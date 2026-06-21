// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 3 implementation: see ew_exec.hpp.
//
// exec() does the per-call prep (result-shape inference + output allocation, const
// materialization, adapter contexts, temp-buffer sizing) ONCE, then hands the traversal to
// broadcastOp(): all operands (inputs + consts-as-Mats + outputs) go in one flat list, and the
// body re-points the program's args at each tile's slices and runs the instruction list. The
// body is op-agnostic-driven: broadcastOp does geometry + 2D tiling + parallelism.

#include "ew_exec.hpp"
#include "ew_broadcast.hpp"
#include <algorithm>
#include <cstring>
#include <vector>

namespace cv { namespace ew {

// Logical shape of a Mat with channels as the innermost dimension; steps in elemsize1 units.
// Used here only to infer the broadcast RESULT shape (spatial + channels) for output allocation;
// broadcastOp does its own channel-aware layout for the traversal itself.
static void matLogical(const Mat& m, MatShape& shp, EwSteps& step, int& esz1)
{
    esz1 = (int)m.elemSize1();
    int nd = m.dims, cn = m.channels();
    shp.resize(nd + 1);
    for (int i = 0; i < nd; i++)
    {
        shp[i] = m.size[i];
        step[i] = m.step[i] / esz1;
    }
    shp[nd] = cn;
    step[nd] = 1;
}

// numpy-style broadcast of several right-aligned shapes.
static bool broadcastShape(const MatShape* shps, int K, MatShape& out)
{
    int nd = 0;
    for (int k = 0; k < K; k++) nd = std::max(nd, shps[k].dims);
    out.assign(nd, 1);
    for (int k = 0; k < K; k++)
    {
        const MatShape& s = shps[k];
        int off = nd - s.dims;
        for (int i = 0; i < s.dims; i++)
        {
            int d = s[i], &o = out[off + i];
            if (o == 1) o = d;
            else if (d != 1 && d != o) return false;
        }
    }
    return true;
}

static void storeScalar(int depth, double v, uchar* p)
{
    switch (depth)
    {
    case CV_8U:  *(uchar*) p = saturate_cast<uchar>(v);  break;
    case CV_8S:  *(schar*) p = saturate_cast<schar>(v);  break;
    case CV_16U: *(ushort*)p = saturate_cast<ushort>(v); break;
    case CV_16S: *(short*) p = saturate_cast<short>(v);  break;
    case CV_32S: *(int*)   p = saturate_cast<int>(v);    break;
    case CV_32F: *(float*) p = saturate_cast<float>(v);  break;
    case CV_64F: *(double*)p = v;                        break;
    default: CV_Error(Error::StsNotImplemented, "ew: unsupported const depth");
    }
}

// Rough per-element cost of an op, in "cycle units" (tuning for parallel_for_ stripe count).
// Unknown ops default to ~division. cv::expression will own this once it drives broadcastOp.
static int opCost(ElemwiseOp op)
{
    switch (op)
    {
    case OP_ADD: case OP_SUB: case OP_MUL: case OP_MIN: case OP_MAX:
    case OP_ABSDIFF: case OP_AND: case OP_OR: case OP_XOR: case OP_NOT:
    case OP_NEG: case OP_ABS: case OP_CAST: case OP_RELU: case OP_COPY_MASK:
    case OP_CMP_EQ: case OP_CMP_NE: case OP_CMP_LT:
    case OP_CMP_LE: case OP_CMP_GT: case OP_CMP_GE: return 1;
    case OP_DIV: case OP_SQRT: case OP_CONVERT_SCALE: return 10;
    case OP_SIN: case OP_COS: case OP_TANH: case OP_ERF:
    case OP_EXP: case OP_LOG: case OP_POW:           return 30;
    default:                                         return 10;
    }
}

EwProgram::EwProgram()
{
    clear();
}

void EwProgram::clear()
{
    prog.allocate(0);
    arginfo.allocate(0);
    bufferOfTemp.allocate(0);
    ninputs = noutputs = ntemps = nbuffers = 0;
}

// Run one resolved instruction over a width x height tile. EW_FN_ELEMWISE calls the broadcasting
// kernel (element steps + the instruction's params block). EW_FN_BINARY calls a core convert
// BinaryFunc (byte steps, Size tile, params = {scale, offset}); a broadcast source (stepx == 0) is
// handled by converting one element per row and replicating it - the executor guarantees the dst
// is contiguous in x. sesz1/desz1 are the source/result element sizes (only used by EW_FN_BINARY).
static inline void runInsn(const EwInsn& ins,
                           const void* p0, size_t y0, size_t x0,
                           const void* p1, size_t y1, size_t x1,
                           const void* p2, size_t y2, size_t x2,
                           void* pr, size_t yr, int w, int h, int sesz1, int desz1)
{
    if (ins.fnkind == EW_FN_BINARY)
    {
        const EwBinaryFunc fn = ins.bfptr;
        void* params = (void*)ins.params.val;
        const size_t srowb = y0 * (size_t)sesz1, drowb = yr * (size_t)desz1;
        if (x0 == 1)                    // contiguous source: one call over the whole tile
            fn((const uchar*)p0, srowb, nullptr, 0, (uchar*)pr, drowb, Size(w, h), params);
        else                            // broadcast source: convert one value per row, replicate
        {
            const uchar* s = (const uchar*)p0; uchar* d = (uchar*)pr;
            for (int i = 0; i < h; i++, s += srowb, d += drowb)
            {
                fn(s, 0, nullptr, 0, d, 0, Size(1, 1), params);
                for (int x = 1; x < w; x++) std::memcpy(d + (size_t)x*desz1, d, (size_t)desz1);
            }
        }
        return;
    }
    int code = ins.fptr(p0, y0, x0, p1, y1, x1, p2, y2, x2, pr, yr, w, h, ins.params.val);
    CV_Assert(code >= 0);
}

struct EwBody {
    const EwInsn* prog;
    // slot -> a NON-NEGATIVE index whose meaning is given by slotKind[s]:
    //   ARG_INPUT / ARG_OUTPUT : index into tile.slices (the broadcast operand list)
    //   ARG_TEMP               : physical temp-buffer id
    //   ARG_CONST              : index into constSlice[]
    const int* slotMap;
    const signed char* slotKind; // [nslots] arginfo[s].kind, for resolving slotMap[s]
    const EwSlice* constSlice;    // [nconsts] fixed scalar slices {ptr=&value, stepy=0, stepx=0}
    const int* bufEszPrefix;  // [nbuffers+1] prefix sums of temp elem sizes; [nbuffers] = total
    const int* esz1Slot;      // [nslots]   element size of each slot's operand (for column offset)
    int ninsn, nslots, nbuffers, capElems;   // capElems: L1 fragment element cap (0 if no temps)
};

void EwProgram::exec(const Mat* inputs, Mat* outputs)
{
    constexpr int LOCAL_HDRS = 3;
    constexpr int LOCAL_CONSTS = 4;
    constexpr int LOCAL_OPS = 16;
    const int nslots = (int)arginfo.size();
    CV_Assert(nslots >= 1 && arginfo[0].kind == ARG_NONE);

    // Scalars (ARG_CONST) never influence the result shape or channel count - the output geometry
    // comes from the real array inputs alone (a scalar broadcasts into whatever they produce).
    int nconsts = 0;
    for (int s = 1; s < nslots; s++)
        if (arginfo[s].kind == ARG_CONST) nconsts++;

    // ---- 1. result shape (spatial dims + channel count), channels innermost ----
    // Fast path: every input shares one shape+channels => the result IS inputs[0]'s shape. Skips
    // the matLogical + broadcastShape rebuild/re-broadcast done only to size the output. Consts are
    // irrelevant here (they don't add dims/channels), so they never break the fast path.
    const int rcn0 = ninputs >= 1 ? inputs[0].channels() : 1;
    bool sameShape = ninputs >= 1;
    for (int i = 1; sameShape && i < ninputs; i++)
    {
        const Mat& a = inputs[i];
        if (a.dims != inputs[0].dims || a.channels() != rcn0) sameShape = false;
        else for (int d = 0; d < a.dims; d++) if (a.size[d] != inputs[0].size[d]) { sameShape = false; break; }
    }

    MatShape spatial;
    int rchannels;
    if (sameShape)
    {
        spatial = inputs[0].size;
        rchannels = rcn0;
    }
    else
    {
        AutoBuffer<MatShape, LOCAL_OPS> bshapes(std::max(ninputs, 1));   // one per real input
        MatShape shp; EwSteps step; int esz1;
        for (int i = 0; i < ninputs; i++) { matLogical(inputs[i], shp, step, esz1); bshapes[i] = shp; }
        MatShape full;
        CV_Assert(broadcastShape(bshapes.data(), ninputs, full) && "ew: inputs not broadcast-compatible");
        const int ndFull = (int)full.size();
        rchannels = full[ndFull - 1];
        spatial = full; spatial.resize(ndFull - 1);
    }

    // ---- 2. ONE pass over slots: allocate outputs, build the operand pointer list + slot map,
    //         per-slot element size, and per-temp-buffer element size. No Mat copies (arr[] holds
    //         pointers); only real inputs + outputs become broadcast operands. ARG_CONST scalars are
    //         NOT broadcast operands and get NO Mat header: each is materialized once into a typed
    //         scratch (constStore) and exposed to the body as a fixed slice {ptr=&value, 0, 0} - a
    //         scalar the kernels read by broadcast. slotMap[s] stays a plain non-negative index;
    //         its meaning (arr / temp / const) is recovered from arginfo[s].kind (see EwBody).
    bool inplace = !(inputs + ninputs <= outputs || outputs + noutputs <= inputs);
    int nhdrs = inplace ? ninputs : 0;

    // A const is stored in ITS OWN depth (ai.depth), independent of the array operands: e.g.
    // multiply(u8, u8, u8, 1./255) carries an FP32 scale even though every matrix is u8. 32 bytes
    // covers the largest single-channel-scope payload (up to 4 channels x 8-byte elements).
    constexpr int CONST_STRIDE = 32;
    AutoBuffer<const Mat*, LOCAL_OPS> arr(ninputs + noutputs);
    AutoBuffer<Mat, LOCAL_HDRS> hdrs(std::max(nhdrs, 1));   // non-owning input headers (in-place incref)
    AutoBuffer<uchar, LOCAL_CONSTS * CONST_STRIDE> constStore((size_t)std::max(nconsts, 1) * CONST_STRIDE);
    AutoBuffer<EwSlice, LOCAL_CONSTS> constSlice(std::max(nconsts, 1));
    AutoBuffer<int, LOCAL_OPS> slotMap(nslots);
    AutoBuffer<signed char, LOCAL_OPS> slotKind(nslots);
    AutoBuffer<int, LOCAL_OPS> esz1Slot(nslots);
    AutoBuffer<int, LOCAL_OPS> bufEsz(std::max(1, nbuffers));
    for (int b = 0; b < nbuffers; b++) bufEsz[b] = 1;
    int narr = 0, nc = 0;
    if (inplace) {
        // save (incref) inputs in the case of in-place operation
        // to protect them from premature deallocation
        for (int j = 0; j < ninputs; j++) hdrs[j] = inputs[j];
        inputs = hdrs.data();
    }
    esz1Slot[0] = 0;
    slotKind[0] = (signed char)ARG_NONE;
    for (int s = 1; s < nslots; s++)
    {
        const EwArgInfo& ai = arginfo[s];
        const int e = esz1Slot[s] = (int)CV_ELEM_SIZE1(ai.depth);
        slotKind[s] = (signed char)ai.kind;
        if (ai.kind == ARG_INPUT) {
            slotMap[s] = narr;
            arr[narr++] = &inputs[ai.index];
        }
        else if (ai.kind == ARG_OUTPUT) {
            outputs[ai.index].create(spatial, CV_MAKETYPE(ai.depth, rchannels));
            slotMap[s] = narr; arr[narr++] = &outputs[ai.index];
        }
        else if (ai.kind == ARG_CONST) {
            const int c = std::max(ai.channels, 1);
            CV_Assert(c <= 4 && (size_t)c * e <= CONST_STRIDE);
            uchar* dst = constStore.data() + (size_t)nc * CONST_STRIDE;
            for (int k = 0; k < c; k++) storeScalar(ai.depth, ai.cval[k], dst + (size_t)k * e);
            constSlice[nc].ptr = dst; constSlice[nc].stepy = 0; constSlice[nc].stepx = 0;
            slotMap[s] = nc;                   // index into constSlice[]
            nc++;
        }
        else { // ARG_TEMP
            const int buf = bufferOfTemp.empty() ? ai.index : bufferOfTemp[ai.index];
            slotMap[s] = buf;                  // physical temp-buffer id
            bufEsz[buf] = std::max(bufEsz[buf], e);
        }
    }

    // Temp-buffer layout precomputed once (point 3): prefix sums give each buffer's byte offset per
    // element, so the body never recomputes offsets per tile - it just does prefix[buf]*region.
    // capElems = how many elements one L1-sized fragment holds (0 when there are no temps at all).
    AutoBuffer<int, LOCAL_OPS> bufEszPrefix(nbuffers + 1);
    bufEszPrefix[0] = 0;
    for (int b = 0; b < nbuffers; b++) bufEszPrefix[b + 1] = bufEszPrefix[b] + bufEsz[b];
    const int totalEsz = bufEszPrefix[nbuffers];
    const int capElems = totalEsz > 0 ? std::max(64, (16 * 1024) / totalEsz) : 0;

    // ---- 3. ONE pass over instructions: summed per-element cost (kernels are resolved at build
    //         time; the convert BinaryFunc is reached directly via EwInsn::bfptr - no contexts). ----
    const int ninsn = (int)prog.size();
    long long costPerElem = 0;
    for (int n = 0; n < ninsn; n++)
        costPerElem += opCost(prog[n].op);

    // ---- 4. parallel work hint: total output scalars x summed per-element op cost / budget. ----
    long long otot = (long long)rchannels;
    for (int d = 0; d < (int)spatial.size(); d++) otot *= spatial[d];
    const double nstripes = (double)otot * (double)std::max<long long>(costPerElem, 1) *
                            (1./ (double)(1 << 18));

    EwBody body;
    body.prog = prog.data();
    body.ninsn = ninsn;
    body.slotMap = slotMap.data();

    // ---- 7. drive: broadcastOp does geometry + 2D tiling + parallelism; the body runs the
    //         frozen program on each tile (temps tile-local, re-pointed from the tile slices).
    //         expandChannels=true => the body always sees single-channel data.
    //
    //         All state the body needs is packed into one POD (EwBody) so the lambda captures a
    //         SINGLE reference: the closure is then one pointer, fits std::function's small-buffer
    //         and never heap-allocates. Inside, the hot fields are copied into locals so the
    //         per-tile/per-insn loops read them from registers, not through the captured pointer. -

    body.bufEszPrefix = bufEszPrefix.data();
    body.esz1Slot = esz1Slot.data();
    body.slotKind = slotKind.data();
    body.constSlice = constSlice.data();
    body.nslots = nslots;
    body.nbuffers = nbuffers;
    body.capElems = capElems;

    broadcastOp(arr.data(), narr, [&](const EwTile& tile)
    {
        EwBody& bc = body;
        const EwInsn* prog = bc.prog;             // hot fields -> locals (registers/stack)
        const int* const slotMap     = bc.slotMap;
        const signed char* const slotKind = bc.slotKind;
        const int* const esz1Slot    = bc.esz1Slot;
        const EwSlice* const constSlice = bc.constSlice;
        const int ninsn = bc.ninsn, nslots = bc.nslots;
        const int w = tile.width, h = tile.height;

        // No temps (single-op add/sub/...): run the whole tile in one pass and pull each operand's
        // slice straight from the tile - no args array, no scratch, no fragmentation. Here a slot is
        // either an array (input/output -> tile.slices) or a const scalar (-> a fixed constSlice);
        // slot 0 (ARG_NONE) is the null operand for unused args (e.g. a binary op's arg2). The
        // dominant pure-array path keeps slotKind[s]==ARG_INPUT/OUTPUT => the branch predicts away.
        if (bc.nbuffers == 0)
        {
            const EwSlice nullSlice;
            auto S = [&](int s) -> const EwSlice& {
                if (s <= 0) return nullSlice;
                return slotKind[s] == ARG_CONST ? constSlice[slotMap[s]] : tile.slices[slotMap[s]];
            };
            for (int n = 0; n < ninsn; n++)
            {
                const EwInsn& ins = prog[n];
                const EwSlice& a0 = S(ins.arg0);
                const EwSlice& a1 = S(ins.arg1);
                const EwSlice& a2 = S(ins.arg2);
                const EwSlice& rr = S(ins.result);
                runInsn(ins, a0.ptr, a0.stepy, a0.stepx, a1.ptr, a1.stepy, a1.stepx,
                        a2.ptr, a2.stepy, a2.stepx, (void*)rr.ptr, rr.stepy, w, h,
                        esz1Slot[ins.arg0], esz1Slot[ins.result]);
            }
            return;
        }

        // Fused temps: run the program over L1-sized column strips so the intermediates stay in
        // cache. Strip width Wf holds <= capElems elements per buffer; the per-buffer offsets are
        // precomputed (bufEszPrefix), so the body only scales by `region` = Wf*h - no per-tile
        // offset loop. Each temp buffer occupies [bufEszPrefix[buf]*region, ...) bytes in tstore.
        const int* const bufEszPrefix = bc.bufEszPrefix;
        const int Wf = std::min(w, std::max(1, bc.capElems / std::max(1, h)));
        const size_t region = alignSize((size_t)Wf * h, 8);
        AutoBuffer<uchar> tstore((size_t)bufEszPrefix[bc.nbuffers] * region);

        AutoBuffer<EwArg, LOCAL_OPS> args(nslots);
        for (int x0 = 0; x0 < w; x0 += Wf)
        {
            const int wf = std::min(Wf, w - x0);
            for (int s = 1; s < nslots; s++)
            {
                EwArg& a = args[s];
                const int k = slotKind[s];
                if (k == ARG_TEMP)                      // contiguous strip-local buffer
                {
                    a.ptr = tstore.data() + (size_t)bufEszPrefix[slotMap[s]] * region;
                    a.stepy = (size_t)wf; a.stepx = 1;
                }
                else if (k == ARG_CONST)               // fixed scalar slice, no x0 offset (stepx 0)
                {
                    const EwSlice& cs = constSlice[slotMap[s]];
                    a.ptr = (uchar*)cs.ptr; a.stepy = 0; a.stepx = 0;
                }
                else                                    // array operand: this strip of the slice
                {
                    const EwSlice& sl = tile.slices[slotMap[s]];
                    a.ptr = (uchar*)sl.ptr + (size_t)x0 * sl.stepx * (size_t)esz1Slot[s];
                    a.stepy = sl.stepy; a.stepx = sl.stepx;
                }
            }

            for (int n = 0; n < ninsn; n++)
            {
                const EwInsn& ins = prog[n];
                const EwArg& a0 = args[ins.arg0]; const EwArg& a1 = args[ins.arg1];
                const EwArg& a2 = args[ins.arg2]; const EwArg& rr = args[ins.result];
                runInsn(ins, a0.ptr, a0.stepy, a0.stepx, a1.ptr, a1.stepy, a1.stepx,
                        a2.ptr, a2.stepy, a2.stepx, (void*)rr.ptr, rr.stepy, wf, h,
                        esz1Slot[ins.arg0], esz1Slot[ins.result]);
            }
        }
    }, true, nstripes);
}

// ---------------------------------------------------------------------------
// Manual program builders.
// ---------------------------------------------------------------------------
static inline EwArgInfo mkArg(ArgKind kind, int depth, int index)
{
    EwArgInfo a; a.kind = kind; a.depth = depth; a.index = index; return a;
}

// [VP] looks like there is a bug here. rank(8s) < rank(16u),
// so when we add 8s to 16u, the coercion type will be 16u,
// however it should be 32s. We need to take into account signness of integer types.
// also, when we subtract 8u from 8s, the result should have 16s type, so
// the coercion result should always be a signed type.
// so, the coercion rule should be:
// * if a == b -> a
// * else if a == 64f || b == 64f -> 64f
// * else if flt(a) || flt(b) -> 32f
// * else if max(a, b) <= 8s -> 16s
// * else if max(a, b) <= 16s -> 32s
// * else if a == 32s || a == 32u || b == 32s || b == 32u -> 64s
// * else 64f
//
// or see below the same rule done with a single 'if'

// numpy-ish promotion of two known depths (float dominates; wider integer otherwise).
static int promote2(int a, int b)
{
    if (a == b) return a;

    constexpr unsigned lbits = 3, lmask = (1u << lbits) - 1u;
    const uint64_t typelut = (uint64_t)((0ULL << CV_8U*lbits) | (0ULL << CV_8S*lbits) |
                                        (1ULL << CV_16U*lbits) | (1ULL << CV_16S*lbits) |
                                        (2ULL << CV_32U*lbits) | (2ULL << CV_32S*lbits) |
                                        (3ULL << CV_16F*lbits) | (3ULL << CV_16BF*lbits) |
                                        (3ULL << CV_32F*lbits) | (4ULL << CV_64F*lbits) |
                                        (4ULL << CV_64S*lbits) | (4ULL << CV_64U*lbits));
    unsigned pr_a = unsigned((typelut >> (a*lbits)) & lmask);
    unsigned pr_b = unsigned((typelut >> (b*lbits)) & lmask);
    unsigned max_pr = std::max(pr_a, pr_b);
    constexpr unsigned dbits = CV_CN_SHIFT, dmask = (1u << dbits) - 1u;
    const unsigned ctypelut = ((CV_16S << 0*dbits) | (CV_32S << 1*dbits) | (CV_64S << 2*dbits) |
                            (CV_32F << 3*dbits) | (CV_64F << 4*dbits));
    return int((ctypelut >> (max_pr*dbits)) & dmask);
}

// A wide type in which add(depth,depth->wdepth) exists and the sum is held without a premature clamp.
static int safeWide(int depth)
{
    constexpr unsigned lbits = 3, lmask = (1u << lbits) - 1u;
    const uint64_t typelut = (uint64_t)((0ULL << CV_8U*lbits) | (0ULL << CV_8S*lbits) |
                                        (1ULL << CV_16U*lbits) | (1ULL << CV_16S*lbits) |
                                        (2ULL << CV_32U*lbits) | (2ULL << CV_32S*lbits) |
                                        (3ULL << CV_16F*lbits) | (3ULL << CV_16BF*lbits) |
                                        (3ULL << CV_32F*lbits) | (4ULL << CV_64F*lbits) |
                                        (4ULL << CV_64S*lbits) | (4ULL << CV_64U*lbits));
    unsigned pr_depth = unsigned((typelut >> (depth*lbits)) & lmask);
    constexpr unsigned dbits = CV_CN_SHIFT, dmask = (1u << dbits) - 1u;
    const unsigned ctypelut = ((CV_16S << 0*dbits) | (CV_32S << 1*dbits) | (CV_64S << 2*dbits) |
                              (CV_32F << 3*dbits) | (CV_64F << 4*dbits));
    return int((ctypelut >> (pr_depth*dbits)) & dmask);
}

// Shared composer for the binary arith ops (ADD, SUB, MUL, DIV): bring both operands to a common
// type. ADD/SUB apply `op` directly to rdepth if a kernel exists, else compute in a safe wide type
// (signed - a-b may go negative) and cast down. MUL/DIV compute in the float work type (matching
// cv::multiply/divide) and cast that down to rdepth.
//
// maskDepth != EW_DEPTH_NONE adds a write-mask (input #2): the arithmetic result lands in a temp
// (type rdepth) and a final OP_COPY_MASK overwrites only the masked subset of the (pre-existing)
// output, leaving the rest UNCHANGED (dst = mask ? result : dst, matching cv::add/... with a mask).
// copyMask is always the LAST instruction, after all ops and conversions. The mask is a single-
// channel 1-byte array (bool/u8/s8) the size of the output spatial shape; it rides the normal
// broadcast machinery (single-channel data => per-element; n-channel => broadcast across the
// channel axis), so nothing special is needed in the executor.
void makeBinaryArithProgram(EwProgram& p, ElemwiseOp op, int depth0, int depth1, int rdepth,
                            int maskDepth, double scale)
{
    p.clear();
    const bool masked = maskDepth != EW_DEPTH_NONE;
    p.ninputs = masked ? 3 : 2; p.noutputs = 1;
    p.arginfo.push_back(mkArg(ARG_NONE,  EW_DEPTH_NONE, -1));               // slot 0
    int sIn0  = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, depth0, 0));
    int sIn1  = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, depth1, 1));
    int sMask = 0;
    if (masked) { sMask = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, maskDepth, 2)); }
    int sOut  = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_OUTPUT, rdepth, 0));

    int ntemps = 0;
    auto addTemp = [&](int depth) {
        int s = (int)p.arginfo.size();
        p.arginfo.push_back(mkArg(ARG_TEMP, depth, ntemps++));
        return s;
    };
    auto addInsn = [&](ElemwiseOp o, int a0, int a1, int r) {
        EwInsn ins; ins.op = o; ins.arg0 = a0; ins.arg1 = a1; ins.arg2 = 0; ins.result = r;
        bool ok = resolveInsnKernel(ins, p.arginfo[a0].depth,
                                    a1 ? p.arginfo[a1].depth : EW_DEPTH_NONE,
                                    EW_DEPTH_NONE, p.arginfo[r].depth);
        CV_Assert(ok);
        p.prog.push_back(ins);
    };

    // The arithmetic result goes to `dst`: the output directly, or a temp when masked (then
    // copyMask moves it into the output). Fast single-op path stays single-op when unmasked.
    int dst = masked ? addTemp(rdepth) : sOut;

    if (op == OP_MUL || op == OP_DIV)
    {
        // cast both operands to a common type C, then mul/div in the float work type Wf (float for
        // <=16-bit/f16/bf/f32, double for 32/64-bit/f64 - matching cv::multiply/divide), then cast
        // Wf down to rdepth. The optional `scale` rides the mul/div instruction's params[0] (mul:
        // a*b*scale; div: a*scale/b), exactly as cv::multiply/divide's scale argument. For div the
        // divide-by-zero policy is decided from the ORIGINAL input types: both-integer => guard /0
        // -> 0; any float => no guard (a/0 -> inf), as cv::divide.
        auto isFlt = [](int d){ return d==CV_16F || d==CV_16BF || d==CV_32F || d==CV_64F; };
        const bool bothInt = !isFlt(depth0) && !isFlt(depth1);
        int C = (depth0 == depth1) ? depth0 : promote2(depth0, depth1);
        if (depth0 != C) { int t = addTemp(C); addInsn(OP_CAST, sIn0, 0, t); sIn0 = t; }
        if (depth1 != C) { int t = addTemp(C); addInsn(OP_CAST, sIn1, 0, t); sIn1 = t; }
        const bool wide = (C==CV_32U || C==CV_32S || C==CV_64U || C==CV_64S || C==CV_64F);
        int Wf = wide ? CV_64F : CV_32F;
        // emit the mul/div into slot r, carrying `scale` in params[0].
        auto emit = [&](int r) {
            EwInsn ins; ins.op = op; ins.arg0 = sIn0; ins.arg1 = sIn1; ins.arg2 = 0; ins.result = r;
            ins.params = Scalar(scale);
            if (op == OP_DIV) {
                // div with the caller-known /0 policy: bypass resolveInsnKernel (which would
                // re-derive `checked` from the common type and miss the wide-int->float case).
                ins.fnkind = EW_FN_ELEMWISE;
                ins.fptr = getDivFunc(p.arginfo[sIn0].depth, p.arginfo[r].depth, bothInt);
                CV_Assert(ins.fptr != nullptr);
            } else {
                bool ok = resolveInsnKernel(ins, p.arginfo[sIn0].depth, p.arginfo[sIn1].depth,
                                            EW_DEPTH_NONE, p.arginfo[r].depth);
                CV_Assert(ok);
            }
            p.prog.push_back(ins);
        };
        // prefer a direct C x C -> rdepth kernel when one exists (mul now has T x T -> T for all but
        // 64-bit types => one fused pass); else compute in Wf and cast down. For div no T x T -> T
        // exists, so this reduces to the old rdepth==Wf test.
        if (rdepth == Wf || getElemwiseFunc(op, C, C, EW_DEPTH_NONE, rdepth))
            emit(dst);
        else
        {
            int tW = addTemp(Wf);
            emit(tW);
            addInsn(OP_CAST, tW, 0, dst);
        }
    }
    else if (depth0 == depth1 && depth0 == rdepth)
        addInsn(op, sIn0, sIn1, dst);
    else
    {
        // ADD/SUB: 1. bring both operands to a common type.
        int depth = (depth0 == depth1) ? depth0 : promote2(depth0, depth1);
        if (depth0 != depth) { int t = addTemp(depth); addInsn(OP_CAST, sIn0, 0, t); sIn0 = t; }
        if (depth1 != depth) { int t = addTemp(depth); addInsn(OP_CAST, sIn1, 0, t); sIn1 = t; }

        // 2. op directly to rdepth if a kernel exists, else compute wide then cast down.
        if (getElemwiseFunc(op, depth, depth, EW_DEPTH_NONE, rdepth))
            addInsn(op, sIn0, sIn1, dst);
        else
        {
            int W = safeWide(depth);
            int tW = addTemp(W);
            addInsn(op, sIn0, sIn1, tW);
            addInsn(OP_CAST, tW, 0, dst);
        }
    }

    if (masked)
        addInsn(OP_COPY_MASK, dst, sMask, sOut);   // last: dst = (mask!=0) ? result : dst

    p.ntemps = ntemps; p.nbuffers = ntemps;
    p.bufferOfTemp.resize(ntemps);
    for (int i = 0; i < ntemps; i++) p.bufferOfTemp[i] = i;
}

void makeAddProgram(EwProgram& p, int depth0, int depth1, int rdepth)
{ makeBinaryArithProgram(p, OP_ADD, depth0, depth1, rdepth); }

// addWeighted(a, alpha, b, beta, gamma) = a*alpha + b*beta + gamma, as two fused convert_scale
// MACs + an add (+ a final cast when the output type differs from the working type W):
//   t0 = cast<W>(a*alpha + gamma) ;  t1 = cast<W>(b*beta) ;  out = cast<rdepth>(t0 + t1)
// The convert_scale steps ride core's optimized scale kernel; alpha/beta/gamma travel in each
// instruction's params block ({scale, offset}), not as operands. The 2-3 temps exercise the body's
// L1 column-fragmentation.
void makeAddWeightedProgram(EwProgram& p, int depth0, int depth1, int rdepth,
                            double alpha, double beta, double gamma)
{
    p.clear();
    p.ninputs = 2; p.noutputs = 1;
    const int W = (depth0 == CV_64F || depth1 == CV_64F || rdepth == CV_64F) ? CV_64F : CV_32F;

    p.arginfo.push_back(mkArg(ARG_NONE, EW_DEPTH_NONE, -1));    // slot 0
    int sA = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, depth0, 0));
    int sB = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, depth1, 1));

    int ntemps = 0;
    auto addTemp = [&](int depth) {
        int s = (int)p.arginfo.size();
        p.arginfo.push_back(mkArg(ARG_TEMP, depth, ntemps++));
        return s;
    };
    auto addInsn = [&](ElemwiseOp op, int a0, int a1, int r, Scalar params = Scalar(1)) {
        EwInsn ins; ins.op = op; ins.arg0 = a0; ins.arg1 = a1; ins.arg2 = 0; ins.result = r;
        ins.params = params;
        bool ok = resolveInsnKernel(ins, p.arginfo[a0].depth,
                                    a1 ? p.arginfo[a1].depth : EW_DEPTH_NONE,
                                    EW_DEPTH_NONE, p.arginfo[r].depth);
        CV_Assert(ok);
        p.prog.push_back(ins);
    };

    int t0 = addTemp(W), t1 = addTemp(W);
    addInsn(OP_CONVERT_SCALE, sA, 0, t0, Scalar(alpha, gamma));   // t0 = a*alpha + gamma
    addInsn(OP_CONVERT_SCALE, sB, 0, t1, Scalar(beta, 0.0));      // t1 = b*beta

    int sOut = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_OUTPUT, rdepth, 0));
    if (rdepth == W)
        addInsn(OP_ADD, t0, t1, sOut);
    else
    {
        int t2 = addTemp(W);
        addInsn(OP_ADD, t0, t1, t2);
        addInsn(OP_CAST, t2, 0, sOut);
    }

    p.ntemps = ntemps; p.nbuffers = ntemps;
    p.bufferOfTemp.resize(ntemps);
    for (int i = 0; i < ntemps; i++) p.bufferOfTemp[i] = i;
}

}} // namespace cv::ew
