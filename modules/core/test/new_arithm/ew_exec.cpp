// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 3 implementation: see ew_exec.hpp.
//
// exec() does the per-call prep (result-shape inference + output allocation, const
// materialization, temp-buffer sizing) ONCE, then hands the traversal to broadcastOp(): all
// operands (inputs + consts + outputs) go in one flat list, and the body re-points the program's
// args at each tile's slices and runs the instruction list. The body is op-agnostic-driven:
// broadcastOp does geometry + 2D tiling + parallelism. A cheap fast path up front handles small
// const-free same-shape continuous arrays directly, skipping geometry/tiling/parallelism.

#include "ew_exec.hpp"
#include "ew_broadcast.hpp"
#include <algorithm>
#include <cstring>

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
static int opCost(TOp op)
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

TExpr::TExpr()
{
    clear();
}

// Reset to an empty program: drop all instructions/slots and re-seat slot 0 = NONE.
void TExpr::clear()
{
    prog.allocate(0);
    arginfo.allocate(0);
    bufferOfTemp.allocate(0);
    ninputs = noutputs = ntemps = nbuffers = 0;
    Arg none;                       // slot 0: the reserved empty operand (kind == NONE)
    arginfo.push_back(none);
}

// ---- program builders: append a slot / instruction, return its index ----
int TExpr::addInput(int depth)
{
    Arg a; a.kind = INPUT; a.depth = depth; a.index = ninputs++;
    arginfo.push_back(a); return (int)arginfo.size() - 1;
}

int TExpr::addOutput(int depth)
{
    Arg a; a.kind = OUTPUT; a.depth = depth; a.index = noutputs++;
    arginfo.push_back(a); return (int)arginfo.size() - 1;
}

int TExpr::addTemp(int depth)
{
    Arg a; a.kind = TEMP; a.depth = depth; a.index = ntemps++;
    arginfo.push_back(a); return (int)arginfo.size() - 1;
}

int TExpr::addConst(int depth, const Scalar& v, int channels)
{
    Arg a; a.kind = CONST; a.depth = depth; a.channels = channels; a.cval = v;
    arginfo.push_back(a); return (int)arginfo.size() - 1;
}

// Append one instruction with a pre-resolved kernel (the caller probed getElemwiseFunc, or knows
// the kernel - e.g. div's /0-aware kernel). No re-resolution.
int TExpr::addInsn(TOp op, int a0, int a1, int a2, int result, const TKernel& kernel, const Scalar& params)
{
    TExpr::Insn ins; ins.op = op; ins.arg0 = a0; ins.arg1 = a1; ins.arg2 = a2; ins.result = result;
    ins.params = params; ins.kernel = kernel;
    prog.push_back(ins);
    return (int)prog.size() - 1;
}

// Append one instruction, resolving its kernel NOW from the operand/result depths (final at build
// time). compile() therefore never re-resolves. A builder that needs a specific kernel (div's /0
// policy) pushes a pre-bound Insn directly instead.
int TExpr::addInsn(TOp op, int a0, int a1, int a2, int result, const Scalar& params)
{
    int d0 = a0 ? arginfo[a0].depth : EW_DEPTH_NONE;
    int d1 = a1 ? arginfo[a1].depth : EW_DEPTH_NONE;
    int d2 = a2 ? arginfo[a2].depth : EW_DEPTH_NONE;
    TKernel k = getElemwiseFunc(op, d0, d1, d2, arginfo[result].depth);
    CV_Assert(k.fptr && "ew: no kernel for this op/type combination");
    return addInsn(op, a0, a1, a2, result, k, params);
}

// Cast `arg` to `depth` only if it is not already that depth (the sole cast-insertion helper for
// the hand builders and emit()'s `place`); returns the slot holding the value at `depth`.
int TExpr::maybeAddCast(int arg, int depth)
{
    if (arginfo[arg].depth == depth) return arg;
    int t = addTemp(depth);
    addInsn(OP_CAST, arg, 0, 0, t);
    return t;
}

// Shared cast-and-emit core (see ew_op.hpp). Casts operands to `computeType`, emits `op` to
// `resultType` directly when a kernel exists, else computes in `wideType` and casts the result down.
int TExpr::emitTyped(TOp op, const int* slots, int nargs, int computeType, int resultType,
                     int wideType, const Scalar& params, int dstSlot)
{
    int s[3] = { 0, 0, 0 };
    for (int k = 0; k < nargs; k++) s[k] = maybeAddCast(slots[k], computeType);
    const int d0 = nargs > 0 ? computeType : EW_DEPTH_NONE;
    const int d1 = nargs > 1 ? computeType : EW_DEPTH_NONE;
    const int d2 = nargs > 2 ? computeType : EW_DEPTH_NONE;

    // div's /0 guard follows the ORIGINAL operand int-ness, NOT computeType: promote2 floats two
    // WIDE integers (e.g. 16U/64S -> 64F), but cv::divide still guards (b==0 -> 0) when both inputs
    // are integer. So resolve div's kernel via getDivFunc with that guard; every other op via
    // getElemwiseFunc (which reads computeType, correct for them).
    auto isFlt = [](int d){ return d==CV_16F || d==CV_16BF || d==CV_32F || d==CV_64F; };
    const bool divGuard = op == OP_DIV && !isFlt(arginfo[slots[0]].depth) && !isFlt(arginfo[slots[1]].depth);
    auto resolve = [&](int rd) {
        return op == OP_DIV ? getDivFunc(computeType, rd, divGuard) : getElemwiseFunc(op, d0, d1, d2, rd);
    };

    TKernel k = resolve(resultType);                           // direct computeType -> resultType?
    if (k.fptr)
    {
        int res = dstSlot >= 0 ? dstSlot : addTemp(resultType);
        addInsn(op, s[0], s[1], s[2], res, k, params);
        return res;
    }
    k = resolve(wideType);                                     // else compute wide, then narrow
    CV_Assert(k.fptr && "ew: no kernel for this op/type combination");
    int wSlot = addTemp(wideType);
    addInsn(op, s[0], s[1], s[2], wSlot, k, params);
    if (dstSlot >= 0) { addInsn(OP_CAST, wSlot, 0, 0, dstSlot); return dstSlot; }
    return maybeAddCast(wSlot, resultType);
}

// Run one resolved instruction over a width x height tile. Every kernel uses ONE calling
// convention - the universal KernelFunc (element steps, the instruction's params block, plus
// kernel.flags/kernel.userdata). OP_CAST / OP_CONVERT_SCALE bind castKernel, which forwards to a
// core convert BinaryFunc (carried in kernel.userdata) over the distinct sub-region and then
// expands it across any broadcast axis (see castKernel/expandKernel in ew_kernels.cpp).
static inline void runInsn(const TExpr::Insn& ins,
                           const void* p0, size_t y0, size_t x0,
                           const void* p1, size_t y1, size_t x1,
                           const void* p2, size_t y2, size_t x2,
                           void* pr, size_t yr, int w, int h)
{
    int code = ins.kernel.fptr(p0, y0, x0, p1, y1, x1, p2, y2, x2, pr, yr, w, h,
                                ins.params.val, ins.kernel.flags, ins.kernel.userdata);
    CV_Assert(code >= 0);
}

struct EwBody {
    const TExpr::Insn* prog;
    // slot -> a NON-NEGATIVE index whose meaning is given by slotKind[s]:
    //   TExpr::INPUT / TExpr::OUTPUT : index into tile.slices (the broadcast operand list)
    //   TExpr::TEMP               : physical temp-buffer id
    //   TExpr::CONST              : index into constSlice[]
    const int* slotMap;
    const signed char* slotKind; // [nslots] arginfo[s].kind, for resolving slotMap[s]
    const EwSlice* constSlice;    // [nconsts] fixed scalar slices {ptr=&value, stepy=0, stepx=0}
    const int* bufEszPrefix;  // [nbuffers+1] prefix sums of temp elem sizes; [nbuffers] = total
    int ninsn, nslots, nbuffers, capElems;   // capElems: L1 fragment element cap (0 if no temps)
};

void TExpr::exec(const Mat* inputs, Mat* outputs)
{
    constexpr int LOCAL_HDRS = 3;
    constexpr int LOCAL_CONSTS = 4;
    constexpr int LOCAL_OPS = 16;
    const int nslots = (int)arginfo.size();
    CV_Assert(nslots >= 1 && arginfo[0].kind == TExpr::NONE);

    // ---- cheap fast path: a const-free program over small, same-shape, continuous arrays. Channels
    //      fold into one flat contiguous run, so the whole job is a flat (total x 1) strip - run the
    //      program directly here and skip the per-call prep loop, broadcastOp's geometry, 2D tiling
    //      and the parallel framework (all pure overhead at this size). Temps are allowed: their
    //      byte layout (bufEszPrefix) is known from compile(), so we size an L1 scratch in O(1) and
    //      walk the strip in L1-sized fragments (the intermediates stay hot). nconsts/nbuffers/
    //      bufEszPrefix come from compile(); the rest is a quick check of the args. ----
    if (nconsts == 0 && ninputs >= 1)
    {
        constexpr size_t EW_FASTPATH_MAX = 1u << 17;   // above this the tiled/parallel path wins
        const Mat& r = inputs[0];
        const int rcn = r.channels();
        bool ok = r.isContinuous();
        for (int i = 1; ok && i < ninputs; i++)
        {
            const Mat& a = inputs[i];
            ok = a.isContinuous() && a.channels() == rcn && a.size == r.size;
        }
        const size_t total = ok ? r.total() * rcn : 0;
        if (ok && total <= EW_FASTPATH_MAX)
        {
            for (int s = 1; s < nslots; s++)
                if (arginfo[s].kind == OUTPUT)
                    outputs[arginfo[s].index].create(r.dims, r.size.p, CV_MAKETYPE(arginfo[s].depth, rcn));
            const int ninsn = (int)prog.size();

            if (nbuffers == 0)
            {
                // No temps (single-op add/sub/mul/...): one flat (total x 1) pass, operands point
                // straight at the Mat data - no scratch, no fragment loop. Slots are only INPUT/OUTPUT.
                auto ptrOf = [&](int s) -> void* {
                    if (s <= 0) return nullptr;
                    const Arg& ai = arginfo[s];
                    return ai.kind == INPUT ? (void*)inputs[ai.index].data : (void*)outputs[ai.index].data;
                };
                const int w = (int)total;
                for (int n = 0; n < ninsn; n++)
                {
                    const Insn& ins = prog[n];
                    runInsn(ins, ptrOf(ins.arg0), 0, 1, ptrOf(ins.arg1), 0, 1,
                            ptrOf(ins.arg2), 0, 1, ptrOf(ins.result), 0, w, 1);
                }
                return;
            }

            // Temps present: walk the strip in L1-sized fragments so the intermediates stay hot.
            // Byte layout (bufEszPrefix) and the fragment cap (capElems) both come from compile();
            // scratch is totalEsz*wf0 (<= ~16KB).
            const int totalEsz = bufEszPrefix[nbuffers];
            const int wf0 = std::min((int)total, capElems);
            AutoBuffer<uchar> scratch((size_t)totalEsz * (size_t)wf0);
            for (int x0 = 0; x0 < (int)total; x0 += wf0)
            {
                const int wf = std::min(wf0, (int)total - x0);
                auto ptrOf = [&](int s) -> void* {
                    if (s <= 0) return nullptr;
                    const Arg& ai = arginfo[s];
                    size_t esz = CV_ELEM_SIZE1(ai.depth);
                    if (ai.kind == INPUT)  return (uchar*)inputs[ai.index].data  + (size_t)x0 * esz;
                    if (ai.kind == OUTPUT) return (uchar*)outputs[ai.index].data + (size_t)x0 * esz;
                    const int b = bufferOfTemp.empty() ? ai.index : bufferOfTemp[ai.index];  // TEMP
                    return scratch.data() + (size_t)bufEszPrefix[b] * (size_t)wf0;            // fragment-local
                };
                for (int n = 0; n < ninsn; n++)
                {
                    const Insn& ins = prog[n];
                    runInsn(ins, ptrOf(ins.arg0), 0, 1, ptrOf(ins.arg1), 0, 1,
                            ptrOf(ins.arg2), 0, 1, ptrOf(ins.result), 0, wf, 1);
                }
            }
            return;
        }
    }

    // Scalars (TExpr::CONST) never influence the result shape or channel count - the output geometry
    // comes from the real array inputs alone (a scalar broadcasts into whatever they produce).
    // A flexible CONST (depth == EW_DEPTH_NONE) is a leftover literal: emit() materializes a typed
    // copy at each use, so the original is dead - it is skipped entirely (no slice, no storeScalar).

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
    //         pointers); only real inputs + outputs become broadcast operands. TExpr::CONST scalars are
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
    int narr = 0, nc = 0;
    if (inplace) {
        // save (incref) inputs in the case of in-place operation
        // to protect them from premature deallocation
        for (int j = 0; j < ninputs; j++) hdrs[j] = inputs[j];
        inputs = hdrs.data();
    }
    slotKind[0] = (signed char)TExpr::NONE;
    for (int s = 1; s < nslots; s++)
    {
        const TExpr::Arg& ai = arginfo[s];
        const size_t esz = CV_ELEM_SIZE1(ai.depth);
        slotKind[s] = (signed char)ai.kind;
        if (ai.kind == TExpr::INPUT) {
            slotMap[s] = narr;
            arr[narr++] = &inputs[ai.index];
        }
        else if (ai.kind == TExpr::OUTPUT) {
            outputs[ai.index].create(spatial, CV_MAKETYPE(ai.depth, rchannels));
            slotMap[s] = narr; arr[narr++] = &outputs[ai.index];
        }
        else if (ai.kind == TExpr::CONST && ai.depth == EW_DEPTH_NONE) {
            slotKind[s] = (signed char)TExpr::NONE;   // dead flexible literal: never referenced
            slotMap[s] = 0;
        }
        else if (ai.kind == TExpr::CONST) {
            const int c = std::max(ai.channels, 1);
            CV_Assert(c <= 4 && (size_t)c * esz <= CONST_STRIDE);
            uchar* dst = constStore.data() + (size_t)nc * CONST_STRIDE;
            for (int k = 0; k < c; k++) storeScalar(ai.depth, ai.cval[k], dst + (size_t)k * esz);
            constSlice[nc].ptr = dst; constSlice[nc].stepy = 0; constSlice[nc].stepx = 0;
            slotMap[s] = nc;                   // index into constSlice[]
            nc++;
        }
        else { // TExpr::TEMP
            slotMap[s] = bufferOfTemp.empty() ? ai.index : bufferOfTemp[ai.index];  // physical buffer id
        }
    }

    // Temp-buffer byte layout (prefix sums, bufEszPrefix) and the L1 fragment cap (capElems) were
    // both precomputed by compile(); the body uses them directly (prefix[buf]*region = a buffer's
    // byte offset). capElems == INT_MAX when there are no temps => the body runs each tile in a
    // single block (no scratch).

    // ---- 3. ONE pass over instructions: summed per-element cost (kernels were bound at compile()). ----
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

    // ---- 5. drive: broadcastOp does geometry + 2D tiling + parallelism; the body runs the
    //         frozen program on each tile (temps tile-local, re-pointed from the tile slices).
    //         expandChannels=true => the body always sees single-channel data.
    //
    //         All state the body needs is packed into one POD (EwBody) so the lambda captures a
    //         SINGLE reference: the closure is then one pointer, fits std::function's small-buffer
    //         and never heap-allocates. Inside, the hot fields are copied into locals so the
    //         per-tile/per-insn loops read them from registers, not through the captured pointer. -

    body.bufEszPrefix = bufEszPrefix.data();
    body.slotKind = slotKind.data();
    body.constSlice = constSlice.data();
    body.nslots = nslots;
    body.nbuffers = nbuffers;
    body.capElems = capElems;

    broadcastOp(arr.data(), narr, [&](const EwTile& tile)
    {
        EwBody& bc = body;
        const TExpr::Insn* prog = bc.prog;             // hot fields -> locals (registers/stack)
        const int* const slotMap     = bc.slotMap;
        const signed char* const slotKind = bc.slotKind;
        const EwSlice* const constSlice = bc.constSlice;
        const int ninsn = bc.ninsn, nslots = bc.nslots;
        const int w = tile.width, h = tile.height;

        // Run the program over L1-sized 2D blocks so the intermediates stay in cache. ONE rule covers
        // every tile shape: keep the strip as WIDE as fits L1 (long inner loop => good SIMD + hits the
        // kernels' width-specific branches), then add as many rows as still fit (bw*bh <= capElems).
        // For the dominant 1D tile (w huge, h==1) this is a column strip; for a tall-thin tile
        // (w==channels, h huge - e.g. a masked op) it instead splits the LONG axis (height), so the
        // temp block stays contiguous and the kernels keep full width. With NO temps capElems==INT_MAX
        // => bw=w, bh=h: one block over the whole tile, region temp store is empty (zero bytes), and
        // each operand slot is re-pointed once straight at its tile slice. Each temp buffer occupies
        // [bufEszPrefix[buf]*region, ...) bytes in tstore; region = bw*bh.
        const int* const bufEszPrefix = bc.bufEszPrefix;
        const int bw = std::min(w, bc.capElems);
        const int bh = std::min(h, std::max(1, bc.capElems / std::max(1, bw)));
        const size_t region = alignSize((size_t)bw * bh, 8);
        AutoBuffer<uchar> tstore((size_t)bufEszPrefix[bc.nbuffers] * region);

        AutoBuffer<EwSlice, LOCAL_OPS> args(nslots);
        for (int y0 = 0; y0 < h; y0 += bh)
        {
            const int hf = std::min(bh, h - y0);
            for (int x0 = 0; x0 < w; x0 += bw)
            {
                const int wf = std::min(bw, w - x0);
                for (int s = 1; s < nslots; s++)
                {
                    EwSlice& a = args[s];
                    const int k = slotKind[s];
                    size_t esz = CV_ELEM_SIZE1(arginfo[s].depth);
                    if (k == TExpr::TEMP)                      // contiguous block-local buffer
                    {
                        a.ptr = tstore.data() + (size_t)bufEszPrefix[slotMap[s]] * region;
                        a.stepy = (size_t)wf*esz; a.stepx = 1;
                    }
                    else if (k == TExpr::CONST)               // fixed scalar slice (stepx/stepy 0)
                    {
                        const EwSlice& cs = constSlice[slotMap[s]];
                        a.ptr = (uchar*)cs.ptr; a.stepy = 0; a.stepx = 0;
                    }
                    else                                    // array operand: this block of the slice
                    {
                        const EwSlice& sl = tile.slices[slotMap[s]];
                        a.ptr = (uchar*)sl.ptr +
                            ((size_t)y0 * sl.stepy + (size_t)x0 * sl.stepx) * esz;
                        a.stepy = sl.stepy*esz; a.stepx = sl.stepx;
                    }
                }

                for (int n = 0; n < ninsn; n++)
                {
                    const TExpr::Insn& ins = prog[n];
                    const EwSlice& a0 = args[ins.arg0]; const EwSlice& a1 = args[ins.arg1];
                    const EwSlice& a2 = args[ins.arg2]; const EwSlice& rr = args[ins.result];
                    runInsn(ins, a0.ptr, a0.stepy, a0.stepx, a1.ptr, a1.stepy, a1.stepx,
                            a2.ptr, a2.stepy, a2.stepx, (void*)rr.ptr, rr.stepy, wf, hf);
                }
            }
        }
    }, true, nstripes);
}

// ---------------------------------------------------------------------------
// Manual program builders (stand-ins for the future engine-backed cv::add etc.): they skip the
// source graph and emit the program directly through TExpr's addInput/addTemp/addOutput/addInsn.
// ---------------------------------------------------------------------------

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
void makeBinaryArithProgram(TExpr& p, TOp op, int depth0, int depth1, int rdepth,
                            int maskDepth, double scale)
{
    p.clear();
    if (rdepth < 0) rdepth = promoteArith(depth0, depth1);   // rdepth == -1 => auto (like cv::'s dtype=-1)
    const bool masked = maskDepth != EW_DEPTH_NONE;
    int sIn0  = p.addInput(depth0);
    int sIn1  = p.addInput(depth1);
    int sMask = masked ? p.addInput(maskDepth) : 0;
    int sOut  = p.addOutput(rdepth);

    // Op-family type policy feeding the shared emitTyped core: the compute type C (both operands cast
    // to it) and the wide fallback W (used only if no direct C x C -> rdepth kernel exists):
    //   MIN/MAX/ABSDIFF: C = rdepth, never widen (the same-type T x T -> T kernel always exists).
    //   MUL/DIV:         C = promote2, W = float/double (matches cv::multiply/divide); `scale` rides
    //                    params[0] (mul: a*b*scale; div: a*scale/b). div's /0 guard is implied by C:
    //                    promote2 is integer iff both inputs are, so getElemwiseFunc picks it right.
    //   ADD/SUB:         C = promote2, W = safeWide(C), signed (the difference may go negative).
    int C, W;
    if (op == OP_MIN || op == OP_MAX || op == OP_ABSDIFF) { C = rdepth; W = rdepth; }
    else
    {
        C = (depth0 == depth1) ? depth0 : promote2(depth0, depth1);
        if (op == OP_MUL || op == OP_DIV)
        {
            const bool wide = (C==CV_32U || C==CV_32S || C==CV_64U || C==CV_64S || C==CV_64F);
            W = wide ? CV_64F : CV_32F;
        }
        else
            W = safeWide(C);
    }

    int args[2] = { sIn0, sIn1 };
    if (masked)
    {
        // result into a temp, then copyMask writes the masked subset into the (pre-existing) output:
        // dst = (mask!=0) ? r : dst, matching cv::add/... with a mask.
        int r = p.emitTyped(op, args, 2, C, rdepth, W, Scalar(scale));
        p.addInsn(OP_COPY_MASK, r, sMask, 0, sOut);
    }
    else
        p.emitTyped(op, args, 2, C, rdepth, W, Scalar(scale), sOut);  // straight into the output

    p.compile();    // pack temp buffers + count consts (kernels already bound by addInsn)
}

// addWeighted(a, alpha, b, beta, gamma) = a*alpha + b*beta + gamma, as two fused convert_scale
// MACs + an add (+ a final cast when the output type differs from the working type W):
//   t0 = cast<W>(a*alpha + gamma) ;  t1 = cast<W>(b*beta) ;  out = cast<rdepth>(t0 + t1)
// The convert_scale steps ride core's optimized scale kernel; alpha/beta/gamma travel in each
// instruction's params block ({scale, offset}), not as operands. The 2-3 temps exercise the body's
// L1 column-fragmentation.
void makeAddWeightedProgram(TExpr& p, int depth0, int depth1, int rdepth,
                            double alpha, double beta, double gamma)
{
    p.clear();
    const int W = (depth0 == CV_64F || depth1 == CV_64F || rdepth == CV_64F) ? CV_64F : CV_32F;

    int sA = p.addInput(depth0);
    int sB = p.addInput(depth1);

    int t0 = p.addTemp(W), t1 = p.addTemp(W);
    p.addInsn(OP_CONVERT_SCALE, sA, 0, 0, t0, Scalar(alpha, gamma));   // t0 = a*alpha + gamma
    p.addInsn(OP_CONVERT_SCALE, sB, 0, 0, t1, Scalar(beta, 0.0));      // t1 = b*beta

    int sOut = p.addOutput(rdepth);
    if (rdepth == W)
        p.addInsn(OP_ADD, t0, t1, 0, sOut);
    else
    {
        int t2 = p.addTemp(W);
        p.addInsn(OP_ADD, t0, t1, 0, t2);
        p.addInsn(OP_CAST, t2, 0, 0, sOut);
    }

    p.compile();    // pack temp buffers + count consts (kernels already bound by addInsn)
}

}} // namespace cv::ew
