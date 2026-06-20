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
#include <vector>

// Engine-side mirror declarations of core's exported convert dispatchers (see the CV_EXPORTS in
// core/src/precomp.hpp). A function's return type is not part of its mangled name, so declaring
// these to return EwBinaryFunc links to cv::getConvertFunc / getConvertScaleFunc (which return
// core's identical BinaryFunc) without pulling in the private precomp.hpp.
namespace cv {
ew::EwBinaryFunc getConvertFunc(int sdepth, int ddepth);
ew::EwBinaryFunc getConvertScaleFunc(int sdepth, int ddepth);
}

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
static bool broadcastShape(const std::vector<MatShape>& shps, MatShape& out)
{
    size_t nd = 0;
    for (size_t k = 0; k < shps.size(); k++) nd = std::max(nd, shps[k].size());
    out.assign(nd, 1);
    for (size_t k = 0; k < shps.size(); k++)
    {
        const MatShape& s = shps[k];
        size_t off = nd - s.size();
        for (size_t i = 0; i < s.size(); i++)
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
    case OP_NEG: case OP_ABS: case OP_CAST: case OP_RELU:
    case OP_CMP_EQ: case OP_CMP_NE: case OP_CMP_LT:
    case OP_CMP_LE: case OP_CMP_GT: case OP_CMP_GE: return 1;
    case OP_DIV: case OP_SQRT: case OP_CONVERT_SCALE: return 10;
    case OP_SIN: case OP_COS: case OP_TANH: case OP_ERF:
    case OP_EXP: case OP_LOG: case OP_POW:           return 30;
    default:                                         return 10;
    }
}

void exec(const EwProgram& program,
          const std::vector<Mat>& inputs,
          std::vector<Mat>& outputs)
{
    CV_Assert((int)inputs.size() == program.ninputs);
    const int nslots = (int)program.arginfo.size();
    CV_Assert(nslots >= 1 && program.arginfo[0].kind == ARG_NONE);

    // ---- 1. broadcast result shape (inputs + const channel counts), channels innermost ----
    std::vector<MatShape> inShp(program.ninputs);
    std::vector<EwSteps>  inStep(program.ninputs);
    std::vector<int>      inEsz1(program.ninputs);
    std::vector<MatShape> bshapes;
    for (int i = 0; i < program.ninputs; i++)
    {
        matLogical(inputs[i], inShp[i], inStep[i], inEsz1[i]);
        bshapes.push_back(inShp[i]);
    }
    for (int s = 1; s < nslots; s++)
    {
        const EwArgInfo& ai = program.arginfo[s];
        if (ai.kind != ARG_CONST) continue;
        MatShape cs; cs.assign(1, std::max(ai.channels, 1));   // just the channel dim
        bshapes.push_back(cs);
    }
    MatShape full;
    CV_Assert(broadcastShape(bshapes, full) && "ew: inputs not broadcast-compatible");
    const int ndFull = (int)full.size();
    const int rchannels = full[ndFull - 1];
    MatShape spatial = full; spatial.resize(ndFull - 1);

    // ---- 2. allocate outputs ----
    outputs.resize(program.noutputs);
    for (int s = 1; s < nslots; s++)
    {
        const EwArgInfo& ai = program.arginfo[s];
        if (ai.kind != ARG_OUTPUT) continue;
        outputs[ai.index].create((int)spatial.size(), spatial.data(),
                                 CV_MAKETYPE(ai.depth, rchannels));
    }

    // ---- 3. flat operand list for broadcastOp (inputs + consts-as-Mats + outputs), plus a
    //         slot -> array-index map and a slot -> temp-buffer map ----
    std::vector<Mat> arrays;
    std::vector<Mat> constMats;                 // backing storage for ARG_CONST operands
    std::vector<int> slotToArray(nslots, -1);
    std::vector<int> slotToTemp(nslots, -1);
    constMats.reserve(nslots);
    for (int s = 1; s < nslots; s++)
    {
        const EwArgInfo& ai = program.arginfo[s];
        if (ai.kind == ARG_INPUT)
        {
            slotToArray[s] = (int)arrays.size();
            arrays.push_back(inputs[ai.index]);
        }
        else if (ai.kind == ARG_OUTPUT)
        {
            slotToArray[s] = (int)arrays.size();
            arrays.push_back(outputs[ai.index]);
        }
        else if (ai.kind == ARG_CONST)
        {
            const int c = std::max(ai.channels, 1);
            CV_Assert(c <= 4);
            Mat cm(1, 1, CV_MAKETYPE(ai.depth, c));
            const int e = (int)CV_ELEM_SIZE1(ai.depth);
            for (int k = 0; k < c; k++)
                storeScalar(ai.depth, ai.cval[k], cm.data + (size_t)k * e);
            constMats.push_back(cm);
            slotToArray[s] = (int)arrays.size();
            arrays.push_back(cm);
        }
        else // ARG_TEMP
        {
            slotToTemp[s] = program.bufferOfTemp.empty() ? ai.index
                                                         : program.bufferOfTemp[ai.index];
        }
    }

    // ---- 4. per-instruction adapter contexts (frozen; shared read-only across threads) ----
    const EwInsn* prog = program.prog.data();
    const int ninsn = (int)program.prog.size();
    AutoBuffer<EwCtx> ctxs(ninsn);
    AutoBuffer<void*> ctxptr(ninsn);
    for (int n = 0; n < ninsn; n++)
    {
        const EwInsn& ins = prog[n];
        void* cx = nullptr;
        if (ins.op == OP_CAST)
        {
            const int sd = program.arginfo[ins.arg0].depth;
            const int dd = program.arginfo[ins.result].depth;
            EwCtx& c = ctxs[n];
            c.cvt.fn = cv::getConvertFunc(sd, dd);
            CV_Assert(c.cvt.fn != nullptr);
            c.cvt.sesz1 = (int)CV_ELEM_SIZE1(sd);
            c.cvt.desz1 = (int)CV_ELEM_SIZE1(dd);
            c.cvt.scale[0] = 1.0; c.cvt.scale[1] = 0.0;     // unused by plain convert
            cx = &c;
        }
        ctxptr[n] = cx;
    }

    // ---- 5. per-temp-buffer element size ----
    const int nbuffers = program.nbuffers;
    AutoBuffer<int> bufEsz(std::max(1, nbuffers));
    for (int b = 0; b < nbuffers; b++) bufEsz[b] = 1;
    for (int s = 1; s < nslots; s++)
        if (program.arginfo[s].kind == ARG_TEMP)
        {
            const int b = slotToTemp[s];
            bufEsz[b] = std::max(bufEsz[b], (int)CV_ELEM_SIZE1(program.arginfo[s].depth));
        }

    // ---- 6. parallel work hint: total output scalars x summed per-element op cost / budget.
    //         (Without this, broadcastOp's default ~100 cyc/elem over-splits cheap ops.) ----
    long long otot = 1;
    for (int d = 0; d < ndFull; d++) otot *= full[d];
    long long costPerElem = 0;
    for (int n = 0; n < ninsn; n++) costPerElem += opCost(prog[n].op);
    const double nstripes = (double)otot * (double)std::max<long long>(costPerElem, 1)
                            / (double)(1 << 18);

    // ---- 7. drive: broadcastOp does geometry + 2D tiling + parallelism; the body runs the
    //         frozen program on each tile (temps tile-local, re-pointed from the tile slices).
    //         expandChannels=true => the body always sees single-channel data. ----
    broadcastOp(arrays.data(), arrays.size(), [&](const EwTile& tile)
    {
        const int w = tile.width, h = tile.height;

        // temp buffers for this tile: contiguous [h x w], stepy=w, stepx=1
        AutoBuffer<size_t> tofs(std::max(1, nbuffers));
        size_t tbytes = 0;
        for (int b = 0; b < nbuffers; b++)
        {
            tofs[b] = tbytes;
            tbytes += (size_t)w * h * bufEsz[b];
        }
        AutoBuffer<uchar> tstore(tbytes);

        AutoBuffer<EwArg> args(nslots);                 // slot 0 stays the null operand
        for (int s = 1; s < nslots; s++)
        {
            EwArg& a = args[s];
            const int ia = slotToArray[s];
            if (ia >= 0)
            {
                const EwSlice& sl = tile.slices[ia];
                a.ptr = sl.ptr; a.stepy = sl.stepy; a.stepx = sl.stepx;
            }
            else                                        // ARG_TEMP
            {
                a.ptr = tstore.data() + tofs[slotToTemp[s]];
                a.stepy = (size_t)w; a.stepx = 1;
            }
        }

        for (int n = 0; n < ninsn; n++)
        {
            const EwInsn& ins = prog[n];
            const EwArg& a0 = args[ins.arg0];
            const EwArg& a1 = args[ins.arg1];
            const EwArg& a2 = args[ins.arg2];
            const EwArg& rr = args[ins.result];
            int code = ins.fptr(a0.ptr, a0.stepy, a0.stepx,
                                a1.ptr, a1.stepy, a1.stepx,
                                a2.ptr, a2.stepy, a2.stepx,
                                (void*)rr.ptr, rr.stepy, w, h, ctxptr[n]);
            CV_Assert(code >= 0);
        }
    }, /*expandChannels*/ true, nstripes);
}

// ---------------------------------------------------------------------------
// Manual program builders.
// ---------------------------------------------------------------------------
static EwArgInfo mkArg(ArgKind kind, int depth, int index)
{
    EwArgInfo a; a.kind = kind; a.depth = depth; a.index = index; return a;
}

EwProgram makeBinaryProgram(ElemwiseOp op, int depth0, int depth1, int rdepth)
{
    CV_Assert(opArity(op) == 2);
    EwProgram p;
    p.ninputs = 2; p.noutputs = 1; p.ntemps = 0; p.nbuffers = 0;
    p.arginfo.resize(4);
    p.arginfo[0] = mkArg(ARG_NONE,   EW_DEPTH_NONE, -1);
    p.arginfo[1] = mkArg(ARG_INPUT,  depth0, 0);
    p.arginfo[2] = mkArg(ARG_INPUT,  depth1, 1);
    p.arginfo[3] = mkArg(ARG_OUTPUT, rdepth, 0);

    EwInsn ins;
    ins.op = op; ins.arg0 = 1; ins.arg1 = 2; ins.arg2 = 0; ins.result = 3;
    ins.fptr = getElemwiseFunc(op, depth0, depth1, EW_DEPTH_NONE, rdepth);
    CV_Assert(ins.fptr != nullptr);
    p.prog.push_back(ins);
    return p;
}

EwProgram makeUnaryProgram(ElemwiseOp op, int depth0, int rdepth)
{
    CV_Assert(opArity(op) == 1);
    EwProgram p;
    p.ninputs = 1; p.noutputs = 1; p.ntemps = 0; p.nbuffers = 0;
    p.arginfo.resize(3);
    p.arginfo[0] = mkArg(ARG_NONE,   EW_DEPTH_NONE, -1);
    p.arginfo[1] = mkArg(ARG_INPUT,  depth0, 0);
    p.arginfo[2] = mkArg(ARG_OUTPUT, rdepth, 0);

    EwInsn ins;
    ins.op = op; ins.arg0 = 1; ins.arg1 = 0; ins.arg2 = 0; ins.result = 2;
    ins.fptr = getElemwiseFunc(op, depth0, EW_DEPTH_NONE, EW_DEPTH_NONE, rdepth);
    CV_Assert(ins.fptr != nullptr);
    p.prog.push_back(ins);
    return p;
}

// [VP] looks like there is a bug here. rank(8s) < rank(16u),
// so when we add 8s to 16u, the coercion type will be 16u,
// however it should be 32s. We need to take into account signness of integer types.
// also, when we subtract 8u from 8s, the result should have 16s type.
// the coercion rule should be:
// * if a == b -> a
// * else if a == 64f || b == 64f -> 64f
// * else if flt(a) || flt(b) -> 32f
// * else if max(a, b) <= 8s -> 16s
// * else if max(a, b) <= 16s -> 32s
// * else if a == 32s || a == 32u || b == 32s || b == 32u -> 64s
// * else 64f
//
// or
//
// if (a == b) return a;
// const uint64_t typelut = (0 << CV_8U*3) | (0 << CV_8S*3) | (1 << CV_16U*3) | (1 << 16S*3) |
//                          (2 << CV_32U*3) | (2 << CV_32S*3) | (3 << CV_16F*3) | (3 << CV_16BF*3) |
//                          (3 << CV_32F*3) | (4 << CV_64F*3) | (4 << CV_64S*3) | (4 << CV_64U*3);
// int idxa = int((typelut >> (a*3)) & 7);
// int idxb = int((typelut >> (b*3)) & 7);
// int maxab = std::max(idxa, idxb);
// const int ctypelut = ((CV_16S << 0*5) | (CV_32S << 1*5) | (CV_64S << 2*5) |
//                       (CV_32F << 3*5) | (CV_64F << 4*5));
// return (ctypelut >> (maxab*5)) & 31;

// numpy-ish promotion of two known depths (float dominates; wider integer otherwise).
static int promote2(int a, int b)
{
    if (a == b) return a;
    auto isF = [](int d){ return d==CV_16F || d==CV_16BF || d==CV_32F || d==CV_64F; };
    if (isF(a) || isF(b))
        return (a==CV_64F || b==CV_64F) ? CV_64F : CV_32F;
    auto rank = [](int d){ switch(d){
        case CV_8U: case CV_8S: return 1;
        case CV_16U: case CV_16S: return 2;
        case CV_32U: case CV_32S: return 3;
        case CV_64U: case CV_64S: return 4; default: return 0; }; };
    return rank(a) >= rank(b) ? a : b;
}

// A wide type in which add(C,C->W) exists and the sum is held without a premature clamp.
static int safeWide(int C)
{
    switch (C)
    {
    case CV_32U: case CV_32S: case CV_64U: case CV_64S: case CV_64F: return CV_64F;
    default:                                                         return CV_32F;
    }
}

EwProgram makeAddProgram(int depth0, int depth1, int rdepth)
{
    EwProgram p;
    p.ninputs = 2; p.noutputs = 1;
    p.arginfo.push_back(mkArg(ARG_NONE,  EW_DEPTH_NONE, -1));   // slot 0
    int sIn0 = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, depth0, 0));
    int sIn1 = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_INPUT, depth1, 1));

    int ntemps = 0;
    auto addTemp = [&](int depth) {
        int s = (int)p.arginfo.size();
        p.arginfo.push_back(mkArg(ARG_TEMP, depth, ntemps++));
        return s;
    };
    auto addInsn = [&](ElemwiseOp op, int a0, int a1, int r) {
        EwInsn ins; ins.op = op; ins.arg0 = a0; ins.arg1 = a1; ins.arg2 = 0; ins.result = r;
        ins.fptr = getElemwiseFunc(op, p.arginfo[a0].depth,
                                   a1 ? p.arginfo[a1].depth : EW_DEPTH_NONE,
                                   EW_DEPTH_NONE, p.arginfo[r].depth);
        CV_Assert(ins.fptr != nullptr);
        p.prog.push_back(ins);
    };

    // 1. bring both operands to a common type C.
    int C = (depth0 == depth1) ? depth0 : promote2(depth0, depth1);
    int op0 = sIn0, op1 = sIn1;
    if (depth0 != C) { int t = addTemp(C); addInsn(OP_CAST, sIn0, 0, t); op0 = t; }
    if (depth1 != C) { int t = addTemp(C); addInsn(OP_CAST, sIn1, 0, t); op1 = t; }

    // 2. add directly to rdepth if a kernel exists, else add wide then cast down.
    int sOut = (int)p.arginfo.size(); p.arginfo.push_back(mkArg(ARG_OUTPUT, rdepth, 0));
    if (getElemwiseFunc(OP_ADD, C, C, EW_DEPTH_NONE, rdepth))
        addInsn(OP_ADD, op0, op1, sOut);
    else
    {
        int W = safeWide(C);
        int tW = addTemp(W);
        addInsn(OP_ADD, op0, op1, tW);
        addInsn(OP_CAST, tW, 0, sOut);
    }

    p.ntemps = ntemps; p.nbuffers = ntemps;
    p.bufferOfTemp.resize(ntemps);
    for (int i = 0; i < ntemps; i++) p.bufferOfTemp[i] = i;
    return p;
}

}} // namespace cv::ew
