// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 3 implementation: see ew_exec.hpp.

#include "ew_exec.hpp"
#include <algorithm>

namespace cv { namespace ew {

// ---------------------------------------------------------------------------
// Per-slot execution plan (shape-collapsed, in elemsize1 units).
// ---------------------------------------------------------------------------
struct SlotPlan
{
    ArgKind kind = ARG_NONE;
    int depth = EW_DEPTH_NONE;
    int esz1 = 0;                  // size of one scalar (one channel value), bytes
    uchar* base = nullptr;         // input/output/const data; temps are per-thread
    int tempBuf = -1;              // physical temp buffer id (ARG_TEMP only)
    EwSteps step{};                // collapsed steps (length m); 0 => broadcast
    std::vector<uchar> constStore; // ARG_CONST backing storage
};

// Logical shape of a Mat with channels as the innermost dimension; steps in elemsize1 units.
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

// Right-align an arg's own (shp,step) to nd dims; broadcast dims get step 0.
static void alignArg(const MatShape& shp, const EwSteps& step, int nd,
                     EwSteps& as, MatShape& ash)
{
    as.fill(0);
    ash.assign(nd, 1);
    int off = nd - (int)shp.size();
    for (int i = 0; i < (int)shp.size(); i++)
    {
        int d = shp[i];
        ash[off + i] = d;
        as[off + i] = (d == 1) ? 0 : step[i];
    }
}

// Collapse adjacent dims that are contiguous (and broadcast-consistent) across all args.
// S/H are per-arg step/shape arrays; D is the iteration extent. Returns collapsed ndims.
static int collapseDims(std::vector<EwSteps>& S,
                        std::vector<MatShape>& H,
                        MatShape& D)
{
    int K = (int)S.size();
    int nd = (int)D.size();
    if (nd <= 1) return nd;

    int j = nd - 1;
    for (int i = j - 1; i >= 0; i--)
    {
        bool contig = true, scalar = true, consist = true;
        for (int k = 0; k < K; k++)
        {
            size_t st = S[k][j] * (size_t)H[k][j];
            bool prevScalar = H[k][j] == 1;
            bool curScalar = H[k][i] == 1;
            contig  = contig  && (st == S[k][i]);
            scalar  = scalar  && curScalar;
            consist = consist && (curScalar == prevScalar);
        }
        if (contig && (consist || scalar))
        {
            for (int k = 0; k < K; k++) H[k][j] *= H[k][i];
            D[j] *= D[i];
        }
        else
        {
            j--;
            if (i < j)
            {
                for (int k = 0; k < K; k++) { H[k][j] = H[k][i]; S[k][j] = S[k][i]; }
                D[j] = D[i];
            }
        }
    }

    int m = nd - j;
    for (int d = 0; d < m; d++)
    {
        D[d] = D[j + d];
        for (int k = 0; k < K; k++) { S[k][d] = S[k][j + d]; H[k][d] = H[k][j + d]; }
    }
    D.resize(m);
    for (int k = 0; k < K; k++) H[k].resize(m);  // S[k] is fixed-size; only [0..m) is read

    // Zero out steps of broadcast (size-1) dims (numpy step==0 trick).
    for (int d = 0; d < m; d++)
        for (int k = 0; k < K; k++)
            if (H[k][d] == 1) S[k][d] = 0;

    return m;
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

void exec(const EwProgram& program,
          const std::vector<Mat>& inputs,
          std::vector<Mat>& outputs)
{
    CV_Assert((int)inputs.size() == program.ninputs);
    const int nslots = (int)program.arginfo.size();
    CV_Assert(nslots >= 1 && program.arginfo[0].kind == ARG_NONE);

    // ---- 1. logical shapes of inputs, broadcast to the full result shape ----
    std::vector<MatShape> inShp(program.ninputs);
    std::vector<EwSteps> inStep(program.ninputs);
    std::vector<int> inEsz1(program.ninputs);
    // Local input headers. An input whose only unit-stride axis is a size-1 dim (e.g. a cropped
    // array adjacent to a dropped channel/size-1 axis) would expose a gapped axis as innermost;
    // kernels require innermost stride in {0,1}, so such inputs are materialized contiguous.
    // Row-gapped ROIs (unit-stride innermost) are kept as-is and handled via stepy.
    std::vector<Mat> ins(inputs.begin(), inputs.end());
    for (int i = 0; i < program.ninputs; i++)
    {
        matLogical(ins[i], inShp[i], inStep[i], inEsz1[i]);
        bool anyBig = false, hasUnit = false;
        for (size_t d = 0; d < inShp[i].size(); d++)
            if (inShp[i][d] > 1) { anyBig = true; if (inStep[i][d] == 1) hasUnit = true; }
        if (anyBig && !hasUnit)
        {
            ins[i] = ins[i].clone();
            matLogical(ins[i], inShp[i], inStep[i], inEsz1[i]);
        }
    }

    MatShape full;
    CV_Assert(broadcastShape(inShp, full));      // includes the channel dim (last)
    const int ndFull = (int)full.size();
    const int rchannels = full[ndFull - 1];
    MatShape spatial = full; spatial.resize(ndFull - 1);  // result shape without channels

    // ---- 2. allocate outputs ----
    outputs.resize(program.noutputs);
    for (int s = 1; s < nslots; s++)
    {
        const EwArgInfo& ai = program.arginfo[s];
        if (ai.kind != ARG_OUTPUT) continue;
        outputs[ai.index].create((int)spatial.size(), spatial.data(),
                                 CV_MAKETYPE(ai.depth, rchannels));
    }

    // ---- 3. build per-slot aligned shape/step; gather collapse set ----
    std::vector<SlotPlan> plan(nslots);
    std::vector<EwSteps> S;     // collapse-set steps
    std::vector<MatShape> H;    // collapse-set shapes
    std::vector<int> setSlot;   // collapse-set -> slot index
    MatShape D = full;          // iteration extent (collapsed in place)

    for (int s = 1; s < nslots; s++)
    {
        const EwArgInfo& ai = program.arginfo[s];
        SlotPlan& sp = plan[s];
        sp.kind = ai.kind;
        sp.depth = ai.depth;

        MatShape shp;
        EwSteps step{};

        if (ai.kind == ARG_INPUT)
        {
            shp = inShp[ai.index]; step = inStep[ai.index];
            sp.esz1 = inEsz1[ai.index];
            sp.base = (uchar*)ins[ai.index].data;
        }
        else if (ai.kind == ARG_OUTPUT)
        {
            int e; matLogical(outputs[ai.index], shp, step, e);
            sp.esz1 = e;
            sp.base = (uchar*)outputs[ai.index].data;
        }
        else if (ai.kind == ARG_CONST)
        {
            sp.esz1 = (int)CV_ELEM_SIZE1(ai.depth);
            int c = std::max(ai.channels, 1);
            CV_Assert(c <= 4);   // per-channel const fits in a cv::Scalar
            sp.constStore.resize((size_t)c * sp.esz1);
            for (int k = 0; k < c; k++)
                storeScalar(ai.depth, ai.cval[k], sp.constStore.data() + (size_t)k * sp.esz1);
            sp.base = sp.constStore.data();
            // logical shape [1,...,1,C]; channel step 1 if per-channel, else fully broadcast
            shp.assign(ndFull, 1); shp[ndFull - 1] = c;
            step.fill(0); step[ndFull - 1] = (c > 1) ? 1 : 0;
        }
        else // ARG_TEMP
        {
            sp.esz1 = (int)CV_ELEM_SIZE1(ai.depth);
            sp.tempBuf = program.bufferOfTemp.empty() ? ai.index
                                                      : program.bufferOfTemp[ai.index];
            continue; // temps are tile-local; not part of broadcast/collapse
        }

        EwSteps as{}; MatShape ash;
        alignArg(shp, step, ndFull, as, ash);
        S.push_back(as); H.push_back(ash); setSlot.push_back(s);
    }

    // Drop size-1 iteration dims (e.g. the channel dim of single-channel arrays, or any
    // broadcast axis that ended up size 1) so the innermost dim is a real extent and tiles
    // aren't degenerate width-1. All operands are size-1 there too, so this is offset-neutral.
    {
        const int K = (int)S.size();
        int w = 0;
        for (int d = 0; d < ndFull; d++)
        {
            if (D[d] == 1) continue;
            D[w] = D[d];
            for (int k = 0; k < K; k++) { S[k][w] = S[k][d]; H[k][w] = H[k][d]; }
            w++;
        }
        if (w == 0)   // fully scalar result: keep one trivial dim
        {
            D[0] = 1;
            for (int k = 0; k < K; k++) { S[k][0] = 0; H[k][0] = 1; }
            w = 1;
        }
        D.resize(w);
        for (int k = 0; k < K; k++) H[k].resize(w);
    }

    // ---- 4. collapse dims across the whole arg set ----
    int m = collapseDims(S, H, D);
    for (size_t t = 0; t < setSlot.size(); t++)
        plan[setSlot[t]].step = S[t];

    // Kernels handle only innermost stride 0 (broadcast) or 1 (contiguous) - no gather.
    // The (rare) operand with a non-unit innermost stride must be materialized contiguous;
    // for now we assert (the copy is a TODO once such an input actually shows up).
    for (size_t t = 0; t < setSlot.size(); t++)
        CV_Assert(plan[setSlot[t]].step[m - 1] <= 1 &&
                  "ew: operand has non-unit innermost stride (materialize a contiguous copy)");

    // ---- 5. tiling parameters ----
    const int W = D[m - 1];
    long long nplanes = 1;
    for (int d = 0; d < m - 1; d++) nplanes *= D[d];

    int maxBufEsz = 1;
    std::vector<int> bufEsz(program.nbuffers, 1);
    for (int s = 1; s < nslots; s++)
        if (plan[s].kind == ARG_TEMP && plan[s].tempBuf >= 0)
        {
            bufEsz[plan[s].tempBuf] = std::max(bufEsz[plan[s].tempBuf], plan[s].esz1);
            maxBufEsz = std::max(maxBufEsz, plan[s].esz1);
        }

    int tileW;
    if (program.nbuffers > 0)
    {
        const size_t l1budget = 16 * 1024;
        tileW = (int)std::max<size_t>(256, l1budget / ((size_t)program.nbuffers * maxBufEsz));
    }
    else
        tileW = 1 << 14;
    tileW = std::max(1, std::min(tileW, W));

    const int ntilesW = (W + tileW - 1) / tileW;
    const long long ntiles = nplanes * ntilesW;
    CV_Assert(ntiles <= (long long)INT_MAX);

    // ---- 6. parallel execution; each worker re-points args at its tile ----
    const EwInsn* prog = program.prog.data();
    const int ninsn = (int)program.prog.size();

    parallel_for_(Range(0, (int)ntiles), [&](const Range& r)
    {
        std::vector<EwArg> args(nslots);
        std::vector<std::vector<uchar> > tbuf(program.nbuffers);
        for (int b = 0; b < program.nbuffers; b++)
            tbuf[b].resize((size_t)tileW * bufEsz[b]);

        std::vector<int> idx(m > 1 ? m - 1 : 0);

        for (int t = r.start; t < r.end; t++)
        {
            int plane = t / ntilesW;
            int wt = t % ntilesW;
            int wofs = wt * tileW;
            int w = std::min(tileW, W - wofs);

            // decode plane index into per-dim indices
            int p = plane;
            for (int d = m - 2; d >= 0; d--) { idx[d] = p % D[d]; p /= D[d]; }

            for (int s = 1; s < nslots; s++)
            {
                const SlotPlan& sp = plan[s];
                EwArg& a = args[s];
                a.depth = sp.depth; a.channels = 0;
                if (sp.kind == ARG_TEMP)
                {
                    a.ptr = tbuf[sp.tempBuf].data();
                    a.stepy = 0; a.stepx = 1;
                    continue;
                }
                size_t off = 0;
                for (int d = 0; d < m - 1; d++) off += (size_t)idx[d] * sp.step[d];
                off += (size_t)wofs * sp.step[m - 1];
                a.ptr = sp.base + off * sp.esz1;
                a.stepy = 0;                  // height==1 for now (no 2D tiling yet)
                a.stepx = sp.step[m - 1];
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
                                    (void*)rr.ptr, rr.stepy, w, 1);
                CV_Assert(code >= 0);
            }
        }
    });
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
