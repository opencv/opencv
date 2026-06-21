// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 2 implementation: see ew_compile.hpp.

#include "ew_compile.hpp"
#include <algorithm>

namespace cv { namespace ew {

// ---------------------------------------------------------------------------
// Graph builder helpers.
// ---------------------------------------------------------------------------
int EwGraph::input(int idx)
{
    EwNode n; n.kind = NODE_INPUT; n.inputIndex = idx;
    ninputs = std::max(ninputs, idx + 1);
    nodes.push_back(n); return (int)nodes.size() - 1;
}

int EwGraph::constant(const Scalar& v, int channels)
{
    EwNode n; n.kind = NODE_CONST; n.cval = v; n.cchannels = channels;
    nodes.push_back(n); return (int)nodes.size() - 1;
}

int EwGraph::unary(ElemwiseOp op, int a)
{
    CV_Assert(opArity(op) == 1);
    EwNode n; n.kind = NODE_OP; n.op = op; n.args[0] = a; n.nargs = 1;
    nodes.push_back(n); return (int)nodes.size() - 1;
}

int EwGraph::binary(ElemwiseOp op, int a, int b)
{
    CV_Assert(opArity(op) == 2);
    EwNode n; n.kind = NODE_OP; n.op = op; n.args[0] = a; n.args[1] = b; n.nargs = 2;
    nodes.push_back(n); return (int)nodes.size() - 1;
}

int EwGraph::ternary(ElemwiseOp op, int a, int b, int c)
{
    CV_Assert(opArity(op) == 3);
    EwNode n; n.kind = NODE_OP; n.op = op; n.args[0] = a; n.args[1] = b; n.args[2] = c; n.nargs = 3;
    nodes.push_back(n); return (int)nodes.size() - 1;
}

int EwGraph::cast(int a, int depth)
{
    EwNode n; n.kind = NODE_OP; n.op = OP_CAST; n.args[0] = a; n.nargs = 1; n.castDepth = depth;
    nodes.push_back(n); return (int)nodes.size() - 1;
}

// ---------------------------------------------------------------------------
// Type inference helpers (deliberately small so they can grow later).
// ---------------------------------------------------------------------------
static bool isFloatDepth(int d)
{
    return d == CV_16F || d == CV_16BF || d == CV_32F || d == CV_64F;
}

static int depthRank(int d)
{
    switch (d)
    {
    case CV_8U: case CV_8S:                          return 1;
    case CV_16U: case CV_16S: case CV_16F: case CV_16BF: return 2;
    case CV_32U: case CV_32S: case CV_32F:           return 3;
    case CV_64U: case CV_64S: case CV_64F:           return 4;
    default:                                         return 0;
    }
}

// numpy-ish promotion with OpenCV same-type preference. A flexible operand (a const, or an
// unused operand) passes EW_DEPTH_NONE and does not force promotion.
static int promoteArith(int a, int b)
{
    if (a == EW_DEPTH_NONE) return b;
    if (b == EW_DEPTH_NONE) return a;
    if (a == b) return a;
    if (isFloatDepth(a) || isFloatDepth(b))
        return (a == CV_64F || b == CV_64F) ? CV_64F : CV_32F;
    return depthRank(a) >= depthRank(b) ? a : b;
}

// The depth in which we actually compute an op when no exact kernel exists for the natural
// types. For now everything numeric collapses to f32 (our only arithmetic kernels), integers
// stay integer for bitwise. This is where richer per-type kernels will plug in later.
static int workingDepth(ElemwiseOp op)
{
    return opCategory(op) == CAT_BITWISE ? CV_32S : CV_32F;
}

// Natural result depth of a node (const stays flexible = EW_DEPTH_NONE).
static int inferNodeDepth(const EwNode& n, int d0, int d1, int d2)
{
    if (n.kind == NODE_INPUT) return n.depth;   // already set
    if (n.kind == NODE_CONST) return EW_DEPTH_NONE;

    int r;
    switch (opCategory(n.op))
    {
    case CAT_CAST:    r = n.castDepth; break;                          // OP_CAST / convert_scale
    case CAT_MATH:    r = (d0 == CV_64F) ? CV_64F : CV_32F; break;     // float domain
    case CAT_COMPARE: r = (n.castDepth != EW_DEPTH_NONE) ? n.castDepth : CV_8U; break;
    case CAT_BITWISE: r = promoteArith(d0, d1); break;
    case CAT_SELECT:  r = promoteArith(d1, d2); break;                 // mask=arg0, data=arg1,arg2
    case CAT_ARITH:
    default:          r = (n.nargs == 1) ? d0 : promoteArith(d0, d1); break;
    }
    if (r == EW_DEPTH_NONE) r = CV_32F;        // e.g. const (op) const
    return r;
}

// ---------------------------------------------------------------------------
// compile(): lower the graph into a frozen EwProgram.
// ---------------------------------------------------------------------------
void compile(const EwGraph& g, EwProgram& prog, const int* inputDepths, size_t ninputs)
{
    CV_Assert((int)ninputs >= g.ninputs);
    const int nnodes = (int)g.nodes.size();

    // ---- 1. infer each node's natural depth (children before parents) ----
    std::vector<int> depth(nnodes, EW_DEPTH_NONE);
    for (int i = 0; i < nnodes; i++)
    {
        const EwNode& n = g.nodes[i];
        if (n.kind == NODE_INPUT) { depth[i] = inputDepths[n.inputIndex]; continue; }
        if (n.kind == NODE_CONST) { depth[i] = EW_DEPTH_NONE; continue; }
        // a const child contributes EW_DEPTH_NONE (flexible) to promotion
        int cd[3] = { EW_DEPTH_NONE, EW_DEPTH_NONE, EW_DEPTH_NONE };
        for (int k = 0; k < n.nargs; k++)
        {
            int c = n.args[k];
            cd[k] = (g.nodes[c].kind == NODE_CONST) ? EW_DEPTH_NONE : depth[c];
        }
        depth[i] = inferNodeDepth(n, cd[0], cd[1], cd[2]);
    }

    // ---- 2. reachability from outputs (drops orphan nodes left by parser backtracking) ----
    std::vector<char> reachable(nnodes, 0);
    for (size_t o = 0; o < g.outputs.size(); o++) reachable[g.outputs[o]] = 1;
    for (int i = nnodes - 1; i >= 0; i--)
        if (reachable[i])
            for (int k = 0; k < g.nodes[i].nargs; k++) reachable[g.nodes[i].args[k]] = 1;

    // ---- precompute use counts + output references (for direct-to-output writes) ----
    std::vector<int> useCount(nnodes, 0), outRef(nnodes, 0);
    for (int i = 0; i < nnodes; i++)
        if (reachable[i])
            for (int k = 0; k < g.nodes[i].nargs; k++)
                useCount[g.nodes[i].args[k]]++;
    for (size_t o = 0; o < g.outputs.size(); o++)
        outRef[g.outputs[o]]++;

    // ---- 3. set up the program & slot table ----
    prog.clear();
    prog.ninputs = g.ninputs;
    prog.noutputs = (int)g.outputs.size();

    auto addSlot = [&](ArgKind kind, int d, int index) -> int
    {
        EwArgInfo ai; ai.kind = kind; ai.depth = d; ai.index = index;
        prog.arginfo.push_back(ai); return (int)prog.arginfo.size() - 1;
    };

    addSlot(ARG_NONE, EW_DEPTH_NONE, -1);                       // slot 0

    std::vector<int> inputSlot(g.ninputs);
    for (int i = 0; i < g.ninputs; i++)
        inputSlot[i] = addSlot(ARG_INPUT, inputDepths[i], i);

    std::vector<int> outSlot(g.outputs.size());
    for (size_t o = 0; o < g.outputs.size(); o++)
        outSlot[o] = addSlot(ARG_OUTPUT, depth[g.outputs[o]], (int)o);

    // a node may be written straight into an output slot iff it is an op consumed nowhere
    // else and used by exactly one output (otherwise it is materialized and copied).
    std::vector<int> destOfNode(nnodes, -1);
    for (size_t o = 0; o < g.outputs.size(); o++)
    {
        int on = g.outputs[o];
        if (g.nodes[on].kind == NODE_OP && useCount[on] == 0 && outRef[on] == 1)
            destOfNode[on] = outSlot[o];
    }

    int ntemps = 0;
    auto newTemp = [&](int d) -> int { int id = ntemps++; return addSlot(ARG_TEMP, d, id); };

    auto emit = [&](ElemwiseOp op, int a0, int a1, int a2, int result)
    {
        EwInsn ins; ins.op = op; ins.arg0 = a0; ins.arg1 = a1; ins.arg2 = a2; ins.result = result;
        int da0 = a0 ? prog.arginfo[a0].depth : EW_DEPTH_NONE;
        int da1 = a1 ? prog.arginfo[a1].depth : EW_DEPTH_NONE;
        int da2 = a2 ? prog.arginfo[a2].depth : EW_DEPTH_NONE;
        ins.fptr = getElemwiseFunc(op, da0, da1, da2, prog.arginfo[result].depth);
        CV_Assert(ins.fptr != nullptr);
        prog.prog.push_back(ins);
    };

    // cast a value already living in `slot` to depth d (no-op if already there).
    auto castSlot = [&](int slot, int d) -> int
    {
        if (prog.arginfo[slot].depth == d) return slot;
        int t = newTemp(d);
        emit(OP_CAST, slot, 0, 0, t);
        return t;
    };

    std::vector<int> nodeSlot(nnodes, 0);

    // produce a slot holding child node `c` at depth `d` (const -> fresh ARG_CONST slot).
    auto lowerOperand = [&](int c, int d) -> int
    {
        if (g.nodes[c].kind == NODE_CONST)
        {
            int cs = addSlot(ARG_CONST, d, -1);
            prog.arginfo[cs].cval = g.nodes[c].cval;
            prog.arginfo[cs].channels = g.nodes[c].cchannels;
            return cs;
        }
        return castSlot(nodeSlot[c], d);
    };

    // lower one op node; `dest` >= 0 means write the final value straight into that slot.
    auto lowerOp = [&](int i, int dest)
    {
        const EwNode& n = g.nodes[i];
        const int rd = depth[i];
        auto finalSlot = [&](int d) { return dest >= 0 ? dest : newTemp(d); };

        // operand "natural" depths (const -> flexible)
        int cd[3] = { EW_DEPTH_NONE, EW_DEPTH_NONE, EW_DEPTH_NONE };
        for (int k = 0; k < n.nargs; k++)
            cd[k] = (g.nodes[n.args[k]].kind == NODE_CONST) ? EW_DEPTH_NONE : depth[n.args[k]];

        // 1) try a direct kernel at the natural operand depths (const takes the result depth)
        int td[3] = { EW_DEPTH_NONE, EW_DEPTH_NONE, EW_DEPTH_NONE };
        for (int k = 0; k < n.nargs; k++) td[k] = (cd[k] == EW_DEPTH_NONE) ? rd : cd[k];

        int op0, op1, op2;
        if (getElemwiseFunc(n.op, td[0], td[1], td[2], rd))
        {
            op0 = n.nargs > 0 ? lowerOperand(n.args[0], td[0]) : 0;
            op1 = n.nargs > 1 ? lowerOperand(n.args[1], td[1]) : 0;
            op2 = n.nargs > 2 ? lowerOperand(n.args[2], td[2]) : 0;
            int res = finalSlot(rd);
            emit(n.op, op0, op1, op2, res);
            nodeSlot[i] = res;
            return;
        }

        // 2) fall back to a working depth, casting operands; cast the result back to rd if needed
        const int W = workingDepth(n.op);
        const int wr = (opCategory(n.op) == CAT_COMPARE) ? rd : W;
        int w0 = n.nargs > 0 ? W : EW_DEPTH_NONE;
        int w1 = n.nargs > 1 ? W : EW_DEPTH_NONE;
        int w2 = n.nargs > 2 ? W : EW_DEPTH_NONE;
        CV_Assert(getElemwiseFunc(n.op, w0, w1, w2, wr) != nullptr);

        op0 = n.nargs > 0 ? lowerOperand(n.args[0], W) : 0;
        op1 = n.nargs > 1 ? lowerOperand(n.args[1], W) : 0;
        op2 = n.nargs > 2 ? lowerOperand(n.args[2], W) : 0;

        if (wr == rd)
        {
            int res = finalSlot(rd);
            emit(n.op, op0, op1, op2, res);
            nodeSlot[i] = res;
        }
        else
        {
            int wSlot = newTemp(wr);
            emit(n.op, op0, op1, op2, wSlot);
            int res = finalSlot(rd);
            emit(OP_CAST, wSlot, 0, 0, res);
            nodeSlot[i] = res;
        }
    };

    // ---- 4. lower every node in evaluation order ----
    for (int i = 0; i < nnodes; i++)
    {
        const EwNode& n = g.nodes[i];
        if (n.kind == NODE_INPUT)      nodeSlot[i] = inputSlot[n.inputIndex];
        else if (n.kind == NODE_CONST) nodeSlot[i] = 0;          // resolved per use
        else                           lowerOp(i, destOfNode[i]);
    }

    // ---- 5. outputs that were not written directly: copy/cast into their output slot ----
    for (size_t o = 0; o < g.outputs.size(); o++)
    {
        int on = g.outputs[o];
        if (destOfNode[on] == outSlot[o]) continue;             // already written in place
        int src = lowerOperand(on, depth[on]);                  // const/input/multi-use op value
        emit(OP_CAST, src, 0, 0, outSlot[o]);                   // same depth => identity copy
    }

    // ---- 6. liveness: pack temps into a minimal set of reusable physical buffers ----
    const int nslots = (int)prog.arginfo.size();
    std::vector<int> tempOfSlot(nslots, -1);
    for (int s = 1; s < nslots; s++)
        if (prog.arginfo[s].kind == ARG_TEMP) tempOfSlot[s] = prog.arginfo[s].index;

    const int ninsn = (int)prog.prog.size();
    std::vector<int> lastUse(ntemps, -1);
    for (int i = 0; i < ninsn; i++)
    {
        const EwInsn& ins = prog.prog[i];
        int as[3] = { ins.arg0, ins.arg1, ins.arg2 };
        for (int k = 0; k < 3; k++)
            if (tempOfSlot[as[k]] >= 0) lastUse[tempOfSlot[as[k]]] = i;
    }

    prog.bufferOfTemp.resize(ntemps);
    for (int t = 0; t < ntemps; t++) prog.bufferOfTemp[t] = -1;
    std::vector<int> freeBufs;
    int nbuffers = 0;
    for (int i = 0; i < ninsn; i++)
    {
        const EwInsn& ins = prog.prog[i];
        int rt = tempOfSlot[ins.result];
        if (rt >= 0 && prog.bufferOfTemp[rt] < 0)
        {
            int b;
            if (!freeBufs.empty()) { b = freeBufs.back(); freeBufs.pop_back(); }
            else b = nbuffers++;
            prog.bufferOfTemp[rt] = b;
        }
        int as[3] = { ins.arg0, ins.arg1, ins.arg2 };
        for (int k = 0; k < 3; k++)
        {
            int t = tempOfSlot[as[k]];
            if (t >= 0 && lastUse[t] == i) freeBufs.push_back(prog.bufferOfTemp[t]);
        }
    }

    prog.ntemps = ntemps;
    prog.nbuffers = nbuffers;
}

}} // namespace cv::ew
