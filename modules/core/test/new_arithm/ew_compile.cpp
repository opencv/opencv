// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 2 implementation: type inference + cast insertion (TExpr::emit) and the finalize pass
// (TExpr::compile = bind kernels + liveness). There is no separate graph - the program is built
// directly (see ew_op.hpp); emit() is the helper that turns "op over these operand slots" into one
// or more typed instructions, exactly the way the hand builders in ew_exec do it by hand.

#include "ew_compile.hpp"
#include <algorithm>
#include <cmath>

namespace cv { namespace ew {

// ---------------------------------------------------------------------------
// Type inference helpers (deliberately small so they can grow later).
// ---------------------------------------------------------------------------
static bool isFloatDepth(int d)
{
    return d == CV_16F || d == CV_16BF || d == CV_32F || d == CV_64F;
}

// Can `depth` hold `v` exactly? Floats: yes (close enough for our promotion). Integers: only if v
// is integral and in range. Used so a const operand (e.g. 2.5 in a*2.5) is NOT quantized into a
// narrow-integer direct kernel - such an op must fall back to the float working type instead.
static bool depthRepresents(double v, int depth)
{
    if (isFloatDepth(depth)) return true;
    if (v != std::floor(v)) return false;
    switch (depth)
    {
    case CV_8U:  return v >= 0            && v <= 255;
    case CV_8S:  return v >= -128         && v <= 127;
    case CV_16U: return v >= 0            && v <= 65535;
    case CV_16S: return v >= -32768       && v <= 32767;
    case CV_32U: return v >= 0            && v <= 4294967295.0;
    case CV_32S: return v >= -2147483648.0 && v <= 2147483647.0;
    case CV_64U: return v >= 0            && v <= 18446744073709551615.0;
    case CV_64S: return v >= -9223372036854775808.0 && v <= 9223372036854775807.0;
    default:     return false;
    }
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
// unused operand) passes EW_DEPTH_NONE and does not force promotion. Declared in ew_op.hpp.
int promoteArith(int a, int b)
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
static int workingDepth(TOp op)
{
    return opCategory(op) == CAT_BITWISE ? CV_32S : CV_32F;
}

// Natural result depth of an op given its operand depths (a flexible/const operand passes
// EW_DEPTH_NONE). `castDepth` is the explicitly requested depth for OP_CAST / typed ops.
static int inferDepth(TOp op, int d0, int d1, int d2, int nargs, int castDepth)
{
    int r;
    switch (opCategory(op))
    {
    case CAT_CAST:    r = castDepth; break;                            // OP_CAST / convert_scale
    case CAT_MATH:    r = (d0 == CV_64F) ? CV_64F : CV_32F; break;     // float domain
    case CAT_COMPARE: r = (castDepth != EW_DEPTH_NONE) ? castDepth : CV_8U; break;
    case CAT_BITWISE: r = promoteArith(d0, d1); break;
    case CAT_SELECT:  r = promoteArith(d1, d2); break;                 // mask=arg0, data=arg1,arg2
    case CAT_ARITH:
    default:          r = (nargs == 1) ? d0 : promoteArith(d0, d1); break;
    }
    if (r == EW_DEPTH_NONE) r = CV_32F;        // e.g. const (op) const
    return r;
}

// Kernel-existence probe (cast-aware: OP_CAST binds a core convert BinaryFunc, not getElemwiseFunc).
static bool kernelExists(TOp op, int d0, int d1, int d2, int rd)
{
    TExpr::Insn probe; probe.op = op;
    return resolveInsnKernel(probe, d0, d1, d2, rd);
}

// ---------------------------------------------------------------------------
// TExpr::emit(): append `op` over the given operand slots with type inference + cast insertion.
// Operand slots may be INPUT/TEMP/OUTPUT (typed) or a flexible CONST (depth == EW_DEPTH_NONE),
// which is materialized at the depth this op wants. Returns the slot holding the result.
// ---------------------------------------------------------------------------
int TExpr::emit(TOp op, const int* args, int nargs, int castDepth)
{
    auto isFlexConst = [&](int s) { return arginfo[s].kind == CONST && arginfo[s].depth == EW_DEPTH_NONE; };

    // operand "natural" depths (a flexible const does not force promotion)
    int cd[3] = { EW_DEPTH_NONE, EW_DEPTH_NONE, EW_DEPTH_NONE };
    for (int k = 0; k < nargs; k++)
        cd[k] = isFlexConst(args[k]) ? EW_DEPTH_NONE : arginfo[args[k]].depth;

    const int rd = inferDepth(op, cd[0], cd[1], cd[2], nargs, castDepth);

    // place a child operand into a slot at depth `d` (flexible const -> fresh typed CONST slot;
    // otherwise cast the existing slot if its depth differs).
    auto place = [&](int s, int d) -> int
    {
        if (isFlexConst(s))
            return addConst(d, arginfo[s].cval, arginfo[s].channels);
        if (arginfo[s].depth == d) return s;
        int t = addTemp(d);
        addInsn(OP_CAST, s, 0, 0, t);
        return t;
    };

    // 1) try a direct kernel with every operand at their COMMON type (the promotion of the non-const
    // operands). Casting both to one type keeps e.g. min(u64,u32) in u64 instead of falling to the
    // f32 working path and losing precision. A flexible const takes that common type too (or rd when
    // every operand is a flexible const, e.g. const (op) const).
    int Cop = EW_DEPTH_NONE;
    for (int k = 0; k < nargs; k++)
        if (!isFlexConst(args[k])) Cop = promoteArith(Cop, arginfo[args[k]].depth);
    const int tdc = (Cop == EW_DEPTH_NONE) ? rd : Cop;

    int td[3] = { EW_DEPTH_NONE, EW_DEPTH_NONE, EW_DEPTH_NONE };
    for (int k = 0; k < nargs; k++) td[k] = tdc;

    // refuse the direct kernel if a flexible const can't be represented exactly at tdc
    // (e.g. 2.5 lowered into a u8 kernel) - fall to the float working path instead.
    bool constsFit = true;
    for (int k = 0; k < nargs && constsFit; k++)
    {
        if (!isFlexConst(args[k])) continue;
        const Arg& a = arginfo[args[k]];
        for (int ch = 0, nch = std::max(1, a.channels); ch < nch; ch++)
            if (!depthRepresents(a.cval[ch], tdc)) { constsFit = false; break; }
    }

    if (constsFit && kernelExists(op, td[0], td[1], td[2], rd))
    {
        int op0 = nargs > 0 ? place(args[0], td[0]) : 0;
        int op1 = nargs > 1 ? place(args[1], td[1]) : 0;
        int op2 = nargs > 2 ? place(args[2], td[2]) : 0;
        int res = addTemp(rd);
        addInsn(op, op0, op1, op2, res);
        return res;
    }

    // 2) fall back to a working depth, casting operands; cast the result back to rd if needed
    const int W = workingDepth(op);
    const int wr = (opCategory(op) == CAT_COMPARE) ? rd : W;
    int w0 = nargs > 0 ? W : EW_DEPTH_NONE;
    int w1 = nargs > 1 ? W : EW_DEPTH_NONE;
    int w2 = nargs > 2 ? W : EW_DEPTH_NONE;
    CV_Assert(kernelExists(op, w0, w1, w2, wr));

    int op0 = nargs > 0 ? place(args[0], W) : 0;
    int op1 = nargs > 1 ? place(args[1], W) : 0;
    int op2 = nargs > 2 ? place(args[2], W) : 0;

    if (wr == rd)
    {
        int res = addTemp(rd);
        addInsn(op, op0, op1, op2, res);
        return res;
    }
    int wSlot = addTemp(wr);
    addInsn(op, op0, op1, op2, wSlot);
    int res = addTemp(rd);
    addInsn(OP_CAST, wSlot, 0, 0, res);
    return res;
}

// ---------------------------------------------------------------------------
// TExpr::output(): expose `rootSlot` as a result tensor. Prefer redirecting the single producer
// instruction to write the OUTPUT slot directly (no copy); else copy via an identity OP_CAST.
// A redirected TEMP slot is simply left unused (it gets no physical buffer in liveness).
// ---------------------------------------------------------------------------
int TExpr::output(int rootSlot)
{
    const int O = addOutput(arginfo[rootSlot].depth);

    if (arginfo[rootSlot].kind == TEMP)
    {
        int producer = -1;
        bool usedAsArg = false;
        const int ninsn = (int)prog.size();
        for (int i = 0; i < ninsn; i++)
        {
            const TExpr::Insn& ins = prog[i];
            if (ins.result == rootSlot) producer = i;
            if (ins.arg0 == rootSlot || ins.arg1 == rootSlot || ins.arg2 == rootSlot) usedAsArg = true;
        }
        if (producer >= 0 && !usedAsArg) { prog[producer].result = O; return O; }
    }
    addInsn(OP_CAST, rootSlot, 0, 0, O);                // same depth => identity copy
    return O;
}

// ---------------------------------------------------------------------------
// TExpr::compile(): finalize the already-typed program. Bind every instruction's kernel (skipping
// any whose pointer was pre-set, e.g. div's caller-known /0 policy) and pack the temps into a
// minimal set of reusable physical buffers (liveness). All per-arg bookkeeping here is transient.
// ---------------------------------------------------------------------------
void TExpr::compile()
{
    const int ninsn = (int)prog.size();

    // ---- bind kernels ----
    for (int i = 0; i < ninsn; i++)
    {
        TExpr::Insn& ins = prog[i];
        if (ins.fptr || ins.bfptr) continue;            // kernel already bound by the builder
        int d0 = ins.arg0 ? arginfo[ins.arg0].depth : EW_DEPTH_NONE;
        int d1 = ins.arg1 ? arginfo[ins.arg1].depth : EW_DEPTH_NONE;
        int d2 = ins.arg2 ? arginfo[ins.arg2].depth : EW_DEPTH_NONE;
        bool ok = resolveInsnKernel(ins, d0, d1, d2, arginfo[ins.result].depth);
        CV_Assert(ok);
    }

    // count materialized CONST slots (flexible literals, depth==NONE, are dead - emit() replaced
    // them with typed copies); sizes the const store and gates exec()'s fast path. Cheap scan, no
    // allocation - kept separate from liveness so the no-temp path can skip out below.
    const int nslots = (int)arginfo.size();
    nconsts = 0;
    for (int s = 1; s < nslots; s++)
        if (arginfo[s].kind == CONST && arginfo[s].depth != EW_DEPTH_NONE) nconsts++;

    // ---- liveness: pack temps into a minimal set of reusable physical buffers. A program with no
    //      temps (single-op add/sub/mul/...) needs none of this - early out with ZERO heap traffic,
    //      so building such a program (the dominant cv::add-style call) allocates nothing. The temp
    //      case uses stack-backed AutoBuffers (inline for typical small expressions). ----
    nbuffers = 0;
    bufEszPrefix.resize(1); bufEszPrefix[0] = 0;
    if (ntemps == 0) return;

    AutoBuffer<int, 32> tempOfSlot(nslots);
    for (int s = 0; s < nslots; s++) tempOfSlot[s] = -1;
    for (int s = 1; s < nslots; s++)
        if (arginfo[s].kind == TEMP) tempOfSlot[s] = arginfo[s].index;

    AutoBuffer<int, 16> lastUse(ntemps);
    for (int t = 0; t < ntemps; t++) lastUse[t] = -1;
    for (int i = 0; i < ninsn; i++)
    {
        const TExpr::Insn& ins = prog[i];
        int as[3] = { ins.arg0, ins.arg1, ins.arg2 };
        for (int k = 0; k < 3; k++)
            if (tempOfSlot[as[k]] >= 0) lastUse[tempOfSlot[as[k]]] = i;
    }

    bufferOfTemp.resize(ntemps);
    for (int t = 0; t < ntemps; t++) bufferOfTemp[t] = -1;
    AutoBuffer<int, 16> freeBufs(ntemps);
    int nfree = 0, nbuf = 0;
    for (int i = 0; i < ninsn; i++)
    {
        const TExpr::Insn& ins = prog[i];
        int rt = tempOfSlot[ins.result];
        if (rt >= 0 && bufferOfTemp[rt] < 0)
            bufferOfTemp[rt] = nfree > 0 ? freeBufs[--nfree] : nbuf++;
        int as[3] = { ins.arg0, ins.arg1, ins.arg2 };
        for (int k = 0; k < 3; k++)
        {
            int t = tempOfSlot[as[k]];
            if (t >= 0 && lastUse[t] == i) freeBufs[nfree++] = bufferOfTemp[t];
        }
    }

    nbuffers = nbuf;

    // Byte layout of the physical temp buffers, per output element: prefix sums of each buffer's
    // max element size. bufEszPrefix[b]*region = buffer b's byte offset in the scratch; [nbuffers] =
    // total temp bytes/element. Precomputed here (depths are build-time) so exec()/its fast path
    // never recompute it per call, and the fast path can size its scratch in O(1).
    bufEszPrefix.resize(nbuffers + 1);
    for (int b = 0; b <= nbuffers; b++) bufEszPrefix[b] = 0;
    for (int s = 1; s < nslots; s++)
        if (arginfo[s].kind == TEMP)
        {
            int b = bufferOfTemp[arginfo[s].index], e = (int)CV_ELEM_SIZE1(arginfo[s].depth);
            if (e > bufEszPrefix[b + 1]) bufEszPrefix[b + 1] = e;     // max elem size in buffer b
        }
    for (int b = 0; b < nbuffers; b++) bufEszPrefix[b + 1] += bufEszPrefix[b];   // -> prefix sums
}

}} // namespace cv::ew
