// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Dispatch layer for the element-wise kernels (arithm_kernels.simd.hpp). Two tiers of plain
// functions sit on top of the per-baseline get*Func_:
//   - get*Func(...)      : forward to the CPU-optimal kernel via CV_CPU_DISPATCH (the useful op-
//                          specific entry points; candidates for CV_EXPORTS later);
//   - getElemwiseFunc(...): the op-level router used by the compiler.

#include "precomp.hpp"
#include "arithm_expr.hpp"
#include "arithm_kernels.simd.hpp"
#include "arithm_kernels.simd_declarations.hpp"

namespace cv { namespace ew {

// ---- tier 2: pick the kernel optimized for the current CPU ---------------------------------------
TKernel getAddFunc(int T, int R)            { CV_CPU_DISPATCH(getAddFunc_,     (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getSubFunc(int T, int R)            { CV_CPU_DISPATCH(getSubFunc_,     (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getMulFunc(int T, int R)            { CV_CPU_DISPATCH(getMulFunc_,     (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getDivFunc(int T, int R, bool chk)  { CV_CPU_DISPATCH(getDivFunc_,     (T, R, chk),     CV_CPU_DISPATCH_MODES_ALL); }
TKernel getPowFunc(int T, int R)            { CV_CPU_DISPATCH(getPowFunc_,     (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getMinFunc(int T, int R)            { CV_CPU_DISPATCH(getMinFunc_,     (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getMaxFunc(int T, int R)            { CV_CPU_DISPATCH(getMaxFunc_,     (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getAbsdiffFunc(int T, int R)        { CV_CPU_DISPATCH(getAbsdiffFunc_, (T, R),          CV_CPU_DISPATCH_MODES_ALL); }
TKernel getCmpFunc(TOp op, int T)           { CV_CPU_DISPATCH(getCmpFunc_,     (op, T),         CV_CPU_DISPATCH_MODES_ALL); }
TKernel getBitwiseFunc(TOp op, int esz)     { CV_CPU_DISPATCH(getBitwiseFunc_, (op, esz),       CV_CPU_DISPATCH_MODES_ALL); }
TKernel getNotFunc(int esz)                 { CV_CPU_DISPATCH(getNotFunc_,     (esz),           CV_CPU_DISPATCH_MODES_ALL); }
TKernel getCopyMaskFunc(int depth)          { CV_CPU_DISPATCH(getCopyMaskFunc_,(depth),         CV_CPU_DISPATCH_MODES_ALL); }
static TKernel getCastFunc(int sd, int dd, bool scaled)
                                            { CV_CPU_DISPATCH(getCastFunc_,    (sd, dd, scaled),CV_CPU_DISPATCH_MODES_ALL); }

// ---- tier 3: op-level dispatcher used by the compiler --------------------------------------------
TKernel getElemwiseFunc(TOp op, int depth0, int depth1, int depth2, int rdepth)
{
    (void)depth2;

    if (op == OP_CAST)          return getCastFunc(depth0, rdepth, false);
    if (op == OP_CONVERT_SCALE) return getCastFunc(depth0, rdepth, true);

    if (op == OP_ADD || op == OP_SUB)
    {
        if (depth0 != depth1) return {};   // operands must be the same type
        return op == OP_ADD ? getAddFunc(depth0, rdepth) : getSubFunc(depth0, rdepth);
    }

    // OP_MUL / OP_DIV / OP_POW: operands same type T; compute in the float work type (rdepth).
    if (op == OP_MUL || op == OP_DIV || op == OP_POW)
    {
        if (depth0 != depth1) return {};
        if (op == OP_MUL) return getMulFunc(depth0, rdepth);
        if (op == OP_POW) return getPowFunc(depth0, rdepth);
        // integer inputs guard divide-by-zero (-> 0); floats do not (a/0 -> inf, matching cv::divide).
        const bool isflt = depth0==CV_16F || depth0==CV_16BF || depth0==CV_32F || depth0==CV_64F;
        return getDivFunc(depth0, rdepth, !isflt);
    }

    // OP_MIN / OP_MAX: T x T -> T.
    if (op == OP_MIN || op == OP_MAX)
    {
        if (depth0 != depth1 || rdepth != depth0) return {};
        return op == OP_MIN ? getMinFunc(depth0, rdepth) : getMaxFunc(depth0, rdepth);
    }

    // OP_ABSDIFF: result is absdiffResultDepth(T) (unsigned same width for signed).
    if (op == OP_ABSDIFF)
    {
        if (depth0 != depth1) return {};
        return getAbsdiffFunc(depth0, rdepth);
    }

    // OP_CMP_*: T x T -> u8 mask (0 / 1 / 255, value via TKernel::flags).
    if (opCategory(op) == CAT_COMPARE)
    {
        if (depth0 != depth1 || rdepth != CV_8U) return {};
        return getCmpFunc(op, depth0);
    }

    // OP_AND / OP_OR / OP_XOR: bit-pattern op, T x T -> T (same depth), dispatched by element size.
    if (op == OP_AND || op == OP_OR || op == OP_XOR)
    {
        if (depth0 != depth1 || rdepth != depth0) return {};
        return getBitwiseFunc(op, CV_ELEM_SIZE1(depth0));
    }

    // OP_NOT: ~x, one input, T -> T (same depth), dispatched by element size.
    if (op == OP_NOT)
    {
        if (rdepth != depth0) return {};
        return getNotFunc(CV_ELEM_SIZE1(depth0));
    }

    // OP_COPY_MASK: data depth == depth0 == rdepth; depth1 is the (1-byte) mask. By element size.
    if (op == OP_COPY_MASK)
        return getCopyMaskFunc(rdepth);

    return {};
}

}} // namespace cv::ew
