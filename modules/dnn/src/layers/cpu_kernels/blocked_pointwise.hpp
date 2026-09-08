// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_BLOCKED_POINTWISE_HPP
#define OPENCV_DNN_BLOCKED_POINTWISE_HPP

#include <opencv2/core/hal/intrin.hpp>

namespace cv { namespace dnn {

#if (CV_SIMD || CV_SIMD_SCALABLE)

#if defined(__GNUC__)
#define CV_DNN_SPAN_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define CV_DNN_SPAN_NOINLINE __declspec(noinline)
#else
#define CV_DNN_SPAN_NOINLINE
#endif

/** True when the (H, W, C0) region of a block-layout tensor is a single contiguous
 *  run, which is what allows a block to be walked flat rather than row by row.
 */
static inline bool blockIsContiguous(int C0, int W,
                                     size_t inStep2, size_t inStep3,
                                     size_t outStep2, size_t outStep3)
{
    return inStep3  == (size_t)C0 && inStep2  == (size_t)W * C0 &&
           outStep3 == (size_t)C0 && outStep2 == (size_t)W * C0;
}

/** True when the block is contiguous and one register covers a whole number of
 *  pixels, i.e. when blockedSpanApply() may be used.
 */
static inline bool blockCanSpan(int C0, int VEC_SZ, int W,
                                size_t inStep2, size_t inStep3,
                                size_t outStep2, size_t outStep3)
{
    return C0 < VEC_SZ && (VEC_SZ % C0) == 0 &&
           blockIsContiguous(C0, W, inStep2, inStep3, outStep2, outStep3);
}

/** Applies a pointwise operation over one contiguous H*W*C0 block of a
 *  block-layout tensor, where the coefficients depend only on the channel and so
 *  repeat with period C0.
 *
 *  Kernels that walk such a block channel-by-channel can only use min(C0, VEC_SZ)
 *  lanes of a register. Since the block is contiguous over (H, W, C0), replicating
 *  the coefficients across the register instead lets one iteration span VEC_SZ/C0
 *  pixels and keep every lane busy. Requires VEC_SZ % C0 == 0.
 *
 *  Callers fill ca[0..C0) before the call; entries [C0, VEC_SZ) are filled here, so
 *  both arrays need room for VEC_SZ floats. Operations using a single coefficient
 *  pass the same array as ca and cb.
 *
 *  Op supplies two overloads, one on v_float32 and one on float:
 *      operator()(x, a, b)
 *
 *  Deliberately not inlined: GCC 15.2 on RISC-V speculates the stores below into
 *  callers whose guard is false, which corrupts elements the caller does not own
 *  (see fastNormGroupBlockF32, where a neighbouring group can share the block).
 */
template<typename Op>
CV_DNN_SPAN_NOINLINE
static void blockedSpanApply(const float* in, float* out, int64_t total,
                             int C0, int VEC_SZ, float* ca, float* cb, const Op& op)
{
    for (int c = C0; c < VEC_SZ; ++c)
    {
        ca[c] = ca[c - C0];
        cb[c] = cb[c - C0];
    }
    v_float32 va = vx_load(ca);
    v_float32 vb = vx_load(cb);

    int64_t idx = 0;
    for (; idx <= total - VEC_SZ; idx += VEC_SZ)
        vx_store(out + idx, op(vx_load(in + idx), va, vb));
    for (; idx < total; ++idx)
    {
        int c = (int)(idx % C0);
        out[idx] = op(in[idx], ca[c], cb[c]);
    }
}

/** y = a*x + b -- BatchNorm, InstanceNorm and GroupNorm scale/shift. */
struct BlockedAffineOp
{
    v_float32 operator()(const v_float32& x, const v_float32& a, const v_float32& b) const
    { return v_fma(x, a, b); }
    float operator()(float x, float a, float b) const { return x * a + b; }
};

/** y = x >= 0 ? x : a*x -- per-channel PReLU. Ignores the second coefficient. */
struct BlockedPReLUOp
{
    v_float32 operator()(const v_float32& x, const v_float32& a, const v_float32&) const
    { return v_select(v_ge(x, vx_setzero_f32()), x, v_mul(x, a)); }
    float operator()(float x, float a, float) const { return x >= 0.f ? x : a * x; }
};

#endif // CV_SIMD || CV_SIMD_SCALABLE

}} // namespace cv::dnn

#endif // OPENCV_DNN_BLOCKED_POINTWISE_HPP
