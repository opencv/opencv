// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Element-wise MATH kernels (sqrt/exp/log/sin/cos/tanh/erf/relu) and SELECT for the new arithmetic
// engine, SIMD-dispatched per CPU baseline - the unary/ternary sibling of arithm.simd.hpp.
//
// This file is compiled once per SIMD baseline (registered via ocv_add_dispatched_file). The per-op
// entry points get*Func_(...) live in cv::ew::CV_CPU_OPTIMIZATION_NAMESPACE and return the kernel
// optimized for that baseline; the regular get*Func dispatchers live in math.dispatch.cpp.
//
// Kernel shape (house style of arithm.simd.hpp):
//  - one 2D tile; per-row outer loop with stepy (bytes); dst contiguous in x; stepx in {0,1}.
//  - continuity collapse 2D->1D when every operand+dst is gap-free.
//  - halide right-edge backoff for the SIMD tail, SUPPRESSED when dst aliases an input (in-place
//    unary math would re-apply Op to already-written values).
//
// Math is T -> T over the four float depths: f32/f64 compute natively (v_exp & co exist for both);
// f16/bf16 ride the f32 hub (vx_load_pair_as widens one native vector into two f32 vectors, the
// saturating v_store_pair_as packs them back) - more accurate than a native f16 polynomial and
// works on every baseline. Integer inputs never reach these kernels: emitUnary computes integer
// math in the float domain and casts.

#include "opencv2/core/hal/intrin.hpp"
#include "convert.hpp"          // typed vx_load_pair_as / v_store_pair_as helpers (cv::)
#include "arithm_expr.hpp"      // the kernel contract: TOp / TKernel / KernelFunc
#include "mathfuncs.hpp"
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace cv {

// Everything outside cv::ew::CV_CPU_OPTIMIZATION_NAMESPACE must be skipped in the
// declarations-only re-includes (one per dispatched mode), or it gets redefined.
#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#if (CV_SIMD || CV_SIMD_SCALABLE)

// f32-pair -> f16/bf16 stores for the half-float hub (convert.hpp covers the other pairs)
static inline void v_store_pair_as(hfloat* p, const v_float32& a, const v_float32& b)
{
    v_pack_store(p, a);
    v_pack_store(p + VTraits<v_float32>::vlanes(), b);
}
static inline void v_store_pair_as(bfloat* p, const v_float32& a, const v_float32& b)
{
    v_pack_store(p, a);
    v_pack_store(p + VTraits<v_float32>::vlanes(), b);
}

#endif
#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace ew {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// ---- per-op kernel entry points for THIS baseline (the regular dispatchers in
//      math.dispatch.cpp reach them through CV_CPU_DISPATCH). ----
TKernel getMathFunc_(TOp op, int T);                 // unary math, T -> T, T in {f16, bf16, f32, f64}
TKernel getPowFunc_(int T, int R);                   // OP_POW, T x T -> T (R must equal T)

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// ===========================================================================
// Op functors: vec(Wvec) over the work vector (f32 or f64), scl(WT) for the scalar path/tail.
// New unary math ops slot in here.
// ===========================================================================
struct MSqrt {
    template<typename V> static V vec(const V& x) { return v_sqrt(x); }
    template<typename W> static W scl(W x) { return std::sqrt(x); }
};
template<typename V> static inline V fastLog(const V& x) { return v_log(x); }
template<typename V> static inline void fastLogPair(const V& x0, const V& x1, V& y0, V& y1,
                                                   const float*)
{
    y0 = v_log(x0);
    y1 = v_log(x1);
}
// FP32 log through a 256-bin table with a cubic correction. This version is defined for every
// input (zero, negative, subnormal, infinity, NaN), so callers need no fallback path and the
// output of a lane never depends on its neighbours or on where the lane sits in the array.
static inline v_float32 tableLog32fAnyInput(const v_float32& x, const float* table)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const v_int32 zero = vx_setzero_s32();
    const v_int32 sourceBits = v_reinterpret_as_s32(x);
    const v_int32 absBits = v_and(sourceBits, vx_setall_s32(0x7fffffff));
    // Subnormals are scaled by 2^23 into the normal range; the exponent is corrected below.
    const v_int32 subnormal = v_and(v_lt(absBits, vx_setall_s32(0x00800000)), v_gt(absBits, zero));
    const v_float32 scaled = v_select(v_reinterpret_as_f32(subnormal),
                                      v_mul(x, vx_setall_f32(8388608.f)), x);
    const v_int32 bits = v_reinterpret_as_s32(scaled);
    const v_int32 exponent = v_and(v_shr<23>(bits), vx_setall_s32(255));
    const v_int32 yi = v_sub(v_sub(exponent, vx_setall_s32(127)), v_and(subnormal, vx_setall_s32(23)));
    const v_int32 xi = v_or(v_and(bits, vx_setall_s32((1 << (23 - 8)) - 1)),
                            vx_setall_s32(127 << 23));
    const v_int32 index = v_and(v_shr<23 - 8 - 1>(bits), vx_setall_s32(510));
    v_float32 y, reciprocal;
    v_lut_deinterleave(table, index, y, reciprocal);
    y = v_fma(v_cvt_f32(yi), vx_setall_f32(0.6931471805599453f), y);
    const v_float32 delta = v_select(v_reinterpret_as_f32(v_eq(index, vx_setall_s32(510))),
                                     vx_setall_f32(-1.f / 512.f), vx_setall_f32(0.f));
    const v_float32 mantissa = v_reinterpret_as_f32(xi);
    const v_float32 reduced = v_fma(v_sub(mantissa, vx_setall_f32(1.f)), reciprocal, delta);
    v_float32 result = v_fma(reduced, vx_setall_f32(1.f / 3.f), vx_setall_f32(-0.5f));
    result = v_fma(result, reduced, vx_setall_f32(1.f));
    result = v_fma(result, reduced, y);

    const v_int32 isZero = v_eq(absBits, zero);
    const v_int32 isInfinity = v_eq(sourceBits, vx_setall_s32(0x7f800000));
    const v_int32 isInvalid = v_or(v_gt(absBits, vx_setall_s32(0x7f800000)),
                                   v_and(v_lt(sourceBits, zero), v_gt(absBits, zero)));
    result = v_select(v_reinterpret_as_f32(isZero),
                      vx_setall_f32(-std::numeric_limits<float>::infinity()), result);
    result = v_select(v_reinterpret_as_f32(isInfinity),
                      vx_setall_f32(std::numeric_limits<float>::infinity()), result);
    return v_select(v_reinterpret_as_f32(isInvalid),
                    vx_setall_f32(std::numeric_limits<float>::quiet_NaN()), result);
#else
    return v_log(x);
#endif
}
// Positive finite normal lanes only: the common case, without the special-value fix-ups.
static inline v_float32 tableLogNormal32f(const v_float32& x, const float* table)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const v_int32 bits = v_reinterpret_as_s32(x);
    const v_int32 exponent = v_and(v_shr<23>(bits), vx_setall_s32(255));
    const v_int32 yi = v_sub(exponent, vx_setall_s32(127));
    const v_int32 xi = v_or(v_and(bits, vx_setall_s32((1 << (23 - 8)) - 1)),
                            vx_setall_s32(127 << 23));
    const v_int32 index = v_and(v_shr<23 - 8 - 1>(bits), vx_setall_s32(510));
    v_float32 y, reciprocal;
    v_lut_deinterleave(table, index, y, reciprocal);
    y = v_fma(v_cvt_f32(yi), vx_setall_f32(0.6931471805599453f), y);
    const v_float32 delta = v_select(v_reinterpret_as_f32(v_eq(index, vx_setall_s32(510))),
                                     vx_setall_f32(-1.f / 512.f), vx_setall_f32(0.f));
    const v_float32 reduced = v_fma(v_sub(v_reinterpret_as_f32(xi), vx_setall_f32(1.f)),
                                    reciprocal, delta);
    v_float32 result = v_fma(reduced, vx_setall_f32(1.f / 3.f), vx_setall_f32(-0.5f));
    result = v_fma(result, reduced, vx_setall_f32(1.f));
    return v_fma(result, reduced, y);
#else
    return v_log(x);
#endif
}
// Lanes that take the fast path give bit-identical results on the any-input path, so the
// output of a lane does not depend on what its neighbours hold.
static inline v_float32 tableLog32f(const v_float32& x, const float* table)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const v_uint32 normal = v_lt(v_sub(v_reinterpret_as_u32(x), vx_setall_u32(0x00800000)),
                                 vx_setall_u32(0x7f000000));
    if (v_check_all(normal))
        return tableLogNormal32f(x, table);
#endif
    return tableLog32fAnyInput(x, table);
}
static inline void fastLogPair(const v_float32& x0, const v_float32& x1,
                               v_float32& y0, v_float32& y1, const float* table)
{
    if (table)
    {
        y0 = tableLog32f(x0, table);
        y1 = tableLog32f(x1, table);
    }
    else
    {
        y0 = v_log(x0);
        y1 = v_log(x1);
    }
}
struct MExp {
    template<typename V> static V vec(const V& x) { return v_exp(x); }
    template<typename W> static W scl(W x) { return std::exp(x); }
};
struct MLog {
    template<typename V> static V vec(const V& x) { return fastLog(x); }
    template<typename W> static W scl(W x) { return std::log(x); }
};
struct MSin {
    template<typename V> static V vec(const V& x) { return v_sin(x); }
    template<typename W> static W scl(W x) { return std::sin(x); }
};
struct MCos {
    template<typename V> static V vec(const V& x) { return v_cos(x); }
    template<typename W> static W scl(W x) { return std::cos(x); }
};
// tanh(x) = (e^2x - 1) / (e^2x + 1), on top of v_exp (no v_tanh intrinsic). The input is clamped
// first: tanh saturates to +/-1 well inside |x| <= 10 (f32) / 20 (f64), while an unclamped large x
// would push e^2x to inf and the ratio to inf/inf = NaN. (A NaN input may map to a saturated value
// on some ISAs instead of NaN - the polynomial v_exp has relaxed NaN semantics anyway.)
struct MTanh {
    static v_float32 vec(const v_float32& x)
    {
        const v_float32 one = vx_setall_f32(1.f), lim = vx_setall_f32(10.f);
        v_float32 cx = v_min(v_max(x, v_sub(vx_setzero_f32(), lim)), lim);
        v_float32 e = v_exp(v_add(cx, cx));
        return v_div(v_sub(e, one), v_add(e, one));
    }
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    static v_float64 vec(const v_float64& x)
    {
        const v_float64 one = vx_setall_f64(1.), lim = vx_setall_f64(20.);
        v_float64 cx = v_min(v_max(x, v_sub(vx_setzero_f64(), lim)), lim);
        v_float64 e = v_exp(v_add(cx, cx));
        return v_div(v_sub(e, one), v_add(e, one));
    }
#endif
    template<typename W> static W scl(W x) { return std::tanh(x); }
};
struct MErf {
    static v_float32 vec(const v_float32& x) { return v_erf(x); }
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    // no f64 SIMD erf primitive: apply std::erf per lane (keeps the one kernel shape; the
    // store/load round-trip is noise next to libm erf itself)
    static v_float64 vec(const v_float64& x)
    {
        double buf[VTraits<v_float64>::max_nlanes];
        v_store(buf, x);
        for (int i = 0; i < VTraits<v_float64>::vlanes(); i++) buf[i] = std::erf(buf[i]);
        return vx_load(buf);
    }
#endif
    template<typename W> static W scl(W x) { return std::erf(x); }
};
struct MRelu {
    template<typename V> static V vec(const V& x) { return v_max(x, v_setzero_<V>()); }
    template<typename W> static W scl(W x) { return x > W(0) ? x : W(0); }
};

// ===========================================================================
// The unary kernel: dst = Op(src), T -> T. Wvec picks the work vector: v_float32 / v_float64 for
// the native depths, v_float32 for the f16/bf16 hub (vx_load_pair_as does the widening).
// ===========================================================================
template<typename T, typename Wvec, class Op>
static int vecUnaryKernel(const void* src0_, size_t s0y, size_t s0x,
                          const void*, size_t, size_t, const void*, size_t, size_t,
                          void* dst_, size_t dsty, int width, int height,
                          const double*, int, void*)
{
    s0y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(s0x <= 1u);

    const T* src0 = (const T*)src0_;
    T* dst = (T*)dst_;
    using WT = typename VTraits<Wvec>::lane_type;

    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width) { width *= height; height = 1; }

    // vertical broadcast (a row expanded into a matrix, s0y == 0): every output row is identical -
    // compute the first one, memcpy the rest (transcendentals cost far more than a row copy)
    const int urows = (s0y == 0 && height > 1) ? 1 : height;

    for (int y = 0; y < urows; y++, src0 += s0y, dst += dsty)
    {
        if (s0x == 0)                     // broadcast-scalar source: one value covers the row
        {
            T v = saturate_cast<T>(Op::scl((WT)src0[0]));
            for (int x = 0; x < width; x++) dst[x] = v;
            continue;
        }
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        const int VECSZ = VTraits<Wvec>::vlanes();
        // in-place (dst == src) forbids the right-edge backoff: it would re-read already-written
        // values and apply Op twice. Those rows finish in the scalar tail instead.
        const bool use_tail_trick = width >= VECSZ*4 && src0_ != dst_;
        for (; x < width; x += VECSZ*2)
        {
            if (x + VECSZ*2 > width) { if (!use_tail_trick || x == 0) break; x = width - VECSZ*2; }
            Wvec a0, a1;
            vx_load_pair_as(src0 + x, a0, a1);
            a0 = Op::vec(a0); a1 = Op::vec(a1);
            v_store_pair_as(dst + x, a0, a1);
        }
#endif
        for (; x < width; x++)
            dst[x] = saturate_cast<T>(Op::scl((WT)src0[x]));
    }
    dst = (T*)dst_;
    for (int y = urows; y < height; y++)
        memcpy(dst + (size_t)y*dsty, dst, (size_t)width*sizeof(T));
    return 0;
}

// Scalar-only variant for (op, depth) pairs with no SIMD primitive (erf on f64; every op's f64
// when the baseline has no 64-bit float SIMD).
template<typename T, typename WT, class Op>
static int scalarUnaryKernel(const void* src0_, size_t s0y, size_t s0x,
                             const void*, size_t, size_t, const void*, size_t, size_t,
                             void* dst_, size_t dsty, int width, int height,
                             const double*, int, void*)
{
    s0y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(s0x <= 1u);
    const T* src0 = (const T*)src0_;
    T* dst = (T*)dst_;
    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width) { width *= height; height = 1; }
    const int urows = (s0y == 0 && height > 1) ? 1 : height;   // vertical broadcast: 1 row + copies
    for (int y = 0; y < urows; y++, src0 += s0y, dst += dsty)
    {
        if (s0x == 0)
        {
            T v = saturate_cast<T>(Op::scl((WT)src0[0]));
            for (int x = 0; x < width; x++) dst[x] = v;
            continue;
        }
        for (int x = 0; x < width; x++)
            dst[x] = saturate_cast<T>(Op::scl((WT)src0[x]));
    }
    dst = (T*)dst_;
    for (int y = urows; y < height; y++)
        memcpy(dst + (size_t)y*dsty, dst, (size_t)width*sizeof(T));
    return 0;
}

static int log32fKernel(const void* src0_, size_t s0y, size_t s0x,
                        const void*, size_t, size_t, const void*, size_t, size_t,
                        void* dst_, size_t dsty, int width, int height,
                        const double*, int, void* userdata)
{
    s0y /= sizeof(float);
    dsty /= sizeof(float);
    CV_Assert(s0x <= 1u);
    const float* src0 = (const float*)src0_;
    float* dst = (float*)dst_;
    [[maybe_unused]] const float* table = (const float*)userdata;

    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width)
    { width *= height; height = 1; }
    const int urows = (s0y == 0 && height > 1) ? 1 : height;

#if (CV_SIMD || CV_SIMD_SCALABLE)
    CV_DbgAssert(table);
    const int vectorSize = VTraits<v_float32>::vlanes();
    float inputTail[VTraits<v_float32>::max_nlanes], outputTail[VTraits<v_float32>::max_nlanes];
#endif
    for (int y = 0; y < urows; y++, src0 += s0y, dst += dsty)
    {
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (s0x == 0)
        {
            // Broadcast input: evaluate once, through the same lanes as the array path.
            vx_store(outputTail, tableLog32f(vx_setall_f32(src0[0]), table));
            for (int x = 0; x < width; x++) dst[x] = outputTail[0];
            continue;
        }
        int x = 0;
        for (; x <= width - vectorSize; x += vectorSize)
            vx_store(dst + x, tableLog32f(vx_load(src0 + x), table));
        if (x < width)
        {
            // The partial tail goes through the same SIMD path so a value's result does not
            // depend on its position; in-place stays safe because the input is copied first.
            const int rest = width - x;
            std::fill(inputTail, inputTail + vectorSize, 1.f);
            std::memcpy(inputTail, src0 + x, rest*sizeof(float));
            vx_store(outputTail, tableLog32f(vx_load(inputTail), table));
            std::memcpy(dst + x, outputTail, rest*sizeof(float));
        }
#else
        if (s0x == 0)
        {
            const float value = std::log(src0[0]);
            for (int x = 0; x < width; x++) dst[x] = value;
            continue;
        }
        for (int x = 0; x < width; x++) dst[x] = std::log(src0[x]);
#endif
    }
    dst = (float*)dst_;
    for (int y = urows; y < height; y++)
        memcpy(dst + (size_t)y*dsty, dst, (size_t)width*sizeof(float));
    return 0;
}

// ===========================================================================
// OP_POW: dst = pow(x, y), T x T -> T over the float depths. Exact std::pow semantics.
//
// The exponent is USUALLY a broadcast scalar (pow(x, 2), texpr literals ride as 0-dim consts with
// stepx == 0) - dispatched PER ROW to the important special cases: y==2 -> x*x, y==3 -> x*x*x,
// y==0.5 -> v_sqrt, y==1 -> copy, y==0 -> fill 1 (std::pow(anything, 0) == 1, NaN included).
// Everything else - and the per-element exponent - runs the general vectorized path
// exp(y * log(x)), which is only valid for x > 0: any lane with x <= 0 falls back to scalar
// std::pow for the whole vector pair (v_check_any per pair; negative/zero bases are rare, and the
// scalar path preserves every std::pow subtlety - signed results for integer y, NaN for
// fractional y, the x == 0 family). One knowing deviation: y==0.5 uses v_sqrt, so pow(-0., .5)
// returns -0. instead of std::pow's +0.
//
// The halide right-edge tail backoff is used in every SIMD loop, SUPPRESSED when dst aliases an
// input: pow is not idempotent, so an in-place backoff would re-read already-written values (the
// overlap region is otherwise just recomputed from the untouched source). Suppressed rows finish
// in the scalar tail.
#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline v_float32 vxSetallW(float v,  const v_float32&) { return vx_setall_f32(v); }
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
static inline v_float64 vxSetallW(double v, const v_float64&) { return vx_setall_f64(v); }
#endif
#endif

// Plain scalar pow for baselines without the needed SIMD float width (f64 without 64-bit SIMD).
template<typename T, typename WT>
static int scalarPowKernel(const void* src0_, size_t s0y, size_t s0x,
                           const void* src1_, size_t s1y, size_t s1x,
                           const void*, size_t, size_t,
                           void* dst_, size_t dsty, int width, int height,
                           const double*, int, void*)
{
    s0y /= sizeof(T); s1y /= sizeof(T); dsty /= sizeof(T);
    CV_Assert(s0x <= 1u && s1x <= 1u);
    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    T* dst = (T*)dst_;
    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width && s1y == s1x*(size_t)width)
    { width *= height; height = 1; }
    for (int y = 0; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
        for (int x = 0; x < width; x++)
            dst[x] = saturate_cast<T>(std::pow((WT)src0[x*s0x], (WT)src1[x*s1x]));
    return 0;
}

template<typename T, typename Wvec>
static int powKernel(const void* src0_, size_t s0y, size_t s0x,
                     const void* src1_, size_t s1y, size_t s1x,
                     const void*, size_t, size_t,
                     void* dst_, size_t dsty, int width, int height,
                     const double*, int, void* userdata)
{
    s0y /= sizeof(T);
    s1y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(s0x <= 1u && s1x <= 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    T* dst = (T*)dst_;
    const float* logTable = (const float*)userdata;
    using WT = typename VTraits<Wvec>::lane_type;

    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width && s1y == s1x*(size_t)width)
    { width *= height; height = 1; }

    [[maybe_unused]] const bool tail_trick = src0_ != dst_ && src1_ != dst_;

    // both operands vertically broadcast: every output row is identical - compute one, copy
    const int urows = (s0y == 0 && s1y == 0 && height > 1) ? 1 : height;

    for (int y = 0; y < urows; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
        if (s1x == 0)                                   // scalar exponent for this row
        {
            const WT p = (WT)src1[0];
#if (CV_SIMD || CV_SIMD_SCALABLE)
            const int VECSZ = VTraits<Wvec>::vlanes();
            if (s0x == 1)
            {
                if (p == WT(2) || p == WT(3))
                {
                    for (; x < width; x += VECSZ*2)
                    {
                        if (x + VECSZ*2 > width) { if (!tail_trick || x == 0) break; x = width - VECSZ*2; }
                        Wvec a0, a1;
                        vx_load_pair_as(src0 + x, a0, a1);
                        Wvec r0 = v_mul(a0, a0), r1 = v_mul(a1, a1);
                        if (p == WT(3)) { r0 = v_mul(r0, a0); r1 = v_mul(r1, a1); }
                        v_store_pair_as(dst + x, r0, r1);
                    }
                }
                else if (p == WT(0.5))
                {
                    for (; x < width; x += VECSZ*2)
                    {
                        if (x + VECSZ*2 > width) { if (!tail_trick || x == 0) break; x = width - VECSZ*2; }
                        Wvec a0, a1;
                        vx_load_pair_as(src0 + x, a0, a1);
                        a0 = v_sqrt(a0); a1 = v_sqrt(a1);
                        v_store_pair_as(dst + x, a0, a1);
                    }
                }
                else if (p == WT(1))
                {
                    if ((const void*)src0 != (const void*)dst)
                        for (; x < width; x++) dst[x] = src0[x];
                    x = width;
                }
                else if (p == WT(0))
                {
                    const T one = saturate_cast<T>(1);
                    for (; x < width; x++) dst[x] = one;
                }
                else if (p == WT(-0.5))
                {
                    const Wvec one = vxSetallW(WT(1), Wvec());
                    for (; x < width; x += VECSZ*2)
                    {
                        if (x + VECSZ*2 > width) { if (!tail_trick || x == 0) break; x = width - VECSZ*2; }
                        Wvec a0, a1;
                        vx_load_pair_as(src0 + x, a0, a1);
                        a0 = v_div(one, v_sqrt(a0)); a1 = v_div(one, v_sqrt(a1));
                        v_store_pair_as(dst + x, a0, a1);
                    }
                }
                else if (p == std::rint(p) && std::abs(p) <= WT(65536))
                {
                    // any other INTEGER exponent: LSB-first binary exponentiation - the same
                    // multiply chain (and order) as the classic iPow, fully vectorized. Also more
                    // accurate than exp(p*log x) (a few ulp vs ~2e-7 rel) and semantically exact
                    // on non-positive bases: the sign falls out of the multiplies, 0^negative
                    // divides to inf - no scalar patching needed.
                    const int ip = (int)p, ap = ip < 0 ? -ip : ip;   // ap >= 1 (0..3 handled above)
                    const Wvec one = vxSetallW(WT(1), Wvec());
                    for (; x < width; x += VECSZ*2)
                    {
                        if (x + VECSZ*2 > width) { if (!tail_trick || x == 0) break; x = width - VECSZ*2; }
                        Wvec b0, b1;
                        vx_load_pair_as(src0 + x, b0, b1);
                        Wvec a0 = one, a1 = one;
                        for (int q = ap; q > 1; q >>= 1)
                        {
                            if (q & 1) { a0 = v_mul(a0, b0); a1 = v_mul(a1, b1); }
                            b0 = v_mul(b0, b0); b1 = v_mul(b1, b1);
                        }
                        a0 = v_mul(a0, b0); a1 = v_mul(a1, b1);
                        if (ip < 0) { a0 = v_div(one, a0); a1 = v_div(one, a1); }
                        v_store_pair_as(dst + x, a0, a1);
                    }
                }
                else                                    // general scalar exponent: exp(p * log(x))
                {
                    const Wvec vp = vxSetallW(p, Wvec()), z = v_setzero_<Wvec>();
                    for (; x < width; x += VECSZ*2)
                    {
                        if (x + VECSZ*2 > width) { if (!tail_trick || x == 0) break; x = width - VECSZ*2; }
                        Wvec a0, a1;
                        vx_load_pair_as(src0 + x, a0, a1);
                        if (v_check_any(v_or(v_le(a0, z), v_le(a1, z))))
                        {                               // exact std::pow for x <= 0 lanes
                            for (int i = 0; i < VECSZ*2; i++)
                                dst[x + i] = saturate_cast<T>(std::pow((WT)src0[x + i], p));
                            continue;
                        }
                        fastLogPair(a0, a1, a0, a1, logTable);
                        a0 = v_mul(vp, a0);
                        a1 = v_mul(vp, a1);
                        a0 = v_exp(a0);
                        a1 = v_exp(a1);
                        v_store_pair_as(dst + x, a0, a1);
                    }
                }
            }
#endif
            for (; x < width; x++)
                dst[x] = saturate_cast<T>(std::pow((WT)src0[x*s0x], p));
            continue;
        }
        // per-element exponent
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (s0x == 1)
        {
            const int VECSZ = VTraits<Wvec>::vlanes();
            const Wvec z = v_setzero_<Wvec>();
            for (; x < width; x += VECSZ*2)
            {
                if (x + VECSZ*2 > width) { if (!tail_trick || x == 0) break; x = width - VECSZ*2; }
                Wvec a0, a1, b0, b1;
                vx_load_pair_as(src0 + x, a0, a1);
                vx_load_pair_as(src1 + x, b0, b1);
                if (v_check_any(v_or(v_le(a0, z), v_le(a1, z))))
                {
                    for (int i = 0; i < VECSZ*2; i++)
                        dst[x + i] = saturate_cast<T>(std::pow((WT)src0[x + i], (WT)src1[x + i]));
                    continue;
                }
                fastLogPair(a0, a1, a0, a1, logTable);
                a0 = v_mul(b0, a0);
                a1 = v_mul(b1, a1);
                a0 = v_exp(a0);
                a1 = v_exp(a1);
                v_store_pair_as(dst + x, a0, a1);
            }
        }
#endif
        for (; x < width; x++)
            dst[x] = saturate_cast<T>(std::pow((WT)src0[x*s0x], (WT)src1[x]));
    }
    dst = (T*)dst_;
    for (int y = urows; y < height; y++)
        memcpy(dst + (size_t)y*dsty, dst, (size_t)width*sizeof(T));
    return 0;
}

TKernel getPowFunc_(int T, int R)
{
    if (R != T)
        return {};
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_16F:  fptr = powKernel<hfloat, v_float32>; break;
    case CV_16BF: fptr = powKernel<bfloat, v_float32>; break;
    case CV_32F:  fptr = powKernel<float,  v_float32>; break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    case CV_64F:  fptr = powKernel<double, v_float64>; break;
#else
    case CV_64F:  fptr = scalarPowKernel<double, double>; break;
#endif
    default: ;
    }
    void* userdata = nullptr;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if (T == CV_32F)
        userdata = (void*)cv::details::getLogTab32f();
#endif
    return {fptr, userdata, 0};
}

// ===========================================================================
// getters for THIS baseline
// ===========================================================================
template<class Op>
static KernelFunc mathByDepth(int T)
{
    switch (T)
    {
    case CV_16F:  return vecUnaryKernel<hfloat, v_float32, Op>;
    case CV_16BF: return vecUnaryKernel<bfloat, v_float32, Op>;
    case CV_32F:
        if constexpr (std::is_same<Op, MLog>::value)
            return log32fKernel;
        else
            return vecUnaryKernel<float, v_float32, Op>;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    case CV_64F:  return vecUnaryKernel<double, v_float64, Op>;
#else
    case CV_64F:  return scalarUnaryKernel<double, double, Op>;
#endif
    default:      return nullptr;
    }
}

TKernel getMathFunc_(TOp op, int T)
{
    KernelFunc f = nullptr;
    switch (op)
    {
    case OP_SQRT: f = mathByDepth<MSqrt>(T); break;
    case OP_EXP:  f = mathByDepth<MExp >(T); break;
    case OP_LOG:  f = mathByDepth<MLog >(T); break;
    case OP_SIN:  f = mathByDepth<MSin >(T); break;
    case OP_COS:  f = mathByDepth<MCos >(T); break;
    case OP_TANH: f = mathByDepth<MTanh>(T); break;
    case OP_RELU: f = mathByDepth<MRelu>(T); break;
    case OP_ERF:  f = mathByDepth<MErf >(T); break;
    default: ;
    }
    void* userdata = nullptr;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if (op == OP_LOG && T == CV_32F)
        userdata = (void*)cv::details::getLogTab32f();
#endif
    return {f, userdata, 0};
}


#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::ew
