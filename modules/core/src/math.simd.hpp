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
#include <cmath>

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
TKernel getSelectFunc_(int mdepth, int T);           // select(mask, x, y): 1-byte mask, x/y/dst of T

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// ===========================================================================
// Op functors: vec(Wvec) over the work vector (f32 or f64), scl(WT) for the scalar path/tail.
// New unary math ops slot in here.
// ===========================================================================
struct MSqrt {
    template<typename V> static V vec(const V& x) { return v_sqrt(x); }
    template<typename W> static W scl(W x) { return std::sqrt(x); }
};
struct MExp {
    template<typename V> static V vec(const V& x) { return v_exp(x); }
    template<typename W> static W scl(W x) { return std::exp(x); }
};
struct MLog {
    template<typename V> static V vec(const V& x) { return v_log(x); }
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

    for (int y = 0; y < height; y++, src0 += s0y, dst += dsty)
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
    for (int y = 0; y < height; y++, src0 += s0y, dst += dsty)
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
    return 0;
}

// ===========================================================================
// select(mask, x, y): dst = mask != 0 ? x : y. The mask is one byte per element (u8/s8/bool,
// never cast); x/y/dst share the depth T - the kernel is depth-agnostic, templated by the element
// SIZE (dispatched over the unsigned int of that width, like the bitwise family), so no float
// lanes ever meet the mask compare (immune to DAZ/FTZ denormal flushing). Branch operands may
// broadcast (stepx == 0).
// ===========================================================================
#if (CV_SIMD || CV_SIMD_SCALABLE)
// load VECSZ(T) mask bytes expanded to T-width all-ones/all-zeros lanes (u8 direct, u16/u32 via
// expand - the compare-kernel trick); the tag argument selects the width.
static inline v_uint8  loadSelectMask(const uchar* m, const v_uint8&)
{ return v_ne(vx_load(m), vx_setzero_u8()); }
static inline v_uint16 loadSelectMask(const uchar* m, const v_uint16&)
{ return v_ne(vx_load_expand(m), vx_setzero_u16()); }
static inline v_uint32 loadSelectMask(const uchar* m, const v_uint32&)
{ return v_ne(vx_load_expand_q(m), vx_setzero_u32()); }

static inline void setallSelect(const uint8_t*  p, v_uint8&  a) { a = vx_setall_u8(*p); }
static inline void setallSelect(const uint16_t* p, v_uint16& a) { a = vx_setall_u16(*p); }
static inline void setallSelect(const uint32_t* p, v_uint32& a) { a = vx_setall_u32(*p); }
#endif

template<typename T, typename Tvec>
static int selectKernel(const void* mask_, size_t smy, size_t smx,
                        const void* src1_, size_t s1y, size_t s1x,
                        const void* src2_, size_t s2y, size_t s2x,
                        void* dst_, size_t dsty, int width, int height,
                        const double*, int, void*)
{
    s1y /= sizeof(T);                    // the 1-byte mask's smy stays in bytes == elements
    s2y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(smx <= 1u && s1x <= 1u && s2x <= 1u);

    const uchar* mask = (const uchar*)mask_;
    const T* src1 = (const T*)src1_;
    const T* src2 = (const T*)src2_;
    T* dst = (T*)dst_;

    if (height > 1 && dsty == (size_t)width && smy == smx*(size_t)width &&
        s1y == s1x*(size_t)width && s2y == s2x*(size_t)width)
    { width *= height; height = 1; }

    for (int y = 0; y < height; y++, mask += smy, src1 += s1y, src2 += s2y, dst += dsty)
    {
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if constexpr (sizeof(T) <= 4)
        {
            if (smx == 1)                                    // per-element mask - the common case
            {
                const int VECSZ = VTraits<Tvec>::vlanes();
                const bool use_tail_trick = width >= VECSZ*4 && src1_ != dst_ && src2_ != dst_;
                Tvec a, b;
                if (s1x == 0) setallSelect(src1, a);
                if (s2x == 0) setallSelect(src2, b);
                for (; x < width; x += VECSZ)
                {
                    if (x + VECSZ > width) { if (!use_tail_trick || x == 0) break; x = width - VECSZ; }
                    Tvec m = loadSelectMask(mask + x, Tvec());
                    if (s1x) a = vx_load(src1 + x);
                    if (s2x) b = vx_load(src2 + x);
                    v_store(dst + x, v_select(m, a, b));
                }
            }
        }
#endif
        for (; x < width; x++)
            dst[x] = mask[x*smx] != 0 ? src1[x*s1x] : src2[x*s2x];
    }
    return 0;
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
    case CV_32F:  return vecUnaryKernel<float,  v_float32, Op>;
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
    return {f, nullptr, 0};
}

TKernel getSelectFunc_(int mdepth, int T)
{
    if (CV_ELEM_SIZE1(mdepth) != 1)      // the mask must be a 1-byte type (u8/s8/bool)
        return {};
    KernelFunc f = nullptr;
    switch (CV_ELEM_SIZE1(T))
    {
    case 1: f = selectKernel<uint8_t,  v_uint8 >; break;
    case 2: f = selectKernel<uint16_t, v_uint16>; break;
    case 4: f = selectKernel<uint32_t, v_uint32>; break;
    case 8: f = selectKernel<uint64_t, v_uint32>; break;   // SIMD path compiled out -> scalar select
    default: ;
    }
    return {f, nullptr, 0};
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::ew
