// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Element-wise kernels for the new arithmetic engine, SIMD-dispatched per CPU baseline.
//
// This file is compiled once per SIMD baseline (registered via ocv_add_dispatched_file). The per-op
// entry points get*Func_(...) live in cv::ew::CV_CPU_OPTIMIZATION_NAMESPACE and return the kernel
// optimized for that baseline; the regular get*Func / getElemwiseFunc dispatchers (which forward here
// through CV_CPU_DISPATCH) live in arithm.dispatch.cpp.
//
// Kernel shape (house style of arithm.simd.hpp + convert.simd.hpp):
//  - one 2D tile; per-row outer loop with stepy; dst contiguous in x.
//  - stepx is restricted to {0,1}: 1 = contiguous (SIMD), 0 = broadcast-scalar along x.
//  - continuity collapse 2D->1D when every operand+dst is gap-free.
//  - halide right-edge backoff for the SIMD tail; scalar tail / scalar path for width<VECSZ.
//  - IN-PLACE SAFE: the right-edge backoff is suppressed (scalar tail) when dst aliases an input.
//
// One template (vecBinaryKernel) covers the whole matrix: the work-vector type selects the native
// same-type saturating path vs the widening f32-hub path. Adding a binary op = add one Op functor.

#include "opencv2/core/hal/intrin.hpp"
#include "convert.hpp"          // typed load/store-as helpers (cv::), and getConvert*Func
#include "arithm_expr.hpp"      // the kernel contract: TOp / TKernel / KernelFunc
#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <type_traits>

namespace cv {

// Everything outside cv::ew::CV_CPU_OPTIMIZATION_NAMESPACE must be skipped in the
// declarations-only re-includes (one per dispatched mode), or it gets redefined.
#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// BinaryFunc and getConvertFunc / getConvertScaleFunc come from core (precomp.hpp / convert.hpp).

#if CV_SIMD128_FP16
#undef CV_SIMD_16F
#define CV_SIMD_16F 1
#elif !defined(CV_SIMD_16F)
#define CV_SIMD_16F 0       // keep `#if CV_SIMD_16F` -Wundef-clean on builds without FP16 SIMD
#endif

#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline void vx_setall_as(const uchar* p, v_uint8& a)
{ a = vx_setall_u8(*p); }
static inline void vx_setall_as(const schar* p, v_int8& a)
{ a = vx_setall_s8(*p); }
static inline void vx_setall_as(const uchar* p, v_int16& a)
{ a = vx_setall_s16(*p); }
static inline void vx_setall_as(const schar* p, v_int16& a)
{ a = vx_setall_s16(*p); }

static inline void vx_setall_as(const ushort* p, v_uint16& a)
{ a = vx_setall_u16(*p); }
static inline void vx_setall_as(const short* p, v_int16& a)
{ a = vx_setall_s16(*p); }
static inline void vx_setall_as(const ushort* p, v_int32& a)
{ a = vx_setall_s32(*p); }
static inline void vx_setall_as(const short* p, v_int32& a)
{ a = vx_setall_s32(*p); }

static inline void vx_setall_as(const unsigned* p, v_uint32& a)
{ a = vx_setall_u32(*p); }
static inline void vx_setall_as(const int* p, v_int32& a)
{ a = vx_setall_s32(*p); }

static inline void vx_setall_as(const uchar* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }
static inline void vx_setall_as(const schar* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }
static inline void vx_setall_as(const ushort* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }
static inline void vx_setall_as(const short* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }
static inline void vx_setall_as(const float* p, v_float32& a)
{ a = vx_setall_f32(*p); }

// ---------------------------------------------------------------------------
// Saturating integer helpers the universal intrinsics lack: v_add_sat/v_sub_sat for 32-bit lanes
// (v_add/v_sub saturate 8/16-bit but WRAP 32-bit), and v_mul_sat - the FULL-precision product
// clamped to the lane type, which is exactly cv::multiply's integer semantics at scale == 1.
// Local for now - the plan is to promote these into core/hal/intrin_*.hpp as proper universal
// intrinsics for all integer types. On NEON: single-instruction add/sub (vqadd/vqsub) and the
// widening-multiply + saturating-narrow pattern (vmull + vqmovn; the vget/vcombine forms compile
// to smull2/sqxtn2 on AArch64). Elsewhere: the Hacker's Delight (2-13) bit tricks for add/sub and
// the portable v_mul_expand + saturating v_pack composition for the 8/16-bit multiplies (RVV/LSX
// also have native ones - later, in the intrinsics). 32-bit lanes have no universal widening
// multiply (no v_mul_expand for s32), so 32-bit v_mul_sat is NEON-only (EW_HAVE_MULSAT32) and
// getMulFunc_ keeps the f64 work-vector kernels on the other backends.
#if defined(__ARM_NEON)
static inline v_int32  v_add_sat(const v_int32& a,  const v_int32& b)  { return v_int32(vqaddq_s32(a.val, b.val)); }
static inline v_int32  v_sub_sat(const v_int32& a,  const v_int32& b)  { return v_int32(vqsubq_s32(a.val, b.val)); }
static inline v_uint32 v_add_sat(const v_uint32& a, const v_uint32& b) { return v_uint32(vqaddq_u32(a.val, b.val)); }
static inline v_uint32 v_sub_sat(const v_uint32& a, const v_uint32& b) { return v_uint32(vqsubq_u32(a.val, b.val)); }
#define EW_HAVE_MULSAT32 1
static inline v_uint8 v_mul_sat(const v_uint8& a, const v_uint8& b)
{
    uint16x8_t p0 = vmull_u8(vget_low_u8(a.val), vget_low_u8(b.val));
    uint16x8_t p1 = vmull_u8(vget_high_u8(a.val), vget_high_u8(b.val));
    return v_uint8(vcombine_u8(vqmovn_u16(p0), vqmovn_u16(p1)));
}
static inline v_int8 v_mul_sat(const v_int8& a, const v_int8& b)
{
    int16x8_t p0 = vmull_s8(vget_low_s8(a.val), vget_low_s8(b.val));
    int16x8_t p1 = vmull_s8(vget_high_s8(a.val), vget_high_s8(b.val));
    return v_int8(vcombine_s8(vqmovn_s16(p0), vqmovn_s16(p1)));
}
static inline v_uint16 v_mul_sat(const v_uint16& a, const v_uint16& b)
{
    uint32x4_t p0 = vmull_u16(vget_low_u16(a.val), vget_low_u16(b.val));
    uint32x4_t p1 = vmull_u16(vget_high_u16(a.val), vget_high_u16(b.val));
    return v_uint16(vcombine_u16(vqmovn_u32(p0), vqmovn_u32(p1)));
}
static inline v_int16 v_mul_sat(const v_int16& a, const v_int16& b)
{
    int32x4_t p0 = vmull_s16(vget_low_s16(a.val), vget_low_s16(b.val));
    int32x4_t p1 = vmull_s16(vget_high_s16(a.val), vget_high_s16(b.val));
    return v_int16(vcombine_s16(vqmovn_s32(p0), vqmovn_s32(p1)));
}
static inline v_uint32 v_mul_sat(const v_uint32& a, const v_uint32& b)
{
    uint64x2_t p0 = vmull_u32(vget_low_u32(a.val), vget_low_u32(b.val));
    uint64x2_t p1 = vmull_u32(vget_high_u32(a.val), vget_high_u32(b.val));
    return v_uint32(vcombine_u32(vqmovn_u64(p0), vqmovn_u64(p1)));
}
static inline v_int32 v_mul_sat(const v_int32& a, const v_int32& b)
{
    int64x2_t p0 = vmull_s32(vget_low_s32(a.val), vget_low_s32(b.val));
    int64x2_t p1 = vmull_s32(vget_high_s32(a.val), vget_high_s32(b.val));
    return v_int32(vcombine_s32(vqmovn_s64(p0), vqmovn_s64(p1)));
}
#else
static inline v_int32 v_add_sat(const v_int32& a, const v_int32& b)
{
    v_int32 res = v_add(a, b);
    v_int32 ov  = v_and(v_xor(a, res), v_xor(b, res));               // sign bit set iff overflow
    v_int32 sat = v_xor(v_shr<31>(a), vx_setall_s32(INT_MAX));       // a >= 0 ? INT_MAX : INT_MIN
    return v_select(v_lt(ov, vx_setzero_s32()), sat, res);
}
static inline v_int32 v_sub_sat(const v_int32& a, const v_int32& b)
{
    v_int32 res = v_sub(a, b);
    v_int32 ov  = v_and(v_xor(a, b), v_xor(a, res));
    v_int32 sat = v_xor(v_shr<31>(a), vx_setall_s32(INT_MAX));
    return v_select(v_lt(ov, vx_setzero_s32()), sat, res);
}
static inline v_uint32 v_add_sat(const v_uint32& a, const v_uint32& b)
{
    v_uint32 res = v_add(a, b);
    return v_or(res, v_lt(res, a));                                  // wrapped => all-ones
}
static inline v_uint32 v_sub_sat(const v_uint32& a, const v_uint32& b)
{
    return v_and(v_sub(a, b), v_ge(a, b));                           // borrow => zero
}
static inline v_uint8 v_mul_sat(const v_uint8& a, const v_uint8& b)
{ v_uint16 p0, p1; v_mul_expand(a, b, p0, p1); return v_pack(p0, p1); }
static inline v_int8 v_mul_sat(const v_int8& a, const v_int8& b)
{ v_int16 p0, p1; v_mul_expand(a, b, p0, p1); return v_pack(p0, p1); }
static inline v_uint16 v_mul_sat(const v_uint16& a, const v_uint16& b)
{ v_uint32 p0, p1; v_mul_expand(a, b, p0, p1); return v_pack(p0, p1); }
static inline v_int16 v_mul_sat(const v_int16& a, const v_int16& b)
{ v_int32 p0, p1; v_mul_expand(a, b, p0, p1); return v_pack(p0, p1); }
#endif
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F   // scalable (RVV) has v_float64 without CV_SIMD_64F
static inline void vx_setall_as(const double*   p, v_float64& a) { a = vx_setall_f64(*p); }
static inline void vx_setall_as(const int*      p, v_float64& a) { a = vx_setall_f64((double)*p); }
static inline void vx_setall_as(const unsigned* p, v_float64& a) { a = vx_setall_f64((double)*p); }
static inline void vx_setall_as(const int64_t*  p, v_float64& a) { a = vx_setall_f64((double)*p); }
static inline void vx_setall_as(const uint64_t* p, v_float64& a) { a = vx_setall_f64((double)*p); }
#endif
static inline void vx_setall_as(const bfloat* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }
static inline void vx_setall_as(const hfloat* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }

static inline void vx_load_pair_as(const uchar* p, v_uint8& a, v_uint8& b)
{
    a = vx_load(p);
    b = vx_load(p + VTraits<v_uint8>::vlanes());
}

static inline void vx_load_pair_as(const schar* p, v_int8& a, v_int8& b)
{
    a = vx_load(p);
    b = vx_load(p + VTraits<v_int8>::vlanes());
}

static inline void vx_load_pair_as(const uchar* p, v_uint32& a, v_uint32& b)
{
    v_uint16 w = vx_load_expand(p);
    v_expand(w, a, b);
}

static inline void v_store_pair_as(uchar* p, const v_uint8& a, const v_uint8& b)
{
    v_store(p, a);
    v_store(p + VTraits<v_uint8>::vlanes(), b);
}

static inline void v_store_pair_as(schar* p, const v_int8& a, const v_int8& b)
{
    v_store(p, a);
    v_store(p + VTraits<v_int8>::vlanes(), b);
}


static inline void v_store_pair_as(int* p, const v_int16& a, const v_int16& b)
{
    const int nlanes32 = VTraits<v_int32>::vlanes();
    v_int32 v0, v1, v2, v3;
    v_expand(a, v0, v1);
    v_expand(b, v2, v3);
    v_store(p, v0);
    v_store(p + nlanes32, v1);
    v_store(p + nlanes32*2, v2);
    v_store(p + nlanes32*3, v3);
}

static inline void v_store_pair_as(float* p, const v_int16& a, const v_int16& b)
{
    const int nlanes32 = VTraits<v_float32>::vlanes();
    v_int32 v0, v1, v2, v3;
    v_expand(a, v0, v1);
    v_expand(b, v2, v3);
    v_store(p, v_cvt_f32(v0));
    v_store(p + nlanes32, v_cvt_f32(v1));
    v_store(p + nlanes32*2, v_cvt_f32(v2));
    v_store(p + nlanes32*3, v_cvt_f32(v3));
}

static inline void v_store_pair_as(float* p, const v_int32& a, const v_int32& b)
{
    v_store(p, v_cvt_f32(a));
    v_store(p + VTraits<v_float32>::vlanes(), v_cvt_f32(b));
}

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

// Reinterpret a comparison result (whose lane type matches the operands) as the UNSIGNED integer
// vector of the same width: u8->u8, s8->u8, u16/s16->u16, u32/s32/f32->u32. The destination type
// drives the overload, so the compare kernel needs no per-width if constexpr.
static inline void v_reinterpret_as(const v_uint8&  s, v_uint8&  d) { d = s; }
static inline void v_reinterpret_as(const v_int8&   s, v_uint8&  d) { d = v_reinterpret_as_u8(s); }
static inline void v_reinterpret_as(const v_uint16& s, v_uint16& d) { d = s; }
static inline void v_reinterpret_as(const v_int16&  s, v_uint16& d) { d = v_reinterpret_as_u16(s); }
static inline void v_reinterpret_as(const v_uint32& s, v_uint32& d) { d = s; }
static inline void v_reinterpret_as(const v_int32&  s, v_uint32& d) { d = v_reinterpret_as_u32(s); }
static inline void v_reinterpret_as(const v_float32& s, v_uint32& d) { d = v_reinterpret_as_u32(s); }
#if CV_SIMD_16F
static inline void v_reinterpret_as(const v_float16& s, v_uint16& d) { d = v_reinterpret_as_u16(s); }
#endif
#if CV_SIMD_64F
static inline void v_reinterpret_as(const v_float64& s, v_uint64& d) { d = v_reinterpret_as_u64(s); }
#endif

// Broadcast the compare mask value (1 or 255) across the unsigned work vector.
static inline void v_setall_mask(v_uint8&  v, uchar t) { v = vx_setall_u8(t); }
static inline void v_setall_mask(v_uint16& v, uchar t) { v = vx_setall_u16(t); }
static inline void v_setall_mask(v_uint32& v, uchar t) { v = vx_setall_u32(t); }

// NOTE: v_store_pair_as(uchar*, v_uint16/v_uint32, ...) - the narrowing pack-to-u8 stores we need -
// already come from core's convert.hpp (included above), so they are not redefined here.

#if CV_SIMD_16F
static inline void vx_setall_as(const hfloat* p, v_float16& a)
{ a = vx_setall_f16(*p); }
static inline void vx_setall_as(const float* p, v_float16& a)
{ a = vx_setall_f16(hfloat(*p)); }
static inline void vx_setall_as(const uchar* p, v_float16& a)
{ a = vx_setall_f16(hfloat(float(*p))); }
static inline void vx_setall_as(const schar* p, v_float16& a)
{ a = vx_setall_f16(hfloat(float(*p))); }
static inline void vx_load_pair_as(const uchar* p, v_float16& a, v_float16& b)
{
    v_uint8 v = vx_load(p);
    v_uint16 v0, v1;
    v_expand(v, v0, v1);
    a = v_cvt_f16(v_reinterpret_as_s16(v0));
    b = v_cvt_f16(v_reinterpret_as_s16(v1));
}
static inline void vx_load_pair_as(const schar* p, v_float16& a, v_float16& b)
{
    v_int8 v = vx_load(p);
    v_int16 v0, v1;
    v_expand(v, v0, v1);
    a = v_cvt_f16(v0);
    b = v_cvt_f16(v1);
}
static inline void vx_load_pair_as(const hfloat* p, v_float16& a, v_float16& b)
{
    a = vx_load(p);
    b = vx_load(p + VTraits<v_float16>::vlanes());
}
static inline void v_store_pair_as(uchar* p, const v_float16& a, const v_float16& b)
{
    v_int16 v0 = v_round(a), v1 = v_round(b);
    v_store(p, v_pack_u(v0, v1));
}
static inline void v_store_pair_as(schar* p, const v_float16& a, const v_float16& b)
{
    v_int16 v0 = v_round(a), v1 = v_round(b);
    v_store(p, v_pack(v0, v1));
}
static inline void v_store_pair_as(hfloat* p, const v_float16& a, const v_float16& b)
{
    v_store(p, a);
    v_store(p + VTraits<v_float16>::vlanes(), b);
}
#endif

#endif

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace ew {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// ---- per-op kernel entry points for THIS baseline (the regular dispatchers in
//      arithm.dispatch.cpp reach them through CV_CPU_DISPATCH). ----
TKernel getAddFunc_(int T, int R);
TKernel getSubFunc_(int T, int R);
TKernel getMulFunc_(int T, int R);
TKernel getDivFunc_(int T, int R, bool checked);
TKernel getMinFunc_(int T, int R);
TKernel getMaxFunc_(int T, int R);
TKernel getAbsdiffFunc_(int T, int R);
TKernel getHypotFunc_(int T, int R);         // hypot = sqrt(x^2+y^2), float depths, T x T -> T
TKernel getAtan2Func_(int T, int R);         // atan2(y, x), radians (-pi, pi], float depths
TKernel getCmpFunc_(TOp op, int T);
TKernel getBitwiseFunc_(TOp op, int esz);                    // OP_AND / OP_OR / OP_XOR, by element size
TKernel getNotFunc_(int esz);                                // OP_NOT, by element size
TKernel getAddWeightedFunc_(int T, int R);                   // OP_ADDW, a*alpha+b*beta+gamma (T x T -> R)
TKernel getSelectFunc_(int mdepth, int T);   // OP_SELECT: 1-byte mask, a/b/dst of T (by esz)
TKernel getClampFunc_(int T);                // OP_CLAMP: min(max(x, lo), hi), all operands of T
TKernel getCastFunc_(int sdepth, int ddepth, bool scaled);   // OP_CAST / OP_CONVERT_SCALE

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// ===========================================================================
// Op functors (vector + scalar). New binary ops slot in here.
// ===========================================================================
struct EwAdd {
    // useScalar: does the op consume the scale scalar (params[0]) in vec()/preproc()? When false,
    // vecBinaryKernel's fast 2-arg branch is taken unconditionally (the check folds at compile time);
    // when true, it is taken only if the runtime scale is exactly 1.
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_add(a, b); }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // 32-bit lanes: v_add wraps - use the saturating version to match the scalar int64+clamp tail
    static v_int32  vec(const v_int32& a,  const v_int32& b)  { return v_add_sat(a, b); }
    static v_uint32 vec(const v_uint32& a, const v_uint32& b) { return v_add_sat(a, b); }
#endif
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    // Accumulate in the promoted type, NOT in W: for the native saturating path W is the narrow
    // lane type (schar/short/...), and (W)(a+b) would wrap in 8/16 bits before saturate_cast<Tr>
    // could clamp. Letting a+b promote (narrow -> int) keeps saturation for 8/16-bit outputs and
    // the natural wrap for 32/64-bit (both matching cv::add). The SIMD path already saturates.
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a + b); }
};

struct EwSub {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_sub(a, b); }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // 32-bit lanes: see EwAdd::vec
    static v_int32  vec(const v_int32& a,  const v_int32& b)  { return v_sub_sat(a, b); }
    static v_uint32 vec(const v_uint32& a, const v_uint32& b) { return v_sub_sat(a, b); }
#endif
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a - b); }   // see EwAdd::scl
    // 64-bit unsigned has no wider WT to hold a-b, so the generic path would wrap on underflow.
    // Saturate at 0 to match cv::subtract (8/16/32-bit already saturate via SIMD floor / wide WT).
    static uint64_t scl(uint64_t a, uint64_t b, uint64_t) { return uint64_t(a >= b)*(a - b); }
};

// mul/div compute in a wide FLOAT work type (W = float for <=16-bit/f16/bf/f32, double for
// 32/64-bit/f64), matching cv::multiply/divide; the executor casts the work-type result down to
// rdepth. scalar (reference) path only for now - SIMD can follow. (No vec(): never instantiated.)
// vec(a, b, scale): the 3rd arg is the scale vector. COMMUTATIVE ops (mul, absdiff) ignore it - their
// scale is folded into the (cheaper) preproc of one operand. DIVISION is NOT commutative, so its scale
// MUST stick to the numerator; it takes the scale in vec (numerator*scale/denominator) and leaves
// preproc as identity. vecBinaryKernel always passes vscalar to vec, so every branch (incl. a broadcast
// denominator) divides correctly. The 2-arg vec (scale==1) is the both-contiguous fast path.
struct EwMul {
    static constexpr bool useScalar = true;
    template<typename V> static V vec(const V& a, const V& b) { return v_mul(a, b); }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // integer lanes: full product clamped to the lane type (v_mul_sat) == cv::multiply semantics
    // at scale 1. These drive the scale==1 fast path on FULL-width registers (Wvec1 = the native
    // lane vector): whole-register loads/stores, widening happens inside the multiply itself.
    // The guard matches the v_mul_sat definitions above (like EwAdd's 32-bit overloads): in
    // no-SIMD builds the vector typedefs still exist (intrin_cpp), but the helpers do not.
    static v_uint8  vec(const v_uint8& a,  const v_uint8& b)  { return v_mul_sat(a, b); }
    static v_int8   vec(const v_int8& a,   const v_int8& b)   { return v_mul_sat(a, b); }
    static v_uint16 vec(const v_uint16& a, const v_uint16& b) { return v_mul_sat(a, b); }
    static v_int16  vec(const v_int16& a,  const v_int16& b)  { return v_mul_sat(a, b); }
#ifdef EW_HAVE_MULSAT32
    static v_uint32 vec(const v_uint32& a, const v_uint32& b) { return v_mul_sat(a, b); }
    static v_int32  vec(const v_int32& a,  const v_int32& b)  { return v_mul_sat(a, b); }
#endif
#endif
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V& s) { return v_mul(a, s); }
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return a * b * s; }
};
// div has two variants by the COMMON INPUT type (matching cv::'s per-type kernel choice): integer
// inputs guard divide-by-zero -> 0 (cv:: iscalar_div); float inputs do NOT guard (cv:: fscalar_div,
// a/0 -> inf), which then saturates on the cast to an integer output exactly like cv::divide.
struct EwDivInt {
    static constexpr bool useScalar = true;
    // integer inputs computed in the float work type: guard b==0 -> 0. Scale rides the numerator (vec).
    template<typename V> static V vec(const V& a, const V& b) {
        const V z = v_setzero_<V>();
        return v_select(v_eq(b, z), z, v_div(a, b));
    }
    template<typename V> static V vec(const V& a, const V& b, const V& s) {
        const V z = v_setzero_<V>();
        return v_select(v_eq(b, z), z, v_div(v_mul(a, s), b));
    }
    template<typename V> static V preproc(const V& a, const V&) { return a; }   // identity: scale is in vec
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return b != W(0) ? a * s / b : W(0); }
};
struct EwDivFlt {
    static constexpr bool useScalar = true;
    template<typename V> static V vec(const V& a, const V& b) { return v_div(a, b); }
    template<typename V> static V vec(const V& a, const V& b, const V& s) { return v_div(v_mul(a, s), b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }   // identity: scale is in vec
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return a * s / b; }
};

// atan2(y, x) in RADIANS over the standard C range (-pi, pi] - the fastAtan2 minimax polynomial
// (mathfuncs_core.simd.hpp v_atan_f32) with the 180/pi factor dropped and the C quadrant logic
// (fastAtan2 returns degrees in [0, 360)). Absolute accuracy ~1e-5 rad, same as cv::fastAtan2.
// Generic over the universal-intrinsic float vector type.
template<typename V>
static inline V v_atan2(const V& y, const V& x)
{
    using LT = typename VTraits<V>::lane_type;
    const V eps  = v_setall_<V>((LT)DBL_EPSILON);
    const V z    = v_setzero_<V>();
    const V p7   = v_setall_<V>((LT)-0.04432655554792128);
    const V p5   = v_setall_<V>((LT)0.1555786518463281);
    const V p3   = v_setall_<V>((LT)-0.3258083974640975);
    const V p1   = v_setall_<V>((LT)0.9997878412794807);
    const V vpi2 = v_setall_<V>((LT)(CV_PI/2));
    const V vpi  = v_setall_<V>((LT)CV_PI);

    V ax = v_abs(x), ay = v_abs(y);
    V c  = v_div(v_min(ax, ay), v_add(v_max(ax, ay), eps));
    V c2 = v_mul(c, c);
    V a  = v_mul(v_fma(v_fma(v_fma(p7, c2, p5), c2, p3), c2, p1), c);
    a = v_select(v_ge(ax, ay), a, v_sub(vpi2, a));
    a = v_select(v_lt(x, z), v_sub(vpi, a), a);
    a = v_select(v_lt(y, z), v_sub(z, a), a);
    return a;
}

struct EwAtan2 {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_atan2(a, b); }   // a = y, b = x
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::atan2(a, b); }
};

// hypot(x, y) = sqrt(x^2 + y^2): NAIVE (matches cv::magnitude; overflow at |x| ~ 1e19+ for f32
// inputs is accepted), computed in the float work type; T x T -> T over the float depths.
struct EwHypot {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_sqrt(v_fma(a, a, v_mul(b, b))); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::sqrt(a*a + b*b); }
};

// min / max / absdiff: T x T -> T (same depth in and out, no scale). v_min/v_max exist for every
// vector lane type (64-bit ints fall back to scalar). absdiff uses v_absdiff (defined for the
// UNSIGNED and float lane types - signed/wide depths go through the scalar path), and the scalar
// |a-b| is computed branch-wise so it never underflows an unsigned work type.
struct EwMin {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_min(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::min(a, b); }
};
struct EwMax {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_max(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::max(a, b); }
};
struct EwAbsdiff {
    static constexpr bool useScalar = false;
    // The absdiff RESULT is the UNSIGNED type of the input width: v_absdiff(v_int8/16/32) already returns
    // v_uint8/16/32 (the true |a-b|, which can exceed the signed max), and v_absdiff on unsigned/float
    // returns the same type. vec() therefore returns THAT type (deduced), not the input V, so the kernel
    // stores it through the matching UNSIGNED v_store_pair_as (an honest same-type store) - no reinterpret,
    // no touching the saturating-narrow overloads' semantics. (This is why the kernel keeps the vec
    // result in its own variable rather than reusing the input operand.)
    template<typename V> static auto vec(const V& a, const V& b) { return v_absdiff(a, b); }
    template<typename V, typename S> static auto vec(const V& a, const V& b, const S&) { return v_absdiff(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return a > b ? W(a - b) : W(b - a); }
};

// Signed T -> SAME signed T (rdepth==T): |a-b| saturated into the signed range, in ONE pass. cv::absdiff
// keeps the signed depth, so the generic path computes absdiff->unsigned then casts back down (2 insns);
// this fuses it. v_absdiff yields the unsigned |a-b|; v_min clamps to the signed max (already >=0), then
// a same-width reinterpret to signed (all values now fit).
struct EwAbsdiffS {
    static constexpr bool useScalar = false;
    static v_int8  vec(const v_int8&  a, const v_int8&  b) { return v_reinterpret_as_s8 (v_min(v_absdiff(a, b), vx_setall_u8 (0x7f))); }
    static v_int16 vec(const v_int16& a, const v_int16& b) { return v_reinterpret_as_s16(v_min(v_absdiff(a, b), vx_setall_u16(0x7fff))); }
    static v_int32 vec(const v_int32& a, const v_int32& b) { return v_reinterpret_as_s32(v_min(v_absdiff(a, b), vx_setall_u32(0x7fffffff))); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return a > b ? W(a - b) : W(b - a); }
};

// compare: T x T -> u8 mask. cmp(a,b) is the scalar relation over a work type W (integers compare
// directly, f16/bf16 through float); vec(a,b) is the vector relation returning an all-ones/zero mask
// in the operand's own lane type. The kernel turns either into 0 / trueVal (1 or 255 via TKernel::flags).
struct EwCmpEq { template<typename W> static bool cmp(W a, W b) { return a == b; }
                 template<typename V> static V vec(const V& a, const V& b) { return v_eq(a, b); } };
struct EwCmpNe { template<typename W> static bool cmp(W a, W b) { return a != b; }
                 template<typename V> static V vec(const V& a, const V& b) { return v_ne(a, b); } };
// LT/LE are synthesized from GT/GE via the EW_KERNEL_SWAP01 flag (a<b == b>a), so no Lt/Le kernels.
struct EwCmpGt { template<typename W> static bool cmp(W a, W b) { return a > b; }
                 template<typename V> static V vec(const V& a, const V& b) { return v_gt(a, b); } };
struct EwCmpGe { template<typename W> static bool cmp(W a, W b) { return a >= b; }
                 template<typename V> static V vec(const V& a, const V& b) { return v_ge(a, b); } };

// bitwise AND/OR/XOR: bit-pattern op, type-agnostic. Run on the UNSIGNED integer whose width matches
// the element (u8/u16/u32/u64), so one functor set covers every depth. No scale, no widening (T x T ->
// T, exactly like min/max); preproc is the identity (min/max share this shape). 64-bit uses the scalar
// path (no widening vector helpers), the rest ride vecBinaryKernel's native same-type path.
struct EwAnd {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_and(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a & b); }
};
struct EwOr {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_or(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a | b); }
};
struct EwXor {
    static constexpr bool useScalar = false;
    template<typename V> static V vec(const V& a, const V& b) { return v_xor(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a ^ b); }
};

// Collapse a gap-free 2D tile to 1D (call with the per-operand x/y-steps).
#define EW_TRY_COLLAPSE(NSRC) \
    if (height > 1 && dsty == (size_t)width && \
        s0y == s0x*(size_t)width && (NSRC < 2 || s1y == s1x*(size_t)width)) \
    { width *= height; height = 1; }

// saturate_cast<T>(float|double) narrows via cvRound() -> int, so a work value past INT_MAX wraps
// to INT_MIN and saturates the wrong way (#28557). Clamping first is a no-op in range. Only
// <=16-bit T: their bounds are exact in float, INT_MAX is not. This guard is scalar-tail-only;
// CV_32S/CV_32U mul (WT=double) has an unclamped overflow of its own in v_store_pair_as
// (convert.hpp), a known, separately tracked gap this function can't reach.
template<typename Tr, typename W>
static inline Tr narrowSaturate(W v)
{
    if constexpr (std::is_floating_point<W>::value && std::is_integral<Tr>::value && sizeof(Tr) <= 2)
    {
        const W lo = (W)std::numeric_limits<Tr>::min(), hi = (W)std::numeric_limits<Tr>::max();
        v = v < lo ? lo : (v > hi ? hi : v);
    }
    return saturate_cast<Tr>(v);
}

// Unified binary kernel: T0 x T1 -> Tr (operands same depth for arithmetic; cast is separate).
//   Wvec     = work vector. Native (v_uint8/...) drives the same-type saturating path
//              (v_add saturates 8/16-bit, wraps 32-bit); v_float32 drives the widening hub.
//   WT       = scalar work type for the tail.
//   Op       = operation functor (vec()/scl()).
//   use_simd = compile-time switch; false => pure scalar (32/64-bit widened outputs, f64).
// stepx in {0,1}; dst contiguous. In-place safe (see file header).
template<typename T, typename Tr, typename WT, class Op, typename ST=WT>
static int scalarBinaryKernel(const void* src0_, size_t s0y, size_t s0x,
                              const void* src1_, size_t s1y, size_t s1x,
                              const void*, size_t, size_t, void* dst_, size_t dsty,
                              int width, int height, const double* params, int, void*)
{
    s0y /= sizeof(T);
    s1y /= sizeof(T);
    dsty /= sizeof(Tr);
    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    Tr* dst = (Tr*)dst_;
    [[maybe_unused]] ST scalar = saturate_cast<ST>(params[0]);   // mul/div scale; ignored by add/sub

    EW_TRY_COLLAPSE(2);
    for (int y = 0; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        if (s0x == s1x) {
            for (int x = 0; x < width; x++)
                dst[x] = narrowSaturate<Tr>(Op::scl((WT)src0[x], (WT)src1[x], scalar));
        }
        else if (s0x == 0) {
            WT sc0 = (WT)src0[0];
            for (int x = 0; x < width; x++)
                dst[x] = narrowSaturate<Tr>(Op::scl(sc0, (WT)src1[x], scalar));
        }
        else {
            WT sc1 = (WT)src1[0];
            for (int x = 0; x < width; x++)
                dst[x] = narrowSaturate<Tr>(Op::scl((WT)src0[x], sc1, scalar));
        }
    }
    return 0;
}

template<typename T, typename WT>
static void expandScalar(const T* sc, size_t sx, int n0, WT* scbuf, int n)
{
    int i = 0;
    for (; i < n0; i++) scbuf[i] = (WT)sc[i*sx];
    for (; i < n; i++) scbuf[i] = scbuf[i - n0];
}

// Decode the per-channel patch bytes (mask + value) from the kernel flags into small arrays. Uniform
// (no EW_CMP_PATCH): mask = trueVal, value = 0 (an ordinary compare). Per-channel: from the 4-bit
// fields. The compare kernels apply  result = (rawmask & mask) | value  per channel - folding the
// former separate patch pass into the compare (one pass).
static inline void cmpUnpackPatch(int flags, uchar trueVal, uchar mvals[4], uchar vvals[4])
{
    if (flags & EW_CMP_PATCH)
        for (int c = 0; c < 4; c++)
        {
            const int f = (flags >> (EW_CMP_PATCH_SHIFT + c*4)) & 0xF;
            mvals[c] = (uchar)cmpPatchByte(f & 3);
            vvals[c] = (uchar)cmpPatchByte((f >> 2) & 3);
        }
    else
        for (int c = 0; c < 4; c++) { mvals[c] = trueVal; vvals[c] = 0; }
}

// Scalar comparison kernel: T x T -> u8 mask. Result is 0 (false) or `trueVal` (true), trueVal read
// from TKernel::flags - 255 (default, matching cv::compare) or 1 (a numpy-style 0/1 mask). Integers
// compare directly (WT == T); f16/bf16 compare through a float WT. This is the fallback for the depths
// the SIMD kernel doesn't cover (f16/bf16/64-bit). stepx in {0,1}; dst is u8, byte step == element step.
template<typename T, typename WT, class Cmp>
static int scalarCompareKernel(const void* src0_, size_t s0y, size_t s0x,
                         const void* src1_, size_t s1y, size_t s1x,
                         const void*, size_t, size_t,
                         void* dst_, size_t dsty, int width, int height,
                         const double*, int flags, void*)
{
    // LT/LE reuse the GT/GE kernel with the operands swapped (a<b == b>a, a<=b == b>=a).
    if (flags & EW_KERNEL_SWAP01) { std::swap(src0_, src1_); std::swap(s0y, s1y); std::swap(s0x, s1x); }
    s0y /= sizeof(T);
    s1y /= sizeof(T);
    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    uchar* dst = (uchar*)dst_;
    const uchar trueVal = (flags & EW_KERNEL_MASK1) ? 1 : 255;
    uchar mvals[4], vvals[4];
    cmpUnpackPatch(flags, trueVal, mvals, vvals);
    const bool perch = (flags & EW_CMP_PATCH) != 0;          // per-channel fix-up (width=cn<=4)
    CV_Assert(!perch || width <= 4);                         // a per-channel patch is a short-row tile
    #define EW_CMP_APPLY(cond, x) (perch ? (uchar)(((cond) ? mvals[x] : 0) | vvals[x]) \
                                         : (uchar)((cond) ? trueVal : 0))

    EW_TRY_COLLAPSE(2);
    for (int y = 0; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        if (s0x == s1x) {
            for (int x = 0; x < width; x++) dst[x] = EW_CMP_APPLY(Cmp::cmp((WT)src0[x], (WT)src1[x]), x);
        }
        else if (s0x == 0) {
            WT a = (WT)src0[0];
            for (int x = 0; x < width; x++) dst[x] = EW_CMP_APPLY(Cmp::cmp(a, (WT)src1[x]), x);
        }
        else {
            WT b = (WT)src1[0];
            for (int x = 0; x < width; x++) dst[x] = EW_CMP_APPLY(Cmp::cmp((WT)src0[x], b), x);
        }
    }
    #undef EW_CMP_APPLY
    return 0;
}

// SIMD comparison kernel: T x T -> u8 mask, for the directly-comparable depths (u8/s8/u16/s16/u32/
// s32/f32). vec(a,b) gives an all-ones/zero mask in the operand's lane type; v_reinterpret_as turns it
// into the unsigned int of the same width; cmpFuse packs to u8 and applies (rawmask & M) | V. The
// per-row SIMD body runs when an operand is contiguous or broadcast; a per-channel patch (M/V differ
// by channel) only ever arrives as a short-row (width=cn) tile and is handled there.
template<typename T, typename Vvec, typename Uvec, class Cmp>
static int vecCompareKernel(const void* src0_, size_t s0y, size_t s0x,
                            const void* src1_, size_t s1y, size_t s1x,
                            const void*, size_t, size_t,
                            void* dst_, size_t dsty, int width, int height,
                            const double*, int flags, void*)
{
    // LT/LE reuse the GT/GE kernel with the operands swapped (a<b == b>a, a<=b == b>=a).
    if (flags & EW_KERNEL_SWAP01) { std::swap(src0_, src1_); std::swap(s0y, s1y); std::swap(s0x, s1x); }
    s0y /= sizeof(T);
    s1y /= sizeof(T);
    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    uchar* dst = (uchar*)dst_;
    const uchar trueVal = (flags & EW_KERNEL_MASK1) ? 1 : 255;
    uchar mvals[4], vvals[4];                                 // per-channel fix-up: (rawmask & m) | v
    cmpUnpackPatch(flags, trueVal, mvals, vvals);
    const bool perch = (flags & EW_CMP_PATCH) != 0;
    CV_Assert(!perch || width <= 4);                          // a per-channel patch is a short-row tile

    EW_TRY_COLLAPSE(2);
    int y = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // Short rows (2/3/4 elements): a per-channel scalar over a multi-channel image arrives as a
    // (width=cn) x (height=pixels) tile with the scalar broadcast over rows (s?y==0). The per-row SIMD
    // below never triggers at such a tiny width, so expand the width<=4 broadcast operand (threshold
    // AND the interleaved per-channel mask/value bytes) across the lanes and compare many rows at once.
    // This is also the ONLY path a per-channel patch (M/V differ by channel) reaches.
    // constexpr: the short-row loads T natively (vx_load->Vvec), so it exists ONLY at native width. The
    // widened f16/bf16->f32 path (sizeof(T) < Vvec lane) and 8-byte f64 fall through to per-row + scalar.
    // (sizeof(T) <= 4 is part of the constexpr gate: for f64 the body would compile to nothing but
    // dead stores - gcc 9 flags scbuf/bsrc/bsx as set-but-not-used there.)
    if constexpr (sizeof(T) == sizeof(typename VTraits<Vvec>::lane_type) && sizeof(T) <= 4)
    if (height > 1 && width <= 4 &&
        ((s0y == 0 && s1y == (size_t)width*s1x) || (s1y == 0 && s0y == (size_t)width*s0x)) &&
        dsty == (size_t)width)
    {
        const int VECSZ  = VTraits<Vvec>::vlanes();          // sizeof(T)-type lanes
        const int VECSZ8 = VTraits<v_uint8>::vlanes();       // u8 output lanes
        constexpr int MAXV8 = VTraits<v_uint8>::max_nlanes;
        T scbuf[MAXV8 * 3] = {};                             // interleaved threshold (elements)
        uchar mbuf[MAXV8 * 3] = {}, vbuf[MAXV8 * 3] = {};    // interleaved mask / value (bytes)
        // ^ the {} inits are for -Wmaybe-uninitialized only: expandScalar/decode fill every lane that
        //   is later read, but the compiler cannot prove it with a runtime VECSZ
        const bool bc0 = (s0y == 0);                         // src0 is the broadcast (short) operand
        const T* bsrc = bc0 ? src0 : src1; const size_t bsx = bc0 ? s0x : s1x;
        const T*& src = bc0 ? src1 : src0;                   // the row-stepping operand (advanced below)
        const size_t sy = bc0 ? s1y : s0y;
        // The compare direction (scalar-first when bc0, array-first otherwise) is hoisted OUT of the row
        // loop with an explicit if(bc0) - the loop is written twice so no per-iteration branch remains.
        // Per branch: own unroll, threshold + interleaved M/V loaded ONCE. sizeof==1 keeps 12 vectors.
        #define EW_CMPR(r, a, b) v_reinterpret_as(Cmp::vec(a, b), r)   // rawmask -> sizeof(T)-byte uint
        if constexpr (sizeof(T) == 1)                        // 3 u8 masks -> 3 u8 stores
        {
            const int ewidth = VECSZ8 * 3;
            expandScalar(bsrc, bsx, width, scbuf, ewidth);
            expandScalar(mvals, (size_t)1, width, mbuf, ewidth);
            expandScalar(vvals, (size_t)1, width, vbuf, ewidth);
            const int dy = ewidth / width;
            Vvec  sc0=vx_load(scbuf),  sc1=vx_load(scbuf+VECSZ),   sc2=vx_load(scbuf+VECSZ*2);
            v_uint8 M0=vx_load(mbuf),  M1=vx_load(mbuf+VECSZ8),    M2=vx_load(mbuf+VECSZ8*2);
            v_uint8 V0=vx_load(vbuf),  V1=vx_load(vbuf+VECSZ8),    V2=vx_load(vbuf+VECSZ8*2);
            #define EW_ROW1(A0,B0,A1,B1,A2,B2) \
                for (; y + dy <= height; y += dy, src += sy*dy, dst += dsty*dy) { \
                    Uvec r0,r1,r2; EW_CMPR(r0,A0,B0); EW_CMPR(r1,A1,B1); EW_CMPR(r2,A2,B2); \
                    v_uint8 o0=v_or(v_and(r0,M0),V0),o1=v_or(v_and(r1,M1),V1),o2=v_or(v_and(r2,M2),V2); \
                    v_store(dst, o0); v_store(dst+VECSZ8, o1); v_store(dst+VECSZ8*2, o2); }
            if (bc0) EW_ROW1(sc0,vx_load(src), sc1,vx_load(src+VECSZ), sc2,vx_load(src+VECSZ*2))
            else     EW_ROW1(vx_load(src),sc0, vx_load(src+VECSZ),sc1, vx_load(src+VECSZ*2),sc2)
            #undef EW_ROW1
        }
        else if constexpr (sizeof(T) == 2)                   // 3 u16 masks -> 1 full + 1 half u8 store
        {
            const int ewidth = VECSZ * 3;
            expandScalar(bsrc, bsx, width, scbuf, ewidth);
            expandScalar(mvals, (size_t)1, width, mbuf, ewidth);
            expandScalar(vvals, (size_t)1, width, vbuf, ewidth);
            const int dy = ewidth / width;
            Vvec  sc0=vx_load(scbuf), sc1=vx_load(scbuf+VECSZ), sc2=vx_load(scbuf+VECSZ*2);
            v_uint8 M0=vx_load(mbuf), M1=vx_load(mbuf+VECSZ8);
            v_uint8 V0=vx_load(vbuf), V1=vx_load(vbuf+VECSZ8);
            #define EW_ROW2(A0,B0,A1,B1,A2,B2) \
                for (; y + dy <= height; y += dy, src += sy*dy, dst += dsty*dy) { \
                    Uvec r0,r1,r2; EW_CMPR(r0,A0,B0); EW_CMPR(r1,A1,B1); EW_CMPR(r2,A2,B2); \
                    v_uint8 b0=v_pack(r0,r1), b1=v_pack(r2,r2); \
                    v_uint8 o0=v_or(v_and(b0,M0),V0), o1=v_or(v_and(b1,M1),V1); \
                    v_store(dst, o0); v_store_low(dst+VECSZ8, o1); }
            if (bc0) EW_ROW2(sc0,vx_load(src), sc1,vx_load(src+VECSZ), sc2,vx_load(src+VECSZ*2))
            else     EW_ROW2(vx_load(src),sc0, vx_load(src+VECSZ),sc1, vx_load(src+VECSZ*2),sc2)
            #undef EW_ROW2
        }
        else if constexpr (sizeof(T) == 4)                   // sizeof==4: 6 u32 -> 3 u16 -> 1 full + 1 half u8
        {
            const int ewidth = VECSZ * 6;
            expandScalar(bsrc, bsx, width, scbuf, ewidth);
            expandScalar(mvals, (size_t)1, width, mbuf, ewidth);
            expandScalar(vvals, (size_t)1, width, vbuf, ewidth);
            const int dy = ewidth / width;
            Vvec sc0=vx_load(scbuf),sc1=vx_load(scbuf+VECSZ),sc2=vx_load(scbuf+VECSZ*2),
                 sc3=vx_load(scbuf+VECSZ*3),sc4=vx_load(scbuf+VECSZ*4),sc5=vx_load(scbuf+VECSZ*5);
            v_uint8 M0=vx_load(mbuf), M1=vx_load(mbuf+VECSZ8);
            v_uint8 V0=vx_load(vbuf), V1=vx_load(vbuf+VECSZ8);
            #define EW_ROW4(A0,B0,A1,B1,A2,B2,A3,B3,A4,B4,A5,B5) \
                for (; y + dy <= height; y += dy, src += sy*dy, dst += dsty*dy) { \
                    Uvec r0,r1,r2,r3,r4,r5; \
                    EW_CMPR(r0,A0,B0); EW_CMPR(r1,A1,B1); EW_CMPR(r2,A2,B2); \
                    EW_CMPR(r3,A3,B3); EW_CMPR(r4,A4,B4); EW_CMPR(r5,A5,B5); \
                    v_uint16 p0=v_pack(r0,r1), p1=v_pack(r2,r3), p2=v_pack(r4,r5); \
                    v_uint8 q0=v_pack(p0,p1), q1=v_pack(p2,p2); \
                    v_uint8 o0=v_or(v_and(q0,M0),V0), o1=v_or(v_and(q1,M1),V1); \
                    v_store(dst, o0); v_store_low(dst+VECSZ8, o1); }
            if (bc0) EW_ROW4(sc0,vx_load(src),sc1,vx_load(src+VECSZ),sc2,vx_load(src+VECSZ*2),
                             sc3,vx_load(src+VECSZ*3),sc4,vx_load(src+VECSZ*4),sc5,vx_load(src+VECSZ*5))
            else     EW_ROW4(vx_load(src),sc0,vx_load(src+VECSZ),sc1,vx_load(src+VECSZ*2),sc2,
                             vx_load(src+VECSZ*3),sc3,vx_load(src+VECSZ*4),sc4,vx_load(src+VECSZ*5),sc5)
            #undef EW_ROW4
        }
        #undef EW_CMPR
    }
#endif
    for (; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
        // SIMD for BOTH the contiguous case AND a broadcast operand (a per-channel scalar const has
        // step 0 - without this the scalar-vs-array compare, incl. every multi-channel scalar compare,
        // fell to the scalar tail below, ~5x slower). vx_setall broadcasts the step-0 operand once.
        // 4-vector unroll + halide right-edge backoff (reprocess the last 4*VECSZ when width is not a
        // multiple); dst is a separate mask buffer so the overlap is a harmless idempotent rewrite.
        const int VECSZ = VTraits<Vvec>::vlanes();
        const int VECSZ8 = VTraits<v_uint8>::vlanes();
        v_uint8 vT; v_setall_mask(vT, trueVal);              // uniform mask (per-channel patch is short-row)
        const bool tailTrick = width >= 4*VECSZ && src0_ != dst_ && src1_ != dst_;
        // Narrow the 4 masks to u8 FIRST, then apply trueVal on the u8 result (1 and per u8 vector, not
        // per wide lane-group). One block per MASK lane width (== sizeof(T) natively, but 4 for the
        // f16/bf16->f32 widened path); a 4-vector unroll spans exactly 4*VECSZ elements.
        constexpr size_t MW = sizeof(typename VTraits<Uvec>::lane_type);
        #define PACK_STORE_CMP_RESULT(m0,m1,m2,m3, D) do { \
            if constexpr (MW == 1u) { \
                v_uint8 o0=v_and(m0,vT),o1=v_and(m1,vT),o2=v_and(m2,vT),o3=v_and(m3,vT); \
                v_store((D),o0); v_store((D)+VECSZ8,o1); v_store((D)+VECSZ8*2,o2); v_store((D)+VECSZ8*3,o3); \
            } else if constexpr (MW == 2u) { \
                v_uint8 o0=v_and(v_pack(m0,m1),vT), o1=v_and(v_pack(m2,m3),vT); \
                v_store((D),o0); v_store((D)+VECSZ8,o1); \
            } else if constexpr (MW == 4u) { \
                v_uint8 o0=v_and(v_pack(v_pack(m0,m1),v_pack(m2,m3)),vT); \
                v_store((D),o0); \
            } else { /* sizeof==8: 4 u64 -> 2 u32 -> 1 u16 -> u8 (low half), 4*VECSZ elems -> store_low */ \
                v_uint16 g=v_pack(v_pack(m0,m1),v_pack(m2,m3)); \
                v_uint8 o0=v_and(v_pack(g,g),vT); \
                v_store_low((D),o0); \
            } } while(0)
        if (s0x == 1u && s1x == 1u)
        {
            for (; x < width; x += 4*VECSZ)
            {
                if (x + 4*VECSZ > width) { if (!tailTrick) break; x = width - 4*VECSZ; }
                Vvec a0, a1, a2, a3, b0, b1, b2, b3;   // vx_load_pair_as widens f16/bf16->f32, else native
                vx_load_pair_as(src0+x, a0, a1); vx_load_pair_as(src0+x+2*VECSZ, a2, a3);
                vx_load_pair_as(src1+x, b0, b1); vx_load_pair_as(src1+x+2*VECSZ, b2, b3);
                Uvec m0, m1, m2, m3;
                v_reinterpret_as(Cmp::vec(a0, b0), m0); v_reinterpret_as(Cmp::vec(a1, b1), m1);
                v_reinterpret_as(Cmp::vec(a2, b2), m2); v_reinterpret_as(Cmp::vec(a3, b3), m3);
                PACK_STORE_CMP_RESULT(m0, m1, m2, m3, dst+x);
            }
        }
        else if (s1x == 0u)                       // src1 (e.g. the scalar) broadcast
        {
            Vvec b0; vx_setall_as(src1, b0);
            for (; x < width; x += 4*VECSZ)
            {
                if (x + 4*VECSZ > width) { if (!tailTrick) break; x = width - 4*VECSZ; }
                Vvec a0, a1, a2, a3;
                vx_load_pair_as(src0+x, a0, a1); vx_load_pair_as(src0+x+2*VECSZ, a2, a3);
                Uvec m0, m1, m2, m3;
                v_reinterpret_as(Cmp::vec(a0, b0), m0); v_reinterpret_as(Cmp::vec(a1, b0), m1);
                v_reinterpret_as(Cmp::vec(a2, b0), m2); v_reinterpret_as(Cmp::vec(a3, b0), m3);
                PACK_STORE_CMP_RESULT(m0, m1, m2, m3, dst+x);
            }
        }
        else if (s0x == 0u)                       // src0 broadcast
        {
            Vvec a0; vx_setall_as(src0, a0);
            for (; x < width; x += 4*VECSZ)
            {
                if (x + 4*VECSZ > width) { if (!tailTrick) break; x = width - 4*VECSZ; }
                Vvec b0, b1, b2, b3;
                vx_load_pair_as(src1+x, b0, b1); vx_load_pair_as(src1+x+2*VECSZ, b2, b3);
                Uvec m0, m1, m2, m3;
                v_reinterpret_as(Cmp::vec(a0, b0), m0); v_reinterpret_as(Cmp::vec(a0, b1), m1);
                v_reinterpret_as(Cmp::vec(a0, b2), m2); v_reinterpret_as(Cmp::vec(a0, b3), m3);
                PACK_STORE_CMP_RESULT(m0, m1, m2, m3, dst+x);
            }
        }
        #undef PACK_STORE_CMP_RESULT
    #endif
        // Fold the fix-up: raw all-ones/zero mask -> (raw & M) | V. Uniform => raw & trueVal (V=0); a
        // per-channel patch (width=cn<=4 here) indexes M/V by the channel x.
        #define EW_CMP_APPLY(cond, x) (perch ? (uchar)(((cond) ? mvals[x] : 0) | vvals[x]) \
                                             : (uchar)((cond) ? trueVal : 0))
        if (s0x == s1x) {
            for (; x < width; x++) dst[x] = EW_CMP_APPLY(Cmp::cmp(src0[x], src1[x]), x);
        }
        else if (s0x == 0) {
            T a = src0[0];
            for (; x < width; x++) dst[x] = EW_CMP_APPLY(Cmp::cmp(a, src1[x]), x);
        }
        else {
            T b = src1[0];
            for (; x < width; x++) dst[x] = EW_CMP_APPLY(Cmp::cmp(src0[x], b), x);
        }
        #undef EW_CMP_APPLY
    }
    return 0;
}

// Unified binary kernel: T0 x T1 -> Tr (operands same depth for arithmetic; cast is separate).
//   Wvec     = work vector. Native (v_uint8/...) drives the same-type saturating path
//              (v_add saturates 8/16-bit, wraps 32-bit); v_float32 drives the widening hub.
//   WT       = scalar work type for the tail.
//   Op       = operation functor (vec()/scl()).
//   use_simd = compile-time switch; false => pure scalar (32/64-bit widened outputs, f64).
// stepx in {0,1}; dst contiguous. In-place safe (see file header).
template<typename T, typename Tr, typename Wvec, typename WT, class Op, typename ST=WT, typename Wvec1=Wvec>
static int vecBinaryKernel(const void* src0_, size_t s0y, size_t s0x,
                         const void* src1_, size_t s1y, size_t s1x,
                         const void*, size_t, size_t,
                         void* dst_, size_t dsty, int width, int height,
                         const double* params, int, void*)
{
    s0y /= sizeof(T);
    s1y /= sizeof(T);
    dsty /= sizeof(Tr);

    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    Tr* dst = (Tr*)dst_;
    [[maybe_unused]] ST scalar = saturate_cast<ST>(params[0]);   // mul/div scale; ignored by add/sub

    EW_TRY_COLLAPSE(2);
    int y = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    using Wlane = typename VTraits<Wvec>::lane_type;
    const int VECSZ = VTraits<Wvec>::vlanes();
    const bool use_tail_trick = width >= VECSZ*4 && src0_ != dst_ && src1_ != dst_;
    [[maybe_unused]] Wvec vscalar;
    vx_setall_as(&scalar, vscalar);
    if (height > 1 && width <= 4 &&
        ((s0y == 0 && s1y == width*s1x) ||
         (s1y == 0 && s0y == width*s0x)) &&
        dsty == (size_t)width) {
        constexpr int MAXVECSZ = VTraits<Wvec>::max_nlanes;
        Wlane scbuf[MAXVECSZ*6] = {};   // {} for -Wmaybe-uninitialized only (filled up to VECSZ*6)
        const int ewidth = VECSZ*6;
        expandScalar(s0y == 0 ? src0 : src1, s0y == 0 ? s0x : s1x, width, scbuf, ewidth);
        int dy = ewidth / width;
        Wvec sc0, sc1, sc2, sc3, sc4, sc5;
        sc0 = Op::preproc(vx_load(scbuf), vscalar);
        sc1 = Op::preproc(vx_load(scbuf + VECSZ), vscalar);
        sc2 = Op::preproc(vx_load(scbuf + VECSZ*2), vscalar);
        sc3 = Op::preproc(vx_load(scbuf + VECSZ*3), vscalar);
        sc4 = Op::preproc(vx_load(scbuf + VECSZ*4), vscalar);
        sc5 = Op::preproc(vx_load(scbuf + VECSZ*5), vscalar);

        if (s0y == 0) {
            for (; y + dy <= height; y += dy, src1 += s1y*dy, dst += dsty*dy) {
                Wvec v0, v1, v2, v3, v4, v5;
                vx_load_pair_as(src1, v0, v1);
                vx_load_pair_as(src1 + VECSZ*2, v2, v3);
                vx_load_pair_as(src1 + VECSZ*4, v4, v5);
                auto w0 = Op::vec(sc0, v0, vscalar), w1 = Op::vec(sc1, v1, vscalar),
                     w2 = Op::vec(sc2, v2, vscalar), w3 = Op::vec(sc3, v3, vscalar),
                     w4 = Op::vec(sc4, v4, vscalar), w5 = Op::vec(sc5, v5, vscalar);
                v_store_pair_as(dst, w0, w1);
                v_store_pair_as(dst + VECSZ*2, w2, w3);
                v_store_pair_as(dst + VECSZ*4, w4, w5);
            }
        }
        else {
            for (; y + dy <= height; y += dy, src0 += s0y*dy, dst += dsty*dy) {
                Wvec v0, v1, v2, v3, v4, v5;
                vx_load_pair_as(src0, v0, v1);
                vx_load_pair_as(src0 + VECSZ*2, v2, v3);
                vx_load_pair_as(src0 + VECSZ*4, v4, v5);
                auto w0 = Op::vec(v0, sc0, vscalar), w1 = Op::vec(v1, sc1, vscalar),
                     w2 = Op::vec(v2, sc2, vscalar), w3 = Op::vec(v3, sc3, vscalar),
                     w4 = Op::vec(v4, sc4, vscalar), w5 = Op::vec(v5, sc5, vscalar);
                v_store_pair_as(dst, w0, w1);
                v_store_pair_as(dst + VECSZ*2, w2, w3);
                v_store_pair_as(dst + VECSZ*4, w4, w5);
            }
        }
    }
#endif
    for (; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
        Wvec a0, a1, a2, a3, b0, b1, b2, b3;
        if (s0x == s1x) {
            if (!Op::useScalar || scalar == ST(1)) {
                // this branch runs on Wvec1, which may be WIDER-laned than Wvec (e.g. the u8 mul
                // path: Wvec1=v_uint16 is 2x the lanes of Wvec=v_float32) - step by ITS lane count,
                // or the pairs overlap and the tail backoff writes past the row end
                const int VECSZ1 = VTraits<Wvec1>::vlanes();
                const bool tail_trick1 = width >= VECSZ1*4 && src0_ != dst_ && src1_ != dst_;
                for (; x < width; x += VECSZ1*4) {
                    Wvec1 a0_, a1_, a2_, a3_, b0_, b1_, b2_, b3_;
                    if (x + VECSZ1*4 > width) { if (!tail_trick1) break; x = width - VECSZ1*4; }
                    vx_load_pair_as(src0 + x, a0_, a1_);
                    vx_load_pair_as(src0 + x + VECSZ1*2, a2_, a3_);
                    vx_load_pair_as(src1 + x, b0_, b1_);
                    vx_load_pair_as(src1 + x + VECSZ1*2, b2_, b3_);
                    auto c0 = Op::vec(a0_, b0_), c1 = Op::vec(a1_, b1_),
                         c2 = Op::vec(a2_, b2_), c3 = Op::vec(a3_, b3_);
                    v_store_pair_as(dst + x, c0, c1);
                    v_store_pair_as(dst + x + VECSZ1*2, c2, c3);
                }
            }
            else {
                for (; x < width; x += VECSZ*4) {
                    if (x + VECSZ*4 > width) { if (!use_tail_trick) break; x = width - VECSZ*4; }
                    vx_load_pair_as(src0 + x, a0, a1);
                    vx_load_pair_as(src0 + x + VECSZ*2, a2, a3);
                    vx_load_pair_as(src1 + x, b0, b1);
                    vx_load_pair_as(src1 + x + VECSZ*2, b2, b3);
                    auto c0 = Op::vec(Op::preproc(a0, vscalar), b0, vscalar),
                         c1 = Op::vec(Op::preproc(a1, vscalar), b1, vscalar),
                         c2 = Op::vec(Op::preproc(a2, vscalar), b2, vscalar),
                         c3 = Op::vec(Op::preproc(a3, vscalar), b3, vscalar);
                    v_store_pair_as(dst + x, c0, c1);
                    v_store_pair_as(dst + x + VECSZ*2, c2, c3);
                }
            }
        }
        else if (s1x == 0) {
            vx_setall_as(src1, b0);
            b0 = Op::preproc(b0, vscalar);
            for (; x < width; x += VECSZ*4) {
                if (x + VECSZ*4 > width) { if (!use_tail_trick) break; x = width - VECSZ*4; }
                vx_load_pair_as(src0 + x, a0, a1);
                vx_load_pair_as(src0 + x + VECSZ*2, a2, a3);
                auto c0 = Op::vec(a0, b0, vscalar), c1 = Op::vec(a1, b0, vscalar),
                     c2 = Op::vec(a2, b0, vscalar), c3 = Op::vec(a3, b0, vscalar);
                v_store_pair_as(dst + x, c0, c1);
                v_store_pair_as(dst + x + VECSZ*2, c2, c3);
            }
        }
        else {
            vx_setall_as(src0, b0);
            b0 = Op::preproc(b0, vscalar);
            for (; x < width; x += VECSZ*4) {
                if (x + VECSZ*4 > width) { if (!use_tail_trick) break; x = width - VECSZ*4; }
                vx_load_pair_as(src1 + x, a0, a1);
                vx_load_pair_as(src1 + x + VECSZ*2, a2, a3);
                auto c0 = Op::vec(b0, a0, vscalar), c1 = Op::vec(b0, a1, vscalar),
                     c2 = Op::vec(b0, a2, vscalar), c3 = Op::vec(b0, a3, vscalar);
                v_store_pair_as(dst + x, c0, c1);
                v_store_pair_as(dst + x + VECSZ*2, c2, c3);
            }
        }
    #endif
        for (; x < width; x++)
            dst[x] = narrowSaturate<Tr>(Op::scl((WT)src0[x*s0x], (WT)src1[x*s1x], scalar));
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    vx_cleanup();
#endif
    return 0;
}

// ===========================================================================
// OP_ADD
// ===========================================================================

// add dispatch: T x T -> R (operands already same depth T).
//  - native saturating path (Wvec = native, use_simd=true) for T -> T on <=32-bit ints + f32
//    (32-bit lanes via the local v_add_sat/v_sub_sat - v_add/v_sub wrap there);
//  - widening f32-hub (Wvec = v_float32, use_simd=true) for the widened/half/float outputs;
//  - scalar (use_simd=false) for 64-bit results and f64 (no vector path yet).
template<class Op>
TKernel getAddSubFunc(int T, int R)
{
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_8U:
        fptr = R == CV_8U ? vecBinaryKernel<uchar, uchar, v_uint8, short, Op, uchar> :
               R == CV_16S ? vecBinaryKernel<uchar, short, v_int16, short, Op, short> :
               R == CV_32S ? vecBinaryKernel<uchar, int, v_int16, short, Op, short> :
               R == CV_32F ? vecBinaryKernel<uchar, float, v_int16, short, Op, short> : nullptr;
        break;
    case CV_8S:
        fptr =
            R == CV_8S ? vecBinaryKernel<schar, schar, v_int8, short, Op, schar> :
            R == CV_16S ? vecBinaryKernel<schar, short, v_int16, short, Op, short> :
            R == CV_32S ? vecBinaryKernel<schar, int, v_int16, short, Op, short> :
            R == CV_32F ? vecBinaryKernel<schar, float, v_int16, short, Op, short> : nullptr;
        break;
    case CV_16U:
        fptr = R == CV_16U ? vecBinaryKernel<ushort, ushort, v_uint16, int, Op, ushort> :
               R == CV_32S ? vecBinaryKernel<ushort, int, v_int32, int, Op, int> :
               R == CV_32F ? vecBinaryKernel<ushort, float, v_int32, int, Op, int> : nullptr;
        break;
    case CV_16S:
        fptr = R == CV_16S ? vecBinaryKernel<short, short, v_int16, int, Op, short> :
               R == CV_32S ? vecBinaryKernel<short, int,   v_int32, int, Op, int> :
               R == CV_32F ? vecBinaryKernel<short, float, v_int32, int, Op, int> : nullptr;
        break;
    case CV_32U:
        fptr = R == CV_32U ? vecBinaryKernel<unsigned, unsigned, v_uint32, int64_t, Op, unsigned> :
                R == CV_64S ? scalarBinaryKernel<unsigned, int64_t, int64_t, Op> :
                R == CV_64F ? scalarBinaryKernel<unsigned, double, int64_t, Op> : nullptr;
        break;
    case CV_32S:
        fptr = R == CV_32S ? vecBinaryKernel<int, int, v_int32, int64_t, Op, int> :
               R == CV_64S ? scalarBinaryKernel<int, int64_t, int64_t, Op> :
               R == CV_64F ? scalarBinaryKernel<int, double, int64_t, Op> : nullptr;
        break;
    case CV_64U:
        fptr = R == CV_64U ? scalarBinaryKernel<uint64_t, uint64_t, uint64_t, Op> :
               R == CV_64F ? scalarBinaryKernel<uint64_t, double, double, Op> : nullptr;
        break;
    case CV_64S:
        fptr = R == CV_64S ? scalarBinaryKernel<int64_t, int64_t, int64_t, Op> :
               R == CV_64F ? scalarBinaryKernel<int64_t, double, double, Op> : nullptr;
        break;
    case CV_16F:
        fptr =
        #if CV_SIMD_16F
        R == CV_16F ? vecBinaryKernel<hfloat, hfloat, v_float16, float, Op, hfloat> :
        #else
        R == CV_16F ? vecBinaryKernel<hfloat, hfloat, v_float32, float, Op, float> :
        #endif
        R == CV_32F ? vecBinaryKernel<hfloat, float, v_float32, float, Op, float> : nullptr;
        break;
    case CV_16BF:
        fptr =
            R == CV_16BF ? vecBinaryKernel<bfloat, bfloat, v_float32, float, Op, float> :
            R == CV_32F ? vecBinaryKernel<bfloat, float,  v_float32, float, Op, float> : nullptr;
        break;
    case CV_32F:
        fptr = R == CV_32F ? vecBinaryKernel<float, float, v_float32, float, Op, float> : nullptr;
        break;
    case CV_64F:
        #if CV_SIMD_64F
        fptr = R == CV_64F ? vecBinaryKernel<double, double, v_float64, double, Op> : nullptr;
        #else
        fptr = R == CV_64F ? scalarBinaryKernel<double, double, double, Op> : nullptr;
        #endif
        break;
    default:
        ;
    }
    return {fptr, nullptr, 0};
}

TKernel getMulFunc_(int T, int R)
{
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_8U:
        fptr =   // scale==1 fast path: whole u8 registers, v_mul_sat widens+clamps inside;
                 // Wvec (f16 where available, else f32) for the scale path
        #if CV_SIMD_16F
            R == CV_8U ? vecBinaryKernel<uchar, uchar, v_float16, float, EwMul, float, v_uint8> :
        #else
            R == CV_8U ? vecBinaryKernel<uchar, uchar, v_float32, float, EwMul, float, v_uint8> :
        #endif
            R == CV_32F ? vecBinaryKernel<uchar, float, v_float32, float, EwMul> : nullptr;
        break;
    case CV_8S:
        fptr =   // scale==1 fast path: whole s8 registers via v_mul_sat
        #if CV_SIMD_16F
            R == CV_8S ? vecBinaryKernel<schar, schar, v_float16, float, EwMul, float, v_int8> :
        #else
            R == CV_8S ? vecBinaryKernel<schar, schar, v_float32, float, EwMul, float, v_int8> :
        #endif
            R == CV_32F ? vecBinaryKernel<schar, float, v_float32, float, EwMul> : nullptr;
        break;
    case CV_16U:
        fptr =   // scale==1 fast path: whole u16 registers via v_mul_sat; Wvec=f32 for scale
            R == CV_16U ? vecBinaryKernel<ushort, ushort, v_float32, float, EwMul, float, v_uint16> :
            R == CV_32F ? vecBinaryKernel<ushort, float,  v_float32, float, EwMul> : nullptr;
        break;
    case CV_16S:
        fptr =   // scale==1 fast path: whole s16 registers via v_mul_sat; Wvec=f32 for scale
            R == CV_16S ? vecBinaryKernel<short, short, v_float32, float, EwMul, float, v_int16> :
            R == CV_32F ? vecBinaryKernel<short, float, v_float32, float, EwMul> : nullptr;
        break;
    case CV_16F:
        fptr =
            R == CV_16F ? vecBinaryKernel<hfloat, hfloat, v_float32, float, EwMul> :
            R == CV_32F ? vecBinaryKernel<hfloat, float,  v_float32, float, EwMul> : nullptr;
        break;
    case CV_16BF:
        fptr =
            R == CV_16BF ? vecBinaryKernel<bfloat, bfloat, v_float32, float, EwMul> :
            R == CV_32F ? vecBinaryKernel<bfloat, float,  v_float32, float, EwMul> : nullptr;
        break;
    case CV_32F:
        fptr = R == CV_32F ? vecBinaryKernel<float, float, v_float32, float, EwMul> : nullptr;
        break;
    case CV_32U:
        fptr =   // scale==1 fast path (where v_mul_sat32 exists): whole u32 registers, widening
                 // multiply + saturating narrow; the f64 work vector serves the scale path
        #if defined(EW_HAVE_MULSAT32) && CV_SIMD_64F
            R == CV_32U ? vecBinaryKernel<unsigned, unsigned, v_float64, double, EwMul, double, v_uint32> :
            R == CV_64F ? vecBinaryKernel<unsigned, double,   v_float64, double, EwMul> : nullptr;
        #elif CV_SIMD_64F
            R == CV_32U ? vecBinaryKernel<unsigned, unsigned, v_float64, double, EwMul> :
            R == CV_64F ? vecBinaryKernel<unsigned, double,   v_float64, double, EwMul> : nullptr;
        #else
            R == CV_32U ? scalarBinaryKernel<unsigned, unsigned, double, EwMul> :
            R == CV_64F ? scalarBinaryKernel<unsigned, double,   double, EwMul> : nullptr;
        #endif
        break;
    case CV_32S:
        fptr =   // scale==1 fast path: see CV_32U
        #if defined(EW_HAVE_MULSAT32) && CV_SIMD_64F
            R == CV_32S ? vecBinaryKernel<int, int,    v_float64, double, EwMul, double, v_int32> :
            R == CV_64F ? vecBinaryKernel<int, double, v_float64, double, EwMul> : nullptr;
        #elif CV_SIMD_64F
            R == CV_32S ? vecBinaryKernel<int, int,    v_float64, double, EwMul> :
            R == CV_64F ? vecBinaryKernel<int, double, v_float64, double, EwMul> : nullptr;
        #else
            R == CV_32S ? scalarBinaryKernel<int, int,    double, EwMul> :
            R == CV_64F ? scalarBinaryKernel<int, double, double, EwMul> : nullptr;
        #endif
        break;
    case CV_64U:
        fptr = R == CV_64F ? scalarBinaryKernel<uint64_t, double, double, EwMul> : nullptr;
        break;
    case CV_64S:
        fptr = R == CV_64F ? scalarBinaryKernel<int64_t,  double, double, EwMul> : nullptr;
        break;
    case CV_64F:
        #if CV_SIMD_64F
        fptr = R == CV_64F ? vecBinaryKernel<double, double, v_float64, double, EwMul> : nullptr;
        #else
        fptr = R == CV_64F ? scalarBinaryKernel<double,   double, double, EwMul> : nullptr;
        #endif
        break;
    default:
        ;
    }
    return {fptr, nullptr, 0};
}

// `checked` (decided by the CALLER from the ORIGINAL input types) selects the divide-by-zero
// policy, INDEPENDENT of the work type R: EwDivInt guards /0 -> 0 for integer-semantics division,
// EwDivFlt does not (a/0 -> inf, saturating on a later cast, as cv::divide does for float inputs).
// It must apply on EVERY row - two wide integers (e.g. 32U / 64S) promote to a 64F work type yet
// still need the integer guard, so the float-work rows can't hardcode EwDivFlt.
TKernel getDivFunc_(int T, int R, bool checked)
{
    KernelFunc fptr = nullptr;
    // SIMD via vecBinaryKernel: div scales the numerator in vec (a*scale/b), EwDivFlt (float a/0 -> inf,
    // saturates on cast) / EwDivInt (v_select guards b==0 -> 0 post-facto). Prefer a DIRECT T->T kernel
    // (1 pass, saturate on store) - emitBinary probes result==T first; a T->f32/f64 kernel serves an
    // explicit float dtype. Work type: f16 for 8-bit (CV_SIMD_16F), else f32; f64 for 32/64-bit.
    #define DIV(T_, Tr_, Wv_, W_) (checked ? vecBinaryKernel<T_, Tr_, Wv_, W_, EwDivInt> \
                                           : vecBinaryKernel<T_, Tr_, Wv_, W_, EwDivFlt>)
#if CV_SIMD_16F
    #define DIV8(T_)   DIV(T_, T_, v_float16, float)                      // 8-bit T->T, f16 work
#else
    #define DIV8(T_)   DIV(T_, T_, v_float32, float)
#endif
#if CV_SIMD_64F
    #define DIVW(T_, Tr_) DIV(T_, Tr_, v_float64, double)                 // 32/64-bit, f64 SIMD
#else
    #define DIVW(T_, Tr_) (checked ? scalarBinaryKernel<T_, Tr_, double, EwDivInt> \
                                    : scalarBinaryKernel<T_, Tr_, double, EwDivFlt>)
#endif
    switch (T)
    {
    case CV_8U:   fptr = R==CV_8U  ? DIV8(uchar)  : R==CV_32F ? DIV(uchar,  float, v_float32, float) : nullptr; break;
    case CV_8S:   fptr = R==CV_8S  ? DIV8(schar)  : R==CV_32F ? DIV(schar,  float, v_float32, float) : nullptr; break;
    case CV_16U:  fptr = R==CV_16U ? DIV(ushort, ushort, v_float32, float) : R==CV_32F ? DIV(ushort, float, v_float32, float) : nullptr; break;
    case CV_16S:  fptr = R==CV_16S ? DIV(short,  short,  v_float32, float) : R==CV_32F ? DIV(short,  float, v_float32, float) : nullptr; break;
    case CV_16F:  fptr = R==CV_16F ? DIV(hfloat, hfloat, v_float32, float) : R==CV_32F ? DIV(hfloat, float, v_float32, float) : nullptr; break;
    case CV_16BF: fptr = R==CV_16BF? DIV(bfloat, bfloat, v_float32, float) : R==CV_32F ? DIV(bfloat, float, v_float32, float) : nullptr; break;
    case CV_32F:  fptr = R==CV_32F ? DIV(float,  float,  v_float32, float) : nullptr; break;
    case CV_32U:  fptr = R==CV_32U ? DIVW(unsigned, unsigned) : R==CV_64F ? DIVW(unsigned, double) : nullptr; break;
    case CV_32S:  fptr = R==CV_32S ? DIVW(int,      int)      : R==CV_64F ? DIVW(int,      double) : nullptr; break;
    case CV_64U:  fptr = R==CV_64F ? DIVW(uint64_t, double) : nullptr; break;
    case CV_64S:  fptr = R==CV_64F ? DIVW(int64_t,  double) : nullptr; break;
    case CV_64F:  fptr = R==CV_64F ? DIVW(double,   double) : nullptr; break;
    default:      ;
    }
    #undef DIV
    #undef DIV8
    #undef DIVW
    return {fptr, nullptr, 0};
}


// min / max: T x T -> T for every depth. Native v_min/v_max on the matching lane type (8/16/32-bit
// ints, f16/bf16/f32); 64-bit ints and f64 use the scalar path. Op = EwMin or EwMax.
template<class Op>
static TKernel getMinMaxFunc(int T)
{
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_8U:   fptr = vecBinaryKernel<uchar,    uchar,    v_uint8,   short,   Op, uchar>; break;
    case CV_8S:   fptr = vecBinaryKernel<schar,    schar,    v_int8,    short,   Op, schar>; break;
    case CV_16U:  fptr = vecBinaryKernel<ushort,   ushort,   v_uint16,  int,     Op, ushort>; break;
    case CV_16S:  fptr = vecBinaryKernel<short,    short,    v_int16,   int,     Op, short>; break;
    case CV_32U:  fptr = vecBinaryKernel<unsigned, unsigned, v_uint32,  int64_t, Op, unsigned>; break;
    case CV_32S:  fptr = vecBinaryKernel<int,      int,      v_int32,   int64_t, Op, int>; break;
    case CV_16F:
        #if CV_SIMD_16F
        fptr = vecBinaryKernel<hfloat,   hfloat,   v_float16, float,   Op, hfloat>;
        #else
        fptr = vecBinaryKernel<hfloat,   hfloat,   v_float32, float,   Op, float>;
        #endif
        break;
    case CV_16BF: fptr = vecBinaryKernel<bfloat, bfloat, v_float32, float, Op, float>; break;
    case CV_32F:  fptr = vecBinaryKernel<float, float, v_float32, float, Op, float>; break;
    case CV_64U:  fptr = scalarBinaryKernel<uint64_t, uint64_t, uint64_t, Op>; break;
    case CV_64S:  fptr = scalarBinaryKernel<int64_t,  int64_t,  int64_t,  Op>; break;
    #if CV_SIMD_64F
    case CV_64F:  fptr = vecBinaryKernel<double, double, v_float64, double, Op, double>; break;
    #else
    case CV_64F:  fptr = scalarBinaryKernel<double,   double,   double,   Op>; break;
    #endif
    default:      ;
    }
    return {fptr, nullptr, 0};
}

// absdiff: |a-b|, T x T -> R.
TKernel getAbsdiffFunc_(int T, int R)
{
    // Signed T has TWO outputs: R==T -> EwAbsdiffS (saturating |a-b| kept in the signed range, native work,
    // ONE pass - the depth cv::absdiff keeps); R==(the unsigned type of the same width) -> EwAbsdiff (the
    // true |a-b| via v_absdiff, for an explicit unsigned dst). Unsigned/float T only produce R==T. Each case
    // returns nullptr for a wrong R (no separate rdepth guard).
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_8U:   fptr = R == CV_8U  ? vecBinaryKernel<uchar,  uchar,  v_uint8,  short,   EwAbsdiff, uchar>  : nullptr; break;
    case CV_16U:  fptr = R == CV_16U ? vecBinaryKernel<ushort, ushort, v_uint16, int,     EwAbsdiff, ushort> : nullptr; break;
    case CV_32U:  fptr = R == CV_32U ? vecBinaryKernel<unsigned, unsigned, v_uint32, int64_t, EwAbsdiff, unsigned> : nullptr; break;
    case CV_8S:   fptr = R == CV_8S  ? vecBinaryKernel<schar, schar, v_int8,  short,   EwAbsdiffS, schar> :
                         R == CV_8U  ? vecBinaryKernel<schar, uchar, v_int8,  short,   EwAbsdiff,  schar> : nullptr; break;
    case CV_16S:  fptr = R == CV_16S ? vecBinaryKernel<short, short, v_int16, int,     EwAbsdiffS, short> :
                         R == CV_16U ? vecBinaryKernel<short, ushort, v_int16, int,     EwAbsdiff,  short> : nullptr; break;
    case CV_32S:  fptr = R == CV_32S ? vecBinaryKernel<int,   int,   v_int32, int64_t, EwAbsdiffS, int> :
                         R == CV_32U ? vecBinaryKernel<int, unsigned, v_int32, int64_t, EwAbsdiff,  int> : nullptr; break;
    #if CV_SIMD_16F
    case CV_16F:  fptr = R == CV_16F ? vecBinaryKernel<hfloat, hfloat, v_float16, float, EwAbsdiff, hfloat> : nullptr; break;
    #else
    case CV_16F:  fptr = R == CV_16F ? vecBinaryKernel<hfloat, hfloat, v_float32, float, EwAbsdiff, float>  : nullptr; break;
    #endif
    case CV_16BF: fptr = R == CV_16BF ? vecBinaryKernel<bfloat, bfloat, v_float32, float, EwAbsdiff, float> : nullptr; break;
    case CV_32F:  fptr = R == CV_32F  ? vecBinaryKernel<float,  float,  v_float32, float, EwAbsdiff, float> : nullptr; break;
    case CV_64U:  fptr = R == CV_64U  ? scalarBinaryKernel<uint64_t, uint64_t, uint64_t, EwAbsdiff> : nullptr; break;
    case CV_64S:  fptr = R == CV_64U  ? scalarBinaryKernel<int64_t,  uint64_t, int64_t,  EwAbsdiff> : nullptr; break;
    #if CV_SIMD_64F
    case CV_64F:  fptr = R == CV_64F  ? vecBinaryKernel<double, double, v_float64, double, EwAbsdiff, double> : nullptr; break;
    #else
    case CV_64F:  fptr = R == CV_64F  ? scalarBinaryKernel<double, double, double, EwAbsdiff> : nullptr; break;
    #endif
    default:      ;
    }
    return {fptr, nullptr, 0};
}

// compare: T x T -> u8 mask. Directly-comparable depths (u8/s8/u16/s16/u32/s32/f32) take the SIMD
// vecCompareKernel; f16/bf16 and 64-bit depths fall back to scalarCompareKernel. The returned kernel
// defaults to a 255 mask in TKernel::flags (cv::compare-compatible); kernel.flags=1 gives a 0/1 mask.
template<class Cmp>
static KernelFunc compareByType(int T)
{
    switch (T)
    {
    case CV_8U:   return vecCompareKernel<uchar,    v_uint8,   v_uint8,  Cmp>;
    case CV_8S:   return vecCompareKernel<schar,    v_int8,    v_uint8,  Cmp>;
    case CV_16U:  return vecCompareKernel<ushort,   v_uint16,  v_uint16, Cmp>;
    case CV_16S:  return vecCompareKernel<short,    v_int16,   v_uint16, Cmp>;
    case CV_32U:  return vecCompareKernel<unsigned, v_uint32,  v_uint32, Cmp>;
    case CV_32S:  return vecCompareKernel<int,      v_int32,   v_uint32, Cmp>;
    case CV_32F:  return vecCompareKernel<float,    v_float32, v_uint32, Cmp>;
    #if CV_SIMD_16F
    case CV_16F:  return vecCompareKernel<hfloat,   v_float16, v_uint16, Cmp>;
    #else
    case CV_16F:  return vecCompareKernel<hfloat,   v_float32, v_uint32, Cmp>;   // widen f16->f32
    #endif
    case CV_16BF: return vecCompareKernel<bfloat,   v_float32, v_uint32, Cmp>;   // widen bf16->f32 (no native)
    case CV_64U:  return scalarCompareKernel<uint64_t, uint64_t, Cmp>;
    case CV_64S:  return scalarCompareKernel<int64_t,  int64_t,  Cmp>;
    #if CV_SIMD_64F
    case CV_64F:  return vecCompareKernel<double,   v_float64, v_uint64, Cmp>;
    #else
    case CV_64F:  return scalarCompareKernel<double,   double,   Cmp>;
    #endif
    default:      return nullptr;
    }
}

TKernel getCmpFunc_(TOp op, int T)
{
    // Only 4 physical kernels (eq/ne/gt/ge): LT/LE reuse GT/GE with the operands swapped
    // (a<b == b>a, a<=b == b>=a) via the EW_KERNEL_SWAP01 flag, honored by the executor.
    KernelFunc f = nullptr;
    int flags = 0;              // mask value: 0 flag bits => 0/255 (cv::compare); EW_KERNEL_MASK1 => 0/1
    switch (op)
    {
    case OP_CMP_EQ: f = compareByType<EwCmpEq>(T); break;
    case OP_CMP_NE: f = compareByType<EwCmpNe>(T); break;
    case OP_CMP_GT: f = compareByType<EwCmpGt>(T); break;
    case OP_CMP_GE: f = compareByType<EwCmpGe>(T); break;
    case OP_CMP_LT: f = compareByType<EwCmpGt>(T); flags = EW_KERNEL_SWAP01; break;
    case OP_CMP_LE: f = compareByType<EwCmpGe>(T); flags = EW_KERNEL_SWAP01; break;
    default:        ;
    }
    return {f, nullptr, flags};
}

// ===========================================================================
// OP_AND / OP_OR / OP_XOR / OP_NOT: bitwise, type-agnostic (by element size)
// ===========================================================================
// A bit-pattern op ignores the operand's semantic type, so we run it on the UNSIGNED integer whose
// width matches the element (1/2/4/8 bytes). One functor set (EwAnd/EwOr/EwXor) times four widths
// covers every depth; the dispatchers below pick by element size. AND/OR/XOR reuse vecBinaryKernel's
// native same-type path (as min/max do); 64-bit falls to the scalar kernel. NOT is unary.

// bitwise NOT: ~x. Single operand -> always a full contiguous array (no broadcast), so just a flat
// per-row complement. SIMD for 1/2/4-byte elements; 8-byte uses the scalar tail (Vvec unused there).
template<typename T, typename Vvec>
static int notKernel(const void* src0_, size_t s0y, size_t s0x,
                     const void*, size_t, size_t, const void*, size_t, size_t,
                     void* dst_, size_t dsty, int width, int height,
                     const double*, int, void*)
{
    s0y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(s0x == 1u || width == 1);
    const T* src0 = (const T*)src0_;
    T* dst = (T*)dst_;
    if (height > 1 && dsty == (size_t)width && s0y == (size_t)width) { width *= height; height = 1; }
    for (int y = 0; y < height; y++, src0 += s0y, dst += dsty)
    {
        int x = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
        if constexpr (sizeof(T) <= 4)
        {
            const int VECSZ = VTraits<Vvec>::vlanes();
            for (; x <= width - VECSZ; x += VECSZ)
                v_store(dst + x, v_not(vx_load(src0 + x)));
        }
    #endif
        for (; x < width; x++) dst[x] = (T)~src0[x];
    }
    return 0;
}

template<class Op>
static KernelFunc bitwiseByEsz(int esz)
{
    switch (esz)
    {
    case 1: return vecBinaryKernel<uint8_t,  uint8_t,  v_uint8,  uint8_t,  Op, uint8_t>;
    case 2: return vecBinaryKernel<uint16_t, uint16_t, v_uint16, uint16_t, Op, uint16_t>;
    case 4: return vecBinaryKernel<uint32_t, uint32_t, v_uint32, uint32_t, Op, uint32_t>;
    case 8: return scalarBinaryKernel<uint64_t, uint64_t, uint64_t, Op>;
    default: return nullptr;
    }
}

TKernel getBitwiseFunc_(TOp op, int esz)
{
    KernelFunc f = nullptr;
    switch (op)
    {
    case OP_AND: f = bitwiseByEsz<EwAnd>(esz); break;
    case OP_OR:  f = bitwiseByEsz<EwOr >(esz); break;
    case OP_XOR: f = bitwiseByEsz<EwXor>(esz); break;
    default:     ;
    }
    return {f, nullptr, 0};
}

TKernel getNotFunc_(int esz)
{
    KernelFunc f = nullptr;
    switch (esz)
    {
    case 1: f = notKernel<uint8_t,  v_uint8 >; break;
    case 2: f = notKernel<uint16_t, v_uint16>; break;
    case 4: f = notKernel<uint32_t, v_uint32>; break;
    case 8: f = notKernel<uint64_t, v_uint32>; break;   // SIMD path compiled out for 8-byte -> scalar ~
    default: ;
    }
    return {f, nullptr, 0};
}

// ===========================================================================
// OP_ADDW (addWeighted): dst = a*alpha + b*beta + gamma, params[0..2] = {alpha, beta, gamma}. Two fused
// v_fma in the work type Wvec - f32 SIMD for u8/s8/u16/s16/f16/bf16/f32; the 32-bit-int/64-bit group
// works in f64 (v_float64 SIMD under CV_SIMD_64F, else use_simd=false scalar). Like vecBinaryKernel but
// WITHOUT its multi-channel short-row
// branch: addWeighted takes plain scalar coefficients (a multi-channel scalar is not optimized, matching
// the classic function). The broadcast branches fold the constant operand's contribution once.
// ===========================================================================
template<typename T, typename Tr, typename Wvec, typename WT, bool use_simd>
static int addWeightedKernel(const void* src0_, size_t s0y, size_t s0x,
                             const void* src1_, size_t s1y, size_t s1x,
                             const void*, size_t, size_t,
                             void* dst_, size_t dsty, int width, int height,
                             const double* params, int, void*)
{
    s0y /= sizeof(T); s1y /= sizeof(T); dsty /= sizeof(Tr);
    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);
    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    Tr* dst = (Tr*)dst_;
    const WT alpha = (WT)params[0], beta = (WT)params[1], gamma = (WT)params[2];
    EW_TRY_COLLAPSE(2);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    Wvec va{}, vb{}, vg{};
    const int VECSZ = VTraits<Wvec>::vlanes();
    const bool tail = width >= VECSZ*4 && src0_ != dst_ && src1_ != dst_;
    if constexpr (use_simd) {
        WT fa=alpha, fb=beta, fg=gamma;
        vx_setall_as(&fa, va); vx_setall_as(&fb, vb); vx_setall_as(&fg, vg);
    }
#endif
    for (int y = 0; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
        if constexpr (use_simd)
        {
            Wvec a0,a1,a2,a3,b0,b1,b2,b3;
            if (s0x == s1x) {                                // both arrays contiguous
                for (; x < width; x += VECSZ*4) {
                    if (x+VECSZ*4 > width) { if (!tail) break; x = width-VECSZ*4; }
                    vx_load_pair_as(src0+x, a0, a1); vx_load_pair_as(src0+x+VECSZ*2, a2, a3);
                    vx_load_pair_as(src1+x, b0, b1); vx_load_pair_as(src1+x+VECSZ*2, b2, b3);
                    a0=v_fma(a0,va,v_fma(b0,vb,vg)); a1=v_fma(a1,va,v_fma(b1,vb,vg));
                    a2=v_fma(a2,va,v_fma(b2,vb,vg)); a3=v_fma(a3,va,v_fma(b3,vb,vg));
                    v_store_pair_as(dst+x, a0, a1); v_store_pair_as(dst+x+VECSZ*2, a2, a3);
                }
            }
            else if (s1x == 0) {                             // src1 broadcast: b*beta+gamma is constant
                Wvec bb; vx_setall_as(src1, bb); Wvec bc = v_fma(bb, vb, vg);
                for (; x < width; x += VECSZ*4) {
                    if (x+VECSZ*4 > width) { if (!tail) break; x = width-VECSZ*4; }
                    vx_load_pair_as(src0+x, a0, a1); vx_load_pair_as(src0+x+VECSZ*2, a2, a3);
                    a0=v_fma(a0,va,bc); a1=v_fma(a1,va,bc); a2=v_fma(a2,va,bc); a3=v_fma(a3,va,bc);
                    v_store_pair_as(dst+x, a0, a1); v_store_pair_as(dst+x+VECSZ*2, a2, a3);
                }
            }
            else {                                           // src0 broadcast: a*alpha+gamma is constant
                Wvec aa; vx_setall_as(src0, aa); Wvec acg = v_fma(aa, va, vg);
                for (; x < width; x += VECSZ*4) {
                    if (x+VECSZ*4 > width) { if (!tail) break; x = width-VECSZ*4; }
                    vx_load_pair_as(src1+x, b0, b1); vx_load_pair_as(src1+x+VECSZ*2, b2, b3);
                    b0=v_fma(b0,vb,acg); b1=v_fma(b1,vb,acg); b2=v_fma(b2,vb,acg); b3=v_fma(b3,vb,acg);
                    v_store_pair_as(dst+x, b0, b1); v_store_pair_as(dst+x+VECSZ*2, b2, b3);
                }
            }
        }
    #endif
        if (s0x == s1x) {
            for (; x < width; x++) dst[x] = narrowSaturate<Tr>((WT)src0[x]*alpha + (WT)src1[x]*beta + gamma);
        } else if (s0x == 0) {
            const WT ac = (WT)src0[0]*alpha + gamma;
            for (; x < width; x++) dst[x] = narrowSaturate<Tr>((WT)src1[x]*beta + ac);
        } else {
            const WT bc = (WT)src1[0]*beta + gamma;
            for (; x < width; x++) dst[x] = narrowSaturate<Tr>((WT)src0[x]*alpha + bc);
        }
    }
    return 0;
}

TKernel getAddWeightedFunc_(int T, int R)
{
    KernelFunc f = nullptr;
    #define AWS(Tt, Rr) addWeightedKernel<Tt, Rr, v_float32, float, true>
#if CV_SIMD_16F
    #define AWH(Tt)     addWeightedKernel<Tt, Tt, v_float16, float, true>            // u8/s8 -> same, f16 work
#else
    #define AWH(Tt)     AWS(Tt, Tt)                                                  // no f16: f32 work
#endif
#if CV_SIMD_64F
    #define AWD(Tt, Rr) addWeightedKernel<Tt, Rr, v_float64, double, true>       // f64 SIMD
#else
    #define AWD(Tt, Rr) addWeightedKernel<Tt, Rr, v_float32, double, false>      // scalar (v_float32 unused)
#endif
    switch (T)
    {
    case CV_8U:  f = R==CV_8U  ? AWH(uint8_t)            : R==CV_32F ? AWS(uint8_t,  float) : nullptr; break;
    case CV_8S:  f = R==CV_8S  ? AWH(int8_t)             : R==CV_32F ? AWS(int8_t,   float) : nullptr; break;
    case CV_16U: f = R==CV_16U ? AWS(uint16_t, uint16_t) : R==CV_32F ? AWS(uint16_t, float) : nullptr; break;
    case CV_16S: f = R==CV_16S ? AWS(int16_t,  int16_t)  : R==CV_32F ? AWS(int16_t,  float) : nullptr; break;
    case CV_16F: f = R==CV_16F ? AWS(hfloat,   hfloat)   : R==CV_32F ? AWS(hfloat,   float) : nullptr; break;
    case CV_16BF:f = R==CV_32F ? AWS(bfloat,   float) : nullptr; break;
    case CV_32F: f = R==CV_32F ? AWS(float,    float) : nullptr; break;
    case CV_32U: f = R==CV_32U ? AWD(unsigned, unsigned) : R==CV_64F ? AWD(unsigned, double) : nullptr; break;
    case CV_32S: f = R==CV_32S ? AWD(int,      int)      : R==CV_64F ? AWD(int,      double) : nullptr; break;
    case CV_64U: f = R==CV_64F ? AWD(uint64_t, double) : nullptr; break;
    case CV_64S: f = R==CV_64F ? AWD(int64_t,  double) : nullptr; break;
    case CV_64F: f = R==CV_64F ? AWD(double,   double) : nullptr; break;
    default: ;
    }
    #undef AWS
    #undef AWH
    #undef AWD
    return {f, nullptr, 0};
}

// ===========================================================================
// OP_SELECT: dst = (mask != 0) ? a : b
// ===========================================================================
// The one masking primitive of the engine. It serves both the public texpr select() and the
// masked-op tail: `cv::add(..., mask)` computes the full result into a temp `r`, then a final
// select(mask, r, dst) -> dst lands the masked subset in the (pre-existing) output and PRESERVES
// the rest - dst rides as both an input and the result. That aliasing is safe even under the
// right-edge tail backoff: re-running select over already-blended elements is IDEMPOTENT
// (mask!=0 lanes stay a, mask==0 lanes stay b/dst). Only dst == mask would break (the store
// rewrites the mask before the backoff re-reads it) - that combination falls to the scalar tail.
//
// The mask is one byte per element (bool/u8/s8 - never parameterized by its depth, we just test
// the byte != 0); a/b/dst share one depth, the kernel is templated by the element SIZE only.
// Channels fall out of the broadcast machinery: single-channel data arrives as a (width x 1) tile
// with a per-element mask (smx == 1); n-channel data as a tall-thin tile - channel axis = width,
// the 1-channel mask broadcasting across it (smx == 0, one mask byte per row) - handled by the
// interleaved fast path for 2..4 channels, per-row otherwise. Branches may broadcast (s1x/s2x == 0).
#if (CV_SIMD || CV_SIMD_SCALABLE)
// VECSZ(T) mask bytes -> T-width lanes (u8 direct, u16/u32 via expand)
static inline v_uint8  loadSelectMask(const uchar* m, const v_uint8&)  { return vx_load(m); }
static inline v_uint16 loadSelectMask(const uchar* m, const v_uint16&) { return vx_load_expand(m); }
static inline v_uint32 loadSelectMask(const uchar* m, const v_uint32&) { return vx_load_expand_q(m); }

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

    int y = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if constexpr (sizeof(T) <= 4)
    {
        const int VECSZ = VTraits<Tvec>::vlanes();
        const Tvec z = v_setzero_<Tvec>();

        // n-channel data under a per-pixel mask (the masked-op shape): channel axis = width (2..4),
        // one mask byte per row. Process VECSZ rows per iteration - expand the mask once and
        // interleave it across the channel lanes.
        if (height > VECSZ && 2 <= width && width <= 4 && smx == 0u && smy == 1u &&
            s1x == 1u && s1y == (size_t)width && s2x == 1u && s2y == (size_t)width &&
            dsty == (size_t)width)
        {
            constexpr int MAXVECSZ = VTraits<Tvec>::max_nlanes;
            T maskbuf[MAXVECSZ*4] = {};      // {} for -Wmaybe-uninitialized only (fully written)
            const int dy = VECSZ;
            for (; y + dy <= height; y += dy, mask += dy, src1 += width*dy, src2 += width*dy,
                                                          dst += width*dy)
            {
                Tvec m0 = v_ne(loadSelectMask(mask, z), z), m1, m2, m3;
                if (width == 2)      v_store_interleave(maskbuf, m0, m0);
                else if (width == 3) v_store_interleave(maskbuf, m0, m0, m0);
                else                 v_store_interleave(maskbuf, m0, m0, m0, m0);
                m0 = vx_load(maskbuf);
                m1 = vx_load(maskbuf + VECSZ);
                if (width > 2) m2 = vx_load(maskbuf + VECSZ*2);
                if (width > 3) m3 = vx_load(maskbuf + VECSZ*3);
                v_store(dst, v_select(m0, vx_load(src1), vx_load(src2)));
                v_store(dst + VECSZ, v_select(m1, vx_load(src1 + VECSZ), vx_load(src2 + VECSZ)));
                if (width > 2)
                    v_store(dst + VECSZ*2,
                            v_select(m2, vx_load(src1 + VECSZ*2), vx_load(src2 + VECSZ*2)));
                if (width > 3)
                    v_store(dst + VECSZ*3,
                            v_select(m3, vx_load(src1 + VECSZ*3), vx_load(src2 + VECSZ*3)));
            }
            // the remaining < VECSZ rows fall through to the per-row path below
        }
        else if (smx == 1)                                   // per-element mask - the common case
        {
            // a branch aliasing dst is fine under the backoff (idempotent select); dst == mask is not
            const bool use_tail_trick = width >= VECSZ*2 && dst_ != mask_;
            Tvec a, b;
            if (s1x == 0) setallSelect(src1, a);
            if (s2x == 0) setallSelect(src2, b);
            for (; y < height; y++, mask += smy, src1 += s1y, src2 += s2y, dst += dsty)
            {
                int x = 0;
                for (; x < width; x += VECSZ)
                {
                    if (x + VECSZ > width) { if (!use_tail_trick || x == 0) break; x = width - VECSZ; }
                    Tvec m = v_ne(loadSelectMask(mask + x, z), z);
                    if (s1x) a = vx_load(src1 + x);
                    if (s2x) b = vx_load(src2 + x);
                    v_store(dst + x, v_select(m, a, b));
                }
                for (; x < width; x++)
                    dst[x] = mask[x] != 0 ? src1[x*s1x] : src2[x*s2x];
            }
            return 0;
        }
    }
#endif
    for (; y < height; y++, mask += smy, src1 += s1y, src2 += s2y, dst += dsty)
    {
        if (smx == 0)                     // one mask byte per row (n-channel data / broadcast mask)
        {
            const T* s = mask[0] != 0 ? src1 : src2;
            const size_t sx = mask[0] != 0 ? s1x : s2x;
            if ((const void*)s != (const void*)dst)      // row select from dst itself is a no-op
                for (int x = 0; x < width; x++) dst[x] = s[x*sx];
        }
        else
            for (int x = 0; x < width; x++)
                dst[x] = mask[x] != 0 ? src1[x*s1x] : src2[x*s2x];
    }
    return 0;
}

TKernel getSelectFunc_(int mdepth, int T)
{
    if (CV_ELEM_SIZE1(mdepth) != 1)      // the mask must be a 1-byte type (u8/s8/bool)
        return {};
    KernelFunc fptr = nullptr;
    switch (CV_ELEM_SIZE1(T))
    {
    case 1: fptr = selectKernel<uint8_t,  v_uint8 >; break;
    case 2: fptr = selectKernel<uint16_t, v_uint16>; break;
    case 4: fptr = selectKernel<uint32_t, v_uint32>; break;
    case 8: fptr = selectKernel<uint64_t, v_uint32>; break;   // SIMD path compiled out -> scalar
    default: ;
    }
    return {fptr, nullptr, 0};
}

// ===========================================================================
// OP_CLAMP: dst = min(max(x, lo), hi)
// ===========================================================================
// All four operands share one depth (emitTernary unifies them); lo/hi are usually broadcast
// scalars (clamp(img, 10, 200) - literals ride as 0-dim consts with stepx == 0) but may be full
// arrays. clamp is IDEMPOTENT, so the right-edge tail backoff stays on when dst aliases x; only
// dst aliasing lo/hi suppresses it (the store would corrupt the bounds before the re-read).
// NaN note: v_min/v_max lane behavior on NaN is ISA-specific, matching the scalar std::min/max
// unspecifiedness - clamp of NaN is not a contract either way.
#if (CV_SIMD || CV_SIMD_SCALABLE)
static inline void setallClamp(const uchar*  p, v_uint8&   a) { a = vx_setall_u8(*p); }
static inline void setallClamp(const schar*  p, v_int8&    a) { a = vx_setall_s8(*p); }
static inline void setallClamp(const ushort* p, v_uint16&  a) { a = vx_setall_u16(*p); }
static inline void setallClamp(const short*  p, v_int16&   a) { a = vx_setall_s16(*p); }
static inline void setallClamp(const unsigned* p, v_uint32& a) { a = vx_setall_u32(*p); }
static inline void setallClamp(const int*    p, v_int32&   a) { a = vx_setall_s32(*p); }
static inline void setallClamp(const float*  p, v_float32& a) { a = vx_setall_f32(*p); }
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
static inline void setallClamp(const double* p, v_float64& a) { a = vx_setall_f64(*p); }
#endif
#endif

template<typename T, typename Tvec>
static int clampKernel(const void* src0_, size_t s0y, size_t s0x,
                       const void* lo_, size_t s1y, size_t s1x,
                       const void* hi_, size_t s2y, size_t s2x,
                       void* dst_, size_t dsty, int width, int height,
                       const double*, int, void*)
{
    s0y /= sizeof(T);
    s1y /= sizeof(T);
    s2y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(s0x <= 1u && s1x <= 1u && s2x <= 1u);

    const T* src0 = (const T*)src0_;
    const T* lo = (const T*)lo_;
    const T* hi = (const T*)hi_;
    T* dst = (T*)dst_;

    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width &&
        s1y == s1x*(size_t)width && s2y == s2x*(size_t)width)
    { width *= height; height = 1; }

    for (int y = 0; y < height; y++, src0 += s0y, lo += s1y, hi += s2y, dst += dsty)
    {
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (s0x == 1)
        {
            const int VECSZ = VTraits<Tvec>::vlanes();
            const bool use_tail_trick = width >= VECSZ*2 && lo_ != dst_ && hi_ != dst_;
            Tvec vlo, vhi;
            if (s1x == 0) setallClamp(lo, vlo);
            if (s2x == 0) setallClamp(hi, vhi);
            for (; x < width; x += VECSZ)
            {
                if (x + VECSZ > width) { if (!use_tail_trick || x == 0) break; x = width - VECSZ; }
                if (s1x) vlo = vx_load(lo + x);
                if (s2x) vhi = vx_load(hi + x);
                v_store(dst + x, v_min(v_max(vx_load(src0 + x), vlo), vhi));
            }
        }
#endif
        for (; x < width; x++)
        {
            T v = src0[x*s0x], l = lo[x*s1x], h = hi[x*s2x];
            dst[x] = std::min(std::max(v, l), h);
        }
    }
    return 0;
}

// Scalar-only clamp for the depths without a native SIMD lane type here (f16/bf16 - compared in
// float; 64-bit ints).
template<typename T, typename WT>
static int scalarClampKernel(const void* src0_, size_t s0y, size_t s0x,
                             const void* lo_, size_t s1y, size_t s1x,
                             const void* hi_, size_t s2y, size_t s2x,
                             void* dst_, size_t dsty, int width, int height,
                             const double*, int, void*)
{
    s0y /= sizeof(T); s1y /= sizeof(T); s2y /= sizeof(T); dsty /= sizeof(T);
    CV_Assert(s0x <= 1u && s1x <= 1u && s2x <= 1u);
    const T* src0 = (const T*)src0_;
    const T* lo = (const T*)lo_;
    const T* hi = (const T*)hi_;
    T* dst = (T*)dst_;
    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width &&
        s1y == s1x*(size_t)width && s2y == s2x*(size_t)width)
    { width *= height; height = 1; }
    for (int y = 0; y < height; y++, src0 += s0y, lo += s1y, hi += s2y, dst += dsty)
        for (int x = 0; x < width; x++)
        {
            WT v = (WT)src0[x*s0x], l = (WT)lo[x*s1x], h = (WT)hi[x*s2x];
            dst[x] = saturate_cast<T>(std::min(std::max(v, l), h));
        }
    return 0;
}

TKernel getClampFunc_(int T)
{
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_8U:   fptr = clampKernel<uchar,    v_uint8  >; break;
    case CV_8S:   fptr = clampKernel<schar,    v_int8   >; break;
    case CV_16U:  fptr = clampKernel<ushort,   v_uint16 >; break;
    case CV_16S:  fptr = clampKernel<short,    v_int16  >; break;
    case CV_32U:  fptr = clampKernel<unsigned, v_uint32 >; break;
    case CV_32S:  fptr = clampKernel<int,      v_int32  >; break;
    case CV_32F:  fptr = clampKernel<float,    v_float32>; break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    case CV_64F:  fptr = clampKernel<double,   v_float64>; break;
#else
    case CV_64F:  fptr = scalarClampKernel<double, double>; break;
#endif
    case CV_16F:  fptr = scalarClampKernel<hfloat, float>; break;
    case CV_16BF: fptr = scalarClampKernel<bfloat, float>; break;
    case CV_64U:  fptr = scalarClampKernel<uint64_t, uint64_t>; break;
    case CV_64S:  fptr = scalarClampKernel<int64_t,  int64_t >; break;
    default: ;
    }
    return {fptr, nullptr, 0};
}

static int expandKernel(size_t s0y, size_t s0x,
                        void* dst_, size_t dsty, int width, int height,
                        int esz1)
{
    uchar* dst = (uchar*)dst_;
    if (s0x == 0u && width > 1) {
        int h = s0y > 0u ? height : 1;
        CV_Assert(esz1 == 1 || esz1 == 2 || esz1 == 4 || esz1 == 8);
        for (int y = 0; y < h; y++) {
            if (esz1 == 1) {
                uchar* rowptr = dst + dsty*y;
                uchar val = rowptr[0];
                for (int x = 1; x < width; x++) rowptr[x] = val;
            }
            else if (esz1 == 2) {
                ushort* rowptr = (ushort*)(dst + dsty*y);
                ushort val = rowptr[0];
                for (int x = 1; x < width; x++) rowptr[x] = val;
            }
            else if (esz1 == 4) {
                unsigned* rowptr = (unsigned*)(dst + dsty*y);
                unsigned val = rowptr[0];
                for (int x = 1; x < width; x++) rowptr[x] = val;
            }
            else {
                uint64_t* rowptr = (uint64_t*)(dst + dsty*y);
                uint64_t val = rowptr[0];
                for (int x = 1; x < width; x++) rowptr[x] = val;
            }
        }
    }

    if (s0y == 0) {
        for (int y = 1; y < height; y++)
            memcpy(dst + dsty*y, dst, (size_t)esz1*width);
    }
    return 0;
}

static int castKernel(const void* src0_, size_t s0y, size_t s0x,
                      const void*, size_t, size_t,
                      const void*, size_t, size_t,
                      void* dst_, size_t dsty, int width, int height,
                      const double* params, int esz1, void* userdata)
{
    BinaryFunc castfunc = (BinaryFunc)userdata;
    castfunc((const uchar*)src0_, s0y, nullptr, 0, (uchar*)dst_, dsty,
             Size((s0x > 0u ? width : 1), (s0y > 0u ? height : 1)), (void*)params);
    return expandKernel(s0y, s0x, dst_, dsty, width, height, esz1);
}

// ===========================================================================
// Per-op entry points (this baseline). The op-level getElemwiseFunc() dispatcher and the regular
// get*Func() forwarders live in arithm.dispatch.cpp.
// ===========================================================================

// OP_CAST / OP_CONVERT_SCALE: wrap core's convert BinaryFunc (carried in kernel.userdata) in
// castKernel, which runs it over the distinct sub-region and then expands across broadcast axes.
TKernel getCastFunc_(int sdepth, int ddepth, bool scaled)
{
    BinaryFunc cvt = scaled ? getConvertScaleFunc(sdepth, ddepth) : getConvertFunc(sdepth, ddepth);
    return {castKernel, (void*)cvt, CV_ELEM_SIZE1(ddepth)};
}

// Non-template wrappers around the templated kernel selectors (ADD/SUB share getAddSubFunc, MIN/MAX
// share getMinMaxFunc). min/max are T x T -> T, so R is ignored (the dispatcher validates R == T).
TKernel getAddFunc_(int T, int R) { return getAddSubFunc<EwAdd>(T, R); }
TKernel getSubFunc_(int T, int R) { return getAddSubFunc<EwSub>(T, R); }
TKernel getMinFunc_(int T, int R) { (void)R; return getMinMaxFunc<EwMin>(T); }
TKernel getMaxFunc_(int T, int R) { (void)R; return getMinMaxFunc<EwMax>(T); }

TKernel getHypotFunc_(int T, int R)
{
    if (R != T)
        return {};
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_16F:  fptr = vecBinaryKernel<hfloat, hfloat, v_float32, float, EwHypot>; break;
    case CV_16BF: fptr = vecBinaryKernel<bfloat, bfloat, v_float32, float, EwHypot>; break;
    case CV_32F:  fptr = vecBinaryKernel<float,  float,  v_float32, float, EwHypot>; break;
#if CV_SIMD_64F || CV_SIMD_SCALABLE_64F
    case CV_64F:  fptr = vecBinaryKernel<double, double, v_float64, double, EwHypot>; break;
#else
    case CV_64F:  fptr = scalarBinaryKernel<double, double, double, EwHypot>; break;
#endif
    default: ;
    }
    return {fptr, nullptr, 0};
}

TKernel getAtan2Func_(int T, int R)
{
    if (R != T)
        return {};
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_16F:  fptr = vecBinaryKernel<hfloat, hfloat, v_float32, float, EwAtan2>; break;
    case CV_16BF: fptr = vecBinaryKernel<bfloat, bfloat, v_float32, float, EwAtan2>; break;
    case CV_32F:  fptr = vecBinaryKernel<float,  float,  v_float32, float, EwAtan2>; break;
    case CV_64F:  fptr = scalarBinaryKernel<double, double, double, EwAtan2>; break;   // exact std::atan2
    default: ;
    }
    return {fptr, nullptr, 0};
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::ew
