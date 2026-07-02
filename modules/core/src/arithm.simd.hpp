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

namespace cv {

// BinaryFunc and getConvertFunc / getConvertScaleFunc come from core (precomp.hpp / convert.hpp).

#if CV_SIMD128_FP16
#undef CV_SIMD_16F
#define CV_SIMD_16F 1
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
#if CV_SIMD_64F
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
    v_uint16 w = v_load_expand(p);
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
    v_store(p + VTraits<v_float32>::nlanes, v_cvt_f32(b));
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

namespace ew {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// ---- per-op kernel entry points for THIS baseline (the regular dispatchers in
//      arithm.dispatch.cpp reach them through CV_CPU_DISPATCH). ----
TKernel getAddFunc_(int T, int R);
TKernel getSubFunc_(int T, int R);
TKernel getMulFunc_(int T, int R);
TKernel getDivFunc_(int T, int R, bool checked);
TKernel getPowFunc_(int T, int R);
TKernel getMinFunc_(int T, int R);
TKernel getMaxFunc_(int T, int R);
TKernel getAbsdiffFunc_(int T, int R);
TKernel getCmpFunc_(TOp op, int T);
TKernel getBitwiseFunc_(TOp op, int esz);                    // OP_AND / OP_OR / OP_XOR, by element size
TKernel getNotFunc_(int esz);                                // OP_NOT, by element size
TKernel getAddWeightedFunc_(int T, int R);                   // OP_ADDW, a*alpha+b*beta+gamma (T x T -> R)
TKernel getCopyMaskFunc_(int depth);
TKernel getCastFunc_(int sdepth, int ddepth, bool scaled);   // OP_CAST / OP_CONVERT_SCALE

// ===========================================================================
// Op functors (vector + scalar). New binary ops slot in here.
// ===========================================================================
struct EwAdd {
    template<typename V> static V vec(const V& a, const V& b) { return v_add(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    // Accumulate in the promoted type, NOT in W: for the native saturating path W is the narrow
    // lane type (schar/short/...), and (W)(a+b) would wrap in 8/16 bits before saturate_cast<Tr>
    // could clamp. Letting a+b promote (narrow -> int) keeps saturation for 8/16-bit outputs and
    // the natural wrap for 32/64-bit (both matching cv::add). The SIMD path already saturates.
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a + b); }
};

struct EwSub {
    template<typename V> static V vec(const V& a, const V& b) { return v_sub(a, b); }
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
    template<typename V> static V vec(const V& a, const V& b) { return v_mul(a, b); }
    static v_uint16 vec(const v_uint16& a, const v_uint16& b) { return v_mul_wrap(a, b); }
    static v_int16 vec(const v_int16& a, const v_int16& b) { return v_mul_wrap(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V& s) { return v_mul(a, s); }
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return a * b * s; }
};
// div has two variants by the COMMON INPUT type (matching cv::'s per-type kernel choice): integer
// inputs guard divide-by-zero -> 0 (cv:: iscalar_div); float inputs do NOT guard (cv:: fscalar_div,
// a/0 -> inf), which then saturates on the cast to an integer output exactly like cv::divide.
struct EwDivInt {
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
    template<typename V> static V vec(const V& a, const V& b) { return v_div(a, b); }
    template<typename V> static V vec(const V& a, const V& b, const V& s) { return v_div(v_mul(a, s), b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }   // identity: scale is in vec
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return a * s / b; }
};

struct EwPow {
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::pow(a, b); }
};

// min / max / absdiff: T x T -> T (same depth in and out, no scale). v_min/v_max exist for every
// vector lane type (64-bit ints fall back to scalar). absdiff uses v_absdiff (defined for the
// UNSIGNED and float lane types - signed/wide depths go through the scalar path), and the scalar
// |a-b| is computed branch-wise so it never underflows an unsigned work type.
struct EwMin {
    template<typename V> static V vec(const V& a, const V& b) { return v_min(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::min(a, b); }
};
struct EwMax {
    template<typename V> static V vec(const V& a, const V& b) { return v_max(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::max(a, b); }
};
struct EwAbsdiff {
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
    template<typename V> static V vec(const V& a, const V& b) { return v_and(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a & b); }
};
struct EwOr {
    template<typename V> static V vec(const V& a, const V& b) { return v_or(a, b); }
    template<typename V, typename S> static V vec(const V& a, const V& b, const S&) { return vec(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a | b); }
};
struct EwXor {
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
                dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x], (WT)src1[x], scalar));
        }
        else if (s0x == 0) {
            WT sc0 = (WT)src0[0];
            for (int x = 0; x < width; x++)
                dst[x] = saturate_cast<Tr>(Op::scl(sc0, (WT)src1[x], scalar));
        }
        else {
            WT sc1 = (WT)src1[0];
            for (int x = 0; x < width; x++)
                dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x], sc1, scalar));
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
    if constexpr (sizeof(T) == sizeof(typename VTraits<Vvec>::lane_type))
    if (sizeof(T) <= 4 && height > 1 && width <= 4 &&
        ((s0y == 0 && s1y == (size_t)width*s1x) || (s1y == 0 && s0y == (size_t)width*s0x)) &&
        dsty == (size_t)width)
    {
        const int VECSZ  = VTraits<Vvec>::vlanes();          // sizeof(T)-type lanes
        const int VECSZ8 = VTraits<v_uint8>::vlanes();       // u8 output lanes
        constexpr int MAXV8 = VTraits<v_uint8>::max_nlanes;
        T scbuf[MAXV8 * 3];                                  // interleaved threshold (elements)
        uchar mbuf[MAXV8 * 3], vbuf[MAXV8 * 3];              // interleaved mask / value (bytes)
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
        Wlane scbuf[MAXVECSZ*6];
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
            if (scalar == ST(1)) {
                for (; x < width; x += VECSZ*4) {
                    Wvec1 a0_, a1_, a2_, a3_, b0_, b1_, b2_, b3_;
                    if (x + VECSZ*4 > width) { if (!use_tail_trick) break; x = width - VECSZ*4; }
                    vx_load_pair_as(src0 + x, a0_, a1_);
                    vx_load_pair_as(src0 + x + VECSZ*2, a2_, a3_);
                    vx_load_pair_as(src1 + x, b0_, b1_);
                    vx_load_pair_as(src1 + x + VECSZ*2, b2_, b3_);
                    auto c0 = Op::vec(a0_, b0_), c1 = Op::vec(a1_, b1_),
                         c2 = Op::vec(a2_, b2_), c3 = Op::vec(a3_, b3_);
                    v_store_pair_as(dst + x, c0, c1);
                    v_store_pair_as(dst + x + VECSZ*2, c2, c3);
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
            dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x*s0x], (WT)src1[x*s1x], scalar));
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
//  - native saturating path (Wvec = native, use_simd=true) for T -> T on <=32-bit ints + f32;
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
        fptr = R == CV_32U ? scalarBinaryKernel<unsigned, unsigned, int64_t, Op> :
                R == CV_64S ? scalarBinaryKernel<unsigned, int64_t, int64_t, Op> :
                R == CV_64F ? scalarBinaryKernel<unsigned, double, int64_t, Op> : nullptr;
        break;
    case CV_32S:
        fptr = R == CV_32S ? scalarBinaryKernel<int, int, int64_t, Op> :
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
        fptr =
        #if CV_SIMD_16F
            R == CV_8U ? vecBinaryKernel<uchar, uchar, v_float16, float, EwMul, float, v_uint16> :
        #else
            R == CV_8U ? vecBinaryKernel<uchar, uchar, v_float32, float, EwMul> :
        #endif
            R == CV_32F ? vecBinaryKernel<uchar, float, v_float32, float, EwMul> : nullptr;
        break;
    case CV_8S:
        fptr =
        #if CV_SIMD_16F
            R == CV_8S ? vecBinaryKernel<schar, schar, v_float16, float, EwMul, float, v_int16> :
        #else
            R == CV_8S ? vecBinaryKernel<schar, schar, v_float32, float, EwMul> :
        #endif
            R == CV_32F ? vecBinaryKernel<schar, float, v_float32, float, EwMul> : nullptr;
        break;
    case CV_16U:
        fptr =
            R == CV_16U ? vecBinaryKernel<ushort, ushort, v_float32, float, EwMul> :
            R == CV_32F ? vecBinaryKernel<ushort, float,  v_float32, float, EwMul> : nullptr;
        break;
    case CV_16S:
        fptr =
            R == CV_16S ? vecBinaryKernel<short, short, v_float32, float, EwMul> :
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
        fptr =
        #if CV_SIMD_64F
            R == CV_32U ? vecBinaryKernel<unsigned, unsigned, v_float64, double, EwMul> :
            R == CV_64F ? vecBinaryKernel<unsigned, double,   v_float64, double, EwMul> : nullptr;
        #else
            R == CV_32U ? scalarBinaryKernel<unsigned, unsigned, double, EwMul> :
            R == CV_64F ? scalarBinaryKernel<unsigned, double,   double, EwMul> : nullptr;
        #endif
        break;
    case CV_32S:
        fptr =
        #if CV_SIMD_64F
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

TKernel getPowFunc_(int T, int R)
{
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_32F:
        fptr = R == CV_32F ? scalarBinaryKernel<float, float, float, EwPow> : nullptr;
        break;
    case CV_64F:
        fptr = R == CV_64F ? scalarBinaryKernel<double, double, double, EwPow> : nullptr;
        break;
    default:
        ;
    }
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

// |a-b| of a value of depth T: unsigned/float -> T, SIGNED integer -> the UNSIGNED type of the same
// width (8s->8u, ...) so the full 0..2^width-1 range fits without saturation (mirrors absdiffResultDepth).
static int absdiffOutDepth(int T)
{
    switch (T)
    {
    case CV_8S:  return CV_8U;
    case CV_16S: return CV_16U;
    case CV_32S: return CV_32U;
    case CV_64S: return CV_64U;
    default:     return T;
    }
}

// absdiff: |a-b|, T x T -> absdiffOutDepth(T). v_absdiff is defined for the UNSIGNED and float lane
// types only (signed v_absdiff would change the lane type), so u8/u16/f16/bf16/f32 take the SIMD path
// and the signed / wide / f64 depths use the scalar kernel (|a-b| in a SIGN-CORRECT work type, stored
// to the unsigned result type for signed inputs). `rdepth` must equal absdiffOutDepth(T).
TKernel getAbsdiffFunc_(int T, int rdepth)
{
    if (rdepth != absdiffOutDepth(T)) return {};
    KernelFunc fptr = nullptr;
    switch (T)
    {
    case CV_8U:   fptr = vecBinaryKernel<uchar,  uchar,  v_uint8,  short, EwAbsdiff, uchar>; break;
    case CV_16U:  fptr = vecBinaryKernel<ushort, ushort, v_uint16, int,   EwAbsdiff, ushort>; break;
    #if CV_SIMD_16F
    case CV_16F:  fptr = vecBinaryKernel<hfloat, hfloat, v_float16, float, EwAbsdiff, hfloat>; break;
    #else
    case CV_16F:  fptr = vecBinaryKernel<hfloat, hfloat, v_float32, float, EwAbsdiff, float>; break;
    #endif
    case CV_16BF: fptr = vecBinaryKernel<bfloat, bfloat, v_float32, float, EwAbsdiff, float>; break;
    case CV_32F:  fptr = vecBinaryKernel<float,  float,  v_float32, float, EwAbsdiff, float>; break;
    // signed: NATIVE work (v_int8/16/32) - v_absdiff returns the matching unsigned result, stored via the
    // honest unsigned v_store_pair_as (vec's own result type). No widening -> full lane count.
    case CV_8S:   fptr = vecBinaryKernel<schar,    uchar,    v_int8,   short,   EwAbsdiff, schar>; break;
    case CV_16S:  fptr = vecBinaryKernel<short,    ushort,   v_int16,  int,     EwAbsdiff, short>; break;
    case CV_32U:  fptr = vecBinaryKernel<unsigned, unsigned, v_uint32, int64_t, EwAbsdiff, unsigned>; break;
    case CV_32S:  fptr = vecBinaryKernel<int,      unsigned, v_int32,  int64_t, EwAbsdiff, int>; break;
    case CV_64U:  fptr = scalarBinaryKernel<uint64_t, uint64_t, uint64_t, EwAbsdiff>; break;   // no 64-bit v_absdiff
    case CV_64S:  fptr = scalarBinaryKernel<int64_t,  uint64_t, int64_t,  EwAbsdiff>; break;
    #if CV_SIMD_64F
    case CV_64F:  fptr = vecBinaryKernel<double,   double,   v_float64, double, EwAbsdiff, double>; break;
    #else
    case CV_64F:  fptr = scalarBinaryKernel<double,   double,   double,   EwAbsdiff>; break;
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
            for (; x < width; x++) dst[x] = saturate_cast<Tr>((WT)src0[x]*alpha + (WT)src1[x]*beta + gamma);
        } else if (s0x == 0) {
            const WT ac = (WT)src0[0]*alpha + gamma;
            for (; x < width; x++) dst[x] = saturate_cast<Tr>((WT)src1[x]*beta + ac);
        } else {
            const WT bc = (WT)src1[0]*beta + gamma;
            for (; x < width; x++) dst[x] = saturate_cast<Tr>((WT)src0[x]*alpha + bc);
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
// OP_COPY_MASK: dst = (mask != 0) ? src : dst   (unmasked elements are PRESERVED)
// ===========================================================================
// The masked tail of an op: the op computes its full result into a temp, then copyMask moves the
// masked subset into the (pre-existing) output, leaving the rest UNCHANGED - matching cv::add/...
// with a mask (dst = mask ? result : dst). src and dst are the data (same depth, contiguous); the
// mask is one byte per pixel (bool/u8/s8 - never parameterized by its depth, we just test the byte
// != 0). Templated only by the element SIZE.
//
// The channel handling falls out of the existing broadcast machinery (no special case here): for
// single-channel data the tile is (width x 1) with the mask aligned per element (s1x == 1); for
// n-channel data the channel axis is the kernel width and the 1-channel mask broadcasts across it
// (s1x == 0), so a whole row of n elements is copied (or left untouched) under one mask test.
// Reference (scalar) implementation for now; a SIMD mask-expand + select can replace it later.
template<typename T, typename Tvec>
static int copyMaskKernel(const void* src0_, size_t s0y, size_t s0x,
                          const void* mask_, size_t s1y, size_t s1x,
                          const void*, size_t, size_t,
                          void* dst_, size_t dsty, int width, int height,
                          const double*, int, void*)
{
    s0y /= sizeof(T);
    dsty /= sizeof(T);

    CV_Assert(s0x == 1 && (s1x == 0 || s1x == 1));   // data contiguous; mask per-element or broadcast
    const T* src = (const T*)src0_;
    const uchar* mask = (const uchar*)mask_;
    T* dst = (T*)dst_;

    EW_TRY_COLLAPSE(2);

    int y = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if constexpr (sizeof(T) <= 4) {
        Tvec z = v_setzero_<Tvec>();
        auto loadExpandMask = [&](const uchar* mask) {
            if constexpr (sizeof(T) == 1u)
                return vx_load(mask);
            else if constexpr (sizeof(T) == 2u)
                return vx_load_expand(mask);
            else
                return vx_load_expand_q(mask);
        };

        const int VECSZ = VTraits<Tvec>::vlanes();
        if (height > 1 && width <= 4 && s0x == 1u && s1x == 0u &&
            s0y == (size_t)width && s1y == 1u && dsty == (size_t)width) {
            constexpr int MAXVECSZ = VTraits<Tvec>::max_nlanes;
            T maskbuf[MAXVECSZ*4]={};
            int dy = VECSZ;

            if (width == 2) {
                for (; y + dy <= height; y += dy, src += width*dy, mask += dy, dst += width*dy) {
                    Tvec m0 = loadExpandMask(mask), m1, s0, s1, d0, d1;
                    m0 = v_eq(m0, z);
                    v_store_interleave(maskbuf, m0, m0);
                    m0 = vx_load(maskbuf);
                    m1 = vx_load(maskbuf + VECSZ);
                    s0 = vx_load(src);
                    s1 = vx_load(src + VECSZ);
                    d0 = vx_load(dst);
                    d1 = vx_load(dst + VECSZ);
                    d0 = v_select(m0, d0, s0);
                    d1 = v_select(m1, d1, s1);
                    v_store(dst, d0);
                    v_store(dst + VECSZ, d1);
                }
            }
            else if (width == 3) {
                for (; y + dy <= height; y += dy, src += width*dy, mask += dy, dst += width*dy) {
                    Tvec m0 = loadExpandMask(mask), m1, m2, s0, s1, s2, d0, d1, d2;
                    m0 = v_eq(m0, z);
                    v_store_interleave(maskbuf, m0, m0, m0);
                    m0 = vx_load(maskbuf);
                    m1 = vx_load(maskbuf + VECSZ);
                    m2 = vx_load(maskbuf + VECSZ*2);
                    s0 = vx_load(src);
                    s1 = vx_load(src + VECSZ);
                    s2 = vx_load(src + VECSZ*2);
                    d0 = vx_load(dst);
                    d1 = vx_load(dst + VECSZ);
                    d2 = vx_load(dst + VECSZ*2);
                    d0 = v_select(m0, d0, s0);
                    d1 = v_select(m1, d1, s1);
                    d2 = v_select(m2, d2, s2);
                    v_store(dst, d0);
                    v_store(dst + VECSZ, d1);
                    v_store(dst + VECSZ*2, d2);
                }
            }
            else if (width == 4) {
                for (; y + dy <= height; y += dy, src += width*dy, mask += dy, dst += width*dy) {
                    Tvec m0 = loadExpandMask(mask), m1, m2, m3, s0, s1, s2, s3, d0, d1, d2, d3;
                    m0 = v_eq(m0, z);
                    v_store_interleave(maskbuf, m0, m0, m0, m0);
                    m0 = vx_load(maskbuf);
                    m1 = vx_load(maskbuf + VECSZ);
                    m2 = vx_load(maskbuf + VECSZ*2);
                    m3 = vx_load(maskbuf + VECSZ*3);
                    s0 = vx_load(src);
                    s1 = vx_load(src + VECSZ);
                    s2 = vx_load(src + VECSZ*2);
                    s3 = vx_load(src + VECSZ*3);
                    d0 = vx_load(dst);
                    d1 = vx_load(dst + VECSZ);
                    d2 = vx_load(dst + VECSZ*2);
                    d3 = vx_load(dst + VECSZ*3);
                    d0 = v_select(m0, d0, s0);
                    d1 = v_select(m1, d1, s1);
                    d2 = v_select(m2, d2, s2);
                    d3 = v_select(m3, d3, s3);
                    v_store(dst, d0);
                    v_store(dst + VECSZ, d1);
                    v_store(dst + VECSZ*2, d2);
                    v_store(dst + VECSZ*3, d3);
                }
            }
        }

        if (width >= VECSZ*2 && dst_ != mask_ && s0x == s1x) {
            for (; y < height; y++, src += s0y, mask += s1y, dst += dsty)
            {
                for (int x = 0; x < width; x += VECSZ*2) {
                    if (x + VECSZ*2 > width) {
                        x = width - VECSZ*2;
                    }
                    Tvec m0, m1, s0, s1, d0, d1;
                    vx_load_pair_as(mask + x, m0, m1);
                    s0 = vx_load(src + x);
                    s1 = vx_load(src + x + VECSZ);
                    d0 = vx_load(dst + x);
                    d1 = vx_load(dst + x + VECSZ);
                    m0 = v_eq(m0, z);
                    m1 = v_eq(m1, z);
                    d0 = v_select(m0, d0, s0);
                    d1 = v_select(m1, d1, s1);
                    v_store(dst + x, d0);
                    v_store(dst + x + VECSZ, d1);
                }
            }
            return 0;
        }
    }
#endif

    for (; y < height; y++, src += s0y, mask += s1y, dst += dsty)
    {
        if (s1x == 0)
        {
            if (mask[0]) for (int x = 0; x < width; x++) dst[x] = src[x];
            // else: leave the row untouched (preserve the existing output)
        }
        else                                // per-element mask (single-channel)
        {
            for (int x = 0; x < width; x++) {
                uchar m = mask[x];
                T s = src[x], d = dst[x];
                dst[x] = d ^ ((d ^ s) & T(-int(m != 0)));
            }
        }
    }
    return 0;
}

TKernel getCopyMaskFunc_(int depth)
{
    KernelFunc fptr = nullptr;
    switch (CV_ELEM_SIZE1(depth))
    {
    case 1: fptr = copyMaskKernel<uchar, v_uint8>; break;
    case 2: fptr = copyMaskKernel<ushort, v_uint16>; break;
    case 4: fptr = copyMaskKernel<unsigned, v_uint32>; break;
    case 8: fptr = copyMaskKernel<uint64_t, v_uint32>; break;
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

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::ew
