// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Layer 1: element-wise kernels + the getElemwiseFunc() dispatcher.
//
// This step implements OP_ADD (the full T x T -> Tr matrix) and OP_CAST (all depth pairs),
// on universal intrinsics (baseline SIMD, no dynamic dispatch yet). Other binary ops
// (sub/mul/div/pow) keep their simple f32 reference kernels until their own matrices land.
//
// Kernel shape (house style of core/src/arithm.simd.hpp + convert.simd.hpp):
//  - one 2D tile; per-row outer loop with stepy; dst contiguous in x.
//  - stepx is restricted to {0,1}: 1 = contiguous (SIMD), 0 = broadcast-scalar along x.
//    The general strided case is excluded (CV_Assert) - the executor guarantees it.
//  - continuity collapse 2D->1D when every operand+dst is gap-free.
//  - halide right-edge backoff for the SIMD tail; scalar tail / scalar path for width<VECSZ
//    and for any broadcast operand (vx_setall vectorization of broadcast is a later step).
//  - IN-PLACE SAFE: the right-edge backoff re-reads already-written lanes, which corrupts the
//    result when dst aliases an input; it is suppressed (fall to the scalar tail) when aliased.
//
// One template (binary_kernel / cast_kernel) covers the whole matrix: the work-vector type
// selects the native same-type saturating path (e.g. v_uint8 + v_add) vs the widening f32-hub
// path (v_float32), and a compile-time use_simd flag drops the SIMD body for the 32/64-bit
// outputs that have no vector path. Adding a new binary op = add one Op functor.

#include "ew_op.hpp"
#include "opencv2/core/hal/intrin.hpp"
// private prototype-only reuse of the typed load/store-as helpers (vx_load_as / v_store_as):
#include "../../src/convert.hpp"
#include <algorithm>
#include <cmath>
#include <cstring>

namespace cv {

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

// ===========================================================================
// Op functors (vector + scalar). New binary ops slot in here.
// ===========================================================================
struct EwAdd {
    template<typename V> static V vec(const V& a, const V& b) { return v_add(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    // Accumulate in the promoted type, NOT in W: for the native saturating path W is the narrow
    // lane type (schar/short/...), and (W)(a+b) would wrap in 8/16 bits before saturate_cast<Tr>
    // could clamp. Letting a+b promote (narrow -> int) keeps saturation for 8/16-bit outputs and
    // the natural wrap for 32/64-bit (both matching cv::add). The SIMD path already saturates.
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a + b); }
};

struct EwSub {
    template<typename V> static V vec(const V& a, const V& b) { return v_sub(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return W(a - b); }   // see EwAdd::scl
    // 64-bit unsigned has no wider WT to hold a-b, so the generic path would wrap on underflow.
    // Saturate at 0 to match cv::subtract (8/16/32-bit already saturate via SIMD floor / wide WT).
    static uint64_t scl(uint64_t a, uint64_t b, uint64_t) { return uint64_t(a >= b)*(a - b); }
};

// mul/div compute in a wide FLOAT work type (W = float for <=16-bit/f16/bf/f32, double for
// 32/64-bit/f64), matching cv::multiply/divide; the executor casts the work-type result down to
// rdepth. scalar (reference) path only for now - SIMD can follow. (No vec(): never instantiated.)
struct EwMul {
    template<typename V> static V vec(const V& a, const V& b) { return v_mul(a, b); }
#ifdef __ARM_NEON
    static v_uint16 vec(const v_uint16& a, const v_uint16& b) { return v_uint16x8(vmulq_u16(a.val, b.val)); }
#endif
    template<typename V> static V preproc(const V& a, const V& s) { return v_mul(a, s); }
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return a * b * s; }
};
// div has two variants by the COMMON INPUT type (matching cv::'s per-type kernel choice): integer
// inputs guard divide-by-zero -> 0 (cv:: iscalar_div); float inputs do NOT guard (cv:: fscalar_div,
// a/0 -> inf), which then saturates on the cast to an integer output exactly like cv::divide.
struct EwDivInt {
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return b != W(0) ? a * s / b : W(0); }
};
struct EwDivFlt {
    template<typename V> static V vec(const V& a, const V& b) { return v_div(a, b); }
    template<typename V> static V preproc(const V& a, const V& s) { return v_mul(a, s); }
    template<typename W, typename ST> static W scl(W a, W b, ST s) { return a * s / b; }
};

// min / max / absdiff: T x T -> T (same depth in and out, no scale). v_min/v_max exist for every
// vector lane type (64-bit ints fall back to scalar). absdiff uses v_absdiff (defined for the
// UNSIGNED and float lane types - signed/wide depths go through the scalar path), and the scalar
// |a-b| is computed branch-wise so it never underflows an unsigned work type.
struct EwMin {
    template<typename V> static V vec(const V& a, const V& b) { return v_min(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::min(a, b); }
};
struct EwMax {
    template<typename V> static V vec(const V& a, const V& b) { return v_max(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return std::max(a, b); }
};
struct EwAbsdiff {
    template<typename V> static V vec(const V& a, const V& b) { return v_absdiff(a, b); }
    template<typename V> static V preproc(const V& a, const V&) { return a; }
    template<typename W, typename ST> static W scl(W a, W b, ST) { return a > b ? W(a - b) : W(b - a); }
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
static int scalar_binary_kernel(const void* src0_, size_t s0y, size_t s0x,
                                const void* src1_, size_t s1y, size_t s1x,
                                const void*, size_t, size_t,
                                void* dst_, size_t dsty,
                                int width, int height, const double* params)
{
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
static void expand_scalar(const T* sc, size_t sx, int n0, WT* scbuf, int n)
{
    int i = 0;
    for (; i < n0; i++) scbuf[i] = (WT)sc[i*sx];
    for (; i < n; i++) scbuf[i] = scbuf[i - n0];
}

// Unified binary kernel: T0 x T1 -> Tr (operands same depth for arithmetic; cast is separate).
//   Wvec     = work vector. Native (v_uint8/...) drives the same-type saturating path
//              (v_add saturates 8/16-bit, wraps 32-bit); v_float32 drives the widening hub.
//   WT       = scalar work type for the tail.
//   Op       = operation functor (vec()/scl()).
//   use_simd = compile-time switch; false => pure scalar (32/64-bit widened outputs, f64).
// stepx in {0,1}; dst contiguous. In-place safe (see file header).
template<typename T, typename Tr, typename Wvec, typename WT, class Op, typename ST=WT, typename Wvec1=Wvec>
static int binary_kernel(const void* src0_, size_t s0y, size_t s0x,
                         const void* src1_, size_t s1y, size_t s1x,
                         const void*, size_t, size_t,
                         void* dst_, size_t dsty, int width, int height, const double* params)
{
    //return 0;
#if 1
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
        expand_scalar(s0y == 0 ? src0 : src1, s0y == 0 ? s0x : s1x, width, scbuf, ewidth);
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
                v0 = Op::vec(sc0, v0);
                v1 = Op::vec(sc1, v1);
                v2 = Op::vec(sc2, v2);
                v3 = Op::vec(sc3, v3);
                v4 = Op::vec(sc4, v4);
                v5 = Op::vec(sc5, v5);
                v_store_pair_as(dst, v0, v1);
                v_store_pair_as(dst + VECSZ*2, v2, v3);
                v_store_pair_as(dst + VECSZ*4, v4, v5);
            }
        }
        else {
            for (; y + dy <= height; y += dy, src0 += s0y*dy, dst += dsty*dy) {
                Wvec v0, v1, v2, v3, v4, v5;
                vx_load_pair_as(src0, v0, v1);
                vx_load_pair_as(src0 + VECSZ*2, v2, v3);
                vx_load_pair_as(src0 + VECSZ*4, v4, v5);
                v0 = Op::vec(v0, sc0);
                v1 = Op::vec(v1, sc1);
                v2 = Op::vec(v2, sc2);
                v3 = Op::vec(v3, sc3);
                v4 = Op::vec(v4, sc4);
                v5 = Op::vec(v5, sc5);
                v_store_pair_as(dst, v0, v1);
                v_store_pair_as(dst + VECSZ*2, v2, v3);
                v_store_pair_as(dst + VECSZ*4, v4, v5);
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
                    a0_ = Op::vec(a0_, b0_);
                    a1_ = Op::vec(a1_, b1_);
                    a2_ = Op::vec(a2_, b2_);
                    a3_ = Op::vec(a3_, b3_);
                    v_store_pair_as(dst + x, a0_, a1_);
                    v_store_pair_as(dst + x + VECSZ*2, a2_, a3_);
                }
            }
            else {
                for (; x < width; x += VECSZ*4) {
                    if (x + VECSZ*4 > width) { if (!use_tail_trick) break; x = width - VECSZ*4; }
                    vx_load_pair_as(src0 + x, a0, a1);
                    vx_load_pair_as(src0 + x + VECSZ*2, a2, a3);
                    vx_load_pair_as(src1 + x, b0, b1);
                    vx_load_pair_as(src1 + x + VECSZ*2, b2, b3);
                    a0 = Op::vec(Op::preproc(a0, vscalar), b0);
                    a1 = Op::vec(Op::preproc(a1, vscalar), b1);
                    a2 = Op::vec(Op::preproc(a2, vscalar), b2);
                    a3 = Op::vec(Op::preproc(a3, vscalar), b3);
                    v_store_pair_as(dst + x, a0, a1);
                    v_store_pair_as(dst + x + VECSZ*2, a2, a3);
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
                a0 = Op::vec(a0, b0);
                a1 = Op::vec(a1, b0);
                a2 = Op::vec(a2, b0);
                a3 = Op::vec(a3, b0);
                v_store_pair_as(dst + x, a0, a1);
                v_store_pair_as(dst + x + VECSZ*2, a2, a3);
            }
        }
        else {
            vx_setall_as(src0, b0);
            b0 = Op::preproc(b0, vscalar);
            for (; x < width; x += VECSZ*4) {
                if (x + VECSZ*4 > width) { if (!use_tail_trick) break; x = width - VECSZ*4; }
                vx_load_pair_as(src1 + x, a0, a1);
                vx_load_pair_as(src1 + x + VECSZ*2, a2, a3);
                a0 = Op::vec(b0, a0);
                a1 = Op::vec(b0, a1);
                a2 = Op::vec(b0, a2);
                a3 = Op::vec(b0, a3);
                v_store_pair_as(dst + x, a0, a1);
                v_store_pair_as(dst + x + VECSZ*2, a2, a3);
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
#endif
}

// ===========================================================================
// OP_ADD
// ===========================================================================

// add dispatch: T x T -> R (operands already same depth T).
//  - native saturating path (Wvec = native, use_simd=true) for T -> T on <=32-bit ints + f32;
//  - widening f32-hub (Wvec = v_float32, use_simd=true) for the widened/half/float outputs;
//  - scalar (use_simd=false) for 64-bit results and f64 (no vector path yet).
template<class Op>
ElemwiseFunc getAddSubFunc(int T, int R)
{
    switch (T)
    {
    case CV_8U:
        if (R == CV_8U)  return binary_kernel<uchar, uchar, v_uint8, short, Op, uchar>;
        if (R == CV_16S) return binary_kernel<uchar, short, v_int16, short, Op, short>;
        if (R == CV_32S) return binary_kernel<uchar, int, v_int16, short, Op, short>;
        if (R == CV_32F) return binary_kernel<uchar, float, v_int16, short, Op, short>;
        return nullptr;
    case CV_8S:
        if (R == CV_8S)  return binary_kernel<schar, schar, v_int8, short, Op, schar>;
        if (R == CV_16S) return binary_kernel<schar, short, v_int16, short, Op, short>;
        if (R == CV_32S) return binary_kernel<schar, int, v_int16, short, Op, short>;
        if (R == CV_32F) return binary_kernel<schar, float, v_int16, short, Op, short>;
        return nullptr;
    case CV_16U:
        if (R == CV_16U) return binary_kernel<ushort, ushort, v_uint16, int, Op, ushort>;
        if (R == CV_32S) return binary_kernel<ushort, int, v_int32, int, Op, int>;
        if (R == CV_32F) return binary_kernel<ushort, float, v_int32, int, Op, int>;
        return nullptr;
    case CV_16S:
        if (R == CV_16S) return binary_kernel<short, short, v_int16, int, Op, short>;
        if (R == CV_32S) return binary_kernel<short, int,   v_int32, int, Op, int>;
        if (R == CV_32F) return binary_kernel<short, float, v_int32, int, Op, int>;
        return nullptr;
    case CV_32U:
        if (R == CV_32U) return binary_kernel<unsigned, unsigned, v_uint32, int64_t, Op, unsigned>;
        if (R == CV_64S) return scalar_binary_kernel<unsigned, int64_t, int64_t, Op>;
        if (R == CV_64F) return scalar_binary_kernel<unsigned, double, int64_t, Op>;
        return nullptr;
    case CV_32S:
        if (R == CV_32S) return binary_kernel<int, int, v_int32, int64_t, Op, int>;
        if (R == CV_64S) return scalar_binary_kernel<int, int64_t, int64_t, Op>;
        if (R == CV_64F) return scalar_binary_kernel<int, double, int64_t, Op>;
        return nullptr;
    case CV_64U:
        if (R == CV_64U) return scalar_binary_kernel<uint64_t, uint64_t, uint64_t, Op>;
        if (R == CV_64F) return scalar_binary_kernel<uint64_t, double, double, Op>;
        return nullptr;
    case CV_64S:
        if (R == CV_64S) return scalar_binary_kernel<int64_t, int64_t, int64_t, Op>;
        if (R == CV_64F) return scalar_binary_kernel<int64_t, double, double, Op>;
        return nullptr;
    case CV_16F:
        #if CV_SIMD_16F
        if (R == CV_16F) return binary_kernel<hfloat, hfloat, v_float16, float, Op, hfloat>;
        #else
        if (R == CV_16F) return binary_kernel<hfloat, hfloat, v_float32, float, Op, float>;
        #endif
        if (R == CV_32F) return binary_kernel<hfloat, float, v_float32, float, Op, float>;
        return nullptr;
    case CV_16BF:
        if (R == CV_16BF) return binary_kernel<bfloat, bfloat, v_float32, float, Op, float>;
        if (R == CV_32F)  return binary_kernel<bfloat, float,  v_float32, float, Op, float>;
        return nullptr;
    case CV_32F:
        if (R == CV_32F) return binary_kernel<float, float, v_float32, float, Op, float>;
        return nullptr;
    case CV_64F:
        if (R == CV_64F) return scalar_binary_kernel<double, double, double, Op>;
        return nullptr;
    default:
        return nullptr;
    }
}

static ElemwiseFunc getMulFunc(int T, int R)
{
    switch (T)
    {
    case CV_8U:
        #if CV_SIMD_16F
        if (R == CV_8U)  return binary_kernel<uchar, uchar, v_float16, float, EwMul, float, v_uint16>;
        #else
        if (R == CV_8U)  return binary_kernel<uchar, uchar, v_float32, float, EwMul>;
        #endif
        if (R == CV_32F) return binary_kernel<uchar, float, v_float32, float, EwMul>;
        return nullptr;
    case CV_8S:
        #if CV_SIMD_16F
        if (R == CV_8S)  return binary_kernel<schar, schar, v_float16, float, EwMul, float, v_int16>;
        #else
        if (R == CV_8S)  return binary_kernel<schar, schar, v_float32, float, EwMul>;
        #endif
        if (R == CV_32F) return binary_kernel<schar, float, v_float32, float, EwMul>;
        return nullptr;
    case CV_16U:
        if (R == CV_16U) return binary_kernel<ushort, ushort, v_float32, float, EwMul>;
        if (R == CV_32F) return binary_kernel<ushort, float,  v_float32, float, EwMul>;
        return nullptr;
    case CV_16S:
        if (R == CV_16S) return binary_kernel<short, short, v_float32, float, EwMul>;
        if (R == CV_32F) return binary_kernel<short, float, v_float32, float, EwMul>;
        return nullptr;
    case CV_16F:
        if (R == CV_16F) return binary_kernel<hfloat, hfloat, v_float32, float, EwMul>;
        if (R == CV_32F) return binary_kernel<hfloat, float,  v_float32, float, EwMul>;
        return nullptr;
    case CV_16BF:
        if (R == CV_16BF) return binary_kernel<bfloat, bfloat, v_float32, float, EwMul>;
        if (R == CV_32F)  return binary_kernel<bfloat, float,  v_float32, float, EwMul>;
        return nullptr;
    case CV_32F:
        return R == CV_32F ? binary_kernel<float, float, v_float32, float, EwMul> : nullptr;
    case CV_32U:
        if (R == CV_32U) return scalar_binary_kernel<unsigned, unsigned, double, EwMul>;
        if (R == CV_64F) return scalar_binary_kernel<unsigned, double,   double, EwMul>;
        return nullptr;
    case CV_32S:
        if (R == CV_32S) return scalar_binary_kernel<int, int,    double, EwMul>;
        if (R == CV_64F) return scalar_binary_kernel<int, double, double, EwMul>;
        return nullptr;
    case CV_64U:
        return R == CV_64F ? scalar_binary_kernel<uint64_t, double, double, EwMul> : nullptr;
    case CV_64S:
        return R == CV_64F ? scalar_binary_kernel<int64_t,  double, double, EwMul> : nullptr;
    case CV_64F:
        return R == CV_64F ? scalar_binary_kernel<double,   double, double, EwMul> : nullptr;
    default:
        return nullptr;
    }
}

// `checked` (decided by the CALLER from the ORIGINAL input types) selects the divide-by-zero
// policy, INDEPENDENT of the work type R: EwDivInt guards /0 -> 0 for integer-semantics division,
// EwDivFlt does not (a/0 -> inf, saturating on a later cast, as cv::divide does for float inputs).
// It must apply on EVERY row - two wide integers (e.g. 32U / 64S) promote to a 64F work type yet
// still need the integer guard, so the float-work rows can't hardcode EwDivFlt.
ElemwiseFunc getDivFunc(int T, int R, bool checked)
{
    #define EW_DIV(T_, W_) (checked ? scalar_binary_kernel<T_, W_, W_, EwDivInt> \
                                    : scalar_binary_kernel<T_, W_, W_, EwDivFlt>)
    switch (T)
    {
    case CV_8U:   return R == CV_32F ? EW_DIV(uchar,    float)  : nullptr;
    case CV_8S:   return R == CV_32F ? EW_DIV(schar,    float)  : nullptr;
    case CV_16U:  return R == CV_32F ? EW_DIV(ushort,   float)  : nullptr;
    case CV_16S:  return R == CV_32F ? EW_DIV(short,    float)  : nullptr;
    case CV_16F:  return R == CV_32F ? EW_DIV(hfloat,   float)  : nullptr;
    case CV_16BF: return R == CV_32F ? EW_DIV(bfloat,   float)  : nullptr;
    case CV_32F:  return R == CV_32F ? EW_DIV(float,    float)  : nullptr;
    case CV_32U:  return R == CV_64F ? EW_DIV(unsigned, double) : nullptr;
    case CV_32S:  return R == CV_64F ? EW_DIV(int,      double) : nullptr;
    case CV_64U:  return R == CV_64F ? EW_DIV(uint64_t, double) : nullptr;
    case CV_64S:  return R == CV_64F ? EW_DIV(int64_t,  double) : nullptr;
    case CV_64F:  return R == CV_64F ? EW_DIV(double,   double) : nullptr;
    default:      return nullptr;
    }
    #undef EW_DIV
}

// min / max: T x T -> T for every depth. Native v_min/v_max on the matching lane type (8/16/32-bit
// ints, f16/bf16/f32); 64-bit ints and f64 use the scalar path. Op = EwMin or EwMax.
template<class Op>
static ElemwiseFunc getMinMaxFunc(int T)
{
    switch (T)
    {
    case CV_8U:   return binary_kernel<uchar,    uchar,    v_uint8,   short,   Op, uchar>;
    case CV_8S:   return binary_kernel<schar,    schar,    v_int8,    short,   Op, schar>;
    case CV_16U:  return binary_kernel<ushort,   ushort,   v_uint16,  int,     Op, ushort>;
    case CV_16S:  return binary_kernel<short,    short,    v_int16,   int,     Op, short>;
    case CV_32U:  return binary_kernel<unsigned, unsigned, v_uint32,  int64_t, Op, unsigned>;
    case CV_32S:  return binary_kernel<int,      int,      v_int32,   int64_t, Op, int>;
    #if CV_SIMD_16F
    case CV_16F:  return binary_kernel<hfloat,   hfloat,   v_float16, float,   Op, hfloat>;
    #else
    case CV_16F:  return binary_kernel<hfloat,   hfloat,   v_float32, float,   Op, float>;
    #endif
    case CV_16BF: return binary_kernel<bfloat,   bfloat,   v_float32, float,   Op, float>;
    case CV_32F:  return binary_kernel<float,    float,    v_float32, float,   Op, float>;
    case CV_64U:  return scalar_binary_kernel<uint64_t, uint64_t, uint64_t, Op>;
    case CV_64S:  return scalar_binary_kernel<int64_t,  int64_t,  int64_t,  Op>;
    case CV_64F:  return scalar_binary_kernel<double,   double,   double,   Op>;
    default:      return nullptr;
    }
}

// absdiff: |a-b|, T x T -> T. v_absdiff is defined for the UNSIGNED and float lane types only
// (signed v_absdiff would change the lane type), so u8/u16/f16/bf16/f32 take the SIMD path and the
// signed / wide / f64 depths use the scalar kernel (|a-b| with the widened work type).
static ElemwiseFunc getAbsdiffFunc(int T)
{
    switch (T)
    {
    case CV_8U:   return binary_kernel<uchar,  uchar,  v_uint8,  short, EwAbsdiff, uchar>;
    case CV_16U:  return binary_kernel<ushort, ushort, v_uint16, int,   EwAbsdiff, ushort>;
    #if CV_SIMD_16F
    case CV_16F:  return binary_kernel<hfloat, hfloat, v_float16, float, EwAbsdiff, hfloat>;
    #else
    case CV_16F:  return binary_kernel<hfloat, hfloat, v_float32, float, EwAbsdiff, float>;
    #endif
    case CV_16BF: return binary_kernel<bfloat, bfloat, v_float32, float, EwAbsdiff, float>;
    case CV_32F:  return binary_kernel<float,  float,  v_float32, float, EwAbsdiff, float>;
    case CV_8S:   return scalar_binary_kernel<schar,    schar,    short,   EwAbsdiff>;
    case CV_16S:  return scalar_binary_kernel<short,    short,    int,     EwAbsdiff>;
    case CV_32U:  return scalar_binary_kernel<unsigned, unsigned, int64_t, EwAbsdiff>;
    case CV_32S:  return scalar_binary_kernel<int,      int,      int64_t, EwAbsdiff>;
    case CV_64U:  return scalar_binary_kernel<uint64_t, uint64_t, uint64_t, EwAbsdiff>;
    case CV_64S:  return scalar_binary_kernel<int64_t,  int64_t,  int64_t,  EwAbsdiff>;
    case CV_64F:  return scalar_binary_kernel<double,   double,   double,   EwAbsdiff>;
    default:      return nullptr;
    }
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
                          void* dst_, size_t dsty, int width, int height, const double*)
{
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

static ElemwiseFunc getCopyMaskFunc(int depth)
{
    switch (CV_ELEM_SIZE1(depth))
    {
    case 1: return copyMaskKernel<uchar, v_uint8>;
    case 2: return copyMaskKernel<ushort, v_uint16>;
    case 4: return copyMaskKernel<unsigned, v_uint32>;
    case 8: return copyMaskKernel<uint64_t, v_uint32>;
    default: return nullptr;
    }
}

// ===========================================================================
// Other binary ops: keep the simple f32 reference kernels until their matrices land.
// ===========================================================================
struct OpPow { static double apply(double a, double b) { return std::pow(a, b); } };

template<class Op, typename T0, typename T1, typename Tr>
static int binaryKernel(const void* src0, size_t s0y, size_t s0x,
                        const void* src1, size_t s1y, size_t s1x,
                        const void*, size_t, size_t,
                        void* dst, size_t dsty, int width, int height, const double*)
{
    const T0* p0 = (const T0*)src0;
    const T1* p1 = (const T1*)src1;
    Tr* pd = (Tr*)dst;
    for (int y = 0; y < height; y++)
    {
        const T0* a = p0 + (size_t)y * s0y;
        const T1* b = p1 + (size_t)y * s1y;
        Tr* c = pd + (size_t)y * dsty;
        for (int x = 0; x < width; x++, a += s0x, b += s1x, c++)
            *c = saturate_cast<Tr>(Op::apply((double)*a, (double)*b));
    }
    return 0;
}

static ElemwiseFunc getBinaryArithF32(int op)
{
    switch (op)
    {
    case OP_POW: return binaryKernel<OpPow, float, float, float>;
    default:     return nullptr;
    }
}

// ===========================================================================
// Public dispatcher.
// ===========================================================================
ElemwiseFunc getElemwiseFunc(TOp op, int depth0, int depth1, int depth2, int rdepth)
{
    (void)depth2;

    // OP_CAST / OP_CONVERT_SCALE are not element-wise kernels: they reuse core's convert
    // BinaryFunc directly (EW_FN_BINARY). resolveInsnKernel routes them; here they are not ours.
    if (op == OP_CAST || op == OP_CONVERT_SCALE)
        return nullptr;

    if (op == OP_ADD || op == OP_SUB)
    {
        if (depth0 != depth1) return nullptr;   // operands must be the same type
        return op == OP_ADD ? getAddSubFunc<EwAdd>(depth0, rdepth) :
                              getAddSubFunc<EwSub>(depth0, rdepth);
    }

    // OP_MUL / OP_DIV: operands same type T; compute in the float work type, output it (rdepth ==
    // the work float). The executor casts the work result down to the real output depth.
    if (op == OP_MUL || op == OP_DIV)
    {
        if (depth0 != depth1) return nullptr;
        if (op == OP_MUL) return getMulFunc(depth0, rdepth);
        // both operands share depth0 here; integer inputs guard divide-by-zero (-> 0), floats do
        // not (a/0 -> inf, matching cv::divide). The full composer overrides this via getDivFunc
        // directly when wide integers promote to a float work type (see makeBinaryArithProgram).
        const bool isflt = depth0==CV_16F || depth0==CV_16BF || depth0==CV_32F || depth0==CV_64F;
        return getDivFunc(depth0, rdepth, !isflt);
    }

    // OP_MIN / OP_MAX / OP_ABSDIFF: operands same type T, result the same type (T x T -> T).
    if (op == OP_MIN || op == OP_MAX || op == OP_ABSDIFF)
    {
        if (depth0 != depth1 || rdepth != depth0) return nullptr;
        if (op == OP_MIN)     return getMinMaxFunc<EwMin>(depth0);
        if (op == OP_MAX)     return getMinMaxFunc<EwMax>(depth0);
        return getAbsdiffFunc(depth0);
    }

    // OP_COPY_MASK: data depth == depth0 == rdepth; depth1 is the mask (any 1-byte type, ignored
    // for dispatch - the kernel just tests bytes). Selected by element size only.
    if (op == OP_COPY_MASK)
        return getCopyMaskFunc(rdepth);

    if (opArity(op) == 2)
    {
        if (depth0 == CV_32F && depth1 == CV_32F && rdepth == CV_32F)
            return getBinaryArithF32(op);
        return nullptr;
    }

    return nullptr;
}

}   // namespace ew

// Engine-side mirror declarations of core's exported convert dispatchers (CV_EXPORTS in
// core/src/precomp.hpp). A function's return type is not part of its mangled name, so declaring
// these to return ew::EwBinaryFunc links to cv::getConvertFunc / getConvertScaleFunc (which
// return core's identical BinaryFunc) without pulling in the private precomp.hpp.
ew::EwBinaryFunc getConvertFunc(int sdepth, int ddepth);
ew::EwBinaryFunc getConvertScaleFunc(int sdepth, int ddepth);

namespace ew {

// Single resolution point for an instruction's kernel + calling convention (see ew_op.hpp).
// OP_CAST / OP_CONVERT_SCALE bind a core convert BinaryFunc (EW_FN_BINARY); every other op binds
// the element-wise kernel from getElemwiseFunc (EW_FN_ELEMWISE).
bool resolveInsnKernel(TExpr::Insn& ins, int depth0, int depth1, int depth2, int rdepth)
{
    if (ins.op == OP_CAST)
    {
        ins.fnkind = EW_FN_BINARY;
        ins.bfptr = cv::getConvertFunc(depth0, rdepth);
        return ins.bfptr != nullptr;
    }
    if (ins.op == OP_CONVERT_SCALE)
    {
        ins.fnkind = EW_FN_BINARY;
        ins.bfptr = cv::getConvertScaleFunc(depth0, rdepth);
        return ins.bfptr != nullptr;
    }
    ins.fnkind = EW_FN_ELEMWISE;
    ins.fptr = getElemwiseFunc(ins.op, depth0, depth1, depth2, rdepth);
    return ins.fptr != nullptr;
}

}} // namespace ew, cv
