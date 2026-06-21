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

static inline void vx_setall_as(const float* p, v_float32& a)
{ a = vx_setall_f32(*p); }
static inline void vx_setall_as(const bfloat* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }
static inline void vx_setall_as(const hfloat* p, v_float32& a)
{ a = vx_setall_f32(float(*p)); }

// Native-width identity load/store-as overloads, complementing the f32-hub helpers in
// convert.hpp (which only target v_float32). These let one binary_kernel<> template serve
// both the native same-type path (work vector = native, e.g. v_uint8 with saturating v_add)
// and the widening f32-hub path (work vector = v_float32). The float/f32 variants already
// live in convert.hpp, so only the integer same-width pairs are added here.
static inline void vx_load_as(const uchar* p, v_uint8& v)
{ v = vx_load(p); }

static inline void vx_load_as(const uchar* p, v_int16& v)
{ v = v_reinterpret_as_s16(vx_load_expand(p)); }

static inline void vx_load_as(const schar* p, v_int8& v)
{ v = vx_load(p); }

static inline void vx_load_as(const schar* p, v_int16& v)
{ v = vx_load_expand(p); }

static inline void vx_load_as(const ushort* p, v_uint16& v)
{ v = vx_load(p); }

static inline void vx_load_as(const ushort* p, v_int32& v)
{ v = v_reinterpret_as_s32(vx_load_expand(p)); }

static inline void vx_load_as(const short* p, v_int16& v)
{ v = vx_load(p); }

static inline void vx_load_as(const short* p, v_int32& v)
{ v = vx_load_expand(p); }

static inline void vx_load_as(const unsigned* p, v_uint32& v)
{ v = vx_load(p); }

static inline void vx_load_as(const int* p, v_int32& v)
{ v = vx_load(p); }

static inline void v_store_as(uchar* p, const v_uint8& v)
{ v_store(p, v); }

static inline void v_store_as(schar* p, const v_int8&  v)
{ v_store(p, v); }

static inline void v_store_as(ushort* p, const v_uint16& v)
{ v_store(p, v); }

static inline void v_store_as(short* p, const v_int16& v)
{ v_store(p, v); }

static inline void v_store_as(int* p, const v_int16& v)
{
    v_int32 a, b;
    v_expand(v, a, b);
    v_store(p, a);
    v_store(p + VTraits<v_int32>::vlanes(), b);
}

static inline void v_store_as(float* p, const v_int16& v)
{
    v_int32 a, b;
    v_expand(v, a, b);
    v_store(p, v_cvt_f32(a));
    v_store(p + VTraits<v_float32>::vlanes(), v_cvt_f32(b));
}

static inline void v_store_as(float* p, const v_int32& v)
{
    v_store(p, v_cvt_f32(v));
}

static inline void v_store_as(unsigned* p, const v_uint32& v)
{ v_store(p, v); }

static inline void v_store_as(int* p, const v_int32& v)
{ v_store(p, v); }

#if CV_SIMD_16F
static inline void vx_setall_as(const hfloat* p, v_float16& v)
{ v = vx_setall_f16(*p); }

static inline void vx_load_as(const hfloat* p, v_float16& v)
{ v = vx_load(p); }

static inline void v_store_as(hfloat* p, const v_float16& v)
{ v_store(p, v); }
#endif

#endif

namespace ew {

// ===========================================================================
// Op functors (vector + scalar). New binary ops slot in here.
// ===========================================================================
struct EwAdd {
    template<typename V> static V vec(const V& a, const V& b) { return v_add(a, b); }
    // Accumulate in the promoted type, NOT in W: for the native saturating path W is the narrow
    // lane type (schar/short/...), and (W)(a+b) would wrap in 8/16 bits before saturate_cast<Tr>
    // could clamp. Letting a+b promote (narrow -> int) keeps saturation for 8/16-bit outputs and
    // the natural wrap for 32/64-bit (both matching cv::add). The SIMD path already saturates.
    template<typename W> static W scl(W a, W b) { return W(a + b); }
};

struct EwSub {
    template<typename V> static V vec(const V& a, const V& b) { return v_sub(a, b); }
    template<typename W> static W scl(W a, W b) { return W(a - b); }   // see EwAdd::scl
    // 64-bit unsigned has no wider WT to hold a-b, so the generic path would wrap on underflow.
    // Saturate at 0 to match cv::subtract (8/16/32-bit already saturate via SIMD floor / wide WT).
    static uint64_t scl(uint64_t a, uint64_t b) { return uint64_t(a >= b)*(a - b); }
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
template<typename T, typename Tr, typename WT, class Op>
static int scalar_binary_kernel(const void* src0_, size_t s0y, size_t s0x,
                                const void* src1_, size_t s1y, size_t s1x,
                                const void*, size_t, size_t,
                                void* dst_, size_t dsty, int width, int height, void*)
{
    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    Tr* dst = (Tr*)dst_;
    EW_TRY_COLLAPSE(2);
    for (int y = 0; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        if (s0x == s1x) {
            for (int x = 0; x < width; x++)
                dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x], (WT)src1[x]));
        }
        else if (s0x == 0) {
            WT sc0 = (WT)src0[0];
            for (int x = 0; x < width; x++)
                dst[x] = saturate_cast<Tr>(Op::scl(sc0, (WT)src1[x]));
        }
        else {
            WT sc1 = (WT)src1[0];
            for (int x = 0; x < width; x++)
                dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x], sc1));
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
template<typename T, typename Tr, typename Wvec, typename WT, class Op>
static int binary_kernel(const void* src0_, size_t s0y, size_t s0x,
                         const void* src1_, size_t s1y, size_t s1x,
                         const void*, size_t, size_t,
                         void* dst_, size_t dsty, int width, int height, void*)
{
    CV_Assert((s0x|s1x) == 1u || (s0x|s1x) + (size_t)width == 1u);

    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    Tr* dst = (Tr*)dst_;
    EW_TRY_COLLAPSE(2);
    int y = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    using Wlane = typename VTraits<Wvec>::lane_type;
    const int VECSZ = VTraits<Wvec>::vlanes();
    const bool use_tail_trick = width >= VECSZ*3 && src0_ != dst_ && src1_ != dst_;

    if (height > 1 && width <= 4 &&
        ((s0y == 0 && s1y == width*s1x) ||
         (s1y == 0 && s0y == width*s0x)) &&
        dsty == (size_t)width) {
        constexpr int MAXVECSZ = VTraits<Wvec>::max_nlanes;
        Wlane scbuf[MAXVECSZ*3];
        const int ewidth = VECSZ*3;
        expand_scalar(s0y == 0 ? src0 : src1, s0y == 0 ? s0x : s1x, width, scbuf, ewidth);
        int dy = ewidth / width;
        Wvec sc0 = vx_load(scbuf), sc1 = vx_load(scbuf + VECSZ), sc2 = vx_load(scbuf + VECSZ*2);

        if (s0y == 0) {
            for (; y + dy <= height; y += dy, src1 += s1y*dy, dst += dsty*dy) {
                Wvec v0, v1, v2;
                vx_load_as(src1, v0);
                vx_load_as(src1 + VECSZ, v1);
                vx_load_as(src1 + VECSZ*2, v2);
                v0 = Op::vec(sc0, v0);
                v1 = Op::vec(sc1, v1);
                v2 = Op::vec(sc2, v2);
                v_store_as(dst, v0);
                v_store_as(dst + VECSZ, v1);
                v_store_as(dst + VECSZ*2, v2);
            }
        }
        else {
            for (; y + dy <= height; y += dy, src0 += s0y*dy, dst += dsty*dy) {
                Wvec v0, v1, v2;
                vx_load_as(src0, v0);
                vx_load_as(src0 + VECSZ, v1);
                vx_load_as(src0 + VECSZ*2, v2);
                v0 = Op::vec(v0, sc0);
                v1 = Op::vec(v1, sc1);
                v2 = Op::vec(v2, sc2);
                v_store_as(dst, v0);
                v_store_as(dst + VECSZ, v1);
                v_store_as(dst + VECSZ*2, v2);
            }
        }
    }
#endif
    for (; y < height; y++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
    #if (CV_SIMD || CV_SIMD_SCALABLE)
        Wvec a0, a1, a2, b0, b1, b2;
        if (s0x == s1x) {
            for (; x < width; x += VECSZ*3) {
                if (x + VECSZ*3 > width) { if (!use_tail_trick) break; x = width - VECSZ*3; }
                vx_load_as(src0 + x, a0);
                vx_load_as(src0 + x + VECSZ, a1);
                vx_load_as(src0 + x + VECSZ*2, a2);
                vx_load_as(src1 + x, b0);
                vx_load_as(src1 + x + VECSZ, b1);
                vx_load_as(src1 + x + VECSZ*2, b2);
                a0 = Op::vec(a0, b0);
                a1 = Op::vec(a1, b1);
                a2 = Op::vec(a2, b2);
                v_store_as(dst + x, a0);
                v_store_as(dst + x + VECSZ, a1);
                v_store_as(dst + x + VECSZ*2, a2);
            }
        }
        else if (s1x == 0) {
            vx_setall_as(src1, b0);
            for (; x < width; x += VECSZ*3) {
                if (x + VECSZ*3 > width) { if (!use_tail_trick) break; x = width - VECSZ*3; }
                vx_load_as(src0 + x, a0);
                vx_load_as(src0 + x + VECSZ, a1);
                vx_load_as(src0 + x + VECSZ*2, a2);
                a0 = Op::vec(a0, b0);
                a1 = Op::vec(a1, b0);
                a2 = Op::vec(a2, b0);
                v_store_as(dst + x, a0);
                v_store_as(dst + x + VECSZ, a1);
                v_store_as(dst + x + VECSZ*2, a2);
            }
        }
        else {
            vx_setall_as(src0, b0);                 // b0 = broadcast src0 (the scalar operand)
            for (; x < width; x += VECSZ*3) {
                if (x + VECSZ*3 > width) { if (!use_tail_trick) break; x = width - VECSZ*3; }
                vx_load_as(src1 + x, a0);
                vx_load_as(src1 + x + VECSZ, a1);
                vx_load_as(src1 + x + VECSZ*2, a2);
                a0 = Op::vec(b0, a0);               // src0 (scalar) OP src1 - order matters for sub
                a1 = Op::vec(b0, a1);
                a2 = Op::vec(b0, a2);
                v_store_as(dst + x, a0);
                v_store_as(dst + x + VECSZ, a1);
                v_store_as(dst + x + VECSZ*2, a2);
            }
        }
    #endif
        for (; x < width; x++)
            dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x*s0x], (WT)src1[x*s1x]));
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
ElemwiseFunc getAddSubFunc(int T, int R)
{
    switch (T)
    {
    case CV_8U:
        if (R == CV_8U)  return binary_kernel<uchar, uchar, v_uint8, short, Op>;
        if (R == CV_16S) return binary_kernel<uchar, short, v_int16, short, Op>;
        if (R == CV_32S) return binary_kernel<uchar, int, v_int16, short, Op>;
        if (R == CV_32F) return binary_kernel<uchar, float, v_int16, short, Op>;
        return nullptr;
    case CV_8S:
        if (R == CV_8S)  return binary_kernel<schar, schar, v_int8, short, Op>;
        if (R == CV_16S) return binary_kernel<schar, short, v_int16, short, Op>;
        if (R == CV_32S) return binary_kernel<schar, int, v_int16, short, Op>;
        if (R == CV_32F) return binary_kernel<schar, float, v_int16, short, Op>;
        return nullptr;
    case CV_16U:
        if (R == CV_16U) return binary_kernel<ushort, ushort, v_uint16, int, Op>;
        if (R == CV_32S) return binary_kernel<ushort, int, v_int32, int, Op>;
        if (R == CV_32F) return binary_kernel<ushort, float, v_int32, int, Op>;
        return nullptr;
    case CV_16S:
        if (R == CV_16S) return binary_kernel<short, short, v_int16, int, Op>;
        if (R == CV_32S) return binary_kernel<short, int,   v_int32, int, Op>;
        if (R == CV_32F) return binary_kernel<short, float, v_int32, int, Op>;
        return nullptr;
    case CV_32U:
        if (R == CV_32U) return binary_kernel<unsigned, unsigned, v_uint32, int64_t, Op>;
        if (R == CV_64S) return scalar_binary_kernel<unsigned, int64_t, int64_t, Op>;
        if (R == CV_64F) return scalar_binary_kernel<unsigned, double, int64_t, Op>;
        return nullptr;
    case CV_32S:
        if (R == CV_32S) return binary_kernel<int, int, v_int32, int64_t, Op>;
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
        if (R == CV_16F) return binary_kernel<hfloat, hfloat, v_float16, float, Op>;
        #else
        if (R == CV_16F) return binary_kernel<hfloat, hfloat, v_float32, float, Op>;
        #endif
        if (R == CV_32F) return binary_kernel<hfloat, float, v_float32, float, Op>;
        return nullptr;
    case CV_16BF:
        if (R == CV_16BF) return binary_kernel<bfloat, bfloat, v_float32, float, Op>;
        if (R == CV_32F)  return binary_kernel<bfloat, float,  v_float32, float, Op>;
        return nullptr;
    case CV_32F:
        if (R == CV_32F) return binary_kernel<float, float, v_float32, float, Op>;
        return nullptr;
    case CV_64F:
        if (R == CV_64F) return scalar_binary_kernel<double, double, double, Op>;
        return nullptr;
    default:
        return nullptr;
    }
}

// ===========================================================================
// OP_CAST / OP_CONVERT_SCALE  ->  adapter over core's getConvertFunc / getConvertScaleFunc.
// ===========================================================================
// A single generic ElemwiseFunc that carries no type info itself: the executor builds an
// EwCtx.cvt (the type-specialized, CPU-dispatched core BinaryFunc + element sizes + optional
// scale/shift) before the parallel loop. The adapter does no arithmetic - it only translates
// element-steps -> byte-steps and the tile extent -> Size, then calls the wrapped BinaryFunc.
// Reusing the existing convert kernels means we don't re-implement the whole cast matrix.
static int convertAdapter(const void* src0, size_t s0y, size_t s0x,
                          const void*, size_t, size_t, const void*, size_t, size_t,
                          void* dst, size_t dsty, int width, int height, void* ctx_)
{
    CV_Assert(ctx_ != nullptr && s0x <= 1);
    const EwCtx* c = (const EwCtx*)ctx_;
    const size_t srowb = s0y * (size_t)c->cvt.sesz1;     // row steps in bytes
    const size_t drowb = dsty * (size_t)c->cvt.desz1;

    if (s0x == 1)   // contiguous source run: one BinaryFunc call over the whole tile
    {
        c->cvt.fn((const uchar*)src0, srowb, nullptr, 0,
                  (uchar*)dst, drowb, Size(width, height), (void*)c->cvt.scale);
        return 0;
    }

    // s0x == 0: broadcast source (a scalar repeated along x). The wrapped BinaryFunc needs a
    // contiguous source, so cast the single value once per row, then replicate it across the row.
    const int desz1 = c->cvt.desz1;
    const uchar* s = (const uchar*)src0;
    uchar* d = (uchar*)dst;
    for (int i = 0; i < height; i++, s += srowb, d += drowb)
    {
        c->cvt.fn(s, 0, nullptr, 0, d, 0, Size(1, 1), (void*)c->cvt.scale);
        for (int x = 1; x < width; x++)
            std::memcpy(d + (size_t)x * desz1, d, (size_t)desz1);
    }
    return 0;
}

// ===========================================================================
// Other binary ops: keep the simple f32 reference kernels until their matrices land.
// ===========================================================================
struct OpMul { static double apply(double a, double b) { return a * b; } };
struct OpDiv { static double apply(double a, double b) { return b != 0 ? a / b : 0; } };
struct OpPow { static double apply(double a, double b) { return std::pow(a, b); } };

template<class Op, typename T0, typename T1, typename Tr>
static int binaryKernel(const void* src0, size_t s0y, size_t s0x,
                        const void* src1, size_t s1y, size_t s1x,
                        const void*, size_t, size_t,
                        void* dst, size_t dsty, int width, int height, void*)
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
    case OP_MUL: return binaryKernel<OpMul, float, float, float>;
    case OP_DIV: return binaryKernel<OpDiv, float, float, float>;
    case OP_POW: return binaryKernel<OpPow, float, float, float>;
    default:     return nullptr;
    }
}

// ===========================================================================
// Public dispatcher.
// ===========================================================================
ElemwiseFunc getElemwiseFunc(ElemwiseOp op, int depth0, int depth1, int depth2, int rdepth)
{
    (void)depth2;

    // OP_CAST and OP_CONVERT_SCALE are both served by the adapter over core's convert kernels;
    // the executor builds the EwCtx (the type-specialized BinaryFunc + scale/offset) - for a plain
    // cast it wraps getConvertFunc with scale {1,0}, for convert_scale getConvertScaleFunc with the
    // {alpha, offset} read from the const operands. The kernel itself is the same plumbing.
    if (op == OP_CAST || op == OP_CONVERT_SCALE)
        return convertAdapter;

    if (op == OP_ADD || op == OP_SUB)
    {
        if (depth0 != depth1) return nullptr;   // operands must be the same type
        return op == OP_ADD ? getAddSubFunc<EwAdd>(depth0, rdepth) :
                              getAddSubFunc<EwSub>(depth0, rdepth);
    }

    if (opArity(op) == 2)
    {
        if (depth0 == CV_32F && depth1 == CV_32F && rdepth == CV_32F)
            return getBinaryArithF32(op);
        return nullptr;
    }

    return nullptr;
}

}} // namespace ew, cv
