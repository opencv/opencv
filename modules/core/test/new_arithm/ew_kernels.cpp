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

#if (CV_SIMD || CV_SIMD_SCALABLE)
// Native-width identity load/store-as overloads, complementing the f32-hub helpers in
// convert.hpp (which only target v_float32). These let one binary_kernel<> template serve
// both the native same-type path (work vector = native, e.g. v_uint8 with saturating v_add)
// and the widening f32-hub path (work vector = v_float32). The float/f32 variants already
// live in convert.hpp, so only the integer same-width pairs are added here.
static inline void vx_load_as(const uchar*    p, v_uint8&  v) { v = vx_load(p); }
static inline void vx_load_as(const schar*    p, v_int8&   v) { v = vx_load(p); }
static inline void vx_load_as(const ushort*   p, v_uint16& v) { v = vx_load(p); }
static inline void vx_load_as(const short*    p, v_int16&  v) { v = vx_load(p); }
static inline void vx_load_as(const unsigned* p, v_uint32& v) { v = vx_load(p); }
static inline void vx_load_as(const int*      p, v_int32&  v) { v = vx_load(p); }
static inline void v_store_as(uchar*    p, const v_uint8&  v) { v_store(p, v); }
static inline void v_store_as(schar*    p, const v_int8&   v) { v_store(p, v); }
static inline void v_store_as(ushort*   p, const v_uint16& v) { v_store(p, v); }
static inline void v_store_as(short*    p, const v_int16&  v) { v_store(p, v); }
static inline void v_store_as(unsigned* p, const v_uint32& v) { v_store(p, v); }
static inline void v_store_as(int*      p, const v_int32&  v) { v_store(p, v); }
#endif

namespace ew {

// ===========================================================================
// Op functors (vector + scalar). New binary ops slot in here.
// ===========================================================================
struct EwAdd {
    template<typename V> static V vec(const V& a, const V& b) { return v_add(a, b); }
    template<typename W> static W scl(W a, W b) { return (W)(a + b); }
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
template<typename T0, typename T1, typename Tr, typename Wvec, typename WT, class Op, bool use_simd>
static int binary_kernel(const void* src0_, size_t s0y, size_t s0x,
                         const void* src1_, size_t s1y, size_t s1x,
                         const void*, size_t, size_t,
                         void* dst_, size_t dsty, int width, int height, void*)
{
    CV_Assert(s0x <= 1 && s1x <= 1);
    const T0* src0 = (const T0*)src0_;
    const T1* src1 = (const T1*)src1_;
    Tr* dst = (Tr*)dst_;
    EW_TRY_COLLAPSE(2);
#if (CV_SIMD || CV_SIMD_SCALABLE)
    // The right-edge backoff re-reads already-written lanes, so it is only valid when there is
    // a full vector to back off into (width >= VECSZ) and dst aliases no input. Otherwise the
    // SIMD loop just breaks to the scalar tail (also covers width < VECSZ and in-place).
    [[maybe_unused]] const int VECSZ = VTraits<Wvec>::vlanes();
    [[maybe_unused]] const bool use_tail_trick =
        width >= VECSZ && src0_ != dst_ && src1_ != dst_;
#endif
    for (int i = 0; i < height; i++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
        if constexpr (use_simd)
        {
        #if (CV_SIMD || CV_SIMD_SCALABLE)
            if (s0x == 1 && s1x == 1)
            {
                for (; x < width; x += VECSZ)
                {
                    if (x + VECSZ > width) { if (!use_tail_trick) break; x = width - VECSZ; }
                    Wvec a, b;
                    vx_load_as(src0 + x, a);
                    vx_load_as(src1 + x, b);
                    v_store_as(dst + x, Op::vec(a, b));
                }
            }
        #endif
        }
        for (; x < width; x++)
            dst[x] = saturate_cast<Tr>(Op::scl((WT)src0[x*s0x], (WT)src1[x*s1x]));
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if constexpr (use_simd) vx_cleanup();
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
static ElemwiseFunc getAddFunc(int T, int R)
{
    switch (T)
    {
    case CV_8U:
        if (R == CV_8U)  return binary_kernel<uchar, uchar, uchar, v_uint8,   int,   EwAdd, true>;
        if (R == CV_16S) return binary_kernel<uchar, uchar, short, v_float32, float, EwAdd, true>;
        if (R == CV_32S) return binary_kernel<uchar, uchar, int,   v_float32, float, EwAdd, true>;
        if (R == CV_32F) return binary_kernel<uchar, uchar, float, v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_8S:
        if (R == CV_8S)  return binary_kernel<schar, schar, schar, v_int8,    int,   EwAdd, true>;
        if (R == CV_16S) return binary_kernel<schar, schar, short, v_float32, float, EwAdd, true>;
        if (R == CV_32F) return binary_kernel<schar, schar, float, v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_16U:
        if (R == CV_16U) return binary_kernel<ushort, ushort, ushort, v_uint16,  int,   EwAdd, true>;
        if (R == CV_32S) return binary_kernel<ushort, ushort, int,    v_float32, float, EwAdd, true>;
        if (R == CV_32F) return binary_kernel<ushort, ushort, float,  v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_16S:
        if (R == CV_16S) return binary_kernel<short, short, short, v_int16,   int,   EwAdd, true>;
        if (R == CV_32S) return binary_kernel<short, short, int,   v_float32, float, EwAdd, true>;
        if (R == CV_32F) return binary_kernel<short, short, float, v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_32U:
        if (R == CV_32U) return binary_kernel<unsigned, unsigned, unsigned, v_uint32, unsigned, EwAdd, true>;
        if (R == CV_64S) return binary_kernel<unsigned, unsigned, int64_t,  v_int32,  int64_t,  EwAdd, false>;
        if (R == CV_64F) return binary_kernel<unsigned, unsigned, double,   v_int32,  double,   EwAdd, false>;
        return nullptr;
    case CV_32S:
        if (R == CV_32S) return binary_kernel<int, int, int,     v_int32, int,     EwAdd, true>;
        if (R == CV_64S) return binary_kernel<int, int, int64_t, v_int32, int64_t, EwAdd, false>;
        if (R == CV_64F) return binary_kernel<int, int, double,  v_int32, double,  EwAdd, false>;
        return nullptr;
    case CV_64U:
        if (R == CV_64U) return binary_kernel<uint64_t, uint64_t, uint64_t, v_int32, uint64_t, EwAdd, false>;
        if (R == CV_64F) return binary_kernel<uint64_t, uint64_t, double,   v_int32, double,   EwAdd, false>;
        return nullptr;
    case CV_64S:
        if (R == CV_64S) return binary_kernel<int64_t, int64_t, int64_t, v_int32, int64_t, EwAdd, false>;
        if (R == CV_64F) return binary_kernel<int64_t, int64_t, double,  v_int32, double,  EwAdd, false>;
        return nullptr;
    case CV_16F:
        if (R == CV_16F) return binary_kernel<hfloat, hfloat, hfloat, v_float32, float, EwAdd, true>;
        if (R == CV_32F) return binary_kernel<hfloat, hfloat, float,  v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_16BF:
        if (R == CV_16BF) return binary_kernel<bfloat, bfloat, bfloat, v_float32, float, EwAdd, true>;
        if (R == CV_32F)  return binary_kernel<bfloat, bfloat, float,  v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_32F:
        if (R == CV_32F) return binary_kernel<float, float, float, v_float32, float, EwAdd, true>;
        return nullptr;
    case CV_64F:
        if (R == CV_64F) return binary_kernel<double, double, double, v_int32, double, EwAdd, false>;
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
struct OpSub { static double apply(double a, double b) { return a - b; } };
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
    case OP_SUB: return binaryKernel<OpSub, float, float, float>;
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

    // OP_CAST is always served by the adapter over core's convert kernels; the type-specialized
    // BinaryFunc is selected by the executor when it builds the EwCtx (it has the depths there).
    if (op == OP_CAST)
        return convertAdapter;

    if (op == OP_ADD)
    {
        if (depth0 != depth1) return nullptr;   // operands must be the same type
        return getAddFunc(depth0, rdepth);
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
