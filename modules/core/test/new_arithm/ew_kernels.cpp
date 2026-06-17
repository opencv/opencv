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

#include "ew_op.hpp"
#include "opencv2/core/hal/intrin.hpp"
// private prototype-only reuse of the typed load/store-as helpers (vx_load_as / v_store_as):
#include "../../src/convert.hpp"
#include <cmath>

namespace cv { namespace ew {

// ===========================================================================
// OP_ADD
// ===========================================================================

// Collapse a gap-free 2D tile to 1D (call with the per-operand x-steps and y-steps).
#define EW_TRY_COLLAPSE(NSRC) \
    if (height > 1 && dsty == (size_t)width && \
        s0y == s0x*(size_t)width && (NSRC < 2 || s1y == s1x*(size_t)width)) \
    { width *= height; height = 1; }

// Same-type add: T x T -> T. Native v_add (saturating for 8/16-bit, wrapping for 32/64-bit).
template<typename T, typename Tvec, typename WT>
static int add_same(const void* src0_, size_t s0y, size_t s0x,
                    const void* src1_, size_t s1y, size_t s1x,
                    const void*, size_t, size_t,
                    void* dst_, size_t dsty, int width, int height)
{
    CV_Assert(s0x <= 1 && s1x <= 1);
    const T* src0 = (const T*)src0_;
    const T* src1 = (const T*)src1_;
    T* dst = (T*)dst_;
    EW_TRY_COLLAPSE(2);
    const bool simd = (s0x == 1 && s1x == 1);
    for (int i = 0; i < height; i++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (simd)
        {
            const int VECSZ = VTraits<Tvec>::vlanes();
            for (; x < width; x += VECSZ)
            {
                if (x + VECSZ > width) { if (x == 0) break; x = width - VECSZ; }
                v_store(dst + x, v_add(vx_load(src0 + x), vx_load(src1 + x)));
            }
        }
#endif
        for (; x < width; x++)
            dst[x] = saturate_cast<T>((WT)src0[x*s0x] + (WT)src1[x*s1x]);
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    vx_cleanup();
#endif
    return 0;
}

// Widening add through the f32 hub: Tin x Tin -> Tr (Tin <= 16-bit or half/float; exact in f32).
template<typename Tin, typename Tr>
static int add_widen32(const void* src0_, size_t s0y, size_t s0x,
                       const void* src1_, size_t s1y, size_t s1x,
                       const void*, size_t, size_t,
                       void* dst_, size_t dsty, int width, int height)
{
    CV_Assert(s0x <= 1 && s1x <= 1);
    const Tin* src0 = (const Tin*)src0_;
    const Tin* src1 = (const Tin*)src1_;
    Tr* dst = (Tr*)dst_;
    EW_TRY_COLLAPSE(2);
    const bool simd = (s0x == 1 && s1x == 1);
    for (int i = 0; i < height; i++, src0 += s0y, src1 += s1y, dst += dsty)
    {
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (simd)
        {
            const int VECSZ = VTraits<v_float32>::vlanes();
            for (; x < width; x += VECSZ)
            {
                if (x + VECSZ > width) { if (x == 0) break; x = width - VECSZ; }
                v_float32 a, b;
                vx_load_as(src0 + x, a);
                vx_load_as(src1 + x, b);
                v_store_as(dst + x, v_add(a, b));
            }
        }
#endif
        for (; x < width; x++)
            dst[x] = saturate_cast<Tr>((float)src0[x*s0x] + (float)src1[x*s1x]);
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    vx_cleanup();
#endif
    return 0;
}

// Scalar add (correct, not vectorized): used for 32/64-bit widened outputs and f64.
template<typename Tin, typename Tr, typename WT>
static int add_scalar(const void* src0_, size_t s0y, size_t s0x,
                      const void* src1_, size_t s1y, size_t s1x,
                      const void*, size_t, size_t,
                      void* dst_, size_t dsty, int width, int height)
{
    CV_Assert(s0x <= 1 && s1x <= 1);
    const Tin* src0 = (const Tin*)src0_;
    const Tin* src1 = (const Tin*)src1_;
    Tr* dst = (Tr*)dst_;
    EW_TRY_COLLAPSE(2);
    for (int i = 0; i < height; i++, src0 += s0y, src1 += s1y, dst += dsty)
        for (int x = 0; x < width; x++)
            dst[x] = saturate_cast<Tr>((WT)src0[x*s0x] + (WT)src1[x*s1x]);
    return 0;
}

// add dispatch: T x T -> R (operands already same depth T).
static ElemwiseFunc getAddFunc(int T, int R)
{
    switch (T)
    {
    case CV_8U:
        if (R == CV_8U)  return add_same<uchar, v_uint8, int>;
        if (R == CV_16S) return add_widen32<uchar, short>;
        if (R == CV_32S) return add_widen32<uchar, int>;
        if (R == CV_32F) return add_widen32<uchar, float>;
        return nullptr;
    case CV_8S:
        if (R == CV_8S)  return add_same<schar, v_int8, int>;
        if (R == CV_16S) return add_widen32<schar, short>;
        if (R == CV_32F) return add_widen32<schar, float>;
        return nullptr;
    case CV_16U:
        if (R == CV_16U) return add_same<ushort, v_uint16, int>;
        if (R == CV_32S) return add_widen32<ushort, int>;
        if (R == CV_32F) return add_widen32<ushort, float>;
        return nullptr;
    case CV_16S:
        if (R == CV_16S) return add_same<short, v_int16, int>;
        if (R == CV_32S) return add_widen32<short, int>;
        if (R == CV_32F) return add_widen32<short, float>;
        return nullptr;
    case CV_32U:
        if (R == CV_32U) return add_same<unsigned, v_uint32, unsigned>;
        if (R == CV_64S) return add_scalar<unsigned, int64_t, int64_t>;
        if (R == CV_64F) return add_scalar<unsigned, double, double>;
        return nullptr;
    case CV_32S:
        if (R == CV_32S) return add_same<int, v_int32, int>;
        if (R == CV_64S) return add_scalar<int, int64_t, int64_t>;
        if (R == CV_64F) return add_scalar<int, double, double>;
        return nullptr;
    case CV_64U:
        if (R == CV_64U) return add_scalar<uint64_t, uint64_t, uint64_t>;
        if (R == CV_64F) return add_scalar<uint64_t, double, double>;
        return nullptr;
    case CV_64S:
        if (R == CV_64S) return add_scalar<int64_t, int64_t, int64_t>;
        if (R == CV_64F) return add_scalar<int64_t, double, double>;
        return nullptr;
    case CV_16F:
        if (R == CV_16F) return add_widen32<hfloat, hfloat>;
        if (R == CV_32F) return add_widen32<hfloat, float>;
        return nullptr;
    case CV_16BF:
        if (R == CV_16BF) return add_widen32<bfloat, bfloat>;
        if (R == CV_32F)  return add_widen32<bfloat, float>;
        return nullptr;
    case CV_32F:
        if (R == CV_32F) return add_same<float, v_float32, float>;
        return nullptr;
    case CV_64F:
        if (R == CV_64F) return add_scalar<double, double, double>;
        return nullptr;
    default:
        return nullptr;
    }
}

// ===========================================================================
// OP_CAST
// ===========================================================================

template<typename Ts, typename Td>
static int cast_scalar(const void* src0_, size_t s0y, size_t s0x,
                       const void*, size_t, size_t, const void*, size_t, size_t,
                       void* dst_, size_t dsty, int width, int height)
{
    CV_Assert(s0x <= 1);
    const Ts* src0 = (const Ts*)src0_;
    Td* dst = (Td*)dst_;
    size_t s1x = 0, s1y = 0; (void)s1x; (void)s1y;
    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width)
    { width *= height; height = 1; }
    for (int i = 0; i < height; i++, src0 += s0y, dst += dsty)
        for (int x = 0; x < width; x++)
            dst[x] = saturate_cast<Td>(src0[x*s0x]);
    return 0;
}

// SIMD cast through the f32 hub (source exact-in-f32, dst has a v_store_as overload).
template<typename Ts, typename Td>
static int cast_f32(const void* src0_, size_t s0y, size_t s0x,
                    const void*, size_t, size_t, const void*, size_t, size_t,
                    void* dst_, size_t dsty, int width, int height)
{
    CV_Assert(s0x <= 1);
    const Ts* src0 = (const Ts*)src0_;
    Td* dst = (Td*)dst_;
    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width)
    { width *= height; height = 1; }
    const bool simd = (s0x == 1);
    for (int i = 0; i < height; i++, src0 += s0y, dst += dsty)
    {
        int x = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        if (simd)
        {
            const int VECSZ = VTraits<v_float32>::vlanes();
            for (; x < width; x += VECSZ)
            {
                if (x + VECSZ > width) { if (x == 0) break; x = width - VECSZ; }
                v_float32 a;
                vx_load_as(src0 + x, a);
                v_store_as(dst + x, a);
            }
        }
#endif
        for (; x < width; x++)
            dst[x] = saturate_cast<Td>(src0[x*s0x]);
    }
#if (CV_SIMD || CV_SIMD_SCALABLE)
    vx_cleanup();
#endif
    return 0;
}

template<typename Ts>
static ElemwiseFunc castScalarDst(int dd)
{
    switch (dd)
    {
    case CV_8U:   return cast_scalar<Ts, uchar>;
    case CV_8S:   return cast_scalar<Ts, schar>;
    case CV_16U:  return cast_scalar<Ts, ushort>;
    case CV_16S:  return cast_scalar<Ts, short>;
    case CV_32U:  return cast_scalar<Ts, unsigned>;
    case CV_32S:  return cast_scalar<Ts, int>;
    case CV_64U:  return cast_scalar<Ts, uint64_t>;
    case CV_64S:  return cast_scalar<Ts, int64_t>;
    case CV_32F:  return cast_scalar<Ts, float>;
    case CV_64F:  return cast_scalar<Ts, double>;
    case CV_16F:  return cast_scalar<Ts, hfloat>;
    case CV_16BF: return cast_scalar<Ts, bfloat>;
    default:      return nullptr;
    }
}

static ElemwiseFunc getCastScalar(int sd, int dd)
{
    switch (sd)
    {
    case CV_8U:   return castScalarDst<uchar>(dd);
    case CV_8S:   return castScalarDst<schar>(dd);
    case CV_16U:  return castScalarDst<ushort>(dd);
    case CV_16S:  return castScalarDst<short>(dd);
    case CV_32U:  return castScalarDst<unsigned>(dd);
    case CV_32S:  return castScalarDst<int>(dd);
    case CV_64U:  return castScalarDst<uint64_t>(dd);
    case CV_64S:  return castScalarDst<int64_t>(dd);
    case CV_32F:  return castScalarDst<float>(dd);
    case CV_64F:  return castScalarDst<double>(dd);
    case CV_16F:  return castScalarDst<hfloat>(dd);
    case CV_16BF: return castScalarDst<bfloat>(dd);
    default:      return nullptr;
    }
}

// Source depths that are represented exactly when widened to f32 (<=16-bit or float).
static bool srcExactInF32(int sd)
{
    return sd == CV_8U || sd == CV_8S || sd == CV_16U || sd == CV_16S ||
           sd == CV_16F || sd == CV_16BF || sd == CV_32F;
}

template<typename Ts>
static ElemwiseFunc castSimdDst(int dd)
{
    switch (dd)   // only depths with a v_store_as(.., v_float32) overload
    {
    case CV_16U:  return cast_f32<Ts, ushort>;
    case CV_16S:  return cast_f32<Ts, short>;
    case CV_32U:  return cast_f32<Ts, unsigned>;
    case CV_32S:  return cast_f32<Ts, int>;
    case CV_32F:  return cast_f32<Ts, float>;
    case CV_16F:  return cast_f32<Ts, hfloat>;
    case CV_16BF: return cast_f32<Ts, bfloat>;
    default:      return nullptr;   // u8/s8/64-bit dst -> scalar
    }
}

static ElemwiseFunc getCastSimd(int sd, int dd)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if (!srcExactInF32(sd)) return nullptr;
    switch (sd)
    {
    case CV_8U:   return castSimdDst<uchar>(dd);
    case CV_8S:   return castSimdDst<schar>(dd);
    case CV_16U:  return castSimdDst<ushort>(dd);
    case CV_16S:  return castSimdDst<short>(dd);
    case CV_16F:  return castSimdDst<hfloat>(dd);
    case CV_16BF: return castSimdDst<bfloat>(dd);
    case CV_32F:  return castSimdDst<float>(dd);
    default:      return nullptr;
    }
#else
    (void)sd; (void)dd; return nullptr;
#endif
}

static ElemwiseFunc getCastFunc(int sd, int dd)
{
    ElemwiseFunc f = getCastSimd(sd, dd);
    return f ? f : getCastScalar(sd, dd);
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
                        void* dst, size_t dsty, int width, int height)
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

    if (op == OP_CAST)
        return getCastFunc(depth0, rdepth);

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

}} // namespace cv::ew
