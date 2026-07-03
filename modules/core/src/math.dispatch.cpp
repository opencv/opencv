// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Dispatch layer for the element-wise MATH + SELECT kernels (math.simd.hpp) - the sibling of
// arithm.dispatch.cpp: plain functions forwarding to the CPU-optimal kernel via CV_CPU_DISPATCH.
// getElemwiseFunc (arithm.dispatch.cpp) routes the corresponding TOps here.

#include "precomp.hpp"
#include "arithm_expr.hpp"
#include "hal_replacement.hpp"
#include "math.simd.hpp"
#include "math.simd_declarations.hpp"

namespace cv { namespace ew {

// ---- pluggable-HAL bridge for exp/log --------------------------------------------------------
// The raw cv_hal_* entry points return int for a reason: without an installed HAL they are stubs
// returning CV_HAL_ERROR_NOT_IMPLEMENTED. getMathFunc PROBES each one once (a 1-element call on
// the safe input 1.0 - fine for both exp and log): implemented -> wrap it as an engine kernel
// (the function pointer rides in TKernel::userdata, castKernel-style) and the engine adds tiling
// and parallelism on top of the vendor code; not implemented -> the engine's own v_exp/v_log
// kernels. Uniform over ANY HAL (an external vendor one, the IPP HAL module, ...) - the get is
// called once per program build, the probe cost is nothing next to the kernel calls that follow.
template<typename T>
static int halUnaryKernel(const void* src0_, size_t s0y, size_t s0x,
                          const void*, size_t, size_t, const void*, size_t, size_t,
                          void* dst_, size_t dsty, int width, int height,
                          const double*, int, void* userdata)
{
    typedef int (*HalFunc)(const T*, T*, int);
    const HalFunc fn = (HalFunc)userdata;
    s0y /= sizeof(T);
    dsty /= sizeof(T);
    CV_Assert(s0x <= 1u);
    const T* src0 = (const T*)src0_;
    T* dst = (T*)dst_;
    if (height > 1 && dsty == (size_t)width && s0y == s0x*(size_t)width) { width *= height; height = 1; }
    const int urows = (s0y == 0 && height > 1) ? 1 : height;   // vertical broadcast: 1 row + copies
    for (int y = 0; y < urows; y++, src0 += s0y, dst += dsty)
    {
        int code;
        if (s0x == 0)                     // broadcast-scalar source: one value covers the row
        {
            T v;
            code = fn(src0, &v, 1);
            for (int x = 0; x < width; x++) dst[x] = v;
        }
        else
            code = fn(src0, dst, width);
        if (code != CV_HAL_ERROR_OK)
            return code;                  // shouldn't happen (probed at get time) - let exec assert
    }
    dst = (T*)dst_;
    for (int y = urows; y < height; y++)
        memcpy(dst + (size_t)y*dsty, dst, (size_t)width*sizeof(T));
    return 0;
}

template<typename T, typename HalFunc>
static TKernel probeHalUnary(HalFunc fn)
{
    T one = (T)1, r = (T)0;
    if (fn(&one, &r, 1) == CV_HAL_ERROR_OK)
        return {halUnaryKernel<T>, (void*)fn, 0};
    return {};
}

#ifdef HAVE_IPP
// IPP is not (yet) routed through the cv_hal_* hooks, so it gets its own tier: thin adapters give
// the ipps calls the HalFunc shape and ride the same halUnaryKernel.
static int ippExp32(const float* s, float* d, int n)  { return CV_INSTRUMENT_FUN_IPP(ippsExp_32f_A21, s, d, n) >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_UNKNOWN; }
static int ippExp64(const double* s, double* d, int n){ return CV_INSTRUMENT_FUN_IPP(ippsExp_64f_A50, s, d, n) >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_UNKNOWN; }
static int ippLog32(const float* s, float* d, int n)  { return CV_INSTRUMENT_FUN_IPP(ippsLn_32f_A21,  s, d, n) >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_UNKNOWN; }
static int ippLog64(const double* s, double* d, int n){ return CV_INSTRUMENT_FUN_IPP(ippsLn_64f_A50,  s, d, n) >= 0 ? CV_HAL_ERROR_OK : CV_HAL_ERROR_UNKNOWN; }
#endif

// The engine's OWN kernel for (op, T) - v_exp/v_log & co, no HAL/IPP tiers. The final fallback of
// getMathFunc, and what hal::exp32f & co use as THEIR built-in implementation (the former table
// kernels are gone), via mathSpanEngine below.
static TKernel getEngineMathFunc(TOp op, int T)
{
    CV_CPU_DISPATCH(getMathFunc_,   (op, T),      CV_CPU_DISPATCH_MODES_ALL);
}

// run the engine's own math kernel over one contiguous span (the shape hal::exp32f & co need)
void mathSpanEngine(TOp op, int depth, const void* src, void* dst, int n)
{
    TKernel k = getEngineMathFunc(op, depth);
    CV_Assert(k.fptr);
    const double noparams[4] = {};
    k.fptr(src, 0, 1, nullptr, 0, 0, nullptr, 0, 0, dst, 0, n, 1, noparams, k.flags, k.userdata);
}

TKernel getMathFunc(TOp op, int T)
{
    if ((op == OP_EXP || op == OP_LOG) && (T == CV_32F || T == CV_64F))
    {
#ifdef HAVE_IPP
        if (ipp::useIPP())   // checked per get (cheap; a program build, not a kernel call)
        {
            if (op == OP_EXP)
                return T == CV_64F ? TKernel{halUnaryKernel<double>, (void*)ippExp64, 0}
                                   : TKernel{halUnaryKernel<float>,  (void*)ippExp32, 0};
            return T == CV_64F ? TKernel{halUnaryKernel<double>, (void*)ippLog64, 0}
                               : TKernel{halUnaryKernel<float>,  (void*)ippLog32, 0};
        }
#endif
        // probe results are process-lifetime stable; cache them (thread-safe magic statics)
        static const TKernel exp32 = probeHalUnary<float >(cv_hal_exp32f);
        static const TKernel exp64 = probeHalUnary<double>(cv_hal_exp64f);
        static const TKernel log32 = probeHalUnary<float >(cv_hal_log32f);
        static const TKernel log64 = probeHalUnary<double>(cv_hal_log64f);
        const TKernel* k = op == OP_EXP ? (T == CV_32F ? &exp32 : &exp64)
                                        : (T == CV_32F ? &log32 : &log64);
        if (k->fptr)
            return *k;
    }
    return getEngineMathFunc(op, T);
}

TKernel getPowFunc(int T, int R)            { CV_CPU_DISPATCH(getPowFunc_,    (T, R),       CV_CPU_DISPATCH_MODES_ALL); }

}} // namespace cv::ew
