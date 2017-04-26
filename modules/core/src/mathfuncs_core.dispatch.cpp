// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "mathfuncs_core.simd.hpp"
#include "mathfuncs_core.simd_declarations.hpp" // defines CV_CPU_DISPATCH_MODES_ALL=AVX2,...,BASELINE based on CMakeLists.txt content

namespace cv { namespace hal {

///////////////////////////////////// ATAN2 ////////////////////////////////////

void fastAtan32f(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(fastAtan32f, cv_hal_fastAtan32f, Y, X, angle, len, angleInDegrees);

    CV_CPU_DISPATCH(fastAtan32f, (Y, X, angle, len, angleInDegrees),
        CV_CPU_DISPATCH_MODES_ALL);
}

void fastAtan64f(const double *Y, const double *X, double *angle, int len, bool angleInDegrees)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(fastAtan64f, cv_hal_fastAtan64f, Y, X, angle, len, angleInDegrees);

    CV_CPU_DISPATCH(fastAtan64f, (Y, X, angle, len, angleInDegrees),
        CV_CPU_DISPATCH_MODES_ALL);
}

// deprecated
void fastAtan2(const float *Y, const float *X, float *angle, int len, bool angleInDegrees )
{
    CV_INSTRUMENT_REGION()

    fastAtan32f(Y, X, angle, len, angleInDegrees);
}

void magnitude32f(const float* x, const float* y, float* mag, int len)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(magnitude32f, cv_hal_magnitude32f, x, y, mag, len);
    CV_IPP_RUN(!IPP_DISABLE_PERF_MAG_SSE42 || (ipp::getIppFeatures()&ippCPUID_AVX), CV_INSTRUMENT_FUN_IPP(ippsMagnitude_32f, x, y, mag, len) >= 0);

    CV_CPU_DISPATCH(magnitude32f, (x, y, mag, len),
        CV_CPU_DISPATCH_MODES_ALL);
}

void magnitude64f(const double* x, const double* y, double* mag, int len)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(magnitude64f, cv_hal_magnitude64f, x, y, mag, len);
    CV_IPP_RUN(!IPP_DISABLE_PERF_MAG_SSE42 || (ipp::getIppFeatures()&ippCPUID_AVX), CV_INSTRUMENT_FUN_IPP(ippsMagnitude_64f, x, y, mag, len) >= 0);

    CV_CPU_DISPATCH(magnitude64f, (x, y, mag, len),
        CV_CPU_DISPATCH_MODES_ALL);
}


void invSqrt32f(const float* src, float* dst, int len)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(invSqrt32f, cv_hal_invSqrt32f, src, dst, len);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsInvSqrt_32f_A21, src, dst, len) >= 0);

    CV_CPU_DISPATCH(invSqrt32f, (src, dst, len),
        CV_CPU_DISPATCH_MODES_ALL);
}


void invSqrt64f(const double* src, double* dst, int len)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(invSqrt64f, cv_hal_invSqrt64f, src, dst, len);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsInvSqrt_64f_A50, src, dst, len) >= 0);

    CV_CPU_DISPATCH(invSqrt64f, (src, dst, len),
        CV_CPU_DISPATCH_MODES_ALL);
}


void sqrt32f(const float* src, float* dst, int len)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(sqrt32f, cv_hal_sqrt32f, src, dst, len);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsSqrt_32f_A21, src, dst, len) >= 0);

    CV_CPU_DISPATCH(sqrt32f, (src, dst, len),
        CV_CPU_DISPATCH_MODES_ALL);
}


void sqrt64f(const double* src, double* dst, int len)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(sqrt64f, cv_hal_sqrt64f, src, dst, len);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsSqrt_64f_A50, src, dst, len) >= 0);

    CV_CPU_DISPATCH(sqrt64f, (src, dst, len),
        CV_CPU_DISPATCH_MODES_ALL);
}

void exp32f(const float *src, float *dst, int n)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(exp32f, cv_hal_exp32f, src, dst, n);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsExp_32f_A21, src, dst, n) >= 0);

    CV_CPU_DISPATCH(exp32f, (src, dst, n),
        CV_CPU_DISPATCH_MODES_ALL);
}

void exp64f(const double *src, double *dst, int n)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(exp64f, cv_hal_exp64f, src, dst, n);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsExp_64f_A50, src, dst, n) >= 0);

    CV_CPU_DISPATCH(exp64f, (src, dst, n),
        CV_CPU_DISPATCH_MODES_ALL);
}

void log32f(const float *src, float *dst, int n)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(log32f, cv_hal_log32f, src, dst, n);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsLn_32f_A21, src, dst, n) >= 0);

    CV_CPU_DISPATCH(log32f, (src, dst, n),
        CV_CPU_DISPATCH_MODES_ALL);
}

void log64f(const double *src, double *dst, int n)
{
    CV_INSTRUMENT_REGION()

    CALL_HAL(log64f, cv_hal_log64f, src, dst, n);
    CV_IPP_RUN_FAST(CV_INSTRUMENT_FUN_IPP(ippsLn_64f_A50, src, dst, n) >= 0);

    CV_CPU_DISPATCH(log64f, (src, dst, n),
        CV_CPU_DISPATCH_MODES_ALL);
}

//=============================================================================
// for compatibility with 3.0

void exp(const float* src, float* dst, int n)
{
    exp32f(src, dst, n);
}

void exp(const double* src, double* dst, int n)
{
    exp64f(src, dst, n);
}

void log(const float* src, float* dst, int n)
{
    log32f(src, dst, n);
}

void log(const double* src, double* dst, int n)
{
    log64f(src, dst, n);
}

void magnitude(const float* x, const float* y, float* dst, int n)
{
    magnitude32f(x, y, dst, n);
}

void magnitude(const double* x, const double* y, double* dst, int n)
{
    magnitude64f(x, y, dst, n);
}

void sqrt(const float* src, float* dst, int len)
{
    sqrt32f(src, dst, len);
}

void sqrt(const double* src, double* dst, int len)
{
    sqrt64f(src, dst, len);
}

void invSqrt(const float* src, float* dst, int len)
{
    invSqrt32f(src, dst, len);
}

void invSqrt(const double* src, double* dst, int len)
{
    invSqrt64f(src, dst, len);
}

}} // namespace cv::hal::

float cv::fastAtan2( float y, float x )
{
    using namespace cv::hal;
    CV_CPU_CALL_BASELINE(fastAtan2, (y, x));
}
