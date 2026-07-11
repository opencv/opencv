// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "reduce_c_generic.hpp"

#if CV_RVV
#include "reduce_c_rvv.hpp"
#endif

#if CV_NEON
#include "reduce_c_neon.hpp"
#endif

#if CV_AVX2
#include "reduce_c_avx2.hpp"
#endif

template<bool isMax>
static void reduceColMinMax_8uC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax8uC1<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax8uC1<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax8uC1<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_8uC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax8uC3<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax8uC3<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax8uC3<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_8uC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax8uC4<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax8uC4<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax8uC4<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_8u(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColMinMax_8uC1<isMax>(srcmat, dstmat);
    else if (cn == 3)
        reduceColMinMax_8uC3<isMax>(srcmat, dstmat);
    else if (cn == 4)
        reduceColMinMax_8uC4<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_8uFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_8u(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_8u<true>(srcmat, dstmat);
}

static void reduceColMin_8u(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_8u<false>(srcmat, dstmat);
}

template<bool isMax>
static void reduceColMinMax_32fC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax32fC1<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax32fC1<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax32fC1<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_32fC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax32fC3<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_32fC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax32fC4<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax32fC4<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax32fC4<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColMinMax_32fC1<isMax>(srcmat, dstmat);
    else if (cn == 3)
        reduceColMinMax_32fC3<isMax>(srcmat, dstmat);
    else if (cn == 4)
        reduceColMinMax_32fC4<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_32fFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_32f(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_32f<true>(srcmat, dstmat);
}

static void reduceColMin_32f(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_32f<false>(srcmat, dstmat);
}

template<typename DT>
static void reduceColSum2_8uC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::sum2_8uC1<DT>(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_8uC1<DT>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_8uC1<DT>(srcmat, dstmat);
#else
    reduceColSum2_8uFallback<DT>(srcmat, dstmat);
#endif
}

template<typename DT>
static void reduceColSum2_8uC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::sum2_8uC3<DT>(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_8uC3<DT>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_8uC3<DT>(srcmat, dstmat);
#else
    reduceColSum2_8uFallback<DT>(srcmat, dstmat);
#endif
}

template<typename DT>
static void reduceColSum2_8uC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::sum2_8uC4<DT>(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_8uC4<DT>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_8uC4<DT>(srcmat, dstmat);
#else
    reduceColSum2_8uFallback<DT>(srcmat, dstmat);
#endif
}

template<typename DT>
static void reduceColSum2_8u(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColSum2_8uC1<DT>(srcmat, dstmat);
    else if (cn == 3)
        reduceColSum2_8uC3<DT>(srcmat, dstmat);
    else if (cn == 4)
        reduceColSum2_8uC4<DT>(srcmat, dstmat);
    else
        reduceColSum2_8uFallback<DT>(srcmat, dstmat);
}

static void reduceColSum2_8u32s(const Mat& srcmat, Mat& dstmat)
{
    reduceColSum2_8u<int>(srcmat, dstmat);
}

static void reduceColSum2_8u32f(const Mat& srcmat, Mat& dstmat)
{
    reduceColSum2_8u<float>(srcmat, dstmat);
}

static void reduceColSum2_8u64f(const Mat& srcmat, Mat& dstmat)
{
    reduceColSum2_8u<double>(srcmat, dstmat);
}

static void reduceColSum2_32f32fC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::sum2_32fC1(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::sum2_32fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_32fC1(srcmat, dstmat);
#else
    reduceColSum2_32f32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f32fC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::sum2_32fC3(srcmat, dstmat);
#else
    reduceColSum2_32f32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f32fC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::sum2_32fC4(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::sum2_32fC4(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_32fC4(srcmat, dstmat);
#else
    reduceColSum2_32f32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    if (cn == 1)
        reduceColSum2_32f32fC1(srcmat, dstmat);
    else if (cn == 3)
        reduceColSum2_32f32fC3(srcmat, dstmat);
    else if (cn == 4)
        reduceColSum2_32f32fC4(srcmat, dstmat);
    else
        reduceColSum2_32f32fFallback(srcmat, dstmat);
}
