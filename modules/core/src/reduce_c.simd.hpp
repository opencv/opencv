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
static void reduceColMinMax_16uC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax16uC1<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax16uC1<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax16uC1<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_16uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_16uC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax16uC4<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax16uC4<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax16uC4<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_16uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_16uC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax16uC3<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax16uC3<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax16uC3<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_16uFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_16u(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() == 1)
        reduceColMinMax_16uC1<isMax>(srcmat, dstmat);
    else if (srcmat.channels() == 3)
        reduceColMinMax_16uC3<isMax>(srcmat, dstmat);
    else if (srcmat.channels() == 4)
        reduceColMinMax_16uC4<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_16uFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_16u(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_16u<true>(srcmat, dstmat);
}

static void reduceColMin_16u(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_16u<false>(srcmat, dstmat);
}

template<bool isMax>
static void reduceColMinMax_16sC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax16sC1<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax16sC1<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax16sC1<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_16sFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_16sC4(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax16sC4<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax16sC4<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax16sC4<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_16sFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_16sC3(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV
    reduce_c_rvv::minMax16sC3<isMax>(srcmat, dstmat);
#elif CV_NEON
    reduce_c_neon::minMax16sC3<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax16sC3<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_16sFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_16s(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() == 1)
        reduceColMinMax_16sC1<isMax>(srcmat, dstmat);
    else if (srcmat.channels() == 3)
        reduceColMinMax_16sC3<isMax>(srcmat, dstmat);
    else if (srcmat.channels() == 4)
        reduceColMinMax_16sC4<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_16sFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_16s(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_16s<true>(srcmat, dstmat);
}

static void reduceColMin_16s(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_16s<false>(srcmat, dstmat);
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

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
template<bool isMax>
static void reduceColMinMax_64fC1(const Mat& srcmat, Mat& dstmat)
{
#if CV_RVV && CV_SIMD_SCALABLE_64F
    reduce_c_rvv::minMax64fC1<isMax>(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::minMax64fC1<isMax>(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::minMax64fC1<isMax>(srcmat, dstmat);
#else
    reduceColMinMax_64fFallback<isMax>(srcmat, dstmat);
#endif
}

template<bool isMax>
static void reduceColMinMax_64f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() == 1)
        reduceColMinMax_64fC1<isMax>(srcmat, dstmat);
    else
        reduceColMinMax_64fFallback<isMax>(srcmat, dstmat);
}

static void reduceColMax_64f(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_64f<true>(srcmat, dstmat);
}

static void reduceColMin_64f(const Mat& srcmat, Mat& dstmat)
{
    reduceColMinMax_64f<false>(srcmat, dstmat);
}
#endif

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

static void reduceColSum2_16u32f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() != 1)
    {
        reduceColSum2_16u32fFallback(srcmat, dstmat);
        return;
    }
#if CV_RVV
    reduce_c_rvv::sum2_16u32fC1(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_16u32fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_16u32fC1(srcmat, dstmat);
#else
    reduceColSum2_16u32fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_16s32f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() != 1)
    {
        reduceColSum2_16s32fFallback(srcmat, dstmat);
        return;
    }
#if CV_RVV
    reduce_c_rvv::sum2_16s32fC1(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_16s32fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_16s32fC1(srcmat, dstmat);
#else
    reduceColSum2_16s32fFallback(srcmat, dstmat);
#endif
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

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)
static void reduceColSum2_16u64f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() != 1)
    {
        reduceColSum2_16u64fFallback(srcmat, dstmat);
        return;
    }
#if CV_RVV && CV_SIMD_SCALABLE_64F
    reduce_c_rvv::sum2_16u64fC1(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_16u64fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_16u64fC1(srcmat, dstmat);
#else
    reduceColSum2_16u64fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_16s64f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() != 1)
    {
        reduceColSum2_16s64fFallback(srcmat, dstmat);
        return;
    }
#if CV_RVV && CV_SIMD_SCALABLE_64F
    reduce_c_rvv::sum2_16s64fC1(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_16s64fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_16s64fC1(srcmat, dstmat);
#else
    reduceColSum2_16s64fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_32f64f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() != 1)
    {
        reduceColSum2_32f64fFallback(srcmat, dstmat);
        return;
    }
#if CV_RVV && CV_SIMD_SCALABLE_64F
    reduce_c_rvv::sum2_32f64fC1(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_32f64fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_32f64fC1(srcmat, dstmat);
#else
    reduceColSum2_32f64fFallback(srcmat, dstmat);
#endif
}

static void reduceColSum2_64f64f(const Mat& srcmat, Mat& dstmat)
{
    if (srcmat.channels() != 1)
    {
        reduceColSum2_64f64fFallback(srcmat, dstmat);
        return;
    }
#if CV_RVV && CV_SIMD_SCALABLE_64F
    reduce_c_rvv::sum2_64f64fC1(srcmat, dstmat);
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
    reduce_c_neon::sum2_64f64fC1(srcmat, dstmat);
#elif CV_AVX2
    reduce_c_avx2::sum2_64f64fC1(srcmat, dstmat);
#else
    reduceColSum2_64f64fFallback(srcmat, dstmat);
#endif
}
#endif
