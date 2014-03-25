/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Fangfang Bai, fangfang@multicorewareinc.com
//    Jin Ma,       jin@multicorewareinc.com
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "perf_precomp.hpp"
#include "opencv2/ts/ocl_perf.hpp"

#ifdef HAVE_OPENCL

namespace cvtest {
namespace ocl {

typedef tuple<Size, MatType, int> FilterParams;
typedef TestBaseWithParam<FilterParams> FilterFixture;

///////////// Blur ////////////////////////

typedef FilterFixture BlurFixture;

OCL_PERF_TEST_P(BlurFixture, Blur,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params), bordertype = BORDER_CONSTANT;
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::blur(src, dst, Size(ksize, ksize), Point(-1, -1), bordertype);

    SANITY_CHECK(dst, eps);
}

///////////// SqrBoxFilter ////////////////////////

typedef tuple<Size, MatType, Size> SqrBoxFilterParams;
typedef TestBaseWithParam<SqrBoxFilterParams> SqrBoxFilterFixture;

OCL_PERF_TEST_P(SqrBoxFilterFixture, SqrBoxFilter,
                ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4),
                                   OCL_PERF_ENUM(Size(3, 3), Size(20, 3), Size(3, 20), Size(20, 20))))
{
    const SqrBoxFilterParams params = GetParam();
    const Size srcSize = get<0>(params), ksize = get<2>(params);
    const int type = get<1>(params), depth = CV_MAT_DEPTH(type),
            ddepth = depth == CV_8U ? CV_32S : CV_32F;
    const double eps = ddepth == CV_32S ? 0 : 5e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, CV_MAKE_TYPE(ddepth, CV_MAT_CN(type)));

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::sqrBoxFilter(src, dst, ddepth, ksize, Point(-1, -1), false);

    SANITY_CHECK(dst, eps);
}

///////////// Laplacian////////////////////////

typedef FilterFixture LaplacianFixture;

OCL_PERF_TEST_P(LaplacianFixture, Laplacian,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::Laplacian(src, dst, -1, ksize, 1);

    SANITY_CHECK(dst, eps);
}

///////////// Erode ////////////////////

typedef FilterFixture ErodeFixture;

OCL_PERF_TEST_P(ErodeFixture, Erode,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);
    const Mat ker = getStructuringElement(MORPH_RECT, Size(ksize, ksize));

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    OCL_TEST_CYCLE() cv::erode(src, dst, ker);

    SANITY_CHECK(dst);
}

///////////// Dilate ////////////////////

typedef FilterFixture DilateFixture;

OCL_PERF_TEST_P(DilateFixture, Dilate,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);
    const Mat ker = getStructuringElement(MORPH_RECT, Size(ksize, ksize));

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    OCL_TEST_CYCLE() cv::dilate(src, dst, ker);

    SANITY_CHECK(dst);
}

///////////// MorphologyEx ////////////////////////

CV_ENUM(MorphOp, MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT)

typedef tuple<Size, MatType, MorphOp, int> MorphologyExParams;
typedef TestBaseWithParam<MorphologyExParams> MorphologyExFixture;

OCL_PERF_TEST_P(MorphologyExFixture, MorphologyEx,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, MorphOp::all(), OCL_PERF_ENUM(3, 5)))
{
    const MorphologyExParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), op = get<2>(params), ksize = get<3>(params);
    const Mat ker = getStructuringElement(MORPH_RECT, Size(ksize, ksize));

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    OCL_TEST_CYCLE() cv::morphologyEx(src, dst, op, ker);

    SANITY_CHECK(dst);
}

///////////// Sobel ////////////////////////

typedef Size_MatType SobelFixture;

OCL_PERF_TEST_P(SobelFixture, Sobel,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), dx = 1, dy = 1;

    checkDeviceMaxMemoryAllocSize(srcSize, type, sizeof(float) * 2);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::Sobel(src, dst, -1, dx, dy);

    SANITY_CHECK(dst, 1e-6);
}

///////////// Scharr ////////////////////////

typedef Size_MatType ScharrFixture;

OCL_PERF_TEST_P(ScharrFixture, Scharr,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), dx = 1, dy = 0;
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type, sizeof(float) * 2);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::Scharr(src, dst, -1, dx, dy);

    SANITY_CHECK(dst, eps);
}

///////////// GaussianBlur ////////////////////////

typedef FilterFixture GaussianBlurFixture;

OCL_PERF_TEST_P(GaussianBlurFixture, GaussianBlur,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5, 7)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 + DBL_EPSILON : 3e-4;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::GaussianBlur(src, dst, Size(ksize, ksize), 0);

    SANITY_CHECK(dst, eps);
}

///////////// Filter2D ////////////////////////

typedef FilterFixture Filter2DFixture;

OCL_PERF_TEST_P(Filter2DFixture, Filter2D,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);
    const double eps = CV_MAT_DEPTH(type) <= CV_32S ? 1 : 1e-5;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, type), kernel(ksize, ksize, CV_32SC1);
    declare.in(src, WARMUP_RNG).in(kernel).out(dst);
    randu(kernel, -3.0, 3.0);

    OCL_TEST_CYCLE() cv::filter2D(src, dst, -1, kernel);

    SANITY_CHECK(dst, eps);
}

///////////// Bilateral ////////////////////////

typedef TestBaseWithParam<Size> BilateralFixture;

OCL_PERF_TEST_P(BilateralFixture, Bilateral, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();
    const int d = 7;
    const double sigmacolor = 50.0, sigmaspace = 50.0;

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    UMat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::bilateralFilter(src, dst, d, sigmacolor, sigmaspace);

    SANITY_CHECK(dst);
}

///////////// MedianBlur ////////////////////////

typedef tuple<Size, int> MedianBlurParams;
typedef TestBaseWithParam<MedianBlurParams> MedianBlurFixture;

OCL_PERF_TEST_P(MedianBlurFixture, Bilateral, ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(3, 5)))
{
    MedianBlurParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int ksize = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    UMat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::medianBlur(src, dst, ksize);

    SANITY_CHECK(dst);
}

} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
