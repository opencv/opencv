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

using namespace perf;
using std::tr1::get;
using std::tr1::tuple;

typedef tuple<Size, MatType, int> FilterParams;
typedef TestBaseWithParam<FilterParams> FilterFixture;

///////////// Blur////////////////////////

typedef FilterFixture BlurFixture;

OCL_PERF_TEST_P(BlurFixture, Blur,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params), bordertype = BORDER_CONSTANT;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::blur(oclSrc, oclDst, Size(ksize, ksize), Point(-1, -1), bordertype);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::blur(src, dst, Size(ksize, ksize), Point(-1, -1), bordertype);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}

///////////// Laplacian////////////////////////

typedef FilterFixture LaplacianFixture;

OCL_PERF_TEST_P(LaplacianFixture, Laplacian,
                ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(1, 3)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::Laplacian(oclSrc, oclDst, -1, ksize, 1);

        oclDst.download(dst);

        SANITY_CHECK(dst, 5e-3);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::Laplacian(src, dst, -1, ksize, 1);

        SANITY_CHECK(dst, 5e-3);
    }
    else
        OCL_PERF_ELSE
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

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type), oclKer(ker);

        OCL_TEST_CYCLE() cv::ocl::erode(oclSrc, oclDst, oclKer);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::erode(src, dst, ker);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
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

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type), oclKer(ker);

        OCL_TEST_CYCLE() cv::ocl::dilate(oclSrc, oclDst, oclKer);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::dilate(src, dst, ker);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// MorphologyEx ////////////////////

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

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type), oclKer(ker);

        OCL_TEST_CYCLE() cv::ocl::morphologyEx(oclSrc, oclDst, op, oclKer);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::morphologyEx(src, dst, op, ker);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
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

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::Sobel(oclSrc, oclDst, -1, dx, dy);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-3);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::Sobel(src, dst, -1, dx, dy);

        SANITY_CHECK(dst, 1e-3);
    }
    else
        OCL_PERF_ELSE
}

///////////// Scharr ////////////////////////

typedef Size_MatType ScharrFixture;

OCL_PERF_TEST_P(ScharrFixture, Scharr,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), dx = 1, dy = 0;

    checkDeviceMaxMemoryAllocSize(srcSize, type, sizeof(float) * 2);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::Scharr(oclSrc, oclDst, -1, dx, dy);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-2);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::Scharr(src, dst, -1, dx, dy);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// GaussianBlur ////////////////////////

typedef FilterFixture GaussianBlurFixture;

OCL_PERF_TEST_P(GaussianBlurFixture, GaussianBlur,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5, 7)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    const double eps = src.depth() == CV_8U ? 1 + DBL_EPSILON : 5e-4;

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::GaussianBlur(oclSrc, oclDst, Size(ksize, ksize), 0);

        oclDst.download(dst);

        SANITY_CHECK(dst, eps);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::GaussianBlur(src, dst, Size(ksize, ksize), 0);

        SANITY_CHECK(dst, eps);
    }
    else
        OCL_PERF_ELSE
}

///////////// filter2D////////////////////////

typedef FilterFixture Filter2DFixture;

OCL_PERF_TEST_P(Filter2DFixture, Filter2D,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, OCL_PERF_ENUM(3, 5)))
{
    const FilterParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst(srcSize, type), kernel(ksize, ksize, CV_32SC1);
    declare.in(src, WARMUP_RNG).in(kernel).out(dst);
    randu(kernel, -3.0, 3.0);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type), oclKernel(kernel);

        OCL_TEST_CYCLE() cv::ocl::filter2D(oclSrc, oclDst, -1, oclKernel);

        oclDst.download(dst);

        SANITY_CHECK(dst, 3e-2);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::filter2D(src, dst, -1, kernel);

        SANITY_CHECK(dst, 1e-2);
    }
    else
        OCL_PERF_ELSE
}

///////////// Bilateral////////////////////////

typedef TestBaseWithParam<Size> BilateralFixture;

OCL_PERF_TEST_P(BilateralFixture, Bilateral, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();
    const int d = 7;
    const double sigmacolor = 50.0, sigmaspace = 50.0;

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    Mat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_8UC1);

        OCL_TEST_CYCLE() cv::ocl::bilateralFilter(oclSrc, oclDst, d, sigmacolor, sigmaspace);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::bilateralFilter(src, dst, d, sigmacolor, sigmaspace);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}

///////////// MedianBlur////////////////////////

typedef tuple<Size, int> MedianBlurParams;
typedef TestBaseWithParam<MedianBlurParams> MedianBlurFixture;

OCL_PERF_TEST_P(MedianBlurFixture, Bilateral, ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(3, 5)))
{
    MedianBlurParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int ksize = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    Mat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, CV_8UC1);

        OCL_TEST_CYCLE() cv::ocl::medianFilter(oclSrc, oclDst, ksize);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::medianBlur(src, dst, ksize);

        SANITY_CHECK(dst);
    }
    else
        OCL_PERF_ELSE
}
