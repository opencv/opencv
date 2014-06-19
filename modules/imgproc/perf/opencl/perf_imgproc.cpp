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

///////////// equalizeHist ////////////////////////

typedef TestBaseWithParam<Size> EqualizeHistFixture;

OCL_PERF_TEST_P(EqualizeHistFixture, EqualizeHist, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();
    const double eps = 1;

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    UMat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::equalizeHist(src, dst);

    SANITY_CHECK(dst, eps);
}

///////////// calcHist ////////////////////////

typedef TestBaseWithParam<Size> CalcHistFixture;

OCL_PERF_TEST_P(CalcHistFixture, CalcHist, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();

    const std::vector<int> channels(1, 0);
    std::vector<float> ranges(2);
    std::vector<int> histSize(1, 256);
    ranges[0] = 0;
    ranges[1] = 256;

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    UMat src(srcSize, CV_8UC1), hist(256, 1, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(hist);

    OCL_TEST_CYCLE() cv::calcHist(std::vector<UMat>(1, src), channels, noArray(), hist, histSize, ranges, false);

    SANITY_CHECK(hist);
}

///////////// calcHist ////////////////////////

typedef TestBaseWithParam<Size> CalcBackProjFixture;

OCL_PERF_TEST_P(CalcBackProjFixture, CalcBackProj, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();

    const std::vector<int> channels(1, 0);
    std::vector<float> ranges(2);
    std::vector<int> histSize(1, 256);
    ranges[0] = 0;
    ranges[1] = 256;

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    UMat src(srcSize, CV_8UC1), hist(256, 1, CV_32FC1), dst(srcSize, CV_8UC1);
    declare.in(src, WARMUP_RNG).out(hist);

    cv::calcHist(std::vector<UMat>(1, src), channels, noArray(), hist, histSize, ranges, false);

    declare.in(src, WARMUP_RNG).out(dst);
    OCL_TEST_CYCLE() cv::calcBackProject(std::vector<UMat>(1,src), channels, hist, dst, ranges, 1);

    SANITY_CHECK_NOTHING();
}


/////////// CopyMakeBorder //////////////////////

CV_ENUM(Border, BORDER_CONSTANT, BORDER_REPLICATE, BORDER_REFLECT, BORDER_WRAP, BORDER_REFLECT_101)

typedef tuple<Size, MatType, Border> CopyMakeBorderParamType;
typedef TestBaseWithParam<CopyMakeBorderParamType> CopyMakeBorderFixture;

OCL_PERF_TEST_P(CopyMakeBorderFixture, CopyMakeBorder,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, Border::all()))
{
    const CopyMakeBorderParamType params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst;
    const Size dstSize = srcSize + Size(12, 12);
    dst.create(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::copyMakeBorder(src, dst, 7, 5, 5, 7, borderType, cv::Scalar(1.0));

    SANITY_CHECK(dst);
}

///////////// CornerMinEigenVal ////////////////////////

typedef Size_MatType CornerMinEigenValFixture;

OCL_PERF_TEST_P(CornerMinEigenValFixture, CornerMinEigenVal,
            ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;
    const int blockSize = 7, apertureSize = 1 + 2 * 3;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::cornerMinEigenVal(src, dst, blockSize, apertureSize, borderType);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

///////////// CornerHarris ////////////////////////

typedef Size_MatType CornerHarrisFixture;

OCL_PERF_TEST_P(CornerHarrisFixture, CornerHarris,
            ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::cornerHarris(src, dst, 5, 7, 0.1, borderType);

    SANITY_CHECK(dst, 5e-6, ERROR_RELATIVE);
}

///////////// PreCornerDetect ////////////////////////

typedef Size_MatType PreCornerDetectFixture;

OCL_PERF_TEST_P(PreCornerDetectFixture, PreCornerDetect,
            ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_8UC1, CV_32FC1)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), borderType = BORDER_REFLECT;

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    UMat src(srcSize, type), dst(srcSize, CV_32FC1);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::preCornerDetect(src, dst, 3, borderType);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

///////////// Integral ////////////////////////

typedef tuple<Size, MatDepth> IntegralParams;
typedef TestBaseWithParam<IntegralParams> IntegralFixture;

OCL_PERF_TEST_P(IntegralFixture, Integral1, ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32S, CV_32F)))
{
    const IntegralParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int ddepth = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, ddepth);

    UMat src(srcSize, CV_8UC1), dst(srcSize + Size(1, 1), ddepth);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::integral(src, dst, ddepth);

    SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
}

OCL_PERF_TEST_P(IntegralFixture, Integral2, ::testing::Combine(OCL_TEST_SIZES, OCL_PERF_ENUM(CV_32S, CV_32F)))
{
    const IntegralParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int ddepth = get<1>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, ddepth);

    UMat src(srcSize, CV_8UC1), sum(srcSize + Size(1, 1), ddepth), sqsum(srcSize + Size(1, 1), CV_32F);
    declare.in(src, WARMUP_RNG).out(sum).out(sqsum);

    OCL_TEST_CYCLE() cv::integral(src, sum, sqsum, ddepth, CV_32F);

    SANITY_CHECK(sum, 1e-6, ERROR_RELATIVE);
    SANITY_CHECK(sqsum, 5e-5, ERROR_RELATIVE);
}

///////////// Threshold ////////////////////////

CV_ENUM(ThreshType, THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO_INV)

typedef tuple<Size, MatType, ThreshType> ThreshParams;
typedef TestBaseWithParam<ThreshParams> ThreshFixture;

OCL_PERF_TEST_P(ThreshFixture, Threshold,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, ThreshType::all()))
{
    const ThreshParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int srcType = get<1>(params);
    const int threshType = get<2>(params);
    const double maxValue = 220.0, threshold = 50;

    checkDeviceMaxMemoryAllocSize(srcSize, srcType);

    UMat src(srcSize, srcType), dst(srcSize, srcType);
    declare.in(src, WARMUP_RNG).out(dst);

    OCL_TEST_CYCLE() cv::threshold(src, dst, threshold, maxValue, threshType);

    SANITY_CHECK(dst);
}

///////////// CLAHE ////////////////////////

typedef TestBaseWithParam<Size> CLAHEFixture;

OCL_PERF_TEST_P(CLAHEFixture, CLAHE, OCL_TEST_SIZES)
{
    const Size srcSize = GetParam();

    checkDeviceMaxMemoryAllocSize(srcSize, CV_8UC1);

    UMat src(srcSize, CV_8UC1), dst(srcSize, CV_8UC1);
    const double clipLimit = 40.0;
    declare.in(src, WARMUP_RNG).out(dst);

    cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(clipLimit);
    OCL_TEST_CYCLE() clahe->apply(src, dst);

    SANITY_CHECK(dst);
}

///////////// Canny ////////////////////////

typedef tuple<int, bool> CannyParams;
typedef TestBaseWithParam<CannyParams> CannyFixture;

OCL_PERF_TEST_P(CannyFixture, Canny, ::testing::Combine(OCL_PERF_ENUM(3, 5), Bool()))
{
    const CannyParams params = GetParam();
    int apertureSize = get<0>(params);
    bool L2Grad = get<1>(params);

    Mat _img = imread(getDataPath("gpu/stereobm/aloe-L.png"), cv::IMREAD_GRAYSCALE);
    ASSERT_TRUE(!_img.empty()) << "can't open aloe-L.png";

    UMat img;
    _img.copyTo(img);
    UMat edges(img.size(), CV_8UC1);

    declare.in(img, WARMUP_RNG).out(edges);

    OCL_TEST_CYCLE() cv::Canny(img, edges, 50.0, 100.0, apertureSize, L2Grad);

    if (apertureSize == 3)
        SANITY_CHECK(edges);
    else
        SANITY_CHECK_NOTHING();
}


} } // namespace cvtest::ocl

#endif // HAVE_OPENCL
