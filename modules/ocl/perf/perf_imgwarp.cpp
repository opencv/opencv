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
using std::tr1::tuple;
using std::tr1::get;

///////////// WarpAffine ////////////////////////

CV_ENUM(InterType, INTER_NEAREST, INTER_LINEAR)

typedef tuple<Size, MatType, InterType> WarpAffineParams;
typedef TestBaseWithParam<WarpAffineParams> WarpAffineFixture;

OCL_PERF_TEST_P(WarpAffineFixture, WarpAffine,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, InterType::all()))
{
    static const double coeffs[2][3] =
    {
        { cos(CV_PI / 6), -sin(CV_PI / 6), 100.0 },
        { sin(CV_PI / 6), cos(CV_PI / 6), -100.0 }
    };
    Mat M(2, 3, CV_64F, (void *)coeffs);

    const WarpAffineParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::warpAffine(oclSrc, oclDst, M, srcSize, interpolation);

        oclDst.download(dst);

        SANITY_CHECK(dst, 5e-4);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::warpAffine(src, dst, M, srcSize, interpolation);

        SANITY_CHECK(dst, 5e-4);
    }
    else
        OCL_PERF_ELSE
}

///////////// WarpPerspective ////////////////////////

typedef WarpAffineParams WarpPerspectiveParams;
typedef TestBaseWithParam<WarpPerspectiveParams> WarpPerspectiveFixture;

OCL_PERF_TEST_P(WarpPerspectiveFixture, WarpPerspective,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, InterType::all()))
{
    static const double coeffs[3][3] =
    {
        {cos(CV_PI / 6), -sin(CV_PI / 6), 100.0},
        {sin(CV_PI / 6), cos(CV_PI / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);

    const WarpPerspectiveParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params);

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        OCL_TEST_CYCLE() cv::ocl::warpPerspective(oclSrc, oclDst, M, srcSize, interpolation);

        oclDst.download(dst);

        SANITY_CHECK(dst, 5e-4);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::warpPerspective(src, dst, M, srcSize, interpolation);

        SANITY_CHECK(dst, 5e-4);
    }
    else
        OCL_PERF_ELSE
}

///////////// Resize ////////////////////////

typedef tuple<Size, MatType, InterType, double> ResizeParams;
typedef TestBaseWithParam<ResizeParams> ResizeFixture;

OCL_PERF_TEST_P(ResizeFixture, Resize,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134,
                               InterType::all(), ::testing::Values(0.5, 2.0)))
{
    const ResizeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interType = get<2>(params);
    double scale = get<3>(params);
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));

    checkDeviceMaxMemoryAllocSize(srcSize, type);
    checkDeviceMaxMemoryAllocSize(dstSize, type);

    Mat src(srcSize, type), dst;
    dst.create(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(dstSize, type);

        OCL_TEST_CYCLE() cv::ocl::resize(oclSrc, oclDst, Size(), scale, scale, interType);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, interType);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}

typedef tuple<Size, MatType, double> ResizeAreaParams;
typedef TestBaseWithParam<ResizeAreaParams> ResizeAreaFixture;

OCL_PERF_TEST_P(ResizeAreaFixture, Resize,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES_134, ::testing::Values(0.3, 0.5, 0.6)))
{
    const ResizeAreaParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params);
    double scale = get<2>(params);
    const Size dstSize(cvRound(srcSize.width * scale), cvRound(srcSize.height * scale));

    checkDeviceMaxMemoryAllocSize(srcSize, type);

    Mat src(srcSize, type), dst;
    dst.create(dstSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(dstSize, type);

        OCL_TEST_CYCLE() cv::ocl::resize(oclSrc, oclDst, Size(), scale, scale, cv::INTER_AREA);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::resize(src, dst, Size(), scale, scale, cv::INTER_AREA);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}

///////////// Remap ////////////////////////

typedef tuple<Size, MatType, InterType> RemapParams;
typedef TestBaseWithParam<RemapParams> RemapFixture;

OCL_PERF_TEST_P(RemapFixture, Remap,
            ::testing::Combine(OCL_TEST_SIZES, OCL_TEST_TYPES, InterType::all()))
{
    const RemapParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), interpolation = get<2>(params);

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    Mat xmap, ymap;
    xmap.create(srcSize, CV_32FC1);
    ymap.create(srcSize, CV_32FC1);

    for (int i = 0; i < srcSize.height; ++i)
    {
        float * const xmap_row = xmap.ptr<float>(i);
        float * const ymap_row = ymap.ptr<float>(i);

        for (int j = 0; j < srcSize.width; ++j)
        {
            xmap_row[j] = (j - srcSize.width * 0.5f) * 0.75f + srcSize.width * 0.5f;
            ymap_row[j] = (i - srcSize.height * 0.5f) * 0.75f + srcSize.height * 0.5f;
        }
    }

    const int borderMode = BORDER_CONSTANT;

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);
        ocl::oclMat oclXMap(xmap), oclYMap(ymap);

        OCL_TEST_CYCLE() cv::ocl::remap(oclSrc, oclDst, oclXMap, oclYMap, interpolation, borderMode);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::remap(src, dst, xmap, ymap, interpolation, borderMode);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else
        OCL_PERF_ELSE
}


///////////// BuildWarpPerspectiveMaps ////////////////////////

static void buildWarpPerspectiveMaps(const Mat &M, bool inverse, Size dsize, Mat &xmap, Mat &ymap)
{
    CV_Assert(M.rows == 3 && M.cols == 3);
    CV_Assert(dsize.area() > 0);

    xmap.create(dsize, CV_32FC1);
    ymap.create(dsize, CV_32FC1);

    float coeffs[3 * 3];
    Mat coeffsMat(3, 3, CV_32F, (void *)coeffs);

    if (inverse)
        M.convertTo(coeffsMat, coeffsMat.type());
    else
    {
        cv::Mat iM;
        invert(M, iM);
        iM.convertTo(coeffsMat, coeffsMat.type());
    }

    for (int y = 0; y < dsize.height; ++y)
    {
        float * const xmap_ptr = xmap.ptr<float>(y);
        float * const ymap_ptr = ymap.ptr<float>(y);

        for (int x = 0; x < dsize.width; ++x)
        {
            float coeff = 1.0f / (x * coeffs[6] + y * coeffs[7] + coeffs[8]);
            xmap_ptr[x] = (x * coeffs[0] + y * coeffs[1] + coeffs[2]) * coeff;
            ymap_ptr[x] = (x * coeffs[3] + y * coeffs[4] + coeffs[5]) * coeff;
        }
    }
}

typedef TestBaseWithParam<Size> BuildWarpPerspectiveMapsFixture;

PERF_TEST_P(BuildWarpPerspectiveMapsFixture, Inverse, OCL_TYPICAL_MAT_SIZES)
{
    static const double coeffs[3][3] =
    {
        {cos(CV_PI / 6), -sin(CV_PI / 6), 100.0},
        {sin(CV_PI / 6), cos(CV_PI / 6), -100.0},
        {0.0, 0.0, 1.0}
    };
    Mat M(3, 3, CV_64F, (void *)coeffs);
    const Size dsize = GetParam();
    const double eps = 5e-4;

    Mat xmap(dsize, CV_32FC1), ymap(dsize, CV_32FC1);
    declare.in(M).out(xmap, ymap);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclXMap(dsize, CV_32FC1), oclYMap(dsize, CV_32FC1);

        OCL_TEST_CYCLE() cv::ocl::buildWarpPerspectiveMaps(M, true, dsize, oclXMap, oclYMap);

        oclXMap.download(xmap);
        oclYMap.download(ymap);

        SANITY_CHECK(xmap, eps);
        SANITY_CHECK(ymap, eps);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() buildWarpPerspectiveMaps(M, true, dsize, xmap, ymap);

        SANITY_CHECK(xmap, eps);
        SANITY_CHECK(ymap, eps);
    }
    else
        OCL_PERF_ELSE
}
