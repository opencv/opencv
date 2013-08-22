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
//     and/or other oclMaterials provided with the distribution.
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

///////////// Blur////////////////////////

CV_ENUM(BlurMatType, CV_8UC1, CV_8UC4)

typedef tuple<Size, BlurMatType> BlurParams;
typedef TestBaseWithParam<BlurParams> BlurFixture;

PERF_TEST_P(BlurFixture, Blur,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               BlurMatType::all()))
{
    // getting params
    BlurParams params = GetParam();
    const Size srcSize = get<0>(params), ksize(3, 3);
    const int type = get<1>(params), bordertype = BORDER_CONSTANT;

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(5);

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::blur(oclSrc, oclDst, ksize, Point(-1, -1), bordertype);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::blur(src, dst, ksize, Point(-1, -1), bordertype);

        SANITY_CHECK(dst, 1 + DBL_EPSILON);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Laplacian////////////////////////

typedef BlurMatType LaplacianMatType;
typedef tuple<Size, LaplacianMatType> LaplacianParams;
typedef TestBaseWithParam<LaplacianParams> LaplacianFixture;

PERF_TEST_P(LaplacianFixture, Laplacian,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               LaplacianMatType::all()))
{
    // getting params
    LaplacianParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = 3;

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(6);

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::Laplacian(oclSrc, oclDst, -1, ksize, 1);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::Laplacian(src, dst, -1, ksize, 1);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Erode ////////////////////

CV_ENUM(ErodeMatType, CV_8UC1, CV_8UC4, CV_32FC1, CV_32FC4)

typedef tuple<Size, ErodeMatType> ErodeParams;
typedef TestBaseWithParam<ErodeParams> ErodeFixture;

PERF_TEST_P(ErodeFixture, Erode,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               ErodeMatType::all()))
{
    // getting params
    ErodeParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = 3;
    const Mat ker = getStructuringElement(MORPH_RECT, Size(ksize, ksize));

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst).in(ker);

    if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(5);

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type), oclKer(ker);

        TEST_CYCLE() cv::ocl::erode(oclSrc, oclDst, oclKer);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::erode(src, dst, ker);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Sobel ////////////////////////

typedef BlurMatType SobelMatType;
typedef tuple<Size, SobelMatType> SobelMatParams;
typedef TestBaseWithParam<SobelMatParams> SobelFixture;

PERF_TEST_P(SobelFixture, Sobel,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               SobelMatType::all()))
{
    // getting params
    SobelMatParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), dx = 1, dy = 1;

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if ((srcSize == OCL_SIZE_2000 && type == CV_8UC4) ||
            (srcSize == OCL_SIZE_4000 && type == CV_8UC1))
        declare.time(5.5);
    else if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(20);

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::Sobel(oclSrc, oclDst, -1, dx, dy);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::Sobel(src, dst, -1, dx, dy);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// Scharr ////////////////////////

typedef BlurMatType ScharrMatType;
typedef tuple<Size, ScharrMatType> ScharrParams;
typedef TestBaseWithParam<ScharrParams> ScharrFixture;

PERF_TEST_P(ScharrFixture, Scharr,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               ScharrMatType::all()))
{
    // getting params
    ScharrParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), dx = 1, dy = 0;

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    if ((srcSize == OCL_SIZE_2000 && type == CV_8UC4) ||
            (srcSize == OCL_SIZE_4000 && type == CV_8UC1))
        declare.time(5.5);
    else if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(21);

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::Scharr(oclSrc, oclDst, -1, dx, dy);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::Scharr(src, dst, -1, dx, dy);

        SANITY_CHECK(dst);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// GaussianBlur ////////////////////////

typedef ErodeMatType GaussianBlurMatType;
typedef tuple<Size, GaussianBlurMatType> GaussianBlurParams;
typedef TestBaseWithParam<GaussianBlurParams> GaussianBlurFixture;

PERF_TEST_P(GaussianBlurFixture, GaussianBlur,
            ::testing::Combine(::testing::Values(OCL_SIZE_1000, OCL_SIZE_2000),
                               GaussianBlurMatType::all()))
{
    // getting params
    GaussianBlurParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = 7;

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type);
    declare.in(src, WARMUP_RNG).out(dst);

    const double eps = src.depth() == CV_8U ? 1 + DBL_EPSILON : 3e-4;

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type);

        TEST_CYCLE() cv::ocl::GaussianBlur(oclSrc, oclDst, Size(ksize, ksize), 0);

        oclDst.download(dst);

        SANITY_CHECK(dst, eps);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::GaussianBlur(src, dst, Size(ksize, ksize), 0);

        SANITY_CHECK(dst, eps);
    }
#ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}

///////////// filter2D////////////////////////

typedef BlurMatType filter2DMatType;
typedef tuple<Size, filter2DMatType> filter2DParams;
typedef TestBaseWithParam<filter2DParams> filter2DFixture;

PERF_TEST_P(filter2DFixture, filter2D,
            ::testing::Combine(OCL_TYPICAL_MAT_SIZES,
                               filter2DMatType::all()))
{
    // getting params
    filter2DParams params = GetParam();
    const Size srcSize = get<0>(params);
    const int type = get<1>(params), ksize = 3;

    const std::string impl = getSelectedImpl();

    Mat src(srcSize, type), dst(srcSize, type), kernel(ksize, ksize, CV_32SC1);
    declare.in(src, WARMUP_RNG).in(kernel).out(dst);
    randu(kernel, -3.0, 3.0);

    if (srcSize == OCL_SIZE_4000 && type == CV_8UC4)
        declare.time(8);

    if (impl == "ocl")
    {
        ocl::oclMat oclSrc(src), oclDst(srcSize, type), oclKernel(kernel);

        TEST_CYCLE() cv::ocl::filter2D(oclSrc, oclDst, -1, oclKernel);

        oclDst.download(dst);

        SANITY_CHECK(dst);
    }
    else if (impl == "plain")
    {
        TEST_CYCLE() cv::filter2D(src, dst, -1, kernel);

        SANITY_CHECK(dst);
    }
 #ifdef HAVE_OPENCV_GPU
    else if (impl == "gpu")
        CV_TEST_FAIL_NO_IMPL();
#endif
    else
        CV_TEST_FAIL_NO_IMPL();
}
