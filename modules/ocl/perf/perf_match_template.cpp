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

/////////// matchTemplate ////////////////////////

typedef Size_MatType CV_TM_CCORRFixture;

OCL_PERF_TEST_P(CV_TM_CCORRFixture, matchTemplate,
                ::testing::Combine(::testing::Values(Size(1000, 1000), Size(2000, 2000)),
                               OCL_PERF_ENUM(CV_32FC1, CV_32FC4)))
{
    const Size_MatType_t params = GetParam();
    const Size srcSize = get<0>(params), templSize(5, 5);
    const int type = get<1>(params);

    Mat src(srcSize, type), templ(templSize, type);
    const Size dstSize(src.cols - templ.cols + 1, src.rows - templ.rows + 1);
    Mat dst(dstSize, CV_32F);
    randu(src, 0.0f, 1.0f);
    randu(templ, 0.0f, 1.0f);
    declare.in(src, templ).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclTempl(templ), oclDst(dstSize, CV_32F);

        OCL_TEST_CYCLE() cv::ocl::matchTemplate(oclSrc, oclTempl, oclDst, CV_TM_CCORR);

        oclDst.download(dst);

        SANITY_CHECK(dst, 1e-4);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::matchTemplate(src, templ, dst, CV_TM_CCORR);

        SANITY_CHECK(dst, 1e-4);
    }
    else
        OCL_PERF_ELSE
}

typedef TestBaseWithParam<Size> CV_TM_CCORR_NORMEDFixture;

OCL_PERF_TEST_P(CV_TM_CCORR_NORMEDFixture, matchTemplate,
                ::testing::Values(Size(1000, 1000), Size(2000, 2000), Size(4000, 4000)))
{
    const Size srcSize = GetParam(), templSize(5, 5);

    Mat src(srcSize, CV_8UC1), templ(templSize, CV_8UC1), dst;
    const Size dstSize(src.cols - templ.cols + 1, src.rows - templ.rows + 1);
    dst.create(dstSize, CV_8UC1);
    declare.in(src, templ, WARMUP_RNG).out(dst);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclSrc(src), oclTempl(templ), oclDst(dstSize, CV_8UC1);

        OCL_TEST_CYCLE() cv::ocl::matchTemplate(oclSrc, oclTempl, oclDst, CV_TM_CCORR_NORMED);

        oclDst.download(dst);

        SANITY_CHECK(dst, 2e-2);
    }
    else if (RUN_PLAIN_IMPL)
    {
        TEST_CYCLE() cv::matchTemplate(src, templ, dst, CV_TM_CCORR_NORMED);

        SANITY_CHECK(dst, 2e-2);
    }
    else
        OCL_PERF_ELSE
}

CV_ENUM(MethodType, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

typedef std::tr1::tuple<Size, Size, MethodType> ImgSize_TmplSize_Method_t;
typedef TestBaseWithParam<ImgSize_TmplSize_Method_t> ImgSize_TmplSize_Method;

OCL_PERF_TEST_P(ImgSize_TmplSize_Method, MatchTemplate,
        ::testing::Combine(
            testing::Values(szSmall128, cv::Size(320, 240),
                            cv::Size(640, 480), cv::Size(800, 600),
                            cv::Size(1024, 768), cv::Size(1280, 1024)),
            testing::Values(cv::Size(12, 12), cv::Size(28, 9),
                            cv::Size(8, 30), cv::Size(16, 16)),
            MethodType::all()
            )
        )
{
    Size imgSz = get<0>(GetParam());
    Size tmplSz = get<1>(GetParam());
    int method = get<2>(GetParam());

    Mat img(imgSz, CV_8UC1);
    Mat tmpl(tmplSz, CV_8UC1);
    Mat result(imgSz - tmplSz + Size(1, 1), CV_32F);

    bool isNormed =
        method == TM_CCORR_NORMED ||
        method == TM_SQDIFF_NORMED ||
        method == TM_CCOEFF_NORMED;

    declare.in(img, tmpl, WARMUP_RNG).out(result).time(30);

    if (RUN_OCL_IMPL)
    {
        ocl::oclMat oclImg(img), oclTmpl(tmpl), oclResult(result.size(), CV_32F);

        OCL_TEST_CYCLE() matchTemplate(oclImg, oclTmpl, oclResult, method);

        oclResult.download(result);
    }
    else if (RUN_PLAIN_IMPL)
    {
        OCL_TEST_CYCLE() matchTemplate(img, tmpl, result, method);
    }
    else
        OCL_PERF_ELSE

    double eps = isNormed ? 3e-2
        : 255 * 255 * tmpl.total() * 1e-4;

    if (method == TM_SQDIFF_NORMED)
        SANITY_CHECK_NOTHING();
    else if (isNormed)
        SANITY_CHECK(result, eps, ERROR_RELATIVE);
    else
        SANITY_CHECK(result, eps);
}
