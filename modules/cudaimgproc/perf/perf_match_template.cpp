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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
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

using namespace std;
using namespace testing;
using namespace perf;

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate8U

CV_ENUM(TemplateMethod, TM_SQDIFF, TM_SQDIFF_NORMED, TM_CCORR, TM_CCORR_NORMED, TM_CCOEFF, TM_CCOEFF_NORMED)

DEF_PARAM_TEST(Sz_TemplateSz_Cn_Method, cv::Size, cv::Size, MatCn, TemplateMethod);

PERF_TEST_P(Sz_TemplateSz_Cn_Method, MatchTemplate8U,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(cv::Size(5, 5), cv::Size(16, 16), cv::Size(30, 30)),
                    CUDA_CHANNELS_1_3_4,
                    TemplateMethod::all()))
{
    declare.time(300.0);

    const cv::Size size = GET_PARAM(0);
    const cv::Size templ_size = GET_PARAM(1);
    const int cn = GET_PARAM(2);
    const int method = GET_PARAM(3);

    cv::Mat image(size, CV_MAKE_TYPE(CV_8U, cn));
    cv::Mat templ(templ_size, CV_MAKE_TYPE(CV_8U, cn));
    declare.in(image, templ, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_image(image);
        const cv::cuda::GpuMat d_templ(templ);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image.type(), method);

        TEST_CYCLE() alg->match(d_image, d_templ, dst);

        CUDA_SANITY_CHECK(dst, 1e-5, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::matchTemplate(image, templ, dst, method);

        CPU_SANITY_CHECK(dst);
    }
};

////////////////////////////////////////////////////////////////////////////////
// MatchTemplate32F

PERF_TEST_P(Sz_TemplateSz_Cn_Method, MatchTemplate32F,
            Combine(CUDA_TYPICAL_MAT_SIZES,
                    Values(cv::Size(5, 5), cv::Size(16, 16), cv::Size(30, 30)),
                    CUDA_CHANNELS_1_3_4,
                    Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))))
{
    declare.time(300.0);

    const cv::Size size = GET_PARAM(0);
    const cv::Size templ_size = GET_PARAM(1);
    const int cn = GET_PARAM(2);
    int method = GET_PARAM(3);

    cv::Mat image(size, CV_MAKE_TYPE(CV_32F, cn));
    cv::Mat templ(templ_size, CV_MAKE_TYPE(CV_32F, cn));
    declare.in(image, templ, WARMUP_RNG);

    if (PERF_RUN_CUDA())
    {
        const cv::cuda::GpuMat d_image(image);
        const cv::cuda::GpuMat d_templ(templ);
        cv::cuda::GpuMat dst;

        cv::Ptr<cv::cuda::TemplateMatching> alg = cv::cuda::createTemplateMatching(image.type(), method);

        TEST_CYCLE() alg->match(d_image, d_templ, dst);

        CUDA_SANITY_CHECK(dst, 1e-6, ERROR_RELATIVE);
    }
    else
    {
        cv::Mat dst;

        TEST_CYCLE() cv::matchTemplate(image, templ, dst, method);

        CPU_SANITY_CHECK(dst);
    }
}
