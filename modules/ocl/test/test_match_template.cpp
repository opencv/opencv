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
//    Peng Xiao, pengxiao@multicorewareinc.com
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


#include "test_precomp.hpp"

#ifdef HAVE_OPENCL
////////////////////////////////////////////////////////////////////////////////
// MatchTemplate
#define ALL_TEMPLATE_METHODS testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR), TemplateMethod(cv::TM_CCOEFF), TemplateMethod(cv::TM_SQDIFF_NORMED), TemplateMethod(cv::TM_CCORR_NORMED), TemplateMethod(cv::TM_CCOEFF_NORMED))

IMPLEMENT_PARAM_CLASS(TemplateSize, cv::Size);

#define MTEMP_SIZES testing::Values(cv::Size(128, 256), cv::Size(1024, 768))

PARAM_TEST_CASE(MatchTemplate8U, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        templ_size = GET_PARAM(1);
        cn = GET_PARAM(2);
        method = GET_PARAM(3);
    }
};

OCL_TEST_P(MatchTemplate8U, Accuracy)
{
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_8U, cn), 0, 255);
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_8U, cn), 0, 255);

    cv::ocl::oclMat dst, ocl_image(image), ocl_templ(templ);
    cv::ocl::matchTemplate(ocl_image, ocl_templ, dst, method);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    cv::Mat mat_dst;
    dst.download(mat_dst);

    EXPECT_MAT_NEAR(dst_gold, mat_dst, templ_size.area() * 1e-1);
}

PARAM_TEST_CASE(MatchTemplate32F, cv::Size, TemplateSize, Channels, TemplateMethod)
{
    cv::Size size;
    cv::Size templ_size;
    int cn;
    int method;

    virtual void SetUp()
    {
        size = GET_PARAM(0);
        templ_size = GET_PARAM(1);
        cn = GET_PARAM(2);
        method = GET_PARAM(3);
    }
};

OCL_TEST_P(MatchTemplate32F, Accuracy)
{
    cv::Mat image = randomMat(size, CV_MAKETYPE(CV_32F, cn), 0, 255);
    cv::Mat templ = randomMat(templ_size, CV_MAKETYPE(CV_32F, cn), 0, 255);

    cv::ocl::oclMat dst, ocl_image(image), ocl_templ(templ);
    cv::ocl::matchTemplate(ocl_image, ocl_templ, dst, method);

    cv::Mat dst_gold;
    cv::matchTemplate(image, templ, dst_gold, method);

    cv::Mat mat_dst;
    dst.download(mat_dst);

    EXPECT_MAT_NEAR(dst_gold, mat_dst, templ_size.area() * 1e-1);
}

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, MatchTemplate8U,
                        testing::Combine(
                            MTEMP_SIZES,
                            testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
                            testing::Values(Channels(1), Channels(3), Channels(4)),
                            ALL_TEMPLATE_METHODS
                        )
                       );

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, MatchTemplate32F, testing::Combine(
                            MTEMP_SIZES,
                            testing::Values(TemplateSize(cv::Size(5, 5)), TemplateSize(cv::Size(16, 16)), TemplateSize(cv::Size(30, 30))),
                            testing::Values(Channels(1), Channels(3), Channels(4)),
                            testing::Values(TemplateMethod(cv::TM_SQDIFF), TemplateMethod(cv::TM_CCORR))));
#endif
