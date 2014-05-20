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

#include "test_precomp.hpp"
#ifdef HAVE_OPENCL

////////////////////////////////////////////////////////
// Canny
IMPLEMENT_PARAM_CLASS(AppertureSize, int)
IMPLEMENT_PARAM_CLASS(L2gradient, bool)

PARAM_TEST_CASE(Canny, AppertureSize, L2gradient)
{
    int apperture_size;
    bool useL2gradient;

    cv::Mat edges_gold;
    virtual void SetUp()
    {
        apperture_size = GET_PARAM(0);
        useL2gradient = GET_PARAM(1);
    }
};

OCL_TEST_P(Canny, Accuracy)
{
    cv::Mat img = readImage("cv/shared/fruits.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(img.empty());

    double low_thresh = 50.0;
    double high_thresh = 100.0;

    cv::ocl::oclMat ocl_img = cv::ocl::oclMat(img);

    cv::ocl::oclMat edges;
    cv::ocl::Canny(ocl_img, edges, low_thresh, high_thresh, apperture_size, useL2gradient);

    cv::Mat edges_gold;
    cv::Canny(img, edges_gold, low_thresh, high_thresh, apperture_size, useL2gradient);

    EXPECT_MAT_SIMILAR(edges_gold, edges, 1e-2);
}

INSTANTIATE_TEST_CASE_P(OCL_ImgProc, Canny, testing::Combine(
                            testing::Values(AppertureSize(3), AppertureSize(5)),
                            testing::Values(L2gradient(false), L2gradient(true))));
#endif
