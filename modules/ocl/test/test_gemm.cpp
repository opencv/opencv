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

using namespace std;

////////////////////////////////////////////////////////////////////////////
// GEMM

PARAM_TEST_CASE(Gemm, int, cv::Size, int)
{
    int      type;
    cv::Size mat_size;
    int		 flags;

    virtual void SetUp()
    {
        type     = GET_PARAM(0);
        mat_size = GET_PARAM(1);
        flags    = GET_PARAM(2);
    }
};

OCL_TEST_P(Gemm, Accuracy)
{
    cv::Mat a = randomMat(mat_size, type, 0.0, 10.0);
    cv::Mat b = randomMat(mat_size, type, 0.0, 10.0);
    cv::Mat c = randomMat(mat_size, type, 0.0, 10.0);

    cv::Mat dst;
    cv::ocl::oclMat ocl_dst;

    cv::gemm(a, b, 1.0, c, 1.0, dst, flags);
    cv::ocl::gemm(cv::ocl::oclMat(a), cv::ocl::oclMat(b), 1.0, cv::ocl::oclMat(c), 1.0, ocl_dst, flags);

    EXPECT_MAT_NEAR(dst, ocl_dst, mat_size.area() * 1e-4);
}

INSTANTIATE_TEST_CASE_P(ocl_gemm, Gemm, testing::Combine(
                            testing::Values(CV_32FC1, CV_32FC2/*, CV_64FC1, CV_64FC2*/),
                            testing::Values(cv::Size(20, 20), cv::Size(300, 300)),
                            testing::Values(0, (int)cv::GEMM_1_T, (int)cv::GEMM_2_T, (int)(cv::GEMM_1_T + cv::GEMM_2_T))));
