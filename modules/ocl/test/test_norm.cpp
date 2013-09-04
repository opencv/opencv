/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#include "test_precomp.hpp"

typedef ::testing::TestWithParam<cv::Size> normFixture;

TEST_P(normFixture, DISABLED_accuracy)
{
    const cv::Size srcSize = GetParam();

    cv::Mat src1(srcSize, CV_8UC1), src2(srcSize, CV_8UC1);
    cv::randu(src1, 0, 2);
    cv::randu(src2, 0, 2);

    cv::ocl::oclMat oclSrc1(src1), oclSrc2(src2);

    double value = cv::norm(src1, src2, cv::NORM_INF);
    double oclValue = cv::ocl::norm(oclSrc1, oclSrc2, cv::NORM_INF);

    ASSERT_EQ(value, oclValue);
}

INSTANTIATE_TEST_CASE_P(oclNormTest, normFixture,
                        ::testing::Values(cv::Size(500, 500), cv::Size(1000, 1000)));
