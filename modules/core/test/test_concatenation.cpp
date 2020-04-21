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

namespace opencv_test { namespace {

TEST(Core_Concatenation, empty)
{
    const Mat mat0x5(0,5, CV_8U, Scalar::all(1));
    const Mat mat10x5(10,5, CV_8U, Scalar::all(1));
    const Mat mat20x5(20,5, CV_8U, Scalar::all(1));

    const Mat mat5x0(5,0, CV_8U, Scalar::all(1));
    const Mat mat5x10(5,10, CV_8U, Scalar::all(1));
    const Mat mat5x20(5,20, CV_8U, Scalar::all(1));

    Mat result;

    cv::hconcat(mat5x0, mat5x0, result);
    EXPECT_MAT_N_DIFF(result, mat5x0, 0);
    cv::hconcat(mat5x0, mat5x10, result);
    EXPECT_MAT_N_DIFF(result, mat5x10, 0);
    cv::hconcat(mat5x10, mat5x0, result);
    EXPECT_MAT_N_DIFF(result, mat5x10, 0);
    cv::hconcat(mat5x10, mat5x10, result);
    EXPECT_MAT_N_DIFF(result, mat5x20, 0);

    cv::vconcat(mat0x5, mat0x5, result);
    EXPECT_MAT_N_DIFF(result, mat0x5, 0);
    cv::vconcat(mat0x5, mat10x5, result);
    EXPECT_MAT_N_DIFF(result, mat10x5, 0);
    cv::vconcat(mat10x5, mat0x5, result);
    EXPECT_MAT_N_DIFF(result, mat10x5, 0);
    cv::vconcat(mat10x5, mat10x5, result);
    EXPECT_MAT_N_DIFF(result, mat20x5, 0);
}

}} // namespace
