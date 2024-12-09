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

#include "../test_precomp.hpp"
#include "opencv2/ts/ocl_test.hpp"

#ifdef HAVE_OPENCL

namespace opencv_test { namespace {

CV_ENUM(DIS_TestPresets, DISOpticalFlow::PRESET_ULTRAFAST, DISOpticalFlow::PRESET_FAST, DISOpticalFlow::PRESET_MEDIUM)

typedef ocl::TSTestWithParam<DIS_TestPresets> OCL_DenseOpticalFlow_DIS;

OCL_TEST_P(OCL_DenseOpticalFlow_DIS, Mat)
{
    int preset = (int)GetParam();
    Mat frame1, frame2, GT;

    frame1 = imread(TS::ptr()->get_data_path() + "optflow/RubberWhale1.png");
    frame2 = imread(TS::ptr()->get_data_path() + "optflow/RubberWhale2.png");

    CV_Assert(!frame1.empty() && !frame2.empty());

    cvtColor(frame1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame2, frame2, COLOR_BGR2GRAY);

    {
        Mat flow;
        UMat ocl_flow;

        Ptr<DenseOpticalFlow> algo = DISOpticalFlow::create(preset);
        OCL_OFF(algo->calc(frame1, frame2, flow));
        OCL_ON(algo->calc(frame1, frame2, ocl_flow));
        ASSERT_EQ(flow.rows, ocl_flow.rows);
        ASSERT_EQ(flow.cols, ocl_flow.cols);

        EXPECT_MAT_SIMILAR(flow, ocl_flow, 6e-3);
    }
}

OCL_INSTANTIATE_TEST_CASE_P(Video, OCL_DenseOpticalFlow_DIS,
                            DIS_TestPresets::all());

}} // namespace

#endif // HAVE_OPENCL
