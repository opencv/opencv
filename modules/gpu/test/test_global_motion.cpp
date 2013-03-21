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

#include "test_precomp.hpp"

#ifdef HAVE_CUDA

using namespace std;
using namespace cv;

struct CompactPoints : testing::TestWithParam<gpu::DeviceInfo>
{
    virtual void SetUp() { gpu::setDevice(GetParam().deviceID()); }
};

GPU_TEST_P(CompactPoints, CanCompactizeSmallInput)
{
    Mat src0(1, 3, CV_32FC2);
    src0.at<Point2f>(0,0) = Point2f(0,0);
    src0.at<Point2f>(0,1) = Point2f(0,1);
    src0.at<Point2f>(0,2) = Point2f(0,2);

    Mat src1(1, 3, CV_32FC2);
    src1.at<Point2f>(0,0) = Point2f(1,0);
    src1.at<Point2f>(0,1) = Point2f(1,1);
    src1.at<Point2f>(0,2) = Point2f(1,2);

    Mat mask(1, 3, CV_8U);
    mask.at<uchar>(0,0) = 1;
    mask.at<uchar>(0,1) = 0;
    mask.at<uchar>(0,2) = 1;

    gpu::GpuMat dsrc0(src0), dsrc1(src1), dmask(mask);
    gpu::compactPoints(dsrc0, dsrc1, dmask);

    dsrc0.download(src0);
    dsrc1.download(src1);

    ASSERT_EQ(2, src0.cols);
    ASSERT_EQ(2, src1.cols);

    ASSERT_TRUE(src0.at<Point2f>(0,0) == Point2f(0,0));
    ASSERT_TRUE(src0.at<Point2f>(0,1) == Point2f(0,2));

    ASSERT_TRUE(src1.at<Point2f>(0,0) == Point2f(1,0));
    ASSERT_TRUE(src1.at<Point2f>(0,1) == Point2f(1,2));
}

INSTANTIATE_TEST_CASE_P(GPU_GlobalMotion, CompactPoints, ALL_DEVICES);

#endif // HAVE_CUDA
