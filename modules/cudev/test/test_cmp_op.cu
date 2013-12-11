/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

using namespace cv;
using namespace cv::cuda;
using namespace cv::cudev;
using namespace cvtest;

typedef ::testing::Types<uchar, ushort, short, int, float> AllTypes;

////////////////////////////////////////////////////////////////////////////////
// LessTest

template <typename T>
class LessTest : public ::testing::Test
{
public:
    void test_gpumat_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<uchar> dst = (d_src1 < d_src2) * 255;

        Mat dst_gold;
        cv::compare(src1, src2, dst_gold, CMP_LT);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(LessTest, AllTypes);

TYPED_TEST(LessTest, GpuMat_GpuMat)
{
    LessTest<TypeParam>::test_gpumat_gpumat();
}

////////////////////////////////////////////////////////////////////////////////
// MinTest

template <typename T>
class MinTest : public ::testing::Test
{
public:
    void test_gpumat_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = min_(d_src1, d_src2);

        Mat dst_gold;
        cv::min(src1, src2, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(MinTest, AllTypes);

TYPED_TEST(MinTest, GpuMat_GpuMat)
{
    MinTest<TypeParam>::test_gpumat_gpumat();
}

////////////////////////////////////////////////////////////////////////////////
// ThreshBinaryTest

typedef ::testing::Types<uchar, short, float> ThreshTypes;

template <typename T>
class ThreshBinaryTest : public ::testing::Test
{
public:
    void test_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);

        GpuMat_<T> dst = threshBinary_(d_src, 128, 0);

        Mat dst_gold;
        cv::threshold(src, dst_gold, 128, 0, THRESH_BINARY);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(ThreshBinaryTest, ThreshTypes);

TYPED_TEST(ThreshBinaryTest, GpuMat)
{
    ThreshBinaryTest<TypeParam>::test_gpumat();
}
