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

////////////////////////////////////////////////////////////////////////////////
// SqrtTest

template <typename T>
class SqrtTest : public ::testing::Test
{
public:
    void test_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);

        GpuMat_<T> dst = sqrt_(d_src);

        Mat dst_gold;
        cv::sqrt(src, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
    }

    void test_expr()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = sqrt_(d_src1 * d_src2);

        Mat dst_gold;
        cv::multiply(src1, src2, dst_gold);
        cv::sqrt(dst_gold, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
    }
};

TYPED_TEST_CASE(SqrtTest, float);

TYPED_TEST(SqrtTest, GpuMat)
{
    SqrtTest<TypeParam>::test_gpumat();
}

TYPED_TEST(SqrtTest, Expr)
{
    SqrtTest<TypeParam>::test_expr();
}

////////////////////////////////////////////////////////////////////////////////
// MagnitudeTest

template <typename T>
class MagnitudeTest : public ::testing::Test
{
public:
    void test_accuracy()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst1 = hypot_(d_src1, d_src2);
        GpuMat_<T> dst2 = magnitude_(d_src1, d_src2);
        GpuMat_<T> dst3 = sqrt_(sqr_(d_src1) + sqr_(d_src2));

        EXPECT_MAT_NEAR(dst1, dst2, 1e-4);
        EXPECT_MAT_NEAR(dst2, dst3, 0.0);
    }
};

TYPED_TEST_CASE(MagnitudeTest, float);

TYPED_TEST(MagnitudeTest, Accuracy)
{
    MagnitudeTest<TypeParam>::test_accuracy();
}

////////////////////////////////////////////////////////////////////////////////
// PowTest

template <typename T>
class PowTest : public ::testing::Test
{
public:
    void test_accuracy()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);

        GpuMat_<T> dst1 = pow_(d_src, 0.5);
        GpuMat_<T> dst2 = sqrt_(d_src);

        EXPECT_MAT_NEAR(dst1, dst2, 1e-5);
    }
};

TYPED_TEST_CASE(PowTest, float);

TYPED_TEST(PowTest, Accuracy)
{
    PowTest<TypeParam>::test_accuracy();
}
