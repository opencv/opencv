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
using namespace cv::gpu;
using namespace cv::cudev;
using namespace cvtest;

typedef ::testing::Types<uchar, ushort, short, int, float> AllTypes;

////////////////////////////////////////////////////////////////////////////////
// MergeTest

template <typename T>
class MergeTest : public ::testing::Test
{
public:
    void test_c2()
    {
        const Size size = randomSize(100, 400);

        const int src_type = DataType<T>::type;

        Mat src1 = randomMat(size, src_type);
        Mat src2 = randomMat(size, src_type);

        GpuMat_<T> d_src1(src1);
        GpuMat_<T> d_src2(src2);

        GpuMat_<typename MakeVec<T, 2>::type> dst;
        gridMerge(zipPtr(d_src1, d_src2), dst);

        Mat dst_gold;
        Mat srcs[] = {src1, src2};
        cv::merge(srcs, 2, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_c3()
    {
        const Size size = randomSize(100, 400);

        const int src_type = DataType<T>::type;

        Mat src1 = randomMat(size, src_type);
        Mat src2 = randomMat(size, src_type);
        Mat src3 = randomMat(size, src_type);

        GpuMat_<T> d_src1(src1);
        GpuMat_<T> d_src2(src2);
        GpuMat_<T> d_src3(src3);

        GpuMat_<typename MakeVec<T, 3>::type> dst;
        gridMerge(zipPtr(d_src1, d_src2, d_src3), dst);

        Mat dst_gold;
        Mat srcs[] = {src1, src2, src3};
        cv::merge(srcs, 3, dst_gold);

        ASSERT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(MergeTest, AllTypes);

TYPED_TEST(MergeTest, C2)
{
    MergeTest<TypeParam>::test_c2();
}

TYPED_TEST(MergeTest, C3)
{
    MergeTest<TypeParam>::test_c3();
}

////////////////////////////////////////////////////////////////////////////////
// SplitTest

template <typename T>
class SplitTest : public ::testing::Test
{
public:
    void test_c3()
    {
        const Size size = randomSize(100, 400);

        const int src_type = CV_MAKE_TYPE(DataType<T>::depth, 3);

        Mat src = randomMat(size, src_type);

        GpuMat_<typename MakeVec<T, 3>::type> d_src(src);

        GpuMat_<T> dst1, dst2, dst3;
        gridSplit(d_src, tie(dst1, dst2, dst3));

        std::vector<Mat> dst;
        cv::split(src, dst);

        ASSERT_MAT_NEAR(dst[0], dst1, 0.0);
        ASSERT_MAT_NEAR(dst[1], dst2, 0.0);
        ASSERT_MAT_NEAR(dst[2], dst3, 0.0);
    }

    void test_c4()
    {
        const Size size = randomSize(100, 400);

        const int src_type = CV_MAKE_TYPE(DataType<T>::depth, 4);

        Mat src = randomMat(size, src_type);

        GpuMat_<typename MakeVec<T, 4>::type> d_src(src);

        GpuMat_<T> dst1, dst2, dst3, dst4;
        gridSplit(d_src, tie(dst1, dst2, dst3, dst4));

        std::vector<Mat> dst;
        cv::split(src, dst);

        ASSERT_MAT_NEAR(dst[0], dst1, 0.0);
        ASSERT_MAT_NEAR(dst[1], dst2, 0.0);
        ASSERT_MAT_NEAR(dst[2], dst3, 0.0);
        ASSERT_MAT_NEAR(dst[3], dst4, 0.0);
    }
};

TYPED_TEST_CASE(SplitTest, AllTypes);

TYPED_TEST(SplitTest, C3)
{
    SplitTest<TypeParam>::test_c3();
}

TYPED_TEST(SplitTest, C4)
{
    SplitTest<TypeParam>::test_c4();
}
