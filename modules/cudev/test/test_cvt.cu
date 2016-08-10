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
typedef ::testing::Types<short, float> Fp16Types;

////////////////////////////////////////////////////////////////////////////////
// CvtTest

template <typename T>
class CvtTest : public ::testing::Test
{
public:
    void test_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);

        GpuMat_<T> dst = cvt_<T>(cvt_<float>(d_src) * 2.0f - 10.0f);

        Mat dst_gold;
        src.convertTo(dst_gold, src.depth(), 2, -10);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

// dummy class
template <typename T>
class CvFp16Test : public ::testing::Test
{
public:
    void test_gpumat()
    {
    }
};

template <>
class CvFp16Test <short> : public ::testing::Test
{
public:
    void test_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<float>::type;

        Mat src = randomMat(size, type), dst, ref;

        GpuMat_<float> g_src(src);
        GpuMat g_dst;

        // Fp32 -> Fp16
        cuda::convertFp16(g_src, g_dst);
        cv::convertFp16(src, dst);
        // Fp16 -> Fp32
        cuda::convertFp16(g_dst.clone(), g_dst);
        cv::convertFp16(dst, ref);

        g_dst.download(dst);
        EXPECT_MAT_NEAR(dst, ref, 0.0);
    }
};

template <>
class CvFp16Test <float> : public ::testing::Test
{
public:
    void test_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<float>::type;

        Mat src = randomMat(size, type), dst, ref;

        GpuMat_<float> g_src(src);
        GpuMat g_dst;

        // Fp32 -> Fp16
        cuda::convertFp16(g_src, g_dst);
        cv::convertFp16(src, ref);

        g_dst.download(dst);
        EXPECT_MAT_NEAR(dst, ref, 0.0);
    }
};

TYPED_TEST_CASE(CvtTest, AllTypes);

TYPED_TEST(CvtTest, GpuMat)
{
    CvtTest<TypeParam>::test_gpumat();
}

TYPED_TEST_CASE(CvFp16Test, Fp16Types);

TYPED_TEST(CvFp16Test, GpuMat)
{
    CvFp16Test<TypeParam>::test_gpumat();
}
