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
typedef ::testing::Types<short, int, float> SignedTypes;

////////////////////////////////////////////////////////////////////////////////
// UnaryMinusTest

template <typename T>
class UnaryMinusTest : public ::testing::Test
{
public:
    void test_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);

        GpuMat_<T> dst = -d_src;

        Mat dst_gold;
        src.convertTo(dst_gold, src.depth(), -1);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_globptr()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);
        GlobPtrSz<T> d_src_ptr = d_src;

        GpuMat_<T> dst = -d_src_ptr;

        Mat dst_gold;
        src.convertTo(dst_gold, src.depth(), -1);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_texptr()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);
        Texture<T> tex_src(d_src);

        GpuMat_<T> dst = -tex_src;

        Mat dst_gold;
        src.convertTo(dst_gold, src.depth(), -1);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_expr()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = -(d_src1 + d_src2);

        Mat dst_gold;
        cv::add(src1, src2, dst_gold);
        dst_gold.convertTo(dst_gold, dst_gold.depth(), -1);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(UnaryMinusTest, SignedTypes);

TYPED_TEST(UnaryMinusTest, GpuMat)
{
    UnaryMinusTest<TypeParam>::test_gpumat();
}

TYPED_TEST(UnaryMinusTest, GlobPtrSz)
{
    UnaryMinusTest<TypeParam>::test_globptr();
}

TYPED_TEST(UnaryMinusTest, TexturePtr)
{
    UnaryMinusTest<TypeParam>::test_texptr();
}

TYPED_TEST(UnaryMinusTest, Expr)
{
    UnaryMinusTest<TypeParam>::test_expr();
}

////////////////////////////////////////////////////////////////////////////////
// PlusTest

template <typename T>
class PlusTest : public ::testing::Test
{
public:
    void test_gpumat_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = d_src1 + d_src2;

        Mat dst_gold;
        cv::add(src1, src2, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_texptr_scalar()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);
        Texture<T> tex_src(d_src);

        GpuMat_<T> dst = tex_src + static_cast<T>(5);

        Mat dst_gold;
        cv::add(src, 5, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_expr_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);
        Mat src3 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2), d_src3(src3);

        GpuMat_<T> dst = d_src1 + d_src2 + d_src3;

        Mat dst_gold;
        cv::add(src1, src2, dst_gold);
        cv::add(dst_gold, src3, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_scalar_expr()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = static_cast<T>(5) + (d_src1 + d_src2);

        Mat dst_gold;
        cv::add(src1, src2, dst_gold);
        cv::add(dst_gold, 5, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(PlusTest, AllTypes);

TYPED_TEST(PlusTest, GpuMat_GpuMat)
{
    PlusTest<TypeParam>::test_gpumat_gpumat();
}

TYPED_TEST(PlusTest, TexturePtr_Scalar)
{
    PlusTest<TypeParam>::test_texptr_scalar();
}

TYPED_TEST(PlusTest, Expr_GpuMat)
{
    PlusTest<TypeParam>::test_expr_gpumat();
}

TYPED_TEST(PlusTest, Scalar_Expr)
{
    PlusTest<TypeParam>::test_scalar_expr();
}

////////////////////////////////////////////////////////////////////////////////
// MinusTest

template <typename T>
class MinusTest : public ::testing::Test
{
public:
    void test_gpumat_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = d_src1 - d_src2;

        Mat dst_gold;
        cv::subtract(src1, src2, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_texptr_scalar()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src = randomMat(size, type);

        GpuMat_<T> d_src(src);
        Texture<T> tex_src(d_src);

        GpuMat_<T> dst = tex_src - static_cast<T>(5);

        Mat dst_gold;
        cv::subtract(src, 5, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_expr_gpumat()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);
        Mat src3 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2), d_src3(src3);

        GpuMat_<T> dst = (d_src1 + d_src2) - d_src3;

        Mat dst_gold;
        cv::add(src1, src2, dst_gold);
        cv::subtract(dst_gold, src3, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }

    void test_scalar_expr()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst = static_cast<T>(5) - (d_src1 + d_src2);

        Mat dst_gold;
        cv::add(src1, src2, dst_gold);
        cv::subtract(5, dst_gold, dst_gold);

        EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
    }
};

TYPED_TEST_CASE(MinusTest, SignedTypes);

TYPED_TEST(MinusTest, GpuMat_GpuMat)
{
    MinusTest<TypeParam>::test_gpumat_gpumat();
}

TYPED_TEST(MinusTest, TexturePtr_Scalar)
{
    MinusTest<TypeParam>::test_texptr_scalar();
}

TYPED_TEST(MinusTest, Expr_GpuMat)
{
    MinusTest<TypeParam>::test_expr_gpumat();
}

TYPED_TEST(MinusTest, Scalar_Expr)
{
    MinusTest<TypeParam>::test_scalar_expr();
}

////////////////////////////////////////////////////////////////////////////////
// AbsDiffTest

template <typename T>
class AbsDiffTest : public ::testing::Test
{
public:
    void test_accuracy()
    {
        const Size size = randomSize(100, 400);
        const int type = DataType<T>::type;

        Mat src1 = randomMat(size, type);
        Mat src2 = randomMat(size, type);

        GpuMat_<T> d_src1(src1), d_src2(src2);

        GpuMat_<T> dst1 = absdiff_(d_src1, d_src2);
        GpuMat_<T> dst2 = abs_(d_src1 - d_src2);

        EXPECT_MAT_NEAR(dst1, dst2, 0.0);
    }
};

TYPED_TEST_CASE(AbsDiffTest, SignedTypes);

TYPED_TEST(AbsDiffTest, Accuracy)
{
    AbsDiffTest<TypeParam>::test_accuracy();
}
