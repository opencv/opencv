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

TEST(Sum, GpuMat)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<float> dst = sum_(d_src);
    float res;
    dst.download(_OutputArray(&res, 1));

    Scalar dst_gold = cv::sum(src);

    ASSERT_FLOAT_EQ(static_cast<float>(dst_gold[0]), res);
}

TEST(Sum, Expr)
{
    const Size size = randomSize(100, 400);

    Mat src1 = randomMat(size, CV_32FC1, 0, 1);
    Mat src2 = randomMat(size, CV_32FC1, 0, 1);

    GpuMat_<float> d_src1(src1), d_src2(src2);

    GpuMat_<float> dst = sum_(abs_(d_src1 - d_src2));
    float res;
    dst.download(_OutputArray(&res, 1));

    Scalar dst_gold = cv::norm(src1, src2, NORM_L1);

    ASSERT_FLOAT_EQ(static_cast<float>(dst_gold[0]), res);
}

TEST(MinVal, GpuMat)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<float> dst = minVal_(d_src);
    float res;
    dst.download(_OutputArray(&res, 1));

    double res_gold;
    cv::minMaxLoc(src, &res_gold, 0);

    ASSERT_FLOAT_EQ(static_cast<float>(res_gold), res);
}

TEST(MaxVal, Expr)
{
    const Size size = randomSize(100, 400);

    Mat src1 = randomMat(size, CV_32SC1);
    Mat src2 = randomMat(size, CV_32SC1);

    GpuMat_<int> d_src1(src1), d_src2(src2);

    GpuMat_<float> dst = maxVal_(abs_(d_src1 - d_src2));
    float res;
    dst.download(_OutputArray(&res, 1));

    double res_gold = cv::norm(src1, src2, NORM_INF);

    ASSERT_FLOAT_EQ(static_cast<float>(res_gold), res);
}

TEST(MinMaxVal, GpuMat)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<float> dst = minMaxVal_(d_src);
    float res[2];
    dst.download(Mat(1, 2, CV_32FC1, res));

    double res_gold[2];
    cv::minMaxLoc(src, &res_gold[0], &res_gold[1]);

    ASSERT_FLOAT_EQ(static_cast<float>(res_gold[0]), res[0]);
    ASSERT_FLOAT_EQ(static_cast<float>(res_gold[1]), res[1]);
}

TEST(NonZeroCount, Accuracy)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1, 0, 5);

    GpuMat_<uchar> d_src(src);

    GpuMat_<int> dst1 = countNonZero_(d_src);
    GpuMat_<int> dst2 = sum_(cvt_<int>(d_src) != 0);

    EXPECT_MAT_NEAR(dst1, dst2, 0.0);
}

TEST(ReduceToRow, Sum)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<int> dst = reduceToRow_<Sum<int> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 0, REDUCE_SUM, CV_32S);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(ReduceToRow, Avg)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<float> dst = reduceToRow_<Avg<float> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 0, REDUCE_AVG, CV_32F);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

TEST(ReduceToRow, Min)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uchar> dst = reduceToRow_<Min<uchar> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 0, REDUCE_MIN);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(ReduceToRow, Max)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uchar> dst = reduceToRow_<Max<uchar> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 0, REDUCE_MAX);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(ReduceToColumn, Sum)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<int> dst = reduceToColumn_<Sum<int> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 1, REDUCE_SUM, CV_32S);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(ReduceToColumn, Avg)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<float> dst = reduceToColumn_<Avg<float> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 1, REDUCE_AVG, CV_32F);

    EXPECT_MAT_NEAR(dst_gold, dst, 1e-4);
}

TEST(ReduceToColumn, Min)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uchar> dst = reduceToColumn_<Min<uchar> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 1, REDUCE_MIN);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(ReduceToColumn, Max)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<uchar> dst = reduceToColumn_<Max<uchar> >(d_src);

    Mat dst_gold;
    cv::reduce(src, dst_gold, 1, REDUCE_MAX);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

static void calcHistGold(const cv::Mat& src, cv::Mat& hist)
{
    hist.create(1, 256, CV_32SC1);
    hist.setTo(cv::Scalar::all(0));

    int* hist_row = hist.ptr<int>();
    for (int y = 0; y < src.rows; ++y)
    {
        const uchar* src_row = src.ptr(y);

        for (int x = 0; x < src.cols; ++x)
            ++hist_row[src_row[x]];
    }
}

TEST(Histogram, GpuMat)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_8UC1);

    GpuMat_<uchar> d_src(src);

    GpuMat_<int> dst = histogram_<256>(d_src);

    Mat dst_gold;
    calcHistGold(src, dst_gold);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}
