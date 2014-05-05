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

// remap

enum { HALF_SIZE=0, UPSIDE_DOWN, REFLECTION_X, REFLECTION_BOTH };

static void generateMap(Mat& mapx, Mat& mapy, int remapMode)
{
    for (int j = 0; j < mapx.rows; ++j)
    {
        for (int i = 0; i < mapx.cols; ++i)
        {
            switch (remapMode)
            {
            case HALF_SIZE:
                if (i > mapx.cols*0.25 && i < mapx.cols*0.75 && j > mapx.rows*0.25 && j < mapx.rows*0.75)
                {
                    mapx.at<float>(j,i) = 2.f * (i - mapx.cols * 0.25f) + 0.5f;
                    mapy.at<float>(j,i) = 2.f * (j - mapx.rows * 0.25f) + 0.5f;
                }
                else
                {
                    mapx.at<float>(j,i) = 0.f;
                    mapy.at<float>(j,i) = 0.f;
                }
                break;
            case UPSIDE_DOWN:
                mapx.at<float>(j,i) = static_cast<float>(i);
                mapy.at<float>(j,i) = static_cast<float>(mapx.rows - j);
                break;
            case REFLECTION_X:
                mapx.at<float>(j,i) = static_cast<float>(mapx.cols - i);
                mapy.at<float>(j,i) = static_cast<float>(j);
                break;
            case REFLECTION_BOTH:
                mapx.at<float>(j,i) = static_cast<float>(mapx.cols - i);
                mapy.at<float>(j,i) = static_cast<float>(mapx.rows - j);
                break;
            } // end of switch
        }
    }
}

static void test_remap(int remapMode)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC1, 0, 1);

    Mat mapx(size, CV_32FC1);
    Mat mapy(size, CV_32FC1);
    generateMap(mapx, mapy, remapMode);

    GpuMat_<float> d_src(src);
    GpuMat_<float> d_mapx(mapx);
    GpuMat_<float> d_mapy(mapy);

    GpuMat_<float> dst = remap_(interNearest(brdReplicate(d_src)), d_mapx, d_mapy);

    Mat dst_gold;
    cv::remap(src, dst_gold, mapx, mapy, INTER_NEAREST, BORDER_REPLICATE);

    EXPECT_MAT_NEAR(dst_gold, dst, 0.0);
}

TEST(Remap, HALF_SIZE)
{
    test_remap(HALF_SIZE);
}

TEST(Remap, UPSIDE_DOWN)
{
    test_remap(UPSIDE_DOWN);
}

TEST(Remap, REFLECTION_X)
{
    test_remap(REFLECTION_X);
}

TEST(Remap, REFLECTION_BOTH)
{
    test_remap(REFLECTION_BOTH);
}

// resize

TEST(Resize, Upscale)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC1, 0, 1);

    GpuMat_<float> d_src(src);
    Texture<float> tex_src(d_src);

    GpuMat_<float> dst1 = resize_(interCubic(tex_src), 2, 2);

    Mat mapx(size.height * 2, size.width * 2, CV_32FC1);
    Mat mapy(size.height * 2, size.width * 2, CV_32FC1);

    for (int y = 0; y < mapx.rows; ++y)
    {
        for (int x = 0; x < mapx.cols; ++x)
        {
            mapx.at<float>(y, x) = static_cast<float>(x / 2);
            mapy.at<float>(y, x) = static_cast<float>(y / 2);
        }
    }

    GpuMat_<float> d_mapx(mapx);
    GpuMat_<float> d_mapy(mapy);

    GpuMat_<float> dst2 = remap_(interCubic(brdReplicate(d_src)), d_mapx, d_mapy);

    EXPECT_MAT_NEAR(dst1, dst2, 0.0);
}

TEST(Resize, Downscale)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC1, 0, 1);
    const float fx = 1.0f / 3.0f;
    const float fy = 1.0f / 3.0f;

    GpuMat_<float> d_src(src);
    Texture<float> tex_src(d_src);

    GpuMat_<float> dst1 = resize_(interArea(tex_src, Size(3, 3)), fx, fy);

    Mat mapx(cv::saturate_cast<int>(size.height * fy), cv::saturate_cast<int>(size.width * fx), CV_32FC1);
    Mat mapy(cv::saturate_cast<int>(size.height * fy), cv::saturate_cast<int>(size.width * fx), CV_32FC1);

    for (int y = 0; y < mapx.rows; ++y)
    {
        for (int x = 0; x < mapx.cols; ++x)
        {
            mapx.at<float>(y, x) = x / fx;
            mapy.at<float>(y, x) = y / fy;
        }
    }

    GpuMat_<float> d_mapx(mapx);
    GpuMat_<float> d_mapy(mapy);

    GpuMat_<float> dst2 = remap_(interArea(brdReplicate(d_src), Size(3, 3)), d_mapx, d_mapy);

    EXPECT_MAT_NEAR(dst1, dst2, 0.0);
}

// warpAffine & warpPerspective

Mat createAffineTransfomMatrix(Size srcSize, float angle, bool perspective)
{
    cv::Mat M(perspective ? 3 : 2, 3, CV_32FC1);

    {
        M.at<float>(0, 0) = std::cos(angle); M.at<float>(0, 1) = -std::sin(angle); M.at<float>(0, 2) = static_cast<float>(srcSize.width / 2);
        M.at<float>(1, 0) = std::sin(angle); M.at<float>(1, 1) =  std::cos(angle); M.at<float>(1, 2) = 0.0f;
    }
    if (perspective)
    {
        M.at<float>(2, 0) = 0.0f           ; M.at<float>(2, 1) =  0.0f           ; M.at<float>(2, 2) = 1.0f;
    }

    return M;
}

TEST(WarpAffine, Rotation)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC1, 0, 1);
    Mat M = createAffineTransfomMatrix(size, static_cast<float>(CV_PI / 4), false);

    GpuMat_<float> d_src(src);
    GpuMat_<float> d_M;
    createContinuous(M.size(), M.type(), d_M);
    d_M.upload(M);

    GpuMat_<float> dst = warpAffine_(interNearest(brdConstant(d_src)), size, d_M);

    Mat dst_gold;
    cv::warpAffine(src, dst_gold, M, size, INTER_NEAREST | WARP_INVERSE_MAP);

    EXPECT_MAT_SIMILAR(dst_gold, dst, 1e-3);
}

TEST(WarpPerspective, Rotation)
{
    const Size size = randomSize(100, 400);

    Mat src = randomMat(size, CV_32FC1, 0, 1);
    Mat M = createAffineTransfomMatrix(size, static_cast<float>(CV_PI / 4), true);

    GpuMat_<float> d_src(src);
    GpuMat_<float> d_M;
    createContinuous(M.size(), M.type(), d_M);
    d_M.upload(M);

    GpuMat_<float> dst = warpPerspective_(interNearest(brdConstant(d_src)), size, d_M);

    Mat dst_gold;
    cv::warpPerspective(src, dst_gold, M, size, INTER_NEAREST | WARP_INVERSE_MAP);

    EXPECT_MAT_SIMILAR(dst_gold, dst, 1e-3);
}
