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

using namespace cvtest;

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HoughLines

PARAM_TEST_CASE(HoughLines, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    static void generateLines(cv::Mat& img)
    {
        img.setTo(cv::Scalar::all(0));

        cv::line(img, cv::Point(20, 0), cv::Point(20, img.rows), cv::Scalar::all(255));
        cv::line(img, cv::Point(0, 50), cv::Point(img.cols, 50), cv::Scalar::all(255));
        cv::line(img, cv::Point(0, 0), cv::Point(img.cols, img.rows), cv::Scalar::all(255));
        cv::line(img, cv::Point(img.cols, 0), cv::Point(0, img.rows), cv::Scalar::all(255));
    }

    static void drawLines(cv::Mat& dst, const std::vector<cv::Vec2f>& lines)
    {
        dst.setTo(cv::Scalar::all(0));

        for (size_t i = 0; i < lines.size(); ++i)
        {
            float rho = lines[i][0], theta = lines[i][1];
            cv::Point pt1, pt2;
            double a = std::cos(theta), b = std::sin(theta);
            double x0 = a*rho, y0 = b*rho;
            pt1.x = cvRound(x0 + 1000*(-b));
            pt1.y = cvRound(y0 + 1000*(a));
            pt2.x = cvRound(x0 - 1000*(-b));
            pt2.y = cvRound(y0 - 1000*(a));
            cv::line(dst, pt1, pt2, cv::Scalar::all(255));
        }
    }
};

GPU_TEST_P(HoughLines, Accuracy)
{
    const cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());
    const cv::Size size = GET_PARAM(1);
    const bool useRoi = GET_PARAM(2);

    const float rho = 1.0f;
    const float theta = (float) (1.5 * CV_PI / 180.0);
    const int threshold = 100;

    cv::Mat src(size, CV_8UC1);
    generateLines(src);

    cv::gpu::GpuMat d_lines;
    cv::gpu::HoughLines(loadMat(src, useRoi), d_lines, rho, theta, threshold);

    std::vector<cv::Vec2f> lines;
    cv::gpu::HoughLinesDownload(d_lines, lines);

    cv::Mat dst(size, CV_8UC1);
    drawLines(dst, lines);

    ASSERT_MAT_NEAR(src, dst, 0.0);
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, HoughLines, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// HoughCircles

PARAM_TEST_CASE(HoughCircles, cv::gpu::DeviceInfo, cv::Size, UseRoi)
{
    static void drawCircles(cv::Mat& dst, const std::vector<cv::Vec3f>& circles, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));

        for (size_t i = 0; i < circles.size(); ++i)
            cv::circle(dst, cv::Point2f(circles[i][0], circles[i][1]), (int)circles[i][2], cv::Scalar::all(255), fill ? -1 : 1);
    }
};

GPU_TEST_P(HoughCircles, Accuracy)
{
    const cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());
    const cv::Size size = GET_PARAM(1);
    const bool useRoi = GET_PARAM(2);

    const float dp = 2.0f;
    const float minDist = 0.0f;
    const int minRadius = 10;
    const int maxRadius = 20;
    const int cannyThreshold = 100;
    const int votesThreshold = 20;

    std::vector<cv::Vec3f> circles_gold(4);
    circles_gold[0] = cv::Vec3i(20, 20, minRadius);
    circles_gold[1] = cv::Vec3i(90, 87, minRadius + 3);
    circles_gold[2] = cv::Vec3i(30, 70, minRadius + 8);
    circles_gold[3] = cv::Vec3i(80, 10, maxRadius);

    cv::Mat src(size, CV_8UC1);
    drawCircles(src, circles_gold, true);

    cv::gpu::GpuMat d_circles;
    cv::gpu::HoughCircles(loadMat(src, useRoi), d_circles, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);

    std::vector<cv::Vec3f> circles;
    cv::gpu::HoughCirclesDownload(d_circles, circles);

    ASSERT_FALSE(circles.empty());

    for (size_t i = 0; i < circles.size(); ++i)
    {
        cv::Vec3f cur = circles[i];

        bool found = false;

        for (size_t j = 0; j < circles_gold.size(); ++j)
        {
            cv::Vec3f gold = circles_gold[j];

            if (std::fabs(cur[0] - gold[0]) < 5 && std::fabs(cur[1] - gold[1]) < 5 && std::fabs(cur[2] - gold[2]) < 5)
            {
                found = true;
                break;
            }
        }

        ASSERT_TRUE(found);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, HoughCircles, testing::Combine(
    ALL_DEVICES,
    DIFFERENT_SIZES,
    WHOLE_SUBMAT));

///////////////////////////////////////////////////////////////////////////////////////////////////////
// GeneralizedHough

PARAM_TEST_CASE(GeneralizedHough, cv::gpu::DeviceInfo, UseRoi)
{
};

GPU_TEST_P(GeneralizedHough, POSITION)
{
    const cv::gpu::DeviceInfo devInfo = GET_PARAM(0);
    cv::gpu::setDevice(devInfo.deviceID());
    const bool useRoi = GET_PARAM(1);

    cv::Mat templ = readImage("../cv/shared/templ.png", cv::IMREAD_GRAYSCALE);
    ASSERT_FALSE(templ.empty());

    cv::Point templCenter(templ.cols / 2, templ.rows / 2);

    const size_t gold_count = 3;
    cv::Point pos_gold[gold_count];
    pos_gold[0] = cv::Point(templCenter.x + 10, templCenter.y + 10);
    pos_gold[1] = cv::Point(2 * templCenter.x + 40, templCenter.y + 10);
    pos_gold[2] = cv::Point(2 * templCenter.x + 40, 2 * templCenter.y + 40);

    cv::Mat image(templ.rows * 3, templ.cols * 3, CV_8UC1, cv::Scalar::all(0));
    for (size_t i = 0; i < gold_count; ++i)
    {
        cv::Rect rec(pos_gold[i].x - templCenter.x, pos_gold[i].y - templCenter.y, templ.cols, templ.rows);
        cv::Mat imageROI = image(rec);
        templ.copyTo(imageROI);
    }

    cv::Ptr<cv::gpu::GeneralizedHough_GPU> hough = cv::gpu::GeneralizedHough_GPU::create(cv::GHT_POSITION);
    hough->set("votesThreshold", 200);

    hough->setTemplate(loadMat(templ, useRoi));

    cv::gpu::GpuMat d_pos;
    hough->detect(loadMat(image, useRoi), d_pos);

    std::vector<cv::Vec4f> pos;
    hough->download(d_pos, pos);

    ASSERT_EQ(gold_count, pos.size());

    for (size_t i = 0; i < gold_count; ++i)
    {
        cv::Point gold = pos_gold[i];

        bool found = false;

        for (size_t j = 0; j < pos.size(); ++j)
        {
            cv::Point2f p(pos[j][0], pos[j][1]);

            if (::fabs(p.x - gold.x) < 2 && ::fabs(p.y - gold.y) < 2)
            {
                found = true;
                break;
            }
        }

        ASSERT_TRUE(found);
    }
}

INSTANTIATE_TEST_CASE_P(GPU_ImgProc, GeneralizedHough, testing::Combine(
    ALL_DEVICES,
    WHOLE_SUBMAT));

#endif // HAVE_CUDA
