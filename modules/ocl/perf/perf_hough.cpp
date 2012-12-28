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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
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
//     and/or other oclMaterials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
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

#include "precomp.hpp"

#ifdef HAVE_OPENCL

using namespace cv;
using namespace perf;

//////////////////////////////////////////////////////////////////////
// HoughCircles

PARAM_TEST_CASE(HoughCircles_Perf, cv::Size, float, float)
{
    static void drawCircles(cv::Mat& dst, const std::vector<cv::Vec3f>& circles, bool fill)
    {
        dst.setTo(cv::Scalar::all(0));

        for (size_t i = 0; i < circles.size(); ++i)
            cv::circle(dst, cv::Point2f(circles[i][0], circles[i][1]), (int)circles[i][2], cv::Scalar::all(255), fill ? -1 : 1);
    }
};

TEST_P(HoughCircles_Perf, Performance)
{
    const cv::Size size = GET_PARAM(0);
    const float dp = GET_PARAM(1);
    const float minDist = GET_PARAM(2);

    const int minRadius = 10;
    const int maxRadius = 30;
    const int cannyThreshold = 100;
    const int votesThreshold = 15;

    cv::RNG rng(123456789);

    cv::Mat src(size, CV_8UC1, cv::Scalar::all(0));

    const int numCircles = rng.uniform(50, 100);
    for (int i = 0; i < numCircles; ++i)
    {
        cv::Point center(rng.uniform(0, src.cols), rng.uniform(0, src.rows));
        const int radius = rng.uniform(minRadius, maxRadius + 1);

        cv::circle(src, center, radius, cv::Scalar::all(255), -1);
    }
    
    cv::ocl::oclMat d_circles;

    double totalgputick = 0;
    double totalgputick_kernel = 0;
    
    double t1 = 0.0;
    double t2 = 0.0;
    for (int j = 0; j < LOOP_TIMES + 1; ++j)
    {

        t1 = (double)cvGetTickCount();//gpu start1

        cv::ocl::oclMat ocl_src = cv::ocl::oclMat(src);//upload

        t2 = (double)cvGetTickCount(); //kernel
        cv::ocl::HoughCircles(ocl_src, d_circles, CV_HOUGH_GRADIENT, dp, minDist, cannyThreshold, votesThreshold, minRadius, maxRadius);
        t2 = (double)cvGetTickCount() - t2;//kernel
    
        cv::Mat cpu_dst;
        if (d_circles.rows > 0)
            d_circles.download (cpu_dst);//download

        t1 = (double)cvGetTickCount() - t1;//gpu end1

        if(j == 0)
            continue;

        totalgputick = t1 + totalgputick;

        totalgputick_kernel = t2 + totalgputick_kernel;        
    }

    std::cout << "average gpu runtime is  " << totalgputick / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << std::endl;
    std::cout << "average gpu runtime without data transfer is  " << totalgputick_kernel / ((double)cvGetTickFrequency()* LOOP_TIMES * 1000.) << "ms" << std::endl;
    
}


INSTANTIATE_TEST_CASE_P(Hough, HoughCircles_Perf,
                        testing::Combine(
                            testing::Values(perf::sz720p, perf::szSXGA, perf::sz1080p),
                            testing::Values(1.0f, 2.0f, 4.0f),
                            testing::Values(1.0f, 10.0f)));

#endif // HAVE_OPENCL
