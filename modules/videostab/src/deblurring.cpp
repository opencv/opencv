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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/videostab/deblurring.hpp"
#include "opencv2/videostab/global_motion.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

float calcBlurriness(const Mat &frame)
{
    Mat Gx, Gy;
    Sobel(frame, Gx, CV_32F, 1, 0);
    Sobel(frame, Gy, CV_32F, 0, 1);
    double normGx = norm(Gx);
    double normGy = norm(Gx);
    double sumSq = normGx*normGx + normGy*normGy;
    return static_cast<float>(1. / (sumSq / frame.size().area() + 1e-6));
}


WeightingDeblurer::WeightingDeblurer()
{
    setSensitivity(0.1f);
}


void WeightingDeblurer::deblur(int idx, Mat &frame)
{
    CV_Assert(frame.type() == CV_8UC3);

    bSum_.create(frame.size());
    gSum_.create(frame.size());
    rSum_.create(frame.size());
    wSum_.create(frame.size());

    for (int y = 0; y < frame.rows; ++y)
    {
        for (int x = 0; x < frame.cols; ++x)
        {
            Point3_<uchar> p = frame.at<Point3_<uchar> >(y,x);
            bSum_(y,x) = p.x;
            gSum_(y,x) = p.y;
            rSum_(y,x) = p.z;
            wSum_(y,x) = 1.f;
        }
    }

    for (int k = idx - radius_; k <= idx + radius_; ++k)
    {
        const Mat &neighbor = at(k, *frames_);
        float bRatio = at(idx, *blurrinessRates_) / at(k, *blurrinessRates_);
        Mat_<float> M = getMotion(idx, k, *motions_);

        if (bRatio > 1.f)
        {
            for (int y = 0; y < frame.rows; ++y)
            {
                for (int x = 0; x < frame.cols; ++x)
                {
                    int x1 = static_cast<int>(M(0,0)*x + M(0,1)*y + M(0,2));
                    int y1 = static_cast<int>(M(1,0)*x + M(1,1)*y + M(1,2));

                    if (x1 >= 0 && x1 < neighbor.cols && y1 >= 0 && y1 < neighbor.rows)
                    {
                        const Point3_<uchar> &p = frame.at<Point3_<uchar> >(y,x);
                        const Point3_<uchar> &p1 = neighbor.at<Point3_<uchar> >(y1,x1);
                        float w = bRatio * sensitivity_ /
                                (sensitivity_ + std::abs(intensity(p1) - intensity(p)));
                        bSum_(y,x) += w * p1.x;
                        gSum_(y,x) += w * p1.y;
                        rSum_(y,x) += w * p1.z;
                        wSum_(y,x) += w;
                    }
                }
            }
        }
    }

    for (int y = 0; y < frame.rows; ++y)
    {
        for (int x = 0; x < frame.cols; ++x)
        {
            float wSumInv = 1.f / wSum_(y,x);
            frame.at<Point3_<uchar> >(y,x) = Point3_<uchar>(
                    static_cast<uchar>(bSum_(y,x)*wSumInv),
                    static_cast<uchar>(gSum_(y,x)*wSumInv),
                    static_cast<uchar>(rSum_(y,x)*wSumInv));
        }
    }
}

} // namespace videostab
} // namespace cv
