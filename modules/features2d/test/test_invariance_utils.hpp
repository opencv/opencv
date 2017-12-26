// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef __OPENCV_TEST_INVARIANCE_UTILS_HPP__
#define __OPENCV_TEST_INVARIANCE_UTILS_HPP__

#include "test_precomp.hpp"

using namespace std;
using namespace cv;

static
Mat generateHomography(float angle)
{
    // angle - rotation around Oz in degrees
    float angleRadian = static_cast<float>(angle * CV_PI / 180);
    Mat H = Mat::eye(3, 3, CV_32FC1);
    H.at<float>(0,0) = H.at<float>(1,1) = std::cos(angleRadian);
    H.at<float>(0,1) = -std::sin(angleRadian);
    H.at<float>(1,0) =  std::sin(angleRadian);

    return H;
}

static
Mat rotateImage(const Mat& srcImage, const Mat& srcMask, float angle, Mat& dstImage, Mat& dstMask)
{
    // angle - rotation around Oz in degrees
    float diag = std::sqrt(static_cast<float>(srcImage.cols * srcImage.cols + srcImage.rows * srcImage.rows));
    Mat LUShift = Mat::eye(3, 3, CV_32FC1); // left up
    LUShift.at<float>(0,2) = static_cast<float>(-srcImage.cols/2);
    LUShift.at<float>(1,2) = static_cast<float>(-srcImage.rows/2);
    Mat RDShift = Mat::eye(3, 3, CV_32FC1); // right down
    RDShift.at<float>(0,2) = diag/2;
    RDShift.at<float>(1,2) = diag/2;
    Size sz(cvRound(diag), cvRound(diag));

    Mat H = RDShift * generateHomography(angle) * LUShift;
    warpPerspective(srcImage, dstImage, H, sz);
    warpPerspective(srcMask, dstMask, H, sz);

    return H;
}

static
float calcCirclesIntersectArea(const Point2f& p0, float r0, const Point2f& p1, float r1)
{
    float c = static_cast<float>(norm(p0 - p1)), sqr_c = c * c;

    float sqr_r0 = r0 * r0;
    float sqr_r1 = r1 * r1;

    if(r0 + r1 <= c)
       return 0;

    float minR = std::min(r0, r1);
    float maxR = std::max(r0, r1);
    if(c + minR <= maxR)
        return static_cast<float>(CV_PI * minR * minR);

    float cos_halfA0 = (sqr_r0 + sqr_c - sqr_r1) / (2 * r0 * c);
    float cos_halfA1 = (sqr_r1 + sqr_c - sqr_r0) / (2 * r1 * c);

    float A0 = 2 * acos(cos_halfA0);
    float A1 = 2 * acos(cos_halfA1);

    return  0.5f * sqr_r0 * (A0 - sin(A0)) +
            0.5f * sqr_r1 * (A1 - sin(A1));
}

static
float calcIntersectRatio(const Point2f& p0, float r0, const Point2f& p1, float r1)
{
    float intersectArea = calcCirclesIntersectArea(p0, r0, p1, r1);
    float unionArea = static_cast<float>(CV_PI) * (r0 * r0 + r1 * r1) - intersectArea;
    return intersectArea / unionArea;
}

static
void scaleKeyPoints(const vector<KeyPoint>& src, vector<KeyPoint>& dst, float scale)
{
    dst.resize(src.size());
    for (size_t i = 0; i < src.size(); i++) {
        dst[i] = src[i];
        dst[i].pt.x *= scale;
        dst[i].pt.y *= scale;
        dst[i].size *= scale;
    }
}

#endif // __OPENCV_TEST_INVARIANCE_UTILS_HPP__
