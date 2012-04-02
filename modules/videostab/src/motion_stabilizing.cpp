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
#include "opencv2/videostab/motion_stabilizing.hpp"
#include "opencv2/videostab/global_motion.hpp"

using namespace std;

namespace cv
{
namespace videostab
{

void MotionFilterBase::stabilize(const Mat *motions, int size, Mat *stabilizationMotions) const
{
    for (int i = 0; i < size; ++i)
        stabilizationMotions[i] = stabilize(i, motions, size);
}


void GaussianMotionFilter::setParams(int radius, float stdev)
{
    radius_ = radius;
    stdev_ = stdev > 0.f ? stdev : sqrt(static_cast<float>(radius_));

    float sum = 0;
    weight_.resize(2*radius_ + 1);
    for (int i = -radius_; i <= radius_; ++i)
        sum += weight_[radius_ + i] = std::exp(-i*i/(stdev_*stdev_));
    for (int i = -radius_; i <= radius_; ++i)
        weight_[radius_ + i] /= sum;
}


Mat GaussianMotionFilter::stabilize(int index, const Mat *motions, int size) const
{
    const Mat &cur = at(index, motions, size);
    Mat res = Mat::zeros(cur.size(), cur.type());
    float sum = 0.f;
    for (int i = std::max(index - radius_, 0); i <= index + radius_; ++i)
    {
        res += weight_[radius_ + i - index] * getMotion(index, i, motions, size);
        sum += weight_[radius_ + i - index];
    }
    return res / sum;
}


static inline int areaSign(Point2f a, Point2f b, Point2f c)
{
    double area = (b-a).cross(c-a);
    if (area < -1e-5) return -1;
    if (area > 1e-5) return 1;
    return 0;
}


static inline bool segmentsIntersect(Point2f a, Point2f b, Point2f c, Point2f d)
{
    return areaSign(a,b,c) * areaSign(a,b,d) < 0 &&
           areaSign(c,d,a) * areaSign(c,d,b) < 0;
}


// Checks if rect a (with sides parallel to axis) is inside rect b (arbitrary).
// Rects must be passed in the [(0,0), (w,0), (w,h), (0,h)] order.
static inline bool isRectInside(const Point2f a[4], const Point2f b[4])
{
    for (int i = 0; i < 4; ++i)
        if (b[i].x > a[0].x && b[i].x < a[2].x && b[i].y > a[0].y && b[i].y < a[2].y)
            return false;
    for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j)
        if (segmentsIntersect(a[i], a[(i+1)%4], b[j], b[(j+1)%4]))
            return false;
    return true;
}


static inline bool isGoodMotion(const float M[], float w, float h, float dx, float dy)
{
    Point2f pt[4] = {Point2f(0,0), Point2f(w,0), Point2f(w,h), Point2f(0,h)};
    Point2f Mpt[4];

    for (int i = 0; i < 4; ++i)
    {
        Mpt[i].x = M[0]*pt[i].x + M[1]*pt[i].y + M[2];
        Mpt[i].y = M[3]*pt[i].x + M[4]*pt[i].y + M[5];
    }

    pt[0] = Point2f(dx, dy);
    pt[1] = Point2f(w - dx, dy);
    pt[2] = Point2f(w - dx, h - dy);
    pt[3] = Point2f(dx, h - dy);

    return isRectInside(pt, Mpt);
}


static inline void relaxMotion(const float M[], float t, float res[])
{
    res[0] = M[0]*(1.f-t) + t;
    res[1] = M[1]*(1.f-t);
    res[2] = M[2]*(1.f-t);
    res[3] = M[3]*(1.f-t);
    res[4] = M[4]*(1.f-t) + t;
    res[5] = M[5]*(1.f-t);
}


Mat ensureInclusionConstraint(const Mat &M, Size size, float trimRatio)
{
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    const float w = static_cast<float>(size.width);
    const float h = static_cast<float>(size.height);
    const float dx = floor(w * trimRatio);
    const float dy = floor(h * trimRatio);
    const float srcM[6] =
            {M.at<float>(0,0), M.at<float>(0,1), M.at<float>(0,2),
             M.at<float>(1,0), M.at<float>(1,1), M.at<float>(1,2)};

    float curM[6];
    float t = 0;
    relaxMotion(srcM, t, curM);
    if (isGoodMotion(curM, w, h, dx, dy))
        return M;

    float l = 0, r = 1;
    while (r - l > 1e-3f)
    {
        t = (l + r) * 0.5f;
        relaxMotion(srcM, t, curM);
        if (isGoodMotion(curM, w, h, dx, dy))
            r = t;
        else
            l = t;
        t = r;
        relaxMotion(srcM, r, curM);
    }

    return (1 - r) * M + r * Mat::eye(3, 3, CV_32F);
}


// TODO can be estimated for O(1) time
float estimateOptimalTrimRatio(const Mat &M, Size size)
{
    CV_Assert(M.size() == Size(3,3) && M.type() == CV_32F);

    const float w = static_cast<float>(size.width);
    const float h = static_cast<float>(size.height);
    Mat_<float> M_(M);

    Point2f pt[4] = {Point2f(0,0), Point2f(w,0), Point2f(w,h), Point2f(0,h)};
    Point2f Mpt[4];

    for (int i = 0; i < 4; ++i)
    {
        Mpt[i].x = M_(0,0)*pt[i].x + M_(0,1)*pt[i].y + M_(0,2);
        Mpt[i].y = M_(1,0)*pt[i].x + M_(1,1)*pt[i].y + M_(1,2);
    }

    float l = 0, r = 0.5f;
    while (r - l > 1e-3f)
    {
        float t = (l + r) * 0.5f;
        float dx = floor(w * t);
        float dy = floor(h * t);
        pt[0] = Point2f(dx, dy);
        pt[1] = Point2f(w - dx, dy);
        pt[2] = Point2f(w - dx, h - dy);
        pt[3] = Point2f(dx, h - dy);
        if (isRectInside(pt, Mpt))
            r = t;
        else
            l = t;
    }

    return r;
}

} // namespace videostab
} // namespace cv
