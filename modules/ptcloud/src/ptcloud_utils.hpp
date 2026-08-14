// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_PTCLOUD_UTILS_HPP
#define OPENCV_PTCLOUD_UTILS_HPP

#include "precomp.hpp"

namespace cv {

// Read an input point cloud (CV_32FC3, or Nx3 / 3xN CV_32F) into a vector<Point3f>.
// Returns an empty vector for empty input. Shared by the point-cloud processing sources.
static inline void toPointVec(InputArray inputCloud, std::vector<Point3f>& points)
{
    points.clear();
    Mat m = inputCloud.getMat();
    if (m.empty())
        return;

    Mat mf;
    if (m.depth() != CV_32F)
        m.convertTo(mf, CV_32F);
    else
        mf = m;

    // getMat() may return a non-contiguous view; reshape() below needs contiguous data.
    if (!mf.isContinuous())
        mf = mf.clone();

    if (mf.channels() == 1)
    {
        // Accept Nx3 (or 3xN) single-channel layouts as well.
        CV_Assert(mf.cols == 3 || mf.rows == 3);
        if (mf.cols != 3 && mf.rows == 3)
            mf = mf.t();
        mf = mf.reshape(3);            // N x 1, CV_32FC3
    }
    CV_Assert(mf.channels() == 3);

    mf = mf.reshape(3, (int)mf.total());   // guarantee N x 1, CV_32FC3
    points.assign(mf.begin<Point3f>(), mf.end<Point3f>());
}

// Normalize a vector; fall back to a fixed unit vector for (near-)zero length.
static inline Vec3f safeNormalize(const Vec3f& v)
{
    float len = (float)norm(v);
    return (len > 1e-12f) ? v * (1.f / len) : Vec3f(0.f, 0.f, 1.f);
}

} // namespace cv

#endif // OPENCV_PTCLOUD_UTILS_HPP
