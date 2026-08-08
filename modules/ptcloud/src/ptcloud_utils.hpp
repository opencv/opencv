// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_PTCLOUD_UTILS_HPP
#define OPENCV_PTCLOUD_UTILS_HPP

#include "precomp.hpp"
#include "../../geometry/src/ptcloud/ptcloud_utils.hpp"   // cv::getPointsMatFromInputArray

namespace cv {

// Read an input point cloud (CV_32FC3, or Nx3 / 3xN CV_32F) into a vector<Point3f>.
// Layout handling is delegated to geometry's getPointsMatFromInputArray so there is a
// single shared implementation. Returns an empty vector for empty input.
static inline void toPointVec(InputArray inputCloud, std::vector<Point3f>& points)
{
    points.clear();
    if (inputCloud.empty())
        return;

    Mat mat;
    getPointsMatFromInputArray(inputCloud, mat, 0, true);  // N x 3, CV_32FC1 (row arrangement)
    mat = mat.reshape(3, mat.rows);                        // N x 1, CV_32FC3
    points.assign(mat.begin<Point3f>(), mat.end<Point3f>());
}

} // namespace cv

#endif // OPENCV_PTCLOUD_UTILS_HPP
