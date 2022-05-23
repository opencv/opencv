// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#ifndef OPENCV_TEST_PTCLOUD_UTILS_HPP
#define OPENCV_TEST_PTCLOUD_UTILS_HPP

#include "test_precomp.hpp"

namespace opencv_test {

/**
 * @brief Generate a specific plane with random points.
 *
 * @param[out] plane_pts Point cloud of plane, only support vector<Point3f> or Mat with Nx3 layout
 *                       in memory.
 * @param model Plane coefficient [a,b,c,d] means ax+by+cz+d=0.
 * @param thr Generate the maximum distance from the point to the plane.
 * @param num The number of points.
 * @param limit The range of xyz coordinates of the generated plane.
 *
 */
void generatePlane(OutputArray plane_pts, const vector<float> &model, float thr, int num,
        const vector<float> &limit);

/**
 * @brief Generate a specific sphere with random points.
 *
 * @param[out] sphere_pts Point cloud of plane, only support vector<Point3f> or Mat with Nx3 layout
 *                       in memory.
 * @param model Plane coefficient [a,b,c,d] means x^2+y^2+z^2=r^2.
 * @param thr Generate the maximum distance from the point to the surface of sphere.
 * @param num The number of points.
 * @param limit The range of vector to make the generated sphere incomplete.
 *
 */
void generateSphere(OutputArray sphere_pts, const vector<float> &model, float thr, int num,
        const vector<float> &limit);

} // opencv_test
#endif //OPENCV_TEST_PTCLOUD_UTILS_HPP
