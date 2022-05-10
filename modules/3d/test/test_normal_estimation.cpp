// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "test_precomp.hpp"
#include "test_ptcloud_utils.hpp"

namespace opencv_test { namespace {

TEST(NormalEstimationTest, PlaneNormalEstimation)
{
    // generate a plane for test
    vector<Point3f> plane_pts;
    vector<float> model({1, 2, 3, 4});
    float thr = 0.f;
    int num = 1000;
    vector<float> limit({5, 55, 5, 55, 0, 0});
    generatePlane(plane_pts, model, thr, num, limit);

    // get knn search result
    Mat knn_idx;
    int k = 10;
    getKNNSearchResultsByKDTree(knn_idx, noArray(), plane_pts, k);

    // estimate normal and curvature
    vector<Point3f> normals;
    vector<float> curvatures;
    normalEstimate(normals, curvatures, plane_pts, knn_idx, k);

    // check each normal
    Point3f plane_norm(model[0], model[1], model[2]);
    float theta = 1.f; // degree of angle between normal of plane and normal of point
    float cos_theta = cos(theta);
    for (Point3f norm: normals)
    {
        float n12 = norm.dot(plane_norm);
        float n1m = norm.dot(norm);
        float n2m = plane_norm.dot(plane_norm);
        ASSERT_GE(n12 * n12 / (n1m * n2m), cos_theta * cos_theta);
    }

    // check each curvature
    float actual_curvature = 0;
    float curvature_thr = 0.0001; // threshold for curvature and actual curvature of the point
    for (float cur: curvatures)
    {
        ASSERT_LE(abs(cur - actual_curvature), curvature_thr);
    }
}

TEST(NormalEstimationTest, SphereNormalEstimation)
{
    // generate a sphere for test
    vector<Point3f> sphere_pts;
    vector<float> model({0, 0, 0, 10});
    float thr = 0.f;
    int num = 1000;
    vector<float> limit({-1, 1, -1, 1, -1, 1});
    generateSphere(sphere_pts, model, thr, num, limit);

    // get knn search result
    Mat knn_idx;
    int k = 10;
    getKNNSearchResultsByKDTree(knn_idx, noArray(), sphere_pts, k);

    // estimate normal and curvature
    vector<Point3f> normals;
    vector<float> curvatures;
    normalEstimate(normals, curvatures, sphere_pts, knn_idx, k);

    // check each normal
    float theta = 1; // degree of angle between normal of plane and normal of point
    float cos_theta = cos(theta);
    for (int i = 0; i < num; ++i)
    {
        Point3f norm = normals[i];
        Point3f actual_norm(sphere_pts[i]);
        float n12 = norm.dot(actual_norm);
        float n1m = norm.dot(norm);
        float n2m = actual_norm.dot(actual_norm);
        ASSERT_GE(n12 * n12 / (n1m * n2m), cos_theta * cos_theta);
    }

    // check each curvature
    float actual_curvature = 0;
    float curvature_thr = 0.01; // threshold for curvature and actual curvature of the point
    for (float cur: curvatures)
    {
        ASSERT_LE(abs(cur - actual_curvature), curvature_thr);
    }
}

} // namespace
} // opencv_test
