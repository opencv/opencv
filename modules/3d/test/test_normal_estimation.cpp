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

    float theta_thr = 1.f; // degree of angle between normal of plane and normal of point
    float curvature_thr = 0.01f; // threshold for curvature and actual curvature of the point
    float actual_curvature = 0.f;

    Point3f n1(model[0], model[1], model[2]);
    float n1m = n1.dot(n1);
    float total_theta = 0.f;
    float total_diff_curvature = 0.f;
    for (int i = 0; i < num; ++i)
    {
        float n12 = n1.dot(normals[i]);
        float n2m = normals[i].dot(normals[i]);
        float cos_theta = n12 / sqrt(n1m * n2m);
        // accuracy problems caused by float numbers, need to be fixed
        cos_theta = cos_theta > 1 ? 1 : cos_theta;
        cos_theta = cos_theta < 0 ? 0 : cos_theta;
        float theta = acos(cos_theta);

        total_theta += theta;
        total_diff_curvature += abs(curvatures[i] - actual_curvature);
    }

    float avg_theta = total_theta / (float) num;
    ASSERT_LE(avg_theta, theta_thr);

    float avg_diff_curvature = total_diff_curvature / (float) num;
    ASSERT_LE(avg_diff_curvature, curvature_thr);
}

TEST(NormalEstimationTest, SphereNormalEstimation)
{
    // generate a sphere for test
    vector<Point3f> sphere_pts;
    vector<float> model({0, 0, 0, 1});
    float thr = 0.f;
    int num = 1000;
    vector<float> limit({-1, 1, -1, 1, -1, 1});
    generateSphere(sphere_pts, model, thr, num, limit);

    // get knn search result
    Mat knn_idx;
    int k = 10;
    getKNNSearchResultsByKDTree(knn_idx, noArray(), sphere_pts, k, nullptr);

    // estimate normal and curvature
    vector<Point3f> normals;
    vector<float> curvatures;
    normalEstimate(normals, curvatures, sphere_pts, knn_idx, k);

    float theta_thr = 1.f; // degree of angle between normal of plane and normal of point
    float curvature_thr = 0.01f; // threshold for curvature and actual curvature of the point
    float actual_curvature = 0.f;

    float total_theta = 0.f;
    float total_diff_curvature = 0.f;
    for (int i = 0; i < num; ++i)
    {
        Point3f n1(sphere_pts[i]);
        Point3f n2 = normals[i];
        float n12 = n1.dot(n2);
        float n1m = n1.dot(n1);
        float n2m = n2.dot(n2);
        float cos_theta = n12 / sqrt(n1m * n2m);
        // accuracy problems caused by float numbers, need to be fixed
        cos_theta = cos_theta > 1 ? 1 : cos_theta;
        cos_theta = cos_theta < 0 ? 0 : cos_theta;
        float theta = acos(cos_theta);

        total_theta += theta;
        total_diff_curvature += abs(curvatures[i] - actual_curvature);
    }

    float avg_theta = total_theta / (float) num;
    ASSERT_LE(avg_theta, theta_thr);

    float avg_diff_curvature = total_diff_curvature / (float) num;
    ASSERT_LE(avg_diff_curvature, curvature_thr);
}

} // namespace
} // opencv_test
