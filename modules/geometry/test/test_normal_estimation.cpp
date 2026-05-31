// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "test_precomp.hpp"
#include "test_ptcloud_utils.hpp"
#include "opencv2/flann.hpp"

namespace opencv_test { namespace {

TEST(NormalEstimationTest, PlaneNormalEstimation)
{
    // generate a plane for test
    Mat plane_pts;
    vector<float> model({1, 2, 3, 4});
    float thr = 0.f;
    int num = 1000;
    vector<float> limit({5, 55, 5, 55, 0, 0});
    generatePlane(plane_pts, model, thr, num, limit);

    // get knn search result
    int k = 10;
    Mat knn_idx(num, k, CV_32S);
    // build kdtree
    flann::Index tree(plane_pts, flann::KDTreeIndexParams());
    tree.knnSearch(plane_pts, knn_idx, noArray(), k);
    // estimate normal and curvature
    vector<Point3f> normals;
    vector<float> curvatures;
    normalEstimate(normals, curvatures, plane_pts, knn_idx, k);

    float theta_thr = 1.f; // threshold for degree of angle between normal of plane and normal of point
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
        cos_theta = cos_theta < -1 ? -1 : cos_theta;
        float theta = acos(abs(cos_theta));

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
