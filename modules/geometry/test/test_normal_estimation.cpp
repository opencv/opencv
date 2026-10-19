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

// nn_idx left empty -> normalEstimate must build the kd-tree internally (k = max_neighbor_num)
// and produce the same plane normals as when the caller supplies the neighbor indices.
TEST(NormalEstimationTest, InternalKnnMatchesExternal)
{
    Mat plane_pts;
    vector<float> model({1, 2, 3, 4});
    int num = 1000;
    vector<float> limit({5, 55, 5, 55, 0, 0});
    generatePlane(plane_pts, model, 0.f, num, limit);

    const int k = 10;

    // reference: caller-supplied neighbors
    Mat knn_idx(num, k, CV_32S);
    flann::Index tree(plane_pts, flann::KDTreeIndexParams());
    tree.knnSearch(plane_pts, knn_idx, noArray(), k);
    vector<Point3f> ref;  vector<float> refCurv;
    normalEstimate(ref, refCurv, plane_pts, knn_idx, k);

    // new path: nn_idx empty -> built internally
    vector<Point3f> got;  vector<float> gotCurv;
    normalEstimate(got, gotCurv, plane_pts, noArray(), k);

    ASSERT_EQ(ref.size(), got.size());

    // The internal-kNN path must reproduce the caller-supplied-neighbor result.
    for (int i = 0; i < num; ++i)
    {
        float denom = std::sqrt(ref[i].dot(ref[i]) * got[i].dot(got[i])) + 1e-12f;
        EXPECT_NEAR(std::abs(ref[i].dot(got[i])) / denom, 1.f, 1e-3f);   // parallel, up to sign
    }

    // ...and both must be parallel to the analytic plane normal.
    Point3f n1(model[0], model[1], model[2]);
    float n1m = n1.dot(n1);
    float total_theta = 0.f;
    for (int i = 0; i < num; ++i)
    {
        float cos_theta = n1.dot(got[i]) / std::sqrt(n1m * got[i].dot(got[i]));
        cos_theta = std::max(-1.f, std::min(1.f, cos_theta));
        total_theta += std::acos(std::abs(cos_theta));
    }
    ASSERT_LE(total_theta / (float)num, 1.f);   // normals parallel to the plane normal
}

TEST(NormalEstimationTest, EmptyAndInvalidInputs)
{
    Mat empty;
    vector<Point3f> normals;
    vector<float> curvatures;
    normalEstimate(normals, curvatures, empty, noArray(), 5);
    EXPECT_TRUE(normals.empty());
    EXPECT_TRUE(curvatures.empty());

    Mat pts(2, 3, CV_32F, Scalar(0));
    std::vector<Mat> badIdx = { Mat(1, 1, CV_8U, Scalar(0)), Mat(1, 1, CV_8U, Scalar(0)) };
    EXPECT_THROW(normalEstimate(normals, curvatures, pts, badIdx, 1), cv::Exception);

    // Column-vector neighbor indices (e.g., std::vector<std::vector<int>> or Mat(K, 1, CV_32S))
    Mat plane_pts;
    vector<float> model({1, 2, 3, 4});
    int num = 50, k = 10;
    vector<float> limit({5, 55, 5, 55, 0, 0});
    generatePlane(plane_pts, model, 0.f, num, limit);

    Mat knn_idx(num, k, CV_32S);
    flann::Index tree(plane_pts, flann::KDTreeIndexParams());
    tree.knnSearch(plane_pts, knn_idx, noArray(), k);

    vector<Point3f> refNormals, colNormals;
    vector<float> refCurv, colCurv;
    normalEstimate(refNormals, refCurv, plane_pts, knn_idx, k);

    std::vector<std::vector<int>> col_knn_idx(num, std::vector<int>(k));
    for (int i = 0; i < num; ++i)
        for (int j = 0; j < k; ++j)
            col_knn_idx[i][j] = knn_idx.at<int>(i, j);
    normalEstimate(colNormals, colCurv, plane_pts, col_knn_idx, k);

    ASSERT_EQ(refNormals.size(), colNormals.size());
    for (int i = 0; i < num; ++i)
    {
        EXPECT_NEAR(refNormals[i].x, colNormals[i].x, 1e-6f);
        EXPECT_NEAR(refNormals[i].y, colNormals[i].y, 1e-6f);
        EXPECT_NEAR(refNormals[i].z, colNormals[i].z, 1e-6f);
        EXPECT_NEAR(refCurv[i], colCurv[i], 1e-6f);
    }
}

} // namespace
} // opencv_test
