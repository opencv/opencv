// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

using namespace cv;

// Compare 2 points in different point clouds.
bool comparePoints(const Mat &m, int rm, const Mat &n, int rn) {
    Mat diff = m.row(rm) != n.row(rn);
    return countNonZero(diff) == 0;
}

// Check whether a point exists in the point cloud.
bool checkExistPoint(const Mat &pt, const Mat &ptcloud) {
    bool flag = false;
    for (int i = 0; i < ptcloud.rows; i++){
        if(comparePoints(ptcloud, i, pt, 0)){
            flag = true;
            break;
        }
    }
    return flag;
}

class SamplingTest : public ::testing::Test
{
protected:
    void SetUp() override {
        // Initialize a cube point cloud with 8 points as vertices.
        point_cloud = Mat(8, 3, CV_32F, point_cloud_info);
    }

public:
    float point_cloud_info[24] = {
            0, 0, 0,  0, 0, 1,  0, 1, 0,  0, 1, 1,
            1, 0, 0,  1, 0, 1,  1, 1, 0,  1, 1, 1
    };
    Mat point_cloud;
    Mat sampled_pts;
};


TEST_F(SamplingTest, VoxelGridFilterSampling) {
    // Set 1.1 as the side length, and there should be only one point after sampling.
    voxelGridSampling(sampled_pts, point_cloud, 1.1, 1.1, 1.1);
    EXPECT_EQ(sampled_pts.rows, 1);
    ASSERT_TRUE(checkExistPoint(sampled_pts.row(0), point_cloud));

    // Set (0.55, 0.55, 1.1) as the side length, and there should be 4 points after sampling.
    voxelGridSampling(sampled_pts, point_cloud, 0.55, 0.55, 1.1);
    EXPECT_EQ(sampled_pts.rows, 4);
    for(int i = 0; i < 4; i++){
        ASSERT_TRUE(checkExistPoint(sampled_pts.row(i), point_cloud));
    }
}

TEST_F(SamplingTest, RandomSampling) {
    // Set 1 as the size, and there should be only one point after sampling.
    randomSampling(sampled_pts, point_cloud, 4);
    EXPECT_EQ(sampled_pts.rows, 4);
    for(int i = 0; i < 4; i++){
        ASSERT_TRUE(checkExistPoint(sampled_pts.row(i), point_cloud));
    }
}

TEST_F(SamplingTest, FarthestPointSampling) {
    // Set 2 as the size, and there should be 2 diagonal points.
    farthestPointSampling(sampled_pts, point_cloud, 2);
    Mat check = sampled_pts.row(0) + sampled_pts.row(1);
    ASSERT_EQ(check.at<float>(0, 0), 1);
    ASSERT_EQ(check.at<float>(0, 1), 1);
    ASSERT_EQ(check.at<float>(0, 2), 1);
}

} // namespace
} // opencv_test