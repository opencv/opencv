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
bool checkExistPoint(const Mat &ptCloud, const Mat &pt) {
    bool flag = false;
    for (int i = 0; i < ptCloud.rows; i++){
        if(comparePoints(ptCloud, i, pt, 0)){
            flag = true;
            break;
        }
    }
    return flag;
}

// Change mask to sampled point cloud
void maskToPointCloud(const Mat &ptCloud, const vector<bool> &mask, Mat &sampledPts){
    for(int i = 0 ; i < mask.size(); i++){
        if(mask.at(i))
            sampledPts.push_back(ptCloud.row(i));
    }
}

class SamplingTest : public ::testing::Test
{
protected:
    void SetUp() override {
        // Initialize a cube point cloud with 8 points as vertices.
        ptCloud = Mat(sizeof(ptCloudInfo)/sizeof(ptCloudInfo[0])/3, 3, CV_32F, ptCloudInfo);
    }

    void TearDown() override{
        ptCloud.release();
        sampledPts.release();
    }

public:
    float ptCloudInfo[24] = {
            0, 0, 0,  0, 0, 1,  0, 1, 0,  0, 1, 1,
            1, 0, 0,  1, 0, 1,  1, 1, 0,  1, 1, 1
    };
    Mat ptCloud;
    Mat sampledPts;
    vector<bool> mask;
};


TEST_F(SamplingTest, VoxelGridFilterSampling) {
    // Set 1.1 as the side length, and there should be only one point after sampling.
    voxelGridSampling(mask, ptCloud, 1.1f, 1.1f, 1.1f);
    maskToPointCloud(ptCloud, mask, sampledPts);
    // Sampled point cloud should have only 1 point.
    EXPECT_EQ(sampledPts.rows, 1);
    // The point should be in a box with a side length of 1.1.
    ASSERT_TRUE(0.0f <= sampledPts.at<float>(0, 0) && sampledPts.at<float>(0, 0) < 1.1f);
    ASSERT_TRUE(0.0f <= sampledPts.at<float>(0, 1) && sampledPts.at<float>(0, 1) < 1.1f);
    ASSERT_TRUE(0.0f <= sampledPts.at<float>(0, 2) && sampledPts.at<float>(0, 2) < 1.1f);

    // Set (0.55, 0.55, 1.1) as the side length, and there should be 4 points after sampling.
    sampledPts.release();
    voxelGridSampling(mask, ptCloud, 0.55f, 0.55f, 1.1f);
    maskToPointCloud(ptCloud, mask, sampledPts);
    // Sampled point cloud should have 4 points.
    EXPECT_EQ(sampledPts.rows, 4);
    // All points should be in 4 different boxes.
    float x[4] = {0.0f, 0.0f, 0.55f, 0.55f}, y[4] = {0.0f, 0.55f, 0.0f, 0.55f};
    for(int i = 0; i < 4; i++){
        bool flag;
        for(int j = 0; j < 4; j++){
            flag =  x[i] <= sampledPts.at<float>(j, 0) && sampledPts.at<float>(j, 0) < x[i] + 0.55f &&
                    y[i] <= sampledPts.at<float>(j, 1) && sampledPts.at<float>(j, 1) < y[i] + 0.55f &&
                    0.0f <= sampledPts.at<float>(j, 2) && sampledPts.at<float>(j, 2) < 1.1f ;
            if(flag) break;
        }
        ASSERT_TRUE(flag);
    }
}

TEST_F(SamplingTest, RandomSampling) {
    // Set 1 as the size, and there should be only one point after sampling.
    randomSampling(sampledPts, ptCloud, 4);
    EXPECT_EQ(sampledPts.rows, 4);
    for(int i = 0; i < 4; i++){
        ASSERT_TRUE(checkExistPoint(ptCloud, sampledPts.row(i)));
    }
}

TEST_F(SamplingTest, FarthestPointSampling) {
    // Set 2 as the size, and there should be 2 diagonal points after sampling.
    farthestPointSampling(mask, ptCloud, 2);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(sampledPts.rows, 2);
    Mat check = sampledPts.row(0) + sampledPts.row(1);
    ASSERT_EQ(check.at<float>(0, 0), 1);
    ASSERT_EQ(check.at<float>(0, 1), 1);
    ASSERT_EQ(check.at<float>(0, 2), 1);


}

} // namespace
} // opencv_test