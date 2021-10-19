// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

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
void maskToPointCloud(const Mat &ptCloud, const vector<char> &mask, Mat &sampledPts){
    sampledPts.release();
    for(int i = 0 ; i < mask.size(); i++){
        if(mask.at(i))
            sampledPts.push_back(ptCloud.row(i).clone());
    }
}

class SamplingTest : public ::testing::Test
{
protected:
    void TearDown() override{
        ptCloud.release();
        sampledPts.release();
    }

public:
    Mat ptCloud = Mat({8, 3}, {
            0.0f, 0.0f, 0.0f,  1.0f, 0.0f, 0.0f,  1.0f, 2.0f, 0.0f,  0.0f, 2.0f, 0.0f,
            0.0f, 0.0f, 3.0f,  1.0f, 0.0f, 3.0f,  1.0f, 2.0f, 3.0f,  0.0f, 2.0f, 3.0f});
    Mat sampledPts;
    vector<char> mask;
};


TEST_F(SamplingTest, VoxelGridFilterSampling) {
    // Set (1.1, 2.1, 3.1) as the side length, and there should be only one point after sampling.
    voxelGridSampling(mask, ptCloud, 1.1f, 2.1f, 3.1f);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(1, sampledPts.rows);
    // The point should be in a box with a side length of (1.1, 2.1, 3.1).
    ASSERT_TRUE(0.0f <= sampledPts.at<float>(0, 0) && sampledPts.at<float>(0, 0) < 1.1f &&
            0.0f <= sampledPts.at<float>(0, 1) && sampledPts.at<float>(0, 1) < 2.1f &&
            0.0f <= sampledPts.at<float>(0, 2) && sampledPts.at<float>(0, 2) < 3.1f);

    // Set (0.55, 1.05, 3.1) as the side length, and there should be 4 points after sampling.
    voxelGridSampling(mask, ptCloud, 0.55f, 1.05f, 3.1f);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(4, sampledPts.rows);
    // All points should be in 4 different boxes.
    float x[4] = {0.0f, 0.0f, 0.55f, 0.55f}, y[4] = {0.0f, 1.05f, 0.0f, 1.05f};
    for(int i = 0; i < 4; i++){
        bool flag;
        for(int j = 0; j < 4; j++){
            flag =  x[i] <= sampledPts.at<float>(j, 0) && sampledPts.at<float>(j, 0) < x[i] + 0.55f &&
                    y[i] <= sampledPts.at<float>(j, 1) && sampledPts.at<float>(j, 1) < y[i] + 1.05f &&
                    0.0f <= sampledPts.at<float>(j, 2) && sampledPts.at<float>(j, 2) < 3.1f ;
            if(flag) break;
        }
        ASSERT_TRUE(flag);
    }
}

TEST_F(SamplingTest, RandomSampling) {
    // Set 4 as the size, and there should have 4 points after sampling.
    randomSampling(sampledPts, ptCloud, 4);
    EXPECT_EQ(4, sampledPts.rows);
    for(int i = 0; i < 4; i++){
        ASSERT_TRUE(checkExistPoint(ptCloud, sampledPts.row(i)));
    }
}

TEST_F(SamplingTest, FarthestPointSampling) {
    Mat ans2 = Mat({1, 3}, {1.0f, 2.0f, 3.0f}),
        check,
        dPtCloud = ptCloud.clone();
    dPtCloud.push_back(ptCloud.clone());

    // Set 2 as the size, and there should be 2 diagonal points after sampling.
    farthestPointSampling(mask, ptCloud, 2);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(2, sampledPts.rows);
    check = sampledPts.row(0) + sampledPts.row(1);
    ASSERT_TRUE(comparePoints(check, 0, ans2, 0));

    // Set 4 as the size, and there should be 4 specific points after sampling.
    farthestPointSampling(mask, ptCloud, 4);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(4, sampledPts.rows);
    // These 4 points should form a plane perpendicular to the X and Y axes.
    for(int i = 0; i < 4; i++){
        check = sampledPts.row(i).clone();
        check.at<float>(0, 2) += 3;
        check.at<float>(0, 2) -= floor(check.at<float>(0, 2) / 6) * 6;
        ASSERT_TRUE(checkExistPoint(sampledPts, check));
    }

    // After doubling the point cloud, 8 points are sampled and each vertex is displayed only once.
    farthestPointSampling(mask, dPtCloud, 8);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(8, sampledPts.rows);
    bool isAppear[8];
    for(bool &item : isAppear) item = false;
    for(int i = 0; i < 8; i++){
        for(int j = 0; j < 8; j++){
            if(comparePoints(sampledPts, i, ptCloud, j)){
                ASSERT_FALSE(isAppear[j]);
                isAppear[j] = true;
                break;
            }
        }
    }

    // Test the dist_lower_limit arguments of FPS function.
    farthestPointSampling(mask, ptCloud, 4, 3);
    maskToPointCloud(ptCloud, mask, sampledPts);
    EXPECT_EQ(2, sampledPts.rows);
}

} // namespace
} // opencv_test