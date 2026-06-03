// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021, Wanli Zhong <zhongwl2018@mail.sustech.edu.cn>

#include "test_precomp.hpp"

namespace opencv_test { namespace {

class SamplingTest : public ::testing::Test {

public:
    int ori_pts_size = 0;

    // Different point clouds to test InputArray compatibility.
    vector<Point3f> pts_vec_Pt3f;
    Mat pts_mat_32F_Nx3;
    Mat pts_mat_64F_3xN;
    Mat pts_mat_32FC3_Nx1;

    // Different masks to test OutputArray compatibility.
    vector<char> mask_vec_char;
    vector<int> mask_vec_int;
    Mat mask_mat_Nx1;
    Mat mask_mat_1xN;

    // Combination of InputArray and OutputArray as information to mention where the test fail.
    string header = "\n===================================================================\n";
    string combination1 = "OutputArray: vector<char>\nInputArray: vector<point3f>\n";
    string combination2 = "OutputArray: Nx1 Mat\nInputArray: Nx3 float Mat\n";
    string combination3 = "OutputArray: 1xN Mat\nInputArray: 3xN double Mat\n";
    string combination4 = "OutputArray: vector<int>\nInputArray: Nx1 3 channels float Mat\n";


    // Initialize point clouds with different data type.
    void dataInitialization(vector<float> &pts_data)
    {
        ori_pts_size = (int) pts_data.size() / 3;

        // point cloud use Nx3 mat as data structure and float as value type.
        pts_mat_32F_Nx3 = Mat(ori_pts_size, 3, CV_32F, pts_data.data());

        // point cloud use vector<Point3f> as data structure.
        pts_vec_Pt3f.clear();
        for (int i = 0; i < ori_pts_size; i++)
        {
            int i3 = i * 3;
            pts_vec_Pt3f.emplace_back(
                    Point3f(pts_data[i3], pts_data[i3 + 1], pts_data[i3 + 2]));
        }

        // point cloud use 3xN mat as data structure and double as value type.
        pts_mat_64F_3xN = pts_mat_32F_Nx3.t();
        pts_mat_64F_3xN.convertTo(pts_mat_64F_3xN, CV_64F);

        // point cloud use Nx1 mat with 3 channels as data structure and float as value type.
        pts_mat_32F_Nx3.convertTo(pts_mat_32FC3_Nx1, CV_32FC3);
    }
};

// Get 1xN mat of mask from OutputArray.
void getMatFromMask(OutputArray &mask, Mat &mat)
{
    int rows = mask.rows(), cols = mask.cols(), channels = mask.channels();

    if (channels == 1 && cols == 1 && rows != 1)
        mat = mask.getMat().t();
    else
        mat = mask.getMat().reshape(1, 1);

    // Use int to store data.
    if (mat.type() != CV_32S)
        mat.convertTo(mat, CV_32S);
}

// Compare 2 rows in different point clouds.
bool compareRows(const Mat &m, int rm, const Mat &n, int rn)
{
    Mat diff = m.row(rm) != n.row(rn);
    return countNonZero(diff) == 0;
}

TEST_F(SamplingTest, VoxelGridFilterSampling)
{
    vector<float> pts_info = {0.0f, 0.0f, 0.0f,
                              -3.0f, -3.0f, -3.0f, 3.0f, -3.0f, -3.0f, 3.0f, 3.0f, -3.0f, -3.0f, 3.0f, -3.0f,
                              -3.0f, -3.0f, 3.0f, 3.0f, -3.0f, 3.0f, 3.0f, 3.0f, 3.0f, -3.0f, 3.0f, 3.0f,
                              -0.9f, -0.9f, -0.9f, 0.9f, -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, -0.9f, 0.9f, -0.9f,
                              -0.9f, -0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f, 0.9f, 0.9f, -0.9f, 0.9f, 0.9f,};
    dataInitialization(pts_info);

    auto testVoxelGridFilterSampling = [this](OutputArray mask, InputArray pts, float *side,
                                              int sampled_pts_size, const Mat &ans, const string &info) {
        // Check the number of point cloud after sampling.
        int res = voxelGridSampling(mask, pts, side[0], side[1], side[2]);
        EXPECT_EQ(sampled_pts_size, res)
        << header << info << "The side length is " << side[0]
        << "\nThe return value of voxelGridSampling() is not equal to " << sampled_pts_size
        << endl;
        EXPECT_EQ(sampled_pts_size, countNonZero(mask))
        << header << info << "The side length is " << side[0]
        << "\nThe number of selected points of mask is not equal to " << sampled_pts_size
        << endl;

        // The mask after sampling should be equal to the answer.
        Mat _mask;
        getMatFromMask(mask, _mask);
        ASSERT_TRUE(compareRows(_mask, 0, ans, 0))
        << header << info << "The side length is " << side[0]
        << "\nThe mask should be " << ans << endl;
    };

    // Set 6.1 as the side1 length, only the center point left.
    Mat ans1({1, (int) (pts_info.size() / 3)}, {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
    float side1[3] = {6.1f, 6.1f, 6.1f};
    testVoxelGridFilterSampling(mask_vec_char, pts_vec_Pt3f, side1, 1, ans1, combination1);
    testVoxelGridFilterSampling(mask_mat_Nx1, pts_mat_32F_Nx3, side1, 1, ans1, combination2);
    testVoxelGridFilterSampling(mask_mat_1xN, pts_mat_64F_3xN, side1, 1, ans1, combination3);
    testVoxelGridFilterSampling(mask_vec_int, pts_mat_32FC3_Nx1, side1, 1, ans1, combination4);

    // Set 2 as the side1 length, only the center point and the vertexes of big cube left.
    Mat ans2({1, (int) (pts_info.size() / 3)}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0});
    float side2[3] = {2.0f, 2.0f, 2.0f};
    testVoxelGridFilterSampling(mask_vec_char, pts_vec_Pt3f, side2, 9, ans2, combination1);
    testVoxelGridFilterSampling(mask_mat_Nx1, pts_mat_32F_Nx3, side2, 9, ans2, combination2);
    testVoxelGridFilterSampling(mask_mat_1xN, pts_mat_64F_3xN, side2, 9, ans2, combination3);
    testVoxelGridFilterSampling(mask_vec_int, pts_mat_32FC3_Nx1, side2, 9, ans2, combination4);

    // Set 1.1 as the side1 length, all points should be left.
    Mat ans3({1, (int) (pts_info.size() / 3)}, {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1});
    float side3[3] = {1.1f, 1.1f, 1.1f};
    testVoxelGridFilterSampling(mask_vec_char, pts_vec_Pt3f, side3, 17, ans3, combination1);
    testVoxelGridFilterSampling(mask_mat_Nx1, pts_mat_32F_Nx3, side3, 17, ans3, combination2);
    testVoxelGridFilterSampling(mask_mat_1xN, pts_mat_64F_3xN, side3, 17, ans3, combination3);
    testVoxelGridFilterSampling(mask_vec_int, pts_mat_32FC3_Nx1, side3, 17, ans3, combination4);
}

TEST_F(SamplingTest, RandomSampling)
{
    vector<float> pts_info = {0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0,
                              0, 0, 3, 1, 0, 3, 1, 2, 3, 0, 2, 3};
    dataInitialization(pts_info);

    auto testRandomSampling = [this](InputArray pts, int sampled_pts_size, const string &info) {
        // Check the number of point cloud after sampling.
        Mat sampled_pts;
        randomSampling(sampled_pts, pts, sampled_pts_size);
        EXPECT_EQ(sampled_pts_size, sampled_pts.rows)
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe number of sampled points is not equal to " << sampled_pts_size << endl;

        // Convert InputArray to Mat of original point cloud.
        int rows = pts.rows(), cols = pts.cols(), channels = pts.channels();
        int total = rows * cols * channels;
        Mat ori_pts;
        if (channels == 1 && rows == 3 && cols != 3)
            ori_pts = pts.getMat().t();
        else
            ori_pts = pts.getMat().reshape(1, (int) (total / 3));
        if (ori_pts.type() != CV_32F)
            ori_pts.convertTo(ori_pts, CV_32F);

        // Each point should be in the original point cloud.
        for (int i = 0; i < sampled_pts_size; i++)
        {
            // Check whether a point exists in the point cloud.
            bool flag = false;
            for (int j = 0; j < ori_pts.rows; j++)
                flag |= compareRows(sampled_pts, i, ori_pts, j);
            ASSERT_TRUE(flag)
            << header << info << "The sampled points size is " << sampled_pts_size
            << "\nThe sampled point " << sampled_pts.row(i)
            << " is not in original point cloud" << endl;
        }
    };

    testRandomSampling(pts_vec_Pt3f, 3, combination1);
    testRandomSampling(pts_mat_32F_Nx3, 4, combination2);
    testRandomSampling(pts_mat_64F_3xN, 5, combination3);
    testRandomSampling(pts_mat_32FC3_Nx1, 6, combination4);
}

TEST_F(SamplingTest, FarthestPointSampling)
{
    vector<float> pts_info = {0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0,
                              0, 0, 3, 1, 0, 3, 1, 2, 3, 0, 2, 3};
    dataInitialization(pts_info);

    auto testFarthestPointSampling = [this](OutputArray mask, InputArray pts, int sampled_pts_size,
                                            const vector<Mat> &ans, const string &info) {
        // Check the number of point cloud after sampling.
        int res = farthestPointSampling(mask, pts, sampled_pts_size);
        EXPECT_EQ(sampled_pts_size, res)
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe return value of farthestPointSampling() is not equal to "
        << sampled_pts_size << endl;
        EXPECT_EQ(sampled_pts_size, countNonZero(mask))
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe number of selected points of mask is not equal to " << sampled_pts_size
        << endl;

        // The mask after sampling should be one of the answers.
        Mat _mask;
        getMatFromMask(mask, _mask);
        bool flag = false;
        for (const Mat &a: ans)
            flag |= compareRows(a, 0, _mask, 0);
        string ans_info;
        ASSERT_TRUE(flag)
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe mask should be one of the answer list" << endl;
    };

    // Set 2 as the size, and there should be 2 diagonal points after sampling.
    vector<Mat> ans_list1;
    ans_list1.emplace_back(Mat({1, ori_pts_size}, {1, 0, 0, 0, 0, 0, 1, 0}));
    ans_list1.emplace_back(Mat({1, ori_pts_size}, {0, 1, 0, 0, 0, 0, 0, 1}));
    ans_list1.emplace_back(Mat({1, ori_pts_size}, {0, 0, 1, 0, 1, 0, 0, 0}));
    ans_list1.emplace_back(Mat({1, ori_pts_size}, {0, 0, 0, 1, 0, 1, 0, 0}));

    testFarthestPointSampling(mask_vec_char, pts_vec_Pt3f, 2, ans_list1, combination1);
    testFarthestPointSampling(mask_mat_Nx1, pts_mat_32F_Nx3, 2, ans_list1, combination2);
    testFarthestPointSampling(mask_mat_1xN, pts_mat_64F_3xN, 2, ans_list1, combination3);
    testFarthestPointSampling(mask_vec_int, pts_mat_32FC3_Nx1, 2, ans_list1, combination4);

    // Set 4 as the size, and there should be 4 specific points form a plane perpendicular to the X and Y axes after sampling.
    vector<Mat> ans_list2;
    ans_list2.emplace_back(Mat({1, ori_pts_size}, {1, 0, 1, 0, 1, 0, 1, 0}));
    ans_list2.emplace_back(Mat({1, ori_pts_size}, {0, 1, 0, 1, 0, 1, 0, 1}));

    testFarthestPointSampling(mask_vec_char, pts_vec_Pt3f, 4, ans_list2, combination1);
    testFarthestPointSampling(mask_mat_Nx1, pts_mat_32F_Nx3, 4, ans_list2, combination2);
    testFarthestPointSampling(mask_mat_1xN, pts_mat_64F_3xN, 4, ans_list2, combination3);
    testFarthestPointSampling(mask_vec_int, pts_mat_32FC3_Nx1, 4, ans_list2, combination4);
}

TEST_F(SamplingTest, FarthestPointSamplingDoublePtCloud)
{
    // After doubling the point cloud, 8 points are sampled and each vertex is displayed only once.
    vector<float> pts_info = {0, 0, 0, 1, 0, 0, 1, 2, 0, 0, 2, 0,
                              0, 0, 3, 1, 0, 3, 1, 2, 3, 0, 2, 3};
    // Double the point cloud.
    pts_info.insert(pts_info.end(), pts_info.begin(), pts_info.end());
    dataInitialization(pts_info);

    auto testFarthestPointSamplingDoublePtCloud = [this](OutputArray mask, InputArray pts, int sampled_pts_size,
                                                         const string &info) {
        // Check the number of point cloud after sampling.
        int res = farthestPointSampling(mask, pts, sampled_pts_size);
        EXPECT_EQ(sampled_pts_size, res)
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe return value of farthestPointSampling() is not equal to "
        << sampled_pts_size << endl;
        EXPECT_EQ(sampled_pts_size, countNonZero(mask))
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe number of selected points of mask is not equal to " << sampled_pts_size
        << endl;


        // One and only one of mask[index] and mask[index + size/2] is true for first eight index.
        Mat _mask;
        getMatFromMask(mask, _mask);
        auto *mask_ptr = (int *) _mask.data;
        int offset = ori_pts_size / 2;
        bool flag = true;
        for (int i = 0; i < offset; i++)
            flag &= (mask_ptr[i] == 0 && mask_ptr[i + offset] != 0) || (mask_ptr[i] != 0 && mask_ptr[i + offset] == 0);
        ASSERT_TRUE(flag)
        << header << info << "The sampled points size is " << sampled_pts_size
        << "\nThe mask should be one of the answer list" << endl;

    };

    testFarthestPointSamplingDoublePtCloud(mask_vec_char, pts_vec_Pt3f, 8, combination1);
    testFarthestPointSamplingDoublePtCloud(mask_mat_Nx1, pts_mat_32F_Nx3, 8, combination2);
    testFarthestPointSamplingDoublePtCloud(mask_mat_1xN, pts_mat_64F_3xN, 8, combination3);
    testFarthestPointSamplingDoublePtCloud(mask_vec_int, pts_mat_32FC3_Nx1, 8, combination4);
}

} // namespace
} // opencv_test
