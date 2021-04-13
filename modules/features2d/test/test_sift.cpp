// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Features2d_SIFT, descriptor_type)
{
    Mat image = imread(cvtest::findDataFile("features2d/tsukuba.png"));
    ASSERT_FALSE(image.empty());

    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    vector<KeyPoint> keypoints;
    Mat descriptorsFloat, descriptorsUchar;
    Ptr<SIFT> siftFloat = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);
    siftFloat->detectAndCompute(gray, Mat(), keypoints, descriptorsFloat, false);
    ASSERT_EQ(descriptorsFloat.type(), CV_32F) << "type mismatch";

    Ptr<SIFT> siftUchar = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_8U);
    siftUchar->detectAndCompute(gray, Mat(), keypoints, descriptorsUchar, false);
    ASSERT_EQ(descriptorsUchar.type(), CV_8U) << "type mismatch";

    Mat descriptorsFloat2;
    descriptorsUchar.assignTo(descriptorsFloat2, CV_32F);
    Mat diff = descriptorsFloat != descriptorsFloat2;
    ASSERT_EQ(countNonZero(diff), 0) << "descriptors are not identical";
}

TEST(Features2d_SIFT, 177_octave_independence)
{
    Ptr<SIFT> sift = cv::SIFT::create(10, 5, 0.01, 10, 1.1, CV_32F);

    Mat image = imread(string(cvtest::TS::ptr()->get_data_path()) + "shared/lena.png");
    ASSERT_FALSE(image.empty());

    vector<KeyPoint> keypoints;
    sift->detect(image, keypoints);
    Mat descriptorsAll, descriptorsOne;
    vector<KeyPoint> oneKeypoint(keypoints.begin(), keypoints.begin()+1);
    sift->compute(image, keypoints, descriptorsAll);
    sift->compute(image, oneKeypoint, descriptorsOne);
    // I should be able to provide all keypoints or one keypoint and get the same descriptor value
    ASSERT_EQ(descriptorsAll.at<float>(0, 1), descriptorsOne.at<float>(0, 1));
}


}} // namespace
