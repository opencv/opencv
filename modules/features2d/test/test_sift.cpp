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
    Mat descriptorsAll, descriptorsSubset;
    vector<KeyPoint> subsetOfKeypoints;
    map<int, KeyPoint> indexes_to_kps;
    for(int i = 0; i < keypoints.size(); ++i) {
        int octave = keypoints[i].octave & 255;
        octave = octave < 128 ? octave : (-128 | octave);
        if(octave > -1) {
            indexes_to_kps[i] = keypoints[i];
            subsetOfKeypoints.push_back(keypoints[i]);
        }
    }
    sift->compute(image, keypoints, descriptorsAll);
    sift->compute(image, subsetOfKeypoints, descriptorsSubset);
    // we should be able to provide all keypoints or a subset of keypoints and get the same descriptors
    for (pair<const int,KeyPoint>& x: indexes_to_kps) {
        std::cout << descriptorsAll.row(x.first) << std::endl;
        std::cout << descriptorsSubset.row(x.first) << std::endl;
        for(int col = 0; col < descriptorsAll.cols; ++col) {
            ASSERT_EQ(descriptorsAll.at<float>(x.first, col), descriptorsSubset.at<float>(x.first, col));
        }
    }

}


}} // namespace
