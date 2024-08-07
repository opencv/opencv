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

TEST(Features2d_SIFT, MaskType)
{
    Mat gray = imread(cvtest::findDataFile("features2d/tsukuba.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty());

    cv::Rect roi(gray.cols/4, gray.rows/4, gray.cols/2, gray.rows/2);
    Mat mask = Mat::zeros(gray.size(), CV_8UC1);
    mask(roi).setTo(255);

    Mat mask_bool = Mat::zeros(gray.size(), CV_BoolC1);
    mask_bool(roi).setTo(255);

    Ptr<SIFT> siftFloat = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);

    vector<KeyPoint> keypoints_mask;
    Mat descriptorsFloat_mask;
    siftFloat->detectAndCompute(gray, mask, keypoints_mask, descriptorsFloat_mask, false);
    ASSERT_EQ(descriptorsFloat_mask.type(), CV_32F) << "type mismatch";

    vector<KeyPoint> keypoints_mask_bool;
    Mat descriptorsFloat_mask_bool;
    siftFloat->detectAndCompute(gray, mask_bool, keypoints_mask_bool, descriptorsFloat_mask_bool, false);
    ASSERT_EQ(descriptorsFloat_mask_bool.type(), CV_32F) << "type mismatch";

    Mat diff = descriptorsFloat_mask_bool != descriptorsFloat_mask;
    ASSERT_EQ(countNonZero(diff), 0) << "descriptors are not identical";
}

CV_ENUM(MaskType, CV_8U, CV_Bool);
typedef testing::TestWithParam<MaskType> SIFTMask;

static bool checkPointinRect(const cv::Point& pt, const cv::Rect& rect)
{
    return (pt.x >= rect.x) && (pt.x <= rect.x+rect.width) && (pt.y >= rect.y) && (pt.y <= rect.y+rect.height);
}

TEST_P(SIFTMask, inRect)
{
    int mask_type = GetParam();

    Mat gray = imread(cvtest::findDataFile("features2d/tsukuba.png"), IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty());

    cv::Rect roi(gray.cols/4, gray.rows/4, gray.cols/2, gray.rows/2);
    Mat mask = Mat::zeros(gray.size(), mask_type);
    mask(roi).setTo(255);

    Ptr<SIFT> siftFloat = cv::SIFT::create(0, 3, 0.04, 10, 1.6, CV_32F);

    vector<KeyPoint> keypoints_mask;
    Mat descriptorsFloat_mask;
    siftFloat->detectAndCompute(gray, mask, keypoints_mask, descriptorsFloat_mask, false);
    ASSERT_EQ(descriptorsFloat_mask.type(), CV_32F) << "type mismatch";

    for (size_t i = 0; i < keypoints_mask.size(); i++)
    {
        ASSERT_TRUE(checkPointinRect(keypoints_mask[i].pt, roi));
    }
}

INSTANTIATE_TEST_CASE_P(/**/, SIFTMask, MaskType::all());


}} // namespace
