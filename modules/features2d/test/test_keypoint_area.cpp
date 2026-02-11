// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(Core_KeyPoint, AreaHelper)
{
    // Test 1: Area not set (default 0)
    cv::KeyPoint kp(10, 10, 20.0f); // size = 20.0f
    EXPECT_EQ(kp.area, 0.0f);

    float expectedAreaFromSize = (float)(CV_PI * 0.25 * 20.0f * 20.0f);
    EXPECT_NEAR(cv::keypointArea(kp), expectedAreaFromSize, 1e-5);

    // Test 2: Area set explicitly
    kp.area = 100.0f;
    EXPECT_EQ(cv::keypointArea(kp), 100.0f);
}

TEST(Features2d_BlobDetector, AreaPopulation)
{
    // Create a synthetic image with a known blob
    // Circle with radius 20 -> diameter 40 -> Area = pi * 20^2 = ~1256.6
    cv::Mat image = cv::Mat::zeros(cv::Size(200, 200), CV_8UC1);
    cv::circle(image, cv::Point(100, 100), 20, cv::Scalar(255), -1);

    SimpleBlobDetector::Params params;
    params.minThreshold = 100;
    params.maxThreshold = 200;
    params.filterByArea = true;
    params.minArea = 1000;
    params.maxArea = 1500;
    // Disable other filters for simplicity to ensure we detect the blob
    params.filterByCircularity = false;
    params.filterByInertia = false;
    params.filterByConvexity = false;
    params.filterByColor = true;
    params.blobColor = 255;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    std::vector<KeyPoint> keypoints;
    detector->detect(image, keypoints);

    ASSERT_EQ(keypoints.size(), 1u);
    KeyPoint kp = keypoints[0];

    // Check area
    // The area computed by moments (m00) for a drawn circle might be slightly different from mathematical area due to discretization
    double exactArea = CV_PI * 20 * 20; // 1256.6
    // Allow some tolerance for rasterization (e.g. +/- 5%)
    EXPECT_NEAR(kp.area, exactArea, exactArea * 0.05);
    EXPECT_GT(kp.area, 0.0f);

    // Check keypointArea helper
    EXPECT_EQ(cv::keypointArea(kp), kp.area);
}

}} // namespace
