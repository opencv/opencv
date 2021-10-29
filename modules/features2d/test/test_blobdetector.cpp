// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {
TEST(Features2d_BlobDetector, bug_6667)
{
    cv::Mat image = cv::Mat(cv::Size(100, 100), CV_8UC1, cv::Scalar(255, 255, 255));
    cv::circle(image, Point(50, 50), 20, cv::Scalar(0), -1);
    SimpleBlobDetector::Params params;
    params.minThreshold = 250;
    params.maxThreshold = 260;
    params.minRepeatability = 1;  // https://github.com/opencv/opencv/issues/6667
    std::vector<KeyPoint> keypoints;

    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    detector->detect(image, keypoints);
    ASSERT_NE((int) keypoints.size(), 0);
}
}} // namespace
