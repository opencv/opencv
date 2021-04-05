#include "test_precomp.hpp"

namespace opencv_test {
    TEST(Features2d_BlobDetector, bug_6667) {
        cv::Mat image = cv::Mat(cv::Size(100, 100), CV_8UC1, cv::Scalar(255,255,255));
        cv::circle(image,Point(50,50), 20, cv::Scalar(0), -1);
        SimpleBlobDetector::Params params;
        params.minThreshold = 250;
        params.maxThreshold = 260;
        std::vector<KeyPoint> keypoints;

        Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
        detector->detect(image, keypoints);
        ASSERT_NE((int) keypoints.size(), 0);
    }
}
