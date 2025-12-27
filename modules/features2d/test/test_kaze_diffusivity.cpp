#include "test_precomp.hpp"

namespace opencv_test { namespace {

TEST(KAZE_Diffusivity, CharbonnierSubpixelPrecision)
{
    cv::Mat img = cv::Mat::zeros(300, 300, CV_8UC1);
    cv::circle(img, cv::Point(150, 150), 80, cv::Scalar(255), -1);

    cv::Ptr<cv::KAZE> kaze = cv::KAZE::create(
        false, false, 0.001f, 4, 4, cv::KAZE::DIFF_CHARBONNIER);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;

    kaze->detectAndCompute(img, cv::noArray(), keypoints, descriptors);

    ASSERT_FALSE(keypoints.empty());

    bool hasSubpixel = false;
    for (const auto& kp : keypoints)
    {
        if (std::abs(kp.pt.x - std::round(kp.pt.x)) > 1e-6 ||
            std::abs(kp.pt.y - std::round(kp.pt.y)) > 1e-6)
        {
            hasSubpixel = true;
            break;
        }
    }

    ASSERT_TRUE(hasSubpixel);
}

}} 
