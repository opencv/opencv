#include "test_precomp.hpp"

TEST(Imgproc_OCL_HoughLines, MinMaxThetaConsistency)
{
    cv::Mat img = cv::Mat::zeros(200,200,CV_8UC1);
    cv::line(img, {20,180}, {180,20}, 255, 2);

    double minT = CV_PI/4;
    double maxT = CV_PI/3;

    std::vector<cv::Vec2f> cpu;
    cv::HoughLines(img, cpu, 1, CV_PI/180, 50, 0, 0, minT, maxT);

    cv::UMat u;
    img.copyTo(u);

    std::vector<cv::Vec2f> ocl;
    cv::HoughLines(u, ocl, 1, CV_PI/180, 50, 0, 0, minT, maxT);

    ASSERT_FALSE(cpu.empty());
    ASSERT_FALSE(ocl.empty());

    ASSERT_NEAR(cpu[0][0], ocl[0][0], 1.0f);
    ASSERT_NEAR(cpu[0][1], ocl[0][1], CV_PI/180);
}
