#include <opencv2/opencv.hpp>
#include <iostream>

int main()
{
    cv::Mat img = cv::Mat::zeros(300, 500, CV_8UC3); 
    cv::putText(img, "OpenCV Blur Test", cv::Point(30, 100), cv::FONT_HERSHEY_SIMPLEX, 1.5, cv::Scalar(255, 255, 255), 2);

    cv::circle(img, cv::Point(150, 200), 40, cv::Scalar(0, 255, 0), -1);
    cv::circle(img, cv::Point(250, 200), 40, cv::Scalar(0, 0, 255), -1);
    cv::circle(img, cv::Point(350, 200), 40, cv::Scalar(255, 0, 0), -1);

    cv::Mat blurred;
    cv::blur(img, blurred, cv::Size(5, 5));

    cv::imshow("Original", img);
    cv::imshow("Blurred", blurred);
    cv::waitKey(0);
    return 0;
}
