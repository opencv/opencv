// Simple example that uses the experimental C++20 module interface for OpenCV core.
// This file is only built when the toolchain supports C++20 modules.

import opencv.core;

int main()
{
    cv::Mat m(2, 2, CV_8U);
    m.setTo(cv::Scalar(0));
    return 0;
}
