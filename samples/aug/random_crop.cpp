#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aug.hpp>
#include <iostream>
#include <vector>


void test_pad_class(cv::Mat src, cv::Mat& dst) {
    cv::Pad pad(cv::Vec4i(100, 100, 100, 100), 255);
    pad.call(src, dst);
}


void test_randomFlip(cv::Mat src, cv::Mat& dst){
    cv::randomFlip(src, dst);
}

int main(int argv, char** argc) {
    const char* filename = argc[1];
    cv::Mat src = cv::imread(filename);
    cv::Mat dst;

//    cv::imshow("lena.png", dst);
//    cv::waitKey(0);

    return 0;
}
