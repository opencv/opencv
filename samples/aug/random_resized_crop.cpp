#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/aug.hpp>
#include <iostream>
#include <vector>


int main(int argv, char** argc) {
    const char* filename = argc[1];
    // lena.png is of size (512, 512)
    cv::Mat src = cv::imread(filename);
    cv::Mat dst;

    cv::randomResizedCrop(src, dst, cv::Size(300, 300));
    cv::imshow("lena.png", dst);
    cv::waitKey(0);

    return 0;
}
