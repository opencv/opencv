// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>

#include <cstdint>
#include <array>
#include <iostream>
#include "raw_pixels.hpp"

#define IMG_ROWS 100
#define IMG_COLS 100

static_assert(IMG_ROWS * IMG_COLS <= RAW_PIXELS_SIZE, "Incompatible size");

int main(void)
{
    // Number of experiment runs
    int no_runs = 2;

    // https://docs.opencv.org/4.x/d3/d63/classcv_1_1Mat.html
    cv::Mat src_new(IMG_ROWS, IMG_COLS, CV_8UC1, (void *)raw_pixels);

    // Set parameters
    int imgCount = 1;
    const int channels[] = {0};
    cv::Mat mask = cv::Mat();
    cv::Mat hist;
    int dims = 1;
    const int hist_sizes[] = {256};
    float Range[] = {0,256};
    const float *ranges[] = {Range};

    // Run calc Hist
    for(int i=0; i < no_runs; i++){
        std::cout << "Running iteration # "<< i << std::endl;
        cv::calcHist(&src_new, imgCount, channels, mask, hist, dims, hist_sizes, ranges);
    }

    return 0;
}
