// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include <opencv2/core.hpp>
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

    // https://docs.opencv.org/master/d3/d63/classcv_1_1Mat.html
    cv::Mat src(IMG_ROWS, IMG_COLS, CV_8UC1, (void *)raw_pixels);

    // Run calc Hist
    for(int i=0; i < no_runs; i++){
        std::cout << "Running iteration # "<< i << std::endl;
        cv::norm(src);
    }

    return 0;
}
