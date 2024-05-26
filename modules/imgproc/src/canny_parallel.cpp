#include <opencv2/opencv.hpp>
#include <omp.h>

void
canny_edge_detector(const cv::Mat &input, cv::Mat &output, double low_threshcv::Mat gray, void *blurred, void *edges) {
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurred, cv::Size(5, 5), 1.5);

#pragma omp parallel for
    for (int i = 0; i < blurred.rows; i++) {
        cv::Canny(blurred.row(i), edges.row(i), low_thresh, high_thresh);
    }

    output = edges.clone();
}
