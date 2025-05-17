// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_TRACKER_CSRT_UTILS
#define OPENCV_TRACKER_CSRT_UTILS

#include <fstream>
#include <iostream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <iterator>

#include "opencv2/core/mat.hpp"

namespace cv
{

inline int modul(int a, int b)
{
    // function calculates the module of two numbers and it takes into account also negative numbers
    return ((a % b) + b) % b;
}

inline double kernel_epan(double x)
{
    return (x <= 1) ? (2.0/3.14)*(1-x) : 0;
}

Mat circshift(Mat matrix, int dx, int dy);
Mat gaussian_shaped_labels(const float sigma, const int w, const int h);
std::vector<Mat> fourier_transform_features(const std::vector<Mat> &M);
Mat divide_complex_matrices(const Mat &A, const Mat &B);
Mat get_subwindow(const Mat &image, const Point2f center,
        const int w, const int h,Rect *valid_pixels = NULL);

float subpixel_peak(const Mat &response, const std::string &s, const Point2f &p);
double get_max(const Mat &m);
double get_min(const Mat &m);

Mat get_hann_win(Size sz);
Mat get_kaiser_win(Size sz, float alpha);
Mat get_chebyshev_win(Size sz, float attenuation);

std::vector<Mat> get_features_rgb(const Mat &patch, const Size &output_size);
std::vector<Mat> get_features_hog(const Mat &im, const int bin_size);
// std::vector<Mat> get_features_cn(const Mat &im, const Size &output_size);

Mat bgr2hsv(const Mat &img);

} //cv namespace

#endif
