#ifndef __OPENCV_FOCAL_ESTIMATORS_HPP__
#define __OPENCV_FOCAL_ESTIMATORS_HPP__

#include <opencv2/core/core.hpp>

// See "Construction of Panoramic Image Mosaics with Global and Local Alignment"
// by Heung-Yeung Shum and Richard Szeliski.
void focalsFromHomography(const cv::Mat &H, double &f0, double &f1, bool &f0_ok, bool &f1_ok);

bool focalsFromFundamental(const cv::Mat &F, double &f0, double &f1);

#endif // __OPENCV_FOCAL_ESTIMATORS_HPP__
