#ifndef _OPENCV_FOCAL_ESTIMATORS_HPP_
#define _OPENCV_FOCAL_ESTIMATORS_HPP_

#include <opencv2/core/core.hpp>

// See "Construction of Panoramic Image Mosaics with Global and Local Alignment"
// by Heung-Yeung Shum and Richard Szeliski.
void focalsFromHomography(const cv::Mat &H, double &f0, double &f1, bool &f0_ok, bool &f1_ok);

bool focalsFromFundamental(const cv::Mat &F, double &f0, double &f1);

#endif // _OPENCV_FOCAL_ESTIMATORS_HPP_
