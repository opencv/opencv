#ifndef __OPENCV_AUTOCALIB_HPP__
#define __OPENCV_AUTOCALIB_HPP__

#include <vector>
#include <opencv2/core/core.hpp>
#include "matchers.hpp"

// See "Construction of Panoramic Image Mosaics with Global and Local Alignment"
// by Heung-Yeung Shum and Richard Szeliski.
void focalsFromHomography(const cv::Mat &H, double &f0, double &f1, bool &f0_ok, bool &f1_ok);

double estimateFocal(const std::vector<ImageFeatures> &features, const std::vector<MatchesInfo> &pairwise_matches);

#endif // __OPENCV_AUTOCALIB_HPP__
