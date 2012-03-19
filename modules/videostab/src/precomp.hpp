#ifndef __OPENCV_PRECOMP_HPP__
#define __OPENCV_PRECOMP_HPP__

#ifdef HAVE_CVCONFIG_H
#include "cvconfig.h"
#endif

#include <stdexcept>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/video.hpp"
#include "opencv2/features2d/features2d.hpp"

inline float sqr(float x) { return x * x; }

inline float intensity(const cv::Point3_<uchar> &bgr)
{
    return 0.3f*bgr.x + 0.59f*bgr.y + 0.11f*bgr.z;
}

template <typename T> inline T& at(int index, std::vector<T> &items)
{
    return items[cv::borderInterpolate(index, items.size(), cv::BORDER_WRAP)];
}

template <typename T> inline const T& at(int index, const std::vector<T> &items)
{
    return items[cv::borderInterpolate(index, items.size(), cv::BORDER_WRAP)];
}

#endif

