/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

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

template <typename T> inline T& at(int index, T *items, int size)
{
    return items[cv::borderInterpolate(index, size, cv::BORDER_WRAP)];
}

template <typename T> inline const T& at(int index, const T *items, int size)
{
    return items[cv::borderInterpolate(index, size, cv::BORDER_WRAP)];
}

template <typename T> inline T& at(int index, std::vector<T> &items)
{
    return at(index, &items[0], static_cast<int>(items.size()));
}

template <typename T> inline const T& at(int index, const std::vector<T> &items)
{
    return items[cv::borderInterpolate(index, static_cast<int>(items.size()), cv::BORDER_WRAP)];
}

#endif

