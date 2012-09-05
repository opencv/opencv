/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#ifndef __OPENCV_SIMPLEFLOW_H__
#define __OPENCV_SIMPLEFLOW_H__

#include <vector>

using namespace std;

#define MASK_TRUE_VALUE 255
#define UNKNOWN_FLOW_THRESH 1e9

namespace cv {

inline static float dist(const Vec3b& p1, const Vec3b& p2) {
  return (float)((p1[0] - p2[0]) * (p1[0] - p2[0]) +
         (p1[1] - p2[1]) * (p1[1] - p2[1]) +
         (p1[2] - p2[2]) * (p1[2] - p2[2]));
}

inline static float dist(const Vec2f& p1, const Vec2f& p2) {
  return (p1[0] - p2[0]) * (p1[0] - p2[0]) +
         (p1[1] - p2[1]) * (p1[1] - p2[1]);
}

inline static float dist(const Point2f& p1, const Point2f& p2) {
  return (p1.x - p2.x) * (p1.x - p2.x) +
         (p1.y - p2.y) * (p1.y - p2.y);
}

inline static float dist(float x1, float y1, float x2, float y2) {
  return (x1 - x2) * (x1 - x2) +
         (y1 - y2) * (y1 - y2);
}

inline static int dist(int x1, int y1, int x2, int y2) {
  return (x1 - x2) * (x1 - x2) +
         (y1 - y2) * (y1 - y2);
}

template<class T>
inline static T min(T t1, T t2, T t3) {
  return (t1 <= t2 && t1 <= t3) ? t1 : min(t2, t3);
}

}

#endif
