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
// any express or bpied warranties, including, but not limited to, the bpied
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

#ifndef __OPENCV_GPU_DEVICE_LBP_HPP_
#define __OPENCV_GPU_DEVICE_LBP_HPP_

#include "internal_shared.hpp"
// #include "opencv2/gpu/device/border_interpolate.hpp"
// #include "opencv2/gpu/device/vec_traits.hpp"
// #include "opencv2/gpu/device/vec_math.hpp"
// #include "opencv2/gpu/device/saturate_cast.hpp"
// #include "opencv2/gpu/device/filters.hpp"

// #define CALC_SUM_(p0, p1, p2, p3, offset) \
//     ((p0)[offset] - (p1)[offset] - (p2)[offset] + (p3)[offset])

// __device__ __forceinline__ int sum(p0, p1, p2, p3, offset)
// {

// }

namespace cv { namespace gpu { namespace device {

    struct Stage
    {
        int    first;
        int    ntrees;
        float  threshold;
        __device__ __forceinline__ Stage(int f = 0, int n = 0, float t = 0.f) : first(f), ntrees(n), threshold(t) {}
        __device__ __forceinline__ Stage(const Stage& other) : first(other.first), ntrees(other.ntrees), threshold(other.threshold) {}
    };

    struct ClNode
    {
        int   featureIdx;
        int   left;
        int   right;
        __device__ __forceinline__  ClNode(int f = 0, int l = 0, int r = 0) : featureIdx(f), left(l), right(r) {}
        __device__ __forceinline__  ClNode(const ClNode& other) : featureIdx(other.featureIdx), left(other.left), right(other.right) {}
    };

    struct Feature
    {
        __device__ __forceinline__ Feature(const Feature& other) {(void)other;}
        __device__ __forceinline__ Feature() {}

        //feature as uchar x, y - left top, z,w - right bottom
        __device__ __forceinline__ uchar operator() (unsigned int y, unsigned int x, uchar4 feature, const DevMem2Di integral) const
        {
            int x_off = 2 * feature.z;
            int y_off = 2 * feature.w;

            // load feature key points
            int anchors[16];
            anchors[0]  = integral(y + feature.y, x + feature.x);
            anchors[1]  = integral(y + feature.y, x + feature.z);
            anchors[2]  = integral(y + feature.y, x + x_off + feature.x);
            anchors[3]  = integral(y + feature.y, x + x_off + feature.z);

            anchors[4]  = integral(y + feature.w, x + feature.x);
            anchors[5]  = integral(y + feature.w, x + feature.z);
            anchors[6]  = integral(y + feature.w, x + x_off + feature.x);
            anchors[7]  = integral(y + feature.w, x + x_off + feature.z);

            anchors[8]  = integral(y + y_off + feature.y, x + feature.x);
            anchors[9]  = integral(y + y_off + feature.y, x + feature.z);
            anchors[10] = integral(y + y_off + feature.y, x + x_off + feature.x);
            anchors[11] = integral(y + y_off + feature.y, x + x_off + feature.z);

            anchors[12] = integral(y + y_off + feature.w, x + feature.x);
            anchors[13] = integral(y + y_off + feature.w, x + feature.z);
            anchors[14] = integral(y + y_off + feature.w, x + x_off + feature.x);
            anchors[15] = integral(y + y_off + feature.w, x + x_off + feature.z);

            // calculate feature
            int sum = anchors[5] - anchors[6] - anchors[9] + anchors[10];

            uchar response = (( (anchors[ 0] - anchors[ 1] - anchors[ 4] + anchors[ 5]) >= sum )? 128 : 0)
                            |(( (anchors[ 1] - anchors[ 2] - anchors[ 5] + anchors[ 6]) >= sum )? 64  : 0)
                            |(( (anchors[ 2] - anchors[ 3] - anchors[ 6] + anchors[ 7]) >= sum )? 32  : 0)
                            |(( (anchors[ 6] - anchors[ 7] - anchors[10] + anchors[11]) >= sum )? 16  : 0)
                            |(( (anchors[10] - anchors[11] - anchors[14] + anchors[15]) >= sum )? 8   : 0)
                            |(( (anchors[ 9] - anchors[10] - anchors[13] + anchors[14]) >= sum )? 4   : 0)
                            |(( (anchors[ 8] - anchors[ 9] - anchors[12] + anchors[13]) >= sum )? 2   : 0)
                            |(( (anchors[ 4] - anchors[ 5] - anchors[ 8] + anchors[ 9]) >= sum )? 1   : 0);

            return response;
        }

    };
} } }// namespaces

#endif