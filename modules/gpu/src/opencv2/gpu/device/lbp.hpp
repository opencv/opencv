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
        __device__ __forceinline__ char operator() ()//(volatile int* ptr, int offset)
        {
            return char(0);
        }

    };
} } }// namespaces

#endif