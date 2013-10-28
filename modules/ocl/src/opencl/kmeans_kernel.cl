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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Xiaopeng Fu, fuxiaopeng2222@163.com
//    Peng Xiao, pengxiao@outlook.com
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
// This software is provided by the copyright holders and contributors as is and
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

#ifdef L1_DIST
#  define DISTANCE(A, B) fabs((A) - (B))
#elif defined L2SQR_DIST
#  define DISTANCE(A, B) ((A) - (B)) * ((A) - (B))
#else
#  define DISTANCE(A, B) ((A) - (B)) * ((A) - (B))
#endif

inline float dist(__global const float * center, __global const float * src, int feature_cols)
{
    float res = 0;
    float4 tmp4;
    int i;
    for(i = 0; i < feature_cols / 4; i += 4, center += 4, src += 4)
    {
        tmp4 = vload4(0, center) - vload4(0, src);
#ifdef L1_DIST
        tmp4 = fabs(tmp4);
#else
        tmp4 *= tmp4;
#endif
        res += tmp4.x + tmp4.y + tmp4.z + tmp4.w;
    }

    for(; i < feature_cols; ++i, ++center, ++src)
    {
        res += DISTANCE(*src, *center);
    }
    return res;
}

// to be distinguished with distanceToCenters in kmeans_kernel.cl
__kernel void distanceToCenters(
    __global const float *src,
    __global const float *centers,
#ifdef USE_INDEX
    __global const int   *indices,
#endif
    __global int   *labels,
    __global float *dists,
    int feature_cols,
    int src_step,
    int centers_step,
    int label_step,
    int input_size,
    int K,
    int offset_src,
    int offset_centers
)
{
    int gid = get_global_id(0);
    float euDist, minval;
    int minCentroid;
    if(gid >= input_size)
    {
        return;
    }
    src += offset_src;
    centers += offset_centers;
#ifdef USE_INDEX
    src += indices[gid] * src_step;
#else
    src += gid * src_step;
#endif
    minval = dist(centers, src, feature_cols);
    minCentroid = 0;
    for(int i = 1 ; i < K; i++)
    {
        euDist = dist(centers + i * centers_step, src, feature_cols);
        if(euDist < minval)
        {
            minval = euDist;
            minCentroid = i;
        }
    }
    labels[gid * label_step] = minCentroid;
    dists[gid] = minval;
}
