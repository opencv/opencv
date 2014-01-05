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

inline float distance_(__global const float * center, __global const float * src, int feature_length)
{
    float res = 0;
    float4 v0, v1, v2;
    int i = 0;

#ifdef L1_DIST
    float4 sum = (float4)(0.0f);
#endif

    for ( ; i <= feature_length - 4; i += 4)
    {
        v0 = vload4(0, center + i);
        v1 = vload4(0, src + i);
        v2 = v1 - v0;
#ifdef L1_DIST
        v0 = fabs(v2);
        sum += v0;
#else
        res += dot(v2, v2);
#endif
    }

#ifdef L1_DIST
    res = sum.x + sum.y + sum.z + sum.w;
#endif

    for ( ; i < feature_length; ++i)
    {
        float t0 = src[i];
        float t1 = center[i];
#ifdef L1_DIST
        res += fabs(t0 - t1);
#else
        float t2 = t0 - t1;
        res += t2 * t2;
#endif
    }

    return res;
}

__kernel void distanceToCenters(__global const float * src, __global const float * centers,
                                __global float * dists, int feature_length,
                                int src_step, int centers_step,
                                int features_count, int centers_count,
                                int src_offset, int centers_offset)
{
    int gid = get_global_id(0);

    if (gid < (features_count * centers_count))
    {
        int feature_index = gid / centers_count;
        int center_index = gid % centers_count;

        int center_idx = mad24(center_index, centers_step, centers_offset);
        int src_idx = mad24(feature_index, src_step, src_offset);

        dists[gid] = distance_(centers + center_idx, src + src_idx, feature_length);
    }
}
