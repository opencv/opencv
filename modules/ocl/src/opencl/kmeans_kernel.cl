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
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other GpuMaterials provided with the distribution.
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

__kernel void distanceToCenters(
    int label_step, int K,
    __global float *src,
    __global int *labels, int dims, int rows,
    __global float *centers,
    __global float *dists)
{
    int gid = get_global_id(1);

    float dist, euDist, min;
    int minCentroid;

    if(gid >= rows)
        return;

    for(int i = 0 ; i < K; i++)
    {
        euDist = 0;
        for(int j = 0; j < dims; j++)
        {
            dist = (src[j + gid * dims]
                    - centers[j + i * dims]);
            euDist += dist * dist;
        }

        if(i == 0)
        {
            min = euDist;
            minCentroid = 0;
        }
        else if(euDist < min)
        {
            min = euDist;
            minCentroid = i;
        }
    }
    dists[gid] = min;
    labels[label_step * gid] = minCentroid;
}
