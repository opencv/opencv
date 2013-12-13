//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Jia Haipeng, jiahaipeng95@gmail.com
//    Xu Pang, pangxu010@163.com
//    Wenju He, wenju@multicorewareinc.com
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
//

__kernel void calculate_histogram(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                                  __global uchar * hist, int total)
{
    int lid = get_local_id(0);
    int id = get_global_id(0);
    int gid = get_group_id(0);

    __local int localhist[BINS];

    for (int i = lid; i < BINS; i += WGS)
        localhist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int grain = HISTS_COUNT * WGS; id < total; id += grain)
    {
        int src_index = mad24(id / src_cols, src_step, src_offset + id % src_cols);
        atomic_inc(localhist + (int)src[src_index]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = lid; i < BINS; i += WGS)
        *(__global int *)(hist + mad24(gid, BINS * (int)sizeof(int), i * (int)sizeof(int))) = localhist[i];
}

__kernel void merge_histogram(__global const int * ghist, __global int * hist)
{
    int lid = get_local_id(0);

    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        hist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int i = 0; i < HISTS_COUNT; ++i)
    {
        #pragma unroll
        for (int j = lid; j < BINS; j += WGS)
            hist[j] += ghist[mad24(i, BINS, j)];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}

__kernel void calcLUT(__global uchar * dst, __constant int * hist, int total)
{
    int lid = get_local_id(0);
    __local int sumhist[BINS];
    __local float scale;

    sumhist[lid] = hist[lid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0)
    {
        int sum = 0, i = 0;
        while (!sumhist[i])
            ++i;

        if (total == sumhist[i])
        {
            scale = 1;
            for (int j = 0; j < BINS; ++j)
                sumhist[i] = i;
        }
        else
        {
            scale = 255.f / (total - sumhist[i]);

            for (sumhist[i++] = 0; i < BINS; i++)
            {
                sum += sumhist[i];
                sumhist[i] = sum;
            }
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        dst[i]= convert_uchar_sat_rte(convert_float(sumhist[i]) * scale);
}
