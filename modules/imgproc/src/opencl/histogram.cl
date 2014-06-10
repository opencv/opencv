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

#ifndef kercn
#define kercn 1
#endif

#ifndef T
#define T uchar
#endif

#define noconvert

__kernel void calculate_histogram(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                                  __global uchar * histptr, int total)
{
    int lid = get_local_id(0);
    int id = get_global_id(0) * kercn;
    int gid = get_group_id(0);

    __local int localhist[BINS];

    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        localhist[i] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    int src_index;

    for (int grain = HISTS_COUNT * WGS * kercn; id < total; id += grain)
    {
#ifdef HAVE_SRC_CONT
        src_index = id;
#else
        src_index = mad24(id / src_cols, src_step, src_offset + id % src_cols);
#endif

#if kercn == 1
        atomic_inc(localhist + convert_int(src[src_index]));
#elif kercn == 4
        int value = *(__global const int *)(src + src_index);
        atomic_inc(localhist + (value & 0xff));
        atomic_inc(localhist + ((value >> 8) & 0xff));
        atomic_inc(localhist + ((value >> 16) & 0xff));
        atomic_inc(localhist + ((value >> 24) & 0xff));
#elif kercn >= 2
        T value = *(__global const T *)(src + src_index);
        atomic_inc(localhist + value.s0);
        atomic_inc(localhist + value.s1);
#if kercn >= 4
        atomic_inc(localhist + value.s2);
        atomic_inc(localhist + value.s3);
#if kercn >= 8
        atomic_inc(localhist + value.s4);
        atomic_inc(localhist + value.s5);
        atomic_inc(localhist + value.s6);
        atomic_inc(localhist + value.s7);
#if kercn == 16
        atomic_inc(localhist + value.s8);
        atomic_inc(localhist + value.s9);
        atomic_inc(localhist + value.sA);
        atomic_inc(localhist + value.sB);
        atomic_inc(localhist + value.sC);
        atomic_inc(localhist + value.sD);
        atomic_inc(localhist + value.sE);
        atomic_inc(localhist + value.sF);
#endif
#endif
#endif
#endif
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    __global int * hist = (__global int *)(histptr + gid * BINS * (int)sizeof(int));
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        hist[i] = localhist[i];
}

#ifndef HT
#define HT int
#endif

#ifndef convertToHT
#define convertToHT noconvert
#endif

__kernel void merge_histogram(__global const int * ghist, __global uchar * histptr, int hist_step, int hist_offset)
{
    int lid = get_local_id(0);

    __global HT * hist = (__global HT *)(histptr + hist_offset);
#if WGS >= BINS
    HT res = (HT)(0);
#else
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        hist[i] = (HT)(0);
#endif

    #pragma unroll
    for (int i = 0; i < HISTS_COUNT; ++i)
    {
        #pragma unroll
        for (int j = lid; j < BINS; j += WGS)
#if WGS >= BINS
            res += convertToHT(ghist[j]);
#else
            hist[j] += convertToHT(ghist[j]);
#endif
        ghist += BINS;
    }

#if WGS >= BINS
    if (lid < BINS)
        *(__global HT *)(histptr + mad24(lid, hist_step, hist_offset)) = res;
#endif
}

__kernel void calcLUT(__global uchar * dst, __global const int * ghist, int total)
{
    int lid = get_local_id(0);
    __local int sumhist[BINS];
    __local float scale;

#if WGS >= BINS
    int res = 0;
#else
    #pragma unroll
    for (int i = lid; i < BINS; i += WGS)
        sumhist[i] = 0;
#endif

    #pragma unroll
    for (int i = 0; i < HISTS_COUNT; ++i)
    {
        #pragma unroll
        for (int j = lid; j < BINS; j += WGS)
#if WGS >= BINS
            res += ghist[j];
#else
            sumhist[j] += ghist[j];
#endif
        ghist += BINS;
    }

#if WGS >= BINS
    if (lid < BINS)
        sumhist[lid] = res;
#endif
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
