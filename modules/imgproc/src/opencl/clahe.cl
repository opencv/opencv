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
//    Sen Liu, swjtuls1987@126.com
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

#ifndef WAVE_SIZE
#define WAVE_SIZE 1
#endif

inline int calc_lut(__local int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid == 0)
        for (int i = 1; i < 256; ++i)
            smem[i] += smem[i - 1];
    barrier(CLK_LOCAL_MEM_FENCE);

    return smem[tid];
}

#ifdef CPU
inline void reduce(volatile __local int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128)
        smem[tid] = val += smem[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64)
        smem[tid] = val += smem[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
        smem[tid] += smem[tid + 32];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16)
        smem[tid] += smem[tid + 16];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8)
        smem[tid] += smem[tid + 8];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 4)
        smem[tid] += smem[tid + 4];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 2)
        smem[tid] += smem[tid + 2];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 1)
        smem[256] = smem[tid] + smem[tid + 1];
    barrier(CLK_LOCAL_MEM_FENCE);
}

#else

inline void reduce(__local volatile int* smem, int val, int tid)
{
    smem[tid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 128)
        smem[tid] = val += smem[tid + 128];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 64)
        smem[tid] = val += smem[tid + 64];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 32)
    {
        smem[tid] += smem[tid + 32];
#if WAVE_SIZE < 32
    } barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 16)
    {
#endif
        smem[tid] += smem[tid + 16];
#if WAVE_SIZE < 16
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (tid < 8)
    {
#endif
        smem[tid] += smem[tid + 8];
        smem[tid] += smem[tid + 4];
        smem[tid] += smem[tid + 2];
        smem[tid] += smem[tid + 1];
    }
}
#endif

__kernel void calcLut(__global __const uchar * src, const int srcStep,
                      const int src_offset, __global uchar * lut,
                      const int dstStep, const int dst_offset,
                      const int2 tileSize, const int tilesX,
                      const int clipLimit, const float lutScale)
{
    __local int smem[512];

    int tx = get_group_id(0);
    int ty = get_group_id(1);
    int tid = get_local_id(1) * get_local_size(0)
                             + get_local_id(0);
    smem[tid] = 0;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = get_local_id(1); i < tileSize.y; i += get_local_size(1))
    {
        __global const uchar* srcPtr = src + mad24(ty * tileSize.y + i, srcStep, tx * tileSize.x + src_offset);
        for (int j = get_local_id(0); j < tileSize.x; j += get_local_size(0))
        {
            const int data = srcPtr[j];
            atomic_inc(&smem[data]);
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    int tHistVal = smem[tid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (clipLimit > 0)
    {
        // clip histogram bar
        int clipped = 0;
        if (tHistVal > clipLimit)
        {
            clipped = tHistVal - clipLimit;
            tHistVal = clipLimit;
        }

        // find number of overall clipped samples
        reduce(smem, clipped, tid);
        barrier(CLK_LOCAL_MEM_FENCE);
#ifdef CPU
        clipped = smem[256];
#else
        clipped = smem[0];
#endif

        // broadcast evaluated value

        __local int totalClipped;

        if (tid == 0)
            totalClipped = clipped;
        barrier(CLK_LOCAL_MEM_FENCE);

        // redistribute clipped samples evenly

        int redistBatch = totalClipped / 256;
        tHistVal += redistBatch;

        int residual = totalClipped - redistBatch * 256;
        int rStep = 256 / residual;
        if (rStep < 1)
            rStep = 1;
        if (tid%rStep == 0 && (tid/rStep)<residual)
            ++tHistVal;
    }

    const int lutVal = calc_lut(smem, tHistVal, tid);
    uint ires = (uint)convert_int_rte(lutScale * lutVal);
    lut[(ty * tilesX + tx) * dstStep + tid + dst_offset] =
        convert_uchar(clamp(ires, (uint)0, (uint)255));
}

__kernel void transform(__global __const uchar * src, const int srcStep, const int src_offset,
                        __global uchar * dst, const int dstStep, const int dst_offset,
                        __global uchar * lut, const int lutStep, int lut_offset,
                        const int cols, const int rows,
                        const int2 tileSize,
                        const int tilesX, const int tilesY)
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    if (x >= cols || y >= rows)
        return;

    const float tyf = (convert_float(y) / tileSize.y) - 0.5f;
    int ty1 = convert_int_rtn(tyf);
    int ty2 = ty1 + 1;
    const float ya = tyf - ty1;
    ty1 = max(ty1, 0);
    ty2 = min(ty2, tilesY - 1);

    const float txf = (convert_float(x) / tileSize.x) - 0.5f;
    int tx1 = convert_int_rtn(txf);
    int tx2 = tx1 + 1;
    const float xa = txf - tx1;
    tx1 = max(tx1, 0);
    tx2 = min(tx2, tilesX - 1);

    const int srcVal = src[mad24(y, srcStep, x + src_offset)];

    float res = 0;

    res += lut[mad24(ty1 * tilesX + tx1, lutStep, srcVal + lut_offset)] * ((1.0f - xa) * (1.0f - ya));
    res += lut[mad24(ty1 * tilesX + tx2, lutStep, srcVal + lut_offset)] * ((xa) * (1.0f - ya));
    res += lut[mad24(ty2 * tilesX + tx1, lutStep, srcVal + lut_offset)] * ((1.0f - xa) * (ya));
    res += lut[mad24(ty2 * tilesX + tx2, lutStep, srcVal + lut_offset)] * ((xa) * (ya));

    uint ires = (uint)convert_int_rte(res);
    dst[mad24(y, dstStep, x + dst_offset)] = convert_uchar(clamp(ires, (uint)0, (uint)255));
}
