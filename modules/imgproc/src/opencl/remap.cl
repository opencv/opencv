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
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Wu Zailong, bullet@yeah.net
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

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

#ifdef INTER_NEAREST
#define convertToWT
#endif

#ifdef BORDER_CONSTANT
#define EXTRAPOLATE(v2, v) v = scalar;
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(v2, v) \
    { \
        v2 = max(min(v2, (int2)(src_cols - 1, src_rows - 1)), zero); \
        v = convertToWT(src[mad24(v2.y, src_step, v2.x + src_offset)]); \
    }
#elif defined BORDER_WRAP
#define EXTRAPOLATE(v2, v) \
    { \
        if (v2.x < 0) \
            v2.x -= ((v2.x - src_cols + 1) / src_cols) * src_cols; \
        if (v2.x >= src_cols) \
            v2.x %= src_cols; \
        \
        if (v2.y < 0) \
            v2.y -= ((v2.y - src_rows + 1) / src_rows) * src_rows; \
        if( v2.y >= src_rows ) \
            v2.y %= src_rows; \
        v = convertToWT(src[mad24(v2.y, src_step, v2.x + src_offset)]); \
    }
#elif defined(BORDER_REFLECT) || defined(BORDER_REFLECT_101)
#ifdef BORDER_REFLECT
#define DELTA int delta = 0
#else
#define DELTA int delta = 1
#endif
#define EXTRAPOLATE(v2, v) \
    { \
        DELTA; \
        if (src_cols == 1) \
            v2.x = 0; \
        else \
            do \
            { \
                if( v2.x < 0 ) \
                    v2.x = -v2.x - 1 + delta; \
                else \
                    v2.x = src_cols - 1 - (v2.x - src_cols) - delta; \
            } \
            while (v2.x >= src_cols || v2.x < 0); \
        \
        if (src_rows == 1) \
            v2.y = 0; \
        else \
            do \
            { \
                if( v2.y < 0 ) \
                    v2.y = -v2.y - 1 + delta; \
                else \
                    v2.y = src_rows - 1 - (v2.y - src_rows) - delta; \
            } \
            while (v2.y >= src_rows || v2.y < 0); \
        v = convertToWT(src[mad24(v2.y, src_step, v2.x + src_offset)]); \
    }
#else
#error No extrapolation method
#endif

#define NEED_EXTRAPOLATION(gx, gy) (gx >= src_cols || gy >= src_rows || gx < 0 || gy < 0)

#ifdef INTER_NEAREST

__kernel void remap_2_32FC1(__global const T * restrict src, __global T * dst,
        __global float * map1, __global float * map2,
        int src_offset, int dst_offset, int map1_offset, int map2_offset,
        int src_step, int dst_step, int map1_step, int map2_step,
        int src_cols, int src_rows, int dst_cols, int dst_rows, T scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int dstIdx = mad24(y, dst_step, x + dst_offset);
        int map1Idx = mad24(y, map1_step, x + map1_offset);
        int map2Idx = mad24(y, map2_step, x + map2_offset);

        int gx = convert_int_sat_rte(map1[map1Idx]);
        int gy = convert_int_sat_rte(map2[map2Idx]);

        if (NEED_EXTRAPOLATION(gx, gy))
        {
            int2 gxy = (int2)(gx, gy), zero = (int2)(0);
            EXTRAPOLATE(gxy, dst[dstIdx]);
        }
        else
        {
            int srcIdx = mad24(gy, src_step, gx + src_offset);
            dst[dstIdx] = src[srcIdx];
        }
    }
}

__kernel void remap_32FC2(__global const T * restrict src, __global T * dst, __global float2 * map1,
        int src_offset, int dst_offset, int map1_offset,
        int src_step, int dst_step, int map1_step,
        int src_cols, int src_rows, int dst_cols, int dst_rows, T scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int dstIdx = mad24(y, dst_step, x + dst_offset);
        int map1Idx = mad24(y, map1_step, x + map1_offset);

        int2 gxy = convert_int2_sat_rte(map1[map1Idx]);
        int gx = gxy.x, gy = gxy.y;

        if (NEED_EXTRAPOLATION(gx, gy))
        {
            int2 zero = (int2)(0);
            EXTRAPOLATE(gxy, dst[dstIdx]);
        }
        else
        {
            int srcIdx = mad24(gy, src_step, gx + src_offset);
            dst[dstIdx] = src[srcIdx];
        }
    }
}

__kernel void remap_16SC2(__global const T * restrict src, __global T * dst, __global short2 * map1,
        int src_offset, int dst_offset, int map1_offset,
        int src_step, int dst_step, int map1_step,
        int src_cols, int src_rows, int dst_cols, int dst_rows, T scalar)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int dstIdx = mad24(y, dst_step, x + dst_offset);
        int map1Idx = mad24(y, map1_step, x + map1_offset);

        int2 gxy = convert_int2(map1[map1Idx]);
        int gx = gxy.x, gy = gxy.y;

        if (NEED_EXTRAPOLATION(gx, gy))
        {
            int2 zero = (int2)(0);
            EXTRAPOLATE(gxy, dst[dstIdx]);
        }
        else
        {
            int srcIdx = mad24(gy, src_step, gx + src_offset);
            dst[dstIdx] = src[srcIdx];
        }
    }
}

#elif INTER_LINEAR

__kernel void remap_2_32FC1(__global T const * restrict  src, __global T * dst,
        __global float * map1, __global float * map2,
        int src_offset, int dst_offset, int map1_offset, int map2_offset,
        int src_step, int dst_step, int map1_step, int map2_step,
        int src_cols, int src_rows, int dst_cols, int dst_rows, T nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int dstIdx = mad24(y, dst_step, x + dst_offset);
        int map1Idx = mad24(y, map1_step, x + map1_offset);
        int map2Idx = mad24(y, map2_step, x + map2_offset);

        float2 map_data = (float2)(map1[map1Idx], map2[map2Idx]);

        int2 map_dataA = convert_int2_sat_rtn(map_data);
        int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
        int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
        int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y +1);
        int2 zero = (int2)(0);

        float2 _u = map_data - convert_float2(map_dataA);
        WT2 u = convertToWT2(convert_int2_rte(convertToWT2(_u) * (WT2)32)) / (WT2)32;
        WT scalar = convertToWT(nVal);
        WT a = scalar, b = scalar, c = scalar, d = scalar;

        if (!NEED_EXTRAPOLATION(map_dataA.x, map_dataA.y))
            a = convertToWT(src[mad24(map_dataA.y, src_step, map_dataA.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataA, a);

        if (!NEED_EXTRAPOLATION(map_dataB.x, map_dataB.y))
            b = convertToWT(src[mad24(map_dataB.y, src_step, map_dataB.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataB, b);

        if (!NEED_EXTRAPOLATION(map_dataC.x, map_dataC.y))
            c = convertToWT(src[mad24(map_dataC.y, src_step, map_dataC.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataC, c);

        if (!NEED_EXTRAPOLATION(map_dataD.x, map_dataD.y))
            d = convertToWT(src[mad24(map_dataD.y, src_step, map_dataD.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataD, d);

        WT dst_data = a * (WT)(1 - u.x) * (WT)(1 - u.y) +
                      b * (WT)(u.x)     * (WT)(1 - u.y) +
                      c * (WT)(1 - u.x) * (WT)(u.y) +
                      d * (WT)(u.x)     * (WT)(u.y);
        dst[dstIdx] = convertToT(dst_data);
    }
}

__kernel void remap_32FC2(__global T const * restrict  src, __global T * dst,
        __global float2 * map1,
        int src_offset, int dst_offset, int map1_offset,
        int src_step, int dst_step, int map1_step,
        int src_cols, int src_rows, int dst_cols, int dst_rows, T nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int dstIdx = mad24(y, dst_step, x + dst_offset);
        int map1Idx = mad24(y, map1_step, x + map1_offset);

        float2 map_data = map1[map1Idx];
        int2 map_dataA = convert_int2_sat_rtn(map_data);
        int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
        int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
        int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y + 1);
        int2 zero = (int2)(0);

        float2 _u = map_data - convert_float2(map_dataA);
        WT2 u = convertToWT2(convert_int2_rte(convertToWT2(_u) * (WT2)32)) / (WT2)32;
        WT scalar = convertToWT(nVal);
        WT a = scalar, b = scalar, c = scalar, d = scalar;

        if (!NEED_EXTRAPOLATION(map_dataA.x, map_dataA.y))
            a = convertToWT(src[mad24(map_dataA.y, src_step, map_dataA.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataA, a);

        if (!NEED_EXTRAPOLATION(map_dataB.x, map_dataB.y))
            b = convertToWT(src[mad24(map_dataB.y, src_step, map_dataB.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataB, b);

        if (!NEED_EXTRAPOLATION(map_dataC.x, map_dataC.y))
            c = convertToWT(src[mad24(map_dataC.y, src_step, map_dataC.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataC, c);

        if (!NEED_EXTRAPOLATION(map_dataD.x, map_dataD.y))
            d = convertToWT(src[mad24(map_dataD.y, src_step, map_dataD.x + src_offset)]);
        else
            EXTRAPOLATE(map_dataD, d);

        WT dst_data = a * (WT)(1 - u.x) * (WT)(1 - u.y) +
                      b * (WT)(u.x)     * (WT)(1 - u.y) +
                      c * (WT)(1 - u.x) * (WT)(u.y) +
                      d * (WT)(u.x)     * (WT)(u.y);
        dst[dstIdx] = convertToT(dst_data);
    }
}

#endif
