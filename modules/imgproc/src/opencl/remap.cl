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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

#if cn != 3
#define loadpix(addr)  *(__global const T*)(addr)
#define storepix(val, addr)  *(__global T*)(addr) = val
#define TSIZE ((int)sizeof(T))
#define convertScalar(a) (a)
#else
#define loadpix(addr)  vload3(0, (__global const T1*)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1*)(addr))
#define TSIZE ((int)sizeof(T1)*3)
#define convertScalar(a) (T)(a.x, a.y, a.z)
#endif

enum
{
    INTER_BITS = 5,
    INTER_TAB_SIZE = 1 << INTER_BITS,
    INTER_TAB_SIZE2 = INTER_TAB_SIZE * INTER_TAB_SIZE
};

#ifdef INTER_NEAREST
#define convertToWT
#endif

#ifdef BORDER_CONSTANT
#define EXTRAPOLATE(v2, v) v = scalar;
#elif defined BORDER_REPLICATE
#define EXTRAPOLATE(v2, v) \
    { \
        v2 = max(min(v2, (int2)(src_cols - 1, src_rows - 1)), (int2)(0)); \
        v = convertToWT(loadpix((__global const T*)(srcptr + mad24(v2.y, src_step, v2.x * TSIZE + src_offset)))); \
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
        v = convertToWT(loadpix((__global const T*)(srcptr + mad24(v2.y, src_step, v2.x * TSIZE + src_offset)))); \
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
        v = convertToWT(loadpix((__global const T*)(srcptr + mad24(v2.y, src_step, v2.x * TSIZE + src_offset)))); \
    }
#else
#error No extrapolation method
#endif

#define NEED_EXTRAPOLATION(gx, gy) (gx >= src_cols || gy >= src_rows || gx < 0 || gy < 0)

#ifdef INTER_NEAREST

__kernel void remap_2_32FC1(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                            __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                            __global const uchar * map1ptr, int map1_step, int map1_offset,
                            __global const uchar * map2ptr, int map2_step, int map2_offset,
                            ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        T scalar = convertScalar(nVal);

        int map1_index = mad24(y, map1_step, mad24(x, (int)sizeof(float), map1_offset));
        int map2_index = mad24(y, map2_step, mad24(x, (int)sizeof(float), map2_offset));
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map1_index += map1_step, map2_index += map2_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const float * map1 = (__global const float *)(map1ptr + map1_index);
                __global const float * map2 = (__global const float *)(map2ptr + map2_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                int gx = convert_int_sat_rte(map1[0]);
                int gy = convert_int_sat_rte(map2[0]);

                if (NEED_EXTRAPOLATION(gx, gy))
                {
#ifndef BORDER_CONSTANT
                    int2 gxy = (int2)(gx, gy);
#endif
                    T v;
                    EXTRAPOLATE(gxy, v)
                    storepix(v, dst);
                }
                else
                {
                    int src_index = mad24(gy, src_step, mad24(gx, TSIZE, src_offset));
                    storepix(loadpix((__global const T*)(srcptr + src_index)), dst);
                }
            }
    }
}

__kernel void remap_32FC2(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                          __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                          __global const uchar * mapptr, int map_step, int map_offset,
                          ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        T scalar = convertScalar(nVal);
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));
        int map_index = mad24(y, map_step, mad24(x, (int)sizeof(float2), map_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map_index += map_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const float2 * map = (__global const float2 *)(mapptr + map_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                int2 gxy = convert_int2_sat_rte(map[0]);
                int gx = gxy.x, gy = gxy.y;

                if (NEED_EXTRAPOLATION(gx, gy))
                {
                    T v;
                    EXTRAPOLATE(gxy, v)
                    storepix(v, dst);
                }
                else
                {
                    int src_index = mad24(gy, src_step, mad24(gx, TSIZE, src_offset));
                    storepix(loadpix((__global const T *)(srcptr + src_index)), dst);
                }
        }
    }
}

__kernel void remap_16SC2(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                          __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                          __global const uchar * mapptr, int map_step, int map_offset,
                          ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        T scalar = convertScalar(nVal);
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));
        int map_index = mad24(y, map_step, mad24(x, (int)sizeof(short2), map_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map_index += map_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const short2 * map = (__global const short2 *)(mapptr + map_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                int2 gxy = convert_int2(map[0]);
                int gx = gxy.x, gy = gxy.y;

                if (NEED_EXTRAPOLATION(gx, gy))
                {
                    T v;
                    EXTRAPOLATE(gxy, v)
                    storepix(v, dst);
                }
                else
                {
                    int src_index = mad24(gy, src_step, mad24(gx, TSIZE, src_offset));
                    storepix(loadpix((__global const T *)(srcptr + src_index)), dst);
                }
            }
    }
}

__kernel void remap_16SC2_16UC1(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                                __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                __global const uchar * map1ptr, int map1_step, int map1_offset,
                                __global const uchar * map2ptr, int map2_step, int map2_offset,
                                ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        T scalar = convertScalar(nVal);
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));
        int map1_index = mad24(y, map1_step, mad24(x, (int)sizeof(short2), map1_offset));
        int map2_index = mad24(y, map2_step, mad24(x, (int)sizeof(ushort), map2_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map1_index += map1_step, map2_index += map2_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const short2 * map1 = (__global const short2 *)(map1ptr + map1_index);
                __global const ushort * map2 = (__global const ushort *)(map2ptr + map2_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                int map2Value = convert_int(map2[0]) & (INTER_TAB_SIZE2 - 1);
                int dx = (map2Value & (INTER_TAB_SIZE - 1)) < (INTER_TAB_SIZE >> 1) ? 1 : 0;
                int dy = (map2Value >> INTER_BITS) < (INTER_TAB_SIZE >> 1) ? 1 : 0;
                int2 gxy = convert_int2(map1[0]) + (int2)(dx, dy);
                int gx = gxy.x, gy = gxy.y;

                if (NEED_EXTRAPOLATION(gx, gy))
                {
                    T v;
                    EXTRAPOLATE(gxy, v)
                    storepix(v, dst);
                }
                else
                {
                    int src_index = mad24(gy, src_step, mad24(gx, TSIZE, src_offset));
                    storepix(loadpix((__global const T *)(srcptr + src_index)), dst);
                }
            }
    }
}

#elif defined INTER_LINEAR

__constant float coeffs[64] =
{ 1.000000f, 0.000000f, 0.968750f, 0.031250f, 0.937500f, 0.062500f, 0.906250f, 0.093750f, 0.875000f, 0.125000f, 0.843750f, 0.156250f,
  0.812500f, 0.187500f, 0.781250f, 0.218750f, 0.750000f, 0.250000f, 0.718750f, 0.281250f, 0.687500f, 0.312500f, 0.656250f, 0.343750f,
  0.625000f, 0.375000f, 0.593750f, 0.406250f, 0.562500f, 0.437500f, 0.531250f, 0.468750f, 0.500000f, 0.500000f, 0.468750f, 0.531250f,
  0.437500f, 0.562500f, 0.406250f, 0.593750f, 0.375000f, 0.625000f, 0.343750f, 0.656250f, 0.312500f, 0.687500f, 0.281250f, 0.718750f,
  0.250000f, 0.750000f, 0.218750f, 0.781250f, 0.187500f, 0.812500f, 0.156250f, 0.843750f, 0.125000f, 0.875000f, 0.093750f, 0.906250f,
  0.062500f, 0.937500f, 0.031250f, 0.968750f };

__kernel void remap_16SC2_16UC1(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                                __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                                __global const uchar * map1ptr, int map1_step, int map1_offset,
                                __global const uchar * map2ptr, int map2_step, int map2_offset,
                                ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        WT scalar = convertToWT(convertScalar(nVal));
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));
        int map1_index = mad24(y, map1_step, mad24(x, (int)sizeof(short2), map1_offset));
        int map2_index = mad24(y, map2_step, mad24(x, (int)sizeof(ushort), map2_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map1_index += map1_step, map2_index += map2_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const short2 * map1 = (__global const short2 *)(map1ptr + map1_index);
                __global const ushort * map2 = (__global const ushort *)(map2ptr + map2_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                int2 map_dataA = convert_int2(map1[0]);
                int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
                int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
                int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y + 1);

                ushort map2Value = (ushort)(map2[0] & (INTER_TAB_SIZE2 - 1));
                WT2 u = (WT2)(map2Value & (INTER_TAB_SIZE - 1), map2Value >> INTER_BITS) / (WT2)(INTER_TAB_SIZE);

                WT a = scalar, b = scalar, c = scalar, d = scalar;

                if (!NEED_EXTRAPOLATION(map_dataA.x, map_dataA.y))
                    a = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataA.y, src_step, map_dataA.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataA, a);

                if (!NEED_EXTRAPOLATION(map_dataB.x, map_dataB.y))
                    b = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataB.y, src_step, map_dataB.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataB, b);

                if (!NEED_EXTRAPOLATION(map_dataC.x, map_dataC.y))
                    c = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataC.y, src_step, map_dataC.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataC, c);

                if (!NEED_EXTRAPOLATION(map_dataD.x, map_dataD.y))
                    d = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataD.y, src_step, map_dataD.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataD, d);

                WT dst_data = a * (1 - u.x) * (1 - u.y) +
                              b * (u.x)     * (1 - u.y) +
                              c * (1 - u.x) * (u.y) +
                              d * (u.x)     * (u.y);
                storepix(convertToT(dst_data), dst);
            }
    }
}

__kernel void remap_2_32FC1(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                            __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                            __global const uchar * map1ptr, int map1_step, int map1_offset,
                            __global const uchar * map2ptr, int map2_step, int map2_offset,
                            ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        WT scalar = convertToWT(convertScalar(nVal));
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));
        int map1_index = mad24(y, map1_step, mad24(x, (int)sizeof(float), map1_offset));
        int map2_index = mad24(y, map2_step, mad24(x, (int)sizeof(float), map2_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map1_index += map1_step, map2_index += map2_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const float * map1 = (__global const float *)(map1ptr + map1_index);
                __global const float * map2 = (__global const float *)(map2ptr + map2_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

#if defined BORDER_CONSTANT
                float xf = map1[0], yf = map2[0];
                int sx = convert_int_sat_rtz(mad(xf, (float)INTER_TAB_SIZE, 0.5f)) >> INTER_BITS;
                int sy = convert_int_sat_rtz(mad(yf, (float)INTER_TAB_SIZE, 0.5f)) >> INTER_BITS;

                __constant float * coeffs_x = coeffs + ((convert_int_rte(xf * INTER_TAB_SIZE) & (INTER_TAB_SIZE - 1)) << 1);
                __constant float * coeffs_y = coeffs + ((convert_int_rte(yf * INTER_TAB_SIZE) & (INTER_TAB_SIZE - 1)) << 1);

                WT sum = (WT)(0), xsum;
                int src_index = mad24(sy, src_step, mad24(sx, TSIZE, src_offset));

                #pragma unroll
                for (int yp = 0; yp < 2; ++yp, src_index += src_step)
                {
                    if (sy + yp >= 0 && sy + yp < src_rows)
                    {
                        xsum = (WT)(0);
                        if (sx >= 0 && sx + 2 < src_cols)
                        {
#if depth == 0 && cn == 1
                            uchar2 value = vload2(0, srcptr + src_index);
                            xsum = dot(convert_float2(value), (float2)(coeffs_x[0], coeffs_x[1]));
#else
                            #pragma unroll
                            for (int xp = 0; xp < 2; ++xp)
                                xsum = fma(convertToWT(loadpix(srcptr + mad24(xp, TSIZE, src_index))), coeffs_x[xp], xsum);
#endif
                        }
                        else
                        {
                            #pragma unroll
                            for (int xp = 0; xp < 2; ++xp)
                                xsum = fma(sx + xp >= 0 && sx + xp < src_cols ?
                                           convertToWT(loadpix(srcptr + mad24(xp, TSIZE, src_index))) : scalar, coeffs_x[xp], xsum);
                        }
                        sum = fma(xsum, coeffs_y[yp], sum);
                    }
                    else
                        sum = fma(scalar, coeffs_y[yp], sum);
                }

                storepix(convertToT(sum), dst);
#else
                float2 map_data = (float2)(map1[0], map2[0]);

                int2 map_dataA = convert_int2_sat_rtn(map_data);
                int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
                int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
                int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y + 1);

                float2 _u = map_data - convert_float2(map_dataA);
                WT2 u = convertToWT2(convert_int2_rte(convertToWT2(_u) * (WT2)INTER_TAB_SIZE)) / (WT2)INTER_TAB_SIZE;
                WT scalar = convertToWT(convertScalar(nVal));
                WT a = scalar, b = scalar, c = scalar, d = scalar;

                if (!NEED_EXTRAPOLATION(map_dataA.x, map_dataA.y))
                    a = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataA.y, src_step, map_dataA.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataA, a);

                if (!NEED_EXTRAPOLATION(map_dataB.x, map_dataB.y))
                    b = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataB.y, src_step, map_dataB.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataB, b);

                if (!NEED_EXTRAPOLATION(map_dataC.x, map_dataC.y))
                    c = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataC.y, src_step, map_dataC.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataC, c);

                if (!NEED_EXTRAPOLATION(map_dataD.x, map_dataD.y))
                    d = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataD.y, src_step, map_dataD.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataD, d);

                WT dst_data = a * (1 - u.x) * (1 - u.y) +
                              b * (u.x)     * (1 - u.y) +
                              c * (1 - u.x) * (u.y) +
                              d * (u.x)     * (u.y);
                storepix(convertToT(dst_data), dst);
#endif
            }
    }
}

__kernel void remap_32FC2(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                          __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                          __global const uchar * mapptr, int map_step, int map_offset,
                          ST nVal)
{
    int x = get_global_id(0);
    int y = get_global_id(1) * rowsPerWI;

    if (x < dst_cols)
    {
        WT scalar = convertToWT(convertScalar(nVal));
        int dst_index = mad24(y, dst_step, mad24(x, TSIZE, dst_offset));
        int map_index = mad24(y, map_step, mad24(x, (int)sizeof(float2), map_offset));

        #pragma unroll
        for (int i = 0; i < rowsPerWI; ++i, ++y,
            map_index += map_step, dst_index += dst_step)
            if (y < dst_rows)
            {
                __global const float2 * map = (__global const float2 *)(mapptr + map_index);
                __global T * dst = (__global T *)(dstptr + dst_index);

                float2 map_data = map[0];
                int2 map_dataA = convert_int2_sat_rtn(map_data);
                int2 map_dataB = (int2)(map_dataA.x + 1, map_dataA.y);
                int2 map_dataC = (int2)(map_dataA.x, map_dataA.y + 1);
                int2 map_dataD = (int2)(map_dataA.x + 1, map_dataA.y + 1);

                float2 _u = map_data - convert_float2(map_dataA);
                WT2 u = convertToWT2(convert_int2_rte(convertToWT2(_u) * (WT2)INTER_TAB_SIZE)) / (WT2)INTER_TAB_SIZE;
                WT a = scalar, b = scalar, c = scalar, d = scalar;

                if (!NEED_EXTRAPOLATION(map_dataA.x, map_dataA.y))
                    a = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataA.y, src_step, map_dataA.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataA, a);

                if (!NEED_EXTRAPOLATION(map_dataB.x, map_dataB.y))
                    b = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataB.y, src_step, map_dataB.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataB, b);

                if (!NEED_EXTRAPOLATION(map_dataC.x, map_dataC.y))
                    c = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataC.y, src_step, map_dataC.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataC, c);

                if (!NEED_EXTRAPOLATION(map_dataD.x, map_dataD.y))
                    d = convertToWT(loadpix((__global const T *)(srcptr + mad24(map_dataD.y, src_step, map_dataD.x * TSIZE + src_offset))));
                else
                    EXTRAPOLATE(map_dataD, d);

                WT dst_data = a * (1 - u.x) * (1 - u.y) +
                              b * (u.x)     * (1 - u.y) +
                              c * (1 - u.x) * (u.y) +
                              d * (u.x)     * (u.y);
                storepix(convertToT(dst_data), dst);
            }
    }
}

#endif
