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
//    Zhang Ying, zhangying913@gmail.com
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
#define CT double
#else
#define CT float
#endif

#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f / INTER_TAB_SIZE
#define AB_BITS max(10, (int)INTER_BITS)
#define AB_SCALE (1 << AB_BITS)
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)

#define noconvert

#ifndef ST
#define ST T
#endif

#if cn != 3
#define loadpix(addr)  *(__global const T*)(addr)
#define storepix(val, addr)  *(__global T*)(addr) = val
#define scalar scalar_
#define pixsize (int)sizeof(T)
#else
#define loadpix(addr)  vload3(0, (__global const T1*)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1*)(addr))
#ifdef INTER_NEAREST
#define scalar (T)(scalar_.x, scalar_.y, scalar_.z)
#else
#define scalar (WT)(scalar_.x, scalar_.y, scalar_.z)
#endif
#define pixsize ((int)sizeof(T1)*3)
#endif

#ifdef INTER_NEAREST

__kernel void warpPerspective(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        CT X0 = M[0] * dx + M[1] * dy + M[2];
        CT Y0 = M[3] * dx + M[4] * dy + M[5];
        CT W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? 1.f / W : 0.0f;
        short sx = convert_short_sat_rte(X0*W);
        short sy = convert_short_sat_rte(Y0*W);

        int dst_index = mad24(dy, dst_step, dx * pixsize + dst_offset);

        if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows)
        {
            int src_index = mad24(sy, src_step, sx * pixsize + src_offset);
            storepix(loadpix(srcptr + src_index), dstptr + dst_index);
        }
        else
            storepix(scalar, dstptr + dst_index);
    }
}

#elif defined INTER_LINEAR

__kernel void warpPerspective(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        CT X0 = M[0] * dx + M[1] * dy + M[2];
        CT Y0 = M[3] * dx + M[4] * dy + M[5];
        CT W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
        int X = rint(X0 * W), Y = rint(Y0 * W);

        short sx = convert_short_sat(X >> INTER_BITS);
        short sy = convert_short_sat(Y >> INTER_BITS);
        short ay = (short)(Y & (INTER_TAB_SIZE - 1));
        short ax = (short)(X & (INTER_TAB_SIZE - 1));

        WT v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ?
            convertToWT(loadpix(srcptr + mad24(sy, src_step, src_offset + sx * pixsize))) : scalar;
        WT v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ?
            convertToWT(loadpix(srcptr + mad24(sy, src_step, src_offset + (sx+1) * pixsize))) : scalar;
        WT v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            convertToWT(loadpix(srcptr + mad24(sy+1, src_step, src_offset + sx * pixsize))) : scalar;
        WT v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            convertToWT(loadpix(srcptr + mad24(sy+1, src_step, src_offset + (sx+1) * pixsize))) : scalar;

        float taby = 1.f/INTER_TAB_SIZE*ay;
        float tabx = 1.f/INTER_TAB_SIZE*ax;

        int dst_index = mad24(dy, dst_step, dst_offset + dx * pixsize);

#if depth <= 4
        int itab0 = convert_short_sat_rte( (1.0f-taby)*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab1 = convert_short_sat_rte( (1.0f-taby)*tabx * INTER_REMAP_COEF_SCALE );
        int itab2 = convert_short_sat_rte( taby*(1.0f-tabx) * INTER_REMAP_COEF_SCALE );
        int itab3 = convert_short_sat_rte( taby*tabx * INTER_REMAP_COEF_SCALE );

        WT val = v0 * itab0 +  v1 * itab1 + v2 * itab2 + v3 * itab3;
        storepix(convertToT((val + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS), dstptr + dst_index);
#else
        float tabx2 = 1.0f - tabx, taby2 = 1.0f - taby;
        WT val = v0 * tabx2 * taby2 +  v1 * tabx * taby2 + v2 * tabx2 * taby + v3 * tabx * taby;
        storepix(convertToT(val), dstptr + dst_index);
#endif
    }
}

#elif defined INTER_CUBIC

inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = ((A*(x + 1.f) - 5.0f*A)*(x + 1.f) + 8.0f*A)*(x + 1.f) - 4.0f*A;
    coeffs[1] = ((A + 2.f)*x - (A + 3.f))*x*x + 1.f;
    coeffs[2] = ((A + 2.f)*(1.f - x) - (A + 3.f))*(1.f - x)*(1.f - x) + 1.f;
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

__kernel void warpPerspective(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        CT X0 = M[0] * dx + M[1] * dy + M[2];
        CT Y0 = M[3] * dx + M[4] * dy + M[5];
        CT W = M[6] * dx + M[7] * dy + M[8];
        W = W != 0.0f ? INTER_TAB_SIZE / W : 0.0f;
        int X = rint(X0 * W), Y = rint(Y0 * W);

        short sx = convert_short_sat(X >> INTER_BITS) - 1;
        short sy = convert_short_sat(Y >> INTER_BITS) - 1;
        short ay = (short)(Y & (INTER_TAB_SIZE-1));
        short ax = (short)(X & (INTER_TAB_SIZE-1));

        WT v[16];
        #pragma unroll
        for (int y = 0; y < 4; y++)
            #pragma unroll
            for (int x = 0; x < 4; x++)
                v[mad24(y, 4, x)] = (sx+x >= 0 && sx+x < src_cols && sy+y >= 0 && sy+y < src_rows) ?
                    convertToWT(loadpix(srcptr + mad24(sy+y, src_step, src_offset + (sx+x) * pixsize))) : scalar;

        float tab1y[4], tab1x[4];

        float ayy = INTER_SCALE * ay;
        float axx = INTER_SCALE * ax;
        interpolateCubic(ayy, tab1y);
        interpolateCubic(axx, tab1x);

        int dst_index = mad24(dy, dst_step, dst_offset + dx * pixsize);

        WT sum = (WT)(0);
#if depth <= 4
        int itab[16];

        #pragma unroll
        for (int i = 0; i < 16; i++)
            itab[i] = rint(tab1y[(i>>2)] * tab1x[(i&3)] * INTER_REMAP_COEF_SCALE);

        #pragma unroll
        for (int i = 0; i < 16; i++)
            sum += v[i] * itab[i];
        storepix(convertToT( (sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ), dstptr + dst_index);
#else
        #pragma unroll
        for (int i = 0; i < 16; i++)
            sum += v[i] * tab1y[(i>>2)] * tab1x[(i&3)];
        storepix(convertToT( sum ), dstptr + dst_index);
#endif
    }
}

#endif
