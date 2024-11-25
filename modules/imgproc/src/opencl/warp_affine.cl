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
#endif

#define INTER_BITS 5
#define INTER_TAB_SIZE (1 << INTER_BITS)
#define INTER_SCALE 1.f/INTER_TAB_SIZE
#define AB_BITS max(10, (int)INTER_BITS)
#define AB_SCALE (1 << AB_BITS)
#define INTER_REMAP_COEF_BITS 15
#define INTER_REMAP_COEF_SCALE (1 << INTER_REMAP_COEF_BITS)
#define ROUND_DELTA (1 << (AB_BITS - INTER_BITS - 1))

#define noconvert

#ifndef ST
#define ST T
#endif

#if CN != 3
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

__kernel void warpAffine(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy0 = get_global_id(1) * ROWS_PER_WI;

    if (dx < dst_cols)
    {
        float X0_ = fma(M[0], (CT)dx, M[2]);
        float Y0_ = fma(M[3], (CT)dx, M[5]);

        for (int dy = dy0, dy1 = min(dst_rows, dy0 + ROWS_PER_WI); dy < dy1; ++dy)
        {
            float X0 = fma(M[1], (CT)dy, X0_);
            float Y0 = fma(M[4], (CT)dy, Y0_);
            int sx = convert_short_rtn(X0);
            int sy = convert_short_rtn(Y0);

            WT v0 = scalar;
            if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows)
            {
                v0 = CONVERT_TO_WT(loadpix(srcptr + mad24(sy, src_step, mad24(sx, pixsize, src_offset))));
            }

            int dst_index = mad24(dy, dst_step, mad24(dx, pixsize, dst_offset));
            storepix(CONVERT_TO_T(v0), dstptr + dst_index);
        }
    }
}

#elif defined INTER_LINEAR

__kernel void warpAffine(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy0 = get_global_id(1) * ROWS_PER_WI;

    if (dx < dst_cols)
    {
        float X0_ = fma(M[0], (CT)dx, M[2]);
        float Y0_ = fma(M[3], (CT)dx, M[5]);

        for (int dy = dy0, dy1 = min(dst_rows, dy0 + ROWS_PER_WI); dy < dy1; ++dy)
        {
            float X0 = fma(M[1], (CT)dy, X0_);
            float Y0 = fma(M[4], (CT)dy, Y0_);
            int sx = convert_short_rtn(X0);
            int sy = convert_short_rtn(Y0);

            float ax = X0 - (CT)sx;
            float ay = Y0 - (CT)sy;

            WT v0 = scalar, v1 = scalar, v2 = scalar, v3 = scalar;
            if (sx >= 0 && sx < src_cols)
            {
                if (sy >= 0 && sy < src_rows)
                    v0 = CONVERT_TO_WT(loadpix(srcptr + mad24(sy, src_step, mad24(sx, pixsize, src_offset))));
                if (sy+1 >= 0 && sy+1 < src_rows)
                    v2 = CONVERT_TO_WT(loadpix(srcptr + mad24(sy+1, src_step, mad24(sx, pixsize, src_offset))));
            }
            if (sx+1 >= 0 && sx+1 < src_cols)
            {
                if (sy >= 0 && sy < src_rows)
                    v1 = CONVERT_TO_WT(loadpix(srcptr + mad24(sy, src_step, mad24(sx+1, pixsize, src_offset))));
                if (sy+1 >= 0 && sy+1 < src_rows)
                    v3 = CONVERT_TO_WT(loadpix(srcptr + mad24(sy+1, src_step, mad24(sx+1, pixsize, src_offset))));
            }

            int dst_index = mad24(dy, dst_step, mad24(dx, pixsize, dst_offset));

            v0 = fma(v1 - v0, ax, v0);
            v2 = fma(v3 - v2, ax, v2);
            v0 = fma(v2 - v0, ay, v0);
            storepix(CONVERT_TO_T(v0), dstptr + dst_index);
        }
    }
}

#elif defined INTER_CUBIC

#ifdef AMD_DEVICE

inline void interpolateCubic( float x, float* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = fma(fma(fma(A, (x + 1.f), - 5.0f*A), (x + 1.f), 8.0f*A), x + 1.f, - 4.0f*A);
    coeffs[1] = fma(fma(A + 2.f, x, - (A + 3.f)), x*x, 1.f);
    coeffs[2] = fma(fma(A + 2.f, 1.f - x, - (A + 3.f)), (1.f - x)*(1.f - x), 1.f);
    coeffs[3] = 1.f - coeffs[0] - coeffs[1] - coeffs[2];
}

#else

__constant float coeffs[128] =
    { 0.000000f, 1.000000f, 0.000000f, 0.000000f, -0.021996f, 0.997841f, 0.024864f, -0.000710f, -0.041199f, 0.991516f, 0.052429f, -0.002747f,
    -0.057747f, 0.981255f, 0.082466f, -0.005974f, -0.071777f, 0.967285f, 0.114746f, -0.010254f, -0.083427f, 0.949837f, 0.149040f, -0.015450f,
    -0.092834f, 0.929138f, 0.185120f, -0.021423f, -0.100136f, 0.905418f, 0.222755f, -0.028038f, -0.105469f, 0.878906f, 0.261719f, -0.035156f,
    -0.108971f, 0.849831f, 0.301781f, -0.042641f, -0.110779f, 0.818420f, 0.342712f, -0.050354f, -0.111031f, 0.784904f, 0.384285f, -0.058159f,
    -0.109863f, 0.749512f, 0.426270f, -0.065918f, -0.107414f, 0.712471f, 0.468437f, -0.073494f, -0.103821f, 0.674011f, 0.510559f, -0.080750f,
    -0.099220f, 0.634361f, 0.552406f, -0.087547f, -0.093750f, 0.593750f, 0.593750f, -0.093750f, -0.087547f, 0.552406f, 0.634361f, -0.099220f,
    -0.080750f, 0.510559f, 0.674011f, -0.103821f, -0.073494f, 0.468437f, 0.712471f, -0.107414f, -0.065918f, 0.426270f, 0.749512f, -0.109863f,
    -0.058159f, 0.384285f, 0.784904f, -0.111031f, -0.050354f, 0.342712f, 0.818420f, -0.110779f, -0.042641f, 0.301781f, 0.849831f, -0.108971f,
    -0.035156f, 0.261719f, 0.878906f, -0.105469f, -0.028038f, 0.222755f, 0.905418f, -0.100136f, -0.021423f, 0.185120f, 0.929138f, -0.092834f,
    -0.015450f, 0.149040f, 0.949837f, -0.083427f, -0.010254f, 0.114746f, 0.967285f, -0.071777f, -0.005974f, 0.082466f, 0.981255f, -0.057747f,
    -0.002747f, 0.052429f, 0.991516f, -0.041199f, -0.000710f, 0.024864f, 0.997841f, -0.021996f };

#endif

__kernel void warpAffine(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        int tmp = (dx << AB_BITS);
        int X0 = rint(M[0] * tmp) + rint(fma(M[1], (CT)dy, M[2]) * AB_SCALE) + ROUND_DELTA;
        int Y0 = rint(M[3] * tmp) + rint(fma(M[4], (CT)dy, M[5]) * AB_SCALE) + ROUND_DELTA;

        X0 = X0 >> (AB_BITS - INTER_BITS);
        Y0 = Y0 >> (AB_BITS - INTER_BITS);

        int sx = (short)(X0 >> INTER_BITS) - 1, sy = (short)(Y0 >> INTER_BITS) - 1;
        int ay = (short)(Y0 & (INTER_TAB_SIZE - 1)), ax = (short)(X0 & (INTER_TAB_SIZE - 1));

#ifdef AMD_DEVICE
        WT v[16];
        #pragma unroll
        for (int y = 0; y < 4; y++)
        {
            if (sy+y >= 0 && sy+y < src_rows)
            {
                #pragma unroll
                for (int x = 0; x < 4; x++)
                    v[mad24(y, 4, x)] = sx+x >= 0 && sx+x < src_cols ?
                        CONVERT_TO_WT(loadpix(srcptr + mad24(sy+y, src_step, mad24(sx+x, pixsize, src_offset)))) : scalar;
            }
            else
            {
                #pragma unroll
                for (int x = 0; x < 4; x++)
                    v[mad24(y, 4, x)] = scalar;
            }
        }

        float tab1y[4], tab1x[4];

        float ayy = INTER_SCALE * ay;
        float axx = INTER_SCALE * ax;
        interpolateCubic(ayy, tab1y);
        interpolateCubic(axx, tab1x);

        int dst_index = mad24(dy, dst_step, mad24(dx, pixsize, dst_offset));

        WT sum = (WT)(0);
#if SRC_DEPTH <= 4
        int itab[16];

        #pragma unroll
        for (int i = 0; i < 16; i++)
            itab[i] = rint(tab1y[(i>>2)] * tab1x[(i&3)] * INTER_REMAP_COEF_SCALE);

        #pragma unroll
        for (int i = 0; i < 16; i++)
            sum = mad24(v[i], itab[i], sum);
        storepix(CONVERT_TO_T( (sum + (1 << (INTER_REMAP_COEF_BITS-1))) >> INTER_REMAP_COEF_BITS ), dstptr + dst_index);
#else
        #pragma unroll
        for (int i = 0; i < 16; i++)
            sum = fma(v[i], tab1y[(i>>2)] * tab1x[(i&3)], sum);
        storepix(CONVERT_TO_T( sum ), dstptr + dst_index);
#endif
#else // INTEL_DEVICE
        __constant float * coeffs_y = coeffs + (ay << 2), * coeffs_x = coeffs + (ax << 2);

        int src_index0 = mad24(sy, src_step, mad24(sx, pixsize, src_offset)), src_index;
        int dst_index = mad24(dy, dst_step, mad24(dx, pixsize, dst_offset));

        WT sum = (WT)(0), xsum;
        #pragma unroll
        for (int y = 0; y < 4; y++)
        {
            src_index = mad24(y, src_step, src_index0);
            if (sy + y >= 0 && sy + y < src_rows)
            {
                xsum = (WT)(0);
                if (sx >= 0 && sx + 4 < src_cols)
                {
#if SRC_DEPTH == 0 && CN == 1
                    uchar4 value = vload4(0, srcptr + src_index);
                    xsum = dot(convert_float4(value), (float4)(coeffs_x[0], coeffs_x[1], coeffs_x[2], coeffs_x[3]));
#else
                    #pragma unroll
                    for (int x = 0; x < 4; x++)
                        xsum = fma(CONVERT_TO_WT(loadpix(srcptr + mad24(x, pixsize, src_index))), coeffs_x[x], xsum);
#endif
                }
                else
                {
                    #pragma unroll
                    for (int x = 0; x < 4; x++)
                        xsum = fma(sx + x >= 0 && sx + x < src_cols ?
                                   CONVERT_TO_WT(loadpix(srcptr + mad24(x, pixsize, src_index))) : scalar, coeffs_x[x], xsum);
                }
                sum = fma(xsum, coeffs_y[y], sum);
            }
            else
                sum = fma(scalar, coeffs_y[y], sum);
        }

        storepix(CONVERT_TO_T(sum), dstptr + dst_index);
#endif
    }
}

#endif
