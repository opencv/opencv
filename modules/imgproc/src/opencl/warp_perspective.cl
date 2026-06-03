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

#define CV_64F 6
#if defined SRC_DEPTH && SRC_DEPTH == CV_64F
#define WT1 double
#else
#define WT1 float
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

__kernel void warpPerspective(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float W  = fma(M[6], (CT)dx, fma(M[7], (CT)dy, M[8]));
        float X0 = fma(M[0], (CT)dx, fma(M[1], (CT)dy, M[2])) / W;
        float Y0 = fma(M[3], (CT)dx, fma(M[4], (CT)dy, M[5])) / W;

        int sx = convert_int_sat(rint(X0));
        int sy = convert_int_sat(rint(Y0));

        T v0 = scalar;
        if (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows)
        {
            v0 = loadpix(srcptr + mad24(sy, src_step, mad24(sx, pixsize, src_offset)));
        }

        int dst_index = mad24(dy, dst_step, mad24(dx, pixsize, dst_offset));
        storepix(v0, dstptr + dst_index);
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
        float W = M[6] * dx + M[7] * dy + M[8];
        float X0 = (M[0] * dx + M[1] * dy + M[2]) / W;
        float Y0 = (M[3] * dx + M[4] * dy + M[5]) / W;

        int sx = convert_short_rtn(X0);
        int sy = convert_short_rtn(Y0);
        float ay = Y0 - (CT)sy;
        float ax = X0 - (CT)sx;

        WT v0 = (sx >= 0 && sx < src_cols && sy >= 0 && sy < src_rows) ?
            CONVERT_TO_WT(loadpix(srcptr + mad24(sy, src_step, src_offset + sx * pixsize))) : scalar;
        WT v1 = (sx+1 >= 0 && sx+1 < src_cols && sy >= 0 && sy < src_rows) ?
            CONVERT_TO_WT(loadpix(srcptr + mad24(sy, src_step, src_offset + (sx+1) * pixsize))) : scalar;
        WT v2 = (sx >= 0 && sx < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            CONVERT_TO_WT(loadpix(srcptr + mad24(sy+1, src_step, src_offset + sx * pixsize))) : scalar;
        WT v3 = (sx+1 >= 0 && sx+1 < src_cols && sy+1 >= 0 && sy+1 < src_rows) ?
            CONVERT_TO_WT(loadpix(srcptr + mad24(sy+1, src_step, src_offset + (sx+1) * pixsize))) : scalar;

        int dst_index = mad24(dy, dst_step, dst_offset + dx * pixsize);

        v0 = fma(v1 - v0, ax, v0);
        v2 = fma(v3 - v2, ax, v2);
        v0 = fma(v2 - v0, ay, v0);
        storepix(CONVERT_TO_T(v0), dstptr + dst_index);
    }
}

#elif defined INTER_CUBIC

inline void interpolateCubic( float x, WT1* coeffs )
{
    const float A = -0.75f;

    coeffs[0] = (WT1)fma(fma(fma(A, (x + 1.f), - 5.0f*A), (x + 1.f), 8.0f*A), x + 1.f, - 4.0f*A);
    coeffs[1] = (WT1)fma(fma(A + 2.f, x, - (A + 3.f)), x*x, 1.f);
    coeffs[2] = (WT1)fma(fma(A + 2.f, 1.f - x, - (A + 3.f)), (1.f - x)*(1.f - x), 1.f);
    coeffs[3] = (WT1)(1. - coeffs[0] - coeffs[1] - coeffs[2]);
}

__kernel void warpPerspective(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __constant CT * M, ST scalar_)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float W = (float)fma(M[7], (CT)dy, fma(M[6], (CT)dx, M[8]));
        float X0 = (float)fma(M[1], (CT)dy, fma(M[0], (CT)dx, M[2])) / W;
        float Y0 = (float)fma(M[4], (CT)dy, fma(M[3], (CT)dx, M[5])) / W;

        int sx = convert_short_rtn(X0);
        int sy = convert_short_rtn(Y0);
        float ax = X0 - (float)sx;
        float ay = Y0 - (float)sy;

        sx--;
        sy--;

        WT1 taby[4], tabx[4];
        interpolateCubic(ay, taby);
        interpolateCubic(ax, tabx);

        WT sum = (WT)(0);

        if (0 <= sy && sy + 4 <= src_rows &&
            0 <= sx && sx + 4 <= src_cols) {
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                int row_offset = mad24(sy+y, src_step, src_offset);
                WT v0 = CONVERT_TO_WT(loadpix(srcptr + mad24(sx, pixsize, row_offset)));
                WT v1 = CONVERT_TO_WT(loadpix(srcptr + mad24(sx + 1, pixsize, row_offset)));
                WT v2 = CONVERT_TO_WT(loadpix(srcptr + mad24(sx + 2, pixsize, row_offset)));
                WT v3 = CONVERT_TO_WT(loadpix(srcptr + mad24(sx + 3, pixsize, row_offset)));
                WT wsum = (WT)(0);
                wsum = fma(v0, tabx[0], wsum);
                wsum = fma(v1, tabx[1], wsum);
                wsum = fma(v2, tabx[2], wsum);
                wsum = fma(v3, tabx[3], wsum);
                sum = fma(wsum, taby[y], sum);
            }
        }
        else {
            #pragma unroll
            for (int y = 0; y < 4; y++) {
                if (sy+y >= 0 && sy+y < src_rows) {
                    int row_offset = mad24(sy+y, src_step, src_offset);
                    #pragma unroll
                    for (int x = 0; x < 4; x++) {
                        WT v = sx+x >= 0 && sx+x < src_cols ?
                            CONVERT_TO_WT(loadpix(srcptr + mad24(sx + x, pixsize, row_offset))) : scalar;
                        sum = fma(v, taby[y] * tabx[x], sum);
                    }
                }
                else {
                    sum = fma(scalar, taby[y], sum);
                }
            }
        }

        int dst_index = mad24(dy, dst_step, mad24(dx, pixsize, dst_offset));
        storepix(CONVERT_TO_T( sum ), dstptr + dst_index);
    }
}

#endif
