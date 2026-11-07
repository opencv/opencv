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
//	  Niko Li, newlife20080214@gmail.com
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

#define INC(x,l) min(x+1,l-1)

#define noconvert

#if CN != 3
#define loadpix(addr)  *(__global const T *)(addr)
#define storepix(val, addr)  *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr)  vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE (int)sizeof(T1)*CN
#endif

#if defined USE_SAMPLER

#if CN == 1
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z).x
#define INTERMEDIATE_TYPE  float
#elif CN == 2
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z).xy
#define INTERMEDIATE_TYPE  float2
#elif CN == 3
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z).xyz
#define INTERMEDIATE_TYPE  float3
#elif CN == 4
#define READ_IMAGE(X,Y,Z)  read_imagef(X,Y,Z)
#define INTERMEDIATE_TYPE  float4
#endif

#define __CAT(x, y) x##y
#define CAT(x, y) __CAT(x, y)
//#define INTERMEDIATE_TYPE CAT(float, CN)
#define float1 float

#if SRC_DEPTH == 0
#define RESULT_SCALE    255.0f
#elif SRC_DEPTH == 1
#define RESULT_SCALE    127.0f
#elif SRC_DEPTH == 2
#define RESULT_SCALE    65535.0f
#elif SRC_DEPTH == 3
#define RESULT_SCALE    32767.0f
#else
#define RESULT_SCALE    1.0f
#endif

__kernel void resizeSampler(__read_only image2d_t srcImage,
                            __global uchar* dstptr, int dststep, int dstoffset,
                            int dstrows, int dstcols,
                            float ifx, float ify)
{
    const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE |
                              CLK_ADDRESS_CLAMP_TO_EDGE |
                              CLK_FILTER_LINEAR;

    int dx = get_global_id(0);
    int dy = get_global_id(1);

    float sx = ((dx+0.5f) * ifx), sy = ((dy+0.5f) * ify);

    INTERMEDIATE_TYPE intermediate = READ_IMAGE(srcImage, sampler, (float2)(sx, sy));

#if SRC_DEPTH <= 4
    T uval = CONVERT_TO_DT(round(intermediate * RESULT_SCALE));
#else
    T uval = CONVERT_TO_DT(intermediate * RESULT_SCALE);
#endif

    if(dx < dstcols && dy < dstrows)
    {
        storepix(uval, dstptr + mad24(dy, dststep, dstoffset + dx*TSIZE));
    }
}

#elif defined INTER_LINEAR_INTEGER

__kernel void resizeLN(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                       __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                       __global const uchar * buffer)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        __global const int * xofs = (__global const int *)(buffer), * yofs = xofs + dst_cols;
        __global const short * ialpha = (__global const short *)(yofs + dst_rows);
        __global const short * ibeta = ialpha + ((dst_cols + dy) << 1);
        ialpha += dx << 1;

        int sx0 = xofs[dx], sy0 = clamp(yofs[dy], 0, src_rows - 1),
        sy1 = clamp(yofs[dy] + 1, 0, src_rows - 1);
        short a0 = ialpha[0], a1 = ialpha[1];
        short b0 = ibeta[0], b1 = ibeta[1];

        int src_index0 = mad24(sy0, src_step, mad24(sx0, TSIZE, src_offset)),
        src_index1 = mad24(sy1, src_step, mad24(sx0, TSIZE, src_offset));
        WT data0 = CONVERT_TO_WT(loadpix(srcptr + src_index0));
        WT data1 = CONVERT_TO_WT(loadpix(srcptr + src_index0 + TSIZE));
        WT data2 = CONVERT_TO_WT(loadpix(srcptr + src_index1));
        WT data3 = CONVERT_TO_WT(loadpix(srcptr + src_index1 + TSIZE));

        WT val = ( (((data0 * a0 + data1 * a1) >> 4) * b0) >> 16) +
                 ( (((data2 * a0 + data3 * a1) >> 4) * b1) >> 16);

        storepix(CONVERT_TO_DT((val + 2) >> 2),
                 dstptr + mad24(dy, dst_step, mad24(dx, TSIZE, dst_offset)));
    }
}

#elif defined INTER_LINEAR

__kernel void resizeLN(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                       __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                       float ifx, float ify)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float sx = ((dx+0.5f) * ifx - 0.5f), sy = ((dy+0.5f) * ify - 0.5f);
        int x = floor(sx), y = floor(sy);

        float u = sx - x, v = sy - y;

        if ( x<0 ) x=0,u=0;
        if ( x>=src_cols ) x=src_cols-1,u=0;
        if ( y<0 ) y=0,v=0;
        if ( y>=src_rows ) y=src_rows-1,v=0;

        int y_ = INC(y, src_rows);
        int x_ = INC(x, src_cols);

#if SRC_DEPTH <= 1  // 8U/8S only, 16U+ cause integer overflows
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
        u = u * INTER_RESIZE_COEF_SCALE;
        v = v * INTER_RESIZE_COEF_SCALE;

        int U = rint(u);
        int V = rint(v);
        int U1 = rint(INTER_RESIZE_COEF_SCALE - u);
        int V1 = rint(INTER_RESIZE_COEF_SCALE - v);

        WT data0 = CONVERT_TO_WT(loadpix(srcptr + mad24(y, src_step, mad24(x, TSIZE, src_offset))));
        WT data1 = CONVERT_TO_WT(loadpix(srcptr + mad24(y, src_step, mad24(x_, TSIZE, src_offset))));
        WT data2 = CONVERT_TO_WT(loadpix(srcptr + mad24(y_, src_step, mad24(x, TSIZE, src_offset))));
        WT data3 = CONVERT_TO_WT(loadpix(srcptr + mad24(y_, src_step, mad24(x_, TSIZE, src_offset))));

        WT val = mul24((WT)mul24(U1, V1), data0) + mul24((WT)mul24(U, V1), data1) +
                   mul24((WT)mul24(U1, V), data2) + mul24((WT)mul24(U, V), data3);

        T uval = CONVERT_TO_DT((val + (1<<(CAST_BITS-1)))>>CAST_BITS);
#else
        float u1 = 1.f - u;
        float v1 = 1.f - v;
        WT data0 = CONVERT_TO_WT(loadpix(srcptr + mad24(y, src_step, mad24(x, TSIZE, src_offset))));
        WT data1 = CONVERT_TO_WT(loadpix(srcptr + mad24(y, src_step, mad24(x_, TSIZE, src_offset))));
        WT data2 = CONVERT_TO_WT(loadpix(srcptr + mad24(y_, src_step, mad24(x, TSIZE, src_offset))));
        WT data3 = CONVERT_TO_WT(loadpix(srcptr + mad24(y_, src_step, mad24(x_, TSIZE, src_offset))));

        T uval = CONVERT_TO_DT((u1 * v1) * data0 + (u * v1) * data1 + (u1 * v) * data2 + (u * v) * data3);
#endif
        storepix(uval, dstptr + mad24(dy, dst_step, mad24(dx, TSIZE, dst_offset)));
    }
}

#elif defined INTER_NEAREST

__kernel void resizeNN(__global const uchar * srcptr, int src_step, int src_offset, int src_rows, int src_cols,
                       __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                       float ifx, float ify)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        float s1 = dx * ifx;
        float s2 = dy * ify;
        int sx = min(convert_int_rtz(s1), src_cols - 1);
        int sy = min(convert_int_rtz(s2), src_rows - 1);

        storepix(loadpix(srcptr + mad24(sy, src_step, mad24(sx, TSIZE, src_offset))),
                 dstptr + mad24(dy, dst_step, mad24(dx, TSIZE, dst_offset)));
    }
}

#elif defined INTER_AREA

#ifdef INTER_AREA_FAST

__kernel void resizeAREA_FAST(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        int dst_index = mad24(dy, dst_step, dst_offset);

        int sx = XSCALE * dx;
        int sy = YSCALE * dy;
        WTV sum = (WTV)(0);

        #pragma unroll
        for (int py = 0; py < YSCALE; ++py)
        {
            int y = min(sy + py, src_rows - 1);
            int src_index = mad24(y, src_step, src_offset);
            #pragma unroll
            for (int px = 0; px < XSCALE; ++px)
            {
                int x = min(sx + px, src_cols - 1);
                sum += CONVERT_TO_WTV(loadpix(src + src_index + x*TSIZE));
            }
        }

        storepix(CONVERT_TO_T(CONVERT_TO_WT2V(sum) * (WT2V)(SCALE)), dst + mad24(dx, TSIZE, dst_index));
    }
}

#else

__kernel void resizeAREA(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                         __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                         float ifx, float ify, __global const int * ofs_tab,
                         __global const int * map_tab, __global const float * alpha_tab)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        int dst_index = mad24(dy, dst_step, dst_offset);

        __global const int * xmap_tab = map_tab;
        __global const int * ymap_tab = (__global const int *)(map_tab + (src_cols << 1));
        __global const float * xalpha_tab = alpha_tab;
        __global const float * yalpha_tab = (__global const float *)(alpha_tab + (src_cols << 1));
        __global const int * xofs_tab = ofs_tab;
        __global const int * yofs_tab = (__global const int *)(ofs_tab + dst_cols + 1);

        int xk0 = xofs_tab[dx], xk1 = xofs_tab[dx + 1];
        int yk0 = yofs_tab[dy], yk1 = yofs_tab[dy + 1];

        int sy0 = ymap_tab[yk0], sy1 = ymap_tab[yk1 - 1];
        int sx0 = xmap_tab[xk0], sx1 = xmap_tab[xk1 - 1];

        WTV sum = (WTV)(0), buf;
        int src_index = mad24(sy0, src_step, src_offset);

        for (int sy = sy0, yk = yk0; sy <= sy1; ++sy, src_index += src_step, ++yk)
        {
            WTV beta = (WTV)(yalpha_tab[yk]);
            buf = (WTV)(0);

            for (int sx = sx0, xk = xk0; sx <= sx1; ++sx, ++xk)
            {
                WTV alpha = (WTV)(xalpha_tab[xk]);
                buf += CONVERT_TO_WTV(loadpix(src + mad24(sx, TSIZE, src_index))) * alpha;
            }
            sum += buf * beta;
        }

    storepix(CONVERT_TO_T(sum), dst + mad24(dx, TSIZE, dst_index));
    }
}

#endif

#endif
