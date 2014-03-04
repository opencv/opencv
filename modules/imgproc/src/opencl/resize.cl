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

#if defined DOUBLE_SUPPORT
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
#define INC(x,l) min(x+1,l-1)


#define noconvert(x) (x)

#if cn != 3
#define loadpix(addr)  *(__global const PIXTYPE*)(addr)
#define storepix(val, addr)  *(__global PIXTYPE*)(addr) = val
#define PIXSIZE ((int)sizeof(PIXTYPE))
#else
#define loadpix(addr)  vload3(0, (__global const PIXTYPE1*)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global PIXTYPE1*)(addr))
#define PIXSIZE ((int)sizeof(PIXTYPE1)*3)
#endif

#if defined INTER_LINEAR

__kernel void resizeLN(__global const uchar* srcptr, int srcstep, int srcoffset,
                       int srcrows, int srccols,
                       __global uchar* dstptr, int dststep, int dstoffset,
                       int dstrows, int dstcols,
                       float ifx, float ify)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    float sx = ((dx+0.5f) * ifx - 0.5f), sy = ((dy+0.5f) * ify - 0.5f);
    int x = floor(sx), y = floor(sy);

    float u = sx - x, v = sy - y;

    if ( x<0 ) x=0,u=0;
    if ( x>=srccols ) x=srccols-1,u=0;
    if ( y<0 ) y=0,v=0;
    if ( y>=srcrows ) y=srcrows-1,v=0;

    int y_ = INC(y,srcrows);
    int x_ = INC(x,srccols);

#if depth <= 4

    u = u * INTER_RESIZE_COEF_SCALE;
    v = v * INTER_RESIZE_COEF_SCALE;

    int U = rint(u);
    int V = rint(v);
    int U1 = rint(INTER_RESIZE_COEF_SCALE - u);
    int V1 = rint(INTER_RESIZE_COEF_SCALE - v);

    WORKTYPE data0 = convertToWT(loadpix(srcptr + mad24(y, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data1 = convertToWT(loadpix(srcptr + mad24(y, srcstep, srcoffset + x_*PIXSIZE)));
    WORKTYPE data2 = convertToWT(loadpix(srcptr + mad24(y_, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data3 = convertToWT(loadpix(srcptr + mad24(y_, srcstep, srcoffset + x_*PIXSIZE)));

    WORKTYPE val = mul24((WORKTYPE)mul24(U1, V1), data0) + mul24((WORKTYPE)mul24(U, V1), data1) +
               mul24((WORKTYPE)mul24(U1, V), data2) + mul24((WORKTYPE)mul24(U, V), data3);

    PIXTYPE uval = convertToDT((val + (1<<(CAST_BITS-1)))>>CAST_BITS);

#else
    float u1 = 1.f - u;
    float v1 = 1.f - v;
    WORKTYPE data0 = convertToWT(loadpix(srcptr + mad24(y, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data1 = convertToWT(loadpix(srcptr + mad24(y, srcstep, srcoffset + x_*PIXSIZE)));
    WORKTYPE data2 = convertToWT(loadpix(srcptr + mad24(y_, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data3 = convertToWT(loadpix(srcptr + mad24(y_, srcstep, srcoffset + x_*PIXSIZE)));

    PIXTYPE uval = u1 * v1 * data0 + u * v1 * data1 + u1 * v *data2 + u * v *data3;

#endif

    if(dx < dstcols && dy < dstrows)
    {
        storepix(uval, dstptr + mad24(dy, dststep, dstoffset + dx*PIXSIZE));
    }
}

#elif defined INTER_NEAREST

__kernel void resizeNN(__global const uchar* srcptr, int srcstep, int srcoffset,
                       int srcrows, int srccols,
                       __global uchar* dstptr, int dststep, int dstoffset,
                       int dstrows, int dstcols,
                       float ifx, float ify)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if( dx < dstcols && dy < dstrows )
    {
        float s1 = dx*ifx;
        float s2 = dy*ify;
        int sx = min(convert_int_rtz(s1), srccols-1);
        int sy = min(convert_int_rtz(s2), srcrows-1);

        storepix(loadpix(srcptr + mad24(sy, srcstep, srcoffset + sx*PIXSIZE)),
                 dstptr + mad24(dy, dststep, dstoffset + dx*PIXSIZE));
    }
}

#elif defined INTER_AREA

#ifdef INTER_AREA_FAST

__kernel void resizeAREA_FAST(__global const uchar * src, int src_step, int src_offset, int src_rows, int src_cols,
                              __global uchar * dst, int dst_step, int dst_offset, int dst_rows, int dst_cols,
                              __global const int * dmap_tab, __global const int * smap_tab)
{
    int dx = get_global_id(0);
    int dy = get_global_id(1);

    if (dx < dst_cols && dy < dst_rows)
    {
        int dst_index = mad24(dy, dst_step, dst_offset);

        __global const int * xmap_tab = dmap_tab;
        __global const int * ymap_tab = dmap_tab + dst_cols;
        __global const int * sxmap_tab = smap_tab;
        __global const int * symap_tab = smap_tab + XSCALE * dst_cols;

        int sx = xmap_tab[dx], sy = ymap_tab[dy];
        WTV sum = (WTV)(0);

        #pragma unroll
        for (int y = 0; y < YSCALE; ++y)
        {
            int src_index = mad24(symap_tab[y + sy], src_step, src_offset);
            #pragma unroll
            for (int x = 0; x < XSCALE; ++x)
                sum += convertToWTV(loadpix(src + src_index + sxmap_tab[sx + x]*PIXSIZE));
        }

        storepix(convertToPIXTYPE(convertToWT2V(sum) * (WT2V)(SCALE)), dst + dst_index + dx*PIXSIZE);
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
                buf += convertToWTV(loadpix(src + src_index + sx*PIXSIZE)) * alpha;
            }
            sum += buf * beta;
        }

        storepix(convertToPIXTYPE(sum), dst + dst_index + dx*PIXSIZE);
    }
}

#endif

#endif
