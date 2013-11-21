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


// resize kernel
// Currently, CV_8UC1  CV_8UC4  CV_32FC1 and CV_32FC4are supported.
// We shall support other types later if necessary.

#if defined DOUBLE_SUPPORT
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#define F double
#else
#define F float
#endif

#define INTER_RESIZE_COEF_BITS 11
#define INTER_RESIZE_COEF_SCALE (1 << INTER_RESIZE_COEF_BITS)
#define CAST_BITS (INTER_RESIZE_COEF_BITS << 1)
#define CAST_SCALE (1.0f/(1<<CAST_BITS))
#define INC(x,l) min(x+1,l-1)

#define PIXSIZE ((int)sizeof(PIXTYPE))
#define noconvert(x) (x)

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
    __global const PIXTYPE* src = (__global const PIXTYPE*)(srcptr + mad24(y, srcstep, srcoffset + x*PIXSIZE));

#if depth == 0
    u = u * INTER_RESIZE_COEF_SCALE;
    v = v * INTER_RESIZE_COEF_SCALE;

    int U = rint(u);
    int V = rint(v);
    int U1 = rint(INTER_RESIZE_COEF_SCALE - u);
    int V1 = rint(INTER_RESIZE_COEF_SCALE - v);

    WORKTYPE data0 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data1 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y, srcstep, srcoffset + x_*PIXSIZE)));
    WORKTYPE data2 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y_, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data3 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y_, srcstep, srcoffset + x_*PIXSIZE)));
    WORKTYPE val = mul24((WORKTYPE)mul24(U1, V1), data0) + mul24((WORKTYPE)mul24(U, V1), data1) +
               mul24((WORKTYPE)mul24(U1, V), data2) + mul24((WORKTYPE)mul24(U, V), data3);

    PIXTYPE uval = convertToDT((val + (1<<(CAST_BITS-1)))>>CAST_BITS);
#else
    float u1 = 1.f-u;
    float v1 = 1.f-v;
    WORKTYPE data0 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data1 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y, srcstep, srcoffset + x_*PIXSIZE)));
    WORKTYPE data2 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y_, srcstep, srcoffset + x*PIXSIZE)));
    WORKTYPE data3 = convertToWT(*(__global const PIXTYPE*)(srcptr + mad24(y_, srcstep, srcoffset + x_*PIXSIZE)));
    PIXTYPE uval = u1 * v1 * s_data1 + u * v1 * s_data2 + u1 * v *s_data3 + u * v *s_data4;
#endif

    if(dx < dstcols && dy < dstrows)
    {
        __global PIXTYPE* dst = (__global PIXTYPE*)(dstptr + mad24(dy, dststep, dstoffset + dx*PIXSIZE));
        dst[0] = uval;
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
        F s1 = dx*ifx;
        F s2 = dy*ify;
        int sx = min(convert_int_rtz(s1), srccols-1);
        int sy = min(convert_int_rtz(s2), srcrows-1);

        __global PIXTYPE* dst = (__global PIXTYPE*)(dstptr + mad24(dy, dststep, dstoffset + dx*PIXSIZE));
        __global const PIXTYPE* src = (__global const PIXTYPE*)(srcptr + mad24(sy, srcstep, srcoffset + sx*PIXSIZE));

        dst[0] = src[0];
    }
}

#endif
