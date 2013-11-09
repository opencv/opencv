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

#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

// threshold type:
// enum { THRESH_BINARY=0, THRESH_BINARY_INV=1, THRESH_TRUNC=2, THRESH_TOZERO=3,
//       THRESH_TOZERO_INV=4, THRESH_MASK=7, THRESH_OTSU=8 };

__kernel void threshold_C1_D0(__global const uchar * restrict src, __global uchar *dst,
                              int src_offset, int src_step,
                              int dst_offset, int dst_rows, int dst_cols, int dst_step,
                              uchar thresh, uchar max_val, int thresh_type
                              )
{
    int gx = get_global_id(0);
    const int gy = get_global_id(1);

    int offset = (dst_offset & 15);
    src_offset -= offset;

    int dstart = (gx << 4) - offset;
    if(dstart < dst_cols && gy < dst_rows)
    {
        uchar16 sdata = vload16(gx, src+src_offset+gy*src_step);
        uchar16 ddata;
        uchar16 zero = 0;
        switch (thresh_type)
        {
            case 0:
                ddata = ((sdata > thresh) ) ? (uchar16)(max_val) : (uchar16)(0);
                break;
            case 1:
                ddata = ((sdata > thresh)) ? zero  : (uchar16)(max_val);
                break;
            case 2:
                ddata = ((sdata > thresh)) ? (uchar16)(thresh) : sdata;
                break;
            case 3:
                ddata = ((sdata > thresh)) ? sdata : zero;
                break;
            case 4:
                ddata = ((sdata > thresh)) ? zero : sdata;
                break;
            default:
                ddata = sdata;
        }
        int16 dpos = (int16)(dstart, dstart+1, dstart+2, dstart+3, dstart+4, dstart+5, dstart+6, dstart+7, dstart+8,
                             dstart+9, dstart+10, dstart+11, dstart+12, dstart+13, dstart+14, dstart+15);
        uchar16 dVal = *(__global uchar16*)(dst+dst_offset+gy*dst_step+dstart);
        int16 con = dpos >= 0 && dpos < dst_cols;
        ddata = convert_uchar16(con != 0) ? ddata : dVal;
        if(dstart < dst_cols)
        {
            *(__global uchar16*)(dst+dst_offset+gy*dst_step+dstart) = ddata;
        }
    }
}


__kernel void threshold_C1_D5(__global const float * restrict src, __global float *dst,
                              int src_offset, int src_step,
                              int dst_offset, int dst_rows, int dst_cols, int dst_step,
                              float thresh, float max_val, int thresh_type
                              )
{
    const int gx = get_global_id(0);
    const int gy = get_global_id(1);

    int offset = (dst_offset & 3);
    src_offset -= offset;

    int dstart = (gx << 2) - offset;
    if(dstart < dst_cols && gy < dst_rows)
    {
        float4 sdata = vload4(gx, src+src_offset+gy*src_step);
        float4 ddata;
        float4 zero = 0;
        switch (thresh_type)
        {
            case 0:
                ddata = sdata > thresh ? (float4)(max_val) : (float4)(0.f);
                break;
            case 1:
                ddata = sdata > thresh ? zero : (float4)max_val;
                break;
            case 2:
                ddata = sdata > thresh ? (float4)thresh : sdata;
                break;
            case 3:
                ddata = sdata > thresh ? sdata : (float4)(0.f);
                break;
            case 4:
                ddata = sdata > thresh ? (float4)(0.f) : sdata;
                break;
            default:
                ddata = sdata;
        }
        int4 dpos = (int4)(dstart, dstart+1, dstart+2, dstart+3);
        float4 dVal = *(__global float4*)(dst+dst_offset+gy*dst_step+dstart);
        int4 con = dpos >= 0 && dpos < dst_cols;
        ddata = convert_float4(con) != (float4)(0) ? ddata : dVal;
        if(dstart < dst_cols)
        {
            *(__global float4*)(dst+dst_offset+gy*dst_step+dstart) = ddata;
        }
    }
}
