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
//    Shengen Yan,yanshengen@gmail.com
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

__kernel void preCornerDetect(__global const uchar * Dxptr, int dx_step, int dx_offset,
                              __global const uchar * Dyptr, int dy_step, int dy_offset,
                              __global const uchar * D2xptr, int d2x_step, int d2x_offset,
                              __global const uchar * D2yptr, int d2y_step, int d2y_offset,
                              __global const uchar * Dxyptr, int dxy_step, int dxy_offset,
                              __global uchar * dstptr, int dst_step, int dst_offset,
                              int dst_rows, int dst_cols, float factor)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int dx_index = mad24(dx_step, y, (int)sizeof(float) * x + dx_offset);
        int dy_index = mad24(dy_step, y, (int)sizeof(float) * x + dy_offset);
        int d2x_index = mad24(d2x_step, y, (int)sizeof(float) * x + d2x_offset);
        int d2y_index = mad24(d2y_step, y, (int)sizeof(float) * x + d2y_offset);
        int dxy_index = mad24(dxy_step, y, (int)sizeof(float) * x + dxy_offset);
        int dst_index = mad24(dst_step, y, (int)sizeof(float) * x + dst_offset);

        float dx = *(__global const float *)(Dxptr + dx_index);
        float dy = *(__global const float *)(Dyptr + dy_index);
        float d2x = *(__global const float *)(D2xptr + d2x_index);
        float d2y = *(__global const float *)(D2yptr + d2y_index);
        float dxy = *(__global const float *)(Dxyptr + dxy_index);
        __global float * dst = (__global float *)(dstptr + dst_index);

        dst[0] = factor * (dx*dx*d2y + dy*dy*d2x - 2*dx*dy*dxy);
    }
}
