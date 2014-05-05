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
// Copyright (C) 2010-2012, Multicoreware, Inc., all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Peng Xiao, pengxiao@multicorewareinc.com
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
// In no event shall the uintel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business uinterruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

inline float2 cmulf(float2 a, float2 b)
{
    return (float2)(mad(a.x, b.x, - a.y * b.y), mad(a.x, b.y, a.y * b.x));
}

inline float2 conjf(float2 a)
{
    return (float2)(a.x, - a.y);
}

__kernel void mulAndScaleSpectrums(__global const uchar * src1ptr, int src1_step, int src1_offset,
                                   __global const uchar * src2ptr, int src2_step, int src2_offset,
                                   __global uchar * dstptr, int dst_step, int dst_offset,
                                   int dst_rows, int dst_cols)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src1_index = mad24(y, src1_step, mad24(x, (int)sizeof(float2), src1_offset));
        int src2_index = mad24(y, src2_step, mad24(x, (int)sizeof(float2), src2_offset));
        int dst_index = mad24(y, dst_step, mad24(x, (int)sizeof(float2), dst_offset));

        float2 src0 = *(__global const float2 *)(src1ptr + src1_index);
        float2 src1 = *(__global const float2 *)(src2ptr + src2_index);
        __global float2 * dst = (__global float2 *)(dstptr + dst_index);

#ifdef CONJ
        float2 v = cmulf(src0, conjf(src1));
#else
        float2 v = cmulf(src0, src1);
#endif
        dst[0] = v;
    }
}
