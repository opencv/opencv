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
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
// In no event shall the copyright holders or contributors be liable for any direct,
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

__kernel void inrange(__global const uchar * src1ptr, int src1_step, int src1_offset,
                      __global uchar * dstptr, int dst_step, int dst_offset, int dst_rows, int dst_cols,
#ifdef HAVE_SCALAR
                      __global const T * src2, __global const T * src3
#else
                      __global const uchar * src2ptr, int src2_step, int src2_offset,
                      __global const uchar * src3ptr, int src3_step, int src3_offset
#endif
                      )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < dst_cols && y < dst_rows)
    {
        int src1_index = mad24(y, src1_step, x*(int)sizeof(T)*cn + src1_offset);
        int dst_index = mad24(y, dst_step, x + dst_offset);
        __global const T * src1 = (__global const T *)(src1ptr + src1_index);
        __global uchar * dst = dstptr + dst_index;

#ifndef HAVE_SCALAR
        int src2_index = mad24(y, src2_step, x*(int)sizeof(T)*cn + src2_offset);
        int src3_index = mad24(y, src3_step, x*(int)sizeof(T)*cn + src3_offset);
        __global const T * src2 = (__global const T *)(src2ptr + src2_index);
        __global const T * src3 = (__global const T *)(src3ptr + src3_index);
#endif

        dst[0] = 255;

        #pragma unroll
        for (int c = 0; c < cn; ++c)
            if ( src2[c] > src1[c] || src3[c] < src1[c] )
            {
                dst[0] = 0;
                break;
            }
    }
}
