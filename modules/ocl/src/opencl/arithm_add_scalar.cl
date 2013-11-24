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
//    Jia Haipeng, jiahaipeng95@gmail.com
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

#if defined (FUNC_ADD)
#define EXPRESSION dst[dst_index] = convertToT(convertToWT(src1[src1_index]) + scalar);
#endif

#if defined (FUNC_SUB)
#define EXPRESSION dst[dst_index] = convertToT(convertToWT(src1[src1_index]) - scalar);
#endif

#if defined (FUNC_MUL)
#define EXPRESSION dst[dst_index] = convertToT(convertToWT(src1[src1_index]) * scalar);
#endif

#if defined (FUNC_DIV)
#define EXPRESSION T zero = (T)(0); \
    dst[dst_index] = src1[src1_index] == zero ? zero : convertToT(scalar / convertToWT(src1[src1_index]));
#endif

#if defined (FUNC_ABS)
#define EXPRESSION \
    T value = src1[src1_index] > (T)(0) ? src1[src1_index] : -src1[src1_index]; \
    dst[dst_index] = value;
#endif

#if defined (FUNC_ABS_DIFF)
#define EXPRESSION WT value = convertToWT(src1[src1_index]) - scalar; \
    value = value > (WT)(0) ? value : -value; \
    dst[dst_index] = convertToT(value);
#endif

///////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Add with scalar /////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////

__kernel void arithm_binary_op_scalar (__global T *src1, int src1_step, int src1_offset,
                                 WT scalar,
                                 __global T *dst,  int dst_step,  int dst_offset,
                                 int cols, int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, src1_step, x + src1_offset);
        int dst_index = mad24(y, dst_step, x + dst_offset);

        EXPRESSION

    }
}
