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
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

typedef struct UMat2D
{
    size_t offset;
    size_t step;
    int rows;
    int cols;
}
UMat2D;

#if defined BINOP_ADD
#define BINOP(x,y) sat_add(x,y)
#elif defined BINOP_ADD_FLT
#define BINOP(x,y) x + y
#elif defined BINOP_SUB
#define BINOP(x,y) sat_sub(x, y)
#elif defined BINOP_SUB_FLT
#define BINOP(x,y) x - y
#elif defined BINOP_ABSDIFF
#define BINOP(x,y) abs_diff(x, y)
#elif defined BINOP_AND
#define BINOP(x,y) x & y
#elif defined BINOP_OR
#define BINOP(x,y) x | y
#elif defined BINOP_XOR
#define BINOP(x,y) x ^ y
#elif defined BINOP_NOT
#define BINOP(x,y) ~x
#elif defined BINOP_MIN
#define BINOP(x,y) min(x, y)
#elif defined BINOP_MAX
#define BINOP(x,y) max(x, y)
#elif defined BINOP_MUL
#define BINOP(x,y) x * y
#elif defined BINOP_DIV
#define BINOP(x,y) x / y
#else
#error "unknown op type"
#endif

#if defined SAMETYPE_MODE
#define srcT
#define convertToWT(x) (x)
#define convertToDstT(x) (x)
#define srcT1 dstT
#define srcT2 dstT
#endif

__kernel void binop(__global const uchar* srcptr1, int srcoffset1, int srcstep1, int cols1, int rows1,
                    __global const uchar* srcptr2, int srcoffset2, int srcstep2, int cols2, int rows2,
                    __global uchar* dstptr, int dstoffset, int dststep, int cols, int rows)
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, srcstep1, x*sizeof(srcT1) + srcoffset1);
        int src2_index = mad24(y, srcstep2, x*sizeof(srcT2) + srcoffset2);
        int dst_index  = mad24(y, dststep, x*sizeof(dstT) + dstoffset);
        *(dstT*)(dstptr + dst_index) = convertToDstT(BINOP(convertToWT(*(srcT1*)(srcptr1 + src1_index)),
                                                           convertToWT(*(srcT2*)(srcptr2 + src2_index))));
    }
}

