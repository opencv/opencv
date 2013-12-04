////////////////////////////////////////////////////////////////////////////////////////
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

#ifdef DOUBLE_SUPPORT
#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif
#endif

#define noconvert

#if defined OP_SUM || defined OP_SUM_ABS || defined OP_SUM_SQR
#if OP_SUM
#define FUNC(a, b) a += b
#elif OP_SUM_ABS
#define FUNC(a, b) a += b >= (dstT)(0) ? b : -b
#elif OP_SUM_SQR
#define FUNC(a, b) a += b * b
#endif
#define DEFINE_ACCUMULATOR \
    dstT accumulator = (dstT)(0)
#define REDUCE_GLOBAL \
    dstT temp = convertToDT(src[0]); \
    FUNC(accumulator, temp)
#define REDUCE_LOCAL_1 \
    localmem[lid - WGS2_ALIGNED] += accumulator
#define REDUCE_LOCAL_2 \
    localmem[lid] += localmem[lid2]

#elif defined OP_COUNT_NON_ZERO
#define dstT int
#define DEFINE_ACCUMULATOR \
    dstT accumulator = (dstT)(0); \
    srcT zero = (srcT)(0), one = (srcT)(1)
#define REDUCE_GLOBAL \
    accumulator += src[0] == zero ? zero : one
#define REDUCE_LOCAL_1 \
    localmem[lid - WGS2_ALIGNED] += accumulator
#define REDUCE_LOCAL_2 \
    localmem[lid] += localmem[lid2]

#else
#error "No operation"

#endif

__kernel void reduce(__global const uchar * srcptr, int step, int offset, int cols,
                     int total, int groupnum, __global uchar * dstptr)
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int  id = get_global_id(0);

    __local dstT localmem[WGS2_ALIGNED];
    DEFINE_ACCUMULATOR;

    for (int grain = groupnum * WGS; id < total; id += grain)
    {
        int src_index = mad24(id / cols, step, offset + (id % cols) * (int)sizeof(srcT));
        __global const srcT * src = (__global const srcT *)(srcptr + src_index);
        REDUCE_GLOBAL;
    }

    if (lid < WGS2_ALIGNED)
        localmem[lid] = accumulator;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED)
        REDUCE_LOCAL_1;
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int lsize = WGS2_ALIGNED >> 1; lsize > 0; lsize >>= 1)
    {
        if (lid < lsize)
        {
           int lid2 = lsize + lid;
           REDUCE_LOCAL_2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0)
    {
        __global dstT * dst = (__global dstT *)(dstptr + (int)sizeof(dstT) * gid);
        dst[0] = localmem[0];
    }
}
