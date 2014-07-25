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

#if defined OP_NORM_INF_MASK

#ifdef DEPTH_0
#define MIN_VAL 0
#define MAX_VAL 255
#elif defined DEPTH_1
#define MIN_VAL -128
#define MAX_VAL 127
#elif defined DEPTH_2
#define MIN_VAL 0
#define MAX_VAL 65535
#elif defined DEPTH_3
#define MIN_VAL -32768
#define MAX_VAL 32767
#elif defined DEPTH_4
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#elif defined DEPTH_5
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#elif defined DEPTH_6
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

#define dstT srcT
#define dstT1 srcT1

#endif // min/max stuff

#define noconvert

#ifndef kercn
#define kercn 1
#endif

#ifdef HAVE_MASK_CONT
#define MASK_INDEX int mask_index = id + mask_offset;
#else
#define MASK_INDEX int mask_index = mad24(id / cols, mask_step, mask_offset + (id % cols))
#endif

#if cn != 3
#define loadpix(addr) *(__global const srcT *)(addr)
#define storepix(val, addr)  *(__global dstT *)(addr) = val
#if kercn == 1
#define srcTSIZE (int)sizeof(srcT)
#else
#define srcTSIZE (int)sizeof(srcT1)
#endif
#define dstTSIZE (int)sizeof(dstT)
#else
#define loadpix(addr) vload3(0, (__global const srcT1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global dstT1 *)(addr))
#define srcTSIZE ((int)sizeof(srcT1)*3)
#define dstTSIZE ((int)sizeof(dstT1)*3)
#endif

#if ddepth <= 4
#define SUM_ABS(a) convertFromU(abs(a))
#define SUM_ABS2(a, b) convertFromU(abs_diff(a, b))
#else
#define SUM_ABS(a) fabs(a)
#define SUM_ABS2(a, b) fabs(a - b)
#endif

#ifdef HAVE_MASK
#ifdef HAVE_SRC2
#define EXTRA_PARAMS , __global const uchar * mask, int mask_step, int mask_offset, __global const uchar * src2ptr, int src2_step, int src2_offset
#else
#define EXTRA_PARAMS , __global const uchar * mask, int mask_step, int mask_offset
#endif
#else
#ifdef HAVE_SRC2
#define EXTRA_PARAMS , __global const uchar * src2ptr, int src2_step, int src2_offset
#else
#define EXTRA_PARAMS
#endif
#endif

// accumulative reduction stuff
#if defined OP_SUM || defined OP_SUM_ABS || defined OP_SUM_SQR || defined OP_DOT

#ifdef OP_DOT
#if ddepth <= 4
#define FUNC(a, b, c) a = mad24(b, c, a)
#else
#define FUNC(a, b, c) a = mad(b, c, a)
#endif

#elif defined OP_SUM
#define FUNC(a, b) a += b

#elif defined OP_SUM_ABS
#define FUNC(a, b) a += SUM_ABS(b)

#elif defined OP_SUM_SQR
#if ddepth <= 4
#define FUNC(a, b) a = mad24(b, b, a)
#else
#define FUNC(a, b) a = mad(b, b, a)
#endif
#endif

#ifdef OP_CALC2
#define DECLARE_LOCAL_MEM \
    __local dstT localmem[WGS2_ALIGNED], localmem2[WGS2_ALIGNED]
#define DEFINE_ACCUMULATOR \
    dstT accumulator = (dstT)(0), accumulator2 = (dstT)(0)
#else
#define DECLARE_LOCAL_MEM \
    __local dstT localmem[WGS2_ALIGNED]
#define DEFINE_ACCUMULATOR \
    dstT accumulator = (dstT)(0)
#endif

#ifdef HAVE_SRC2
#ifdef OP_CALC2
#define PROCESS_ELEMS \
    dstT temp = convertToDT(loadpix(srcptr + src_index)); \
    dstT temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    temp2 = SUM_ABS(temp2); \
    FUNC(accumulator2, temp2); \
    FUNC(accumulator, temp)
#else
#define PROCESS_ELEMS \
    dstT temp = convertToDT(loadpix(srcptr + src_index)); \
    dstT temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    FUNC(accumulator, temp)
#endif
#else
#define PROCESS_ELEMS \
    dstT temp = convertToDT(loadpix(srcptr + src_index)); \
    FUNC(accumulator, temp)
#endif

#ifdef HAVE_MASK
#define REDUCE_GLOBAL \
    MASK_INDEX; \
    if (mask[mask_index]) \
    { \
        PROCESS_ELEMS; \
    }
#elif defined OP_DOT

#ifdef HAVE_SRC2_CONT
#define SRC2_INDEX int src2_index = mad24(id, srcTSIZE, src2_offset);
#else
#define SRC2_INDEX int src2_index = mad24(id / cols, src2_step, mad24(id % cols, srcTSIZE, src2_offset))
#endif

#if kercn == 1
#define REDUCE_GLOBAL \
    SRC2_INDEX; \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)), temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    FUNC(accumulator, temp, temp2)
#elif kercn == 2
#define REDUCE_GLOBAL \
    SRC2_INDEX; \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)), temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    FUNC(accumulator, temp.s0, temp2.s0); \
    FUNC(accumulator, temp.s1, temp2.s1)
#elif kercn == 4
#define REDUCE_GLOBAL \
    SRC2_INDEX; \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)), temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    FUNC(accumulator, temp.s0, temp2.s0); \
    FUNC(accumulator, temp.s1, temp2.s1); \
    FUNC(accumulator, temp.s2, temp2.s2); \
    FUNC(accumulator, temp.s3, temp2.s3)
#elif kercn == 8
#define REDUCE_GLOBAL \
    SRC2_INDEX; \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)), temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    FUNC(accumulator, temp.s0, temp2.s0); \
    FUNC(accumulator, temp.s1, temp2.s1); \
    FUNC(accumulator, temp.s2, temp2.s2); \
    FUNC(accumulator, temp.s3, temp2.s3); \
    FUNC(accumulator, temp.s4, temp2.s4); \
    FUNC(accumulator, temp.s5, temp2.s5); \
    FUNC(accumulator, temp.s6, temp2.s6); \
    FUNC(accumulator, temp.s7, temp2.s7)
#elif kercn == 16
#define REDUCE_GLOBAL \
    SRC2_INDEX; \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)), temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    FUNC(accumulator, temp.s0, temp2.s0); \
    FUNC(accumulator, temp.s1, temp2.s1); \
    FUNC(accumulator, temp.s2, temp2.s2); \
    FUNC(accumulator, temp.s3, temp2.s3); \
    FUNC(accumulator, temp.s4, temp2.s4); \
    FUNC(accumulator, temp.s5, temp2.s5); \
    FUNC(accumulator, temp.s6, temp2.s6); \
    FUNC(accumulator, temp.s7, temp2.s7); \
    FUNC(accumulator, temp.s8, temp2.s8); \
    FUNC(accumulator, temp.s9, temp2.s9); \
    FUNC(accumulator, temp.sA, temp2.sA); \
    FUNC(accumulator, temp.sB, temp2.sB); \
    FUNC(accumulator, temp.sC, temp2.sC); \
    FUNC(accumulator, temp.sD, temp2.sD); \
    FUNC(accumulator, temp.sE, temp2.sE); \
    FUNC(accumulator, temp.sF, temp2.sF)
#endif

#else // sum or norm with 2 args
#ifdef HAVE_SRC2
#ifdef OP_CALC2 // norm relative
#if kercn == 1
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    temp2 = SUM_ABS(temp2); \
    FUNC(accumulator, temp); \
    FUNC(accumulator2, temp2)
#elif kercn == 2
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    temp2 = SUM_ABS(temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator2, temp2.s0); \
    FUNC(accumulator2, temp2.s1)
#elif kercn == 4
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    temp2 = SUM_ABS(temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator2, temp2.s0); \
    FUNC(accumulator2, temp2.s1); \
    FUNC(accumulator2, temp2.s2); \
    FUNC(accumulator2, temp2.s3)
#elif kercn == 8
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    temp2 = SUM_ABS(temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator, temp.s4); \
    FUNC(accumulator, temp.s5); \
    FUNC(accumulator, temp.s6); \
    FUNC(accumulator, temp.s7); \
    FUNC(accumulator2, temp2.s0); \
    FUNC(accumulator2, temp2.s1); \
    FUNC(accumulator2, temp2.s2); \
    FUNC(accumulator2, temp2.s3); \
    FUNC(accumulator2, temp2.s4); \
    FUNC(accumulator2, temp2.s5); \
    FUNC(accumulator2, temp2.s6); \
    FUNC(accumulator2, temp2.s7)
#elif kercn == 16
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    temp2 = SUM_ABS(temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator, temp.s4); \
    FUNC(accumulator, temp.s5); \
    FUNC(accumulator, temp.s6); \
    FUNC(accumulator, temp.s7); \
    FUNC(accumulator, temp.s8); \
    FUNC(accumulator, temp.s9); \
    FUNC(accumulator, temp.sA); \
    FUNC(accumulator, temp.sB); \
    FUNC(accumulator, temp.sC); \
    FUNC(accumulator, temp.sD); \
    FUNC(accumulator, temp.sE); \
    FUNC(accumulator, temp.sF); \
    FUNC(accumulator2, temp2.s0); \
    FUNC(accumulator2, temp2.s1); \
    FUNC(accumulator2, temp2.s2); \
    FUNC(accumulator2, temp2.s3); \
    FUNC(accumulator2, temp2.s4); \
    FUNC(accumulator2, temp2.s5); \
    FUNC(accumulator2, temp2.s6); \
    FUNC(accumulator2, temp2.s7); \
    FUNC(accumulator2, temp2.s8); \
    FUNC(accumulator2, temp2.s9); \
    FUNC(accumulator2, temp2.sA); \
    FUNC(accumulator2, temp2.sB); \
    FUNC(accumulator2, temp2.sC); \
    FUNC(accumulator2, temp2.sD); \
    FUNC(accumulator2, temp2.sE); \
    FUNC(accumulator2, temp2.sF)
#endif
#else // norm with 2 args
#if kercn == 1
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    FUNC(accumulator, temp)
#elif kercn == 2
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1)
#elif kercn == 4
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3)
#elif kercn == 8
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator, temp.s4); \
    FUNC(accumulator, temp.s5); \
    FUNC(accumulator, temp.s6); \
    FUNC(accumulator, temp.s7)
#elif kercn == 16
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    dstTK temp2 = convertToDT(loadpix(src2ptr + src2_index)); \
    temp = SUM_ABS2(temp, temp2); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator, temp.s4); \
    FUNC(accumulator, temp.s5); \
    FUNC(accumulator, temp.s6); \
    FUNC(accumulator, temp.s7); \
    FUNC(accumulator, temp.s8); \
    FUNC(accumulator, temp.s9); \
    FUNC(accumulator, temp.sA); \
    FUNC(accumulator, temp.sB); \
    FUNC(accumulator, temp.sC); \
    FUNC(accumulator, temp.sD); \
    FUNC(accumulator, temp.sE); \
    FUNC(accumulator, temp.sF)
#endif
#endif

#else // sum
#if kercn == 1
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    FUNC(accumulator, temp)
#elif kercn == 2
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1)
#elif kercn == 4
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3)
#elif kercn == 8
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator, temp.s4); \
    FUNC(accumulator, temp.s5); \
    FUNC(accumulator, temp.s6); \
    FUNC(accumulator, temp.s7)
#elif kercn == 16
#define REDUCE_GLOBAL \
    dstTK temp = convertToDT(loadpix(srcptr + src_index)); \
    FUNC(accumulator, temp.s0); \
    FUNC(accumulator, temp.s1); \
    FUNC(accumulator, temp.s2); \
    FUNC(accumulator, temp.s3); \
    FUNC(accumulator, temp.s4); \
    FUNC(accumulator, temp.s5); \
    FUNC(accumulator, temp.s6); \
    FUNC(accumulator, temp.s7); \
    FUNC(accumulator, temp.s8); \
    FUNC(accumulator, temp.s9); \
    FUNC(accumulator, temp.sA); \
    FUNC(accumulator, temp.sB); \
    FUNC(accumulator, temp.sC); \
    FUNC(accumulator, temp.sD); \
    FUNC(accumulator, temp.sE); \
    FUNC(accumulator, temp.sF)
#endif
#endif
#endif

#ifdef OP_CALC2
#define SET_LOCAL_1 \
    localmem[lid] = accumulator; \
    localmem2[lid] = accumulator2
#define REDUCE_LOCAL_1 \
    localmem[lid - WGS2_ALIGNED] += accumulator; \
    localmem2[lid - WGS2_ALIGNED] += accumulator2
#define REDUCE_LOCAL_2 \
    localmem[lid] += localmem[lid2]; \
    localmem2[lid] += localmem2[lid2]
#define CALC_RESULT \
    storepix(localmem[0], dstptr + dstTSIZE * gid); \
    storepix(localmem2[0], dstptr + mad24(groupnum, dstTSIZE, dstTSIZE * gid))
#else
#define SET_LOCAL_1 \
    localmem[lid] = accumulator
#define REDUCE_LOCAL_1 \
    localmem[lid - WGS2_ALIGNED] += accumulator
#define REDUCE_LOCAL_2 \
    localmem[lid] += localmem[lid2]
#define CALC_RESULT \
    storepix(localmem[0], dstptr + dstTSIZE * gid)
#endif

// countNonZero stuff
#elif defined OP_COUNT_NON_ZERO
#define dstT int
#define DECLARE_LOCAL_MEM \
    __local dstT localmem[WGS2_ALIGNED]
#define DEFINE_ACCUMULATOR \
    dstT accumulator = (dstT)(0); \
    srcT1 zero = (srcT1)(0), one = (srcT1)(1)
#if kercn == 1
#define REDUCE_GLOBAL \
    accumulator += loadpix(srcptr + src_index) == zero ? zero : one
#elif kercn == 2
#define REDUCE_GLOBAL \
    srcT value = loadpix(srcptr + src_index); \
    accumulator += value.s0 == zero ? zero : one; \
    accumulator += value.s1 == zero ? zero : one
#elif kercn == 4
#define REDUCE_GLOBAL \
    srcT value = loadpix(srcptr + src_index); \
    accumulator += value.s0 == zero ? zero : one; \
    accumulator += value.s1 == zero ? zero : one; \
    accumulator += value.s2 == zero ? zero : one; \
    accumulator += value.s3 == zero ? zero : one
#elif kercn == 8
#define REDUCE_GLOBAL \
    srcT value = loadpix(srcptr + src_index); \
    accumulator += value.s0 == zero ? zero : one; \
    accumulator += value.s1 == zero ? zero : one; \
    accumulator += value.s2 == zero ? zero : one; \
    accumulator += value.s3 == zero ? zero : one; \
    accumulator += value.s4 == zero ? zero : one; \
    accumulator += value.s5 == zero ? zero : one; \
    accumulator += value.s6 == zero ? zero : one; \
    accumulator += value.s7 == zero ? zero : one
#elif kercn == 16
#define REDUCE_GLOBAL \
    srcT value = loadpix(srcptr + src_index); \
    accumulator += value.s0 == zero ? zero : one; \
    accumulator += value.s1 == zero ? zero : one; \
    accumulator += value.s2 == zero ? zero : one; \
    accumulator += value.s3 == zero ? zero : one; \
    accumulator += value.s4 == zero ? zero : one; \
    accumulator += value.s5 == zero ? zero : one; \
    accumulator += value.s6 == zero ? zero : one; \
    accumulator += value.s7 == zero ? zero : one; \
    accumulator += value.s8 == zero ? zero : one; \
    accumulator += value.s9 == zero ? zero : one; \
    accumulator += value.sA == zero ? zero : one; \
    accumulator += value.sB == zero ? zero : one; \
    accumulator += value.sC == zero ? zero : one; \
    accumulator += value.sD == zero ? zero : one; \
    accumulator += value.sE == zero ? zero : one; \
    accumulator += value.sF == zero ? zero : one
#endif

#define SET_LOCAL_1 \
    localmem[lid] = accumulator
#define REDUCE_LOCAL_1 \
    localmem[lid - WGS2_ALIGNED] += accumulator
#define REDUCE_LOCAL_2 \
    localmem[lid] += localmem[lid2]
#define CALC_RESULT \
    storepix(localmem[0], dstptr + dstTSIZE * gid)

#else
#error "No operation"
#endif

#ifdef OP_DOT
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , __global uchar * src2ptr, int src2_step, int src2_offset
#endif

__kernel void reduce(__global const uchar * srcptr, int src_step, int src_offset, int cols,
                     int total, int groupnum, __global uchar * dstptr EXTRA_PARAMS)
{
    int lid = get_local_id(0);
    int gid = get_group_id(0);
    int  id = get_global_id(0) * kercn;

    srcptr += src_offset;
#ifdef HAVE_SRC2
    src2ptr += src2_offset;
#endif

    DECLARE_LOCAL_MEM;
    DEFINE_ACCUMULATOR;

    for (int grain = groupnum * WGS * kercn; id < total; id += grain)
    {
#ifdef HAVE_SRC_CONT
        int src_index = mul24(id, srcTSIZE);
#else
        int src_index = mad24(id / cols, src_step, mul24(id % cols, srcTSIZE));
#endif
#ifdef HAVE_SRC2
#ifdef HAVE_SRC2_CONT
        int src2_index = mul24(id, srcTSIZE);
#else
        int src2_index = mad24(id / cols, src2_step, mul24(id % cols, srcTSIZE));
#endif
#endif
        REDUCE_GLOBAL;
    }

    if (lid < WGS2_ALIGNED)
    {
        SET_LOCAL_1;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid >= WGS2_ALIGNED && total >= WGS2_ALIGNED)
    {
        REDUCE_LOCAL_1;
    }
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
        CALC_RESULT;
    }
}
