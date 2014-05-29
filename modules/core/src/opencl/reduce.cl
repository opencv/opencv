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

#if defined OP_NORM_INF_MASK || defined OP_MIN_MAX_LOC || defined OP_MIN_MAX_LOC_MASK

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

#ifdef HAVE_MASK
#define EXTRA_PARAMS , __global const uchar * mask, int mask_step, int mask_offset
#else
#define EXTRA_PARAMS
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
#define FUNC(a, b) a += b >= (dstT)(0) ? b : -b

#elif defined OP_SUM_SQR
#if ddepth <= 4
#define FUNC(a, b) a = mad24(b, b, a)
#else
#define FUNC(a, b) a = mad(b, b, a)
#endif
#endif

#define DECLARE_LOCAL_MEM \
    __local dstT localmem[WGS2_ALIGNED]
#define DEFINE_ACCUMULATOR \
    dstT accumulator = (dstT)(0)

#ifdef HAVE_MASK
#define REDUCE_GLOBAL \
    MASK_INDEX; \
    if (mask[mask_index]) \
    { \
        dstT temp = convertToDT(loadpix(srcptr + src_index)); \
        FUNC(accumulator, temp); \
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

#else
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

#define SET_LOCAL_1 \
    localmem[lid] = accumulator
#define REDUCE_LOCAL_1 \
    localmem[lid - WGS2_ALIGNED] += accumulator
#define REDUCE_LOCAL_2 \
    localmem[lid] += localmem[lid2]
#define CALC_RESULT \
    storepix(localmem[0], dstptr + dstTSIZE * gid)

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

// norm (NORM_INF) with cn > 1 and mask
#elif defined OP_NORM_INF_MASK

#define DECLARE_LOCAL_MEM \
    __local srcT localmem_max[WGS2_ALIGNED]
#define DEFINE_ACCUMULATOR \
    srcT maxval = MIN_VAL, temp
#define REDUCE_GLOBAL \
    MASK_INDEX; \
    if (mask[mask_index]) \
    { \
        temp = loadpix(srcptr + src_index); \
        maxval = max(maxval, (srcT)(temp >= (srcT)(0) ? temp : -temp)); \
    }
#define SET_LOCAL_1 \
    localmem_max[lid] = maxval
#define REDUCE_LOCAL_1 \
    localmem_max[lid - WGS2_ALIGNED] = max(maxval, localmem_max[lid - WGS2_ALIGNED])
#define REDUCE_LOCAL_2 \
    localmem_max[lid] = max(localmem_max[lid], localmem_max[lid2])
#define CALC_RESULT \
    storepix(localmem_max[0], dstptr + dstTSIZE * gid)

// minMaxLoc stuff
#elif defined OP_MIN_MAX_LOC || defined OP_MIN_MAX_LOC_MASK

#define DECLARE_LOCAL_MEM \
    __local srcT localmem_min[WGS2_ALIGNED]; \
    __local srcT localmem_max[WGS2_ALIGNED]; \
    __local int localmem_minloc[WGS2_ALIGNED]; \
    __local int localmem_maxloc[WGS2_ALIGNED]
#define DEFINE_ACCUMULATOR \
    srcT minval = MAX_VAL; \
    srcT maxval = MIN_VAL; \
    int negative = -1; \
    int minloc = negative; \
    int maxloc = negative; \
    srcT temp; \
    int temploc
#define REDUCE_GLOBAL \
    temp = loadpix(srcptr + src_index); \
    temploc = id; \
    srcT temp_minval = minval, temp_maxval = maxval; \
    minval = min(minval, temp); \
    maxval = max(maxval, temp); \
    minloc = (minval == temp_minval) ? (temp_minval == MAX_VAL) ? temploc : minloc : temploc; \
    maxloc = (maxval == temp_maxval) ? (temp_maxval == MIN_VAL) ? temploc : maxloc : temploc
#define SET_LOCAL_1 \
    localmem_min[lid] = minval; \
    localmem_max[lid] = maxval; \
    localmem_minloc[lid] = minloc; \
    localmem_maxloc[lid] = maxloc
#define REDUCE_LOCAL_1 \
    srcT oldmin = localmem_min[lid-WGS2_ALIGNED]; \
    srcT oldmax = localmem_max[lid-WGS2_ALIGNED]; \
    localmem_min[lid - WGS2_ALIGNED] = min(minval, localmem_min[lid-WGS2_ALIGNED]); \
    localmem_max[lid - WGS2_ALIGNED] = max(maxval, localmem_max[lid-WGS2_ALIGNED]); \
    srcT minv = localmem_min[lid - WGS2_ALIGNED], maxv = localmem_max[lid - WGS2_ALIGNED]; \
    localmem_minloc[lid - WGS2_ALIGNED] = (minv == minval) ? (minv == oldmin) ? \
        min(minloc, localmem_minloc[lid-WGS2_ALIGNED]) : minloc : localmem_minloc[lid-WGS2_ALIGNED]; \
    localmem_maxloc[lid - WGS2_ALIGNED] = (maxv == maxval) ? (maxv == oldmax) ? \
        min(maxloc, localmem_maxloc[lid-WGS2_ALIGNED]) : maxloc : localmem_maxloc[lid-WGS2_ALIGNED]
#define REDUCE_LOCAL_2 \
    srcT oldmin = localmem_min[lid]; \
    srcT oldmax = localmem_max[lid]; \
    localmem_min[lid] = min(localmem_min[lid], localmem_min[lid2]); \
    localmem_max[lid] = max(localmem_max[lid], localmem_max[lid2]); \
    srcT min1 = localmem_min[lid], min2 = localmem_min[lid2]; \
    localmem_minloc[lid] = (localmem_minloc[lid] == negative) ? localmem_minloc[lid2] : (localmem_minloc[lid2] == negative) ? \
        localmem_minloc[lid] : (min1 == min2) ? (min1 == oldmin) ? min(localmem_minloc[lid2],localmem_minloc[lid]) : \
        localmem_minloc[lid2] : localmem_minloc[lid]; \
    srcT max1 = localmem_max[lid], max2 = localmem_max[lid2]; \
    localmem_maxloc[lid] = (localmem_maxloc[lid] == negative) ? localmem_maxloc[lid2] : (localmem_maxloc[lid2] == negative) ? \
        localmem_maxloc[lid] : (max1 == max2) ? (max1 == oldmax) ? min(localmem_maxloc[lid2],localmem_maxloc[lid]) : \
        localmem_maxloc[lid2] : localmem_maxloc[lid]
#define CALC_RESULT \
    storepix(localmem_min[0], dstptr + dstTSIZE * gid); \
    storepix(localmem_max[0], dstptr2 + dstTSIZE * gid); \
    dstlocptr[gid] = localmem_minloc[0]; \
    dstlocptr2[gid] = localmem_maxloc[0]

#if defined OP_MIN_MAX_LOC_MASK
#undef DEFINE_ACCUMULATOR
#define DEFINE_ACCUMULATOR \
    srcT minval = MAX_VAL; \
    srcT maxval = MIN_VAL; \
    int negative = -1; \
    int minloc = negative; \
    int maxloc = negative; \
    srcT temp, temp_mask, zeroVal = (srcT)(0); \
    int temploc
#undef REDUCE_GLOBAL
#define REDUCE_GLOBAL \
    temp = loadpix(srcptr + src_index); \
    temploc = id; \
    MASK_INDEX; \
    __global const uchar * mask = (__global const uchar *)(maskptr + mask_index); \
    temp_mask = mask[0]; \
    srcT temp_minval = minval, temp_maxval = maxval; \
    minval = (temp_mask == zeroVal) ? minval : min(minval, temp); \
    maxval = (temp_mask == zeroVal) ? maxval : max(maxval, temp); \
    minloc = (temp_mask == zeroVal) ? minloc : (minval == temp_minval) ? (temp_minval == MAX_VAL) ? temploc : minloc : temploc; \
    maxloc = (temp_mask == zeroVal) ? maxloc : (maxval == temp_maxval) ? (temp_maxval == MIN_VAL) ? temploc : maxloc : temploc
#endif

#else
#error "No operation"
#endif // end of minMaxLoc stuff

#ifdef OP_MIN_MAX_LOC
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , __global uchar * dstptr2, __global int * dstlocptr, __global int * dstlocptr2

#elif defined OP_MIN_MAX_LOC_MASK
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , __global uchar * dstptr2, __global int * dstlocptr, __global int * dstlocptr2, \
    __global const uchar * maskptr, int mask_step, int mask_offset

#elif defined OP_DOT
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

    DECLARE_LOCAL_MEM;
    DEFINE_ACCUMULATOR;

    for (int grain = groupnum * WGS * kercn; id < total; id += grain)
    {
#ifdef HAVE_SRC_CONT
        int src_index = mul24(id, srcTSIZE);
#else
        int src_index = mad24(id / cols, src_step, mul24(id % cols, srcTSIZE));
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
