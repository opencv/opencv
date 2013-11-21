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
// In no event shall the copyright holders or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*
  Usage:
     after compiling this program user gets a single kernel called KF.
     the following flags should be passed:
     1) one of "-D BINARY_OP", "-D UNARY_OP", "-D MASK_BINARY_OP" or "-D MASK_UNARY_OP"
     2) the actual operation performed, one of "-D OP_...", see below the list of operations.
     2a) "-D dstDepth=<destination depth> [-D cn=<num channels]"
         for some operations, like min/max/and/or/xor it's enough
     2b) "-D srcDepth1=<source1 depth> -D srcDepth2=<source2 depth> -D dstDepth=<destination depth>
          -D workDepth=<work depth> [-D cn=<num channels>]" - for mixed-type operations
*/

#if defined (DOUBLE_SUPPORT)
#ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#elif defined (cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#endif
#endif

#define CV_32S 4
#define CV_32F 5

#define dstelem *(__global dstT*)(dstptr + dst_index)
#define noconvert(x) x

#ifndef workT

    #define srcT1 dstT
    #define srcT2 dstT
    #define workT dstT
    #define srcelem1 *(__global dstT*)(srcptr1 + src1_index)
    #define srcelem2 *(__global dstT*)(srcptr2 + src2_index)
    #define convertToDT noconvert

#else

    #define srcelem1 convertToWT1(*(__global srcT1*)(srcptr1 + src1_index))
    #define srcelem2 convertToWT2(*(__global srcT2*)(srcptr2 + src2_index))

#endif

#define EXTRA_PARAMS

#if defined OP_ADD_SAT
#define PROCESS_ELEM dstelem = add_sat(srcelem1, srcelem2)

#elif defined OP_ADD
#define PROCESS_ELEM dstelem = convertToDT(srcelem1 + srcelem2)

#elif defined OP_SUB_SAT
#define PROCESS_ELEM dstelem = sub_sat(srcelem1, srcelem2)

#elif defined OP_SUB
#define PROCESS_ELEM dstelem = convertToDT(srcelem1 - srcelem2)

#elif defined OP_RSUB_SAT
#define PROCESS_ELEM dstelem = sub_sat(srcelem2, srcelem1)

#elif defined OP_RSUB
#define PROCESS_ELEM dstelem = convertToDT(srcelem2 - srcelem1)

#elif defined OP_ABSDIFF
#define PROCESS_ELEM dstelem = abs_diff(srcelem1, srcelem2)

#elif defined OP_AND
#define PROCESS_ELEM dstelem = srcelem1 & srcelem2

#elif defined OP_OR
#define PROCESS_ELEM dstelem = srcelem1 | srcelem2

#elif defined OP_XOR
#define PROCESS_ELEM dstelem = srcelem1 ^ srcelem2

#elif defined OP_NOT
#define PROCESS_ELEM dstelem = ~srcelem1

#elif defined OP_MIN
#define PROCESS_ELEM dstelem = min(srcelem1, srcelem2)

#elif defined OP_MAX
#define PROCESS_ELEM dstelem = max(srcelem1, srcelem2)

#elif defined OP_MUL
#define PROCESS_ELEM dstelem = convertToDT(srcelem1 * srcelem2)

#elif defined OP_MUL_SCALE
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , workT scale
#define PROCESS_ELEM dstelem = convertToDT(srcelem1 * srcelem2 * scale)

#elif defined OP_DIV
#define PROCESS_ELEM \
        workT e2 = srcelem2, zero = (workT)(0); \
        dstelem = convertToDT(e2 != zero ? srcelem1 / e2 : zero)

#elif defined OP_DIV_SCALE
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , workT scale
#define PROCESS_ELEM \
        workT e2 = srcelem2, zero = (workT)(0); \
        dstelem = convertToDT(e2 != zero ? srcelem1 * scale / e2 : zero)

#elif defined OP_RECIP_SCALE
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , workT scale
#define PROCESS_ELEM \
        workT e1 = srcelem1, zero = (workT)(0); \
        dstelem = convertToDT(e1 != zero ? scale / e1 : zero)

#elif defined OP_ADDW
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , workT alpha, workT beta, workT gamma
#define PROCESS_ELEM dstelem = convertToDT(srcelem1*alpha + srcelem2*beta + gamma)

#elif defined OP_MAG
#define PROCESS_ELEM dstelem = hypot(srcelem1, srcelem2)

#elif defined OP_PHASE_RADIANS
#define PROCESS_ELEM \
        workT tmp = atan2(srcelem2, srcelem1); \
        if(tmp < 0) tmp += 6.283185307179586232; \
        dstelem = tmp

#elif defined OP_PHASE_DEGREES
    #define PROCESS_ELEM \
    workT tmp = atan2(srcelem2, srcelem1)*57.29577951308232286465; \
    if(tmp < 0) tmp += 360; \
    dstelem = tmp

#elif defined OP_EXP
#define PROCESS_ELEM dstelem = exp(srcelem1)

#elif defined OP_SQRT
#define PROCESS_ELEM dstelem = sqrt(srcelem1)

#elif defined OP_LOG
#define PROCESS_ELEM dstelem = log(abs(srcelem1))

#elif defined OP_CMP
#define PROCESS_ELEM dstelem = convert_uchar(srcelem1 CMP_OPERATOR srcelem2 ? 255 : 0)

#elif defined OP_CONVERT
#define PROCESS_ELEM dstelem = convertToDT(srcelem1)

#elif defined OP_CONVERT_SCALE
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , workT alpha, workT beta
#define PROCESS_ELEM dstelem = convertToDT(srcelem1*alpha + beta)

#else
#error "unknown op type"
#endif

#if defined UNARY_OP || defined MASK_UNARY_OP
#undef srcelem2
#if defined OP_AND || defined OP_OR || defined OP_XOR || defined OP_ADD || defined OP_SAT_ADD || \
    defined OP_SUB || defined OP_SAT_SUB || defined OP_RSUB || defined OP_SAT_RSUB || \
    defined OP_ABSDIFF || defined OP_CMP || defined OP_MIN || defined OP_MAX
    #undef EXTRA_PARAMS
    #define EXTRA_PARAMS , workT srcelem2
#endif
#endif

#if defined BINARY_OP

__kernel void KF(__global const uchar* srcptr1, int srcstep1, int srcoffset1,
                 __global const uchar* srcptr2, int srcstep2, int srcoffset2,
                 __global uchar* dstptr, int dststep, int dstoffset,
                 int rows, int cols EXTRA_PARAMS )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, srcstep1, x*(int)sizeof(srcT1) + srcoffset1);
        int src2_index = mad24(y, srcstep2, x*(int)sizeof(srcT2) + srcoffset2);
        int dst_index  = mad24(y, dststep, x*(int)sizeof(dstT) + dstoffset);

        PROCESS_ELEM;
        //printf("(x=%d, y=%d). %d, %d, %d\n", x, y, (int)srcelem1, (int)srcelem2, (int)dstelem);
    }
}

#elif defined MASK_BINARY_OP

__kernel void KF(__global const uchar* srcptr1, int srcstep1, int srcoffset1,
                 __global const uchar* srcptr2, int srcstep2, int srcoffset2,
                 __global const uchar* mask, int maskstep, int maskoffset,
                 __global uchar* dstptr, int dststep, int dstoffset,
                 int rows, int cols EXTRA_PARAMS )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int mask_index = mad24(y, maskstep, x + maskoffset);
        if( mask[mask_index] )
        {
            int src1_index = mad24(y, srcstep1, x*(int)sizeof(srcT1) + srcoffset1);
            int src2_index = mad24(y, srcstep2, x*(int)sizeof(srcT2) + srcoffset2);
            int dst_index  = mad24(y, dststep, x*(int)sizeof(dstT) + dstoffset);

            PROCESS_ELEM;
        }
    }
}

#elif defined UNARY_OP

__kernel void KF(__global const uchar* srcptr1, int srcstep1, int srcoffset1,
                 __global uchar* dstptr, int dststep, int dstoffset,
                 int rows, int cols EXTRA_PARAMS )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int src1_index = mad24(y, srcstep1, x*(int)sizeof(srcT1) + srcoffset1);
        int dst_index  = mad24(y, dststep, x*(int)sizeof(dstT) + dstoffset);

        PROCESS_ELEM;
    }
}

#elif defined MASK_UNARY_OP

__kernel void KF(__global const uchar* srcptr1, int srcstep1, int srcoffset1,
                 __global const uchar* mask, int maskstep, int maskoffset,
                 __global uchar* dstptr, int dststep, int dstoffset,
                 int rows, int cols EXTRA_PARAMS )
{
    int x = get_global_id(0);
    int y = get_global_id(1);

    if (x < cols && y < rows)
    {
        int mask_index = mad24(y, maskstep, x + maskoffset);
        if( mask[mask_index] )
        {
            int src1_index = mad24(y, srcstep1, x*(int)sizeof(srcT1) + srcoffset1);
            int dst_index  = mad24(y, dststep, x*(int)sizeof(dstT) + dstoffset);

            PROCESS_ELEM;
        }
    }
}

#else

#error "Unknown operation type"

#endif
