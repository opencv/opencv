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

#if dstDepth == 0
#define dstT_ uchar
#elif dstDepth == 1
#define dstT_ char
#elif dstDepth == 2
#define dstT_ ushort
#elif dstDepth == 3
#define dstT_ short
#elif dstDepth == 4
#define dstT_ int
#elif dstDepth == 5
#define dstT_ float
#elif dstDepth == 6
#define dstT_ double
#elif dstDepth == 7
/* specially for bit & byte-level operations */
#define dstT_ long
#endif

#define PASTE(a, b) a##b

#if defined cn && cn != 1
#define ADD_CN(s) PASTE(s, cn)
#else
#define ADD_CN(s) s
#endif

#define dstT ADD_CN(dstT_)
#define dstelem *(dstT*)(dstptr + dst_index)

#ifndef workDepth

    #define srcT1_ dstT_
    #define srcT1 dstT
    #define srcT2_ dstT_
    #define srcT2 dstT
    #define workT_ dstT_
    #define workT dstT
    #define srcelem1 *(dstT*)(srcptr1 + src1_index)
    #define srcelem2 *(dstT*)(srcptr2 + src2_index)

    #ifdef OP_ADD
    #undef OP_ADD
    #define OP_SAT_ADD
    #endif

    #ifdef OP_SUB
    #undef OP_SUB
    #define OP_SAT_SUB
    #endif

    #ifdef OP_RSUB
    #undef OP_RSUB
    #define OP_SAT_RSUB
    #endif

#else

    #define srcT1_ PASTE(TYPE, srcDepth1)
    #define srcT1 ADD_CN(srcT1_)
    #define srcT2_ PASTE(TYPE, srcDepth2)
    #define srcT2 ADD_CN(srcT2_)
    #define workT_ PASTE(TYPE, workDepth)
    #define workT ADD_CN(workT_)

    #if workDepth == srcDepth1
        #define convertToWT1
    #elif workDepth > srcDepth1
        #define convertToWT1 PASTE(convert_, workT)
    #elif workDepth < CV_32S
        #if srcDepth1 >= CV_32F
            #define convertToWT1 PASTE(PASTE(convert_, workT), _sat_rte)
        #else
            #define convertToWT1 PASTE(PASTE(convert_, workT), _sat)
        #endif
    #else
        #define convertToWT1 PASTE(PASTE(convert_, workT), _rte)
    #endif

    #if workDepth == srcDepth2
        #define convertToWT2
    #elif workDepth > srcDepth2
        #define convertToWT2 convert_##workT
    #elif workDepth < CV_32S
        #if srcDepth2 >= CV_32F
            #define convertToWT2 PASTE(PASTE(convert_, workT), _sat_rte)
        #else
            #define convertToWT2 PASTE(PASTE(convert_, workT), _sat)
        #endif
    #else
        #define convertToWT2 PASTE(PASTE(convert_, workT), _rte)
    #endif

    #if workDepth == dstDepth
        #define convertToDT
    #elif dstDepth < CV_32S
        #if workDepth >= CV_32F
            #define convertToDT PASTE(PASTE(convert_, dstT), _sat_rte)
        #else
            #define convertToDT PASTE(PASTE(convert_, dstT), _sat)
        #endif
    #elif dstDepth == CV_32S && workDepth >= CV_32F
        #define convertToDT PASTE(PASTE(convert_, dstT), _rte)
    #else
        #define convertToDT PASTE(convert_, dstT)
    #endif

    #define srcelem1 convertToWT1(*(srcT1*)(srcptr1 + src1_index))
    #define srcelem2 convertToWT2(*(srcT2*)(srcptr2 + src2_index))

#endif

#define EXTRA_PARAMS

#if defined OP_SAT_ADD
#define PROCESS_ELEM dstelem = sat_add(srcelem1, srcelem2)

#elif defined OP_ADD
#define PROCESS_ELEM dstelem = convertToDT(srcelem1 + srcelem2)

#elif defined OP_SAT_SUB
#define PROCESS_ELEM dstelem = sat_sub(srcelem1, srcelem2)

#elif defined OP_SUB
#define PROCESS_ELEM dstelem = convertToDT(srcelem1 - srcelem2)

#elif defined OP_SAT_RSUB
#define PROCESS_ELEM dstelem = sat_sub(srcelem2, srcelem1)

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

#elif defined OP_SET
#undef EXTRA_PARAMS
#define EXTRA_PARAMS , dstT value
#define PROCESS_ELEM dstelem = value

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
        int src1_index = mad24(y, srcstep1, x*sizeof(srcT1) + srcoffset1);
        int src2_index = mad24(y, srcstep2, x*sizeof(srcT2) + srcoffset2);
        int dst_index  = mad24(y, dststep, x*sizeof(dstT) + dstoffset);

        PROCESS_ELEM;
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
            int src1_index = mad24(y, srcstep1, x*sizeof(srcT1) + srcoffset1);
            int src2_index = mad24(y, srcstep2, x*sizeof(srcT2) + srcoffset2);
            int dst_index  = mad24(y, dststep, x*sizeof(dstT) + dstoffset);

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
        int src1_index = mad24(y, srcstep1, x*sizeof(srcT1) + srcoffset1);
        int dst_index  = mad24(y, dststep, x*sizeof(dstT) + dstoffset);

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
            int src1_index = mad24(y, srcstep1, x*sizeof(srcT1) + srcoffset1);
            int dst_index  = mad24(y, dststep, x*sizeof(dstT) + dstoffset);

            PROCESS_ELEM;
        }
    }
}

#else

#error "Unknown operation type"

#endif




