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
//     and/or other oclMaterials provided with the distribution.
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

/**************************************PUBLICFUNC*************************************/
#if defined (DOUBLE_SUPPORT)
    #ifdef cl_khr_fp64
#pragma OPENCL EXTENSION cl_khr_fp64:enable
    #elif defined (cl_amd_fp64)
        #pragma OPENCL EXTENSION cl_amd_fp64:enable
    #endif
#define RES_TYPE double8
#define CONVERT_RES_TYPE convert_double8
#else
#define RES_TYPE float8
#define CONVERT_RES_TYPE convert_float8
#endif

#if defined (DEPTH_0)
#define VEC_TYPE uchar8
#endif
#if defined (DEPTH_1)
#define VEC_TYPE char8
#endif
#if defined (DEPTH_2)
#define VEC_TYPE ushort8
#endif
#if defined (DEPTH_3)
#define VEC_TYPE short8
#endif
#if defined (DEPTH_4)
#define VEC_TYPE int8
#endif
#if defined (DEPTH_5)
#define VEC_TYPE float8
#endif
#if defined (DEPTH_6)
#define VEC_TYPE double8
#endif

#if defined (FUNC_TYPE_0)
#define FUNC(a,b) b += a;
#endif
#if defined (FUNC_TYPE_1)
#define FUNC(a,b) b = b + (a >= 0 ? a : -a);
#endif
#if defined (FUNC_TYPE_2)
#define FUNC(a,b) b = b + a * a;
#endif

#if defined (REPEAT_S0)
#define repeat_s(a) a = a;
#endif
#if defined (REPEAT_S1)
#define repeat_s(a) a.s0 = 0;
#endif
#if defined (REPEAT_S2)
#define repeat_s(a) a.s0 = 0;a.s1 = 0;
#endif
#if defined (REPEAT_S3)
#define repeat_s(a) a.s0 = 0;a.s1 = 0;a.s2 = 0;
#endif
#if defined (REPEAT_S4)
#define repeat_s(a) a.s0 = 0;a.s1 = 0;a.s2 = 0;a.s3 = 0;
#endif
#if defined (REPEAT_S5)
#define repeat_s(a) a.s0 = 0;a.s1 = 0;a.s2 = 0;a.s3 = 0;a.s4 = 0;
#endif
#if defined (REPEAT_S6)
#define repeat_s(a) a.s0 = 0;a.s1 = 0;a.s2 = 0;a.s3 = 0;a.s4 = 0;a.s5 = 0;
#endif
#if defined (REPEAT_S7)
#define repeat_s(a) a.s0 = 0;a.s1 = 0;a.s2 = 0;a.s3 = 0;a.s4 = 0;a.s5 = 0;a.s6 = 0;
#endif

#if defined (REPEAT_E0)
#define repeat_e(a) a = a;
#endif
#if defined (REPEAT_E1)
#define repeat_e(a) a.s7 = 0;
#endif
#if defined (REPEAT_E2)
#define repeat_e(a) a.s7 = 0;a.s6 = 0;
#endif
#if defined (REPEAT_E3)
#define repeat_e(a) a.s7 = 0;a.s6 = 0;a.s5 = 0;
#endif
#if defined (REPEAT_E4)
#define repeat_e(a) a.s7 = 0;a.s6 = 0;a.s5 = 0;a.s4 = 0;
#endif
#if defined (REPEAT_E5)
#define repeat_e(a) a.s7 = 0;a.s6 = 0;a.s5 = 0;a.s4 = 0;a.s3 = 0;
#endif
#if defined (REPEAT_E6)
#define repeat_e(a) a.s7 = 0;a.s6 = 0;a.s5 = 0;a.s4 = 0;a.s3 = 0;a.s2 = 0;
#endif
#if defined (REPEAT_E7)
#define repeat_e(a) a.s7 = 0;a.s6 = 0;a.s5 = 0;a.s4 = 0;a.s3 = 0;a.s2 = 0;a.s1 = 0;
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable

/**************************************Array buffer SUM**************************************/
__kernel void arithm_op_sum (int cols,int invalid_cols,int offset,int elemnum,int groupnum,
                                __global VEC_TYPE *src, __global RES_TYPE *dst)
{
   unsigned int lid = get_local_id(0);
   unsigned int gid = get_group_id(0);
   unsigned int  id = get_global_id(0);
   unsigned int idx = offset + id + (id / cols) * invalid_cols;
   __local RES_TYPE localmem_sum[128];
   RES_TYPE sum = 0,temp;
   if(id < elemnum)
   {
       temp = CONVERT_RES_TYPE(src[idx]);
       if(id % cols == 0 )
       {
           repeat_s(temp);
       }
       if(id % cols == cols - 1)
       {
           repeat_e(temp);
       }
       FUNC(temp,sum);
   }
   else
   {
       sum = 0;
   }
   for(id=id + (groupnum << 8); id < elemnum;id = id + (groupnum << 8))
   {
       idx = offset + id + (id / cols) * invalid_cols;
       temp = CONVERT_RES_TYPE(src[idx]);
       if(id % cols == 0 )
       {
               repeat_s(temp);
       }
       if(id % cols == cols - 1)
       {
               repeat_e(temp);
       }
       FUNC(temp,sum);
   }
   if(lid > 127)
   {
       localmem_sum[lid - 128] = sum;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   if(lid < 128)
   {
       localmem_sum[lid] = sum + localmem_sum[lid];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int lsize = 64; lsize > 0; lsize >>= 1)
   {
       if(lid < lsize)
       {
           int lid2 = lsize + lid;
           localmem_sum[lid] = localmem_sum[lid] + localmem_sum[lid2];
       }
       barrier(CLK_LOCAL_MEM_FENCE);
   }
   if( lid == 0)
   {
       dst[gid] = localmem_sum[0];
   }
}

