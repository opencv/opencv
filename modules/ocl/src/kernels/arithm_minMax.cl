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
//     and/or other GpuMaterials provided with the distribution.
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
#endif

#if defined (DEPTH_0)
#define VEC_TYPE uchar8
#define CONVERT_TYPE convert_uchar8
#define MIN_VAL 0
#define MAX_VAL 255
#endif
#if defined (DEPTH_1)
#define VEC_TYPE char8
#define CONVERT_TYPE convert_char8
#define MIN_VAL -128
#define MAX_VAL 127
#endif
#if defined (DEPTH_2)
#define VEC_TYPE ushort8
#define CONVERT_TYPE convert_ushort8
#define MIN_VAL 0
#define MAX_VAL 65535
#endif
#if defined (DEPTH_3)
#define VEC_TYPE short8
#define CONVERT_TYPE convert_short8
#define MIN_VAL -32768
#define MAX_VAL 32767
#endif
#if defined (DEPTH_4)
#define VEC_TYPE int8
#define CONVERT_TYPE convert_int8
#define MIN_VAL INT_MIN
#define MAX_VAL INT_MAX
#endif
#if defined (DEPTH_5)
#define VEC_TYPE float8
#define CONVERT_TYPE convert_float8
#define MIN_VAL (-FLT_MAX)
#define MAX_VAL FLT_MAX
#endif
#if defined (DEPTH_6)
#define VEC_TYPE double8
#define CONVERT_TYPE convert_double8
#define MIN_VAL (-DBL_MAX)
#define MAX_VAL DBL_MAX
#endif

#if defined (REPEAT_S0)
#define repeat_s(a) a = a;
#endif
#if defined (REPEAT_S1)
#define repeat_s(a) a.s0 = a.s1;
#endif
#if defined (REPEAT_S2)
#define repeat_s(a) a.s0 = a.s2;a.s1 = a.s2;
#endif
#if defined (REPEAT_S3)
#define repeat_s(a) a.s0 = a.s3;a.s1 = a.s3;a.s2 = a.s3;
#endif
#if defined (REPEAT_S4)
#define repeat_s(a) a.s0 = a.s4;a.s1 = a.s4;a.s2 = a.s4;a.s3 = a.s4;
#endif
#if defined (REPEAT_S5)
#define repeat_s(a) a.s0 = a.s5;a.s1 = a.s5;a.s2 = a.s5;a.s3 = a.s5;a.s4 = a.s5;
#endif
#if defined (REPEAT_S6)
#define repeat_s(a) a.s0 = a.s6;a.s1 = a.s6;a.s2 = a.s6;a.s3 = a.s6;a.s4 = a.s6;a.s5 = a.s6;
#endif
#if defined (REPEAT_S7)
#define repeat_s(a) a.s0 = a.s7;a.s1 = a.s7;a.s2 = a.s7;a.s3 = a.s7;a.s4 = a.s7;a.s5 = a.s7;a.s6 = a.s7;
#endif

#if defined (REPEAT_E0)
#define repeat_e(a) a = a;
#endif
#if defined (REPEAT_E1)
#define repeat_e(a) a.s7 = a.s6;
#endif
#if defined (REPEAT_E2)
#define repeat_e(a) a.s7 = a.s5;a.s6 = a.s5;
#endif
#if defined (REPEAT_E3)
#define repeat_e(a) a.s7 = a.s4;a.s6 = a.s4;a.s5 = a.s4;
#endif
#if defined (REPEAT_E4)
#define repeat_e(a) a.s7 = a.s3;a.s6 = a.s3;a.s5 = a.s3;a.s4 = a.s3;
#endif
#if defined (REPEAT_E5)
#define repeat_e(a) a.s7 = a.s2;a.s6 = a.s2;a.s5 = a.s2;a.s4 = a.s2;a.s3 = a.s2;
#endif
#if defined (REPEAT_E6)
#define repeat_e(a) a.s7 = a.s1;a.s6 = a.s1;a.s5 = a.s1;a.s4 = a.s1;a.s3 = a.s1;a.s2 = a.s1;
#endif
#if defined (REPEAT_E7)
#define repeat_e(a) a.s7 = a.s0;a.s6 = a.s0;a.s5 = a.s0;a.s4 = a.s0;a.s3 = a.s0;a.s2 = a.s0;a.s1 = a.s0;
#endif

#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics:enable
#pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics:enable

/**************************************Array minMax**************************************/
__kernel void arithm_op_minMax (int cols,int invalid_cols,int offset,int elemnum,int groupnum,
                                  __global VEC_TYPE *src, __global VEC_TYPE *dst)
{
   unsigned int lid = get_local_id(0);
   unsigned int gid = get_group_id(0);
   unsigned int  id = get_global_id(0);
   unsigned int idx = offset + id + (id / cols) * invalid_cols;
   __local VEC_TYPE localmem_max[128],localmem_min[128];
   VEC_TYPE minval,maxval,temp;
   if(id < elemnum)
   {
       temp = src[idx];
       if(id % cols == 0 )
       {
           repeat_s(temp);
       }
       if(id % cols == cols - 1)
       {
           repeat_e(temp);
       }
       minval = temp;
       maxval = temp;
   }
   else
   {
       minval = MAX_VAL;
       maxval = MIN_VAL;
   }
   for(id=id + (groupnum << 8); id < elemnum;id = id + (groupnum << 8))
   {
       idx = offset + id + (id / cols) * invalid_cols;
       temp = src[idx];
       if(id % cols == 0 )
       {
               repeat_s(temp);
       }
       if(id % cols == cols - 1)
       {
               repeat_e(temp);
       }
       minval = min(minval,temp);
       maxval = max(maxval,temp);
   }
   if(lid > 127)
   {
       localmem_min[lid - 128] = minval;
       localmem_max[lid - 128] = maxval;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   if(lid < 128)
   {
       localmem_min[lid] = min(minval,localmem_min[lid]);
       localmem_max[lid] = max(maxval,localmem_max[lid]);
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int lsize = 64; lsize > 0; lsize >>= 1)
   {
       if(lid < lsize)
       {
           int lid2 = lsize + lid;
           localmem_min[lid] = min(localmem_min[lid] , localmem_min[lid2]);
           localmem_max[lid] = max(localmem_max[lid] , localmem_max[lid2]);
       }
       barrier(CLK_LOCAL_MEM_FENCE);
   }
   if( lid == 0)
   {
       dst[gid] = localmem_min[0];
       dst[gid + groupnum] = localmem_max[0];
   }
}
