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
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#define RES_TYPE double4
#define CONVERT_RES_TYPE convert_double4
#else
#define RES_TYPE float4
#define CONVERT_RES_TYPE convert_float4
#endif

#if defined (DEPTH_0)
#define VEC_TYPE uchar4
#endif
#if defined (DEPTH_1)
#define VEC_TYPE char4
#endif
#if defined (DEPTH_2)
#define VEC_TYPE ushort4
#endif
#if defined (DEPTH_3)
#define VEC_TYPE short4
#endif
#if defined (DEPTH_4)
#define VEC_TYPE int4
#endif
#if defined (DEPTH_5)
#define VEC_TYPE float4
#endif
#if defined (DEPTH_6)
#define VEC_TYPE double4
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
#define repeat_s(a,b,c) a=a; b =b; c=c;
#endif
#if defined (REPEAT_S1)
#define repeat_s(a,b,c) a.s0=0; b=b; c=c;
#endif
#if defined (REPEAT_S2)
#define repeat_s(a,b,c) a.s0=0; a.s1=0; b=b; c=c;
#endif
#if defined (REPEAT_S3)
#define repeat_s(a,b,c) a.s0=0; a.s1=0; a.s2=0; b=b; c=c;
#endif
#if defined (REPEAT_S4)
#define repeat_s(a,b,c) a=0;b=b; c=c;
#endif
#if defined (REPEAT_S5)
#define repeat_s(a,b,c) a=0; b.s0=0;c=c;
#endif
#if defined (REPEAT_S6)
#define repeat_s(a,b,c) a=0; b.s0=0; b.s1=0; c=c;
#endif
#if defined (REPEAT_S7)
#define repeat_s(a,b,c) a=0; b.s0=0; b.s1=0; b.s2=0; c=c;
#endif
#if defined (REPEAT_S8)
#define repeat_s(a,b,c) a=0; b=0; c=c;
#endif
#if defined (REPEAT_S9)
#define repeat_s(a,b,c) a=0; b=0; c.s0=0;
#endif
#if defined (REPEAT_S10)
#define repeat_s(a,b,c) a=0; b=0; c.s0=0; c.s1=0;
#endif
#if defined (REPEAT_S11)
#define repeat_s(a,b,c) a=0; b=0; c.s0=0; c.s1=0; c.s2=0;
#endif

#if defined (REPEAT_E0)
#define repeat_e(a,b,c) a=a; b =b; c=c;
#endif
#if defined (REPEAT_E1)
#define repeat_e(a,b,c) a=a; b=b; c.s3=0;
#endif
#if defined (REPEAT_E2)
#define repeat_e(a,b,c) a=a; b=b; c.s3=0; c.s2=0;
#endif
#if defined (REPEAT_E3)
#define repeat_e(a,b,c) a=a; b=b; c.s3=0; c.s2=0; c.s1=0;
#endif
#if defined (REPEAT_E4)
#define repeat_e(a,b,c) a=a; b=b; c=0;
#endif
#if defined (REPEAT_E5)
#define repeat_e(a,b,c) a=a; b.s3=0; c=0;
#endif
#if defined (REPEAT_E6)
#define repeat_e(a,b,c) a=a; b.s3=0; b.s2=0; c=0;
#endif
#if defined (REPEAT_E7)
#define repeat_e(a,b,c) a=a; b.s3=0; b.s2=0; b.s1=0; c=0;
#endif
#if defined (REPEAT_E8)
#define repeat_e(a,b,c) a=a; b=0; c=0;
#endif
#if defined (REPEAT_E9)
#define repeat_e(a,b,c) a.s3=0; b=0; c=0;
#endif
#if defined (REPEAT_E10)
#define repeat_e(a,b,c) a.s3=0; a.s2=0; b=0; c=0;
#endif
#if defined (REPEAT_E11)
#define repeat_e(a,b,c) a.s3=0; a.s2=0; a.s1=0; b=0; c=0;
#endif

__kernel void arithm_op_sum_3 (int cols,int invalid_cols,int offset,int elemnum,int groupnum,  
                                __global VEC_TYPE *src, __global RES_TYPE *dst)
{
   unsigned int lid = get_local_id(0);
   unsigned int gid = get_group_id(0);
   unsigned int id = get_global_id(0);
   unsigned int idx = offset + id + (id  / cols) * invalid_cols;
   idx = idx * 3;
   __local RES_TYPE localmem_sum1[128];
   __local RES_TYPE localmem_sum2[128];
   __local RES_TYPE localmem_sum3[128];
   RES_TYPE sum1 = 0,sum2 = 0,sum3 = 0,temp1,temp2,temp3;
   if(id < elemnum)
   {
       temp1 = CONVERT_RES_TYPE(src[idx]);
       temp2 = CONVERT_RES_TYPE(src[idx+1]);
       temp3 = CONVERT_RES_TYPE(src[idx+2]);
       if(id % cols == 0 ) 
       {
           repeat_s(temp1,temp2,temp3);
       }
       if(id % cols == cols - 1)
       {
           repeat_e(temp1,temp2,temp3);
       }
       FUNC(temp1,sum1);
       FUNC(temp2,sum2);
       FUNC(temp3,sum3);
   }
   else
   {
       sum1 = 0;
       sum2 = 0;
       sum3 = 0;
   }
   for(id=id + (groupnum << 8); id < elemnum;id = id + (groupnum << 8))
   {
       idx = offset + id + (id / cols) * invalid_cols;
       idx = idx * 3;
       temp1 = CONVERT_RES_TYPE(src[idx]);
       temp2 = CONVERT_RES_TYPE(src[idx+1]);
       temp3 = CONVERT_RES_TYPE(src[idx+2]);
       if(id % cols == 0 ) 
       {
               repeat_s(temp1,temp2,temp3);
       }
       if(id % cols == cols - 1)
       {
               repeat_e(temp1,temp2,temp3);
       }
       FUNC(temp1,sum1);
       FUNC(temp2,sum2);
       FUNC(temp3,sum3);
   }
   if(lid > 127)
   {
       localmem_sum1[lid - 128] = sum1;
       localmem_sum2[lid - 128] = sum2;
       localmem_sum3[lid - 128] = sum3;
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   if(lid < 128)
   {
       localmem_sum1[lid] = sum1 + localmem_sum1[lid];
       localmem_sum2[lid] = sum2 + localmem_sum2[lid];
       localmem_sum3[lid] = sum3 + localmem_sum3[lid];
   }
   barrier(CLK_LOCAL_MEM_FENCE);
   for(int lsize = 64; lsize > 0; lsize >>= 1)
   {
       if(lid < lsize)
       {
           int lid2 = lsize + lid;
           localmem_sum1[lid] = localmem_sum1[lid] + localmem_sum1[lid2];
           localmem_sum2[lid] = localmem_sum2[lid] + localmem_sum2[lid2];
           localmem_sum3[lid] = localmem_sum3[lid] + localmem_sum3[lid2];
       }
       barrier(CLK_LOCAL_MEM_FENCE);
   }
   if( lid == 0)
   {
       dst[gid*3]   = localmem_sum1[0];
       dst[gid*3+1] = localmem_sum2[0];
       dst[gid*3+2] = localmem_sum3[0];
   }
}

