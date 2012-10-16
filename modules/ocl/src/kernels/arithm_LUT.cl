//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science, all rights reserved.
// Copyright (C) 2010-2012, Advanced Micro Devices, Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// @Authors
//    Niko Li, newlife20080214@gmail.com
//    Rock Li, Rock.li@amd.com
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
//

#if defined (DOUBLE_SUPPORT)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

__kernel
void LUT_C1_D0( __global uchar *dst,
      __global const uchar *src,
      __constant uchar *table,
      int rows,
      int cols,
      int channels,
      int whole_rows,
      int whole_cols,
      int src_offset,
      int dst_offset,
      int lut_offset,
      int src_step,
      int dst_step)
{
    int gidx = get_global_id(0)<<2;
    int gidy = get_global_id(1);
    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local uchar l[256];
    l[(lidy<<4)+lidx] = table[(lidy<<4)+lidx+lut_offset];
    //mem_fence(CLK_LOCAL_MEM_FENCE);


    //clamp(gidx,mask,cols-1);
    gidx = gidx >= cols-4?cols-4:gidx;
    gidy = gidy >= rows?rows-1:gidy;

    int src_index = src_offset + mad24(gidy,src_step,gidx);
    int dst_index = dst_offset + mad24(gidy,dst_step,gidx);
    uchar4 p,q;
    barrier(CLK_LOCAL_MEM_FENCE);
    p.x = src[src_index];
    p.y = src[src_index+1];
    p.z = src[src_index+2];
    p.w = src[src_index+3];

    q.x = l[p.x];
    q.y = l[p.y];
    q.z = l[p.z];
    q.w = l[p.w];
    *(__global uchar4*)(dst + dst_index) = q;
}

__kernel
void LUT2_C1_D0( __global uchar *dst,
      __global const uchar *src,
      __constant uchar *table,
      int rows,
      int precols,
      int channels,
      int whole_rows,
      int cols,
      int src_offset,
      int dst_offset,
      int lut_offset,
      int src_step,
      int dst_step)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);
    //int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    __local uchar l[256];
    l[lidy] = table[lidy+lut_offset];
    //mem_fence(CLK_LOCAL_MEM_FENCE);


    //clamp(gidx,mask,cols-1);
    gidx = gidx >= precols ? cols+gidx : gidx;
    gidy = gidy >= rows?rows-1:gidy;

    int src_index = src_offset + mad24(gidy,src_step,gidx);
    int dst_index = dst_offset + mad24(gidy,dst_step,gidx);
    //uchar4 p,q;
    barrier(CLK_LOCAL_MEM_FENCE);
    uchar p = src[src_index];
    uchar q = l[p];
    dst[dst_index] = q;
}

__kernel
void LUT_C4_D0( __global uchar4 *dst,
      __global uchar4 *src,
      __constant uchar *table,
      int rows,
      int cols,
      int channels,
      int whole_rows,
      int whole_cols,
      int src_offset,
      int dst_offset,
      int lut_offset,
      int src_step,
      int dst_step)
{
    int gidx = get_global_id(0);
    int gidy = get_global_id(1);

    int lidx = get_local_id(0);
    int lidy = get_local_id(1);

    int src_index = mad24(gidy,src_step,gidx+src_offset);
    int dst_index = mad24(gidy,dst_step,gidx+dst_offset);
    __local uchar l[256];
    l[lidy*16+lidx] = table[lidy*16+lidx+lut_offset];
    //mem_fence(CLK_LOCAL_MEM_FENCE);
    barrier(CLK_LOCAL_MEM_FENCE);

    if(gidx<cols && gidy<rows)
    {
        uchar4 p = src[src_index];
        uchar4 q;
        q.x = l[p.x];
        q.y = l[p.y];
        q.z = l[p.z];
        q.w = l[p.w];
        dst[dst_index] = q;
    }
}
