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
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
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
// This software is provided by the copyright holders and contributors "as is" and
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

#if defined(cl_khr_fp16)
#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)
#define KERNEL_ARG_DTYPE float

__kernel void TEMPLATE(matvec_mul4,Dtype)(
          __global const Dtype * A,
          int offA,
          unsigned int A_col_size,
          unsigned int trail_item,
          __global const Dtype * v,
          int offv,
          KERNEL_ARG_DTYPE alpha,
          KERNEL_ARG_DTYPE beta,
          __global Dtype4* result,
          int offr,
          __local Dtype4* work)
{
  unsigned int row_gid = get_group_id(0);
  unsigned int lid = get_local_id(0);
  const __global Dtype *src0_read = A + row_gid * 4 * A_col_size + offA;
  const __global Dtype *src1_read = v + offv;
  result = (__global Dtype4*)((__global Dtype*)result + offr);
  Dtype4 dot0 = (Dtype4)(0.f);
  Dtype4 dot1 = (Dtype4)(0.f);
  Dtype4 dot2 = (Dtype4)(0.f);
  Dtype4 dot3 = (Dtype4)(0.f);

  unsigned int i = lid;
  while( i < A_col_size / 4) {
    const Dtype4 a0 = vload4(i, src0_read);
    const Dtype4 a1 = vload4(i, src0_read + A_col_size);
    const Dtype4 a2 = vload4(i, src0_read + 2 * A_col_size);
    const Dtype4 a3 = vload4(i, src0_read + 3 * A_col_size);

    const Dtype4 b0 = vload4(i, src1_read);

    dot0 += a0 * b0;
    dot1 += a1 * b0;
    dot2 += a2 * b0;
    dot3 += a3 * b0;

    i += get_local_size(0);
  }

  work[lid].s0 = dot0.x + dot0.y + dot0.z + dot0.w;
  work[lid].s1 = dot1.x + dot1.y + dot1.z + dot1.w;
  work[lid].s2 = dot2.x + dot2.y + dot2.z + dot2.w;
  work[lid].s3 = dot3.x + dot3.y + dot3.z + dot3.w;

  if(i == A_col_size / 4)
  {
    if(trail_item != 0)
    {
      const __global Dtype *src0_trail = src0_read + i * 4;
      const __global Dtype *src1_trail = src1_read + i * 4;
      for(unsigned int i = 0; i < trail_item; ++i) {
        const Dtype at0 = src0_trail[i];
        const Dtype at1 = src0_trail[i + A_col_size];
        const Dtype at2 = src0_trail[i + 2 * A_col_size];
        const Dtype at3 = src0_trail[i + 3 * A_col_size];

        const Dtype bt = src1_trail[i];

        work[lid].s0 += at0 * bt;
        work[lid].s1 += at1 * bt;
        work[lid].s2 += at2 * bt;
        work[lid].s3 += at3 * bt;
      }
    }

  }

  for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lid < stride)
        work[lid] += work[lid+stride];
  }
  if(lid == 0) {
    if(beta == (Dtype)0)
      result[row_gid] = convert_Dtype(alpha) * work[0];
    else
      result[row_gid] = convert_Dtype(alpha) * work[0] + convert_Dtype(beta) * result[row_gid];
  }
}

/* This kernel used for the trailing rows when row_of_A %4 !=0 */
__kernel void TEMPLATE(matvec_mul1,Dtype)(
          __global const Dtype * A,
          int offA,
          unsigned int A_col_size,
          unsigned int row_offset,
          unsigned int trail_item,
          __global const Dtype * v,
          int offv,
          KERNEL_ARG_DTYPE alpha,
          KERNEL_ARG_DTYPE beta,
          __global Dtype * result,
          int offr,
          __local Dtype * work)
{
  unsigned int row_gid = get_group_id(0);
  unsigned int lid = get_local_id(0);

  const __global Dtype *src0_read = A + (row_offset + row_gid) * A_col_size + offA;
  const __global Dtype *src1_read = v + + offv;
  result = result + offr;
  Dtype4 dot0 = (Dtype4)(0.f);

  unsigned int i = lid;
  while( i < A_col_size / 4)
  {
    const Dtype4 a0 = vload4(i, src0_read);
    const Dtype4 b0 = vload4(i, src1_read);

    dot0 += a0 * b0;
    i += get_local_size(0);
  }

  work[lid] = dot0.x + dot0.y + dot0.z + dot0.w;

  if(i == A_col_size / 4)
  {
    if(trail_item != 0)
    {
      const __global Dtype *src0_trail = src0_read + i * 4;
      const __global Dtype *src1_trail = src1_read + i * 4;
      for(unsigned int i = 0; i < trail_item; ++i) {
        const Dtype at0 = src0_trail[i];
        const Dtype bt = src1_trail[i];

        work[lid] += at0 * bt;
      }
    }

  }
  for(unsigned int stride=get_local_size(0)/2 ; stride>0 ; stride>>=1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lid < stride)
        work[lid] += work[lid+stride];
  }

  if(lid == 0) {
    if(beta == (Dtype)0) {
      result[row_gid+row_offset] = convert_Dtype(alpha) * work[0];
    } else {
      result[row_gid+row_offset] *= convert_Dtype(beta);
      result[row_gid+row_offset] += convert_Dtype(alpha) * work[0];
    }
  }
}
