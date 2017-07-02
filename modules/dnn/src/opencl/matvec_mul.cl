#ifndef __OPENCL_VERSION__
#define __kernel
#define __global
#define __constant
#define __local
#define get_global_id(x) 0
#define get_global_size(x) 0
#define get_local_id(x) 0
#define get_local_size(x) 0
#define FLT_MAX 0
#define FLT_MIN 0
#define cl_khr_fp64
#define cl_amd_fp64
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#define CLK_LOCAL_MEM_FENCE
#define CLK_GLOBAL_MEM_FENCE
#define Dtype float
#define barrier(x)
#define atomic_cmpxchg(x, y, z) x
#define signbit(x) x
#define int_tp long
#define uint_tp unsigned long
#define int_tpc long
#define uint_tpc unsigned long
#endif

#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

#define TYPE_FLOAT 1
#define TYPE_DOUBLE 2

#if defined(cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#elif defined(cl_amd_fp64)
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#ifndef DISABLE_DOUBLE_SUPPORT
#define DOUBLE_SUPPORT_AVAILABLE
#endif //DISABLE_DOUBLE_SUPPORT
#endif

#if defined(cl_khr_int64_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics : enable
#define ATOMICS_64_AVAILABLE
#endif

#if defined(cl_khr_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

#if defined(cl_khr_global_int32_base_atomics)
#pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
#define ATOMICS_32_AVAILABLE
#endif

// Types used for parameters, offset computations and so on
#define int_tp int
#define uint_tp unsigned int

// Definitions used to cast the types above as needed
#define int_tpc int
#define uint_tpc unsigned int

#define Dtype float

__kernel void TEMPLATE(matvec_mul4,Dtype)(
          __global const float * A,
          int offA,
          unsigned int A_col_size,
          unsigned int trail_item,
          __global const float * v,
          int offv,
          float alpha,
          float beta,
          __global float4 * result,
          int offr,
          __local float4 * work)
{
  unsigned int row_gid = get_group_id(0);
  unsigned int lid = get_local_id(0);
  const __global float *src0_read = A + row_gid * 4 * A_col_size + offA;
  const __global float *src1_read = v + offv;
  result = (__global float4*)((__global float*)result + offr);
  float4 dot0 = (float4)(0.f);
  float4 dot1 = (float4)(0.f);
  float4 dot2 = (float4)(0.f);
  float4 dot3 = (float4)(0.f);

  unsigned int i = lid;
  while( i < A_col_size / 4) {
    const float4 a0 = vload4(i, src0_read);
    const float4 a1 = vload4(i, src0_read + A_col_size);
    const float4 a2 = vload4(i, src0_read + 2 * A_col_size);
    const float4 a3 = vload4(i, src0_read + 3 * A_col_size);

    const float4 b0 = vload4(i, src1_read);

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
      const __global float *src0_trail = src0_read + i * 4;
      const __global float *src1_trail = src1_read + i * 4;
      for(unsigned int i = 0; i < trail_item; ++i) {
        const float at0 = src0_trail[i];
        const float at1 = src0_trail[i + A_col_size];
        const float at2 = src0_trail[i + 2 * A_col_size];
        const float at3 = src0_trail[i + 3 * A_col_size];

        const float bt = src1_trail[i];

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
      result[row_gid] = alpha * work[0];
    else
      result[row_gid] = alpha * work[0] + beta * result[row_gid];
  }
}

/* This kernel used for the trailing rows when row_of_A %4 !=0 */
__kernel void TEMPLATE(matvec_mul1,Dtype)(
          __global const float * A,
          int offA,
          unsigned int A_col_size,
          unsigned int row_offset,
          unsigned int trail_item,
          __global const float * v,
          int offv,
          float alpha,
          float beta,
          __global float * result,
          int offr,
          __local float * work)
{
  unsigned int row_gid = get_group_id(0);
  unsigned int lid = get_local_id(0);

  const __global float *src0_read = A + (row_offset + row_gid) * A_col_size + offA;
  const __global float *src1_read = v + + offv;
  result = result + offr;
  float4 dot0 = (float4)(0.f);

  unsigned int i = lid;
  while( i < A_col_size / 4)
  {
    const float4 a0 = vload4(i, src0_read);
    const float4 b0 = vload4(i, src1_read);

    dot0 += a0 * b0;
    i += get_local_size(0);
  }

  work[lid] = dot0.x + dot0.y + dot0.z + dot0.w;

  if(i == A_col_size / 4)
  {
    if(trail_item != 0)
    {
      const __global float *src0_trail = src0_read + i * 4;
      const __global float *src1_trail = src1_read + i * 4;
      for(unsigned int i = 0; i < trail_item; ++i) {
        const float at0 = src0_trail[i];
        const float bt = src1_trail[i];

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
      result[row_gid+row_offset] = alpha * work[0];
    } else {
      result[row_gid+row_offset] *= beta;
      result[row_gid+row_offset] += alpha * work[0];
    }
  }
}
