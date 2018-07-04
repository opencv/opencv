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
#define TYPE_FLOAT  1
#define TYPE_HALF   2

#if TYPE == TYPE_HALF
#define Dtype  half
#define Dtype2 half2
#define Dtype4 half4
#define Dtype8 half8
#define Dtype16 half16

#define as_Dtype  as_half
#define as_Dtype2 as_half2
#define as_Dtype4 as_half4
#define as_Dtype8 as_half8
#define as_Dtype16 as_half16
#else
#define Dtype  float
#define Dtype2 float2
#define Dtype4 float4
#define Dtype8 float8
#define Dtype16 float16

#define as_Dtype  as_float
#define as_Dtype2 as_float2
#define as_Dtype4 as_float4
#define as_Dtype8 as_float8
#define as_Dtype16 as_float16
#endif

#if TYPE == TYPE_HALF
#define SHUFFLE_TYPE2(val) as_ushort2(val)
#define SHUFFLE_TYPE8(val) as_ushort8(val)
#define SIMD_SIZE_GEMM 16
#else
#define SHUFFLE_TYPE2(val) val
#define SHUFFLE_TYPE8(val) val
#define SIMD_SIZE_GEMM 8
#endif

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#if TYPE == TYPE_HALF
#define TILE_K          32
#define TILE_N          64
#else
#define TILE_K          16
#define TILE_N          32
#endif

__attribute__((reqd_work_group_size(SIMD_SIZE_GEMM, LWG_HEIGHT, 1)))
__attribute__((intel_reqd_sub_group_size(SIMD_SIZE_GEMM)))
__kernel void TEMPLATE(gemm_buffer_NN, Dtype)(
    const __global Dtype *src0, int off0,
    const __global Dtype *src1, int off1,
    __global Dtype *dst, int offd,
    int M,
    int N,
    int K,
    KERNEL_ARG_DTYPE alpha_in,
    KERNEL_ARG_DTYPE beta_in,
    int start_index)
{
    const Dtype alpha = (Dtype)alpha_in;
    const Dtype beta = (Dtype)beta_in;
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    Dtype4 brow;
    Dtype2 arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7;

    __global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global Dtype *src0_read = src0 + local_x * (TILE_K / SIMD_SIZE_GEMM) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * K + start_index + off0;

    const __global Dtype *src1_read0 = src1 + local_x * VEC_SIZE + (group_x * TILE_N) + start_index * N + off1;

    int border = -(group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M);

    int row0 = mad24(global_y, TILE_M, 0) < M ? 0 : border;
    int row1 = mad24(global_y, TILE_M, 1) < M ? 1 : border;
    int row2 = mad24(global_y, TILE_M, 2) < M ? 2 : border;
    int row3 = mad24(global_y, TILE_M, 3) < M ? 3 : border;
    int row4 = mad24(global_y, TILE_M, 4) < M ? 4 : border;
    int row5 = mad24(global_y, TILE_M, 5) < M ? 5 : border;
    int row6 = mad24(global_y, TILE_M, 6) < M ? 6 : border;
    int row7 = mad24(global_y, TILE_M, 7) < M ? 7 : border;

    Dtype4 dot00 = (start_index != 0) ? vload4(0, dst_write0) : beta * vload4(0, dst_write0);
    Dtype4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * N) : beta * vload4(0, dst_write0 + 1 * N);
    Dtype4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * N) : beta * vload4(0, dst_write0 + 2 * N);
    Dtype4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * N) : beta * vload4(0, dst_write0 + 3 * N);
    Dtype4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * N) : beta * vload4(0, dst_write0 + 4 * N);
    Dtype4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * N) : beta * vload4(0, dst_write0 + 5 * N);
    Dtype4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * N) : beta * vload4(0, dst_write0 + 6 * N);
    Dtype4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * N) : beta * vload4(0, dst_write0 + 7 * N);

    int end_index = min(start_index + 256, K);
    int w = start_index;
    while( w + TILE_K <= end_index ) {
        arow0 = alpha * vload2(0, src0_read + row0 * K);
        arow1 = alpha * vload2(0, src0_read + row1 * K);
        arow2 = alpha * vload2(0, src0_read + row2 * K);
        arow3 = alpha * vload2(0, src0_read + row3 * K);
        arow4 = alpha * vload2(0, src0_read + row4 * K);
        arow5 = alpha * vload2(0, src0_read + row5 * K);
        arow6 = alpha * vload2(0, src0_read + row6 * K);
        arow7 = alpha * vload2(0, src0_read + row7 * K);

#define MM_DOT_PRODUCT( index, suffix )   \
        brow = vload4(0, src1_read0);  src1_read0 += N; \
        dot00 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow0), index )).s##suffix), brow, dot00 ); \
        dot01 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow1), index )).s##suffix), brow, dot01 ); \
        dot02 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow2), index )).s##suffix), brow, dot02 ); \
        dot03 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow3), index )).s##suffix), brow, dot03 ); \
        dot04 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow4), index )).s##suffix), brow, dot04 ); \
        dot05 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow5), index )).s##suffix), brow, dot05 ); \
        dot06 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow6), index )).s##suffix), brow, dot06 ); \
        dot07 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow7), index )).s##suffix), brow, dot07 );

        MM_DOT_PRODUCT(0, 0);
        MM_DOT_PRODUCT(0, 1);
        MM_DOT_PRODUCT(1, 0);
        MM_DOT_PRODUCT(1, 1);
        MM_DOT_PRODUCT(2, 0);
        MM_DOT_PRODUCT(2, 1);
        MM_DOT_PRODUCT(3, 0);
        MM_DOT_PRODUCT(3, 1);
        MM_DOT_PRODUCT(4, 0);
        MM_DOT_PRODUCT(4, 1);
        MM_DOT_PRODUCT(5, 0);
        MM_DOT_PRODUCT(5, 1);
        MM_DOT_PRODUCT(6, 0);
        MM_DOT_PRODUCT(6, 1);
        MM_DOT_PRODUCT(7, 0);
        MM_DOT_PRODUCT(7, 1);
#if TYPE == TYPE_HALF
        MM_DOT_PRODUCT(8, 0);
        MM_DOT_PRODUCT(8, 1);
        MM_DOT_PRODUCT(9, 0);
        MM_DOT_PRODUCT(9, 1);
        MM_DOT_PRODUCT(10, 0);
        MM_DOT_PRODUCT(10, 1);
        MM_DOT_PRODUCT(11, 0);
        MM_DOT_PRODUCT(11, 1);
        MM_DOT_PRODUCT(12, 0);
        MM_DOT_PRODUCT(12, 1);
        MM_DOT_PRODUCT(13, 0);
        MM_DOT_PRODUCT(13, 1);
        MM_DOT_PRODUCT(14, 0);
        MM_DOT_PRODUCT(14, 1);
        MM_DOT_PRODUCT(15, 0);
        MM_DOT_PRODUCT(15, 1);
#endif
#undef MM_DOT_PRODUCT

        src0_read += TILE_K;
        w += TILE_K;
    }

    if(w < end_index) {
        arow0.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row0 * K)[0] : 0.0f;
        arow0.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row0 * K)[1] : 0.0f;
        arow1.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row1 * K)[0] : 0.0f;
        arow1.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row1 * K)[1] : 0.0f;
        arow2.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row2 * K)[0] : 0.0f;
        arow2.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row2 * K)[1] : 0.0f;
        arow3.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row3 * K)[0] : 0.0f;
        arow3.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row3 * K)[1] : 0.0f;
        arow4.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row4 * K)[0] : 0.0f;
        arow4.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row4 * K)[1] : 0.0f;
        arow5.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row5 * K)[0] : 0.0f;
        arow5.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row5 * K)[1] : 0.0f;
        arow6.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row6 * K)[0] : 0.0f;
        arow6.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row6 * K)[1] : 0.0f;
        arow7.x = ((w + local_x * 2) < K) ? alpha * (src0_read + row7 * K)[0] : 0.0f;
        arow7.y = ((w + local_x * 2 + 1) < K) ? alpha * (src0_read + row7 * K)[1] : 0.0f;

#define MM_DOT_PRODUCT( index, suffix )   \
        brow = (w < K) ? vload4(0, src1_read0) : (Dtype4)0.0f;  src1_read0 += N; w++; \
        dot00 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow0), index )).s##suffix), brow, dot00 ); \
        dot01 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow1), index )).s##suffix), brow, dot01 ); \
        dot02 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow2), index )).s##suffix), brow, dot02 ); \
        dot03 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow3), index )).s##suffix), brow, dot03 ); \
        dot04 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow4), index )).s##suffix), brow, dot04 ); \
        dot05 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow5), index )).s##suffix), brow, dot05 ); \
        dot06 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow6), index )).s##suffix), brow, dot06 ); \
        dot07 = mad( (Dtype4)(as_Dtype2(intel_sub_group_shuffle( SHUFFLE_TYPE2(arow7), index )).s##suffix), brow, dot07 );

        MM_DOT_PRODUCT(0, 0);
        MM_DOT_PRODUCT(0, 1);
        MM_DOT_PRODUCT(1, 0);
        MM_DOT_PRODUCT(1, 1);
        MM_DOT_PRODUCT(2, 0);
        MM_DOT_PRODUCT(2, 1);
        MM_DOT_PRODUCT(3, 0);
        MM_DOT_PRODUCT(3, 1);
        MM_DOT_PRODUCT(4, 0);
        MM_DOT_PRODUCT(4, 1);
        MM_DOT_PRODUCT(5, 0);
        MM_DOT_PRODUCT(5, 1);
        MM_DOT_PRODUCT(6, 0);
        MM_DOT_PRODUCT(6, 1);
        MM_DOT_PRODUCT(7, 0);
        MM_DOT_PRODUCT(7, 1);
#if TYPE == TYPE_HALF
        MM_DOT_PRODUCT(8, 0);
        MM_DOT_PRODUCT(8, 1);
        MM_DOT_PRODUCT(9, 0);
        MM_DOT_PRODUCT(9, 1);
        MM_DOT_PRODUCT(10, 0);
        MM_DOT_PRODUCT(10, 1);
        MM_DOT_PRODUCT(11, 0);
        MM_DOT_PRODUCT(11, 1);
        MM_DOT_PRODUCT(12, 0);
        MM_DOT_PRODUCT(12, 1);
        MM_DOT_PRODUCT(13, 0);
        MM_DOT_PRODUCT(13, 1);
        MM_DOT_PRODUCT(14, 0);
        MM_DOT_PRODUCT(14, 1);
        MM_DOT_PRODUCT(15, 0);
        MM_DOT_PRODUCT(15, 1);
#endif
#undef MM_DOT_PRODUCT
    }

    if(global_x * 4 < N && global_y * 8 < M) {
        if(mad24(global_x, 4, 3) < N) {
            vstore4(dot00, 0, dst_write0); dst_write0 += N;
            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }
        } else if(mad24(global_x, 4, 2) < N) {
            vstore2(dot00.xy, 0, dst_write0);
            dst_write0[2] = dot00.z;
            dst_write0 += N;
            if(mad24(global_y, 8, 1) < M) {
                vstore2(dot01.xy, 0, dst_write0);
                dst_write0[2] = dot01.z;
                dst_write0 += N;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                vstore2(dot02.xy, 0, dst_write0);
                dst_write0[2] = dot02.z;
                dst_write0 += N;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                vstore2(dot03.xy, 0, dst_write0);
                dst_write0[2] = dot03.z;
                dst_write0 += N;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                vstore2(dot04.xy, 0, dst_write0);
                dst_write0[2] = dot04.z;
                dst_write0 += N;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                vstore2(dot05.xy, 0, dst_write0);
                dst_write0[2] = dot05.z;
                dst_write0 += N;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                vstore2(dot06.xy, 0, dst_write0);
                dst_write0[2] = dot06.z;
                dst_write0 += N;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                vstore2(dot07.xy, 0, dst_write0);
                dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            vstore2(dot00.xy, 0, dst_write0); dst_write0 += N;
            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }
        } else {
            dst_write0[0] = dot00.x; dst_write0 += N;
            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += N; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write0[0] = dot07.x; }
        }
    }
}

#undef VEC_SIZE
#undef LWG_HEIGHT
#undef TILE_M
#undef TILE_K
#undef TILE_N

#define VEC_SIZE        1
#define TILE_M          8
#define TILE_N          8
#define SLM_BLOCK       128

#if TYPE == TYPE_HALF
#define LWG_HEIGHT      2
#define TILE_K          64
#else
#define LWG_HEIGHT      4
#define TILE_K          32
#endif

#if TYPE == TYPE_HALF
__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void TEMPLATE(gemm_buffer_NT, Dtype)(
    const __global Dtype *src0, int off0,
    const __global Dtype *src1, int off1,
    __global Dtype *dst, int offd,
    int M,
    int N,
    int K,
    KERNEL_ARG_DTYPE alpha_in,
    KERNEL_ARG_DTYPE beta_in)
{
    const Dtype alpha = (Dtype)alpha_in;
    const Dtype beta = (Dtype)beta_in;
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    Dtype8 dot00 = 0.f;
    Dtype8 dot01 = 0.f;
    Dtype8 dot02 = 0.f;
    Dtype8 dot03 = 0.f;
    Dtype8 dot04 = 0.f;
    Dtype8 dot05 = 0.f;
    Dtype8 dot06 = 0.f;
    Dtype8 dot07 = 0.f;

    Dtype8 brow0;
    Dtype8 brow1;
    Dtype8 brow2;
    Dtype8 brow3;
    Dtype8 brow4;
    Dtype8 brow5;
    Dtype8 brow6;
    Dtype8 brow7;

    __global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global Dtype *src0_read = src0 + local_x * (TILE_K / 8) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * K + off0;

    const __global Dtype *src1_read0 = src1 + (group_x * TILE_N) * K + off1;

    __local Dtype slm_brow[8 * SLM_BLOCK];
    __local Dtype* slm_brow0;

    int local_index = mad24(local_y, 8, local_x) * 8;
    int w;
    for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(0, K, local_index))), 0, (__local float *)(slm_brow + mad24(0, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(1, K, local_index))), 0, (__local float *)(slm_brow + mad24(1, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(2, K, local_index))), 0, (__local float *)(slm_brow + mad24(2, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(3, K, local_index))), 0, (__local float *)(slm_brow + mad24(3, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(4, K, local_index))), 0, (__local float *)(slm_brow + mad24(4, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(5, K, local_index))), 0, (__local float *)(slm_brow + mad24(5, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(6, K, local_index))), 0, (__local float *)(slm_brow + mad24(6, SLM_BLOCK, local_index)));
        vstore4(vload4(0, (__global float *)(src1_read0 + mad24(7, K, local_index))), 0, (__local float *)(slm_brow + mad24(7, SLM_BLOCK, local_index)));
        barrier(CLK_LOCAL_MEM_FENCE);

        slm_brow0 = slm_brow + local_x * (TILE_K / 8);
        w = b_tile;
        int end_w = min(b_tile + SLM_BLOCK, K);
        while( w + TILE_K <= end_w ) {
            Dtype8 arow;

            brow0 = as_half8(vload4(0, (__local float *)(slm_brow0 + 0 * SLM_BLOCK)));
            brow1 = as_half8(vload4(0, (__local float *)(slm_brow0 + 1 * SLM_BLOCK)));
            brow2 = as_half8(vload4(0, (__local float *)(slm_brow0 + 2 * SLM_BLOCK)));
            brow3 = as_half8(vload4(0, (__local float *)(slm_brow0 + 3 * SLM_BLOCK)));
            brow4 = as_half8(vload4(0, (__local float *)(slm_brow0 + 4 * SLM_BLOCK)));
            brow5 = as_half8(vload4(0, (__local float *)(slm_brow0 + 5 * SLM_BLOCK)));
            brow6 = as_half8(vload4(0, (__local float *)(slm_brow0 + 6 * SLM_BLOCK)));
            brow7 = as_half8(vload4(0, (__local float *)(slm_brow0 + 7 * SLM_BLOCK)));

#define MM_DOT_PRODUCT( _row, _dot )   \
            arow = as_half8(vload4(0, (__global float *)(src0_read + _row * K)));                           \
            _dot = mad( (Dtype8)(arow.s0), (Dtype8)(brow0.s0, brow1.s0, brow2.s0, brow3.s0, brow4.s0, brow5.s0, brow6.s0, brow7.s0), _dot ); \
            _dot = mad( (Dtype8)(arow.s1), (Dtype8)(brow0.s1, brow1.s1, brow2.s1, brow3.s1, brow4.s1, brow5.s1, brow6.s1, brow7.s1), _dot ); \
            _dot = mad( (Dtype8)(arow.s2), (Dtype8)(brow0.s2, brow1.s2, brow2.s2, brow3.s2, brow4.s2, brow5.s2, brow6.s2, brow7.s2), _dot ); \
            _dot = mad( (Dtype8)(arow.s3), (Dtype8)(brow0.s3, brow1.s3, brow2.s3, brow3.s3, brow4.s3, brow5.s3, brow6.s3, brow7.s3), _dot ); \
            _dot = mad( (Dtype8)(arow.s4), (Dtype8)(brow0.s4, brow1.s4, brow2.s4, brow3.s4, brow4.s4, brow5.s4, brow6.s4, brow7.s4), _dot ); \
            _dot = mad( (Dtype8)(arow.s5), (Dtype8)(brow0.s5, brow1.s5, brow2.s5, brow3.s5, brow4.s5, brow5.s5, brow6.s5, brow7.s5), _dot ); \
            _dot = mad( (Dtype8)(arow.s6), (Dtype8)(brow0.s6, brow1.s6, brow2.s6, brow3.s6, brow4.s6, brow5.s6, brow6.s6, brow7.s6), _dot ); \
            _dot = mad( (Dtype8)(arow.s7), (Dtype8)(brow0.s7, brow1.s7, brow2.s7, brow3.s7, brow4.s7, brow5.s7, brow6.s7, brow7.s7), _dot );

            MM_DOT_PRODUCT( 0, dot00 );
            MM_DOT_PRODUCT( 1, dot01 );
            MM_DOT_PRODUCT( 2, dot02 );
            MM_DOT_PRODUCT( 3, dot03 );
            MM_DOT_PRODUCT( 4, dot04 );
            MM_DOT_PRODUCT( 5, dot05 );
            MM_DOT_PRODUCT( 6, dot06 );
            MM_DOT_PRODUCT( 7, dot07 );
#undef MM_DOT_PRODUCT

            src0_read += TILE_K;
            slm_brow0 += TILE_K;
            w += TILE_K;
        }
        src1_read0 += SLM_BLOCK;
    }

    if(w < K) {
        Dtype8 arow;

#define READ_BROW(_brow, _row) \
        _brow = as_half8(vload4(0, (__local float *)(slm_brow0 + _row * SLM_BLOCK))); \
        _brow.s0 = (mad24(local_x, 8, w) < K) ? _brow.s0 : 0.0f; \
        _brow.s1 = (mad24(local_x, 8, w + 1) < K) ? _brow.s1 : 0.0f; \
        _brow.s2 = (mad24(local_x, 8, w + 2) < K) ? _brow.s2 : 0.0f; \
        _brow.s3 = (mad24(local_x, 8, w + 3) < K) ? _brow.s3 : 0.0f; \
        _brow.s4 = (mad24(local_x, 8, w + 4) < K) ? _brow.s4 : 0.0f; \
        _brow.s5 = (mad24(local_x, 8, w + 5) < K) ? _brow.s5 : 0.0f; \
        _brow.s6 = (mad24(local_x, 8, w + 6) < K) ? _brow.s6 : 0.0f; \
        _brow.s7 = (mad24(local_x, 8, w + 7) < K) ? _brow.s7 : 0.0f;

        READ_BROW(brow0, 0);
        READ_BROW(brow1, 1);
        READ_BROW(brow2, 2);
        READ_BROW(brow3, 3);
        READ_BROW(brow4, 4);
        READ_BROW(brow5, 5);
        READ_BROW(brow6, 6);
        READ_BROW(brow7, 7);

#undef READ_BROW

#define MM_DOT_PRODUCT( _row, _dot )   \
        arow = as_half8(vload4(0, (__global float *)(src0_read + _row * K)));                           \
        arow.s0 = (mad24(local_x, 8, w) < K) ? arow.s0 : 0.0f; \
        arow.s1 = (mad24(local_x, 8, w + 1) < K) ? arow.s1 : 0.0f; \
        arow.s2 = (mad24(local_x, 8, w + 2) < K) ? arow.s2 : 0.0f; \
        arow.s3 = (mad24(local_x, 8, w + 3) < K) ? arow.s3 : 0.0f; \
        arow.s4 = (mad24(local_x, 8, w + 4) < K) ? arow.s4 : 0.0f; \
        arow.s5 = (mad24(local_x, 8, w + 5) < K) ? arow.s5 : 0.0f; \
        arow.s6 = (mad24(local_x, 8, w + 6) < K) ? arow.s6 : 0.0f; \
        arow.s7 = (mad24(local_x, 8, w + 7) < K) ? arow.s7 : 0.0f; \
        _dot = mad( (Dtype8)(arow.s0), (Dtype8)(brow0.s0, brow1.s0, brow2.s0, brow3.s0, brow4.s0, brow5.s0, brow6.s0, brow7.s0), _dot ); \
        _dot = mad( (Dtype8)(arow.s1), (Dtype8)(brow0.s1, brow1.s1, brow2.s1, brow3.s1, brow4.s1, brow5.s1, brow6.s1, brow7.s1), _dot ); \
        _dot = mad( (Dtype8)(arow.s2), (Dtype8)(brow0.s2, brow1.s2, brow2.s2, brow3.s2, brow4.s2, brow5.s2, brow6.s2, brow7.s2), _dot ); \
        _dot = mad( (Dtype8)(arow.s3), (Dtype8)(brow0.s3, brow1.s3, brow2.s3, brow3.s3, brow4.s3, brow5.s3, brow6.s3, brow7.s3), _dot ); \
        _dot = mad( (Dtype8)(arow.s4), (Dtype8)(brow0.s4, brow1.s4, brow2.s4, brow3.s4, brow4.s4, brow5.s4, brow6.s4, brow7.s4), _dot ); \
        _dot = mad( (Dtype8)(arow.s5), (Dtype8)(brow0.s5, brow1.s5, brow2.s5, brow3.s5, brow4.s5, brow5.s5, brow6.s5, brow7.s5), _dot ); \
        _dot = mad( (Dtype8)(arow.s6), (Dtype8)(brow0.s6, brow1.s6, brow2.s6, brow3.s6, brow4.s6, brow5.s6, brow6.s6, brow7.s6), _dot ); \
        _dot = mad( (Dtype8)(arow.s7), (Dtype8)(brow0.s7, brow1.s7, brow2.s7, brow3.s7, brow4.s7, brow5.s7, brow6.s7, brow7.s7), _dot );

        MM_DOT_PRODUCT( 0, dot00 );
        MM_DOT_PRODUCT( 1, dot01 );
        MM_DOT_PRODUCT( 2, dot02 );
        MM_DOT_PRODUCT( 3, dot03 );
        MM_DOT_PRODUCT( 4, dot04 );
        MM_DOT_PRODUCT( 5, dot05 );
        MM_DOT_PRODUCT( 6, dot06 );
        MM_DOT_PRODUCT( 7, dot07 );
#undef MM_DOT_PRODUCT
    }

#define REDUCE(_dot) \
    _dot = as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 0)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 1)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 2)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 3)) +  \
           as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 4)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 5)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 6)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 7));

    REDUCE(dot00);
    REDUCE(dot01);
    REDUCE(dot02);
    REDUCE(dot03);
    REDUCE(dot04);
    REDUCE(dot05);
    REDUCE(dot06);
    REDUCE(dot07);
#undef REDUCE

    Dtype output = 0.0f;
#define OUTPUT( _dot) \
    output = (local_x == 0) ? _dot.s0 : output; \
    output = (local_x == 1) ? _dot.s1 : output; \
    output = (local_x == 2) ? _dot.s2 : output; \
    output = (local_x == 3) ? _dot.s3 : output; \
    output = (local_x == 4) ? _dot.s4 : output; \
    output = (local_x == 5) ? _dot.s5 : output; \
    output = (local_x == 6) ? _dot.s6 : output; \
    output = (local_x == 7) ? _dot.s7 : output; \
    dst_write0[0] = mad(output, alpha, beta * dst_write0[0]); \
    dst_write0 += N;

    if(global_x < N && global_y * 8 < M) {
        OUTPUT(dot00);
        if(mad24(global_y, 8, 1) < M) { OUTPUT(dot01); }
        if(mad24(global_y, 8, 2) < M) { OUTPUT(dot02); }
        if(mad24(global_y, 8, 3) < M) { OUTPUT(dot03); }
        if(mad24(global_y, 8, 4) < M) { OUTPUT(dot04); }
        if(mad24(global_y, 8, 5) < M) { OUTPUT(dot05); }
        if(mad24(global_y, 8, 6) < M) { OUTPUT(dot06); }
        if(mad24(global_y, 8, 7) < M) { OUTPUT(dot07); }
    }
#undef OUTPUT
}

#else

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__attribute__((intel_reqd_sub_group_size(8)))
__kernel void TEMPLATE(gemm_buffer_NT, Dtype)(
    const __global Dtype *src0, int off0,
    const __global Dtype *src1, int off1,
    __global Dtype *dst, int offd,
    int M,
    int N,
    int K,
    KERNEL_ARG_DTYPE alpha_in,
    KERNEL_ARG_DTYPE beta_in)
{
    const Dtype alpha = (Dtype)alpha_in;
    const Dtype beta = (Dtype)beta_in;
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    Dtype8 dot00 = 0.f;
    Dtype8 dot01 = 0.f;
    Dtype8 dot02 = 0.f;
    Dtype8 dot03 = 0.f;
    Dtype8 dot04 = 0.f;
    Dtype8 dot05 = 0.f;
    Dtype8 dot06 = 0.f;
    Dtype8 dot07 = 0.f;

    Dtype4 brow0;
    Dtype4 brow1;
    Dtype4 brow2;
    Dtype4 brow3;
    Dtype4 brow4;
    Dtype4 brow5;
    Dtype4 brow6;
    Dtype4 brow7;

    __global Dtype *dst_write0 = dst + local_x * VEC_SIZE + (group_x * TILE_N) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global Dtype *src0_read = src0 + local_x * (TILE_K / 8) + (group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * K + off0;

    const __global Dtype *src1_read0 = src1 + (group_x * TILE_N) * K + off1;

    __local Dtype slm_brow[8 * SLM_BLOCK];
    __local Dtype* slm_brow0;

    int local_index = mad24(local_y, 8, local_x) * 4;
    int w;
    for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        vstore4(vload4(0, src1_read0 + mad24(0, K, local_index)), 0, slm_brow + mad24(0, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(1, K, local_index)), 0, slm_brow + mad24(1, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(2, K, local_index)), 0, slm_brow + mad24(2, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(3, K, local_index)), 0, slm_brow + mad24(3, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(4, K, local_index)), 0, slm_brow + mad24(4, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(5, K, local_index)), 0, slm_brow + mad24(5, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(6, K, local_index)), 0, slm_brow + mad24(6, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(7, K, local_index)), 0, slm_brow + mad24(7, SLM_BLOCK, local_index));
        barrier(CLK_LOCAL_MEM_FENCE);

        slm_brow0 = slm_brow + local_x * (TILE_K / 8);
        w = b_tile;
        int end_w = min(b_tile + SLM_BLOCK, K);
        while( w + TILE_K <= end_w ) {
            Dtype4 arow;

            brow0 = vload4(0, slm_brow0 + 0 * SLM_BLOCK);
            brow1 = vload4(0, slm_brow0 + 1 * SLM_BLOCK);
            brow2 = vload4(0, slm_brow0 + 2 * SLM_BLOCK);
            brow3 = vload4(0, slm_brow0 + 3 * SLM_BLOCK);
            brow4 = vload4(0, slm_brow0 + 4 * SLM_BLOCK);
            brow5 = vload4(0, slm_brow0 + 5 * SLM_BLOCK);
            brow6 = vload4(0, slm_brow0 + 6 * SLM_BLOCK);
            brow7 = vload4(0, slm_brow0 + 7 * SLM_BLOCK);

#define MM_DOT_PRODUCT( _row, _dot )   \
            arow = vload4(0, src0_read + _row * K);                           \
            _dot = mad( (Dtype8)(arow.x), (Dtype8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \
            _dot = mad( (Dtype8)(arow.y), (Dtype8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \
            _dot = mad( (Dtype8)(arow.z), (Dtype8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \
            _dot = mad( (Dtype8)(arow.w), (Dtype8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );

            MM_DOT_PRODUCT( 0, dot00 );
            MM_DOT_PRODUCT( 1, dot01 );
            MM_DOT_PRODUCT( 2, dot02 );
            MM_DOT_PRODUCT( 3, dot03 );
            MM_DOT_PRODUCT( 4, dot04 );
            MM_DOT_PRODUCT( 5, dot05 );
            MM_DOT_PRODUCT( 6, dot06 );
            MM_DOT_PRODUCT( 7, dot07 );
#undef MM_DOT_PRODUCT

            src0_read += TILE_K;
            slm_brow0 += TILE_K;
            w += TILE_K;
        }
        src1_read0 += SLM_BLOCK;
    }

    if(w < K) {
        Dtype4 arow;

#define READ_BROW(_brow, _row) \
        _brow = vload4(0, slm_brow0 + _row * SLM_BLOCK); \
        _brow.x = (mad24(local_x, 4, w) < K) ? _brow.x : 0.0f; \
        _brow.y = (mad24(local_x, 4, w + 1) < K) ? _brow.y : 0.0f; \
        _brow.z = (mad24(local_x, 4, w + 2) < K) ? _brow.z : 0.0f; \
        _brow.w = (mad24(local_x, 4, w + 3) < K) ? _brow.w : 0.0f;

        READ_BROW(brow0, 0);
        READ_BROW(brow1, 1);
        READ_BROW(brow2, 2);
        READ_BROW(brow3, 3);
        READ_BROW(brow4, 4);
        READ_BROW(brow5, 5);
        READ_BROW(brow6, 6);
        READ_BROW(brow7, 7);

#undef READ_BROW

#define MM_DOT_PRODUCT( _row, _dot )   \
        arow = vload4(0, src0_read + _row * K);                           \
        arow.x = (mad24(local_x, 4, w) < K) ? arow.x : 0.0f; \
        arow.y = (mad24(local_x, 4, w + 1) < K) ? arow.y : 0.0f; \
        arow.z = (mad24(local_x, 4, w + 2) < K) ? arow.z : 0.0f; \
        arow.w = (mad24(local_x, 4, w + 3) < K) ? arow.w : 0.0f; \
        _dot = mad( (Dtype8)(arow.x), (Dtype8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \
        _dot = mad( (Dtype8)(arow.y), (Dtype8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \
        _dot = mad( (Dtype8)(arow.z), (Dtype8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \
        _dot = mad( (Dtype8)(arow.w), (Dtype8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );

        MM_DOT_PRODUCT( 0, dot00 );
        MM_DOT_PRODUCT( 1, dot01 );
        MM_DOT_PRODUCT( 2, dot02 );
        MM_DOT_PRODUCT( 3, dot03 );
        MM_DOT_PRODUCT( 4, dot04 );
        MM_DOT_PRODUCT( 5, dot05 );
        MM_DOT_PRODUCT( 6, dot06 );
        MM_DOT_PRODUCT( 7, dot07 );
#undef MM_DOT_PRODUCT
    }

#define REDUCE(_dot) \
    _dot = as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 0)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 1)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 2)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 3)) +  \
           as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 4)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 5)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 6)) + as_Dtype8(intel_sub_group_shuffle(SHUFFLE_TYPE8(_dot), 7));

    REDUCE(dot00);
    REDUCE(dot01);
    REDUCE(dot02);
    REDUCE(dot03);
    REDUCE(dot04);
    REDUCE(dot05);
    REDUCE(dot06);
    REDUCE(dot07);
#undef REDUCE

    Dtype output = 0.0f;
#define OUTPUT( _dot) \
    output = (local_x == 0) ? _dot.s0 : output; \
    output = (local_x == 1) ? _dot.s1 : output; \
    output = (local_x == 2) ? _dot.s2 : output; \
    output = (local_x == 3) ? _dot.s3 : output; \
    output = (local_x == 4) ? _dot.s4 : output; \
    output = (local_x == 5) ? _dot.s5 : output; \
    output = (local_x == 6) ? _dot.s6 : output; \
    output = (local_x == 7) ? _dot.s7 : output; \
    dst_write0[0] = mad(output, alpha, beta * dst_write0[0]); \
    dst_write0 += N;

    if(global_x < N && global_y * 8 < M) {
        OUTPUT(dot00);
        if(mad24(global_y, 8, 1) < M) { OUTPUT(dot01); }
        if(mad24(global_y, 8, 2) < M) { OUTPUT(dot02); }
        if(mad24(global_y, 8, 3) < M) { OUTPUT(dot03); }
        if(mad24(global_y, 8, 4) < M) { OUTPUT(dot04); }
        if(mad24(global_y, 8, 5) < M) { OUTPUT(dot05); }
        if(mad24(global_y, 8, 6) < M) { OUTPUT(dot06); }
        if(mad24(global_y, 8, 7) < M) { OUTPUT(dot07); }
    }
#undef OUTPUT
}
#endif

#undef VEC_SIZE
#undef LWG_HEIGHT
#undef TILE_M
#undef TILE_K
#undef TILE_N
#undef SLM_BLOCK

#define SLM_SIZE 64
void TEMPLATE(gemm_buffer_NT_M_2_edgerows,Dtype)(
                           const __global Dtype* srca_read0,
                           const __global Dtype* srca_read1,
                           const __global Dtype* srcb_read,
                           __local Dtype4* work0,
                           __local Dtype4* work1,
                           int N,
                           int K,
                           int x_gid,
                           int lid,
                           Dtype alpha,
                           Dtype beta,
                           __global Dtype* dstc0,
                           __global Dtype* dstc1)
{
  __local Dtype* work_each0 = (__local Dtype*)work0;
  __local Dtype* work_each1 = (__local Dtype*)work1;

  int rows = N - x_gid * 4;

  Dtype4 dot0[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
  Dtype4 dot1[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};

  int i = lid;
  while( i < K / 4) {
    const Dtype4 b0 = {srca_read0[i*4], srca_read0[(i*4+1)], srca_read0[(i*4+2)], srca_read0[(i*4+3)]};
    const Dtype4 b1 = {srca_read1[i*4], srca_read1[(i*4+1)], srca_read1[(i*4+2)], srca_read1[(i*4+3)]};
#pragma unroll
    for(int j = 0; j < rows; ++j) {
      dot0[j] += b0 * vload4(i, srcb_read + j * K);
      dot1[j] += b1 * vload4(i, srcb_read + j * K);
    }

    i += get_local_size(0);
  }
#pragma unroll
  for(int j = 0; j < rows; ++j) {
    work_each0[lid * 4 + j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;
    work_each1[lid * 4 + j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;
  }

  if(i == K / 4) {
    short tail_items = K % 4;

    if(tail_items != 0) {
      const __global Dtype *srcb_tail = srcb_read + i * 4;
      const __global Dtype *srca_tail0 = srca_read0 + i * 4;
      const __global Dtype *srca_tail1 = srca_read1 + i * 4;
#pragma unroll
      for(short i = 0; i < tail_items; ++i) {
        const Dtype at0 = srca_tail0[i];
        const Dtype at1 = srca_tail1[i];
#pragma unroll
        for(int j = 0; j < rows; ++j) {
          work_each0[lid * 4 + j] += at0 * srcb_tail[i + j * K];
          work_each1[lid * 4 + j] += at1 * srcb_tail[i + j * K];
        }
      }
    }
  }

  for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < stride) {
      work0[lid] += work0[lid+stride];
      work1[lid] += work1[lid+stride];
    }
  }

  if(lid == 0) {
#pragma unroll
    for(int j = 0; j < rows; ++j) {
      dstc0[(x_gid * 4  + j)] = alpha * work_each0[j] + beta * dstc0[(x_gid * 4 + j)];
      dstc1[(x_gid * 4  + j)] = alpha * work_each1[j] + beta * dstc1[(x_gid * 4 + j)];
    }
  }
}

__kernel void TEMPLATE(gemm_buffer_NT_M_2,Dtype)(
          __global const Dtype * A,
          int offA,
          __global const Dtype * B,
          int offB,
          __global Dtype * C,
          int offC,
          int M,
          int N,
          int K,
          KERNEL_ARG_DTYPE alpha_f,
          KERNEL_ARG_DTYPE beta_f)
{
  Dtype alpha = (Dtype)alpha_f;
  Dtype beta = (Dtype)beta_f;
  int x_gid = get_group_id(0);
  int lid = get_local_id(0);

  const __global Dtype *srca_read0 = A + offA;
  const __global Dtype *srca_read1 = srca_read0 + K;

  const __global Dtype *srcb_read = B + x_gid * 4 * K + offB;

  __global Dtype4 *dstc0 = (__global Dtype4*)(C + offC);
  __global Dtype4 *dstc1 = (__global Dtype4*)((__global Dtype*)(dstc0) + N);

  __local Dtype4 work0[SLM_SIZE];
  __local Dtype4 work1[SLM_SIZE];
  __local Dtype* work_each0 = (__local Dtype*)work0;
  __local Dtype* work_each1 = (__local Dtype*)work1;

  if(x_gid == N / 4) {
    TEMPLATE(gemm_buffer_NT_M_2_edgerows,Dtype) \
         (srca_read0, srca_read1, srcb_read, work0, work1, N, K, x_gid, lid, alpha, beta, (__global Dtype*)dstc0, (__global Dtype*)dstc1);
  } else {
    Dtype4 dot0[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
    Dtype4 dot1[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
    int i = lid;
    while( i < K / 4) {
      const Dtype4 b0 = vload4(i, srca_read0);
      const Dtype4 b1 = vload4(i, srca_read1);
#pragma unroll
      for(int j = 0; j < 4; ++j) {
        Dtype4 a = vload4(i, srcb_read + j * K);
        dot0[j] += b0 * a;
        dot1[j] += b1 * a;
      }
      i += get_local_size(0);
    }

#pragma unroll
    for(int j = 0; j < 4; ++j) {
      work_each0[lid * 4 + j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;
      work_each1[lid * 4 + j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;
    }

    if(i == K / 4) {
      short tail_items = K % 4;
      if(tail_items != 0) {
        const __global Dtype *srcb_tail = srcb_read + i * 4;

        const __global Dtype *srca_tail0 = srca_read0 + i * 4;
        const __global Dtype *srca_tail1 = srca_read1 + i * 4;
#pragma unroll
        for(short i = 0; i < tail_items; ++i) {
          const Dtype at0 = srca_tail0[i];
          const Dtype at1 = srca_tail1[i];
#pragma unroll
          for(int j = 0; j < 4; ++j) {
            work_each0[lid * 4 + j] += at0 * srcb_tail[i + j * K];
            work_each1[lid * 4 + j] += at1 * srcb_tail[i + j * K];
          }
        }
      }
    }

    for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lid < stride) {
        work0[lid] += work0[lid+stride];
        work1[lid] += work1[lid+stride];
      }
    }

    if(lid == 0) {
      dstc0[x_gid] = alpha * work0[0] + beta * dstc0[x_gid];
      dstc1[x_gid] = alpha * work1[0] + beta * dstc1[x_gid];
    }
  }
}
#undef SLM_SIZE

#define SLM_SIZE 32
void TEMPLATE(gemm_buffer_NT_M_4_edgerows,Dtype)(
                           const __global Dtype* srca_read0,
                           const __global Dtype* srca_read1,
                           const __global Dtype* srca_read2,
                           const __global Dtype* srca_read3,
                           const __global Dtype* srcb_read,
                           __local Dtype4* work0,
                           __local Dtype4* work1,
                           __local Dtype4* work2,
                           __local Dtype4* work3,
                           int N,
                           int K,
                           int x_gid,
                           int lid,
                           Dtype alpha,
                           Dtype beta,
                           __global Dtype* dstc0,
                           __global Dtype* dstc1,
                           __global Dtype* dstc2,
                           __global Dtype* dstc3)
{
  __local Dtype* work_each0 = (__local Dtype*)(work0 + lid);
  __local Dtype* work_each1 = (__local Dtype*)(work1 + lid);
  __local Dtype* work_each2 = (__local Dtype*)(work2 + lid);
  __local Dtype* work_each3 = (__local Dtype*)(work3 + lid);

  int rows = N - x_gid * 4;

  Dtype4 dot0[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
  Dtype4 dot1[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
  Dtype4 dot2[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
  Dtype4 dot3[3] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};

  int i = lid;
  while( i < K / 4) {
    const Dtype4 a0 = {srca_read0[i*4], srca_read0[(i*4+1)], srca_read0[(i*4+2)], srca_read0[(i*4+3)]};
    const Dtype4 a1 = {srca_read1[i*4], srca_read1[(i*4+1)], srca_read1[(i*4+2)], srca_read1[(i*4+3)]};
    const Dtype4 a2 = {srca_read2[i*4], srca_read2[(i*4+1)], srca_read2[(i*4+2)], srca_read2[(i*4+3)]};
    const Dtype4 a3 = {srca_read3[i*4], srca_read3[(i*4+1)], srca_read3[(i*4+2)], srca_read3[(i*4+3)]};
#pragma unrol
    for(int j = 0; j < rows; ++j) {
      dot0[j] += a0 * vload4(i, srcb_read + j * K);
      dot1[j] += a1 * vload4(i, srcb_read + j * K);
      dot2[j] += a2 * vload4(i, srcb_read + j * K);
      dot3[j] += a3 * vload4(i, srcb_read + j * K);
    }

    i += get_local_size(0);
  }
#pragma unroll
  for(int j = 0; j < rows; ++j) {
    work_each0[j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;
    work_each1[j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;
    work_each2[j] = dot2[j].x + dot2[j].y + dot2[j].z + dot2[j].w;
    work_each3[j] = dot3[j].x + dot3[j].y + dot3[j].z + dot3[j].w;
  }

  if(i == K / 4) {
    short tail_items = K % 4;

    if(tail_items != 0) {
      const __global Dtype *srcb_tail = srcb_read + i * 4;

      const __global Dtype *srca_tail0 = srca_read0 + i * 4;
      const __global Dtype *srca_tail1 = srca_read1 + i * 4;
      const __global Dtype *srca_tail2 = srca_read2 + i * 4;
      const __global Dtype *srca_tail3 = srca_read3 + i * 4;
#pragma unroll
      for(short i = 0; i < tail_items; ++i) {
        const Dtype at0 = srca_tail0[i];
        const Dtype at1 = srca_tail1[i];
        const Dtype at2 = srca_tail2[i];
        const Dtype at3 = srca_tail3[i];
#pragma unroll
        for(int j = 0; j < rows; ++j) {
          work_each0[j] += at0 * srcb_tail[i + j * K];
          work_each1[j] += at1 * srcb_tail[i + j * K];
          work_each2[j] += at2 * srcb_tail[i + j * K];
          work_each3[j] += at3 * srcb_tail[i + j * K];
        }
      }
    }
  }

  for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < stride) {
      work0[lid] += work0[lid+stride];
      work1[lid] += work1[lid+stride];
      work2[lid] += work2[lid+stride];
      work3[lid] += work3[lid+stride];
    }
  }

  if(lid == 0) {
#pragma unroll
    for(int j = 0; j < rows; ++j) {
      dstc0[(x_gid * 4  + j)] = alpha * work_each0[j] + beta * dstc0[(x_gid * 4 + j)];
      dstc1[(x_gid * 4  + j)] = alpha * work_each1[j] + beta * dstc1[(x_gid * 4 + j)];
      dstc2[(x_gid * 4  + j)] = alpha * work_each2[j] + beta * dstc2[(x_gid * 4 + j)];
      dstc3[(x_gid * 4  + j)] = alpha * work_each3[j] + beta * dstc3[(x_gid * 4 + j)];
    }
  }
}

__kernel void TEMPLATE(gemm_buffer_NT_M_4,Dtype)(
          __global const Dtype * A,
          int offA,
          __global const Dtype * B,
          int offB,
          __global Dtype * C,
          int offC,
          int M,
          int N,
          int K,
          KERNEL_ARG_DTYPE alpha_f,
          KERNEL_ARG_DTYPE beta_f)
{
  Dtype alpha = (Dtype)alpha_f;
  Dtype beta = (Dtype)beta_f;
  int x_gid = get_group_id(0);
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  const __global Dtype *srca_read0 = A + offA;
  const __global Dtype *srca_read1 = srca_read0 + K;
  const __global Dtype *srca_read2 = srca_read1 + K;
  const __global Dtype *srca_read3 = srca_read2 + K;

  const __global Dtype *srcb_read = B + x_gid * 4 * K + offB;

  __global Dtype4 *dstc0 = (__global Dtype4*)(C + offC);
  __global Dtype4 *dstc1 = (__global Dtype4*)((__global Dtype*)(dstc0) + N);
  __global Dtype4 *dstc2 = (__global Dtype4*)((__global Dtype*)(dstc1) + N);
  __global Dtype4 *dstc3 = (__global Dtype4*)((__global Dtype*)(dstc2) + N);

  __local Dtype4 work0[SLM_SIZE];
  __local Dtype4 work1[SLM_SIZE];
  __local Dtype4 work2[SLM_SIZE];
  __local Dtype4 work3[SLM_SIZE];
  __local Dtype* work_each0 = (__local Dtype*)(work0 + lid);
  __local Dtype* work_each1 = (__local Dtype*)(work1 + lid);
  __local Dtype* work_each2 = (__local Dtype*)(work2 + lid);
  __local Dtype* work_each3 = (__local Dtype*)(work3 + lid);

  if(x_gid == N / 4) {
    TEMPLATE(gemm_buffer_NT_M_4_edgerows,Dtype) \
         (srca_read0, srca_read1, srca_read2, srca_read3, srcb_read, \
         work0, work1, work2, work3, N, K, x_gid, lid, alpha, beta, \
         (__global Dtype*)dstc0, (__global Dtype*)dstc1, (__global Dtype*)dstc2, (__global Dtype*)dstc3);
  } else {
    Dtype4 dot0[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
    Dtype4 dot1[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
    Dtype4 dot2[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};
    Dtype4 dot3[4] = {(Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.), (Dtype4)(0.)};

    int kid = lid;
    while( kid < K / 4) {
      const Dtype4 b0 = vload4(kid, srca_read0);
      const Dtype4 b1 = vload4(kid, srca_read1);
      const Dtype4 b2 = vload4(kid, srca_read2);
      const Dtype4 b3 = vload4(kid, srca_read3);
#pragma unroll
      for(int j = 0; j < 4; ++j) {
        Dtype4 a = vload4(kid, srcb_read + j * K);
        dot0[j] += b0 * a;
        dot1[j] += b1 * a;
        dot2[j] += b2 * a;
        dot3[j] += b3 * a;
      }
      kid += lsize;
    }
#pragma unroll
    for(int j = 0; j < 4; ++j) {
      work_each0[j] = dot0[j].x + dot0[j].y + dot0[j].z + dot0[j].w;
      work_each1[j] = dot1[j].x + dot1[j].y + dot1[j].z + dot1[j].w;
      work_each2[j] = dot2[j].x + dot2[j].y + dot2[j].z + dot2[j].w;
      work_each3[j] = dot3[j].x + dot3[j].y + dot3[j].z + dot3[j].w;
    }

    if(kid == (K >> 2)) {
      short tail_items = K % 4;
      if(tail_items != 0) {
        int offset = kid << 2;
        const __global Dtype *srcb_tail = srcb_read + offset;

        const __global Dtype *srca_tail0 = srca_read0 + offset;
        const __global Dtype *srca_tail1 = srca_read1 + offset;
        const __global Dtype *srca_tail2 = srca_read2 + offset;
        const __global Dtype *srca_tail3 = srca_read3 + offset;
#pragma unroll
        for(short i = 0; i < tail_items; ++i) {
          const Dtype at0 = srca_tail0[i];
          const Dtype at1 = srca_tail1[i];
          const Dtype at2 = srca_tail2[i];
          const Dtype at3 = srca_tail3[i];
#pragma unroll
          for(int j = 0; j < 4; ++j) {
            work_each0[j] += at0 * srcb_tail[i + j * K];
            work_each1[j] += at1 * srcb_tail[i + j * K];
            work_each2[j] += at2 * srcb_tail[i + j * K];
            work_each3[j] += at3 * srcb_tail[i + j * K];
          }
        }
      }
    }

    for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
      barrier(CLK_LOCAL_MEM_FENCE);
      if(lid < stride) {
        work0[lid] += work0[lid+stride];
        work1[lid] += work1[lid+stride];
        work2[lid] += work2[lid+stride];
        work3[lid] += work3[lid+stride];
      }
    }

    if(lid == 0) {
      dstc0[x_gid] = alpha * work0[0] + beta * dstc0[x_gid];
      dstc1[x_gid] = alpha * work1[0] + beta * dstc1[x_gid];
      dstc2[x_gid] = alpha * work2[0] + beta * dstc2[x_gid];
      dstc3[x_gid] = alpha * work3[0] + beta * dstc3[x_gid];
    }
  }
}
#undef SLM_SIZE

#define SLM_SIZE 16
__kernel void TEMPLATE(gemm_buffer_NT_M_8,Dtype)(
          __global const Dtype * A,
          int offA,
          __global const Dtype * B,
          int offB,
          __global Dtype * C,
          int offC,
          int M,
          int N,
          int K,
          KERNEL_ARG_DTYPE alpha_f,
          KERNEL_ARG_DTYPE beta_f)
{
  Dtype alpha = (Dtype)alpha_f;
  Dtype beta = (Dtype)beta_f;
  int x_gid = get_group_id(0);
  int lid = get_local_id(0);
  int lsize = get_local_size(0);

  const __global Dtype *srca_read0 = A + offA;
  const __global Dtype *srca_read1 = srca_read0 + K;
  const __global Dtype *srca_read2 = srca_read1 + K;
  const __global Dtype *srca_read3 = srca_read2 + K;
  const __global Dtype *srca_read4 = srca_read3 + K;
  const __global Dtype *srca_read5 = srca_read4 + K;
  const __global Dtype *srca_read6 = srca_read5 + K;
  const __global Dtype *srca_read7 = srca_read6 + K;

  const __global Dtype *srcb_read = B + x_gid * K + offB;

  __global Dtype *dstc0 = C + offC;
  __global Dtype *dstc1 = dstc0 + N;
  __global Dtype *dstc2 = dstc1 + N;
  __global Dtype *dstc3 = dstc2 + N;
  __global Dtype *dstc4 = dstc3 + N;
  __global Dtype *dstc5 = dstc4 + N;
  __global Dtype *dstc6 = dstc5 + N;
  __global Dtype *dstc7 = dstc6 + N;

  __local Dtype work0[SLM_SIZE];
  __local Dtype work1[SLM_SIZE];
  __local Dtype work2[SLM_SIZE];
  __local Dtype work3[SLM_SIZE];
  __local Dtype work4[SLM_SIZE];
  __local Dtype work5[SLM_SIZE];
  __local Dtype work6[SLM_SIZE];
  __local Dtype work7[SLM_SIZE];

  Dtype4 dot0 = (Dtype4)(0.);
  Dtype4 dot1 = (Dtype4)(0.);
  Dtype4 dot2 = (Dtype4)(0.);
  Dtype4 dot3 = (Dtype4)(0.);
  Dtype4 dot4 = (Dtype4)(0.);
  Dtype4 dot5 = (Dtype4)(0.);
  Dtype4 dot6 = (Dtype4)(0.);
  Dtype4 dot7 = (Dtype4)(0.);

  int kid = lid;
  while( kid < K / 4) {
    const Dtype4 a0 = vload4(kid, srca_read0);
    const Dtype4 a1 = vload4(kid, srca_read1);
    const Dtype4 a2 = vload4(kid, srca_read2);
    const Dtype4 a3 = vload4(kid, srca_read3);
    const Dtype4 a4 = vload4(kid, srca_read4);
    const Dtype4 a5 = vload4(kid, srca_read5);
    const Dtype4 a6 = vload4(kid, srca_read6);
    const Dtype4 a7 = vload4(kid, srca_read7);
    Dtype4 b = vload4(kid, srcb_read);
    dot0 += a0 * b;
    dot1 += a1 * b;
    dot2 += a2 * b;
    dot3 += a3 * b;
    dot4 += a4 * b;
    dot5 += a5 * b;
    dot6 += a6 * b;
    dot7 += a7 * b;

    kid += lsize;
  }
  work0[lid] = dot0.x + dot0.y + dot0.z + dot0.w;
  work1[lid] = dot1.x + dot1.y + dot1.z + dot1.w;
  work2[lid] = dot2.x + dot2.y + dot2.z + dot2.w;
  work3[lid] = dot3.x + dot3.y + dot3.z + dot3.w;
  work4[lid] = dot4.x + dot4.y + dot4.z + dot4.w;
  work5[lid] = dot5.x + dot5.y + dot5.z + dot5.w;
  work6[lid] = dot6.x + dot6.y + dot6.z + dot6.w;
  work7[lid] = dot7.x + dot7.y + dot7.z + dot7.w;

  if(kid == (K >> 2)) {
    short tail_items = K % 4;
    if(tail_items != 0) {
      int offset = kid << 2;
      const __global Dtype *srcb_tail = srcb_read + offset;

      const __global Dtype *srca_tail0 = srca_read0 + offset;
      const __global Dtype *srca_tail1 = srca_read1 + offset;
      const __global Dtype *srca_tail2 = srca_read2 + offset;
      const __global Dtype *srca_tail3 = srca_read3 + offset;
      const __global Dtype *srca_tail4 = srca_read4 + offset;
      const __global Dtype *srca_tail5 = srca_read5 + offset;
      const __global Dtype *srca_tail6 = srca_read6 + offset;
      const __global Dtype *srca_tail7 = srca_read7 + offset;
#pragma unroll
      for(short item = 0; item < tail_items; ++item) {
        work0[lid] += srca_tail0[item] * srcb_tail[item];
        work1[lid] += srca_tail1[item] * srcb_tail[item];
        work2[lid] += srca_tail2[item] * srcb_tail[item];
        work3[lid] += srca_tail3[item] * srcb_tail[item];
        work4[lid] += srca_tail4[item] * srcb_tail[item];
        work5[lid] += srca_tail5[item] * srcb_tail[item];
        work6[lid] += srca_tail6[item] * srcb_tail[item];
        work7[lid] += srca_tail7[item] * srcb_tail[item];
      }
    }
  }

  for(int stride = get_local_size(0) >> 1; stride > 0 ; stride >>= 1) {
    barrier(CLK_LOCAL_MEM_FENCE);
    if(lid < stride) {
      work0[lid] += work0[lid+stride];
      work1[lid] += work1[lid+stride];
      work2[lid] += work2[lid+stride];
      work3[lid] += work3[lid+stride];
      work4[lid] += work4[lid+stride];
      work5[lid] += work5[lid+stride];
      work6[lid] += work6[lid+stride];
      work7[lid] += work7[lid+stride];
    }
  }

  if(lid == 0) {
    dstc0[x_gid] = alpha * work0[0] + beta * dstc0[x_gid];
    dstc1[x_gid] = alpha * work1[0] + beta * dstc1[x_gid];
    dstc2[x_gid] = alpha * work2[0] + beta * dstc2[x_gid];
    dstc3[x_gid] = alpha * work3[0] + beta * dstc3[x_gid];
    dstc4[x_gid] = alpha * work4[0] + beta * dstc4[x_gid];
    dstc5[x_gid] = alpha * work5[0] + beta * dstc5[x_gid];
    dstc6[x_gid] = alpha * work6[0] + beta * dstc6[x_gid];
    dstc7[x_gid] = alpha * work7[0] + beta * dstc7[x_gid];
  }
}
#undef SLM_SIZE

#undef VEC_SIZE
#undef LWG_HEIGHT
#undef TILE_M
#undef TILE_K
#undef TILE_N
#undef SIMD_SIZE_GEMM
#undef SHUFFLE_TYPE2
#undef SHUFFLE_TYPE8
