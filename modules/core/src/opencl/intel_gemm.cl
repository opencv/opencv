// Copyright (c) 2017, Intel Corporation
//
// The MIT License (MIT)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files(the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions :
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#if defined(cl_intel_subgroups)

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void intelblas_gemm_buffer_NN_sp(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int ldA,
    int ldB,
    int ldC,
    int start_index,
    int stride)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float4 brow;
    float2 arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd;

    const __global float *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * ldA + start_index + off0;

    const __global float *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * ldB + off1;

    float4 dot00 = (start_index != 0) ? vload4(0, dst_write0)           : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0          )) : (float4)(0.0));
    float4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 1 * ldC)) : (float4)(0.0));
    float4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 2 * ldC)) : (float4)(0.0));
    float4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 3 * ldC)) : (float4)(0.0));
    float4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 4 * ldC)) : (float4)(0.0));
    float4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 5 * ldC)) : (float4)(0.0));
    float4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 6 * ldC)) : (float4)(0.0));
    float4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 7 * ldC)) : (float4)(0.0));

    int end_index = min(start_index + stride, K);
    int w = start_index;
    while( w + TILE_K <= end_index ) {
        arow0 = (float)alpha * vload2(0, src0_read + 0 * ldA);
        arow1 = (float)alpha * vload2(0, src0_read + 1 * ldA);
        arow2 = (float)alpha * vload2(0, src0_read + 2 * ldA);
        arow3 = (float)alpha * vload2(0, src0_read + 3 * ldA);
        arow4 = (float)alpha * vload2(0, src0_read + 4 * ldA);
        arow5 = (float)alpha * vload2(0, src0_read + 5 * ldA);
        arow6 = (float)alpha * vload2(0, src0_read + 6 * ldA);
        arow7 = (float)alpha * vload2(0, src0_read + 7 * ldA);

#define MM_DOT_PRODUCT(index, suffix)   \
        brow = vload4(0, src1_read0);  src1_read0 += ldB; \
        dot00 = mad((float4)(intel_sub_group_shuffle(arow0.s##suffix,index)),brow,dot00); \
        dot01 = mad((float4)(intel_sub_group_shuffle(arow1.s##suffix,index)),brow,dot01); \
        dot02 = mad((float4)(intel_sub_group_shuffle(arow2.s##suffix,index)),brow,dot02); \
        dot03 = mad((float4)(intel_sub_group_shuffle(arow3.s##suffix,index)),brow,dot03); \
        dot04 = mad((float4)(intel_sub_group_shuffle(arow4.s##suffix,index)),brow,dot04); \
        dot05 = mad((float4)(intel_sub_group_shuffle(arow5.s##suffix,index)),brow,dot05); \
        dot06 = mad((float4)(intel_sub_group_shuffle(arow6.s##suffix,index)),brow,dot06); \
        dot07 = mad((float4)(intel_sub_group_shuffle(arow7.s##suffix,index)),brow,dot07);

        MM_DOT_PRODUCT(0,0);
        MM_DOT_PRODUCT(0,1);
        MM_DOT_PRODUCT(1,0);
        MM_DOT_PRODUCT(1,1);
        MM_DOT_PRODUCT(2,0);
        MM_DOT_PRODUCT(2,1);
        MM_DOT_PRODUCT(3,0);
        MM_DOT_PRODUCT(3,1);
        MM_DOT_PRODUCT(4,0);
        MM_DOT_PRODUCT(4,1);
        MM_DOT_PRODUCT(5,0);
        MM_DOT_PRODUCT(5,1);
        MM_DOT_PRODUCT(6,0);
        MM_DOT_PRODUCT(6,1);
        MM_DOT_PRODUCT(7,0);
        MM_DOT_PRODUCT(7,1);
#undef MM_DOT_PRODUCT

        src0_read += TILE_K;
        w += TILE_K;
    }

    vstore4(dot00, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot01, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot02, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot03, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot04, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot05, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot06, 0, dst_write0); dst_write0 += ldC;
    vstore4(dot07, 0, dst_write0); dst_write0 += ldC;
}

#undef VEC_SIZE
#undef LWG_HEIGHT
#undef TILE_M
#undef TILE_K
#undef TILE_N

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void intelblas_gemm_buffer_NN(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int ldA,
    int ldB,
    int ldC,
    int start_index,
    int stride)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float4 brow;
    float2 arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd;

    const __global float *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * ldA + start_index + off0;

    const __global float *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * ldB + off1;

    int border = -(group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M);

    int row0 = mad24(global_y, TILE_M, 0) < M ? 0 : border;
    int row1 = mad24(global_y, TILE_M, 1) < M ? 1 : border;
    int row2 = mad24(global_y, TILE_M, 2) < M ? 2 : border;
    int row3 = mad24(global_y, TILE_M, 3) < M ? 3 : border;
    int row4 = mad24(global_y, TILE_M, 4) < M ? 4 : border;
    int row5 = mad24(global_y, TILE_M, 5) < M ? 5 : border;
    int row6 = mad24(global_y, TILE_M, 6) < M ? 6 : border;
    int row7 = mad24(global_y, TILE_M, 7) < M ? 7 : border;

    float4 dot00 = (start_index != 0) ? vload4(0, dst_write0)           : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0          )) : (float4)(0.0));
    float4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 1 * ldC)) : (float4)(0.0));
    float4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 2 * ldC)) : (float4)(0.0));
    float4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 3 * ldC)) : (float4)(0.0));
    float4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 4 * ldC)) : (float4)(0.0));
    float4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 5 * ldC)) : (float4)(0.0));
    float4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 6 * ldC)) : (float4)(0.0));
    float4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 7 * ldC)) : (float4)(0.0));

    int end_index = min(start_index + stride, K);
    int w = start_index;
    while( w + TILE_K <= end_index ) {
        arow0 = (float)alpha * vload2(0, src0_read + row0 * ldA);
        arow1 = (float)alpha * vload2(0, src0_read + row1 * ldA);
        arow2 = (float)alpha * vload2(0, src0_read + row2 * ldA);
        arow3 = (float)alpha * vload2(0, src0_read + row3 * ldA);
        arow4 = (float)alpha * vload2(0, src0_read + row4 * ldA);
        arow5 = (float)alpha * vload2(0, src0_read + row5 * ldA);
        arow6 = (float)alpha * vload2(0, src0_read + row6 * ldA);
        arow7 = (float)alpha * vload2(0, src0_read + row7 * ldA);

#define MM_DOT_PRODUCT(index,suffix) \
        brow = vload4(0, src1_read0);  src1_read0 += ldB; \
        dot00 = mad((float4)(intel_sub_group_shuffle(arow0.s##suffix,index)),brow,dot00); \
        dot01 = mad((float4)(intel_sub_group_shuffle(arow1.s##suffix,index)),brow,dot01); \
        dot02 = mad((float4)(intel_sub_group_shuffle(arow2.s##suffix,index)),brow,dot02); \
        dot03 = mad((float4)(intel_sub_group_shuffle(arow3.s##suffix,index)),brow,dot03); \
        dot04 = mad((float4)(intel_sub_group_shuffle(arow4.s##suffix,index)),brow,dot04); \
        dot05 = mad((float4)(intel_sub_group_shuffle(arow5.s##suffix,index)),brow,dot05); \
        dot06 = mad((float4)(intel_sub_group_shuffle(arow6.s##suffix,index)),brow,dot06); \
        dot07 = mad((float4)(intel_sub_group_shuffle(arow7.s##suffix,index)),brow,dot07);

        MM_DOT_PRODUCT(0,0);
        MM_DOT_PRODUCT(0,1);
        MM_DOT_PRODUCT(1,0);
        MM_DOT_PRODUCT(1,1);
        MM_DOT_PRODUCT(2,0);
        MM_DOT_PRODUCT(2,1);
        MM_DOT_PRODUCT(3,0);
        MM_DOT_PRODUCT(3,1);
        MM_DOT_PRODUCT(4,0);
        MM_DOT_PRODUCT(4,1);
        MM_DOT_PRODUCT(5,0);
        MM_DOT_PRODUCT(5,1);
        MM_DOT_PRODUCT(6,0);
        MM_DOT_PRODUCT(6,1);
        MM_DOT_PRODUCT(7,0);
        MM_DOT_PRODUCT(7,1);
#undef MM_DOT_PRODUCT

        src0_read += TILE_K;
        w += TILE_K;
    }

    if(w < end_index) {
        arow0.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row0 * ldA)[0] : 0.0f;
        arow0.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row0 * ldA)[1] : 0.0f;
        arow1.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row1 * ldA)[0] : 0.0f;
        arow1.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row1 * ldA)[1] : 0.0f;
        arow2.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row2 * ldA)[0] : 0.0f;
        arow2.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row2 * ldA)[1] : 0.0f;
        arow3.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row3 * ldA)[0] : 0.0f;
        arow3.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row3 * ldA)[1] : 0.0f;
        arow4.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row4 * ldA)[0] : 0.0f;
        arow4.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row4 * ldA)[1] : 0.0f;
        arow5.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row5 * ldA)[0] : 0.0f;
        arow5.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row5 * ldA)[1] : 0.0f;
        arow6.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row6 * ldA)[0] : 0.0f;
        arow6.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row6 * ldA)[1] : 0.0f;
        arow7.x = ((w + local_x * 2) < K) ? (float)alpha * (src0_read + row7 * ldA)[0] : 0.0f;
        arow7.y = ((w + local_x * 2 + 1) < K) ? (float)alpha * (src0_read + row7 * ldA)[1] : 0.0f;

#define MM_DOT_PRODUCT(index,suffix)   \
        brow = (w < K) ? vload4(0, src1_read0) : (float)0.0f;  src1_read0 += ldB; w++; \
        dot00 = mad((float4)(intel_sub_group_shuffle( arow0.s##suffix, index )),brow,dot00 ); \
        dot01 = mad((float4)(intel_sub_group_shuffle( arow1.s##suffix, index )),brow,dot01 ); \
        dot02 = mad((float4)(intel_sub_group_shuffle( arow2.s##suffix, index )),brow,dot02 ); \
        dot03 = mad((float4)(intel_sub_group_shuffle( arow3.s##suffix, index )),brow,dot03 ); \
        dot04 = mad((float4)(intel_sub_group_shuffle( arow4.s##suffix, index )),brow,dot04 ); \
        dot05 = mad((float4)(intel_sub_group_shuffle( arow5.s##suffix, index )),brow,dot05 ); \
        dot06 = mad((float4)(intel_sub_group_shuffle( arow6.s##suffix, index )),brow,dot06 ); \
        dot07 = mad((float4)(intel_sub_group_shuffle( arow7.s##suffix, index )),brow,dot07 );

        MM_DOT_PRODUCT(0,0);
        MM_DOT_PRODUCT(0,1);
        MM_DOT_PRODUCT(1,0);
        MM_DOT_PRODUCT(1,1);
        MM_DOT_PRODUCT(2,0);
        MM_DOT_PRODUCT(2,1);
        MM_DOT_PRODUCT(3,0);
        MM_DOT_PRODUCT(3,1);
        MM_DOT_PRODUCT(4,0);
        MM_DOT_PRODUCT(4,1);
        MM_DOT_PRODUCT(5,0);
        MM_DOT_PRODUCT(5,1);
        MM_DOT_PRODUCT(6,0);
        MM_DOT_PRODUCT(6,1);
        MM_DOT_PRODUCT(7,0);
        MM_DOT_PRODUCT(7,1);
#undef MM_DOT_PRODUCT
    }

    if(global_x * 4 < N && global_y * 8 < M) {
        if(mad24(global_x, 4, 3) < N) {
            vstore4(dot00, 0, dst_write0); dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }
        } else if(mad24(global_x, 4, 2) < N) {
            vstore2(dot00.xy, 0, dst_write0);
            dst_write0[2] = dot00.z;
            dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) {
                vstore2(dot01.xy, 0, dst_write0);
                dst_write0[2] = dot01.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                vstore2(dot02.xy, 0, dst_write0);
                dst_write0[2] = dot02.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                vstore2(dot03.xy, 0, dst_write0);
                dst_write0[2] = dot03.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                vstore2(dot04.xy, 0, dst_write0);
                dst_write0[2] = dot04.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                vstore2(dot05.xy, 0, dst_write0);
                dst_write0[2] = dot05.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                vstore2(dot06.xy, 0, dst_write0);
                dst_write0[2] = dot06.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                vstore2(dot07.xy, 0, dst_write0);
                dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            vstore2(dot00.xy, 0, dst_write0); dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }
        } else {
            dst_write0[0] = dot00.x; dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += ldC; }
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
#define LWG_HEIGHT      16
#define TILE_M          8
#define TILE_K          32
#define TILE_N          8
#define SLM_BLOCK       512

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void intelblas_gemm_buffer_NT(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int ldA,
    int ldB,
    int ldC)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float8 dot00 = 0.f;
    float8 dot01 = 0.f;
    float8 dot02 = 0.f;
    float8 dot03 = 0.f;
    float8 dot04 = 0.f;
    float8 dot05 = 0.f;
    float8 dot06 = 0.f;
    float8 dot07 = 0.f;

    float4 brow0;
    float4 brow1;
    float4 brow2;
    float4 brow3;
    float4 brow4;
    float4 brow5;
    float4 brow6;
    float4 brow7;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd;

    const __global float *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * ldA + off0;

    const __global float *src1_read0 = src1 + ( group_x * TILE_N ) * ldB + off1;

    __local float slm_brow[8 * SLM_BLOCK];
    __local float* slm_brow0;

    int local_index = mad24(local_y, 8, local_x) * 4;
    int w;
    for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        vstore4(vload4(0, src1_read0 + mad24(0, ldB, local_index)), 0, slm_brow + mad24(0, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(1, ldB, local_index)), 0, slm_brow + mad24(1, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(2, ldB, local_index)), 0, slm_brow + mad24(2, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(3, ldB, local_index)), 0, slm_brow + mad24(3, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(4, ldB, local_index)), 0, slm_brow + mad24(4, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(5, ldB, local_index)), 0, slm_brow + mad24(5, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(6, ldB, local_index)), 0, slm_brow + mad24(6, SLM_BLOCK, local_index));
        vstore4(vload4(0, src1_read0 + mad24(7, ldB, local_index)), 0, slm_brow + mad24(7, SLM_BLOCK, local_index));
        barrier(CLK_LOCAL_MEM_FENCE);

        slm_brow0 = slm_brow + local_x * (TILE_K / 8);
        w = b_tile;
        int end_w = min(b_tile + SLM_BLOCK, K);
        while( w + TILE_K <= end_w ) {
            float4 arow;

            brow0 = vload4(0, slm_brow0 + 0 * SLM_BLOCK);
            brow1 = vload4(0, slm_brow0 + 1 * SLM_BLOCK);
            brow2 = vload4(0, slm_brow0 + 2 * SLM_BLOCK);
            brow3 = vload4(0, slm_brow0 + 3 * SLM_BLOCK);
            brow4 = vload4(0, slm_brow0 + 4 * SLM_BLOCK);
            brow5 = vload4(0, slm_brow0 + 5 * SLM_BLOCK);
            brow6 = vload4(0, slm_brow0 + 6 * SLM_BLOCK);
            brow7 = vload4(0, slm_brow0 + 7 * SLM_BLOCK);

#define MM_DOT_PRODUCT(_row,_dot)   \
            arow = vload4(0, src0_read + _row * ldA);                           \
            _dot = mad( (float8)(arow.x), (float8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \
            _dot = mad( (float8)(arow.y), (float8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \
            _dot = mad( (float8)(arow.z), (float8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \
            _dot = mad( (float8)(arow.w), (float8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );

            MM_DOT_PRODUCT(0,dot00);
            MM_DOT_PRODUCT(1,dot01);
            MM_DOT_PRODUCT(2,dot02);
            MM_DOT_PRODUCT(3,dot03);
            MM_DOT_PRODUCT(4,dot04);
            MM_DOT_PRODUCT(5,dot05);
            MM_DOT_PRODUCT(6,dot06);
            MM_DOT_PRODUCT(7,dot07);
#undef MM_DOT_PRODUCT

            src0_read += TILE_K;
            slm_brow0 += TILE_K;
            w += TILE_K;
        }
        src1_read0 += SLM_BLOCK;
    }

    if(w < K) {
        float4 arow;

#define READ_BROW(_brow,_row) \
        _brow = vload4(0, slm_brow0 + _row * SLM_BLOCK); \
        _brow.x = (mad24(local_x, 4, w) < K) ? _brow.x : 0.0f; \
        _brow.y = (mad24(local_x, 4, w + 1) < K) ? _brow.y : 0.0f; \
        _brow.z = (mad24(local_x, 4, w + 2) < K) ? _brow.z : 0.0f; \
        _brow.w = (mad24(local_x, 4, w + 3) < K) ? _brow.w : 0.0f;

        READ_BROW(brow0,0);
        READ_BROW(brow1,1);
        READ_BROW(brow2,2);
        READ_BROW(brow3,3);
        READ_BROW(brow4,4);
        READ_BROW(brow5,5);
        READ_BROW(brow6,6);
        READ_BROW(brow7,7);

#define MM_DOT_PRODUCT(_row,_dot)   \
        arow = vload4(0, src0_read + _row * ldA);  \
        arow.x = (mad24(local_x, 4, w) < K) ? arow.x : 0.0f; \
        arow.y = (mad24(local_x, 4, w + 1) < K) ? arow.y : 0.0f; \
        arow.z = (mad24(local_x, 4, w + 2) < K) ? arow.z : 0.0f; \
        arow.w = (mad24(local_x, 4, w + 3) < K) ? arow.w : 0.0f; \
        _dot = mad( (float8)(arow.x), (float8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \
        _dot = mad( (float8)(arow.y), (float8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \
        _dot = mad( (float8)(arow.z), (float8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \
        _dot = mad( (float8)(arow.w), (float8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );

        MM_DOT_PRODUCT(0,dot00);
        MM_DOT_PRODUCT(1,dot01);
        MM_DOT_PRODUCT(2,dot02);
        MM_DOT_PRODUCT(3,dot03);
        MM_DOT_PRODUCT(4,dot04);
        MM_DOT_PRODUCT(5,dot05);
        MM_DOT_PRODUCT(6,dot06);
        MM_DOT_PRODUCT(7,dot07);
#undef MM_DOT_PRODUCT
    }

#define REDUCE(_dot) \
    _dot.s0 = intel_sub_group_shuffle(_dot.s0, 0) + intel_sub_group_shuffle(_dot.s0, 1) + intel_sub_group_shuffle(_dot.s0, 2) + intel_sub_group_shuffle(_dot.s0, 3) +  \
           intel_sub_group_shuffle(_dot.s0, 4) + intel_sub_group_shuffle(_dot.s0, 5) + intel_sub_group_shuffle(_dot.s0, 6) + intel_sub_group_shuffle(_dot.s0, 7); \
    _dot.s1 = intel_sub_group_shuffle(_dot.s1, 0) + intel_sub_group_shuffle(_dot.s1, 1) + intel_sub_group_shuffle(_dot.s1, 2) + intel_sub_group_shuffle(_dot.s1, 3) +  \
           intel_sub_group_shuffle(_dot.s1, 4) + intel_sub_group_shuffle(_dot.s1, 5) + intel_sub_group_shuffle(_dot.s1, 6) + intel_sub_group_shuffle(_dot.s1, 7); \
    _dot.s2 = intel_sub_group_shuffle(_dot.s2, 0) + intel_sub_group_shuffle(_dot.s2, 1) + intel_sub_group_shuffle(_dot.s2, 2) + intel_sub_group_shuffle(_dot.s2, 3) +  \
           intel_sub_group_shuffle(_dot.s2, 4) + intel_sub_group_shuffle(_dot.s2, 5) + intel_sub_group_shuffle(_dot.s2, 6) + intel_sub_group_shuffle(_dot.s2, 7); \
    _dot.s3 = intel_sub_group_shuffle(_dot.s3, 0) + intel_sub_group_shuffle(_dot.s3, 1) + intel_sub_group_shuffle(_dot.s3, 2) + intel_sub_group_shuffle(_dot.s3, 3) +  \
           intel_sub_group_shuffle(_dot.s3, 4) + intel_sub_group_shuffle(_dot.s3, 5) + intel_sub_group_shuffle(_dot.s3, 6) + intel_sub_group_shuffle(_dot.s3, 7); \
    _dot.s4 = intel_sub_group_shuffle(_dot.s4, 0) + intel_sub_group_shuffle(_dot.s4, 1) + intel_sub_group_shuffle(_dot.s4, 2) + intel_sub_group_shuffle(_dot.s4, 3) +  \
           intel_sub_group_shuffle(_dot.s4, 4) + intel_sub_group_shuffle(_dot.s4, 5) + intel_sub_group_shuffle(_dot.s4, 6) + intel_sub_group_shuffle(_dot.s4, 7); \
    _dot.s5 = intel_sub_group_shuffle(_dot.s5, 0) + intel_sub_group_shuffle(_dot.s5, 1) + intel_sub_group_shuffle(_dot.s5, 2) + intel_sub_group_shuffle(_dot.s5, 3) +  \
           intel_sub_group_shuffle(_dot.s5, 4) + intel_sub_group_shuffle(_dot.s5, 5) + intel_sub_group_shuffle(_dot.s5, 6) + intel_sub_group_shuffle(_dot.s5, 7); \
    _dot.s6 = intel_sub_group_shuffle(_dot.s6, 0) + intel_sub_group_shuffle(_dot.s6, 1) + intel_sub_group_shuffle(_dot.s6, 2) + intel_sub_group_shuffle(_dot.s6, 3) +  \
           intel_sub_group_shuffle(_dot.s6, 4) + intel_sub_group_shuffle(_dot.s6, 5) + intel_sub_group_shuffle(_dot.s6, 6) + intel_sub_group_shuffle(_dot.s6, 7); \
    _dot.s7 = intel_sub_group_shuffle(_dot.s7, 0) + intel_sub_group_shuffle(_dot.s7, 1) + intel_sub_group_shuffle(_dot.s7, 2) + intel_sub_group_shuffle(_dot.s7, 3) +  \
           intel_sub_group_shuffle(_dot.s7, 4) + intel_sub_group_shuffle(_dot.s7, 5) + intel_sub_group_shuffle(_dot.s7, 6) + intel_sub_group_shuffle(_dot.s7, 7);

    REDUCE(dot00);
    REDUCE(dot01);
    REDUCE(dot02);
    REDUCE(dot03);
    REDUCE(dot04);
    REDUCE(dot05);
    REDUCE(dot06);
    REDUCE(dot07);
#undef REDUCE

    float output = 0.0f;
#define OUTPUT(_dot) \
    output = (local_x == 0) ? _dot.s0 : output; \
    output = (local_x == 1) ? _dot.s1 : output; \
    output = (local_x == 2) ? _dot.s2 : output; \
    output = (local_x == 3) ? _dot.s3 : output; \
    output = (local_x == 4) ? _dot.s4 : output; \
    output = (local_x == 5) ? _dot.s5 : output; \
    output = (local_x == 6) ? _dot.s6 : output; \
    output = (local_x == 7) ? _dot.s7 : output; \
    if (beta != 0.0) \
        dst_write0[0] = mad(output, (float)alpha, ((float)beta * dst_write0[0])); \
    else \
        dst_write0[0] = output * (float)alpha; \
    dst_write0 += ldC;

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

#undef VEC_SIZE
#undef LWG_HEIGHT
#undef TILE_M
#undef TILE_K
#undef TILE_N
#undef SLM_BLOCK

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void intelblas_gemm_buffer_TN(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int ldA,
    int ldB,
    int ldC,
    int start_index,
    int stride)

{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float4 brow;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd;

    const __global float *src0_read = src0 + (local_x * ( TILE_K / 8 ) + start_index) * ldA + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0;

    const __global float *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * ldB + off1;

    float4 dot00 = (start_index != 0) ? vload4(0, dst_write0)           : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0          )) : (float4)(0.0));
    float4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + 1 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 1 * ldC)) : (float4)(0.0));
    float4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 2 * ldC)) : (float4)(0.0));
    float4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 3 * ldC)) : (float4)(0.0));
    float4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 4 * ldC)) : (float4)(0.0));
    float4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 5 * ldC)) : (float4)(0.0));
    float4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 6 * ldC)) : (float4)(0.0));
    float4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : ((beta != 0.0) ? ((float)beta * vload4(0, dst_write0 + 7 * ldC)) : (float4)(0.0));

    int end_index = min(start_index + stride, K);
    while( start_index + TILE_K <= end_index ) {
        float8 arow0 = (float)alpha * vload8(0, src0_read);
        float8 arow1 = (float)alpha * vload8(0, src0_read + ldA);

#define MM_DOT_PRODUCT(_arow,index) \
        brow = vload4(0, src1_read0);  src1_read0 += ldB; \
        dot00 = mad( (float4)(intel_sub_group_shuffle(_arow.s0, index)), brow, dot00 ); \
        dot01 = mad( (float4)(intel_sub_group_shuffle(_arow.s1, index)), brow, dot01 ); \
        dot02 = mad( (float4)(intel_sub_group_shuffle(_arow.s2, index)), brow, dot02 ); \
        dot03 = mad( (float4)(intel_sub_group_shuffle(_arow.s3, index)), brow, dot03 ); \
        dot04 = mad( (float4)(intel_sub_group_shuffle(_arow.s4, index)), brow, dot04 ); \
        dot05 = mad( (float4)(intel_sub_group_shuffle(_arow.s5, index)), brow, dot05 ); \
        dot06 = mad( (float4)(intel_sub_group_shuffle(_arow.s6, index)), brow, dot06 ); \
        dot07 = mad( (float4)(intel_sub_group_shuffle(_arow.s7, index)), brow, dot07 );

        MM_DOT_PRODUCT(arow0,0);
        MM_DOT_PRODUCT(arow1,0);
        MM_DOT_PRODUCT(arow0,1);
        MM_DOT_PRODUCT(arow1,1);
        MM_DOT_PRODUCT(arow0,2);
        MM_DOT_PRODUCT(arow1,2);
        MM_DOT_PRODUCT(arow0,3);
        MM_DOT_PRODUCT(arow1,3);
        MM_DOT_PRODUCT(arow0,4);
        MM_DOT_PRODUCT(arow1,4);
        MM_DOT_PRODUCT(arow0,5);
        MM_DOT_PRODUCT(arow1,5);
        MM_DOT_PRODUCT(arow0,6);
        MM_DOT_PRODUCT(arow1,6);
        MM_DOT_PRODUCT(arow0,7);
        MM_DOT_PRODUCT(arow1,7);
#undef MM_DOT_PRODUCT

        src0_read += TILE_K * ldA;
        start_index += TILE_K;
    }

    if(start_index < end_index) {
        float8 arow0 = ((start_index + local_x * 2) < K) ? ((float)alpha * vload8(0, src0_read)) : (float)0.0f;
        float8 arow1 = ((start_index + local_x * 2 + 1) < K) ? ((float)alpha * vload8(0, src0_read + ldA)) : (float)0.0f;

#define MM_DOT_PRODUCT(_arow,index) \
        brow = (start_index < K) ? vload4(0, src1_read0) : (float)0.0f;  src1_read0 += ldB; start_index++; \
        dot00 = mad( (float4)(intel_sub_group_shuffle(_arow.s0, index)), brow, dot00 ); \
        dot01 = mad( (float4)(intel_sub_group_shuffle(_arow.s1, index)), brow, dot01 ); \
        dot02 = mad( (float4)(intel_sub_group_shuffle(_arow.s2, index)), brow, dot02 ); \
        dot03 = mad( (float4)(intel_sub_group_shuffle(_arow.s3, index)), brow, dot03 ); \
        dot04 = mad( (float4)(intel_sub_group_shuffle(_arow.s4, index)), brow, dot04 ); \
        dot05 = mad( (float4)(intel_sub_group_shuffle(_arow.s5, index)), brow, dot05 ); \
        dot06 = mad( (float4)(intel_sub_group_shuffle(_arow.s6, index)), brow, dot06 ); \
        dot07 = mad( (float4)(intel_sub_group_shuffle(_arow.s7, index)), brow, dot07 );

        MM_DOT_PRODUCT(arow0,0);
        MM_DOT_PRODUCT(arow1,0);
        MM_DOT_PRODUCT(arow0,1);
        MM_DOT_PRODUCT(arow1,1);
        MM_DOT_PRODUCT(arow0,2);
        MM_DOT_PRODUCT(arow1,2);
        MM_DOT_PRODUCT(arow0,3);
        MM_DOT_PRODUCT(arow1,3);
        MM_DOT_PRODUCT(arow0,4);
        MM_DOT_PRODUCT(arow1,4);
        MM_DOT_PRODUCT(arow0,5);
        MM_DOT_PRODUCT(arow1,5);
        MM_DOT_PRODUCT(arow0,6);
        MM_DOT_PRODUCT(arow1,6);
        MM_DOT_PRODUCT(arow0,7);
        MM_DOT_PRODUCT(arow1,7);
#undef MM_DOT_PRODUCT
    }

    if(global_x * 4 < N && global_y * 8 < M) {
        if(mad24(global_x, 4, 3) < N) {
            vstore4(dot00, 0, dst_write0); dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }
        } else if(mad24(global_x, 4, 2) < N) {
            vstore2(dot00.xy, 0, dst_write0);
            dst_write0[2] = dot00.z;
            dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) {
                vstore2(dot01.xy, 0, dst_write0);
                dst_write0[2] = dot01.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                vstore2(dot02.xy, 0, dst_write0);
                dst_write0[2] = dot02.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                vstore2(dot03.xy, 0, dst_write0);
                dst_write0[2] = dot03.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                vstore2(dot04.xy, 0, dst_write0);
                dst_write0[2] = dot04.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                vstore2(dot05.xy, 0, dst_write0);
                dst_write0[2] = dot05.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                vstore2(dot06.xy, 0, dst_write0);
                dst_write0[2] = dot06.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                vstore2(dot07.xy, 0, dst_write0);
                dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            vstore2(dot00.xy, 0, dst_write0); dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }
        } else {
            dst_write0[0] = dot00.x; dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += ldC; }
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

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void intelblas_gemm_buffer_TT(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int ldA,
    int ldB,
    int ldC,
    int start_index,
    int stride)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float8 dot0 = 0.f;
    float8 dot1 = 0.f;
    float8 dot2 = 0.f;
    float8 dot3 = 0.f;

    float16 brow0;
    float16 brow1;
    float16 brow2;
    float16 brow3;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * ldC + offd;

    const __global float *src0_read = src0 + (local_x * ( TILE_K / 8 ) + start_index) * ldA + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0;

    const __global float *src1_read0 = src1 + (local_x * VEC_SIZE + ( group_x * TILE_N )) * ldB + start_index + off1;

    float4 dot00 = (start_index != 0) ? vload4(0, dst_write0)           : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0          )) : (float4)(0.0));
    float4 dot01 = (start_index != 0) ? vload4(0, dst_write0 + ldC)     : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + ldC    )) : (float4)(0.0));
    float4 dot02 = (start_index != 0) ? vload4(0, dst_write0 + 2 * ldC) : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + 2 * ldC)) : (float4)(0.0));
    float4 dot03 = (start_index != 0) ? vload4(0, dst_write0 + 3 * ldC) : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + 3 * ldC)) : (float4)(0.0));
    float4 dot04 = (start_index != 0) ? vload4(0, dst_write0 + 4 * ldC) : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + 4 * ldC)) : (float4)(0.0));
    float4 dot05 = (start_index != 0) ? vload4(0, dst_write0 + 5 * ldC) : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + 5 * ldC)) : (float4)(0.0));
    float4 dot06 = (start_index != 0) ? vload4(0, dst_write0 + 6 * ldC) : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + 6 * ldC)) : (float4)(0.0));
    float4 dot07 = (start_index != 0) ? vload4(0, dst_write0 + 7 * ldC) : ((beta != 0.0)? ((float)beta * vload4(0, dst_write0 + 7 * ldC)) : (float4)(0.0));

    int end_index = min(start_index + stride, K);
    while( start_index + TILE_K <= end_index ) {
        brow0 = vload16(0, src1_read0);
        brow1 = vload16(0, src1_read0 + ldB);
        brow2 = vload16(0, src1_read0 + 2 * ldB);
        brow3 = vload16(0, src1_read0 + 3 * ldB);

        float8 arow0 = (float)alpha * vload8(0, src0_read);
        float8 arow1 = (float)alpha * vload8(0, src0_read + ldA);

#define DOT_PRODUCT( _dot, _arow, index, _brow) \
        _dot.s0 = mad( intel_sub_group_shuffle( _arow.s0, index ), _brow, _dot.s0 ); \
        _dot.s1 = mad( intel_sub_group_shuffle( _arow.s1, index ), _brow, _dot.s1 ); \
        _dot.s2 = mad( intel_sub_group_shuffle( _arow.s2, index ), _brow, _dot.s2 ); \
        _dot.s3 = mad( intel_sub_group_shuffle( _arow.s3, index ), _brow, _dot.s3 ); \
        _dot.s4 = mad( intel_sub_group_shuffle( _arow.s4, index ), _brow, _dot.s4 ); \
        _dot.s5 = mad( intel_sub_group_shuffle( _arow.s5, index ), _brow, _dot.s5 ); \
        _dot.s6 = mad( intel_sub_group_shuffle( _arow.s6, index ), _brow, _dot.s6 ); \
        _dot.s7 = mad( intel_sub_group_shuffle( _arow.s7, index ), _brow, _dot.s7 );

#define MM_DOT_PRODUCT( _brow, _dot) \
        DOT_PRODUCT(_dot, arow0, 0, _brow.s0); \
        DOT_PRODUCT(_dot, arow1, 0, _brow.s1); \
        DOT_PRODUCT(_dot, arow0, 1, _brow.s2); \
        DOT_PRODUCT(_dot, arow1, 1, _brow.s3); \
        DOT_PRODUCT(_dot, arow0, 2, _brow.s4); \
        DOT_PRODUCT(_dot, arow1, 2, _brow.s5); \
        DOT_PRODUCT(_dot, arow0, 3, _brow.s6); \
        DOT_PRODUCT(_dot, arow1, 3, _brow.s7); \
        DOT_PRODUCT(_dot, arow0, 4, _brow.s8); \
        DOT_PRODUCT(_dot, arow1, 4, _brow.s9); \
        DOT_PRODUCT(_dot, arow0, 5, _brow.sa); \
        DOT_PRODUCT(_dot, arow1, 5, _brow.sb); \
        DOT_PRODUCT(_dot, arow0, 6, _brow.sc); \
        DOT_PRODUCT(_dot, arow1, 6, _brow.sd); \
        DOT_PRODUCT(_dot, arow0, 7, _brow.se); \
        DOT_PRODUCT(_dot, arow1, 7, _brow.sf);

        MM_DOT_PRODUCT( brow0, dot0 );
        MM_DOT_PRODUCT( brow1, dot1 );
        MM_DOT_PRODUCT( brow2, dot2 );
        MM_DOT_PRODUCT( brow3, dot3 );
#undef MM_DOT_PRODUCT
#undef DOT_PRODUCT

        src1_read0 += TILE_K;
        src0_read += TILE_K * ldA;
        start_index += TILE_K;
    }

    if(start_index < end_index) {
        brow0 = vload16(0, src1_read0);  src1_read0 += ldB;
        brow1 = vload16(0, src1_read0);  src1_read0 += ldB;
        brow2 = vload16(0, src1_read0);  src1_read0 += ldB;
        brow3 = vload16(0, src1_read0);

        float8 arow0 = (float)alpha * vload8(0, src0_read);
        float8 arow1 = (float)alpha * vload8(0, src0_read + ldA);

#define DOT_PRODUCT( _dot, _arow, index, _brow) \
        _dot.s0 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s0, index ), _brow, _dot.s0 ) : _dot.s0; \
        _dot.s1 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s1, index ), _brow, _dot.s1 ) : _dot.s1; \
        _dot.s2 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s2, index ), _brow, _dot.s2 ) : _dot.s2; \
        _dot.s3 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s3, index ), _brow, _dot.s3 ) : _dot.s3; \
        _dot.s4 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s4, index ), _brow, _dot.s4 ) : _dot.s4; \
        _dot.s5 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s5, index ), _brow, _dot.s5 ) : _dot.s5; \
        _dot.s6 = (w < K) ? mad( intel_sub_group_shuffle( _arow.s6, index ), _brow, _dot.s6 ) : _dot.s6; \
        _dot.s7 = (w++ < K) ? mad( intel_sub_group_shuffle( _arow.s7, index ), _brow, _dot.s7 ) : _dot.s7;

#define MM_DOT_PRODUCT( _brow, _dot) \
        DOT_PRODUCT(_dot, arow0, 0, _brow.s0); \
        DOT_PRODUCT(_dot, arow1, 0, _brow.s1); \
        DOT_PRODUCT(_dot, arow0, 1, _brow.s2); \
        DOT_PRODUCT(_dot, arow1, 1, _brow.s3); \
        DOT_PRODUCT(_dot, arow0, 2, _brow.s4); \
        DOT_PRODUCT(_dot, arow1, 2, _brow.s5); \
        DOT_PRODUCT(_dot, arow0, 3, _brow.s6); \
        DOT_PRODUCT(_dot, arow1, 3, _brow.s7); \
        DOT_PRODUCT(_dot, arow0, 4, _brow.s8); \
        DOT_PRODUCT(_dot, arow1, 4, _brow.s9); \
        DOT_PRODUCT(_dot, arow0, 5, _brow.sa); \
        DOT_PRODUCT(_dot, arow1, 5, _brow.sb); \
        DOT_PRODUCT(_dot, arow0, 6, _brow.sc); \
        DOT_PRODUCT(_dot, arow1, 6, _brow.sd); \
        DOT_PRODUCT(_dot, arow0, 7, _brow.se); \
        DOT_PRODUCT(_dot, arow1, 7, _brow.sf);

        int w = start_index;
        MM_DOT_PRODUCT( brow0, dot0 );
        w = start_index;
        MM_DOT_PRODUCT( brow1, dot1 );
        w = start_index;
        MM_DOT_PRODUCT( brow2, dot2 );
        w = start_index;
        MM_DOT_PRODUCT( brow3, dot3 );
#undef MM_DOT_PRODUCT
#undef DOT_PRODUCT
    }

    dot00 += (float4)(dot0.s0, dot1.s0, dot2.s0, dot3.s0);
    dot01 += (float4)(dot0.s1, dot1.s1, dot2.s1, dot3.s1);
    dot02 += (float4)(dot0.s2, dot1.s2, dot2.s2, dot3.s2);
    dot03 += (float4)(dot0.s3, dot1.s3, dot2.s3, dot3.s3);
    dot04 += (float4)(dot0.s4, dot1.s4, dot2.s4, dot3.s4);
    dot05 += (float4)(dot0.s5, dot1.s5, dot2.s5, dot3.s5);
    dot06 += (float4)(dot0.s6, dot1.s6, dot2.s6, dot3.s6);
    dot07 += (float4)(dot0.s7, dot1.s7, dot2.s7, dot3.s7);

    if(global_x * 4 < N && global_y * 8 < M) {
        if(mad24(global_x, 4, 3) < N) {
            vstore4(dot00, 0, dst_write0); dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { vstore4(dot01, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore4(dot02, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore4(dot03, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore4(dot04, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore4(dot05, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore4(dot06, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore4(dot07, 0, dst_write0); }
        } else if(mad24(global_x, 4, 2) < N) {
            vstore2(dot00.xy, 0, dst_write0);
            dst_write0[2] = dot00.z;
            dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) {
                vstore2(dot01.xy, 0, dst_write0);
                dst_write0[2] = dot01.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                vstore2(dot02.xy, 0, dst_write0);
                dst_write0[2] = dot02.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                vstore2(dot03.xy, 0, dst_write0);
                dst_write0[2] = dot03.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                vstore2(dot04.xy, 0, dst_write0);
                dst_write0[2] = dot04.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                vstore2(dot05.xy, 0, dst_write0);
                dst_write0[2] = dot05.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                vstore2(dot06.xy, 0, dst_write0);
                dst_write0[2] = dot06.z;
                dst_write0 += ldC;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                vstore2(dot07.xy, 0, dst_write0);
                dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            vstore2(dot00.xy, 0, dst_write0); dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { vstore2(dot01.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { vstore2(dot02.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { vstore2(dot03.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { vstore2(dot04.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { vstore2(dot05.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { vstore2(dot06.xy, 0, dst_write0); dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 7) < M) { vstore2(dot07.xy, 0, dst_write0); }
        } else {
            dst_write0[0] = dot00.x; dst_write0 += ldC;
            if(mad24(global_y, 8, 1) < M) { dst_write0[0] = dot01.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write0[0] = dot02.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write0[0] = dot03.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write0[0] = dot04.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write0[0] = dot05.x; dst_write0 += ldC; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write0[0] = dot06.x; dst_write0 += ldC; }
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

#endif
