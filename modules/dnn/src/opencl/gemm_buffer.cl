#define CONCAT(A,B) A##_##B
#define TEMPLATE(name,type) CONCAT(name,type)

#define TYPE_FLOAT 1
#define TYPE_DOUBLE 2

#define Dtype float
#define Dtype4 float4

#if defined(cl_intel_subgroups)
#pragma OPENCL EXTENSION  cl_intel_subgroups : enable
#endif

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void TEMPLATE(gemm_buffer_NN, Dtype)(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int start_index)
{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float4 brow;
    float2 arow0, arow1, arow2, arow3, arow4, arow5, arow6, arow7;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global float *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * K + start_index + off0;

    const __global float *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * N + off1;

    int border = -(group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M);

    int row0 = mad24(global_y, TILE_M, 0) < M ? 0 : border;
    int row1 = mad24(global_y, TILE_M, 1) < M ? 1 : border;
    int row2 = mad24(global_y, TILE_M, 2) < M ? 2 : border;
    int row3 = mad24(global_y, TILE_M, 3) < M ? 3 : border;
    int row4 = mad24(global_y, TILE_M, 4) < M ? 4 : border;
    int row5 = mad24(global_y, TILE_M, 5) < M ? 5 : border;
    int row6 = mad24(global_y, TILE_M, 6) < M ? 6 : border;
    int row7 = mad24(global_y, TILE_M, 7) < M ? 7 : border;

    float4 dot00 = (start_index != 0) ? ((__global float4 *)dst_write0)[0] : beta * ((__global float4 *)dst_write0)[0];
    float4 dot01 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 1 * N))[0] : beta * ((__global float4 *)(dst_write0 + 1 * N))[0];
    float4 dot02 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 2 * N))[0] : beta * ((__global float4 *)(dst_write0 + 2 * N))[0];
    float4 dot03 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 3 * N))[0] : beta * ((__global float4 *)(dst_write0 + 3 * N))[0];
    float4 dot04 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 4 * N))[0] : beta * ((__global float4 *)(dst_write0 + 4 * N))[0];
    float4 dot05 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 5 * N))[0] : beta * ((__global float4 *)(dst_write0 + 5 * N))[0];
    float4 dot06 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 6 * N))[0] : beta * ((__global float4 *)(dst_write0 + 6 * N))[0];
    float4 dot07 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 7 * N))[0] : beta * ((__global float4 *)(dst_write0 + 7 * N))[0];

    int end_index = min(start_index + 256, K);
    int w = start_index;
    while( w + TILE_K <= end_index ) {
        arow0 = alpha * ((__global float2 *)(src0_read + row0 * K))[0];
        arow1 = alpha * ((__global float2 *)(src0_read + row1 * K))[0];
        arow2 = alpha * ((__global float2 *)(src0_read + row2 * K))[0];
        arow3 = alpha * ((__global float2 *)(src0_read + row3 * K))[0];
        arow4 = alpha * ((__global float2 *)(src0_read + row4 * K))[0];
        arow5 = alpha * ((__global float2 *)(src0_read + row5 * K))[0];
        arow6 = alpha * ((__global float2 *)(src0_read + row6 * K))[0];
        arow7 = alpha * ((__global float2 *)(src0_read + row7 * K))[0];

#define MM_DOT_PRODUCT( index, suffix )   \
        brow = ((__global float4 *)src1_read0)[0];  src1_read0 += N; \
        dot00 = mad( (float4)(intel_sub_group_shuffle( arow0, index ).s##suffix), brow, dot00 ); \
        dot01 = mad( (float4)(intel_sub_group_shuffle( arow1, index ).s##suffix), brow, dot01 ); \
        dot02 = mad( (float4)(intel_sub_group_shuffle( arow2, index ).s##suffix), brow, dot02 ); \
        dot03 = mad( (float4)(intel_sub_group_shuffle( arow3, index ).s##suffix), brow, dot03 ); \
        dot04 = mad( (float4)(intel_sub_group_shuffle( arow4, index ).s##suffix), brow, dot04 ); \
        dot05 = mad( (float4)(intel_sub_group_shuffle( arow5, index ).s##suffix), brow, dot05 ); \
        dot06 = mad( (float4)(intel_sub_group_shuffle( arow6, index ).s##suffix), brow, dot06 ); \
        dot07 = mad( (float4)(intel_sub_group_shuffle( arow7, index ).s##suffix), brow, dot07 );

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
        brow = (w < K) ? ((__global float4 *)src1_read0)[0] : 0.0f;  src1_read0 += N; w++; \
        dot00 = mad( (float4)(intel_sub_group_shuffle( arow0, index ).s##suffix), brow, dot00 ); \
        dot01 = mad( (float4)(intel_sub_group_shuffle( arow1, index ).s##suffix), brow, dot01 ); \
        dot02 = mad( (float4)(intel_sub_group_shuffle( arow2, index ).s##suffix), brow, dot02 ); \
        dot03 = mad( (float4)(intel_sub_group_shuffle( arow3, index ).s##suffix), brow, dot03 ); \
        dot04 = mad( (float4)(intel_sub_group_shuffle( arow4, index ).s##suffix), brow, dot04 ); \
        dot05 = mad( (float4)(intel_sub_group_shuffle( arow5, index ).s##suffix), brow, dot05 ); \
        dot06 = mad( (float4)(intel_sub_group_shuffle( arow6, index ).s##suffix), brow, dot06 ); \
        dot07 = mad( (float4)(intel_sub_group_shuffle( arow7, index ).s##suffix), brow, dot07 );

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
#undef MM_DOT_PRODUCT
    }

    if(global_x * 4 < N && global_y * 8 < M) {
        if(mad24(global_x, 4, 3) < N) {
            __global float4 *dst_write = (__global float4 *)dst_write0;
            dst_write[0] = dot00; dst_write0 += N; dst_write = (__global float4 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) { dst_write[0] = dot01; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write[0] = dot02; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write[0] = dot03; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write[0] = dot04; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write[0] = dot05; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write[0] = dot06; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write[0] = dot07; }
        } else if(mad24(global_x, 4, 2) < N) {
            __global float2 *dst_write = (__global float2 *)dst_write0;
            dst_write[0] = dot00.xy;
            dst_write0[2] = dot00.z;
            dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) {
                dst_write[0] = dot01.xy;
                dst_write0[2] = dot01.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                dst_write[0] = dot02.xy;
                dst_write0[2] = dot02.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                dst_write[0] = dot03.xy;
                dst_write0[2] = dot03.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                dst_write[0] = dot04.xy;
                dst_write0[2] = dot04.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                dst_write[0] = dot05.xy;
                dst_write0[2] = dot05.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                dst_write[0] = dot06.xy;
                dst_write0[2] = dot06.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                dst_write[0] = dot07.xy;
                dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            __global float2 *dst_write = (__global float2 *)dst_write0;
            dst_write[0] = dot00.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) { dst_write[0] = dot01.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write[0] = dot02.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write[0] = dot03.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write[0] = dot04.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write[0] = dot05.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write[0] = dot06.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write[0] = dot07.xy; }
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
#define LWG_HEIGHT      16
#define TILE_M          8
#define TILE_K          32
#define TILE_N          8
#define SLM_BLOCK       512

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void TEMPLATE(gemm_buffer_NT, Dtype)(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta)
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

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global float *src0_read = src0 + local_x * ( TILE_K / 8 ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M ) * K + off0;

    const __global float *src1_read0 = src1 + ( group_x * TILE_N ) * K + off1;

    __local float slm_brow[8 * SLM_BLOCK];
    __local float* slm_brow0;

    int local_index = mad24(local_y, 8, local_x) * 4;
    int w;
    for(int b_tile = 0; b_tile < K; b_tile += SLM_BLOCK) {
        barrier(CLK_LOCAL_MEM_FENCE);
        ((__local float4 *)(slm_brow + mad24(0, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(0, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(1, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(1, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(2, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(2, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(3, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(3, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(4, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(4, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(5, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(5, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(6, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(6, K, local_index)))[0];
        ((__local float4 *)(slm_brow + mad24(7, SLM_BLOCK, local_index)))[0] = ((__global float4 *)(src1_read0 + mad24(7, K, local_index)))[0];
        barrier(CLK_LOCAL_MEM_FENCE);

        slm_brow0 = slm_brow + local_x * (TILE_K / 8);
        w = b_tile;
        int end_w = min(b_tile + SLM_BLOCK, K);
        while( w + TILE_K <= end_w ) {
            float4 arow;

            brow0 = ((__local float4 *)(slm_brow0 + 0 * SLM_BLOCK))[0];
            brow1 = ((__local float4 *)(slm_brow0 + 1 * SLM_BLOCK))[0];
            brow2 = ((__local float4 *)(slm_brow0 + 2 * SLM_BLOCK))[0];
            brow3 = ((__local float4 *)(slm_brow0 + 3 * SLM_BLOCK))[0];
            brow4 = ((__local float4 *)(slm_brow0 + 4 * SLM_BLOCK))[0];
            brow5 = ((__local float4 *)(slm_brow0 + 5 * SLM_BLOCK))[0];
            brow6 = ((__local float4 *)(slm_brow0 + 6 * SLM_BLOCK))[0];
            brow7 = ((__local float4 *)(slm_brow0 + 7 * SLM_BLOCK))[0];

#define MM_DOT_PRODUCT( _row, _dot )   \
            arow = ((__global float4 *)(src0_read + _row * K))[0];                           \
            _dot = mad( (float8)(arow.x), (float8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \
            _dot = mad( (float8)(arow.y), (float8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \
            _dot = mad( (float8)(arow.z), (float8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \
            _dot = mad( (float8)(arow.w), (float8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );

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
        float4 arow;

#define READ_BROW(_brow, _row) \
        _brow = ((__local float4 *)(slm_brow0 + _row * SLM_BLOCK))[0]; \
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

#define MM_DOT_PRODUCT( _row, _dot )   \
        arow = ((__global float4 *)(src0_read + _row * K))[0];                           \
        arow.x = (mad24(local_x, 4, w) < K) ? arow.x : 0.0f; \
        arow.y = (mad24(local_x, 4, w + 1) < K) ? arow.y : 0.0f; \
        arow.z = (mad24(local_x, 4, w + 2) < K) ? arow.z : 0.0f; \
        arow.w = (mad24(local_x, 4, w + 3) < K) ? arow.w : 0.0f; \
        _dot = mad( (float8)(arow.x), (float8)(brow0.x, brow1.x, brow2.x, brow3.x, brow4.x, brow5.x, brow6.x, brow7.x), _dot ); \
        _dot = mad( (float8)(arow.y), (float8)(brow0.y, brow1.y, brow2.y, brow3.y, brow4.y, brow5.y, brow6.y, brow7.y), _dot ); \
        _dot = mad( (float8)(arow.z), (float8)(brow0.z, brow1.z, brow2.z, brow3.z, brow4.z, brow5.z, brow6.z, brow7.z), _dot ); \
        _dot = mad( (float8)(arow.w), (float8)(brow0.w, brow1.w, brow2.w, brow3.w, brow4.w, brow5.w, brow6.w, brow7.w), _dot );

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
    _dot = intel_sub_group_shuffle(_dot, 0) + intel_sub_group_shuffle(_dot, 1) + intel_sub_group_shuffle(_dot, 2) + intel_sub_group_shuffle(_dot, 3) +  \
           intel_sub_group_shuffle(_dot, 4) + intel_sub_group_shuffle(_dot, 5) + intel_sub_group_shuffle(_dot, 6) + intel_sub_group_shuffle(_dot, 7);

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
          float alpha_f,
          float beta_f)
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
          float alpha_f,
          float beta_f)
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
          float alpha_f,
          float beta_f)
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

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void TEMPLATE(gemm_buffer_TN, Dtype)(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int start_index)

{
    const int group_x = get_group_id(0);
    const int group_y = get_group_id(1);
    const int local_x = get_local_id(0);
    const int local_y = get_local_id(1);
    const int global_x = get_global_id(0);
    const int global_y = get_global_id(1);

    float4 brow;

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global float *src0_read = src0 + (local_x * ( TILE_K / 8 ) + start_index) * M + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0;

    const __global float *src1_read0 = src1 + local_x * VEC_SIZE + ( group_x * TILE_N ) + start_index * N + off1;

    float4 dot00 = (start_index != 0) ? ((__global float4 *)dst_write0)[0] : (beta * ((__global float4 *)dst_write0)[0]);
    float4 dot01 = (start_index != 0) ? ((__global float4 *)(dst_write0 + N))[0] : (beta * ((__global float4 *)(dst_write0 + N))[0]);
    float4 dot02 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 2 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 2 * N))[0]);
    float4 dot03 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 3 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 3 * N))[0]);
    float4 dot04 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 4 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 4 * N))[0]);
    float4 dot05 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 5 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 5 * N))[0]);
    float4 dot06 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 6 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 6 * N))[0]);
    float4 dot07 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 7 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 7 * N))[0]);

    int end_index = min(start_index + 256, K);
    while( start_index + TILE_K <= end_index ) {
        float8 arow0 = alpha * ((__global float8 *)src0_read)[0];
        float8 arow1 = alpha * ((__global float8 *)(src0_read + M))[0];

#define MM_DOT_PRODUCT( _arow ) \
        brow = ((__global float4 *)src1_read0)[0];  src1_read0 += N; \
        dot00 = mad( (float4)(_arow.s0), brow, dot00 ); \
        dot01 = mad( (float4)(_arow.s1), brow, dot01 ); \
        dot02 = mad( (float4)(_arow.s2), brow, dot02 ); \
        dot03 = mad( (float4)(_arow.s3), brow, dot03 ); \
        dot04 = mad( (float4)(_arow.s4), brow, dot04 ); \
        dot05 = mad( (float4)(_arow.s5), brow, dot05 ); \
        dot06 = mad( (float4)(_arow.s6), brow, dot06 ); \
        dot07 = mad( (float4)(_arow.s7), brow, dot07 );

        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 0 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 0 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 1 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 1 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 2 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 2 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 3 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 3 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 4 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 4 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 5 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 5 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 6 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 6 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 7 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 7 ) );
#undef MM_DOT_PRODUCT

        src0_read += TILE_K * M;
        start_index += TILE_K;
    }

    if(start_index < end_index) {
        float8 arow0 = ((start_index + local_x * 2) < K) ? (alpha * ((__global float8 *)src0_read)[0]) : 0.0f;
        float8 arow1 = ((start_index + local_x * 2 + 1) < K) ? (alpha * ((__global float8 *)(src0_read + M))[0]) : 0.0f;

#define MM_DOT_PRODUCT( _arow ) \
        brow = (start_index < K) ? ((__global float4 *)src1_read0)[0] : 0.0f;  src1_read0 += N; start_index++; \
        dot00 = mad( (float4)(_arow.s0), brow, dot00 ); \
        dot01 = mad( (float4)(_arow.s1), brow, dot01 ); \
        dot02 = mad( (float4)(_arow.s2), brow, dot02 ); \
        dot03 = mad( (float4)(_arow.s3), brow, dot03 ); \
        dot04 = mad( (float4)(_arow.s4), brow, dot04 ); \
        dot05 = mad( (float4)(_arow.s5), brow, dot05 ); \
        dot06 = mad( (float4)(_arow.s6), brow, dot06 ); \
        dot07 = mad( (float4)(_arow.s7), brow, dot07 );

        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 0 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 0 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 1 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 1 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 2 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 2 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 3 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 3 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 4 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 4 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 5 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 5 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 6 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 6 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow0, 7 ) );
        MM_DOT_PRODUCT( intel_sub_group_shuffle( arow1, 7 ) );
#undef MM_DOT_PRODUCT
    }

    if(global_x * 4 < N && global_y * 8 < M) {
        if(mad24(global_x, 4, 3) < N) {
            __global float4 *dst_write = (__global float4 *)dst_write0;
            dst_write[0] = dot00; dst_write0 += N; dst_write = (__global float4 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) { dst_write[0] = dot01; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write[0] = dot02; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write[0] = dot03; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write[0] = dot04; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write[0] = dot05; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write[0] = dot06; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write[0] = dot07; }
        } else if(mad24(global_x, 4, 2) < N) {
            __global float2 *dst_write = (__global float2 *)dst_write0;
            dst_write[0] = dot00.xy; dst_write0[2] = dot00.z;
            dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) {
                dst_write[0] = dot01.xy; dst_write0[2] = dot01.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                dst_write[0] = dot02.xy; dst_write0[2] = dot02.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                dst_write[0] = dot03.xy; dst_write0[2] = dot03.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                dst_write[0] = dot04.xy; dst_write0[2] = dot04.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                dst_write[0] = dot05.xy; dst_write0[2] = dot05.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                dst_write[0] = dot06.xy; dst_write0[2] = dot06.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                dst_write[0] = dot07.xy; dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            __global float2 *dst_write = (__global float2 *)dst_write0;
            dst_write[0] = dot00.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) { dst_write[0] = dot01.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write[0] = dot02.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write[0] = dot03.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write[0] = dot04.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write[0] = dot05.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write[0] = dot06.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write[0] = dot07.xy; }
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

#define VEC_SIZE        4
#define LWG_HEIGHT      4
#define TILE_M          8
#define TILE_K          16
#define TILE_N          32

__attribute__((reqd_work_group_size(8, LWG_HEIGHT, 1)))
__kernel void TEMPLATE(gemm_buffer_TT, Dtype)(
    const __global float *src0, int off0,
    const __global float *src1, int off1,
    __global float *dst, int offd,
    int M,
    int N,
    int K,
    float alpha,
    float beta,
    int start_index)

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

    __global float *dst_write0 = dst + local_x * VEC_SIZE + ( group_x * TILE_N ) + ( group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M) * N + offd;

    const __global float *src0_read = src0 + (local_x * ( TILE_K / 8 ) + start_index) * M + group_y * LWG_HEIGHT * TILE_M + local_y * TILE_M + off0;

    const __global float *src1_read0 = src1 + (local_x * VEC_SIZE + ( group_x * TILE_N )) * K + start_index + off1;

    float4 dot00 = (start_index != 0) ? ((__global float4 *)dst_write0)[0] : (beta * ((__global float4 *)dst_write0)[0]);
    float4 dot01 = (start_index != 0) ? ((__global float4 *)(dst_write0 + N))[0] : (beta * ((__global float4 *)(dst_write0 + N))[0]);
    float4 dot02 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 2 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 2 * N))[0]);
    float4 dot03 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 3 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 3 * N))[0]);
    float4 dot04 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 4 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 4 * N))[0]);
    float4 dot05 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 5 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 5 * N))[0]);
    float4 dot06 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 6 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 6 * N))[0]);
    float4 dot07 = (start_index != 0) ? ((__global float4 *)(dst_write0 + 7 * N))[0] : (beta * ((__global float4 *)(dst_write0 + 7 * N))[0]);

    int end_index = min(start_index + 256, K);
    while( start_index + TILE_K <= end_index ) {
        brow0 = ((__global float16 *)src1_read0)[0];
        brow1 = ((__global float16 *)(src1_read0 + K))[0];
        brow2 = ((__global float16 *)(src1_read0 + 2 * K))[0];
        brow3 = ((__global float16 *)(src1_read0 + 3 * K))[0];

        float8 arow0 = alpha * ((__global float8 *)src0_read)[0];
        float8 arow1 = alpha * ((__global float8 *)(src0_read + M))[0];

#define MM_DOT_PRODUCT( _brow, _dot) \
        _dot = mad( intel_sub_group_shuffle( arow0, 0 ), (float8)_brow.s0, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 0 ), (float8)_brow.s1, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 1 ), (float8)_brow.s2, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 1 ), (float8)_brow.s3, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 2 ), (float8)_brow.s4, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 2 ), (float8)_brow.s5, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 3 ), (float8)_brow.s6, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 3 ), (float8)_brow.s7, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 4 ), (float8)_brow.s8, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 4 ), (float8)_brow.s9, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 5 ), (float8)_brow.sa, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 5 ), (float8)_brow.sb, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 6 ), (float8)_brow.sc, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 6 ), (float8)_brow.sd, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow0, 7 ), (float8)_brow.se, _dot ); \
        _dot = mad( intel_sub_group_shuffle( arow1, 7 ), (float8)_brow.sf, _dot );

        MM_DOT_PRODUCT( brow0, dot0 );
        MM_DOT_PRODUCT( brow1, dot1 );
        MM_DOT_PRODUCT( brow2, dot2 );
        MM_DOT_PRODUCT( brow3, dot3 );
#undef MM_DOT_PRODUCT

        src1_read0 += TILE_K;
        src0_read += TILE_K * M;
        start_index += TILE_K;
    }

    if(start_index < end_index) {
        brow0 = ((__global float16 *)src1_read0)[0];  src1_read0 += K;
        brow1 = ((__global float16 *)src1_read0)[0];  src1_read0 += K;
        brow2 = ((__global float16 *)src1_read0)[0];  src1_read0 += K;
        brow3 = ((__global float16 *)src1_read0)[0];

        float8 arow0 = alpha * ((__global float8 *)src0_read)[0];
        float8 arow1 = alpha * ((__global float8 *)(src0_read + M))[0];

#define MM_DOT_PRODUCT( _brow, _dot) \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 0 ), (float8)_brow.s0, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 0 ), (float8)_brow.s1, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 1 ), (float8)_brow.s2, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 1 ), (float8)_brow.s3, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 2 ), (float8)_brow.s4, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 2 ), (float8)_brow.s5, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 3 ), (float8)_brow.s6, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 3 ), (float8)_brow.s7, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 4 ), (float8)_brow.s8, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 4 ), (float8)_brow.s9, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 5 ), (float8)_brow.sa, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 5 ), (float8)_brow.sb, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 6 ), (float8)_brow.sc, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 6 ), (float8)_brow.sd, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow0, 7 ), (float8)_brow.se, _dot ) : _dot; \
        _dot = (w++ < K) ? mad( intel_sub_group_shuffle( arow1, 7 ), (float8)_brow.sf, _dot ) : _dot;

        int w = start_index;
        MM_DOT_PRODUCT( brow0, dot0 );
        w = start_index;
        MM_DOT_PRODUCT( brow1, dot1 );
        w = start_index;
        MM_DOT_PRODUCT( brow2, dot2 );
        w = start_index;
        MM_DOT_PRODUCT( brow3, dot3 );
#undef MM_DOT_PRODUCT
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
            __global float4 *dst_write = (__global float4 *)dst_write0;
            dst_write[0] = dot00; dst_write0 += N; dst_write = (__global float4 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) { dst_write[0] = dot01; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write[0] = dot02; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write[0] = dot03; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write[0] = dot04; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write[0] = dot05; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write[0] = dot06; dst_write0 += N; dst_write = (__global float4 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write[0] = dot07; }
        } else if(mad24(global_x, 4, 2) < N) {
            __global float2 *dst_write = (__global float2 *)dst_write0;
            dst_write[0] = dot00.xy; dst_write0[2] = dot00.z;
            dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) {
                dst_write[0] = dot01.xy; dst_write0[2] = dot01.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 2) < M) {
                dst_write[0] = dot02.xy; dst_write0[2] = dot02.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 3) < M) {
                dst_write[0] = dot03.xy; dst_write0[2] = dot03.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 4) < M) {
                dst_write[0] = dot04.xy; dst_write0[2] = dot04.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 5) < M) {
                dst_write[0] = dot05.xy; dst_write0[2] = dot05.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 6) < M) {
                dst_write[0] = dot06.xy; dst_write0[2] = dot06.z;
                dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            } else
                return;
            if(mad24(global_y, 8, 7) < M) {
                dst_write[0] = dot07.xy; dst_write0[2] = dot07.z;
            }
        } else if(mad24(global_x, 4, 1) < N) {
            __global float2 *dst_write = (__global float2 *)dst_write0;
            dst_write[0] = dot00.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0;
            if(mad24(global_y, 8, 1) < M) { dst_write[0] = dot01.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 2) < M) { dst_write[0] = dot02.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 3) < M) { dst_write[0] = dot03.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 4) < M) { dst_write[0] = dot04.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 5) < M) { dst_write[0] = dot05.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 6) < M) { dst_write[0] = dot06.xy; dst_write0 += N; dst_write = (__global float2 *)dst_write0; }
            else return;
            if(mad24(global_y, 8, 7) < M) { dst_write[0] = dot07.xy; }
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
