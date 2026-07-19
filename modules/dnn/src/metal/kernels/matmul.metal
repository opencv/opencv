#include <metal_stdlib>

using namespace metal;

struct matmul_params
{
    uint output_count;
    uint m;
    uint n;
    uint k;
    uint a_rows;
    uint a_columns;
    uint b_rows;
    uint b_columns;
    uint trans_a;
    uint trans_b;
    float alpha;
    float beta;
};

kernel void kernel_matmul_f32(
    device const float* input_a [[buffer(0)]],
    device const float* input_b [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant matmul_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= params.output_count)
        return;

    const uint output_matrix_size = params.m * params.n;
    const uint batch = gid / output_matrix_size;
    const uint matrix_index = gid % output_matrix_size;
    const uint row = matrix_index / params.n;
    const uint column = matrix_index % params.n;
    const uint input_a_base = batch * params.a_rows * params.a_columns;
    const uint input_b_base = batch * params.b_rows * params.b_columns;

    float sum = 0.0f;
    for (uint inner = 0; inner < params.k; ++inner)
    {
        const uint input_a_index = params.trans_a
            ? input_a_base + inner * params.a_columns + row
            : input_a_base + row * params.a_columns + inner;
        const uint input_b_index = params.trans_b
            ? input_b_base + column * params.b_columns + inner
            : input_b_base + inner * params.b_columns + column;
        sum += input_a[input_a_index] * input_b[input_b_index];
    }
    output[gid] = params.alpha * sum + params.beta * bias[column];
}

constant uint MATMUL_TILE_M = 16;
constant uint MATMUL_TILE_N = 32;
constant uint MATMUL_TILE_K = 16;

kernel void kernel_matmul_tiled_f32(
    device const float* input_a [[buffer(0)]],
    device const float* input_b [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant matmul_params& params [[buffer(4)]],
    uint3 threadgroup_position [[threadgroup_position_in_grid]],
    uint3 thread_position [[thread_position_in_threadgroup]])
{
    threadgroup float input_a_tile[MATMUL_TILE_M * MATMUL_TILE_K];
    threadgroup float input_b_tile[MATMUL_TILE_K * MATMUL_TILE_N];

    const uint batch = threadgroup_position.z;
    const uint row = threadgroup_position.y * MATMUL_TILE_M + thread_position.y;
    const uint column0 = threadgroup_position.x * MATMUL_TILE_N + thread_position.x;
    const uint column1 = column0 + MATMUL_TILE_K;
    const uint input_a_tile_index = thread_position.y * MATMUL_TILE_K + thread_position.x;
    const uint input_b_tile_index = thread_position.y * MATMUL_TILE_N + thread_position.x;
    const uint input_a_base = batch * params.a_rows * params.a_columns;
    const uint input_b_base = batch * params.b_rows * params.b_columns;

    float sum0 = 0.0f;
    float sum1 = 0.0f;
    for (uint tile_start = 0; tile_start < params.k; tile_start += MATMUL_TILE_K)
    {
        const uint input_a_column = tile_start + thread_position.x;
        const uint input_b_row = tile_start + thread_position.y;
        input_a_tile[input_a_tile_index] = row < params.m && input_a_column < params.k
            ? input_a[input_a_base + row * params.a_columns + input_a_column]
            : 0.0f;
        input_b_tile[input_b_tile_index] = input_b_row < params.k && column0 < params.n
            ? input_b[input_b_base + input_b_row * params.b_columns + column0]
            : 0.0f;
        input_b_tile[input_b_tile_index + MATMUL_TILE_K] =
            input_b_row < params.k && column1 < params.n
            ? input_b[input_b_base + input_b_row * params.b_columns + column1]
            : 0.0f;

        threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma unroll
        for (uint inner = 0; inner < MATMUL_TILE_K; ++inner)
        {
            const float input_a_value = input_a_tile[thread_position.y * MATMUL_TILE_K + inner];
            sum0 = fma(input_a_value,
                       input_b_tile[inner * MATMUL_TILE_N + thread_position.x], sum0);
            sum1 = fma(input_a_value,
                       input_b_tile[inner * MATMUL_TILE_N + thread_position.x + MATMUL_TILE_K], sum1);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (row < params.m && column0 < params.n)
    {
        const uint output_index = batch * params.m * params.n + row * params.n + column0;
        output[output_index] = params.alpha * sum0 + params.beta * bias[column0];
    }
    if (row < params.m && column1 < params.n)
    {
        const uint output_index = batch * params.m * params.n + row * params.n + column1;
        output[output_index] = params.alpha * sum1 + params.beta * bias[column1];
    }
}
