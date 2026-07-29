#include <metal_stdlib>

using namespace metal;

struct affine_params
{
    uint count;
    uint parameter_count;
    uint inner_size;
};

kernel void kernel_affine_f32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant affine_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint parameter = (gid / params.inner_size) % params.parameter_count;
    output[gid] = input[gid] * scale[parameter] + bias[parameter];
}

kernel void kernel_affine_contiguous_f32(
    device const float4* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float4* output [[buffer(3)]],
    constant affine_params& params [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    const uint parameter = row % params.parameter_count;
    const uint vectors_per_row = params.inner_size / 4;
    const uint row_offset = row * vectors_per_row;
    const float4 scale_value = float4(scale[parameter]);
    const float4 bias_value = float4(bias[parameter]);
    for (uint i = lid; i < vectors_per_row; i += threads_per_threadgroup)
        output[row_offset + i] = fma(input[row_offset + i], scale_value, bias_value);
}
