#include <metal_stdlib>

using namespace metal;

struct fully_connected_params
{
    uint output_count;
    uint inner_size;
    uint output_channels;
};

kernel void kernel_fully_connected_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant fully_connected_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint sample = gid / params.output_channels;
    const uint output_channel = gid % params.output_channels;
    const uint input_base = sample * params.inner_size;
    const uint weight_base = output_channel * params.inner_size;
    float sum = bias[output_channel];
    for (uint i = 0; i < params.inner_size; ++i)
        sum += input[input_base + i] * weights[weight_base + i];
    output[gid] = sum;
}
