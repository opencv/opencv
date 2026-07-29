#include <metal_stdlib>

using namespace metal;

struct reshape_params
{
    uint count;
};

kernel void kernel_reshape_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant reshape_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = input[gid];
}
