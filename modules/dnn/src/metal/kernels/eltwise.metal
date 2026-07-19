#include <metal_stdlib>

using namespace metal;

struct eltwise_params
{
    uint count;
};

kernel void kernel_add_f32(
    device const float* input0 [[buffer(0)]],
    device const float* input1 [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant eltwise_params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = input0[gid] + input1[gid];
}
