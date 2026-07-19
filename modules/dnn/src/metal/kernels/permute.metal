#include <metal_stdlib>

using namespace metal;

constant uint METAL_MAX_RANK = 8;

struct permute_params
{
    uint count;
    uint rank;
    uint order[METAL_MAX_RANK];
    uint input_stride[METAL_MAX_RANK];
    uint output_stride[METAL_MAX_RANK];
};

kernel void kernel_permute_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant permute_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    uint remainder = gid;
    uint input_index = 0;
    for (uint output_axis = 0; output_axis < params.rank; ++output_axis)
    {
        const uint coordinate = remainder / params.output_stride[output_axis];
        remainder %= params.output_stride[output_axis];
        input_index += coordinate * params.input_stride[params.order[output_axis]];
    }
    output[gid] = input[input_index];
}
