#include <metal_stdlib>

using namespace metal;

struct concat_params
{
    uint count;
    uint inner_size;
    uint input_axis_size;
    uint output_axis_size;
    uint axis_offset;
};

kernel void kernel_concat_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant concat_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint inner = gid % params.inner_size;
    const uint axis = (gid / params.inner_size) % params.input_axis_size;
    const uint outer = gid / (params.inner_size * params.input_axis_size);
    const uint output_index =
        (outer * params.output_axis_size + params.axis_offset + axis) * params.inner_size + inner;
    output[output_index] = input[gid];
}
