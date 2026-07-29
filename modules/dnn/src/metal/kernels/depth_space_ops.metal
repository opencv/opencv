#include <metal_stdlib>

using namespace metal;

struct depth_space_params
{
    uint count;
    uint input_channels;
    uint input_height;
    uint input_width;
    uint output_channels;
    uint output_height;
    uint output_width;
    uint block_size;
    uint element_size;
    uint mode;
};

inline uint depth_to_space_input_index(
    constant depth_space_params& params, uint gid)
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint output_channel = (gid / output_plane_size) % params.output_channels;
    const uint batch = gid / (output_plane_size * params.output_channels);
    const uint block_x = output_x % params.block_size;
    const uint block_y = output_y % params.block_size;
    const uint input_x = output_x / params.block_size;
    const uint input_y = output_y / params.block_size;
    const uint input_channel = params.mode == 1
        ? (output_channel * params.block_size + block_y) * params.block_size + block_x
        : (block_y * params.block_size + block_x) * params.output_channels + output_channel;
    return ((batch * params.input_channels + input_channel) * params.input_height + input_y) *
           params.input_width + input_x;
}

inline uint space_to_depth_input_index(
    constant depth_space_params& params, uint gid)
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint output_channel = (gid / output_plane_size) % params.output_channels;
    const uint batch = gid / (output_plane_size * params.output_channels);
    const uint block_index = output_channel / params.input_channels;
    const uint input_channel = output_channel % params.input_channels;
    const uint block_x = block_index % params.block_size;
    const uint block_y = block_index / params.block_size;
    const uint input_x = output_x * params.block_size + block_x;
    const uint input_y = output_y * params.block_size + block_y;
    return ((batch * params.input_channels + input_channel) * params.input_height + input_y) *
           params.input_width + input_x;
}

#define DEFINE_DEPTH_TO_SPACE_KERNEL(NAME, TYPE) \
kernel void NAME( \
    device const TYPE* input [[buffer(0)]], \
    device TYPE* output [[buffer(1)]], \
    constant depth_space_params& params [[buffer(2)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    output[gid] = input[depth_to_space_input_index(params, gid)]; \
}

#define DEFINE_SPACE_TO_DEPTH_KERNEL(NAME, TYPE) \
kernel void NAME( \
    device const TYPE* input [[buffer(0)]], \
    device TYPE* output [[buffer(1)]], \
    constant depth_space_params& params [[buffer(2)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    output[gid] = input[space_to_depth_input_index(params, gid)]; \
}

DEFINE_DEPTH_TO_SPACE_KERNEL(kernel_depth_to_space_u8, uchar)
DEFINE_DEPTH_TO_SPACE_KERNEL(kernel_depth_to_space_u16, ushort)
DEFINE_DEPTH_TO_SPACE_KERNEL(kernel_depth_to_space_u32, uint)
DEFINE_DEPTH_TO_SPACE_KERNEL(kernel_depth_to_space_u64, ulong)
DEFINE_DEPTH_TO_SPACE_KERNEL(kernel_depth_to_space_u128, uint4)

DEFINE_SPACE_TO_DEPTH_KERNEL(kernel_space_to_depth_u8, uchar)
DEFINE_SPACE_TO_DEPTH_KERNEL(kernel_space_to_depth_u16, ushort)
DEFINE_SPACE_TO_DEPTH_KERNEL(kernel_space_to_depth_u32, uint)
DEFINE_SPACE_TO_DEPTH_KERNEL(kernel_space_to_depth_u64, ulong)
DEFINE_SPACE_TO_DEPTH_KERNEL(kernel_space_to_depth_u128, uint4)

kernel void kernel_depth_to_space_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant depth_space_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint input_index = depth_to_space_input_index(params, gid);
    for (uint byte = 0; byte < params.element_size; ++byte)
        output[gid * params.element_size + byte] =
            input[input_index * params.element_size + byte];
}

kernel void kernel_space_to_depth_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant depth_space_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint input_index = space_to_depth_input_index(params, gid);
    for (uint byte = 0; byte < params.element_size; ++byte)
        output[gid * params.element_size + byte] =
            input[input_index * params.element_size + byte];
}
