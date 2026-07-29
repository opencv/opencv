#include <metal_stdlib>

using namespace metal;

constant uint slice_max_dims = 10;

struct slice_params
{
    uint count;
    uint rank;
    uint element_size;
    uint reserved;
    uint output_shape[slice_max_dims];
    uint input_strides[slice_max_dims];
    uint starts[slice_max_dims];
    uint steps[slice_max_dims];
    uint flipped[slice_max_dims];
};

inline uint slice_input_index(constant slice_params& params, uint gid)
{
    uint remainder = gid;
    uint input_index = 0;
    for (int dim = int(params.rank) - 1; dim >= 0; --dim)
    {
        uint coordinate = remainder % params.output_shape[dim];
        remainder /= params.output_shape[dim];
        if (params.flipped[dim] != 0)
            coordinate = params.output_shape[dim] - 1 - coordinate;
        input_index += (params.starts[dim] + coordinate * params.steps[dim]) *
                       params.input_strides[dim];
    }
    return input_index;
}

#define DEFINE_SLICE_KERNEL(NAME, TYPE) \
kernel void NAME( \
    device const TYPE* input [[buffer(0)]], \
    device TYPE* output [[buffer(1)]], \
    constant slice_params& params [[buffer(2)]], \
    uint gid [[thread_position_in_grid]]) \
{ \
    output[gid] = input[slice_input_index(params, gid)]; \
}

DEFINE_SLICE_KERNEL(kernel_slice_u8, uchar)
DEFINE_SLICE_KERNEL(kernel_slice_u16, ushort)
DEFINE_SLICE_KERNEL(kernel_slice_u32, uint)
DEFINE_SLICE_KERNEL(kernel_slice_u64, ulong)
DEFINE_SLICE_KERNEL(kernel_slice_u128, uint4)

kernel void kernel_slice_bytes(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant slice_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint input_index = slice_input_index(params, gid);
    for (uint byte = 0; byte < params.element_size; ++byte)
        output[gid * params.element_size + byte] =
            input[input_index * params.element_size + byte];
}
