#include <metal_stdlib>

using namespace metal;

struct const_params
{
    uint byte_count;
};

constant uint const_copy_bytes_per_thread = 16;

kernel void kernel_const_copy(
    device const uchar* input [[buffer(0)]],
    device uchar* output [[buffer(1)]],
    constant const_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint begin = gid * const_copy_bytes_per_thread;
    const uint end = min(begin + const_copy_bytes_per_thread, params.byte_count);
    for (uint offset = begin; offset < end; ++offset)
        output[offset] = input[offset];
}
