#include <metal_stdlib>

using namespace metal;

kernel void kernel_nary_copy_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant uint& count [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    output[gid] = input[gid];
}

constant uint METAL_MAX_RANK = 8;

struct broadcast_params
{
    uint count;
    uint rank;
    uint operation;
    uint lhs_shape[METAL_MAX_RANK];
    uint rhs_shape[METAL_MAX_RANK];
    uint output_shape[METAL_MAX_RANK];
    uint lhs_stride[METAL_MAX_RANK];
    uint rhs_stride[METAL_MAX_RANK];
    uint output_stride[METAL_MAX_RANK];
};

kernel void kernel_nary_eltwise_f32(
    device const float* lhs [[buffer(0)]],
    device const float* rhs [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant broadcast_params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    uint remainder = gid;
    uint lhs_index = 0;
    uint rhs_index = 0;
    for (uint i = 0; i < params.rank; ++i)
    {
        const uint coordinate = remainder / params.output_stride[i];
        remainder %= params.output_stride[i];
        if (params.lhs_shape[i] != 1)
            lhs_index += coordinate * params.lhs_stride[i];
        if (params.rhs_shape[i] != 1)
            rhs_index += coordinate * params.rhs_stride[i];
    }

    const float a = lhs[lhs_index];
    const float b = rhs[rhs_index];
    float result = a + b;
    switch (params.operation)
    {
        case 1: result = a * b; break;
        case 2: result = a - b; break;
        case 3: result = a / b; break;
        case 4: result = max(a, b); break;
        case 5: result = min(a, b); break;
        default: break;
    }
    output[gid] = result;
}
