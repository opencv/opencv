#include <metal_stdlib>

using namespace metal;

struct instance_norm_params
{
    uint group_count;
    uint channels;
    uint normalization_size;
    float epsilon;
};

inline float instance_norm_threadgroup_sum(
    float value,
    threadgroup float* partial_values,
    uint simd_lane,
    uint simd_group,
    uint threads_per_threadgroup)
{
    constexpr uint simd_width = 32;
    value = simd_sum(value);
    if (simd_lane == 0)
        partial_values[simd_group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint simd_group_count =
        (threads_per_threadgroup + simd_width - 1) / simd_width;
    if (simd_group == 0)
    {
        value = simd_lane < simd_group_count ? partial_values[simd_lane] : 0.0f;
        value = simd_sum(value);
        if (simd_lane == 0)
            partial_values[0] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return partial_values[0];
}

kernel void kernel_instance_norm_f32(
    device const float* input [[buffer(0)]],
    device const float* scale [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant instance_norm_params& params [[buffer(4)]],
    uint group [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    threadgroup float partial_values[32];
    const uint group_offset = group * params.normalization_size;

    float sum = 0.0f;
    for (uint i = lid; i < params.normalization_size; i += threads_per_threadgroup)
        sum += input[group_offset + i];
    const float mean = instance_norm_threadgroup_sum(
        sum, partial_values, simd_lane, simd_group, threads_per_threadgroup) /
        params.normalization_size;

    float sum_squared_difference = 0.0f;
    for (uint i = lid; i < params.normalization_size; i += threads_per_threadgroup)
    {
        const float difference = input[group_offset + i] - mean;
        sum_squared_difference = fma(difference, difference, sum_squared_difference);
    }
    const float variance = instance_norm_threadgroup_sum(
        sum_squared_difference, partial_values, simd_lane, simd_group,
        threads_per_threadgroup) / params.normalization_size;
    const float normalizer = metal::precise::rsqrt(variance + params.epsilon);
    const uint channel = group % params.channels;
    const float group_scale = scale[channel];
    const float group_bias = bias[channel];

    for (uint i = lid; i < params.normalization_size; i += threads_per_threadgroup)
    {
        const float normalized = (input[group_offset + i] - mean) * normalizer;
        output[group_offset + i] = fma(normalized, group_scale, group_bias);
    }
}
