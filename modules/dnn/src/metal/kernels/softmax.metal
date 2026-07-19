#include <metal_stdlib>

using namespace metal;

struct softmax_params
{
    uint count;
    uint channels;
    uint inner_size;
    uint log_softmax;
    float scale;
};

kernel void kernel_softmax_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant softmax_params& params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr uint simd_width = 32;
    threadgroup float partial_values[simd_width];

    const uint inner = row % params.inner_size;
    const uint outer = row / params.inner_size;
    const uint base = outer * params.channels * params.inner_size + inner;
    float maximum = -INFINITY;
    for (uint c = lid; c < params.channels; c += threads_per_threadgroup)
        maximum = max(maximum, input[base + c * params.inner_size] * params.scale);

    maximum = simd_max(maximum);
    const uint simd_group_count = (threads_per_threadgroup + simd_width - 1) / simd_width;
    if (simd_group_count > 1)
    {
        if (simd_lane == 0)
            partial_values[simd_group] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0)
        {
            maximum = simd_lane < simd_group_count ? partial_values[simd_lane] : -INFINITY;
            maximum = simd_max(maximum);
            if (simd_lane == 0)
                partial_values[0] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = partial_values[0];
    }

    if (isinf(maximum))
    {
        if (maximum < 0.0f)
        {
            for (uint c = lid; c < params.channels; c += threads_per_threadgroup)
                output[base + c * params.inner_size] = params.log_softmax ? -INFINITY : 0.0f;
            return;
        }

        float maximum_count = 0.0f;
        for (uint c = lid; c < params.channels; c += threads_per_threadgroup)
        {
            const float value = input[base + c * params.inner_size] * params.scale;
            maximum_count += isinf(value) && value > 0.0f;
        }
        maximum_count = simd_sum(maximum_count);
        if (simd_group_count > 1)
        {
            if (simd_lane == 0)
                partial_values[simd_group] = maximum_count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0)
            {
                maximum_count = simd_lane < simd_group_count ? partial_values[simd_lane] : 0.0f;
                maximum_count = simd_sum(maximum_count);
                if (simd_lane == 0)
                    partial_values[0] = maximum_count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum_count = partial_values[0];
        }

        for (uint c = lid; c < params.channels; c += threads_per_threadgroup)
        {
            const uint index = base + c * params.inner_size;
            const float value = input[index] * params.scale;
            const bool is_maximum = isinf(value) && value > 0.0f;
            output[index] = params.log_softmax
                ? (is_maximum ? -log(maximum_count) : -INFINITY)
                : (is_maximum ? 1.0f / maximum_count : 0.0f);
        }
        return;
    }

    float sum = 0.0f;
    for (uint c = lid; c < params.channels; c += threads_per_threadgroup)
        sum += exp(input[base + c * params.inner_size] * params.scale - maximum);

    sum = simd_sum(sum);
    if (simd_group_count > 1)
    {
        if (simd_lane == 0)
            partial_values[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0)
        {
            sum = simd_lane < simd_group_count ? partial_values[simd_lane] : 0.0f;
            sum = simd_sum(sum);
            if (simd_lane == 0)
                partial_values[0] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = partial_values[0];
    }

    const float log_sum = params.log_softmax ? log(sum) : 0.0f;
    const float inverse_sum = params.log_softmax ? 0.0f : 1.0f / sum;
    for (uint c = lid; c < params.channels; c += threads_per_threadgroup)
    {
        const uint index = base + c * params.inner_size;
        const float value = input[index] * params.scale - maximum;
        output[index] = params.log_softmax ? value - log_sum : exp(value) * inverse_sum;
    }
}

kernel void kernel_softmax_contiguous_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant softmax_params& params [[buffer(2)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_group [[simdgroup_index_in_threadgroup]])
{
    constexpr uint simd_width = 32;
    constexpr uint reads_per_thread = 4;
    threadgroup float partial_values[simd_width];

    const uint row_offset = row * params.channels;
    const uint first_channel = lid * reads_per_thread;
    float values[reads_per_thread];
    float maximum = -INFINITY;
    for (uint i = 0; i < reads_per_thread; ++i)
    {
        const uint channel = first_channel + i;
        values[i] = channel < params.channels
            ? input[row_offset + channel] * params.scale
            : -INFINITY;
        maximum = max(maximum, values[i]);
    }

    maximum = simd_max(maximum);
    const uint simd_group_count = (threads_per_threadgroup + simd_width - 1) / simd_width;
    if (simd_group_count > 1)
    {
        if (simd_lane == 0)
            partial_values[simd_group] = maximum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0)
        {
            maximum = simd_lane < simd_group_count ? partial_values[simd_lane] : -INFINITY;
            maximum = simd_max(maximum);
            if (simd_lane == 0)
                partial_values[0] = maximum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        maximum = partial_values[0];
    }

    if (isinf(maximum))
    {
        if (maximum < 0.0f)
        {
            for (uint i = 0; i < reads_per_thread; ++i)
            {
                const uint channel = first_channel + i;
                if (channel < params.channels)
                    output[row_offset + channel] = 0.0f;
            }
            return;
        }

        float maximum_count = 0.0f;
        for (uint i = 0; i < reads_per_thread; ++i)
            maximum_count += isinf(values[i]) && values[i] > 0.0f;
        maximum_count = simd_sum(maximum_count);
        if (simd_group_count > 1)
        {
            if (simd_lane == 0)
                partial_values[simd_group] = maximum_count;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (simd_group == 0)
            {
                maximum_count = simd_lane < simd_group_count ? partial_values[simd_lane] : 0.0f;
                maximum_count = simd_sum(maximum_count);
                if (simd_lane == 0)
                    partial_values[0] = maximum_count;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            maximum_count = partial_values[0];
        }

        const float inverse_maximum_count = 1.0f / maximum_count;
        for (uint i = 0; i < reads_per_thread; ++i)
        {
            const uint channel = first_channel + i;
            if (channel < params.channels)
                output[row_offset + channel] = isinf(values[i]) && values[i] > 0.0f
                    ? inverse_maximum_count : 0.0f;
        }
        return;
    }

    float sum = 0.0f;
    for (uint i = 0; i < reads_per_thread; ++i)
    {
        values[i] = exp(values[i] - maximum);
        sum += values[i];
    }

    sum = simd_sum(sum);
    if (simd_group_count > 1)
    {
        if (simd_lane == 0)
            partial_values[simd_group] = sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_group == 0)
        {
            sum = simd_lane < simd_group_count ? partial_values[simd_lane] : 0.0f;
            sum = simd_sum(sum);
            if (simd_lane == 0)
                partial_values[0] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        sum = partial_values[0];
    }

    const float inverse_sum = 1.0f / sum;
    for (uint i = 0; i < reads_per_thread; ++i)
    {
        const uint channel = first_channel + i;
        if (channel < params.channels)
            output[row_offset + channel] = values[i] * inverse_sum;
    }
}
