#include <metal_stdlib>

using namespace metal;

struct topk_params
{
    uint row_count;
    uint axis_size;
    uint inner_size;
    uint k;
};

template<bool largest>
inline bool topk_is_better(float lhs_value, uint lhs_index,
                           float rhs_value, uint rhs_index)
{
    if (lhs_value == rhs_value)
        return lhs_index < rhs_index;
    return largest ? lhs_value > rhs_value : lhs_value < rhs_value;
}

typedef void (topk_f32)(
    device const float* input [[buffer(0)]],
    device float* output_values [[buffer(1)]],
    device long* output_indices [[buffer(2)]],
    constant topk_params& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]);

template<bool largest>
kernel void kernel_topk_f32(
    device const float* input [[buffer(0)]],
    device float* output_values [[buffer(1)]],
    device long* output_indices [[buffer(2)]],
    constant topk_params& params [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup float candidate_values[256];
    threadgroup uint candidate_indices[256];

    const uint outer = row / params.inner_size;
    const uint inner = row - outer * params.inner_size;
    const uint input_offset = outer * params.axis_size * params.inner_size + inner;
    const uint output_offset = outer * params.k * params.inner_size + inner;
    const float invalid_value = largest ? -INFINITY : INFINITY;

    float local_value = invalid_value;
    uint local_index = UINT_MAX;
    float previous_value = invalid_value;
    uint previous_index = UINT_MAX;
    for (uint index = lid; index < params.axis_size; index += threads_per_threadgroup)
    {
        const float value = input[input_offset + index * params.inner_size];
        if (local_index == UINT_MAX ||
            topk_is_better<largest>(value, index, local_value, local_index))
        {
            local_value = value;
            local_index = index;
        }
    }

    for (uint output_index = 0; output_index < params.k; ++output_index)
    {
        candidate_values[lid] = local_value;
        candidate_indices[lid] = local_index;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2)
        {
            if (lid < stride)
            {
                const float other_value = candidate_values[lid + stride];
                const uint other_index = candidate_indices[lid + stride];
                if (other_index != UINT_MAX &&
                    (candidate_indices[lid] == UINT_MAX ||
                     topk_is_better<largest>(other_value, other_index,
                                             candidate_values[lid], candidate_indices[lid])))
                {
                    candidate_values[lid] = other_value;
                    candidate_indices[lid] = other_index;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const float selected_value = candidate_values[0];
        const uint selected_index = candidate_indices[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lid == 0)
        {
            const uint destination = output_offset + output_index * params.inner_size;
            output_values[destination] = selected_value;
            output_indices[destination] = static_cast<long>(selected_index);
        }

        if (local_index == selected_index)
        {
            previous_value = local_value;
            previous_index = local_index;
            local_value = invalid_value;
            local_index = UINT_MAX;
            for (uint index = lid; index < params.axis_size;
                 index += threads_per_threadgroup)
            {
                const float value = input[input_offset + index * params.inner_size];
                if (!topk_is_better<largest>(previous_value, previous_index, value, index))
                    continue;
                if (local_index == UINT_MAX ||
                    topk_is_better<largest>(value, index, local_value, local_index))
                {
                    local_value = value;
                    local_index = index;
                }
            }
        }
    }
}

template [[host_name("kernel_topk_largest_f32")]]
kernel topk_f32 kernel_topk_f32<true>;

template [[host_name("kernel_topk_smallest_f32")]]
kernel topk_f32 kernel_topk_f32<false>;
