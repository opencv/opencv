#include <metal_stdlib>

using namespace metal;

constant uint reduce_max_dims = 10;

struct reduce_params
{
    uint output_count;
    uint reduction_count;
    uint output_dims;
    uint reduction_dims;
    uint contiguous_reduction;
    uint output_shape[reduce_max_dims];
    uint output_strides[reduce_max_dims];
    uint reduction_shape[reduce_max_dims];
    uint reduction_strides[reduce_max_dims];
};

enum reduce_type : ushort
{
    reduce_max,
    reduce_min,
    reduce_mean,
    reduce_sum,
    reduce_l1,
    reduce_l2,
    reduce_prod,
    reduce_sum_square,
    reduce_log_sum,
    reduce_log_sum_exp
};

template<reduce_type operation>
inline float reduce_identity()
{
    if (operation == reduce_max)
        return -INFINITY;
    if (operation == reduce_min)
        return INFINITY;
    if (operation == reduce_prod)
        return 1.0f;
    return 0.0f;
}

template<reduce_type operation>
inline float reduce_transform(float value)
{
    if (operation == reduce_l1)
        return abs(value);
    if (operation == reduce_l2 || operation == reduce_sum_square)
        return value * value;
    if (operation == reduce_log_sum_exp)
        return metal::precise::exp(value);
    return value;
}

template<reduce_type operation>
inline float reduce_combine(float lhs, float rhs)
{
    if (operation == reduce_max)
        return rhs > lhs ? rhs : lhs;
    if (operation == reduce_min)
        return rhs > lhs ? lhs : rhs;
    if (operation == reduce_prod)
        return lhs * rhs;
    return lhs + rhs;
}

template<reduce_type operation>
inline float reduce_finalize(float value, uint reduction_count)
{
    if (operation == reduce_mean)
        return value / static_cast<float>(reduction_count);
    if (operation == reduce_l2)
        return metal::precise::sqrt(value);
    if (operation == reduce_log_sum || operation == reduce_log_sum_exp)
        return metal::precise::log(value);
    return value;
}

inline uint reduce_input_offset(
    uint linear_index,
    constant uint* shape,
    constant uint* strides,
    uint dims)
{
    uint offset = 0;
    for (int dim = static_cast<int>(dims) - 1; dim >= 0; --dim)
    {
        const uint coordinate = linear_index % shape[dim];
        linear_index /= shape[dim];
        offset += coordinate * strides[dim];
    }
    return offset;
}

typedef void (reduce_f32)(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant reduce_params& params [[buffer(2)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]]);

template<reduce_type operation>
kernel void kernel_reduce_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant reduce_params& params [[buffer(2)]],
    uint output_index [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint threads_per_threadgroup [[threads_per_threadgroup]])
{
    threadgroup float partial_values[256];
    threadgroup uint base_offset;

    if (lid == 0)
    {
        base_offset = params.contiguous_reduction
            ? output_index * params.reduction_count
            : reduce_input_offset(output_index, params.output_shape,
                                  params.output_strides, params.output_dims);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float value = reduce_identity<operation>();
    for (uint reduction_index = lid; reduction_index < params.reduction_count;
         reduction_index += threads_per_threadgroup)
    {
        const uint offset = base_offset + (params.contiguous_reduction
            ? reduction_index
            : reduce_input_offset(reduction_index, params.reduction_shape,
                                  params.reduction_strides, params.reduction_dims));
        value = reduce_combine<operation>(
            value, reduce_transform<operation>(input[offset]));
    }
    partial_values[lid] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint stride = threads_per_threadgroup / 2; stride > 0; stride /= 2)
    {
        if (lid < stride)
        {
            partial_values[lid] = reduce_combine<operation>(
                partial_values[lid], partial_values[lid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    if (lid == 0)
    {
        output[output_index] = reduce_finalize<operation>(
            partial_values[0], params.reduction_count);
    }
}

template [[host_name("kernel_reduce_max_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_max>;
template [[host_name("kernel_reduce_min_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_min>;
template [[host_name("kernel_reduce_mean_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_mean>;
template [[host_name("kernel_reduce_sum_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_sum>;
template [[host_name("kernel_reduce_l1_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_l1>;
template [[host_name("kernel_reduce_l2_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_l2>;
template [[host_name("kernel_reduce_prod_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_prod>;
template [[host_name("kernel_reduce_sum_square_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_sum_square>;
template [[host_name("kernel_reduce_log_sum_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_log_sum>;
template [[host_name("kernel_reduce_log_sum_exp_f32")]]
kernel reduce_f32 kernel_reduce_f32<reduce_log_sum_exp>;
