#include <metal_stdlib>
#include <metal_simdgroup_matrix>

using namespace metal;

struct conv2d_params
{
    uint batch;
    uint input_channels;
    uint input_height;
    uint input_width;
    uint output_channels;
    uint output_height;
    uint output_width;
    uint kernel_height;
    uint kernel_width;
    uint stride_height;
    uint stride_width;
    uint dilation_height;
    uint dilation_width;
    uint pad_top;
    uint pad_left;
    uint groups;
};

kernel void kernel_conv2d_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant conv2d_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint output_channel = (gid / output_plane_size) % params.output_channels;
    const uint batch = gid / (output_plane_size * params.output_channels);

    const uint input_channels_per_group = params.input_channels / params.groups;
    const uint output_channels_per_group = params.output_channels / params.groups;
    const uint group = output_channel / output_channels_per_group;
    const uint input_channel_begin = group * input_channels_per_group;
    const uint input_plane_size = params.input_height * params.input_width;

    const int input_y_begin = int(output_y * params.stride_height) - int(params.pad_top);
    const int input_x_begin = int(output_x * params.stride_width) - int(params.pad_left);
    const uint kernel_y_begin = input_y_begin < 0
        ? uint((-input_y_begin + int(params.dilation_height) - 1) / int(params.dilation_height)) : 0;
    const uint kernel_x_begin = input_x_begin < 0
        ? uint((-input_x_begin + int(params.dilation_width) - 1) / int(params.dilation_width)) : 0;
    const uint kernel_y_end = min(
        params.kernel_height,
        input_y_begin >= int(params.input_height) ? 0u :
            uint((int(params.input_height) - 1 - input_y_begin) / int(params.dilation_height) + 1));
    const uint kernel_x_end = min(
        params.kernel_width,
        input_x_begin >= int(params.input_width) ? 0u :
            uint((int(params.input_width) - 1 - input_x_begin) / int(params.dilation_width) + 1));

    float sum = bias[output_channel];
    for (uint local_input_channel = 0; local_input_channel < input_channels_per_group;
         ++local_input_channel)
    {
        const uint input_channel = input_channel_begin + local_input_channel;
        const uint input_base = (batch * params.input_channels + input_channel) * input_plane_size;
        const uint weight_base =
            (output_channel * input_channels_per_group + local_input_channel) *
            params.kernel_height * params.kernel_width;

        for (uint kernel_y = kernel_y_begin; kernel_y < kernel_y_end; ++kernel_y)
        {
            const uint input_y = uint(input_y_begin + int(kernel_y * params.dilation_height));

            for (uint kernel_x = kernel_x_begin; kernel_x < kernel_x_end; ++kernel_x)
            {
                const uint input_x = uint(input_x_begin + int(kernel_x * params.dilation_width));
                const uint input_index = input_base + input_y * params.input_width + input_x;
                const uint weight_index = weight_base + kernel_y * params.kernel_width + kernel_x;
                sum += input[input_index] * weights[weight_index];
            }
        }
    }
    output[gid] = sum;
}

kernel void kernel_conv2d_oc4_f32(
    device const float* input [[buffer(0)]],
    device const float4* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant conv2d_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_block_count = params.output_channels / 4;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint output_block = (gid / output_plane_size) % output_block_count;
    const uint batch = gid / (output_plane_size * output_block_count);
    const uint output_channel = output_block * 4;

    const uint input_channels_per_group = params.input_channels / params.groups;
    const uint output_channels_per_group = params.output_channels / params.groups;
    const uint group = output_channel / output_channels_per_group;
    const uint input_channel_begin = group * input_channels_per_group;
    const uint input_plane_size = params.input_height * params.input_width;
    const uint kernel_size = params.kernel_height * params.kernel_width;

    const int input_y_begin = int(output_y * params.stride_height) - int(params.pad_top);
    const int input_x_begin = int(output_x * params.stride_width) - int(params.pad_left);
    const uint kernel_y_begin = input_y_begin < 0
        ? uint((-input_y_begin + int(params.dilation_height) - 1) / int(params.dilation_height)) : 0;
    const uint kernel_x_begin = input_x_begin < 0
        ? uint((-input_x_begin + int(params.dilation_width) - 1) / int(params.dilation_width)) : 0;
    const uint kernel_y_end = min(
        params.kernel_height,
        input_y_begin >= int(params.input_height) ? 0u :
            uint((int(params.input_height) - 1 - input_y_begin) / int(params.dilation_height) + 1));
    const uint kernel_x_end = min(
        params.kernel_width,
        input_x_begin >= int(params.input_width) ? 0u :
            uint((int(params.input_width) - 1 - input_x_begin) / int(params.dilation_width) + 1));

    float4 sum = float4(bias[output_channel], bias[output_channel + 1],
                        bias[output_channel + 2], bias[output_channel + 3]);
    for (uint local_input_channel = 0; local_input_channel < input_channels_per_group;
         ++local_input_channel)
    {
        const uint input_channel = input_channel_begin + local_input_channel;
        const uint input_base = (batch * params.input_channels + input_channel) * input_plane_size;
        const uint weight_base =
            (output_block * input_channels_per_group + local_input_channel) * kernel_size;

        for (uint kernel_y = kernel_y_begin; kernel_y < kernel_y_end; ++kernel_y)
        {
            const uint input_y = uint(input_y_begin + int(kernel_y * params.dilation_height));
            for (uint kernel_x = kernel_x_begin; kernel_x < kernel_x_end; ++kernel_x)
            {
                const uint input_x = uint(input_x_begin + int(kernel_x * params.dilation_width));
                const float value = input[input_base + input_y * params.input_width + input_x];
                const float4 weight = weights[weight_base + kernel_y * params.kernel_width + kernel_x];
                sum = fma(float4(value), weight, sum);
            }
        }
    }

    const uint output_offset =
        (batch * params.output_channels + output_channel) * output_plane_size +
        output_y * params.output_width + output_x;
    output[output_offset] = sum.x;
    output[output_offset + output_plane_size] = sum.y;
    output[output_offset + output_plane_size * 2] = sum.z;
    output[output_offset + output_plane_size * 3] = sum.w;
}

kernel void kernel_conv2d_implicit_gemm_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant conv2d_params& params [[buffer(4)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
    constexpr uint tile_m = 32;
    constexpr uint tile_n = 32;
    constexpr uint tile_k = 16;
    threadgroup float input_tile[tile_m * tile_k];
    threadgroup float weight_tile[tile_k * tile_n];
    threadgroup float result_tile[tile_m * tile_n];

    const uint output_pixels = params.batch * params.output_height * params.output_width;
    const uint output_channels_per_group = params.output_channels / params.groups;
    const uint input_channels_per_group = params.input_channels / params.groups;
    const uint tiles_m = (output_pixels + tile_m - 1) / tile_m;
    const uint tiles_n = (output_channels_per_group + tile_n - 1) / tile_n;
    uint remaining = threadgroup_index;
    const uint output_channel_tile = remaining % tiles_n;
    remaining /= tiles_n;
    const uint output_pixel_tile = remaining % tiles_m;
    const uint group = remaining / tiles_m;

    const uint kernel_size = params.kernel_height * params.kernel_width;
    const uint implicit_k = input_channels_per_group * kernel_size;
    const uint output_plane_size = params.output_height * params.output_width;
    const uint input_plane_size = params.input_height * params.input_width;

    simdgroup_matrix<float, 8, 8> accumulator[4];
    for (uint i = 0; i < 4; ++i)
    {
        accumulator[i].thread_elements()[0] = 0.0f;
        accumulator[i].thread_elements()[1] = 0.0f;
    }
    for (uint k_base = 0; k_base < implicit_k; k_base += tile_k)
    {
        for (uint load_index = lid; load_index < tile_m * tile_k; load_index += 128)
        {
            const uint row = load_index / tile_k;
            const uint local_k = load_index % tile_k;
            const uint pixel = output_pixel_tile * tile_m + row;
            const uint k = k_base + local_k;
            float value = 0.0f;
            if (pixel < output_pixels && k < implicit_k)
            {
                const uint batch = pixel / output_plane_size;
                const uint output_position = pixel % output_plane_size;
                const uint output_y = output_position / params.output_width;
                const uint output_x = output_position % params.output_width;
                const uint local_input_channel = k / kernel_size;
                const uint kernel_index = k % kernel_size;
                const uint kernel_y = kernel_index / params.kernel_width;
                const uint kernel_x = kernel_index % params.kernel_width;
                const int input_y = int(output_y * params.stride_height +
                                       kernel_y * params.dilation_height) - int(params.pad_top);
                const int input_x = int(output_x * params.stride_width +
                                       kernel_x * params.dilation_width) - int(params.pad_left);
                if (input_y >= 0 && input_y < int(params.input_height) &&
                    input_x >= 0 && input_x < int(params.input_width))
                {
                    const uint input_channel = group * input_channels_per_group + local_input_channel;
                    value = input[(batch * params.input_channels + input_channel) * input_plane_size +
                                  uint(input_y) * params.input_width + uint(input_x)];
                }
            }
            input_tile[load_index] = value;
        }

        for (uint load_index = lid; load_index < tile_k * tile_n; load_index += 128)
        {
            const uint local_k = load_index / tile_n;
            const uint column = load_index % tile_n;
            const uint k = k_base + local_k;
            const uint local_channel = output_channel_tile * tile_n + column;
            float value = 0.0f;
            if (k < implicit_k && local_channel < output_channels_per_group)
            {
                const uint channel = group * output_channels_per_group + local_channel;
                value = weights[channel * implicit_k + k];
            }
            weight_tile[load_index] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint block_row = (simdgroup_index / 2) * 16;
        const uint block_column = (simdgroup_index % 2) * 16;
        simdgroup_matrix<float, 8, 8> input_low[2];
        simdgroup_matrix<float, 8, 8> input_high[2];
        simdgroup_matrix<float, 8, 8> weight_low[2];
        simdgroup_matrix<float, 8, 8> weight_high[2];
        simdgroup_load(input_low[0], input_tile + block_row * tile_k, tile_k);
        simdgroup_load(input_high[0], input_tile + block_row * tile_k + 8, tile_k);
        simdgroup_load(input_low[1], input_tile + (block_row + 8) * tile_k, tile_k);
        simdgroup_load(input_high[1], input_tile + (block_row + 8) * tile_k + 8, tile_k);
        simdgroup_load(weight_low[0], weight_tile + block_column, tile_n);
        simdgroup_load(weight_high[0], weight_tile + 8 * tile_n + block_column, tile_n);
        simdgroup_load(weight_low[1], weight_tile + block_column + 8, tile_n);
        simdgroup_load(weight_high[1], weight_tile + 8 * tile_n + block_column + 8, tile_n);
        for (uint row = 0; row < 2; ++row)
        {
            for (uint column = 0; column < 2; ++column)
            {
                const uint index = row * 2 + column;
                simdgroup_multiply_accumulate(accumulator[index], input_low[row],
                                              weight_low[column], accumulator[index]);
                simdgroup_multiply_accumulate(accumulator[index], input_high[row],
                                              weight_high[column], accumulator[index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint block_row = (simdgroup_index / 2) * 16;
    const uint block_column = (simdgroup_index % 2) * 16;
    for (uint row = 0; row < 2; ++row)
    {
        for (uint column = 0; column < 2; ++column)
        {
            simdgroup_store(accumulator[row * 2 + column],
                            result_tile + (block_row + row * 8) * tile_n +
                                block_column + column * 8,
                            tile_n);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint store_index = lid; store_index < tile_m * tile_n; store_index += 128)
    {
        const uint row = store_index / tile_n;
        const uint column = store_index % tile_n;
        const uint output_pixel = output_pixel_tile * tile_m + row;
        const uint local_output_channel = output_channel_tile * tile_n + column;
        if (output_pixel >= output_pixels || local_output_channel >= output_channels_per_group)
            continue;
        const uint output_channel = group * output_channels_per_group + local_output_channel;
        const uint batch = output_pixel / output_plane_size;
        const uint output_position = output_pixel % output_plane_size;
        output[(batch * params.output_channels + output_channel) * output_plane_size +
               output_position] = result_tile[store_index] + bias[output_channel];
    }
}

constant float winograd_input_transform[8][8] = {
    {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
    {0.00f, 1.00f, -1.00f, 0.50f, -0.50f, 2.00f, -2.00f, -1.00f},
    {-5.25f, 1.00f, 1.00f, 0.25f, 0.25f, 4.00f, 4.00f, 0.00f},
    {0.00f, -4.25f, 4.25f, -2.50f, 2.50f, -2.50f, 2.50f, 5.25f},
    {5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.00f, -5.00f, 0.00f},
    {0.00f, 1.00f, -1.00f, 2.00f, -2.00f, 0.50f, -0.50f, -5.25f},
    {-1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 0.00f},
    {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
};

constant float winograd_output_transform[8][6] = {
    {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
    {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
    {1.00f, -1.00f, 1.00f, -1.00f, 1.00f, -1.00f},
    {1.00f, 2.00f, 4.00f, 8.00f, 16.00f, 32.00f},
    {1.00f, -2.00f, 4.00f, -8.00f, 16.00f, -32.00f},
    {1.00f, 0.50f, 0.25f, 0.125f, 0.0625f, 0.03125f},
    {1.00f, -0.50f, 0.25f, -0.125f, 0.0625f, -0.03125f},
    {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
};

kernel void kernel_conv2d_winograd_input_f32(
    device const float* input [[buffer(0)]],
    device float* transformed [[buffer(1)]],
    constant conv2d_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint tiles_x = (params.output_width + 5) / 6;
    const uint tiles_y = (params.output_height + 5) / 6;
    const uint tile_count = params.batch * tiles_x * tiles_y;
    const uint channel = gid % params.input_channels;
    const uint tile = gid / params.input_channels;
    const uint batch = tile / (tiles_x * tiles_y);
    const uint spatial_tile = tile % (tiles_x * tiles_y);
    const uint tile_y = spatial_tile / tiles_x;
    const uint tile_x = spatial_tile % tiles_x;
    const int input_origin_y = int(tile_y * 6) - int(params.pad_top);
    const int input_origin_x = int(tile_x * 6) - int(params.pad_left);
    const uint input_plane_size = params.input_height * params.input_width;

    float source[64];
    float temporary[64];
    float result[64];
    for (uint y = 0; y < 8; ++y)
    {
        for (uint x = 0; x < 8; ++x)
        {
            const int input_y = input_origin_y + int(y);
            const int input_x = input_origin_x + int(x);
            source[y * 8 + x] = input_y >= 0 && input_y < int(params.input_height) &&
                                        input_x >= 0 && input_x < int(params.input_width)
                ? input[(batch * params.input_channels + channel) * input_plane_size +
                        uint(input_y) * params.input_width + uint(input_x)] : 0.0f;
        }
    }
    for (uint y = 0; y < 8; ++y)
    {
        for (uint x = 0; x < 8; ++x)
        {
            float value = 0.0f;
            for (uint k = 0; k < 8; ++k)
                value = fma(winograd_input_transform[k][y], source[k * 8 + x], value);
            temporary[y * 8 + x] = value;
        }
    }
    for (uint y = 0; y < 8; ++y)
    {
        for (uint x = 0; x < 8; ++x)
        {
            float value = 0.0f;
            for (uint k = 0; k < 8; ++k)
                value = fma(temporary[y * 8 + k], winograd_input_transform[k][x], value);
            result[y * 8 + x] = value;
        }
    }
    for (uint component = 0; component < 64; ++component)
        transformed[(component * tile_count + tile) * params.input_channels + channel] =
            result[component];
}

kernel void kernel_conv2d_winograd_gemm_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant conv2d_params& params [[buffer(4)]],
    uint threadgroup_index [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]],
    uint simdgroup_index [[simdgroup_index_in_threadgroup]])
{
    constexpr uint tile_m = 32;
    constexpr uint tile_n = 32;
    constexpr uint tile_k = 16;
    threadgroup float input_tile[tile_m * tile_k];
    threadgroup float weight_tile[tile_k * tile_n];
    threadgroup float result_tile[tile_m * tile_n];

    const uint tiles_x = (params.output_width + 5) / 6;
    const uint tiles_y = (params.output_height + 5) / 6;
    const uint tile_count = params.batch * tiles_x * tiles_y;
    const uint tiles_m = (tile_count + tile_m - 1) / tile_m;
    const uint tiles_n = (params.output_channels + tile_n - 1) / tile_n;
    uint remaining = threadgroup_index;
    const uint output_channel_tile = remaining % tiles_n;
    remaining /= tiles_n;
    const uint output_tile = remaining % tiles_m;
    const uint component = remaining / tiles_m;

    simdgroup_matrix<float, 8, 8> accumulator[4];
    for (uint i = 0; i < 4; ++i)
    {
        accumulator[i].thread_elements()[0] = 0.0f;
        accumulator[i].thread_elements()[1] = 0.0f;
    }
    for (uint k_base = 0; k_base < params.input_channels; k_base += tile_k)
    {
        for (uint load_index = lid; load_index < tile_m * tile_k; load_index += 128)
        {
            const uint row = load_index / tile_k;
            const uint k = k_base + load_index % tile_k;
            input_tile[load_index] = output_tile * tile_m + row < tile_count &&
                                           k < params.input_channels
                ? input[(component * tile_count + output_tile * tile_m + row) *
                        params.input_channels + k] : 0.0f;
        }
        for (uint load_index = lid; load_index < tile_k * tile_n; load_index += 128)
        {
            const uint k = k_base + load_index / tile_n;
            const uint channel = output_channel_tile * tile_n + load_index % tile_n;
            weight_tile[load_index] = k < params.input_channels && channel < params.output_channels
                ? weights[(component * params.output_channels + channel) *
                          params.input_channels + k] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint block_row = (simdgroup_index / 2) * 16;
        const uint block_column = (simdgroup_index % 2) * 16;
        simdgroup_matrix<float, 8, 8> input_low[2];
        simdgroup_matrix<float, 8, 8> input_high[2];
        simdgroup_matrix<float, 8, 8> weight_low[2];
        simdgroup_matrix<float, 8, 8> weight_high[2];
        simdgroup_load(input_low[0], input_tile + block_row * tile_k, tile_k);
        simdgroup_load(input_high[0], input_tile + block_row * tile_k + 8, tile_k);
        simdgroup_load(input_low[1], input_tile + (block_row + 8) * tile_k, tile_k);
        simdgroup_load(input_high[1], input_tile + (block_row + 8) * tile_k + 8, tile_k);
        simdgroup_load(weight_low[0], weight_tile + block_column, tile_n);
        simdgroup_load(weight_high[0], weight_tile + 8 * tile_n + block_column, tile_n);
        simdgroup_load(weight_low[1], weight_tile + block_column + 8, tile_n);
        simdgroup_load(weight_high[1], weight_tile + 8 * tile_n + block_column + 8, tile_n);
        for (uint row = 0; row < 2; ++row)
        {
            for (uint column = 0; column < 2; ++column)
            {
                const uint index = row * 2 + column;
                simdgroup_multiply_accumulate(accumulator[index], input_low[row],
                                              weight_low[column], accumulator[index]);
                simdgroup_multiply_accumulate(accumulator[index], input_high[row],
                                              weight_high[column], accumulator[index]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const uint block_row = (simdgroup_index / 2) * 16;
    const uint block_column = (simdgroup_index % 2) * 16;
    for (uint row = 0; row < 2; ++row)
    {
        for (uint column = 0; column < 2; ++column)
        {
            simdgroup_store(accumulator[row * 2 + column],
                            result_tile + (block_row + row * 8) * tile_n +
                                block_column + column * 8,
                            tile_n);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint store_index = lid; store_index < tile_m * tile_n; store_index += 128)
    {
        const uint row = store_index / tile_n;
        const uint column = store_index % tile_n;
        const uint tile = output_tile * tile_m + row;
        const uint output_channel = output_channel_tile * tile_n + column;
        if (tile < tile_count && output_channel < params.output_channels)
            output[(component * tile_count + tile) * params.output_channels + output_channel] =
                result_tile[store_index];
    }
}

kernel void kernel_conv2d_winograd_output_f32(
    device const float* transformed [[buffer(0)]],
    device const float* bias [[buffer(1)]],
    device float* output [[buffer(2)]],
    constant conv2d_params& params [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    const uint tiles_x = (params.output_width + 5) / 6;
    const uint tiles_y = (params.output_height + 5) / 6;
    const uint tile_count = params.batch * tiles_x * tiles_y;
    const uint output_channel = gid % params.output_channels;
    const uint tile = gid / params.output_channels;
    const uint batch = tile / (tiles_x * tiles_y);
    const uint spatial_tile = tile % (tiles_x * tiles_y);
    const uint tile_y = spatial_tile / tiles_x;
    const uint tile_x = spatial_tile % tiles_x;

    float source[64];
    float temporary[48];
    for (uint component = 0; component < 64; ++component)
        source[component] = transformed[(component * tile_count + tile) *
                                        params.output_channels + output_channel];
    for (uint y = 0; y < 8; ++y)
    {
        for (uint x = 0; x < 6; ++x)
        {
            float value = 0.0f;
            for (uint k = 0; k < 8; ++k)
                value = fma(source[y * 8 + k], winograd_output_transform[k][x], value);
            temporary[y * 6 + x] = value;
        }
    }

    const uint output_plane_size = params.output_height * params.output_width;
    for (uint y = 0; y < 6; ++y)
    {
        const uint output_y = tile_y * 6 + y;
        if (output_y >= params.output_height)
            continue;
        for (uint x = 0; x < 6; ++x)
        {
            const uint output_x = tile_x * 6 + x;
            if (output_x >= params.output_width)
                continue;
            float value = 0.0f;
            for (uint k = 0; k < 8; ++k)
                value = fma(winograd_output_transform[k][y], temporary[k * 6 + x], value);
            output[(batch * params.output_channels + output_channel) * output_plane_size +
                   output_y * params.output_width + output_x] = value + bias[output_channel];
        }
    }
}

kernel void kernel_depthwise_conv2d_3x3_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant conv2d_params& params [[buffer(4)]],
    uint tile_index [[threadgroup_position_in_grid]],
    uint lid [[thread_position_in_threadgroup]])
{
    constexpr uint output_tile_width = 8;
    constexpr uint output_tile_height = 8;
    constexpr uint maximum_input_tile_width = 17;
    threadgroup float input_tile[maximum_input_tile_width * maximum_input_tile_width];

    const uint tiles_x = (params.output_width + output_tile_width - 1) / output_tile_width;
    const uint tiles_y = (params.output_height + output_tile_height - 1) / output_tile_height;
    uint remaining = tile_index;
    const uint tile_x = remaining % tiles_x;
    remaining /= tiles_x;
    const uint tile_y = remaining % tiles_y;
    remaining /= tiles_y;
    const uint channel = remaining % params.output_channels;
    const uint batch = remaining / params.output_channels;

    const uint input_tile_width =
        (output_tile_width - 1) * params.stride_width + params.kernel_width;
    const uint input_tile_height =
        (output_tile_height - 1) * params.stride_height + params.kernel_height;
    const int input_origin_x =
        int(tile_x * output_tile_width * params.stride_width) - int(params.pad_left);
    const int input_origin_y =
        int(tile_y * output_tile_height * params.stride_height) - int(params.pad_top);
    const uint input_plane_size = params.input_height * params.input_width;
    const uint input_base = (batch * params.input_channels + channel) * input_plane_size;

    for (uint index = lid; index < input_tile_width * input_tile_height; index += 64)
    {
        const uint tile_y_index = index / input_tile_width;
        const uint tile_x_index = index % input_tile_width;
        const int input_y = input_origin_y + int(tile_y_index);
        const int input_x = input_origin_x + int(tile_x_index);
        input_tile[index] = input_y >= 0 && input_y < int(params.input_height) &&
                                  input_x >= 0 && input_x < int(params.input_width)
            ? input[input_base + uint(input_y) * params.input_width + uint(input_x)] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const uint local_output_y = lid / output_tile_width;
    const uint local_output_x = lid % output_tile_width;
    const uint output_y = tile_y * output_tile_height + local_output_y;
    const uint output_x = tile_x * output_tile_width + local_output_x;
    if (output_y >= params.output_height || output_x >= params.output_width)
        return;

    float sum = bias[channel];
    const uint kernel_base = channel * 9;
    const uint tile_input_y = local_output_y * params.stride_height;
    const uint tile_input_x = local_output_x * params.stride_width;
    for (uint kernel_y = 0; kernel_y < 3; ++kernel_y)
    {
        for (uint kernel_x = 0; kernel_x < 3; ++kernel_x)
        {
            const float value = input_tile[
                (tile_input_y + kernel_y) * input_tile_width + tile_input_x + kernel_x];
            sum = fma(value, weights[kernel_base + kernel_y * 3 + kernel_x], sum);
        }
    }

    const uint output_plane_size = params.output_height * params.output_width;
    output[(batch * params.output_channels + channel) * output_plane_size +
           output_y * params.output_width + output_x] = sum;
}
