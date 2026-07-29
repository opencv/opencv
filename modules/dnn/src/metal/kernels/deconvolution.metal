#include <metal_stdlib>

using namespace metal;

struct deconv2d_params
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

kernel void kernel_deconv2d_f32(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant deconv2d_params& params [[buffer(4)]],
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
    const uint local_output_channel = output_channel % output_channels_per_group;
    const uint input_channel_begin = group * input_channels_per_group;
    const uint input_plane_size = params.input_height * params.input_width;
    const uint kernel_size = params.kernel_height * params.kernel_width;

    float sum = bias[output_channel];
    for (uint local_input_channel = 0;
         local_input_channel < input_channels_per_group; ++local_input_channel)
    {
        const uint input_channel = input_channel_begin + local_input_channel;
        const uint input_base =
            (batch * params.input_channels + input_channel) * input_plane_size;
        const uint weight_base =
            (input_channel * output_channels_per_group + local_output_channel) * kernel_size;
        for (uint kernel_y = 0; kernel_y < params.kernel_height; ++kernel_y)
        {
            const int input_y_numerator = int(output_y + params.pad_top) -
                                          int(kernel_y * params.dilation_height);
            if (input_y_numerator < 0 ||
                input_y_numerator % int(params.stride_height) != 0)
                continue;
            const int input_y = input_y_numerator / int(params.stride_height);
            if (input_y >= int(params.input_height))
                continue;

            for (uint kernel_x = 0; kernel_x < params.kernel_width; ++kernel_x)
            {
                const int input_x_numerator = int(output_x + params.pad_left) -
                                              int(kernel_x * params.dilation_width);
                if (input_x_numerator < 0 ||
                    input_x_numerator % int(params.stride_width) != 0)
                    continue;
                const int input_x = input_x_numerator / int(params.stride_width);
                if (input_x >= int(params.input_width))
                    continue;

                const uint input_index = input_base + uint(input_y) * params.input_width +
                                         uint(input_x);
                const uint weight_index =
                    weight_base + kernel_y * params.kernel_width + kernel_x;
                sum = fma(input[input_index], weights[weight_index], sum);
            }
        }
    }
    output[gid] = sum;
}

kernel void kernel_deconv2d_oc4_f32(
    device const float* input [[buffer(0)]],
    device const float4* weights [[buffer(1)]],
    device const float* bias [[buffer(2)]],
    device float* output [[buffer(3)]],
    constant deconv2d_params& params [[buffer(4)]],
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
    const uint output_blocks_per_group = output_channels_per_group / 4;
    const uint group = output_block / output_blocks_per_group;
    const uint input_channel_begin = group * input_channels_per_group;
    const uint input_plane_size = params.input_height * params.input_width;
    const uint kernel_size = params.kernel_height * params.kernel_width;

    float4 sum = float4(bias[output_channel], bias[output_channel + 1],
                        bias[output_channel + 2], bias[output_channel + 3]);
    for (uint local_input_channel = 0;
         local_input_channel < input_channels_per_group; ++local_input_channel)
    {
        const uint input_channel = input_channel_begin + local_input_channel;
        const uint input_base =
            (batch * params.input_channels + input_channel) * input_plane_size;
        const uint weight_base =
            (output_block * input_channels_per_group + local_input_channel) * kernel_size;
        for (uint kernel_y = 0; kernel_y < params.kernel_height; ++kernel_y)
        {
            const int input_y_numerator = int(output_y + params.pad_top) -
                                          int(kernel_y * params.dilation_height);
            if (input_y_numerator < 0 ||
                input_y_numerator % int(params.stride_height) != 0)
                continue;
            const int input_y = input_y_numerator / int(params.stride_height);
            if (input_y >= int(params.input_height))
                continue;

            for (uint kernel_x = 0; kernel_x < params.kernel_width; ++kernel_x)
            {
                const int input_x_numerator = int(output_x + params.pad_left) -
                                              int(kernel_x * params.dilation_width);
                if (input_x_numerator < 0 ||
                    input_x_numerator % int(params.stride_width) != 0)
                    continue;
                const int input_x = input_x_numerator / int(params.stride_width);
                if (input_x >= int(params.input_width))
                    continue;

                const uint input_index = input_base + uint(input_y) * params.input_width +
                                         uint(input_x);
                const uint weight_index =
                    weight_base + kernel_y * params.kernel_width + kernel_x;
                sum = fma(float4(input[input_index]), weights[weight_index], sum);
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
