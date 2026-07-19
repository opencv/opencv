#include <metal_stdlib>

using namespace metal;

struct avg_pool2d_params
{
    uint input_height;
    uint input_width;
    uint output_height;
    uint output_width;
    uint kernel_height;
    uint kernel_width;
    uint stride_height;
    uint stride_width;
    uint pad_top;
    uint pad_bottom;
    uint pad_left;
    uint pad_right;
    uint include_padding;
};

kernel void kernel_avg_pool2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant avg_pool2d_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint plane = gid / output_plane_size;

    const int unclamped_start_y = int(output_y * params.stride_height) - int(params.pad_top);
    const int unclamped_start_x = int(output_x * params.stride_width) - int(params.pad_left);
    const int unclamped_end_y = min(unclamped_start_y + int(params.kernel_height),
                                  int(params.input_height + params.pad_bottom));
    const int unclamped_end_x = min(unclamped_start_x + int(params.kernel_width),
                                  int(params.input_width + params.pad_right));

    const int start_y = max(unclamped_start_y, 0);
    const int start_x = max(unclamped_start_x, 0);
    const int end_y = min(unclamped_end_y, int(params.input_height));
    const int end_x = min(unclamped_end_x, int(params.input_width));

    const uint input_plane_size = params.input_height * params.input_width;
    const uint input_base = plane * input_plane_size;
    float sum = 0.0f;
    for (int y = start_y; y < end_y; ++y)
    {
        for (int x = start_x; x < end_x; ++x)
            sum += input[input_base + uint(y) * params.input_width + uint(x)];
    }

    const int padded_height = unclamped_end_y - unclamped_start_y;
    const int padded_width = unclamped_end_x - unclamped_start_x;
    const int valid_height = end_y - start_y;
    const int valid_width = end_x - start_x;
    const uint divisor = params.include_padding != 0
        ? uint(padded_height * padded_width)
        : uint(valid_height * valid_width);
    output[gid] = sum / float(divisor);
}
struct max_pool2d_params
{
    uint input_height;
    uint input_width;
    uint output_height;
    uint output_width;
    uint kernel_height;
    uint kernel_width;
    uint stride_height;
    uint stride_width;
    uint pad_top;
    uint pad_left;
};

kernel void kernel_max_pool2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device long* indices [[buffer(2)]],
    constant max_pool2d_params& params [[buffer(3)]],
    uint gid [[thread_position_in_grid]])
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint plane = gid / output_plane_size;

    const int start_y = max(int(output_y * params.stride_height) - int(params.pad_top), 0);
    const int start_x = max(int(output_x * params.stride_width) - int(params.pad_left), 0);
    const int end_y = min(int(output_y * params.stride_height) - int(params.pad_top) +
                             int(params.kernel_height),
                         int(params.input_height));
    const int end_x = min(int(output_x * params.stride_width) - int(params.pad_left) +
                             int(params.kernel_width),
                         int(params.input_width));

    const uint input_plane_size = params.input_height * params.input_width;
    const uint input_base = plane * input_plane_size;
    float result = -3.402823466e+38f;
    long result_index = -1;
    for (int y = start_y; y < end_y; ++y)
    {
        for (int x = start_x; x < end_x; ++x)
        {
            const uint index = uint(y) * params.input_width + uint(x);
            const float value = input[input_base + index];
            if (value > result)
            {
                result = value;
                result_index = long(index);
            }
        }
    }
    output[gid] = result;
    indices[gid] = result_index;
}
