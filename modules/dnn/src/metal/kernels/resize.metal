#include <metal_stdlib>

using namespace metal;

struct resize_params
{
    uint input_height;
    uint input_width;
    uint output_height;
    uint output_width;
    uint interpolation;
    uint align_corners;
    uint half_pixel_centers;
    float scale_height;
    float scale_width;
};

kernel void kernel_resize2d_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant resize_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint output_plane_size = params.output_height * params.output_width;
    const uint output_x = gid % params.output_width;
    const uint output_y = (gid / params.output_width) % params.output_height;
    const uint plane = gid / output_plane_size;
    const uint input_base = plane * params.input_height * params.input_width;

    if (params.interpolation == 0)
    {
        const float y = params.half_pixel_centers
            ? float(output_y) * params.scale_height + 0.5f * params.scale_height
            : float(output_y) * params.scale_height;
        const float x = params.half_pixel_centers
            ? float(output_x) * params.scale_width + 0.5f * params.scale_width
            : float(output_x) * params.scale_width;
        const uint input_y = min(uint(params.half_pixel_centers ? floor(y) :
                                     (params.align_corners ? floor(y + 0.5f) : floor(y))),
                                params.input_height - 1);
        const uint input_x = min(uint(params.half_pixel_centers ? floor(x) :
                                     (params.align_corners ? floor(x + 0.5f) : floor(x))),
                                params.input_width - 1);
        output[gid] = input[input_base + input_y * params.input_width + input_x];
        return;
    }

    const float input_y = params.half_pixel_centers
        ? max((float(output_y) + 0.5f) * params.scale_height - 0.5f, 0.0f)
        : float(output_y) * params.scale_height;
    const float input_x = params.half_pixel_centers
        ? max((float(output_x) + 0.5f) * params.scale_width - 0.5f, 0.0f)
        : float(output_x) * params.scale_width;
    const uint y0 = min(uint(input_y), params.input_height - 1);
    const uint x0 = min(uint(input_x), params.input_width - 1);
    const uint y1 = min(y0 + 1, params.input_height - 1);
    const uint x1 = min(x0 + 1, params.input_width - 1);
    const float dy = input_y - float(y0);
    const float dx = input_x - float(x0);
    const float v00 = input[input_base + y0 * params.input_width + x0];
    const float v01 = input[input_base + y0 * params.input_width + x1];
    const float v10 = input[input_base + y1 * params.input_width + x0];
    const float v11 = input[input_base + y1 * params.input_width + x1];
    output[gid] = mix(mix(v00, v01, dx), mix(v10, v11, dx), dy);
}
