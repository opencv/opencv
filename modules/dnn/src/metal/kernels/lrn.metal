#include <metal_stdlib>

using namespace metal;

struct lrn_params
{
    uint count;
    uint channels;
    uint height;
    uint width;
    uint local_size;
    float alpha;
    float beta;
    float bias;
    uint type;
};

kernel void kernel_lrn_f32(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    constant lrn_params& params [[buffer(2)]],
    uint gid [[thread_position_in_grid]])
{
    const uint plane_size = params.height * params.width;
    const uint x = gid % params.width;
    const uint y = (gid / params.width) % params.height;
    const uint channel = (gid / plane_size) % params.channels;
    const uint batch = gid / (plane_size * params.channels);
    const int radius = int(params.local_size / 2);
    float sum = 0.0f;

    if (params.type == 0)
    {
        const int first = max(0, int(channel) - radius);
        const int last = min(int(params.channels) - 1, int(channel) + radius);
        for (int c = first; c <= last; ++c)
        {
            const uint index = ((batch * params.channels + uint(c)) * params.height + y) *
                               params.width + x;
            sum += input[index] * input[index];
        }
    }
    else
    {
        const int first_y = max(0, int(y) - radius);
        const int last_y = min(int(params.height) - 1, int(y) + radius);
        const int first_x = max(0, int(x) - radius);
        const int last_x = min(int(params.width) - 1, int(x) + radius);
        for (int yy = first_y; yy <= last_y; ++yy)
        {
            for (int xx = first_x; xx <= last_x; ++xx)
            {
                const uint index = ((batch * params.channels + channel) * params.height +
                                    uint(yy)) * params.width + uint(xx);
                sum += input[index] * input[index];
            }
        }
    }

    output[gid] = input[gid] * pow(params.bias + params.alpha * sum, -params.beta);
}
