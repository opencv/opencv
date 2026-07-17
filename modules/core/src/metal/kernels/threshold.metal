R"metal(
#include <metal_stdlib>
using namespace metal;

struct ThresholdParams
{
    ulong srcOffset;
    ulong dstOffset;
    ulong srcStep;
    ulong dstStep;
    int rows;
    int cols;
    int depth;
    int channels;
    int thresholdType;
    float thresh;
    float maxval;
};

inline float applyThreshold(float value, constant ThresholdParams& p)
{
    if (p.thresholdType == 0)
        return value > p.thresh ? p.maxval : 0.0f;
    if (p.thresholdType == 1)
        return value <= p.thresh ? p.maxval : 0.0f;
    if (p.thresholdType == 2)
        return min(value, p.thresh);
    if (p.thresholdType == 3)
        return value > p.thresh ? value : 0.0f;
    return value <= p.thresh ? value : 0.0f;
}

kernel void thresholdKernel(device const uchar* src [[buffer(0)]],
                            device uchar* dst [[buffer(1)]],
                            constant ThresholdParams& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)
        return;

    if (p.depth == 0)
    {
        ulong srcBase = p.srcOffset + (ulong)gid.y * p.srcStep + (ulong)gid.x * (ulong)p.channels;
        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels;
        for (int c = 0; c < p.channels; ++c)
        {
            float value = applyThreshold((float)src[srcBase + c], p);
            dst[dstBase + c] = (uchar)clamp(rint(value), 0.0f, 255.0f);
        }
    }
    else
    {
        ulong srcBase = p.srcOffset + (ulong)gid.y * p.srcStep + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        device const float* srcf = (device const float*)(src + srcBase);
        device float* dstf = (device float*)(dst + dstBase);
        for (int c = 0; c < p.channels; ++c)
            dstf[c] = applyThreshold(srcf[c], p);
    }
}
)metal"
