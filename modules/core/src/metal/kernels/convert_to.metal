R"metal(
#include <metal_stdlib>
using namespace metal;

struct ConvertToParams
{
    ulong srcOffset;
    ulong dstOffset;
    ulong srcStep;
    ulong dstStep;
    int rows;
    int cols;
    int sdepth;
    int ddepth;
    int channels;
    float alpha;
    float beta;
};

inline uchar saturateToUchar(float value)
{
    return (uchar)clamp(rint(value), 0.0f, 255.0f);
}

kernel void convertToKernel(device const uchar* src [[buffer(0)]],
                            device uchar* dst [[buffer(1)]],
                            constant ConvertToParams& p [[buffer(2)]],
                            uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)
        return;

    ulong srcBase = p.srcOffset + (ulong)gid.y * p.srcStep;
    ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep;

    if (p.sdepth == 0 && p.ddepth == 0)
    {
        srcBase += (ulong)gid.x * (ulong)p.channels;
        dstBase += (ulong)gid.x * (ulong)p.channels;
        for (int c = 0; c < p.channels; ++c)
        {
            float value = (float)src[srcBase + c] * p.alpha + p.beta;
            dst[dstBase + c] = saturateToUchar(value);
        }
    }
    else if (p.sdepth == 0)
    {
        srcBase += (ulong)gid.x * (ulong)p.channels;
        dstBase += (ulong)gid.x * (ulong)p.channels * sizeof(float);
        device float* dstf = (device float*)(dst + dstBase);
        for (int c = 0; c < p.channels; ++c)
            dstf[c] = (float)src[srcBase + c] * p.alpha + p.beta;
    }
    else if (p.ddepth == 0)
    {
        srcBase += (ulong)gid.x * (ulong)p.channels * sizeof(float);
        dstBase += (ulong)gid.x * (ulong)p.channels;
        device const float* srcf = (device const float*)(src + srcBase);
        for (int c = 0; c < p.channels; ++c)
        {
            float value = srcf[c] * p.alpha + p.beta;
            dst[dstBase + c] = saturateToUchar(value);
        }
    }
    else
    {
        srcBase += (ulong)gid.x * (ulong)p.channels * sizeof(float);
        dstBase += (ulong)gid.x * (ulong)p.channels * sizeof(float);
        device const float* srcf = (device const float*)(src + srcBase);
        device float* dstf = (device float*)(dst + dstBase);
        for (int c = 0; c < p.channels; ++c)
            dstf[c] = srcf[c] * p.alpha + p.beta;
    }
}
)metal"
