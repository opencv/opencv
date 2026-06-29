R"metal(
#include <metal_stdlib>
using namespace metal;

struct MultiplyParams
{
    ulong src1Offset;
    ulong src2Offset;
    ulong dstOffset;
    ulong src1Step;
    ulong src2Step;
    ulong dstStep;
    int rows;
    int cols;
    int depth;
    int channels;
    float scale;
};

kernel void multiplyKernel(device const uchar* src1 [[buffer(0)]],
                           device const uchar* src2 [[buffer(1)]],
                           device uchar* dst [[buffer(2)]],
                           constant MultiplyParams& p [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)
        return;

    if (p.depth == 0)
    {
        ulong src1Base = p.src1Offset + (ulong)gid.y * p.src1Step + (ulong)gid.x * (ulong)p.channels;
        ulong src2Base = p.src2Offset + (ulong)gid.y * p.src2Step + (ulong)gid.x * (ulong)p.channels;
        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels;
        for (int c = 0; c < p.channels; ++c)
        {
            float value = (float)src1[src1Base + c] * (float)src2[src2Base + c] * p.scale;
            dst[dstBase + c] = (uchar)clamp(rint(value), 0.0f, 255.0f);
        }
    }
    else
    {
        ulong src1Base = p.src1Offset + (ulong)gid.y * p.src1Step + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        ulong src2Base = p.src2Offset + (ulong)gid.y * p.src2Step + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        device const float* src1f = (device const float*)(src1 + src1Base);
        device const float* src2f = (device const float*)(src2 + src2Base);
        device float* dstf = (device float*)(dst + dstBase);
        for (int c = 0; c < p.channels; ++c)
            dstf[c] = src1f[c] * src2f[c] * p.scale;
    }
}
)metal"
