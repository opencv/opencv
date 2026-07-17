R"metal(
#include <metal_stdlib>
using namespace metal;

struct AddParams
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
};

kernel void addKernel(device const uchar* src1 [[buffer(0)]],
                      device const uchar* src2 [[buffer(1)]],
                      device uchar* dst [[buffer(2)]],
                      constant AddParams& p [[buffer(3)]],
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
            int sum = (int)src1[src1Base + c] + (int)src2[src2Base + c];
            dst[dstBase + c] = (uchar)min(sum, 255);
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
            dstf[c] = src1f[c] + src2f[c];
    }
}
)metal"
