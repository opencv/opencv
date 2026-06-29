R"metal(
#include <metal_stdlib>
using namespace metal;

struct CompareParams
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
    int op;
};

inline bool compareValues(float a, float b, int op)
{
    if (op == 0)
        return a == b;
    if (op == 1)
        return a > b;
    if (op == 2)
        return a >= b;
    if (op == 3)
        return a < b;
    if (op == 4)
        return a <= b;
    return a != b;
}

kernel void compareKernel(device const uchar* src1 [[buffer(0)]],
                          device const uchar* src2 [[buffer(1)]],
                          device uchar* dst [[buffer(2)]],
                          constant CompareParams& p [[buffer(3)]],
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
            dst[dstBase + c] = compareValues((float)src1[src1Base + c], (float)src2[src2Base + c], p.op) ? 255 : 0;
    }
    else
    {
        ulong src1Base = p.src1Offset + (ulong)gid.y * p.src1Step + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        ulong src2Base = p.src2Offset + (ulong)gid.y * p.src2Step + (ulong)gid.x * (ulong)p.channels * sizeof(float);
        ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.channels;
        device const float* src1f = (device const float*)(src1 + src1Base);
        device const float* src2f = (device const float*)(src2 + src2Base);
        for (int c = 0; c < p.channels; ++c)
            dst[dstBase + c] = compareValues(src1f[c], src2f[c], p.op) ? 255 : 0;
    }
}
)metal"
