R"metal(
#include <metal_stdlib>
using namespace metal;

struct BitwiseParams
{
    ulong src1Offset;
    ulong src2Offset;
    ulong dstOffset;
    ulong src1Step;
    ulong src2Step;
    ulong dstStep;
    int rows;
    int cols;
    int elemSize;
    int op;
};

kernel void bitwiseKernel(device const uchar* src1 [[buffer(0)]],
                          device const uchar* src2 [[buffer(1)]],
                          device uchar* dst [[buffer(2)]],
                          constant BitwiseParams& p [[buffer(3)]],
                          uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)
        return;

    ulong src1Base = p.src1Offset + (ulong)gid.y * p.src1Step + (ulong)gid.x * (ulong)p.elemSize;
    ulong src2Base = p.src2Offset + (ulong)gid.y * p.src2Step + (ulong)gid.x * (ulong)p.elemSize;
    ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.elemSize;

    for (int c = 0; c < p.elemSize; ++c)
    {
        uchar a = src1[src1Base + c];
        uchar b = src2[src2Base + c];
        uchar value = 0;
        if (p.op == 9)
            value = a & b;
        else if (p.op == 10)
            value = a | b;
        else if (p.op == 11)
            value = a ^ b;
        else
            value = ~a;
        dst[dstBase + c] = value;
    }
}
)metal"
