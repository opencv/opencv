R"metal(
#include <metal_stdlib>
using namespace metal;

struct CopyToMaskParams
{
    ulong srcOffset;
    ulong maskOffset;
    ulong dstOffset;
    ulong srcStep;
    ulong maskStep;
    ulong dstStep;
    int rows;
    int cols;
    int elemSize;
    int depthSize;
    int channels;
    int maskChannels;
    int haveDstUninit;
};

kernel void copyToMaskKernel(device const uchar* src [[buffer(0)]],
                             device const uchar* mask [[buffer(1)]],
                             device uchar* dst [[buffer(2)]],
                             constant CopyToMaskParams& p [[buffer(3)]],
                             uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)
        return;

    ulong srcBase = p.srcOffset + (ulong)gid.y * p.srcStep + (ulong)gid.x * (ulong)p.elemSize;
    ulong maskBase = p.maskOffset + (ulong)gid.y * p.maskStep + (ulong)gid.x * (ulong)p.maskChannels;
    ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.elemSize;
    if (p.maskChannels == 1)
    {
        if (mask[maskBase])
        {
            for (int i = 0; i < p.elemSize; ++i)
                dst[dstBase + i] = src[srcBase + i];
        }
        else if (p.haveDstUninit)
        {
            for (int i = 0; i < p.elemSize; ++i)
                dst[dstBase + i] = 0;
        }
    }
    else
    {
        for (int c = 0; c < p.channels; ++c)
        {
            ulong srcChannel = srcBase + (ulong)c * (ulong)p.depthSize;
            ulong dstChannel = dstBase + (ulong)c * (ulong)p.depthSize;
            if (mask[maskBase + c])
            {
                for (int i = 0; i < p.depthSize; ++i)
                    dst[dstChannel + i] = src[srcChannel + i];
            }
            else if (p.haveDstUninit)
            {
                for (int i = 0; i < p.depthSize; ++i)
                    dst[dstChannel + i] = 0;
            }
        }
    }
}
)metal"
