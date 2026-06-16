R"metal(
#include <metal_stdlib>
using namespace metal;

struct SetToParams
{
    ulong dstOffset;
    ulong maskOffset;
    ulong dstStep;
    ulong maskStep;
    int rows;
    int cols;
    int elemSize;
    int depthSize;
    int channels;
    int maskChannels;
    int haveMask;
    uchar scalar[16];
};

kernel void setToKernel(device uchar* dst [[buffer(0)]],
                        device const uchar* mask [[buffer(1)]],
                        constant SetToParams& p [[buffer(2)]],
                        uint2 gid [[thread_position_in_grid]])
{
    if ((int)gid.x >= p.cols || (int)gid.y >= p.rows)
        return;

    ulong dstBase = p.dstOffset + (ulong)gid.y * p.dstStep + (ulong)gid.x * (ulong)p.elemSize;
    if (!p.haveMask)
    {
        for (int i = 0; i < p.elemSize; ++i)
            dst[dstBase + i] = p.scalar[i];
        return;
    }

    ulong maskBase = p.maskOffset + (ulong)gid.y * p.maskStep + (ulong)gid.x * (ulong)p.maskChannels;
    if (p.maskChannels == 1)
    {
        if (mask[maskBase])
        {
            for (int i = 0; i < p.elemSize; ++i)
                dst[dstBase + i] = p.scalar[i];
        }
    }
    else
    {
        for (int c = 0; c < p.channels; ++c)
        {
            if (mask[maskBase + c])
            {
                ulong dstChannel = dstBase + (ulong)c * (ulong)p.depthSize;
                int scalarChannel = c * p.depthSize;
                for (int i = 0; i < p.depthSize; ++i)
                    dst[dstChannel + i] = p.scalar[scalarChannel + i];
            }
        }
    }
}
)metal"
