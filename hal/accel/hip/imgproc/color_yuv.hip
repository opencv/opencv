// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// NV12/NV21 to BGR(A)/RGB(A) with cvtColor's BT.601 fixed-point coefficients (bit-exact).
// Each thread converts 2x2 pixels sharing one UV pair, the only hipbench variant above the noise.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

// ITUR_BT_601_*
const int CY = 1220542, CUB = 2116026, CUG = -409993, CVG = -852492, CVR = 1673527, SHIFT = 20;

__device__ __forceinline__ unsigned char sat8(int v)
{
    return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

template<int DCN, bool SWAP_BLUE, int UIDX>
__global__ void __launch_bounds__(256)
nv12ToBgrKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                int rows, int cols)
{
    const int x0 = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    const int y0 = (blockIdx.y * blockDim.y + threadIdx.y) * 2;
    if (x0 >= cols || y0 >= rows)
        return;

    // the UV plane follows the rows Y rows of the same buffer
    const unsigned char* uv = src + (size_t)(rows + y0 / 2) * sstep + x0;
    const int u = (int)uv[UIDX] - 128, v = (int)uv[1 - UIDX] - 128;
    const int half = 1 << (SHIFT - 1);
    const int ruv = half + CVR * v;
    const int guv = half + CVG * v + CUG * u;
    const int buv = half + CUB * u;

#pragma unroll
    for (int dy = 0; dy < 2; dy++)
    {
        const unsigned char* ys = src + (size_t)(y0 + dy) * sstep + x0;
        unsigned char* d = dst + (size_t)(y0 + dy) * dstep + (size_t)x0 * DCN;
#pragma unroll
        for (int dx = 0; dx < 2; dx++)
        {
            const int y = max(0, (int)ys[dx] - 16) * CY;
            const unsigned char r = sat8((y + ruv) >> SHIFT);
            const unsigned char g = sat8((y + guv) >> SHIFT);
            const unsigned char b = sat8((y + buv) >> SHIFT);
            d[dx * DCN + 0] = SWAP_BLUE ? r : b;
            d[dx * DCN + 1] = g;
            d[dx * DCN + 2] = SWAP_BLUE ? b : r;
            if (DCN == 4)
                d[dx * DCN + 3] = 255;
        }
    }
}

template<int DCN, bool SWAP_BLUE>
const void* byUIdx(int uIdx)
{
    return uIdx ? reinterpret_cast<const void*>(nv12ToBgrKernel<DCN, SWAP_BLUE, 1>)
                : reinterpret_cast<const void*>(nv12ToBgrKernel<DCN, SWAP_BLUE, 0>);
}

template<int DCN>
const void* bySwap(bool swapBlue, int uIdx)
{
    return swapBlue ? byUIdx<DCN, true>(uIdx) : byUIdx<DCN, false>(uIdx);
}

} // namespace

const void* nv12ToBgrKernelFor(int dcn, bool swapBlue, int uIdx)
{
    return dcn == 4 ? bySwap<4>(swapBlue, uIdx) : dcn == 3 ? bySwap<3>(swapBlue, uIdx) : 0;
}

hipError_t nv12ToBgr(const void* kernel, const unsigned char* src, size_t sstep,
                     unsigned char* dst, size_t dstep, int rows, int cols,
                     int blockX, int blockY, hipStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return hipSuccess;
    const dim3 block(blockX, blockY);
    const dim3 grid((cols / 2 + blockX - 1) / blockX, (rows / 2 + blockY - 1) / blockY);
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols };
    return hipLaunchKernel(kernel, grid, block, args, 0, s);
}

}}}} // namespace cv::accel::hip_impl::kernels
