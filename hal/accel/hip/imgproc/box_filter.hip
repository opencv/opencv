// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Box filter of 8-bit images: the two of 47 hipbench variants kept, bit-exact with cv::boxFilter.

#include "../common.hpp"
#include "kernels.hpp"

namespace cv { namespace accel { namespace hip_impl { namespace kernels {

namespace {

// cv::BorderTypes
enum { B_CONSTANT = 0, B_REPLICATE = 1, B_REFLECT = 2, B_REFLECT_101 = 4 };

// @p p mapped inside a row or column of @p len, as cv::borderInterpolate; -1 for BORDER_CONSTANT
template<int BORDER>
__device__ __forceinline__ int borderIdx(int p, int len)
{
    if ((unsigned)p < (unsigned)len)
        return p;
    if (BORDER == B_REPLICATE)
        return p < 0 ? 0 : len - 1;
    if (BORDER == B_REFLECT || BORDER == B_REFLECT_101)
    {
        if (len == 1)
            return 0;
        const int d = BORDER == B_REFLECT_101 ? 1 : 0;
        do
        {
            if (p < 0)
                p = -p - 1 + d;
            else
                p = len - 1 - (p - len) - d;
        }
        while ((unsigned)p >= (unsigned)len);
        return p;
    }
    return -1;
}

template<bool FIXED>
__device__ __forceinline__ unsigned char boxOut(int sum, const BoxScale& scale)
{
    // saturates like the CPU SIMD path
    if (FIXED)
    {
        const unsigned v = __umul24((unsigned)sum + scale.divDelta, scale.divScale) >> 23;
        return (unsigned char)(v > 255 ? 255 : v);
    }
    const int v = __float2int_rn((float)sum * scale.scale);
    return (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
}

// Each thread slides a column sum down 4 rows, adding the entering row and dropping the leaving
// one; any ksize. Threads whose window is inside the image skip the border arithmetic.
const int SLIDE_BX = 64, SLIDE_BY = 4, SLIDE_PPY = 4;

template<int CN, int BORDER, bool FIXED>
__global__ void __launch_bounds__(SLIDE_BX * SLIDE_BY)
boxSlidingKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
                 int rows, int cols, int kw, int kh, int ax, int ay, BoxScale scale)
{
    const int x = blockIdx.x * SLIDE_BX + threadIdx.x;
    const int y0 = (blockIdx.y * SLIDE_BY + threadIdx.y) * SLIDE_PPY;
    if (x >= cols || y0 >= rows)
        return;

    // every tap of every row this thread produces is inside the image
    const bool inside = x - ax >= 0 && y0 - ay >= 0 &&
                        x - ax + kw <= cols && y0 + SLIDE_PPY - 1 - ay + kh <= rows;

    int col[CN];
#pragma unroll
    for (int c = 0; c < CN; c++)
        col[c] = 0;

    for (int j = 0; j < kh; j++)
    {
        const int sy = inside ? y0 - ay + j : borderIdx<BORDER>(y0 - ay + j, rows);
        if (sy < 0)
            continue;
        const unsigned char* r = src + (size_t)sy * sstep;
        if (inside)
        {
            const unsigned char* p = r + (size_t)(x - ax) * CN;
            for (int i = 0; i < kw; i++)
#pragma unroll
                for (int c = 0; c < CN; c++)
                    col[c] += p[i * CN + c];
        }
        else
        {
            for (int i = 0; i < kw; i++)
            {
                const int sx = borderIdx<BORDER>(x - ax + i, cols);
                if (sx < 0)
                    continue;
#pragma unroll
                for (int c = 0; c < CN; c++)
                    col[c] += r[(size_t)sx * CN + c];
            }
        }
    }

    unsigned char* d = dst + (size_t)y0 * dstep + (size_t)x * CN;
#pragma unroll
    for (int c = 0; c < CN; c++)
        d[c] = boxOut<FIXED>(col[c], scale);

#pragma unroll
    for (int g = 1; g < SLIDE_PPY; g++)
    {
        const int y = y0 + g;
        if (y >= rows)
            break;
        int drop = y - 1 - ay, add = y - 1 - ay + kh;
        if (!inside)
        {
            drop = borderIdx<BORDER>(drop, rows);
            add = borderIdx<BORDER>(add, rows);
        }
        const unsigned char* rd = drop < 0 ? 0 : src + (size_t)drop * sstep;
        const unsigned char* ra = add < 0 ? 0 : src + (size_t)add * sstep;
        for (int i = 0; i < kw; i++)
        {
            const int sx = inside ? x - ax + i : borderIdx<BORDER>(x - ax + i, cols);
            if (sx < 0)
                continue;
#pragma unroll
            for (int c = 0; c < CN; c++)
                col[c] += (ra ? (int)ra[(size_t)sx * CN + c] : 0) - (rd ? (int)rd[(size_t)sx * CN + c] : 0);
        }
        d = dst + (size_t)y * dstep + (size_t)x * CN;
#pragma unroll
        for (int c = 0; c < CN; c++)
            d[c] = boxOut<FIXED>(col[c], scale);
    }
}

// The block stages its tile plus halo in LDS once and each thread produces 2 pixels of a row;
// fastest on every GPU measured. The window is compiled in: 3x3 and 5x5 only.
const int LDS_BX = 64, LDS_BY = 4, LDS_PPT = 2;

static_assert(SLIDE_BX * SLIDE_BY == BOX_BLOCK_THREADS && LDS_BX * LDS_BY == BOX_BLOCK_THREADS,
              "the eligibility check in ops.cpp assumes BOX_BLOCK_THREADS threads per block");

template<int CN, int BORDER, int K, bool FIXED>
__global__ void __launch_bounds__(LDS_BX * LDS_BY)
boxLdsKernel(const unsigned char* src, size_t sstep, unsigned char* dst, size_t dstep,
             int rows, int cols, int ax, int ay, BoxScale scale)
{
    constexpr int TW = LDS_BX * LDS_PPT + K - 1;
    constexpr int TH = LDS_BY + K - 1;
    // 4 channels: one word per pixel, so every tap is one aligned load; else one plane per channel
    constexpr bool PACKED = CN == 4;
    __shared__ unsigned char tile[PACKED ? 1 : CN * TH * TW];
    __shared__ unsigned tile32[PACKED ? TH * TW : 1];

    const int ox = blockIdx.x * LDS_BX * LDS_PPT - ax;
    const int oy = blockIdx.y * LDS_BY - ay;
    for (int idx = threadIdx.y * LDS_BX + threadIdx.x; idx < TH * TW; idx += LDS_BX * LDS_BY)
    {
        const int ty = idx / TW, tx = idx - ty * TW;
        const int sx = borderIdx<BORDER>(ox + tx, cols);
        const int sy = borderIdx<BORDER>(oy + ty, rows);
        const unsigned char* p = sx < 0 || sy < 0 ? 0 : src + (size_t)sy * sstep + (size_t)sx * CN;
        if (PACKED)
            tile32[ty * TW + tx] = p ? p[0] | (p[1] << 8) | (p[2] << 16) | ((unsigned)p[3] << 24) : 0u;
        else
        {
#pragma unroll
            for (int c = 0; c < CN; c++)
                tile[(c * TH + ty) * TW + tx] = p ? p[c] : 0;
        }
    }
    __syncthreads();

    const int y = blockIdx.y * LDS_BY + threadIdx.y;
    if (y >= rows)
        return;
    const int lx = threadIdx.x * LDS_PPT;
    unsigned char* d = dst + (size_t)y * dstep;
#pragma unroll
    for (int p = 0; p < LDS_PPT; p++)
    {
        const int x = blockIdx.x * LDS_BX * LDS_PPT + lx + p;
        if (x >= cols)
            break;
        int sum[CN];
#pragma unroll
        for (int c = 0; c < CN; c++)
            sum[c] = 0;
#pragma unroll
        for (int j = 0; j < K; j++)
#pragma unroll
            for (int i = 0; i < K; i++)
            {
                if (PACKED)
                {
                    const unsigned v = tile32[(threadIdx.y + j) * TW + lx + p + i];
#pragma unroll
                    for (int c = 0; c < CN; c++)
                        sum[c] += (v >> (8 * c)) & 255;
                }
                else
                {
#pragma unroll
                    for (int c = 0; c < CN; c++)
                        sum[c] += tile[(c * TH + threadIdx.y + j) * TW + lx + p + i];
                }
            }
#pragma unroll
        for (int c = 0; c < CN; c++)
            d[(size_t)x * CN + c] = boxOut<FIXED>(sum[c], scale);
    }
}

template<int CN, int K, bool FIXED>
const void* ldsByBorder(int border)
{
    switch (border)
    {
    case B_CONSTANT:    return reinterpret_cast<const void*>(boxLdsKernel<CN, B_CONSTANT, K, FIXED>);
    case B_REPLICATE:   return reinterpret_cast<const void*>(boxLdsKernel<CN, B_REPLICATE, K, FIXED>);
    case B_REFLECT:     return reinterpret_cast<const void*>(boxLdsKernel<CN, B_REFLECT, K, FIXED>);
    case B_REFLECT_101: return reinterpret_cast<const void*>(boxLdsKernel<CN, B_REFLECT_101, K, FIXED>);
    default:            return 0;
    }
}

template<int K, bool FIXED>
const void* ldsByCn(int cn, int border)
{
    switch (cn)
    {
    case 1: return ldsByBorder<1, K, FIXED>(border);
    case 3: return ldsByBorder<3, K, FIXED>(border);
    case 4: return ldsByBorder<4, K, FIXED>(border);
    default: return 0;
    }
}

template<int K>
const void* ldsByScale(int cn, int border, bool fixed)
{
    return fixed ? ldsByCn<K, true>(cn, border) : ldsByCn<K, false>(cn, border);
}

template<int CN, bool FIXED>
const void* slidingByBorder(int border)
{
    switch (border)
    {
    case B_CONSTANT:    return reinterpret_cast<const void*>(boxSlidingKernel<CN, B_CONSTANT, FIXED>);
    case B_REPLICATE:   return reinterpret_cast<const void*>(boxSlidingKernel<CN, B_REPLICATE, FIXED>);
    case B_REFLECT:     return reinterpret_cast<const void*>(boxSlidingKernel<CN, B_REFLECT, FIXED>);
    case B_REFLECT_101: return reinterpret_cast<const void*>(boxSlidingKernel<CN, B_REFLECT_101, FIXED>);
    default:            return 0;
    }
}

template<bool FIXED>
const void* slidingByCn(int cn, int border)
{
    switch (cn)
    {
    case 1: return slidingByBorder<1, FIXED>(border);
    case 3: return slidingByBorder<3, FIXED>(border);
    case 4: return slidingByBorder<4, FIXED>(border);
    default: return 0;
    }
}

} // namespace

const void* boxLdsKernelFor(int cn, int ksize, int border, bool fixed)
{
    return ksize == 3 ? ldsByScale<3>(cn, border, fixed) : ksize == 5 ? ldsByScale<5>(cn, border, fixed) : 0;
}

const void* boxSlidingKernelFor(int cn, int border, bool fixed)
{
    return fixed ? slidingByCn<true>(cn, border) : slidingByCn<false>(cn, border);
}

hipError_t boxFilterLds(const void* kernel, const unsigned char* src, size_t sstep,
                        unsigned char* dst, size_t dstep, int rows, int cols,
                        int ax, int ay, BoxScale scale, hipStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return hipSuccess;
    const dim3 block(LDS_BX, LDS_BY);
    const dim3 grid((cols + LDS_BX * LDS_PPT - 1) / (LDS_BX * LDS_PPT), (rows + LDS_BY - 1) / LDS_BY);
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols, &ax, &ay, &scale };
    return hipLaunchKernel(kernel, grid, block, args, 0, s);
}

hipError_t boxFilterSliding(const void* kernel, const unsigned char* src, size_t sstep,
                            unsigned char* dst, size_t dstep, int rows, int cols,
                            int kw, int kh, int ax, int ay, BoxScale scale, hipStream_t s)
{
    if (rows <= 0 || cols <= 0)
        return hipSuccess;
    const dim3 block(SLIDE_BX, SLIDE_BY);
    const dim3 grid((cols + SLIDE_BX - 1) / SLIDE_BX,
                    (rows + SLIDE_BY * SLIDE_PPY - 1) / (SLIDE_BY * SLIDE_PPY));
    void* args[] = { &src, &sstep, &dst, &dstep, &rows, &cols, &kw, &kh, &ax, &ay, &scale };
    return hipLaunchKernel(kernel, grid, block, args, 0, s);
}

}}}} // namespace cv::accel::hip_impl::kernels
