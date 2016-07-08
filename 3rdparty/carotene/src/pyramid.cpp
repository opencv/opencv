/*
 * By downloading, copying, installing or using the software you agree to this license.
 * If you do not agree to this license, do not download, install,
 * copy or use the software.
 *
 *
 *                           License Agreement
 *                For Open Source Computer Vision Library
 *                        (3-clause BSD License)
 *
 * Copyright (C) 2012-2015, NVIDIA Corporation, all rights reserved.
 * Third party copyrights are property of their respective owners.
 *
 * Redistribution and use in source and binary forms, with or without modification,
 * are permitted provided that the following conditions are met:
 *
 *   * Redistributions of source code must retain the above copyright notice,
 *     this list of conditions and the following disclaimer.
 *
 *   * Redistributions in binary form must reproduce the above copyright notice,
 *     this list of conditions and the following disclaimer in the documentation
 *     and/or other materials provided with the distribution.
 *
 *   * Neither the names of the copyright holders nor the names of the contributors
 *     may be used to endorse or promote products derived from this software
 *     without specific prior written permission.
 *
 * This software is provided by the copyright holders and contributors "as is" and
 * any express or implied warranties, including, but not limited to, the implied
 * warranties of merchantability and fitness for a particular purpose are disclaimed.
 * In no event shall copyright holders or contributors be liable for any direct,
 * indirect, incidental, special, exemplary, or consequential damages
 * (including, but not limited to, procurement of substitute goods or services;
 * loss of use, data, or profits; or business interruption) however caused
 * and on any theory of liability, whether in contract, strict liability,
 * or tort (including negligence or otherwise) arising in any way out of
 * the use of this software, even if advised of the possibility of such damage.
 */

#include "common.hpp"

#include <vector>

namespace CAROTENE_NS {

bool isGaussianPyramidDownRTZSupported(const Size2D &srcSize, const Size2D &dstSize, BORDER_MODE border_mode)
{
    if (!isSupportedConfiguration())
        return false;
    // Need at least 8 pixels for vectorization.
    // Need to make sure dst width is half the src width.
    // Don't care about dst height.
    if ( dstSize.width < 8 || std::abs((ptrdiff_t)dstSize.width*2 - (ptrdiff_t)srcSize.width) > 2 )
        return false;

    // Current implementation only supports Reflect101 (ie: UNDEFINED mode)
    if (border_mode != BORDER_MODE_UNDEFINED)
        return false;

    return true;
}

bool isGaussianPyramidDownU8Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn)
{
    if (!isSupportedConfiguration())
        return false;
    if ( (dstSize.width * cn) < 8 ||
         (cn != 1 && cn !=3 && cn!=4) ||
         std::abs((ptrdiff_t)dstSize.width*2 - (ptrdiff_t)srcSize.width) > 2 ||
         std::abs((ptrdiff_t)dstSize.height*2 - (ptrdiff_t)srcSize.height) > 2 )
        return false;

    return true;
}

bool isGaussianPyramidDownS16Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn)
{
    if (!isSupportedConfiguration())
        return false;
    if ( (dstSize.width * cn) < 4 ||
         (cn != 1 && cn !=3 && cn!=4) ||
         std::abs((ptrdiff_t)dstSize.width*2 - (ptrdiff_t)srcSize.width) > 2 ||
         std::abs((ptrdiff_t)dstSize.height*2 - (ptrdiff_t)srcSize.height) > 2 )
        return false;

    return true;
}

bool isGaussianPyramidDownF32Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn)
{
    if (!isSupportedConfiguration())
        return false;
    if ( (dstSize.width * cn) < 4 ||
         (cn != 1 && cn !=3 && cn!=4) ||
         std::abs((ptrdiff_t)dstSize.width*2 - (ptrdiff_t)srcSize.width) > 2 ||
         std::abs((ptrdiff_t)dstSize.height*2 - (ptrdiff_t)srcSize.height) > 2 )
        return false;

    return true;
}

bool isGaussianPyramidUpU8Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn)
{
    if (!isSupportedConfiguration())
        return false;
    if ( (srcSize.width * cn) < 8 ||
         (cn != 1 && cn !=3 && cn!=4) ||
         std::abs((ptrdiff_t)dstSize.width - (ptrdiff_t)srcSize.width*2) != (ptrdiff_t)dstSize.width % 2 ||
         std::abs((ptrdiff_t)dstSize.height - (ptrdiff_t)srcSize.height*2) != (ptrdiff_t)dstSize.height % 2 )
        return false;

    return true;
}

bool isGaussianPyramidUpS16Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn)
{
    if (!isSupportedConfiguration())
        return false;
    if ( (srcSize.width * cn) < 12 ||
         (cn != 1 && cn !=3 && cn!=4) ||
         std::abs((ptrdiff_t)dstSize.width - (ptrdiff_t)srcSize.width*2) != (ptrdiff_t)dstSize.width % 2 ||
         std::abs((ptrdiff_t)dstSize.height - (ptrdiff_t)srcSize.height*2) != (ptrdiff_t)dstSize.height % 2 )
        return false;

    return true;
}

#ifdef CAROTENE_NEON

namespace {

ptrdiff_t borderInterpolate101(ptrdiff_t p, ptrdiff_t len)
{
    if (len == 1)
        return 0;
    else
    {
        while ((unsigned)p >= (unsigned)len)
        {
            if (p < 0)
                p = -p;
            else
                p = (len - 1)*2 - p;
        }
    }
    return p;
}

} // namespace

#endif

void gaussianPyramidDownRTZ(const Size2D &srcSize,
                            const u8 *srcBase, ptrdiff_t srcStride,
                            const Size2D &dstSize,
                            u8 *dstBase, ptrdiff_t dstStride,
                            BORDER_MODE border, u8 borderValue)
{
    internal::assertSupportedConfiguration(isGaussianPyramidDownRTZSupported(srcSize, dstSize, border));
#ifdef CAROTENE_NEON
    // Single-core NEON code
    const size_t dwidth = dstSize.width;
    const size_t dheight = dstSize.height;
    const size_t swidth = srcSize.width;
    const size_t sheight = srcSize.height;

    ptrdiff_t idx_l1 = borderInterpolate101(-1, swidth);
    ptrdiff_t idx_l2 = borderInterpolate101(-2, swidth);
    ptrdiff_t idx_r1 = borderInterpolate101(swidth + 0, swidth);
    ptrdiff_t idx_r2 = borderInterpolate101(swidth + 1, swidth);

    //1-line buffer
    std::vector<u16> _buf((swidth + 4) + 32/sizeof(u16));
    u16* lane = internal::alignPtr(&_buf[2], 32);

    uint8x8_t vc6u8 = vmov_n_u8(6);
    uint16x8_t vc6u16 = vmovq_n_u16(6);
    uint16x8_t vc4u16 = vmovq_n_u16(4);

    u8* dst = dstBase;

    for (size_t i = 0; i < dheight; ++i, dst += dstStride)
    {
        //vertical convolution
        const u8* ln0 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-2, sheight));
        const u8* ln1 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-1, sheight));
        const u8* ln2 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+0, sheight));
        const u8* ln3 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+1, sheight));
        const u8* ln4 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+2, sheight));

        size_t x = 0;
        for (; x <= swidth - 8; x += 8)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, x % 5 - 2));
            uint8x8_t v0 = vld1_u8(ln0+x);
            uint8x8_t v1 = vld1_u8(ln1+x);
            uint8x8_t v2 = vld1_u8(ln2+x);
            uint8x8_t v3 = vld1_u8(ln3+x);
            uint8x8_t v4 = vld1_u8(ln4+x);

            uint16x8_t v = vaddl_u8(v0, v4);
            uint16x8_t v13 = vaddl_u8(v1, v3);

            v = vmlal_u8(v, v2, vc6u8);
            v = vmlaq_u16(v, v13, vc4u16);

            vst1q_u16(lane + x, v);
        }
        for (; x < swidth; ++x)
        {
            lane[x] = ln0[x] + ln4[x] + 4u * (ln1[x] + ln3[x]) + 6u * ln2[x];
        }

        //left&right borders
        lane[-1] = lane[idx_l1];
        lane[-2] = lane[idx_l2];

        lane[swidth] = lane[idx_r1];
        lane[swidth+1] = lane[idx_r2];

        //horizontal convolution
        x = 0;
        size_t vw = (swidth/2) - 7;    // Using 7 instead of 8 allows swidth of 14 or 15.
        for (; x < vw; x += 8)
        {
            internal::prefetch(lane + 2 * x);
            uint16x8x2_t vLane0 = vld2q_u16(lane + 2*x-2);  // L0[0] = x0 x2 x4 x6 x8 x10 x12 x14   L0[1] = x1 x3 x5 x7 x9 x11 x13 x15
            uint16x8x2_t vLane1 = vld2q_u16(lane + 2*x-1);  // L1[0] = x1 x3 x5 x7 x9 x11 x13 x15   L1[1] = x2 x4 x6 x8 x10 x12 x14 x16
            uint16x8x2_t vLane2 = vld2q_u16(lane + 2*x+0);  // L2[0] = x2 x4 x6 x8 x10 x12 x14 x16  L2[1] = x3 x5 x7 x9 x11 x13 x15 x17
            uint16x8x2_t vLane3 = vld2q_u16(lane + 2*x+1);  // L3[0] = x3 x5 x7 x9 x11 x13 x15 x17  L3[1] = x4 x6 x8 x10 x12 x14 x16 x18
            uint16x8x2_t vLane4 = vld2q_u16(lane + 2*x+2);  // L4[0] = x4 x6 x8 x10 x12 x14 x16 x18 L4[1] = x5 x7 x9 x11 x13 x15 x17 x19
            uint16x8_t vSum_0_4 = vaddq_u16(vLane0.val[0], vLane4.val[0]);
            uint16x8_t vSum_1_3 = vaddq_u16(vLane1.val[0], vLane3.val[0]);
            vSum_0_4 = vmlaq_u16(vSum_0_4, vLane2.val[0], vc6u16);
            vSum_0_4 = vmlaq_u16(vSum_0_4, vSum_1_3, vc4u16);
            uint8x8_t vRes = vshrn_n_u16(vSum_0_4, 8);

            vst1_u8(dst + x, vRes);
        }

        for (; x < dwidth; x++)
        {
            dst[x] = u8((lane[2*x-2] + lane[2*x+2] + 4u * (lane[2*x-1] + lane[2*x+1]) + 6u * lane[2*x]) >> 8);
        }
    }
#else
    // Remove 'unused parameter' warnings.
    (void)srcSize;
    (void)srcBase;
    (void)srcStride;
    (void)dstSize;
    (void)dstBase;
    (void)dstStride;
    (void)border;
#endif
    (void)borderValue;
}

void gaussianPyramidDown(const Size2D &srcSize,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         const Size2D &dstSize,
                         u8 *dstBase, ptrdiff_t dstStride, u8 cn)
{
    internal::assertSupportedConfiguration(isGaussianPyramidDownU8Supported(srcSize, dstSize, cn));
#ifdef CAROTENE_NEON
    size_t dcolcn = dstSize.width*cn;
    size_t scolcn = srcSize.width*cn;
    size_t roiw8 = dcolcn - 7;

    size_t idx_l1 = borderInterpolate101(-1, srcSize.width) * cn;
    size_t idx_l2 = borderInterpolate101(-2, srcSize.width) * cn;
    size_t idx_r1 = borderInterpolate101(srcSize.width + 0, srcSize.width) * cn;
    size_t idx_r2 = borderInterpolate101(srcSize.width + 1, srcSize.width) * cn;

    //1-line buffer
    std::vector<u16> _buf(cn*(srcSize.width + 4) + 32/sizeof(u16));
    u16* lane = internal::alignPtr(&_buf[2*cn], 32);

    uint8x8_t vc6u8 = vmov_n_u8(6);
    uint16x8_t vc6u16 = vmovq_n_u16(6);
    uint16x8_t vc4u16 = vmovq_n_u16(4);

    for (size_t i = 0; i < dstSize.height; ++i)
    {
        u8* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        const u8* ln0 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-2, srcSize.height));
        const u8* ln1 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-1, srcSize.height));
        const u8* ln2 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+0, srcSize.height));
        const u8* ln3 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+1, srcSize.height));
        const u8* ln4 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+2, srcSize.height));

        size_t x = 0;
        for (; x <= scolcn - 8; x += 8)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, (ptrdiff_t)x % 5 - 2));
            uint8x8_t v0 = vld1_u8(ln0+x);
            uint8x8_t v1 = vld1_u8(ln1+x);
            uint8x8_t v2 = vld1_u8(ln2+x);
            uint8x8_t v3 = vld1_u8(ln3+x);
            uint8x8_t v4 = vld1_u8(ln4+x);

            uint16x8_t v = vaddl_u8(v0, v4);
            uint16x8_t v13 = vaddl_u8(v1, v3);

            v = vmlal_u8(v, v2, vc6u8);
            v = vmlaq_u16(v, v13, vc4u16);

            vst1q_u16(lane + x, v);
        }
        for (; x < scolcn; ++x)
        {
            lane[x] = ln0[x] + ln4[x] + 4u * (ln1[x] + ln3[x]) + 6u * ln2[x];
        }

        //left&right borders
        for (u32 k = 0; k < cn; ++k)
        {
            lane[(s32)(-cn+k)] = lane[idx_l1 + k];
            lane[(s32)(-cn-cn+k)] = lane[idx_l2 + k];

            lane[scolcn+k] = lane[idx_r1 + k];
            lane[scolcn+cn+k] = lane[idx_r2 + k];
        }

        //horizontal convolution
        x = 0;
        switch(cn)
        {
        case 1:
            for (; x < roiw8; x += 8)
            {
                internal::prefetch(lane + 2 * x);
#if __GNUC_MINOR__ < 7
                __asm__ (
                    "vld2.16 {d0-d3}, [%[in0]]                               \n\t"
                    "vld2.16 {d4-d7}, [%[in4]]                               \n\t"
                    "vld2.16 {d12-d15}, [%[in1]]                             \n\t"
                    "vld2.16 {d16-d19}, [%[in3]]                             \n\t"
                    "vld2.16 {d8-d11}, [%[in2],:256]                         \n\t"
                    "vadd.i16 q0, q2                  /*q0 = v0 + v4*/       \n\t"
                    "vadd.i16 q6, q8                  /*q6 = v1 + v3*/       \n\t"
                    "vmla.i16 q0, q4, %q[c6]          /*q0 += v2 * 6*/       \n\t"
                    "vmla.i16 q0, q6, %q[c4]          /*q1 += (v1+v3) * 4*/  \n\t"
                    "vrshrn.u16 d8, q0, #8                                   \n\t"
                    "vst1.8 {d8}, [%[out]]                                   \n\t"
                    : /*no output*/
                    : [out] "r" (dst + x),
                      [in0] "r" (lane + 2*x-2),
                      [in1] "r" (lane + 2*x-1),
                      [in2] "r" (lane + 2*x+0),
                      [in3] "r" (lane + 2*x+1),
                      [in4] "r" (lane + 2*x+2),
                      [c4] "w" (vc4u16), [c6] "w" (vc6u16)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19"
                );
#else
                uint16x8x2_t vLane0 = vld2q_u16(lane + 2*x-2);
                uint16x8x2_t vLane1 = vld2q_u16(lane + 2*x-1);
                uint16x8x2_t vLane2 = vld2q_u16(lane + 2*x+0);
                uint16x8x2_t vLane3 = vld2q_u16(lane + 2*x+1);
                uint16x8x2_t vLane4 = vld2q_u16(lane + 2*x+2);

                uint16x8_t vSum_0_4 = vaddq_u16(vLane0.val[0], vLane4.val[0]);
                uint16x8_t vSum_1_3 = vaddq_u16(vLane1.val[0], vLane3.val[0]);
                vSum_0_4 = vmlaq_u16(vSum_0_4, vLane2.val[0], vc6u16);
                vSum_0_4 = vmlaq_u16(vSum_0_4, vSum_1_3, vc4u16);
                uint8x8_t vRes = vrshrn_n_u16(vSum_0_4, 8);

                vst1_u8(dst + x, vRes);
#endif
            }
            break;
        case 3:
        {
            uint16x4_t vx1 = vld1_u16(lane - 2*3);
            uint16x4_t vx2 = vld1_u16(lane - 1*3);
            uint16x4_t vx3 = vld1_u16(lane + 0*3);
            uint16x8_t v0 = vcombine_u16(vx1, vx3);

            uint8x8_t map = vreinterpret_u8_u64(vmov_n_u64(0xFFFF060504020100ULL));
            for (; x < roiw8; x += 6)
            {
                internal::prefetch(lane + 2 * x + 12);

                uint16x4_t vx_ = vld1_u16(lane + 2*x-1*3 + 6);
                uint16x4_t vx4 = vld1_u16(lane + 2*x+0*3 + 6);
                uint16x4_t vx5 = vld1_u16(lane + 2*x+1*3 + 6);
                uint16x4_t vx6 = vld1_u16(lane + 2*x+2*3 + 6);

                uint16x8_t v1 = vcombine_u16(vx2, vx_);
                uint16x8_t v2 = vcombine_u16(vget_high_u16(v0), vx4);
                uint16x8_t v3 = vcombine_u16(vx_, vx5);
                uint16x8_t v4 = vcombine_u16(vx4, vx6);
                vx2 = vx5;

                uint16x8_t v = vaddq_u16(v0, v4);
                uint16x8_t v13 = vaddq_u16(v1, v3);

                v = vmlaq_u16(v, v2, vc6u16);
                v = vmlaq_u16(v, v13, vc4u16);

                uint8x8_t v8 = vrshrn_n_u16(v, 8);

                v0 = v4;

                vst1_u8(dst + x, vtbl1_u8(v8, map));
            }
        }
        break;
        case 4:
        {
            uint16x4_t vx1 = vld1_u16(lane - 2*4);
            uint16x4_t vx2 = vld1_u16(lane - 1*4);
            uint16x4_t vx3 = vld1_u16(lane + 0*4);
            uint16x8_t v0 = vcombine_u16(vx1, vx3);

            for (; x < roiw8; x += 8)
            {
                internal::prefetch(lane + 2 * x + 16);

                uint16x4_t vx_ = vld1_u16(lane + 2 * x - 1*4 + 8);
                uint16x4_t vx4 = vld1_u16(lane + 2 * x + 0*4 + 8);
                uint16x4_t vx5 = vld1_u16(lane + 2 * x + 1*4 + 8);
                uint16x4_t vx6 = vld1_u16(lane + 2 * x + 2*4 + 8);

                uint16x8_t v1 = vcombine_u16(vx2, vx_);
                uint16x8_t v2 = vcombine_u16(vget_high_u16(v0), vx4);
                uint16x8_t v3 = vcombine_u16(vx_, vx5);
                uint16x8_t v4 = vcombine_u16(vx4, vx6);
                vx2 = vx5;

                uint16x8_t v = vaddq_u16(v0, v4);
                uint16x8_t v13 = vaddq_u16(v1, v3);

                v = vmlaq_u16(v, v2, vc6u16);
                v = vmlaq_u16(v, v13, vc4u16);

                uint8x8_t v8 = vrshrn_n_u16(v, 8);

                v0 = v4;

                vst1_u8(dst + x, v8);
            }
        }
        break;
        }

        for (u32 h = 0; h < cn; ++h)
        {
            u16* ln = lane + h;
            u8* dt = dst + h;
            for (size_t k = x; k < dcolcn; k += cn)
                dt[k] = u8((ln[2*k-2*cn] + ln[2*k+2*cn] + 4u * (ln[2*k-cn] + ln[2*k+cn]) + 6u * ln[2*k] + (1 << 7)) >> 8);
        }
    }
#else
    // Remove 'unused parameter' warnings.
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void gaussianPyramidDown(const Size2D &srcSize,
                         const s16 *srcBase, ptrdiff_t srcStride,
                         const Size2D &dstSize,
                         s16 *dstBase, ptrdiff_t dstStride, u8 cn)
{
    internal::assertSupportedConfiguration(isGaussianPyramidDownS16Supported(srcSize, dstSize, cn));
#ifdef CAROTENE_NEON
    size_t dcolcn = dstSize.width*cn;
    size_t scolcn = srcSize.width*cn;
    size_t roiw4 = dcolcn - 3;

    size_t idx_l1 = borderInterpolate101(-1, srcSize.width) * cn;
    size_t idx_l2 = borderInterpolate101(-2, srcSize.width) * cn;
    size_t idx_r1 = borderInterpolate101(srcSize.width + 0, srcSize.width) * cn;
    size_t idx_r2 = borderInterpolate101(srcSize.width + 1, srcSize.width) * cn;

    //1-line buffer
    std::vector<s32> _buf(cn*(srcSize.width + 4) + 32/sizeof(s32));
    s32* lane = internal::alignPtr(&_buf[2*cn], 32);

    int16x4_t vc6s16 = vmov_n_s16(6);
    int32x4_t vc6s32 = vmovq_n_s32(6);
    int32x4_t vc4s32 = vmovq_n_s32(4);

    for (size_t i = 0; i < dstSize.height; ++i)
    {
        s16* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        const s16* ln0 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-2, srcSize.height));
        const s16* ln1 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-1, srcSize.height));
        const s16* ln2 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+0, srcSize.height));
        const s16* ln3 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+1, srcSize.height));
        const s16* ln4 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+2, srcSize.height));

        size_t x = 0;
        for (; x <= scolcn - 4; x += 4)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, (ptrdiff_t)x % 5 - 2));
            int16x4_t v0 = vld1_s16(ln0 + x);
            int16x4_t v1 = vld1_s16(ln1 + x);
            int16x4_t v2 = vld1_s16(ln2 + x);
            int16x4_t v3 = vld1_s16(ln3 + x);
            int16x4_t v4 = vld1_s16(ln4 + x);

            int32x4_t v = vaddl_s16(v0, v4);
            int32x4_t v13 = vaddl_s16(v1, v3);

            v = vmlal_s16(v, v2, vc6s16);
            v = vmlaq_s32(v, v13, vc4s32);

            vst1q_s32(lane + x, v);
        }
        for (; x < scolcn; ++x)
        {
            lane[x] = ln0[x] + ln4[x] + 4 * (ln1[x] + ln3[x]) + 6 * ln2[x];
        }

        //left&right borders
        for (u32 k = 0; k < cn; ++k)
        {
            lane[(s32)(-cn+k)] = lane[idx_l1 + k];
            lane[(s32)(-cn-cn+k)] = lane[idx_l2 + k];

            lane[scolcn+k] = lane[idx_r1 + k];
            lane[scolcn+cn+k] = lane[idx_r2 + k];
        }

        //horizontal convolution
        x = 0;
        switch(cn)
        {
        case 1:
            for (; x < roiw4; x += 4)
            {
                internal::prefetch(lane + 2 * x);
#if __GNUC_MINOR__ < 7
                __asm__ (
                    "vld2.32 {d0-d3}, [%[in0]]                              \n\t"
                    "vld2.32 {d4-d7}, [%[in4]]                              \n\t"
                    "vld2.32 {d12-d15}, [%[in1]]                            \n\t"
                    "vld2.32 {d16-d19}, [%[in3]]                            \n\t"
                    "vld2.32 {d8-d11}, [%[in2],:256]                        \n\t"
                    "vadd.i32 q0, q2                                        \n\t"
                    "vadd.i32 q6, q8                                        \n\t"
                    "vmla.i32 q0, q4, %q[c6]                                \n\t"
                    "vmla.i32 q0, q6, %q[c4]                                \n\t"
                    "vrshrn.s32 d8, q0, #8                                  \n\t"
                    "vst1.16 {d8}, [%[out]]                                 \n\t"
                    : /*no output*/
                    : [out] "r" (dst + x),
                      [in0] "r" (lane + 2*x-2),
                      [in1] "r" (lane + 2*x-1),
                      [in2] "r" (lane + 2*x+0),
                      [in3] "r" (lane + 2*x+1),
                      [in4] "r" (lane + 2*x+2),
                      [c4] "w" (vc4s32), [c6] "w" (vc6s32)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19"
                );
#else
                int32x4x2_t vLane0 = vld2q_s32(lane + 2*x-2);
                int32x4x2_t vLane1 = vld2q_s32(lane + 2*x-1);
                int32x4x2_t vLane2 = vld2q_s32(lane + 2*x+0);
                int32x4x2_t vLane3 = vld2q_s32(lane + 2*x+1);
                int32x4x2_t vLane4 = vld2q_s32(lane + 2*x+2);

                int32x4_t vSum_0_4 = vaddq_s32(vLane0.val[0], vLane4.val[0]);
                int32x4_t vSum_1_3 = vaddq_s32(vLane1.val[0], vLane3.val[0]);
                vSum_0_4 = vmlaq_s32(vSum_0_4, vLane2.val[0], vc6s32);
                vSum_0_4 = vmlaq_s32(vSum_0_4, vSum_1_3, vc4s32);
                int16x4_t vRes = vrshrn_n_s32(vSum_0_4, 8);

                vst1_s16(dst + x, vRes);
#endif
            }
            break;
        case 3:
        {
            int32x4_t v0 = vld1q_s32(lane - 2*3);
            int32x4_t v1 = vld1q_s32(lane - 1*3);
            int32x4_t v2 = vld1q_s32(lane + 0*3);
            for (; x < roiw4; x += 3)
            {
                internal::prefetch(lane + 2 * x);

                int32x4_t v3 = vld1q_s32(lane + 2 * x + 1*3);
                int32x4_t v4 = vld1q_s32(lane + 2 * x + 2*3);

                int32x4_t v = vaddq_s32(v0, v4);
                int32x4_t v13 = vaddq_s32(v1, v3);

                v = vmlaq_s32(v, v2, vc6s32);
                v = vmlaq_s32(v, v13, vc4s32);

                int16x4_t vv = vrshrn_n_s32(v, 8);

                v0 = v2;
                v1 = v3;
                v2 = v4;

                vst1_s16(dst + x, vv);
            }
        }
        break;
        case 4:
        {
            int32x4_t v0 = vld1q_s32(lane - 2*4);
            int32x4_t v1 = vld1q_s32(lane - 1*4);
            int32x4_t v2 = vld1q_s32(lane + 0*4);
            for (; x < roiw4; x += 4)
            {
                internal::prefetch(lane + 2 * x + 8);
                int32x4_t v3 = vld1q_s32(lane + 2 * x + 1*4);
                int32x4_t v4 = vld1q_s32(lane + 2 * x + 2*4);

                int32x4_t v = vaddq_s32(v0, v4);
                int32x4_t v13 = vaddq_s32(v1, v3);

                v = vmlaq_s32(v, v2, vc6s32);
                v = vmlaq_s32(v, v13, vc4s32);

                int16x4_t vv = vrshrn_n_s32(v, 8);

                v0 = v2;
                v1 = v3;
                v2 = v4;

                vst1_s16(dst + x, vv);
            }
        }
        break;
        }

        for (u32 h = 0; h < cn; ++h)
        {
            s32* ln = lane + h;
            s16* dt = dst + h;
            for (size_t k = x; k < dcolcn; k += cn)
                dt[k] = s16((ln[2*k-2*cn] + ln[2*k+2*cn] + 4 * (ln[2*k-cn] + ln[2*k+cn]) + 6 * ln[2*k] + (1 << 7)) >> 8);
        }
    }
#else
    // Remove 'unused parameter' warnings.
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void gaussianPyramidDown(const Size2D &srcSize,
                         const f32 *srcBase, ptrdiff_t srcStride,
                         const Size2D &dstSize,
                         f32 *dstBase, ptrdiff_t dstStride, u8 cn)
{
    internal::assertSupportedConfiguration(isGaussianPyramidDownF32Supported(srcSize, dstSize, cn));
#ifdef CAROTENE_NEON
    size_t dcolcn = dstSize.width*cn;
    size_t scolcn = srcSize.width*cn;
    size_t roiw4 = dcolcn - 3;

    size_t idx_l1 = borderInterpolate101(-1, srcSize.width) * cn;
    size_t idx_l2 = borderInterpolate101(-2, srcSize.width) * cn;
    size_t idx_r1 = borderInterpolate101(srcSize.width + 0, srcSize.width) * cn;
    size_t idx_r2 = borderInterpolate101(srcSize.width + 1, srcSize.width) * cn;

    //1-line buffer
    std::vector<f32> _buf(cn*(srcSize.width + 4) + 32/sizeof(f32));
    f32* lane = internal::alignPtr(&_buf[2*cn], 32);

#if __GNUC_MINOR__ < 7
    register float32x4_t vc6d4f32  asm ("q11") = vmovq_n_f32(1.5f);  // 6/4
    register float32x4_t vc1d4f32  asm ("q12") = vmovq_n_f32(0.25f); // 1/4

    register float32x4_t vc1d64f32 asm ("q13") = vmovq_n_f32(0.015625f); //1/4/16
    register float32x4_t vc4d64f32 asm ("q14") = vmovq_n_f32(0.0625f);   //4/4/16
    register float32x4_t vc6d64f32 asm ("q15") = vmovq_n_f32(0.09375f);  //6/4/16
#else
    register float32x4_t vc6d4f32  = vmovq_n_f32(1.5f);  // 6/4
    register float32x4_t vc1d4f32  = vmovq_n_f32(0.25f); // 1/4

    register float32x4_t vc1d64f32 = vmovq_n_f32(0.015625f); //1/4/16
    register float32x4_t vc4d64f32 = vmovq_n_f32(0.0625f);   //4/4/16
    register float32x4_t vc6d64f32 = vmovq_n_f32(0.09375f);  //6/4/16
#endif

    for (size_t i = 0; i < dstSize.height; ++i)
    {
        f32* dst = internal::getRowPtr(dstBase, dstStride, i);
        //vertical convolution
        const f32* ln0 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-2, srcSize.height));
        const f32* ln1 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2-1, srcSize.height));
        const f32* ln2 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+0, srcSize.height));
        const f32* ln3 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+1, srcSize.height));
        const f32* ln4 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i*2+2, srcSize.height));

        size_t x = 0;
        for (; x <= scolcn - 4; x += 4)
        {
            internal::prefetch(internal::getRowPtr(ln2 + x, srcStride, (ptrdiff_t)x % 5 - 2));
            float32x4_t v0 = vld1q_f32((const float32_t*)ln0 + x);
            float32x4_t v1 = vld1q_f32((const float32_t*)ln1 + x);
            float32x4_t v2 = vld1q_f32((const float32_t*)ln2 + x);
            float32x4_t v3 = vld1q_f32((const float32_t*)ln3 + x);
            float32x4_t v4 = vld1q_f32((const float32_t*)ln4 + x);

            float32x4_t v   = vaddq_f32(v1, v3);
            float32x4_t v04 = vaddq_f32(v0, v4);

            v = vmlaq_f32(v, v2, vc6d4f32);
            v = vmlaq_f32(v, v04, vc1d4f32);

            vst1q_f32(lane + x, v);
        }
        for (; x < scolcn; ++x)
        {
            lane[x] = 0.25f*(ln0[x] + ln4[x]) + (ln1[x] + ln3[x]) + 1.5f * ln2[x];
        }

        //left&right borders
        for (u32 k = 0; k < cn; ++k)
        {
            lane[(s32)(-cn+k)] = lane[idx_l1 + k];
            lane[(s32)(-cn-cn+k)] = lane[idx_l2 + k];

            lane[scolcn+k] = lane[idx_r1 + k];
            lane[scolcn+cn+k] = lane[idx_r2 + k];
        }

        //horizontal convolution
        x = 0;
        switch(cn)
        {
        case 1:
            for (; x < roiw4; x += 4)
            {
                internal::prefetch(lane + 2 * x);
#if __GNUC_MINOR__ < 7
                __asm__ __volatile__ (
                    "vld2.32 {d0-d3}, [%[in0]]                              \n\t"
                    "vld2.32 {d8-d11}, [%[in4]]                             \n\t"
                    "vld2.32 {d14-d17}, [%[in2],:256]                       \n\t"
                    "vld2.32 {d10-d13}, [%[in1]]                            \n\t"
                    "vld2.32 {d16-d19}, [%[in3]]                            \n\t"
                    "vmul.f32 q7, %q[c6d64]                                 \n\t"
                    "vadd.f32 q0, q4         @v04                           \n\t"
                    "vadd.f32 q5, q8         @v13                           \n\t"
                    "vmla.f32 q7, q0, %q[c1d64]                             \n\t"
                    "vmla.f32 q7, q5, %q[c4d64]                             \n\t"
                    "vst1.32 {d14-d15}, [%[out]]                            \n\t"
                    :
                    : [out] "r" (dst + x),
                      [in0] "r" (lane + 2*x-2),
                      [in1] "r" (lane + 2*x-1),
                      [in2] "r" (lane + 2*x+0),
                      [in3] "r" (lane + 2*x+1),
                      [in4] "r" (lane + 2*x+2),
                      [c4d64] "w" (vc4d64f32), [c6d64] "w" (vc6d64f32), [c1d64] "w" (vc1d64f32)
                    : "d0","d1","d2","d3","d4",/*"d5","d6","d7",*/"d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19" //ugly compiler "bug" - can't touch d5-d7
                );
#else
                float32x4x2_t vLane0 = vld2q_f32(lane + 2*x-2);
                float32x4x2_t vLane1 = vld2q_f32(lane + 2*x-1);
                float32x4x2_t vLane2 = vld2q_f32(lane + 2*x+0);
                float32x4x2_t vLane3 = vld2q_f32(lane + 2*x+1);
                float32x4x2_t vLane4 = vld2q_f32(lane + 2*x+2);

                float32x4_t vSum_0_4 = vaddq_f32(vLane0.val[0], vLane4.val[0]);
                float32x4_t vSum_1_3 = vaddq_f32(vLane1.val[0], vLane3.val[0]);
                float32x4_t vRes = vmulq_f32(vLane2.val[0], vc6d64f32);
                vRes = vmlaq_f32(vRes, vSum_0_4, vc1d64f32);
                vRes = vmlaq_f32(vRes, vSum_1_3, vc4d64f32);

                vst1q_f32(dst + x, vRes);
#endif
            }
            break;
        case 3:
        {
            float32x4_t v0 = vld1q_f32((const float32_t*)lane - 2*3);
            float32x4_t v1 = vld1q_f32((const float32_t*)lane - 1*3);
            float32x4_t v2 = vld1q_f32((const float32_t*)lane + 0*3);

            for (; x < roiw4; x += 3)
            {
                internal::prefetch(lane + 2 * x);

                float32x4_t v3 = vld1q_f32((const float32_t*)lane + 2 * x + 1*3);
                float32x4_t v4 = vld1q_f32((const float32_t*)lane + 2 * x + 2*3);

                float32x4_t v04 = vaddq_f32(v0, v4);
                float32x4_t v13 = vaddq_f32(v1, v3);

                float32x4_t v = vmulq_f32(v2, vc6d64f32);
                v = vmlaq_f32(v, v04, vc1d64f32);
                v = vmlaq_f32(v, v13, vc4d64f32);

                v0 = v2;
                v1 = v3;
                v2 = v4;

                vst1q_f32(dst + x, v);
            }
        }
        break;
        case 4:
        {
            float32x4_t v0 = vld1q_f32((const float32_t*)lane - 2*4);
            float32x4_t v1 = vld1q_f32((const float32_t*)lane - 1*4);
            float32x4_t v2 = vld1q_f32((const float32_t*)lane + 0*4);

            for (; x < roiw4; x += 4)
            {
                internal::prefetch(lane + 2 * x + 8);

                float32x4_t v3 = vld1q_f32((const float32_t*)lane + 2 * x + 1*4);
                float32x4_t v4 = vld1q_f32((const float32_t*)lane + 2 * x + 2*4);

                float32x4_t v04 = vaddq_f32(v0, v4);
                float32x4_t v13 = vaddq_f32(v1, v3);

                float32x4_t v = vmulq_f32(v2, vc6d64f32);
                v = vmlaq_f32(v, v04, vc1d64f32);
                v = vmlaq_f32(v, v13, vc4d64f32);

                v0 = v2;
                v1 = v3;
                v2 = v4;

                vst1q_f32(dst + x, v);
            }
        }
        break;
        }

        for (u32 h = 0; h < cn; ++h)
        {
            f32* ln = lane + h;
            f32* dt = dst + h;
            for (size_t k = x; k < dcolcn; k += cn)
                dt[k] = 0.015625f * (ln[2*k-2*cn] + ln[2*k+2*cn]) + 0.0625f * (ln[2*k-cn] + ln[2*k+cn]) + 0.09375f * ln[2*k];
        }
    }
#else
    // Remove 'unused parameter' warnings.
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void gaussianPyramidUp(const Size2D &srcSize,
                       const u8 *srcBase, ptrdiff_t srcStride,
                       const Size2D &dstSize,
                       u8 *dstBase, ptrdiff_t dstStride, u8 cn)
{
    internal::assertSupportedConfiguration(isGaussianPyramidUpU8Supported(srcSize, dstSize, cn));
#ifdef CAROTENE_NEON
    size_t dcolshn = (dstSize.width/2) * cn;
    size_t dcolshw = ((dstSize.width+1)/2) * cn;
    size_t scolsn = srcSize.width*cn;

    size_t idx_l =  (borderInterpolate101(-2, 2 * srcSize.width)/2) * cn;
    size_t idx_r1 = (borderInterpolate101(2 * srcSize.width + 0, 2 * srcSize.width)/2) * cn;
    size_t idx_r2 = (borderInterpolate101(2 * srcSize.width + 2, 2 * srcSize.width + 2)/2) * cn;

    //2-lines buffer
    std::vector<u16> _buf(2*(cn*(srcSize.width + 3) + 32/sizeof(u16)));
    u16* lane0 = internal::alignPtr(&_buf[cn], 32);
    u16* lane1 = internal::alignPtr(lane0 + (3 + srcSize.width)*cn, 32);

    uint8x8_t vc6u8 = vmov_n_u8(6);
    uint16x8_t vc6u16 = vmovq_n_u16(6);

    for (size_t i = 0; i < (dstSize.height + 1)/2; ++i)
    {
        u8* dst = internal::getRowPtr(dstBase, dstStride, 2*i);
        //vertical convolution
        const u8* ln0 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i * 2 - 2, srcSize.height * 2)/2);
        const u8* ln1 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i * 2 + 0, srcSize.height * 2)/2);
        const u8* ln2 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i * 2 + 2, srcSize.height * 2)/2);

        size_t x = 0;
        for (; x <= scolsn - 8; x += 8)
        {
            internal::prefetch(internal::getRowPtr(ln1 + x, srcStride, (ptrdiff_t)x % 3 - 1));
            uint8x8_t v0 = vld1_u8(ln0+x);
            uint8x8_t v2 = vld1_u8(ln2+x);
            uint8x8_t v1 = vld1_u8(ln1+x);

            uint16x8_t vl0 = vaddl_u8(v0, v2);
            uint16x8_t vl1 = vaddl_u8(v1, v2);

            vl0 = vmlal_u8(vl0, v1, vc6u8);
            vl1 = vshlq_n_u16(vl1, 2);

            vst1q_u16(lane0 + x, vl0);
            vst1q_u16(lane1 + x, vl1);
        }
        for (; x < scolsn; ++x)
        {
            lane0[x] = ln0[x] + ln2[x] + 6u * ln1[x];
            lane1[x] = 4u * (ln1[x] + ln2[x]);
        }

        //left&right borders
        for (u32 k = 0; k < cn; ++k)
        {
            lane0[(s32)(-cn+k)] = lane0[idx_l + k];
            lane1[(s32)(-cn+k)] = lane1[idx_l + k];

            lane0[scolsn+k] = lane0[idx_r1 + k];
            lane0[scolsn+cn+k] = lane0[idx_r2 + k];
            lane1[scolsn+k] = lane1[idx_r1 + k];
            lane1[scolsn+cn+k] = lane1[idx_r2 + k];
        }

        //horizontal convolution
        const u16* lane = lane0;
pyrUp8uHorizontalConvolution:
        x = 0;
        size_t lim;
        switch(cn)
        {
        case 1:
            lim = dcolshn > 7 ? dcolshn - 7 : 0;
            for (; x < lim; x += 8)
            {
                internal::prefetch(lane + x);
#if defined(__GNUC__) && defined(__arm__)
                __asm__ (
                    "vld1.16 {d0-d1}, [%[in0]]       /*q0 = v0*/            \n\t"
                    "vld1.16 {d2-d3}, [%[in2]]       /*q1 = v2*/            \n\t"
                    "vld1.16 {d4-d5}, [%[in1],:128]  /*q2 = v1*/            \n\t"
                    "vadd.i16 q0, q1                 /*q0 = v0 + v2*/       \n\t"
                    "vadd.i16 q3, q1, q2             /*q3 = v1 + v2*/       \n\t"
                    "vmla.i16 q0, q2, %q[c6]         /*q0 += v1*6*/         \n\t"
                    "vrshrn.u16 d9, q3, #4                                  \n\t"
                    "vrshrn.u16 d8, q0, #6                                  \n\t"
                    "vst2.8 {d8-d9}, [%[out]]                               \n\t"
                    : /*no output*/
                    : [out] "r" (dst + x*2),
                      [in0] "r" (lane + x - 1),
                      [in1] "r" (lane + x + 0),
                      [in2] "r" (lane + x + 1),
                      [c6] "w" (vc6u16)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
                );
#else
                uint16x8_t vLane0 = vld1q_u16(lane + x - 1);
                uint16x8_t vLane1 = vld1q_u16(lane + x + 0);
                uint16x8_t vLane2 = vld1q_u16(lane + x + 1);

                vLane0 = vaddq_u16(vLane0, vLane2);
                vLane2 = vaddq_u16(vLane2, vLane1);
                vLane0 = vmlaq_u16(vLane0, vLane1, vc6u16);
                uint8x8x2_t vRes;
                vRes.val[0] = vrshrn_n_u16(vLane0, 6);
                vRes.val[1] = vrshrn_n_u16(vLane2, 4);

                vst2_u8(dst + x*2, vRes);
#endif
            }
            break;
        case 3:
        {
            lim = dcolshn > 23 ? dcolshn - 23 : 0;
            for (; x < lim; x += 24)
            {
                internal::prefetch(lane + x);
#if defined(__GNUC__) && defined(__arm__)
                __asm__ (
                    "vmov.u16 q9, #6                                           \n\t"
                    "vld3.16 {d0, d2, d4}, [%[in0]]        /*v0*/              \n\t"
                    "vld3.16 {d1, d3, d5}, [%[in02]]                           \n\t"
                    "vld3.16 {d6, d8, d10}, [%[in2]]       /*v2*/              \n\t"
                    "vld3.16 {d7, d9, d11}, [%[in22]]                          \n\t"
                    "vld3.16 {d12, d14, d16}, [%[in1]]     /*v1*/              \n\t"
                    "vld3.16 {d13, d15, d17}, [%[in12]]                        \n\t"
                    "vadd.i16 q0, q3                       /*v0 + v2*/         \n\t"
                    "vadd.i16 q1, q4                       /*v0 + v2*/         \n\t"
                    "vadd.i16 q2, q5                       /*v0 + v2*/         \n\t"
                    "vadd.i16 q3, q6                       /*v1 + v2*/         \n\t"
                    "vadd.i16 q4, q7                       /*v1 + v2*/         \n\t"
                    "vadd.i16 q5, q8                       /*v1 + v2*/         \n\t"
                    "vmla.i16 q0, q6, q9                   /*v0 + v2 + v1*6 */ \n\t"
                    "vmla.i16 q1, q7, q9                   /*v0 + v2 + v1*6 */ \n\t"
                    "vmla.i16 q2, q8, q9                   /*v0 + v2 + v1*6 */ \n\t"
                    "vrshrn.u16 d19, q3, #4                                    \n\t"
                    "vrshrn.u16 d21, q4, #4                                    \n\t"
                    "vrshrn.u16 d23, q5, #4                                    \n\t"
                    "vrshrn.u16 d18, q0, #6                                    \n\t"
                    "vrshrn.u16 d20, q1, #6                                    \n\t"
                    "vrshrn.u16 d22, q2, #6                                    \n\t"
                    "vzip.8 d18, d19                                           \n\t"
                    "vzip.8 d20, d21                                           \n\t"
                    "vzip.8 d22, d23                                           \n\t"
                    "vst3.8 {d18, d20, d22}, [%[out1]]                         \n\t"
                    "vst3.8 {d19, d21, d23}, [%[out2]]                         \n\t"
                    : /*no output*/
                    : [out1] "r" (dst + 2 * x),
                      [out2] "r" (dst + 2 * x + 24),
                      [in0]  "r" (lane + x - 3),
                      [in02] "r" (lane + x + 9),
                      [in1]  "r" (lane + x),
                      [in12] "r" (lane + x + 12),
                      [in2]  "r" (lane + x + 3),
                      [in22] "r" (lane + x + 15)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23"
                );
#else
                uint16x8_t vc6 = vmovq_n_u16(6);
                uint16x8x3_t vLane0  = vld3q_u16(lane + x - 3);
                uint16x8x3_t vLane1  = vld3q_u16(lane + x + 0);
                uint16x8x3_t vLane2  = vld3q_u16(lane + x + 3);

                uint16x8_t vSum_0_3 = vaddq_u16(vLane0.val[0], vLane2.val[0]);
                uint16x8_t vSum_1_4 = vaddq_u16(vLane0.val[1], vLane2.val[1]);
                uint16x8_t vSum_2_5 = vaddq_u16(vLane0.val[2], vLane2.val[2]);
                uint16x8_t vSum_3_6 = vaddq_u16(vLane2.val[0], vLane1.val[0]);
                uint16x8_t vSum_4_7 = vaddq_u16(vLane2.val[1], vLane1.val[1]);
                uint16x8_t vSum_5_8 = vaddq_u16(vLane2.val[2], vLane1.val[2]);

                vSum_0_3 = vmlaq_u16(vSum_0_3, vLane1.val[0], vc6);
                vSum_1_4 = vmlaq_u16(vSum_1_4, vLane1.val[1], vc6);
                vSum_2_5 = vmlaq_u16(vSum_2_5, vLane1.val[2], vc6);

                uint8x8x2_t vSumShr3;
                vSumShr3.val[0] = vrshrn_n_u16(vSum_3_6, 4);
                vSumShr3.val[1] = vrshrn_n_u16(vSum_0_3, 6);;
                uint8x8x2_t vSumShr4;
                vSumShr4.val[0] = vrshrn_n_u16(vSum_4_7, 4);
                vSumShr4.val[1] = vrshrn_n_u16(vSum_1_4, 6);
                uint8x8x2_t vSumShr5;
                vSumShr5.val[0] = vrshrn_n_u16(vSum_5_8, 4);
                vSumShr5.val[1] = vrshrn_n_u16(vSum_2_5, 6);

                vSumShr3 = vzip_u8(vSumShr3.val[1], vSumShr3.val[0]);
                vSumShr4 = vzip_u8(vSumShr4.val[1], vSumShr4.val[0]);
                vSumShr5 = vzip_u8(vSumShr5.val[1], vSumShr5.val[0]);

                uint8x8x3_t vRes1;
                vRes1.val[0] = vSumShr3.val[0];
                vRes1.val[1] = vSumShr4.val[0];
                vRes1.val[2] = vSumShr5.val[0];
                vst3_u8(dst + 2 * x,      vRes1);

                uint8x8x3_t vRes2;
                vRes2.val[0] = vSumShr3.val[1];
                vRes2.val[1] = vSumShr4.val[1];
                vRes2.val[2] = vSumShr5.val[1];
                vst3_u8(dst + 2 * x + 24, vRes2);
#endif
            }
        }
        break;
        case 4:
            lim = dcolshn > 7 ? dcolshn - 7 : 0;
            for (; x < lim; x += 8)
            {
                internal::prefetch(lane + x);
#if defined(__GNUC__) && defined(__arm__)
                __asm__ (
                    "vld1.16 {d0-d1}, [%[in0]]       /*q0 = v0*/            \n\t"
                    "vld1.16 {d2-d3}, [%[in2]]       /*q1 = v2*/            \n\t"
                    "vld1.16 {d4-d5}, [%[in1],:128]  /*q2 = v1*/            \n\t"
                    "vadd.i16 q0, q1                 /*q0 = v0 + v2*/       \n\t"
                    "vadd.i16 q3, q1, q2             /*q3 = v1 + v2*/       \n\t"
                    "vmla.i16 q0, q2, %q[c6]         /*q0 += v1*6*/         \n\t"
                    "vrshrn.u16 d9, q3, #4                                  \n\t"
                    "vrshrn.u16 d8, q0, #6                                  \n\t"
                    "vst2.32 {d8-d9}, [%[out]]                              \n\t"
                    : /*no output*/
                    : [out] "r" (dst + x*2),
                      [in0] "r" (lane + x-4),
                      [in1] "r" (lane + x),
                      [in2] "r" (lane + x+4),
                      [c6] "w" (vc6u16)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
                );
#else
                uint16x8_t vLane0 = vld1q_u16(lane + x-4);
                uint16x8_t vLane1 = vld1q_u16(lane + x+0);
                uint16x8_t vLane2 = vld1q_u16(lane + x+4);

                vLane0 = vaddq_u16(vLane0, vLane2);
                vLane2 = vaddq_u16(vLane2, vLane1);
                vLane0 = vmlaq_u16(vLane0, vLane1, vc6u16);
                uint32x2x2_t vRes;
                vRes.val[1] = vreinterpret_u32_u8(vrshrn_n_u16(vLane2, 4));
                vRes.val[0] = vreinterpret_u32_u8(vrshrn_n_u16(vLane0, 6));

                vst2_u32((uint32_t*)(dst + x*2), vRes);
#endif
            }
            break;
        };

        for (u32 h = 0; h < cn; ++h)
        {
            const u16* ln = lane + h;
            u8* dt = dst + h;
            size_t k = x;
            for (; k < dcolshn; k += cn)
            {
                dt[2*k+0] = u8((ln[(ptrdiff_t)(k-cn)] + ln[k+cn] + 6u * ln[k] + (1 << 5)) >> 6);
                dt[2*k+cn] = u8(((ln[k] + ln[k+cn]) * 4u + (1 << 5)) >> 6);
            }
            for (; k < dcolshw; k += cn)
                dt[2*k] = u8((ln[(ptrdiff_t)(k-cn)] + ln[k+cn] + 6u * ln[k] + (1 << 5)) >> 6);
        }
        dst = internal::getRowPtr(dstBase, dstStride, 2*i+1);

        //second row
        if (lane == lane0 && 2*i+1 < dstSize.height)
        {
            lane = lane1;
            goto pyrUp8uHorizontalConvolution;
        }
    }
#else
    // Remove 'unused parameter' warnings.
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void gaussianPyramidUp(const Size2D &srcSize,
                       const s16 *srcBase, ptrdiff_t srcStride,
                       const Size2D &dstSize,
                       s16 *dstBase, ptrdiff_t dstStride, u8 cn)
{
    internal::assertSupportedConfiguration(isGaussianPyramidUpS16Supported(srcSize, dstSize, cn));
#ifdef CAROTENE_NEON
    size_t dcolshn = (dstSize.width/2) * cn;
    size_t dcolshw = ((dstSize.width+1)/2) * cn;
    size_t scolsn = srcSize.width*cn;

    size_t idx_l =  (borderInterpolate101(-2, 2 * srcSize.width)/2) * cn;
    size_t idx_r1 = (borderInterpolate101(2 * srcSize.width + 0, 2 * srcSize.width)/2) * cn;
    size_t idx_r2 = (borderInterpolate101(2 * srcSize.width + 2, 2 * srcSize.width + 2)/2) * cn;

    //2-lines buffer
    std::vector<s32> _buf(2*(cn*(srcSize.width + 3) + 32/sizeof(s32)));
    s32* lane0 = internal::alignPtr(&_buf[cn], 32);
    s32* lane1 = internal::alignPtr(lane0 + (3 + srcSize.width)*cn, 32);

    int16x4_t vc6s16 = vmov_n_s16(6);
    int32x4_t vc6s32 = vmovq_n_s32(6);

    for (size_t i = 0; i < (dstSize.height + 1)/2; ++i)
    {
        s16* dst = internal::getRowPtr(dstBase, dstStride, 2*i);
        //vertical convolution
        const s16* ln0 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i * 2 - 2, srcSize.height * 2)/2);
        const s16* ln1 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i * 2 + 0, srcSize.height * 2)/2);
        const s16* ln2 = internal::getRowPtr(srcBase, srcStride, borderInterpolate101(i * 2 + 2, srcSize.height * 2)/2);

        size_t x = 0;
        for (; x <= scolsn - 4; x += 4)
        {
            internal::prefetch(internal::getRowPtr(ln1 + x, srcStride, (ptrdiff_t)x % 3 - 1));
            int16x4_t v0 = vld1_s16(ln0 + x);
            int16x4_t v2 = vld1_s16(ln2 + x);
            int16x4_t v1 = vld1_s16(ln1 + x);

            int32x4_t vl0 = vaddl_s16(v0, v2);
            int32x4_t vl1 = vaddl_s16(v1, v2);

            vl0 = vmlal_s16(vl0, v1, vc6s16);
            vl1 = vshlq_n_s32(vl1, 2);

            vst1q_s32(lane0 + x, vl0);
            vst1q_s32(lane1 + x, vl1);
        }
        for (; x < scolsn; ++x)
        {
            lane0[x] = ln0[x] + ln2[x] + 6 * ln1[x];
            lane1[x] = 4 * (ln1[x] + ln2[x]);
        }

        //left&right borders
        for (u32 k = 0; k < cn; ++k)
        {
            lane0[(s32)(-cn+k)] = lane0[idx_l + k];
            lane1[(s32)(-cn+k)] = lane1[idx_l + k];

            lane0[scolsn+k] = lane0[idx_r1 + k];
            lane0[scolsn+cn+k] = lane0[idx_r2 + k];
            lane1[scolsn+k] = lane1[idx_r1 + k];
            lane1[scolsn+cn+k] = lane1[idx_r2 + k];
        }

        //horizontal convolution
        const s32* lane = lane0;
pyrUp16sHorizontalConvolution:
        x = 0;
        size_t lim;
        switch(cn)
        {
        case 1:
            lim = dcolshn > 3 ? dcolshn - 3 : 0;
            for (; x < lim; x += 4)
            {
                internal::prefetch(lane + x);
#if defined(__GNUC__) && defined(__arm__)
                __asm__ (
                    "vld1.32 {d0-d1}, [%[in0]]       /*q0 = v0*/            \n\t"
                    "vld1.32 {d2-d3}, [%[in2]]       /*q1 = v2*/            \n\t"
                    "vld1.32 {d4-d5}, [%[in1],:128]  /*q2 = v1*/            \n\t"
                    "vadd.i32 q0, q0, q1             /*q0 = v0 + v2*/       \n\t"
                    "vadd.i32 q3, q1, q2             /*q3 = v1 + v2*/       \n\t"
                    "vmla.i32 q0, q2, %q[c6]         /*q0 += v1*6*/         \n\t"
                    "vrshrn.s32 d9, q3, #4                                  \n\t"
                    "vrshrn.s32 d8, q0, #6                                  \n\t"
                    "vst2.16 {d8-d9}, [%[out]]                              \n\t"
                    : /*no output*/
                    : [out] "r" (dst + x * 2),
                      [in0] "r" (lane + x - 1),
                      [in1] "r" (lane + x),
                      [in2] "r" (lane + x + 1),
                      [c6] "w" (vc6s32)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
                );
#else
                int32x4_t vLane0 = vld1q_s32(lane + x - 1);
                int32x4_t vLane1 = vld1q_s32(lane + x);
                int32x4_t vLane2 = vld1q_s32(lane + x + 1);

                vLane0 = vaddq_s32(vLane0, vLane2);
                vLane2 = vaddq_s32(vLane2, vLane1);
                vLane0 = vmlaq_s32(vLane0, vLane1, vc6s32);
                int16x4x2_t vRes;
                vRes.val[0] = vrshrn_n_s32(vLane0, 6);
                vRes.val[1] = vrshrn_n_s32(vLane2, 4);

                vst2_s16(dst + x * 2, vRes);
#endif
            }
            break;
        case 3:
        {
            lim = dcolshn > 11 ? dcolshn - 11 : 0;
            for (; x < lim; x += 12)
            {
                internal::prefetch(lane + x + 3);
#if defined(__GNUC__) && defined(__arm__)
                __asm__ (
                    "vmov.s32 q9, #6                                           \n\t"
                    "vld3.32 {d0, d2, d4}, [%[in0]]        /*v0*/              \n\t"
                    "vld3.32 {d1, d3, d5}, [%[in2]]                            \n\t"
                    "vld3.32 {d6, d8, d10}, [%[in2]]       /*v2*/              \n\t"
                    "vld3.32 {d7, d9, d11}, [%[in22]]                          \n\t"
                    "vld3.32 {d12, d14, d16}, [%[in1]]     /*v1*/              \n\t"
                    "vld3.32 {d13, d15, d17}, [%[in12]]                        \n\t"
                    "vadd.i32 q0, q3                       /*v0 + v2*/         \n\t"
                    "vadd.i32 q1, q4                       /*v0 + v2*/         \n\t"
                    "vadd.i32 q2, q5                       /*v0 + v2*/         \n\t"
                    "vadd.i32 q3, q6                       /*v1 + v2*/         \n\t"
                    "vadd.i32 q4, q7                       /*v1 + v2*/         \n\t"
                    "vadd.i32 q5, q8                       /*v1 + v2*/         \n\t"
                    "vmla.i32 q0, q6, q9                   /*v0 + v2 + v1*6 */ \n\t"
                    "vmla.i32 q1, q7, q9                   /*v0 + v2 + v1*6 */ \n\t"
                    "vmla.i32 q2, q8, q9                   /*v0 + v2 + v1*6 */ \n\t"
                    "vrshrn.s32 d19, q3, #4                                    \n\t"
                    "vrshrn.s32 d21, q4, #4                                    \n\t"
                    "vrshrn.s32 d23, q5, #4                                    \n\t"
                    "vrshrn.s32 d18, q0, #6                                    \n\t"
                    "vrshrn.s32 d20, q1, #6                                    \n\t"
                    "vrshrn.s32 d22, q2, #6                                    \n\t"
                    "vzip.16 d18, d19                                          \n\t"
                    "vzip.16 d20, d21                                          \n\t"
                    "vzip.16 d22, d23                                          \n\t"
                    "vst3.16 {d18, d20, d22}, [%[out1]]                        \n\t"
                    "vst3.16 {d19, d21, d23}, [%[out2]]                        \n\t"
                    : /*no output*/
                    : [out1] "r" (dst + 2*x),
                      [out2] "r" (dst + 2*x + 12),
                      [in0]  "r" (lane + x - 3),
                      [in1]  "r" (lane + x),
                      [in12] "r" (lane + x + 6),
                      [in2]  "r" (lane + x + 3),
                      [in22] "r" (lane + x + 9)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23"
                );
#else
                int32x4_t vc6 = vmovq_n_s32(6);
                int32x4x3_t vLane0  = vld3q_s32(lane + x - 3);
                int32x4x3_t vLane1  = vld3q_s32(lane + x);
                int32x4x3_t vLane2  = vld3q_s32(lane + x + 3);

                int32x4_t vSum_0_3 = vaddq_s32(vLane0.val[0], vLane2.val[0]);
                int32x4_t vSum_1_4 = vaddq_s32(vLane0.val[1], vLane2.val[1]);
                int32x4_t vSum_2_5 = vaddq_s32(vLane0.val[2], vLane2.val[2]);
                int32x4_t vSum_3_6 = vaddq_s32(vLane2.val[0], vLane1.val[0]);
                int32x4_t vSum_4_7 = vaddq_s32(vLane2.val[1], vLane1.val[1]);
                int32x4_t vSum_5_8 = vaddq_s32(vLane2.val[2], vLane1.val[2]);

                vSum_0_3 = vmlaq_s32(vSum_0_3, vLane1.val[0], vc6);
                vSum_1_4 = vmlaq_s32(vSum_1_4, vLane1.val[1], vc6);
                vSum_2_5 = vmlaq_s32(vSum_2_5, vLane1.val[2], vc6);

                int16x4x2_t vSumShr1;
                vSumShr1.val[1] = vrshrn_n_s32(vSum_3_6, 4);
                vSumShr1.val[0] = vrshrn_n_s32(vSum_0_3, 6);

                int16x4x2_t vSumShr2;
                vSumShr2.val[1] = vrshrn_n_s32(vSum_4_7, 4);
                vSumShr2.val[0] = vrshrn_n_s32(vSum_1_4, 6);

                int16x4x2_t vSumShr3;
                vSumShr3.val[1] = vrshrn_n_s32(vSum_5_8, 4);
                vSumShr3.val[0] = vrshrn_n_s32(vSum_2_5, 6);

                vSumShr1 = vzip_s16(vSumShr1.val[0], vSumShr1.val[1]);
                vSumShr2 = vzip_s16(vSumShr2.val[0], vSumShr2.val[1]);
                vSumShr3 = vzip_s16(vSumShr3.val[0], vSumShr3.val[1]);

                int16x4x3_t vRes1;
                vRes1.val[0] = vSumShr1.val[0];
                vRes1.val[1] = vSumShr2.val[0];
                vRes1.val[2] = vSumShr3.val[0];
                vst3_s16((int16_t*)(dst + 2 * x), vRes1);

                int16x4x3_t vRes2;
                vRes2.val[0] = vSumShr1.val[1];
                vRes2.val[1] = vSumShr2.val[1];
                vRes2.val[2] = vSumShr3.val[1];
                vst3_s16(dst + 2 * x + 12, vRes2);
#endif
            }
        }
        break;
        case 4:
            lim = dcolshn > 3 ? dcolshn - 3 : 0;
            for (; x < lim; x += 4)
            {
                internal::prefetch(lane + x);
#if defined(__GNUC__) && defined(__arm__)
                __asm__ (
                    "vld1.32 {d0-d1}, [%[in0]]       /*q0 = v0*/            \n\t"
                    "vld1.32 {d2-d3}, [%[in2]]       /*q1 = v2*/            \n\t"
                    "vld1.32 {d4-d5}, [%[in1],:128]  /*q2 = v1*/            \n\t"
                    "vadd.i32 q0, q1                 /*q0 = v0 + v2*/       \n\t"
                    "vadd.i32 q3, q1, q2             /*q3 = v1 + v2*/       \n\t"
                    "vmla.i32 q0, q2, %q[c6]         /*q0 += v1*6*/         \n\t"
                    "vrshrn.s32 d9, q3, #4                                  \n\t"
                    "vrshrn.s32 d8, q0, #6                                  \n\t"
                    "vst1.16 {d8-d9}, [%[out]]                              \n\t"
                    : /*no output*/
                    : [out] "r" (dst + x * 2),
                      [in0] "r" (lane + x - 4),
                      [in1] "r" (lane + x),
                      [in2] "r" (lane + x + 4),
                      [c6] "w" (vc6s32)
                    : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
                );
#else
                int32x4_t vLane0 = vld1q_s32(lane + x - 4);
                int32x4_t vLane1 = vld1q_s32(lane + x);
                int32x4_t vLane2 = vld1q_s32(lane + x + 4);

                vLane0 = vaddq_s32(vLane0, vLane2);
                vLane2 = vaddq_s32(vLane2, vLane1);
                vLane0 = vmlaq_s32(vLane0, vLane1, vc6s32);
                int16x4x2_t vRes;
                vRes.val[0] = vrshrn_n_s32(vLane0, 6);
                vRes.val[1] = vrshrn_n_s32(vLane2, 4);

                vst1q_s16(dst + x * 2, vcombine_s16(vRes.val[0], vRes.val[1]));
#endif
            }
            break;
        };

        for (u32 h = 0; h < cn; ++h)
        {
            const s32* ln = lane + h;
            s16* dt = dst + h;
            size_t k = x;
            for (; k < dcolshn; k += cn)
            {
                dt[2*k+0] = s16((ln[(ptrdiff_t)(k-cn)] + ln[k+cn] + 6 * ln[k] + (1 << 5)) >> 6);
                dt[2*k+cn] = s16(((ln[k] + ln[k+cn]) * 4 + (1 << 5)) >> 6);
            }
            for (; k < dcolshw; k += cn)
                dt[2*k] = s16((ln[(ptrdiff_t)(k-cn)] + ln[k+cn] + 6 * ln[k] + (1 << 5)) >> 6);
        }
        dst = internal::getRowPtr(dstBase, dstStride, 2*i+1);

        //second row
        if (lane == lane0 && 2*i+1 < dstSize.height)
        {
            lane = lane1;
            goto pyrUp16sHorizontalConvolution;
        }
    }
#else
    // Remove 'unused parameter' warnings.
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

} // namespace CAROTENE_NS
