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
 * Copyright (C) 2015, NVIDIA Corporation, all rights reserved.
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

#include "remap.hpp"

namespace CAROTENE_NS {

bool isWarpAffineNearestNeighborSupported(const Size2D &ssize)
{
#if SIZE_MAX > UINT32_MAX
    return !(ssize.width > 0xffffFFFF || ssize.height > 0xffffFFFF) && // Restrict image size since internal index evaluation
                                                                       // is performed with u32
           isSupportedConfiguration();
#else
    (void)ssize;
    return isSupportedConfiguration();
#endif
}

bool isWarpAffineLinearSupported(const Size2D &ssize)
{
#if SIZE_MAX > UINT32_MAX
    return !(ssize.width > 0xffffFFFF || ssize.height > 0xffffFFFF) && // Restrict image size since internal index evaluation
                                                                       // is performed with u32
           isSupportedConfiguration();
#else
    (void)ssize;
    return isSupportedConfiguration();
#endif
}

void warpAffineNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                               const u8 * srcBase, ptrdiff_t srcStride,
                               const f32 * m,
                               u8 * dstBase, ptrdiff_t dstStride,
                               BORDER_MODE borderMode, u8 borderValue)
{
    internal::assertSupportedConfiguration(isWarpAffineNearestNeighborSupported(ssize));
#ifdef CAROTENE_NEON
    using namespace internal;

    s32 _map[BLOCK_SIZE * BLOCK_SIZE + 16];
    s32 * map = alignPtr(_map, 16);

    int32x4_t v_width4 = vdupq_n_s32(ssize.width - 1), v_height4 = vdupq_n_s32(ssize.height - 1);
    int32x4_t v_step4 = vdupq_n_s32(srcStride);
    float32x4_t v_4 = vdupq_n_f32(4.0f);

    float32x4_t v_m0 = vdupq_n_f32(m[0]);
    float32x4_t v_m1 = vdupq_n_f32(m[1]);
    float32x4_t v_m2 = vdupq_n_f32(m[2]);
    float32x4_t v_m3 = vdupq_n_f32(m[3]);
    float32x4_t v_m4 = vdupq_n_f32(m[4]);
    float32x4_t v_m5 = vdupq_n_f32(m[5]);

    if (borderMode == BORDER_MODE_REPLICATE)
    {
        int32x4_t v_zero4 = vdupq_n_s32(0);

        for (size_t i = 0; i < dsize.height; i += BLOCK_SIZE)
        {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, dsize.height - i);
            for (size_t j = 0; j < dsize.width; j += BLOCK_SIZE)
            {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, dsize.width - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y)
                {
                    s32 * map_row = getRowPtr(&map[0], blockWidth * sizeof(s32), y);

                    size_t x = 0, y_ = y + i;
                    f32 indeces[4] = { j + 0.0f, j + 1.0f, j + 2.0f, j + 3.0f };
                    float32x4_t v_x = vld1q_f32(indeces), v_y = vdupq_n_f32(y_);
                    float32x4_t v_yx = vmlaq_f32(v_m4, v_m2, v_y), v_yy = vmlaq_f32(v_m5, v_m3, v_y);

                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4_t v_src_xf = vmlaq_f32(v_yx, v_m0, v_x);
                        float32x4_t v_src_yf = vmlaq_f32(v_yy, v_m1, v_x);

                        int32x4_t v_src_x = vmaxq_s32(v_zero4, vminq_s32(v_width4, vcvtq_s32_f32(v_src_xf)));
                        int32x4_t v_src_y = vmaxq_s32(v_zero4, vminq_s32(v_height4, vcvtq_s32_f32(v_src_yf)));
                        int32x4_t v_src_index = vmlaq_s32(v_src_x, v_src_y, v_step4);
                        vst1q_s32(map_row + x, v_src_index);

                        v_x = vaddq_f32(v_x, v_4);
                    }

                    f32 yx = m[2] * y_ + m[4], yy = m[3] * y_ + m[5];
                    for (ptrdiff_t x_ = x + j; x < blockWidth; ++x, ++x_)
                    {
                        f32 src_x_f = m[0] * x_ + yx;
                        f32 src_y_f = m[1] * x_ + yy;
                        s32 src_x = floorf(src_x_f), src_y = floorf(src_y_f);

                        src_x = std::max(0, std::min<s32>(ssize.width - 1, src_x));
                        src_y = std::max(0, std::min<s32>(ssize.height - 1, src_y));
                        map_row[x] = src_y * srcStride + src_x;
                    }
                }

                // make remap
                remapNearestNeighborReplicate(Size2D(blockWidth, blockHeight), srcBase, &map[0],
                                                        getRowPtr(dstBase, dstStride, i) + j, dstStride);
            }
        }
    }
    else if (borderMode == BORDER_MODE_CONSTANT)
    {
        int32x4_t v_m1_4 = vdupq_n_s32(-1);
        float32x4_t v_zero4 = vdupq_n_f32(0.0f);

        for (size_t i = 0; i < dsize.height; i += BLOCK_SIZE)
        {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, dsize.height - i);
            for (size_t j = 0; j < dsize.width; j += BLOCK_SIZE)
            {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, dsize.width - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y)
                {
                    s32 * map_row = getRowPtr(&map[0], blockWidth * sizeof(s32), y);

                    size_t x = 0, y_ = y + i;
                    f32 indeces[4] = { j + 0.0f, j + 1.0f, j + 2.0f, j + 3.0f };
                    float32x4_t v_x = vld1q_f32(indeces), v_y = vdupq_n_f32(y_);
                    float32x4_t v_yx = vmlaq_f32(v_m4, v_m2, v_y), v_yy = vmlaq_f32(v_m5, v_m3, v_y);

                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4_t v_src_xf = vmlaq_f32(v_yx, v_m0, v_x);
                        float32x4_t v_src_yf = vmlaq_f32(v_yy, v_m1, v_x);

                        int32x4_t v_src_x = vcvtq_s32_f32(v_src_xf);
                        int32x4_t v_src_y = vcvtq_s32_f32(v_src_yf);
                        uint32x4_t v_mask = vandq_u32(vandq_u32(vcgeq_f32(v_src_xf, v_zero4), vcleq_s32(v_src_x, v_width4)),
                                                      vandq_u32(vcgeq_f32(v_src_yf, v_zero4), vcleq_s32(v_src_y, v_height4)));
                        int32x4_t v_src_index = vbslq_s32(v_mask, vmlaq_s32(v_src_x, v_src_y, v_step4), v_m1_4);
                        vst1q_s32(map_row + x, v_src_index);

                        v_x = vaddq_f32(v_x, v_4);
                    }

                    f32 yx = m[2] * y_ + m[4], yy = m[3] * y_ + m[5];
                    for (ptrdiff_t x_ = x + j; x < blockWidth; ++x, ++x_)
                    {
                        f32 src_x_f = m[0] * x_ + yx;
                        f32 src_y_f = m[1] * x_ + yy;
                        s32 src_x = floorf(src_x_f), src_y = floorf(src_y_f);

                        map_row[x] = (src_x >= 0) && (src_x < (s32)ssize.width) &&
                                     (src_y >= 0) && (src_y < (s32)ssize.height) ? src_y * srcStride + src_x : -1;
                    }
                }

                // make remap
                remapNearestNeighborConst(Size2D(blockWidth, blockHeight), srcBase, &map[0],
                                                    getRowPtr(dstBase, dstStride, i) + j, dstStride, borderValue);
            }
        }
    }
#else
    (void)ssize;
    (void)dsize;
    (void)srcBase;
    (void)srcStride;
    (void)m;
    (void)dstBase;
    (void)dstStride;
    (void)borderMode;
    (void)borderValue;
#endif
}

void warpAffineLinear(const Size2D &ssize, const Size2D &dsize,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      const f32 * m,
                      u8 * dstBase, ptrdiff_t dstStride,
                      BORDER_MODE borderMode, u8 borderValue)
{
    internal::assertSupportedConfiguration(isWarpAffineLinearSupported(ssize));
#ifdef CAROTENE_NEON
    using namespace internal;

    s32 _map[((BLOCK_SIZE * BLOCK_SIZE) << 2) + 16];
    f32 _coeffs[((BLOCK_SIZE * BLOCK_SIZE) << 1) + 16];
    s32 * map = alignPtr(_map, 16);
    f32 * coeffs = alignPtr(_coeffs, 16);

    int32x4_t v_width4 = vdupq_n_s32(ssize.width - 1), v_height4 = vdupq_n_s32(ssize.height - 1);
    int32x4_t v_step4 = vdupq_n_s32(srcStride), v_1 = vdupq_n_s32(1);
    float32x4_t v_zero4f = vdupq_n_f32(0.0f), v_one4f = vdupq_n_f32(1.0f);

    float32x4_t v_m0 = vdupq_n_f32(m[0]);
    float32x4_t v_m1 = vdupq_n_f32(m[1]);
    float32x4_t v_m2 = vdupq_n_f32(m[2]);
    float32x4_t v_m3 = vdupq_n_f32(m[3]);
    float32x4_t v_m4 = vdupq_n_f32(m[4]);
    float32x4_t v_m5 = vdupq_n_f32(m[5]);

    if (borderMode == BORDER_MODE_REPLICATE)
    {
        int32x4_t v_zero4 = vdupq_n_s32(0);

        for (size_t i = 0; i < dsize.height; i += BLOCK_SIZE)
        {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, dsize.height - i);
            for (size_t j = 0; j < dsize.width; j += BLOCK_SIZE)
            {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, dsize.width - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y)
                {
                    s32 * map_row = getRowPtr(map, blockWidth * sizeof(s32) * 4, y);
                    f32 * coeff_row = getRowPtr(coeffs, blockWidth * sizeof(f32) * 2, y);

                    size_t x = 0, y_ = y + i;
                    f32 indeces[4] = { j + 0.0f, j + 1.0f, j + 2.0f, j + 3.0f };
                    float32x4_t v_x = vld1q_f32(indeces), v_y = vdupq_n_f32(y_), v_4 = vdupq_n_f32(4.0f);
                    float32x4_t v_yx = vmlaq_f32(v_m4, v_m2, v_y), v_yy = vmlaq_f32(v_m5, v_m3, v_y);

                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4_t v_src_xf = vmlaq_f32(v_yx, v_m0, v_x);
                        float32x4_t v_src_yf = vmlaq_f32(v_yy, v_m1, v_x);

                        int32x4_t v_src_x = vcvtq_s32_f32(v_src_xf);
                        int32x4_t v_src_y = vcvtq_s32_f32(v_src_yf);

                        float32x4x2_t v_coeff;
                        v_coeff.val[0] = vsubq_f32(v_src_xf, vcvtq_f32_s32(v_src_x));
                        v_coeff.val[1] = vsubq_f32(v_src_yf, vcvtq_f32_s32(v_src_y));
                        uint32x4_t v_maskx = vcltq_f32(v_coeff.val[0], v_zero4f);
                        uint32x4_t v_masky = vcltq_f32(v_coeff.val[1], v_zero4f);
                        v_coeff.val[0] = vbslq_f32(v_maskx, vaddq_f32(v_one4f, v_coeff.val[0]), v_coeff.val[0]);
                        v_coeff.val[1] = vbslq_f32(v_masky, vaddq_f32(v_one4f, v_coeff.val[1]), v_coeff.val[1]);
                        v_src_x = vbslq_s32(v_maskx, vsubq_s32(v_src_x, v_1), v_src_x);
                        v_src_y = vbslq_s32(v_masky, vsubq_s32(v_src_y, v_1), v_src_y);

                        int32x4_t v_dst0_x = vmaxq_s32(v_zero4, vminq_s32(v_width4, v_src_x));
                        int32x4_t v_dst0_y = vmaxq_s32(v_zero4, vminq_s32(v_height4, v_src_y));
                        int32x4_t v_dst1_x = vmaxq_s32(v_zero4, vminq_s32(v_width4, vaddq_s32(v_1, v_src_x)));
                        int32x4_t v_dst1_y = vmaxq_s32(v_zero4, vminq_s32(v_height4, vaddq_s32(v_1, v_src_y)));

                        int32x4x4_t v_dst_index;
                        v_dst_index.val[0] = vmlaq_s32(v_dst0_x, v_dst0_y, v_step4);
                        v_dst_index.val[1] = vmlaq_s32(v_dst1_x, v_dst0_y, v_step4);
                        v_dst_index.val[2] = vmlaq_s32(v_dst0_x, v_dst1_y, v_step4);
                        v_dst_index.val[3] = vmlaq_s32(v_dst1_x, v_dst1_y, v_step4);

                        vst2q_f32(coeff_row + (x << 1), v_coeff);
                        vst4q_s32(map_row + (x << 2), v_dst_index);

                        v_x = vaddq_f32(v_x, v_4);
                    }

                    f32 yx = m[2] * y_ + m[4], yy = m[3] * y_ + m[5];
                    for (ptrdiff_t x_ = x + j; x < blockWidth; ++x, ++x_)
                    {
                        f32 src_x_f = m[0] * x_ + yx;
                        f32 src_y_f = m[1] * x_ + yy;

                        s32 src0_x = (s32)floorf(src_x_f);
                        s32 src0_y = (s32)floorf(src_y_f);

                        coeff_row[(x << 1) + 0] = src_x_f - src0_x;
                        coeff_row[(x << 1) + 1] = src_y_f - src0_y;

                        s32 src1_y = std::max(0, std::min<s32>(ssize.height - 1, src0_y + 1));
                        src0_y = std::max(0, std::min<s32>(ssize.height - 1, src0_y));
                        s32 src1_x = std::max(0, std::min<s32>(ssize.width - 1, src0_x + 1));
                        src0_x = std::max(0, std::min<s32>(ssize.width - 1, src0_x));

                        map_row[(x << 2) + 0] = src0_y * srcStride + src0_x;
                        map_row[(x << 2) + 1] = src0_y * srcStride + src1_x;
                        map_row[(x << 2) + 2] = src1_y * srcStride + src0_x;
                        map_row[(x << 2) + 3] = src1_y * srcStride + src1_x;
                    }
                }

                remapLinearReplicate(Size2D(blockWidth, blockHeight),
                                     srcBase, &map[0], &coeffs[0],
                                     getRowPtr(dstBase, dstStride, i) + j, dstStride);
            }
        }
    }
    else if (borderMode == BORDER_MODE_CONSTANT)
    {
        float32x4_t v_zero4 = vdupq_n_f32(0.0f);
        int32x4_t v_m1_4 = vdupq_n_s32(-1);

        for (size_t i = 0; i < dsize.height; i += BLOCK_SIZE)
        {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, dsize.height - i);
            for (size_t j = 0; j < dsize.width; j += BLOCK_SIZE)
            {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, dsize.width - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y)
                {
                    s32 * map_row = getRowPtr(map, blockWidth * sizeof(s32) * 4, y);
                    f32 * coeff_row = getRowPtr(coeffs, blockWidth * sizeof(f32) * 2, y);

                    size_t x = 0, y_ = y + i;
                    f32 indeces[4] = { j + 0.0f, j + 1.0f, j + 2.0f, j + 3.0f };
                    float32x4_t v_x = vld1q_f32(indeces), v_y = vdupq_n_f32(y_), v_4 = vdupq_n_f32(4.0f);
                    float32x4_t v_yx = vmlaq_f32(v_m4, v_m2, v_y), v_yy = vmlaq_f32(v_m5, v_m3, v_y);

                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4_t v_src_xf = vmlaq_f32(v_yx, v_m0, v_x);
                        float32x4_t v_src_yf = vmlaq_f32(v_yy, v_m1, v_x);

                        int32x4_t v_src_x0 = vcvtq_s32_f32(v_src_xf);
                        int32x4_t v_src_y0 = vcvtq_s32_f32(v_src_yf);

                        float32x4x2_t v_coeff;
                        v_coeff.val[0] = vsubq_f32(v_src_xf, vcvtq_f32_s32(v_src_x0));
                        v_coeff.val[1] = vsubq_f32(v_src_yf, vcvtq_f32_s32(v_src_y0));
                        uint32x4_t v_maskx = vcltq_f32(v_coeff.val[0], v_zero4f);
                        uint32x4_t v_masky = vcltq_f32(v_coeff.val[1], v_zero4f);
                        v_coeff.val[0] = vbslq_f32(v_maskx, vaddq_f32(v_one4f, v_coeff.val[0]), v_coeff.val[0]);
                        v_coeff.val[1] = vbslq_f32(v_masky, vaddq_f32(v_one4f, v_coeff.val[1]), v_coeff.val[1]);
                        v_src_x0 = vbslq_s32(v_maskx, vsubq_s32(v_src_x0, v_1), v_src_x0);
                        v_src_y0 = vbslq_s32(v_masky, vsubq_s32(v_src_y0, v_1), v_src_y0);

                        int32x4_t v_src_x1 = vaddq_s32(v_src_x0, v_1);
                        int32x4_t v_src_y1 = vaddq_s32(v_src_y0, v_1);

                        int32x4x4_t v_dst_index;
                        v_dst_index.val[0] = vmlaq_s32(v_src_x0, v_src_y0, v_step4);
                        v_dst_index.val[1] = vmlaq_s32(v_src_x1, v_src_y0, v_step4);
                        v_dst_index.val[2] = vmlaq_s32(v_src_x0, v_src_y1, v_step4);
                        v_dst_index.val[3] = vmlaq_s32(v_src_x1, v_src_y1, v_step4);

                        uint32x4_t v_mask_x0 = vandq_u32(vcgeq_f32(v_src_xf, v_zero4), vcleq_s32(v_src_x0, v_width4));
                        uint32x4_t v_mask_x1 = vandq_u32(vcgeq_f32(vaddq_f32(v_src_xf, v_one4f), v_zero4), vcleq_s32(v_src_x1, v_width4));
                        uint32x4_t v_mask_y0 = vandq_u32(vcgeq_f32(v_src_yf, v_zero4), vcleq_s32(v_src_y0, v_height4));
                        uint32x4_t v_mask_y1 = vandq_u32(vcgeq_f32(vaddq_f32(v_src_yf, v_one4f), v_zero4), vcleq_s32(v_src_y1, v_height4));

                        v_dst_index.val[0] = vbslq_s32(vandq_u32(v_mask_x0, v_mask_y0), v_dst_index.val[0], v_m1_4);
                        v_dst_index.val[1] = vbslq_s32(vandq_u32(v_mask_x1, v_mask_y0), v_dst_index.val[1], v_m1_4);
                        v_dst_index.val[2] = vbslq_s32(vandq_u32(v_mask_x0, v_mask_y1), v_dst_index.val[2], v_m1_4);
                        v_dst_index.val[3] = vbslq_s32(vandq_u32(v_mask_x1, v_mask_y1), v_dst_index.val[3], v_m1_4);

                        vst2q_f32(coeff_row + (x << 1), v_coeff);
                        vst4q_s32(map_row + (x << 2), v_dst_index);

                        v_x = vaddq_f32(v_x, v_4);
                    }

                    f32 yx = m[2] * y_ + m[4], yy = m[3] * y_ + m[5];
                    for (ptrdiff_t x_ = x + j; x < blockWidth; ++x, ++x_)
                    {
                        f32 src_x_f = m[0] * x_ + yx;
                        f32 src_y_f = m[1] * x_ + yy;

                        s32 src0_x = (s32)floorf(src_x_f), src1_x = src0_x + 1;
                        s32 src0_y = (s32)floorf(src_y_f), src1_y = src0_y + 1;

                        coeff_row[(x << 1) + 0] = src_x_f - src0_x;
                        coeff_row[(x << 1) + 1] = src_y_f - src0_y;

                        map_row[(x << 2) + 0] = (src0_x >= 0) && (src0_x < (s32)ssize.width) &&
                                                (src0_y >= 0) && (src0_y < (s32)ssize.height) ? src0_y * srcStride + src0_x : -1;
                        map_row[(x << 2) + 1] = (src1_x >= 0) && (src1_x < (s32)ssize.width) &&
                                                (src0_y >= 0) && (src0_y < (s32)ssize.height) ? src0_y * srcStride + src1_x : -1;
                        map_row[(x << 2) + 2] = (src0_x >= 0) && (src0_x < (s32)ssize.width) &&
                                                (src1_y >= 0) && (src1_y < (s32)ssize.height) ? src1_y * srcStride + src0_x : -1;
                        map_row[(x << 2) + 3] = (src1_x >= 0) && (src1_x < (s32)ssize.width) &&
                                                (src1_y >= 0) && (src1_y < (s32)ssize.height) ? src1_y * srcStride + src1_x : -1;
                    }
                }

                remapLinearConst(Size2D(blockWidth, blockHeight),
                                 srcBase, &map[0], &coeffs[0],
                                 getRowPtr(dstBase, dstStride, i) + j, dstStride, borderValue);
            }
        }
    }
#else
    (void)ssize;
    (void)dsize;
    (void)srcBase;
    (void)srcStride;
    (void)m;
    (void)dstBase;
    (void)dstStride;
    (void)borderMode;
    (void)borderValue;
#endif
}

} // namespace CAROTENE_NS
