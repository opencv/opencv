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

#ifdef CAROTENE_NEON

namespace internal {

void remapNearestNeighborReplicate(const Size2D size,
                                   const u8 * srcBase,
                                   const s32 * map,
                                   u8 * dstBase, ptrdiff_t dstStride)
{
    for (size_t y = 0; y < size.height; ++y)
    {
        const s32 * map_row = internal::getRowPtr(map, size.width * sizeof(s32), y);
        u8 * dst_row = internal::getRowPtr(dstBase, dstStride, y);

        for (size_t x = 0; x < size.width; ++x)
        {
            dst_row[x] = srcBase[map_row[x]];
        }
    }
}

void remapNearestNeighborConst(const Size2D size,
                               const u8 * srcBase,
                               const s32 * map,
                               u8 * dstBase, ptrdiff_t dstStride,
                               u8 borderValue)
{
    for (size_t y = 0; y < size.height; ++y)
    {
        const s32 * map_row = internal::getRowPtr(map, size.width * sizeof(s32), y);
        u8 * dst_row = internal::getRowPtr(dstBase, dstStride, y);

        for (size_t x = 0; x < size.width; ++x)
        {
            s32 src_idx = map_row[x];
            dst_row[x] = src_idx >= 0 ? srcBase[map_row[x]] : borderValue;
        }
    }
}

void remapLinearReplicate(const Size2D size,
                          const u8 * srcBase,
                          const s32 * map,
                          const f32 * coeffs,
                          u8 * dstBase, ptrdiff_t dstStride)
{
    int16x8_t v_zero16 = vdupq_n_s16(0);

    for (size_t y = 0; y < size.height; ++y)
    {
        const s32 * map_row = internal::getRowPtr(map, size.width * sizeof(s32) * 4, y);
        const f32 * coeff_row = internal::getRowPtr(coeffs, size.width * sizeof(f32) * 2, y);

        u8 * dst_row = internal::getRowPtr(dstBase, dstStride, y);

        size_t x = 0;
        for ( ; x + 8 < size.width; x += 8)
        {
            int16x8_t v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2)]], v_zero16, 0);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 4]], v_src00, 1);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 8]], v_src00, 2);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 12]], v_src00, 3);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 16]], v_src00, 4);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 20]], v_src00, 5);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 24]], v_src00, 6);
            v_src00 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 28]], v_src00, 7);

            int16x8_t v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 1]], v_zero16, 0);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 5]], v_src01, 1);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 9]], v_src01, 2);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 13]], v_src01, 3);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 17]], v_src01, 4);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 21]], v_src01, 5);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 25]], v_src01, 6);
            v_src01 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 29]], v_src01, 7);

            int16x8_t v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 2]], v_zero16, 0);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 6]], v_src10, 1);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 10]], v_src10, 2);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 14]], v_src10, 3);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 18]], v_src10, 4);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 22]], v_src10, 5);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 26]], v_src10, 6);
            v_src10 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 30]], v_src10, 7);

            int16x8_t v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 3]], v_zero16, 0);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 7]], v_src11, 1);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 11]], v_src11, 2);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 15]], v_src11, 3);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 19]], v_src11, 4);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 23]], v_src11, 5);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 27]], v_src11, 6);
            v_src11 = vsetq_lane_s16(srcBase[map_row[(x << 2) + 31]], v_src11, 7);

            // first part
            float32x4_t v_src00_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src00)));
            float32x4_t v_src10_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src10)));

            float32x4x2_t v_coeff = vld2q_f32(coeff_row + (x << 1));
            float32x4_t v_dst_0 = vmlaq_f32(v_src00_f, vcvtq_f32_s32(vsubl_s16(vget_low_s16(v_src01),
                                                                               vget_low_s16(v_src00))), v_coeff.val[0]);
            float32x4_t v_dst_1 = vmlaq_f32(v_src10_f, vcvtq_f32_s32(vsubl_s16(vget_low_s16(v_src11),
                                                                               vget_low_s16(v_src10))), v_coeff.val[0]);

            float32x4_t v_dst = vmlaq_f32(v_dst_0, vsubq_f32(v_dst_1, v_dst_0), v_coeff.val[1]);
            uint16x4_t v_dst0 = vmovn_u32(vcvtq_u32_f32(v_dst));

            // second part
            v_src00_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src00)));
            v_src10_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src10)));

            v_coeff = vld2q_f32(coeff_row + (x << 1) + 8);
            v_dst_0 = vmlaq_f32(v_src00_f, vcvtq_f32_s32(vsubl_s16(vget_high_s16(v_src01),
                                                                   vget_high_s16(v_src00))), v_coeff.val[0]);
            v_dst_1 = vmlaq_f32(v_src10_f, vcvtq_f32_s32(vsubl_s16(vget_high_s16(v_src11),
                                                                   vget_high_s16(v_src10))), v_coeff.val[0]);

            v_dst = vmlaq_f32(v_dst_0, vsubq_f32(v_dst_1, v_dst_0), v_coeff.val[1]);
            uint16x4_t v_dst1 = vmovn_u32(vcvtq_u32_f32(v_dst));

            // store
            vst1_u8(dst_row + x, vmovn_u16(vcombine_u16(v_dst0, v_dst1)));
        }

        for ( ; x < size.width; ++x)
        {
            s32 src00_index = map_row[(x << 2)];
            s32 src10_index = map_row[(x << 2) + 2];
            f32 dst_val_0 = (srcBase[map_row[(x << 2) + 1]] - srcBase[src00_index]) * coeff_row[x << 1] +
                             srcBase[src00_index];
            f32 dst_val_1 = (srcBase[map_row[(x << 2) + 3]] - srcBase[src10_index]) * coeff_row[x << 1] +
                             srcBase[src10_index];
            dst_row[x] = floorf((dst_val_1 - dst_val_0) * coeff_row[(x << 1) + 1] + dst_val_0);
        }
    }
}

void remapLinearConst(const Size2D size,
                      const u8 * srcBase,
                      const s32 * map,
                      const f32 * coeffs,
                      u8 * dstBase, ptrdiff_t dstStride,
                      u8 borderValue)
{
    int16x8_t v_zero16 = vdupq_n_s16(0);

    for (size_t y = 0; y < size.height; ++y)
    {
        const s32 * map_row = internal::getRowPtr(map, size.width * sizeof(s32) * 4, y);
        const f32 * coeff_row = internal::getRowPtr(coeffs, size.width * sizeof(f32) * 2, y);

        u8 * dst_row = internal::getRowPtr(dstBase, dstStride, y);

        size_t x = 0;
        for ( ; x + 8 < size.width; x += 8)
        {
            int16x8_t v_src00 = vsetq_lane_s16(map_row[(x << 2)] >= 0 ? srcBase[map_row[(x << 2)]] : borderValue, v_zero16, 0);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) +  4] >= 0 ? srcBase[map_row[(x << 2) +  4]] : borderValue, v_src00, 1);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) +  8] >= 0 ? srcBase[map_row[(x << 2) +  8]] : borderValue, v_src00, 2);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) + 12] >= 0 ? srcBase[map_row[(x << 2) + 12]] : borderValue, v_src00, 3);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) + 16] >= 0 ? srcBase[map_row[(x << 2) + 16]] : borderValue, v_src00, 4);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) + 20] >= 0 ? srcBase[map_row[(x << 2) + 20]] : borderValue, v_src00, 5);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) + 24] >= 0 ? srcBase[map_row[(x << 2) + 24]] : borderValue, v_src00, 6);
            v_src00 = vsetq_lane_s16(map_row[(x << 2) + 28] >= 0 ? srcBase[map_row[(x << 2) + 28]] : borderValue, v_src00, 7);

            int16x8_t v_src01 = vsetq_lane_s16(map_row[(x << 2) + 1] >= 0 ? srcBase[map_row[(x << 2) + 1]] : borderValue, v_zero16, 0);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) +  5] >= 0 ? srcBase[map_row[(x << 2) +  5]] : borderValue, v_src01, 1);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) +  9] >= 0 ? srcBase[map_row[(x << 2) +  9]] : borderValue, v_src01, 2);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) + 13] >= 0 ? srcBase[map_row[(x << 2) + 13]] : borderValue, v_src01, 3);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) + 17] >= 0 ? srcBase[map_row[(x << 2) + 17]] : borderValue, v_src01, 4);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) + 21] >= 0 ? srcBase[map_row[(x << 2) + 21]] : borderValue, v_src01, 5);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) + 25] >= 0 ? srcBase[map_row[(x << 2) + 25]] : borderValue, v_src01, 6);
            v_src01 = vsetq_lane_s16(map_row[(x << 2) + 29] >= 0 ? srcBase[map_row[(x << 2) + 29]] : borderValue, v_src01, 7);

            int16x8_t v_src10 = vsetq_lane_s16(map_row[(x << 2) + 2] >= 0 ? srcBase[map_row[(x << 2) + 2]] : borderValue, v_zero16, 0);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) +  6] >= 0 ? srcBase[map_row[(x << 2) +  6]] : borderValue, v_src10, 1);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) + 10] >= 0 ? srcBase[map_row[(x << 2) + 10]] : borderValue, v_src10, 2);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) + 14] >= 0 ? srcBase[map_row[(x << 2) + 14]] : borderValue, v_src10, 3);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) + 18] >= 0 ? srcBase[map_row[(x << 2) + 18]] : borderValue, v_src10, 4);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) + 22] >= 0 ? srcBase[map_row[(x << 2) + 22]] : borderValue, v_src10, 5);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) + 26] >= 0 ? srcBase[map_row[(x << 2) + 26]] : borderValue, v_src10, 6);
            v_src10 = vsetq_lane_s16(map_row[(x << 2) + 30] >= 0 ? srcBase[map_row[(x << 2) + 30]] : borderValue, v_src10, 7);

            int16x8_t v_src11 = vsetq_lane_s16(map_row[(x << 2) + 3] >= 0 ? srcBase[map_row[(x << 2) + 3]] : borderValue, v_zero16, 0);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) +  7] >= 0 ? srcBase[map_row[(x << 2) +  7]] : borderValue, v_src11, 1);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) + 11] >= 0 ? srcBase[map_row[(x << 2) + 11]] : borderValue, v_src11, 2);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) + 15] >= 0 ? srcBase[map_row[(x << 2) + 15]] : borderValue, v_src11, 3);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) + 19] >= 0 ? srcBase[map_row[(x << 2) + 19]] : borderValue, v_src11, 4);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) + 23] >= 0 ? srcBase[map_row[(x << 2) + 23]] : borderValue, v_src11, 5);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) + 27] >= 0 ? srcBase[map_row[(x << 2) + 27]] : borderValue, v_src11, 6);
            v_src11 = vsetq_lane_s16(map_row[(x << 2) + 31] >= 0 ? srcBase[map_row[(x << 2) + 31]] : borderValue, v_src11, 7);

            // first part
            float32x4_t v_src00_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src00)));
            float32x4_t v_src10_f = vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src10)));

            float32x4x2_t v_coeff = vld2q_f32(coeff_row + (x << 1));
            float32x4_t v_dst_0 = vmlaq_f32(v_src00_f, vcvtq_f32_s32(vsubl_s16(vget_low_s16(v_src01),
                                                                               vget_low_s16(v_src00))), v_coeff.val[0]);
            float32x4_t v_dst_1 = vmlaq_f32(v_src10_f, vcvtq_f32_s32(vsubl_s16(vget_low_s16(v_src11),
                                                                               vget_low_s16(v_src10))), v_coeff.val[0]);

            float32x4_t v_dst = vmlaq_f32(v_dst_0, vsubq_f32(v_dst_1, v_dst_0), v_coeff.val[1]);
            uint16x4_t v_dst0 = vmovn_u32(vcvtq_u32_f32(v_dst));

            // second part
            v_src00_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src00)));
            v_src10_f = vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src10)));

            v_coeff = vld2q_f32(coeff_row + (x << 1) + 8);
            v_dst_0 = vmlaq_f32(v_src00_f, vcvtq_f32_s32(vsubl_s16(vget_high_s16(v_src01),
                                                                   vget_high_s16(v_src00))), v_coeff.val[0]);
            v_dst_1 = vmlaq_f32(v_src10_f, vcvtq_f32_s32(vsubl_s16(vget_high_s16(v_src11),
                                                                   vget_high_s16(v_src10))), v_coeff.val[0]);

            v_dst = vmlaq_f32(v_dst_0, vsubq_f32(v_dst_1, v_dst_0), v_coeff.val[1]);
            uint16x4_t v_dst1 = vmovn_u32(vcvtq_u32_f32(v_dst));

            // store
            vst1_u8(dst_row + x, vmovn_u16(vcombine_u16(v_dst0, v_dst1)));
        }

        for ( ; x < size.width; ++x)
        {
            s16 src00 = map_row[(x << 2) + 0] >= 0 ? srcBase[map_row[(x << 2) + 0]] : borderValue;
            s16 src01 = map_row[(x << 2) + 1] >= 0 ? srcBase[map_row[(x << 2) + 1]] : borderValue;
            s16 src10 = map_row[(x << 2) + 2] >= 0 ? srcBase[map_row[(x << 2) + 2]] : borderValue;
            s16 src11 = map_row[(x << 2) + 3] >= 0 ? srcBase[map_row[(x << 2) + 3]] : borderValue;

            f32 dst_val_0 = (src01 - src00) * coeff_row[(x << 1)] + src00;
            f32 dst_val_1 = (src11 - src10) * coeff_row[(x << 1)] + src10;
            dst_row[x] = floorf((dst_val_1 - dst_val_0) * coeff_row[(x << 1) + 1] + dst_val_0);
        }
    }
}

} // namespace internal

#endif // CAROTENE_NEON

bool isRemapNearestNeighborSupported(const Size2D &ssize)
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

bool isRemapLinearSupported(const Size2D &ssize)
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

void remapNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                          const u8 * srcBase, ptrdiff_t srcStride,
                          const f32 * tableBase, ptrdiff_t tableStride,
                          u8 * dstBase, ptrdiff_t dstStride,
                          BORDER_MODE borderMode, u8 borderValue)
{
    internal::assertSupportedConfiguration(isRemapNearestNeighborSupported(ssize));
#ifdef CAROTENE_NEON
    using namespace internal;

    s32 _map[BLOCK_SIZE * BLOCK_SIZE + 16];
    s32 * map = alignPtr(_map, 16);

    int32x4_t v_width4 = vdupq_n_s32(ssize.width - 1), v_height4 = vdupq_n_s32(ssize.height - 1);
    int32x2_t v_width2 = vdup_n_s32(ssize.width - 1), v_height2 = vdup_n_s32(ssize.height - 1);
    int32x4_t v_step4 = vdupq_n_s32(srcStride);
    int32x2_t v_step2 = vdup_n_s32(srcStride);

    if (borderMode == BORDER_MODE_REPLICATE)
    {
        int32x4_t v_zero4 = vdupq_n_s32(0);
        int32x2_t v_zero2 = vdup_n_s32(0);

        for (size_t i = 0; i < dsize.height; i += BLOCK_SIZE)
        {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, dsize.height - i);
            for (size_t j = 0; j < dsize.width; j += BLOCK_SIZE)
            {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, dsize.width - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y)
                {
                    const f32 * table_row = getRowPtr(tableBase, tableStride, i + y) + (j << 1);
                    s32 * map_row = getRowPtr(&map[0], blockWidth * sizeof(s32), y);

                    size_t x = 0;
                    for ( ; x + 8 <= blockWidth; x += 8)
                    {
                        float32x4x2_t v_table0 = vld2q_f32(table_row + (x << 1)),
                                      v_table1 = vld2q_f32(table_row + (x << 1) + 8);

                        int32x4_t v_dst_x = vmaxq_s32(v_zero4, vminq_s32(v_width4, vcvtq_s32_f32(v_table0.val[0])));
                        int32x4_t v_dst_y = vmaxq_s32(v_zero4, vminq_s32(v_height4, vcvtq_s32_f32(v_table0.val[1])));
                        int32x4_t v_dst_index = vmlaq_s32(v_dst_x, v_dst_y, v_step4);
                        vst1q_s32(map_row + x, v_dst_index);

                        v_dst_x = vmaxq_s32(v_zero4, vminq_s32(v_width4, vcvtq_s32_f32(v_table1.val[0])));
                        v_dst_y = vmaxq_s32(v_zero4, vminq_s32(v_height4, vcvtq_s32_f32(v_table1.val[1])));
                        v_dst_index = vmlaq_s32(v_dst_x, v_dst_y, v_step4);
                        vst1q_s32(map_row + x + 4, v_dst_index);
                    }

                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4x2_t v_table0 = vld2q_f32(table_row + (x << 1));

                        int32x4_t v_dst_x = vmaxq_s32(v_zero4, vminq_s32(v_width4, vcvtq_s32_f32(v_table0.val[0])));
                        int32x4_t v_dst_y = vmaxq_s32(v_zero4, vminq_s32(v_height4, vcvtq_s32_f32(v_table0.val[1])));
                        int32x4_t v_dst_index = vmlaq_s32(v_dst_x, v_dst_y, v_step4);
                        vst1q_s32(map_row + x, v_dst_index);
                    }

                    for ( ; x + 2 <= blockWidth; x += 2)
                    {
                        float32x2x2_t v_table0 = vld2_f32(table_row + (x << 1));

                        int32x2_t v_dst_x = vmax_s32(v_zero2, vmin_s32(v_width2, vcvt_s32_f32(v_table0.val[0])));
                        int32x2_t v_dst_y = vmax_s32(v_zero2, vmin_s32(v_height2, vcvt_s32_f32(v_table0.val[1])));
                        int32x2_t v_dst_index = vmla_s32(v_dst_x, v_dst_y, v_step2);
                        vst1_s32(map_row + x, v_dst_index);
                    }

                    for ( ; x < blockWidth; ++x)
                    {
                        s32 src_x = std::max(0, std::min<s32>(ssize.width - 1, (s32)floorf(table_row[(x << 1) + 0])));
                        s32 src_y = std::max(0, std::min<s32>(ssize.height - 1, (s32)floorf(table_row[(x << 1) + 1])));
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
        int32x2_t v_m1_2 = vdup_n_s32(-1);
        float32x4_t v_zero4 = vdupq_n_f32(0.0f);
        float32x2_t v_zero2 = vdup_n_f32(0.0f);

        for (size_t i = 0; i < dsize.height; i += BLOCK_SIZE)
        {
            size_t blockHeight = std::min<size_t>(BLOCK_SIZE, dsize.height - i);
            for (size_t j = 0; j < dsize.width; j += BLOCK_SIZE)
            {
                size_t blockWidth = std::min<size_t>(BLOCK_SIZE, dsize.width - j);

                // compute table
                for (size_t y = 0; y < blockHeight; ++y)
                {
                    const f32 * table_row = getRowPtr(tableBase, tableStride, i + y) + (j << 1);
                    s32 * map_row = getRowPtr(&map[0], blockWidth * sizeof(s32), y);

                    size_t x = 0;
                    for ( ; x + 8 <= blockWidth; x += 8)
                    {
                        float32x4x2_t v_table0 = vld2q_f32(table_row + (x << 1)),
                                      v_table1 = vld2q_f32(table_row + (x << 1) + 8);

                        int32x4_t v_dst_x = vcvtq_s32_f32(v_table0.val[0]);
                        int32x4_t v_dst_y = vcvtq_s32_f32(v_table0.val[1]);
                        uint32x4_t v_mask = vandq_u32(vandq_u32(vcgeq_f32(v_table0.val[0], v_zero4), vcleq_s32(v_dst_x, v_width4)),
                                                      vandq_u32(vcgeq_f32(v_table0.val[1], v_zero4), vcleq_s32(v_dst_y, v_height4)));
                        int32x4_t v_dst_index = vbslq_s32(v_mask, vmlaq_s32(v_dst_x, v_dst_y, v_step4), v_m1_4);
                        vst1q_s32(map_row + x, v_dst_index);

                        v_dst_x = vcvtq_s32_f32(v_table1.val[0]);
                        v_dst_y = vcvtq_s32_f32(v_table1.val[1]);
                        v_mask = vandq_u32(vandq_u32(vcgeq_f32(v_table1.val[0], v_zero4), vcleq_s32(v_dst_x, v_width4)),
                                           vandq_u32(vcgeq_f32(v_table1.val[1], v_zero4), vcleq_s32(v_dst_y, v_height4)));
                        v_dst_index = vbslq_s32(v_mask, vmlaq_s32(v_dst_x, v_dst_y, v_step4), v_m1_4);
                        vst1q_s32(map_row + x + 4, v_dst_index);
                    }

                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4x2_t v_table0 = vld2q_f32(table_row + (x << 1));

                        int32x4_t v_dst_x = vcvtq_s32_f32(v_table0.val[0]);
                        int32x4_t v_dst_y = vcvtq_s32_f32(v_table0.val[1]);
                        uint32x4_t v_mask = vandq_u32(vandq_u32(vcgeq_f32(v_table0.val[0], v_zero4), vcleq_s32(v_dst_x, v_width4)),
                                                      vandq_u32(vcgeq_f32(v_table0.val[1], v_zero4), vcleq_s32(v_dst_y, v_height4)));
                        int32x4_t v_dst_index = vbslq_s32(v_mask, vmlaq_s32(v_dst_x, v_dst_y, v_step4), v_m1_4);
                        vst1q_s32(map_row + x, v_dst_index);
                    }

                    for ( ; x + 2 <= blockWidth; x += 2)
                    {
                        float32x2x2_t v_table0 = vld2_f32(table_row + (x << 1));

                        int32x2_t v_dst_x = vcvt_s32_f32(v_table0.val[0]);
                        int32x2_t v_dst_y = vcvt_s32_f32(v_table0.val[1]);
                        uint32x2_t v_mask = vand_u32(vand_u32(vcge_f32(v_table0.val[0], v_zero2), vcle_s32(v_dst_x, v_width2)),
                                                     vand_u32(vcge_f32(v_table0.val[1], v_zero2), vcle_s32(v_dst_y, v_height2)));
                        int32x2_t v_dst_index = vbsl_s32(v_mask, vmla_s32(v_dst_x, v_dst_y, v_step2), v_m1_2);
                        vst1_s32(map_row + x, v_dst_index);
                    }

                    for ( ; x < blockWidth; ++x)
                    {
                        s32 src_x = (s32)floorf(table_row[(x << 1) + 0]);
                        s32 src_y = (s32)floorf(table_row[(x << 1) + 1]);
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
    (void)tableBase;
    (void)tableStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderMode;
    (void)borderValue;
#endif
}

void remapLinear(const Size2D &ssize, const Size2D &dsize,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 const f32 * tableBase, ptrdiff_t tableStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE borderMode, u8 borderValue)
{
    internal::assertSupportedConfiguration(isRemapLinearSupported(ssize));
#ifdef CAROTENE_NEON
    using namespace internal;

    s32 _map[((BLOCK_SIZE * BLOCK_SIZE) << 2) + 16];
    f32 _coeffs[((BLOCK_SIZE * BLOCK_SIZE) << 1) + 16];

    s32 * map = alignPtr(_map, 16);
    f32 * coeffs = alignPtr(_coeffs, 16);

    int32x4_t v_width4 = vdupq_n_s32(ssize.width - 1), v_height4 = vdupq_n_s32(ssize.height - 1);
    int32x4_t v_step4 = vdupq_n_s32(srcStride), v_1 = vdupq_n_s32(1);
    float32x4_t v_zero4f = vdupq_n_f32(0.0f), v_one4f = vdupq_n_f32(1.0f);

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
                    const f32 * table_row = getRowPtr(tableBase, tableStride, i + y) + (j << 1);

                    s32 * map_row = getRowPtr(map, blockWidth * sizeof(s32) * 4, y);
                    f32 * coeff_row = getRowPtr(coeffs, blockWidth * sizeof(f32) * 2, y);

                    size_t x = 0;
                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4x2_t v_table = vld2q_f32(table_row + (x << 1));

                        int32x4_t v_src_x = vcvtq_s32_f32(v_table.val[0]);
                        int32x4_t v_src_y = vcvtq_s32_f32(v_table.val[1]);

                        float32x4x2_t  v_coeff;
                        v_coeff.val[0] = vsubq_f32(v_table.val[0], vcvtq_f32_s32(v_src_x));
                        v_coeff.val[1] = vsubq_f32(v_table.val[1], vcvtq_f32_s32(v_src_y));
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
                    }

                    for ( ; x < blockWidth; ++x)
                    {
                        f32 src_x_f = table_row[(x << 1) + 0];
                        f32 src_y_f = table_row[(x << 1) + 1];

                        s32 src0_x = (s32)floorf(src_x_f);
                        s32 src0_y = (s32)floorf(src_y_f);

                        coeff_row[x << 1] = src_x_f - src0_x;
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
                    const f32 * table_row = getRowPtr(tableBase, tableStride, i + y) + (j << 1);

                    s32 * map_row = getRowPtr(map, blockWidth * sizeof(s32) * 4, y);
                    f32 * coeff_row = getRowPtr(coeffs, blockWidth * sizeof(f32) * 2, y);

                    size_t x = 0;
                    for ( ; x + 4 <= blockWidth; x += 4)
                    {
                        float32x4x2_t v_table = vld2q_f32(table_row + (x << 1));

                        int32x4_t v_src_x0 = vcvtq_s32_f32(v_table.val[0]);
                        int32x4_t v_src_y0 = vcvtq_s32_f32(v_table.val[1]);

                        float32x4x2_t v_coeff;
                        v_coeff.val[0] = vsubq_f32(v_table.val[0], vcvtq_f32_s32(v_src_x0));
                        v_coeff.val[1] = vsubq_f32(v_table.val[1], vcvtq_f32_s32(v_src_y0));
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

                        uint32x4_t v_mask_x0 = vandq_u32(vcgeq_f32(v_table.val[0], v_zero4), vcleq_s32(v_src_x0, v_width4));
                        uint32x4_t v_mask_x1 = vandq_u32(vcgeq_f32(vaddq_f32(v_table.val[0], v_one4f), v_zero4), vcleq_s32(v_src_x1, v_width4));
                        uint32x4_t v_mask_y0 = vandq_u32(vcgeq_f32(v_table.val[1], v_zero4), vcleq_s32(v_src_y0, v_height4));
                        uint32x4_t v_mask_y1 = vandq_u32(vcgeq_f32(vaddq_f32(v_table.val[1], v_one4f), v_zero4), vcleq_s32(v_src_y1, v_height4));

                        v_dst_index.val[0] = vbslq_s32(vandq_u32(v_mask_x0, v_mask_y0), v_dst_index.val[0], v_m1_4);
                        v_dst_index.val[1] = vbslq_s32(vandq_u32(v_mask_x1, v_mask_y0), v_dst_index.val[1], v_m1_4);
                        v_dst_index.val[2] = vbslq_s32(vandq_u32(v_mask_x0, v_mask_y1), v_dst_index.val[2], v_m1_4);
                        v_dst_index.val[3] = vbslq_s32(vandq_u32(v_mask_x1, v_mask_y1), v_dst_index.val[3], v_m1_4);

                        vst2q_f32(coeff_row + (x << 1), v_coeff);
                        vst4q_s32(map_row + (x << 2), v_dst_index);
                    }

                    for ( ; x < blockWidth; ++x)
                    {
                        f32 src_x_f = table_row[(x << 1) + 0];
                        f32 src_y_f = table_row[(x << 1) + 1];

                        s32 src0_x = (s32)floorf(src_x_f), src1_x = src0_x + 1;
                        s32 src0_y = (s32)floorf(src_y_f), src1_y = src0_y + 1;

                        coeff_row[(x << 1)] = src_x_f - src0_x;
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
    (void)tableBase;
    (void)tableStride;
    (void)dstBase;
    (void)dstStride;
    (void)borderMode;
    (void)borderValue;
#endif
}

} // namespace CAROTENE_NS
