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
 * Copyright (C) 2014, NVIDIA Corporation, all rights reserved.
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

#include <cmath>

namespace CAROTENE_NS {

void meanStdDev(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                f32 * pMean, f32 * pStdDev)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    f64 fsum = 0.0f, fsqsum = 0.0f;
    sqsum(size, srcBase, srcStride, &fsum, &fsqsum, 1);

    // calc mean and stddev
    f64 itotal = 1.0 / size.total();
    f64 mean = fsum * itotal;
    f64 stddev = sqrt(std::max(fsqsum * itotal - mean * mean, 0.0));

    if (pMean)
        *pMean = mean;
    if (pStdDev)
        *pStdDev = stddev;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMean;
    (void)pStdDev;
#endif
}

void meanStdDev(const Size2D &size,
                const u16 * srcBase, ptrdiff_t srcStride,
                f32 * pMean, f32 * pStdDev)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t blockSize0 = 1 << 10, roiw4 = size.width & ~3;
    f64 fsum = 0.0f, fsqsum = 0.0f;

    f32 arsum[8];
    uint32x4_t v_zero = vdupq_n_u32(0u), v_sum;
    float32x4_t v_zero_f = vdupq_n_f32(0.0f), v_sqsum;

    for (size_t i = 0; i < size.height; ++i)
    {
        const u16 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0u;

        while (j < roiw4)
        {
            size_t blockSize = std::min(roiw4 - j, blockSize0) + j;
            v_sum = v_zero;
            v_sqsum = v_zero_f;

            for ( ; j + 16 < blockSize ; j += 16)
            {
                internal::prefetch(src + j);
                uint16x8_t v_src0 = vld1q_u16(src + j), v_src1 = vld1q_u16(src + j + 8);

                // 0
                uint32x4_t v_srclo = vmovl_u16(vget_low_u16(v_src0));
                uint32x4_t v_srchi = vmovl_u16(vget_high_u16(v_src0));
                v_sum = vaddq_u32(v_sum, vaddq_u32(v_srclo, v_srchi));
                float32x4_t v_srclo_f = vcvtq_f32_u32(v_srclo);
                float32x4_t v_srchi_f = vcvtq_f32_u32(v_srchi);
                v_sqsum = vmlaq_f32(v_sqsum, v_srclo_f, v_srclo_f);
                v_sqsum = vmlaq_f32(v_sqsum, v_srchi_f, v_srchi_f);

                // 1
                v_srclo = vmovl_u16(vget_low_u16(v_src1));
                v_srchi = vmovl_u16(vget_high_u16(v_src1));
                v_sum = vaddq_u32(v_sum, vaddq_u32(v_srclo, v_srchi));
                v_srclo_f = vcvtq_f32_u32(v_srclo);
                v_srchi_f = vcvtq_f32_u32(v_srchi);
                v_sqsum = vmlaq_f32(v_sqsum, v_srclo_f, v_srclo_f);
                v_sqsum = vmlaq_f32(v_sqsum, v_srchi_f, v_srchi_f);
            }

            for ( ; j < blockSize; j += 4)
            {
                uint32x4_t v_src = vmovl_u16(vld1_u16(src + j));
                float32x4_t v_src_f = vcvtq_f32_u32(v_src);
                v_sum = vaddq_u32(v_sum, v_src);
                v_sqsum = vmlaq_f32(v_sqsum, v_src_f, v_src_f);
            }

            vst1q_f32(arsum, vcvtq_f32_u32(v_sum));
            vst1q_f32(arsum + 4, v_sqsum);

            fsum += (f64)arsum[0] + arsum[1] + arsum[2] + arsum[3];
            fsqsum += (f64)arsum[4] + arsum[5] + arsum[6] + arsum[7];
        }

        // collect a few last elements in the current row
        for ( ; j < size.width; ++j)
        {
            f32 srcval = src[j];
            fsum += srcval;
            fsqsum += srcval * srcval;
        }
    }

    // calc mean and stddev
    f64 itotal = 1.0 / size.total();
    f64 mean = fsum * itotal;
    f64 stddev = sqrt(std::max(fsqsum * itotal - mean * mean, 0.0));

    if (pMean)
        *pMean = mean;
    if (pStdDev)
        *pStdDev = stddev;
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)pMean;
    (void)pStdDev;
#endif
}

} // namespace CAROTENE_NS
