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

#include "vtransform.hpp"

namespace CAROTENE_NS {

bool isSumSupported(u32 channels)
{
    return (channels && channels < 5);
}

void sum(const Size2D &_size,
         const u8 * srcBase, ptrdiff_t srcStride,
         u32 * sumdst, u32 channels)
{
    internal::assertSupportedConfiguration(isSumSupported(channels));
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    const ptrdiff_t width = size.width * channels;

    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src = internal::getRowPtr( srcBase,  srcStride, k);
        ptrdiff_t i = 0;

        if (channels == 3)
        {
            uint32x4_t vs1231 = vdupq_n_u32(0);
            uint32x4_t vs3123 = vdupq_n_u32(0);
            uint32x4_t vs2312 = vdupq_n_u32(0);
            for (; i <= width - 257*8*3; i += 257*8*3, src += 257*8*3)
            {
                uint16x8_t s1 = vmovl_u8(vld1_u8(src +  0));
                uint16x8_t s2 = vmovl_u8(vld1_u8(src +  8));
                uint16x8_t s3 = vmovl_u8(vld1_u8(src + 16));

                for (ptrdiff_t j = 8*3; j < 257*8*3; j+= 8*3)
                {
                    internal::prefetch(src + j + 24);
                    s1 = vaddw_u8(s1, vld1_u8(src + j +  0));
                    s2 = vaddw_u8(s2, vld1_u8(src + j +  8));
                    s3 = vaddw_u8(s3, vld1_u8(src + j + 16));
                }

                vs1231 = vqaddq_u32(vs1231, vaddl_u16(vget_low_u16(s1), vget_high_u16(s2)));
                vs3123 = vqaddq_u32(vs3123, vaddl_u16(vget_low_u16(s2), vget_high_u16(s3)));
                vs2312 = vqaddq_u32(vs2312, vaddl_u16(vget_low_u16(s3), vget_high_u16(s1)));
            }
            if (i <= width - 8*3)
            {
                uint16x8_t s1 = vmovl_u8(vld1_u8(src +  0));
                uint16x8_t s2 = vmovl_u8(vld1_u8(src +  8));
                uint16x8_t s3 = vmovl_u8(vld1_u8(src + 16));

                for (i += 8*3, src += 8*3; i <= width - 8*3; i += 8*3, src += 8*3)
                {
                    internal::prefetch(src + 24);
                    s1 = vaddw_u8(s1, vld1_u8(src +  0));
                    s2 = vaddw_u8(s2, vld1_u8(src +  8));
                    s3 = vaddw_u8(s3, vld1_u8(src + 16));
                }

                vs1231 = vqaddq_u32(vs1231, vaddl_u16(vget_low_u16(s1), vget_high_u16(s2)));
                vs3123 = vqaddq_u32(vs3123, vaddl_u16(vget_low_u16(s2), vget_high_u16(s3)));
                vs2312 = vqaddq_u32(vs2312, vaddl_u16(vget_low_u16(s3), vget_high_u16(s1)));
            }

            u32 sum[12];
            vst1q_u32(sum+0, vs1231);
            vst1q_u32(sum+4, vs2312);
            vst1q_u32(sum+8, vs3123);

            for (; i < width; i += 3, src += 3)
            {
                sumdst[0] += src[0];
                sumdst[1] += src[1];
                sumdst[2] += src[2];
            }

            sumdst[0] += sum[0] + sum[3] + sum[6] + sum[9];
            sumdst[1] += sum[1] + sum[4] + sum[7] + sum[10];
            sumdst[2] += sum[2] + sum[5] + sum[8] + sum[11];
        }
        else
        {
            uint32x4_t vs = vdupq_n_u32(0);
            for (; i <= width - 257*8; i += 257*8, src += 257 * 8)
            {
                uint16x8_t s1 = vmovl_u8(vld1_u8(src));

                for (int j = 8; j < 257 * 8; j += 8)
                {
                    internal::prefetch(src + j);
                    s1 = vaddw_u8(s1, vld1_u8(src + j));
                }

                vs = vqaddq_u32(vs, vaddl_u16(vget_low_u16(s1), vget_high_u16(s1)));
            }
            if (i < width - 7)
            {
                uint16x8_t s1 = vmovl_u8(vld1_u8(src));

                for(i+=8,src+=8; i < width-7; i+=8,src+=8)
                {
                    internal::prefetch(src);
                    s1 = vaddw_u8(s1, vld1_u8(src));
                }
                vs = vqaddq_u32(vs, vaddl_u16(vget_low_u16(s1), vget_high_u16(s1)));
            }

            if (channels == 1)
            {
                uint32x2_t vs2 = vqadd_u32(vget_low_u32(vs), vget_high_u32(vs));
                uint32x2_t vs1 = vreinterpret_u32_u64(vpaddl_u32(vs2));

                u32 s0 = vget_lane_u32(vs1, 0);
                for(; i < width; ++i,++src)
                    s0 += src[0];
                sumdst[0] += s0;
            }
            else if (channels == 4)
            {
                vst1q_u32(sumdst, vqaddq_u32(vs, vld1q_u32(sumdst)));

                for(; i < width; i+=4,src+=4)
                {
                    sumdst[0] += src[0];
                    sumdst[1] += src[1];
                    sumdst[2] += src[2];
                    sumdst[3] += src[3];
                }
            }
            else//if (channels == 2)
            {
                uint32x2_t vs2 = vqadd_u32(vget_low_u32(vs), vget_high_u32(vs));
                vst1_u32(sumdst, vqadd_u32(vs2, vld1_u32(sumdst)));

                for(; i < width; i+=2,src+=2)
                {
                    sumdst[0] += src[0];
                    sumdst[1] += src[1];
                }
            }
        }//channels != 3
    }
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;
    (void)sumdst;
    (void)channels;
#endif
}

void sum(const Size2D &_size,
         const f32 * srcBase, ptrdiff_t srcStride,
         f64 * sumdst, u32 channels)
{
    internal::assertSupportedConfiguration(isSumSupported(channels));
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    const ptrdiff_t width = size.width * channels;

    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src = internal::getRowPtr( srcBase,  srcStride, k);
        ptrdiff_t i = 0;

        if (channels == 3)
        {
            float32x4_t vs1231 = vdupq_n_f32(0);
            float32x4_t vs2312 = vdupq_n_f32(0);
            float32x4_t vs3123 = vdupq_n_f32(0);
            for(; i <= width-12; i += 12)
            {
                internal::prefetch(src + i + 12);
                vs1231 = vaddq_f32(vs1231, vld1q_f32(src + i + 0));
                vs2312 = vaddq_f32(vs2312, vld1q_f32(src + i + 4));
                vs3123 = vaddq_f32(vs3123, vld1q_f32(src + i + 8));
            }

            f32 s[12];
            vst1q_f32(s + 0, vs1231);
            vst1q_f32(s + 4, vs2312);
            vst1q_f32(s + 8, vs3123);

            sumdst[0] += s[0] + s[3] + s[6] + s[9];
            sumdst[1] += s[1] + s[4] + s[7] + s[10];
            sumdst[2] += s[2] + s[5] + s[8] + s[11];
            for( ; i < width; i+=3)
            {
                sumdst[0] += src[i];
                sumdst[1] += src[i+1];
                sumdst[2] += src[i+2];
            }
        }
        else
        {
            float32x4_t vs = vdupq_n_f32(0);
            for(; i <= width-4; i += 4)
            {
                internal::prefetch(src + i);
                vs = vaddq_f32(vs, vld1q_f32(src+i));
            }

            if (channels == 1)
            {
                float32x2_t vs2 = vpadd_f32(vget_low_f32(vs), vget_high_f32(vs));
                f32 s[2];
                vst1_f32(s, vs2);

                sumdst[0] += s[0] + s[1];
                for( ; i < width; i++)
                    sumdst[0] += src[i];
            }
            else if (channels == 4)
            {
                f32 s[4];
                vst1q_f32(s, vs);

                sumdst[0] += s[0];
                sumdst[1] += s[1];
                sumdst[2] += s[2];
                sumdst[3] += s[3];
            }
            else//if (channels == 2)
            {
                float32x2_t vs2 = vadd_f32(vget_low_f32(vs), vget_high_f32(vs));
                f32 s[2];
                vst1_f32(s, vs2);

                sumdst[0] += s[0];
                sumdst[1] += s[1];

                if(i < width)
                {
                    sumdst[0] += src[i];
                    sumdst[1] += src[i+1];
                }
            }
        }//channels != 3
    }
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;
    (void)sumdst;
    (void)channels;
#endif
}

bool isSqsumSupported(u32 channels)
{
    return (channels && ((4/channels)*channels == 4));
}

void sqsum(const Size2D &_size,
           const u8 * srcBase, ptrdiff_t srcStride,
           f64 * sumdst, f64 * sqsumdst, u32 channels)
{
    internal::assertSupportedConfiguration(isSqsumSupported(channels));
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width*channels))
    {
        size.width *= size.height;
        size.height = 1;
    }
    const size_t width = size.width * channels;

    size_t blockSize0 = 1 << 23;
    size_t roiw8 = width & ~7;

    uint32x4_t v_zero = vdupq_n_u32(0u);

    for (size_t i = 0; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        size_t j = 0u;

        while (j < roiw8)
        {
            size_t blockSize = std::min(roiw8 - j, blockSize0) + j;
            uint32x4_t v_sum = v_zero;
            uint32x4_t v_sqsum = v_zero;

            for ( ; j < blockSize ; j += 8, src += 8)
            {
                internal::prefetch(src);
                uint8x8_t v_src0 = vld1_u8(src);

                uint16x8_t v_src = vmovl_u8(v_src0);
                uint16x4_t v_srclo = vget_low_u16(v_src), v_srchi = vget_high_u16(v_src);
                v_sum = vaddq_u32(v_sum, vaddl_u16(v_srclo, v_srchi));
                v_sqsum = vmlal_u16(v_sqsum, v_srclo, v_srclo);
                v_sqsum = vmlal_u16(v_sqsum, v_srchi, v_srchi);
            }

            u32 arsum[8];
            vst1q_u32(arsum, v_sum);
            vst1q_u32(arsum + 4, v_sqsum);

            sumdst[0] += (f64)arsum[0];
            sumdst[1 % channels] += (f64)arsum[1];
            sumdst[2 % channels] += (f64)arsum[2];
            sumdst[3 % channels] += (f64)arsum[3];
            sqsumdst[0] += (f64)arsum[4];
            sqsumdst[1 % channels] += (f64)arsum[5];
            sqsumdst[2 % channels] += (f64)arsum[6];
            sqsumdst[3 % channels] += (f64)arsum[7];
        }
        // collect a few last elements in the current row
        // it's ok to process channels elements per step
        // since we could handle 1,2 or 4 channels
        // we always have channels-fold amount of elements remaining
        for ( ; j < width; j+=channels, src+=channels)
        {
            for (u32 kk = 0; kk < channels; kk++)
            {
                u32 srcval = src[kk];
                sumdst[kk] += srcval;
                sqsumdst[kk] += srcval * srcval;
            }
        }
    }
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;
    (void)sumdst;
    (void)sqsumdst;
    (void)channels;
#endif
}

} // namespace CAROTENE_NS
