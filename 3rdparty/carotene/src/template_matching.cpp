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
 * Copyright (C) 2013-2015, NVIDIA Corporation, all rights reserved.
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
#include <cstring>

namespace CAROTENE_NS {

#define ENABLE4LINESMATCHING false  //Disabled since overall time for simultaneous 4 lines matching is greater than
                                    //time for simultaneous 2 lines matching for the same amount of data

bool isMatchTemplateSupported(const Size2D &tmplSize)
{
    return isSupportedConfiguration() &&
           tmplSize.width >= 8 && // Actually the function could process even shorter templates
                                  // but there will be no NEON optimization in this case
           (tmplSize.width * tmplSize.height) <= 256;
}

void matchTemplate(const Size2D &srcSize,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   const Size2D &tmplSize,
                   const u8 * tmplBase, ptrdiff_t tmplStride,
                   f32 * dstBase, ptrdiff_t dstStride,
                   bool normalize)
{
    internal::assertSupportedConfiguration(isMatchTemplateSupported(tmplSize));
#ifdef CAROTENE_NEON
    const size_t tmplW = tmplSize.width;
    const size_t tmplH = tmplSize.height;
    const size_t dstW  = srcSize.width  - tmplSize.width  + 1;
    const size_t dstH  = srcSize.height - tmplSize.height + 1;

    //template correlation part
    {
#if ENABLE4LINESMATCHING
        const size_t dstroiw4 = dstW & ~3u;
#endif
        const size_t dstroiw2 = dstW & ~1u;
        const size_t tmplroiw = tmplW & ~7u;
        const size_t dstride = dstStride >> 2;

        f32 *corr = dstBase;
        const u8  *imgrrow = srcBase;
        for(size_t r = 0; r < dstH; ++r, corr+=dstride, imgrrow+=srcStride)
        {
            size_t c = 0;
#if ENABLE4LINESMATCHING
            for(; c < dstroiw4; c+=4)
            {
                u32 dot[4] = {0, 0, 0, 0};
                uint32x4_t vdot0 = vmovq_n_u32(0);
                uint32x4_t vdot1 = vmovq_n_u32(0);
                uint32x4_t vdot2 = vmovq_n_u32(0);
                uint32x4_t vdot3 = vmovq_n_u32(0);

                const u8  *img = imgrrow;
                const u8 *tmpl = tmplBase;
                for(size_t i = 0; i < tmplH; ++i, tmpl+=tmplStride, img+=srcStride)
                {
                    size_t j = 0;
                    for(; j < tmplroiw; j+=8)
                    {
                        uint8x8_t vtmpl = vld1_u8(tmpl + j);

                        uint8x8_t vimg0 = vld1_u8(img + j + c + 0);
                        uint8x8_t vimg1 = vld1_u8(img + j + c + 1);
                        uint8x8_t vimg2 = vld1_u8(img + j + c + 2);
                        uint8x8_t vimg3 = vld1_u8(img + j + c + 3);

                        uint16x8_t vd0 = vmull_u8(vtmpl, vimg0);
                        uint16x8_t vd1 = vmull_u8(vtmpl, vimg1);
                        uint16x8_t vd2 = vmull_u8(vtmpl, vimg2);
                        uint16x8_t vd3 = vmull_u8(vtmpl, vimg3);

                        vdot0 = vpadalq_u16(vdot0, vd0);
                        vdot1 = vpadalq_u16(vdot1, vd1);
                        vdot2 = vpadalq_u16(vdot2, vd2);
                        vdot3 = vpadalq_u16(vdot3, vd3);
                    }
                    for(; j < tmplW; ++j)
                    {
                        dot[0] += tmpl[j] * img[j + c + 0];
                        dot[1] += tmpl[j] * img[j + c + 1];
                        dot[2] += tmpl[j] * img[j + c + 2];
                        dot[3] += tmpl[j] * img[j + c + 3];
                    }
                }
                uint32x4_t vdotx   = vld1q_u32(dot);
                uint32x2_t vdot_0  = vpadd_u32(vget_low_u32(vdot0), vget_high_u32(vdot0));
                uint32x2_t vdot_1  = vpadd_u32(vget_low_u32(vdot1), vget_high_u32(vdot1));
                uint32x2_t vdot_2  = vpadd_u32(vget_low_u32(vdot2), vget_high_u32(vdot2));
                uint32x2_t vdot_3  = vpadd_u32(vget_low_u32(vdot3), vget_high_u32(vdot3));
                uint32x2_t vdot_01 = vpadd_u32(vdot_0, vdot_1);
                uint32x2_t vdot_23 = vpadd_u32(vdot_2, vdot_3);

                vst1q_f32(corr + c, vcvtq_f32_u32(vaddq_u32(vdotx, vcombine_u32(vdot_01, vdot_23))));
            }
#endif

            for(; c < dstroiw2; c+=2)
            {
                u32 dot[2] = {0, 0};
                uint32x4_t vdot0 = vmovq_n_u32(0);
                uint32x4_t vdot1 = vmovq_n_u32(0);
                const u8  *img = imgrrow;
                const u8 *tmpl = tmplBase;
                for(size_t i = 0; i < tmplH; ++i, tmpl+=tmplStride, img+=srcStride)
                {
                    size_t j = 0;
                    for(; j < tmplroiw; j+=8)
                    {
                        uint8x8_t vtmpl = vld1_u8(tmpl + j);

                        uint8x8_t vimg0 = vld1_u8(img + j + c + 0);
                        uint8x8_t vimg1 = vld1_u8(img + j + c + 1);

                        uint16x8_t vd0 = vmull_u8(vtmpl, vimg0);
                        uint16x8_t vd1 = vmull_u8(vtmpl, vimg1);

                        vdot0 = vpadalq_u16(vdot0, vd0);
                        vdot1 = vpadalq_u16(vdot1, vd1);
                    }
                    for(; j < tmplW; ++j)
                    {
                        dot[0] += tmpl[j] * img[j + c + 0];
                        dot[1] += tmpl[j] * img[j + c + 1];
                    }
                }
                uint32x2_t vdotx  = vld1_u32(dot);
                uint32x2_t vdot_0 = vpadd_u32(vget_low_u32(vdot0), vget_high_u32(vdot0));
                uint32x2_t vdot_1 = vpadd_u32(vget_low_u32(vdot1), vget_high_u32(vdot1));
                uint32x2_t vdot_  = vpadd_u32(vdot_0, vdot_1);
                vst1_f32(corr + c, vcvt_f32_u32(vadd_u32(vdotx, vdot_)));
            }

            for(; c < dstW; ++c)
            {
                u32 dot = 0;
                uint32x4_t vdot = vmovq_n_u32(0);
                const u8  *img = imgrrow;
                const u8 *tmpl = tmplBase;
                for(size_t i = 0; i < tmplH; ++i, tmpl+=tmplStride, img+=srcStride)
                {
                    size_t j = 0;
                    for(; j < tmplroiw; j+=8)
                    {
                        uint8x8_t vtmpl = vld1_u8(tmpl + j);
                        uint8x8_t vimg  = vld1_u8(img + j + c);
                        uint16x8_t vd   = vmull_u8(vtmpl, vimg);
                        vdot = vpadalq_u16(vdot, vd);
                    }
                    for(; j < tmplW; ++j)
                        dot += tmpl[j] * img[j + c];
                }
                u32 wdot[2];
                vst1_u32(wdot, vpadd_u32(vget_low_u32(vdot), vget_high_u32(vdot)));
                dot += wdot[0] + wdot[1];
                corr[c] = (f32)dot;
            }
        }
    }

    if(normalize)
    {
        f32 tn = std::sqrt((f32)normL2(tmplSize, tmplBase, tmplStride));

        size_t iw = srcSize.width+1;
        size_t ih = srcSize.height+1;
        std::vector<f64> _sqsum(iw*ih);
        f64 *sqsum = &_sqsum[0];
        memset(sqsum, 0, iw*sizeof(f64));
        for(size_t i = 1; i < ih; ++i)
            sqsum[iw*i] = 0.;
        sqrIntegral(srcSize, srcBase, srcStride, sqsum + iw + 1, iw*sizeof(f64));

        for(size_t i = 0; i < dstH; ++i)
        {
            f32 *result = internal::getRowPtr(dstBase, dstStride, i);
            for(size_t j = 0; j < dstW; ++j)
            {
                double s2 = sqsum[iw*i + j] +
                            sqsum[iw*(i + tmplSize.height) + j + tmplSize.width] -
                            sqsum[iw*(i + tmplSize.height) + j] -
                            sqsum[iw*i + j + tmplSize.width];

                result[j] /= tn * std::sqrt(s2);
            }
        }
    }
#else
    (void)srcSize;
    (void)srcBase;
    (void)srcStride;
    (void)tmplBase;
    (void)tmplStride;
    (void)dstBase;
    (void)dstStride;
    (void)normalize;
#endif
}

} // namespace CAROTENE_NS
