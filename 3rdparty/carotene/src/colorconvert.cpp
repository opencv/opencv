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

#include "saturate_cast.hpp"
#include "vround_helper.hpp"

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

namespace {

enum
{
    SHIFT = 15,
    SHIFT_DELTA = 1 << (SHIFT - 1),

    R2Y_BT601   = 9798,
    G2Y_BT601   = 19235,
    B2Y_BT601   = 3735,

    R2Y_BT709   = 3483,
    G2Y_BT709   = 11718,
    B2Y_BT709   = 1183,
};

inline uint8x8_t convertToGray(const uint16x8_t & v_r,
                               const uint16x8_t & v_g,
                               const uint16x8_t & v_b,
                               const uint16x4_t & v_r2y,
                               const uint16x4_t & v_g2y,
                               const uint16x4_t & v_b2y)
{
    uint32x4_t v_dst0 = vmull_u16(vget_low_u16(v_g), v_g2y);
    uint32x4_t v_dst1 = vmull_u16(vget_high_u16(v_g), v_g2y);

    v_dst0 = vmlal_u16(v_dst0, vget_low_u16(v_r), v_r2y);
    v_dst1 = vmlal_u16(v_dst1, vget_high_u16(v_r), v_r2y);

    v_dst0 = vmlal_u16(v_dst0, vget_low_u16(v_b), v_b2y);
    v_dst1 = vmlal_u16(v_dst1, vget_high_u16(v_b), v_b2y);

    uint8x8_t v_gray = vqmovn_u16(vcombine_u16(vrshrn_n_u32(v_dst0, SHIFT),
                                               vrshrn_n_u32(v_dst1, SHIFT)));

    return v_gray;
}

} // namespace

#endif

void rgb2gray(const Size2D &size, COLOR_SPACE color_space,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    const u32 R2Y = color_space == COLOR_SPACE_BT601 ? R2Y_BT601 : R2Y_BT709;
    const u32 G2Y = color_space == COLOR_SPACE_BT601 ? G2Y_BT601 : G2Y_BT709;
    const u32 B2Y = color_space == COLOR_SPACE_BT601 ? B2Y_BT601 : B2Y_BT709;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register int16x4_t v_r2y asm ("d31") = vmov_n_s16(R2Y);
    register int16x4_t v_g2y asm ("d30") = vmov_n_s16(G2Y);
    register int16x4_t v_b2y asm ("d29") = vmov_n_s16(B2Y);
#else
    uint16x4_t v_r2y = vdup_n_u16(R2Y),
               v_g2y = vdup_n_u16(G2Y),
               v_b2y = vdup_n_u16(B2Y);

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
        for (; dj < roiw8; sj += 24, dj += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
            "vld3.8 {d0-d2}, [%[in]] @RGB                       \n\t"
            "vmovl.u8 q2, d0         @R (d4,d5)                 \n\t"
            "vmovl.u8 q3, d1         @G (d6,d7)                 \n\t"
            "vmovl.u8 q4, d2         @B (d8,d9)                 \n\t"
            "vmull.u16 q5, d6, d30   @Y (q5,q6):  G             \n\t"
            "vmull.u16 q6, d7, d30   @Y (q5,q6):  G             \n\t"
            "vmlal.s16 q5, d8, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q6, d9, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q5, d4, d31   @Y (q5,q6):  GBR           \n\t"
            "vmlal.s16 q6, d5, d31   @Y (q5,q6):  GBR           \n\t"
            "vrshrn.s32 d8, q5, #14  @Y  -> q4                  \n\t"
            "vrshrn.s32 d9, q6, #14  @Y  -> q4                  \n\t"
            "vqmovn.u16 d4, q4                                  \n\t"
            "vst1.8 {d4}, [%[out]]                              \n\t"
            : /*no output*/
            : [out] "r" (dst + dj), [in] "r" (src + sj), "w" (v_r2y), "w" (v_g2y), "w" (v_b2y)
            : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
            );
        }
#else
        for (; dj < roiw16; sj += 48, dj += 16)
        {
            internal::prefetch(src + sj);
            uint8x16x3_t v_src0 = vld3q_u8(src + sj);
            // 0
            uint16x8_t v_r = vmovl_u8(vget_low_u8(v_src0.val[0])),
                       v_g = vmovl_u8(vget_low_u8(v_src0.val[1])),
                       v_b = vmovl_u8(vget_low_u8(v_src0.val[2]));
            uint8x8_t v_gray0 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            v_r = vmovl_u8(vget_high_u8(v_src0.val[0])),
            v_g = vmovl_u8(vget_high_u8(v_src0.val[1])),
            v_b = vmovl_u8(vget_high_u8(v_src0.val[2]));
            uint8x8_t v_gray1 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1q_u8(dst + dj, vcombine_u8(v_gray0, v_gray1));
        }

        if (dj < roiw8)
        {
            uint8x8x3_t v_src = vld3_u8(src + sj);
            uint16x8_t v_r = vmovl_u8(v_src.val[0]),
                       v_g = vmovl_u8(v_src.val[1]),
                       v_b = vmovl_u8(v_src.val[2]);
            uint8x8_t v_gray = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1_u8(dst + dj, v_gray);
            sj += 24; dj += 8;
        }
#endif

        for (; dj < size.width; sj += 3, dj++)
        {
            u32 val = src[sj] * R2Y + src[sj + 1] * G2Y + src[sj + 2] * B2Y;
            dst[dj] = internal::saturate_cast<u8>((val + SHIFT_DELTA) >> SHIFT);
        }
    }
#else
    (void)size;
    (void)color_space;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgbx2gray(const Size2D &size, COLOR_SPACE color_space,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    const u32 R2Y = color_space == COLOR_SPACE_BT601 ? R2Y_BT601 : R2Y_BT709;
    const u32 G2Y = color_space == COLOR_SPACE_BT601 ? G2Y_BT601 : G2Y_BT709;
    const u32 B2Y = color_space == COLOR_SPACE_BT601 ? B2Y_BT601 : B2Y_BT709;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register int16x4_t v_r2y asm ("d31") = vmov_n_s16(R2Y);
    register int16x4_t v_g2y asm ("d30") = vmov_n_s16(G2Y);
    register int16x4_t v_b2y asm ("d29") = vmov_n_s16(B2Y);
#else
    uint16x4_t v_r2y = vdup_n_u16(R2Y),
               v_g2y = vdup_n_u16(G2Y),
               v_b2y = vdup_n_u16(B2Y);

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
        for (; dj < roiw8; sj += 32, dj += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
            "vld4.8 {d0-d3}, [%[in]] @RGBA                      \n\t"
            "vmovl.u8 q2, d0         @R (d4,d5)                 \n\t"
            "vmovl.u8 q3, d1         @G (d6,d7)                 \n\t"
            "vmovl.u8 q4, d2         @B (d8,d9)                 \n\t"
            "vmull.u16 q5, d6, d30   @Y (q5,q6):  G             \n\t"
            "vmull.u16 q6, d7, d30   @Y (q5,q6):  G             \n\t"
            "vmlal.s16 q5, d8, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q6, d9, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q5, d4, d31   @Y (q5,q6):  GBR           \n\t"
            "vmlal.s16 q6, d5, d31   @Y (q5,q6):  GBR           \n\t"
            "vrshrn.s32 d8, q5, #14  @Y  -> q4                  \n\t"
            "vrshrn.s32 d9, q6, #14  @Y  -> q4                  \n\t"
            "vqmovn.u16 d4, q4                                  \n\t"
            "vst1.8 {d4}, [%[out]]                              \n\t"
            : /*no output*/
            : [out] "r" (dst + dj), [in] "r" (src + sj), "w" (v_r2y), "w" (v_g2y), "w" (v_b2y)
            : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
            );
        }
#else
        for (; dj < roiw16; sj += 64, dj += 16)
        {
            internal::prefetch(src + sj);
            uint8x16x4_t v_src0 = vld4q_u8(src + sj);

            // 0
            uint16x8_t v_r = vmovl_u8(vget_low_u8(v_src0.val[0])),
                       v_g = vmovl_u8(vget_low_u8(v_src0.val[1])),
                       v_b = vmovl_u8(vget_low_u8(v_src0.val[2]));
            uint8x8_t v_gray0 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            v_r = vmovl_u8(vget_high_u8(v_src0.val[0])),
            v_g = vmovl_u8(vget_high_u8(v_src0.val[1])),
            v_b = vmovl_u8(vget_high_u8(v_src0.val[2]));
            uint8x8_t v_gray1 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1q_u8(dst + dj, vcombine_u8(v_gray0, v_gray1));
        }

        if (dj < roiw8)
        {
            uint8x8x4_t v_src = vld4_u8(src + sj);
            uint16x8_t v_r = vmovl_u8(v_src.val[0]),
                       v_g = vmovl_u8(v_src.val[1]),
                       v_b = vmovl_u8(v_src.val[2]);
            uint8x8_t v_gray = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1_u8(dst + dj, v_gray);
            sj += 32; dj += 8;
        }
#endif

        for (; dj < size.width; sj += 4, dj++)
        {
            u32 val = src[sj] * R2Y + src[sj + 1] * G2Y + src[sj + 2] * B2Y;
            dst[dj] = internal::saturate_cast<u8>((val + SHIFT_DELTA) >> SHIFT);
        }
    }
#else
    (void)size;
    (void)color_space;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bgr2gray(const Size2D &size, COLOR_SPACE color_space,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    const u32 R2Y = color_space == COLOR_SPACE_BT601 ? R2Y_BT601 : R2Y_BT709;
    const u32 G2Y = color_space == COLOR_SPACE_BT601 ? G2Y_BT601 : G2Y_BT709;
    const u32 B2Y = color_space == COLOR_SPACE_BT601 ? B2Y_BT601 : B2Y_BT709;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register int16x4_t v_r2y asm ("d31") = vmov_n_s16(R2Y);
    register int16x4_t v_g2y asm ("d30") = vmov_n_s16(G2Y);
    register int16x4_t v_b2y asm ("d29") = vmov_n_s16(B2Y);
#else
    uint16x4_t v_r2y = vdup_n_u16(R2Y),
               v_g2y = vdup_n_u16(G2Y),
               v_b2y = vdup_n_u16(B2Y);

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
        for (; dj < roiw8; sj += 24, dj += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
            "vld3.8 {d0-d2}, [%[in]] @BGR                       \n\t"
            "vmovl.u8 q2, d2         @R (d4,d5)                 \n\t"
            "vmovl.u8 q3, d1         @G (d6,d7)                 \n\t"
            "vmovl.u8 q4, d0         @B (d8,d9)                 \n\t"
            "vmull.u16 q5, d6, d30   @Y (q5,q6):  G             \n\t"
            "vmull.u16 q6, d7, d30   @Y (q5,q6):  G             \n\t"
            "vmlal.s16 q5, d8, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q6, d9, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q5, d4, d31   @Y (q5,q6):  GBR           \n\t"
            "vmlal.s16 q6, d5, d31   @Y (q5,q6):  GBR           \n\t"
            "vrshrn.s32 d8, q5, #14  @Y  -> q4                  \n\t"
            "vrshrn.s32 d9, q6, #14  @Y  -> q4                  \n\t"
            "vqmovn.u16 d4, q4                                  \n\t"
            "vst1.8 {d4}, [%[out]]                              \n\t"
            : /*no output*/
            : [out] "r" (dst + dj), [in] "r" (src + sj), "w" (v_r2y), "w" (v_g2y), "w" (v_b2y)
            : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
            );
        }
#else
        for (; dj < roiw16; sj += 48, dj += 16)
        {
            internal::prefetch(src + sj);
            uint8x16x3_t v_src0 = vld3q_u8(src + sj);

            // 0
            uint16x8_t v_b = vmovl_u8(vget_low_u8(v_src0.val[0])),
                       v_g = vmovl_u8(vget_low_u8(v_src0.val[1])),
                       v_r = vmovl_u8(vget_low_u8(v_src0.val[2]));
            uint8x8_t v_gray0 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            v_b = vmovl_u8(vget_high_u8(v_src0.val[0])),
            v_g = vmovl_u8(vget_high_u8(v_src0.val[1])),
            v_r = vmovl_u8(vget_high_u8(v_src0.val[2]));
            uint8x8_t v_gray1 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1q_u8(dst + dj, vcombine_u8(v_gray0, v_gray1));
        }

        if (dj < roiw8)
        {
            uint8x8x3_t v_src = vld3_u8(src + sj);
            uint16x8_t v_b = vmovl_u8(v_src.val[0]),
                       v_g = vmovl_u8(v_src.val[1]),
                       v_r = vmovl_u8(v_src.val[2]);
            uint8x8_t v_gray = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1_u8(dst + dj, v_gray);
            sj += 24; dj += 8;
        }
#endif

        for (; dj < size.width; sj += 3, dj++)
        {
            u32 val = src[sj] * B2Y + src[sj + 1] * G2Y + src[sj + 2] * R2Y;
            dst[dj] = internal::saturate_cast<u8>((val + SHIFT_DELTA) >> SHIFT);
        }
    }
#else
    (void)size;
    (void)color_space;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bgrx2gray(const Size2D &size, COLOR_SPACE color_space,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    const u32 R2Y = color_space == COLOR_SPACE_BT601 ? R2Y_BT601 : R2Y_BT709;
    const u32 G2Y = color_space == COLOR_SPACE_BT601 ? G2Y_BT601 : G2Y_BT709;
    const u32 B2Y = color_space == COLOR_SPACE_BT601 ? B2Y_BT601 : B2Y_BT709;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register int16x4_t v_r2y asm ("d31") = vmov_n_s16(R2Y);
    register int16x4_t v_g2y asm ("d30") = vmov_n_s16(G2Y);
    register int16x4_t v_b2y asm ("d29") = vmov_n_s16(B2Y);
#else
    uint16x4_t v_r2y = vdup_n_u16(R2Y),
               v_g2y = vdup_n_u16(G2Y),
               v_b2y = vdup_n_u16(B2Y);

    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
        for (; dj < roiw8; sj += 32, dj += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
            "vld4.8 {d0-d3}, [%[in]] @BGRA                      \n\t"
            "vmovl.u8 q2, d2         @R (d4,d5)                 \n\t"
            "vmovl.u8 q3, d1         @G (d6,d7)                 \n\t"
            "vmovl.u8 q4, d0         @B (d8,d9)                 \n\t"
            "vmull.u16 q5, d6, d30   @Y (q5,q6):  G             \n\t"
            "vmull.u16 q6, d7, d30   @Y (q5,q6):  G             \n\t"
            "vmlal.s16 q5, d8, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q6, d9, d29   @Y (q5,q6):  GB            \n\t"
            "vmlal.s16 q5, d4, d31   @Y (q5,q6):  GBR           \n\t"
            "vmlal.s16 q6, d5, d31   @Y (q5,q6):  GBR           \n\t"
            "vrshrn.s32 d8, q5, #14  @Y  -> q4                  \n\t"
            "vrshrn.s32 d9, q6, #14  @Y  -> q4                  \n\t"
            "vqmovn.u16 d4, q4                                  \n\t"
            "vst1.8 {d4}, [%[out]]                              \n\t"
            : /*no output*/
            : [out] "r" (dst + dj), [in] "r" (src + sj), "w" (v_r2y), "w" (v_g2y), "w" (v_b2y)
            : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
            );
        }
#else
        for (; dj < roiw16; sj += 64, dj += 16)
        {
            internal::prefetch(src + sj);
            uint8x16x4_t v_src0 = vld4q_u8(src + sj);

            // 0
            uint16x8_t v_b = vmovl_u8(vget_low_u8(v_src0.val[0])),
                       v_g = vmovl_u8(vget_low_u8(v_src0.val[1])),
                       v_r = vmovl_u8(vget_low_u8(v_src0.val[2]));
            uint8x8_t v_gray0 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            v_b = vmovl_u8(vget_high_u8(v_src0.val[0])),
            v_g = vmovl_u8(vget_high_u8(v_src0.val[1])),
            v_r = vmovl_u8(vget_high_u8(v_src0.val[2]));
            uint8x8_t v_gray1 = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1q_u8(dst + dj, vcombine_u8(v_gray0, v_gray1));
        }

        if (dj < roiw8)
        {
            uint8x8x4_t v_src = vld4_u8(src + sj);
            uint16x8_t v_b = vmovl_u8(v_src.val[0]),
                       v_g = vmovl_u8(v_src.val[1]),
                       v_r = vmovl_u8(v_src.val[2]);
            uint8x8_t v_gray = convertToGray(v_r, v_g, v_b, v_r2y, v_g2y, v_b2y);

            vst1_u8(dst + dj, v_gray);
            sj += 32; dj += 8;
        }
#endif

        for (; dj < size.width; sj += 4, dj++)
        {
            u32 val = src[sj] * B2Y + src[sj + 1] * G2Y + src[sj + 2] * R2Y;
            dst[dj] = internal::saturate_cast<u8>((val + SHIFT_DELTA) >> SHIFT);
        }
    }
#else
    (void)size;
    (void)color_space;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void gray2rgb(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

        for (; sj < roiw16; sj += 16, dj += 48)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
            "vld1.8 {d0-d1}, [%[in0]]            \n\t"
            "vmov.8 q1, q0                       \n\t"
            "vmov.8 q2, q0                       \n\t"
            "vmov.8 q3, q1                       \n\t"
            "vst3.8 {d2, d4, d6}, [%[out0]]      \n\t"
            "vst3.8 {d3, d5, d7}, [%[out1]]      \n\t"
            : /*no output*/
            : [out0] "r" (dst + dj),      [out1] "r" (dst + dj + 24),
              [in0] "r" (src + sj)
            : "d0","d1","d2","d3","d4","d5","d6","d7"
            );
#else
            uint8x16x3_t vRgb1;
            vRgb1.val[0] = vld1q_u8(src + sj);

            vRgb1.val[1] = vRgb1.val[0];
            vRgb1.val[2] = vRgb1.val[0];

            vst3q_u8(dst + dj, vRgb1);
#endif
        }

        if (sj < roiw8)
        {
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
            "vld1.8 {d0}, [%[in]]                \n\t"
            "vmov.8 d1, d0                       \n\t"
            "vmov.8 d2, d0                       \n\t"
            "vst3.8 {d0-d2}, [%[out]]            \n\t"
            : /*no output*/
            : [out] "r" (dst + dj), [in] "r" (src + sj)
            : "d0","d1","d2"
            );
#else
            uint8x8x3_t vRgb2;
            vRgb2.val[0] = vld1_u8(src + sj);
            vRgb2.val[1] = vRgb2.val[0];
            vRgb2.val[2] = vRgb2.val[0];

            vst3_u8(dst + dj, vRgb2);
#endif
            sj += 8; dj += 24;
        }

        for (; sj < size.width; sj++, dj += 3)
        {
            dst[dj+0] = src[sj];
            dst[dj+1] = src[sj];
            dst[dj+2] = src[sj];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void gray2rgbx(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register uint8x16_t vc255   asm ("q4") = vmovq_n_u8(255);
#else
    uint8x16x4_t vRgba;
    uint8x8x4_t  vRgba2;
    vRgba.val[3]  = vmovq_n_u8(255);
    vRgba2.val[3] = vget_low_u8(vRgba.val[3]);
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

        for (; sj < roiw16; sj += 16, dj += 64)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
            "vld1.8 {d0-d1}, [%[in0]]            \n\t"
            "vmov.8 q1, q0                       \n\t"
            "vmov.8 q2, q0                       \n\t"
            "vmov.8 q3, q1                       \n\t"
            "vst4.8 {d2, d4, d6, d8}, [%[out0]]  \n\t"
            "vst4.8 {d3, d5, d7, d9}, [%[out1]]  \n\t"
            : /*no output*/
            : [out0] "r" (dst + dj),      [out1] "r" (dst + dj + 32),
              [in0] "r" (src + sj),
              "w" (vc255)
            : "d0","d1","d2","d3","d4","d5","d6","d7"
            );
#else
            vRgba.val[0]  = vld1q_u8(src + sj);

            vRgba.val[1] = vRgba.val[0];
            vRgba.val[2] = vRgba.val[0];

            vst4q_u8(dst + dj, vRgba);
#endif
        }

        if (sj < roiw8)
        {
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
            "vld1.8 {d5}, [%[in]]                \n\t"
            "vmov.8 d6, d5                       \n\t"
            "vmov.8 d7, d5                       \n\t"
            "vst4.8 {d5-d8}, [%[out]]            \n\t"
            : /*no output*/
            : [out] "r" (dst + dj), [in] "r" (src + sj), "w" (vc255)
            : "d5","d6","d7"
            );
#else
            vRgba2.val[0] = vld1_u8(src + sj);
            vRgba2.val[1] = vRgba2.val[0];
            vRgba2.val[2] = vRgba2.val[0];

            vst4_u8(dst + dj, vRgba2);
#endif
            sj += 8; dj += 32;
        }

        for (; sj < size.width; sj++, dj += 4)
        {
            dst[dj+0] = src[sj];
            dst[dj+1] = src[sj];
            dst[dj+2] = src[sj];
            dst[dj+3] = 255;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgb2rgbx(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
    register uint8x8_t vc255_0  asm ("d3") = vmov_n_u8(255);
#else
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    union { uint8x16x4_t v4; uint8x16x3_t v3; } v_dst0;
    v_dst0.v4.val[3] = vdupq_n_u8(255);
    union { uint8x8x4_t v4; uint8x8x3_t v3; } v_dst;
    v_dst.v4.val[3] = vdup_n_u8(255);
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
        for (; j < roiw8; sj += 24, dj += 32, j += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
                "vld3.8 {d0, d1, d2}, [%[in0]]             \n\t"
                "vst4.8 {d0, d1, d2, d3}, [%[out0]]        \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [in0]  "r" (src + sj),
                  "w" (vc255_0)
                : "d0","d1","d2"
            );
        }
#else
        for (; j < roiw16; sj += 48, dj += 64, j += 16)
        {
            internal::prefetch(src + sj);
            v_dst0.v3 = vld3q_u8(src + sj);
            vst4q_u8(dst + dj, v_dst0.v4);
        }

        if (j < roiw8)
        {
            v_dst.v3 = vld3_u8(src + sj);
            vst4_u8(dst + dj, v_dst.v4);
            sj += 24; dj += 32; j += 8;
        }
#endif

        for (; j < size.width; ++j, sj += 3, dj += 4)
        {
            dst[dj] = src[sj];
            dst[dj + 1] = src[sj + 1];
            dst[dj + 2] = src[sj + 2];
            dst[dj + 3] = 255;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgbx2rgb(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
#if !(!defined(__aarch64__) && defined(__GNUC__) && defined(__arm__))
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
    union { uint8x16x4_t v4; uint8x16x3_t v3; } v_dst0;
    union { uint8x8x4_t v4; uint8x8x3_t v3; } v_dst;
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
        for (; j < roiw8; sj += 32, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
                "vld4.8 {d0, d1, d2, d3}, [%[in0]]         \n\t"
                "vst3.8 {d0, d1, d2}, [%[out0]]            \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [in0]  "r" (src + sj)
                : "d0","d1","d2","d3"
            );
        }
#else
        for (; j < roiw16; sj += 64, dj += 48, j += 16)
        {
            internal::prefetch(src + sj);
            v_dst0.v4 = vld4q_u8(src + sj);
            vst3q_u8(dst + dj, v_dst0.v3);
        }

        if (j < roiw8)
        {
            v_dst.v4 = vld4_u8(src + sj);
            vst3_u8(dst + dj, v_dst.v3);
            sj += 32; dj += 24; j += 8;
        }
#endif

        for (; j < size.width; ++j, sj += 4, dj += 3)
        {
            dst[dj] = src[sj];
            dst[dj + 1] = src[sj + 1];
            dst[dj + 2] = src[sj + 2];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgb2bgr(const Size2D &size,
             const u8 * srcBase, ptrdiff_t srcStride,
             u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#if !(!defined(__aarch64__) && defined(__GNUC__) && defined(__arm__))
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;


#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
        for (; j < roiw8; sj += 24, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
                "vld3.8 {d0, d1, d2}, [%[in0]]             \n\t"
                "vswp d0, d2                               \n\t"
                "vst3.8 {d0, d1, d2}, [%[out0]]            \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [in0]  "r" (src + sj)
                : "d0","d1","d2"
            );
        }
#else
        for (; j < roiw16; sj += 48, dj += 48, j += 16)
        {
            internal::prefetch(src + sj);
            uint8x16x3_t vals0 = vld3q_u8(src + sj);

            std::swap(vals0.val[0], vals0.val[2]);

            vst3q_u8(dst + dj, vals0);
        }

        if (j < roiw8)
        {
            uint8x8x3_t vals = vld3_u8(src + sj);
            std::swap(vals.val[0], vals.val[2]);
            vst3_u8(dst + dj, vals);
            sj += 24; dj += 24; j += 8;
        }
#endif

        for (; j < size.width; ++j, sj += 3, dj += 3)
        {
            u8 b = src[sj + 2];//Handle src == dst case
            dst[dj + 2] = src[sj    ];
            dst[dj + 1] = src[sj + 1];
            dst[dj    ] = b;
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgbx2bgrx(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#if !(!defined(__aarch64__) && defined(__GNUC__) && defined(__arm__))
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
        for (; j < roiw8; sj += 32, dj += 32, j += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
                "vld4.8 {d0, d1, d2, d3}, [%[in0]]         \n\t"
                "vswp d0, d2                               \n\t"
                "vst4.8 {d0, d1, d2, d3}, [%[out0]]        \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [in0]  "r" (src + sj)
                : "d0","d1","d2","d3"
            );
        }
#else
        for (; j < roiw16; sj += 64, dj += 64, j += 16)
        {
            internal::prefetch(src + sj);
            uint8x16x4_t vals0 = vld4q_u8(src + sj);

            std::swap(vals0.val[0], vals0.val[2]);

            vst4q_u8(dst + dj, vals0);
        }

        if (j < roiw8)
        {
            uint8x8x4_t vals = vld4_u8(src + sj);
            std::swap(vals.val[0], vals.val[2]);
            vst4_u8(dst + dj, vals);
            sj += 32; dj += 32; j += 8;
        }
#endif

        for (; j < size.width; ++j, sj += 4, dj += 4)
        {
            u8 b = src[sj + 2];//Handle src == dst case
            dst[dj + 2] = src[sj    ];
            dst[dj + 1] = src[sj + 1];
            dst[dj    ] = b;
            dst[dj + 3] = src[sj + 3];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgbx2bgr(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#if !(!defined(__aarch64__) && defined(__GNUC__) && defined(__arm__))
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
        for (; j < roiw8; sj += 32, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
                "vld4.8 {d0, d1, d2, d3}, [%[in0]]         \n\t"
                "vswp d0, d2                               \n\t"
                "vst3.8 {d0, d1, d2}, [%[out0]]            \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [in0]  "r" (src + sj)
                : "d0","d1","d2","d3"
            );
        }
#else
        for (; j < roiw16; sj += 64, dj += 48, j += 16)
        {
            internal::prefetch(src + sj);
            union { uint8x16x4_t v4; uint8x16x3_t v3; } vals0;
            vals0.v4 = vld4q_u8(src + sj);
            std::swap(vals0.v3.val[0], vals0.v3.val[2]);
            vst3q_u8(dst + dj, vals0.v3);
        }

        if (j < roiw8)
        {
            union { uint8x8x4_t v4; uint8x8x3_t v3; } vals;
            vals.v4 = vld4_u8(src + sj);
            std::swap(vals.v3.val[0], vals.v3.val[2]);
            vst3_u8(dst + dj, vals.v3);
            sj += 32; dj += 24; j += 8;
        }
#endif

        for (; j < size.width; ++j, sj += 4, dj += 3)
        {
            dst[dj + 2] = src[sj    ];
            dst[dj + 1] = src[sj + 1];
            dst[dj    ] = src[sj + 2];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgb2bgrx(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
    register uint8x8_t vc255  asm ("d3") = vmov_n_u8(255);
#else
    union { uint8x16x4_t v4; uint8x16x3_t v3; } vals0;
    vals0.v4.val[3] = vmovq_n_u8(255);
    union { uint8x8x4_t v4; uint8x8x3_t v3; } vals8;
    vals8.v4.val[3] = vmov_n_u8(255);
#endif

#if !(!defined(__aarch64__) && defined(__GNUC__) && defined(__arm__))
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

#if !defined(__aarch64__) && defined(__GNUC__) && defined(__arm__)
        for (; j < roiw8; sj += 24, dj += 32, j += 8)
        {
            internal::prefetch(src + sj);
            __asm__ (
                "vld3.8 {d0, d1, d2}, [%[in0]]             \n\t"
                "vswp d0, d2                               \n\t"
                "vst4.8 {d0, d1, d2, d3}, [%[out0]]        \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [in0]  "r" (src + sj),
                   "w" (vc255)
                : "d0","d1","d2"
            );
        }
#else
        for (; j < roiw16; sj += 48, dj += 64, j += 16)
        {
            internal::prefetch(src + sj);
            vals0.v3 = vld3q_u8(src + sj);
            std::swap(vals0.v4.val[0], vals0.v4.val[2]);
            vst4q_u8(dst + dj, vals0.v4);
        }

        if (j < roiw8)
        {
            vals8.v3 = vld3_u8(src + sj);
            std::swap(vals8.v4.val[0], vals8.v4.val[2]);
            vst4_u8(dst + dj, vals8.v4);
            sj += 24; dj += 32; j += 8;
        }
#endif

        for (; j < size.width; ++j, sj += 3, dj += 4)
        {
            dst[dj + 3] = 255;
            dst[dj + 2] = src[sj    ];
            dst[dj + 1] = src[sj + 1];
            dst[dj    ] = src[sj + 2];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

namespace {

#ifdef CAROTENE_NEON
inline uint8x8x3_t convertToHSV(const uint8x8_t vR, const uint8x8_t vG, const uint8x8_t vB,
                                const s32 hrange )
{
    const s32 hsv_shift = 12;
    const f32 vsdiv_table = f32(255 << hsv_shift);
    f32 vhdiv_table = f32(hrange << hsv_shift);
    const s32 vhrange = hrange;
    const s32 v0 = s32(0);
    const s32 vshift = s32(1 << (hsv_shift-1));
    const s32 v6 = s32(6);

    uint8x8_t vMin = vmin_u8(vR, vG);
    uint8x8_t vMax = vmax_u8(vR, vG);

    uint16x8_t vR_u16 = vmovl_u8(vR);
    uint16x8_t vG_u16 = vmovl_u8(vG);

    vMax = vmax_u8(vMax, vB);
    vMin = vmin_u8(vMin, vB);
    uint16x8_t vB_u16 = vmovl_u8(vB);

    uint16x8_t vDiff = vsubl_u8(vMax, vMin);

    uint16x8_t vV = vmovl_u8(vMax);
    uint16x8_t vDiffx2 = vaddq_u16(vDiff, vDiff);
    uint32x4_t vDiffL = vmovl_u16(vget_low_u16(vDiff));
    uint32x4_t vDiffH = vmovl_u16(vget_high_u16(vDiff));

    uint16x8_t vVEqR = vceqq_u16(vR_u16, vV);
    uint16x8_t vVEqG = vceqq_u16(vG_u16, vV);

    int16x8_t vG_B = vsubq_s16(vreinterpretq_s16_u16(vG_u16), vreinterpretq_s16_u16(vB_u16));
    uint16x8_t vInvR = vmvnq_u16(vVEqR);
    int16x8_t vB_R = vsubq_s16(vreinterpretq_s16_u16(vB_u16), vreinterpretq_s16_u16(vR_u16));
    int16x8_t vR_G = vsubq_s16(vreinterpretq_s16_u16(vR_u16), vreinterpretq_s16_u16(vG_u16));

    uint16x8_t vMask2 = vandq_u16(vVEqG, vInvR);
    vR_u16 = vandq_u16(vreinterpretq_u16_s16(vG_B), vVEqR);
    int16x8_t vH2 = vaddq_s16(vB_R, vreinterpretq_s16_u16(vDiffx2));

    vVEqR = vmvnq_u16(vVEqG);
    vB_R = vaddq_s16(vreinterpretq_s16_u16(vDiffx2), vreinterpretq_s16_u16(vDiffx2));
    vG_B = vandq_s16(vreinterpretq_s16_u16(vInvR), vreinterpretq_s16_u16(vVEqR));
    vInvR = vandq_u16(vreinterpretq_u16_s16(vH2), vMask2);
    vR_G = vaddq_s16(vR_G, vB_R);
    int16x8_t vH = vaddq_s16(vreinterpretq_s16_u16(vR_u16), vreinterpretq_s16_u16(vInvR));

    uint32x4_t vV_L = vmovl_u16(vget_low_u16(vV));
    vR_G = vandq_s16(vR_G, vG_B);
    uint32x4_t vV_H = vmovl_u16(vget_high_u16(vV));
    int16x8_t vDiff4 = vaddq_s16(vH, vR_G);

    int32x4_t vc6 = vdupq_n_s32(v6);
    uint32x4_t vLine1 = vmulq_u32(vDiffL, vreinterpretq_u32_s32(vc6));
    uint32x4_t vLine2 = vmulq_u32(vDiffH, vreinterpretq_u32_s32(vc6));

    float32x4_t vF1 = vcvtq_f32_u32(vV_L);
    float32x4_t vF2 = vcvtq_f32_u32(vV_H);
    float32x4_t vHF1 = vcvtq_f32_u32(vLine1);
    float32x4_t vHF2 = vcvtq_f32_u32(vLine2);

    float32x4_t vXInv1 = vrecpeq_f32(vF1);
    float32x4_t vXInv2 = vrecpeq_f32(vF2);
    float32x4_t vXInv3 = vrecpeq_f32(vHF1);
    float32x4_t vXInv4 = vrecpeq_f32(vHF2);

    float32x4_t vSt1 = vrecpsq_f32(vXInv1, vF1);
    float32x4_t vSt2 = vrecpsq_f32(vXInv2, vF2);
    float32x4_t vSt3 = vrecpsq_f32(vXInv3, vHF1);
    float32x4_t vSt4 = vrecpsq_f32(vXInv4, vHF2);

    vF1 = vmulq_f32(vXInv1, vSt1);
    vF2 = vmulq_f32(vXInv2, vSt2);
    vHF1 = vmulq_f32(vXInv3, vSt3);
    vHF2 = vmulq_f32(vXInv4, vSt4);

    float32x4_t vDivTab = vdupq_n_f32(vsdiv_table);
    vSt1 = vmulq_f32(vF1, vDivTab);
    vSt2 = vmulq_f32(vF2, vDivTab);
    vDivTab = vdupq_n_f32(vhdiv_table);
    vSt3 = vmulq_f32(vHF1, vDivTab);
    vSt4 = vmulq_f32(vHF2, vDivTab);

    uint32x4_t vRes1 = internal::vroundq_u32_f32(vSt1);
    uint32x4_t vRes2 = internal::vroundq_u32_f32(vSt2);
    uint32x4_t vRes3 = internal::vroundq_u32_f32(vSt3);
    uint32x4_t vRes4 = internal::vroundq_u32_f32(vSt4);

    int32x4_t vH_L = vmovl_s16(vget_low_s16(vDiff4));
    int32x4_t vH_H = vmovl_s16(vget_high_s16(vDiff4));

    uint32x4_t vDiff_Res1 = vmulq_u32(vDiffL, vRes1);
    uint32x4_t vDiff_Res2 = vmulq_u32(vDiffH, vRes2);
    uint32x4_t vDiff_Res3 = vmulq_u32(vreinterpretq_u32_s32(vH_L), vRes3);
    uint32x4_t vDiff_Res4 = vmulq_u32(vreinterpretq_u32_s32(vH_H), vRes4);

    int32x4_t vShift = vdupq_n_s32(vshift);
    uint32x4_t vAddRes1 = vaddq_u32(vDiff_Res1, vreinterpretq_u32_s32(vShift));
    uint32x4_t vAddRes2 = vaddq_u32(vDiff_Res2, vreinterpretq_u32_s32(vShift));
    uint32x4_t vAddRes3 = vaddq_u32(vDiff_Res3, vreinterpretq_u32_s32(vShift));
    uint32x4_t vAddRes4 = vaddq_u32(vDiff_Res4, vreinterpretq_u32_s32(vShift));
    int16x4_t vShrRes1 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes1), 8);
    int16x4_t vShrRes2 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes2), 8);
    int16x4_t vShrRes3 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes3), 8);
    int16x4_t vShrRes4 = vshrn_n_s32(vreinterpretq_s32_u32(vAddRes4), 8);

    int16x8_t vc0 = vdupq_n_s16((s16)v0);
    int8x8_t vShrRes1_s8 = vshrn_n_s16(vcombine_s16(vShrRes1, vShrRes2), 4);
    uint16x8_t vCltRes_u16 = vcltq_s16(vcombine_s16(vShrRes3, vShrRes4), vc0);
    int8x8_t vShrRes2_s8 = vshrn_n_s16(vcombine_s16(vShrRes3, vShrRes4), 4);

    int8x8_t vCltRes_s8 = vmovn_s16(vreinterpretq_s16_u16(vCltRes_u16));
    int8x8_t vcHRange = vdup_n_s8((s8)vhrange);
    uint8x8_t vHResAdd = vand_u8(vreinterpret_u8_s8(vCltRes_s8), vreinterpret_u8_s8(vcHRange));
    int8x8_t vHRes = vadd_s8(vShrRes2_s8, vreinterpret_s8_u8(vHResAdd));

    uint8x8x3_t vHsv;
    vHsv.val[0] = vreinterpret_u8_s8(vHRes);
    vHsv.val[1] = vreinterpret_u8_s8(vShrRes1_s8);
    vHsv.val[2] = vMax;

    return vHsv;
}

const u8 fastSaturate8u[] =
{
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255
};

inline void convertToHSV(const s32 r, const s32 g, const s32 b,
                         const s32 &hrange, const s32 &hsv_shift,
                         u8* dst)
{
    s32 h, s, v = b;
    s32 vmin = b, diff;
    s32 vr, vg;

    v += fastSaturate8u[g-v+256];
    v += fastSaturate8u[r-v+256];
    vmin -= fastSaturate8u[vmin-g+256];
    vmin -= fastSaturate8u[vmin-r+256];

    diff = v - vmin;
    vr = v == r ? -1 : 0;
    vg = v == g ? -1 : 0;

    s = (s32(diff * (255 << hsv_shift) * (1.0f/(f32)v)) + (1 << (hsv_shift-1))) >> hsv_shift;
    h = (vr & (g - b)) + (~vr & ((vg & (b - r + 2 * diff)) + ((~vg) & (r - g + 4 * diff))));
    h = ((h * s32((hrange << hsv_shift)/(6.f*diff) + 0.5)) + (1 << (hsv_shift-1))) >> hsv_shift;
    h += h < 0 ? hrange : 0;

    dst[0] = internal::saturate_cast<u8>(h);
    dst[1] = (u8)s;
    dst[2] = (u8)v;
}

#define CONVERT_TO_HSV_ASM(loadop, rreg, breg)                        \
            __asm__ (                                                 \
               #loadop    ", [%[in]] @RGB                       \n\t" \
            "vmin.u8 d3, d0, d1      @VMin (d3)                 \n\t" \
            "vmax.u8 d6, d0, d1      @V (d6)                    \n\t" \
            "vmovl.u8 q2, " #rreg "  @V16_R (d4,d5)             \n\t" \
            "vmovl.u8 q4, d1         @V16_G (d8,d9)             \n\t" \
            "vmax.u8 d6, d6, d2                                 \n\t" \
            "vmin.u8 d3, d3, d2                                 \n\t" \
            "vmovl.u8 q0, " #breg "  @V16_B (d0,d1)             \n\t" \
            "vsubl.u8 q8, d6, d3     @V16_Diff (d16,d17)        \n\t" \
                                                                      \
            "vmovl.u8 q5, d6         @V16_V (d10,d11)           \n\t" \
            "vadd.s16 q10, q8, q8    @V16_Diff_2 (d20,d21)      \n\t" \
            "vmovl.u16 q9, d16       @V32_Diff_L (d18,d19)      \n\t" \
            "vmovl.u16 q11, d17      @V32_Diff_H (d22,d23)      \n\t" \
            "vceq.u16 q12, q2, q5    @V==R(d24,d25)             \n\t" \
            "vceq.u16 q13, q4, q5    @V==G(d26,d27)             \n\t" \
                                                                      \
            "vsub.s16 q8, q4, q0     @V16_G-B (d16,d17)         \n\t" \
            "vmvn.u16 q15, q12       @V16~R                     \n\t" \
            "vsub.s16 q6, q0, q2     @V16_B-R (d12,d13)         \n\t" \
            "vsub.s16 q7, q2, q4     @V16_R-G (d14,d15)         \n\t" \
            "vand.u16 q1, q13, q15   @VMask2                    \n\t" \
            "vand.u16 q2, q8, q12    @V16_H(d4,d5)              \n\t" \
            "vadd.s16 q4, q6, q10    @V16_H2                    \n\t" \
            "vmvn.u16 q12, q13       @V16~G                     \n\t" \
            "vadd.s16 q6, q10, q10   @VDiff16_4 (d12,d13)       \n\t" \
            "vand.u16 q8, q15, q12   @VMask3                    \n\t" \
            "vand.u16 q15, q4, q1    @vH2(d30,d31)              \n\t" \
            "vadd.s16 q7, q7, q6     @V16_H3 (d14,d15)          \n\t" \
            "vadd.s16 q14, q2, q15   @vH16                      \n\t" \
            "vmovl.u16 q12, d10      @V32_V_L                   \n\t" \
            "vand.s16 q7, q7, q8     @vH16                      \n\t" \
            "vmovl.u16 q13, d11      @V32_V_H                   \n\t" \
            "vadd.s16 q2, q14, q7    @V16_Diff_4                \n\t" \
                                                                      \
            "vdup.32 q4, %[v6]                                  \n\t" \
            "vmul.u32 q14, q9, q4                               \n\t" \
            "vmul.u32 q15, q11, q4                              \n\t" \
            "vcvt.f32.u32 q4, q12     @VF1 (d8,d9)              \n\t" \
            "vcvt.f32.u32 q8, q13     @VF2                      \n\t" \
            "vcvt.f32.u32 q0, q14     @HF1                      \n\t" \
            "vcvt.f32.u32 q1, q15     @HF2                      \n\t" \
            "vrecpe.f32 q12, q4       @Vxinv                    \n\t" \
            "vrecpe.f32 q13, q8       @Vxinv                    \n\t" \
            "vrecpe.f32 q5, q0        @Vxinv                    \n\t" \
            "vrecpe.f32 q7, q1        @Vxinv                    \n\t" \
            "vrecps.f32 q14, q12, q4  @Vst1                     \n\t" \
            "vrecps.f32 q15, q13, q8  @Vst1                     \n\t" \
            "vrecps.f32 q10, q5, q0   @Vst1                     \n\t" \
            "vrecps.f32 q6, q7, q1    @Vst1                     \n\t" \
            "vmul.f32 q4, q12, q14                              \n\t" \
            "vmul.f32 q8, q13, q15                              \n\t" \
            "vmul.f32 q0, q5, q10                               \n\t" \
            "vmul.f32 q1, q7, q6                                \n\t" \
            "vdup.32 q12, %[vsdiv_table]                        \n\t" \
            "vmul.f32 q14, q4, q12                              \n\t" \
            "vmul.f32 q15, q8, q12                              \n\t" \
            "vdup.32 q12, %[vhdiv_table]                        \n\t" \
            "vmul.f32 q10, q0, q12                              \n\t" \
            "vmul.f32 q6, q1, q12                               \n\t" \
                                                                      \
            "vdup.32 q12, %[bias]                               \n\t" \
                                                                      \
            "vadd.f32 q7, q14, q12                              \n\t" \
            "vadd.f32 q13, q15, q12                             \n\t" \
            "vcvt.u32.f32 q4, q7                                \n\t" \
            "vcvt.u32.f32 q8, q13                               \n\t" \
                                                                      \
            "vadd.f32 q14, q10, q12                             \n\t" \
            "vadd.f32 q7, q6, q12                               \n\t" \
            "vcvt.u32.f32 q0, q14                               \n\t" \
            "vcvt.u32.f32 q1, q7      @Vres                     \n\t" \
                                                                      \
            "vmovl.s16 q7, d4         @V32_H_L (d14,d15)        \n\t" \
            "vmovl.s16 q5, d5         @V32_H_H (d10,d11)        \n\t" \
            "vmul.u32 q14, q9, q4                               \n\t" \
            "vmul.u32 q15, q11, q8                              \n\t" \
            "vmul.u32 q10, q7, q0                               \n\t" \
            "vmul.u32 q6, q5, q1                                \n\t" \
                                                                      \
            "vdup.32 q12, %[vshift]                             \n\t" \
            "vadd.u32 q13, q14, q12                             \n\t" \
            "vadd.u32 q8, q15, q12                              \n\t" \
            "vadd.u32 q0, q10, q12                              \n\t" \
            "vadd.u32 q1, q6, q12                               \n\t" \
            "vshrn.s32 d8, q13, #8                              \n\t" \
            "vshrn.s32 d9, q8, #8                               \n\t" \
            "vshrn.s32 d10, q0, #8                              \n\t" \
            "vshrn.s32 d11, q1, #8                              \n\t" \
                                                                      \
            "vdup.16 q8, %[v0]                                  \n\t" \
            "vshrn.s16 d5, q4, #4                               \n\t" \
            "vclt.s16 q9, q5, q8                                \n\t" \
            "vshrn.s16 d4, q5, #4                               \n\t" \
                                                                      \
            "vmovn.s16 d9, q9                                   \n\t" \
            "vdup.8 d7, %[vhrange]                              \n\t" \
            "vand.u8 d10, d9, d7                                \n\t" \
            "vadd.s8 d4, d4, d10                                \n\t" \
            "vst3.8 {d4-d6}, [%[out]] @HSV                      \n\t" \
            : /*no output*/                                           \
            : [out] "r" (dst + dj), [in] "r" (src + sj),              \
                        [vsdiv_table] "r" (vsdiv_table),              \
                        [vshift] "r" (vshift),                        \
                        [vhdiv_table] "r" (vhdiv_table),              \
                        [v6] "r" (v6), [vhrange] "r" (vhrange),       \
                        [v0] "r" (v0), [bias] "r" (bias)              \
            : "d0","d1","d2","d3","d4","d5","d6","d7",                \
              "d8","d9","d10","d11","d12","d13","d14","d15",          \
              "d16","d17","d18","d19","d20","d21","d22","d23",        \
              "d24","d25","d26","d27","d28","d29","d30","d31"         \
            );

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)

#define YCRCB_CONSTS                                                        \
    register int16x4_t vcYR  asm ("d31") = vmov_n_s16(4899);                \
    register int16x4_t vcYG  asm ("d30") = vmov_n_s16(9617);                \
    register int16x4_t vcYB  asm ("d29") = vmov_n_s16(1868);                \
    register int16x4_t vcCrG asm ("d28") = vmov_n_s16(6860);                \
    register int16x4_t vcCrB asm ("d27") = vmov_n_s16(1332);                \
    register int16x4_t vcCbR asm ("d26") = vmov_n_s16(2765);                \
    register int16x4_t vcCbG asm ("d25") = vmov_n_s16(5427);

#else

#define YCRCB_CONSTS                                                        \
    const s16       convertCoeffs[] = { 4899, 4899, 4899, 4899,             \
                                        9617, 9617, 9617, 9617,             \
                                        1868, 1868, 1868, 1868,             \
                                        6860, 6860, 6860, 6860,             \
                                        1332, 1332, 1332, 1332,             \
                                        2765, 2765, 2765, 2765,             \
                                        5427, 5427, 5427, 5427  };          \
    const int16x8_t vcYRG  = vld1q_s16(convertCoeffs);      /*YR and YG*/   \
    const int16x4_t vcYB   = vld1_s16(convertCoeffs + 8);   /*YB*/          \
    const int16x8_t vcCrGB = vld1q_s16(convertCoeffs + 12); /*CrG and CrB*/ \
    const int16x8_t vcCbRG = vld1q_s16(convertCoeffs + 20); /*CbR and CbG*/

#endif

#define CONVERTTOYCRCB(loadcmd, rreg, greg, breg)                           \
    __asm__ (                                                               \
       #loadcmd   ", [%[in]] @RGB                       \n\t"               \
    "vmovl.u8 q2, " #rreg "  @R (d4,d5)                 \n\t"               \
    "vmovl.u8 q3, " #greg "  @G (d6,d7)                 \n\t"               \
    "vmovl.u8 q4, " #breg "  @B (d8,d9)                 \n\t"               \
                                                                            \
    "vshll.u16 q7, d4, #13   @Cr(q7,q8):  R             \n\t"               \
    "vmull.u16 q5, d6, d30   @Y (q5,q6):  G             \n\t"               \
    "vshll.u16 q9, d8, #13   @Cb(q9,q10): B             \n\t"               \
    "vshll.u16 q8, d5, #13   @Cr(q7,q8):  R             \n\t"               \
    "vmull.u16 q6, d7, d30   @Y (q5,q6):  G             \n\t"               \
    "vshll.u16 q10, d9, #13  @Cb(q9,q10): B             \n\t"               \
                                                                            \
    "vmlsl.s16 q7, d6, d28   @Cr(q7,q8):  RG            \n\t"               \
    "vmlal.s16 q5, d8, d29   @Y (q5,q6):  GB            \n\t"               \
    "vmlsl.s16 q9, d4, d26   @Cb(q9,q10): BR            \n\t"               \
    "vmlsl.s16 q8, d7, d28   @Cr(q7,q8):  RG            \n\t"               \
    "vmlal.s16 q6, d9, d29   @Y (q5,q6):  GB            \n\t"               \
    "vmlsl.s16 q10, d5, d26  @Cb(q9,q10): BR            \n\t"               \
                                                                            \
    "vmlsl.s16 q7, d8, d27   @Cr(q7,q8):  RGB           \n\t"               \
    "vmlal.s16 q5, d4, d31   @Y (q5,q6):  GBR           \n\t"               \
    "vmlsl.s16 q9, d6, d25   @Cb(q9,q10): BRG           \n\t"               \
    "vmlsl.s16 q8, d9, d27   @Cr(q7,q8):  RGB           \n\t"               \
    "vmlal.s16 q6, d5, d31   @Y (q5,q6):  GBR           \n\t"               \
    "vmlsl.s16 q10, d7, d25  @Cb(q9,q10): BRG           \n\t"               \
                                                                            \
    "vrshrn.s32 d4, q7, #14  @Cr -> q2                  \n\t"               \
    "vrshrn.s32 d8, q5, #14  @Y  -> q4                  \n\t"               \
    "vrshrn.s32 d6, q9, #14  @Cb -> q3                  \n\t"               \
    "vrshrn.s32 d5, q8, #14  @Cr -> q2                  \n\t"               \
    "vrshrn.s32 d9, q6, #14  @Y  -> q4                  \n\t"               \
    "vrshrn.s32 d7, q10, #14 @Cb -> q3                  \n\t"               \
                                                                            \
    "vmov.s16 q5, #128                                  \n\t"               \
    "vmov.s16 q6, #128                                  \n\t"               \
    "vadd.i16 q5, q2         @Cr -> q5                  \n\t"               \
    "vadd.i16 q6, q3         @Cb -> q6                  \n\t"               \
                                                                            \
    "vqmovn.u16 d4, q4                                  \n\t"               \
    "vqmovun.s16 d5, q5                                 \n\t"               \
    "vqmovun.s16 d6, q6                                 \n\t"               \
                                                                            \
    "vst3.8 {d4-d6}, [%[out]]                           \n\t"               \
    : /*no output*/                                                         \
    : [out] "r" (dst + dj), [in] "r" (src + sj),                            \
      "w" (vcYR), "w" (vcYG), "w" (vcYB),                                   \
      "w" (vcCrB), "w" (vcCrG), "w" (vcCbG), "w" (vcCbR)                    \
    : "d0","d1","d2","d3","d4","d5","d6","d7",                              \
      "d8","d9","d10","d11","d12","d13","d14","d15",                        \
      "d16","d17","d18","d19","d20","d21"                                   \
    );


inline uint8x8x3_t convertToYCrCb( const int16x8_t& vR, const int16x8_t& vG, const int16x8_t& vB,
                                   const int16x8_t& vcYRG, const int16x4_t& vcYB,
                                   const int16x8_t& vcCrGB, const int16x8_t& vcCbRG )
{
    int32x4_t vCrL = vshll_n_s16(vget_low_s16(vR), 13);                  // R
    int32x4_t vCrH = vshll_n_s16(vget_high_s16(vR), 13);                 // R
    int32x4_t vYL  = vmull_s16(vget_low_s16(vG), vget_high_s16(vcYRG));  // G
    int32x4_t vYH  = vmull_s16(vget_high_s16(vG), vget_high_s16(vcYRG)); // G
    int32x4_t vCbL = vshll_n_s16(vget_low_s16(vB), 13);                  // B
    int32x4_t vCbH = vshll_n_s16(vget_high_s16(vB), 13);                 // B

    vCrL = vmlsl_s16(vCrL, vget_low_s16(vG), vget_low_s16(vcCrGB));      // RG
    vCrH = vmlsl_s16(vCrH, vget_high_s16(vG), vget_low_s16(vcCrGB));     // RG
    vYL  = vmlal_s16(vYL, vget_low_s16(vB), vcYB);                       // GB
    vYH  = vmlal_s16(vYH, vget_high_s16(vB), vcYB);                      // GB
    vCbL = vmlsl_s16(vCbL, vget_low_s16(vR), vget_low_s16(vcCbRG));      // BR
    vCbH = vmlsl_s16(vCbH, vget_high_s16(vR), vget_low_s16(vcCbRG));     // BR

    vCrL = vmlsl_s16(vCrL, vget_low_s16(vB), vget_high_s16(vcCrGB));     // RGB
    vCrH = vmlsl_s16(vCrH, vget_high_s16(vB), vget_high_s16(vcCrGB));    // RGB
    vYL  = vmlal_s16(vYL, vget_low_s16(vR), vget_low_s16(vcYRG));        // GBR
    vYH  = vmlal_s16(vYH, vget_high_s16(vR), vget_low_s16(vcYRG));       // GBR
    vCbL = vmlsl_s16(vCbL, vget_low_s16(vG), vget_high_s16(vcCbRG));     // BRG
    vCbH = vmlsl_s16(vCbH, vget_high_s16(vG), vget_high_s16(vcCbRG));    // BRG

    int16x4_t vCrL_ = vrshrn_n_s32(vCrL, 14);
    int16x4_t vCrH_ = vrshrn_n_s32(vCrH, 14);
    int16x4_t vYL_  = vrshrn_n_s32(vYL, 14);
    int16x4_t vYH_  = vrshrn_n_s32(vYH, 14);
    int16x4_t vCbL_ = vrshrn_n_s32(vCbL, 14);
    int16x4_t vCbH_ = vrshrn_n_s32(vCbH, 14);

    int16x8_t vCr = vmovq_n_s16(128);
    int16x8_t vCb = vmovq_n_s16(128);

    vCr = vaddq_s16(vCr, vcombine_s16(vCrL_, vCrH_));
    vCb = vaddq_s16(vCb, vcombine_s16(vCbL_, vCbH_));

    uint8x8x3_t vYCrCb;
    vYCrCb.val[0] = vqmovn_u16(vreinterpretq_u16_s16(vcombine_s16(vYL_, vYH_)));
    vYCrCb.val[1] = vqmovun_s16(vCr);
    vYCrCb.val[2] = vqmovun_s16(vCb);

    return vYCrCb;
}

#define S_CONVERTTOYCRCB(R, G, B)                                           \
    s32 Y =         (R * 4899    + G * 9617 + B * 1868 + (1 << 13)) >> 14;  \
    s32 Cr = 128 + ((R * 8192    - G * 6860 - B * 1332 + (1 << 13)) >> 14); \
    s32 Cb = 128 + ((R * (-2765) - G * 5427 + B * 8192 + (1 << 13)) >> 14); \
    dst[dj + 0] = internal::saturate_cast<u8>(Y);                           \
    dst[dj + 1] = internal::saturate_cast<u8>(Cr);                          \
    dst[dj + 2] = internal::saturate_cast<u8>(Cb);

#define COEFF_Y   (   149)
#define COEFF_BU  (   129)
#define COEFF_RV  (   102)
#define COEFF_GU  (    25)
#define COEFF_GV  (    52)
#define COEFF_R   (-14248)
#define COEFF_G   (  8663)
#define COEFF_B   (-17705)

#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
#define YUV420ALPHA3_CONST
#define YUV420ALPHA4_CONST register uint8x16_t c255  asm ("q13") = vmovq_n_u8(255);
#define YUV420ALPHA3_CONVERT
#define YUV420ALPHA4_CONVERT , "w" (c255)
#define YUV420STORE1CMD3 "vst3.8 {d20, d22, d24}"
#define YUV420STORE2CMD3 "vst3.8 {d21, d23, d25}"
#define YUV420STORE1CMD4 "vst4.8 {d20, d22, d24, d26}"
#define YUV420STORE2CMD4 "vst4.8 {d21, d23, d25, d27}"

#define YUV420_CONSTS(cn, bIdx, vIdx)                            \
    register const s32 cR = s16(COEFF_R);                        \
    register const s32 cG = s16(COEFF_G);                        \
    register const s32 cB = s16(COEFF_B);                        \
                                                                 \
    register uint8x16_t vc16  asm ("q15") = vmovq_n_u8(16);      \
    register uint8x8_t cGU    asm ("d14") = vmov_n_u8(COEFF_GU); \
    register uint8x8_t cGV    asm ("d15") = vmov_n_u8(COEFF_GV); \
    register uint8x8_t cRV    asm ("d16") = vmov_n_u8(COEFF_RV); \
    register uint8x8_t cBU    asm ("d17") = vmov_n_u8(COEFF_BU); \
    register uint8x16_t cRGBY asm ("q3")  = vmovq_n_u8(COEFF_Y); \
    YUV420ALPHA##cn##_CONST

#define CONVERTYUV420TORGB(cn, ureg, vreg, rreg, breg)                                                    \
    __asm__ (                                                                                             \
        "vld2.8 {d0-d1}, [%[inUV]]                      @UV                                         \n\t" \
        "vdup.16 q4, %[cG]                              @cG                                         \n\t" \
        "vld2.8 {d2-d3}, [%[inY1]]                      @YY                                         \n\t" \
        "vdup.16 "#rreg", %[cR]                         @cR                                         \n\t" \
        "vld2.8 {d4-d5}, [%[inY2]]                      @YY                                         \n\t" \
        "vdup.16 "#breg", %[cB]                         @cB                                         \n\t" \
        "vmlsl.u8 q4, "#ureg", d14                      @cG-25u                                     \n\t" \
        "vmax.u8 q1, q15                                @max(Y,16)                                  \n\t" \
        "vmlal.u8 "#rreg", "#vreg", d16                 @cR+102*v                                   \n\t" \
        "vmlal.u8 "#breg", "#ureg", d17                 @cB+129*u                                   \n\t" \
        "vmax.u8 q2, q15                                @max(Y,16)                                  \n\t" \
        "vmlsl.u8 q4, "#vreg", d15                      @cG-25u-52v                                 \n\t" \
                                                                         /*q10,q11,q12,q13 - for output*/ \
        "vmull.u8 q9, d3, d6                            @h 149*y                                    \n\t" \
        "vmull.u8 q10, d2, d7                           @l 149*y                                    \n\t" \
        "vshr.u16 q9, #1                                @h (149*y)/2                                \n\t" \
        "vshr.u16 q10, #1                               @l (149*y)/2                                \n\t" \
                                                                                                          \
        "vhadd.s16 q0, q9, q4                           @hG ((149*y)/2 + cG - 25*u - 52*v)/2        \n\t" \
        "vhadd.s16 q12, q10, q6                         @lB ((149*y)/2 + cB + 129*u)/2              \n\t" \
        "vhadd.s16 q1, q9, q5                           @hR ((149*y)/2 + cR + 102*v)/2              \n\t" \
        "vhadd.s16 q11, q10, q4                         @lG ((149*y)/2 + cG - 25*u - 52*v)/2        \n\t" \
        "vhadd.s16 q9, q6                               @hB ((149*y)/2 + cB + 129*u)/2              \n\t" \
        "vhadd.s16 q10, q5                              @lR ((149*y)/2 + cR + 102*v)/2              \n\t" \
                                                                                                          \
        "vqrshrun.s16 d24, q12, #5                      @lB ((149*y)/2 + cB + 129*u)/2/32           \n\t" \
        "vqrshrun.s16 d22, q11, #5                      @lG ((149*y)/2 + cG - 25*u - 52*v)/2/32     \n\t" \
        "vqrshrun.s16 d20, q10, #5                      @lR ((149*y)/2 + cR + 102*v)/2/32           \n\t" \
        "vqrshrun.s16 d23, q0, #5                       @hG ((149*y)/2 + cG - 25*u - 52*v)/2/32     \n\t" \
        "vqrshrun.s16 d21, q1, #5                       @hR ((149*y)/2 + cR + 102*v)/2/32           \n\t" \
        "vqrshrun.s16 d25, q9, #5                       @hB ((149*y)/2 + cB + 129*u)/2/32           \n\t" \
                                                                                                          \
        "vzip.8 d22, d23                                @G                \n\t"                           \
        "vzip.8 d20, d21                                @R                \n\t"                           \
        "vzip.8 d24, d25                                @B                \n\t"                           \
                                                                                                          \
        YUV420STORE1CMD##cn", [%[out1]]                                \n\t"                              \
        YUV420STORE2CMD##cn", [%[out1x]]                               \n\t"                              \
                                                                                                          \
        "vmull.u8 q9, d5, d6                            @h 149*y                \n\t"                     \
        "vmull.u8 q10, d4, d7                           @l 149*y                \n\t"                     \
        "vshr.u16 q9, #1                                @h (149*y)/2            \n\t"                     \
        "vshr.u16 q10, #1                               @l (149*y)/2            \n\t"                     \
                                                                                                          \
        "vhadd.s16 q0, q9, q4                           @hG ((149*y)/2 + cG - 25*u - 52*v)/2        \n\t" \
        "vhadd.s16 q12, q10, q6                         @lB ((149*y)/2 + cB + 129*u)/2              \n\t" \
        "vhadd.s16 q1, q9, q5                           @hR ((149*y)/2 + cR + 102*v)/2              \n\t" \
        "vhadd.s16 q11, q10, q4                         @lG ((149*y)/2 + cG - 25*u - 52*v)/2        \n\t" \
        "vhadd.s16 q9, q6                               @hB ((149*y)/2 + cB + 129*u)/2              \n\t" \
        "vhadd.s16 q10, q5                              @lR ((149*y)/2 + cR + 102*v)/2              \n\t" \
                                                                                                          \
        "vqrshrun.s16 d24, q12, #5                      @lB ((149*y)/2 + cB + 129*u)/2/32           \n\t" \
        "vqrshrun.s16 d22, q11, #5                      @lG ((149*y)/2 + cG - 25*u - 52*v)/2/32     \n\t" \
        "vqrshrun.s16 d20, q10, #5                      @lR ((149*y)/2 + cR + 102*v)/2/32           \n\t" \
        "vqrshrun.s16 d23, q0, #5                       @hG ((149*y)/2 + cG - 25*u - 52*v)/2/32     \n\t" \
        "vqrshrun.s16 d21, q1, #5                       @hR ((149*y)/2 + cR + 102*v)/2/32           \n\t" \
        "vqrshrun.s16 d25, q9, #5                       @hB ((149*y)/2 + cB + 129*u)/2/32           \n\t" \
                                                                                                          \
        "vzip.8 d22, d23                                @G                \n\t"                           \
        "vzip.8 d20, d21                                @R                \n\t"                           \
        "vzip.8 d24, d25                                @B                \n\t"                           \
                                                                                                          \
        YUV420STORE1CMD##cn", [%[out2]]                                \n\t"                              \
        YUV420STORE2CMD##cn", [%[out2x]]                               \n\t"                              \
                                                                                                          \
        : /*no output*/                                                                                   \
        : [out1] "r" (dst1 + dj), [out2] "r" (dst2 + dj),                                                 \
          [out1x] "r" (dst1 + dj+cn*8), [out2x] "r" (dst2 + dj+cn*8),                                     \
          [inUV] "r" (uv+j), [inY1] "r" (y1+j), [inY2] "r" (y2+j),                                        \
          [cR] "r" (cR), [cG] "r" (cG), [cB] "r" (cB),                                                    \
          "w" (vc16), "w" (cGU), "w" (cGV), "w" (cBU), "w" (cRV), "w" (cRGBY) YUV420ALPHA##cn##_CONVERT   \
        : "d0","d1","d2","d3","d4","d5","d8","d9","d10","d11","d12",                                      \
          "d13","d18","d19","d20","d21","d22","d23","d24","d25"                                           \
    );

#else

template<int bIdx>
struct _convertYUV420Internals
{
    uint16x8_t vc14216;
    uint16x8_t vc17672;
    uint16x8_t vc8696;
    uint8x8_t  vc102;
    uint8x8_t  vc25;
    uint8x8_t  vc129;
    uint8x8_t  vc52;
    uint16x8_t vc_1;
    uint8x8_t  vc149;
    uint8x8_t  vc16;
    _convertYUV420Internals()
    {
        vc14216 = vdupq_n_u16(-COEFF_R);
        vc17672 = vdupq_n_u16(-COEFF_B);
        vc8696  = vdupq_n_u16(COEFF_G);
        vc102   = vdup_n_u8(COEFF_RV);
        vc25    = vdup_n_u8(COEFF_GU);
        vc129   = vdup_n_u8(COEFF_BU);
        vc52    = vdup_n_u8(COEFF_GV);
        vc_1    = vdupq_n_u16((uint16_t)-1);
        vc149   = vdup_n_u8(COEFF_Y);
        vc16    = vdup_n_u8(16);
    }

    inline void UVrgbToRGB( const int16x8_t &ruv, const int16x8_t &guv, const int16x8_t &buv,
                            const u8 *y, uint8x16x3_t &rgbl )
    {
        //y get line
        uint8x8x2_t yl = vld2_u8(y);
        yl.val[0] = vmax_u8(yl.val[0], vc16);
        yl.val[1] = vmax_u8(yl.val[1], vc16);

        //y part line
        uint16x8_t yodd1 = vmlal_u8(vc_1, yl.val[0], vc149); //(-1+149*y)
        uint16x8_t yevn1 = vmlal_u8(vc_1, yl.val[1], vc149); //(-1+149*y)
        int16x8_t yodd1h = (int16x8_t)vshrq_n_u16(yodd1, 1);  //(-1+149*y)/2
        int16x8_t yevn1h = (int16x8_t)vshrq_n_u16(yevn1, 1);  //(-1+149*y)/2

        //y line calc rgb
        int16x8_t rodd1w = vhsubq_s16(yodd1h, ruv); //((-1+149*y)/2 - (14216-102*v))/2
        int16x8_t gevn1w = vhaddq_s16(yevn1h, guv); //((-1+149*y)/2 + ((8696-25*u)-52*v))/2
        int16x8_t bodd1w = vhsubq_s16(yodd1h, buv); //((-1+149*y)/2 - (17672-129*u))/2
        int16x8_t revn1w = vhsubq_s16(yevn1h, ruv); //((-1+149*y)/2 - (14216-102*v))/2
        int16x8_t godd1w = vhaddq_s16(yodd1h, guv); //((-1+149*y)/2 + ((8696-25*u)-52*v))/2
        int16x8_t bevn1w = vhsubq_s16(yevn1h, buv); //((-1+149*y)/2 - (17672-129*u))/2

        //y line clamp + narrow
        uint8x8_t rodd1n = vqshrun_n_s16(rodd1w, 5);
        uint8x8_t revn1n = vqshrun_n_s16(revn1w, 5);
        uint8x8_t godd1n = vqshrun_n_s16(godd1w, 5);
        uint8x8x2_t r1 = vzip_u8 (rodd1n, revn1n);
        uint8x8_t gevn1n = vqshrun_n_s16(gevn1w, 5);
        uint8x8_t bodd1n = vqshrun_n_s16(bodd1w, 5);
        uint8x8x2_t g1 = vzip_u8 (godd1n, gevn1n);
        uint8x8_t bevn1n = vqshrun_n_s16(bevn1w, 5);
        uint8x8x2_t b1 = vzip_u8 (bodd1n, bevn1n);
        rgbl.val[2 - bIdx] = vcombine_u8(r1.val[0], r1.val[1]);
        rgbl.val[1]        = vcombine_u8(g1.val[0], g1.val[1]);
        rgbl.val[0 + bIdx] = vcombine_u8(b1.val[0], b1.val[1]);
    }
};

template<int cn, int bIdx, int vIdx>
struct _convertYUV420
{
    _convertYUV420Internals<bIdx> convertYUV420Internals;

    inline void ToRGB( const u8 *y1, const u8 *y2, const u8 *uv,
                       u8 *dst1, u8 *dst2 )
    {
        uint8x8x2_t raw_uv = vld2_u8(uv);
        uint16x8_t gu =            vmlsl_u8(convertYUV420Internals.vc8696,  raw_uv.val[1-vIdx], convertYUV420Internals.vc25);  //(8696-25*u)
        int16x8_t ruv = (int16x8_t)vmlsl_u8(convertYUV420Internals.vc14216, raw_uv.val[vIdx], convertYUV420Internals.vc102); //(14216-102*v)

        int16x8_t buv = (int16x8_t)vmlsl_u8(convertYUV420Internals.vc17672, raw_uv.val[1-vIdx], convertYUV420Internals.vc129); //(17672-129*u)
        int16x8_t guv = (int16x8_t)vmlsl_u8(gu,      raw_uv.val[vIdx], convertYUV420Internals.vc52);  //((8696-25*u)-52*v))

        uint8x16x3_t rgbl;
        //y line1
        convertYUV420Internals.UVrgbToRGB(ruv, guv, buv, y1, rgbl);
        vst3q_u8(dst1, rgbl);
        //y line2
        convertYUV420Internals.UVrgbToRGB(ruv, guv, buv, y2, rgbl);
        vst3q_u8(dst2, rgbl);
    }
};

template<int bIdx, int vIdx>
struct _convertYUV420<4, bIdx, vIdx>
{
    _convertYUV420Internals<bIdx> convertYUV420Internals;

    inline void ToRGB( const u8 *y1, const u8 *y2, const u8 *uv,
                       u8 *dst1, u8 *dst2 )
    {
        uint8x8x2_t raw_uv = vld2_u8(uv);
        uint16x8_t gu =            vmlsl_u8(convertYUV420Internals.vc8696,  raw_uv.val[1-vIdx], convertYUV420Internals.vc25);  //(8696-25*u)
        int16x8_t ruv = (int16x8_t)vmlsl_u8(convertYUV420Internals.vc14216, raw_uv.val[vIdx], convertYUV420Internals.vc102); //(14216-102*v)

        int16x8_t buv = (int16x8_t)vmlsl_u8(convertYUV420Internals.vc17672, raw_uv.val[1-vIdx], convertYUV420Internals.vc129); //(17672-129*u)
        int16x8_t guv = (int16x8_t)vmlsl_u8(gu,      raw_uv.val[vIdx], convertYUV420Internals.vc52);  //((8696-25*u)-52*v))

        union { uint8x16x4_t v4; uint8x16x3_t v3; } rgbl;
        rgbl.v4.val[3] = vdupq_n_u8(0xff);
        //y line1
        convertYUV420Internals.UVrgbToRGB(ruv, guv, buv, y1, rgbl.v3);
        vst4q_u8(dst1, rgbl.v4);
        //y line2
        convertYUV420Internals.UVrgbToRGB(ruv, guv, buv, y2, rgbl.v3);
        vst4q_u8(dst2, rgbl.v4);
    }
};

#define YUV420_CONSTS(cn, bIdx, vIdx) _convertYUV420<cn, bIdx, vIdx> convertYUV420;

#endif

template <int cn> inline void fillAlpha(u8 *, u8 *){}
template <> inline void fillAlpha<4>(u8 *dst1, u8 *dst2)
{
    dst1[3] = 255;
    dst1[7] = 255;
    dst2[3] = 255;
    dst2[7] = 255;
}
template <int cn, int bIdx, int vIdx>
inline void convertYUV420ToRGB(const u8 *y1, const u8 *y2, const u8 *uv, u8* dst1, u8 *dst2)
{
    int Y11 = y1[0];
    int Y12 = y1[1];
    int Y21 = y2[0];
    int Y22 = y2[1];

    int U = uv[1 - vIdx];
    int V = uv[vIdx];

    int y11 = (COEFF_Y * std::max(16, Y11)) >> 1;
    int y12 = (COEFF_Y * std::max(16, Y12)) >> 1;
    int y21 = (COEFF_Y * std::max(16, Y21)) >> 1;
    int y22 = (COEFF_Y * std::max(16, Y22)) >> 1;

    int uvR = COEFF_R +                COEFF_RV * V;
    int uvG = COEFF_G - COEFF_GU * U - COEFF_GV * V;
    int uvB = COEFF_B + COEFF_BU * U;

    dst1[2-bIdx] = internal::saturate_cast<u8>((((y11 + uvR) >> 1) + (1 << 4)) >> 5);
    dst1[1] = internal::saturate_cast<u8>((((y11 + uvG) >> 1) + (1 << 4)) >> 5);
    dst1[bIdx] = internal::saturate_cast<u8>((((y11 + uvB) >> 1) + (1 << 4)) >> 5);

    dst1[cn+2-bIdx] = internal::saturate_cast<u8>((((y12 + uvR) >> 1) + (1 << 4)) >> 5);
    dst1[cn+1] = internal::saturate_cast<u8>((((y12 + uvG) >> 1) + (1 << 4)) >> 5);
    dst1[cn+bIdx] = internal::saturate_cast<u8>((((y12 + uvB) >> 1) + (1 << 4)) >> 5);

    dst2[2-bIdx] = internal::saturate_cast<u8>((((y21 + uvR) >> 1) + (1 << 4)) >> 5);
    dst2[1] = internal::saturate_cast<u8>((((y21 + uvG) >> 1) + (1 << 4)) >> 5);
    dst2[bIdx] = internal::saturate_cast<u8>((((y21 + uvB) >> 1) + (1 << 4)) >> 5);

    dst2[cn+2-bIdx] = internal::saturate_cast<u8>((((y22 + uvR) >> 1) + (1 << 4)) >> 5);
    dst2[cn+1] = internal::saturate_cast<u8>((((y22 + uvG) >> 1) + (1 << 4)) >> 5);
    dst2[cn+bIdx] = internal::saturate_cast<u8>((((y22 + uvB) >> 1) + (1 << 4)) >> 5);

    fillAlpha<cn>(dst1, dst2);
}

// converts R, G, B (B, G, R) pixels to  RGB(BGR)565 format respectively
inline uint8x16x2_t convertTo565( const uint8x16_t& vR, const uint8x16_t& vG, const uint8x16_t& vB )
{
    uint8x16x2_t vRgb565;                               // rrrrRRRR ggggGGGG bbbbBBBB

    vRgb565.val[1] = vsriq_n_u8(vB, vG, 5);             // xxxxxxxx bbbbBggg
    vRgb565.val[0] = vshlq_n_u8(vG, 3);                 // gGGGG000 bbbbBggg
    vRgb565.val[0] = vsriq_n_u8(vRgb565.val[0], vR, 3); // gGGrrrrR bbbbBggg

    return vRgb565;
}
inline void convertTo565( const u16 R, const u16 G, const u16 B, u8 * dst )
{
    *((u16*)dst) = (R >> 3)|((G&~3) << 3)|((B&~7) << 8);
}
#endif

} //namespace

void rgb2hsv(const Size2D &size,
             const u8 * srcBase, ptrdiff_t srcStride,
             u8 * dstBase, ptrdiff_t dstStride,
             s32 hrange)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    const s32 hsv_shift = 12;
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register const f32 vsdiv_table = f32(255 << hsv_shift);
    register f32 vhdiv_table = f32(hrange << hsv_shift);
    register const s32 vhrange = hrange;
    register const s32 v0 = s32(0);
    register const s32 vshift = s32(1 << (hsv_shift-1));
    register const s32 v6 = s32(6);
    register const f32 bias = 0.5f;
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 24, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERT_TO_HSV_ASM(vld3.8 {d0-d2}, d0, d2)
#else
            uint8x8x3_t vRgb = vld3_u8(src + sj);
            uint8x8x3_t vHsv = convertToHSV(vRgb.val[0], vRgb.val[1], vRgb.val[2], hrange);
            vst3_u8(dst + dj, vHsv);
#endif
        }

        for (; j < size.width; ++j, sj += 3, dj += 3)
        {
            convertToHSV(src[sj], src[sj+1], src[sj+2], hrange, hsv_shift, dst+dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)hrange;
#endif
}

void rgbx2hsv(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              s32 hrange)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    const s32 hsv_shift = 12;
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register const f32 vsdiv_table = f32(255 << hsv_shift);
    register f32 vhdiv_table = f32(hrange << hsv_shift);
    register const s32 vhrange = hrange;
    register const s32 v0 = s32(0);
    register const s32 vshift = s32(1 << (hsv_shift-1));
    register const s32 v6 = s32(6);
    register const f32 bias = 0.5f;
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 32, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERT_TO_HSV_ASM(vld4.8 {d0-d3}, d0, d2)
#else
            uint8x8x4_t vRgb = vld4_u8(src + sj);
            uint8x8x3_t vHsv = convertToHSV(vRgb.val[0], vRgb.val[1], vRgb.val[2], hrange);
            vst3_u8(dst + dj, vHsv);
#endif
        }

        for (; j < size.width; ++j, sj += 4, dj += 3)
        {
            convertToHSV(src[sj], src[sj+1], src[sj+2], hrange, hsv_shift, dst+dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)hrange;
#endif
}

void bgr2hsv(const Size2D &size,
             const u8 * srcBase, ptrdiff_t srcStride,
             u8 * dstBase, ptrdiff_t dstStride,
             s32 hrange)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    const s32 hsv_shift = 12;
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register const f32 vsdiv_table = f32(255 << hsv_shift);
    register f32 vhdiv_table = f32(hrange << hsv_shift);
    register const s32 vhrange = hrange;
    register const s32 v0 = s32(0);
    register const s32 vshift = s32(1 << (hsv_shift-1));
    register const s32 v6 = s32(6);
    register const f32 bias = 0.5f;
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 24, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERT_TO_HSV_ASM(vld3.8 {d0-d2}, d2, d0)
#else
            uint8x8x3_t vRgb = vld3_u8(src + sj);
            uint8x8x3_t vHsv = convertToHSV(vRgb.val[2], vRgb.val[1], vRgb.val[0], hrange);
            vst3_u8(dst + dj, vHsv);
#endif
        }

        for (; j < size.width; ++j, sj += 3, dj += 3)
        {
            convertToHSV(src[sj+2], src[sj+1], src[sj], hrange, hsv_shift, dst+dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)hrange;
#endif
}

void bgrx2hsv(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              s32 hrange)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;
    const s32 hsv_shift = 12;
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
    register const f32 vsdiv_table = f32(255 << hsv_shift);
    register f32 vhdiv_table = f32(hrange << hsv_shift);
    register const s32 vhrange = hrange;
    register const s32 v0 = s32(0);
    register const s32 vshift = s32(1 << (hsv_shift-1));
    register const s32 v6 = s32(6);
    register const f32 bias = 0.5f;
#endif

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 32, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERT_TO_HSV_ASM(vld4.8 {d0-d3}, d2, d0)
#else
            uint8x8x4_t vRgb = vld4_u8(src + sj);
            uint8x8x3_t vHsv = convertToHSV(vRgb.val[2], vRgb.val[1], vRgb.val[0], hrange);
            vst3_u8(dst + dj, vHsv);
#endif
        }

        for (; j < size.width; ++j, sj += 4, dj += 3)
        {
            convertToHSV(src[sj+2], src[sj+1], src[sj], hrange, hsv_shift, dst+dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)hrange;
#endif
}

void rgbx2bgr565(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw16; sj += 64, dj += 32, j += 16)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
                "vld4.8 {d2, d4, d6, d8}, [%[in0]]        @  q0       q1       q2       q3       q4       \n\t"
                "vld4.8 {d3, d5, d7, d9}, [%[in1]]        @  xxxxxxxx rrrrRRRR ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vsri.8 q1, q2, #5                        @  xxxxxxxx rrrrRggg ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vshl.u8 q0, q2, #3                       @  gGGGG000 rrrrRggg ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vsri.8 q0, q3, #3                        @  gGGbbbbB rrrrRggg ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vst2.8 {d0, d2}, [%[out0]]                                                               \n\t"
                "vst2.8 {d1, d3}, [%[out1]]                                                               \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [out1] "r" (dst + dj + 16),
                  [in0]  "r" (src + sj),
                  [in1]  "r" (src + sj + 32)
                : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
            );
#else
            uint8x16x4_t vRgba = vld4q_u8(src + sj);
            uint8x16x2_t vVal565 = convertTo565(vRgba.val[2], vRgba.val[1], vRgba.val[0]);
            vst2q_u8(dst + dj, vVal565);
#endif
        }

        for (; j < size.width; ++j, sj += 4, dj += 2)
        {
            convertTo565(src[sj + 2], src[sj + 1], src[sj], dst + dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgb2bgr565(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw16; sj += 48, dj += 32, j += 16)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
                "vld3.8 {d2, d4, d6}, [%[in0]]       @  q0       q1       q2       q3       q4       \n\t"
                "vld3.8 {d3, d5, d7}, [%[in1]]       @  xxxxxxxx rrrrRRRR ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vsri.8 q1, q2, #5                   @  xxxxxxxx rrrrRggg ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vshl.u8 q0, q2, #3                  @  gGGGG000 rrrrRggg ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vsri.8 q0, q3, #3                   @  gGGbbbbB rrrrRggg ggggGGGG bbbbBBBB xxxxxxxx \n\t"
                "vst2.8 {d0, d2}, [%[out0]]                                                          \n\t"
                "vst2.8 {d1, d3}, [%[out1]]                                                          \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [out1] "r" (dst + dj + 16),
                  [in0]  "r" (src + sj),
                  [in1]  "r" (src + sj + 24)
                : "d0","d1","d2","d3","d4","d5","d6","d7"
            );
#else
            uint8x16x3_t vRgba = vld3q_u8(src + sj);
            uint8x16x2_t vVal565 = convertTo565(vRgba.val[2], vRgba.val[1], vRgba.val[0]);
            vst2q_u8(dst + dj, vVal565);
#endif
        }

        for (; j < size.width; ++j, sj += 3, dj += 2)
        {
            convertTo565(src[sj + 2], src[sj + 1], src[sj], dst + dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgbx2rgb565(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw16; sj += 64, dj += 32, j += 16)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
                "vld4.8 {d0, d2, d4, d6}, [%[in0]]    @  q0       q1       q2       q3         \n\t"
                "vld4.8 {d1, d3, d5, d7}, [%[in1]]    @  rrrrRRRR ggggGGGG bbbbBBBB aaaaAAAA   \n\t"
                "vsri.8 q2, q1, #5                    @  rrrrRRRR ggggGGGG bbbbBggg aaaaAAAA   \n\t"
                "vshl.u8 q1, #3                       @  rrrrRRRR gGGGG000 bbbbBggg aaaaAAAA   \n\t"
                "vsri.8 q1, q0, #3                    @  rrrrRRRR gGGrrrrR bbbbBggg aaaaAAAA   \n\t"
                "vst2.8 {d2, d4}, [%[out0]]                                                    \n\t"
                "vst2.8 {d3, d5}, [%[out1]]                                                    \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [out1] "r" (dst + dj + 16),
                  [in0]  "r" (src + sj),
                  [in1]  "r" (src + sj + 32)
                : "d0","d1","d2","d3","d4","d5","d6","d7"
            );
#else
            uint8x16x4_t vRgba = vld4q_u8(src + sj);
            uint8x16x2_t vVal565 = convertTo565(vRgba.val[0], vRgba.val[1], vRgba.val[2]);
            vst2q_u8(dst + dj, vVal565);
#endif
        }

        for (; j < size.width; ++j, sj += 4, dj += 2)
        {
            convertTo565(src[sj], src[sj + 1], src[sj + 2], dst + dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgb2rgb565(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw16; sj += 48, dj += 32, j += 16)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            __asm__ (
                "vld3.8 {d0, d2, d4}, [%[in0]]        @  q0       q1       q2       q3         \n\t"
                "vld3.8 {d1, d3, d5}, [%[in1]]        @  rrrrRRRR ggggGGGG bbbbBBBB xxxxxxxx   \n\t"
                "vsri.8 q2, q1, #5                    @  rrrrRRRR ggggGGGG bbbbBggg xxxxxxxx   \n\t"
                "vshl.u8 q1, #3                       @  rrrrRRRR gGGGG000 bbbbBggg xxxxxxxx   \n\t"
                "vsri.8 q1, q0, #3                    @  rrrrRRRR gGGrrrrR bbbbBggg xxxxxxxx   \n\t"
                "vst2.8 {d2, d4}, [%[out0]]                                                    \n\t"
                "vst2.8 {d3, d5}, [%[out1]]                                                    \n\t"
                : /*no output*/
                : [out0] "r" (dst + dj),
                  [out1] "r" (dst + dj + 16),
                  [in0]  "r" (src + sj),
                  [in1]  "r" (src + sj + 24)
                : "d0","d1","d2","d3","d4","d5"
            );
#else
            uint8x16x3_t vRgba = vld3q_u8(src + sj);
            uint8x16x2_t vVal565 = convertTo565(vRgba.val[0], vRgba.val[1], vRgba.val[2]);
            vst2q_u8(dst + dj, vVal565);
#endif
        }

        for (; j < size.width; ++j, sj += 3, dj += 2)
        {
            convertTo565(src[sj], src[sj + 1], src[sj + 2], dst + dj);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgb2ycrcb(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YCRCB_CONSTS
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 24, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTTOYCRCB(vld3.8 {d0-d2}, d0, d1, d2)
#else
            uint8x8x3_t vRgb = vld3_u8(src + sj);
            int16x8_t vR = vreinterpretq_s16_u16(vmovl_u8(vRgb.val[0]));
            int16x8_t vG = vreinterpretq_s16_u16(vmovl_u8(vRgb.val[1]));
            int16x8_t vB = vreinterpretq_s16_u16(vmovl_u8(vRgb.val[2]));
            uint8x8x3_t vYCrCb = convertToYCrCb(vR, vG, vB, vcYRG, vcYB, vcCrGB, vcCbRG);
            vst3_u8(dst + dj, vYCrCb);
#endif
        }

        for (; j < size.width; ++j, sj += 3, dj += 3)
        {
            S_CONVERTTOYCRCB(src[sj], src[sj + 1], src[sj + 2]);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void rgbx2ycrcb(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YCRCB_CONSTS
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 32, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTTOYCRCB(vld4.8 {d0-d3}, d0, d1, d2)
#else
            uint8x8x4_t vRgba = vld4_u8(src + sj);
            int16x8_t vR = vreinterpretq_s16_u16(vmovl_u8(vRgba.val[0]));
            int16x8_t vG = vreinterpretq_s16_u16(vmovl_u8(vRgba.val[1]));
            int16x8_t vB = vreinterpretq_s16_u16(vmovl_u8(vRgba.val[2]));
            uint8x8x3_t vYCrCb = convertToYCrCb(vR, vG, vB, vcYRG, vcYB, vcCrGB, vcCbRG);
            vst3_u8(dst + dj, vYCrCb);
#endif
        }

        for (; j < size.width; ++j, sj += 4, dj += 3)
        {
            S_CONVERTTOYCRCB(src[sj], src[sj + 1], src[sj + 2]);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bgr2ycrcb(const Size2D &size,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YCRCB_CONSTS
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 24, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTTOYCRCB(vld3.8 {d0-d2}, d2, d1, d0)
#else
            uint8x8x3_t vBgr = vld3_u8(src + sj);
            int16x8_t vB = vreinterpretq_s16_u16(vmovl_u8(vBgr.val[0]));
            int16x8_t vG = vreinterpretq_s16_u16(vmovl_u8(vBgr.val[1]));
            int16x8_t vR = vreinterpretq_s16_u16(vmovl_u8(vBgr.val[2]));
            uint8x8x3_t vYCrCb = convertToYCrCb(vR, vG, vB, vcYRG, vcYB, vcCrGB, vcCbRG);
            vst3_u8(dst + dj, vYCrCb);
#endif
        }

        for (; j < size.width; ++j, sj += 3, dj += 3)
        {
            S_CONVERTTOYCRCB(src[sj + 2], src[sj + 1], src[sj]);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void bgrx2ycrcb(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YCRCB_CONSTS
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u, j = 0u;

        for (; j < roiw8; sj += 32, dj += 24, j += 8)
        {
            internal::prefetch(src + sj);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTTOYCRCB(vld4.8 {d0-d3}, d2, d1, d0)
#else
            uint8x8x4_t vBgra = vld4_u8(src + sj);
            int16x8_t vB = vreinterpretq_s16_u16(vmovl_u8(vBgra.val[0]));
            int16x8_t vG = vreinterpretq_s16_u16(vmovl_u8(vBgra.val[1]));
            int16x8_t vR = vreinterpretq_s16_u16(vmovl_u8(vBgra.val[2]));
            uint8x8x3_t vYCrCb = convertToYCrCb(vR, vG, vB, vcYRG, vcYB, vcCrGB, vcCbRG);
            vst3_u8(dst + dj, vYCrCb);
#endif
        }

        for (; j < size.width; ++j, sj += 4, dj += 3)
        {
            S_CONVERTTOYCRCB(src[sj + 2], src[sj + 1], src[sj]);
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420sp2rgb(const Size2D &size,
                  const u8 *  yBase, ptrdiff_t  yStride,
                  const u8 * uvBase, ptrdiff_t uvStride,
                  u8 * dstBase, ptrdiff_t dstStride)
{
    // input data:
    ////////////// Y matrix:
    // {y1, y2,   y3, y4,   y5, y6,   y7, y8,   y9, y10, y11, y12, y13, y14, y15, y16}
    // {Y1, Y2,   Y3, Y4,   Y5, Y6,   Y7, Y8,   Y9, Y10, Y11, Y12, Y13, Y14, Y15, Y16}
    ////////////// UV matrix:
    // {v12, u12, v34, u34, v56, u56, v78, u78, v90 u90, V12, U12, V34, U34, V56, U56}

    // fp version
    // R = 1.164(Y - 16) + 1.596(V - 128)
    // G = 1.164(Y - 16) - 0.813(V - 128) - 0.391(U - 128)
    // B = 1.164(Y - 16)                  + 2.018(U - 128)

    // integer version
    // R = [((149*y)/2 + (-14248+102*v)      )/2]/32
    // G = [((149*y)/2 + ((8663- 25*u)-52*v))/2]/32
    // B = [((149*y)/2 + (-17705+129*u)      )/2]/32

    // error estimation:
    //Rerr = 0.0000625 * y - 0.00225 * v                - 0.287
    //Gerr = 0.0000625 * y + 0.0005  * v + 0.000375 * u + 0.128625
    //Berr = 0.0000625 * y               - 0.002375 * u - 0.287375

    //real error test:
    //=================
    //R: 1 less: 520960       ==  3.11% of full space
    //G: 1 less: 251425       ==  1.50% of full space
    //B: 1 less: 455424       ==  2.71% of full space
    //=================
    //R: 1 more: 642048       ==  3.83% of full space
    //G: 1 more: 192458       ==  1.15% of full space
    //B: 1 more: 445184       ==  2.65% of full space

    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(3, 2, 0)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 48, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(3, d1, d0, q5, q6)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 6)
        {
            convertYUV420ToRGB<3, 2, 0>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420sp2rgbx(const Size2D &size,
                   const u8 *  yBase, ptrdiff_t  yStride,
                   const u8 * uvBase, ptrdiff_t uvStride,
                   u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(4, 2, 0)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 64, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(4, d1, d0, q5, q6)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 8)
        {
            convertYUV420ToRGB<4, 2, 0>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420i2rgb(const Size2D &size,
                 const u8 *  yBase, ptrdiff_t  yStride,
                 const u8 * uvBase, ptrdiff_t uvStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(3, 2, 1)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 48, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(3, d0, d1, q5, q6)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 6)
        {
            convertYUV420ToRGB<3, 2, 1>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420i2rgbx(const Size2D &size,
                  const u8 *  yBase, ptrdiff_t  yStride,
                  const u8 * uvBase, ptrdiff_t uvStride,
                  u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(4, 2, 1)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 64, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(4, d0, d1, q5, q6)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 8)
        {
            convertYUV420ToRGB<4, 2, 1>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420sp2bgr(const Size2D &size,
                  const u8 *  yBase, ptrdiff_t  yStride,
                  const u8 * uvBase, ptrdiff_t uvStride,
                  u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(3, 0, 0)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 48, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(3, d1, d0, q6, q5)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 6)
        {
            convertYUV420ToRGB<3, 0, 0>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420sp2bgrx(const Size2D &size,
                   const u8 *  yBase, ptrdiff_t  yStride,
                   const u8 * uvBase, ptrdiff_t uvStride,
                   u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(4, 0, 0)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 64, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(4, d1, d0, q6, q5)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 8)
        {
            convertYUV420ToRGB<4, 0, 0>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420i2bgr(const Size2D &size,
                 const u8 *  yBase, ptrdiff_t  yStride,
                 const u8 * uvBase, ptrdiff_t uvStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(3, 0, 1)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 48, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(3, d0, d1, q6, q5)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 6)
        {
            convertYUV420ToRGB<3, 0, 1>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void yuv420i2bgrx(const Size2D &size,
                  const u8 *  yBase, ptrdiff_t  yStride,
                  const u8 * uvBase, ptrdiff_t uvStride,
                  u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    YUV420_CONSTS(4, 0, 1)
    size_t roiw16 = size.width >= 15 ? size.width - 15 : 0;

    for (size_t i = 0u; i < size.height; i+=2)
    {
        const u8 * uv = internal::getRowPtr(uvBase, uvStride, i>>1);
        const u8 * y1 = internal::getRowPtr(yBase, yStride, i);
        const u8 * y2 = internal::getRowPtr(yBase, yStride, i+1);
        u8 * dst1 = internal::getRowPtr(dstBase, dstStride, i);
        u8 * dst2 = internal::getRowPtr(dstBase, dstStride, i+1);

        size_t dj = 0u, j = 0u;
        for (; j < roiw16; dj += 64, j += 16)
        {
            internal::prefetch(uv + j);
            internal::prefetch(y1 + j);
            internal::prefetch(y2 + j);
#if !defined(__aarch64__) && defined(__GNUC__) && __GNUC__ == 4 &&  __GNUC_MINOR__ < 7 && !defined(__clang__)
            CONVERTYUV420TORGB(4, d0, d1, q6, q5)
#else
            convertYUV420.ToRGB(y1 + j, y2 + j, uv + j, dst1 + dj, dst2 + dj);
#endif
        }
        for (; j + 2 <= size.width; j+=2, dj += 8)
        {
            convertYUV420ToRGB<4, 0, 1>(y1+j, y2+j, uv+j, dst1 + dj, dst2 + dj);
        }
    }
#else
    (void)size;
    (void)yBase;
    (void)yStride;
    (void)uvBase;
    (void)uvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

} // namespace CAROTENE_NS
