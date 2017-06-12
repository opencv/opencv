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

namespace CAROTENE_NS {

#ifdef CAROTENE_NEON

#define CVT_FUNC(T1, T2, SIMD_SIZE, CVTINIT, CVTROW)                            \
    void convert(const Size2D &_size,                                           \
                 const T1 * srcBase, ptrdiff_t srcStride,                       \
                 T2 * dstBase, ptrdiff_t dstStride)                             \
    {                                                                           \
        internal::assertSupportedConfiguration();                               \
        Size2D size(_size);                                                     \
        if (srcStride == dstStride &&                                           \
            srcStride == (ptrdiff_t)(size.width))                               \
        {                                                                       \
            size.width *= size.height;                                          \
            size.height = 1;                                                    \
        }                                                                       \
        const ptrdiff_t sstep = srcStride / sizeof(T1);                         \
        const ptrdiff_t dstep = dstStride / sizeof(T2);                         \
        const size_t w = size.width & ~(SIMD_SIZE-1);                           \
        if (size.width >= SIMD_SIZE)                                            \
        {                                                                       \
            const T1* _src = srcBase;                                           \
            T2* _dst = dstBase;                                                 \
            CVTINIT                                                             \
            for (ptrdiff_t h = size.height; h--; _src += sstep, _dst += dstep ) \
                CVTROW                                                          \
        }                                                                       \
        if(w < size.width)                                                      \
        {                                                                       \
            const T1* _src = srcBase;                                           \
            T2* _dst = dstBase;                                                 \
            for (ptrdiff_t h = size.height; h--; _src += sstep, _dst += dstep ) \
                for(size_t i = w; i < size.width; i++ )                         \
                    _dst[i] = internal::saturate_cast<T2>(_src[i]);             \
        }                                                                       \
    }

#else

#define CVT_FUNC(T1, T2, SIMD_SIZE, CVTINIT, CVTROW)                            \
    void convert(const Size2D &,                                                \
                 const T1 *, ptrdiff_t,                                         \
                 T2 *, ptrdiff_t)                                               \
    {                                                                           \
        internal::assertSupportedConfiguration();                               \
    }

#endif

CVT_FUNC(u8, s8, 16,
     uint8x16_t v127 = vdupq_n_u8(127);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         uint8x16_t vu8 = vld1q_u8(_src + i);
         int8x16_t vu1 = vreinterpretq_s8_u8(vminq_u8(vu8, v127));
         vst1q_s8(_dst + i, vu1);
     }
})

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(u8, u16, 16,
     register uint8x16_t zero0 asm ("q1") = vmovq_n_u8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vst2.8 {d0,d2}, [%[dst1]]                             \n\t"
             "vst2.8 {d1,d3}, [%[dst2]]                             \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 8),
               "w" (zero0)
             : "d0","d1"
         );
     }
})
#else
CVT_FUNC(u8, u16, 16,
     uint8x16x2_t vline;
     vline.val[1] = vmovq_n_u8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         vline.val[0] = vld1q_u8(_src + i);
         vst2q_u8((uint8_t*)(_dst + i), vline);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(u8, s32, 16,
     register uint8x16_t zero0 asm ("q1") = vmovq_n_u8(0);
     register uint8x16_t zero1 asm ("q2") = vmovq_n_u8(0);
     register uint8x16_t zero2 asm ("q3") = vmovq_n_u8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vst4.8 {d0,d2,d4,d6}, [%[dst1]]                       \n\t"
             "vst4.8 {d1,d3,d5,d7}, [%[dst2]]                       \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 8),
               "w" (zero0), "w" (zero1), "w" (zero2)
             : "d0","d1"
         );
     }
})
#else
CVT_FUNC(u8, s32, 16,
     uint8x16x4_t vline;
     vline.val[1] = vmovq_n_u8(0);
     vline.val[2] = vmovq_n_u8(0);
     vline.val[3] = vmovq_n_u8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
        vline.val[0] = vld1q_u8(_src + i);
        vst4q_u8((uint8_t*)(_dst + i), vline);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(u8, f32, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.u8 q1, d0                                       \n\t"
             "vmovl.u8 q2, d1                                       \n\t"
             "vmovl.u16 q3, d2                                      \n\t"
             "vmovl.u16 q4, d3                                      \n\t"
             "vmovl.u16 q5, d4                                      \n\t"
             "vmovl.u16 q6, d5                                      \n\t"
             "vcvt.f32.u32 q7, q3                                   \n\t"
             "vcvt.f32.u32 q8, q4                                   \n\t"
             "vcvt.f32.u32 q9, q5                                   \n\t"
             "vcvt.f32.u32 q10, q6                                  \n\t"
             "vst1.32 {d14-d15}, [%[dst1]]                          \n\t"
             "vst1.32 {d16-d17}, [%[dst2]]                          \n\t"
             "vst1.32 {d18-d19}, [%[dst3]]                          \n\t"
             "vst1.32 {d20-d21}, [%[dst4]]                          \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 4),
               [dst3] "r" (_dst + i + 8),
               [dst4] "r" (_dst + i + 12)
             : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21"
         );
     }
})
#else
CVT_FUNC(u8, f32, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         uint8x16_t vline_u8 = vld1q_u8(_src + i);

         uint16x8_t vline1_u16 = vmovl_u8(vget_low_u8(vline_u8));
         uint16x8_t vline2_u16 = vmovl_u8(vget_high_u8(vline_u8));

         uint32x4_t vline1_u32 = vmovl_u16(vget_low_u16(vline1_u16));
         uint32x4_t vline2_u32 = vmovl_u16(vget_high_u16(vline1_u16));
         uint32x4_t vline3_u32 = vmovl_u16(vget_low_u16(vline2_u16));
         uint32x4_t vline4_u32 = vmovl_u16(vget_high_u16(vline2_u16));

         float32x4_t vline1_f32 = vcvtq_f32_u32(vline1_u32);
         float32x4_t vline2_f32 = vcvtq_f32_u32(vline2_u32);
         float32x4_t vline3_f32 = vcvtq_f32_u32(vline3_u32);
         float32x4_t vline4_f32 = vcvtq_f32_u32(vline4_u32);

         vst1q_f32(_dst + i, vline1_f32);
         vst1q_f32(_dst + i + 4, vline2_f32);
         vst1q_f32(_dst + i + 8, vline3_f32);
         vst1q_f32(_dst + i + 12, vline4_f32);
     }
})
#endif

CVT_FUNC(s8, u8, 16,
     int8x16_t vZero = vdupq_n_s8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         int8x16_t vu8 = vld1q_s8(_src + i);
         uint8x16_t vu1 = vreinterpretq_u8_s8(vmaxq_s8(vu8, vZero));
         vst1q_u8(_dst + i, vu1);
     }
})

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(s8, u16, 16,
     register uint8x16_t zero0 asm ("q1") = vmovq_n_u8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vmax.s8 q0, q1                                        \n\t"
             "vst2.8 {d0,d2}, [%[dst1]]                             \n\t"
             "vst2.8 {d1,d3}, [%[dst2]]                             \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 8),
               "w" (zero0)
             : "d0","d1"
         );
     }
})
#else
CVT_FUNC(s8, u16, 16,
     int8x16x2_t vline_s8;
     vline_s8.val[1] = vmovq_n_s8(0);,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         vline_s8.val[0] = vld1q_s8(_src + i);
         vline_s8.val[0] = vmaxq_s8(vline_s8.val[0], vline_s8.val[1]);
         vst2q_s8((int8_t*)(_dst + i), vline_s8);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s8, s16, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.s8 q1, d0                                       \n\t"
             "vmovl.s8 q2, d1                                       \n\t"
             "vst1.16 {d2-d3}, [%[dst1]]                            \n\t"
             "vst1.16 {d4-d5}, [%[dst2]]                            \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 8)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(s8, s16, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         int8x16_t vline_s8 = vld1q_s8(_src + i);

         int16x8_t vline1_s16 = vmovl_s8(vget_low_s8(vline_s8));
         int16x8_t vline2_s16 = vmovl_s8(vget_high_s8(vline_s8));

         vst1q_s16(_dst + i, vline1_s16);
         vst1q_s16(_dst + i + 8, vline2_s16);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(s8, s32, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.s8 q1, d0                                       \n\t"
             "vmovl.s8 q2, d1                                       \n\t"
             "vmovl.s16 q3, d2                                      \n\t"
             "vmovl.s16 q4, d3                                      \n\t"
             "vmovl.s16 q5, d4                                      \n\t"
             "vmovl.s16 q6, d5                                      \n\t"
             "vst1.32 {d6-d7}, [%[dst1]]                            \n\t"
             "vst1.32 {d8-d9}, [%[dst2]]                            \n\t"
             "vst1.32 {d10-d11}, [%[dst3]]                          \n\t"
             "vst1.32 {d12-d13}, [%[dst4]]                          \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 4),
               [dst3] "r" (_dst + i + 8),
               [dst4] "r" (_dst + i + 12)
             : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
         );
     }
})
#else
CVT_FUNC(s8, s32, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         int8x16_t vline_s8 = vld1q_s8(_src + i);

         int16x8_t vline1_s16 = vmovl_s8(vget_low_s8(vline_s8));
         int16x8_t vline2_s16 = vmovl_s8(vget_high_s8(vline_s8));

         int32x4_t vline1_s32 = vmovl_s16(vget_low_s16(vline1_s16));
         int32x4_t vline2_s32 = vmovl_s16(vget_high_s16(vline1_s16));
         int32x4_t vline3_s32 = vmovl_s16(vget_low_s16(vline2_s16));
         int32x4_t vline4_s32 = vmovl_s16(vget_high_s16(vline2_s16));

         vst1q_s32(_dst + i, vline1_s32);
         vst1q_s32(_dst + i + 4, vline2_s32);
         vst1q_s32(_dst + i + 8, vline3_s32);
         vst1q_s32(_dst + i + 12, vline4_s32);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s8, f32, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.s8 q1, d0                                       \n\t"
             "vmovl.s8 q2, d1                                       \n\t"
             "vmovl.s16 q3, d2                                      \n\t"
             "vmovl.s16 q4, d3                                      \n\t"
             "vmovl.s16 q5, d4                                      \n\t"
             "vmovl.s16 q6, d5                                      \n\t"
             "vcvt.f32.s32 q7, q3                                   \n\t"
             "vcvt.f32.s32 q8, q4                                   \n\t"
             "vcvt.f32.s32 q9, q5                                   \n\t"
             "vcvt.f32.s32 q10, q6                                  \n\t"
             "vst1.32 {d14-d15}, [%[dst1]]                          \n\t"
             "vst1.32 {d16-d17}, [%[dst2]]                          \n\t"
             "vst1.32 {d18-d19}, [%[dst3]]                          \n\t"
             "vst1.32 {d20-d21}, [%[dst4]]                          \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 4),
               [dst3] "r" (_dst + i + 8),
               [dst4] "r" (_dst + i + 12)
             : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21"
         );
     }
})
#else
CVT_FUNC(s8, f32, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         int8x16_t vline_s8 = vld1q_s8(_src + i);

         int16x8_t vline1_s16 = vmovl_s8(vget_low_s8(vline_s8));
         int16x8_t vline2_s16 = vmovl_s8(vget_high_s8(vline_s8));

         int32x4_t vline1_s32 = vmovl_s16(vget_low_s16(vline1_s16));
         int32x4_t vline2_s32 = vmovl_s16(vget_high_s16(vline1_s16));
         int32x4_t vline3_s32 = vmovl_s16(vget_low_s16(vline2_s16));
         int32x4_t vline4_s32 = vmovl_s16(vget_high_s16(vline2_s16));

         float32x4_t vline1_f32 = vcvtq_f32_s32(vline1_s32);
         float32x4_t vline2_f32 = vcvtq_f32_s32(vline2_s32);
         float32x4_t vline3_f32 = vcvtq_f32_s32(vline3_s32);
         float32x4_t vline4_f32 = vcvtq_f32_s32(vline4_s32);

         vst1q_f32(_dst + i, vline1_f32);
         vst1q_f32(_dst + i + 4, vline2_f32);
         vst1q_f32(_dst + i + 8, vline3_f32);
         vst1q_f32(_dst + i + 12, vline4_f32);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(u16, u8, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src1]]                             \n\t"
             "vqmovn.u16 d4, q0                                     \n\t"
             "vld1.8 {d2-d3}, [%[src2]]                             \n\t"
             "vqmovn.u16 d5, q1                                     \n\t"
             "vst1.8 {d4-d5}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src1] "r" (_src + i),
               [src2] "r" (_src + i + 8),
               [dst] "r" (_dst + i + 0)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(u16, u8, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         uint16x8_t vline1_u16 = vld1q_u16(_src + i);
         uint16x8_t vline2_u16 = vld1q_u16(_src + i + 8);

         uint8x8_t vline1_u8 = vqmovn_u16(vline1_u16);
         uint8x8_t vline2_u8 = vqmovn_u16(vline2_u16);

         vst1q_u8(_dst + i, vcombine_u8(vline1_u8, vline2_u8));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(u16, s8, 16,
    register uint8x16_t v127 asm ("q4") = vmovq_n_u8(127);,
{
    for (size_t i = 0; i < w; i += 16)
    {
        internal::prefetch(_src + i);
        __asm__ (
            "vld1.8 {d0-d1}, [%[src1]]                             \n\t"
            "vqmovn.u16 d4, q0                                     \n\t"
            "vld1.8 {d2-d3}, [%[src2]]                             \n\t"
            "vqmovn.u16 d5, q1                                     \n\t"
            "vmin.u8 q3, q2, q4                                    \n\t"
            "vst1.8 {d6-d7}, [%[dst]]                              \n\t"
            : /*no output*/
            : [src1] "r" (_src + i),
              [src2] "r" (_src + i + 8),
              [dst] "r" (_dst + i + 0),
              "w" (v127)
            : "d0","d1","d2","d3","d4","d5","d6","d7"
         );
    }
})
#else
CVT_FUNC(u16, s8, 16,
    uint8x8_t v127 = vmov_n_u8(127);,
{
    for (size_t i = 0; i < w; i += 16)
    {
        internal::prefetch(_src + i);
        uint16x8_t vline1_u16 = vld1q_u16(_src + i);
        uint16x8_t vline2_u16 = vld1q_u16(_src + i + 8);

        uint8x8_t vline1_u8 = vqmovn_u16(vline1_u16);
        uint8x8_t vline2_u8 = vqmovn_u16(vline2_u16);
        vline1_u8 = vmin_u8(vline1_u8, v127);
        vline2_u8 = vmin_u8(vline2_u8, v127);

        vst1q_s8(_dst + i, vcombine_s8(vreinterpret_s8_u8(vline1_u8), vreinterpret_s8_u8(vline2_u8)));
    }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(u16, s16, 8,
     register uint16x8_t v32767 asm ("q4") = vmovq_n_u16(0x7FFF);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.16 {d0-d1}, [%[src]]                              \n\t"
             "vmin.u16 q1, q0, q4                                    \n\t"
             "vst1.16 {d2-d3}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst] "r" (_dst + i + 0),
               "w" (v32767)
             : "d0","d1","d2","d3"
         );
     }
})
#else
CVT_FUNC(u16, s16, 8,
     uint16x8_t v32767 = vmovq_n_u16(0x7FFF);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         uint16x8_t vline_u16 = vld1q_u16(_src + i);
         vline_u16 = vminq_u16(vline_u16, v32767);
         vst1q_s16((_dst + i), vreinterpretq_s16_u16(vline_u16));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(u16, s32, 8,
     register uint16x8_t zero0 asm ("q1") = vmovq_n_u16(0);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.16 {d0-d1}, [%[src]]                        \n\t"
             "vst2.16 {d0,d2}, [%[dst1]]                       \n\t"
             "vst2.16 {d1,d3}, [%[dst2]]                       \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i),
               [dst2] "r" (_dst + i + 4),
               "w" (zero0)
             : "d0","d1"//,"d2","d3"//,"d4","d5","d6","d7"
         );
     }
})
#else
CVT_FUNC(u16, s32, 8,
     uint16x8x2_t vline;
     vline.val[1] = vmovq_n_u16(0);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         vline.val[0] = vld1q_u16(_src + i);
         vst2q_u16((uint16_t*)(_dst + i), vline);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(u16, f32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.16 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.u16 q1, d0                                       \n\t"
             "vmovl.u16 q2, d1                                       \n\t"
             "vcvt.f32.u32 q3, q1                                    \n\t"
             "vcvt.f32.u32 q4, q2                                    \n\t"
             "vst1.32 {d6-d7}, [%[dst1]]                             \n\t"
             "vst1.32 {d8-d9}, [%[dst2]]                             \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 4)
             : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
         );
     }
})
#else
CVT_FUNC(u16, f32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         uint16x8_t vline_u16 = vld1q_u16(_src + i);

         uint32x4_t vline_u32_lo = vmovl_u16(vget_low_u16(vline_u16));
         uint32x4_t vline_u32_hi = vmovl_u16(vget_high_u16(vline_u16));

         float32x4_t vline_f32_lo = vcvtq_f32_u32(vline_u32_lo);
         float32x4_t vline_f32_hi = vcvtq_f32_u32(vline_u32_hi);

         vst1q_f32(_dst + i, vline_f32_lo);
         vst1q_f32(_dst + i + 4, vline_f32_hi);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s16, u8, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src1]]                             \n\t"
             "vld1.8 {d2-d3}, [%[src2]]                             \n\t"
             "vqmovun.s16 d4, q0                                    \n\t"
             "vqmovun.s16 d5, q1                                    \n\t"
             "vst1.8 {d4-d5}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src1] "r" (_src + i),
               [src2] "r" (_src + i + 8),
               [dst] "r" (_dst + i + 0)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(s16, u8, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         int16x8_t vline1_s16 = vld1q_s16(_src + i);
         int16x8_t vline2_s16 = vld1q_s16(_src + i + 8);

         uint8x8_t vline1_u8 = vqmovun_s16(vline1_s16);
         uint8x8_t vline2_u8 = vqmovun_s16(vline2_s16);

         vst1q_u8(_dst + i, vcombine_u8(vline1_u8, vline2_u8));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s16, s8, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.8 {d0-d1}, [%[src1]]                             \n\t"
             "vld1.8 {d2-d3}, [%[src2]]                             \n\t"
             "vqmovn.s16 d4, q0                                     \n\t"
             "vqmovn.s16 d5, q1                                     \n\t"
             "vst1.8 {d4-d5}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src1] "r" (_src + i),
               [src2] "r" (_src + i + 8),
               [dst] "r" (_dst + i + 0)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(s16, s8, 16,
,
{
     for (size_t i = 0; i < w; i += 16)
     {
         internal::prefetch(_src + i);
         int16x8_t vline1_s16 = vld1q_s16(_src + i);
         int16x8_t vline2_s16 = vld1q_s16(_src + i + 8);

         int8x8_t vline1_s8 = vqmovn_s16(vline1_s16);
         int8x8_t vline2_s8 = vqmovn_s16(vline2_s16);

         vst1q_s8(_dst + i, vcombine_s8(vline1_s8, vline2_s8));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7
CVT_FUNC(s16, u16, 8,
     register int16x8_t vZero asm ("q4") = vmovq_n_s16(0);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.16 {d0-d1}, [%[src]]                              \n\t"
             "vmax.s16 q1, q0, q4                                    \n\t"
             "vst1.16 {d2-d3}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst] "r" (_dst + i + 0),
               "w" (vZero)
             : "d0","d1","d2","d3"
         );
     }
})
#else
CVT_FUNC(s16, u16, 8,
     int16x4_t vZero = vmov_n_s16(0);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int16x8_t vline_s16 = vld1q_s16(_src + i);

         int16x4_t vline_s16_lo = vmax_s16(vget_low_s16(vline_s16), vZero);
         int16x4_t vline_s16_hi = vmax_s16(vget_high_s16(vline_s16), vZero);

         vst1q_u16(_dst + i, vcombine_u16(vreinterpret_u16_s16(vline_s16_lo), vreinterpret_u16_s16(vline_s16_hi)));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s16, s32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.16 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.s16 q1, d0                                       \n\t"
             "vmovl.s16 q2, d1                                       \n\t"
             "vst1.32 {d2-d3}, [%[dst1]]                             \n\t"
             "vst1.32 {d4-d5}, [%[dst2]]                             \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 4)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(s16, s32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int16x8_t vline_s16 = vld1q_s16(_src + i);

         int32x4_t vline_s32_lo = vmovl_s16(vget_low_s16(vline_s16));
         int32x4_t vline_s32_hi = vmovl_s16(vget_high_s16(vline_s16));

         vst1q_s32(_dst + i, vline_s32_lo);
         vst1q_s32(_dst + i + 4, vline_s32_hi);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s16, f32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.16 {d0-d1}, [%[src]]                              \n\t"
             "vmovl.s16 q1, d0                                       \n\t"
             "vmovl.s16 q2, d1                                       \n\t"
             "vcvt.f32.s32 q3, q1                                    \n\t"
             "vcvt.f32.s32 q4, q2                                    \n\t"
             "vst1.32 {d6-d7}, [%[dst1]]                             \n\t"
             "vst1.32 {d8-d9}, [%[dst2]]                             \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst1] "r" (_dst + i + 0),
               [dst2] "r" (_dst + i + 4)
             : "d0","d1","d2","d3","d4","d5","d6","d7","d8","d9"
         );
     }
})
#else
CVT_FUNC(s16, f32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int16x8_t vline_s16 = vld1q_s16(_src + i);

         int32x4_t vline_s32_lo = vmovl_s16(vget_low_s16(vline_s16));
         int32x4_t vline_s32_hi = vmovl_s16(vget_high_s16(vline_s16));
         float32x4_t vline_f32_lo = vcvtq_f32_s32(vline_s32_lo);
         float32x4_t vline_f32_hi = vcvtq_f32_s32(vline_s32_hi);

         vst1q_f32(_dst + i, vline_f32_lo);
         vst1q_f32(_dst + i + 4, vline_f32_hi);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s32, u8, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d0-d1}, [%[src1]]                              \n\t"
             "vld1.32 {d2-d3}, [%[src2]]                              \n\t"
             "vqmovun.s32 d4, q0                                      \n\t"
             "vqmovun.s32 d5, q1                                      \n\t"
             "vqmovn.u16  d6, q2                                      \n\t"
             "vst1.8 {d6}, [%[dst]]                                   \n\t"
             : /*no output*/
             : [src1] "r" (_src + i + 0),
               [src2] "r" (_src + i + 4),
               [dst] "r" (_dst + i)
             : "d0","d1","d2","d3","d4","d5","d6"
         );
     }
})
#else
CVT_FUNC(s32, u8, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int32x4_t vline1_s32 = vld1q_s32(_src + i);
         int32x4_t vline2_s32 = vld1q_s32(_src + i + 4);

         uint16x4_t vline1_u16 = vqmovun_s32(vline1_s32);
         uint16x4_t vline2_u16 = vqmovun_s32(vline2_s32);
         uint8x8_t vline_u8 = vqmovn_u16(vcombine_u16(vline1_u16, vline2_u16));

         vst1_u8(_dst + i, vline_u8);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s32, s8, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d0-d1}, [%[src1]]                              \n\t"
             "vld1.32 {d2-d3}, [%[src2]]                              \n\t"
             "vqmovn.s32 d4, q0                                       \n\t"
             "vqmovn.s32 d5, q1                                       \n\t"
             "vqmovn.s16  d6, q2                                      \n\t"
             "vst1.8 {d6}, [%[dst]]                                   \n\t"
             : /*no output*/
             : [src1] "r" (_src + i + 0),
               [src2] "r" (_src + i + 4),
               [dst] "r" (_dst + i)
             : "d0","d1","d2","d3","d4","d5","d6"
         );
     }
})
#else
CVT_FUNC(s32, s8, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int32x4_t vline1_s32 = vld1q_s32(_src + i);
         int32x4_t vline2_s32 = vld1q_s32(_src + i + 4);

         int16x4_t vline1_s16 = vqmovn_s32(vline1_s32);
         int16x4_t vline2_s16 = vqmovn_s32(vline2_s32);
         int8x8_t vline_s8 = vqmovn_s16(vcombine_s16(vline1_s16, vline2_s16));

         vst1_s8(_dst + i, vline_s8);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s32, u16, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d0-d1}, [%[src1]]                              \n\t"
             "vld1.32 {d2-d3}, [%[src2]]                              \n\t"
             "vqmovun.s32 d4, q0                                      \n\t"
             "vqmovun.s32 d5, q1                                      \n\t"
             "vst1.16 {d4-d5}, [%[dst]]                               \n\t"
             : /*no output*/
             : [src1] "r" (_src + i + 0),
               [src2] "r" (_src + i + 4),
               [dst] "r" (_dst + i)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(s32, u16, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int32x4_t vline1_s32 = vld1q_s32(_src + i);
         int32x4_t vline2_s32 = vld1q_s32(_src + i + 4);

         uint16x4_t vline1_u16 = vqmovun_s32(vline1_s32);
         uint16x4_t vline2_u16 = vqmovun_s32(vline2_s32);

         vst1q_u16(_dst + i, vcombine_u16(vline1_u16, vline2_u16));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s32, s16, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d0-d1}, [%[src1]]                              \n\t"
             "vld1.32 {d2-d3}, [%[src2]]                              \n\t"
             "vqmovn.s32 d4, q0                                       \n\t"
             "vqmovn.s32 d5, q1                                       \n\t"
             "vst1.8 {d4-d5}, [%[dst]]                                \n\t"
             : /*no output*/
             : [src1] "r" (_src + i + 0),
               [src2] "r" (_src + i + 4),
               [dst] "r" (_dst + i)
             : "d0","d1","d2","d3","d4","d5"
         );
     }
})
#else
CVT_FUNC(s32, s16, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int32x4_t vline1_s32 = vld1q_s32(_src + i);
         int32x4_t vline2_s32 = vld1q_s32(_src + i + 4);

         int16x4_t vline1_s16 = vqmovn_s32(vline1_s32);
         int16x4_t vline2_s16 = vqmovn_s32(vline2_s32);

         vst1q_s16(_dst + i, vcombine_s16(vline1_s16, vline2_s16));
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(s32, f32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d0-d1}, [%[src]]                              \n\t"
             "vcvt.f32.s32 q1, q0                                    \n\t"
             "vst1.32 {d2-d3}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst] "r" (_dst + i)
             : "d0","d1","d2","d3"//,"d4","d5"
         );
         __asm__ (
             "vld1.32 {d0-d1}, [%[src]]                              \n\t"
             "vcvt.f32.s32 q1, q0                                    \n\t"
             "vst1.32 {d2-d3}, [%[dst]]                              \n\t"
             : /*no output*/
             : [src] "r" (_src + i + 4),
               [dst] "r" (_dst + i + 4)
             : "d0","d1","d2","d3"//,"d4","d5"
         );
     }
})
#else
CVT_FUNC(s32, f32, 8,
,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         int32x4_t vline_s32 = vld1q_s32(_src + i);
         float32x4_t vline_f32 = vcvtq_f32_s32(vline_s32);
         vst1q_f32(_dst + i, vline_f32);

         vline_s32 = vld1q_s32(_src + i + 4);
         vline_f32 = vcvtq_f32_s32(vline_s32);
         vst1q_f32(_dst + i + 4, vline_f32);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(f32, u8, 8,
    register float32x4_t vmult asm ("q0") = vdupq_n_f32((float)(1 << 16));
    register uint32x4_t  vmask asm ("q1") = vdupq_n_u32(1<<16);,
{
    for (size_t i = 0; i < w; i += 8)
    {
        internal::prefetch(_src + i);
        __asm__ (
            "vld1.32 {d4-d5}, [%[src1]]                              \n\t"
            "vld1.32 {d6-d7}, [%[src2]]                              \n\t"
            "vmul.f32 q4, q2, q0                                     \n\t"
            "vmul.f32 q5, q3, q0                                     \n\t"
            "vcvt.u32.f32 q6, q4                                     \n\t"
            "vcvt.u32.f32 q7, q5                                     \n\t"
            "vbic q8, q1, q6                                         \n\t"
            "vbic q9, q1, q7                                         \n\t"
            "vshr.u32 q10, q8, #16                                   \n\t"
            "vshr.u32 q11, q9, #16                                   \n\t"
            "vqsub.u32 q12, q6, q10                                  \n\t"
            "vqsub.u32 q13, q7, q11                                  \n\t"
            "vqrshrn.u32 d28, q12, #16                               \n\t"
            "vqrshrn.u32 d29, q13, #16                               \n\t"
            "vqmovn.u16 d30, q14                                     \n\t"
            "vst1.8 {d30}, [%[dst]]                                  \n\t"
            : /*no output*/
            : [src1] "r" (_src + i + 0),
              [src2] "r" (_src + i + 4),
              [dst] "r" (_dst + i),
              "w" (vmult), "w" (vmask)
            : "d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17","d18","d19","d20","d21","d22","d23","d24","d25","d26","d27","d28","d29","d30"
        );
     }
})
#else
CVT_FUNC(f32, u8, 8,
    float32x4_t vmult = vdupq_n_f32((float)(1 << 16));
    uint32x4_t  vmask = vdupq_n_u32(1<<16);,
{
    for (size_t i = 0; i < w; i += 8)
    {
        internal::prefetch(_src + i);
        float32x4_t vline1_f32 = vld1q_f32(_src + i);
        float32x4_t vline2_f32 = vld1q_f32(_src + i + 4);

        float32x4_t vline1w_f32 = vmulq_f32(vline1_f32, vmult);
        float32x4_t vline2w_f32 = vmulq_f32(vline2_f32, vmult);

        uint32x4_t vline1_u32 = vcvtq_u32_f32(vline1w_f32);
        uint32x4_t vline2_u32 = vcvtq_u32_f32(vline2w_f32);

        uint32x4_t vl1_masked = vbicq_u32(vmask, vline1_u32);
        uint32x4_t vl2_masked = vbicq_u32(vmask, vline2_u32);
        uint32x4_t vl1_masked2 = vshrq_n_u32(vl1_masked, 16);
        uint32x4_t vl2_masked2 = vshrq_n_u32(vl2_masked, 16);
        uint32x4_t vline1r_u32 = vqsubq_u32(vline1_u32, vl1_masked2);
        uint32x4_t vline2r_u32 = vqsubq_u32(vline2_u32, vl2_masked2);

        uint16x4_t vline1_u16 = vqrshrn_n_u32(vline1r_u32, 16);
        uint16x4_t vline2_u16 = vqrshrn_n_u32(vline2r_u32, 16);

        uint8x8_t vline_u8 = vqmovn_u16(vcombine_u16(vline1_u16, vline2_u16));
        vst1_u8(_dst + i, vline_u8);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(f32, s8, 8,
     register float32x4_t vhalf asm ("q0") = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d2-d3}, [%[src1]]                              \n\t"
             "vld1.32 {d4-d5}, [%[src2]]                              \n\t"
             "vadd.f32 q3, q1, q0                                     \n\t"
             "vadd.f32 q4, q2, q0                                     \n\t"
             "vcvt.s32.f32 q5, q3                                     \n\t"
             "vcvt.s32.f32 q6, q4                                     \n\t"
             "vqmovn.s32 d14, q5                                      \n\t"
             "vqmovn.s32 d15, q6                                      \n\t"
             "vqmovn.s16 d16, q7                                      \n\t"
             "vst1.8 {d16}, [%[dst]]                                  \n\t"
             : /*no output*/
             : [src1] "r" (_src + i + 0),
               [src2] "r" (_src + i + 4),
               [dst] "r" (_dst + i),
               "w" (vhalf)
             : "d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13","d14","d15","d16","d17"
         );
     }
})
#else
CVT_FUNC(f32, s8, 8,
     float32x4_t vhalf = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         float32x4_t vline1_f32 = vld1q_f32(_src + i);
         float32x4_t vline2_f32 = vld1q_f32(_src + i + 4);

         vline1_f32 = vaddq_f32(vline1_f32, vhalf);
         vline2_f32 = vaddq_f32(vline2_f32, vhalf);

         int32x4_t vline1_s32 = vcvtq_s32_f32(vline1_f32);
         int32x4_t vline2_s32 = vcvtq_s32_f32(vline2_f32);
         int16x4_t vline1_s16 = vqmovn_s32(vline1_s32);
         int16x4_t vline2_s16 = vqmovn_s32(vline2_s32);

         int8x8_t vline_s8 = vqmovn_s16(vcombine_s16(vline1_s16, vline2_s16));

         vst1_s8(_dst + i, vline_s8);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(f32, u16, 8,
     register float32x4_t vhalf asm ("q0") = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d2-d3}, [%[src]]                               \n\t"
             "vadd.f32 q2, q1, q0                                     \n\t"
             "vcvt.u32.f32 q3, q2                                     \n\t"
             "vqmovn.u32 d8, q3                                       \n\t"
             "vst1.16 {d8}, [%[dst]]                                  \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst] "r" (_dst + i),
               "w" (vhalf)
             : "d2","d3","d4","d5","d6","d7","d8"
         );
         __asm__ (
             "vld1.32 {d2-d3}, [%[src]]                               \n\t"
             "vadd.f32 q2, q1, q0                                     \n\t"
             "vcvt.u32.f32 q3, q2                                     \n\t"
             "vqmovn.u32 d8, q3                                       \n\t"
             "vst1.16 {d8}, [%[dst]]                                  \n\t"
             : /*no output*/
             : [src] "r" (_src + i + 4),
               [dst] "r" (_dst + i + 4),
               "w" (vhalf)
             : "d2","d3","d4","d5","d6","d7","d8"
         );
     }
})
#else
CVT_FUNC(f32, u16, 8,
     float32x4_t vhalf = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         float32x4_t vline_f32 = vld1q_f32(_src + i);

         vline_f32 = vaddq_f32(vline_f32, vhalf);
         uint32x4_t vline_u32 = vcvtq_u32_f32(vline_f32);
         uint16x4_t vline_u16 = vqmovn_u32(vline_u32);

         vst1_u16(_dst + i, vline_u16);

         vline_f32 = vld1q_f32(_src + i + 4);

         vline_f32 = vaddq_f32(vline_f32, vhalf);
         vline_u32 = vcvtq_u32_f32(vline_f32);
         vline_u16 = vqmovn_u32(vline_u32);

         vst1_u16(_dst + i + 4, vline_u16);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(f32, s16, 8,
     register float32x4_t vhalf asm ("q0") = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d2-d3}, [%[src]]                               \n\t"
             "vadd.f32 q2, q1, q0                                     \n\t"
             "vcvt.s32.f32 q3, q2                                     \n\t"
             "vqmovn.s32 d8, q3                                       \n\t"
             "vst1.16 {d8}, [%[dst]]                                  \n\t"
             : /*no output*/
             : [src] "r" (_src + i),
               [dst] "r" (_dst + i),
               "w" (vhalf)
             : "d2","d3","d4","d5","d6","d7","d8"
         );
         __asm__ (
             "vld1.32 {d2-d3}, [%[src]]                               \n\t"
             "vadd.f32 q2, q1, q0                                     \n\t"
             "vcvt.s32.f32 q3, q2                                     \n\t"
             "vqmovn.s32 d8, q3                                       \n\t"
             "vst1.16 {d8}, [%[dst]]                                  \n\t"
             : /*no output*/
             : [src] "r" (_src + i + 4),
               [dst] "r" (_dst + i + 4),
               "w" (vhalf)
             : "d2","d3","d4","d5","d6","d7","d8"
         );
     }
})
#else
CVT_FUNC(f32, s16, 8,
     float32x4_t vhalf = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         float32x4_t vline_f32 = vld1q_f32(_src + i);

         vline_f32 = vaddq_f32(vline_f32, vhalf);
         int32x4_t vline_s32 = vcvtq_s32_f32(vline_f32);
         int16x4_t vline_s16 = vqmovn_s32(vline_s32);

         vst1_s16(_dst + i, vline_s16);

         vline_f32 = vld1q_f32(_src + i + 4);

         vline_f32 = vaddq_f32(vline_f32, vhalf);
         vline_s32 = vcvtq_s32_f32(vline_f32);
         vline_s16 = vqmovn_s32(vline_s32);

         vst1_s16(_dst + i + 4, vline_s16);
     }
})
#endif

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 6
CVT_FUNC(f32, s32, 8,
     register float32x4_t vhalf asm ("q0") = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         __asm__ (
             "vld1.32 {d2-d3}, [%[src1]]                              \n\t"
             "vld1.32 {d4-d5}, [%[src2]]                              \n\t"
             "vadd.f32 q3, q1, q0                                     \n\t"
             "vadd.f32 q4, q2, q0                                     \n\t"
             "vcvt.s32.f32 q5, q3                                     \n\t"
             "vcvt.s32.f32 q6, q4                                     \n\t"
             "vst1.32 {q5}, [%[dst1]]                                 \n\t"
             "vst1.32 {q6}, [%[dst2]]                                 \n\t"
             : /*no output*/
             : [src1] "r" (_src + i),
               [src2] "r" (_src + i + 4),
               [dst1] "r" (_dst + i),
               [dst2] "r" (_dst + i + 4),
               "w" (vhalf)
             : "d2","d3","d4","d5","d6","d7","d8","d9","d10","d11","d12","d13"
         );
     }
})
#else
CVT_FUNC(f32, s32, 8,
     float32x4_t vhalf = vdupq_n_f32(0.5f);,
{
     for (size_t i = 0; i < w; i += 8)
     {
         internal::prefetch(_src + i);
         float32x4_t vline_f32 = vld1q_f32(_src + i);

         vline_f32 = vaddq_f32(vline_f32, vhalf);
         int32x4_t vline_s32 = vcvtq_s32_f32(vline_f32);

         vst1q_s32(_dst + i, vline_s32);

         vline_f32 = vld1q_f32(_src + i + 4);

         vline_f32 = vaddq_f32(vline_f32, vhalf);
         vline_s32 = vcvtq_s32_f32(vline_f32);

         vst1q_s32(_dst + i + 4, vline_s32);
     }
})
#endif

void convert(const Size2D &_size,
             const u8 * srcBase, ptrdiff_t srcStride,
             s16 * dstBase, ptrdiff_t dstStride)
{
    convert(_size, srcBase, srcStride, (u16*)dstBase, dstStride);
}

} // namespace CAROTENE_NS
