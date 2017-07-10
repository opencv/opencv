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
 * Copyright (C) 2014-2015, NVIDIA Corporation, all rights reserved.
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

void extract2(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              u32 coi)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#ifndef __ANDROID__
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#ifndef __ANDROID__
        for (; dj < roiw32; sj += 64, dj += 32)
        {
            internal::prefetch(src + sj);

            uint8x16x2_t v_src = vld2q_u8(src + sj);
            vst1q_u8(dst + dj, v_src.val[coi]);

            v_src = vld2q_u8(src + sj + 32);
            vst1q_u8(dst + dj + 16, v_src.val[coi]);
        }
#endif

        for (; dj < roiw8; sj += 16, dj += 8)
        {
            uint8x8x2_t v_src = vld2_u8(src + sj);
            vst1_u8(dst + dj, v_src.val[coi]);
        }

        for (; dj < size.width; sj += 2, ++dj)
        {
            dst[dj] = src[sj + coi];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)coi;
#endif
}

void extract3(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              u32 coi)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#ifndef __ANDROID__
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#ifndef __ANDROID__
        for (; dj < roiw32; sj += 96, dj += 32)
        {
            internal::prefetch(src + sj);

            uint8x16x3_t v_src = vld3q_u8(src + sj);
            vst1q_u8(dst + dj, v_src.val[coi]);

            v_src = vld3q_u8(src + sj + 48);
            vst1q_u8(dst + dj + 16, v_src.val[coi]);
        }
#endif

        for (; dj < roiw8; sj += 24, dj += 8)
        {
            uint8x8x3_t v_src = vld3_u8(src + sj);
            vst1_u8(dst + dj, v_src.val[coi]);
        }

        for (; dj < size.width; sj += 3, ++dj)
        {
            dst[dj] = src[sj + coi];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)coi;
#endif
}

void extract4(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              u32 coi)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#ifndef __ANDROID__
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * src = internal::getRowPtr(srcBase, srcStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t sj = 0u, dj = 0u;

#ifndef __ANDROID__
        for (; dj < roiw32; sj += 128, dj += 32)
        {
            internal::prefetch(src + sj);

            uint8x16x4_t v_src = vld4q_u8(src + sj);
            vst1q_u8(dst + dj, v_src.val[coi]);

            v_src = vld4q_u8(src + sj + 64);
            vst1q_u8(dst + dj + 16, v_src.val[coi]);
        }
#endif

        for (; dj < roiw8; sj += 32, dj += 8)
        {
            uint8x8x4_t v_src = vld4_u8(src + sj);
            vst1_u8(dst + dj, v_src.val[coi]);
        }

        for (; dj < size.width; sj += 4, ++dj)
        {
            dst[dj] = src[sj + coi];
        }
    }
#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)coi;
#endif
}

#define FILL_LINES2(macro,type) \
            macro##_LINE(type,0) \
            macro##_LINE(type,1)
#define FILL_LINES3(macro,type) \
            FILL_LINES2(macro,type) \
            macro##_LINE(type,2)
#define FILL_LINES4(macro,type) \
            FILL_LINES3(macro,type) \
            macro##_LINE(type,3)

#define FARG_LINE(type, n) , type * dst##n##Base, ptrdiff_t dst##n##Stride

#ifdef CAROTENE_NEON

#define VROW_LINE(type, n) type * dst##n = internal::getRowPtr(dst##n##Base, dst##n##Stride, i);
#define VST1Q_LINE(type, n) vst1q_##type(dst##n + dj, v_src.val[n]);
#define VST1_LINE(type, n) vst1_##type(dst##n + dj, v_src.val[n]);
#define SST_LINE(type, n) dst##n[dj] = src[sj + n];

#define MUL2(val) (val << 1)
#define MUL3(val) (MUL2(val) + val)
#define MUL4(val) (val << 2)

#define CONTDST2 srcStride == dst0Stride && \
                 srcStride == dst1Stride &&
#define CONTDST3 srcStride == dst0Stride && \
                 srcStride == dst1Stride && \
                 srcStride == dst2Stride &&
#define CONTDST4 srcStride == dst0Stride && \
                 srcStride == dst1Stride && \
                 srcStride == dst2Stride && \
                 srcStride == dst3Stride &&

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7

#define SPLIT_ASM2(sgn, bits) __asm__ ( \
                                          "vld2." #bits " {d0, d2}, [%[in0]]            \n\t" \
                                          "vld2." #bits " {d1, d3}, [%[in1]]            \n\t" \
                                          "vst1." #bits " {d0-d1}, [%[out0]]            \n\t" \
                                          "vst1." #bits " {d2-d3}, [%[out1]]            \n\t" \
                                          : \
                                          : [out0] "r" (dst0 + dj), [out1] "r" (dst1 + dj), \
                                            [in0]  "r" (src + sj), [in1]  "r" (src + sj + MUL2(8)/sizeof(sgn##bits)) \
                                          : "d0","d1","d2","d3" \
                                      );
#define SPLIT_ASM3(sgn, bits) __asm__ ( \
                                          "vld3." #bits " {d0, d2, d4}, [%[in0]]        \n\t" \
                                          "vld3." #bits " {d1, d3, d5}, [%[in1]]        \n\t" \
                                          "vst1." #bits " {d0-d1}, [%[out0]]            \n\t" \
                                          "vst1." #bits " {d2-d3}, [%[out1]]            \n\t" \
                                          "vst1." #bits " {d4-d5}, [%[out2]]            \n\t" \
                                          : \
                                          : [out0] "r" (dst0 + dj), [out1] "r" (dst1 + dj), [out2] "r" (dst2 + dj), \
                                            [in0]  "r" (src + sj), [in1]  "r" (src + sj + MUL3(8)/sizeof(sgn##bits)) \
                                          : "d0","d1","d2","d3","d4","d5" \
                                      );
#define SPLIT_ASM4(sgn, bits) __asm__ ( \
                                          "vld4." #bits " {d0, d2, d4, d6}, [%[in0]]    \n\t" \
                                          "vld4." #bits " {d1, d3, d5, d7}, [%[in1]]    \n\t" \
                                          "vst1." #bits " {d0-d1}, [%[out0]]            \n\t" \
                                          "vst1." #bits " {d2-d3}, [%[out1]]            \n\t" \
                                          "vst1." #bits " {d4-d5}, [%[out2]]            \n\t" \
                                          "vst1." #bits " {d6-d7}, [%[out3]]            \n\t" \
                                          : \
                                          : [out0] "r" (dst0 + dj), [out1] "r" (dst1 + dj), [out2] "r" (dst2 + dj), [out3] "r" (dst3 + dj), \
                                            [in0]  "r" (src + sj), [in1]  "r" (src + sj + MUL4(8)/sizeof(sgn##bits)) \
                                          : "d0","d1","d2","d3","d4","d5","d6","d7" \
                                      );

#define SPLIT_QUAD(sgn, bits, n) { \
                                     internal::prefetch(src + sj); \
                                     SPLIT_ASM##n(sgn, bits) \
                                 }

#else

#define SPLIT_QUAD(sgn, bits, n) { \
                                     internal::prefetch(src + sj); \
                                     vec128 v_src = vld##n##q_##sgn##bits(src + sj); \
                                     FILL_LINES##n(VST1Q, sgn##bits) \
                                 }

#endif

#define SPLIT(sgn,bits,n) void split##n(const Size2D &_size,                                            \
                                    const sgn##bits * srcBase, ptrdiff_t srcStride                      \
                                    FILL_LINES##n(FARG, sgn##bits) )                                    \
{                                                                                                       \
    internal::assertSupportedConfiguration();                                                           \
    Size2D size(_size);                                                                                 \
    if (CONTDST##n                                                                                      \
        dst0Stride == (ptrdiff_t)(size.width))                                                          \
    {                                                                                                   \
        size.width *= size.height;                                                                      \
        size.height = 1;                                                                                \
    }                                                                                                   \
    typedef internal::VecTraits<sgn##bits, n>::vec128 vec128;                                           \
    size_t roiw16 = size.width >= (16/sizeof(sgn##bits)-1) ? size.width - (16/sizeof(sgn##bits)-1) : 0; \
    typedef internal::VecTraits<sgn##bits, n>::vec64 vec64;                                             \
    size_t roiw8 = size.width >= (8/sizeof(sgn##bits)-1) ? size.width - (8/sizeof(sgn##bits)-1) : 0;    \
                                                                                                        \
    for (size_t i = 0u; i < size.height; ++i)                                                           \
    {                                                                                                   \
        const sgn##bits * src = internal::getRowPtr(srcBase, srcStride, i);                             \
        FILL_LINES##n(VROW, sgn##bits)                                                                  \
        size_t sj = 0u, dj = 0u;                                                                        \
                                                                                                        \
        for (; dj < roiw16; sj += MUL##n(16)/sizeof(sgn##bits), dj += 16/sizeof(sgn##bits))             \
            SPLIT_QUAD(sgn, bits, n)                                                                    \
                                                                                                        \
        if (dj < roiw8)                                                                                 \
        {                                                                                               \
            vec64 v_src = vld##n##_##sgn##bits(src + sj);                                               \
            FILL_LINES##n(VST1, sgn##bits)                                                              \
            sj += MUL##n(8)/sizeof(sgn##bits);                                                          \
            dj += 8/sizeof(sgn##bits);                                                                  \
        }                                                                                               \
                                                                                                        \
        for (; dj < size.width; sj += n, ++dj)                                                          \
        {                                                                                               \
            FILL_LINES##n(SST, sgn##bits)                                                               \
        }                                                                                               \
    }                                                                                                   \
}

#define SPLIT64(sgn,n) void split##n(const Size2D &_size,                                               \
                                     const sgn##64 * srcBase, ptrdiff_t srcStride                       \
                                     FILL_LINES##n(FARG, sgn##64) )                                     \
{                                                                                                       \
    internal::assertSupportedConfiguration();                                                           \
    Size2D size(_size);                                                                                 \
    if (CONTDST##n                                                                                      \
        dst0Stride == (ptrdiff_t)(size.width))                                                          \
    {                                                                                                   \
        size.width *= size.height;                                                                      \
        size.height = 1;                                                                                \
    }                                                                                                   \
    typedef internal::VecTraits<sgn##64, n>::vec64 vec64;                                               \
                                                                                                        \
    for (size_t i = 0u; i < size.height; ++i)                                                           \
    {                                                                                                   \
        const sgn##64 * src = internal::getRowPtr(srcBase, srcStride, i);                               \
        FILL_LINES##n(VROW, sgn##64)                                                                    \
        size_t sj = 0u, dj = 0u;                                                                        \
                                                                                                        \
        for (; dj < size.width; sj += n, ++dj)                                                          \
        {                                                                                               \
            vec64 v_src = vld##n##_##sgn##64(src + sj);                                                 \
            FILL_LINES##n(VST1, sgn##64)                                                                \
        }                                                                                               \
    }                                                                                                   \
}

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7

#define ALPHA_QUAD(sgn, bits) { \
                                  internal::prefetch(src + sj); \
                                  __asm__ ( \
                                      "vld4." #bits " {d0, d2, d4, d6}, [%[in0]]    \n\t" \
                                      "vld4." #bits " {d1, d3, d5, d7}, [%[in1]]    \n\t" \
                                      "vst3." #bits " {d0, d2, d4}, [%[out3_1]]     \n\t" \
                                      "vst3." #bits " {d1, d3, d5}, [%[out3_2]]     \n\t" \
                                      "vst1." #bits " {d6-d7}, [%[out1]]            \n\t" \
                                      : \
                                      : [out3_1] "r" (dst3 + d3j), [out3_2] "r" (dst3 + d3j + 24/sizeof(sgn##bits)), [out1] "r" (dst1 + d1j), \
                                        [in0]  "r" (src + sj), [in1]  "r" (src + sj + 32/sizeof(sgn##bits)) \
                                      : "d0","d1","d2","d3","d4","d5","d6","d7" \
                                  ); \
                              }

#else

#define ALPHA_QUAD(sgn, bits) { \
                                  internal::prefetch(src + sj); \
                                  union { vec128_4 v4; vec128_3 v3; } vals; \
                                  vals.v4 = vld4q_##sgn##bits(src + sj); \
                                  vst3q_##sgn##bits(dst3 + d3j, vals.v3); \
                                  vst1q_##sgn##bits(dst1 + d1j, vals.v4.val[3]); \
                              }

#endif 

#define SPLIT4ALPHA(sgn,bits) void split4(const Size2D &_size,                                          \
                                          const sgn##bits * srcBase, ptrdiff_t srcStride,               \
                                          sgn##bits * dst3Base, ptrdiff_t dst3Stride,                   \
                                          sgn##bits * dst1Base, ptrdiff_t dst1Stride)                   \
{                                                                                                       \
    internal::assertSupportedConfiguration();                                                           \
    Size2D size(_size);                                                                                 \
    if (srcStride == dst3Stride &&                                                                      \
        srcStride == dst1Stride &&                                                                      \
        srcStride == (ptrdiff_t)(size.width))                                                           \
    {                                                                                                   \
        size.width *= size.height;                                                                      \
        size.height = 1;                                                                                \
    }                                                                                                   \
    typedef internal::VecTraits<sgn##bits, 4>::vec128 vec128_4;                                         \
    typedef internal::VecTraits<sgn##bits, 3>::vec128 vec128_3;                                         \
    size_t roiw16 = size.width >= (16/sizeof(sgn##bits)-1) ? size.width - (16/sizeof(sgn##bits)-1) : 0; \
    typedef internal::VecTraits<sgn##bits, 4>::vec64 vec64_4;                                           \
    typedef internal::VecTraits<sgn##bits, 3>::vec64 vec64_3;                                           \
    size_t roiw8 = size.width >= (8/sizeof(sgn##bits)-1) ? size.width - (8/sizeof(sgn##bits)-1) : 0;    \
                                                                                                        \
    for (size_t i = 0u; i < size.height; ++i)                                                           \
    {                                                                                                   \
        const sgn##bits * src = internal::getRowPtr(srcBase, srcStride, i);                             \
        sgn##bits * dst3 = internal::getRowPtr(dst3Base, dst3Stride, i);                                \
        sgn##bits * dst1 = internal::getRowPtr(dst1Base, dst1Stride, i);                                \
        size_t sj = 0u, d3j = 0u, d1j = 0u;                                                             \
                                                                                                        \
        for (; d1j < roiw16; sj += MUL4(16)/sizeof(sgn##bits), d3j += MUL3(16)/sizeof(sgn##bits),       \
                                                               d1j += 16/sizeof(sgn##bits))             \
            ALPHA_QUAD(sgn, bits)                                                                       \
                                                                                                        \
        if (d1j < roiw8)                                                                                \
        {                                                                                               \
            union { vec64_4 v4; vec64_3 v3; } vals;                                                     \
            vals.v4 = vld4_##sgn##bits(src + sj);                                                       \
            vst3_u8(dst3 + d3j, vals.v3);                                                               \
            vst1_u8(dst1 + d1j, vals.v4.val[3]);                                                        \
            sj += MUL4(8)/sizeof(sgn##bits);                                                            \
            d3j += MUL3(8)/sizeof(sgn##bits);                                                           \
            d1j += 8/sizeof(sgn##bits);                                                                 \
        }                                                                                               \
                                                                                                        \
        for (; d1j < size.width; sj += 4, d3j += 3, ++d1j)                                              \
        {                                                                                               \
            dst3[d3j+0] = src[sj + 0];                                                                  \
            dst3[d3j+1] = src[sj + 1];                                                                  \
            dst3[d3j+2] = src[sj + 2];                                                                  \
            dst1[d1j]   = src[sj + 3];                                                                  \
        }                                                                                               \
    }                                                                                                   \
}

#else

#define VOID_LINE(type, n) (void)dst##n##Base; (void)dst##n##Stride;

#define SPLIT(sgn,bits,n) void split##n(const Size2D &size,                                          \
                                    const sgn##bits * srcBase, ptrdiff_t srcStride                   \
                                    FILL_LINES##n(FARG, sgn##bits) )                                 \
{                                                                                                    \
    internal::assertSupportedConfiguration();                                                        \
    (void)size;                                                                                      \
    (void)srcBase;                                                                                   \
    (void)srcStride;                                                                                 \
    FILL_LINES##n(VOID, sgn##bits)                                                                   \
}

#define SPLIT64(sgn,n) SPLIT(sgn,64,n)

#define SPLIT4ALPHA(sgn,bits) void split4(const Size2D &size,                                        \
                                          const sgn##bits * srcBase, ptrdiff_t srcStride,            \
                                          sgn##bits * dst3Base, ptrdiff_t dst3Stride,                \
                                          sgn##bits * dst1Base, ptrdiff_t dst1Stride)                \
{                                                                                                    \
    internal::assertSupportedConfiguration();                                                        \
    (void)size;                                                                                      \
    (void)srcBase;                                                                                   \
    (void)srcStride;                                                                                 \
    (void)dst3Base;                                                                                  \
    (void)dst3Stride;                                                                                \
    (void)dst1Base;                                                                                  \
    (void)dst1Stride;                                                                                \
}

#endif //CAROTENE_NEON

SPLIT(u, 8,2)
SPLIT(u, 8,3)
SPLIT(u, 8,4)
SPLIT(u,16,2)
SPLIT(u,16,3)
SPLIT(u,16,4)
SPLIT(s,32,2)
SPLIT(s,32,3)
SPLIT(s,32,4)

SPLIT64(s, 2)
SPLIT64(s, 3)
SPLIT64(s, 4)

SPLIT4ALPHA(u,8)

} // namespace CAROTENE_NS
