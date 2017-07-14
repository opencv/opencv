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

#define FILL_LINES2(macro,type) \
            macro##_LINE(type,0) \
            macro##_LINE(type,1)
#define FILL_LINES3(macro,type) \
            FILL_LINES2(macro,type) \
            macro##_LINE(type,2)
#define FILL_LINES4(macro,type) \
            FILL_LINES3(macro,type) \
            macro##_LINE(type,3)

#define  FARG_LINE(type, n) , const type * src##n##Base, ptrdiff_t src##n##Stride

#ifdef CAROTENE_NEON

#define  VROW_LINE(type, n) const type * src##n = internal::getRowPtr(src##n##Base, src##n##Stride, i);
#define  PREF_LINE(type, n) internal::prefetch(src##n + sj);
#define VLD1Q_LINE(type, n) v_dst.val[n] = vld1q_##type(src##n + sj);
#define  PRLD_LINE(type, n) internal::prefetch(src##n + sj); v_dst.val[n] = vld1q_##type(src##n + sj);
#define  VLD1_LINE(type, n) v_dst.val[n] = vld1_##type(src##n + sj);
#define   SLD_LINE(type, n) dst[dj + n] = src##n[sj];

#define MUL2(val) (val << 1)
#define MUL3(val) (MUL2(val) + val)
#define MUL4(val) (val << 2)

#define CONTSRC2 dstStride == src0Stride && \
                 dstStride == src1Stride &&
#define CONTSRC3 dstStride == src0Stride && \
                 dstStride == src1Stride && \
                 dstStride == src2Stride &&
#define CONTSRC4 dstStride == src0Stride && \
                 dstStride == src1Stride && \
                 dstStride == src2Stride && \
                 dstStride == src3Stride &&

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ < 7

#define MERGE_ASM2(sgn, bits) __asm__ ( \
                                          "vld1." #bits " {d0-d1}, [%[in0]]             \n\t" \
                                          "vld1." #bits " {d2-d3}, [%[in1]]             \n\t" \
                                          "vst2." #bits " {d0, d2}, [%[out0]]           \n\t" \
                                          "vst2." #bits " {d1, d3}, [%[out1]]           \n\t" \
                                          : \
                                          : [in0] "r" (src0 + sj), [in1] "r" (src1 + sj), \
                                            [out0]  "r" (dst + dj), [out1]  "r" (dst + dj + MUL2(8)/sizeof(sgn##bits)) \
                                          : "d0","d1","d2","d3" \
                                      );
#define MERGE_ASM3(sgn, bits) __asm__ ( \
                                          "vld1." #bits " {d0-d1}, [%[in0]]             \n\t" \
                                          "vld1." #bits " {d2-d3}, [%[in1]]             \n\t" \
                                          "vld1." #bits " {d4-d5}, [%[in2]]             \n\t" \
                                          "vst3." #bits " {d0, d2, d4}, [%[out0]]       \n\t" \
                                          "vst3." #bits " {d1, d3, d5}, [%[out1]]       \n\t" \
                                          : \
                                          : [in0] "r" (src0 + sj), [in1] "r" (src1 + sj), [in2] "r" (src2 + sj), \
                                            [out0]  "r" (dst + dj), [out1]  "r" (dst + dj + MUL3(8)/sizeof(sgn##bits)) \
                                          : "d0","d1","d2","d3","d4","d5" \
                                      );
#define MERGE_ASM4(sgn, bits) __asm__ ( \
                                          "vld1." #bits " {d0-d1}, [%[in0]]             \n\t" \
                                          "vld1." #bits " {d2-d3}, [%[in1]]             \n\t" \
                                          "vld1." #bits " {d4-d5}, [%[in2]]             \n\t" \
                                          "vld1." #bits " {d6-d7}, [%[in3]]             \n\t" \
                                          "vst4." #bits " {d0, d2, d4, d6}, [%[out0]]   \n\t" \
                                          "vst4." #bits " {d1, d3, d5, d7}, [%[out1]]   \n\t" \
                                          : \
                                          : [in0] "r" (src0 + sj), [in1] "r" (src1 + sj), [in2] "r" (src2 + sj), [in3] "r" (src3 + sj), \
                                            [out0]  "r" (dst + dj), [out1]  "r" (dst + dj + MUL4(8)/sizeof(sgn##bits)) \
                                          : "d0","d1","d2","d3","d4","d5","d6","d7" \
                                      );

#define MERGE_QUAD(sgn, bits, n) { \
                                     FILL_LINES##n(PREF, sgn##bits) \
                                     MERGE_ASM##n(sgn, bits) \
                                 }

#else

#define MERGE_QUAD(sgn, bits, n) { \
                                     vec128 v_dst; \
                                     /*FILL_LINES##n(PREF, sgn##bits) \
                                     FILL_LINES##n(VLD1Q, sgn##bits)*/ \
                                     FILL_LINES##n(PRLD, sgn##bits) \
                                     vst##n##q_##sgn##bits(dst + dj, v_dst); \
                                 }

#endif

#define COMBINE(sgn,bits,n) void combine##n(const Size2D &_size                                             \
                                        FILL_LINES##n(FARG, sgn##bits),                                     \
                                        sgn##bits * dstBase, ptrdiff_t dstStride)                           \
{                                                                                                           \
    internal::assertSupportedConfiguration();                                                               \
    Size2D size(_size);                                                                                     \
    if (CONTSRC##n                                                                                          \
        dstStride == (ptrdiff_t)(size.width))                                                               \
    {                                                                                                       \
        size.width *= size.height;                                                                          \
        size.height = 1;                                                                                    \
    }                                                                                                       \
    typedef internal::VecTraits<sgn##bits, n>::vec128 vec128;                                               \
    size_t roiw16 = size.width >= (16/sizeof(sgn##bits) - 1) ? size.width - (16/sizeof(sgn##bits) - 1) : 0; \
    typedef internal::VecTraits<sgn##bits, n>::vec64 vec64;                                                 \
    size_t roiw8 = size.width >= (8/sizeof(sgn##bits) - 1) ? size.width - (8/sizeof(sgn##bits) - 1) : 0;    \
                                                                                                            \
    for (size_t i = 0u; i < size.height; ++i)                                                               \
    {                                                                                                       \
        FILL_LINES##n(VROW, sgn##bits)                                                                      \
        sgn##bits * dst = internal::getRowPtr(dstBase, dstStride, i);                                       \
        size_t sj = 0u, dj = 0u;                                                                            \
                                                                                                            \
        for (; sj < roiw16; sj += 16/sizeof(sgn##bits), dj += MUL##n(16)/sizeof(sgn##bits))                 \
            MERGE_QUAD(sgn, bits, n)                                                                        \
                                                                                                            \
        if ( sj < roiw8 )                                                                                   \
        {                                                                                                   \
            vec64 v_dst;                                                                                    \
            FILL_LINES##n(VLD1, sgn##bits)                                                                  \
            vst##n##_##sgn##bits(dst + dj, v_dst);                                                          \
            sj += 8/sizeof(sgn##bits); dj += MUL##n(8)/sizeof(sgn##bits);                                   \
        }                                                                                                   \
                                                                                                            \
        for (; sj < size.width; ++sj, dj += n)                                                              \
        {                                                                                                   \
            FILL_LINES##n(SLD, sgn##bits)                                                                   \
        }                                                                                                   \
    }                                                                                                       \
}

#define COMBINE64(sgn,n) void combine##n(const Size2D &_size                                                \
                                               FILL_LINES##n(FARG, sgn##64),                                \
                                               sgn##64 * dstBase, ptrdiff_t dstStride)                      \
{                                                                                                           \
    internal::assertSupportedConfiguration();                                                               \
    Size2D size(_size);                                                                                     \
    if (CONTSRC##n                                                                                          \
        dstStride == (ptrdiff_t)(size.width))                                                               \
    {                                                                                                       \
        size.width *= size.height;                                                                          \
        size.height = 1;                                                                                    \
    }                                                                                                       \
    typedef internal::VecTraits<sgn##64, n>::vec64 vec64;                                                   \
                                                                                                            \
    for (size_t i = 0u; i < size.height; ++i)                                                               \
    {                                                                                                       \
        FILL_LINES##n(VROW, sgn##64)                                                                        \
        sgn##64 * dst = internal::getRowPtr(dstBase, dstStride, i);                                         \
        size_t sj = 0u, dj = 0u;                                                                            \
                                                                                                            \
        for (; sj < size.width; ++sj, dj += n)                                                              \
        {                                                                                                   \
            vec64 v_dst;                                                                                    \
            FILL_LINES##n(VLD1, sgn##64)                                                                    \
            vst##n##_##sgn##64(dst + dj, v_dst);                                                            \
            /*FILL_LINES##n(SLD, sgn##64)*/                                                                 \
        }                                                                                                   \
    }                                                                                                       \
}

#else

#define  VOID_LINE(type, n) (void)src##n##Base; (void)src##n##Stride;

#define COMBINE(sgn,bits,n) void combine##n(const Size2D &size                                              \
                                        FILL_LINES##n(FARG, sgn##bits),                                     \
                                        sgn##bits * dstBase, ptrdiff_t dstStride)                           \
{                                                                                                           \
    internal::assertSupportedConfiguration();                                                               \
    (void)size;                                                                                             \
    FILL_LINES##n(VOID, sgn##bits)                                                                          \
    (void)dstBase;                                                                                          \
    (void)dstStride;                                                                                        \
}
#define COMBINE64(sgn,n) COMBINE(sgn,64,n)

#endif //CAROTENE_NEON

COMBINE(u, 8,2)
COMBINE(u, 8,3)
COMBINE(u, 8,4)
COMBINE(u,16,2)
COMBINE(u,16,3)
COMBINE(u,16,4)
COMBINE(s,32,2)
COMBINE(s,32,3)
COMBINE(s,32,4)
COMBINE64(s, 2)
COMBINE64(s, 3)
COMBINE64(s, 4)

void combineYUYV(const Size2D &size,
                 const u8 * srcyBase, ptrdiff_t srcyStride,
                 const u8 * srcuBase, ptrdiff_t srcuStride,
                 const u8 * srcvBase, ptrdiff_t srcvStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#ifndef __ANDROID__
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; i += 1)
    {
        const u8 * srcy = internal::getRowPtr(srcyBase, srcyStride, i);
        const u8 * srcu = internal::getRowPtr(srcuBase, srcuStride, i);
        const u8 * srcv = internal::getRowPtr(srcvBase, srcvStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t syj = 0u, sj = 0u, dj = 0u;

#ifndef __ANDROID__
        for (; sj < roiw32; sj += 32, syj += 64, dj += 128)
        {
            internal::prefetch(srcy + syj);
            internal::prefetch(srcu + sj);
            internal::prefetch(srcv + sj);

            uint8x16x2_t v_y = vld2q_u8(srcy + syj);
            uint8x16x4_t v_dst;
            v_dst.val[0] = v_y.val[0];
            v_dst.val[1] = vld1q_u8(srcu + sj);
            v_dst.val[2] = v_y.val[1];
            v_dst.val[3] = vld1q_u8(srcv + sj);
            vst4q_u8(dst + dj, v_dst);

            v_y = vld2q_u8(srcy + syj + 32);
            v_dst.val[0] = v_y.val[0];
            v_dst.val[1] = vld1q_u8(srcu + sj + 16);
            v_dst.val[2] = v_y.val[1];
            v_dst.val[3] = vld1q_u8(srcv + sj + 16);
            vst4q_u8(dst + dj + 64, v_dst);
        }
#endif

        for (; sj < roiw8; sj += 8, syj += 16, dj += 32)
        {
            uint8x8x2_t v_y = vld2_u8(srcy + syj);
            uint8x8x4_t v_dst;
            v_dst.val[0] = v_y.val[0];
            v_dst.val[1] = vld1_u8(srcu + sj);
            v_dst.val[2] = v_y.val[1];
            v_dst.val[3] = vld1_u8(srcv + sj);
            vst4_u8(dst + dj, v_dst);
        }

        for (; sj < size.width; ++sj, syj += 2, dj += 4)
        {
            dst[dj] = srcy[syj];
            dst[dj + 1] = srcu[sj];
            dst[dj + 2] = srcy[syj + 1];
            dst[dj + 3] = srcv[sj];
        }
    }
#else
    (void)size;
    (void)srcyBase;
    (void)srcyStride;
    (void)srcuBase;
    (void)srcuStride;
    (void)srcvBase;
    (void)srcvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

void combineUYVY(const Size2D &size,
                 const u8 * srcyBase, ptrdiff_t srcyStride,
                 const u8 * srcuBase, ptrdiff_t srcuStride,
                 const u8 * srcvBase, ptrdiff_t srcvStride,
                 u8 * dstBase, ptrdiff_t dstStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
#ifndef __ANDROID__
    size_t roiw32 = size.width >= 31 ? size.width - 31 : 0;
#endif
    size_t roiw8 = size.width >= 7 ? size.width - 7 : 0;

    for (size_t i = 0u; i < size.height; ++i)
    {
        const u8 * srcy = internal::getRowPtr(srcyBase, srcyStride, i);
        const u8 * srcu = internal::getRowPtr(srcuBase, srcuStride, i);
        const u8 * srcv = internal::getRowPtr(srcvBase, srcvStride, i);
        u8 * dst = internal::getRowPtr(dstBase, dstStride, i);
        size_t syj = 0u, sj = 0u, dj = 0u;

#ifndef __ANDROID__
        for (; sj < roiw32; sj += 32, syj += 64, dj += 128)
        {
            internal::prefetch(srcy + syj);
            internal::prefetch(srcu + sj);
            internal::prefetch(srcv + sj);

            uint8x16x2_t v_y = vld2q_u8(srcy + syj);
            uint8x16x4_t v_dst;
            v_dst.val[0] = vld1q_u8(srcu + sj);
            v_dst.val[1] = v_y.val[0];
            v_dst.val[2] = vld1q_u8(srcv + sj);
            v_dst.val[3] = v_y.val[1];
            vst4q_u8(dst + dj, v_dst);

            v_y = vld2q_u8(srcy + syj + 32);
            v_dst.val[0] = vld1q_u8(srcu + sj + 16);
            v_dst.val[1] = v_y.val[0];
            v_dst.val[2] = vld1q_u8(srcv + sj + 16);
            v_dst.val[3] = v_y.val[1];
            vst4q_u8(dst + dj + 64, v_dst);
        }
#endif

        for (; sj < roiw8; sj += 8, syj += 16, dj += 32)
        {
            uint8x8x2_t v_y = vld2_u8(srcy + syj);
            uint8x8x4_t v_dst;
            v_dst.val[0] = vld1_u8(srcu + sj);
            v_dst.val[1] = v_y.val[0];
            v_dst.val[2] = vld1_u8(srcv + sj);
            v_dst.val[3] = v_y.val[1];
            vst4_u8(dst + dj, v_dst);
        }

        for (; sj < size.width; ++sj, syj += 2, dj += 4)
        {
            dst[dj] = srcu[sj];
            dst[dj + 1] = srcy[syj];
            dst[dj + 2] = srcv[sj];
            dst[dj + 3] = srcy[syj + 1];
        }
    }
#else
    (void)size;
    (void)srcyBase;
    (void)srcyStride;
    (void)srcuBase;
    (void)srcuStride;
    (void)srcvBase;
    (void)srcvStride;
    (void)dstBase;
    (void)dstStride;
#endif
}

} // namespace CAROTENE_NS
