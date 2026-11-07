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

#include <limits>

namespace CAROTENE_NS {

s32 countNonZero(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw16 = size.width & ~15u;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u8* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        #define COUNTNONZERO8U_BLOCK_SIZE (16*255)
        uint8x16_t vc1 = vmovq_n_u8(1);
        for (; i < roiw16;)
        {
            size_t lim = std::min(i + COUNTNONZERO8U_BLOCK_SIZE, size.width) - 16;
            uint8x16_t vs = vmovq_n_u8(0);

            for (; i <= lim; i+= 16)
            {
                internal::prefetch(src + i);
                uint8x16_t vln = vld1q_u8(src + i);
                uint8x16_t vnz = vminq_u8(vln, vc1);
                vs = vaddq_u8(vs, vnz);
            }

            uint32x4_t vs4 = vpaddlq_u16(vpaddlq_u8(vs));
            uint32x2_t vs2 = vadd_u32(vget_low_u32(vs4), vget_high_u32(vs4));

            s32 s[2];
            vst1_u32((u32*)s, vs2);

            if (s[0] < 0 || s[1] < 0)//saturate in case of overflow ~ 2GB of non-zeros...
            {
                return 0x7fFFffFF;
            }
            result += (s[0] += s[1]);
            if (s[0] < 0 || result < 0)
            {
                return 0x7fFFffFF;
            }
        }
        for (; i < size.width; i++)
            result += (src[i] != 0)?1:0;
        if (result < 0)//saturate in case of overflow ~ 2GB of non-zeros...
        {
            return 0x7fFFffFF;
        }
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 countNonZero(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw8 = size.width & ~7u;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u16* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        #define COUNTNONZERO16U_BLOCK_SIZE (8*(256*256-1))
        uint16x8_t vc1 = vmovq_n_u16(1);
        for (; i < roiw8;)
        {
            size_t lim = std::min(i + COUNTNONZERO16U_BLOCK_SIZE, size.width) - 8;
            uint16x8_t vs = vmovq_n_u16(0);

            for (; i <= lim; i+= 8)
            {
                internal::prefetch(src + i);
                uint16x8_t vln = vld1q_u16(src + i);
                uint16x8_t vnz = vminq_u16(vln, vc1);
                vs = vaddq_u16(vs, vnz);
            }

            uint32x4_t vs4 = vpaddlq_u16(vs);
            uint32x2_t vs2 = vadd_u32(vget_low_u32(vs4), vget_high_u32(vs4));

            s32 s[2];
            vst1_u32((u32*)s, vs2);

            if (s[0] < 0 || s[1] < 0)//saturate in case of overflow ~ 4GB of non-zeros...
            {
                return 0x7fFFffFF;
            }
            result += (s[0] += s[1]);
            if (s[0] < 0 || result < 0)
            {
                return 0x7fFFffFF;
            }
        }
        for (; i < size.width; i++)
            result += (src[i] != 0)?1:0;
        if (result < 0)//saturate in case of overflow ~ 4GB of non-zeros...
        {
            return 0x7fFFffFF;
        }
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 countNonZero(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width & ~3u;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const u32* src = (const u32*)internal::getRowPtr( srcBase,  srcStride, k);
        u32 i = 0;

        uint32x4_t vc1 = vmovq_n_u32(1);
        uint32x4_t vs = vmovq_n_u32(0);

        for (; i < roiw4; i += 4 )
        {
            internal::prefetch(src + i);
            uint32x4_t vln = vld1q_u32(src + i);
            uint32x4_t vnz = vminq_u32(vln, vc1);
            vs = vqaddq_u32(vs, vnz);
        }

        uint32x2_t vs2 = vqadd_u32(vget_low_u32(vs), vget_high_u32(vs));

        s32 s[2];
        vst1_u32((u32*)s, vs2);

        if (s[0] < 0 || s[1] < 0)//saturate in case of overflow ~ 8GB of non-zeros...
        {
            return 0x7fFFffFF;
        }
        result += (s[0] += s[1]);
        if (s[0] < 0 || result < 0)
        {
            return 0x7fFFffFF;
        }

        for (; i < size.width; i++)
            result += (src[i] != 0)?1:0;
        if (result < 0)//saturate in case of overflow ~ 8GB of non-zeros...
        {
            return 0x7fFFffFF;
        }
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 countNonZero(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw4 = size.width & ~3u;
    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f32* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        float32x4_t vc0 = vmovq_n_f32(0);
        int32x4_t vs = vmovq_n_s32(0);

        for (; i < roiw4; i += 4 )
        {
            internal::prefetch(src + i);
            float32x4_t vln = vld1q_f32(src + i);
            int32x4_t vnz = vreinterpretq_s32_u32(vmvnq_u32(vceqq_f32(vln, vc0)));
            vs = vqaddq_s32(vs, vnz);
        }

        int32x2_t vs2 = vqneg_s32(vqadd_s32(vget_low_s32(vs), vget_high_s32(vs)));

        int s[2];
        vst1_s32(s, vs2);

        result += (s[0] += s[1]);
        if (s[0] < 0 || result < 0)//case of overflow ~ 8GB of non-zeros...
        {
            return 0x7fFFffFF;
        }

        for (; i < size.width; i++)
            result += (src[i] < std::numeric_limits<float>::min() && src[i] > -std::numeric_limits<float>::min())?0:1;

        if (result < 0)
        {
            return 0x7fFFffFF;
        }
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

s32 countNonZero(const Size2D &_size,
                 const f64 * srcBase, ptrdiff_t srcStride)
{
    internal::assertSupportedConfiguration();
#ifdef CAROTENE_NEON
    Size2D size(_size);
    if (srcStride == (ptrdiff_t)(size.width))
    {
        size.width *= size.height;
        size.height = 1;
    }
    size_t roiw8 = size.width & ~7u;
    size_t roiw4 = size.width & ~3u;
    size_t roiw2 = size.width & ~1u;
    uint64x2_t vmask1 = vdupq_n_u64(0x7fFFffFFffFFffFFULL); //will treat denormals as non-zero
    uint32x4_t vc0 = vmovq_n_u32(0);

    s32 result = 0;
    for(size_t k = 0; k < size.height; ++k)
    {
        const f64* src = internal::getRowPtr( srcBase,  srcStride, k);
        size_t i = 0;

        int32x2_t vs1 = vmov_n_s32(0);
        int32x2_t vs2 = vmov_n_s32(0);
        int32x2_t vs3 = vmov_n_s32(0);
        int32x2_t vs4 = vmov_n_s32(0);

        for (; i < roiw8; i += 8 )
        {
            internal::prefetch(src + i + 6);
            uint64x2_t vln1 = vld1q_u64((const u64*)(src + i));
            uint64x2_t vln2 = vld1q_u64((const u64*)(src + i + 2));
            uint64x2_t vln3 = vld1q_u64((const u64*)(src + i + 4));
            uint64x2_t vln4 = vld1q_u64((const u64*)(src + i + 6));

            uint64x2_t vm1 = vandq_u64(vln1, vmask1);
            uint64x2_t vm2 = vandq_u64(vln2, vmask1);
            uint64x2_t vm3 = vandq_u64(vln3, vmask1);
            uint64x2_t vm4 = vandq_u64(vln4, vmask1);

            uint32x4_t vequ1 = vceqq_u32(vreinterpretq_u32_u64(vm1), vc0);
            uint32x4_t vequ2 = vceqq_u32(vreinterpretq_u32_u64(vm2), vc0);
            uint32x4_t vequ3 = vceqq_u32(vreinterpretq_u32_u64(vm3), vc0);
            uint32x4_t vequ4 = vceqq_u32(vreinterpretq_u32_u64(vm4), vc0);

            uint32x4_t vlx1 = vmvnq_u32(vequ1);
            uint32x4_t vlx2 = vmvnq_u32(vequ2);
            uint32x4_t vlx3 = vmvnq_u32(vequ3);
            uint32x4_t vlx4 = vmvnq_u32(vequ4);

            int32x2_t vnz1 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx1), vget_high_u32(vlx1)));
            int32x2_t vnz2 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx2), vget_high_u32(vlx2)));
            int32x2_t vnz3 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx3), vget_high_u32(vlx3)));
            int32x2_t vnz4 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx4), vget_high_u32(vlx4)));

            vs1 = vqadd_s32(vs1, vnz1);
            vs2 = vqadd_s32(vs2, vnz2);
            vs3 = vqadd_s32(vs3, vnz3);
            vs4 = vqadd_s32(vs4, vnz4);
        }

        if (i < roiw4)
        {
            internal::prefetch(src + i + 2);
            uint64x2_t vln1 = vld1q_u64((const u64*)(src + i));
            uint64x2_t vln2 = vld1q_u64((const u64*)(src + i + 2));

            uint64x2_t vm1 = vandq_u64(vln1, vmask1);
            uint64x2_t vm2 = vandq_u64(vln2, vmask1);

            uint32x4_t vequ1 = vceqq_u32(vreinterpretq_u32_u64(vm1), vc0);
            uint32x4_t vequ2 = vceqq_u32(vreinterpretq_u32_u64(vm2), vc0);

            uint32x4_t vlx1 = vmvnq_u32(vequ1);
            uint32x4_t vlx2 = vmvnq_u32(vequ2);

            int32x2_t vnz1 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx1), vget_high_u32(vlx1)));
            int32x2_t vnz2 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx2), vget_high_u32(vlx2)));

            vs1 = vqadd_s32(vs1, vnz1);
            vs2 = vqadd_s32(vs2, vnz2);
            i += 4;
        }

        if (i < roiw2)
        {
            internal::prefetch(src + i);
            uint64x2_t vln1 = vld1q_u64((const u64*)(src + i));

            uint64x2_t vm1 = vandq_u64(vln1, vmask1);

            uint32x4_t vequ1 = vceqq_u32(vreinterpretq_u32_u64(vm1), vc0);

            uint32x4_t vlx1 = vmvnq_u32(vequ1);

            int32x2_t vnz1 = vreinterpret_s32_u32(vpmax_u32(vget_low_u32(vlx1), vget_high_u32(vlx1)));

            vs1 = vqadd_s32(vs1, vnz1);
            i += 2;
        }

        vs1 = vqadd_s32(vs1, vs2);
        vs3 = vqadd_s32(vs3, vs4);
        vs1 = vqadd_s32(vs1, vs3);
        int32x2_t vsneg = vqneg_s32(vs1);

        s32 s[2];
        vst1_s32(s, vsneg);

        result += (s[0] += s[1]);
        if (s[0] < 0 || result < 0)//case of overflow ~ 16GB of non-zeros...
        {
            return 0x7fFFffFF;
        }

        for (; i < size.width; i++)
            result += (src[i] < std::numeric_limits<double>::min() && src[i] > -std::numeric_limits<double>::min())?0:1;
        if (result < 0)
        {
            return 0x7fFFffFF;
        }
    }
    return result;
#else
    (void)_size;
    (void)srcBase;
    (void)srcStride;

    return 0;
#endif
}

} // namespace CAROTENE_NS
