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
#include "vtransform.hpp"

#include <cstring>

namespace CAROTENE_NS {

bool isFlipSupported(FLIP_MODE flipMode, u32 elemSize)
{
    bool supportedElemSize = (elemSize == 1) || (elemSize == 2) || (elemSize == 3) || (elemSize == 4);
    return isSupportedConfiguration() &&
            ((supportedElemSize && ((flipMode == FLIP_BOTH_MODE) || (flipMode == FLIP_HORIZONTAL_MODE))) ||
             (flipMode == FLIP_VERTICAL_MODE));
}

#ifdef CAROTENE_NEON

namespace {

template <typename T>
void flip(const Size2D & size,
          const void * srcBase, ptrdiff_t srcStride,
          void * dstBase, ptrdiff_t dstStride,
          FLIP_MODE flipMode)
{
    using namespace internal;

    typedef typename VecTraits<T>::vec128 vec128;
    typedef typename VecTraits<T>::vec64 vec64;

    u32 step_base = 16 / sizeof(T), step_tail = 8 / sizeof(T);
    size_t roiw_base = size.width >= (step_base - 1) ? size.width - step_base + 1 : 0;
    size_t roiw_tail = size.width >= (step_tail - 1) ? size.width - step_tail + 1 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src = getRowPtr((const T *)srcBase, srcStride, i);
        T * dst = getRowPtr((T *)dstBase, dstStride, (flipMode & FLIP_VERTICAL_MODE) != 0 ? size.height - i - 1 : i);
        size_t js = 0, jd = size.width;

        for (; js < roiw_base; js += step_base, jd -= step_base)
        {
            prefetch(src + js);

            vec128 v_src = vld1q(src + js);
            vec128 v_dst = vrev64q(v_src);
            v_dst = vcombine(vget_high(v_dst), vget_low(v_dst));
            vst1q(dst + jd - step_base, v_dst);
        }
        for (; js < roiw_tail; js += step_tail, jd -= step_tail)
        {
            vec64 v_src = vld1(src + js);
            vst1(dst + jd - step_tail, vrev64(v_src));
        }

        for (--jd; js < size.width; ++js, --jd)
            dst[jd] = src[js];
    }
}

template <typename T>
void flip3(const Size2D & size,
           const void * srcBase, ptrdiff_t srcStride,
           void * dstBase, ptrdiff_t dstStride,
           FLIP_MODE flipMode)
{
    using namespace internal;

#ifndef ANDROID
    typedef typename VecTraits<T, 3>::vec128 vec128;
#endif
    typedef typename VecTraits<T, 3>::vec64 vec64;

#ifndef ANDROID
    u32 step_base = 16 / sizeof(T), step_base3 = step_base * 3;
    size_t roiw_base = size.width >= (step_base - 1) ? size.width - step_base + 1 : 0;
#endif
    u32 step_tail = 8 / sizeof(T), step_tail3 = step_tail * 3;
    size_t roiw_tail = size.width >= (step_tail - 1) ? size.width - step_tail + 1 : 0;

    for (size_t i = 0; i < size.height; ++i)
    {
        const T * src = getRowPtr((const T *)srcBase, srcStride, i);
        T * dst = getRowPtr((T *)dstBase, dstStride, (flipMode & FLIP_VERTICAL_MODE) != 0 ? size.height - i - 1 : i);
        size_t j = 0, js = 0, jd = size.width * 3;

#ifndef ANDROID
        for (; j < roiw_base; j += step_base, js += step_base3, jd -= step_base3)
        {
            prefetch(src + js);

            vec128 v_src = vld3q(src + js), v_dst;
            v_src.val[0] = vrev64q(v_src.val[0]);
            v_src.val[1] = vrev64q(v_src.val[1]);
            v_src.val[2] = vrev64q(v_src.val[2]);

            v_dst.val[0] = vcombine(vget_high(v_src.val[0]), vget_low(v_src.val[0]));
            v_dst.val[1] = vcombine(vget_high(v_src.val[1]), vget_low(v_src.val[1]));
            v_dst.val[2] = vcombine(vget_high(v_src.val[2]), vget_low(v_src.val[2]));

            vst3q(dst + jd - step_base3, v_dst);
        }
#endif // ANDROID

        for (; j < roiw_tail; j += step_tail, js += step_tail3, jd -= step_tail3)
        {
            vec64 v_src = vld3(src + js), v_dst;
            v_dst.val[0] = vrev64(v_src.val[0]);
            v_dst.val[1] = vrev64(v_src.val[1]);
            v_dst.val[2] = vrev64(v_src.val[2]);

            vst3(dst + jd - step_tail3, v_dst);
        }

        for (jd -= 3; j < size.width; ++j, js += 3, jd -= 3)
        {
            dst[jd] = src[js];
            dst[jd + 1] = src[js + 1];
            dst[jd + 2] = src[js + 2];
        }
    }
}

typedef void (* flipFunc)(const Size2D &size,
                  const void * srcBase, ptrdiff_t srcStride,
                  void * dstBase, ptrdiff_t dstStride,
                  FLIP_MODE flipMode);

} // namespace

#endif

void flip(const Size2D &size,
          const u8 * srcBase, ptrdiff_t srcStride,
          u8 * dstBase, ptrdiff_t dstStride,
          FLIP_MODE flipMode, u32 elemSize)
{
    internal::assertSupportedConfiguration(isFlipSupported(flipMode, elemSize));
#ifdef CAROTENE_NEON

    if (flipMode == FLIP_VERTICAL_MODE)
    {
        for (size_t y = 0; y < size.height; ++y)
        {
            const u8 * src_row = internal::getRowPtr(srcBase, srcStride, y);
            u8 * dst_row = internal::getRowPtr(dstBase, dstStride, size.height - y - 1);

            std::memcpy(dst_row, src_row, elemSize * size.width);
        }
        return;
    }

    flipFunc func = NULL;

    if (elemSize == (u32)sizeof(u8))
        func = &flip<u8>;
    if (elemSize == (u32)sizeof(u16))
        func = &flip<u16>;
    if (elemSize == (u32)sizeof(u32))
        func = &flip<u32>;
    if (elemSize == (u32)sizeof(u8) * 3)
        func = &flip3<u8>;

    if (func == NULL)
        return;

    func(size,
         srcBase, srcStride,
         dstBase, dstStride,
         flipMode);

#else
    (void)size;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)flipMode;
    (void)elemSize;
#endif
}

} // namespace CAROTENE_NS
