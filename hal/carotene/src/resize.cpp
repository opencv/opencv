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

#include "common.hpp"
#include "vtransform.hpp"

#include <cmath>
#include <vector>
#include <algorithm>

namespace CAROTENE_NS {

bool isResizeNearestNeighborSupported(const Size2D &ssize, u32 elemSize)
{
#if SIZE_MAX <= UINT32_MAX
    (void)ssize;
#endif
    bool supportedElemSize = (elemSize == 1) || (elemSize == 3) || (elemSize == 4);
    return isSupportedConfiguration()
#if SIZE_MAX > UINT32_MAX
           && !(ssize.width > 0xffffFFFF || ssize.height > 0xffffFFFF)// Restrict image size since internally used resizeGeneric performs
                                                                      // index evaluation with u32
#endif
           && supportedElemSize;
}

bool isResizeAreaSupported(f32 wr, f32 hr, u32 channels)
{
    bool supportedRatio = false;

    if (channels == 1)
        supportedRatio = (hr == wr) && ((wr == 2.0f) || (wr == 4.0f) || (wr == 0.5));
    else if (channels == 3)
        supportedRatio = (hr == wr) && ((wr == 2.0f) || (wr == 4.0f) || (wr == 0.5f));
    else if (channels == 4)
        supportedRatio = (hr == wr) && ((wr == 2.0f) || (wr == 4.0f) || (wr == 0.5f));

    return isSupportedConfiguration() && supportedRatio;
}

bool isResizeLinearSupported(const Size2D &ssize, const Size2D &dsize,
                             f32 wr, f32 hr, u32 channels)
{
    if ((wr <= 2.0f) && (hr <= 2.0f))
    {
        bool channelsSupport = (channels == 1) || (channels == 3) || (channels == 4);
        return (ssize.width >= 16) && (dsize.height >= 8) &&
                (dsize.width >= 8) && channelsSupport;
    }

    return false;
}

bool isResizeLinearOpenCVSupported(const Size2D &ssize, const Size2D &dsize, u32 channels)
{
    switch(channels)
    {
    case 1:
        if (ssize.width >= 8
#if SIZE_MAX > UINT32_MAX
            && !(ssize.width > 0xffffFFFF || ssize.height > 0xffffFFFF)// Restrict image size since internal index evaluation
                                                                       // is performed with u32
#endif
            && dsize.width >= 8 && dsize.height >= 8)
            return isSupportedConfiguration();
        return false;
    case 4:
        if (ssize.width >= 2
#if SIZE_MAX > UINT32_MAX
            && !(ssize.width > 0xffffFFFF || ssize.height > 0xffffFFFF)// Restrict image size since internal index evaluation
                                                                       // is performed with u32
#endif
            && dsize.width >= 2 && dsize.height >= 8
            && (2*dsize.width != ssize.width || 2*dsize.height != ssize.height)) // 2x downscaling is performed as area in OpenCV which differs from this implementation
            return isSupportedConfiguration();
        return false;
    default:
        return false;
    };
}

#ifdef CAROTENE_NEON

namespace {

u32 * calcLUT(size_t size, f32 ratio,
              std::vector<u32> & _ofs)
{
    _ofs.resize(size);
    u32 * ofs = &_ofs[0];

    size_t roiw8 = size >= 7 ? size - 7 : 0;
    size_t roiw4 = size >= 3 ? size - 3 : 0;
    size_t x = 0;

    f32 indeces[4] = { 0, 1, 2, 3 };
    float32x4_t v_index = vld1q_f32(indeces), v_inc = vdupq_n_f32(4);
    float32x4_t v_05 = vdupq_n_f32(0.5f), v_ratio = vdupq_n_f32(ratio);

    for ( ; x < roiw8; x += 8)
    {
        float32x4_t v_dstf = vmulq_f32(vaddq_f32(v_index, v_05), v_ratio);
        vst1q_u32(ofs + x, vcvtq_u32_f32(v_dstf));
        v_index = vaddq_f32(v_index, v_inc);

        v_dstf = vmulq_f32(vaddq_f32(v_index, v_05), v_ratio);
        vst1q_u32(ofs + x + 4, vcvtq_u32_f32(v_dstf));
        v_index = vaddq_f32(v_index, v_inc);
    }

    for ( ; x < roiw4; x += 4)
    {
        float32x4_t v_dstf = vmulq_f32(vaddq_f32(v_index, v_05), v_ratio);
        vst1q_u32(ofs + x, vcvtq_u32_f32(v_dstf));
        v_index = vaddq_f32(v_index, v_inc);
    }

    for ( ; x < size; ++x)
    {
        ofs[x] = static_cast<u32>(floorf((x + 0.5f) * ratio));
    }

    return ofs;
}

template <typename T>
void resizeGeneric(const Size2D &dsize,
                   const void * srcBase, ptrdiff_t srcStride,
                   void * dstBase, ptrdiff_t dstStride,
                   f32 wr, f32 hr)
{
    std::vector<u32> _x_ofs;
    u32 * x_ofs = calcLUT(dsize.width, wr, _x_ofs);//32bit LUT is used so we could get issues on src image dimensions greater than (2^32-1)

    for (size_t dst_y = 0; dst_y < dsize.height; ++dst_y)
    {
        size_t src_y = static_cast<size_t>(floorf((dst_y + 0.5f) * hr));
        const T * src = internal::getRowPtr(static_cast<const T *>(srcBase), srcStride, src_y);
        T * dst = internal::getRowPtr(static_cast<T *>(dstBase), dstStride, dst_y);

        for (size_t dst_x = 0; dst_x < dsize.width; ++dst_x)
        {
            internal::prefetch(src + dst_x);
            dst[dst_x] = src[x_ofs[dst_x]];
        }
    }
}

typedef struct _24bit_
{
    u8 a[3];
} _24bit;

} // namespace


#endif

void resizeNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                           const void * srcBase, ptrdiff_t srcStride,
                           void * dstBase, ptrdiff_t dstStride,
                           f32 wr, f32 hr, u32 elemSize)
{
    internal::assertSupportedConfiguration(wr > 0 && hr > 0 &&
                                           (dsize.width - 0.5) * wr < ssize.width &&
                                           (dsize.height - 0.5) * hr < ssize.height &&  // Ensure we have enough source data
                                           (dsize.width + 0.5) * wr >= ssize.width &&
                                           (dsize.height + 0.5) * hr >= ssize.height && // Ensure source isn't too big
                                           isResizeNearestNeighborSupported(ssize, elemSize));
#ifdef CAROTENE_NEON

    if (elemSize == 1)
    {
        resizeGeneric<u8>(dsize,
                          srcBase, srcStride,
                          dstBase, dstStride,
                          wr, hr);
    }
    else if (elemSize == 3)
    {
        resizeGeneric<_24bit>(dsize,
                              srcBase, srcStride,
                              dstBase, dstStride,
                              wr, hr);
    }
    else if (elemSize == 4)
    {
        resizeGeneric<u32>(dsize,
                           srcBase, srcStride,
                           dstBase, dstStride,
                           wr, hr);
    }

#else
    (void)dsize;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)wr;
    (void)hr;
#endif
}

#ifdef CAROTENE_NEON
template <bool opencv_like, int shiftsize>
inline uint8x8_t areaDownsamplingDivision(uint16x8_t data)
{
    return vshrn_n_u16(data, shiftsize);
}
template <>
inline uint8x8_t areaDownsamplingDivision<true,2>(uint16x8_t data)
{
    // rounding
    return vrshrn_n_u16(data,2);
}
template <>
inline uint8x8_t areaDownsamplingDivision<true,4>(uint16x8_t data)
{
    // bankers rounding
    return vrshrn_n_u16(vqsubq_u16(data, vshrq_n_u16(vbicq_u16(vdupq_n_u16(1<<4), data), 4)),4);
}

template <bool opencv_like, int shiftsize>
inline u8 areaDownsamplingDivision(u16 data)
{
    return data >> shiftsize;
}
template <>
inline u8 areaDownsamplingDivision<true,2>(u16 data)
{
    // rounding
    return (data + 2) >> 2;
}
template <>
inline u8 areaDownsamplingDivision<true,4>(u16 data)
{
    // bankers rounding
    return (data - (((1<<4) & ~data) >> 4) + 8) >> 4;
}
#endif

template <bool opencv_like>
inline void resizeAreaRounding(const Size2D &ssize, const Size2D &dsize,
                               const u8 * srcBase, ptrdiff_t srcStride,
                               u8 * dstBase, ptrdiff_t dstStride,
                               f32 wr, f32 hr, u32 channels)
{
    internal::assertSupportedConfiguration(isResizeAreaSupported(wr, hr, channels) &&
                                           std::abs(dsize.width  * wr -  ssize.width) < 0.1 &&
                                           std::abs(dsize.height * hr - ssize.height) < 0.1);
#ifdef CAROTENE_NEON
    if (channels == 1)
    {
        if ((wr == 2.0f) && (hr == 2.0f))
        {
            size_t roiw8 = dsize.width >= 7 ? dsize.width - 7 : 0;

            for (size_t i = 0; i < dsize.height; ++i)
            {
                const u8 * src0_row = internal::getRowPtr(srcBase, srcStride, i << 1);
                const u8 * src1_row = internal::getRowPtr(srcBase, srcStride, (i << 1) + 1);
                u8 * dst_row = internal::getRowPtr(dstBase, dstStride, i);
                size_t sj = 0, dj = 0;

                for ( ; dj < roiw8; dj += 8, sj += 16)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);

                    uint16x8_t vSum1 = vpaddlq_u8(vld1q_u8(src0_row + sj));
                    uint16x8_t vSum2 = vpaddlq_u8(vld1q_u8(src1_row + sj));
                    uint8x8_t vRes1 = areaDownsamplingDivision<opencv_like,2>(vaddq_u16(vSum1, vSum2));

                    vst1_u8(dst_row + dj, vRes1);
                }

                for ( ; dj < dsize.width; ++dj, sj += 2)
                {
                    dst_row[dj] = areaDownsamplingDivision<opencv_like,2>(
                                      (u16)src0_row[sj] + src0_row[sj + 1] +
                                      src1_row[sj] + src1_row[sj + 1]);
                }
            }
        }
        else if ((wr == 0.5f) && (hr == 0.5f))
        {
            size_t roiw32 = dsize.width >= 31 ? dsize.width - 31 : 0;
            size_t roiw16 = dsize.width >= 15 ? dsize.width - 15 : 0;

            for (size_t i = 0; i < dsize.height; i += 2)
            {
                const u8 * src_row = internal::getRowPtr(srcBase, srcStride, i >> 1);
                u8 * dst0_row = internal::getRowPtr(dstBase, dstStride, i);
                u8 * dst1_row = internal::getRowPtr(dstBase, dstStride, std::min(i + 1, dsize.height - 1));
                size_t sj = 0, dj = 0;

                for ( ; dj < roiw32; dj += 32, sj += 16)
                {
                    internal::prefetch(src_row + sj);

                    uint8x16x2_t v_dst;
                    v_dst.val[0] = v_dst.val[1] = vld1q_u8(src_row + sj);

                    vst2q_u8(dst0_row + dj, v_dst);
                    vst2q_u8(dst1_row + dj, v_dst);
                }

                for ( ; dj < roiw16; dj += 16, sj += 8)
                {
                    uint8x8x2_t v_dst;
                    v_dst.val[0] = v_dst.val[1] = vld1_u8(src_row + sj);

                    vst2_u8(dst0_row + dj, v_dst);
                    vst2_u8(dst1_row + dj, v_dst);
                }

                for ( ; dj < dsize.width; dj += 2, ++sj)
                {
                    u8 src_val = src_row[sj];
                    dst0_row[dj] = dst0_row[dj + 1] = src_val;
                    dst1_row[dj] = dst1_row[dj + 1] = src_val;
                }
            }
        }
        else //if ((wr == 4.0f) && (hr == 4.0f)) //the only scale that lasts after isSupported check
        {
#ifndef __ANDROID__
            size_t roiw16 = dsize.width >= 15 ? dsize.width - 15 : 0;
#endif
            size_t roiw8 = dsize.width >= 7 ? dsize.width - 7 : 0;

            for (size_t i = 0; i < dsize.height; ++i)
            {
                const u8 * src0_row = internal::getRowPtr(srcBase, srcStride, i << 2);
                const u8 * src1_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 1);
                const u8 * src2_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 2);
                const u8 * src3_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 3);
                u8 * dst_row = internal::getRowPtr(dstBase, dstStride, i);
                size_t sj = 0, dj = 0;

#ifndef __ANDROID__
                for ( ; dj < roiw16; dj += 16, sj += 64)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);
                    internal::prefetch(src2_row + sj);
                    internal::prefetch(src3_row + sj);

                    uint8x16x4_t vLane1 = vld4q_u8(src0_row + sj);
                    uint8x16x4_t vLane2 = vld4q_u8(src1_row + sj);
                    uint8x16x4_t vLane3 = vld4q_u8(src2_row + sj);
                    uint8x16x4_t vLane4 = vld4q_u8(src3_row + sj);

                    uint16x8_t vSum_0 = vaddl_u8(vget_low_u8(vLane1.val[0]), vget_low_u8(vLane1.val[1]));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane1.val[2]), vget_low_u8(vLane1.val[3])));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane2.val[0]), vget_low_u8(vLane2.val[1])));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane2.val[2]), vget_low_u8(vLane2.val[3])));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane3.val[0]), vget_low_u8(vLane3.val[1])));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane3.val[2]), vget_low_u8(vLane3.val[3])));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane4.val[0]), vget_low_u8(vLane4.val[1])));
                    vSum_0 = vaddq_u16(vSum_0, vaddl_u8(vget_low_u8(vLane4.val[2]), vget_low_u8(vLane4.val[3])));

                    uint16x8_t vSum_1 = vaddl_u8(vget_high_u8(vLane1.val[0]), vget_high_u8(vLane1.val[1]));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane1.val[2]), vget_high_u8(vLane1.val[3])));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane2.val[0]), vget_high_u8(vLane2.val[1])));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane2.val[2]), vget_high_u8(vLane2.val[3])));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane3.val[0]), vget_high_u8(vLane3.val[1])));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane3.val[2]), vget_high_u8(vLane3.val[3])));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane4.val[0]), vget_high_u8(vLane4.val[1])));
                    vSum_1 = vaddq_u16(vSum_1, vaddl_u8(vget_high_u8(vLane4.val[2]), vget_high_u8(vLane4.val[3])));

                    uint8x8_t vRes_0 = areaDownsamplingDivision<opencv_like,4>(vSum_0);
                    uint8x8_t vRes_1 = areaDownsamplingDivision<opencv_like,4>(vSum_1);

                    vst1q_u8(dst_row + dj, vcombine_u8(vRes_0, vRes_1));
                }
#endif

                for ( ; dj < roiw8; dj += 8, sj += 32)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);
                    internal::prefetch(src2_row + sj);
                    internal::prefetch(src3_row + sj);

                    uint8x8x4_t vLane1 = vld4_u8(src0_row + sj);
                    uint8x8x4_t vLane2 = vld4_u8(src1_row + sj);
                    uint8x8x4_t vLane3 = vld4_u8(src2_row + sj);
                    uint8x8x4_t vLane4 = vld4_u8(src3_row + sj);

                    uint16x8_t vSum = vaddl_u8(vLane1.val[0], vLane1.val[1]);
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane1.val[2], vLane1.val[3]));
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane2.val[0], vLane2.val[1]));
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane2.val[2], vLane2.val[3]));
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane3.val[0], vLane3.val[1]));
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane3.val[2], vLane3.val[3]));
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane4.val[0], vLane4.val[1]));
                    vSum = vaddq_u16(vSum, vaddl_u8(vLane4.val[2], vLane4.val[3]));

                    vst1_u8(dst_row + dj, (areaDownsamplingDivision<opencv_like,4>(vSum)));
                }

                for ( ; dj < dsize.width; ++dj, sj += 4)
                {
                    dst_row[dj] = areaDownsamplingDivision<opencv_like,4>(
                                      (u16)src0_row[sj] + src0_row[sj + 1] + src0_row[sj + 2] + src0_row[sj + 3] +
                                      src1_row[sj] + src1_row[sj + 1] + src1_row[sj + 2] + src1_row[sj + 3] +
                                      src2_row[sj] + src2_row[sj + 1] + src2_row[sj + 2] + src2_row[sj + 3] +
                                      src3_row[sj] + src3_row[sj + 1] + src3_row[sj + 2] + src3_row[sj + 3]);
                }
            }
        }
    }
    else if (channels == 4)
    {
        if ((wr == 2.0f) && (hr == 2.0f))
        {
#ifndef __ANDROID__
            size_t roiw4 = dsize.width >= 3 ? (dsize.width - 3) << 2 : 0;
#endif
            size_t roiw2 = dsize.width >= 1 ? (dsize.width - 1) << 2 : 0;

            for (size_t i = 0; i < dsize.height; ++i)
            {
                const u8 * src0_row = internal::getRowPtr(srcBase, srcStride, i << 1);
                const u8 * src1_row = internal::getRowPtr(srcBase, srcStride, (i << 1) + 1);
                u8 * dst_row = internal::getRowPtr(dstBase, dstStride, i);
                size_t sj = 0, dj = 0;

#ifndef __ANDROID__
                for ( ; dj < roiw4; dj += 16, sj += 32)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);

                    uint8x8_t vRes_0, vRes_1;

                    {
                        uint8x16_t vLane1 = vld1q_u8(src0_row + sj);
                        uint8x16_t vLane2 = vld1q_u8(src1_row + sj);

                        uint16x8_t vLane_l = vaddl_u8(vget_low_u8(vLane1), vget_low_u8(vLane2));
                        uint16x8_t vLane_h = vaddl_u8(vget_high_u8(vLane1), vget_high_u8(vLane2));

                        uint16x4_t vSum_l = vadd_u16(vget_low_u16(vLane_l), vget_high_u16(vLane_l));
                        uint16x4_t vSum_h = vadd_u16(vget_low_u16(vLane_h), vget_high_u16(vLane_h));

                        vRes_0 = areaDownsamplingDivision<opencv_like,2>(vcombine_u16(vSum_l, vSum_h));
                    }

                    {
                        uint8x16_t vLane1 = vld1q_u8(src0_row + sj + 16);
                        uint8x16_t vLane2 = vld1q_u8(src1_row + sj + 16);

                        uint16x8_t vLane_l = vaddl_u8(vget_low_u8(vLane1), vget_low_u8(vLane2));
                        uint16x8_t vLane_h = vaddl_u8(vget_high_u8(vLane1), vget_high_u8(vLane2));

                        uint16x4_t vSum_l = vadd_u16(vget_low_u16(vLane_l), vget_high_u16(vLane_l));
                        uint16x4_t vSum_h = vadd_u16(vget_low_u16(vLane_h), vget_high_u16(vLane_h));

                        vRes_1 = areaDownsamplingDivision<opencv_like,2>(vcombine_u16(vSum_l, vSum_h));
                    }

                    vst1q_u8(dst_row + dj, vcombine_u8(vRes_0, vRes_1));
                }
#endif

                for ( ; dj < roiw2; dj += 8, sj += 16)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);

                    uint8x16_t vLane1 = vld1q_u8(src0_row + sj);
                    uint8x16_t vLane2 = vld1q_u8(src1_row + sj);

                    uint16x8_t vLane_l = vaddl_u8(vget_low_u8(vLane1), vget_low_u8(vLane2));
                    uint16x8_t vLane_h = vaddl_u8(vget_high_u8(vLane1), vget_high_u8(vLane2));

                    uint16x4_t vSum_l = vadd_u16(vget_low_u16(vLane_l), vget_high_u16(vLane_l));
                    uint16x4_t vSum_h = vadd_u16(vget_low_u16(vLane_h), vget_high_u16(vLane_h));

                    uint8x8_t vRes = areaDownsamplingDivision<opencv_like,2>(vcombine_u16(vSum_l, vSum_h));
                    vst1_u8(dst_row + dj, vRes);
                }

                for (size_t dwidth = dsize.width << 2; dj < dwidth; dj += 4, sj += 8)
                {
                    dst_row[dj    ] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj    ] + src0_row[sj + 4] +
                                               src1_row[sj    ] + src1_row[sj + 4]);
                    dst_row[dj + 1] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj + 1] + src0_row[sj + 5] +
                                               src1_row[sj + 1] + src1_row[sj + 5]);
                    dst_row[dj + 2] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj + 2] + src0_row[sj + 6] +
                                               src1_row[sj + 2] + src1_row[sj + 6]);
                    dst_row[dj + 3] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj + 3] + src0_row[sj + 7] +
                                               src1_row[sj + 3] + src1_row[sj + 7]);
                }
            }
        }
        else if ((wr == 0.5f) && (hr == 0.5f))
        {
#ifndef __ANDROID__
            size_t roiw32 = dsize.width >= 31 ? (dsize.width - 31) << 2 : 0;
#endif
            size_t roiw16 = dsize.width >= 15 ? (dsize.width - 15) << 2 : 0;

            for (size_t i = 0; i < dsize.height; i += 2)
            {
                const u8 * src_row = internal::getRowPtr(srcBase, srcStride, i >> 1);
                u8 * dst0_row = internal::getRowPtr(dstBase, dstStride, i);
                u8 * dst1_row = internal::getRowPtr(dstBase, dstStride, std::min(i + 1, dsize.height - 1));
                size_t sj = 0, dj = 0;

#ifndef __ANDROID__
                for ( ; dj < roiw32; dj += 128, sj += 64)
                {
                    internal::prefetch(src_row + sj);

                    uint8x16x4_t v_src = vld4q_u8(src_row + sj);
                    uint8x16x2_t v_c0 = vzipq_u8(v_src.val[0], v_src.val[0]);
                    uint8x16x2_t v_c1 = vzipq_u8(v_src.val[1], v_src.val[1]);
                    uint8x16x2_t v_c2 = vzipq_u8(v_src.val[2], v_src.val[2]);
                    uint8x16x2_t v_c3 = vzipq_u8(v_src.val[3], v_src.val[3]);

                    uint8x16x4_t v_dst;
                    v_dst.val[0] = v_c0.val[0];
                    v_dst.val[1] = v_c1.val[0];
                    v_dst.val[2] = v_c2.val[0];
                    v_dst.val[3] = v_c3.val[0];
                    vst4q_u8(dst0_row + dj, v_dst);
                    vst4q_u8(dst1_row + dj, v_dst);

                    v_dst.val[0] = v_c0.val[1];
                    v_dst.val[1] = v_c1.val[1];
                    v_dst.val[2] = v_c2.val[1];
                    v_dst.val[3] = v_c3.val[1];
                    vst4q_u8(dst0_row + dj + 64, v_dst);
                    vst4q_u8(dst1_row + dj + 64, v_dst);
                }
#endif

                for ( ; dj < roiw16; dj += 64, sj += 32)
                {
                    internal::prefetch(src_row + sj);

                    uint8x8x4_t v_src = vld4_u8(src_row + sj);
                    uint8x8x2_t v_c0 = vzip_u8(v_src.val[0], v_src.val[0]);
                    uint8x8x2_t v_c1 = vzip_u8(v_src.val[1], v_src.val[1]);
                    uint8x8x2_t v_c2 = vzip_u8(v_src.val[2], v_src.val[2]);
                    uint8x8x2_t v_c3 = vzip_u8(v_src.val[3], v_src.val[3]);

                    uint8x16x4_t v_dst;
                    v_dst.val[0] = vcombine_u8(v_c0.val[0], v_c0.val[1]);
                    v_dst.val[1] = vcombine_u8(v_c1.val[0], v_c1.val[1]);
                    v_dst.val[2] = vcombine_u8(v_c2.val[0], v_c2.val[1]);
                    v_dst.val[3] = vcombine_u8(v_c3.val[0], v_c3.val[1]);
                    vst4q_u8(dst0_row + dj, v_dst);
                    vst4q_u8(dst1_row + dj, v_dst);
                }

                for (size_t dwidth = dsize.width << 2; dj < dwidth; dj += 8, sj += 4)
                {
                    u8 src_val = src_row[sj];
                    dst0_row[dj] = dst0_row[dj + 4] = src_val;
                    dst1_row[dj] = dst1_row[dj + 4] = src_val;

                    src_val = src_row[sj + 1];
                    dst0_row[dj + 1] = dst0_row[dj + 5] = src_val;
                    dst1_row[dj + 1] = dst1_row[dj + 5] = src_val;

                    src_val = src_row[sj + 2];
                    dst0_row[dj + 2] = dst0_row[dj + 6] = src_val;
                    dst1_row[dj + 2] = dst1_row[dj + 6] = src_val;

                    src_val = src_row[sj + 3];
                    dst0_row[dj + 3] = dst0_row[dj + 7] = src_val;
                    dst1_row[dj + 3] = dst1_row[dj + 7] = src_val;
                }
            }
        }
        else //if ((hr == 4.0f) && (wr == 4.0f)) //the only scale that lasts after isSupported check
        {
            size_t roiw4 = dsize.width >= 3 ? (dsize.width - 3) << 2 : 0;
            size_t roiw2 = dsize.width >= 1 ? (dsize.width - 1) << 2 : 0;

            for (size_t i = 0; i < dsize.height; ++i)
            {
                const u8 * src0_row = internal::getRowPtr(srcBase, srcStride, i << 2);
                const u8 * src1_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 1);
                const u8 * src2_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 2);
                const u8 * src3_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 3);
                u8 * dst_row = internal::getRowPtr(dstBase, dstStride, i);
                size_t sj = 0, dj = 0;

                for ( ; dj < roiw4; dj += 16, sj += 64)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);
                    internal::prefetch(src2_row + sj);
                    internal::prefetch(src3_row + sj);

                    uint8x16_t vLane10 = vld1q_u8(src0_row + sj), vLane11 = vld1q_u8(src0_row + sj + 16);
                    uint8x16_t vLane20 = vld1q_u8(src1_row + sj), vLane21 = vld1q_u8(src1_row + sj + 16);
                    uint8x16_t vLane30 = vld1q_u8(src2_row + sj), vLane31 = vld1q_u8(src2_row + sj + 16);
                    uint8x16_t vLane40 = vld1q_u8(src3_row + sj), vLane41 = vld1q_u8(src3_row + sj + 16);

                    uint16x8_t v_part_0, v_part_1;
                    {
                        uint16x8_t v_sum0 = vaddl_u8(vget_low_u8(vLane10), vget_high_u8(vLane10));
                        v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane20), vget_high_u8(vLane20)));
                        v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane30), vget_high_u8(vLane30)));
                        v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane40), vget_high_u8(vLane40)));

                        uint16x8_t v_sum1 = vaddl_u8(vget_low_u8(vLane11), vget_high_u8(vLane11));
                        v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane21), vget_high_u8(vLane21)));
                        v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane31), vget_high_u8(vLane31)));
                        v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane41), vget_high_u8(vLane41)));

                        v_part_0 = vcombine_u16(vadd_u16(vget_low_u16(v_sum0), vget_high_u16(v_sum0)),
                                                vadd_u16(vget_low_u16(v_sum1), vget_high_u16(v_sum1)));
                    }

                    vLane10 = vld1q_u8(src0_row + sj + 32);
                    vLane11 = vld1q_u8(src0_row + sj + 48);
                    vLane20 = vld1q_u8(src1_row + sj + 32);
                    vLane21 = vld1q_u8(src1_row + sj + 48);
                    vLane30 = vld1q_u8(src2_row + sj + 32);
                    vLane31 = vld1q_u8(src2_row + sj + 48);
                    vLane40 = vld1q_u8(src3_row + sj + 32);
                    vLane41 = vld1q_u8(src3_row + sj + 48);

                    {
                        uint16x8_t v_sum0 = vaddl_u8(vget_low_u8(vLane10), vget_high_u8(vLane10));
                        v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane20), vget_high_u8(vLane20)));
                        v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane30), vget_high_u8(vLane30)));
                        v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane40), vget_high_u8(vLane40)));

                        uint16x8_t v_sum1 = vaddl_u8(vget_low_u8(vLane11), vget_high_u8(vLane11));
                        v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane21), vget_high_u8(vLane21)));
                        v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane31), vget_high_u8(vLane31)));
                        v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane41), vget_high_u8(vLane41)));

                        v_part_1 = vcombine_u16(vadd_u16(vget_low_u16(v_sum0), vget_high_u16(v_sum0)),
                                                vadd_u16(vget_low_u16(v_sum1), vget_high_u16(v_sum1)));
                    }

                    vst1q_u8(dst_row + dj, vcombine_u8(areaDownsamplingDivision<opencv_like,4>(v_part_0),
                                                       areaDownsamplingDivision<opencv_like,4>(v_part_1)));
                }

                for ( ; dj < roiw2; dj += 8, sj += 32)
                {
                    uint8x16_t vLane10 = vld1q_u8(src0_row + sj), vLane11 = vld1q_u8(src0_row + sj + 16);
                    uint8x16_t vLane20 = vld1q_u8(src1_row + sj), vLane21 = vld1q_u8(src1_row + sj + 16);
                    uint8x16_t vLane30 = vld1q_u8(src2_row + sj), vLane31 = vld1q_u8(src2_row + sj + 16);
                    uint8x16_t vLane40 = vld1q_u8(src3_row + sj), vLane41 = vld1q_u8(src3_row + sj + 16);

                    uint16x8_t v_sum0 = vaddl_u8(vget_low_u8(vLane10), vget_high_u8(vLane10));
                    v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane20), vget_high_u8(vLane20)));
                    v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane30), vget_high_u8(vLane30)));
                    v_sum0 = vaddq_u16(v_sum0, vaddl_u8(vget_low_u8(vLane40), vget_high_u8(vLane40)));

                    uint16x8_t v_sum1 = vaddl_u8(vget_low_u8(vLane11), vget_high_u8(vLane11));
                    v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane21), vget_high_u8(vLane21)));
                    v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane31), vget_high_u8(vLane31)));
                    v_sum1 = vaddq_u16(v_sum1, vaddl_u8(vget_low_u8(vLane41), vget_high_u8(vLane41)));

                    uint16x8_t v_sum = vcombine_u16(vadd_u16(vget_low_u16(v_sum0), vget_high_u16(v_sum0)),
                                                    vadd_u16(vget_low_u16(v_sum1), vget_high_u16(v_sum1)));

                    vst1_u8(dst_row + dj, (areaDownsamplingDivision<opencv_like,4>(v_sum)));
                }

                for (size_t dwidth = dsize.width << 2; dj < dwidth; dj += 4, sj += 16)
                {
                    dst_row[dj    ] = areaDownsamplingDivision<opencv_like,4>(
                                            (u16)src0_row[sj     ] + src0_row[sj +  4] +
                                                 src0_row[sj +  8] + src0_row[sj + 12] +
                                                 src1_row[sj     ] + src1_row[sj +  4] +
                                                 src1_row[sj +  8] + src1_row[sj + 12] +
                                                 src2_row[sj     ] + src2_row[sj +  4] +
                                                 src2_row[sj +  8] + src2_row[sj + 12] +
                                                 src3_row[sj     ] + src3_row[sj +  4] +
                                                 src3_row[sj +  8] + src3_row[sj + 12]);

                    dst_row[dj + 1] = areaDownsamplingDivision<opencv_like,4>(
                                            (u16)src0_row[sj +  1] + src0_row[sj +  5] +
                                                 src0_row[sj +  9] + src0_row[sj + 13] +
                                                 src1_row[sj +  1] + src1_row[sj +  5] +
                                                 src1_row[sj +  9] + src1_row[sj + 13] +
                                                 src2_row[sj +  1] + src2_row[sj +  5] +
                                                 src2_row[sj +  9] + src2_row[sj + 13] +
                                                 src3_row[sj +  1] + src3_row[sj +  5] +
                                                 src3_row[sj +  9] + src3_row[sj + 13]);

                    dst_row[dj + 2] = areaDownsamplingDivision<opencv_like,4>(
                                            (u16)src0_row[sj +  2] + src0_row[sj +  6] +
                                                 src0_row[sj + 10] + src0_row[sj + 14] +
                                                 src1_row[sj +  2] + src1_row[sj +  6] +
                                                 src1_row[sj + 10] + src1_row[sj + 14] +
                                                 src2_row[sj +  2] + src2_row[sj +  6] +
                                                 src2_row[sj + 10] + src2_row[sj + 14] +
                                                 src3_row[sj +  2] + src3_row[sj +  6] +
                                                 src3_row[sj + 10] + src3_row[sj + 14]);

                    dst_row[dj + 3] = areaDownsamplingDivision<opencv_like,4>(
                                            (u16)src0_row[sj +  3] + src0_row[sj +  7] +
                                                 src0_row[sj + 11] + src0_row[sj + 15] +
                                                 src1_row[sj +  3] + src1_row[sj +  7] +
                                                 src1_row[sj + 11] + src1_row[sj + 15] +
                                                 src2_row[sj +  3] + src2_row[sj +  7] +
                                                 src2_row[sj + 11] + src2_row[sj + 15] +
                                                 src3_row[sj +  3] + src3_row[sj +  7] +
                                                 src3_row[sj + 11] + src3_row[sj + 15]);
                }
            }
        }
    }
    else if (channels == 3)
    {
        if ((wr == 2.0f) && (hr == 2.0f))
        {
#ifndef __ANDROID__
            size_t roiw16 = dsize.width >= 15 ? (dsize.width - 15) * 3 : 0;
#endif
            size_t roiw8 = dsize.width >= 7 ? (dsize.width - 7) * 3 : 0;

            for (size_t i = 0; i < dsize.height; ++i)
            {
                const u8 * src0_row = internal::getRowPtr(srcBase, srcStride, i << 1);
                const u8 * src1_row = internal::getRowPtr(srcBase, srcStride, (i << 1) + 1);
                u8 * dst_row = internal::getRowPtr(dstBase, dstStride, i);
                size_t sj = 0, dj = 0;

#ifndef __ANDROID__
                for ( ; dj < roiw16; dj += 48, sj += 96)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);

                    uint8x16x3_t vLane1 = vld3q_u8(src0_row + sj);
                    uint8x16x3_t vLane2 = vld3q_u8(src1_row + sj);

                    uint8x8x3_t v_dst0, v_dst1;
                    {
                        uint16x8_t v_el0 = vpaddlq_u8(vLane1.val[0]);
                        uint16x8_t v_el1 = vpaddlq_u8(vLane1.val[1]);
                        uint16x8_t v_el2 = vpaddlq_u8(vLane1.val[2]);
                        v_el0 = vpadalq_u8(v_el0, vLane2.val[0]);
                        v_el1 = vpadalq_u8(v_el1, vLane2.val[1]);
                        v_el2 = vpadalq_u8(v_el2, vLane2.val[2]);

                        v_dst0.val[0] = areaDownsamplingDivision<opencv_like,2>(v_el0);
                        v_dst0.val[1] = areaDownsamplingDivision<opencv_like,2>(v_el1);
                        v_dst0.val[2] = areaDownsamplingDivision<opencv_like,2>(v_el2);
                    }

                    vLane1 = vld3q_u8(src0_row + sj + 48);
                    vLane2 = vld3q_u8(src1_row + sj + 48);
                    {
                        uint16x8_t v_el0 = vpaddlq_u8(vLane1.val[0]);
                        uint16x8_t v_el1 = vpaddlq_u8(vLane1.val[1]);
                        uint16x8_t v_el2 = vpaddlq_u8(vLane1.val[2]);
                        v_el0 = vpadalq_u8(v_el0, vLane2.val[0]);
                        v_el1 = vpadalq_u8(v_el1, vLane2.val[1]);
                        v_el2 = vpadalq_u8(v_el2, vLane2.val[2]);

                        v_dst1.val[0] = areaDownsamplingDivision<opencv_like,2>(v_el0);
                        v_dst1.val[1] = areaDownsamplingDivision<opencv_like,2>(v_el1);
                        v_dst1.val[2] = areaDownsamplingDivision<opencv_like,2>(v_el2);
                    }

                    uint8x16x3_t v_dst;
                    v_dst.val[0] = vcombine_u8(v_dst0.val[0], v_dst1.val[0]);
                    v_dst.val[1] = vcombine_u8(v_dst0.val[1], v_dst1.val[1]);
                    v_dst.val[2] = vcombine_u8(v_dst0.val[2], v_dst1.val[2]);

                    vst3q_u8(dst_row + dj, v_dst);
                }
#endif

                for ( ; dj < roiw8; dj += 24, sj += 48)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);

                    uint8x16x3_t vLane1 = vld3q_u8(src0_row + sj);
                    uint8x16x3_t vLane2 = vld3q_u8(src1_row + sj);

                    uint16x8_t v_el0 = vpaddlq_u8(vLane1.val[0]);
                    uint16x8_t v_el1 = vpaddlq_u8(vLane1.val[1]);
                    uint16x8_t v_el2 = vpaddlq_u8(vLane1.val[2]);
                    v_el0 = vpadalq_u8(v_el0, vLane2.val[0]);
                    v_el1 = vpadalq_u8(v_el1, vLane2.val[1]);
                    v_el2 = vpadalq_u8(v_el2, vLane2.val[2]);

                    uint8x8x3_t v_dst;
                    v_dst.val[0] = areaDownsamplingDivision<opencv_like,2>(v_el0);
                    v_dst.val[1] = areaDownsamplingDivision<opencv_like,2>(v_el1);
                    v_dst.val[2] = areaDownsamplingDivision<opencv_like,2>(v_el2);

                    vst3_u8(dst_row + dj, v_dst);
                }

                for (size_t dwidth = dsize.width * 3; dj < dwidth; dj += 3, sj += 6)
                {
                    dst_row[dj    ] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj    ] + src0_row[sj + 3] +
                                               src1_row[sj    ] + src1_row[sj + 3]);
                    dst_row[dj + 1] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj + 1] + src0_row[sj + 4] +
                                               src1_row[sj + 1] + src1_row[sj + 4]);
                    dst_row[dj + 2] = areaDownsamplingDivision<opencv_like,2>(
                                          (u16)src0_row[sj + 2] + src0_row[sj + 5] +
                                               src1_row[sj + 2] + src1_row[sj + 5]);
                }
            }
        }
        else if ((wr == 0.5f) && (hr == 0.5f))
        {
#ifndef __ANDROID__
            size_t roiw32 = dsize.width >= 31 ? (dsize.width - 31) * 3 : 0;
#endif
            size_t roiw16 = dsize.width >= 15 ? (dsize.width - 15) * 3 : 0;

            for (size_t i = 0; i < dsize.height; i += 2)
            {
                const u8 * src_row = internal::getRowPtr(srcBase, srcStride, i >> 1);
                u8 * dst0_row = internal::getRowPtr(dstBase, dstStride, i);
                u8 * dst1_row = internal::getRowPtr(dstBase, dstStride, std::min(i + 1, dsize.height - 1));
                size_t sj = 0, dj = 0;

#ifndef __ANDROID__
                for ( ; dj < roiw32; dj += 96, sj += 48)
                {
                    internal::prefetch(src_row + sj);

                    uint8x16x3_t v_src = vld3q_u8(src_row + sj);
                    uint8x16x2_t v_c0 = vzipq_u8(v_src.val[0], v_src.val[0]);
                    uint8x16x2_t v_c1 = vzipq_u8(v_src.val[1], v_src.val[1]);
                    uint8x16x2_t v_c2 = vzipq_u8(v_src.val[2], v_src.val[2]);

                    uint8x16x3_t v_dst;
                    v_dst.val[0] = v_c0.val[0];
                    v_dst.val[1] = v_c1.val[0];
                    v_dst.val[2] = v_c2.val[0];
                    vst3q_u8(dst0_row + dj, v_dst);
                    vst3q_u8(dst1_row + dj, v_dst);

                    v_dst.val[0] = v_c0.val[1];
                    v_dst.val[1] = v_c1.val[1];
                    v_dst.val[2] = v_c2.val[1];
                    vst3q_u8(dst0_row + dj + 48, v_dst);
                    vst3q_u8(dst1_row + dj + 48, v_dst);
                }
#endif

                for ( ; dj < roiw16; dj += 48, sj += 24)
                {
                    internal::prefetch(src_row + sj);

                    uint8x8x3_t v_src = vld3_u8(src_row + sj);
                    uint8x8x2_t v_c0 = vzip_u8(v_src.val[0], v_src.val[0]);
                    uint8x8x2_t v_c1 = vzip_u8(v_src.val[1], v_src.val[1]);
                    uint8x8x2_t v_c2 = vzip_u8(v_src.val[2], v_src.val[2]);

                    uint8x16x3_t v_dst;
                    v_dst.val[0] = vcombine_u8(v_c0.val[0], v_c0.val[1]);
                    v_dst.val[1] = vcombine_u8(v_c1.val[0], v_c1.val[1]);
                    v_dst.val[2] = vcombine_u8(v_c2.val[0], v_c2.val[1]);
                    vst3q_u8(dst0_row + dj, v_dst);
                    vst3q_u8(dst1_row + dj, v_dst);
                }

                for (size_t dwidth = dsize.width * 3; dj < dwidth; dj += 6, sj += 3)
                {
                    u8 src_val = src_row[sj];
                    dst0_row[dj] = dst0_row[dj + 3] = src_val;
                    dst1_row[dj] = dst1_row[dj + 3] = src_val;

                    src_val = src_row[sj + 1];
                    dst0_row[dj + 1] = dst0_row[dj + 4] = src_val;
                    dst1_row[dj + 1] = dst1_row[dj + 4] = src_val;

                    src_val = src_row[sj + 2];
                    dst0_row[dj + 2] = dst0_row[dj + 5] = src_val;
                    dst1_row[dj + 2] = dst1_row[dj + 5] = src_val;
                }
            }
        }
        else //if ((hr == 4.0f) && (wr == 4.0f)) //the only scale that lasts after isSupported check
        {
#ifndef __ANDROID__
            size_t roiw8 = dsize.width >= 7 ? (dsize.width - 7) * 3 : 0;
#endif

            for (size_t i = 0; i < dsize.height; ++i)
            {
                const u8 * src0_row = internal::getRowPtr(srcBase, srcStride, i << 2);
                const u8 * src1_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 1);
                const u8 * src2_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 2);
                const u8 * src3_row = internal::getRowPtr(srcBase, srcStride, (i << 2) + 3);
                u8 * dst_row = internal::getRowPtr(dstBase, dstStride, i);
                size_t sj = 0, dj = 0;

#ifndef __ANDROID__
                for ( ; dj < roiw8; dj += 24, sj += 96)
                {
                    internal::prefetch(src0_row + sj);
                    internal::prefetch(src1_row + sj);
                    internal::prefetch(src2_row + sj);
                    internal::prefetch(src3_row + sj);

                    uint8x16x3_t vLane10 = vld3q_u8(src0_row + sj), vLane11 = vld3q_u8(src0_row + sj + 48);
                    uint8x16x3_t vLane20 = vld3q_u8(src1_row + sj), vLane21 = vld3q_u8(src1_row + sj + 48);
                    uint8x16x3_t vLane30 = vld3q_u8(src2_row + sj), vLane31 = vld3q_u8(src2_row + sj + 48);
                    uint8x16x3_t vLane40 = vld3q_u8(src3_row + sj), vLane41 = vld3q_u8(src3_row + sj + 48);

                    uint8x8x3_t v_dst;

                    // channel 0
                    {
                        uint16x8_t v_lane0 = vpaddlq_u8(vLane10.val[0]);
                        uint16x8_t v_lane1 = vpaddlq_u8(vLane20.val[0]);
                        uint16x8_t v_lane2 = vpaddlq_u8(vLane30.val[0]);
                        uint16x8_t v_lane3 = vpaddlq_u8(vLane40.val[0]);
                        v_lane0 = vaddq_u16(v_lane0, v_lane1);
                        v_lane0 = vaddq_u16(v_lane0, v_lane2);
                        v_lane0 = vaddq_u16(v_lane0, v_lane3);

                        uint16x8_t v_lane0_ = vpaddlq_u8(vLane11.val[0]);
                        uint16x8_t v_lane1_ = vpaddlq_u8(vLane21.val[0]);
                        uint16x8_t v_lane2_ = vpaddlq_u8(vLane31.val[0]);
                        uint16x8_t v_lane3_ = vpaddlq_u8(vLane41.val[0]);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane1_);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane2_);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane3_);

                        v_dst.val[0] = areaDownsamplingDivision<opencv_like,4>(
                                           vcombine_u16(vmovn_u32(vpaddlq_u16(v_lane0)),
                                                        vmovn_u32(vpaddlq_u16(v_lane0_))));
                    }

                    // channel 1
                    {
                        uint16x8_t v_lane0 = vpaddlq_u8(vLane10.val[1]);
                        uint16x8_t v_lane1 = vpaddlq_u8(vLane20.val[1]);
                        uint16x8_t v_lane2 = vpaddlq_u8(vLane30.val[1]);
                        uint16x8_t v_lane3 = vpaddlq_u8(vLane40.val[1]);
                        v_lane0 = vaddq_u16(v_lane0, v_lane1);
                        v_lane0 = vaddq_u16(v_lane0, v_lane2);
                        v_lane0 = vaddq_u16(v_lane0, v_lane3);

                        uint16x8_t v_lane0_ = vpaddlq_u8(vLane11.val[1]);
                        uint16x8_t v_lane1_ = vpaddlq_u8(vLane21.val[1]);
                        uint16x8_t v_lane2_ = vpaddlq_u8(vLane31.val[1]);
                        uint16x8_t v_lane3_ = vpaddlq_u8(vLane41.val[1]);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane1_);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane2_);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane3_);

                        v_dst.val[1] = areaDownsamplingDivision<opencv_like,4>(
                                           vcombine_u16(vmovn_u32(vpaddlq_u16(v_lane0)),
                                                        vmovn_u32(vpaddlq_u16(v_lane0_))));
                    }

                    // channel 2
                    {
                        uint16x8_t v_lane0 = vpaddlq_u8(vLane10.val[2]);
                        uint16x8_t v_lane1 = vpaddlq_u8(vLane20.val[2]);
                        uint16x8_t v_lane2 = vpaddlq_u8(vLane30.val[2]);
                        uint16x8_t v_lane3 = vpaddlq_u8(vLane40.val[2]);
                        v_lane0 = vaddq_u16(v_lane0, v_lane1);
                        v_lane0 = vaddq_u16(v_lane0, v_lane2);
                        v_lane0 = vaddq_u16(v_lane0, v_lane3);

                        uint16x8_t v_lane0_ = vpaddlq_u8(vLane11.val[2]);
                        uint16x8_t v_lane1_ = vpaddlq_u8(vLane21.val[2]);
                        uint16x8_t v_lane2_ = vpaddlq_u8(vLane31.val[2]);
                        uint16x8_t v_lane3_ = vpaddlq_u8(vLane41.val[2]);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane1_);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane2_);
                        v_lane0_ = vaddq_u16(v_lane0_, v_lane3_);

                        v_dst.val[2] = areaDownsamplingDivision<opencv_like,4>(
                                           vcombine_u16(vmovn_u32(vpaddlq_u16(v_lane0)),
                                                        vmovn_u32(vpaddlq_u16(v_lane0_))));
                    }

                    vst3_u8(dst_row + dj, v_dst);
                }
#endif

                for (size_t dwidth = dsize.width * 3; dj < dwidth; dj += 3, sj += 12)
                {
                    dst_row[dj    ] = areaDownsamplingDivision<opencv_like,4>(
                                          (u16)src0_row[sj    ] + src0_row[sj +  3] +
                                               src0_row[sj + 6] + src0_row[sj +  9] +
                                               src1_row[sj    ] + src1_row[sj +  3] +
                                               src1_row[sj + 6] + src1_row[sj +  9] +
                                               src2_row[sj    ] + src2_row[sj +  3] +
                                               src2_row[sj + 6] + src2_row[sj +  9] +
                                               src3_row[sj    ] + src3_row[sj +  3] +
                                               src3_row[sj + 6] + src3_row[sj +  9]);

                    dst_row[dj + 1] = areaDownsamplingDivision<opencv_like,4>(
                                          (u16)src0_row[sj + 1] + src0_row[sj +  4] +
                                               src0_row[sj + 7] + src0_row[sj + 10] +
                                               src1_row[sj + 1] + src1_row[sj +  4] +
                                               src1_row[sj + 7] + src1_row[sj + 10] +
                                               src2_row[sj + 1] + src2_row[sj +  4] +
                                               src2_row[sj + 7] + src2_row[sj + 10] +
                                               src3_row[sj + 1] + src3_row[sj +  4] +
                                               src3_row[sj + 7] + src3_row[sj + 10]);

                    dst_row[dj + 2] = areaDownsamplingDivision<opencv_like,4>(
                                          (u16)src0_row[sj + 2] + src0_row[sj +  5] +
                                               src0_row[sj + 8] + src0_row[sj + 11] +
                                               src1_row[sj + 2] + src1_row[sj +  5] +
                                               src1_row[sj + 8] + src1_row[sj + 11] +
                                               src2_row[sj + 2] + src2_row[sj +  5] +
                                               src2_row[sj + 8] + src2_row[sj + 11] +
                                               src3_row[sj + 2] + src3_row[sj +  5] +
                                               src3_row[sj + 8] + src3_row[sj + 11]);
                }
            }
        }
    }
#else
    (void)dsize;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)wr;
    (void)hr;
#endif
    (void)ssize;
}

void resizeAreaOpenCV(const Size2D &ssize, const Size2D &dsize,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                f32 wr, f32 hr, u32 channels)
{
   resizeAreaRounding<true>(ssize, dsize, srcBase, srcStride, dstBase, dstStride, wr, hr, channels);
}

void resizeArea(const Size2D &ssize, const Size2D &dsize,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                f32 wr, f32 hr, u32 channels)
{
   resizeAreaRounding<false>(ssize, dsize, srcBase, srcStride, dstBase, dstStride, wr, hr, channels);
}

#ifdef CAROTENE_NEON

namespace {

uint8x8_t resizeLinearStep(uint8x16_t vr1, uint8x16_t vr2,
                           uint8x8_t vlutl, uint8x8_t vluth,
                           float32x4_t vrw, float32x4_t vcw0, float32x4_t vcw1)
{
    uint8x8_t vr1l = internal::vqtbl1_u8(vr1, vlutl);
    uint8x8_t vr1h = internal::vqtbl1_u8(vr1, vluth);
    uint8x8_t vr2l = internal::vqtbl1_u8(vr2, vlutl);
    uint8x8_t vr2h = internal::vqtbl1_u8(vr2, vluth);

    uint16x8_t v1hw = vmovl_u8(vr1h);
    uint16x8_t v2hw = vmovl_u8(vr2h);

    int16x8_t v1df = vreinterpretq_s16_u16(vsubl_u8(vr1l, vr1h));
    int16x8_t v2df = vreinterpretq_s16_u16(vsubl_u8(vr2l, vr2h));

    float32x4_t v1L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v1hw)));
    float32x4_t v1H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v1hw)));
    float32x4_t v2L = vcvtq_f32_u32(vmovl_u16(vget_low_u16(v2hw)));
    float32x4_t v2H = vcvtq_f32_u32(vmovl_u16(vget_high_u16(v2hw)));

    v1L = vmlaq_f32(v1L, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v1df))), vcw0);
    v1H = vmlaq_f32(v1H, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v1df))), vcw1);
    v2L = vmlaq_f32(v2L, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v2df))), vcw0);
    v2H = vmlaq_f32(v2H, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v2df))), vcw1);

    float32x4_t vdiffL = vsubq_f32(v1L, v2L);
    float32x4_t vdiffH = vsubq_f32(v1H, v2H);

    float32x4_t vL = vmlaq_f32(v2L, vdiffL, vrw);
    float32x4_t vH = vmlaq_f32(v2H, vdiffH, vrw);
    uint16x4_t vL_ = vmovn_u32(vcvtq_u32_f32(vL));
    uint16x4_t vH_ = vmovn_u32(vcvtq_u32_f32(vH));
    return vmovn_u16(vcombine_u16(vL_, vH_));
}

} // namespace

namespace {

void resize_bilinear_rows(const Size2D &ssize, const Size2D &dsize,
                        const u8 * srcBase, ptrdiff_t srcStride,
                        u8 * dstBase, ptrdiff_t dstStride,
                        f32 hr, const u8** gcols, u8* gcweight, u8* buf)
{
    f32 scale_y_offset = 0.5f * hr - 0.5f;

    size_t dst_h8 = dsize.height & ~7;
    size_t dst_w8 = dsize.width & ~7;
    size_t src_w8 = ssize.width & ~7;

    size_t r = 0;
    for (; r < dst_h8; r += 8)
    {
resize8u_xystretch:
        const u8* rows[16];
        u8 rweight[8];

        for (u32 i = 0; i < 8; ++i)
        {
            f32 w = (i + r) * hr + scale_y_offset;
            ptrdiff_t src_row = floorf(w);
            ptrdiff_t src_row2 = src_row + 1;

            rweight[i] = (u8)((src_row2-w) * 128);

            if (src_row < 0)
                src_row = 0;
            if (src_row2 >= (ptrdiff_t)ssize.height)
                src_row2 = ssize.height-1;

            rows[2 * i] = srcBase + src_row * srcStride;
            rows[2 * i + 1] = srcBase + src_row2 * srcStride;
        }

        uint8x8_t vr0w = vdup_n_u8(rweight[0]);
        uint8x8_t vr1w = vdup_n_u8(rweight[1]);
        uint8x8_t vr2w = vdup_n_u8(rweight[2]);
        uint8x8_t vr3w = vdup_n_u8(rweight[3]);
        uint8x8_t vr4w = vdup_n_u8(rweight[4]);
        uint8x8_t vr5w = vdup_n_u8(rweight[5]);
        uint8x8_t vr6w = vdup_n_u8(rweight[6]);
        uint8x8_t vr7w = vdup_n_u8(rweight[7]);

        uint8x8_t vr0w2 = vdup_n_u8(128 - rweight[0]);
        uint8x8_t vr1w2 = vdup_n_u8(128 - rweight[1]);
        uint8x8_t vr2w2 = vdup_n_u8(128 - rweight[2]);
        uint8x8_t vr3w2 = vdup_n_u8(128 - rweight[3]);
        uint8x8_t vr4w2 = vdup_n_u8(128 - rweight[4]);
        uint8x8_t vr5w2 = vdup_n_u8(128 - rweight[5]);
        uint8x8_t vr6w2 = vdup_n_u8(128 - rweight[6]);
        uint8x8_t vr7w2 = vdup_n_u8(128 - rweight[7]);

        size_t col = 0;
        for(; col < src_w8; col += 8)
        {
            internal::prefetch(rows[3] + col);
            internal::prefetch(rows[7] + col);
            internal::prefetch(rows[11] + col);
            internal::prefetch(rows[15] + col);
resize8u_ystretch:
            uint8x8_t vsrc0l1 = vld1_u8(rows[0] + col);
            uint8x8_t vsrc0l2 = vld1_u8(rows[1] + col);
            uint8x8_t vsrc1l1 = vld1_u8(rows[2] + col);
            uint8x8_t vsrc1l2 = vld1_u8(rows[3] + col);

            // (l1 * w + l2 * (128 - w) + 64) / 128
            uint16x8_t vdst0l = vmull_u8(vsrc0l1, vr0w);
            uint16x8_t vdst1l = vmull_u8(vsrc1l1, vr1w);

            uint8x8_t vsrc2l1 = vld1_u8(rows[4] + col);
            uint8x8_t vsrc2l2 = vld1_u8(rows[5] + col);
            uint8x8_t vsrc3l1 = vld1_u8(rows[6] + col);
            uint8x8_t vsrc3l2 = vld1_u8(rows[7] + col);

            vdst0l = vmlal_u8(vdst0l, vsrc0l2, vr0w2);
            vdst1l = vmlal_u8(vdst1l, vsrc1l2, vr1w2);
            uint16x8_t vdst2l = vmull_u8(vsrc2l1, vr2w);
            uint16x8_t vdst3l = vmull_u8(vsrc3l1, vr3w);

            uint8x8_t vsrc4l1 = vld1_u8(rows[8] + col);
            uint8x8_t vsrc4l2 = vld1_u8(rows[9] + col);
            uint8x8_t vsrc5l1 = vld1_u8(rows[10] + col);
            uint8x8_t vsrc5l2 = vld1_u8(rows[11] + col);

            vdst2l = vmlal_u8(vdst2l, vsrc2l2, vr2w2);
            vdst3l = vmlal_u8(vdst3l, vsrc3l2, vr3w2);
            uint16x8_t vdst4l = vmull_u8(vsrc4l1, vr4w);
            uint16x8_t vdst5l = vmull_u8(vsrc5l1, vr5w);

            uint8x8_t vsrc6l1 = vld1_u8(rows[12] + col);
            uint8x8_t vsrc6l2 = vld1_u8(rows[13] + col);
            uint8x8_t vsrc7l1 = vld1_u8(rows[14] + col);
            uint8x8_t vsrc7l2 = vld1_u8(rows[15] + col);

            uint8x8_t vdst0 = vrshrn_n_u16(vdst0l, 7);
            uint8x8_t vdst1 = vrshrn_n_u16(vdst1l, 7);
            vdst4l = vmlal_u8(vdst4l, vsrc4l2, vr4w2);
            vdst5l = vmlal_u8(vdst5l, vsrc5l2, vr5w2);
            uint16x8_t vdst6l = vmull_u8(vsrc6l1, vr6w);
            uint16x8_t vdst7l = vmull_u8(vsrc7l1, vr7w);

            uint8x8_t vdst2 = vrshrn_n_u16(vdst2l, 7);
            uint8x8_t vdst3 = vrshrn_n_u16(vdst3l, 7);
            vdst6l = vmlal_u8(vdst6l, vsrc6l2, vr6w2);
            vdst7l = vmlal_u8(vdst7l, vsrc7l2, vr7w2);

            uint8x8_t vdst4 = vrshrn_n_u16(vdst4l, 7);
            uint8x8_t vdst5 = vrshrn_n_u16(vdst5l, 7);
            uint8x8_t vdst6 = vrshrn_n_u16(vdst6l, 7);
            uint8x8_t vdst7 = vrshrn_n_u16(vdst7l, 7);

            // == 8x8 matrix transpose ==

            //00 01 02 03 04 05 06 07   d0
            //10 11 12 13 14 15 16 17   d1
            //20 21 22 23 24 25 26 27   d2
            //30 31 32 33 34 35 36 37   d3
            //40 41 42 43 44 45 46 47   d4
            //50 51 52 53 54 55 56 57   d5
            //60 61 62 63 64 65 66 67   d6
            //70 71 72 73 74 75 76 77   d7

            uint8x8x2_t vdst10t = vtrn_u8(vdst0, vdst1);
            uint8x8x2_t vdst32t = vtrn_u8(vdst2, vdst3);
            uint8x8x2_t vdst54t = vtrn_u8(vdst4, vdst5);
            uint8x8x2_t vdst76t = vtrn_u8(vdst6, vdst7);

            uint8x16_t vd1d0 = vcombine_u8(vdst10t.val[0], vdst10t.val[1]);
            uint8x16_t vd3d2 = vcombine_u8(vdst32t.val[0], vdst32t.val[1]);
            uint8x16_t vd5d4 = vcombine_u8(vdst54t.val[0], vdst54t.val[1]);
            uint8x16_t vd7d6 = vcombine_u8(vdst76t.val[0], vdst76t.val[1]);

            //00 10 02 12 04 14 06 16   d0
            //01 11 03 13 05 15 07 17   d1
            //20 30 22 32 24 34 26 36   d2
            //21 31 23 33 25 35 27 37   d3
            //40 50 42 52 44 54 46 56   d4
            //41 51 43 53 45 55 47 57   d5
            //60 70 62 72 64 74 66 76   d6
            //61 71 63 73 65 75 67 77   d7

            uint16x8x2_t vq1q0t = vtrnq_u16((uint16x8_t)vd1d0, (uint16x8_t)vd3d2);
            uint16x8x2_t vq3q2t = vtrnq_u16((uint16x8_t)vd5d4, (uint16x8_t)vd7d6);

            //00 10 20 30 04 14 24 34   d0
            //01 11 21 31 05 15 25 35   d1
            //02 12 22 32 06 16 26 36   d2
            //03 13 23 33 07 17 27 37   d3
            //40 50 60 70 44 54 64 74   d4
            //41 51 61 71 45 55 65 75   d5
            //42 52 62 72 46 56 66 76   d6
            //43 53 63 73 47 57 67 77   d7

            uint32x4x2_t vq2q0t = vtrnq_u32((uint32x4_t)vq1q0t.val[0], (uint32x4_t)vq3q2t.val[0]);
            uint32x4x2_t vq3q1t = vtrnq_u32((uint32x4_t)vq1q0t.val[1], (uint32x4_t)vq3q2t.val[1]);

            //00 10 20 30 40 50 60 70   d0
            //01 11 21 31 41 51 61 71   d1
            //02 12 22 32 42 52 62 72   d2
            //03 13 23 33 43 53 63 73   d3
            //04 14 24 34 44 54 64 74   d4
            //05 15 25 35 45 55 65 75   d5
            //06 16 26 36 46 56 66 76   d6
            //07 17 27 37 47 57 67 77   d7

            vst1q_u8(buf + col * 8 +  0, (uint8x16_t)vq2q0t.val[0]);
            vst1q_u8(buf + col * 8 + 16, (uint8x16_t)vq3q1t.val[0]);
            vst1q_u8(buf + col * 8 + 32, (uint8x16_t)vq2q0t.val[1]);
            vst1q_u8(buf + col * 8 + 48, (uint8x16_t)vq3q1t.val[1]);
        }

        if (col < ssize.width)
        {
            col = ssize.width - 8;
            goto resize8u_ystretch;
        }

        u8* dst_data = dstBase + r * dstStride;
        const u8** cols = gcols;
        u8* cweight = gcweight;

        size_t dcol = 0;
        for (; dcol < dst_w8; dcol += 8, cols += 16, cweight += 8)
        {
            internal::prefetch(cols[0], 64*4);
resize8u_xstretch:
            uint8x8_t vc0w = vdup_n_u8(cweight[0]);
            uint8x8_t vc1w = vdup_n_u8(cweight[1]);
            uint8x8_t vc2w = vdup_n_u8(cweight[2]);
            uint8x8_t vc3w = vdup_n_u8(cweight[3]);
            uint8x8_t vc4w = vdup_n_u8(cweight[4]);
            uint8x8_t vc5w = vdup_n_u8(cweight[5]);
            uint8x8_t vc6w = vdup_n_u8(cweight[6]);
            uint8x8_t vc7w = vdup_n_u8(cweight[7]);

            uint8x8_t vc0w2 = vdup_n_u8(128 - cweight[0]);
            uint8x8_t vc1w2 = vdup_n_u8(128 - cweight[1]);
            uint8x8_t vc2w2 = vdup_n_u8(128 - cweight[2]);
            uint8x8_t vc3w2 = vdup_n_u8(128 - cweight[3]);
            uint8x8_t vc4w2 = vdup_n_u8(128 - cweight[4]);
            uint8x8_t vc5w2 = vdup_n_u8(128 - cweight[5]);
            uint8x8_t vc6w2 = vdup_n_u8(128 - cweight[6]);
            uint8x8_t vc7w2 = vdup_n_u8(128 - cweight[7]);

            uint8x8_t vsrc0l1 = vld1_u8(cols[0]);
            uint8x8_t vsrc0l2 = vld1_u8(cols[1]);
            uint8x8_t vsrc1l1 = vld1_u8(cols[2]);
            uint8x8_t vsrc1l2 = vld1_u8(cols[3]);
            uint8x8_t vsrc2l1 = vld1_u8(cols[4]);
            uint8x8_t vsrc2l2 = vld1_u8(cols[5]);
            uint8x8_t vsrc3l1 = vld1_u8(cols[6]);
            uint8x8_t vsrc3l2 = vld1_u8(cols[7]);
            uint8x8_t vsrc4l1 = vld1_u8(cols[8]);
            uint8x8_t vsrc4l2 = vld1_u8(cols[9]);
            uint8x8_t vsrc5l1 = vld1_u8(cols[10]);
            uint8x8_t vsrc5l2 = vld1_u8(cols[11]);
            uint8x8_t vsrc6l1 = vld1_u8(cols[12]);
            uint8x8_t vsrc6l2 = vld1_u8(cols[13]);
            uint8x8_t vsrc7l1 = vld1_u8(cols[14]);
            uint8x8_t vsrc7l2 = vld1_u8(cols[15]);

            // (l1 * w + l2 * (128 - w) + 64) / 128
            uint16x8_t vdst0l = vmull_u8(vsrc0l1, vc0w);
            uint16x8_t vdst1l = vmull_u8(vsrc1l1, vc1w);
            uint16x8_t vdst2l = vmull_u8(vsrc2l1, vc2w);
            uint16x8_t vdst3l = vmull_u8(vsrc3l1, vc3w);
            uint16x8_t vdst4l = vmull_u8(vsrc4l1, vc4w);
            uint16x8_t vdst5l = vmull_u8(vsrc5l1, vc5w);
            uint16x8_t vdst6l = vmull_u8(vsrc6l1, vc6w);
            uint16x8_t vdst7l = vmull_u8(vsrc7l1, vc7w);

            vdst0l = vmlal_u8(vdst0l, vsrc0l2, vc0w2);
            vdst1l = vmlal_u8(vdst1l, vsrc1l2, vc1w2);
            vdst2l = vmlal_u8(vdst2l, vsrc2l2, vc2w2);
            vdst3l = vmlal_u8(vdst3l, vsrc3l2, vc3w2);
            vdst4l = vmlal_u8(vdst4l, vsrc4l2, vc4w2);
            vdst5l = vmlal_u8(vdst5l, vsrc5l2, vc5w2);
            vdst6l = vmlal_u8(vdst6l, vsrc6l2, vc6w2);
            vdst7l = vmlal_u8(vdst7l, vsrc7l2, vc7w2);

            uint8x8_t vdst0 = vrshrn_n_u16(vdst0l, 7);
            uint8x8_t vdst1 = vrshrn_n_u16(vdst1l, 7);
            uint8x8_t vdst2 = vrshrn_n_u16(vdst2l, 7);
            uint8x8_t vdst3 = vrshrn_n_u16(vdst3l, 7);
            uint8x8_t vdst4 = vrshrn_n_u16(vdst4l, 7);
            uint8x8_t vdst5 = vrshrn_n_u16(vdst5l, 7);
            uint8x8_t vdst6 = vrshrn_n_u16(vdst6l, 7);
            uint8x8_t vdst7 = vrshrn_n_u16(vdst7l, 7);

            // == 8x8 matrix transpose ==
            uint8x8x2_t vdst10t = vtrn_u8(vdst0, vdst1);
            uint8x8x2_t vdst32t = vtrn_u8(vdst2, vdst3);
            uint8x8x2_t vdst54t = vtrn_u8(vdst4, vdst5);
            uint8x8x2_t vdst76t = vtrn_u8(vdst6, vdst7);
            uint8x16_t vd1d0 = vcombine_u8(vdst10t.val[0], vdst10t.val[1]);
            uint8x16_t vd3d2 = vcombine_u8(vdst32t.val[0], vdst32t.val[1]);
            uint8x16_t vd5d4 = vcombine_u8(vdst54t.val[0], vdst54t.val[1]);
            uint8x16_t vd7d6 = vcombine_u8(vdst76t.val[0], vdst76t.val[1]);
            uint16x8x2_t vq1q0t = vtrnq_u16((uint16x8_t)vd1d0, (uint16x8_t)vd3d2);
            uint16x8x2_t vq3q2t = vtrnq_u16((uint16x8_t)vd5d4, (uint16x8_t)vd7d6);
            uint32x4x2_t vq2q0t = vtrnq_u32((uint32x4_t)vq1q0t.val[0], (uint32x4_t)vq3q2t.val[0]);
            uint32x4x2_t vq3q1t = vtrnq_u32((uint32x4_t)vq1q0t.val[1], (uint32x4_t)vq3q2t.val[1]);

            //save results
            vst1_u8(dst_data + 0 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq2q0t.val[0]));
            vst1_u8(dst_data + 1 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq2q0t.val[0]));
            vst1_u8(dst_data + 2 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq3q1t.val[0]));
            vst1_u8(dst_data + 3 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq3q1t.val[0]));
            vst1_u8(dst_data + 4 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq2q0t.val[1]));
            vst1_u8(dst_data + 5 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq2q0t.val[1]));
            vst1_u8(dst_data + 6 * dstStride + dcol, (uint8x8_t)vget_low_u32(vq3q1t.val[1]));
            vst1_u8(dst_data + 7 * dstStride + dcol, (uint8x8_t)vget_high_u32(vq3q1t.val[1]));
        }

        if (dcol < dsize.width)
        {
            dcol = dsize.width - 8;
            cols = gcols + dcol * 2;
            cweight = gcweight + dcol;
            goto resize8u_xstretch;
        }
    }

    if (r < dsize.height)
    {
        r = dsize.height - 8;
        goto resize8u_xystretch;
    }
}

template <int channels> struct resizeLinearInternals;
template <> struct resizeLinearInternals<1>
{
    int32x4_t vc_upd;
    int32x4_t vc0;
    int32x4_t vcmax;

    inline resizeLinearInternals(int32x4_t & vi, u32 srccols)
    {
        vc_upd = vdupq_n_s32(4);
        vc0 = vdupq_n_s32(0);
        vcmax = vdupq_n_s32(srccols-1);

        s32 tmp0123[] = {0, 1, 2, 3 };
        vi = vld1q_s32(tmp0123);
    }
    inline void updateIndexes(int32x4_t & vi, int32x4_t & vsrch, int32x4_t & vsrcl)
    {
        vsrch = vminq_s32(vsrch, vcmax);
        vsrcl = vmaxq_s32(vsrcl, vc0);
        vsrcl = vminq_s32(vsrcl, vcmax);//for safe tail
        vsrch = vshlq_n_s32(vsrch, 3);
        vsrcl = vshlq_n_s32(vsrcl, 3);
        vi = vaddq_s32(vi, vc_upd);
    }
};
template <> struct resizeLinearInternals<4>
{
    int32x4_t vc_upd;
    int32x4_t vc0;
    int32x4_t vcmax;
    int32x4_t v0123x8;

    inline resizeLinearInternals(int32x4_t & vi, u32 srccols)
    {
        vc_upd = vdupq_n_s32(1);
        vc0 = vdupq_n_s32(0);
        vcmax = vdupq_n_s32(srccols-1);
        s32 tmp0123x8[] = {0, 8, 16, 24};
        v0123x8 = vld1q_s32(tmp0123x8);

        vi = vc0;
    }
    inline void updateIndexes(int32x4_t & vi, int32x4_t & vsrch, int32x4_t & vsrcl)
    {
        vsrch = vminq_s32(vsrch, vcmax);
        vsrcl = vmaxq_s32(vsrcl, vc0);
        vsrch = vshlq_n_s32(vsrch, 5);
        vsrcl = vshlq_n_s32(vsrcl, 5);
        vsrch = vaddq_s32(vsrch, v0123x8);
        vsrcl = vaddq_s32(vsrcl, v0123x8);
        vi = vaddq_s32(vi, vc_upd);
    }
};

template <int channels>
void resizeLinearOpenCVchan(const Size2D &_ssize, const Size2D &_dsize,
                            const u8 * srcBase, ptrdiff_t srcStride,
                            u8 * dstBase, ptrdiff_t dstStride,
                            f32 wr, f32 hr)
{
    float scale_x_offset = 0.5f * wr - 0.5f;

    Size2D ssize(_ssize.width*channels, _ssize.height);
    Size2D dsize(_dsize.width*channels, _dsize.height);

    std::vector<u8> gcweight((dsize.width + 7) & ~7);
    std::vector<const u8*> gcols(((dsize.width + 7) & ~7) * 2);
    std::vector<u8> buf(((ssize.width + 7) & ~7) * 8); // (8 rows) x (width of src)

    float32x4_t vscale_x = vdupq_n_f32(wr);
    float32x4_t vscale_x_offset = vdupq_n_f32(scale_x_offset);
    int32x4_t vc1 = vdupq_n_s32(1);
    float32x4_t vc128f = vdupq_n_f32(128.0f);

    int32x4_t vi;
    resizeLinearInternals<channels> indexes(vi, _ssize.width);//u32 is used to store indexes
                                                              //so we could get issues on src image dimensions greater than (2^32-1)

    for (size_t dcol = 0; dcol < dsize.width; dcol += 8)
    {
        s32 idx[16];

        float32x4_t vif = vcvtq_f32_s32(vi);
        float32x4_t vw = vmlaq_f32(vscale_x_offset, vscale_x, vif);
        int32x4_t vwi = vcvtq_s32_f32(vw);
        float32x4_t vwif = vcvtq_f32_s32(vwi);
        int32x4_t vmask = (int32x4_t)vcltq_f32(vwif, vw);
        int32x4_t vsrch = vsubq_s32(vwi, vmask);
        int32x4_t vsrcl = vsubq_s32(vsrch, vc1);
        float32x4_t vsrchf = vcvtq_f32_s32(vsrch);
        float32x4_t vw2 = vsubq_f32(vsrchf, vw);

        vw2 = vmulq_f32(vw2, vc128f);
        uint32x4_t vw32u = vcvtq_u32_f32(vw2);
        uint16x4_t vw16ul = vmovn_u32(vw32u);
        indexes.updateIndexes(vi, vsrch, vsrcl);

        vst1q_s32(idx + 0, vsrcl);
        vst1q_s32(idx + 8, vsrch);

        vif = vcvtq_f32_s32(vi);
        vw = vmlaq_f32(vscale_x_offset, vscale_x, vif);
        vwi = vcvtq_s32_f32(vw);
        vwif = vcvtq_f32_s32(vwi);
        vmask = (int32x4_t)vcltq_f32(vwif, vw);
        vsrch = vsubq_s32(vwi, vmask);
        vsrcl = vsubq_s32(vsrch, vc1);
        vsrchf = vcvtq_f32_s32(vsrch);
        vw2 = vsubq_f32(vsrchf, vw);

        vw2 = vmulq_f32(vw2, vc128f);
        vw32u = vcvtq_u32_f32(vw2);
        indexes.updateIndexes(vi, vsrch, vsrcl);

        uint16x4_t vw16uh = vmovn_u32(vw32u);

        vst1q_s32(idx + 4, vsrcl);
        vst1q_s32(idx + 12, vsrch);

        uint8x8_t vw8u = vmovn_u16(vcombine_u16(vw16ul, vw16uh));

        for (u32 i = 0; i < 8; ++i)
        {
            gcols[dcol * 2 + i*2] = &buf[idx[i]];
            gcols[dcol * 2 + i*2 + 1] = &buf[idx[i + 8]];
        }

        vst1_u8(&gcweight[dcol], vw8u);
    }

    resize_bilinear_rows(ssize, dsize, srcBase, srcStride, dstBase, dstStride, hr, &gcols[0], &gcweight[0], &buf[0]);
}

void downsample_bilinear_8uc1(const Size2D &ssize, const Size2D &dsize,
                              const u8 * srcBase, ptrdiff_t srcStride,
                              u8 * dstBase, ptrdiff_t dstStride,
                              f32 wr, f32 hr)
{
    internal::assertSupportedConfiguration(wr <= 2.f && hr <= 2.f);

    enum { SHIFT_BITS = 11 };

    f32 scale_x_offset = 0.5f * wr - 0.5f;
    f32 scale_y_offset = 0.5f * hr - 0.5f;

    std::vector<s32> _buf(dsize.height*(2*(sizeof(ptrdiff_t)/sizeof(s32))+1)+1);
    ptrdiff_t* buf = (ptrdiff_t*)&_buf[0];
    s32* buf2 = (s32*)buf+2*(sizeof(ptrdiff_t)/sizeof(s32))*dsize.height;
    for(size_t row = 0; row < (size_t)dsize.height; ++row)
    {
        f32 r = row * hr + scale_y_offset;
        ptrdiff_t src_row = floorf(r);
        ptrdiff_t src_row2 = src_row + 1;

        f32 rweight = src_row2 - r;
        buf2[row] = floorf(rweight * (1 << SHIFT_BITS) + 0.5f);
        buf[0 * dsize.height + row] = std::max<ptrdiff_t>(0, src_row);
        buf[1 * dsize.height + row] = std::min((ptrdiff_t)ssize.height-1, src_row2);
    }

#define USE_CORRECT_VERSION 0

    ptrdiff_t col = 0;
/***********************************************/
    for(; col <= (ptrdiff_t)dsize.width-16; col+=16)
    {
        ptrdiff_t col1[16];
        ptrdiff_t col2[16];
        s16 cwi[16];

        for(s32 k = 0; k < 16; ++k)
        {
            f32 c = (col + k) * wr + scale_x_offset;
            col1[k] = (ptrdiff_t)c;
            col2[k] = col1[k] + 1;

            cwi[k] = (short)floorf((col2[k] - c) * (1 << SHIFT_BITS) + 0.5f);

            if(col1[k] < 0) col1[k] = 0;
            if(col2[k] >= (ptrdiff_t)ssize.width) col2[k] = ssize.width-1;
        }

        ptrdiff_t x = std::min(col1[0], (ptrdiff_t)ssize.width-16);
        ptrdiff_t y = std::min(col1[8], (ptrdiff_t)ssize.width-16);
        u8 lutl[16];
        u8 luth[16];
        for(s32 k = 0; k < 8; ++k)
        {
            lutl[k] = (u8)(col1[k] - x);
            luth[k] = (u8)(col2[k] - x);
            lutl[k+8] = (u8)(col1[k+8] - y);
            luth[k+8] = (u8)(col2[k+8] - y);
        }

        uint8x8_t vlutl = vld1_u8(lutl);
        uint8x8_t vluth = vld1_u8(luth);
        int16x8_t vcw = vld1q_s16(cwi);

        uint8x8_t vlutl_ = vld1_u8(lutl+8);
        uint8x8_t vluth_ = vld1_u8(luth+8);
        int16x8_t vcw_ = vld1q_s16(cwi+8);

        for(ptrdiff_t row = 0; row < (ptrdiff_t)dsize.height; ++row)
        {
#if USE_CORRECT_VERSION
            int32x4_t vrw = vdupq_n_s32(buf2[row]);
#else
            int16x8_t vrw = vdupq_n_s16((int16_t)buf2[row]);
            int16x8_t vrW = vdupq_n_s16((int16_t)((1 << SHIFT_BITS) - buf2[row]));
#endif

            internal::prefetch(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + x, 2*srcStride);
            internal::prefetch(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + x, 3*srcStride);

            {
                union { uint8x16_t v; uint8x8x2_t w; } vr1 = { vld1q_u8(internal::getRowPtr(srcBase, srcStride, buf[0*dsize.height + row]) + x) };
                union { uint8x16_t v; uint8x8x2_t w; } vr2 = { vld1q_u8(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + x) };

                uint8x8_t vr1l = vtbl2_u8(vr1.w, vlutl);
                uint8x8_t vr1h = vtbl2_u8(vr1.w, vluth);
                uint8x8_t vr2l = vtbl2_u8(vr2.w, vlutl);
                uint8x8_t vr2h = vtbl2_u8(vr2.w, vluth);

                uint16x8_t v1hw = vmovl_u8(vr1h);
                uint16x8_t v2hw = vmovl_u8(vr2h);

                int16x8_t v1df = vreinterpretq_s16_u16(vsubl_u8(vr1l, vr1h));
                int16x8_t v2df = vreinterpretq_s16_u16(vsubl_u8(vr2l, vr2h));

                int32x4_t v1L = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(v1hw),  SHIFT_BITS));
                int32x4_t v1H = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(v1hw), SHIFT_BITS));
                int32x4_t v2L = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(v2hw),  SHIFT_BITS));
                int32x4_t v2H = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(v2hw), SHIFT_BITS));

                v1L = vmlal_s16(v1L, vget_low_s16(v1df), vget_low_s16(vcw));
                v1H = vmlal_s16(v1H, vget_high_s16(v1df), vget_high_s16(vcw));
                v2L = vmlal_s16(v2L, vget_low_s16(v2df), vget_low_s16(vcw));
                v2H = vmlal_s16(v2H, vget_high_s16(v2df), vget_high_s16(vcw));

#if USE_CORRECT_VERSION
                /* correct version */
                int32x4_t vL = vshlq_n_s32(v2L, SHIFT_BITS);
                int32x4_t vH = vshlq_n_s32(v2H, SHIFT_BITS);
                int32x4_t vdiffL = vsubq_s32(v1L, v2L);
                int32x4_t vdiffH = vsubq_s32(v1H, v2H);

                vL = vmlaq_s32(vL, vdiffL, vrw);
                vH = vmlaq_s32(vH, vdiffH, vrw);
                uint16x4_t vL_ = vqrshrun_n_s32(vL, 2*SHIFT_BITS - 8);
                uint16x4_t vH_ = vqrshrun_n_s32(vH, 2*SHIFT_BITS - 8);
                uint8x8_t vres = vrshrn_n_u16(vcombine_u16(vL_, vH_), 8);
                vst1_u8(internal::getRowPtr(dstBase, dstStride, row) + col, vres);
#else
                /* ugly version matching to OpenCV's SSE optimization */
                int16x4_t v1Ls = vshrn_n_s32(v1L, 4);
                int16x4_t v1Hs = vshrn_n_s32(v1H, 4);
                int16x4_t v2Ls = vshrn_n_s32(v2L, 4);
                int16x4_t v2Hs = vshrn_n_s32(v2H, 4);

                int16x8_t v1s = vqdmulhq_s16(vcombine_s16(v1Ls, v1Hs), vrw);
                int16x8_t v2s = vqdmulhq_s16(vcombine_s16(v2Ls, v2Hs), vrW);

                int16x8_t vsum = vaddq_s16(vshrq_n_s16(v1s,1), vshrq_n_s16(v2s,1));
                uint8x8_t vres = vqrshrun_n_s16(vsum, 2);

                vst1_u8(internal::getRowPtr(dstBase, dstStride, row) + col, vres);
#endif
            }

            {
                union { uint8x16_t v; uint8x8x2_t w; } vr1 = { vld1q_u8(internal::getRowPtr(srcBase, srcStride, buf[0*dsize.height + row]) + y) };
                union { uint8x16_t v; uint8x8x2_t w; } vr2 = { vld1q_u8(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + y) };

                uint8x8_t vr1l = vtbl2_u8(vr1.w, vlutl_);
                uint8x8_t vr1h = vtbl2_u8(vr1.w, vluth_);
                uint8x8_t vr2l = vtbl2_u8(vr2.w, vlutl_);
                uint8x8_t vr2h = vtbl2_u8(vr2.w, vluth_);

                uint16x8_t v1hw = vmovl_u8(vr1h);
                uint16x8_t v2hw = vmovl_u8(vr2h);

                int16x8_t v1df = vreinterpretq_s16_u16(vsubl_u8(vr1l, vr1h));
                int16x8_t v2df = vreinterpretq_s16_u16(vsubl_u8(vr2l, vr2h));

                int32x4_t v1L = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(v1hw),  SHIFT_BITS));
                int32x4_t v1H = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(v1hw), SHIFT_BITS));
                int32x4_t v2L = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(v2hw),  SHIFT_BITS));
                int32x4_t v2H = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(v2hw), SHIFT_BITS));

                v1L = vmlal_s16(v1L, vget_low_s16(v1df), vget_low_s16(vcw_));
                v1H = vmlal_s16(v1H, vget_high_s16(v1df), vget_high_s16(vcw_));
                v2L = vmlal_s16(v2L, vget_low_s16(v2df), vget_low_s16(vcw_));
                v2H = vmlal_s16(v2H, vget_high_s16(v2df), vget_high_s16(vcw_));

#if USE_CORRECT_VERSION
                /* correct version */
                int32x4_t vL = vshlq_n_s32(v2L, SHIFT_BITS);
                int32x4_t vH = vshlq_n_s32(v2H, SHIFT_BITS);
                int32x4_t vdiffL = vsubq_s32(v1L, v2L);
                int32x4_t vdiffH = vsubq_s32(v1H, v2H);

                vL = vmlaq_s32(vL, vdiffL, vrw);
                vH = vmlaq_s32(vH, vdiffH, vrw);
                uint16x4_t vL_ = vqrshrun_n_s32(vL, 2*SHIFT_BITS - 8);
                uint16x4_t vH_ = vqrshrun_n_s32(vH, 2*SHIFT_BITS - 8);
                uint8x8_t vres = vrshrn_n_u16(vcombine_u16(vL_, vH_), 8);
                vst1_u8(internal::getRowPtr(dstBase, dstStride, row) + col + 8, vres);
#else
                /* ugly version matching to OpenCV's SSE optimization */
                int16x4_t v1Ls = vshrn_n_s32(v1L, 4);
                int16x4_t v1Hs = vshrn_n_s32(v1H, 4);
                int16x4_t v2Ls = vshrn_n_s32(v2L, 4);
                int16x4_t v2Hs = vshrn_n_s32(v2H, 4);

                int16x8_t v1s = vqdmulhq_s16(vcombine_s16(v1Ls, v1Hs), vrw);
                int16x8_t v2s = vqdmulhq_s16(vcombine_s16(v2Ls, v2Hs), vrW);

                int16x8_t vsum = vaddq_s16(vshrq_n_s16(v1s,1), vshrq_n_s16(v2s,1));
                uint8x8_t vres = vqrshrun_n_s16(vsum, 2);

                vst1_u8(internal::getRowPtr(dstBase, dstStride, row) + col + 8, vres);
#endif
            }
        }
    }
/***********************************************/
    for(; col <= (ptrdiff_t)dsize.width-8; col+=8)
    {
downsample_bilinear_8uc1_col_loop8:
        ptrdiff_t col1[8];
        ptrdiff_t col2[8];
        s16 cwi[8];

        for(s32 k = 0; k < 8; ++k)
        {
            f32 c = (col + k) * wr + scale_x_offset;
            col1[k] = (ptrdiff_t)c;
            col2[k] = col1[k] + 1;

            cwi[k] = (s16)floorf((col2[k] - c) * (1 << SHIFT_BITS) + 0.5f);

            if(col1[k] < 0) col1[k] = 0;
            if(col2[k] >= (ptrdiff_t)ssize.width) col2[k] = (ptrdiff_t)ssize.width-1;
        }

        ptrdiff_t x = std::min(col1[0], (ptrdiff_t)ssize.width-16);
        u8 lutl[8];
        u8 luth[8];
        for(s32 k = 0; k < 8; ++k)
        {
            lutl[k] = (u8)(col1[k] - x);
            luth[k] = (u8)(col2[k] - x);
        }

        uint8x8_t vlutl = vld1_u8(lutl);
        uint8x8_t vluth = vld1_u8(luth);
        int16x8_t vcw = vld1q_s16(cwi);

        for(ptrdiff_t row = 0; row < (ptrdiff_t)dsize.height; ++row)
        {
#if USE_CORRECT_VERSION
            int32x4_t vrw = vdupq_n_s32(buf2[row]);
#else
            int16x8_t vrw = vdupq_n_s16((int16_t)buf2[row]);
            int16x8_t vrW = vdupq_n_s16((int16_t)((1 << SHIFT_BITS) - buf2[row]));
#endif

            internal::prefetch(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + x, 2*srcStride);
            internal::prefetch(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + x, 3*srcStride);

            union { uint8x16_t v; uint8x8x2_t w; } vr1 = { vld1q_u8(internal::getRowPtr(srcBase, srcStride, buf[0*dsize.height + row]) + x) };
            union { uint8x16_t v; uint8x8x2_t w; } vr2 = { vld1q_u8(internal::getRowPtr(srcBase, srcStride, buf[1*dsize.height + row]) + x) };

            uint8x8_t vr1l = vtbl2_u8(vr1.w, vlutl);
            uint8x8_t vr1h = vtbl2_u8(vr1.w, vluth);
            uint8x8_t vr2l = vtbl2_u8(vr2.w, vlutl);
            uint8x8_t vr2h = vtbl2_u8(vr2.w, vluth);

            uint16x8_t v1hw = vmovl_u8(vr1h);
            uint16x8_t v2hw = vmovl_u8(vr2h);

            int16x8_t v1df = vreinterpretq_s16_u16(vsubl_u8(vr1l, vr1h));
            int16x8_t v2df = vreinterpretq_s16_u16(vsubl_u8(vr2l, vr2h));

            int32x4_t v1L = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(v1hw),  SHIFT_BITS));
            int32x4_t v1H = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(v1hw), SHIFT_BITS));
            int32x4_t v2L = vreinterpretq_s32_u32(vshll_n_u16(vget_low_u16(v2hw),  SHIFT_BITS));
            int32x4_t v2H = vreinterpretq_s32_u32(vshll_n_u16(vget_high_u16(v2hw), SHIFT_BITS));

            v1L = vmlal_s16(v1L, vget_low_s16(v1df), vget_low_s16(vcw));
            v1H = vmlal_s16(v1H, vget_high_s16(v1df), vget_high_s16(vcw));
            v2L = vmlal_s16(v2L, vget_low_s16(v2df), vget_low_s16(vcw));
            v2H = vmlal_s16(v2H, vget_high_s16(v2df), vget_high_s16(vcw));

#if USE_CORRECT_VERSION
            /* correct version */
            int32x4_t vL = vshlq_n_s32(v2L, SHIFT_BITS);
            int32x4_t vH = vshlq_n_s32(v2H, SHIFT_BITS);
            int32x4_t vdiffL = vsubq_s32(v1L, v2L);
            int32x4_t vdiffH = vsubq_s32(v1H, v2H);

            vL = vmlaq_s32(vL, vdiffL, vrw);
            vH = vmlaq_s32(vH, vdiffH, vrw);
            uint16x4_t vL_ = vqrshrun_n_s32(vL, 2*SHIFT_BITS - 8);
            uint16x4_t vH_ = vqrshrun_n_s32(vH, 2*SHIFT_BITS - 8);
            uint8x8_t vres = vrshrn_n_u16(vcombine_u16(vL_, vH_), 8);
            vst1_u8(internal::getRowPtr(dstBase, dstStride, row) + col, vres);
#else
            /* ugly version matching to OpenCV's SSE optimization */
            int16x4_t v1Ls = vshrn_n_s32(v1L, 4);
            int16x4_t v1Hs = vshrn_n_s32(v1H, 4);
            int16x4_t v2Ls = vshrn_n_s32(v2L, 4);
            int16x4_t v2Hs = vshrn_n_s32(v2H, 4);

            int16x8_t v1s = vqdmulhq_s16(vcombine_s16(v1Ls, v1Hs), vrw);
            int16x8_t v2s = vqdmulhq_s16(vcombine_s16(v2Ls, v2Hs), vrW);

            int16x8_t vsum = vaddq_s16(vshrq_n_s16(v1s,1), vshrq_n_s16(v2s,1));
            uint8x8_t vres = vqrshrun_n_s16(vsum, 2);

            vst1_u8(internal::getRowPtr(dstBase, dstStride, row) + col, vres);
#endif
        }
    }
    if (col < (ptrdiff_t)dsize.width)
    {
        col = dsize.width - 8;
        goto downsample_bilinear_8uc1_col_loop8;
    }
}

} // namespace

#endif

void resizeLinearOpenCV(const Size2D &ssize, const Size2D &dsize,
                        const u8 * srcBase, ptrdiff_t srcStride,
                        u8 * dstBase, ptrdiff_t dstStride,
                        f32 wr, f32 hr, u32 channels)
{
    internal::assertSupportedConfiguration(wr > 0 && hr > 0 &&
                                           (dsize.width - 0.5) * wr - 0.5 < ssize.width &&
                                           (dsize.height - 0.5) * hr - 0.5 < ssize.height &&  // Ensure we have enough source data
                                           (dsize.width + 0.5) * wr + 0.5 >= ssize.width &&
                                           (dsize.height + 0.5) * hr + 0.5 >= ssize.height && // Ensure source isn't too big
                                           isResizeLinearOpenCVSupported(ssize, dsize, channels));
#ifdef CAROTENE_NEON
        if(1 == channels)
        {
            if (wr <= 1.f && hr <= 1.f)
                resizeLinearOpenCVchan<1>(ssize, dsize, srcBase, srcStride, dstBase, dstStride, wr, hr);
            else if (wr <= 2.0f && hr <= 2.0f && ssize.width >= 16)
                downsample_bilinear_8uc1(ssize, dsize, srcBase, srcStride, dstBase, dstStride, wr, hr);
            else
                resizeLinearOpenCVchan<1>(ssize, dsize, srcBase, srcStride, dstBase, dstStride, wr, hr);
        }
        else if(4 == channels)
            resizeLinearOpenCVchan<4>(ssize, dsize, srcBase, srcStride, dstBase, dstStride, wr, hr);
#else
    (void)ssize;
    (void)dsize;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)wr;
    (void)hr;
    (void)channels;
#endif
}

void resizeLinear(const Size2D &ssize, const Size2D &dsize,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  f32 wr, f32 hr, u32 channels)
{
    internal::assertSupportedConfiguration(wr > 0 && hr > 0 &&
                                           (dsize.width - 0.5) * wr - 0.5 < ssize.width &&
                                           (dsize.height - 0.5) * hr - 0.5 < ssize.height &&  // Ensure we have enough source data
                                           (dsize.width + 0.5) * wr + 0.5 >= ssize.width &&
                                           (dsize.height + 0.5) * hr + 0.5 >= ssize.height && // Ensure source isn't too big
                                           isResizeLinearSupported(ssize, dsize,
                                                                   wr, hr, channels));
#ifdef CAROTENE_NEON
    f32 scale_x = wr;
    f32 scale_x_offset = 0.5f * scale_x - 0.5f;
    f32 scale_y = hr;
    f32 scale_y_offset = 0.5f * scale_y - 0.5f;

    std::vector<ptrdiff_t> _buf(dsize.height * 3 + 1);
    std::vector<f32> coeff(dsize.height);
    ptrdiff_t * buf = &_buf[0];

    for (size_t row = 0; row < dsize.height; ++row)
    {
        f32 r = row * scale_y + scale_y_offset;
        ptrdiff_t src_row = floorf(r);
        ptrdiff_t src_row2 = src_row + 1;

        f32 rweight = src_row2 - r;
        buf[0 * dsize.height + row] = std::max<ptrdiff_t>(0, src_row);
        buf[1 * dsize.height + row] = std::min<ptrdiff_t>(ssize.height - 1, src_row2);
        coeff[row] = rweight;
    }

    size_t col = 0;
    for ( ; col + 16 <= dsize.width; col += 16)
    {
        ptrdiff_t col1[16], col2[16];
        f32 cwi[16];

        for(s32 k = 0; k < 16; ++k)
        {
            f32 c = (col + k) * scale_x + scale_x_offset;
            col1[k] = floorf(c);
            col2[k] = col1[k] + 1;

            cwi[k] = col2[k] - c;

            if (col1[k] < 0)
                col1[k] = 0;
            if (col2[k] >= (ptrdiff_t)ssize.width)
                col2[k] = ssize.width - 1;
        }

        ptrdiff_t x = std::min<ptrdiff_t>(col1[0], ssize.width - 16);
        ptrdiff_t y = std::min<ptrdiff_t>(col1[8], ssize.width - 16);
        u8 lutl[16], luth[16];

        for (s32 k = 0; k < 8; ++k)
        {
            lutl[k] = (u8)(col1[k] - x);
            luth[k] = (u8)(col2[k] - x);
            lutl[k + 8] = (u8)(col1[k + 8] - y);
            luth[k + 8] = (u8)(col2[k + 8] - y);
        }

        uint8x8_t vlutl = vld1_u8(lutl);
        uint8x8_t vluth = vld1_u8(luth);
        float32x4_t vcw0 = vld1q_f32(cwi);
        float32x4_t vcw1 = vld1q_f32(cwi + 4);

        uint8x8_t vlutl_ = vld1_u8(lutl + 8);
        uint8x8_t vluth_ = vld1_u8(luth + 8);
        float32x4_t vcw0_ = vld1q_f32(cwi + 8);
        float32x4_t vcw1_ = vld1q_f32(cwi + 12);

        if (channels == 1)
        {
            for (size_t row = 0; row < dsize.height; ++row)
            {
                float32x4_t vrw = vdupq_n_f32(coeff[row]);

                const u8 * srow0 = internal::getRowPtr(srcBase, srcStride, buf[0 * dsize.height + row]);
                const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, buf[1 * dsize.height + row]);
                u8 * drow = internal::getRowPtr(dstBase, dstStride, row);

                internal::prefetch(srow0 + x + 2 * srcStride);
                internal::prefetch(srow1 + x + 2 * srcStride);

                uint8x8_t vres0 = resizeLinearStep(vld1q_u8(srow0 + x), vld1q_u8(srow1 + x),
                                                   vlutl, vluth,
                                                   vrw, vcw0, vcw1);

                uint8x8_t vres1 = resizeLinearStep(vld1q_u8(srow0 + y), vld1q_u8(srow1 + y),
                                                   vlutl_, vluth_,
                                                   vrw, vcw0_, vcw1_);

                vst1q_u8(drow + col, vcombine_u8(vres0, vres1));
            }
        }
        else if (channels == 3)
        {
            for (size_t row = 0; row < dsize.height; ++row)
            {
                float32x4_t vrw = vdupq_n_f32(coeff[row]);

                const u8 * srow0 = internal::getRowPtr(srcBase, srcStride, buf[0 * dsize.height + row]);
                const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, buf[1 * dsize.height + row]);
                u8 * drow = internal::getRowPtr(dstBase, dstStride, row);

                internal::prefetch(srow0 + x + 2 * srcStride);
                internal::prefetch(srow1 + x + 2 * srcStride);

                uint8x16x3_t v_src10 = vld3q_u8(srow0 + (x * 3));
                uint8x16x3_t v_src20 = vld3q_u8(srow1 + (x * 3));

                uint8x16x3_t v_src11 = vld3q_u8(srow0 + (y * 3));
                uint8x16x3_t v_src21 = vld3q_u8(srow1 + (y * 3));

                uint8x16x3_t v_dst;

                v_dst.val[0] = vcombine_u8(resizeLinearStep(v_src10.val[0], v_src20.val[0], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[0], v_src21.val[0], vlutl_, vluth_, vrw, vcw0_, vcw1_));
                v_dst.val[1] = vcombine_u8(resizeLinearStep(v_src10.val[1], v_src20.val[1], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[1], v_src21.val[1], vlutl_, vluth_, vrw, vcw0_, vcw1_));
                v_dst.val[2] = vcombine_u8(resizeLinearStep(v_src10.val[2], v_src20.val[2], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[2], v_src21.val[2], vlutl_, vluth_, vrw, vcw0_, vcw1_));

                vst3q_u8(drow + (col * 3), v_dst);
            }
        }
        else if (channels == 4)
        {
            for (size_t row = 0; row < dsize.height; ++row)
            {
                float32x4_t vrw = vdupq_n_f32(coeff[row]);

                const u8 * srow0 = internal::getRowPtr(srcBase, srcStride, buf[0 * dsize.height + row]);
                const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, buf[1 * dsize.height + row]);
                u8 * drow = internal::getRowPtr(dstBase, dstStride, row);

                internal::prefetch(srow0 + x + 2 * srcStride);
                internal::prefetch(srow1 + x + 2 * srcStride);

                uint8x16x4_t v_src10 = vld4q_u8(srow0 + (x << 2));
                uint8x16x4_t v_src20 = vld4q_u8(srow1 + (x << 2));

                uint8x16x4_t v_src11 = vld4q_u8(srow0 + (y << 2));
                uint8x16x4_t v_src21 = vld4q_u8(srow1 + (y << 2));

                uint8x16x4_t v_dst;

                v_dst.val[0] = vcombine_u8(resizeLinearStep(v_src10.val[0], v_src20.val[0], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[0], v_src21.val[0], vlutl_, vluth_, vrw, vcw0_, vcw1_));
                v_dst.val[1] = vcombine_u8(resizeLinearStep(v_src10.val[1], v_src20.val[1], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[1], v_src21.val[1], vlutl_, vluth_, vrw, vcw0_, vcw1_));
                v_dst.val[2] = vcombine_u8(resizeLinearStep(v_src10.val[2], v_src20.val[2], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[2], v_src21.val[2], vlutl_, vluth_, vrw, vcw0_, vcw1_));
                v_dst.val[3] = vcombine_u8(resizeLinearStep(v_src10.val[3], v_src20.val[3], vlutl, vluth, vrw, vcw0, vcw1),
                                           resizeLinearStep(v_src11.val[3], v_src21.val[3], vlutl_, vluth_, vrw, vcw0_, vcw1_));

                vst4q_u8(drow + (col << 2), v_dst);
            }
        }
    }

    for ( ; col + 8 <= dsize.width; col += 8)
    {
downsample_bilinear_8uc1_col_loop8:
        ptrdiff_t col1[8], col2[8];
        f32 cwi[8];

        for (s32 k = 0; k < 8; ++k)
        {
            f32 c = (col + k) * scale_x + scale_x_offset;
            col1[k] = floorf(c);
            col2[k] = col1[k] + 1;

            cwi[k] = col2[k] - c;

            if (col1[k] < 0)
                col1[k] = 0;
            if (col2[k] >= (ptrdiff_t)ssize.width)
                col2[k] = ssize.width - 1;
        }

        ptrdiff_t x = std::min<ptrdiff_t>(col1[0], ssize.width - 16);
        u8 lutl[8], luth[8];
        for (s32 k = 0; k < 8; ++k)
        {
            lutl[k] = (u8)(col1[k] - x);
            luth[k] = (u8)(col2[k] - x);
        }

        uint8x8_t vlutl = vld1_u8(lutl);
        uint8x8_t vluth = vld1_u8(luth);
        float32x4_t vcw0 = vld1q_f32(cwi);
        float32x4_t vcw1 = vld1q_f32(cwi + 4);

        if (channels == 1)
        {
            for (size_t row = 0; row < dsize.height; ++row)
            {
                float32x4_t vrw = vdupq_n_f32(coeff[row]);

                const u8 * srow0 = internal::getRowPtr(srcBase, srcStride, buf[0 * dsize.height + row]);
                const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, buf[1 * dsize.height + row]);
                u8 * drow = internal::getRowPtr(dstBase, dstStride, row);

                internal::prefetch(srow0 + x + 2 * srcStride);
                internal::prefetch(srow1 + x + 2 * srcStride);

                uint8x8_t vres = resizeLinearStep(vld1q_u8(srow0 + x), vld1q_u8(srow1 + x),
                                                  vlutl, vluth,
                                                  vrw, vcw0, vcw1);
                vst1_u8(drow + col, vres);
            }
        }
        else if (channels == 3)
        {
            for (size_t row = 0; row < dsize.height; ++row)
            {
                float32x4_t vrw = vdupq_n_f32(coeff[row]);

                const u8 * srow0 = internal::getRowPtr(srcBase, srcStride, buf[0 * dsize.height + row]);
                const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, buf[1 * dsize.height + row]);
                u8 * drow = internal::getRowPtr(dstBase, dstStride, row);

                internal::prefetch(srow0 + x + 2 * srcStride);
                internal::prefetch(srow1 + x + 2 * srcStride);

                uint8x16x3_t v_src1 = vld3q_u8(srow0 + (x * 3));
                uint8x16x3_t v_src2 = vld3q_u8(srow1 + (x * 3));

                uint8x8x3_t v_dst;

                v_dst.val[0] = resizeLinearStep(v_src1.val[0], v_src2.val[0], vlutl, vluth, vrw, vcw0, vcw1);
                v_dst.val[1] = resizeLinearStep(v_src1.val[1], v_src2.val[1], vlutl, vluth, vrw, vcw0, vcw1);
                v_dst.val[2] = resizeLinearStep(v_src1.val[2], v_src2.val[2], vlutl, vluth, vrw, vcw0, vcw1);

                vst3_u8(drow + (col * 3), v_dst);
            }
        }
        else if (channels == 4)
        {
            for (size_t row = 0; row < dsize.height; ++row)
            {
                float32x4_t vrw = vdupq_n_f32(coeff[row]);

                const u8 * srow0 = internal::getRowPtr(srcBase, srcStride, buf[0 * dsize.height + row]);
                const u8 * srow1 = internal::getRowPtr(srcBase, srcStride, buf[1 * dsize.height + row]);
                u8 * drow = internal::getRowPtr(dstBase, dstStride, row);

                internal::prefetch(srow0 + x + 2 * srcStride);
                internal::prefetch(srow1 + x + 2 * srcStride);

                uint8x16x4_t v_src1 = vld4q_u8(srow0 + (x << 2));
                uint8x16x4_t v_src2 = vld4q_u8(srow1 + (x << 2));

                uint8x8x4_t v_dst;

                v_dst.val[0] = resizeLinearStep(v_src1.val[0], v_src2.val[0], vlutl, vluth, vrw, vcw0, vcw1);
                v_dst.val[1] = resizeLinearStep(v_src1.val[1], v_src2.val[1], vlutl, vluth, vrw, vcw0, vcw1);
                v_dst.val[2] = resizeLinearStep(v_src1.val[2], v_src2.val[2], vlutl, vluth, vrw, vcw0, vcw1);
                v_dst.val[3] = resizeLinearStep(v_src1.val[3], v_src2.val[3], vlutl, vluth, vrw, vcw0, vcw1);

                vst4_u8(drow + (col << 2), v_dst);
            }
        }
    }

    if (col < dsize.width)
    {
        col = dsize.width - 8;
        goto downsample_bilinear_8uc1_col_loop8;
    }

#else
    (void)ssize;
    (void)dsize;
    (void)srcBase;
    (void)srcStride;
    (void)dstBase;
    (void)dstStride;
    (void)wr;
    (void)hr;
    (void)channels;
#endif
}

} // namespace CAROTENE_NS
