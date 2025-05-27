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

#ifndef CAROTENE_FUNCTIONS_HPP
#define CAROTENE_FUNCTIONS_HPP

#include <carotene/definitions.hpp>
#include <carotene/types.hpp>

namespace CAROTENE_NS {
    /* If this returns false, none of the functions will work. */
    bool isSupportedConfiguration();

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] + src1[p]
    */
    void add(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             u8 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const s8 *src0Base, ptrdiff_t src0Stride,
             const s8 *src1Base, ptrdiff_t src1Stride,
             s8 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const s16 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const u16 * src0Base, ptrdiff_t src0Stride,
             const u16 * src1Base, ptrdiff_t src1Stride,
             u16 * dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const s32 * src0Base, ptrdiff_t src0Stride,
             const s32 * src1Base, ptrdiff_t src1Stride,
             s32 * dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const u32 * src0Base, ptrdiff_t src0Stride,
             const u32 * src1Base, ptrdiff_t src1Stride,
             u32 * dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void add(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] - src1[p]
    */
    void sub(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             u8 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             f32 *dstBase, ptrdiff_t dstStride);

    void sub(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const s16 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const s16 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const s8 *src0Base, ptrdiff_t src0Stride,
             const s8 *src1Base, ptrdiff_t src1Stride,
             s8 *dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const u16 * src0Base, ptrdiff_t src0Stride,
             const u16 * src1Base, ptrdiff_t src1Stride,
             u16 * dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const s32 * src0Base, ptrdiff_t src0Stride,
             const s32 * src1Base, ptrdiff_t src1Stride,
             s32 * dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const u32 * src0Base, ptrdiff_t src0Stride,
             const u32 * src1Base, ptrdiff_t src1Stride,
             u32 * dstBase, ptrdiff_t dstStride,
             CONVERT_POLICY policy);

    void sub(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] * alpha + src1[p] * beta + gamma
    */
    void addWeighted(const Size2D &size,
                     const u8 * src0Base, ptrdiff_t src0Stride,
                     const u8 * src1Base, ptrdiff_t src1Stride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    void addWeighted(const Size2D &size,
                     const s8 * src0Base, ptrdiff_t src0Stride,
                     const s8 * src1Base, ptrdiff_t src1Stride,
                     s8 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    void addWeighted(const Size2D &size,
                     const u16 * src0Base, ptrdiff_t src0Stride,
                     const u16 * src1Base, ptrdiff_t src1Stride,
                     u16 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    void addWeighted(const Size2D &size,
                     const s16 * src0Base, ptrdiff_t src0Stride,
                     const s16 * src1Base, ptrdiff_t src1Stride,
                     s16 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    void addWeighted(const Size2D &size,
                     const u32 * src0Base, ptrdiff_t src0Stride,
                     const u32 * src1Base, ptrdiff_t src1Stride,
                     u32 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    void addWeighted(const Size2D &size,
                     const s32 * src0Base, ptrdiff_t src0Stride,
                     const s32 * src1Base, ptrdiff_t src1Stride,
                     s32 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    void addWeighted(const Size2D &size,
                     const f32 * src0Base, ptrdiff_t src0Stride,
                     const f32 * src1Base, ptrdiff_t src1Stride,
                     f32 * dstBase, ptrdiff_t dstStride,
                     f32 alpha, f32 beta, f32 gamma);

    /*
        For each point `p` within `size`, do:
        dst[p] = min(src0[p], src1[p])
    */
    void min(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             u8 *dstBase, ptrdiff_t dstStride);

    void min(const Size2D &size,
             const s8 *src0Base, ptrdiff_t src0Stride,
             const s8 *src1Base, ptrdiff_t src1Stride,
             s8 *dstBase, ptrdiff_t dstStride);

    void min(const Size2D &size,
             const u16 * src0Base, ptrdiff_t src0Stride,
             const u16 * src1Base, ptrdiff_t src1Stride,
             u16 * dstBase, ptrdiff_t dstStride);

    void min(const Size2D &size,
             const s16 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride);

    void min(const Size2D &size,
             const s32 * src0Base, ptrdiff_t src0Stride,
             const s32 * src1Base, ptrdiff_t src1Stride,
             s32 * dstBase, ptrdiff_t dstStride);

    void min(const Size2D &size,
             const u32 * src0Base, ptrdiff_t src0Stride,
             const u32 * src1Base, ptrdiff_t src1Stride,
             u32 * dstBase, ptrdiff_t dstStride);

    void min(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = max(src0[p], src1[p])
    */
    void max(const Size2D &size,
             const u8 *src0Base, ptrdiff_t src0Stride,
             const u8 *src1Base, ptrdiff_t src1Stride,
             u8 *dstBase, ptrdiff_t dstStride);

    void max(const Size2D &size,
             const s8 *src0Base, ptrdiff_t src0Stride,
             const s8 *src1Base, ptrdiff_t src1Stride,
             s8 *dstBase, ptrdiff_t dstStride);

    void max(const Size2D &size,
             const u16 * src0Base, ptrdiff_t src0Stride,
             const u16 * src1Base, ptrdiff_t src1Stride,
             u16 * dstBase, ptrdiff_t dstStride);

    void max(const Size2D &size,
             const s16 *src0Base, ptrdiff_t src0Stride,
             const s16 *src1Base, ptrdiff_t src1Stride,
             s16 *dstBase, ptrdiff_t dstStride);

    void max(const Size2D &size,
             const s32 * src0Base, ptrdiff_t src0Stride,
             const s32 * src1Base, ptrdiff_t src1Stride,
             s32 * dstBase, ptrdiff_t dstStride);

    void max(const Size2D &size,
             const u32 * src0Base, ptrdiff_t src0Stride,
             const u32 * src1Base, ptrdiff_t src1Stride,
             u32 * dstBase, ptrdiff_t dstStride);

    void max(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] * src1[p] * scale

        NOTE: ROUND_TO_ZERO convert policy is used
    */
    void mul(const Size2D &size,
             const u8 * src0Base, ptrdiff_t src0Stride,
             const u8 * src1Base, ptrdiff_t src1Stride,
             u8 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const u8 * src0Base, ptrdiff_t src0Stride,
             const u8 * src1Base, ptrdiff_t src1Stride,
             s16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const u8 * src0Base, ptrdiff_t src0Stride,
             const s16 * src1Base, ptrdiff_t src1Stride,
             s16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const s8 * src0Base, ptrdiff_t src0Stride,
             const s8 * src1Base, ptrdiff_t src1Stride,
             s8 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const u16 * src0Base, ptrdiff_t src0Stride,
             const u16 * src1Base, ptrdiff_t src1Stride,
             u16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const s16 * src0Base, ptrdiff_t src0Stride,
             const s16 * src1Base, ptrdiff_t src1Stride,
             s16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const s32 * src0Base, ptrdiff_t src0Stride,
             const s32 * src1Base, ptrdiff_t src1Stride,
             s32 * dstBase, ptrdiff_t dstStride,
             f64 scale,
             CONVERT_POLICY cpolicy);

    void mul(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride,
             f32 scale);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] * scale / src1[p]

        NOTE: ROUND_TO_ZERO convert policy is used
    */
    void div(const Size2D &size,
             const u8 * src0Base, ptrdiff_t src0Stride,
             const u8 * src1Base, ptrdiff_t src1Stride,
             u8 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const u8 * src0Base, ptrdiff_t src0Stride,
             const u8 * src1Base, ptrdiff_t src1Stride,
             s16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const u8 * src0Base, ptrdiff_t src0Stride,
             const s16 * src1Base, ptrdiff_t src1Stride,
             s16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const s8 * src0Base, ptrdiff_t src0Stride,
             const s8 * src1Base, ptrdiff_t src1Stride,
             s8 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const u16 * src0Base, ptrdiff_t src0Stride,
             const u16 * src1Base, ptrdiff_t src1Stride,
             u16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const s16 * src0Base, ptrdiff_t src0Stride,
             const s16 * src1Base, ptrdiff_t src1Stride,
             s16 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const s32 * src0Base, ptrdiff_t src0Stride,
             const s32 * src1Base, ptrdiff_t src1Stride,
             s32 * dstBase, ptrdiff_t dstStride,
             f32 scale,
             CONVERT_POLICY cpolicy);

    void div(const Size2D &size,
             const f32 * src0Base, ptrdiff_t src0Stride,
             const f32 * src1Base, ptrdiff_t src1Stride,
             f32 * dstBase, ptrdiff_t dstStride,
             f32 scale);

    /*
        For each point `p` within `size`, do:
        dst[p] = scale / src[p]

        NOTE: ROUND_TO_ZERO convert policy is used
    */
    void reciprocal(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    f32 scale,
                    CONVERT_POLICY cpolicy);

    void reciprocal(const Size2D &size,
                    const s8 * srcBase, ptrdiff_t srcStride,
                    s8 * dstBase, ptrdiff_t dstStride,
                    f32 scale,
                    CONVERT_POLICY cpolicy);

    void reciprocal(const Size2D &size,
                    const u16 * srcBase, ptrdiff_t srcStride,
                    u16 * dstBase, ptrdiff_t dstStride,
                    f32 scale,
                    CONVERT_POLICY cpolicy);

    void reciprocal(const Size2D &size,
                    const s16 * srcBase, ptrdiff_t srcStride,
                    s16 * dstBase, ptrdiff_t dstStride,
                    f32 scale,
                    CONVERT_POLICY cpolicy);

    void reciprocal(const Size2D &size,
                    const s32 * srcBase, ptrdiff_t srcStride,
                    s32 * dstBase, ptrdiff_t dstStride,
                    f32 scale,
                    CONVERT_POLICY cpolicy);

    void reciprocal(const Size2D &size,
                    const f32 * srcBase, ptrdiff_t srcStride,
                    f32 * dstBase, ptrdiff_t dstStride,
                    f32 scale);

    /*
        For each point `p` within `size`, set `dst[p]` to the median
        of `src[p]` and the 8 points around it. If `srcMargin` is
        zero on any side, get the neighbors on that side by replicating
        the edge.
    */
    bool isMedianFilter3x3Supported(const Size2D &size, u32 numChannels);
    void medianFilter3x3(const Size2D &size, u32 numChannels,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         const Margin &srcMargin,
                         u8 *dstBase, ptrdiff_t dstStride);

    /*
        Apply a half Gaussian filter + half Scale, as one level of a Gaussian
        pyramid. For all `p` within `dstSize`, set `dst[p]` to `f[2 * p]`, where
        `f` is an image of size srcSize obtained by filtering src with the 5x5
        Gaussian kernel ([1 4 6 4 1]'*[1 4 6 4 1]/256) using the border mode
        passed in, and round-to-zero rounding.
        dstSize must be (srcSize.width / 2, srcSize.height / 2), rounded by any method.
     */
    bool isGaussianPyramidDownRTZSupported(const Size2D &srcSize, const Size2D &dstSize, BORDER_MODE border);
    void gaussianPyramidDownRTZ(const Size2D &srcSize,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         const Size2D &dstSize,
                         u8 *dstBase, ptrdiff_t dstStride,
                         BORDER_MODE border, u8 borderValue);

    /* Same as above, but uses round-half-up rounding. */

    bool isGaussianPyramidDownU8Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn);
    void gaussianPyramidDown(const Size2D &srcSize,
                             const u8 *srcBase, ptrdiff_t srcStride,
                             const Size2D &dstSize,
                             u8 *dstBase, ptrdiff_t dstStride, u8 cn);


    bool isGaussianPyramidDownS16Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn);
    void gaussianPyramidDown(const Size2D &srcSize,
                             const s16 *srcBase, ptrdiff_t srcStride,
                             const Size2D &dstSize,
                             s16 *dstBase, ptrdiff_t dstStride, u8 cn);

    bool isGaussianPyramidDownF32Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn);
    void gaussianPyramidDown(const Size2D &srcSize,
                             const f32 *srcBase, ptrdiff_t srcStride,
                             const Size2D &dstSize,
                             f32 *dstBase, ptrdiff_t dstStride, u8 cn);

    bool isGaussianPyramidUpU8Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn);
    void gaussianPyramidUp(const Size2D &srcSize,
                           const u8 *srcBase, ptrdiff_t srcStride,
                           const Size2D &dstSize,
                           u8 *dstBase, ptrdiff_t dstStride, u8 cn);

    bool isGaussianPyramidUpS16Supported(const Size2D &srcSize, const Size2D &dstSize, u8 cn);
    void gaussianPyramidUp(const Size2D &srcSize,
                           const s16 *srcBase, ptrdiff_t srcStride,
                           const Size2D &dstSize,
                           s16 *dstBase, ptrdiff_t dstStride, u8 cn);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] > threshold ? trueValue : falseValue
    */
    void thresholdBinary(const Size2D &size,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         u8 *dstBase, ptrdiff_t dstStride,
                         u8 threshold, u8 trueValue = 255, u8 falseValue = 0);

    /*
        For each point `p` within `size`, do:
        dst[p] = lower <= src[p] && src[p] <= upper ? trueValue : falseValue
    */
    void thresholdRange(const Size2D &size,
                        const u8 *srcBase, ptrdiff_t srcStride,
                        u8 *dstBase, ptrdiff_t dstStride,
                        u8 lowerThreshold, u8 upperThreshold,
                        u8 trueValue = 255, u8 falseValue = 0);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] > threshold ? value : 0
    */
    void thresholdBinary(const Size2D &size,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         u8 *dstBase, ptrdiff_t dstStride,
                         u8 threshold, u8 value);

    void thresholdBinary(const Size2D &size,
                         const s8 *srcBase, ptrdiff_t srcStride,
                         s8 *dstBase, ptrdiff_t dstStride,
                         s8 threshold, s8 value);

    void thresholdBinary(const Size2D &size,
                         const u16 *srcBase, ptrdiff_t srcStride,
                         u16 *dstBase, ptrdiff_t dstStride,
                         u16 threshold, u16 value);

    void thresholdBinary(const Size2D &size,
                         const s16 *srcBase, ptrdiff_t srcStride,
                         s16 *dstBase, ptrdiff_t dstStride,
                         s16 threshold, s16 value);

    void thresholdBinary(const Size2D &size,
                         const s32 *srcBase, ptrdiff_t srcStride,
                         s32 *dstBase, ptrdiff_t dstStride,
                         s32 threshold, s32 value);

    void thresholdBinary(const Size2D &size,
                         const f32 *srcBase, ptrdiff_t srcStride,
                         f32 *dstBase, ptrdiff_t dstStride,
                         f32 threshold, f32 value);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] > threshold ? 0 : value
    */
    void thresholdBinaryInv(const Size2D &size,
                            const u8 *srcBase, ptrdiff_t srcStride,
                            u8 *dstBase, ptrdiff_t dstStride,
                            u8 threshold, u8 value);

    void thresholdBinaryInv(const Size2D &size,
                            const s8 *srcBase, ptrdiff_t srcStride,
                            s8 *dstBase, ptrdiff_t dstStride,
                            s8 threshold, s8 value);

    void thresholdBinaryInv(const Size2D &size,
                            const u16 *srcBase, ptrdiff_t srcStride,
                            u16 *dstBase, ptrdiff_t dstStride,
                            u16 threshold, u16 value);

    void thresholdBinaryInv(const Size2D &size,
                            const s16 *srcBase, ptrdiff_t srcStride,
                            s16 *dstBase, ptrdiff_t dstStride,
                            s16 threshold, s16 value);

    void thresholdBinaryInv(const Size2D &size,
                            const s32 *srcBase, ptrdiff_t srcStride,
                            s32 *dstBase, ptrdiff_t dstStride,
                            s32 threshold, s32 value);

    void thresholdBinaryInv(const Size2D &size,
                            const f32 *srcBase, ptrdiff_t srcStride,
                            f32 *dstBase, ptrdiff_t dstStride,
                            f32 threshold, f32 value);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] > threshold ? threshold : src[p]
    */
    void thresholdTruncate(const Size2D &size,
                           const u8 *srcBase, ptrdiff_t srcStride,
                           u8 *dstBase, ptrdiff_t dstStride,
                           u8 threshold);

    void thresholdTruncate(const Size2D &size,
                           const s8 *srcBase, ptrdiff_t srcStride,
                           s8 *dstBase, ptrdiff_t dstStride,
                           s8 threshold);

    void thresholdTruncate(const Size2D &size,
                           const u16 *srcBase, ptrdiff_t srcStride,
                           u16 *dstBase, ptrdiff_t dstStride,
                           u16 threshold);

    void thresholdTruncate(const Size2D &size,
                           const s16 *srcBase, ptrdiff_t srcStride,
                           s16 *dstBase, ptrdiff_t dstStride,
                           s16 threshold);

    void thresholdTruncate(const Size2D &size,
                           const s32 *srcBase, ptrdiff_t srcStride,
                           s32 *dstBase, ptrdiff_t dstStride,
                           s32 threshold);

    void thresholdTruncate(const Size2D &size,
                           const f32 *srcBase, ptrdiff_t srcStride,
                           f32 *dstBase, ptrdiff_t dstStride,
                           f32 threshold);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] > threshold ? src[p] : 0
    */
    void thresholdToZero(const Size2D &size,
                         const u8 *srcBase, ptrdiff_t srcStride,
                         u8 *dstBase, ptrdiff_t dstStride,
                         u8 threshold);

    void thresholdToZero(const Size2D &size,
                         const s8 *srcBase, ptrdiff_t srcStride,
                         s8 *dstBase, ptrdiff_t dstStride,
                         s8 threshold);

    void thresholdToZero(const Size2D &size,
                         const u16 *srcBase, ptrdiff_t srcStride,
                         u16 *dstBase, ptrdiff_t dstStride,
                         u16 threshold);

    void thresholdToZero(const Size2D &size,
                         const s16 *srcBase, ptrdiff_t srcStride,
                         s16 *dstBase, ptrdiff_t dstStride,
                         s16 threshold);

    void thresholdToZero(const Size2D &size,
                         const s32 *srcBase, ptrdiff_t srcStride,
                         s32 *dstBase, ptrdiff_t dstStride,
                         s32 threshold);

    void thresholdToZero(const Size2D &size,
                         const f32 *srcBase, ptrdiff_t srcStride,
                         f32 *dstBase, ptrdiff_t dstStride,
                         f32 threshold);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] > threshold ? 0 : src[p]
    */
    void thresholdToZeroInv(const Size2D &size,
                            const u8 *srcBase, ptrdiff_t srcStride,
                            u8 *dstBase, ptrdiff_t dstStride,
                            u8 threshold);

    void thresholdToZeroInv(const Size2D &size,
                            const s8 *srcBase, ptrdiff_t srcStride,
                            s8 *dstBase, ptrdiff_t dstStride,
                            s8 threshold);

    void thresholdToZeroInv(const Size2D &size,
                            const u16 *srcBase, ptrdiff_t srcStride,
                            u16 *dstBase, ptrdiff_t dstStride,
                            u16 threshold);

    void thresholdToZeroInv(const Size2D &size,
                            const s16 *srcBase, ptrdiff_t srcStride,
                            s16 *dstBase, ptrdiff_t dstStride,
                            s16 threshold);

    void thresholdToZeroInv(const Size2D &size,
                            const s32 *srcBase, ptrdiff_t srcStride,
                            s32 *dstBase, ptrdiff_t dstStride,
                            s32 threshold);

    void thresholdToZeroInv(const Size2D &size,
                            const f32 *srcBase, ptrdiff_t srcStride,
                            f32 *dstBase, ptrdiff_t dstStride,
                            f32 threshold);

    /*
        For each point `p` within `size`, do:
        dst[p] = abs(src0[p] - src1[p])
    */
    void absDiff(const Size2D &size,
                 const u8 *src0Base, ptrdiff_t src0Stride,
                 const u8 *src1Base, ptrdiff_t src1Stride,
                 u8 *dstBase, ptrdiff_t dstStride);

    void absDiff(const Size2D &size,
                 const u16 *src0Base, ptrdiff_t src0Stride,
                 const u16 *src1Base, ptrdiff_t src1Stride,
                 u16 *dstBase, ptrdiff_t dstStride);

    void absDiff(const Size2D &size,
                 const s8 *src0Base, ptrdiff_t src0Stride,
                 const s8 *src1Base, ptrdiff_t src1Stride,
                 s8 *dstBase, ptrdiff_t dstStride);

    void absDiff(const Size2D &size,
                 const s16 *src0Base, ptrdiff_t src0Stride,
                 const s16 *src1Base, ptrdiff_t src1Stride,
                 s16 *dstBase, ptrdiff_t dstStride);

    void absDiff(const Size2D &size,
                 const s32 * src0Base, ptrdiff_t src0Stride,
                 const s32 * src1Base, ptrdiff_t src1Stride,
                 s32 * dstBase, ptrdiff_t dstStride);

    void absDiff(const Size2D &size,
                 const f32 * src0Base, ptrdiff_t src0Stride,
                 const f32 * src1Base, ptrdiff_t src1Stride,
                 f32 * dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = ~src[p]
    */
    void bitwiseNot(const Size2D &size,
                    const u8 *srcBase, ptrdiff_t srcStride,
                    u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] & src1[p]
    */
    void bitwiseAnd(const Size2D &size,
                    const u8 *src0Base, ptrdiff_t src0Stride,
                    const u8 *src1Base, ptrdiff_t src1Stride,
                    u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] | src1[p]
    */
    void bitwiseOr(const Size2D &size,
                   const u8 *src0Base, ptrdiff_t src0Stride,
                   const u8 *src1Base, ptrdiff_t src1Stride,
                   u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] ^ src1[p]
    */
    void bitwiseXor(const Size2D &size,
                    const u8 *src0Base, ptrdiff_t src0Stride,
                    const u8 *src1Base, ptrdiff_t src1Stride,
                    u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] == src1[p] ? 255 : 0
    */
    void cmpEQ(const Size2D &size,
               const u8 *src0Base, ptrdiff_t src0Stride,
               const u8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpEQ(const Size2D &size,
               const s8 *src0Base, ptrdiff_t src0Stride,
               const s8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpEQ(const Size2D &size,
               const u16 *src0Base, ptrdiff_t src0Stride,
               const u16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpEQ(const Size2D &size,
               const s16 *src0Base, ptrdiff_t src0Stride,
               const s16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpEQ(const Size2D &size,
               const u32 *src0Base, ptrdiff_t src0Stride,
               const u32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpEQ(const Size2D &size,
               const s32 *src0Base, ptrdiff_t src0Stride,
               const s32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpEQ(const Size2D &size,
               const f32 *src0Base, ptrdiff_t src0Stride,
               const f32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] != src1[p] ? 255 : 0
    */
    void cmpNE(const Size2D &size,
               const u8 *src0Base, ptrdiff_t src0Stride,
               const u8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpNE(const Size2D &size,
               const s8 *src0Base, ptrdiff_t src0Stride,
               const s8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpNE(const Size2D &size,
               const u16 *src0Base, ptrdiff_t src0Stride,
               const u16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpNE(const Size2D &size,
               const s16 *src0Base, ptrdiff_t src0Stride,
               const s16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpNE(const Size2D &size,
               const u32 *src0Base, ptrdiff_t src0Stride,
               const u32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpNE(const Size2D &size,
               const s32 *src0Base, ptrdiff_t src0Stride,
               const s32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpNE(const Size2D &size,
               const f32 *src0Base, ptrdiff_t src0Stride,
               const f32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] > src1[p] ? 255 : 0
    */
    void cmpGT(const Size2D &size,
               const u8 *src0Base, ptrdiff_t src0Stride,
               const u8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGT(const Size2D &size,
               const s8 *src0Base, ptrdiff_t src0Stride,
               const s8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGT(const Size2D &size,
               const u16 *src0Base, ptrdiff_t src0Stride,
               const u16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGT(const Size2D &size,
               const s16 *src0Base, ptrdiff_t src0Stride,
               const s16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGT(const Size2D &size,
               const u32 *src0Base, ptrdiff_t src0Stride,
               const u32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGT(const Size2D &size,
               const s32 *src0Base, ptrdiff_t src0Stride,
               const s32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGT(const Size2D &size,
               const f32 *src0Base, ptrdiff_t src0Stride,
               const f32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src0[p] >= src1[p] ? 255 : 0
    */
    void cmpGE(const Size2D &size,
               const u8 *src0Base, ptrdiff_t src0Stride,
               const u8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGE(const Size2D &size,
               const s8 *src0Base, ptrdiff_t src0Stride,
               const s8 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGE(const Size2D &size,
               const u16 *src0Base, ptrdiff_t src0Stride,
               const u16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGE(const Size2D &size,
               const s16 *src0Base, ptrdiff_t src0Stride,
               const s16 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGE(const Size2D &size,
               const u32 *src0Base, ptrdiff_t src0Stride,
               const u32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGE(const Size2D &size,
               const s32 *src0Base, ptrdiff_t src0Stride,
               const s32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    void cmpGE(const Size2D &size,
               const f32 *src0Base, ptrdiff_t src0Stride,
               const f32 *src1Base, ptrdiff_t src1Stride,
               u8 *dstBase, ptrdiff_t dstStride);

    /*
        Calculates dot product
    */
    f64 dotProduct(const Size2D &size,
                   const u8 * src0Base, ptrdiff_t src0Stride,
                   const u8 * src1Base, ptrdiff_t src1Stride);

    f64 dotProduct(const Size2D &size,
                   const s8 * src0Base, ptrdiff_t src0Stride,
                   const s8 * src1Base, ptrdiff_t src1Stride);

    f64 dotProduct(const Size2D &size,
                   const f32 * src0Base, ptrdiff_t src0Stride,
                   const f32 * src1Base, ptrdiff_t src1Stride);

    /*
        Calculates mean and stddev
    */
    void meanStdDev(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    f32 * pMean, f32 * pStdDev);

    void meanStdDev(const Size2D &size,
                const u16 * srcBase, ptrdiff_t srcStride,
                f32 * pMean, f32 * pStdDev);

    /*
        For each point `p` within `size`, do:
        dst[p] = sqrt(src0[p] ^ 2 + src1[p] ^ 2)
    */
    void magnitude(const Size2D &size,
                   const s16 *src0Base, ptrdiff_t src0Stride,
                   const s16 *src1Base, ptrdiff_t src1Stride,
                   s16 *dstBase, ptrdiff_t dstStride);

    void magnitude(const Size2D &size,
                   const f32 *src0Base, ptrdiff_t src0Stride,
                   const f32 *src1Base, ptrdiff_t src1Stride,
                   f32 *dstBase, ptrdiff_t dstStride);

    /*
        Compute an integral image
    */
    void integral(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u32 * sumBase, ptrdiff_t sumStride);

    /*
        Compute an integral of squared image values
    */
    void sqrIntegral(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     f64 * sqsumBase, ptrdiff_t sqsumStride);

    /*
        Among each pixel `p` within `src` find min and max values
    */
    void minMaxVals(const Size2D &size,
                    const u8 *srcBase, ptrdiff_t srcStride,
                    u8 * minVal, u8 * maxVal);

    void minMaxVals(const Size2D &size,
                    const s16 *srcBase, ptrdiff_t srcStride,
                    s16 * minVal, s16 * maxVal);

    void minMaxVals(const Size2D &size,
                    const u16 *srcBase, ptrdiff_t srcStride,
                    u16 * minVal, u16 * maxVal);

    void minMaxVals(const Size2D &size,
                    const s32 *srcBase, ptrdiff_t srcStride,
                    s32 * minVal, s32 * maxVal);

    void minMaxVals(const Size2D &size,
                    const u32 *srcBase, ptrdiff_t srcStride,
                    u32 * minVal, u32 * maxVal);

    /*
        Fill the arrays `minLocPtr`, `maxLocPtr` with locations of
        given values `minVal`, `maxVal`
    */
    void fillMinMaxLocs(const Size2D & size,
                        const u8 *srcBase, ptrdiff_t srcStride,
                        u8 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                        u8 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity);

    void fillMinMaxLocs(const Size2D & size,
                        const u16 *srcBase, ptrdiff_t srcStride,
                        u16 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                        u16 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity);

    void fillMinMaxLocs(const Size2D & size,
                        const s16 *srcBase, ptrdiff_t srcStride,
                        s16 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                        s16 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity);

    void fillMinMaxLocs(const Size2D & size,
                        const u32 *srcBase, ptrdiff_t srcStride,
                        u32 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                        u32 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity);

    void fillMinMaxLocs(const Size2D & size,
                        const s32 *srcBase, ptrdiff_t srcStride,
                        s32 minVal, size_t * minLocPtr, s32 & minLocCount, s32 minLocCapacity,
                        s32 maxVal, size_t * maxLocPtr, s32 & maxLocCount, s32 maxLocCapacity);

    /*
        Among each pixel `p` within `src` find min and max values and its first occurrences
    */
    void minMaxLoc(const Size2D &size,
                   const s8 * srcBase, ptrdiff_t srcStride,
                   s8 &minVal, size_t &minCol, size_t &minRow,
                   s8 &maxVal, size_t &maxCol, size_t &maxRow);

    void minMaxLoc(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 &minVal, size_t &minCol, size_t &minRow,
                   u8 &maxVal, size_t &maxCol, size_t &maxRow);

    void minMaxLoc(const Size2D &size,
                   const s16 * srcBase, ptrdiff_t srcStride,
                   s16 &minVal, size_t &minCol, size_t &minRow,
                   s16 &maxVal, size_t &maxCol, size_t &maxRow);

    void minMaxLoc(const Size2D &size,
                   const u16 * srcBase, ptrdiff_t srcStride,
                   u16 &minVal, size_t &minCol, size_t &minRow,
                   u16 &maxVal, size_t &maxCol, size_t &maxRow);

    void minMaxLoc(const Size2D &size,
                   const s32 * srcBase, ptrdiff_t srcStride,
                   s32 &minVal, size_t &minCol, size_t &minRow,
                   s32 &maxVal, size_t &maxCol, size_t &maxRow);

    void minMaxLoc(const Size2D &size,
                   const f32 * srcBase, ptrdiff_t srcStride,
                   f32 &minVal, size_t &minCol, size_t &minRow,
                   f32 &maxVal, size_t &maxCol, size_t &maxRow);

    void minMaxLoc(const Size2D &size,
                   const f32 * srcBase, ptrdiff_t srcStride,
                   const u8 * maskBase, ptrdiff_t maskStride,
                   f32 &minVal, size_t &minCol, size_t &minRow,
                   f32 &maxVal, size_t &maxCol, size_t &maxRow);

    /*
        For each point `p` within `size`, do:
        dst[p] += src[p]
    */
    void accumulate(const Size2D &size,
                    const u8 *srcBase, ptrdiff_t srcStride,
                    s16 *dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = (dst[p] + ((src[p] ^ 2) >> shift))
    */
    void accumulateSquare(const Size2D &size,
                          const u8 *srcBase, ptrdiff_t srcStride,
                          s16 *dstBase, ptrdiff_t dstStride,
                          u32 shift);

    /*
        For each point `p` within `size`, do:
        dst[p] = (1 - alpha) * dst[p] + alpha * src[p]
    */
    void accumulateWeighted(const Size2D &size,
                            const u8 *srcBase, ptrdiff_t srcStride,
                            u8 *dstBase, ptrdiff_t dstStride,
                            f32 alpha);

    /*
        orient[p] = atan2(src0[p], src1[p])
    */
    void phase(const Size2D &size,
               const s16 * src0Base, ptrdiff_t src0Stride,
               const s16 * src1Base, ptrdiff_t src1Stride,
               u8 * orientBase, ptrdiff_t orientStride);

    void phase(const Size2D &size,
               const f32 * src0Base, ptrdiff_t src0Stride,
               const f32 * src1Base, ptrdiff_t src1Stride,
               f32 * orientBase, ptrdiff_t orientStride,
               f32 scale);

    /*
        Combine 2 planes to a single one
    */
    void combine2(const Size2D &size,
                  const u8 * src0Base, ptrdiff_t src0Stride,
                  const u8 * src1Base, ptrdiff_t src1Stride,
                  u8 * dstBase, ptrdiff_t dstStride);

    void combine2(const Size2D &size,
                  const u16 * src0Base, ptrdiff_t src0Stride,
                  const u16 * src1Base, ptrdiff_t src1Stride,
                  u16 * dstBase, ptrdiff_t dstStride);

    void combine2(const Size2D &size,
                  const s32 * src0Base, ptrdiff_t src0Stride,
                  const s32 * src1Base, ptrdiff_t src1Stride,
                  s32 * dstBase, ptrdiff_t dstStride);

    void combine2(const Size2D &size,
                  const s64 * src0Base, ptrdiff_t src0Stride,
                  const s64 * src1Base, ptrdiff_t src1Stride,
                  s64 * dstBase, ptrdiff_t dstStride);

    /*
        Combine 3 planes to a single one
    */
    void combine3(const Size2D &size,
                  const u8 * src0Base, ptrdiff_t src0Stride,
                  const u8 * src1Base, ptrdiff_t src1Stride,
                  const u8 * src2Base, ptrdiff_t src2Stride,
                  u8 * dstBase, ptrdiff_t dstStride);

    void combine3(const Size2D &size,
                  const u16 * src0Base, ptrdiff_t src0Stride,
                  const u16 * src1Base, ptrdiff_t src1Stride,
                  const u16 * src2Base, ptrdiff_t src2Stride,
                  u16 * dstBase, ptrdiff_t dstStride);

    void combine3(const Size2D &size,
                  const s32 * src0Base, ptrdiff_t src0Stride,
                  const s32 * src1Base, ptrdiff_t src1Stride,
                  const s32 * src2Base, ptrdiff_t src2Stride,
                  s32 * dstBase, ptrdiff_t dstStride);

    void combine3(const Size2D &size,
                  const s64 * src0Base, ptrdiff_t src0Stride,
                  const s64 * src1Base, ptrdiff_t src1Stride,
                  const s64 * src2Base, ptrdiff_t src2Stride,
                  s64 * dstBase, ptrdiff_t dstStride);

    /*
        Combine 4 planes to a single one
    */
    void combine4(const Size2D &size,
                  const u8 * src0Base, ptrdiff_t src0Stride,
                  const u8 * src1Base, ptrdiff_t src1Stride,
                  const u8 * src2Base, ptrdiff_t src2Stride,
                  const u8 * src3Base, ptrdiff_t src3Stride,
                  u8 * dstBase, ptrdiff_t dstStride);

    void combine4(const Size2D &size,
                  const u16 * src0Base, ptrdiff_t src0Stride,
                  const u16 * src1Base, ptrdiff_t src1Stride,
                  const u16 * src2Base, ptrdiff_t src2Stride,
                  const u16 * src3Base, ptrdiff_t src3Stride,
                  u16 * dstBase, ptrdiff_t dstStride);

    void combine4(const Size2D &size,
                  const s32 * src0Base, ptrdiff_t src0Stride,
                  const s32 * src1Base, ptrdiff_t src1Stride,
                  const s32 * src2Base, ptrdiff_t src2Stride,
                  const s32 * src3Base, ptrdiff_t src3Stride,
                  s32 * dstBase, ptrdiff_t dstStride);

    void combine4(const Size2D &size,
                  const s64 * src0Base, ptrdiff_t src0Stride,
                  const s64 * src1Base, ptrdiff_t src1Stride,
                  const s64 * src2Base, ptrdiff_t src2Stride,
                  const s64 * src3Base, ptrdiff_t src3Stride,
                  s64 * dstBase, ptrdiff_t dstStride);

    /*
        Combine 3 planes to YUYV one
    */
    void combineYUYV(const Size2D &size,
                     const u8 * srcyBase, ptrdiff_t srcyStride,
                     const u8 * srcuBase, ptrdiff_t srcuStride,
                     const u8 * srcvBase, ptrdiff_t srcvStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Combine 3 planes to UYVY one
    */
    void combineUYVY(const Size2D &size,
                     const u8 * srcyBase, ptrdiff_t srcyStride,
                     const u8 * srcuBase, ptrdiff_t srcuStride,
                     const u8 * srcvBase, ptrdiff_t srcvStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to grayscale one
    */
    void rgb2gray(const Size2D &size, COLOR_SPACE color_space,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGBX image to grayscale one
    */
    void rgbx2gray(const Size2D &size, COLOR_SPACE color_space,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert BGR image to grayscale one
    */
    void bgr2gray(const Size2D &size, COLOR_SPACE color_space,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert BGRX image to grayscale one
    */
    void bgrx2gray(const Size2D &size, COLOR_SPACE color_space,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert grayscale image to RGB one
    */
    void gray2rgb(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert grayscale image to RGBX one
    */
    void gray2rgbx(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to RGBX
    */
    void rgb2rgbx(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGBX image to RGB
    */
    void rgbx2rgb(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to BGR
    */
    void rgb2bgr(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGBX image to BGRX
    */
    void rgbx2bgrx(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGBX image to BGR
    */
    void rgbx2bgr(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to BGRX
    */
    void rgb2bgrx(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to HSV
    */
    void rgb2hsv(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 s32 hrange);

    /*
        Convert RGBX image to HSV
    */
    void rgbx2hsv(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  s32 hrange);

    /*
        Convert BGR image to HSV
    */
    void bgr2hsv(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 s32 hrange);

    /*
        Convert BGRX image to HSV
    */
    void bgrx2hsv(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  s32 hrange);

    /*
        Convert RGBX image to BGR565
        RRRRrrrr GGGGgggg BBBBbbbb XXXXxxxx -> GggBBBBb RRRRrGGG
    */
    void rgbx2bgr565(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to BGR565
        RRRRrrrr GGGGgggg BBBBbbbb -> GggBBBBb RRRRrGGG
    */
    void rgb2bgr565(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGBX image to RGB565
        RRRRrrrr GGGGgggg BBBBbbbb XXXXxxxx -> GggRRRRr BBBBbGGG
    */
    void rgbx2rgb565(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to RGB565
        RRRRrrrr GGGGgggg BBBBbbbb -> GggRRRRr BBBBbGGG
    */
    void rgb2rgb565(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGB image to YCrCb
    */
    void rgb2ycrcb(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert RGBX image to YCrCb
    */
    void rgbx2ycrcb(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert BGR image to YCrCb
    */
    void bgr2ycrcb(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert BGRX image to YCrCb
    */
    void bgrx2ycrcb(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420sp image to RGB
    */
    void yuv420sp2rgb(const Size2D &size,
                      const u8 *  yBase, ptrdiff_t  yStride,
                      const u8 * uvBase, ptrdiff_t uvStride,
                      u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420sp image to RGBX
    */
    void yuv420sp2rgbx(const Size2D &size,
                       const u8 *  yBase, ptrdiff_t  yStride,
                       const u8 * uvBase, ptrdiff_t uvStride,
                       u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420i image to RGB
    */
    void yuv420i2rgb(const Size2D &size,
                     const u8 *  yBase, ptrdiff_t  yStride,
                     const u8 * uvBase, ptrdiff_t uvStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420i image to RGBX
    */
    void yuv420i2rgbx(const Size2D &size,
                      const u8 *  yBase, ptrdiff_t  yStride,
                      const u8 * uvBase, ptrdiff_t uvStride,
                      u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420sp image to BGR
    */
    void yuv420sp2bgr(const Size2D &size,
                      const u8 *  yBase, ptrdiff_t  yStride,
                      const u8 * uvBase, ptrdiff_t uvStride,
                      u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420sp image to BGRX
    */
    void yuv420sp2bgrx(const Size2D &size,
                       const u8 *  yBase, ptrdiff_t  yStride,
                       const u8 * uvBase, ptrdiff_t uvStride,
                       u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420i image to BGR
    */
    void yuv420i2bgr(const Size2D &size,
                     const u8 *  yBase, ptrdiff_t  yStride,
                     const u8 * uvBase, ptrdiff_t uvStride,
                     u8 * dstBase, ptrdiff_t dstStride);

    /*
        Convert YUV420i image to BGRX
    */
    void yuv420i2bgrx(const Size2D &size,
                      const u8 *  yBase, ptrdiff_t  yStride,
                      const u8 * uvBase, ptrdiff_t uvStride,
                      u8 * dstBase, ptrdiff_t dstStride);

    /*
        For each point `p` within `size`, do:
        dst[p] = src[p] << shift
    */
    void lshift(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                s16 * dstBase, ptrdiff_t dstStride,
                u32 shift);

    /*
        For each point `p` within `size`, do sign-extending shift:
        dst[p] = src[p] >> shift
    */
    void rshift(const Size2D &size,
                const s16 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                u32 shift, CONVERT_POLICY cpolicy);

    /*
        For each point `p` within `size`, set `dst[p]` to the average
        of `src[p]` and the 8 (or 24 for blur5x5) points around it
        NOTE: the function cannot operate inplace
    */
    bool isBlur3x3Supported(const Size2D &size, BORDER_MODE border);
    void blur3x3(const Size2D &size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE border, u8 borderValue);

    bool isBlurU8Supported(const Size2D &size, s32 cn, BORDER_MODE border);
    void blur3x3(const Size2D &size, s32 cn,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE borderType, u8 borderValue);

    void blur5x5(const Size2D &size, s32 cn,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE borderType, u8 borderValue);

    /*
        For each point `p` within `size`, set `dst[p]` to the average
        of `src[p]` and the 8 points around it
        NOTE: the function can operate inplace
    */
    bool isBlurF32Supported(const Size2D &size, s32 cn, BORDER_MODE border);
    void blur3x3(const Size2D &size, s32 cn,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 f32 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE borderType, f32 borderValue, Margin borderMargin);

    bool isBlurS32Supported(const Size2D &size, s32 cn, BORDER_MODE border);
    void blur3x3(const Size2D &size, s32 cn,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 s32 * dstBase, ptrdiff_t dstStride,
                 BORDER_MODE borderType, s32 borderValue, Margin borderMargin);

    /*
        For each point `p` within `size`, set `dst[p]` to gaussian smooth
        of `src[p]` and the 8(24 for 5x5 version) points around it
        NOTE: the function cannot operate inplace
    */
    bool isGaussianBlur3x3Supported(const Size2D &size, BORDER_MODE border);
    void gaussianBlur3x3(const Size2D &size,
                         const u8 * srcBase, ptrdiff_t srcStride,
                         u8 * dstBase, ptrdiff_t dstStride,
                         BORDER_MODE border, u8 borderValue);
    bool isGaussianBlur3x3MarginSupported(const Size2D &size, BORDER_MODE border, Margin borderMargin = Margin());
    void gaussianBlur3x3Margin(const Size2D &size,
                               const u8 * srcBase, ptrdiff_t srcStride,
                               u8 * dstBase, ptrdiff_t dstStride,
                               BORDER_MODE border, u8 borderValue, Margin borderMargin = Margin());

    bool isGaussianBlur5x5Supported(const Size2D &size, s32 cn, BORDER_MODE border);
    void gaussianBlur5x5(const Size2D &size, s32 cn,
                         const u8 * srcBase, ptrdiff_t srcStride,
                         u8 * dstBase, ptrdiff_t dstStride,
                         BORDER_MODE borderType, u8 borderValue, Margin borderMargin);

    void gaussianBlur5x5(const Size2D &size, s32 cn,
                         const u16 * srcBase, ptrdiff_t srcStride,
                         u16 * dstBase, ptrdiff_t dstStride,
                         BORDER_MODE borderType, u16 borderValue, Margin borderMargin);

    void gaussianBlur5x5(const Size2D &size, s32 cn,
                         const s16 * srcBase, ptrdiff_t srcStride,
                         s16 * dstBase, ptrdiff_t dstStride,
                         BORDER_MODE borderType, s16 borderValue, Margin borderMargin);

    void gaussianBlur5x5(const Size2D &size, s32 cn,
                         const s32 * srcBase, ptrdiff_t srcStride,
                         s32 * dstBase, ptrdiff_t dstStride,
                         BORDER_MODE borderType, s32 borderValue, Margin borderMargin);

    /*
        Calculation of Sobel operator
        NOTE: the function cannot operate inplace
    */
    bool isSobel3x3Supported(const Size2D &size, BORDER_MODE border, s32 dx, s32 dy, Margin borderMargin = Margin());
    void Sobel3x3(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  s16 * dstBase, ptrdiff_t dstStride,
                  s32 dx, s32 dy,
                  BORDER_MODE border, u8 borderValue, Margin borderMargin = Margin());

    /*
        Calculation of Sobel operator for f32 data
        NOTE: the function can operate inplace
    */
    bool isSobel3x3f32Supported(const Size2D &size, BORDER_MODE border, s32 dx, s32 dy);
    void Sobel3x3(const Size2D &size,
                  const f32 * srcBase, ptrdiff_t srcStride,
                  f32 * dstBase, ptrdiff_t dstStride,
                  s32 dx, s32 dy,
                  BORDER_MODE borderType, f32 borderValue);

    /*
        Calculation of Scharr operator
        NOTE: the function cannot operate inplace
    */
    bool isScharr3x3Supported(const Size2D &size, BORDER_MODE border, s32 dx, s32 dy, Margin borderMargin = Margin());
    void Scharr3x3(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   s16 * dstBase, ptrdiff_t dstStride,
                   s32 dx, s32 dy,
                   BORDER_MODE borderType, u8 borderValue, Margin borderMargin = Margin());

    void ScharrDeriv(const Size2D &size, s32 cn,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     s16 * dstBase, ptrdiff_t dstStride);

    /*
        Calculation of generic separable filtering operator
        rowFilter/colFilter define filter weights
        0 - predefined  1  2  1
        1 - predefined -1  0  1
        2 - predefined  1 -2  1
        3 - weights provided as xw/yw
    */
    bool isSeparableFilter3x3Supported(const Size2D &size, BORDER_MODE border, s32 dx, s32 dy, Margin borderMargin = Margin());
    void SeparableFilter3x3(const Size2D &size,
                            const u8 * srcBase, ptrdiff_t srcStride,
                            s16 * dstBase, ptrdiff_t dstStride,
                            const u8 rowFilter, const u8 colFilter, const s16 *xw, const s16 *yw,
                            BORDER_MODE border, u8 borderValue, Margin borderMargin = Margin());

    /*
        Extract a single plane from 2 channel image
    */
    void extract2(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  u32 coi);

    /*
        Extract a single plane from 3 channel image
    */
    void extract3(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  u32 coi);

    /*
        Extract a single plane from 4 channel image
    */
    void extract4(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  u32 coi);

    /*
        Split 2 channel image to separate planes
    */
    void split2(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dst0Base, ptrdiff_t dst0Stride,
                u8 * dst1Base, ptrdiff_t dst1Stride);

    void split2(const Size2D &size,
                const u16* srcBase, ptrdiff_t srcStride,
                u16 * dst0Base, ptrdiff_t dst0Stride,
                u16 * dst1Base, ptrdiff_t dst1Stride);

    void split2(const Size2D &size,
                const s32 * srcBase, ptrdiff_t srcStride,
                s32 * dst0Base, ptrdiff_t dst0Stride,
                s32 * dst1Base, ptrdiff_t dst1Stride);

    void split2(const Size2D &size,
                const s64 * srcBase, ptrdiff_t srcStride,
                s64 * dst0Base, ptrdiff_t dst0Stride,
                s64 * dst1Base, ptrdiff_t dst1Stride);

    /*
        Split 3 channel image to separate planes
    */
    void split3(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dst0Base, ptrdiff_t dst0Stride,
                u8 * dst1Base, ptrdiff_t dst1Stride,
                u8 * dst2Base, ptrdiff_t dst2Stride);

    void split3(const Size2D &size,
                const u16* srcBase, ptrdiff_t srcStride,
                u16 * dst0Base, ptrdiff_t dst0Stride,
                u16 * dst1Base, ptrdiff_t dst1Stride,
                u16 * dst2Base, ptrdiff_t dst2Stride);

    void split3(const Size2D &size,
                const s32 * srcBase, ptrdiff_t srcStride,
                s32 * dst0Base, ptrdiff_t dst0Stride,
                s32 * dst1Base, ptrdiff_t dst1Stride,
                s32 * dst2Base, ptrdiff_t dst2Stride);

    void split3(const Size2D &size,
                const s64 * srcBase, ptrdiff_t srcStride,
                s64 * dst0Base, ptrdiff_t dst0Stride,
                s64 * dst1Base, ptrdiff_t dst1Stride,
                s64 * dst2Base, ptrdiff_t dst2Stride);

    /*
        Split 4 channel image to separate planes
    */
    void split4(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dst0Base, ptrdiff_t dst0Stride,
                u8 * dst1Base, ptrdiff_t dst1Stride,
                u8 * dst2Base, ptrdiff_t dst2Stride,
                u8 * dst3Base, ptrdiff_t dst3Stride);

    void split4(const Size2D &size,
                const u16* srcBase, ptrdiff_t srcStride,
                u16 * dst0Base, ptrdiff_t dst0Stride,
                u16 * dst1Base, ptrdiff_t dst1Stride,
                u16 * dst2Base, ptrdiff_t dst2Stride,
                u16 * dst3Base, ptrdiff_t dst3Stride);

    void split4(const Size2D &size,
                const s32 * srcBase, ptrdiff_t srcStride,
                s32 * dst0Base, ptrdiff_t dst0Stride,
                s32 * dst1Base, ptrdiff_t dst1Stride,
                s32 * dst2Base, ptrdiff_t dst2Stride,
                s32 * dst3Base, ptrdiff_t dst3Stride);

    void split4(const Size2D &size,
                const s64 * srcBase, ptrdiff_t srcStride,
                s64 * dst0Base, ptrdiff_t dst0Stride,
                s64 * dst1Base, ptrdiff_t dst1Stride,
                s64 * dst2Base, ptrdiff_t dst2Stride,
                s64 * dst3Base, ptrdiff_t dst3Stride);

    /*
        Split 4 channel image to 3 channel image and 1 channel image
    */
    void split4(const Size2D &size,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dst3Base, ptrdiff_t dst3Stride,
                u8 * dst1Base, ptrdiff_t dst1Stride);

    /*
        Flip image using specified flip mode
    */
    bool isFlipSupported(FLIP_MODE flipMode, u32 elemSize);
    void flip(const Size2D &size,
              const u8 * srcBase, ptrdiff_t srcStride,
              u8 * dstBase, ptrdiff_t dstStride,
              FLIP_MODE flipMode, u32 elemSize);

    /*
        For each point `p` within `size`, set `dst[p]` to the maximum
        of `src[p]` and the 8 points around it
        NOTE: the function cannot operate inplace
    */
    bool isMorph3x3Supported(const Size2D &size, BORDER_MODE border);

    void erode3x3(const Size2D &size,
                  const u8 * srcBase, ptrdiff_t srcStride,
                  u8 * dstBase, ptrdiff_t dstStride,
                  BORDER_MODE border, u8 borderValue);

    void dilate3x3(const Size2D &size,
                   const u8 * srcBase, ptrdiff_t srcStride,
                   u8 * dstBase, ptrdiff_t dstStride,
                   BORDER_MODE border, u8 borderValue);

    void erode(const Size2D &ssize, u32 cn,
               const u8 * srcBase, ptrdiff_t srcStride,
               u8 * dstBase, ptrdiff_t dstStride,
               const Size2D &ksize,
               size_t anchorX, size_t anchorY,
               BORDER_MODE rowBorderType, BORDER_MODE columnBorderType,
               const u8 * borderValues, Margin borderMargin);

    void dilate(const Size2D &ssize, u32 cn,
                const u8 * srcBase, ptrdiff_t srcStride,
                u8 * dstBase, ptrdiff_t dstStride,
                const Size2D &ksize,
                size_t anchorX, size_t anchorY,
                BORDER_MODE rowBorderType, BORDER_MODE columnBorderType,
                const u8 * borderValues, Margin borderMargin);

    /*
        Resize a source image using "nearest neighbor" interpolation type

        wr = src_width / dst_width
        hr = src_height / dst_height
    */
    bool isResizeNearestNeighborSupported(const Size2D &ssize, u32 elemSize);
    void resizeNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                               const void * srcBase, ptrdiff_t srcStride,
                               void * dstBase, ptrdiff_t dstStride,
                               f32 wr, f32 hr, u32 elemSize);

    /*
        Resize a source image using "area" interpolation type

        wr = src_width / dst_width
        hr = src_height / dst_height
    */
    bool isResizeAreaSupported(f32 wr, f32 hr, u32 channels);
    void resizeAreaOpenCV(const Size2D &ssize, const Size2D &dsize,
                          const u8 * srcBase, ptrdiff_t srcStride,
                          u8 * dstBase, ptrdiff_t dstStride,
                          f32 wr, f32 hr, u32 channels);
    void resizeArea(const Size2D &ssize, const Size2D &dsize,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    f32 wr, f32 hr, u32 channels);

    /*
        Resize a source image using "linear" interpolation type

        wr = src_width / dst_width
        hr = src_height / dst_height
    */
    bool isResizeLinearOpenCVSupported(const Size2D &ssize, const Size2D &dsize, u32 channels);
    bool isResizeLinearSupported(const Size2D &ssize, const Size2D &dsize,
                                 f32 wr, f32 hr, u32 channels);
    void resizeLinearOpenCV(const Size2D &ssize, const Size2D &dsize,
                            const u8 * srcBase, ptrdiff_t srcStride,
                            u8 * dstBase, ptrdiff_t dstStride,
                            f32 wr, f32 hr, u32 channels);
    void resizeLinear(const Size2D &ssize, const Size2D &dsize,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f32 wr, f32 hr, u32 channels);

    /*
        For each point `p` within `size`, set `dst[p]` to convolution
        of `src[p]` and the (ksize * ksize - 1) points around it
        The function uses OpenVX semantic (so, in order to use this function
        in OpenCV you should flip kernel in both directions)
        NOTE: the function cannot operate inplace
    */
    bool isConvolutionSupported(const Size2D &size, const Size2D &ksize, BORDER_MODE border);
    void convolution(const Size2D &size,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE border, u8 borderValue,
                     const Size2D & ksize, s16 * kernelBase, u32 scale);

    /*
        For each point `p` within `dstSize`, does convolution
        of tmpl points and size*size square of src points starting with `src[p]`.
        Src should be of size (dstSize+size-1)*(dstSize+size-1)
        NOTE: the function cannot operate inplace
    */
    bool isMatchTemplateSupported(const Size2D &tmplSize);
    void matchTemplate(const Size2D &srcSize,
                       const u8 * srcBase, ptrdiff_t srcStride,
                       const Size2D &tmplSize,
                       const u8 * tmplBase, ptrdiff_t tmplStride,
                       f32 * dstBase, ptrdiff_t dstStride,
                       bool normalize);

    /*
        Calculation of Laplacian operator

        1  1  1
        1 -8  1
        1  1  1

        NOTE: the function cannot operate inplace
    */
    bool isLaplacian3x3Supported(const Size2D &size, BORDER_MODE border);
    void Laplacian3x3(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      BORDER_MODE border, u8 borderValue);

    /*
        OpenCV like calculation of Laplacian operator

        kernel 1    kernel 3    kernel 5
        0  1  0     2  0  2     1   2   2   2   1
        1 -4  1     0 -8  0     2   0  -4   0   2
        0  1  0     2  0  2     2  -4 -12  -4   2
                                2   0  -4   0   2
                                1   2   2   2   1

        NOTE: the function cannot operate inplace
    */
    bool isLaplacianOpenCVSupported(const Size2D &size, BORDER_MODE border);
    void Laplacian1OpenCV(const Size2D &size,
                          const u8 * srcBase, ptrdiff_t srcStride,
                          s16 * dstBase, ptrdiff_t dstStride,
                          BORDER_MODE border, u8 borderValue);
    void Laplacian3OpenCV(const Size2D &size,
                          const u8 * srcBase, ptrdiff_t srcStride,
                          s16 * dstBase, ptrdiff_t dstStride,
                          BORDER_MODE border, u8 borderValue);
    void Laplacian5OpenCV(const Size2D &size,
                          const u8 * srcBase, ptrdiff_t srcStride,
                          s16 * dstBase, ptrdiff_t dstStride,
                          BORDER_MODE border, u8 borderValue);

    /*
        Detect image edges using Canny algorithm
        These functions perform derivatives estimation using sobel algorithm
    */
    bool isCanny3x3Supported(const Size2D &size);
    void Canny3x3L1(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    f64 low_thresh, f64 high_thresh,
                    Margin borderMargin);

    void Canny3x3L2(const Size2D &size,
                    const u8 * srcBase, ptrdiff_t srcStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    f64 low_thresh, f64 high_thresh,
                    Margin borderMargin);

    /*
        Detect image edges using Canny algorithm
        These functions don't estimate derivatives and thus require
        precomputed derivatives estimation instead of source image
    */
    void Canny3x3L1(const Size2D &size, s32 cn,
                    s16 * dxBase, ptrdiff_t dxStride,
                    s16 * dyBase, ptrdiff_t dyStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    f64 low_thresh, f64 high_thresh);

    void Canny3x3L2(const Size2D &size, s32 cn,
                    s16 * dxBase, ptrdiff_t dxStride,
                    s16 * dyBase, ptrdiff_t dyStride,
                    u8 * dstBase, ptrdiff_t dstStride,
                    f64 low_thresh, f64 high_thresh);

    /*
        Performs detection of FAST features
    */
    void FAST(const Size2D &size,
              u8 *srcBase, ptrdiff_t srcStride,
              KeypointStore *keypoints,
              u8 threshold, bool nonmax_suppression);

    /*
        Remap a source image using table and specified
        extrapolation method
    */
    bool isRemapNearestNeighborSupported(const Size2D &ssize);
    void remapNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                              const u8 * srcBase, ptrdiff_t srcStride,
                              const f32 * tableBase, ptrdiff_t tableStride,
                              u8 * dstBase, ptrdiff_t dstStride,
                              BORDER_MODE borderMode, u8 borderValue);

    bool isRemapLinearSupported(const Size2D &ssize);
    void remapLinear(const Size2D &ssize, const Size2D &dsize,
                     const u8 * srcBase, ptrdiff_t srcStride,
                     const f32 * tableBase, ptrdiff_t tableStride,
                     u8 * dstBase, ptrdiff_t dstStride,
                     BORDER_MODE borderMode, u8 borderValue);

    /*
        Perform an affine transform on an input image

        src_x = dst_x * m[0] + dst_y * m[2] + m[4]
        src_y = dst_x * m[1] + dst_y * m[3] + m[5]
    */
    bool isWarpAffineNearestNeighborSupported(const Size2D &ssize);
    void warpAffineNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                                   const u8 * srcBase, ptrdiff_t srcStride,
                                   const f32 * m,
                                   u8 * dstBase, ptrdiff_t dstStride,
                                   BORDER_MODE borderMode, u8 borderValue);

    bool isWarpAffineLinearSupported(const Size2D &ssize);
    void warpAffineLinear(const Size2D &ssize, const Size2D &dsize,
                          const u8 * srcBase, ptrdiff_t srcStride,
                          const f32 * m,
                          u8 * dstBase, ptrdiff_t dstStride,
                          BORDER_MODE borderMode, u8 borderValue);

    /*
        Perform a perspective transform on an input image

        src_x = dst_x * m[0] + dst_y * m[3] + m[6]
        src_y = dst_x * m[1] + dst_y * m[4] + m[7]
        w     = dst_x * m[2] + dst_y * m[5] + m[8]

        src_x = w == 0 ? 0 : src_x / w
        src_y = w == 0 ? 0 : src_y / w
    */
    bool isWarpPerspectiveNearestNeighborSupported(const Size2D &ssize);
    void warpPerspectiveNearestNeighbor(const Size2D &ssize, const Size2D &dsize,
                                        const u8 * srcBase, ptrdiff_t srcStride,
                                        const f32 * m,
                                        u8 * dstBase, ptrdiff_t dstStride,
                                        BORDER_MODE borderMode, u8 borderValue);

    bool isWarpPerspectiveLinearSupported(const Size2D &ssize);
    void warpPerspectiveLinear(const Size2D &ssize, const Size2D &dsize,
                               const u8 * srcBase, ptrdiff_t srcStride,
                               const f32 * m,
                               u8 * dstBase, ptrdiff_t dstStride,
                               BORDER_MODE borderMode, u8 borderValue);

    /*
        Convert data from source to destination type
    */
    void convert(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 s8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 u16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 s32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 f32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s8 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s8 * srcBase, ptrdiff_t srcStride,
                 u16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s8 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s8 * srcBase, ptrdiff_t srcStride,
                 s32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s8 * srcBase, ptrdiff_t srcStride,
                 f32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride,
                 s8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride,
                 s32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride,
                 f32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 s8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 u16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 s32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 f32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 s8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 u16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 f32 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 s8 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 u16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 s16 * dstBase, ptrdiff_t dstStride);

    void convert(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 s32 * dstBase, ptrdiff_t dstStride);

    /*
        Convert data from source to destination type with scaling
        dst = saturate_cast<dst_type>(src * alpha + beta)
    */
    void convertScale(const Size2D &_size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      u16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s8 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s8 * srcBase, ptrdiff_t srcStride,
                      s8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s8 * srcBase, ptrdiff_t srcStride,
                      u16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s8 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s8 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s8 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u16 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u16 * srcBase, ptrdiff_t srcStride,
                      s8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u16 * srcBase, ptrdiff_t srcStride,
                      u16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u16 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u16 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const u16 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s16 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s16 * srcBase, ptrdiff_t srcStride,
                      s8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s16 * srcBase, ptrdiff_t srcStride,
                      u16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s16 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s16 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s16 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s32 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s32 * srcBase, ptrdiff_t srcStride,
                      s8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s32 * srcBase, ptrdiff_t srcStride,
                      u16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s32 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s32 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const s32 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      s8 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      u16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      s16 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    void convertScale(const Size2D &_size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase, ptrdiff_t dstStride,
                      f64 alpha, f64 beta);

    /*
        Reduce matrix to a vector by calculatin given operation for each column
    */
    void reduceColSum(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      s32 * dstBase);

    void reduceColMax(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase);

    void reduceColMin(const Size2D &size,
                      const u8 * srcBase, ptrdiff_t srcStride,
                      u8 * dstBase);

    void reduceColSum(const Size2D &size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase);

    void reduceColMax(const Size2D &size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase);

    void reduceColMin(const Size2D &size,
                      const f32 * srcBase, ptrdiff_t srcStride,
                      f32 * dstBase);

    /*
        For each point `p` within `size`, do:
        dst[p] = (rng1[p] <= src[p] && src[p] <= rng2[p]) ? 255 : 0
    */

    void inRange(const Size2D &_size,
                 const u8 * srcBase, ptrdiff_t srcStride,
                 const u8 * rng1Base, ptrdiff_t rng1Stride,
                 const u8 * rng2Base, ptrdiff_t rng2Stride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void inRange(const Size2D &_size,
                 const s8 * srcBase, ptrdiff_t srcStride,
                 const s8 * rng1Base, ptrdiff_t rng1Stride,
                 const s8 * rng2Base, ptrdiff_t rng2Stride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void inRange(const Size2D &_size,
                 const u16 * srcBase, ptrdiff_t srcStride,
                 const u16 * rng1Base, ptrdiff_t rng1Stride,
                 const u16 * rng2Base, ptrdiff_t rng2Stride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void inRange(const Size2D &_size,
                 const s16 * srcBase, ptrdiff_t srcStride,
                 const s16 * rng1Base, ptrdiff_t rng1Stride,
                 const s16 * rng2Base, ptrdiff_t rng2Stride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void inRange(const Size2D &_size,
                 const s32 * srcBase, ptrdiff_t srcStride,
                 const s32 * rng1Base, ptrdiff_t rng1Stride,
                 const s32 * rng2Base, ptrdiff_t rng2Stride,
                 u8 * dstBase, ptrdiff_t dstStride);

    void inRange(const Size2D &_size,
                 const f32 * srcBase, ptrdiff_t srcStride,
                 const f32 * rng1Base, ptrdiff_t rng1Stride,
                 const f32 * rng2Base, ptrdiff_t rng2Stride,
                 u8 * dstBase, ptrdiff_t dstStride);

    /*
        Estimate amount of non zero elements
    */
    s32 countNonZero(const Size2D &_size,
                     const u8 * srcBase, ptrdiff_t srcStride);

    s32 countNonZero(const Size2D &_size,
                     const u16 * srcBase, ptrdiff_t srcStride);

    s32 countNonZero(const Size2D &_size,
                     const s32 * srcBase, ptrdiff_t srcStride);

    s32 countNonZero(const Size2D &_size,
                     const f32 * srcBase, ptrdiff_t srcStride);

    s32 countNonZero(const Size2D &_size,
                     const f64 * srcBase, ptrdiff_t srcStride);

    /*
        Calculates sum of all image pixel values and squared values
    */
    bool isSumSupported(u32 channels);

    void sum(const Size2D &_size,
             const u8 * srcBase, ptrdiff_t srcStride,
             u32 * sumdst, u32 channels);

    void sum(const Size2D &_size,
             const f32 * srcBase, ptrdiff_t srcStride,
             f64 * sumdst, u32 channels);

    bool isSqsumSupported(u32 channels);

    void sqsum(const Size2D &_size,
               const u8 * srcBase, ptrdiff_t srcStride,
               f64 * sumdst, f64 * sqsumdst, u32 channels);

    /*
        Calculates norm
    */
    s32 normInf(const Size2D &_size,
                const u8 * srcBase, ptrdiff_t srcStride);

    s32 normInf(const Size2D &_size,
                const s8 * srcBase, ptrdiff_t srcStride);

    s32 normInf(const Size2D &_size,
                const u16 * srcBase, ptrdiff_t srcStride);

    s32 normInf(const Size2D &_size,
                const s16 * srcBase, ptrdiff_t srcStride);

    s32 normInf(const Size2D &_size,
                const s32 * srcBase, ptrdiff_t srcStride);

    f32 normInf(const Size2D &_size,
                const f32 * srcBase, ptrdiff_t srcStride);

    s32 normL1(const Size2D &_size,
               const u8 * srcBase, ptrdiff_t srcStride);

    s32 normL1(const Size2D &_size,
               const s8 * srcBase, ptrdiff_t srcStride);

    s32 normL1(const Size2D &_size,
               const u16 * srcBase, ptrdiff_t srcStride);

    s32 normL1(const Size2D &_size,
               const s16 * srcBase, ptrdiff_t srcStride);

    f64 normL1(const Size2D &_size,
               const s32 * srcBase, ptrdiff_t srcStride);

    f64 normL1(const Size2D &_size,
               const f32 * srcBase, ptrdiff_t srcStride);

    s32 normL2(const Size2D &_size,
               const u8 * srcBase, ptrdiff_t srcStride);

    s32 normL2(const Size2D &_size,
               const s8 * srcBase, ptrdiff_t srcStride);

    f64 normL2(const Size2D &_size,
               const u16 * srcBase, ptrdiff_t srcStride);

    f64 normL2(const Size2D &_size,
               const s16 * srcBase, ptrdiff_t srcStride);

    f64 normL2(const Size2D &_size,
               const s32 * srcBase, ptrdiff_t srcStride);

    f64 normL2(const Size2D &_size,
               const f32 * srcBase, ptrdiff_t srcStride);

    /*
        Calculates norm of per element difference
    */
    s32 diffNormInf(const Size2D &_size,
                    const u8 * src0Base, ptrdiff_t src0Stride,
                    const u8 * src1Base, ptrdiff_t src1Stride);

    f32 diffNormInf(const Size2D &_size,
                    const f32 * src0Base, ptrdiff_t src0Stride,
                    const f32 * src1Base, ptrdiff_t src1Stride);

    s32 diffNormL1(const Size2D &_size,
                   const u8 * src0Base, ptrdiff_t src0Stride,
                   const u8 * src1Base, ptrdiff_t src1Stride);

    f64 diffNormL1(const Size2D &_size,
                   const f32 * src0Base, ptrdiff_t src0Stride,
                   const f32 * src1Base, ptrdiff_t src1Stride);

    s32 diffNormL2(const Size2D &_size,
                   const u8 * src0Base, ptrdiff_t src0Stride,
                   const u8 * src1Base, ptrdiff_t src1Stride);

    f64 diffNormL2(const Size2D &_size,
                   const f32 * src0Base, ptrdiff_t src0Stride,
                   const f32 * src1Base, ptrdiff_t src1Stride);

    /*
     *        Pyramidal Lucas-Kanade Optical Flow level processing
     */
    void pyrLKOptFlowLevel(const Size2D &size, s32 cn,
                           const u8 *prevData, ptrdiff_t prevStride,
                           const s16 *prevDerivData, ptrdiff_t prevDerivStride,
                           const u8 *nextData, ptrdiff_t nextStride,
                           u32 ptCount,
                           const f32 *prevPts, f32 *nextPts,
                           u8 *status, f32 *err,
                           const Size2D &winSize,
                           u32 terminationCount, f64 terminationEpsilon,
                           bool getMinEigenVals,
                           f32 minEigThreshold);
}

#endif
