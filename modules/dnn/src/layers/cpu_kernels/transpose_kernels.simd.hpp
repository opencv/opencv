// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include <cstdint>
#include <algorithm>

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// Blocked 2D float transpose.
void transpose2D_f32_(const float* inp, float* out,
                      int64_t outer, int64_t rows, int64_t cols);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#if (CV_SIMD || CV_SIMD_SCALABLE)
// AVX2 8x8 f32 in-register transpose. v_transpose4x4 transposes the two
// 128-bit halves independently; v_combine_low/high then exchange halves so
// element k of each target row comes from the right source row.
static inline void transpose_8x8_f32_avx2_(const float* src, int64_t src_stride,
                                           float* dst, int64_t dst_stride)
{
    v_float32 r0 = vx_load(src + 0 * src_stride);
    v_float32 r1 = vx_load(src + 1 * src_stride);
    v_float32 r2 = vx_load(src + 2 * src_stride);
    v_float32 r3 = vx_load(src + 3 * src_stride);
    v_float32 r4 = vx_load(src + 4 * src_stride);
    v_float32 r5 = vx_load(src + 5 * src_stride);
    v_float32 r6 = vx_load(src + 6 * src_stride);
    v_float32 r7 = vx_load(src + 7 * src_stride);

    v_float32 a0, a1, a2, a3, a4, a5, a6, a7;
    v_transpose4x4(r0, r1, r2, r3, a0, a1, a2, a3);
    v_transpose4x4(r4, r5, r6, r7, a4, a5, a6, a7);

    v_float32 b0 = v_combine_low (a0, a4);
    v_float32 b1 = v_combine_low (a1, a5);
    v_float32 b2 = v_combine_low (a2, a6);
    v_float32 b3 = v_combine_low (a3, a7);
    v_float32 b4 = v_combine_high(a0, a4);
    v_float32 b5 = v_combine_high(a1, a5);
    v_float32 b6 = v_combine_high(a2, a6);
    v_float32 b7 = v_combine_high(a3, a7);

    v_store(dst + 0 * dst_stride, b0);
    v_store(dst + 1 * dst_stride, b1);
    v_store(dst + 2 * dst_stride, b2);
    v_store(dst + 3 * dst_stride, b3);
    v_store(dst + 4 * dst_stride, b4);
    v_store(dst + 5 * dst_stride, b5);
    v_store(dst + 6 * dst_stride, b6);
    v_store(dst + 7 * dst_stride, b7);
}
#endif

void transpose2D_f32_(const float* inp, float* out,
                      int64_t outer, int64_t rows, int64_t cols)
{
    const int TILE = 32;
    int64_t batchStride = rows * cols;

    int64_t rtiles = (rows + TILE - 1) / TILE;
    int64_t ctiles = (cols + TILE - 1) / TILE;
    int64_t total = outer * rtiles * ctiles;

    parallel_for_(Range(0, (int)total), [&](const Range& range) {
        for (int64_t idx = range.start; idx < range.end; idx++) {
            int64_t rem = idx;
            int64_t b = rem / (rtiles * ctiles);
            rem -= b * rtiles * ctiles;
            int64_t ti = rem / ctiles;
            int64_t tj = rem - ti * ctiles;
            int64_t r0 = ti * TILE, r1 = std::min(r0 + TILE, rows);
            int64_t c0 = tj * TILE, c1 = std::min(c0 + TILE, cols);

            const float* inB  = inp + b * batchStride;
            float* outB = out + b * batchStride;

#if (CV_SIMD || CV_SIMD_SCALABLE)
            if (VTraits<v_float32>::vlanes() == 8) {
                int64_t r = r0;
                for (; r + 8 <= r1; r += 8) {
                    int64_t c = c0;
                    for (; c + 8 <= c1; c += 8) {
                        transpose_8x8_f32_avx2_(inB + c * rows + r, rows,
                                                outB + r * cols + c, cols);
                    }
                    for (; c < c1; c++) {
                        for (int k = 0; k < 8; k++)
                            outB[(r + k) * cols + c] = inB[c * rows + (r + k)];
                    }
                }
                for (; r < r1; r++) {
                    for (int64_t c = c0; c < c1; c++)
                        outB[r * cols + c] = inB[c * rows + r];
                }
                continue;
            }
#endif
            for (int64_t r = r0; r < r1; r++) {
                for (int64_t c = c0; c < c1; c++)
                    outB[r * cols + c] = inB[c * rows + r];
            }
        }
    });
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
