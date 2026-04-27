// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include <opencv2/core.hpp>
#include "opencv2/core/hal/intrin.hpp"
#include <cstddef>

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// SIMD-over-channels bilinear kernel for f32 GridSample, INTERIOR positions only.
void gridSampleBilinearF32InteriorRow_(
    const float* baseN,
    float* outBase,
    const int* px, const int* py,
    const float* dx, const float* dy,
    const unsigned char* interior,
    int C, int Wout,
    size_t xCStride, size_t xHStride,
    size_t yCStride);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void gridSampleBilinearF32InteriorRow_(
    const float* baseN,
    float* outBase,
    const int* px, const int* py,
    const float* dx, const float* dy,
    const unsigned char* interior,
    int C, int Wout,
    size_t xCStride, size_t xHStride,
    size_t yCStride)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int L = VTraits<v_float32>::vlanes();
#endif
    for (int w = 0; w < Wout; w++) {
        if (!interior[w]) continue;
        const int x0 = px[w], y0 = py[w];
        const float fx = dx[w], fy = dy[w];
        const float w00 = (1.f - fx) * (1.f - fy);
        const float w01 = fx * (1.f - fy);
        const float w10 = (1.f - fx) * fy;
        const float w11 = fx * fy;
        const size_t off00 = (size_t)y0 * xHStride + x0;
        const size_t off01 = off00 + 1;
        const size_t off10 = off00 + xHStride;
        const size_t off11 = off10 + 1;

        int c = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
        v_float32 vw00 = vx_setall_f32(w00);
        v_float32 vw01 = vx_setall_f32(w01);
        v_float32 vw10 = vx_setall_f32(w10);
        v_float32 vw11 = vx_setall_f32(w11);
        float buf00[VTraits<v_float32>::max_nlanes];
        float buf01[VTraits<v_float32>::max_nlanes];
        float buf10[VTraits<v_float32>::max_nlanes];
        float buf11[VTraits<v_float32>::max_nlanes];
        float bufo[VTraits<v_float32>::max_nlanes];
        for (; c + L <= C; c += L) {
            for (int k = 0; k < L; k++) {
                const float* pNC = baseN + (c + k) * xCStride;
                buf00[k] = pNC[off00];
                buf01[k] = pNC[off01];
                buf10[k] = pNC[off10];
                buf11[k] = pNC[off11];
            }
            v_float32 v00 = vx_load(buf00);
            v_float32 v01 = vx_load(buf01);
            v_float32 v10 = vx_load(buf10);
            v_float32 v11 = vx_load(buf11);
            v_float32 res = v_add(v_add(v_mul(v00, vw00), v_mul(v01, vw01)),
                                  v_add(v_mul(v10, vw10), v_mul(v11, vw11)));
            v_store(bufo, res);
            for (int k = 0; k < L; k++) {
                outBase[(size_t)(c + k) * yCStride + w] = bufo[k];
            }
        }
#endif
        for (; c < C; c++) {
            const float* pNC = baseN + c * xCStride;
            float v00 = pNC[off00];
            float v01 = pNC[off01];
            float v10 = pNC[off10];
            float v11 = pNC[off11];
            outBase[(size_t)c * yCStride + w] = w00 * v00 + w01 * v01 + w10 * v10 + w11 * v11;
        }
    }
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
