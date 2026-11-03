// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"
#include <algorithm>
#include <cfloat>
#include <cmath>

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void flashAttnSoftmaxBlock(float* S, int rows, int n, float* O, int Dv, float* m, float* l, bool first);

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // namespace cv::dnn

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace cv { namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

static float flash_attn_row_max(const float* p, int n) {
    float mx = -FLT_MAX;
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VL = VTraits<v_float32>::vlanes();
    if (n >= VL) {
        v_float32 vm = vx_load(p);
        for (j = VL; j + VL <= n; j += VL)
            vm = v_max(vm, vx_load(p + j));
        mx = v_reduce_max(vm);
    }
#endif
    for (; j < n; j++)
        mx = std::max(mx, p[j]);
    return mx;
}

static float flash_attn_exp_sum(float* p, int n, float mx) {
    float s = 0.f;
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VL = VTraits<v_float32>::vlanes();
    v_float32 vmx = vx_setall_f32(mx), vs = vx_setzero_f32();
    for (; j + VL <= n; j += VL) {
        v_float32 e = v_exp(v_sub(vx_load(p + j), vmx));
        vx_store(p + j, e);
        vs = v_add(vs, e);
    }
    s = v_reduce_sum(vs);
#endif
    for (; j < n; j++) {
        p[j] = std::exp(p[j] - mx);
        s += p[j];
    }
    return s;
}

static void flash_attn_scale_row(float* p, int n, float a) {
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int VL = VTraits<v_float32>::vlanes();
    v_float32 va = vx_setall_f32(a);
    for (; j + VL <= n; j += VL)
        vx_store(p + j, v_mul(vx_load(p + j), va));
#endif
    for (; j < n; j++)
        p[j] *= a;
}

void flashAttnSoftmaxBlock(float* S, int rows, int n, float* O, int Dv, float* m, float* l, bool first) {
    for (int i = 0; i < rows; i++) {
        float* s = S + (size_t)i * n;
        const float mnew = std::max(m[i], flash_attn_row_max(s, n));
        const float rowsum = flash_attn_exp_sum(s, n, mnew);
        if (!first) {
            const float a = std::exp(m[i] - mnew);
            l[i] = l[i] * a + rowsum;
            flash_attn_scale_row(O + (size_t)i * Dv, Dv, a);
        } else {
            l[i] = rowsum;
        }
        m[i] = mnew;
    }
}

CV_CPU_OPTIMIZATION_NAMESPACE_END
}}  // namespace cv::dnn

#endif  // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
