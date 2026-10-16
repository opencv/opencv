// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "flash_attention.hpp"
#include "fast_gemm.hpp"
#include "opencv2/core/hal/intrin.hpp"

#include <cfloat>

namespace cv { namespace dnn {

static constexpr int FLASH_ATTN_MAX_Q_BLOCK = 128;  // a 128 x 256 score tile (128 KB) stays in L2
static constexpr int FLASH_ATTN_MIN_Q_BLOCK = 16;   // smallest query block when splitting for threads
static constexpr int FLASH_ATTN_KV_BLOCK = 256;     // the key block the MLAS path uses

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

void flashAttention(const float* Q, const float* K, const float* V, float* out,
                    int B, int H, int Sq, int Skv, int D, int Dv, float scale) {
    CV_Assert(B > 0 && H > 0 && Sq > 0 && Skv > 0 && D > 0 && Dv > 0);
    const int nthreads = std::max(1, getNumThreads());
    int qb = std::min(FLASH_ATTN_MAX_Q_BLOCK, Sq);
    while (qb > FLASH_ATTN_MIN_Q_BLOCK &&
           (int64_t)B * H * ((Sq + qb - 1) / qb) < 4 * (int64_t)nthreads)
        qb /= 2;
    const int kb = std::min(FLASH_ATTN_KV_BLOCK, Skv);
    const int q_tiles = (Sq + qb - 1) / qb;
    const int tasks = B * H * q_tiles;

    FastGemmOpt opt;
    opt.init();
    opt.multi_thread = false;

    parallel_for_(Range(0, tasks), [&](const Range& r) {
        FastGemmOpt lopt = opt;
        AutoBuffer<float> buf((size_t)qb * kb + (size_t)qb * Dv + 2 * (size_t)qb);
        float* S = buf.data();
        float* O = S + (size_t)qb * kb;
        float* m = O + (size_t)qb * Dv;
        float* l = m + qb;

        for (int t = r.start; t < r.end; t++) {
            const int bh = t / q_tiles;
            const int q0 = (t % q_tiles) * qb;
            const int mq = std::min(qb, Sq - q0);
            const float* q = Q + ((size_t)bh * Sq + q0) * D;
            const float* k = K + (size_t)bh * Skv * D;
            const float* v = V + (size_t)bh * Skv * Dv;

            for (int i = 0; i < mq; i++) {
                m[i] = -FLT_MAX;
                l[i] = 0.f;
            }
            for (int k0 = 0; k0 < Skv; k0 += kb) {
                const int nk = std::min(kb, Skv - k0);
                fastGemm(false, true, mq, D, nk, D, scale, q, D, 1,
                         k + (size_t)k0 * D, D, 1, 0.f, S, nk, lopt);
                for (int i = 0; i < mq; i++) {
                    float* s = S + (size_t)i * nk;
                    const float mnew = std::max(m[i], flash_attn_row_max(s, nk));
                    const float rowsum = flash_attn_exp_sum(s, nk, mnew);
                    if (k0 > 0) {
                        const float a = std::exp(m[i] - mnew);
                        l[i] = l[i] * a + rowsum;
                        flash_attn_scale_row(O + (size_t)i * Dv, Dv, a);
                    } else {
                        l[i] = rowsum;
                    }
                    m[i] = mnew;
                }
                fastGemm(false, false, mq, nk, nk, Dv, 1.f, S, nk, 1,
                         v + (size_t)k0 * Dv, Dv, 1, k0 > 0 ? 1.f : 0.f, O, Dv, lopt);
            }

            const int b = bh / H, h = bh % H;
            for (int i = 0; i < mq; i++) {
                float* dst = out + (((size_t)b * Sq + q0 + i) * H + h) * Dv;
                const float* src = O + (size_t)i * Dv;
                const float inv = 1.f / l[i];
                for (int j = 0; j < Dv; j++)
                    dst[j] = src[j] * inv;
            }
        }
    });
}

}}  // namespace cv::dnn
