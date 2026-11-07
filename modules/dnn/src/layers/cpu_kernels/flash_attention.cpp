// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "flash_attention.hpp"
#include "fast_gemm.hpp"

#include "flash_attention.simd.hpp"
#include "layers/cpu_kernels/flash_attention.simd_declarations.hpp"

#include <cfloat>

namespace cv { namespace dnn {

static constexpr int FLASH_ATTN_MAX_Q_BLOCK = 128;  // a 128 x 256 score tile (128 KB) stays in L2
static constexpr int FLASH_ATTN_MIN_Q_BLOCK = 32;   // smallest query block when splitting for threads
static constexpr int FLASH_ATTN_KV_BLOCK = 256;     // the key block the MLAS path uses

static void flash_attn_softmax_block(float* S, int rows, int n, float* O, int Dv, float* m, float* l, bool first) {
    CV_CPU_DISPATCH(flashAttnSoftmaxBlock, (S, rows, n, O, Dv, m, l, first), CV_CPU_DISPATCH_MODES_ALL);
}

// Adds the mask tile for rows [q0, q0 + rows) and keys [k0, k0 + n) to the score tile S.
static void flash_attn_add_mask(float* S, int rows, int n, const float* mask, const size_t* steps, int q0, int k0) {
    const float* mq = mask + (size_t)q0 * steps[2] + (size_t)k0 * steps[3];
    for (int i = 0; i < rows; i++) {
        float* s = S + (size_t)i * n;
        const float* mr = mq + (size_t)i * steps[2];
        if (steps[3] == 1) {
            for (int j = 0; j < n; j++)
                s[j] += mr[j];
        } else {
            for (int j = 0; j < n; j++)
                s[j] += mr[(size_t)j * steps[3]];
        }
    }
}

void flashAttention(const float* Q, const float* K, const float* V, float* out,
                    int B, int H, int Sq, int Skv, int D, int Dv, float scale,
                    const float* mask, const size_t* mask_steps, bool k_transposed) {
    CV_Assert(B > 0 && H > 0 && Sq > 0 && Skv > 0 && D > 0 && Dv > 0);
    CV_Assert(!mask || mask_steps);
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
            const int b = bh / H, h = bh % H;
            const int q0 = (t % q_tiles) * qb;
            const int mq = std::min(qb, Sq - q0);
            const float* q = Q + ((size_t)bh * Sq + q0) * D;
            const float* k = K + (size_t)bh * Skv * D;
            const float* v = V + (size_t)bh * Skv * Dv;
            const float* mask_bh = mask ? mask + b * mask_steps[0] + h * mask_steps[1] : nullptr;

            for (int i = 0; i < mq; i++) {
                m[i] = -FLT_MAX;
                l[i] = 0.f;
            }
            for (int k0 = 0; k0 < Skv; k0 += kb) {
                const int nk = std::min(kb, Skv - k0);
                if (k_transposed)
                    fastGemm(false, false, mq, D, D, nk, scale, q, D, 1,
                             k + k0, Skv, 1, 0.f, S, nk, lopt);
                else
                    fastGemm(false, true, mq, D, nk, D, scale, q, D, 1,
                             k + (size_t)k0 * D, D, 1, 0.f, S, nk, lopt);
                if (mask_bh)
                    flash_attn_add_mask(S, mq, nk, mask_bh, mask_steps, q0, k0);
                flash_attn_softmax_block(S, mq, nk, O, Dv, m, l, k0 == 0);
                fastGemm(false, false, mq, nk, nk, Dv, 1.f, S, nk, 1,
                         v + (size_t)k0 * Dv, Dv, 1, k0 > 0 ? 1.f : 0.f, O, Dv, lopt);
            }

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
