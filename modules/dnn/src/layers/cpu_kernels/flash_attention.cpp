// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "flash_attention.hpp"
#include "gemm_select.hpp"
#include "mlas_gemm.hpp"
#include <opencv2/core/hal/hal.hpp>
#include <opencv2/core/private.hpp>
#include <opencv2/core/hal/intrin.hpp>

namespace cv { namespace dnn {

bool flashAttentionEnabled()
{
#if CV_NEON && CV_NEON_AARCH64
    const GemmPath path = gemmPath();
    if (path != GEMM_PATH_AUTO)
        return path == GEMM_PATH_ENGINE;
    if (!mlasAvailable())
        return true;  // nothing else streams attention here
    // Measured on Apple M5: ahead of MLAS with SME, up to 20% behind it on NEON alone.
    static const bool sme = hal::gemm32fUsesSME();
    return sme;
#else
    return false;
#endif
}

// dst = max(dst, src) for n values.
static void maxInto(float* dst, const float* src, int n)
{
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int nl = VTraits<v_float32>::vlanes();
    for (; j <= n - nl; j += nl)
        vx_store(dst + j, v_max(vx_load(dst + j), vx_load(src + j)));
#endif
    for (; j < n; j++)
        dst[j] = std::max(dst[j], src[j]);
}

// p = exp(p - mx) and sum += p, elementwise for n values.
static void expSubAdd(float* p, const float* mx, float* sum, int n)
{
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int nl = VTraits<v_float32>::vlanes();
    for (; j <= n - nl; j += nl)
    {
        const v_float32 e = v_exp(v_sub(vx_load(p + j), vx_load(mx + j)));
        vx_store(p + j, e);
        vx_store(sum + j, v_add(vx_load(sum + j), e));
    }
#endif
    for (; j < n; j++)
    {
        p[j] = std::exp(p[j] - mx[j]);
        sum[j] += p[j];
    }
}

// dst = src * a for n values.
static void scaleRow(const float* src, float* dst, int n, float a)
{
    int j = 0;
#if (CV_SIMD || CV_SIMD_SCALABLE)
    const int nl = VTraits<v_float32>::vlanes();
    const v_float32 va = vx_setall_f32(a);
    for (; j <= n - nl; j += nl)
        vx_store(dst + j, v_mul(vx_load(src + j), va));
#endif
    for (; j < n; j++)
        dst[j] = src[j] * a;
}

void flashAttention(const float* Q, const float* K, const float* V, float* out,
                    int B, int H, int Sq, int Skv, int D, int Dv, float scale, bool kTransposed)
{
    CV_Assert(Q && K && V && out && B > 0 && H > 0 && Sq > 0 && Skv > 0 && D > 0 && Dv > 0);
    // Same block sizes as the MLAS kernel: Q rows per task and K/V rows per step.
    const int QB = std::min(Sq, 256), KB = std::min(Skv, 256);
    const int qBlocks = (Sq + QB - 1) / QB, tasks = B * H * qBlocks;
    const size_t fsz = sizeof(float);
    const size_t qPackBytes = alignSize(hal::gemm32fPackBSize(QB, D, CV_HAL_GEMM_2_T), 64);

    parallel_for_(Range(0, tasks), [&](const Range& r) {
        // Per task: packed Q block, scores S^T (KB x QB), output O (QB x Dv), and per-row max, sum, new max, step sum.
        AutoBuffer<uchar> qbuf(qPackBytes);
        AutoBuffer<float> buf((size_t)KB * QB + (size_t)QB * Dv + 4 * (size_t)QB);
        float* St = buf.data();
        float* O = St + (size_t)KB * QB;
        float* m = O + (size_t)QB * Dv;
        float* l = m + QB;
        float* mNew = l + QB;
        float* lStep = mNew + QB;
        const size_t zero = 0;
        for (int t = r.start; t < r.end; t++)
        {
            const int bh = t / qBlocks, q0 = t % qBlocks * QB, qn = std::min(QB, Sq - q0);
            const float* k = K + (size_t)bh * Skv * D;
            const float* v = V + (size_t)bh * Skv * Dv;
            // Q block is the B operand of S^T = K_blk * Q_blk^T, packed once for all K/V steps.
            hal::gemm32fPackB(Q + ((size_t)bh * Sq + q0) * D, D * fsz, qn, D,
                              CV_HAL_GEMM_2_T | CV_HAL_GEMM_SINGLE_THREAD, qbuf.data());
            for (int i = 0; i < qn; i++)
            {
                m[i] = -FLT_MAX;
                l[i] = 0.f;
            }
            for (int j0 = 0; j0 < Skv; j0 += KB)
            {
                const int kn = std::min(KB, Skv - j0);
                // S^T = scale * K_blk * Q_blk^T (kn x qn)
                if (kTransposed)
                    hal::gemm32fPacked(k + j0, Skv * fsz, qbuf.data(), St, qn * fsz, kn, qn, D, scale, 0.f,
                                       CV_HAL_GEMM_1_T | CV_HAL_GEMM_SINGLE_THREAD);
                else
                    hal::gemm32fPacked(k + (size_t)j0 * D, D * fsz, qbuf.data(), St, qn * fsz, kn, qn, D, scale, 0.f,
                                       CV_HAL_GEMM_SINGLE_THREAD);
                // Online softmax over each column of S^T: new max, exp, column sums.
                std::copy(m, m + qn, mNew);
                for (int j = 0; j < kn; j++)
                    maxInto(mNew, St + (size_t)j * qn, qn);
                std::fill(lStep, lStep + qn, 0.f);
                for (int j = 0; j < kn; j++)
                    expSubAdd(St + (size_t)j * qn, mNew, lStep, qn);
                for (int i = 0; i < qn; i++)
                {
                    const float corr = std::exp(m[i] - mNew[i]);
                    l[i] = l[i] * corr + lStep[i];
                    m[i] = mNew[i];
                    if (j0 > 0)
                        scaleRow(O + (size_t)i * Dv, O + (size_t)i * Dv, Dv, corr);
                }
                // O += P * V_blk with P = (S^T)^T; the first block writes O.
                hal::gemm32fBatch(1, St, &zero, qn * fsz, v + (size_t)j0 * Dv, &zero, Dv * fsz, O, &zero, Dv * fsz,
                                  qn, Dv, kn, 1.f, j0 == 0 ? 0.f : 1.f, CV_HAL_GEMM_1_T | CV_HAL_GEMM_SINGLE_THREAD);
            }
            // out[b, q0 + i, h, :] = O[i] / l[i]
            const int b = bh / H, h = bh % H;
            float* dst = out + (((size_t)b * Sq + q0) * H + h) * Dv;
            for (int i = 0; i < qn; i++)
                scaleRow(O + (size_t)i * Dv, dst + (size_t)i * H * Dv, Dv, 1.f / l[i]);
        }
    }, tasks);
}

}} // cv::dnn
