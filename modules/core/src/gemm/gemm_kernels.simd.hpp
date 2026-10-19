// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// The 8x12 kernel is moved from modules/dnn/src/layers/cpu_kernels/fast_gemm_kernels.simd.hpp,
// which is modified from ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "opencv2/core/hal/intrin.hpp"
#include "gemm_engine.hpp"

namespace cv { namespace gemm_engine {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

#if CV_NEON && CV_NEON_AARCH64

static inline void transpose4x4(float32x4_t r0, float32x4_t r1, float32x4_t r2, float32x4_t r3,
                                float32x4_t& c0, float32x4_t& c1, float32x4_t& c2, float32x4_t& c3)
{
    float32x4_t t0 = vtrn1q_f32(r0, r1), t1 = vtrn2q_f32(r0, r1);
    float32x4_t t2 = vtrn1q_f32(r2, r3), t3 = vtrn2q_f32(r2, r3);
    c0 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    c1 = vreinterpretq_f32_f64(vtrn1q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
    c2 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t0), vreinterpretq_f64_f32(t2)));
    c3 = vreinterpretq_f32_f64(vtrn2q_f64(vreinterpretq_f64_f32(t1), vreinterpretq_f64_f32(t3)));
}

// Writes W values of X(p, j0..j0+W-1) for each p into dst (k x W), where X(p, j) = X[p*ld0 + j*ld1].
// Columns past `cols` are zero. Used for B panels (X = B) and for A panels (X = A^T).
static inline void packPanel(int cols, int k, const float* X, size_t ld0, size_t ld1, int W, float* dst)
{
    if (cols == W && ld1 == 1)
    {
        for (int p = 0; p < k; p++)
        {
            const float* src = X + p * ld0;
            float* out = dst + (size_t)p * W;
            int c = 0;
            for (; c <= W - 4; c += 4)
                vst1q_f32(out + c, vld1q_f32(src + c));
            for (; c < W; c++)
                out[c] = src[c];
        }
    }
    else if (cols == W && ld0 == 1 && W % 4 == 0)
    {
        // Columns are contiguous along p: transpose 4x4 blocks.
        int p = 0;
        for (; p <= k - 4; p += 4)
        {
            float* out = dst + (size_t)p * W;
            for (int c = 0; c < W; c += 4)
            {
                const float* s = X + c * ld1 + p;
                float32x4_t t0, t1, t2, t3;
                transpose4x4(vld1q_f32(s), vld1q_f32(s + ld1), vld1q_f32(s + ld1 * 2), vld1q_f32(s + ld1 * 3),
                             t0, t1, t2, t3);
                vst1q_f32(out + c, t0);
                vst1q_f32(out + W + c, t1);
                vst1q_f32(out + W * 2 + c, t2);
                vst1q_f32(out + W * 3 + c, t3);
            }
        }
        for (; p < k; p++)
            for (int c = 0; c < W; c++)
                dst[(size_t)p * W + c] = X[c * ld1 + p];
    }
    else
    {
        for (int p = 0; p < k; p++)
        {
            float* out = dst + (size_t)p * W;
            int c = 0;
            for (; c < cols; c++)
                out[c] = X[p * ld0 + c * ld1];
            for (; c < W; c++)
                out[c] = 0.f;
        }
    }
}

// Packs n columns of B into W-column panels of k rows each, one after another.
static inline void packBPanels(int n, int k, const float* B, size_t ldb0, size_t ldb1, int W, float* dst)
{
    for (int j = 0; j < n; j += W, dst += (size_t)W * k)
        packPanel(std::min(W, n - j), k, B + j * ldb1, ldb0, ldb1, W, dst);
}

// Packs m rows of A into 8-row panels (the last one has 4 rows when it fits).
static inline void packA8(int m, int k, const float* A, size_t lda0, size_t lda1, float* dst)
{
    for (int i = 0; i < m; i += GEMM_MR)
    {
        const int mr = std::min((int)GEMM_MR, m - i), R = mr <= 4 ? 4 : GEMM_MR;
        packPanel(mr, k, A + i * lda0, lda1, lda0, R, dst);
        dst += (size_t)R * k;
    }
}

// Writes alpha*acc (+ beta*C) for one row of up to 12 columns, `beta` already resolved for this k block.
static inline void storeRow12(float* c, float32x4_t s0, float32x4_t s1, float32x4_t s2,
                              float32x4_t va, float beta, int nr)
{
    if (nr == GEMM_NR)
    {
        if (beta == 0.f)
        {
            vst1q_f32(c, vmulq_f32(s0, va));
            vst1q_f32(c + 4, vmulq_f32(s1, va));
            vst1q_f32(c + 8, vmulq_f32(s2, va));
        }
        else if (beta == 1.f)
        {
            vst1q_f32(c, vfmaq_f32(vld1q_f32(c), s0, va));
            vst1q_f32(c + 4, vfmaq_f32(vld1q_f32(c + 4), s1, va));
            vst1q_f32(c + 8, vfmaq_f32(vld1q_f32(c + 8), s2, va));
        }
        else
        {
            const float32x4_t vb = vdupq_n_f32(beta);
            vst1q_f32(c, vfmaq_f32(vmulq_f32(vld1q_f32(c), vb), s0, va));
            vst1q_f32(c + 4, vfmaq_f32(vmulq_f32(vld1q_f32(c + 4), vb), s1, va));
            vst1q_f32(c + 8, vfmaq_f32(vmulq_f32(vld1q_f32(c + 8), vb), s2, va));
        }
        return;
    }
    // Edge: only the first nr columns are written.
    float t[GEMM_NR];
    vst1q_f32(t, s0); vst1q_f32(t + 4, s1); vst1q_f32(t + 8, s2);
    const float alpha = vgetq_lane_f32(va, 0);
    for (int j = 0; j < nr; j++)
        c[j] = beta == 0.f ? t[j] * alpha : std::fma(t[j], alpha, beta == 1.f ? c[j] : c[j] * beta);
}

// C(mr x nr) = alpha * packedA(8 x k) * packedB(k x 12) + beta * C.
static inline void gemm8x12(int k, const float* a, const float* b, float* c, size_t ldc,
                            float alpha, float beta, int mr, int nr)
{
    float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00;
    float32x4_t s10 = s00, s11 = s00, s12 = s00;
    float32x4_t s20 = s00, s21 = s00, s22 = s00;
    float32x4_t s30 = s00, s31 = s00, s32 = s00;
    float32x4_t s40 = s00, s41 = s00, s42 = s00;
    float32x4_t s50 = s00, s51 = s00, s52 = s00;
    float32x4_t s60 = s00, s61 = s00, s62 = s00;
    float32x4_t s70 = s00, s71 = s00, s72 = s00;

    for (int p = 0; p < k; p++, a += GEMM_MR, b += GEMM_NR)
    {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);

        s00 = vfmaq_laneq_f32(s00, b0, a0, 0);
        s01 = vfmaq_laneq_f32(s01, b1, a0, 0);
        s02 = vfmaq_laneq_f32(s02, b2, a0, 0);
        s10 = vfmaq_laneq_f32(s10, b0, a0, 1);
        s11 = vfmaq_laneq_f32(s11, b1, a0, 1);
        s12 = vfmaq_laneq_f32(s12, b2, a0, 1);

        s20 = vfmaq_laneq_f32(s20, b0, a0, 2);
        s21 = vfmaq_laneq_f32(s21, b1, a0, 2);
        s22 = vfmaq_laneq_f32(s22, b2, a0, 2);
        s30 = vfmaq_laneq_f32(s30, b0, a0, 3);
        s31 = vfmaq_laneq_f32(s31, b1, a0, 3);
        s32 = vfmaq_laneq_f32(s32, b2, a0, 3);

        a0 = vld1q_f32(a + 4);

        s40 = vfmaq_laneq_f32(s40, b0, a0, 0);
        s41 = vfmaq_laneq_f32(s41, b1, a0, 0);
        s42 = vfmaq_laneq_f32(s42, b2, a0, 0);
        s50 = vfmaq_laneq_f32(s50, b0, a0, 1);
        s51 = vfmaq_laneq_f32(s51, b1, a0, 1);
        s52 = vfmaq_laneq_f32(s52, b2, a0, 1);

        s60 = vfmaq_laneq_f32(s60, b0, a0, 2);
        s61 = vfmaq_laneq_f32(s61, b1, a0, 2);
        s62 = vfmaq_laneq_f32(s62, b2, a0, 2);
        s70 = vfmaq_laneq_f32(s70, b0, a0, 3);
        s71 = vfmaq_laneq_f32(s71, b1, a0, 3);
        s72 = vfmaq_laneq_f32(s72, b2, a0, 3);
    }

    const float32x4_t va = vdupq_n_f32(alpha);
    storeRow12(c, s00, s01, s02, va, beta, nr);
    if (mr > 1) storeRow12(c + ldc, s10, s11, s12, va, beta, nr);
    if (mr > 2) storeRow12(c + ldc * 2, s20, s21, s22, va, beta, nr);
    if (mr > 3) storeRow12(c + ldc * 3, s30, s31, s32, va, beta, nr);
    if (mr > 4) storeRow12(c + ldc * 4, s40, s41, s42, va, beta, nr);
    if (mr > 5) storeRow12(c + ldc * 5, s50, s51, s52, va, beta, nr);
    if (mr > 6) storeRow12(c + ldc * 6, s60, s61, s62, va, beta, nr);
    if (mr > 7) storeRow12(c + ldc * 7, s70, s71, s72, va, beta, nr);
}

// Same for a 4-row panel, used for row remainders of at most 4.
static inline void gemm4x12(int k, const float* a, const float* b, float* c, size_t ldc,
                            float alpha, float beta, int mr, int nr)
{
    float32x4_t s00 = vdupq_n_f32(0.f), s01 = s00, s02 = s00;
    float32x4_t s10 = s00, s11 = s00, s12 = s00;
    float32x4_t s20 = s00, s21 = s00, s22 = s00;
    float32x4_t s30 = s00, s31 = s00, s32 = s00;
    for (int p = 0; p < k; p++, a += 4, b += GEMM_NR)
    {
        float32x4_t a0 = vld1q_f32(a);
        float32x4_t b0 = vld1q_f32(b), b1 = vld1q_f32(b + 4), b2 = vld1q_f32(b + 8);
        s00 = vfmaq_laneq_f32(s00, b0, a0, 0);
        s01 = vfmaq_laneq_f32(s01, b1, a0, 0);
        s02 = vfmaq_laneq_f32(s02, b2, a0, 0);
        s10 = vfmaq_laneq_f32(s10, b0, a0, 1);
        s11 = vfmaq_laneq_f32(s11, b1, a0, 1);
        s12 = vfmaq_laneq_f32(s12, b2, a0, 1);
        s20 = vfmaq_laneq_f32(s20, b0, a0, 2);
        s21 = vfmaq_laneq_f32(s21, b1, a0, 2);
        s22 = vfmaq_laneq_f32(s22, b2, a0, 2);
        s30 = vfmaq_laneq_f32(s30, b0, a0, 3);
        s31 = vfmaq_laneq_f32(s31, b1, a0, 3);
        s32 = vfmaq_laneq_f32(s32, b2, a0, 3);
    }
    const float32x4_t va = vdupq_n_f32(alpha);
    storeRow12(c, s00, s01, s02, va, beta, nr);
    if (mr > 1) storeRow12(c + ldc, s10, s11, s12, va, beta, nr);
    if (mr > 2) storeRow12(c + ldc * 2, s20, s21, s22, va, beta, nr);
    if (mr > 3) storeRow12(c + ldc * 3, s30, s31, s32, va, beta, nr);
}

// C(m x n) = alpha * packedA * packedB + beta * C. B panels are 12 wide and `bPanelStride` floats apart.
static inline void macroKernel(int m, int n, int k, const float* packedA, const float* packedB, size_t bPanelStride,
                               float alpha, float beta, float* C, size_t ldc, bool rowsOuter)
{
    if (rowsOuter)
    {
        // Small B block: keep one A panel in L1 and sweep the B panels.
        for (int i = 0; i < m; i += GEMM_MR)
        {
            const int mr = std::min((int)GEMM_MR, m - i);
            const float* a = packedA + (size_t)i * k;
            for (int j = 0; j < n; j += GEMM_NR)
            {
                const int nr = std::min((int)GEMM_NR, n - j);
                const float* b = packedB + (size_t)(j / GEMM_NR) * bPanelStride;
                if (mr > 4)
                    gemm8x12(k, a, b, C + i * ldc + j, ldc, alpha, beta, mr, nr);
                else
                    gemm4x12(k, a, b, C + i * ldc + j, ldc, alpha, beta, mr, nr);
            }
        }
        return;
    }
    for (int j = 0; j < n; j += GEMM_NR)
    {
        const int nr = std::min((int)GEMM_NR, n - j);
        const float* b = packedB + (size_t)(j / GEMM_NR) * bPanelStride;
        for (int i = 0; i < m; i += GEMM_MR)
        {
            const int mr = std::min((int)GEMM_MR, m - i);
            const float* a = packedA + (size_t)i * k;
            float* c = C + i * ldc + j;
            if (mr > 4)
                gemm8x12(k, a, b, c, ldc, alpha, beta, mr, nr);
            else
                gemm4x12(k, a, b, c, ldc, alpha, beta, mr, nr);
        }
    }
}

// ---------------- small M: A is read in place, B is a k x (4*NV) row-major block ----------------

template<int MR, int NV, int Q>
static inline void smallStep(float32x4_t (&acc)[MR][NV], const float32x4_t (&a)[MR], const float* b)
{
    float32x4_t bv[NV];
    for (int v = 0; v < NV; v++)
        bv[v] = vld1q_f32(b + v * 4);
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < NV; v++)
            acc[r][v] = vfmaq_laneq_f32(acc[r][v], bv[v], a[r], Q);
}

// C(MR x cols) = alpha * A(MR x k) * B(k x cols) + beta * C, cols <= 4*NV, B row stride ldb.
template<int MR, int NV>
static inline void smallBlock(int k, const float* A, size_t lda, const float* b, size_t ldb,
                              float* c, size_t ldc, float alpha, float beta, int cols)
{
    float32x4_t acc[MR][NV];
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < NV; v++)
            acc[r][v] = vdupq_n_f32(0.f);

    int p = 0;
    for (; p <= k - 4; p += 4, b += ldb * 4)
    {
        float32x4_t a[MR];
        for (int r = 0; r < MR; r++)
            a[r] = vld1q_f32(A + r * lda + p);
        smallStep<MR, NV, 0>(acc, a, b);
        smallStep<MR, NV, 1>(acc, a, b + ldb);
        smallStep<MR, NV, 2>(acc, a, b + ldb * 2);
        smallStep<MR, NV, 3>(acc, a, b + ldb * 3);
    }
    for (; p < k; p++, b += ldb)
    {
        float32x4_t a[MR];
        for (int r = 0; r < MR; r++)
            a[r] = vdupq_n_f32(A[r * lda + p]);
        smallStep<MR, NV, 0>(acc, a, b);
    }

    const float32x4_t va = vdupq_n_f32(alpha), vb = vdupq_n_f32(beta);
    for (int r = 0; r < MR; r++)
    {
        float* cr = c + r * ldc;
        if (cols == NV * 4)
        {
            for (int v = 0; v < NV; v++)
            {
                float32x4_t res = beta == 0.f ? vmulq_f32(acc[r][v], va)
                                : vfmaq_f32(beta == 1.f ? vld1q_f32(cr + v * 4) : vmulq_f32(vld1q_f32(cr + v * 4), vb),
                                            acc[r][v], va);
                vst1q_f32(cr + v * 4, res);
            }
        }
        else
        {
            float t[NV * 4];
            for (int v = 0; v < NV; v++)
                vst1q_f32(t + v * 4, acc[r][v]);
            for (int j = 0; j < cols; j++)
                cr[j] = beta == 0.f ? t[j] * alpha : std::fma(t[j], alpha, beta == 1.f ? cr[j] : cr[j] * beta);
        }
    }
}

// Runs smallBlock with a run-time row count 1..MAXR.
template<int NV>
static inline void smallBlockRows(int mr, int k, const float* A, size_t lda, const float* b, size_t ldb,
                                  float* c, size_t ldc, float alpha, float beta, int cols)
{
    if (mr == 1) { smallBlock<1, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return; }
    if (mr == 2) { smallBlock<2, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return; }
    if constexpr (NV <= 4)
    {
        switch (mr)
        {
        case 3: smallBlock<3, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return;
        case 4: smallBlock<4, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return;
        case 5: smallBlock<5, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return;
        default: break;
        }
    }
    if constexpr (NV <= 3)
    {
        if (mr == 6) { smallBlock<6, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return; }
        if (mr == 7) { smallBlock<7, NV>(k, A, lda, b, ldb, c, ldc, alpha, beta, cols); return; }
    }
    CV_Error(Error::StsInternal, "too many rows for the small-M kernel");
}

// Most rows one small kernel holds in registers for NV column vectors.
static inline int smallMaxRows(int NV) { return NV >= 8 ? 2 : NV >= 4 ? 5 : 7; }

// ---------------- small M with B stored transposed: dot products along k ----------------

// C(MR x cols) = alpha * A(MR x k) * Bt(cols x k)^T + beta * C, cols <= 4, Bt row stride ldbt.
template<int MR>
static inline void dotBlock(int k, const float* A, size_t lda, const float* Bt, size_t ldbt,
                            float* c, size_t ldc, float alpha, float beta, int cols)
{
    const float* bt[4];
    for (int j = 0; j < 4; j++)
        bt[j] = Bt + (j < cols ? j : 0) * ldbt;
    float32x4_t acc[MR][4];
    for (int r = 0; r < MR; r++)
        for (int j = 0; j < 4; j++)
            acc[r][j] = vdupq_n_f32(0.f);
    int p = 0;
    for (; p <= k - 4; p += 4)
    {
        float32x4_t bv[4], a[MR];
        for (int j = 0; j < 4; j++)
            bv[j] = vld1q_f32(bt[j] + p);
        for (int r = 0; r < MR; r++)
            a[r] = vld1q_f32(A + r * lda + p);
        for (int r = 0; r < MR; r++)
            for (int j = 0; j < 4; j++)
                acc[r][j] = vfmaq_f32(acc[r][j], a[r], bv[j]);
    }
    for (int r = 0; r < MR; r++)
    {
        float s[4];
        for (int j = 0; j < 4; j++)
        {
            float v = vaddvq_f32(acc[r][j]);
            for (int q = p; q < k; q++)
                v = std::fma(A[r * lda + q], bt[j][q], v);
            s[j] = v;
        }
        float* cr = c + r * ldc;
        for (int j = 0; j < cols; j++)
            cr[j] = beta == 0.f ? s[j] * alpha : std::fma(s[j], alpha, beta == 1.f ? cr[j] : cr[j] * beta);
    }
}

static inline void dotBlockRows(int mr, int k, const float* A, size_t lda, const float* Bt, size_t ldbt,
                                float* c, size_t ldc, float alpha, float beta, int cols)
{
    switch (mr)
    {
    case 1: dotBlock<1>(k, A, lda, Bt, ldbt, c, ldc, alpha, beta, cols); break;
    case 2: dotBlock<2>(k, A, lda, Bt, ldbt, c, ldc, alpha, beta, cols); break;
    case 3: dotBlock<3>(k, A, lda, Bt, ldbt, c, ldc, alpha, beta, cols); break;
    default: dotBlock<4>(k, A, lda, Bt, ldbt, c, ldc, alpha, beta, cols); break;
    }
}

#endif // CV_NEON && CV_NEON_AARCH64

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::gemm_engine
