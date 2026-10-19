// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

// Built with the SME flags only when CMake found SME support. Everything here runs in streaming
// mode, so no NEON code and no library calls are allowed inside these functions.

#include "gemm_sme.hpp"

#if defined(CV_GEMM_HAVE_SME)
#include <arm_sme.h>

namespace cv { namespace gemm_engine { namespace sme {

__arm_locally_streaming static int vectorLengthImpl()
{
    return (int)svcntw();
}

int vectorLength()
{
    return vectorLengthImpl();
}

// Writes alpha*acc (+ beta*C) for one row of ZA through the column mask pc.
#define CV_SME_STORE(tile, slice, ptr, pc)                                      \
    do {                                                                        \
        svfloat32_t acc = svread_hor_za32_f32_m(svundef_f32(), pt, tile, slice); \
        if (beta == 0.f)                                                        \
            svst1_f32(pc, ptr, svmul_f32_x(pt, acc, va));                       \
        else                                                                    \
        {                                                                       \
            svfloat32_t cv = svld1_f32(pc, ptr);                                \
            if (beta != 1.f)                                                    \
                cv = svmul_f32_x(pt, cv, vb);                                   \
            svst1_f32(pc, ptr, svmla_f32_x(pt, cv, acc, va));                   \
        }                                                                       \
    } while (0)

// 2VL x 2VL tiles: A panels of 2VL rows times B panels of 2VL columns, all four ZA tiles.
__arm_new("za") __arm_locally_streaming
static void kernel2x2(int m, int n, int k, const float* A, size_t aps, const float* B, size_t bps,
                      float alpha, float beta, float* C, size_t ldc)
{
    const int vl = (int)svcntw(), W = 2 * vl;
    const svbool_t pt = svptrue_b32();
    const svfloat32_t va = svdup_f32(alpha), vb = svdup_f32(beta);
    for (int j = 0; j < n; j += W)
    {
        const float* bp = B + (size_t)(j / W) * bps;
        const svbool_t pc0 = svwhilelt_b32(j, n), pc1 = svwhilelt_b32(j + vl, n);
        for (int i = 0; i < m; i += W)
        {
            const float* a = A + (size_t)(i / W) * aps;
            const float* b = bp;
            svzero_za();
            for (int p = 0; p < k; p++, a += W, b += W)
            {
                svfloat32_t a0 = svld1_f32(pt, a), a1 = svld1_f32(pt, a + vl);
                svfloat32_t b0 = svld1_f32(pt, b), b1 = svld1_f32(pt, b + vl);
                svmopa_za32_f32_m(0, pt, pt, a0, b0);
                svmopa_za32_f32_m(1, pt, pt, a0, b1);
                svmopa_za32_f32_m(2, pt, pt, a1, b0);
                svmopa_za32_f32_m(3, pt, pt, a1, b1);
            }
            const int rows = m - i < W ? m - i : W;
            for (int r = 0; r < rows; r++)
            {
                float* c = C + (size_t)(i + r) * ldc + j;
                if (r < vl)
                {
                    CV_SME_STORE(0, r, c, pc0);
                    CV_SME_STORE(1, r, c + vl, pc1);
                }
                else
                {
                    CV_SME_STORE(2, r - vl, c, pc0);
                    CV_SME_STORE(3, r - vl, c + vl, pc1);
                }
            }
        }
    }
}

// VL x 4VL tiles for short A: one A panel of VL rows times two B panels.
__arm_new("za") __arm_locally_streaming
static void kernel1x4(int m, int n, int k, const float* A, size_t aps, const float* B, size_t bps,
                      float alpha, float beta, float* C, size_t ldc)
{
    const int vl = (int)svcntw(), W = 2 * vl;
    const svbool_t pt = svptrue_b32();
    const svfloat32_t va = svdup_f32(alpha), vb = svdup_f32(beta);
    for (int j = 0; j < n; j += 2 * W)
    {
        const float* bp0 = B + (size_t)(j / W) * bps;
        const float* bp1 = j + W < n ? bp0 + bps : bp0;  // no second panel: reuse and mask it out
        const svbool_t pc0 = svwhilelt_b32(j, n), pc1 = svwhilelt_b32(j + vl, n);
        const svbool_t pc2 = svwhilelt_b32(j + W, n), pc3 = svwhilelt_b32(j + W + vl, n);
        for (int i = 0; i < m; i += vl)
        {
            const float* a = A + (size_t)(i / vl) * aps;
            const float* b0p = bp0;
            const float* b1p = bp1;
            svzero_za();
            for (int p = 0; p < k; p++, a += vl, b0p += W, b1p += W)
            {
                svfloat32_t a0 = svld1_f32(pt, a);
                svmopa_za32_f32_m(0, pt, pt, a0, svld1_f32(pt, b0p));
                svmopa_za32_f32_m(1, pt, pt, a0, svld1_f32(pt, b0p + vl));
                svmopa_za32_f32_m(2, pt, pt, a0, svld1_f32(pt, b1p));
                svmopa_za32_f32_m(3, pt, pt, a0, svld1_f32(pt, b1p + vl));
            }
            const int rows = m - i < vl ? m - i : vl;
            for (int r = 0; r < rows; r++)
            {
                float* c = C + (size_t)(i + r) * ldc + j;
                CV_SME_STORE(0, r, c, pc0);
                CV_SME_STORE(1, r, c + vl, pc1);
                CV_SME_STORE(2, r, c + W, pc2);
                CV_SME_STORE(3, r, c + W + vl, pc3);
            }
        }
    }
}
#undef CV_SME_STORE

void gemmPanels(int m, int n, int k, int R, const float* A, size_t aPanelStride,
                const float* B, size_t bPanelStride, float alpha, float beta, float* C, size_t ldc)
{
    if (R == vectorLength())
        kernel1x4(m, n, k, A, aPanelStride, B, bPanelStride, alpha, beta, C, ldc);
    else
        kernel2x2(m, n, k, A, aPanelStride, B, bPanelStride, alpha, beta, C, ldc);
}

}}} // cv::gemm_engine::sme

#endif // CV_GEMM_HAVE_SME
