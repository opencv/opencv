// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "gemm_dispatch.hpp"
#include "mlas_gemm.hpp"

namespace cv { namespace dnn {

void fastGemmPackB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckTypeEQ(B.type(), CV_32F, "fastGemmPackB: only float32 is supported for now");

    auto B_shape = shape(B);
    int batch = total(B_shape, 0, B_shape.size() - 2),
        rows = B_shape[B_shape.size() - 2], cols = B_shape.back();
    size_t size_packed_B = hal::fastGemmPackBSize(trans ? rows : cols, trans ? cols : rows, opt);
    packed_B.resize(size_packed_B * batch);
    const float *b = B.ptr<const float>();
    for (int i = 0; i < batch; i++)
        hal::fastGemmPackB(trans, cols, rows, b + (size_t)i * rows * cols, cols,
                           packed_B.data() + (size_t)i * size_packed_B, opt);
}

#ifdef HAVE_MLAS
// fastGemm runs these products on its own unblocked path, which beats a call into MLAS.
static inline bool fast_gemm_small(int M, int N, int K) {
    return hal::fastGemmThinEligible(M, N, K) || (uint64_t)M * N * K <= 10000;
}
#endif

void dispatchGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
                  float alpha, const float *A, int lda0, int lda1, const float *B, int ldb0, int ldb1,
                  float beta, float *C, int ldc, FastGemmOpt &opt) {
#ifdef HAVE_MLAS
    const int M = trans_a ? na : ma, N = trans_b ? mb : nb, K = trans_a ? ma : na;
    const int a0 = trans_a ? lda1 : lda0, a1 = trans_a ? lda0 : lda1;
    const int b0 = trans_b ? ldb1 : ldb0, b1 = trans_b ? ldb0 : ldb1;
    const bool a_row_major = (a0 == 1 || a1 == 1);
    const bool b_row_major = (b0 == 1 || b1 == 1);
    if (!(!trans_b && b1 == 1 && fast_gemm_small(M, N, K)) &&
        !opt.use_rvv && a_row_major && b_row_major) {
        const int phys_lda = std::max(a0, a1);
        const int phys_ldb = std::max(b0, b1);
        if (mlasSgemm(trans_a, trans_b, M, N, K,
                      alpha, A, phys_lda, B, phys_ldb,
                      beta, C, ldc)) {
            return;
        }
    }
#endif
    hal::fastGemm(trans_a, trans_b, ma, na, mb, nb, alpha, A, lda0, lda1, B, ldb0, ldb1, beta, C, ldc, opt);
}

void dispatchGemm(bool trans_a, bool trans_b,
                  float alpha, const Mat &A, const Mat &B,
                  float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/dispatchGemm: only support float32 for now");
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/dispatchGemm: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/dispatchGemm: B and C should have the same type");

    const auto shape_a = shape(A);
    CV_CheckEQ(shape_a.size(), static_cast<size_t>(2), "DNN/dispatchGemm: A must be 2-dimensional");
    const auto shape_b = shape(B);
    CV_CheckEQ(shape_b.size(), static_cast<size_t>(2), "DNN/dispatchGemm: B must be 2-dimensional");
    const auto shape_c = shape(C);
    CV_CheckEQ(shape_c.size(), static_cast<size_t>(2), "DNN/dispatchGemm: C must be 2-dimensional");

    int ma = shape_a[0], na = shape_a[1];
    int mb = shape_b[0], nb = shape_b[1];

    int lda0 = na, lda1 = 1, ldb0 = nb, ldb1 = 1, ldc = shape_c[1];

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    dispatchGemm(trans_a, trans_b, ma, na, mb, nb,
                 alpha, a, lda0, lda1, b, ldb0, ldb1,
                 beta, c, ldc, opt);
}

void dispatchGemmBatch(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                       int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                       const float *B, int ldb0, int ldb1, float beta, float *C, int ldc, FastGemmOpt &opt) {
#ifdef HAVE_MLAS
    bool a_ok = false, b_ok = false;
    bool mlas_trans_a = false, mlas_trans_b = false;
    int  mlas_lda = 0, mlas_ldb = 0;
    if (lda1 == 1)      { a_ok = true; mlas_trans_a = false; mlas_lda = lda0; }
    else if (lda0 == 1) { a_ok = true; mlas_trans_a = true;  mlas_lda = lda1; }
    if (ldb1 == 1)      { b_ok = true; mlas_trans_b = false; mlas_ldb = ldb0; }
    else if (ldb0 == 1) { b_ok = true; mlas_trans_b = true;  mlas_ldb = ldb1; }
    if (!(ldb1 == 1 && fast_gemm_small(M, N, K)) && !opt.use_rvv && a_ok && b_ok) {
        if (mlasSgemmBatch(batch, A_offsets, B_offsets, C_offsets,
                            mlas_trans_a, mlas_trans_b, M, N, K,
                            alpha, A, mlas_lda, B, mlas_ldb,
                            beta, C, ldc)) {
            return;
        }
    }
#endif
    hal::fastGemmBatch(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, A, lda0, lda1,
                       B, ldb0, ldb1, beta, C, ldc, opt);
}

void dispatchGemmBatch(bool trans_a, bool trans_b,
                       float alpha, const Mat &A, const Mat &B,
                       float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/dispatchGemmBatch: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/dispatchGemmBatch: B and C should have the same type");
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/dispatchGemmBatch: only support float32 for now");

    const auto shape_a = shape(A);
    const auto shape_b = shape(B);
    const auto shape_c = shape(C);
    CV_CheckGE(shape_a.size(), static_cast<size_t>(2), "DNN/dispatchGemmBatch: A must be n-dimensional (n >= 2)");
    CV_CheckGE(shape_b.size(), static_cast<size_t>(2), "DNN/dispatchGemmBatch: B must be n-dimensional (n >= 2)");

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    MatMulHelper helper;
    helper.compute(trans_a, trans_b, shape_a, shape_b, shape_c);

    dispatchGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                      helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1, b, helper.ldb0,
                      helper.ldb1, beta, c, helper.ldc, opt);
}

}} // cv::dnn
