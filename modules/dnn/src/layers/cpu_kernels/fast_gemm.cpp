// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "../../precomp.hpp"
#include "fast_gemm.hpp"
#include "mlas_gemm.hpp"

namespace cv { namespace dnn {

int fastGemmMC(const FastGemmOpt &opt) { return hal::fastGemmMC(opt); }
int fastGemmNC(const FastGemmOpt &opt) { return hal::fastGemmNC(opt); }
int fastGemmKC(const FastGemmOpt &opt) { return hal::fastGemmKC(opt); }
int fastGemmNR(const FastGemmOpt &opt) { return hal::fastGemmNR(opt); }

size_t fastGemmPackBSize(size_t N, size_t K, const FastGemmOpt &opt) {
    return hal::fastGemmPackBSize(N, K, opt);
}

void fastGemmPackB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckTypeEQ(B.type(), CV_32F, "fastGemmPackB: only float32 is supported for now");

    auto B_shape = shape(B);
    int batch = total(B_shape, 0, B_shape.size() - 2),
        rows = B_shape[B_shape.size() - 2], cols = B_shape.back();
    // hal::fastGemmPackB takes the stored width and height of B and transposes it when asked
    size_t size_packed_B = hal::fastGemmPackBSize(trans ? rows : cols, trans ? cols : rows, opt);
    packed_B.resize(size_packed_B * batch);
    const float *b = B.ptr<const float>();
    for (int i = 0; i < batch; i++)
        hal::fastGemmPackB(trans, cols, rows, b + (size_t)i * rows * cols, cols,
                           packed_B.data() + (size_t)i * size_packed_B, opt);
}

void fastGemmPackB(bool trans, size_t N, size_t K, const float *B, size_t ldb, float *packed_B, const FastGemmOpt &opt) {
    hal::fastGemmPackB(trans, N, K, B, ldb, packed_B, opt);
}

bool fastGemmThinEligible(int M, int N, int K) {
    return hal::fastGemmThinEligible(M, N, K);
}

size_t fastGemmThinPackBSize(int N, int K) {
    return hal::fastGemmThinPackBSize(N, K);
}

void fastGemmThinPackB(int N, int K, const float* B, size_t ldb_K, size_t ldb_N, float* packed_B) {
    hal::fastGemmThinPackB(N, K, B, ldb_K, ldb_N, packed_B);
}

void fastGemmThin(int M, int N, int K, float alpha,
                  const float* A, int lda0, int lda1,
                  const float* packed_B, float beta,
                  float* C, int ldc, bool multi_thread) {
    hal::fastGemmThin(M, N, K, alpha, A, lda0, lda1, packed_B, beta, C, ldc, multi_thread);
}

void fastGemm(bool trans_a, int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *packed_B, float beta,
              float *C, int ldc, FastGemmOpt &opt) {
    hal::fastGemm(trans_a, M, N, K, alpha, A, lda, packed_B, beta, C, ldc, opt);
}

#ifdef HAVE_MLAS
// hal::fastGemm and hal::fastGemmBatch run these products on their own unblocked path,
// which is faster than MLAS for them; MLAS gets the rest.
static inline bool fast_gemm_small(int M, int N, int K) {
    return fastGemmThinEligible(M, N, K) || (uint64_t)M * N * K <= 10000;
}
#endif

void fastGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
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

void fastGemm(bool trans_a, bool trans_b,
              float alpha, const Mat &A, const Mat &B,
              float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/fastGemm: only support float32 for now");
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/fastGemm: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/fastGemm: B and C should have the same type");

    const auto shape_a = shape(A);
    CV_CheckEQ(shape_a.size(), static_cast<size_t>(2), "DNN/fastGemm: A must be 2-dimensional");
    const auto shape_b = shape(B);
    CV_CheckEQ(shape_b.size(), static_cast<size_t>(2), "DNN/fastGemm: B must be 2-dimensional");
    const auto shape_c = shape(C);
    CV_CheckEQ(shape_c.size(), static_cast<size_t>(2), "DNN/fastGemm: C must be 2-dimensional");

    int ma = shape_a[0], na = shape_a[1];
    int mb = shape_b[0], nb = shape_b[1];

    int lda0 = na, lda1 = 1, ldb0 = nb, ldb1 = 1, ldc = shape_c[1];

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    fastGemm(trans_a, trans_b, ma, na, mb, nb,
             alpha, a, lda0, lda1, b, ldb0, ldb1,
             beta, c, ldc, opt);
}

void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
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

void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *packed_B_offsets, const size_t *C_offsets,
                   int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                   const float *packed_B, float beta, float *C, int ldc, FastGemmOpt &opt) {
    hal::fastGemmBatch(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, A, lda0, lda1,
                       packed_B, beta, C, ldc, opt);
}

void fastGemmBatch(bool trans_a, bool trans_b,
                   float alpha, const Mat &A, const Mat &B,
                   float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/fastGemmBatch: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/fastGemmBatch: B and C should have the same type");
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/fastGemmBatch: only support float32 for now");

    const auto shape_a = shape(A);
    const auto shape_b = shape(B);
    const auto shape_c = shape(C);
    CV_CheckGE(shape_a.size(), static_cast<size_t>(2), "DNN/fastGemmBatch: A must be n-dimensional (n >= 2)");
    CV_CheckGE(shape_b.size(), static_cast<size_t>(2), "DNN/fastGemmBatch: B must be n-dimensional (n >= 2)");

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    MatMulHelper helper;
    helper.compute(trans_a, trans_b, shape_a, shape_b, shape_c);

    fastGemmBatch(helper.batch, helper.A_offsets.data(), helper.B_offsets.data(), helper.C_offsets.data(),
                  helper.M, helper.N, helper.K, alpha, a, helper.lda0, helper.lda1, b, helper.ldb0,
                  helper.ldb1, beta, c, helper.ldc, opt);
}

void fastGemmBatch(size_t batch,
                   const std::vector<size_t> &A_offsets, const std::vector<size_t> &B_offsets, const std::vector<size_t> &C_offsets,
                   int M, int N, int K, float alpha, const Mat&A, int lda0, int lda1,
                   const Mat&B, int ldb0, int ldb1, float beta, Mat&C, int ldc, FastGemmOpt &opt){
    hal::fastGemmBatch(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha,
                       A.ptr<const float>(), lda0, lda1, B.ptr<const float>(), ldb0, ldb1,
                       beta, C.ptr<float>(), ldc, opt);
}

void pagedAttnQKGemm(
    const Mat& Q,const std::vector<Mat> &K, Mat& A,
    int T_q, int Nq, int N_k, int T_s, int D, size_t T_k,
    float sm_scale, const FastGemmOpt &opt
) {
    size_t esz = Q.elemSize();

    for (size_t s = 0; s < K.size(); s++){
        CV_CheckTypeEQ(Q.type(), K[s].type(), "pagedAttnQKGemmKernel: Q and K should have the same type");
        CV_CheckTrue(esz == K[s].elemSize(), "pagedAttnQKGemmKernel: Q and K should have the same element size");
    }
    CV_CheckTypeEQ(Q.type(), A.type(), "pagedAttnQKGemmKernel: Q and A should have the same type");

    CV_CheckTrue(
        T_s % fastGemmNR(opt) == 0,
        "pagedAttnQKGemmKernel: T_s should be divisible by the macro tile size"
    );

    const auto shape_q = shape(Q);
    CV_CheckTrue(
        ((shape_q.size() == 3) || (shape_q.size() == 4)),
        "pagedAttnQKGemmKernel: Q must be 3D or 4D (T_q x Nq x D) or (kq_groups x T_q x Nq x D)"
    );
    const auto shape_a = shape(A);
    CV_CheckTrue( shape_a.size() == 4, "pagedAttnQKGemmKernel: A must be 4D (B x N_q x T_q x D)" );

    const int B = shape_a[0];
    for (size_t s = 0; s < K.size(); s++) {
        const auto shape_k = shape(K[s]);
        CV_CheckTrue(
            shape_k.size() == 3,
            "pagedAttnQKGemmKernel: each K must be 4D (B x N_k x T_s x D)"
        );
        CV_CheckEQ(shape_k[0], B, "pagedAttnQKGemmKernel: the batch size of K should be the same as A");
        CV_CheckEQ(shape_k[1], N_k, "pagedAttnQKGemmKernel: the number of heads in K should match that of Q");
        CV_Assert(shape_k[2] == D * T_s);
    }

    std::vector<const char*> packed_K;
    for (size_t s = 0; s < K.size(); s++){
        packed_K.push_back(K[s].ptr<const char>());
    }

    char*a = A.ptr<char>();
    bool isQ3D = shape_q.size() == 3;
    hal::fastGemmPagedQK(Q.ptr<const char>(), packed_K, a,
                         B, T_q, Nq, N_k, T_s, D, T_k,
                         sm_scale, esz, isQ3D, opt);
}

void pagedAttnAVGemm(
    const Mat& A,const std::vector<Mat> &V, Mat& Out,
    int T_q, int Nq, int N_k, int T_s, int D, int T_v,
    const FastGemmOpt &opt
) {
    size_t esz = A.elemSize();

    for (size_t s = 0; s < V.size(); s++) {
        CV_CheckTypeEQ(A.type(), V[s].type(), "pagedAttnAVGemmKernel: A and V should have the same type");
        CV_CheckTrue(esz == V[s].elemSize(), "pagedAttnAVGemmKernel: A and V should have the same element size");
    }
    CV_CheckTypeEQ(A.type(), Out.type(), "pagedAttnAVGemmKernel: A and Out should have the same type");

    CV_CheckTrue(
        T_s % fastGemmKC(opt) == 0,
        "pagedAttnAVGemmKernel: T_s should be divisible by the macro tile size"
    );

    const auto shape_a = shape(A);
    CV_CheckTrue(
        shape_a.size() == 4,
        "pagedAttnAVGemmKernel: A must be 4D (B x n_q x T_q x D)"
    );

    const int B = shape_a[0];
    for (size_t s = 0; s < V.size(); s++) {
        const auto shape_v = shape(V[s]);
        CV_CheckTrue(
            shape_v.size() == 3,
            "pagedAttnAVGemmKernel: each V must be 3D (B x N_k x (T_s * D))"
        );
        CV_CheckEQ(shape_v[0], B, "pagedAttnAVGemmKernel: the batch size of V should be the same as A");
        CV_CheckEQ(shape_v[1], N_k, "pagedAttnAVGemmKernel: the number of heads in V should match that of A");
        CV_CheckEQ(shape_v[2], (int)fastGemmPackBSize(D, T_s, opt),
                    "pagedAttnAVGemmKernel: the last dimension of V-pages should be be equal to packed D x T_s Mat size");
    }

    std::vector<const char*> packed_V;
    for (size_t s = 0; s < V.size(); s++){
        packed_V.push_back(V[s].ptr<const char>());
    }

    bool canonical_layout = shape(Out).size() == 3;
    hal::fastGemmPagedAV(A.ptr<const char>(), packed_V, Out.ptr<char>(),
                         B, T_q, Nq, N_k, T_s, D, T_v,
                         esz, canonical_layout, fastGemmPackBSize(D, T_s, opt), opt);
}

}} // cv::dnn
