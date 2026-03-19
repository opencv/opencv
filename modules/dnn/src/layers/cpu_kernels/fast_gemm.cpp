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

#define CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "fast_gemm_kernels.simd.hpp"
#include "layers/cpu_kernels/fast_gemm_kernels.simd_declarations.hpp"
#undef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
#include "fast_gemm_kernels.default.hpp"

namespace cv { namespace dnn {

int fastGemmMC(const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmMC();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmMC();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmMC();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmMC();
    } else
#endif
    {
        return cpu_baseline::fastGemmMC();
    }
}

int fastGemmNC(const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmNC();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmNC();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmNC();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmNC();
    } else
#endif
    {
        return cpu_baseline::fastGemmNC();
    }
}

int fastGemmKC(const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return opt_NEON::fastGemmKC();
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return opt_AVX2::fastGemmKC();
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return opt_AVX::fastGemmKC();
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return opt_LASX::fastGemmKC();
    } else
#endif
    {
        return cpu_baseline::fastGemmKC();
    }
}

size_t fastGemmPackBSize(size_t N, size_t K, const FastGemmOpt &opt) {
#if CV_TRY_NEON
    if (opt.use_neon) {
        return static_cast<size_t>(opt_NEON::fastGemmPackBSize(N, K));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        return static_cast<size_t>(opt_AVX2::fastGemmPackBSize(N, K));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        return static_cast<size_t>(opt_AVX::fastGemmPackBSize(N, K));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        return static_cast<size_t>(opt_LASX::fastGemmPackBSize(N, K));
    } else
#endif
    {
        return static_cast<size_t>(cpu_baseline::fastGemmPackBSize(N, K));
    }
}

void fastGemmPackB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckTypeEQ(B.type(), CV_32F, "fastGemmPackB: only float32 is supported for now");

    auto B_shape = shape(B);
    int batch = total(B_shape, 0, B_shape.size() - 2),
        K = B_shape[B_shape.size() - 2], N = B_shape.back(), ldb0 = N, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

    const auto *b = B.ptr<const char>();
    int esz = B.elemSize();

#if CV_TRY_NEON
    if (opt.use_neon) {
        size_t size_packed_B = opt_NEON::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_NEON::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        size_t size_packed_B = opt_AVX2::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_AVX2::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        size_t size_packed_B = opt_AVX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_AVX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        size_t size_packed_B = opt_LASX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_LASX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    } else
#endif
    {
        size_t size_packed_B = cpu_baseline::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            cpu_baseline::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += (size_t)N * (size_t)K * (size_t)esz;
            packed_b += size_packed_B * (size_t)esz;
        }
    }
}

void fastGemmPackB(bool trans, size_t N, size_t K, const float *B, size_t ldb, float *packed_B, const FastGemmOpt &opt) {
    size_t ldb0 = ldb, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

    const auto &b = (const char *)B;
    auto *packed_b = (char *)packed_B;

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, sizeof(float));
    }
}

static void fast_gemm_thin(float alpha, float beta, int M, int N, int K,
                           const char *a_, int lda0, int lda1,
                           const char *b_, int ldb,
                           char *c_, int ldc, bool multi_thread) {
    const float* a = (const float*)a_;

    auto fn = [&](const Range &r) {
        for(int start = r.start ; start < r.end; start++ ) {
            float* c_i = (float*)c_ + start * ldc;
            if (beta == 0.f)
                for(int j = 0; j < N; j++ ) c_i[j] = 0.f;
            else if (beta != 1.f)
                for(int j = 0; j < N; j++ ) c_i[j] *= beta;
            for(int k = 0; k < K; k++ ) {
                const float* b_k = (const float*)b_ + k * ldb;
                float aval = alpha * a[start * lda0 + k * lda1];
                for(int j = 0; j < N; j++ )
                    c_i[j] += aval * b_k[j];
            }
        }
    };

    if (multi_thread) {
        int total = M; // outer loops
        int cost_per_thread = static_cast<int>(K * N); // inner loops
        double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
        parallel_for_(Range(0, total), fn, nstripes);
    } else {
        fn(Range(0, M));
    }
}

void fastGemm(bool trans_a, int M, int N, int K,
              float alpha, const float *A, int lda,
              const float *packed_B, float beta,
              float *C, int ldc, FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *packed_b = (const char *)packed_B;
    char *c = (char *)C;

    int lda0 = lda, lda1 = 1;
    if (trans_a) {
        std::swap(lda0, lda1);
    }

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
    {
        cpu_baseline::fastGemmKernel(M, N, K, alpha, a, lda0, lda1, packed_b, beta, c, ldc, sizeof(float), opt.multi_thread);
    }
}

void fastGemm(bool trans_a, bool trans_b, int ma, int na, int mb, int nb,
              float alpha, const float *A, int lda0, int lda1, const float *B, int ldb0, int ldb1,
              float beta, float *C, int ldc, FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *b = (const char *)B;
    char *c = (char *)C;

    int M = trans_a ? na : ma;
    int N = trans_b ? mb : nb;
    int K = trans_a ? ma : na;

    if (trans_a) {
        std::swap(lda0, lda1);
    }
    if (trans_b) {
        std::swap(ldb0, ldb1);
    }

    if (!trans_b && ldb1 == 1 && (M <= 4 || (uint64_t)M * N * K <= 10000)) {
        return fast_gemm_thin(alpha, beta, M, N, K, a, lda0, lda1, b, ldb0, c, ldc, opt.multi_thread);
    }

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                 b, ldb0, ldb1, beta,
                                 c, ldc, sizeof(float), opt.multi_thread);
    } else
#endif
    {
        cpu_baseline::fastGemmKernel(M, N, K, alpha, a, lda0, lda1,
                                     b, ldb0, ldb1, beta,
                                     c, ldc, sizeof(float), opt.multi_thread);
    }
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
    const char *a = (const char *)A;
    const char *b = (const char *)B;
    char *c = (char *)C;

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmBatchKernel(batch, A_offsets, B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    }
}

void fastGemmBatch(size_t batch, const size_t *A_offsets, const size_t *packed_B_offsets, const size_t *C_offsets,
                   int M, int N, int K, float alpha, const float *A, int lda0, int lda1,
                   const float *packed_B, float beta, float *C, int ldc, FastGemmOpt &opt) {
    const char *a = (const char *)A;
    const char *b = (const char *)packed_B;
    char *c = (char *)C;

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmBatchKernel(batch, A_offsets, packed_B_offsets, C_offsets, M, N, K, alpha, a, lda0, lda1, b, beta, c, ldc, sizeof(float));
    }
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
    const char *a = (const char *)A.ptr<const float>();
    const char *b = (const char *)B.ptr<const float>();
    char *c = (char *)C.ptr<float>();

#if CV_TRY_NEON
    if (opt.use_neon) {
        opt_NEON::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        opt_AVX::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        opt_LASX::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    } else
#endif
    {
        cpu_baseline::fastGemmBatchKernel(batch, A_offsets.data(), B_offsets.data(), C_offsets.data(), M, N, K, alpha, a, lda0, lda1, b, ldb0, ldb1, beta, c, ldc, sizeof(float));
    }
}


void pagedAttnQKGemm(
    const Mat& Q,const std::vector<Mat> &K, Mat& A,
    int T_q, int Nq, int N_k, int T_s, int D,
    const FastGemmOpt &opt
) {
    size_t esz = Q.elemSize();

    for (size_t s = 0; s < K.size(); s++){
        CV_CheckTypeEQ(Q.type(), K[s].type(), "pagedAttnQKGemmKernel: Q and K should have the same type");
        CV_CheckTrue(esz == K[s].elemSize(), "pagedAttnQKGemmKernel: Q and K should have the same element size");
    }
    CV_CheckTypeEQ(Q.type(), A.type(), "pagedAttnQKGemmKernel: Q and A should have the same type");

    CV_CheckTrue(
        T_s % fastGemmNC(opt) == 0,
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
        CV_CheckEQ(shape_k[2], D * T_s, "pagedAttnQKGemmKernel: the head dimension of K should match that of Q and A");
    }

    std::vector<const char*> packed_K;
    for (size_t s = 0; s < K.size(); s++){
        packed_K.push_back(K[s].ptr<const char>());
    }

    char*a = A.ptr<char>();
    bool isQ3D = shape_q.size() == 3;
#if CV_TRY_NEON
    if (opt.use_neon)
        opt_NEON::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2)
        opt_AVX2::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
#if CV_TRY_AVX
    if (opt.use_avx)
        opt_AVX::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx)
        opt_LASX::pagedAttnQKGemmKernel(
            Q.ptr<const char>(), packed_K, a,
            B, T_q, Nq, N_k, T_s, D,
            esz, isQ3D
        );
    else
#endif
    cpu_baseline::pagedAttnQKGemmKernel(
        Q.ptr<const char>(), packed_K, a,
        B, T_q, Nq, N_k, T_s, D,
        esz, isQ3D
    );


}


void pagedAttnAVGemm(
    const Mat& A,const std::vector<Mat> &V, Mat& Out,
    int T_q, int Nq, int N_k, int T_s, int D,
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
#if CV_TRY_NEON
    if (opt.use_neon)
        opt_NEON::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        opt_AVX2::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    }
    else
#endif
#if CV_TRY_AVX
    if (opt.use_avx){
        opt_AVX::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    }
    else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx){
        opt_LASX::pagedAttnAVGemmKernel(
            A.ptr<const char>(), packed_V, Out.ptr<char>(),
            B, T_q, Nq, N_k, T_s, D,
            esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
        );
    }
    else
#endif
    {
        cpu_baseline::pagedAttnAVGemmKernel(
                A.ptr<const char>(), packed_V, Out.ptr<char>(),
                B, T_q, Nq, N_k, T_s, D,
                esz, canonical_layout, fastGemmPackBSize(D, T_s, opt)
            );

    }


}

}} // cv::dnn
