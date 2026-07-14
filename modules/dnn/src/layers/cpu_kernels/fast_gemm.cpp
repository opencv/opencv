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
        int size_packed_B = opt_NEON::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_NEON::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += N * K * esz;
            packed_b += size_packed_B * esz;
        }
    } else
#endif
#if CV_TRY_AVX2
    if (opt.use_avx2) {
        int size_packed_B = opt_AVX2::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_AVX2::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += N * K * esz;
            packed_b += size_packed_B * esz;
        }
    } else
#endif
#if CV_TRY_AVX
    if (opt.use_avx) {
        int size_packed_B = opt_AVX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_AVX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += N * K * esz;
            packed_b += size_packed_B * esz;
        }
    } else
#endif
#if CV_TRY_LASX
    if (opt.use_lasx) {
        int size_packed_B = opt_LASX::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            opt_LASX::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += N * K * esz;
            packed_b += size_packed_B * esz;
        }
    } else
#endif
    {
        int size_packed_B = cpu_baseline::fastGemmPackBSize(N, K);
        packed_B.resize(size_packed_B * batch);
        auto *packed_b = (char*)packed_B.data();
        for (int i = 0; i < batch; i++) {
            cpu_baseline::fastGemmPackBKernel(b, packed_b, N, K, ldb0, ldb1, esz);
            b += N * K * esz;
            packed_b += size_packed_B * esz;
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

}} // cv::dnn
