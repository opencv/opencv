// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include "fast_gemm.hpp"

#include "fast_gemm_kernels.simd.hpp"
#include "layers/cpu_kernels/fast_gemm_kernels.simd_declarations.hpp"

namespace cv { namespace dnn {

#define FAST_GEMM_IMPLEMENT_PACK(N, suffix, styp, dtyp) \
static void fast_gemm_pack##N##suffix( int m, int k, const void* A_, \
                                      int lda0, int lda1, void* packA_ ) \
{ \
    const styp* A = (const styp*)A_; \
    dtyp* packA = (dtyp*)packA_; \
    for( int i = 0; i < m; i += N ) { \
        if (i + N-1 < m) { \
            const styp* a_ptr = A + lda0*i; \
            for( int j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                FAST_GEMM_LOAD_TO_BUF_##N(styp); \
                FAST_GEMM_PACK##suffix##_##N(buf, packA); \
            } \
        } else { \
            const styp* a_ptr[N]; \
            for (int k = 0; k < N; k++) a_ptr[k] = A + lda0*(i+k < m ? i+k : i); \
            for( int j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                FAST_GEMM_LOAD_TO_BUF_BORDERS_##N(styp); \
                FAST_GEMM_PACK##suffix##_##N(buf, packA); \
            } \
        } \
    } \
}

#define FAST_GEMM_LOAD_TO_BUF_6(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_6(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j] }

#define FAST_GEMM_LOAD_TO_BUF_8(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_8(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j] }

#define FAST_GEMM_LOAD_TO_BUF_12(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7], \
        a_ptr[j+lda0*8], a_ptr[j+lda0*9], a_ptr[j+lda0*10], a_ptr[j+lda0*11] }

#define FAST_GEMM_LOAD_TO_BUF_BORDERS_12(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j], \
        a_ptr[8][j], a_ptr[9][j], a_ptr[10][j], a_ptr[11][j] }

#define FAST_GEMM_PACK_COPY(src, dst, N) \
    memcpy((dst), (src), N*sizeof(src[0]))
#define FAST_GEMM_PACK_f32_8(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 8)
#define FAST_GEMM_PACK_f32_12(src, dst) FAST_GEMM_PACK_COPY((src), (dst), 12)

FAST_GEMM_IMPLEMENT_PACK(8, _f32, float, float)
FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float)

void fastGemmPackB(const Mat &B, std::vector<float> &packed_B, bool trans, FastGemmOpt &opt) {
    CV_CheckEQ(B.dims, 2, "fastGemmPackB: input mat should be two-dimensional");
    CV_CheckTypeEQ(B.type(), CV_32F, "fastGemmPackB: only float32 is supported for now");

    auto B_shape = shape(B);
    int K = B_shape[0], N = B_shape[1], ldb0 = N, ldb1 = 1;
    if (trans) {
        std::swap(K, N);
        std::swap(ldb0, ldb1);
    }

    int esz = B.elemSize(),
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_NR = FAST_GEMM_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    if (!packed_B.empty()) {
        packed_B.clear();
    }
    int total_size = static_cast<int>((N + NC - 1) / NC) * NC * K;
    packed_B.resize(total_size);

    const auto *ptr_B = B.ptr<char>();
    auto ptr_packed = packed_B.data();
    int n_tiles = (N + NC - 1) / NC;
    for (int r = 0; r < n_tiles; ++r) {
        int j0 = r * NC;
        int nc = N - j0 < NC ? N - j0 : NC;
        int _nc = static_cast<int>((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
        for (int k = 0; k < K; k += KC) {
            int kc = K - k < KC ? K - k : KC;
#if CV_TRY_AVX || CV_TRY_AVX2
            if (opt.use_avx || opt.use_avx2) {
                fast_gemm_pack8_f32(nc, kc, ptr_B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, ptr_packed);
            } else
#endif
#if CV_TRY_NEON
            if (opt.use_neon_aarch64) {
                fast_gemm_pack12_f32(nc, kc, ptr_B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, ptr_packed);
            } else
#endif
            {
                fast_gemm_pack8_f32(nc, kc, ptr_B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, ptr_packed);
            }
            ptr_packed += _nc * kc;
        }
    }
}

static void fast_gemm_f32(int k, const char *a_, const char *b_,
                          char *c_, int ldc, const void* palpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;
    float alpha = *(const float*)palpha;

    float sbuf[FAST_GEMM_F32_MR * FAST_GEMM_F32_NR];
    memset(sbuf, 0, sizeof(sbuf));
    for(int p = 0; p < k; p++) {
        for( int i = 0; i < FAST_GEMM_F32_MR; i++ ) {
            float ai = a[FAST_GEMM_F32_MR * p + i];
            for( int j = 0; j < FAST_GEMM_F32_NR; j++ )
                sbuf[i * FAST_GEMM_F32_NR + j] += b[FAST_GEMM_F32_NR * p + j] * ai;
        }
    }
    for (int i = 0; i < FAST_GEMM_F32_MR; i++) {
        for (int j = 0; j < FAST_GEMM_F32_NR; j++)
            c[i * ldc + j] += alpha * sbuf[i * FAST_GEMM_F32_NR + j];
    }
}

static void fast_gemm_macro_kernel(int m, int n, int k,
                                   const char *packA, const char *packB,
                                   const void* palpha, char *c, int ldc0,
                                   int MR, int NR, FastGemmOpt &opt) {
    int esz = sizeof(float);
    int ldc0_esz = ldc0 * esz;

    double tempC[FAST_GEMM_F32_MR * FAST_GEMM_F32_NR]; // make sure the buffer is big enough
    for(int i = 0; i < m; i += MR) {
        for(int j = 0; j < n; j += NR) {
            char* cptr0 = &c[i * ldc0_esz + j * esz];
            char* cptr = cptr0;
            int ldc = ldc0;
            int mr = m - i < MR ? m - i : MR;
            int nr = n - j < NR ? n - j : NR;
            int nr_esz = nr * esz;
            bool partial = (bool)((mr < MR) | (nr < NR));
            if (partial) {
                memset(tempC, 0, sizeof(tempC));
                cptr = (char*)tempC;
                ldc = NR;
                for(int p = 0; p < mr; p++)
                    memcpy(cptr + p*(ldc*esz), cptr0 + p*ldc0_esz, nr_esz);
            }
#if CV_TRY_AVX2
            if (opt.use_avx2) {
                opt_AVX2::fast_gemm_micro_kernel_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha, MR, NR);
            } else
#endif
#if CV_TRY_AVX
            if (opt.use_avx) {
                opt_AVX::fast_gemm_micro_kernel_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha, MR, NR);
            } else
#endif
#if CV_TRY_NEON && CV_NEON_AARCH64
            if (opt.use_neon_aarch64) {
                opt_NEON_AARCH64::fast_gemm8x12_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha, MR, NR);
            } else
#endif
            {
                fast_gemm_f32(k, packA + i * k * esz, packB + j * k * esz, cptr, ldc, palpha);
            }

            if (partial) {
                for(int p = 0; p < mr; p++)
                    memcpy(cptr0 + p*ldc0_esz, cptr + p*(ldc*esz), nr_esz);
            }
        }
    }
}

static void fast_gemm_thin(float alpha, float beta, int M, int N, int K,
                         const char *a_, int lda0, int lda1,
                         const char *b_, int ldb,
                         char *c_, int ldc) {
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

    int total = M; // outer loops
    int cost_per_thread = static_cast<int>(K * N); // inner loops
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
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

    const void* palpha = (const void*)&alpha;

    int esz = sizeof(float);
    int GEMM_MC = FAST_GEMM_F32_MC,
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_MR = FAST_GEMM_F32_MR,
        GEMM_NR = FAST_GEMM_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    size_t buff_size = KC * MC * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_MAX_STACKBUF;
    int m_tiles = (M + MC - 1) / MC;
    int n_tiles = (N + NC - 1) / NC;
    int total_tiles = m_tiles * n_tiles;

    auto fn = [&](const Range &r) {
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size)); // TODO: use AutoBuffer
        const char *packed_b_ = packed_b;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = c + (i0 * ldc + j0) * esz;
            packed_b_ = packed_b + j0 * K * esz;

            if (beta == 0.f) {
                for(int i = 0; i < mc; i++)
                    memset(c_block + i * ldc_block * esz, 0, nc * esz);
            } else if (beta != 1.f) {
                for(int i = 0; i < mc; i++) {
                    float* c_i = (float*)c_block + i * ldc_block;
                    for(int j = 0; j < nc; j++)
                        c_i[j] *= beta;
                }
            }

            int _nc = static_cast<int>((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR * esz;
            for(int k0 = 0; k0 < K; k0 += KC)
            {
                int kc = K - k0 < KC ? K - k0 : KC;
#if CV_TRY_NEON
                if (opt.use_neon_aarch64) {
                    fast_gemm_pack8_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                } else
#endif
                { // default, AVX, AVX2
                    fast_gemm_pack12_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                }
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b_, palpha,
                                           c_block, ldc_block, GEMM_MR, GEMM_NR, opt);
                packed_b_ += _nc * kc;
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    int total = total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
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

    const void* palpha = (const void*)&alpha;

    if (!trans_b && ldb1 == 1 && (M <= 4 || (uint64_t)M * N * K <= 10000)) {
        return fast_gemm_thin(alpha, beta, M, N, K, a, lda0, lda1, b, ldb0, c, ldc);
    }

    int esz = sizeof(float);

    int GEMM_MC = FAST_GEMM_F32_MC,
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_MR = FAST_GEMM_F32_MR,
        GEMM_NR = FAST_GEMM_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = FAST_GEMM_STORAGE / ((MC + NC) * esz);
    KC = KC > 8 ? KC : 8;
    KC = KC < K ? KC : K;

    size_t buff_size = KC * (MC + NC) * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_MAX_STACKBUF;
    int m_tiles = (M + MC - 1) / MC;
    int n_tiles = (N + NC - 1) / NC;
    int total_tiles = m_tiles * n_tiles;

    auto fn = [&](const Range &r) {
        char* pack_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        char* pack_b = pack_a + KC * MC * esz;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = c + (i0 * ldc + j0) * esz;

            if (beta == 0.f) {
                for(int i = 0; i < mc; i++)
                    memset(c_block + i * ldc_block * esz, 0, nc * esz);
            } else if (beta != 1.f) {
                for(int i = 0; i < mc; i++) {
                    float* c_i = (float*)c_block + i * ldc_block;
                    for(int j = 0; j < nc; j++)
                        c_i[j] *= beta;
                }
            }

            for(int k0 = 0; k0 < K; k0 += KC)
            {
                int kc = K - k0 < KC ? K - k0 : KC;
#if CV_TRY_NEON && CV_NEON_AARCH64
                if (opt.use_neon_aarch64) {
                    fast_gemm_pack8_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, pack_a);
                    fast_gemm_pack12_f32(nc, kc, b + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, pack_b);
                } else
#endif
                { // default, AVX, AVX2
                    fast_gemm_pack12_f32(mc, kc, a + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, pack_a);
                    fast_gemm_pack8_f32(nc, kc, b + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, pack_b);
                }
                fast_gemm_macro_kernel(mc, nc, kc, pack_a, pack_b, palpha,
                                       c_block, ldc_block, GEMM_MR, GEMM_NR, opt);
            }
        }

        if (!use_stackbuff) {
            free(pack_a);
        }
    };

    int total = total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fastGemm(bool trans_a, bool trans_b,
               float alpha, const Mat &A, const Mat &B,
               float beta, Mat &C, FastGemmOpt &opt) {
    CV_CheckTypeEQ(A.type(), B.type(), "DNN/gemm: A and B should have the same type");
    CV_CheckTypeEQ(B.type(), C.type(), "DNN/gemm: B and C should have the same type");
    CV_CheckTypeEQ(A.type(), CV_32F, "DNN/gemm: only support float32 for now");

    const auto shape_a = shape(A);
    const auto shape_b = shape(B);
    const auto shape_c = shape(C);

    int ma = shape_a[0], na = shape_a[1];
    int mb = shape_b[0], nb = shape_b[1];

    int lda0 = na, lda1 = 1, ldb0 = nb, ldb1 = 1, ldc = shape_c[1];

    // printf("trans_a=%d, trans_b=%d, ma=%d, na=%d, mb=%d, nb=%d, alpha=%f, beta=%f, lda0=%d, lda1=%d, ldb0=%d, ldb1=%d\n", static_cast<int>(trans_a), static_cast<int>(trans_b), ma, na, mb, nb, alpha, beta, lda0, lda1, ldb0, ldb1);

    const float *a = A.ptr<const float>();
    const float *b = B.ptr<const float>();
    float *c = C.ptr<float>();

    fastGemm(trans_a, trans_b, ma, na, mb, nb,
             alpha, a, lda0, lda1, b, ldb0, ldb1,
             beta, c, ldc, opt);
}

}} // cv::dnn
