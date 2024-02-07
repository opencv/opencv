// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// This file is modified from the ficus (https://github.com/vpisarev/ficus/blob/master/runtime/ficus/impl/gemm.impl.h).
// Here is the original license:
/*
    This file is a part of ficus language project.
    See ficus/LICENSE for the licensing terms
*/

#include <opencv2/core/hal/intrin.hpp>
#include <opencv2/core/utility.hpp> // parallel_for_

#define FAST_GEMM_STORAGE (1<<20) // 2^20
#define FAST_GEMM_MAX_STACKBUF (1 << 14)

#define FAST_GEMM_F32_MC 64
#define FAST_GEMM_F32_NC 240
#define FAST_GEMM_F32_MR 8
#define FAST_GEMM_F32_NR 12
#define FAST_GEMM_F32_PACKED_STRIDE_K 64

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

namespace cv { namespace dnn { namespace cpu_baseline {

int fastGemmPackBSize(int N, int K);

void fastGemmPackBKernel(const char *B, char *packed_B, int N, int K, int ldb0, int ldb1, int esz);

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *B, int ldb0, int ldb1,
                    float beta, char *C, int ldc, int esz, bool multi_thread);
void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *packed_B, float beta, char *C, int ldc, int esz, bool multi_thread);

void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *B, int ldb0, int ldb1, float beta, char *C, int ldc, int esz);
void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *packed_B, float beta, char *C, int ldc, int esz);

FAST_GEMM_IMPLEMENT_PACK(8, _f32, float, float)
FAST_GEMM_IMPLEMENT_PACK(12, _f32, float, float)

int fastGemmPackBSize(int N, int K) {
    int GEMM_NC = FAST_GEMM_F32_NC, GEMM_NR = FAST_GEMM_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

    return static_cast<int>((N + NC - 1) / NC) * NC * K;
}

void fastGemmPackBKernel(const char *B, char *packed_B, int N, int K, int ldb0, int ldb1, int esz) {
    int GEMM_NC = FAST_GEMM_F32_NC, GEMM_NR = FAST_GEMM_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    int n_tiles = (N + NC - 1) / NC;
    for (int r = 0; r < n_tiles; ++r) {
        int j0 = r * NC;
        int nc = N - j0 < NC ? N - j0 : NC;
        int _nc = static_cast<int>((nc + GEMM_NR - 1) / GEMM_NR) * GEMM_NR * esz;
        for (int k = 0; k < K; k += KC) {
            int kc = K - k < KC ? K - k : KC;
            fast_gemm_pack12_f32(nc, kc, B + (k * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_B);
            packed_B += _nc * kc;
        }
    }
}

static inline void fast_gemm_f32(int k, const char *a_, const char *b_,
                                 char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

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
                                   const char *packed_A, const char *packed_B,
                                   float alpha, char *c, int ldc0, int esz) {
    int ldc0_esz = ldc0 * esz;

    double tempC[FAST_GEMM_F32_MR * FAST_GEMM_F32_NR]; // make sure the buffer is big enough
    for(int i = 0; i < m; i += FAST_GEMM_F32_MR) {
        for(int j = 0; j < n; j += FAST_GEMM_F32_NR) {
            char* cptr0 = &c[i * ldc0_esz + j * esz];
            char* cptr = cptr0;
            int ldc = ldc0;
            int mr = m - i < FAST_GEMM_F32_MR ? m - i : FAST_GEMM_F32_MR;
            int nr = n - j < FAST_GEMM_F32_NR ? n - j : FAST_GEMM_F32_NR;
            int nr_esz = nr * esz;
            bool partial = (bool)((mr < FAST_GEMM_F32_MR) | (nr < FAST_GEMM_F32_NR));
            if (partial) {
                memset(tempC, 0, sizeof(tempC));
                cptr = (char *)tempC;
                ldc = FAST_GEMM_F32_NR;
                for(int p = 0; p < mr; p++)
                    memcpy(cptr + p * (ldc * esz), cptr0 + p * ldc0_esz, nr_esz);
            }
            fast_gemm_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);

            if (partial) {
                for(int p = 0; p < mr; p++)
                    memcpy(cptr0 + p * ldc0_esz, cptr + p * (ldc * esz), nr_esz);
            }
        }
    }
}

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *B, int ldb0, int ldb1,
                    float beta, char *C, int ldc, int esz, bool multi_thread) {
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
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        char* packed_b = packed_a + KC * MC * esz;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = C + (i0 * ldc + j0) * esz;

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
                fast_gemm_pack8_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                fast_gemm_pack12_f32(nc, kc, B + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b, alpha, c_block, ldc_block, esz);
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    if (multi_thread) {
        int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
        double nstripes = (size_t)total_tiles * cost_per_thread * (1 / 1024.0);
        parallel_for_(Range(0, total_tiles), fn, nstripes);
    } else {
        fn(Range(0, total_tiles));
    }
}

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *packed_B, float beta, char *C, int ldc, int esz, bool multi_thread) {
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
        const char *packed_b_ = packed_B;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            int i0 = (tile_idx / n_tiles) * MC;
            int j0 = (tile_idx % n_tiles) * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            char* c_block = C + (i0 * ldc + j0) * esz;
            packed_b_ = packed_B + j0 * K * esz;

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
                fast_gemm_pack8_f32(mc, kc, A + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b_, alpha, c_block, ldc_block, esz);
                packed_b_ += _nc * kc;
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    if (multi_thread) {
        int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
        double nstripes = (size_t)total_tiles * cost_per_thread * (1 / 1024.0);
        parallel_for_(Range(0, total_tiles), fn, nstripes);
    } else {
        fn(Range(0, total_tiles));
    }
}

void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *B, int ldb0, int ldb1, float beta, char *C, int ldc, int esz) {
    int GEMM_MC = FAST_GEMM_F32_MC,
        GEMM_NC = FAST_GEMM_F32_NC,
        GEMM_MR = FAST_GEMM_F32_MR,
        GEMM_NR = FAST_GEMM_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_F32_PACKED_STRIDE_K, K);

    size_t buff_size = KC * (MC + NC) * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_MAX_STACKBUF;
    int m_tiles = (M + MC - 1) / MC;
    int n_tiles = (N + NC - 1) / NC;
    int total_tiles = m_tiles * n_tiles;

    auto fn = [&](const Range &r) {
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        char* packed_b = packed_a + KC * MC * esz;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            const int batch_index = static_cast<int>(tile_idx / total_tiles);
            const int m_tiles_index = static_cast<int>((tile_idx - batch_index * total_tiles) / n_tiles);
            const int n_tiles_index = static_cast<int>(tile_idx % n_tiles);

            int i0 = m_tiles_index * MC;
            int j0 = n_tiles_index * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            const char *a_block = A + A_offsets[batch_index] * esz;
            const char *b_block = B + B_offsets[batch_index] * esz;
            char* c_block = C + C_offsets[batch_index] * esz + (i0 * ldc + j0) * esz;

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
                // pack a
                fast_gemm_pack8_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);

                // pack b
                fast_gemm_pack12_f32(nc, kc, b_block + (k0 * ldb0 + j0 * ldb1) * esz, ldb1, ldb0, packed_b);

                // run kernel
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b, alpha, c_block, ldc_block, esz);
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    int total = batch * total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fastGemmBatchKernel(size_t batch, const size_t *A_offsets, const size_t *B_offsets, const size_t *C_offsets,
                         int M, int N, int K, float alpha, const char *A, int lda0, int lda1,
                         const char *packed_B, float beta, char *C, int ldc, int esz) {
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
        char* packed_a = (char*)(use_stackbuff ? alloca(buff_size) : malloc(buff_size));
        const char *packed_b = packed_B;
        int start = r.start;
        int end = r.end;

        for (int tile_idx = start; tile_idx < end; tile_idx++) {
            const int batch_index = static_cast<int>(tile_idx / total_tiles);
            const int m_tiles_index = static_cast<int>((tile_idx - batch_index * total_tiles) / n_tiles);
            const int n_tiles_index = static_cast<int>(tile_idx % n_tiles);

            int i0 = m_tiles_index * MC;
            int j0 = n_tiles_index * NC;
            int mc = M - i0 < MC ? M - i0 : MC;
            int nc = N - j0 < NC ? N - j0 : NC;
            int ldc_block = ldc;
            const char *a_block = A + A_offsets[batch_index] * esz;
            packed_b = packed_B + B_offsets[batch_index] * esz + j0 * K * esz;
            char* c_block = C + C_offsets[batch_index] * esz + (i0 * ldc + j0) * esz;

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
                // pack a
                fast_gemm_pack8_f32(mc, kc, a_block + (i0 * lda0 + k0 * lda1) * esz, lda0, lda1, packed_a);

                // run kernel
                fast_gemm_macro_kernel(mc, nc, kc, packed_a, packed_b, alpha, c_block, ldc_block, esz);
                packed_b += _nc * kc;
            }
        }

        if (!use_stackbuff) {
            free(packed_a);
        }
    };

    int total = batch * total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

}}} // cv::dnn::cpu_baseline

#undef FAST_GEMM_STORAGE
#undef FAST_GEMM_MAX_STACKBUF
#ifdef FAST_GEMM_F32_MC
#undef FAST_GEMM_F32_MC
#endif
#ifdef FAST_GEMM_F32_NC
#undef FAST_GEMM_F32_NC
#endif
#ifdef FAST_GEMM_F32_MR
#undef FAST_GEMM_F32_MR
#endif
#ifdef FAST_GEMM_F32_NR
#undef FAST_GEMM_F32_NR
#endif
#ifdef FAST_GEMM_F32_PACKED_STRIDE_K
#undef FAST_GEMM_F32_PACKED_STRIDE_K
#endif
#undef FAST_GEMM_IMPLEMENT_PACK
#undef FAST_GEMM_LOAD_TO_BUF_8
#undef FAST_GEMM_LOAD_TO_BUF_BORDERS_8
#undef FAST_GEMM_LOAD_TO_BUF_12
#undef FAST_GEMM_LOAD_TO_BUF_BORDERS_12
#undef FAST_GEMM_PACK_COPY
#undef FAST_GEMM_PACK_f32_8
#undef FAST_GEMM_PACK_f32_12
