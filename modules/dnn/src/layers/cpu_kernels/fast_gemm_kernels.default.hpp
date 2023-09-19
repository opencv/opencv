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

#define FAST_GEMM_DEFAULT_STORAGE (1<<20) // 2^20
#define FAST_GEMM_DEFAULT_MAX_STACKBUF (1 << 14)

#define FAST_GEMM_DEFAULT_F32_MC 64
#define FAST_GEMM_DEFAULT_F32_NC 240
#define FAST_GEMM_DEFAULT_F32_MR 8
#define FAST_GEMM_DEFAULT_F32_NR 12
#define FAST_GEMM_DEFAULT_F32_PACKED_STRIDE_K 256

#define FAST_GEMM_DEFAULT_IMPLEMENT_PACK(N, suffix, styp, dtyp) \
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
                FAST_GEMM_DEFAULT_LOAD_TO_BUF_##N(styp); \
                FAST_GEMM_DEFAULT_PACK##suffix##_##N(buf, packA); \
            } \
        } else { \
            const styp* a_ptr[N]; \
            for (int k = 0; k < N; k++) a_ptr[k] = A + lda0*(i+k < m ? i+k : i); \
            for( int j = 0; j < k*lda1; packA += N, j += lda1 ) \
            { \
                FAST_GEMM_DEFAULT_LOAD_TO_BUF_BORDERS_##N(styp); \
                FAST_GEMM_DEFAULT_PACK##suffix##_##N(buf, packA); \
            } \
        } \
    } \
}

#define FAST_GEMM_DEFAULT_LOAD_TO_BUF_8(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7] }

#define FAST_GEMM_DEFAULT_LOAD_TO_BUF_BORDERS_8(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j] }

#define FAST_GEMM_DEFAULT_LOAD_TO_BUF_12(styp) \
    styp buf[] = { \
        a_ptr[j], a_ptr[j+lda0], a_ptr[j+lda0*2], a_ptr[j+lda0*3], \
        a_ptr[j+lda0*4], a_ptr[j+lda0*5], a_ptr[j+lda0*6], a_ptr[j+lda0*7], \
        a_ptr[j+lda0*8], a_ptr[j+lda0*9], a_ptr[j+lda0*10], a_ptr[j+lda0*11] }

#define FAST_GEMM_DEFAULT_LOAD_TO_BUF_BORDERS_12(styp) \
    styp buf[] = { \
        a_ptr[0][j], a_ptr[1][j], a_ptr[2][j], a_ptr[3][j], \
        a_ptr[4][j], a_ptr[5][j], a_ptr[6][j], a_ptr[7][j], \
        a_ptr[8][j], a_ptr[9][j], a_ptr[10][j], a_ptr[11][j] }

#define FAST_GEMM_DEFAULT_PACK_COPY(src, dst, N) \
    memcpy((dst), (src), N*sizeof(src[0]))
#define FAST_GEMM_DEFAULT_PACK_f32_8(src, dst) FAST_GEMM_DEFAULT_PACK_COPY((src), (dst), 8)
#define FAST_GEMM_DEFAULT_PACK_f32_12(src, dst) FAST_GEMM_DEFAULT_PACK_COPY((src), (dst), 12)

namespace cv { namespace dnn { namespace cpu_baseline {

int fastGemmPackBSize(int N, int K);

void fastGemmPackBKernel(const char *B, char *packed_B, int N, int K, int ldb0, int ldb1, int esz);

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *B, int ldb0, int ldb1,
                    float beta, char *C, int ldc, int esz);
void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *packed_B, float beta, char *C, int ldc, int esz);

FAST_GEMM_DEFAULT_IMPLEMENT_PACK(8, _f32, float, float)
FAST_GEMM_DEFAULT_IMPLEMENT_PACK(12, _f32, float, float)

int fastGemmPackBSize(int N, int K) {
    int GEMM_NC = FAST_GEMM_DEFAULT_F32_NC, GEMM_NR = FAST_GEMM_DEFAULT_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;

    return static_cast<int>((N + NC - 1) / NC) * NC * K;
}

void fastGemmPackBKernel(const char *B, char *packed_B, int N, int K, int ldb0, int ldb1, int esz) {
    int GEMM_NC = FAST_GEMM_DEFAULT_F32_NC, GEMM_NR = FAST_GEMM_DEFAULT_F32_NR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_DEFAULT_F32_PACKED_STRIDE_K, K);

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

#if CV_SIMD128
static void fast_gemm8x12_f32(int k, const char *a_, const char *b_,
                       char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

    v_float32x4 s00 = v_setzero_f32(), s01 = s00, s02 = s00;
    v_float32x4 s10 = s00, s11 = s00, s12 = s00;
    v_float32x4 s20 = s00, s21 = s00, s22 = s00;
    v_float32x4 s30 = s00, s31 = s00, s32 = s00;
    v_float32x4 s40 = s00, s41 = s00, s42 = s00;
    v_float32x4 s50 = s00, s51 = s00, s52 = s00;
    v_float32x4 s60 = s00, s61 = s00, s62 = s00;
    v_float32x4 s70 = s00, s71 = s00, s72 = s00;

    for(int p = 0; p < k; p++, a += FAST_GEMM_DEFAULT_F32_MR, b += FAST_GEMM_DEFAULT_F32_NR) {
        v_float32x4 b0 = v_load(b), b1 = v_load(b + 4), b2 = v_load(b + 8);

        v_float32x4 a0 = v_setall_f32(*a);
        s00 = v_fma(b0, a0, s00);
        s01 = v_fma(b1, a0, s01);
        s02 = v_fma(b2, a0, s02);
        v_float32x4 a1 = v_setall_f32(*(a + 1));
        s10 = v_fma(b0, a1, s10);
        s11 = v_fma(b1, a1, s11);
        s12 = v_fma(b2, a1, s12);

        v_float32x4 a2 = v_setall_f32(*(a + 2));
        s20 = v_fma(b0, a2, s20);
        s21 = v_fma(b1, a2, s21);
        s22 = v_fma(b2, a2, s22);
        v_float32x4 a3 = v_setall_f32(*(a + 3));
        s30 = v_fma(b0, a3, s30);
        s31 = v_fma(b1, a3, s31);
        s32 = v_fma(b2, a3, s32);

        a0 = v_setall_f32(*(a + 4));
        s40 = v_fma(b0, a0, s40);
        s41 = v_fma(b1, a0, s41);
        s42 = v_fma(b2, a0, s42);
        a1 = v_setall_f32(*(a + 5));
        s50 = v_fma(b0, a1, s50);
        s51 = v_fma(b1, a1, s51);
        s52 = v_fma(b2, a1, s52);

        a2 = v_setall_f32(*(a + 6));
        s60 = v_fma(b0, a2, s60);
        s61 = v_fma(b1, a2, s61);
        s62 = v_fma(b2, a2, s62);
        a3 = v_setall_f32(*(a + 7));
        s70 = v_fma(b0, a3, s70);
        s71 = v_fma(b1, a3, s71);
        s72 = v_fma(b2, a3, s72);
    }

    v_float32x4 c0, c1, c2, c3, c4, c5, v_alpha = v_setall_f32(alpha);
#define FAST_GEMM_FINALE(row0, row1)       \
    c0 = v_load(c + row0 * ldc);         \
    c1 = v_load(c + row0 * ldc + 4);     \
    c2 = v_load(c + row0 * ldc + 8);     \
    c3 = v_load(c + row1 * ldc);         \
    c4 = v_load(c + row1 * ldc + 4);     \
    c5 = v_load(c + row1 * ldc + 8);     \
    c0 = v_fma(s##row0##0, v_alpha, c0); \
    c1 = v_fma(s##row0##1, v_alpha, c1); \
    c2 = v_fma(s##row0##2, v_alpha, c2); \
    c3 = v_fma(s##row1##0, v_alpha, c3); \
    c4 = v_fma(s##row1##1, v_alpha, c4); \
    c5 = v_fma(s##row1##2, v_alpha, c5); \
    v_store(c + row0 * ldc, c0);         \
    v_store(c + row0 * ldc + 4, c1);     \
    v_store(c + row0 * ldc + 8, c2);     \
    v_store(c + row1 * ldc, c3);         \
    v_store(c + row1 * ldc + 4, c4);     \
    v_store(c + row1 * ldc + 8, c5);

    FAST_GEMM_FINALE(0, 1);
    FAST_GEMM_FINALE(2, 3);
    FAST_GEMM_FINALE(4, 5);
    FAST_GEMM_FINALE(6, 7);
#undef FAST_GEMM_FINALE
}

#else
static void fast_gemm_f32(int k, const char *a_, const char *b_,
                          char *c_, int ldc, float alpha) {
    const float* a = (const float*)a_;
    const float* b = (const float*)b_;
    float* c = (float*)c_;

    float sbuf[FAST_GEMM_DEFAULT_F32_MR * FAST_GEMM_DEFAULT_F32_NR];
    memset(sbuf, 0, sizeof(sbuf));
    for(int p = 0; p < k; p++) {
        for( int i = 0; i < FAST_GEMM_DEFAULT_F32_MR; i++ ) {
            float ai = a[FAST_GEMM_DEFAULT_F32_MR * p + i];
            for( int j = 0; j < FAST_GEMM_DEFAULT_F32_NR; j++ )
                sbuf[i * FAST_GEMM_DEFAULT_F32_NR + j] += b[FAST_GEMM_DEFAULT_F32_NR * p + j] * ai;
        }
    }
    for (int i = 0; i < FAST_GEMM_DEFAULT_F32_MR; i++) {
        for (int j = 0; j < FAST_GEMM_DEFAULT_F32_NR; j++)
            c[i * ldc + j] += alpha * sbuf[i * FAST_GEMM_DEFAULT_F32_NR + j];
    }
}
#endif // CV_SIMD128

static void fast_gemm_macro_kernel(int m, int n, int k,
                                   const char *packed_A, const char *packed_B,
                                   float alpha, char *c, int ldc0, int esz) {
    int ldc0_esz = ldc0 * esz;

    double tempC[FAST_GEMM_DEFAULT_F32_MR * FAST_GEMM_DEFAULT_F32_NR]; // make sure the buffer is big enough
    for(int i = 0; i < m; i += FAST_GEMM_DEFAULT_F32_MR) {
        for(int j = 0; j < n; j += FAST_GEMM_DEFAULT_F32_NR) {
            char* cptr0 = &c[i * ldc0_esz + j * esz];
            char* cptr = cptr0;
            int ldc = ldc0;
            int mr = m - i < FAST_GEMM_DEFAULT_F32_MR ? m - i : FAST_GEMM_DEFAULT_F32_MR;
            int nr = n - j < FAST_GEMM_DEFAULT_F32_NR ? n - j : FAST_GEMM_DEFAULT_F32_NR;
            int nr_esz = nr * esz;
            bool partial = (bool)((mr < FAST_GEMM_DEFAULT_F32_MR) | (nr < FAST_GEMM_DEFAULT_F32_NR));
            if (partial) {
                memset(tempC, 0, sizeof(tempC));
                cptr = (char *)tempC;
                ldc = FAST_GEMM_DEFAULT_F32_NR;
                for(int p = 0; p < mr; p++)
                    memcpy(cptr + p * (ldc * esz), cptr0 + p * ldc0_esz, nr_esz);
            }
#if CV_SIMD128
            fast_gemm8x12_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);
#else
            fast_gemm_f32(k, packed_A + i * k * esz, packed_B + j * k * esz, cptr, ldc, alpha);
#endif

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
                    float beta, char *C, int ldc, int esz) {
    int GEMM_MC = FAST_GEMM_DEFAULT_F32_MC,
        GEMM_NC = FAST_GEMM_DEFAULT_F32_NC,
        GEMM_MR = FAST_GEMM_DEFAULT_F32_MR,
        GEMM_NR = FAST_GEMM_DEFAULT_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = FAST_GEMM_DEFAULT_STORAGE / ((MC + NC) * esz);
    KC = KC > 8 ? KC : 8;
    KC = KC < K ? KC : K;

    size_t buff_size = KC * (MC + NC) * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_DEFAULT_MAX_STACKBUF;
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

    int total = total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

void fastGemmKernel(int M, int N, int K,
                    float alpha, const char *A, int lda0, int lda1,
                    const char *packed_B, float beta, char *C, int ldc, int esz) {
    int GEMM_MC = FAST_GEMM_DEFAULT_F32_MC,
        GEMM_NC = FAST_GEMM_DEFAULT_F32_NC,
        GEMM_MR = FAST_GEMM_DEFAULT_F32_MR,
        GEMM_NR = FAST_GEMM_DEFAULT_F32_NR;

    int MC = (((GEMM_MC < M ? GEMM_MC : M) + GEMM_MR - 1) / GEMM_MR) * GEMM_MR;
    int NC = (((GEMM_NC < N ? GEMM_NC : N) + GEMM_NR - 1) / GEMM_NR) * GEMM_NR;
    int KC = std::min(FAST_GEMM_DEFAULT_F32_PACKED_STRIDE_K, K);

    size_t buff_size = KC * MC * esz;
    bool use_stackbuff = buff_size <= FAST_GEMM_DEFAULT_MAX_STACKBUF;
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

    int total = total_tiles;
    int cost_per_thread = static_cast<int>((K / KC) * (MC / GEMM_MR) * (NC / GEMM_NR));
    double nstripes = (size_t)total * cost_per_thread * (1 / 1024.0);
    parallel_for_(Range(0, total), fn, nstripes);
}

}}} // cv::dnn::cpu_baseline
