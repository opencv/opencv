// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"
#include "matmul_int8_kernels.hpp"
#include <type_traits>

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

bool matMulInt8PairedB();
void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, void* Y, const MatMulInt8OutLayout& lay,
                     int M, int N, int K, int a_zp, bool aIsU8,
                     const float* scale, const float* bias);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY)

// scale and bias round apart, as the unfused Mul and Add do: a DynamicQuantizeLinear downstream
// turns a one-ulp difference into a different integer, so GCC must not fuse them into an FMA
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif

static constexpr int MMI8_NR = 16, MMI8_KR = 4;
static constexpr int MMI8_PSTEP = MMI8_NR * MMI8_KR; // packed bytes per group of KR rows

// dpbusd matches the vector width: zmm with AVX-512 VNNI, ymm with AVX-VNNI in a 256-bit build
#if CV_SIMD512 && CV_AVX_512VNNI
#define MMI8_USE_VNNI 1
#define MMI8_USE_AVX2 0
static constexpr int MMI8_MR = 4;
#elif CV_SIMD256 && !CV_SIMD512 && CV_AVX_VNNI
#define MMI8_USE_VNNI 1
#define MMI8_USE_AVX2 0
// dpbusd needs about ten independent sums in flight to stay busy: 6 rows x 2 vectors is 12,
// which with B and A still fits the 16 ymm registers
static constexpr int MMI8_MR = 6;
#elif CV_SIMD256 && !CV_SIMD512 && CV_AVX2
#define MMI8_USE_VNNI 0
#define MMI8_USE_AVX2 1
// 4 rows x 2 sums, the 4 widened B vectors and the A pairs fit the 16 ymm registers
static constexpr int MMI8_MR = 4;
#else
#define MMI8_USE_VNNI 0
#define MMI8_USE_AVX2 0
static constexpr int MMI8_MR = 4;
#endif

#if CV_SIMD
// nlanes, not vlanes(): the register arrays below need a compile-time count
static constexpr int MMI8_NV = MMI8_PSTEP / VTraits<v_int8>::nlanes;

// The last n < 4 bytes of a row, zero-padded. A variable-length memcpy would be a call, and the
// compiler would then save every running sum before it, inside the K loop.
static CV_ALWAYS_INLINE int loadTail4(const uchar* p, int n)
{
    uchar t[MMI8_KR] = {};
    if (n > 0) t[0] = p[0];
    if (n > 1) t[1] = p[1];
    if (n > 2) t[2] = p[2];
    int v;
    memcpy(&v, t, MMI8_KR);
    return v;
}

// lane i accumulates the products of bytes 4i..4i+3; with VNNI a is read as uint8
static inline v_int32 dot4(const v_int8& a, const v_int8& b, const v_int32& c)
{
#if MMI8_USE_VNNI && CV_SIMD512
    return v_int32(_mm512_dpbusd_epi32(c.val, a.val, b.val));
#elif MMI8_USE_VNNI
    return v_int32(_mm256_dpbusd_epi32(c.val, a.val, b.val));
#else
    return v_dotprod_expand(a, b, c);
#endif
}

// each 4 bytes of A are broadcast straight from memory; XOR flips them to the operand's signedness
template<int MR, bool XOR>
static CV_ALWAYS_INLINE void macStep(v_int32 (&acc)[MR][MMI8_NV], const int8_t* bp,
                                     const uchar* const (&a)[MR], int k, const v_int8& vxor)
{
    v_int8 b[MMI8_NV];
    for (int j = 0; j < MMI8_NV; j++)
        b[j] = vx_load(bp + j * VTraits<v_int8>::vlanes());
    for (int i = 0; i < MR; i++) {
        int a4;
        memcpy(&a4, a[i] + k, MMI8_KR);
        v_int8 av = v_reinterpret_as_s8(vx_setall_s32(a4));
        if constexpr (XOR)
            av = v_xor(av, vxor);
        for (int j = 0; j < MMI8_NV; j++)
            acc[i][j] = dot4(av, b[j], acc[i][j]);
    }
}
#endif

#if MMI8_USE_VNNI && !CV_SIMD512
#define MMI8_ROW6(i, a4) { \
        __m256i av = _mm256_set1_epi32(a4); \
        if constexpr (XOR) av = _mm256_xor_si256(av, vxor); \
        c##i##0 = _mm256_dpbusd_epi32(c##i##0, av, b0); \
        c##i##1 = _mm256_dpbusd_epi32(c##i##1, av, b1); }

template<bool XOR>
static void accumulateVnni6(const uchar* const (&a)[6], const int8_t* bp, int K,
                            int (&accbuf)[6][MMI8_NR])
{
    const __m256i vxor = _mm256_set1_epi32((int)0x80808080u);
    __m256i c00 = _mm256_setzero_si256(), c01 = c00, c10 = c00, c11 = c00, c20 = c00, c21 = c00,
            c30 = c00, c31 = c00, c40 = c00, c41 = c00, c50 = c00, c51 = c00;
    int k = 0;
    auto load4 = [&](int i) { int v; memcpy(&v, a[i] + k, MMI8_KR); return v; };
    for (; k + MMI8_KR <= K; k += MMI8_KR, bp += MMI8_PSTEP) {
        __m256i b0 = _mm256_loadu_si256((const __m256i*)bp);
        __m256i b1 = _mm256_loadu_si256((const __m256i*)(bp + 32));
        MMI8_ROW6(0, load4(0)) MMI8_ROW6(1, load4(1)) MMI8_ROW6(2, load4(2))
        MMI8_ROW6(3, load4(3)) MMI8_ROW6(4, load4(4)) MMI8_ROW6(5, load4(5))
    }
    __m256i* out = (__m256i*)&accbuf[0][0];
    _mm256_store_si256(out + 0, c00); _mm256_store_si256(out + 1, c01);
    _mm256_store_si256(out + 2, c10); _mm256_store_si256(out + 3, c11);
    _mm256_store_si256(out + 4, c20); _mm256_store_si256(out + 5, c21);
    _mm256_store_si256(out + 6, c30); _mm256_store_si256(out + 7, c31);
    _mm256_store_si256(out + 8, c40); _mm256_store_si256(out + 9, c41);
    _mm256_store_si256(out + 10, c50); _mm256_store_si256(out + 11, c51);
    // The K % 4 tail works on the stored sums: with all 12 still live after the loop, the
    // compiler would save them to the stack on every iteration instead
    if (k < K) {
        __m256i b0 = _mm256_loadu_si256((const __m256i*)bp);
        __m256i b1 = _mm256_loadu_si256((const __m256i*)(bp + 32));
        for (int i = 0; i < 6; i++) {
            __m256i av = _mm256_set1_epi32(loadTail4(a[i] + k, K - k));
            if constexpr (XOR)
                av = _mm256_xor_si256(av, vxor);
            out[2 * i] = _mm256_dpbusd_epi32(out[2 * i], av, b0);
            out[2 * i + 1] = _mm256_dpbusd_epi32(out[2 * i + 1], av, b1);
        }
    }
}
#undef MMI8_ROW6
#endif

#if MMI8_USE_AVX2
// AVX2 has no u8 x s8 dot product that cannot saturate, so A and B are widened to int16 for
// vpmaddwd; B is packed as k pairs per column (see matMulInt8PairedB), two sums per row.
#define MMI8_AVX2_ROW(i) if constexpr (i < R) { \
        int p01, p23; \
        memcpy(&p01, aw[i] + k, sizeof(p01)); \
        memcpy(&p23, aw[i] + k + 2, sizeof(p23)); \
        __m256i av = _mm256_set1_epi32(p01); \
        c##i##0 = _mm256_add_epi32(c##i##0, _mm256_madd_epi16(av, b0)); \
        c##i##1 = _mm256_add_epi32(c##i##1, _mm256_madd_epi16(av, b1)); \
        av = _mm256_set1_epi32(p23); \
        c##i##0 = _mm256_add_epi32(c##i##0, _mm256_madd_epi16(av, b2)); \
        c##i##1 = _mm256_add_epi32(c##i##1, _mm256_madd_epi16(av, b3)); }

template<int R, bool XOR>
static void accumulateAvx2(const uchar* const (&a)[R], const int8_t* bp, int K,
                           int (&accbuf)[R][MMI8_NR])
{
    constexpr int KC = 256;
    int16_t CV_DECL_ALIGNED(32) aw[R][KC];
    const __m128i vxor = _mm_set1_epi8((char)0x80);
    __m256i c00 = _mm256_setzero_si256(), c01 = c00, c10 = c00, c11 = c00,
            c20 = c00, c21 = c00, c30 = c00, c31 = c00;
    const int Kpad = (K + MMI8_KR - 1) & ~(MMI8_KR - 1);
    for (int k0 = 0; k0 < Kpad; k0 += KC) {
        const int kc = std::min(KC, Kpad - k0);
        // this chunk of each row as int16; past K it is zero, like the packed B there
        for (int i = 0; i < R; i++) {
            const uchar* ai = a[i] + k0;
            int k = 0;
            for (; k + 16 <= kc && k0 + k + 16 <= K; k += 16) {
                __m128i t = _mm_loadu_si128((const __m128i*)(ai + k));
                if constexpr (XOR)
                    t = _mm_xor_si128(t, vxor);
                _mm256_store_si256((__m256i*)(aw[i] + k), _mm256_cvtepu8_epi16(t));
            }
            for (; k < kc; k++)
                aw[i][k] = k0 + k < K ? (int16_t)(uchar)(ai[k] ^ (XOR ? 0x80 : 0)) : 0;
        }
        for (int k = 0; k < kc; k += MMI8_KR, bp += MMI8_PSTEP) {
            __m256i b0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)bp));
            __m256i b1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(bp + 16)));
            __m256i b2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(bp + 32)));
            __m256i b3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(bp + 48)));
            MMI8_AVX2_ROW(0) MMI8_AVX2_ROW(1) MMI8_AVX2_ROW(2) MMI8_AVX2_ROW(3)
        }
    }
    __m256i* out = (__m256i*)&accbuf[0][0];
    out[0] = c00; out[1] = c01;
    if constexpr (R > 1) { out[2] = c10; out[3] = c11; }
    if constexpr (R > 2) { out[4] = c20; out[5] = c21; }
    if constexpr (R > 3) { out[6] = c30; out[7] = c31; }
}
#undef MMI8_AVX2_ROW
#endif

// rows [r, r+mr) x one panel; rows past mr repeat the last row and are not stored.
// A float Y gets the int32 sum times scale plus bias, while it is still in registers
template<int MR, bool CORR, bool XOR, typename _Tp>
static void gemmTile(const uchar* A, size_t lda, int r, int mr, const int8_t* bp, int K, int zp,
                     const int* colsum, const int* corr, const int* rt, const float* scale,
                     const float* fbias, _Tp* Y, const MatMulInt8OutLayout& lay, int nvalid)
{
    constexpr bool FLT = std::is_same<_Tp, float>::value;
    const uchar* a[MR];
    for (int i = 0; i < MR; i++)
        a[i] = A + std::min(i, mr - 1) * lda;
#if CV_SIMD
    const int VLI = VTraits<v_int32>::vlanes();
    int CV_DECL_ALIGNED(CV_SIMD_WIDTH) accbuf[MR][MMI8_NR];
#if MMI8_USE_AVX2
    accumulateAvx2<MR, XOR>(a, bp, K, accbuf);
#else
#if MMI8_USE_VNNI && !CV_SIMD512
    if constexpr (MR == 6) {
        accumulateVnni6<XOR>(a, bp, K, accbuf);
    } else
#endif
    {
        const v_int8 vxor = v_reinterpret_as_s8(vx_setall_s32((int)0x80808080u));
        v_int32 acc[MR][MMI8_NV];
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < MMI8_NV; j++)
                acc[i][j] = vx_setzero_s32();

        int k = 0;
        for (; k + MMI8_KR <= K; k += MMI8_KR, bp += MMI8_PSTEP)
            macStep<MR, XOR>(acc, bp, a, k, vxor);
        for (int i = 0; i < MR; i++)
            for (int j = 0; j < MMI8_NV; j++)
                v_store_aligned(accbuf[i] + j * VLI, acc[i][j]);
        // the K % 4 tail works on the stored sums, as in accumulateVnni6
        if (k < K) {
            for (int i = 0; i < MR; i++) {
                v_int8 av = v_reinterpret_as_s8(vx_setall_s32(loadTail4(a[i] + k, K - k)));
                if constexpr (XOR)
                    av = v_xor(av, vxor);
                for (int j = 0; j < MMI8_NV; j++)
                    v_store_aligned(accbuf[i] + j * VLI,
                                    dot4(av, vx_load(bp + j * VTraits<v_int8>::vlanes()),
                                         vx_load_aligned(accbuf[i] + j * VLI)));
            }
        }
    }
#endif

    v_int32 vzp = vx_setall_s32(-zp), bias[MMI8_NV];
    for (int j = 0; j < MMI8_NV; j++)
        bias[j] = v_mul(vzp, vx_load(colsum + j * VLI));
    for (int i = 0; i < mr; i++) {
        _Tp buf[MMI8_NR];
        _Tp* yrow = Y + lay.rowOfs(r + i);
        _Tp* y = nvalid == MMI8_NR ? yrow : buf;
        for (int j = 0; j < MMI8_NV; j++) {
            v_int32 s = v_add(vx_load_aligned(accbuf[i] + j * VLI), bias[j]);
            if constexpr (CORR)
                s = v_muladd(vx_load(corr + j * VLI), vx_setall_s32(rt[i]), s);
            if constexpr (FLT)
                v_store(y + j * VLI, v_add(v_mul(v_cvt_f32(s), vx_load(scale + j * VLI)),
                                           vx_load(fbias + j * VLI)));
            else
                v_store(y + j * VLI, s);
        }
        if (y == buf)
            for (int j = 0; j < nvalid; j++)
                yrow[j] = buf[j];
    }
#else
    for (int i = 0; i < mr; i++) {
        _Tp* yrow = Y + lay.rowOfs(r + i);
        for (int j = 0; j < nvalid; j++) {
            int s = 0;
            for (int k = 0; k < K; k++)
                s += (int)(schar)(a[i][k] ^ (XOR ? 0x80 : 0)) *
                     bp[(k / MMI8_KR) * MMI8_PSTEP + j * MMI8_KR + k % MMI8_KR];
            s -= zp * colsum[j];
            if constexpr (CORR)
                s += corr[j] * rt[i];
            if constexpr (FLT)
                yrow[j] = (float)s * scale[j] + fbias[j];
            else
                yrow[j] = s;
        }
    }
#endif
}

template<bool CORR, bool XOR, typename _Tp>
static void gemmInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                          const int* corr, const int* rowterm, const float* scale,
                          const float* bias, _Tp* Y, const MatMulInt8OutLayout& lay,
                          int M, int N, int K, int zp)
{
    constexpr int TASK_ROWS = 16 * MMI8_MR, TASK_PANELS = 2;
    const int npanels = (N + MMI8_NR - 1) / MMI8_NR;
    const size_t panelSize = (size_t)((K + MMI8_KR - 1) / MMI8_KR) * MMI8_PSTEP;
    const int ngroups = (npanels + TASK_PANELS - 1) / TASK_PANELS;
    const int ntasks = (M + TASK_ROWS - 1) / TASK_ROWS * ngroups;

    parallel_for_(Range(0, ntasks), [&](const Range& range) {
        for (int t = range.start; t < range.end; t++) {
            int r0 = t / ngroups * TASK_ROWS, r1 = std::min(M, r0 + TASK_ROWS);
            int p0 = t % ngroups * TASK_PANELS, p1 = std::min(npanels, p0 + TASK_PANELS);
            for (int p = p0; p < p1; p++) {
                int n0 = p * MMI8_NR, nvalid = std::min(MMI8_NR, N - n0);
                const int8_t* bp = packedB + p * panelSize;
                _Tp* Yp = Y + lay.colOfs(n0);
                for (int r = r0; r < r1; r += MMI8_MR) {
                    int mr = std::min(MMI8_MR, r1 - r);
                    // a lone row (token-by-token decoding) would otherwise be computed MR times
                    if (mr == 1)
                        gemmTile<1, CORR, XOR>(A + r * lda, lda, r, 1, bp, K, zp, colsum + n0,
                                               CORR ? corr + n0 : nullptr,
                                               CORR ? rowterm + r : nullptr,
                                               scale ? scale + n0 : nullptr,
                                               bias ? bias + n0 : nullptr, Yp, lay, nvalid);
                    else
                        gemmTile<MMI8_MR, CORR, XOR>(A + r * lda, lda, r, mr, bp, K, zp,
                                                     colsum + n0, CORR ? corr + n0 : nullptr,
                                                     CORR ? rowterm + r : nullptr,
                                                     scale ? scale + n0 : nullptr,
                                                     bias ? bias + n0 : nullptr, Yp, lay, nvalid);
                }
            }
        }
    });
}

template<bool XOR, typename _Tp>
static void gemmInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                          const int* corr, const int* rowterm, const float* scale,
                          const float* bias, _Tp* Y, const MatMulInt8OutLayout& lay,
                          int M, int N, int K, int zp)
{
    if (corr)
        gemmInt8Int32<true, XOR>(A, lda, packedB, colsum, corr, rowterm, scale, bias, Y, lay,
                                 M, N, K, zp);
    else
        gemmInt8Int32<false, XOR>(A, lda, packedB, colsum, corr, rowterm, scale, bias, Y, lay,
                                  M, N, K, zp);
}

bool matMulInt8PairedB()
{
    return MMI8_USE_AVX2 != 0;
}

void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, void* Y, const MatMulInt8OutLayout& lay,
                     int M, int N, int K, int a_zp, bool aIsU8,
                     const float* scale, const float* bias)
{
    // xor 0x80 turns uint8 A into the int8 operand (or back) and moves its zero point by 128
#if MMI8_USE_VNNI || MMI8_USE_AVX2
    const bool operandU8 = true;
#else
    const bool operandU8 = false;
#endif
    const bool flip = aIsU8 != operandU8;
    const int zp = !flip ? a_zp : aIsU8 ? a_zp - 128 : a_zp + 128;
    CV_Assert(!scale == !bias);
    if (scale) {
        if (flip)
            gemmInt8Int32<true>(A, lda, packedB, colsum, corr, rowterm, scale, bias, (float*)Y,
                                lay, M, N, K, zp);
        else
            gemmInt8Int32<false>(A, lda, packedB, colsum, corr, rowterm, scale, bias, (float*)Y,
                                 lay, M, N, K, zp);
    } else {
        if (flip)
            gemmInt8Int32<true>(A, lda, packedB, colsum, corr, rowterm, scale, bias, (int*)Y,
                                lay, M, N, K, zp);
        else
            gemmInt8Int32<false>(A, lda, packedB, colsum, corr, rowterm, scale, bias, (int*)Y,
                                 lay, M, N, K, zp);
    }
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::dnn
