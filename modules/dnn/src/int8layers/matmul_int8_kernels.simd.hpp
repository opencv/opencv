// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
namespace dnn {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, int* Y, size_t ldy,
                     int M, int N, int K, int a_zp, bool aIsU8);

#if !defined(CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY)

static constexpr int MMI8_MR = 4, MMI8_NR = 16, MMI8_KR = 4;
static constexpr int MMI8_PSTEP = MMI8_NR * MMI8_KR; // packed bytes per group of KR rows

// dpbusd takes 256-bit registers, so only a build whose widest vectors are 256 bits uses it
#if CV_SIMD256 && !CV_SIMD512 && CV_AVX_VNNI
#define MMI8_USE_VNNI 1
#else
#define MMI8_USE_VNNI 0
#endif

#if CV_SIMD
// nlanes, not vlanes(): the register arrays below need a compile-time count
static constexpr int MMI8_NV = MMI8_PSTEP / VTraits<v_int8>::nlanes;

// lane i accumulates the products of bytes 4i..4i+3; on AVX-VNNI a is read as uint8
static inline v_int32 dot4(const v_int8& a, const v_int8& b, const v_int32& c)
{
#if MMI8_USE_VNNI
    return v_int32(_mm256_dpbusd_epi32(c.val, a.val, b.val));
#else
    return v_dotprod_expand(a, b, c);
#endif
}

static CV_ALWAYS_INLINE void macStep(v_int32 (&acc)[MMI8_MR][MMI8_NV], const int8_t* bp,
                                     const int (&a4)[MMI8_MR])
{
    v_int8 b[MMI8_NV];
    for (int j = 0; j < MMI8_NV; j++)
        b[j] = vx_load(bp + j * VTraits<v_int8>::vlanes());
    for (int i = 0; i < MMI8_MR; i++) {
        v_int8 a = v_reinterpret_as_s8(vx_setall_s32(a4[i]));
        for (int j = 0; j < MMI8_NV; j++)
            acc[i][j] = dot4(a, b[j], acc[i][j]);
    }
}
#endif

// rows [r, r+mr) x one panel; rows past mr repeat the last row and are not stored
template<bool CORR>
static void gemmTile(const uchar* A, size_t lda, int mr, const int8_t* bp, int K, int xor4,
                     int zp, const int* colsum, const int* corr, const int* rt,
                     int* Y, size_t ldy, int nvalid)
{
    const uchar* a[MMI8_MR];
    for (int i = 0; i < MMI8_MR; i++)
        a[i] = A + std::min(i, mr - 1) * lda;
#if CV_SIMD
    const int VLI = VTraits<v_int32>::vlanes();
    v_int32 acc[MMI8_MR][MMI8_NV];
    for (int i = 0; i < MMI8_MR; i++)
        for (int j = 0; j < MMI8_NV; j++)
            acc[i][j] = vx_setzero_s32();

    int k = 0;
    for (; k + MMI8_KR <= K; k += MMI8_KR, bp += MMI8_PSTEP) {
        int a4[MMI8_MR];
        for (int i = 0; i < MMI8_MR; i++) {
            memcpy(&a4[i], a[i] + k, MMI8_KR);
            a4[i] ^= xor4;
        }
        macStep(acc, bp, a4);
    }
    if (k < K) {
        // the padded weights are zero, so the padded bytes of A may hold anything
        int a4[MMI8_MR] = {};
        for (int i = 0; i < MMI8_MR; i++) {
            memcpy(&a4[i], a[i] + k, K - k);
            a4[i] ^= xor4;
        }
        macStep(acc, bp, a4);
    }

    v_int32 vzp = vx_setall_s32(-zp), bias[MMI8_NV];
    for (int j = 0; j < MMI8_NV; j++)
        bias[j] = v_mul(vzp, vx_load(colsum + j * VLI));
    for (int i = 0; i < mr; i++) {
        int buf[MMI8_NR];
        int* y = nvalid == MMI8_NR ? Y + i * ldy : buf;
        for (int j = 0; j < MMI8_NV; j++) {
            v_int32 s = v_add(acc[i][j], bias[j]);
            if constexpr (CORR)
                s = v_muladd(vx_load(corr + j * VLI), vx_setall_s32(rt[i]), s);
            v_store(y + j * VLI, s);
        }
        if (y == buf)
            for (int j = 0; j < nvalid; j++)
                Y[i * ldy + j] = buf[j];
    }
#else
    for (int i = 0; i < mr; i++) {
        for (int j = 0; j < nvalid; j++) {
            int s = 0;
            for (int k = 0; k < K; k++)
                s += (int)(schar)(a[i][k] ^ (xor4 & 255)) *
                     bp[(k / MMI8_KR) * MMI8_PSTEP + j * MMI8_KR + k % MMI8_KR];
            s -= zp * colsum[j];
            if constexpr (CORR)
                s += corr[j] * rt[i];
            Y[i * ldy + j] = s;
        }
    }
#endif
}

template<bool CORR>
static void gemmInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                          const int* corr, const int* rowterm, int* Y, size_t ldy,
                          int M, int N, int K, int xor4, int zp)
{
    constexpr int TASK_ROWS = 64, TASK_PANELS = 2;
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
                for (int r = r0; r < r1; r += MMI8_MR)
                    gemmTile<CORR>(A + r * lda, lda, std::min(MMI8_MR, r1 - r), bp, K, xor4, zp,
                                   colsum + n0, CORR ? corr + n0 : nullptr,
                                   CORR ? rowterm + r : nullptr, Y + r * ldy + n0, ldy, nvalid);
            }
        }
    });
}

void matMulInt8Int32(const uchar* A, size_t lda, const int8_t* packedB, const int* colsum,
                     const int* corr, const int* rowterm, int* Y, size_t ldy,
                     int M, int N, int K, int a_zp, bool aIsU8)
{
    // xor 0x80 turns uint8 A into the int8 operand (or back) and moves its zero point by 128
#if MMI8_USE_VNNI
    const bool operandU8 = true;
#else
    const bool operandU8 = false;
#endif
    const int xor4 = aIsU8 == operandU8 ? 0 : (int)0x80808080u;
    const int zp = aIsU8 == operandU8 ? a_zp : aIsU8 ? a_zp - 128 : a_zp + 128;
    if (corr)
        gemmInt8Int32<true>(A, lda, packedB, colsum, corr, rowterm, Y, ldy, M, N, K, xor4, zp);
    else
        gemmInt8Int32<false>(A, lda, packedB, colsum, corr, rowterm, Y, ldy, M, N, K, xor4, zp);
}

#endif // !CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // namespace cv::dnn
