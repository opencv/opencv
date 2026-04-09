// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#include "precomp.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

typedef void (*ReduceSumFunc)(const Mat& src, Mat& dst);
ReduceSumFunc getReduceCSumFunc(int sdepth, int ddepth);
ReduceSumFunc getReduceRSumFunc(int sdepth, int ddepth);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

// =====================================================================
//  Col reduce SUM (dim=1): sum each row into cn output values
// =====================================================================

#if (CV_SIMD || CV_SIMD_SCALABLE)

// --- uchar → int ---
// Uses u16 intermediate accumulator to halve the number of widen operations:
// u8→u16 per iteration, u16→u32 flush every 128 iterations (max u16 value: 128*255=32640 < 65535)
static void reduceColSum_8u32s(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int cols = srcmat.cols;
    const int width = cols * cn;

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            int* dst = dstmat.ptr<int>(y);

            if (cn == 1)
            {
#if defined(CV_NEON_DOT)
                // ARMv8.2 DOTPROD: vdotq_u32(acc, src, ones) — single instruction byte sum
                // u32 lane max 1020/iter, overflow at ~4.2M iter (67MB) — safe without flush
                const uint8x16_t ones = vdupq_n_u8(1);
                uint32x4_t v_sum = vdupq_n_u32(0);
                int x = 0;
                for (; x <= width - 16; x += 16)
                    v_sum = vdotq_u32(v_sum, vld1q_u8(src + x), ones);
                int total = (int)vaddvq_u32(v_sum);
                for (; x < width; x++)
                    total += (int)src[x];
                dst[0] = total;
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
                // AArch64 fallback: vpaddlq chain
                uint32x4_t v_sum = vdupq_n_u32(0);
                int x = 0;
                for (; x <= width - 16; x += 16)
                    v_sum = vaddq_u32(v_sum, vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + x))));
                int total = (int)vaddvq_u32(v_sum);
                for (; x < width; x++)
                    total += (int)src[x];
                dst[0] = total;
#elif CV_AVX2
                // Intel AVX2: _mm256_sad_epu8 — 32 bytes → 4×u64 sum, 1 cycle
                const __m256i zero = _mm256_setzero_si256();
                __m256i vsum = zero;
                int x = 0;
                for (; x <= width - 32; x += 32)
                    vsum = _mm256_add_epi64(vsum, _mm256_sad_epu8(
                        _mm256_loadu_si256((const __m256i*)(src + x)), zero));
                __m128i lo128 = _mm256_castsi256_si128(vsum);
                __m128i hi128 = _mm256_extracti128_si256(vsum, 1);
                __m128i s = _mm_add_epi64(lo128, hi128);
                s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
                int total = _mm_cvtsi128_si32(s);
                for (; x < width; x++)
                    total += (int)src[x];
                dst[0] = total;
#elif CV_SSE2
                // Intel SSE2: _mm_sad_epu8 — 16 bytes → 2×u64 sum
                const __m128i zero = _mm_setzero_si128();
                __m128i vsum = zero;
                int x = 0;
                for (; x <= width - 16; x += 16)
                    vsum = _mm_add_epi64(vsum, _mm_sad_epu8(
                        _mm_loadu_si128((const __m128i*)(src + x)), zero));
                __m128i s = _mm_add_epi64(vsum, _mm_unpackhi_epi64(vsum, vsum));
                int total = _mm_cvtsi128_si32(s);
                for (; x < width; x++)
                    total += (int)src[x];
                dst[0] = total;
#else
                const int vlanes8 = VTraits<v_uint8>::vlanes();
                v_uint32 v_sum = vx_setzero_u32();
                v_uint16 v_sum16 = vx_setzero_u16();
                int x = 0, batch = 0;
                const int flush_at = 128;

                for (; x <= width - vlanes8; x += vlanes8)
                {
                    v_uint16 lo, hi;
                    v_expand(vx_load(src + x), lo, hi);
                    v_sum16 = v_add(v_sum16, v_add(lo, hi));

                    if (++batch >= flush_at)
                    {
                        v_uint32 a, b;
                        v_expand(v_sum16, a, b);
                        v_sum = v_add(v_sum, v_add(a, b));
                        v_sum16 = vx_setzero_u16();
                        batch = 0;
                    }
                }
                {
                    v_uint32 a, b;
                    v_expand(v_sum16, a, b);
                    v_sum = v_add(v_sum, v_add(a, b));
                }

                int total = (int)v_reduce_sum(v_sum);
                for (; x < width; x++)
                    total += (int)src[x];
                dst[0] = total;
#endif
            }
            else if (cn == 4)
            {
#if defined(CV_NEON_DOT)
                // ARMv8.2 DOTPROD: vld4q_u8 deinterleave + vdotq_u32 per-channel
                // vld4q_u8 reads 16 pixels (64 bytes), deinterleaves into 4 × u8x16
                // vdotq_u32 sums each channel's 16 bytes into 4 partial u32 sums
                const uint8x16_t ones = vdupq_n_u8(1);
                uint32x4_t acc0 = vdupq_n_u32(0), acc1 = vdupq_n_u32(0);
                uint32x4_t acc2 = vdupq_n_u32(0), acc3 = vdupq_n_u32(0);
                int x = 0;
                for (; x <= cols - 16; x += 16)
                {
                    uint8x16x4_t v = vld4q_u8(src + x * 4);
                    acc0 = vdotq_u32(acc0, v.val[0], ones);
                    acc1 = vdotq_u32(acc1, v.val[1], ones);
                    acc2 = vdotq_u32(acc2, v.val[2], ones);
                    acc3 = vdotq_u32(acc3, v.val[3], ones);
                }
                dst[0] = (int)vaddvq_u32(acc0);
                dst[1] = (int)vaddvq_u32(acc1);
                dst[2] = (int)vaddvq_u32(acc2);
                dst[3] = (int)vaddvq_u32(acc3);
                for (; x < cols; x++)
                {
                    dst[0] += (int)src[x * 4];
                    dst[1] += (int)src[x * 4 + 1];
                    dst[2] += (int)src[x * 4 + 2];
                    dst[3] += (int)src[x * 4 + 3];
                }
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
                // AArch64 fallback: vld4q_u8 deinterleave + vaddlvq_u8 per-channel
                int sums[4] = {0, 0, 0, 0};
                int x = 0;
                for (; x <= cols - 16; x += 16)
                {
                    uint8x16x4_t v = vld4q_u8(src + x * 4);
                    sums[0] += (int)vaddlvq_u8(v.val[0]);
                    sums[1] += (int)vaddlvq_u8(v.val[1]);
                    sums[2] += (int)vaddlvq_u8(v.val[2]);
                    sums[3] += (int)vaddlvq_u8(v.val[3]);
                }
                dst[0] = sums[0]; dst[1] = sums[1];
                dst[2] = sums[2]; dst[3] = sums[3];
                for (; x < cols; x++)
                {
                    dst[0] += (int)src[x * 4];
                    dst[1] += (int)src[x * 4 + 1];
                    dst[2] += (int)src[x * 4 + 2];
                    dst[3] += (int)src[x * 4 + 3];
                }
#elif CV_AVX2
                // Intel AVX2: shuffle to group channels + _mm256_sad_epu8
                // 32 bytes = 8 pixels × 4 channels per iteration
                // Shuffle within each 128-bit lane:
                //   [B0G0R0A0 B1G1R1A1 B2G2R2A2 B3G3R3A3] → [B0B1B2B3 G0G1G2G3 R0R1R2R3 A0A1A2A3]
                // Then mask/shift to isolate channel pairs, SAD each against zero
                const __m256i zero = _mm256_setzero_si256();
                const __m256i shuf_mask = _mm256_setr_epi8(
                    0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15,
                    0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
                const __m256i mask_lo32 = _mm256_set1_epi64x(0x00000000FFFFFFFFLL);
                __m256i sum_br = zero, sum_ga = zero;
                int x = 0;
                for (; x <= cols - 8; x += 8)
                {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 4));
                    __m256i grouped = _mm256_shuffle_epi8(v, shuf_mask);
                    // 64-bit lane0: [B0B1B2B3 G0G1G2G3], lane1: [R0R1R2R3 A0A1A2A3]
                    __m256i br = _mm256_and_si256(grouped, mask_lo32);
                    __m256i ga = _mm256_srli_epi64(grouped, 32);
                    sum_br = _mm256_add_epi64(sum_br, _mm256_sad_epu8(br, zero));
                    sum_ga = _mm256_add_epi64(sum_ga, _mm256_sad_epu8(ga, zero));
                }
                // sum_br: [B_lo, R_lo, B_hi, R_hi], sum_ga: [G_lo, A_lo, G_hi, A_hi]
                __m128i br_lo = _mm256_castsi256_si128(sum_br);
                __m128i br_hi = _mm256_extracti128_si256(sum_br, 1);
                __m128i br_s = _mm_add_epi64(br_lo, br_hi);
                __m128i ga_lo = _mm256_castsi256_si128(sum_ga);
                __m128i ga_hi = _mm256_extracti128_si256(sum_ga, 1);
                __m128i ga_s = _mm_add_epi64(ga_lo, ga_hi);
                dst[0] = _mm_cvtsi128_si32(br_s);
                dst[1] = _mm_cvtsi128_si32(ga_s);
                dst[2] = _mm_cvtsi128_si32(_mm_srli_si128(br_s, 8));
                dst[3] = _mm_cvtsi128_si32(_mm_srli_si128(ga_s, 8));
                for (; x < cols; x++)
                {
                    dst[0] += (int)src[x * 4];
                    dst[1] += (int)src[x * 4 + 1];
                    dst[2] += (int)src[x * 4 + 2];
                    dst[3] += (int)src[x * 4 + 3];
                }
#elif CV_SSSE3
                // Intel SSSE3: _mm_shuffle_epi8 deinterleave + _mm_sad_epu8
                // 16 bytes = 4 pixels × 4 channels per iteration
                const __m128i zero = _mm_setzero_si128();
                const __m128i shuf_mask = _mm_setr_epi8(
                    0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
                const __m128i mask_lo32 = _mm_set1_epi64x(0x00000000FFFFFFFFLL);
                __m128i sum_br = zero, sum_ga = zero;
                int x = 0;
                for (; x <= cols - 4; x += 4)
                {
                    __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 4));
                    __m128i grouped = _mm_shuffle_epi8(v, shuf_mask);
                    __m128i br = _mm_and_si128(grouped, mask_lo32);
                    __m128i ga = _mm_srli_epi64(grouped, 32);
                    sum_br = _mm_add_epi64(sum_br, _mm_sad_epu8(br, zero));
                    sum_ga = _mm_add_epi64(sum_ga, _mm_sad_epu8(ga, zero));
                }
                dst[0] = _mm_cvtsi128_si32(sum_br);
                dst[1] = _mm_cvtsi128_si32(sum_ga);
                dst[2] = _mm_cvtsi128_si32(_mm_srli_si128(sum_br, 8));
                dst[3] = _mm_cvtsi128_si32(_mm_srli_si128(sum_ga, 8));
                for (; x < cols; x++)
                {
                    dst[0] += (int)src[x * 4];
                    dst[1] += (int)src[x * 4 + 1];
                    dst[2] += (int)src[x * 4 + 2];
                    dst[3] += (int)src[x * 4 + 3];
                }
#else
                const int vlanes8 = VTraits<v_uint8>::vlanes();
                v_uint32 vsum0 = vx_setzero_u32(), vsum1 = vx_setzero_u32();
                v_uint32 vsum2 = vx_setzero_u32(), vsum3 = vx_setzero_u32();
                v_uint16 vs16_0 = vx_setzero_u16(), vs16_1 = vx_setzero_u16();
                v_uint16 vs16_2 = vx_setzero_u16(), vs16_3 = vx_setzero_u16();
                int x = 0, batch = 0;
                const int flush_at = 128;

                for (; x <= cols - vlanes8; x += vlanes8)
                {
                    v_uint8 ch0, ch1, ch2, ch3;
                    v_load_deinterleave(src + x * 4, ch0, ch1, ch2, ch3);

                    v_uint16 lo, hi;
                    v_expand(ch0, lo, hi); vs16_0 = v_add(vs16_0, v_add(lo, hi));
                    v_expand(ch1, lo, hi); vs16_1 = v_add(vs16_1, v_add(lo, hi));
                    v_expand(ch2, lo, hi); vs16_2 = v_add(vs16_2, v_add(lo, hi));
                    v_expand(ch3, lo, hi); vs16_3 = v_add(vs16_3, v_add(lo, hi));

                    if (++batch >= flush_at)
                    {
                        v_uint32 a, b;
                        v_expand(vs16_0, a, b); vsum0 = v_add(vsum0, v_add(a, b));
                        v_expand(vs16_1, a, b); vsum1 = v_add(vsum1, v_add(a, b));
                        v_expand(vs16_2, a, b); vsum2 = v_add(vsum2, v_add(a, b));
                        v_expand(vs16_3, a, b); vsum3 = v_add(vsum3, v_add(a, b));
                        vs16_0 = vx_setzero_u16(); vs16_1 = vx_setzero_u16();
                        vs16_2 = vx_setzero_u16(); vs16_3 = vx_setzero_u16();
                        batch = 0;
                    }
                }
                {
                    v_uint32 a, b;
                    v_expand(vs16_0, a, b); vsum0 = v_add(vsum0, v_add(a, b));
                    v_expand(vs16_1, a, b); vsum1 = v_add(vsum1, v_add(a, b));
                    v_expand(vs16_2, a, b); vsum2 = v_add(vsum2, v_add(a, b));
                    v_expand(vs16_3, a, b); vsum3 = v_add(vsum3, v_add(a, b));
                }

                dst[0] = (int)v_reduce_sum(vsum0);
                dst[1] = (int)v_reduce_sum(vsum1);
                dst[2] = (int)v_reduce_sum(vsum2);
                dst[3] = (int)v_reduce_sum(vsum3);

                for (; x < cols; x++)
                {
                    dst[0] += (int)src[x * 4];
                    dst[1] += (int)src[x * 4 + 1];
                    dst[2] += (int)src[x * 4 + 2];
                    dst[3] += (int)src[x * 4 + 3];
                }
#endif
            }
            else
            {
                // generic cn: scalar fallback
                for (int c = 0; c < cn; c++)
                    dst[c] = (int)src[c];
                for (int x = cn; x < width; x += cn)
                    for (int c = 0; c < cn; c++)
                        dst[c] += (int)src[x + c];
            }
        }
    };

    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}

// --- uchar → float ---
static void reduceColSum_8u32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int cols = srcmat.cols;
    const int width = cols * cn;

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const uchar* src = srcmat.ptr<uchar>(y);
            float* dst = dstmat.ptr<float>(y);

            // compute in int, then convert to float
            AutoBuffer<int> ibuf(cn);
            int* sums = ibuf.data();
            for (int c = 0; c < cn; c++)
                sums[c] = 0;

            if (cn == 1)
            {
#if defined(CV_NEON_DOT)
                const uint8x16_t ones = vdupq_n_u8(1);
                uint32x4_t v_sum = vdupq_n_u32(0);
                int x = 0;
                for (; x <= width - 16; x += 16)
                    v_sum = vdotq_u32(v_sum, vld1q_u8(src + x), ones);
                sums[0] = (int)vaddvq_u32(v_sum);
                for (; x < width; x++)
                    sums[0] += (int)src[x];
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
                uint32x4_t v_sum = vdupq_n_u32(0);
                int x = 0;
                for (; x <= width - 16; x += 16)
                    v_sum = vaddq_u32(v_sum, vpaddlq_u16(vpaddlq_u8(vld1q_u8(src + x))));
                sums[0] = (int)vaddvq_u32(v_sum);
                for (; x < width; x++)
                    sums[0] += (int)src[x];
#elif CV_AVX2
                const __m256i zero = _mm256_setzero_si256();
                __m256i vsum = zero;
                int x = 0;
                for (; x <= width - 32; x += 32)
                    vsum = _mm256_add_epi64(vsum, _mm256_sad_epu8(
                        _mm256_loadu_si256((const __m256i*)(src + x)), zero));
                __m128i lo128 = _mm256_castsi256_si128(vsum);
                __m128i hi128 = _mm256_extracti128_si256(vsum, 1);
                __m128i s = _mm_add_epi64(lo128, hi128);
                s = _mm_add_epi64(s, _mm_unpackhi_epi64(s, s));
                sums[0] = _mm_cvtsi128_si32(s);
                for (; x < width; x++)
                    sums[0] += (int)src[x];
#elif CV_SSE2
                const __m128i zero = _mm_setzero_si128();
                __m128i vsum = zero;
                int x = 0;
                for (; x <= width - 16; x += 16)
                    vsum = _mm_add_epi64(vsum, _mm_sad_epu8(
                        _mm_loadu_si128((const __m128i*)(src + x)), zero));
                __m128i s = _mm_add_epi64(vsum, _mm_unpackhi_epi64(vsum, vsum));
                sums[0] = _mm_cvtsi128_si32(s);
                for (; x < width; x++)
                    sums[0] += (int)src[x];
#else
                const int vlanes8 = VTraits<v_uint8>::vlanes();
                v_uint32 v_sum = vx_setzero_u32();
                v_uint16 v_sum16 = vx_setzero_u16();
                int x = 0, batch = 0;

                for (; x <= width - vlanes8; x += vlanes8)
                {
                    v_uint16 lo, hi;
                    v_expand(vx_load(src + x), lo, hi);
                    v_sum16 = v_add(v_sum16, v_add(lo, hi));
                    if (++batch >= 128)
                    {
                        v_uint32 a, b;
                        v_expand(v_sum16, a, b);
                        v_sum = v_add(v_sum, v_add(a, b));
                        v_sum16 = vx_setzero_u16();
                        batch = 0;
                    }
                }
                v_uint32 a, b;
                v_expand(v_sum16, a, b);
                v_sum = v_add(v_sum, v_add(a, b));

                sums[0] = (int)v_reduce_sum(v_sum);
                for (; x < width; x++)
                    sums[0] += (int)src[x];
#endif
            }
            else if (cn == 4)
            {
#if defined(CV_NEON_DOT)
                const uint8x16_t ones = vdupq_n_u8(1);
                uint32x4_t acc0 = vdupq_n_u32(0), acc1 = vdupq_n_u32(0);
                uint32x4_t acc2 = vdupq_n_u32(0), acc3 = vdupq_n_u32(0);
                int x = 0;
                for (; x <= cols - 16; x += 16)
                {
                    uint8x16x4_t v = vld4q_u8(src + x * 4);
                    acc0 = vdotq_u32(acc0, v.val[0], ones);
                    acc1 = vdotq_u32(acc1, v.val[1], ones);
                    acc2 = vdotq_u32(acc2, v.val[2], ones);
                    acc3 = vdotq_u32(acc3, v.val[3], ones);
                }
                sums[0] = (int)vaddvq_u32(acc0);
                sums[1] = (int)vaddvq_u32(acc1);
                sums[2] = (int)vaddvq_u32(acc2);
                sums[3] = (int)vaddvq_u32(acc3);
                for (; x < cols; x++)
                {
                    sums[0] += (int)src[x * 4];
                    sums[1] += (int)src[x * 4 + 1];
                    sums[2] += (int)src[x * 4 + 2];
                    sums[3] += (int)src[x * 4 + 3];
                }
#elif CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
                int x = 0;
                for (; x <= cols - 16; x += 16)
                {
                    uint8x16x4_t v = vld4q_u8(src + x * 4);
                    sums[0] += (int)vaddlvq_u8(v.val[0]);
                    sums[1] += (int)vaddlvq_u8(v.val[1]);
                    sums[2] += (int)vaddlvq_u8(v.val[2]);
                    sums[3] += (int)vaddlvq_u8(v.val[3]);
                }
                for (; x < cols; x++)
                {
                    sums[0] += (int)src[x * 4];
                    sums[1] += (int)src[x * 4 + 1];
                    sums[2] += (int)src[x * 4 + 2];
                    sums[3] += (int)src[x * 4 + 3];
                }
#elif CV_AVX2
                const __m256i zero = _mm256_setzero_si256();
                const __m256i shuf_mask = _mm256_setr_epi8(
                    0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15,
                    0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
                const __m256i mask_lo32 = _mm256_set1_epi64x(0x00000000FFFFFFFFLL);
                __m256i sum_br = zero, sum_ga = zero;
                int x = 0;
                for (; x <= cols - 8; x += 8)
                {
                    __m256i v = _mm256_loadu_si256((const __m256i*)(src + x * 4));
                    __m256i grouped = _mm256_shuffle_epi8(v, shuf_mask);
                    __m256i br = _mm256_and_si256(grouped, mask_lo32);
                    __m256i ga = _mm256_srli_epi64(grouped, 32);
                    sum_br = _mm256_add_epi64(sum_br, _mm256_sad_epu8(br, zero));
                    sum_ga = _mm256_add_epi64(sum_ga, _mm256_sad_epu8(ga, zero));
                }
                __m128i br_lo = _mm256_castsi256_si128(sum_br);
                __m128i br_hi = _mm256_extracti128_si256(sum_br, 1);
                __m128i br_s = _mm_add_epi64(br_lo, br_hi);
                __m128i ga_lo = _mm256_castsi256_si128(sum_ga);
                __m128i ga_hi = _mm256_extracti128_si256(sum_ga, 1);
                __m128i ga_s = _mm_add_epi64(ga_lo, ga_hi);
                sums[0] = _mm_cvtsi128_si32(br_s);
                sums[1] = _mm_cvtsi128_si32(ga_s);
                sums[2] = _mm_cvtsi128_si32(_mm_srli_si128(br_s, 8));
                sums[3] = _mm_cvtsi128_si32(_mm_srli_si128(ga_s, 8));
                for (; x < cols; x++)
                {
                    sums[0] += (int)src[x * 4];
                    sums[1] += (int)src[x * 4 + 1];
                    sums[2] += (int)src[x * 4 + 2];
                    sums[3] += (int)src[x * 4 + 3];
                }
#elif CV_SSSE3
                // Intel SSSE3: _mm_shuffle_epi8 deinterleave + _mm_sad_epu8
                const __m128i zero = _mm_setzero_si128();
                const __m128i shuf_mask = _mm_setr_epi8(
                    0,4,8,12, 1,5,9,13, 2,6,10,14, 3,7,11,15);
                const __m128i mask_lo32 = _mm_set1_epi64x(0x00000000FFFFFFFFLL);
                __m128i sum_br = zero, sum_ga = zero;
                int x = 0;
                for (; x <= cols - 4; x += 4)
                {
                    __m128i v = _mm_loadu_si128((const __m128i*)(src + x * 4));
                    __m128i grouped = _mm_shuffle_epi8(v, shuf_mask);
                    __m128i br = _mm_and_si128(grouped, mask_lo32);
                    __m128i ga = _mm_srli_epi64(grouped, 32);
                    sum_br = _mm_add_epi64(sum_br, _mm_sad_epu8(br, zero));
                    sum_ga = _mm_add_epi64(sum_ga, _mm_sad_epu8(ga, zero));
                }
                sums[0] = _mm_cvtsi128_si32(sum_br);
                sums[1] = _mm_cvtsi128_si32(sum_ga);
                sums[2] = _mm_cvtsi128_si32(_mm_srli_si128(sum_br, 8));
                sums[3] = _mm_cvtsi128_si32(_mm_srli_si128(sum_ga, 8));
                for (; x < cols; x++)
                {
                    sums[0] += (int)src[x * 4];
                    sums[1] += (int)src[x * 4 + 1];
                    sums[2] += (int)src[x * 4 + 2];
                    sums[3] += (int)src[x * 4 + 3];
                }
#else
                for (int x = 0; x < width; x++)
                    sums[x % cn] += (int)src[x];
#endif
            }
            else
            {
                for (int x = 0; x < width; x++)
                    sums[x % cn] += (int)src[x];
            }
            for (int c = 0; c < cn; c++)
                dst[c] = (float)sums[c];
        }
    };
    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}

// --- ushort → float ---
static void reduceColSum_16u32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int width = srcmat.cols * cn;
    const int vlanes16 = VTraits<v_uint16>::vlanes();

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const ushort* src = srcmat.ptr<ushort>(y);
            float* dst = dstmat.ptr<float>(y);

            if (cn == 1)
            {
                v_uint32 v_sum = vx_setzero_u32();
                int x = 0;
                for (; x <= width - vlanes16; x += vlanes16)
                {
                    v_uint32 lo, hi;
                    v_expand(vx_load(src + x), lo, hi);
                    v_sum = v_add(v_sum, v_add(lo, hi));
                }
                dst[0] = (float)v_reduce_sum(v_sum);
                for (; x < width; x++)
                    dst[0] += (float)src[x];
            }
            else
            {
                for (int c = 0; c < cn; c++)
                    dst[c] = 0;
                for (int x = 0; x < width; x++)
                    dst[x % cn] += (float)src[x];
            }
        }
    };
    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}

// --- short → float ---
static void reduceColSum_16s32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int width = srcmat.cols * cn;
    const int vlanes16 = VTraits<v_int16>::vlanes();

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const short* src = srcmat.ptr<short>(y);
            float* dst = dstmat.ptr<float>(y);

            if (cn == 1)
            {
                v_int32 v_sum = vx_setzero_s32();
                int x = 0;
                for (; x <= width - vlanes16; x += vlanes16)
                {
                    v_int32 lo, hi;
                    v_expand(vx_load(src + x), lo, hi);
                    v_sum = v_add(v_sum, v_add(lo, hi));
                }
                dst[0] = (float)v_reduce_sum(v_sum);
                for (; x < width; x++)
                    dst[0] += (float)src[x];
            }
            else
            {
                for (int c = 0; c < cn; c++)
                    dst[c] = 0;
                for (int x = 0; x < width; x++)
                    dst[x % cn] += (float)src[x];
            }
        }
    };
    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}

// --- float → float ---
static void reduceColSum_32f32f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int width = srcmat.cols * cn;
    const int vlanes32 = VTraits<v_float32>::vlanes();

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            float* dst = dstmat.ptr<float>(y);

            if (cn == 1)
            {
                v_float32 v_sum = vx_setzero_f32();
                int x = 0;
                for (; x <= width - vlanes32; x += vlanes32)
                    v_sum = v_add(v_sum, vx_load(src + x));

                dst[0] = v_reduce_sum(v_sum);
                for (; x < width; x++)
                    dst[0] += src[x];
            }
            else
            {
                for (int c = 0; c < cn; c++)
                    dst[c] = 0;
                for (int x = 0; x < width; x++)
                    dst[x % cn] += src[x];
            }
        }
    };
    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}

#if CV_SIMD_64F
// --- float → double ---
static void reduceColSum_32f64f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int width = srcmat.cols * cn;
    const int vlanes32 = VTraits<v_float32>::vlanes();

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const float* src = srcmat.ptr<float>(y);
            double* dst = dstmat.ptr<double>(y);

            if (cn == 1)
            {
                v_float64 v_sum0 = vx_setzero_f64();
                v_float64 v_sum1 = vx_setzero_f64();
                int x = 0;
                for (; x <= width - vlanes32; x += vlanes32)
                {
                    v_float32 v = vx_load(src + x);
                    v_sum0 = v_add(v_sum0, v_cvt_f64(v));
                    v_sum1 = v_add(v_sum1, v_cvt_f64_high(v));
                }
                v_float64 v_total = v_add(v_sum0, v_sum1);

                double total = v_reduce_sum(v_total);
                for (; x < width; x++)
                    total += (double)src[x];
                dst[0] = total;
            }
            else
            {
                for (int c = 0; c < cn; c++)
                    dst[c] = 0;
                for (int x = 0; x < width; x++)
                    dst[x % cn] += (double)src[x];
            }
        }
    };
    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}

// --- double → double ---
static void reduceColSum_64f64f(const Mat& srcmat, Mat& dstmat)
{
    const int cn = srcmat.channels();
    const int width = srcmat.cols * cn;
    const int vlanes64 = VTraits<v_float64>::vlanes();

    auto body = [&](const Range& range) {
        for (int y = range.start; y < range.end; y++)
        {
            const double* src = srcmat.ptr<double>(y);
            double* dst = dstmat.ptr<double>(y);

            if (cn == 1)
            {
                v_float64 v_sum = vx_setzero_f64();
                int x = 0;
                for (; x <= width - vlanes64; x += vlanes64)
                    v_sum = v_add(v_sum, vx_load(src + x));

                double total = v_reduce_sum(v_sum);
                for (; x < width; x++)
                    total += src[x];
                dst[0] = total;
            }
            else
            {
                for (int c = 0; c < cn; c++)
                    dst[c] = 0;
                for (int x = 0; x < width; x++)
                    dst[x % cn] += src[x];
            }
        }
    };
    parallel_for_(Range(0, srcmat.rows), body);
    v_cleanup();
}
#endif // CV_SIMD_64F

// =====================================================================
//  Row reduce SUM (dim=0): sum each column across all rows
// =====================================================================

// --- uchar → int ---
// Uses u16 intermediate accumulator for vertical sum: accumulate u8→u16 per row,
// flush u16→u32 every 256 rows (max u16 value: 256*255=65280 < 65535)
static void reduceRowSum_8u32s(const Mat& srcmat, Mat& dstmat)
{
    const int width_cn = srcmat.cols * srcmat.channels();
    const int height = srcmat.rows;
    const int vlanes16 = VTraits<v_uint16>::vlanes();
    const int vlanes32 = VTraits<v_int32>::vlanes();

    auto body = [&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        const int len = end - start;

        AutoBuffer<int> buf32_storage(len);
        AutoBuffer<ushort> buf16_storage(len);
        int* buf32 = buf32_storage.data();
        ushort* buf16 = buf16_storage.data();

        // init from first row
        const uchar* src0 = srcmat.ptr<uchar>(0) + start;
        for (int i = 0; i < len; i++)
            buf32[i] = (int)src0[i];

        if (height <= 1)
        {
            int* dst = dstmat.ptr<int>(0) + start;
            for (int i = 0; i < len; i++)
                dst[i] = buf32[i];
            return;
        }

        memset(buf16, 0, len * sizeof(ushort));
        int flush_count = 0;

        for (int row = 1; row < height; row++)
        {
            const uchar* src = srcmat.ptr<uchar>(row) + start;
            int i = 0;

#if CV_NEON && (defined(__aarch64__) || defined(_M_ARM64))
            // AArch64: vaddw_u8 — single-instruction widening add (u16 += u8)
            for (; i <= len - 16; i += 16)
            {
                uint8x16_t v = vld1q_u8(src + i);
                uint16x8_t acc_lo = vld1q_u16(buf16 + i);
                uint16x8_t acc_hi = vld1q_u16(buf16 + i + 8);
                vst1q_u16(buf16 + i,     vaddw_u8(acc_lo, vget_low_u8(v)));
                vst1q_u16(buf16 + i + 8, vaddw_high_u8(acc_hi, v));
            }
#else
            const int vlanes8 = VTraits<v_uint8>::vlanes();
            for (; i <= len - vlanes8; i += vlanes8)
            {
                v_uint16 lo, hi;
                v_expand(vx_load(src + i), lo, hi);
                v_store(buf16 + i, v_add(vx_load(buf16 + i), lo));
                v_store(buf16 + i + vlanes16, v_add(vx_load(buf16 + i + vlanes16), hi));
            }
#endif
            for (; i < len; i++)
                buf16[i] += (ushort)src[i];

            if (++flush_count >= 256 || row == height - 1)
            {
                // flush u16 → u32
                int j = 0;
                for (; j <= len - vlanes16; j += vlanes16)
                {
                    v_uint32 lo, hi;
                    v_expand(vx_load(buf16 + j), lo, hi);
                    v_store(buf32 + j, v_add(vx_load(buf32 + j), v_reinterpret_as_s32(lo)));
                    v_store(buf32 + j + vlanes32, v_add(vx_load(buf32 + j + vlanes32), v_reinterpret_as_s32(hi)));
                }
                for (; j < len; j++)
                    buf32[j] += (int)buf16[j];
                memset(buf16, 0, len * sizeof(ushort));
                flush_count = 0;
            }
        }

        int* dst = dstmat.ptr<int>(0) + start;
        for (int i = 0; i < len; i++)
            dst[i] = buf32[i];
    };

    parallel_for_(Range(0, width_cn), body, width_cn * CV_ELEM_SIZE(srcmat.depth()) / 64);
    v_cleanup();
}

// --- uchar → float ---
static void reduceRowSum_8u32f(const Mat& srcmat, Mat& dstmat)
{
    // compute in int, then convert to float
    Mat temp(dstmat.rows, dstmat.cols, CV_32SC(srcmat.channels()));
    reduceRowSum_8u32s(srcmat, temp);
    temp.convertTo(dstmat, dstmat.type());
}

// --- ushort → float ---
static void reduceRowSum_16u32f(const Mat& srcmat, Mat& dstmat)
{
    const int width_cn = srcmat.cols * srcmat.channels();
    const int height = srcmat.rows;
    const int vlanes16 = VTraits<v_uint16>::vlanes();
    const int vlanes32 = VTraits<v_uint32>::vlanes();

    auto body = [&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        const int len = end - start;

        AutoBuffer<int> buf32_storage(len);
        int* buf32 = buf32_storage.data();

        const ushort* src0 = srcmat.ptr<ushort>(0) + start;
        for (int i = 0; i < len; i++)
            buf32[i] = (int)src0[i];

        for (int row = 1; row < height; row++)
        {
            const ushort* src = srcmat.ptr<ushort>(row) + start;
            int i = 0;
            for (; i <= len - vlanes16; i += vlanes16)
            {
                v_uint32 lo, hi;
                v_expand(vx_load(src + i), lo, hi);
                v_store(buf32 + i, v_add(vx_load(buf32 + i), v_reinterpret_as_s32(lo)));
                v_store(buf32 + i + vlanes32, v_add(vx_load(buf32 + i + vlanes32), v_reinterpret_as_s32(hi)));
            }
            for (; i < len; i++)
                buf32[i] += (int)src[i];
        }

        float* dst = dstmat.ptr<float>(0) + start;
        for (int i = 0; i < len; i++)
            dst[i] = (float)buf32[i];
    };

    parallel_for_(Range(0, width_cn), body, width_cn * CV_ELEM_SIZE(srcmat.depth()) / 64);
    v_cleanup();
}

// --- short → float ---
static void reduceRowSum_16s32f(const Mat& srcmat, Mat& dstmat)
{
    const int width_cn = srcmat.cols * srcmat.channels();
    const int height = srcmat.rows;
    const int vlanes16 = VTraits<v_int16>::vlanes();
    const int vlanes32 = VTraits<v_int32>::vlanes();

    auto body = [&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        const int len = end - start;

        AutoBuffer<int> buf32_storage(len);
        int* buf32 = buf32_storage.data();

        const short* src0 = srcmat.ptr<short>(0) + start;
        for (int i = 0; i < len; i++)
            buf32[i] = (int)src0[i];

        for (int row = 1; row < height; row++)
        {
            const short* src = srcmat.ptr<short>(row) + start;
            int i = 0;
            for (; i <= len - vlanes16; i += vlanes16)
            {
                v_int32 lo, hi;
                v_expand(vx_load(src + i), lo, hi);
                v_store(buf32 + i, v_add(vx_load(buf32 + i), lo));
                v_store(buf32 + i + vlanes32, v_add(vx_load(buf32 + i + vlanes32), hi));
            }
            for (; i < len; i++)
                buf32[i] += (int)src[i];
        }

        float* dst = dstmat.ptr<float>(0) + start;
        for (int i = 0; i < len; i++)
            dst[i] = (float)buf32[i];
    };

    parallel_for_(Range(0, width_cn), body, width_cn * CV_ELEM_SIZE(srcmat.depth()) / 64);
    v_cleanup();
}

// --- float → float ---
static void reduceRowSum_32f32f(const Mat& srcmat, Mat& dstmat)
{
    const int width_cn = srcmat.cols * srcmat.channels();
    const int height = srcmat.rows;
    const int vlanes = VTraits<v_float32>::vlanes();

    auto body = [&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        const int len = end - start;

        AutoBuffer<float> buf_storage(len);
        float* buf = buf_storage.data();

        const float* src0 = srcmat.ptr<float>(0) + start;
        memcpy(buf, src0, len * sizeof(float));

        for (int row = 1; row < height; row++)
        {
            const float* src = srcmat.ptr<float>(row) + start;
            int i = 0;
            for (; i <= len - vlanes; i += vlanes)
                v_store(buf + i, v_add(vx_load(buf + i), vx_load(src + i)));
            for (; i < len; i++)
                buf[i] += src[i];
        }

        float* dst = dstmat.ptr<float>(0) + start;
        memcpy(dst, buf, len * sizeof(float));
    };

    parallel_for_(Range(0, width_cn), body, width_cn * CV_ELEM_SIZE(srcmat.depth()) / 64);
    v_cleanup();
}

#if CV_SIMD_64F
// --- float → double ---
static void reduceRowSum_32f64f(const Mat& srcmat, Mat& dstmat)
{
    const int width_cn = srcmat.cols * srcmat.channels();
    const int height = srcmat.rows;
    const int vlanes32 = VTraits<v_float32>::vlanes();
    const int vlanes64 = VTraits<v_float64>::vlanes();

    auto body = [&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        const int len = end - start;

        AutoBuffer<double> buf_storage(len);
        double* buf = buf_storage.data();

        const float* src0 = srcmat.ptr<float>(0) + start;
        for (int i = 0; i < len; i++)
            buf[i] = (double)src0[i];

        for (int row = 1; row < height; row++)
        {
            const float* src = srcmat.ptr<float>(row) + start;
            int i = 0;
            for (; i <= len - vlanes32; i += vlanes32)
            {
                v_float32 v = vx_load(src + i);
                v_store(buf + i, v_add(vx_load(buf + i), v_cvt_f64(v)));
                v_store(buf + i + vlanes64, v_add(vx_load(buf + i + vlanes64), v_cvt_f64_high(v)));
            }
            for (; i < len; i++)
                buf[i] += (double)src[i];
        }

        double* dst = dstmat.ptr<double>(0) + start;
        memcpy(dst, buf, len * sizeof(double));
    };

    parallel_for_(Range(0, width_cn), body, width_cn * CV_ELEM_SIZE(srcmat.depth()) / 64);
    v_cleanup();
}

// --- double → double ---
static void reduceRowSum_64f64f(const Mat& srcmat, Mat& dstmat)
{
    const int width_cn = srcmat.cols * srcmat.channels();
    const int height = srcmat.rows;
    const int vlanes = VTraits<v_float64>::vlanes();

    auto body = [&](const Range& range) {
        const int start = range.start;
        const int end = range.end;
        const int len = end - start;

        AutoBuffer<double> buf_storage(len);
        double* buf = buf_storage.data();

        const double* src0 = srcmat.ptr<double>(0) + start;
        memcpy(buf, src0, len * sizeof(double));

        for (int row = 1; row < height; row++)
        {
            const double* src = srcmat.ptr<double>(row) + start;
            int i = 0;
            for (; i <= len - vlanes; i += vlanes)
                v_store(buf + i, v_add(vx_load(buf + i), vx_load(src + i)));
            for (; i < len; i++)
                buf[i] += src[i];
        }

        double* dst = dstmat.ptr<double>(0) + start;
        memcpy(dst, buf, len * sizeof(double));
    };

    parallel_for_(Range(0, width_cn), body, width_cn * CV_ELEM_SIZE(srcmat.depth()) / 64);
    v_cleanup();
}
#endif // CV_SIMD_64F

#endif // CV_SIMD || CV_SIMD_SCALABLE

// =====================================================================
//  Dispatchers
// =====================================================================

ReduceSumFunc getReduceCSumFunc(int sdepth, int ddepth)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if (sdepth == CV_8U && ddepth == CV_32S) return reduceColSum_8u32s;
    if (sdepth == CV_8U && ddepth == CV_32F) return reduceColSum_8u32f;
    if (sdepth == CV_16U && ddepth == CV_32F) return reduceColSum_16u32f;
    if (sdepth == CV_16S && ddepth == CV_32F) return reduceColSum_16s32f;
    if (sdepth == CV_32F && ddepth == CV_32F) return reduceColSum_32f32f;
#if CV_SIMD_64F
    if (sdepth == CV_32F && ddepth == CV_64F) return reduceColSum_32f64f;
    if (sdepth == CV_64F && ddepth == CV_64F) return reduceColSum_64f64f;
#endif
#else
    CV_UNUSED(sdepth);
    CV_UNUSED(ddepth);
#endif
    return nullptr;
}

ReduceSumFunc getReduceRSumFunc(int sdepth, int ddepth)
{
#if (CV_SIMD || CV_SIMD_SCALABLE)
    if (sdepth == CV_8U && ddepth == CV_32S) return reduceRowSum_8u32s;
    if (sdepth == CV_8U && ddepth == CV_32F) return reduceRowSum_8u32f;
    if (sdepth == CV_16U && ddepth == CV_32F) return reduceRowSum_16u32f;
    if (sdepth == CV_16S && ddepth == CV_32F) return reduceRowSum_16s32f;
    if (sdepth == CV_32F && ddepth == CV_32F) return reduceRowSum_32f32f;
#if CV_SIMD_64F
    if (sdepth == CV_32F && ddepth == CV_64F) return reduceRowSum_32f64f;
    if (sdepth == CV_64F && ddepth == CV_64F) return reduceRowSum_64f64f;
#endif
#else
    CV_UNUSED(sdepth);
    CV_UNUSED(ddepth);
#endif
    return nullptr;
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
