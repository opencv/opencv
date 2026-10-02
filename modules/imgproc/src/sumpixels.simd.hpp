/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2020 Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "opencv2/core/hal/intrin.hpp"

#if CV_AVX512_SKX
#include "sumpixels.avx512_skx.hpp"
#endif

namespace cv { namespace hal {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// forward declarations
bool integral_SIMD(
        int depth, int sdepth, int sqdepth,
        const uchar* src, size_t srcstep,
        uchar* sum, size_t sumstep,
        uchar* sqsum, size_t sqsumstep,
        uchar* tilted, size_t tstep,
        int width, int height, int cn);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY
namespace {

template <typename T, typename ST, typename QT>
struct Integral_SIMD
{
    bool operator()(const T *, size_t,
                    ST *, size_t,
                    QT *, size_t,
                    ST *, size_t,
                    int, int, int) const
    {
        return false;
    }
};

#if CV_SIMD && CV_SIMD_WIDTH <= 64

// Squared-pixel integral: store one vector block.
// The horizontal prefix sum of squares is computed in int32 (two halves, lo/hi);
// here it is converted to the sqsum output type and added to the integral row above.
// Overloaded on the output element type so the same int32 prefix feeds CV_64F /
// CV_32F / CV_32S sqsum outputs (Integral_SIMD<uchar,int,QT> below).
#if CV_SIMD_64F
static inline void v_store_sqsum_block(double* dst, const double* prev,
                                       const v_int32& lo, const v_int32& hi)
{
    const int vl = VTraits<v_float64>::vlanes();
    // Convert the int32 squared prefix to f64 via int64. v_cvt_f64(v_int32) goes
    // through float32 on AArch64 NEON (vcvt_f64_f32(vcvt_f32_s32(...))) and loses
    // precision once the prefix exceeds 2^24 (~16.7M, reached well within one HD
    // row of squares); the int64 round-trip keeps it exact on every backend.
    v_int64 lo0, lo1, hi0, hi1;
    v_expand(lo, lo0, lo1);
    v_expand(hi, hi0, hi1);
    v_store(dst,            v_add(v_cvt_f64(lo0), vx_load(prev)));
    v_store(dst +     vl,   v_add(v_cvt_f64(lo1), vx_load(prev +     vl)));
    v_store(dst + 2 * vl,   v_add(v_cvt_f64(hi0), vx_load(prev + 2 * vl)));
    v_store(dst + 3 * vl,   v_add(v_cvt_f64(hi1), vx_load(prev + 3 * vl)));
}
#endif // CV_SIMD_64F
static inline void v_store_sqsum_block(float* dst, const float* prev,
                                       const v_int32& lo, const v_int32& hi)
{
    const int vl = VTraits<v_float32>::vlanes();
    v_store(dst,          v_add(v_cvt_f32(lo), vx_load(prev)));
    v_store(dst +    vl,  v_add(v_cvt_f32(hi), vx_load(prev +    vl)));
}
static inline void v_store_sqsum_block(int* dst, const int* prev,
                                       const v_int32& lo, const v_int32& hi)
{
    const int vl = VTraits<v_int32>::vlanes();
    v_store(dst,          v_add(lo, vx_load(prev)));
    v_store(dst +    vl,  v_add(hi, vx_load(prev +    vl)));
}

// In-register horizontal prefix sum of squared 8-bit pixels for one vector block.
// px holds the block's pixels expanded to uint16; the running per-row carry
// prev_sq is folded in and updated. The prefix runs in int32 (callers gate on
// width * 255^2 <= INT32_MAX via sqsum_cn_ok, so it never overflows), using the
// same rotate/broadcast idiom as the plain sum. Returns the two int32 halves.
static inline void v_sqsum_prefix(const v_uint16& px, v_int32& prev_sq,
                                  v_int32& sqlo, v_int32& sqhi)
{
    v_uint32 sqlo_u, sqhi_u;
    v_mul_expand(px, px, sqlo_u, sqhi_u);
    sqlo = v_reinterpret_as_s32(sqlo_u);
    sqhi = v_reinterpret_as_s32(sqhi_u);
    sqlo = v_add(sqlo, v_rotate_left<1>(sqlo));
    sqlo = v_add(sqlo, v_rotate_left<2>(sqlo));
    sqhi = v_add(sqhi, v_rotate_left<1>(sqhi));
    sqhi = v_add(sqhi, v_rotate_left<2>(sqhi));
#if CV_SIMD_WIDTH >= 32
    sqlo = v_add(sqlo, v_rotate_left<4>(sqlo));
    sqhi = v_add(sqhi, v_rotate_left<4>(sqhi));
#if CV_SIMD_WIDTH == 64
    sqlo = v_add(sqlo, v_rotate_left<8>(sqlo));
    sqhi = v_add(sqhi, v_rotate_left<8>(sqhi));
#endif
#endif
    sqlo = v_add(sqlo, prev_sq);
    sqhi = v_add(sqhi, v_broadcast_highest(sqlo));
    prev_sq = v_broadcast_highest(sqhi);
}

template <typename QT>
struct Integral_SIMD<uchar, int, QT>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
                    int * sum, size_t _sumstep,
                    QT * sqsum, size_t _sqsumstep,
                    int * tilted, size_t,
                    int width, int height, int cn) const
    {
        // sqsum is vectorized for cn 1..3 at any SIMD width, and cn==4 only at
        // 128-bit width (where the 4 channels map to two pixels per vector). Wider
        // cn==4 sqsum and the tilted case still fall back to the scalar path.
        // The squared prefix runs in int32, so it is valid only while a row's
        // sum of squares (width * 255^2) fits in int32; wider rows fall back to scalar.
        const bool sqsum_cn_ok = (cn == 1 || cn == 2 || cn == 3 || (cn == 4 && CV_SIMD_WIDTH == 16))
                                 && width <= 0x7FFFFFFF / (255 * 255);
        if ((sqsum && !sqsum_cn_ok) || tilted || cn > 4)
            return false;
#if !CV_SSE4_1 && CV_SSE2
        // 3 channel code is slower for SSE2 & SSE3
        if (cn == 3)
            return false;
#endif

        width *= cn;

        // the first iteration
        memset(sum, 0, (width + cn) * sizeof(int));
        if (sqsum)
            memset(sqsum, 0, (width + cn) * sizeof(QT));

        if (cn == 1)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + 1;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + 1;

                sum_row[-1] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + 1 : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + 1 : 0;
                if (sqsum)
                    sqsum_row[-1] = 0;

                v_int32 prev = vx_setzero_s32();
                v_int32 prev_sq = vx_setzero_s32();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_uint16 px = vx_load_expand(src_row + j);
                    v_int16 el8 = v_reinterpret_as_s16(px);
                    v_int32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum)), prev.val);
                    el4h.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum)), _mm256_permutevar8x32_epi32(el4l.val, shmask));
                    prev.val = _mm256_permutevar8x32_epi32(el4h.val, shmask);
#else
                    el8 = v_add(el8, v_rotate_left<1>(el8));
                    el8 = v_add(el8, v_rotate_left<2>(el8));
#if CV_SIMD_WIDTH >= 32
                    el8 = v_add(el8, v_rotate_left<4>(el8));
#if CV_SIMD_WIDTH == 64
                    el8 = v_add(el8, v_rotate_left<8>(el8));
#endif
#endif
                    v_expand(el8, el4l, el4h);
                    el4l = v_add(el4l, prev);
                    el4h = v_add(el4h, el4l);
                    prev = v_broadcast_highest(el4h);
#endif
                    v_store(sum_row + j                  , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes())));

                    if (sqsum)
                    {
                        v_int32 sqlo, sqhi;
                        v_sqsum_prefix(px, prev_sq, sqlo, sqhi);
                        v_store_sqsum_block(sqsum_row + j, prev_sqsum_row + j, sqlo, sqhi);
                    }
                }

                int jt = j;
                for (int v = sum_row[jt - 1] - prev_sum_row[jt - 1]; jt < width; ++jt)
                    sum_row[jt] = (v += src_row[jt]) + prev_sum_row[jt];
                if (sqsum)
                {
                    QT s = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    for ( ; j < width; ++j)
                    {
                        int it = src_row[j];
                        s += (QT)it * it;
                        sqsum_row[j] = s + prev_sqsum_row[j];
                    }
                }
            }
        }
        else if (cn == 2)
        {
            // the others
            v_int16 mask = vx_setall_s16((short)0xff);
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + cn;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = 0;

                v_int32 prev_1 = vx_setzero_s32(), prev_2 = vx_setzero_s32();
                v_int32 prev_sq_1 = vx_setzero_s32(), prev_sq_2 = vx_setzero_s32();
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() * cn <= width; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_int16 v_src_row = v_reinterpret_as_s16(vx_load(src_row + j));
                    v_int16 el8_1 = v_and(v_src_row, mask);
                    v_int16 el8_2 = v_reinterpret_as_s16(v_shr<8>(v_reinterpret_as_u16(v_src_row)));

                    if (sqsum)
                    {
                        // Per-channel squared prefix (el8_1/el8_2 hold the raw channel
                        // values, one per pixel); computed before the sum prefix below
                        // overwrites them, then re-interleaved to match the output layout.
                        v_int32 sqlo_1, sqhi_1, sqlo_2, sqhi_2;
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_1), prev_sq_1, sqlo_1, sqhi_1);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_2), prev_sq_2, sqlo_2, sqhi_2);
                        v_int32 z1, z2, z3, z4;
                        v_zip(sqlo_1, sqlo_2, z1, z2);
                        v_zip(sqhi_1, sqhi_2, z3, z4);
                        const int vl = VTraits<v_int32>::vlanes();
                        v_store_sqsum_block(sqsum_row + j,          prev_sqsum_row + j,          z1, z2);
                        v_store_sqsum_block(sqsum_row + j + 2 * vl, prev_sqsum_row + j + 2 * vl, z3, z4);
                    }

                    v_int32 el4l_1, el4h_1, el4l_2, el4h_2;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1)), prev_1.val);
                    el4l_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2)), prev_2.val);
                    el4h_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1)), _mm256_permutevar8x32_epi32(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2)), _mm256_permutevar8x32_epi32(el4l_2.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_epi32(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_epi32(el4h_2.val, shmask);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
#endif
#endif
                    v_expand(el8_1, el4l_1, el4h_1);
                    v_expand(el8_2, el4l_2, el4h_2);
                    el4l_1 = v_add(el4l_1, prev_1);
                    el4l_2 = v_add(el4l_2, prev_2);
                    el4h_1 = v_add(el4h_1, el4l_1);
                    el4h_2 = v_add(el4h_2, el4l_2);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
#endif
                    v_int32 el4_1, el4_2, el4_3, el4_4;
                    v_zip(el4l_1, el4l_2, el4_1, el4_2);
                    v_zip(el4h_1, el4h_2, el4_3, el4_4);
                    v_store(sum_row + j                      , v_add(el4_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes()    , v_add(el4_2, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes())));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 2, v_add(el4_3, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 3, v_add(el4_4, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 3)));
                }

                int jt = j;
                for (int v2 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                         v1 = sum_row[jt - 2] - prev_sum_row[jt - 2]; jt < width; jt += 2)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                }
                if (sqsum)
                {
                    QT sq2 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    QT sq1 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    for ( ; j < width; j += 2)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1];
                        sq1 += (QT)i1 * i1;
                        sq2 += (QT)i2 * i2;
                        sqsum_row[j]     = sq1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = sq2 + prev_sqsum_row[j + 1];
                    }
                }
            }
        }
#if CV_SSE4_1 || !CV_SSE2
        else if (cn == 3)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + cn;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + cn;
                int row_cache[VTraits<v_int32>::max_nlanes * 6];
                int sq_cache[VTraits<v_int32>::max_nlanes * 6];

                sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = sqsum_row[-3] = 0;

                v_int32 prev_1 = vx_setzero_s32(), prev_2 = vx_setzero_s32(),
                        prev_3 = vx_setzero_s32();
                v_int32 prev_sq_1 = vx_setzero_s32(), prev_sq_2 = vx_setzero_s32(),
                        prev_sq_3 = vx_setzero_s32();
                int j = 0;
                const int j_max =
                        ((_srcstep * i + (width - VTraits<v_uint16>::vlanes() * cn + VTraits<v_uint8>::vlanes() * cn)) >= _srcstep * height)
                        ? width - VTraits<v_uint8>::vlanes() * cn    // uint8 in v_load_deinterleave()
                        : width - VTraits<v_uint16>::vlanes() * cn;  // v_expand_low
                for ( ; j <= j_max; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_uint8 v_src_row_1, v_src_row_2, v_src_row_3;
                    v_load_deinterleave(src_row + j, v_src_row_1, v_src_row_2, v_src_row_3);
                    v_int16 el8_1 = v_reinterpret_as_s16(v_expand_low(v_src_row_1));
                    v_int16 el8_2 = v_reinterpret_as_s16(v_expand_low(v_src_row_2));
                    v_int16 el8_3 = v_reinterpret_as_s16(v_expand_low(v_src_row_3));

                    if (sqsum)
                    {
                        // Per-channel squared prefix (el8_x hold raw channel values),
                        // then re-interleaved through sq_cache like the sum below.
                        v_int32 sl1, sh1, sl2, sh2, sl3, sh3;
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_1), prev_sq_1, sl1, sh1);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_2), prev_sq_2, sl2, sh2);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_3), prev_sq_3, sl3, sh3);
                        const int vl = VTraits<v_int32>::vlanes();
                        v_store_interleave(sq_cache         , sl1, sl2, sl3);
                        v_store_interleave(sq_cache + vl * 3, sh1, sh2, sh3);
                        v_int32 z0 = vx_load(sq_cache         ), z1 = vx_load(sq_cache + vl    );
                        v_int32 z2 = vx_load(sq_cache + vl * 2), z3 = vx_load(sq_cache + vl * 3);
                        v_int32 z4 = vx_load(sq_cache + vl * 4), z5 = vx_load(sq_cache + vl * 5);
                        v_store_sqsum_block(sqsum_row + j         , prev_sqsum_row + j         , z0, z1);
                        v_store_sqsum_block(sqsum_row + j + vl * 2, prev_sqsum_row + j + vl * 2, z2, z3);
                        v_store_sqsum_block(sqsum_row + j + vl * 4, prev_sqsum_row + j + vl * 4, z4, z5);
                    }

                    v_int32 el4l_1, el4h_1, el4l_2, el4h_2, el4l_3, el4h_3;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    __m256i vsum_3 = _mm256_add_epi16(el8_3.val, _mm256_slli_si256(el8_3.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1)), prev_1.val);
                    el4l_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2)), prev_2.val);
                    el4l_3.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_3)), prev_3.val);
                    el4h_1.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1)), _mm256_permutevar8x32_epi32(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2)), _mm256_permutevar8x32_epi32(el4l_2.val, shmask));
                    el4h_3.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_3)), _mm256_permutevar8x32_epi32(el4l_3.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_epi32(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_epi32(el4h_2.val, shmask);
                    prev_3.val = _mm256_permutevar8x32_epi32(el4h_3.val, shmask);
#else
                    el8_1 = v_add(el8_1,v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2,v_rotate_left<1>(el8_2));
                    el8_3 = v_add(el8_3,v_rotate_left<1>(el8_3));
                    el8_1 = v_add(el8_1,v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2,v_rotate_left<2>(el8_2));
                    el8_3 = v_add(el8_3,v_rotate_left<2>(el8_3));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<4>(el8_3));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<8>(el8_3));
#endif
#endif
                    v_expand(el8_1, el4l_1, el4h_1);
                    v_expand(el8_2, el4l_2, el4h_2);
                    v_expand(el8_3, el4l_3, el4h_3);
                    el4l_1 = v_add(el4l_1, prev_1);
                    el4l_2 = v_add(el4l_2, prev_2);
                    el4l_3 = v_add(el4l_3, prev_3);
                    el4h_1 = v_add(el4h_1, el4l_1);
                    el4h_2 = v_add(el4h_2, el4l_2);
                    el4h_3 = v_add(el4h_3, el4l_3);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
                    prev_3 = v_broadcast_highest(el4h_3);
#endif
                    v_store_interleave(row_cache                      , el4l_1, el4l_2, el4l_3);
                    v_store_interleave(row_cache + VTraits<v_int32>::vlanes() * 3, el4h_1, el4h_2, el4h_3);
                    el4l_1 = vx_load(row_cache                      );
                    el4l_2 = vx_load(row_cache + VTraits<v_int32>::vlanes()    );
                    el4l_3 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 2);
                    el4h_1 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 3);
                    el4h_2 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 4);
                    el4h_3 = vx_load(row_cache + VTraits<v_int32>::vlanes() * 5);
                    v_store(sum_row + j                      ,            v_add(el4l_1, vx_load(prev_sum_row + j                      )));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes()    , v_add(el4l_2, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes()    )));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 2, v_add(el4l_3, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 3, v_add(el4h_1, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 4, v_add(el4h_2, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes() * 5, v_add(el4h_3, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes() * 5)));
                }

                int jt = j;
                for (int v3 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                         v2 = sum_row[jt - 2] - prev_sum_row[jt - 2],
                         v1 = sum_row[jt - 3] - prev_sum_row[jt - 3]; jt < width; jt += 3)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                    sum_row[jt + 2] = (v3 += src_row[jt + 2]) + prev_sum_row[jt + 2];
                }
                if (sqsum)
                {
                    QT s3 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    QT s2 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    QT s1 = sqsum_row[j - 3] - prev_sqsum_row[j - 3];
                    for ( ; j < width; j += 3)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1], i3 = src_row[j + 2];
                        s1 += (QT)i1 * i1; s2 += (QT)i2 * i2; s3 += (QT)i3 * i3;
                        sqsum_row[j]     = s1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = s2 + prev_sqsum_row[j + 1];
                        sqsum_row[j + 2] = s3 + prev_sqsum_row[j + 2];
                    }
                }
            }
        }
#endif
        else if (cn == 4)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                int * prev_sum_row = (int *)((uchar *)sum + _sumstep * i) + cn;
                int * sum_row = (int *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = sqsum_row[-3] = sqsum_row[-4] = 0;

                v_int32 prev = vx_setzero_s32();
#if CV_SIMD_WIDTH == 16
                v_int32 prev_sq = vx_setzero_s32();
#endif
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_uint16 px = vx_load_expand(src_row + j);
                    v_int16 el8 = v_reinterpret_as_s16(px);
                    v_int32 el4l, el4h;
#if CV_SIMD_WIDTH == 16
                    // cn==4 sqsum is enabled only at 128-bit width (see bail): the 8
                    // expanded pixels are exactly two 4-channel pixels, so the squared
                    // prefix is the same two-pixel accumulate as the sum below.
                    if (sqsum)
                    {
                        v_uint32 sq_lo_u, sq_hi_u;
                        v_mul_expand(px, px, sq_lo_u, sq_hi_u);
                        v_int32 sq_l = v_reinterpret_as_s32(sq_lo_u);
                        v_int32 sq_h = v_reinterpret_as_s32(sq_hi_u);
                        sq_l = v_add(sq_l, prev_sq);
                        sq_h = v_add(sq_h, sq_l);
                        prev_sq = sq_h;
                        v_store_sqsum_block(sqsum_row + j, prev_sqsum_row + j, sq_l, sq_h);
                    }
#endif
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 8));
                    el4l.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_low(vsum)), prev.val);
                    el4h.val = _mm256_add_epi32(_mm256_cvtepi16_epi32(_v256_extract_high(vsum)), _mm256_permute2x128_si256(el4l.val, el4l.val, 0x31));
                    prev.val = _mm256_permute2x128_si256(el4h.val, el4h.val, 0x31);
#else
#if CV_SIMD_WIDTH >= 32
                    el8 = v_add(el8, v_rotate_left<4>(el8));
#if CV_SIMD_WIDTH == 64
                    el8 = v_add(el8, v_rotate_left<8>(el8));
#endif
#endif
                    v_expand(el8, el4l, el4h);
                    el4l = v_add(el4l, prev);
                    el4h = v_add(el4h, el4l);
#if CV_SIMD_WIDTH == 16
                    prev = el4h;
#elif CV_SIMD_WIDTH == 32
                    prev = v_combine_high(el4h, el4h);
#else
                    v_int32 t = v_rotate_right<12>(el4h);
                    t = v_or(t, v_rotate_left<4>(t));
                    prev = v_combine_low(t, t);
#endif
#endif
                    v_store(sum_row + j                  , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_int32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_int32>::vlanes())));
                }

                int jt = j;
                for (int v4 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                         v3 = sum_row[jt - 2] - prev_sum_row[jt - 2],
                         v2 = sum_row[jt - 3] - prev_sum_row[jt - 3],
                         v1 = sum_row[jt - 4] - prev_sum_row[jt - 4]; jt < width; jt += 4)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                    sum_row[jt + 2] = (v3 += src_row[jt + 2]) + prev_sum_row[jt + 2];
                    sum_row[jt + 3] = (v4 += src_row[jt + 3]) + prev_sum_row[jt + 3];
                }
                if (sqsum)
                {
                    QT s4 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    QT s3 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    QT s2 = sqsum_row[j - 3] - prev_sqsum_row[j - 3];
                    QT s1 = sqsum_row[j - 4] - prev_sqsum_row[j - 4];
                    for ( ; j < width; j += 4)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1], i3 = src_row[j + 2], i4 = src_row[j + 3];
                        s1 += (QT)i1 * i1; s2 += (QT)i2 * i2; s3 += (QT)i3 * i3; s4 += (QT)i4 * i4;
                        sqsum_row[j]     = s1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = s2 + prev_sqsum_row[j + 1];
                        sqsum_row[j + 2] = s3 + prev_sqsum_row[j + 2];
                        sqsum_row[j + 3] = s4 + prev_sqsum_row[j + 3];
                    }
                }
            }
        }
        else
        {
            return false;
        }
        vx_cleanup();

        return true;
    }
};

template <typename QT>
struct Integral_SIMD<uchar, float, QT>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
        float * sum, size_t _sumstep,
        QT * sqsum, size_t _sqsumstep,
        float * tilted, size_t,
        int width, int height, int cn) const
    {
        // sqsum is vectorized for cn 1..3 at any SIMD width, and cn==4 only at
        // 128-bit width; wider cn==4 sqsum and the tilted case fall back to scalar.
        // The squared prefix runs in int32, so it is valid only while a row's
        // sum of squares (width * 255^2) fits in int32; wider rows fall back to scalar.
        const bool sqsum_cn_ok = (cn == 1 || cn == 2 || cn == 3 || (cn == 4 && CV_SIMD_WIDTH == 16))
                                 && width <= 0x7FFFFFFF / (255 * 255);
        if ((sqsum && !sqsum_cn_ok) || tilted || cn > 4)
            return false;

        width *= cn;

        // the first iteration
        memset(sum, 0, (width + cn) * sizeof(float));
        if (sqsum)
            memset(sqsum, 0, (width + cn) * sizeof(QT));

        if (cn == 1)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + 1;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + 1;

                sum_row[-1] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + 1 : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + 1 : 0;
                if (sqsum)
                    sqsum_row[-1] = 0;

                v_float32 prev = vx_setzero_f32();
                v_int32 prev_sq = vx_setzero_s32();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_uint16 px = vx_load_expand(src_row + j);
                    v_int16 el8 = v_reinterpret_as_s16(px);
                    v_float32 el4l, el4h;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum))), prev.val);
                    el4h.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum))), _mm256_permutevar8x32_ps(el4l.val, shmask));
                    prev.val = _mm256_permutevar8x32_ps(el4h.val, shmask);
#else
                    el8 = v_add(el8, v_rotate_left<1>(el8));
                    el8 = v_add(el8, v_rotate_left<2>(el8));
#if CV_SIMD_WIDTH >= 32
                    el8 = v_add(el8, v_rotate_left<4>(el8));
#if CV_SIMD_WIDTH == 64
                    el8 = v_add(el8, v_rotate_left<8>(el8));
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4l = v_add(v_cvt_f32(el4li), prev);
                    el4h = v_add(v_cvt_f32(el4hi), el4l);
                    prev = v_broadcast_highest(el4h);
#endif
                    v_store(sum_row + j                    , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));

                    if (sqsum)
                    {
                        v_int32 sqlo, sqhi;
                        v_sqsum_prefix(px, prev_sq, sqlo, sqhi);
                        v_store_sqsum_block(sqsum_row + j, prev_sqsum_row + j, sqlo, sqhi);
                    }
                }

                int jt = j;
                for (float v = sum_row[jt - 1] - prev_sum_row[jt - 1]; jt < width; ++jt)
                    sum_row[jt] = (v += src_row[jt]) + prev_sum_row[jt];
                if (sqsum)
                {
                    QT s = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    for ( ; j < width; ++j)
                    {
                        int it = src_row[j];
                        s += (QT)it * it;
                        sqsum_row[j] = s + prev_sqsum_row[j];
                    }
                }
            }
        }
        else if (cn == 2)
        {
            // the others
            v_int16 mask = vx_setall_s16((short)0xff);
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + cn;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = 0;

                v_float32 prev_1 = vx_setzero_f32(), prev_2 = vx_setzero_f32();
                v_int32 prev_sq_1 = vx_setzero_s32(), prev_sq_2 = vx_setzero_s32();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() * cn <= width; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_int16 v_src_row = v_reinterpret_as_s16(vx_load(src_row + j));
                    v_int16 el8_1 = v_and(v_src_row, mask);
                    v_int16 el8_2 = v_reinterpret_as_s16(v_shr<8>(v_reinterpret_as_u16(v_src_row)));

                    if (sqsum)
                    {
                        v_int32 sqlo_1, sqhi_1, sqlo_2, sqhi_2;
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_1), prev_sq_1, sqlo_1, sqhi_1);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_2), prev_sq_2, sqlo_2, sqhi_2);
                        v_int32 z1, z2, z3, z4;
                        v_zip(sqlo_1, sqlo_2, z1, z2);
                        v_zip(sqhi_1, sqhi_2, z3, z4);
                        const int vl = VTraits<v_int32>::vlanes();
                        v_store_sqsum_block(sqsum_row + j,          prev_sqsum_row + j,          z1, z2);
                        v_store_sqsum_block(sqsum_row + j + 2 * vl, prev_sqsum_row + j + 2 * vl, z3, z4);
                    }

                    v_float32 el4l_1, el4h_1, el4l_2, el4h_2;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1))), prev_1.val);
                    el4l_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2))), prev_2.val);
                    el4h_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1))), _mm256_permutevar8x32_ps(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2))), _mm256_permutevar8x32_ps(el4l_2.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_ps(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_ps(el4h_2.val, shmask);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    el4l_1 = v_add(v_cvt_f32(el4li_1), prev_1);
                    el4l_2 = v_add(v_cvt_f32(el4li_2), prev_2);
                    el4h_1 = v_add(v_cvt_f32(el4hi_1), el4l_1);
                    el4h_2 = v_add(v_cvt_f32(el4hi_2), el4l_2);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
#endif
                    v_float32 el4_1, el4_2, el4_3, el4_4;
                    v_zip(el4l_1, el4l_2, el4_1, el4_2);
                    v_zip(el4h_1, el4h_2, el4_3, el4_4);
                    v_store(sum_row + j                        , v_add(el4_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes()    , v_add(el4_2, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 2, v_add(el4_3, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 3, v_add(el4_4, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 3)));
                }

                int jt = j;
                for (float v2 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                           v1 = sum_row[jt - 2] - prev_sum_row[jt - 2]; jt < width; jt += 2)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                }
                if (sqsum)
                {
                    QT sq2 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    QT sq1 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    for ( ; j < width; j += 2)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1];
                        sq1 += (QT)i1 * i1;
                        sq2 += (QT)i2 * i2;
                        sqsum_row[j]     = sq1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = sq2 + prev_sqsum_row[j + 1];
                    }
                }
            }
        }
        else if (cn == 3)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + cn;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + cn;
                float row_cache[VTraits<v_float32>::max_nlanes * 6];
                int sq_cache[VTraits<v_int32>::max_nlanes * 6];

                sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = sqsum_row[-3] = 0;

                v_float32 prev_1 = vx_setzero_f32(), prev_2 = vx_setzero_f32(),
                          prev_3 = vx_setzero_f32();
                v_int32 prev_sq_1 = vx_setzero_s32(), prev_sq_2 = vx_setzero_s32(),
                        prev_sq_3 = vx_setzero_s32();
                int j = 0;
                const int j_max =
                        ((_srcstep * i + (width - VTraits<v_uint16>::vlanes() * cn + VTraits<v_uint8>::vlanes() * cn)) >= _srcstep * height)
                        ? width - VTraits<v_uint8>::vlanes() * cn    // uint8 in v_load_deinterleave()
                        : width - VTraits<v_uint16>::vlanes() * cn;  // v_expand_low
                for ( ; j <= j_max; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_uint8 v_src_row_1, v_src_row_2, v_src_row_3;
                    v_load_deinterleave(src_row + j, v_src_row_1, v_src_row_2, v_src_row_3);
                    v_int16 el8_1 = v_reinterpret_as_s16(v_expand_low(v_src_row_1));
                    v_int16 el8_2 = v_reinterpret_as_s16(v_expand_low(v_src_row_2));
                    v_int16 el8_3 = v_reinterpret_as_s16(v_expand_low(v_src_row_3));

                    if (sqsum)
                    {
                        v_int32 sl1, sh1, sl2, sh2, sl3, sh3;
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_1), prev_sq_1, sl1, sh1);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_2), prev_sq_2, sl2, sh2);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_3), prev_sq_3, sl3, sh3);
                        const int vl = VTraits<v_int32>::vlanes();
                        v_store_interleave(sq_cache         , sl1, sl2, sl3);
                        v_store_interleave(sq_cache + vl * 3, sh1, sh2, sh3);
                        v_int32 z0 = vx_load(sq_cache         ), z1 = vx_load(sq_cache + vl    );
                        v_int32 z2 = vx_load(sq_cache + vl * 2), z3 = vx_load(sq_cache + vl * 3);
                        v_int32 z4 = vx_load(sq_cache + vl * 4), z5 = vx_load(sq_cache + vl * 5);
                        v_store_sqsum_block(sqsum_row + j         , prev_sqsum_row + j         , z0, z1);
                        v_store_sqsum_block(sqsum_row + j + 2 * vl, prev_sqsum_row + j + 2 * vl, z2, z3);
                        v_store_sqsum_block(sqsum_row + j + 4 * vl, prev_sqsum_row + j + 4 * vl, z4, z5);
                    }

                    v_float32 el4l_1, el4h_1, el4l_2, el4h_2, el4l_3, el4h_3;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    __m256i vsum_3 = _mm256_add_epi16(el8_3.val, _mm256_slli_si256(el8_3.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 8));
                    __m256i shmask = _mm256_set1_epi32(7);
                    el4l_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_1))), prev_1.val);
                    el4l_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_2))), prev_2.val);
                    el4l_3.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum_3))), prev_3.val);
                    el4h_1.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_1))), _mm256_permutevar8x32_ps(el4l_1.val, shmask));
                    el4h_2.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_2))), _mm256_permutevar8x32_ps(el4l_2.val, shmask));
                    el4h_3.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum_3))), _mm256_permutevar8x32_ps(el4l_3.val, shmask));
                    prev_1.val = _mm256_permutevar8x32_ps(el4h_1.val, shmask);
                    prev_2.val = _mm256_permutevar8x32_ps(el4h_2.val, shmask);
                    prev_3.val = _mm256_permutevar8x32_ps(el4h_3.val, shmask);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<1>(el8_3));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<2>(el8_3));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<4>(el8_3));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<8>(el8_3));
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2, el4li_3, el4hi_3;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    v_expand(el8_3, el4li_3, el4hi_3);
                    el4l_1 = v_add(v_cvt_f32(el4li_1), prev_1);
                    el4l_2 = v_add(v_cvt_f32(el4li_2), prev_2);
                    el4l_3 = v_add(v_cvt_f32(el4li_3), prev_3);
                    el4h_1 = v_add(v_cvt_f32(el4hi_1), el4l_1);
                    el4h_2 = v_add(v_cvt_f32(el4hi_2), el4l_2);
                    el4h_3 = v_add(v_cvt_f32(el4hi_3), el4l_3);
                    prev_1 = v_broadcast_highest(el4h_1);
                    prev_2 = v_broadcast_highest(el4h_2);
                    prev_3 = v_broadcast_highest(el4h_3);
#endif
                    v_store_interleave(row_cache                        , el4l_1, el4l_2, el4l_3);
                    v_store_interleave(row_cache + VTraits<v_float32>::vlanes() * 3, el4h_1, el4h_2, el4h_3);
                    el4l_1 = vx_load(row_cache                        );
                    el4l_2 = vx_load(row_cache + VTraits<v_float32>::vlanes()    );
                    el4l_3 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 2);
                    el4h_1 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 3);
                    el4h_2 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 4);
                    el4h_3 = vx_load(row_cache + VTraits<v_float32>::vlanes() * 5);
                    v_store(sum_row + j                        , v_add(el4l_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes()    , v_add(el4l_2, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 2, v_add(el4l_3, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 3, v_add(el4h_1, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 4, v_add(el4h_2, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes() * 5, v_add(el4h_3, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes() * 5)));
                }

                int jt = j;
                for (float v3 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                           v2 = sum_row[jt - 2] - prev_sum_row[jt - 2],
                           v1 = sum_row[jt - 3] - prev_sum_row[jt - 3]; jt < width; jt += 3)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                    sum_row[jt + 2] = (v3 += src_row[jt + 2]) + prev_sum_row[jt + 2];
                }
                if (sqsum)
                {
                    QT s3 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    QT s2 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    QT s1 = sqsum_row[j - 3] - prev_sqsum_row[j - 3];
                    for ( ; j < width; j += 3)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1], i3 = src_row[j + 2];
                        s1 += (QT)i1 * i1; s2 += (QT)i2 * i2; s3 += (QT)i3 * i3;
                        sqsum_row[j]     = s1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = s2 + prev_sqsum_row[j + 1];
                        sqsum_row[j + 2] = s3 + prev_sqsum_row[j + 2];
                    }
                }
            }
        }
        else if (cn == 4)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                float * prev_sum_row = (float *)((uchar *)sum + _sumstep * i) + cn;
                float * sum_row = (float *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

                QT * prev_sqsum_row = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                QT * sqsum_row      = sqsum ? (QT *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = sqsum_row[-3] = sqsum_row[-4] = 0;

                v_float32 prev = vx_setzero_f32();
#if CV_SIMD_WIDTH == 16
                v_int32 prev_sq = vx_setzero_s32();
#endif
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_uint16 px = vx_load_expand(src_row + j);
                    v_int16 el8 = v_reinterpret_as_s16(px);
                    v_float32 el4l, el4h;
#if CV_SIMD_WIDTH == 16
                    if (sqsum)
                    {
                        v_uint32 sq_lo_u, sq_hi_u;
                        v_mul_expand(px, px, sq_lo_u, sq_hi_u);
                        v_int32 sq_l = v_reinterpret_as_s32(sq_lo_u);
                        v_int32 sq_h = v_reinterpret_as_s32(sq_hi_u);
                        sq_l = v_add(sq_l, prev_sq);
                        sq_h = v_add(sq_h, sq_l);
                        prev_sq = sq_h;
                        v_store_sqsum_block(sqsum_row + j, prev_sqsum_row + j, sq_l, sq_h);
                    }
#endif
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 8));
                    el4l.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_low(vsum))), prev.val);
                    el4h.val = _mm256_add_ps(_mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_v256_extract_high(vsum))), _mm256_permute2f128_ps(el4l.val, el4l.val, 0x31));
                    prev.val = _mm256_permute2f128_ps(el4h.val, el4h.val, 0x31);
#else
#if CV_SIMD_WIDTH >= 32
                    el8 = v_add(el8, v_rotate_left<4>(el8));
#if CV_SIMD_WIDTH == 64
                    el8 = v_add(el8, v_rotate_left<8>(el8));
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4l = v_add(v_cvt_f32(el4li), prev);
                    el4h = v_add(v_cvt_f32(el4hi), el4l);
#if CV_SIMD_WIDTH == 16
                    prev = el4h;
#elif CV_SIMD_WIDTH == 32
                    prev = v_combine_high(el4h, el4h);
#else
                    v_float32 t = v_rotate_right<12>(el4h);
                    t = v_or(t, v_rotate_left<4>(t));
                    prev = v_combine_low(t, t);
#endif
#endif
                    v_store(sum_row + j                    , v_add(el4l, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float32>::vlanes(), v_add(el4h, vx_load(prev_sum_row + j + VTraits<v_float32>::vlanes())));
                }

                int jt = j;
                for (float v4 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                           v3 = sum_row[jt - 2] - prev_sum_row[jt - 2],
                           v2 = sum_row[jt - 3] - prev_sum_row[jt - 3],
                           v1 = sum_row[jt - 4] - prev_sum_row[jt - 4]; jt < width; jt += 4)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                    sum_row[jt + 2] = (v3 += src_row[jt + 2]) + prev_sum_row[jt + 2];
                    sum_row[jt + 3] = (v4 += src_row[jt + 3]) + prev_sum_row[jt + 3];
                }
                if (sqsum)
                {
                    QT s4 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    QT s3 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    QT s2 = sqsum_row[j - 3] - prev_sqsum_row[j - 3];
                    QT s1 = sqsum_row[j - 4] - prev_sqsum_row[j - 4];
                    for ( ; j < width; j += 4)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1], i3 = src_row[j + 2], i4 = src_row[j + 3];
                        s1 += (QT)i1 * i1; s2 += (QT)i2 * i2; s3 += (QT)i3 * i3; s4 += (QT)i4 * i4;
                        sqsum_row[j]     = s1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = s2 + prev_sqsum_row[j + 1];
                        sqsum_row[j + 2] = s3 + prev_sqsum_row[j + 2];
                        sqsum_row[j + 3] = s4 + prev_sqsum_row[j + 3];
                    }
                }
            }
        }
        else
        {
            return false;
        }
        vx_cleanup();

        return true;
    }
};

#if CV_SIMD128_64F
template <>
struct Integral_SIMD<uchar, double, double>
{
    Integral_SIMD() {}

    bool operator()(const uchar * src, size_t _srcstep,
        double * sum, size_t _sumstep,
        double * sqsum, size_t _sqsumstep,
        double * tilted, size_t,
        int width, int height, int cn) const
    {
#if CV_AVX512_SKX
        if (!tilted && cn <= 4 && (cn > 1 || sqsum))
        {
            calculate_integral_avx512(src, _srcstep, sum, _sumstep, sqsum, _sqsumstep, width, height, cn);
            return true;
        }
#endif
        // sqsum is vectorized for cn 1..3 at any SIMD width and cn==4 at 128-bit
        // width (the AVX-512 path above already covers the multi-channel case);
        // wider cn==4 sqsum and the tilted case fall back to scalar here.
        // The squared prefix runs in int32, so it is valid only while a row's
        // sum of squares (width * 255^2) fits in int32; wider rows fall back to scalar.
        const bool sqsum_cn_ok = (cn == 1 || cn == 2 || cn == 3 || (cn == 4 && CV_SIMD_WIDTH == 16))
                                 && width <= 0x7FFFFFFF / (255 * 255);
        if ((sqsum && !sqsum_cn_ok) || tilted || cn > 4)
            return false;

        width *= cn;

        // the first iteration
        memset(sum, 0, (width + cn) * sizeof(double));
        if (sqsum)
            memset(sqsum, 0, (width + cn) * sizeof(double));

        if (cn == 1)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + 1;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + 1;

                sum_row[-1] = 0;

                double * prev_sqsum_row = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * i) + 1 : 0;
                double * sqsum_row      = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * (i + 1)) + 1 : 0;
                if (sqsum)
                    sqsum_row[-1] = 0;

                v_float64 prev = vx_setzero_f64();
                v_int32 prev_sq = vx_setzero_s32();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_uint16 px = vx_load_expand(src_row + j);
                    v_int16 el8 = v_reinterpret_as_s16(px);
                    v_float64 el4ll, el4lh, el4hl, el4hh;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 2));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 4));
                    vsum = _mm256_add_epi16(vsum, _mm256_slli_si256(vsum, 8));
                    __m256i el4l_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum));
                    __m256i el4h_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum));
                    el4ll.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l_32)), prev.val);
                    el4lh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l_32)), prev.val);
                    __m256d el4d = _mm256_permute4x64_pd(el4lh.val, 0xff);
                    el4hl.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h_32)), el4d);
                    el4hh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h_32)), el4d);
                    prev.val = _mm256_permute4x64_pd(el4hh.val, 0xff);
#else
                    el8 = v_add(el8, v_rotate_left<1>(el8));
                    el8 = v_add(el8, v_rotate_left<2>(el8));
#if CV_SIMD_WIDTH >= 32
                    el8 = v_add(el8, v_rotate_left<4>(el8));
#if CV_SIMD_WIDTH == 64
                    el8 = v_add(el8, v_rotate_left<8>(el8));
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4ll = v_add(v_cvt_f64(el4li), prev);
                    el4lh = v_add(v_cvt_f64_high(el4li), prev);
                    el4hl = v_add(v_cvt_f64(el4hi), el4ll);
                    el4hh = v_add(v_cvt_f64_high(el4hi), el4lh);
                    prev = vx_setall_f64(v_extract_highest(el4hh));
//                    prev = v_broadcast_highest(el4hh);
#endif
                    v_store(sum_row + j                        , v_add(el4ll, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()    , v_add(el4lh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2, v_add(el4hl, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3, v_add(el4hh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));

                    if (sqsum)
                    {
                        v_int32 sqlo, sqhi;
                        v_sqsum_prefix(px, prev_sq, sqlo, sqhi);
                        v_store_sqsum_block(sqsum_row + j, prev_sqsum_row + j, sqlo, sqhi);
                    }
                }

                int jt = j;
                for (double v = sum_row[jt - 1] - prev_sum_row[jt - 1]; jt < width; ++jt)
                    sum_row[jt] = (v += src_row[jt]) + prev_sum_row[jt];
                if (sqsum)
                {
                    double s = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    for ( ; j < width; ++j)
                    {
                        int it = src_row[j];
                        s += (double)it * it;
                        sqsum_row[j] = s + prev_sqsum_row[j];
                    }
                }
            }
        }
        else if (cn == 2)
        {
            // the others
            v_int16 mask = vx_setall_s16((short)0xff);
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + cn;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = 0;

                double * prev_sqsum_row = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                double * sqsum_row      = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = 0;

                v_float64 prev_1 = vx_setzero_f64(), prev_2 = vx_setzero_f64();
                v_int32 prev_sq_1 = vx_setzero_s32(), prev_sq_2 = vx_setzero_s32();
                int j = 0;
                for (; j + VTraits<v_uint16>::vlanes() * cn <= width; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_int16 v_src_row = v_reinterpret_as_s16(vx_load(src_row + j));
                    v_int16 el8_1 = v_and(v_src_row, mask);
                    v_int16 el8_2 = v_reinterpret_as_s16(v_shr<8>(v_reinterpret_as_u16(v_src_row)));

                    if (sqsum)
                    {
                        v_int32 sqlo_1, sqhi_1, sqlo_2, sqhi_2;
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_1), prev_sq_1, sqlo_1, sqhi_1);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_2), prev_sq_2, sqlo_2, sqhi_2);
                        v_int32 z1, z2, z3, z4;
                        v_zip(sqlo_1, sqlo_2, z1, z2);
                        v_zip(sqhi_1, sqhi_2, z3, z4);
                        const int vl = VTraits<v_int32>::vlanes();
                        v_store_sqsum_block(sqsum_row + j,          prev_sqsum_row + j,          z1, z2);
                        v_store_sqsum_block(sqsum_row + j + 2 * vl, prev_sqsum_row + j + 2 * vl, z3, z4);
                    }

                    v_float64 el4ll_1, el4lh_1, el4hl_1, el4hh_1, el4ll_2, el4lh_2, el4hl_2, el4hh_2;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    __m256i el4l1_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_1));
                    __m256i el4l2_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_2));
                    __m256i el4h1_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_1));
                    __m256i el4h2_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_2));
                    el4ll_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l1_32)), prev_1.val);
                    el4ll_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l2_32)), prev_2.val);
                    el4lh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l1_32)), prev_1.val);
                    el4lh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l2_32)), prev_2.val);
                    __m256d el4d_1 = _mm256_permute4x64_pd(el4lh_1.val, 0xff);
                    __m256d el4d_2 = _mm256_permute4x64_pd(el4lh_2.val, 0xff);
                    el4hl_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h1_32)), el4d_1);
                    el4hl_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h2_32)), el4d_2);
                    el4hh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h1_32)), el4d_1);
                    el4hh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h2_32)), el4d_2);
                    prev_1.val = _mm256_permute4x64_pd(el4hh_1.val, 0xff);
                    prev_2.val = _mm256_permute4x64_pd(el4hh_2.val, 0xff);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    el4ll_1 = v_add(v_cvt_f64(el4li_1), prev_1);
                    el4ll_2 = v_add(v_cvt_f64(el4li_2), prev_2);
                    el4lh_1 = v_add(v_cvt_f64_high(el4li_1), prev_1);
                    el4lh_2 = v_add(v_cvt_f64_high(el4li_2), prev_2);
                    el4hl_1 = v_add(v_cvt_f64(el4hi_1), el4ll_1);
                    el4hl_2 = v_add(v_cvt_f64(el4hi_2), el4ll_2);
                    el4hh_1 = v_add(v_cvt_f64_high(el4hi_1), el4lh_1);
                    el4hh_2 = v_add(v_cvt_f64_high(el4hi_2), el4lh_2);
                    prev_1 = vx_setall_f64(v_extract_highest(el4hh_1));
                    prev_2 = vx_setall_f64(v_extract_highest(el4hh_2));
//                    prev_1 = v_broadcast_highest(el4hh_1);
//                    prev_2 = v_broadcast_highest(el4hh_2);
#endif
                    v_float64 el4_1, el4_2, el4_3, el4_4, el4_5, el4_6, el4_7, el4_8;
                    v_zip(el4ll_1, el4ll_2, el4_1, el4_2);
                    v_zip(el4lh_1, el4lh_2, el4_3, el4_4);
                    v_zip(el4hl_1, el4hl_2, el4_5, el4_6);
                    v_zip(el4hh_1, el4hh_2, el4_7, el4_8);
                    v_store(sum_row + j                        , v_add(el4_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()    , v_add(el4_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2, v_add(el4_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3, v_add(el4_4, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 4, v_add(el4_5, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 5, v_add(el4_6, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 5)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 6, v_add(el4_7, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 6)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 7, v_add(el4_8, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 7)));
                }

                int jt = j;
                for (double v2 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                            v1 = sum_row[jt - 2] - prev_sum_row[jt - 2]; jt < width; jt += 2)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                }
                if (sqsum)
                {
                    double sq2 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    double sq1 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    for ( ; j < width; j += 2)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1];
                        sq1 += (double)i1 * i1;
                        sq2 += (double)i2 * i2;
                        sqsum_row[j]     = sq1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = sq2 + prev_sqsum_row[j + 1];
                    }
                }
            }
        }
        else if (cn == 3)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + cn;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + cn;
                double row_cache[VTraits<v_float64>::max_nlanes * 12];
                int sq_cache[VTraits<v_int32>::max_nlanes * 6];

                sum_row[-1] = sum_row[-2] = sum_row[-3] = 0;

                double * prev_sqsum_row = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                double * sqsum_row      = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = sqsum_row[-3] = 0;

                v_float64 prev_1 = vx_setzero_f64(), prev_2 = vx_setzero_f64(),
                          prev_3 = vx_setzero_f64();
                v_int32 prev_sq_1 = vx_setzero_s32(), prev_sq_2 = vx_setzero_s32(),
                        prev_sq_3 = vx_setzero_s32();
                int j = 0;
                const int j_max =
                        ((_srcstep * i + (width - VTraits<v_uint16>::vlanes() * cn + VTraits<v_uint8>::vlanes() * cn)) >= _srcstep * height)
                        ? width - VTraits<v_uint8>::vlanes() * cn    // uint8 in v_load_deinterleave()
                        : width - VTraits<v_uint16>::vlanes() * cn;  // v_expand_low
                for ( ; j <= j_max; j += VTraits<v_uint16>::vlanes() * cn)
                {
                    v_uint8 v_src_row_1, v_src_row_2, v_src_row_3;
                    v_load_deinterleave(src_row + j, v_src_row_1, v_src_row_2, v_src_row_3);
                    v_int16 el8_1 = v_reinterpret_as_s16(v_expand_low(v_src_row_1));
                    v_int16 el8_2 = v_reinterpret_as_s16(v_expand_low(v_src_row_2));
                    v_int16 el8_3 = v_reinterpret_as_s16(v_expand_low(v_src_row_3));

                    if (sqsum)
                    {
                        v_int32 sl1, sh1, sl2, sh2, sl3, sh3;
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_1), prev_sq_1, sl1, sh1);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_2), prev_sq_2, sl2, sh2);
                        v_sqsum_prefix(v_reinterpret_as_u16(el8_3), prev_sq_3, sl3, sh3);
                        const int vl = VTraits<v_int32>::vlanes();
                        v_store_interleave(sq_cache         , sl1, sl2, sl3);
                        v_store_interleave(sq_cache + vl * 3, sh1, sh2, sh3);
                        v_int32 z0 = vx_load(sq_cache         ), z1 = vx_load(sq_cache + vl    );
                        v_int32 z2 = vx_load(sq_cache + vl * 2), z3 = vx_load(sq_cache + vl * 3);
                        v_int32 z4 = vx_load(sq_cache + vl * 4), z5 = vx_load(sq_cache + vl * 5);
                        v_store_sqsum_block(sqsum_row + j         , prev_sqsum_row + j         , z0, z1);
                        v_store_sqsum_block(sqsum_row + j + 2 * vl, prev_sqsum_row + j + 2 * vl, z2, z3);
                        v_store_sqsum_block(sqsum_row + j + 4 * vl, prev_sqsum_row + j + 4 * vl, z4, z5);
                    }

                    v_float64 el4ll_1, el4lh_1, el4hl_1, el4hh_1, el4ll_2, el4lh_2, el4hl_2, el4hh_2, el4ll_3, el4lh_3, el4hl_3, el4hh_3;
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum_1 = _mm256_add_epi16(el8_1.val, _mm256_slli_si256(el8_1.val, 2));
                    __m256i vsum_2 = _mm256_add_epi16(el8_2.val, _mm256_slli_si256(el8_2.val, 2));
                    __m256i vsum_3 = _mm256_add_epi16(el8_3.val, _mm256_slli_si256(el8_3.val, 2));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 4));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 4));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 4));
                    vsum_1 = _mm256_add_epi16(vsum_1, _mm256_slli_si256(vsum_1, 8));
                    vsum_2 = _mm256_add_epi16(vsum_2, _mm256_slli_si256(vsum_2, 8));
                    vsum_3 = _mm256_add_epi16(vsum_3, _mm256_slli_si256(vsum_3, 8));
                    __m256i el4l1_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_1));
                    __m256i el4l2_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_2));
                    __m256i el4l3_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum_3));
                    __m256i el4h1_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_1));
                    __m256i el4h2_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_2));
                    __m256i el4h3_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum_3));
                    el4ll_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l1_32)), prev_1.val);
                    el4ll_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l2_32)), prev_2.val);
                    el4ll_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l3_32)), prev_3.val);
                    el4lh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l1_32)), prev_1.val);
                    el4lh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l2_32)), prev_2.val);
                    el4lh_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l3_32)), prev_3.val);
                    __m256d el4d_1 = _mm256_permute4x64_pd(el4lh_1.val, 0xff);
                    __m256d el4d_2 = _mm256_permute4x64_pd(el4lh_2.val, 0xff);
                    __m256d el4d_3 = _mm256_permute4x64_pd(el4lh_3.val, 0xff);
                    el4hl_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h1_32)), el4d_1);
                    el4hl_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h2_32)), el4d_2);
                    el4hl_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h3_32)), el4d_3);
                    el4hh_1.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h1_32)), el4d_1);
                    el4hh_2.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h2_32)), el4d_2);
                    el4hh_3.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h3_32)), el4d_3);
                    prev_1.val = _mm256_permute4x64_pd(el4hh_1.val, 0xff);
                    prev_2.val = _mm256_permute4x64_pd(el4hh_2.val, 0xff);
                    prev_3.val = _mm256_permute4x64_pd(el4hh_3.val, 0xff);
#else
                    el8_1 = v_add(el8_1, v_rotate_left<1>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<1>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<1>(el8_3));
                    el8_1 = v_add(el8_1, v_rotate_left<2>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<2>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<2>(el8_3));
#if CV_SIMD_WIDTH >= 32
                    el8_1 = v_add(el8_1, v_rotate_left<4>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<4>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<4>(el8_3));
#if CV_SIMD_WIDTH == 64
                    el8_1 = v_add(el8_1, v_rotate_left<8>(el8_1));
                    el8_2 = v_add(el8_2, v_rotate_left<8>(el8_2));
                    el8_3 = v_add(el8_3, v_rotate_left<8>(el8_3));
#endif
#endif
                    v_int32 el4li_1, el4hi_1, el4li_2, el4hi_2, el4li_3, el4hi_3;
                    v_expand(el8_1, el4li_1, el4hi_1);
                    v_expand(el8_2, el4li_2, el4hi_2);
                    v_expand(el8_3, el4li_3, el4hi_3);
                    el4ll_1 = v_add(v_cvt_f64(el4li_1), prev_1);
                    el4ll_2 = v_add(v_cvt_f64(el4li_2), prev_2);
                    el4ll_3 = v_add(v_cvt_f64(el4li_3), prev_3);
                    el4lh_1 = v_add(v_cvt_f64_high(el4li_1), prev_1);
                    el4lh_2 = v_add(v_cvt_f64_high(el4li_2), prev_2);
                    el4lh_3 = v_add(v_cvt_f64_high(el4li_3), prev_3);
                    el4hl_1 = v_add(v_cvt_f64(el4hi_1), el4ll_1);
                    el4hl_2 = v_add(v_cvt_f64(el4hi_2), el4ll_2);
                    el4hl_3 = v_add(v_cvt_f64(el4hi_3), el4ll_3);
                    el4hh_1 = v_add(v_cvt_f64_high(el4hi_1), el4lh_1);
                    el4hh_2 = v_add(v_cvt_f64_high(el4hi_2), el4lh_2);
                    el4hh_3 = v_add(v_cvt_f64_high(el4hi_3), el4lh_3);
                    prev_1 = vx_setall_f64(v_extract_highest(el4hh_1));
                    prev_2 = vx_setall_f64(v_extract_highest(el4hh_2));
                    prev_3 = vx_setall_f64(v_extract_highest(el4hh_3));
//                    prev_1 = v_broadcast_highest(el4hh_1);
//                    prev_2 = v_broadcast_highest(el4hh_2);
//                    prev_3 = v_broadcast_highest(el4hh_3);
#endif
                    v_store_interleave(row_cache                        , el4ll_1, el4ll_2, el4ll_3);
                    v_store_interleave(row_cache + VTraits<v_float64>::vlanes() * 3, el4lh_1, el4lh_2, el4lh_3);
                    v_store_interleave(row_cache + VTraits<v_float64>::vlanes() * 6, el4hl_1, el4hl_2, el4hl_3);
                    v_store_interleave(row_cache + VTraits<v_float64>::vlanes() * 9, el4hh_1, el4hh_2, el4hh_3);
                    el4ll_1 = vx_load(row_cache                         );
                    el4ll_2 = vx_load(row_cache + VTraits<v_float64>::vlanes()     );
                    el4ll_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 2 );
                    el4lh_1 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 3 );
                    el4lh_2 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 4 );
                    el4lh_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 5 );
                    el4hl_1 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 6 );
                    el4hl_2 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 7 );
                    el4hl_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 8 );
                    el4hh_1 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 9 );
                    el4hh_2 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 10);
                    el4hh_3 = vx_load(row_cache + VTraits<v_float64>::vlanes() * 11);
                    v_store(sum_row + j                         , v_add(el4ll_1, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()     , v_add(el4ll_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2 , v_add(el4ll_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3 , v_add(el4lh_1, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 4 , v_add(el4lh_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 4)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 5 , v_add(el4lh_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 5)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 6 , v_add(el4hl_1, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 6)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 7 , v_add(el4hl_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 7)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 8 , v_add(el4hl_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 8)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 9 , v_add(el4hh_1, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 9)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 10, v_add(el4hh_2, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 10)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 11, v_add(el4hh_3, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 11)));
                }

                int jt = j;
                for (double v3 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                            v2 = sum_row[jt - 2] - prev_sum_row[jt - 2],
                            v1 = sum_row[jt - 3] - prev_sum_row[jt - 3]; jt < width; jt += 3)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                    sum_row[jt + 2] = (v3 += src_row[jt + 2]) + prev_sum_row[jt + 2];
                }
                if (sqsum)
                {
                    double s3 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    double s2 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    double s1 = sqsum_row[j - 3] - prev_sqsum_row[j - 3];
                    for ( ; j < width; j += 3)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1], i3 = src_row[j + 2];
                        s1 += (double)i1 * i1; s2 += (double)i2 * i2; s3 += (double)i3 * i3;
                        sqsum_row[j]     = s1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = s2 + prev_sqsum_row[j + 1];
                        sqsum_row[j + 2] = s3 + prev_sqsum_row[j + 2];
                    }
                }
            }
        }
        else if (cn == 4)
        {
            // the others
            for (int i = 0; i < height; ++i)
            {
                const uchar * src_row = src + _srcstep * i;
                double * prev_sum_row = (double *)((uchar *)sum + _sumstep * i) + cn;
                double * sum_row = (double *)((uchar *)sum + _sumstep * (i + 1)) + cn;

                sum_row[-1] = sum_row[-2] = sum_row[-3] = sum_row[-4] = 0;

                double * prev_sqsum_row = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * i) + cn : 0;
                double * sqsum_row      = sqsum ? (double *)((uchar *)sqsum + _sqsumstep * (i + 1)) + cn : 0;
                if (sqsum)
                    sqsum_row[-1] = sqsum_row[-2] = sqsum_row[-3] = sqsum_row[-4] = 0;

                v_float64 prev_1 = vx_setzero_f64(), prev_2 = vx_setzero_f64();
#if CV_SIMD_WIDTH == 16
                v_int32 prev_sq = vx_setzero_s32();
#endif
                int j = 0;
                for ( ; j + VTraits<v_uint16>::vlanes() <= width; j += VTraits<v_uint16>::vlanes())
                {
                    v_uint16 px = vx_load_expand(src_row + j);
                    v_int16 el8 = v_reinterpret_as_s16(px);
                    v_float64 el4ll, el4lh, el4hl, el4hh;
#if CV_SIMD_WIDTH == 16
                    if (sqsum)
                    {
                        v_uint32 sq_lo_u, sq_hi_u;
                        v_mul_expand(px, px, sq_lo_u, sq_hi_u);
                        v_int32 sq_l = v_reinterpret_as_s32(sq_lo_u);
                        v_int32 sq_h = v_reinterpret_as_s32(sq_hi_u);
                        sq_l = v_add(sq_l, prev_sq);
                        sq_h = v_add(sq_h, sq_l);
                        prev_sq = sq_h;
                        v_store_sqsum_block(sqsum_row + j, prev_sqsum_row + j, sq_l, sq_h);
                    }
#endif
#if CV_AVX2 && CV_SIMD_WIDTH == 32
                    __m256i vsum = _mm256_add_epi16(el8.val, _mm256_slli_si256(el8.val, 8));
                    __m256i el4l_32 = _mm256_cvtepi16_epi32(_v256_extract_low(vsum));
                    __m256i el4h_32 = _mm256_cvtepi16_epi32(_v256_extract_high(vsum));
                    el4ll.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4l_32)), prev_1.val);
                    el4lh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4l_32)), prev_2.val);
                    el4hl.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_low(el4h_32)), el4lh.val);
                    el4hh.val = _mm256_add_pd(_mm256_cvtepi32_pd(_v256_extract_high(el4h_32)), el4lh.val);
                    prev_1.val = prev_2.val = el4hh.val;
#else
#if CV_SIMD_WIDTH >= 32
                    el8 = v_add(el8, v_rotate_left<4>(el8));
#if CV_SIMD_WIDTH == 64
                    el8 = v_add(el8, v_rotate_left<8>(el8));
#endif
#endif
                    v_int32 el4li, el4hi;
                    v_expand(el8, el4li, el4hi);
                    el4ll = v_add(v_cvt_f64(el4li), prev_1);
                    el4lh = v_add(v_cvt_f64_high(el4li), prev_2);
                    el4hl = v_add(v_cvt_f64(el4hi), el4ll);
                    el4hh = v_add(v_cvt_f64_high(el4hi), el4lh);
#if CV_SIMD_WIDTH == 16
                    prev_1 = el4hl;
                    prev_2 = el4hh;
#elif CV_SIMD_WIDTH == 32
                    prev_1 = prev_2 = el4hh;
#else
                    prev_1 = prev_2 = v_combine_high(el4hh, el4hh);
#endif
#endif
                    v_store(sum_row + j                        , v_add(el4ll, vx_load(prev_sum_row + j)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes()    , v_add(el4lh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes())));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 2, v_add(el4hl, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 2)));
                    v_store(sum_row + j + VTraits<v_float64>::vlanes() * 3, v_add(el4hh, vx_load(prev_sum_row + j + VTraits<v_float64>::vlanes() * 3)));
                }

                int jt = j;
                for (double v4 = sum_row[jt - 1] - prev_sum_row[jt - 1],
                            v3 = sum_row[jt - 2] - prev_sum_row[jt - 2],
                            v2 = sum_row[jt - 3] - prev_sum_row[jt - 3],
                            v1 = sum_row[jt - 4] - prev_sum_row[jt - 4]; jt < width; jt += 4)
                {
                    sum_row[jt]     = (v1 += src_row[jt])     + prev_sum_row[jt];
                    sum_row[jt + 1] = (v2 += src_row[jt + 1]) + prev_sum_row[jt + 1];
                    sum_row[jt + 2] = (v3 += src_row[jt + 2]) + prev_sum_row[jt + 2];
                    sum_row[jt + 3] = (v4 += src_row[jt + 3]) + prev_sum_row[jt + 3];
                }
                if (sqsum)
                {
                    double s4 = sqsum_row[j - 1] - prev_sqsum_row[j - 1];
                    double s3 = sqsum_row[j - 2] - prev_sqsum_row[j - 2];
                    double s2 = sqsum_row[j - 3] - prev_sqsum_row[j - 3];
                    double s1 = sqsum_row[j - 4] - prev_sqsum_row[j - 4];
                    for ( ; j < width; j += 4)
                    {
                        int i1 = src_row[j], i2 = src_row[j + 1], i3 = src_row[j + 2], i4 = src_row[j + 3];
                        s1 += (double)i1 * i1; s2 += (double)i2 * i2; s3 += (double)i3 * i3; s4 += (double)i4 * i4;
                        sqsum_row[j]     = s1 + prev_sqsum_row[j];
                        sqsum_row[j + 1] = s2 + prev_sqsum_row[j + 1];
                        sqsum_row[j + 2] = s3 + prev_sqsum_row[j + 2];
                        sqsum_row[j + 3] = s4 + prev_sqsum_row[j + 3];
                    }
                }
            }
        }
        else
        {
            return false;
        }
        vx_cleanup();

        return true;
    }
};
#endif

#endif

} // namespace anon

bool integral_SIMD(
        int depth, int sdepth, int sqdepth,
        const uchar* src, size_t srcstep,
        uchar* sum, size_t sumstep,
        uchar* sqsum, size_t sqsumstep,
        uchar* tilted, size_t tstep,
        int width, int height, int cn)
{
    CV_INSTRUMENT_REGION();

#define ONE_CALL(T, ST, QT) \
    return Integral_SIMD<T, ST, QT>()((const T*)src, srcstep, (ST*)sum, sumstep, (QT*)sqsum, sqsumstep, (ST*)tilted, tstep, width, height, cn)

    if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_64F )
#if CV_SIMD_64F
        ONE_CALL(uchar, int, double);   // f64 sqsum: needs 64-bit-float SIMD
#else
        return false;                   // e.g. 32-bit ARMv7 NEON -> scalar path
#endif
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32F )
        ONE_CALL(uchar, int, float);
    else if( depth == CV_8U && sdepth == CV_32S && sqdepth == CV_32S )
        ONE_CALL(uchar, int, int);
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_64F )
#if CV_SIMD_64F
        ONE_CALL(uchar, float, double);
#else
        return false;
#endif
    else if( depth == CV_8U && sdepth == CV_32F && sqdepth == CV_32F )
        ONE_CALL(uchar, float, float);
    else if( depth == CV_8U && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(uchar, double, double);
    else if( depth == CV_16U && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(ushort, double, double);
    else if( depth == CV_16S && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(short, double, double);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_64F )
        ONE_CALL(float, float, double);
    else if( depth == CV_32F && sdepth == CV_32F && sqdepth == CV_32F )
        ONE_CALL(float, float, float);
    else if( depth == CV_32F && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(float, double, double);
    else if( depth == CV_64F && sdepth == CV_64F && sqdepth == CV_64F )
        ONE_CALL(double, double, double);
    else
        return false;

#undef ONE_CALL
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
}} // cv::hal::
