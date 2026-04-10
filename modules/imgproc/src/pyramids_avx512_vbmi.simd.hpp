// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "opencv2/core/hal/intrin.hpp"

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

// AVX-512 VBMI vpermb-based PyrDownVecH for uchar->ushort.
int PyrDownVecH_uchar_ushort_1_vbmi(const uchar* src, ushort* row, int width);
int PyrDownVecH_uchar_ushort_2_vbmi(const uchar* src, ushort* row, int width);
int PyrDownVecH_uchar_ushort_3_vbmi(const uchar* src, ushort* row, int width);
int PyrDownVecH_uchar_ushort_4_vbmi(const uchar* src, ushort* row, int width);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

int PyrDownVecH_uchar_ushort_1_vbmi(const uchar* src, ushort* row, int width)
{
    int x = 0;

#if !CV_AVX_512VBMI
    CV_UNUSED(src); CV_UNUSED(row); CV_UNUSED(width);
#else
    // cn=1: 32 output pixels per iteration.
    // Source stride = 2 bytes/pixel. Tap k for pixel i: src[2*i + k], k=0..4.
    // Max source byte: 2*31 + 4 = 66. Fits in 2 zmm (128 bytes).
    __m512i vidx0 = _v512_set_epu8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        62,60,58,56,54,52,50,48,46,44,42,40,38,36,34,32,
        30,28,26,24,22,20,18,16,14,12,10, 8, 6, 4, 2, 0);
    __m512i vidx1 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(1));
    __m512i vidx2 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(2));
    __m512i vidx3 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(3));
    __m512i vidx4 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(4));
    __m512i v6 = _mm512_set1_epi16(6);
    for (; x <= width - 32; x += 32, src += 64, row += 32)
    {
        __m512i sA = _mm512_loadu_si512(src);
        __m512i sB = _mm512_loadu_si512(src + 64);

        __m512i t0 = _mm512_permutex2var_epi8(sA, vidx0, sB);
        __m512i t1 = _mm512_permutex2var_epi8(sA, vidx1, sB);
        __m512i t2 = _mm512_permutex2var_epi8(sA, vidx2, sB);
        __m512i t3 = _mm512_permutex2var_epi8(sA, vidx3, sB);
        __m512i t4 = _mm512_permutex2var_epi8(sA, vidx4, sB);

        __m512i w0 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t0));
        __m512i w1 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t1));
        __m512i w2 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t2));
        __m512i w3 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t3));
        __m512i w4 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t4));

        __m512i res = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mullo_epi16(w2, v6),
                             _mm512_slli_epi16(_mm512_add_epi16(w1, w3), 2)),
            _mm512_add_epi16(w0, w4));

        _mm512_storeu_si512(row, res);
    }
    _mm256_zeroupper();
#endif
    return x;
}

int PyrDownVecH_uchar_ushort_2_vbmi(const uchar* src, ushort* row, int width)
{
    int x = 0;

#if !CV_AVX_512VBMI
    CV_UNUSED(src); CV_UNUSED(row); CV_UNUSED(width);
#else
    // cn=2: 16 output pixels x 2 channels = 32 output ushorts per iteration.
    // Source stride = 4 bytes/pixel. Tap k for pixel i, ch j: src[4*i + 2*k + j].
    // Max source byte: 4*15 + 8 + 1 = 69. Fits in 2 zmm.
    __m512i vidx0 = _v512_set_epu8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        61,60,57,56,53,52,49,48,45,44,41,40,37,36,33,32,
        29,28,25,24,21,20,17,16,13,12, 9, 8, 5, 4, 1, 0);
    __m512i vidx1 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(2));
    __m512i vidx2 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(4));
    __m512i vidx3 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(6));
    __m512i vidx4 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(8));
    __m512i v6 = _mm512_set1_epi16(6);

    for (; x <= width - 32; x += 32, src += 64, row += 32)
    {
        __m512i sA = _mm512_loadu_si512(src);
        __m512i sB = _mm512_loadu_si512(src + 64);

        __m512i t0 = _mm512_permutex2var_epi8(sA, vidx0, sB);
        __m512i t1 = _mm512_permutex2var_epi8(sA, vidx1, sB);
        __m512i t2 = _mm512_permutex2var_epi8(sA, vidx2, sB);
        __m512i t3 = _mm512_permutex2var_epi8(sA, vidx3, sB);
        __m512i t4 = _mm512_permutex2var_epi8(sA, vidx4, sB);

        __m512i w0 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t0));
        __m512i w1 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t1));
        __m512i w2 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t2));
        __m512i w3 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t3));
        __m512i w4 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t4));

        __m512i res = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mullo_epi16(w2, v6),
                             _mm512_slli_epi16(_mm512_add_epi16(w1, w3), 2)),
            _mm512_add_epi16(w0, w4));

        _mm512_storeu_si512(row, res);
    }
    _mm256_zeroupper();
#endif
    return x;
}

int PyrDownVecH_uchar_ushort_3_vbmi(const uchar* src, ushort* row, int width)
{
    int x = 0;

#if !CV_AVX_512VBMI
    CV_UNUSED(src); CV_UNUSED(row); CV_UNUSED(width);
#else
    // Each iteration: 16 output pixels × 3 channels = 48 output ushorts.
    // Source span: 16×6 + 4*3 = 108 bytes. Load 128 contiguous bytes in 2 zmm.
    __m512i vidx0 = _v512_set_epu8(
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        92,91,90, 86,85,84, 80,79,78, 74,73,72, 68,67,66, 62,
        61,60, 56,55,54, 50,49,48, 44,43,42, 38,37,36, 32,31,
        30, 26,25,24, 20,19,18, 14,13,12,  8, 7, 6,  2, 1, 0);
    __m512i vidx1 = _v512_set_epu8(
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        95,94,93, 89,88,87, 83,82,81, 77,76,75, 71,70,69, 65,
        64,63, 59,58,57, 53,52,51, 47,46,45, 41,40,39, 35,34,
        33, 29,28,27, 23,22,21, 17,16,15, 11,10, 9,  5, 4, 3);
    __m512i vidx2 = _v512_set_epu8(
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        98,97,96, 92,91,90, 86,85,84, 80,79,78, 74,73,72, 68,
        67,66, 62,61,60, 56,55,54, 50,49,48, 44,43,42, 38,37,
        36, 32,31,30, 26,25,24, 20,19,18, 14,13,12,  8, 7, 6);
    __m512i vidx3 = _v512_set_epu8(
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       101,100,99, 95,94,93, 89,88,87, 83,82,81, 77,76,75, 71,
        70,69, 65,64,63, 59,58,57, 53,52,51, 47,46,45, 41,40,
        39, 35,34,33, 29,28,27, 23,22,21, 17,16,15, 11,10, 9);
    __m512i vidx4 = _v512_set_epu8(
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
       104,103,102, 98,97,96, 92,91,90, 86,85,84, 80,79,78, 74,
        73,72, 68,67,66, 62,61,60, 56,55,54, 50,49,48, 44,43,
        42, 38,37,36, 32,31,30, 26,25,24, 20,19,18, 14,13,12);

    __m512i v6 = _mm512_set1_epi16(6);

    //16 output pixels × 3 channels
    for (; x <= width - 48; x += 48, src += 96, row += 48)
    {
        __m512i sA = _mm512_loadu_si512(src);
        __m512i sB = _mm512_loadu_si512(src + 64);

        __m512i t0 = _mm512_permutex2var_epi8(sA, vidx0, sB);
        __m512i t1 = _mm512_permutex2var_epi8(sA, vidx1, sB);
        __m512i t2 = _mm512_permutex2var_epi8(sA, vidx2, sB);
        __m512i t3 = _mm512_permutex2var_epi8(sA, vidx3, sB);
        __m512i t4 = _mm512_permutex2var_epi8(sA, vidx4, sB);

        // Expand lower 32 bytes of each tap to u16 (first 32 of 48 bytes)
        __m512i t0_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t0));
        __m512i t1_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t1));
        __m512i t2_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t2));
        __m512i t3_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t3));
        __m512i t4_lo = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t4));

        // Expand upper 16 bytes (bytes 32..47) to u16
        __m256i t0_hi = _mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(t0, 2));
        __m256i t1_hi = _mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(t1, 2));
        __m256i t2_hi = _mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(t2, 2));
        __m256i t3_hi = _mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(t3, 2));
        __m256i t4_hi = _mm256_cvtepu8_epi16(_mm512_extracti32x4_epi32(t4, 2));

        // Compute low half (32xu16 values):
        __m512i sum13_lo = _mm512_add_epi16(t1_lo, t3_lo);
        __m512i res_lo   = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mullo_epi16(t2_lo, v6),
                             _mm512_slli_epi16(sum13_lo, 2)),
            _mm512_add_epi16(t0_lo, t4_lo));

        // Compute high half (16xu16 values):
        __m256i v6_256   = _mm256_set1_epi16(6);
        __m256i sum13_hi = _mm256_add_epi16(t1_hi, t3_hi);
        __m256i res_hi   = _mm256_add_epi16(
            _mm256_add_epi16(_mm256_mullo_epi16(t2_hi, v6_256),
                             _mm256_slli_epi16(sum13_hi, 2)),
            _mm256_add_epi16(t0_hi, t4_hi));

        _mm512_storeu_si512(row, res_lo); //32 ushort
        _mm256_storeu_si256((__m256i*)(row + 32), res_hi); //16 ushort
    }
    _mm256_zeroupper();
#endif // CV_AVX_512VBMI

    return x;
}

int PyrDownVecH_uchar_ushort_4_vbmi(const uchar* src, ushort* row, int width)
{
    int x = 0;

#if !CV_AVX_512VBMI
    CV_UNUSED(src); CV_UNUSED(row); CV_UNUSED(width);
#else
    // cn=4: 8 output pixels x 4 channels = 32 output ushorts per iteration.
    // Source stride = 8 bytes/pixel. Tap k for pixel i, ch j: src[8*i + 4*k + j].
    // Max source byte: 8*7 + 4*4 + 3 = 75. Fits in 2 zmm.
    __m512i vidx0 = _v512_set_epu8(
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        59,58,57,56,51,50,49,48,43,42,41,40,35,34,33,32,
        27,26,25,24,19,18,17,16,11,10, 9, 8, 3, 2, 1, 0);
    __m512i vidx1 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(4));
    __m512i vidx2 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(8));
    __m512i vidx3 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(12));
    __m512i vidx4 = _mm512_add_epi8(vidx0, _mm512_set1_epi8(16));

    __m512i v6 = _mm512_set1_epi16(6);

    for (; x <= width - 32; x += 32, src += 64, row += 32)
    {
        __m512i sA = _mm512_loadu_si512(src);
        __m512i sB = _mm512_loadu_si512(src + 64);

        __m512i t0 = _mm512_permutex2var_epi8(sA, vidx0, sB);
        __m512i t1 = _mm512_permutex2var_epi8(sA, vidx1, sB);
        __m512i t2 = _mm512_permutex2var_epi8(sA, vidx2, sB);
        __m512i t3 = _mm512_permutex2var_epi8(sA, vidx3, sB);
        __m512i t4 = _mm512_permutex2var_epi8(sA, vidx4, sB);

        __m512i w0 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t0));
        __m512i w1 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t1));
        __m512i w2 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t2));
        __m512i w3 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t3));
        __m512i w4 = _mm512_cvtepu8_epi16(_mm512_castsi512_si256(t4));

        __m512i res = _mm512_add_epi16(
            _mm512_add_epi16(_mm512_mullo_epi16(w2, v6),
                             _mm512_slli_epi16(_mm512_add_epi16(w1, w3), 2)),
            _mm512_add_epi16(w0, w4));

        _mm512_storeu_si512(row, res);
    }
    _mm256_zeroupper();
#endif
    return x;
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
