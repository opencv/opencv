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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2014-2015, Itseez Inc., all rights reserved.
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

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"
#include "corner.hpp"

namespace cv
{

// load three 8-packed float vector and deinterleave
// probably it's better to write down somewhere else
static void load_deinterleave(const float* ptr, __m256& a, __m256& b, __m256& c)
{
    __m256 s0 = _mm256_loadu_ps(ptr);                    // a0, b0, c0, a1, b1, c1, a2, b2,
    __m256 s1 = _mm256_loadu_ps(ptr + 8);                // c2, a3, b3, c3, a4, b4, c4, a5,
    __m256 s2 = _mm256_loadu_ps(ptr + 16);               // b5, c5, a6, b6, c6, a7, b7, c7,
    __m256 s3 = _mm256_permute2f128_ps(s1, s2, 0x21);    // a4, b4, c4, a5, b5, c5, a6, b6,
    __m256 s4 = _mm256_permute2f128_ps(s2, s2, 0x33);    // c6, a7, b7, c7, c6, a7, b7, c7,

    __m256 v00 = _mm256_unpacklo_ps(s0, s3);             // a0, a4, b0, b4, b1, b5, c1, c5,
    __m256 v01 = _mm256_unpackhi_ps(s0, s3);             // c0, c4, a1, a5, a2, a6, b2, b6,
    __m256 v02 = _mm256_unpacklo_ps(s1, s4);             // c2, c6, a3, a7, x,  x,  x,  x,
    __m256 v03 = _mm256_unpackhi_ps(s1, s4);             // b3, b7, c3, c7, x,  x,  x,  x,
    __m256 v04 = _mm256_permute2f128_ps(v02, v03, 0x20); // c2, c6, a3, a7, b3, b7, c3, c7,
    __m256 v05 = _mm256_permute2f128_ps(v01, v03, 0x21); // a2, a6, b2, b6, b3, b7, c3, c7,

    __m256 v10 = _mm256_unpacklo_ps(v00, v05);           // a0, a2, a4, a6, b1, b3, b5, b7,
    __m256 v11 = _mm256_unpackhi_ps(v00, v05);           // b0, b2, b4, b6, c1, c3, c5, c7,
    __m256 v12 = _mm256_unpacklo_ps(v01, v04);           // c0, c2, c4, c6, x,  x,  x,  x,
    __m256 v13 = _mm256_unpackhi_ps(v01, v04);           // a1, a3, a5, a7, x,  x,  x,  x,
    __m256 v14 = _mm256_permute2f128_ps(v11, v12, 0x20); // b0, b2, b4, b6, c0, c2, c4, c6,
    __m256 v15 = _mm256_permute2f128_ps(v10, v11, 0x31); // b1, b3, b5, b7, c1, c3, c5, c7,

    __m256 v20 = _mm256_unpacklo_ps(v14, v15);           // b0, b1, b2, b3, c0, c1, c2, c3,
    __m256 v21 = _mm256_unpackhi_ps(v14, v15);           // b4, b5, b6, b7, c4, c5, c6, c7,
    __m256 v22 = _mm256_unpacklo_ps(v10, v13);           // a0, a1, a2, a3, x,  x,  x,  x,
    __m256 v23 = _mm256_unpackhi_ps(v10, v13);           // a4, a5, a6, a7, x,  x,  x,  x,

    a = _mm256_permute2f128_ps(v22, v23, 0x20);          // a0, a1, a2, a3, a4, a5, a6, a7,
    b = _mm256_permute2f128_ps(v20, v21, 0x20);          // b0, b1, b2, b3, b4, b5, b6, b7,
    c = _mm256_permute2f128_ps(v20, v21, 0x31);          // c0, c1, c2, c3, c4, c5, c6, c7,
}

// realign four 3-packed vector to three 4-packed vector
static void v_pack4x3to3x4(const __m128i& s0, const __m128i& s1, const __m128i& s2, const __m128i& s3, __m128i& d0, __m128i& d1, __m128i& d2)
{
    d0 = _mm_or_si128(s0, _mm_slli_si128(s1, 12));
    d1 = _mm_or_si128(_mm_srli_si128(s1, 4), _mm_slli_si128(s2, 8));
    d2 = _mm_or_si128(_mm_srli_si128(s2, 8), _mm_slli_si128(s3, 4));
}

// separate high and low 128 bit and cast to __m128i
static void v_separate_lo_hi(const __m256& src, __m128i& lo, __m128i& hi)
{
    lo = _mm_castps_si128(_mm256_castps256_ps128(src));
    hi = _mm_castps_si128(_mm256_extractf128_ps(src, 1));
}

// interleave three 8-float vector and store
static void store_interleave(float* ptr, const __m256& a, const __m256& b, const __m256& c)
{
    __m128i a0, a1, b0, b1, c0, c1;
    v_separate_lo_hi(a, a0, a1);
    v_separate_lo_hi(b, b0, b1);
    v_separate_lo_hi(c, c0, c1);

    v_uint32x4 z = v_setzero_u32();
    v_uint32x4 u0, u1, u2, u3;
    v_transpose4x4(v_uint32x4(a0), v_uint32x4(b0), v_uint32x4(c0), z, u0, u1, u2, u3);
    v_pack4x3to3x4(u0.val, u1.val, u2.val, u3.val, a0, b0, c0);
    v_transpose4x4(v_uint32x4(a1), v_uint32x4(b1), v_uint32x4(c1), z, u0, u1, u2, u3);
    v_pack4x3to3x4(u0.val, u1.val, u2.val, u3.val, a1, b1, c1);

#if !defined(__GNUC__) || defined(__INTEL_COMPILER)
    _mm256_storeu_ps(ptr, _mm256_setr_m128(_mm_castsi128_ps(a0), _mm_castsi128_ps(b0)));
    _mm256_storeu_ps(ptr + 8, _mm256_setr_m128(_mm_castsi128_ps(c0), _mm_castsi128_ps(a1)));
    _mm256_storeu_ps(ptr + 16,  _mm256_setr_m128(_mm_castsi128_ps(b1), _mm_castsi128_ps(c1)));
#else
    // GCC: workaround for missing AVX intrinsic: "_mm256_setr_m128()"
    _mm256_storeu_ps(ptr, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(a0)), _mm_castsi128_ps(b0), 1));
    _mm256_storeu_ps(ptr + 8, _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(c0)), _mm_castsi128_ps(a1), 1));
    _mm256_storeu_ps(ptr + 16,  _mm256_insertf128_ps(_mm256_castps128_ps256(_mm_castsi128_ps(b1)), _mm_castsi128_ps(c1), 1));
#endif
}

int calcMinEigenValLine_AVX(const float* cov, float* dst, int width)
{
    int j = 0;
    __m256 half = _mm256_set1_ps(0.5f);
    for (; j <= width - 8; j += 8)
    {
        __m256 v_a, v_b, v_c, v_t;
        load_deinterleave(cov + j * 3, v_a, v_b, v_c);
        v_a = _mm256_mul_ps(v_a, half);
        v_c = _mm256_mul_ps(v_c, half);
        v_t = _mm256_sub_ps(v_a, v_c);
        v_t = _mm256_add_ps(_mm256_mul_ps(v_b, v_b), _mm256_mul_ps(v_t, v_t));
        _mm256_storeu_ps(dst + j, _mm256_sub_ps(_mm256_add_ps(v_a, v_c), _mm256_sqrt_ps(v_t)));
    }
    return j;
}

int calcHarrisLine_AVX(const float* cov, float* dst, double k, int width)
{
    int j = 0;
    __m256 v_k = _mm256_set1_ps((float)k);

    for (; j <= width - 8; j += 8)
    {
        __m256 v_a, v_b, v_c;
        load_deinterleave(cov + j * 3, v_a, v_b, v_c);

        __m256 v_ac_bb = _mm256_sub_ps(_mm256_mul_ps(v_a, v_c), _mm256_mul_ps(v_b, v_b));
        __m256 v_ac = _mm256_add_ps(v_a, v_c);
        __m256 v_dst = _mm256_sub_ps(v_ac_bb, _mm256_mul_ps(v_k, _mm256_mul_ps(v_ac, v_ac)));
        _mm256_storeu_ps(dst + j, v_dst);
    }
    return j;
}

int cornerEigenValsVecsLine_AVX(const float* dxdata, const float* dydata, float* cov_data, int width)
{
    int j = 0;
    for (; j <= width - 8; j += 8)
    {
        __m256 v_dx = _mm256_loadu_ps(dxdata + j);
        __m256 v_dy = _mm256_loadu_ps(dydata + j);

        __m256 v_dst0, v_dst1, v_dst2;
        v_dst0 = _mm256_mul_ps(v_dx, v_dx);
        v_dst1 = _mm256_mul_ps(v_dx, v_dy);
        v_dst2 = _mm256_mul_ps(v_dy, v_dy);

        store_interleave(cov_data + j * 3, v_dst0, v_dst1, v_dst2);
    }
    return j;
}

}
/* End of file */
