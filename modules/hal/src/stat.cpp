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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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

namespace cv { namespace hal {

static const uchar popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

static const uchar popCountTable2[] =
{
    0, 1, 1, 1, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4,
    2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4, 2, 3, 3, 3, 3, 4, 4, 4, 3, 4, 4, 4, 3, 4, 4, 4
};

static const uchar popCountTable4[] =
{
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
};

int normHamming(const uchar* a, int n)
{
    int i = 0;
    int result = 0;
#if CV_NEON
    {
        uint32x4_t bits = vmovq_n_u32(0);
        for (; i <= n - 16; i += 16) {
            uint8x16_t A_vec = vld1q_u8 (a + i);
            uint8x16_t bitsSet = vcntq_u8 (A_vec);
            uint16x8_t bitSet8 = vpaddlq_u8 (bitsSet);
            uint32x4_t bitSet4 = vpaddlq_u16 (bitSet8);
            bits = vaddq_u32(bits, bitSet4);
        }
        uint64x2_t bitSet2 = vpaddlq_u32 (bits);
        result = vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),0);
        result += vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),2);
    }
#endif
        for( ; i <= n - 4; i += 4 )
            result += popCountTable[a[i]] + popCountTable[a[i+1]] +
            popCountTable[a[i+2]] + popCountTable[a[i+3]];
    for( ; i < n; i++ )
        result += popCountTable[a[i]];
    return result;
}

int normHamming(const uchar* a, const uchar* b, int n)
{
    int i = 0;
    int result = 0;
#if CV_NEON
    {
        uint32x4_t bits = vmovq_n_u32(0);
        for (; i <= n - 16; i += 16) {
            uint8x16_t A_vec = vld1q_u8 (a + i);
            uint8x16_t B_vec = vld1q_u8 (b + i);
            uint8x16_t AxorB = veorq_u8 (A_vec, B_vec);
            uint8x16_t bitsSet = vcntq_u8 (AxorB);
            uint16x8_t bitSet8 = vpaddlq_u8 (bitsSet);
            uint32x4_t bitSet4 = vpaddlq_u16 (bitSet8);
            bits = vaddq_u32(bits, bitSet4);
        }
        uint64x2_t bitSet2 = vpaddlq_u32 (bits);
        result = vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),0);
        result += vgetq_lane_s32 (vreinterpretq_s32_u64(bitSet2),2);
    }
#endif
        for( ; i <= n - 4; i += 4 )
            result += popCountTable[a[i] ^ b[i]] + popCountTable[a[i+1] ^ b[i+1]] +
                    popCountTable[a[i+2] ^ b[i+2]] + popCountTable[a[i+3] ^ b[i+3]];
    for( ; i < n; i++ )
        result += popCountTable[a[i] ^ b[i]];
    return result;
}

int normHamming(const uchar* a, int n, int cellSize)
{
    if( cellSize == 1 )
        return normHamming(a, n);
    const uchar* tab = 0;
    if( cellSize == 2 )
        tab = popCountTable2;
    else if( cellSize == 4 )
        tab = popCountTable4;
    else
        return -1;
    int i = 0;
    int result = 0;
#if CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
        result += tab[a[i]] + tab[a[i+1]] + tab[a[i+2]] + tab[a[i+3]];
#endif
    for( ; i < n; i++ )
        result += tab[a[i]];
    return result;
}

int normHamming(const uchar* a, const uchar* b, int n, int cellSize)
{
    if( cellSize == 1 )
        return normHamming(a, b, n);
    const uchar* tab = 0;
    if( cellSize == 2 )
        tab = popCountTable2;
    else if( cellSize == 4 )
        tab = popCountTable4;
    else
        return -1;
    int i = 0;
    int result = 0;
    #if CV_ENABLE_UNROLLED
    for( ; i <= n - 4; i += 4 )
        result += tab[a[i] ^ b[i]] + tab[a[i+1] ^ b[i+1]] +
                tab[a[i+2] ^ b[i+2]] + tab[a[i+3] ^ b[i+3]];
    #endif
    for( ; i < n; i++ )
        result += tab[a[i] ^ b[i]];
    return result;
}

float normL2Sqr_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if CV_SSE
    float CV_DECL_ALIGNED(16) buf[4];
    __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();

    for( ; j <= n - 8; j += 8 )
    {
        __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
        __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
        d0 = _mm_add_ps(d0, _mm_mul_ps(t0, t0));
        d1 = _mm_add_ps(d1, _mm_mul_ps(t1, t1));
    }
    _mm_store_ps(buf, _mm_add_ps(d0, d1));
    d = buf[0] + buf[1] + buf[2] + buf[3];
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            float t0 = a[j] - b[j], t1 = a[j+1] - b[j+1], t2 = a[j+2] - b[j+2], t3 = a[j+3] - b[j+3];
            d += t0*t0 + t1*t1 + t2*t2 + t3*t3;
        }
    }

    for( ; j < n; j++ )
    {
        float t = a[j] - b[j];
        d += t*t;
    }
    return d;
}


float normL1_(const float* a, const float* b, int n)
{
    int j = 0; float d = 0.f;
#if CV_SSE
    float CV_DECL_ALIGNED(16) buf[4];
    static const int CV_DECL_ALIGNED(16) absbuf[4] = {0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff};
    __m128 d0 = _mm_setzero_ps(), d1 = _mm_setzero_ps();
    __m128 absmask = _mm_load_ps((const float*)absbuf);

    for( ; j <= n - 8; j += 8 )
    {
        __m128 t0 = _mm_sub_ps(_mm_loadu_ps(a + j), _mm_loadu_ps(b + j));
        __m128 t1 = _mm_sub_ps(_mm_loadu_ps(a + j + 4), _mm_loadu_ps(b + j + 4));
        d0 = _mm_add_ps(d0, _mm_and_ps(t0, absmask));
        d1 = _mm_add_ps(d1, _mm_and_ps(t1, absmask));
    }
    _mm_store_ps(buf, _mm_add_ps(d0, d1));
    d = buf[0] + buf[1] + buf[2] + buf[3];
#elif CV_NEON
    float32x4_t v_sum = vdupq_n_f32(0.0f);
    for ( ; j <= n - 4; j += 4)
        v_sum = vaddq_f32(v_sum, vabdq_f32(vld1q_f32(a + j), vld1q_f32(b + j)));

    float CV_DECL_ALIGNED(16) buf[4];
    vst1q_f32(buf, v_sum);
    d = buf[0] + buf[1] + buf[2] + buf[3];
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            d += std::abs(a[j] - b[j]) + std::abs(a[j+1] - b[j+1]) +
            std::abs(a[j+2] - b[j+2]) + std::abs(a[j+3] - b[j+3]);
        }
    }

    for( ; j < n; j++ )
        d += std::abs(a[j] - b[j]);
    return d;
}

int normL1_(const uchar* a, const uchar* b, int n)
{
    int j = 0, d = 0;
#if CV_SSE
    __m128i d0 = _mm_setzero_si128();

    for( ; j <= n - 16; j += 16 )
    {
        __m128i t0 = _mm_loadu_si128((const __m128i*)(a + j));
        __m128i t1 = _mm_loadu_si128((const __m128i*)(b + j));

        d0 = _mm_add_epi32(d0, _mm_sad_epu8(t0, t1));
    }

    for( ; j <= n - 4; j += 4 )
    {
        __m128i t0 = _mm_cvtsi32_si128(*(const int*)(a + j));
        __m128i t1 = _mm_cvtsi32_si128(*(const int*)(b + j));

        d0 = _mm_add_epi32(d0, _mm_sad_epu8(t0, t1));
    }
    d = _mm_cvtsi128_si32(_mm_add_epi32(d0, _mm_unpackhi_epi64(d0, d0)));
#elif CV_NEON
    uint32x4_t v_sum = vdupq_n_u32(0.0f);
    for ( ; j <= n - 16; j += 16)
    {
        uint8x16_t v_dst = vabdq_u8(vld1q_u8(a + j), vld1q_u8(b + j));
        uint16x8_t v_low = vmovl_u8(vget_low_u8(v_dst)), v_high = vmovl_u8(vget_high_u8(v_dst));
        v_sum = vaddq_u32(v_sum, vaddl_u16(vget_low_u16(v_low), vget_low_u16(v_high)));
        v_sum = vaddq_u32(v_sum, vaddl_u16(vget_high_u16(v_low), vget_high_u16(v_high)));
    }

    uint CV_DECL_ALIGNED(16) buf[4];
    vst1q_u32(buf, v_sum);
    d = buf[0] + buf[1] + buf[2] + buf[3];
#endif
    {
        for( ; j <= n - 4; j += 4 )
        {
            d += std::abs(a[j] - b[j]) + std::abs(a[j+1] - b[j+1]) +
            std::abs(a[j+2] - b[j+2]) + std::abs(a[j+3] - b[j+3]);
        }
    }
    for( ; j < n; j++ )
        d += std::abs(a[j] - b[j]);
    return d;
}

}} //cv::hal
