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
#include <climits>
#include <limits>
#include "opencv2/core/hal/intrin.hpp"

#include "opencl_kernels_core.hpp"

#include "opencv2/core/openvx/ovx_defs.hpp"

namespace cv
{

/****************************************************************************************\
*                                        sum                                             *
\****************************************************************************************/

template <typename T, typename ST>
struct Sum_SIMD
{
    int operator () (const T *, const uchar *, ST *, int, int) const
    {
        return 0;
    }
};

template <typename ST, typename DT>
inline void addChannels(DT * dst, ST * buf, int cn)
{
    for (int i = 0; i < 4; ++i)
        dst[i % cn] += buf[i];
}

#if CV_SSE2

template <>
struct Sum_SIMD<schar, int>
{
    int operator () (const schar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128i v_zero = _mm_setzero_si128(), v_sum = v_zero;

        for ( ; x <= len - 16; x += 16)
        {
            __m128i v_src = _mm_loadu_si128((const __m128i *)(src0 + x));
            __m128i v_half = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src), 8);

            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_half), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_half), 16));

            v_half = _mm_srai_epi16(_mm_unpackhi_epi8(v_zero, v_src), 8);
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_half), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_half), 16));
        }

        for ( ; x <= len - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src0 + x))), 8);

            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
        }

        int CV_DECL_ALIGNED(16) ar[4];
        _mm_store_si128((__m128i*)ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<int, double>
{
    int operator () (const int * src0, const uchar * mask, double * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128d v_zero = _mm_setzero_pd(), v_sum0 = v_zero, v_sum1 = v_zero;

        for ( ; x <= len - 4; x += 4)
        {
            __m128i v_src = _mm_loadu_si128((__m128i const *)(src0 + x));
            v_sum0 = _mm_add_pd(v_sum0, _mm_cvtepi32_pd(v_src));
            v_sum1 = _mm_add_pd(v_sum1, _mm_cvtepi32_pd(_mm_srli_si128(v_src, 8)));
        }

        double CV_DECL_ALIGNED(16) ar[4];
        _mm_store_pd(ar, v_sum0);
        _mm_store_pd(ar + 2, v_sum1);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<float, double>
{
    int operator () (const float * src0, const uchar * mask, double * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128d v_zero = _mm_setzero_pd(), v_sum0 = v_zero, v_sum1 = v_zero;

        for ( ; x <= len - 4; x += 4)
        {
            __m128 v_src = _mm_loadu_ps(src0 + x);
            v_sum0 = _mm_add_pd(v_sum0, _mm_cvtps_pd(v_src));
            v_src = _mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(v_src), 8));
            v_sum1 = _mm_add_pd(v_sum1, _mm_cvtps_pd(v_src));
        }

        double CV_DECL_ALIGNED(16) ar[4];
        _mm_store_pd(ar, v_sum0);
        _mm_store_pd(ar + 2, v_sum1);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};


#elif CV_NEON

template <>
struct Sum_SIMD<uchar, int>
{
    int operator () (const uchar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;

        int x = 0;
        uint32x4_t v_sum = vdupq_n_u32(0u);

        for ( ; x <= len - 16; x += 16)
        {
            uint8x16_t v_src = vld1q_u8(src0 + x);
            uint16x8_t v_half = vmovl_u8(vget_low_u8(v_src));

            v_sum = vaddw_u16(v_sum, vget_low_u16(v_half));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_half));

            v_half = vmovl_u8(vget_high_u8(v_src));
            v_sum = vaddw_u16(v_sum, vget_low_u16(v_half));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_half));
        }

        for ( ; x <= len - 8; x += 8)
        {
            uint16x8_t v_src = vmovl_u8(vld1_u8(src0 + x));

            v_sum = vaddw_u16(v_sum, vget_low_u16(v_src));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_src));
        }

        unsigned int CV_DECL_ALIGNED(16) ar[4];
        vst1q_u32(ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<schar, int>
{
    int operator () (const schar * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;

        int x = 0;
        int32x4_t v_sum = vdupq_n_s32(0);

        for ( ; x <= len - 16; x += 16)
        {
            int8x16_t v_src = vld1q_s8(src0 + x);
            int16x8_t v_half = vmovl_s8(vget_low_s8(v_src));

            v_sum = vaddw_s16(v_sum, vget_low_s16(v_half));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_half));

            v_half = vmovl_s8(vget_high_s8(v_src));
            v_sum = vaddw_s16(v_sum, vget_low_s16(v_half));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_half));
        }

        for ( ; x <= len - 8; x += 8)
        {
            int16x8_t v_src = vmovl_s8(vld1_s8(src0 + x));

            v_sum = vaddw_s16(v_sum, vget_low_s16(v_src));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_src));
        }

        int CV_DECL_ALIGNED(16) ar[4];
        vst1q_s32(ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<ushort, int>
{
    int operator () (const ushort * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;

        int x = 0;
        uint32x4_t v_sum = vdupq_n_u32(0u);

        for ( ; x <= len - 8; x += 8)
        {
            uint16x8_t v_src = vld1q_u16(src0 + x);

            v_sum = vaddw_u16(v_sum, vget_low_u16(v_src));
            v_sum = vaddw_u16(v_sum, vget_high_u16(v_src));
        }

        for ( ; x <= len - 4; x += 4)
            v_sum = vaddw_u16(v_sum, vld1_u16(src0 + x));

        unsigned int CV_DECL_ALIGNED(16) ar[4];
        vst1q_u32(ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

template <>
struct Sum_SIMD<short, int>
{
    int operator () (const short * src0, const uchar * mask, int * dst, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2 && cn != 4))
            return 0;

        int x = 0;
        int32x4_t v_sum = vdupq_n_s32(0u);

        for ( ; x <= len - 8; x += 8)
        {
            int16x8_t v_src = vld1q_s16(src0 + x);

            v_sum = vaddw_s16(v_sum, vget_low_s16(v_src));
            v_sum = vaddw_s16(v_sum, vget_high_s16(v_src));
        }

        for ( ; x <= len - 4; x += 4)
            v_sum = vaddw_s16(v_sum, vld1_s16(src0 + x));

        int CV_DECL_ALIGNED(16) ar[4];
        vst1q_s32(ar, v_sum);

        addChannels(dst, ar, cn);

        return x / cn;
    }
};

#endif

template<typename T, typename ST>
static int sum_(const T* src0, const uchar* mask, ST* dst, int len, int cn )
{
    const T* src = src0;
    if( !mask )
    {
        Sum_SIMD<T, ST> vop;
        int i = vop(src0, mask, dst, len, cn), k = cn % 4;
        src += i * cn;

        if( k == 1 )
        {
            ST s0 = dst[0];

            #if CV_ENABLE_UNROLLED
            for(; i <= len - 4; i += 4, src += cn*4 )
                s0 += src[0] + src[cn] + src[cn*2] + src[cn*3];
            #endif
            for( ; i < len; i++, src += cn )
                s0 += src[0];
            dst[0] = s0;
        }
        else if( k == 2 )
        {
            ST s0 = dst[0], s1 = dst[1];
            for( ; i < len; i++, src += cn )
            {
                s0 += src[0];
                s1 += src[1];
            }
            dst[0] = s0;
            dst[1] = s1;
        }
        else if( k == 3 )
        {
            ST s0 = dst[0], s1 = dst[1], s2 = dst[2];
            for( ; i < len; i++, src += cn )
            {
                s0 += src[0];
                s1 += src[1];
                s2 += src[2];
            }
            dst[0] = s0;
            dst[1] = s1;
            dst[2] = s2;
        }

        for( ; k < cn; k += 4 )
        {
            src = src0 + i*cn + k;
            ST s0 = dst[k], s1 = dst[k+1], s2 = dst[k+2], s3 = dst[k+3];
            for( ; i < len; i++, src += cn )
            {
                s0 += src[0]; s1 += src[1];
                s2 += src[2]; s3 += src[3];
            }
            dst[k] = s0;
            dst[k+1] = s1;
            dst[k+2] = s2;
            dst[k+3] = s3;
        }
        return len;
    }

    int i, nzm = 0;
    if( cn == 1 )
    {
        ST s = dst[0];
        for( i = 0; i < len; i++ )
            if( mask[i] )
            {
                s += src[i];
                nzm++;
            }
        dst[0] = s;
    }
    else if( cn == 3 )
    {
        ST s0 = dst[0], s1 = dst[1], s2 = dst[2];
        for( i = 0; i < len; i++, src += 3 )
            if( mask[i] )
            {
                s0 += src[0];
                s1 += src[1];
                s2 += src[2];
                nzm++;
            }
        dst[0] = s0;
        dst[1] = s1;
        dst[2] = s2;
    }
    else
    {
        for( i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                int k = 0;
                #if CV_ENABLE_UNROLLED
                for( ; k <= cn - 4; k += 4 )
                {
                    ST s0, s1;
                    s0 = dst[k] + src[k];
                    s1 = dst[k+1] + src[k+1];
                    dst[k] = s0; dst[k+1] = s1;
                    s0 = dst[k+2] + src[k+2];
                    s1 = dst[k+3] + src[k+3];
                    dst[k+2] = s0; dst[k+3] = s1;
                }
                #endif
                for( ; k < cn; k++ )
                    dst[k] += src[k];
                nzm++;
            }
    }
    return nzm;
}


static int sum8u( const uchar* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum8s( const schar* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum16u( const ushort* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum16s( const short* src, const uchar* mask, int* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum32s( const int* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum32f( const float* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

static int sum64f( const double* src, const uchar* mask, double* dst, int len, int cn )
{ return sum_(src, mask, dst, len, cn); }

typedef int (*SumFunc)(const uchar*, const uchar* mask, uchar*, int, int);

static SumFunc getSumFunc(int depth)
{
    static SumFunc sumTab[] =
    {
        (SumFunc)GET_OPTIMIZED(sum8u), (SumFunc)sum8s,
        (SumFunc)sum16u, (SumFunc)sum16s,
        (SumFunc)sum32s,
        (SumFunc)GET_OPTIMIZED(sum32f), (SumFunc)sum64f,
        0
    };

    return sumTab[depth];
}

template<typename T>
static int countNonZero_(const T* src, int len )
{
    int i=0, nz = 0;
    #if CV_ENABLE_UNROLLED
    for(; i <= len - 4; i += 4 )
        nz += (src[i] != 0) + (src[i+1] != 0) + (src[i+2] != 0) + (src[i+3] != 0);
    #endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero8u( const uchar* src, int len )
{
    int i=0, nz = 0;
#if CV_SSE2
    if(USE_SSE2)//5x-6x
    {
        __m128i v_zero = _mm_setzero_si128();
        __m128i sum = _mm_setzero_si128();

        for (; i<=len-16; i+=16)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src+i));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_cmpeq_epi8(r0, v_zero)), v_zero));
        }
        nz = i - _mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum)));
    }
#elif CV_NEON
    int len0 = len & -16, blockSize1 = (1 << 8) - 16, blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    uint8x16_t v_zero = vdupq_n_u8(0), v_1 = vdupq_n_u8(1);
    const uchar * src0 = src;

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint8x16_t v_pz = v_zero;

            for( ; k <= blockSizej - 16; k += 16 )
                v_pz = vaddq_u8(v_pz, vandq_u8(vceqq_u8(vld1q_u8(src0 + k), v_zero), v_1));

            uint16x8_t v_p1 = vmovl_u8(vget_low_u8(v_pz)), v_p2 = vmovl_u8(vget_high_u8(v_pz));
            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_p1), vget_high_u16(v_p1)), v_nz);
            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_p2), vget_high_u16(v_p2)), v_nz);

            src0 += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    for( ; i < len; i++ )
        nz += src[i] != 0;
    return nz;
}

static int countNonZero16u( const ushort* src, int len )
{
    int i = 0, nz = 0;
#if CV_SSE2
    if (USE_SSE2)
    {
        __m128i v_zero = _mm_setzero_si128 ();
        __m128i sum = _mm_setzero_si128();

        for ( ; i <= len - 8; i += 8)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src + i));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_cmpeq_epi16(r0, v_zero)), v_zero));
        }

        nz = i - (_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum))) >> 1);
        src += i;
    }
#elif CV_NEON
    int len0 = len & -8, blockSize1 = (1 << 15), blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    uint16x8_t v_zero = vdupq_n_u16(0), v_1 = vdupq_n_u16(1);

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint16x8_t v_pz = v_zero;

            for( ; k <= blockSizej - 8; k += 8 )
                v_pz = vaddq_u16(v_pz, vandq_u16(vceqq_u16(vld1q_u16(src + k), v_zero), v_1));

            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_pz), vget_high_u16(v_pz)), v_nz);

            src += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    return nz + countNonZero_(src, len - i);
}

static int countNonZero32s( const int* src, int len )
{
    int i = 0, nz = 0;
#if CV_SSE2
    if (USE_SSE2)
    {
        __m128i v_zero = _mm_setzero_si128 ();
        __m128i sum = _mm_setzero_si128();

        for ( ; i <= len - 4; i += 4)
        {
            __m128i r0 = _mm_loadu_si128((const __m128i*)(src + i));
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_cmpeq_epi32(r0, v_zero)), v_zero));
        }

        nz = i - (_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum))) >> 2);
        src += i;
    }
#elif CV_NEON
    int len0 = len & -8, blockSize1 = (1 << 15), blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    int32x4_t v_zero = vdupq_n_s32(0.0f);
    uint16x8_t v_1 = vdupq_n_u16(1u), v_zerou = vdupq_n_u16(0u);

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint16x8_t v_pz = v_zerou;

            for( ; k <= blockSizej - 8; k += 8 )
                v_pz = vaddq_u16(v_pz, vandq_u16(vcombine_u16(vmovn_u32(vceqq_s32(vld1q_s32(src + k), v_zero)),
                                                              vmovn_u32(vceqq_s32(vld1q_s32(src + k + 4), v_zero))), v_1));

            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_pz), vget_high_u16(v_pz)), v_nz);

            src += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    return nz + countNonZero_(src, len - i);
}

static int countNonZero32f( const float* src, int len )
{
    int i = 0, nz = 0;
#if CV_SSE2
    if (USE_SSE2)
    {
        __m128 v_zero_f = _mm_setzero_ps();
        __m128i v_zero = _mm_setzero_si128 ();
        __m128i sum = _mm_setzero_si128();

        for ( ; i <= len - 4; i += 4)
        {
            __m128 r0 = _mm_loadu_ps(src + i);
            sum = _mm_add_epi32(sum, _mm_sad_epu8(_mm_sub_epi8(v_zero, _mm_castps_si128(_mm_cmpeq_ps(r0, v_zero_f))), v_zero));
        }

        nz = i - (_mm_cvtsi128_si32(_mm_add_epi32(sum, _mm_unpackhi_epi64(sum, sum))) >> 2);
        src += i;
    }
#elif CV_NEON
    int len0 = len & -8, blockSize1 = (1 << 15), blockSize0 = blockSize1 << 6;
    uint32x4_t v_nz = vdupq_n_u32(0u);
    float32x4_t v_zero = vdupq_n_f32(0.0f);
    uint16x8_t v_1 = vdupq_n_u16(1u), v_zerou = vdupq_n_u16(0u);

    while( i < len0 )
    {
        int blockSizei = std::min(len0 - i, blockSize0), j = 0;

        while (j < blockSizei)
        {
            int blockSizej = std::min(blockSizei - j, blockSize1), k = 0;
            uint16x8_t v_pz = v_zerou;

            for( ; k <= blockSizej - 8; k += 8 )
                v_pz = vaddq_u16(v_pz, vandq_u16(vcombine_u16(vmovn_u32(vceqq_f32(vld1q_f32(src + k), v_zero)),
                                                              vmovn_u32(vceqq_f32(vld1q_f32(src + k + 4), v_zero))), v_1));

            v_nz = vaddq_u32(vaddl_u16(vget_low_u16(v_pz), vget_high_u16(v_pz)), v_nz);

            src += blockSizej;
            j += blockSizej;
        }

        i += blockSizei;
    }

    CV_DECL_ALIGNED(16) unsigned int buf[4];
    vst1q_u32(buf, v_nz);
    nz += i - saturate_cast<int>(buf[0] + buf[1] + buf[2] + buf[3]);
#endif
    return nz + countNonZero_(src, len - i);
}

static int countNonZero64f( const double* src, int len )
{
    return countNonZero_(src, len);
}

typedef int (*CountNonZeroFunc)(const uchar*, int);

static CountNonZeroFunc getCountNonZeroTab(int depth)
{
    static CountNonZeroFunc countNonZeroTab[] =
    {
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero8u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero16u),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32s), (CountNonZeroFunc)GET_OPTIMIZED(countNonZero32f),
        (CountNonZeroFunc)GET_OPTIMIZED(countNonZero64f), 0
    };

    return countNonZeroTab[depth];
}

template <typename T, typename ST, typename SQT>
struct SumSqr_SIMD
{
    int operator () (const T *, const uchar *, ST *, SQT *, int, int) const
    {
        return 0;
    }
};

template <typename T>
inline void addSqrChannels(T * sum, T * sqsum, T * buf, int cn)
{
    for (int i = 0; i < 4; ++i)
    {
        sum[i % cn] += buf[i];
        sqsum[i % cn] += buf[4 + i];
    }
}

#if CV_SSE2

template <>
struct SumSqr_SIMD<uchar, int, int>
{
    int operator () (const uchar * src0, const uchar * mask, int * sum, int * sqsum, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128i v_zero = _mm_setzero_si128(), v_sum = v_zero, v_sqsum = v_zero;
        const int len_16 = len & ~15;

        for ( ; x <= len_16 - 16; )
        {
            const int len_tmp = min(x + 2048, len_16);
            __m128i v_sum_tmp = v_zero;
            for ( ; x <= len_tmp - 16; x += 16)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src0 + x));
                __m128i v_half_0 = _mm_unpacklo_epi8(v_src, v_zero);
                __m128i v_half_1 = _mm_unpackhi_epi8(v_src, v_zero);
                v_sum_tmp = _mm_add_epi16(v_sum_tmp, _mm_add_epi16(v_half_0, v_half_1));
                __m128i v_half_2 = _mm_unpacklo_epi16(v_half_0, v_half_1);
                __m128i v_half_3 = _mm_unpackhi_epi16(v_half_0, v_half_1);
                v_sqsum = _mm_add_epi32(v_sqsum, _mm_madd_epi16(v_half_2, v_half_2));
                v_sqsum = _mm_add_epi32(v_sqsum, _mm_madd_epi16(v_half_3, v_half_3));
            }
            v_sum = _mm_add_epi32(v_sum, _mm_unpacklo_epi16(v_sum_tmp, v_zero));
            v_sum = _mm_add_epi32(v_sum, _mm_unpackhi_epi16(v_sum_tmp, v_zero));
        }

        for ( ; x <= len - 8; x += 8)
        {
            __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src0 + x)), v_zero);
            __m128i v_half_0 = _mm_unpackhi_epi64(v_src, v_src);
            __m128i v_sum_tmp = _mm_add_epi16(v_src, v_half_0);
            __m128i v_half_1 = _mm_unpacklo_epi16(v_src, v_half_0);

            v_sum = _mm_add_epi32(v_sum, _mm_unpacklo_epi16(v_sum_tmp, v_zero));
            v_sqsum = _mm_add_epi32(v_sqsum, _mm_madd_epi16(v_half_1, v_half_1));
        }

        int CV_DECL_ALIGNED(16) ar[8];
        _mm_store_si128((__m128i*)ar, v_sum);
        _mm_store_si128((__m128i*)(ar + 4), v_sqsum);

        addSqrChannels(sum, sqsum, ar, cn);

        return x / cn;
    }
};

template <>
struct SumSqr_SIMD<schar, int, int>
{
    int operator () (const schar * src0, const uchar * mask, int * sum, int * sqsum, int len, int cn) const
    {
        if (mask || (cn != 1 && cn != 2) || !USE_SSE2)
            return 0;

        int x = 0;
        __m128i v_zero = _mm_setzero_si128(), v_sum = v_zero, v_sqsum = v_zero;
        const int len_16 = len & ~15;

        for ( ; x <= len_16 - 16; )
        {
            const int len_tmp = min(x + 2048, len_16);
            __m128i v_sum_tmp = v_zero;
            for ( ; x <= len_tmp - 16; x += 16)
            {
                __m128i v_src = _mm_loadu_si128((const __m128i *)(src0 + x));
                __m128i v_half_0 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src), 8);
                __m128i v_half_1 = _mm_srai_epi16(_mm_unpackhi_epi8(v_zero, v_src), 8);
                v_sum_tmp = _mm_add_epi16(v_sum_tmp, _mm_add_epi16(v_half_0, v_half_1));
                __m128i v_half_2 = _mm_unpacklo_epi16(v_half_0, v_half_1);
                __m128i v_half_3 = _mm_unpackhi_epi16(v_half_0, v_half_1);
                v_sqsum = _mm_add_epi32(v_sqsum, _mm_madd_epi16(v_half_2, v_half_2));
                v_sqsum = _mm_add_epi32(v_sqsum, _mm_madd_epi16(v_half_3, v_half_3));
            }
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_sum_tmp), 16));
            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_sum_tmp), 16));
        }

        for ( ; x <= len - 8; x += 8)
        {
            __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src0 + x))), 8);
            __m128i v_half_0 = _mm_unpackhi_epi64(v_src, v_src);
            __m128i v_sum_tmp = _mm_add_epi16(v_src, v_half_0);
            __m128i v_half_1 = _mm_unpacklo_epi16(v_src, v_half_0);

            v_sum = _mm_add_epi32(v_sum, _mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_sum_tmp), 16));
            v_sqsum = _mm_add_epi32(v_sqsum, _mm_madd_epi16(v_half_1, v_half_1));
        }

        int CV_DECL_ALIGNED(16) ar[8];
        _mm_store_si128((__m128i*)ar, v_sum);
        _mm_store_si128((__m128i*)(ar + 4), v_sqsum);

        addSqrChannels(sum, sqsum, ar, cn);

        return x / cn;
    }
};

#endif

template<typename T, typename ST, typename SQT>
static int sumsqr_(const T* src0, const uchar* mask, ST* sum, SQT* sqsum, int len, int cn )
{
    const T* src = src0;

    if( !mask )
    {
        SumSqr_SIMD<T, ST, SQT> vop;
        int i = vop(src0, mask, sum, sqsum, len, cn), k = cn % 4;
        src += i * cn;

        if( k == 1 )
        {
            ST s0 = sum[0];
            SQT sq0 = sqsum[0];
            for( ; i < len; i++, src += cn )
            {
                T v = src[0];
                s0 += v; sq0 += (SQT)v*v;
            }
            sum[0] = s0;
            sqsum[0] = sq0;
        }
        else if( k == 2 )
        {
            ST s0 = sum[0], s1 = sum[1];
            SQT sq0 = sqsum[0], sq1 = sqsum[1];
            for( ; i < len; i++, src += cn )
            {
                T v0 = src[0], v1 = src[1];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
            }
            sum[0] = s0; sum[1] = s1;
            sqsum[0] = sq0; sqsum[1] = sq1;
        }
        else if( k == 3 )
        {
            ST s0 = sum[0], s1 = sum[1], s2 = sum[2];
            SQT sq0 = sqsum[0], sq1 = sqsum[1], sq2 = sqsum[2];
            for( ; i < len; i++, src += cn )
            {
                T v0 = src[0], v1 = src[1], v2 = src[2];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
                s2 += v2; sq2 += (SQT)v2*v2;
            }
            sum[0] = s0; sum[1] = s1; sum[2] = s2;
            sqsum[0] = sq0; sqsum[1] = sq1; sqsum[2] = sq2;
        }

        for( ; k < cn; k += 4 )
        {
            src = src0 + k;
            ST s0 = sum[k], s1 = sum[k+1], s2 = sum[k+2], s3 = sum[k+3];
            SQT sq0 = sqsum[k], sq1 = sqsum[k+1], sq2 = sqsum[k+2], sq3 = sqsum[k+3];
            for( ; i < len; i++, src += cn )
            {
                T v0, v1;
                v0 = src[0], v1 = src[1];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
                v0 = src[2], v1 = src[3];
                s2 += v0; sq2 += (SQT)v0*v0;
                s3 += v1; sq3 += (SQT)v1*v1;
            }
            sum[k] = s0; sum[k+1] = s1;
            sum[k+2] = s2; sum[k+3] = s3;
            sqsum[k] = sq0; sqsum[k+1] = sq1;
            sqsum[k+2] = sq2; sqsum[k+3] = sq3;
        }
        return len;
    }

    int i, nzm = 0;

    if( cn == 1 )
    {
        ST s0 = sum[0];
        SQT sq0 = sqsum[0];
        for( i = 0; i < len; i++ )
            if( mask[i] )
            {
                T v = src[i];
                s0 += v; sq0 += (SQT)v*v;
                nzm++;
            }
        sum[0] = s0;
        sqsum[0] = sq0;
    }
    else if( cn == 3 )
    {
        ST s0 = sum[0], s1 = sum[1], s2 = sum[2];
        SQT sq0 = sqsum[0], sq1 = sqsum[1], sq2 = sqsum[2];
        for( i = 0; i < len; i++, src += 3 )
            if( mask[i] )
            {
                T v0 = src[0], v1 = src[1], v2 = src[2];
                s0 += v0; sq0 += (SQT)v0*v0;
                s1 += v1; sq1 += (SQT)v1*v1;
                s2 += v2; sq2 += (SQT)v2*v2;
                nzm++;
            }
        sum[0] = s0; sum[1] = s1; sum[2] = s2;
        sqsum[0] = sq0; sqsum[1] = sq1; sqsum[2] = sq2;
    }
    else
    {
        for( i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    T v = src[k];
                    ST s = sum[k] + v;
                    SQT sq = sqsum[k] + (SQT)v*v;
                    sum[k] = s; sqsum[k] = sq;
                }
                nzm++;
            }
    }
    return nzm;
}


static int sqsum8u( const uchar* src, const uchar* mask, int* sum, int* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum8s( const schar* src, const uchar* mask, int* sum, int* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum16u( const ushort* src, const uchar* mask, int* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum16s( const short* src, const uchar* mask, int* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum32s( const int* src, const uchar* mask, double* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum32f( const float* src, const uchar* mask, double* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

static int sqsum64f( const double* src, const uchar* mask, double* sum, double* sqsum, int len, int cn )
{ return sumsqr_(src, mask, sum, sqsum, len, cn); }

typedef int (*SumSqrFunc)(const uchar*, const uchar* mask, uchar*, uchar*, int, int);

static SumSqrFunc getSumSqrTab(int depth)
{
    static SumSqrFunc sumSqrTab[] =
    {
        (SumSqrFunc)GET_OPTIMIZED(sqsum8u), (SumSqrFunc)sqsum8s, (SumSqrFunc)sqsum16u, (SumSqrFunc)sqsum16s,
        (SumSqrFunc)sqsum32s, (SumSqrFunc)GET_OPTIMIZED(sqsum32f), (SumSqrFunc)sqsum64f, 0
    };

    return sumSqrTab[depth];
}

#ifdef HAVE_OPENCL

template <typename T> Scalar ocl_part_sum(Mat m)
{
    CV_Assert(m.rows == 1);

    Scalar s = Scalar::all(0);
    int cn = m.channels();
    const T * const ptr = m.ptr<T>(0);

    for (int x = 0, w = m.cols * cn; x < w; )
        for (int c = 0; c < cn; ++c, ++x)
            s[c] += ptr[x];

    return s;
}

enum { OCL_OP_SUM = 0, OCL_OP_SUM_ABS =  1, OCL_OP_SUM_SQR = 2 };

static bool ocl_sum( InputArray _src, Scalar & res, int sum_op, InputArray _mask = noArray(),
                     InputArray _src2 = noArray(), bool calc2 = false, const Scalar & res2 = Scalar() )
{
    CV_Assert(sum_op == OCL_OP_SUM || sum_op == OCL_OP_SUM_ABS || sum_op == OCL_OP_SUM_SQR);

    const ocl::Device & dev = ocl::Device::getDefault();
    bool doubleSupport = dev.doubleFPConfig() > 0,
        haveMask = _mask.kind() != _InputArray::NONE,
        haveSrc2 = _src2.kind() != _InputArray::NONE;
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            kercn = cn == 1 && !haveMask ? ocl::predictOptimalVectorWidth(_src, _src2) : 1,
            mcn = std::max(cn, kercn);
    CV_Assert(!haveSrc2 || _src2.type() == type);
    int convert_cn = haveSrc2 ? mcn : cn;

    if ( (!doubleSupport && depth == CV_64F) || cn > 4 )
        return false;

    int ngroups = dev.maxComputeUnits(), dbsize = ngroups * (calc2 ? 2 : 1);
    size_t wgs = dev.maxWorkGroupSize();

    int ddepth = std::max(sum_op == OCL_OP_SUM_SQR ? CV_32F : CV_32S, depth),
            dtype = CV_MAKE_TYPE(ddepth, cn);
    CV_Assert(!haveMask || _mask.type() == CV_8UC1);

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    static const char * const opMap[3] = { "OP_SUM", "OP_SUM_ABS", "OP_SUM_SQR" };
    char cvt[2][40];
    String opts = format("-D srcT=%s -D srcT1=%s -D dstT=%s -D dstTK=%s -D dstT1=%s -D ddepth=%d -D cn=%d"
                         " -D convertToDT=%s -D %s -D WGS=%d -D WGS2_ALIGNED=%d%s%s%s%s -D kercn=%d%s%s%s -D convertFromU=%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, mcn)), ocl::typeToStr(depth),
                         ocl::typeToStr(dtype), ocl::typeToStr(CV_MAKE_TYPE(ddepth, mcn)),
                         ocl::typeToStr(ddepth), ddepth, cn,
                         ocl::convertTypeStr(depth, ddepth, mcn, cvt[0]),
                         opMap[sum_op], (int)wgs, wgs2_aligned,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         haveMask ? " -D HAVE_MASK" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : "",
                         haveMask && _mask.isContinuous() ? " -D HAVE_MASK_CONT" : "", kercn,
                         haveSrc2 ? " -D HAVE_SRC2" : "", calc2 ? " -D OP_CALC2" : "",
                         haveSrc2 && _src2.isContinuous() ? " -D HAVE_SRC2_CONT" : "",
                         depth <= CV_32S && ddepth == CV_32S ? ocl::convertTypeStr(CV_8U, ddepth, convert_cn, cvt[1]) : "noconvert");

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc, opts);
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), src2 = _src2.getUMat(),
        db(1, dbsize, dtype), mask = _mask.getUMat();

    ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
            dbarg = ocl::KernelArg::PtrWriteOnly(db),
            maskarg = ocl::KernelArg::ReadOnlyNoSize(mask),
            src2arg = ocl::KernelArg::ReadOnlyNoSize(src2);

    if (haveMask)
    {
        if (haveSrc2)
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, maskarg, src2arg);
        else
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, maskarg);
    }
    else
    {
        if (haveSrc2)
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg, src2arg);
        else
            k.args(srcarg, src.cols, (int)src.total(), ngroups, dbarg);
    }

    size_t globalsize = ngroups * wgs;
    if (k.run(1, &globalsize, &wgs, false))
    {
        typedef Scalar (*part_sum)(Mat m);
        part_sum funcs[3] = { ocl_part_sum<int>, ocl_part_sum<float>, ocl_part_sum<double> },
                func = funcs[ddepth - CV_32S];

        Mat mres = db.getMat(ACCESS_READ);
        if (calc2)
            const_cast<Scalar &>(res2) = func(mres.colRange(ngroups, dbsize));

        res = func(mres.colRange(0, ngroups));
        return true;
    }
    return false;
}

#endif

#ifdef HAVE_IPP
static bool ipp_sum(Mat &src, Scalar &_res)
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 >= 700
    int cn = src.channels();
    if (cn > 4)
        return false;
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims == 2 || (src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        typedef IppStatus (CV_STDCALL* ippiSumFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
        typedef IppStatus (CV_STDCALL* ippiSumFuncNoHint)(const void*, int, IppiSize, double *);
        ippiSumFuncHint ippiSumHint =
            type == CV_32FC1 ? (ippiSumFuncHint)ippiSum_32f_C1R :
            type == CV_32FC3 ? (ippiSumFuncHint)ippiSum_32f_C3R :
            type == CV_32FC4 ? (ippiSumFuncHint)ippiSum_32f_C4R :
            0;
        ippiSumFuncNoHint ippiSum =
            type == CV_8UC1 ? (ippiSumFuncNoHint)ippiSum_8u_C1R :
            type == CV_8UC3 ? (ippiSumFuncNoHint)ippiSum_8u_C3R :
            type == CV_8UC4 ? (ippiSumFuncNoHint)ippiSum_8u_C4R :
            type == CV_16UC1 ? (ippiSumFuncNoHint)ippiSum_16u_C1R :
            type == CV_16UC3 ? (ippiSumFuncNoHint)ippiSum_16u_C3R :
            type == CV_16UC4 ? (ippiSumFuncNoHint)ippiSum_16u_C4R :
            type == CV_16SC1 ? (ippiSumFuncNoHint)ippiSum_16s_C1R :
            type == CV_16SC3 ? (ippiSumFuncNoHint)ippiSum_16s_C3R :
            type == CV_16SC4 ? (ippiSumFuncNoHint)ippiSum_16s_C4R :
            0;
        CV_Assert(!ippiSumHint || !ippiSum);
        if( ippiSumHint || ippiSum )
        {
            Ipp64f res[4];
            IppStatus ret = ippiSumHint ?
                            CV_INSTRUMENT_FUN_IPP(ippiSumHint, src.ptr(), (int)src.step[0], sz, res, ippAlgHintAccurate) :
                            CV_INSTRUMENT_FUN_IPP(ippiSum, src.ptr(), (int)src.step[0], sz, res);
            if( ret >= 0 )
            {
                for( int i = 0; i < cn; i++ )
                    _res[i] = res[i];
                return true;
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(_res);
#endif
    return false;
}
#endif

}

cv::Scalar cv::sum( InputArray _src )
{
    CV_INSTRUMENT_REGION()

#if defined HAVE_OPENCL || defined HAVE_IPP
    Scalar _res;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_sum(_src, _res, OCL_OP_SUM),
                _res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_sum(src, _res), _res);

    int k, cn = src.channels(), depth = src.depth();
    SumFunc func = getSumFunc(depth);
    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1];
    NAryMatIterator it(arrays, ptrs);
    Scalar s;
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0;
    AutoBuffer<int> _buf;
    int* buf = (int*)&s[0];
    size_t esz = 0;
    bool blockSum = depth < CV_32S;

    if( blockSum )
    {
        intSumBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        _buf.allocate(cn);
        buf = _buf;

        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], 0, (uchar*)buf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += buf[k];
                    buf[k] = 0;
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
        }
    }
    return s;
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_countNonZero( InputArray _src, int & res )
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), kercn = ocl::predictOptimalVectorWidth(_src);
    bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0;

    if (depth == CV_64F && !doubleSupport)
        return false;

    int dbsize = ocl::Device::getDefault().maxComputeUnits();
    size_t wgs = ocl::Device::getDefault().maxWorkGroupSize();

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    ocl::Kernel k("reduce", ocl::core::reduce_oclsrc,
                  format("-D srcT=%s -D srcT1=%s -D cn=1 -D OP_COUNT_NON_ZERO"
                         " -D WGS=%d -D kercn=%d -D WGS2_ALIGNED=%d%s%s",
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)),
                         ocl::typeToStr(depth), (int)wgs, kercn,
                         wgs2_aligned, doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : ""));
    if (k.empty())
        return false;

    UMat src = _src.getUMat(), db(1, dbsize, CV_32SC1);
    k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
           dbsize, ocl::KernelArg::PtrWriteOnly(db));

    size_t globalsize = dbsize * wgs;
    if (k.run(1, &globalsize, &wgs, true))
        return res = saturate_cast<int>(cv::sum(db.getMat(ACCESS_READ))[0]), true;
    return false;
}

}

#endif

#if defined HAVE_IPP
namespace cv {

static bool ipp_countNonZero( Mat &src, int &res )
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 < 201801
    // Poor performance of SSE42
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

    Ipp32s  count = 0;
    int     depth = src.depth();

    if(src.dims <= 2)
    {
        IppStatus status;
        IppiSize  size = {src.cols*src.channels(), src.rows};

        if(depth == CV_8U)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, (const Ipp8u *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else if(depth == CV_32F)
            status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, (const Ipp32f *)src.ptr(), (int)src.step, size, &count, 0, 0);
        else
            return false;

        if(status < 0)
            return false;

        res = size.width*size.height - count;
    }
    else
    {
        IppStatus       status;
        const Mat      *arrays[] = {&src, NULL};
        Mat            planes[1];
        NAryMatIterator it(arrays, planes, 1);
        IppiSize        size  = {(int)it.size*src.channels(), 1};
        res = 0;
        for (size_t i = 0; i < it.nplanes; i++, ++it)
        {
            if(depth == CV_8U)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_8u_C1R, it.planes->ptr<Ipp8u>(), (int)it.planes->step, size, &count, 0, 0);
            else if(depth == CV_32F)
                status = CV_INSTRUMENT_FUN_IPP(ippiCountInRange_32f_C1R, it.planes->ptr<Ipp32f>(), (int)it.planes->step, size, &count, 0, 0);
            else
                return false;

            if(status < 0 || (int)it.planes->total()*src.channels() < count)
                return false;

            res += (int)it.planes->total()*src.channels() - count;
        }
    }

    return true;
}
}
#endif


int cv::countNonZero( InputArray _src )
{
    CV_INSTRUMENT_REGION()

    int type = _src.type(), cn = CV_MAT_CN(type);
    CV_Assert( cn == 1 );

#if defined HAVE_OPENCL || defined HAVE_IPP
    int res = -1;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_countNonZero(_src, res),
                res)
#endif

    Mat src = _src.getMat();
    CV_IPP_RUN_FAST(ipp_countNonZero(src, res), res);

    CountNonZeroFunc func = getCountNonZeroTab(src.depth());
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, 0};
    uchar* ptrs[1];
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, nz = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        nz += func( ptrs[0], total );

    return nz;
}

#if defined HAVE_IPP
namespace cv
{
static bool ipp_mean( Mat &src, Mat &mask, Scalar &ret )
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 >= 700
    size_t total_size = src.total();
    int cn = src.channels();
    if (cn > 4)
        return false;
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskMeanFuncC1 ippiMean_C1MR =
            type == CV_8UC1 ? (ippiMaskMeanFuncC1)ippiMean_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMeanFuncC1)ippiMean_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMeanFuncC1)ippiMean_32f_C1MR :
            0;
            if( ippiMean_C1MR )
            {
                Ipp64f res;
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_C1MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, &res) >= 0 )
                {
                    ret = Scalar(res);
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskMeanFuncC3 ippiMean_C3MR =
            type == CV_8UC3 ? (ippiMaskMeanFuncC3)ippiMean_8u_C3CMR :
            type == CV_16UC3 ? (ippiMaskMeanFuncC3)ippiMean_16u_C3CMR :
            type == CV_32FC3 ? (ippiMaskMeanFuncC3)ippiMean_32f_C3CMR :
            0;
            if( ippiMean_C3MR )
            {
                Ipp64f res1, res2, res3;
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 1, &res1) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 2, &res2) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_C3MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 3, &res3) >= 0 )
                {
                    ret = Scalar(res1, res2, res3);
                    return true;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanFuncHint)(const void*, int, IppiSize, double *, IppHintAlgorithm);
            typedef IppStatus (CV_STDCALL* ippiMeanFuncNoHint)(const void*, int, IppiSize, double *);
            ippiMeanFuncHint ippiMeanHint =
                type == CV_32FC1 ? (ippiMeanFuncHint)ippiMean_32f_C1R :
                type == CV_32FC3 ? (ippiMeanFuncHint)ippiMean_32f_C3R :
                type == CV_32FC4 ? (ippiMeanFuncHint)ippiMean_32f_C4R :
                0;
            ippiMeanFuncNoHint ippiMean =
                type == CV_8UC1 ? (ippiMeanFuncNoHint)ippiMean_8u_C1R :
                type == CV_8UC3 ? (ippiMeanFuncNoHint)ippiMean_8u_C3R :
                type == CV_8UC4 ? (ippiMeanFuncNoHint)ippiMean_8u_C4R :
                type == CV_16UC1 ? (ippiMeanFuncNoHint)ippiMean_16u_C1R :
                type == CV_16UC3 ? (ippiMeanFuncNoHint)ippiMean_16u_C3R :
                type == CV_16UC4 ? (ippiMeanFuncNoHint)ippiMean_16u_C4R :
                type == CV_16SC1 ? (ippiMeanFuncNoHint)ippiMean_16s_C1R :
                type == CV_16SC3 ? (ippiMeanFuncNoHint)ippiMean_16s_C3R :
                type == CV_16SC4 ? (ippiMeanFuncNoHint)ippiMean_16s_C4R :
                0;
            // Make sure only zero or one version of the function pointer is valid
            CV_Assert(!ippiMeanHint || !ippiMean);
            if( ippiMeanHint || ippiMean )
            {
                Ipp64f res[4];
                IppStatus status = ippiMeanHint ? CV_INSTRUMENT_FUN_IPP(ippiMeanHint, src.ptr(), (int)src.step[0], sz, res, ippAlgHintAccurate) :
                                CV_INSTRUMENT_FUN_IPP(ippiMean, src.ptr(), (int)src.step[0], sz, res);
                if( status >= 0 )
                {
                    for( int i = 0; i < cn; i++ )
                        ret[i] = res[i];
                    return true;
                }
            }
        }
    }
    return false;
#else
    return false;
#endif
}
}
#endif

cv::Scalar cv::mean( InputArray _src, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_Assert( mask.empty() || mask.type() == CV_8U );

    int k, cn = src.channels(), depth = src.depth();
    Scalar s;

    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_mean(src, mask, s), s)

    SumFunc func = getSumFunc(depth);

    CV_Assert( cn <= 4 && func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0;
    AutoBuffer<int> _buf;
    int* buf = (int*)&s[0];
    bool blockSum = depth <= CV_16S;
    size_t esz = 0, nz0 = 0;

    if( blockSum )
    {
        intSumBlockSize = depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        _buf.allocate(cn);
        buf = _buf;

        for( k = 0; k < cn; k++ )
            buf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            int nz = func( ptrs[0], ptrs[1], (uchar*)buf, bsz, cn );
            count += nz;
            nz0 += nz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += buf[k];
                    buf[k] = 0;
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }
    return s*(nz0 ? 1./nz0 : 0);
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_meanStdDev( InputArray _src, OutputArray _mean, OutputArray _sdv, InputArray _mask )
{
    CV_INSTRUMENT_REGION_OPENCL()

    bool haveMask = _mask.kind() != _InputArray::NONE;
    int nz = haveMask ? -1 : (int)_src.total();
    Scalar mean(0), stddev(0);
    const int cn = _src.channels();
    if (cn > 4)
        return false;

    {
        int type = _src.type(), depth = CV_MAT_DEPTH(type);
        bool doubleSupport = ocl::Device::getDefault().doubleFPConfig() > 0,
                isContinuous = _src.isContinuous(),
                isMaskContinuous = _mask.isContinuous();
        const ocl::Device &defDev = ocl::Device::getDefault();
        int groups = defDev.maxComputeUnits();
        if (defDev.isIntel())
        {
            static const int subSliceEUCount = 10;
            groups = (groups / subSliceEUCount) * 2;
        }
        size_t wgs = defDev.maxWorkGroupSize();

        int ddepth = std::max(CV_32S, depth), sqddepth = std::max(CV_32F, depth),
                dtype = CV_MAKE_TYPE(ddepth, cn),
                sqdtype = CV_MAKETYPE(sqddepth, cn);
        CV_Assert(!haveMask || _mask.type() == CV_8UC1);

        int wgs2_aligned = 1;
        while (wgs2_aligned < (int)wgs)
            wgs2_aligned <<= 1;
        wgs2_aligned >>= 1;

        if ( (!doubleSupport && depth == CV_64F) )
            return false;

        char cvt[2][40];
        String opts = format("-D srcT=%s -D srcT1=%s -D dstT=%s -D dstT1=%s -D sqddepth=%d"
                             " -D sqdstT=%s -D sqdstT1=%s -D convertToSDT=%s -D cn=%d%s%s"
                             " -D convertToDT=%s -D WGS=%d -D WGS2_ALIGNED=%d%s%s",
                             ocl::typeToStr(type), ocl::typeToStr(depth),
                             ocl::typeToStr(dtype), ocl::typeToStr(ddepth), sqddepth,
                             ocl::typeToStr(sqdtype), ocl::typeToStr(sqddepth),
                             ocl::convertTypeStr(depth, sqddepth, cn, cvt[0]),
                             cn, isContinuous ? " -D HAVE_SRC_CONT" : "",
                             isMaskContinuous ? " -D HAVE_MASK_CONT" : "",
                             ocl::convertTypeStr(depth, ddepth, cn, cvt[1]),
                             (int)wgs, wgs2_aligned, haveMask ? " -D HAVE_MASK" : "",
                             doubleSupport ? " -D DOUBLE_SUPPORT" : "");

        ocl::Kernel k("meanStdDev", ocl::core::meanstddev_oclsrc, opts);
        if (k.empty())
            return false;

        int dbsize = groups * ((haveMask ? CV_ELEM_SIZE1(CV_32S) : 0) +
                               CV_ELEM_SIZE(sqdtype) + CV_ELEM_SIZE(dtype));
        UMat src = _src.getUMat(), db(1, dbsize, CV_8UC1), mask = _mask.getUMat();

        ocl::KernelArg srcarg = ocl::KernelArg::ReadOnlyNoSize(src),
                dbarg = ocl::KernelArg::PtrWriteOnly(db),
                maskarg = ocl::KernelArg::ReadOnlyNoSize(mask);

        if (haveMask)
            k.args(srcarg, src.cols, (int)src.total(), groups, dbarg, maskarg);
        else
            k.args(srcarg, src.cols, (int)src.total(), groups, dbarg);

        size_t globalsize = groups * wgs;

        if(!k.run(1, &globalsize, &wgs, false))
            return false;

        typedef Scalar (* part_sum)(Mat m);
        part_sum funcs[3] = { ocl_part_sum<int>, ocl_part_sum<float>, ocl_part_sum<double> };
        Mat dbm = db.getMat(ACCESS_READ);

        mean = funcs[ddepth - CV_32S](Mat(1, groups, dtype, dbm.ptr()));
        stddev = funcs[sqddepth - CV_32S](Mat(1, groups, sqdtype, dbm.ptr() + groups * CV_ELEM_SIZE(dtype)));

        if (haveMask)
            nz = saturate_cast<int>(funcs[0](Mat(1, groups, CV_32SC1, dbm.ptr() +
                                                 groups * (CV_ELEM_SIZE(dtype) +
                                                           CV_ELEM_SIZE(sqdtype))))[0]);
    }

    double total = nz != 0 ? 1.0 / nz : 0;
    int k, j;
    for (int i = 0; i < cn; ++i)
    {
        mean[i] *= total;
        stddev[i] = std::sqrt(std::max(stddev[i] * total - mean[i] * mean[i] , 0.));
    }

    for( j = 0; j < 2; j++ )
    {
        const double * const sptr = j == 0 ? &mean[0] : &stddev[0];
        _OutputArray _dst = j == 0 ? _mean : _sdv;
        if( !_dst.needed() )
            continue;

        if( !_dst.fixedSize() )
            _dst.create(cn, 1, CV_64F, -1, true);
        Mat dst = _dst.getMat();
        int dcn = (int)dst.total();
        CV_Assert( dst.type() == CV_64F && dst.isContinuous() &&
                   (dst.cols == 1 || dst.rows == 1) && dcn >= cn );
        double* dptr = dst.ptr<double>();
        for( k = 0; k < cn; k++ )
            dptr[k] = sptr[k];
        for( ; k < dcn; k++ )
            dptr[k] = 0;
    }

    return true;
}

}

#endif

#ifdef HAVE_OPENVX
namespace cv
{
    static bool openvx_meanStdDev(Mat& src, OutputArray _mean, OutputArray _sdv, Mat& mask)
    {
        size_t total_size = src.total();
        int rows = src.size[0], cols = rows ? (int)(total_size / rows) : 0;
        if (src.type() != CV_8UC1|| !mask.empty() ||
               (src.dims != 2 && !(src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size))
           )
        return false;

        try
        {
            ivx::Context ctx = ovx::getOpenVXContext();
#ifndef VX_VERSION_1_1
            if (ctx.vendorID() == VX_ID_KHRONOS)
                return false; // Do not use OpenVX meanStdDev estimation for sample 1.0.1 implementation due to lack of accuracy
#endif

            ivx::Image
                ia = ivx::Image::createFromHandle(ctx, VX_DF_IMAGE_U8,
                    ivx::Image::createAddressing(cols, rows, 1, (vx_int32)(src.step[0])), src.ptr());

            vx_float32 mean_temp, stddev_temp;
            ivx::IVX_CHECK_STATUS(vxuMeanStdDev(ctx, ia, &mean_temp, &stddev_temp));

            if (_mean.needed())
            {
                if (!_mean.fixedSize())
                    _mean.create(1, 1, CV_64F, -1, true);
                Mat mean = _mean.getMat();
                CV_Assert(mean.type() == CV_64F && mean.isContinuous() &&
                    (mean.cols == 1 || mean.rows == 1) && mean.total() >= 1);
                double *pmean = mean.ptr<double>();
                pmean[0] = mean_temp;
                for (int c = 1; c < (int)mean.total(); c++)
                    pmean[c] = 0;
            }

            if (_sdv.needed())
            {
                if (!_sdv.fixedSize())
                    _sdv.create(1, 1, CV_64F, -1, true);
                Mat stddev = _sdv.getMat();
                CV_Assert(stddev.type() == CV_64F && stddev.isContinuous() &&
                    (stddev.cols == 1 || stddev.rows == 1) && stddev.total() >= 1);
                double *pstddev = stddev.ptr<double>();
                pstddev[0] = stddev_temp;
                for (int c = 1; c < (int)stddev.total(); c++)
                    pstddev[c] = 0;
            }
        }
        catch (ivx::RuntimeError & e)
        {
            VX_DbgThrow(e.what());
        }
        catch (ivx::WrapperError & e)
        {
            VX_DbgThrow(e.what());
        }

        return true;
    }
}
#endif

#ifdef HAVE_IPP
namespace cv
{
static bool ipp_meanStdDev(Mat& src, OutputArray _mean, OutputArray _sdv, Mat& mask)
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 >= 700
    int cn = src.channels();

#if IPP_VERSION_X100 < 201801
    // IPP_DISABLE: C3C functions can read outside of allocated memory
    if (cn > 1)
        return false;
#endif

    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( src.dims == 2 || (src.isContinuous() && mask.isContinuous() && cols > 0 && (size_t)rows*cols == total_size) )
    {
        Ipp64f mean_temp[3];
        Ipp64f stddev_temp[3];
        Ipp64f *pmean = &mean_temp[0];
        Ipp64f *pstddev = &stddev_temp[0];
        Mat mean, stddev;
        int dcn_mean = -1;
        if( _mean.needed() )
        {
            if( !_mean.fixedSize() )
                _mean.create(cn, 1, CV_64F, -1, true);
            mean = _mean.getMat();
            dcn_mean = (int)mean.total();
            pmean = mean.ptr<Ipp64f>();
        }
        int dcn_stddev = -1;
        if( _sdv.needed() )
        {
            if( !_sdv.fixedSize() )
                _sdv.create(cn, 1, CV_64F, -1, true);
            stddev = _sdv.getMat();
            dcn_stddev = (int)stddev.total();
            pstddev = stddev.ptr<Ipp64f>();
        }
        for( int c = cn; c < dcn_mean; c++ )
            pmean[c] = 0;
        for( int c = cn; c < dcn_stddev; c++ )
            pstddev[c] = 0;
        IppiSize sz = { cols, rows };
        int type = src.type();
        if( !mask.empty() )
        {
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC1 ippiMean_StdDev_C1MR =
            type == CV_8UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_8u_C1MR :
            type == CV_16UC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_16u_C1MR :
            type == CV_32FC1 ? (ippiMaskMeanStdDevFuncC1)ippiMean_StdDev_32f_C1MR :
            0;
            if( ippiMean_StdDev_C1MR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C1MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, pmean, pstddev) >= 0 )
                {
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskMeanStdDevFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMaskMeanStdDevFuncC3 ippiMean_StdDev_C3CMR =
            type == CV_8UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CMR :
            type == CV_16UC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CMR :
            type == CV_32FC3 ? (ippiMaskMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CMR :
            0;
            if( ippiMean_StdDev_C3CMR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CMR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                {
                    return true;
                }
            }
        }
        else
        {
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC1)(const void *, int, IppiSize, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC1 ippiMean_StdDev_C1R =
            type == CV_8UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_8u_C1R :
            type == CV_16UC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_16u_C1R :
#if (IPP_VERSION_X100 >= 810)
            type == CV_32FC1 ? (ippiMeanStdDevFuncC1)ippiMean_StdDev_32f_C1R ://Aug 2013: bug in IPP 7.1, 8.0
#endif
            0;
            if( ippiMean_StdDev_C1R )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C1R, src.ptr(), (int)src.step[0], sz, pmean, pstddev) >= 0 )
                {
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMeanStdDevFuncC3)(const void *, int, IppiSize, int, Ipp64f *, Ipp64f *);
            ippiMeanStdDevFuncC3 ippiMean_StdDev_C3CR =
            type == CV_8UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_8u_C3CR :
            type == CV_16UC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_16u_C3CR :
            type == CV_32FC3 ? (ippiMeanStdDevFuncC3)ippiMean_StdDev_32f_C3CR :
            0;
            if( ippiMean_StdDev_C3CR )
            {
                if( CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src.ptr(), (int)src.step[0], sz, 1, &pmean[0], &pstddev[0]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src.ptr(), (int)src.step[0], sz, 2, &pmean[1], &pstddev[1]) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiMean_StdDev_C3CR, src.ptr(), (int)src.step[0], sz, 3, &pmean[2], &pstddev[2]) >= 0 )
                {
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(_mean); CV_UNUSED(_sdv); CV_UNUSED(mask);
#endif
    return false;
}
}
#endif

void cv::meanStdDev( InputArray _src, OutputArray _mean, OutputArray _sdv, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    CV_OCL_RUN(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
               ocl_meanStdDev(_src, _mean, _sdv, _mask))

    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_Assert( mask.empty() || mask.type() == CV_8UC1 );

    CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_MEAN_STDDEV>(src.cols, src.rows),
               openvx_meanStdDev(src, _mean, _sdv, mask))

    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_meanStdDev(src, _mean, _sdv, mask));

    int k, cn = src.channels(), depth = src.depth();

    SumSqrFunc func = getSumSqrTab(depth);

    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int total = (int)it.size, blockSize = total, intSumBlockSize = 0;
    int j, count = 0, nz0 = 0;
    AutoBuffer<double> _buf(cn*4);
    double *s = (double*)_buf, *sq = s + cn;
    int *sbuf = (int*)s, *sqbuf = (int*)sq;
    bool blockSum = depth <= CV_16S, blockSqSum = depth <= CV_8S;
    size_t esz = 0;

    for( k = 0; k < cn; k++ )
        s[k] = sq[k] = 0;

    if( blockSum )
    {
        intSumBlockSize = 1 << 15;
        blockSize = std::min(blockSize, intSumBlockSize);
        sbuf = (int*)(sq + cn);
        if( blockSqSum )
            sqbuf = sbuf + cn;
        for( k = 0; k < cn; k++ )
            sbuf[k] = sqbuf[k] = 0;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            int nz = func( ptrs[0], ptrs[1], (uchar*)sbuf, (uchar*)sqbuf, bsz, cn );
            count += nz;
            nz0 += nz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                for( k = 0; k < cn; k++ )
                {
                    s[k] += sbuf[k];
                    sbuf[k] = 0;
                }
                if( blockSqSum )
                {
                    for( k = 0; k < cn; k++ )
                    {
                        sq[k] += sqbuf[k];
                        sqbuf[k] = 0;
                    }
                }
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }

    double scale = nz0 ? 1./nz0 : 0.;
    for( k = 0; k < cn; k++ )
    {
        s[k] *= scale;
        sq[k] = std::sqrt(std::max(sq[k]*scale - s[k]*s[k], 0.));
    }

    for( j = 0; j < 2; j++ )
    {
        const double* sptr = j == 0 ? s : sq;
        _OutputArray _dst = j == 0 ? _mean : _sdv;
        if( !_dst.needed() )
            continue;

        if( !_dst.fixedSize() )
            _dst.create(cn, 1, CV_64F, -1, true);
        Mat dst = _dst.getMat();
        int dcn = (int)dst.total();
        CV_Assert( dst.type() == CV_64F && dst.isContinuous() &&
                   (dst.cols == 1 || dst.rows == 1) && dcn >= cn );
        double* dptr = dst.ptr<double>();
        for( k = 0; k < cn; k++ )
            dptr[k] = sptr[k];
        for( ; k < dcn; k++ )
            dptr[k] = 0;
    }
}

/****************************************************************************************\
*                                       minMaxLoc                                        *
\****************************************************************************************/

namespace cv
{

template<typename T, typename WT> static void
minMaxIdx_( const T* src, const uchar* mask, WT* _minVal, WT* _maxVal,
            size_t* _minIdx, size_t* _maxIdx, int len, size_t startIdx )
{
    WT minVal = *_minVal, maxVal = *_maxVal;
    size_t minIdx = *_minIdx, maxIdx = *_maxIdx;

    if( !mask )
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }
    else
    {
        for( int i = 0; i < len; i++ )
        {
            T val = src[i];
            if( mask[i] && val < minVal )
            {
                minVal = val;
                minIdx = startIdx + i;
            }
            if( mask[i] && val > maxVal )
            {
                maxVal = val;
                maxIdx = startIdx + i;
            }
        }
    }

    *_minIdx = minIdx;
    *_maxIdx = maxIdx;
    *_minVal = minVal;
    *_maxVal = maxVal;
}

static void minMaxIdx_8u(const uchar* src, const uchar* mask, int* minval, int* maxval,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_8s(const schar* src, const uchar* mask, int* minval, int* maxval,
                         size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_16u(const ushort* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_16s(const short* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_32s(const int* src, const uchar* mask, int* minval, int* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_32f(const float* src, const uchar* mask, float* minval, float* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

static void minMaxIdx_64f(const double* src, const uchar* mask, double* minval, double* maxval,
                          size_t* minidx, size_t* maxidx, int len, size_t startidx )
{ minMaxIdx_(src, mask, minval, maxval, minidx, maxidx, len, startidx ); }

typedef void (*MinMaxIdxFunc)(const uchar*, const uchar*, int*, int*, size_t*, size_t*, int, size_t);

static MinMaxIdxFunc getMinmaxTab(int depth)
{
    static MinMaxIdxFunc minmaxTab[] =
    {
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_8u), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_8s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_16u), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_16s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_32s),
        (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_32f), (MinMaxIdxFunc)GET_OPTIMIZED(minMaxIdx_64f),
        0
    };

    return minmaxTab[depth];
}

static void ofs2idx(const Mat& a, size_t ofs, int* idx)
{
    int i, d = a.dims;
    if( ofs > 0 )
    {
        ofs--;
        for( i = d-1; i >= 0; i-- )
        {
            int sz = a.size[i];
            idx[i] = (int)(ofs % sz);
            ofs /= sz;
        }
    }
    else
    {
        for( i = d-1; i >= 0; i-- )
            idx[i] = -1;
    }
}

#ifdef HAVE_OPENCL

#define MINMAX_STRUCT_ALIGNMENT 8 // sizeof double

template <typename T>
void getMinMaxRes(const Mat & db, double * minVal, double * maxVal,
                  int* minLoc, int* maxLoc,
                  int groupnum, int cols, double * maxVal2)
{
    uint index_max = std::numeric_limits<uint>::max();
    T minval = std::numeric_limits<T>::max();
    T maxval = std::numeric_limits<T>::min() > 0 ? -std::numeric_limits<T>::max() : std::numeric_limits<T>::min(), maxval2 = maxval;
    uint minloc = index_max, maxloc = index_max;

    size_t index = 0;
    const T * minptr = NULL, * maxptr = NULL, * maxptr2 = NULL;
    const uint * minlocptr = NULL, * maxlocptr = NULL;
    if (minVal || minLoc)
    {
        minptr = db.ptr<T>();
        index += sizeof(T) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (maxVal || maxLoc)
    {
        maxptr = (const T *)(db.ptr() + index);
        index += sizeof(T) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (minLoc)
    {
        minlocptr = (const uint *)(db.ptr() + index);
        index += sizeof(uint) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (maxLoc)
    {
        maxlocptr = (const uint *)(db.ptr() + index);
        index += sizeof(uint) * groupnum;
        index = alignSize(index, MINMAX_STRUCT_ALIGNMENT);
    }
    if (maxVal2)
        maxptr2 = (const T *)(db.ptr() + index);

    for (int i = 0; i < groupnum; i++)
    {
        if (minptr && minptr[i] <= minval)
        {
            if (minptr[i] == minval)
            {
                if (minlocptr)
                    minloc = std::min(minlocptr[i], minloc);
            }
            else
            {
                if (minlocptr)
                    minloc = minlocptr[i];
                minval = minptr[i];
            }
        }
        if (maxptr && maxptr[i] >= maxval)
        {
            if (maxptr[i] == maxval)
            {
                if (maxlocptr)
                    maxloc = std::min(maxlocptr[i], maxloc);
            }
            else
            {
                if (maxlocptr)
                    maxloc = maxlocptr[i];
                maxval = maxptr[i];
            }
        }
        if (maxptr2 && maxptr2[i] > maxval2)
            maxval2 = maxptr2[i];
    }
    bool zero_mask = (minLoc && minloc == index_max) ||
            (maxLoc && maxloc == index_max);

    if (minVal)
        *minVal = zero_mask ? 0 : (double)minval;
    if (maxVal)
        *maxVal = zero_mask ? 0 : (double)maxval;
    if (maxVal2)
        *maxVal2 = zero_mask ? 0 : (double)maxval2;

    if (minLoc)
    {
        minLoc[0] = zero_mask ? -1 : minloc / cols;
        minLoc[1] = zero_mask ? -1 : minloc % cols;
    }
    if (maxLoc)
    {
        maxLoc[0] = zero_mask ? -1 : maxloc / cols;
        maxLoc[1] = zero_mask ? -1 : maxloc % cols;
    }
}

typedef void (*getMinMaxResFunc)(const Mat & db, double * minVal, double * maxVal,
                                 int * minLoc, int *maxLoc, int gropunum, int cols, double * maxVal2);

static bool ocl_minMaxIdx( InputArray _src, double* minVal, double* maxVal, int* minLoc, int* maxLoc, InputArray _mask,
                           int ddepth = -1, bool absValues = false, InputArray _src2 = noArray(), double * maxVal2 = NULL)
{
    const ocl::Device & dev = ocl::Device::getDefault();

#ifdef __ANDROID__
    if (dev.isNVidia())
        return false;
#endif

    bool doubleSupport = dev.doubleFPConfig() > 0, haveMask = !_mask.empty(),
        haveSrc2 = _src2.kind() != _InputArray::NONE;
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type),
            kercn = haveMask ? cn : std::min(4, ocl::predictOptimalVectorWidth(_src, _src2));

    // disabled following modes since it occasionally fails on AMD devices (e.g. A10-6800K, sep. 2014)
    if ((haveMask || type == CV_32FC1) && dev.isAMD())
        return false;

    CV_Assert( (cn == 1 && (!haveMask || _mask.type() == CV_8U)) ||
              (cn >= 1 && !minLoc && !maxLoc) );

    if (ddepth < 0)
        ddepth = depth;

    CV_Assert(!haveSrc2 || _src2.type() == type);

    if (depth == CV_32S)
        return false;

    if ((depth == CV_64F || ddepth == CV_64F) && !doubleSupport)
        return false;

    int groupnum = dev.maxComputeUnits();
    size_t wgs = dev.maxWorkGroupSize();

    int wgs2_aligned = 1;
    while (wgs2_aligned < (int)wgs)
        wgs2_aligned <<= 1;
    wgs2_aligned >>= 1;

    bool needMinVal = minVal || minLoc, needMinLoc = minLoc != NULL,
            needMaxVal = maxVal || maxLoc, needMaxLoc = maxLoc != NULL;

    // in case of mask we must know whether mask is filled with zeros or not
    // so let's calculate min or max location, if it's undefined, so mask is zeros
    if (!(needMaxLoc || needMinLoc) && haveMask)
    {
        if (needMinVal)
            needMinLoc = true;
        else
            needMaxLoc = true;
    }

    char cvt[2][40];
    String opts = format("-D DEPTH_%d -D srcT1=%s%s -D WGS=%d -D srcT=%s"
                         " -D WGS2_ALIGNED=%d%s%s%s -D kercn=%d%s%s%s%s"
                         " -D dstT1=%s -D dstT=%s -D convertToDT=%s%s%s%s%s -D wdepth=%d -D convertFromU=%s"
                         " -D MINMAX_STRUCT_ALIGNMENT=%d",
                         depth, ocl::typeToStr(depth), haveMask ? " -D HAVE_MASK" : "", (int)wgs,
                         ocl::typeToStr(CV_MAKE_TYPE(depth, kercn)), wgs2_aligned,
                         doubleSupport ? " -D DOUBLE_SUPPORT" : "",
                         _src.isContinuous() ? " -D HAVE_SRC_CONT" : "",
                         _mask.isContinuous() ? " -D HAVE_MASK_CONT" : "", kercn,
                         needMinVal ? " -D NEED_MINVAL" : "", needMaxVal ? " -D NEED_MAXVAL" : "",
                         needMinLoc ? " -D NEED_MINLOC" : "", needMaxLoc ? " -D NEED_MAXLOC" : "",
                         ocl::typeToStr(ddepth), ocl::typeToStr(CV_MAKE_TYPE(ddepth, kercn)),
                         ocl::convertTypeStr(depth, ddepth, kercn, cvt[0]),
                         absValues ? " -D OP_ABS" : "",
                         haveSrc2 ? " -D HAVE_SRC2" : "", maxVal2 ? " -D OP_CALC2" : "",
                         haveSrc2 && _src2.isContinuous() ? " -D HAVE_SRC2_CONT" : "", ddepth,
                         depth <= CV_32S && ddepth == CV_32S ? ocl::convertTypeStr(CV_8U, ddepth, kercn, cvt[1]) : "noconvert",
                         MINMAX_STRUCT_ALIGNMENT);

    ocl::Kernel k("minmaxloc", ocl::core::minmaxloc_oclsrc, opts);
    if (k.empty())
        return false;

    int esz = CV_ELEM_SIZE(ddepth), esz32s = CV_ELEM_SIZE1(CV_32S),
            dbsize = groupnum * ((needMinVal ? esz : 0) + (needMaxVal ? esz : 0) +
                                 (needMinLoc ? esz32s : 0) + (needMaxLoc ? esz32s : 0) +
                                 (maxVal2 ? esz : 0))
                     + 5 * MINMAX_STRUCT_ALIGNMENT;
    UMat src = _src.getUMat(), src2 = _src2.getUMat(), db(1, dbsize, CV_8UC1), mask = _mask.getUMat();

    if (cn > 1 && !haveMask)
    {
        src = src.reshape(1);
        src2 = src2.reshape(1);
    }

    if (haveSrc2)
    {
        if (!haveMask)
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db), ocl::KernelArg::ReadOnlyNoSize(src2));
        else
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db), ocl::KernelArg::ReadOnlyNoSize(mask),
                   ocl::KernelArg::ReadOnlyNoSize(src2));
    }
    else
    {
        if (!haveMask)
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db));
        else
            k.args(ocl::KernelArg::ReadOnlyNoSize(src), src.cols, (int)src.total(),
                   groupnum, ocl::KernelArg::PtrWriteOnly(db), ocl::KernelArg::ReadOnlyNoSize(mask));
    }

    size_t globalsize = groupnum * wgs;
    if (!k.run(1, &globalsize, &wgs, true))
        return false;

    static const getMinMaxResFunc functab[7] =
    {
        getMinMaxRes<uchar>,
        getMinMaxRes<char>,
        getMinMaxRes<ushort>,
        getMinMaxRes<short>,
        getMinMaxRes<int>,
        getMinMaxRes<float>,
        getMinMaxRes<double>
    };

    getMinMaxResFunc func = functab[ddepth];

    int locTemp[2];
    func(db.getMat(ACCESS_READ), minVal, maxVal,
         needMinLoc ? minLoc ? minLoc : locTemp : minLoc,
         needMaxLoc ? maxLoc ? maxLoc : locTemp : maxLoc,
         groupnum, src.cols, maxVal2);

    return true;
}

#endif

#ifdef HAVE_OPENVX
namespace ovx {
    template <> inline bool skipSmallImages<VX_KERNEL_MINMAXLOC>(int w, int h) { return w*h < 3840 * 2160; }
}
static bool openvx_minMaxIdx(Mat &src, double* minVal, double* maxVal, int* minIdx, int* maxIdx, Mat &mask)
{
    int stype = src.type();
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size / rows) : 0;
    if ((stype != CV_8UC1 && stype != CV_16SC1) || !mask.empty() ||
        (src.dims != 2 && !(src.isContinuous() && cols > 0 && (size_t)rows*cols == total_size))
        )
        return false;

    try
    {
        ivx::Context ctx = ovx::getOpenVXContext();
        ivx::Image
            ia = ivx::Image::createFromHandle(ctx, stype == CV_8UC1 ? VX_DF_IMAGE_U8 : VX_DF_IMAGE_S16,
                ivx::Image::createAddressing(cols, rows, stype == CV_8UC1 ? 1 : 2, (vx_int32)(src.step[0])), src.ptr());

        ivx::Scalar vxMinVal = ivx::Scalar::create(ctx, stype == CV_8UC1 ? VX_TYPE_UINT8 : VX_TYPE_INT16, 0);
        ivx::Scalar vxMaxVal = ivx::Scalar::create(ctx, stype == CV_8UC1 ? VX_TYPE_UINT8 : VX_TYPE_INT16, 0);
        ivx::Array vxMinInd, vxMaxInd;
        ivx::Scalar vxMinCount, vxMaxCount;
        if (minIdx)
        {
            vxMinInd = ivx::Array::create(ctx, VX_TYPE_COORDINATES2D, 1);
            vxMinCount = ivx::Scalar::create(ctx, VX_TYPE_UINT32, 0);
        }
        if (maxIdx)
        {
            vxMaxInd = ivx::Array::create(ctx, VX_TYPE_COORDINATES2D, 1);
            vxMaxCount = ivx::Scalar::create(ctx, VX_TYPE_UINT32, 0);
        }

        ivx::IVX_CHECK_STATUS(vxuMinMaxLoc(ctx, ia, vxMinVal, vxMaxVal, vxMinInd, vxMaxInd, vxMinCount, vxMaxCount));

        if (minVal)
        {
            *minVal = stype == CV_8UC1 ? vxMinVal.getValue<vx_uint8>() : vxMinVal.getValue<vx_int16>();
        }
        if (maxVal)
        {
            *maxVal = stype == CV_8UC1 ? vxMaxVal.getValue<vx_uint8>() : vxMaxVal.getValue<vx_int16>();
        }
        if (minIdx)
        {
            if(vxMinCount.getValue<vx_uint32>()<1) throw ivx::RuntimeError(VX_ERROR_INVALID_VALUE, std::string(__func__) + "(): minimum value location not found");
            vx_coordinates2d_t loc;
            vxMinInd.copyRangeTo(0, 1, &loc);
            size_t minidx = loc.y * cols + loc.x + 1;
            ofs2idx(src, minidx, minIdx);
        }
        if (maxIdx)
        {
            if (vxMaxCount.getValue<vx_uint32>()<1) throw ivx::RuntimeError(VX_ERROR_INVALID_VALUE, std::string(__func__) + "(): maximum value location not found");
            vx_coordinates2d_t loc;
            vxMaxInd.copyRangeTo(0, 1, &loc);
            size_t maxidx = loc.y * cols + loc.x + 1;
            ofs2idx(src, maxidx, maxIdx);
        }
    }
    catch (ivx::RuntimeError & e)
    {
        VX_DbgThrow(e.what());
    }
    catch (ivx::WrapperError & e)
    {
        VX_DbgThrow(e.what());
    }

    return true;
}
#endif

#ifdef HAVE_IPP
static IppStatus ipp_minMaxIndex_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint* pMinIndex, IppiPoint* pMaxIndex, const Ipp8u*, int)
{
    switch(dataType)
    {
    case ipp8u:  return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp16u: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    default:     return ippStsDataTypeErr;
    }
}

static IppStatus ipp_minMaxIndexMask_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint* pMinIndex, IppiPoint* pMaxIndex, const Ipp8u* pMask, int maskStep)
{
    switch(dataType)
    {
    case ipp8u:  return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_8u_C1MR, (const Ipp8u*)pSrc, srcStep, pMask, maskStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp16u: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_16u_C1MR, (const Ipp16u*)pSrc, srcStep, pMask, maskStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinMaxIndx_32f_C1MR, (const Ipp32f*)pSrc, srcStep, pMask, maskStep, size, pMinVal, pMaxVal, pMinIndex, pMaxIndex);
    default:     return ippStsDataTypeErr;
    }
}

static IppStatus ipp_minMax_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint*, IppiPoint*, const Ipp8u*, int)
{
    IppStatus status;

    switch(dataType)
    {
#if IPP_VERSION_X100 > 201701 // wrong min values
    case ipp8u:
    {
        Ipp8u val[2];
        status = CV_INSTRUMENT_FUN_IPP(ippiMinMax_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, &val[0], &val[1]);
        *pMinVal = val[0];
        *pMaxVal = val[1];
        return status;
    }
#endif
    case ipp16u:
    {
        Ipp16u val[2];
        status = CV_INSTRUMENT_FUN_IPP(ippiMinMax_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, &val[0], &val[1]);
        *pMinVal = val[0];
        *pMaxVal = val[1];
        return status;
    }
    case ipp16s:
    {
        Ipp16s val[2];
        status = CV_INSTRUMENT_FUN_IPP(ippiMinMax_16s_C1R, (const Ipp16s*)pSrc, srcStep, size, &val[0], &val[1]);
        *pMinVal = val[0];
        *pMaxVal = val[1];
        return status;
    }
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinMax_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMinVal, pMaxVal);
    default:     return ipp_minMaxIndex_wrap(pSrc, srcStep, size, dataType, pMinVal, pMaxVal, NULL, NULL, NULL, 0);
    }
}

static IppStatus ipp_minIdx_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float*, IppiPoint* pMinIndex, IppiPoint*, const Ipp8u*, int)
{
    IppStatus status;

    switch(dataType)
    {
    case ipp8u:
    {
        Ipp8u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMinIndx_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, &val, &pMinIndex->x, &pMinIndex->y);
        *pMinVal = val;
        return status;
    }
    case ipp16u:
    {
        Ipp16u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMinIndx_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, &val, &pMinIndex->x, &pMinIndex->y);
        *pMinVal = val;
        return status;
    }
    case ipp16s:
    {
        Ipp16s val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMinIndx_16s_C1R, (const Ipp16s*)pSrc, srcStep, size, &val, &pMinIndex->x, &pMinIndex->y);
        *pMinVal = val;
        return status;
    }
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMinIndx_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMinVal, &pMinIndex->x, &pMinIndex->y);
    default:     return ipp_minMaxIndex_wrap(pSrc, srcStep, size, dataType, pMinVal, NULL, pMinIndex, NULL, NULL, 0);
    }
}

static IppStatus ipp_maxIdx_wrap(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float*, float* pMaxVal, IppiPoint*, IppiPoint* pMaxIndex, const Ipp8u*, int)
{
    IppStatus status;

    switch(dataType)
    {
    case ipp8u:
    {
        Ipp8u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_8u_C1R, (const Ipp8u*)pSrc, srcStep, size, &val, &pMaxIndex->x, &pMaxIndex->y);
        *pMaxVal = val;
        return status;
    }
    case ipp16u:
    {
        Ipp16u val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_16u_C1R, (const Ipp16u*)pSrc, srcStep, size, &val, &pMaxIndex->x, &pMaxIndex->y);
        *pMaxVal = val;
        return status;
    }
    case ipp16s:
    {
        Ipp16s val;
        status = CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_16s_C1R, (const Ipp16s*)pSrc, srcStep, size, &val, &pMaxIndex->x, &pMaxIndex->y);
        *pMaxVal = val;
        return status;
    }
    case ipp32f: return CV_INSTRUMENT_FUN_IPP(ippiMaxIndx_32f_C1R, (const Ipp32f*)pSrc, srcStep, size, pMaxVal, &pMaxIndex->x, &pMaxIndex->y);
    default:     return ipp_minMaxIndex_wrap(pSrc, srcStep, size, dataType, NULL, pMaxVal, NULL, pMaxIndex, NULL, 0);
    }
}

typedef IppStatus (*IppMinMaxSelector)(const void* pSrc, int srcStep, IppiSize size, IppDataType dataType,
    float* pMinVal, float* pMaxVal, IppiPoint* pMinIndex, IppiPoint* pMaxIndex, const Ipp8u* pMask, int maskStep);

static bool ipp_minMaxIdx(Mat &src, double* _minVal, double* _maxVal, int* _minIdx, int* _maxIdx, Mat &mask)
{
#if IPP_VERSION_X100 >= 700
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 < 201800
    // cv::minMaxIdx problem with NaN input
    // Disable 32F processing only
    if(src.depth() == CV_32F && cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;
#endif

#if IPP_VERSION_X100 < 201801
    // cv::minMaxIdx problem with index positions on AVX
    if(!mask.empty() && _maxIdx && cv::ipp::getIppTopFeatures() != ippCPUID_SSE42)
        return false;
#endif

    IppStatus   status;
    IppDataType dataType = ippiGetDataType(src.depth());
    float       minVal = 0;
    float       maxVal = 0;
    IppiPoint   minIdx = {-1, -1};
    IppiPoint   maxIdx = {-1, -1};

    float       *pMinVal = (_minVal || _minIdx)?&minVal:NULL;
    float       *pMaxVal = (_maxVal || _maxIdx)?&maxVal:NULL;
    IppiPoint   *pMinIdx = (_minIdx)?&minIdx:NULL;
    IppiPoint   *pMaxIdx = (_maxIdx)?&maxIdx:NULL;

    IppMinMaxSelector ippMinMaxFun = ipp_minMaxIndexMask_wrap;
    if(mask.empty())
    {
        if(_maxVal && _maxIdx && !_minVal && !_minIdx)
            ippMinMaxFun = ipp_maxIdx_wrap;
        else if(!_maxVal && !_maxIdx && _minVal && _minIdx)
            ippMinMaxFun = ipp_minIdx_wrap;
        else if(_maxVal && !_maxIdx && _minVal && !_minIdx)
            ippMinMaxFun = ipp_minMax_wrap;
        else if(!_maxVal && !_maxIdx && !_minVal && !_minIdx)
            return false;
        else
            ippMinMaxFun = ipp_minMaxIndex_wrap;
    }

    if(src.dims <= 2)
    {
        IppiSize size = ippiSize(src.size());
        size.width *= src.channels();

        status = ippMinMaxFun(src.ptr(), (int)src.step, size, dataType, pMinVal, pMaxVal, pMinIdx, pMaxIdx, (Ipp8u*)mask.ptr(), (int)mask.step);
        if(status < 0)
            return false;
        if(_minVal)
            *_minVal = minVal;
        if(_maxVal)
            *_maxVal = maxVal;
        if(_minIdx)
        {
#if IPP_VERSION_X100 < 201801
            // Should be just ippStsNoOperation check, but there is a bug in the function so we need additional checks
            if(status == ippStsNoOperation && !mask.empty() && !pMinIdx->x && !pMinIdx->y)
#else
            if(status == ippStsNoOperation)
#endif
            {
                _minIdx[0] = -1;
                _minIdx[1] = -1;
            }
            else
            {
                _minIdx[0] = minIdx.y;
                _minIdx[1] = minIdx.x;
            }
        }
        if(_maxIdx)
        {
#if IPP_VERSION_X100 < 201801
            // Should be just ippStsNoOperation check, but there is a bug in the function so we need additional checks
            if(status == ippStsNoOperation && !mask.empty() && !pMaxIdx->x && !pMaxIdx->y)
#else
            if(status == ippStsNoOperation)
#endif
            {
                _maxIdx[0] = -1;
                _maxIdx[1] = -1;
            }
            else
            {
                _maxIdx[0] = maxIdx.y;
                _maxIdx[1] = maxIdx.x;
            }
        }
    }
    else
    {
        const Mat *arrays[] = {&src, mask.empty()?NULL:&mask, NULL};
        uchar     *ptrs[3]  = {NULL};
        NAryMatIterator it(arrays, ptrs);
        IppiSize size = ippiSize(it.size*src.channels(), 1);
        int srcStep      = (int)(size.width*src.elemSize1());
        int maskStep     = size.width;
        size_t idxPos    = 1;
        size_t minIdxAll = 0;
        size_t maxIdxAll = 0;
        float  minValAll = IPP_MAXABS_32F;
        float  maxValAll = -IPP_MAXABS_32F;

        for(size_t i = 0; i < it.nplanes; i++, ++it, idxPos += size.width)
        {
            status = ippMinMaxFun(ptrs[0], srcStep, size, dataType, pMinVal, pMaxVal, pMinIdx, pMaxIdx, ptrs[1], maskStep);
            if(status < 0)
                return false;
#if IPP_VERSION_X100 > 201701
            // Zero-mask check, function should return ippStsNoOperation warning
            if(status == ippStsNoOperation)
                    continue;
#else
            // Crude zero-mask check, waiting for fix in IPP function
            if(ptrs[1])
            {
                Mat localMask(Size(size.width, 1), CV_8U, ptrs[1], maskStep);
                if(!cv::countNonZero(localMask))
                    continue;
            }
#endif

            if(_minVal && minVal < minValAll)
            {
                minValAll = minVal;
                minIdxAll = idxPos+minIdx.x;
            }
            if(_maxVal && maxVal > maxValAll)
            {
                maxValAll = maxVal;
                maxIdxAll = idxPos+maxIdx.x;
            }
        }
        if(!src.empty() && mask.empty())
        {
            if(minIdxAll == 0)
                minIdxAll = 1;
            if(maxValAll == 0)
                maxValAll = 1;
        }

        if(_minVal)
            *_minVal = minValAll;
        if(_maxVal)
            *_maxVal = maxValAll;
        if(_minIdx)
            ofs2idx(src, minIdxAll, _minIdx);
        if(_maxIdx)
            ofs2idx(src, maxIdxAll, _maxIdx);
    }

    return true;
#else
    CV_UNUSED(src); CV_UNUSED(minVal); CV_UNUSED(maxVal); CV_UNUSED(minIdx); CV_UNUSED(maxIdx); CV_UNUSED(mask);
    return false;
#endif
}
#endif

}

void cv::minMaxIdx(InputArray _src, double* minVal,
                   double* maxVal, int* minIdx, int* maxIdx,
                   InputArray _mask)
{
    CV_INSTRUMENT_REGION()

    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    CV_Assert( (cn == 1 && (_mask.empty() || _mask.type() == CV_8U)) ||
        (cn > 1 && _mask.empty() && !minIdx && !maxIdx) );

    CV_OCL_RUN(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2  && (_mask.empty() || _src.size() == _mask.size()),
               ocl_minMaxIdx(_src, minVal, maxVal, minIdx, maxIdx, _mask))

    Mat src = _src.getMat(), mask = _mask.getMat();

    CV_OVX_RUN(!ovx::skipSmallImages<VX_KERNEL_MINMAXLOC>(src.cols, src.rows),
               openvx_minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask))

    CV_IPP_RUN_FAST(ipp_minMaxIdx(src, minVal, maxVal, minIdx, maxIdx, mask))

    MinMaxIdxFunc func = getMinmaxTab(depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);

    size_t minidx = 0, maxidx = 0;
    int iminval = INT_MAX, imaxval = INT_MIN;
    float  fminval = std::numeric_limits<float>::infinity(),  fmaxval = -fminval;
    double dminval = std::numeric_limits<double>::infinity(), dmaxval = -dminval;
    size_t startidx = 1;
    int *minval = &iminval, *maxval = &imaxval;
    int planeSize = (int)it.size*cn;

    if( depth == CV_32F )
        minval = (int*)&fminval, maxval = (int*)&fmaxval;
    else if( depth == CV_64F )
        minval = (int*)&dminval, maxval = (int*)&dmaxval;

    for( size_t i = 0; i < it.nplanes; i++, ++it, startidx += planeSize )
        func( ptrs[0], ptrs[1], minval, maxval, &minidx, &maxidx, planeSize, startidx );

    if (!src.empty() && mask.empty())
    {
        if( minidx == 0 )
             minidx = 1;
         if( maxidx == 0 )
             maxidx = 1;
    }

    if( minidx == 0 )
        dminval = dmaxval = 0;
    else if( depth == CV_32F )
        dminval = fminval, dmaxval = fmaxval;
    else if( depth <= CV_32S )
        dminval = iminval, dmaxval = imaxval;

    if( minVal )
        *minVal = dminval;
    if( maxVal )
        *maxVal = dmaxval;

    if( minIdx )
        ofs2idx(src, minidx, minIdx);
    if( maxIdx )
        ofs2idx(src, maxidx, maxIdx);
}

void cv::minMaxLoc( InputArray _img, double* minVal, double* maxVal,
                    Point* minLoc, Point* maxLoc, InputArray mask )
{
    CV_INSTRUMENT_REGION()

    CV_Assert(_img.dims() <= 2);

    minMaxIdx(_img, minVal, maxVal, (int*)minLoc, (int*)maxLoc, mask);
    if( minLoc )
        std::swap(minLoc->x, minLoc->y);
    if( maxLoc )
        std::swap(maxLoc->x, maxLoc->y);
}

/****************************************************************************************\
*                                         norm                                           *
\****************************************************************************************/

namespace cv
{

template<typename T, typename ST> int
normInf_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result = std::max(result, normInf<T, ST>(src, len*cn));
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result = std::max(result, ST(cv_abs(src[k])));
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL1_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL1<T, ST>(src, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result += cv_abs(src[k]);
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normL2_(const T* src, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL2Sqr<T, ST>(src, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    T v = src[k];
                    result += (ST)v*v;
                }
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffInf_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result = std::max(result, normInf<T, ST>(src1, src2, len*cn));
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result = std::max(result, (ST)std::abs(src1[k] - src2[k]));
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL1_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL1<T, ST>(src1, src2, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                    result += std::abs(src1[k] - src2[k]);
            }
    }
    *_result = result;
    return 0;
}

template<typename T, typename ST> int
normDiffL2_(const T* src1, const T* src2, const uchar* mask, ST* _result, int len, int cn)
{
    ST result = *_result;
    if( !mask )
    {
        result += normL2Sqr<T, ST>(src1, src2, len*cn);
    }
    else
    {
        for( int i = 0; i < len; i++, src1 += cn, src2 += cn )
            if( mask[i] )
            {
                for( int k = 0; k < cn; k++ )
                {
                    ST v = src1[k] - src2[k];
                    result += v*v;
                }
            }
    }
    *_result = result;
    return 0;
}

Hamming::ResultType Hamming::operator()( const unsigned char* a, const unsigned char* b, int size ) const
{
    return cv::hal::normHamming(a, b, size);
}

#define CV_DEF_NORM_FUNC(L, suffix, type, ntype) \
    static int norm##L##_##suffix(const type* src, const uchar* mask, ntype* r, int len, int cn) \
{ return norm##L##_(src, mask, r, len, cn); } \
    static int normDiff##L##_##suffix(const type* src1, const type* src2, \
    const uchar* mask, ntype* r, int len, int cn) \
{ return normDiff##L##_(src1, src2, mask, r, (int)len, cn); }

#define CV_DEF_NORM_ALL(suffix, type, inftype, l1type, l2type) \
    CV_DEF_NORM_FUNC(Inf, suffix, type, inftype) \
    CV_DEF_NORM_FUNC(L1, suffix, type, l1type) \
    CV_DEF_NORM_FUNC(L2, suffix, type, l2type)

CV_DEF_NORM_ALL(8u, uchar, int, int, int)
CV_DEF_NORM_ALL(8s, schar, int, int, int)
CV_DEF_NORM_ALL(16u, ushort, int, int, double)
CV_DEF_NORM_ALL(16s, short, int, int, double)
CV_DEF_NORM_ALL(32s, int, int, double, double)
CV_DEF_NORM_ALL(32f, float, float, double, double)
CV_DEF_NORM_ALL(64f, double, double, double, double)


typedef int (*NormFunc)(const uchar*, const uchar*, uchar*, int, int);
typedef int (*NormDiffFunc)(const uchar*, const uchar*, const uchar*, uchar*, int, int);

static NormFunc getNormFunc(int normType, int depth)
{
    static NormFunc normTab[3][8] =
    {
        {
            (NormFunc)GET_OPTIMIZED(normInf_8u), (NormFunc)GET_OPTIMIZED(normInf_8s), (NormFunc)GET_OPTIMIZED(normInf_16u), (NormFunc)GET_OPTIMIZED(normInf_16s),
            (NormFunc)GET_OPTIMIZED(normInf_32s), (NormFunc)GET_OPTIMIZED(normInf_32f), (NormFunc)normInf_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL1_8u), (NormFunc)GET_OPTIMIZED(normL1_8s), (NormFunc)GET_OPTIMIZED(normL1_16u), (NormFunc)GET_OPTIMIZED(normL1_16s),
            (NormFunc)GET_OPTIMIZED(normL1_32s), (NormFunc)GET_OPTIMIZED(normL1_32f), (NormFunc)normL1_64f, 0
        },
        {
            (NormFunc)GET_OPTIMIZED(normL2_8u), (NormFunc)GET_OPTIMIZED(normL2_8s), (NormFunc)GET_OPTIMIZED(normL2_16u), (NormFunc)GET_OPTIMIZED(normL2_16s),
            (NormFunc)GET_OPTIMIZED(normL2_32s), (NormFunc)GET_OPTIMIZED(normL2_32f), (NormFunc)normL2_64f, 0
        }
    };

    return normTab[normType][depth];
}

static NormDiffFunc getNormDiffFunc(int normType, int depth)
{
    static NormDiffFunc normDiffTab[3][8] =
    {
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffInf_8u), (NormDiffFunc)normDiffInf_8s,
            (NormDiffFunc)normDiffInf_16u, (NormDiffFunc)normDiffInf_16s,
            (NormDiffFunc)normDiffInf_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffInf_32f),
            (NormDiffFunc)normDiffInf_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL1_8u), (NormDiffFunc)normDiffL1_8s,
            (NormDiffFunc)normDiffL1_16u, (NormDiffFunc)normDiffL1_16s,
            (NormDiffFunc)normDiffL1_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL1_32f),
            (NormDiffFunc)normDiffL1_64f, 0
        },
        {
            (NormDiffFunc)GET_OPTIMIZED(normDiffL2_8u), (NormDiffFunc)normDiffL2_8s,
            (NormDiffFunc)normDiffL2_16u, (NormDiffFunc)normDiffL2_16s,
            (NormDiffFunc)normDiffL2_32s, (NormDiffFunc)GET_OPTIMIZED(normDiffL2_32f),
            (NormDiffFunc)normDiffL2_64f, 0
        }
    };

    return normDiffTab[normType][depth];
}

#ifdef HAVE_OPENCL

static bool ocl_norm( InputArray _src, int normType, InputArray _mask, double & result )
{
    const ocl::Device & d = ocl::Device::getDefault();

#ifdef __ANDROID__
    if (d.isNVidia())
        return false;
#endif
    const int cn = _src.channels();
    if (cn > 4)
        return false;
    int type = _src.type(), depth = CV_MAT_DEPTH(type);
    bool doubleSupport = d.doubleFPConfig() > 0,
            haveMask = _mask.kind() != _InputArray::NONE;

    if ( !(normType == NORM_INF || normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR) ||
         (!doubleSupport && depth == CV_64F))
        return false;

    UMat src = _src.getUMat();

    if (normType == NORM_INF)
    {
        if (!ocl_minMaxIdx(_src, NULL, &result, NULL, NULL, _mask,
                           std::max(depth, CV_32S), depth != CV_8U && depth != CV_16U))
            return false;
    }
    else if (normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR)
    {
        Scalar sc;
        bool unstype = depth == CV_8U || depth == CV_16U;

        if ( !ocl_sum(haveMask ? src : src.reshape(1), sc, normType == NORM_L2 || normType == NORM_L2SQR ?
                    OCL_OP_SUM_SQR : (unstype ? OCL_OP_SUM : OCL_OP_SUM_ABS), _mask) )
            return false;

        double s = 0.0;
        for (int i = 0; i < (haveMask ? cn : 1); ++i)
            s += sc[i];

        result = normType == NORM_L1 || normType == NORM_L2SQR ? s : std::sqrt(s);
    }

    return true;
}

#endif

#ifdef HAVE_IPP
static bool ipp_norm(Mat &src, int normType, Mat &mask, double &result)
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 >= 700
    size_t total_size = src.total();
    int rows = src.size[0], cols = rows ? (int)(total_size/rows) : 0;

    if( (src.dims == 2 || (src.isContinuous() && mask.isContinuous()))
        && cols > 0 && (size_t)rows*cols == total_size )
    {
        if( !mask.empty() )
        {
            IppiSize sz = { cols, rows };
            int type = src.type();

            typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC1)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormFuncC1 ippiNorm_C1MR =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_Inf_32f_C1MR :
                0) :
            normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L1_32f_C1MR :
                0) :
            normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormFuncC1)ippiNorm_L2_32f_C1MR :
                0) : 0;
            if( ippiNorm_C1MR )
            {
                Ipp64f norm;
                if( CV_INSTRUMENT_FUN_IPP(ippiNorm_C1MR, src.ptr(), (int)src.step[0], mask.ptr(), (int)mask.step[0], sz, &norm) >= 0 )
                {
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskNormFuncC3)(const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskNormFuncC3 ippiNorm_C3CMR =
                normType == NORM_INF ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_Inf_32f_C3CMR :
                0) :
            normType == NORM_L1 ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L1_32f_C3CMR :
                0) :
            normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormFuncC3)ippiNorm_L2_32f_C3CMR :
                0) : 0;
            if( ippiNorm_C3CMR )
            {
                Ipp64f norm1, norm2, norm3;
                if( CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 1, &norm1) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 2, &norm2) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNorm_C3CMR, src.data, (int)src.step[0], mask.data, (int)mask.step[0], sz, 3, &norm3) >= 0)
                {
                    Ipp64f norm =
                        normType == NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                        normType == NORM_L1 ? norm1 + norm2 + norm3 :
                        normType == NORM_L2 || normType == NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                        0;
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
        }
        else
        {
            IppiSize sz = { cols*src.channels(), rows };
            int type = src.depth();

            typedef IppStatus (CV_STDCALL* ippiNormFuncHint)(const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            typedef IppStatus (CV_STDCALL* ippiNormFuncNoHint)(const void *, int, IppiSize, Ipp64f *);
            ippiNormFuncHint ippiNormHint =
                normType == NORM_L1 ?
                (type == CV_32FC1 ? (ippiNormFuncHint)ippiNorm_L1_32f_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_32FC1 ? (ippiNormFuncHint)ippiNorm_L2_32f_C1R :
                0) : 0;
            ippiNormFuncNoHint ippiNorm =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_8u_C1R :
                type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_16u_C1R :
                type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_16s_C1R :
                type == CV_32FC1 ? (ippiNormFuncNoHint)ippiNorm_Inf_32f_C1R :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_L1_8u_C1R :
                type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_L1_16u_C1R :
                type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_L1_16s_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiNormFuncNoHint)ippiNorm_L2_8u_C1R :
                type == CV_16UC1 ? (ippiNormFuncNoHint)ippiNorm_L2_16u_C1R :
                type == CV_16SC1 ? (ippiNormFuncNoHint)ippiNorm_L2_16s_C1R :
                0) : 0;
            if( ippiNormHint || ippiNorm )
            {
                Ipp64f norm;
                IppStatus ret = ippiNormHint ? CV_INSTRUMENT_FUN_IPP(ippiNormHint, src.ptr(), (int)src.step[0], sz, &norm, ippAlgHintAccurate) :
                                CV_INSTRUMENT_FUN_IPP(ippiNorm, src.ptr(), (int)src.step[0], sz, &norm);
                if( ret >= 0 )
                {
                    result = (normType == NORM_L2SQR) ? norm * norm : norm;
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(src); CV_UNUSED(normType); CV_UNUSED(mask); CV_UNUSED(result);
#endif
    return false;
}
#endif
}

double cv::norm( InputArray _src, int normType, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    normType &= NORM_TYPE_MASK;
    CV_Assert( normType == NORM_INF || normType == NORM_L1 ||
               normType == NORM_L2 || normType == NORM_L2SQR ||
               ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && _src.type() == CV_8U) );

#if defined HAVE_OPENCL || defined HAVE_IPP
    double _result = 0;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src.isUMat()) && _src.dims() <= 2,
                ocl_norm(_src, normType, _mask, _result),
                _result)
#endif

    Mat src = _src.getMat(), mask = _mask.getMat();
    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_norm(src, normType, mask, _result), _result);

    int depth = src.depth(), cn = src.channels();
    if( src.isContinuous() && mask.empty() )
    {
        size_t len = src.total()*cn;
        if( len == (size_t)(int)len )
        {
            if( depth == CV_32F )
            {
                const float* data = src.ptr<float>();

                if( normType == NORM_L2 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL2_32f)(data, 0, &result, (int)len, 1);
                    return std::sqrt(result);
                }
                if( normType == NORM_L2SQR )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL2_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_L1 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normL1_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_INF )
                {
                    float result = 0;
                    GET_OPTIMIZED(normInf_32f)(data, 0, &result, (int)len, 1);
                    return result;
                }
            }
            if( depth == CV_8U )
            {
                const uchar* data = src.ptr<uchar>();

                if( normType == NORM_HAMMING )
                {
                    return hal::normHamming(data, (int)len);
                }

                if( normType == NORM_HAMMING2 )
                {
                    return hal::normHamming(data, (int)len, 2);
                }
            }
        }
    }

    CV_Assert( mask.empty() || mask.type() == CV_8U );

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_and(src, mask, temp);
            return norm(temp, normType);
        }
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        const Mat* arrays[] = {&src, 0};
        uchar* ptrs[1];
        NAryMatIterator it(arrays, ptrs);
        int total = (int)it.size;
        int result = 0;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            result += hal::normHamming(ptrs[0], total, cellSize);
        }

        return result;
    }

    NormFunc func = getNormFunc(normType >> 1, depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &mask, 0};
    uchar* ptrs[2];
    union
    {
        double d;
        int i;
        float f;
    }
    result;
    result.d = 0;
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)it.size, blockSize = total, intSumBlockSize = 0, count = 0;
    bool blockSum = (normType == NORM_L1 && depth <= CV_16S) ||
            ((normType == NORM_L2 || normType == NORM_L2SQR) && depth <= CV_8S);
    int isum = 0;
    int *ibuf = &result.i;
    size_t esz = 0;

    if( blockSum )
    {
        intSumBlockSize = (normType == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15))/cn;
        blockSize = std::min(blockSize, intSumBlockSize);
        ibuf = &isum;
        esz = src.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], ptrs[1], (uchar*)ibuf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                result.d += isum;
                isum = 0;
                count = 0;
            }
            ptrs[0] += bsz*esz;
            if( ptrs[1] )
                ptrs[1] += bsz;
        }
    }

    if( normType == NORM_INF )
    {
        if( depth == CV_64F )
            ;
        else if( depth == CV_32F )
            result.d = result.f;
        else
            result.d = result.i;
    }
    else if( normType == NORM_L2 )
        result.d = std::sqrt(result.d);

    return result.d;
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_norm( InputArray _src1, InputArray _src2, int normType, InputArray _mask, double & result )
{
#ifdef __ANDROID__
    if (ocl::Device::getDefault().isNVidia())
        return false;
#endif

    Scalar sc1, sc2;
    int cn = _src1.channels();
    if (cn > 4)
        return false;
    int type = _src1.type(), depth = CV_MAT_DEPTH(type);
    bool relative = (normType & NORM_RELATIVE) != 0;
    normType &= ~NORM_RELATIVE;
    bool normsum = normType == NORM_L1 || normType == NORM_L2 || normType == NORM_L2SQR;

#ifdef __APPLE__
    if(normType == NORM_L1 && type == CV_16UC3 && !_mask.empty())
        return false;
#endif

    if (normsum)
    {
        if (!ocl_sum(_src1, sc1, normType == NORM_L2 || normType == NORM_L2SQR ?
                     OCL_OP_SUM_SQR : OCL_OP_SUM, _mask, _src2, relative, sc2))
            return false;
    }
    else
    {
        if (!ocl_minMaxIdx(_src1, NULL, &sc1[0], NULL, NULL, _mask, std::max(CV_32S, depth),
                           false, _src2, relative ? &sc2[0] : NULL))
            return false;
        cn = 1;
    }

    double s2 = 0;
    for (int i = 0; i < cn; ++i)
    {
        result += sc1[i];
        if (relative)
            s2 += sc2[i];
    }

    if (normType == NORM_L2)
    {
        result = std::sqrt(result);
        if (relative)
            s2 = std::sqrt(s2);
    }

    if (relative)
        result /= (s2 + DBL_EPSILON);

    return true;
}

}

#endif

#ifdef HAVE_IPP
namespace cv
{
static bool ipp_norm(InputArray _src1, InputArray _src2, int normType, InputArray _mask, double &result)
{
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 >= 700
    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();

    if( normType & CV_RELATIVE )
    {
        normType &= NORM_TYPE_MASK;

        size_t total_size = src1.total();
        int rows = src1.size[0], cols = rows ? (int)(total_size/rows) : 0;
        if( (src1.dims == 2 || (src1.isContinuous() && src2.isContinuous() && mask.isContinuous()))
            && cols > 0 && (size_t)rows*cols == total_size )
        {
            if( !mask.empty() )
            {
                IppiSize sz = { cols, rows };
                int type = src1.type();

                typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
                ippiMaskNormDiffFuncC1 ippiNormRel_C1MR =
                    normType == NORM_INF ?
                    (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_8u_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_Inf_32f_C1MR :
                    0) :
                    normType == NORM_L1 ?
                    (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_8u_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L1_32f_C1MR :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_8u_C1MR :
                    type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_16u_C1MR :
                    type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormRel_L2_32f_C1MR :
                    0) : 0;
                if( ippiNormRel_C1MR )
                {
                    Ipp64f norm;
                    if( CV_INSTRUMENT_FUN_IPP(ippiNormRel_C1MR, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], mask.ptr(), (int)mask.step[0], sz, &norm) >= 0 )
                    {
                        result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                        return true;
                    }
                }
            }
            else
            {
                IppiSize sz = { cols*src1.channels(), rows };
                int type = src1.depth();

                typedef IppStatus (CV_STDCALL* ippiNormRelFuncHint)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
                typedef IppStatus (CV_STDCALL* ippiNormRelFuncNoHint)(const void *, int, const void *, int, IppiSize, Ipp64f *);
                ippiNormRelFuncHint ippiNormRelHint =
                    normType == NORM_L1 ?
                    (type == CV_32F ? (ippiNormRelFuncHint)ippiNormRel_L1_32f_C1R :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_32F ? (ippiNormRelFuncHint)ippiNormRel_L2_32f_C1R :
                    0) : 0;
                ippiNormRelFuncNoHint ippiNormRel =
                    normType == NORM_INF ?
                    (type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_8u_C1R :
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_16s_C1R :
                    type == CV_32F ? (ippiNormRelFuncNoHint)ippiNormRel_Inf_32f_C1R :
                    0) :
                    normType == NORM_L1 ?
                    (type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_L1_8u_C1R :
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_L1_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_L1_16s_C1R :
                    0) :
                    normType == NORM_L2 || normType == NORM_L2SQR ?
                    (type == CV_8U ? (ippiNormRelFuncNoHint)ippiNormRel_L2_8u_C1R :
                    type == CV_16U ? (ippiNormRelFuncNoHint)ippiNormRel_L2_16u_C1R :
                    type == CV_16S ? (ippiNormRelFuncNoHint)ippiNormRel_L2_16s_C1R :
                    0) : 0;
                if( ippiNormRelHint || ippiNormRel )
                {
                    Ipp64f norm;
                    IppStatus ret = ippiNormRelHint ? CV_INSTRUMENT_FUN_IPP(ippiNormRelHint, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm, ippAlgHintAccurate) :
                                    CV_INSTRUMENT_FUN_IPP(ippiNormRel, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm);
                    if( ret >= 0 )
                    {
                        result = (normType == NORM_L2SQR) ? norm * norm : norm;
                        return true;
                    }
                }
            }
        }
        return false;
    }

    normType &= NORM_TYPE_MASK;

    size_t total_size = src1.total();
    int rows = src1.size[0], cols = rows ? (int)(total_size/rows) : 0;
    if( (src1.dims == 2 || (src1.isContinuous() && src2.isContinuous() && mask.isContinuous()))
        && cols > 0 && (size_t)rows*cols == total_size )
    {
        if( !mask.empty() )
        {
            IppiSize sz = { cols, rows };
            int type = src1.type();

            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC1)(const void *, int, const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiMaskNormDiffFuncC1 ippiNormDiff_C1MR =
                normType == NORM_INF ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_Inf_32f_C1MR :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L1_32f_C1MR :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_8u_C1MR :
                type == CV_16UC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_16u_C1MR :
                type == CV_32FC1 ? (ippiMaskNormDiffFuncC1)ippiNormDiff_L2_32f_C1MR :
                0) : 0;
            if( ippiNormDiff_C1MR )
            {
                Ipp64f norm;
                if( CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C1MR, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], mask.ptr(), (int)mask.step[0], sz, &norm) >= 0 )
                {
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
            typedef IppStatus (CV_STDCALL* ippiMaskNormDiffFuncC3)(const void *, int, const void *, int, const void *, int, IppiSize, int, Ipp64f *);
            ippiMaskNormDiffFuncC3 ippiNormDiff_C3CMR =
                normType == NORM_INF ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_Inf_32f_C3CMR :
                0) :
                normType == NORM_L1 ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L1_32f_C3CMR :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_8u_C3CMR :
                type == CV_16UC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_16u_C3CMR :
                type == CV_32FC3 ? (ippiMaskNormDiffFuncC3)ippiNormDiff_L2_32f_C3CMR :
                0) : 0;
            if( ippiNormDiff_C3CMR )
            {
                Ipp64f norm1, norm2, norm3;
                if( CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 1, &norm1) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 2, &norm2) >= 0 &&
                    CV_INSTRUMENT_FUN_IPP(ippiNormDiff_C3CMR, src1.data, (int)src1.step[0], src2.data, (int)src2.step[0], mask.data, (int)mask.step[0], sz, 3, &norm3) >= 0)
                {
                    Ipp64f norm =
                        normType == NORM_INF ? std::max(std::max(norm1, norm2), norm3) :
                        normType == NORM_L1 ? norm1 + norm2 + norm3 :
                        normType == NORM_L2 || normType == NORM_L2SQR ? std::sqrt(norm1 * norm1 + norm2 * norm2 + norm3 * norm3) :
                        0;
                    result = (normType == NORM_L2SQR ? (double)(norm * norm) : (double)norm);
                    return true;
                }
            }
        }
        else
        {
            IppiSize sz = { cols*src1.channels(), rows };
            int type = src1.depth();

            typedef IppStatus (CV_STDCALL* ippiNormDiffFuncHint)(const void *, int, const void *, int, IppiSize, Ipp64f *, IppHintAlgorithm hint);
            typedef IppStatus (CV_STDCALL* ippiNormDiffFuncNoHint)(const void *, int, const void *, int, IppiSize, Ipp64f *);
            ippiNormDiffFuncHint ippiNormDiffHint =
                normType == NORM_L1 ?
                (type == CV_32F ? (ippiNormDiffFuncHint)ippiNormDiff_L1_32f_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_32F ? (ippiNormDiffFuncHint)ippiNormDiff_L2_32f_C1R :
                0) : 0;
            ippiNormDiffFuncNoHint ippiNormDiff =
                normType == NORM_INF ?
                (type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_8u_C1R :
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_16s_C1R :
                type == CV_32F ? (ippiNormDiffFuncNoHint)ippiNormDiff_Inf_32f_C1R :
                0) :
                normType == NORM_L1 ?
                (type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_8u_C1R :
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_L1_16s_C1R :
                0) :
                normType == NORM_L2 || normType == NORM_L2SQR ?
                (type == CV_8U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_8u_C1R :
                type == CV_16U ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_16u_C1R :
                type == CV_16S ? (ippiNormDiffFuncNoHint)ippiNormDiff_L2_16s_C1R :
                0) : 0;
            if( ippiNormDiffHint || ippiNormDiff )
            {
                Ipp64f norm;
                IppStatus ret = ippiNormDiffHint ? CV_INSTRUMENT_FUN_IPP(ippiNormDiffHint, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm, ippAlgHintAccurate) :
                                CV_INSTRUMENT_FUN_IPP(ippiNormDiff, src1.ptr(), (int)src1.step[0], src2.ptr(), (int)src2.step[0], sz, &norm);
                if( ret >= 0 )
                {
                    result = (normType == NORM_L2SQR) ? norm * norm : norm;
                    return true;
                }
            }
        }
    }
#else
    CV_UNUSED(_src1); CV_UNUSED(_src2); CV_UNUSED(normType); CV_UNUSED(_mask); CV_UNUSED(result);
#endif
    return false;
}
}
#endif


double cv::norm( InputArray _src1, InputArray _src2, int normType, InputArray _mask )
{
    CV_INSTRUMENT_REGION()

    CV_Assert( _src1.sameSize(_src2) && _src1.type() == _src2.type() );

#if defined HAVE_OPENCL || defined HAVE_IPP
    double _result = 0;
#endif

#ifdef HAVE_OPENCL
    CV_OCL_RUN_(OCL_PERFORMANCE_CHECK(_src1.isUMat()),
                ocl_norm(_src1, _src2, normType, _mask, _result),
                _result)
#endif

    CV_IPP_RUN(IPP_VERSION_X100 >= 700, ipp_norm(_src1, _src2, normType, _mask, _result), _result);

    if( normType & CV_RELATIVE )
    {
        return norm(_src1, _src2, normType & ~CV_RELATIVE, _mask)/(norm(_src2, normType, _mask) + DBL_EPSILON);
    }

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    int depth = src1.depth(), cn = src1.channels();

    normType &= 7;
    CV_Assert( normType == NORM_INF || normType == NORM_L1 ||
               normType == NORM_L2 || normType == NORM_L2SQR ||
              ((normType == NORM_HAMMING || normType == NORM_HAMMING2) && src1.type() == CV_8U) );

    if( src1.isContinuous() && src2.isContinuous() && mask.empty() )
    {
        size_t len = src1.total()*src1.channels();
        if( len == (size_t)(int)len )
        {
            if( src1.depth() == CV_32F )
            {
                const float* data1 = src1.ptr<float>();
                const float* data2 = src2.ptr<float>();

                if( normType == NORM_L2 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL2_32f)(data1, data2, 0, &result, (int)len, 1);
                    return std::sqrt(result);
                }
                if( normType == NORM_L2SQR )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL2_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_L1 )
                {
                    double result = 0;
                    GET_OPTIMIZED(normDiffL1_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
                if( normType == NORM_INF )
                {
                    float result = 0;
                    GET_OPTIMIZED(normDiffInf_32f)(data1, data2, 0, &result, (int)len, 1);
                    return result;
                }
            }
        }
    }

    CV_Assert( mask.empty() || mask.type() == CV_8U );

    if( normType == NORM_HAMMING || normType == NORM_HAMMING2 )
    {
        if( !mask.empty() )
        {
            Mat temp;
            bitwise_xor(src1, src2, temp);
            bitwise_and(temp, mask, temp);
            return norm(temp, normType);
        }
        int cellSize = normType == NORM_HAMMING ? 1 : 2;

        const Mat* arrays[] = {&src1, &src2, 0};
        uchar* ptrs[2];
        NAryMatIterator it(arrays, ptrs);
        int total = (int)it.size;
        int result = 0;

        for( size_t i = 0; i < it.nplanes; i++, ++it )
        {
            result += hal::normHamming(ptrs[0], ptrs[1], total, cellSize);
        }

        return result;
    }

    NormDiffFunc func = getNormDiffFunc(normType >> 1, depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src1, &src2, &mask, 0};
    uchar* ptrs[3];
    union
    {
        double d;
        float f;
        int i;
        unsigned u;
    }
    result;
    result.d = 0;
    NAryMatIterator it(arrays, ptrs);
    int j, total = (int)it.size, blockSize = total, intSumBlockSize = 0, count = 0;
    bool blockSum = (normType == NORM_L1 && depth <= CV_16S) ||
            ((normType == NORM_L2 || normType == NORM_L2SQR) && depth <= CV_8S);
    unsigned isum = 0;
    unsigned *ibuf = &result.u;
    size_t esz = 0;

    if( blockSum )
    {
        intSumBlockSize = normType == NORM_L1 && depth <= CV_8S ? (1 << 23) : (1 << 15);
        blockSize = std::min(blockSize, intSumBlockSize);
        ibuf = &isum;
        esz = src1.elemSize();
    }

    for( size_t i = 0; i < it.nplanes; i++, ++it )
    {
        for( j = 0; j < total; j += blockSize )
        {
            int bsz = std::min(total - j, blockSize);
            func( ptrs[0], ptrs[1], ptrs[2], (uchar*)ibuf, bsz, cn );
            count += bsz;
            if( blockSum && (count + blockSize >= intSumBlockSize || (i+1 >= it.nplanes && j+bsz >= total)) )
            {
                result.d += isum;
                isum = 0;
                count = 0;
            }
            ptrs[0] += bsz*esz;
            ptrs[1] += bsz*esz;
            if( ptrs[2] )
                ptrs[2] += bsz;
        }
    }

    if( normType == NORM_INF )
    {
        if( depth == CV_64F )
            ;
        else if( depth == CV_32F )
            result.d = result.f;
        else
            result.d = result.u;
    }
    else if( normType == NORM_L2 )
        result.d = std::sqrt(result.d);

    return result.d;
}


///////////////////////////////////// batch distance ///////////////////////////////////////

namespace cv
{

template<typename _Tp, typename _Rt>
void batchDistL1_(const _Tp* src1, const _Tp* src2, size_t step2,
                  int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normL1<_Tp, _Rt>(src1, src2 + step2*i, len);
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normL1<_Tp, _Rt>(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
void batchDistL2Sqr_(const _Tp* src1, const _Tp* src2, size_t step2,
                     int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len);
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len) : val0;
    }
}

template<typename _Tp, typename _Rt>
void batchDistL2_(const _Tp* src1, const _Tp* src2, size_t step2,
                  int nvecs, int len, _Rt* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = std::sqrt(normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len));
    }
    else
    {
        _Rt val0 = std::numeric_limits<_Rt>::max();
        for( int i = 0; i < nvecs; i++ )
            dist[i] = mask[i] ? std::sqrt(normL2Sqr<_Tp, _Rt>(src1, src2 + step2*i, len)) : val0;
    }
}

static void batchDistHamming(const uchar* src1, const uchar* src2, size_t step2,
                             int nvecs, int len, int* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
             dist[i] = hal::normHamming(src1, src2 + step2*i, len);
    }
    else
    {
        int val0 = INT_MAX;
        for( int i = 0; i < nvecs; i++ )
        {
            if (mask[i])
                dist[i] = hal::normHamming(src1, src2 + step2*i, len);
            else
                dist[i] = val0;
        }
    }
}

static void batchDistHamming2(const uchar* src1, const uchar* src2, size_t step2,
                              int nvecs, int len, int* dist, const uchar* mask)
{
    step2 /= sizeof(src2[0]);
    if( !mask )
    {
        for( int i = 0; i < nvecs; i++ )
            dist[i] = hal::normHamming(src1, src2 + step2*i, len, 2);
    }
    else
    {
        int val0 = INT_MAX;
        for( int i = 0; i < nvecs; i++ )
        {
            if (mask[i])
                dist[i] = hal::normHamming(src1, src2 + step2*i, len, 2);
            else
                dist[i] = val0;
        }
    }
}

static void batchDistL1_8u32s(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, int* dist, const uchar* mask)
{
    batchDistL1_<uchar, int>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL1_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL1_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_8u32s(const uchar* src1, const uchar* src2, size_t step2,
                                  int nvecs, int len, int* dist, const uchar* mask)
{
    batchDistL2Sqr_<uchar, int>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                                  int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2Sqr_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2_8u32f(const uchar* src1, const uchar* src2, size_t step2,
                               int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2_<uchar, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL1_32f(const float* src1, const float* src2, size_t step2,
                             int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL1_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2Sqr_32f(const float* src1, const float* src2, size_t step2,
                                int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2Sqr_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

static void batchDistL2_32f(const float* src1, const float* src2, size_t step2,
                             int nvecs, int len, float* dist, const uchar* mask)
{
    batchDistL2_<float, float>(src1, src2, step2, nvecs, len, dist, mask);
}

typedef void (*BatchDistFunc)(const uchar* src1, const uchar* src2, size_t step2,
                              int nvecs, int len, uchar* dist, const uchar* mask);


struct BatchDistInvoker : public ParallelLoopBody
{
    BatchDistInvoker( const Mat& _src1, const Mat& _src2,
                      Mat& _dist, Mat& _nidx, int _K,
                      const Mat& _mask, int _update,
                      BatchDistFunc _func)
    {
        src1 = &_src1;
        src2 = &_src2;
        dist = &_dist;
        nidx = &_nidx;
        K = _K;
        mask = &_mask;
        update = _update;
        func = _func;
    }

    void operator()(const Range& range) const
    {
        AutoBuffer<int> buf(src2->rows);
        int* bufptr = buf;

        for( int i = range.start; i < range.end; i++ )
        {
            func(src1->ptr(i), src2->ptr(), src2->step, src2->rows, src2->cols,
                 K > 0 ? (uchar*)bufptr : dist->ptr(i), mask->data ? mask->ptr(i) : 0);

            if( K > 0 )
            {
                int* nidxptr = nidx->ptr<int>(i);
                // since positive float's can be compared just like int's,
                // we handle both CV_32S and CV_32F cases with a single branch
                int* distptr = (int*)dist->ptr(i);

                int j, k;

                for( j = 0; j < src2->rows; j++ )
                {
                    int d = bufptr[j];
                    if( d < distptr[K-1] )
                    {
                        for( k = K-2; k >= 0 && distptr[k] > d; k-- )
                        {
                            nidxptr[k+1] = nidxptr[k];
                            distptr[k+1] = distptr[k];
                        }
                        nidxptr[k+1] = j + update;
                        distptr[k+1] = d;
                    }
                }
            }
        }
    }

    const Mat *src1;
    const Mat *src2;
    Mat *dist;
    Mat *nidx;
    const Mat *mask;
    int K;
    int update;
    BatchDistFunc func;
};

}

void cv::batchDistance( InputArray _src1, InputArray _src2,
                        OutputArray _dist, int dtype, OutputArray _nidx,
                        int normType, int K, InputArray _mask,
                        int update, bool crosscheck )
{
    CV_INSTRUMENT_REGION()

    Mat src1 = _src1.getMat(), src2 = _src2.getMat(), mask = _mask.getMat();
    int type = src1.type();
    CV_Assert( type == src2.type() && src1.cols == src2.cols &&
               (type == CV_32F || type == CV_8U));
    CV_Assert( _nidx.needed() == (K > 0) );

    if( dtype == -1 )
    {
        dtype = normType == NORM_HAMMING || normType == NORM_HAMMING2 ? CV_32S : CV_32F;
    }
    CV_Assert( (type == CV_8U && dtype == CV_32S) || dtype == CV_32F);

    K = std::min(K, src2.rows);

    _dist.create(src1.rows, (K > 0 ? K : src2.rows), dtype);
    Mat dist = _dist.getMat(), nidx;
    if( _nidx.needed() )
    {
        _nidx.create(dist.size(), CV_32S);
        nidx = _nidx.getMat();
    }

    if( update == 0 && K > 0 )
    {
        dist = Scalar::all(dtype == CV_32S ? (double)INT_MAX : (double)FLT_MAX);
        nidx = Scalar::all(-1);
    }

    if( crosscheck )
    {
        CV_Assert( K == 1 && update == 0 && mask.empty() );
        Mat tdist, tidx;
        batchDistance(src2, src1, tdist, dtype, tidx, normType, K, mask, 0, false);

        // if an idx-th element from src1 appeared to be the nearest to i-th element of src2,
        // we update the minimum mutual distance between idx-th element of src1 and the whole src2 set.
        // As a result, if nidx[idx] = i*, it means that idx-th element of src1 is the nearest
        // to i*-th element of src2 and i*-th element of src2 is the closest to idx-th element of src1.
        // If nidx[idx] = -1, it means that there is no such ideal couple for it in src2.
        // This O(N) procedure is called cross-check and it helps to eliminate some false matches.
        if( dtype == CV_32S )
        {
            for( int i = 0; i < tdist.rows; i++ )
            {
                int idx = tidx.at<int>(i);
                int d = tdist.at<int>(i), d0 = dist.at<int>(idx);
                if( d < d0 )
                {
                    dist.at<int>(idx) = d;
                    nidx.at<int>(idx) = i + update;
                }
            }
        }
        else
        {
            for( int i = 0; i < tdist.rows; i++ )
            {
                int idx = tidx.at<int>(i);
                float d = tdist.at<float>(i), d0 = dist.at<float>(idx);
                if( d < d0 )
                {
                    dist.at<float>(idx) = d;
                    nidx.at<int>(idx) = i + update;
                }
            }
        }
        return;
    }

    BatchDistFunc func = 0;
    if( type == CV_8U )
    {
        if( normType == NORM_L1 && dtype == CV_32S )
            func = (BatchDistFunc)batchDistL1_8u32s;
        else if( normType == NORM_L1 && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL1_8u32f;
        else if( normType == NORM_L2SQR && dtype == CV_32S )
            func = (BatchDistFunc)batchDistL2Sqr_8u32s;
        else if( normType == NORM_L2SQR && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL2Sqr_8u32f;
        else if( normType == NORM_L2 && dtype == CV_32F )
            func = (BatchDistFunc)batchDistL2_8u32f;
        else if( normType == NORM_HAMMING && dtype == CV_32S )
            func = (BatchDistFunc)batchDistHamming;
        else if( normType == NORM_HAMMING2 && dtype == CV_32S )
            func = (BatchDistFunc)batchDistHamming2;
    }
    else if( type == CV_32F && dtype == CV_32F )
    {
        if( normType == NORM_L1 )
            func = (BatchDistFunc)batchDistL1_32f;
        else if( normType == NORM_L2SQR )
            func = (BatchDistFunc)batchDistL2Sqr_32f;
        else if( normType == NORM_L2 )
            func = (BatchDistFunc)batchDistL2_32f;
    }

    if( func == 0 )
        CV_Error_(CV_StsUnsupportedFormat,
                  ("The combination of type=%d, dtype=%d and normType=%d is not supported",
                   type, dtype, normType));

    parallel_for_(Range(0, src1.rows),
                  BatchDistInvoker(src1, src2, dist, nidx, K, mask, update, func));
}


void cv::findNonZero( InputArray _src, OutputArray _idx )
{
    CV_INSTRUMENT_REGION()

    Mat src = _src.getMat();
    CV_Assert( src.type() == CV_8UC1 );
    int n = countNonZero(src);
    if( n == 0 )
    {
        _idx.release();
        return;
    }
    if( _idx.kind() == _InputArray::MAT && !_idx.getMatRef().isContinuous() )
        _idx.release();
    _idx.create(n, 1, CV_32SC2);
    Mat idx = _idx.getMat();
    CV_Assert(idx.isContinuous());
    Point* idx_ptr = idx.ptr<Point>();

    for( int i = 0; i < src.rows; i++ )
    {
        const uchar* bin_ptr = src.ptr(i);
        for( int j = 0; j < src.cols; j++ )
            if( bin_ptr[j] )
                *idx_ptr++ = Point(j, i);
    }
}

double cv::PSNR(InputArray _src1, InputArray _src2)
{
    CV_INSTRUMENT_REGION()

    //Input arrays must have depth CV_8U
    CV_Assert( _src1.depth() == CV_8U && _src2.depth() == CV_8U );

    double diff = std::sqrt(norm(_src1, _src2, NORM_L2SQR)/(_src1.total()*_src1.channels()));
    return 20*log10(255./(diff+DBL_EPSILON));
}


CV_IMPL CvScalar cvSum( const CvArr* srcarr )
{
    cv::Scalar sum = cv::sum(cv::cvarrToMat(srcarr, false, true, 1));
    if( CV_IS_IMAGE(srcarr) )
    {
        int coi = cvGetImageCOI((IplImage*)srcarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            sum = cv::Scalar(sum[coi-1]);
        }
    }
    return sum;
}

CV_IMPL int cvCountNonZero( const CvArr* imgarr )
{
    cv::Mat img = cv::cvarrToMat(imgarr, false, true, 1);
    if( img.channels() > 1 )
        cv::extractImageCOI(imgarr, img);
    return countNonZero(img);
}


CV_IMPL  CvScalar
cvAvg( const void* imgarr, const void* maskarr )
{
    cv::Mat img = cv::cvarrToMat(imgarr, false, true, 1);
    cv::Scalar mean = !maskarr ? cv::mean(img) : cv::mean(img, cv::cvarrToMat(maskarr));
    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = cv::Scalar(mean[coi-1]);
        }
    }
    return mean;
}


CV_IMPL  void
cvAvgSdv( const CvArr* imgarr, CvScalar* _mean, CvScalar* _sdv, const void* maskarr )
{
    cv::Scalar mean, sdv;

    cv::Mat mask;
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);

    cv::meanStdDev(cv::cvarrToMat(imgarr, false, true, 1), mean, sdv, mask );

    if( CV_IS_IMAGE(imgarr) )
    {
        int coi = cvGetImageCOI((IplImage*)imgarr);
        if( coi )
        {
            CV_Assert( 0 < coi && coi <= 4 );
            mean = cv::Scalar(mean[coi-1]);
            sdv = cv::Scalar(sdv[coi-1]);
        }
    }

    if( _mean )
        *(cv::Scalar*)_mean = mean;
    if( _sdv )
        *(cv::Scalar*)_sdv = sdv;
}


CV_IMPL void
cvMinMaxLoc( const void* imgarr, double* _minVal, double* _maxVal,
             CvPoint* _minLoc, CvPoint* _maxLoc, const void* maskarr )
{
    cv::Mat mask, img = cv::cvarrToMat(imgarr, false, true, 1);
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);
    if( img.channels() > 1 )
        cv::extractImageCOI(imgarr, img);

    cv::minMaxLoc( img, _minVal, _maxVal,
                   (cv::Point*)_minLoc, (cv::Point*)_maxLoc, mask );
}


CV_IMPL  double
cvNorm( const void* imgA, const void* imgB, int normType, const void* maskarr )
{
    cv::Mat a, mask;
    if( !imgA )
    {
        imgA = imgB;
        imgB = 0;
    }

    a = cv::cvarrToMat(imgA, false, true, 1);
    if( maskarr )
        mask = cv::cvarrToMat(maskarr);

    if( a.channels() > 1 && CV_IS_IMAGE(imgA) && cvGetImageCOI((const IplImage*)imgA) > 0 )
        cv::extractImageCOI(imgA, a);

    if( !imgB )
        return !maskarr ? cv::norm(a, normType) : cv::norm(a, normType, mask);

    cv::Mat b = cv::cvarrToMat(imgB, false, true, 1);
    if( b.channels() > 1 && CV_IS_IMAGE(imgB) && cvGetImageCOI((const IplImage*)imgB) > 0 )
        cv::extractImageCOI(imgB, b);

    return !maskarr ? cv::norm(a, b, normType) : cv::norm(a, b, normType, mask);
}

namespace cv { namespace hal {

extern const uchar popCountTable[256] =
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
#if CV_AVX2
    float CV_DECL_ALIGNED(32) buf[8];
    __m256 d0 = _mm256_setzero_ps();

    for( ; j <= n - 8; j += 8 )
    {
        __m256 t0 = _mm256_sub_ps(_mm256_loadu_ps(a + j), _mm256_loadu_ps(b + j));
#if CV_FMA3
        d0 = _mm256_fmadd_ps(t0, t0, d0);
#else
        d0 = _mm256_add_ps(d0, _mm256_mul_ps(t0, t0));
#endif
    }
    _mm256_store_ps(buf, d0);
    d = buf[0] + buf[1] + buf[2] + buf[3] + buf[4] + buf[5] + buf[6] + buf[7];
#elif CV_SSE
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
