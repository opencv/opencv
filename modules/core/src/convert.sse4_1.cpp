// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#include "precomp.hpp"
#include "convert.hpp"

namespace cv
{
namespace opt_SSE4_1
{

int cvtScale_SIMD_u8u16f32_SSE41(const uchar * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_unpacklo_epi8(_mm_loadl_epi64((__m128i const *)(src + x)), v_zero);
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_s8u16f32_SSE41(const schar * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, _mm_loadl_epi64((__m128i const *)(src + x))), 8);
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_u16u16f32_SSE41(const ushort * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src, v_zero));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src, v_zero));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_s16u16f32_SSE41(const short * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128i v_zero = _mm_setzero_si128();
    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
        __m128 v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src), 16));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        v_src_f = _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src), 16));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src_f, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_s32u16f32_SSE41(const int * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128i v_src = _mm_loadu_si128((__m128i const *)(src + x));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

        v_src = _mm_loadu_si128((__m128i const *)(src + x + 4));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(_mm_cvtepi32_ps(v_src), v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_f32u16f32_SSE41(const float * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128 v_src = _mm_loadu_ps(src + x);
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        v_src = _mm_loadu_ps(src + x + 4);
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int cvtScale_SIMD_f64u16f32_SSE41(const double * src, ushort * dst, int width, float scale, float shift)
{
    int x = 0;

    __m128 v_scale = _mm_set1_ps(scale), v_shift = _mm_set1_ps(shift);

    for ( ; x <= width - 8; x += 8)
    {
        __m128 v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x)),
                                        _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2)));
        __m128 v_dst_0 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        v_src = _mm_movelh_ps(_mm_cvtpd_ps(_mm_loadu_pd(src + x + 4)),
                                _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6)));
        __m128 v_dst_1 = _mm_add_ps(_mm_mul_ps(v_src, v_scale), v_shift);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_dst_0),
                                            _mm_cvtps_epi32(v_dst_1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

int Cvt_SIMD_f64u16_SSE41(const double * src, ushort * dst, int width)
{
    int x = 0;

    for ( ; x <= width - 8; x += 8)
    {
        __m128 v_src0 = _mm_cvtpd_ps(_mm_loadu_pd(src + x));
        __m128 v_src1 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 2));
        __m128 v_src2 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 4));
        __m128 v_src3 = _mm_cvtpd_ps(_mm_loadu_pd(src + x + 6));

        v_src0 = _mm_movelh_ps(v_src0, v_src1);
        v_src1 = _mm_movelh_ps(v_src2, v_src3);

        __m128i v_dst = _mm_packus_epi32(_mm_cvtps_epi32(v_src0),
                                            _mm_cvtps_epi32(v_src1));
        _mm_storeu_si128((__m128i *)(dst + x), v_dst);
    }

    return x;
}

}
} // cv::

/* End of file. */
