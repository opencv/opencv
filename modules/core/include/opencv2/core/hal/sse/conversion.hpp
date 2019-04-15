// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Round ////

inline v_int32x4 v_round(const v_float32x4& a)
{ return _mm_cvtps_epi32(a); }

inline v_int32x4 v_round(const v_float64x2& a)
{ return _mm_cvtpd_epi32(a); }

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{ return _mm_unpacklo_epi64(_mm_cvtpd_epi32(a), _mm_cvtpd_epi32(b)); }

//// Ceil ////

inline v_int32x4 v_ceil(const v_float32x4& a)
{
#if CV_SSE4_1
    return _mm_cvttps_epi32(_mm_ceil_ps(a));
#else
    __m128i a1 = _mm_cvtps_epi32(a);
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(a, _mm_cvtepi32_ps(a1)));
    return _mm_sub_epi32(a1, mask);
#endif
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
#if CV_SSE4_1
    return _mm_cvttpd_epi32(_mm_ceil_pd(a));
#else
    __m128i a1 = _mm_cvtpd_epi32(a);
    __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(a, _mm_cvtepi32_pd(a1)));
    mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
    return _mm_sub_epi32(a1, mask);
#endif
}

//// Floor ////

inline v_int32x4 v_floor(const v_float32x4& a)
{
#if CV_SSE4_1
    return _mm_cvttps_epi32(_mm_floor_ps(a));
#else
    __m128i a1 = _mm_cvtps_epi32(a);
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(_mm_cvtepi32_ps(a1), a));
    return _mm_add_epi32(a1, mask);
#endif
}

inline v_int32x4 v_floor(const v_float64x2& a)
{
#if CV_SSE4_1
    return _mm_cvttpd_epi32(_mm_floor_pd(a));
#else
    __m128i a1 = _mm_cvtpd_epi32(a);
    __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(_mm_cvtepi32_pd(a1), a));
    mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
    return _mm_add_epi32(a1, mask);
#endif
}

//// Truncate ////

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return _mm_cvttps_epi32(a); }

inline v_int32x4 v_trunc(const v_float64x2& a)
{ return _mm_cvttpd_epi32(a); }

//// To Signle Precision ////

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{ return _mm_cvtepi32_ps(a); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{ return _mm_cvtpd_ps(a); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{ return _mm_movelh_ps(_mm_cvtpd_ps(a), _mm_cvtpd_ps(b)); }

//// To Double Precision ////

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{ return _mm_cvtepi32_pd(a); }

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{ return _mm_cvtepi32_pd(_mm_srli_si128(a, 8)); }

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{ return _mm_cvtps_pd(a); }

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{ return _mm_cvtps_pd(_mm_movehl_ps(a, a)); }