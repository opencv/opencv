// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Min/Max ////

OPENCV_HAL_IMPL_SSE_BIN_FUN(v_min, v_uint8x16,  _mm_min_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_max, v_uint8x16,  _mm_max_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_min, v_int16x8,   _mm_min_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_max, v_int16x8,   _mm_max_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_min, v_float32x4, _mm_min_ps)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_max, v_float32x4, _mm_max_ps)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_min, v_float64x2, _mm_min_pd)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_max, v_float64x2, _mm_max_pd)

inline v_int8x16 v_min(const v_int8x16& a, const v_int8x16& b)
{
#if CV_SSE4_1
    return _mm_min_epi8(a, b);
#else
    const __m128i sbit = _mm_set1_epi8((char)-128);
    return _mm_xor_si128(sbit, _mm_min_epu8(
        _mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit))
    );
#endif
}
inline v_int8x16 v_max(const v_int8x16& a, const v_int8x16& b)
{
#if CV_SSE4_1
    return _mm_max_epi8(a, b);
#else
    const __m128i sbit = _mm_set1_epi8((char)-128);
    return _mm_xor_si128(sbit, _mm_max_epu8(
        _mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit))
    );
#endif
}
inline v_uint16x8 v_min(const v_uint16x8& a, const v_uint16x8& b)
{
#if CV_SSE4_1
    return _mm_min_epu16(a, b);
#else
    return a - (a - b);
#endif
}
inline v_uint16x8 v_max(const v_uint16x8& a, const v_uint16x8& b)
{
#if CV_SSE4_1
    return _mm_max_epu16(a, b);
#else
    return (a - b) + b;
#endif
}
inline v_uint32x4 v_min(const v_uint32x4& a, const v_uint32x4& b)
{
#if CV_SSE4_1
    return _mm_min_epu32(a, b);
#else
    return v_select(a > b, b, a);
#endif
}
inline v_int32x4 v_min(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    return _mm_min_epi32(a, b);
#else
    return v_select(a > b, b, a);
#endif
}
inline v_uint32x4 v_max(const v_uint32x4& a, const v_uint32x4& b)
{
#if CV_SSE4_1
    return _mm_max_epu32(a, b);
#else
    return v_select(a > b, a, b);
#endif
}
inline v_int32x4 v_max(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    return _mm_max_epi32(a, b);
#else
    return v_select(a > b, a, b);
#endif
}

//// Sqrt ////

inline v_float32x4 v_sqrt(const v_float32x4& a)
{ return _mm_sqrt_ps(a); }
inline v_float64x2 v_sqrt(const v_float64x2& a)
{ return _mm_sqrt_pd(a); }

inline v_float32x4 v_invsqrt(const v_float32x4& a)
{
    const __m128 _0_5 = _mm_set1_ps(0.5f), _1_5 = _mm_set1_ps(1.5f);
    __m128 h = _mm_mul_ps(a, _0_5);
    __m128 t = _mm_rsqrt_ps(a);
    t = _mm_mul_ps(t, _mm_sub_ps(_1_5, _mm_mul_ps(_mm_mul_ps(t, t), h)));
    return t;
}
inline v_float64x2 v_invsqrt(const v_float64x2& a)
{
    const __m128d v_1 = _mm_set1_pd(1.);
    return _mm_div_pd(v_1, _mm_sqrt_pd(a));
}

//// Magnitude ////

inline v_float32x4 v_magnitude(const v_float32x4& a, const v_float32x4& b)
{ return _mm_sqrt_ps(v_fma(a, a, b * b)); }
inline v_float64x2 v_magnitude(const v_float64x2& a, const v_float64x2& b)
{ return _mm_sqrt_pd(v_fma(a, a, b * b)); }

inline v_float32x4 v_sqr_magnitude(const v_float32x4& a, const v_float32x4& b)
{ return v_fma(a, a, b * b); }
inline v_float64x2 v_sqr_magnitude(const v_float64x2& a, const v_float64x2& b)
{ return v_fma(a, a, b * b); }

//// Absolute value ////

inline v_uint8x16 v_abs(const v_int8x16& a)
{
#if CV_SSSE3
    return _mm_abs_epi8(a);
#else
    const __m128i z = _mm_setzero_si128();
    return _mm_min_epu8(a, _mm_sub_epi8(z, a));
#endif
}

inline v_uint16x8 v_abs(const v_int16x8& a)
{
#if CV_SSSE3
    return _mm_abs_epi16(a);
#else
    const __m128i z = _mm_setzero_si128();
    return _mm_max_epi16(a, _mm_sub_epi16(z, a));
#endif
}

inline v_uint32x4 v_abs(const v_int32x4& a)
{
#if CV_SSSE3
    return _mm_abs_epi32(a);
#else
    __m128i s = _mm_srli_epi32(a, 31);
    __m128i f = _mm_srai_epi32(a, 31);
    return _mm_add_epi32(_mm_xor_si128(a, f), s);
#endif
}

inline v_float32x4 v_abs(const v_float32x4& a)
{ return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))); }
inline v_float64x2 v_abs(const v_float64x2& a)
{ return _mm_and_pd(a, _mm_castsi128_pd(_mm_srli_epi64(_mm_set1_epi32(-1), 1))); }

//// Absolute difference ////

inline v_uint8x16 v_absdiff(const v_uint8x16& a, const v_uint8x16& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint16x8 v_absdiff(const v_uint16x8& a, const v_uint16x8& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b)
{ return v_max(a, b) - v_min(a, b); }

inline v_uint8x16 v_absdiff(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = v_sub_wrap(a, b);
    v_int8x16 m = v_int8x16(a < b);
    return v_uint8x16(v_sub_wrap(d ^ m, m));
}
inline v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b)
{
    return v_uint16x8(v_sub_wrap(v_max(a, b), v_min(a, b)));
}
inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{
    v_int32x4 d = a - b;
    v_int32x4 m = v_int32x4(a < b);
    return v_uint32x4((d ^ m) - m);
}

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{ return v_abs(a - b); }

inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{ return v_abs(a - b); }

//// Saturating absolute difference ////

inline v_int8x16 v_absdiffs(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = a - b;
    v_int8x16 m = v_int8x16(a < b);
    return (d ^ m) - m;
 }
inline v_int16x8 v_absdiffs(const v_int16x8& a, const v_int16x8& b)
{ return v_max(a, b) - v_min(a, b); }
