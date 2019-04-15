// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Non-saturated Arithmetic ////

#define OPENCV_HAL_IMPL_SSE_BIN_FUN(fun, _Tvec, intrin) \
    inline _Tvec fun(const _Tvec& a, const _Tvec& b)    \
    { return intrin(a, b); }

OPENCV_HAL_IMPL_SSE_BIN_FUN(v_add_wrap, v_uint8x16, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_add_wrap, v_int8x16,  _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_add_wrap, v_uint16x8, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_add_wrap, v_int16x8,  _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_sub_wrap, v_uint8x16, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_sub_wrap, v_int8x16,  _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_sub_wrap, v_uint16x8, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_sub_wrap, v_int16x8,  _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_mul_wrap, v_uint16x8, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUN(v_mul_wrap, v_int16x8,  _mm_mullo_epi16)

inline v_uint8x16 v_mul_wrap(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i ad = _mm_srai_epi16(a, 8);
    __m128i bd = _mm_srai_epi16(b, 8);
    v_uint8x16 p0 = _mm_mullo_epi16(a, b); // even
    v_uint8x16 p1 = _mm_slli_epi16(_mm_mullo_epi16(ad, bd), 8); // odd
    const v_mask8x16 mask = _mm_set1_epi32(0xFF00FF00);
    return v_select(mask, p1, p0);
}
inline v_int8x16 v_mul_wrap(const v_int8x16& a, const v_int8x16& b)
{ return v_int8x16(v_mul_wrap(v_uint8x16(a), v_uint8x16(b))); }

//// Dot Product ////

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{ return _mm_madd_epi16(a, b); }

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_dotprod(a, b) + c; }

//// Multiply and Add ////

inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{ return a * b + c; }

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
#if CV_FMA3
    return _mm_fmadd_ps(a, b, c);
#else
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
#if CV_FMA3
    return _mm_fmadd_pd(a, b, c);
#else
    return _mm_add_pd(_mm_mul_pd(a, b), c);
#endif
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{ return v_fma(a, b, c); }
inline v_float32x4 v_muladd(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{ return v_fma(a, b, c); }
inline v_float64x2 v_muladd(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{ return v_fma(a, b, c); }

//// Multiply and extract high ////

inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{ return _mm_mulhi_epu16(a, b); }
inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{ return _mm_mulhi_epi16(a, b); }

//// Multiply and expand ////

inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    v_uint16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    v_int16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    __m128i v0 = _mm_mullo_epi16(a, b);
    __m128i v1 = _mm_mulhi_epi16(a, b);
    c = _mm_unpacklo_epi16(v0, v1);
    d = _mm_unpackhi_epi16(v0, v1);
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    __m128i v0 = _mm_mullo_epi16(a, b);
    __m128i v1 = _mm_mulhi_epu16(a, b);
    c = _mm_unpacklo_epi16(v0, v1);
    d = _mm_unpackhi_epi16(v0, v1);
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    __m128i c0 = _mm_mul_epu32(a, b);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
    c = _mm_unpacklo_epi64(c0, c1);
    d = _mm_unpackhi_epi64(c0, c1);
}

//// Matrix multiplication  ////

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    __m128 v0 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)), m0);
    __m128 v1 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)), m1);
    __m128 v2 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)), m2);
    __m128 v3 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(3, 3, 3, 3)), m3);
    return _mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, v3));
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    __m128 v0 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(0, 0, 0, 0)), m0);
    __m128 v1 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(1, 1, 1, 1)), m1);
    __m128 v2 = _mm_mul_ps(_mm_shuffle_ps(v, v, _MM_SHUFFLE(2, 2, 2, 2)), m2);
    return _mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, a));
}