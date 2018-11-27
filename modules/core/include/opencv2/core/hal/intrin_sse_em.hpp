// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_INTRIN_SSE_EM_HPP
#define OPENCV_HAL_INTRIN_SSE_EM_HPP

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define OPENCV_HAL_SSE_WRAP_1(fun, tp) \
    inline tp _v128_##fun(const tp& a) \
    { return _mm_##fun(a); }

#define OPENCV_HAL_SSE_WRAP_2(fun, tp) \
    inline tp _v128_##fun(const tp& a, const tp& b) \
    { return _mm_##fun(a, b); }

#define OPENCV_HAL_SSE_WRAP_3(fun, tp) \
    inline tp _v128_##fun(const tp& a, const tp& b, const tp& c) \
    { return _mm_##fun(a, b, c); }

///////////////////////////// XOP /////////////////////////////

// [todo] define CV_XOP
#if 1 // CV_XOP
inline __m128i _v128_comgt_epu32(const __m128i& a, const __m128i& b)
{
    const __m128i delta = _mm_set1_epi32((int)0x80000000);
    return _mm_cmpgt_epi32(_mm_xor_si128(a, delta), _mm_xor_si128(b, delta));
}
// wrapping XOP
#else
OPENCV_HAL_SSE_WRAP_2(_v128_comgt_epu32, __m128i)
#endif // !CV_XOP

///////////////////////////// SSE4.1 /////////////////////////////

#if !CV_SSE4_1

/** Swizzle **/
inline __m128i _v128_blendv_epi8(const __m128i& a, const __m128i& b, const __m128i& mask)
{ return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(b, a), mask)); }

/** Convert **/
// 8 >> 16
inline __m128i _v128_cvtepu8_epi16(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi8(a, z);
}
inline __m128i _v128_cvtepi8_epi16(const __m128i& a)
{ return _mm_srai_epi16(_mm_unpacklo_epi8(a, a), 8); }
// 8 >> 32
inline __m128i _v128_cvtepu8_epi32(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(a, z), z);
}
inline __m128i _v128_cvtepi8_epi32(const __m128i& a)
{
    __m128i r = _mm_unpacklo_epi8(a, a);
    r = _mm_unpacklo_epi8(r, r);
    return _mm_srai_epi32(r, 24);
}
// 16 >> 32
inline __m128i _v128_cvtepu16_epi32(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi16(a, z);
}
inline __m128i _v128_cvtepi16_epi32(const __m128i& a)
{ return _mm_srai_epi32(_mm_unpacklo_epi16(a, a), 16); }
// 32 >> 64
inline __m128i _v128_cvtepu32_epi64(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi32(a, z);
}
inline __m128i _v128_cvtepi32_epi64(const __m128i& a)
{ return _mm_unpacklo_epi32(a, _mm_srai_epi32(a, 31)); }

/** Arithmetic **/
inline __m128i _v128_mullo_epi32(const __m128i& a, const __m128i& b)
{
    __m128i c0 = _mm_mul_epu32(a, b);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
    __m128i d0 = _mm_unpacklo_epi32(c0, c1);
    __m128i d1 = _mm_unpackhi_epi32(c0, c1);
    return _mm_unpacklo_epi64(d0, d1);
}

/** Math **/
inline __m128i _v128_min_epu32(const __m128i& a, const __m128i& b)
{ return _v128_blendv_epi8(a, b, _v128_comgt_epu32(a, b)); }

// wrapping SSE4.1
#else
OPENCV_HAL_SSE_WRAP_1(cvtepu8_epi16, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepi8_epi16, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepu8_epi32, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepi8_epi32, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepu16_epi32, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepi16_epi32, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepu32_epi64, __m128i)
OPENCV_HAL_SSE_WRAP_1(cvtepi32_epi64, __m128i)
OPENCV_HAL_SSE_WRAP_2(min_epu32, __m128i)
OPENCV_HAL_SSE_WRAP_2(mullo_epi32, __m128i)
OPENCV_HAL_SSE_WRAP_3(blendv_epi8, __m128i)
#endif // !CV_SSE4_1

///////////////////////////// Revolutionary /////////////////////////////

/** Convert **/
// 16 << 8
inline __m128i _v128_cvtepu8_epi16_high(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpackhi_epi8(a, z);
}
inline __m128i _v128_cvtepi8_epi16_high(const __m128i& a)
{ return _mm_srai_epi16(_mm_unpackhi_epi8(a, a), 8); }
// 32 << 16
inline __m128i _v128_cvtepu16_epi32_high(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpackhi_epi16(a, z);
}
inline __m128i _v128_cvtepi16_epi32_high(const __m128i& a)
{ return _mm_srai_epi32(_mm_unpackhi_epi16(a, a), 16); }
// 64 << 32
inline __m128i _v128_cvtepu32_epi64_high(const __m128i& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpackhi_epi32(a, z);
}
inline __m128i _v128_cvtepi32_epi64_high(const __m128i& a)
{ return _mm_unpackhi_epi32(a, _mm_srai_epi32(a, 31)); }

/** Miscellaneous **/
inline __m128i _v128_packs_epu32(const __m128i& a, const __m128i& b)
{
    const __m128i m = _mm_set1_epi32(65535);
    __m128i am = _v128_min_epu32(a, m);
    __m128i bm = _v128_min_epu32(b, m);
#if CV_SSE4_1
    return _mm_packus_epi32(am, bm);
#else
    const __m128i d = _mm_set1_epi32(32768), nd = _mm_set1_epi16(-32768);
    am = _mm_sub_epi32(am, d);
    bm = _mm_sub_epi32(bm, d);
    am = _mm_packs_epi32(am, bm);
    return _mm_sub_epi16(am, nd);
#endif
}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // OPENCV_HAL_INTRIN_SSE_EM_HPP