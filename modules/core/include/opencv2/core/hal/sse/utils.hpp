// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

///// workaround /////
#if !defined(__x86_64__) && !defined(_M_X64)

inline __m128i _mm_set_epi64x(int64 ai, int64 bi)
{ return _mm_set_epi32((int)(ai >> 32), (int)ai, (int)(bi >> 32), (int)bi); }

inline __m128i _mm_set1_epi64x(int64 ai)
{ return _mm_set_epi64x(ai, ai); }

inline int64 _mm_cvtsi128_si64(const __m128i& a)
{
    uint64 ai = _mm_cvtsi128_si32(a);
    uint64 bi = _mm_cvtsi128_si32(_mm_srli_epi64(a, 32));
    return (int64)(ai | bi << 32);
}
#elif defined(_MSC_VER) && _MSC_VER <= 1900
// workaround unexpected results on MSVC, up to 2015
inline __m128i _mm_set_epi64x(int64 ai, int64 bi)
{
    __m128i a = _mm_cvtsi64_si128(ai);
    __m128i b = _mm_cvtsi64_si128(bi);
    return _mm_unpacklo_epi64(b, a);
}
inline __m128i _mm_set1_epi64x(int64 ai)
{
    __m128i a = _mm_cvtsi64_si128(ai);
    return _mm_unpacklo_epi64(a, a);
}
#endif

///// Cast /////
// to __m128i
inline __m128i _mm_castsi128_non(const __m128i& a)
{ return a; }
inline __m128i _mm_castsi128_non(const __m128& a)
{ return _mm_castps_si128(a); }
inline __m128i _mm_castsi128_non(const __m128d& a)
{ return _mm_castpd_si128(a); }
// to __m128
inline __m128 _mm_castps_non(const __m128& a)
{ return a; }
inline __m128 _mm_castps_non(const __m128d& a)
{ return _mm_castpd_ps(a); }
inline __m128 _mm_castps_non(const __m128i& a)
{ return _mm_castsi128_ps(a); }
// to __m128d
inline __m128d _mm_castpd_non(const __m128d& a)
{ return a; }
inline __m128d _mm_castpd_non(const __m128& a)
{ return _mm_castps_pd(a); }
inline __m128d _mm_castpd_non(const __m128i& a)
{ return _mm_castsi128_pd(a); }