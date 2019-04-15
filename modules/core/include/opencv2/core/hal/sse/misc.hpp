// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Select ///

#if CV_SSE4_1
#define OPENCV_HAL_IMPL_SSE_SELECT(_Tvec, _Tmvec, suffix, cast) \
    inline _Tvec v_select(const _Tmvec& mask, const _Tvec& a, const _Tvec& b) \
    { return _mm_blendv_##suffix(b, a, cast(mask)); }
#else
#define OPENCV_HAL_IMPL_SSE_SELECT(_Tvec, _Tmvec, suffix, cast) \
    inline _Tvec v_select(const _Tmvec& mask, const _Tvec& a, const _Tvec& b) \
    { return b ^ ((b ^ a) & _Tvec(cast(mask))); }
#endif

OPENCV_HAL_IMPL_SSE_SELECT(v_uint8x16,  v_mask8x16, epi8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_SELECT(v_int8x16,   v_mask8x16, epi8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint16x8,  v_mask16x8, epi8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_SELECT(v_int16x8,   v_mask16x8, epi8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint32x4,  v_mask32x4, epi8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_SELECT(v_int32x4,   v_mask32x4, epi8, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_SELECT(v_float32x4, v_mask32x4, ps, _mm_castsi128_ps)
OPENCV_HAL_IMPL_SSE_SELECT(v_float64x2, v_mask64x2, pd, _mm_castsi128_pd)

//// convert to bitmask from the most significant bit ////
inline int v_signmask(const v_uint8x16& a)
{ return _mm_movemask_epi8(a); }
inline int v_signmask(const v_int8x16& a)
{ return _mm_movemask_epi8(a); }
inline int v_signmask(const v_mask8x16& a)
{ return _mm_movemask_epi8(a); }

inline int v_signmask(const v_int16x8& a)
{ return _mm_movemask_epi8(_mm_packs_epi16(a, a)) & 255; }
inline int v_signmask(const v_uint16x8& a)
{ return _mm_movemask_epi8(_mm_packs_epi16(a, a)) & 255; }
inline int v_signmask(const v_mask16x8& a)
{ return _mm_movemask_epi8(_mm_packs_epi16(a, a)) & 255; }

inline int v_signmask(const v_uint32x4& a)
{
    __m128i a1 = _mm_packs_epi32(a, a);
    a1 = _mm_packs_epi16(a1, a1);
    return _mm_movemask_epi8(a1) & 15;
}
inline int v_signmask(const v_int32x4& a)
{ return v_signmask(v_uint32x4(a)); }
inline int v_signmask(const v_mask32x4& a)
{ return v_signmask(v_uint32x4(a)); }

inline int v_signmask(const v_float32x4& a)
{ return _mm_movemask_ps(a); }
inline int v_signmask(const v_float64x2& a)
{ return _mm_movemask_pd(a); }

//// Checks ////

#define OPENCV_HAL_IMPL_SSE_CHECK(_Tvec, and_op, allmask)   \
    inline bool v_check_all(const _Tvec& a)                 \
    {                                                       \
        int bitmask = _mm_movemask_epi8(a);                 \
        return and_op(bitmask, allmask) == allmask;         \
    }                                                       \
    inline bool v_check_any(const _Tvec& a)                 \
    {                                                       \
        int bitmask = _mm_movemask_epi8(a);                 \
        return and_op(bitmask, allmask) != 0;               \
    }

OPENCV_HAL_IMPL_SSE_CHECK(v_uint8x16,  OPENCV_HAL_1ST, 65535)
OPENCV_HAL_IMPL_SSE_CHECK(v_int8x16,   OPENCV_HAL_1ST, 65535)
OPENCV_HAL_IMPL_SSE_CHECK(v_uint16x8,  OPENCV_HAL_AND, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK(v_int16x8,   OPENCV_HAL_AND, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK(v_uint32x4,  OPENCV_HAL_AND, (int)0x8888)
OPENCV_HAL_IMPL_SSE_CHECK(v_int32x4,   OPENCV_HAL_AND, (int)0x8888)

#define OPENCV_HAL_IMPL_SSE_CHECK_FP(_Tvec, allmask)  \
    inline bool v_check_all(const _Tvec& a)           \
    {                                                 \
        int mask = v_signmask(a);                     \
        return mask == allmask;                       \
    }                                                 \
    inline bool v_check_any(const _Tvec& a)           \
    {                                                 \
        int mask = v_signmask(a);                     \
        return mask != 0;                             \
    }

OPENCV_HAL_IMPL_SSE_CHECK_FP(v_float32x4, 15)
OPENCV_HAL_IMPL_SSE_CHECK_FP(v_float64x2, 3)

//// Nan ////

inline v_float32x4 v_not_nan(const v_float32x4& a)
{ return _mm_cmpord_ps(a, a); }
inline v_float64x2 v_not_nan(const v_float64x2& a)
{ return _mm_cmpord_pd(a, a); }