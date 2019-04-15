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

//// Reduce ////

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(_Tpvec, scalartype, func, suffix, sbit) \
inline scalartype v_reduce_##func(const v_##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
    return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype v_reduce_##func(const v_u##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    __m128i smask = _mm_set1_epi16(sbit); \
    val = _mm_xor_si128(val, smask); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
    return (unsigned scalartype)(_mm_cvtsi128_si32(val) ^  sbit); \
}
#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8_SUM(_Tpvec, scalartype, suffix) \
inline scalartype v_reduce_sum(const v_##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 8)); \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 4)); \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 2)); \
    return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype v_reduce_sum(const v_u##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 8)); \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 4)); \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 2)); \
    return (unsigned scalartype)_mm_cvtsi128_si32(val); \
}
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, max, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, min, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8_SUM(int16x8, short, 16)

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(_Tpvec, scalartype, regtype, suffix, cast_from, cast_to, extract) \
inline scalartype v_reduce_sum(const _Tpvec& a) \
{ \
    regtype val = a.val; \
    val = _mm_add_##suffix(val, cast_to(_mm_srli_si128(cast_from(val), 8))); \
    val = _mm_add_##suffix(val, cast_to(_mm_srli_si128(cast_from(val), 4))); \
    return (scalartype)_mm_cvt##extract(val); \
}

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(_Tpvec, scalartype, func, scalar_func) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    scalartype CV_DECL_ALIGNED(16) buf[4]; \
    v_store_aligned(buf, a); \
    scalartype s0 = scalar_func(buf[0], buf[1]); \
    scalartype s1 = scalar_func(buf[2], buf[3]); \
    return scalar_func(s0, s1); \
}

OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_uint32x4, unsigned, __m128i, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP, si128_si32)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_int32x4, int, __m128i, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP, si128_si32)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_float32x4, float, __m128, ps, _mm_castps_si128, _mm_castsi128_ps, ss_f32)

inline double v_reduce_sum(const v_float64x2& a)
{
    double CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return idx[0] + idx[1];
}

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
#if CV_SSE3
    __m128 ab = _mm_hadd_ps(a.val, b.val);
    __m128 cd = _mm_hadd_ps(c.val, d.val);
    return v_float32x4(_mm_hadd_ps(ab, cd));
#else
    __m128 ac = _mm_add_ps(_mm_unpacklo_ps(a.val, c.val), _mm_unpackhi_ps(a.val, c.val));
    __m128 bd = _mm_add_ps(_mm_unpacklo_ps(b.val, d.val), _mm_unpackhi_ps(b.val, d.val));
    return v_float32x4(_mm_add_ps(_mm_unpacklo_ps(ac, bd), _mm_unpackhi_ps(ac, bd)));
#endif
}

OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, min, std::min)

inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
{
    return (unsigned)_mm_cvtsi128_si32(_mm_sad_epu8(a.val, b.val));
}
inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
{
    __m128i half = _mm_set1_epi8(0x7f);
    return (unsigned)_mm_cvtsi128_si32(_mm_sad_epu8(_mm_add_epi8(a.val, half),
                                                    _mm_add_epi8(b.val, half)));
}

v_uint16x8 v_absdiff(const v_uint16x8& a, const v_uint16x8& b);
v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b);
v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b);
v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b);
v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b);

inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint32x4 l, h;
    v_expand(v_absdiff(a, b), l, h);
    return v_reduce_sum(l + h);
}
inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
{
    v_uint32x4 l, h;
    v_expand(v_absdiff(a, b), l, h);
    return v_reduce_sum(l + h);
}
inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}
inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}
inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}

#define OPENCV_HAL_IMPL_SSE_POPCOUNT(_Tpvec) \
inline v_uint32x4 v_popcount(const _Tpvec& a) \
{ \
    __m128i m1 = _mm_set1_epi32(0x55555555); \
    __m128i m2 = _mm_set1_epi32(0x33333333); \
    __m128i m4 = _mm_set1_epi32(0x0f0f0f0f); \
    __m128i p = a.val; \
    p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 1), m1), _mm_and_si128(p, m1)); \
    p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 2), m2), _mm_and_si128(p, m2)); \
    p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 4), m4), _mm_and_si128(p, m4)); \
    p = _mm_adds_epi8(p, _mm_srli_si128(p, 1)); \
    p = _mm_adds_epi8(p, _mm_srli_si128(p, 2)); \
    return v_uint32x4(_mm_and_si128(p, _mm_set1_epi32(0x000000ff))); \
}

OPENCV_HAL_IMPL_SSE_POPCOUNT(v_uint8x16)
OPENCV_HAL_IMPL_SSE_POPCOUNT(v_uint16x8)
OPENCV_HAL_IMPL_SSE_POPCOUNT(v_uint32x4)
OPENCV_HAL_IMPL_SSE_POPCOUNT(v_int8x16)
OPENCV_HAL_IMPL_SSE_POPCOUNT(v_int16x8)
OPENCV_HAL_IMPL_SSE_POPCOUNT(v_int32x4)