// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Shift ////

#define OPENCV_HAL_IMPL_SSE_OP_SHIFT(_Tuvec, _Tsvec, suffix, rsuffix)  \
    inline _Tuvec operator << (const _Tuvec& a, int imm)               \
    { return _mm_slli_##suffix(a, imm); }                              \
    inline _Tsvec operator << (const _Tsvec& a, int imm)               \
    { return _mm_slli_##suffix(a, imm); }                              \
    inline _Tuvec operator >> (const _Tuvec& a, int imm)               \
    { return _mm_srli_##suffix(a, imm); }                              \
    inline _Tsvec operator >> (const _Tsvec& a, int imm)               \
    { return _mm_srai_##rsuffix(a, imm); }                             \
    template<int imm>                                                  \
    inline _Tuvec v_shl(const _Tuvec& a)                               \
    { return _mm_slli_##suffix(a, imm); }                              \
    template<int imm>                                                  \
    inline _Tsvec v_shl(const _Tsvec& a)                               \
    { return _mm_slli_##suffix(a, imm); }                              \
    template<int imm>                                                  \
    inline _Tuvec v_shr(const _Tuvec& a)                               \
    { return _mm_srli_##suffix(a, imm); }                              \
    template<int imm>                                                  \
    inline _Tsvec v_shr(const _Tsvec& a)                               \
    { return _mm_srai_##rsuffix(a, imm); }

OPENCV_HAL_IMPL_SSE_OP_SHIFT(v_uint16x8, v_int16x8, epi16, epi16)
OPENCV_HAL_IMPL_SSE_OP_SHIFT(v_uint32x4, v_int32x4, epi32, epi32)

inline __m128i _mm_srai_epi64_non(__m128i a, int imm)
{
    const __m128i d = _mm_set1_epi64x((int64)1 << 63);
    __m128i r = _mm_srli_epi64(_mm_add_epi64(a, d), imm);
    return _mm_sub_epi64(r, _mm_srli_epi64(d, imm));
}
OPENCV_HAL_IMPL_SSE_OP_SHIFT(v_uint64x2, v_int64x2, epi64, epi64_non)

//// Logic ////

#define OPENCV_HAL_IMPL_SSE_OP_BIN(bin_op, _Tvec, intrin)         \
    inline _Tvec operator bin_op (const _Tvec& a, const _Tvec& b) \
    { return intrin(a, b); }                                      \
    inline _Tvec& operator bin_op##= (_Tvec& a, const _Tvec& b)   \
    { a = intrin(a, b); return a; }

#define OPENCV_HAL_IMPL_SSE_OP_LOGIC(_Tvec, suffix, cast)   \
    OPENCV_HAL_IMPL_SSE_OP_BIN(&, _Tvec, _mm_and_##suffix)  \
    OPENCV_HAL_IMPL_SSE_OP_BIN(|, _Tvec, _mm_or_##suffix)   \
    OPENCV_HAL_IMPL_SSE_OP_BIN(^, _Tvec, _mm_xor_##suffix)  \
    inline _Tvec operator ~ (const _Tvec& a)                \
    {                                                       \
        const _Tvec mask = cast(_mm_set1_epi32(-1));        \
        return a ^ mask;                                    \
    }

OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_uint8x16,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_int8x16,   si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_uint16x8,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_int16x8,   si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_uint32x4,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_int32x4,   si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_uint64x2,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_int64x2,   si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_float32x4, ps, _mm_castsi128_ps)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_float64x2, pd, _mm_castsi128_pd)

OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_mask8x16,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_mask16x8,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_mask32x4,  si128, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_OP_LOGIC(v_mask64x2,  si128, OPENCV_HAL_NOP)

//// Comparison ////

//// Unsigned Greater Than and Equal ////
inline v_mask8x16 operator >= (const v_uint8x16& a, const v_uint8x16& b)
{
#if CV_XOP
    return _mm_comge_epu8(a, b);
#else
    return _mm_cmpeq_epi8(a, _mm_max_epu8(a, b));
#endif
}
inline v_mask16x8 operator >= (const v_uint16x8& a, const v_uint16x8& b)
{
#if CV_XOP
    return _mm_comge_epu16(a, b);
#elif CV_SSE4_1
    return _mm_cmpeq_epi16(a, _mm_max_epu16(a, b));
#else
    return _mm_cmpeq_epi16(_mm_subs_epu16(b, a), _mm_setzero_si128());
#endif
}
inline v_mask32x4 operator >= (const v_uint32x4& a, const v_uint32x4& b)
{
#if CV_XOP
    return _mm_comge_epu32(a, b);
#elif CV_SSE4_1
    return _mm_cmpeq_epi32(a, _mm_max_epu32(a, b));
#else
    return ~(v_mask32x4(_mm_cmpgt_epi32(b, a)));
#endif
}

//// Rest Comparison for all int except 64bit ////
#define OPENCV_HAL_IMPL_SSE_OP_CMP_OV(_Tvec, _Tmvec)           \
    inline _Tmvec operator <  (const _Tvec& a, const _Tvec& b) \
    { return b > a; }                                          \
    inline _Tmvec operator <= (const _Tvec& a, const _Tvec& b) \
    { return b >= a; }

#if CV_XOP
    #define OPENCV_HAL_IMPL_SSE_OP_CMP_X(_Tuvec, _Tsvec, _Tmvec, len, sign) \
        inline _Tmvec operator != (const _Tuvec& a, const _Tuvec& b)        \
        { return _mm_comneq_epi##len(a, b); }                               \
        inline _Tmvec operator > (const _Tuvec& a, const _Tuvec& b)         \
        { return _mm_comgt_epu##len(a, b); }                                \
        inline _Tmvec operator != (const _Tsvec& a, const _Tsvec& b)        \
        { return _mm_comneq_epi##len(a, b); }                               \
        inline _Tmvec operator >= (const _Tsvec& a, const _Tsvec& b)        \
        { return _mm_comge_epi##len(a, b); }

#else
    #define OPENCV_HAL_IMPL_SSE_OP_CMP_X(_Tuvec, _Tsvec, _Tmvec, len, sign) \
        inline _Tmvec operator != (const _Tuvec& a, const _Tuvec& b)        \
        { return ~(a == b); }                                               \
        inline _Tmvec operator > (const _Tuvec& a, const _Tuvec& b)         \
        {                                                                   \
            const __m128i sbit = _mm_set1_epi##len(sign);                   \
            return _mm_cmpgt_epi##len(                                      \
                _mm_xor_si128(a, sbit), _mm_xor_si128(b, sbit)              \
            );                                                              \
        }                                                                   \
        inline _Tmvec operator != (const _Tsvec& a, const _Tsvec& b)        \
        { return ~(a == b); }                                               \
        inline _Tmvec operator >= (const _Tsvec& a, const _Tsvec& b)        \
        { return ~(b > a); }
#endif

#define OPENCV_HAL_IMPL_SSE_OP_CMP(_Tuvec, _Tsvec, _Tmvec, len, sign) \
    inline _Tmvec operator == (const _Tuvec& a, const _Tuvec& b)      \
    { return _mm_cmpeq_epi##len(a, b); }                              \
    inline _Tmvec operator == (const _Tsvec& a, const _Tsvec& b)      \
    { return _mm_cmpeq_epi##len(a, b); }                              \
    inline _Tmvec operator > (const _Tsvec& a, const _Tsvec& b)       \
    { return _mm_cmpgt_epi##len(a, b); }                              \
    OPENCV_HAL_IMPL_SSE_OP_CMP_X(_Tuvec, _Tsvec, _Tmvec, len, sign)   \
    OPENCV_HAL_IMPL_SSE_OP_CMP_OV(_Tuvec, _Tmvec)                     \
    OPENCV_HAL_IMPL_SSE_OP_CMP_OV(_Tsvec, _Tmvec)

OPENCV_HAL_IMPL_SSE_OP_CMP(v_uint8x16, v_int8x16, v_mask8x16, 8,  (char)-128)
OPENCV_HAL_IMPL_SSE_OP_CMP(v_uint16x8, v_int16x8, v_mask16x8, 16, (short)-32768)
OPENCV_HAL_IMPL_SSE_OP_CMP(v_uint32x4, v_int32x4, v_mask32x4, 32, (int)0x80000000)

//// 64-bit Comparison ////
// signed
inline v_mask64x2 operator == (const v_int64x2& a, const v_int64x2& b)
{
#if CV_SSE4_1
    return _mm_cmpeq_epi64(a, b);
#else
    return _mm_castpd_si128(_mm_cmpeq_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b)));
#endif
}
inline v_mask64x2 operator != (const v_int64x2& a, const v_int64x2& b)
{
#if CV_XOP
    return _mm_comneq_epi64(a, b);
#else
    return ~(a == b);
#endif
}
inline v_mask64x2 operator > (const v_int64x2& a, const v_int64x2& b)
{
#if CV_SSE4_2
    return _mm_cmpgt_epi64(a, b);
#else
    return _mm_castpd_si128(_mm_cmpgt_pd(_mm_castsi128_pd(a), _mm_castsi128_pd(b)));
#endif
}
inline v_mask64x2 operator >= (const v_int64x2& a, const v_int64x2& b)
{
#if CV_XOP
    return _mm_comge_epi64(a, b);
#else
    return ~(b > a);
#endif
}
OPENCV_HAL_IMPL_SSE_OP_CMP_OV(v_int64x2, v_mask64x2)

// unsigned
inline v_mask64x2 operator == (const v_uint64x2& a, const v_uint64x2& b)
{ return v_int64x2(a) == v_int64x2(b); }
inline v_mask64x2 operator != (const v_uint64x2& a, const v_uint64x2& b)
{ return v_int64x2(a) != v_int64x2(b); }

inline v_mask64x2 operator > (const v_uint64x2& a, const v_uint64x2& b)
{
#if CV_XOP
    return _mm_comgt_epu64(a, b);
#else
    const v_int64x2 sbit(0x7FFFFFFFFFFFFFFF);
    return (v_int64x2(a) ^ sbit) > (v_int64x2(b) ^ sbit);
#endif
}
inline v_mask64x2 operator >= (const v_uint64x2& a, const v_uint64x2& b)
{
#if CV_XOP
    return _mm_comge_epu64(a, b);
#else
    return ~(b > a);
#endif
}
OPENCV_HAL_IMPL_SSE_OP_CMP_OV(v_uint64x2, v_mask64x2)


//// Floating-point Comparison ////

#define OPENCV_HAL_IMPL_SSE_OP_CMP_FP(_Tvec, _Tmvec, suffix)        \
    inline _Tmvec operator == (const _Tvec& a, const _Tvec& b)      \
    { return _mm_cast##suffix##_si128(_mm_cmpeq_##suffix(a, b)); }  \
    inline _Tmvec operator != (const _Tvec& a, const _Tvec& b)      \
    { return _mm_cast##suffix##_si128(_mm_cmpneq_##suffix(a, b)); } \
    inline _Tmvec operator < (const _Tvec& a, const _Tvec& b)       \
    { return _mm_cast##suffix##_si128(_mm_cmplt_##suffix(a, b)); }  \
    inline _Tmvec operator > (const _Tvec& a, const _Tvec& b)       \
    { return _mm_cast##suffix##_si128(_mm_cmpgt_##suffix(a, b)); }  \
    inline _Tmvec operator <= (const _Tvec& a, const _Tvec& b)      \
    { return _mm_cast##suffix##_si128(_mm_cmple_##suffix(a, b)); }  \
    inline _Tmvec operator >= (const _Tvec& a, const _Tvec& b)      \
    { return _mm_cast##suffix##_si128(_mm_cmpge_##suffix(a, b)); }

OPENCV_HAL_IMPL_SSE_OP_CMP_FP(v_float32x4, v_mask32x4, ps)
OPENCV_HAL_IMPL_SSE_OP_CMP_FP(v_float64x2, v_mask64x2, pd)


//// Arithmetic ////

// saturated
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_uint8x16,  _mm_adds_epu8)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_uint8x16,  _mm_subs_epu8)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_int8x16,   _mm_adds_epi8)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_int8x16,   _mm_subs_epi8)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_uint16x8,  _mm_adds_epu16)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_uint16x8,  _mm_subs_epu16)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_int16x8,   _mm_adds_epi16)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_int16x8,   _mm_subs_epi16)

#define OPENCV_HAL_IMPL_SSE_MUL_SAT(_Tvec, _Twvec)            \
    inline _Tvec operator * (const _Tvec& a, const _Tvec& b)  \
    {                                                         \
        _Twvec c, d;                                          \
        v_mul_expand(a, b, c, d);                             \
        return v_pack(c, d);                                  \
    }                                                         \
    inline _Tvec& operator *= (_Tvec& a, const _Tvec& b)      \
    { a = a * b; return a; }

OPENCV_HAL_IMPL_SSE_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_SSE_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_SSE_MUL_SAT(v_uint16x8, v_uint32x4)
OPENCV_HAL_IMPL_SSE_MUL_SAT(v_int16x8,  v_int32x4)

// non-saturated
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_uint32x4,  _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_uint32x4,  _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_int32x4,   _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_int32x4,   _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_float32x4, _mm_add_ps)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_float32x4, _mm_sub_ps)
OPENCV_HAL_IMPL_SSE_OP_BIN(*, v_float32x4, _mm_mul_ps)
OPENCV_HAL_IMPL_SSE_OP_BIN(/, v_float32x4, _mm_div_ps)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_float64x2, _mm_add_pd)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_float64x2, _mm_sub_pd)
OPENCV_HAL_IMPL_SSE_OP_BIN(*, v_float64x2, _mm_mul_pd)
OPENCV_HAL_IMPL_SSE_OP_BIN(/, v_float64x2, _mm_div_pd)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_uint64x2,  _mm_add_epi64)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_uint64x2,  _mm_sub_epi64)
OPENCV_HAL_IMPL_SSE_OP_BIN(+, v_int64x2,   _mm_add_epi64)
OPENCV_HAL_IMPL_SSE_OP_BIN(-, v_int64x2,   _mm_sub_epi64)

inline v_uint32x4 operator * (const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i c0 = _mm_mul_epu32(a, b);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a, 32), _mm_srli_epi64(b, 32));
    __m128i d0 = _mm_unpacklo_epi32(c0, c1);
    __m128i d1 = _mm_unpackhi_epi32(c0, c1);
    return _mm_unpacklo_epi64(d0, d1);
}
inline v_uint32x4& operator *= (v_uint32x4& a, const v_uint32x4& b)
{ a = a * b; return a; }

inline v_int32x4 operator * (const v_int32x4& a, const v_int32x4& b)
{ return v_int32x4(v_uint32x4(a) * v_uint32x4(b)); }
inline v_int32x4& operator *= (v_int32x4& a, const v_int32x4& b)
{ a = a * b; return a; }