// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Rotate ////

#if !CV_SSSE3
template<int imm>
inline __m128i _mm_alignr_epi8_non(const __m128i& a, const __m128i& b)
{
    enum {imm2 = (16 - imm) & 0xFF};
    return _mm_or_si128(_mm_srli_si128(b, imm), _mm_slli_si128(a, imm2));
}
template<>
inline __m128i _mm_alignr_epi8_non<8>(const __m128i& a, const __m128i& b)
{ return _mm_unpacklo_epi64(_mm_unpackhi_epi64(b, b), a); }
template<>
inline __m128i _mm_alignr_epi8_non<16>(const __m128i& a, const __m128i&)
{ return a; }
template<>
inline __m128i _mm_alignr_epi8_non<0>(const __m128i&, const __m128i&b)
{ return b; }

#else
template<int imm>
inline __m128i _mm_alignr_epi8_non(const __m128i& a, const __m128i& b)
{ return _mm_alignr_epi8(a, b, imm); }
#endif // !CV_SSSE3

template<int imm, typename Tvec>
inline Tvec v_rotate_right(const Tvec& a)
{
    enum {immb = (imm * sizeof(typename Tvec::lane_type))};
    return Tvec::cast(_mm_srli_si128(_mm_castsi128_non(a), immb));
}
template<int imm, typename Tvec>
inline Tvec v_rotate_left(const Tvec& a)
{
    enum {immb = (imm * sizeof(typename Tvec::lane_type))};
    return Tvec::cast(_mm_slli_si128(_mm_castsi128_non(a), immb));
}

template<int imm, typename Tvec>
inline Tvec v_rotate_right(const Tvec& a, const Tvec& b)
{
    enum {immb = (imm * sizeof(typename Tvec::lane_type))};
    return Tvec::cast(_mm_alignr_epi8_non<immb>(_mm_castsi128_non(b), _mm_castsi128_non(a)));
}
template<int imm, typename Tvec>
inline Tvec v_rotate_left(const Tvec& a, const Tvec& b)
{
    enum {immb = ((Tvec::nlanes - imm) * sizeof(typename Tvec::lane_type))};
    return Tvec::cast(_mm_alignr_epi8_non<immb>(_mm_castsi128_non(a), _mm_castsi128_non(b)));
}

template<int s, typename _Tvec>
inline _Tvec v_extract(const _Tvec& a, const _Tvec& b)
{ return v_rotate_right<s>(a, b); }

//// Combine ////

template<typename Tvec>
inline Tvec v_combine_low(const Tvec& a, const Tvec& b)
{
    return Tvec::cast(_mm_unpacklo_epi64(
        _mm_castsi128_non(a), _mm_castsi128_non(b)
    ));
}

template<typename Tvec>
inline Tvec v_combine_high(const Tvec& a, const Tvec& b)
{
    return Tvec::cast(_mm_unpackhi_epi64(
        _mm_castsi128_non(a), _mm_castsi128_non(b)
    ));
}

template<typename Tvec>
inline void v_recombine(const Tvec& a, const Tvec& b, Tvec& c, Tvec& d)
{
    c = v_combine_low(a, b);
    d = v_combine_high(a, b);
}

//// Zip ////

#define OPENCV_HAL_IMPL_SSE_ZIP(_Tvec, suffix)          \
    inline void v_zip(const _Tvec& a0, const _Tvec& a1, \
                            _Tvec& b0,       _Tvec& b1) \
    {                                                   \
        b0 = _mm_unpacklo_##suffix(a0, a1);             \
        b1 = _mm_unpackhi_##suffix(a0, a1);             \
    }

OPENCV_HAL_IMPL_SSE_ZIP(v_uint8x16,  epi8)
OPENCV_HAL_IMPL_SSE_ZIP(v_int8x16,   epi8)
OPENCV_HAL_IMPL_SSE_ZIP(v_uint16x8,  epi16)
OPENCV_HAL_IMPL_SSE_ZIP(v_int16x8,   epi16)
OPENCV_HAL_IMPL_SSE_ZIP(v_uint32x4,  epi32)
OPENCV_HAL_IMPL_SSE_ZIP(v_int32x4,   epi32)
OPENCV_HAL_IMPL_SSE_ZIP(v_float32x4, ps)
OPENCV_HAL_IMPL_SSE_ZIP(v_float64x2, pd)

//// Transpose ////

template<typename Tvec>
inline void v_transpose4x4(const Tvec& a0, const Tvec& a1,
                           const Tvec& a2, const Tvec& a3,
                           Tvec& b0, Tvec& b1,
                           Tvec& b2, Tvec& b3)
{
    Tvec t0, t1, t2, t3;
    v_zip(a0, a1, t0, t2);
    v_zip(a2, a3, t1, t3);
    v_recombine(t0, t1, b0, b1);
    v_recombine(t2, t3, b2, b3);
}

//// Low Expand ////

inline v_uint16x8 v_expand_low(const v_uint8x16& a)
{
#if CV_SSE4_1
    return _mm_cvtepu8_epi16(a);
#else
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi8(a, z);
#endif
}
inline v_int16x8 v_expand_low(const v_int8x16& a)
{
#if CV_SSE4_1
    return _mm_cvtepi8_epi16(a);
#else
    return _mm_srai_epi16(_mm_unpacklo_epi8(a, a), 8);
#endif
}

inline v_uint32x4 v_expand_low(const v_uint16x8& a)
{
#if CV_SSE4_1
    return _mm_cvtepu16_epi32(a);
#else
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi16(a, z);
#endif
}
inline v_int32x4 v_expand_low(const v_int16x8& a)
{
#if CV_SSE4_1
    return _mm_cvtepi16_epi32(a);
#else
    return _mm_srai_epi32(_mm_unpacklo_epi16(a, a), 16);
#endif
}
inline v_uint64x2 v_expand_low(const v_uint32x4& a)
{
#if CV_SSE4_1
    return _mm_cvtepu32_epi64(a);
#else
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi32(a, z);
#endif
}
inline v_int64x2 v_expand_low(const v_int32x4& a)
{
#if CV_SSE4_1
    return _mm_cvtepi32_epi64(a);
#else
    return _mm_unpacklo_epi32(a, _mm_srai_epi32(a, 31));
#endif
}

inline v_mask16x8 v_expand_low(const v_mask8x16& a)
{ return _mm_unpacklo_epi8(a, a); }
inline v_mask32x4 v_expand_low(const v_mask16x8& a)
{ return _mm_unpacklo_epi16(a, a); }
inline v_mask64x2 v_expand_low(const v_mask32x4& a)
{ return _mm_unpacklo_epi32(a, a); }

//// High Expand ////

inline v_uint16x8 v_expand_high(const v_uint8x16& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpackhi_epi8(a, z);
}
inline v_int16x8 v_expand_high(const v_int8x16& a)
{ return _mm_srai_epi16(_mm_unpackhi_epi8(a, a), 8); }

inline v_uint32x4 v_expand_high(const v_uint16x8& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpackhi_epi16(a, z);
}
inline v_int32x4 v_expand_high(const v_int16x8& a)
{ return _mm_srai_epi32(_mm_unpackhi_epi16(a, a), 16); }

inline v_uint64x2 v_expand_high(const v_uint32x4& a)
{
    const __m128i z = _mm_setzero_si128();
    return _mm_unpackhi_epi32(a, z);
}
inline v_int64x2 v_expand_high(const v_int32x4& a)
{ return _mm_unpackhi_epi32(a, _mm_srai_epi32(a, 31)); }

inline v_mask16x8 v_expand_high(const v_mask8x16& a)
{ return _mm_unpackhi_epi8(a, a); }
inline v_mask32x4 v_expand_high(const v_mask16x8& a)
{ return _mm_unpackhi_epi16(a, a); }
inline v_mask64x2 v_expand_high(const v_mask32x4& a)
{ return _mm_unpackhi_epi32(a, a); }

//// Expand low and high ////

template<typename Tvec, typename Twvec>
inline void v_expand(const Tvec& a, Twvec& b0, Twvec& b1)
{
    b0 = v_expand_low(a);
    b1 = v_expand_high(a);
}

//// Load low and expand ////

template<typename Tp>
inline typename V128_Traits<Tp>::v_twice v_load_expand(const Tp* ptr)
{ return v_expand_low(v_load_low(ptr)); }

//// Quad expand ////

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    __m128i a = _mm_cvtsi32_si128(*(const int*)ptr);
#if CV_SSE4_1
    return _mm_cvtepu8_epi32(a);
#else
    const __m128i z = _mm_setzero_si128();
    return _mm_unpacklo_epi16(_mm_unpacklo_epi8(a, z), z);
#endif
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    __m128i a = _mm_cvtsi32_si128(*(const int*)ptr);
#if CV_SSE4_1
    return _mm_cvtepi8_epi32(a);
#else
    __m128i r = _mm_unpacklo_epi8(a, a);
    r = _mm_unpacklo_epi8(r, r);
    return _mm_srai_epi32(r, 24);
#endif
}

//// FP16 expand ////

inline v_float32x4 v_load_expand(const float16_t* ptr)
{
    const __m128i z = _mm_setzero_si128(), delta = _mm_set1_epi32(0x38000000);
    const __m128i signmask = _mm_set1_epi32(0x80000000), maxexp = _mm_set1_epi32(0x7c000000);
    const __m128 deltaf = _mm_castsi128_ps(_mm_set1_epi32(0x38800000));
    __m128i bits = _mm_unpacklo_epi16(z, _mm_loadl_epi64((const __m128i*)ptr)); // h << 16
    __m128i e = _mm_and_si128(bits, maxexp), sign = _mm_and_si128(bits, signmask);
    v_int32x4 t = _mm_add_epi32(_mm_srli_epi32(_mm_xor_si128(bits, sign), 3), delta); // ((h & 0x7fff) << 13) + delta
    v_int32x4 zt = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_add_epi32(t, _mm_set1_epi32(1 << 23))), deltaf));
    t = _mm_add_epi32(t, _mm_and_si128(delta, _mm_cmpeq_epi32(maxexp, e)));
    v_mask32x4 zmask = _mm_cmpeq_epi32(e, z);
    __m128i ft = v_select(zmask, zt, t);
    return _mm_castsi128_ps(_mm_or_si128(ft, sign));
}

//// Pack ////

// 16 to 8
inline v_uint8x16 v_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    const v_uint16x8 max(255);
    return _mm_packus_epi16(v_min(a, max), v_min(b, max));
}

inline void v_pack_store(uchar* ptr, const v_uint16x8& a)
{
    const v_uint16x8 max(255);
    __m128i a1 = v_min(a, max);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b)
{ return _mm_packus_epi16(a, b); }

inline void v_pack_u_store(uchar* ptr, const v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a, a)); }

template<int n> inline
v_uint8x16 v_rshr_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    const __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return _mm_packus_epi16(
        _mm_srli_epi16(_mm_adds_epu16(a, delta), n),
        _mm_srli_epi16(_mm_adds_epu16(b, delta), n)
    );
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x8& a)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    const __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srli_epi16(_mm_adds_epu16(a, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

template<int n> inline
v_uint8x16 v_rshr_pack_u(const v_int16x8& a, const v_int16x8& b)
{
    const __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return _mm_packus_epi16(
        _mm_srai_epi16(_mm_adds_epi16(a, delta), n),
        _mm_srai_epi16(_mm_adds_epi16(b, delta), n)
    );
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x8& a)
{
    const __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

inline v_int8x16 v_pack(const v_int16x8& a, const v_int16x8& b)
{ return _mm_packs_epi16(a, b); }

inline void v_pack_store(schar* ptr, const v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a, a)); }

template<int n> inline
v_int8x16 v_rshr_pack(const v_int16x8& a, const v_int16x8& b)
{
    const __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return _mm_packs_epi16(
        _mm_srai_epi16(_mm_adds_epi16(a, delta), n),
        _mm_srai_epi16(_mm_adds_epi16(b, delta), n)
    );
}

template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x8& a)
{
    const __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a1, a1));
}

// 32 to 16
inline v_uint16x8 v_pack_u(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    return _mm_packus_epi32(a, b);
#else
    const __m128i cvti16 = _mm_set1_epi32(-32768);
    const __m128i cvtu16 = _mm_set1_epi16(-32768);

    // preliminary saturate negative values to zero
    __m128i a1 = _mm_and_si128(a, _mm_cmpgt_epi32(a, _mm_set1_epi32(0)));
    __m128i b1 = _mm_and_si128(b, _mm_cmpgt_epi32(b, _mm_set1_epi32(0)));

    a1 = _mm_add_epi32(a1, cvti16);
    b1 = _mm_add_epi32(b1, cvti16);
    return _mm_add_epi16(_mm_packs_epi32(a1, b1), cvtu16);
#endif
}

inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    const v_uint32x4 max(65535);
    return v_pack_u(v_int32x4(v_min(a, max)), v_int32x4(v_min(b, max)));
}

inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
{
#if CV_SSE4_1
    __m128i a1 = _mm_packus_epi32(a, a);
#else
    const __m128i cvti16 = _mm_set1_epi32(-32768);
    const __m128i cvtu16 = _mm_set1_epi16(-32768);

     // preliminary saturate negative values to zero
    __m128i a1 = _mm_and_si128(a, _mm_cmpgt_epi32(a, _mm_set1_epi32(0)));

    a1 = _mm_add_epi32(a1, cvti16);
    a1 = _mm_add_epi16(_mm_packs_epi32(a1, a1), cvtu16);
#endif
    _mm_storel_epi64((__m128i*)ptr, a1);
}

inline void v_pack_store(ushort* ptr, const v_uint32x4& a)
{
    const v_uint32x4 max(65535);
    v_pack_u_store(ptr, v_int32x4(v_min(a, max)));
}

template<int n> inline
v_uint16x8 v_rshr_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    const __m128i delta = _mm_set1_epi32(1 << (n-1));
    v_int32x4 a1 = _mm_srli_epi32(_mm_add_epi32(a, delta), n);
    v_int32x4 b1 = _mm_srli_epi32(_mm_add_epi32(b, delta), n);
    return v_pack_u(a1, b1);
}

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x4& a)
{
    const __m128i delta = _mm_set1_epi32(1 << (n-1));
    v_int32x4 a1 = _mm_srli_epi32(_mm_add_epi32(a, delta), n);
    v_pack_u_store(ptr, a1);
}

template<int n> inline
v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    const __m128i delta = _mm_set1_epi32(1 << (n - 1));
    v_int32x4 a1 = _mm_srai_epi32(_mm_add_epi32(a, delta), n);
    v_int32x4 b1 = _mm_srai_epi32(_mm_add_epi32(b, delta), n);
    return v_pack_u(a1, b1);
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    const __m128i delta = _mm_set1_epi32(1 << (n - 1));
    v_int32x4 a1 = _mm_srai_epi32(_mm_add_epi32(a, delta), n);
    v_pack_u_store(ptr, a1);
}

inline v_int16x8 v_pack(const v_int32x4& a, const v_int32x4& b)
{ return _mm_packs_epi32(a, b); }

inline void v_pack_store(short* ptr, const v_int32x4& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a, a)); }

template<int n> inline
v_int16x8 v_rshr_pack(const v_int32x4& a, const v_int32x4& b)
{
    const __m128i delta = _mm_set1_epi32(1 << (n-1));
    return _mm_packs_epi32(
        _mm_srai_epi32(_mm_add_epi32(a, delta), n),
        _mm_srai_epi32(_mm_add_epi32(b, delta), n)
    );
}

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x4& a)
{
    const __m128i delta = _mm_set1_epi32(1 << (n-1));
    __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a1, a1));
}

// 64 to 32
// [a0 0 | b0 0]  [a1 0 | b1 0]
inline v_uint32x4 v_pack(const v_uint64x2& a, const v_uint64x2& b)
{
    __m128i v0 = _mm_unpacklo_epi32(a, b); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a, b); // b0 b1 0 0
    return _mm_unpacklo_epi32(v0, v1);
}

inline void v_pack_store(unsigned* ptr, const v_uint64x2& a)
{
    __m128i a1 = _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a1);
}

// [a0 0 | b0 0]  [a1 0 | b1 0]
inline v_int32x4 v_pack(const v_int64x2& a, const v_int64x2& b)
{
    __m128i v0 = _mm_unpacklo_epi32(a, b); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a, b); // b0 b1 0 0
    return _mm_unpacklo_epi32(v0, v1);
}

inline void v_pack_store(int* ptr, const v_int64x2& a)
{
    __m128i a1 = _mm_shuffle_epi32(a, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a1);
}

template<int n> inline
v_uint32x4 v_rshr_pack(const v_uint64x2& a, const v_uint64x2& b)
{
    const __m128i delta = _mm_set1_epi64x((int64)1 << (n-1));
    __m128i a1 = _mm_srli_epi64(_mm_add_epi64(a, delta), n);
    __m128i b1 = _mm_srli_epi64(_mm_add_epi64(b, delta), n);
    __m128i v0 = _mm_unpacklo_epi32(a1, b1); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a1, b1); // b0 b1 0 0
    return _mm_unpacklo_epi32(v0, v1);
}

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x2& a)
{
    const __m128i delta = _mm_set1_epi64x((int64)1 << (n-1));
    __m128i a1 = _mm_srli_epi64(_mm_add_epi64(a, delta), n);
    __m128i a2 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

template<int n> inline
v_int32x4 v_rshr_pack(const v_int64x2& a, const v_int64x2& b)
{
    const v_int64x2 delta = v_int64x2((int64)1 << (n-1));
    __m128i a1 = (a + delta) >> n;
    __m128i b1 = (b + delta) >> n;
    __m128i v0 = _mm_unpacklo_epi32(a1, b1); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a1, b1); // b0 b1 0 0
    return _mm_unpacklo_epi32(v0, v1);
}

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x2& a)
{
    const v_int64x2 delta = v_int64x2((int64)1 << (n-1));
    __m128i a1 = (a + delta) >> n;
    __m128i a2 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

// pack mask
inline v_mask8x16 v_pack(const v_mask16x8& a, const v_mask16x8& b)
{ return _mm_packs_epi16(a, b); }

inline v_mask16x8 v_pack(const v_mask32x4& a, const v_mask32x4& b)
{ return _mm_packs_epi32(a, b); }

inline v_mask8x16 v_pack(const v_mask32x4& a, const v_mask32x4& b,
                         const v_mask32x4& c,  const v_mask32x4& d)
{
    __m128i ab = _mm_packs_epi32(a, b);
    __m128i cd = _mm_packs_epi32(c, d);
    return _mm_packs_epi16(ab, cd);
}

inline v_mask8x16 v_pack(const v_mask64x2& a, const v_mask64x2& b, const v_mask64x2& c,
                         const v_mask64x2& d, const v_mask64x2& e, const v_mask64x2& f,
                         const v_mask64x2& g, const v_mask64x2& h)
{
    __m128i ab = _mm_packs_epi32(a, b);
    __m128i cd = _mm_packs_epi32(c, d);
    __m128i ef = _mm_packs_epi32(e, f);
    __m128i gh = _mm_packs_epi32(g, h);

    __m128i abcd = _mm_packs_epi32(ab, cd);
    __m128i efgh = _mm_packs_epi32(ef, gh);
    return _mm_packs_epi16(abcd, efgh);
}

// triplets
inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
#if CV_SSSE3
    return _mm_shuffle_epi8(vec, _mm_set_epi64x(0xffffff0f0e0d0c0a, 0x0908060504020100));
#else
    __m128i mask = _mm_set1_epi64x(0x00000000FFFFFFFF);
    __m128i a = _mm_srli_si128(_mm_or_si128(
        _mm_andnot_si128(mask, vec),
        _mm_and_si128(mask, _mm_sll_epi32(vec, _mm_set_epi64x(0, 8)))
    ), 1);
    return _mm_srli_si128(_mm_shufflelo_epi16(a, _MM_SHUFFLE(2, 1, 0, 3)), 2);
#endif
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec)
{ return v_uint8x16(v_pack_triplets(v_int8x16(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
#if CV_SSSE3
    return _mm_shuffle_epi8(vec, _mm_set_epi64x(0xffff0f0e0d0c0b0a, 0x0908050403020100));
#else
    return _mm_srli_si128(_mm_shufflelo_epi16(vec, _MM_SHUFFLE(2, 1, 0, 3)), 2);
#endif
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec)
{ return v_uint16x8(v_pack_triplets(v_int16x8(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec)
{ return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec)
{ return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec)
{ return vec; }

//// FP16 pack ////

inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
{
    const __m128i signmask = _mm_set1_epi32(0x80000000);
    const __m128i rval = _mm_set1_epi32(0x3f000000);

    v_int32x4 t = _mm_castps_si128(v);
    __m128i sign = _mm_srai_epi32(_mm_and_si128(t, signmask), 16);
    t = _mm_andnot_si128(signmask, t);

    v_mask32x4 finitemask = _mm_cmpgt_epi32(_mm_set1_epi32(0x47800000), t);
    v_mask32x4 isnan = _mm_cmpgt_epi32(t, _mm_set1_epi32(0x7f800000));
    v_int32x4 naninf = v_select(isnan, v_int32x4(_mm_set1_epi32(0x7e00)), v_int32x4(_mm_set1_epi32(0x7c00)));
    v_mask32x4 tinymask = _mm_cmpgt_epi32(_mm_set1_epi32(0x38800000), t);
    v_int32x4 tt = _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(t), _mm_castsi128_ps(rval)));
    tt = _mm_sub_epi32(tt, rval);
    __m128i odd = _mm_and_si128(_mm_srli_epi32(t, 13), _mm_set1_epi32(1));
    v_int32x4 nt = _mm_add_epi32(t, _mm_set1_epi32(0xc8000fff));
    nt = _mm_srli_epi32(_mm_add_epi32(nt, odd), 13);
    t = v_select(tinymask, tt, nt);
    t = v_select(finitemask, t, naninf);
    t = _mm_or_si128(t, sign);
    t = _mm_packs_epi32(t, t);
    _mm_storel_epi64((__m128i*)ptr, t);
}