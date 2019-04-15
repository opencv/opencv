// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Pairs ////

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
#if CV_SSSE3
    return _mm_shuffle_epi8(vec, _mm_set_epi64x(0x0f0d0e0c0b090a08, 0x0705060403010200));
#else
    __m128i a = _mm_shufflelo_epi16(vec, _MM_SHUFFLE(3, 1, 2, 0));
    a = _mm_shufflehi_epi16(a, _MM_SHUFFLE(3, 1, 2, 0));
    a = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_unpacklo_epi8(a, _mm_unpackhi_epi64(a, a));
#endif
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec)
{ return v_uint8x16(v_interleave_pairs(v_int8x16(vec))); }

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
#if CV_SSSE3
    return _mm_shuffle_epi8(vec, _mm_set_epi64x(0x0f0e0b0a0d0c0908, 0x0706030205040100));
#else
    __m128i a = _mm_shufflelo_epi16(vec, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_shufflehi_epi16(a, _MM_SHUFFLE(3, 1, 2, 0));
#endif
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec)
{ return v_uint16x8(v_interleave_pairs(v_int16x8(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{ return _mm_shuffle_epi32(vec, _MM_SHUFFLE(3, 1, 2, 0)); }
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec)
{ return v_uint32x4(v_interleave_pairs(v_int32x4(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec)
{ return v_float32x4::cast(v_interleave_pairs(v_int32x4::cast(vec))); }

//// Quad ////

inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
#if CV_SSSE3
    return _mm_shuffle_epi8(vec, _mm_set_epi64x(0x0f0b0e0a0d090c08, 0x0703060205010400));
#else
    __m128i a = _mm_shuffle_epi32(vec, _MM_SHUFFLE(3, 1, 2, 0));
    return _mm_unpacklo_epi8(a, _mm_unpackhi_epi64(a, a));
#endif
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec)
{ return v_uint8x16(v_interleave_quads(v_int8x16(vec))); }

inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
#if CV_SSSE3
    return _mm_shuffle_epi8(vec, _mm_set_epi64x(0x0f0e07060d0c0504, 0x0b0a030209080100));
#else
    return _mm_unpacklo_epi16(vec, _mm_unpackhi_epi64(vec, vec));
#endif
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec)
{ return v_uint16x8(v_interleave_quads(v_int16x8(vec))); }

//// Interleave ////

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi8(a, b);
    __m128i v1 = _mm_unpackhi_epi8(a, b);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 16), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 16), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 16), v1);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
#if CV_SSE4_1
    const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    __m128i a0 = _mm_shuffle_epi8(a, sh_a);
    __m128i b0 = _mm_shuffle_epi8(b, sh_b);
    __m128i c0 = _mm_shuffle_epi8(c, sh_c);

    const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
    __m128i v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
    __m128i v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);
#elif CV_SSSE3
    const __m128i m0 = _mm_setr_epi8(0, 6, 11, 1, 7, 12, 2, 8, 13, 3, 9, 14, 4, 10, 15, 5);
    const __m128i m1 = _mm_setr_epi8(5, 11, 0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10);
    const __m128i m2 = _mm_setr_epi8(10, 0, 5, 11, 1, 6, 12, 2, 7, 13, 3, 8, 14, 4, 9, 15);

    __m128i t0 = _mm_alignr_epi8(b, _mm_slli_si128(a, 10), 5);
    t0 = _mm_alignr_epi8(c, t0, 5);
    __m128i v0 = _mm_shuffle_epi8(t0, m0);

    __m128i t1 = _mm_alignr_epi8(_mm_srli_si128(b, 5), _mm_slli_si128(a, 5), 6);
    t1 = _mm_alignr_epi8(_mm_srli_si128(c, 5), t1, 5);
    __m128i v1 = _mm_shuffle_epi8(t1, m1);

    __m128i t2 = _mm_alignr_epi8(_mm_srli_si128(c, 10), b, 11);
    t2 = _mm_alignr_epi8(t2, a, 11);
    __m128i v2 = _mm_shuffle_epi8(t2, m2);
#else
    __m128i z = _mm_setzero_si128();
    __m128i ab0 = _mm_unpacklo_epi8(a, b);
    __m128i ab1 = _mm_unpackhi_epi8(a, b);
    __m128i c0 = _mm_unpacklo_epi8(c, z);
    __m128i c1 = _mm_unpackhi_epi8(c, z);

    __m128i p00 = _mm_unpacklo_epi16(ab0, c0);
    __m128i p01 = _mm_unpackhi_epi16(ab0, c0);
    __m128i p02 = _mm_unpacklo_epi16(ab1, c1);
    __m128i p03 = _mm_unpackhi_epi16(ab1, c1);

    __m128i p10 = _mm_unpacklo_epi32(p00, p01);
    __m128i p11 = _mm_unpackhi_epi32(p00, p01);
    __m128i p12 = _mm_unpacklo_epi32(p02, p03);
    __m128i p13 = _mm_unpackhi_epi32(p02, p03);

    __m128i p20 = _mm_unpacklo_epi64(p10, p11);
    __m128i p21 = _mm_unpackhi_epi64(p10, p11);
    __m128i p22 = _mm_unpacklo_epi64(p12, p13);
    __m128i p23 = _mm_unpackhi_epi64(p12, p13);

    p20 = _mm_slli_si128(p20, 1);
    p22 = _mm_slli_si128(p22, 1);

    __m128i p30 = _mm_slli_epi64(_mm_unpacklo_epi32(p20, p21), 8);
    __m128i p31 = _mm_srli_epi64(_mm_unpackhi_epi32(p20, p21), 8);
    __m128i p32 = _mm_slli_epi64(_mm_unpacklo_epi32(p22, p23), 8);
    __m128i p33 = _mm_srli_epi64(_mm_unpackhi_epi32(p22, p23), 8);

    __m128i p40 = _mm_unpacklo_epi64(p30, p31);
    __m128i p41 = _mm_unpackhi_epi64(p30, p31);
    __m128i p42 = _mm_unpacklo_epi64(p32, p33);
    __m128i p43 = _mm_unpackhi_epi64(p32, p33);

    __m128i v0 = _mm_or_si128(_mm_srli_si128(p40, 2), _mm_slli_si128(p41, 10));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(p41, 6), _mm_slli_si128(p42, 6));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(p42, 10), _mm_slli_si128(p43, 2));
#endif

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 16), v1);
        _mm_stream_si128((__m128i*)(ptr + 32), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 16), v1);
        _mm_store_si128((__m128i*)(ptr + 32), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 16), v1);
        _mm_storeu_si128((__m128i*)(ptr + 32), v2);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, const v_uint8x16& d,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    __m128i u0 = _mm_unpacklo_epi8(a, c); // a0 c0 a1 c1 ...
    __m128i u1 = _mm_unpackhi_epi8(a, c); // a8 c8 a9 c9 ...
    __m128i u2 = _mm_unpacklo_epi8(b, d); // b0 d0 b1 d1 ...
    __m128i u3 = _mm_unpackhi_epi8(b, d); // b8 d8 b9 d9 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2); // a0 b0 c0 d0 ...
    __m128i v1 = _mm_unpackhi_epi8(u0, u2); // a4 b4 c4 d4 ...
    __m128i v2 = _mm_unpacklo_epi8(u1, u3); // a8 b8 c8 d8 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a12 b12 c12 d12 ...

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 16), v1);
        _mm_stream_si128((__m128i*)(ptr + 32), v2);
        _mm_stream_si128((__m128i*)(ptr + 48), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 16), v1);
        _mm_store_si128((__m128i*)(ptr + 32), v2);
        _mm_store_si128((__m128i*)(ptr + 48), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 16), v1);
        _mm_storeu_si128((__m128i*)(ptr + 32), v2);
        _mm_storeu_si128((__m128i*)(ptr + 48), v3);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi16(a, b);
    __m128i v1 = _mm_unpackhi_epi16(a, b);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 8), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 8), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 8), v1);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a,
                                const v_uint16x8& b, const v_uint16x8& c,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
#if CV_SSE4_1
    const __m128i sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m128i sh_b = _mm_setr_epi8(10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    const __m128i sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    __m128i a0 = _mm_shuffle_epi8(a, sh_a);
    __m128i b0 = _mm_shuffle_epi8(b, sh_b);
    __m128i c0 = _mm_shuffle_epi8(c, sh_c);

    __m128i v0 = _mm_blend_epi16(_mm_blend_epi16(a0, b0, 0x92), c0, 0x24);
    __m128i v1 = _mm_blend_epi16(_mm_blend_epi16(c0, a0, 0x92), b0, 0x24);
    __m128i v2 = _mm_blend_epi16(_mm_blend_epi16(b0, c0, 0x92), a0, 0x24);
#else
    __m128i z = _mm_setzero_si128();
    __m128i ab0 = _mm_unpacklo_epi16(a, b);
    __m128i ab1 = _mm_unpackhi_epi16(a, b);
    __m128i c0 = _mm_unpacklo_epi16(c, z);
    __m128i c1 = _mm_unpackhi_epi16(c, z);

    __m128i p10 = _mm_unpacklo_epi32(ab0, c0);
    __m128i p11 = _mm_unpackhi_epi32(ab0, c0);
    __m128i p12 = _mm_unpacklo_epi32(ab1, c1);
    __m128i p13 = _mm_unpackhi_epi32(ab1, c1);

    __m128i p20 = _mm_unpacklo_epi64(p10, p11);
    __m128i p21 = _mm_unpackhi_epi64(p10, p11);
    __m128i p22 = _mm_unpacklo_epi64(p12, p13);
    __m128i p23 = _mm_unpackhi_epi64(p12, p13);

    p20 = _mm_slli_si128(p20, 2);
    p22 = _mm_slli_si128(p22, 2);

    __m128i p30 = _mm_unpacklo_epi64(p20, p21);
    __m128i p31 = _mm_unpackhi_epi64(p20, p21);
    __m128i p32 = _mm_unpacklo_epi64(p22, p23);
    __m128i p33 = _mm_unpackhi_epi64(p22, p23);

    __m128i v0 = _mm_or_si128(_mm_srli_si128(p30, 2), _mm_slli_si128(p31, 10));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(p31, 6), _mm_slli_si128(p32, 6));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(p32, 10), _mm_slli_si128(p33, 2));
#endif
    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 8), v1);
        _mm_stream_si128((__m128i*)(ptr + 16), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 8), v1);
        _mm_store_si128((__m128i*)(ptr + 16), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 8), v1);
        _mm_storeu_si128((__m128i*)(ptr + 16), v2);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                const v_uint16x8& c, const v_uint16x8& d,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    __m128i u0 = _mm_unpacklo_epi16(a, c); // a0 c0 a1 c1 ...
    __m128i u1 = _mm_unpackhi_epi16(a, c); // a4 c4 a5 c5 ...
    __m128i u2 = _mm_unpacklo_epi16(b, d); // b0 d0 b1 d1 ...
    __m128i u3 = _mm_unpackhi_epi16(b, d); // b4 d4 b5 d5 ...

    __m128i v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    __m128i v1 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    __m128i v2 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    __m128i v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 8), v1);
        _mm_stream_si128((__m128i*)(ptr + 16), v2);
        _mm_stream_si128((__m128i*)(ptr + 24), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 8), v1);
        _mm_store_si128((__m128i*)(ptr + 16), v2);
        _mm_store_si128((__m128i*)(ptr + 24), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 8), v1);
        _mm_storeu_si128((__m128i*)(ptr + 16), v2);
        _mm_storeu_si128((__m128i*)(ptr + 24), v3);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi32(a, b);
    __m128i v1 = _mm_unpackhi_epi32(a, b);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 4), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 4), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 4), v1);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                const v_uint32x4& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v_uint32x4 z = v_setzero_u32(), u0, u1, u2, u3;
    v_transpose4x4(a, b, c, z, u0, u1, u2, u3);

    __m128i v0 = _mm_or_si128(u0, _mm_slli_si128(u1, 12));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(u1, 4), _mm_slli_si128(u2, 8));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(u2, 8), _mm_slli_si128(u3, 4));

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 4), v1);
        _mm_stream_si128((__m128i*)(ptr + 8), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 4), v1);
        _mm_store_si128((__m128i*)(ptr + 8), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 4), v1);
        _mm_storeu_si128((__m128i*)(ptr + 8), v2);
    }
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               const v_uint32x4& c, const v_uint32x4& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v_uint32x4 v0, v1, v2, v3;
    v_transpose4x4(a, b, c, d, v0, v1, v2, v3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 4), v1);
        _mm_stream_si128((__m128i*)(ptr + 8), v2);
        _mm_stream_si128((__m128i*)(ptr + 12), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 4), v1);
        _mm_store_si128((__m128i*)(ptr + 8), v2);
        _mm_store_si128((__m128i*)(ptr + 12), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 4), v1);
        _mm_storeu_si128((__m128i*)(ptr + 8), v2);
        _mm_storeu_si128((__m128i*)(ptr + 12), v3);
    }
}

// 2-channel, float only
inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128 v0 = _mm_unpacklo_ps(a, b); // a0 b0 a1 b1
    __m128 v1 = _mm_unpackhi_ps(a, b); // a2 b2 a3 b3

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_ps(ptr, v0);
        _mm_stream_ps(ptr + 4, v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_ps(ptr, v0);
        _mm_store_ps(ptr + 4, v1);
    }
    else
    {
        _mm_storeu_ps(ptr, v0);
        _mm_storeu_ps(ptr + 4, v1);
    }
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128 u0 = _mm_shuffle_ps(a, b, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 u1 = _mm_shuffle_ps(c, a, _MM_SHUFFLE(1, 1, 0, 0));
    __m128 v0 = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u2 = _mm_shuffle_ps(b, c, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 u3 = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1 = _mm_shuffle_ps(u2, u3, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u4 = _mm_shuffle_ps(c, a, _MM_SHUFFLE(3, 3, 2, 2));
    __m128 u5 = _mm_shuffle_ps(b, c, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 v2 = _mm_shuffle_ps(u4, u5, _MM_SHUFFLE(2, 0, 2, 0));

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_ps(ptr, v0);
        _mm_stream_ps(ptr + 4, v1);
        _mm_stream_ps(ptr + 8, v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_ps(ptr, v0);
        _mm_store_ps(ptr + 4, v1);
        _mm_store_ps(ptr + 8, v2);
    }
    else
    {
        _mm_storeu_ps(ptr, v0);
        _mm_storeu_ps(ptr + 4, v1);
        _mm_storeu_ps(ptr + 8, v2);
    }
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, const v_float32x4& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128 u0 = _mm_unpacklo_ps(a, c);
    __m128 u1 = _mm_unpacklo_ps(b, d);
    __m128 u2 = _mm_unpackhi_ps(a, c);
    __m128 u3 = _mm_unpackhi_ps(b, d);
    __m128 v0 = _mm_unpacklo_ps(u0, u1);
    __m128 v2 = _mm_unpacklo_ps(u2, u3);
    __m128 v1 = _mm_unpackhi_ps(u0, u1);
    __m128 v3 = _mm_unpackhi_ps(u2, u3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_ps(ptr, v0);
        _mm_stream_ps(ptr + 4, v1);
        _mm_stream_ps(ptr + 8, v2);
        _mm_stream_ps(ptr + 12, v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_ps(ptr, v0);
        _mm_store_ps(ptr + 4, v1);
        _mm_store_ps(ptr + 8, v2);
        _mm_store_ps(ptr + 12, v3);
    }
    else
    {
        _mm_storeu_ps(ptr, v0);
        _mm_storeu_ps(ptr + 4, v1);
        _mm_storeu_ps(ptr + 8, v2);
        _mm_storeu_ps(ptr + 12, v3);
    }
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi64(a, b);
    __m128i v1 = _mm_unpackhi_epi64(a, b);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 2), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 2), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 2), v1);
    }
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi64(a, b);
    __m128i v1 = _mm_unpacklo_epi64(c, _mm_unpackhi_epi64(a, a));
    __m128i v2 = _mm_unpackhi_epi64(b, c);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 2), v1);
        _mm_stream_si128((__m128i*)(ptr + 4), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 2), v1);
        _mm_store_si128((__m128i*)(ptr + 4), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 2), v1);
        _mm_storeu_si128((__m128i*)(ptr + 4), v2);
    }
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, const v_uint64x2& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi64(a, b);
    __m128i v1 = _mm_unpacklo_epi64(c, d);
    __m128i v2 = _mm_unpackhi_epi64(a, b);
    __m128i v3 = _mm_unpackhi_epi64(c, d);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 2), v1);
        _mm_stream_si128((__m128i*)(ptr + 4), v2);
        _mm_stream_si128((__m128i*)(ptr + 6), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 2), v1);
        _mm_store_si128((__m128i*)(ptr + 4), v2);
        _mm_store_si128((__m128i*)(ptr + 6), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 2), v1);
        _mm_storeu_si128((__m128i*)(ptr + 4), v2);
        _mm_storeu_si128((__m128i*)(ptr + 6), v3);
    }
}

#define OPENCV_HAL_IMPL_SSE_SIGNED_INTERLEAVE(_Tvec0, _Tp0, _Tvec1, _Tp1)         \
    inline void v_store_interleave(_Tp0* ptr, const _Tvec0& a0, const _Tvec0& b0, \
                                   hal::StoreMode mode = hal::STORE_UNALIGNED)    \
    {                                                                             \
        _Tvec1 a1 = _Tvec1::cast(a0);                                             \
        _Tvec1 b1 = _Tvec1::cast(b0);                                             \
        v_store_interleave((_Tp1*)ptr, a1, b1, mode);                             \
    }                                                                             \
    inline void v_store_interleave(_Tp0* ptr, const _Tvec0& a0, const _Tvec0& b0, \
                                                                const _Tvec0& c0, \
                                   hal::StoreMode mode = hal::STORE_UNALIGNED)    \
    {                                                                             \
        _Tvec1 a1 = _Tvec1::cast(a0);                                             \
        _Tvec1 b1 = _Tvec1::cast(b0);                                             \
        _Tvec1 c1 = _Tvec1::cast(c0);                                             \
        v_store_interleave((_Tp1*)ptr, a1, b1, c1, mode);                         \
    }                                                                             \
    inline void v_store_interleave(_Tp0* ptr, const _Tvec0& a0, const _Tvec0& b0, \
                                   const _Tvec0& c0, const _Tvec0& d0,            \
                                   hal::StoreMode mode = hal::STORE_UNALIGNED)    \
    {                                                                             \
        _Tvec1 a1 = _Tvec1::cast(a0);                                             \
        _Tvec1 b1 = _Tvec1::cast(b0);                                             \
        _Tvec1 c1 = _Tvec1::cast(c0);                                             \
        _Tvec1 d1 = _Tvec1::cast(d0);                                             \
        v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1, mode);                     \
    }

OPENCV_HAL_IMPL_SSE_SIGNED_INTERLEAVE(v_int8x16,   schar,  v_uint8x16, uchar)
OPENCV_HAL_IMPL_SSE_SIGNED_INTERLEAVE(v_int16x8,   short,  v_uint16x8, ushort)
OPENCV_HAL_IMPL_SSE_SIGNED_INTERLEAVE(v_int32x4,   int,    v_uint32x4, unsigned)
OPENCV_HAL_IMPL_SSE_SIGNED_INTERLEAVE(v_int64x2,   int64,  v_uint64x2, uint64)
OPENCV_HAL_IMPL_SSE_SIGNED_INTERLEAVE(v_float64x2, double, v_uint64x2, uint64)