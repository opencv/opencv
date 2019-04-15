// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b)
{
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 16));

    __m128i t10 = _mm_unpacklo_epi8(t00, t01);
    __m128i t11 = _mm_unpackhi_epi8(t00, t01);

    __m128i t20 = _mm_unpacklo_epi8(t10, t11);
    __m128i t21 = _mm_unpackhi_epi8(t10, t11);

    __m128i t30 = _mm_unpacklo_epi8(t20, t21);
    __m128i t31 = _mm_unpackhi_epi8(t20, t21);

    a = _mm_unpacklo_epi8(t30, t31);
    b = _mm_unpackhi_epi8(t30, t31);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c)
{
#if CV_SSE4_1
    const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i s0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i s1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i s2 = _mm_loadu_si128((const __m128i*)(ptr + 32));
    __m128i a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
    __m128i b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
    __m128i c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
    const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
    const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
    const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    a = _mm_shuffle_epi8(a0, sh_b);
    b = _mm_shuffle_epi8(b0, sh_g);
    c = _mm_shuffle_epi8(c0, sh_r);
#elif CV_SSSE3
    const __m128i m0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
    const __m128i m1 = _mm_alignr_epi8(m0, m0, 11);
    const __m128i m2 = _mm_alignr_epi8(m0, m0, 6);

    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i t2 = _mm_loadu_si128((const __m128i*)(ptr + 32));

    __m128i s0 = _mm_shuffle_epi8(t0, m0);
    __m128i s1 = _mm_shuffle_epi8(t1, m1);
    __m128i s2 = _mm_shuffle_epi8(t2, m2);

    t0 = _mm_alignr_epi8(s1, _mm_slli_si128(s0, 10), 5);
    a = _mm_alignr_epi8(s2, t0, 5);

    t1 = _mm_alignr_epi8(_mm_srli_si128(s1, 5), _mm_slli_si128(s0, 5), 6);
    b = _mm_alignr_epi8(_mm_srli_si128(s2, 5), t1, 5);

    t2 = _mm_alignr_epi8(_mm_srli_si128(s2, 10), s1, 11);
    c = _mm_alignr_epi8(t2, s0, 11);
#else
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i t02 = _mm_loadu_si128((const __m128i*)(ptr + 32));

    __m128i t10 = _mm_unpacklo_epi8(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi8(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi8(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi8(t11, _mm_unpackhi_epi64(t12, t12));

    __m128i t30 = _mm_unpacklo_epi8(t20, _mm_unpackhi_epi64(t21, t21));
    __m128i t31 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t20, t20), t22);
    __m128i t32 = _mm_unpacklo_epi8(t21, _mm_unpackhi_epi64(t22, t22));

    a = _mm_unpacklo_epi8(t30, _mm_unpackhi_epi64(t31, t31));
    b = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t30, t30), t32);
    c = _mm_unpacklo_epi8(t31, _mm_unpackhi_epi64(t32, t32));
#endif
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c, v_uint8x16& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 32)); // a8 b8 c8 d8 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 48)); // a12 b12 c12 d12 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2); // a0 a8 b0 b8 ...
    __m128i v1 = _mm_unpackhi_epi8(u0, u2); // a2 a10 b2 b10 ...
    __m128i v2 = _mm_unpacklo_epi8(u1, u3); // a4 a12 b4 b12 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a6 a14 b6 b14 ...

    u0 = _mm_unpacklo_epi8(v0, v2); // a0 a4 a8 a12 ...
    u1 = _mm_unpacklo_epi8(v1, v3); // a2 a6 a10 a14 ...
    u2 = _mm_unpackhi_epi8(v0, v2); // a1 a5 a9 a13 ...
    u3 = _mm_unpackhi_epi8(v1, v3); // a3 a7 a11 a15 ...

    v0 = _mm_unpacklo_epi8(u0, u1); // a0 a2 a4 a6 ...
    v1 = _mm_unpacklo_epi8(u2, u3); // a1 a3 a5 a7 ...
    v2 = _mm_unpackhi_epi8(u0, u1); // c0 c2 c4 c6 ...
    v3 = _mm_unpackhi_epi8(u2, u3); // c1 c3 c5 c7 ...

    a = _mm_unpacklo_epi8(v0, v1);
    b = _mm_unpackhi_epi8(v0, v1);
    c = _mm_unpacklo_epi8(v2, v3);
    d = _mm_unpackhi_epi8(v2, v3);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b)
{
    __m128i v0 = _mm_loadu_si128((__m128i*)(ptr));     // a0 b0 a1 b1 a2 b2 a3 b3
    __m128i v1 = _mm_loadu_si128((__m128i*)(ptr + 8)); // a4 b4 a5 b5 a6 b6 a7 b7

    __m128i v2 = _mm_unpacklo_epi16(v0, v1); // a0 a4 b0 b4 a1 a5 b1 b5
    __m128i v3 = _mm_unpackhi_epi16(v0, v1); // a2 a6 b2 b6 a3 a7 b3 b7
    __m128i v4 = _mm_unpacklo_epi16(v2, v3); // a0 a2 a4 a6 b0 b2 b4 b6
    __m128i v5 = _mm_unpackhi_epi16(v2, v3); // a1 a3 a5 a7 b1 b3 b5 b7

    a = _mm_unpacklo_epi16(v4, v5); // a0 a1 a2 a3 a4 a5 a6 a7
    b = _mm_unpackhi_epi16(v4, v5); // b0 b1 ab b3 b4 b5 b6 b7
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c)
{
#if CV_SSE4_1
    __m128i v0 = _mm_loadu_si128((__m128i*)(ptr));
    __m128i v1 = _mm_loadu_si128((__m128i*)(ptr + 8));
    __m128i v2 = _mm_loadu_si128((__m128i*)(ptr + 16));
    __m128i a0 = _mm_blend_epi16(_mm_blend_epi16(v0, v1, 0x92), v2, 0x24);
    __m128i b0 = _mm_blend_epi16(_mm_blend_epi16(v2, v0, 0x92), v1, 0x24);
    __m128i c0 = _mm_blend_epi16(_mm_blend_epi16(v1, v2, 0x92), v0, 0x24);

    const __m128i sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m128i sh_b = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    const __m128i sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    a0 = _mm_shuffle_epi8(a0, sh_a);
    b0 = _mm_shuffle_epi8(b0, sh_b);
    c0 = _mm_shuffle_epi8(c0, sh_c);

    a = a0;
    b = b0;
    c = c0;
#else
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 8));
    __m128i t02 = _mm_loadu_si128((const __m128i*)(ptr + 16));

    __m128i t10 = _mm_unpacklo_epi16(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi16(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi16(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi16(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi16(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi16(t11, _mm_unpackhi_epi64(t12, t12));

    a = _mm_unpacklo_epi16(t20, _mm_unpackhi_epi64(t21, t21));
    b = _mm_unpacklo_epi16(_mm_unpackhi_epi64(t20, t20), t22);
    c = _mm_unpacklo_epi16(t21, _mm_unpackhi_epi64(t22, t22));
#endif
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c, v_uint16x8& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 8)); // a2 b2 c2 d2 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 24)); // a6 b6 c6 d6 ...

    __m128i v0 = _mm_unpacklo_epi16(u0, u2); // a0 a4 b0 b4 ...
    __m128i v1 = _mm_unpackhi_epi16(u0, u2); // a1 a5 b1 b5 ...
    __m128i v2 = _mm_unpacklo_epi16(u1, u3); // a2 a6 b2 b6 ...
    __m128i v3 = _mm_unpackhi_epi16(u1, u3); // a3 a7 b3 b7 ...

    u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    a = _mm_unpacklo_epi16(u0, u1);
    b = _mm_unpackhi_epi16(u0, u1);
    c = _mm_unpacklo_epi16(u2, u3);
    d = _mm_unpackhi_epi16(u2, u3);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b)
{
    __m128i v0 = _mm_loadu_si128((__m128i*)(ptr));     // a0 b0 a1 b1
    __m128i v1 = _mm_loadu_si128((__m128i*)(ptr + 4)); // a2 b2 a3 b3

    __m128i v2 = _mm_unpacklo_epi32(v0, v1); // a0 a2 b0 b2
    __m128i v3 = _mm_unpackhi_epi32(v0, v1); // a1 a3 b1 b3

    a = _mm_unpacklo_epi32(v2, v3); // a0 a1 a2 a3
    b = _mm_unpackhi_epi32(v2, v3); // b0 b1 ab b3
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c)
{
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 4));
    __m128i t02 = _mm_loadu_si128((const __m128i*)(ptr + 8));

    __m128i t10 = _mm_unpacklo_epi32(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi32(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi32(t01, _mm_unpackhi_epi64(t02, t02));

    a = _mm_unpacklo_epi32(t10, _mm_unpackhi_epi64(t11, t11));
    b = _mm_unpacklo_epi32(_mm_unpackhi_epi64(t10, t10), t12);
    c = _mm_unpacklo_epi32(t11, _mm_unpackhi_epi64(t12, t12));
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c, v_uint32x4& d)
{
    v_uint32x4 s0(_mm_loadu_si128((const __m128i*)ptr));        // a0 b0 c0 d0
    v_uint32x4 s1(_mm_loadu_si128((const __m128i*)(ptr + 4)));  // a1 b1 c1 d1
    v_uint32x4 s2(_mm_loadu_si128((const __m128i*)(ptr + 8)));  // a2 b2 c2 d2
    v_uint32x4 s3(_mm_loadu_si128((const __m128i*)(ptr + 12))); // a3 b3 c3 d3

    v_transpose4x4(s0, s1, s2, s3, a, b, c, d);
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b)
{
    __m128 u0 = _mm_loadu_ps(ptr);       // a0 b0 a1 b1
    __m128 u1 = _mm_loadu_ps((ptr + 4)); // a2 b2 a3 b3

    a = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(2, 0, 2, 0)); // a0 a1 a2 a3
    b = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(3, 1, 3, 1)); // b0 b1 ab b3
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c)
{
    __m128 t0 = _mm_loadu_ps(ptr + 0);
    __m128 t1 = _mm_loadu_ps(ptr + 4);
    __m128 t2 = _mm_loadu_ps(ptr + 8);

    __m128 at12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 1, 0, 2));
    a = _mm_shuffle_ps(t0, at12, _MM_SHUFFLE(2, 0, 3, 0));

    __m128 bt01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 bt12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 2, 0, 3));
    b = _mm_shuffle_ps(bt01, bt12, _MM_SHUFFLE(2, 0, 2, 0));

    __m128 ct01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 1, 0, 2));
    c = _mm_shuffle_ps(ct01, t2, _MM_SHUFFLE(3, 0, 2, 0));
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c, v_float32x4& d)
{
    __m128 t0 = _mm_loadu_ps(ptr +  0);
    __m128 t1 = _mm_loadu_ps(ptr +  4);
    __m128 t2 = _mm_loadu_ps(ptr +  8);
    __m128 t3 = _mm_loadu_ps(ptr + 12);
    __m128 t02lo = _mm_unpacklo_ps(t0, t2);
    __m128 t13lo = _mm_unpacklo_ps(t1, t3);
    __m128 t02hi = _mm_unpackhi_ps(t0, t2);
    __m128 t13hi = _mm_unpackhi_ps(t1, t3);
    a = _mm_unpacklo_ps(t02lo, t13lo);
    b = _mm_unpackhi_ps(t02lo, t13lo);
    c = _mm_unpacklo_ps(t02hi, t13hi);
    d = _mm_unpackhi_ps(t02hi, t13hi);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b)
{
    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 2));

    a = _mm_unpacklo_epi64(t0, t1);
    b = _mm_unpackhi_epi64(t0, t1);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b, v_uint64x2& c)
{
    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr); // a0, b0
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 2)); // c0, a1
    __m128i t2 = _mm_loadu_si128((const __m128i*)(ptr + 4)); // b1, c1

    t1 = _mm_shuffle_epi32(t1, 0x4e); // a1, c0

    a = _mm_unpacklo_epi64(t0, t1);
    b = _mm_unpacklo_epi64(_mm_unpackhi_epi64(t0, t0), t2);
    c = _mm_unpackhi_epi64(t1, t2);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a,
                                v_uint64x2& b, v_uint64x2& c, v_uint64x2& d)
{
    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 2)); // c0 d0
    __m128i t2 = _mm_loadu_si128((const __m128i*)(ptr + 4)); // a1 b1
    __m128i t3 = _mm_loadu_si128((const __m128i*)(ptr + 6)); // c1 d1

    a = _mm_unpacklo_epi64(t0, t2);
    b = _mm_unpackhi_epi64(t0, t2);
    c = _mm_unpacklo_epi64(t1, t3);
    d = _mm_unpackhi_epi64(t1, t3);
}


#define OPENCV_HAL_IMPL_SSE_SIGNED_DEINTERLEAVE(_Tvec0, _Tp0, _Tvec1, _Tp1)       \
    inline void v_load_deinterleave(const _Tp0* ptr, _Tvec0& a0, _Tvec0& b0)      \
    {                                                                             \
        _Tvec1 a1, b1;                                                            \
        v_load_deinterleave((const _Tp1*)ptr, a1, b1);                            \
        a0 = _Tvec0::cast(a1);                                                    \
        b0 = _Tvec0::cast(b1);                                                    \
    }                                                                             \
    inline void v_load_deinterleave(const _Tp0* ptr, _Tvec0& a0, _Tvec0& b0,      \
                                                                 _Tvec0& c0)      \
    {                                                                             \
        _Tvec1 a1, b1, c1;                                                        \
        v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1);                        \
        a0 = _Tvec0::cast(a1);                                                    \
        b0 = _Tvec0::cast(b1);                                                    \
        c0 = _Tvec0::cast(c1);                                                    \
    }                                                                             \
    inline void v_load_deinterleave(const _Tp0* ptr, _Tvec0& a0, _Tvec0& b0,      \
                                                     _Tvec0& c0, _Tvec0& d0)      \
    {                                                                             \
        _Tvec1 a1, b1, c1, d1;                                                    \
        v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1, d1);                    \
        a0 = _Tvec0::cast(a1);                                                    \
        b0 = _Tvec0::cast(b1);                                                    \
        c0 = _Tvec0::cast(c1);                                                    \
        d0 = _Tvec0::cast(d1);                                                    \
    }

OPENCV_HAL_IMPL_SSE_SIGNED_DEINTERLEAVE(v_int8x16,   schar,  v_uint8x16, uchar)
OPENCV_HAL_IMPL_SSE_SIGNED_DEINTERLEAVE(v_int16x8,   short,  v_uint16x8, ushort)
OPENCV_HAL_IMPL_SSE_SIGNED_DEINTERLEAVE(v_int32x4,   int,    v_uint32x4, unsigned)
OPENCV_HAL_IMPL_SSE_SIGNED_DEINTERLEAVE(v_int64x2,   int64,  v_uint64x2, uint64)
OPENCV_HAL_IMPL_SSE_SIGNED_DEINTERLEAVE(v_float64x2, double, v_uint64x2, uint64)