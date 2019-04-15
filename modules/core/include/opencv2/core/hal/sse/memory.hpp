// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV_SIMD128
    #error "Not a standalone header"
#endif

//// Load ////

template<typename Tp>
inline typename V128_Traits<Tp>::v v_load(const Tp* ptr)
{ return _mm_loadu_si128((const __m128i*)ptr); }
inline v_float32x4 v_load(const float* ptr)
{ return _mm_loadu_ps(ptr); }
inline v_float64x2 v_load(const double* ptr)
{ return _mm_loadu_pd(ptr); }

template<typename Tp>
inline typename V128_Traits<Tp>::v v_load_aligned(const Tp* ptr)
{ return _mm_load_si128((const __m128i*)ptr); }
inline v_float32x4 v_load_aligned(const float* ptr)
{ return _mm_load_ps(ptr); }
inline v_float64x2 v_load_aligned(const double* ptr)
{ return _mm_load_pd(ptr); }

template<typename Tp>
inline typename V128_Traits<Tp>::v v_load_low(const Tp* ptr)
{ return V128_Traits<Tp>::v::cast(_mm_loadl_epi64((const __m128i*)ptr)); }

template<typename Tp>
inline typename V128_Traits<Tp>::v v_load_halves(const Tp* ptr0, const Tp* ptr1)
{
    return V128_Traits<Tp>::v::cast(_mm_unpacklo_epi64(
        _mm_loadl_epi64((const __m128i*)ptr0),
        _mm_loadl_epi64((const __m128i*)ptr1))
    );
}

//// Store ////

template<typename Tp>
inline void v_store(Tp* ptr, const __m128i& a)
{ _mm_storeu_si128((__m128i*)ptr, a); }
inline void v_store(float* ptr, const v_float32x4& a)
{ _mm_storeu_ps(ptr, a); }
inline void v_store(double* ptr, const v_float64x2& a)
{ _mm_storeu_pd(ptr, a); }

template<typename Tp>
inline void v_store_aligned(Tp* ptr, const __m128i& a)
{ _mm_store_si128((__m128i*)ptr, a); }
inline void v_store_aligned(float* ptr, const v_float32x4& a)
{ _mm_store_ps(ptr, a); }
inline void v_store_aligned(double* ptr, const v_float64x2& a)
{ _mm_store_pd(ptr, a); }

template<typename Tp>
inline void v_store_aligned_nocache(Tp* ptr, const __m128i& a)
{ _mm_stream_si128((__m128i*)ptr, a); }
inline void v_store_aligned_nocache(float* ptr, const v_float32x4& a)
{ _mm_stream_ps(ptr, a); }
inline void v_store_aligned_nocache(double* ptr, const v_float64x2& a)
{ _mm_stream_pd(ptr, a); }

template<typename Tp, typename Tpvec>
inline void v_store(Tp* ptr, const Tpvec& a, hal::StoreMode mode)
{
    switch(mode)
    {
    case hal::STORE_ALIGNED_NOCACHE:
        v_store_aligned_nocache(ptr, a);
        break;
    case hal::STORE_ALIGNED:
        v_store_aligned(ptr, a);
        break;
    default:
        v_store(ptr, a);
    }
}

template<typename Tp>
inline void v_store_low(Tp* ptr, const __m128i& a)
{ _mm_storel_epi64((__m128i*)ptr, a); }
inline void v_store_low(float* ptr, const v_float32x4& a)
{ _mm_storel_pi((__m64*)ptr, a); }
inline void v_store_low(double* ptr, const v_float64x2& a)
{ _mm_storel_pd(ptr, a); }

template<typename Tp>
inline void v_store_high(Tp* ptr, const __m128i& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a, a)); }
inline void v_store_high(float* ptr, const v_float32x4& a)
{
    __m128i ai = _mm_castps_si128(a);
    _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(ai, ai));
}
inline void v_store_high(double* ptr, const v_float64x2& a)
{ _mm_storel_pd(ptr, _mm_unpackhi_pd(a, a)); }


//// Half-precision ////

#if CV_FP16
inline v_float32x4 v128_load_fp16_f32(const short* ptr)
{ return _mm_cvtph_ps(_mm_loadu_si128((const __m128i*)ptr)); }

inline void v_store_fp16(short* ptr, const v_float32x4& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_cvtps_ph(a, 0)); }
#endif

//// Lookup table access ////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
#if defined(_MSC_VER)
    return _mm_setr_epi8(
        tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
        tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]
    );
#else
    return _mm_setr_epi64(
        _mm_setr_pi8(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]]),
        _mm_setr_pi8(tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]])
    );
#endif
}

inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
#if defined(_MSC_VER)
    return _mm_setr_epi16(
        *(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]), *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3]),
        *(const short*)(tab + idx[4]), *(const short*)(tab + idx[5]), *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7])
    );
#else
    return _mm_setr_epi64(
        _mm_setr_pi16(
            *(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]),
            *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3])
        ),
        _mm_setr_pi16(
            *(const short*)(tab + idx[4]), *(const short*)(tab + idx[5]),
            *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7])
        )
    );
#endif
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
#if defined(_MSC_VER)
    return _mm_setr_epi32(
        *(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
        *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])
    );
#else
    return _mm_setr_epi64(
        _mm_setr_pi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1])),
        _mm_setr_pi32(*(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]))
    );
#endif
}
inline v_uint8x16 v_lut(const uchar* tab, const int* idx)
{ return v_uint8x16(v_lut((const schar *)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx)
{ return v_uint8x16(v_lut_pairs((const schar *)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx)
{ return v_uint8x16(v_lut_quads((const schar *)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
#if defined(_MSC_VER)
    return _mm_setr_epi16(
        tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
        tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]
    );
#else
    return _mm_setr_epi64(
        _mm_setr_pi16(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]),
        _mm_setr_pi16(tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]])
    );
#endif
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
#if defined(_MSC_VER)
    return _mm_setr_epi32(
        *(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
        *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])
    );
#else
    return _mm_setr_epi64(
        _mm_setr_pi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1])),
        _mm_setr_pi32(*(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]))
    );
#endif
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    return _mm_set_epi64x(*(const int64_t*)(tab + idx[1]), *(const int64_t*)(tab + idx[0]));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx)
{ return v_uint16x8(v_lut((const short *)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx)
{ return v_uint16x8(v_lut_pairs((const short *)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx)
{ return v_uint16x8(v_lut_quads((const short *)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
#if defined(_MSC_VER)
    return _mm_setr_epi32(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
#else
    return _mm_setr_epi64(
        _mm_setr_pi32(tab[idx[0]], tab[idx[1]]),
        _mm_setr_pi32(tab[idx[2]], tab[idx[3]])
    );
#endif
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    return _mm_set_epi64x(*(const int64_t*)(tab + idx[1]), *(const int64_t*)(tab + idx[0]));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return _mm_loadu_si128((const __m128i*)(tab + idx[0]));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx)
{ return v_uint32x4(v_lut((const int *)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx)
{ return v_uint32x4(v_lut_pairs((const int *)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx)
{ return v_uint32x4(v_lut_quads((const int *)tab, idx)); }

inline v_int64x2 v_lut(const int64* tab, const int* idx)
{
    return _mm_set_epi64x(tab[idx[1]], tab[idx[0]]);
}
inline v_int64x2 v_lut_pairs(const int64* tab, const int* idx)
{
    return _mm_loadu_si128((const __m128i*)(tab + idx[0]));
}
inline v_uint64x2 v_lut(const uint64* tab, const int* idx)
{ return v_uint64x2(v_lut((const int64 *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64* tab, const int* idx)
{ return v_uint64x2(v_lut_pairs((const int64 *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    return _mm_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx)
{ return v_float32x4::cast(v_lut_pairs((const int *)tab, idx)); }
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{ return v_float32x4::cast(v_lut_quads((const int *)tab, idx)); }

inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    return _mm_setr_pd(tab[idx[0]], tab[idx[1]]);
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{ return _mm_castsi128_pd(_mm_loadu_si128((const __m128i*)(tab + idx[0]))); }

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(16) idx[4];
    v_store_aligned(idx, idxvec);
    return _mm_setr_epi32(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}
inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{ return v_uint32x4(v_lut((const int *)tab, idxvec)); }

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(16) idx[4];
    v_store_aligned(idx, idxvec);
    return _mm_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]);
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    int idx[2];
    v_store_low(idx, idxvec);
    return _mm_setr_pd(tab[idx[0]], tab[idx[1]]);
}


// loads pairs from the table and deinterleaves them, e.g. returns:
//   x = (tab[idxvec[0], tab[idxvec[1]], tab[idxvec[2]], tab[idxvec[3]]),
//   y = (tab[idxvec[0]+1], tab[idxvec[1]+1], tab[idxvec[2]+1], tab[idxvec[3]+1])
// note that the indices are float's indices, not the float-pair indices.
// in theory, this function can be used to implement bilinear interpolation,
// when idxvec are the offsets within the image.
inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    int CV_DECL_ALIGNED(16) idx[4];
    v_store_aligned(idx, idxvec);
    __m128 z = _mm_setzero_ps();
    __m128 xy01 = _mm_loadl_pi(z, (__m64*)(tab + idx[0]));
    __m128 xy23 = _mm_loadl_pi(z, (__m64*)(tab + idx[2]));
    xy01 = _mm_loadh_pi(xy01, (__m64*)(tab + idx[1]));
    xy23 = _mm_loadh_pi(xy23, (__m64*)(tab + idx[3]));
    __m128 xxyy02 = _mm_unpacklo_ps(xy01, xy23);
    __m128 xxyy13 = _mm_unpackhi_ps(xy01, xy23);
    x = _mm_unpacklo_ps(xxyy02, xxyy13);
    y = _mm_unpackhi_ps(xxyy02, xxyy13);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int idx[2];
    v_store_low(idx, idxvec);
    __m128d xy0 = _mm_loadu_pd(tab + idx[0]);
    __m128d xy1 = _mm_loadu_pd(tab + idx[1]);
    x = _mm_unpacklo_pd(xy0, xy1);
    y = _mm_unpackhi_pd(xy0, xy1);
}