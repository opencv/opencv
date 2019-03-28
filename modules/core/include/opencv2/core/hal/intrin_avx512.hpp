// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_INTRIN_AVX512_HPP
#define OPENCV_HAL_INTRIN_AVX512_HPP

#define CV_SIMD512 1
#define CV_SIMD512_64F 1
#define CV_SIMD512_FP16 0  // no native operations with FP16 type. Only load/store from float32x8 are available (if CV_FP16 == 1)

///////// Utils ////////////

namespace
{

inline __m512i _v512_combine(const __m256i& lo, const __m256i& hi)
{ return _mm512_inserti32x8(_mm512_castsi256_si512(lo), hi, 1); }

inline __m512 _v512_combine(const __m256& lo, const __m256& hi)
{ return _mm512_insertf32x8(_mm512_castps256_ps512(lo), hi, 1); }

inline __m512d _v512_combine(const __m256d& lo, const __m256d& hi)
{ return _mm512_insertf64x4(_mm512_castpd256_pd512(lo), hi, 1); }

inline int _v_cvtsi512_si32(const __m512i& a)
{ return _mm_cvtsi128_si32(_mm512_castsi512_si128(a)); }

inline __m256i _v512_extract_high(const __m512i& v)
{ return _mm512_extracti32x8_epi32(v, 1); }

inline __m256  _v512_extract_high(const __m512& v)
{ return _mm512_extractf32x8_ps(v, 1); }

inline __m256d _v512_extract_high(const __m512d& v)
{ return _mm512_extractf64x4_pd(v, 1); }

inline __m256i _v512_extract_low(const __m512i& v)
{ return _mm512_castsi512_si256(v); }

inline __m256  _v512_extract_low(const __m512& v)
{ return _mm512_castps512_ps256(v); }

inline __m256d _v512_extract_low(const __m512d& v)
{ return _mm512_castpd512_pd256(v); }

inline __m512i _v512_insert(const __m512i& a, const __m256i& b)
{ return _mm512_inserti32x8(a, b, 0); }

inline __m512 _v512_insert(const __m512& a, const __m256& b)
{ return _mm512_insertf32x8(a, b, 0); }

inline __m512d _v512_insert(const __m512d& a, const __m256d& b)
{ return _mm512_insertf64x4(a, b, 0); }

}
/*inline __m512i _v512_shuffle_odd_64(const __m512i& v)
{ return _mm512_permutex_epi64(v, _MM_SHUFFLE(3, 1, 2, 0)); }

inline __m512d _v512_shuffle_odd_64(const __m512d& v)
{ return _mm512_permutex_pd(v, _MM_SHUFFLE(3, 1, 2, 0)); }

template<int imm, typename _Tpvec>
inline _Tpvec v512_permute2x128(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_v512_permute2x128<imm>(a.val, b.val)); }

template<int imm>
inline __m512i _v512_permute4x64(const __m512i& a)
{ return _mm256_permute4x64_epi64(a, imm); }

template<int imm>
inline __m512d _v512_permute4x64(const __m512d& a)
{ return _mm256_permute4x64_pd(a, imm); }

template<int imm, typename _Tpvec>
inline _Tpvec v512_permute4x64(const _Tpvec& a)
{ return _Tpvec(_v512_permute4x64<imm>(a.val)); }

inline __m512i _v512_packs_epu32(const __m512i& a, const __m512i& b)
{
    const __m512i m = _mm256_set1_epi32(65535);
    __m512i am = _mm256_min_epu32(a, m);
    __m512i bm = _mm256_min_epu32(b, m);
    return _mm256_packus_epi32(am, bm);
}

//////////////// Variant Value reordering ///////////////

// shuffle
// todo: emluate 64bit
#define OPENCV_HAL_IMPL_AVX_SHUFFLE(_Tpvec, intrin)  \
template<int m>                                  \
inline _Tpvec v512_shuffle(const _Tpvec& a)      \
{ return _Tpvec(_mm256_##intrin(a.val, m)); }

OPENCV_HAL_IMPL_AVX_SHUFFLE(v_uint32x8,  shuffle_epi32)
OPENCV_HAL_IMPL_AVX_SHUFFLE(v_int32x8,   shuffle_epi32)
OPENCV_HAL_IMPL_AVX_SHUFFLE(v_float32x8, permute_ps)
OPENCV_HAL_IMPL_AVX_SHUFFLE(v_float64x4, permute_pd)

template<typename _Tpvec>
inline _Tpvec v512_alignr_128(const _Tpvec& a, const _Tpvec& b)
{ return v512_permute2x128<0x21>(a, b); }

template<typename _Tpvec>
inline _Tpvec v512_alignr_64(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_mm256_alignr_epi8(a.val, b.val, 8)); }
inline v_float64x4 v512_alignr_64(const v_float64x4& a, const v_float64x4& b)
{ return v_float64x4(_mm256_shuffle_pd(b.val, a.val, _MM_SHUFFLE(0, 0, 1, 1))); }
// todo: emulate float32

template<typename _Tpvec>
inline _Tpvec v512_swap_halves(const _Tpvec& a)
{ return v512_permute2x128<1>(a, a); }

template<typename _Tpvec>
inline _Tpvec v512_reverse_64(const _Tpvec& a)
{ return v512_permute4x64<_MM_SHUFFLE(0, 1, 2, 3)>(a); }

*/

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Types ////////////

struct v_uint8x64
{
    typedef uchar lane_type;
    enum { nlanes = 64 };
    __m512i val;

    explicit v_uint8x64(__m512i v) : val(v) {}
    v_uint8x64(uchar v0,  uchar v1,  uchar v2,  uchar v3,
               uchar v4,  uchar v5,  uchar v6,  uchar v7,
               uchar v8,  uchar v9,  uchar v10, uchar v11,
               uchar v12, uchar v13, uchar v14, uchar v15,
               uchar v16, uchar v17, uchar v18, uchar v19,
               uchar v20, uchar v21, uchar v22, uchar v23,
               uchar v24, uchar v25, uchar v26, uchar v27,
               uchar v28, uchar v29, uchar v30, uchar v31,
               uchar v32, uchar v33, uchar v34, uchar v35,
               uchar v36, uchar v37, uchar v38, uchar v39,
               uchar v40, uchar v41, uchar v42, uchar v43,
               uchar v44, uchar v45, uchar v46, uchar v47,
               uchar v48, uchar v49, uchar v50, uchar v51,
               uchar v52, uchar v53, uchar v54, uchar v55,
               uchar v56, uchar v57, uchar v58, uchar v59,
               uchar v60, uchar v61, uchar v62, uchar v63)
    {
        val = _mm512_set_epi8((char)v63, (char)v62, (char)v61, (char)v60, (char)v59, (char)v58, (char)v57, (char)v56,
                              (char)v55, (char)v54, (char)v53, (char)v52, (char)v51, (char)v50, (char)v49, (char)v48,
                              (char)v47, (char)v46, (char)v45, (char)v44, (char)v43, (char)v42, (char)v41, (char)v40,
                              (char)v39, (char)v38, (char)v37, (char)v36, (char)v35, (char)v34, (char)v33, (char)v32,
                              (char)v31, (char)v30, (char)v29, (char)v28, (char)v27, (char)v26, (char)v25, (char)v24,
                              (char)v23, (char)v22, (char)v21, (char)v20, (char)v19, (char)v18, (char)v17, (char)v16,
                              (char)v15, (char)v14, (char)v13, (char)v12, (char)v11, (char)v10, (char)v9,  (char)v8,
                              (char)v7,  (char)v6,  (char)v5,  (char)v4,  (char)v3,  (char)v2,  (char)v1,  (char)v0);
    }
    v_uint8x64() : val(_mm512_setzero_si512()) {}
    uchar get0() const { return (uchar)_v_cvtsi512_si32(val); }
};

struct v_int8x64
{
    typedef schar lane_type;
    enum { nlanes = 64 };
    __m512i val;

    explicit v_int8x64(__m512i v) : val(v) {}
    v_int8x64(schar v0,  schar v1,  schar v2,  schar v3,
              schar v4,  schar v5,  schar v6,  schar v7,
              schar v8,  schar v9,  schar v10, schar v11,
              schar v12, schar v13, schar v14, schar v15,
              schar v16, schar v17, schar v18, schar v19,
              schar v20, schar v21, schar v22, schar v23,
              schar v24, schar v25, schar v26, schar v27,
              schar v28, schar v29, schar v30, schar v31,
              schar v32, schar v33, schar v34, schar v35,
              schar v36, schar v37, schar v38, schar v39,
              schar v40, schar v41, schar v42, schar v43,
              schar v44, schar v45, schar v46, schar v47,
              schar v48, schar v49, schar v50, schar v51,
              schar v52, schar v53, schar v54, schar v55,
              schar v56, schar v57, schar v58, schar v59,
              schar v60, schar v61, schar v62, schar v63)
    {
        val = _mm512_set_epi8(v63, v62, v61, v60, v59, v58, v57, v56, v55, v54, v53, v52, v51, v50, v49, v48,
                              v47, v46, v45, v44, v43, v42, v41, v40, v39, v38, v37, v36, v35, v34, v33, v32,
                              v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16,
                              v15, v14, v13, v12, v11, v10, v9,  v8,  v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
    }
    v_int8x64() : val(_mm512_setzero_si512()) {}
    schar get0() const { return (schar)_v_cvtsi512_si32(val); }
};

struct v_uint16x32
{
    typedef ushort lane_type;
    enum { nlanes = 32 };
    __m512i val;

    explicit v_uint16x32(__m512i v) : val(v) {}
    v_uint16x32(ushort v0,  ushort v1,  ushort v2,  ushort v3,
                ushort v4,  ushort v5,  ushort v6,  ushort v7,
                ushort v8,  ushort v9,  ushort v10, ushort v11,
                ushort v12, ushort v13, ushort v14, ushort v15,
                ushort v16, ushort v17, ushort v18, ushort v19,
                ushort v20, ushort v21, ushort v22, ushort v23,
                ushort v24, ushort v25, ushort v26, ushort v27,
                ushort v28, ushort v29, ushort v30, ushort v31)
    {
        val = _mm512_set_epi16((short)v31, (short)v30, (short)v29, (short)v28, (short)v27, (short)v26, (short)v25, (short)v24,
                               (short)v23, (short)v22, (short)v21, (short)v20, (short)v19, (short)v18, (short)v17, (short)v16,
                               (short)v15, (short)v14, (short)v13, (short)v12, (short)v11, (short)v10, (short)v9,  (short)v8,
                               (short)v7,  (short)v6,  (short)v5,  (short)v4,  (short)v3,  (short)v2,  (short)v1,  (short)v0);
    }
    v_uint16x32() : val(_mm512_setzero_si512()) {}
    ushort get0() const { return (ushort)_v_cvtsi256_si32(val); }
};

struct v_int16x32
{
    typedef short lane_type;
    enum { nlanes = 32 };
    __m512i val;

    explicit v_int16x32(__m512i v) : val(v) {}
    v_int16x32(short v0,  short v1,  short v2,  short v3,  short v4,  short v5,  short v6,  short v7,
               short v8,  short v9,  short v10, short v11, short v12, short v13, short v14, short v15,
               short v16, short v17, short v18, short v19, short v20, short v21, short v22, short v23,
               short v24, short v25, short v26, short v27, short v28, short v29, short v30, short v31)
    {
        val = _mm512_set_epi16(v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16,
                               v15, v14, v13, v12, v11, v10, v9,  v8,  v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
    }
    v_int16x32() : val(_mm512_setzero_si512()) {}
    short get0() const { return (short)_v_cvtsi512_si32(val); }
};

struct v_uint32x16
{
    typedef unsigned lane_type;
    enum { nlanes = 16 };
    __m512i val;

    explicit v_uint32x16(__m512i v) : val(v) {}
    v_uint32x16(unsigned v0,  unsigned v1,  unsigned v2,  unsigned v3,
                unsigned v4,  unsigned v5,  unsigned v6,  unsigned v7,
                unsigned v8,  unsigned v9,  unsigned v10, unsigned v11,
                unsigned v12, unsigned v13, unsigned v14, unsigned v15)
    {
        val = _mm512_setr_epi32((unsigned)v0,  (unsigned)v1,  (unsigned)v2,  (unsigned)v3,
                                (unsigned)v4,  (unsigned)v5,  (unsigned)v6,  (unsigned)v7,
                                (unsigned)v8,  (unsigned)v9,  (unsigned)v10, (unsigned)v11,
                                (unsigned)v12, (unsigned)v13, (unsigned)v14, (unsigned)v15);
    }
    v_uint32x16() : val(_mm512_setzero_si512()) {}
    unsigned get0() const { return (unsigned)_v_cvtsi512_si32(val); }
};

struct v_int32x16
{
    typedef int lane_type;
    enum { nlanes = 16 };
    __m512i val;

    explicit v_int32x16(__m512i v) : val(v) {}
    v_int32x16(int v0, int v1, int v2,  int v3,  int v4,  int v5,  int v6,  int v7,
               int v8, int v9, int v10, int v11, int v12, int v13, int v14, int v15)
    {
        val = _mm512_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }
    v_int32x16() : val(_mm512_setzero_si512()) {}
    int get0() const { return _v_cvtsi512_si32(val); }
};

struct v_float32x16
{
    typedef float lane_type;
    enum { nlanes = 16 };
    __m512 val;

    explicit v_float32x16(__m512 v) : val(v) {}
    v_float32x16(float v0, float v1, float v2,  float v3,  float v4,  float v5,  float v6,  float v7,
                 float v8, float v9, float v10, float v11, float v12, float v13, float v14, float v15)
    {
        val = _mm512_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }
    v_float32x16() : val(_mm512_setzero_ps()) {}
    float get0() const { return _mm_cvtss_f32(_mm512_castps512_ps128(val)); }
};

struct v_uint64x8
{
    typedef uint64 lane_type;
    enum { nlanes = 8 };
    __m512i val;

    explicit v_uint64x8(__m512i v) : val(v) {}
    v_uint64x8(uint64 v0, uint64 v1, uint64 v2, uint64 v3, uint64 v4, uint64 v5, uint64 v6, uint64 v7)
    { val = _mm512_setr_epi64((int64)v0, (int64)v1, (int64)v2, (int64)v3, (int64)v4, (int64)v5, (int64)v6, (int64)v7); }
    v_uint64x8() : val(_mm512_setzero_si512()) {}
    uint64 get0() const
    {
    #if defined __x86_64__ || defined _M_X64
        return (uint64)_mm_cvtsi128_si64(_mm512_castsi512_si128(val));
    #else
        int a = _mm_cvtsi128_si32(_mm512_castsi512_si128(val));
        int b = _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_srli_epi64(val, 32)));
        return (unsigned)a | ((uint64)(unsigned)b << 32);
    #endif
    }
};

struct v_int64x8
{
    typedef int64 lane_type;
    enum { nlanes = 8 };
    __m512i val;

    explicit v_int64x8(__m512i v) : val(v) {}
    v_int64x8(int64 v0, int64 v1, int64 v2, int64 v3, int64 v4, int64 v5, int64 v6, int64 v7)
    { val = _mm512_setr_epi64(v0, v1, v2, v3, v4, v5, v6, v7); }
    v_int64x8() : val(_mm512_setzero_si512()) {}

    int64 get0() const
    {
    #if defined __x86_64__ || defined _M_X64
        return (int64)_mm_cvtsi128_si64(_mm512_castsi512_si128(val));
    #else
        int a = _mm_cvtsi128_si32(_mm512_castsi512_si128(val));
        int b = _mm_cvtsi128_si32(_mm512_castsi512_si128(_mm512_srli_epi64(val, 32)));
        return (int64)((unsigned)a | ((uint64)(unsigned)b << 32));
    #endif
    }
};

struct v_float64x8
{
    typedef double lane_type;
    enum { nlanes = 8 };
    __m512d val;

    explicit v_float64x8(__m512d v) : val(v) {}
    v_float64x8(double v0, double v1, double v2, double v3, double v4, double v5, double v6, double v7)
    { val = _mm512_setr_pd(v0, v1, v2, v3, v4, v5, v6, v7); }
    v_float64x8() : val(_mm512_setzero_pd()) {}
    double get0() const { return _mm_cvtsd_f64(_mm512_castpd512_pd128(val)); }
};

//////////////// Load and store operations ///////////////

#define OPENCV_HAL_IMPL_AVX512_LOADSTORE(_Tpvec, _Tp)                    \
    inline _Tpvec v512_load(const _Tp* ptr)                           \
    { return _Tpvec(_mm512_loadu_si512((const __m512i*)ptr)); }       \
    inline _Tpvec v512_load_aligned(const _Tp* ptr)                   \
    { return _Tpvec(_mm512_load_si512((const __m512i*)ptr)); }        \
    inline _Tpvec v512_load_low(const _Tp* ptr)                       \
    {                                                                 \
        __m256i v256 = _mm256_loadu_si256((const __m256i*)ptr);       \
        return _Tpvec(_mm512_castsi256_si512(v256));                  \
    }                                                                 \
    inline _Tpvec v512_load_halves(const _Tp* ptr0, const _Tp* ptr1)  \
    {                                                                 \
        __m256i vlo = _mm256_loadu_si256((const __m256i*)ptr0);       \
        __m256i vhi = _mm256_loadu_si256((const __m256i*)ptr1);       \
        return _Tpvec(_v512_combine(vlo, vhi));                       \
    }                                                                 \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                    \
    { _mm512_storeu_si512((__m512i*)ptr, a.val); }                    \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)            \
    { _mm512_store_si512((__m512i*)ptr, a.val); }                     \
    inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a)    \
    { _mm512_stream_si512((__m512i*)ptr, a.val); }                    \
    inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
    { \
        if( mode == hal::STORE_UNALIGNED ) \
            _mm512_storeu_si512((__m512i*)ptr, a.val); \
        else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
            _mm512_stream_si512((__m512i*)ptr, a.val); \
        else \
            _mm512_store_si512((__m512i*)ptr, a.val); \
    } \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                \
    { _mm256_storeu_si256((__m256i*)ptr, _v512_extract_low(a.val)); }    \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)               \
    { _mm256_storeu_si256((__m256i*)ptr, _v512_extract_high(a.val)); }

OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_uint8x64,  uchar)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_int8x64,   schar)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_uint16x32, ushort)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_int16x32,  short)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_uint32x16,  unsigned)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_int32x16,   int)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_uint64x8,  uint64)
OPENCV_HAL_IMPL_AVX512_LOADSTORE(v_int64x8,   int64)

#define OPENCV_HAL_IMPL_AVX512_LOADSTORE_FLT(_Tpvec, _Tp, suffix, halfreg)   \
    inline _Tpvec v512_load(const _Tp* ptr)                               \
    { return _Tpvec(_mm512_loadu_##suffix(ptr)); }                        \
    inline _Tpvec v512_load_aligned(const _Tp* ptr)                       \
    { return _Tpvec(_mm512_load_##suffix(ptr)); }                         \
    inline _Tpvec v512_load_low(const _Tp* ptr)                           \
    {                                                                     \
        return _Tpvec(_mm512_cast##suffix##256_##suffix##512              \
                     (_mm256_loadu_##suffix(ptr)));                       \
    }                                                                     \
    inline _Tpvec v512_load_halves(const _Tp* ptr0, const _Tp* ptr1)      \
    {                                                                     \
        halfreg vlo = _mm256_loadu_##suffix(ptr0);                        \
        halfreg vhi = _mm256_loadu_##suffix(ptr1);                        \
        return _Tpvec(_v512_combine(vlo, vhi));                           \
    }                                                                     \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                        \
    { _mm512_storeu_##suffix(ptr, a.val); }                               \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)                \
    { _mm512_store_##suffix(ptr, a.val); }                                \
    inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a)        \
    { _mm512_stream_##suffix(ptr, a.val); }                               \
    inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
    { \
        if( mode == hal::STORE_UNALIGNED ) \
            _mm512_storeu_##suffix(ptr, a.val); \
        else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
            _mm512_stream_##suffix(ptr, a.val); \
        else \
            _mm512_store_##suffix(ptr, a.val); \
    } \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                    \
    { _mm256_storeu_##suffix(ptr, _v512_extract_low(a.val)); }            \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)                   \
    { _mm256_storeu_##suffix(ptr, _v512_extract_high(a.val)); }

OPENCV_HAL_IMPL_AVX512_LOADSTORE_FLT(v_float32x16, float,  ps, __m256)
OPENCV_HAL_IMPL_AVX512_LOADSTORE_FLT(v_float64x8, double, pd, __m256d)

#define OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, _Tpvecf, suffix, cast) \
    inline _Tpvec v_reinterpret_as_##suffix(const _Tpvecf& a)   \
    { return _Tpvec(cast(a.val)); }

#define OPENCV_HAL_IMPL_AVX512_INIT(_Tpvec, _Tp, suffix, ssuffix, ctype_s)         \
    inline _Tpvec v512_setzero_##suffix()                                          \
    { return _Tpvec(_mm512_setzero_si512()); }                                     \
    inline _Tpvec v512_setall_##suffix(_Tp v)                                      \
    { return _Tpvec(_mm512_set1_##ssuffix((ctype_s)v)); }                          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint8x64,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int8x64,    suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint16x32,  suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int16x32,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint32x16,  suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int32x16,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint64x8,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int64x8,    suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_float32x16, suffix, _mm256_castps_si256) \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_float64x8,  suffix, _mm256_castpd_si256)

OPENCV_HAL_IMPL_AVX512_INIT(v_uint8x64,  uchar,    u8,  epi8,   char)
OPENCV_HAL_IMPL_AVX512_INIT(v_int8x64,   schar,    s8,  epi8,   char)
OPENCV_HAL_IMPL_AVX512_INIT(v_uint16x32, ushort,   u16, epi16,  short)
OPENCV_HAL_IMPL_AVX512_INIT(v_int16x32,  short,    s16, epi16,  short)
OPENCV_HAL_IMPL_AVX512_INIT(v_uint32x16, unsigned, u32, epi32,  int)
OPENCV_HAL_IMPL_AVX512_INIT(v_int32x16,  int,      s32, epi32,  int)
OPENCV_HAL_IMPL_AVX512_INIT(v_uint64x8,  uint64,   u64, epi64,  int64)
OPENCV_HAL_IMPL_AVX512_INIT(v_int64x8,   int64,    s64, epi64,  int64)

#define OPENCV_HAL_IMPL_AVX512_INIT_FLT(_Tpvec, _Tp, suffix, zsuffix, cast) \
    inline _Tpvec v512_setzero_##suffix()                                   \
    { return _Tpvec(_mm512_setzero_##zsuffix()); }                          \
    inline _Tpvec v512_setall_##suffix(_Tp v)                               \
    { return _Tpvec(_mm512_set1_##zsuffix(v)); }                            \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint8x64,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int8x64,   suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint16x32, suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int16x32,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint32x16, suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int32x16,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint64x8,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int64x8,   suffix, cast)

OPENCV_HAL_IMPL_AVX512_INIT_FLT(v_float32x16, float,  f32, ps, _mm512_castsi512_ps)
OPENCV_HAL_IMPL_AVX512_INIT_FLT(v_float64x8,  double, f64, pd, _mm512_castsi512_pd)

inline v_float32x16 v_reinterpret_as_f32(const v_float32x16& a)
{ return a; }
inline v_float32x16 v_reinterpret_as_f32(const v_float64x8& a)
{ return v_float32x16(_mm512_castpd_ps(a.val)); }

inline v_float64x8 v_reinterpret_as_f64(const v_float64x8& a)
{ return a; }
inline v_float64x8 v_reinterpret_as_f64(const v_float32x16& a)
{ return v_float64x8(_mm512_castps_pd(a.val)); }

// FP16
inline v_float32x16 v512_load_expand(const float16_t* ptr)
{
    return v_float32x16(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)ptr)));
}

inline void v_pack_store(float16_t* ptr, const v_float32x16& a)
{
    __m256i ah = _mm512_cvtps_ph(a.val, 0);
    _mm256_storeu_si256((__m256i*)ptr, ah);
}

/* Recombine & ZIP */
#define OPENCV_HAL_IMPL_AVX512_ZIP(_Tpvec, suffix)                                         \
    inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)                          \
    { return _v512_combine(_v512_extract_low(a.val), _v512_extract_low(b.val)); }          \
    inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)                         \
    { return _v512_insert(b.val, _v512_extract_high(a.val)); }                             \
    inline void v_recombine(const _Tpvec& a, const _Tpvec& b,                              \
                                  _Tpvec& c, _Tpvec& d)                                    \
    {                                                                                      \
        _Tpvec a1b0 = _v512_combine(_v512_extract_high(a), _v512_extract_low(b));          \
        c = _v_512_combine(_v512_extract_low(a),_v512_extract_low(b));                     \
        d = _v_512_insert(b,_v512_extract_high(a));                                        \
    }                                                                                      \
    inline void v_zip(const _Tpvec& a, const _Tpvec& b,                                    \
                            _Tpvec& ab0, _Tpvec& ab1)                                      \
    {                                                                                      \
        ab0.val = _mm512_maskz_expand_##suffix(0x55555555, a);                             \
        ab1.val = _mm512_maskz_expand_##suffix(0x55555555 << _Tpvec::nlanes/2, a);         \
        ab0.val = _mm512_mask_expand_##suffix(ab0.val, 0xAAAAAAAA, b);                     \
        ab1.val = _mm512_mask_expand_##suffix(ab1.val, 0xAAAAAAAA << _Tpvec::nlanes/2, b); \
    }

OPENCV_HAL_IMPL_AVX512_ZIP(v_uint8x64, epi8)
OPENCV_HAL_IMPL_AVX512_ZIP(v_int8x64, epi8)
OPENCV_HAL_IMPL_AVX512_ZIP(v_uint16x32, epi16)
OPENCV_HAL_IMPL_AVX512_ZIP(v_int16x32, epi16)
OPENCV_HAL_IMPL_AVX512_ZIP(v_uint32x16, epi32)
OPENCV_HAL_IMPL_AVX512_ZIP(v_int32x16, epi32)
OPENCV_HAL_IMPL_AVX512_ZIP(v_uint64x8, epi64)
OPENCV_HAL_IMPL_AVX512_ZIP(v_int64x8, epi64)
OPENCV_HAL_IMPL_AVX512_ZIP(v_float32x16, ps)
OPENCV_HAL_IMPL_AVX512_ZIP(v_float64x8, pd)

////////// Arithmetic, bitwise and comparison operations /////////

/* Element-wise binary and unary operations */

/** Non-saturating arithmetics **/
#define OPENCV_HAL_IMPL_AVX512_BIN_FUNC(func, _Tpvec, intrin) \
    inline _Tpvec func(const _Tpvec& a, const _Tpvec& b)      \
    { return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_add_wrap, v_uint8x64, _mm512_add_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_add_wrap, v_int8x64, _mm512_add_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_add_wrap, v_uint16x32, _mm512_add_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_add_wrap, v_int16x32, _mm512_add_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_sub_wrap, v_uint8x64, _mm512_sub_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_sub_wrap, v_int8x64, _mm512_sub_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_sub_wrap, v_uint16x32, _mm512_sub_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_sub_wrap, v_int16x32, _mm512_sub_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_mul_wrap, v_uint16x32, _mm512_mullo_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_mul_wrap, v_int16x32, _mm512_mullo_epi16)

inline v_uint8x64 v_mul_wrap(const v_uint8x64& a, const v_uint8x64& b)
{
    __m512i ad = _mm512_srai_epi16(a.val, 8);
    __m512i bd = _mm512_srai_epi16(b.val, 8);
    __m512i p0 = _mm512_mullo_epi16(a.val, b.val); // even
    __m512i p1 = _mm512_slli_epi16(_mm512_mullo_epi16(ad, bd), 8); // odd
    return v_uint8x64(_mm512_mask_blend_epi8(0x5555555555555555, p0, p1));
}
inline v_int8x64 v_mul_wrap(const v_int8x64& a, const v_int8x64& b)
{
    return v_reinterpret_as_s8(v_mul_wrap(v_reinterpret_as_u8(a), v_reinterpret_as_u8(b)));
}

#define OPENCV_HAL_IMPL_AVX512_BIN_OP(bin_op, _Tpvec, intrin)            \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b)     \
    { return _Tpvec(intrin(a.val, b.val)); }                             \
    inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b)       \
    { a.val = intrin(a.val, b.val); return a; }

OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_uint32x16, _mm512_add_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_uint32x16, _mm512_sub_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_int32x16, _mm512_add_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_int32x16, _mm512_sub_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_uint64x8, _mm512_add_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_uint64x8, _mm512_sub_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_int64x8, _mm512_add_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_int64x8, _mm512_sub_epi64)

OPENCV_HAL_IMPL_AVX512_BIN_OP(*, v_uint32x16, _mm512_mullo_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(*, v_int32x16, _mm512_mullo_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(*, v_uint64x8, _mm512_mullo_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(*, v_int64x8, _mm512_mullo_epi64)

/** Saturating arithmetics **/
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_uint8x64,  _mm512_adds_epu8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_uint8x64,  _mm512_subs_epu8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_int8x64,   _mm512_adds_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_int8x64,   _mm512_subs_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_uint16x32, _mm512_adds_epu16)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_uint16x32, _mm512_subs_epu16)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_int16x32,  _mm512_adds_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_int16x32,  _mm512_subs_epi16)

OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_float32x16, _mm512_add_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_float32x16, _mm512_sub_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(*, v_float32x16, _mm512_mul_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(/, v_float32x16, _mm512_div_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(+, v_float64x8, _mm512_add_pd)
OPENCV_HAL_IMPL_AVX512_BIN_OP(-, v_float64x8, _mm512_sub_pd)
OPENCV_HAL_IMPL_AVX512_BIN_OP(*, v_float64x8, _mm512_mul_pd)
OPENCV_HAL_IMPL_AVX512_BIN_OP(/, v_float64x8, _mm512_div_pd)

// saturating multiply
inline v_uint8x64 operator * (const v_uint8x64& a, const v_uint8x64& b)
{
    v_uint16x32 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_int8x64 operator * (const v_int8x64& a, const v_int8x64& b)
{
    v_int16x32 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_uint16x32 operator * (const v_uint16x32& a, const v_uint16x32& b)
{
    __m512i pl = _mm512_mullo_epi16(a.val, b.val);
    __m512i ph = _mm512_mulhi_epu16(a.val, b.val);
    __m512i p0 = _mm512_unpacklo_epi16(pl, ph);
    __m512i p1 = _mm512_unpackhi_epi16(pl, ph);
    return v_uint16x32(_v512_packs_epu32(p0, p1));
}
inline v_int16x32 operator * (const v_int16x32& a, const v_int16x32& b)
{
    __m512i pl = _mm512_mullo_epi16(a.val, b.val);
    __m512i ph = _mm512_mulhi_epi16(a.val, b.val);
    __m512i p0 = _mm512_unpacklo_epi16(pl, ph);
    __m512i p1 = _mm512_unpackhi_epi16(pl, ph);
    return v_int16x32(_mm512_packs_epi32(p0, p1));
}

inline v_uint8x64& operator *= (v_uint8x64& a, const v_uint8x64& b)
{ a = a * b; return a; }
inline v_int8x64& operator *= (v_int8x64& a, const v_int8x64& b)
{ a = a * b; return a; }
inline v_uint16x32& operator *= (v_uint16x32& a, const v_uint16x32& b)
{ a = a * b; return a; }
inline v_int16x32& operator *= (v_int16x32& a, const v_int16x32& b)
{ a = a * b; return a; }

inline v_int16x32 v_mul_hi(const v_int16x32& a, const v_int16x32& b) { return v_int16x32(_mm512_mulhi_epi16(a.val, b.val)); }
inline v_uint16x32 v_mul_hi(const v_uint16x32& a, const v_uint16x32& b) { return v_uint16x32(_mm512_mulhi_epu16(a.val, b.val)); }

//  Multiply and expand
inline void v_mul_expand(const v_uint8x64& a, const v_uint8x64& b,
                         v_uint16x32& c, v_uint16x32& d)
{
    v_uint16x32 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int8x64& a, const v_int8x64& b,
                         v_int16x32& c, v_int16x32& d)
{
    v_int16x32 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int16x32& a, const v_int16x32& b,
                         v_int32x16& c, v_int32x16& d)
{
    v_int16x32 v0, v1;
    v_zip(v_mul_wrap(a, b), v_mul_hi(a, b), v0, v1);

    c = v_reinterpret_as_s32(v0);
    d = v_reinterpret_as_s32(v1);
}

inline void v_mul_expand(const v_uint16x32& a, const v_uint16x32& b,
                         v_uint32x16& c, v_uint32x16& d)
{
    v_uint16x16 v0, v1;
    v_zip(v_mul_wrap(a, b), v_mul_hi(a, b), v0, v1);

    c = v_reinterpret_as_u32(v0);
    d = v_reinterpret_as_u32(v1);
}

inline void v_mul_expand(const v_uint32x16& a, const v_uint32x16& b,
                         v_uint64x8& c, v_uint64x8& d)
{
    v_zip(v_uint64x8(_mm512_mul_epu32(a.val, b.val)),
          v_uint64x8(_mm512_mul_epu32(_mm512_srli_epi64(a.val, 32), _mm512_srli_epi64(b.val, 32))), c, d);
}

inline void v_mul_expand(const v_int32x16& a, const v_int32x16& b,
    v_int64x8& c, v_int64x8& d)
{
    v_zip(v_int64x8(_mm512_mul_epi32(a.val, b.val)),
          v_int64x8(_mm512_mul_epi32(_mm512_srli_epi64(a.val, 32), _mm512_srli_epi64(b.val, 32))), c, d);
}

/** Bitwise shifts **/
#define OPENCV_HAL_IMPL_AVX512_SHIFT_OP(_Tpuvec, _Tpsvec, suffix) \
    inline _Tpuvec operator << (const _Tpuvec& a, int imm)        \
    { return _Tpuvec(_mm512_slli_##suffix(a.val, imm)); }         \
    inline _Tpsvec operator << (const _Tpsvec& a, int imm)        \
    { return _Tpsvec(_mm512_slli_##suffix(a.val, imm)); }         \
    inline _Tpuvec operator >> (const _Tpuvec& a, int imm)        \
    { return _Tpuvec(_mm512_srli_##suffix(a.val, imm)); }         \
    inline _Tpsvec operator >> (const _Tpsvec& a, int imm)        \
    { return _Tpsvec(_mm512_srai_##suffix(a.val, imm)); }         \
    template<int imm>                                             \
    inline _Tpuvec v_shl(const _Tpuvec& a)                        \
    { return _Tpuvec(_mm512_slli_##suffix(a.val, imm)); }         \
    template<int imm>                                             \
    inline _Tpsvec v_shl(const _Tpsvec& a)                        \
    { return _Tpsvec(_mm512_slli_##suffix(a.val, imm)); }         \
    template<int imm>                                             \
    inline _Tpuvec v_shr(const _Tpuvec& a)                        \
    { return _Tpuvec(_mm512_srli_##suffix(a.val, imm)); }         \
    template<int imm>                                             \
    inline _Tpsvec v_shr(const _Tpsvec& a)                        \
    { return _Tpsvec(_mm512_srai_##suffix(a.val, imm)); }

OPENCV_HAL_IMPL_AVX512_SHIFT_OP(v_uint16x32, v_int16x32, epi16)
OPENCV_HAL_IMPL_AVX512_SHIFT_OP(v_uint32x16, v_int32x16, epi32)
OPENCV_HAL_IMPL_AVX512_SHIFT_OP(v_uint64x8,  v_int64x8,  epi64)


/** Bitwise logic **/
#define OPENCV_HAL_IMPL_AVX512_LOGIC_OP(_Tpvec, suffix, not_const) \
    OPENCV_HAL_IMPL_AVX512_BIN_OP(&, _Tpvec, _mm512_and_##suffix)  \
    OPENCV_HAL_IMPL_AVX512_BIN_OP(|, _Tpvec, _mm512_or_##suffix)   \
    OPENCV_HAL_IMPL_AVX512_BIN_OP(^, _Tpvec, _mm512_xor_##suffix)  \
    inline _Tpvec operator ~ (const _Tpvec& a)                     \
    { return _Tpvec(_mm512_xor_##suffix(a.val, not_const)); }

OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_uint8x64,   si512, _mm512_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_int8x64,    si512, _mm512_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_uint16x32,  si512, _mm512_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_int16x32,   si512, _mm512_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_uint32x16,  si512, _mm512_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_int32x16,   si512, _mm512_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_uint64x8,   si512, _mm512_set1_epi64(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_int64x8,    si512, _mm512_set1_epi64(-1))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_float32x16, ps,    _mm512_castsi512_ps(_mm512_set1_epi32(-1)))
OPENCV_HAL_IMPL_AVX512_LOGIC_OP(v_float64x8,  pd,    _mm512_castsi512_pd(_mm512_set1_epi32(-1)))

/** Select **/
#define OPENCV_HAL_IMPL_AVX512_SELECT(_Tpvec, suffix)                            \
    inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
    { return _Tpvec(_mm512_mask_blend_##suffix(_mm512_cmp_##suffix##_mask(mask.val, _mm512_setzero_si512(), _MM_CMPINT_EQ), a.val, b.val)); }

OPENCV_HAL_IMPL_AVX512_SELECT(v_uint8x64,   epi8)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int8x64,    epi8)
OPENCV_HAL_IMPL_AVX512_SELECT(v_uint16x32, epi16)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int16x32,  epi16)
OPENCV_HAL_IMPL_AVX512_SELECT(v_uint32x16, epi32)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int32x16,  epi32)
OPENCV_HAL_IMPL_AVX512_SELECT(v_uint64x8,  epi64)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int64x8,   epi64)
OPENCV_HAL_IMPL_AVX512_SELECT(v_float32x16,   ps)
OPENCV_HAL_IMPL_AVX512_SELECT(v_float64x8,    pd)

/** Comparison **/
#define OPENCV_HAL_IMPL_AVX512_CMP_INT(bin_op, imm8, _Tpvec, sufcmp, sufset, tval) \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b)               \
    { return _Tpvec(_mm512_maskz_set1_##sufset(_mm512_cmp_##sufcmp##_mask(a.val, b.val, imm8), tval)); }

#define OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(_Tpvec, sufcmp, sufset, tval)           \
    OPENCV_HAL_IMPL_AVX_CMP_INT(==, _MM_CMPINT_EQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX_CMP_INT(!=, _MM_CMPINT_NE,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX_CMP_INT(<,  _MM_CMPINT_LT,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX_CMP_INT(>,  _MM_CMPINT_NLE, _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX_CMP_INT(<=, _MM_CMPINT_LE,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX_CMP_INT(>=, _MM_CMPINT_NLT, _Tpvec, sufcmp, sufset, tval)

OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint8x64,   epu8,  epi8, (char)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int8x64,    epi8,  epi8, (char)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint16x32, epu16, epi16, (short)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int16x32,  epi16, epi16, (short)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint32x16, epu32, epi32, (int)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int32x16,  epi32, epi32, (int)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint64x8,  epu64, epi64, (int64)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int64x8,   epi64, epi64, (int64)-1)

#define OPENCV_HAL_IMPL_AVX512_CMP_FLT(bin_op, imm8, _Tpvec, sufcmp, sufset, tval) \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b)               \
    { return _Tpvec(_mm512_castsi512_##sufcmp(_mm512_maskz_set1_##sufset(_mm512_cmp_##sufcmp##_mask(a.val, b.val, imm8), tval))); }

#define OPENCV_HAL_IMPL_AVX512_CMP_OP_FLT(_Tpvec, sufcmp, sufset, tval)           \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(==, _CMP_EQ_OQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(!=, _CMP_NEQ_OQ, _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(<,  _CMP_LT_OQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(>,  _CMP_GT_OQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(<=, _CMP_LE_OQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(>=, _CMP_GE_OQ,  _Tpvec, sufcmp, sufset, tval)

OPENCV_HAL_IMPL_AVX512_CMP_OP_FLT(v_float32x16, ps, epi32, (int)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_FLT(v_float64x8,  pd, epi64, (int64)-1)

inline v_float32x16 v_not_nan(const v_float32x16& a)
{ return v_float32x16(_mm512_castsi512_ps(_mm512_maskz_set1_epi32(_mm512_cmp_ps_mask(a.val, a.val, _CMP_ORD_Q), (int)-1))); }
inline v_float64x8 v_not_nan(const v_float64x8& a)
{ return v_float64x8(_mm512_castsi512_pd(_mm512_maskz_set1_epi64(_mm512_cmp_pd_mask(a.val, a.val, _CMP_ORD_Q), (int64)-1))); }

/** min/max **/
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_uint8x64,   _mm512_min_epu8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_uint8x64,   _mm512_max_epu8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_int8x64,    _mm512_min_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_int8x64,    _mm512_max_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_uint16x32,  _mm512_min_epu16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_uint16x32,  _mm512_max_epu16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_int16x32,   _mm512_min_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_int16x32,   _mm512_max_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_uint32x16,  _mm512_min_epu32)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_uint32x16,  _mm512_max_epu32)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_int32x16,   _mm512_min_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_int32x16,   _mm512_max_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_uint64x8,   _mm512_min_epu64)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_uint64x8,   _mm512_max_epu64)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_int64x8,    _mm512_min_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_int64x8,    _mm512_max_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_float32x16, _mm512_min_ps)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_float32x16, _mm512_max_ps)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_min, v_float64x8,  _mm512_min_pd)
OPENCV_HAL_IMPL_AVX512_BIN_FUNC(v_max, v_float64x8,  _mm512_max_pd)

/** Rotate **/
#define OPENCV_HAL_IMPL_AVX512_ROTATE(_Tpvec, suffix)                                                                                    \
template<int imm>                                                                                                                        \
inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b)                                                                            \
{                                                                                                                                        \
    enum { SHIFT2 = _Tpvec::nlanes - imm };                                                                                              \
    if (imm == 0) return a;                                                                                                              \
    if (imm == _Tpvec::nlanes) return b;                                                                                                 \
    if (imm >= 2*_Tpvec::nlanes) return _Tpvec();                                                                                        \
    return _Tpvec(_mm512_mask_expand_##suffix(_mm512_maskz_compress_##suffix(-1 << SHIFT2, b), -1 << imm, a));                           \
  /*const _Tpvec::lane_type idx[] = { 0x00+SHIFT2,0x01+SHIFT2,0x02+SHIFT2,0x03+SHIFT2,0x04+SHIFT2,0x05+SHIFT2,0x06+SHIFT2,0x07+SHIFT2,   \
                                      0x08+SHIFT2,0x09+SHIFT2,0x0a+SHIFT2,0x0b+SHIFT2,0x0c+SHIFT2,0x0d+SHIFT2,0x0e+SHIFT2,0x0f+SHIFT2,   \
                                      0x10+SHIFT2,0x11+SHIFT2,0x12+SHIFT2,0x13+SHIFT2,0x14+SHIFT2,0x15+SHIFT2,0x16+SHIFT2,0x17+SHIFT2,   \
                                      0x18+SHIFT2,0x19+SHIFT2,0x1a+SHIFT2,0x1b+SHIFT2,0x1c+SHIFT2,0x1d+SHIFT2,0x1e+SHIFT2,0x1f+SHIFT2,   \
                                      0x20+SHIFT2,0x21+SHIFT2,0x22+SHIFT2,0x23+SHIFT2,0x24+SHIFT2,0x25+SHIFT2,0x26+SHIFT2,0x27+SHIFT2,   \
                                      0x28+SHIFT2,0x29+SHIFT2,0x2a+SHIFT2,0x2b+SHIFT2,0x2c+SHIFT2,0x2d+SHIFT2,0x2e+SHIFT2,0x2f+SHIFT2,   \
                                      0x30+SHIFT2,0x31+SHIFT2,0x32+SHIFT2,0x33+SHIFT2,0x34+SHIFT2,0x35+SHIFT2,0x36+SHIFT2,0x37+SHIFT2,   \
                                      0x38+SHIFT2,0x39+SHIFT2,0x3a+SHIFT2,0x3b+SHIFT2,0x3c+SHIFT2,0x3d+SHIFT2,0x3e+SHIFT2,0x3f+SHIFT2 }; \
    return _Tpvec(_mm512_permutex2var_##suffix(b, _mm512_load_si512(idx), a));*/                                                         \
}                                                                                                                                        \
template<int imm>                                                                                                                        \
inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b)                                                                           \
{                                                                                                                                        \
    enum { SHIFT2 = _Tpvec::nlanes - imm };                                                                                              \
    if (imm == 0) return a;                                                                                                              \
    if (imm == _Tpvec::nlanes) return b;                                                                                                 \
    if (imm >= 2*_Tpvec::nlanes) return _Tpvec();                                                                                        \
    return _Tpvec(_mm512_mask_expand_##suffix(_mm512_maskz_compress_##suffix(-1 << imm, a), -1 << SHIFT2, b));                           \
  /*const _Tpvec::lane_type idx[] = { 0x00+imm,0x01+imm,0x02+imm,0x03+imm,0x04+imm,0x05+imm,0x06+imm,0x07+imm,                           \
                                      0x08+imm,0x09+imm,0x0a+imm,0x0b+imm,0x0c+imm,0x0d+imm,0x0e+imm,0x0f+imm,                           \
                                      0x10+imm,0x11+imm,0x12+imm,0x13+imm,0x14+imm,0x15+imm,0x16+imm,0x17+imm,                           \
                                      0x18+imm,0x19+imm,0x1a+imm,0x1b+imm,0x1c+imm,0x1d+imm,0x1e+imm,0x1f+imm,                           \
                                      0x20+imm,0x21+imm,0x22+imm,0x23+imm,0x24+imm,0x25+imm,0x26+imm,0x27+imm,                           \
                                      0x28+imm,0x29+imm,0x2a+imm,0x2b+imm,0x2c+imm,0x2d+imm,0x2e+imm,0x2f+imm,                           \
                                      0x30+imm,0x31+imm,0x32+imm,0x33+imm,0x34+imm,0x35+imm,0x36+imm,0x37+imm,                           \
                                      0x38+imm,0x39+imm,0x3a+imm,0x3b+imm,0x3c+imm,0x3d+imm,0x3e+imm,0x3f+imm };                         \
    return _Tpvec(_mm512_permutex2var_##suffix(a, _mm512_load_si512(idx), b));*/                                                         \
}                                                                                                                                        \
template<int imm>                                                                                                                        \
inline _Tpvec v_rotate_left(const _Tpvec& a)                                                                                             \
{                                                                                                                                        \
    enum { SHIFT2 = _Tpvec::nlanes - imm };                                                                                              \
    if (imm == 0) return a;                                                                                                              \
    if (imm >= _Tpvec::nlanes) return _Tpvec();                                                                                          \
    return _Tpvec(_mm512_maskz_expand_##suffix(-1 << imm, a));                                                                           \
}                                                                                                                                        \
template<int imm>                                                                                                                        \
inline _Tpvec v_rotate_right(const _Tpvec& a)                                                                                            \
{                                                                                                                                        \
    enum { SHIFT2 = _Tpvec::nlanes - imm };                                                                                              \
    if (imm == 0) return a;                                                                                                              \
    if (imm >= _Tpvec::nlanes) return _Tpvec();                                                                                          \
    return _Tpvec(_mm512_maskz_compress_##suffix(-1 << imm, a));                                                                         \
}

#define OPENCV_HAL_IMPL_AVX_ROTATE(_Tpvec)                                  \
    OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_left,  _Tpvec, OPENCV_HAL_NOP) \
    OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_right, _Tpvec, OPENCV_HAL_NOP)

OPENCV_HAL_IMPL_AVX512_ROTATE(v_uint8x64,   epi8)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_int8x64,    epi8)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_uint16x32, epi16)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_int16x32,  epi16)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_uint32x16, epi32)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_int32x16,  epi32)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_uint64x8,  epi64)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_int64x8,   epi64)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_float32x16,   ps)
OPENCV_HAL_IMPL_AVX512_ROTATE(v_float64x8,    pd)

////////// Reduce and mask /////////

/** Reduce **/
#define OPENCV_HAL_IMPL_AVX512_REDUCE(func, ifunc, _Tpvec, sctype, suffix) \
    inline sctype v_reduce_##func(const _Tpvec& a)                         \
    { return _mm512_reduce_##ifunc##_##suffix(a); }

OPENCV_HAL_IMPL_AVX512_REDUCE(min, min, v_uint32x16,  uint,   _epu32)
OPENCV_HAL_IMPL_AVX512_REDUCE(min, min, v_int32x16,   int,    _epi32)
OPENCV_HAL_IMPL_AVX512_REDUCE(min, min, v_uint64x8,   uint64, _epu64)
OPENCV_HAL_IMPL_AVX512_REDUCE(min, min, v_int64x8,    int64,  _epi64)
OPENCV_HAL_IMPL_AVX512_REDUCE(min, min, v_float32x16, float,  _ps)
OPENCV_HAL_IMPL_AVX512_REDUCE(min, min, v_float64x8,  double, _pd)

OPENCV_HAL_IMPL_AVX512_REDUCE(max, max, v_uint32x16,  uint,   _epu32)
OPENCV_HAL_IMPL_AVX512_REDUCE(max, max, v_int32x16,   int,    _epi32)
OPENCV_HAL_IMPL_AVX512_REDUCE(max, max, v_uint64x8,   uint64, _epu64)
OPENCV_HAL_IMPL_AVX512_REDUCE(max, max, v_int64x8,    int64,  _epi64)
OPENCV_HAL_IMPL_AVX512_REDUCE(max, max, v_float32x16, float,  _ps)
OPENCV_HAL_IMPL_AVX512_REDUCE(max, max, v_float64x8,  double, _pd)

OPENCV_HAL_IMPL_AVX512_REDUCE(sum, add, v_uint32x16,  uint,   _epu32)
OPENCV_HAL_IMPL_AVX512_REDUCE(sum, add, v_int32x16,   int,    _epi32)
OPENCV_HAL_IMPL_AVX512_REDUCE(sum, add, v_uint64x8,   uint64, _epu64)
OPENCV_HAL_IMPL_AVX512_REDUCE(sum, add, v_int64x8,    int64,  _epi64)
OPENCV_HAL_IMPL_AVX512_REDUCE(sum, add, v_float32x16, float,  _ps)
OPENCV_HAL_IMPL_AVX512_REDUCE(sum, add, v_float64x8,  double, _pd)

#define OPENCV_HAL_IMPL_AVX512_REDUCE_32(func, ifunc, _Tpvec, sctype, suffix) \
    inline sctype v_reduce_##func(const _Tpvec& a)                            \
    { return (sctype)_mm512_reduce_##ifunc##_epi32(_mm512_cvt##suffix##_epi32(_mm256_##ifunc##_##suffix(_v512_extract_low(a.val), _v512_extract_high(a.val)))); }
#define OPENCV_HAL_IMPL_AVX512_REDUCE_32_SUM(func, ifunc, _Tpvec, sctype, suffix) \
    inline sctype v_reduce_##func(const _Tpvec& a)                                \
    { return (sctype)_mm512_reduce_##ifunc##_epi32(_mm512_##ifunc##_##suffix(_mm512_cvt##suffix##_epi32(_v512_extract_low(a.val)), _mm512_cvt##suffix##_epi32(_v512_extract_high(a.val)))); }

OPENCV_HAL_IMPL_AVX512_REDUCE_32(min, min, v_uint16x32, ushort, epu16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32(min, min, v_int16x32,  short,  epi16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32(max, max, v_uint16x32, ushort, epu16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32(max, max, v_int16x32,  short,  epi16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32_SUM(sum, add, v_uint16x32, uint, epu16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32_SUM(sum, add, v_int16x32,  int,  epi16)

#define OPENCV_HAL_IMPL_AVX512_REDUCE_64(func, ifunc, _Tpvec, sctype, suffix)                                      \
    inline sctype v_reduce_##func(const _Tpvec& a)                                                                 \
    {                                                                                                              \
        __m256i half = _mm256_##ifunc##_##suffix(_v512_extract_low(a.val), _v512_extract_high(a.val));             \
        __m128i quarter = _mm_##ifunc##_##suffix(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1)); \
        return (sctype)_mm512_reduce_##ifunc##_epi32(_mm512_cvt##suffix##_epi32(quarter));                         \
    }
#define OPENCV_HAL_IMPL_AVX512_REDUCE_64_SUM(func, ifunc, _Tpvec, sctype, suffix)                                  \
    inline sctype v_reduce_##func(const _Tpvec& a)                                                                 \
    {                                                                                                              \
        __m512i half = _mm512_##ifunc##_##suffix(_mm512_cvt##suffix##_epi16(_v512_extract_low(a.val)),             \
                                                 _mm512_cvt##suffix##_epi16(_v512_extract_high(a.val)));           \
        __m256i quarter = _mm256_##ifunc##_##suffix(_v512_extract_low(half), _v512_extract_high(half));            \
        return (sctype)_mm512_reduce_##ifunc##_epi32(_mm512_cvtepi16_epi32(quarter));                              \
    }

OPENCV_HAL_IMPL_AVX512_REDUCE_64(min, min, v_uint8x64, uchar, epu8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64(min, min, v_int8x64,  char,  epi8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64(max, max, v_uint8x64, uchar, epu8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64(max, max, v_int8x64,  char,  epi8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64_SUM(sum, add, v_uint8x64, uint, epu8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64_SUM(sum, add, v_int8x64,  int,  epi8)

inline v_float32x16 v_reduce_sum4(const v_float32x16& a, const v_float32x16& b,
                                  const v_float32x16& c, const v_float32x16& d)
{
    __m256 abl = _mm256_hadd_ps(_v512_extract_low(a.val), _v512_extract_low(b.val));
    __m256 abh = _mm256_hadd_ps(_v512_extract_high(a.val), _v512_extract_high(b.val));
    __m256 cdl = _mm256_hadd_ps(_v512_extract_low(c.val), _v512_extract_low(d.val));
    __m256 cdh = _mm256_hadd_ps(_v512_extract_high(c.val), _v512_extract_high(d.val));
    return v_float32x16(_v512_combine(_mm256_hadd_ps(abl, cdl), _mm256_hadd_ps(abh, cdh)));
}

inline unsigned v_reduce_sad(const v_uint8x64& a, const v_uint8x64& b)
{
    return (unsigned)_mm512_reduce_add_epi32(_mm512_sad_epu8(a.val, b.val));
}
inline unsigned v_reduce_sad(const v_int8x64& a, const v_int8x64& b)
{
    __m512i half = _mm512_set1_epi8(0x80);
    return (unsigned)_mm512_reduce_add_epi32(_mm512_sad_epu8(_mm512_add_epi8(a.val, half), _mm512_add_epi8(b.val, half)));
}
inline unsigned v_reduce_sad(const v_uint16x32& a, const v_uint16x32& b)
{ return v_reduce_sum(v_add_wrap(a - b, b - a)); }
inline unsigned v_reduce_sad(const v_int16x32& a, const v_int16x32& b)
{ return v_reduce_sum(v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b)))); }
inline unsigned v_reduce_sad(const v_uint32x16& a, const v_uint32x16& b)
{ return v_reduce_sum(v_max(a, b) - v_min(a, b)); }
inline unsigned v_reduce_sad(const v_int32x16& a, const v_int32x16& b)
{ return v_reduce_sum(v_reinterpret_as_u32(v_max(a, b) - v_min(a, b))); }
inline float v_reduce_sad(const v_float32x16& a, const v_float32x16& b)
{ return v_reduce_sum((a - b) & v_float32x8(_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff)))); }
inline double v_reduce_sad(const v_float64x8& a, const v_float64x8& b)
{ return v_reduce_sum((a - b) & v_float64x8(_mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffff)))); }

/** Popcount **/
#define OPENCV_HAL_IMPL_AVX512_POPCOUNT(_Tpvec, _Tpuvec, suffix) \
    inline _Tpuvec v_popcount(const _Tpvec& a)                   \
    { return _Tpuvec(_mm512_popcnt_##suffix(a.val)); }           \
    inline _Tpuvec v_popcount(const _Tupvec& a)                  \
    { return _Tpuvec(_mm512_popcnt_##suffix(a.val)); }

OPENCV_HAL_IMPL_AVX512_POPCOUNT(v_int8x64,  v_uint8x64,  epi8)
OPENCV_HAL_IMPL_AVX512_POPCOUNT(v_int16x32, v_uint16x32, epi16)
OPENCV_HAL_IMPL_AVX512_POPCOUNT(v_int32x16, v_uint32x16, epi32)
OPENCV_HAL_IMPL_AVX512_POPCOUNT(v_int64x8,  v_uint64x8,  epi64)

/** Mask **/
inline int v_signmask(const v_int8x32& a)
{ return _mm256_movemask_epi8(a.val); }
inline int v_signmask(const v_uint8x32& a)
{ return v_signmask(v_reinterpret_as_s8(a)); }

inline int v_signmask(const v_int16x16& a)
{
    v_int8x32 v = v_int8x32(_mm256_packs_epi16(a.val, a.val));
    return v_signmask(v) & 255;
}
inline int v_signmask(const v_uint16x16& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }

inline int v_signmask(const v_int32x8& a)
{
    __m512i a16 = _mm256_packs_epi32(a.val, a.val);
    v_int8x32 v = v_int8x32(_mm256_packs_epi16(a16, a16));
    return v_signmask(v) & 15;
}
inline int v_signmask(const v_uint32x8& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }

inline int v_signmask(const v_float32x8& a)
{ return _mm256_movemask_ps(a.val); }
inline int v_signmask(const v_float64x4& a)
{ return _mm256_movemask_pd(a.val); }

/** Checks **/
#define OPENCV_HAL_IMPL_AVX_CHECK(_Tpvec, and_op, allmask)  \
    inline bool v_check_all(const _Tpvec& a)                \
    {                                                       \
        int mask = v_signmask(v_reinterpret_as_s8(a));      \
        return and_op(mask, allmask) == allmask;            \
    }                                                       \
    inline bool v_check_any(const _Tpvec& a)                \
    {                                                       \
        int mask = v_signmask(v_reinterpret_as_s8(a));      \
        return and_op(mask, allmask) != 0;                  \
    }

OPENCV_HAL_IMPL_AVX_CHECK(v_uint8x32,  OPENCV_HAL_1ST, -1)
OPENCV_HAL_IMPL_AVX_CHECK(v_int8x32,   OPENCV_HAL_1ST, -1)
OPENCV_HAL_IMPL_AVX_CHECK(v_uint16x16, OPENCV_HAL_AND, (int)0xaaaa)
OPENCV_HAL_IMPL_AVX_CHECK(v_int16x16,  OPENCV_HAL_AND, (int)0xaaaa)
OPENCV_HAL_IMPL_AVX_CHECK(v_uint32x8,  OPENCV_HAL_AND, (int)0x8888)
OPENCV_HAL_IMPL_AVX_CHECK(v_int32x8,   OPENCV_HAL_AND, (int)0x8888)

#define OPENCV_HAL_IMPL_AVX_CHECK_FLT(_Tpvec, allmask) \
    inline bool v_check_all(const _Tpvec& a)           \
    {                                                  \
        int mask = v_signmask(a);                      \
        return mask == allmask;                        \
    }                                                  \
    inline bool v_check_any(const _Tpvec& a)           \
    {                                                  \
        int mask = v_signmask(a);                      \
        return mask != 0;                              \
    }

OPENCV_HAL_IMPL_AVX_CHECK_FLT(v_float32x8, 255)
OPENCV_HAL_IMPL_AVX_CHECK_FLT(v_float64x4, 15)


////////// Other math /////////

/** Some frequent operations **/
#define OPENCV_HAL_IMPL_AVX_MULADD(_Tpvec, suffix)                            \
    inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)    \
    { return _Tpvec(_mm256_fmadd_##suffix(a.val, b.val, c.val)); }            \
    inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
    { return _Tpvec(_mm256_fmadd_##suffix(a.val, b.val, c.val)); }            \
    inline _Tpvec v_sqrt(const _Tpvec& x)                                     \
    { return _Tpvec(_mm256_sqrt_##suffix(x.val)); }                           \
    inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b)           \
    { return v_fma(a, a, b * b); }                                            \
    inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b)               \
    { return v_sqrt(v_fma(a, a, b*b)); }

OPENCV_HAL_IMPL_AVX_MULADD(v_float32x8, ps)
OPENCV_HAL_IMPL_AVX_MULADD(v_float64x4, pd)

inline v_int32x8 v_fma(const v_int32x8& a, const v_int32x8& b, const v_int32x8& c)
{
    return a * b + c;
}

inline v_int32x8 v_muladd(const v_int32x8& a, const v_int32x8& b, const v_int32x8& c)
{
    return v_fma(a, b, c);
}

inline v_float32x8 v_invsqrt(const v_float32x8& x)
{
    v_float32x8 half = x * v512_setall_f32(0.5);
    v_float32x8 t  = v_float32x8(_mm256_rsqrt_ps(x.val));
    // todo: _mm256_fnmsub_ps
    t *= v512_setall_f32(1.5) - ((t * t) * half);
    return t;
}

inline v_float64x4 v_invsqrt(const v_float64x4& x)
{
    return v512_setall_f64(1.) / v_sqrt(x);
}

/** Absolute values **/
#define OPENCV_HAL_IMPL_AVX_ABS(_Tpvec, suffix)         \
    inline v_u##_Tpvec v_abs(const v_##_Tpvec& x)       \
    { return v_u##_Tpvec(_mm256_abs_##suffix(x.val)); }

OPENCV_HAL_IMPL_AVX_ABS(int8x32,  epi8)
OPENCV_HAL_IMPL_AVX_ABS(int16x16, epi16)
OPENCV_HAL_IMPL_AVX_ABS(int32x8,  epi32)

inline v_float32x8 v_abs(const v_float32x8& x)
{ return x & v_float32x8(_mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff))); }
inline v_float64x4 v_abs(const v_float64x4& x)
{ return x & v_float64x4(_mm256_castsi256_pd(_mm256_srli_epi64(_mm256_set1_epi64x(-1), 1))); }

/** Absolute difference **/
inline v_uint8x32 v_absdiff(const v_uint8x32& a, const v_uint8x32& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint16x16 v_absdiff(const v_uint16x16& a, const v_uint16x16& b)
{ return v_add_wrap(a - b,  b - a); }
inline v_uint32x8 v_absdiff(const v_uint32x8& a, const v_uint32x8& b)
{ return v_max(a, b) - v_min(a, b); }

inline v_uint8x32 v_absdiff(const v_int8x32& a, const v_int8x32& b)
{
    v_int8x32 d = v_sub_wrap(a, b);
    v_int8x32 m = a < b;
    return v_reinterpret_as_u8(v_sub_wrap(d ^ m, m));
}

inline v_uint16x16 v_absdiff(const v_int16x16& a, const v_int16x16& b)
{ return v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b))); }

inline v_uint32x8 v_absdiff(const v_int32x8& a, const v_int32x8& b)
{
    v_int32x8 d = a - b;
    v_int32x8 m = a < b;
    return v_reinterpret_as_u32((d ^ m) - m);
}

inline v_float32x8 v_absdiff(const v_float32x8& a, const v_float32x8& b)
{ return v_abs(a - b); }

inline v_float64x4 v_absdiff(const v_float64x4& a, const v_float64x4& b)
{ return v_abs(a - b); }

/** Saturating absolute difference **/
inline v_int8x32 v_absdiffs(const v_int8x32& a, const v_int8x32& b)
{
    v_int8x32 d = a - b;
    v_int8x32 m = a < b;
    return (d ^ m) - m;
}
inline v_int16x16 v_absdiffs(const v_int16x16& a, const v_int16x16& b)
{ return v_max(a, b) - v_min(a, b); }

////////// Conversions /////////

/** Rounding **/
inline v_int32x8 v_round(const v_float32x8& a)
{ return v_int32x8(_mm256_cvtps_epi32(a.val)); }

inline v_int32x8 v_round(const v_float64x4& a)
{ return v_int32x8(_mm256_castsi128_si256(_mm256_cvtpd_epi32(a.val))); }

inline v_int32x8 v_round(const v_float64x4& a, const v_float64x4& b)
{
    __m256i ai = _mm256_cvtpd_epi32(a.val), bi = _mm256_cvtpd_epi32(b.val);
    return v_int32x8(_v512_combine(ai, bi));
}

inline v_int32x8 v_trunc(const v_float32x8& a)
{ return v_int32x8(_mm256_cvttps_epi32(a.val)); }

inline v_int32x8 v_trunc(const v_float64x4& a)
{ return v_int32x8(_mm256_castsi128_si256(_mm256_cvttpd_epi32(a.val))); }

inline v_int32x8 v_floor(const v_float32x8& a)
{ return v_int32x8(_mm256_cvttps_epi32(_mm256_floor_ps(a.val))); }

inline v_int32x8 v_floor(const v_float64x4& a)
{ return v_trunc(v_float64x4(_mm256_floor_pd(a.val))); }

inline v_int32x8 v_ceil(const v_float32x8& a)
{ return v_int32x8(_mm256_cvttps_epi32(_mm256_ceil_ps(a.val))); }

inline v_int32x8 v_ceil(const v_float64x4& a)
{ return v_trunc(v_float64x4(_mm256_ceil_pd(a.val))); }

/** To float **/
inline v_float32x8 v_cvt_f32(const v_int32x8& a)
{ return v_float32x8(_mm256_cvtepi32_ps(a.val)); }

inline v_float32x8 v_cvt_f32(const v_float64x4& a)
{ return v_float32x8(_mm256_castps128_ps256(_mm256_cvtpd_ps(a.val))); }

inline v_float32x8 v_cvt_f32(const v_float64x4& a, const v_float64x4& b)
{
    __m256 af = _mm256_cvtpd_ps(a.val), bf = _mm256_cvtpd_ps(b.val);
    return v_float32x8(_mm256_insertf128_ps(_mm256_castps128_ps256(af), bf, 1));
}

inline v_float64x4 v_cvt_f64(const v_int32x8& a)
{ return v_float64x4(_mm256_cvtepi32_pd(_v512_extract_low(a.val))); }

inline v_float64x4 v_cvt_f64_high(const v_int32x8& a)
{ return v_float64x4(_mm256_cvtepi32_pd(_v512_extract_high(a.val))); }

inline v_float64x4 v_cvt_f64(const v_float32x8& a)
{ return v_float64x4(_mm256_cvtps_pd(_v512_extract_low(a.val))); }

inline v_float64x4 v_cvt_f64_high(const v_float32x8& a)
{ return v_float64x4(_mm256_cvtps_pd(_v512_extract_high(a.val))); }

////////////// Lookup table access ////////////////////

inline v_int8x32 v512_lut(const schar* tab, const int* idx)
{
    return v_int8x32(_mm256_setr_epi8(tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
                                      tab[idx[ 8]], tab[idx[ 9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]],
                                      tab[idx[16]], tab[idx[17]], tab[idx[18]], tab[idx[19]], tab[idx[20]], tab[idx[21]], tab[idx[22]], tab[idx[23]],
                                      tab[idx[24]], tab[idx[25]], tab[idx[26]], tab[idx[27]], tab[idx[28]], tab[idx[29]], tab[idx[30]], tab[idx[31]]));
}
inline v_int8x32 v512_lut_pairs(const schar* tab, const int* idx)
{
    return v_int8x32(_mm256_setr_epi16(*(const short*)(tab + idx[ 0]), *(const short*)(tab + idx[ 1]), *(const short*)(tab + idx[ 2]), *(const short*)(tab + idx[ 3]),
                                       *(const short*)(tab + idx[ 4]), *(const short*)(tab + idx[ 5]), *(const short*)(tab + idx[ 6]), *(const short*)(tab + idx[ 7]),
                                       *(const short*)(tab + idx[ 8]), *(const short*)(tab + idx[ 9]), *(const short*)(tab + idx[10]), *(const short*)(tab + idx[11]),
                                       *(const short*)(tab + idx[12]), *(const short*)(tab + idx[13]), *(const short*)(tab + idx[14]), *(const short*)(tab + idx[15])));
}
inline v_int8x32 v512_lut_quads(const schar* tab, const int* idx)
{
    return v_int8x32(_mm256_i32gather_epi32((const int*)tab, _mm256_loadu_si256((const __m512i*)idx), 1));
}
inline v_uint8x32 v512_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut((const schar *)tab, idx)); }
inline v_uint8x32 v512_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut_pairs((const schar *)tab, idx)); }
inline v_uint8x32 v512_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut_quads((const schar *)tab, idx)); }

inline v_int16x16 v512_lut(const short* tab, const int* idx)
{
    return v_int16x16(_mm256_setr_epi16(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
                                        tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]));
}
inline v_int16x16 v512_lut_pairs(const short* tab, const int* idx)
{
    return v_int16x16(_mm256_i32gather_epi32((const int*)tab, _mm256_loadu_si256((const __m512i*)idx), 2));
}
inline v_int16x16 v512_lut_quads(const short* tab, const int* idx)
{
#if defined(__GNUC__)
    return v_int16x16(_mm256_i32gather_epi64((const long long int*)tab, _mm_loadu_si128((const __m256i*)idx), 2));//Looks like intrinsic has wrong definition
#else
    return v_int16x16(_mm256_i32gather_epi64((const int64*)tab, _mm_loadu_si128((const __m256i*)idx), 2));
#endif
}
inline v_uint16x16 v512_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut((const short *)tab, idx)); }
inline v_uint16x16 v512_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut_pairs((const short *)tab, idx)); }
inline v_uint16x16 v512_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut_quads((const short *)tab, idx)); }

inline v_int32x8 v512_lut(const int* tab, const int* idx)
{
    return v_int32x8(_mm256_i32gather_epi32(tab, _mm256_loadu_si256((const __m512i*)idx), 4));
}
inline v_int32x8 v512_lut_pairs(const int* tab, const int* idx)
{
#if defined(__GNUC__)
    return v_int32x8(_mm256_i32gather_epi64((const long long int*)tab, _mm_loadu_si128((const __m256i*)idx), 4));
#else
    return v_int32x8(_mm256_i32gather_epi64((const int64*)tab, _mm_loadu_si128((const __m256i*)idx), 4));
#endif
}
inline v_int32x8 v512_lut_quads(const int* tab, const int* idx)
{
    return v_int32x8(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m256i*)(tab + idx[0]))), _mm_loadu_si128((const __m256i*)(tab + idx[1])), 0x1));
}
inline v_uint32x8 v512_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut((const int *)tab, idx)); }
inline v_uint32x8 v512_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut_pairs((const int *)tab, idx)); }
inline v_uint32x8 v512_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut_quads((const int *)tab, idx)); }

inline v_int64x4 v512_lut(const int64* tab, const int* idx)
{
#if defined(__GNUC__)
    return v_int64x4(_mm256_i32gather_epi64((const long long int*)tab, _mm_loadu_si128((const __m256i*)idx), 8));
#else
    return v_int64x4(_mm256_i32gather_epi64(tab, _mm_loadu_si128((const __m256i*)idx), 8));
#endif
}
inline v_int64x4 v512_lut_pairs(const int64* tab, const int* idx)
{
    return v_int64x4(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_loadu_si128((const __m256i*)(tab + idx[0]))), _mm_loadu_si128((const __m256i*)(tab + idx[1])), 0x1));
}
inline v_uint64x4 v512_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v512_lut((const int64 *)tab, idx)); }
inline v_uint64x4 v512_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v512_lut_pairs((const int64 *)tab, idx)); }

inline v_float32x8 v512_lut(const float* tab, const int* idx)
{
    return v_float32x8(_mm256_i32gather_ps(tab, _mm256_loadu_si256((const __m512i*)idx), 4));
}
inline v_float32x8 v512_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v512_lut_pairs((const int *)tab, idx)); }
inline v_float32x8 v512_lut_quads(const float* tab, const int* idx) { return v_reinterpret_as_f32(v512_lut_quads((const int *)tab, idx)); }

inline v_float64x4 v512_lut(const double* tab, const int* idx)
{
    return v_float64x4(_mm256_i32gather_pd(tab, _mm_loadu_si128((const __m256i*)idx), 8));
}
inline v_float64x4 v512_lut_pairs(const double* tab, const int* idx) { return v_float64x4(_mm256_insertf128_pd(_mm256_castpd128_pd256(_mm_loadu_pd(tab + idx[0])), _mm_loadu_pd(tab + idx[1]), 0x1)); }

inline v_int32x8 v_lut(const int* tab, const v_int32x8& idxvec)
{
    return v_int32x8(_mm256_i32gather_epi32(tab, idxvec.val, 4));
}

inline v_uint32x8 v_lut(const unsigned* tab, const v_int32x8& idxvec)
{
    return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
}

inline v_float32x8 v_lut(const float* tab, const v_int32x8& idxvec)
{
    return v_float32x8(_mm256_i32gather_ps(tab, idxvec.val, 4));
}

inline v_float64x4 v_lut(const double* tab, const v_int32x8& idxvec)
{
    return v_float64x4(_mm256_i32gather_pd(tab, _mm256_castsi256_si128(idxvec.val), 8));
}

inline void v_lut_deinterleave(const float* tab, const v_int32x8& idxvec, v_float32x8& x, v_float32x8& y)
{
    int CV_DECL_ALIGNED(32) idx[8];
    v_store_aligned(idx, idxvec);
    __m256 z = _mm_setzero_ps();
    __m256 xy01, xy45, xy23, xy67;
    xy01 = _mm_loadl_pi(z, (const __m64*)(tab + idx[0]));
    xy01 = _mm_loadh_pi(xy01, (const __m64*)(tab + idx[1]));
    xy45 = _mm_loadl_pi(z, (const __m64*)(tab + idx[4]));
    xy45 = _mm_loadh_pi(xy45, (const __m64*)(tab + idx[5]));
    __m512 xy0145 = _v512_combine(xy01, xy45);
    xy23 = _mm_loadl_pi(z, (const __m64*)(tab + idx[2]));
    xy23 = _mm_loadh_pi(xy23, (const __m64*)(tab + idx[3]));
    xy67 = _mm_loadl_pi(z, (const __m64*)(tab + idx[6]));
    xy67 = _mm_loadh_pi(xy67, (const __m64*)(tab + idx[7]));
    __m512 xy2367 = _v512_combine(xy23, xy67);

    __m512 xxyy0145 = _mm256_unpacklo_ps(xy0145, xy2367);
    __m512 xxyy2367 = _mm256_unpackhi_ps(xy0145, xy2367);

    x = v_float32x8(_mm256_unpacklo_ps(xxyy0145, xxyy2367));
    y = v_float32x8(_mm256_unpackhi_ps(xxyy0145, xxyy2367));
}

inline void v_lut_deinterleave(const double* tab, const v_int32x8& idxvec, v_float64x4& x, v_float64x4& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_low(idx, idxvec);
    __m256d xy0 = _mm_loadu_pd(tab + idx[0]);
    __m256d xy2 = _mm_loadu_pd(tab + idx[2]);
    __m256d xy1 = _mm_loadu_pd(tab + idx[1]);
    __m256d xy3 = _mm_loadu_pd(tab + idx[3]);
    __m512d xy02 = _v512_combine(xy0, xy2);
    __m512d xy13 = _v512_combine(xy1, xy3);

    x = v_float64x4(_mm256_unpacklo_pd(xy02, xy13));
    y = v_float64x4(_mm256_unpackhi_pd(xy02, xy13));
}

inline v_int8x32 v_interleave_pairs(const v_int8x32& vec)
{
    return v_int8x32(_mm256_shuffle_epi8(vec.val, _mm256_set_epi64x(0x0f0d0e0c0b090a08, 0x0705060403010200, 0x0f0d0e0c0b090a08, 0x0705060403010200)));
}
inline v_uint8x32 v_interleave_pairs(const v_uint8x32& vec) { return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x32 v_interleave_quads(const v_int8x32& vec)
{
    return v_int8x32(_mm256_shuffle_epi8(vec.val, _mm256_set_epi64x(0x0f0b0e0a0d090c08, 0x0703060205010400, 0x0f0b0e0a0d090c08, 0x0703060205010400)));
}
inline v_uint8x32 v_interleave_quads(const v_uint8x32& vec) { return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x16 v_interleave_pairs(const v_int16x16& vec)
{
    return v_int16x16(_mm256_shuffle_epi8(vec.val, _mm256_set_epi64x(0x0f0e0b0a0d0c0908, 0x0706030205040100, 0x0f0e0b0a0d0c0908, 0x0706030205040100)));
}
inline v_uint16x16 v_interleave_pairs(const v_uint16x16& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x16 v_interleave_quads(const v_int16x16& vec)
{
    return v_int16x16(_mm256_shuffle_epi8(vec.val, _mm256_set_epi64x(0x0f0e07060d0c0504, 0x0b0a030209080100, 0x0f0e07060d0c0504, 0x0b0a030209080100)));
}
inline v_uint16x16 v_interleave_quads(const v_uint16x16& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x8 v_interleave_pairs(const v_int32x8& vec)
{
    return v_int32x8(_mm256_shuffle_epi32(vec.val, _MM_SHUFFLE(3, 1, 2, 0)));
}
inline v_uint32x8 v_interleave_pairs(const v_uint32x8& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x8 v_interleave_pairs(const v_float32x8& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x32 v_pack_triplets(const v_int8x32& vec)
{
    return v_int8x32(_mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(vec.val, _mm256_broadcastsi128_si256(_mm_set_epi64x(0xffffff0f0e0d0c0a, 0x0908060504020100))),
                                                 _mm256_set_epi64x(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}
inline v_uint8x32 v_pack_triplets(const v_uint8x32& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x16 v_pack_triplets(const v_int16x16& vec)
{
    return v_int16x16(_mm256_permutevar8x32_epi32(_mm256_shuffle_epi8(vec.val, _mm256_broadcastsi128_si256(_mm_set_epi64x(0xffff0f0e0d0c0b0a, 0x0908050403020100))),
                                                  _mm256_set_epi64x(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}
inline v_uint16x16 v_pack_triplets(const v_uint16x16& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x8 v_pack_triplets(const v_int32x8& vec)
{
    return v_int32x8(_mm256_permutevar8x32_epi32(vec.val, _mm256_set_epi64x(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}
inline v_uint32x8 v_pack_triplets(const v_uint32x8& vec) { return v_reinterpret_as_u32(v_pack_triplets(v_reinterpret_as_s32(vec))); }
inline v_float32x8 v_pack_triplets(const v_float32x8& vec)
{
    return v_float32x8(_mm256_permutevar8x32_ps(vec.val, _mm256_set_epi64x(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}

////////// Matrix operations /////////

inline v_int32x8 v_dotprod(const v_int16x16& a, const v_int16x16& b)
{ return v_int32x8(_mm256_madd_epi16(a.val, b.val)); }

inline v_int32x8 v_dotprod(const v_int16x16& a, const v_int16x16& b, const v_int32x8& c)
{ return v_dotprod(a, b) + c; }

#define OPENCV_HAL_AVX_SPLAT2_PS(a, im) \
    v_float32x8(_mm256_permute_ps(a.val, _MM_SHUFFLE(im, im, im, im)))

inline v_float32x8 v_matmul(const v_float32x8& v, const v_float32x8& m0,
                            const v_float32x8& m1, const v_float32x8& m2,
                            const v_float32x8& m3)
{
    v_float32x8 v04 = OPENCV_HAL_AVX_SPLAT2_PS(v, 0);
    v_float32x8 v15 = OPENCV_HAL_AVX_SPLAT2_PS(v, 1);
    v_float32x8 v26 = OPENCV_HAL_AVX_SPLAT2_PS(v, 2);
    v_float32x8 v37 = OPENCV_HAL_AVX_SPLAT2_PS(v, 3);
    return v_fma(v04, m0, v_fma(v15, m1, v_fma(v26, m2, v37 * m3)));
}

inline v_float32x8 v_matmuladd(const v_float32x8& v, const v_float32x8& m0,
                               const v_float32x8& m1, const v_float32x8& m2,
                               const v_float32x8& a)
{
    v_float32x8 v04 = OPENCV_HAL_AVX_SPLAT2_PS(v, 0);
    v_float32x8 v15 = OPENCV_HAL_AVX_SPLAT2_PS(v, 1);
    v_float32x8 v26 = OPENCV_HAL_AVX_SPLAT2_PS(v, 2);
    return v_fma(v04, m0, v_fma(v15, m1, v_fma(v26, m2, a)));
}

#define OPENCV_HAL_IMPL_AVX_TRANSPOSE4x4(_Tpvec, suffix, cast_from, cast_to)    \
    inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1,              \
                               const _Tpvec& a2, const _Tpvec& a3,              \
                               _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3)  \
    {                                                                           \
        __m512i t0 = cast_from(_mm256_unpacklo_##suffix(a0.val, a1.val));       \
        __m512i t1 = cast_from(_mm256_unpacklo_##suffix(a2.val, a3.val));       \
        __m512i t2 = cast_from(_mm256_unpackhi_##suffix(a0.val, a1.val));       \
        __m512i t3 = cast_from(_mm256_unpackhi_##suffix(a2.val, a3.val));       \
        b0.val = cast_to(_mm256_unpacklo_epi64(t0, t1));                        \
        b1.val = cast_to(_mm256_unpackhi_epi64(t0, t1));                        \
        b2.val = cast_to(_mm256_unpacklo_epi64(t2, t3));                        \
        b3.val = cast_to(_mm256_unpackhi_epi64(t2, t3));                        \
    }

OPENCV_HAL_IMPL_AVX_TRANSPOSE4x4(v_uint32x8,  epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_AVX_TRANSPOSE4x4(v_int32x8,   epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_AVX_TRANSPOSE4x4(v_float32x8, ps, _mm256_castps_si256, _mm256_castsi256_ps)

//////////////// Value reordering ///////////////

/* Expand */
#define OPENCV_HAL_IMPL_AVX_EXPAND(_Tpvec, _Tpwvec, _Tp, intrin)    \
    inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
    {                                                               \
        b0.val = intrin(_v512_extract_low(a.val));                  \
        b1.val = intrin(_v512_extract_high(a.val));                 \
    }                                                               \
    inline _Tpwvec v_expand_low(const _Tpvec& a)                    \
    { return _Tpwvec(intrin(_v512_extract_low(a.val))); }           \
    inline _Tpwvec v_expand_high(const _Tpvec& a)                   \
    { return _Tpwvec(intrin(_v512_extract_high(a.val))); }          \
    inline _Tpwvec v512_load_expand(const _Tp* ptr)                 \
    {                                                               \
        __m256i a = _mm_loadu_si128((const __m256i*)ptr);           \
        return _Tpwvec(intrin(a));                                  \
    }

OPENCV_HAL_IMPL_AVX_EXPAND(v_uint8x32,  v_uint16x16, uchar,    _mm256_cvtepu8_epi16)
OPENCV_HAL_IMPL_AVX_EXPAND(v_int8x32,   v_int16x16,  schar,    _mm256_cvtepi8_epi16)
OPENCV_HAL_IMPL_AVX_EXPAND(v_uint16x16, v_uint32x8,  ushort,   _mm256_cvtepu16_epi32)
OPENCV_HAL_IMPL_AVX_EXPAND(v_int16x16,  v_int32x8,   short,    _mm256_cvtepi16_epi32)
OPENCV_HAL_IMPL_AVX_EXPAND(v_uint32x8,  v_uint64x4,  unsigned, _mm256_cvtepu32_epi64)
OPENCV_HAL_IMPL_AVX_EXPAND(v_int32x8,   v_int64x4,   int,      _mm256_cvtepi32_epi64)

#define OPENCV_HAL_IMPL_AVX_EXPAND_Q(_Tpvec, _Tp, intrin)   \
    inline _Tpvec v512_load_expand_q(const _Tp* ptr)        \
    {                                                       \
        __m256i a = _mm_loadl_epi64((const __m256i*)ptr);   \
        return _Tpvec(intrin(a));                           \
    }

OPENCV_HAL_IMPL_AVX_EXPAND_Q(v_uint32x8, uchar, _mm256_cvtepu8_epi32)
OPENCV_HAL_IMPL_AVX_EXPAND_Q(v_int32x8,  schar, _mm256_cvtepi8_epi32)

/* pack */
// 16
inline v_int8x32 v_pack(const v_int16x16& a, const v_int16x16& b)
{ return v_int8x32(_v512_shuffle_odd_64(_mm256_packs_epi16(a.val, b.val))); }

inline v_uint8x32 v_pack(const v_uint16x16& a, const v_uint16x16& b)
{
    __m512i t = _mm256_set1_epi16(255);
    __m512i a1 = _mm256_min_epu16(a.val, t);
    __m512i b1 = _mm256_min_epu16(b.val, t);
    return v_uint8x32(_v512_shuffle_odd_64(_mm256_packus_epi16(a1, b1)));
}

inline v_uint8x32 v_pack_u(const v_int16x16& a, const v_int16x16& b)
{
    return v_uint8x32(_v512_shuffle_odd_64(_mm256_packus_epi16(a.val, b.val)));
}

inline void v_pack_store(schar* ptr, const v_int16x16& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(uchar* ptr, const v_uint16x16& a)
{
    const __m512i m = _mm256_set1_epi16(255);
    __m512i am = _mm256_min_epu16(a.val, m);
            am =  _v512_shuffle_odd_64(_mm256_packus_epi16(am, am));
    v_store_low(ptr, v_uint8x32(am));
}

inline void v_pack_u_store(uchar* ptr, const v_int16x16& a)
{ v_store_low(ptr, v_pack_u(a, a)); }

template<int n> inline
v_uint8x32 v_rshr_pack(const v_uint16x16& a, const v_uint16x16& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    v_uint16x16 delta = v512_setall_u16((short)(1 << (n-1)));
    return v_pack_u(v_reinterpret_as_s16((a + delta) >> n),
                    v_reinterpret_as_s16((b + delta) >> n));
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x16& a)
{
    v_uint16x16 delta = v512_setall_u16((short)(1 << (n-1)));
    v_pack_u_store(ptr, v_reinterpret_as_s16((a + delta) >> n));
}

template<int n> inline
v_uint8x32 v_rshr_pack_u(const v_int16x16& a, const v_int16x16& b)
{
    v_int16x16 delta = v512_setall_s16((short)(1 << (n-1)));
    return v_pack_u((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x16& a)
{
    v_int16x16 delta = v512_setall_s16((short)(1 << (n-1)));
    v_pack_u_store(ptr, (a + delta) >> n);
}

template<int n> inline
v_int8x32 v_rshr_pack(const v_int16x16& a, const v_int16x16& b)
{
    v_int16x16 delta = v512_setall_s16((short)(1 << (n-1)));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x16& a)
{
    v_int16x16 delta = v512_setall_s16((short)(1 << (n-1)));
    v_pack_store(ptr, (a + delta) >> n);
}

// 32
inline v_int16x16 v_pack(const v_int32x8& a, const v_int32x8& b)
{ return v_int16x16(_v512_shuffle_odd_64(_mm256_packs_epi32(a.val, b.val))); }

inline v_uint16x16 v_pack(const v_uint32x8& a, const v_uint32x8& b)
{ return v_uint16x16(_v512_shuffle_odd_64(_v512_packs_epu32(a.val, b.val))); }

inline v_uint16x16 v_pack_u(const v_int32x8& a, const v_int32x8& b)
{ return v_uint16x16(_v512_shuffle_odd_64(_mm256_packus_epi32(a.val, b.val))); }

inline void v_pack_store(short* ptr, const v_int32x8& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(ushort* ptr, const v_uint32x8& a)
{
    const __m512i m = _mm256_set1_epi32(65535);
    __m512i am = _mm256_min_epu32(a.val, m);
            am = _v512_shuffle_odd_64(_mm256_packus_epi32(am, am));
    v_store_low(ptr, v_uint16x16(am));
}

inline void v_pack_u_store(ushort* ptr, const v_int32x8& a)
{ v_store_low(ptr, v_pack_u(a, a)); }


template<int n> inline
v_uint16x16 v_rshr_pack(const v_uint32x8& a, const v_uint32x8& b)
{
    // we assume that n > 0, and so the shifted 32-bit values can be treated as signed numbers.
    v_uint32x8 delta = v512_setall_u32(1 << (n-1));
    return v_pack_u(v_reinterpret_as_s32((a + delta) >> n),
                    v_reinterpret_as_s32((b + delta) >> n));
}

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x8& a)
{
    v_uint32x8 delta = v512_setall_u32(1 << (n-1));
    v_pack_u_store(ptr, v_reinterpret_as_s32((a + delta) >> n));
}

template<int n> inline
v_uint16x16 v_rshr_pack_u(const v_int32x8& a, const v_int32x8& b)
{
    v_int32x8 delta = v512_setall_s32(1 << (n-1));
    return v_pack_u((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x8& a)
{
    v_int32x8 delta = v512_setall_s32(1 << (n-1));
    v_pack_u_store(ptr, (a + delta) >> n);
}

template<int n> inline
v_int16x16 v_rshr_pack(const v_int32x8& a, const v_int32x8& b)
{
    v_int32x8 delta = v512_setall_s32(1 << (n-1));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x8& a)
{
    v_int32x8 delta = v512_setall_s32(1 << (n-1));
    v_pack_store(ptr, (a + delta) >> n);
}

// 64
// Non-saturating pack
inline v_uint32x8 v_pack(const v_uint64x4& a, const v_uint64x4& b)
{
    __m512i a0 = _mm256_shuffle_epi32(a.val, _MM_SHUFFLE(0, 0, 2, 0));
    __m512i b0 = _mm256_shuffle_epi32(b.val, _MM_SHUFFLE(0, 0, 2, 0));
    __m512i ab = _mm256_unpacklo_epi64(a0, b0); // a0, a1, b0, b1, a2, a3, b2, b3
    return v_uint32x8(_v512_shuffle_odd_64(ab));
}

inline v_int32x8 v_pack(const v_int64x4& a, const v_int64x4& b)
{ return v_reinterpret_as_s32(v_pack(v_reinterpret_as_u64(a), v_reinterpret_as_u64(b))); }

inline void v_pack_store(unsigned* ptr, const v_uint64x4& a)
{
    __m512i a0 = _mm256_shuffle_epi32(a.val, _MM_SHUFFLE(0, 0, 2, 0));
    v_store_low(ptr, v_uint32x8(_v512_shuffle_odd_64(a0)));
}

inline void v_pack_store(int* ptr, const v_int64x4& b)
{ v_pack_store((unsigned*)ptr, v_reinterpret_as_u64(b)); }

template<int n> inline
v_uint32x8 v_rshr_pack(const v_uint64x4& a, const v_uint64x4& b)
{
    v_uint64x4 delta = v512_setall_u64((uint64)1 << (n-1));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x4& a)
{
    v_uint64x4 delta = v512_setall_u64((uint64)1 << (n-1));
    v_pack_store(ptr, (a + delta) >> n);
}

template<int n> inline
v_int32x8 v_rshr_pack(const v_int64x4& a, const v_int64x4& b)
{
    v_int64x4 delta = v512_setall_s64((int64)1 << (n-1));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x4& a)
{
    v_int64x4 delta = v512_setall_s64((int64)1 << (n-1));
    v_pack_store(ptr, (a + delta) >> n);
}

// pack boolean
inline v_uint8x32 v_pack_b(const v_uint16x16& a, const v_uint16x16& b)
{
    __m512i ab = _mm256_packs_epi16(a.val, b.val);
    return v_uint8x32(_v512_shuffle_odd_64(ab));
}

inline v_uint8x32 v_pack_b(const v_uint32x8& a, const v_uint32x8& b,
                           const v_uint32x8& c, const v_uint32x8& d)
{
    __m512i ab = _mm256_packs_epi32(a.val, b.val);
    __m512i cd = _mm256_packs_epi32(c.val, d.val);

    __m512i abcd = _v512_shuffle_odd_64(_mm256_packs_epi16(ab, cd));
    return v_uint8x32(_mm256_shuffle_epi32(abcd, _MM_SHUFFLE(3, 1, 2, 0)));
}

inline v_uint8x32 v_pack_b(const v_uint64x4& a, const v_uint64x4& b, const v_uint64x4& c,
                           const v_uint64x4& d, const v_uint64x4& e, const v_uint64x4& f,
                           const v_uint64x4& g, const v_uint64x4& h)
{
    __m512i ab = _mm256_packs_epi32(a.val, b.val);
    __m512i cd = _mm256_packs_epi32(c.val, d.val);
    __m512i ef = _mm256_packs_epi32(e.val, f.val);
    __m512i gh = _mm256_packs_epi32(g.val, h.val);

    __m512i abcd = _mm256_packs_epi32(ab, cd);
    __m512i efgh = _mm256_packs_epi32(ef, gh);
    __m512i pkall = _v512_shuffle_odd_64(_mm256_packs_epi16(abcd, efgh));

    __m512i rev = _mm256_alignr_epi8(pkall, pkall, 8);
    return v_uint8x32(_mm256_unpacklo_epi16(pkall, rev));
}

/* Recombine */
// its up there with load and store operations

/* Extract */
#define OPENCV_HAL_IMPL_AVX_EXTRACT(_Tpvec)                    \
    template<int s>                                            \
    inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)  \
    { return v_rotate_right<s>(a, b); }

OPENCV_HAL_IMPL_AVX_EXTRACT(v_uint8x32)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_int8x32)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_uint16x16)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_int16x16)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_uint32x8)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_int32x8)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_uint64x4)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_int64x4)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_float32x8)
OPENCV_HAL_IMPL_AVX_EXTRACT(v_float64x4)


///////////////////// load deinterleave /////////////////////////////

inline void v_load_deinterleave( const uchar* ptr, v_uint8x32& a, v_uint8x32& b )
{
    __m512i ab0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i ab1 = _mm256_loadu_si256((const __m512i*)(ptr + 32));

    const __m512i sh = _mm256_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
                                               0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15);
    __m512i p0 = _mm256_shuffle_epi8(ab0, sh);
    __m512i p1 = _mm256_shuffle_epi8(ab1, sh);
    __m512i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2*16);
    __m512i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3*16);
    __m512i a0 = _mm256_unpacklo_epi64(pl, ph);
    __m512i b0 = _mm256_unpackhi_epi64(pl, ph);
    a = v_uint8x32(a0);
    b = v_uint8x32(b0);
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x16& a, v_uint16x16& b )
{
    __m512i ab0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i ab1 = _mm256_loadu_si256((const __m512i*)(ptr + 16));

    const __m512i sh = _mm256_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
                                               0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15);
    __m512i p0 = _mm256_shuffle_epi8(ab0, sh);
    __m512i p1 = _mm256_shuffle_epi8(ab1, sh);
    __m512i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2*16);
    __m512i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3*16);
    __m512i a0 = _mm256_unpacklo_epi64(pl, ph);
    __m512i b0 = _mm256_unpackhi_epi64(pl, ph);
    a = v_uint16x16(a0);
    b = v_uint16x16(b0);
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x8& a, v_uint32x8& b )
{
    __m512i ab0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i ab1 = _mm256_loadu_si256((const __m512i*)(ptr + 8));

    const int sh = 0+2*4+1*16+3*64;
    __m512i p0 = _mm256_shuffle_epi32(ab0, sh);
    __m512i p1 = _mm256_shuffle_epi32(ab1, sh);
    __m512i pl = _mm256_permute2x128_si256(p0, p1, 0 + 2*16);
    __m512i ph = _mm256_permute2x128_si256(p0, p1, 1 + 3*16);
    __m512i a0 = _mm256_unpacklo_epi64(pl, ph);
    __m512i b0 = _mm256_unpackhi_epi64(pl, ph);
    a = v_uint32x8(a0);
    b = v_uint32x8(b0);
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x4& a, v_uint64x4& b )
{
    __m512i ab0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i ab1 = _mm256_loadu_si256((const __m512i*)(ptr + 4));

    __m512i pl = _mm256_permute2x128_si256(ab0, ab1, 0 + 2*16);
    __m512i ph = _mm256_permute2x128_si256(ab0, ab1, 1 + 3*16);
    __m512i a0 = _mm256_unpacklo_epi64(pl, ph);
    __m512i b0 = _mm256_unpackhi_epi64(pl, ph);
    a = v_uint64x4(a0);
    b = v_uint64x4(b0);
}

inline void v_load_deinterleave( const uchar* ptr, v_uint8x32& b, v_uint8x32& g, v_uint8x32& r )
{
    __m512i bgr0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgr1 = _mm256_loadu_si256((const __m512i*)(ptr + 32));
    __m512i bgr2 = _mm256_loadu_si256((const __m512i*)(ptr + 64));

    __m512i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2*16);
    __m512i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3*16);

    const __m512i m0 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
                                               0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    const __m512i m1 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
                                               -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

    __m512i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
    __m512i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);
    __m512i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);

    const __m512i
    sh_b = _mm256_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
                            0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13),
    sh_g = _mm256_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
                            1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14),
    sh_r = _mm256_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
                            2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    b0 = _mm256_shuffle_epi8(b0, sh_b);
    g0 = _mm256_shuffle_epi8(g0, sh_g);
    r0 = _mm256_shuffle_epi8(r0, sh_r);

    b = v_uint8x32(b0);
    g = v_uint8x32(g0);
    r = v_uint8x32(r0);
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x16& b, v_uint16x16& g, v_uint16x16& r )
{
    __m512i bgr0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgr1 = _mm256_loadu_si256((const __m512i*)(ptr + 16));
    __m512i bgr2 = _mm256_loadu_si256((const __m512i*)(ptr + 32));

    __m512i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2*16);
    __m512i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3*16);

    const __m512i m0 = _mm256_setr_epi8(0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
                                               0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);
    const __m512i m1 = _mm256_setr_epi8(0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
                                               -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);
    __m512i b0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_low, s02_high, m0), bgr1, m1);
    __m512i g0 = _mm256_blendv_epi8(_mm256_blendv_epi8(bgr1, s02_low, m0), s02_high, m1);
    __m512i r0 = _mm256_blendv_epi8(_mm256_blendv_epi8(s02_high, s02_low, m1), bgr1, m0);
    const __m512i sh_b = _mm256_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
                                                 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m512i sh_g = _mm256_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13,
                                                 2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    const __m512i sh_r = _mm256_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
                                                 4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    b0 = _mm256_shuffle_epi8(b0, sh_b);
    g0 = _mm256_shuffle_epi8(g0, sh_g);
    r0 = _mm256_shuffle_epi8(r0, sh_r);

    b = v_uint16x16(b0);
    g = v_uint16x16(g0);
    r = v_uint16x16(r0);
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x8& b, v_uint32x8& g, v_uint32x8& r )
{
    __m512i bgr0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgr1 = _mm256_loadu_si256((const __m512i*)(ptr + 8));
    __m512i bgr2 = _mm256_loadu_si256((const __m512i*)(ptr + 16));

    __m512i s02_low = _mm256_permute2x128_si256(bgr0, bgr2, 0 + 2*16);
    __m512i s02_high = _mm256_permute2x128_si256(bgr0, bgr2, 1 + 3*16);

    __m512i b0 = _mm256_blend_epi32(_mm256_blend_epi32(s02_low, s02_high, 0x24), bgr1, 0x92);
    __m512i g0 = _mm256_blend_epi32(_mm256_blend_epi32(s02_high, s02_low, 0x92), bgr1, 0x24);
    __m512i r0 = _mm256_blend_epi32(_mm256_blend_epi32(bgr1, s02_low, 0x24), s02_high, 0x92);

    b0 = _mm256_shuffle_epi32(b0, 0x6c);
    g0 = _mm256_shuffle_epi32(g0, 0xb1);
    r0 = _mm256_shuffle_epi32(r0, 0xc6);

    b = v_uint32x8(b0);
    g = v_uint32x8(g0);
    r = v_uint32x8(r0);
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x4& b, v_uint64x4& g, v_uint64x4& r )
{
    __m512i bgr0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgr1 = _mm256_loadu_si256((const __m512i*)(ptr + 4));
    __m512i bgr2 = _mm256_loadu_si256((const __m512i*)(ptr + 8));

    __m512i s01 = _mm256_blend_epi32(bgr0, bgr1, 0xf0);
    __m512i s12 = _mm256_blend_epi32(bgr1, bgr2, 0xf0);
    __m512i s20r = _mm256_permute4x64_epi64(_mm256_blend_epi32(bgr2, bgr0, 0xf0), 0x1b);
    __m512i b0 = _mm256_unpacklo_epi64(s01, s20r);
    __m512i g0 = _mm256_alignr_epi8(s12, s01, 8);
    __m512i r0 = _mm256_unpackhi_epi64(s20r, s12);

    b = v_uint64x4(b0);
    g = v_uint64x4(g0);
    r = v_uint64x4(r0);
}

inline void v_load_deinterleave( const uchar* ptr, v_uint8x32& b, v_uint8x32& g, v_uint8x32& r, v_uint8x32& a )
{
    __m512i bgr0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgr1 = _mm256_loadu_si256((const __m512i*)(ptr + 32));
    __m512i bgr2 = _mm256_loadu_si256((const __m512i*)(ptr + 64));
    __m512i bgr3 = _mm256_loadu_si256((const __m512i*)(ptr + 96));
    const __m512i sh = _mm256_setr_epi8(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
                                               0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);

    __m512i p0 = _mm256_shuffle_epi8(bgr0, sh);
    __m512i p1 = _mm256_shuffle_epi8(bgr1, sh);
    __m512i p2 = _mm256_shuffle_epi8(bgr2, sh);
    __m512i p3 = _mm256_shuffle_epi8(bgr3, sh);

    __m512i p01l = _mm256_unpacklo_epi32(p0, p1);
    __m512i p01h = _mm256_unpackhi_epi32(p0, p1);
    __m512i p23l = _mm256_unpacklo_epi32(p2, p3);
    __m512i p23h = _mm256_unpackhi_epi32(p2, p3);

    __m512i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2*16);
    __m512i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3*16);
    __m512i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2*16);
    __m512i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3*16);

    __m512i b0 = _mm256_unpacklo_epi32(pll, plh);
    __m512i g0 = _mm256_unpackhi_epi32(pll, plh);
    __m512i r0 = _mm256_unpacklo_epi32(phl, phh);
    __m512i a0 = _mm256_unpackhi_epi32(phl, phh);

    b = v_uint8x32(b0);
    g = v_uint8x32(g0);
    r = v_uint8x32(r0);
    a = v_uint8x32(a0);
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x16& b, v_uint16x16& g, v_uint16x16& r, v_uint16x16& a )
{
    __m512i bgr0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgr1 = _mm256_loadu_si256((const __m512i*)(ptr + 16));
    __m512i bgr2 = _mm256_loadu_si256((const __m512i*)(ptr + 32));
    __m512i bgr3 = _mm256_loadu_si256((const __m512i*)(ptr + 48));
    const __m512i sh = _mm256_setr_epi8(0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15,
                                               0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15);
    __m512i p0 = _mm256_shuffle_epi8(bgr0, sh);
    __m512i p1 = _mm256_shuffle_epi8(bgr1, sh);
    __m512i p2 = _mm256_shuffle_epi8(bgr2, sh);
    __m512i p3 = _mm256_shuffle_epi8(bgr3, sh);

    __m512i p01l = _mm256_unpacklo_epi32(p0, p1);
    __m512i p01h = _mm256_unpackhi_epi32(p0, p1);
    __m512i p23l = _mm256_unpacklo_epi32(p2, p3);
    __m512i p23h = _mm256_unpackhi_epi32(p2, p3);

    __m512i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2*16);
    __m512i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3*16);
    __m512i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2*16);
    __m512i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3*16);

    __m512i b0 = _mm256_unpacklo_epi32(pll, plh);
    __m512i g0 = _mm256_unpackhi_epi32(pll, plh);
    __m512i r0 = _mm256_unpacklo_epi32(phl, phh);
    __m512i a0 = _mm256_unpackhi_epi32(phl, phh);

    b = v_uint16x16(b0);
    g = v_uint16x16(g0);
    r = v_uint16x16(r0);
    a = v_uint16x16(a0);
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x8& b, v_uint32x8& g, v_uint32x8& r, v_uint32x8& a )
{
    __m512i p0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i p1 = _mm256_loadu_si256((const __m512i*)(ptr + 8));
    __m512i p2 = _mm256_loadu_si256((const __m512i*)(ptr + 16));
    __m512i p3 = _mm256_loadu_si256((const __m512i*)(ptr + 24));

    __m512i p01l = _mm256_unpacklo_epi32(p0, p1);
    __m512i p01h = _mm256_unpackhi_epi32(p0, p1);
    __m512i p23l = _mm256_unpacklo_epi32(p2, p3);
    __m512i p23h = _mm256_unpackhi_epi32(p2, p3);

    __m512i pll = _mm256_permute2x128_si256(p01l, p23l, 0 + 2*16);
    __m512i plh = _mm256_permute2x128_si256(p01l, p23l, 1 + 3*16);
    __m512i phl = _mm256_permute2x128_si256(p01h, p23h, 0 + 2*16);
    __m512i phh = _mm256_permute2x128_si256(p01h, p23h, 1 + 3*16);

    __m512i b0 = _mm256_unpacklo_epi32(pll, plh);
    __m512i g0 = _mm256_unpackhi_epi32(pll, plh);
    __m512i r0 = _mm256_unpacklo_epi32(phl, phh);
    __m512i a0 = _mm256_unpackhi_epi32(phl, phh);

    b = v_uint32x8(b0);
    g = v_uint32x8(g0);
    r = v_uint32x8(r0);
    a = v_uint32x8(a0);
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x4& b, v_uint64x4& g, v_uint64x4& r, v_uint64x4& a )
{
    __m512i bgra0 = _mm256_loadu_si256((const __m512i*)ptr);
    __m512i bgra1 = _mm256_loadu_si256((const __m512i*)(ptr + 4));
    __m512i bgra2 = _mm256_loadu_si256((const __m512i*)(ptr + 8));
    __m512i bgra3 = _mm256_loadu_si256((const __m512i*)(ptr + 12));

    __m512i l02 = _mm256_permute2x128_si256(bgra0, bgra2, 0 + 2*16);
    __m512i h02 = _mm256_permute2x128_si256(bgra0, bgra2, 1 + 3*16);
    __m512i l13 = _mm256_permute2x128_si256(bgra1, bgra3, 0 + 2*16);
    __m512i h13 = _mm256_permute2x128_si256(bgra1, bgra3, 1 + 3*16);

    __m512i b0 = _mm256_unpacklo_epi64(l02, l13);
    __m512i g0 = _mm256_unpackhi_epi64(l02, l13);
    __m512i r0 = _mm256_unpacklo_epi64(h02, h13);
    __m512i a0 = _mm256_unpackhi_epi64(h02, h13);

    b = v_uint64x4(b0);
    g = v_uint64x4(g0);
    r = v_uint64x4(r0);
    a = v_uint64x4(a0);
}

///////////////////////////// store interleave /////////////////////////////////////

inline void v_store_interleave( uchar* ptr, const v_uint8x32& x, const v_uint8x32& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i xy_l = _mm256_unpacklo_epi8(x.val, y.val);
    __m512i xy_h = _mm256_unpackhi_epi8(x.val, y.val);

    __m512i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2*16);
    __m512i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, xy0);
        _mm256_stream_si256((__m512i*)(ptr + 32), xy1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, xy0);
        _mm256_store_si256((__m512i*)(ptr + 32), xy1);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, xy0);
        _mm256_storeu_si256((__m512i*)(ptr + 32), xy1);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x16& x, const v_uint16x16& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i xy_l = _mm256_unpacklo_epi16(x.val, y.val);
    __m512i xy_h = _mm256_unpackhi_epi16(x.val, y.val);

    __m512i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2*16);
    __m512i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, xy0);
        _mm256_stream_si256((__m512i*)(ptr + 16), xy1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, xy0);
        _mm256_store_si256((__m512i*)(ptr + 16), xy1);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, xy0);
        _mm256_storeu_si256((__m512i*)(ptr + 16), xy1);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x8& x, const v_uint32x8& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i xy_l = _mm256_unpacklo_epi32(x.val, y.val);
    __m512i xy_h = _mm256_unpackhi_epi32(x.val, y.val);

    __m512i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2*16);
    __m512i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, xy0);
        _mm256_stream_si256((__m512i*)(ptr + 8), xy1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, xy0);
        _mm256_store_si256((__m512i*)(ptr + 8), xy1);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, xy0);
        _mm256_storeu_si256((__m512i*)(ptr + 8), xy1);
    }
}

inline void v_store_interleave( uint64* ptr, const v_uint64x4& x, const v_uint64x4& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i xy_l = _mm256_unpacklo_epi64(x.val, y.val);
    __m512i xy_h = _mm256_unpackhi_epi64(x.val, y.val);

    __m512i xy0 = _mm256_permute2x128_si256(xy_l, xy_h, 0 + 2*16);
    __m512i xy1 = _mm256_permute2x128_si256(xy_l, xy_h, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, xy0);
        _mm256_stream_si256((__m512i*)(ptr + 4), xy1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, xy0);
        _mm256_store_si256((__m512i*)(ptr + 4), xy1);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, xy0);
        _mm256_storeu_si256((__m512i*)(ptr + 4), xy1);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x32& b, const v_uint8x32& g, const v_uint8x32& r,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    const __m512i sh_b = _mm256_setr_epi8(
            0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
            0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    const __m512i sh_g = _mm256_setr_epi8(
            5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
            5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    const __m512i sh_r = _mm256_setr_epi8(
            10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
            10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

    __m512i b0 = _mm256_shuffle_epi8(b.val, sh_b);
    __m512i g0 = _mm256_shuffle_epi8(g.val, sh_g);
    __m512i r0 = _mm256_shuffle_epi8(r.val, sh_r);

    const __m512i m0 = _mm256_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
                                               0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    const __m512i m1 = _mm256_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
                                               0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);

    __m512i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
    __m512i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
    __m512i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

    __m512i bgr0 = _mm256_permute2x128_si256(p0, p1, 0 + 2*16);
    __m512i bgr1 = _mm256_permute2x128_si256(p2, p0, 0 + 3*16);
    __m512i bgr2 = _mm256_permute2x128_si256(p1, p2, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgr0);
        _mm256_stream_si256((__m512i*)(ptr + 32), bgr1);
        _mm256_stream_si256((__m512i*)(ptr + 64), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgr0);
        _mm256_store_si256((__m512i*)(ptr + 32), bgr1);
        _mm256_store_si256((__m512i*)(ptr + 64), bgr2);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgr0);
        _mm256_storeu_si256((__m512i*)(ptr + 32), bgr1);
        _mm256_storeu_si256((__m512i*)(ptr + 64), bgr2);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x16& b, const v_uint16x16& g, const v_uint16x16& r,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    const __m512i sh_b = _mm256_setr_epi8(
         0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
         0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m512i sh_g = _mm256_setr_epi8(
         10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5,
         10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    const __m512i sh_r = _mm256_setr_epi8(
         4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
         4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

    __m512i b0 = _mm256_shuffle_epi8(b.val, sh_b);
    __m512i g0 = _mm256_shuffle_epi8(g.val, sh_g);
    __m512i r0 = _mm256_shuffle_epi8(r.val, sh_r);

    const __m512i m0 = _mm256_setr_epi8(0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
                                               0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);
    const __m512i m1 = _mm256_setr_epi8(0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
                                               -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);

    __m512i p0 = _mm256_blendv_epi8(_mm256_blendv_epi8(b0, g0, m0), r0, m1);
    __m512i p1 = _mm256_blendv_epi8(_mm256_blendv_epi8(g0, r0, m0), b0, m1);
    __m512i p2 = _mm256_blendv_epi8(_mm256_blendv_epi8(r0, b0, m0), g0, m1);

    __m512i bgr0 = _mm256_permute2x128_si256(p0, p2, 0 + 2*16);
    //__m512i bgr1 = p1;
    __m512i bgr2 = _mm256_permute2x128_si256(p0, p2, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgr0);
        _mm256_stream_si256((__m512i*)(ptr + 16), p1);
        _mm256_stream_si256((__m512i*)(ptr + 32), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgr0);
        _mm256_store_si256((__m512i*)(ptr + 16), p1);
        _mm256_store_si256((__m512i*)(ptr + 32), bgr2);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgr0);
        _mm256_storeu_si256((__m512i*)(ptr + 16), p1);
        _mm256_storeu_si256((__m512i*)(ptr + 32), bgr2);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x8& b, const v_uint32x8& g, const v_uint32x8& r,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i b0 = _mm256_shuffle_epi32(b.val, 0x6c);
    __m512i g0 = _mm256_shuffle_epi32(g.val, 0xb1);
    __m512i r0 = _mm256_shuffle_epi32(r.val, 0xc6);

    __m512i p0 = _mm256_blend_epi32(_mm256_blend_epi32(b0, g0, 0x92), r0, 0x24);
    __m512i p1 = _mm256_blend_epi32(_mm256_blend_epi32(g0, r0, 0x92), b0, 0x24);
    __m512i p2 = _mm256_blend_epi32(_mm256_blend_epi32(r0, b0, 0x92), g0, 0x24);

    __m512i bgr0 = _mm256_permute2x128_si256(p0, p1, 0 + 2*16);
    //__m512i bgr1 = p2;
    __m512i bgr2 = _mm256_permute2x128_si256(p0, p1, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgr0);
        _mm256_stream_si256((__m512i*)(ptr + 8), p2);
        _mm256_stream_si256((__m512i*)(ptr + 16), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgr0);
        _mm256_store_si256((__m512i*)(ptr + 8), p2);
        _mm256_store_si256((__m512i*)(ptr + 16), bgr2);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgr0);
        _mm256_storeu_si256((__m512i*)(ptr + 8), p2);
        _mm256_storeu_si256((__m512i*)(ptr + 16), bgr2);
    }
}

inline void v_store_interleave( uint64* ptr, const v_uint64x4& b, const v_uint64x4& g, const v_uint64x4& r,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i s01 = _mm256_unpacklo_epi64(b.val, g.val);
    __m512i s12 = _mm256_unpackhi_epi64(g.val, r.val);
    __m512i s20 = _mm256_blend_epi32(r.val, b.val, 0xcc);

    __m512i bgr0 = _mm256_permute2x128_si256(s01, s20, 0 + 2*16);
    __m512i bgr1 = _mm256_blend_epi32(s01, s12, 0x0f);
    __m512i bgr2 = _mm256_permute2x128_si256(s20, s12, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgr0);
        _mm256_stream_si256((__m512i*)(ptr + 4), bgr1);
        _mm256_stream_si256((__m512i*)(ptr + 8), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgr0);
        _mm256_store_si256((__m512i*)(ptr + 4), bgr1);
        _mm256_store_si256((__m512i*)(ptr + 8), bgr2);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgr0);
        _mm256_storeu_si256((__m512i*)(ptr + 4), bgr1);
        _mm256_storeu_si256((__m512i*)(ptr + 8), bgr2);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x32& b, const v_uint8x32& g,
                                const v_uint8x32& r, const v_uint8x32& a,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i bg0 = _mm256_unpacklo_epi8(b.val, g.val);
    __m512i bg1 = _mm256_unpackhi_epi8(b.val, g.val);
    __m512i ra0 = _mm256_unpacklo_epi8(r.val, a.val);
    __m512i ra1 = _mm256_unpackhi_epi8(r.val, a.val);

    __m512i bgra0_ = _mm256_unpacklo_epi16(bg0, ra0);
    __m512i bgra1_ = _mm256_unpackhi_epi16(bg0, ra0);
    __m512i bgra2_ = _mm256_unpacklo_epi16(bg1, ra1);
    __m512i bgra3_ = _mm256_unpackhi_epi16(bg1, ra1);

    __m512i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2*16);
    __m512i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3*16);
    __m512i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2*16);
    __m512i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgra0);
        _mm256_stream_si256((__m512i*)(ptr + 32), bgra1);
        _mm256_stream_si256((__m512i*)(ptr + 64), bgra2);
        _mm256_stream_si256((__m512i*)(ptr + 96), bgra3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgra0);
        _mm256_store_si256((__m512i*)(ptr + 32), bgra1);
        _mm256_store_si256((__m512i*)(ptr + 64), bgra2);
        _mm256_store_si256((__m512i*)(ptr + 96), bgra3);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgra0);
        _mm256_storeu_si256((__m512i*)(ptr + 32), bgra1);
        _mm256_storeu_si256((__m512i*)(ptr + 64), bgra2);
        _mm256_storeu_si256((__m512i*)(ptr + 96), bgra3);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x16& b, const v_uint16x16& g,
                                const v_uint16x16& r, const v_uint16x16& a,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i bg0 = _mm256_unpacklo_epi16(b.val, g.val);
    __m512i bg1 = _mm256_unpackhi_epi16(b.val, g.val);
    __m512i ra0 = _mm256_unpacklo_epi16(r.val, a.val);
    __m512i ra1 = _mm256_unpackhi_epi16(r.val, a.val);

    __m512i bgra0_ = _mm256_unpacklo_epi32(bg0, ra0);
    __m512i bgra1_ = _mm256_unpackhi_epi32(bg0, ra0);
    __m512i bgra2_ = _mm256_unpacklo_epi32(bg1, ra1);
    __m512i bgra3_ = _mm256_unpackhi_epi32(bg1, ra1);

    __m512i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2*16);
    __m512i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3*16);
    __m512i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2*16);
    __m512i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgra0);
        _mm256_stream_si256((__m512i*)(ptr + 16), bgra1);
        _mm256_stream_si256((__m512i*)(ptr + 32), bgra2);
        _mm256_stream_si256((__m512i*)(ptr + 48), bgra3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgra0);
        _mm256_store_si256((__m512i*)(ptr + 16), bgra1);
        _mm256_store_si256((__m512i*)(ptr + 32), bgra2);
        _mm256_store_si256((__m512i*)(ptr + 48), bgra3);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgra0);
        _mm256_storeu_si256((__m512i*)(ptr + 16), bgra1);
        _mm256_storeu_si256((__m512i*)(ptr + 32), bgra2);
        _mm256_storeu_si256((__m512i*)(ptr + 48), bgra3);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x8& b, const v_uint32x8& g,
                                const v_uint32x8& r, const v_uint32x8& a,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i bg0 = _mm256_unpacklo_epi32(b.val, g.val);
    __m512i bg1 = _mm256_unpackhi_epi32(b.val, g.val);
    __m512i ra0 = _mm256_unpacklo_epi32(r.val, a.val);
    __m512i ra1 = _mm256_unpackhi_epi32(r.val, a.val);

    __m512i bgra0_ = _mm256_unpacklo_epi64(bg0, ra0);
    __m512i bgra1_ = _mm256_unpackhi_epi64(bg0, ra0);
    __m512i bgra2_ = _mm256_unpacklo_epi64(bg1, ra1);
    __m512i bgra3_ = _mm256_unpackhi_epi64(bg1, ra1);

    __m512i bgra0 = _mm256_permute2x128_si256(bgra0_, bgra1_, 0 + 2*16);
    __m512i bgra2 = _mm256_permute2x128_si256(bgra0_, bgra1_, 1 + 3*16);
    __m512i bgra1 = _mm256_permute2x128_si256(bgra2_, bgra3_, 0 + 2*16);
    __m512i bgra3 = _mm256_permute2x128_si256(bgra2_, bgra3_, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgra0);
        _mm256_stream_si256((__m512i*)(ptr + 8), bgra1);
        _mm256_stream_si256((__m512i*)(ptr + 16), bgra2);
        _mm256_stream_si256((__m512i*)(ptr + 24), bgra3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgra0);
        _mm256_store_si256((__m512i*)(ptr + 8), bgra1);
        _mm256_store_si256((__m512i*)(ptr + 16), bgra2);
        _mm256_store_si256((__m512i*)(ptr + 24), bgra3);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgra0);
        _mm256_storeu_si256((__m512i*)(ptr + 8), bgra1);
        _mm256_storeu_si256((__m512i*)(ptr + 16), bgra2);
        _mm256_storeu_si256((__m512i*)(ptr + 24), bgra3);
    }
}

inline void v_store_interleave( uint64* ptr, const v_uint64x4& b, const v_uint64x4& g,
                                const v_uint64x4& r, const v_uint64x4& a,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i bg0 = _mm256_unpacklo_epi64(b.val, g.val);
    __m512i bg1 = _mm256_unpackhi_epi64(b.val, g.val);
    __m512i ra0 = _mm256_unpacklo_epi64(r.val, a.val);
    __m512i ra1 = _mm256_unpackhi_epi64(r.val, a.val);

    __m512i bgra0 = _mm256_permute2x128_si256(bg0, ra0, 0 + 2*16);
    __m512i bgra1 = _mm256_permute2x128_si256(bg1, ra1, 0 + 2*16);
    __m512i bgra2 = _mm256_permute2x128_si256(bg0, ra0, 1 + 3*16);
    __m512i bgra3 = _mm256_permute2x128_si256(bg1, ra1, 1 + 3*16);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm256_stream_si256((__m512i*)ptr, bgra0);
        _mm256_stream_si256((__m512i*)(ptr + 4), bgra1);
        _mm256_stream_si256((__m512i*)(ptr + 8), bgra2);
        _mm256_stream_si256((__m512i*)(ptr + 12), bgra3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm256_store_si256((__m512i*)ptr, bgra0);
        _mm256_store_si256((__m512i*)(ptr + 4), bgra1);
        _mm256_store_si256((__m512i*)(ptr + 8), bgra2);
        _mm256_store_si256((__m512i*)(ptr + 12), bgra3);
    }
    else
    {
        _mm256_storeu_si256((__m512i*)ptr, bgra0);
        _mm256_storeu_si256((__m512i*)(ptr + 4), bgra1);
        _mm256_storeu_si256((__m512i*)(ptr + 8), bgra2);
        _mm256_storeu_si256((__m512i*)(ptr + 12), bgra3);
    }
}

#define OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1) \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0 ) \
{ \
    _Tpvec1 a1, b1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0 ) \
{ \
    _Tpvec1 a1, b1, c1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0, _Tpvec0& d0 ) \
{ \
    _Tpvec1 a1, b1, c1, d1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1, d1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
    d0 = v_reinterpret_as_##suffix0(d1); \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                hal::StoreMode mode=hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, mode);      \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, const _Tpvec0& c0, \
                                hal::StoreMode mode=hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, mode);  \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, const _Tpvec0& d0, \
                                hal::StoreMode mode=hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    _Tpvec1 d1 = v_reinterpret_as_##suffix1(d0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1, mode); \
}

OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(v_int8x32, schar, s8, v_uint8x32, uchar, u8)
OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(v_int16x16, short, s16, v_uint16x16, ushort, u16)
OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(v_int32x8, int, s32, v_uint32x8, unsigned, u32)
OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(v_float32x8, float, f32, v_uint32x8, unsigned, u32)
OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(v_int64x4, int64, s64, v_uint64x4, uint64, u64)
OPENCV_HAL_IMPL_AVX_LOADSTORE_INTERLEAVE(v_float64x4, double, f64, v_uint64x4, uint64, u64)

inline void v512_cleanup() { _mm256_zeroall(); }

//! @name Check SIMD256 support
//! @{
//! @brief Check CPU capability of SIMD operation
static inline bool hasSIMD512()
{
    return (CV_CPU_HAS_SUPPORT_AVX512_SKX) ? true : false;
}
//! @}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // OPENCV_HAL_INTRIN_AVX_HPP
