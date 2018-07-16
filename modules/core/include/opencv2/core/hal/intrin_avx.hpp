// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_INTRIN_AVX_HPP
#define OPENCV_HAL_INTRIN_AVX_HPP

#define CV_SIMD256 1
#define CV_SIMD256_64F 1

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Utils ////////////

inline __m256i _v256_combine(const __m128i& lo, const __m128i& hi)
{ return _mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1); }

inline __m256 _v256_combine(const __m128& lo, const __m128& hi)
{ return _mm256_insertf128_ps(_mm256_castps128_ps256(lo), hi, 1); }

inline __m256d _v256_combine(const __m128d& lo, const __m128d& hi)
{ return _mm256_insertf128_pd(_mm256_castpd128_pd256(lo), hi, 1); }

inline int _v_cvtsi256_si32(const __m256i& a)
{ return _mm_cvtsi128_si32(_mm256_castsi256_si128(a)); }

inline __m256i _v256_shuffle_odd_64(const __m256i& v)
{ return _mm256_permute4x64_epi64(v, _MM_SHUFFLE(3, 1, 2, 0)); }

inline __m256d _v256_shuffle_odd_64(const __m256d& v)
{ return _mm256_permute4x64_pd(v, _MM_SHUFFLE(3, 1, 2, 0)); }

template<int imm>
inline __m256i _v256_permute2x128(const __m256i& a, const __m256i& b)
{ return _mm256_permute2x128_si256(a, b, imm); }

template<int imm>
inline __m256 _v256_permute2x128(const __m256& a, const __m256& b)
{ return _mm256_permute2f128_ps(a, b, imm); }

template<int imm>
inline __m256d _v256_permute2x128(const __m256d& a, const __m256d& b)
{ return _mm256_permute2f128_pd(a, b, imm); }

template<int imm, typename _Tpvec>
inline _Tpvec v256_permute2x128(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_v256_permute2x128<imm>(a.val, b.val)); }

template<int imm>
inline __m256i _v256_permute4x64(const __m256i& a)
{ return _mm256_permute4x64_epi64(a, imm); }

template<int imm>
inline __m256d _v256_permute4x64(const __m256d& a)
{ return _mm256_permute4x64_pd(a, imm); }

template<int imm, typename _Tpvec>
inline _Tpvec v256_permute4x64(const _Tpvec& a)
{ return _Tpvec(_v256_permute4x64<imm>(a.val)); }

inline __m128i _v256_extract_high(const __m256i& v)
{ return _mm256_extracti128_si256(v, 1); }

inline __m128  _v256_extract_high(const __m256& v)
{ return _mm256_extractf128_ps(v, 1); }

inline __m128d _v256_extract_high(const __m256d& v)
{ return _mm256_extractf128_pd(v, 1); }

inline __m128i _v256_extract_low(const __m256i& v)
{ return _mm256_castsi256_si128(v); }

inline __m128  _v256_extract_low(const __m256& v)
{ return _mm256_castps256_ps128(v); }

inline __m128d _v256_extract_low(const __m256d& v)
{ return _mm256_castpd256_pd128(v); }

///////// Types ////////////

struct v_uint8x32
{
    typedef uchar lane_type;
    enum { nlanes = 32 };
    __m256i val;

    explicit v_uint8x32(__m256i v) : val(v) {}
    v_uint8x32(uchar v0,  uchar v1,  uchar v2,  uchar v3,
               uchar v4,  uchar v5,  uchar v6,  uchar v7,
               uchar v8,  uchar v9,  uchar v10, uchar v11,
               uchar v12, uchar v13, uchar v14, uchar v15,
               uchar v16, uchar v17, uchar v18, uchar v19,
               uchar v20, uchar v21, uchar v22, uchar v23,
               uchar v24, uchar v25, uchar v26, uchar v27,
               uchar v28, uchar v29, uchar v30, uchar v31)
    {
        val = _mm256_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
            (char)v4,  (char)v5,  (char)v6 , (char)v7,  (char)v8,  (char)v9,
            (char)v10, (char)v11, (char)v12, (char)v13, (char)v14, (char)v15,
            (char)v16, (char)v17, (char)v18, (char)v19, (char)v20, (char)v21,
            (char)v22, (char)v23, (char)v24, (char)v25, (char)v26, (char)v27,
            (char)v28, (char)v29, (char)v30, (char)v31);
    }
    v_uint8x32() : val(_mm256_setzero_si256()) {}
    uchar get0() const { return (uchar)_v_cvtsi256_si32(val); }
};

struct v_int8x32
{
    typedef schar lane_type;
    enum { nlanes = 32 };
    __m256i val;

    explicit v_int8x32(__m256i v) : val(v) {}
    v_int8x32(schar v0,  schar v1,  schar v2,  schar v3,
              schar v4,  schar v5,  schar v6,  schar v7,
              schar v8,  schar v9,  schar v10, schar v11,
              schar v12, schar v13, schar v14, schar v15,
              schar v16, schar v17, schar v18, schar v19,
              schar v20, schar v21, schar v22, schar v23,
              schar v24, schar v25, schar v26, schar v27,
              schar v28, schar v29, schar v30, schar v31)
    {
        val = _mm256_setr_epi8(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31);
    }
    v_int8x32() : val(_mm256_setzero_si256()) {}
    schar get0() const { return (schar)_v_cvtsi256_si32(val); }
};

struct v_uint16x16
{
    typedef ushort lane_type;
    enum { nlanes = 16 };
    __m256i val;

    explicit v_uint16x16(__m256i v) : val(v) {}
    v_uint16x16(ushort v0,  ushort v1,  ushort v2,  ushort v3,
                ushort v4,  ushort v5,  ushort v6,  ushort v7,
                ushort v8,  ushort v9,  ushort v10, ushort v11,
                ushort v12, ushort v13, ushort v14, ushort v15)
    {
        val = _mm256_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
            (short)v4,  (short)v5,  (short)v6,  (short)v7,  (short)v8,  (short)v9,
            (short)v10, (short)v11, (short)v12, (short)v13, (short)v14, (short)v15);
    }
    v_uint16x16() : val(_mm256_setzero_si256()) {}
    ushort get0() const { return (ushort)_v_cvtsi256_si32(val); }
};

struct v_int16x16
{
    typedef short lane_type;
    enum { nlanes = 16 };
    __m256i val;

    explicit v_int16x16(__m256i v) : val(v) {}
    v_int16x16(short v0,  short v1,  short v2,  short v3,
               short v4,  short v5,  short v6,  short v7,
               short v8,  short v9,  short v10, short v11,
               short v12, short v13, short v14, short v15)
    {
        val = _mm256_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7,
            v8, v9, v10, v11, v12, v13, v14, v15);
    }
    v_int16x16() : val(_mm256_setzero_si256()) {}
    short get0() const { return (short)_v_cvtsi256_si32(val); }
};

struct v_uint32x8
{
    typedef unsigned lane_type;
    enum { nlanes = 8 };
    __m256i val;

    explicit v_uint32x8(__m256i v) : val(v) {}
    v_uint32x8(unsigned v0, unsigned v1, unsigned v2, unsigned v3,
               unsigned v4, unsigned v5, unsigned v6, unsigned v7)
    {
        val = _mm256_setr_epi32((unsigned)v0, (unsigned)v1, (unsigned)v2,
            (unsigned)v3, (unsigned)v4, (unsigned)v5, (unsigned)v6, (unsigned)v7);
    }
    v_uint32x8() : val(_mm256_setzero_si256()) {}
    unsigned get0() const { return (unsigned)_v_cvtsi256_si32(val); }
};

struct v_int32x8
{
    typedef int lane_type;
    enum { nlanes = 8 };
    __m256i val;

    explicit v_int32x8(__m256i v) : val(v) {}
    v_int32x8(int v0, int v1, int v2, int v3,
              int v4, int v5, int v6, int v7)
    {
        val = _mm256_setr_epi32(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    v_int32x8() : val(_mm256_setzero_si256()) {}
    int get0() const { return _v_cvtsi256_si32(val); }
};

struct v_float32x8
{
    typedef float lane_type;
    enum { nlanes = 8 };
    __m256 val;

    explicit v_float32x8(__m256 v) : val(v) {}
    v_float32x8(float v0, float v1, float v2, float v3,
                float v4, float v5, float v6, float v7)
    {
        val = _mm256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    v_float32x8() : val(_mm256_setzero_ps()) {}
    float get0() const { return _mm_cvtss_f32(_mm256_castps256_ps128(val)); }
};

struct v_uint64x4
{
    typedef uint64 lane_type;
    enum { nlanes = 4 };
    __m256i val;

    explicit v_uint64x4(__m256i v) : val(v) {}
    v_uint64x4(uint64 v0, uint64 v1, uint64 v2, uint64 v3)
    { val = _mm256_setr_epi64x((int64)v0, (int64)v1, (int64)v2, (int64)v3); }
    v_uint64x4() : val(_mm256_setzero_si256()) {}
    uint64 get0() const
    { return (uint64)_mm_cvtsi128_si64(_mm256_castsi256_si128(val)); }
};

struct v_int64x4
{
    typedef int64 lane_type;
    enum { nlanes = 4 };
    __m256i val;

    explicit v_int64x4(__m256i v) : val(v) {}
    v_int64x4(int64 v0, int64 v1, int64 v2, int64 v3)
    { val = _mm256_setr_epi64x(v0, v1, v2, v3); }
    v_int64x4() : val(_mm256_setzero_si256()) {}
    int64 get0() const { return (int64)_mm_cvtsi128_si64(_mm256_castsi256_si128(val)); }
};

struct v_float64x4
{
    typedef double lane_type;
    enum { nlanes = 4 };
    __m256d val;

    explicit v_float64x4(__m256d v) : val(v) {}
    v_float64x4(double v0, double v1, double v2, double v3)
    { val = _mm256_setr_pd(v0, v1, v2, v3); }
    v_float64x4() : val(_mm256_setzero_pd()) {}
    double get0() const { return _mm_cvtsd_f64(_mm256_castpd256_pd128(val)); }
};

struct v_float16x16
{
    typedef short lane_type;
    enum { nlanes = 16 };
    __m256i val;

    explicit v_float16x16(__m256i v) : val(v) {}
    v_float16x16(short v0, short v1, short v2, short v3,
                 short v4, short v5, short v6, short v7,
                 short v8, short v9, short v10, short v11,
                 short v12, short v13, short v14, short v15)
    {
        val = _mm256_setr_epi16(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }
    v_float16x16() : val(_mm256_setzero_si256()) {}
    short get0() const { return (short)_v_cvtsi256_si32(val); }
};
inline v_float16x16 v256_setzero_f16() { return v_float16x16(_mm256_setzero_si256()); }
inline v_float16x16 v256_setall_f16(short val) { return v_float16x16(_mm256_set1_epi16(val)); }

//////////////// Load and store operations ///////////////

#define OPENCV_HAL_IMPL_AVX_LOADSTORE(_Tpvec, _Tp)                    \
    inline _Tpvec v256_load(const _Tp* ptr)                           \
    { return _Tpvec(_mm256_loadu_si256((const __m256i*)ptr)); }       \
    inline _Tpvec v256_load_aligned(const _Tp* ptr)                   \
    { return _Tpvec(_mm256_load_si256((const __m256i*)ptr)); }        \
    inline _Tpvec v256_load_low(const _Tp* ptr)                       \
    {                                                                 \
        __m128i v128 = _mm_loadu_si128((const __m128i*)ptr);          \
        return _Tpvec(_mm256_castsi128_si256(v128));                  \
    }                                                                 \
    inline _Tpvec v256_load_halves(const _Tp* ptr0, const _Tp* ptr1)  \
    {                                                                 \
        __m128i vlo = _mm_loadu_si128((const __m128i*)ptr0);          \
        __m128i vhi = _mm_loadu_si128((const __m128i*)ptr1);          \
        return _Tpvec(_v256_combine(vlo, vhi));                       \
    }                                                                 \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                    \
    { _mm256_storeu_si256((__m256i*)ptr, a.val); }                    \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)            \
    { _mm256_store_si256((__m256i*)ptr, a.val); }                     \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                \
    { _mm_storeu_si128((__m128i*)ptr, _v256_extract_low(a.val)); }    \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)               \
    { _mm_storeu_si128((__m128i*)ptr, _v256_extract_high(a.val)); }

OPENCV_HAL_IMPL_AVX_LOADSTORE(v_uint8x32,  uchar)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_int8x32,   schar)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_uint16x16, ushort)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_int16x16,  short)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_uint32x8,  unsigned)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_int32x8,   int)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_uint64x4,  uint64)
OPENCV_HAL_IMPL_AVX_LOADSTORE(v_int64x4,   int64)

#define OPENCV_HAL_IMPL_AVX_LOADSTORE_FLT(_Tpvec, _Tp, suffix, halfreg)   \
    inline _Tpvec v256_load(const _Tp* ptr)                               \
    { return _Tpvec(_mm256_loadu_##suffix(ptr)); }                        \
    inline _Tpvec v256_load_aligned(const _Tp* ptr)                       \
    { return _Tpvec(_mm256_load_##suffix(ptr)); }                         \
    inline _Tpvec v256_load_low(const _Tp* ptr)                           \
    {                                                                     \
        return _Tpvec(_mm256_cast##suffix##128_##suffix##256              \
                     (_mm_loadu_##suffix(ptr)));                          \
    }                                                                     \
    inline _Tpvec v256_load_halves(const _Tp* ptr0, const _Tp* ptr1)      \
    {                                                                     \
        halfreg vlo = _mm_loadu_##suffix(ptr0);                           \
        halfreg vhi = _mm_loadu_##suffix(ptr1);                           \
        return _Tpvec(_v256_combine(vlo, vhi));                           \
    }                                                                     \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                        \
    { _mm256_storeu_##suffix(ptr, a.val); }                               \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)                \
    { _mm256_store_##suffix(ptr, a.val); }                                \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                    \
    { _mm_storeu_##suffix(ptr, _v256_extract_low(a.val)); }               \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)                   \
    { _mm_storeu_##suffix(ptr, _v256_extract_high(a.val)); }

OPENCV_HAL_IMPL_AVX_LOADSTORE_FLT(v_float32x8, float,  ps, __m128)
OPENCV_HAL_IMPL_AVX_LOADSTORE_FLT(v_float64x4, double, pd, __m128d)

#define OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, _Tpvecf, suffix, cast) \
    inline _Tpvec v_reinterpret_as_##suffix(const _Tpvecf& a)   \
    { return _Tpvec(cast(a.val)); }

#define OPENCV_HAL_IMPL_AVX_INIT(_Tpvec, _Tp, suffix, ssuffix, ctype_s)          \
    inline _Tpvec v256_setzero_##suffix()                                        \
    { return _Tpvec(_mm256_setzero_si256()); }                                   \
    inline _Tpvec v256_setall_##suffix(_Tp v)                                    \
    { return _Tpvec(_mm256_set1_##ssuffix((ctype_s)v)); }                        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint8x32,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int8x32,   suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint16x16, suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int16x16,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint32x8,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int32x8,   suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint64x4,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int64x4,   suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_float32x8, suffix, _mm256_castps_si256)   \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_float64x4, suffix, _mm256_castpd_si256)

OPENCV_HAL_IMPL_AVX_INIT(v_uint8x32,  uchar,    u8,  epi8,   char)
OPENCV_HAL_IMPL_AVX_INIT(v_int8x32,   schar,    s8,  epi8,   char)
OPENCV_HAL_IMPL_AVX_INIT(v_uint16x16, ushort,   u16, epi16,  short)
OPENCV_HAL_IMPL_AVX_INIT(v_int16x16,  short,    s16, epi16,  short)
OPENCV_HAL_IMPL_AVX_INIT(v_uint32x8,  unsigned, u32, epi32,  int)
OPENCV_HAL_IMPL_AVX_INIT(v_int32x8,   int,      s32, epi32,  int)
OPENCV_HAL_IMPL_AVX_INIT(v_uint64x4,  uint64,   u64, epi64x, int64)
OPENCV_HAL_IMPL_AVX_INIT(v_int64x4,   int64,    s64, epi64x, int64)

#define OPENCV_HAL_IMPL_AVX_INIT_FLT(_Tpvec, _Tp, suffix, zsuffix, cast) \
    inline _Tpvec v256_setzero_##suffix()                                \
    { return _Tpvec(_mm256_setzero_##zsuffix()); }                       \
    inline _Tpvec v256_setall_##suffix(_Tp v)                            \
    { return _Tpvec(_mm256_set1_##zsuffix(v)); }                         \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint8x32,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int8x32,   suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint16x16, suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int16x16,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint32x8,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int32x8,   suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_uint64x4,  suffix, cast)          \
    OPENCV_HAL_IMPL_AVX_CAST(_Tpvec, v_int64x4,   suffix, cast)

OPENCV_HAL_IMPL_AVX_INIT_FLT(v_float32x8, float,  f32, ps, _mm256_castsi256_ps)
OPENCV_HAL_IMPL_AVX_INIT_FLT(v_float64x4, double, f64, pd, _mm256_castsi256_pd)

inline v_float32x8 v_reinterpret_as_f32(const v_float32x8& a)
{ return a; }
inline v_float32x8 v_reinterpret_as_f32(const v_float64x4& a)
{ return v_float32x8(_mm256_castpd_ps(a.val)); }

inline v_float64x4 v_reinterpret_as_f64(const v_float64x4& a)
{ return a; }
inline v_float64x4 v_reinterpret_as_f64(const v_float32x8& a)
{ return v_float64x4(_mm256_castps_pd(a.val)); }

inline v_float16x16 v256_load_f16(const short* ptr)
{ return v_float16x16(_mm256_loadu_si256((const __m256i*)ptr)); }
inline v_float16x16 v256_load_f16_aligned(const short* ptr)
{ return v_float16x16(_mm256_load_si256((const __m256i*)ptr)); }

inline void v_store(short* ptr, const v_float16x16& a)
{ _mm256_storeu_si256((__m256i*)ptr, a.val); }
inline void v_store_aligned(short* ptr, const v_float16x16& a)
{ _mm256_store_si256((__m256i*)ptr, a.val); }

/* Recombine */
/*#define OPENCV_HAL_IMPL_AVX_COMBINE(_Tpvec, perm)                    \
    inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)    \
    { return _Tpvec(perm(a.val, b.val, 0x20)); }                     \
    inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)   \
    { return _Tpvec(perm(a.val, b.val, 0x31)); }                     \
    inline void v_recombine(const _Tpvec& a, const _Tpvec& b,        \
                             _Tpvec& c, _Tpvec& d)                   \
    { c = v_combine_low(a, b); d = v_combine_high(a, b); }

#define OPENCV_HAL_IMPL_AVX_UNPACKS(_Tpvec, suffix)                  \
    OPENCV_HAL_IMPL_AVX_COMBINE(_Tpvec, _mm256_permute2x128_si256)   \
    inline void v_zip(const _Tpvec& a0, const _Tpvec& a1,            \
                             _Tpvec& b0, _Tpvec& b1)                 \
    {                                                                \
        __m256i v0 = _v256_shuffle_odd_64(a0.val);                   \
        __m256i v1 = _v256_shuffle_odd_64(a1.val);                   \
        b0.val = _mm256_unpacklo_##suffix(v0, v1);                   \
        b1.val = _mm256_unpackhi_##suffix(v0, v1);                   \
    }

OPENCV_HAL_IMPL_AVX_UNPACKS(v_uint8x32,  epi8)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_int8x32,   epi8)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_uint16x16, epi16)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_int16x16,  epi16)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_uint32x8,  epi32)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_int32x8,   epi32)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_uint64x4,  epi64)
OPENCV_HAL_IMPL_AVX_UNPACKS(v_int64x4,   epi64)
OPENCV_HAL_IMPL_AVX_COMBINE(v_float32x8, _mm256_permute2f128_ps)
OPENCV_HAL_IMPL_AVX_COMBINE(v_float64x4, _mm256_permute2f128_pd)

inline void v_zip(const v_float32x8& a0, const v_float32x8& a1, v_float32x8& b0, v_float32x8& b1)
{
    __m256 v0 = _mm256_unpacklo_ps(a0.val, a1.val);
    __m256 v1 = _mm256_unpackhi_ps(a0.val, a1.val);
    v_recombine(v_float32x8(v0), v_float32x8(v1), b0, b1);
}

inline void v_zip(const v_float64x4& a0, const v_float64x4& a1, v_float64x4& b0, v_float64x4& b1)
{
    __m256d v0 = _v_shuffle_odd_64(a0.val);
    __m256d v1 = _v_shuffle_odd_64(a1.val);
    b0.val = _mm256_unpacklo_pd(v0, v1);
    b1.val = _mm256_unpackhi_pd(v0, v1);
}*/

//////////////// Variant Value reordering ///////////////

// unpacks
#define OPENCV_HAL_IMPL_AVX_UNPACK(_Tpvec, suffix)                 \
    inline _Tpvec v256_unpacklo(const _Tpvec& a, const _Tpvec& b)  \
    { return _Tpvec(_mm256_unpacklo_##suffix(a.val, b.val)); }     \
    inline _Tpvec v256_unpackhi(const _Tpvec& a, const _Tpvec& b)  \
    { return _Tpvec(_mm256_unpackhi_##suffix(a.val, b.val)); }

OPENCV_HAL_IMPL_AVX_UNPACK(v_uint8x32,  epi8)
OPENCV_HAL_IMPL_AVX_UNPACK(v_int8x32,   epi8)
OPENCV_HAL_IMPL_AVX_UNPACK(v_uint16x16, epi16)
OPENCV_HAL_IMPL_AVX_UNPACK(v_int16x16,  epi16)
OPENCV_HAL_IMPL_AVX_UNPACK(v_uint32x8,  epi32)
OPENCV_HAL_IMPL_AVX_UNPACK(v_int32x8,   epi32)
OPENCV_HAL_IMPL_AVX_UNPACK(v_uint64x4,  epi64)
OPENCV_HAL_IMPL_AVX_UNPACK(v_int64x4,   epi64)
OPENCV_HAL_IMPL_AVX_UNPACK(v_float32x8, ps)
OPENCV_HAL_IMPL_AVX_UNPACK(v_float64x4, pd)

// blend
#define OPENCV_HAL_IMPL_AVX_BLEND(_Tpvec, suffix)               \
    template<int m>                                             \
    inline _Tpvec v256_blend(const _Tpvec& a, const _Tpvec& b)  \
    { return _Tpvec(_mm256_blend_##suffix(a.val, b.val, m)); }

OPENCV_HAL_IMPL_AVX_BLEND(v_uint16x16, epi16)
OPENCV_HAL_IMPL_AVX_BLEND(v_int16x16,  epi16)
OPENCV_HAL_IMPL_AVX_BLEND(v_uint32x8,  epi32)
OPENCV_HAL_IMPL_AVX_BLEND(v_int32x8,   epi32)
OPENCV_HAL_IMPL_AVX_BLEND(v_float32x8, ps)
OPENCV_HAL_IMPL_AVX_BLEND(v_float64x4, pd)

template<int m>
inline v_uint64x4 v256_blend(const v_uint64x4& a, const v_uint64x4& b)
{
    enum {M0 = m};
    enum {M1 = (M0 | (M0 << 2)) & 0x33};
    enum {M2 = (M1 | (M1 << 1)) & 0x55};
    enum {MM =  M2 | (M2 << 1)};
    return v_uint64x4(_mm256_blend_epi32(a.val, b.val, MM));
}
template<int m>
inline v_int64x4 v256_blend(const v_int64x4& a, const v_int64x4& b)
{ return v_int64x4(v256_blend<m>(v_uint64x4(a.val), v_uint64x4(b.val)).val); }

// shuffle
// todo: emluate 64bit
#define OPENCV_HAL_IMPL_AVX_SHUFFLE(_Tpvec, intrin)  \
    template<int m>                                  \
    inline _Tpvec v256_shuffle(const _Tpvec& a)      \
    { return _Tpvec(_mm256_##intrin(a.val, m)); }

OPENCV_HAL_IMPL_AVX_SHUFFLE(v_uint32x8,  shuffle_epi32)
OPENCV_HAL_IMPL_AVX_SHUFFLE(v_int32x8,   shuffle_epi32)
OPENCV_HAL_IMPL_AVX_SHUFFLE(v_float32x8, permute_ps)
OPENCV_HAL_IMPL_AVX_SHUFFLE(v_float64x4, permute_pd)

template<typename _Tpvec>
inline void v256_zip(const _Tpvec& a, const _Tpvec& b, _Tpvec& ab0, _Tpvec& ab1)
{
    ab0 = v256_unpacklo(a, b);
    ab1 = v256_unpackhi(a, b);
}

template<typename _Tpvec>
inline _Tpvec v256_combine_diagonal(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_mm256_blend_epi32(a.val, b.val, 0b11110000)); }

inline v_float32x8 v256_combine_diagonal(const v_float32x8& a, const v_float32x8& b)
{ return v256_blend<0b11110000>(a, b); }

inline v_float64x4 v256_combine_diagonal(const v_float64x4& a, const v_float64x4& b)
{ return v256_blend<0b1100>(a, b); }

template<typename _Tpvec>
inline _Tpvec v256_alignr_128(const _Tpvec& a, const _Tpvec& b)
{ return v256_permute2x128<0x21>(a, b); }

template<typename _Tpvec>
inline _Tpvec v256_alignr_64(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_mm256_alignr_epi8(a.val, b.val, 8)); }
inline v_float64x4 v256_alignr_64(const v_float64x4& a, const v_float64x4& b)
{ return v_float64x4(_mm256_shuffle_pd(b.val, a.val, _MM_SHUFFLE(0, 0, 1, 1))); }
// todo: emulate float32

template<typename _Tpvec>
inline _Tpvec v256_swap_halves(const _Tpvec& a)
{ return v256_permute2x128<1>(a, a); }

template<typename _Tpvec>
inline _Tpvec v256_reverse_64(const _Tpvec& a)
{ return v256_permute4x64<_MM_SHUFFLE(0, 1, 2, 3)>(a); }

// ZIP
#define OPENCV_HAL_IMPL_AVX_ZIP(_Tpvec)                              \
    inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)    \
    { return v256_permute2x128<0x20>(a, b); }                        \
    inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)   \
    { return v256_permute2x128<0x31>(a, b); }                        \
    inline void v_recombine(const _Tpvec& a, const _Tpvec& b,        \
                             _Tpvec& c, _Tpvec& d)                   \
    {                                                                \
        _Tpvec a1b0 = v256_alignr_128(a, b);                         \
        c = v256_combine_diagonal(a, a1b0);                          \
        d = v256_combine_diagonal(a1b0, b);                          \
    }                                                                \
    inline void v_zip(const _Tpvec& a, const _Tpvec& b,              \
                      _Tpvec& ab0, _Tpvec& ab1)                      \
    {                                                                \
        _Tpvec ab0ab2, ab1ab3;                                       \
        v256_zip(a, b, ab0ab2, ab1ab3);                              \
        v_recombine(ab0ab2, ab1ab3, ab0, ab1);                       \
    }

OPENCV_HAL_IMPL_AVX_ZIP(v_uint8x32)
OPENCV_HAL_IMPL_AVX_ZIP(v_int8x32)
OPENCV_HAL_IMPL_AVX_ZIP(v_uint16x16)
OPENCV_HAL_IMPL_AVX_ZIP(v_int16x16)
OPENCV_HAL_IMPL_AVX_ZIP(v_uint32x8)
OPENCV_HAL_IMPL_AVX_ZIP(v_int32x8)
OPENCV_HAL_IMPL_AVX_ZIP(v_uint64x4)
OPENCV_HAL_IMPL_AVX_ZIP(v_int64x4)
OPENCV_HAL_IMPL_AVX_ZIP(v_float32x8)
OPENCV_HAL_IMPL_AVX_ZIP(v_float64x4)

////////// Arithmetic, bitwise and comparison operations /////////

/* Element-wise binary and unary operations */

/** Arithmetics **/
#define OPENCV_HAL_IMPL_AVX_BIN_OP(bin_op, _Tpvec, intrin)            \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b)  \
    { return _Tpvec(intrin(a.val, b.val)); }                          \
    inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b)    \
    { a.val = intrin(a.val, b.val); return a; }

OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_uint8x32,  _mm256_adds_epu8)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_uint8x32,  _mm256_subs_epu8)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_int8x32,   _mm256_adds_epi8)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_int8x32,   _mm256_subs_epi8)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_uint16x16, _mm256_adds_epu16)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_uint16x16, _mm256_subs_epu16)
OPENCV_HAL_IMPL_AVX_BIN_OP(*, v_uint16x16, _mm256_mullo_epi16)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_int16x16,  _mm256_adds_epi16)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_int16x16,  _mm256_subs_epi16)
OPENCV_HAL_IMPL_AVX_BIN_OP(*, v_int16x16,  _mm256_mullo_epi16)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_uint32x8,  _mm256_add_epi32)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_uint32x8,  _mm256_sub_epi32)
OPENCV_HAL_IMPL_AVX_BIN_OP(*, v_uint32x8,  _mm256_mullo_epi32)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_int32x8,   _mm256_add_epi32)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_int32x8,   _mm256_sub_epi32)
OPENCV_HAL_IMPL_AVX_BIN_OP(*, v_int32x8,   _mm256_mullo_epi32)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_uint64x4,  _mm256_add_epi64)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_uint64x4,  _mm256_sub_epi64)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_int64x4,   _mm256_add_epi64)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_int64x4,   _mm256_sub_epi64)

OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_float32x8, _mm256_add_ps)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_float32x8, _mm256_sub_ps)
OPENCV_HAL_IMPL_AVX_BIN_OP(*, v_float32x8, _mm256_mul_ps)
OPENCV_HAL_IMPL_AVX_BIN_OP(/, v_float32x8, _mm256_div_ps)
OPENCV_HAL_IMPL_AVX_BIN_OP(+, v_float64x4, _mm256_add_pd)
OPENCV_HAL_IMPL_AVX_BIN_OP(-, v_float64x4, _mm256_sub_pd)
OPENCV_HAL_IMPL_AVX_BIN_OP(*, v_float64x4, _mm256_mul_pd)
OPENCV_HAL_IMPL_AVX_BIN_OP(/, v_float64x4, _mm256_div_pd)

inline void v_mul_expand(const v_int16x16& a, const v_int16x16& b,
                         v_int32x8& c, v_int32x8& d)
{
    v_int16x16 vhi = v_int16x16(_mm256_mulhi_epi16(a.val, b.val));

    v_int16x16 v0, v1;
    v_zip(a * b, vhi, v0, v1);

    c = v_reinterpret_as_s32(v0);
    d = v_reinterpret_as_s32(v1);
}

inline void v_mul_expand(const v_uint16x16& a, const v_uint16x16& b,
                         v_uint32x8& c, v_uint32x8& d)
{
    v_uint16x16 vhi = v_uint16x16(_mm256_mulhi_epu16(a.val, b.val));

    v_uint16x16 v0, v1;
    v_zip(a * b, vhi, v0, v1);

    c = v_reinterpret_as_u32(v0);
    d = v_reinterpret_as_u32(v1);
}

inline void v_mul_expand(const v_uint32x8& a, const v_uint32x8& b,
                         v_uint64x4& c, v_uint64x4& d)
{
    __m256i v0 = _mm256_mul_epu32(a.val, b.val);
    __m256i v1 = _mm256_mul_epu32(_mm256_srli_epi64(a.val, 32), _mm256_srli_epi64(b.val, 32));
    v_zip(v_uint64x4(v0), v_uint64x4(v1), c, d);
}


/** Non-saturating arithmetics **/
#define OPENCV_HAL_IMPL_AVX_BIN_FUNC(func, _Tpvec, intrin) \
    inline _Tpvec func(const _Tpvec& a, const _Tpvec& b)   \
    { return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_add_wrap, v_uint8x32,  _mm256_add_epi8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_add_wrap, v_int8x32,   _mm256_add_epi8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_add_wrap, v_uint16x16, _mm256_add_epi16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_add_wrap, v_int16x16,  _mm256_add_epi16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_sub_wrap, v_uint8x32,  _mm256_sub_epi8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_sub_wrap, v_int8x32,   _mm256_sub_epi8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_sub_wrap, v_uint16x16, _mm256_sub_epi16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_sub_wrap, v_int16x16,  _mm256_sub_epi16)

/** Bitwise shifts **/
#define OPENCV_HAL_IMPL_AVX_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, srai)  \
    inline _Tpuvec operator << (const _Tpuvec& a, int imm)            \
    { return _Tpuvec(_mm256_slli_##suffix(a.val, imm)); }             \
    inline _Tpsvec operator << (const _Tpsvec& a, int imm)            \
    { return _Tpsvec(_mm256_slli_##suffix(a.val, imm)); }             \
    inline _Tpuvec operator >> (const _Tpuvec& a, int imm)            \
    { return _Tpuvec(_mm256_srli_##suffix(a.val, imm)); }             \
    inline _Tpsvec operator >> (const _Tpsvec& a, int imm)            \
    { return _Tpsvec(srai(a.val, imm)); }                             \
    template<int imm>                                                 \
    inline _Tpuvec v_shl(const _Tpuvec& a)                            \
    { return _Tpuvec(_mm256_slli_##suffix(a.val, imm)); }             \
    template<int imm>                                                 \
    inline _Tpsvec v_shl(const _Tpsvec& a)                            \
    { return _Tpsvec(_mm256_slli_##suffix(a.val, imm)); }             \
    template<int imm>                                                 \
    inline _Tpuvec v_shr(const _Tpuvec& a)                            \
    { return _Tpuvec(_mm256_srli_##suffix(a.val, imm)); }             \
    template<int imm>                                                 \
    inline _Tpsvec v_shr(const _Tpsvec& a)                            \
    { return _Tpsvec(srai(a.val, imm)); }

OPENCV_HAL_IMPL_AVX_SHIFT_OP(v_uint16x16, v_int16x16, epi16, _mm256_srai_epi16)
OPENCV_HAL_IMPL_AVX_SHIFT_OP(v_uint32x8,  v_int32x8,  epi32, _mm256_srai_epi32)

inline __m256i _mm256_srai_epi64xx(const __m256i a, int imm)
{
    __m256i d = _mm256_set1_epi64x((int64)1 << 63);
    __m256i r = _mm256_srli_epi64(_mm256_add_epi64(a, d), imm);
    return _mm256_sub_epi64(r, _mm256_srli_epi64(d, imm));
}
OPENCV_HAL_IMPL_AVX_SHIFT_OP(v_uint64x4,  v_int64x4,  epi64, _mm256_srai_epi64xx)


/** Bitwise logic **/
#define OPENCV_HAL_IMPL_AVX_LOGIC_OP(_Tpvec, suffix, not_const)  \
    OPENCV_HAL_IMPL_AVX_BIN_OP(&, _Tpvec, _mm256_and_##suffix)   \
    OPENCV_HAL_IMPL_AVX_BIN_OP(|, _Tpvec, _mm256_or_##suffix)    \
    OPENCV_HAL_IMPL_AVX_BIN_OP(^, _Tpvec, _mm256_xor_##suffix)   \
    inline _Tpvec operator ~ (const _Tpvec& a)                   \
    { return _Tpvec(_mm256_xor_##suffix(a.val, not_const)); }

OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_uint8x32,   si256, _mm256_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_int8x32,    si256, _mm256_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_uint16x16,  si256, _mm256_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_int16x16,   si256, _mm256_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_uint32x8,   si256, _mm256_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_int32x8,    si256, _mm256_set1_epi32(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_uint64x4,   si256, _mm256_set1_epi64x(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_int64x4,    si256, _mm256_set1_epi64x(-1))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_float32x8,  ps,    _mm256_castsi256_ps(_mm256_set1_epi32(-1)))
OPENCV_HAL_IMPL_AVX_LOGIC_OP(v_float64x4,  pd,    _mm256_castsi256_pd(_mm256_set1_epi32(-1)))

/** Select **/
#define OPENCV_HAL_IMPL_AVX_SELECT(_Tpvec, suffix)                               \
    inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
    { return _Tpvec(_mm256_blendv_##suffix(b.val, a.val, mask.val)); }

OPENCV_HAL_IMPL_AVX_SELECT(v_uint8x32,  epi8)
OPENCV_HAL_IMPL_AVX_SELECT(v_int8x32,   epi8)
OPENCV_HAL_IMPL_AVX_SELECT(v_uint16x16, epi8)
OPENCV_HAL_IMPL_AVX_SELECT(v_int16x16,  epi8)
OPENCV_HAL_IMPL_AVX_SELECT(v_uint32x8,  epi8)
OPENCV_HAL_IMPL_AVX_SELECT(v_int32x8,   epi8)
OPENCV_HAL_IMPL_AVX_SELECT(v_float32x8, ps)
OPENCV_HAL_IMPL_AVX_SELECT(v_float64x4, pd)

/** Comparison **/
#define OPENCV_HAL_IMPL_AVX_CMP_OP_OV(_Tpvec)                     \
    inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b)  \
    { return ~(a == b); }                                         \
    inline _Tpvec operator <  (const _Tpvec& a, const _Tpvec& b)  \
    { return b > a; }                                             \
    inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b)  \
    { return ~(a < b); }                                          \
    inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b)  \
    { return b >= a; }

#define OPENCV_HAL_IMPL_AVX_CMP_OP_INT(_Tpuvec, _Tpsvec, suffix, sbit)   \
    inline _Tpuvec operator == (const _Tpuvec& a, const _Tpuvec& b)      \
    { return _Tpuvec(_mm256_cmpeq_##suffix(a.val, b.val)); }             \
    inline _Tpuvec operator > (const _Tpuvec& a, const _Tpuvec& b)       \
    {                                                                    \
        __m256i smask = _mm256_set1_##suffix(sbit);                      \
        return _Tpuvec(_mm256_cmpgt_##suffix(                            \
                       _mm256_xor_si256(a.val, smask),                   \
                       _mm256_xor_si256(b.val, smask)));                 \
    }                                                                    \
    inline _Tpsvec operator == (const _Tpsvec& a, const _Tpsvec& b)      \
    { return _Tpsvec(_mm256_cmpeq_##suffix(a.val, b.val)); }             \
    inline _Tpsvec operator > (const _Tpsvec& a, const _Tpsvec& b)       \
    { return _Tpsvec(_mm256_cmpgt_##suffix(a.val, b.val)); }             \
    OPENCV_HAL_IMPL_AVX_CMP_OP_OV(_Tpuvec)                               \
    OPENCV_HAL_IMPL_AVX_CMP_OP_OV(_Tpsvec)

OPENCV_HAL_IMPL_AVX_CMP_OP_INT(v_uint8x32,  v_int8x32,  epi8,  (char)-128)
OPENCV_HAL_IMPL_AVX_CMP_OP_INT(v_uint16x16, v_int16x16, epi16, (short)-32768)
OPENCV_HAL_IMPL_AVX_CMP_OP_INT(v_uint32x8,  v_int32x8,  epi32, (int)0x80000000)

#define OPENCV_HAL_IMPL_AVX_CMP_OP_64BIT(_Tpvec)                 \
    inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
    { return _Tpvec(_mm256_cmpeq_epi64(a.val, b.val)); }         \
    inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
    { return ~(a == b); }

OPENCV_HAL_IMPL_AVX_CMP_OP_64BIT(v_uint64x4)
OPENCV_HAL_IMPL_AVX_CMP_OP_64BIT(v_int64x4)

#define OPENCV_HAL_IMPL_AVX_CMP_FLT(bin_op, imm8, _Tpvec, suffix)    \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
    { return _Tpvec(_mm256_cmp_##suffix(a.val, b.val, imm8)); }

#define OPENCV_HAL_IMPL_AVX_CMP_OP_FLT(_Tpvec, suffix)               \
    OPENCV_HAL_IMPL_AVX_CMP_FLT(==, _CMP_EQ_OQ,  _Tpvec, suffix)     \
    OPENCV_HAL_IMPL_AVX_CMP_FLT(!=, _CMP_NEQ_OQ, _Tpvec, suffix)     \
    OPENCV_HAL_IMPL_AVX_CMP_FLT(<,  _CMP_LT_OQ,  _Tpvec, suffix)     \
    OPENCV_HAL_IMPL_AVX_CMP_FLT(>,  _CMP_GT_OQ,  _Tpvec, suffix)     \
    OPENCV_HAL_IMPL_AVX_CMP_FLT(<=, _CMP_LE_OQ,  _Tpvec, suffix)     \
    OPENCV_HAL_IMPL_AVX_CMP_FLT(>=, _CMP_GE_OQ,  _Tpvec, suffix)

OPENCV_HAL_IMPL_AVX_CMP_OP_FLT(v_float32x8, ps)
OPENCV_HAL_IMPL_AVX_CMP_OP_FLT(v_float64x4, pd)

/** min/max **/
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_uint8x32,  _mm256_min_epu8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_uint8x32,  _mm256_max_epu8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_int8x32,   _mm256_min_epi8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_int8x32,   _mm256_max_epi8)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_uint16x16, _mm256_min_epu16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_uint16x16, _mm256_max_epu16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_int16x16,  _mm256_min_epi16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_int16x16,  _mm256_max_epi16)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_uint32x8,  _mm256_min_epu32)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_uint32x8,  _mm256_max_epu32)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_int32x8,   _mm256_min_epi32)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_int32x8,   _mm256_max_epi32)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_float32x8, _mm256_min_ps)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_float32x8, _mm256_max_ps)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_min, v_float64x4, _mm256_min_pd)
OPENCV_HAL_IMPL_AVX_BIN_FUNC(v_max, v_float64x4, _mm256_max_pd)

/** Rotate **/
template<int imm>
inline v_uint8x32 v_rotate_left(const v_uint8x32& a, const v_uint8x32& b)
{
    __m256i swap = _mm256_permute2x128_si256(a.val, b.val, 0x03);

    switch(imm)
    {
        case 0:  return a;
        case 32: return b;
        case 16: return v_uint8x32(swap);
    }

    if (imm < 16) return v_uint8x32(_mm256_alignr_epi8(a.val, swap, 16 - imm));
    if (imm < 32) return v_uint8x32(_mm256_alignr_epi8(swap, b.val, 32 - imm));

    return v_uint8x32();
}

template<int imm>
inline v_uint8x32 v_rotate_right(const v_uint8x32& a, const v_uint8x32& b)
{
    __m256i swap = _mm256_permute2x128_si256(a.val, b.val, 0x21);

    switch(imm)
    {
        case 0:  return a;
        case 32: return b;
        case 16: return v_uint8x32(swap);
    }

    if (imm < 16) return v_uint8x32(_mm256_alignr_epi8(swap, a.val, imm));
    if (imm < 32) return v_uint8x32(_mm256_alignr_epi8(b.val, swap, imm - 16));

    return v_uint8x32();
}

template<int imm>
inline v_uint8x32 v_rotate_left(const v_uint8x32& a)
{
    v_uint8x32 res;
    // ESAC control[3] ? [127:0] = 0
    __m256i swapz = _mm256_permute2x128_si256(a.val, a.val, _MM_SHUFFLE(0, 0, 2, 0));

    if (imm == 0)
        return a;
    if (imm == 16)
        res.val = swapz;
    else if (imm < 16)
        res.val = _mm256_alignr_epi8(a.val, swapz, 16 - imm);
    else if (imm < 32)
        res.val = _mm256_slli_si256(swapz, imm - 16);
    else
        return v_uint8x32();
    return res;
}

template<int imm>
inline v_uint8x32 v_rotate_right(const v_uint8x32& a)
{
    v_uint8x32 res;
    // ESAC control[3] ? [127:0] = 0
    __m256i swapz = _mm256_permute2x128_si256(a.val, a.val, _MM_SHUFFLE(2, 0, 0, 1));

    if (imm == 0)
        return a;
    if (imm == 16)
        res.val = swapz;
    else if (imm < 16)
        res.val = _mm256_alignr_epi8(swapz, a.val, imm);
    else if (imm < 32)
        res.val = _mm256_srli_si256(swapz, imm - 16);
    else
        return v_uint8x32();
    return res;
}

#define OPENCV_HAL_IMPL_AVX_ROTATE_CAST(intrin, _Tpvec, cast)   \
    template<int imm>                                           \
    inline _Tpvec intrin(const _Tpvec& a, const _Tpvec& b)      \
    {                                                           \
        const int w = sizeof(typename _Tpvec::lane_type);       \
        v_uint8x32 ret = intrin<imm*w>(v_reinterpret_as_u8(a),  \
                                       v_reinterpret_as_u8(b)); \
        return _Tpvec(cast(ret.val));                           \
    }                                                           \
    template<int imm>                                           \
    inline _Tpvec intrin(const _Tpvec& a)                       \
    {                                                           \
        const int w = sizeof(typename _Tpvec::lane_type);       \
        v_uint8x32 ret = intrin<imm*w>(v_reinterpret_as_u8(a)); \
        return _Tpvec(cast(ret.val));                           \
    }

#define OPENCV_HAL_IMPL_AVX_ROTATE(_Tpvec)                                  \
    OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_left,  _Tpvec, OPENCV_HAL_NOP) \
    OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_right, _Tpvec, OPENCV_HAL_NOP)

OPENCV_HAL_IMPL_AVX_ROTATE(v_int8x32)
OPENCV_HAL_IMPL_AVX_ROTATE(v_uint16x16)
OPENCV_HAL_IMPL_AVX_ROTATE(v_int16x16)
OPENCV_HAL_IMPL_AVX_ROTATE(v_uint32x8)
OPENCV_HAL_IMPL_AVX_ROTATE(v_int32x8)
OPENCV_HAL_IMPL_AVX_ROTATE(v_uint64x4)
OPENCV_HAL_IMPL_AVX_ROTATE(v_int64x4)

OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_left,  v_float32x8, _mm256_castsi256_ps)
OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_right, v_float32x8, _mm256_castsi256_ps)
OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_left,  v_float64x4, _mm256_castsi256_pd)
OPENCV_HAL_IMPL_AVX_ROTATE_CAST(v_rotate_right, v_float64x4, _mm256_castsi256_pd)

////////// Reduce and mask /////////

/** Reduce **/
#define OPENCV_HAL_IMPL_AVX_REDUCE_16(_Tpvec, sctype, func, intrin) \
    inline sctype v_reduce_##func(const _Tpvec& a)                  \
    {                                                               \
        __m128i v0 = _v256_extract_low(a.val);                      \
        __m128i v1 = _v256_extract_high(a.val);                     \
        v0 = intrin(v0, v1);                                        \
        v0 = intrin(v0, _mm_srli_si128(v0, 8));                     \
        v0 = intrin(v0, _mm_srli_si128(v0, 4));                     \
        v0 = intrin(v0, _mm_srli_si128(v0, 2));                     \
        return (sctype) _mm_cvtsi128_si32(v0);                      \
    }

OPENCV_HAL_IMPL_AVX_REDUCE_16(v_uint16x16, ushort, min, _mm_min_epu16)
OPENCV_HAL_IMPL_AVX_REDUCE_16(v_int16x16,  short,  min, _mm_min_epi16)
OPENCV_HAL_IMPL_AVX_REDUCE_16(v_uint16x16, ushort, max, _mm_max_epu16)
OPENCV_HAL_IMPL_AVX_REDUCE_16(v_int16x16,  short,  max, _mm_max_epi16)

#define OPENCV_HAL_IMPL_AVX_REDUCE_8(_Tpvec, sctype, func, intrin) \
    inline sctype v_reduce_##func(const _Tpvec& a)                 \
    {                                                              \
        __m128i v0 = _v256_extract_low(a.val);                     \
        __m128i v1 = _v256_extract_high(a.val);                    \
        v0 = intrin(v0, v1);                                       \
        v0 = intrin(v0, _mm_srli_si128(v0, 8));                    \
        v0 = intrin(v0, _mm_srli_si128(v0, 4));                    \
        return (sctype) _mm_cvtsi128_si32(v0);                     \
    }

OPENCV_HAL_IMPL_AVX_REDUCE_8(v_uint32x8, unsigned, min, _mm_min_epu32)
OPENCV_HAL_IMPL_AVX_REDUCE_8(v_int32x8,  int,      min, _mm_min_epi32)
OPENCV_HAL_IMPL_AVX_REDUCE_8(v_uint32x8, unsigned, max, _mm_max_epu32)
OPENCV_HAL_IMPL_AVX_REDUCE_8(v_int32x8,  int,      max, _mm_max_epi32)

#define OPENCV_HAL_IMPL_AVX_REDUCE_FLT(func, intrin)                  \
    inline float v_reduce_##func(const v_float32x8& a)                \
    {                                                                 \
        __m128 v0 = _v256_extract_low(a.val);                         \
        __m128 v1 = _v256_extract_high(a.val);                        \
        v0 = intrin(v0, v1);                                          \
        v0 = intrin(v0, _mm_permute_ps(v0, _MM_SHUFFLE(0, 0, 3, 2))); \
        v0 = intrin(v0, _mm_permute_ps(v0, _MM_SHUFFLE(0, 0, 0, 3))); \
        return _mm_cvtss_f32(v0);                                     \
    }

OPENCV_HAL_IMPL_AVX_REDUCE_FLT(min, _mm_min_ps)
OPENCV_HAL_IMPL_AVX_REDUCE_FLT(max, _mm_max_ps)

inline ushort v_reduce_sum(const v_uint16x16& a)
{
    __m128i a0 = _v256_extract_low(a.val);
    __m128i a1 = _v256_extract_high(a.val);

    __m128i s0 = _mm_adds_epu16(a0, a1);
            s0 = _mm_adds_epu16(s0, _mm_srli_si128(s0, 8));
            s0 = _mm_adds_epu16(s0, _mm_srli_si128(s0, 4));
            s0 = _mm_adds_epu16(s0, _mm_srli_si128(s0, 2));

    return (ushort)_mm_cvtsi128_si32(s0);
}

inline short v_reduce_sum(const v_int16x16& a)
{
    __m256i s0 = _mm256_hadds_epi16(a.val, a.val);
            s0 = _mm256_hadds_epi16(s0, s0);
            s0 = _mm256_hadds_epi16(s0, s0);

    __m128i s1 = _v256_extract_high(s0);
            s1 = _mm_adds_epi16(_v256_extract_low(s0), s1);

    return (short)_mm_cvtsi128_si32(s1);
}

inline int v_reduce_sum(const v_int32x8& a)
{
    __m256i s0 = _mm256_hadd_epi32(a.val, a.val);
            s0 = _mm256_hadd_epi32(s0, s0);

    __m128i s1 = _v256_extract_high(s0);
            s1 = _mm_add_epi32(_v256_extract_low(s0), s1);

    return _mm_cvtsi128_si32(s1);
}

inline unsigned v_reduce_sum(const v_uint32x8& a)
{ return v_reduce_sum(v_reinterpret_as_s32(a)); }

inline float v_reduce_sum(const v_float32x8& a)
{
    __m256 s0 = _mm256_hadd_ps(a.val, a.val);
           s0 = _mm256_hadd_ps(s0, s0);

    __m128 s1 = _v256_extract_high(s0);
           s1 = _mm_add_ps(_v256_extract_low(s0), s1);

    return _mm_cvtss_f32(s1);
}

inline v_float32x8 v_reduce_sum4(const v_float32x8& a, const v_float32x8& b,
                                 const v_float32x8& c, const v_float32x8& d)
{
    __m256 ab = _mm256_hadd_ps(a.val, b.val);
    __m256 cd = _mm256_hadd_ps(c.val, d.val);
    return v_float32x8(_mm256_hadd_ps(ab, cd));
}

/** Popcount **/
#define OPENCV_HAL_IMPL_AVX_POPCOUNT(_Tpvec)                     \
    inline v_uint32x8 v_popcount(const _Tpvec& a)                \
    {                                                            \
        const v_uint32x8 m1 = v256_setall_u32(0x55555555);       \
        const v_uint32x8 m2 = v256_setall_u32(0x33333333);       \
        const v_uint32x8 m4 = v256_setall_u32(0x0f0f0f0f);       \
        v_uint32x8 p  = v_reinterpret_as_u32(a);                 \
        p = ((p >> 1) & m1) + (p & m1);                          \
        p = ((p >> 2) & m2) + (p & m2);                          \
        p = ((p >> 4) & m4) + (p & m4);                          \
        p.val = _mm256_sad_epu8(p.val, _mm256_setzero_si256());  \
        return p;                                                \
    }

OPENCV_HAL_IMPL_AVX_POPCOUNT(v_uint8x32)
OPENCV_HAL_IMPL_AVX_POPCOUNT(v_int8x32)
OPENCV_HAL_IMPL_AVX_POPCOUNT(v_uint16x16)
OPENCV_HAL_IMPL_AVX_POPCOUNT(v_int16x16)
OPENCV_HAL_IMPL_AVX_POPCOUNT(v_uint32x8)
OPENCV_HAL_IMPL_AVX_POPCOUNT(v_int32x8)

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
    __m256i a16 = _mm256_packs_epi32(a.val, a.val);
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

inline v_float32x8 v_invsqrt(const v_float32x8& x)
{
    v_float32x8 half = x * v256_setall_f32(0.5);
    v_float32x8 t  = v_float32x8(_mm256_rsqrt_ps(x.val));
    // todo: _mm256_fnmsub_ps
    t *= v256_setall_f32(1.5) - ((t * t) * half);
    return t;
}

inline v_float64x4 v_invsqrt(const v_float64x4& x)
{
    return v256_setall_f64(1.) / v_sqrt(x);
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

////////// Conversions /////////

/** Rounding **/
inline v_int32x8 v_round(const v_float32x8& a)
{ return v_int32x8(_mm256_cvtps_epi32(a.val)); }

inline v_int32x8 v_round(const v_float64x4& a)
{ return v_int32x8(_mm256_castsi128_si256(_mm256_cvtpd_epi32(a.val))); }

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
    __m128 af = _mm256_cvtpd_ps(a.val), bf = _mm256_cvtpd_ps(b.val);
    return v_float32x8(_mm256_insertf128_ps(_mm256_castps128_ps256(af), bf, 1));
}

inline v_float64x4 v_cvt_f64(const v_int32x8& a)
{ return v_float64x4(_mm256_cvtepi32_pd(_v256_extract_low(a.val))); }

inline v_float64x4 v_cvt_f64_high(const v_int32x8& a)
{ return v_float64x4(_mm256_cvtepi32_pd(_v256_extract_high(a.val))); }

inline v_float64x4 v_cvt_f64(const v_float32x8& a)
{ return v_float64x4(_mm256_cvtps_pd(_v256_extract_low(a.val))); }

inline v_float64x4 v_cvt_f64_high(const v_float32x8& a)
{ return v_float64x4(_mm256_cvtps_pd(_v256_extract_high(a.val))); }

#if CV_FP16
inline v_float32x8 v_cvt_f32(const v_float16x16& a)
{ return v_float32x8(_mm256_cvtph_ps(_v256_extract_low(a.val))); }

inline v_float32x8 v_cvt_f32_high(const v_float16x16& a)
{ return v_float32x8(_mm256_cvtph_ps(_v256_extract_high(a.val))); }

inline v_float16x16 v_cvt_f16(const v_float32x8& a, const v_float32x8& b)
{
    __m128i ah = _mm256_cvtps_ph(a.val, 0), bh = _mm256_cvtps_ph(b.val, 0);
    return v_float16x16(_mm256_inserti128_si256(_mm256_castsi128_si256(ah), bh, 1));
}
#endif

////////////// Lookup table access ////////////////////

inline v_int32x8 v_lut(const int* tab, const v_int32x8& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[8];
    v_store_aligned(idx, idxvec);
    return v_int32x8(_mm256_setr_epi32(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                                       tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]));
}

inline v_float32x8 v_lut(const float* tab, const v_int32x8& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[8];
    v_store_aligned(idx, idxvec);
    return v_float32x8(_mm256_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                                      tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]));
}

inline v_float64x4 v_lut(const double* tab, const v_int32x8& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[8];
    v_store_aligned(idx, idxvec);
    return v_float64x4(_mm256_setr_pd(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}

inline void v_lut_deinterleave(const float* tab, const v_int32x8& idxvec, v_float32x8& x, v_float32x8& y)
{
    int CV_DECL_ALIGNED(32) idx[8];
    v_store_aligned(idx, idxvec);
    __m128 z = _mm_setzero_ps();
    __m128 xy01, xy45, xy23, xy67;
    xy01 = _mm_loadl_pi(z, (const __m64*)(tab + idx[0]));
    xy01 = _mm_loadh_pi(xy01, (const __m64*)(tab + idx[1]));
    xy45 = _mm_loadl_pi(z, (const __m64*)(tab + idx[4]));
    xy45 = _mm_loadh_pi(xy45, (const __m64*)(tab + idx[5]));
    __m256 xy0145 = _v256_combine(xy01, xy45);
    xy23 = _mm_loadl_pi(z, (const __m64*)(tab + idx[2]));
    xy23 = _mm_loadh_pi(xy23, (const __m64*)(tab + idx[3]));
    xy67 = _mm_loadl_pi(z, (const __m64*)(tab + idx[6]));
    xy67 = _mm_loadh_pi(xy67, (const __m64*)(tab + idx[7]));
    __m256 xy2367 = _v256_combine(xy23, xy67);

    __m256 xxyy0145 = _mm256_unpacklo_ps(xy0145, xy2367);
    __m256 xxyy2367 = _mm256_unpackhi_ps(xy0145, xy2367);

    x = v_float32x8(_mm256_unpacklo_ps(xxyy0145, xxyy2367));
    y = v_float32x8(_mm256_unpackhi_ps(xxyy0145, xxyy2367));
}

inline void v_lut_deinterleave(const double* tab, const v_int32x8& idxvec, v_float64x4& x, v_float64x4& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_low(idx, idxvec);
    __m128d xy0 = _mm_loadu_pd(tab + idx[0]);
    __m128d xy2 = _mm_loadu_pd(tab + idx[2]);
    __m128d xy1 = _mm_loadu_pd(tab + idx[1]);
    __m128d xy3 = _mm_loadu_pd(tab + idx[3]);
    __m256d xy02 = _v256_combine(xy0, xy2);
    __m256d xy13 = _v256_combine(xy1, xy3);

    x = v_float64x4(_mm256_unpacklo_pd(xy02, xy13));
    y = v_float64x4(_mm256_unpackhi_pd(xy02, xy13));
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
        __m256i t0 = cast_from(_mm256_unpacklo_##suffix(a0.val, a1.val));       \
        __m256i t1 = cast_from(_mm256_unpacklo_##suffix(a2.val, a3.val));       \
        __m256i t2 = cast_from(_mm256_unpackhi_##suffix(a0.val, a1.val));       \
        __m256i t3 = cast_from(_mm256_unpackhi_##suffix(a2.val, a3.val));       \
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
        b0.val = intrin(_v256_extract_low(a.val));                  \
        b1.val = intrin(_v256_extract_high(a.val));                 \
    }                                                               \
    inline _Tpwvec v256_load_expand(const _Tp* ptr)                 \
    {                                                               \
        __m128i a = _mm_loadu_si128((const __m128i*)ptr);           \
        return _Tpwvec(intrin(a));                                  \
    }

OPENCV_HAL_IMPL_AVX_EXPAND(v_uint8x32,  v_uint16x16, uchar,    _mm256_cvtepu8_epi16)
OPENCV_HAL_IMPL_AVX_EXPAND(v_int8x32,   v_int16x16,  schar,    _mm256_cvtepi8_epi16)
OPENCV_HAL_IMPL_AVX_EXPAND(v_uint16x16, v_uint32x8,  ushort,   _mm256_cvtepu16_epi32)
OPENCV_HAL_IMPL_AVX_EXPAND(v_int16x16,  v_int32x8,   short,    _mm256_cvtepi16_epi32)
OPENCV_HAL_IMPL_AVX_EXPAND(v_uint32x8,  v_uint64x4,  unsigned, _mm256_cvtepu32_epi64)
OPENCV_HAL_IMPL_AVX_EXPAND(v_int32x8,   v_int64x4,   int,      _mm256_cvtepi32_epi64)

#define OPENCV_HAL_IMPL_AVX_EXPAND_Q(_Tpvec, _Tp, intrin)   \
    inline _Tpvec v256_load_expand_q(const _Tp* ptr)        \
    {                                                       \
        __m128i a = _mm_loadl_epi64((const __m128i*)ptr);   \
        return _Tpvec(intrin(a));                           \
    }

OPENCV_HAL_IMPL_AVX_EXPAND_Q(v_uint32x8, uchar, _mm256_cvtepu8_epi32)
OPENCV_HAL_IMPL_AVX_EXPAND_Q(v_int32x8,  schar, _mm256_cvtepi8_epi32)

/* pack */
// 16
inline v_int8x32 v_pack(const v_int16x16& a, const v_int16x16& b)
{ return v_int8x32(_v256_shuffle_odd_64(_mm256_packs_epi16(a.val, b.val))); }

inline v_uint8x32 v_pack(const v_uint16x16& a, const v_uint16x16& b)
{ return v_uint8x32(_v256_shuffle_odd_64(_mm256_packus_epi16(a.val, b.val))); }

inline v_uint8x32 v_pack_u(const v_int16x16& a, const v_int16x16& b)
{ return v_pack(v_reinterpret_as_u16(a), v_reinterpret_as_u16(b)); }

inline void v_pack_store(schar* ptr, const v_int16x16& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(uchar* ptr, const v_uint16x16& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_u_store(uchar* ptr, const v_int16x16& a)
{ v_store_low(ptr, v_pack_u(a, a)); }

template<int n> inline
v_uint8x32 v_rshr_pack(const v_uint16x16& a, const v_uint16x16& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    v_uint16x16 delta = v256_setall_u16((short)(1 << (n-1)));
    return v_pack_u(v_reinterpret_as_s16((a + delta) >> n),
                    v_reinterpret_as_s16((b + delta) >> n));
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x16& a)
{
    v_uint16x16 delta = v256_setall_u16((short)(1 << (n-1)));
    v_pack_u_store(ptr, v_reinterpret_as_s16((a + delta) >> n));
}

template<int n> inline
v_uint8x32 v_rshr_pack_u(const v_int16x16& a, const v_int16x16& b)
{
    v_int16x16 delta = v256_setall_s16((short)(1 << (n-1)));
    return v_pack_u((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x16& a)
{
    v_int16x16 delta = v256_setall_s16((short)(1 << (n-1)));
    v_pack_u_store(ptr, (a + delta) >> n);
}

template<int n> inline
v_int8x32 v_rshr_pack(const v_int16x16& a, const v_int16x16& b)
{
    v_int16x16 delta = v256_setall_s16((short)(1 << (n-1)));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x16& a)
{
    v_int16x16 delta = v256_setall_s16((short)(1 << (n-1)));
    v_pack_store(ptr, (a + delta) >> n);
}

// 32
inline v_int16x16 v_pack(const v_int32x8& a, const v_int32x8& b)
{ return v_int16x16(_v256_shuffle_odd_64(_mm256_packs_epi32(a.val, b.val))); }

inline v_uint16x16 v_pack(const v_uint32x8& a, const v_uint32x8& b)
{ return v_uint16x16(_v256_shuffle_odd_64(_mm256_packus_epi32(a.val, b.val))); }

inline v_uint16x16 v_pack_u(const v_int32x8& a, const v_int32x8& b)
{ return v_pack(v_reinterpret_as_u32(a), v_reinterpret_as_u32(b)); }

inline void v_pack_store(short* ptr, const v_int32x8& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(ushort* ptr, const v_uint32x8& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_u_store(ushort* ptr, const v_int32x8& a)
{ v_store_low(ptr, v_pack_u(a, a)); }


template<int n> inline
v_uint16x16 v_rshr_pack(const v_uint32x8& a, const v_uint32x8& b)
{
    // we assume that n > 0, and so the shifted 32-bit values can be treated as signed numbers.
    v_uint32x8 delta = v256_setall_u32(1 << (n-1));
    return v_pack_u(v_reinterpret_as_s32((a + delta) >> n),
                    v_reinterpret_as_s32((b + delta) >> n));
}

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x8& a)
{
    v_uint32x8 delta = v256_setall_u32(1 << (n-1));
    v_pack_u_store(ptr, v_reinterpret_as_s32((a + delta) >> n));
}

template<int n> inline
v_uint16x16 v_rshr_pack_u(const v_int32x8& a, const v_int32x8& b)
{
    v_int32x8 delta = v256_setall_s32(1 << (n-1));
    return v_pack_u((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x8& a)
{
    v_int32x8 delta = v256_setall_s32(1 << (n-1));
    v_pack_u_store(ptr, (a + delta) >> n);
}

template<int n> inline
v_int16x16 v_rshr_pack(const v_int32x8& a, const v_int32x8& b)
{
    v_int32x8 delta = v256_setall_s32(1 << (n-1));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x8& a)
{
    v_int32x8 delta = v256_setall_s32(1 << (n-1));
    v_pack_store(ptr, (a + delta) >> n);
}

// 64
// Non-saturating pack
inline v_uint32x8 v_pack(const v_uint64x4& a, const v_uint64x4& b)
{
    __m256i a0 = _mm256_shuffle_epi32(a.val, _MM_SHUFFLE(0, 0, 2, 0));
    __m256i b0 = _mm256_shuffle_epi32(b.val, _MM_SHUFFLE(0, 0, 2, 0));
    __m256i ab = _mm256_unpacklo_epi64(a0, b0); // a0, a1, b0, b1, a2, a3, b2, b3
    return v_uint32x8(_v256_shuffle_odd_64(ab));
}

inline v_int32x8 v_pack(const v_int64x4& a, const v_int64x4& b)
{ return v_reinterpret_as_s32(v_pack(v_reinterpret_as_u64(a), v_reinterpret_as_u64(b))); }

inline void v_pack_store(unsigned* ptr, const v_uint64x4& a)
{
    __m256i a0 = _mm256_shuffle_epi32(a.val, _MM_SHUFFLE(0, 0, 2, 0));
    v_store_low(ptr, v_uint32x8(_v256_shuffle_odd_64(a0)));
}

inline void v_pack_store(int* ptr, const v_int64x4& b)
{ v_pack_store((unsigned*)ptr, v_reinterpret_as_u64(b)); }

template<int n> inline
v_uint32x8 v_rshr_pack(const v_uint64x4& a, const v_uint64x4& b)
{
    v_uint64x4 delta = v256_setall_u64((uint64)1 << (n-1));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x4& a)
{
    v_uint64x4 delta = v256_setall_u64((uint64)1 << (n-1));
    v_pack_store(ptr, (a + delta) >> n);
}

template<int n> inline
v_int32x8 v_rshr_pack(const v_int64x4& a, const v_int64x4& b)
{
    v_int64x4 delta = v256_setall_s64((int64)1 << (n-1));
    return v_pack((a + delta) >> n, (b + delta) >> n);
}

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x4& a)
{
    v_int64x4 delta = v256_setall_s64((int64)1 << (n-1));
    v_pack_store(ptr, (a + delta) >> n);
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


/** Reinterpret **/
// its up there with load and store operations

/* de&interleave */
#define OPENCV_HAL_IMPL_AVX_INTERLEAVE_2CH(_Tpvec, _Tp, suffix)                 \
    inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a, _Tpvec& b)       \
    { return v256_load_deinterleave_##suffix(ptr, a, b); }                      \
    inline void v_store_interleave(_Tp* ptr, const _Tpvec& a, const _Tpvec& b)  \
    { return v256_store_interleave_2ch(ptr, a, b); }

#define OPENCV_HAL_IMPL_AVX_INTERLEAVE_3CH(_Tpvec, _Tp, suffix)   \
    inline void v_load_deinterleave                               \
    (const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c)             \
    { return v256_load_deinterleave_##suffix(ptr, a, b, c); }     \
    inline void v_store_interleave                                \
    (_Tp* ptr, const _Tpvec& a,const _Tpvec& b, const _Tpvec& c)  \
    { return v256_store_interleave_##suffix(ptr, a, b, c); }

#define OPENCV_HAL_IMPL_AVX_INTERLEAVE_4CH(_Tpvec, _Tp, suffix)                    \
    inline void v_load_deinterleave                                                \
    (const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c, _Tpvec& d)                   \
    { return v256_load_deinterleave_##suffix(ptr, a, b, c, d); }                   \
    inline void v_store_interleave                                                 \
    (_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c, const _Tpvec& d) \
    { return v256_store_interleave_##suffix(ptr, a, b, c, d); }

#define OPENCV_HAL_IMPL_AVX_INTERLEAVE_3n4CH(_Tpvec, _Tp, suffix) \
    OPENCV_HAL_IMPL_AVX_INTERLEAVE_3CH(_Tpvec, _Tp, suffix)       \
    OPENCV_HAL_IMPL_AVX_INTERLEAVE_4CH(_Tpvec, _Tp, suffix)

#define OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(_Tpvec, _Tp, suffix) \
    OPENCV_HAL_IMPL_AVX_INTERLEAVE_2CH(_Tpvec, _Tp, suffix)     \
    OPENCV_HAL_IMPL_AVX_INTERLEAVE_3n4CH(_Tpvec, _Tp, suffix)

/* **** */
//
template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_2ch(_Tp* ptr, const _Tpvec& a, const _Tpvec& b)
{
    _Tpvec ab0, ab1;
    v_zip(a, b, ab0, ab1);
    v_store(ptr, ab0);
    v_store(ptr + _Tpvec::nlanes, ab1);
}

template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l4(const _Tp* ptr, _Tpvec& a, _Tpvec& b)
{
    _Tpvec ab0 = v256_load(ptr);
    _Tpvec ab1 = v256_load(ptr + _Tpvec::nlanes);
    _Tpvec ab00, ab11;
    v_recombine(ab0, ab1, ab00, ab11);
    v256_zip(ab00, ab11, a, b);
}

///
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l4(const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c)
{
    _Tpvec abc0 = v256_load(ptr);
    _Tpvec abc1 = v256_load(ptr + _Tpvec::nlanes);
    _Tpvec abc2 = v256_load(ptr + _Tpvec::nlanes * 2);

    _Tpvec ab0 = v256_combine_diagonal(abc0, abc1);
    _Tpvec bc1 = v256_combine_diagonal(abc1, abc2);
    _Tpvec ac1 = v256_reverse_64(v256_combine_diagonal(abc2, abc0));

    a = v256_unpacklo(ab0, ac1);
    c = v256_unpackhi(ac1, bc1);
    b = v256_alignr_64(bc1, ab0);
}


template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l4(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)
{
    _Tpvec ab0 = v256_unpacklo(a, b);
    _Tpvec bc1 = v256_unpackhi(b, c);
    _Tpvec ca10 = v256_swap_halves(v256_blend<0b1010>(c, a));

    v_store(ptr, v256_combine_diagonal(ab0, ca10));
    v_store(ptr + _Tpvec::nlanes, v256_combine_diagonal(bc1, ab0));
    v_store(ptr + _Tpvec::nlanes * 2, v256_combine_diagonal(ca10, bc1));
}

////
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l4(const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c, _Tpvec& d)
{
    _Tpvec abcd0 = v256_load(ptr);
    _Tpvec abcd1 = v256_load(ptr + _Tpvec::nlanes);
    _Tpvec abcd2 = v256_load(ptr + _Tpvec::nlanes * 2);
    _Tpvec abcd3 = v256_load(ptr + _Tpvec::nlanes * 3);

    _Tpvec cd0ab0 = v256_alignr_128(abcd0, abcd2);
    _Tpvec cd1ab1 = v256_alignr_128(abcd1, abcd3);

    _Tpvec ab0 = v256_combine_diagonal(abcd0, cd0ab0);
    _Tpvec ab1 = v256_combine_diagonal(abcd1, cd1ab1);
    _Tpvec cd0 = v256_combine_diagonal(cd0ab0, abcd2);
    _Tpvec cd1 = v256_combine_diagonal(cd1ab1, abcd3);

    v256_zip(ab0, ab1, a, b);
    v256_zip(cd0, cd1, c, d);
}

template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l4(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c, const _Tpvec& d)
{
    _Tpvec ab0, ab1, cd0, cd1;
    v256_zip(a, b, ab0, ab1);
    v256_zip(c, d, cd0, cd1);

    _Tpvec ab0cd0 = v256_alignr_128(ab0, cd0);
    _Tpvec ab1cd1 = v256_alignr_128(ab1, cd1);

    v_store(ptr, v256_combine_diagonal(ab0, ab0cd0));
    v_store(ptr + _Tpvec::nlanes, v256_combine_diagonal(ab1, ab1cd1));
    v_store(ptr + _Tpvec::nlanes * 2, v256_combine_diagonal(ab0cd0, cd0));
    v_store(ptr + _Tpvec::nlanes * 3, v256_combine_diagonal(ab1cd1, cd1));
}

OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_uint64x4,  uint64, l4)
OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_int64x4,   int64,  l4)
OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_float64x4, double, l4)

/* **** **** */
//
inline void v256_load_deinterleave_l8(const float* ptr, v_float32x8& a, v_float32x8& b)
{
    v_float32x8 ab0 = v256_load(ptr);
    v_float32x8 ab1 = v256_load(ptr + 8);

    v_float32x8 ab0ab2, ab1ab3;
    v_recombine(ab0, ab1, ab0ab2, ab1ab3);

    a.val = _mm256_shuffle_ps(ab0ab2.val, ab1ab3.val, _MM_SHUFFLE(2, 0, 2, 0));
    b.val = _mm256_shuffle_ps(ab0ab2.val, ab1ab3.val, _MM_SHUFFLE(3, 1, 3, 1));
}

template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l8(const _Tp* ptr, _Tpvec& a, _Tpvec& b)
{
    v_float32x8 fa, fb;
    v256_load_deinterleave_l8((float*)ptr, fa, fb);
    a.val = v_reinterpret_as_u32(fa).val;
    b.val = v_reinterpret_as_u32(fb).val;
}
///
template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l8(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)
{
    _Tpvec ab0, ab1, bc0, bc1;
    v256_zip(a, b, ab0, ab1);
    v256_zip(b, c, bc0, bc1);

    _Tpvec cazg = v256_blend<0b10101010>(c, a);
    _Tpvec abc0abc1(_mm256_unpacklo_epi64(ab0.val, cazg.val));
    _Tpvec abc1abc2(_mm256_unpackhi_epi64(cazg.val, bc1.val));
    _Tpvec abc2abc0 = v256_reverse_64(v256_blend<0b11001100>(ab1, bc0));

    _Tpvec abc0 = v256_combine_diagonal(abc0abc1, abc2abc0);
    _Tpvec abc1 = v256_combine_diagonal(abc1abc2, abc0abc1);
    _Tpvec abc2 = v256_combine_diagonal(abc2abc0, abc1abc2);

    v_store(ptr, abc0);
    v_store(ptr + _Tpvec::nlanes, abc1);
    v_store(ptr + _Tpvec::nlanes * 2, abc2);
}

inline void v256_store_interleave_l8(float* ptr, const v_float32x8& a, const v_float32x8& b, const v_float32x8& c)
{
    v_float32x8 ab0, ab1, bc0, bc1;
    v256_zip(a, b, ab0, ab1);
    v256_zip(b, c, bc0, bc1);

    v_float32x8 cazg = v256_blend<0b10101010>(c, a);
    v_float32x8 abc0abc1(_mm256_shuffle_ps(ab0.val, cazg.val, _MM_SHUFFLE(1, 0, 1, 0)));
    v_float32x8 abc1abc2(_mm256_shuffle_ps(cazg.val, bc1.val, _MM_SHUFFLE(3, 2, 3, 2)));

    v_float32x8 abc0abc2(_mm256_shuffle_ps(bc0.val, ab1.val, _MM_SHUFFLE(1, 0, 3, 2)));
    v_float32x8 abc2abc0 = v256_swap_halves(abc0abc2);

    v_float32x8 abc0 = v256_combine_diagonal(abc0abc1, abc2abc0);
    v_float32x8 abc1 = v256_combine_diagonal(abc1abc2, abc0abc1);
    v_float32x8 abc2 = v256_combine_diagonal(abc2abc0, abc1abc2);

    v_store(ptr, abc0);
    v_store(ptr + 8, abc1);
    v_store(ptr + 16, abc2);
}

template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l8(const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c)
{
    _Tpvec abc02 = v256_load(ptr);
    _Tpvec abc1  = v256_load(ptr + _Tpvec::nlanes);
    _Tpvec abc20 = v256_load(ptr + _Tpvec::nlanes * 2);

    _Tpvec abc2 = v256_alignr_128(abc02, abc20);
    _Tpvec abc0 = v256_combine_diagonal(abc02, abc20);

    a = v256_blend<0b10010010>(abc0, abc1);
    a = v256_blend<0b01000100>(a, abc2);

    b = v256_blend<0b00100100>(abc0, abc1);
    b = v256_blend<0b10011001>(b, abc2);

    c = v256_blend<0b01001001>(abc0, abc1);
    c = v256_blend<0b00100010>(c, abc2);

    a = v256_shuffle<_MM_SHUFFLE(1, 2, 3, 0)>(a);
    b = v256_shuffle<_MM_SHUFFLE(2, 3, 0, 1)>(b);
    c = v256_shuffle<_MM_SHUFFLE(3, 0, 1, 2)>(c);
}
/////
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l8(const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c, _Tpvec& d)
{
    _Tpvec ab0, ab1, cd0, cd1;
    v256_load_deinterleave_l4(ptr, ab0, cd0, ab1, cd1);
    v256_zip(ab0, ab1, a, b);
    v256_zip(cd0, cd1, c, d);
}

template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l8(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c, const _Tpvec& d)
{
    _Tpvec ac0, ac1, bd0, bd1;
    v256_zip(a, c, ac0, ac1);
    v256_zip(b, d, bd0, bd1);

    _Tpvec abcd0, abcd1, abcd2, abcd3;
    v256_zip(ac0, bd0, abcd0, abcd1);
    v256_zip(ac1, bd1, abcd2, abcd3);

    _Tpvec abcd01, abcd23, abcd45, abcd67;
    v_recombine(abcd0, abcd1, abcd01, abcd45);
    v_recombine(abcd2, abcd3, abcd23, abcd67);

    v_store(ptr, abcd01);
    v_store(ptr + _Tpvec::nlanes, abcd23);
    v_store(ptr + _Tpvec::nlanes * 2, abcd45);
    v_store(ptr + _Tpvec::nlanes * 3, abcd67);
}

OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_uint32x8,  unsigned, l8)
OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_int32x8,   int,      l8)
OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_float32x8, float,    l8)

/* ********  ******** */
//
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l16(const _Tp* ptr, _Tpvec& a, _Tpvec& b)
{
    const __m256i sep = _mm256_setr_epi8(
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15,
        0, 1, 4, 5, 8, 9, 12, 13, 2, 3, 6, 7, 10, 11, 14, 15
    );

    _Tpvec ab0, ab1;
    v_recombine(v256_load(ptr), v256_load(ptr + _Tpvec::nlanes), ab0, ab1);

    __m256i a0b0 = _mm256_shuffle_epi8(ab0.val, sep);
    __m256i a1b1 = _mm256_shuffle_epi8(ab1.val, sep);

    a.val = _mm256_unpacklo_epi64(a0b0, a1b1);
    b.val = _mm256_unpackhi_epi64(a0b0, a1b1);
}
///
template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l16(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)
{
    v_uint32x8 ab0 = v_reinterpret_as_u32(v256_unpacklo(a, b));
    v_uint32x8 ab1 = v_reinterpret_as_u32(v256_unpackhi(a, b));
    v_uint32x8 bc0 = v_reinterpret_as_u32(v256_unpacklo(b, c));
    v_uint32x8 bc1 = v_reinterpret_as_u32(v256_unpackhi(b, c));

    v_uint32x8 cazg = v_reinterpret_as_u32(v256_blend<0b10101010>(c, a));
               cazg = v256_shuffle<_MM_SHUFFLE(2, 1, 0, 3)>(cazg);

    v_uint32x8 ac1ab1 = v256_blend<0b10101010>(ab1, bc1);
               ac1ab1 = v256_shuffle<_MM_SHUFFLE(2, 1, 0, 3)>(ac1ab1);

    v_uint32x8 abc001 = v256_blend<0b10101010>(ab0, cazg);
    v_uint32x8 cabc0 = v256_blend<0b10101010>(cazg, bc0);

    v_uint32x8 cabc1 = v256_unpacklo(cabc0, ac1ab1);
    v_uint32x8 bcab0 = v256_unpackhi(cabc1, abc001);

    v_uint64x4 abc01 = v256_unpacklo(v_reinterpret_as_u64(abc001), v_reinterpret_as_u64(bcab0));
    v_uint64x4 abc21 = v256_unpackhi(v_reinterpret_as_u64(cabc0), v_reinterpret_as_u64(bcab0));
               abc21 = v256_swap_halves(abc21);
    v_uint64x4 abc12 = v_reinterpret_as_u64(v256_alignr_64(cabc1, ac1ab1));

    v_uint64x4 abc0 = v256_combine_diagonal(abc01, abc21);
    v_uint64x4 abc1 = v256_combine_diagonal(abc12, abc01);
    v_uint64x4 abc2 = v256_combine_diagonal(abc21, abc12);

    v_store(ptr, _Tpvec(abc0.val));
    v_store(ptr + _Tpvec::nlanes, _Tpvec(abc1.val));
    v_store(ptr + _Tpvec::nlanes * 2, _Tpvec(abc2.val));
}
// todo:
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l16(const _Tp*, _Tpvec&, _Tpvec&, _Tpvec&)
{}
////
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l16(const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c, _Tpvec& d)
{
    _Tpvec ab0, ab1, cd0, cd1;
    v256_load_deinterleave_l8(ptr, ab0, cd0, ab1, cd1);
    v256_zip(ab0, ab1, a, b);
    v256_zip(cd0, cd1, c, d);
}

template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l16(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c, const _Tpvec& d)
{ v256_store_interleave_l8(ptr, a, b, c, d); }

OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_uint16x16,  ushort, l16)
OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_int16x16,   short,  l16)

/* **************** **************** */
//
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l32(const _Tp* ptr, _Tpvec& a, _Tpvec& b)
{
    const __m256i sep = _mm256_setr_epi8(
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15,
        0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15
    );

    _Tpvec ab0, ab1;
    v_recombine(v256_load(ptr), v256_load(ptr + _Tpvec::nlanes), ab0, ab1);

    __m256i a0b0 = _mm256_shuffle_epi8(ab0.val, sep);
    __m256i a1b1 = _mm256_shuffle_epi8(ab1.val, sep);

    a.val = _mm256_unpacklo_epi64(a0b0, a1b1);
    b.val = _mm256_unpackhi_epi64(a0b0, a1b1);
}

/// todo
template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l32(_Tp*, const _Tpvec&, const _Tpvec&, const _Tpvec&)
{}
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l32(const _Tp*, _Tpvec&, _Tpvec&, _Tpvec&)
{}
////
template<typename _Tp, typename _Tpvec>
inline void v256_load_deinterleave_l32(const _Tp* ptr, _Tpvec& a, _Tpvec& b, _Tpvec& c, _Tpvec& d)
{
    const __m256i sep = _mm256_setr_epi8(
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15,
        0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15
    );

    _Tpvec abcd0, abcd1, abcd2, abcd3;
    v_recombine(v256_load(ptr), v256_load(ptr + _Tpvec::nlanes * 2), abcd0, abcd1);
    v_recombine(v256_load(ptr + _Tpvec::nlanes), v256_load(ptr + _Tpvec::nlanes * 3), abcd2, abcd3);

    __m256i ab0cd0 = _mm256_shuffle_epi8(abcd0.val, sep);
    __m256i ab1cd1 = _mm256_shuffle_epi8(abcd1.val, sep);
    __m256i ab2cd2 = _mm256_shuffle_epi8(abcd2.val, sep);
    __m256i ab3cd3 = _mm256_shuffle_epi8(abcd3.val, sep);

    __m256i ab0 = _mm256_unpacklo_epi32(ab0cd0, ab1cd1);
    __m256i ab1 = _mm256_unpacklo_epi32(ab2cd2, ab3cd3);
    __m256i cd0 = _mm256_unpackhi_epi32(ab0cd0, ab1cd1);
    __m256i cd1 = _mm256_unpackhi_epi32(ab2cd2, ab3cd3);

    a.val = _mm256_unpacklo_epi64(ab0, ab1);
    b.val = _mm256_unpackhi_epi64(ab0, ab1);
    c.val = _mm256_unpacklo_epi64(cd0, cd1);
    d.val = _mm256_unpackhi_epi64(cd0, cd1);
}

template<typename _Tp, typename _Tpvec>
inline void v256_store_interleave_l32(_Tp* ptr, const _Tpvec& a, const _Tpvec& b, const _Tpvec& c, const _Tpvec& d)
{ v256_store_interleave_l8(ptr, a, b, c, d); }

OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_uint8x32,  uchar, l32)
OPENCV_HAL_IMPL_AVX_INTERLEAVE_ACH(v_int8x32,   schar, l32)

inline void v256_cleanup() { _mm256_zeroupper(); }

//! @name Check SIMD256 support
//! @{
//! @brief Check CPU capability of SIMD operation
static inline bool hasSIMD256()
{
    return (CV_CPU_HAS_SUPPORT_AVX2) ? true : false;
}
//! @}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // OPENCV_HAL_INTRIN_AVX_HPP
