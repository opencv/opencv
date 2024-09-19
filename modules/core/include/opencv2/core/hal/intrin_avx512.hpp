// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_INTRIN_AVX512_HPP
#define OPENCV_HAL_INTRIN_AVX512_HPP

#if defined(_MSC_VER) && (_MSC_VER < 1920/*MSVS2019*/)
# pragma warning(disable:4146)  // unary minus operator applied to unsigned type, result still unsigned
# pragma warning(disable:4309)  // 'argument': truncation of constant value
# pragma warning(disable:4310)  // cast truncates constant value
#endif

#define CVT_ROUND_MODES_IMPLEMENTED 0

#define CV_SIMD512 1
#define CV_SIMD512_64F 1
#define CV_SIMD512_FP16 0  // no native operations with FP16 type. Only load/store from float32x8 are available (if CV_FP16 == 1)

#define _v512_set_epu64(a7, a6, a5, a4, a3, a2, a1, a0) _mm512_set_epi64((int64)(a7),(int64)(a6),(int64)(a5),(int64)(a4),(int64)(a3),(int64)(a2),(int64)(a1),(int64)(a0))
#define _v512_set_epu32(a15, a14, a13, a12, a11, a10,  a9,  a8,  a7,  a6,  a5,  a4,  a3,  a2,  a1,  a0) \
        _mm512_set_epi64(((int64)(a15)<<32)|(int64)(a14), ((int64)(a13)<<32)|(int64)(a12), ((int64)(a11)<<32)|(int64)(a10), ((int64)( a9)<<32)|(int64)( a8), \
                         ((int64)( a7)<<32)|(int64)( a6), ((int64)( a5)<<32)|(int64)( a4), ((int64)( a3)<<32)|(int64)( a2), ((int64)( a1)<<32)|(int64)( a0))
#define _v512_set_epu16(a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16, \
                        a15, a14, a13, a12, a11, a10,  a9,  a8,  a7,  a6,  a5,  a4,  a3,  a2,  a1,  a0) \
        _v512_set_epu32(((unsigned)(a31)<<16)|(unsigned)(a30), ((unsigned)(a29)<<16)|(unsigned)(a28), ((unsigned)(a27)<<16)|(unsigned)(a26), ((unsigned)(a25)<<16)|(unsigned)(a24), \
                        ((unsigned)(a23)<<16)|(unsigned)(a22), ((unsigned)(a21)<<16)|(unsigned)(a20), ((unsigned)(a19)<<16)|(unsigned)(a18), ((unsigned)(a17)<<16)|(unsigned)(a16), \
                        ((unsigned)(a15)<<16)|(unsigned)(a14), ((unsigned)(a13)<<16)|(unsigned)(a12), ((unsigned)(a11)<<16)|(unsigned)(a10), ((unsigned)( a9)<<16)|(unsigned)( a8), \
                        ((unsigned)( a7)<<16)|(unsigned)( a6), ((unsigned)( a5)<<16)|(unsigned)( a4), ((unsigned)( a3)<<16)|(unsigned)( a2), ((unsigned)( a1)<<16)|(unsigned)( a0))
#define _v512_set_epu8(a63, a62, a61, a60, a59, a58, a57, a56, a55, a54, a53, a52, a51, a50, a49, a48, \
                       a47, a46, a45, a44, a43, a42, a41, a40, a39, a38, a37, a36, a35, a34, a33, a32, \
                       a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16, \
                       a15, a14, a13, a12, a11, a10,  a9,  a8,  a7,  a6,  a5,  a4,  a3,  a2,  a1,  a0) \
        _v512_set_epu32(((unsigned)(a63)<<24)|((unsigned)(a62)<<16)|((unsigned)(a61)<<8)|(unsigned)(a60),((unsigned)(a59)<<24)|((unsigned)(a58)<<16)|((unsigned)(a57)<<8)|(unsigned)(a56), \
                        ((unsigned)(a55)<<24)|((unsigned)(a54)<<16)|((unsigned)(a53)<<8)|(unsigned)(a52),((unsigned)(a51)<<24)|((unsigned)(a50)<<16)|((unsigned)(a49)<<8)|(unsigned)(a48), \
                        ((unsigned)(a47)<<24)|((unsigned)(a46)<<16)|((unsigned)(a45)<<8)|(unsigned)(a44),((unsigned)(a43)<<24)|((unsigned)(a42)<<16)|((unsigned)(a41)<<8)|(unsigned)(a40), \
                        ((unsigned)(a39)<<24)|((unsigned)(a38)<<16)|((unsigned)(a37)<<8)|(unsigned)(a36),((unsigned)(a35)<<24)|((unsigned)(a34)<<16)|((unsigned)(a33)<<8)|(unsigned)(a32), \
                        ((unsigned)(a31)<<24)|((unsigned)(a30)<<16)|((unsigned)(a29)<<8)|(unsigned)(a28),((unsigned)(a27)<<24)|((unsigned)(a26)<<16)|((unsigned)(a25)<<8)|(unsigned)(a24), \
                        ((unsigned)(a23)<<24)|((unsigned)(a22)<<16)|((unsigned)(a21)<<8)|(unsigned)(a20),((unsigned)(a19)<<24)|((unsigned)(a18)<<16)|((unsigned)(a17)<<8)|(unsigned)(a16), \
                        ((unsigned)(a15)<<24)|((unsigned)(a14)<<16)|((unsigned)(a13)<<8)|(unsigned)(a12),((unsigned)(a11)<<24)|((unsigned)(a10)<<16)|((unsigned)( a9)<<8)|(unsigned)( a8), \
                        ((unsigned)( a7)<<24)|((unsigned)( a6)<<16)|((unsigned)( a5)<<8)|(unsigned)( a4),((unsigned)( a3)<<24)|((unsigned)( a2)<<16)|((unsigned)( a1)<<8)|(unsigned)( a0))
#define _v512_set_epi8(a63, a62, a61, a60, a59, a58, a57, a56, a55, a54, a53, a52, a51, a50, a49, a48, \
                       a47, a46, a45, a44, a43, a42, a41, a40, a39, a38, a37, a36, a35, a34, a33, a32, \
                       a31, a30, a29, a28, a27, a26, a25, a24, a23, a22, a21, a20, a19, a18, a17, a16, \
                       a15, a14, a13, a12, a11, a10,  a9,  a8,  a7,  a6,  a5,  a4,  a3,  a2,  a1,  a0) \
        _v512_set_epu8((uchar)(a63), (uchar)(a62), (uchar)(a61), (uchar)(a60), (uchar)(a59), (uchar)(a58), (uchar)(a57), (uchar)(a56), \
                       (uchar)(a55), (uchar)(a54), (uchar)(a53), (uchar)(a52), (uchar)(a51), (uchar)(a50), (uchar)(a49), (uchar)(a48), \
                       (uchar)(a47), (uchar)(a46), (uchar)(a45), (uchar)(a44), (uchar)(a43), (uchar)(a42), (uchar)(a41), (uchar)(a40), \
                       (uchar)(a39), (uchar)(a38), (uchar)(a37), (uchar)(a36), (uchar)(a35), (uchar)(a34), (uchar)(a33), (uchar)(a32), \
                       (uchar)(a31), (uchar)(a30), (uchar)(a29), (uchar)(a28), (uchar)(a27), (uchar)(a26), (uchar)(a25), (uchar)(a24), \
                       (uchar)(a23), (uchar)(a22), (uchar)(a21), (uchar)(a20), (uchar)(a19), (uchar)(a18), (uchar)(a17), (uchar)(a16), \
                       (uchar)(a15), (uchar)(a14), (uchar)(a13), (uchar)(a12), (uchar)(a11), (uchar)(a10), (uchar)( a9), (uchar)( a8), \
                       (uchar)( a7), (uchar)( a6), (uchar)( a5), (uchar)( a4), (uchar)( a3), (uchar)( a2), (uchar)( a1), (uchar)( a0))

#ifndef _mm512_cvtpd_pslo
#ifdef _mm512_zextsi256_si512
#define _mm512_cvtpd_pslo(a) _mm512_zextps256_ps512(_mm512_cvtpd_ps(a))
#else
//if preferred way to extend with zeros is unavailable
#define _mm512_cvtpd_pslo(a) _mm512_castps256_ps512(_mm512_cvtpd_ps(a))
#endif
#endif
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
        val = _v512_set_epu8(v63, v62, v61, v60, v59, v58, v57, v56, v55, v54, v53, v52, v51, v50, v49, v48,
                             v47, v46, v45, v44, v43, v42, v41, v40, v39, v38, v37, v36, v35, v34, v33, v32,
                             v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16,
                             v15, v14, v13, v12, v11, v10, v9,  v8,  v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
    }
    v_uint8x64() {}

    static inline v_uint8x64 zero() { return v_uint8x64(_mm512_setzero_si512()); }

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
        val = _v512_set_epi8(v63, v62, v61, v60, v59, v58, v57, v56, v55, v54, v53, v52, v51, v50, v49, v48,
                             v47, v46, v45, v44, v43, v42, v41, v40, v39, v38, v37, v36, v35, v34, v33, v32,
                             v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16,
                             v15, v14, v13, v12, v11, v10, v9,  v8,  v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
    }
    v_int8x64() {}

    static inline v_int8x64 zero() { return v_int8x64(_mm512_setzero_si512()); }

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
        val = _v512_set_epu16(v31, v30, v29, v28, v27, v26, v25, v24, v23, v22, v21, v20, v19, v18, v17, v16,
                              v15, v14, v13, v12, v11, v10, v9,  v8,  v7,  v6,  v5,  v4,  v3,  v2,  v1,  v0);
    }
    v_uint16x32() {}

    static inline v_uint16x32 zero() { return v_uint16x32(_mm512_setzero_si512()); }

    ushort get0() const { return (ushort)_v_cvtsi512_si32(val); }
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
        val = _v512_set_epu16((ushort)v31, (ushort)v30, (ushort)v29, (ushort)v28, (ushort)v27, (ushort)v26, (ushort)v25, (ushort)v24,
                              (ushort)v23, (ushort)v22, (ushort)v21, (ushort)v20, (ushort)v19, (ushort)v18, (ushort)v17, (ushort)v16,
                              (ushort)v15, (ushort)v14, (ushort)v13, (ushort)v12, (ushort)v11, (ushort)v10, (ushort)v9 , (ushort)v8,
                              (ushort)v7 , (ushort)v6 , (ushort)v5 , (ushort)v4 , (ushort)v3 , (ushort)v2 , (ushort)v1 , (ushort)v0);
    }
    v_int16x32() {}

    static inline v_int16x32 zero() { return v_int16x32(_mm512_setzero_si512()); }

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
        val = _mm512_setr_epi32((int)v0,  (int)v1,  (int)v2,  (int)v3, (int)v4,  (int)v5,  (int)v6,  (int)v7,
                                (int)v8,  (int)v9,  (int)v10, (int)v11, (int)v12, (int)v13, (int)v14, (int)v15);
    }
    v_uint32x16() {}

    static inline v_uint32x16 zero() { return v_uint32x16(_mm512_setzero_si512()); }

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
    v_int32x16() {}

    static inline v_int32x16 zero() { return v_int32x16(_mm512_setzero_si512()); }

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
    v_float32x16() {}

    static inline v_float32x16 zero() { return v_float32x16(_mm512_setzero_ps()); }

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
    v_uint64x8() {}

    static inline v_uint64x8 zero() { return v_uint64x8(_mm512_setzero_si512()); }

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
    v_int64x8() {}

    static inline v_int64x8 zero() { return v_int64x8(_mm512_setzero_si512()); }

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
    v_float64x8() {}

    static inline v_float64x8 zero() { return v_float64x8(_mm512_setzero_pd()); }

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
    inline _Tpvec v_setzero(_Tpvec /*unused*/)                                     \
    { return v512_setzero_##suffix(); }                                            \
    inline _Tpvec v_setall(_Tp v, _Tpvec /*unused*/)                               \
    { return v512_setall_##suffix(v); }                                            \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint8x64,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int8x64,    suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint16x32,  suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int16x32,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint32x16,  suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int32x16,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_uint64x8,   suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_int64x8,    suffix, OPENCV_HAL_NOP)      \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_float32x16, suffix, _mm512_castps_si512) \
    OPENCV_HAL_IMPL_AVX512_CAST(_Tpvec, v_float64x8,  suffix, _mm512_castpd_si512)

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
    inline _Tpvec v_setzero(_Tpvec /*unused*/)                              \
    { return v512_setzero_##suffix(); }                                     \
    inline _Tpvec v_setall(_Tp v, _Tpvec /*unused*/)                        \
    { return v512_setall_##suffix(v); }                                     \
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
inline v_float32x16 v512_load_expand(const hfloat* ptr)
{
    return v_float32x16(_mm512_cvtph_ps(_mm256_loadu_si256((const __m256i*)ptr)));
}

inline void v_pack_store(hfloat* ptr, const v_float32x16& a)
{
    __m256i ah = _mm512_cvtps_ph(a.val, 0);
    _mm256_storeu_si256((__m256i*)ptr, ah);
}

/* Recombine & ZIP */
inline void v_zip(const v_int8x64& a, const v_int8x64& b, v_int8x64& ab0, v_int8x64& ab1)
{
#if CV_AVX_512VBMI
    __m512i mask0 = _v512_set_epu8( 95,  31,  94,  30,  93,  29,  92,  28,  91,  27,  90,  26,  89,  25,  88,  24,
                                    87,  23,  86,  22,  85,  21,  84,  20,  83,  19,  82,  18,  81,  17,  80,  16,
                                    79,  15,  78,  14,  77,  13,  76,  12,  75,  11,  74,  10,  73,   9,  72,   8,
                                    71,   7,  70,   6,  69,   5,  68,   4,  67,   3,  66,   2,  65,   1,  64,   0);
    ab0 = v_int8x64(_mm512_permutex2var_epi8(a.val, mask0, b.val));
    __m512i mask1 = _v512_set_epu8(127,  63, 126,  62, 125,  61, 124,  60, 123,  59, 122,  58, 121,  57, 120,  56,
                                   119,  55, 118,  54, 117,  53, 116,  52, 115,  51, 114,  50, 113,  49, 112,  48,
                                   111,  47, 110,  46, 109,  45, 108,  44, 107,  43, 106,  42, 105,  41, 104,  40,
                                   103,  39, 102,  38, 101,  37, 100,  36,  99,  35,  98,  34,  97,  33,  96,  32);
    ab1 = v_int8x64(_mm512_permutex2var_epi8(a.val, mask1, b.val));
#else
    __m512i low  = _mm512_unpacklo_epi8(a.val, b.val);
    __m512i high = _mm512_unpackhi_epi8(a.val, b.val);
    ab0 = v_int8x64(_mm512_permutex2var_epi64(low, _v512_set_epu64(11, 10, 3, 2,  9,  8, 1, 0), high));
    ab1 = v_int8x64(_mm512_permutex2var_epi64(low, _v512_set_epu64(15, 14, 7, 6, 13, 12, 5, 4), high));
#endif
}
inline void v_zip(const v_int16x32& a, const v_int16x32& b, v_int16x32& ab0, v_int16x32& ab1)
{
    __m512i mask0 = _v512_set_epu16(47, 15, 46, 14, 45, 13, 44, 12, 43, 11, 42, 10, 41,  9, 40,  8,
                                    39,  7, 38,  6, 37,  5, 36,  4, 35,  3, 34,  2, 33,  1, 32,  0);
    ab0 = v_int16x32(_mm512_permutex2var_epi16(a.val, mask0, b.val));
    __m512i mask1 = _v512_set_epu16(63, 31, 62, 30, 61, 29, 60, 28, 59, 27, 58, 26, 57, 25, 56, 24,
                                    55, 23, 54, 22, 53, 21, 52, 20, 51, 19, 50, 18, 49, 17, 48, 16);
    ab1 = v_int16x32(_mm512_permutex2var_epi16(a.val, mask1, b.val));
}
inline void v_zip(const v_int32x16& a, const v_int32x16& b, v_int32x16& ab0, v_int32x16& ab1)
{
    __m512i mask0 = _v512_set_epu32(23,  7, 22,  6, 21,  5, 20,  4, 19,  3, 18,  2, 17, 1, 16, 0);
    ab0 = v_int32x16(_mm512_permutex2var_epi32(a.val, mask0, b.val));
    __m512i mask1 = _v512_set_epu32(31, 15, 30, 14, 29, 13, 28, 12, 27, 11, 26, 10, 25, 9, 24, 8);
    ab1 = v_int32x16(_mm512_permutex2var_epi32(a.val, mask1, b.val));
}
inline void v_zip(const v_int64x8& a, const v_int64x8& b, v_int64x8& ab0, v_int64x8& ab1)
{
    __m512i mask0 = _v512_set_epu64(11, 3, 10, 2,  9, 1,  8, 0);
    ab0 = v_int64x8(_mm512_permutex2var_epi64(a.val, mask0, b.val));
    __m512i mask1 = _v512_set_epu64(15, 7, 14, 6, 13, 5, 12, 4);
    ab1 = v_int64x8(_mm512_permutex2var_epi64(a.val, mask1, b.val));
}

inline void v_zip(const v_uint8x64&  a, const v_uint8x64&  b, v_uint8x64& ab0, v_uint8x64& ab1)
{
    v_int8x64 i0, i1;
    v_zip(v_reinterpret_as_s8(a), v_reinterpret_as_s8(b), i0, i1);
    ab0 = v_reinterpret_as_u8(i0);
    ab1 = v_reinterpret_as_u8(i1);
}
inline void v_zip(const v_uint16x32&  a, const v_uint16x32&  b, v_uint16x32& ab0, v_uint16x32& ab1)
{
    v_int16x32 i0, i1;
    v_zip(v_reinterpret_as_s16(a), v_reinterpret_as_s16(b), i0, i1);
    ab0 = v_reinterpret_as_u16(i0);
    ab1 = v_reinterpret_as_u16(i1);
}
inline void v_zip(const v_uint32x16&  a, const v_uint32x16&  b, v_uint32x16& ab0, v_uint32x16& ab1)
{
    v_int32x16 i0, i1;
    v_zip(v_reinterpret_as_s32(a), v_reinterpret_as_s32(b), i0, i1);
    ab0 = v_reinterpret_as_u32(i0);
    ab1 = v_reinterpret_as_u32(i1);
}
inline void v_zip(const v_uint64x8&  a, const v_uint64x8&  b, v_uint64x8& ab0, v_uint64x8& ab1)
{
    v_int64x8 i0, i1;
    v_zip(v_reinterpret_as_s64(a), v_reinterpret_as_s64(b), i0, i1);
    ab0 = v_reinterpret_as_u64(i0);
    ab1 = v_reinterpret_as_u64(i1);
}
inline void v_zip(const v_float32x16&  a, const v_float32x16&  b, v_float32x16& ab0, v_float32x16& ab1)
{
    v_int32x16 i0, i1;
    v_zip(v_reinterpret_as_s32(a), v_reinterpret_as_s32(b), i0, i1);
    ab0 = v_reinterpret_as_f32(i0);
    ab1 = v_reinterpret_as_f32(i1);
}
inline void v_zip(const v_float64x8&  a, const v_float64x8&  b, v_float64x8& ab0, v_float64x8& ab1)
{
    v_int64x8 i0, i1;
    v_zip(v_reinterpret_as_s64(a), v_reinterpret_as_s64(b), i0, i1);
    ab0 = v_reinterpret_as_f64(i0);
    ab1 = v_reinterpret_as_f64(i1);
}

#define OPENCV_HAL_IMPL_AVX512_COMBINE(_Tpvec, suffix)                                    \
    inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)                         \
    { return _Tpvec(_v512_combine(_v512_extract_low(a.val), _v512_extract_low(b.val))); } \
    inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)                        \
    { return _Tpvec(_v512_insert(b.val, _v512_extract_high(a.val))); }                    \
    inline void v_recombine(const _Tpvec& a, const _Tpvec& b,                             \
                                  _Tpvec& c, _Tpvec& d)                                   \
    {                                                                                     \
        c.val = _v512_combine(_v512_extract_low(a.val),_v512_extract_low(b.val));         \
        d.val = _v512_insert(b.val,_v512_extract_high(a.val));                            \
    }


OPENCV_HAL_IMPL_AVX512_COMBINE(v_uint8x64,   epi8)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_int8x64,    epi8)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_uint16x32,  epi16)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_int16x32,   epi16)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_uint32x16,  epi32)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_int32x16,   epi32)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_uint64x8,   epi64)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_int64x8,    epi64)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_float32x16, ps)
OPENCV_HAL_IMPL_AVX512_COMBINE(v_float64x8,  pd)

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
    return v_uint8x64(_mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, p0, p1));
}
inline v_int8x64 v_mul_wrap(const v_int8x64& a, const v_int8x64& b)
{
    return v_reinterpret_as_s8(v_mul_wrap(v_reinterpret_as_u8(a), v_reinterpret_as_u8(b)));
}

#define OPENCV_HAL_IMPL_AVX512_BIN_OP(bin_op, _Tpvec, intrin)            \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)               \
    { return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_uint32x16, _mm512_add_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_uint32x16, _mm512_sub_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_int32x16, _mm512_add_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_int32x16, _mm512_sub_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_uint64x8, _mm512_add_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_uint64x8, _mm512_sub_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_int64x8, _mm512_add_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_int64x8, _mm512_sub_epi64)

OPENCV_HAL_IMPL_AVX512_BIN_OP(v_mul, v_uint32x16, _mm512_mullo_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_mul, v_int32x16, _mm512_mullo_epi32)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_mul, v_uint64x8, _mm512_mullo_epi64)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_mul, v_int64x8, _mm512_mullo_epi64)

/** Saturating arithmetics **/
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_uint8x64,  _mm512_adds_epu8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_uint8x64,  _mm512_subs_epu8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_int8x64,   _mm512_adds_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_int8x64,   _mm512_subs_epi8)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_uint16x32, _mm512_adds_epu16)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_uint16x32, _mm512_subs_epu16)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_int16x32,  _mm512_adds_epi16)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_int16x32,  _mm512_subs_epi16)

OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_float32x16, _mm512_add_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_float32x16, _mm512_sub_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_mul, v_float32x16, _mm512_mul_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_div, v_float32x16, _mm512_div_ps)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_add, v_float64x8, _mm512_add_pd)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_sub, v_float64x8, _mm512_sub_pd)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_mul, v_float64x8, _mm512_mul_pd)
OPENCV_HAL_IMPL_AVX512_BIN_OP(v_div, v_float64x8, _mm512_div_pd)

// saturating multiply
inline v_uint8x64 v_mul(const v_uint8x64& a, const v_uint8x64& b)
{
    v_uint16x32 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_int8x64 v_mul(const v_int8x64& a, const v_int8x64& b)
{
    v_int16x32 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_uint16x32 v_mul(const v_uint16x32& a, const v_uint16x32& b)
{
    __m512i pl = _mm512_mullo_epi16(a.val, b.val);
    __m512i ph = _mm512_mulhi_epu16(a.val, b.val);
    __m512i p0 = _mm512_unpacklo_epi16(pl, ph);
    __m512i p1 = _mm512_unpackhi_epi16(pl, ph);

    const __m512i m = _mm512_set1_epi32(65535);
    return v_uint16x32(_mm512_packus_epi32(_mm512_min_epu32(p0, m), _mm512_min_epu32(p1, m)));
}
inline v_int16x32 v_mul(const v_int16x32& a, const v_int16x32& b)
{
    __m512i pl = _mm512_mullo_epi16(a.val, b.val);
    __m512i ph = _mm512_mulhi_epi16(a.val, b.val);
    __m512i p0 = _mm512_unpacklo_epi16(pl, ph);
    __m512i p1 = _mm512_unpackhi_epi16(pl, ph);
    return v_int16x32(_mm512_packs_epi32(p0, p1));
}

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
    v_uint16x32 v0, v1;
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
    inline _Tpuvec v_shl(const _Tpuvec& a, int imm)               \
    { return _Tpuvec(_mm512_slli_##suffix(a.val, imm)); }         \
    inline _Tpsvec v_shl(const _Tpsvec& a, int imm)               \
    { return _Tpsvec(_mm512_slli_##suffix(a.val, imm)); }         \
    inline _Tpuvec v_shr(const _Tpuvec& a, int imm)               \
    { return _Tpuvec(_mm512_srli_##suffix(a.val, imm)); }         \
    inline _Tpsvec v_shr(const _Tpsvec& a, int imm)               \
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
    OPENCV_HAL_IMPL_AVX512_BIN_OP(v_and, _Tpvec, _mm512_and_##suffix)  \
    OPENCV_HAL_IMPL_AVX512_BIN_OP(v_or, _Tpvec, _mm512_or_##suffix)    \
    OPENCV_HAL_IMPL_AVX512_BIN_OP(v_xor, _Tpvec, _mm512_xor_##suffix)  \
    inline _Tpvec v_not(const _Tpvec& a)                               \
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
#define OPENCV_HAL_IMPL_AVX512_SELECT(_Tpvec, suffix, zsuf)                      \
    inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
    { return _Tpvec(_mm512_mask_blend_##suffix(_mm512_cmp_##suffix##_mask(mask.val, _mm512_setzero_##zsuf(), _MM_CMPINT_EQ), a.val, b.val)); }

OPENCV_HAL_IMPL_AVX512_SELECT(v_uint8x64,   epi8, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int8x64,    epi8, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_uint16x32, epi16, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int16x32,  epi16, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_uint32x16, epi32, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int32x16,  epi32, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_uint64x8,  epi64, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_int64x8,   epi64, si512)
OPENCV_HAL_IMPL_AVX512_SELECT(v_float32x16,   ps,    ps)
OPENCV_HAL_IMPL_AVX512_SELECT(v_float64x8,    pd,    pd)

/** Comparison **/
#define OPENCV_HAL_IMPL_AVX512_CMP_INT(bin_op, imm8, _Tpvec, sufcmp, sufset, tval) \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)                         \
    { return _Tpvec(_mm512_maskz_set1_##sufset(_mm512_cmp_##sufcmp##_mask(a.val, b.val, imm8), tval)); }

#define OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(_Tpvec, sufcmp, sufset, tval)              \
    OPENCV_HAL_IMPL_AVX512_CMP_INT(v_eq, _MM_CMPINT_EQ,  _Tpvec, sufcmp, sufset, tval)  \
    OPENCV_HAL_IMPL_AVX512_CMP_INT(v_ne, _MM_CMPINT_NE,  _Tpvec, sufcmp, sufset, tval)  \
    OPENCV_HAL_IMPL_AVX512_CMP_INT(v_lt,  _MM_CMPINT_LT,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_INT(v_gt,  _MM_CMPINT_NLE, _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_INT(v_le, _MM_CMPINT_LE,  _Tpvec, sufcmp, sufset, tval)  \
    OPENCV_HAL_IMPL_AVX512_CMP_INT(v_ge, _MM_CMPINT_NLT, _Tpvec, sufcmp, sufset, tval)

OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint8x64,   epu8,  epi8, (char)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int8x64,    epi8,  epi8, (char)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint16x32, epu16, epi16, (short)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int16x32,  epi16, epi16, (short)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint32x16, epu32, epi32, (int)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int32x16,  epi32, epi32, (int)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_uint64x8,  epu64, epi64, (int64)-1)
OPENCV_HAL_IMPL_AVX512_CMP_OP_INT(v_int64x8,   epi64, epi64, (int64)-1)

#define OPENCV_HAL_IMPL_AVX512_CMP_FLT(bin_op, imm8, _Tpvec, sufcmp, sufset, tval) \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)                        \
    { return _Tpvec(_mm512_castsi512_##sufcmp(_mm512_maskz_set1_##sufset(_mm512_cmp_##sufcmp##_mask(a.val, b.val, imm8), tval))); }

#define OPENCV_HAL_IMPL_AVX512_CMP_OP_FLT(_Tpvec, sufcmp, sufset, tval)           \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(v_eq, _CMP_EQ_OQ,  _Tpvec, sufcmp, sufset, tval)  \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(v_ne, _CMP_NEQ_OQ, _Tpvec, sufcmp, sufset, tval)  \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(v_lt,  _CMP_LT_OQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(v_gt,  _CMP_GT_OQ,  _Tpvec, sufcmp, sufset, tval) \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(v_le, _CMP_LE_OQ,  _Tpvec, sufcmp, sufset, tval)  \
    OPENCV_HAL_IMPL_AVX512_CMP_FLT(v_ge, _CMP_GE_OQ,  _Tpvec, sufcmp, sufset, tval)

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
namespace {
    template<bool prec, int imm4, bool part, int imm32>
    struct _v_rotate_right { static inline v_int8x64 eval(const v_int8x64&, const v_int8x64&) { return v_int8x64(); }};
    template<int imm4, int imm32>
    struct _v_rotate_right<true, imm4, false, imm32> { static inline v_int8x64 eval(const v_int8x64& a, const v_int8x64& b)
    {
        return v_int8x64(_mm512_or_si512(_mm512_srli_epi32(_mm512_alignr_epi32(b.val, a.val, imm32    ),    imm4 *8),
                                         _mm512_slli_epi32(_mm512_alignr_epi32(b.val, a.val, imm32 + 1), (4-imm4)*8)));
    }};
    template<int imm4>
    struct _v_rotate_right<true, imm4, false, 15> { static inline v_int8x64 eval(const v_int8x64& a, const v_int8x64& b)
    {
        return v_int8x64(_mm512_or_si512(_mm512_srli_epi32(_mm512_alignr_epi32(b.val, a.val, 15),    imm4 *8),
                                         _mm512_slli_epi32(                                b.val, (4-imm4)*8)));
    }};
    template<int imm4, int imm32>
    struct _v_rotate_right<true, imm4, true, imm32> { static inline v_int8x64 eval(const v_int8x64&, const v_int8x64& b)
    {
        return v_int8x64(_mm512_or_si512(_mm512_srli_epi32(_mm512_alignr_epi32(_mm512_setzero_si512(), b.val, imm32 - 16),    imm4 *8),
                                         _mm512_slli_epi32(_mm512_alignr_epi32(_mm512_setzero_si512(), b.val, imm32 - 15), (4-imm4)*8)));
    }};
    template<int imm4>
    struct _v_rotate_right<true, imm4, true, 31> { static inline v_int8x64 eval(const v_int8x64&, const v_int8x64& b)
    { return v_int8x64(_mm512_srli_epi32(_mm512_alignr_epi32(_mm512_setzero_si512(), b.val, 15), imm4*8)); }};
    template<int imm32>
    struct _v_rotate_right<false, 0, false, imm32> { static inline v_int8x64 eval(const v_int8x64& a, const v_int8x64& b)
    { return v_int8x64(_mm512_alignr_epi32(b.val, a.val, imm32)); }};
    template<>
    struct _v_rotate_right<false, 0, false, 0> { static inline v_int8x64 eval(const v_int8x64& a, const v_int8x64&) { return a; }};
    template<int imm32>
    struct _v_rotate_right<false, 0, true, imm32> { static inline v_int8x64 eval(const v_int8x64&, const v_int8x64& b)
    { return v_int8x64(_mm512_alignr_epi32(_mm512_setzero_si512(), b.val, imm32 - 16)); }};
    template<>
    struct _v_rotate_right<false, 0, true, 16> { static inline v_int8x64 eval(const v_int8x64&, const v_int8x64& b) { return b; }};
    template<>
    struct _v_rotate_right<false, 0, true, 32> { static inline v_int8x64 eval(const v_int8x64&, const v_int8x64&) { return v_int8x64(); }};
}
template<int imm> inline v_int8x64 v_rotate_right(const v_int8x64& a, const v_int8x64& b)
{
    return imm >= 128 ? v_int8x64() :
#if CV_AVX_512VBMI
    v_int8x64(_mm512_permutex2var_epi8(a.val,
    _v512_set_epu8(0x3f + imm, 0x3e + imm, 0x3d + imm, 0x3c + imm, 0x3b + imm, 0x3a + imm, 0x39 + imm, 0x38 + imm,
                   0x37 + imm, 0x36 + imm, 0x35 + imm, 0x34 + imm, 0x33 + imm, 0x32 + imm, 0x31 + imm, 0x30 + imm,
                   0x2f + imm, 0x2e + imm, 0x2d + imm, 0x2c + imm, 0x2b + imm, 0x2a + imm, 0x29 + imm, 0x28 + imm,
                   0x27 + imm, 0x26 + imm, 0x25 + imm, 0x24 + imm, 0x23 + imm, 0x22 + imm, 0x21 + imm, 0x20 + imm,
                   0x1f + imm, 0x1e + imm, 0x1d + imm, 0x1c + imm, 0x1b + imm, 0x1a + imm, 0x19 + imm, 0x18 + imm,
                   0x17 + imm, 0x16 + imm, 0x15 + imm, 0x14 + imm, 0x13 + imm, 0x12 + imm, 0x11 + imm, 0x10 + imm,
                   0x0f + imm, 0x0e + imm, 0x0d + imm, 0x0c + imm, 0x0b + imm, 0x0a + imm, 0x09 + imm, 0x08 + imm,
                   0x07 + imm, 0x06 + imm, 0x05 + imm, 0x04 + imm, 0x03 + imm, 0x02 + imm, 0x01 + imm, 0x00 + imm), b.val));
#else
    _v_rotate_right<imm%4!=0, imm%4, (imm/4 > 15), imm/4>::eval(a, b);
#endif
}
template<int imm>
inline v_int8x64 v_rotate_left(const v_int8x64& a, const v_int8x64& b)
{
    if (imm == 0) return a;
    if (imm == 64) return b;
    if (imm >= 128) return v_int8x64();
#if CV_AVX_512VBMI
    return v_int8x64(_mm512_permutex2var_epi8(b.val,
           _v512_set_epi8(0x7f - imm,0x7e - imm,0x7d - imm,0x7c - imm,0x7b - imm,0x7a - imm,0x79 - imm,0x78 - imm,
                          0x77 - imm,0x76 - imm,0x75 - imm,0x74 - imm,0x73 - imm,0x72 - imm,0x71 - imm,0x70 - imm,
                          0x6f - imm,0x6e - imm,0x6d - imm,0x6c - imm,0x6b - imm,0x6a - imm,0x69 - imm,0x68 - imm,
                          0x67 - imm,0x66 - imm,0x65 - imm,0x64 - imm,0x63 - imm,0x62 - imm,0x61 - imm,0x60 - imm,
                          0x5f - imm,0x5e - imm,0x5d - imm,0x5c - imm,0x5b - imm,0x5a - imm,0x59 - imm,0x58 - imm,
                          0x57 - imm,0x56 - imm,0x55 - imm,0x54 - imm,0x53 - imm,0x52 - imm,0x51 - imm,0x50 - imm,
                          0x4f - imm,0x4e - imm,0x4d - imm,0x4c - imm,0x4b - imm,0x4a - imm,0x49 - imm,0x48 - imm,
                          0x47 - imm,0x46 - imm,0x45 - imm,0x44 - imm,0x43 - imm,0x42 - imm,0x41 - imm,0x40 - imm), a.val));
#else
    return imm < 64 ? v_rotate_right<64 - imm>(b, a) : v_rotate_right<128 - imm>(v512_setzero_s8(), b);
#endif
}
template<int imm>
inline v_int8x64 v_rotate_right(const v_int8x64& a)
{
    if (imm == 0) return a;
    if (imm >= 64) return v_int8x64();
#if CV_AVX_512VBMI
    return v_int8x64(_mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFF >> imm,
           _v512_set_epu8(0x3f + imm,0x3e + imm,0x3d + imm,0x3c + imm,0x3b + imm,0x3a + imm,0x39 + imm,0x38 + imm,
                          0x37 + imm,0x36 + imm,0x35 + imm,0x34 + imm,0x33 + imm,0x32 + imm,0x31 + imm,0x30 + imm,
                          0x2f + imm,0x2e + imm,0x2d + imm,0x2c + imm,0x2b + imm,0x2a + imm,0x29 + imm,0x28 + imm,
                          0x27 + imm,0x26 + imm,0x25 + imm,0x24 + imm,0x23 + imm,0x22 + imm,0x21 + imm,0x20 + imm,
                          0x1f + imm,0x1e + imm,0x1d + imm,0x1c + imm,0x1b + imm,0x1a + imm,0x19 + imm,0x18 + imm,
                          0x17 + imm,0x16 + imm,0x15 + imm,0x14 + imm,0x13 + imm,0x12 + imm,0x11 + imm,0x10 + imm,
                          0x0f + imm,0x0e + imm,0x0d + imm,0x0c + imm,0x0b + imm,0x0a + imm,0x09 + imm,0x08 + imm,
                          0x07 + imm,0x06 + imm,0x05 + imm,0x04 + imm,0x03 + imm,0x02 + imm,0x01 + imm,0x00 + imm), a.val));
#else
    return v_rotate_right<imm>(a, v512_setzero_s8());
#endif
}
template<int imm>
inline v_int8x64 v_rotate_left(const v_int8x64& a)
{
    if (imm == 0) return a;
    if (imm >= 64) return v_int8x64();
#if CV_AVX_512VBMI
    return v_int8x64(_mm512_maskz_permutexvar_epi8(0xFFFFFFFFFFFFFFFF << imm,
           _v512_set_epi8(0x3f - imm,0x3e - imm,0x3d - imm,0x3c - imm,0x3b - imm,0x3a - imm,0x39 - imm,0x38 - imm,
                          0x37 - imm,0x36 - imm,0x35 - imm,0x34 - imm,0x33 - imm,0x32 - imm,0x31 - imm,0x30 - imm,
                          0x2f - imm,0x2e - imm,0x2d - imm,0x2c - imm,0x2b - imm,0x2a - imm,0x29 - imm,0x28 - imm,
                          0x27 - imm,0x26 - imm,0x25 - imm,0x24 - imm,0x23 - imm,0x22 - imm,0x21 - imm,0x20 - imm,
                          0x1f - imm,0x1e - imm,0x1d - imm,0x1c - imm,0x1b - imm,0x1a - imm,0x19 - imm,0x18 - imm,
                          0x17 - imm,0x16 - imm,0x15 - imm,0x14 - imm,0x13 - imm,0x12 - imm,0x11 - imm,0x10 - imm,
                          0x0f - imm,0x0e - imm,0x0d - imm,0x0c - imm,0x0b - imm,0x0a - imm,0x09 - imm,0x08 - imm,
                          0x07 - imm,0x06 - imm,0x05 - imm,0x04 - imm,0x03 - imm,0x02 - imm,0x01 - imm,0x00 - imm), a.val));
#else
    return v_rotate_right<64 - imm>(v512_setzero_s8(), a);
#endif
}

#define OPENCV_HAL_IMPL_AVX512_ROTATE_PM(_Tpvec, suffix)                                                                                   \
template<int imm> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b)                                                            \
{ return v_reinterpret_as_##suffix(v_rotate_left<imm * sizeof(_Tpvec::lane_type)>(v_reinterpret_as_s8(a), v_reinterpret_as_s8(b))); }      \
template<int imm> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b)                                                           \
{ return v_reinterpret_as_##suffix(v_rotate_right<imm * sizeof(_Tpvec::lane_type)>(v_reinterpret_as_s8(a), v_reinterpret_as_s8(b))); }     \
template<int imm> inline _Tpvec v_rotate_left(const _Tpvec& a)                                                                             \
{ return v_reinterpret_as_##suffix(v_rotate_left<imm * sizeof(_Tpvec::lane_type)>(v_reinterpret_as_s8(a))); }                              \
template<int imm> inline _Tpvec v_rotate_right(const _Tpvec& a)                                                                            \
{ return v_reinterpret_as_##suffix(v_rotate_right<imm * sizeof(_Tpvec::lane_type)>(v_reinterpret_as_s8(a))); }

#define OPENCV_HAL_IMPL_AVX512_ROTATE_EC(_Tpvec, suffix)                                                                                   \
template<int imm>                                                                                                                          \
inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b)                                                                              \
{                                                                                                                                          \
    enum { SHIFT2 = (_Tpvec::nlanes - imm) };                                                                                              \
    enum { MASK = ((1 << _Tpvec::nlanes) - 1) };                                                                                           \
    if (imm == 0) return a;                                                                                                                \
    if (imm == _Tpvec::nlanes) return b;                                                                                                   \
    if (imm >= 2*_Tpvec::nlanes) return _Tpvec::zero();                                                                                    \
    return _Tpvec(_mm512_mask_expand_##suffix(_mm512_maskz_compress_##suffix((MASK << SHIFT2)&MASK, b.val), (MASK << (imm))&MASK, a.val)); \
}                                                                                                                                          \
template<int imm>                                                                                                                          \
inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b)                                                                             \
{                                                                                                                                          \
    enum { SHIFT2 = (_Tpvec::nlanes - imm) };                                                                                              \
    enum { MASK = ((1 << _Tpvec::nlanes) - 1) };                                                                                           \
    if (imm == 0) return a;                                                                                                                \
    if (imm == _Tpvec::nlanes) return b;                                                                                                   \
    if (imm >= 2*_Tpvec::nlanes) return _Tpvec::zero();                                                                                    \
    return _Tpvec(_mm512_mask_expand_##suffix(_mm512_maskz_compress_##suffix((MASK << (imm))&MASK, a.val), (MASK << SHIFT2)&MASK, b.val)); \
}                                                                                                                                          \
template<int imm>                                                                                                                          \
inline _Tpvec v_rotate_left(const _Tpvec& a)                                                                                               \
{                                                                                                                                          \
    if (imm == 0) return a;                                                                                                                \
    if (imm >= _Tpvec::nlanes) return _Tpvec::zero();                                                                                      \
    return _Tpvec(_mm512_maskz_expand_##suffix((1 << _Tpvec::nlanes) - (1 << (imm)), a.val));                                              \
}                                                                                                                                          \
template<int imm>                                                                                                                          \
inline _Tpvec v_rotate_right(const _Tpvec& a)                                                                                              \
{                                                                                                                                          \
    if (imm == 0) return a;                                                                                                                \
    if (imm >= _Tpvec::nlanes) return _Tpvec::zero();                                                                                      \
    return _Tpvec(_mm512_maskz_compress_##suffix((1 << _Tpvec::nlanes) - (1 << (imm)), a.val));                                            \
}

OPENCV_HAL_IMPL_AVX512_ROTATE_PM(v_uint8x64,   u8)
OPENCV_HAL_IMPL_AVX512_ROTATE_PM(v_uint16x32,  u16)
OPENCV_HAL_IMPL_AVX512_ROTATE_PM(v_int16x32,   s16)
OPENCV_HAL_IMPL_AVX512_ROTATE_EC(v_uint32x16,  epi32)
OPENCV_HAL_IMPL_AVX512_ROTATE_EC(v_int32x16,   epi32)
OPENCV_HAL_IMPL_AVX512_ROTATE_EC(v_uint64x8,   epi64)
OPENCV_HAL_IMPL_AVX512_ROTATE_EC(v_int64x8,    epi64)
OPENCV_HAL_IMPL_AVX512_ROTATE_EC(v_float32x16, ps)
OPENCV_HAL_IMPL_AVX512_ROTATE_EC(v_float64x8,  pd)

/** Reverse **/
inline v_uint8x64 v_reverse(const v_uint8x64 &a)
{
#if CV_AVX_512VBMI
    static const __m512i perm = _mm512_set_epi32(
            0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
            0x10111213, 0x14151617, 0x18191a1b, 0x1c1d1e1f,
            0x20212223, 0x24252627, 0x28292a2b, 0x2c2d2e2f,
            0x30313233, 0x34353637, 0x38393a3b, 0x3c3d3e3f);
    return v_uint8x64(_mm512_permutexvar_epi8(perm, a.val));
#else
    static const __m512i shuf = _mm512_set_epi32(
            0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
            0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
            0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f,
            0x00010203, 0x04050607, 0x08090a0b, 0x0c0d0e0f);
    static const __m512i perm = _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6);
    __m512i vec = _mm512_shuffle_epi8(a.val, shuf);
    return v_uint8x64(_mm512_permutexvar_epi64(perm, vec));
#endif
}

inline v_int8x64 v_reverse(const v_int8x64 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x32 v_reverse(const v_uint16x32 &a)
{
#if CV_AVX_512VBMI
    static const __m512i perm = _mm512_set_epi32(
            0x00000001, 0x00020003, 0x00040005, 0x00060007,
            0x00080009, 0x000a000b, 0x000c000d, 0x000e000f,
            0x00100011, 0x00120013, 0x00140015, 0x00160017,
            0x00180019, 0x001a001b, 0x001c001d, 0x001e001f);
    return v_uint16x32(_mm512_permutexvar_epi16(perm, a.val));
#else
    static const __m512i shuf = _mm512_set_epi32(
            0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e,
            0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e,
            0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e,
            0x01000302, 0x05040706, 0x09080b0a, 0x0d0c0f0e);
    static const __m512i perm = _mm512_set_epi64(1, 0, 3, 2, 5, 4, 7, 6);
    __m512i vec = _mm512_shuffle_epi8(a.val, shuf);
    return v_uint16x32(_mm512_permutexvar_epi64(perm, vec));
#endif
}

inline v_int16x32 v_reverse(const v_int16x32 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x16 v_reverse(const v_uint32x16 &a)
{
    static const __m512i perm = _mm512_set_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,14, 15);
    return v_uint32x16(_mm512_permutexvar_epi32(perm, a.val));
}

inline v_int32x16 v_reverse(const v_int32x16 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x16 v_reverse(const v_float32x16 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x8 v_reverse(const v_uint64x8 &a)
{
    static const __m512i perm = _mm512_set_epi64(0, 1, 2, 3, 4, 5, 6, 7);
    return v_uint64x8(_mm512_permutexvar_epi64(perm, a.val));
}

inline v_int64x8 v_reverse(const v_int64x8 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x8 v_reverse(const v_float64x8 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }

////////// Reduce /////////

/** Reduce **/
#define OPENCV_HAL_IMPL_AVX512_REDUCE_ADD64(a, b) a + b
#define OPENCV_HAL_IMPL_AVX512_REDUCE_8(sctype, func, _Tpvec, ifunc, scop)                                          \
    inline sctype v_reduce_##func(const _Tpvec& a)                                                                  \
    { __m256i half = _mm256_##ifunc(_v512_extract_low(a.val), _v512_extract_high(a.val));                           \
      sctype CV_DECL_ALIGNED(64) idx[2];                                                                            \
      _mm_store_si128((__m128i*)idx, _mm_##ifunc(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1))); \
      return scop(idx[0], idx[1]); }
OPENCV_HAL_IMPL_AVX512_REDUCE_8(uint64, min, v_uint64x8, min_epu64, min)
OPENCV_HAL_IMPL_AVX512_REDUCE_8(uint64, max, v_uint64x8, max_epu64, max)
OPENCV_HAL_IMPL_AVX512_REDUCE_8(uint64, sum, v_uint64x8, add_epi64, OPENCV_HAL_IMPL_AVX512_REDUCE_ADD64)
OPENCV_HAL_IMPL_AVX512_REDUCE_8(int64,  min, v_int64x8,  min_epi64, min)
OPENCV_HAL_IMPL_AVX512_REDUCE_8(int64,  max, v_int64x8,  max_epi64, max)
OPENCV_HAL_IMPL_AVX512_REDUCE_8(int64,  sum, v_int64x8,  add_epi64, OPENCV_HAL_IMPL_AVX512_REDUCE_ADD64)

#define OPENCV_HAL_IMPL_AVX512_REDUCE_8F(func, ifunc, scop)                                         \
    inline double v_reduce_##func(const v_float64x8& a)                                             \
    { __m256d half = _mm256_##ifunc(_v512_extract_low(a.val), _v512_extract_high(a.val));           \
      double CV_DECL_ALIGNED(64) idx[2];                                                            \
      _mm_store_pd(idx, _mm_##ifunc(_mm256_castpd256_pd128(half), _mm256_extractf128_pd(half, 1))); \
      return scop(idx[0], idx[1]); }
OPENCV_HAL_IMPL_AVX512_REDUCE_8F(min, min_pd, min)
OPENCV_HAL_IMPL_AVX512_REDUCE_8F(max, max_pd, max)
OPENCV_HAL_IMPL_AVX512_REDUCE_8F(sum, add_pd, OPENCV_HAL_IMPL_AVX512_REDUCE_ADD64)

#define OPENCV_HAL_IMPL_AVX512_REDUCE_16(sctype, func, _Tpvec, ifunc)                                 \
    inline sctype v_reduce_##func(const _Tpvec& a)                                                    \
    { __m256i half = _mm256_##ifunc(_v512_extract_low(a.val), _v512_extract_high(a.val));             \
      __m128i quarter = _mm_##ifunc(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1)); \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 8));                                     \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 4));                                     \
      return (sctype)_mm_cvtsi128_si32(quarter); }
OPENCV_HAL_IMPL_AVX512_REDUCE_16(uint, min, v_uint32x16, min_epu32)
OPENCV_HAL_IMPL_AVX512_REDUCE_16(uint, max, v_uint32x16, max_epu32)
OPENCV_HAL_IMPL_AVX512_REDUCE_16(int,  min, v_int32x16,  min_epi32)
OPENCV_HAL_IMPL_AVX512_REDUCE_16(int,  max, v_int32x16,  max_epi32)

#define OPENCV_HAL_IMPL_AVX512_REDUCE_16F(func, ifunc)                                            \
    inline float v_reduce_##func(const v_float32x16& a)                                           \
    { __m256 half = _mm256_##ifunc(_v512_extract_low(a.val), _v512_extract_high(a.val));          \
      __m128 quarter = _mm_##ifunc(_mm256_castps256_ps128(half), _mm256_extractf128_ps(half, 1)); \
      quarter = _mm_##ifunc(quarter, _mm_permute_ps(quarter, _MM_SHUFFLE(0, 0, 3, 2)));           \
      quarter = _mm_##ifunc(quarter, _mm_permute_ps(quarter, _MM_SHUFFLE(0, 0, 0, 1)));           \
      return _mm_cvtss_f32(quarter); }
OPENCV_HAL_IMPL_AVX512_REDUCE_16F(min, min_ps)
OPENCV_HAL_IMPL_AVX512_REDUCE_16F(max, max_ps)

inline float v_reduce_sum(const v_float32x16& a)
{
    __m256 half = _mm256_add_ps(_v512_extract_low(a.val), _v512_extract_high(a.val));
    __m128 quarter = _mm_add_ps(_mm256_castps256_ps128(half), _mm256_extractf128_ps(half, 1));
    quarter = _mm_hadd_ps(quarter, quarter);
    return _mm_cvtss_f32(_mm_hadd_ps(quarter, quarter));
}
inline int v_reduce_sum(const v_int32x16& a)
{
    __m256i half = _mm256_add_epi32(_v512_extract_low(a.val), _v512_extract_high(a.val));
    __m128i quarter = _mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1));
    quarter = _mm_hadd_epi32(quarter, quarter);
    return _mm_cvtsi128_si32(_mm_hadd_epi32(quarter, quarter));
}
inline uint v_reduce_sum(const v_uint32x16& a)
{ return (uint)v_reduce_sum(v_reinterpret_as_s32(a)); }

#define OPENCV_HAL_IMPL_AVX512_REDUCE_32(sctype, func, _Tpvec, ifunc)                                 \
    inline sctype v_reduce_##func(const _Tpvec& a)                                                    \
    { __m256i half = _mm256_##ifunc(_v512_extract_low(a.val), _v512_extract_high(a.val));             \
      __m128i quarter = _mm_##ifunc(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1)); \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 8));                                     \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 4));                                     \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 2));                                     \
      return (sctype)_mm_cvtsi128_si32(quarter); }
OPENCV_HAL_IMPL_AVX512_REDUCE_32(ushort, min, v_uint16x32, min_epu16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32(ushort, max, v_uint16x32, max_epu16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32(short,  min, v_int16x32,  min_epi16)
OPENCV_HAL_IMPL_AVX512_REDUCE_32(short,  max, v_int16x32,  max_epi16)

inline int v_reduce_sum(const v_int16x32& a)
{ return v_reduce_sum(v_add(v_expand_low(a), v_expand_high(a))); }
inline uint v_reduce_sum(const v_uint16x32& a)
{ return v_reduce_sum(v_add(v_expand_low(a), v_expand_high(a))); }

#define OPENCV_HAL_IMPL_AVX512_REDUCE_64(sctype, func, _Tpvec, ifunc)                                 \
    inline sctype v_reduce_##func(const _Tpvec& a)                                                    \
    { __m256i half = _mm256_##ifunc(_v512_extract_low(a.val), _v512_extract_high(a.val));             \
      __m128i quarter = _mm_##ifunc(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1)); \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 8));                                     \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 4));                                     \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 2));                                     \
      quarter = _mm_##ifunc(quarter, _mm_srli_si128(quarter, 1));                                     \
      return (sctype)_mm_cvtsi128_si32(quarter); }
OPENCV_HAL_IMPL_AVX512_REDUCE_64(uchar, min, v_uint8x64, min_epu8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64(uchar, max, v_uint8x64, max_epu8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64(schar, min, v_int8x64,  min_epi8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64(schar, max, v_int8x64,  max_epi8)

#define OPENCV_HAL_IMPL_AVX512_REDUCE_64_SUM(sctype, _Tpvec, suffix)                                    \
    inline sctype v_reduce_sum(const _Tpvec& a)                                                         \
    {   __m512i a16 = _mm512_add_epi16(_mm512_cvt##suffix##_epi16(_v512_extract_low(a.val)),            \
                                       _mm512_cvt##suffix##_epi16(_v512_extract_high(a.val)));          \
        a16 = _mm512_cvtepi16_epi32(_mm256_add_epi16(_v512_extract_low(a16), _v512_extract_high(a16))); \
        __m256i a8 = _mm256_add_epi32(_v512_extract_low(a16), _v512_extract_high(a16));                 \
        __m128i a4 = _mm_add_epi32(_mm256_castsi256_si128(a8), _mm256_extracti128_si256(a8, 1));        \
        a4 = _mm_hadd_epi32(a4, a4);                                                                    \
        return (sctype)_mm_cvtsi128_si32(_mm_hadd_epi32(a4, a4)); }
OPENCV_HAL_IMPL_AVX512_REDUCE_64_SUM(uint, v_uint8x64, epu8)
OPENCV_HAL_IMPL_AVX512_REDUCE_64_SUM(int,  v_int8x64,  epi8)

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
    __m512i val = _mm512_sad_epu8(a.val, b.val);
    __m256i half = _mm256_add_epi32(_v512_extract_low(val), _v512_extract_high(val));
    __m128i quarter = _mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1));
    return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(quarter, _mm_unpackhi_epi64(quarter, quarter)));
}
inline unsigned v_reduce_sad(const v_int8x64& a, const v_int8x64& b)
{
    __m512i val = _mm512_set1_epi8(-128);
    val = _mm512_sad_epu8(_mm512_add_epi8(a.val, val), _mm512_add_epi8(b.val, val));
    __m256i half = _mm256_add_epi32(_v512_extract_low(val), _v512_extract_high(val));
    __m128i quarter = _mm_add_epi32(_mm256_castsi256_si128(half), _mm256_extracti128_si256(half, 1));
    return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(quarter, _mm_unpackhi_epi64(quarter, quarter)));
}
inline unsigned v_reduce_sad(const v_uint16x32& a, const v_uint16x32& b)
{ return v_reduce_sum(v_add_wrap(v_sub(a, b), v_sub(b, a))); }
inline unsigned v_reduce_sad(const v_int16x32& a, const v_int16x32& b)
{ return v_reduce_sum(v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b)))); }
inline unsigned v_reduce_sad(const v_uint32x16& a, const v_uint32x16& b)
{ return v_reduce_sum(v_sub(v_max(a, b), v_min(a, b))); }
inline unsigned v_reduce_sad(const v_int32x16& a, const v_int32x16& b)
{ return v_reduce_sum(v_reinterpret_as_u32(v_sub(v_max(a, b), v_min(a, b)))); }
inline float v_reduce_sad(const v_float32x16& a, const v_float32x16& b)
{ return v_reduce_sum(v_and(v_sub(a, b), v_float32x16(_mm512_castsi512_ps(_mm512_set1_epi32(0x7fffffff))))); }
inline double v_reduce_sad(const v_float64x8& a, const v_float64x8& b)
{ return v_reduce_sum(v_and(v_sub(a, b), v_float64x8(_mm512_castsi512_pd(_mm512_set1_epi64(0x7fffffffffffffff))))); }

/** Popcount **/
inline v_uint8x64 v_popcount(const v_int8x64& a)
{
#if CV_AVX_512BITALG
    return v_uint8x64(_mm512_popcnt_epi8(a.val));
#elif CV_AVX_512VBMI
    __m512i _popcnt_table0 = _v512_set_epu8(7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
                                            5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
                                            5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1,
                                            4, 3, 3, 2, 3, 2, 2, 1, 3, 2, 2, 1, 2, 1, 1, 0);
    __m512i _popcnt_table1 = _v512_set_epu8(7, 6, 6, 5, 6, 5, 5, 4, 6, 5, 5, 4, 5, 4, 4, 3,
                                            6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
                                            6, 5, 5, 4, 5, 4, 4, 3, 5, 4, 4, 3, 4, 3, 3, 2,
                                            5, 4, 4, 3, 4, 3, 3, 2, 4, 3, 3, 2, 3, 2, 2, 1);
    return v_uint8x64(_mm512_sub_epi8(_mm512_permutex2var_epi8(_popcnt_table0, a.val, _popcnt_table1), _mm512_movm_epi8(_mm512_movepi8_mask(a.val))));
#else
    __m512i _popcnt_table = _mm512_set4_epi32(0x04030302, 0x03020201, 0x03020201, 0x02010100);
    __m512i _popcnt_mask = _mm512_set1_epi8(0x0F);

    return v_uint8x64(_mm512_add_epi8(_mm512_shuffle_epi8(_popcnt_table, _mm512_and_si512(                  a.val,     _popcnt_mask)),
                                      _mm512_shuffle_epi8(_popcnt_table, _mm512_and_si512(_mm512_srli_epi16(a.val, 4), _popcnt_mask))));
#endif
}
inline v_uint16x32 v_popcount(const v_int16x32& a)
{
#if CV_AVX_512BITALG
    return v_uint16x32(_mm512_popcnt_epi16(a.val));
#elif CV_AVX_512VPOPCNTDQ
    __m512i zero = _mm512_setzero_si512();
    return v_uint16x32(_mm512_packs_epi32(_mm512_popcnt_epi32(_mm512_unpacklo_epi16(a.val, zero)),
                                          _mm512_popcnt_epi32(_mm512_unpackhi_epi16(a.val, zero))));
#else
    v_uint8x64 p = v_popcount(v_reinterpret_as_s8(a));
    p = v_add(p, v_rotate_right<1>(p));
    return v_and(v_reinterpret_as_u16(p), v512_setall_u16(0x00ff));
#endif
}
inline v_uint32x16 v_popcount(const v_int32x16& a)
{
#if CV_AVX_512VPOPCNTDQ
    return v_uint32x16(_mm512_popcnt_epi32(a.val));
#else
    v_uint8x64 p = v_popcount(v_reinterpret_as_s8(a));
    p = v_add(p, v_rotate_right<1>(p));
    p = v_add(p, v_rotate_right<2>(p));
    return v_and(v_reinterpret_as_u32(p), v512_setall_u32(0x000000ff));
#endif
}
inline v_uint64x8 v_popcount(const v_int64x8& a)
{
#if CV_AVX_512VPOPCNTDQ
    return v_uint64x8(_mm512_popcnt_epi64(a.val));
#else
    return v_uint64x8(_mm512_sad_epu8(v_popcount(v_reinterpret_as_s8(a)).val, _mm512_setzero_si512()));
#endif
}


inline v_uint8x64  v_popcount(const v_uint8x64&  a) { return v_popcount(v_reinterpret_as_s8 (a)); }
inline v_uint16x32 v_popcount(const v_uint16x32& a) { return v_popcount(v_reinterpret_as_s16(a)); }
inline v_uint32x16 v_popcount(const v_uint32x16& a) { return v_popcount(v_reinterpret_as_s32(a)); }
inline v_uint64x8  v_popcount(const v_uint64x8&  a) { return v_popcount(v_reinterpret_as_s64(a)); }


////////// Other math /////////

/** Some frequent operations **/
#if CV_FMA3
#define OPENCV_HAL_IMPL_AVX512_MULADD(_Tpvec, suffix)                         \
    inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)    \
    { return _Tpvec(_mm512_fmadd_##suffix(a.val, b.val, c.val)); }            \
    inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
    { return _Tpvec(_mm512_fmadd_##suffix(a.val, b.val, c.val)); }
#else
#define OPENCV_HAL_IMPL_AVX512_MULADD(_Tpvec, suffix)                                 \
    inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)            \
    { return _Tpvec(_mm512_add_##suffix(_mm512_mul_##suffix(a.val, b.val), c.val)); } \
    inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)         \
    { return _Tpvec(_mm512_add_##suffix(_mm512_mul_##suffix(a.val, b.val), c.val)); }
#endif

#define OPENCV_HAL_IMPL_AVX512_MISC(_Tpvec, suffix)                           \
    inline _Tpvec v_sqrt(const _Tpvec& x)                                     \
    { return _Tpvec(_mm512_sqrt_##suffix(x.val)); }                           \
    inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b)           \
    { return v_fma(a, a, v_mul(b, b)); }                                      \
    inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b)               \
    { return v_sqrt(v_fma(a, a, v_mul(b, b))); }

OPENCV_HAL_IMPL_AVX512_MULADD(v_float32x16, ps)
OPENCV_HAL_IMPL_AVX512_MULADD(v_float64x8,  pd)
OPENCV_HAL_IMPL_AVX512_MISC(v_float32x16, ps)
OPENCV_HAL_IMPL_AVX512_MISC(v_float64x8,  pd)

inline v_int32x16 v_fma(const v_int32x16& a, const v_int32x16& b, const v_int32x16& c)
{ return v_add(v_mul(a, b), c); }
inline v_int32x16 v_muladd(const v_int32x16& a, const v_int32x16& b, const v_int32x16& c)
{ return v_fma(a, b, c); }

inline v_float32x16 v_invsqrt(const v_float32x16& x)
{
#if CV_AVX_512ER
    return v_float32x16(_mm512_rsqrt28_ps(x.val));
#else
    v_float32x16 half = v_mul(x, v512_setall_f32(0.5));
    v_float32x16 t  = v_float32x16(_mm512_rsqrt14_ps(x.val));
    t = v_mul(t, v_sub(v512_setall_f32(1.5), v_mul(v_mul(t, t), half)));
    return t;
#endif
}

inline v_float64x8 v_invsqrt(const v_float64x8& x)
{
#if CV_AVX_512ER
    return v_float64x8(_mm512_rsqrt28_pd(x.val));
#else
    return v_div(v512_setall_f64(1.), v_sqrt(x));
//    v_float64x8 half = x * v512_setall_f64(0.5);
//    v_float64x8 t = v_float64x8(_mm512_rsqrt14_pd(x.val));
//    t *= v512_setall_f64(1.5) - ((t * t) * half);
//    t *= v512_setall_f64(1.5) - ((t * t) * half);
//    return t;
#endif
}

/** Absolute values **/
#define OPENCV_HAL_IMPL_AVX512_ABS(_Tpvec, _Tpuvec, suffix) \
    inline _Tpuvec v_abs(const _Tpvec& x)                   \
    { return _Tpuvec(_mm512_abs_##suffix(x.val)); }

OPENCV_HAL_IMPL_AVX512_ABS(v_int8x64,    v_uint8x64,    epi8)
OPENCV_HAL_IMPL_AVX512_ABS(v_int16x32,   v_uint16x32,  epi16)
OPENCV_HAL_IMPL_AVX512_ABS(v_int32x16,   v_uint32x16,  epi32)
OPENCV_HAL_IMPL_AVX512_ABS(v_int64x8,    v_uint64x8,   epi64)

inline v_float32x16 v_abs(const v_float32x16& x)
{
#ifdef _mm512_abs_pd
    return v_float32x16(_mm512_abs_ps(x.val));
#else
    return v_float32x16(_mm512_castsi512_ps(_mm512_and_si512(_mm512_castps_si512(x.val),
                        _v512_set_epu64(0x7FFFFFFF7FFFFFFF, 0x7FFFFFFF7FFFFFFF, 0x7FFFFFFF7FFFFFFF, 0x7FFFFFFF7FFFFFFF,
                                        0x7FFFFFFF7FFFFFFF, 0x7FFFFFFF7FFFFFFF, 0x7FFFFFFF7FFFFFFF, 0x7FFFFFFF7FFFFFFF))));
#endif
}

inline v_float64x8 v_abs(const v_float64x8& x)
{
#ifdef _mm512_abs_pd
    #if defined __GNUC__ && (__GNUC__ < 7 || (__GNUC__ == 7 && __GNUC_MINOR__ <= 3) || (__GNUC__ == 8 && __GNUC_MINOR__ <= 2))
        // Workaround for https://gcc.gnu.org/bugzilla/show_bug.cgi?id=87476
        return v_float64x8(_mm512_abs_pd(_mm512_castpd_ps(x.val)));
    #else
        return v_float64x8(_mm512_abs_pd(x.val));
    #endif
#else
    return v_float64x8(_mm512_castsi512_pd(_mm512_and_si512(_mm512_castpd_si512(x.val),
                       _v512_set_epu64(0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF,
                                       0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF, 0x7FFFFFFFFFFFFFFF))));
#endif
}

/** Absolute difference **/
inline v_uint8x64 v_absdiff(const v_uint8x64& a, const v_uint8x64& b)
{ return v_add_wrap(v_sub(a, b), v_sub(b, a)); }
inline v_uint16x32 v_absdiff(const v_uint16x32& a, const v_uint16x32& b)
{ return v_add_wrap(v_sub(a, b), v_sub(b, a)); }
inline v_uint32x16 v_absdiff(const v_uint32x16& a, const v_uint32x16& b)
{ return v_sub(v_max(a, b), v_min(a, b)); }

inline v_uint8x64 v_absdiff(const v_int8x64& a, const v_int8x64& b)
{
    v_int8x64 d = v_sub_wrap(a, b);
    v_int8x64 m = v_lt(a, b);
    return v_reinterpret_as_u8(v_sub_wrap(v_xor(d, m), m));
}

inline v_uint16x32 v_absdiff(const v_int16x32& a, const v_int16x32& b)
{ return v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b))); }

inline v_uint32x16 v_absdiff(const v_int32x16& a, const v_int32x16& b)
{
    v_int32x16 d = v_sub(a, b);
    v_int32x16 m = v_lt(a, b);
    return v_reinterpret_as_u32(v_sub(v_xor(d, m), m));
}

inline v_float32x16 v_absdiff(const v_float32x16& a, const v_float32x16& b)
{ return v_abs(v_sub(a, b)); }

inline v_float64x8 v_absdiff(const v_float64x8& a, const v_float64x8& b)
{ return v_abs(v_sub(a, b)); }

/** Saturating absolute difference **/
inline v_int8x64 v_absdiffs(const v_int8x64& a, const v_int8x64& b)
{
    v_int8x64 d = v_sub(a, b);
    v_int8x64 m = v_lt(a, b);
    return v_sub(v_xor(d, m), m);
}
inline v_int16x32 v_absdiffs(const v_int16x32& a, const v_int16x32& b)
{ return v_sub(v_max(a, b), v_min(a, b)); }

////////// Conversions /////////

/** Rounding **/
inline v_int32x16 v_round(const v_float32x16& a)
{ return v_int32x16(_mm512_cvtps_epi32(a.val)); }

inline v_int32x16 v_round(const v_float64x8& a)
{ return v_int32x16(_mm512_castsi256_si512(_mm512_cvtpd_epi32(a.val))); }

inline v_int32x16 v_round(const v_float64x8& a, const v_float64x8& b)
{ return v_int32x16(_v512_combine(_mm512_cvtpd_epi32(a.val), _mm512_cvtpd_epi32(b.val))); }

inline v_int32x16 v_trunc(const v_float32x16& a)
{ return v_int32x16(_mm512_cvttps_epi32(a.val)); }

inline v_int32x16 v_trunc(const v_float64x8& a)
{ return v_int32x16(_mm512_castsi256_si512(_mm512_cvttpd_epi32(a.val))); }

#if CVT_ROUND_MODES_IMPLEMENTED
inline v_int32x16 v_floor(const v_float32x16& a)
{ return v_int32x16(_mm512_cvt_roundps_epi32(a.val, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC)); }

inline v_int32x16 v_floor(const v_float64x8& a)
{ return v_int32x16(_mm512_castsi256_si512(_mm512_cvt_roundpd_epi32(a.val, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC))); }

inline v_int32x16 v_ceil(const v_float32x16& a)
{ return v_int32x16(_mm512_cvt_roundps_epi32(a.val, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC)); }

inline v_int32x16 v_ceil(const v_float64x8& a)
{ return v_int32x16(_mm512_castsi256_si512(_mm512_cvt_roundpd_epi32(a.val, _MM_FROUND_TO_POS_INF | _MM_FROUND_NO_EXC))); }
#else
inline v_int32x16 v_floor(const v_float32x16& a)
{ return v_int32x16(_mm512_cvtps_epi32(_mm512_roundscale_ps(a.val, 1))); }

inline v_int32x16 v_floor(const v_float64x8& a)
{ return v_int32x16(_mm512_castsi256_si512(_mm512_cvtpd_epi32(_mm512_roundscale_pd(a.val, 1)))); }

inline v_int32x16 v_ceil(const v_float32x16& a)
{ return v_int32x16(_mm512_cvtps_epi32(_mm512_roundscale_ps(a.val, 2))); }

inline v_int32x16 v_ceil(const v_float64x8& a)
{ return v_int32x16(_mm512_castsi256_si512(_mm512_cvtpd_epi32(_mm512_roundscale_pd(a.val, 2)))); }
#endif

/** To float **/
inline v_float32x16 v_cvt_f32(const v_int32x16& a)
{ return v_float32x16(_mm512_cvtepi32_ps(a.val)); }

inline v_float32x16 v_cvt_f32(const v_float64x8& a)
{ return v_float32x16(_mm512_cvtpd_pslo(a.val)); }

inline v_float32x16 v_cvt_f32(const v_float64x8& a, const v_float64x8& b)
{ return v_float32x16(_v512_combine(_mm512_cvtpd_ps(a.val), _mm512_cvtpd_ps(b.val))); }

inline v_float64x8 v_cvt_f64(const v_int32x16& a)
{ return v_float64x8(_mm512_cvtepi32_pd(_v512_extract_low(a.val))); }

inline v_float64x8 v_cvt_f64_high(const v_int32x16& a)
{ return v_float64x8(_mm512_cvtepi32_pd(_v512_extract_high(a.val))); }

inline v_float64x8 v_cvt_f64(const v_float32x16& a)
{ return v_float64x8(_mm512_cvtps_pd(_v512_extract_low(a.val))); }

inline v_float64x8 v_cvt_f64_high(const v_float32x16& a)
{ return v_float64x8(_mm512_cvtps_pd(_v512_extract_high(a.val))); }

// from (Mysticial and wim) https://stackoverflow.com/q/41144668
inline v_float64x8 v_cvt_f64(const v_int64x8& v)
{
#if CV_AVX_512DQ
    return v_float64x8(_mm512_cvtepi64_pd(v.val));
#else
    // constants encoded as floating-point
    __m512i magic_i_lo   = _mm512_set1_epi64(0x4330000000000000); // 2^52
    __m512i magic_i_hi32 = _mm512_set1_epi64(0x4530000080000000); // 2^84 + 2^63
    __m512i magic_i_all  = _mm512_set1_epi64(0x4530000080100000); // 2^84 + 2^63 + 2^52
    __m512d magic_d_all  = _mm512_castsi512_pd(magic_i_all);

    // Blend the 32 lowest significant bits of v with magic_int_lo
    __m512i v_lo         = _mm512_mask_blend_epi32(0x5555, magic_i_lo, v.val);
    // Extract the 32 most significant bits of v
    __m512i v_hi         = _mm512_srli_epi64(v.val, 32);
    // Flip the msb of v_hi and blend with 0x45300000
            v_hi         = _mm512_xor_si512(v_hi, magic_i_hi32);
    // Compute in double precision
    __m512d v_hi_dbl     = _mm512_sub_pd(_mm512_castsi512_pd(v_hi), magic_d_all);
    // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition
    __m512d result       = _mm512_add_pd(v_hi_dbl, _mm512_castsi512_pd(v_lo));
    return v_float64x8(result);
#endif
}

////////////// Lookup table access ////////////////////

inline v_int8x64 v512_lut(const schar* tab, const int* idx)
{
    __m128i p0 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx    ), (const int *)tab, 1));
    __m128i p1 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx + 1), (const int *)tab, 1));
    __m128i p2 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx + 2), (const int *)tab, 1));
    __m128i p3 = _mm512_cvtepi32_epi8(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx + 3), (const int *)tab, 1));
    return v_int8x64(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(p0), p1, 1), p2, 2), p3, 3));
}
inline v_int8x64 v512_lut_pairs(const schar* tab, const int* idx)
{
    __m256i p0 = _mm512_cvtepi32_epi16(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx    ), (const int *)tab, 1));
    __m256i p1 = _mm512_cvtepi32_epi16(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx + 1), (const int *)tab, 1));
    return v_int8x64(_v512_combine(p0, p1));
}
inline v_int8x64 v512_lut_quads(const schar* tab, const int* idx)
{
    return v_int8x64(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx), (const int *)tab, 1));
}
inline v_uint8x64 v512_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut((const schar *)tab, idx)); }
inline v_uint8x64 v512_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut_pairs((const schar *)tab, idx)); }
inline v_uint8x64 v512_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v512_lut_quads((const schar *)tab, idx)); }

inline v_int16x32 v512_lut(const short* tab, const int* idx)
{
    __m256i p0 = _mm512_cvtepi32_epi16(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx    ), (const int *)tab, 2));
    __m256i p1 = _mm512_cvtepi32_epi16(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx + 1), (const int *)tab, 2));
    return v_int16x32(_v512_combine(p0, p1));
}
inline v_int16x32 v512_lut_pairs(const short* tab, const int* idx)
{
    return v_int16x32(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx), (const int *)tab, 2));
}
inline v_int16x32 v512_lut_quads(const short* tab, const int* idx)
{
#if defined(__GNUC__)
    return v_int16x32(_mm512_i32gather_epi64(_mm256_loadu_si256((const __m256i*)idx), (const long long int*)tab, 2));
#else
    return v_int16x32(_mm512_i32gather_epi64(_mm256_loadu_si256((const __m256i*)idx), (const int64*)tab, 2));
#endif
}
inline v_uint16x32 v512_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut((const short *)tab, idx)); }
inline v_uint16x32 v512_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut_pairs((const short *)tab, idx)); }
inline v_uint16x32 v512_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v512_lut_quads((const short *)tab, idx)); }

inline v_int32x16 v512_lut(const int* tab, const int* idx)
{
    return v_int32x16(_mm512_i32gather_epi32(_mm512_loadu_si512((const __m512i*)idx), tab, 4));
}
inline v_int32x16 v512_lut_pairs(const int* tab, const int* idx)
{
#if defined(__GNUC__)
    return v_int32x16(_mm512_i32gather_epi64(_mm256_loadu_si256((const __m256i*)idx), (const long long int*)tab, 4));
#else
    return v_int32x16(_mm512_i32gather_epi64(_mm256_loadu_si256((const __m256i*)idx), (const int64*)tab, 4));
#endif
}
inline v_int32x16 v512_lut_quads(const int* tab, const int* idx)
{
    return v_int32x16(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(
                          _mm_loadu_si128((const __m128i*)(tab + idx[0]))),
                          _mm_loadu_si128((const __m128i*)(tab + idx[1])), 1),
                          _mm_loadu_si128((const __m128i*)(tab + idx[2])), 2),
                          _mm_loadu_si128((const __m128i*)(tab + idx[3])), 3));
}
inline v_uint32x16 v512_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut((const int *)tab, idx)); }
inline v_uint32x16 v512_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut_pairs((const int *)tab, idx)); }
inline v_uint32x16 v512_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v512_lut_quads((const int *)tab, idx)); }

inline v_int64x8 v512_lut(const int64* tab, const int* idx)
{
#if defined(__GNUC__)
    return v_int64x8(_mm512_i32gather_epi64(_mm256_loadu_si256((const __m256i*)idx), (const long long int*)tab, 8));
#else
    return v_int64x8(_mm512_i32gather_epi64(_mm256_loadu_si256((const __m256i*)idx), tab , 8));
#endif
}
inline v_int64x8 v512_lut_pairs(const int64* tab, const int* idx)
{
    return v_int64x8(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_inserti32x4(_mm512_castsi128_si512(
                         _mm_loadu_si128((const __m128i*)(tab + idx[0]))),
                         _mm_loadu_si128((const __m128i*)(tab + idx[1])), 1),
                         _mm_loadu_si128((const __m128i*)(tab + idx[2])), 2),
                         _mm_loadu_si128((const __m128i*)(tab + idx[3])), 3));
}
inline v_uint64x8 v512_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v512_lut((const int64 *)tab, idx)); }
inline v_uint64x8 v512_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v512_lut_pairs((const int64 *)tab, idx)); }

inline v_float32x16 v512_lut(const float* tab, const int* idx)
{
    return v_float32x16(_mm512_i32gather_ps(_mm512_loadu_si512((const __m512i*)idx), tab, 4));
}
inline v_float32x16 v512_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v512_lut_pairs((const int *)tab, idx)); }
inline v_float32x16 v512_lut_quads(const float* tab, const int* idx) { return v_reinterpret_as_f32(v512_lut_quads((const int *)tab, idx)); }

inline v_float64x8 v512_lut(const double* tab, const int* idx)
{
    return v_float64x8(_mm512_i32gather_pd(_mm256_loadu_si256((const __m256i*)idx), tab, 8));
}
inline v_float64x8 v512_lut_pairs(const double* tab, const int* idx)
{
        return v_float64x8(_mm512_insertf64x2(_mm512_insertf64x2(_mm512_insertf64x2(_mm512_castpd128_pd512(
                               _mm_loadu_pd(tab + idx[0])),
                               _mm_loadu_pd(tab + idx[1]), 1),
                               _mm_loadu_pd(tab + idx[2]), 2),
                               _mm_loadu_pd(tab + idx[3]), 3));
}

inline v_int32x16 v_lut(const int* tab, const v_int32x16& idxvec)
{
    return v_int32x16(_mm512_i32gather_epi32(idxvec.val, tab, 4));
}

inline v_uint32x16 v_lut(const unsigned* tab, const v_int32x16& idxvec)
{
    return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
}

inline v_float32x16 v_lut(const float* tab, const v_int32x16& idxvec)
{
    return v_float32x16(_mm512_i32gather_ps(idxvec.val, tab, 4));
}

inline v_float64x8 v_lut(const double* tab, const v_int32x16& idxvec)
{
    return v_float64x8(_mm512_i32gather_pd(_v512_extract_low(idxvec.val), tab, 8));
}

inline void v_lut_deinterleave(const float* tab, const v_int32x16& idxvec, v_float32x16& x, v_float32x16& y)
{
    x.val = _mm512_i32gather_ps(idxvec.val, tab, 4);
    y.val = _mm512_i32gather_ps(idxvec.val, &tab[1], 4);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x16& idxvec, v_float64x8& x, v_float64x8& y)
{
    x.val = _mm512_i32gather_pd(_v512_extract_low(idxvec.val), tab, 8);
    y.val = _mm512_i32gather_pd(_v512_extract_low(idxvec.val), &tab[1], 8);
}

inline v_int8x64 v_interleave_pairs(const v_int8x64& vec)
{
    return v_int8x64(_mm512_shuffle_epi8(vec.val, _mm512_set4_epi32(0x0f0d0e0c, 0x0b090a08, 0x07050604, 0x03010200)));
}
inline v_uint8x64 v_interleave_pairs(const v_uint8x64& vec) { return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x64 v_interleave_quads(const v_int8x64& vec)
{
    return v_int8x64(_mm512_shuffle_epi8(vec.val, _mm512_set4_epi32(0x0f0b0e0a, 0x0d090c08, 0x07030602, 0x05010400)));
}
inline v_uint8x64 v_interleave_quads(const v_uint8x64& vec) { return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x32 v_interleave_pairs(const v_int16x32& vec)
{
    return v_int16x32(_mm512_shuffle_epi8(vec.val, _mm512_set4_epi32(0x0f0e0b0a, 0x0d0c0908, 0x07060302, 0x05040100)));
}
inline v_uint16x32 v_interleave_pairs(const v_uint16x32& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x32 v_interleave_quads(const v_int16x32& vec)
{
    return v_int16x32(_mm512_shuffle_epi8(vec.val, _mm512_set4_epi32(0x0f0e0706, 0x0d0c0504, 0x0b0a0302, 0x09080100)));
}
inline v_uint16x32 v_interleave_quads(const v_uint16x32& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x16 v_interleave_pairs(const v_int32x16& vec)
{
    return v_int32x16(_mm512_shuffle_epi32(vec.val, _MM_PERM_ACBD));
}
inline v_uint32x16 v_interleave_pairs(const v_uint32x16& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x16 v_interleave_pairs(const v_float32x16& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x64 v_pack_triplets(const v_int8x64& vec)
{
    return v_int8x64(_mm512_permutexvar_epi32(_v512_set_epu64(0x0000000f0000000f, 0x0000000f0000000f, 0x0000000e0000000d, 0x0000000c0000000a,
                                                              0x0000000900000008, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000),
                                              _mm512_shuffle_epi8(vec.val, _mm512_set4_epi32(0xffffff0f, 0x0e0d0c0a, 0x09080605, 0x04020100))));
}
inline v_uint8x64 v_pack_triplets(const v_uint8x64& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x32 v_pack_triplets(const v_int16x32& vec)
{
    return v_int16x32(_mm512_permutexvar_epi16(_v512_set_epu64(0x001f001f001f001f, 0x001f001f001f001f, 0x001e001d001c001a, 0x0019001800160015,
                                                               0x0014001200110010, 0x000e000d000c000a, 0x0009000800060005, 0x0004000200010000), vec.val));
}
inline v_uint16x32 v_pack_triplets(const v_uint16x32& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x16 v_pack_triplets(const v_int32x16& vec)
{
    return v_int32x16(_mm512_permutexvar_epi32(_v512_set_epu64(0x0000000f0000000f, 0x0000000f0000000f, 0x0000000e0000000d, 0x0000000c0000000a,
                                                               0x0000000900000008, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000), vec.val));
}
inline v_uint32x16 v_pack_triplets(const v_uint32x16& vec) { return v_reinterpret_as_u32(v_pack_triplets(v_reinterpret_as_s32(vec))); }
inline v_float32x16 v_pack_triplets(const v_float32x16& vec)
{
    return v_float32x16(_mm512_permutexvar_ps(_v512_set_epu64(0x0000000f0000000f, 0x0000000f0000000f, 0x0000000e0000000d, 0x0000000c0000000a,
                                                              0x0000000900000008, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000), vec.val));
}

////////// Matrix operations /////////

//////// Dot Product ////////

// 16 >> 32
inline v_int32x16 v_dotprod(const v_int16x32& a, const v_int16x32& b)
{ return v_int32x16(_mm512_madd_epi16(a.val, b.val)); }
inline v_int32x16 v_dotprod(const v_int16x32& a, const v_int16x32& b, const v_int32x16& c)
{ return v_add(v_dotprod(a, b), c); }

// 32 >> 64
inline v_int64x8 v_dotprod(const v_int32x16& a, const v_int32x16& b)
{
    __m512i even = _mm512_mul_epi32(a.val, b.val);
    __m512i odd = _mm512_mul_epi32(_mm512_srli_epi64(a.val, 32), _mm512_srli_epi64(b.val, 32));
    return v_int64x8(_mm512_add_epi64(even, odd));
}
inline v_int64x8 v_dotprod(const v_int32x16& a, const v_int32x16& b, const v_int64x8& c)
{ return v_add(v_dotprod(a, b), c); }

// 8 >> 32
inline v_uint32x16 v_dotprod_expand(const v_uint8x64& a, const v_uint8x64& b)
{
    __m512i even_a = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, a.val, _mm512_setzero_si512());
    __m512i odd_a  = _mm512_srli_epi16(a.val, 8);

    __m512i even_b = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, b.val, _mm512_setzero_si512());
    __m512i odd_b  = _mm512_srli_epi16(b.val, 8);

    __m512i prod0  = _mm512_madd_epi16(even_a, even_b);
    __m512i prod1  = _mm512_madd_epi16(odd_a, odd_b);
    return v_uint32x16(_mm512_add_epi32(prod0, prod1));
}
inline v_uint32x16 v_dotprod_expand(const v_uint8x64& a, const v_uint8x64& b, const v_uint32x16& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int32x16 v_dotprod_expand(const v_int8x64& a, const v_int8x64& b)
{
    __m512i even_a = _mm512_srai_epi16(_mm512_bslli_epi128(a.val, 1), 8);
    __m512i odd_a  = _mm512_srai_epi16(a.val, 8);

    __m512i even_b = _mm512_srai_epi16(_mm512_bslli_epi128(b.val, 1), 8);
    __m512i odd_b  = _mm512_srai_epi16(b.val, 8);

    __m512i prod0  = _mm512_madd_epi16(even_a, even_b);
    __m512i prod1  = _mm512_madd_epi16(odd_a, odd_b);
    return v_int32x16(_mm512_add_epi32(prod0, prod1));
}
inline v_int32x16 v_dotprod_expand(const v_int8x64& a, const v_int8x64& b, const v_int32x16& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 16 >> 64
inline v_uint64x8 v_dotprod_expand(const v_uint16x32& a, const v_uint16x32& b)
{
    __m512i mullo = _mm512_mullo_epi16(a.val, b.val);
    __m512i mulhi = _mm512_mulhi_epu16(a.val, b.val);
    __m512i mul0  = _mm512_unpacklo_epi16(mullo, mulhi);
    __m512i mul1  = _mm512_unpackhi_epi16(mullo, mulhi);

    __m512i p02   = _mm512_mask_blend_epi32(0xAAAA, mul0, _mm512_setzero_si512());
    __m512i p13   = _mm512_srli_epi64(mul0, 32);
    __m512i p46   = _mm512_mask_blend_epi32(0xAAAA, mul1, _mm512_setzero_si512());
    __m512i p57   = _mm512_srli_epi64(mul1, 32);

    __m512i p15_  = _mm512_add_epi64(p02, p13);
    __m512i p9d_  = _mm512_add_epi64(p46, p57);

    return v_uint64x8(_mm512_add_epi64(
        _mm512_unpacklo_epi64(p15_, p9d_),
        _mm512_unpackhi_epi64(p15_, p9d_)
    ));
}
inline v_uint64x8 v_dotprod_expand(const v_uint16x32& a, const v_uint16x32& b, const v_uint64x8& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int64x8 v_dotprod_expand(const v_int16x32& a, const v_int16x32& b)
{
    __m512i prod = _mm512_madd_epi16(a.val, b.val);
    __m512i even = _mm512_srai_epi64(_mm512_bslli_epi128(prod, 4), 32);
    __m512i odd  = _mm512_srai_epi64(prod, 32);
    return v_int64x8(_mm512_add_epi64(even, odd));
}
inline v_int64x8 v_dotprod_expand(const v_int16x32& a, const v_int16x32& b, const v_int64x8& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 32 >> 64f
inline v_float64x8 v_dotprod_expand(const v_int32x16& a, const v_int32x16& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x8 v_dotprod_expand(const v_int32x16& a, const v_int32x16& b, const v_float64x8& c)
{ return v_add(v_dotprod_expand(a, b), c); }

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x16 v_dotprod_fast(const v_int16x32& a, const v_int16x32& b)
{ return v_dotprod(a, b); }
inline v_int32x16 v_dotprod_fast(const v_int16x32& a, const v_int16x32& b, const v_int32x16& c)
{ return v_dotprod(a, b, c); }

// 32 >> 64
inline v_int64x8 v_dotprod_fast(const v_int32x16& a, const v_int32x16& b)
{ return v_dotprod(a, b); }
inline v_int64x8 v_dotprod_fast(const v_int32x16& a, const v_int32x16& b, const v_int64x8& c)
{ return v_dotprod(a, b, c); }

// 8 >> 32
inline v_uint32x16 v_dotprod_expand_fast(const v_uint8x64& a, const v_uint8x64& b)
{ return v_dotprod_expand(a, b); }
inline v_uint32x16 v_dotprod_expand_fast(const v_uint8x64& a, const v_uint8x64& b, const v_uint32x16& c)
{ return v_dotprod_expand(a, b, c); }

inline v_int32x16 v_dotprod_expand_fast(const v_int8x64& a, const v_int8x64& b)
{ return v_dotprod_expand(a, b); }
inline v_int32x16 v_dotprod_expand_fast(const v_int8x64& a, const v_int8x64& b, const v_int32x16& c)
{ return v_dotprod_expand(a, b, c); }

// 16 >> 64
inline v_uint64x8 v_dotprod_expand_fast(const v_uint16x32& a, const v_uint16x32& b)
{
    __m512i mullo = _mm512_mullo_epi16(a.val, b.val);
    __m512i mulhi = _mm512_mulhi_epu16(a.val, b.val);
    __m512i mul0  = _mm512_unpacklo_epi16(mullo, mulhi);
    __m512i mul1  = _mm512_unpackhi_epi16(mullo, mulhi);

    __m512i p02   = _mm512_mask_blend_epi32(0xAAAA, mul0, _mm512_setzero_si512());
    __m512i p13   = _mm512_srli_epi64(mul0, 32);
    __m512i p46   = _mm512_mask_blend_epi32(0xAAAA, mul1, _mm512_setzero_si512());
    __m512i p57   = _mm512_srli_epi64(mul1, 32);

    __m512i p15_  = _mm512_add_epi64(p02, p13);
    __m512i p9d_  = _mm512_add_epi64(p46, p57);
    return v_uint64x8(_mm512_add_epi64(p15_, p9d_));
}
inline v_uint64x8 v_dotprod_expand_fast(const v_uint16x32& a, const v_uint16x32& b, const v_uint64x8& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

inline v_int64x8 v_dotprod_expand_fast(const v_int16x32& a, const v_int16x32& b)
{ return v_dotprod_expand(a, b); }
inline v_int64x8 v_dotprod_expand_fast(const v_int16x32& a, const v_int16x32& b, const v_int64x8& c)
{ return v_dotprod_expand(a, b, c); }

// 32 >> 64f
inline v_float64x8 v_dotprod_expand_fast(const v_int32x16& a, const v_int32x16& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x8 v_dotprod_expand_fast(const v_int32x16& a, const v_int32x16& b, const v_float64x8& c)
{ return v_add(v_dotprod_expand(a, b), c); }


#define OPENCV_HAL_AVX512_SPLAT2_PS(a, im) \
    v_float32x16(_mm512_permute_ps(a.val, _MM_SHUFFLE(im, im, im, im)))

inline v_float32x16 v_matmul(const v_float32x16& v,
                             const v_float32x16& m0, const v_float32x16& m1,
                             const v_float32x16& m2, const v_float32x16& m3)
{
    v_float32x16 v04 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 0);
    v_float32x16 v15 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 1);
    v_float32x16 v26 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 2);
    v_float32x16 v37 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 3);
    return v_fma(v04, m0, v_fma(v15, m1, v_fma(v26, m2, v_mul(v37, m3))));
}

inline v_float32x16 v_matmuladd(const v_float32x16& v,
                                const v_float32x16& m0, const v_float32x16& m1,
                                const v_float32x16& m2, const v_float32x16& a)
{
    v_float32x16 v04 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 0);
    v_float32x16 v15 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 1);
    v_float32x16 v26 = OPENCV_HAL_AVX512_SPLAT2_PS(v, 2);
    return v_fma(v04, m0, v_fma(v15, m1, v_fma(v26, m2, a)));
}

#define OPENCV_HAL_IMPL_AVX512_TRANSPOSE4x4(_Tpvec, suffix, cast_from, cast_to) \
    inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1,              \
                               const _Tpvec& a2, const _Tpvec& a3,              \
                               _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3)  \
    {                                                                           \
        __m512i t0 = cast_from(_mm512_unpacklo_##suffix(a0.val, a1.val));       \
        __m512i t1 = cast_from(_mm512_unpacklo_##suffix(a2.val, a3.val));       \
        __m512i t2 = cast_from(_mm512_unpackhi_##suffix(a0.val, a1.val));       \
        __m512i t3 = cast_from(_mm512_unpackhi_##suffix(a2.val, a3.val));       \
        b0.val = cast_to(_mm512_unpacklo_epi64(t0, t1));                        \
        b1.val = cast_to(_mm512_unpackhi_epi64(t0, t1));                        \
        b2.val = cast_to(_mm512_unpacklo_epi64(t2, t3));                        \
        b3.val = cast_to(_mm512_unpackhi_epi64(t2, t3));                        \
    }

OPENCV_HAL_IMPL_AVX512_TRANSPOSE4x4(v_uint32x16,  epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_AVX512_TRANSPOSE4x4(v_int32x16,   epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_AVX512_TRANSPOSE4x4(v_float32x16, ps, _mm512_castps_si512, _mm512_castsi512_ps)

//////////////// Value reordering ///////////////

/* Expand */
#define OPENCV_HAL_IMPL_AVX512_EXPAND(_Tpvec, _Tpwvec, _Tp, intrin) \
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
        __m256i a = _mm256_loadu_si256((const __m256i*)ptr);        \
        return _Tpwvec(intrin(a));                                  \
    }

OPENCV_HAL_IMPL_AVX512_EXPAND(v_uint8x64,  v_uint16x32, uchar,    _mm512_cvtepu8_epi16)
OPENCV_HAL_IMPL_AVX512_EXPAND(v_int8x64,   v_int16x32,  schar,    _mm512_cvtepi8_epi16)
OPENCV_HAL_IMPL_AVX512_EXPAND(v_uint16x32, v_uint32x16, ushort,   _mm512_cvtepu16_epi32)
OPENCV_HAL_IMPL_AVX512_EXPAND(v_int16x32,  v_int32x16,  short,    _mm512_cvtepi16_epi32)
OPENCV_HAL_IMPL_AVX512_EXPAND(v_uint32x16, v_uint64x8,  unsigned, _mm512_cvtepu32_epi64)
OPENCV_HAL_IMPL_AVX512_EXPAND(v_int32x16,  v_int64x8,   int,      _mm512_cvtepi32_epi64)

#define OPENCV_HAL_IMPL_AVX512_EXPAND_Q(_Tpvec, _Tp, intrin) \
    inline _Tpvec v512_load_expand_q(const _Tp* ptr)         \
    {                                                        \
        __m128i a = _mm_loadu_si128((const __m128i*)ptr);    \
        return _Tpvec(intrin(a));                            \
    }

OPENCV_HAL_IMPL_AVX512_EXPAND_Q(v_uint32x16, uchar, _mm512_cvtepu8_epi32)
OPENCV_HAL_IMPL_AVX512_EXPAND_Q(v_int32x16,  schar, _mm512_cvtepi8_epi32)

/* pack */
// 16
inline v_int8x64 v_pack(const v_int16x32& a, const v_int16x32& b)
{ return v_int8x64(_mm512_permutexvar_epi64(_v512_set_epu64(7, 5, 3, 1, 6, 4, 2, 0), _mm512_packs_epi16(a.val, b.val))); }

inline v_uint8x64 v_pack(const v_uint16x32& a, const v_uint16x32& b)
{
    const __m512i t = _mm512_set1_epi16(255);
    return v_uint8x64(_v512_combine(_mm512_cvtepi16_epi8(_mm512_min_epu16(a.val, t)), _mm512_cvtepi16_epi8(_mm512_min_epu16(b.val, t))));
}

inline v_uint8x64 v_pack_u(const v_int16x32& a, const v_int16x32& b)
{
    return v_uint8x64(_mm512_permutexvar_epi64(_v512_set_epu64(7, 5, 3, 1, 6, 4, 2, 0), _mm512_packus_epi16(a.val, b.val)));
}

inline void v_pack_store(schar* ptr, const v_int16x32& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(uchar* ptr, const v_uint16x32& a)
{
    const __m512i m = _mm512_set1_epi16(255);
    _mm256_storeu_si256((__m256i*)ptr, _mm512_cvtepi16_epi8(_mm512_min_epu16(a.val, m)));
}

inline void v_pack_u_store(uchar* ptr, const v_int16x32& a)
{ v_store_low(ptr, v_pack_u(a, a)); }

template<int n> inline
v_uint8x64 v_rshr_pack(const v_uint16x32& a, const v_uint16x32& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    v_uint16x32 delta = v512_setall_u16((short)(1 << (n-1)));
    return v_pack_u(v_reinterpret_as_s16(v_shr(v_add(a, delta), n)),
                    v_reinterpret_as_s16(v_shr(v_add(b, delta), n)));
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x32& a)
{
    v_uint16x32 delta = v512_setall_u16((short)(1 << (n-1)));
    v_pack_u_store(ptr, v_reinterpret_as_s16(v_shr(v_add(a, delta), n)));
}

template<int n> inline
v_uint8x64 v_rshr_pack_u(const v_int16x32& a, const v_int16x32& b)
{
    v_int16x32 delta = v512_setall_s16((short)(1 << (n-1)));
    return v_pack_u(v_shr(v_add(a, delta), n), v_shr(v_add(b, delta), n));
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x32& a)
{
    v_int16x32 delta = v512_setall_s16((short)(1 << (n-1)));
    v_pack_u_store(ptr, v_shr(v_add(a, delta), n));
}

template<int n> inline
v_int8x64 v_rshr_pack(const v_int16x32& a, const v_int16x32& b)
{
    v_int16x32 delta = v512_setall_s16((short)(1 << (n-1)));
    return v_pack(v_shr(v_add(a, delta), n), v_shr(v_add(b, delta), n));
}

template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x32& a)
{
    v_int16x32 delta = v512_setall_s16((short)(1 << (n-1)));
    v_pack_store(ptr, v_shr(v_add(a, delta), n));
}

// 32
inline v_int16x32 v_pack(const v_int32x16& a, const v_int32x16& b)
{ return v_int16x32(_mm512_permutexvar_epi64(_v512_set_epu64(7, 5, 3, 1, 6, 4, 2, 0), _mm512_packs_epi32(a.val, b.val))); }

inline v_uint16x32 v_pack(const v_uint32x16& a, const v_uint32x16& b)
{
    const __m512i m = _mm512_set1_epi32(65535);
    return v_uint16x32(_v512_combine(_mm512_cvtepi32_epi16(_mm512_min_epu32(a.val, m)), _mm512_cvtepi32_epi16(_mm512_min_epu32(b.val, m))));
}

inline v_uint16x32 v_pack_u(const v_int32x16& a, const v_int32x16& b)
{ return v_uint16x32(_mm512_permutexvar_epi64(_v512_set_epu64(7, 5, 3, 1, 6, 4, 2, 0), _mm512_packus_epi32(a.val, b.val))); }

inline void v_pack_store(short* ptr, const v_int32x16& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(ushort* ptr, const v_uint32x16& a)
{
    const __m512i m = _mm512_set1_epi32(65535);
    _mm256_storeu_si256((__m256i*)ptr, _mm512_cvtepi32_epi16(_mm512_min_epu32(a.val, m)));
}

inline void v_pack_u_store(ushort* ptr, const v_int32x16& a)
{ v_store_low(ptr, v_pack_u(a, a)); }


template<int n> inline
v_uint16x32 v_rshr_pack(const v_uint32x16& a, const v_uint32x16& b)
{
    v_uint32x16 delta = v512_setall_u32(1 << (n-1));
    return v_pack_u(v_reinterpret_as_s32(v_shr(v_add(a, delta), n)),
                    v_reinterpret_as_s32(v_shr(v_add(b, delta), n)));
}

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x16& a)
{
    v_uint32x16 delta = v512_setall_u32(1 << (n-1));
    v_pack_u_store(ptr, v_reinterpret_as_s32(v_shr(v_add(a, delta), n)));
}

template<int n> inline
v_uint16x32 v_rshr_pack_u(const v_int32x16& a, const v_int32x16& b)
{
    v_int32x16 delta = v512_setall_s32(1 << (n-1));
    return v_pack_u(v_shr(v_add(a, delta), n), v_shr(v_add(b, delta), n));
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x16& a)
{
    v_int32x16 delta = v512_setall_s32(1 << (n-1));
    v_pack_u_store(ptr, v_shr(v_add(a, delta), n));
}

template<int n> inline
v_int16x32 v_rshr_pack(const v_int32x16& a, const v_int32x16& b)
{
    v_int32x16 delta = v512_setall_s32(1 << (n-1));
    return v_pack(v_shr(v_add(a, delta), n), v_shr(v_add(b, delta), n));
}

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x16& a)
{
    v_int32x16 delta = v512_setall_s32(1 << (n-1));
    v_pack_store(ptr, v_shr(v_add(a, delta), n));
}

// 64
// Non-saturating pack
inline v_uint32x16 v_pack(const v_uint64x8& a, const v_uint64x8& b)
{ return v_uint32x16(_v512_combine(_mm512_cvtepi64_epi32(a.val), _mm512_cvtepi64_epi32(b.val))); }

inline v_int32x16 v_pack(const v_int64x8& a, const v_int64x8& b)
{ return v_reinterpret_as_s32(v_pack(v_reinterpret_as_u64(a), v_reinterpret_as_u64(b))); }

inline void v_pack_store(unsigned* ptr, const v_uint64x8& a)
{ _mm256_storeu_si256((__m256i*)ptr, _mm512_cvtepi64_epi32(a.val)); }

inline void v_pack_store(int* ptr, const v_int64x8& b)
{ v_pack_store((unsigned*)ptr, v_reinterpret_as_u64(b)); }

template<int n> inline
v_uint32x16 v_rshr_pack(const v_uint64x8& a, const v_uint64x8& b)
{
    v_uint64x8 delta = v512_setall_u64((uint64)1 << (n-1));
    return v_pack(v_shr(v_add(a, delta), n), v_shr(v_add(b, delta), n));
}

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x8& a)
{
    v_uint64x8 delta = v512_setall_u64((uint64)1 << (n-1));
    v_pack_store(ptr, v_shr(v_add(a, delta), n));
}

template<int n> inline
v_int32x16 v_rshr_pack(const v_int64x8& a, const v_int64x8& b)
{
    v_int64x8 delta = v512_setall_s64((int64)1 << (n-1));
    return v_pack(v_shr(v_add(a, delta), n), v_shr(v_add(b, delta), n));
}

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x8& a)
{
    v_int64x8 delta = v512_setall_s64((int64)1 << (n-1));
    v_pack_store(ptr, v_shr(v_add(a, delta), n));
}

// pack boolean
inline v_uint8x64 v_pack_b(const v_uint16x32& a, const v_uint16x32& b)
{ return v_uint8x64(_mm512_permutexvar_epi64(_v512_set_epu64(7, 5, 3, 1, 6, 4, 2, 0), _mm512_packs_epi16(a.val, b.val))); }

inline v_uint8x64 v_pack_b(const v_uint32x16& a, const v_uint32x16& b,
                           const v_uint32x16& c, const v_uint32x16& d)
{
    __m512i ab = _mm512_packs_epi32(a.val, b.val);
    __m512i cd = _mm512_packs_epi32(c.val, d.val);

    return v_uint8x64(_mm512_permutexvar_epi32(_v512_set_epu32(15, 11, 7, 3, 14, 10, 6, 2, 13, 9, 5, 1, 12, 8, 4, 0), _mm512_packs_epi16(ab, cd)));
}

inline v_uint8x64 v_pack_b(const v_uint64x8& a, const v_uint64x8& b, const v_uint64x8& c,
                           const v_uint64x8& d, const v_uint64x8& e, const v_uint64x8& f,
                           const v_uint64x8& g, const v_uint64x8& h)
{
    __m512i ab = _mm512_packs_epi32(a.val, b.val);
    __m512i cd = _mm512_packs_epi32(c.val, d.val);
    __m512i ef = _mm512_packs_epi32(e.val, f.val);
    __m512i gh = _mm512_packs_epi32(g.val, h.val);

    __m512i abcd = _mm512_packs_epi32(ab, cd);
    __m512i efgh = _mm512_packs_epi32(ef, gh);

    return v_uint8x64(_mm512_permutexvar_epi16(_v512_set_epu16(31, 23, 15, 7, 30, 22, 14, 6, 29, 21, 13, 5, 28, 20, 12, 4,
                                                               27, 19, 11, 3, 26, 18, 10, 2, 25, 17,  9, 1, 24, 16,  8, 0), _mm512_packs_epi16(abcd, efgh)));
}

/* Recombine */
// its up there with load and store operations

/* Extract */
#define OPENCV_HAL_IMPL_AVX512_EXTRACT(_Tpvec)                \
    template<int s>                                           \
    inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b) \
    { return v_rotate_right<s>(a, b); }

OPENCV_HAL_IMPL_AVX512_EXTRACT(v_uint8x64)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_int8x64)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_uint16x32)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_int16x32)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_uint32x16)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_int32x16)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_uint64x8)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_int64x8)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_float32x16)
OPENCV_HAL_IMPL_AVX512_EXTRACT(v_float64x8)

#define OPENCV_HAL_IMPL_AVX512_EXTRACT_N(_Tpvec, _Tp) \
template<int i> inline _Tp v_extract_n(_Tpvec v) { return v_rotate_right<i>(v).get0(); }

OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_uint8x64, uchar)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_int8x64, schar)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_uint16x32, ushort)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_int16x32, short)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_uint32x16, uint)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_int32x16, int)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_uint64x8, uint64)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_int64x8, int64)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_float32x16, float)
OPENCV_HAL_IMPL_AVX512_EXTRACT_N(v_float64x8, double)

template<int i>
inline v_uint32x16 v_broadcast_element(v_uint32x16 a)
{
    static const __m512i perm = _mm512_set1_epi32((char)i);
    return v_uint32x16(_mm512_permutexvar_epi32(perm, a.val));
}

template<int i>
inline v_int32x16 v_broadcast_element(const v_int32x16 &a)
{ return v_reinterpret_as_s32(v_broadcast_element<i>(v_reinterpret_as_u32(a))); }

template<int i>
inline v_float32x16 v_broadcast_element(const v_float32x16 &a)
{ return v_reinterpret_as_f32(v_broadcast_element<i>(v_reinterpret_as_u32(a))); }


///////////////////// load deinterleave /////////////////////////////

inline void v_load_deinterleave( const uchar* ptr, v_uint8x64& a, v_uint8x64& b )
{
    __m512i ab0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i ab1 = _mm512_loadu_si512((const __m512i*)(ptr + 64));
#if CV_AVX_512VBMI
    __m512i mask0 = _v512_set_epu8(126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96,
                                    94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74,  72,  70,  68, 66, 64,
                                    62,  60,  58,  56,  54,  52,  50,  48,  46,  44,  42,  40,  38,  36, 34, 32,
                                    30,  28,  26,  24,  22,  20,  18,  16,  14,  12,  10,   8,   6,   4,  2,  0);
    __m512i mask1 = _v512_set_epu8(127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97,
                                    95,  93,  91,  89,  87,  85,  83,  81,  79,  77,  75,  73,  71,  69, 67, 65,
                                    63,  61,  59,  57,  55,  53,  51,  49,  47,  45,  43,  41,  39,  37, 35, 33,
                                    31,  29,  27,  25,  23,  21,  19,  17,  15,  13,  11,   9,   7,   5,  3,  1);
    a = v_uint8x64(_mm512_permutex2var_epi8(ab0, mask0, ab1));
    b = v_uint8x64(_mm512_permutex2var_epi8(ab0, mask1, ab1));
#else
    __m512i mask0 = _mm512_set4_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
    __m512i a0b0 = _mm512_shuffle_epi8(ab0, mask0);
    __m512i a1b1 = _mm512_shuffle_epi8(ab1, mask0);
    __m512i mask1 = _v512_set_epu64(14, 12, 10, 8, 6, 4, 2, 0);
    __m512i mask2 = _v512_set_epu64(15, 13, 11, 9, 7, 5, 3, 1);
    a = v_uint8x64(_mm512_permutex2var_epi64(a0b0, mask1, a1b1));
    b = v_uint8x64(_mm512_permutex2var_epi64(a0b0, mask2, a1b1));
#endif
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x32& a, v_uint16x32& b )
{
    __m512i ab0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i ab1 = _mm512_loadu_si512((const __m512i*)(ptr + 32));
    __m512i mask0 = _v512_set_epu16(62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
                                    30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,  8,  6,  4,  2,  0);
    __m512i mask1 = _v512_set_epu16(63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33,
                                    31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11,  9,  7,  5,  3,  1);
    a = v_uint16x32(_mm512_permutex2var_epi16(ab0, mask0, ab1));
    b = v_uint16x32(_mm512_permutex2var_epi16(ab0, mask1, ab1));
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x16& a, v_uint32x16& b )
{
    __m512i ab0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i ab1 = _mm512_loadu_si512((const __m512i*)(ptr + 16));
    __m512i mask0 = _v512_set_epu32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i mask1 = _v512_set_epu32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);
    a = v_uint32x16(_mm512_permutex2var_epi32(ab0, mask0, ab1));
    b = v_uint32x16(_mm512_permutex2var_epi32(ab0, mask1, ab1));
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x8& a, v_uint64x8& b )
{
    __m512i ab0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i ab1 = _mm512_loadu_si512((const __m512i*)(ptr + 8));
    __m512i mask0 = _v512_set_epu64(14, 12, 10, 8, 6, 4, 2, 0);
    __m512i mask1 = _v512_set_epu64(15, 13, 11, 9, 7, 5, 3, 1);
    a = v_uint64x8(_mm512_permutex2var_epi64(ab0, mask0, ab1));
    b = v_uint64x8(_mm512_permutex2var_epi64(ab0, mask1, ab1));
}

inline void v_load_deinterleave( const uchar* ptr, v_uint8x64& a, v_uint8x64& b, v_uint8x64& c )
{
    __m512i bgr0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgr1 = _mm512_loadu_si512((const __m512i*)(ptr + 64));
    __m512i bgr2 = _mm512_loadu_si512((const __m512i*)(ptr + 128));

#if CV_AVX_512VBMI2
    __m512i mask0 = _v512_set_epu8(126, 123, 120, 117, 114, 111, 108, 105, 102,  99,  96,  93,  90,  87,  84, 81,
                                    78,  75,  72,  69,  66,  63,  60,  57,  54,  51,  48,  45,  42,  39,  36, 33,
                                    30,  27,  24,  21,  18,  15,  12,   9,   6,   3,   0,  62,  59,  56,  53, 50,
                                    47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,  2);
    __m512i r0b01 = _mm512_permutex2var_epi8(bgr0, mask0, bgr1);
    __m512i b1g12 = _mm512_permutex2var_epi8(bgr1, mask0, bgr2);
    __m512i r12b2 = _mm512_permutex2var_epi8(bgr1,
                    _v512_set_epu8(125, 122, 119, 116, 113, 110, 107, 104, 101,  98,  95,  92,  89,  86,  83, 80,
                                    77,  74,  71,  68,  65, 127, 124, 121, 118, 115, 112, 109, 106, 103, 100, 97,
                                    94,  91,  88,  85,  82,  79,  76,  73,  70,  67,  64,  61,  58,  55,  52, 49,
                                    46,  43,  40,  37,  34,  31,  28,  25,  22,  19,  16,  13,  10,   7,   4,  1), bgr2);
    a = v_uint8x64(_mm512_mask_compress_epi8(r12b2, 0xffffffffffe00000, r0b01));
    b = v_uint8x64(_mm512_mask_compress_epi8(b1g12, 0x2492492492492492, bgr0));
    c = v_uint8x64(_mm512_mask_expand_epi8(r0b01, 0xffffffffffe00000, r12b2));
#elif CV_AVX_512VBMI
    __m512i b0g0b1 = _mm512_mask_blend_epi8(0xb6db6db6db6db6db, bgr1, bgr0);
    __m512i g1r1g2 = _mm512_mask_blend_epi8(0xb6db6db6db6db6db, bgr2, bgr1);
    __m512i r2b2r0 = _mm512_mask_blend_epi8(0xb6db6db6db6db6db, bgr0, bgr2);
    a = v_uint8x64(_mm512_permutex2var_epi8(b0g0b1, _v512_set_epu8(125, 122, 119, 116, 113, 110, 107, 104, 101,  98,  95,  92,  89,  86,  83,  80,
                                                                    77,  74,  71,  68,  65,  63,  61,  60,  58,  57,  55,  54,  52,  51,  49,  48,
                                                                    46,  45,  43,  42,  40,  39,  37,  36,  34,  33,  31,  30,  28,  27,  25,  24,
                                                                    23,  21,  20,  18,  17,  15,  14,  12,  11,   9,   8,   6,   5,   3,   2,   0), bgr2));
    b = v_uint8x64(_mm512_permutex2var_epi8(g1r1g2, _v512_set_epu8( 63,  61,  60,  58,  57,  55,  54,  52,  51,  49,  48,  46,  45,  43,  42,  40,
                                                                    39,  37,  36,  34,  33,  31,  30,  28,  27,  25,  24,  23,  21,  20,  18,  17,
                                                                    15,  14,  12,  11,   9,   8,   6,   5,   3,   2,   0, 126, 123, 120, 117, 114,
                                                                   111, 108, 105, 102,  99,  96,  93,  90,  87,  84,  81,  78,  75,  72,  69,  66), bgr0));
    c = v_uint8x64(_mm512_permutex2var_epi8(r2b2r0, _v512_set_epu8( 63,  60,  57,  54,  51,  48,  45,  42,  39,  36,  33,  30,  27,  24,  21,  18,
                                                                    15,  12,   9,   6,   3,   0, 125, 122, 119, 116, 113, 110, 107, 104, 101,  98,
                                                                    95,  92,  89,  86,  83,  80,  77,  74,  71,  68,  65,  62,  59,  56,  53,  50,
                                                                    47,  44,  41,  38,  35,  32,  29,  26,  23,  20,  17,  14,  11,   8,   5,   2), bgr1));
#else
    __m512i mask0 = _v512_set_epu16(61, 58, 55, 52, 49, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48,
                                    45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12,  9,  6,  3,  0);
    __m512i b01g1 = _mm512_permutex2var_epi16(bgr0, mask0, bgr1);
    __m512i r12b2 = _mm512_permutex2var_epi16(bgr1, mask0, bgr2);
    __m512i g20r0 = _mm512_permutex2var_epi16(bgr2, mask0, bgr0);

    __m512i b0g0 = _mm512_mask_blend_epi32(0xf800, b01g1, r12b2);
    __m512i r0b1 = _mm512_permutex2var_epi16(bgr1, _v512_set_epu16(42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 26, 23, 20, 17,
                                                                   14, 11,  8,  5,  2, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43), g20r0);
    __m512i g1r1 = _mm512_alignr_epi32(r12b2, g20r0, 11);
    a = v_uint8x64(_mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, b0g0, r0b1));
    c = v_uint8x64(_mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, r0b1, g1r1));
    b = v_uint8x64(_mm512_shuffle_epi8(_mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, g1r1, b0g0), _mm512_set4_epi32(0x0e0f0c0d, 0x0a0b0809, 0x06070405, 0x02030001)));
#endif
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x32& a, v_uint16x32& b, v_uint16x32& c )
{
    __m512i bgr0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgr1 = _mm512_loadu_si512((const __m512i*)(ptr + 32));
    __m512i bgr2 = _mm512_loadu_si512((const __m512i*)(ptr + 64));

    __m512i mask0 = _v512_set_epu16(61, 58, 55, 52, 49, 46, 43, 40, 37, 34, 63, 60, 57, 54, 51, 48,
                                    45, 42, 39, 36, 33, 30, 27, 24, 21, 18, 15, 12,  9,  6,  3,  0);
    __m512i b01g1 = _mm512_permutex2var_epi16(bgr0, mask0, bgr1);
    __m512i r12b2 = _mm512_permutex2var_epi16(bgr1, mask0, bgr2);
    __m512i g20r0 = _mm512_permutex2var_epi16(bgr2, mask0, bgr0);

    a = v_uint16x32(_mm512_mask_blend_epi32(0xf800, b01g1, r12b2));
    b = v_uint16x32(_mm512_permutex2var_epi16(bgr1, _v512_set_epu16(42, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 29, 26, 23, 20, 17,
                                                                    14, 11,  8,  5,  2, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 43), g20r0));
    c = v_uint16x32(_mm512_alignr_epi32(r12b2, g20r0, 11));
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x16& a, v_uint32x16& b, v_uint32x16& c )
{
    __m512i bgr0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgr1 = _mm512_loadu_si512((const __m512i*)(ptr + 16));
    __m512i bgr2 = _mm512_loadu_si512((const __m512i*)(ptr + 32));

    __m512i mask0 = _v512_set_epu32(29, 26, 23, 20, 17, 30, 27, 24, 21, 18, 15, 12, 9, 6, 3, 0);
    __m512i b01r1 = _mm512_permutex2var_epi32(bgr0, mask0, bgr1);
    __m512i g12b2 = _mm512_permutex2var_epi32(bgr1, mask0, bgr2);
    __m512i r20g0 = _mm512_permutex2var_epi32(bgr2, mask0, bgr0);

    a = v_uint32x16(_mm512_mask_blend_epi32(0xf800, b01r1, g12b2));
    b = v_uint32x16(_mm512_alignr_epi32(g12b2, r20g0, 11));
    c = v_uint32x16(_mm512_permutex2var_epi32(bgr1, _v512_set_epu32(21, 20, 19, 18, 17, 16, 13, 10, 7, 4, 1, 26, 25, 24, 23, 22), r20g0));
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x8& a, v_uint64x8& b, v_uint64x8& c )
{
    __m512i bgr0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgr1 = _mm512_loadu_si512((const __m512i*)(ptr + 8));
    __m512i bgr2 = _mm512_loadu_si512((const __m512i*)(ptr + 16));

    __m512i mask0 = _v512_set_epu64(13, 10, 15, 12, 9, 6, 3, 0);
    __m512i b01g1 = _mm512_permutex2var_epi64(bgr0, mask0, bgr1);
    __m512i r12b2 = _mm512_permutex2var_epi64(bgr1, mask0, bgr2);
    __m512i g20r0 = _mm512_permutex2var_epi64(bgr2, mask0, bgr0);

    a = v_uint64x8(_mm512_mask_blend_epi64(0xc0, b01g1, r12b2));
    c = v_uint64x8(_mm512_alignr_epi64(r12b2, g20r0, 6));
    b = v_uint64x8(_mm512_permutex2var_epi64(bgr1, _v512_set_epu64(10, 9, 8, 5, 2, 13, 12, 11), g20r0));
}

inline void v_load_deinterleave( const uchar* ptr, v_uint8x64& a, v_uint8x64& b, v_uint8x64& c, v_uint8x64& d )
{
    __m512i bgra0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgra1 = _mm512_loadu_si512((const __m512i*)(ptr + 64));
    __m512i bgra2 = _mm512_loadu_si512((const __m512i*)(ptr + 128));
    __m512i bgra3 = _mm512_loadu_si512((const __m512i*)(ptr + 192));

#if CV_AVX_512VBMI
    __m512i mask0 = _v512_set_epu8(126, 124, 122, 120, 118, 116, 114, 112, 110, 108, 106, 104, 102, 100, 98, 96,
                                    94,  92,  90,  88,  86,  84,  82,  80,  78,  76,  74,  72,  70,  68, 66, 64,
                                    62,  60,  58,  56,  54,  52,  50,  48,  46,  44,  42,  40,  38,  36, 34, 32,
                                    30,  28,  26,  24,  22,  20,  18,  16,  14,  12,  10,   8,   6,   4,  2,  0);
    __m512i mask1 = _v512_set_epu8(127, 125, 123, 121, 119, 117, 115, 113, 111, 109, 107, 105, 103, 101, 99, 97,
                                    95,  93,  91,  89,  87,  85,  83,  81,  79,  77,  75,  73,  71,  69, 67, 65,
                                    63,  61,  59,  57,  55,  53,  51,  49,  47,  45,  43,  41,  39,  37, 35, 33,
                                    31,  29,  27,  25,  23,  21,  19,  17,  15,  13,  11,   9,   7,   5,  3,  1);

    __m512i br01 = _mm512_permutex2var_epi8(bgra0, mask0, bgra1);
    __m512i ga01 = _mm512_permutex2var_epi8(bgra0, mask1, bgra1);
    __m512i br23 = _mm512_permutex2var_epi8(bgra2, mask0, bgra3);
    __m512i ga23 = _mm512_permutex2var_epi8(bgra2, mask1, bgra3);

    a = v_uint8x64(_mm512_permutex2var_epi8(br01, mask0, br23));
    c = v_uint8x64(_mm512_permutex2var_epi8(br01, mask1, br23));
    b = v_uint8x64(_mm512_permutex2var_epi8(ga01, mask0, ga23));
    d = v_uint8x64(_mm512_permutex2var_epi8(ga01, mask1, ga23));
#else
    __m512i mask = _mm512_set4_epi32(0x0f0b0703, 0x0e0a0602, 0x0d090501, 0x0c080400);
    __m512i b0g0r0a0 = _mm512_shuffle_epi8(bgra0, mask);
    __m512i b1g1r1a1 = _mm512_shuffle_epi8(bgra1, mask);
    __m512i b2g2r2a2 = _mm512_shuffle_epi8(bgra2, mask);
    __m512i b3g3r3a3 = _mm512_shuffle_epi8(bgra3, mask);

    __m512i mask0 = _v512_set_epu32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i mask1 = _v512_set_epu32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    __m512i br01 = _mm512_permutex2var_epi32(b0g0r0a0, mask0, b1g1r1a1);
    __m512i ga01 = _mm512_permutex2var_epi32(b0g0r0a0, mask1, b1g1r1a1);
    __m512i br23 = _mm512_permutex2var_epi32(b2g2r2a2, mask0, b3g3r3a3);
    __m512i ga23 = _mm512_permutex2var_epi32(b2g2r2a2, mask1, b3g3r3a3);

    a = v_uint8x64(_mm512_permutex2var_epi32(br01, mask0, br23));
    c = v_uint8x64(_mm512_permutex2var_epi32(br01, mask1, br23));
    b = v_uint8x64(_mm512_permutex2var_epi32(ga01, mask0, ga23));
    d = v_uint8x64(_mm512_permutex2var_epi32(ga01, mask1, ga23));
#endif
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x32& a, v_uint16x32& b, v_uint16x32& c, v_uint16x32& d )
{
    __m512i bgra0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgra1 = _mm512_loadu_si512((const __m512i*)(ptr + 32));
    __m512i bgra2 = _mm512_loadu_si512((const __m512i*)(ptr + 64));
    __m512i bgra3 = _mm512_loadu_si512((const __m512i*)(ptr + 96));

    __m512i mask0 = _v512_set_epu16(62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
                                    30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,  8,  6,  4,  2,  0);
    __m512i mask1 = _v512_set_epu16(63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33,
                                    31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11,  9,  7,  5,  3,  1);

    __m512i br01 = _mm512_permutex2var_epi16(bgra0, mask0, bgra1);
    __m512i ga01 = _mm512_permutex2var_epi16(bgra0, mask1, bgra1);
    __m512i br23 = _mm512_permutex2var_epi16(bgra2, mask0, bgra3);
    __m512i ga23 = _mm512_permutex2var_epi16(bgra2, mask1, bgra3);

    a = v_uint16x32(_mm512_permutex2var_epi16(br01, mask0, br23));
    c = v_uint16x32(_mm512_permutex2var_epi16(br01, mask1, br23));
    b = v_uint16x32(_mm512_permutex2var_epi16(ga01, mask0, ga23));
    d = v_uint16x32(_mm512_permutex2var_epi16(ga01, mask1, ga23));
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x16& a, v_uint32x16& b, v_uint32x16& c, v_uint32x16& d )
{
    __m512i bgra0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgra1 = _mm512_loadu_si512((const __m512i*)(ptr + 16));
    __m512i bgra2 = _mm512_loadu_si512((const __m512i*)(ptr + 32));
    __m512i bgra3 = _mm512_loadu_si512((const __m512i*)(ptr + 48));

    __m512i mask0 = _v512_set_epu32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0);
    __m512i mask1 = _v512_set_epu32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1);

    __m512i br01 = _mm512_permutex2var_epi32(bgra0, mask0, bgra1);
    __m512i ga01 = _mm512_permutex2var_epi32(bgra0, mask1, bgra1);
    __m512i br23 = _mm512_permutex2var_epi32(bgra2, mask0, bgra3);
    __m512i ga23 = _mm512_permutex2var_epi32(bgra2, mask1, bgra3);

    a = v_uint32x16(_mm512_permutex2var_epi32(br01, mask0, br23));
    c = v_uint32x16(_mm512_permutex2var_epi32(br01, mask1, br23));
    b = v_uint32x16(_mm512_permutex2var_epi32(ga01, mask0, ga23));
    d = v_uint32x16(_mm512_permutex2var_epi32(ga01, mask1, ga23));
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x8& a, v_uint64x8& b, v_uint64x8& c, v_uint64x8& d )
{
    __m512i bgra0 = _mm512_loadu_si512((const __m512i*)ptr);
    __m512i bgra1 = _mm512_loadu_si512((const __m512i*)(ptr + 8));
    __m512i bgra2 = _mm512_loadu_si512((const __m512i*)(ptr + 16));
    __m512i bgra3 = _mm512_loadu_si512((const __m512i*)(ptr + 24));

    __m512i mask0 = _v512_set_epu64(14, 12, 10, 8, 6, 4, 2, 0);
    __m512i mask1 = _v512_set_epu64(15, 13, 11, 9, 7, 5, 3, 1);

    __m512i br01 = _mm512_permutex2var_epi64(bgra0, mask0, bgra1);
    __m512i ga01 = _mm512_permutex2var_epi64(bgra0, mask1, bgra1);
    __m512i br23 = _mm512_permutex2var_epi64(bgra2, mask0, bgra3);
    __m512i ga23 = _mm512_permutex2var_epi64(bgra2, mask1, bgra3);

    a = v_uint64x8(_mm512_permutex2var_epi64(br01, mask0, br23));
    c = v_uint64x8(_mm512_permutex2var_epi64(br01, mask1, br23));
    b = v_uint64x8(_mm512_permutex2var_epi64(ga01, mask0, ga23));
    d = v_uint64x8(_mm512_permutex2var_epi64(ga01, mask1, ga23));
}

///////////////////////////// store interleave /////////////////////////////////////

inline void v_store_interleave( uchar* ptr, const v_uint8x64& x, const v_uint8x64& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint8x64 low, high;
    v_zip(x, y, low, high);
    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, low.val);
        _mm512_stream_si512((__m512i*)(ptr + 64), high.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, low.val);
        _mm512_store_si512((__m512i*)(ptr + 64), high.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, low.val);
        _mm512_storeu_si512((__m512i*)(ptr + 64), high.val);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x32& x, const v_uint16x32& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint16x32 low, high;
    v_zip(x, y, low, high);
    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, low.val);
        _mm512_stream_si512((__m512i*)(ptr + 32), high.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, low.val);
        _mm512_store_si512((__m512i*)(ptr + 32), high.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, low.val);
        _mm512_storeu_si512((__m512i*)(ptr + 32), high.val);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x16& x, const v_uint32x16& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint32x16 low, high;
    v_zip(x, y, low, high);
    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, low.val);
        _mm512_stream_si512((__m512i*)(ptr + 16), high.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, low.val);
        _mm512_store_si512((__m512i*)(ptr + 16), high.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, low.val);
        _mm512_storeu_si512((__m512i*)(ptr + 16), high.val);
    }
}

inline void v_store_interleave( uint64* ptr, const v_uint64x8& x, const v_uint64x8& y,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint64x8 low, high;
    v_zip(x, y, low, high);
    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, low.val);
        _mm512_stream_si512((__m512i*)(ptr + 8), high.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, low.val);
        _mm512_store_si512((__m512i*)(ptr + 8), high.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, low.val);
        _mm512_storeu_si512((__m512i*)(ptr + 8), high.val);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x64& a, const v_uint8x64& b, const v_uint8x64& c,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
#if CV_AVX_512VBMI
    __m512i mask0 = _v512_set_epu8(127,  84,  20, 126,  83,  19, 125,  82,  18, 124,  81,  17, 123,  80,  16, 122,
                                    79,  15, 121,  78,  14, 120,  77,  13, 119,  76,  12, 118,  75,  11, 117,  74,
                                    10, 116,  73,   9, 115,  72,   8, 114,  71,   7, 113,  70,   6, 112,  69,   5,
                                   111,  68,   4, 110,  67,   3, 109,  66,   2, 108,  65,   1, 107,  64,   0, 106);
    __m512i mask1 = _v512_set_epu8( 21,  42, 105,  20,  41, 104,  19,  40, 103,  18,  39, 102,  17,  38, 101,  16,
                                    37, 100,  15,  36,  99,  14,  35,  98,  13,  34,  97,  12,  33,  96,  11,  32,
                                    95,  10,  31,  94,   9,  30,  93,   8,  29,  92,   7,  28,  91,   6,  27,  90,
                                     5,  26,  89,   4,  25,  88,   3,  24,  87,   2,  23,  86,   1,  22,  85,   0);
    __m512i mask2 = _v512_set_epu8(106, 127,  63, 105, 126,  62, 104, 125,  61, 103, 124,  60, 102, 123,  59, 101,
                                   122,  58, 100, 121,  57,  99, 120,  56,  98, 119,  55,  97, 118,  54,  96, 117,
                                    53,  95, 116,  52,  94, 115,  51,  93, 114,  50,  92, 113,  49,  91, 112,  48,
                                    90, 111,  47,  89, 110,  46,  88, 109,  45,  87, 108,  44,  86, 107,  43,  85);
    __m512i r2g0r0 = _mm512_permutex2var_epi8(b.val, mask0, c.val);
    __m512i b0r1b1 = _mm512_permutex2var_epi8(a.val, mask1, c.val);
    __m512i g1b2g2 = _mm512_permutex2var_epi8(a.val, mask2, b.val);

    __m512i bgr0 = _mm512_mask_blend_epi8(0x9249249249249249, r2g0r0, b0r1b1);
    __m512i bgr1 = _mm512_mask_blend_epi8(0x9249249249249249, b0r1b1, g1b2g2);
    __m512i bgr2 = _mm512_mask_blend_epi8(0x9249249249249249, g1b2g2, r2g0r0);
#else
    __m512i g1g0 = _mm512_shuffle_epi8(b.val, _mm512_set4_epi32(0x0e0f0c0d, 0x0a0b0809, 0x06070405, 0x02030001));
    __m512i b0g0 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, a.val, g1g0);
    __m512i r0b1 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, c.val, a.val);
    __m512i g1r1 = _mm512_mask_blend_epi8(0xAAAAAAAAAAAAAAAA, g1g0, c.val);

    __m512i mask0 = _v512_set_epu16(42, 10, 31, 41,  9, 30, 40,  8, 29, 39,  7, 28, 38,  6, 27, 37,
                                     5, 26, 36,  4, 25, 35,  3, 24, 34,  2, 23, 33,  1, 22, 32,  0);
    __m512i mask1 = _v512_set_epu16(21, 52, 41, 20, 51, 40, 19, 50, 39, 18, 49, 38, 17, 48, 37, 16,
                                    47, 36, 15, 46, 35, 14, 45, 34, 13, 44, 33, 12, 43, 32, 11, 42);
    __m512i mask2 = _v512_set_epu16(63, 31, 20, 62, 30, 19, 61, 29, 18, 60, 28, 17, 59, 27, 16, 58,
                                    26, 15, 57, 25, 14, 56, 24, 13, 55, 23, 12, 54, 22, 11, 53, 21);
    __m512i b0g0b2 = _mm512_permutex2var_epi16(b0g0, mask0, r0b1);
    __m512i r1b1r0 = _mm512_permutex2var_epi16(b0g0, mask1, g1r1);
    __m512i g2r2g1 = _mm512_permutex2var_epi16(r0b1, mask2, g1r1);

    __m512i bgr0 = _mm512_mask_blend_epi16(0x24924924, b0g0b2, r1b1r0);
    __m512i bgr1 = _mm512_mask_blend_epi16(0x24924924, r1b1r0, g2r2g1);
    __m512i bgr2 = _mm512_mask_blend_epi16(0x24924924, g2r2g1, b0g0b2);
#endif

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgr0);
        _mm512_stream_si512((__m512i*)(ptr + 64), bgr1);
        _mm512_stream_si512((__m512i*)(ptr + 128), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgr0);
        _mm512_store_si512((__m512i*)(ptr + 64), bgr1);
        _mm512_store_si512((__m512i*)(ptr + 128), bgr2);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgr0);
        _mm512_storeu_si512((__m512i*)(ptr + 64), bgr1);
        _mm512_storeu_si512((__m512i*)(ptr + 128), bgr2);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x32& a, const v_uint16x32& b, const v_uint16x32& c,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i mask0 = _v512_set_epu16(42, 10, 31, 41,  9, 30, 40,  8, 29, 39,  7, 28, 38,  6, 27, 37,
                                     5, 26, 36,  4, 25, 35,  3, 24, 34,  2, 23, 33,  1, 22, 32,  0);
    __m512i mask1 = _v512_set_epu16(21, 52, 41, 20, 51, 40, 19, 50, 39, 18, 49, 38, 17, 48, 37, 16,
                                    47, 36, 15, 46, 35, 14, 45, 34, 13, 44, 33, 12, 43, 32, 11, 42);
    __m512i mask2 = _v512_set_epu16(63, 31, 20, 62, 30, 19, 61, 29, 18, 60, 28, 17, 59, 27, 16, 58,
                                    26, 15, 57, 25, 14, 56, 24, 13, 55, 23, 12, 54, 22, 11, 53, 21);
    __m512i b0g0b2 = _mm512_permutex2var_epi16(a.val, mask0, b.val);
    __m512i r1b1r0 = _mm512_permutex2var_epi16(a.val, mask1, c.val);
    __m512i g2r2g1 = _mm512_permutex2var_epi16(b.val, mask2, c.val);

    __m512i bgr0 = _mm512_mask_blend_epi16(0x24924924, b0g0b2, r1b1r0);
    __m512i bgr1 = _mm512_mask_blend_epi16(0x24924924, r1b1r0, g2r2g1);
    __m512i bgr2 = _mm512_mask_blend_epi16(0x24924924, g2r2g1, b0g0b2);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgr0);
        _mm512_stream_si512((__m512i*)(ptr + 32), bgr1);
        _mm512_stream_si512((__m512i*)(ptr + 64), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgr0);
        _mm512_store_si512((__m512i*)(ptr + 32), bgr1);
        _mm512_store_si512((__m512i*)(ptr + 64), bgr2);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgr0);
        _mm512_storeu_si512((__m512i*)(ptr + 32), bgr1);
        _mm512_storeu_si512((__m512i*)(ptr + 64), bgr2);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x16& a, const v_uint32x16& b, const v_uint32x16& c,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i mask0 = _v512_set_epu32(26, 31, 15, 25, 30, 14, 24, 29, 13, 23, 28, 12, 22, 27, 11, 21);
    __m512i mask1 = _v512_set_epu32(31, 10, 25, 30,  9, 24, 29,  8, 23, 28,  7, 22, 27,  6, 21, 26);
    __m512i g1b2g2 = _mm512_permutex2var_epi32(a.val, mask0, b.val);
    __m512i r2r1b1 = _mm512_permutex2var_epi32(a.val, mask1, c.val);

    __m512i bgr0 = _mm512_mask_expand_epi32(_mm512_mask_expand_epi32(_mm512_maskz_expand_epi32(0x9249, a.val), 0x2492, b.val), 0x4924, c.val);
    __m512i bgr1 = _mm512_mask_blend_epi32(0x9249, r2r1b1, g1b2g2);
    __m512i bgr2 = _mm512_mask_blend_epi32(0x9249, g1b2g2, r2r1b1);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgr0);
        _mm512_stream_si512((__m512i*)(ptr + 16), bgr1);
        _mm512_stream_si512((__m512i*)(ptr + 32), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgr0);
        _mm512_store_si512((__m512i*)(ptr + 16), bgr1);
        _mm512_store_si512((__m512i*)(ptr + 32), bgr2);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgr0);
        _mm512_storeu_si512((__m512i*)(ptr + 16), bgr1);
        _mm512_storeu_si512((__m512i*)(ptr + 32), bgr2);
    }
}

inline void v_store_interleave( uint64* ptr, const v_uint64x8& a, const v_uint64x8& b, const v_uint64x8& c,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    __m512i mask0 = _v512_set_epu64( 5, 12,  7,  4, 11,  6,  3, 10);
    __m512i mask1 = _v512_set_epu64(15,  7,  4, 14,  6,  3, 13,  5);
    __m512i r1b1b2 = _mm512_permutex2var_epi64(a.val, mask0, c.val);
    __m512i g2r2g1 = _mm512_permutex2var_epi64(b.val, mask1, c.val);

    __m512i bgr0 = _mm512_mask_expand_epi64(_mm512_mask_expand_epi64(_mm512_maskz_expand_epi64(0x49, a.val), 0x92, b.val), 0x24, c.val);
    __m512i bgr1 = _mm512_mask_blend_epi64(0xdb, g2r2g1, r1b1b2);
    __m512i bgr2 = _mm512_mask_blend_epi64(0xdb, r1b1b2, g2r2g1);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgr0);
        _mm512_stream_si512((__m512i*)(ptr + 8), bgr1);
        _mm512_stream_si512((__m512i*)(ptr + 16), bgr2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgr0);
        _mm512_store_si512((__m512i*)(ptr + 8), bgr1);
        _mm512_store_si512((__m512i*)(ptr + 16), bgr2);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgr0);
        _mm512_storeu_si512((__m512i*)(ptr + 8), bgr1);
        _mm512_storeu_si512((__m512i*)(ptr + 16), bgr2);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x64& a, const v_uint8x64& b,
                                const v_uint8x64& c, const v_uint8x64& d,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint8x64 br01, br23, ga01, ga23;
    v_zip(a, c, br01, br23);
    v_zip(b, d, ga01, ga23);
    v_uint8x64 bgra0, bgra1, bgra2, bgra3;
    v_zip(br01, ga01, bgra0, bgra1);
    v_zip(br23, ga23, bgra2, bgra3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgra0.val);
        _mm512_stream_si512((__m512i*)(ptr + 64), bgra1.val);
        _mm512_stream_si512((__m512i*)(ptr + 128), bgra2.val);
        _mm512_stream_si512((__m512i*)(ptr + 192), bgra3.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgra0.val);
        _mm512_store_si512((__m512i*)(ptr + 64), bgra1.val);
        _mm512_store_si512((__m512i*)(ptr + 128), bgra2.val);
        _mm512_store_si512((__m512i*)(ptr + 192), bgra3.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgra0.val);
        _mm512_storeu_si512((__m512i*)(ptr + 64), bgra1.val);
        _mm512_storeu_si512((__m512i*)(ptr + 128), bgra2.val);
        _mm512_storeu_si512((__m512i*)(ptr + 192), bgra3.val);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x32& a, const v_uint16x32& b,
                                const v_uint16x32& c, const v_uint16x32& d,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint16x32 br01, br23, ga01, ga23;
    v_zip(a, c, br01, br23);
    v_zip(b, d, ga01, ga23);
    v_uint16x32 bgra0, bgra1, bgra2, bgra3;
    v_zip(br01, ga01, bgra0, bgra1);
    v_zip(br23, ga23, bgra2, bgra3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgra0.val);
        _mm512_stream_si512((__m512i*)(ptr + 32), bgra1.val);
        _mm512_stream_si512((__m512i*)(ptr + 64), bgra2.val);
        _mm512_stream_si512((__m512i*)(ptr + 96), bgra3.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgra0.val);
        _mm512_store_si512((__m512i*)(ptr + 32), bgra1.val);
        _mm512_store_si512((__m512i*)(ptr + 64), bgra2.val);
        _mm512_store_si512((__m512i*)(ptr + 96), bgra3.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgra0.val);
        _mm512_storeu_si512((__m512i*)(ptr + 32), bgra1.val);
        _mm512_storeu_si512((__m512i*)(ptr + 64), bgra2.val);
        _mm512_storeu_si512((__m512i*)(ptr + 96), bgra3.val);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x16& a, const v_uint32x16& b,
                                const v_uint32x16& c, const v_uint32x16& d,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint32x16 br01, br23, ga01, ga23;
    v_zip(a, c, br01, br23);
    v_zip(b, d, ga01, ga23);
    v_uint32x16 bgra0, bgra1, bgra2, bgra3;
    v_zip(br01, ga01, bgra0, bgra1);
    v_zip(br23, ga23, bgra2, bgra3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgra0.val);
        _mm512_stream_si512((__m512i*)(ptr + 16), bgra1.val);
        _mm512_stream_si512((__m512i*)(ptr + 32), bgra2.val);
        _mm512_stream_si512((__m512i*)(ptr + 48), bgra3.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgra0.val);
        _mm512_store_si512((__m512i*)(ptr + 16), bgra1.val);
        _mm512_store_si512((__m512i*)(ptr + 32), bgra2.val);
        _mm512_store_si512((__m512i*)(ptr + 48), bgra3.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgra0.val);
        _mm512_storeu_si512((__m512i*)(ptr + 16), bgra1.val);
        _mm512_storeu_si512((__m512i*)(ptr + 32), bgra2.val);
        _mm512_storeu_si512((__m512i*)(ptr + 48), bgra3.val);
    }
}

inline void v_store_interleave( uint64* ptr, const v_uint64x8& a, const v_uint64x8& b,
                                const v_uint64x8& c, const v_uint64x8& d,
                                hal::StoreMode mode=hal::STORE_UNALIGNED )
{
    v_uint64x8 br01, br23, ga01, ga23;
    v_zip(a, c, br01, br23);
    v_zip(b, d, ga01, ga23);
    v_uint64x8 bgra0, bgra1, bgra2, bgra3;
    v_zip(br01, ga01, bgra0, bgra1);
    v_zip(br23, ga23, bgra2, bgra3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm512_stream_si512((__m512i*)ptr, bgra0.val);
        _mm512_stream_si512((__m512i*)(ptr + 8), bgra1.val);
        _mm512_stream_si512((__m512i*)(ptr + 16), bgra2.val);
        _mm512_stream_si512((__m512i*)(ptr + 24), bgra3.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm512_store_si512((__m512i*)ptr, bgra0.val);
        _mm512_store_si512((__m512i*)(ptr + 8), bgra1.val);
        _mm512_store_si512((__m512i*)(ptr + 16), bgra2.val);
        _mm512_store_si512((__m512i*)(ptr + 24), bgra3.val);
    }
    else
    {
        _mm512_storeu_si512((__m512i*)ptr, bgra0.val);
        _mm512_storeu_si512((__m512i*)(ptr + 8), bgra1.val);
        _mm512_storeu_si512((__m512i*)(ptr + 16), bgra2.val);
        _mm512_storeu_si512((__m512i*)(ptr + 24), bgra3.val);
    }
}

#define OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1) \
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

OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(v_int8x64, schar, s8, v_uint8x64, uchar, u8)
OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(v_int16x32, short, s16, v_uint16x32, ushort, u16)
OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(v_int32x16, int, s32, v_uint32x16, unsigned, u32)
OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(v_float32x16, float, f32, v_uint32x16, unsigned, u32)
OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(v_int64x8, int64, s64, v_uint64x8, uint64, u64)
OPENCV_HAL_IMPL_AVX512_LOADSTORE_INTERLEAVE(v_float64x8, double, f64, v_uint64x8, uint64, u64)

////////// Mask and checks /////////

/** Mask **/
inline int64 v_signmask(const v_int8x64& a) { return (int64)_mm512_movepi8_mask(a.val); }
inline int v_signmask(const v_int16x32& a) { return (int)_mm512_cmp_epi16_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_LT); }
inline int v_signmask(const v_int32x16& a) { return (int)_mm512_cmp_epi32_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_LT); }
inline int v_signmask(const v_int64x8& a) { return (int)_mm512_cmp_epi64_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_LT); }

inline int64 v_signmask(const v_uint8x64& a) { return v_signmask(v_reinterpret_as_s8(a)); }
inline int v_signmask(const v_uint16x32& a) { return v_signmask(v_reinterpret_as_s16(a)); }
inline int v_signmask(const v_uint32x16& a) { return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_uint64x8& a) { return v_signmask(v_reinterpret_as_s64(a)); }
inline int v_signmask(const v_float32x16& a) { return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_float64x8& a) { return v_signmask(v_reinterpret_as_s64(a)); }

/** Checks **/
inline bool v_check_all(const v_int8x64& a) { return !(bool)_mm512_cmp_epi8_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_NLT); }
inline bool v_check_any(const v_int8x64& a) { return (bool)_mm512_movepi8_mask(a.val); }
inline bool v_check_all(const v_int16x32& a) { return !(bool)_mm512_cmp_epi16_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_NLT); }
inline bool v_check_any(const v_int16x32& a) { return (bool)_mm512_cmp_epi16_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_LT); }
inline bool v_check_all(const v_int32x16& a) { return !(bool)_mm512_cmp_epi32_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_NLT); }
inline bool v_check_any(const v_int32x16& a) { return (bool)_mm512_cmp_epi32_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_LT); }
inline bool v_check_all(const v_int64x8& a) { return !(bool)_mm512_cmp_epi64_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_NLT); }
inline bool v_check_any(const v_int64x8& a) { return (bool)_mm512_cmp_epi64_mask(a.val, _mm512_setzero_si512(), _MM_CMPINT_LT); }

inline bool v_check_all(const v_float32x16& a) { return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float32x16& a) { return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_float64x8& a) { return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_float64x8& a) { return v_check_any(v_reinterpret_as_s64(a)); }
inline bool v_check_all(const v_uint8x64& a) { return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_all(const v_uint16x32& a) { return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_all(const v_uint32x16& a) { return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_uint64x8& a) { return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_uint8x64& a) { return v_check_any(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint16x32& a) { return v_check_any(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint32x16& a) { return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_uint64x8& a) { return v_check_any(v_reinterpret_as_s64(a)); }

inline int v_scan_forward(const v_int8x64& a)
{
    int64 mask = _mm512_movepi8_mask(a.val);
    int mask32 = (int)mask;
    return mask != 0 ? mask32 != 0 ? trailingZeros32(mask32) : 32 + trailingZeros32((int)(mask >> 32)) : 0;
}
inline int v_scan_forward(const v_uint8x64& a) { return v_scan_forward(v_reinterpret_as_s8(a)); }
inline int v_scan_forward(const v_int16x32& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))); }
inline int v_scan_forward(const v_uint16x32& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))); }
inline int v_scan_forward(const v_int32x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))) / 2; }
inline int v_scan_forward(const v_uint32x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))) / 2; }
inline int v_scan_forward(const v_float32x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))) / 2; }
inline int v_scan_forward(const v_int64x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))) / 4; }
inline int v_scan_forward(const v_uint64x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))) / 4; }
inline int v_scan_forward(const v_float64x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s16(a))) / 4; }

inline void v512_cleanup() { _mm256_zeroall(); }

#include "intrin_math.hpp"
inline v_float32x16 v_exp(v_float32x16 x) { return v_exp_default_32f<v_float32x16, v_int32x16>(x); }
inline v_float32x16 v_log(v_float32x16 x) { return v_log_default_32f<v_float32x16, v_int32x16>(x); }
inline v_float32x16 v_erf(v_float32x16 x) { return v_erf_default_32f<v_float32x16, v_int32x16>(x); }

inline v_float64x8 v_exp(v_float64x8 x) { return v_exp_default_64f<v_float64x8, v_int64x8>(x); }
inline v_float64x8 v_log(v_float64x8 x) { return v_log_default_64f<v_float64x8, v_int64x8>(x); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // OPENCV_HAL_INTRIN_AVX_HPP
