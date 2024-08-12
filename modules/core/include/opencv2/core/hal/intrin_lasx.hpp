// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_INTRIN_LASX_HPP
#define OPENCV_HAL_INTRIN_LASX_HPP

#include <lsxintrin.h>
#include <lasxintrin.h>

#define CV_SIMD256 1
#define CV_SIMD256_64F 1
#define CV_SIMD256_FP16 0

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Utils ////////////

inline __m256i _v256_setr_b(char v0, char v1, char v2, char v3, char v4, char v5, char v6, char v7, char v8,  char v9,
                    char v10, char v11, char v12, char v13, char v14, char v15, char v16, char v17, char v18, char v19,
                    char v20, char v21, char v22, char v23, char v24, char v25, char v26, char v27, char v28, char v29,
                    char v30, char v31)
{
    return (__m256i)v32i8{ v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
                           v10, v11, v12, v13, v14, v15, v16, v17, v18, v19,
                           v20, v21, v22, v23, v24, v25, v26, v27, v28, v29,
                           v30, v31 };
}

inline __m256i _v256_set_b(char v0, char v1, char v2, char v3, char v4, char v5, char v6, char v7, char v8,  char v9,
                   char v10, char v11, char v12, char v13, char v14, char v15, char v16, char v17, char v18, char v19,
                   char v20, char v21, char v22, char v23, char v24, char v25, char v26, char v27, char v28, char v29,
                   char v30, char v31)
{
    return (__m256i)v32i8{ v31, v30,
                           v29, v28, v27, v26, v25, v24, v23, v22, v21, v20,
                           v19, v18, v17, v16, v15, v14, v13, v12, v11, v10,
                           v9, v8, v7, v6, v5, v4, v3, v2, v1, v0 };
}

inline __m256i _v256_setr_h(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7,
                            short v8,  short v9, short v10, short v11, short v12, short v13, short v14, short v15)
{
    return (__m256i)v16i16{ v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15 };
}

inline __m256i _v256_setr_w(int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7)
{
    return (__m256i)v8i32{ v0, v1, v2, v3, v4, v5, v6, v7 };
}

inline __m256i _v256_set_w(int v0, int v1, int v2, int v3, int v4, int v5, int v6, int v7)
{
    return (__m256i)v8i32{ v7, v6, v5, v4, v3, v2, v1, v0 };
}

inline __m256i _v256_setall_w(int v0)
{
    return (__m256i)v8i32{ v0, v0, v0, v0, v0, v0, v0, v0 };
}

inline __m256i _v256_setr_d(int64 v0, int64 v1, int64 v2, int64 v3)
{
    return (__m256i)v4i64{ v0, v1, v2, v3 };
}

inline __m256i _v256_set_d(int64 v0, int64 v1, int64 v2, int64 v3)
{
    return (__m256i)v4i64{ v3, v2, v1, v0 };
}

inline __m256 _v256_setr_ps(float v0, float v1, float v2, float v3, float v4, float v5, float v6, float v7)
{
    return (__m256)v8f32{ v0, v1, v2, v3, v4, v5, v6, v7 };
}

inline __m256 _v256_setall_ps(float f32)
{
    return (__m256)v8f32{ f32, f32, f32, f32, f32, f32, f32, f32 };
}

inline __m256d _v256_setr_pd(double v0, double v1, double v2, double v3)
{
    return (__m256d)v4f64{ v0, v1, v2, v3 };
}

inline __m256d _v256_setall_pd(double f64)
{
    return (__m256d)v4f64{ f64, f64, f64, f64 };
}

inline __m256i _lasx_packus_h(const __m256i& a, const __m256i& b)
{
    return __lasx_xvssrarni_bu_h(b, a, 0);
}

inline __m256i _lasx_packs_h(const __m256i& a, const __m256i& b)
{
    return __lasx_xvssrarni_b_h(b, a, 0);
}

inline __m256i _lasx_packus_w(const __m256i& a, const __m256i& b)
{
    return __lasx_xvssrarni_hu_w(b, a, 0);
}

inline __m256i _lasx_packs_w(const __m256i& a, const __m256i& b)
{
    return __lasx_xvssrarni_h_w(b, a, 0);
}

inline __m256i _v256_combine(const __m128i& lo, const __m128i& hi)
{ return __lasx_xvpermi_q(*((__m256i*)&lo), *((__m256i*)&hi), 0x02); }

inline __m256 _v256_combine(const __m128& lo, const __m128& hi)
{ return __m256(__lasx_xvpermi_q(*((__m256i*)&lo), *((__m256i*)&hi), 0x02)); }

inline __m256d _v256_combine(const __m128d& lo, const __m128d& hi)
{ return __m256d(__lasx_xvpermi_q(*((__m256i*)&lo), *((__m256i*)&hi), 0x02)); }

inline __m256i _v256_shuffle_odd_64(const __m256i& v)
{ return __lasx_xvpermi_d(v, 0xd8); }

inline __m256d _v256_shuffle_odd_64(const __m256d& v)
{ return __m256d(__lasx_xvpermi_d(*((__m256i*)&v), 0xd8)); }

//LASX: only use for permute WITHOUT zero clearing
template<int imm>
inline __m256i _v256_permute2x128(const __m256i& a, const __m256i& b)
{ return __lasx_xvpermi_q(a, b, imm); }

template<int imm>
inline __m256 _v256_permute2x128(const __m256& a, const __m256& b)
{ return __m256(__lasx_xvpermi_q(*((__m256i*)&a), *((__m256i*)&b), imm)); }

template<int imm>
inline __m256d _v256_permute2x128(const __m256d& a, const __m256d& b)
{ return __m256d(__lasx_xvpermi_q(*((__m256i*)&a), *((__m256i*)&b), imm)); }

template<int imm, typename _Tpvec>
inline _Tpvec v256_permute2x128(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_v256_permute2x128<imm>(a.val, b.val)); }

template<int imm>
inline __m256i _v256_permute4x64(const __m256i& a)
{ return __lasx_xvpermi_d(a, imm); }

template<int imm>
inline __m256d _v256_permute4x64(const __m256d& a)
{ return __m256d(__lasx_xvpermi_d(*((__m256i*)&a), imm)); }

template<int imm, typename _Tpvec>
inline _Tpvec v256_permute4x64(const _Tpvec& a)
{ return _Tpvec(_v256_permute4x64<imm>(a.val)); }

inline __m128i _v256_extract_high(const __m256i& v)
{ __m256i temp256i = __lasx_xvpermi_d(v, 0x4E);
  return *((__m128i*)&temp256i); }

inline __m128  _v256_extract_high(const __m256& v)
{ return __m128(_v256_extract_high(*((__m256i*)&v))); }

inline __m128d _v256_extract_high(const __m256d& v)
{ return __m128d(_v256_extract_high(*((__m256i*)&v))); }

inline __m128i _v256_extract_low(const __m256i& v)
{ return *((__m128i*)&v); }

inline __m128  _v256_extract_low(const __m256& v)
{ return __m128(_v256_extract_low(*((__m256i*)&v))); }

inline __m128d _v256_extract_low(const __m256d& v)
{ return __m128d(_v256_extract_low(*((__m256i*)&v))); }

inline __m256i _v256_packs_epu32(const __m256i& a, const __m256i& b)
{
    return __lasx_xvssrlrni_hu_w(b, a, 0);
}

template<int i>
inline int _v256_extract_b(const __m256i& a)
{
    int des[1] = {0};
    __lasx_xvstelm_b(a, des, 0, i);
    return des[0];
}

template<int i>
inline int _v256_extract_h(const __m256i& a)
{
    int des[1] = {0};
    __lasx_xvstelm_h(a, des, 0, i);
    return des[0];
}

template<int i>
inline int _v256_extract_w(const __m256i& a)
{
    return __lasx_xvpickve2gr_w(a, i);
}

template<int i>
inline int64 _v256_extract_d(const __m256i& a)
{
    return __lasx_xvpickve2gr_d(a, i);
}

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
        val = _v256_setr_b((char)v0, (char)v1, (char)v2, (char)v3,
            (char)v4,  (char)v5,  (char)v6 , (char)v7,  (char)v8,  (char)v9,
            (char)v10, (char)v11, (char)v12, (char)v13, (char)v14, (char)v15,
            (char)v16, (char)v17, (char)v18, (char)v19, (char)v20, (char)v21,
            (char)v22, (char)v23, (char)v24, (char)v25, (char)v26, (char)v27,
            (char)v28, (char)v29, (char)v30, (char)v31);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_uint8x32() {}

    uchar get0() const {
        uchar des[1] = {0};
        __lasx_xvstelm_b(val, des, 0, 0);
        return des[0];
    }
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
        val = _v256_setr_b(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9,
            v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20,
            v21, v22, v23, v24, v25, v26, v27, v28, v29, v30, v31);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_int8x32() {}

    schar get0() const {
        schar des[1] = {0};
        __lasx_xvstelm_b(val, des, 0, 0);
        return des[0];
    }
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
        val = _v256_setr_h((short)v0, (short)v1, (short)v2, (short)v3,
            (short)v4,  (short)v5,  (short)v6,  (short)v7,  (short)v8,  (short)v9,
            (short)v10, (short)v11, (short)v12, (short)v13, (short)v14, (short)v15);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_uint16x16() {}

    ushort get0() const {
        ushort des[1] = {0};
        __lasx_xvstelm_h(val, des, 0, 0);
        return des[0];
    }
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
        val = _v256_setr_h(v0, v1, v2, v3, v4, v5, v6, v7,
            v8, v9, v10, v11, v12, v13, v14, v15);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_int16x16() {}

    short get0() const {
        short des[1] = {0};
        __lasx_xvstelm_h(val, des, 0, 0);
        return des[0];
    }
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
        val = _v256_setr_w((unsigned)v0, (unsigned)v1, (unsigned)v2,
            (unsigned)v3, (unsigned)v4, (unsigned)v5, (unsigned)v6, (unsigned)v7);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_uint32x8() {}

    unsigned get0() const { return __lasx_xvpickve2gr_wu(val, 0); }
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
        val = _v256_setr_w(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_int32x8() {}

    int get0() const { return __lasx_xvpickve2gr_w(val, 0); }
};

struct v_float32x8
{
    typedef float lane_type;
    enum { nlanes = 8 };
    __m256 val;

    explicit v_float32x8(__m256 v) : val(v) {}
    explicit v_float32x8(__m256i v) { val = *((__m256*)&v); }
    v_float32x8(float v0, float v1, float v2, float v3,
                float v4, float v5, float v6, float v7)
    {
        val = _v256_setr_ps(v0, v1, v2, v3, v4, v5, v6, v7);
    }
    /* coverity[uninit_ctor]: suppress warning */
    v_float32x8() {}

    float get0() const {
        float des[1] = {0};
        __lasx_xvstelm_w(*((__m256i*)&val), des, 0, 0);
        return des[0];
    }

    int get0toint() const {
        int des[1] = {0};
        __lasx_xvstelm_w(*((__m256i*)&val), des, 0, 0);
        return des[0];
    }
};

struct v_uint64x4
{
    typedef uint64 lane_type;
    enum { nlanes = 4 };
    __m256i val;

    explicit v_uint64x4(__m256i v) : val(v) {}
    v_uint64x4(uint64 v0, uint64 v1, uint64 v2, uint64 v3)
    { val = _v256_setr_d((int64)v0, (int64)v1, (int64)v2, (int64)v3); }
    /* coverity[uninit_ctor]: suppress warning */
    v_uint64x4() {}

    uint64 get0() const
    {
        return __lasx_xvpickve2gr_du(val, 0);
    }
};

struct v_int64x4
{
    typedef int64 lane_type;
    enum { nlanes = 4 };
    __m256i val;

    explicit v_int64x4(__m256i v) : val(v) {}
    v_int64x4(int64 v0, int64 v1, int64 v2, int64 v3)
    { val = _v256_setr_d(v0, v1, v2, v3); }
    /* coverity[uninit_ctor]: suppress warning */
    v_int64x4() {}

    int64 get0() const
    {
        return __lasx_xvpickve2gr_d(val, 0);
    }
};

struct v_float64x4
{
    typedef double lane_type;
    enum { nlanes = 4 };
    __m256d val;

    explicit v_float64x4(__m256d v) : val(v) {}
    explicit v_float64x4(__m256i v) { val = *((__m256d*)&v); }
    v_float64x4(double v0, double v1, double v2, double v3)
    { val = _v256_setr_pd(v0, v1, v2, v3); }
    /* coverity[uninit_ctor]: suppress warning */
    v_float64x4() {}

    double get0() const {
        double des[1] = {0};
        __lasx_xvstelm_d(*((__m256i*)&val), des, 0, 0);
        return des[0];
    }

    int64 get0toint64() const {
        int64 des[1] = {0};
        __lasx_xvstelm_d(*((__m256i*)&val), des, 0, 0);
        return des[0];
    }
};

//////////////// Load and store operations ///////////////

#define OPENCV_HAL_IMPL_LASX_LOADSTORE(_Tpvec, _Tp)                   \
    inline _Tpvec v256_load(const _Tp* ptr)                           \
    { return _Tpvec(__lasx_xvld(ptr, 0)); }                           \
    inline _Tpvec v256_load_aligned(const _Tp* ptr)                   \
    { return _Tpvec(__lasx_xvld(ptr, 0)); }                           \
    inline _Tpvec v256_load_low(const _Tp* ptr)                       \
    {                                                                 \
        __m128i v128 = __lsx_vld(ptr, 0);                             \
        return _Tpvec(*((__m256i*)&v128));                            \
    }                                                                 \
    inline _Tpvec v256_load_halves(const _Tp* ptr0, const _Tp* ptr1)  \
    {                                                                 \
        __m128i vlo = __lsx_vld(ptr0, 0);                             \
        __m128i vhi = __lsx_vld(ptr1, 0);                             \
        return _Tpvec(_v256_combine(vlo, vhi));                       \
    }                                                                 \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                    \
    { __lasx_xvst(a.val, ptr, 0); }                                   \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)            \
    { __lasx_xvst(a.val, ptr, 0); }                                   \
    inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a)    \
    { __lasx_xvst(a.val, ptr, 0); }                                   \
    inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
    { \
        if( mode == hal::STORE_UNALIGNED ) \
            __lasx_xvst(a.val, ptr, 0); \
        else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
            __lasx_xvst(a.val, ptr, 0); \
        else \
            __lasx_xvst(a.val, ptr, 0); \
    } \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                \
    { __lsx_vst(_v256_extract_low(a.val), ptr, 0); }                  \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)               \
    { __lsx_vst(_v256_extract_high(a.val), ptr, 0); }

OPENCV_HAL_IMPL_LASX_LOADSTORE(v_uint8x32,  uchar)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_int8x32,   schar)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_uint16x16, ushort)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_int16x16,  short)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_uint32x8,  unsigned)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_int32x8,   int)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_uint64x4,  uint64)
OPENCV_HAL_IMPL_LASX_LOADSTORE(v_int64x4,   int64)


#define OPENCV_HAL_IMPL_LASX_LOADSTORE_FLT(_Tpvec, _Tp, halfreg)          \
    inline _Tpvec v256_load(const _Tp* ptr)                               \
    { return _Tpvec(__lasx_xvld(ptr, 0)); }                               \
    inline _Tpvec v256_load_aligned(const _Tp* ptr)                       \
    { return _Tpvec(__lasx_xvld(ptr, 0)); }                               \
    inline _Tpvec v256_load_low(const _Tp* ptr)                           \
    {                                                                     \
        __m128i v128 = __lsx_vld(ptr, 0);                                 \
        return _Tpvec(*((__m256i*)&v128));                                \
    }                                                                     \
    inline _Tpvec v256_load_halves(const _Tp* ptr0, const _Tp* ptr1)      \
    {                                                                     \
        halfreg vlo = __lsx_vld(ptr0, 0);                                 \
        halfreg vhi = __lsx_vld(ptr1, 0);                                 \
        return _Tpvec(_v256_combine(vlo, vhi));                           \
    }                                                                     \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                        \
    { __lasx_xvst(a.val, ptr, 0); }                                       \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)                \
    { __lasx_xvst(a.val, ptr, 0); }                                       \
    inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a)        \
    { __lasx_xvst(a.val, ptr, 0); }                                       \
    inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
    { \
        if( mode == hal::STORE_UNALIGNED ) \
            __lasx_xvst(a.val, ptr, 0); \
        else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
            __lasx_xvst(a.val, ptr, 0); \
        else \
            __lasx_xvst(a.val, ptr, 0); \
    } \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                    \
    { __lsx_vst(_v256_extract_low(a.val), ptr, 0); }                      \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)                   \
    { __lsx_vst(_v256_extract_high(a.val), ptr, 0); }

OPENCV_HAL_IMPL_LASX_LOADSTORE_FLT(v_float32x8, float, __m128i)
OPENCV_HAL_IMPL_LASX_LOADSTORE_FLT(v_float64x4, double, __m128i)


inline __m256i _lasx_256_castps_si256(const __m256& v)
{ return __m256i(v); }

inline __m256i _lasx_256_castpd_si256(const __m256d& v)
{ return __m256i(v); }

#define OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, _Tpvecf, suffix, cast) \
    inline _Tpvec v_reinterpret_as_##suffix(const _Tpvecf& a)   \
    { return _Tpvec(cast(a.val)); }

#define OPENCV_HAL_IMPL_LASX_INIT(_Tpvec, _Tp, suffix, ssuffix, ctype_s)          \
    inline _Tpvec v256_setzero_##suffix()                                         \
    { return _Tpvec(__lasx_xvreplgr2vr_d(0)); }                                   \
    inline _Tpvec v256_setall_##suffix(_Tp v)                                     \
    { return _Tpvec(__lasx_xvreplgr2vr_##ssuffix((ctype_s)v)); }                  \
    template <> inline _Tpvec v_setzero_()                                        \
    { return v256_setzero_##suffix(); }                                           \
    template <> inline _Tpvec v_setall_(_Tp v)                                    \
    { return v256_setall_##suffix(v); }                                           \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint8x32,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int8x32,   suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint16x16, suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int16x16,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint32x8,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int32x8,   suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint64x4,  suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int64x4,   suffix, OPENCV_HAL_NOP)        \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_float32x8, suffix, _lasx_256_castps_si256) \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_float64x4, suffix, _lasx_256_castpd_si256)

OPENCV_HAL_IMPL_LASX_INIT(v_uint8x32,  uchar,    u8,  b,   int)
OPENCV_HAL_IMPL_LASX_INIT(v_int8x32,   schar,    s8,  b,   int)
OPENCV_HAL_IMPL_LASX_INIT(v_uint16x16, ushort,   u16, h,  int)
OPENCV_HAL_IMPL_LASX_INIT(v_int16x16,  short,    s16, h,  int)
OPENCV_HAL_IMPL_LASX_INIT(v_uint32x8,  unsigned, u32, w,  int)
OPENCV_HAL_IMPL_LASX_INIT(v_int32x8,   int,      s32, w,  int)
OPENCV_HAL_IMPL_LASX_INIT(v_uint64x4,  uint64,   u64, d, long int)
OPENCV_HAL_IMPL_LASX_INIT(v_int64x4,   int64,    s64, d, long int)


inline __m256 _lasx_256_castsi256_ps(const __m256i &v)
{ return __m256(v); }

inline __m256d _lasx_256_castsi256_pd(const __m256i &v)
{ return __m256d(v); }

#define OPENCV_HAL_IMPL_LASX_INIT_FLT(_Tpvec, _Tp, suffix, zsuffix, cast) \
    inline _Tpvec v256_setzero_##suffix()                                 \
    { return _Tpvec(__lasx_xvreplgr2vr_d(0)); }                           \
    inline _Tpvec v256_setall_##suffix(_Tp v)                             \
    { return _Tpvec(_v256_setall_##zsuffix(v)); }                         \
    template <> inline _Tpvec v_setzero_()                                \
    { return v256_setzero_##suffix(); }                                   \
    template <> inline _Tpvec v_setall_(_Tp v)                            \
    { return v256_setall_##suffix(v); }                                   \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint8x32,  suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int8x32,   suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint16x16, suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int16x16,  suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint32x8,  suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int32x8,   suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_uint64x4,  suffix, cast)          \
    OPENCV_HAL_IMPL_LASX_CAST(_Tpvec, v_int64x4,   suffix, cast)

OPENCV_HAL_IMPL_LASX_INIT_FLT(v_float32x8, float,  f32, ps, _lasx_256_castsi256_ps)
OPENCV_HAL_IMPL_LASX_INIT_FLT(v_float64x4, double, f64, pd, _lasx_256_castsi256_pd)

inline v_float32x8 v_reinterpret_as_f32(const v_float32x8& a)
{ return a; }
inline v_float32x8 v_reinterpret_as_f32(const v_float64x4& a)
{ return v_float32x8(_lasx_256_castps_si256(__m256(a.val))); }

inline v_float64x4 v_reinterpret_as_f64(const v_float64x4& a)
{ return a; }
inline v_float64x4 v_reinterpret_as_f64(const v_float32x8& a)
{ return v_float64x4(_lasx_256_castpd_si256(__m256d(a.val))); }


//////////////// Variant Value reordering ///////////////

// unpacks
#define OPENCV_HAL_IMPL_LASX_UNPACK(_Tpvec, suffix)                 \
    inline _Tpvec v256_unpacklo(const _Tpvec& a, const _Tpvec& b)   \
    { return _Tpvec(__lasx_xvilvl_##suffix(__m256i(b.val), __m256i(a.val))); }        \
    inline _Tpvec v256_unpackhi(const _Tpvec& a, const _Tpvec& b)   \
    { return _Tpvec(__lasx_xvilvh_##suffix(__m256i(b.val), __m256i(a.val))); }

OPENCV_HAL_IMPL_LASX_UNPACK(v_uint8x32,  b)
OPENCV_HAL_IMPL_LASX_UNPACK(v_int8x32,   b)
OPENCV_HAL_IMPL_LASX_UNPACK(v_uint16x16, h)
OPENCV_HAL_IMPL_LASX_UNPACK(v_int16x16,  h)
OPENCV_HAL_IMPL_LASX_UNPACK(v_uint32x8,  w)
OPENCV_HAL_IMPL_LASX_UNPACK(v_int32x8,   w)
OPENCV_HAL_IMPL_LASX_UNPACK(v_uint64x4,  d)
OPENCV_HAL_IMPL_LASX_UNPACK(v_int64x4,   d)
OPENCV_HAL_IMPL_LASX_UNPACK(v_float32x8, w)
OPENCV_HAL_IMPL_LASX_UNPACK(v_float64x4, d)


// shuffle
// todo: emulate 64bit
#define OPENCV_HAL_IMPL_LASX_SHUFFLE(_Tpvec, intrin)  \
    template<int m>                                  \
    inline _Tpvec v256_shuffle(const _Tpvec& a)      \
    { return _Tpvec(__lasx_xvshuf4i_##intrin(a.val, m)); }

OPENCV_HAL_IMPL_LASX_SHUFFLE(v_uint32x8,  w)
OPENCV_HAL_IMPL_LASX_SHUFFLE(v_int32x8,   w)

template<int m>
inline v_float32x8 v256_shuffle(const v_float32x8 &a)
{ return v_float32x8(__lasx_xvshuf4i_w(*((__m256i*)&a.val), m)); }

template<int m>
inline v_float64x4 v256_shuffle(const v_float64x4 &a)
{
    const int m1 = m & 0b1;
    const int m2 = m & 0b10;
    const int m3 = m & 0b100;
    const int m4 = m & 0b1000;
    const int m5 = m2 << 1;
    const int m6 = m3 << 2;
    const int m7 = m4 << 3;
    const int m8 = m1 & m5 & m6 & m7;

    return v_float64x4(__lasx_xvshuf4i_d(*((__m256i*)&a.val), *((__m256i*)&a.val), m8));
}

template<typename _Tpvec>
inline void v256_zip(const _Tpvec& a, const _Tpvec& b, _Tpvec& ab0, _Tpvec& ab1)
{
    ab0 = v256_unpacklo(a, b);
    ab1 = v256_unpackhi(a, b);
}

template<typename _Tpvec>
inline _Tpvec v256_combine_diagonal(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(__lasx_xvpermi_q(a.val, b.val, 0x12)); }

inline v_float32x8 v256_combine_diagonal(const v_float32x8& a, const v_float32x8& b)
{ return v_float32x8(__lasx_xvpermi_q(a.val, b.val, 0x12)); }

inline v_float64x4 v256_combine_diagonal(const v_float64x4& a, const v_float64x4& b)
{ return v_float64x4(__lasx_xvpermi_q(a.val, b.val, 0x12)); }

template<typename _Tpvec>
inline _Tpvec v256_alignr_128(const _Tpvec& a, const _Tpvec& b)
{ return v256_permute2x128<0x03>(a, b); }

inline __m256i _v256_alignr_b(const __m256i &a, const __m256i &b, const int imm)
{
    if (imm == 8) {
        return __lasx_xvshuf4i_d(b, a, 0x9); // b.d1 a.d0 b.d3 a.d2
    } else {
        __m256i byteIndex = _v256_setr_b(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
                                         0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
        return __lasx_xvshuf_b(a, b, __lasx_xvadd_b(__lasx_xvreplgr2vr_b(imm), byteIndex));
    }
}

template<typename _Tpvec>
inline _Tpvec v256_alignr_64(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(_v256_alignr_b(a.val, b.val, 8)); }
inline v_float64x4 v256_alignr_64(const v_float64x4& a, const v_float64x4& b)
{ return v_float64x4(__lasx_xvshuf4i_d(b.val, a.val, 0x9)); } // b.d1 a.d0 b.d3 a.d2
// todo: emulate float32

template<typename _Tpvec>
inline _Tpvec v256_swap_halves(const _Tpvec& a)
{ return v256_permute2x128<1>(a, a); }

template<typename _Tpvec>
inline _Tpvec v256_reverse_64(const _Tpvec& a)
{ return v256_permute4x64<0x1b>(a); }


// ZIP
#define OPENCV_HAL_IMPL_LASX_ZIP(_Tpvec)                             \
    inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)    \
    { return v256_permute2x128<0x02>(a, b); }                        \
    inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)   \
    { return v256_permute2x128<0x13>(a, b); }                        \
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

OPENCV_HAL_IMPL_LASX_ZIP(v_uint8x32)
OPENCV_HAL_IMPL_LASX_ZIP(v_int8x32)
OPENCV_HAL_IMPL_LASX_ZIP(v_uint16x16)
OPENCV_HAL_IMPL_LASX_ZIP(v_int16x16)
OPENCV_HAL_IMPL_LASX_ZIP(v_uint32x8)
OPENCV_HAL_IMPL_LASX_ZIP(v_int32x8)
OPENCV_HAL_IMPL_LASX_ZIP(v_uint64x4)
OPENCV_HAL_IMPL_LASX_ZIP(v_int64x4)
OPENCV_HAL_IMPL_LASX_ZIP(v_float32x8)
OPENCV_HAL_IMPL_LASX_ZIP(v_float64x4)

////////// Arithmetic, bitwise and comparison operations /////////

/** Arithmetics **/
#define OPENCV_HAL_IMPL_LASX_BIN_OP(bin_op, _Tpvec, intrin)           \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)            \
    { return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_uint8x32,  __lasx_xvsadd_bu)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_uint8x32,  __lasx_xvssub_bu)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_int8x32,   __lasx_xvsadd_b)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_int8x32,   __lasx_xvssub_b)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_uint16x16, __lasx_xvsadd_hu)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_uint16x16, __lasx_xvssub_hu)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_int16x16,  __lasx_xvsadd_h)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_int16x16,  __lasx_xvssub_h)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_uint32x8,  __lasx_xvadd_w)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_uint32x8,  __lasx_xvsub_w)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_mul, v_uint32x8,  __lasx_xvmul_w)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_int32x8,   __lasx_xvadd_w)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_int32x8,   __lasx_xvsub_w)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_mul, v_int32x8,   __lasx_xvmul_w)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_uint64x4,  __lasx_xvadd_d)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_uint64x4,  __lasx_xvsub_d)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_int64x4,   __lasx_xvadd_d)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_int64x4,   __lasx_xvsub_d)

OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_float32x8, __lasx_xvfadd_s)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_float32x8, __lasx_xvfsub_s)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_mul, v_float32x8, __lasx_xvfmul_s)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_div, v_float32x8, __lasx_xvfdiv_s)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_add, v_float64x4, __lasx_xvfadd_d)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_sub, v_float64x4, __lasx_xvfsub_d)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_mul, v_float64x4, __lasx_xvfmul_d)
OPENCV_HAL_IMPL_LASX_BIN_OP(v_div, v_float64x4, __lasx_xvfdiv_d)

// saturating multiply 8-bit, 16-bit
inline v_uint8x32 v_mul(const v_uint8x32& a, const v_uint8x32& b)
{
    v_uint16x16 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_int8x32 v_mul(const v_int8x32& a, const v_int8x32& b)
{
    v_int16x16 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_uint16x16 v_mul(const v_uint16x16& a, const v_uint16x16& b)
{
    __m256i pl = __lasx_xvmul_h(a.val, b.val);
    __m256i ph = __lasx_xvmuh_hu(a.val, b.val);
    __m256i p0 = __lasx_xvilvl_h(ph, pl);
    __m256i p1 = __lasx_xvilvh_h(ph, pl);
    return v_uint16x16(_v256_packs_epu32(p0, p1));
}
inline v_int16x16 v_mul(const v_int16x16& a, const v_int16x16& b)
{
    __m256i pl = __lasx_xvmul_h(a.val, b.val);
    __m256i ph = __lasx_xvmuh_h(a.val, b.val);
    __m256i p0 = __lasx_xvilvl_h(ph, pl);
    __m256i p1 = __lasx_xvilvh_h(ph, pl);
    return v_int16x16(_lasx_packs_w(p0, p1));
}

/** Non-saturating arithmetics **/

#define OPENCV_HAL_IMPL_LASX_BIN_FUNC(func, _Tpvec, intrin) \
    inline _Tpvec func(const _Tpvec& a, const _Tpvec& b)    \
    { return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_add_wrap, v_uint8x32,  __lasx_xvadd_b)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_add_wrap, v_int8x32,   __lasx_xvadd_b)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_add_wrap, v_uint16x16, __lasx_xvadd_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_add_wrap, v_int16x16,  __lasx_xvadd_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_sub_wrap, v_uint8x32,  __lasx_xvsub_b)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_sub_wrap, v_int8x32,   __lasx_xvsub_b)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_sub_wrap, v_uint16x16, __lasx_xvsub_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_sub_wrap, v_int16x16,  __lasx_xvsub_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_mul_wrap, v_uint16x16, __lasx_xvmul_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_mul_wrap, v_int16x16,  __lasx_xvmul_h)

inline v_uint8x32 v_mul_wrap(const v_uint8x32& a, const v_uint8x32& b)
{
    __m256i p0 = __lasx_xvmulwev_h_bu(a.val, b.val);
    __m256i p1 = __lasx_xvmulwod_h_bu(a.val, b.val);
    return v_uint8x32(__lasx_xvpackev_b(p1, p0));
}

inline v_int8x32 v_mul_wrap(const v_int8x32& a, const v_int8x32& b)
{
    return v_reinterpret_as_s8(v_mul_wrap(v_reinterpret_as_u8(a), v_reinterpret_as_u8(b)));
}

//  Multiply and expand
inline void v_mul_expand(const v_uint8x32& a, const v_uint8x32& b,
                         v_uint16x16& c, v_uint16x16& d)
{
    v_uint16x16 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int8x32& a, const v_int8x32& b,
                         v_int16x16& c, v_int16x16& d)
{
    v_int16x16 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int16x16& a, const v_int16x16& b,
                         v_int32x8& c, v_int32x8& d)
{
    v_int16x16 vhi = v_int16x16(__lasx_xvmuh_h(a.val, b.val));

    v_int16x16 v0, v1;
    v_zip(v_mul_wrap(a, b), vhi, v0, v1);

    c = v_reinterpret_as_s32(v0);
    d = v_reinterpret_as_s32(v1);
}

inline void v_mul_expand(const v_uint16x16& a, const v_uint16x16& b,
                         v_uint32x8& c, v_uint32x8& d)
{
    v_uint16x16 vhi = v_uint16x16(__lasx_xvmuh_hu(a.val, b.val));

    v_uint16x16 v0, v1;
    v_zip(v_mul_wrap(a, b), vhi, v0, v1);

    c = v_reinterpret_as_u32(v0);
    d = v_reinterpret_as_u32(v1);
}

inline void v_mul_expand(const v_uint32x8& a, const v_uint32x8& b,
                         v_uint64x4& c, v_uint64x4& d)
{
    __m256i v0 = __lasx_xvmulwev_d_wu(a.val, b.val);
    __m256i v1 = __lasx_xvmulwod_d_wu(a.val, b.val);
    v_zip(v_uint64x4(v0), v_uint64x4(v1), c, d);
}

inline v_int16x16 v_mul_hi(const v_int16x16& a, const v_int16x16& b) { return v_int16x16(__lasx_xvmuh_h(a.val, b.val)); }
inline v_uint16x16 v_mul_hi(const v_uint16x16& a, const v_uint16x16& b) { return v_uint16x16(__lasx_xvmuh_hu(a.val, b.val)); }

/** Bitwise shifts **/
#define OPENCV_HAL_IMPL_LASX_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, srai)                             \
    inline _Tpuvec v_shl(const _Tpuvec& a, int imm)                                               \
    { return _Tpuvec(__lasx_xvsll_##suffix(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }           \
    inline _Tpsvec v_shl(const _Tpsvec& a, int imm)                                               \
    { return _Tpsvec(__lasx_xvsll_##suffix(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }           \
    inline _Tpuvec V_shr(const _Tpuvec& a, int imm)                                               \
    { return _Tpuvec(__lasx_xvsrl_##suffix(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }           \
    inline _Tpsvec v_shr(const _Tpsvec& a, int imm)                                               \
    { return _Tpsvec(srai(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }                            \
    template<int imm>                                                                             \
    inline _Tpuvec v_shl(const _Tpuvec& a)                                                        \
    { return _Tpuvec(__lasx_xvsll_##suffix(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }           \
    template<int imm>                                                                             \
    inline _Tpsvec v_shl(const _Tpsvec& a)                                                        \
    { return _Tpsvec(__lasx_xvsll_##suffix(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }           \
    template<int imm>                                                                             \
    inline _Tpuvec v_shr(const _Tpuvec& a)                                                        \
    { return _Tpuvec(__lasx_xvsrl_##suffix(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }           \
    template<int imm>                                                                             \
    inline _Tpsvec v_shr(const _Tpsvec& a)                                                        \
    { return _Tpsvec(srai(a.val, __lasx_xvreplgr2vr_##suffix(imm))); }

OPENCV_HAL_IMPL_LASX_SHIFT_OP(v_uint16x16, v_int16x16, h, __lasx_xvsra_h)
OPENCV_HAL_IMPL_LASX_SHIFT_OP(v_uint32x8,  v_int32x8,  w, __lasx_xvsra_w)
OPENCV_HAL_IMPL_LASX_SHIFT_OP(v_uint64x4,  v_int64x4,  d, __lasx_xvsra_d)


/** Bitwise logic **/
#define OPENCV_HAL_IMPL_LASX_LOGIC_OP(_Tpvec, suffix, not_const)    \
    OPENCV_HAL_IMPL_LASX_BIN_OP(v_and, _Tpvec, __lasx_xvand_##suffix)  \
    OPENCV_HAL_IMPL_LASX_BIN_OP(v_or, _Tpvec, __lasx_xvor_##suffix)    \
    OPENCV_HAL_IMPL_LASX_BIN_OP(v_xor, _Tpvec, __lasx_xvxor_##suffix)  \
    inline _Tpvec v_not(const _Tpvec& a)                               \
    { return _Tpvec(__lasx_xvnori_b(a.val, 0)); }

OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_uint8x32,   v, __lasx_xvreplgr2vr_w(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_int8x32,    v, __lasx_xvreplgr2vr_w(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_uint16x16,  v, __lasx_xvreplgr2vr_w(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_int16x16,   v, __lasx_xvreplgr2vr_w(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_uint32x8,   v, __lasx_xvreplgr2vr_w(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_int32x8,    v, __lasx_xvreplgr2vr_w(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_uint64x4,   v, __lasx_xvreplgr2vr_d(-1))
OPENCV_HAL_IMPL_LASX_LOGIC_OP(v_int64x4,    v, __lasx_xvreplgr2vr_d(-1))

#define OPENCV_HAL_IMPL_LASX_FLOAT_BIN_OP(bin_op, _Tpvec, intrin, cast)                         \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)                                      \
    { return _Tpvec(intrin(*((__m256i*)(&a.val)), *((__m256i*)(&b.val)))); }

#define OPENCV_HAL_IMPL_LASX_FLOAT_LOGIC_OP(_Tpvec, suffix, not_const, cast)       \
    OPENCV_HAL_IMPL_LASX_FLOAT_BIN_OP(v_and, _Tpvec, __lasx_xvand_##suffix, cast)  \
    OPENCV_HAL_IMPL_LASX_FLOAT_BIN_OP(v_or, _Tpvec, __lasx_xvor_##suffix, cast)    \
    OPENCV_HAL_IMPL_LASX_FLOAT_BIN_OP(v_xor, _Tpvec, __lasx_xvxor_##suffix, cast)  \
    inline _Tpvec v_not(const _Tpvec& a)                                           \
    { return _Tpvec(__lasx_xvxor_##suffix(*((__m256i*)(&a.val)), not_const)); }

OPENCV_HAL_IMPL_LASX_FLOAT_LOGIC_OP(v_float32x8,  v, __lasx_xvreplgr2vr_w(-1), _lasx_256_castsi256_ps)
OPENCV_HAL_IMPL_LASX_FLOAT_LOGIC_OP(v_float64x4,  v, __lasx_xvreplgr2vr_d(-1), _lasx_256_castsi256_pd)

/** Select **/
#define OPENCV_HAL_IMPL_LASX_SELECT(_Tpvec)                                      \
    inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
    { return _Tpvec(__lasx_xvbitsel_v(b.val, a.val, mask.val)); }

OPENCV_HAL_IMPL_LASX_SELECT(v_uint8x32)
OPENCV_HAL_IMPL_LASX_SELECT(v_int8x32)
OPENCV_HAL_IMPL_LASX_SELECT(v_uint16x16)
OPENCV_HAL_IMPL_LASX_SELECT(v_int16x16)
OPENCV_HAL_IMPL_LASX_SELECT(v_uint32x8)
OPENCV_HAL_IMPL_LASX_SELECT(v_int32x8)

inline v_float32x8 v_select(const v_float32x8 &mask, const v_float32x8 &a, const v_float32x8 &b)
{ return v_float32x8(__lasx_xvbitsel_v(*((__m256i*)&b.val), *((__m256i*)&a.val), *((__m256i*)&mask.val))); }

inline v_float64x4 v_select(const v_float64x4 &mask, const v_float64x4 &a, const v_float64x4 &b)
{ return v_float64x4(__lasx_xvbitsel_v(*((__m256i*)&b.val), *((__m256i*)&a.val), *((__m256i*)&mask.val))); }

/** Comparison **/
#define OPENCV_HAL_IMPL_LASX_CMP_OP_OV(_Tpvec)                     \
    inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b)           \
    { return v_not(v_eq(a, b)); }                                  \
    inline _Tpvec v_lt(const _Tpvec& a, const _Tpvec& b)           \
    { return v_gt(b, a); }                                         \
    inline _Tpvec v_ge(const _Tpvec& a, const _Tpvec& b)           \
    { return v_not(v_lt(a, b)); }                                  \
    inline _Tpvec v_le(const _Tpvec& a, const _Tpvec& b)           \
    { return v_ge(b, a); }

#define OPENCV_HAL_IMPL_LASX_CMP_OP_INT(_Tpuvec, _Tpsvec, suffix, usuffix)   \
    inline _Tpuvec v_eq(const _Tpuvec& a, const _Tpuvec& b)                  \
    { return _Tpuvec(__lasx_xvseq_##suffix(a.val, b.val)); }                 \
    inline _Tpuvec v_gt(const _Tpuvec& a, const _Tpuvec& b)                  \
    {                                                                        \
        return _Tpuvec(__lasx_xvslt_##usuffix(b.val, a.val));                \
    }                                                                        \
    inline _Tpsvec v_eq(const _Tpsvec& a, const _Tpsvec& b)                  \
    { return _Tpsvec(__lasx_xvseq_##suffix(a.val, b.val)); }                 \
    inline _Tpsvec v_gt(const _Tpsvec& a, const _Tpsvec& b)                  \
    { return _Tpsvec(__lasx_xvslt_##suffix(b.val, a.val)); }                 \
    OPENCV_HAL_IMPL_LASX_CMP_OP_OV(_Tpuvec)                                  \
    OPENCV_HAL_IMPL_LASX_CMP_OP_OV(_Tpsvec)

OPENCV_HAL_IMPL_LASX_CMP_OP_INT(v_uint8x32,  v_int8x32,  b, bu)
OPENCV_HAL_IMPL_LASX_CMP_OP_INT(v_uint16x16, v_int16x16, h, hu)
OPENCV_HAL_IMPL_LASX_CMP_OP_INT(v_uint32x8,  v_int32x8,  w, wu)

#define OPENCV_HAL_IMPL_LASX_CMP_OP_64BIT(_Tpvec, suffix)         \
    inline _Tpvec v_eq(const _Tpvec& a, const _Tpvec& b)          \
    { return _Tpvec(__lasx_xvseq_##suffix(a.val, b.val)); }       \
    inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b)          \
    { return v_not(v_eq(a, b)); }

OPENCV_HAL_IMPL_LASX_CMP_OP_64BIT(v_uint64x4, d)
OPENCV_HAL_IMPL_LASX_CMP_OP_64BIT(v_int64x4, d)

#define OPENCV_HAL_IMPL_LASX_CMP_FLT(bin_op, suffix, _Tpvec, ssuffix)    \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)               \
    { return _Tpvec(__lasx_##suffix##_##ssuffix(a.val, b.val)); }

#define OPENCV_HAL_IMPL_LASX_CMP_OP_FLT(_Tpvec, ssuffix)              \
    OPENCV_HAL_IMPL_LASX_CMP_FLT(v_eq, xvfcmp_ceq, _Tpvec, ssuffix)   \
    OPENCV_HAL_IMPL_LASX_CMP_FLT(v_ne, xvfcmp_cne, _Tpvec, ssuffix)   \
    OPENCV_HAL_IMPL_LASX_CMP_FLT(v_lt,  xvfcmp_clt, _Tpvec, ssuffix)  \
    OPENCV_HAL_IMPL_LASX_CMP_FLT(v_le, xvfcmp_cle, _Tpvec, ssuffix)

OPENCV_HAL_IMPL_LASX_CMP_OP_FLT(v_float32x8, s)
OPENCV_HAL_IMPL_LASX_CMP_OP_FLT(v_float64x4, d)

inline v_float32x8 v_gt(const v_float32x8 &a, const v_float32x8 &b)
{ return v_float32x8(__lasx_xvfcmp_clt_s(b.val, a.val)); }

inline v_float32x8 v_ge(const v_float32x8 &a, const v_float32x8 &b)
{ return v_float32x8(__lasx_xvfcmp_cle_s(b.val, a.val)); }

inline v_float64x4 v_gt(const v_float64x4 &a, const v_float64x4 &b)
{ return v_float64x4(__lasx_xvfcmp_clt_d(b.val, a.val)); }

inline v_float64x4 v_ge(const v_float64x4 &a, const v_float64x4 &b)
{ return v_float64x4(__lasx_xvfcmp_cle_d(b.val, a.val)); }

inline v_float32x8 v_not_nan(const v_float32x8& a)
{ return v_float32x8(__lasx_xvfcmp_cor_s(a.val, a.val)); }
inline v_float64x4 v_not_nan(const v_float64x4& a)
{ return v_float64x4(__lasx_xvfcmp_cor_d(a.val, a.val)); }

/** min/max **/
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_uint8x32,  __lasx_xvmin_bu)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_uint8x32,  __lasx_xvmax_bu)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_int8x32,   __lasx_xvmin_b)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_int8x32,   __lasx_xvmax_b)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_uint16x16, __lasx_xvmin_hu)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_uint16x16, __lasx_xvmax_hu)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_int16x16,  __lasx_xvmin_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_int16x16,  __lasx_xvmax_h)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_uint32x8,  __lasx_xvmin_wu)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_uint32x8,  __lasx_xvmax_wu)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_int32x8,   __lasx_xvmin_w)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_int32x8,   __lasx_xvmax_w)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_float32x8, __lasx_xvfmin_s)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_float32x8, __lasx_xvfmax_s)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_min, v_float64x4, __lasx_xvfmin_d)
OPENCV_HAL_IMPL_LASX_BIN_FUNC(v_max, v_float64x4, __lasx_xvfmax_d)

/** Rotate **/
template<int imm>
inline v_uint8x32 v_rotate_left(const v_uint8x32& a, const v_uint8x32& b)
{
    enum {IMM_R = (16 - imm) & 0xFF};
    enum {IMM_R2 = (32 - imm) & 0xFF};

    if (imm == 0)  return a;
    if (imm == 32) return b;
    if (imm > 32)  return v_uint8x32();

    __m256i swap = _v256_permute2x128<0x21>(a.val, b.val);
    if (imm == 16) return v_uint8x32(swap);
    if (imm < 16)  return v_uint8x32(_v256_alignr_b(a.val, swap, IMM_R));
    return v_uint8x32(_v256_alignr_b(swap, b.val, IMM_R2)); // imm < 32
}

template<int imm>
inline v_uint8x32 v_rotate_right(const v_uint8x32& a, const v_uint8x32& b)
{
    enum {IMM_L = (imm - 16) & 0xFF};

    if (imm == 0)  return a;
    if (imm == 32) return b;
    if (imm > 32)  return v_uint8x32();

    __m256i swap = _v256_permute2x128<0x03>(a.val, b.val);
    if (imm == 16) return v_uint8x32(swap);
    if (imm < 16)  return v_uint8x32(_v256_alignr_b(swap, a.val, imm));
    return v_uint8x32(_v256_alignr_b(b.val, swap, IMM_L));
}

template<int imm>
inline v_uint8x32 v_rotate_left(const v_uint8x32& a)
{
    enum {IMM_L = ((imm - 16) & 0xFF) > 31 ? 31 : ((imm - 16) & 0xFF)};
    enum {IMM_R = (16 - imm) & 0xFF};

    if (imm == 0) return a;
    if (imm > 32) return v_uint8x32();

    // ESAC control[3] ? [127:0] = 0
    __m256i vzero = __lasx_xvreplgr2vr_w(0);
    __m256i swapz = __lasx_xvpermi_q(a.val, vzero, 0x20);;
    if (imm == 16) return v_uint8x32(swapz);
    if (imm < 16)  return v_uint8x32(_v256_alignr_b(a.val, swapz, IMM_R));
    return v_uint8x32(__lasx_xvbsll_v(swapz, IMM_L));
}

template<int imm>
inline v_uint8x32 v_rotate_right(const v_uint8x32& a)
{
    enum {IMM_L = ((imm - 16) & 0xFF) > 31 ? 31 : ((imm - 16) & 0xFF)};

    if (imm == 0) return a;
    if (imm > 32) return v_uint8x32();

    // ESAC control[3] ? [127:0] = 0
    __m256i vzero = __lasx_xvreplgr2vr_w(0);
    __m256i swapz = __lasx_xvpermi_q(vzero, a.val, 0x21);;
    if (imm == 16) return v_uint8x32(swapz);
    if (imm < 16)  return v_uint8x32(_v256_alignr_b(swapz, a.val, imm));
    return v_uint8x32(__lasx_xvbsrl_v(swapz, IMM_L));
}

#define OPENCV_HAL_IMPL_LASX_ROTATE_CAST(intrin, _Tpvec, cast)    \
    template<int imm>                                             \
    inline _Tpvec intrin(const _Tpvec& a, const _Tpvec& b)        \
    {                                                             \
        enum {IMMxW = imm * sizeof(typename _Tpvec::lane_type)};  \
        v_uint8x32 ret = intrin<IMMxW>(v_reinterpret_as_u8(a),    \
                                       v_reinterpret_as_u8(b));   \
        return _Tpvec(cast(ret.val));                             \
    }                                                             \
    template<int imm>                                             \
    inline _Tpvec intrin(const _Tpvec& a)                         \
    {                                                             \
        enum {IMMxW = imm * sizeof(typename _Tpvec::lane_type)};  \
        v_uint8x32 ret = intrin<IMMxW>(v_reinterpret_as_u8(a));   \
        return _Tpvec(cast(ret.val));                             \
    }

#define OPENCV_HAL_IMPL_LASX_ROTATE(_Tpvec)                                  \
    OPENCV_HAL_IMPL_LASX_ROTATE_CAST(v_rotate_left,  _Tpvec, OPENCV_HAL_NOP) \
    OPENCV_HAL_IMPL_LASX_ROTATE_CAST(v_rotate_right, _Tpvec, OPENCV_HAL_NOP)

OPENCV_HAL_IMPL_LASX_ROTATE(v_int8x32)
OPENCV_HAL_IMPL_LASX_ROTATE(v_uint16x16)
OPENCV_HAL_IMPL_LASX_ROTATE(v_int16x16)
OPENCV_HAL_IMPL_LASX_ROTATE(v_uint32x8)
OPENCV_HAL_IMPL_LASX_ROTATE(v_int32x8)
OPENCV_HAL_IMPL_LASX_ROTATE(v_uint64x4)
OPENCV_HAL_IMPL_LASX_ROTATE(v_int64x4)

OPENCV_HAL_IMPL_LASX_ROTATE_CAST(v_rotate_left,  v_float32x8, _lasx_256_castsi256_ps)
OPENCV_HAL_IMPL_LASX_ROTATE_CAST(v_rotate_right, v_float32x8, _lasx_256_castsi256_ps)
OPENCV_HAL_IMPL_LASX_ROTATE_CAST(v_rotate_left,  v_float64x4, _lasx_256_castsi256_pd)
OPENCV_HAL_IMPL_LASX_ROTATE_CAST(v_rotate_right, v_float64x4, _lasx_256_castsi256_pd)

/** Reverse **/
inline v_uint8x32 v_reverse(const v_uint8x32 &a)
{
    static const __m256i perm = _v256_setr_b(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    __m256i vec = __lasx_xvshuf_b(a.val, a.val, perm);
    return v_uint8x32(__lasx_xvpermi_q(vec, vec, 1));
}

inline v_int8x32 v_reverse(const v_int8x32 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x16 v_reverse(const v_uint16x16 &a)
{
    __m256i vec = __lasx_xvshuf4i_h(a.val, 0x1B);
    vec = __lasx_xvshuf4i_w(vec, 0x4E);
    return v_uint16x16(__lasx_xvpermi_d(vec, 0x4E));
}

inline v_int16x16 v_reverse(const v_int16x16 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x8 v_reverse(const v_uint32x8 &a)
{
    __m256i vec = __lasx_xvshuf4i_w(a.val, 0x1B);
    return v_uint32x8(__lasx_xvpermi_d(vec, 0x4E));
}

inline v_int32x8 v_reverse(const v_int32x8 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x8 v_reverse(const v_float32x8 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x4 v_reverse(const v_uint64x4 &a)
{
    return v_uint64x4(__lasx_xvpermi_d(a.val, 0x1b));
}

inline v_int64x4 v_reverse(const v_int64x4 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x4 v_reverse(const v_float64x4 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }

////////// Reduce and mask /////////

/** Reduce **/
// this function is return a[0]+a[1]+...+a[31]
inline unsigned v_reduce_sum(const v_uint8x32& a)
{
    __m256i t1 = __lasx_xvhaddw_hu_bu(a.val, a.val);
    __m256i t2 = __lasx_xvhaddw_wu_hu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_du_wu(t2, t2);
    __m256i t4 = __lasx_xvhaddw_qu_du(t3, t3);
    return (unsigned)(((v8u32)t4)[0]+((v8u32)t4)[4]);
}

inline int v_reduce_sum(const v_int8x32& a)
{
    __m256i t1 = __lasx_xvhaddw_h_b(a.val, a.val);
    __m256i t2 = __lasx_xvhaddw_w_h(t1, t1);
    __m256i t3 = __lasx_xvhaddw_d_w(t2, t2);
    __m256i t4 = __lasx_xvhaddw_q_d(t3, t3);
    return (int)(((v8i32)t4)[0]+((v8i32)t4)[4]);
}

#define OPENCV_HAL_IMPL_LASX_REDUCE_32(_Tpvec, sctype, func, intrin) \
    inline sctype v_reduce_##func(const _Tpvec& a) \
    { \
        __m128i val = intrin(_v256_extract_low(a.val), _v256_extract_high(a.val)); \
        val = intrin(val, __lsx_vbsrl_v(val,8));    \
        val = intrin(val, __lsx_vbsrl_v(val,4));    \
        val = intrin(val, __lsx_vbsrl_v(val,2));    \
        val = intrin(val, __lsx_vbsrl_v(val,1));    \
        return (sctype)__lsx_vpickve2gr_w(val, 0);  \
    }

OPENCV_HAL_IMPL_LASX_REDUCE_32(v_uint8x32, uchar, min, __lsx_vmin_bu)
OPENCV_HAL_IMPL_LASX_REDUCE_32(v_int8x32,  schar, min, __lsx_vmin_b)
OPENCV_HAL_IMPL_LASX_REDUCE_32(v_uint8x32, uchar, max, __lsx_vmax_bu)
OPENCV_HAL_IMPL_LASX_REDUCE_32(v_int8x32,  schar, max, __lsx_vmax_b)

#define OPENCV_HAL_IMPL_LASX_REDUCE_16(_Tpvec, sctype, func, intrin) \
    inline sctype v_reduce_##func(const _Tpvec& a)                   \
    {                                                                \
        __m128i v0 = _v256_extract_low(a.val);                       \
        __m128i v1 = _v256_extract_high(a.val);                      \
        v0 = intrin(v0, v1);                                         \
        v0 = intrin(v0, __lsx_vbsrl_v(v0, 8));                       \
        v0 = intrin(v0, __lsx_vbsrl_v(v0, 4));                       \
        v0 = intrin(v0, __lsx_vbsrl_v(v0, 2));                       \
        return (sctype) __lsx_vpickve2gr_w(v0, 0);                   \
    }

OPENCV_HAL_IMPL_LASX_REDUCE_16(v_uint16x16, ushort, min, __lsx_vmin_hu)
OPENCV_HAL_IMPL_LASX_REDUCE_16(v_int16x16,  short,  min, __lsx_vmin_h)
OPENCV_HAL_IMPL_LASX_REDUCE_16(v_uint16x16, ushort, max, __lsx_vmax_hu)
OPENCV_HAL_IMPL_LASX_REDUCE_16(v_int16x16,  short,  max, __lsx_vmax_h)

#define OPENCV_HAL_IMPL_LASX_REDUCE_8(_Tpvec, sctype, func, intrin) \
    inline sctype v_reduce_##func(const _Tpvec& a)                  \
    {                                                               \
        __m128i v0 = _v256_extract_low(a.val);                      \
        __m128i v1 = _v256_extract_high(a.val);                     \
        v0 = intrin(v0, v1);                                        \
        v0 = intrin(v0, __lsx_vbsrl_v(v0, 8));                      \
        v0 = intrin(v0, __lsx_vbsrl_v(v0, 4));                      \
        return (sctype) __lsx_vpickve2gr_w(v0, 0);                  \
    }

OPENCV_HAL_IMPL_LASX_REDUCE_8(v_uint32x8, unsigned, min, __lsx_vmin_wu)
OPENCV_HAL_IMPL_LASX_REDUCE_8(v_int32x8,  int,      min, __lsx_vmin_w)
OPENCV_HAL_IMPL_LASX_REDUCE_8(v_uint32x8, unsigned, max, __lsx_vmax_wu)
OPENCV_HAL_IMPL_LASX_REDUCE_8(v_int32x8,  int,      max, __lsx_vmax_w)

#define OPENCV_HAL_IMPL_LASX_REDUCE_FLT(func, intrin)                 \
    inline float v_reduce_##func(const v_float32x8& a)                \
    {                                                                 \
        __m128 v0 = _v256_extract_low(a.val);                         \
        __m128 v1 = _v256_extract_high(a.val);                        \
        v0 = intrin(v0, v1);                                          \
        v0 = intrin(v0, __m128(__lsx_vpermi_w(*((__m128i*)&v0), *((__m128i*)&v0), 0x0e))); \
        v0 = intrin(v0, __m128(__lsx_vpermi_w(*((__m128i*)&v0), *((__m128i*)&v0), 0x01))); \
        float *fvalue = (float*)&v0;                                  \
        return fvalue[0];                                             \
    }

OPENCV_HAL_IMPL_LASX_REDUCE_FLT(min, __lsx_vfmin_s)
OPENCV_HAL_IMPL_LASX_REDUCE_FLT(max, __lsx_vfmax_s)

inline int v_reduce_sum(const v_int32x8& a)
{
    __m256i t1 = __lasx_xvhaddw_d_w(a.val, a.val);
    __m256i t2 = __lasx_xvhaddw_q_d(t1, t1);
    return (int)(((v8i32)t2)[0]+((v8i32)t2)[4]);
}

inline unsigned v_reduce_sum(const v_uint32x8& a)
{ return v_reduce_sum(v_reinterpret_as_s32(a)); }

inline int v_reduce_sum(const v_int16x16& a)
{ return v_reduce_sum(v_add(v_expand_low(a), v_expand_high(a))); }
inline unsigned v_reduce_sum(const v_uint16x16& a)
{ return v_reduce_sum(v_add(v_expand_low(a), v_expand_high(a))); }

inline float v_reduce_sum(const v_float32x8& a)
{
    float result = 0;
    float *pa = (float*)&a;
    for (int i = 0; i < 2; ++i) {
        result += pa[i*4] + pa[i*4+1] + pa[i*4+2] + pa[i*4+3];
    }
    return result;
}

inline uint64 v_reduce_sum(const v_uint64x4& a)
{
    __m256i t0 = __lasx_xvhaddw_qu_du(a.val, a.val);
    return (uint64)(((v4u64)t0)[0] + ((v4u64)t0)[2]);
}
inline int64 v_reduce_sum(const v_int64x4& a)
{
    __m256i t0 = __lasx_xvhaddw_q_d(a.val, a.val);
    return (int64)(((v4i64)t0)[0] + ((v4i64)t0)[2]);
}
inline double v_reduce_sum(const v_float64x4& a)
{
    double *pa = (double*)&a;
    return pa[0] + pa[1] + pa[2] + pa[3];
}

inline v_float32x8 v_reduce_sum4(const v_float32x8& a, const v_float32x8& b,
                                 const v_float32x8& c, const v_float32x8& d)
{
    float *pa = (float*)&a;
    float *pb = (float*)&b;
    float *pc = (float*)&c;
    float *pd = (float*)&d;

    float v0 = pa[0] + pa[1] + pa[2] + pa[3];
    float v1 = pb[0] + pb[1] + pb[2] + pb[3];
    float v2 = pc[0] + pc[1] + pc[2] + pc[3];
    float v3 = pd[0] + pd[1] + pd[2] + pd[3];
    float v4 = pa[4] + pa[5] + pa[6] + pa[7];
    float v5 = pb[4] + pb[5] + pb[6] + pb[7];
    float v6 = pc[4] + pc[5] + pc[6] + pc[7];
    float v7 = pd[4] + pd[5] + pd[6] + pd[7];
    return v_float32x8(v0, v1, v2, v3, v4, v5, v6, v7);
}

inline unsigned v_reduce_sad(const v_uint8x32& a, const v_uint8x32& b)
{
    __m256i t0 = __lasx_xvabsd_bu(a.val, b.val);
    __m256i t1 = __lasx_xvhaddw_hu_bu(t0, t0);
    __m256i t2 = __lasx_xvhaddw_wu_hu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_du_wu(t2, t2);
    __m256i t4 = __lasx_xvhaddw_qu_du(t3, t3);
    return (unsigned)(((v8u32)t4)[0]+((v8u32)t4)[4]);
}
inline unsigned v_reduce_sad(const v_int8x32& a, const v_int8x32& b)
{
    __m256i t0 = __lasx_xvabsd_b(a.val, b.val);
    __m256i t1 = __lasx_xvhaddw_hu_bu(t0, t0);
    __m256i t2 = __lasx_xvhaddw_wu_hu(t1, t1);
    __m256i t3 = __lasx_xvhaddw_du_wu(t2, t2);
    __m256i t4 = __lasx_xvhaddw_qu_du(t3, t3);
    return (unsigned)(((v8u32)t4)[0]+((v8u32)t4)[4]);
}
inline unsigned v_reduce_sad(const v_uint16x16& a, const v_uint16x16& b)
{
    v_uint32x8 l, h;
    v_expand(v_add_wrap(v_sub(a, b), v_sub(b, a)), l, h);
    return v_reduce_sum(v_add(l, h));
}
inline unsigned v_reduce_sad(const v_int16x16& a, const v_int16x16& b)
{
    v_uint32x8 l, h;
    v_expand(v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b))), l, h);
    return v_reduce_sum(v_add(l, h));
}
inline unsigned v_reduce_sad(const v_uint32x8& a, const v_uint32x8& b)
{
    return v_reduce_sum(v_sub(v_max(a, b), v_min(a, b)));
}
inline unsigned v_reduce_sad(const v_int32x8& a, const v_int32x8& b)
{
    v_int32x8 m = v_lt(a, b);
    return v_reduce_sum(v_reinterpret_as_u32(v_sub(v_xor(v_sub(a, b), m), m)));
}
inline float v_reduce_sad(const v_float32x8& a, const v_float32x8& b)
{
    v_float32x8 a_b = v_sub(a, b);
    return v_reduce_sum(v_float32x8(*((__m256i*)&a_b.val) & __lasx_xvreplgr2vr_w(0x7fffffff)));
}

/** Popcount **/
inline v_uint8x32 v_popcount(const v_uint8x32& a)
{ return v_uint8x32(__lasx_xvpcnt_b(a.val)); }
inline v_uint16x16 v_popcount(const v_uint16x16& a)
{ return v_uint16x16(__lasx_xvpcnt_h(a.val)); }
inline v_uint32x8 v_popcount(const v_uint32x8& a)
{ return v_uint32x8(__lasx_xvpcnt_w(a.val)); }
inline v_uint64x4 v_popcount(const v_uint64x4& a)
{ return v_uint64x4(__lasx_xvpcnt_d(a.val)); }
inline v_uint8x32 v_popcount(const v_int8x32& a)
{ return v_popcount(v_reinterpret_as_u8(a)); }
inline v_uint16x16 v_popcount(const v_int16x16& a)
{ return v_popcount(v_reinterpret_as_u16(a)); }
inline v_uint32x8 v_popcount(const v_int32x8& a)
{ return v_popcount(v_reinterpret_as_u32(a)); }
inline v_uint64x4 v_popcount(const v_int64x4& a)
{ return v_popcount(v_reinterpret_as_u64(a)); }

inline int v_signmask(const v_int8x32& a)
{
    __m256i result = __lasx_xvmskltz_b(a.val);
    int mask = __lasx_xvpickve2gr_w(result, 0);
    mask |= (__lasx_xvpickve2gr_w(result, 4) << 16);
    return mask;
}
inline int v_signmask(const v_uint8x32& a)
{ return v_signmask(v_reinterpret_as_s8(a)); }

inline int v_signmask(const v_int16x16& a)
{ return v_signmask(v_pack(a, a)) & 0xFFFF; }
inline int v_signmask(const v_uint16x16& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }

inline int v_signmask(const v_int32x8& a)
{
    __m256i result = __lasx_xvmskltz_w(a.val);
    int mask = __lasx_xvpickve2gr_w(result, 0);
    mask |= (__lasx_xvpickve2gr_w(result, 4) << 4);
    return mask;
}
inline int v_signmask(const v_uint32x8& a)
{ return v_signmask(*(v_int32x8*)(&a)); }

inline int v_signmask(const v_int64x4& a)
{
    __m256i result = __lasx_xvmskltz_d(a.val);
    int mask = __lasx_xvpickve2gr_d(result, 0);
    mask |= (__lasx_xvpickve2gr_w(result, 4) << 2);
    return mask;
}
inline int v_signmask(const v_uint64x4& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }

inline int v_signmask(const v_float32x8& a)
{ return v_signmask(*(v_int32x8*)(&a)); }

inline int v_signmask(const v_float64x4& a)
{ return v_signmask(*(v_int64x4*)(&a)); }

inline int v_scan_forward(const v_int8x32& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_uint8x32& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_int16x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_uint16x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_int32x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_uint32x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_float32x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_int64x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_uint64x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_float64x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }

/** Checks **/
#define OPENCV_HAL_IMPL_LASX_CHECK(_Tpvec, allmask) \
    inline bool v_check_all(const _Tpvec& a) { return v_signmask(a) == allmask; } \
    inline bool v_check_any(const _Tpvec& a) { return v_signmask(a) != 0; }
OPENCV_HAL_IMPL_LASX_CHECK(v_uint8x32, -1)
OPENCV_HAL_IMPL_LASX_CHECK(v_int8x32, -1)
OPENCV_HAL_IMPL_LASX_CHECK(v_uint32x8, 255)
OPENCV_HAL_IMPL_LASX_CHECK(v_int32x8, 255)
OPENCV_HAL_IMPL_LASX_CHECK(v_uint64x4, 15)
OPENCV_HAL_IMPL_LASX_CHECK(v_int64x4, 15)
OPENCV_HAL_IMPL_LASX_CHECK(v_float32x8, 255)
OPENCV_HAL_IMPL_LASX_CHECK(v_float64x4, 15)

#define OPENCV_HAL_IMPL_LASX_CHECK_SHORT(_Tpvec)  \
    inline bool v_check_all(const _Tpvec& a) { return (v_signmask(v_reinterpret_as_s8(a)) & 0xaaaaaaaa) == 0xaaaaaaaa; } \
    inline bool v_check_any(const _Tpvec& a) { return (v_signmask(v_reinterpret_as_s8(a)) & 0xaaaaaaaa) != 0; }
OPENCV_HAL_IMPL_LASX_CHECK_SHORT(v_uint16x16)
OPENCV_HAL_IMPL_LASX_CHECK_SHORT(v_int16x16)

////////// Other math /////////

/** Some frequent operations **/
#define OPENCV_HAL_IMPL_LASX_MULADD(_Tpvec, suffix)                            \
    inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)     \
    { return _Tpvec(__lasx_xvfmadd_##suffix(a.val, b.val, c.val)); }           \
    inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)  \
    { return _Tpvec(__lasx_xvfmadd_##suffix(a.val, b.val, c.val)); }           \
    inline _Tpvec v_sqrt(const _Tpvec& x)                                      \
    { return _Tpvec(__lasx_xvfsqrt_##suffix(x.val)); }                         \
    inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b)            \
    { return v_fma(a, a, v_mul(b, b)); }                                       \
    inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b)                \
    { return v_sqrt(v_fma(a, a, v_mul(b, b))); }

OPENCV_HAL_IMPL_LASX_MULADD(v_float32x8, s)
OPENCV_HAL_IMPL_LASX_MULADD(v_float64x4, d)

inline v_int32x8 v_fma(const v_int32x8& a, const v_int32x8& b, const v_int32x8& c)
{
    return v_int32x8(__lasx_xvmadd_w(c.val, a.val, b.val));
}

inline v_int32x8 v_muladd(const v_int32x8& a, const v_int32x8& b, const v_int32x8& c)
{
    return v_fma(a, b, c);
}

inline v_float32x8 v_invsqrt(const v_float32x8& x)
{ return v_float32x8(__lasx_xvfrsqrt_s(x.val)); }

inline v_float64x4 v_invsqrt(const v_float64x4& x)
{ return v_float64x4(__lasx_xvfrsqrt_d(x.val)); }

/** Absolute values **/
#define OPENCV_HAL_IMPL_LASX_ABS(_Tpvec, suffix)         \
    inline v_u##_Tpvec v_abs(const v_##_Tpvec& x)        \
    { return v_u##_Tpvec(__lasx_xvabsd_##suffix(x.val, __lasx_xvreplgr2vr_w(0))); }

OPENCV_HAL_IMPL_LASX_ABS(int8x32,  b)
OPENCV_HAL_IMPL_LASX_ABS(int16x16, h)
OPENCV_HAL_IMPL_LASX_ABS(int32x8,  w)

inline v_float32x8 v_abs(const v_float32x8& x)
{ return v_float32x8(*((__m256i*)&x) & __lasx_xvreplgr2vr_w(0x7fffffff)); }
inline v_float64x4 v_abs(const v_float64x4& x)
{ return v_float64x4(*((__m256i*)&x) & __lasx_xvreplgr2vr_d(0x7fffffffffffffff)); }

/** Absolute difference **/
inline v_uint8x32 v_absdiff(const v_uint8x32& a, const v_uint8x32& b)
{ return (v_uint8x32)__lasx_xvabsd_bu(a.val, b.val); }
inline v_uint16x16 v_absdiff(const v_uint16x16& a, const v_uint16x16& b)
{ return (v_uint16x16)__lasx_xvabsd_hu(a.val, b.val); }
inline v_uint32x8 v_absdiff(const v_uint32x8& a, const v_uint32x8& b)
{ return (v_uint32x8)__lasx_xvabsd_wu(a.val, b.val); }

inline v_uint8x32 v_absdiff(const v_int8x32& a, const v_int8x32& b)
{ return (v_uint8x32)__lasx_xvabsd_b(a.val, b.val); }
inline v_uint16x16 v_absdiff(const v_int16x16& a, const v_int16x16& b)
{ return (v_uint16x16)__lasx_xvabsd_h(a.val, b.val); }
inline v_uint32x8 v_absdiff(const v_int32x8& a, const v_int32x8& b)
{ return (v_uint32x8)__lasx_xvabsd_w(a.val, b.val); }

inline v_float32x8 v_absdiff(const v_float32x8& a, const v_float32x8& b)
{ return v_abs(v_sub(a, b)); }

inline v_float64x4 v_absdiff(const v_float64x4& a, const v_float64x4& b)
{ return v_abs(v_sub(a, b)); }

/** Saturating absolute difference **/
inline v_int8x32 v_absdiffs(const v_int8x32& a, const v_int8x32& b)
{
    v_int8x32 d = v_sub(a, b);
    v_int8x32 m = v_lt(a, b);
    return v_sub(v_xor(d, m), m);
}
inline v_int16x16 v_absdiffs(const v_int16x16& a, const v_int16x16& b)
{ return v_sub(v_max(a, b), v_min(a, b)); }

////////// Conversions /////////

/** Rounding **/
inline v_int32x8 v_round(const v_float32x8& a)
{ return v_int32x8(__lasx_xvftint_w_s(a.val)); }

inline v_int32x8 v_round(const v_float64x4& a)
{ __m256i t = __lasx_xvftint_w_d(a.val, a.val);
  return v_int32x8(__lasx_xvpermi_d(t, 0x88)); }

inline v_int32x8 v_round(const v_float64x4& a, const v_float64x4& b)
{
    __m256i abi = __lasx_xvftint_w_d(b.val, a.val);
    return v_int32x8(__lasx_xvpermi_d(abi, 0b11011000)); //3120
}

inline v_int32x8 v_trunc(const v_float32x8& a)
{ return v_int32x8(__lasx_xvftintrz_w_s(a.val)); }

inline v_int32x8 v_trunc(const v_float64x4& a)
{ __m256i t = __lasx_xvftintrz_w_d(a.val, a.val);
  return v_int32x8(__lasx_xvpermi_d(t, 0x88)); }

inline v_int32x8 v_floor(const v_float32x8& a)
{ return v_int32x8(__lasx_xvftintrz_w_s(__m256(__lasx_xvfrintrm_s(a.val)))); }

inline v_int32x8 v_floor(const v_float64x4& a)
{ return v_trunc(v_float64x4(__lasx_xvfrintrm_d(a.val))); }

inline v_int32x8 v_ceil(const v_float32x8& a)
{ return v_int32x8(__lasx_xvftintrz_w_s(__m256(__lasx_xvfrintrp_s(a.val)))); }

inline v_int32x8 v_ceil(const v_float64x4& a)
{ return v_trunc(v_float64x4(__lasx_xvfrintrp_d(a.val))); }

/** To float **/
inline v_float32x8 v_cvt_f32(const v_int32x8& a)
{ return v_float32x8(__lasx_xvffint_s_w(a.val)); }

inline v_float32x8 v_cvt_f32(const v_float64x4& a)
{ return v_float32x8(__lasx_xvpermi_d(__lasx_xvfcvt_s_d(a.val, a.val), 0x88)); }

inline v_float32x8 v_cvt_f32(const v_float64x4& a, const v_float64x4& b)
{
    __m256 abf = __lasx_xvfcvt_s_d(a.val, b.val);  //warnning: order of a,b is diff from instruction xvfcvt.s.d
    return v_float32x8(__lasx_xvpermi_d(abf, 0x8D));
}

inline v_float64x4 v_cvt_f64(const v_int32x8& a)
{
    __m256i alow = __lasx_xvpermi_d(a.val, 0x10);
    return v_float64x4(__lasx_xvffintl_d_w(alow));
}

inline v_float64x4 v_cvt_f64_high(const v_int32x8& a)
{
    __m256i ahigh = __lasx_xvpermi_d(a.val, 0x32);
    return v_float64x4(__lasx_xvffintl_d_w(ahigh));
}

inline v_float64x4 v_cvt_f64(const v_float32x8& a)
{
    __m256i alow = __lasx_xvpermi_d(a.val, 0x10);
    return v_float64x4(__lasx_xvfcvtl_d_s((__m256)alow));
}

inline v_float64x4 v_cvt_f64_high(const v_float32x8& a)
{
    __m256i ahigh = __lasx_xvpermi_d(a.val, 0x32);
    return v_float64x4(__lasx_xvfcvtl_d_s((__m256)ahigh));
}

inline v_float64x4 v_cvt_f64(const v_int64x4& v)
{ return v_float64x4(__lasx_xvffint_d_l(v.val)); }

////////////// Lookup table access ////////////////////

inline v_int8x32 v256_lut(const schar* tab, const int* idx)
{
    return v_int8x32(_v256_setr_b(tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]],
                                  tab[idx[ 6]], tab[idx[ 7]], tab[idx[ 8]], tab[idx[ 9]], tab[idx[10]], tab[idx[11]],
                                  tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]], tab[idx[16]], tab[idx[17]],
                                  tab[idx[18]], tab[idx[19]], tab[idx[20]], tab[idx[21]], tab[idx[22]], tab[idx[23]],
                                  tab[idx[24]], tab[idx[25]], tab[idx[26]], tab[idx[27]], tab[idx[28]], tab[idx[29]],
                                  tab[idx[30]], tab[idx[31]]));
}
inline v_int8x32 v256_lut_pairs(const schar* tab, const int* idx)
{
    return v_int8x32(_v256_setr_h(*(const short*)(tab + idx[ 0]), *(const short*)(tab + idx[ 1]), *(const short*)(tab + idx[ 2]),
                                  *(const short*)(tab + idx[ 3]), *(const short*)(tab + idx[ 4]), *(const short*)(tab + idx[ 5]),
                                  *(const short*)(tab + idx[ 6]), *(const short*)(tab + idx[ 7]), *(const short*)(tab + idx[ 8]),
                                  *(const short*)(tab + idx[ 9]), *(const short*)(tab + idx[10]), *(const short*)(tab + idx[11]),
                                  *(const short*)(tab + idx[12]), *(const short*)(tab + idx[13]), *(const short*)(tab + idx[14]),
                                  *(const short*)(tab + idx[15])));
}
inline v_int8x32 v256_lut_quads(const schar* tab, const int* idx)
{
    return v_int8x32(_v256_setr_w(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                                  *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]),
                                  *(const int*)(tab + idx[4]), *(const int*)(tab + idx[5]),
                                  *(const int*)(tab + idx[6]), *(const int*)(tab + idx[7])));
}
inline v_uint8x32 v256_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v256_lut((const schar *)tab, idx)); }
inline v_uint8x32 v256_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v256_lut_pairs((const schar *)tab, idx)); }
inline v_uint8x32 v256_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v256_lut_quads((const schar *)tab, idx)); }

inline v_int16x16 v256_lut(const short* tab, const int* idx)
{
    return v_int16x16(_v256_setr_h(tab[idx[ 0]], tab[idx[ 1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]],
                                   tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]], tab[idx[ 8]], tab[idx[ 9]],
                                   tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]],
                                   tab[idx[15]]));
}
inline v_int16x16 v256_lut_pairs(const short* tab, const int* idx)
{
    return v_int16x16(_v256_setr_w(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                                   *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]),
                                   *(const int*)(tab + idx[4]), *(const int*)(tab + idx[5]),
                                   *(const int*)(tab + idx[6]), *(const int*)(tab + idx[7]) ));
}
inline v_int16x16 v256_lut_quads(const short* tab, const int* idx)
{
    return v_int16x16(_v256_setr_d(*(const long long int*)(tab + idx[0]), *(const long long int*)(tab + idx[1]),
                                   *(const long long int*)(tab + idx[2]), *(const long long int*)(tab + idx[3]) ));

}
inline v_uint16x16 v256_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v256_lut((const short *)tab, idx)); }
inline v_uint16x16 v256_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v256_lut_pairs((const short *)tab, idx)); }
inline v_uint16x16 v256_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v256_lut_quads((const short *)tab, idx)); }

inline v_int32x8 v256_lut(const int* tab, const int* idx)
{
    return v_int32x8(_v256_setr_w(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                                  *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]),
                                  *(const int*)(tab + idx[4]), *(const int*)(tab + idx[5]),
                                  *(const int*)(tab + idx[6]), *(const int*)(tab + idx[7]) ));
}
inline v_int32x8 v256_lut_pairs(const int* tab, const int* idx)
{
    return v_int32x8(_v256_setr_d(*(const long long int*)(tab + idx[0]), *(const long long int*)(tab + idx[1]),
                                  *(const long long int*)(tab + idx[2]), *(const long long int*)(tab + idx[3]) ));
}
inline v_int32x8 v256_lut_quads(const int* tab, const int* idx)
{
    return v_int32x8(_v256_combine(__lsx_vld(tab + idx[0], 0), __lsx_vld(tab + idx[1], 0)));
}
inline v_uint32x8 v256_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v256_lut((const int *)tab, idx)); }
inline v_uint32x8 v256_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v256_lut_pairs((const int *)tab, idx)); }
inline v_uint32x8 v256_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v256_lut_quads((const int *)tab, idx)); }

inline v_int64x4 v256_lut(const int64* tab, const int* idx)
{
    return v_int64x4(_v256_setr_d(*(const long long int*)(tab + idx[0]), *(const long long int*)(tab + idx[1]),
                                  *(const long long int*)(tab + idx[2]), *(const long long int*)(tab + idx[3]) ));
}
inline v_int64x4 v256_lut_pairs(const int64* tab, const int* idx)
{
    return v_int64x4(_v256_combine(__lsx_vld(tab + idx[0], 0), __lsx_vld(tab + idx[1], 0)));
}
inline v_uint64x4 v256_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v256_lut((const int64 *)tab, idx)); }
inline v_uint64x4 v256_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v256_lut_pairs((const int64 *)tab, idx)); }

inline v_float32x8 v256_lut(const float* tab, const int* idx)
{
    return v_float32x8(_v256_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                                     tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]));
}
inline v_float32x8 v256_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v256_lut_pairs((const int *)tab, idx)); }
inline v_float32x8 v256_lut_quads(const float* tab, const int* idx) { return v_reinterpret_as_f32(v256_lut_quads((const int *)tab, idx)); }

inline v_float64x4 v256_lut(const double* tab, const int* idx)
{
    return v_float64x4(_v256_setr_pd(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}
inline v_float64x4 v256_lut_pairs(const double* tab, const int* idx)
{ return v_float64x4(_v256_combine(__lsx_vld(tab + idx[0], 0), __lsx_vld(tab + idx[1], 0))); }

inline v_int32x8 v_lut(const int* tab, const v_int32x8& idxvec)
{
    int *idx = (int*)&idxvec.val;
    return v256_lut(tab, idx);
}

inline v_uint32x8 v_lut(const unsigned* tab, const v_int32x8& idxvec)
{
    return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
}

inline v_float32x8 v_lut(const float* tab, const v_int32x8& idxvec)
{
    const int *idx = (const int*)&idxvec.val;
    return v256_lut(tab, idx);
}

inline v_float64x4 v_lut(const double* tab, const v_int32x8& idxvec)
{
    const int *idx = (const int*)&idxvec.val;
    return v256_lut(tab, idx);
}

inline void v_lut_deinterleave(const float* tab, const v_int32x8& idxvec, v_float32x8& x, v_float32x8& y)
{
    const int *idx = (const int*)&idxvec.val;
    __m128i xy01, xy45, xy23, xy67;
    xy01 = __lsx_vld(tab + idx[0], 0);
    xy01 = __lsx_vextrins_d(xy01, __lsx_vld(tab + idx[1], 0), 0x10);
    xy45 = __lsx_vld(tab + idx[4], 0);
    xy45 = __lsx_vextrins_d(xy45, __lsx_vld(tab + idx[5], 0), 0x10);
    __m256i xy0145 = _v256_combine(xy01, xy45);
    xy23 = __lsx_vld(tab + idx[2], 0);
    xy23 = __lsx_vextrins_d(xy23, __lsx_vld(tab + idx[3], 0), 0x10);
    xy67 = __lsx_vld(tab + idx[6], 0);
    xy67 = __lsx_vextrins_d(xy67, __lsx_vld(tab + idx[7], 0), 0x10);
    __m256i xy2367 = _v256_combine(xy23, xy67);

    __m256i xxyy0145 = __lasx_xvilvl_w(xy2367, xy0145);
    __m256i xxyy2367 = __lasx_xvilvh_w(xy2367, xy0145);

    x = v_float32x8(__lasx_xvilvl_w(xxyy2367, xxyy0145));
    y = v_float32x8(__lasx_xvilvh_w(xxyy2367, xxyy0145));
}

inline void v_lut_deinterleave(const double* tab, const v_int32x8& idxvec, v_float64x4& x, v_float64x4& y)
{
    //int CV_DECL_ALIGNED(32) idx[4];
    const int *idx = (const int*)&idxvec.val;
    __m128i xy0 = __lsx_vld(tab + idx[0], 0);
    __m128i xy2 = __lsx_vld(tab + idx[2], 0);
    __m128i xy1 = __lsx_vld(tab + idx[1], 0);
    __m128i xy3 = __lsx_vld(tab + idx[3], 0);
    __m256i xy02 = _v256_combine(xy0, xy2);
    __m256i xy13 = _v256_combine(xy1, xy3);

    x = v_float64x4(__lasx_xvilvl_d(xy13, xy02));
    y = v_float64x4(__lasx_xvilvh_d(xy13, xy02));
}

inline v_int8x32 v_interleave_pairs(const v_int8x32& vec)
{
    return v_int8x32(__lasx_xvshuf_b(vec.val, vec.val,
                       _v256_set_d(0x0f0d0e0c0b090a08, 0x0705060403010200, 0x0f0d0e0c0b090a08, 0x0705060403010200)));
}
inline v_uint8x32 v_interleave_pairs(const v_uint8x32& vec)
{ return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x32 v_interleave_quads(const v_int8x32& vec)
{
    return v_int8x32(__lasx_xvshuf_b(vec.val, vec.val,
                       _v256_set_d(0x0f0b0e0a0d090c08, 0x0703060205010400, 0x0f0b0e0a0d090c08, 0x0703060205010400)));
}
inline v_uint8x32 v_interleave_quads(const v_uint8x32& vec)
{ return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x16 v_interleave_pairs(const v_int16x16& vec)
{
    return v_int16x16(__lasx_xvshuf_b(vec.val, vec.val,
                        _v256_set_d(0x0f0e0b0a0d0c0908, 0x0706030205040100, 0x0f0e0b0a0d0c0908, 0x0706030205040100)));
}
inline v_uint16x16 v_interleave_pairs(const v_uint16x16& vec)
{ return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x16 v_interleave_quads(const v_int16x16& vec)
{
    return v_int16x16(__lasx_xvshuf_b(vec.val, vec.val,
                        _v256_set_d(0x0f0e07060d0c0504, 0x0b0a030209080100, 0x0f0e07060d0c0504, 0x0b0a030209080100)));
}
inline v_uint16x16 v_interleave_quads(const v_uint16x16& vec)
{ return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x8 v_interleave_pairs(const v_int32x8& vec)
{
    return v_int32x8(__lasx_xvshuf4i_w(vec.val, 0xd8));
}
inline v_uint32x8 v_interleave_pairs(const v_uint32x8& vec)
{ return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x8 v_interleave_pairs(const v_float32x8& vec)
{ return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x32 v_pack_triplets(const v_int8x32& vec)
{
    __m256i vzero = __lasx_xvreplgr2vr_w(0);
    __m256i t1 = __lasx_xvshuf_b(vzero, vec.val,
                   _v256_set_d(0x1211100f0e0d0c0a, 0x0908060504020100, 0x1211100f0e0d0c0a, 0x0908060504020100));
    return v_int8x32(__lasx_xvperm_w(t1,
                       _v256_set_d(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}
inline v_uint8x32 v_pack_triplets(const v_uint8x32& vec)
{ return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x16 v_pack_triplets(const v_int16x16& vec)
{
    __m256i vzero = __lasx_xvreplgr2vr_w(0);
    __m256i t1 = __lasx_xvshuf_b(vzero, vec.val,
                   _v256_set_d(0x11100f0e0d0c0b0a, 0x0908050403020100, 0x11100f0e0d0c0b0a, 0x0908050403020100));
    return v_int16x16(__lasx_xvperm_w(t1,
                        _v256_set_d(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}
inline v_uint16x16 v_pack_triplets(const v_uint16x16& vec)
{ return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x8 v_pack_triplets(const v_int32x8& vec)
{
    return v_int32x8(__lasx_xvperm_w(vec.val,
                       _v256_set_d(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}
inline v_uint32x8 v_pack_triplets(const v_uint32x8& vec)
{ return v_reinterpret_as_u32(v_pack_triplets(v_reinterpret_as_s32(vec))); }
inline v_float32x8 v_pack_triplets(const v_float32x8& vec)
{
    return v_float32x8(__lasx_xvperm_w(*(__m256i*)(&vec.val),
                         _v256_set_d(0x0000000700000007, 0x0000000600000005, 0x0000000400000002, 0x0000000100000000)));
}

////////// Matrix operations /////////

//////// Dot Product ////////

// 16 >> 32
inline v_int32x8 v_dotprod(const v_int16x16& a, const v_int16x16& b)
{ return v_int32x8(__lasx_xvadd_w(__lasx_xvmulwev_w_h(a.val, b.val), __lasx_xvmulwod_w_h(a.val, b.val))); }

inline v_int32x8 v_dotprod(const v_int16x16& a, const v_int16x16& b, const v_int32x8& c)
{ return v_add(v_dotprod(a, b), c); }

// 32 >> 64
inline v_int64x4 v_dotprod(const v_int32x8& a, const v_int32x8& b)
{
    __m256i even = __lasx_xvmulwev_d_w(a.val, b.val);
    return v_int64x4(__lasx_xvmaddwod_d_w(even, a.val, b.val));
}
inline v_int64x4 v_dotprod(const v_int32x8& a, const v_int32x8& b, const v_int64x4& c)
{
    __m256i even = __lasx_xvmaddwev_d_w(c.val, a.val, b.val);
    return v_int64x4(__lasx_xvmaddwod_d_w(even, a.val, b.val));
}

// 8 >> 32
inline v_uint32x8 v_dotprod_expand(const v_uint8x32& a, const v_uint8x32& b)
{
    __m256i even  = __lasx_xvmulwev_h_bu(a.val, b.val);
    __m256i odd   = __lasx_xvmulwod_h_bu(a.val, b.val);
    __m256i prod0 = __lasx_xvhaddw_wu_hu(even, even);
    __m256i prod1 = __lasx_xvhaddw_wu_hu(odd, odd);
    return v_uint32x8(__lasx_xvadd_w(prod0, prod1));
}
inline v_uint32x8 v_dotprod_expand(const v_uint8x32& a, const v_uint8x32& b, const v_uint32x8& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int32x8 v_dotprod_expand(const v_int8x32& a, const v_int8x32& b)
{
    __m256i even  = __lasx_xvmulwev_h_b(a.val, b.val);
    __m256i odd   = __lasx_xvmulwod_h_b(a.val, b.val);
    __m256i prod0 = __lasx_xvhaddw_w_h(even, even);
    __m256i prod1 = __lasx_xvhaddw_w_h(odd, odd);
    return v_int32x8(__lasx_xvadd_w(prod0, prod1));
}
inline v_int32x8 v_dotprod_expand(const v_int8x32& a, const v_int8x32& b, const v_int32x8& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 16 >> 64
inline v_uint64x4 v_dotprod_expand(const v_uint16x16& a, const v_uint16x16& b)
{
    __m256i even  = __lasx_xvmulwev_w_hu(a.val, b.val);
    __m256i odd   = __lasx_xvmulwod_w_hu(a.val, b.val);
    __m256i prod0 = __lasx_xvhaddw_du_wu(even, even);
    __m256i prod1 = __lasx_xvhaddw_du_wu(odd, odd);
    return v_uint64x4(__lasx_xvadd_d(prod0, prod1));
}
inline v_uint64x4 v_dotprod_expand(const v_uint16x16& a, const v_uint16x16& b, const v_uint64x4& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int64x4 v_dotprod_expand(const v_int16x16& a, const v_int16x16& b)
{
    __m256i even  = __lasx_xvmulwev_w_h(a.val, b.val);
    __m256i odd   = __lasx_xvmulwod_w_h(a.val, b.val);
    __m256i prod0 = __lasx_xvhaddw_d_w(even, even);
    __m256i prod1 = __lasx_xvhaddw_d_w(odd, odd);
    return v_int64x4(__lasx_xvadd_d(prod0, prod1));
}

inline v_int64x4 v_dotprod_expand(const v_int16x16& a, const v_int16x16& b, const v_int64x4& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 32 >> 64f
inline v_float64x4 v_dotprod_expand(const v_int32x8& a, const v_int32x8& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x4 v_dotprod_expand(const v_int32x8& a, const v_int32x8& b, const v_float64x4& c)
{ return v_add(v_dotprod_expand(a, b), c); }

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x8 v_dotprod_fast(const v_int16x16& a, const v_int16x16& b)
{ return v_dotprod(a, b); }
inline v_int32x8 v_dotprod_fast(const v_int16x16& a, const v_int16x16& b, const v_int32x8& c)
{ return v_dotprod(a, b, c); }

// 32 >> 64
inline v_int64x4 v_dotprod_fast(const v_int32x8& a, const v_int32x8& b)
{ return v_dotprod(a, b); }
inline v_int64x4 v_dotprod_fast(const v_int32x8& a, const v_int32x8& b, const v_int64x4& c)
{ return v_dotprod(a, b, c); }

// 8 >> 32
inline v_uint32x8 v_dotprod_expand_fast(const v_uint8x32& a, const v_uint8x32& b)
{ return v_dotprod_expand(a, b); }
inline v_uint32x8 v_dotprod_expand_fast(const v_uint8x32& a, const v_uint8x32& b, const v_uint32x8& c)
{ return v_dotprod_expand(a, b, c); }

inline v_int32x8 v_dotprod_expand_fast(const v_int8x32& a, const v_int8x32& b)
{ return v_dotprod_expand(a, b); }
inline v_int32x8 v_dotprod_expand_fast(const v_int8x32& a, const v_int8x32& b, const v_int32x8& c)
{ return v_dotprod_expand(a, b, c); }

// 16 >> 64
inline v_uint64x4 v_dotprod_expand_fast(const v_uint16x16& a, const v_uint16x16& b)
{
    __m256i even  = __lasx_xvmulwev_w_hu(a.val, b.val);
    __m256i odd   = __lasx_xvmulwod_w_hu(a.val, b.val);
    __m256i prod0 = __lasx_xvhaddw_du_wu(even, even);
    __m256i prod1 = __lasx_xvhaddw_du_wu(odd, odd);
    return v_uint64x4(__lasx_xvadd_d(__lasx_xvilvl_d(prod1, prod0), __lasx_xvilvh_d(prod1, prod0)));
}
inline v_uint64x4 v_dotprod_expand_fast(const v_uint16x16& a, const v_uint16x16& b, const v_uint64x4& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

inline v_int64x4 v_dotprod_expand_fast(const v_int16x16& a, const v_int16x16& b)
{
    __m256i prod = __lasx_xvadd_w(__lasx_xvmulwev_w_h(a.val, b.val), __lasx_xvmulwod_w_h(a.val, b.val));
    __m256i sign = __lasx_xvsrai_w(prod, 31);
    __m256i lo = __lasx_xvilvl_w(sign, prod);
    __m256i hi = __lasx_xvilvh_w(sign, prod);
    return v_int64x4(__lasx_xvadd_d(lo, hi));
}
inline v_int64x4 v_dotprod_expand_fast(const v_int16x16& a, const v_int16x16& b, const v_int64x4& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

// 32 >> 64f
inline v_float64x4 v_dotprod_expand_fast(const v_int32x8& a, const v_int32x8& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x4 v_dotprod_expand_fast(const v_int32x8& a, const v_int32x8& b, const v_float64x4& c)
{ return v_dotprod_expand(a, b, c); }


#define OPENCV_HAL_LASX_SPLAT2_PS(a, im) \
    v_float32x8(__lasx_xvpermi_w(a.val, a.val, im))

inline v_float32x8 v_matmul(const v_float32x8& v, const v_float32x8& m0,
                            const v_float32x8& m1, const v_float32x8& m2,
                            const v_float32x8& m3)
{
    v_float32x8 v04 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0);
    v_float32x8 v15 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0x55);
    v_float32x8 v26 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0xAA);
    v_float32x8 v37 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0xFF);
    return v_fma(v04, m0, v_fma(v15, m1, v_fma(v26, m2, v_mul(v37, m3))));
}

inline v_float32x8 v_matmuladd(const v_float32x8& v, const v_float32x8& m0,
                               const v_float32x8& m1, const v_float32x8& m2,
                               const v_float32x8& a)
{
    v_float32x8 v04 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0);
    v_float32x8 v15 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0x55);
    v_float32x8 v26 = OPENCV_HAL_LASX_SPLAT2_PS(v, 0xAA);
    return v_fma(v04, m0, v_fma(v15, m1, v_fma(v26, m2, a)));
}


#define OPENCV_HAL_IMPL_LASX_TRANSPOSE4x4(_Tpvec, cast_from, cast_to)           \
    inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1,              \
                               const _Tpvec& a2, const _Tpvec& a3,              \
                               _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3)  \
    {                                                                           \
        __m256i t0 = cast_from(__lasx_xvilvl_w(a1.val, a0.val));                \
        __m256i t1 = cast_from(__lasx_xvilvl_w(a3.val, a2.val));                \
        __m256i t2 = cast_from(__lasx_xvilvh_w(a1.val, a0.val));                \
        __m256i t3 = cast_from(__lasx_xvilvh_w(a3.val, a2.val));                \
        b0.val = cast_to(__lasx_xvilvl_d(t1, t0));                              \
        b1.val = cast_to(__lasx_xvilvh_d(t1, t0));                              \
        b2.val = cast_to(__lasx_xvilvl_d(t3, t2));                              \
        b3.val = cast_to(__lasx_xvilvh_d(t3, t2));                              \
    }

OPENCV_HAL_IMPL_LASX_TRANSPOSE4x4(v_uint32x8, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_LASX_TRANSPOSE4x4(v_int32x8,  OPENCV_HAL_NOP, OPENCV_HAL_NOP)

inline void v_transpose4x4(const v_float32x8 &a0, const v_float32x8 &a1,
                           const v_float32x8 &a2, const v_float32x8 &a3,
                           v_float32x8 &b0, v_float32x8 &b1, v_float32x8 &b2, v_float32x8 &b3)
{
    __m256i t0 = __lasx_xvilvl_w(__m256i(a1.val), __m256i(a0.val));
    __m256i t1 = __lasx_xvilvl_w(__m256i(a3.val), __m256i(a2.val));
    __m256i t2 = __lasx_xvilvh_w(__m256i(a1.val), __m256i(a0.val));
    __m256i t3 = __lasx_xvilvh_w(__m256i(a3.val), __m256i(a2.val));
    b0.val = __m256(__lasx_xvilvl_d(t1, t0));
    b1.val = __m256(__lasx_xvilvh_d(t1, t0));
    b2.val = __m256(__lasx_xvilvl_d(t3, t2));
    b3.val = __m256(__lasx_xvilvh_d(t3, t2));
}

//////////////// Value reordering ///////////////

/* Expand */
#define OPENCV_HAL_IMPL_LASX_EXPAND(_Tpvec, _Tpwvec, _Tp, intrin)     \
    inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1)   \
    {                                                                 \
        b0.val = intrin(a.val);                                       \
        b1.val = intrin(__lasx_xvpermi_q(a.val, a.val, 0x11));        \
    }                                                                 \
    inline _Tpwvec v_expand_low(const _Tpvec& a)                      \
    { return _Tpwvec(intrin(a.val)); }                                \
    inline _Tpwvec v_expand_high(const _Tpvec& a)                     \
    { return _Tpwvec(intrin(__lasx_xvpermi_q(a.val, a.val, 0x11))); } \
    inline _Tpwvec v256_load_expand(const _Tp* ptr)                   \
    {                                                                 \
        __m128i a = __lsx_vld(ptr, 0);                                \
        return _Tpwvec(intrin(*((__m256i*)&a)));                      \
    }

OPENCV_HAL_IMPL_LASX_EXPAND(v_uint8x32,  v_uint16x16, uchar,    __lasx_vext2xv_hu_bu)
OPENCV_HAL_IMPL_LASX_EXPAND(v_int8x32,   v_int16x16,  schar,    __lasx_vext2xv_h_b)
OPENCV_HAL_IMPL_LASX_EXPAND(v_uint16x16, v_uint32x8,  ushort,   __lasx_vext2xv_wu_hu)
OPENCV_HAL_IMPL_LASX_EXPAND(v_int16x16,  v_int32x8,   short,    __lasx_vext2xv_w_h)
OPENCV_HAL_IMPL_LASX_EXPAND(v_uint32x8,  v_uint64x4,  unsigned, __lasx_vext2xv_du_wu)
OPENCV_HAL_IMPL_LASX_EXPAND(v_int32x8,   v_int64x4,   int,      __lasx_vext2xv_d_w)

#define OPENCV_HAL_IMPL_LASX_EXPAND_Q(_Tpvec, _Tp, intrin)   \
    inline _Tpvec v256_load_expand_q(const _Tp* ptr)         \
    {                                                        \
        __m128i a = __lsx_vld(ptr, 0);                       \
        return _Tpvec(intrin(*((__m256i*)&a)));              \
    }

OPENCV_HAL_IMPL_LASX_EXPAND_Q(v_uint32x8, uchar, __lasx_vext2xv_wu_bu)
OPENCV_HAL_IMPL_LASX_EXPAND_Q(v_int32x8,  schar, __lasx_vext2xv_w_b)

/* pack */
// 16
inline v_int8x32 v_pack(const v_int16x16& a, const v_int16x16& b)
{ return v_int8x32(_v256_shuffle_odd_64(_lasx_packs_h(a.val, b.val))); }

inline v_uint8x32 v_pack(const v_uint16x16& a, const v_uint16x16& b)
{ return v_uint8x32(_v256_shuffle_odd_64(__lasx_xvssrlrni_bu_h(b.val, a.val, 0))); }

inline v_uint8x32 v_pack_u(const v_int16x16& a, const v_int16x16& b)
{
    return v_uint8x32(_v256_shuffle_odd_64(_lasx_packus_h(a.val, b.val)));
}

inline void v_pack_store(schar* ptr, const v_int16x16& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(uchar *ptr, const v_uint16x16& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_u_store(uchar* ptr, const v_int16x16& a)
{ v_store_low(ptr, v_pack_u(a, a)); }

template<int n> inline
v_uint8x32 v_rshr_pack(const v_uint16x16& a, const v_uint16x16& b)
{
    __m256i res = __lasx_xvssrlrni_bu_h(b.val, a.val, n);
    return v_uint8x32(_v256_shuffle_odd_64(res));
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x16& a)
{
    __m256i res = __lasx_xvssrlrni_bu_h(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

template<int n> inline
v_uint8x32 v_rshr_pack_u(const v_int16x16& a, const v_int16x16& b)
{
    __m256i res = __lasx_xvssrarni_bu_h(b.val, a.val, n);
    return v_uint8x32(_v256_shuffle_odd_64(res));
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x16& a)
{
    __m256i res = __lasx_xvssrarni_bu_h(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

template<int n> inline
v_int8x32 v_rshr_pack(const v_int16x16& a, const v_int16x16& b)
{
    __m256i res = __lasx_xvssrarni_b_h(b.val, a.val, n);
    return v_int8x32(_v256_shuffle_odd_64(res));
}

template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x16& a)
{
    __m256i res = __lasx_xvssrarni_b_h(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

// 32
inline v_int16x16 v_pack(const v_int32x8& a, const v_int32x8& b)
{ return v_int16x16(_v256_shuffle_odd_64(_lasx_packs_w(a.val, b.val))); }

inline v_uint16x16 v_pack(const v_uint32x8& a, const v_uint32x8& b)
{ return v_uint16x16(_v256_shuffle_odd_64(_v256_packs_epu32(a.val, b.val))); }

inline v_uint16x16 v_pack_u(const v_int32x8& a, const v_int32x8& b)
{ return v_uint16x16(_v256_shuffle_odd_64(_lasx_packus_w(a.val, b.val))); }

inline void v_pack_store(short* ptr, const v_int32x8& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(ushort* ptr, const v_uint32x8& a)
{
    __m256i res = __lasx_xvssrlrni_hu_w(a.val, a.val, 0);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

inline void v_pack_u_store(ushort* ptr, const v_int32x8& a)
{ v_store_low(ptr, v_pack_u(a, a)); }

template<int n> inline
v_uint16x16 v_rshr_pack(const v_uint32x8& a, const v_uint32x8& b)
{ return v_uint16x16(_v256_shuffle_odd_64(__lasx_xvssrlrni_hu_w(b.val, a.val, n))); }

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x8& a)
{
    __m256i res = __lasx_xvssrlrni_hu_w(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

template<int n> inline
v_uint16x16 v_rshr_pack_u(const v_int32x8& a, const v_int32x8& b)
{ return v_uint16x16(_v256_shuffle_odd_64(__lasx_xvssrarni_hu_w(b.val, a.val, n))); }

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x8& a)
{
    __m256i res = __lasx_xvssrarni_hu_w(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

template<int n> inline
v_int16x16 v_rshr_pack(const v_int32x8& a, const v_int32x8& b)
{ return v_int16x16(_v256_shuffle_odd_64(__lasx_xvssrarni_h_w(b.val, a.val, n))); }

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x8& a)
{
    __m256i res = __lasx_xvssrarni_h_w(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

// 64
// Non-saturating pack
inline v_uint32x8 v_pack(const v_uint64x4& a, const v_uint64x4& b)
{
    __m256i ab = __lasx_xvpickev_w(b.val, a.val);
    return v_uint32x8(_v256_shuffle_odd_64(ab));
}

inline v_int32x8 v_pack(const v_int64x4& a, const v_int64x4& b)
{ return v_reinterpret_as_s32(v_pack(v_reinterpret_as_u64(a), v_reinterpret_as_u64(b))); }

inline void v_pack_store(unsigned* ptr, const v_uint64x4& a)
{
    __m256i a0 = __lasx_xvshuf4i_w(a.val, 0x08);
    v_store_low(ptr, v_uint32x8(_v256_shuffle_odd_64(a0)));
}

inline void v_pack_store(int* ptr, const v_int64x4& b)
{ v_pack_store((unsigned*)ptr, v_reinterpret_as_u64(b)); }

template<int n> inline
v_uint32x8 v_rshr_pack(const v_uint64x4& a, const v_uint64x4& b)
{ return v_uint32x8(_v256_shuffle_odd_64(__lasx_xvsrlrni_w_d(b.val, a.val, n))); }

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x4& a)
{
    __m256i res = __lasx_xvsrlrni_w_d(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

template<int n> inline
v_int32x8 v_rshr_pack(const v_int64x4& a, const v_int64x4& b)
{ return v_int32x8(_v256_shuffle_odd_64(__lasx_xvsrarni_w_d(b.val, a.val, n))); }

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x4& a)
{
    __m256i res = __lasx_xvsrarni_w_d(a.val, a.val, n);
    __lasx_xvstelm_d(res, ptr, 0, 0);
    __lasx_xvstelm_d(res, ptr, 8, 2);
}

// pack boolean
inline v_uint8x32 v_pack_b(const v_uint16x16& a, const v_uint16x16& b)
{
    __m256i ab = _lasx_packs_h(a.val, b.val);
    return v_uint8x32(_v256_shuffle_odd_64(ab));
}

inline v_uint8x32 v_pack_b(const v_uint32x8& a, const v_uint32x8& b,
                           const v_uint32x8& c, const v_uint32x8& d)
{
    __m256i ab = _lasx_packs_w(a.val, b.val);
    __m256i cd = _lasx_packs_w(c.val, d.val);

    __m256i abcd = _v256_shuffle_odd_64(_lasx_packs_h(ab, cd));
    return v_uint8x32(__lasx_xvshuf4i_w(abcd, 0xd8));
}

inline v_uint8x32 v_pack_b(const v_uint64x4& a, const v_uint64x4& b, const v_uint64x4& c,
                           const v_uint64x4& d, const v_uint64x4& e, const v_uint64x4& f,
                           const v_uint64x4& g, const v_uint64x4& h)
{
    __m256i ab = _lasx_packs_w(a.val, b.val);
    __m256i cd = _lasx_packs_w(c.val, d.val);
    __m256i ef = _lasx_packs_w(e.val, f.val);
    __m256i gh = _lasx_packs_w(g.val, h.val);

    __m256i abcd = _lasx_packs_w(ab, cd);
    __m256i efgh = _lasx_packs_w(ef, gh);
    __m256i pkall = _v256_shuffle_odd_64(_lasx_packs_h(abcd, efgh));

    __m256i rev = _v256_alignr_b(pkall, pkall, 8);
    return v_uint8x32(__lasx_xvilvl_h(rev, pkall));
}

/* Recombine */
// its up there with load and store operations

/* Extract */
#define OPENCV_HAL_IMPL_LASX_EXTRACT(_Tpvec)                    \
    template<int s>                                             \
    inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)   \
    { return v_rotate_right<s>(a, b); }

OPENCV_HAL_IMPL_LASX_EXTRACT(v_uint8x32)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_int8x32)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_uint16x16)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_int16x16)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_uint32x8)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_int32x8)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_uint64x4)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_int64x4)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_float32x8)
OPENCV_HAL_IMPL_LASX_EXTRACT(v_float64x4)

template<int i>
inline uchar v_extract_n(v_uint8x32 a)
{
    return (uchar)_v256_extract_b<i>(a.val);
}

template<int i>
inline schar v_extract_n(v_int8x32 a)
{
    return (schar)v_extract_n<i>(v_reinterpret_as_u8(a));
}

template<int i>
inline ushort v_extract_n(v_uint16x16 a)
{
    return (ushort)_v256_extract_h<i>(a.val);
}

template<int i>
inline short v_extract_n(v_int16x16 a)
{
    return (short)v_extract_n<i>(v_reinterpret_as_u16(a));
}

template<int i>
inline uint v_extract_n(v_uint32x8 a)
{
    return (uint)_v256_extract_w<i>(a.val);
}

template<int i>
inline int v_extract_n(v_int32x8 a)
{
    return (int)v_extract_n<i>(v_reinterpret_as_u32(a));
}

template<int i>
inline uint64 v_extract_n(v_uint64x4 a)
{
    return (uint64)_v256_extract_d<i>(a.val);
}

template<int i>
inline int64 v_extract_n(v_int64x4 v)
{
    return (int64)v_extract_n<i>(v_reinterpret_as_u64(v));
}

template<int i>
inline float v_extract_n(v_float32x8 v)
{
    union { uint iv; float fv; } d;
    d.iv = v_extract_n<i>(v_reinterpret_as_u32(v));
    return d.fv;
}

template<int i>
inline double v_extract_n(v_float64x4 v)
{
    union { uint64 iv; double dv; } d;
    d.iv = v_extract_n<i>(v_reinterpret_as_u64(v));
    return d.dv;
}

template<int i>
inline v_uint32x8 v_broadcast_element(v_uint32x8 a)
{
    static const __m256i perm = __lasx_xvreplgr2vr_w((char)i);
    return v_uint32x8(__lasx_xvperm_w(a.val, perm));
}

template<int i>
inline v_int32x8 v_broadcast_element(const v_int32x8 &a)
{ return v_reinterpret_as_s32(v_broadcast_element<i>(v_reinterpret_as_u32(a))); }

template<int i>
inline v_float32x8 v_broadcast_element(const v_float32x8 &a)
{ return v_reinterpret_as_f32(v_broadcast_element<i>(v_reinterpret_as_u32(a))); }

///////////////////// load deinterleave /////////////////////////////

inline void v_load_deinterleave(const uchar* ptr, v_uint8x32& a, v_uint8x32& b)
{
    __m256i t0 = __lasx_xvld(ptr, 0);
    __m256i t1 = __lasx_xvld(ptr, 32);

    __m256i p0 = __lasx_xvpickev_b(t1, t0);
    __m256i p1 = __lasx_xvpickod_b(t1, t0);

    a.val = __lasx_xvpermi_d(p0, 0xd8);
    b.val = __lasx_xvpermi_d(p1, 0xd8);
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x16& a, v_uint16x16& b )
{
    __m256i t0 = __lasx_xvld(ptr, 0);
    __m256i t1 = __lasx_xvld(ptr, 32);

    __m256i p0 = __lasx_xvpickev_h(t1, t0);
    __m256i p1 = __lasx_xvpickod_h(t1, t0);

    a.val = __lasx_xvpermi_d(p0, 0xd8);
    b.val = __lasx_xvpermi_d(p1, 0xd8);
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x8& a, v_uint32x8& b )
{
    __m256i t0 = __lasx_xvld(ptr, 0);
    __m256i t1 = __lasx_xvld(ptr, 32);

    __m256i p0 = __lasx_xvpickev_w(t1, t0);
    __m256i p1 = __lasx_xvpickod_w(t1, t0);

    a.val = __lasx_xvpermi_d(p0, 0xd8);
    b.val = __lasx_xvpermi_d(p1, 0xd8);
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x4& a, v_uint64x4& b )
{
    __m256i ab0 = __lasx_xvld(ptr, 0);
    __m256i ab1 = __lasx_xvld(ptr, 32);

    __m256i pl = __lasx_xvpermi_q(ab0, ab1, 0x02);
    __m256i ph = __lasx_xvpermi_q(ab0, ab1, 0x13);
    __m256i a0 = __lasx_xvilvl_d(ph, pl);
    __m256i b0 = __lasx_xvilvh_d(ph, pl);
    a = v_uint64x4(a0);
    b = v_uint64x4(b0);
}

inline void v_load_deinterleave( const uchar* ptr, v_uint8x32& a, v_uint8x32& b, v_uint8x32& c )
{
    __m256i bgr0 = __lasx_xvld(ptr, 0);
    __m256i bgr1 = __lasx_xvld(ptr, 32);
    __m256i bgr2 = __lasx_xvld(ptr, 64);

    __m256i s02_low = __lasx_xvpermi_q(bgr0, bgr2, 0x02);
    __m256i s02_high = __lasx_xvpermi_q(bgr0, bgr2, 0x13);

    const __m256i m0 = _v256_setr_b(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
                                    0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    const __m256i m1 = _v256_setr_b(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
                                    -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1);

    __m256i b0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(s02_low, s02_high, m0), bgr1, m1);
    __m256i g0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(s02_high, s02_low, m1), bgr1, m0);
    __m256i r0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(bgr1, s02_low, m0), s02_high, m1);

    const __m256i
    sh_b = _v256_setr_b(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13,
                        0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13),
    sh_g = _v256_setr_b(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14,
                        1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14),
    sh_r = _v256_setr_b(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15,
                        2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    b0 = __lasx_xvshuf_b(b0, b0, sh_b);
    g0 = __lasx_xvshuf_b(g0, g0, sh_g);
    r0 = __lasx_xvshuf_b(r0, r0, sh_r);

    a = v_uint8x32(b0);
    b = v_uint8x32(g0);
    c = v_uint8x32(r0);
}

inline void v_load_deinterleave( const ushort* ptr, v_uint16x16& a, v_uint16x16& b, v_uint16x16& c )
{
    __m256i bgr0 = __lasx_xvld(ptr, 0);
    __m256i bgr1 = __lasx_xvld(ptr, 32);
    __m256i bgr2 = __lasx_xvld(ptr, 64);

    __m256i s02_low = __lasx_xvpermi_q(bgr0, bgr2, 0x02);
    __m256i s02_high = __lasx_xvpermi_q(bgr0, bgr2, 0x13);

    const __m256i m0 = _v256_setr_b(0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
                                    0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);
    const __m256i m1 = _v256_setr_b(0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
                                    -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);
    __m256i b0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(s02_low, s02_high, m0), bgr1, m1);
    __m256i g0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(bgr1, s02_low, m0), s02_high, m1);
    __m256i r0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(s02_high, s02_low, m1), bgr1, m0);
    const __m256i sh_b = _v256_setr_b(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
                                      0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m256i sh_g = _v256_setr_b(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13,
                                      2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    const __m256i sh_r = _v256_setr_b(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
                                      4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    b0 = __lasx_xvshuf_b(b0, b0, sh_b);
    g0 = __lasx_xvshuf_b(g0, g0, sh_g);
    r0 = __lasx_xvshuf_b(r0, r0, sh_r);

    a = v_uint16x16(b0);
    b = v_uint16x16(g0);
    c = v_uint16x16(r0);
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x8& a, v_uint32x8& b, v_uint32x8& c )
{
    __m256i bgr0 = __lasx_xvld(ptr, 0);
    __m256i bgr1 = __lasx_xvld(ptr, 32);
    __m256i bgr2 = __lasx_xvld(ptr, 64);

    __m256i s02_low = __lasx_xvpermi_q(bgr0, bgr2, 0x02);
    __m256i s02_high = __lasx_xvpermi_q(bgr0, bgr2, 0x13);

    __m256i m24 = _v256_set_w(0, 0, -1, 0, 0, -1, 0, 0);
    __m256i m92 = _v256_set_w(-1, 0, 0, -1, 0, 0, -1, 0);
    __m256i b0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(s02_low, s02_high, m24), bgr1, m92);
    __m256i g0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(s02_high, s02_low, m92), bgr1, m24);
    __m256i r0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(bgr1, s02_low, m24), s02_high, m92);

    b0 = __lasx_xvshuf4i_w(b0, 0x6c);
    g0 = __lasx_xvshuf4i_w(g0, 0xb1);
    r0 = __lasx_xvshuf4i_w(r0, 0xc6);

    a = v_uint32x8(b0);
    b = v_uint32x8(g0);
    c = v_uint32x8(r0);
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x4& a, v_uint64x4& b, v_uint64x4& c )
{
    __m256i bgr0 = __lasx_xvld(ptr, 0);
    __m256i bgr1 = __lasx_xvld(ptr, 32);
    __m256i bgr2 = __lasx_xvld(ptr, 64);

    __m256i s01 = __lasx_xvpermi_q(bgr0, bgr1, 0x12); // get bgr0 low 128 and bgr1 high 128
    __m256i s12 = __lasx_xvpermi_q(bgr1, bgr2, 0x12);
    __m256i s20r = __lasx_xvpermi_d(__lasx_xvpermi_q(bgr2, bgr0, 0x12), 0x1b);
    __m256i b0 = __lasx_xvilvl_d(s20r, s01);
    __m256i g0 = _v256_alignr_b(s12, s01, 8);
    __m256i r0 = __lasx_xvilvh_d(s12, s20r);

    a = v_uint64x4(b0);
    b = v_uint64x4(g0);
    c = v_uint64x4(r0);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x32& a, v_uint8x32& b, v_uint8x32& c, v_uint8x32& d)
{
    __m256i t0 = __lasx_xvld(ptr, 0);
    __m256i t1 = __lasx_xvld(ptr, 32);
    __m256i t2 = __lasx_xvld(ptr, 64);
    __m256i t3 = __lasx_xvld(ptr, 96);

    const __m256i sh = _v256_setr_w(0, 4, 1, 5, 2, 6, 3, 7);
    __m256i ac_lo = __lasx_xvpickev_b(t1, t0);
    __m256i bd_lo = __lasx_xvpickod_b(t1, t0);
    __m256i ac_hi = __lasx_xvpickev_b(t3, t2);
    __m256i bd_hi = __lasx_xvpickod_b(t3, t2);

    __m256i a_pre = __lasx_xvpickev_b(ac_hi, ac_lo);
    __m256i c_pre = __lasx_xvpickod_b(ac_hi, ac_lo);
    __m256i b_pre = __lasx_xvpickev_b(bd_hi, bd_lo);
    __m256i d_pre = __lasx_xvpickod_b(bd_hi, bd_lo);

    a.val = __lasx_xvperm_w(a_pre, sh);
    b.val = __lasx_xvperm_w(b_pre, sh);
    c.val = __lasx_xvperm_w(c_pre, sh);
    d.val = __lasx_xvperm_w(d_pre, sh);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x16& a, v_uint16x16& b, v_uint16x16& c, v_uint16x16& d)
{
    __m256i t0 = __lasx_xvld(ptr, 0);
    __m256i t1 = __lasx_xvld(ptr, 32);
    __m256i t2 = __lasx_xvld(ptr, 64);
    __m256i t3 = __lasx_xvld(ptr, 96);

    const __m256i sh = _v256_setr_w(0, 4, 1, 5, 2, 6, 3, 7);
    __m256i ac_lo = __lasx_xvpickev_h(t1, t0);
    __m256i bd_lo = __lasx_xvpickod_h(t1, t0);
    __m256i ac_hi = __lasx_xvpickev_h(t3, t2);
    __m256i bd_hi = __lasx_xvpickod_h(t3, t2);

    __m256i a_pre = __lasx_xvpickev_h(ac_hi, ac_lo);
    __m256i c_pre = __lasx_xvpickod_h(ac_hi, ac_lo);
    __m256i b_pre = __lasx_xvpickev_h(bd_hi, bd_lo);
    __m256i d_pre = __lasx_xvpickod_h(bd_hi, bd_lo);

    a.val = __lasx_xvperm_w(a_pre, sh);
    b.val = __lasx_xvperm_w(b_pre, sh);
    c.val = __lasx_xvperm_w(c_pre, sh);
    d.val = __lasx_xvperm_w(d_pre, sh);
}

inline void v_load_deinterleave( const unsigned* ptr, v_uint32x8& a, v_uint32x8& b, v_uint32x8& c, v_uint32x8& d )
{
    __m256i p0 = __lasx_xvld(ptr, 0);
    __m256i p1 = __lasx_xvld(ptr, 32);
    __m256i p2 = __lasx_xvld(ptr, 64);
    __m256i p3 = __lasx_xvld(ptr, 96);

    __m256i p01l = __lasx_xvilvl_w(p1, p0);
    __m256i p01h = __lasx_xvilvh_w(p1, p0);
    __m256i p23l = __lasx_xvilvl_w(p3, p2);
    __m256i p23h = __lasx_xvilvh_w(p3, p2);

    __m256i pll = __lasx_xvpermi_q(p01l, p23l, 0x02);
    __m256i plh = __lasx_xvpermi_q(p01l, p23l, 0x13);
    __m256i phl = __lasx_xvpermi_q(p01h, p23h, 0x02);
    __m256i phh = __lasx_xvpermi_q(p01h, p23h, 0x13);

    __m256i b0 = __lasx_xvilvl_w(plh, pll);
    __m256i g0 = __lasx_xvilvh_w(plh, pll);
    __m256i r0 = __lasx_xvilvl_w(phh, phl);
    __m256i a0 = __lasx_xvilvh_w(phh, phl);

    a = v_uint32x8(b0);
    b = v_uint32x8(g0);
    c = v_uint32x8(r0);
    d = v_uint32x8(a0);
}

inline void v_load_deinterleave( const uint64* ptr, v_uint64x4& a, v_uint64x4& b, v_uint64x4& c, v_uint64x4& d )
{
    __m256i bgra0 = __lasx_xvld(ptr, 0);
    __m256i bgra1 = __lasx_xvld(ptr, 32);
    __m256i bgra2 = __lasx_xvld(ptr, 64);
    __m256i bgra3 = __lasx_xvld(ptr, 96);

    __m256i l02 = __lasx_xvpermi_q(bgra0, bgra2, 0x02);
    __m256i h02 = __lasx_xvpermi_q(bgra0, bgra2, 0x13);
    __m256i l13 = __lasx_xvpermi_q(bgra1, bgra3, 0x02);
    __m256i h13 = __lasx_xvpermi_q(bgra1, bgra3, 0x13);

    __m256i b0 = __lasx_xvilvl_d(l13, l02);
    __m256i g0 = __lasx_xvilvh_d(l13, l02);
    __m256i r0 = __lasx_xvilvl_d(h13, h02);
    __m256i a0 = __lasx_xvilvh_d(h13, h02);

    a = v_uint64x4(b0);
    b = v_uint64x4(g0);
    c = v_uint64x4(r0);
    d = v_uint64x4(a0);
}

///////////////////////////// store interleave /////////////////////////////////////

inline void v_store_interleave( uchar* ptr, const v_uint8x32& x, const v_uint8x32& y,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i xy_l = __lasx_xvilvl_b(y.val, x.val);
    __m256i xy_h = __lasx_xvilvh_b(y.val, x.val);

    __m256i xy0 = __lasx_xvpermi_q(xy_h, xy_l, 0 + 2*16);
    __m256i xy1 = __lasx_xvpermi_q(xy_h, xy_l, 1 + 3*16);

    __lasx_xvst(xy0, (__m256i*)ptr, 0);
    __lasx_xvst(xy1, (__m256i*)ptr, 32*1);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x16& x, const v_uint16x16& y,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i xy_l = __lasx_xvilvl_h(y.val, x.val);
    __m256i xy_h = __lasx_xvilvh_h(y.val, x.val);

    __m256i xy0 = __lasx_xvpermi_q(xy_h, xy_l, 0 + 2*16);
    __m256i xy1 = __lasx_xvpermi_q(xy_h, xy_l, 1 + 3*16);

    __lasx_xvst(xy0, (__m256i*)ptr, 0);
    __lasx_xvst(xy1, (__m256i*)ptr, 16*2);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x8& x, const v_uint32x8& y,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i xy_l = __lasx_xvilvl_w(y.val, x.val);
    __m256i xy_h = __lasx_xvilvh_w(y.val, x.val);

    __m256i xy0 = __lasx_xvpermi_q(xy_h, xy_l, 0 + 2*16);
    __m256i xy1 = __lasx_xvpermi_q(xy_h, xy_l, 1 + 3*16);

    __lasx_xvst(xy0, (__m256i*)ptr, 0);
    __lasx_xvst(xy1, (__m256i*)ptr, 8*4);
}

inline void v_store_interleave( uint64* ptr, const v_uint64x4& x, const v_uint64x4& y,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i xy_l = __lasx_xvilvl_d(y.val, x.val);
    __m256i xy_h = __lasx_xvilvh_d(y.val, x.val);

    __m256i xy0 = __lasx_xvpermi_q(xy_h, xy_l, 0 + 2*16);
    __m256i xy1 = __lasx_xvpermi_q(xy_h, xy_l, 1 + 3*16);

    __lasx_xvst(xy0, (__m256i*)ptr, 0);
    __lasx_xvst(xy1, (__m256i*)ptr, 4*8);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x32& a, const v_uint8x32& b, const v_uint8x32& c,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    const __m256i sh_b = _v256_setr_b(
            0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5,
            0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    const __m256i sh_g = _v256_setr_b(
            5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10,
            5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    const __m256i sh_r = _v256_setr_b(
            10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15,
            10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);

    __m256i b0 = __lasx_xvshuf_b(a.val, a.val, sh_b);
    __m256i g0 = __lasx_xvshuf_b(b.val, b.val, sh_g);
    __m256i r0 = __lasx_xvshuf_b(c.val, c.val, sh_r);

    const __m256i m0 = _v256_setr_b(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0,
                                    0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    const __m256i m1 = _v256_setr_b(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0,
                                    0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);

    __m256i p0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(b0, g0, m0), r0, m1);
    __m256i p1 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(g0, r0, m0), b0, m1);
    __m256i p2 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(r0, b0, m0), g0, m1);

    __m256i bgr0 = __lasx_xvpermi_q(p1, p0, 0 + 2*16);
    __m256i bgr1 = __lasx_xvpermi_q(p0, p2, 0 + 3*16);
    __m256i bgr2 = __lasx_xvpermi_q(p2, p1, 1 + 3*16);

    __lasx_xvst(bgr0, (__m256i*)ptr, 0);
    __lasx_xvst(bgr1, (__m256i*)ptr, 32);
    __lasx_xvst(bgr2, (__m256i*)ptr, 64);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x16& a, const v_uint16x16& b, const v_uint16x16& c,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    const __m256i sh_b = _v256_setr_b(
         0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11,
         0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m256i sh_g = _v256_setr_b(
         10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5,
         10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    const __m256i sh_r = _v256_setr_b(
         4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15,
         4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);

    __m256i b0 = __lasx_xvshuf_b(a.val, a.val, sh_b);
    __m256i g0 = __lasx_xvshuf_b(b.val, b.val, sh_g);
    __m256i r0 = __lasx_xvshuf_b(c.val, c.val, sh_r);

    const __m256i m0 = _v256_setr_b(0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1,
                                    0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0);
    const __m256i m1 = _v256_setr_b(0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0,
                                    -1, -1, 0, 0, 0, 0, -1, -1, 0, 0, 0, 0, -1, -1, 0, 0);

    __m256i p0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(b0, g0, m0), r0, m1);
    __m256i p1 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(g0, r0, m0), b0, m1);
    __m256i p2 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(r0, b0, m0), g0, m1);

    __m256i bgr0 = __lasx_xvpermi_q(p2, p0, 0 + 2*16);
    __m256i bgr2 = __lasx_xvpermi_q(p2, p0, 1 + 3*16);

    __lasx_xvst(bgr0, (__m256i*)ptr, 0);
    __lasx_xvst(p1,   (__m256i*)ptr, 16*2);
    __lasx_xvst(bgr2, (__m256i*)ptr, 32*2);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x8& a, const v_uint32x8& b, const v_uint32x8& c,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i b0 = __lasx_xvshuf4i_w(a.val, 0x6c);
    __m256i g0 = __lasx_xvshuf4i_w(b.val, 0xb1);
    __m256i r0 = __lasx_xvshuf4i_w(c.val, 0xc6);

    __m256i bitmask_1 = _v256_set_w(-1, 0, 0, -1, 0, 0, -1, 0);
    __m256i bitmask_2 = _v256_set_w(0, 0, -1, 0, 0, -1, 0, 0);

    __m256i p0 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(b0, g0, bitmask_1), r0, bitmask_2);
    __m256i p1 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(g0, r0, bitmask_1), b0, bitmask_2);
    __m256i p2 = __lasx_xvbitsel_v(__lasx_xvbitsel_v(r0, b0, bitmask_1), g0, bitmask_2);

    __m256i bgr0 = __lasx_xvpermi_q(p1, p0, 0 + 2*16);
    __m256i bgr2 = __lasx_xvpermi_q(p1, p0, 1 + 3*16);

    __lasx_xvst(bgr0, (__m256i*)ptr, 0);
    __lasx_xvst(p2,   (__m256i*)ptr, 8*4);
    __lasx_xvst(bgr2, (__m256i*)ptr, 16*4);
}

inline void v_store_interleave( uint64* ptr, const v_uint64x4& a, const v_uint64x4& b, const v_uint64x4& c,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i s01 = __lasx_xvilvl_d(b.val, a.val);
    __m256i s12 = __lasx_xvilvh_d(c.val, b.val);
    __m256i s20 = __lasx_xvpermi_w(a.val, c.val, 0xe4);

    __m256i bgr0 = __lasx_xvpermi_q(s20, s01, 0 + 2*16);
    __m256i bgr1 = __lasx_xvpermi_q(s01, s12, 0x30);
    __m256i bgr2 = __lasx_xvpermi_q(s12, s20, 1 + 3*16);

    __lasx_xvst(bgr0, (__m256i*)ptr, 0);
    __lasx_xvst(bgr1, (__m256i*)ptr, 4*8);
    __lasx_xvst(bgr2, (__m256i*)ptr, 8*8);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x32& a, const v_uint8x32& b,
                                const v_uint8x32& c, const v_uint8x32& d,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i bg0 = __lasx_xvilvl_b(b.val, a.val);
    __m256i bg1 = __lasx_xvilvh_b(b.val, a.val);
    __m256i ra0 = __lasx_xvilvl_b(d.val, c.val);
    __m256i ra1 = __lasx_xvilvh_b(d.val, c.val);

    __m256i bgra0_ = __lasx_xvilvl_h(ra0, bg0);
    __m256i bgra1_ = __lasx_xvilvh_h(ra0, bg0);
    __m256i bgra2_ = __lasx_xvilvl_h(ra1, bg1);
    __m256i bgra3_ = __lasx_xvilvh_h(ra1, bg1);

    __m256i bgra0 = __lasx_xvpermi_q(bgra1_, bgra0_, 0 + 2*16);
    __m256i bgra2 = __lasx_xvpermi_q(bgra1_, bgra0_, 1 + 3*16);
    __m256i bgra1 = __lasx_xvpermi_q(bgra3_, bgra2_, 0 + 2*16);
    __m256i bgra3 = __lasx_xvpermi_q(bgra3_, bgra2_, 1 + 3*16);

    __lasx_xvst(bgra0, (__m256i*)ptr, 0);
    __lasx_xvst(bgra1, (__m256i*)ptr, 32);
    __lasx_xvst(bgra2, (__m256i*)ptr, 64);
    __lasx_xvst(bgra3, (__m256i*)ptr, 96);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x16& a, const v_uint16x16& b,
                                const v_uint16x16& c, const v_uint16x16& d,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i bg0 = __lasx_xvilvl_h(b.val, a.val);
    __m256i bg1 = __lasx_xvilvh_h(b.val, a.val);
    __m256i ra0 = __lasx_xvilvl_h(d.val, c.val);
    __m256i ra1 = __lasx_xvilvh_h(d.val, c.val);

    __m256i bgra0_ = __lasx_xvilvl_w(ra0, bg0);
    __m256i bgra1_ = __lasx_xvilvh_w(ra0, bg0);
    __m256i bgra2_ = __lasx_xvilvl_w(ra1, bg1);
    __m256i bgra3_ = __lasx_xvilvh_w(ra1, bg1);

    __m256i bgra0 = __lasx_xvpermi_q(bgra1_, bgra0_, 0 + 2*16);
    __m256i bgra2 = __lasx_xvpermi_q(bgra1_, bgra0_, 1 + 3*16);
    __m256i bgra1 = __lasx_xvpermi_q(bgra3_, bgra2_, 0 + 2*16);
    __m256i bgra3 = __lasx_xvpermi_q(bgra3_, bgra2_, 1 + 3*16);

    __lasx_xvst(bgra0, (__m256i*)ptr, 0);
    __lasx_xvst(bgra1, (__m256i*)ptr, 16*2);
    __lasx_xvst(bgra2, (__m256i*)ptr, 32*2);
    __lasx_xvst(bgra3, (__m256i*)ptr, 48*2);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x8& a, const v_uint32x8& b,
                                const v_uint32x8& c, const v_uint32x8& d,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i bg0 = __lasx_xvilvl_w(b.val, a.val);
    __m256i bg1 = __lasx_xvilvh_w(b.val, a.val);
    __m256i ra0 = __lasx_xvilvl_w(d.val, c.val);
    __m256i ra1 = __lasx_xvilvh_w(d.val, c.val);

    __m256i bgra0_ = __lasx_xvilvl_d(ra0, bg0);
    __m256i bgra1_ = __lasx_xvilvh_d(ra0, bg0);
    __m256i bgra2_ = __lasx_xvilvl_d(ra1, bg1);
    __m256i bgra3_ = __lasx_xvilvh_d(ra1, bg1);

    __m256i bgra0 = __lasx_xvpermi_q(bgra1_, bgra0_, 0 + 2*16);
    __m256i bgra2 = __lasx_xvpermi_q(bgra1_, bgra0_, 1 + 3*16);
    __m256i bgra1 = __lasx_xvpermi_q(bgra3_, bgra2_, 0 + 2*16);
    __m256i bgra3 = __lasx_xvpermi_q(bgra3_, bgra2_, 1 + 3*16);

    __lasx_xvst(bgra0, (__m256i*)ptr, 0);
    __lasx_xvst(bgra1, (__m256i*)ptr, 8*4);
    __lasx_xvst(bgra2, (__m256i*)ptr, 16*4);
    __lasx_xvst(bgra3, (__m256i*)ptr, 24*4);
}

inline void v_store_interleave( uint64* ptr, const v_uint64x4& a, const v_uint64x4& b,
                                const v_uint64x4& c, const v_uint64x4& d,
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED )
{
    __m256i bg0 = __lasx_xvilvl_d(b.val, a.val);
    __m256i bg1 = __lasx_xvilvh_d(b.val, a.val);
    __m256i ra0 = __lasx_xvilvl_d(d.val, c.val);
    __m256i ra1 = __lasx_xvilvh_d(d.val, c.val);

    __m256i bgra0 = __lasx_xvpermi_q(ra0, bg0, 0 + 2*16);
    __m256i bgra1 = __lasx_xvpermi_q(ra1, bg1, 0 + 2*16);
    __m256i bgra2 = __lasx_xvpermi_q(ra0, bg0, 1 + 3*16);
    __m256i bgra3 = __lasx_xvpermi_q(ra1, bg1, 1 + 3*16);

    __lasx_xvst(bgra0, (__m256i*)ptr, 0);
    __lasx_xvst(bgra1, (__m256i*)(ptr), 4*8);
    __lasx_xvst(bgra2, (__m256i*)(ptr), 8*8);
    __lasx_xvst(bgra3, (__m256i*)(ptr), 12*8);
}


#define OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1) \
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
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    v_store_interleave((_Tp1*)ptr, a1, b1/*, mode*/);      \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, const _Tpvec0& c0, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1/*, mode*/);  \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, const _Tpvec0& d0, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    _Tpvec1 d1 = v_reinterpret_as_##suffix1(d0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1/*, mode*/); \
}

OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(v_int8x32, schar, s8, v_uint8x32, uchar, u8)
OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(v_int16x16, short, s16, v_uint16x16, ushort, u16)
OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(v_int32x8, int, s32, v_uint32x8, unsigned, u32)
OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(v_float32x8, float, f32, v_uint32x8, unsigned, u32)
OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(v_int64x4, int64, s64, v_uint64x4, uint64, u64)
OPENCV_HAL_IMPL_LASX_LOADSTORE_INTERLEAVE(v_float64x4, double, f64, v_uint64x4, uint64, u64)

//
// FP16
//

inline v_float32x8 v256_load_expand(const hfloat* ptr)
{
#if CV_FP16
    //1-load128, 2-permi, 3-cvt
   return v_float32x8(__lasx_xvfcvtl_s_h(__lasx_xvpermi_d(__lsx_vld((const __m128i*)ptr, 0), 0x10)));
#else
    float CV_DECL_ALIGNED(32) buf[8];
    for (int i = 0; i < 8; i++)
        buf[i] = (float)ptr[i];
    return v256_load_aligned(buf);
#endif
}

inline void v_pack_store(hfloat* ptr, const v_float32x8& a)
{
#if CV_FP16
    __m256i ah = __lasx_xvfcvt_h_s(a.val, a.val);
    __lsx_vst((_m128i)ah, ptr, 0);
#else
    float CV_DECL_ALIGNED(32) buf[8];
    v_store_aligned(buf, a);
    for (int i = 0; i < 8; i++)
        ptr[i] = hfloat(buf[i]);
#endif
}

//
// end of FP16
//

inline void v256_cleanup() {}

#include "intrin_math.hpp"
inline v_float32x8 v_exp(v_float32x8 x) { return v_exp_default_32f<v_float32x8, v_int32x8>(x); }
inline v_float32x8 v_log(v_float32x8 x) { return v_log_default_32f<v_float32x8, v_int32x8>(x); }
inline void v_sincos(const v_float32x8& x, v_float32x8& s, v_float32x8& c) { v_sincos_default_32f<v_float32x8, v_int32x8>(x, s, c); }
inline v_float32x8 v_sin(const v_float32x8& x) { return v_sin_default_32f<v_float32x8, v_int32x8>(x); }
inline v_float32x8 v_cos(const v_float32x8& x) { return v_cos_default_32f<v_float32x8, v_int32x8>(x); }
inline v_float32x8 v_erf(v_float32x8 x) { return v_erf_default_32f<v_float32x8, v_int32x8>(x); }

inline v_float64x4 v_exp(v_float64x4 x) { return v_exp_default_64f<v_float64x4, v_int64x4>(x); }
inline v_float64x4 v_log(v_float64x4 x) { return v_log_default_64f<v_float64x4, v_int64x4>(x); }
inline void v_sincos(const v_float64x4& x, v_float64x4& s, v_float64x4& c) { v_sincos_default_64f<v_float64x4, v_int64x4>(x, s, c); }
inline v_float64x4 v_sin(const v_float64x4& x) { return v_sin_default_64f<v_float64x4, v_int64x4>(x); }
inline v_float64x4 v_cos(const v_float64x4& x) { return v_cos_default_64f<v_float64x4, v_int64x4>(x); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // OPENCV_HAL_INTRIN_LASX_HPP
