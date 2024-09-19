// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_INTRIN_LSX_HPP
#define OPENCV_HAL_INTRIN_LSX_HPP

#include <lsxintrin.h>

#define CV_SIMD128 1
#define CV_SIMD128_64F 1
#define CV_SIMD128_FP16 0

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

/////////// Utils ////////

inline __m128i _v128_setr_b(char v0, char v1, char v2, char v3, char v4, char v5, char v6,
        char v7, char v8, char v9, char v10, char v11, char v12, char v13, char v14, char v15)
{
    return (__m128i)v16i8{ v0, v1, v2, v3, v4, v5, v6, v7,
                           v8, v9, v10, v11, v12, v13, v14, v15 };
}

inline __m128i _v128_set_b(char v0, char v1, char v2, char v3, char v4, char v5, char v6,
        char v7, char v8, char v9, char v10, char v11, char v12, char v13, char v14, char v15)
{
    return (__m128i)v16i8{ v15, v14, v13, v12, v11, v10, v9, v8,
                           v7, v6, v5, v4, v3, v2, v1, v0 };
}

inline __m128i _v128_setr_h(short v0, short v1, short v2, short v3, short v4, short v5,
       short v6, short v7)
{
    return (__m128i)v8i16{ v0, v1, v2, v3, v4, v5, v6, v7 };
}

inline __m128i _v128_setr_w(int v0, int v1, int v2, int v3)
{
    return (__m128i)v4i32{ v0, v1, v2, v3 };
}

inline __m128i _v128_set_w(int v0, int v1, int v2, int v3)
{
    return (__m128i)v4i32{ v3, v2, v1, v0 };
}

inline __m128i _v128_setall_w(int v0)
{
    return __lsx_vreplgr2vr_w(v0);
}

inline __m128i _v128_setr_d(int64 v0, int64 v1)
{
    return (__m128i)v2i64{ v0, v1 };
}

inline __m128i _v128_set_d(int64 v0, int64 v1)
{
    return (__m128i)v2i64{ v1, v0 };
}

inline __m128 _v128_setr_ps(float v0, float v1, float v2, float v3)
{
    return (__m128)v4f32{ v0, v1, v2, v3 };
}

inline __m128 _v128_setall_ps(float v0)
{
    return (__m128)v4f32{ v0, v0, v0, v0 };
}

inline __m128d _v128_setr_pd(double v0, double v1)
{
    return (__m128d)v2f64{ v0, v1 };
}

inline __m128d _v128_setall_pd(double v0)
{
    return (__m128d)v2f64{ v0, v0 };
}

inline __m128i _lsx_packus_h(const __m128i& a, const __m128i& b)
{
    return __lsx_vssrarni_bu_h(b, a, 0);
}

inline __m128i _lsx_packs_h(const __m128i& a, const __m128i& b)
{
    return __lsx_vssrarni_b_h(b, a, 0);
}

inline __m128i _lsx_packus_w(const __m128i& a, const __m128i& b)
{
    return __lsx_vssrarni_hu_w(b, a, 0);
}

/////// Types ///////

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16};

    v_uint8x16() {}
    explicit v_uint8x16(__m128i v): val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
             uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        val = _v128_setr_b(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }

    uchar get0() const
    {
        return (uchar)__lsx_vpickve2gr_bu(val, 0);
    }

    __m128i val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(__m128i v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
            schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        val = _v128_setr_b(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15);
    }

    schar get0() const
    {
        return (schar)__lsx_vpickve2gr_b(val, 0);
    }

    __m128i val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(__m128i v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        val = _v128_setr_h(v0, v1, v2, v3, v4, v5, v6, v7);
    }

    ushort get0() const
    {
        return (ushort)__lsx_vpickve2gr_hu(val, 0);
    }

    __m128i val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(__m128i v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        val = _v128_setr_h(v0, v1, v2, v3, v4, v5, v6, v7);
    }

    short get0() const
    {
        return (short)__lsx_vpickve2gr_h(val, 0);
    }

    __m128i val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(__m128i v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        val = _v128_setr_w(v0, v1, v2, v3);
    }

    unsigned get0() const
    {
        return (unsigned)__lsx_vpickve2gr_wu(val, 0);
    }

    __m128i val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(__m128i v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        val = _v128_setr_w(v0, v1, v2, v3);
    }

    int get0() const
    {
        return (int)__lsx_vpickve2gr_w(val, 0);
    }

    __m128i val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4};

    v_float32x4() {}
    explicit v_float32x4(__m128 v) : val(v) {}
    explicit v_float32x4(__m128i v) { val = *((__m128*)&v); }
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        val = _v128_setr_ps(v0, v1, v2, v3);
    }

    float get0() const
    {
        union { int iv; float fv; } d;
        d.iv = __lsx_vpickve2gr_w(val, 0);
        return d.fv;
    }

    int get0toint() const
    {
        __m128i result = __lsx_vftintrz_w_s(val);
        return (int)__lsx_vpickve2gr_w(result, 0);
    }

    __m128 val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2};

    v_uint64x2() {}
    explicit v_uint64x2(__m128i v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        val = _v128_setr_d(v0, v1);
    }

    uint64 get0() const
    {
        return __lsx_vpickve2gr_du(val, 0);
    }

    __m128i val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2};

    v_int64x2() {}
    explicit v_int64x2(__m128i v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        val = _v128_setr_d(v0, v1);
    }

    uint64 get0() const
    {
        return __lsx_vpickve2gr_d(val, 0);
    }

    __m128i val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2};

    v_float64x2() {}
    explicit v_float64x2(__m128d v) : val(v) {}
    explicit v_float64x2(__m128i v) { val = *((__m128d*)&v); }
    v_float64x2(double v0, double v1)
    {
        val = _v128_setr_pd(v0, v1);
    }

    double get0() const
    {
        union { int64 iv; double fv; } d;
        d.iv = __lsx_vpickve2gr_d(val, 0);
        return d.fv;
    }

    int64 get0toint64() const
    {
        __m128i result = __lsx_vftintrz_l_d(val);
        return (int64)__lsx_vpickve2gr_d(result, 0);
    }

    __m128d val;
};

////////////// Load and store operations /////////

#define OPENCV_HAL_IMPL_LSX_LOADSTORE(_Tpvec, _Tp)                     \
    inline _Tpvec v_load(const _Tp* ptr)                               \
    { return _Tpvec(__lsx_vld(ptr, 0)); }                              \
    inline _Tpvec v_load_aligned(const _Tp* ptr)                       \
    { return _Tpvec(__lsx_vld(ptr, 0)); }                              \
    inline _Tpvec v_load_low(const _Tp* ptr)                           \
    { return _Tpvec(__lsx_vldrepl_d(ptr, 0)); }                        \
    inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1)      \
    {                                                                  \
        __m128i vl = __lsx_vldrepl_d(ptr0, 0);                         \
        __m128i vh = __lsx_vldrepl_d(ptr1, 0);                         \
        return _Tpvec(__lsx_vilvl_d(vh, vl));                          \
    }                                                                  \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                     \
    { __lsx_vst(a.val, ptr, 0); }                                      \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)             \
    { __lsx_vst(a.val, ptr, 0); }                                      \
    inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a)     \
    { __lsx_vst(a.val, ptr, 0); }                                      \
    inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode)\
    {                                                                  \
        if ( mode == hal::STORE_UNALIGNED)                             \
            __lsx_vst(a.val, ptr, 0);                                  \
        else if ( mode == hal::STORE_ALIGNED_NOCACHE)                  \
            __lsx_vst(a.val, ptr, 0);                                  \
        else                                                           \
            __lsx_vst(a.val, ptr, 0);                                  \
    }                                                                  \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                 \
    {  __lsx_vstelm_d(a.val, ptr, 0, 0); }                             \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)                \
    {  __lsx_vstelm_d(a.val, ptr, 0, 1); }                             \

OPENCV_HAL_IMPL_LSX_LOADSTORE(v_uint8x16,  uchar)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_int8x16,   schar)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_uint16x8, ushort)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_int16x8,  short)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_uint32x4,  unsigned)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_int32x4,   int)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_uint64x2,  uint64)
OPENCV_HAL_IMPL_LSX_LOADSTORE(v_int64x2,   int64)

#define OPENCV_HAL_IMPL_LSX_LOADSTORE_FLT(_Tpvec, _Tp, halfreg)        \
    inline _Tpvec v_load(const _Tp* ptr)                               \
    { return _Tpvec((halfreg)__lsx_vld(ptr, 0)); }                     \
    inline _Tpvec v_load_aligned(const _Tp* ptr)                       \
    { return _Tpvec((halfreg)__lsx_vld(ptr, 0)); }                     \
    inline _Tpvec v_load_low(const _Tp* ptr)                           \
    { return _Tpvec((halfreg)__lsx_vldrepl_d(ptr, 0)); }               \
    inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1)      \
    {                                                                  \
        __m128i vl = __lsx_vldrepl_d(ptr0, 0);                         \
        __m128i vh = __lsx_vldrepl_d(ptr1, 0);                         \
        return _Tpvec((halfreg)__lsx_vilvl_d(vh, vl));                 \
    }                                                                  \
    inline void v_store(_Tp* ptr, const _Tpvec& a)                     \
    {  __lsx_vst((__m128i)a.val, ptr, 0); }                            \
    inline void v_store_aligned(_Tp* ptr, const _Tpvec& a)             \
    {  __lsx_vst((__m128i)a.val, ptr, 0); }                            \
    inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a)     \
    {  __lsx_vst((__m128i)a.val, ptr, 0); }                            \
    inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode)\
    {                                                                  \
        if( mode == hal::STORE_UNALIGNED)                              \
            __lsx_vst((__m128i)a.val, ptr, 0);                         \
        else if( mode == hal::STORE_ALIGNED_NOCACHE)                   \
            __lsx_vst((__m128i)a.val, ptr, 0);                         \
        else                                                           \
            __lsx_vst((__m128i)a.val, ptr, 0);                         \
    }                                                                  \
    inline void v_store_low(_Tp* ptr, const _Tpvec& a)                 \
    {  __lsx_vstelm_d((__m128i)a.val, ptr, 0, 0); }                    \
    inline void v_store_high(_Tp* ptr, const _Tpvec& a)                \
    {  __lsx_vstelm_d((__m128i)a.val, ptr, 0, 1); }                    \

OPENCV_HAL_IMPL_LSX_LOADSTORE_FLT(v_float32x4, float, __m128)
OPENCV_HAL_IMPL_LSX_LOADSTORE_FLT(v_float64x2, double, __m128d)

inline __m128i _lsx_128_castps_si128(const __m128& v)
{ return __m128i(v); }

inline __m128i _lsx_128_castpd_si128(const __m128d& v)
{ return __m128i(v); }

#define OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, _Tpvecf, suffix, cast)  \
    inline _Tpvec v_reinterpret_as_##suffix(const _Tpvecf& a)    \
    { return _Tpvec(cast(a.val)); }

#define OPENCV_HAL_IMPL_LSX_INIT(_Tpvec, _Tp, suffix, ssuffix, ctype_s)           \
    inline _Tpvec v_setzero_##suffix()                                            \
    { return _Tpvec(__lsx_vldi(0)); }                                             \
    inline _Tpvec v_setall_##suffix(_Tp v)                                        \
    { return _Tpvec(__lsx_vreplgr2vr_##ssuffix((ctype_s)v)); }                    \
    inline _Tpvec v_setzero(_Tpvec /*unused*/)                                    \
    { return v_setzero_##suffix(); }                                              \
    inline _Tpvec v_setall(_Tp v, _Tpvec /*unused*/)                              \
    { return v_setall_##suffix(v); }                                              \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint8x16,  suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int8x16,   suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint16x8,  suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int16x8,   suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint32x4,  suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int32x4,   suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint64x2,  suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int64x2,   suffix, OPENCV_HAL_NOP)         \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_float32x4, suffix, _lsx_128_castps_si128)  \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_float64x2, suffix, _lsx_128_castpd_si128)  \

OPENCV_HAL_IMPL_LSX_INIT(v_uint8x16,  uchar,    u8,   b,  int)
OPENCV_HAL_IMPL_LSX_INIT(v_int8x16,   schar,    s8,   b,  int)
OPENCV_HAL_IMPL_LSX_INIT(v_uint16x8,  ushort,   u16,  h,  int)
OPENCV_HAL_IMPL_LSX_INIT(v_int16x8,   short,    s16,  h,  int)
OPENCV_HAL_IMPL_LSX_INIT(v_uint32x4,  unsigned, u32,  w,  int)
OPENCV_HAL_IMPL_LSX_INIT(v_int32x4,   int,      s32,  w,  int)
OPENCV_HAL_IMPL_LSX_INIT(v_uint64x2,  uint64,   u64,  d,  long int)
OPENCV_HAL_IMPL_LSX_INIT(v_int64x2,   int64,    s64,  d,  long int)

inline __m128 _lsx_128_castsi128_ps(const __m128i &v)
{ return __m128(v); }

inline __m128d _lsx_128_castsi128_pd(const __m128i &v)
{ return __m128d(v); }

#define OPENCV_HAL_IMPL_LSX_INIT_FLT(_Tpvec, _Tp, suffix, zsuffix, cast)    \
    inline _Tpvec v_setzero_##suffix()                                      \
    { return _Tpvec(__lsx_vldi(0)); }                                       \
    inline _Tpvec v_setall_##suffix(_Tp v)                                  \
    { return _Tpvec(_v128_setall_##zsuffix(v)); }                           \
    inline _Tpvec v_setzero(_Tpvec /*unused*/)                              \
    { return v_setzero_##suffix(); }                                        \
    inline _Tpvec v_setall(_Tp v, _Tpvec /*unused*/)                        \
    { return v_setall_##suffix(v); }                                        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint8x16,     suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int8x16,      suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint16x8,     suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int16x8,      suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint32x4,     suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int32x4,      suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_uint64x2,     suffix,   cast)        \
    OPENCV_HAL_IMPL_LSX_CAST(_Tpvec, v_int64x2,      suffix,   cast)        \

OPENCV_HAL_IMPL_LSX_INIT_FLT(v_float32x4, float,  f32, ps, _lsx_128_castsi128_ps)
OPENCV_HAL_IMPL_LSX_INIT_FLT(v_float64x2, double, f64, pd, _lsx_128_castsi128_pd)

inline v_float32x4 v_reinterpret_as_f32(const v_float32x4& a)
{ return a; }
inline v_float32x4 v_reinterpret_as_f32(const v_float64x2& a)
{ return v_float32x4(_lsx_128_castps_si128(__m128(a.val))); }

inline v_float64x2 v_reinterpret_as_f64(const v_float64x2& a)
{ return a; }
inline v_float64x2 v_reinterpret_as_f64(const v_float32x4& a)
{ return v_float64x2(_lsx_128_castpd_si128(__m128d(a.val))); }

//////////////// Variant Value reordering ///////////////

// unpacks
#define OPENCV_HAL_IMPL_LSX_UNPACK(_Tpvec, suffix)                            \
    inline _Tpvec v128_unpacklo(const _Tpvec& a, const _Tpvec& b)             \
    { return _Tpvec(__lsx_vilvl_##suffix(__m128i(b.val), __m128i(a.val))); }  \
    inline _Tpvec v128_unpackhi(const _Tpvec& a, const _Tpvec& b)             \
    { return _Tpvec(__lsx_vilvh_##suffix(__m128i(b.val), __m128i(a.val))); }  \

OPENCV_HAL_IMPL_LSX_UNPACK(v_uint8x16,  b)
OPENCV_HAL_IMPL_LSX_UNPACK(v_int8x16,   b)
OPENCV_HAL_IMPL_LSX_UNPACK(v_uint16x8,  h)
OPENCV_HAL_IMPL_LSX_UNPACK(v_int16x8,   h)
OPENCV_HAL_IMPL_LSX_UNPACK(v_uint32x4,  w)
OPENCV_HAL_IMPL_LSX_UNPACK(v_int32x4,   w)
OPENCV_HAL_IMPL_LSX_UNPACK(v_uint64x2,  d)
OPENCV_HAL_IMPL_LSX_UNPACK(v_int64x2,   d)
OPENCV_HAL_IMPL_LSX_UNPACK(v_float32x4, w)
OPENCV_HAL_IMPL_LSX_UNPACK(v_float64x2, d)

//ZIP
#define OPENCV_HAL_IMPL_LSX_ZIP(_Tpvec)                               \
    inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)     \
    { return (_Tpvec)__lsx_vilvl_d((__m128i)b.val, (__m128i)a.val); } \
    inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)    \
    { return (_Tpvec)__lsx_vilvh_d((__m128i)b.val, (__m128i)a.val); } \
    inline void v_recombine(const _Tpvec& a, const _Tpvec& b,         \
                            _Tpvec& c, _Tpvec& d)                     \
    {                                                                 \
        __m128i a1 = (__m128i)a.val,  b1 = (__m128i)b.val;            \
        c = _Tpvec(__lsx_vilvl_d(b1, a1));                            \
        d = _Tpvec(__lsx_vilvh_d(b1, a1));                            \
    }                                                                 \
    inline void v_zip(const _Tpvec& a, const _Tpvec& b,               \
                      _Tpvec& ab0, _Tpvec& ab1)                       \
    {                                                                 \
        ab0 = v128_unpacklo(a, b);                                    \
        ab1 = v128_unpackhi(a, b);                                    \
    }

OPENCV_HAL_IMPL_LSX_ZIP(v_uint8x16)
OPENCV_HAL_IMPL_LSX_ZIP(v_int8x16)
OPENCV_HAL_IMPL_LSX_ZIP(v_uint16x8)
OPENCV_HAL_IMPL_LSX_ZIP(v_int16x8)
OPENCV_HAL_IMPL_LSX_ZIP(v_uint32x4)
OPENCV_HAL_IMPL_LSX_ZIP(v_int32x4)
OPENCV_HAL_IMPL_LSX_ZIP(v_uint64x2)
OPENCV_HAL_IMPL_LSX_ZIP(v_int64x2)
OPENCV_HAL_IMPL_LSX_ZIP(v_float32x4)
OPENCV_HAL_IMPL_LSX_ZIP(v_float64x2)

////////// Arithmetic, bitwise and comparison operations /////////

/** Arithmetics **/
#define OPENCV_HAL_IMPL_LSX_BIN_OP(bin_op, _Tpvec, intrin)           \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)  \
    { return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_uint8x16,  __lsx_vsadd_bu)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_uint8x16,  __lsx_vssub_bu)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_int8x16,   __lsx_vsadd_b)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_int8x16,   __lsx_vssub_b)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_uint16x8,  __lsx_vsadd_hu)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_uint16x8,  __lsx_vssub_hu)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_int16x8,   __lsx_vsadd_h)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_int16x8,   __lsx_vssub_h)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_uint32x4,  __lsx_vadd_w)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_uint32x4,  __lsx_vsub_w)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_mul, v_uint32x4,  __lsx_vmul_w)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_int32x4,   __lsx_vadd_w)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_int32x4,   __lsx_vsub_w)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_mul, v_int32x4,   __lsx_vmul_w)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_uint64x2,  __lsx_vadd_d)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_uint64x2,  __lsx_vsub_d)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_int64x2,   __lsx_vadd_d)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_int64x2,   __lsx_vsub_d)

OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_float32x4, __lsx_vfadd_s)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_float32x4, __lsx_vfsub_s)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_mul, v_float32x4, __lsx_vfmul_s)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_div, v_float32x4, __lsx_vfdiv_s)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_add, v_float64x2, __lsx_vfadd_d)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_sub, v_float64x2, __lsx_vfsub_d)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_mul, v_float64x2, __lsx_vfmul_d)
OPENCV_HAL_IMPL_LSX_BIN_OP(v_div, v_float64x2, __lsx_vfdiv_d)

// saturating multiply 8-bit, 16-bit
inline v_uint8x16 v_mul(const v_uint8x16& a, const v_uint8x16& b)
{
    v_uint16x8 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_int8x16 v_mul(const v_int8x16& a, const v_int8x16& b)
{
    v_int16x8 c, d;
    v_mul_expand(a, b, c, d);
    return v_pack(c, d);
}
inline v_uint16x8 v_mul(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i pev = __lsx_vmulwev_w_hu(a0, b0);
    __m128i pod = __lsx_vmulwod_w_hu(a0, b0);
    __m128i pl  = __lsx_vilvl_w(pod, pev);
    __m128i ph  = __lsx_vilvh_w(pod, pev);
    return (v_uint16x8)__lsx_vssrlrni_hu_w(ph, pl, 0);
}
inline v_int16x8 v_mul(const v_int16x8& a, const v_int16x8& b)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i pev = __lsx_vmulwev_w_h(a0, b0);
    __m128i pod = __lsx_vmulwod_w_h(a0, b0);
    __m128i pl  = __lsx_vilvl_w(pod, pev);
    __m128i ph  = __lsx_vilvh_w(pod, pev);
    return (v_int16x8)__lsx_vssrarni_h_w(ph, pl, 0);
}

/** Non-saturating arithmetics **/

#define OPENCV_HAL_IMPL_LSX_BIN_FUNC(func, _Tpvec, intrin)         \
    inline _Tpvec func(const _Tpvec& a, const _Tpvec& b)           \
    { return _Tpvec(intrin(a.val, b.val)); }                       \

OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_add_wrap, v_uint8x16,  __lsx_vadd_b)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_add_wrap, v_int8x16,   __lsx_vadd_b)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_add_wrap, v_uint16x8,  __lsx_vadd_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_add_wrap, v_int16x8,   __lsx_vadd_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_sub_wrap, v_uint8x16,  __lsx_vsub_b)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_sub_wrap, v_int8x16,   __lsx_vsub_b)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_sub_wrap, v_uint16x8,  __lsx_vsub_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_sub_wrap, v_int16x8,   __lsx_vsub_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_mul_wrap, v_uint16x8,  __lsx_vmul_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_mul_wrap, v_int16x8,   __lsx_vmul_h)

inline v_uint8x16 v_mul_wrap(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i p0 = __lsx_vmulwev_h_bu(a0, b0);
    __m128i p1 = __lsx_vmulwod_h_bu(a0, b0);
    return v_uint8x16(__lsx_vpackev_b(p1, p0));
}

inline v_int8x16 v_mul_wrap(const v_int8x16& a, const v_int8x16& b)
{
    return v_reinterpret_as_s8(v_mul_wrap(v_reinterpret_as_u8(a), v_reinterpret_as_u8(b)));
}

// Multiply and expand
inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i p0 = __lsx_vmulwev_h_bu(a0, b0);
    __m128i p1 = __lsx_vmulwod_h_bu(a0, b0);
    c.val = __lsx_vilvl_h(p1, p0);
    d.val = __lsx_vilvh_h(p1, p0);
}
inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i p0 = __lsx_vmulwev_h_b(a0, b0);
    __m128i p1 = __lsx_vmulwod_h_b(a0, b0);
    c.val = __lsx_vilvl_h(p1, p0);
    d.val = __lsx_vilvh_h(p1, p0);
}
inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i p0 = __lsx_vmulwev_w_h(a0, b0);
    __m128i p1 = __lsx_vmulwod_w_h(a0, b0);
    c.val = __lsx_vilvl_w(p1, p0);
    d.val = __lsx_vilvh_w(p1, p0);
}
inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i p0 = __lsx_vmulwev_w_hu(a0, b0);
    __m128i p1 = __lsx_vmulwod_w_hu(a0, b0);
    c.val = __lsx_vilvl_w(p1, p0);
    d.val = __lsx_vilvh_w(p1, p0);
}
inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    __m128i a0 = a.val, b0 = b.val;
    __m128i p0 = __lsx_vmulwev_d_wu(a0, b0);
    __m128i p1 = __lsx_vmulwod_d_wu(a0, b0);
    c.val = __lsx_vilvl_d(p1, p0);
    d.val = __lsx_vilvh_d(p1, p0);
}
inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{ return v_int16x8(__lsx_vmuh_h(a.val, b.val)); }
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{ return v_uint16x8(__lsx_vmuh_hu(a.val, b.val)); }

/** Bitwise shifts **/
#define OPENCV_HAL_IMPL_LSX_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, srai)                 \
    inline _Tpuvec v_shl(const _Tpuvec& a, int imm)                                  \
    { return _Tpuvec(__lsx_vsll_##suffix(a.val, __lsx_vreplgr2vr_##suffix(imm))); }  \
    inline _Tpsvec v_shl(const _Tpsvec& a, int imm)                                  \
    { return _Tpsvec(__lsx_vsll_##suffix(a.val, __lsx_vreplgr2vr_##suffix(imm))); }  \
    inline _Tpuvec v_shr(const _Tpuvec& a, int imm)                                  \
    { return _Tpuvec(__lsx_vsrl_##suffix(a.val, __lsx_vreplgr2vr_##suffix(imm))); }  \
    inline _Tpsvec v_shr(const _Tpsvec& a, int imm)                                  \
    { return _Tpsvec(srai(a.val, __lsx_vreplgr2vr_##suffix(imm))); }                 \
    template<int imm>                                                                \
    inline _Tpuvec v_shl(const _Tpuvec& a)                                           \
    { return _Tpuvec(__lsx_vslli_##suffix(a.val, imm)); }                            \
    template<int imm>                                                                \
    inline _Tpsvec v_shl(const _Tpsvec& a)                                           \
    { return _Tpsvec(__lsx_vslli_##suffix(a.val, imm)); }                            \
    template<int imm>                                                                \
    inline _Tpuvec v_shr(const _Tpuvec& a)                                           \
    { return _Tpuvec(__lsx_vsrli_##suffix(a.val, imm)); }                            \
    template<int imm>                                                                \
    inline _Tpsvec v_shr(const _Tpsvec& a)                                           \
    { return _Tpsvec(__lsx_vsrai_##suffix(a.val, imm)); }                            \

OPENCV_HAL_IMPL_LSX_SHIFT_OP(v_uint16x8, v_int16x8, h, __lsx_vsra_h)
OPENCV_HAL_IMPL_LSX_SHIFT_OP(v_uint32x4, v_int32x4, w, __lsx_vsra_w)
OPENCV_HAL_IMPL_LSX_SHIFT_OP(v_uint64x2, v_int64x2, d, __lsx_vsra_d)

/** Bitwise logic **/
#define OPENCV_HAL_IMPL_LSX_LOGIC_OP(_Tpvec, suffix)                                 \
    OPENCV_HAL_IMPL_LSX_BIN_OP(v_and, _Tpvec, __lsx_vand_##suffix)                   \
    OPENCV_HAL_IMPL_LSX_BIN_OP(v_or, _Tpvec, __lsx_vor_##suffix)                     \
    OPENCV_HAL_IMPL_LSX_BIN_OP(v_xor, _Tpvec, __lsx_vxor_##suffix)                   \
    inline _Tpvec v_not(const _Tpvec& a)                                             \
    { return _Tpvec(__lsx_vnori_b(a.val, 0)); }                                      \

OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_uint8x16,   v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_int8x16,    v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_uint16x8,   v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_int16x8,    v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_uint32x4,   v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_int32x4,    v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_uint64x2,   v)
OPENCV_HAL_IMPL_LSX_LOGIC_OP(v_int64x2,    v)

#define OPENCV_HAL_IMPL_LSX_FLOAT_BIN_OP(bin_op, _Tpvec, intrin, cast)               \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)                           \
    { return _Tpvec(intrin((__m128i)(a.val), (__m128i)(b.val))); }

#define OPENCV_HAL_IMPL_LSX_FLOAT_LOGIC_OP(_Tpvec, cast)                             \
    OPENCV_HAL_IMPL_LSX_FLOAT_BIN_OP(v_and, _Tpvec, __lsx_vand_v, cast)              \
    OPENCV_HAL_IMPL_LSX_FLOAT_BIN_OP(v_or, _Tpvec, __lsx_vor_v, cast)                \
    OPENCV_HAL_IMPL_LSX_FLOAT_BIN_OP(v_xor, _Tpvec, __lsx_vxor_v, cast)              \
    inline _Tpvec v_not(const _Tpvec& a)                                             \
    { return _Tpvec(__lsx_vnori_b((__m128i)(a.val), 0)); }                           \

OPENCV_HAL_IMPL_LSX_FLOAT_LOGIC_OP(v_float32x4, _lsx_128_castsi128_ps)
OPENCV_HAL_IMPL_LSX_FLOAT_LOGIC_OP(v_float64x2, _lsx_128_castsi128_pd)

/** Select **/
#define OPENCV_HAL_IMPL_LSX_SELECT(_Tpvec)                                           \
    inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b)     \
    { return _Tpvec(__lsx_vbitsel_v(b.val, a.val, mask.val)); }                      \

OPENCV_HAL_IMPL_LSX_SELECT(v_uint8x16)
OPENCV_HAL_IMPL_LSX_SELECT(v_int8x16)
OPENCV_HAL_IMPL_LSX_SELECT(v_uint16x8)
OPENCV_HAL_IMPL_LSX_SELECT(v_int16x8)
OPENCV_HAL_IMPL_LSX_SELECT(v_uint32x4)
OPENCV_HAL_IMPL_LSX_SELECT(v_int32x4)

inline v_float32x4 v_select(const v_float32x4 &mask, const v_float32x4 &a, const v_float32x4 &b)
{ return v_float32x4(__lsx_vbitsel_v((__m128i)b.val, (__m128i)a.val, (__m128i)mask.val)); }
inline v_float64x2 v_select(const v_float64x2 &mask, const v_float64x2 &a, const v_float64x2 &b)
{ return v_float64x2(__lsx_vbitsel_v((__m128i)b.val, (__m128i)a.val, (__m128i)mask.val)); }

/** Comparison **/
#define OPENCV_HAL_IMPL_LSX_CMP_OP_OV(_Tpvec)                            \
    inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b)                 \
    { return v_not(v_eq(a, b)); }                                        \
    inline _Tpvec v_lt(const _Tpvec& a, const _Tpvec& b)                 \
    { return v_gt(b, a); }                                               \
    inline _Tpvec v_ge(const _Tpvec& a, const _Tpvec& b)                 \
    { return v_not(v_lt(a, b)); }                                        \
    inline _Tpvec v_le(const _Tpvec& a, const _Tpvec& b)                 \
    { return v_ge(b, a); }                                               \

#define OPENCV_HAL_IMPL_LSX_CMP_OP_INT(_Tpuvec, _Tpsvec, suffix, usuffix)    \
    inline _Tpuvec v_eq(const _Tpuvec& a, const _Tpuvec& b)                  \
    { return _Tpuvec(__lsx_vseq_##suffix(a.val, b.val)); }                   \
    inline _Tpuvec v_gt(const _Tpuvec& a, const _Tpuvec& b)                  \
    { return _Tpuvec(__lsx_vslt_##usuffix(b.val, a.val)); }                  \
    inline _Tpsvec v_eq(const _Tpsvec& a, const _Tpsvec& b)                  \
    { return _Tpsvec(__lsx_vseq_##suffix(a.val, b.val)); }                   \
    inline _Tpsvec v_gt(const _Tpsvec& a, const _Tpsvec& b)                  \
    { return _Tpsvec(__lsx_vslt_##suffix(b.val, a.val)); }                   \
    OPENCV_HAL_IMPL_LSX_CMP_OP_OV(_Tpuvec)                                   \
    OPENCV_HAL_IMPL_LSX_CMP_OP_OV(_Tpsvec)

OPENCV_HAL_IMPL_LSX_CMP_OP_INT(v_uint8x16,  v_int8x16,  b, bu)
OPENCV_HAL_IMPL_LSX_CMP_OP_INT(v_uint16x8,  v_int16x8,  h, hu)
OPENCV_HAL_IMPL_LSX_CMP_OP_INT(v_uint32x4,  v_int32x4,  w, wu)

#define OPENCV_HAL_IMPL_LSX_CMP_OP_64BIT(_Tpvec, suffix)          \
    inline _Tpvec v_eq(const _Tpvec& a, const _Tpvec& b)          \
    { return _Tpvec(__lsx_vseq_##suffix(a.val, b.val)); }         \
    inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b)          \
    { return v_not(v_eq(a, b)); }

OPENCV_HAL_IMPL_LSX_CMP_OP_64BIT(v_uint64x2, d)
OPENCV_HAL_IMPL_LSX_CMP_OP_64BIT(v_int64x2, d)

#define OPENCV_HAL_IMPL_LSX_CMP_FLT(bin_op, suffix, _Tpvec, ssuffix)       \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b)                 \
    { return _Tpvec(__lsx_##suffix##_##ssuffix(a.val, b.val)); }           \

#define OPENCV_HAL_IMPL_LSX_CMP_OP_FLT(_Tpvec, ssuffix)                    \
    OPENCV_HAL_IMPL_LSX_CMP_FLT(v_eq, vfcmp_ceq, _Tpvec, ssuffix)          \
    OPENCV_HAL_IMPL_LSX_CMP_FLT(v_ne, vfcmp_cne, _Tpvec, ssuffix)          \
    OPENCV_HAL_IMPL_LSX_CMP_FLT(v_lt,  vfcmp_clt, _Tpvec, ssuffix)         \
    OPENCV_HAL_IMPL_LSX_CMP_FLT(v_le, vfcmp_cle, _Tpvec, ssuffix)          \

OPENCV_HAL_IMPL_LSX_CMP_OP_FLT(v_float32x4, s)
OPENCV_HAL_IMPL_LSX_CMP_OP_FLT(v_float64x2, d)

inline v_float32x4 v_gt(const v_float32x4 &a, const v_float32x4 &b)
{ return v_float32x4(__lsx_vfcmp_clt_s(b.val, a.val)); }

inline v_float32x4 v_ge(const v_float32x4 &a, const v_float32x4 &b)
{ return v_float32x4(__lsx_vfcmp_cle_s(b.val, a.val)); }

inline v_float64x2 v_gt(const v_float64x2 &a, const v_float64x2 &b)
{ return v_float64x2(__lsx_vfcmp_clt_d(b.val, a.val)); }

inline v_float64x2 v_ge(const v_float64x2 &a, const v_float64x2 &b)
{ return v_float64x2(__lsx_vfcmp_cle_d(b.val, a.val)); }

inline v_float32x4 v_not_nan(const v_float32x4& a)
{ return v_float32x4(__lsx_vfcmp_cor_s(a.val, a.val)); }

inline v_float64x2 v_not_nan(const v_float64x2& a)
{ return v_float64x2(__lsx_vfcmp_cor_d(a.val, a.val)); }

/** min/max **/
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_uint8x16,  __lsx_vmin_bu)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_uint8x16,  __lsx_vmax_bu)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_int8x16,   __lsx_vmin_b)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_int8x16,   __lsx_vmax_b)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_uint16x8,  __lsx_vmin_hu)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_uint16x8,  __lsx_vmax_hu)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_int16x8,   __lsx_vmin_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_int16x8,   __lsx_vmax_h)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_uint32x4,  __lsx_vmin_wu)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_uint32x4,  __lsx_vmax_wu)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_int32x4,   __lsx_vmin_w)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_int32x4,   __lsx_vmax_w)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_float32x4, __lsx_vfmin_s)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_float32x4, __lsx_vfmax_s)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_min, v_float64x2, __lsx_vfmin_d)
OPENCV_HAL_IMPL_LSX_BIN_FUNC(v_max, v_float64x2, __lsx_vfmax_d)

template <int imm,
    bool is_invalid = ((imm < 0) || (imm > 16)),
    bool is_first = (imm == 0),
    bool is_half = (imm == 8),
    bool is_second = (imm == 16),
    bool is_other = (((imm > 0) && (imm < 8)) || ((imm > 8) && (imm < 16)))>
class v_lsx_palignr_u8_class;

template <int imm>
class v_lsx_palignr_u8_class<imm, true, false, false, false, false>;

template <int imm>
class v_lsx_palignr_u8_class<imm, false, true, false, false, false>
{
public:
    inline __m128i operator()(const __m128i& a, const __m128i& b) const
    {
        CV_UNUSED(b);
        return a;
    }
};

template <int imm>
class v_lsx_palignr_u8_class<imm, false, false, true, false, false>
{
public:
    inline __m128i operator()(const __m128i& a, const __m128i& b) const
    {
        return __lsx_vshuf4i_d(a, b, 0x9);
    }
};

template <int imm>
class v_lsx_palignr_u8_class<imm, false, false, false, true, false>
{
public:
    inline __m128i operator()(const __m128i& a, const __m128i& b) const
    {
        CV_UNUSED(a);
        return b;
    }
};

template <int imm>
class v_lsx_palignr_u8_class<imm, false, false, false, false, true>
{
public:
    inline __m128i operator()(const __m128i& a, const __m128i& b) const
    {
        enum { imm2 = (sizeof(__m128i) - imm) };
        return __lsx_vor_v(__lsx_vbsrl_v(a, imm), __lsx_vbsll_v(b, imm2));
    }
};

template <int imm>
inline __m128i v_lsx_palignr_u8(const __m128i& a, const __m128i& b)
{
    CV_StaticAssert((imm >= 0) && (imm <= 16), "Invalid imm for v_lsx_palignr_u8");
    return v_lsx_palignr_u8_class<imm>()(a, b);
}
/** Rotate **/
#define OPENCV_HAL_IMPL_LSX_ROTATE_CAST(_Tpvec, cast)                                   \
    template<int imm>                                                                   \
    inline _Tpvec v_rotate_right(const _Tpvec &a)                                       \
    {                                                                                   \
        enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type))};                      \
        __m128i ret = __lsx_vbsrl_v((__m128i)a.val, imm2);                              \
        return _Tpvec(cast(ret));                                                       \
    }                                                                                   \
    template<int imm>                                                                   \
    inline _Tpvec v_rotate_left(const _Tpvec &a)                                        \
    {                                                                                   \
        enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type))};                      \
        __m128i ret = __lsx_vbsll_v((__m128i)a.val, imm2);                              \
        return _Tpvec(cast(ret));                                                       \
    }                                                                                   \
    template<int imm>                                                                   \
    inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b)                      \
    {                                                                                   \
        enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type))};                      \
        return _Tpvec(cast(v_lsx_palignr_u8<imm2>((__m128i)a.val, (__m128i)b.val)));    \
    }                                                                                   \
    template<int imm>                                                                   \
    inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b)                       \
    {                                                                                   \
        enum { imm2 = ((_Tpvec::nlanes - imm) * sizeof(typename _Tpvec::lane_type))};   \
        return _Tpvec(cast(v_lsx_palignr_u8<imm2>((__m128i)b.val, (__m128i)a.val)));    \
    }

OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_uint8x16, OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_int8x16,  OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_uint16x8, OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_int16x8,  OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_uint32x4, OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_int32x4,  OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_uint64x2, OPENCV_HAL_NOP)                             \
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_int64x2,  OPENCV_HAL_NOP)                             \

OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_float32x4, _lsx_128_castsi128_ps)
OPENCV_HAL_IMPL_LSX_ROTATE_CAST(v_float64x2, _lsx_128_castsi128_pd)

/** Rverse **/
inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
    __m128i vec = __lsx_vshuf4i_b(a.val, 0x1B);
    return v_uint8x16(__lsx_vshuf4i_w(vec, 0x1B));
}

inline v_int8x16 v_reverse(const v_int8x16 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
    __m128i vec = __lsx_vshuf4i_h(a.val, 0x1B);
    return v_uint16x8(__lsx_vshuf4i_w(vec, 0x4E));
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{ return v_uint32x4(__lsx_vshuf4i_w(a.val, 0x1B)); }

inline v_int32x4 v_reverse(const v_int32x4 &a)
{ return v_int32x4(__lsx_vshuf4i_w(a.val, 0x1B)); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{ return v_uint64x2(__lsx_vshuf4i_w(a.val, 0x4E)); }

inline v_int64x2 v_reverse(const v_int64x2 &a)
{ return v_int64x2(__lsx_vshuf4i_w(a.val, 0x4E)); }

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float64x2 v_reverse(const v_float64x2 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }

////////////// Reduce and mask ////////////

/** Reduce **/
// this function is return a[0]+a[1]+...+a[31]
inline unsigned v_reduce_sum(const v_uint8x16& a)
{
    __m128i t1 = __lsx_vhaddw_hu_bu(a.val, a.val);
    __m128i t2 = __lsx_vhaddw_wu_hu(t1, t1);
    __m128i t3 = __lsx_vhaddw_du_wu(t2, t2);
    __m128i t4 = __lsx_vhaddw_qu_du(t3, t3);
    return (unsigned)__lsx_vpickve2gr_w(t4, 0);
}

inline int v_reduce_sum(const v_int8x16 &a)
{
    __m128i t1 = __lsx_vhaddw_h_b(a.val, a.val);
    __m128i t2 = __lsx_vhaddw_w_h(t1, t1);
    __m128i t3 = __lsx_vhaddw_d_w(t2, t2);
    __m128i t4 = __lsx_vhaddw_q_d(t3, t3);
    return (int)__lsx_vpickve2gr_w(t4, 0);
}

#define OPENCV_HAL_IMPL_LSX_REDUCE_16(_Tpvec, sctype, func, intrin)            \
    inline sctype v_reduce_##func(const _Tpvec& a)                             \
    {                                                                          \
        __m128i val = intrin(a.val, __lsx_vbsrl_v(a.val, 8));                  \
        val = intrin(val, __lsx_vbsrl_v(val, 4));                              \
        val = intrin(val, __lsx_vbsrl_v(val, 2));                              \
        val = intrin(val, __lsx_vbsrl_v(val, 1));                              \
        return (sctype)__lsx_vpickve2gr_b(val, 0);                             \
    }

OPENCV_HAL_IMPL_LSX_REDUCE_16(v_uint8x16, uchar, min, __lsx_vmin_bu)
OPENCV_HAL_IMPL_LSX_REDUCE_16(v_uint8x16, uchar, max, __lsx_vmax_bu)
OPENCV_HAL_IMPL_LSX_REDUCE_16(v_int8x16,  schar, min, __lsx_vmin_b)
OPENCV_HAL_IMPL_LSX_REDUCE_16(v_int8x16,  schar, max, __lsx_vmax_b)

#define OPENCV_HAL_IMPL_LSX_REDUCE_8(_Tpvec, sctype, func, intrin)             \
    inline sctype v_reduce_##func(const _Tpvec &a)                             \
    {                                                                          \
        __m128i val = intrin(a.val, __lsx_vbsrl_v(a.val, 8));                  \
        val = intrin(val, __lsx_vbsrl_v(val, 4));                              \
        val = intrin(val, __lsx_vbsrl_v(val, 2));                              \
        return (sctype)__lsx_vpickve2gr_h(val, 0);                             \
    }

OPENCV_HAL_IMPL_LSX_REDUCE_8(v_uint16x8, ushort, min, __lsx_vmin_hu)
OPENCV_HAL_IMPL_LSX_REDUCE_8(v_uint16x8, ushort, max, __lsx_vmax_hu)
OPENCV_HAL_IMPL_LSX_REDUCE_8(v_int16x8,  short,  min, __lsx_vmin_h)
OPENCV_HAL_IMPL_LSX_REDUCE_8(v_int16x8,  short,  max, __lsx_vmax_h)

#define OPENCV_HAL_IMPL_LSX_REDUCE_4(_Tpvec, sctype, func, intrin)             \
    inline sctype v_reduce_##func(const _Tpvec &a)                             \
    {                                                                          \
        __m128i val = intrin(a.val, __lsx_vbsrl_v(a.val, 8));                  \
        val = intrin(val, __lsx_vbsrl_v(val, 4));                              \
        return (sctype)__lsx_vpickve2gr_w(val, 0);                             \
    }

OPENCV_HAL_IMPL_LSX_REDUCE_4(v_uint32x4, unsigned, min, __lsx_vmin_wu)
OPENCV_HAL_IMPL_LSX_REDUCE_4(v_uint32x4, unsigned, max, __lsx_vmax_wu)
OPENCV_HAL_IMPL_LSX_REDUCE_4(v_int32x4,  int,      min, __lsx_vmin_w)
OPENCV_HAL_IMPL_LSX_REDUCE_4(v_int32x4,  int,      max, __lsx_vmax_w)

#define OPENCV_HAL_IMPL_LSX_REDUCE_FLT(func, intrin)                           \
    inline float v_reduce_##func(const v_float32x4 &a)                         \
    {                                                                          \
        __m128 val   = a.val;                                                  \
        val = intrin(val, (__m128)__lsx_vbsrl_v((__m128i)val, 8));             \
        val = intrin(val, (__m128)__lsx_vbsrl_v((__m128i)val, 4));             \
        float *fval = (float*)&val;                                            \
        return fval[0];                                                        \
    }

OPENCV_HAL_IMPL_LSX_REDUCE_FLT(min, __lsx_vfmin_s)
OPENCV_HAL_IMPL_LSX_REDUCE_FLT(max, __lsx_vfmax_s)

inline int v_reduce_sum(const v_int32x4 &a)
{
    __m128i t1 = __lsx_vhaddw_d_w(a.val, a.val);
    __m128i t2 = __lsx_vhaddw_q_d(t1, t1);
    return (int)__lsx_vpickve2gr_w(t2, 0);
}

inline unsigned v_reduce_sum(const v_uint32x4 &a)
{
    __m128i t1 = __lsx_vhaddw_du_wu(a.val, a.val);
    __m128i t2 = __lsx_vhaddw_qu_du(t1, t1);
    return (int)__lsx_vpickve2gr_w(t2, 0);
}

inline int v_reduce_sum(const v_int16x8 &a)
{
    __m128i t1 = __lsx_vhaddw_w_h(a.val, a.val);
    __m128i t2 = __lsx_vhaddw_d_w(t1, t1);
    __m128i t3 = __lsx_vhaddw_q_d(t2, t2);
    return (int)__lsx_vpickve2gr_w(t3, 0);
}

inline unsigned v_reduce_sum(const v_uint16x8 &a)
{
    __m128i t1 = __lsx_vhaddw_wu_hu(a.val, a.val);
    __m128i t2 = __lsx_vhaddw_du_wu(t1, t1);
    __m128i t3 = __lsx_vhaddw_qu_du(t2, t2);
    return (int)__lsx_vpickve2gr_w(t3, 0);
}

inline float v_reduce_sum(const v_float32x4 &a)
{
    __m128i val = (__m128i)a.val;
    val = __lsx_vbsrl_v(val, 8);
    __m128 result = __lsx_vfadd_s(a.val, (__m128)val);
    float *pa = (float*)&result;
    return (float)(pa[0] + pa[1]);
}

inline uint64 v_reduce_sum(const v_uint64x2 &a)
{
    __m128i t0 = __lsx_vhaddw_qu_du(a.val, a.val);
    return (uint64)__lsx_vpickve2gr_du(t0, 0);
}

inline int64 v_reduce_sum(const v_int64x2 &a)
{
    __m128i t0 = __lsx_vhaddw_q_d(a.val, a.val);
    return (int64)__lsx_vpickve2gr_d(t0, 0);
}

inline double v_reduce_sum(const v_float64x2 &a)
{
    double *pa = (double*)&a;
    return pa[0] + pa[1];
}

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    __m128i a0 = (__m128i)a.val;
    __m128i b0 = (__m128i)b.val;
    __m128i c0 = (__m128i)c.val;
    __m128i d0 = (__m128i)d.val;
    __m128i ac_l = __lsx_vilvl_w(c0, a0);
    __m128i ac_h = __lsx_vilvh_w(c0, a0);
    __m128i bd_l = __lsx_vilvl_w(d0, b0);
    __m128i bd_h = __lsx_vilvh_w(d0, b0);
    __m128  ac   = __lsx_vfadd_s((__m128)ac_l, (__m128)ac_h);
    __m128  bd   = __lsx_vfadd_s((__m128)bd_l, (__m128)bd_h);
    return v_float32x4(__lsx_vfadd_s((__m128)__lsx_vilvl_w((__m128i)bd, (__m128i)ac),
                       (__m128)__lsx_vilvh_w((__m128i)bd, (__m128i)ac)));
}

inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
{
    __m128i t0 = __lsx_vabsd_b(a.val, b.val);
    __m128i t1 = __lsx_vhaddw_hu_bu(t0, t0);
    __m128i t2 = __lsx_vhaddw_wu_hu(t1, t1);
    __m128i t3 = __lsx_vhaddw_du_wu(t2, t2);
    __m128i t4 = __lsx_vhaddw_qu_du(t3, t3);
    return (unsigned)__lsx_vpickve2gr_w(t4, 0);
}

inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i t0 = __lsx_vabsd_bu(a.val, b.val);
    __m128i t1 = __lsx_vhaddw_hu_bu(t0, t0);
    __m128i t2 = __lsx_vhaddw_wu_hu(t1, t1);
    __m128i t3 = __lsx_vhaddw_du_wu(t2, t2);
    __m128i t4 = __lsx_vhaddw_qu_du(t3, t3);
    return (unsigned)__lsx_vpickve2gr_w(t4, 0);
}

inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i t0 = __lsx_vabsd_hu(a.val, b.val);
    __m128i t1 = __lsx_vhaddw_wu_hu(t0, t0);
    __m128i t2 = __lsx_vhaddw_du_wu(t1, t1);
    __m128i t3 = __lsx_vhaddw_qu_du(t2, t2);
    return (unsigned)__lsx_vpickve2gr_w(t3, 0);
}

inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
{
    __m128i t0 = __lsx_vabsd_h(a.val, b.val);
    __m128i t1 = __lsx_vhaddw_wu_hu(t0, t0);
    __m128i t2 = __lsx_vhaddw_du_wu(t1, t1);
    __m128i t3 = __lsx_vhaddw_qu_du(t2, t2);
    return (unsigned)__lsx_vpickve2gr_w(t3, 0);
}

inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i t0 = __lsx_vabsd_wu(a.val, b.val);
    __m128i t1 = __lsx_vhaddw_du_wu(t0, t0);
    __m128i t2 = __lsx_vhaddw_qu_du(t1, t1);
    return (unsigned)__lsx_vpickve2gr_w(t2, 0);
}

inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
{
    __m128i t0 = __lsx_vabsd_w(a.val, b.val);
    __m128i t1 = __lsx_vhaddw_du_wu(t0, t0);
    __m128i t2 = __lsx_vhaddw_qu_du(t1, t1);
    return (unsigned)__lsx_vpickve2gr_w(t2, 0);
}

inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 a_b = v_sub(a, b);
    return v_reduce_sum(v_float32x4((__m128i)a_b.val & __lsx_vreplgr2vr_w(0x7fffffff)));
}

/** Popcount **/
#define OPENCV_HAL_IMPL_LSX_POPCOUNT(_Tpvec, _Tp, suffix)                  \
inline _Tpvec v_popcount(const _Tp& a)                                     \
{ return _Tpvec(__lsx_vpcnt_##suffix(a.val)); }

OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint8x16,  v_uint8x16,  b);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint8x16,  v_int8x16,   b);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint16x8,  v_uint16x8,  h);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint16x8,  v_int16x8,   h);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint32x4,  v_uint32x4,  w);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint32x4,  v_int32x4,   w);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint64x2,  v_uint64x2,  d);
OPENCV_HAL_IMPL_LSX_POPCOUNT(v_uint64x2,  v_int64x2,   d);

/** Mask **/
#define OPENCV_HAL_IMPL_REINTERPRET_INT(ft, tt)              \
inline tt reinterpret_int(ft x) { union {ft l; tt i;} v; v.l = x; return v.i; }
OPENCV_HAL_IMPL_REINTERPRET_INT(uchar, schar)
OPENCV_HAL_IMPL_REINTERPRET_INT(schar, schar)
OPENCV_HAL_IMPL_REINTERPRET_INT(ushort, short)
OPENCV_HAL_IMPL_REINTERPRET_INT(short, short)
OPENCV_HAL_IMPL_REINTERPRET_INT(unsigned, int)
OPENCV_HAL_IMPL_REINTERPRET_INT(int, int)
OPENCV_HAL_IMPL_REINTERPRET_INT(float, int)
OPENCV_HAL_IMPL_REINTERPRET_INT(uint64, int64)
OPENCV_HAL_IMPL_REINTERPRET_INT(int64, int64)
OPENCV_HAL_IMPL_REINTERPRET_INT(double, int64)

inline int v_signmask(const v_int8x16& a)
{
    __m128i result = __lsx_vmskltz_b(a.val);
    return __lsx_vpickve2gr_w(result, 0);
}
inline int v_signmask(const v_uint8x16& a)
{ return v_signmask(v_reinterpret_as_s8(a)) ;}

inline int v_signmask(const v_int16x8 &a)
{
    __m128i result = __lsx_vmskltz_h(a.val);
    return __lsx_vpickve2gr_w(result, 0);
}
inline int v_signmask(const v_uint16x8 &a)
{ return v_signmask(v_reinterpret_as_s16(a)); }

inline int v_signmask(const v_uint32x4& a)
{
    __m128i result = __lsx_vmskltz_w(a.val);
    return __lsx_vpickve2gr_w(result, 0);
}
inline int v_signmask(const v_int32x4& a)
{ return v_signmask(v_reinterpret_as_u32(a)); }

inline int v_signmask(const v_uint64x2& a)
{
    __m128i result = __lsx_vmskltz_d(a.val);
    return __lsx_vpickve2gr_w(result, 0);
}
inline int v_signmask(const v_int64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }

inline int v_signmask(const v_float32x4& a)
{ return v_signmask(*(v_int32x4*)(&a)); }

inline int v_signmask(const v_float64x2& a)
{ return v_signmask(*(v_int64x2*)(&a)); }

inline int v_scan_forward(const v_int8x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_uint8x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_int16x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_uint16x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_int32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_uint32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_float32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_int64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_uint64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_float64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }

/** Checks **/
#define OPENCV_HAL_IMPL_LSX_CHECK(_Tpvec, allmask) \
    inline bool v_check_all(const _Tpvec& a) { return v_signmask(a) == allmask; } \
    inline bool v_check_any(const _Tpvec& a) { return v_signmask(a) != 0; }
OPENCV_HAL_IMPL_LSX_CHECK(v_uint8x16, 65535)
OPENCV_HAL_IMPL_LSX_CHECK(v_int8x16, 65535)
OPENCV_HAL_IMPL_LSX_CHECK(v_uint16x8, 255);
OPENCV_HAL_IMPL_LSX_CHECK(v_int16x8, 255);
OPENCV_HAL_IMPL_LSX_CHECK(v_uint32x4, 15)
OPENCV_HAL_IMPL_LSX_CHECK(v_int32x4, 15)
OPENCV_HAL_IMPL_LSX_CHECK(v_uint64x2, 3)
OPENCV_HAL_IMPL_LSX_CHECK(v_int64x2, 3)
OPENCV_HAL_IMPL_LSX_CHECK(v_float32x4, 15)
OPENCV_HAL_IMPL_LSX_CHECK(v_float64x2, 3)

///////////// Other math /////////////

/** Some frequent operations **/
#define OPENCV_HAL_IMPL_LSX_MULADD(_Tpvec, suffix)                              \
    inline _Tpvec v_fma(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)      \
    { return _Tpvec(__lsx_vfmadd_##suffix(a.val, b.val, c.val)); }              \
    inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec &b, const _Tpvec& c)   \
    { return _Tpvec(__lsx_vfmadd_##suffix(a.val, b.val, c.val)); }              \
    inline _Tpvec v_sqrt(const _Tpvec& x)                                       \
    { return _Tpvec(__lsx_vfsqrt_##suffix(x.val)); }                            \
    inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b)             \
    { return v_fma(a, a, v_mul(b, b)); }                                        \
    inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b)                 \
    { return v_sqrt(v_fma(a, a, v_mul(b, b))); }

OPENCV_HAL_IMPL_LSX_MULADD(v_float32x4, s)
OPENCV_HAL_IMPL_LSX_MULADD(v_float64x2, d)

inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{ return v_int32x4(__lsx_vmadd_w(c.val, a.val, b.val)); }

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{ return v_fma(a, b, c); }

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{
    return v_float32x4(__lsx_vfrsqrt_s(x.val));
}

inline v_float64x2 v_invsqrt(const v_float64x2& x)
{
    return v_float64x2(__lsx_vfrsqrt_d(x.val));
}

/** Absolute values **/
#define OPENCV_HAL_IMPL_LSX_ABS(_Tpvec, suffix)                          \
    inline v_u##_Tpvec v_abs(const v_##_Tpvec& x)                        \
    { return v_u##_Tpvec(__lsx_vabsd_##suffix(x.val, __lsx_vldi(0))); }

OPENCV_HAL_IMPL_LSX_ABS(int8x16, b)
OPENCV_HAL_IMPL_LSX_ABS(int16x8, h)
OPENCV_HAL_IMPL_LSX_ABS(int32x4, w)

inline v_float32x4 v_abs(const v_float32x4& x)
{ return v_float32x4(*((__m128i*)&x) & __lsx_vreplgr2vr_w(0x7fffffff)); }
inline v_float64x2 v_abs(const v_float64x2& x)
{ return v_float64x2(*((__m128i*)&x) & __lsx_vreplgr2vr_d(0x7fffffffffffffff)); }

/** Absolute difference **/

inline v_uint8x16 v_absdiff(const v_uint8x16& a, const v_uint8x16& b)
{ return (v_uint8x16)__lsx_vabsd_bu(a.val, b.val); }
inline v_uint16x8 v_absdiff(const v_uint16x8& a, const v_uint16x8& b)
{ return (v_uint16x8)__lsx_vabsd_hu(a.val, b.val); }
inline v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b)
{ return (v_uint32x4)__lsx_vabsd_wu(a.val, b.val); }

inline v_uint8x16 v_absdiff(const v_int8x16& a, const v_int8x16& b)
{ return (v_uint8x16)__lsx_vabsd_b(a.val, b.val); }
inline v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b)
{ return (v_uint16x8)__lsx_vabsd_h(a.val, b.val); }
inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{ return (v_uint32x4)__lsx_vabsd_w(a.val, b.val); }

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{ return v_abs(v_sub(a, b)); }

inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{ return v_abs(v_sub(a, b)); }

/** Saturating absolute difference **/
inline v_int8x16 v_absdiffs(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = v_sub(a, b);
    v_int8x16 m = v_lt(a, b);
    return v_sub(v_xor(d, m), m);
}
inline v_int16x8 v_absdiffs(const v_int16x8& a, const v_int16x8& b)
{ return v_sub(v_max(a, b), v_min(a, b)); }

///////// Conversions /////////

/** Rounding **/
inline v_int32x4 v_round(const v_float32x4& a)
{ return v_int32x4(__lsx_vftint_w_s(a.val)); }

inline v_int32x4 v_round(const v_float64x2& a)
{ return v_int32x4(__lsx_vftint_w_d(a.val, a.val)); }

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{ return v_int32x4(__lsx_vftint_w_d(b.val, a.val)); }

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(__lsx_vftintrz_w_s(a.val)); }

inline v_int32x4 v_trunc(const v_float64x2& a)
{ return v_int32x4(__lsx_vftintrz_w_d(a.val, a.val)); }

inline v_int32x4 v_floor(const v_float32x4& a)
{ return v_int32x4(__lsx_vftintrz_w_s(__m128(__lsx_vfrintrm_s(a.val)))); }

inline v_int32x4 v_floor(const v_float64x2& a)
{ return v_trunc(v_float64x2(__lsx_vfrintrm_d(a.val))); }

inline v_int32x4 v_ceil(const v_float32x4& a)
{ return v_int32x4(__lsx_vftintrz_w_s(__m128(__lsx_vfrintrp_s(a.val)))); }

inline v_int32x4 v_ceil(const v_float64x2& a)
{ return v_trunc(v_float64x2(__lsx_vfrintrp_d(a.val))); }

/** To float **/
inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{ return v_float32x4(__lsx_vffint_s_w(a.val)); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{ return v_float32x4(__lsx_vfcvt_s_d(a.val, a.val)); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{ return v_float32x4(__lsx_vfcvt_s_d(b.val, a.val)); }

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{ return v_float64x2(__lsx_vffintl_d_w(a.val)); }

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{ return v_float64x2(__lsx_vffinth_d_w(a.val)); }

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{ return v_float64x2(__lsx_vfcvtl_d_s(a.val)); }

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{ return v_float64x2(__lsx_vfcvth_d_s(a.val)); }

inline v_float64x2 v_cvt_f64(const v_int64x2& v)
{ return v_float64x2(__lsx_vffint_d_l(v.val)); }


//////////////// Lookup table access ////////////////
inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
    return v_int8x16(_v128_setr_b(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                     tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]], tab[idx[8]],
                     tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]],
                     tab[idx[14]], tab[idx[15]]));
}

inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
    return v_int8x16(_v128_setr_h(*(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]),
           *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3]), *(const short*)(tab + idx[4]),
           *(const short*)(tab + idx[5]), *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7])));
}

inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    return v_int8x16(_v128_setr_w(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
}

inline v_uint8x16 v_lut(const uchar* tab, const int* idx)
{ return v_reinterpret_as_u8(v_lut((const schar*)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx)
{ return v_reinterpret_as_u8(v_lut_pairs((const schar*)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx)
{ return v_reinterpret_as_u8(v_lut_quads((const schar*)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    return v_int16x8(_v128_setr_h(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                     tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]));
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    return v_int16x8(_v128_setr_w(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    return v_int16x8(_v128_setr_d(*(const int64_t*)(tab + idx[0]), *(const int64_t*)(tab + idx[1])));
}

inline v_uint16x8 v_lut(const ushort* tab, const int* idx)
{ return v_reinterpret_as_u16(v_lut((const short *)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx)
{ return v_reinterpret_as_u16(v_lut_pairs((const short *)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx)
{ return v_reinterpret_as_u16(v_lut_quads((const short *)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    return v_int32x4(_v128_setr_w(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}
inline v_int32x4 v_lut_pairs(const int *tab, const int* idx)
{
    return v_int32x4(_v128_setr_d(*(const int64_t*)(tab + idx[0]), *(const int64_t*)(tab + idx[1])));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(__lsx_vld(tab + idx[0], 0));
}

inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((const int *)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((const int *)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((const int *)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int *idx)
{
    return v_int64x2(_v128_setr_d(tab[idx[0]], tab[idx[1]]));
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(__lsx_vld(tab + idx[0], 0));
}

inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    return v_float32x4(_v128_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx)
{
    return v_float32x4((__m128)_v128_setr_pd(*(const double*)(tab + idx[0]), *(const double*)(tab + idx[1])));
}
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{
    return v_float32x4((__m128)__lsx_vld(tab + idx[0], 0));
}

inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    return v_float64x2(_v128_setr_pd(tab[idx[0]], tab[idx[1]]));
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2((__m128d)__lsx_vld(tab + idx[0], 0));
}

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int *idx = (int*)&idxvec.val;
    return v_lut(tab, idx);
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    const int *idx = (const int*)&idxvec.val;
    return v_lut(tab, idx);
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    const int *idx = (const int*)&idxvec.val;
    return v_lut(tab, idx);
}

inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    const int *idx = (const int*)&idxvec.val;
    __m128i xy0  = __lsx_vld(tab + idx[0], 0);
    __m128i xy1  = __lsx_vld(tab + idx[1], 0);
    __m128i xy2  = __lsx_vld(tab + idx[2], 0);
    __m128i xy3  = __lsx_vld(tab + idx[3], 0);
    __m128i xy01 = __lsx_vilvl_d(xy1, xy0);
    __m128i xy23 = __lsx_vilvl_d(xy3, xy2);
    __m128i xxyy02 = __lsx_vilvl_w(xy23, xy01);
    __m128i xxyy13 = __lsx_vilvh_w(xy23, xy01);
    x = v_float32x4((__m128)__lsx_vilvl_w(xxyy13, xxyy02));
    y = v_float32x4((__m128)__lsx_vilvh_w(xxyy13, xxyy02));
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    const int* idx = (const int*)&idxvec.val;
    __m128i xy0 = __lsx_vld(tab + idx[0], 0);
    __m128i xy1 = __lsx_vld(tab + idx[1], 0);
    x = v_float64x2((__m128d)__lsx_vilvl_d(xy1, xy0));
    y = v_float64x2((__m128d)__lsx_vilvh_d(xy1, xy0));
}

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
    return v_int8x16(__lsx_vshuf_b(vec.val, vec.val,
                _v128_setr_d(0x0705060403010200, 0x0f0d0e0c0b090a08)));
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
    return v_int8x16(__lsx_vshuf_b(vec.val, vec.val,
                _v128_setr_d(0x0703060205010400, 0x0f0b0e0a0d090c08)));
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
    return v_int16x8(__lsx_vshuf_b(vec.val, vec.val,
                _v128_setr_d(0x0706030205040100, 0x0f0e0b0a0d0c0908)));
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec)
{ return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
    return v_int16x8(__lsx_vshuf_b(vec.val, vec.val,
                _v128_setr_d(0x0b0a030209080100, 0x0f0e07060d0c0504)));
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec)
{ return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    return v_int32x4(__lsx_vshuf4i_w(vec.val, 0xd8));
}
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec)
{ return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_float32x4 v_interleave_pairs(const v_float32x4& vec)
{ return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
    __m128i zero = __lsx_vldi(0);
    return v_int8x16(__lsx_vshuf_b(zero, vec.val,
           _v128_set_d(0x1211100f0e0d0c0a, 0x0908060504020100)));
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec)
{ return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
    __m128i zero = __lsx_vldi(0);
    return v_int16x8(__lsx_vshuf_b(zero, vec.val,
           _v128_set_d(0x11100f0e0d0c0b0a, 0x0908050403020100)));
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec)
{ return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

//////////// Matrix operations /////////

/////////// Dot Product /////////

// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    __m128i x = a.val, y = b.val;
    return v_int32x4(__lsx_vmaddwod_w_h(__lsx_vmulwev_w_h(x, y), x, y));
}
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    __m128i x = a.val, y = b.val, z = c.val;
    __m128i t = __lsx_vmaddwev_w_h(z, x, y);
    return v_int32x4(__lsx_vmaddwod_w_h(t, x, y));
}

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    __m128i x = a.val, y = b.val;
    return v_int64x2(__lsx_vmaddwod_d_w(__lsx_vmulwev_d_w(x, y), x, y));
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    __m128i x = a.val, y = b.val, z = c.val;
    __m128i t = __lsx_vmaddwev_d_w(z, x, y);
    return v_int64x2(__lsx_vmaddwod_d_w(t, x, y));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i x = a.val, y = b.val;
    __m128i even  = __lsx_vmulwev_h_bu(x, y);
    __m128i odd   = __lsx_vmulwod_h_bu(x, y);
    __m128i prod0 = __lsx_vhaddw_wu_hu(even, even);
    __m128i prod1 = __lsx_vhaddw_wu_hu(odd, odd);
    return v_uint32x4(__lsx_vadd_w(prod0, prod1));
}

inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_add(v_dotprod_expand(a, b), c) ;}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    __m128i x = a.val, y = b.val;
    __m128i even  = __lsx_vmulwev_h_b(x, y);
    __m128i odd   = __lsx_vmulwod_h_b(x, y);
    __m128i prod0 = __lsx_vhaddw_w_h(even, even);
    __m128i prod1 = __lsx_vhaddw_w_h(odd, odd);
    return v_int32x4(__lsx_vadd_w(prod0, prod1));
}
inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i x = a.val, y = b.val;
    __m128i even  = __lsx_vmulwev_w_hu(x, y);
    __m128i odd   = __lsx_vmulwod_w_hu(x, y);
    __m128i prod0 = __lsx_vhaddw_du_wu(even, even);
    __m128i prod1 = __lsx_vhaddw_du_wu(odd, odd);
    return v_uint64x2(__lsx_vadd_d(prod0, prod1));
}
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    __m128i x = a.val, y = b.val;
    __m128i even  = __lsx_vmulwev_w_h(x, y);
    __m128i odd   = __lsx_vmulwod_w_h(x, y);
    __m128i prod0 = __lsx_vhaddw_d_w(even, even);
    __m128i prod1 = __lsx_vhaddw_d_w(odd, odd);
    return v_int64x2(__lsx_vadd_d(prod0, prod1));
}
inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

//32 >> 64f
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }


///////// Fast Dot Product //////

// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod(a, b); }
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_dotprod(a, b, c); }

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod(a, b); }
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_dotprod(a, b, c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_dotprod_expand(a, b, c); }

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{ return v_dotprod_expand(a, b); }
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_dotprod_expand(a, b, c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i x = a.val, y = b.val;
    __m128i even  = __lsx_vmulwev_w_hu(x, y);
    __m128i odd   = __lsx_vmulwod_w_hu(x, y);
    __m128i prod0 = __lsx_vhaddw_du_wu(even, even);
    __m128i prod1 = __lsx_vhaddw_du_wu(odd, odd);
    return v_uint64x2(__lsx_vilvl_d(__lsx_vhaddw_qu_du(prod0, prod0), __lsx_vhaddw_qu_du(prod1, prod1)));
}
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    __m128i x = a.val, y = b.val;
    __m128i prod = __lsx_vmaddwod_w_h(__lsx_vmulwev_w_h(x, y), x, y);
    __m128i sign = __lsx_vsrai_w(prod, 31);
    __m128i lo   = __lsx_vilvl_w(sign, prod);
    __m128i hi   = __lsx_vilvh_w(sign, prod);
    return v_int64x2(__lsx_vadd_d(lo, hi));
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod_expand(a, b); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_dotprod_expand(a, b, c); }

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2, const v_float32x4& m3)
{
    __m128i x = (__m128i)v.val;
    __m128 v0 = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(x, 0x0), m0.val);
    __m128 v1 = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(x, 0x55), m1.val);
    __m128 v2 = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(x, 0xAA), m2.val);
    __m128 v3 = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(x, 0xFF), m3.val);

    return v_float32x4(__lsx_vfadd_s(__lsx_vfadd_s(v0, v1), __lsx_vfadd_s(v2, v3)));
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const  v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2, const v_float32x4& a)
{
    __m128i x = (__m128i)v.val;
    __m128 v0 = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(x, 0x0), m0.val);
    __m128 v1 = __lsx_vfmul_s((__m128)__lsx_vshuf4i_w(x, 0x55), m1.val);
    __m128 v2 = __lsx_vfmadd_s((__m128)__lsx_vshuf4i_w(x, 0xAA), m2.val, a.val);

    return v_float32x4(__lsx_vfadd_s(__lsx_vfadd_s(v0, v1), v2));
}

#define OPENCV_HAL_IMPL_LSX_TRANSPOSE4X4(_Tpvec, cast_from, cast_to)                          \
    inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1,                            \
                               const _Tpvec& a2, const _Tpvec& a3,                            \
                               _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3)                \
   {                                                                                          \
       __m128i t0 = cast_from(__lsx_vilvl_w(a1.val, a0.val));                                 \
       __m128i t1 = cast_from(__lsx_vilvl_w(a3.val, a2.val));                                 \
       __m128i t2 = cast_from(__lsx_vilvh_w(a1.val, a0.val));                                 \
       __m128i t3 = cast_from(__lsx_vilvh_w(a3.val, a2.val));                                 \
       b0.val = cast_to(__lsx_vilvl_d(t1, t0));                                               \
       b1.val = cast_to(__lsx_vilvh_d(t1, t0));                                               \
       b2.val = cast_to(__lsx_vilvl_d(t3, t2));                                               \
       b3.val = cast_to(__lsx_vilvh_d(t3, t2));                                               \
   }

OPENCV_HAL_IMPL_LSX_TRANSPOSE4X4(v_uint32x4, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_LSX_TRANSPOSE4X4(v_int32x4, OPENCV_HAL_NOP, OPENCV_HAL_NOP)

inline void v_transpose4x4(const v_float32x4& a0, const v_float32x4& a1,
                           const v_float32x4& a2, const v_float32x4& a3,
                           v_float32x4& b0, v_float32x4& b1, v_float32x4& b2, v_float32x4& b3)
{
    __m128i vec0 = (__m128i)a0.val, vec1 = (__m128i)a1.val;
    __m128i vec2 = (__m128i)a2.val, vec3 = (__m128i)a3.val;
    __m128i t0 = __lsx_vilvl_w(vec1, vec0);
    __m128i t1 = __lsx_vilvl_w(vec3, vec2);
    __m128i t2 = __lsx_vilvh_w(vec1, vec0);
    __m128i t3 = __lsx_vilvh_w(vec3, vec2);
    b0.val = __m128(__lsx_vilvl_d(t1, t0));
    b1.val = __m128(__lsx_vilvh_d(t1, t0));
    b2.val = __m128(__lsx_vilvl_d(t3, t2));
    b3.val = __m128(__lsx_vilvh_d(t3, t2));
}

////////////////// Value reordering ////////////////

/* Expand */
#define OPENCV_HAL_IMPL_LSX_EXPAND(_Tpvec, _Tpwvec, _Tp, intrin_lo, intrin_hi)     \
    inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1)                \
    {                                                                              \
        b0.val = intrin_lo(a.val, 0);                                              \
        b1.val = intrin_hi(a.val);                                                 \
    }                                                                              \
    inline _Tpwvec v_expand_low(const _Tpvec& a)                                   \
    { return _Tpwvec(intrin_lo(a.val, 0)); }                                       \
    inline _Tpwvec v_expand_high(const _Tpvec& a)                                  \
    { return _Tpwvec(intrin_hi(a.val)); }                                          \
    inline _Tpwvec v_load_expand(const _Tp* ptr)                                   \
    {                                                                              \
        __m128i a = __lsx_vld(ptr, 0);                                             \
        return _Tpwvec(intrin_lo(a, 0));                                           \
    }

OPENCV_HAL_IMPL_LSX_EXPAND(v_uint8x16, v_uint16x8, uchar,     __lsx_vsllwil_hu_bu, __lsx_vexth_hu_bu)
OPENCV_HAL_IMPL_LSX_EXPAND(v_int8x16,  v_int16x8,  schar,     __lsx_vsllwil_h_b,   __lsx_vexth_h_b)
OPENCV_HAL_IMPL_LSX_EXPAND(v_uint16x8, v_uint32x4, ushort,    __lsx_vsllwil_wu_hu, __lsx_vexth_wu_hu)
OPENCV_HAL_IMPL_LSX_EXPAND(v_int16x8,  v_int32x4,  short,     __lsx_vsllwil_w_h,   __lsx_vexth_w_h)
OPENCV_HAL_IMPL_LSX_EXPAND(v_uint32x4, v_uint64x2, unsigned,  __lsx_vsllwil_du_wu, __lsx_vexth_du_wu)
OPENCV_HAL_IMPL_LSX_EXPAND(v_int32x4,  v_int64x2,  int,       __lsx_vsllwil_d_w,   __lsx_vexth_d_w)

#define OPENCV_HAL_IMPL_LSX_EXPAND_Q(_Tpvec, _Tp, intrin_lo, intrin_hi)          \
    inline _Tpvec v_load_expand_q(const _Tp* ptr)                                \
    {                                                                            \
        __m128i a = __lsx_vld(ptr, 0);                                           \
        __m128i b = intrin_lo(a, 0);                                             \
        return _Tpvec(intrin_hi(b, 0));                                          \
    }

OPENCV_HAL_IMPL_LSX_EXPAND_Q(v_uint32x4, uchar, __lsx_vsllwil_hu_bu, __lsx_vsllwil_wu_hu)
OPENCV_HAL_IMPL_LSX_EXPAND_Q(v_int32x4,  schar, __lsx_vsllwil_h_b,   __lsx_vsllwil_w_h)

/* pack */
// 16
inline v_int8x16 v_pack(const v_int16x8& a, const v_int16x8& b)
{ return v_int8x16(_lsx_packs_h(a.val, b.val)); }

inline v_uint8x16 v_pack(const v_uint16x8& a, const v_uint16x8& b)
{ return v_uint8x16(__lsx_vssrlrni_bu_h(b.val, a.val, 0)); }

inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b)
{ return v_uint8x16(_lsx_packus_h(a.val, b.val)); }

inline void v_pack_store(schar* ptr, const v_int16x8& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(uchar* ptr, const v_uint16x8& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_u_store(uchar* ptr, const v_int16x8& a)
{ v_store_low(ptr, v_pack_u(a, a)); }

template<int n> inline
v_uint8x16 v_rshr_pack(const v_uint16x8& a, const v_uint16x8& b)
{ return v_uint8x16(__lsx_vssrlrni_bu_h(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x8& a)
{ __lsx_vstelm_d(__lsx_vssrlrni_bu_h(a.val, a.val, n), ptr, 0, 0); }

template<int n> inline
v_uint8x16 v_rshr_pack_u(const v_int16x8& a, const v_int16x8& b)
{ return v_uint8x16(__lsx_vssrarni_bu_h(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x8& a)
{ __lsx_vstelm_d(__lsx_vssrarni_bu_h(a.val, a.val, n), ptr, 0, 0); }

template<int n> inline
v_int8x16 v_rshr_pack(const v_int16x8& a, const v_int16x8& b)
{ return v_int8x16(__lsx_vssrarni_b_h(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x8& a)
{ __lsx_vstelm_d(__lsx_vssrarni_b_h(a.val, a.val, n), ptr, 0, 0); }

//32
inline v_int16x8 v_pack(const v_int32x4& a, const v_int32x4& b)
{ return v_int16x8(__lsx_vssrarni_h_w(b.val, a.val, 0)); }

inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
{ return v_uint16x8(__lsx_vssrlrni_hu_w(b.val, a.val, 0)); }

inline v_uint16x8 v_pack_u(const v_int32x4& a, const v_int32x4& b)
{ return v_uint16x8(__lsx_vssrarni_hu_w(b.val, a.val, 0)); }

inline void v_pack_store(short* ptr, const v_int32x4& a)
{ v_store_low(ptr, v_pack(a, a)); }

inline void v_pack_store(ushort *ptr, const v_uint32x4& a)
{ __lsx_vstelm_d(__lsx_vssrlrni_hu_w(a.val, a.val, 0), ptr,  0, 0); }

inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
{ __lsx_vstelm_d(__lsx_vssrarni_hu_w(a.val, a.val, 0), ptr, 0, 0); }

template<int n> inline
v_uint16x8 v_rshr_pack(const v_uint32x4& a, const v_uint32x4& b)
{ return v_uint16x8(__lsx_vssrlrni_hu_w(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x4& a)
{ __lsx_vstelm_d(__lsx_vssrlrni_hu_w(a.val, a.val, n), ptr, 0, 0); }

template<int n> inline
v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
{ return v_uint16x8(__lsx_vssrarni_hu_w(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
{ __lsx_vstelm_d(__lsx_vssrarni_hu_w(a.val, a.val, n), ptr, 0, 0); }

template<int n> inline
v_int16x8 v_rshr_pack(const v_int32x4& a, const v_int32x4& b)
{ return v_int16x8(__lsx_vssrarni_h_w(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x4& a)
{ __lsx_vstelm_d(__lsx_vssrarni_h_w(a.val, a.val, n), ptr, 0, 0); }

// 64
// Non-saturaing pack
inline v_uint32x4 v_pack(const v_uint64x2& a, const v_uint64x2& b)
{ return v_uint32x4(__lsx_vpickev_w(b.val, a.val)); }

inline v_int32x4 v_pack(const v_int64x2& a, const v_int64x2& b)
{ return v_reinterpret_as_s32(v_pack(v_reinterpret_as_u64(a), v_reinterpret_as_u64(b))); }

inline void v_pack_store(unsigned* ptr, const v_uint64x2& a)
{ __lsx_vstelm_d(__lsx_vshuf4i_w(a.val, 0x08), ptr, 0, 0); }

inline void v_pack_store(int *ptr, const v_int64x2& a)
{ v_pack_store((unsigned*)ptr, v_reinterpret_as_u64(a)); }

template<int n> inline
v_uint32x4 v_rshr_pack(const v_uint64x2& a, const v_uint64x2& b)
{ return v_uint32x4(__lsx_vsrlrni_w_d(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x2& a)
{ __lsx_vstelm_d(__lsx_vsrlrni_w_d(a.val, a.val, n), ptr, 0, 0); }

template<int n> inline
v_int32x4 v_rshr_pack(const v_int64x2& a, const v_int64x2& b)
{ return v_int32x4(__lsx_vsrarni_w_d(b.val, a.val, n)); }

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x2& a)
{ __lsx_vstelm_d(__lsx_vsrarni_w_d(a.val, a.val, n), ptr, 0, 0); }

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{ return v_uint8x16(__lsx_vssrarni_b_h(b.val, a.val, 0)); }

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    __m128i ab = __lsx_vssrarni_h_w(b.val, a.val, 0);
    __m128i cd = __lsx_vssrarni_h_w(d.val, c.val, 0);
    return v_uint8x16(__lsx_vssrarni_b_h(cd, ab, 0));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    __m128i ab = __lsx_vssrarni_w_d(b.val, a.val, 0);
    __m128i cd = __lsx_vssrarni_w_d(d.val, c.val, 0);
    __m128i ef = __lsx_vssrarni_w_d(f.val, e.val, 0);
    __m128i gh = __lsx_vssrarni_w_d(h.val, g.val, 0);

    __m128i abcd = __lsx_vssrarni_h_w(cd, ab, 0);
    __m128i efgh = __lsx_vssrarni_h_w(gh, ef, 0);
    return v_uint8x16(__lsx_vssrarni_b_h(efgh, abcd, 0));
}

/* Recombine */
// its up there with load and store operations

/* Extract */
#define OPENCV_HAL_IMPL_LSX_EXTRACT(_Tpvec)                    \
    template<int s>                                            \
    inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)  \
    { return v_rotate_right<s>(a, b); }

OPENCV_HAL_IMPL_LSX_EXTRACT(v_uint8x16)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_int8x16)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_uint16x8)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_int16x8)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_uint32x4)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_int32x4)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_uint64x2)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_int64x2)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_float32x4)
OPENCV_HAL_IMPL_LSX_EXTRACT(v_float64x2)

#define OPENCV_HAL_IMPL_LSX_EXTRACT_N(_Tpvec, _Twvec, intrin)             \
template<int i>                                                           \
inline _Twvec v_extract_n(const _Tpvec& a)                                \
{ return (_Twvec)intrin(a.val, i); }

OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_uint8x16, uchar,   __lsx_vpickve2gr_b)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_int8x16,  schar,   __lsx_vpickve2gr_b)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_uint16x8, ushort,  __lsx_vpickve2gr_h)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_int16x8,  short,   __lsx_vpickve2gr_h)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_uint32x4, uint,    __lsx_vpickve2gr_w)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_int32x4,  int,     __lsx_vpickve2gr_w)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_uint64x2, uint64,  __lsx_vpickve2gr_d)
OPENCV_HAL_IMPL_LSX_EXTRACT_N(v_int64x2,  int64,   __lsx_vpickve2gr_d)

template<int i>
inline float v_extract_n(const v_float32x4& v)
{
    union { uint iv; float fv; } d;
    d.iv = __lsx_vpickve2gr_w(v.val, i);
    return d.fv;
}

template<int i>
inline double v_extract_n(const v_float64x2& v)
{
    union { uint64 iv; double dv; } d;
    d.iv = __lsx_vpickve2gr_d(v.val, i);
    return d.dv;
}

template<int i>
inline v_uint32x4 v_broadcast_element(const v_uint32x4& a)
{ return v_uint32x4(__lsx_vreplvei_w(a.val, i)); }

template<int i>
inline v_int32x4 v_broadcast_element(const v_int32x4& a)
{ return v_int32x4(__lsx_vreplvei_w(a.val, i)); }

template<int i>
inline v_float32x4 v_broadcast_element(const v_float32x4& a)
{ return v_float32x4((__m128)__lsx_vreplvei_w((__m128i)a.val, i)); }

/////////////////// load deinterleave //////////////////////////////

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);

    a.val = __lsx_vpickev_b(t1, t0);
    b.val = __lsx_vpickod_b(t1, t0);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    a.val = __lsx_vpickev_h(t1, t0);
    b.val = __lsx_vpickod_h(t1, t0);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    a.val = __lsx_vpickev_w(t1, t0);
    b.val = __lsx_vpickod_w(t1, t0);
}

inline void v_load_deinterleave(const uint64* ptr, v_uint64x2& a, v_uint64x2& b)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    a.val = __lsx_vilvl_d(t1, t0);
    b.val = __lsx_vilvh_d(t1, t0);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);
    const __m128i shuff0 = _v128_setr_b(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i shuff1 = _v128_setr_b(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i a0 = __lsx_vbitsel_v(t0, t1, shuff0);
    __m128i b0 = __lsx_vbitsel_v(t1, t0, shuff1);
    __m128i c0 = __lsx_vbitsel_v(t1, t0, shuff0);
    const __m128i shuff_a = _v128_setr_b(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 17, 20, 23, 26, 29);
    const __m128i shuff_b = _v128_setr_b(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30);
    const __m128i shuff_c = _v128_setr_b(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31);

    a.val = __lsx_vshuf_b(t2, a0, shuff_a);
    b.val = __lsx_vshuf_b(t2, b0, shuff_b);
    c.val = __lsx_vshuf_b(t2, c0, shuff_c);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);
    const __m128i shuff0 = _v128_setr_h(0, 0, -1, 0, 0, -1, 0, 0);
    const __m128i shuff1 = _v128_setr_h(0, -1, 0, 0, -1, 0, 0, -1);

    __m128i a0 = __lsx_vbitsel_v(t0, t1, shuff1);
    __m128i b0 = __lsx_vbitsel_v(t0, t1, shuff0);
    __m128i c0 = __lsx_vbitsel_v(t1, t0, shuff0);

    const __m128i shuff_a = _v128_setr_b(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 20, 21, 26, 27);
    const __m128i shuff_b = _v128_setr_b(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 16, 17, 22, 23, 28, 29);
    const __m128i shuff_c = _v128_setr_b(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 18, 19, 24, 25, 30, 31);

    a.val = __lsx_vshuf_b(t2, a0, shuff_a);
    b.val = __lsx_vshuf_b(t2, b0, shuff_b);
    c.val = __lsx_vshuf_b(t2, c0, shuff_c);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);

    __m128i a0 = __lsx_vpermi_w(t1, t0, 0xAC);
    __m128i b0 = __lsx_vpermi_w(t1, t0, 0xC5);
    __m128i c0 = __lsx_vpermi_w(t1, t0, 0x5A);

    a.val = __lsx_vextrins_w(a0, t2, 0x31);
    b0    = __lsx_vshuf4i_w(b0, 0x38);
    c0    = __lsx_vshuf4i_w(c0, 0x8);
    b.val = __lsx_vextrins_w(b0, t2, 0x32);
    c.val = __lsx_vpermi_w(t2, c0, 0xC4);
}

inline void v_load_deinterleave(const uint64* ptr, v_uint64x2& a, v_uint64x2& b, v_uint64x2& c)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);

    a.val = __lsx_vshuf4i_d(t0, t1, 0xC);
    b.val = __lsx_vshuf4i_d(t0, t2, 0x9);
    c.val = __lsx_vshuf4i_d(t1, t2, 0xC);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c, v_uint8x16& d)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);
    __m128i t3 = __lsx_vld(ptr, 48);

    __m128i ac_lo = __lsx_vpickev_b(t1, t0);
    __m128i bd_lo = __lsx_vpickod_b(t1, t0);
    __m128i ac_hi = __lsx_vpickev_b(t3, t2);
    __m128i bd_hi = __lsx_vpickod_b(t3, t2);

    a.val = __lsx_vpickev_b(ac_hi, ac_lo);
    c.val = __lsx_vpickod_b(ac_hi, ac_lo);
    b.val = __lsx_vpickev_b(bd_hi, bd_lo);
    d.val = __lsx_vpickod_b(bd_hi, bd_lo);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c, v_uint16x8& d)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);
    __m128i t3 = __lsx_vld(ptr, 48);

    __m128i ac_lo = __lsx_vpickev_h(t1, t0);
    __m128i bd_lo = __lsx_vpickod_h(t1, t0);
    __m128i ac_hi = __lsx_vpickev_h(t3, t2);
    __m128i bd_hi = __lsx_vpickod_h(t3, t2);

    a.val = __lsx_vpickev_h(ac_hi, ac_lo);
    c.val = __lsx_vpickod_h(ac_hi, ac_lo);
    b.val = __lsx_vpickev_h(bd_hi, bd_lo);
    d.val = __lsx_vpickod_h(bd_hi, bd_lo);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c, v_uint32x4& d)
{
    __m128i p0 = __lsx_vld(ptr, 0);
    __m128i p1 = __lsx_vld(ptr, 16);
    __m128i p2 = __lsx_vld(ptr, 32);
    __m128i p3 = __lsx_vld(ptr, 48);

    __m128i t0 = __lsx_vilvl_w(p1, p0);
    __m128i t1 = __lsx_vilvl_w(p3, p2);
    __m128i t2 = __lsx_vilvh_w(p1, p0);
    __m128i t3 = __lsx_vilvh_w(p3, p2);
    a.val = __lsx_vilvl_d(t1, t0);
    b.val = __lsx_vilvh_d(t1, t0);
    c.val = __lsx_vilvl_d(t3, t2);
    d.val = __lsx_vilvh_d(t3, t2);
}

inline void v_load_deinterleave(const uint64* ptr, v_uint64x2& a, v_uint64x2& b, v_uint64x2& c, v_uint64x2& d)
{
    __m128i t0 = __lsx_vld(ptr, 0);
    __m128i t1 = __lsx_vld(ptr, 16);
    __m128i t2 = __lsx_vld(ptr, 32);
    __m128i t3 = __lsx_vld(ptr, 48);

    a.val = __lsx_vilvl_d(t2, t0);
    b.val = __lsx_vilvh_d(t2, t0);
    c.val = __lsx_vilvl_d(t3, t1);
    d.val = __lsx_vilvh_d(t3, t1);
}

////////////////////////// store interleave ////////////////////////////////

inline void v_store_interleave(uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i v0 = __lsx_vilvl_b(b.val, a.val);
    __m128i v1 = __lsx_vilvh_b(b.val, a.val);

    __lsx_vst(v0, ptr, 0);
    __lsx_vst(v1, ptr, 16);
}

inline void v_store_interleave(ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i v0 = __lsx_vilvl_h(b.val, a.val);
    __m128i v1 = __lsx_vilvh_h(b.val, a.val);

    __lsx_vst(v0, ptr, 0);
    __lsx_vst(v1, ptr, 16);
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i v0 = __lsx_vilvl_w(b.val, a.val);
    __m128i v1 = __lsx_vilvh_w(b.val, a.val);

    __lsx_vst(v0, ptr, 0);
    __lsx_vst(v1, ptr, 16);
}

inline void v_store_interleave(uint64* ptr, const v_uint64x2& a, const v_uint64x2& b,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i v0 = __lsx_vilvl_d(b.val, a.val);
    __m128i v1 = __lsx_vilvh_d(b.val, a.val);

    __lsx_vst(v0, ptr, 0);
    __lsx_vst(v1, ptr, 16);
}

inline void v_store_interleave(uchar* ptr, const v_uint8x16& a, const v_uint8x16& b, const v_uint8x16& c,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i ab_lo = __lsx_vilvl_b(b.val, a.val);
    __m128i ab_hi = __lsx_vilvh_b(b.val, a.val);
    __m128i v_c = c.val;
    const __m128i shuff0 = _v128_setr_b(0, 1, 16, 2, 3, 17, 4, 5, 18, 6, 7, 19, 8, 9, 20, 10);
    const __m128i shuff1 = _v128_setr_b(11, 21, 12, 13, 22, 14, 15, 23, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i shuff2 = _v128_setr_b(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 24, 18, 19, 25, 20, 21);
    const __m128i shuff3 = _v128_setr_b(26, 6, 7, 27, 8, 9, 28, 10, 11, 29, 12, 13, 30, 14, 15, 31);
    __m128i abc = __lsx_vpermi_w(v_c, ab_hi, 0xE4);

    __m128i dst0 = __lsx_vshuf_b(v_c, ab_lo, shuff0);
    __m128i dst1 = __lsx_vshuf_b(v_c, ab_lo, shuff1);
    __m128i dst2 = __lsx_vshuf_b(v_c, ab_hi, shuff3);
    dst1 = __lsx_vshuf_b(abc, dst1, shuff2);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
}

inline void v_store_interleave(ushort* ptr, const v_uint16x8& a, const v_uint16x8& b, const v_uint16x8& c,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i ab_lo = __lsx_vilvl_h(b.val, a.val);
    __m128i ab_hi = __lsx_vilvh_h(b.val, a.val);
    __m128i v_c = c.val;
    const __m128i shuff0 = _v128_setr_b(0, 1, 2, 3, 16, 17, 4, 5, 6, 7, 18, 19, 8, 9, 10, 11);
    const __m128i shuff1 = _v128_setr_b(20, 21, 12, 13, 14, 15, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i shuff2 = _v128_setr_b(0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 24, 25, 20, 21);
    const __m128i shuff3 = _v128_setr_b(6, 7, 26, 27, 8, 9, 10, 11, 28, 29, 12, 13, 14, 15, 30, 31);
    __m128i abc = __lsx_vpermi_w(v_c, ab_hi, 0xE4);

    __m128i dst0 = __lsx_vshuf_b(v_c, ab_lo, shuff0);
    __m128i dst1 = __lsx_vshuf_b(v_c, ab_lo, shuff1);
    __m128i dst2 = __lsx_vshuf_b(v_c, ab_hi, shuff3);
    dst1 = __lsx_vshuf_b(abc, dst1, shuff2);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b, const v_uint32x4& c,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i v_c = c.val;
    __m128i ab_lo = __lsx_vilvl_w(b.val, a.val);  //a0 b0 a1 b1
    __m128i ab_hi = __lsx_vilvh_w(b.val, a.val);  //a2 b2 a3 b3
    __m128i bc_od = __lsx_vpackod_w(v_c, b.val); // b1 c1 b3 c3

    __m128i dst0 = __lsx_vshuf4i_w(ab_lo, 0xB4);  //a0 b0 b1 a1
    __m128i dst1 = __lsx_vilvl_d(ab_hi, bc_od); //b1 c1 a2 b2
    __m128i dst2 = __lsx_vpermi_w(bc_od, ab_hi, 0xE8); //a2, a3, b3, c3

    dst0 = __lsx_vextrins_w(dst0, v_c, 0x20);
    dst2 = __lsx_vextrins_w(dst2, v_c, 0x2);
    __lsx_vst(dst0, ptr, 0);  //a0 b0 c0 a1
    __lsx_vst(dst1, ptr, 16); //b1 c1 a2 b2
    __lsx_vst(dst2, ptr, 32); //c2 a3 b3 c3
}

inline void v_store_interleave(uint64* ptr, const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i dst0 = __lsx_vilvl_d(b.val, a.val);
    __m128i dst1 = __lsx_vpermi_w(a.val, c.val, 0xE4);
    __m128i dst2 = __lsx_vilvh_d(c.val, b.val);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
}

inline void v_store_interleave(uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                               const v_uint8x16& c, const v_uint8x16& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i ab_lo = __lsx_vilvl_b(b.val, a.val);
    __m128i ab_hi = __lsx_vilvh_b(b.val, a.val);
    __m128i cd_lo = __lsx_vilvl_b(d.val, c.val);
    __m128i cd_hi = __lsx_vilvh_b(d.val, c.val);

    __m128i dst0 = __lsx_vilvl_h(cd_lo, ab_lo);
    __m128i dst1 = __lsx_vilvh_h(cd_lo, ab_lo);
    __m128i dst2 = __lsx_vilvl_h(cd_hi, ab_hi);
    __m128i dst3 = __lsx_vilvh_h(cd_hi, ab_hi);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
    __lsx_vst(dst3, ptr, 48);
}

inline void v_store_interleave(ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                               const v_uint16x8& c, const v_uint16x8& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i ab_lo = __lsx_vilvl_h(b.val, a.val);
    __m128i ab_hi = __lsx_vilvh_h(b.val, a.val);
    __m128i cd_lo = __lsx_vilvl_h(d.val, c.val);
    __m128i cd_hi = __lsx_vilvh_h(d.val, c.val);

    __m128i dst0 = __lsx_vilvl_w(cd_lo, ab_lo);
    __m128i dst1 = __lsx_vilvh_w(cd_lo, ab_lo);
    __m128i dst2 = __lsx_vilvl_w(cd_hi, ab_hi);
    __m128i dst3 = __lsx_vilvh_w(cd_hi, ab_hi);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
    __lsx_vst(dst3, ptr, 48);
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               const v_uint32x4& c, const v_uint32x4& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i ab_lo = __lsx_vilvl_w(b.val, a.val);
    __m128i ab_hi = __lsx_vilvh_w(b.val, a.val);
    __m128i cd_lo = __lsx_vilvl_w(d.val, c.val);
    __m128i cd_hi = __lsx_vilvh_w(d.val, c.val);

    __m128i dst0 = __lsx_vilvl_d(cd_lo, ab_lo);
    __m128i dst1 = __lsx_vilvh_d(cd_lo, ab_lo);
    __m128i dst2 = __lsx_vilvl_d(cd_hi, ab_hi);
    __m128i dst3 = __lsx_vilvh_d(cd_hi, ab_hi);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
    __lsx_vst(dst3, ptr, 48);
}

inline void v_store_interleave(uint64* ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, const v_uint64x2& d,
                               hal::StoreMode /*mode*/ = hal::STORE_UNALIGNED)
{
    __m128i dst0 = __lsx_vilvl_d(b.val, a.val);
    __m128i dst2 = __lsx_vilvh_d(b.val, a.val);
    __m128i dst1 = __lsx_vilvl_d(d.val, c.val);
    __m128i dst3 = __lsx_vilvh_d(d.val, c.val);

    __lsx_vst(dst0, ptr, 0);
    __lsx_vst(dst1, ptr, 16);
    __lsx_vst(dst2, ptr, 32);
    __lsx_vst(dst3, ptr, 48);
}

#define OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1)  \
inline void v_load_deinterleave(const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0)                        \
{                                                                                                 \
    _Tpvec1 a1, b1;                                                                               \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1);                                                \
    a0 = v_reinterpret_as_##suffix0(a1);                                                          \
    b0 = v_reinterpret_as_##suffix0(b1);                                                          \
}                                                                                                 \
inline void v_load_deinterleave(const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0)           \
{                                                                                                 \
    _Tpvec1 a1, b1, c1;                                                                           \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1);                                            \
    a0 = v_reinterpret_as_##suffix0(a1);                                                          \
    b0 = v_reinterpret_as_##suffix0(b1);                                                          \
    c0 = v_reinterpret_as_##suffix0(c1);                                                          \
}                                                                                                 \
inline void v_load_deinterleave(const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0,                        \
                                _Tpvec0& c0, _Tpvec0& d0)                                         \
{                                                                                                 \
    _Tpvec1 a1, b1, c1, d1;                                                                       \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1, d1);                                        \
    a0 = v_reinterpret_as_##suffix0(a1);                                                          \
    b0 = v_reinterpret_as_##suffix0(b1);                                                          \
    c0 = v_reinterpret_as_##suffix0(c1);                                                          \
    d0 = v_reinterpret_as_##suffix0(d1);                                                          \
}                                                                                                 \
inline void v_store_interleave(_Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0,                   \
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)                      \
{                                                                                                 \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0);                                                  \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0);                                                  \
    v_store_interleave((_Tp1*)ptr, a1, b1);                                                     \
}                                                                                                 \
inline void v_store_interleave(_Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, const _Tpvec0& c0,\
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)                      \
{                                                                                                 \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0);                                                  \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0);                                                  \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0);                                                  \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1);                                                 \
}                                                                                                 \
inline void v_store_interleave(_Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0,                   \
                               const _Tpvec0& c0, const _Tpvec0& d0,                              \
                               hal::StoreMode /*mode*/=hal::STORE_UNALIGNED)                      \
{                                                                                                 \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0);                                                  \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0);                                                  \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0);                                                  \
    _Tpvec1 d1 = v_reinterpret_as_##suffix1(d0);                                                  \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1);                                             \
}

OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(v_int8x16, schar, s8, v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(v_int16x8, short, s16, v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(v_int32x4, int, s32, v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(v_float32x4, float, f32, v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(v_int64x2, int64, s64, v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_LSX_LOADSTORE_INTERLEAVE(v_float64x2, double, f64, v_uint64x2, uint64, u64)

//
// FP16
//

inline v_float32x4 v_load_expand(const hfloat* ptr)
{
#if CV_FP16
    return v_float32x4(__lsx_vfcvtl_s_h((__m128)__lsx_vld(ptr, 0)));
#else
    float CV_DECL_ALIGNED(32) buf[4];
    for (int i = 0; i < 4; i++)
        buf[i] = (float)ptr[i];
    return v_float32x4((__m128)__lsx_vld(buf, 0));
#endif
}

inline void v_pack_store(hfloat* ptr, const v_float32x4& a)
{
#if CV_FP16
    __m128i res = (__m218i)__lsx_vfcvt_h_s(a.val, a.val);
    __lsx_vstelm_d(res, ptr, 0, 0);
#else
    float CV_DECL_ALIGNED(32) buf[4];
    v_store_aligned(buf, a);
    for (int i = 0; i < 4; i++)
        ptr[i] = hfloat(buf[i]);
#endif
}

//
// end of FP16
//

inline void v_cleanup() {}

#include "intrin_math.hpp"
inline v_float32x4 v_exp(v_float32x4 x) { return v_exp_default_32f<v_float32x4, v_int32x4>(x); }
inline v_float32x4 v_log(v_float32x4 x) { return v_log_default_32f<v_float32x4, v_int32x4>(x); }
inline v_float32x4 v_erf(v_float32x4 x) { return v_erf_default_32f<v_float32x4, v_int32x4>(x); }

inline v_float64x2 v_exp(v_float64x2 x) { return v_exp_default_64f<v_float64x2, v_int64x2>(x); }
inline v_float64x2 v_log(v_float64x2 x) { return v_log_default_64f<v_float64x2, v_int64x2>(x); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::

#endif // OPENCV_HAL_INTRIN_LSX_HPP
