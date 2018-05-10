// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef OPENCV_HAL_VSX_HPP
#define OPENCV_HAL_VSX_HPP

#include <algorithm>
#include "opencv2/core/utility.hpp"

#define CV_SIMD128 1
#define CV_SIMD128_64F 1

/**
 * todo: supporting half precision for power9
 * convert instractions xvcvhpsp, xvcvsphp
**/

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Types ////////////

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };
    vec_uchar16 val;

    explicit v_uint8x16(const vec_uchar16& v) : val(v)
    {}
    v_uint8x16() : val(vec_uchar16_z)
    {}
    v_uint8x16(vec_bchar16 v) : val(vec_uchar16_c(v))
    {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
        : val(vec_uchar16_set(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15))
    {}
    uchar get0() const
    { return vec_extract(val, 0); }
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };
    vec_char16 val;

    explicit v_int8x16(const vec_char16& v) : val(v)
    {}
    v_int8x16() : val(vec_char16_z)
    {}
    v_int8x16(vec_bchar16 v) : val(vec_char16_c(v))
    {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
              schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
        : val(vec_char16_set(v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15))
    {}
    schar get0() const
    { return vec_extract(val, 0); }
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };
    vec_ushort8 val;

    explicit v_uint16x8(const vec_ushort8& v) : val(v)
    {}
    v_uint16x8() : val(vec_ushort8_z)
    {}
    v_uint16x8(vec_bshort8 v) : val(vec_ushort8_c(v))
    {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
        : val(vec_ushort8_set(v0, v1, v2, v3, v4, v5, v6, v7))
    {}
    ushort get0() const
    { return vec_extract(val, 0); }
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };
    vec_short8 val;

    explicit v_int16x8(const vec_short8& v) : val(v)
    {}
    v_int16x8() : val(vec_short8_z)
    {}
    v_int16x8(vec_bshort8 v) : val(vec_short8_c(v))
    {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
        : val(vec_short8_set(v0, v1, v2, v3, v4, v5, v6, v7))
    {}
    short get0() const
    { return vec_extract(val, 0); }
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };
    vec_uint4 val;

    explicit v_uint32x4(const vec_uint4& v) : val(v)
    {}
    v_uint32x4() : val(vec_uint4_z)
    {}
    v_uint32x4(vec_bint4 v) : val(vec_uint4_c(v))
    {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3) : val(vec_uint4_set(v0, v1, v2, v3))
    {}
    uint get0() const
    { return vec_extract(val, 0); }
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };
    vec_int4 val;

    explicit v_int32x4(const vec_int4& v) : val(v)
    {}
    v_int32x4() : val(vec_int4_z)
    {}
    v_int32x4(vec_bint4 v) : val(vec_int4_c(v))
    {}
    v_int32x4(int v0, int v1, int v2, int v3) : val(vec_int4_set(v0, v1, v2, v3))
    {}
    int get0() const
    { return vec_extract(val, 0); }
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };
    vec_float4 val;

    explicit v_float32x4(const vec_float4& v) : val(v)
    {}
    v_float32x4() : val(vec_float4_z)
    {}
    v_float32x4(vec_bint4 v) : val(vec_float4_c(v))
    {}
    v_float32x4(float v0, float v1, float v2, float v3) : val(vec_float4_set(v0, v1, v2, v3))
    {}
    float get0() const
    { return vec_extract(val, 0); }
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };
    vec_udword2 val;

    explicit v_uint64x2(const vec_udword2& v) : val(v)
    {}
    v_uint64x2() : val(vec_udword2_z)
    {}
    v_uint64x2(vec_bdword2 v) : val(vec_udword2_c(v))
    {}
    v_uint64x2(uint64 v0, uint64 v1) : val(vec_udword2_set(v0, v1))
    {}
    uint64 get0() const
    { return vec_extract(val, 0); }
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };
    vec_dword2 val;

    explicit v_int64x2(const vec_dword2& v) : val(v)
    {}
    v_int64x2() : val(vec_dword2_z)
    {}
    v_int64x2(vec_bdword2 v) : val(vec_dword2_c(v))
    {}
    v_int64x2(int64 v0, int64 v1) : val(vec_dword2_set(v0, v1))
    {}
    int64 get0() const
    { return vec_extract(val, 0); }
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };
    vec_double2 val;

    explicit v_float64x2(const vec_double2& v) : val(v)
    {}
    v_float64x2() : val(vec_double2_z)
    {}
    v_float64x2(vec_bdword2 v) : val(vec_double2_c(v))
    {}
    v_float64x2(double v0, double v1) : val(vec_double2_set(v0, v1))
    {}
    double get0() const
    { return vec_extract(val, 0); }
};

//////////////// Load and store operations ///////////////

/*
 * clang-5 aborted during parse "vec_xxx_c" only if it's
 * inside a function template which is defined by preprocessor macro.
 *
 * if vec_xxx_c defined as C++ cast, clang-5 will pass it
*/
#define OPENCV_HAL_IMPL_VSX_INITVEC(_Tpvec, _Tp, suffix, cast)                        \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(); }                               \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(vec_splats((_Tp)v));}          \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0 &a)  \
{ return _Tpvec((cast)a.val); }

OPENCV_HAL_IMPL_VSX_INITVEC(v_uint8x16, uchar, u8, vec_uchar16)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int8x16, schar, s8, vec_char16)
OPENCV_HAL_IMPL_VSX_INITVEC(v_uint16x8, ushort, u16, vec_ushort8)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int16x8, short, s16, vec_short8)
OPENCV_HAL_IMPL_VSX_INITVEC(v_uint32x4, uint, u32, vec_uint4)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int32x4, int, s32, vec_int4)
OPENCV_HAL_IMPL_VSX_INITVEC(v_uint64x2, uint64, u64, vec_udword2)
OPENCV_HAL_IMPL_VSX_INITVEC(v_int64x2, int64, s64, vec_dword2)
OPENCV_HAL_IMPL_VSX_INITVEC(v_float32x4, float, f32, vec_float4)
OPENCV_HAL_IMPL_VSX_INITVEC(v_float64x2, double, f64, vec_double2)

#define OPENCV_HAL_IMPL_VSX_LOADSTORE_C(_Tpvec, _Tp, ld, ld_a, st, st_a)    \
inline _Tpvec v_load(const _Tp* ptr)                                        \
{ return _Tpvec(ld(0, ptr)); }                                              \
inline _Tpvec v_load_aligned(VSX_UNUSED(const _Tp* ptr))                    \
{ return _Tpvec(ld_a(0, ptr)); }                                            \
inline _Tpvec v_load_low(const _Tp* ptr)                                    \
{ return _Tpvec(vec_ld_l8(ptr)); }                                          \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1)               \
{ return _Tpvec(vec_mergesqh(vec_ld_l8(ptr0), vec_ld_l8(ptr1))); }          \
inline void v_store(_Tp* ptr, const _Tpvec& a)                              \
{ st(a.val, 0, ptr); }                                                      \
inline void v_store_aligned(VSX_UNUSED(_Tp* ptr), const _Tpvec& a)          \
{ st_a(a.val, 0, ptr); }                                                    \
inline void v_store_low(_Tp* ptr, const _Tpvec& a)                          \
{ vec_st_l8(a.val, ptr); }                                                  \
inline void v_store_high(_Tp* ptr, const _Tpvec& a)                         \
{ vec_st_h8(a.val, ptr); }

#define OPENCV_HAL_IMPL_VSX_LOADSTORE(_Tpvec, _Tp) \
OPENCV_HAL_IMPL_VSX_LOADSTORE_C(_Tpvec, _Tp, vsx_ld, vec_ld, vsx_st, vec_st)

OPENCV_HAL_IMPL_VSX_LOADSTORE(v_uint8x16,  uchar)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_int8x16,   schar)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_uint16x8,  ushort)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_int16x8,   short)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_uint32x4,  uint)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_int32x4,   int)
OPENCV_HAL_IMPL_VSX_LOADSTORE(v_float32x4, float)

OPENCV_HAL_IMPL_VSX_LOADSTORE_C(v_float64x2, double, vsx_ld,  vsx_ld,  vsx_st,  vsx_st)
OPENCV_HAL_IMPL_VSX_LOADSTORE_C(v_uint64x2,  uint64, vsx_ld2, vsx_ld2, vsx_st2, vsx_st2)
OPENCV_HAL_IMPL_VSX_LOADSTORE_C(v_int64x2,    int64, vsx_ld2, vsx_ld2, vsx_st2, vsx_st2)

//////////////// Value reordering ///////////////

/* de&interleave */
#define OPENCV_HAL_IMPL_VSX_INTERLEAVE(_Tp, _Tpvec)                          \
inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a, _Tpvec& b)        \
{ vec_ld_deinterleave(ptr, a.val, b.val);}                                   \
inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a,                   \
                                _Tpvec& b, _Tpvec& c)                        \
{ vec_ld_deinterleave(ptr, a.val, b.val, c.val); }                           \
inline void v_load_deinterleave(const _Tp* ptr, _Tpvec& a, _Tpvec& b,        \
                                                _Tpvec& c, _Tpvec& d)        \
{ vec_ld_deinterleave(ptr, a.val, b.val, c.val, d.val); }                    \
inline void v_store_interleave(_Tp* ptr, const _Tpvec& a, const _Tpvec& b)   \
{ vec_st_interleave(a.val, b.val, ptr); }                                    \
inline void v_store_interleave(_Tp* ptr, const _Tpvec& a,                    \
                               const _Tpvec& b, const _Tpvec& c)             \
{ vec_st_interleave(a.val, b.val, c.val, ptr); }                             \
inline void v_store_interleave(_Tp* ptr, const _Tpvec& a, const _Tpvec& b,   \
                                         const _Tpvec& c, const _Tpvec& d)   \
{ vec_st_interleave(a.val, b.val, c.val, d.val, ptr); }

OPENCV_HAL_IMPL_VSX_INTERLEAVE(uchar, v_uint8x16)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(schar, v_int8x16)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(ushort, v_uint16x8)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(short, v_int16x8)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(uint, v_uint32x4)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(int, v_int32x4)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(float, v_float32x4)
OPENCV_HAL_IMPL_VSX_INTERLEAVE(double, v_float64x2)

/* Expand */
#define OPENCV_HAL_IMPL_VSX_EXPAND(_Tpvec, _Tpwvec, _Tp, fl, fh)  \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1)   \
{                                                                 \
    b0.val = fh(a.val);                                           \
    b1.val = fl(a.val);                                           \
}                                                                 \
inline _Tpwvec v_load_expand(const _Tp* ptr)                      \
{ return _Tpwvec(fh(vec_ld_l8(ptr))); }

OPENCV_HAL_IMPL_VSX_EXPAND(v_uint8x16, v_uint16x8, uchar, vec_unpacklu, vec_unpackhu)
OPENCV_HAL_IMPL_VSX_EXPAND(v_int8x16, v_int16x8, schar, vec_unpackl, vec_unpackh)
OPENCV_HAL_IMPL_VSX_EXPAND(v_uint16x8, v_uint32x4, ushort, vec_unpacklu, vec_unpackhu)
OPENCV_HAL_IMPL_VSX_EXPAND(v_int16x8, v_int32x4, short, vec_unpackl, vec_unpackh)
OPENCV_HAL_IMPL_VSX_EXPAND(v_uint32x4, v_uint64x2, uint, vec_unpacklu, vec_unpackhu)
OPENCV_HAL_IMPL_VSX_EXPAND(v_int32x4, v_int64x2, int, vec_unpackl, vec_unpackh)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{ return v_uint32x4(vec_uint4_set(ptr[0], ptr[1], ptr[2], ptr[3])); }

inline v_int32x4 v_load_expand_q(const schar* ptr)
{ return v_int32x4(vec_int4_set(ptr[0], ptr[1], ptr[2], ptr[3])); }

/* pack */
#define OPENCV_HAL_IMPL_VSX_PACK(_Tpvec, _Tp, _Tpwvec, _Tpvn, _Tpdel, sfnc, pkfnc, addfnc, pack)    \
inline _Tpvec v_##pack(const _Tpwvec& a, const _Tpwvec& b)                                          \
{                                                                                                   \
    return _Tpvec(pkfnc(a.val, b.val));                                                             \
}                                                                                                   \
inline void v_##pack##_store(_Tp* ptr, const _Tpwvec& a)                                            \
{                                                                                                   \
    vec_st_l8(pkfnc(a.val, a.val), ptr);                                                            \
}                                                                                                   \
template<int n>                                                                                     \
inline _Tpvec v_rshr_##pack(const _Tpwvec& a, const _Tpwvec& b)                                     \
{                                                                                                   \
    const __vector _Tpvn vn = vec_splats((_Tpvn)n);                                                 \
    const __vector _Tpdel delta = vec_splats((_Tpdel)((_Tpdel)1 << (n-1)));                         \
    return _Tpvec(pkfnc(sfnc(addfnc(a.val, delta), vn), sfnc(addfnc(b.val, delta), vn)));           \
}                                                                                                   \
template<int n>                                                                                     \
inline void v_rshr_##pack##_store(_Tp* ptr, const _Tpwvec& a)                                       \
{                                                                                                   \
    const __vector _Tpvn vn = vec_splats((_Tpvn)n);                                                 \
    const __vector _Tpdel delta = vec_splats((_Tpdel)((_Tpdel)1 << (n-1)));                         \
    vec_st_l8(pkfnc(sfnc(addfnc(a.val, delta), vn), delta), ptr);                                   \
}

OPENCV_HAL_IMPL_VSX_PACK(v_uint8x16, uchar, v_uint16x8, unsigned short, unsigned short,
                         vec_sr, vec_packs, vec_adds, pack)
OPENCV_HAL_IMPL_VSX_PACK(v_int8x16, schar, v_int16x8, unsigned short, short,
                         vec_sra, vec_packs, vec_adds, pack)

OPENCV_HAL_IMPL_VSX_PACK(v_uint16x8, ushort, v_uint32x4, unsigned int, unsigned int,
                         vec_sr, vec_packs, vec_add, pack)
OPENCV_HAL_IMPL_VSX_PACK(v_int16x8, short, v_int32x4, unsigned int, int,
                         vec_sra, vec_packs, vec_add, pack)

OPENCV_HAL_IMPL_VSX_PACK(v_uint32x4, uint, v_uint64x2, unsigned long long, unsigned long long,
                         vec_sr, vec_pack, vec_add, pack)
OPENCV_HAL_IMPL_VSX_PACK(v_int32x4, int, v_int64x2, unsigned long long, long long,
                         vec_sra, vec_pack, vec_add, pack)

OPENCV_HAL_IMPL_VSX_PACK(v_uint8x16, uchar, v_int16x8, unsigned short, short,
                         vec_sra, vec_packsu, vec_adds, pack_u)
OPENCV_HAL_IMPL_VSX_PACK(v_uint16x8, ushort, v_int32x4, unsigned int, int,
                         vec_sra, vec_packsu, vec_add, pack_u)
// Following variant is not implemented on other platforms:
//OPENCV_HAL_IMPL_VSX_PACK(v_uint32x4, uint, v_int64x2, unsigned long long, long long,
//                         vec_sra, vec_packsu, vec_add, pack_u)

/* Recombine */
template <typename _Tpvec>
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1)
{
    b0.val = vec_mergeh(a0.val, a1.val);
    b1.val = vec_mergel(a0.val, a1.val);
}

template <typename _Tpvec>
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(vec_mergesql(a.val, b.val)); }

template <typename _Tpvec>
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b)
{ return _Tpvec(vec_mergesqh(a.val, b.val)); }

template <typename _Tpvec>
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d)
{
    c.val = vec_mergesqh(a.val, b.val);
    d.val = vec_mergesql(a.val, b.val);
}

////////// Arithmetic, bitwise and comparison operations /////////

/* Element-wise binary and unary operations */
/** Arithmetics **/
#define OPENCV_HAL_IMPL_VSX_BIN_OP(bin_op, _Tpvec, intrin)       \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(intrin(a.val, b.val)); }                         \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b)   \
{ a.val = intrin(a.val, b.val); return a; }

OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_uint8x16, vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_uint8x16, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_int8x16,  vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_int8x16, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_uint16x8, vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_uint16x8, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(*, v_uint16x8, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_int16x8, vec_adds)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_int16x8, vec_subs)
OPENCV_HAL_IMPL_VSX_BIN_OP(*, v_int16x8, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_uint32x4, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_uint32x4, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(*, v_uint32x4, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_int32x4, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_int32x4, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(*, v_int32x4, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_float32x4, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_float32x4, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(*, v_float32x4, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(/, v_float32x4, vec_div)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_float64x2, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_float64x2, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(*, v_float64x2, vec_mul)
OPENCV_HAL_IMPL_VSX_BIN_OP(/, v_float64x2, vec_div)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_uint64x2, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_uint64x2, vec_sub)
OPENCV_HAL_IMPL_VSX_BIN_OP(+, v_int64x2, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_OP(-, v_int64x2, vec_sub)

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b, v_int32x4& c, v_int32x4& d)
{
    c.val = vec_mul(vec_unpackh(a.val), vec_unpackh(b.val));
    d.val = vec_mul(vec_unpackl(a.val), vec_unpackl(b.val));
}
inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b, v_uint32x4& c, v_uint32x4& d)
{
    c.val = vec_mul(vec_unpackhu(a.val), vec_unpackhu(b.val));
    d.val = vec_mul(vec_unpacklu(a.val), vec_unpacklu(b.val));
}
inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b, v_uint64x2& c, v_uint64x2& d)
{
    c.val = vec_mul(vec_unpackhu(a.val), vec_unpackhu(b.val));
    d.val = vec_mul(vec_unpacklu(a.val), vec_unpacklu(b.val));
}

/** Non-saturating arithmetics **/
#define OPENCV_HAL_IMPL_VSX_BIN_FUNC(func, intrin)    \
template<typename _Tpvec>                             \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b)  \
{ return _Tpvec(intrin(a.val, b.val)); }

OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_add_wrap, vec_add)
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_sub_wrap, vec_sub)

/** Bitwise shifts **/
#define OPENCV_HAL_IMPL_VSX_SHIFT_OP(_Tpvec, shr, splfunc)   \
inline _Tpvec operator << (const _Tpvec& a, int imm)         \
{ return _Tpvec(vec_sl(a.val, splfunc(imm))); }              \
inline _Tpvec operator >> (const _Tpvec& a, int imm)         \
{ return _Tpvec(shr(a.val, splfunc(imm))); }                 \
template<int imm> inline _Tpvec v_shl(const _Tpvec& a)       \
{ return _Tpvec(vec_sl(a.val, splfunc(imm))); }              \
template<int imm> inline _Tpvec v_shr(const _Tpvec& a)       \
{ return _Tpvec(shr(a.val, splfunc(imm))); }

OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint8x16, vec_sr, vec_uchar16_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint16x8, vec_sr, vec_ushort8_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint32x4, vec_sr, vec_uint4_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_uint64x2, vec_sr, vec_udword2_sp)
// algebraic right shift
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int8x16, vec_sra, vec_uchar16_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int16x8, vec_sra, vec_ushort8_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int32x4, vec_sra, vec_uint4_sp)
OPENCV_HAL_IMPL_VSX_SHIFT_OP(v_int64x2, vec_sra, vec_udword2_sp)

/** Bitwise logic **/
#define OPENCV_HAL_IMPL_VSX_LOGIC_OP(_Tpvec)    \
OPENCV_HAL_IMPL_VSX_BIN_OP(&, _Tpvec, vec_and)  \
OPENCV_HAL_IMPL_VSX_BIN_OP(|, _Tpvec, vec_or)   \
OPENCV_HAL_IMPL_VSX_BIN_OP(^, _Tpvec, vec_xor)  \
inline _Tpvec operator ~ (const _Tpvec& a)      \
{ return _Tpvec(vec_not(a.val)); }

OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint8x16)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int8x16)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint16x8)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int16x8)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint32x4)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int32x4)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_uint64x2)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_int64x2)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_float32x4)
OPENCV_HAL_IMPL_VSX_LOGIC_OP(v_float64x2)

/** Bitwise select **/
#define OPENCV_HAL_IMPL_VSX_SELECT(_Tpvec, cast)                             \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(vec_sel(b.val, a.val, cast(mask.val))); }

OPENCV_HAL_IMPL_VSX_SELECT(v_uint8x16, vec_bchar16_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_int8x16, vec_bchar16_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_uint16x8, vec_bshort8_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_int16x8, vec_bshort8_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_uint32x4, vec_bint4_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_int32x4, vec_bint4_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_float32x4, vec_bint4_c)
OPENCV_HAL_IMPL_VSX_SELECT(v_float64x2, vec_bdword2_c)

/** Comparison **/
#define OPENCV_HAL_IMPL_VSX_INT_CMP_OP(_Tpvec)                 \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b)   \
{ return _Tpvec(vec_cmpeq(a.val, b.val)); }                    \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b)   \
{ return _Tpvec(vec_cmpne(a.val, b.val)); }                    \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b)    \
{ return _Tpvec(vec_cmplt(a.val, b.val)); }                    \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b)    \
{ return _Tpvec(vec_cmpgt(a.val, b.val)); }                    \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b)   \
{ return _Tpvec(vec_cmple(a.val, b.val)); }                    \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b)   \
{ return _Tpvec(vec_cmpge(a.val, b.val)); }

OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint8x16)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int8x16)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint16x8)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int16x8)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint32x4)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int32x4)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_float32x4)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_float64x2)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_uint64x2)
OPENCV_HAL_IMPL_VSX_INT_CMP_OP(v_int64x2)

/** min/max **/
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_min, vec_min)
OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_max, vec_max)

/** Rotate **/
#define OPENCV_IMPL_VSX_ROTATE(_Tpvec, suffix, shf, cast)                       \
template<int imm>                                                               \
inline _Tpvec v_rotate_##suffix(const _Tpvec& a)                                \
{                                                                               \
    const int wd = imm * sizeof(typename _Tpvec::lane_type);                    \
    if (wd > 15)                                                                \
        return _Tpvec();                                                        \
    return _Tpvec((cast)shf(vec_uchar16_c(a.val), vec_uchar16_sp(wd << 3)));    \
}

#define OPENCV_IMPL_VSX_ROTATE_LR(_Tpvec, cast)     \
OPENCV_IMPL_VSX_ROTATE(_Tpvec, left, vec_slo, cast) \
OPENCV_IMPL_VSX_ROTATE(_Tpvec, right, vec_sro, cast)

OPENCV_IMPL_VSX_ROTATE_LR(v_uint8x16, vec_uchar16)
OPENCV_IMPL_VSX_ROTATE_LR(v_int8x16,  vec_char16)
OPENCV_IMPL_VSX_ROTATE_LR(v_uint16x8, vec_ushort8)
OPENCV_IMPL_VSX_ROTATE_LR(v_int16x8,  vec_short8)
OPENCV_IMPL_VSX_ROTATE_LR(v_uint32x4, vec_uint4)
OPENCV_IMPL_VSX_ROTATE_LR(v_int32x4,  vec_int4)
OPENCV_IMPL_VSX_ROTATE_LR(v_float32x4, vec_float4)
OPENCV_IMPL_VSX_ROTATE_LR(v_uint64x2, vec_udword2)
OPENCV_IMPL_VSX_ROTATE_LR(v_int64x2,  vec_dword2)
OPENCV_IMPL_VSX_ROTATE_LR(v_float64x2, vec_double2)

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b)
{
    enum { CV_SHIFT = 16 - imm * (sizeof(typename _Tpvec::lane_type)) };
    if (CV_SHIFT == 16)
        return a;
#ifdef __IBMCPP__
    return _Tpvec(vec_sld(b.val, a.val, CV_SHIFT & 15));
#else
    return _Tpvec(vec_sld(b.val, a.val, CV_SHIFT));
#endif
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b)
{
    enum { CV_SHIFT = imm * (sizeof(typename _Tpvec::lane_type)) };
    if (CV_SHIFT == 16)
        return b;
    return _Tpvec(vec_sld(a.val, b.val, CV_SHIFT));
}

#define OPENCV_IMPL_VSX_ROTATE_64_2RG(_Tpvec, suffix, rg1, rg2)   \
template<int imm>                                                 \
inline _Tpvec v_rotate_##suffix(const _Tpvec& a, const _Tpvec& b) \
{                                                                 \
    if (imm == 1)                                                 \
        return _Tpvec(vec_permi(rg1.val, rg2.val, 2));            \
    return imm ? b : a;                                           \
}

#define OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(_Tpvec)    \
OPENCV_IMPL_VSX_ROTATE_64_2RG(_Tpvec, left,  b, a)  \
OPENCV_IMPL_VSX_ROTATE_64_2RG(_Tpvec, right, a, b)

OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(v_float64x2)
OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(v_uint64x2)
OPENCV_IMPL_VSX_ROTATE_64_2RG_LR(v_int64x2)

/* Extract */
template<int s, typename _Tpvec>
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)
{ return v_rotate_right<s>(a, b); }

////////// Reduce and mask /////////

/** Reduce **/
inline short v_reduce_sum(const v_int16x8& a)
{
    const vec_int4 zero = vec_int4_z;
    return saturate_cast<short>(vec_extract(vec_sums(vec_sum4s(a.val, zero), zero), 3));
}
inline ushort v_reduce_sum(const v_uint16x8& a)
{
    const vec_int4 v4 = vec_int4_c(vec_unpackhu(vec_adds(a.val, vec_sld(a.val, a.val, 8))));
    return saturate_cast<ushort>(vec_extract(vec_sums(v4, vec_int4_z), 3));
}

#define OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(_Tpvec, _Tpvec2, scalartype, suffix, func) \
inline scalartype v_reduce_##suffix(const _Tpvec& a)                               \
{                                                                                  \
    const _Tpvec2 rs = func(a.val, vec_sld(a.val, a.val, 8));                      \
    return vec_extract(func(rs, vec_sld(rs, rs, 4)), 0);                           \
}
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_uint32x4, vec_uint4, uint, sum, vec_add)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_uint32x4, vec_uint4, uint, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_uint32x4, vec_uint4, uint, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_int32x4, vec_int4, int, sum, vec_add)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_int32x4, vec_int4, int, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_int32x4, vec_int4, int, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_float32x4, vec_float4, float, sum, vec_add)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_float32x4, vec_float4, float, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_4(v_float32x4, vec_float4, float, min, vec_min)

#define OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(_Tpvec, _Tpvec2, scalartype, suffix, func) \
inline scalartype v_reduce_##suffix(const _Tpvec& a)                               \
{                                                                                  \
    _Tpvec2 rs = func(a.val, vec_sld(a.val, a.val, 8));                            \
    rs = func(rs, vec_sld(rs, rs, 4));                                             \
    return vec_extract(func(rs, vec_sld(rs, rs, 2)), 0);                           \
}
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_uint16x8, vec_ushort8, ushort, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_uint16x8, vec_ushort8, ushort, min, vec_min)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_int16x8, vec_short8, short, max, vec_max)
OPENCV_HAL_IMPL_VSX_REDUCE_OP_8(v_int16x8, vec_short8, short, min, vec_min)

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    vec_float4 ac = vec_add(vec_mergel(a.val, c.val), vec_mergeh(a.val, c.val));
    ac = vec_add(ac, vec_sld(ac, ac, 8));

    vec_float4 bd = vec_add(vec_mergel(b.val, d.val), vec_mergeh(b.val, d.val));
    bd = vec_add(bd, vec_sld(bd, bd, 8));
    return v_float32x4(vec_mergeh(ac, bd));
}

/** Popcount **/
template<typename _Tpvec>
inline v_uint32x4 v_popcount(const _Tpvec& a)
{ return v_uint32x4(vec_popcntu(vec_uint4_c(a.val))); }

/** Mask **/
inline int v_signmask(const v_uint8x16& a)
{
    vec_uchar16 sv  = vec_sr(a.val, vec_uchar16_sp(7));
    static const vec_uchar16 slm = {0, 1, 2, 3, 4, 5, 6, 7, 0, 1, 2, 3, 4, 5, 6, 7};
    sv = vec_sl(sv, slm);
    vec_uint4 sv4 = vec_sum4s(sv, vec_uint4_z);
    static const vec_uint4 slm4 = {0, 0, 8, 8};
    sv4 = vec_sl(sv4, slm4);
    return vec_extract(vec_sums((vec_int4) sv4, vec_int4_z), 3);
}
inline int v_signmask(const v_int8x16& a)
{ return v_signmask(v_reinterpret_as_u8(a)); }

inline int v_signmask(const v_int16x8& a)
{
    static const vec_ushort8 slm = {0, 1, 2, 3, 4, 5, 6, 7};
    vec_short8 sv = vec_sr(a.val, vec_ushort8_sp(15));
    sv = vec_sl(sv, slm);
    vec_int4 svi = vec_int4_z;
    svi = vec_sums(vec_sum4s(sv, svi), svi);
    return vec_extract(svi, 3);
}
inline int v_signmask(const v_uint16x8& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }

inline int v_signmask(const v_int32x4& a)
{
    static const vec_uint4 slm = {0, 1, 2, 3};
    vec_int4 sv = vec_sr(a.val, vec_uint4_sp(31));
    sv = vec_sl(sv, slm);
    sv = vec_sums(sv, vec_int4_z);
    return vec_extract(sv, 3);
}
inline int v_signmask(const v_uint32x4& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_float32x4& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }

inline int v_signmask(const v_int64x2& a)
{
    VSX_UNUSED(const vec_dword2) sv = vec_sr(a.val, vec_udword2_sp(63));
    return (int)vec_extract(sv, 0) | (int)vec_extract(sv, 1) << 1;
}
inline int v_signmask(const v_uint64x2& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
inline int v_signmask(const v_float64x2& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }

template<typename _Tpvec>
inline bool v_check_all(const _Tpvec& a)
{ return vec_all_lt(a.val, _Tpvec().val); }
inline bool v_check_all(const v_uint8x16& a)
{ return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_all(const v_uint16x8& a)
{ return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_all(const v_uint32x4& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_float32x4& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_all(const v_float64x2& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }

template<typename _Tpvec>
inline bool v_check_any(const _Tpvec& a)
{ return vec_any_lt(a.val, _Tpvec().val); }
inline bool v_check_any(const v_uint8x16& a)
{ return v_check_any(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint16x8& a)
{ return v_check_any(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint32x4& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float32x4& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float64x2& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }

////////// Other math /////////

/** Some frequent operations **/
inline v_float32x4 v_sqrt(const v_float32x4& x)
{ return v_float32x4(vec_sqrt(x.val)); }
inline v_float64x2 v_sqrt(const v_float64x2& x)
{ return v_float64x2(vec_sqrt(x.val)); }

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{ return v_float32x4(vec_rsqrt(x.val)); }
inline v_float64x2 v_invsqrt(const v_float64x2& x)
{ return v_float64x2(vec_rsqrt(x.val)); }

#define OPENCV_HAL_IMPL_VSX_MULADD(_Tpvec)                                  \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b)                 \
{ return _Tpvec(vec_sqrt(vec_madd(a.val, a.val, vec_mul(b.val, b.val)))); } \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b)             \
{ return _Tpvec(vec_madd(a.val, a.val, vec_mul(b.val, b.val))); }           \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c)   \
{ return _Tpvec(vec_madd(a.val, b.val, c.val)); }

OPENCV_HAL_IMPL_VSX_MULADD(v_float32x4)
OPENCV_HAL_IMPL_VSX_MULADD(v_float64x2)

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{ return a * b + c; }

// TODO: exp, log, sin, cos

/** Absolute values **/
inline v_uint8x16 v_abs(const v_int8x16& x)
{ return v_uint8x16(vec_uchar16_c(vec_abs(x.val))); }

inline v_uint16x8 v_abs(const v_int16x8& x)
{ return v_uint16x8(vec_ushort8_c(vec_abs(x.val))); }

inline v_uint32x4 v_abs(const v_int32x4& x)
{ return v_uint32x4(vec_uint4_c(vec_abs(x.val))); }

inline v_float32x4 v_abs(const v_float32x4& x)
{ return v_float32x4(vec_abs(x.val)); }

inline v_float64x2 v_abs(const v_float64x2& x)
{ return v_float64x2(vec_abs(x.val)); }

OPENCV_HAL_IMPL_VSX_BIN_FUNC(v_absdiff, vec_absd)

#define OPENCV_HAL_IMPL_VSX_BIN_FUNC2(_Tpvec, _Tpvec2, cast, func, intrin)  \
inline _Tpvec2 func(const _Tpvec& a, const _Tpvec& b)                       \
{ return _Tpvec2(cast(intrin(a.val, b.val))); }

OPENCV_HAL_IMPL_VSX_BIN_FUNC2(v_int8x16, v_uint8x16, vec_uchar16_c, v_absdiff, vec_absd)
OPENCV_HAL_IMPL_VSX_BIN_FUNC2(v_int16x8, v_uint16x8, vec_ushort8_c, v_absdiff, vec_absd)
OPENCV_HAL_IMPL_VSX_BIN_FUNC2(v_int32x4, v_uint32x4, vec_uint4_c, v_absdiff, vec_absd)
OPENCV_HAL_IMPL_VSX_BIN_FUNC2(v_int64x2, v_uint64x2, vec_udword2_c, v_absdiff, vec_absd)

////////// Conversions /////////

/** Rounding **/
inline v_int32x4 v_round(const v_float32x4& a)
{ return v_int32x4(vec_cts(vec_round(a.val))); }

inline v_int32x4 v_round(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_round(a.val)), vec_int4_z)); }

inline v_int32x4 v_floor(const v_float32x4& a)
{ return v_int32x4(vec_cts(vec_floor(a.val))); }

inline v_int32x4 v_floor(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_floor(a.val)), vec_int4_z)); }

inline v_int32x4 v_ceil(const v_float32x4& a)
{ return v_int32x4(vec_cts(vec_ceil(a.val))); }

inline v_int32x4 v_ceil(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(vec_ceil(a.val)), vec_int4_z)); }

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(vec_cts(a.val)); }

inline v_int32x4 v_trunc(const v_float64x2& a)
{ return v_int32x4(vec_mergesqo(vec_ctso(a.val), vec_int4_z)); }

/** To float **/
inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{ return v_float32x4(vec_ctf(a.val)); }

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{ return v_float32x4(vec_mergesqo(vec_cvfo(a.val), vec_float4_z)); }

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{ return v_float64x2(vec_ctdo(vec_mergeh(a.val, a.val))); }

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{ return v_float64x2(vec_ctdo(vec_mergel(a.val, a.val))); }

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{ return v_float64x2(vec_cvfo(vec_mergeh(a.val, a.val))); }

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{ return v_float64x2(vec_cvfo(vec_mergel(a.val, a.val))); }

/** Reinterpret **/
/** its up there with load and store operations **/

////////// Matrix operations /////////

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{ return v_int32x4(vec_msum(a.val, b.val, vec_int4_z)); }

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_int32x4(vec_msum(a.val, b.val, c.val)); }

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    const vec_float4 v0 = vec_splat(v.val, 0);
    const vec_float4 v1 = vec_splat(v.val, 1);
    const vec_float4 v2 = vec_splat(v.val, 2);
    VSX_UNUSED(const vec_float4) v3 = vec_splat(v.val, 3);
    return v_float32x4(vec_madd(v0, m0.val, vec_madd(v1, m1.val, vec_madd(v2, m2.val, vec_mul(v3, m3.val)))));
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    const vec_float4 v0 = vec_splat(v.val, 0);
    const vec_float4 v1 = vec_splat(v.val, 1);
    const vec_float4 v2 = vec_splat(v.val, 2);
    return v_float32x4(vec_madd(v0, m0.val, vec_madd(v1, m1.val, vec_madd(v2, m2.val, a.val))));
}

#define OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(_Tpvec, _Tpvec2)                        \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1,                   \
                           const _Tpvec& a2, const _Tpvec& a3,                   \
                           _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3)       \
{                                                                                \
    _Tpvec2 a02 = vec_mergeh(a0.val, a2.val);                                    \
    _Tpvec2 a13 = vec_mergeh(a1.val, a3.val);                                    \
    b0.val = vec_mergeh(a02, a13);                                               \
    b1.val = vec_mergel(a02, a13);                                               \
    a02 = vec_mergel(a0.val, a2.val);                                            \
    a13 = vec_mergel(a1.val, a3.val);                                            \
    b2.val  = vec_mergeh(a02, a13);                                              \
    b3.val  = vec_mergel(a02, a13);                                              \
}
OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(v_uint32x4, vec_uint4)
OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(v_int32x4, vec_int4)
OPENCV_HAL_IMPL_VSX_TRANSPOSE4x4(v_float32x4, vec_float4)

//! @name Check SIMD support
//! @{
//! @brief Check CPU capability of SIMD operation
static inline bool hasSIMD128()
{
    return (CV_CPU_HAS_SUPPORT_VSX) ? true : false;
}

//! @}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif // OPENCV_HAL_VSX_HPP
