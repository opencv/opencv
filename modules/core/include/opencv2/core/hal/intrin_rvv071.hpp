// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// Copyright (C) 2015, PingTouGe Semiconductor Co., Ltd., all rights reserved.

#ifndef OPENCV_HAL_INTRIN_RISCVV_HPP
#define OPENCV_HAL_INTRIN_RISCVV_HPP

#include <float.h>
#include <algorithm>
#include "opencv2/core/utility.hpp"

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD128 1
#define CV_SIMD128_64F 0
//////////// Types ////////////
struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() {}
    explicit v_uint8x16(vuint8m1_t v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = (vuint8m1_t)vle_v_u8m1((unsigned char*)v, 16);
    }
    uchar get0() const
    {
        return vmv_x_s_u8m1_u8(val, 16);
    }

    vuint8m1_t val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(vint8m1_t v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
               schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = (vint8m1_t)vle_v_i8m1((schar*)v, 16);
    }
    schar get0() const
    {
        return vmv_x_s_i8m1_i8(val, 16);
    }

    vint8m1_t val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(vuint16m1_t v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = (vuint16m1_t)vle_v_u16m1((unsigned short*)v, 8);
    }
    ushort get0() const
    {
        return vmv_x_s_u16m1_u16(val, 8);
    }

    vuint16m1_t val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(vint16m1_t v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = (vint16m1_t)vle_v_i16m1((signed short*)v, 8);
    }
    short get0() const
    {
        return vmv_x_s_i16m1_i16(val, 8);
    }

    vint16m1_t val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(vuint32m1_t v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        val = (vuint32m1_t)vle_v_u32m1((unsigned int*)v, 4);
    }
    unsigned get0() const
    {
        return vmv_x_s_u32m1_u32(val, 4);
    }

    vuint32m1_t val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(vint32m1_t v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = (vint32m1_t)vle_v_i32m1((signed int*)v, 4);
    }
    int get0() const
    {
        return vmv_x_s_i32m1_i32(val, 4);
    }
    vint32m1_t val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(vfloat32m1_t v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = (vfloat32m1_t)vle_v_f32m1((float*)v, 4);
    }
    float get0() const
    {
        return vfmv_f_s_f32m1_f32(val, 4);
    }
    vfloat32m1_t val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(vuint64m1_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = (vuint64m1_t)vle_v_u64m1((unsigned long*)v, 2);
    }
    uint64 get0() const
    {
        return vmv_x_s_u64m1_u64(val, 2);
    }
    vuint64m1_t val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(vint64m1_t v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = (vint64m1_t)vle_v_i64m1((long*)v, 2);
    }
    int64 get0() const
    {
        return vmv_x_s_i64m1_i64(val, 2);
    }
    vint64m1_t val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(vfloat64m1_t v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = (vfloat64m1_t)vle_v_f64m1((double*)v, 2);
    }
    double get0() const
    {
        return vfmv_f_s_f64m1_f64(val, 2);
    }
    vfloat64m1_t val;
};

#define OPENCV_HAL_IMPL_RISCVV_INIT(_Tpv, _Tp, suffix) \
inline _Tp##m1_t vreinterpretq_##suffix##_##suffix(_Tp##m1_t v) { return v; } \
inline v_uint8x16 v_reinterpret_as_u8(const v_##_Tpv& v) { return v_uint8x16((vuint8m1_t)(v.val)); } \
inline v_int8x16 v_reinterpret_as_s8(const v_##_Tpv& v) { return v_int8x16((vint8m1_t)(v.val)); } \
inline v_uint16x8 v_reinterpret_as_u16(const v_##_Tpv& v) { return v_uint16x8((vuint16m1_t)(v.val)); } \
inline v_int16x8 v_reinterpret_as_s16(const v_##_Tpv& v) { return v_int16x8((vint16m1_t)(v.val)); } \
inline v_uint32x4 v_reinterpret_as_u32(const v_##_Tpv& v) { return v_uint32x4((vuint32m1_t)(v.val)); } \
inline v_int32x4 v_reinterpret_as_s32(const v_##_Tpv& v) { return v_int32x4((vint32m1_t)(v.val)); } \
inline v_uint64x2 v_reinterpret_as_u64(const v_##_Tpv& v) { return v_uint64x2((vuint64m1_t)(v.val)); } \
inline v_int64x2 v_reinterpret_as_s64(const v_##_Tpv& v) { return v_int64x2((vint64m1_t)(v.val)); } \
inline v_float32x4 v_reinterpret_as_f32(const v_##_Tpv& v) { return v_float32x4((vfloat32m1_t)(v.val)); }\
inline v_float64x2 v_reinterpret_as_f64(const v_##_Tpv& v) { return v_float64x2((vfloat64m1_t)(v.val)); }


OPENCV_HAL_IMPL_RISCVV_INIT(uint8x16, vuint8, u8)
OPENCV_HAL_IMPL_RISCVV_INIT(int8x16, vint8, s8)
OPENCV_HAL_IMPL_RISCVV_INIT(uint16x8, vuint16, u16)
OPENCV_HAL_IMPL_RISCVV_INIT(int16x8, vint16, s16)
OPENCV_HAL_IMPL_RISCVV_INIT(uint32x4, vuint32, u32)
OPENCV_HAL_IMPL_RISCVV_INIT(int32x4, vint32, s32)
OPENCV_HAL_IMPL_RISCVV_INIT(uint64x2, vuint64, u64)
OPENCV_HAL_IMPL_RISCVV_INIT(int64x2, vint64, s64)
OPENCV_HAL_IMPL_RISCVV_INIT(float64x2, vfloat64, f64)
OPENCV_HAL_IMPL_RISCVV_INIT(float32x4, vfloat32, f32)
#define OPENCV_HAL_IMPL_RISCVV_INIT_SET(__Tp, _Tp, suffix, len, num) \
inline v_##_Tp##x##num v_setzero_##suffix() { return v_##_Tp##x##num((v##_Tp##m1_t){0}); }     \
inline v_##_Tp##x##num v_setall_##suffix(__Tp v) { return v_##_Tp##x##num(vmv_v_x_##len##m1(v, num)); }

OPENCV_HAL_IMPL_RISCVV_INIT_SET(uchar, uint8, u8, u8, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(char, int8, s8, i8, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(ushort, uint16, u16, u16, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(short, int16, s16, i16, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(unsigned int, uint32, u32, u32, 4)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(int, int32, s32, i32, 4)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(unsigned long, uint64, u64, u64, 2)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(long, int64, s64, i64, 2)
inline v_float32x4 v_setzero_f32() { return v_float32x4((vfloat32m1_t){0}); }
inline v_float32x4 v_setall_f32(float v) { return v_float32x4(vfmv_v_f_f32m1(v, 4)); }

inline v_float64x2 v_setzero_f64() { return v_float64x2(vfmv_v_f_f64m1(0, 2)); }
inline v_float64x2 v_setall_f64(double v) { return v_float64x2(vfmv_v_f_f64m1(v, 2)); }


#define OPENCV_HAL_IMPL_RISCVV_BIN_OP(bin_op, _Tpvec, intrin) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a.val = intrin(a.val, b.val); \
    return a; \
}

#define OPENCV_HAL_IMPL_RISCVV_BIN_OPN(bin_op, _Tpvec, intrin, num) \
inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val, num)); \
} \
inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
{ \
    a.val = intrin(a.val, b.val, num); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint8x16, vsaddu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint8x16, vssubu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int8x16, vsadd_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int8x16, vssub_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint16x8, vsaddu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint16x8, vssubu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int16x8, vsadd_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int16x8, vssub_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int32x4, vadd_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int32x4, vsub_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_int32x4, vmul_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint32x4, vadd_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint32x4, vsub_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_uint32x4, vmul_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int64x2, vadd_vv_i64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int64x2, vsub_vv_i64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint64x2, vadd_vv_u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint64x2, vsub_vv_u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float32x4, vfadd_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float32x4, vfsub_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float32x4, vfmul_vv_f32m1, 4)
inline v_float32x4 operator / (const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4(vfdiv_vv_f32m1(a.val, b.val, 4));
}
inline v_float32x4& operator /= (v_float32x4& a, const v_float32x4& b)
{
    a.val = vfdiv_vv_f32m1(a.val, b.val, 4);
    return a;
}

OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float64x2, vfadd_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float64x2, vfsub_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float64x2, vfmul_vv_f64m1, 2)
inline v_float64x2 operator / (const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2(vfdiv_vv_f64m1(a.val, b.val, 2));
}
inline v_float64x2& operator /= (v_float64x2& a, const v_float64x2& b)
{
    a.val = vfdiv_vv_f64m1(a.val, b.val, 2);
    return a;
}
// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_RISCVV_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

#define OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(_Tpvec, func, intrin, num) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val, num)); \
}
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_min, vminu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_max, vmaxu_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_min, vmin_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_max, vmax_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_min, vminu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_max, vmaxu_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_min, vmin_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_max, vmax_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x4, v_min, vminu_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x4, v_max, vmaxu_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x4, v_min, vmin_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x4, v_max, vmax_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x4, v_min, vfmin_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x4, v_max, vfmax_vv_f32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x2, v_min, vfmin_vv_f64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x2, v_max, vfmax_vv_f64m1, 2)

inline v_float32x4 v_sqrt(const v_float32x4& x)
{
    return v_float32x4(vfsqrt_v_f32m1(x.val, 4));
}

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{
    return v_float32x4(vfrdiv_vf_f32m1(vfsqrt_v_f32m1(x.val, 4), 1, 4));
}

inline v_float32x4 v_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 x(vfmacc_vv_f32m1(vfmul_vv_f32m1(a.val, a.val, 4), b.val, b.val, 4));
    return v_sqrt(x);
}

inline v_float32x4 v_sqr_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4(vfmacc_vv_f32m1(vfmul_vv_f32m1(a.val, a.val, 4), b.val, b.val, 4));
}

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_float32x4(vfmacc_vv_f32m1(c.val, a.val, b.val, 4));
}

inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_int32x4(vmacc_vv_i32m1(c.val, a.val, b.val, 4));
}

inline v_float32x4 v_muladd(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_fma(a, b, c);
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_fma(a, b, c);
}

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0.val, v.val[0], 4);//vmuli_f32(m0.val, v.val, 0);
    res = vfmacc_vf_f32m1(res, v.val[1], m1.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m1(res, v.val[2], m2.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m1(res, v.val[3], m3.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x4(res);
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    vfloat32m1_t res = vfmul_vf_f32m1(m0.val, v.val[0], 4);//vmuli_f32(m0.val, v.val, 0);
    res = vfmacc_vf_f32m1(res, v.val[1], m1.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmacc_vf_f32m1(res, v.val[2], m2.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfadd_vv_f32m1(res, a.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x4(res);
}

inline v_float64x2 v_sqrt(const v_float64x2& x)
{
    return v_float64x2(vfsqrt_v_f64m1(x.val, 2));
}

inline v_float64x2 v_invsqrt(const v_float64x2& x)
{
    return v_float64x2(vfrdiv_vf_f64m1(vfsqrt_v_f64m1(x.val, 2), 1, 2));
}

inline v_float64x2 v_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    v_float64x2 x(vfmacc_vv_f64m1(vfmul_vv_f64m1(a.val, a.val, 2), b.val, b.val, 2));
    return v_sqrt(x);
}

inline v_float64x2 v_sqr_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2(vfmacc_vv_f64m1(vfmul_vv_f64m1(a.val, a.val, 2), b.val, b.val, 2));
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_float64x2(vfmacc_vv_f64m1(c.val, a.val, b.val, 2));
}

inline v_float64x2 v_muladd(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_fma(a, b, c);
}

#define OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(_Tpvec, suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(&, _Tpvec, vand_vv_##suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(|, _Tpvec, vor_vv_##suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(^, _Tpvec, vxor_vv_##suffix, num) \
    inline _Tpvec operator ~ (const _Tpvec & a) \
    { \
        return _Tpvec(vnot_v_##suffix(a.val, num)); \
    }

OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint8x16, u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint16x8, u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint32x4, u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint64x2, u64m1, 2)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int8x16,  i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int16x8,  i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int32x4,  i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int64x2,  i64m1, 2)

#define OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(bin_op, intrin) \
inline v_float32x4 operator bin_op (const v_float32x4& a, const v_float32x4& b) \
{ \
    return v_float32x4(vfloat32m1_t(intrin(vint32m1_t(a.val), vint32m1_t(b.val), 4))); \
} \
inline v_float32x4& operator bin_op##= (v_float32x4& a, const v_float32x4& b) \
{ \
    a.val = vfloat32m1_t(intrin(vint32m1_t(a.val), vint32m1_t(b.val), 4)); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(&, vand_vv_i32m1)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(|, vor_vv_i32m1)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(^, vxor_vv_i32m1)

inline v_float32x4 operator ~ (const v_float32x4& a)
{
    return v_float32x4((vfloat32m1_t)(vnot_v_i32m1((vint32m1_t)(a.val), 4)));
}

#define OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(bin_op, intrin) \
inline v_float64x2 operator bin_op (const v_float64x2& a, const v_float64x2& b) \
{ \
    return v_float64x2(vfloat64m1_t(intrin(vint64m1_t(a.val), vint64m1_t(b.val), 2))); \
} \
inline v_float64x2& operator bin_op##= (v_float64x2& a, const v_float64x2& b) \
{ \
    a.val = vfloat64m1_t(intrin(vint64m1_t(a.val), vint64m1_t(b.val), 2)); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(&, vand_vv_i64m1)
OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(|, vor_vv_i64m1)
OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(^, vxor_vv_i64m1)

inline v_float64x2 operator ~ (const v_float64x2& a)
{
    return v_float64x2((vfloat64m1_t)(vnot_v_i64m1((vint64m1_t)(a.val), 2)));
}
inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(vmulh_vv_i16m1(a.val, b.val, 8));
}
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(vmulhu_vv_u16m1(a.val, b.val, 8));
}

//#define OPENCV_HAL_IMPL_RISCVV_ABS(_Tpuvec, _Tpsvec, usuffix, ssuffix) \
//inline _Tpuvec v_abs(const _Tpsvec& a) {    \
//    E##xm1_t mask=vmflt_vf_e32xm1_f32m1(x.val, 0.0, 4);

//OPENCV_HAL_IMPL_RISCVV_ABS(v_uint8x16, v_int8x16, u8, s8)
//OPENCV_HAL_IMPL_RISCVV_ABS(v_uint16x8, v_int16x8, u16, s16)
//OPENCV_HAL_IMPL_RISCVV_ABS(v_uint32x4, v_int32x4, u32, s32)

inline v_uint32x4 v_abs(v_int32x4 x)
{
    vbool32_t mask=vmslt_vx_i32m1_b32(x.val, 0, 4);
    return v_uint32x4((vuint32m1_t)vrsub_vx_i32m1_m(mask, x.val, x.val, 0, 4));
}

inline v_uint16x8 v_abs(v_int16x8 x)
{
    vbool16_t mask=vmslt_vx_i16m1_b16(x.val, 0, 8);
    return v_uint16x8((vuint16m1_t)vrsub_vx_i16m1_m(mask, x.val, x.val, 0, 8));
}

inline v_uint8x16 v_abs(v_int8x16 x)
{
    vbool8_t mask=vmslt_vx_i8m1_b8(x.val, 0, 16);
    return v_uint8x16((vuint8m1_t)vrsub_vx_i8m1_m(mask, x.val, x.val, 0, 16));
}

inline v_float32x4 v_abs(v_float32x4 x)
{
    return (v_float32x4)vfsgnjx_vv_f32m1(x.val, x.val, 4);
}

inline v_float64x2 v_abs(v_float64x2 x)
{
    return (v_float64x2)vfsgnjx_vv_f64m1(x.val, x.val, 2);
}

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{
    vfloat32m1_t ret = vfsub_vv_f32m1(a.val, b.val, 4);
    return (v_float32x4)vfsgnjx_vv_f32m1(ret, ret, 4);
}

inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{
    vfloat64m1_t ret = vfsub_vv_f64m1(a.val, b.val, 2);
    return (v_float64x2)vfsgnjx_vv_f64m1(ret, ret, 2);
}

#define OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(bit, num) \
inline v_uint##bit##x##num v_absdiff(v_uint##bit##x##num a, v_uint##bit##x##num b){    \
    vuint##bit##m1_t vmax = vmaxu_vv_u##bit##m1(a.val, b.val, num);    \
    vuint##bit##m1_t vmin = vminu_vv_u##bit##m1(a.val, b.val, num);    \
    return v_uint##bit##x##num(vsub_vv_u##bit##m1(vmax, vmin, num));\
}

OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(8, 16)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(16, 8)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(32, 4)

/** Saturating absolute difference **/
inline v_int8x16 v_absdiffs(v_int8x16 a, v_int8x16 b){
    vint8m1_t vmax = vmax_vv_i8m1(a.val, b.val, 16);
    vint8m1_t vmin = vmin_vv_i8m1(a.val, b.val, 16);
    return v_int8x16(vssub_vv_i8m1(vmax, vmin, 16));
}
inline v_int16x8 v_absdiffs(v_int16x8 a, v_int16x8 b){
    vint16m1_t vmax = vmax_vv_i16m1(a.val, b.val, 8);
    vint16m1_t vmin = vmin_vv_i16m1(a.val, b.val, 8);
    return v_int16x8(vssub_vv_i16m1(vmax, vmin, 8));
}

#define OPENCV_HAL_IMPL_RISCVV_ABSDIFF(_Tpvec, _Tpv, num) \
inline v_uint##_Tpvec v_absdiff(v_int##_Tpvec a, v_int##_Tpvec b){    \
     vint##_Tpv##_t max = vmax_vv_i##_Tpv(a.val, b.val, num);\
     vint##_Tpv##_t min = vmin_vv_i##_Tpv(a.val, b.val, num);\
    return v_uint##_Tpvec((vuint##_Tpv##_t)vsub_vv_i##_Tpv(max, min, num));    \
}

OPENCV_HAL_IMPL_RISCVV_ABSDIFF(8x16, 8m1, 16)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF(16x8, 16m1, 8)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF(32x4, 32m1, 4)

//  Multiply and expand
inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    vint16m2_t res = vundefined_i16m2();
    res = vwmul_vv_i16m2(a.val, b.val, 16);
    c.val = vget_i16m2_i16m1(res, 0);
    d.val = vget_i16m2_i16m1(res, 1);
}

inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    vuint16m2_t res = vundefined_u16m2();
    res = vwmulu_vv_u16m2(a.val, b.val, 16);
    c.val = vget_u16m2_u16m1(res, 0);
    d.val = vget_u16m2_u16m1(res, 1);
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    vint32m2_t res = vundefined_i32m2();
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    c.val = vget_i32m2_i32m1(res, 0);
    d.val = vget_i32m2_i32m1(res, 1);
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    vuint32m2_t res = vundefined_u32m2();
    res = vwmulu_vv_u32m2(a.val, b.val, 8);
    c.val = vget_u32m2_u32m1(res, 0);
    d.val = vget_u32m2_u32m1(res, 1);
}

inline void v_mul_expand(const v_int32x4& a, const v_int32x4& b,
                         v_int64x2& c, v_int64x2& d)
{
    vint64m2_t res = vundefined_i64m2();
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    c.val = vget_i64m2_i64m1(res, 0);
    d.val = vget_i64m2_i64m1(res, 1);
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    vuint64m2_t res = vundefined_u64m2();
    res = vwmulu_vv_u64m2(a.val, b.val, 4);
    c.val = vget_u64m2_u64m1(res, 0);
    d.val = vget_u64m2_u64m1(res, 1);
}

OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_add_wrap, vadd_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_add_wrap, vadd_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_add_wrap, vadd_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_add_wrap, vadd_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_sub_wrap, vsub_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_sub_wrap, vsub_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_sub_wrap, vsub_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_sub_wrap, vsub_vv_i16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_mul_wrap, vmul_vv_u8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_mul_wrap, vmul_vv_i8m1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_mul_wrap, vmul_vv_u16m1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_mul_wrap, vmul_vv_i16m1, 8)
//////// Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t res = vundefined_i32m2();
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    res = vrgather_vv_i32m2(res, (vuint32m2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(res, 0), vget_i32m2_i32m1(res, 1), 4));
}
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    vint32m2_t res = vundefined_i32m2();
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    res = vrgather_vv_i32m2(res, (vuint32m2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(res, 0),vget_i32m2_i32m1(res, 1), 4), c.val, 4));
}

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t res = vundefined_i64m2();
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    res = vrgather_vv_i64m2(res, (vuint64m2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(res, 0), vget_i64m2_i64m1(res, 1), 2));
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    vint64m2_t res = vundefined_i64m2();
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    res = vrgather_vv_i64m2(res, (vuint64m2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(res, 0), vget_i64m2_i64m1(res, 1), 2), c.val, 2));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    vuint16m2_t v1 = vundefined_u16m2();
    vuint32m2_t v2 = vundefined_u32m2();
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v1 = vrgather_vv_u16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4));
}

inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b,
                                   const v_uint32x4& c)
{
    vuint16m2_t v1 = vundefined_u16m2();
    vuint32m2_t v2 = vundefined_u32m2();
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v1 = vrgather_vv_u16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4), c.val, 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    vint16m2_t v1 = vundefined_i16m2();
    vint32m2_t v2 = vundefined_i32m2();
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v1 = vrgather_vv_i16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b,
                                   const v_int32x4& c)
{
    vint16m2_t v1 = vundefined_i16m2();
    vint32m2_t v2 = vundefined_i32m2();
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v1 = vrgather_vv_i16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4), c.val, 4));
}

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint32m2_t v1 = vundefined_u32m2();
    vuint64m2_t v2 = vundefined_u64m2();
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v1 = vrgather_vv_u32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2));
}

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b,
                                   const v_uint64x2& c)
{
    vuint32m2_t v1 = vundefined_u32m2();
    vuint64m2_t v2 = vundefined_u64m2();
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v1 = vrgather_vv_u32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2), c.val, 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1 = vundefined_i32m2();
    vint64m2_t v2 = vundefined_i64m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v1 = vrgather_vv_i32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b,
                                   const v_int64x2& c)
{
    vint32m2_t v1 = vundefined_i32m2();
    vint64m2_t v2 = vundefined_i64m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v1 = vrgather_vv_i32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2), c.val, 2));
}

//////// Fast Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1 = vundefined_i32m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4));
}

inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    vint32m2_t v1 = vundefined_i32m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4), c.val, 4));
}

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t v1 = vundefined_i64m2();
    v1 = vwmul_vv_i64m2(a.val, b.val, 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 2));
}
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    vint64m2_t v1 = vundefined_i64m2();
    v1 = vwmul_vv_i64m2(a.val, b.val, 8);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 4), c.val, 4));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{
    vuint16m2_t v1 = vundefined_u16m2();
    vuint32m2_t v2 = vundefined_u32m2();
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4));
}

inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{
    vuint16m2_t v1 = vundefined_u16m2();
    vuint32m2_t v2 = vundefined_u32m2();
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4), c.val, 4));
}

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
    vint16m2_t v1 = vundefined_i16m2();
    vint32m2_t v2 = vundefined_i32m2();
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4));
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{
    vint16m2_t v1 = vundefined_i16m2();
    vint32m2_t v2 = vundefined_i32m2();
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4), c.val, 4));
}

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint32m2_t v1 = vundefined_u32m2();
    vuint64m2_t v2 = vundefined_u64m2();
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2));
}
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{
    vuint32m2_t v1 = vundefined_u32m2();
    vuint64m2_t v2 = vundefined_u64m2();
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2), c.val, 2));
}

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1 = vundefined_i32m2();
    vint64m2_t v2 = vundefined_i64m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2));
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{
    vint32m2_t v1 = vundefined_i32m2();
    vint64m2_t v2 = vundefined_i64m2();
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2), c.val, 2));
}


#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(_Tpvec, _Tpvec2, len, scalartype, func, intrin, num) \
inline scalartype v_reduce_##func(const v_##_Tpvec##x##num& a) \
{\
    v##_Tpvec2##m1_t val = vmv_v_x_##len##m1(0, num); \
    val = intrin(val, a.val, val, num);    \
    return vmv_x_s_##len##m1_##len(val, num);    \
}


#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(_Tpvec, _Tpvec2, scalartype, func, funcu, num) \
inline scalartype v_reduce_##func(const v_##_Tpvec##x##num& a) \
{\
    v##_Tpvec##m1_t val = (v##_Tpvec##m1_t)vmv_v_x_i8m1(0, num); \
    val = v##funcu##_vs_##_Tpvec2##m1_##_Tpvec2##m1(val, a.val, a.val, num);    \
    return val[0];    \
}
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int8, int16, i16, int, sum, vwredsum_vs_i8m1_i16m1, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int16, int32, i32, int, sum, vwredsum_vs_i16m1_i32m1, 8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int32, int64, i64, int, sum, vwredsum_vs_i32m1_i64m1, 4)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint8, uint16, u16, unsigned, sum, vwredsumu_vs_u8m1_u16m1, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint16, uint32, u32, unsigned, sum, vwredsumu_vs_u16m1_u32m1, 8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint32, uint64, u64, unsigned, sum, vwredsumu_vs_u32m1_u64m1, 4)
inline float v_reduce_sum(const v_float32x4& a) \
{\
    vfloat32m1_t val = vfmv_v_f_f32m1(0.0, 4); \
    val = vfredsum_vs_f32m1_f32m1(val, a.val, val, 4);    \
    return vfmv_f_s_f32m1_f32(val, 4);    \
}
inline double v_reduce_sum(const v_float64x2& a) \
{\
    vfloat64m1_t val = vfmv_v_f_f64m1(0.0, 2); \
    val = vfredsum_vs_f64m1_f64m1(val, a.val, val, 2);    \
    return vfmv_f_s_f64m1_f64(val, 2);    \
}
inline uint64 v_reduce_sum(const v_uint64x2& a)
{ return vext_x_v_u64m1_u64((vuint64m1_t)a.val, 0, 2)+vext_x_v_u64m1_u64((vuint64m1_t)a.val, 1, 2); }

inline int64 v_reduce_sum(const v_int64x2& a)
{ return vext_x_v_i64m1_i64((vint64m1_t)a.val, 0, 2)+vext_x_v_i64m1_i64((vint64m1_t)a.val, 1, 2); }

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(func)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int8,  i8, int, func, red##func, 16)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int16, i16, int, func, red##func, 8)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int32, i32, int, func, red##func, 4)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int64, i64, int, func, red##func, 2)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint8,  u8, unsigned, func, red##func##u, 16)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint16, u16, unsigned, func, red##func##u, 8)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint32, u32, unsigned, func, red##func##u, 4)    \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(float32, f32, float, func, fred##func, 4)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(max)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(min)

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    vfloat32m1_t a0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t b0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t c0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t d0 = vfmv_v_f_f32m1(0.0, 4);
    a0 = vfredsum_vs_f32m1_f32m1(a0, a.val, a0, 4);
    b0 = vfredsum_vs_f32m1_f32m1(b0, b.val, b0, 4);
    c0 = vfredsum_vs_f32m1_f32m1(c0, c.val, c0, 4);
    d0 = vfredsum_vs_f32m1_f32m1(d0, d.val, d0, 4);
    return v_float32x4(a0[0], b0[0], c0[0], d0[0]);
}

inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    vfloat32m1_t a0 = vfmv_v_f_f32m1(0.0, 4);
    vfloat32m1_t x = vfsub_vv_f32m1(a.val, b.val, 4);
    vbool32_t mask=vmflt_vf_f32m1_b32(x, 0, 4);
    vfloat32m1_t val = vfrsub_vf_f32m1_m(mask, x, x, 0, 4);
    a0 = vfredsum_vs_f32m1_f32m1(a0, val, a0, 4);
    return a0[0];
}

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(_Tpvec, _Tpvec2) \
inline unsigned v_reduce_sad(const _Tpvec& a, const _Tpvec&b){    \
    _Tpvec2 x = v_absdiff(a, b);    \
    return v_reduce_sum(x);    \
}

OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int8x16, v_uint8x16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint8x16, v_uint8x16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int16x8, v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint16x8, v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_int32x4, v_uint32x4)
OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(v_uint32x4, v_uint32x4)

#define OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(_Tpvec, _Tp, _T, num, uv) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmseq_vv_##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmsne_vv_##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmslt##uv##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmslt##uv##_Tp##_b##_T(b.val, a.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmsle##uv##_Tp##_b##_T(a.val, b.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ \
    vbool##_T##_t mask = vmsle##uv##_Tp##_b##_T(b.val, a.val, num);    \
    return _Tpvec(vmerge_vxm_##_Tp(mask, vmv_v_x_##_Tp(0, num), -1, num));    \
} \

OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int8x16, i8m1,  8, 16, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int16x8, i16m1, 16, 8, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int32x4, i32m1, 32, 4, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int64x2, i64m1, 64, 2, _vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint8x16, u8m1, 8, 16, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint16x8, u16m1, 16, 8, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint32x4, u32m1, 32, 4, u_vv_)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint64x2, u64m1, 64, 2, u_vv_)

//TODO: ==
inline v_float32x4 operator == (const v_float32x4& a, const v_float32x4& b)
{
    vbool32_t mask = vmfeq_vv_f32m1_b32(a.val, b.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}
inline v_float32x4 operator != (const v_float32x4& a, const v_float32x4& b)
{
    vbool32_t mask = vmfne_vv_f32m1_b32(a.val, b.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}
inline v_float32x4 operator < (const v_float32x4& a, const v_float32x4& b)
{
    vbool32_t mask = vmflt_vv_f32m1_b32(a.val, b.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}
inline v_float32x4 operator <= (const v_float32x4& a, const v_float32x4& b)
{
    vbool32_t mask = vmfle_vv_f32m1_b32(a.val, b.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}
inline v_float32x4 operator > (const v_float32x4& a, const v_float32x4& b)
{
    vbool32_t mask = vmfgt_vv_f32m1_b32(a.val, b.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}
inline v_float32x4 operator >= (const v_float32x4& a, const v_float32x4& b)
{
    vbool32_t mask = vmfge_vv_f32m1_b32(a.val, b.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}
inline v_float32x4 v_not_nan(const v_float32x4& a)
{
    vbool32_t mask = vmford_vv_f32m1_b32(a.val, a.val, 4);
    vint32m1_t res = vmerge_vxm_i32m1(mask, vmv_v_x_i32m1(0.0, 4), -1, 4);
    return v_float32x4((vfloat32m1_t)res);
}

//TODO: ==
inline v_float64x2 operator == (const v_float64x2& a, const v_float64x2& b)
{
    vbool64_t mask = vmfeq_vv_f64m1_b64(a.val, b.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
inline v_float64x2 operator != (const v_float64x2& a, const v_float64x2& b)
{
    vbool64_t mask = vmfne_vv_f64m1_b64(a.val, b.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
inline v_float64x2 operator < (const v_float64x2& a, const v_float64x2& b)
{
    vbool64_t mask = vmflt_vv_f64m1_b64(a.val, b.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
inline v_float64x2 operator <= (const v_float64x2& a, const v_float64x2& b)
{
    vbool64_t mask = vmfle_vv_f64m1_b64(a.val, b.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
inline v_float64x2 operator > (const v_float64x2& a, const v_float64x2& b)
{
    vbool64_t mask = vmfgt_vv_f64m1_b64(a.val, b.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
inline v_float64x2 operator >= (const v_float64x2& a, const v_float64x2& b)
{
    vbool64_t mask = vmfge_vv_f64m1_b64(a.val, b.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
inline v_float64x2 v_not_nan(const v_float64x2& a)
{
    vbool64_t mask = vmford_vv_f64m1_b64(a.val, a.val, 2);
    vint64m1_t res = vmerge_vxm_i64m1(mask, vmv_v_x_i64m1(0.0, 2), -1, 2);
    return v_float64x2((vfloat64m1_t)res);
}
#define OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(_Tp, _T) \
inline void v_transpose4x4(const v_##_Tp##32x4& a0, const v_##_Tp##32x4& a1, \
                         const v_##_Tp##32x4& a2, const v_##_Tp##32x4& a3, \
                         v_##_Tp##32x4& b0, v_##_Tp##32x4& b1, \
                         v_##_Tp##32x4& b2, v_##_Tp##32x4& b3) \
{ \
    v##_Tp##32m4_t val = vundefined_##_T##m4();    \
    val = vset_##_T##m4(val, 0, a0.val);    \
    val = vset_##_T##m4(val, 1, a1.val);    \
    val = vset_##_T##m4(val, 2, a2.val);    \
    val = vset_##_T##m4(val, 3, a3.val);   \
    val = vrgather_vv_##_T##m4(val, (vuint32m4_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);    \
    b0.val = vget_##_T##m4_##_T##m1(val, 0);   \
    b1.val = vget_##_T##m4_##_T##m1(val, 1);   \
    b2.val = vget_##_T##m4_##_T##m1(val, 2);   \
    b3.val = vget_##_T##m4_##_T##m1(val, 3);   \
}
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(uint, u32)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(int, i32)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(float, f32)


#define OPENCV_HAL_IMPL_RISCVV_SHIFT_LEFT(_Tpvec, suffix, _T, num) \
inline _Tpvec operator << (const _Tpvec& a, int n) \
{ return _Tpvec((vsll_vx_##_T##m1(a.val, n, num))); } \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return _Tpvec((vsll_vx_##_T##m1(a.val, n, num))); }

#define OPENCV_HAL_IMPL_RISCVV_SHIFT_RIGHT(_Tpvec, suffix, _T, num, intric) \
inline _Tpvec operator >> (const _Tpvec& a, int n) \
{ return _Tpvec((v##intric##_vx_##_T##m1(a.val, n, num))); } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return _Tpvec((v##intric##_vx_##_T##m1(a.val, n, num))); }\
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ return _Tpvec((v##intric##_vx_##_T##m1(vadd_vx_##_T##m1(a.val, 1<<(n-1), num), n, num))); }

// trade efficiency for convenience
#define OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(suffix, _T, num, intrin) \
OPENCV_HAL_IMPL_RISCVV_SHIFT_LEFT(v_##suffix##x##num, suffix, _T, num) \
OPENCV_HAL_IMPL_RISCVV_SHIFT_RIGHT(v_##suffix##x##num, suffix, _T, num, intrin)

OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint8, u8, 16, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint16, u16, 8, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint32, u32, 4, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint64, u64, 2, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int8, i8, 16, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int16, i16, 8, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int32, i32, 4, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int64, i64, 2, sra)

#if 0
#define VUP4(n) {0, 1, 2, 3}
#define VUP8(n) {0, 1, 2, 3, 4, 5, 6, 7}
#define VUP16(n) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
#define VUP2(n) {0, 1}
#endif
#define OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(_Tpvec, suffix, _T, num, num2, vmv, len) \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{    \
    suffix##m1_t tmp = vmv##_##_T##m1(0, num);\
        tmp = vslideup_vx_##_T##m1_m(vmset_m_##len(num), tmp, a.val, n, num);\
        return _Tpvec(tmp);\
} \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{     \
        return _Tpvec(vslidedown_vx_##_T##m1(a.val, n, num));\
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    suffix##m2_t tmp = vundefined_##_T##m2();    \
    tmp = vset_##_T##m2(tmp, 0, a.val);          \
    tmp = vset_##_T##m2(tmp, 1, b.val);          \
        tmp = vslidedown_vx_##_T##m2(tmp, n, num2);\
        return _Tpvec(vget_##_T##m2_##_T##m1(tmp, 0));\
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    suffix##m2_t tmp = vundefined_##_T##m2();    \
    tmp = vset_##_T##m2(tmp, 0, b.val);    \
    tmp = vset_##_T##m2(tmp, 1, a.val);    \
        tmp = vslideup_vx_##_T##m2(tmp, n, num2);\
        return _Tpvec(vget_##_T##m2_##_T##m1(tmp, 1));\
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ \
    CV_UNUSED(b); return a; \
}

OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint8x16, vuint8, u8, 16, 32, vmv_v_x, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int8x16, vint8, i8, 16, 32, vmv_v_x, b8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint16x8, vuint16, u16, 8, 16, vmv_v_x, b16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int16x8, vint16, i16, 8, 16, vmv_v_x, b16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint32x4, vuint32, u32, 4, 8, vmv_v_x, b32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int32x4, vint32, i32, 4, 8, vmv_v_x, b32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint64x2, vuint64, u64, 2, 4, vmv_v_x, b64)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int64x2, vint64, i64, 2, 4, vmv_v_x, b64)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float32x4, vfloat32, f32, 4, 8, vfmv_v_f, b32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float64x2, vfloat64, f64, 2, 4, vfmv_v_f, b64)

#define OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(_Tpvec, _Tp, _Tp2, len, hnum, num) \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
  typedef uint64 CV_DECL_ALIGNED(1) unaligned_uint64; \
  vuint64m1_t tmp = {*(unaligned_uint64*)ptr0, *(unaligned_uint64*)ptr1};\
    return _Tpvec(_Tp2##_t(tmp)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ return _Tpvec(vle_v_##len(ptr, hnum)); }\
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(vle_v_##len(ptr, num)); } \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec((_Tp2##_t)vle_v_##len((const _Tp *)ptr, num)); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, hnum);}\
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
  _Tp2##_t a0 = vslidedown_vx_##len(a.val, hnum, num);    \
  vse_v_##len(ptr, a0, hnum);}\
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, num); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, num); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ vse_v_##len(ptr, a.val, num); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ vse_v_##len(ptr, a.val, num); }

OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint8x16, uchar, vuint8m1, u8m1, 8, 16)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int8x16,  schar, vint8m1, i8m1, 8, 16)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint16x8, ushort, vuint16m1, u16m1, 4, 8)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int16x8,  short,  vint16m1, i16m1, 4, 8)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint32x4, unsigned, vuint32m1, u32m1, 2, 4)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int32x4,  int,     vint32m1, i32m1, 2, 4)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint64x2, unsigned long, vuint64m1, u64m1, 1, 2)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int64x2,  long,     vint64m1, i64m1, 1, 2)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float32x4, float, vfloat32m1, f32m1, 2, 4)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float64x2, double, vfloat64m1, f64m1, 1, 2)


////////////// Lookup table access ////////////////////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
#if 1
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[ 0]],
        tab[idx[ 1]],
        tab[idx[ 2]],
        tab[idx[ 3]],
        tab[idx[ 4]],
        tab[idx[ 5]],
        tab[idx[ 6]],
        tab[idx[ 7]],
        tab[idx[ 8]],
        tab[idx[ 9]],
        tab[idx[10]],
        tab[idx[11]],
        tab[idx[12]],
        tab[idx[13]],
        tab[idx[14]],
        tab[idx[15]]
    };
    return v_int8x16(vle_v_i8m1(elems, 16));
#else
    int32xm4_t index32 = vlev_int32xm4(idx, 16);
    vint16m2_t index16 = vnsra_vx_i16m2_int32xm4(index32, 0, 16);
    vint8m1_t index = vnsra_vx_i8m1_i16m2(index16, 0, 16);
    return v_int8x16(vlxbv_i8m1(tab, index, 16));
#endif
}

inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx){
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[4]],
        tab[idx[4] + 1],
        tab[idx[5]],
        tab[idx[5] + 1],
        tab[idx[6]],
        tab[idx[6] + 1],
        tab[idx[7]],
        tab[idx[7] + 1]
    };
    return v_int8x16(vle_v_i8m1(elems, 16));
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
    schar CV_DECL_ALIGNED(32) elems[16] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[2] + 2],
        tab[idx[2] + 3],
        tab[idx[3]],
        tab[idx[3] + 1],
        tab[idx[3] + 2],
        tab[idx[3] + 3]
    };
    return v_int8x16(vle_v_i8m1(elems, 16));
}

inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]],
        tab[idx[4]],
        tab[idx[5]],
        tab[idx[6]],
        tab[idx[7]]
    };
    return v_int16x8(vle_v_i16m1(elems, 8));
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[2]],
        tab[idx[2] + 1],
        tab[idx[3]],
        tab[idx[3] + 1]
    };
    return v_int16x8(vle_v_i16m1(elems, 8));
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    short CV_DECL_ALIGNED(32) elems[8] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[0] + 2],
        tab[idx[0] + 3],
        tab[idx[1]],
        tab[idx[1] + 1],
        tab[idx[1] + 2],
        tab[idx[1] + 3]
    };
    return v_int16x8(vle_v_i16m1(elems, 8));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_int32x4(vle_v_i32m1(elems, 4));
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[0] + 1],
        tab[idx[1]],
        tab[idx[1] + 1]
    };
    return v_int32x4(vle_v_i32m1(elems, 4));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(vle_v_i32m1(tab+idx[0], 4));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    vint64m1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_int64x2(res);
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(vle_v_i64m1(tab+idx[0], 2));
}

inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx)
{
    vuint64m1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_uint64x2(res);
}
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx)
{
    return v_uint64x2(vle_v_u64m1(tab+idx[0], 2));
}

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[1]],
        tab[idx[2]],
        tab[idx[3]]
    };
    return v_float32x4(vle_v_f32m1(elems, 4));
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idx[0]],
        tab[idx[0]+1],
        tab[idx[1]],
        tab[idx[1]+1]
    };
    return v_float32x4(vle_v_f32m1(elems, 4));
}
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{
    return v_float32x4(vle_v_f32m1(tab + idx[0], 4));
}
inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    vfloat64m1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_float64x2(res);
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2(vle_v_f64m1(tab+idx[0], 2));
}

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idxvec.val[0]],
        tab[idxvec.val[1]],
        tab[idxvec.val[2]],
        tab[idxvec.val[3]]
    };
    return v_int32x4(vle_v_i32m1(elems, 4));
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    unsigned CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idxvec.val[0]],
        tab[idxvec.val[1]],
        tab[idxvec.val[2]],
        tab[idxvec.val[3]]
    };
    return v_uint32x4(vle_v_u32m1(elems, 4));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    float CV_DECL_ALIGNED(32) elems[4] =
    {
        tab[idxvec.val[0]],
        tab[idxvec.val[1]],
        tab[idxvec.val[2]],
        tab[idxvec.val[3]]
    };
    return v_float32x4(vle_v_f32m1(elems, 4));
}
inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    vfloat64m1_t res = {tab[idxvec.val[0]], tab[idxvec.val[1]]};
    return v_float64x2(res);
}
inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    vint32m1_t index_x = vmul_vx_i32m1(idxvec.val, 4, 4);
    vint32m1_t index_y = vadd_vx_i32m1(index_x, 4, 4);

    x.val = vlxe_v_f32m1(tab, index_x, 4);
    y.val = vlxe_v_f32m1(tab, index_y, 4);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    x = v_float64x2(tab[idx[0]], tab[idx[1]]);
    y = v_float64x2(tab[idx[0]+1], tab[idx[1]+1]);
}

#define OPENCV_HAL_IMPL_RISCVV_PACKS(_Tp, _Tp2, _T2, num2, _T1, num, intrin, shr, _Type) \
inline v_##_Tp##x##num v_pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    v##_Tp2##m2_t  tmp = vundefined_##_T2##m2();    \
    tmp = vset_##_T2##m2(tmp, 0, a.val);    \
    tmp = vset_##_T2##m2(tmp, 1, b.val);    \
    return v_##_Tp##x##num(shr##_##_T1##m1(tmp, 0, num)); \
}\
template<int n> inline \
v_##_Tp##x##num v_rshr_pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    v##_Tp2##m2_t  tmp = vundefined_##_T2##m2();    \
    tmp = vset_##_T2##m2(tmp, 0, a.val);    \
    tmp = vset_##_T2##m2(tmp, 1, b.val);    \
    return v_##_Tp##x##num(intrin##_##_T1##m1(tmp, n, num)); \
}\
inline void v_pack_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    v##_Tp2##m2_t tmp = vundefined_##_T2##m2();    \
    tmp = vset_##_T2##m2(tmp, 0, a.val);    \
    tmp = vset_##_T2##m2(tmp, 1, vmv_v_x_##_T2##m1(0, num2));    \
    asm("" ::: "memory");                                       \
    vse_v_##_T1##m1(ptr, shr##_##_T1##m1(tmp, 0, num), num2); \
}\
template<int n> inline \
void v_rshr_pack_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    v##_Tp2##m2_t tmp = vundefined_##_T2##m2();    \
    tmp = vset_##_T2##m2(tmp, 0, a.val);    \
    tmp = vset_##_T2##m2(tmp, 1, vmv_v_x_##_T2##m1(0, num2));    \
    vse_v_##_T1##m1(ptr, intrin##_##_T1##m1(tmp, n, num), num2); \
}
OPENCV_HAL_IMPL_RISCVV_PACKS(int8, int16, i16, 8, i8, 16, vnclip_vx, vnclip_vx, signed char)
OPENCV_HAL_IMPL_RISCVV_PACKS(int16, int32, i32, 4, i16, 8, vnclip_vx, vnclip_vx, signed short)
OPENCV_HAL_IMPL_RISCVV_PACKS(int32, int64, i64, 2, i32, 4, vnclip_vx, vnsra_vx, int)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint8, uint16, u16, 8, u8, 16, vnclipu_vx, vnclipu_vx, unsigned char)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint16, uint32, u32, 4, u16, 8, vnclipu_vx, vnclipu_vx, unsigned short)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint32, uint64, u64, 2, u32, 4, vnclipu_vx, vnsrl_vx, unsigned int)

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint16m2_t tmp = vundefined_u16m2();    \
    tmp = vset_u16m2(tmp, 0, a.val);    \
    tmp = vset_u16m2(tmp, 1, b.val);    \
    return v_uint8x16(vnsrl_vx_u8m1(tmp, 0, 16));
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    vuint32m4_t vabcd = vundefined_u32m4();    \
    vuint16m2_t v16 = vundefined_u16m2();    \
    vabcd = vset_u32m4(vabcd, 0, a.val);    \
    vabcd = vset_u32m4(vabcd, 1, b.val);    \
    vabcd = vset_u32m4(vabcd, 2, c.val);    \
    vabcd = vset_u32m4(vabcd, 3, d.val);    \
    v16 = vnsrl_vx_u16m2(vabcd, 0, 16);
    return v_uint8x16(vnsrl_vx_u8m1(v16, 0, 16));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    vuint64m8_t v64 = vundefined_u64m8();    \
    vuint32m4_t v32 = vundefined_u32m4();    \
    vuint16m2_t v16 = vundefined_u16m2();    \
    v64 = vset_u64m8(v64, 0, a.val);    \
    v64 = vset_u64m8(v64, 1, b.val);    \
    v64 = vset_u64m8(v64, 2, c.val);    \
    v64 = vset_u64m8(v64, 3, d.val);    \
    v64 = vset_u64m8(v64, 4, e.val);    \
    v64 = vset_u64m8(v64, 5, f.val);    \
    v64 = vset_u64m8(v64, 6, g.val);    \
    v64 = vset_u64m8(v64, 7, h.val);    \
    v32 = vnsrl_vx_u32m4(v64, 0, 16);
    v16 = vnsrl_vx_u16m2(v32, 0, 16);
    return v_uint8x16(vnsrl_vx_u8m1(v16, 0, 16));
}

//inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b) \
//{ \
//    int16xm2_u tmp;    \
//    tmp.m1[0] = (vint16m1_t)a.val;    \
//    tmp.m1[1] = (vint16m1_t)b.val;    \
//    e8xm1_t mask = (e8xm1_t)vmsge_vx_e16xm2_i16m2(tmp.v, 0, 16);\
//    return v_uint8x16(vnclipuvi_mask_u8m1_u16m2(vmv_v_x_u8m1(0, 16), (vuint16m2_t)tmp.v, 0, mask, 16));
//}

#define OPENCV_HAL_IMPL_RISCVV_PACK_U(tp1, num1, tp2, num2, _Tp) \
inline v_uint##tp1##x##num1 v_pack_u(const v_int##tp2##x##num2& a, const v_int##tp2##x##num2& b) \
{ \
    vint##tp2##m2_t tmp = vundefined_##i##tp2##m2();    \
    tmp = vset_##i##tp2##m2(tmp, 0, a.val);    \
    tmp = vset_##i##tp2##m2(tmp, 1, b.val);    \
    vint##tp2##m2_t val = vmax_vx_i##tp2##m2(tmp, 0, num1);\
    return v_uint##tp1##x##num1(vnclipu_vx_u##tp1##m1((vuint##tp2##m2_t)val, 0, num1));    \
} \
inline void v_pack_u_store(_Tp* ptr, const v_int##tp2##x##num2& a) \
{ \
    vint##tp2##m2_t tmp = vundefined_##i##tp2##m2();    \
    tmp = vset_##i##tp2##m2(tmp, 0, a.val);    \
    vint##tp2##m2_t val = vmax_vx_i##tp2##m2(tmp, 0, num1);\
    return vse_v_u##tp1##m1(ptr, vnclipu_vx_u##tp1##m1((vuint##tp2##m2_t)val, 0, num1), num2);    \
} \
template<int n> inline \
v_uint##tp1##x##num1 v_rshr_pack_u(const v_int##tp2##x##num2& a, const v_int##tp2##x##num2& b) \
{ \
    vint##tp2##m2_t tmp = vundefined_##i##tp2##m2();    \
    tmp = vset_##i##tp2##m2(tmp, 0, a.val);    \
    tmp = vset_##i##tp2##m2(tmp, 1, b.val);    \
    vint##tp2##m2_t val = vmax_vx_i##tp2##m2(tmp, 0, num1);\
    return v_uint##tp1##x##num1(vnclipu_vx_u##tp1##m1((vuint##tp2##m2_t)val, n, num1));    \
} \
template<int n> inline \
void v_rshr_pack_u_store(_Tp* ptr, const v_int##tp2##x##num2& a) \
{ \
    vint##tp2##m2_t tmp = vundefined_##i##tp2##m2();    \
    tmp = vset_##i##tp2##m2(tmp, 0, a.val);    \
    vint##tp2##m2_t val_ = vmax_vx_i##tp2##m2(tmp, 0, num1);\
    vuint##tp1##m1_t val = vnclipu_vx_u##tp1##m1((vuint##tp2##m2_t)val_, n, num1);    \
    return vse_v_u##tp1##m1(ptr, val, num2);\
}
OPENCV_HAL_IMPL_RISCVV_PACK_U(8, 16, 16, 8, unsigned char )
OPENCV_HAL_IMPL_RISCVV_PACK_U(16, 8, 32, 4, unsigned short)

#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif

// saturating multiply 8-bit, 16-bit
#define OPENCV_HAL_IMPL_RISCVV_MUL_SAT(_Tpvec, _Tpwvec)            \
    inline _Tpvec operator * (const _Tpvec& a, const _Tpvec& b)  \
    {                                                            \
        _Tpwvec c, d;                                            \
        v_mul_expand(a, b, c, d);                                \
        return v_pack(c, d);                                     \
    }                                                            \
    inline _Tpvec& operator *= (_Tpvec& a, const _Tpvec& b)      \
    { a = a * b; return a; }

OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_int16x8,  v_int32x4)
OPENCV_HAL_IMPL_RISCVV_MUL_SAT(v_uint16x8, v_uint32x4)

#ifdef __GNUC__
#pragma GCC diagnostic pop
#endif
static const signed char popCountTable[256] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8,
};

inline vuint8m1_t vcnt_u8(vuint8m1_t val){
    vuint8m1_t v0 = val & 1;
    return vlxe_v_u8m1((unsigned char*)popCountTable, val >> 1, 16)+v0;
}

inline v_uint8x16
v_popcount(const v_uint8x16& a)
{
    return v_uint8x16(vcnt_u8(a.val));
}

inline v_uint8x16
v_popcount(const v_int8x16& a)
{
    return v_uint8x16(vcnt_u8((vuint8m1_t)a.val));
}

inline v_uint16x8
v_popcount(const v_uint16x8& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0E0C0A0806040200, 0, 0x0F0D0B0907050301, 0};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 8);
    return v_uint16x8(vget_u16m2_u16m1(res, 0));
}

inline v_uint16x8
v_popcount(const v_int16x8& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0E0C0A0806040200, 0, 0x0F0D0B0907050301, 0};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 8);
    return v_uint16x8(vget_u16m2_u16m1(res, 0));
}

inline v_uint32x4
v_popcount(const v_uint32x4& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0xFFFFFFFF0C080400, 0xFFFFFFFF0D090501,
                     0xFFFFFFFF0E0A0602, 0xFFFFFFFF0F0B0703};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res_ = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 16);
    vuint32m2_t res  = vwaddu_vv_u32m2(vget_u16m2_u16m1(res_, 0), vget_u16m2_u16m1(res_, 1), 8);
    return v_uint32x4(vget_u32m2_u32m1(res, 0));
}

inline v_uint32x4
v_popcount(const v_int32x4& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0xFFFFFFFF0C080400, 0xFFFFFFFF0D090501,
                     0xFFFFFFFF0E0A0602, 0xFFFFFFFF0F0B0703};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint16m2_t res_ = vwaddu_vv_u16m2(vget_u8m2_u8m1(tmp, 0), vget_u8m2_u8m1(tmp, 1), 16);
    vuint32m2_t res  = vwaddu_vv_u32m2(vget_u16m2_u16m1(res_, 0), vget_u16m2_u16m1(res_, 1), 8);
    return v_uint32x4(vget_u32m2_u32m1(res, 0));
}

inline v_uint64x2
v_popcount(const v_uint64x2& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0706050403020100, 0x0000000000000000,
                     0x0F0E0D0C0B0A0908, 0x0000000000000000};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint8m1_t zero = vmv_v_x_u8m1(0, 16);
    vuint8m1_t res1 = zero;
    vuint8m1_t res2 = zero;
    res1 = vredsum_vs_u8m1_u8m1(res1, vget_u8m2_u8m1(tmp, 0), zero, 8);
    res2 = vredsum_vs_u8m1_u8m1(res2, vget_u8m2_u8m1(tmp, 1), zero, 8);

    return v_uint64x2((unsigned long)vmv_x_s_u8m1_u8(res1, 8), (unsigned long)vmv_x_s_u8m1_u8(res2, 8));
}

inline v_uint64x2
v_popcount(const v_int64x2& a)
{
    vuint8m2_t tmp = vundefined_u8m2();
    tmp = vset_u8m2(tmp, 0, vcnt_u8((vuint8m1_t)a.val));
    vuint64m2_t mask = (vuint64m2_t){0x0706050403020100, 0x0000000000000000,
                     0x0F0E0D0C0B0A0908, 0x0000000000000000};
    tmp = vrgather_vv_u8m2(tmp, (vuint8m2_t)mask, 32);    \
    vuint8m1_t zero = vmv_v_x_u8m1(0, 16);
    vuint8m1_t res1 = zero;
    vuint8m1_t res2 = zero;
    res1 = vredsum_vs_u8m1_u8m1(res1, vget_u8m2_u8m1(tmp, 0), zero, 8);
    res2 = vredsum_vs_u8m1_u8m1(res2, vget_u8m2_u8m1(tmp, 1), zero, 8);

    return v_uint64x2((unsigned long)vmv_x_s_u8m1_u8(res1, 8), (unsigned long)vmv_x_s_u8m1_u8(res2, 8));
}

#define SMASK 1, 2, 4, 8, 16, 32, 64, 128
inline int v_signmask(const v_uint8x16& a)
{
    vuint8m1_t t0  = vsrl_vx_u8m1(a.val, 7, 16);
    vuint8m1_t m1  = (vuint8m1_t){SMASK, SMASK};
    vuint16m2_t t1 = vwmulu_vv_u16m2(t0, m1, 16);
    vuint32m1_t res = vmv_v_x_u32m1(0, 4);
    vuint32m2_t t2 = vwmulu_vx_u32m2(vget_u16m2_u16m1(t1, 1), 256, 8);
    res = vredsum_vs_u32m2_u32m1(res, t2, res, 8);
    res = vwredsumu_vs_u16m1_u32m1(res, vget_u16m2_u16m1(t1, 0), res, 8);
    return vmv_x_s_u32m1_u32(res, 8);
}
inline int v_signmask(const v_int8x16& a)
{
    vuint8m1_t t0 = vsrl_vx_u8m1((vuint8m1_t)a.val, 7, 16);
    vuint8m1_t m1 = (vuint8m1_t){SMASK, SMASK};
    vint16m2_t t1 = (vint16m2_t)vwmulu_vv_u16m2(t0, m1, 16);
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m2_t t2 = vwmul_vx_i32m2(vget_i16m2_i16m1(t1, 1), 256, 8);
    res = vredsum_vs_i32m2_i32m1(res, t2, res, 8);
    res = vwredsum_vs_i16m1_i32m1(res, vget_i16m2_i16m1(t1, 0), res, 8);
    return vmv_x_s_i32m1_i32(res, 8);
}

inline int v_signmask(const v_int16x8& a)
{
    vint16m1_t t0 = (vint16m1_t)vsrl_vx_u16m1((vuint16m1_t)a.val, 15, 8);
    vint16m1_t m1 = (vint16m1_t){SMASK};
    vint16m1_t t1 = vmul_vv_i16m1(t0, m1, 8);
    vint16m1_t res = vmv_v_x_i16m1(0, 8);
    res = vredsum_vs_i16m1_i16m1(res, t1, res, 8);
    return vmv_x_s_i16m1_i16(res, 8);
}
inline int v_signmask(const v_uint16x8& a)
{
    vint16m1_t t0 = (vint16m1_t)vsrl_vx_u16m1((vuint16m1_t)a.val, 15, 8);
    vint16m1_t m1 = (vint16m1_t){SMASK};
    vint16m1_t t1 = vmul_vv_i16m1(t0, m1, 8);
    vint16m1_t res = vmv_v_x_i16m1(0, 8);
    res = vredsum_vs_i16m1_i16m1(res, t1, res, 8);
    return vmv_x_s_i16m1_i16(res, 8);
}
inline int v_signmask(const v_int32x4& a)
{
    vint32m1_t t0 = (vint32m1_t)vsrl_vx_u32m1((vuint32m1_t)a.val, 31, 4);
    vint32m1_t m1 = (vint32m1_t){1, 2, 4, 8};
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m1_t t1 = vmul_vv_i32m1(t0, m1, 4);
    res = vredsum_vs_i32m1_i32m1(res, t1, res, 4);
    return vmv_x_s_i32m1_i32(res, 4);
}
inline int v_signmask(const v_uint32x4& a)
{
    vint32m1_t t0 = (vint32m1_t)vsrl_vx_u32m1(a.val, 31, 4);
    vint32m1_t m1 = (vint32m1_t){1, 2, 4, 8};
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m1_t t1 = vmul_vv_i32m1(t0, m1, 4);
    res = vredsum_vs_i32m1_i32m1(res, t1, res, 4);
    return vmv_x_s_i32m1_i32(res, 4);
}
inline int v_signmask(const v_uint64x2& a)
{
    vuint64m1_t v0 = vsrl_vx_u64m1(a.val, 63, 2);
    int res = (int)vext_x_v_u64m1_u64(v0, 0, 2) + ((int)vext_x_v_u64m1_u64(v0, 1, 2) << 1);
    return res;
}
inline int v_signmask(const v_int64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float32x4& a)
{
    vint32m1_t t0 = (vint32m1_t)vsrl_vx_u32m1((vuint32m1_t)a.val, 31, 4);
    vint32m1_t m1 = (vint32m1_t){1, 2, 4, 8};
    vint32m1_t res = vmv_v_x_i32m1(0, 4);
    vint32m1_t t1 = vmul_vv_i32m1(t0, m1, 4);
    res = vredsum_vs_i32m1_i32m1(res, t1, res, 4);
    return vmv_x_s_i32m1_i32(res, 4);
}

inline int v_scan_forward(const v_int8x16& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_uint8x16& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_int16x8& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_uint16x8& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_int32x4& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_uint32x4& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_float32x4& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_int64x2& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }
inline int v_scan_forward(const v_uint64x2& a) {
int val = v_signmask(a);
if(val==0) return 0;
else return trailingZeros32(val); }

#define OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(_Tpvec, suffix, _T, shift, num) \
inline bool v_check_all(const v_##_Tpvec& a) \
{ \
    suffix##m1_t v0 = vsrl_vx_##_T(vnot_v_##_T(a.val, num), shift, num); \
    vuint32m1_t v1 = vuint32m1_t(v0); \
    return (v1[0] | v1[1] | v1[2] | v1[3]) == 0; \
} \
inline bool v_check_any(const v_##_Tpvec& a) \
{ \
    suffix##m1_t v0 = vsrl_vx_##_T(a.val, shift, num); \
    vuint32m1_t v1 = vuint32m1_t(v0); \
    return (v1[0] | v1[1] | v1[2] | v1[3]) != 0; \
}

OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint8x16, vuint8,  u8m1, 7, 16)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint16x8, vuint16, u16m1, 15, 8)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint32x4, vuint32, u32m1, 31, 4)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint64x2, vuint64, u64m1, 63, 2)

inline bool v_check_all(const v_int8x16& a)
{ return v_check_all(v_reinterpret_as_u8(a)); }
inline bool v_check_all(const v_int16x8& a)
{ return v_check_all(v_reinterpret_as_u16(a)); }
inline bool v_check_all(const v_int32x4& a)
{ return v_check_all(v_reinterpret_as_u32(a)); }
inline bool v_check_all(const v_float32x4& a)
{ return v_check_all(v_reinterpret_as_u32(a)); }
inline bool v_check_all(const v_int64x2& a)
{ return v_check_all(v_reinterpret_as_u64(a)); }
inline bool v_check_all(const v_float64x2& a)
{ return v_check_all(v_reinterpret_as_u64(a)); }

inline bool v_check_any(const v_int8x16& a)
{ return v_check_any(v_reinterpret_as_u8(a)); }
inline bool v_check_any(const v_int16x8& a)
{ return v_check_any(v_reinterpret_as_u16(a)); }
inline bool v_check_any(const v_int32x4& a)
{ return v_check_any(v_reinterpret_as_u32(a)); }
inline bool v_check_any(const v_float32x4& a)
{ return v_check_any(v_reinterpret_as_u32(a)); }
inline bool v_check_any(const v_int64x2& a)
{ return v_check_any(v_reinterpret_as_u64(a)); }
inline bool v_check_any(const v_float64x2& a)
{ return v_check_any(v_reinterpret_as_u64(a)); }

#define OPENCV_HAL_IMPL_RISCVV_SELECT(_Tpvec, suffix, _Tpvec2, num) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(vmerge_vvm_##suffix(_Tpvec2(mask.val), b.val, a.val, num)); \
}

OPENCV_HAL_IMPL_RISCVV_SELECT(v_int8x16,  i8m1, vbool8_t, 16)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_int16x8,  i16m1, vbool16_t, 8)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_int32x4,  i32m1, vbool32_t, 4)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint8x16, u8m1, vbool8_t, 16)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint16x8, u16m1, vbool16_t, 8)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint32x4, u32m1, vbool32_t, 4)
inline v_float32x4 v_select(const v_float32x4& mask, const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4((vfloat32m1_t)vmerge_vvm_u32m1((vbool32_t)mask.val, (vuint32m1_t)b.val, (vuint32m1_t)a.val, 4));
}
inline v_float64x2 v_select(const v_float64x2& mask, const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2((vfloat64m1_t)vmerge_vvm_u64m1((vbool64_t)mask.val, (vuint64m1_t)b.val, (vuint64m1_t)a.val, 2));
}

#define OPENCV_HAL_IMPL_RISCVV_EXPAND(add, _Tpvec, _Tpwvec, _Tp, _Tp1, num1, _Tp2, num2, _T1, _T2) \
inline void v_expand(const _Tpvec& a, v_##_Tpwvec& b0, v_##_Tpwvec& b1) \
{ \
    _T1##_t b = vw##add##_vv_##_Tp2##m2(a.val, vmv_v_x_##_Tp1(0, num1), num1);    \
    b0.val = vget_##_Tp2##m2_##_Tp2##m1(b, 0);  \
    b1.val = vget_##_Tp2##m2_##_Tp2##m1(b, 1);  \
} \
inline v_##_Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _T1##_t b = vw##add##_vv_##_Tp2##m2(a.val, vmv_v_x_##_Tp1(0, num2), num2);    \
    return v_##_Tpwvec(vget_##_Tp2##m2_##_Tp2##m1(b, 0)); \
} \
inline v_##_Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _T1##_t b = vw##add##_vv_##_Tp2##m2(a.val, vmv_v_x_##_Tp1(0, num1), num1);    \
    return v_##_Tpwvec(vget_##_Tp2##m2_##_Tp2##m1(b, 1)); \
} \
inline v_##_Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    _T2##_t val = vle##_v_##_Tp1(ptr, num2);    \
    _T1##_t b = vw##add##_vv_##_Tp2##m2(val, vmv_v_x_##_Tp1(0, num2), num2);    \
    return v_##_Tpwvec(vget_##_Tp2##m2_##_Tp2##m1(b, 0)); \
}

OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint8x16, uint16x8, uchar, u8m1, 16, u16, 8, vuint16m2, vuint8m1)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint16x8, uint32x4, ushort,  u16m1, 8, u32, 4, vuint32m2, vuint16m1)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint32x4, uint64x2, uint,  u32m1, 4, u64, 2, vuint64m2, vuint32m1)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int8x16, int16x8, schar,  i8m1, 16, i16, 8, vint16m2, vint8m1)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int16x8, int32x4, short,  i16m1, 8, i32, 4, vint32m2, vint16m1)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int32x4, int64x2, int,  i32m1, 4, i64, 2, vint64m2, vint32m1)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    vuint16m2_t b = vundefined_u16m2();
    vuint32m2_t c = vundefined_u32m2();
    vuint8m1_t val = vle_v_u8m1(ptr, 4);    \
    b = vwaddu_vv_u16m2(val, vmv_v_x_u8m1(0, 4), 4);    \
    c = vwaddu_vv_u32m2(vget_u16m2_u16m1(b, 0), vmv_v_x_u16m1(0, 4), 4);    \
    return v_uint32x4(vget_u32m2_u32m1(c, 0));
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    vint16m2_t b = vundefined_i16m2();
    vint32m2_t c = vundefined_i32m2();
    vint8m1_t val = vle_v_i8m1(ptr, 4);    \
    b = vwadd_vv_i16m2(val, vmv_v_x_i8m1(0, 4), 4);    \
    c = vwadd_vv_i32m2(vget_i16m2_i16m1(b, 0), vmv_v_x_i16m1(0, 4), 4);    \
    return v_int32x4(vget_i32m2_i32m1(c, 0));
}
#define VITL_16 (vuint32m2_t){0x11011000, 0x13031202, 0x15051404, 0x17071606, 0x19091808, 0x1B0B1A0A, 0x1D0D1C0C, 0x1F0F1E0E}
#define VITL_8 (vuint32m2_t){0x00080000, 0x00090001, 0x000A0002, 0x000B0003, 0x000C0004, 0x000D0005, 0x000E0006, 0x000F0007}
#define VITL_4 (vuint32m2_t){0x00000000, 0x00000004, 0x00000001, 0x00000005, 0x00000002, 0x00000006, 0x00000003, 0x00000007}
#define VITL_2 (vuint32m2_t){0, 0, 2, 0, 1, 0, 3, 0}

#define OPENCV_HAL_IMPL_RISCVV_UNPACKS(_Tpvec, _Tp, _T, _UTp, _UT, num, num2, len, numh) \
inline void v_zip(const v_##_Tpvec& a0, const v_##_Tpvec& a1, v_##_Tpvec& b0, v_##_Tpvec& b1) \
{ \
    v##_Tp##m2_t tmp = vundefined_##_T##m2();\
    tmp = vset_##_T##m2(tmp, 0, a0.val); \
    tmp = vset_##_T##m2(tmp, 1, a1.val); \
    vuint32m2_t mask = VITL_##num;    \
    tmp = (v##_Tp##m2_t)vrgather_vv_##_T##m2((v##_Tp##m2_t)tmp, (v##_UTp##m2_t)mask, num2);    \
    b0.val = vget_##_T##m2_##_T##m1(tmp, 0); \
    b1.val = vget_##_T##m2_##_T##m1(tmp, 1); \
} \
inline v_##_Tpvec v_combine_low(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    v##_Tp##m1_t b0 = vslideup_vx_##_T##m1_m(vmset_m_##len(num), a.val, b.val, numh, num);    \
    return v_##_Tpvec(b0);\
} \
inline v_##_Tpvec v_combine_high(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    v##_Tp##m1_t b0 = vslidedown_vx_##_T##m1(b.val, numh, num);    \
    v##_Tp##m1_t a0 = vslidedown_vx_##_T##m1(a.val, numh, num);    \
    v##_Tp##m1_t b1 = vslideup_vx_##_T##m1_m(vmset_m_##len(num), a0, b0, numh, num);    \
    return v_##_Tpvec(b1);\
} \
inline void v_recombine(const v_##_Tpvec& a, const v_##_Tpvec& b, v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    c.val = vslideup_vx_##_T##m1_m(vmset_m_##len(num), a.val, b.val, numh, num);    \
    v##_Tp##m1_t b0 = vslidedown_vx_##_T##m1(b.val, numh, num);    \
    v##_Tp##m1_t a0 = vslidedown_vx_##_T##m1(a.val, numh, num);    \
    d.val = vslideup_vx_##_T##m1_m(vmset_m_##len(num), a0, b0, numh, num);    \
}

OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint8x16, uint8, u8, uint8, u8, 16, 32, b8, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int8x16, int8, i8, uint8, u8, 16, 32, b8, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint16x8, uint16, u16, uint16, u16, 8, 16, b16, 4)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int16x8, int16, i16, uint16, u16, 8, 16, b16, 4)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint32x4, uint32, u32, uint32, u32, 4, 8, b32, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int32x4, int32, i32, uint32, u32, 4, 8, b32, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float32x4, float32, f32, uint32, u32, 4, 8, b32, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float64x2, float64, f64, uint64, u64, 2, 4, b64, 1)

inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
    vuint64m1_t mask = (vuint64m1_t){0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_uint8x16(vrgather_vv_u8m1(a.val, (vuint8m1_t)mask, 16));
}
inline v_int8x16 v_reverse(const v_int8x16 &a)
{
    vint64m1_t mask = (vint64m1_t){0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_int8x16(vrgather_vv_i8m1(a.val, (vuint8m1_t)mask, 16));
}

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
    vuint64m1_t mask = (vuint64m1_t){0x0004000500060007, 0x000000100020003};
    return v_uint16x8(vrgather_vv_u16m1(a.val, (vuint16m1_t)mask, 8));
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{
    vint64m1_t mask = (vint64m1_t){0x0004000500060007, 0x000000100020003};
    return v_int16x8(vrgather_vv_i16m1(a.val, (vuint16m1_t)mask, 8));
}
inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{
    return v_uint32x4(vrgather_vv_u32m1(a.val, (vuint32m1_t){3, 2, 1, 0}, 4));
}

inline v_int32x4 v_reverse(const v_int32x4 &a)
{
    return v_int32x4(vrgather_vv_i32m1(a.val, (vuint32m1_t){3, 2, 1, 0}, 4));
}

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{
    return v_uint64x2(a.val[1], a.val[0]);
}

inline v_int64x2 v_reverse(const v_int64x2 &a)
{
    return v_int64x2(a.val[1], a.val[0]);
}

inline v_float64x2 v_reverse(const v_float64x2 &a)
{
    return v_float64x2(a.val[1], a.val[0]);
}

#define OPENCV_HAL_IMPL_RISCVV_EXTRACT(_Tpvec, suffix, size) \
template <int n> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b) \
{ return v_rotate_right<n>(a, b);}
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint8x16, u8, 0)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int8x16, s8, 0)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint16x8, u16, 1)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int16x8, s16, 1)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint32x4, u32, 2)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int32x4, s32, 2)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_uint64x2, u64, 3)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_int64x2, s64, 3)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_float32x4, f32, 2)
OPENCV_HAL_IMPL_RISCVV_EXTRACT(v_float64x2, f64, 3)


#define OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(_Tpvec, _Tp, suffix) \
template<int i> inline _Tp v_extract_n(_Tpvec v) { return v.val[i]; }

OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int8x16, schar, s8)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int16x8, short, s16)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint32x4, uint, u32)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int32x4, int, s32)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_int64x2, int64, s64)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_float32x4, float, f32)
OPENCV_HAL_IMPL_RISCVV_EXTRACT_N(v_float64x2, double, f64)

#define OPENCV_HAL_IMPL_RISCVV_BROADCAST(_Tpvec, _Tp, num) \
template<int i> inline _Tpvec v_broadcast_element(_Tpvec v) { return _Tpvec(vrgather_vx_##_Tp##m1(v.val, i, num)); }

OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint8x16, u8, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int8x16, i8, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint16x8, u16, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int16x8, i16, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint32x4, u32, 4)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int32x4, i32, 4)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint64x2, u64, 2)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int64x2, i64, 2)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_float32x4, f32, 4)
inline v_int32x4 v_round(const v_float32x4& a)
{
    __builtin_riscv_fsrm(0);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}
inline v_int32x4 v_floor(const v_float32x4& a)
{
    __builtin_riscv_fsrm(2);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    __builtin_riscv_fsrm(3);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{
    __builtin_riscv_fsrm(1);
    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)a.val, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), a.val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_round(const v_float64x2& a)
{
    __builtin_riscv_fsrm(0);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    //_val = vset_f64m2(_val, 1, a.val);
    _val = vset_f64m2(_val, 1, vfmv_v_f_f64m1(0, 2));
    vint32m1_t val = vfncvt_x_f_v_i32m1(_val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}
inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    __builtin_riscv_fsrm(0);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    _val = vset_f64m2(_val, 1, b.val);
    vint32m1_t val = vfncvt_x_f_v_i32m1(_val, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}
inline v_int32x4 v_floor(const v_float64x2& a)
{
    __builtin_riscv_fsrm(2);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);

    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)aval, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), aval, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    __builtin_riscv_fsrm(3);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);

    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)aval, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), aval, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{
    __builtin_riscv_fsrm(1);
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);

    vint32m1_t nan = vand_vx_i32m1((vint32m1_t)aval, 0x7f800000, 4);
    vbool32_t mask = vmsne_vx_i32m1_b32(nan, 0x7f800000, 4);
    vint32m1_t val = vfcvt_x_f_v_i32m1_m(mask, vmv_v_x_i32m1(0, 4), aval, 4);
    __builtin_riscv_fsrm(0);
    return v_int32x4(val);
}

#define OPENCV_HAL_IMPL_RISCVV_LOAD_DEINTERLEAVED(intrin, _Tpvec, num, _Tp, _T)    \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b) \
{ \
    v##_Tpvec##m1x2_t ret = intrin##2e_v_##_T##m1x2(ptr, num);\
    a.val = vget_##_T##m1x2_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x2_##_T##m1(ret, 1);  \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, v_##_Tpvec##x##num& c) \
{ \
    v##_Tpvec##m1x3_t ret = intrin##3e_v_##_T##m1x3(ptr, num);\
    a.val = vget_##_T##m1x3_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x3_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x3_##_T##m1(ret, 2);  \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, \
                                v_##_Tpvec##x##num& c, v_##_Tpvec##x##num& d) \
{ \
    v##_Tpvec##m1x4_t ret = intrin##4e_v_##_T##m1x4(ptr, num);\
    a.val = vget_##_T##m1x4_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x4_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x4_##_T##m1(ret, 2);  \
    d.val = vget_##_T##m1x4_##_T##m1(ret, 3);  \
} \

#define OPENCV_HAL_IMPL_RISCVV_STORE_INTERLEAVED(intrin, _Tpvec, num, _Tp, _T)    \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x2_t ret = vundefined_##_T##m1x2();      \
    ret = vset_##_T##m1x2(ret, 0, a.val);  \
    ret = vset_##_T##m1x2(ret, 1, b.val);  \
    intrin##2e_v_##_T##m1x2(ptr, ret, num); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x3_t ret = vundefined_##_T##m1x3();       \
    ret = vset_##_T##m1x3(ret, 0, a.val);  \
    ret = vset_##_T##m1x3(ret, 1, b.val);  \
    ret = vset_##_T##m1x3(ret, 2, c.val);  \
    intrin##3e_v_##_T##m1x3(ptr, ret, num); \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, const v_##_Tpvec##x##num& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    v##_Tpvec##m1x4_t ret = vundefined_##_T##m1x4();             \
    ret = vset_##_T##m1x4(ret, 0, a.val);  \
    ret = vset_##_T##m1x4(ret, 1, b.val);  \
    ret = vset_##_T##m1x4(ret, 2, c.val);  \
    ret = vset_##_T##m1x4(ret, 3, d.val);  \
    intrin##4e_v_##_T##m1x4(ptr, ret, num); \
}

#define OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(_Tpvec, _Tp, num, ld, st, _T) \
OPENCV_HAL_IMPL_RISCVV_LOAD_DEINTERLEAVED(ld, _Tpvec, num, _Tp, _T)    \
OPENCV_HAL_IMPL_RISCVV_STORE_INTERLEAVED(st, _Tpvec, num, _Tp, _T)

//OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint8, uchar, )
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(int8, schar, 16, vlseg, vsseg, i8)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(int16, short, 8, vlseg, vsseg, i16)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(int32, int, 4, vlseg, vsseg, i32)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint8, unsigned char, 16, vlseg, vsseg, u8)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint16, unsigned short, 8, vlseg, vsseg, u16)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED(uint32, unsigned int, 4, vlseg, vsseg, u32)

#define OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(_Tpvec, _Tp, num, _T) \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b) \
{ \
    v##_Tpvec##m1x2_t ret = vlseg2e_v_##_T##m1x2(ptr, num); \
    a.val = vget_##_T##m1x2_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x2_##_T##m1(ret, 1);  \
} \
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, v_##_Tpvec##x##num& c) \
{ \
    v##_Tpvec##m1x3_t ret = vlseg3e_v_##_T##m1x3(ptr, num);    \
    a.val = vget_##_T##m1x3_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x3_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x3_##_T##m1(ret, 2);  \
}\
inline void v_load_deinterleave(const _Tp* ptr, v_##_Tpvec##x##num& a, v_##_Tpvec##x##num& b, \
                                v_##_Tpvec##x##num& c, v_##_Tpvec##x##num& d) \
{ \
    v##_Tpvec##m1x4_t ret = vlseg4e_v_##_T##m1x4(ptr, num);    \
    a.val = vget_##_T##m1x4_##_T##m1(ret, 0);  \
    b.val = vget_##_T##m1x4_##_T##m1(ret, 1);  \
    c.val = vget_##_T##m1x4_##_T##m1(ret, 2);  \
    d.val = vget_##_T##m1x4_##_T##m1(ret, 3);  \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x2_t ret = vundefined_##_T##m1x2();    \
    ret = vset_##_T##m1x2(ret, 0, a.val);  \
    ret = vset_##_T##m1x2(ret, 1, b.val);  \
    vsseg2e_v_##_T##m1x2(ptr, ret, num);    \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, hal::StoreMode /*mode*/=hal::STORE_UNALIGNED) \
{ \
    v##_Tpvec##m1x3_t ret = vundefined_##_T##m1x3();    \
    ret = vset_##_T##m1x3(ret, 0, a.val);  \
    ret = vset_##_T##m1x3(ret, 1, b.val);  \
    ret = vset_##_T##m1x3(ret, 2, c.val);  \
    vsseg3e_v_##_T##m1x3(ptr, ret, num);    \
} \
inline void v_store_interleave( _Tp* ptr, const v_##_Tpvec##x##num& a, const v_##_Tpvec##x##num& b, \
                                const v_##_Tpvec##x##num& c, const v_##_Tpvec##x##num& d, \
                                hal::StoreMode /*mode*/=hal::STORE_UNALIGNED ) \
{ \
    v##_Tpvec##m1x4_t ret = vundefined_##_T##m1x4();    \
    ret = vset_##_T##m1x4(ret, 0, a.val);  \
    ret = vset_##_T##m1x4(ret, 1, b.val);  \
    ret = vset_##_T##m1x4(ret, 2, c.val);  \
    ret = vset_##_T##m1x4(ret, 3, d.val);  \
    vsseg4e_v_##_T##m1x4(ptr, ret, num);    \
}
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(float32, float, 4, f32)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(float64, double, 2, f64)

OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(uint64, unsigned long, 2, u64)
OPENCV_HAL_IMPL_RISCVV_INTERLEAVED_(int64, long, 2, i64)

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(vfcvt_f_x_v_f32m1(a.val, 4));
}

#if CV_SIMD128_64F
inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 2);
    return v_float32x4(aval);
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{
    vfloat64m2_t _val = vundefined_f64m2();
    _val = vset_f64m2(_val, 0, a.val);
    _val = vset_f64m2(_val, 1, b.val);
    vfloat32m1_t aval = vfncvt_f_f_v_f32m1(_val, 4);
    return v_float32x4(aval);
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    vfloat32m1_t val = vfcvt_f_x_v_f32m1(a.val, 4);
    vfloat64m2_t _val = vfwcvt_f_f_v_f64m2(val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 0));
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
    vfloat32m1_t val = vfcvt_f_x_v_f32m1(a.val, 4);
    vfloat64m2_t _val = vfwcvt_f_f_v_f64m2(val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 1));
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    vfloat64m2_t _val  = vfwcvt_f_f_v_f64m2(a.val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 0));
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    vfloat64m2_t _val  = vfwcvt_f_f_v_f64m2(a.val, 4);
    return v_float64x2(vget_f64m2_f64m1(_val, 1));
}

inline v_float64x2 v_cvt_f64(const v_int64x2& a)
{
    return v_float64x2(vfcvt_f_x_v_f64m1(a.val, 2));
}

#endif
inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
    vuint64m1_t m0 = {0x0705060403010200, 0x0F0D0E0C0B090A08};
    return v_int8x16(vrgather_vv_i8m1(vec.val, (vuint8m1_t)m0, 16));
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec)
{
    return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec)));
}

inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
    vuint64m1_t m0 = {0x0703060205010400, 0x0F0B0E0A0D090C08};
    return v_int8x16(vrgather_vv_i8m1(vec.val, (vuint8m1_t)m0, 16));
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec)
{
    return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec)));
}

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
    vuint64m1_t m0 = {0x0706030205040100, 0x0F0E0B0A0D0C0908};
    return v_int16x8((vint16m1_t)vrgather_vv_u8m1((vuint8m1_t)vec.val, (vuint8m1_t)m0, 16));
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
    vuint64m1_t m0 = {0x0B0A030209080100, 0x0F0E07060D0C0504};
    return v_int16x8((vint16m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    vuint64m1_t m0 = {0x0B0A090803020100, 0x0F0E0D0C07060504};
    return v_int32x4((vint32m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
    vuint64m1_t m0 = {0x0908060504020100, 0xFFFFFFFF0E0D0C0A};
    return v_int8x16((vint8m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
    vuint64m1_t m0 = {0x0908050403020100, 0xFFFFFFFF0D0C0B0A};
    return v_int16x8((vint16m1_t)vrgather_vv_u8m1((vuint8m1_t)(vec.val), (vuint8m1_t)m0, 16));
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

#if CV_SIMD128_64F
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{ return v_cvt_f64(v_dotprod(a, b)); }
inline v_float64x2 v_dotprod_expand(const v_int32x4& a,   const v_int32x4& b,
                                    const v_float64x2& c)
{ return v_dotprod_expand(a, b) + c; }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t v1 = vwmul_vv_i64m2(a.val, b.val, 4);
    vfloat64m1_t res = vfcvt_f_x_v_f64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 2), 2);
    return v_float64x2(res);
}
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ v_float64x2 res = v_dotprod_expand_fast(a, b);
  return res + c; }
#endif
////// FP16 support ///////
inline v_float32x4 v_load_expand(const float16_t* ptr)
{
    vfloat16m1_t v = vle_v_f16m1((__fp16*)ptr, 4);
    vfloat32m2_t v32 = vfwcvt_f_f_v_f32m2(v, 4);
    return v_float32x4(vget_f32m2_f32m1(v32, 0));
}

inline void v_pack_store(float16_t* ptr, const v_float32x4& v)
{
    vfloat32m2_t v32 = vundefined_f32m2();
    v32 = vset_f32m2(v32, 0, v.val);
    vfloat16m1_t hv = vfncvt_f_f_v_f16m1(v32, 4);
    vse_v_f16m1((__fp16*)ptr, hv, 4);
}


inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // namespace cv
#endif
