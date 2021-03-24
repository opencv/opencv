/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Copyright (C) 2021, PingTouGe Semiconductor Co., Ltd all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

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
#define CV_SIMD128_64F 1
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
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int32x4, vsadd_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int32x4, vssub_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_int32x4, vmul_vv_i32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint32x4, vadd_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint32x4, vsub_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_uint32x4, vmul_vv_u32m1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int64x2, vsadd_vv_i64m1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int64x2, vssub_vv_i64m1, 2)
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
    vint16m2_t res;
    res = vwmul_vv_i16m2(a.val, b.val, 16);
    c.val = vget_i16m2_i16m1(res, 0);
    d.val = vget_i16m2_i16m1(res, 1);
}

inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    vuint16m2_t res;
    res = vwmulu_vv_u16m2(a.val, b.val, 16);
    c.val = vget_u16m2_u16m1(res, 0);
    d.val = vget_u16m2_u16m1(res, 1);
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    vint32m2_t res;
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    c.val = vget_i32m2_i32m1(res, 0);
    d.val = vget_i32m2_i32m1(res, 1);
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    vuint32m2_t res;
    res = vwmulu_vv_u32m2(a.val, b.val, 8);
    c.val = vget_u32m2_u32m1(res, 0);
    d.val = vget_u32m2_u32m1(res, 1);
}

inline void v_mul_expand(const v_int32x4& a, const v_int32x4& b,
                         v_int64x2& c, v_int64x2& d)
{
    vint64m2_t res;
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    c.val = vget_i64m2_i64m1(res, 0);
    d.val = vget_i64m2_i64m1(res, 1);
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    vuint64m2_t res;
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
    vint32m2_t res;
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    res = vrgather_vv_i32m2(res, (vuint32m2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(res, 0), vget_i32m2_i32m1(res, 1), 4));
}
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    vint32m2_t res;
    res = vwmul_vv_i32m2(a.val, b.val, 8);
    res = vrgather_vv_i32m2(res, (vuint32m2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(res, 0),vget_i32m2_i32m1(res, 1), 4), c.val, 4));
}

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t res;
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    res = vrgather_vv_i64m2(res, (vuint64m2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(res, 0), vget_i64m2_i64m1(res, 1), 2));
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    vint64m2_t res;
    res = vwmul_vv_i64m2(a.val, b.val, 4);
    res = vrgather_vv_i64m2(res, (vuint64m2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(res, 0), vget_i64m2_i64m1(res, 1), 2), c.val, 2));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    vuint16m2_t v1;
    vuint32m2_t v2;
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v1 = vrgather_vv_u16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4));
}

inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b,
                                   const v_uint32x4& c)
{
    vuint16m2_t v1;
    vuint32m2_t v2;
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v1 = vrgather_vv_u16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4), c.val, 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    vint16m2_t v1;
    vint32m2_t v2;
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v1 = vrgather_vv_i16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b,
                                   const v_int32x4& c)
{
    vint16m2_t v1;
    vint32m2_t v2;
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v1 = vrgather_vv_i16m2(v1, (vuint16m2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4), c.val, 4));
}

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint32m2_t v1;
    vuint64m2_t v2;
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v1 = vrgather_vv_u32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2));
}

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b,
                                   const v_uint64x2& c)
{
    vuint32m2_t v1;
    vuint64m2_t v2;
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v1 = vrgather_vv_u32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2), c.val, 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1;
    vint64m2_t v2;
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v1 = vrgather_vv_i32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b,
                                   const v_int64x2& c)
{
    vint32m2_t v1;
    vint64m2_t v2;
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v1 = vrgather_vv_i32m2(v1, (vuint32m2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2), c.val, 2));
}

//////// Fast Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1;
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4));
}

inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    vint32m2_t v1;
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4), c.val, 4));
}

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{
    vint64m2_t v1;
    v1 = vwmul_vv_i64m2(a.val, b.val, 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 2));
}
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    vint64m2_t v1;
    v1 = vwmul_vv_i64m2(a.val, b.val, 8);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v1, 0), vget_i64m2_i64m1(v1, 1), 4), c.val, 4));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{
    vuint16m2_t v1;
    vuint32m2_t v2;
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4));
}

inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{
    vuint16m2_t v1;
    vuint32m2_t v2;
    v1 = vwmulu_vv_u16m2(a.val, b.val, 16);
    v2 = vwaddu_vv_u32m2(vget_u16m2_u16m1(v1, 0), vget_u16m2_u16m1(v1, 1), 8);
    return v_uint32x4(vadd_vv_u32m1(vadd_vv_u32m1(vget_u32m2_u32m1(v2, 0), vget_u32m2_u32m1(v2, 1), 4), c.val, 4));
}

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
    vint16m2_t v1;
    vint32m2_t v2;
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4));
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{
    vint16m2_t v1;
    vint32m2_t v2;
    v1 = vwmul_vv_i16m2(a.val, b.val, 16);
    v2 = vwadd_vv_i32m2(vget_i16m2_i16m1(v1, 0), vget_i16m2_i16m1(v1, 1), 8);
    return v_int32x4(vadd_vv_i32m1(vadd_vv_i32m1(vget_i32m2_i32m1(v2, 0), vget_i32m2_i32m1(v2, 1), 4), c.val, 4));
}

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    vuint32m2_t v1;
    vuint64m2_t v2;
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2));
}
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{
    vuint32m2_t v1;
    vuint64m2_t v2;
    v1 = vwmulu_vv_u32m2(a.val, b.val, 8);
    v2 = vwaddu_vv_u64m2(vget_u32m2_u32m1(v1, 0), vget_u32m2_u32m1(v1, 1), 4);
    return v_uint64x2(vadd_vv_u64m1(vadd_vv_u64m1(vget_u64m2_u64m1(v2, 0), vget_u64m2_u64m1(v2, 1), 2), c.val, 2));
}

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    vint32m2_t v1;
    vint64m2_t v2;
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2));
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{
    vint32m2_t v1;
    vint64m2_t v2;
    v1 = vwmul_vv_i32m2(a.val, b.val, 8);
    v2 = vwadd_vv_i64m2(vget_i32m2_i32m1(v1, 0), vget_i32m2_i32m1(v1, 1), 4);
    return v_int64x2(vadd_vv_i64m1(vadd_vv_i64m1(vget_i64m2_i64m1(v2, 0), vget_i64m2_i64m1(v2, 1), 2), c.val, 2));
}


inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}
#endif
