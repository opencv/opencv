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

#include<float.h>
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
    explicit v_uint8x16(uint8xm1_t v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        uchar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = (uint8xm1_t)vlbuv_uint8xm1((unsigned char*)v, 16);
    }
    uchar get0() const
    {
        return val[0];
    }

    uint8xm1_t val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(int8xm1_t v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
               schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        schar v[] = {v0, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15};
        val = (int8xm1_t)vlbv_int8xm1((schar*)v, 16);
    }
    schar get0() const
    {
        return val[0];
    }

    int8xm1_t val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(uint16xm1_t v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        ushort v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = (uint16xm1_t)vlev_uint16xm1((unsigned short*)v, 8);
    }
    ushort get0() const
    {
        return val[0];
    }

    uint16xm1_t val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(int16xm1_t v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        short v[] = {v0, v1, v2, v3, v4, v5, v6, v7};
        val = (int16xm1_t)vlev_int16xm1((signed short*)v, 8);
    }
    short get0() const
    {
        return val[0];
    }

    int16xm1_t val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(uint32xm1_t v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        unsigned v[] = {v0, v1, v2, v3};
        val = (uint32xm1_t)vlev_uint32xm1((unsigned int*)v, 4);
    }
    unsigned get0() const
    {
        return val[0];
    }

    uint32xm1_t val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(int32xm1_t v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        int v[] = {v0, v1, v2, v3};
        val = (int32xm1_t)vlev_int32xm1((signed int*)v, 4);
    }
    int get0() const
    {
        return val[0];
    }
    int32xm1_t val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(float32xm1_t v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        float v[] = {v0, v1, v2, v3};
        val = (float32xm1_t)vlev_float32xm1((float*)v, 4);
    }
    float get0() const
    {
        return val[0];
    }
    float32xm1_t val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(uint64xm1_t v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        uint64 v[] = {v0, v1};
        val = (uint64xm1_t)vlev_uint64xm1((unsigned long*)v, 2);
    }
    uint64 get0() const
    {
        return val[0];
    }
    uint64xm1_t val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(int64xm1_t v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        int64 v[] = {v0, v1};
        val = (int64xm1_t)vlev_int64xm1((long*)v, 2);
    }
    int64 get0() const
    {
        return val[0];
    }
    int64xm1_t val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(float64xm1_t v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        double v[] = {v0, v1};
        val = (float64xm1_t)vlev_float64xm1((double*)v, 2);
    }
    double get0() const
    {
        return val[0];
    }
    float64xm1_t val;
};

#define OPENCV_HAL_IMPL_RISCVV_INIT(_Tpv, _Tp, suffix) \
inline _Tp##xm1_t vreinterpretq_##suffix##_##suffix(_Tp##xm1_t v) { return v; } \
inline v_uint8x16 v_reinterpret_as_u8(const v_##_Tpv& v) { return v_uint8x16((uint8xm1_t)(v.val)); } \
inline v_int8x16 v_reinterpret_as_s8(const v_##_Tpv& v) { return v_int8x16((int8xm1_t)(v.val)); } \
inline v_uint16x8 v_reinterpret_as_u16(const v_##_Tpv& v) { return v_uint16x8((uint16xm1_t)(v.val)); } \
inline v_int16x8 v_reinterpret_as_s16(const v_##_Tpv& v) { return v_int16x8((int16xm1_t)(v.val)); } \
inline v_uint32x4 v_reinterpret_as_u32(const v_##_Tpv& v) { return v_uint32x4((uint32xm1_t)(v.val)); } \
inline v_int32x4 v_reinterpret_as_s32(const v_##_Tpv& v) { return v_int32x4((int32xm1_t)(v.val)); } \
inline v_uint64x2 v_reinterpret_as_u64(const v_##_Tpv& v) { return v_uint64x2((uint64xm1_t)(v.val)); } \
inline v_int64x2 v_reinterpret_as_s64(const v_##_Tpv& v) { return v_int64x2((int64xm1_t)(v.val)); } \
inline v_float32x4 v_reinterpret_as_f32(const v_##_Tpv& v) { return v_float32x4((float32xm1_t)(v.val)); }\
inline v_float64x2 v_reinterpret_as_f64(const v_##_Tpv& v) { return v_float64x2((float64xm1_t)(v.val)); }


OPENCV_HAL_IMPL_RISCVV_INIT(uint8x16, uint8, u8)
OPENCV_HAL_IMPL_RISCVV_INIT(int8x16, int8, s8)
OPENCV_HAL_IMPL_RISCVV_INIT(uint16x8, uint16, u16)
OPENCV_HAL_IMPL_RISCVV_INIT(int16x8, int16, s16)
OPENCV_HAL_IMPL_RISCVV_INIT(uint32x4, uint32, u32)
OPENCV_HAL_IMPL_RISCVV_INIT(int32x4, int32, s32)
OPENCV_HAL_IMPL_RISCVV_INIT(uint64x2, uint64, u64)
OPENCV_HAL_IMPL_RISCVV_INIT(int64x2, int64, s64)
OPENCV_HAL_IMPL_RISCVV_INIT(float64x2, float64, f64)
OPENCV_HAL_IMPL_RISCVV_INIT(float32x4, float32, f32)
#define OPENCV_HAL_IMPL_RISCVV_INIT_SET(__Tp, _Tp, suffix, num) \
inline v_##_Tp##x##num v_setzero_##suffix() { return v_##_Tp##x##num(vmvvx_##_Tp##xm1(0, num)); }     \
inline v_##_Tp##x##num v_setall_##suffix(__Tp v) { return v_##_Tp##x##num(vmvvx_##_Tp##xm1(v, num)); }

OPENCV_HAL_IMPL_RISCVV_INIT_SET(uchar, uint8, u8, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(char, int8, s8, 16)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(ushort, uint16, u16, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(short, int16, s16, 8)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(unsigned int, uint32, u32, 4)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(int, int32, s32, 4)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(unsigned long, uint64, u64, 2)
OPENCV_HAL_IMPL_RISCVV_INIT_SET(long, int64, s64, 2)
inline v_float32x4 v_setzero_f32() { return v_float32x4(vfmvvf_float32xm1(0, 4)); }
inline v_float32x4 v_setall_f32(float v) { return v_float32x4(vfmvvf_float32xm1(v, 4)); }

inline v_float64x2 v_setzero_f64() { return v_float64x2(vfmvvf_float64xm1(0, 2)); }
inline v_float64x2 v_setall_f64(double v) { return v_float64x2(vfmvvf_float64xm1(v, 2)); }

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

OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint8x16, vsadduvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint8x16, vssubuvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int8x16, vsaddvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int8x16, vssubvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint16x8, vsadduvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint16x8, vssubuvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int16x8, vsaddvv_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int16x8, vssubvv_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int32x4, vsaddvv_int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int32x4, vssubvv_int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_int32x4, vmulvv_int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint32x4, vaddvv_uint32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint32x4, vsubvv_uint32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_uint32x4, vmulvv_uint32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_int64x2, vsaddvv_int64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_int64x2, vssubvv_int64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_uint64x2, vaddvv_uint64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_uint64x2, vsubvv_uint64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float32x4, vfaddvv_float32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float32x4, vfsubvv_float32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float32x4, vfmulvv_float32xm1, 4)
inline v_float32x4 operator / (const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4(vfdivvv_float32xm1(a.val, b.val, 4));
}
inline v_float32x4& operator /= (v_float32x4& a, const v_float32x4& b)
{
    a.val = vfdivvv_float32xm1(a.val, b.val, 4);
    return a;
}

OPENCV_HAL_IMPL_RISCVV_BIN_OPN(+, v_float64x2, vfaddvv_float64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(-, v_float64x2, vfsubvv_float64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BIN_OPN(*, v_float64x2, vfmulvv_float64xm1, 2)
inline v_float64x2 operator / (const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2(vfdivvv_float64xm1(a.val, b.val, 2));
}
inline v_float64x2& operator /= (v_float64x2& a, const v_float64x2& b)
{
    a.val = vfdivvv_float64xm1(a.val, b.val, 2);
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
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_min, vminuvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_max, vmaxuvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_min, vminvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_max, vmaxvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_min, vminuvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_max, vmaxuvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_min, vminvv_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_max, vmaxvv_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x4, v_min, vminuvv_uint32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint32x4, v_max, vmaxuvv_uint32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x4, v_min, vminvv_int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int32x4, v_max, vmaxvv_int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x4, v_min, vfminvv_float32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float32x4, v_max, vfmaxvv_float32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x2, v_min, vfminvv_float64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_float64x2, v_max, vfmaxvv_float64xm1, 2)

inline v_float32x4 v_sqrt(const v_float32x4& x)
{
    return v_float32x4(vfsqrtv_float32xm1(x.val, 4));
}

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{
    return v_float32x4(vfrdivvf_float32xm1(vfsqrtv_float32xm1(x.val, 4), 1, 4));
}

inline v_float32x4 v_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    v_float32x4 x(vfmaccvv_float32xm1(vfmulvv_float32xm1(a.val, a.val, 4), b.val, b.val, 4));
    return v_sqrt(x);
}

inline v_float32x4 v_sqr_magnitude(const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4(vfmaccvv_float32xm1(vfmulvv_float32xm1(a.val, a.val, 4), b.val, b.val, 4));
}

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
    return v_float32x4(vfmaccvv_float32xm1(c.val, a.val, b.val, 4));
}

inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_int32x4(vmaccvv_int32xm1(a.val, b.val, c.val, 4));
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
    float32xm1_t res = vfmulvf_float32xm1(m0.val, v.val[0], 4);//vmuli_f32(m0.val, v.val, 0);
    res = vfmaccvf_float32xm1(res, v.val[1], m1.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmaccvf_float32xm1(res, v.val[2], m2.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmaccvf_float32xm1(res, v.val[3], m3.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x4(res);
}

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    float32xm1_t res = vfmulvf_float32xm1(m0.val, v.val[0], 4);//vmuli_f32(m0.val, v.val, 0);
    res = vfmaccvf_float32xm1(res, v.val[1], m1.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfmaccvf_float32xm1(res, v.val[2], m2.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    res = vfaddvv_float32xm1(res, a.val, 4);//vmulai_f32(res, m1.val, v.val, 1);
    return v_float32x4(res);
}

inline v_float64x2 v_sqrt(const v_float64x2& x)
{
    return v_float64x2(vfsqrtv_float64xm1(x.val, 2));
}

inline v_float64x2 v_invsqrt(const v_float64x2& x)
{
    return v_float64x2(vfrdivvf_float64xm1(vfsqrtv_float64xm1(x.val, 2), 1, 2));
}

inline v_float64x2 v_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    v_float64x2 x(vfmaccvv_float64xm1(vfmulvv_float64xm1(a.val, a.val, 2), b.val, b.val, 2));
    return v_sqrt(x);
}

inline v_float64x2 v_sqr_magnitude(const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2(vfmaccvv_float64xm1(vfmulvv_float64xm1(a.val, a.val, 2), b.val, b.val, 2));
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_float64x2(vfmaccvv_float64xm1(c.val, a.val, b.val, 2));
}

inline v_float64x2 v_muladd(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
    return v_fma(a, b, c);
}

#define OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(_Tpvec, suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(&, _Tpvec, vandvv_##suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(|, _Tpvec, vorvv_##suffix, num) \
    OPENCV_HAL_IMPL_RISCVV_BIN_OPN(^, _Tpvec, vxorvv_##suffix, num) \
    inline _Tpvec operator ~ (const _Tpvec & a) \
    { \
        return _Tpvec(vnotv_##suffix(a.val, num)); \
    }

OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint8x16, uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint16x8, uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint32x4, uint32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_uint64x2, uint64xm1, 2)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int8x16, int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int16x8, int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int32x4, int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_LOGIC_OPN(v_int64x2, int64xm1, 2)

#define OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(bin_op, intrin) \
inline v_float32x4 operator bin_op (const v_float32x4& a, const v_float32x4& b) \
{ \
    return v_float32x4(float32xm1_t(intrin(int32xm1_t(a.val), int32xm1_t(b.val), 4))); \
} \
inline v_float32x4& operator bin_op##= (v_float32x4& a, const v_float32x4& b) \
{ \
    a.val = float32xm1_t(intrin(int32xm1_t(a.val), int32xm1_t(b.val), 4)); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(&, vandvv_int32xm1)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(|, vorvv_int32xm1)
OPENCV_HAL_IMPL_RISCVV_FLT_BIT_OP(^, vxorvv_int32xm1)

inline v_float32x4 operator ~ (const v_float32x4& a)
{
    return v_float32x4((float32xm1_t)(vnotv_int32xm1((int32xm1_t)(a.val), 4)));
}

#define OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(bin_op, intrin) \
inline v_float64x2 operator bin_op (const v_float64x2& a, const v_float64x2& b) \
{ \
    return v_float64x2(float64xm1_t(intrin(int64xm1_t(a.val), int64xm1_t(b.val), 2))); \
} \
inline v_float64x2& operator bin_op##= (v_float64x2& a, const v_float64x2& b) \
{ \
    a.val = float64xm1_t(intrin(int64xm1_t(a.val), int64xm1_t(b.val), 2)); \
    return a; \
}

OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(&, vandvv_int64xm1)
OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(|, vorvv_int64xm1)
OPENCV_HAL_IMPL_RISCVV_FLT_64BIT_OP(^, vxorvv_int64xm1)

inline v_float64x2 operator ~ (const v_float64x2& a)
{
    return v_float64x2((float64xm1_t)(vnotv_int64xm1((int64xm1_t)(a.val), 2)));
}

inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b)
{
    return v_int16x8(vmulhvv_int16xm1(a.val, b.val, 8));
}
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(vmulhuvv_uint16xm1(a.val, b.val, 8));
}

inline v_uint32x4 v_abs(v_int32x4 x)
{
    e32xm1_t mask=vmsltvx_e32xm1_int32xm1(x.val, 0, 4);
    return v_uint32x4((uint32xm1_t)vrsubvx_mask_int32xm1(x.val, x.val, 0, mask, 4));
}

inline v_uint16x8 v_abs(v_int16x8 x)
{
    e16xm1_t mask=vmsltvx_e16xm1_int16xm1(x.val, 0, 8);
    return v_uint16x8((uint16xm1_t)vrsubvx_mask_int16xm1(x.val, x.val, 0, mask, 8));
}

inline v_uint8x16 v_abs(v_int8x16 x)
{
    e8xm1_t mask=vmsltvx_e8xm1_int8xm1(x.val, 0, 16);
    return v_uint8x16((uint8xm1_t)vrsubvx_mask_int8xm1(x.val, x.val, 0, mask, 16));
}

inline v_float32x4 v_abs(v_float32x4 x)
{
    e32xm1_t mask=vmfltvf_e32xm1_float32xm1(x.val, 0, 4);
    return v_float32x4(vfrsubvf_mask_float32xm1(x.val, x.val, 0, mask, 4));
}

inline v_float64x2 v_abs(v_float64x2 x)
{
    e64xm1_t mask=vmfltvf_e64xm1_float64xm1(x.val, 0, 2);
    return v_float64x2(vfrsubvf_mask_float64xm1(x.val, x.val, 0, mask, 2));
}

inline v_float32x4 v_absdiff(const v_float32x4& a, const v_float32x4& b)
{
    float32xm1_t vmax = vfmaxvv_float32xm1(a.val, b.val, 4);
    float32xm1_t vmin = vfminvv_float32xm1(a.val, b.val, 4);
    return v_float32x4(vfsubvv_float32xm1(vmax, vmin, 4));
}

inline v_float64x2 v_absdiff(const v_float64x2& a, const v_float64x2& b)
{
    float64xm1_t vmax = vfmaxvv_float64xm1(a.val, b.val, 2);
    float64xm1_t vmin = vfminvv_float64xm1(a.val, b.val, 2);
    return v_float64x2(vfsubvv_float64xm1(vmax, vmin, 2));
}

#define OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(bit, num) \
inline v_uint##bit##x##num v_absdiff(v_uint##bit##x##num a, v_uint##bit##x##num b){ \
    uint##bit##xm1_t vmax = vmaxuvv_uint##bit##xm1(a.val, b.val, num);  \
    uint##bit##xm1_t vmin = vminuvv_uint##bit##xm1(a.val, b.val, num);  \
    return v_uint##bit##x##num(vsubvv_uint##bit##xm1(vmax, vmin, num)); \
}

OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(8, 16)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(16, 8)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF_U(32, 4)

/** Saturating absolute difference **/
inline v_int8x16 v_absdiffs(v_int8x16 a, v_int8x16 b){
    int8xm1_t vmax = vmaxvv_int8xm1(a.val, b.val, 16);
    int8xm1_t vmin = vminvv_int8xm1(a.val, b.val, 16);
    return v_int8x16(vssubvv_int8xm1(vmax, vmin, 16));
}
inline v_int16x8 v_absdiffs(v_int16x8 a, v_int16x8 b){
    int16xm1_t vmax = vmaxvv_int16xm1(a.val, b.val, 8);
    int16xm1_t vmin = vminvv_int16xm1(a.val, b.val, 8);
    return v_int16x8(vssubvv_int16xm1(vmax, vmin, 8));
}

#define OPENCV_HAL_IMPL_RISCVV_ABSDIFF(_Tpvec, _Tpvec2, _Tpv, num)  \
inline v_uint##_Tpvec v_absdiff(v_int##_Tpvec a, v_int##_Tpvec b){  \
    int##_Tpvec2##_t val = vwsubvv_int##_Tpvec2##_int##_Tpv(a.val, b.val, num); \
    e##_Tpvec2##_t mask=vmsltvx_e##_Tpvec2##_int##_Tpvec2(val, 0.0, num);       \
    val = vwsubvv_mask_int##_Tpvec2##_int##_Tpv(val, b.val, a.val, mask, num);  \
    return v_uint##_Tpvec(vnclipuvx_uint##_Tpv##_uint##_Tpvec2 ((uint##_Tpvec2##_t)val, 0, num));\
}

OPENCV_HAL_IMPL_RISCVV_ABSDIFF(8x16, 16xm2, 8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF(16x8, 32xm2, 16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_ABSDIFF(32x4, 64xm2, 32xm1, 4)

//  Multiply and expand
inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    int16xm2_u res;
    res.v = vwmulvv_int16xm2_int8xm1(a.val, b.val, 16);
    c.val = res.m1[0];
    d.val = res.m1[1];
}

inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    uint16xm2_u res;
    res.v = vwmuluvv_uint16xm2_uint8xm1(a.val, b.val, 16);
    c.val = res.m1[0];
    d.val = res.m1[1];
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    int32xm2_u res;
    res.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    c.val = res.m1[0];
    d.val = res.m1[1];
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    uint32xm2_u res;
    res.v = vwmuluvv_uint32xm2_uint16xm1(a.val, b.val, 8);
    c.val = res.m1[0];
    d.val = res.m1[1];
}

inline void v_mul_expand(const v_int32x4& a, const v_int32x4& b,
                         v_int64x2& c, v_int64x2& d)
{
    int64xm2_u res;
    res.v = vwmulvv_int64xm2_int32xm1(a.val, b.val, 4);
    c.val = res.m1[0];
    d.val = res.m1[1];
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    uint64xm2_u res;
    res.v = vwmuluvv_uint64xm2_uint32xm1(a.val, b.val, 4);
    c.val = res.m1[0];
    d.val = res.m1[1];
}

OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_add_wrap, vaddvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_add_wrap, vaddvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_add_wrap, vaddvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_add_wrap, vaddvv_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_sub_wrap, vsubvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_sub_wrap, vsubvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_sub_wrap, vsubvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_sub_wrap, vsubvv_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint8x16, v_mul_wrap, vmulvv_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int8x16, v_mul_wrap, vmulvv_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_uint16x8, v_mul_wrap, vmulvv_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_BINN_FUNC(v_int16x8, v_mul_wrap, vmulvv_int16xm1, 8)

//////// Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    int32xm2_u res;
    res.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    res.v = vrgathervv_int32xm2_uint32xm2(res.v, (uint32xm2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vaddvv_int32xm1(res.m1[0], res.m1[1], 4));
}
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    int32xm2_u res;
    res.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    res.v = vrgathervv_int32xm2_uint32xm2(res.v, (uint32xm2_t){0, 2, 4, 6, 1, 3, 5, 7}, 8);
    return v_int32x4(vaddvv_int32xm1(vaddvv_int32xm1(res.m1[0], res.m1[1], 4), c.val, 4));
}

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
    int64xm2_u res;
    res.v = vwmulvv_int64xm2_int32xm1(a.val, b.val, 4);
    res.v = vrgathervv_int64xm2_uint64xm2(res.v, (uint64xm2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vaddvv_int64xm1(res.m1[0], res.m1[1], 2));
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    int64xm2_u res;
    res.v = vwmulvv_int64xm2_int32xm1(a.val, b.val, 4);
    res.v = vrgathervv_int64xm2_uint64xm2(res.v, (uint64xm2_t){0, 2, 1, 3}, 4);
    return v_int64x2(vaddvv_int64xm1(vaddvv_int64xm1(res.m1[0], res.m1[1], 2), c.val, 2));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    uint16xm2_u v1;
    uint32xm2_u v2;
    v1.v = vwmuluvv_uint16xm2_uint8xm1(a.val, b.val, 16);
    v1.v = vrgathervv_uint16xm2(v1.v, (uint16xm2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2.v = vwadduvv_uint32xm2_uint16xm1(v1.m1[0], v1.m1[1], 8);
    return v_uint32x4(vaddvv_uint32xm1(v2.m1[0], v2.m1[1], 4));
}

inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b,
                                   const v_uint32x4& c)
{
    uint16xm2_u v1;
    uint32xm2_u v2;
    v1.v = vwmuluvv_uint16xm2_uint8xm1(a.val, b.val, 16);
    v1.v = vrgathervv_uint16xm2(v1.v, (uint16xm2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2.v = vwadduvv_uint32xm2_uint16xm1(v1.m1[0], v1.m1[1], 8);
    return v_uint32x4(vaddvv_uint32xm1(vaddvv_uint32xm1(v2.m1[0], v2.m1[1], 4), c.val, 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    int16xm2_u v1;
    int32xm2_u v2;
    v1.v = vwmulvv_int16xm2_int8xm1(a.val, b.val, 16);
    v1.v = vrgathervv_int16xm2_uint16xm2(v1.v, (uint16xm2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2.v = vwaddvv_int32xm2_int16xm1(v1.m1[0], v1.m1[1], 8);
    return v_int32x4(vaddvv_int32xm1(v2.m1[0], v2.m1[1], 4));
}

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b,
                                   const v_int32x4& c)
{
    int16xm2_u v1;
    int32xm2_u v2;
    v1.v = vwmulvv_int16xm2_int8xm1(a.val, b.val, 16);
    v1.v = vrgathervv_int16xm2_uint16xm2(v1.v, (uint16xm2_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);
    v2.v = vwaddvv_int32xm2_int16xm1(v1.m1[0], v1.m1[1], 8);
    return v_int32x4(vaddvv_int32xm1(vaddvv_int32xm1(v2.m1[0], v2.m1[1], 4), c.val, 4));
}

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    uint32xm2_u v1;
    uint64xm2_u v2;
    v1.v = vwmuluvv_uint32xm2_uint16xm1(a.val, b.val, 8);
    v1.v = vrgathervv_uint32xm2(v1.v, (uint32xm2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2.v = vwadduvv_uint64xm2_uint32xm1(v1.m1[0], v1.m1[1], 4);
    return v_uint64x2(vaddvv_uint64xm1(v2.m1[0], v2.m1[1], 2));
}

inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b,
                                   const v_uint64x2& c)
{
    uint32xm2_u v1;
    uint64xm2_u v2;
    v1.v = vwmuluvv_uint32xm2_uint16xm1(a.val, b.val, 8);
    v1.v = vrgathervv_uint32xm2(v1.v, (uint32xm2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2.v = vwadduvv_uint64xm2_uint32xm1(v1.m1[0], v1.m1[1], 4);
    return v_uint64x2(vaddvv_uint64xm1(vaddvv_uint64xm1(v2.m1[0], v2.m1[1], 2), c.val, 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    int32xm2_u v1;
    int64xm2_u v2;
    v1.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    v1.v = vrgathervv_int32xm2_uint32xm2(v1.v, (uint32xm2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2.v = vwaddvv_int64xm2_int32xm1(v1.m1[0], v1.m1[1], 4);
    return v_int64x2(vaddvv_int64xm1(v2.m1[0], v2.m1[1], 2));
}

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b,
                                   const v_int64x2& c)
{
    int32xm2_u v1;
    int64xm2_u v2;
    v1.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    v1.v = vrgathervv_int32xm2_uint32xm2(v1.v, (uint32xm2_t){0, 4, 1, 5, 2, 6, 3, 7}, 8);
    v2.v = vwaddvv_int64xm2_int32xm1(v1.m1[0], v1.m1[1], 4);
    return v_int64x2(vaddvv_int64xm1(vaddvv_int64xm1(v2.m1[0], v2.m1[1], 2), c.val, 2));
}
//////// Fast Dot Product ////////
// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{
    int32xm2_u v1;
    v1.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    return v_int32x4(vaddvv_int32xm1(v1.m1[0], v1.m1[1], 4));
}

inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{
    int32xm2_u v1;
    v1.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    return v_int32x4(vaddvv_int32xm1(vaddvv_int32xm1(v1.m1[0], v1.m1[1], 4), c.val, 4));
}

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{
    int64xm2_u v1;
    v1.v = vwmulvv_int64xm2_int32xm1(a.val, b.val, 4);
    return v_int64x2(vaddvv_int64xm1(v1.m1[0], v1.m1[1], 2));
}
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{
    int64xm2_u v1;
    v1.v = vwmulvv_int64xm2_int32xm1(a.val, b.val, 8);
    return v_int64x2(vaddvv_int64xm1(vaddvv_int64xm1(v1.m1[0], v1.m1[1], 4), c.val, 4));
}

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{
    uint16xm2_u v1;
    uint32xm2_u v2;
    v1.v = vwmuluvv_uint16xm2_uint8xm1(a.val, b.val, 16);
    v2.v = vwadduvv_uint32xm2_uint16xm1(v1.m1[0], v1.m1[1], 8);
    return v_uint32x4(vaddvv_uint32xm1(v2.m1[0], v2.m1[1], 4));
}

inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{
    uint16xm2_u v1;
    uint32xm2_u v2;
    v1.v = vwmuluvv_uint16xm2_uint8xm1(a.val, b.val, 16);
    v2.v = vwadduvv_uint32xm2_uint16xm1(v1.m1[0], v1.m1[1], 8);
    return v_uint32x4(vaddvv_uint32xm1(vaddvv_uint32xm1(v2.m1[0], v2.m1[1], 4), c.val, 4));
}

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
    int16xm2_u v1;
    int32xm2_u v2;
    v1.v = vwmulvv_int16xm2_int8xm1(a.val, b.val, 16);
    v2.v = vwaddvv_int32xm2_int16xm1(v1.m1[0], v1.m1[1], 8);
    return v_int32x4(vaddvv_int32xm1(v2.m1[0], v2.m1[1], 4));
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{
    int16xm2_u v1;
    int32xm2_u v2;
    v1.v = vwmulvv_int16xm2_int8xm1(a.val, b.val, 16);
    v2.v = vwaddvv_int32xm2_int16xm1(v1.m1[0], v1.m1[1], 8);
    return v_int32x4(vaddvv_int32xm1(vaddvv_int32xm1(v2.m1[0], v2.m1[1], 4), c.val, 4));
}

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    uint32xm2_u v1;
    uint64xm2_u v2;
    v1.v = vwmuluvv_uint32xm2_uint16xm1(a.val, b.val, 8);
    v2.v = vwadduvv_uint64xm2_uint32xm1(v1.m1[0], v1.m1[1], 4);
    return v_uint64x2(vaddvv_uint64xm1(v2.m1[0], v2.m1[1], 2));
}
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{
    uint32xm2_u v1;
    uint64xm2_u v2;
    v1.v = vwmuluvv_uint32xm2_uint16xm1(a.val, b.val, 8);
    v2.v = vwadduvv_uint64xm2_uint32xm1(v1.m1[0], v1.m1[1], 4);
    return v_uint64x2(vaddvv_uint64xm1(vaddvv_uint64xm1(v2.m1[0], v2.m1[1], 2), c.val, 2));
}

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    int32xm2_u v1;
    int64xm2_u v2;
    v1.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    v2.v = vwaddvv_int64xm2_int32xm1(v1.m1[0], v1.m1[1], 4);
    return v_int64x2(vaddvv_int64xm1(v2.m1[0], v2.m1[1], 2));
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{
    int32xm2_u v1;
    int64xm2_u v2;
    v1.v = vwmulvv_int32xm2_int16xm1(a.val, b.val, 8);
    v2.v = vwaddvv_int64xm2_int32xm1(v1.m1[0], v1.m1[1], 4);
    return v_int64x2(vaddvv_int64xm1(vaddvv_int64xm1(v2.m1[0], v2.m1[1], 2), c.val, 2));
}

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(_Tpvec, _Tpvec2, scalartype, func, intrin, num) \
inline scalartype v_reduce_##func(const v_##_Tpvec##x##num& a) \
{\
    _Tpvec2##xm2_t val = intrin(a.val, vmvvx_##_Tpvec2##xm2(0, num), num);  \
    return val[0];  \
}

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(_Tpvec, scalartype, func, funcu, num) \
inline scalartype v_reduce_##func(const v_##_Tpvec##x##num& a) \
{\
    _Tpvec##xm1_t val = v##funcu##vs_##_Tpvec##xm1(a.val, a.val, num);  \
    return val[0];  \
}
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int8, int16, int, sum, vwredsumvs_int16xm2_int8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int16, int32, int, sum, vwredsumvs_int32xm2_int16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(int32, int64, int, sum, vwredsumvs_int64xm2_int32xm1, 4)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint8, uint16, unsigned, sum, vwredsumuvs_uint16xm2_uint8xm1, 16)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint16, uint32, unsigned, sum, vwredsumuvs_uint32xm2_uint16xm1, 8)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_W(uint32, uint64, unsigned, sum, vwredsumuvs_uint64xm2_uint32xm1, 4)
inline float v_reduce_sum(const v_float32x4& a) \
{\
    float32xm1_t val = vfredsumvs_float32xm1(a.val, vfmvvf_float32xm1(0.0, 4), 4);  \
    return val[0];  \
}
inline double v_reduce_sum(const v_float64x2& a) \
{\
    return a.val[0]+a.val[1];   \
}
inline uint64 v_reduce_sum(const v_uint64x2& a)
{ return a.val[0]+a.val[1]; }
inline int64 v_reduce_sum(const v_int64x2& a)
{ return a.val[0]+a.val[1]; }

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(func)  \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int8,  int, func, red##func, 16)  \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int16, int, func, red##func, 8)   \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int32, int, func, red##func, 4)   \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(int64, int, func, red##func, 2)   \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint8,  unsigned, func, red##func##u, 16) \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint16, unsigned, func, red##func##u, 8)  \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint32, unsigned, func, red##func##u, 4)  \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(uint64, unsigned, func, red##func##u, 2)  \
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP_(float32, float, func, fred##func, 4)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(max)
OPENCV_HAL_IMPL_RISCVV_REDUCE_OP(min)

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
    float32xm1_t m0 = vfmvvf_float32xm1(0.0, 4);
    float32xm1_t a0 = vfredsumvs_float32xm1(a.val, m0, 4);
    float32xm1_t b0 = vfredsumvs_float32xm1(b.val, m0, 4);
    float32xm1_t c0 = vfredsumvs_float32xm1(c.val, m0, 4);
    float32xm1_t d0 = vfredsumvs_float32xm1(d.val, m0, 4);
    return v_float32x4(a0[0], b0[0], c0[0], d0[0]);
}

inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    float32xm1_t x = vfsubvv_float32xm1(a.val, b.val, 4);
    e32xm1_t mask=vmfltvf_e32xm1_float32xm1(x, 0, 4);
    float32xm1_t val = vfrsubvf_mask_float32xm1(x, x, 0, mask, 4);
    float32xm1_t m0 = vfmvvf_float32xm1(0.0, 4);
    float32xm1_t a0 = vfredsumvs_float32xm1(val, m0, 4);
    return a0[0];
}

#define OPENCV_HAL_IMPL_RISCVV_REDUCE_SAD(_Tpvec, _Tpvec2) \
inline unsigned v_reduce_sad(const _Tpvec& a, const _Tpvec&b){  \
    _Tpvec2 x = v_absdiff(a, b);    \
    return v_reduce_sum(x); \
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
    e##_Tp##_t mask = vmseqvv_e##_Tp##_##_T(a.val, b.val, num);    \
    return _Tpvec(vmergevxm_mask_##_T(vmvvx_##_T(0, num), -1, mask, num));    \
} \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ \
    e##_Tp##_t mask = vmsnevv_e##_Tp##_##_T(a.val, b.val, num);    \
    return _Tpvec(vmergevxm_mask_##_T(vmvvx_##_T(0, num), -1, mask, num));    \
} \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ \
    e##_Tp##_t mask = vmslt##uv##_e##_Tp##_##_T(a.val, b.val, num);    \
    return _Tpvec(vmergevxm_mask_##_T(vmvvx_##_T(0, num), -1, mask, num));    \
} \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ \
    e##_Tp##_t mask = vmslt##uv##_e##_Tp##_##_T(b.val, a.val, num);    \
    return _Tpvec(vmergevxm_mask_##_T(vmvvx_##_T(0, num), -1, mask, num));    \
} \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ \
    e##_Tp##_t mask = vmsle##uv##_e##_Tp##_##_T(a.val, b.val, num);    \
    return _Tpvec(vmergevxm_mask_##_T(vmvvx_##_T(0, num), -1, mask, num));    \
} \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ \
    e##_Tp##_t mask = vmsle##uv##_e##_Tp##_##_T(b.val, a.val, num);    \
    return _Tpvec(vmergevxm_mask_##_T(vmvvx_##_T(0, num), -1, mask, num));    \
} \

OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int8x16, 8xm1, int8xm1, 16, vv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int16x8, 16xm1, int16xm1, 8, vv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int32x4, 32xm1, int32xm1, 4, vv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_int64x2, 64xm1, int64xm1, 2, vv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint8x16, 8xm1, uint8xm1, 16, uvv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint16x8, 16xm1, uint16xm1, 8, uvv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint32x4, 32xm1, uint32xm1, 4, uvv)
OPENCV_HAL_IMPL_RISCVV_INT_CMP_OP(v_uint64x2, 64xm1, uint64xm1, 2, uvv)

//TODO: ==
inline v_float32x4 operator == (const v_float32x4& a, const v_float32x4& b)
{
    e32xm1_t mask = vmfeqvv_e32xm1_float32xm1(a.val, b.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}
inline v_float32x4 operator != (const v_float32x4& a, const v_float32x4& b)
{
    e32xm1_t mask = vmfnevv_e32xm1_float32xm1(a.val, b.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}
inline v_float32x4 operator < (const v_float32x4& a, const v_float32x4& b)
{
    e32xm1_t mask = vmfltvv_e32xm1_float32xm1(a.val, b.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}
inline v_float32x4 operator <= (const v_float32x4& a, const v_float32x4& b)
{
    e32xm1_t mask = vmflevv_e32xm1_float32xm1(a.val, b.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}
inline v_float32x4 operator > (const v_float32x4& a, const v_float32x4& b)
{
    e32xm1_t mask = vmfgtvv_e32xm1_float32xm1(a.val, b.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}
inline v_float32x4 operator >= (const v_float32x4& a, const v_float32x4& b)
{
    e32xm1_t mask = vmfgevv_e32xm1_float32xm1(a.val, b.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}
inline v_float32x4 v_not_nan(const v_float32x4& a)
{
    e32xm1_t mask = vmfordvv_e32xm1_float32xm1(a.val, a.val, 4);
    int32xm1_t res = vmergevxm_mask_int32xm1(vmvvx_int32xm1(0.0, 4), -1, mask, 4);
    return v_float32x4((float32xm1_t)res);
}

//TODO: ==
inline v_float64x2 operator == (const v_float64x2& a, const v_float64x2& b)
{
    e64xm1_t mask = vmfeqvv_e64xm1_float64xm1(a.val, b.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}
inline v_float64x2 operator != (const v_float64x2& a, const v_float64x2& b)
{
    e64xm1_t mask = vmfnevv_e64xm1_float64xm1(a.val, b.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}
inline v_float64x2 operator < (const v_float64x2& a, const v_float64x2& b)
{
    e64xm1_t mask = vmfltvv_e64xm1_float64xm1(a.val, b.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}
inline v_float64x2 operator <= (const v_float64x2& a, const v_float64x2& b)
{
    e64xm1_t mask = vmflevv_e64xm1_float64xm1(a.val, b.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}
inline v_float64x2 operator > (const v_float64x2& a, const v_float64x2& b)
{
    e64xm1_t mask = vmfgtvv_e64xm1_float64xm1(a.val, b.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}
inline v_float64x2 operator >= (const v_float64x2& a, const v_float64x2& b)
{
    e64xm1_t mask = vmfgevv_e64xm1_float64xm1(a.val, b.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}
inline v_float64x2 v_not_nan(const v_float64x2& a)
{
    e64xm1_t mask = vmfordvv_e64xm1_float64xm1(a.val, a.val, 2);
    int64xm1_t res = vmergevxm_mask_int64xm1(vmvvx_int64xm1(0.0, 2), -1, mask, 2);
    return v_float64x2((float64xm1_t)res);
}

inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}
#endif
