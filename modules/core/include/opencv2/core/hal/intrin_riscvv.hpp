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

#define OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(_Tp, tmp) \
inline void v_transpose4x4(const v_##_Tp##32x4& a0, const v_##_Tp##32x4& a1, \
                         const v_##_Tp##32x4& a2, const v_##_Tp##32x4& a3, \
                         v_##_Tp##32x4& b0, v_##_Tp##32x4& b1, \
                         v_##_Tp##32x4& b2, v_##_Tp##32x4& b3) \
{ \
    _Tp##32xm4_u val;    \
    val.m1[0] = a0.val;    \
    val.m1[1] = a1.val;    \
    val.m1[2] = a2.val;    \
    val.m1[3] = a3.val;     \
    val.v = vrgathervv##tmp##uint32xm4(val.v, (uint32xm4_t){0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15}, 16);    \
    b0.val = val.m1[0];    \
    b1.val = val.m1[1];    \
    b2.val = val.m1[2];    \
    b3.val = val.m1[3];    \
}
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(uint, _)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(int, _int32xm4_)
OPENCV_HAL_IMPL_RISCVV_TRANSPOSE4x4(float, _float32xm4_)


#define OPENCV_HAL_IMPL_RISCVV_SHIFT_LEFT(_Tpvec, suffix, num) \
inline _Tpvec operator << (const _Tpvec& a, int n) \
{ return _Tpvec((vsllvx_##suffix##xm1(a.val, n, num))); } \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ return _Tpvec((vsllvx_##suffix##xm1(a.val, n, num))); }

#define OPENCV_HAL_IMPL_RISCVV_SHIFT_RIGHT(_Tpvec, suffix, num, intric) \
inline _Tpvec operator >> (const _Tpvec& a, int n) \
{ return _Tpvec((v##intric##vx_##suffix##xm1(a.val, n, num))); } \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ return _Tpvec((v##intric##vx_##suffix##xm1(a.val, n, num))); }\
template<int n> inline _Tpvec v_rshr(const _Tpvec& a) \
{ return _Tpvec((v##intric##vx_##suffix##xm1(vaddvx_##suffix##xm1(a.val, 1<<(n-1), num), n, num))); }

// trade efficiency for convenience
#define OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(suffix, num, intrin) \
OPENCV_HAL_IMPL_RISCVV_SHIFT_LEFT(v_##suffix##x##num, suffix, num) \
OPENCV_HAL_IMPL_RISCVV_SHIFT_RIGHT(v_##suffix##x##num, suffix, num, intrin)

OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint8, 16, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint16, 8, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint32, 4, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(uint64, 2, srl)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int8, 16, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int16, 8, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int32, 4, sra)
OPENCV_HAL_IMPL_RISCVV_SHIFT_OP(int64, 2, sra)
#define VUP4(n) {0, 1, 2, 3}
#define VUP8(n) {0, 1, 2, 3, 4, 5, 6, 7}
#define VUP16(n) {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}
#define VUP2(n) {0, 1}
#define OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(_Tpvec, suffix, num, num2, vmv, len) \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{    \
    suffix##xm1_t tmp;    \
    tmp = vmv##_##suffix##xm1(0, num);\
        tmp = vslideupvx_mask_##suffix##xm1(tmp, a.val, n, vmsetm_e##len##xm1(num), num);\
        return _Tpvec(tmp);\
} \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{     \
        return _Tpvec(vslidedownvx_##suffix##xm1(a.val, n, num));\
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    suffix##xm2_u tmp;    \
    tmp.m1[0] = a.val;\
    tmp.m1[1] = b.val;\
        tmp.v = vslidedownvx_##suffix##xm2(tmp.v, n, num2);\
        return _Tpvec(tmp.m1[0]);\
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    suffix##xm2_u tmp;    \
    tmp.m1[0] = b.val;\
    tmp.m1[1] = a.val;\
        tmp.v = vslideupvx_##suffix##xm2(tmp.v, n, num2);\
        return _Tpvec(tmp.m1[1]);\
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ \
    CV_UNUSED(b); return a; \
}

OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint8x16, uint8, 16, 32, vmvvx, 8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int8x16, int8, 16, 32, vmvvx, 8)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint16x8, uint16, 8, 16, vmvvx, 16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int16x8, int16, 8, 16, vmvvx, 16)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint32x4, uint32, 4, 8, vmvvx, 32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int32x4, int32, 4, 8, vmvvx, 32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_uint64x2, uint64, 2, 4, vmvvx, 64)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_int64x2, int64, 2, 4, vmvvx, 64)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float32x4, float32, 4, 8, vfmvvf, 32)
OPENCV_HAL_IMPL_RISCVV_ROTATE_OP(v_float64x2, float64, 2, 4, vfmvvf, 64)

#define OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(_Tpvec, _Tp, _Tp2, hnum, num) \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
  typedef uint64 CV_DECL_ALIGNED(1) unaligned_uint64; \
  uint64xm1_t tmp = {*(unaligned_uint64*)ptr0, *(unaligned_uint64*)ptr1};\
    return _Tpvec(_Tp2##_t(tmp)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ return _Tpvec(vlev_##_Tp2(ptr, hnum)); }\
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(vlev_##_Tp2(ptr, num)); } \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec((_Tp2##_t)vlev_uint8xm1((unsigned char *)ptr, 16)); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ vsev_##_Tp2(ptr, a.val, hnum);}\
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
  _Tp2##_t a0 = vslidedownvx_##_Tp2(a.val, hnum, num);    \
  vsev_##_Tp2(ptr, a0, hnum);}\
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ vsev_uint8xm1((unsigned char *)ptr, (uint8xm1_t)a.val, 16); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ vsev_##_Tp2(ptr, a.val, num); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ vsev_##_Tp2(ptr, a.val, num); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ vsev_##_Tp2(ptr, a.val, num); }

OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint8x16, uchar, uint8xm1, 8, 16)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int8x16,  schar, int8xm1, 8, 16)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint16x8, ushort, uint16xm1, 4, 8)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int16x8,  short,  int16xm1, 4, 8)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint32x4, unsigned, uint32xm1, 2, 4)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int32x4,  int,     int32xm1, 2, 4)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_uint64x2, unsigned long, uint64xm1, 1, 2)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_int64x2,  long,     int64xm1, 1, 2)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float32x4, float, float32xm1, 2, 4)
OPENCV_HAL_IMPL_RISCVV_LOADSTORE_OP(v_float64x2, double, float64xm1, 1, 2)


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
    return v_int8x16(vlev_int8xm1(elems, 16));
#else
    int32xm4_t index32 = vlev_int32xm4(idx, 16);
    int16xm2_t index16 = vnsravx_int16xm2_int32xm4(index32, 0, 16);
    int8xm1_t index = vnsravx_int8xm1_int16xm2(index16, 0, 16);
    return v_int8x16(vlxbv_int8xm1(tab, index, 16));
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
    return v_int8x16(vlev_int8xm1(elems, 16));
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
    return v_int8x16(vlev_int8xm1(elems, 16));
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
    return v_int16x8(vlev_int16xm1(elems, 8));
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
    return v_int16x8(vlev_int16xm1(elems, 8));
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
    return v_int16x8(vlev_int16xm1(elems, 8));
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
    return v_int32x4(vlev_int32xm1(elems, 4));
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
    return v_int32x4(vlev_int32xm1(elems, 4));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(vlev_int32xm1(tab+idx[0], 4));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    int64xm1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_int64x2(res);
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(vlev_int64xm1(tab+idx[0], 2));
}

inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx)
{
    uint64xm1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_uint64x2(res);
}
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx)
{
    return v_uint64x2(vlev_uint64xm1(tab+idx[0], 2));
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
    return v_float32x4(vlev_float32xm1(elems, 4));
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
    return v_float32x4(vlev_float32xm1(elems, 4));
}
inline v_float32x4 v_lut_quads(const float* tab, const int* idx)
{
    return v_float32x4(vlev_float32xm1(tab + idx[0], 4));
}
inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    float64xm1_t res = {tab[idx[0]], tab[idx[1]]};
    return v_float64x2(res);
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx)
{
    return v_float64x2(vlev_float64xm1(tab+idx[0], 2));
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
    return v_int32x4(vlev_int32xm1(elems, 4));
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
    return v_uint32x4(vlev_uint32xm1(elems, 4));
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
    return v_float32x4(vlev_float32xm1(elems, 4));
}
inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    float64xm1_t res = {tab[idxvec.val[0]], tab[idxvec.val[1]]};
    return v_float64x2(res);
}
inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    int32xm1_t index_x = vmulvx_int32xm1(idxvec.val, 4, 4);
    int32xm1_t index_y = vaddvx_int32xm1(index_x, 4, 4);

    x.val = vlxev_float32xm1(tab, index_x, 4);
    y.val = vlxev_float32xm1(tab, index_y, 4);
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);

    x = v_float64x2(tab[idx[0]], tab[idx[1]]);
    y = v_float64x2(tab[idx[0]+1], tab[idx[1]+1]);
}


#define OPENCV_HAL_IMPL_RISCVV_PACKS(_Tp2, num2, _Tp, num, pack, intrin, shr, _Type) \
inline v_##_Tp##x##num v_##pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    _Tp2##xm2_u tmp;    \
    tmp.m1[0] = a.val;    \
    tmp.m1[1] = b.val;    \
    return v_##_Tp##x##num(shr##_##_Tp##xm1_##_Tp2##xm2(tmp.v, 0, num)); \
}\
template<int n> inline \
v_##_Tp##x##num v_rshr_##pack(const v_##_Tp2##x##num2& a, const v_##_Tp2##x##num2& b) \
{ \
    _Tp2##xm2_u tmp;    \
    tmp.m1[0] = a.val;    \
    tmp.m1[1] = b.val;    \
    return v_##_Tp##x##num(intrin##_##_Tp##xm1_##_Tp2##xm2(tmp.v, n, num)); \
}\
inline void v_##pack##_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    _Tp2##xm2_u tmp;    \
    tmp.m1[0] = a.val;    \
    tmp.m1[1] = vmvvx_##_Tp2##xm1(0, num2);    \
    vsev_##_Tp##xm1(ptr, shr##_##_Tp##xm1_##_Tp2##xm2(tmp.v, 0, num), num2); \
}\
template<int n> inline \
void v_rshr_##pack##_store(_Type* ptr, const v_##_Tp2##x##num2& a) \
{ \
    _Tp2##xm2_u tmp;    \
    tmp.m1[0] = a.val;    \
    tmp.m1[1] = vmvvx_##_Tp2##xm1(0, num2);    \
    vsev_##_Tp##xm1(ptr, intrin##_##_Tp##xm1_##_Tp2##xm2(tmp.v, n, num), num2); \
}
OPENCV_HAL_IMPL_RISCVV_PACKS(int16, 8, int8, 16, pack, vnclipvx, vnclipvx, signed char)
OPENCV_HAL_IMPL_RISCVV_PACKS(int32, 4, int16, 8, pack, vnclipvx, vnclipvx, signed short)
OPENCV_HAL_IMPL_RISCVV_PACKS(int64, 2, int32, 4, pack, vnclipvx, vnsravx, int)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint16, 8, uint8, 16, pack, vnclipuvx, vnclipuvx, unsigned char)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint32, 4, uint16, 8, pack, vnclipuvx, vnclipuvx, unsigned short)
OPENCV_HAL_IMPL_RISCVV_PACKS(uint64, 2, uint32, 4, pack, vnclipuvx, vnsrlvx, unsigned int)

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    uint16xm2_u tmp;    \
    tmp.m1[0] = a.val;    \
    tmp.m1[1] = b.val;    \
    return v_uint8x16(vnsrlvi_uint8xm1_uint16xm2(tmp.v, 0, 16));
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    uint32xm2_u vab;    \
    uint32xm2_u vcd;    \
    uint16xm2_u v16;    \
    vab.m1[0] = a.val;    \
    vab.m1[1] = b.val;    \
    v16.m1[0] = vnsrlvi_uint16xm1_uint32xm2(vab.v, 0, 8);
    vcd.m1[0] = c.val;    \
    vcd.m1[1] = d.val;    \
    v16.m1[1] = vnsrlvi_uint16xm1_uint32xm2(vcd.v, 0, 8);
    return v_uint8x16(vnsrlvi_uint8xm1_uint16xm2(v16.v, 0, 16));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    uint64xm2_u vab;    \
    uint64xm2_u vcd;    \
    uint32xm2_u vabcd;    \
    uint16xm2_u vres;    \
    vab.m1[0] = a.val;    \
    vab.m1[1] = b.val;    \
    vabcd.m1[0] = vnsrlvi_uint32xm1_uint64xm2(vab.v, 0, 4);
    vcd.m1[0] = c.val;    \
    vcd.m1[1] = d.val;    \
    vabcd.m1[1] = vnsrlvi_uint32xm1_uint64xm2(vcd.v, 0, 4);
    vres.m1[0] = vnsrlvi_uint16xm1_uint32xm2(vabcd.v, 0, 8);
    uint64xm2_u vef;    \
    uint64xm2_u vgh;    \
    uint32xm2_u vefgh;    \
    vef.m1[0] = e.val;    \
    vef.m1[1] = f.val;    \
    vefgh.m1[0] = vnsrlvi_uint32xm1_uint64xm2(vef.v, 0, 4);
    vgh.m1[0] = g.val;    \
    vgh.m1[1] = h.val;    \
    vefgh.m1[1] = vnsrlvi_uint32xm1_uint64xm2(vgh.v, 0, 4);
    vres.m1[1] = vnsrlvi_uint16xm1_uint32xm2(vefgh.v, 0, 8);

    return v_uint8x16(vnsrlvi_uint8xm1_uint16xm2(vres.v, 0, 16));
}

#define OPENCV_HAL_IMPL_RISCVV_PACK_U(tp1, num1, tp2, num2, _Tp) \
inline v_uint##tp1##x##num1 v_pack_u(const v_int##tp2##x##num2& a, const v_int##tp2##x##num2& b) \
{ \
    int##tp2##xm2_u tmp;    \
    tmp.m1[0] = (int##tp2##xm1_t)a.val;    \
    tmp.m1[1] = (int##tp2##xm1_t)b.val;    \
    int##tp2##xm2_t val = vmaxvx_int##tp2##xm2(tmp.v, 0, num1);\
    return v_uint##tp1##x##num1(vnclipuvi_uint##tp1##xm1_uint##tp2##xm2((uint##tp2##xm2_t)val, 0, num1));    \
} \
inline void v_pack_u_store(_Tp* ptr, const v_int##tp2##x##num2& a) \
{ \
    int##tp2##xm2_u tmp;    \
    tmp.m1[0] = (int##tp2##xm1_t)a.val;    \
    int##tp2##xm2_t val = vmaxvx_int##tp2##xm2(tmp.v, 0, num1);\
    return vsev_uint##tp1##xm1(ptr, vnclipuvi_uint##tp1##xm1_uint##tp2##xm2((uint##tp2##xm2_t)val, 0, num1), num2);    \
} \
template<int n> inline \
v_uint##tp1##x##num1 v_rshr_pack_u(const v_int##tp2##x##num2& a, const v_int##tp2##x##num2& b) \
{ \
    int##tp2##xm2_u tmp;    \
    tmp.m1[0] = (int##tp2##xm1_t)a.val;    \
    tmp.m1[1] = (int##tp2##xm1_t)b.val;    \
    int##tp2##xm2_t val = vmaxvx_int##tp2##xm2(tmp.v, 0, num1);\
    return v_uint##tp1##x##num1(vnclipuvi_uint##tp1##xm1_uint##tp2##xm2((uint##tp2##xm2_t)val, n, num1));    \
} \
template<int n> inline \
void v_rshr_pack_u_store(_Tp* ptr, const v_int##tp2##x##num2& a) \
{ \
    int##tp2##xm2_u tmp;    \
    tmp.m1[0] = (int##tp2##xm1_t)a.val;    \
    int##tp2##xm2_t val_ = vmaxvx_int##tp2##xm2(tmp.v, 0, num1);\
    uint##tp1##xm1_t val = vnclipuvi_uint##tp1##xm1_uint##tp2##xm2((uint##tp2##xm2_t)val_, n, num1);    \
    return vsev_uint##tp1##xm1(ptr, val, num2);\
}
OPENCV_HAL_IMPL_RISCVV_PACK_U(8, 16, 16, 8, unsigned char )
OPENCV_HAL_IMPL_RISCVV_PACK_U(16, 8, 32, 4, unsigned short)

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

inline uint8xm1_t vcnt_u8(uint8xm1_t val){
    uint8xm1_t v0 = val & 1;
    return vlxbuv_uint8xm1((unsigned char*)popCountTable, val >> 1, 16)+v0;
}

inline v_uint8x16
v_popcount(const v_uint8x16& a)
{
    return v_uint8x16(vcnt_u8(a.val));
}

inline v_uint8x16
v_popcount(const v_int8x16& a)
{
    return v_uint8x16(vcnt_u8((uint8xm1_t)a.val));
}

inline v_uint16x8
v_popcount(const v_uint16x8& a)
{
    uint8xm2_u tmp;
    tmp.m1[0] = vcnt_u8((uint8xm1_t)a.val);
    uint64xm2_t mask = (uint64xm2_t){0x0E0C0A0806040200, 0, 0x0F0D0B0907050301, 0};
    tmp.v = vrgathervv_uint8xm2(tmp.v, (uint8xm2_t)mask, 32);    \
    uint16xm2_u res;
    res.v = vwadduvv_uint16xm2_uint8xm1(tmp.m1[0], tmp.m1[1], 8);
    return v_uint16x8(res.m1[0]);
}

inline v_uint16x8
v_popcount(const v_int16x8& a)
{
    uint8xm2_u tmp;
    tmp.m1[0] = vcnt_u8((uint8xm1_t)a.val);
    uint64xm2_t mask = (uint64xm2_t){0x0E0C0A0806040200, 0, 0x0F0D0B0907050301, 0};
    tmp.v = vrgathervv_uint8xm2(tmp.v, (uint8xm2_t)mask, 32);    \
    uint16xm2_u res;
    res.v = vwadduvv_uint16xm2_uint8xm1(tmp.m1[0], tmp.m1[1], 8);
    return v_uint16x8(res.m1[0]);
}

inline v_uint32x4
v_popcount(const v_uint32x4& a)
{
    uint8xm2_u tmp;
    tmp.m1[0] = vcnt_u8((uint8xm1_t)a.val);
    uint64xm2_t mask = (uint64xm2_t){0xFFFFFFFF0C080400, 0xFFFFFFFF0D090501,
    0xFFFFFFFF0E0A0602, 0xFFFFFFFF0F0B0703};
    tmp.v = vrgathervv_uint8xm2(tmp.v, (uint8xm2_t)mask, 32);    \
    uint16xm2_u res_;
    res_.v = vwadduvv_uint16xm2_uint8xm1(tmp.m1[0], tmp.m1[1], 16);
    uint32xm2_u res;
    res.v = vwadduvv_uint32xm2_uint16xm1(res_.m1[0], res_.m1[1], 4);
    return v_uint32x4(res.m1[0]);
}

inline v_uint32x4
v_popcount(const v_int32x4& a)
{
    uint8xm2_u tmp;
    tmp.m1[0] = vcnt_u8((uint8xm1_t)a.val);
    uint64xm2_t mask = (uint64xm2_t){0xFFFFFFFF0C080400, 0xFFFFFFFF0D090501,
    0xFFFFFFFF0E0A0602, 0xFFFFFFFF0F0B0703};
    tmp.v = vrgathervv_uint8xm2(tmp.v, (uint8xm2_t)mask, 32);    \
    uint16xm2_u res_;
    res_.v = vwadduvv_uint16xm2_uint8xm1(tmp.m1[0], tmp.m1[1], 16);
    uint32xm2_u res;
    res.v = vwadduvv_uint32xm2_uint16xm1(res_.m1[0], res_.m1[1], 4);
    return v_uint32x4(res.m1[0]);
}

inline v_uint64x2
v_popcount(const v_uint64x2& a)
{
    uint8xm2_u tmp;
    tmp.m1[0] = vcnt_u8((uint8xm1_t)a.val);
    uint64xm2_t mask = (uint64xm2_t){0x0706050403020100, 0x0000000000000000,
    0x0F0E0D0C0B0A0908, 0x0000000000000000};
    tmp.v = vrgathervv_uint8xm2(tmp.v, (uint8xm2_t)mask, 32);    \
    uint8xm1_t zero = vmvvx_uint8xm1(0, 16);
    tmp.m1[0] = vredsumvs_uint8xm1(tmp.m1[0], zero, 8);
    tmp.m1[1] = vredsumvs_uint8xm1(tmp.m1[1], zero, 8);

    return v_uint64x2((unsigned long)tmp.m1[0][0], (unsigned long)tmp.m1[1][0]);
}

inline v_uint64x2
v_popcount(const v_int64x2& a)
{
    uint8xm2_u tmp;
    tmp.m1[0] = vcnt_u8((uint8xm1_t)a.val);
    uint64xm2_t mask = (uint64xm2_t){0x0706050403020100, 0x0000000000000000,
    0x0F0E0D0C0B0A0908, 0x0000000000000000};
    tmp.v = vrgathervv_uint8xm2(tmp.v, (uint8xm2_t)mask, 32);    \
    uint8xm1_t zero = vmvvx_uint8xm1(0, 16);
    tmp.m1[0] = vredsumvs_uint8xm1(tmp.m1[0], zero, 8);
    tmp.m1[1] = vredsumvs_uint8xm1(tmp.m1[1], zero, 8);

    return v_uint64x2((unsigned long)tmp.m1[0][0], (unsigned long)tmp.m1[1][0]);
}


#define SMASK 1, 2, 4, 8, 16, 32, 64, 128
inline int v_signmask(const v_uint8x16& a)
{
    uint8xm1_t t0 = vsrlvx_uint8xm1(a.val, 7, 16);
    uint8xm1_t m1 = (uint8xm1_t){SMASK, SMASK};
    uint16xm2_u t1;
    t1.v = vwmuluvv_uint16xm2_uint8xm1(t0, m1, 16);
    uint32xm2_t t2, res;
    res = vmvvx_uint32xm2(0, 8);
    t2 = vwmuluvx_uint32xm2_uint16xm1(t1.m1[1], 256, 8);
    res = vredsumvs_uint32xm2(t2, res, 8);
    res = vwredsumuvs_uint32xm2_uint16xm1(t1.m1[0], res, 8);
    return res[0];
}
inline int v_signmask(const v_int8x16& a)
{
    uint8xm1_t t0 = vsrlvx_uint8xm1((uint8xm1_t)a.val, 7, 16);
    uint8xm1_t m1 = (uint8xm1_t){SMASK, SMASK};
    int16xm2_u t1;
    t1.v = (int16xm2_t)vwmuluvv_uint16xm2_uint8xm1(t0, m1, 16);
    int32xm2_t t2, res;
    res = vmvvx_int32xm2(0, 8);
    t2 = vwmulvx_int32xm2_int16xm1(t1.m1[1], 256, 8);
    res = vredsumvs_int32xm2(t2, res, 8);
    res = vwredsumvs_int32xm2_int16xm1(t1.m1[0], res, 8);
    return res[0];
}

inline int v_signmask(const v_int16x8& a)
{
    int16xm1_t t0 = (int16xm1_t)vsrlvx_uint16xm1((uint16xm1_t)a.val, 15, 8);
    int16xm1_t m1 = (int16xm1_t){SMASK};
    int16xm1_t t1 = vmulvv_int16xm1(t0, m1, 8);
    int16xm1_t res = vmvvx_int16xm1(0, 8);
    res = vredsumvs_int16xm1(t1, res, 8);
    return res[0];
}
inline int v_signmask(const v_uint16x8& a)
{
    int16xm1_t t0 = (int16xm1_t)vsrlvx_uint16xm1((uint16xm1_t)a.val, 15, 8);
    int16xm1_t m1 = (int16xm1_t){SMASK};
    int16xm1_t t1 = vmulvv_int16xm1(t0, m1, 8);
    int16xm1_t res = vmvvx_int16xm1(0, 8);
    res = vredsumvs_int16xm1(t1, res, 8);
    return res[0];
}
inline int v_signmask(const v_int32x4& a)
{
    int32xm1_t t0 = (int32xm1_t)vsrlvx_uint32xm1((uint32xm1_t)a.val, 31, 4);
    int32xm1_t m1 = (int32xm1_t){1, 2, 4, 8};
    int32xm1_t res;
    res = vmvvx_int32xm1(0, 8);
    int32xm1_t t1 = vmulvv_int32xm1(t0, m1, 4);
    res = vredsumvs_int32xm1(t1, res, 4);
    return res[0];
}
inline int v_signmask(const v_uint32x4& a)
{
    int32xm1_t t0 = (int32xm1_t)vsrlvx_uint32xm1(a.val, 31, 4);
    int32xm1_t m1 = (int32xm1_t){1, 2, 4, 8};
    int32xm1_t res;
    res = vmvvx_int32xm1(0, 8);
    int32xm1_t t1 = vmulvv_int32xm1(t0, m1, 4);
    res = vredsumvs_int32xm1(t1, res, 4);
    return res[0];
}
inline int v_signmask(const v_uint64x2& a)
{
    uint64xm1_t v0 = vsrlvx_uint64xm1(a.val, 63, 2);
    int res = (int)v0[0] + ((int)v0[1] << 1);
    return res;
}
inline int v_signmask(const v_int64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float64x2& a)
{ return v_signmask(v_reinterpret_as_u64(a)); }
inline int v_signmask(const v_float32x4& a)
{
    int32xm1_t t0 = (int32xm1_t)vsrlvx_uint32xm1((uint32xm1_t)a.val, 31, 4);
    int32xm1_t m1 = (int32xm1_t){1, 2, 4, 8};
    int32xm1_t res;
    res = vmvvx_int32xm1(0, 8);
    int32xm1_t t1 = vmulvv_int32xm1(t0, m1, 4);
    res = vredsumvs_int32xm1(t1, res, 4);
    return res[0];
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

#define OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(_Tpvec, suffix, shift, num) \
inline bool v_check_all(const v_##_Tpvec& a) \
{ \
    suffix##_t v0 = vsrlvx_##suffix(vnotv_##suffix(a.val, num), shift, num); \
    uint64xm1_t v1 = uint64xm1_t(v0); \
    return (v1[0] | v1[1]) == 0; \
} \
inline bool v_check_any(const v_##_Tpvec& a) \
{ \
    suffix##_t v0 = vsrlvx_##suffix(a.val, shift, num); \
    uint64xm1_t v1 = uint64xm1_t(v0); \
    return (v1[0] | v1[1]) != 0; \
}

OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint8x16, uint8xm1, 7, 16)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint16x8, uint16xm1, 15, 8)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint32x4, uint32xm1, 31, 4)
OPENCV_HAL_IMPL_RISCVV_CHECK_ALLANY(uint64x2, uint64xm1, 63, 2)
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
    return _Tpvec(vmergevvm_mask_##suffix(b.val, a.val, _Tpvec2(mask.val), num)); \
}

OPENCV_HAL_IMPL_RISCVV_SELECT(v_int8x16,  int8xm1, e8xm1_t, 16)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_int16x8,  int16xm1, e16xm1_t, 8)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_int32x4,  int32xm1, e32xm1_t, 4)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint8x16, uint8xm1, e8xm1_t, 16)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint16x8, uint16xm1, e16xm1_t, 8)
OPENCV_HAL_IMPL_RISCVV_SELECT(v_uint32x4, uint32xm1, e32xm1_t, 4)
inline v_float32x4 v_select(const v_float32x4& mask, const v_float32x4& a, const v_float32x4& b)
{
    return v_float32x4((float32xm1_t)vmergevvm_mask_uint32xm1((uint32xm1_t)b.val, (uint32xm1_t)a.val, e32xm1_t(vfcvtxufv_uint32xm1_float32xm1(mask.val, 4)), 4));
}
inline v_float64x2 v_select(const v_float64x2& mask, const v_float64x2& a, const v_float64x2& b)
{
    return v_float64x2((float64xm1_t)vmergevvm_mask_uint64xm1((uint64xm1_t)b.val, (uint64xm1_t)a.val, e64xm1_t(vfcvtxufv_uint64xm1_float64xm1(mask.val, 2)), 2));
}

#define OPENCV_HAL_IMPL_RISCVV_EXPAND(add, _Tpvec, _Tpwvec, _Tp, _Tp1, num1, _Tp2, num2) \
inline void v_expand(const _Tpvec& a, v_##_Tpwvec& b0, v_##_Tpwvec& b1) \
{ \
    _Tp2##_u b;\
    b.v = vw##add##vv_##_Tp2##_##_Tp1(a.val, vmvvx_##_Tp1(0, num1), num1);    \
    b0.val = b.m1[0]; \
    b1.val = b.m1[1]; \
} \
inline v_##_Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _Tp2##_u b;    \
    b.v = vw##add##vv_##_Tp2##_##_Tp1(a.val, vmvvx_##_Tp1(0, num2), num2);    \
    return v_##_Tpwvec(b.m1[0]); \
} \
inline v_##_Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _Tp2##_u b;\
    b.v = vw##add##vv_##_Tp2##_##_Tp1(a.val, vmvvx_##_Tp1(0, num1), num1);    \
    return v_##_Tpwvec(b.m1[1]); \
} \
inline v_##_Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    _Tp2##_u b;    \
    _Tp1##_t val = vlev_##_Tp1(ptr, num2);    \
    b.v = vw##add##vv_##_Tp2##_##_Tp1(val, vmvvx_##_Tp1(0, num2), num2);    \
    return v_##_Tpwvec(b.m1[0]); \
}

OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint8x16, uint16x8, uchar, uint8xm1, 16, uint16xm2, 8)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint16x8, uint32x4, ushort,  uint16xm1, 8, uint32xm2, 4)
OPENCV_HAL_IMPL_RISCVV_EXPAND(addu, v_uint32x4, uint64x2, uint,  uint32xm1, 4, uint64xm2, 2)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int8x16, int16x8, schar,  int8xm1, 16, int16xm2, 8)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int16x8, int32x4, short,  int16xm1, 8, int32xm2, 4)
OPENCV_HAL_IMPL_RISCVV_EXPAND(add, v_int32x4, int64x2, int,  int32xm1, 4, int64xm2, 2)

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    uint16xm2_u b;
    uint32xm2_u c;
    uint8xm1_t val = vlev_uint8xm1(ptr, 4);    \
    b.v = vwadduvv_uint16xm2_uint8xm1(val, vmvvx_uint8xm1(0, 4), 4);    \
    c.v = vwadduvv_uint32xm2_uint16xm1(b.m1[0], vmvvx_uint16xm1(0, 4), 4);    \
    return v_uint32x4(c.m1[0]);
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    int16xm2_u b;
    int32xm2_u c;
    int8xm1_t val = vlev_int8xm1(ptr, 4);    \
    b.v = vwaddvv_int16xm2_int8xm1(val, vmvvx_int8xm1(0, 4), 4);    \
    c.v = vwaddvv_int32xm2_int16xm1(b.m1[0], vmvvx_int16xm1(0, 4), 4);    \
    return v_int32x4(c.m1[0]);
}
#define VITL_16 (uint64xm2_t){0x1303120211011000, 0x1707160615051404, 0x1B0B1A0A19091808, 0x1F0F1E0E1D0D1C0C}
#define VITL_8 (uint64xm2_t){0x0009000100080000, 0x000B0003000A0002, 0x000D0005000C0004, 0x000F0007000E0006}
#define VITL_4 (uint64xm2_t){0x0000000400000000, 0x0000000500000001, 0x0000000600000002, 0x0000000700000003}
#define VITL_2 (uint64xm2_t){0, 2, 1, 3}
#define LOW_4  0x0000000100000000, 0x0000000500000004
#define LOW_8  0x0003000200010000, 0x000B000A00090008
#define LOW_16 0x0706050403020100, 0x1716151413121110
#define HIGH_4  0x0000000300000002, 0x0000000700000006
#define HIGH_8  0x0007000600050004, 0x000F000E000D000C
#define HIGH_16 0x0F0E0D0C0B0A0908,  0x1F1E1D1C1B1A1918
#define OPENCV_HAL_IMPL_RISCVV_UNPACKS(_Tpvec, _Tp, _UTp, num, num2, len, numh) \
inline void v_zip(const v_##_Tpvec& a0, const v_##_Tpvec& a1, v_##_Tpvec& b0, v_##_Tpvec& b1) \
{ \
    _Tp##xm2_u tmp;\
    tmp.m1[0] = a0.val;    \
    tmp.m1[1] = a1.val;    \
    uint64xm2_t mask = VITL_##num;    \
    tmp.v = (_Tp##xm2_t)vrgathervv_##_UTp##xm2((_UTp##xm2_t)tmp.v, (_UTp##xm2_t)mask, num2);    \
    b0.val = tmp.m1[0]; \
    b1.val = tmp.m1[1]; \
} \
inline v_##_Tpvec v_combine_low(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    _Tp##xm1_t b0 = vslideupvx_mask_##_Tp##xm1(a.val, b.val, numh, vmsetm_e##len##xm1(num), num);    \
    return v_##_Tpvec(b0);\
} \
inline v_##_Tpvec v_combine_high(const v_##_Tpvec& a, const v_##_Tpvec& b) \
{ \
    _Tp##xm1_t b0 = vslidedownvx_##_Tp##xm1(b.val, numh, num);    \
    _Tp##xm1_t a0 = vslidedownvx_##_Tp##xm1(a.val, numh, num);    \
    _Tp##xm1_t b1 = vslideupvx_mask_##_Tp##xm1(a0, b0, numh, vmsetm_e##len##xm1(num), num);    \
    return v_##_Tpvec(b1);\
} \
inline void v_recombine(const v_##_Tpvec& a, const v_##_Tpvec& b, v_##_Tpvec& c, v_##_Tpvec& d) \
{ \
    c.val = vslideupvx_mask_##_Tp##xm1(a.val, b.val, numh, vmsetm_e##len##xm1(num), num);    \
    _Tp##xm1_t b0 = vslidedownvx_##_Tp##xm1(b.val, numh, num);    \
    _Tp##xm1_t a0 = vslidedownvx_##_Tp##xm1(a.val, numh, num);    \
    d.val = vslideupvx_mask_##_Tp##xm1(a0, b0, numh, vmsetm_e##len##xm1(num), num);    \
}

OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint8x16, uint8, uint8, 16, 32, 8, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int8x16, int8, uint8, 16, 32, 8, 8)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint16x8, uint16, uint16, 8, 16, 16, 4)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int16x8, int16, uint16, 8, 16, 16, 4)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(uint32x4, uint32, uint32, 4, 8, 32, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(int32x4, int32, uint32, 4, 8, 32, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float32x4, float32, uint32, 4, 8, 32, 2)
OPENCV_HAL_IMPL_RISCVV_UNPACKS(float64x2, float64, uint64, 2, 4, 64, 1)

inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
    uint64xm1_t mask = (uint64xm1_t){0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_uint8x16(vrgathervv_uint8xm1(a.val, (uint8xm1_t)mask, 16));
}
inline v_int8x16 v_reverse(const v_int8x16 &a)
{
    int64xm1_t mask = (int64xm1_t){0x08090A0B0C0D0E0F, 0x0001020304050607};
    return v_int8x16(vrgathervv_int8xm1_uint8xm1(a.val, (uint8xm1_t)mask, 16));
}

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
    uint64xm1_t mask = (uint64xm1_t){0x0004000500060007, 0x000000100020003};
    return v_uint16x8(vrgathervv_uint16xm1(a.val, (uint16xm1_t)mask, 8));
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{
    int64xm1_t mask = (int64xm1_t){0x0004000500060007, 0x000000100020003};
    return v_int16x8(vrgathervv_int16xm1_uint16xm1(a.val, (uint16xm1_t)mask, 8));
}
inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{
    return v_uint32x4(vrgathervv_uint32xm1(a.val, (uint32xm1_t){3, 2, 1, 0}, 4));
}

inline v_int32x4 v_reverse(const v_int32x4 &a)
{
    return v_int32x4(vrgathervv_int32xm1_uint32xm1(a.val, (uint32xm1_t){3, 2, 1, 0}, 4));
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
template<int i> inline _Tpvec v_broadcast_element(_Tpvec v) { return _Tpvec(vrgathervi_##_Tp##xm1(v.val, i, num)); }

OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint8x16, uint8, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int8x16, int8, 16)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint16x8, uint16, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int16x8, int16, 8)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint32x4, uint32, 4)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int32x4, int32, 4)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_uint64x2, uint64, 2)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_int64x2, int64, 2)
OPENCV_HAL_IMPL_RISCVV_BROADCAST(v_float32x4, float32, 4)


inline void round_mode_set(int a){
    asm volatile(
    "fsrm %0\n\t"
    :
    :"r"(a)
    :);
}

inline int round_mode_read(){
    int a;
    asm volatile(
    "frrm %0\n\t"
    :"=r"(a)
    :
    :);
    return a;
}
inline v_int32x4 v_round(const v_float32x4& a)
{
    round_mode_set(0);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(a.val, a.val, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), a.val, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(a.val, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(a.val, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}
inline v_int32x4 v_floor(const v_float32x4& a)
{
    round_mode_set(2);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(a.val, a.val, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), a.val, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(a.val, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(a.val, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    round_mode_set(3);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(a.val, a.val, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), a.val, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(a.val, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(a.val, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{
    round_mode_set(1);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(a.val, a.val, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), a.val, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(a.val, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(a.val, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}

inline v_int32x4 v_round(const v_float64x2& a)
{
    round_mode_set(0);
    float64xm2_u _val;
    _val.m1[0] = a.val;
    _val.m1[1] = vfmvvf_float64xm1(0, 2);
    int32xm1_t val = vfncvtxfv_int32xm1_float64xm2(_val.v, 4);
    round_mode_set(0);
    return v_int32x4(val);
}
inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    round_mode_set(0);
    float64xm2_u _val;
    _val.m1[0] = a.val;
    _val.m1[1] = b.val;
    int32xm1_t val = vfncvtxfv_int32xm1_float64xm2(_val.v, 4);
    round_mode_set(0);
    return v_int32x4(val);
}
inline v_int32x4 v_floor(const v_float64x2& a)
{
    round_mode_set(2);
    float64xm2_u _val;
    _val.m1[0] = a.val;
    float32xm1_t aval = vfncvtffv_float32xm1_float64xm2(_val.v, 2);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(aval, aval, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), aval, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(aval, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(aval, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    round_mode_set(3);
    float64xm2_u _val;
    _val.m1[0] = a.val;
    float32xm1_t aval = vfncvtffv_float32xm1_float64xm2(_val.v, 2);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(aval, aval, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), aval, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(aval, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(aval, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{
    round_mode_set(1);
    float64xm2_u _val;
    _val.m1[0] = a.val;
    float32xm1_t aval = vfncvtffv_float32xm1_float64xm2(_val.v, 2);
    e32xm1_t nan = vmfordvv_e32xm1_float32xm1(aval, aval, 4);
    int32xm1_t val = vfcvtxfv_mask_int32xm1_float32xm1(vmvvx_int32xm1(0, 4), aval, nan, 4);
    e32xm1_t mask = vmormm_e32xm1(vmfeqvf_e32xm1_float32xm1(aval, INFINITY, 4), vmfeqvf_e32xm1_float32xm1(aval, -INFINITY, 4), 4);
    val = vmergevxm_mask_int32xm1(val, 0, mask, 4);
    round_mode_set(0);
    return v_int32x4(val);
}


inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}
#endif
