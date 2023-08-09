// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_HAL_INTRIN_RVV_COMPAT_OVERLOAD_HPP
#define OPENCV_HAL_INTRIN_RVV_COMPAT_OVERLOAD_HPP

// This file requires VTraits to be defined for vector types

#define OPENCV_HAL_IMPL_RVV_FUN_AND(REG, SUF) \
inline static REG vand(const REG & op1, const REG & op2, size_t vl) \
{ \
    return vand_vv_##SUF(op1, op2, vl); \
}

OPENCV_HAL_IMPL_RVV_FUN_AND(vint8m1_t, i8m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vuint8m1_t, u8m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vint16m1_t, i16m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vuint16m1_t, u16m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vint32m1_t, i32m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vuint32m1_t, u32m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vint64m1_t, i64m1)
OPENCV_HAL_IMPL_RVV_FUN_AND(vuint64m1_t, u64m1)

#define OPENCV_HAL_IMPL_RVV_FUN_LOXEI(REG, SUF, INDX, ISUF) \
inline static REG vloxe##ISUF(const VTraits<REG>::lane_type *base, INDX bindex, size_t vl) \
{ \
    return vloxe##ISUF##_v_##SUF(base, bindex, vl); \
}

OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint8m1_t, i8m1, vuint8m1_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint8m2_t, i8m2, vuint8m2_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint8m4_t, i8m4, vuint8m4_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint8m8_t, i8m8, vuint8m8_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint8m1_t, i8m1, vuint32m4_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint8m2_t, i8m2, vuint32m8_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint16m1_t, i16m1, vuint32m2_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint32m1_t, i32m1, vuint32m1_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint32m2_t, i32m2, vuint32m2_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint32m4_t, i32m4, vuint32m4_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint32m8_t, i32m8, vuint32m8_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vint64m1_t, i64m1, vuint32mf2_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vuint8m1_t, u8m1, vuint8m1_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vuint8m2_t, u8m2, vuint8m2_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vuint8m4_t, u8m4, vuint8m4_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vuint8m8_t, u8m8, vuint8m8_t, i8)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vfloat32m1_t, f32m1, vuint32m1_t, i32)
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vuint32m1_t, u32m1, vuint32m1_t, i32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_FUN_LOXEI(vfloat64m1_t, f64m1, vuint32mf2_t, i32)
#endif

#define OPENCV_HAL_IMPL_RVV_FUN_MUL(REG, SUF) \
inline static REG##m1_t vmul(const REG##m1_t & op1, const REG##m1_t & op2, size_t vl) \
{ \
    return vmul_vv_##SUF##m1(op1, op2, vl); \
} \
inline static REG##m1_t vmul(const REG##m1_t & op1, VTraits<REG##m1_t>::lane_type op2, size_t vl) \
{ \
    return vmul_vx_##SUF##m1(op1, op2, vl); \
} \
inline static REG##m2_t vmul(const REG##m2_t & op1, const REG##m2_t & op2, size_t vl) \
{ \
    return vmul_vv_##SUF##m2(op1, op2, vl); \
} \
inline static REG##m2_t vmul(const REG##m2_t & op1, VTraits<REG##m2_t>::lane_type op2, size_t vl) \
{ \
    return vmul_vx_##SUF##m2(op1, op2, vl); \
} \
inline static REG##m4_t vmul(const REG##m4_t & op1, const REG##m4_t & op2, size_t vl) \
{ \
    return vmul_vv_##SUF##m4(op1, op2, vl); \
} \
inline static REG##m4_t vmul(const REG##m4_t & op1, VTraits<REG##m4_t>::lane_type op2, size_t vl) \
{ \
    return vmul_vx_##SUF##m4(op1, op2, vl); \
} \
inline static REG##m8_t vmul(const REG##m8_t & op1, const REG##m8_t & op2, size_t vl) \
{ \
    return vmul_vv_##SUF##m8(op1, op2, vl); \
} \
inline static REG##m8_t vmul(const REG##m8_t & op1, VTraits<REG##m8_t>::lane_type op2, size_t vl) \
{ \
    return vmul_vx_##SUF##m8(op1, op2, vl); \
}

OPENCV_HAL_IMPL_RVV_FUN_MUL(vint8, i8)
OPENCV_HAL_IMPL_RVV_FUN_MUL(vuint8, u8)
OPENCV_HAL_IMPL_RVV_FUN_MUL(vint16, i16)
OPENCV_HAL_IMPL_RVV_FUN_MUL(vuint16, u16)
OPENCV_HAL_IMPL_RVV_FUN_MUL(vint32, i32)
OPENCV_HAL_IMPL_RVV_FUN_MUL(vuint32, u32)

#define OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(REG1, SUF1, REG2, SUF2) \
inline static REG1##m1_t vreinterpret_##SUF1##m1(const REG2##m1_t & src) \
{\
    return vreinterpret_v_##SUF2##m1_##SUF1##m1(src); \
} \
inline static REG1##m2_t vreinterpret_##SUF1##m2(const REG2##m2_t & src) \
{\
    return vreinterpret_v_##SUF2##m2_##SUF1##m2(src); \
} \
inline static REG1##m4_t vreinterpret_##SUF1##m4(const REG2##m4_t & src) \
{\
    return vreinterpret_v_##SUF2##m4_##SUF1##m4(src); \
} \
inline static REG1##m8_t vreinterpret_##SUF1##m8(const REG2##m8_t & src) \
{\
    return vreinterpret_v_##SUF2##m8_##SUF1##m8(src); \
}

OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vint8, i8, vuint8, u8)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vint16, i16, vuint16, u16)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vint32, i32, vuint32, u32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vfloat32, f32, vuint32, u32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vfloat32, f32, vint32, i32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint32, u32, vfloat32, f32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vint32, i32, vfloat32, f32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint8, u8, vint8, i8)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint8, u8, vuint16, u16)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint8, u8, vuint32, u32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint8, u8, vuint64, u64)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint16, u16, vint16, i16)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint16, u16, vuint8, u8)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint16, u16, vuint32, u32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint16, u16, vuint64, u64)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint32, u32, vint32, i32)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint32, u32, vuint8, u8)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint32, u32, vuint16, u16)
OPENCV_HAL_IMPL_RVV_FUN_REINTERPRET(vuint32, u32, vuint64, u64)

#define OPENCV_HAL_IMPL_RVV_FUN_STORE(REG, SUF, SZ) \
inline static void vse##SZ(VTraits<REG>::lane_type *base, REG value, size_t vl) \
{ \
    return vse##SZ##_v_##SUF##m1(base, value, vl); \
}

OPENCV_HAL_IMPL_RVV_FUN_STORE(v_uint8, u8, 8)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_int8, i8, 8)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_uint16, u16, 16)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_int16, i16, 16)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_uint32, u32, 32)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_int32, i32, 32)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_uint64, u64, 64)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_int64, i64, 64)
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_float32, f32, 32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_FUN_STORE(v_float64, f64, 64)
#endif

#define OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(REG, SUF) \
inline static VTraits<REG>::lane_type vmv_x(const REG & reg) \
{\
    return vmv_x_s_##SUF##m1_##SUF(reg); \
}
#define OPENCV_HAL_IMPL_RVV_FUN_EXTRACT_F(REG, SUF) \
inline static VTraits<REG>::lane_type vfmv_f(const REG & reg) \
{\
    return vfmv_f_s_##SUF##m1_##SUF(reg); \
}

OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_uint8, u8)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_int8, i8)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_uint16, u16)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_int16, i16)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_int32, i32)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_uint64, u64)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT(v_int64, i64)
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT_F(v_float32, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_FUN_EXTRACT_F(v_float64, f64)
#endif

#define OPENCV_HAL_IMPL_RVV_FUN_SLIDE(REG, SUF) \
inline static REG vslidedown(const REG & dst, const REG & src, size_t offset, size_t vl) \
{ \
    return vslidedown_vx_##SUF##m1(dst, src, offset, vl); \
} \
inline static REG vslideup(const REG & dst, const REG & src, size_t offset, size_t vl) \
{ \
    return vslideup_vx_##SUF##m1(dst, src, offset, vl); \
}

OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_uint8, u8)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_int8, i8)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_uint16, u16)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_int16, i16)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_int32, i32)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_float32, f32)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_uint64, u64)
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_int64, i64)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_FUN_SLIDE(v_float64, f64)
#endif

inline static vuint32mf2_t vmul(const vuint32mf2_t & op1, uint32_t op2, size_t vl)
{
    return vmul_vx_u32mf2(op1, op2, vl);
}

inline static vuint32mf2_t vreinterpret_u32mf2(vint32mf2_t val)
{
    return vreinterpret_v_i32mf2_u32mf2(val);
}

#endif //OPENCV_HAL_INTRIN_RVV_COMPAT_OVERLOAD_HPP
