// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

#include <riscv_vector.h>

namespace cv { namespace cv_hal_rvv {

enum RVV_LMUL
{
    LMUL_1 = 1,
    LMUL_2 = 2,
    LMUL_4 = 4,
    LMUL_8 = 8,
    LMUL_f2,
    LMUL_f4,
    LMUL_f8,
};

template <typename T, RVV_LMUL LMUL>
struct RVV;

// -------------------------------Supported types--------------------------------

#define HAL_RVV_TYPE_ALIAS(TYPE_NAME, ELEM_TYPE)              \
    using RVV##TYPE_NAME##M1 = struct RVV<ELEM_TYPE, LMUL_1>; \
    using RVV##TYPE_NAME##M2 = struct RVV<ELEM_TYPE, LMUL_2>; \
    using RVV##TYPE_NAME##M4 = struct RVV<ELEM_TYPE, LMUL_4>; \
    using RVV##TYPE_NAME##M8 = struct RVV<ELEM_TYPE, LMUL_8>;

HAL_RVV_TYPE_ALIAS(U8, uint8_t)
using RVVU8MF2 = struct RVV<uint8_t, LMUL_f2>;
using RVVU8MF4 = struct RVV<uint8_t, LMUL_f4>;
using RVVU8MF8 = struct RVV<uint8_t, LMUL_f8>;

HAL_RVV_TYPE_ALIAS(I8, int8_t)
using RVVI8MF2 = struct RVV<int8_t, LMUL_f2>;
using RVVI8MF4 = struct RVV<int8_t, LMUL_f4>;
using RVVI8MF8 = struct RVV<int8_t, LMUL_f8>;

HAL_RVV_TYPE_ALIAS(U16, uint16_t)
using RVVU16MF2 = struct RVV<uint16_t, LMUL_f2>;
using RVVU16MF4 = struct RVV<uint16_t, LMUL_f4>;

HAL_RVV_TYPE_ALIAS(I16, int16_t)
using RVVI16MF2 = struct RVV<int16_t, LMUL_f2>;
using RVVI16MF4 = struct RVV<int16_t, LMUL_f4>;

HAL_RVV_TYPE_ALIAS(U32, uint32_t)
using RVVU32MF2 = struct RVV<uint32_t, LMUL_f2>;

HAL_RVV_TYPE_ALIAS(I32, int32_t)
using RVVI32MF2 = struct RVV<int32_t, LMUL_f2>;

HAL_RVV_TYPE_ALIAS(U64, uint64_t)

HAL_RVV_TYPE_ALIAS(I64, int64_t)

HAL_RVV_TYPE_ALIAS(F32, float)
using RVVF32MF2 = struct RVV<float, LMUL_f2>;

HAL_RVV_TYPE_ALIAS(F64, double)

// -------------------------------Implementation details--------------------------------

#define HAL_RVV_SIZE_RELATED(EEW, TYPE, LMUL, S_OR_F, X_OR_F, IS_U, IS_F)         \
static inline size_t setvlmax() { return __riscv_vsetvlmax_e##EEW##LMUL(); }      \
static inline size_t setvl(size_t vl) { return __riscv_vsetvl_e##EEW##LMUL(vl); } \
static inline VecType vload(const ElemType* ptr, size_t vl) {                     \
    return __riscv_vle##EEW##_v_##TYPE##LMUL(ptr, vl);                            \
}                                                                                 \
static inline VecType vload(BoolType vm, const ElemType* ptr, size_t vl) {        \
    return __riscv_vle##EEW(vm, ptr, vl);                                         \
}                                                                                 \
static inline void vstore(ElemType* ptr, VecType v, size_t vl) {                  \
    __riscv_vse##EEW(ptr, v, vl);                                                 \
}                                                                                 \
static inline void vstore(BoolType vm, ElemType* ptr, VecType v, size_t vl) {     \
    __riscv_vse##EEW(vm, ptr, v, vl);                                             \
}                                                                                 \
static inline VecType vundefined() { return __riscv_vundefined_##TYPE##LMUL(); }  \
static inline VecType vmv(ElemType a, size_t b) {                                 \
    return __riscv_v##IS_F##mv_v_##X_OR_F##_##TYPE##LMUL(a, b);                   \
}                                                                                 \
HAL_RVV_UNSIGNED_ONLY(EEW, TYPE, LMUL)

#define HAL_RVV_SIZE_UNRELATED(S_OR_F, X_OR_F, IS_U, IS_F)                                      \
static inline ElemType vmv_x(VecType vs2) { return __riscv_v##IS_F##mv_##X_OR_F(vs2); }         \
                                                                                                \
static inline BoolType vmlt(VecType vs2, VecType vs1, size_t vl) {                              \
    return __riscv_vm##S_OR_F##lt##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmle(VecType vs2, VecType vs1, size_t vl) {                              \
    return __riscv_vm##S_OR_F##le##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmgt(VecType vs2, VecType vs1, size_t vl) {                              \
    return __riscv_vm##S_OR_F##gt##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmge(VecType vs2, VecType vs1, size_t vl) {                              \
    return __riscv_vm##S_OR_F##ge##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmlt_mu(BoolType vm, BoolType vd, VecType vs2, VecType vs1, size_t vl) { \
    return __riscv_vm##S_OR_F##lt##IS_U##_mu(vm, vd, vs2, vs1, vl);                             \
}                                                                                               \
static inline BoolType vmle_mu(BoolType vm, BoolType vd, VecType vs2, VecType vs1, size_t vl) { \
    return __riscv_vm##S_OR_F##le##IS_U##_mu(vm, vd, vs2, vs1, vl);                             \
}                                                                                               \
static inline BoolType vmgt_mu(BoolType vm, BoolType vd, VecType vs2, VecType vs1, size_t vl) { \
    return __riscv_vm##S_OR_F##gt##IS_U##_mu(vm, vd, vs2, vs1, vl);                             \
}                                                                                               \
static inline BoolType vmge_mu(BoolType vm, BoolType vd, VecType vs2, VecType vs1, size_t vl) { \
    return __riscv_vm##S_OR_F##ge##IS_U##_mu(vm, vd, vs2, vs1, vl);                             \
}                                                                                               \
                                                                                                \
static inline VecType vmin(VecType vs2, VecType vs1, size_t vl) {                               \
    return __riscv_v##IS_F##min##IS_U(vs2, vs1, vl);                                            \
}                                                                                               \
static inline VecType vmax(VecType vs2, VecType vs1, size_t vl) {                               \
    return __riscv_v##IS_F##max##IS_U(vs2, vs1, vl);                                            \
}                                                                                               \
static inline VecType vmin_tu(VecType vd, VecType vs2, VecType vs1, size_t vl) {                \
    return __riscv_v##IS_F##min##IS_U##_tu(vd, vs2, vs1, vl);                                   \
}                                                                                               \
static inline VecType vmax_tu(VecType vd, VecType vs2, VecType vs1, size_t vl) {                \
    return __riscv_v##IS_F##max##IS_U##_tu(vd, vs2, vs1, vl);                                   \
}                                                                                               \
static inline VecType vmin_tumu(BoolType vm, VecType vd, VecType vs2, VecType vs1, size_t vl) { \
    return __riscv_v##IS_F##min##IS_U##_tumu(vm, vd, vs2, vs1, vl);                             \
}                                                                                               \
static inline VecType vmax_tumu(BoolType vm, VecType vd, VecType vs2, VecType vs1, size_t vl) { \
    return __riscv_v##IS_F##max##IS_U##_tumu(vm, vd, vs2, vs1, vl);                             \
}                                                                                               \
                                                                                                \
static inline BaseType vredmin(VecType vs2, BaseType vs1, size_t vl) {                          \
    return __riscv_v##IS_F##redmin##IS_U(vs2, vs1, vl);                                         \
}                                                                                               \
static inline BaseType vredmax(VecType vs2, BaseType vs1, size_t vl) {                          \
    return __riscv_v##IS_F##redmax##IS_U(vs2, vs1, vl);                                         \
}

#define HAL_RVV_DEFINE_ONE(ELEM_TYPE, VEC_TYPE, BOOL_TYPE, BASE_TYPE, LMUL_TYPE, \
                           EEW, TYPE, LMUL, ...)                                 \
    template <> struct RVV<ELEM_TYPE, LMUL_TYPE>                                 \
    {                                                                            \
        using ElemType = ELEM_TYPE;                                              \
        using VecType = VEC_TYPE;                                                \
        using BoolType = BOOL_TYPE;                                              \
        using BaseType = BASE_TYPE;                                              \
                                                                                 \
        HAL_RVV_SIZE_RELATED(EEW, TYPE, LMUL, __VA_ARGS__)                       \
        HAL_RVV_SIZE_UNRELATED(__VA_ARGS__)                                      \
    };

#define HAL_RVV_DEFINE_ALL(ELEM_TYPE, VEC_TYPE, BOOL1, BOOL2, BOOL4, BOOL8,            \
                           EEW, TYPE, ...)                                             \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, v##VEC_TYPE##m1_t, BOOL1, v##VEC_TYPE##m1_t, LMUL_1, \
                       EEW, TYPE, m1, __VA_ARGS__)                                     \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, v##VEC_TYPE##m2_t, BOOL2, v##VEC_TYPE##m1_t, LMUL_2, \
                       EEW, TYPE, m2, __VA_ARGS__)                                     \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, v##VEC_TYPE##m4_t, BOOL4, v##VEC_TYPE##m1_t, LMUL_4, \
                       EEW, TYPE, m4, __VA_ARGS__)                                     \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, v##VEC_TYPE##m8_t, BOOL8, v##VEC_TYPE##m1_t, LMUL_8, \
                       EEW, TYPE, m8, __VA_ARGS__)

#define HAL_RVV_SIGNED_PARAM   s,x, ,
#define HAL_RVV_UNSIGNED_PARAM s,x,u,
#define HAL_RVV_FLOAT_PARAM    f,f, ,f

// Unsigned Integer
#define HAL_RVV_UNSIGNED_ONLY(EEW, TYPE, LMUL) \
    static inline VecType vid(size_t vl) { return __riscv_vid_v_##TYPE##LMUL(vl); }

// LMUL = 1, 2, 4, 8
HAL_RVV_DEFINE_ALL(
     uint8_t,  uint8,  vbool8_t,  vbool4_t,  vbool2_t, vbool1_t,  8,  u8, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ALL(
    uint16_t, uint16, vbool16_t,  vbool8_t,  vbool4_t, vbool2_t, 16, u16, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ALL(
    uint32_t, uint32, vbool32_t, vbool16_t,  vbool8_t, vbool4_t, 32, u32, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ALL(
    uint64_t, uint64, vbool64_t, vbool32_t, vbool16_t, vbool8_t, 64, u64, HAL_RVV_UNSIGNED_PARAM)

// LMUL = f2
HAL_RVV_DEFINE_ONE(
     uint8_t,  vuint8mf2_t, vbool16_t,  vuint8m1_t, LMUL_f2,  8,  u8, mf2, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ONE(
    uint16_t, vuint16mf2_t, vbool32_t, vuint16m1_t, LMUL_f2, 16, u16, mf2, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ONE(
    uint32_t, vuint32mf2_t, vbool64_t, vuint32m1_t, LMUL_f2, 32, u32, mf2, HAL_RVV_UNSIGNED_PARAM)

// LMUL = f4
HAL_RVV_DEFINE_ONE(
     uint8_t,  vuint8mf4_t, vbool32_t,  vuint8m1_t, LMUL_f4,  8,  u8, mf4, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ONE(
    uint16_t, vuint16mf4_t, vbool64_t, vuint16m1_t, LMUL_f4, 16, u16, mf4, HAL_RVV_UNSIGNED_PARAM)

// LMUL = f8
HAL_RVV_DEFINE_ONE(
     uint8_t,  vuint8mf8_t, vbool64_t,  vuint8m1_t, LMUL_f8,  8,  u8, mf8, HAL_RVV_UNSIGNED_PARAM)

#undef HAL_RVV_UNSIGNED_ONLY
// Signed Integer
#define HAL_RVV_UNSIGNED_ONLY(EEW, TYPE, LMUL)

// LMUL = 1, 2, 4, 8
HAL_RVV_DEFINE_ALL(
     int8_t,  int8,  vbool8_t,  vbool4_t,  vbool2_t, vbool1_t,  8,  i8, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ALL(
    int16_t, int16, vbool16_t,  vbool8_t,  vbool4_t, vbool2_t, 16, i16, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ALL(
    int32_t, int32, vbool32_t, vbool16_t,  vbool8_t, vbool4_t, 32, i32, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ALL(
    int64_t, int64, vbool64_t, vbool32_t, vbool16_t, vbool8_t, 64, i64, HAL_RVV_SIGNED_PARAM)

// LMUL = f2
HAL_RVV_DEFINE_ONE(
     int8_t,  vint8mf2_t, vbool16_t,  vint8m1_t, LMUL_f2,  8,  i8, mf2, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ONE(
    int16_t, vint16mf2_t, vbool32_t, vint16m1_t, LMUL_f2, 16, i16, mf2, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ONE(
    int32_t, vint32mf2_t, vbool64_t, vint32m1_t, LMUL_f2, 32, i32, mf2, HAL_RVV_SIGNED_PARAM)

// LMUL = f4
HAL_RVV_DEFINE_ONE(
     int8_t,  vint8mf4_t, vbool32_t,  vint8m1_t, LMUL_f4,  8,  i8, mf4, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ONE(
    int16_t, vint16mf4_t, vbool64_t, vint16m1_t, LMUL_f4, 16, i16, mf4, HAL_RVV_SIGNED_PARAM)

// LMUL = f8
HAL_RVV_DEFINE_ONE(
     int8_t,  vint8mf8_t, vbool64_t,  vint8m1_t, LMUL_f8,  8,  i8, mf8, HAL_RVV_SIGNED_PARAM)

// Float
// LMUL = 1, 2, 4, 8
HAL_RVV_DEFINE_ALL(
     float, float32, vbool32_t, vbool16_t,  vbool8_t, vbool4_t, 32, f32, HAL_RVV_FLOAT_PARAM)
HAL_RVV_DEFINE_ALL(
    double, float64, vbool64_t, vbool32_t, vbool16_t, vbool8_t, 64, f64, HAL_RVV_FLOAT_PARAM)

// LMUL = f2
HAL_RVV_DEFINE_ONE(
    float, vfloat32mf2_t, vbool64_t, vfloat32m1_t, LMUL_f2, 32, f32, mf2, HAL_RVV_FLOAT_PARAM)

#undef HAL_RVV_UNSIGNED_ONLY
#undef HAL_RVV_DEFINE_ALL
#undef HAL_RVV_DEFINE_ONE
#undef HAL_RVV_SIZE_UNRELATED
#undef HAL_RVV_SIZE_RELATED

}}  // namespace cv::cv_hal_rvv
