// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2025, Institute of Software, Chinese Academy of Sciences.

#ifndef OPENCV_HAL_RVV_TYPES_HPP_INCLUDED
#define OPENCV_HAL_RVV_TYPES_HPP_INCLUDED

#include <riscv_vector.h>
#include <type_traits>

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

using RVV_U8M1 = struct RVV<uint8_t, LMUL_1>;
using RVV_U8M2 = struct RVV<uint8_t, LMUL_2>;
using RVV_U8M4 = struct RVV<uint8_t, LMUL_4>;
using RVV_U8M8 = struct RVV<uint8_t, LMUL_8>;
using RVV_U8MF2 = struct RVV<uint8_t, LMUL_f2>;
using RVV_U8MF4 = struct RVV<uint8_t, LMUL_f4>;
using RVV_U8MF8 = struct RVV<uint8_t, LMUL_f8>;

using RVV_I8M1 = struct RVV<int8_t, LMUL_1>;
using RVV_I8M2 = struct RVV<int8_t, LMUL_2>;
using RVV_I8M4 = struct RVV<int8_t, LMUL_4>;
using RVV_I8M8 = struct RVV<int8_t, LMUL_8>;
using RVV_I8MF2 = struct RVV<int8_t, LMUL_f2>;
using RVV_I8MF4 = struct RVV<int8_t, LMUL_f4>;
using RVV_I8MF8 = struct RVV<int8_t, LMUL_f8>;

using RVV_U16M1 = struct RVV<uint16_t, LMUL_1>;
using RVV_U16M2 = struct RVV<uint16_t, LMUL_2>;
using RVV_U16M4 = struct RVV<uint16_t, LMUL_4>;
using RVV_U16M8 = struct RVV<uint16_t, LMUL_8>;
using RVV_U16MF2 = struct RVV<uint16_t, LMUL_f2>;
using RVV_U16MF4 = struct RVV<uint16_t, LMUL_f4>;

using RVV_I16M1 = struct RVV<int16_t, LMUL_1>;
using RVV_I16M2 = struct RVV<int16_t, LMUL_2>;
using RVV_I16M4 = struct RVV<int16_t, LMUL_4>;
using RVV_I16M8 = struct RVV<int16_t, LMUL_8>;
using RVV_I16MF2 = struct RVV<int16_t, LMUL_f2>;
using RVV_I16MF4 = struct RVV<int16_t, LMUL_f4>;

using RVV_U32M1 = struct RVV<uint32_t, LMUL_1>;
using RVV_U32M2 = struct RVV<uint32_t, LMUL_2>;
using RVV_U32M4 = struct RVV<uint32_t, LMUL_4>;
using RVV_U32M8 = struct RVV<uint32_t, LMUL_8>;
using RVV_U32MF2 = struct RVV<uint32_t, LMUL_f2>;

using RVV_I32M1 = struct RVV<int32_t, LMUL_1>;
using RVV_I32M2 = struct RVV<int32_t, LMUL_2>;
using RVV_I32M4 = struct RVV<int32_t, LMUL_4>;
using RVV_I32M8 = struct RVV<int32_t, LMUL_8>;
using RVV_I32MF2 = struct RVV<int32_t, LMUL_f2>;

using RVV_U64M1 = struct RVV<uint64_t, LMUL_1>;
using RVV_U64M2 = struct RVV<uint64_t, LMUL_2>;
using RVV_U64M4 = struct RVV<uint64_t, LMUL_4>;
using RVV_U64M8 = struct RVV<uint64_t, LMUL_8>;

using RVV_I64M1 = struct RVV<int64_t, LMUL_1>;
using RVV_I64M2 = struct RVV<int64_t, LMUL_2>;
using RVV_I64M4 = struct RVV<int64_t, LMUL_4>;
using RVV_I64M8 = struct RVV<int64_t, LMUL_8>;

using RVV_F32M1 = struct RVV<float, LMUL_1>;
using RVV_F32M2 = struct RVV<float, LMUL_2>;
using RVV_F32M4 = struct RVV<float, LMUL_4>;
using RVV_F32M8 = struct RVV<float, LMUL_8>;
using RVV_F32MF2 = struct RVV<float, LMUL_f2>;

using RVV_F64M1 = struct RVV<double, LMUL_1>;
using RVV_F64M2 = struct RVV<double, LMUL_2>;
using RVV_F64M4 = struct RVV<double, LMUL_4>;
using RVV_F64M8 = struct RVV<double, LMUL_8>;

template <typename Dst_T, typename RVV_T>
using RVV_SameLen =
    RVV<Dst_T, RVV_LMUL(static_cast<int>((RVV_T::lmul <= 8 ? RVV_T::lmul * static_cast<float>(sizeof(Dst_T)) : RVV_T::lmul == 9 ? static_cast<float>(sizeof(Dst_T)) / 2 : RVV_T::lmul == 10 ? static_cast<float>(sizeof(Dst_T)) / 4 : static_cast<float>(sizeof(Dst_T)) / 8) / sizeof(typename RVV_T::ElemType) == 0.5   ? 9  : \
                                         (RVV_T::lmul <= 8 ? RVV_T::lmul * static_cast<float>(sizeof(Dst_T)) : RVV_T::lmul == 9 ? static_cast<float>(sizeof(Dst_T)) / 2 : RVV_T::lmul == 10 ? static_cast<float>(sizeof(Dst_T)) / 4 : static_cast<float>(sizeof(Dst_T)) / 8) / sizeof(typename RVV_T::ElemType) == 0.25  ? 10 : \
                                         (RVV_T::lmul <= 8 ? RVV_T::lmul * static_cast<float>(sizeof(Dst_T)) : RVV_T::lmul == 9 ? static_cast<float>(sizeof(Dst_T)) / 2 : RVV_T::lmul == 10 ? static_cast<float>(sizeof(Dst_T)) / 4 : static_cast<float>(sizeof(Dst_T)) / 8) / sizeof(typename RVV_T::ElemType) == 0.125 ? 11 : \
                                         (RVV_T::lmul <= 8 ? RVV_T::lmul * static_cast<float>(sizeof(Dst_T)) : RVV_T::lmul == 9 ? static_cast<float>(sizeof(Dst_T)) / 2 : RVV_T::lmul == 10 ? static_cast<float>(sizeof(Dst_T)) / 4 : static_cast<float>(sizeof(Dst_T)) / 8) / sizeof(typename RVV_T::ElemType)))>;

template <size_t DstSize> struct RVV_ToIntHelper;
template <size_t DstSize> struct RVV_ToUintHelper;
template <size_t DstSize> struct RVV_ToFloatHelper;

template <typename RVV_T>
using RVV_ToInt =
    RVV<typename RVV_ToIntHelper<sizeof(typename RVV_T::ElemType)>::type, RVV_T::lmul>;

template <typename RVV_T>
using RVV_ToUint =
    RVV<typename RVV_ToUintHelper<sizeof(typename RVV_T::ElemType)>::type, RVV_T::lmul>;

template <typename RVV_T>
using RVV_ToFloat =
    RVV<typename RVV_ToFloatHelper<sizeof(typename RVV_T::ElemType)>::type, RVV_T::lmul>;

template <typename RVV_T>
using RVV_BaseType = RVV<typename RVV_T::ElemType, LMUL_1>;

// -------------------------------Supported operations--------------------------------

#define HAL_RVV_SIZE_RELATED(EEW, TYPE, LMUL, S_OR_F, X_OR_F, IS_U, IS_F, IS_O)                      \
static inline size_t setvlmax() { return __riscv_vsetvlmax_e##EEW##LMUL(); }                         \
static inline size_t setvl(size_t vl) { return __riscv_vsetvl_e##EEW##LMUL(vl); }                    \
static inline VecType vload(const ElemType* ptr, size_t vl) {                                        \
    return __riscv_vle##EEW##_v_##TYPE##LMUL(ptr, vl);                                               \
}                                                                                                    \
static inline VecType vload(BoolType vm, const ElemType* ptr, size_t vl) {                           \
    return __riscv_vle##EEW(vm, ptr, vl);                                                            \
}                                                                                                    \
static inline VecType vload_stride(const ElemType* ptr, ptrdiff_t unit, size_t vl) {                 \
    return __riscv_vlse##EEW##_v_##TYPE##LMUL(ptr, unit, vl);                                        \
}                                                                                                    \
static inline VecType vload_stride(BoolType vm, const ElemType* ptr, ptrdiff_t unit, size_t vl) {    \
    return __riscv_vlse##EEW(vm, ptr, unit, vl);                                                     \
}                                                                                                    \
static inline void vstore(ElemType* ptr, VecType v, size_t vl) {                                     \
    __riscv_vse##EEW(ptr, v, vl);                                                                    \
}                                                                                                    \
static inline void vstore(BoolType vm, ElemType* ptr, VecType v, size_t vl) {                        \
    __riscv_vse##EEW(vm, ptr, v, vl);                                                                \
}                                                                                                    \
static inline void vstore_stride(ElemType* ptr, ptrdiff_t unit, VecType v, size_t vl) {              \
    __riscv_vsse##EEW(ptr, unit, v, vl);                                                             \
}                                                                                                    \
static inline void vstore_stride(BoolType vm, ElemType* ptr, ptrdiff_t unit, VecType v, size_t vl) { \
    __riscv_vsse##EEW(vm, ptr, unit, v, vl);                                                         \
}                                                                                                    \
static inline VecType vundefined() { return __riscv_vundefined_##TYPE##LMUL(); }                     \
static inline VecType vmv(ElemType a, size_t vl) {                                                   \
    return __riscv_v##IS_F##mv_v_##X_OR_F##_##TYPE##LMUL(a, vl);                                     \
}                                                                                                    \
static inline VecType vmv_s(ElemType a, size_t vl) {                                                 \
    return __riscv_v##IS_F##mv_s_##X_OR_F##_##TYPE##LMUL(a, vl);                                     \
}                                                                                                    \
static inline VecType vslideup(VecType vs2, VecType vs1, size_t n, size_t vl) {                      \
    return __riscv_vslideup_vx_##TYPE##LMUL(vs2, vs1, n, vl);                                        \
}                                                                                                    \
static inline VecType vslidedown(VecType vs, size_t n, size_t vl) {                                  \
    return __riscv_vslidedown_vx_##TYPE##LMUL(vs, n, vl);                                            \
}                                                                                                    \
HAL_RVV_SIZE_RELATED_CUSTOM(EEW, TYPE, LMUL)

#define HAL_RVV_SIZE_UNRELATED(S_OR_F, X_OR_F, IS_U, IS_F, IS_O)                                \
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
static inline BoolType vmle(VecType vs2, ElemType vs1, size_t vl) {                             \
    return __riscv_vm##S_OR_F##le##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmgt(VecType vs2, ElemType vs1, size_t vl) {                             \
    return __riscv_vm##S_OR_F##gt##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmge(VecType vs2, VecType vs1, size_t vl) {                              \
    return __riscv_vm##S_OR_F##ge##IS_U(vs2, vs1, vl);                                          \
}                                                                                               \
static inline BoolType vmeq(VecType vs2, ElemType vs1, size_t vl) {                             \
    return __riscv_vm##S_OR_F##eq(vs2, vs1, vl);                                                \
}                                                                                               \
static inline BoolType vmne(VecType vs2, ElemType vs1, size_t vl) {                             \
    return __riscv_vm##S_OR_F##ne(vs2, vs1, vl);                                                \
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
static inline VecType vadd(VecType vs2, VecType vs1, size_t vl) {                               \
    return __riscv_v##IS_F##add(vs2, vs1, vl);                                                  \
}                                                                                               \
static inline VecType vsub(VecType vs2, VecType vs1, size_t vl) {                               \
    return __riscv_v##IS_F##sub(vs2, vs1, vl);                                                  \
}                                                                                               \
static inline VecType vadd_tu(VecType vd, VecType vs2, VecType vs1, size_t vl) {                \
    return __riscv_v##IS_F##add_tu(vd, vs2, vs1, vl);                                           \
}                                                                                               \
static inline VecType vsub_tu(VecType vd, VecType vs2, VecType vs1, size_t vl) {                \
    return __riscv_v##IS_F##sub_tu(vd, vs2, vs1, vl);                                           \
}                                                                                               \
static inline VecType vmul(VecType vs2, VecType vs1, size_t vl) {                               \
    return __riscv_v##IS_F##mul(vs2, vs1, vl);                                                  \
}                                                                                               \
                                                                                                \
static inline VecType vslide1down(VecType vs2, ElemType vs1, size_t vl) {                       \
    return __riscv_v##IS_F##slide1down(vs2, vs1, vl);                                           \
}                                                                                               \
static inline VecType vslide1up(VecType vs2, ElemType vs1, size_t vl) {                         \
    return __riscv_v##IS_F##slide1up(vs2, vs1, vl);                                             \
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
}                                                                                               \
static inline BaseType vredsum(VecType vs2, BaseType vs1, size_t vl) {                          \
    return __riscv_v##IS_F##red##IS_O##sum(vs2, vs1, vl);                                       \
}

#define HAL_RVV_BOOL_TYPE(S_OR_F, X_OR_F, IS_U, IS_F, IS_O) \
    decltype(__riscv_vm##S_OR_F##eq(std::declval<VecType>(), std::declval<VecType>(), 0))

#define HAL_RVV_DEFINE_ONE(ELEM_TYPE, VEC_TYPE, LMUL_TYPE, \
                           EEW, TYPE, LMUL, ...)           \
    template <> struct RVV<ELEM_TYPE, LMUL_TYPE>           \
    {                                                      \
        using ElemType = ELEM_TYPE;                        \
        using VecType = v##VEC_TYPE##LMUL##_t;             \
        using BoolType = HAL_RVV_BOOL_TYPE(__VA_ARGS__);   \
        using BaseType = v##VEC_TYPE##m1_t;                \
                                                           \
        static constexpr RVV_LMUL lmul = LMUL_TYPE;        \
                                                           \
        HAL_RVV_SIZE_RELATED(EEW, TYPE, LMUL, __VA_ARGS__) \
        HAL_RVV_SIZE_UNRELATED(__VA_ARGS__)                \
                                                           \
        template <typename FROM>                           \
        inline static VecType cast(FROM v, size_t vl);     \
        template <typename FROM>                           \
        inline static VecType reinterpret(FROM v);         \
    };                                                     \
                                                           \
    template <>                                            \
    inline RVV<ELEM_TYPE, LMUL_TYPE>::VecType              \
    RVV<ELEM_TYPE, LMUL_TYPE>::cast(                       \
        RVV<ELEM_TYPE, LMUL_TYPE>::VecType v,              \
        [[maybe_unused]] size_t vl                         \
    )                                                      \
    {                                                      \
        return v;                                          \
    }

// -------------------------------Define all types--------------------------------

#define HAL_RVV_DEFINE_ALL(ELEM_TYPE, VEC_TYPE,     \
                           EEW, TYPE, ...)          \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, VEC_TYPE, LMUL_1, \
                       EEW, TYPE, m1, __VA_ARGS__)  \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, VEC_TYPE, LMUL_2, \
                       EEW, TYPE, m2, __VA_ARGS__)  \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, VEC_TYPE, LMUL_4, \
                       EEW, TYPE, m4, __VA_ARGS__)  \
    HAL_RVV_DEFINE_ONE(ELEM_TYPE, VEC_TYPE, LMUL_8, \
                       EEW, TYPE, m8, __VA_ARGS__)

#define HAL_RVV_SIGNED_PARAM   s,x, , ,
#define HAL_RVV_UNSIGNED_PARAM s,x,u, ,
#define HAL_RVV_FLOAT_PARAM    f,f, ,f,o

// -------------------------------Define Unsigned Integer--------------------------------

#define HAL_RVV_SIZE_RELATED_CUSTOM(EEW, TYPE, LMUL) \
    static inline VecType vid(size_t vl) { return __riscv_vid_v_##TYPE##LMUL(vl); }

// LMUL = 1, 2, 4, 8
HAL_RVV_DEFINE_ALL( uint8_t,  uint8,  8,  u8, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ALL(uint16_t, uint16, 16, u16, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ALL(uint32_t, uint32, 32, u32, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ALL(uint64_t, uint64, 64, u64, HAL_RVV_UNSIGNED_PARAM)

// LMUL = f2
HAL_RVV_DEFINE_ONE( uint8_t,  uint8, LMUL_f2,  8,  u8, mf2, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ONE(uint16_t, uint16, LMUL_f2, 16, u16, mf2, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ONE(uint32_t, uint32, LMUL_f2, 32, u32, mf2, HAL_RVV_UNSIGNED_PARAM)

// LMUL = f4
HAL_RVV_DEFINE_ONE( uint8_t,  uint8, LMUL_f4,  8,  u8, mf4, HAL_RVV_UNSIGNED_PARAM)
HAL_RVV_DEFINE_ONE(uint16_t, uint16, LMUL_f4, 16, u16, mf4, HAL_RVV_UNSIGNED_PARAM)

// LMUL = f8
HAL_RVV_DEFINE_ONE( uint8_t,  uint8, LMUL_f8,  8,  u8, mf8, HAL_RVV_UNSIGNED_PARAM)

#undef HAL_RVV_SIZE_RELATED_CUSTOM

// -------------------------------Define Signed Integer--------------------------------

#define HAL_RVV_SIZE_RELATED_CUSTOM(EEW, TYPE, LMUL)

// LMUL = 1, 2, 4, 8
HAL_RVV_DEFINE_ALL( int8_t,  int8,  8,  i8, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ALL(int16_t, int16, 16, i16, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ALL(int32_t, int32, 32, i32, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ALL(int64_t, int64, 64, i64, HAL_RVV_SIGNED_PARAM)

// LMUL = f2
HAL_RVV_DEFINE_ONE( int8_t,  int8, LMUL_f2,  8,  i8, mf2, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ONE(int16_t, int16, LMUL_f2, 16, i16, mf2, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ONE(int32_t, int32, LMUL_f2, 32, i32, mf2, HAL_RVV_SIGNED_PARAM)

// LMUL = f4
HAL_RVV_DEFINE_ONE( int8_t,  int8, LMUL_f4,  8,  i8, mf4, HAL_RVV_SIGNED_PARAM)
HAL_RVV_DEFINE_ONE(int16_t, int16, LMUL_f4, 16, i16, mf4, HAL_RVV_SIGNED_PARAM)

// LMUL = f8
HAL_RVV_DEFINE_ONE( int8_t,  int8, LMUL_f8,  8,  i8, mf8, HAL_RVV_SIGNED_PARAM)

// -------------------------------Define Floating Point--------------------------------

// LMUL = 1, 2, 4, 8
HAL_RVV_DEFINE_ALL( float, float32, 32, f32, HAL_RVV_FLOAT_PARAM)
HAL_RVV_DEFINE_ALL(double, float64, 64, f64, HAL_RVV_FLOAT_PARAM)

// LMUL = f2
HAL_RVV_DEFINE_ONE( float, float32, LMUL_f2, 32, f32, mf2, HAL_RVV_FLOAT_PARAM)

#undef HAL_RVV_SIZE_RELATED_CUSTOM
#undef HAL_RVV_DEFINE_ALL
#undef HAL_RVV_DEFINE_ONE
#undef HAL_RVV_BOOL_TYPE
#undef HAL_RVV_SIZE_UNRELATED
#undef HAL_RVV_SIZE_RELATED

// -------------------------------Define cast--------------------------------

template <> struct RVV_ToIntHelper<1> {using type = int8_t;};
template <> struct RVV_ToIntHelper<2> {using type = int16_t;};
template <> struct RVV_ToIntHelper<4> {using type = int32_t;};
template <> struct RVV_ToIntHelper<8> {using type = int64_t;};

template <> struct RVV_ToUintHelper<1> {using type = uint8_t;};
template <> struct RVV_ToUintHelper<2> {using type = uint16_t;};
template <> struct RVV_ToUintHelper<4> {using type = uint32_t;};
template <> struct RVV_ToUintHelper<8> {using type = uint64_t;};

template <> struct RVV_ToFloatHelper<2> {using type = _Float16;};
template <> struct RVV_ToFloatHelper<4> {using type = float;};
template <> struct RVV_ToFloatHelper<8> {using type = double;};

#define HAL_RVV_CVT(ONE, TWO)                                                                   \
    template <>                                                                                 \
    inline ONE::VecType ONE::cast(TWO::VecType v, size_t vl) { return __riscv_vncvt_x(v, vl); } \
    template <>                                                                                 \
    inline TWO::VecType TWO::cast(ONE::VecType v, size_t vl) { return __riscv_vsext_vf2(v, vl); }

HAL_RVV_CVT(RVV_I8M4, RVV_I16M8)
HAL_RVV_CVT(RVV_I8M2, RVV_I16M4)
HAL_RVV_CVT(RVV_I8M1, RVV_I16M2)
HAL_RVV_CVT(RVV_I8MF2, RVV_I16M1)
HAL_RVV_CVT(RVV_I8MF4, RVV_I16MF2)
HAL_RVV_CVT(RVV_I8MF8, RVV_I16MF4)

HAL_RVV_CVT(RVV_I16M4, RVV_I32M8)
HAL_RVV_CVT(RVV_I16M2, RVV_I32M4)
HAL_RVV_CVT(RVV_I16M1, RVV_I32M2)
HAL_RVV_CVT(RVV_I16MF2, RVV_I32M1)
HAL_RVV_CVT(RVV_I16MF4, RVV_I32MF2)

HAL_RVV_CVT(RVV_I32M4, RVV_I64M8)
HAL_RVV_CVT(RVV_I32M2, RVV_I64M4)
HAL_RVV_CVT(RVV_I32M1, RVV_I64M2)
HAL_RVV_CVT(RVV_I32MF2, RVV_I64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(ONE, TWO)                                                                   \
    template <>                                                                                 \
    inline ONE::VecType ONE::cast(TWO::VecType v, size_t vl) { return __riscv_vncvt_x(v, vl); } \
    template <>                                                                                 \
    inline TWO::VecType TWO::cast(ONE::VecType v, size_t vl) { return __riscv_vzext_vf2(v, vl); }

HAL_RVV_CVT(RVV_U8M4, RVV_U16M8)
HAL_RVV_CVT(RVV_U8M2, RVV_U16M4)
HAL_RVV_CVT(RVV_U8M1, RVV_U16M2)
HAL_RVV_CVT(RVV_U8MF2, RVV_U16M1)
HAL_RVV_CVT(RVV_U8MF4, RVV_U16MF2)
HAL_RVV_CVT(RVV_U8MF8, RVV_U16MF4)

HAL_RVV_CVT(RVV_U16M4, RVV_U32M8)
HAL_RVV_CVT(RVV_U16M2, RVV_U32M4)
HAL_RVV_CVT(RVV_U16M1, RVV_U32M2)
HAL_RVV_CVT(RVV_U16MF2, RVV_U32M1)
HAL_RVV_CVT(RVV_U16MF4, RVV_U32MF2)

HAL_RVV_CVT(RVV_U32M4, RVV_U64M8)
HAL_RVV_CVT(RVV_U32M2, RVV_U64M4)
HAL_RVV_CVT(RVV_U32M1, RVV_U64M2)
HAL_RVV_CVT(RVV_U32MF2, RVV_U64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(ONE, FOUR)                                                                      \
    template <>                                                                                     \
    inline FOUR::VecType FOUR::cast(ONE::VecType v, size_t vl) { return __riscv_vsext_vf4(v, vl); } \
    template <>                                                                                     \
    inline ONE::VecType ONE::cast(FOUR::VecType v, size_t vl) {                                     \
        return __riscv_vncvt_x(__riscv_vncvt_x(v, vl), vl);                                         \
    }

HAL_RVV_CVT(RVV_I8M2, RVV_I32M8)
HAL_RVV_CVT(RVV_I8M1, RVV_I32M4)
HAL_RVV_CVT(RVV_I8MF2, RVV_I32M2)
HAL_RVV_CVT(RVV_I8MF4, RVV_I32M1)
HAL_RVV_CVT(RVV_I8MF8, RVV_I32MF2)

HAL_RVV_CVT(RVV_I16M2, RVV_I64M8)
HAL_RVV_CVT(RVV_I16M1, RVV_I64M4)
HAL_RVV_CVT(RVV_I16MF2, RVV_I64M2)
HAL_RVV_CVT(RVV_I16MF4, RVV_I64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(ONE, FOUR)                                                                      \
    template <>                                                                                     \
    inline FOUR::VecType FOUR::cast(ONE::VecType v, size_t vl) { return __riscv_vzext_vf4(v, vl); } \
    template <>                                                                                     \
    inline ONE::VecType ONE::cast(FOUR::VecType v, size_t vl) {                                     \
        return __riscv_vncvt_x(__riscv_vncvt_x(v, vl), vl);                                         \
    }

HAL_RVV_CVT(RVV_U8M2, RVV_U32M8)
HAL_RVV_CVT(RVV_U8M1, RVV_U32M4)
HAL_RVV_CVT(RVV_U8MF2, RVV_U32M2)
HAL_RVV_CVT(RVV_U8MF4, RVV_U32M1)
HAL_RVV_CVT(RVV_U8MF8, RVV_U32MF2)

HAL_RVV_CVT(RVV_U16M2, RVV_U64M8)
HAL_RVV_CVT(RVV_U16M1, RVV_U64M4)
HAL_RVV_CVT(RVV_U16MF2, RVV_U64M2)
HAL_RVV_CVT(RVV_U16MF4, RVV_U64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(ONE, EIGHT)                                                                       \
    template <>                                                                                       \
    inline EIGHT::VecType EIGHT::cast(ONE::VecType v, size_t vl) { return __riscv_vsext_vf8(v, vl); } \
    template <>                                                                                       \
    inline ONE::VecType ONE::cast(EIGHT::VecType v, size_t vl) {                                      \
        return __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vncvt_x(v ,vl), vl), vl);                      \
    }

HAL_RVV_CVT(RVV_I8M1, RVV_I64M8)
HAL_RVV_CVT(RVV_I8MF2, RVV_I64M4)
HAL_RVV_CVT(RVV_I8MF4, RVV_I64M2)
HAL_RVV_CVT(RVV_I8MF8, RVV_I64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(ONE, EIGHT)                                                                       \
    template <>                                                                                       \
    inline EIGHT::VecType EIGHT::cast(ONE::VecType v, size_t vl) { return __riscv_vzext_vf8(v, vl); } \
    template <>                                                                                       \
    inline ONE::VecType ONE::cast(EIGHT::VecType v, size_t vl) {                                      \
        return __riscv_vncvt_x(__riscv_vncvt_x(__riscv_vncvt_x(v ,vl), vl), vl);                      \
    }

HAL_RVV_CVT(RVV_U8M1, RVV_U64M8)
HAL_RVV_CVT(RVV_U8MF2, RVV_U64M4)
HAL_RVV_CVT(RVV_U8MF4, RVV_U64M2)
HAL_RVV_CVT(RVV_U8MF8, RVV_U64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(F32, F64)                                                                    \
    template <>                                                                                  \
    inline F32::VecType F32::cast(F64::VecType v, size_t vl) { return __riscv_vfncvt_f(v, vl); } \
    template <>                                                                                  \
    inline F64::VecType F64::cast(F32::VecType v, size_t vl) { return __riscv_vfwcvt_f(v, vl); }

HAL_RVV_CVT(RVV_F32M4, RVV_F64M8)
HAL_RVV_CVT(RVV_F32M2, RVV_F64M4)
HAL_RVV_CVT(RVV_F32M1, RVV_F64M2)
HAL_RVV_CVT(RVV_F32MF2, RVV_F64M1)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_TYPE, LMUL, IS_U)                              \
    template <>                                                                               \
    inline RVV<A, LMUL_TYPE>::VecType RVV<A, LMUL_TYPE>::cast(                                \
        RVV<B, LMUL_TYPE>::VecType v, size_t vl                                               \
    ) {                                                                                       \
        return __riscv_vfcvt_f_x##IS_U##_v_##A_TYPE##LMUL(v, vl);                             \
    }                                                                                         \
    template <>                                                                               \
    inline RVV<B, LMUL_TYPE>::VecType RVV<B, LMUL_TYPE>::cast(                                \
        RVV<A, LMUL_TYPE>::VecType v, size_t vl                                               \
    ) {                                                                                       \
        return __riscv_vfcvt_x##IS_U##_f_v_##B_TYPE##LMUL(v, vl);                             \
    }

HAL_RVV_CVT( float,  int32_t, f32, i32,  LMUL_1,  m1, )
HAL_RVV_CVT( float,  int32_t, f32, i32,  LMUL_2,  m2, )
HAL_RVV_CVT( float,  int32_t, f32, i32,  LMUL_4,  m4, )
HAL_RVV_CVT( float,  int32_t, f32, i32,  LMUL_8,  m8, )
HAL_RVV_CVT( float,  int32_t, f32, i32, LMUL_f2, mf2, )

HAL_RVV_CVT( float, uint32_t, f32, u32,  LMUL_1,  m1, u)
HAL_RVV_CVT( float, uint32_t, f32, u32,  LMUL_2,  m2, u)
HAL_RVV_CVT( float, uint32_t, f32, u32,  LMUL_4,  m4, u)
HAL_RVV_CVT( float, uint32_t, f32, u32,  LMUL_8,  m8, u)
HAL_RVV_CVT( float, uint32_t, f32, u32, LMUL_f2, mf2, u)

HAL_RVV_CVT(double,  int64_t, f64, i64,  LMUL_1,  m1, )
HAL_RVV_CVT(double,  int64_t, f64, i64,  LMUL_2,  m2, )
HAL_RVV_CVT(double,  int64_t, f64, i64,  LMUL_4,  m4, )
HAL_RVV_CVT(double,  int64_t, f64, i64,  LMUL_8,  m8, )

HAL_RVV_CVT(double, uint64_t, f64, u64,  LMUL_1,  m1, u)
HAL_RVV_CVT(double, uint64_t, f64, u64,  LMUL_2,  m2, u)
HAL_RVV_CVT(double, uint64_t, f64, u64,  LMUL_4,  m4, u)
HAL_RVV_CVT(double, uint64_t, f64, u64,  LMUL_8,  m8, u)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_TYPE, LMUL)                                           \
    template <>                                                                                      \
    inline RVV<A, LMUL_TYPE>::VecType RVV<A, LMUL_TYPE>::reinterpret(RVV<B, LMUL_TYPE>::VecType v) { \
        return __riscv_vreinterpret_##A_TYPE##LMUL(v);                                               \
    }                                                                                                \
    template <>                                                                                      \
    inline RVV<B, LMUL_TYPE>::VecType RVV<B, LMUL_TYPE>::reinterpret(RVV<A, LMUL_TYPE>::VecType v) { \
        return __riscv_vreinterpret_##B_TYPE##LMUL(v);                                               \
    }

#define HAL_RVV_CVT2(A, B, A_TYPE, B_TYPE)        \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_1, m1) \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_2, m2) \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_4, m4) \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_8, m8)

HAL_RVV_CVT2( uint8_t,  int8_t,  u8,  i8)
HAL_RVV_CVT2(uint16_t, int16_t, u16, i16)
HAL_RVV_CVT2(uint32_t, int32_t, u32, i32)
HAL_RVV_CVT2(uint64_t, int64_t, u64, i64)

HAL_RVV_CVT2( int32_t,  float, i32, f32)
HAL_RVV_CVT2(uint32_t,  float, u32, f32)
HAL_RVV_CVT2( int64_t, double, i64, f64)
HAL_RVV_CVT2(uint64_t, double, u64, f64)

#undef HAL_RVV_CVT2

HAL_RVV_CVT( uint8_t,  int8_t,  u8,  i8, LMUL_f2, mf2)
HAL_RVV_CVT(uint16_t, int16_t, u16, i16, LMUL_f2, mf2)
HAL_RVV_CVT(uint32_t, int32_t, u32, i32, LMUL_f2, mf2)
HAL_RVV_CVT( int32_t,   float, i32, f32, LMUL_f2, mf2)
HAL_RVV_CVT(uint32_t,   float, u32, f32, LMUL_f2, mf2)

HAL_RVV_CVT( uint8_t,  int8_t,  u8,  i8, LMUL_f4, mf4)
HAL_RVV_CVT(uint16_t, int16_t, u16, i16, LMUL_f4, mf4)

HAL_RVV_CVT( uint8_t,  int8_t,  u8,  i8, LMUL_f8, mf8)

#undef HAL_RVV_CVT

#define HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_TYPE, LMUL)                                                                \
    template <>                                                                                                           \
    inline RVV<A, LMUL_TYPE>::VecType RVV<A, LMUL_TYPE>::cast(RVV<B, LMUL_TYPE>::VecType v, [[maybe_unused]] size_t vl) { \
        return __riscv_vreinterpret_##A_TYPE##LMUL(v);                                                                    \
    }                                                                                                                     \
    template <>                                                                                                           \
    inline RVV<B, LMUL_TYPE>::VecType RVV<B, LMUL_TYPE>::cast(RVV<A, LMUL_TYPE>::VecType v, [[maybe_unused]] size_t vl) { \
        return __riscv_vreinterpret_##B_TYPE##LMUL(v);                                                                    \
    }

#define HAL_RVV_CVT2(A, B, A_TYPE, B_TYPE)        \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_1, m1) \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_2, m2) \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_4, m4) \
    HAL_RVV_CVT(A, B, A_TYPE, B_TYPE, LMUL_8, m8)

HAL_RVV_CVT2( uint8_t,  int8_t,  u8,  i8)
HAL_RVV_CVT2(uint16_t, int16_t, u16, i16)
HAL_RVV_CVT2(uint32_t, int32_t, u32, i32)
HAL_RVV_CVT2(uint64_t, int64_t, u64, i64)

#undef HAL_RVV_CVT2
#undef HAL_RVV_CVT

#define HAL_RVV_CVT(FROM, INTERMEDIATE, TO)                      \
    template <>                                                  \
    inline TO::VecType TO::cast(FROM::VecType v, size_t vl) {    \
        return TO::cast(INTERMEDIATE::cast(v, vl), vl);          \
    }                                                            \
    template <>                                                  \
    inline FROM::VecType FROM::cast(TO::VecType v, size_t vl) {  \
        return FROM::cast(INTERMEDIATE::cast(v, vl), vl);        \
    }

// Integer and Float conversions
HAL_RVV_CVT(RVV_I8M1, RVV_I32M4, RVV_F32M4)
HAL_RVV_CVT(RVV_I8M2, RVV_I32M8, RVV_F32M8)
HAL_RVV_CVT(RVV_I8M1, RVV_I64M8, RVV_F64M8)

HAL_RVV_CVT(RVV_I16M1, RVV_I32M2, RVV_F32M2)
HAL_RVV_CVT(RVV_I16M2, RVV_I32M4, RVV_F32M4)
HAL_RVV_CVT(RVV_I16M4, RVV_I32M8, RVV_F32M8)
HAL_RVV_CVT(RVV_I16M1, RVV_I64M4, RVV_F64M4)
HAL_RVV_CVT(RVV_I16M2, RVV_I64M8, RVV_F64M8)

HAL_RVV_CVT(RVV_I32M1, RVV_I64M2, RVV_F64M2)
HAL_RVV_CVT(RVV_I32M2, RVV_I64M4, RVV_F64M4)
HAL_RVV_CVT(RVV_I32M4, RVV_I64M8, RVV_F64M8)

HAL_RVV_CVT(RVV_U8M1, RVV_U32M4, RVV_F32M4)
HAL_RVV_CVT(RVV_U8M2, RVV_U32M8, RVV_F32M8)
HAL_RVV_CVT(RVV_U8M1, RVV_U64M8, RVV_F64M8)

HAL_RVV_CVT(RVV_U16M1, RVV_U32M2, RVV_F32M2)
HAL_RVV_CVT(RVV_U16M2, RVV_U32M4, RVV_F32M4)
HAL_RVV_CVT(RVV_U16M4, RVV_U32M8, RVV_F32M8)
HAL_RVV_CVT(RVV_U16M1, RVV_U64M4, RVV_F64M4)
HAL_RVV_CVT(RVV_U16M2, RVV_U64M8, RVV_F64M8)

HAL_RVV_CVT(RVV_U32M1, RVV_U64M2, RVV_F64M2)
HAL_RVV_CVT(RVV_U32M2, RVV_U64M4, RVV_F64M4)
HAL_RVV_CVT(RVV_U32M4, RVV_U64M8, RVV_F64M8)

// Signed and Unsigned conversions
HAL_RVV_CVT(RVV_U8M1, RVV_U16M2, RVV_I16M2)
HAL_RVV_CVT(RVV_U8M2, RVV_U16M4, RVV_I16M4)
HAL_RVV_CVT(RVV_U8M4, RVV_U16M8, RVV_I16M8)

HAL_RVV_CVT(RVV_U8M1, RVV_U32M4, RVV_I32M4)
HAL_RVV_CVT(RVV_U8M2, RVV_U32M8, RVV_I32M8)

HAL_RVV_CVT(RVV_U8M1, RVV_U64M8, RVV_I64M8)

#undef HAL_RVV_CVT

// ---------------------------- Define Register Group Operations -------------------------------

#if defined(__clang__) && __clang_major__ <= 17
#define HAL_RVV_GROUP(ONE, TWO, TYPE, ONE_LMUL, TWO_LMUL)                     \
    template <size_t idx>                                                     \
    inline ONE::VecType vget(TWO::VecType v) {                                \
        return __riscv_vget_v_##TYPE##TWO_LMUL##_##TYPE##ONE_LMUL(v, idx);    \
    }                                                                         \
    template <size_t idx>                                                     \
    inline void vset(TWO::VecType v, ONE::VecType val) {                      \
        __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##TWO_LMUL(v, idx, val);      \
    }                                                                         \
    inline TWO::VecType vcreate(ONE::VecType v0, ONE::VecType v1) {           \
        TWO::VecType v{};                                                     \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##TWO_LMUL(v, 0, v0);     \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##TWO_LMUL(v, 1, v1);     \
        return v;                                                             \
    }
#else
#define HAL_RVV_GROUP(ONE, TWO, TYPE, ONE_LMUL, TWO_LMUL)                     \
    template <size_t idx>                                                     \
    inline ONE::VecType vget(TWO::VecType v) {                                \
        return __riscv_vget_v_##TYPE##TWO_LMUL##_##TYPE##ONE_LMUL(v, idx);    \
    }                                                                         \
    template <size_t idx>                                                     \
    inline void vset(TWO::VecType v, ONE::VecType val) {                      \
        __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##TWO_LMUL(v, idx, val);      \
    }                                                                         \
    inline TWO::VecType vcreate(ONE::VecType v0, ONE::VecType v1) {           \
        return __riscv_vcreate_v_##TYPE##ONE_LMUL##_##TYPE##TWO_LMUL(v0, v1); \
    }
#endif

HAL_RVV_GROUP(RVV_I8M1, RVV_I8M2, i8, m1, m2)
HAL_RVV_GROUP(RVV_I8M2, RVV_I8M4, i8, m2, m4)
HAL_RVV_GROUP(RVV_I8M4, RVV_I8M8, i8, m4, m8)

HAL_RVV_GROUP(RVV_I16M1, RVV_I16M2, i16, m1, m2)
HAL_RVV_GROUP(RVV_I16M2, RVV_I16M4, i16, m2, m4)
HAL_RVV_GROUP(RVV_I16M4, RVV_I16M8, i16, m4, m8)

HAL_RVV_GROUP(RVV_I32M1, RVV_I32M2, i32, m1, m2)
HAL_RVV_GROUP(RVV_I32M2, RVV_I32M4, i32, m2, m4)
HAL_RVV_GROUP(RVV_I32M4, RVV_I32M8, i32, m4, m8)

HAL_RVV_GROUP(RVV_I64M1, RVV_I64M2, i64, m1, m2)
HAL_RVV_GROUP(RVV_I64M2, RVV_I64M4, i64, m2, m4)
HAL_RVV_GROUP(RVV_I64M4, RVV_I64M8, i64, m4, m8)

HAL_RVV_GROUP(RVV_U8M1, RVV_U8M2, u8, m1, m2)
HAL_RVV_GROUP(RVV_U8M2, RVV_U8M4, u8, m2, m4)
HAL_RVV_GROUP(RVV_U8M4, RVV_U8M8, u8, m4, m8)

HAL_RVV_GROUP(RVV_U16M1, RVV_U16M2, u16, m1, m2)
HAL_RVV_GROUP(RVV_U16M2, RVV_U16M4, u16, m2, m4)
HAL_RVV_GROUP(RVV_U16M4, RVV_U16M8, u16, m4, m8)

HAL_RVV_GROUP(RVV_U32M1, RVV_U32M2, u32, m1, m2)
HAL_RVV_GROUP(RVV_U32M2, RVV_U32M4, u32, m2, m4)
HAL_RVV_GROUP(RVV_U32M4, RVV_U32M8, u32, m4, m8)

HAL_RVV_GROUP(RVV_U64M1, RVV_U64M2, u64, m1, m2)
HAL_RVV_GROUP(RVV_U64M2, RVV_U64M4, u64, m2, m4)
HAL_RVV_GROUP(RVV_U64M4, RVV_U64M8, u64, m4, m8)

HAL_RVV_GROUP(RVV_F32M1, RVV_F32M2, f32, m1, m2)
HAL_RVV_GROUP(RVV_F32M2, RVV_F32M4, f32, m2, m4)
HAL_RVV_GROUP(RVV_F32M4, RVV_F32M8, f32, m4, m8)

HAL_RVV_GROUP(RVV_F64M1, RVV_F64M2, f64, m1, m2)
HAL_RVV_GROUP(RVV_F64M2, RVV_F64M4, f64, m2, m4)
HAL_RVV_GROUP(RVV_F64M4, RVV_F64M8, f64, m4, m8)

#undef HAL_RVV_GROUP

#if defined(__clang__) && __clang_major__ <= 17
#define HAL_RVV_GROUP(ONE, FOUR, TYPE, ONE_LMUL, FOUR_LMUL)                                            \
    template <size_t idx>                                                                              \
    inline ONE::VecType vget(FOUR::VecType v) {                                                        \
        return __riscv_vget_v_##TYPE##FOUR_LMUL##_##TYPE##ONE_LMUL(v, idx);                            \
    }                                                                                                  \
    template <size_t idx>                                                                              \
    inline void vset(FOUR::VecType v, ONE::VecType val) {                                              \
        __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v, idx, val);                              \
    }                                                                                                  \
    inline FOUR::VecType vcreate(ONE::VecType v0, ONE::VecType v1, ONE::VecType v2, ONE::VecType v3) { \
        FOUR::VecType v{};                                                                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v, 0, v0);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v, 1, v1);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v, 2, v2);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v, 3, v3);                             \
        return v;                                                                                      \
    }
#else
#define HAL_RVV_GROUP(ONE, FOUR, TYPE, ONE_LMUL, FOUR_LMUL)                                            \
    template <size_t idx>                                                                              \
    inline ONE::VecType vget(FOUR::VecType v) {                                                        \
        return __riscv_vget_v_##TYPE##FOUR_LMUL##_##TYPE##ONE_LMUL(v, idx);                            \
    }                                                                                                  \
    template <size_t idx>                                                                              \
    inline void vset(FOUR::VecType v, ONE::VecType val) {                                              \
        __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v, idx, val);                              \
    }                                                                                                  \
    inline FOUR::VecType vcreate(ONE::VecType v0, ONE::VecType v1, ONE::VecType v2, ONE::VecType v3) { \
        return __riscv_vcreate_v_##TYPE##ONE_LMUL##_##TYPE##FOUR_LMUL(v0, v1, v2, v3);                 \
    }
#endif

HAL_RVV_GROUP(RVV_I8M1, RVV_I8M4, i8, m1, m4)
HAL_RVV_GROUP(RVV_I8M2, RVV_I8M8, i8, m2, m8)

HAL_RVV_GROUP(RVV_U8M1, RVV_U8M4, u8, m1, m4)
HAL_RVV_GROUP(RVV_U8M2, RVV_U8M8, u8, m2, m8)

HAL_RVV_GROUP(RVV_I16M1, RVV_I16M4, i16, m1, m4)
HAL_RVV_GROUP(RVV_I16M2, RVV_I16M8, i16, m2, m8)

HAL_RVV_GROUP(RVV_U16M1, RVV_U16M4, u16, m1, m4)
HAL_RVV_GROUP(RVV_U16M2, RVV_U16M8, u16, m2, m8)

HAL_RVV_GROUP(RVV_I32M1, RVV_I32M4, i32, m1, m4)
HAL_RVV_GROUP(RVV_I32M2, RVV_I32M8, i32, m2, m8)

HAL_RVV_GROUP(RVV_U32M1, RVV_U32M4, u32, m1, m4)
HAL_RVV_GROUP(RVV_U32M2, RVV_U32M8, u32, m2, m8)

HAL_RVV_GROUP(RVV_I64M1, RVV_I64M4, i64, m1, m4)
HAL_RVV_GROUP(RVV_I64M2, RVV_I64M8, i64, m2, m8)

HAL_RVV_GROUP(RVV_U64M1, RVV_U64M4, u64, m1, m4)
HAL_RVV_GROUP(RVV_U64M2, RVV_U64M8, u64, m2, m8)

HAL_RVV_GROUP(RVV_F32M1, RVV_F32M4, f32, m1, m4)
HAL_RVV_GROUP(RVV_F32M2, RVV_F32M8, f32, m2, m8)

HAL_RVV_GROUP(RVV_F64M1, RVV_F64M4, f64, m1, m4)
HAL_RVV_GROUP(RVV_F64M2, RVV_F64M8, f64, m2, m8)

#undef HAL_RVV_GROUP

#if defined(__clang__) && __clang_major__ <= 17
#define HAL_RVV_GROUP(ONE, EIGHT, TYPE, ONE_LMUL, EIGHT_LMUL)                                           \
    template <size_t idx>                                                                               \
    inline ONE::VecType vget(EIGHT::VecType v) {                                                        \
        return __riscv_vget_v_##TYPE##EIGHT_LMUL##_##TYPE##ONE_LMUL(v, idx);                            \
    }                                                                                                   \
    template <size_t idx>                                                                               \
    inline void vset(EIGHT::VecType v, ONE::VecType val) {                                              \
        __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, idx, val);                              \
    }                                                                                                   \
    inline EIGHT::VecType vcreate(ONE::VecType v0, ONE::VecType v1, ONE::VecType v2, ONE::VecType v3,   \
        ONE::VecType v4, ONE::VecType v5, ONE::VecType v6, ONE::VecType v7) {                           \
        EIGHT::VecType v{};                                                                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 0, v0);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 1, v1);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 2, v2);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 3, v3);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 4, v4);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 5, v5);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 6, v6);                             \
        v = __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, 7, v7);                             \
        return v;                                                                                       \
    }
#else
#define HAL_RVV_GROUP(ONE, EIGHT, TYPE, ONE_LMUL, EIGHT_LMUL)                                           \
    template <size_t idx>                                                                               \
    inline ONE::VecType vget(EIGHT::VecType v) {                                                        \
        return __riscv_vget_v_##TYPE##EIGHT_LMUL##_##TYPE##ONE_LMUL(v, idx);                            \
    }                                                                                                   \
    template <size_t idx>                                                                               \
    inline void vset(EIGHT::VecType v, ONE::VecType val) {                                              \
        __riscv_vset_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v, idx, val);                              \
    }                                                                                                   \
    inline EIGHT::VecType vcreate(ONE::VecType v0, ONE::VecType v1, ONE::VecType v2, ONE::VecType v3,   \
        ONE::VecType v4, ONE::VecType v5, ONE::VecType v6, ONE::VecType v7) {                           \
        return __riscv_vcreate_v_##TYPE##ONE_LMUL##_##TYPE##EIGHT_LMUL(v0, v1, v2, v3, v4, v5, v6, v7); \
    }
#endif

HAL_RVV_GROUP(RVV_I8M1, RVV_I8M8, i8, m1, m8)
HAL_RVV_GROUP(RVV_U8M1, RVV_U8M8, u8, m1, m8)

HAL_RVV_GROUP(RVV_I16M1, RVV_I16M8, i16, m1, m8)
HAL_RVV_GROUP(RVV_U16M1, RVV_U16M8, u16, m1, m8)

HAL_RVV_GROUP(RVV_I32M1, RVV_I32M8, i32, m1, m8)
HAL_RVV_GROUP(RVV_U32M1, RVV_U32M8, u32, m1, m8)

HAL_RVV_GROUP(RVV_I64M1, RVV_I64M8, i64, m1, m8)
HAL_RVV_GROUP(RVV_U64M1, RVV_U64M8, u64, m1, m8)

HAL_RVV_GROUP(RVV_F32M1, RVV_F32M8, f32, m1, m8)
HAL_RVV_GROUP(RVV_F64M1, RVV_F64M8, f64, m1, m8)

#undef HAL_RVV_GROUP

}}  // namespace cv::cv_hal_rvv

#endif //OPENCV_HAL_RVV_TYPES_HPP_INCLUDED
