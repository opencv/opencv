// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.
#pragma once

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

// Only for dst type lmul >= 1
template <typename Dst_T, typename RVV_T>
using RVV_SameLen =
    RVV<Dst_T, RVV_LMUL(RVV_T::lmul / sizeof(typename RVV_T::ElemType) * sizeof(Dst_T))>;

// -------------------------------Supported operations--------------------------------

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
HAL_RVV_SIZE_RELATED_CUSTOM(EEW, TYPE, LMUL)

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

#define HAL_RVV_BOOL_TYPE(S_OR_F, X_OR_F, IS_U, IS_F) \
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
        static constexpr size_t lmul = LMUL_TYPE;          \
                                                           \
        HAL_RVV_SIZE_RELATED(EEW, TYPE, LMUL, __VA_ARGS__) \
        HAL_RVV_SIZE_UNRELATED(__VA_ARGS__)                \
                                                           \
        template <typename FROM>                           \
        inline static VecType cast(FROM v, size_t vl);     \
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

#define HAL_RVV_SIGNED_PARAM   s,x, ,
#define HAL_RVV_UNSIGNED_PARAM s,x,u,
#define HAL_RVV_FLOAT_PARAM    f,f, ,f

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

#define HAL_RVV_CVT(ONE, TWO)                                                                   \
    template <>                                                                                 \
    inline ONE::VecType ONE::cast(TWO::VecType v, size_t vl) { return __riscv_vncvt_x(v, vl); } \
    template <>                                                                                 \
    inline TWO::VecType TWO::cast(ONE::VecType v, size_t vl) { return __riscv_vwcvt_x(v, vl); }

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
    inline TWO::VecType TWO::cast(ONE::VecType v, size_t vl) { return __riscv_vwcvtu_x(v, vl); }

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

#define HAL_RVV_CVT(ONE, FOUR) \
    template <>                \
    inline FOUR::VecType FOUR::cast(ONE::VecType v, size_t vl) { return __riscv_vsext_vf4(v, vl); }

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

#define HAL_RVV_CVT(ONE, FOUR) \
    template <>                \
    inline FOUR::VecType FOUR::cast(ONE::VecType v, size_t vl) { return __riscv_vzext_vf4(v, vl); }

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

#define HAL_RVV_CVT(ONE, EIGHT) \
    template <>                 \
    inline EIGHT::VecType EIGHT::cast(ONE::VecType v, size_t vl) { return __riscv_vzext_vf8(v, vl); }

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

}}  // namespace cv::cv_hal_rvv
