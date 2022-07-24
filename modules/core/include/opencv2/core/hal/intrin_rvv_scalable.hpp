
#ifndef OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP
#define OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP

#include <initializer_list>
#include <assert.h>
#include <vector>

#ifndef CV_RVV_MAX_VLEN
#define CV_RVV_MAX_VLEN 1024
#endif

namespace cv
{
CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

#define CV_SIMD_SCALABLE 1
#define CV_SIMD_SCALABLE_64F 1

using v_uint8 = vuint8m1_t;
using v_int8 = vint8m1_t;
using v_uint16 = vuint16m1_t;
using v_int16 = vint16m1_t;
using v_uint32 = vuint32m1_t;
using v_int32 = vint32m1_t;
using v_uint64 = vuint64m1_t;
using v_int64 = vint64m1_t;

using v_float32 = vfloat32m1_t;
#if CV_SIMD_SCALABLE_64F
using v_float64 = vfloat64m1_t;
#endif

using uchar = unsigned char;
using schar = signed char;
using ushort = unsigned short;
using uint = unsigned int;
using uint64 = unsigned long int;
using int64 = long int;

static const int __cv_rvv_e8_nlanes = vsetvlmax_e8m1();
static const int __cv_rvv_e16_nlanes = vsetvlmax_e16m1();
static const int __cv_rvv_e32_nlanes = vsetvlmax_e32m1();
static const int __cv_rvv_e64_nlanes = vsetvlmax_e64m1();

template <class T>
struct VTraits;

template <>
struct VTraits<v_uint8>
{
    static inline int vlanes() { return __cv_rvv_e8_nlanes; }
    using lane_type = uchar;
    static const int max_nlanes = CV_RVV_MAX_VLEN/8;
};

template <>
struct VTraits<v_int8>
{
    static inline int vlanes() { return __cv_rvv_e8_nlanes; }
    using lane_type = schar;
    static const int max_nlanes = CV_RVV_MAX_VLEN/8;
};
template <>
struct VTraits<v_uint16>
{
    static inline int vlanes() { return __cv_rvv_e16_nlanes; }
    using lane_type = ushort;
    static const int max_nlanes = CV_RVV_MAX_VLEN/16;
};
template <>
struct VTraits<v_int16>
{
    static inline int vlanes() { return __cv_rvv_e16_nlanes; }
    using lane_type = short;
    static const int max_nlanes = CV_RVV_MAX_VLEN/16;
};
template <>
struct VTraits<v_uint32>
{
    static inline int vlanes() { return __cv_rvv_e32_nlanes; }
    using lane_type = uint;
    static const int max_nlanes = CV_RVV_MAX_VLEN/32;
};
template <>
struct VTraits<v_int32>
{
    static inline int vlanes() { return __cv_rvv_e32_nlanes; }
    using lane_type = int;
    static const int max_nlanes = CV_RVV_MAX_VLEN/32;
};

template <>
struct VTraits<v_float32>
{
    static inline int vlanes() { return __cv_rvv_e32_nlanes; }
    using lane_type = float;
    static const int max_nlanes = CV_RVV_MAX_VLEN/32;
};
template <>
struct VTraits<v_uint64>
{
    static inline int vlanes() { return __cv_rvv_e64_nlanes; }
    using lane_type = uint64;
    static const int max_nlanes = CV_RVV_MAX_VLEN/64;
};
template <>
struct VTraits<v_int64>
{
    static inline int vlanes() { return __cv_rvv_e64_nlanes; }
    using lane_type = int64;
    static const int max_nlanes = CV_RVV_MAX_VLEN/64;
};
#if CV_SIMD_SCALABLE_64F
template <>
struct VTraits<v_float64>
{
    static inline int vlanes() { return __cv_rvv_e64_nlanes; }
    using lane_type = double;
    static const int max_nlanes = CV_RVV_MAX_VLEN/64;
};
#endif

//////////// get0 ////////////
#define OPENCV_HAL_IMPL_RVV_GRT0_INT(_Tpvec, _Tp) \
inline _Tp v_get0(const v_##_Tpvec& v) \
{ \
    return vmv_x(v); \
}

OPENCV_HAL_IMPL_RVV_GRT0_INT(uint8, uchar)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int8, schar)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint16, ushort)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int16, short)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint32, unsigned)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int32, int)
OPENCV_HAL_IMPL_RVV_GRT0_INT(uint64, uint64)
OPENCV_HAL_IMPL_RVV_GRT0_INT(int64, int64)

inline float v_get0(const v_float32& v) \
{ \
    return vfmv_f(v); \
}
#if CV_SIMD_SCALABLE_64F
inline double v_get0(const v_float64& v) \
{ \
    return vfmv_f(v); \
}
#endif

//////////// Initial ////////////

#define OPENCV_HAL_IMPL_RVV_INIT_INTEGER(_Tpvec, _Tp, suffix1, suffix2, vl) \
inline v_##_Tpvec v_setzero_##suffix1() \
{ \
    return vmv_v_x_##suffix2##m1(0, vl); \
} \
inline v_##_Tpvec v_setall_##suffix1(_Tp v) \
{ \
    return vmv_v_x_##suffix2##m1(v, vl); \
}

OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint8, uchar, u8, u8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int8, schar, s8, i8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint16, ushort, u16, u16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int16, short, s16, i16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint32, uint, u32, u32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int32, int, s32, i32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(uint64, uint64, u64, u64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_INIT_INTEGER(int64, int64, s64, i64, VTraits<v_int64>::vlanes())

#define OPENCV_HAL_IMPL_RVV_INIT_FP(_Tpv, _Tp, suffix, vl) \
inline v_##_Tpv v_setzero_##suffix() \
{ \
    return vfmv_v_f_##suffix##m1(0, vl); \
} \
inline v_##_Tpv v_setall_##suffix(_Tp v) \
{ \
    return vfmv_v_f_##suffix##m1(v, vl); \
}

OPENCV_HAL_IMPL_RVV_INIT_FP(float32, float, f32, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_INIT_FP(float64, double, f64, VTraits<v_float64>::vlanes())
#endif

//////////// Reinterpret ////////////
#define OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(_Tpvec1, suffix1) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec1& v) \
{ \
    return v;\
}
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint8, u8)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint16, u16)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint32, u32)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(uint64, u64)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int8, s8)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int16, s16)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int32, s32)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(int64, s64)
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(float32, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_NOTHING_REINTERPRET(float64, f64)
#endif
// TODO: can be simplified by using overloaded RV intrinsic
#define OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, nsuffix1, nsuffix2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) \
{ \
    return v_##_Tpvec1(vreinterpret_v_##nsuffix2##m1_##nsuffix1##m1(v));\
} \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) \
{ \
    return v_##_Tpvec2(vreinterpret_v_##nsuffix1##m1_##nsuffix2##m1(v));\
}

OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, int8, u8, s8, u8, i8)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, int16, u16, s16, u16, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, int32, u32, s32, u32, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, float32, u32, f32, u32, f32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int32, float32, s32, f32, i32, f32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint64, int64, u64, s64, u64, i64)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint64, float64, u64, f64, u64, f64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int64, float64, s64, f64, i64, f64)
#endif
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint16, u8, u16, u8, u16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint32, u8, u32, u8, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint8, uint64, u8, u64, u8, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, uint32, u16, u32, u16, u32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint16, uint64, u16, u64, u16, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(uint32, uint64, u32, u64, u32, u64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int16, s8, s16, i8, i16)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int32, s8, s32, i8, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int8, int64, s8, s64, i8, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, int32, s16, s32, i16, i32)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int16, int64, s16, s64, i16, i64)
OPENCV_HAL_IMPL_RVV_NATIVE_REINTERPRET(int32, int64, s32, s64, i32, i64)


#define OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(_Tpvec1, _Tpvec2, suffix1, suffix2, nsuffix1, nsuffix2, width1, width2) \
inline v_##_Tpvec1 v_reinterpret_as_##suffix1(const v_##_Tpvec2& v) \
{ \
    return vreinterpret_v_##nsuffix1##width2##m1_##nsuffix1##width1##m1(vreinterpret_v_##nsuffix2##width2##m1_##nsuffix1##width2##m1(v));\
} \
inline v_##_Tpvec2 v_reinterpret_as_##suffix2(const v_##_Tpvec1& v) \
{ \
    return vreinterpret_v_##nsuffix1##width2##m1_##nsuffix2##width2##m1(vreinterpret_v_##nsuffix1##width1##m1_##nsuffix1##width2##m1(v));\
}

OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int16, u8, s16, u, i, 8, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int32, u8, s32, u, i, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, int64, u8, s64, u, i, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int8, u16, s8, u, i, 16, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int32, u16, s32, u, i, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, int64, u16, s64, u, i, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int8, u32, s8, u, i, 32, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int16, u32, s16, u, i, 32, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, int64, u32, s64, u, i, 32, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int8, u64, s8, u, i, 64, 8)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int16, u64, s16, u, i, 64, 16)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, int32, u64, s32, u, i, 64, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float32, u8, f32, u, f, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, float32, u16, f32, u, f, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint64, float32, u64, f32, u, f, 64, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float32, s8, f32, i, f, 8, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16, float32, s16, f32, i, f, 16, 32)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int64, float32, s64, f32, i, f, 64, 32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint8, float64, u8, f64, u, f, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint16, float64, u16, f64, u, f, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(uint32, float64, u32, f64, u, f, 32, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int8, float64, s8, f64, i, f, 8, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int16, float64, s16, f64, i, f, 16, 64)
OPENCV_HAL_IMPL_RVV_TWO_TIMES_REINTERPRET(int32, float64, s32, f64, i, f, 32, 64)
// Three times reinterpret
inline v_float32 v_reinterpret_as_f32(const v_float64& v) \
{ \
    return vreinterpret_v_u32m1_f32m1(vreinterpret_v_u64m1_u32m1(vreinterpret_v_f64m1_u64m1(v)));\
}

inline v_float64 v_reinterpret_as_f64(const v_float32& v) \
{ \
    return vreinterpret_v_u64m1_f64m1(vreinterpret_v_u32m1_u64m1(vreinterpret_v_f32m1_u32m1(v)));\
}
#endif


////////////// Load/Store //////////////
#define OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(_Tpvec, _nTpvec, _Tp, hvl, vl, width, suffix, vmv) \
inline _Tpvec v_load(const _Tp* ptr) \
{ \
    return vle##width##_v_##suffix##m1(ptr, vl); \
} \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ \
    return vle##width##_v_##suffix##m1(ptr, vl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode /*mode*/) \
{ \
    vse##width##_v_##suffix##m1(ptr, a, vl); \
} \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ \
    return vle##width##_v_##suffix##m1(ptr, hvl); \
} \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return vslideup(vle##width##_v_##suffix##m1(ptr0, hvl), vle##width##_v_##suffix##m1(ptr1, hvl), hvl, vl); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, vl); \
} \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, vl); \
} \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, vl); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, a, hvl); \
} \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    vse##width(ptr, vslidedown_vx_##suffix##m1(vmv(0, vl), a, hvl, vl), hvl); \
} \
inline _Tpvec v_load(std::initializer_list<_Tp> nScalars) \
{ \
    assert(nScalars.size() == vl); \
    return vle##width##_v_##suffix##m1(nScalars.begin(), nScalars.size()); \
} \
template<typename... Targs> \
_Tpvec v_load_##suffix(Targs... nScalars) \
{ \
    return v_load({nScalars...}); \
}


OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint8, vuint8m1_t, uchar, VTraits<v_uint8>::vlanes() / 2, VTraits<v_uint8>::vlanes(), 8, u8, vmv_v_x_u8m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int8, vint8m1_t, schar, VTraits<v_int8>::vlanes() / 2, VTraits<v_int8>::vlanes(), 8, i8, vmv_v_x_i8m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint16, vuint16m1_t, ushort, VTraits<v_uint16>::vlanes() / 2, VTraits<v_uint16>::vlanes(), 16, u16, vmv_v_x_u16m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int16, vint16m1_t, short, VTraits<v_int16>::vlanes() / 2, VTraits<v_int16>::vlanes(), 16, i16, vmv_v_x_i16m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint32, vuint32m1_t, unsigned int, VTraits<v_uint32>::vlanes() / 2, VTraits<v_uint32>::vlanes(), 32, u32, vmv_v_x_u32m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int32, vint32m1_t, int, VTraits<v_int32>::vlanes() / 2, VTraits<v_int32>::vlanes(), 32, i32, vmv_v_x_i32m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_uint64, vuint64m1_t, uint64, VTraits<v_uint64>::vlanes() / 2, VTraits<v_uint64>::vlanes(), 64, u64, vmv_v_x_u64m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_int64, vint64m1_t, int64, VTraits<v_int64>::vlanes() / 2, VTraits<v_int64>::vlanes(), 64, i64, vmv_v_x_i64m1)
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float32, vfloat32m1_t, float, VTraits<v_float32>::vlanes() /2 , VTraits<v_float32>::vlanes(), 32, f32, vfmv_v_f_f32m1)

#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_LOADSTORE_OP(v_float64, vfloat64m1_t, double, VTraits<v_float64>::vlanes() / 2, VTraits<v_float64>::vlanes(), 64, f64, vfmv_v_f_f64m1)
#endif

////////////// Lookup table access ////////////////////
#define OPENCV_HAL_IMPL_RVV_LUT(_Tpvec, _Tp, suffix) \
inline _Tpvec v_lut(const _Tp* tab, const int* idx) \
{ \
    vuint32##suffix##_t vidx = vmul(vreinterpret_u32##suffix(vle32_v_i32##suffix(idx, VTraits<_Tpvec>::vlanes())), sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return vloxei32(tab, vidx, VTraits<_Tpvec>::vlanes()); \
} \
inline _Tpvec v_lut_pairs(const _Tp* tab, const int* idx) \
{ \
    std::vector<uint> idx_; \
    for (size_t i = 0; i < VTraits<v_int16>::vlanes(); ++i) { \
        idx_.push_back(idx[i]); \
        idx_.push_back(idx[i]+1); \
    } \
    vuint32##suffix##_t vidx = vmul(vle32_v_u32##suffix(idx_.data(), VTraits<_Tpvec>::vlanes()), sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return vloxei32(tab, vidx, VTraits<_Tpvec>::vlanes()); \
} \
inline _Tpvec v_lut_quads(const _Tp* tab, const int* idx) \
{ \
    std::vector<uint> idx_; \
    for (size_t i = 0; i < VTraits<v_int32>::vlanes(); ++i) { \
        idx_.push_back(idx[i]); \
        idx_.push_back(idx[i]+1); \
        idx_.push_back(idx[i]+2); \
        idx_.push_back(idx[i]+3); \
    } \
    vuint32##suffix##_t vidx = vmul(vle32_v_u32##suffix(idx_.data(), VTraits<_Tpvec>::vlanes()), sizeof(_Tp), VTraits<_Tpvec>::vlanes()); \
    return vloxei32(tab, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_LUT(v_int8, schar, m4)
OPENCV_HAL_IMPL_RVV_LUT(v_int16, short, m2)
OPENCV_HAL_IMPL_RVV_LUT(v_int32, int, m1)
OPENCV_HAL_IMPL_RVV_LUT(v_int64, int64_t, mf2)
OPENCV_HAL_IMPL_RVV_LUT(v_float32, float, m1)

inline v_uint8 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((schar*)tab, idx)); }
inline v_uint8 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((schar*)tab, idx)); }
inline v_uint8 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((schar*)tab, idx)); }
inline v_uint16 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((short*)tab, idx)); }
inline v_uint16 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((short*)tab, idx)); }
inline v_uint16 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((short*)tab, idx)); }
inline v_uint32 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((int*)tab, idx)); }
inline v_uint32 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((int*)tab, idx)); }
inline v_uint32 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((int*)tab, idx)); }
inline v_uint64 v_lut(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64 v_lut_pairs(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }
inline v_uint64 v_lut_quads(const uint64* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_quads((const int64_t*)tab, idx)); }


////////////// Min/Max //////////////

#define OPENCV_HAL_IMPL_RVV_BIN_FUNC(_Tpvec, func, intrin, vl) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return intrin(a, b, vl); \
}

OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8, v_min, vminu, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint8, v_max, vmaxu, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8, v_min, vmin, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int8, v_max, vmax, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16, v_min, vminu, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint16, v_max, vmaxu, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16, v_min, vmin, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int16, v_max, vmax, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32, v_min, vminu, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint32, v_max, vmaxu, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32, v_min, vmin, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int32, v_max, vmax, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32, v_min, vfmin, VTraits<v_float32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float32, v_max, vfmax, VTraits<v_float32>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint64, v_min, vminu, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_uint64, v_max, vmaxu, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int64, v_min, vmin, VTraits<v_int64>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_int64, v_max, vmax, VTraits<v_int64>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_min, vfmin, VTraits<v_float64>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_max, vfmax, VTraits<v_float64>::vlanes())
#endif


//////////// Value reordering ////////////

#define OPENCV_HAL_IMPL_RVV_EXPAND(_Tp, _Tpwvec, _Tpwvec_m2, _Tpvec, width, suffix, suffix2, cvt) \
inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
{ \
    _Tpwvec_m2 temp = cvt(a, vsetvlmax_e##width##m1()); \
    b0 = vget_##suffix##m1(temp, 0); \
    b1 = vget_##suffix##m1(temp, 1); \
} \
inline _Tpwvec v_expand_low(const _Tpvec& a) \
{ \
    _Tpwvec_m2 temp = cvt(a, vsetvlmax_e##width##m1()); \
    return vget_##suffix##m1(temp, 0); \
} \
inline _Tpwvec v_expand_high(const _Tpvec& a) \
{ \
    _Tpwvec_m2 temp = cvt(a, vsetvlmax_e##width##m1()); \
    return vget_##suffix##m1(temp, 1); \
} \
inline _Tpwvec v_load_expand(const _Tp* ptr) \
{ \
    return cvt(vle##width##_v_##suffix2##mf2(ptr, vsetvlmax_e##width##m1()), vsetvlmax_e##width##m1()); \
}

OPENCV_HAL_IMPL_RVV_EXPAND(uchar, v_uint16, vuint16m2_t, v_uint8, 8, u16, u8, vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(schar, v_int16, vint16m2_t, v_int8, 8, i16, i8, vwcvt_x)
OPENCV_HAL_IMPL_RVV_EXPAND(ushort, v_uint32, vuint32m2_t, v_uint16, 16, u32, u16, vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(short, v_int32, vint32m2_t, v_int16, 16, i32, i16, vwcvt_x)
OPENCV_HAL_IMPL_RVV_EXPAND(uint, v_uint64, vuint64m2_t, v_uint32, 32, u64, u32, vwcvtu_x)
OPENCV_HAL_IMPL_RVV_EXPAND(int, v_int64, vint64m2_t, v_int32, 32, i64, i32, vwcvt_x)

inline v_uint32 v_load_expand_q(const uchar* ptr)
{
    return vwcvtu_x(vwcvtu_x(vle8_v_u8mf4(ptr, VTraits<v_uint32>::vlanes()), VTraits<v_uint32>::vlanes()), VTraits<v_uint32>::vlanes());
}

inline v_int32 v_load_expand_q(const schar* ptr)
{
    return vwcvt_x(vwcvt_x(vle8_v_i8mf4(ptr, VTraits<v_int32>::vlanes()), VTraits<v_int32>::vlanes()), VTraits<v_int32>::vlanes());
}


////// FP16 support ///////

inline v_float32 v_load_expand(const float16_t* ptr)
{
    // TODO
    return vundefined_f32m1();
}

inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

} //namespace cv

#endif //OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP