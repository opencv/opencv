
#ifndef OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP
#define OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP

#include <initializer_list>
#include <assert.h>
#include <vector>
#include <opencv2/core/check.hpp>

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

//////////// Extract //////////////

#define OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(_Tpvec, _Tp, suffix, vl) \
template <int s = 0> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b, int i = s) \
{ \
    return vslideup(vslidedown(v_setzero_##suffix(), a, i, vl), b, VTraits<_Tpvec>::vlanes() - i, vl); \
} \
template<int s = 0> inline _Tp v_extract_n(_Tpvec v, int i = s) \
{ \
    return vmv_x(vslidedown(v_setzero_##suffix(), v, i, vl)); \
}


OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint8, uchar, u8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int8, schar, s8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint16, ushort, u16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int16, short, s16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint32, unsigned int, u32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int32, int, s32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_uint64, uint64, u64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT_INTEGER(v_int64, int64, s64, VTraits<v_int64>::vlanes())

#define OPENCV_HAL_IMPL_RVV_EXTRACT_FP(_Tpvec, _Tp, suffix, vl) \
template <int s = 0> \
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b, int i = s) \
{ \
    return vslideup(vslidedown(v_setzero_##suffix(), a, i, vl), b, VTraits<_Tpvec>::vlanes() - i, vl); \
} \
template<int s = 0> inline _Tp v_extract_n(_Tpvec v, int i = s) \
{ \
    return vfmv_f(vslidedown(v_setzero_##suffix(), v, i, vl)); \
}

OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float32, float, f32, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_EXTRACT_FP(v_float64, double, f64, VTraits<v_float64>::vlanes())
#endif

#define OPENCV_HAL_IMPL_RVV_EXTRACT(_Tpvec, _Tp, vl) \
inline _Tp v_extract_highest(_Tpvec v) \
{ \
    return v_extract_n(v, vl-1); \
}

OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint8, uchar, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int8, schar, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint16, ushort, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int16, short, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint32, unsigned int, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int32, int, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_uint64, uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_int64, int64, VTraits<v_int64>::vlanes())
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float32, float, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_EXTRACT(v_float64, double, VTraits<v_float64>::vlanes())
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
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_LUT(v_float64, double, mf2)
#endif

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

////////////// Pack boolean ////////////////////
/* TODO */

////////////// Arithmetics //////////////
#define OPENCV_HAL_IMPL_RVV_BIN_OP(_Tpvec, ocv_intrin, rvv_intrin) \
inline _Tpvec v_##ocv_intrin(const _Tpvec& a, const _Tpvec& b) \
{ \
    return rvv_intrin(a, b, VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, add, vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, sub, vssubu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, add, vsadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, sub, vssub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, add, vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, sub, vssubu)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, add, vsadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, sub, vssub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, add, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, sub, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint32, mul, vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, add, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, sub, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int32, mul, vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, add, vfadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, sub, vfsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, mul, vfmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float32, div, vfdiv)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint64, add, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint64, sub, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int64, add, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int64, sub, vsub)

#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, add, vfadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, sub, vfsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, mul, vfmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_float64, div, vfdiv)
#endif

#define OPENCV_HAL_IMPL_RVV_BIN_MADD(_Tpvec, rvv_add) \
template<typename... Args> \
inline _Tpvec v_add(const _Tpvec& f1, const _Tpvec& f2, const Args&... vf) { \
    return v_add(rvv_add(f1, f2, VTraits<_Tpvec>::vlanes()), vf...); \
}
#define OPENCV_HAL_IMPL_RVV_BIN_MMUL(_Tpvec, rvv_mul) \
template<typename... Args> \
inline _Tpvec v_mul(const _Tpvec& f1, const _Tpvec& f2, const Args&... vf) { \
    return v_mul(rvv_mul(f1, f2, VTraits<_Tpvec>::vlanes()), vf...); \
}
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint8, vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int8, vsadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint16, vsaddu)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int16, vsadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint32, vadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int32, vadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_float32, vfadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_uint64, vadd)
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_int64, vadd)

OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_uint32, vmul)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_int32, vmul)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_float32, vfmul)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_MADD(v_float64, vfadd)
OPENCV_HAL_IMPL_RVV_BIN_MMUL(v_float64, vfmul)
#endif

#define OPENCV_HAL_IMPL_RVV_MUL_EXPAND(_Tpvec, _Tpwvec, _TpwvecM2, suffix, wmul) \
inline void v_mul_expand(const _Tpvec& a, const _Tpvec& b, _Tpwvec& c, _Tpwvec& d) \
{ \
    _TpwvecM2 temp = wmul(a, b, VTraits<_Tpvec>::vlanes()); \
    c = vget_##suffix##m1(temp, 0); \
    d = vget_##suffix##m1(temp, 1); \
}

OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint8, v_uint16, vuint16m2_t, u16, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int8, v_int16, vint16m2_t, i16, vwmul)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint16, v_uint32, vuint32m2_t, u32, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_int16, v_int32, vint32m2_t, i32, vwmul)
OPENCV_HAL_IMPL_RVV_MUL_EXPAND(v_uint32, v_uint64, vuint64m2_t, u64, vwmulu)

inline v_int16 v_mul_hi(const v_int16& a, const v_int16& b)
{
    return vmulh(a, b, VTraits<v_int16>::vlanes());
}
inline v_uint16 v_mul_hi(const v_uint16& a, const v_uint16& b)
{
    return vmulhu(a, b, VTraits<v_uint16>::vlanes());
}

////////////// Arithmetics (wrap)//////////////
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, add_wrap, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, add_wrap, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, add_wrap, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, add_wrap, vadd)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, sub_wrap, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, sub_wrap, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, sub_wrap, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, sub_wrap, vsub)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint8, mul_wrap, vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int8, mul_wrap, vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_uint16, mul_wrap, vmul)
OPENCV_HAL_IMPL_RVV_BIN_OP(v_int16, mul_wrap, vmul)

//////// Saturating Multiply ////////
#define OPENCV_HAL_IMPL_RVV_MUL_SAT(_Tpvec, _clip, _wmul) \
inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _clip(_wmul(a, b, VTraits<_Tpvec>::vlanes()), 0, VTraits<_Tpvec>::vlanes()); \
} \
template<typename... Args> \
inline _Tpvec v_mul(const _Tpvec& a1, const _Tpvec& a2, const Args&... va) { \
    return v_mul(_clip(_wmul(a1, a2, VTraits<_Tpvec>::vlanes()), 0, VTraits<_Tpvec>::vlanes()), va...); \
}

OPENCV_HAL_IMPL_RVV_MUL_SAT(v_uint8, vnclipu, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_int8, vnclip, vwmul)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_uint16, vnclipu, vwmulu)
OPENCV_HAL_IMPL_RVV_MUL_SAT(v_int16, vnclip, vwmul)

////////////// Bitwise logic //////////////

#define OPENCV_HAL_IMPL_RVV_LOGIC_OP(_Tpvec, vl) \
inline _Tpvec v_and(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vand(a, b, vl); \
} \
inline _Tpvec v_or(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vor(a, b, vl); \
} \
inline _Tpvec v_xor(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vxor(a, b, vl); \
} \
inline _Tpvec v_not (const _Tpvec& a) \
{ \
    return vnot(a, vl); \
}

OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_LOGIC_OP(v_int64, VTraits<v_int64>::vlanes())



////////////// Bitwise shifts //////////////

#define OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(_Tpvec, vl) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ \
    return _Tpvec(vsll(a, uint8_t(n), vl)); \
} \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ \
    return _Tpvec(vsrl(a, uint8_t(n), vl)); \
}

#define OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(_Tpvec, vl) \
template<int n> inline _Tpvec v_shl(const _Tpvec& a) \
{ \
    return _Tpvec(vsll(a, uint8_t(n), vl)); \
} \
template<int n> inline _Tpvec v_shr(const _Tpvec& a) \
{ \
    return _Tpvec(vsra(a, uint8_t(n), vl)); \
}

OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_UNSIGNED_SHIFT_OP(v_uint64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_SIGNED_SHIFT_OP(v_int64, VTraits<v_int64>::vlanes())

////////////// Comparison //////////////
// TODO

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
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_min, vfmin, VTraits<v_float64>::vlanes())
OPENCV_HAL_IMPL_RVV_BIN_FUNC(v_float64, v_max, vfmax, VTraits<v_float64>::vlanes())
#endif

////////////// Reduce //////////////

#define OPENCV_HAL_IMPL_RVV_REDUCE_SUM(_Tpvec, _wTpvec, _nwTpvec, scalartype, wsuffix, vl, red) \
inline scalartype v_reduce_sum(const _Tpvec& a)  \
{ \
    _nwTpvec zero = vmv_v_x_##wsuffix##m1(0, vl); \
    _nwTpvec res = vmv_v_x_##wsuffix##m1(0, vl); \
    res = v##red(res, a, zero, vl); \
    return (scalartype)v_get0(res); \
}
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint8, v_uint16, vuint16m1_t, unsigned, u16, VTraits<v_uint8>::vlanes(), wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int8, v_int16, vint16m1_t, int, i16, VTraits<v_int8>::vlanes(), wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint16, v_uint32, vuint32m1_t, unsigned, u32, VTraits<v_uint16>::vlanes(), wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int16, v_int32, vint32m1_t, int, i32, VTraits<v_int16>::vlanes(), wredsum)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_uint32, v_uint64, vuint64m1_t, unsigned, u64, VTraits<v_uint32>::vlanes(), wredsumu)
OPENCV_HAL_IMPL_RVV_REDUCE_SUM(v_int32, v_int64, vint64m1_t, int, i64, VTraits<v_int32>::vlanes(), wredsum)

#define OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(_Tpvec, _wTpvec, _nwTpvec, scalartype, wsuffix, vl) \
inline scalartype v_reduce_sum(const _Tpvec& a)  \
{ \
    _nwTpvec zero = vfmv_v_f_##wsuffix##m1(0, vl); \
    _nwTpvec res = vfmv_v_f_##wsuffix##m1(0, vl); \
    res = vfredosum(res, a, zero, vl); \
    return (scalartype)v_get0(res); \
}
OPENCV_HAL_IMPL_RVV_REDUCE_SUM_FP(v_float32, v_float32, vfloat32m1_t, float, f32, VTraits<v_float32>::vlanes())

#define OPENCV_HAL_IMPL_RVV_REDUCE(_Tpvec, func, scalartype, suffix, vl, red) \
inline scalartype v_reduce_##func(const _Tpvec& a)  \
{ \
    _Tpvec res = _Tpvec(v##red(a, a, a, vl)); \
    return (scalartype)v_get0(res); \
}

OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8, min, uchar, u8, VTraits<v_uint8>::vlanes(), redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8, min, schar, i8, VTraits<v_int8>::vlanes(), redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16, min, ushort, u16, VTraits<v_uint16>::vlanes(), redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16, min, short, i16, VTraits<v_int16>::vlanes(), redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32, min, unsigned, u32, VTraits<v_uint32>::vlanes(), redminu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32, min, int, i32, VTraits<v_int32>::vlanes(), redmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32, min, float, f32, VTraits<v_float32>::vlanes(), fredmin)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint8, max, uchar, u8, VTraits<v_uint8>::vlanes(), redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int8, max, schar, i8, VTraits<v_int8>::vlanes(), redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint16, max, ushort, u16, VTraits<v_uint16>::vlanes(), redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int16, max, short, i16, VTraits<v_int16>::vlanes(), redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_uint32, max, unsigned, u32, VTraits<v_uint32>::vlanes(), redmaxu)
OPENCV_HAL_IMPL_RVV_REDUCE(v_int32, max, int, i32, VTraits<v_int32>::vlanes(), redmax)
OPENCV_HAL_IMPL_RVV_REDUCE(v_float32, max, float, f32, VTraits<v_float32>::vlanes(), fredmax)

//TODO: v_reduce_sum4

////////////// Square-Root //////////////

inline v_float32 v_sqrt(const v_float32& x)
{
    return vfsqrt(x, VTraits<v_float32>::vlanes());
}

inline v_float32 v_invsqrt(const v_float32& x)
{
    v_float32 one = v_setall_f32(1.0f);
    return v_div(one, v_sqrt(x));
}

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_sqrt(const v_float64& x)
{
    return vfsqrt(x, VTraits<v_float64>::vlanes());
}

inline v_float64 v_invsqrt(const v_float64& x)
{
    v_float64 one = v_setall_f64(1.0f);
    return v_div(one, v_sqrt(x));
}
#endif

inline v_float32 v_magnitude(const v_float32& a, const v_float32& b)
{
    v_float32 x = vfmacc(vfmul(a, a, VTraits<v_float32>::vlanes()), b, b, VTraits<v_float32>::vlanes());
    return v_sqrt(x);
}

inline v_float32 v_sqr_magnitude(const v_float32& a, const v_float32& b)
{
    return v_float32(vfmacc(vfmul(a, a, VTraits<v_float32>::vlanes()), b, b, VTraits<v_float32>::vlanes()));
}

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_magnitude(const v_float64& a, const v_float64& b)
{
    v_float64 x = vfmacc(vfmul(a, a, VTraits<v_float64>::vlanes()), b, b, VTraits<v_float64>::vlanes());
    return v_sqrt(x);
}

inline v_float64 v_sqr_magnitude(const v_float64& a, const v_float64& b)
{
    return vfmacc(vfmul(a, a, VTraits<v_float64>::vlanes()), b, b, VTraits<v_float64>::vlanes());
}
#endif

////////////// Multiply-Add //////////////

inline v_float32 v_fma(const v_float32& a, const v_float32& b, const v_float32& c)
{
    return vfmacc(c, a, b, VTraits<v_float32>::vlanes());
}
inline v_int32 v_fma(const v_int32& a, const v_int32& b, const v_int32& c)
{
    return vmacc(c, a, b, VTraits<v_float32>::vlanes());
}

inline v_float32 v_muladd(const v_float32& a, const v_float32& b, const v_float32& c)
{
    return v_fma(a, b, c);
}

inline v_int32 v_muladd(const v_int32& a, const v_int32& b, const v_int32& c)
{
    return v_fma(a, b, c);
}

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_fma(const v_float64& a, const v_float64& b, const v_float64& c)
{
    return vfmacc_vv_f64m1(c, a, b, VTraits<v_float64>::vlanes());
}

inline v_float64 v_muladd(const v_float64& a, const v_float64& b, const v_float64& c)
{
    return v_fma(a, b, c);
}
#endif

////////////// Check all/any //////////////

#define OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(_Tpvec, vl) \
inline bool v_check_all(const _Tpvec& a) \
{ \
    return vcpop(vmslt(a, 0, vl), vl) == vl; \
} \
inline bool v_check_any(const _Tpvec& a) \
{ \
    return vcpop(vmslt(a, 0, vl), vl) != 0; \
}

OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_CHECK_ALLANY(v_int64, VTraits<v_int64>::vlanes())


inline bool v_check_all(const v_uint8& a)
{ return v_check_all(v_reinterpret_as_s8(a)); }
inline bool v_check_any(const v_uint8& a)
{ return v_check_any(v_reinterpret_as_s8(a)); }

inline bool v_check_all(const v_uint16& a)
{ return v_check_all(v_reinterpret_as_s16(a)); }
inline bool v_check_any(const v_uint16& a)
{ return v_check_any(v_reinterpret_as_s16(a)); }

inline bool v_check_all(const v_uint32& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_uint32& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }

inline bool v_check_all(const v_float32& a)
{ return v_check_all(v_reinterpret_as_s32(a)); }
inline bool v_check_any(const v_float32& a)
{ return v_check_any(v_reinterpret_as_s32(a)); }

inline bool v_check_all(const v_uint64& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_uint64& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }

#if CV_SIMD_SCALABLE_64F
inline bool v_check_all(const v_float64& a)
{ return v_check_all(v_reinterpret_as_s64(a)); }
inline bool v_check_any(const v_float64& a)
{ return v_check_any(v_reinterpret_as_s64(a)); }
#endif

////////////// abs //////////////

#define OPENCV_HAL_IMPL_RVV_ABSDIFF(_Tpvec, abs) \
inline _Tpvec v_##abs(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_sub(v_max(a, b), v_min(a, b)); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint8, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint16, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_uint32, absdiff)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float32, absdiff)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_float64, absdiff)
#endif
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int8, absdiffs)
OPENCV_HAL_IMPL_RVV_ABSDIFF(v_int16, absdiffs)

#define OPENCV_HAL_IMPL_RVV_ABSDIFF_S(_Tpvec, _rTpvec, width) \
inline _rTpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vnclipu(vreinterpret_u##width##m2(vwsub_vv(v_max(a, b), v_min(a, b), VTraits<_Tpvec>::vlanes())), 0, VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int8, v_uint8, 16)
OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int16, v_uint16, 32)
OPENCV_HAL_IMPL_RVV_ABSDIFF_S(v_int32, v_uint32, 64)

#define OPENCV_HAL_IMPL_RVV_ABS(_Tprvec, _Tpvec, suffix) \
inline _Tprvec v_abs(const _Tpvec& a) \
{ \
    return v_absdiff(a, v_setzero_##suffix()); \
}

OPENCV_HAL_IMPL_RVV_ABS(v_uint8, v_int8, s8)
OPENCV_HAL_IMPL_RVV_ABS(v_uint16, v_int16, s16)
OPENCV_HAL_IMPL_RVV_ABS(v_uint32, v_int32, s32)
OPENCV_HAL_IMPL_RVV_ABS(v_float32, v_float32, f32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_ABS(v_float64, v_float64, f64)
#endif


#define OPENCV_HAL_IMPL_RVV_REDUCE_SAD(_Tpvec, scalartype) \
inline scalartype v_reduce_sad(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_reduce_sum(v_absdiff(a, b)); \
}

OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int8, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int16, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_uint32, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_int32, unsigned)
OPENCV_HAL_IMPL_RVV_REDUCE_SAD(v_float32, float)

////////////// Select //////////////

#define OPENCV_HAL_IMPL_RVV_SELECT(_Tpvec, vl) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return vmerge(vmsne(mask, 0, vl), b, a, vl); \
}

OPENCV_HAL_IMPL_RVV_SELECT(v_uint8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_uint16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_uint32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_int8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_int16, VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_SELECT(v_int32, VTraits<v_int32>::vlanes())

inline v_float32 v_select(const v_float32& mask, const v_float32& a, const v_float32& b) \
{ \
    return vmerge(vmfne(mask, 0, VTraits<v_float32>::vlanes()), b, a, VTraits<v_float32>::vlanes()); \
}

#if CV_SIMD_SCALABLE_64F
inline v_float64 v_select(const v_float64& mask, const v_float64& a, const v_float64& b) \
{ \
    return vmerge(vmfne(mask, 0, VTraits<v_float64>::vlanes()), b, a, VTraits<v_float64>::vlanes()); \
}
#endif

////////////// Rotate shift //////////////

#define OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(_Tpvec, suffix, vl) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return vslidedown(vmv_v_x_##suffix##m1(0, vl), a, n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return vslideup(vmv_v_x_##suffix##m1(0, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vmv_v_x_##suffix##m1(0, vl), a, n, vl), b, VTraits<_Tpvec>::vlanes() - n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vmv_v_x_##suffix##m1(0, vl), b, VTraits<_Tpvec>::vlanes() - n, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ CV_UNUSED(b); return a; }

OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint8, u8, VTraits<v_uint8>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int8, i8, VTraits<v_int8>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint16, u16, VTraits<v_uint16>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int16, i16,  VTraits<v_int16>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint32, u32, VTraits<v_uint32>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int32, i32, VTraits<v_int32>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_uint64, u64, VTraits<v_uint64>::vlanes())
OPENCV_HAL_IMPL_RVV_ROTATE_INTEGER(v_int64, i64, VTraits<v_int64>::vlanes())

#define OPENCV_HAL_IMPL_RVV_ROTATE_FP(_Tpvec, suffix, vl) \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a) \
{ \
    return vslidedown(vfmv_v_f_##suffix##m1(0, vl), a, n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a) \
{ \
    return vslideup(vfmv_v_f_##suffix##m1(0, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a) \
{ return a; } \
template<int n> inline _Tpvec v_rotate_right(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vfmv_v_f_##suffix##m1(0, vl), a, n, vl), b, VTraits<_Tpvec>::vlanes() - n, vl); \
} \
template<int n> inline _Tpvec v_rotate_left(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(vslidedown(vfmv_v_f_##suffix##m1(0, vl), b, VTraits<_Tpvec>::vlanes() - n, vl), a, n, vl); \
} \
template<> inline _Tpvec v_rotate_left<0>(const _Tpvec& a, const _Tpvec& b) \
{ CV_UNUSED(b); return a; }

OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float32, f32, VTraits<v_float32>::vlanes())
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_ROTATE_FP(v_float64, f64,  VTraits<v_float64>::vlanes())
#endif

////////////// Convert to float //////////////
// TODO

//////////// Broadcast //////////////

#define OPENCV_HAL_IMPL_RVV_BROADCAST(_Tpvec, suffix) \
template<int s = 0> inline _Tpvec v_broadcast_element(_Tpvec v, int i = s) \
{ \
    return v_setall_##suffix(v_extract_n(v, i)); \
} \
inline _Tpvec v_broadcast_highest(_Tpvec v) \
{ \
    return v_setall_##suffix(v_extract_n(v, VTraits<_Tpvec>::vlanes()-1)); \
}

OPENCV_HAL_IMPL_RVV_BROADCAST(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_int32, s32)
OPENCV_HAL_IMPL_RVV_BROADCAST(v_float32, f32)


////////////// Reverse //////////////
#define OPENCV_HAL_IMPL_RVV_REVERSE(_Tpvec, width) \
inline _Tpvec v_reverse(const _Tpvec& a)  \
{ \
    vuint##width##m1_t vidx = vrsub(vid_v_u##width##m1(VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()-1, VTraits<_Tpvec>::vlanes()); \
    return vrgather(a, vidx, VTraits<_Tpvec>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint8, 8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int8, 8)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint16, 16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int16, 16)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint32, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int32, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_float32, 32)
OPENCV_HAL_IMPL_RVV_REVERSE(v_uint64, 64)
OPENCV_HAL_IMPL_RVV_REVERSE(v_int64, 64)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_REVERSE(v_float64, 64)
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


/* void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1)
  a0 = {A1 A2 A3 A4}
  a1 = {B1 B2 B3 B4}
---------------
  {A1 B1 A2 B2} and {A3 B3 A4 B4}
*/

#define OPENCV_HAL_IMPL_RVV_ZIP(_Tpvec, _wTpvec, suffix, width, width2, convert2um2, convert2um1) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) { \
    _wTpvec temp = vreinterpret_##suffix##m2(convert2um2( \
        vor(vzext_vf2(convert2um1(a0), VTraits<_Tpvec>::vlanes()*2), \
            vreinterpret_u##width2##m2(vslide1up(vreinterpret_u##width##m2(vzext_vf2(convert2um1(a1), VTraits<_Tpvec>::vlanes()*2)), 0, VTraits<_Tpvec>::vlanes()*2)), \
            VTraits<_Tpvec>::vlanes()))); \
    b0 = vget_##suffix##m1(temp, 0); \
    b1 = vget_##suffix##m1(temp, 1); \
}
OPENCV_HAL_IMPL_RVV_ZIP(v_uint8, vuint8m2_t, u8, 8, 16, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP(v_int8, vint8m2_t, i8, 8, 16, vreinterpret_u8m2, vreinterpret_u8m1)
OPENCV_HAL_IMPL_RVV_ZIP(v_uint16, vuint16m2_t, u16, 16, 32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP(v_int16, vint16m2_t, i16, 16, 32, vreinterpret_u16m2, vreinterpret_u16m1)
OPENCV_HAL_IMPL_RVV_ZIP(v_uint32, vuint32m2_t, u32, 32, 64, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP(v_int32, vint32m2_t, i32, 32, 64, vreinterpret_u32m2, vreinterpret_u32m1)
OPENCV_HAL_IMPL_RVV_ZIP(v_float32, vfloat32m2_t, f32, 32, 64, vreinterpret_u32m2, vreinterpret_u32m1)

#define OPENCV_HAL_IMPL_RVV_UNPACKS(_Tpvec, width) \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup(a, b, VTraits<_Tpvec>::vlanes()/2, VTraits<_Tpvec>::vlanes());\
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    return vslideup( \
            vslidedown(a, a, VTraits<_Tpvec>::vlanes()/2, VTraits<_Tpvec>::vlanes()), \
            vslidedown(b, b, VTraits<_Tpvec>::vlanes()/2, VTraits<_Tpvec>::vlanes()), \
            VTraits<_Tpvec>::vlanes()/2, \
            VTraits<_Tpvec>::vlanes()); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    c = v_combine_low(a, b); \
    d = v_combine_high(a, b); \
}

OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint8, 8)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int8, 8)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint16, 16)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int16, 16)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_uint32, 32)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_int32, 32)
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float32, 32)
#if CV_SIMD_SCALABLE_64F
OPENCV_HAL_IMPL_RVV_UNPACKS(v_float64, 64)
#endif

static uint64_t idx_interleave_pairs[] = { \
    0x0705060403010200, 0x0f0d0e0c0b090a08, 0x1715161413111210, 0x1f1d1e1c1b191a18, \
    0x2725262423212220, 0x2f2d2e2c2b292a28, 0x3735363433313230, 0x3f3d3e3c3b393a38, \
    0x4745464443414240, 0x4f4d4e4c4b494a48, 0x5755565453515250, 0x5f5d5e5c5b595a58, \
    0x6765666463616260, 0x6f6d6e6c6b696a68, 0x7775767473717270, 0x7f7d7e7c7b797a78};

static uint64_t idx_interleave_quads[] = { \
    0x0703060205010400, 0x0f0b0e0a0d090c08, 0x1713161215111410, 0x1f1b1e1a1d191c18, \
    0x2723262225212420, 0x2f2b2e2a2d292c28, 0x3733363235313430, 0x3f3b3e3a3d393c38, \
    0x4743464245414440, 0x4f4b4e4a4d494c48, 0x5753565255515450, 0x5f5b5e5a5d595c58, \
    0x6763666265616460, 0x6f6b6e6a6d696c68, 0x7773767275717470, 0x7f7b7e7a7d797c78};

#define OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(_Tpvec, func) \
inline _Tpvec v_interleave_##func(const _Tpvec& vec) { \
    CV_CheckLE(VTraits<_Tpvec>::vlanes(), VTraits<_Tpvec>::max_nlanes, "RVV implementation only supports VLEN in the range [128, 1024]"); \
    vuint8m1_t vidx = vundefined_u8m1();\
    vidx = vreinterpret_u8m1(vle64_v_u64m1(idx_interleave_##func, 16)); \
    return vrgather(vec, vidx, VTraits<v_uint8>::vlanes()); \
}
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_uint8, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_int8, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_uint8, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ_NOEXPEND(v_int8, quads)

#define OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(_Tpvec, width, vzext_vfx, func) \
inline _Tpvec v_interleave_##func(const _Tpvec& vec) { \
    CV_CheckLE(VTraits<_Tpvec>::vlanes(), VTraits<_Tpvec>::max_nlanes, "RVV implementation only supports VLEN in the range [128, 1024]"); \
    vuint##width##m1_t vidx = vundefined_u##width##m1();\
    vidx = vget_u##width##m1(vzext_vfx(vreinterpret_u8m1(vle64_v_u64m1(idx_interleave_##func, 16)), VTraits<v_uint8>::vlanes()), 0); \
    return vrgather(vec, vidx, VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint16, 16, vzext_vf2, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int16, 16, vzext_vf2, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint32, 32, vzext_vf4, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int32, 32, vzext_vf4, pairs)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_float32, 32, vzext_vf4, pairs)

OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint16, 16, vzext_vf2, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int16, 16, vzext_vf2, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_uint32, 32, vzext_vf4, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_int32, 32, vzext_vf4, quads)
OPENCV_HAL_IMPL_RVV_INTERLEAVED_PQ(v_float32, 32, vzext_vf4, quads)

////////////// Transpose4x4 //////////////
#define OPENCV_HAL_IMPL_RVV_ZIP4(_Tpvec, _wTpvec, suffix, convert2u, convert) \
static inline void v_zip4(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) { \
    int vl = 4; \
    _wTpvec temp = vreinterpret_##suffix##m2(convert2u( \
        vor(vzext_vf2(convert(a0), vl), \
            vreinterpret_u64m2(vslide1up(vreinterpret_u32m2(vzext_vf2(convert(a1), vl)), 0, vl*2)), \
            vl))); \
    b0 = vget_##suffix##m1(temp, 0); \
    b1 = vget_##suffix##m1(vrgather(temp, vadd(vid_v_u32m2(vl), 4, vl)/*{4,5,6,7} */, vl) ,0); \
}

OPENCV_HAL_IMPL_RVV_ZIP4(v_uint32, vuint32m2_t, u32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_RVV_ZIP4(v_int32, vint32m2_t, i32, vreinterpret_u32m2, vreinterpret_u32m1)
OPENCV_HAL_IMPL_RVV_ZIP4(v_float32, vfloat32m2_t, f32, vreinterpret_u32m2, vreinterpret_u32m1)

#define OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(_Tpvec, suffix) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, const _Tpvec& a2, const _Tpvec& a3, _Tpvec& b0, _Tpvec& b1, _Tpvec& b2, _Tpvec& b3) { \
    _Tpvec t0,t1,t2,t3= vundefined_##suffix##m1(); \
    v_zip4(a0, a2, t0, t2); \
    v_zip4(a1, a3, t1, t3); \
    v_zip4(t0, t1, b0, b1); \
    v_zip4(t2, t3, b2, b3); \
}

OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(v_uint32, u32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(v_int32, i32)
OPENCV_HAL_IMPL_RVV_TRANSPOSE4x4(v_float32, f32)

//////////// PopCount //////////
// TODO

//////////// SignMask ////////////
#define OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(_Tpvec) \
inline int v_signmask(const _Tpvec& a) \
{ \
    uint8_t ans[4] = {0}; \
    vsm(ans, vmslt(a, 0, VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()); \
    return *(reinterpret_cast<int*>(ans)); \
} \
inline int v_scan_forward(const _Tpvec& a) \
{ \
    return (int)vfirst(vmslt(a, 0, VTraits<_Tpvec>::vlanes()), VTraits<_Tpvec>::vlanes()); \
}

OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int8)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int16)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int32)
OPENCV_HAL_IMPL_RVV_SIGNMASK_OP(v_int64)

inline int64 v_signmask(const v_uint8& a)
{ return v_signmask(v_reinterpret_as_s8(a)); }
inline int64 v_signmask(const v_uint16& a)
{ return v_signmask(v_reinterpret_as_s16(a)); }
inline int v_signmask(const v_uint32& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_float32& a)
{ return v_signmask(v_reinterpret_as_s32(a)); }
inline int v_signmask(const v_uint64& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
#if CV_SIMD_SCALABLE_64F
inline int v_signmask(const v_float64& a)
{ return v_signmask(v_reinterpret_as_s64(a)); }
#endif

//////////// Scan forward ////////////
inline int v_scan_forward(const v_uint8& a)
{ return v_scan_forward(v_reinterpret_as_s8(a)); }
inline int v_scan_forward(const v_uint16& a)
{ return v_scan_forward(v_reinterpret_as_s16(a)); }
inline int v_scan_forward(const v_uint32& a)
{ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_float32& a)
{ return v_scan_forward(v_reinterpret_as_s32(a)); }
inline int v_scan_forward(const v_uint64& a)
{ return v_scan_forward(v_reinterpret_as_s64(a)); }
#if CV_SIMD_SCALABLE_64F
inline int v_scan_forward(const v_float64& a)
{ return v_scan_forward(v_reinterpret_as_s64(a)); }
#endif

//////////// Pack triplets ////////////
// TODO


////// FP16 support ///////

inline v_float32 v_load_expand(const float16_t* ptr)
{
    // TODO
    return vundefined_f32m1();
}

////////////// Rounding //////////////
// TODO

//////// Dot Product ////////
// TODO

//////// Fast Dot Product ////////
// TODO

inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

} //namespace cv

#endif //OPENCV_HAL_INTRIN_RVV_SCALABLE_HPP