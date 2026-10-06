// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

#ifndef CV__SIMD_FORWARD
#error "Need to pre-define forward width"
#endif

namespace cv
{

//! @cond IGNORED

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

/** Types **/
#if CV__SIMD_FORWARD == 1024
// [todo] 1024
#error "1024-long ops not implemented yet"
#elif CV__SIMD_FORWARD == 512
// 512
#define __CV_VX(fun)   v512_##fun
#define __CV_V_UINT8   v_uint8x64
#define __CV_V_INT8    v_int8x64
#define __CV_V_UINT16  v_uint16x32
#define __CV_V_INT16   v_int16x32
#define __CV_V_UINT32  v_uint32x16
#define __CV_V_INT32   v_int32x16
#define __CV_V_UINT64  v_uint64x8
#define __CV_V_INT64   v_int64x8
#define __CV_V_FLOAT32 v_float32x16
#define __CV_V_FLOAT64 v_float64x8
struct v_uint8x64;
struct v_int8x64;
struct v_uint16x32;
struct v_int16x32;
struct v_uint32x16;
struct v_int32x16;
struct v_uint64x8;
struct v_int64x8;
struct v_float32x16;
struct v_float64x8;
#elif CV__SIMD_FORWARD == 256
// 256
#define __CV_VX(fun)   v256_##fun
#define __CV_V_UINT8   v_uint8x32
#define __CV_V_INT8    v_int8x32
#define __CV_V_UINT16  v_uint16x16
#define __CV_V_INT16   v_int16x16
#define __CV_V_UINT32  v_uint32x8
#define __CV_V_INT32   v_int32x8
#define __CV_V_UINT64  v_uint64x4
#define __CV_V_INT64   v_int64x4
#define __CV_V_FLOAT32 v_float32x8
#define __CV_V_FLOAT64 v_float64x4
struct v_uint8x32;
struct v_int8x32;
struct v_uint16x16;
struct v_int16x16;
struct v_uint32x8;
struct v_int32x8;
struct v_uint64x4;
struct v_int64x4;
struct v_float32x8;
struct v_float64x4;
#else
// 128
#define __CV_VX(fun)   v_##fun
#define __CV_V_UINT8   v_uint8x16
#define __CV_V_INT8    v_int8x16
#define __CV_V_UINT16  v_uint16x8
#define __CV_V_INT16   v_int16x8
#define __CV_V_UINT32  v_uint32x4
#define __CV_V_INT32   v_int32x4
#define __CV_V_UINT64  v_uint64x2
#define __CV_V_INT64   v_int64x2
#define __CV_V_FLOAT32 v_float32x4
#define __CV_V_FLOAT64 v_float64x2
struct v_uint8x16;
struct v_int8x16;
struct v_uint16x8;
struct v_int16x8;
struct v_uint32x4;
struct v_int32x4;
struct v_uint64x2;
struct v_int64x2;
struct v_float32x4;
struct v_float64x2;
#endif

/** Value reordering **/

// Expansion
void v_expand(const __CV_V_UINT8&,  __CV_V_UINT16&, __CV_V_UINT16&);
void v_expand(const __CV_V_INT8&,   __CV_V_INT16&,  __CV_V_INT16&);
void v_expand(const __CV_V_UINT16&, __CV_V_UINT32&, __CV_V_UINT32&);
void v_expand(const __CV_V_INT16&,  __CV_V_INT32&,  __CV_V_INT32&);
void v_expand(const __CV_V_UINT32&, __CV_V_UINT64&, __CV_V_UINT64&);
void v_expand(const __CV_V_INT32&,  __CV_V_INT64&,  __CV_V_INT64&);
// Low Expansion
__CV_V_UINT16 v_expand_low(const __CV_V_UINT8&);
__CV_V_INT16  v_expand_low(const __CV_V_INT8&);
__CV_V_UINT32 v_expand_low(const __CV_V_UINT16&);
__CV_V_INT32  v_expand_low(const __CV_V_INT16&);
__CV_V_UINT64 v_expand_low(const __CV_V_UINT32&);
__CV_V_INT64  v_expand_low(const __CV_V_INT32&);
// High Expansion
__CV_V_UINT16 v_expand_high(const __CV_V_UINT8&);
__CV_V_INT16  v_expand_high(const __CV_V_INT8&);
__CV_V_UINT32 v_expand_high(const __CV_V_UINT16&);
__CV_V_INT32  v_expand_high(const __CV_V_INT16&);
__CV_V_UINT64 v_expand_high(const __CV_V_UINT32&);
__CV_V_INT64  v_expand_high(const __CV_V_INT32&);
// Load & Low Expansion
__CV_V_UINT16 __CV_VX(load_expand)(const uchar*);
__CV_V_INT16  __CV_VX(load_expand)(const schar*);
__CV_V_UINT32 __CV_VX(load_expand)(const ushort*);
__CV_V_INT32  __CV_VX(load_expand)(const short*);
__CV_V_UINT64 __CV_VX(load_expand)(const uint*);
__CV_V_INT64  __CV_VX(load_expand)(const int*);
// Load lower 8-bit and expand into 32-bit
__CV_V_UINT32 __CV_VX(load_expand_q)(const uchar*);
__CV_V_INT32  __CV_VX(load_expand_q)(const schar*);

// Saturating Pack
__CV_V_UINT8  v_pack(const __CV_V_UINT16&, const __CV_V_UINT16&);
__CV_V_INT8   v_pack(const __CV_V_INT16&,  const __CV_V_INT16&);
__CV_V_UINT16 v_pack(const __CV_V_UINT32&, const __CV_V_UINT32&);
__CV_V_INT16  v_pack(const __CV_V_INT32&,  const __CV_V_INT32&);
// Non-saturating Pack
__CV_V_UINT32 v_pack(const __CV_V_UINT64&, const __CV_V_UINT64&);
__CV_V_INT32  v_pack(const __CV_V_INT64&,  const __CV_V_INT64&);
// Pack signed integers with unsigned saturation
__CV_V_UINT8  v_pack_u(const __CV_V_INT16&, const __CV_V_INT16&);
__CV_V_UINT16 v_pack_u(const __CV_V_INT32&, const __CV_V_INT32&);

/** Arithmetic, bitwise and comparison operations **/

// Non-saturating multiply
#if CV_VSX
template<typename Tvec>
Tvec v_mul_wrap(const Tvec& a, const Tvec& b);
#else
__CV_V_UINT8  v_mul_wrap(const __CV_V_UINT8&,  const __CV_V_UINT8&);
__CV_V_INT8   v_mul_wrap(const __CV_V_INT8&,   const __CV_V_INT8&);
__CV_V_UINT16 v_mul_wrap(const __CV_V_UINT16&, const __CV_V_UINT16&);
__CV_V_INT16  v_mul_wrap(const __CV_V_INT16&,  const __CV_V_INT16&);
#endif

//  Multiply and expand
#if CV_VSX
template<typename Tvec, typename Twvec>
void v_mul_expand(const Tvec& a, const Tvec& b, Twvec& c, Twvec& d);
#else
void v_mul_expand(const __CV_V_UINT8&,  const __CV_V_UINT8&,  __CV_V_UINT16&, __CV_V_UINT16&);
void v_mul_expand(const __CV_V_INT8&,   const __CV_V_INT8&,   __CV_V_INT16&,  __CV_V_INT16&);
void v_mul_expand(const __CV_V_UINT16&, const __CV_V_UINT16&, __CV_V_UINT32&, __CV_V_UINT32&);
void v_mul_expand(const __CV_V_INT16&,  const __CV_V_INT16&,  __CV_V_INT32&,  __CV_V_INT32&);
void v_mul_expand(const __CV_V_UINT32&, const __CV_V_UINT32&, __CV_V_UINT64&, __CV_V_UINT64&);
void v_mul_expand(const __CV_V_INT32&,  const __CV_V_INT32&,  __CV_V_INT64&,  __CV_V_INT64&);
#endif

// Conversions
__CV_V_FLOAT32 v_cvt_f32(const __CV_V_INT32& a);
__CV_V_FLOAT32 v_cvt_f32(const __CV_V_FLOAT64& a);
__CV_V_FLOAT32 v_cvt_f32(const __CV_V_FLOAT64& a, const __CV_V_FLOAT64& b);
__CV_V_FLOAT64 v_cvt_f64(const __CV_V_INT32& a);
__CV_V_FLOAT64 v_cvt_f64_high(const __CV_V_INT32& a);
__CV_V_FLOAT64 v_cvt_f64(const __CV_V_FLOAT32& a);
__CV_V_FLOAT64 v_cvt_f64_high(const __CV_V_FLOAT32& a);
__CV_V_FLOAT64 v_cvt_f64(const __CV_V_INT64& a);

/** Cleanup **/
#undef CV__SIMD_FORWARD
#undef __CV_VX
#undef __CV_V_UINT8
#undef __CV_V_INT8
#undef __CV_V_UINT16
#undef __CV_V_INT16
#undef __CV_V_UINT32
#undef __CV_V_INT32
#undef __CV_V_UINT64
#undef __CV_V_INT64
#undef __CV_V_FLOAT32
#undef __CV_V_FLOAT64

//! @brief Saturating 64->32 pack for backends with no native saturating narrow.
//!
//! Defines v_pack(int64)->int32, v_pack(uint64)->uint32 and v_pack_u(int64)->uint32 plus their
//! _store forms, on top of the backend own truncating narrow (PACK_TRUNC_S / PACK_TRUNC_U).
//!
//! Everything is decided from the two 32-bit halves, so only 32-bit compares and selects are used.
//! A 64-bit ordering compare would not do: at the SSE and NEON baselines v_gt on 64-bit lanes is
//! the sign of (b - a), so it overflows - and answers the wrong way - exactly when the operands sit
//! at opposite ends of the 64-bit range, which is the case a saturating narrow has to get right.
//!
//! SSE, AVX2, AVX-512, NEON and RVV have a native saturating narrow and define these directly.
#define OPENCV_HAL_IMPL_SATURATING_PACK_64(SV64, UV64, SV32, UV32, SETALL_S32, PACK_TRUNC_S, PACK_TRUNC_U) \
inline SV32 v_pack(const SV64& a, const SV64& b) \
{ \
    SV32 lo = PACK_TRUNC_S(a, b); \
    SV32 hi = PACK_TRUNC_S(v_shr<32>(a), v_shr<32>(b)); \
    SV32 fits = v_eq(hi, v_shr<31>(lo));              /* hi == sext(lo) */ \
    SV32 sat  = v_xor(v_shr<31>(hi), SETALL_S32(INT_MAX)); \
    return v_select(fits, lo, sat); \
} \
inline UV32 v_pack(const UV64& a, const UV64& b) \
{ \
    UV32 lo = PACK_TRUNC_U(a, b); \
    UV32 hi = PACK_TRUNC_U(v_shr<32>(a), v_shr<32>(b)); \
    UV32 fits = v_eq(hi, v_reinterpret_as_u32(SETALL_S32(0))); \
    return v_select(fits, lo, v_reinterpret_as_u32(SETALL_S32(-1))); \
} \
inline UV32 v_pack_u(const SV64& a, const SV64& b) \
{ \
    SV32 lo   = PACK_TRUNC_S(a, b); \
    SV32 hi   = PACK_TRUNC_S(v_shr<32>(a), v_shr<32>(b)); \
    SV32 zero = SETALL_S32(0); \
    SV32 fits = v_eq(hi, zero);                       /* value is in [0, UINT32_MAX] */ \
    SV32 sat  = v_gt(hi, zero);                       /* all ones above the range, 0 below it */ \
    return v_reinterpret_as_u32(v_select(fits, lo, sat)); \
} \
inline void v_pack_store(int* ptr, const SV64& a) \
{ v_store_low(ptr, v_pack(a, a)); } \
inline void v_pack_store(unsigned* ptr, const UV64& a) \
{ v_store_low(ptr, v_pack(a, a)); } \
inline void v_pack_u_store(unsigned* ptr, const SV64& a) \
{ v_store_low(ptr, v_pack_u(a, a)); }

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::
