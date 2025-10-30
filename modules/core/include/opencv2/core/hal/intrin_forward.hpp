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

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

} // cv::
