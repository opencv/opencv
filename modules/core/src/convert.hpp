// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#ifndef SRC_CONVERT_HPP
#define SRC_CONVERT_HPP

#include "opencv2/core/types.hpp"

namespace cv
{

#if CV_SIMD

static inline void vx_load_as(const uchar* ptr, v_float32& a)
{ a = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand_q(ptr))); }

static inline void vx_load_as(const schar* ptr, v_float32& a)
{ a = v_cvt_f32(vx_load_expand_q(ptr)); }

static inline void vx_load_as(const ushort* ptr, v_float32& a)
{ a = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(ptr))); }

static inline void vx_load_as(const short* ptr, v_float32& a)
{ a = v_cvt_f32(v_reinterpret_as_s32(vx_load_expand(ptr))); }

static inline void vx_load_as(const int* ptr, v_float32& a)
{ a = v_cvt_f32(vx_load(ptr)); }

static inline void vx_load_as(const float* ptr, v_float32& a)
{ a = vx_load(ptr); }

static inline void vx_load_as(const float16_t* ptr, v_float32& a)
{ a = vx_load_expand(ptr); }

static inline void v_store_as(ushort* ptr, const v_float32& a)
{ v_pack_u_store(ptr, v_round(a)); }

static inline void v_store_as(short* ptr, const v_float32& a)
{ v_pack_store(ptr, v_round(a)); }

static inline void v_store_as(int* ptr, const v_float32& a)
{ v_store(ptr, v_round(a)); }

static inline void v_store_as(float* ptr, const v_float32& a)
{ v_store(ptr, a); }

static inline void v_store_as(float16_t* ptr, const v_float32& a)
{ v_pack_store(ptr, a); }

static inline void vx_load_pair_as(const uchar* ptr, v_uint16& a, v_uint16& b)
{ v_expand(vx_load(ptr), a, b); }

static inline void vx_load_pair_as(const schar* ptr, v_uint16& a, v_uint16& b)
{
    const v_int8 z = vx_setzero_s8();
    v_int16 sa, sb;
    v_expand(v_max(vx_load(ptr), z), sa, sb);
    a = v_reinterpret_as_u16(sa);
    b = v_reinterpret_as_u16(sb);
}

static inline void vx_load_pair_as(const ushort* ptr, v_uint16& a, v_uint16& b)
{ a = vx_load(ptr); b = vx_load(ptr + v_uint16::nlanes); }

static inline void vx_load_pair_as(const uchar* ptr, v_int16& a, v_int16& b)
{
    v_uint16 ua, ub;
    v_expand(vx_load(ptr), ua, ub);
    a = v_reinterpret_as_s16(ua);
    b = v_reinterpret_as_s16(ub);
}

static inline void vx_load_pair_as(const schar* ptr, v_int16& a, v_int16& b)
{ v_expand(vx_load(ptr), a, b); }

static inline void vx_load_pair_as(const short* ptr, v_int16& a, v_int16& b)
{ a = vx_load(ptr); b = vx_load(ptr + v_uint16::nlanes); }

static inline void vx_load_pair_as(const uchar* ptr, v_int32& a, v_int32& b)
{
    v_uint32 ua, ub;
    v_expand(vx_load_expand(ptr), ua, ub);
    a = v_reinterpret_as_s32(ua);
    b = v_reinterpret_as_s32(ub);
}

static inline void vx_load_pair_as(const schar* ptr, v_int32& a, v_int32& b)
{ v_expand(vx_load_expand(ptr), a, b); }

static inline void vx_load_pair_as(const ushort* ptr, v_int32& a, v_int32& b)
{
    v_uint32 ua, ub;
    v_expand(vx_load(ptr), ua, ub);
    a = v_reinterpret_as_s32(ua);
    b = v_reinterpret_as_s32(ub);
}

static inline void vx_load_pair_as(const short* ptr, v_int32& a, v_int32& b)
{
    v_expand(vx_load(ptr), a, b);
}

static inline void vx_load_pair_as(const int* ptr, v_int32& a, v_int32& b)
{
    a = vx_load(ptr);
    b = vx_load(ptr + v_int32::nlanes);
}

static inline void vx_load_pair_as(const uchar* ptr, v_float32& a, v_float32& b)
{
    v_uint32 ua, ub;
    v_expand(vx_load_expand(ptr), ua, ub);
    a = v_cvt_f32(v_reinterpret_as_s32(ua));
    b = v_cvt_f32(v_reinterpret_as_s32(ub));
}

static inline void vx_load_pair_as(const schar* ptr, v_float32& a, v_float32& b)
{
    v_int32 ia, ib;
    v_expand(vx_load_expand(ptr), ia, ib);
    a = v_cvt_f32(ia);
    b = v_cvt_f32(ib);
}

static inline void vx_load_pair_as(const ushort* ptr, v_float32& a, v_float32& b)
{
    v_uint32 ua, ub;
    v_expand(vx_load(ptr), ua, ub);
    a = v_cvt_f32(v_reinterpret_as_s32(ua));
    b = v_cvt_f32(v_reinterpret_as_s32(ub));
}

static inline void vx_load_pair_as(const short* ptr, v_float32& a, v_float32& b)
{
    v_int32 ia, ib;
    v_expand(vx_load(ptr), ia, ib);
    a = v_cvt_f32(ia);
    b = v_cvt_f32(ib);
}

static inline void vx_load_pair_as(const int* ptr, v_float32& a, v_float32& b)
{
    v_int32 ia = vx_load(ptr), ib = vx_load(ptr + v_int32::nlanes);
    a = v_cvt_f32(ia);
    b = v_cvt_f32(ib);
}

static inline void vx_load_pair_as(const float* ptr, v_float32& a, v_float32& b)
{ a = vx_load(ptr); b = vx_load(ptr + v_float32::nlanes); }

//static inline void vx_load_pair_as(const float16_t* ptr, v_float32& a, v_float32& b)
//{
//    a = vx_load_expand(ptr);
//    b = vx_load_expand(ptr + v_float32::nlanes);
//}


static inline void v_store_pair_as(uchar* ptr, const v_uint16& a, const v_uint16& b)
{
    v_store(ptr, v_pack(a, b));
}

static inline void v_store_pair_as(schar* ptr, const v_uint16& a, const v_uint16& b)
{
    const v_uint8 maxval = vx_setall_u8((uchar)std::numeric_limits<schar>::max());
    v_uint8 v = v_pack(a, b);
    v_store(ptr, v_reinterpret_as_s8(v_min(v, maxval)));
}

static inline void v_store_pair_as(ushort* ptr, const v_uint16& a, const v_uint16& b)
{ v_store(ptr, a); v_store(ptr + v_uint16::nlanes, b); }

static inline void v_store_pair_as(uchar* ptr, const v_int16& a, const v_int16& b)
{ v_store(ptr, v_pack_u(a, b)); }

static inline void v_store_pair_as(schar* ptr, const v_int16& a, const v_int16& b)
{ v_store(ptr, v_pack(a, b)); }

static inline void v_store_pair_as(short* ptr, const v_int16& a, const v_int16& b)
{ v_store(ptr, a); v_store(ptr + v_int16::nlanes, b); }

static inline void v_store_pair_as(uchar* ptr, const v_int32& a, const v_int32& b)
{ v_pack_u_store(ptr, v_pack(a, b)); }

static inline void v_store_pair_as(schar* ptr, const v_int32& a, const v_int32& b)
{ v_pack_store(ptr, v_pack(a, b)); }

static inline void v_store_pair_as(ushort* ptr, const v_int32& a, const v_int32& b)
{ v_store(ptr, v_pack_u(a, b)); }

static inline void v_store_pair_as(short* ptr, const v_int32& a, const v_int32& b)
{ v_store(ptr, v_pack(a, b)); }

static inline void v_store_pair_as(int* ptr, const v_int32& a, const v_int32& b)
{
    v_store(ptr, a);
    v_store(ptr + v_int32::nlanes, b);
}

static inline void v_store_pair_as(uchar* ptr, const v_float32& a, const v_float32& b)
{ v_pack_u_store(ptr, v_pack(v_round(a), v_round(b))); }

static inline void v_store_pair_as(schar* ptr, const v_float32& a, const v_float32& b)
{ v_pack_store(ptr, v_pack(v_round(a), v_round(b))); }

static inline void v_store_pair_as(ushort* ptr, const v_float32& a, const v_float32& b)
{ v_store(ptr, v_pack_u(v_round(a), v_round(b))); }

static inline void v_store_pair_as(short* ptr, const v_float32& a, const v_float32& b)
{ v_store(ptr, v_pack(v_round(a), v_round(b))); }

static inline void v_store_pair_as(int* ptr, const v_float32& a, const v_float32& b)
{
    v_int32 ia = v_round(a), ib = v_round(b);
    v_store(ptr, ia);
    v_store(ptr + v_int32::nlanes, ib);
}

static inline void v_store_pair_as(float* ptr, const v_float32& a, const v_float32& b)
{ v_store(ptr, a); v_store(ptr + v_float32::nlanes, b); }

#if CV_SIMD_64F

static inline void vx_load_as(const double* ptr, v_float32& a)
{
    v_float64 v0 = vx_load(ptr), v1 = vx_load(ptr + v_float64::nlanes);
    a = v_cvt_f32(v0, v1);
}

static inline void vx_load_pair_as(const double* ptr, v_int32& a, v_int32& b)
{
    v_float64 v0 = vx_load(ptr), v1 = vx_load(ptr + v_float64::nlanes);
    v_float64 v2 = vx_load(ptr + v_float64::nlanes*2), v3 = vx_load(ptr + v_float64::nlanes*3);
    v_int32 iv0 = v_round(v0), iv1 = v_round(v1);
    v_int32 iv2 = v_round(v2), iv3 = v_round(v3);
    a = v_combine_low(iv0, iv1);
    b = v_combine_low(iv2, iv3);
}

static inline void vx_load_pair_as(const double* ptr, v_float32& a, v_float32& b)
{
    v_float64 v0 = vx_load(ptr), v1 = vx_load(ptr + v_float64::nlanes);
    v_float64 v2 = vx_load(ptr + v_float64::nlanes*2), v3 = vx_load(ptr + v_float64::nlanes*3);
    a = v_cvt_f32(v0, v1);
    b = v_cvt_f32(v2, v3);
}

static inline void vx_load_pair_as(const uchar* ptr, v_float64& a, v_float64& b)
{
    v_int32 v0 = v_reinterpret_as_s32(vx_load_expand_q(ptr));
    a = v_cvt_f64(v0);
    b = v_cvt_f64_high(v0);
}

static inline void vx_load_pair_as(const schar* ptr, v_float64& a, v_float64& b)
{
    v_int32 v0 = vx_load_expand_q(ptr);
    a = v_cvt_f64(v0);
    b = v_cvt_f64_high(v0);
}

static inline void vx_load_pair_as(const ushort* ptr, v_float64& a, v_float64& b)
{
    v_int32 v0 = v_reinterpret_as_s32(vx_load_expand(ptr));
    a = v_cvt_f64(v0);
    b = v_cvt_f64_high(v0);
}

static inline void vx_load_pair_as(const short* ptr, v_float64& a, v_float64& b)
{
    v_int32 v0 = vx_load_expand(ptr);
    a = v_cvt_f64(v0);
    b = v_cvt_f64_high(v0);
}

static inline void vx_load_pair_as(const int* ptr, v_float64& a, v_float64& b)
{
    v_int32 v0 = vx_load(ptr);
    a = v_cvt_f64(v0);
    b = v_cvt_f64_high(v0);
}

static inline void vx_load_pair_as(const float* ptr, v_float64& a, v_float64& b)
{
    v_float32 v0 = vx_load(ptr);
    a = v_cvt_f64(v0);
    b = v_cvt_f64_high(v0);
}

static inline void vx_load_pair_as(const double* ptr, v_float64& a, v_float64& b)
{
    a = vx_load(ptr);
    b = vx_load(ptr + v_float64::nlanes);
}

//static inline void vx_load_pair_as(const float16_t* ptr, v_float64& a, v_float64& b)
//{
//    v_float32 v0 = vx_load_expand(ptr);
//    a = v_cvt_f64(v0);
//    b = v_cvt_f64_high(v0);
//}

static inline void v_store_as(double* ptr, const v_float32& a)
{
    v_float64 fa0 = v_cvt_f64(a), fa1 = v_cvt_f64_high(a);
    v_store(ptr, fa0);
    v_store(ptr + v_float64::nlanes, fa1);
}

static inline void v_store_pair_as(double* ptr, const v_int32& a, const v_int32& b)
{
    v_float64 fa0 = v_cvt_f64(a), fa1 = v_cvt_f64_high(a);
    v_float64 fb0 = v_cvt_f64(b), fb1 = v_cvt_f64_high(b);

    v_store(ptr, fa0);
    v_store(ptr + v_float64::nlanes, fa1);
    v_store(ptr + v_float64::nlanes*2, fb0);
    v_store(ptr + v_float64::nlanes*3, fb1);
}

static inline void v_store_pair_as(double* ptr, const v_float32& a, const v_float32& b)
{
    v_float64 fa0 = v_cvt_f64(a), fa1 = v_cvt_f64_high(a);
    v_float64 fb0 = v_cvt_f64(b), fb1 = v_cvt_f64_high(b);

    v_store(ptr, fa0);
    v_store(ptr + v_float64::nlanes, fa1);
    v_store(ptr + v_float64::nlanes*2, fb0);
    v_store(ptr + v_float64::nlanes*3, fb1);
}

static inline void v_store_pair_as(double* ptr, const v_float64& a, const v_float64& b)
{
    v_store(ptr, a);
    v_store(ptr + v_float64::nlanes, b);
}

static inline void v_store_pair_as(int* ptr, const v_float64& a, const v_float64& b)
{
    v_int32 ia = v_round(a), ib = v_round(b);
    v_store(ptr, v_combine_low(ia, ib));
}

static inline void v_store_pair_as(float* ptr, const v_float64& a, const v_float64& b)
{
    v_float32 v = v_cvt_f32(a, b);
    v_store(ptr, v);
}

//static inline void v_store_pair_as(float16_t* ptr, const v_float64& a, const v_float64& b)
//{
//    v_float32 v = v_cvt_f32(a, b);
//    v_pack_store(ptr, v);
//}

#else

static inline void vx_load_as(const double* ptr, v_float32& a)
{
    const int VECSZ = v_float32::nlanes;
    float buf[VECSZ*2];

    for( int i = 0; i < VECSZ; i++ )
        buf[i] = saturate_cast<float>(ptr[i]);
    a = vx_load(buf);
}

template<typename _Tdvec>
static inline void vx_load_pair_as(const double* ptr, _Tdvec& a, _Tdvec& b)
{
    const int VECSZ = _Tdvec::nlanes;
    typename _Tdvec::lane_type buf[VECSZ*2];

    for( int i = 0; i < VECSZ*2; i++ )
        buf[i] = saturate_cast<typename _Tdvec::lane_type>(ptr[i]);
    a = vx_load(buf);
    b = vx_load(buf + VECSZ);
}

static inline void v_store_as(double* ptr, const v_float32& a)
{
    const int VECSZ = v_float32::nlanes;
    float buf[VECSZ];

    v_store(buf, a);
    for( int i = 0; i < VECSZ; i++ )
        ptr[i] = (double)buf[i];
}

template<typename _Tsvec>
static inline void v_store_pair_as(double* ptr, const _Tsvec& a, const _Tsvec& b)
{
    const int VECSZ = _Tsvec::nlanes;
    typename _Tsvec::lane_type buf[VECSZ*2];

    v_store(buf, a); v_store(buf + VECSZ, b);
    for( int i = 0; i < VECSZ*2; i++ )
        ptr[i] = (double)buf[i];
}

#endif /////////// CV_SIMD_64F

#endif /////////// CV_SIMD

}

#endif // SRC_CONVERT_HPP
