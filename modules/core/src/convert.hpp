// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html


#ifndef SRC_CONVERT_HPP
#define SRC_CONVERT_HPP

#include "opencv2/core/types.hpp"
#include "opencv2/core/hal/intrin.hpp"

namespace cv
{

#if (CV_SIMD || CV_SIMD_SCALABLE)

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

static inline void vx_load_as(const unsigned* ptr, v_float32& a)
{
    v_uint32 delta = vx_setall_u32(0x80000000U);
    v_uint32 ua = vx_load(ptr);
    v_uint32 mask_a = v_and(v_ge(ua, delta), delta);
    v_float32 fmask_a = v_cvt_f32(v_reinterpret_as_s32(mask_a)); // 0.f or (float)(-(1 << 31))
    a = v_cvt_f32(v_reinterpret_as_s32(v_sub(ua, mask_a)));
    // restore the original values
    a = v_sub(a, fmask_a); // subtract 0 or a large negative number
}

static inline void vx_load_as(const float* ptr, v_float32& a)
{ a = vx_load(ptr); }

static inline void vx_load_as(const hfloat* ptr, v_float32& a)
{ a = vx_load_expand(ptr); }

static inline void vx_load_as(const bfloat* ptr, v_float32& a)
{ a = vx_load_expand(ptr); }

static inline void v_store_as(ushort* ptr, const v_float32& a)
{ v_pack_u_store(ptr, v_round(a)); }

static inline void v_store_as(short* ptr, const v_float32& a)
{ v_pack_store(ptr, v_round(a)); }

static inline void v_store_as(int* ptr, const v_float32& a)
{ v_store(ptr, v_round(a)); }

static inline void v_store_as(unsigned* ptr, const v_float32& a)
{
    v_float32 z = vx_setzero_f32();
    v_store(ptr, v_reinterpret_as_u32(v_round(v_max(a, z))));
}

static inline void v_store_as(float* ptr, const v_float32& a)
{ v_store(ptr, a); }

static inline void v_store_as(hfloat* ptr, const v_float32& a)
{ v_pack_store(ptr, a); }

static inline void v_store_as(bfloat* ptr, const v_float32& a)
{ v_pack_store(ptr, a); }

static inline void v_store_as(int64_t* ptr, const v_float32& a)
{
    v_int32 ia = v_round(a);
    v_int64 ia_0, ia_1;
    v_expand(ia, ia_0, ia_1);
    v_store(ptr, ia_0);
    v_store(ptr + VTraits<v_uint64>::vlanes(), ia_1);
}

static inline void v_store_as(uint64_t* ptr, const v_float32& a)
{
    v_int32 ia = v_round(a);
    v_uint64 ia_0, ia_1;
    ia = v_max(ia, vx_setzero_s32());
    v_expand(v_reinterpret_as_u32(ia), ia_0, ia_1);
    v_store(ptr, ia_0);
    v_store(ptr + VTraits<v_uint64>::vlanes(), ia_1);
}

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
{ a = vx_load(ptr); b = vx_load(ptr + VTraits<v_uint16>::vlanes()); }

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
{ a = vx_load(ptr); b = vx_load(ptr + VTraits<v_uint16>::vlanes()); }

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
    b = vx_load(ptr + VTraits<v_int32>::vlanes());
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
    v_int32 ia = vx_load(ptr), ib = vx_load(ptr + VTraits<v_int32>::vlanes());
    a = v_cvt_f32(ia);
    b = v_cvt_f32(ib);
}

static inline void vx_load_pair_as(const int64_t* ptr, v_int32& a, v_int32& b)
{
    const int int64_nlanes = VTraits<v_uint64>::vlanes();
    a = v_pack(vx_load(ptr), vx_load(ptr + int64_nlanes));
    b = v_pack(vx_load(ptr + int64_nlanes*2), vx_load(ptr + int64_nlanes*3));
}

static inline void vx_load_pair_as(const int64_t* ptr, v_uint64& a, v_uint64& b)
{
    v_int64 z = vx_setzero_s64();
    v_int64 ia = vx_load(ptr), ib = vx_load(ptr + VTraits<v_uint64>::vlanes());
    ia = v_and(ia, v_gt(ia, z));
    ib = v_and(ib, v_gt(ib, z));
    a = v_reinterpret_as_u64(ia);
    b = v_reinterpret_as_u64(ib);
}

static inline void vx_load_pair_as(const int64_t* ptr, v_uint32& a, v_uint32& b)
{
    const int nlanes = VTraits<v_uint64>::vlanes();
    v_int64 z = vx_setzero_s64();
    v_int64 ia0 = vx_load(ptr), ia1 = vx_load(ptr + nlanes);
    v_int64 ib0 = vx_load(ptr + nlanes*2), ib1 = vx_load(ptr + nlanes*3);
    ia0 = v_and(ia0, v_gt(ia0, z));
    ia1 = v_and(ia1, v_gt(ia1, z));
    ib0 = v_and(ib0, v_gt(ib0, z));
    ib1 = v_and(ib1, v_gt(ib1, z));
    a = v_pack(v_reinterpret_as_u64(ia0), v_reinterpret_as_u64(ia1));
    b = v_pack(v_reinterpret_as_u64(ib0), v_reinterpret_as_u64(ib1));
}

static inline void vx_load_pair_as(const uint64_t* ptr, v_float32& a, v_float32& b)
{
    const int nlanes = VTraits<v_uint64>::vlanes();
    float buf[VTraits<v_uint64>::max_nlanes*4];
    for (int i = 0; i < nlanes*4; i++) {
        buf[i] = (float)ptr[i];
    }
    a = vx_load(buf);
    b = vx_load(buf + nlanes*2);
}

static inline void vx_load_pair_as(const int64_t* ptr, v_float32& a, v_float32& b)
{
    const int nlanes = VTraits<v_uint64>::vlanes();
    float buf[VTraits<v_uint64>::max_nlanes*4];
    for (int i = 0; i < nlanes*4; i++) {
        buf[i] = (float)ptr[i];
    }
    a = vx_load(buf);
    b = vx_load(buf + nlanes*2);
}

static inline void vx_load_pair_as(const bool* ptr, v_float32& a, v_float32& b)
{
    v_uint16 z = vx_setzero_u16();
    v_uint16 uab = vx_load_expand((const uchar*)ptr);
    uab = v_shr<15>(v_gt(uab, z));
    v_int32 ia, ib;
    v_expand(v_reinterpret_as_s16(uab), ia, ib);
    a = v_cvt_f32(ia);
    b = v_cvt_f32(ib);
}

static inline void vx_load_as(const bool* ptr, v_float32& a)
{
    v_uint32 z = vx_setzero_u32();
    v_uint32 ua = vx_load_expand_q((const uchar*)ptr);
    ua = v_shr<31>(v_gt(ua, z));
    a = v_cvt_f32(v_reinterpret_as_s32(ua));
}

static inline void vx_load_pair_as(const schar* ptr, v_uint32& a, v_uint32& b)
{
    v_int16 ab = v_max(vx_load_expand(ptr), vx_setzero_s16());
    v_expand(v_reinterpret_as_u16(ab), a, b);
}

static inline void vx_load_pair_as(const short* ptr, v_uint32& a, v_uint32& b)
{
    v_int16 ab = v_max(vx_load(ptr), vx_setzero_s16());
    v_expand(v_reinterpret_as_u16(ab), a, b);
}

static inline void vx_load_pair_as(const int* ptr, v_uint32& a, v_uint32& b)
{
    v_int32 z = vx_setzero_s32();
    v_int32 ia = v_max(vx_load(ptr), z);
    v_int32 ib = v_max(vx_load(ptr + VTraits<v_int32>::vlanes()), z);
    a = v_reinterpret_as_u32(ia);
    b = v_reinterpret_as_u32(ib);
}

static inline void vx_load_pair_as(const uint64_t* ptr, v_uint32& a, v_uint32& b)
{
    const int int64_nlanes = VTraits<v_uint64>::vlanes();
    a = v_pack(vx_load(ptr), vx_load(ptr + int64_nlanes));
    b = v_pack(vx_load(ptr + int64_nlanes*2), vx_load(ptr + int64_nlanes*3));
}

static inline void vx_load_pair_as(const uint64_t* ptr, v_int32& a, v_int32& b)
{
    const int int64_nlanes = VTraits<v_uint64>::vlanes();
    v_uint32 ua = v_pack(vx_load(ptr), vx_load(ptr + int64_nlanes));
    v_uint32 ub = v_pack(vx_load(ptr + int64_nlanes*2), vx_load(ptr + int64_nlanes*3));
    a = v_reinterpret_as_s32(ua);
    b = v_reinterpret_as_s32(ub);
}

static inline void vx_load_pair_as(const float* ptr, v_float32& a, v_float32& b)
{ a = vx_load(ptr); b = vx_load(ptr + VTraits<v_float32>::vlanes()); }

static inline void vx_load_pair_as(const hfloat* ptr, v_float32& a, v_float32& b)
{
    a = vx_load_expand(ptr);
    b = vx_load_expand(ptr + VTraits<v_float32>::vlanes());
}

static inline void vx_load_pair_as(const bfloat* ptr, v_float32& a, v_float32& b)
{
    a = vx_load_expand(ptr);
    b = vx_load_expand(ptr + VTraits<v_float32>::vlanes());
}

static inline void vx_load_pair_as(const unsigned* ptr, v_uint32& a, v_uint32& b)
{
    a = vx_load(ptr);
    b = vx_load(ptr + VTraits<v_uint32>::vlanes());
}

static inline void vx_load_pair_as(const unsigned* ptr, v_int32& a, v_int32& b)
{
    a = v_reinterpret_as_s32(vx_load(ptr));
    b = v_reinterpret_as_s32(vx_load(ptr + VTraits<v_uint32>::vlanes()));
}

static inline void vx_load_pair_as(const unsigned* ptr, v_float32& a, v_float32& b)
{
    v_uint32 delta = vx_setall_u32(0x80000000U);
    v_uint32 ua = vx_load(ptr);
    v_uint32 ub = vx_load(ptr + VTraits<v_uint32>::vlanes());
    v_uint32 mask_a = v_and(v_ge(ua, delta), delta), mask_b = v_and(v_ge(ub, delta), delta);
    v_float32 fmask_a = v_cvt_f32(v_reinterpret_as_s32(mask_a)); // 0.f or (float)(-(1 << 31))
    v_float32 fmask_b = v_cvt_f32(v_reinterpret_as_s32(mask_b)); // 0.f or (float)(-(1 << 31))
    a = v_cvt_f32(v_reinterpret_as_s32(v_sub(ua, mask_a)));
    b = v_cvt_f32(v_reinterpret_as_s32(v_sub(ub, mask_b)));
    // restore the original values
    a = v_sub(a, fmask_a); // subtract 0 or a large negative number
    b = v_sub(b, fmask_b); // subtract 0 or a large negative number
}

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
{ v_store(ptr, a); v_store(ptr + VTraits<v_uint16>::vlanes(), b); }

static inline void v_store_pair_as(uchar* ptr, const v_int16& a, const v_int16& b)
{ v_store(ptr, v_pack_u(a, b)); }

static inline void v_store_pair_as(schar* ptr, const v_int16& a, const v_int16& b)
{ v_store(ptr, v_pack(a, b)); }

static inline void v_store_pair_as(short* ptr, const v_int16& a, const v_int16& b)
{ v_store(ptr, a); v_store(ptr + VTraits<v_int16>::vlanes(), b); }

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
    v_store(ptr + VTraits<v_int32>::vlanes(), b);
}

static inline void v_store_pair_as(int64_t* ptr, const v_int32& a, const v_int32& b)
{
    v_int64 q0, q1, q2, q3;
    v_expand(a, q0, q1);
    v_expand(b, q2, q3);
    const int nlanes = VTraits<v_uint64>::vlanes();
    v_store(ptr, q0);
    v_store(ptr + nlanes, q1);
    v_store(ptr + nlanes*2, q2);
    v_store(ptr + nlanes*3, q3);
}

static inline void v_store_pair_as(uchar* ptr, const v_float32& a, const v_float32& b)
{ v_pack_u_store(ptr, v_pack(v_round(a), v_round(b))); }

static inline void v_store_pair_as(schar* ptr, const v_float32& a, const v_float32& b)
{ v_pack_store(ptr, v_pack(v_round(a), v_round(b))); }

static inline void v_store_pair_as(bool* ptr, const v_float32& a, const v_float32& b)
{
    v_float32 z = vx_setzero_f32();
    v_uint32 ma = v_shr<31>(v_reinterpret_as_u32(v_ne(a, z)));
    v_uint32 mb = v_shr<31>(v_reinterpret_as_u32(v_ne(b, z)));
    v_uint16 mab = v_pack(ma, mb);
    v_pack_store((uchar*)ptr, mab);
}

static inline void v_store_pair_as(ushort* ptr, const v_float32& a, const v_float32& b)
{ v_store(ptr, v_pack_u(v_round(a), v_round(b))); }

static inline void v_store_pair_as(short* ptr, const v_float32& a, const v_float32& b)
{ v_store(ptr, v_pack(v_round(a), v_round(b))); }

static inline void v_store_pair_as(int* ptr, const v_float32& a, const v_float32& b)
{
    v_int32 ia = v_round(a), ib = v_round(b);
    v_store(ptr, ia);
    v_store(ptr + VTraits<v_int32>::vlanes(), ib);
}

static inline void v_store_pair_as(float* ptr, const v_float32& a, const v_float32& b)
{ v_store(ptr, a); v_store(ptr + VTraits<v_float32>::vlanes(), b); }

static inline void v_store_pair_as(unsigned* ptr, const v_float32& a, const v_float32& b)
{
    v_int32 z = vx_setzero_s32();
    v_int32 ia = v_max(v_round(a), z);
    v_int32 ib = v_max(v_round(b), z);
    v_store(ptr, v_reinterpret_as_u32(ia));
    v_store(ptr + VTraits<v_int32>::vlanes(), v_reinterpret_as_u32(ib));
}

static inline void v_store_pair_as(uchar* ptr, const v_uint32& a, const v_uint32& b)
{
    v_pack_store(ptr, v_pack(a, b));
}

static inline void v_store_pair_as(ushort* ptr, const v_uint32& a, const v_uint32& b)
{
    v_store(ptr, v_pack(a, b));
}

static inline void v_store_pair_as(unsigned* ptr, const v_uint32& a, const v_uint32& b)
{
    v_store(ptr, a);
    v_store(ptr + VTraits<v_uint32>::vlanes(), b);
}

static inline void v_store_pair_as(uint64_t* ptr, const v_uint32& a, const v_uint32& b)
{
    v_uint64 q0, q1, q2, q3;
    v_expand(a, q0, q1);
    v_expand(b, q2, q3);
    const int nlanes = VTraits<v_uint64>::vlanes();
    v_store(ptr, q0);
    v_store(ptr + nlanes, q1);
    v_store(ptr + nlanes*2, q2);
    v_store(ptr + nlanes*3, q3);
}

static inline void v_store_pair_as(uint64_t* ptr, const v_uint64& a, const v_uint64& b)
{
    v_store(ptr, a);
    v_store(ptr + VTraits<v_uint64>::vlanes(), b);
}

#if (CV_SIMD_64F || CV_SIMD_SCALABLE_64F)

static inline void vx_load_as(const uint64_t* ptr, v_float32& a)
{
    v_float64 a_0 = v_cvt_f64(v_reinterpret_as_s64(vx_load(ptr)));
    v_float64 a_1 = v_cvt_f64(v_reinterpret_as_s64(vx_load(ptr + VTraits<v_uint64>::vlanes())));
    a = v_cvt_f32(a_0, a_1);
}

static inline void vx_load_as(const int64_t* ptr, v_float32& a)
{
    v_float64 a_0 = v_cvt_f64(vx_load(ptr));
    v_float64 a_1 = v_cvt_f64(vx_load(ptr + VTraits<v_uint64>::vlanes()));
    a = v_cvt_f32(a_0, a_1);
}

static inline void vx_load_as(const double* ptr, v_float32& a)
{
    v_float64 v0 = vx_load(ptr), v1 = vx_load(ptr + VTraits<v_float64>::vlanes());
    a = v_cvt_f32(v0, v1);
}

static inline void vx_load_pair_as(const bool* ptr, v_float64& a, v_float64& b)
{
    v_uint32 z = vx_setzero_u32();
    v_uint32 uab = vx_load_expand_q((const uchar*)ptr);
    uab = v_shr<31>(v_gt(uab, z));
    v_float32 fab = v_cvt_f32(v_reinterpret_as_s32(uab));
    a = v_cvt_f64(fab);
    b = v_cvt_f64_high(fab);
}

static inline void vx_load_pair_as(const hfloat* ptr, v_float64& a, v_float64& b)
{
    v_float32 fab = vx_load_expand(ptr);
    a = v_cvt_f64(fab);
    b = v_cvt_f64_high(fab);
}

static inline void vx_load_pair_as(const bfloat* ptr, v_float64& a, v_float64& b)
{
    v_float32 fab = vx_load_expand(ptr);
    a = v_cvt_f64(fab);
    b = v_cvt_f64_high(fab);
}

static inline void vx_load_pair_as(const double* ptr, v_int32& a, v_int32& b)
{
    v_float64 v0 = vx_load(ptr), v1 = vx_load(ptr + VTraits<v_float64>::vlanes());
    v_float64 v2 = vx_load(ptr + VTraits<v_float64>::vlanes()*2), v3 = vx_load(ptr + VTraits<v_float64>::vlanes()*3);
    v_int32 iv0 = v_round(v0), iv1 = v_round(v1);
    v_int32 iv2 = v_round(v2), iv3 = v_round(v3);
    a = v_combine_low(iv0, iv1);
    b = v_combine_low(iv2, iv3);
}

static inline void vx_load_pair_as(const uint64_t* ptr, v_float64& a, v_float64& b)
{
    const int int64_nlanes = VTraits<v_uint64>::vlanes();
    a = v_cvt_f64(v_reinterpret_as_s64(vx_load(ptr)));
    b = v_cvt_f64(v_reinterpret_as_s64(vx_load(ptr + int64_nlanes)));
}

static inline void vx_load_pair_as(const double* ptr, v_float32& a, v_float32& b)
{
    v_float64 v0 = vx_load(ptr), v1 = vx_load(ptr + VTraits<v_float64>::vlanes());
    v_float64 v2 = vx_load(ptr + VTraits<v_float64>::vlanes()*2), v3 = vx_load(ptr + VTraits<v_float64>::vlanes()*3);
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
    b = vx_load(ptr + VTraits<v_float64>::vlanes());
}

static inline void vx_load_pair_as(const int64_t* ptr, v_float64& a, v_float64& b)
{
    a = v_cvt_f64(vx_load(ptr));
    b = v_cvt_f64(vx_load(ptr + VTraits<v_float64>::vlanes()));
}

static inline void vx_load_pair_as(const unsigned* ptr, v_float64& a, v_float64& b)
{
    const int nlanes = VTraits<v_uint64>::vlanes();
    double buf[VTraits<v_uint64>::max_nlanes*2];
    for (int i = 0; i < nlanes*2; i++)
        buf[i] = (double)ptr[i];
    a = vx_load(buf);
    b = vx_load(buf + nlanes);
}

static inline void v_store_as(double* ptr, const v_float32& a)
{
    v_float64 fa0 = v_cvt_f64(a), fa1 = v_cvt_f64_high(a);
    v_store(ptr, fa0);
    v_store(ptr + VTraits<v_float64>::vlanes(), fa1);
}

static inline void v_store_pair_as(double* ptr, const v_int32& a, const v_int32& b)
{
    v_float64 fa0 = v_cvt_f64(a), fa1 = v_cvt_f64_high(a);
    v_float64 fb0 = v_cvt_f64(b), fb1 = v_cvt_f64_high(b);

    v_store(ptr, fa0);
    v_store(ptr + VTraits<v_float64>::vlanes(), fa1);
    v_store(ptr + VTraits<v_float64>::vlanes()*2, fb0);
    v_store(ptr + VTraits<v_float64>::vlanes()*3, fb1);
}

static inline void v_store_pair_as(double* ptr, const v_float32& a, const v_float32& b)
{
    v_float64 fa0 = v_cvt_f64(a), fa1 = v_cvt_f64_high(a);
    v_float64 fb0 = v_cvt_f64(b), fb1 = v_cvt_f64_high(b);

    v_store(ptr, fa0);
    v_store(ptr + VTraits<v_float64>::vlanes(), fa1);
    v_store(ptr + VTraits<v_float64>::vlanes()*2, fb0);
    v_store(ptr + VTraits<v_float64>::vlanes()*3, fb1);
}

static inline void v_store_pair_as(double* ptr, const v_float64& a, const v_float64& b)
{
    v_store(ptr, a);
    v_store(ptr + VTraits<v_float64>::vlanes(), b);
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

static inline void v_store_pair_as(hfloat* ptr, const v_float64& a, const v_float64& b)
{
    v_float32 v = v_cvt_f32(a, b);
    v_pack_store(ptr, v);
}

static inline void v_store_pair_as(uint64_t* ptr, const v_float64& a, const v_float64& b)
{
    v_float64 z = vx_setzero_f64();
    v_int64 ia, ib;
    v_expand(v_round(v_max(a, z), v_max(b, z)), ia, ib);
    v_store(ptr, v_reinterpret_as_u64(ia));
    v_store(ptr + VTraits<v_uint64>::vlanes(), v_reinterpret_as_u64(ib));
}

static inline void v_store_pair_as(int64_t* ptr, const v_float64& a, const v_float64& b)
{
    v_int64 ia, ib;
    v_expand(v_round(a, b), ia, ib);
    v_store(ptr, ia);
    v_store(ptr + VTraits<v_uint64>::vlanes(), ib);
}

static inline void v_store_pair_as(unsigned* ptr, const v_float64& a, const v_float64& b)
{
    v_int32 iab = v_max(v_round(a, b), vx_setzero_s32());
    v_store(ptr, v_reinterpret_as_u32(iab));
}

#else

static inline void vx_load_as(const double* ptr, v_float32& a)
{
    const int VECSZ = VTraits<v_float32>::vlanes();
    float buf[VTraits<v_float32>::max_nlanes*2];

    for( int i = 0; i < VECSZ; i++ )
        buf[i] = saturate_cast<float>(ptr[i]);
    a = vx_load(buf);
}

static inline void vx_load_as(const uint64_t* ptr, v_float32& a)
{
    const int VECSZ = VTraits<v_float32>::vlanes();
    float buf[VTraits<v_float32>::max_nlanes*2];

    for( int i = 0; i < VECSZ; i++ )
        buf[i] = saturate_cast<float>(ptr[i]);
    a = vx_load(buf);
}

static inline void vx_load_as(const int64_t* ptr, v_float32& a)
{
    const int VECSZ = VTraits<v_float32>::vlanes();
    float buf[VTraits<v_float32>::max_nlanes*2];

    for( int i = 0; i < VECSZ; i++ )
        buf[i] = saturate_cast<float>(ptr[i]);
    a = vx_load(buf);
}

template<typename _Tdvec>
static inline void vx_load_pair_as(const double* ptr, _Tdvec& a, _Tdvec& b)
{
    const int VECSZ = VTraits<_Tdvec>::vlanes();
    typename VTraits<_Tdvec>::lane_type buf[VTraits<_Tdvec>::max_nlanes*2];

    for( int i = 0; i < VECSZ*2; i++ )
        buf[i] = saturate_cast<typename VTraits<_Tdvec>::lane_type>(ptr[i]);
    a = vx_load(buf);
    b = vx_load(buf + VECSZ);
}

static inline void v_store_as(double* ptr, const v_float32& a)
{
    const int VECSZ = VTraits<v_float32>::vlanes();
    float buf[VTraits<v_float32>::max_nlanes];

    v_store(buf, a);
    for( int i = 0; i < VECSZ; i++ )
        ptr[i] = (double)buf[i];
}

template<typename _Tsvec>
static inline void v_store_pair_as(double* ptr, const _Tsvec& a, const _Tsvec& b)
{
    const int VECSZ = VTraits<_Tsvec>::vlanes();
    typename VTraits<_Tsvec>::lane_type buf[VTraits<_Tsvec>::max_nlanes*2];

    v_store(buf, a); v_store(buf + VECSZ, b);
    for( int i = 0; i < VECSZ*2; i++ )
        ptr[i] = (double)buf[i];
}

#endif /////////// CV_SIMD_64F || CV_SIMD_SCALABLE_64F

#endif /////////// CV_SIMD || CV_SIMD_SCALABLE

}

#endif // SRC_CONVERT_HPP
