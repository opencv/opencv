/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2015, Itseez Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#ifndef OPENCV_HAL_SSE_HPP
#define OPENCV_HAL_SSE_HPP

#include <algorithm>
#include "opencv2/core/utility.hpp"

#define CV_SIMD128 1
#define CV_SIMD128_64F 1

namespace cv
{

//! @cond IGNORED

struct v_uint8x16
{
    typedef uchar lane_type;
    enum { nlanes = 16 };

    v_uint8x16() {}
    explicit v_uint8x16(__m128i v) : val(v) {}
    v_uint8x16(uchar v0, uchar v1, uchar v2, uchar v3, uchar v4, uchar v5, uchar v6, uchar v7,
               uchar v8, uchar v9, uchar v10, uchar v11, uchar v12, uchar v13, uchar v14, uchar v15)
    {
        val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                            (char)v4, (char)v5, (char)v6, (char)v7,
                            (char)v8, (char)v9, (char)v10, (char)v11,
                            (char)v12, (char)v13, (char)v14, (char)v15);
    }
    uchar get0() const
    {
        return (uchar)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_int8x16
{
    typedef schar lane_type;
    enum { nlanes = 16 };

    v_int8x16() {}
    explicit v_int8x16(__m128i v) : val(v) {}
    v_int8x16(schar v0, schar v1, schar v2, schar v3, schar v4, schar v5, schar v6, schar v7,
              schar v8, schar v9, schar v10, schar v11, schar v12, schar v13, schar v14, schar v15)
    {
        val = _mm_setr_epi8((char)v0, (char)v1, (char)v2, (char)v3,
                            (char)v4, (char)v5, (char)v6, (char)v7,
                            (char)v8, (char)v9, (char)v10, (char)v11,
                            (char)v12, (char)v13, (char)v14, (char)v15);
    }
    schar get0() const
    {
        return (schar)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_uint16x8
{
    typedef ushort lane_type;
    enum { nlanes = 8 };

    v_uint16x8() {}
    explicit v_uint16x8(__m128i v) : val(v) {}
    v_uint16x8(ushort v0, ushort v1, ushort v2, ushort v3, ushort v4, ushort v5, ushort v6, ushort v7)
    {
        val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                             (short)v4, (short)v5, (short)v6, (short)v7);
    }
    ushort get0() const
    {
        return (ushort)_mm_cvtsi128_si32(val);
    }

    __m128i val;
};

struct v_int16x8
{
    typedef short lane_type;
    enum { nlanes = 8 };

    v_int16x8() {}
    explicit v_int16x8(__m128i v) : val(v) {}
    v_int16x8(short v0, short v1, short v2, short v3, short v4, short v5, short v6, short v7)
    {
        val = _mm_setr_epi16((short)v0, (short)v1, (short)v2, (short)v3,
                             (short)v4, (short)v5, (short)v6, (short)v7);
    }
    short get0() const
    {
        return (short)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_uint32x4
{
    typedef unsigned lane_type;
    enum { nlanes = 4 };

    v_uint32x4() {}
    explicit v_uint32x4(__m128i v) : val(v) {}
    v_uint32x4(unsigned v0, unsigned v1, unsigned v2, unsigned v3)
    {
        val = _mm_setr_epi32((int)v0, (int)v1, (int)v2, (int)v3);
    }
    unsigned get0() const
    {
        return (unsigned)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_int32x4
{
    typedef int lane_type;
    enum { nlanes = 4 };

    v_int32x4() {}
    explicit v_int32x4(__m128i v) : val(v) {}
    v_int32x4(int v0, int v1, int v2, int v3)
    {
        val = _mm_setr_epi32(v0, v1, v2, v3);
    }
    int get0() const
    {
        return _mm_cvtsi128_si32(val);
    }
    __m128i val;
};

struct v_float32x4
{
    typedef float lane_type;
    enum { nlanes = 4 };

    v_float32x4() {}
    explicit v_float32x4(__m128 v) : val(v) {}
    v_float32x4(float v0, float v1, float v2, float v3)
    {
        val = _mm_setr_ps(v0, v1, v2, v3);
    }
    float get0() const
    {
        return _mm_cvtss_f32(val);
    }
    __m128 val;
};

struct v_uint64x2
{
    typedef uint64 lane_type;
    enum { nlanes = 2 };

    v_uint64x2() {}
    explicit v_uint64x2(__m128i v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
        val = _mm_setr_epi32((int)v0, (int)(v0 >> 32), (int)v1, (int)(v1 >> 32));
    }
    uint64 get0() const
    {
        int a = _mm_cvtsi128_si32(val);
        int b = _mm_cvtsi128_si32(_mm_srli_epi64(val, 32));
        return (unsigned)a | ((uint64)(unsigned)b << 32);
    }
    __m128i val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    enum { nlanes = 2 };

    v_int64x2() {}
    explicit v_int64x2(__m128i v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
        val = _mm_setr_epi32((int)v0, (int)(v0 >> 32), (int)v1, (int)(v1 >> 32));
    }
    int64 get0() const
    {
        int a = _mm_cvtsi128_si32(val);
        int b = _mm_cvtsi128_si32(_mm_srli_epi64(val, 32));
        return (int64)((unsigned)a | ((uint64)(unsigned)b << 32));
    }
    __m128i val;
};

struct v_float64x2
{
    typedef double lane_type;
    enum { nlanes = 2 };

    v_float64x2() {}
    explicit v_float64x2(__m128d v) : val(v) {}
    v_float64x2(double v0, double v1)
    {
        val = _mm_setr_pd(v0, v1);
    }
    double get0() const
    {
        return _mm_cvtsd_f64(val);
    }
    __m128d val;
};

#if defined(HAVE_FP16)
struct v_float16x4
{
    typedef short lane_type;
    enum { nlanes = 4 };

    v_float16x4() {}
    explicit v_float16x4(__m128i v) : val(v) {}
    v_float16x4(short v0, short v1, short v2, short v3)
    {
        val = _mm_setr_epi16(v0, v1, v2, v3, 0, 0, 0, 0);
    }
    short get0() const
    {
        return (short)_mm_cvtsi128_si32(val);
    }
    __m128i val;
};
#endif

#define OPENCV_HAL_IMPL_SSE_INITVEC(_Tpvec, _Tp, suffix, zsuffix, ssuffix, _Tps, cast) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(_mm_setzero_##zsuffix()); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(_mm_set1_##ssuffix((_Tps)v)); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(cast(a.val)); }

OPENCV_HAL_IMPL_SSE_INITVEC(v_uint8x16, uchar, u8, si128, epi8, char, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int8x16, schar, s8, si128, epi8, char, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_uint16x8, ushort, u16, si128, epi16, short, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int16x8, short, s16, si128, epi16, short, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_uint32x4, unsigned, u32, si128, epi32, int, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int32x4, int, s32, si128, epi32, int, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_float32x4, float, f32, ps, ps, float, _mm_castsi128_ps)
OPENCV_HAL_IMPL_SSE_INITVEC(v_float64x2, double, f64, pd, pd, double, _mm_castsi128_pd)

inline v_uint64x2 v_setzero_u64() { return v_uint64x2(_mm_setzero_si128()); }
inline v_int64x2 v_setzero_s64() { return v_int64x2(_mm_setzero_si128()); }
inline v_uint64x2 v_setall_u64(uint64 val) { return v_uint64x2(val, val); }
inline v_int64x2 v_setall_s64(int64 val) { return v_int64x2(val, val); }

template<typename _Tpvec> inline
v_uint64x2 v_reinterpret_as_u64(const _Tpvec& a) { return v_uint64x2(a.val); }
template<typename _Tpvec> inline
v_int64x2 v_reinterpret_as_s64(const _Tpvec& a) { return v_int64x2(a.val); }
inline v_float32x4 v_reinterpret_as_f32(const v_uint64x2& a)
{ return v_float32x4(_mm_castsi128_ps(a.val)); }
inline v_float32x4 v_reinterpret_as_f32(const v_int64x2& a)
{ return v_float32x4(_mm_castsi128_ps(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_uint64x2& a)
{ return v_float64x2(_mm_castsi128_pd(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_int64x2& a)
{ return v_float64x2(_mm_castsi128_pd(a.val)); }

#define OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(_Tpvec, suffix) \
inline _Tpvec v_reinterpret_as_##suffix(const v_float32x4& a) \
{ return _Tpvec(_mm_castps_si128(a.val)); } \
inline _Tpvec v_reinterpret_as_##suffix(const v_float64x2& a) \
{ return _Tpvec(_mm_castpd_si128(a.val)); }

OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint8x16, u8)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int8x16, s8)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint16x8, u16)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int16x8, s16)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint32x4, u32)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int32x4, s32)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_uint64x2, u64)
OPENCV_HAL_IMPL_SSE_INIT_FROM_FLT(v_int64x2, s64)

inline v_float32x4 v_reinterpret_as_f32(const v_float32x4& a) {return a; }
inline v_float64x2 v_reinterpret_as_f64(const v_float64x2& a) {return a; }
inline v_float32x4 v_reinterpret_as_f32(const v_float64x2& a) {return v_float32x4(_mm_castpd_ps(a.val)); }
inline v_float64x2 v_reinterpret_as_f64(const v_float32x4& a) {return v_float64x2(_mm_castps_pd(a.val)); }

//////////////// PACK ///////////////
inline v_uint8x16 v_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i delta = _mm_set1_epi16(255);
    return v_uint8x16(_mm_packus_epi16(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, delta)),
                                       _mm_subs_epu16(b.val, _mm_subs_epu16(b.val, delta))));
}

inline void v_pack_store(uchar* ptr, const v_uint16x8& a)
{
    __m128i delta = _mm_set1_epi16(255);
    __m128i a1 = _mm_subs_epu16(a.val, _mm_subs_epu16(a.val, delta));
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

inline v_uint8x16 v_pack_u(const v_int16x8& a, const v_int16x8& b)
{ return v_uint8x16(_mm_packus_epi16(a.val, b.val)); }

inline void v_pack_u_store(uchar* ptr, const v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a.val, a.val)); }

template<int n> inline
v_uint8x16 v_rshr_pack(const v_uint16x8& a, const v_uint16x8& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_uint8x16(_mm_packus_epi16(_mm_srli_epi16(_mm_adds_epu16(a.val, delta), n),
                                       _mm_srli_epi16(_mm_adds_epu16(b.val, delta), n)));
}

template<int n> inline
void v_rshr_pack_store(uchar* ptr, const v_uint16x8& a)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srli_epi16(_mm_adds_epu16(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

template<int n> inline
v_uint8x16 v_rshr_pack_u(const v_int16x8& a, const v_int16x8& b)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_uint8x16(_mm_packus_epi16(_mm_srai_epi16(_mm_adds_epi16(a.val, delta), n),
                                       _mm_srai_epi16(_mm_adds_epi16(b.val, delta), n)));
}

template<int n> inline
void v_rshr_pack_u_store(uchar* ptr, const v_int16x8& a)
{
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi16(a1, a1));
}

inline v_int8x16 v_pack(const v_int16x8& a, const v_int16x8& b)
{ return v_int8x16(_mm_packs_epi16(a.val, b.val)); }

inline void v_pack_store(schar* ptr, v_int16x8& a)
{ _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a.val, a.val)); }

template<int n> inline
v_int8x16 v_rshr_pack(const v_int16x8& a, const v_int16x8& b)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    return v_int8x16(_mm_packs_epi16(_mm_srai_epi16(_mm_adds_epi16(a.val, delta), n),
                                     _mm_srai_epi16(_mm_adds_epi16(b.val, delta), n)));
}
template<int n> inline
void v_rshr_pack_store(schar* ptr, const v_int16x8& a)
{
    // we assume that n > 0, and so the shifted 16-bit values can be treated as signed numbers.
    __m128i delta = _mm_set1_epi16((short)(1 << (n-1)));
    __m128i a1 = _mm_srai_epi16(_mm_adds_epi16(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi16(a1, a1));
}


// bit-wise "mask ? a : b"
inline __m128i v_select_si128(__m128i mask, __m128i a, __m128i b)
{
    return _mm_xor_si128(b, _mm_and_si128(_mm_xor_si128(a, b), mask));
}

inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
    __m128i b1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, b.val), maxval32, b.val), delta32);
    __m128i r = _mm_packs_epi32(a1, b1);
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}

inline void v_pack_store(ushort* ptr, const v_uint32x4& a)
{
    __m128i z = _mm_setzero_si128(), maxval32 = _mm_set1_epi32(65535), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(v_select_si128(_mm_cmpgt_epi32(z, a.val), maxval32, a.val), delta32);
    __m128i r = _mm_packs_epi32(a1, a1);
    _mm_storel_epi64((__m128i*)ptr, _mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}

template<int n> inline
v_uint16x8 v_rshr_pack(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i b1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    return v_uint16x8(_mm_sub_epi16(_mm_packs_epi32(a1, b1), _mm_set1_epi16(-32768)));
}

template<int n> inline
void v_rshr_pack_store(ushort* ptr, const v_uint32x4& a)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srli_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

inline v_uint16x8 v_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i r = _mm_packs_epi32(_mm_sub_epi32(a.val, delta32), _mm_sub_epi32(b.val, delta32));
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
}

inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(a.val, delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
}

template<int n> inline
v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    __m128i b1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    __m128i b2 = _mm_sub_epi16(_mm_packs_epi32(b1, b1), _mm_set1_epi16(-32768));
    return v_uint16x8(_mm_unpacklo_epi64(a2, b2));
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

inline v_int16x8 v_pack(const v_int32x4& a, const v_int32x4& b)
{ return v_int16x8(_mm_packs_epi32(a.val, b.val)); }

inline void v_pack_store(short* ptr, const v_int32x4& a)
{
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a.val, a.val));
}

template<int n> inline
v_int16x8 v_rshr_pack(const v_int32x4& a, const v_int32x4& b)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    return v_int16x8(_mm_packs_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n),
                                     _mm_srai_epi32(_mm_add_epi32(b.val, delta), n)));
}

template<int n> inline
void v_rshr_pack_store(short* ptr, const v_int32x4& a)
{
    __m128i delta = _mm_set1_epi32(1 << (n-1));
    __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packs_epi32(a1, a1));
}


// [a0 0 | b0 0]  [a1 0 | b1 0]
inline v_uint32x4 v_pack(const v_uint64x2& a, const v_uint64x2& b)
{
    __m128i v0 = _mm_unpacklo_epi32(a.val, b.val); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a.val, b.val); // b0 b1 0 0
    return v_uint32x4(_mm_unpacklo_epi32(v0, v1));
}

inline void v_pack_store(unsigned* ptr, const v_uint64x2& a)
{
    __m128i a1 = _mm_shuffle_epi32(a.val, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a1);
}

// [a0 0 | b0 0]  [a1 0 | b1 0]
inline v_int32x4 v_pack(const v_int64x2& a, const v_int64x2& b)
{
    __m128i v0 = _mm_unpacklo_epi32(a.val, b.val); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a.val, b.val); // b0 b1 0 0
    return v_int32x4(_mm_unpacklo_epi32(v0, v1));
}

inline void v_pack_store(int* ptr, const v_int64x2& a)
{
    __m128i a1 = _mm_shuffle_epi32(a.val, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a1);
}

template<int n> inline
v_uint32x4 v_rshr_pack(const v_uint64x2& a, const v_uint64x2& b)
{
    uint64 delta = (uint64)1 << (n-1);
    v_uint64x2 delta2(delta, delta);
    __m128i a1 = _mm_srli_epi64(_mm_add_epi64(a.val, delta2.val), n);
    __m128i b1 = _mm_srli_epi64(_mm_add_epi64(b.val, delta2.val), n);
    __m128i v0 = _mm_unpacklo_epi32(a1, b1); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a1, b1); // b0 b1 0 0
    return v_uint32x4(_mm_unpacklo_epi32(v0, v1));
}

template<int n> inline
void v_rshr_pack_store(unsigned* ptr, const v_uint64x2& a)
{
    uint64 delta = (uint64)1 << (n-1);
    v_uint64x2 delta2(delta, delta);
    __m128i a1 = _mm_srli_epi64(_mm_add_epi64(a.val, delta2.val), n);
    __m128i a2 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

inline __m128i v_sign_epi64(__m128i a)
{
    return _mm_shuffle_epi32(_mm_srai_epi32(a, 31), _MM_SHUFFLE(3, 3, 1, 1)); // x m0 | x m1
}

inline __m128i v_srai_epi64(__m128i a, int imm)
{
    __m128i smask = v_sign_epi64(a);
    return _mm_xor_si128(_mm_srli_epi64(_mm_xor_si128(a, smask), imm), smask);
}

template<int n> inline
v_int32x4 v_rshr_pack(const v_int64x2& a, const v_int64x2& b)
{
    int64 delta = (int64)1 << (n-1);
    v_int64x2 delta2(delta, delta);
    __m128i a1 = v_srai_epi64(_mm_add_epi64(a.val, delta2.val), n);
    __m128i b1 = v_srai_epi64(_mm_add_epi64(b.val, delta2.val), n);
    __m128i v0 = _mm_unpacklo_epi32(a1, b1); // a0 a1 0 0
    __m128i v1 = _mm_unpackhi_epi32(a1, b1); // b0 b1 0 0
    return v_int32x4(_mm_unpacklo_epi32(v0, v1));
}

template<int n> inline
void v_rshr_pack_store(int* ptr, const v_int64x2& a)
{
    int64 delta = (int64)1 << (n-1);
    v_int64x2 delta2(delta, delta);
    __m128i a1 = v_srai_epi64(_mm_add_epi64(a.val, delta2.val), n);
    __m128i a2 = _mm_shuffle_epi32(a1, _MM_SHUFFLE(0, 2, 2, 0));
    _mm_storel_epi64((__m128i*)ptr, a2);
}

inline v_float32x4 v_matmul(const v_float32x4& v, const v_float32x4& m0,
                            const v_float32x4& m1, const v_float32x4& m2,
                            const v_float32x4& m3)
{
    __m128 v0 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(0, 0, 0, 0)), m0.val);
    __m128 v1 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(1, 1, 1, 1)), m1.val);
    __m128 v2 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(2, 2, 2, 2)), m2.val);
    __m128 v3 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(3, 3, 3, 3)), m3.val);

    return v_float32x4(_mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, v3)));
}


#define OPENCV_HAL_IMPL_SSE_BIN_OP(bin_op, _Tpvec, intrin) \
    inline _Tpvec operator bin_op (const _Tpvec& a, const _Tpvec& b) \
    { \
        return _Tpvec(intrin(a.val, b.val)); \
    } \
    inline _Tpvec& operator bin_op##= (_Tpvec& a, const _Tpvec& b) \
    { \
        a.val = intrin(a.val, b.val); \
        return a; \
    }

OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint8x16, _mm_adds_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint8x16, _mm_subs_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int8x16, _mm_adds_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int8x16, _mm_subs_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint16x8, _mm_adds_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint16x8, _mm_subs_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_uint16x8, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int16x8, _mm_adds_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int16x8, _mm_subs_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_int16x8, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint32x4, _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint32x4, _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int32x4, _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int32x4, _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_float32x4, _mm_add_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_float32x4, _mm_sub_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_float32x4, _mm_mul_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(/, v_float32x4, _mm_div_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_float64x2, _mm_add_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_float64x2, _mm_sub_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(*, v_float64x2, _mm_mul_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(/, v_float64x2, _mm_div_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_uint64x2, _mm_add_epi64)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_uint64x2, _mm_sub_epi64)
OPENCV_HAL_IMPL_SSE_BIN_OP(+, v_int64x2, _mm_add_epi64)
OPENCV_HAL_IMPL_SSE_BIN_OP(-, v_int64x2, _mm_sub_epi64)

inline v_uint32x4 operator * (const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i c0 = _mm_mul_epu32(a.val, b.val);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    __m128i d0 = _mm_unpacklo_epi32(c0, c1);
    __m128i d1 = _mm_unpackhi_epi32(c0, c1);
    return v_uint32x4(_mm_unpacklo_epi64(d0, d1));
}
inline v_int32x4 operator * (const v_int32x4& a, const v_int32x4& b)
{
    __m128i c0 = _mm_mul_epu32(a.val, b.val);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    __m128i d0 = _mm_unpacklo_epi32(c0, c1);
    __m128i d1 = _mm_unpackhi_epi32(c0, c1);
    return v_int32x4(_mm_unpacklo_epi64(d0, d1));
}
inline v_uint32x4& operator *= (v_uint32x4& a, const v_uint32x4& b)
{
    a = a * b;
    return a;
}
inline v_int32x4& operator *= (v_int32x4& a, const v_int32x4& b)
{
    a = a * b;
    return a;
}

inline void v_mul_expand(const v_int16x8& a, const v_int16x8& b,
                         v_int32x4& c, v_int32x4& d)
{
    __m128i v0 = _mm_mullo_epi16(a.val, b.val);
    __m128i v1 = _mm_mulhi_epi16(a.val, b.val);
    c.val = _mm_unpacklo_epi16(v0, v1);
    d.val = _mm_unpackhi_epi16(v0, v1);
}

inline void v_mul_expand(const v_uint16x8& a, const v_uint16x8& b,
                         v_uint32x4& c, v_uint32x4& d)
{
    __m128i v0 = _mm_mullo_epi16(a.val, b.val);
    __m128i v1 = _mm_mulhi_epu16(a.val, b.val);
    c.val = _mm_unpacklo_epi16(v0, v1);
    d.val = _mm_unpackhi_epi16(v0, v1);
}

inline void v_mul_expand(const v_uint32x4& a, const v_uint32x4& b,
                         v_uint64x2& c, v_uint64x2& d)
{
    __m128i c0 = _mm_mul_epu32(a.val, b.val);
    __m128i c1 = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    c.val = _mm_unpacklo_epi64(c0, c1);
    d.val = _mm_unpackhi_epi64(c0, c1);
}

inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{
    return v_int32x4(_mm_madd_epi16(a.val, b.val));
}

#define OPENCV_HAL_IMPL_SSE_LOGIC_OP(_Tpvec, suffix, not_const) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(&, _Tpvec, _mm_and_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(|, _Tpvec, _mm_or_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(^, _Tpvec, _mm_xor_##suffix) \
    inline _Tpvec operator ~ (const _Tpvec& a) \
    { \
        return _Tpvec(_mm_xor_##suffix(a.val, not_const)); \
    }

OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint8x16, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int8x16, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint16x8, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int16x8, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint32x4, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int32x4, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_uint64x2, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_int64x2, si128, _mm_set1_epi32(-1))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_float32x4, ps, _mm_castsi128_ps(_mm_set1_epi32(-1)))
OPENCV_HAL_IMPL_SSE_LOGIC_OP(v_float64x2, pd, _mm_castsi128_pd(_mm_set1_epi32(-1)))

inline v_float32x4 v_sqrt(const v_float32x4& x)
{ return v_float32x4(_mm_sqrt_ps(x.val)); }

inline v_float32x4 v_invsqrt(const v_float32x4& x)
{
    static const __m128 _0_5 = _mm_set1_ps(0.5f), _1_5 = _mm_set1_ps(1.5f);
    __m128 t = x.val;
    __m128 h = _mm_mul_ps(t, _0_5);
    t = _mm_rsqrt_ps(t);
    t = _mm_mul_ps(t, _mm_sub_ps(_1_5, _mm_mul_ps(_mm_mul_ps(t, t), h)));
    return v_float32x4(t);
}

inline v_float64x2 v_sqrt(const v_float64x2& x)
{ return v_float64x2(_mm_sqrt_pd(x.val)); }

inline v_float64x2 v_invsqrt(const v_float64x2& x)
{
    static const __m128d v_1 = _mm_set1_pd(1.);
    return v_float64x2(_mm_div_pd(v_1, _mm_sqrt_pd(x.val)));
}

#define OPENCV_HAL_IMPL_SSE_ABS_INT_FUNC(_Tpuvec, _Tpsvec, func, suffix, subWidth) \
inline _Tpuvec v_abs(const _Tpsvec& x) \
{ return _Tpuvec(_mm_##func##_ep##suffix(x.val, _mm_sub_ep##subWidth(_mm_setzero_si128(), x.val))); }

OPENCV_HAL_IMPL_SSE_ABS_INT_FUNC(v_uint8x16, v_int8x16, min, u8, i8)
OPENCV_HAL_IMPL_SSE_ABS_INT_FUNC(v_uint16x8, v_int16x8, max, i16, i16)
inline v_uint32x4 v_abs(const v_int32x4& x)
{
    __m128i s = _mm_srli_epi32(x.val, 31);
    __m128i f = _mm_srai_epi32(x.val, 31);
    return v_uint32x4(_mm_add_epi32(_mm_xor_si128(x.val, f), s));
}
inline v_float32x4 v_abs(const v_float32x4& x)
{ return v_float32x4(_mm_and_ps(x.val, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff)))); }
inline v_float64x2 v_abs(const v_float64x2& x)
{
    return v_float64x2(_mm_and_pd(x.val,
        _mm_castsi128_pd(_mm_srli_epi64(_mm_set1_epi32(-1), 1))));
}

// TODO: exp, log, sin, cos

#define OPENCV_HAL_IMPL_SSE_BIN_FUNC(_Tpvec, func, intrin) \
inline _Tpvec func(const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(intrin(a.val, b.val)); \
}

OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_min, _mm_min_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_max, _mm_max_epu8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_min, _mm_min_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_max, _mm_max_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float32x4, v_min, _mm_min_ps)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float32x4, v_max, _mm_max_ps)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float64x2, v_min, _mm_min_pd)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_float64x2, v_max, _mm_max_pd)

inline v_int8x16 v_min(const v_int8x16& a, const v_int8x16& b)
{
    __m128i delta = _mm_set1_epi8((char)-128);
    return v_int8x16(_mm_xor_si128(delta, _mm_min_epu8(_mm_xor_si128(a.val, delta),
                                                       _mm_xor_si128(b.val, delta))));
}
inline v_int8x16 v_max(const v_int8x16& a, const v_int8x16& b)
{
    __m128i delta = _mm_set1_epi8((char)-128);
    return v_int8x16(_mm_xor_si128(delta, _mm_max_epu8(_mm_xor_si128(a.val, delta),
                                                       _mm_xor_si128(b.val, delta))));
}
inline v_uint16x8 v_min(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, b.val)));
}
inline v_uint16x8 v_max(const v_uint16x8& a, const v_uint16x8& b)
{
    return v_uint16x8(_mm_adds_epu16(_mm_subs_epu16(a.val, b.val), b.val));
}
inline v_uint32x4 v_min(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i delta = _mm_set1_epi32((int)0x80000000);
    __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
    return v_uint32x4(v_select_si128(mask, b.val, a.val));
}
inline v_uint32x4 v_max(const v_uint32x4& a, const v_uint32x4& b)
{
    __m128i delta = _mm_set1_epi32((int)0x80000000);
    __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
    return v_uint32x4(v_select_si128(mask, a.val, b.val));
}
inline v_int32x4 v_min(const v_int32x4& a, const v_int32x4& b)
{
    return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), b.val, a.val));
}
inline v_int32x4 v_max(const v_int32x4& a, const v_int32x4& b)
{
    return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), a.val, b.val));
}

#define OPENCV_HAL_IMPL_SSE_INT_CMP_OP(_Tpuvec, _Tpsvec, suffix, sbit) \
inline _Tpuvec operator == (const _Tpuvec& a, const _Tpuvec& b) \
{ return _Tpuvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpuvec operator != (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpuvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec operator == (const _Tpsvec& a, const _Tpsvec& b) \
{ return _Tpsvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpsvec operator != (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpuvec operator < (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask))); \
} \
inline _Tpuvec operator > (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask))); \
} \
inline _Tpuvec operator <= (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpuvec operator >= (const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpsvec operator < (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(b.val, a.val)); \
} \
inline _Tpsvec operator > (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(a.val, b.val)); \
} \
inline _Tpsvec operator <= (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec operator >= (const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(b.val, a.val), not_mask)); \
}

OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint8x16, v_int8x16, epi8, (char)-128)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint16x8, v_int16x8, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint32x4, v_int32x4, epi32, (int)0x80000000)

#define OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(_Tpvec, suffix) \
inline _Tpvec operator == (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpvec operator != (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpneq_##suffix(a.val, b.val)); } \
inline _Tpvec operator < (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmplt_##suffix(a.val, b.val)); } \
inline _Tpvec operator > (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpgt_##suffix(a.val, b.val)); } \
inline _Tpvec operator <= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmple_##suffix(a.val, b.val)); } \
inline _Tpvec operator >= (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpge_##suffix(a.val, b.val)); }

OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(v_float32x4, ps)
OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(v_float64x2, pd)

OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_sub_wrap, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_sub_wrap, _mm_sub_epi16)

#define OPENCV_HAL_IMPL_SSE_ABSDIFF_8_16(_Tpuvec, _Tpsvec, bits, smask32) \
inline _Tpuvec v_absdiff(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    return _Tpuvec(_mm_add_epi##bits(_mm_subs_epu##bits(a.val, b.val), _mm_subs_epu##bits(b.val, a.val))); \
} \
inline _Tpuvec v_absdiff(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i smask = _mm_set1_epi32(smask32); \
    __m128i a1 = _mm_xor_si128(a.val, smask); \
    __m128i b1 = _mm_xor_si128(b.val, smask); \
    return _Tpuvec(_mm_add_epi##bits(_mm_subs_epu##bits(a1, b1), _mm_subs_epu##bits(b1, a1))); \
}

OPENCV_HAL_IMPL_SSE_ABSDIFF_8_16(v_uint8x16, v_int8x16, 8, (int)0x80808080)
OPENCV_HAL_IMPL_SSE_ABSDIFF_8_16(v_uint16x8, v_int16x8, 16, (int)0x80008000)

inline v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b)
{
    return v_max(a, b) - v_min(a, b);
}

inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{
    __m128i d = _mm_sub_epi32(a.val, b.val);
    __m128i m = _mm_cmpgt_epi32(b.val, a.val);
    return v_uint32x4(_mm_sub_epi32(_mm_xor_si128(d, m), m));
}

#define OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(_Tpvec, _Tp, _Tpreg, suffix, absmask_vec) \
inline _Tpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg absmask = _mm_castsi128_##suffix(absmask_vec); \
    return _Tpvec(_mm_and_##suffix(_mm_sub_##suffix(a.val, b.val), absmask)); \
} \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg res = _mm_add_##suffix(_mm_mul_##suffix(a.val, a.val), _mm_mul_##suffix(b.val, b.val)); \
    return _Tpvec(_mm_sqrt_##suffix(res)); \
} \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg res = _mm_add_##suffix(_mm_mul_##suffix(a.val, a.val), _mm_mul_##suffix(b.val, b.val)); \
    return _Tpvec(res); \
} \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{ \
    return _Tpvec(_mm_add_##suffix(_mm_mul_##suffix(a.val, b.val), c.val)); \
}

OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(v_float32x4, float, __m128, ps, _mm_set1_epi32((int)0x7fffffff))
OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(v_float64x2, double, __m128d, pd, _mm_srli_epi64(_mm_set1_epi32(-1), 1))

#define OPENCV_HAL_IMPL_SSE_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, srai) \
inline _Tpuvec operator << (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(_mm_slli_##suffix(a.val, imm)); \
} \
inline _Tpsvec operator << (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(_mm_slli_##suffix(a.val, imm)); \
} \
inline _Tpuvec operator >> (const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(_mm_srli_##suffix(a.val, imm)); \
} \
inline _Tpsvec operator >> (const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(srai(a.val, imm)); \
} \
template<int imm> \
inline _Tpuvec v_shl(const _Tpuvec& a) \
{ \
    return _Tpuvec(_mm_slli_##suffix(a.val, imm)); \
} \
template<int imm> \
inline _Tpsvec v_shl(const _Tpsvec& a) \
{ \
    return _Tpsvec(_mm_slli_##suffix(a.val, imm)); \
} \
template<int imm> \
inline _Tpuvec v_shr(const _Tpuvec& a) \
{ \
    return _Tpuvec(_mm_srli_##suffix(a.val, imm)); \
} \
template<int imm> \
inline _Tpsvec v_shr(const _Tpsvec& a) \
{ \
    return _Tpsvec(srai(a.val, imm)); \
}

OPENCV_HAL_IMPL_SSE_SHIFT_OP(v_uint16x8, v_int16x8, epi16, _mm_srai_epi16)
OPENCV_HAL_IMPL_SSE_SHIFT_OP(v_uint32x4, v_int32x4, epi32, _mm_srai_epi32)
OPENCV_HAL_IMPL_SSE_SHIFT_OP(v_uint64x2, v_int64x2, epi64, v_srai_epi64)

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(_Tpvec, _Tp) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(_mm_loadu_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(_mm_load_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return _Tpvec(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)ptr0), \
                                     _mm_loadl_epi64((const __m128i*)ptr1))); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ _mm_storeu_si128((__m128i*)ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ _mm_store_si128((__m128i*)ptr, a.val); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, a.val); } \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a.val, a.val)); }

OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint8x16, uchar)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int8x16, schar)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint16x8, ushort)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int16x8, short)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint32x4, unsigned)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int32x4, int)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_uint64x2, uint64)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(v_int64x2, int64)

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(_Tpvec, _Tp, suffix) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(_mm_loadu_##suffix(ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(_mm_load_##suffix(ptr)); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return _Tpvec(_mm_castsi128_##suffix( \
        _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)ptr0), \
                           _mm_loadl_epi64((const __m128i*)ptr1)))); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ _mm_storeu_##suffix(ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ _mm_store_##suffix(ptr, a.val); } \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, _mm_cast##suffix##_si128(a.val)); } \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    __m128i a1 = _mm_cast##suffix##_si128(a.val); \
    _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a1, a1)); \
}

OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(v_float32x4, float, ps)
OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(v_float64x2, double, pd)

#if defined(HAVE_FP16)
inline v_float16x4 v_load_f16(const short* ptr)
{ return v_float16x4(_mm_loadl_epi64((const __m128i*)ptr)); }
inline void v_store_f16(short* ptr, v_float16x4& a)
{ _mm_storel_epi64((__m128i*)ptr, a.val); }
#endif

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(_Tpvec, scalartype, func, suffix, sbit) \
inline scalartype v_reduce_##func(const v_##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
    return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype v_reduce_##func(const v_u##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    __m128i smask = _mm_set1_epi16(sbit); \
    val = _mm_xor_si128(val, smask); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_##suffix(val, _mm_srli_si128(val,2)); \
    return (unsigned scalartype)(_mm_cvtsi128_si32(val) ^  sbit); \
}
#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_8_SUM(_Tpvec, scalartype, suffix) \
inline scalartype v_reduce_sum(const v_##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 8)); \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 4)); \
    val = _mm_adds_epi##suffix(val, _mm_srli_si128(val, 2)); \
    return (scalartype)_mm_cvtsi128_si32(val); \
} \
inline unsigned scalartype v_reduce_sum(const v_u##_Tpvec& a) \
{ \
    __m128i val = a.val; \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 8)); \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 4)); \
    val = _mm_adds_epu##suffix(val, _mm_srli_si128(val, 2)); \
    return (unsigned scalartype)_mm_cvtsi128_si32(val); \
}
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, max, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, min, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8_SUM(int16x8, short, 16)

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(_Tpvec, scalartype, func, scalar_func) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    scalartype CV_DECL_ALIGNED(16) buf[4]; \
    v_store_aligned(buf, a); \
    scalartype s0 = scalar_func(buf[0], buf[1]); \
    scalartype s1 = scalar_func(buf[2], buf[3]); \
    return scalar_func(s0, s1); \
}

OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, sum, OPENCV_HAL_ADD)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, sum, OPENCV_HAL_ADD)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, sum, OPENCV_HAL_ADD)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, min, std::min)

#define OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(_Tpvec, suffix, pack_op, and_op, signmask, allmask) \
inline int v_signmask(const _Tpvec& a) \
{ \
    return and_op(_mm_movemask_##suffix(pack_op(a.val)), signmask); \
} \
inline bool v_check_all(const _Tpvec& a) \
{ return and_op(_mm_movemask_##suffix(a.val), allmask) == allmask; } \
inline bool v_check_any(const _Tpvec& a) \
{ return and_op(_mm_movemask_##suffix(a.val), allmask) != 0; }

#define OPENCV_HAL_PACKS(a) _mm_packs_epi16(a, a)
inline __m128i v_packq_epi32(__m128i a)
{
    __m128i b = _mm_packs_epi32(a, a);
    return _mm_packs_epi16(b, b);
}

OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 65535, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int16x8, epi8, OPENCV_HAL_PACKS, OPENCV_HAL_AND, 255, (int)0xaaaa)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint32x4, epi8, v_packq_epi32, OPENCV_HAL_AND, 15, (int)0x8888)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int32x4, epi8, v_packq_epi32, OPENCV_HAL_AND, 15, (int)0x8888)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float32x4, ps, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 15, 15)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float64x2, pd, OPENCV_HAL_NOP, OPENCV_HAL_1ST, 3, 3)

#define OPENCV_HAL_IMPL_SSE_SELECT(_Tpvec, suffix) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(_mm_xor_##suffix(b.val, _mm_and_##suffix(_mm_xor_##suffix(b.val, a.val), mask.val))); \
}

OPENCV_HAL_IMPL_SSE_SELECT(v_uint8x16, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_int8x16, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint16x8, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_int16x8, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint32x4, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_int32x4, si128)
// OPENCV_HAL_IMPL_SSE_SELECT(v_uint64x2, si128)
// OPENCV_HAL_IMPL_SSE_SELECT(v_int64x2, si128)
OPENCV_HAL_IMPL_SSE_SELECT(v_float32x4, ps)
OPENCV_HAL_IMPL_SSE_SELECT(v_float64x2, pd)

#define OPENCV_HAL_IMPL_SSE_EXPAND(_Tpuvec, _Tpwuvec, _Tpu, _Tpsvec, _Tpwsvec, _Tps, suffix, wsuffix, shift) \
inline void v_expand(const _Tpuvec& a, _Tpwuvec& b0, _Tpwuvec& b1) \
{ \
    __m128i z = _mm_setzero_si128(); \
    b0.val = _mm_unpacklo_##suffix(a.val, z); \
    b1.val = _mm_unpackhi_##suffix(a.val, z); \
} \
inline _Tpwuvec v_load_expand(const _Tpu* ptr) \
{ \
    __m128i z = _mm_setzero_si128(); \
    return _Tpwuvec(_mm_unpacklo_##suffix(_mm_loadl_epi64((const __m128i*)ptr), z)); \
} \
inline void v_expand(const _Tpsvec& a, _Tpwsvec& b0, _Tpwsvec& b1) \
{ \
    b0.val = _mm_srai_##wsuffix(_mm_unpacklo_##suffix(a.val, a.val), shift); \
    b1.val = _mm_srai_##wsuffix(_mm_unpackhi_##suffix(a.val, a.val), shift); \
} \
inline _Tpwsvec v_load_expand(const _Tps* ptr) \
{ \
    __m128i a = _mm_loadl_epi64((const __m128i*)ptr); \
    return _Tpwsvec(_mm_srai_##wsuffix(_mm_unpacklo_##suffix(a, a), shift)); \
}

OPENCV_HAL_IMPL_SSE_EXPAND(v_uint8x16, v_uint16x8, uchar, v_int8x16, v_int16x8, schar, epi8, epi16, 8)
OPENCV_HAL_IMPL_SSE_EXPAND(v_uint16x8, v_uint32x4, ushort, v_int16x8, v_int32x4, short, epi16, epi32, 16)

inline void v_expand(const v_uint32x4& a, v_uint64x2& b0, v_uint64x2& b1)
{
    __m128i z = _mm_setzero_si128();
    b0.val = _mm_unpacklo_epi32(a.val, z);
    b1.val = _mm_unpackhi_epi32(a.val, z);
}
inline v_uint64x2 v_load_expand(const unsigned* ptr)
{
    __m128i z = _mm_setzero_si128();
    return v_uint64x2(_mm_unpacklo_epi32(_mm_loadl_epi64((const __m128i*)ptr), z));
}
inline void v_expand(const v_int32x4& a, v_int64x2& b0, v_int64x2& b1)
{
    __m128i s = _mm_srai_epi32(a.val, 31);
    b0.val = _mm_unpacklo_epi32(a.val, s);
    b1.val = _mm_unpackhi_epi32(a.val, s);
}
inline v_int64x2 v_load_expand(const int* ptr)
{
    __m128i a = _mm_loadl_epi64((const __m128i*)ptr);
    __m128i s = _mm_srai_epi32(a, 31);
    return v_int64x2(_mm_unpacklo_epi32(a, s));
}

inline v_uint32x4 v_load_expand_q(const uchar* ptr)
{
    __m128i z = _mm_setzero_si128();
    __m128i a = _mm_cvtsi32_si128(*(const int*)ptr);
    return v_uint32x4(_mm_unpacklo_epi16(_mm_unpacklo_epi8(a, z), z));
}

inline v_int32x4 v_load_expand_q(const schar* ptr)
{
    __m128i a = _mm_cvtsi32_si128(*(const int*)ptr);
    a = _mm_unpacklo_epi8(a, a);
    a = _mm_unpacklo_epi8(a, a);
    return v_int32x4(_mm_srai_epi32(a, 24));
}

#define OPENCV_HAL_IMPL_SSE_UNPACKS(_Tpvec, suffix, cast_from, cast_to) \
inline void v_zip(const _Tpvec& a0, const _Tpvec& a1, _Tpvec& b0, _Tpvec& b1) \
{ \
    b0.val = _mm_unpacklo_##suffix(a0.val, a1.val); \
    b1.val = _mm_unpackhi_##suffix(a0.val, a1.val); \
} \
inline _Tpvec v_combine_low(const _Tpvec& a, const _Tpvec& b) \
{ \
    __m128i a1 = cast_from(a.val), b1 = cast_from(b.val); \
    return _Tpvec(cast_to(_mm_unpacklo_epi64(a1, b1))); \
} \
inline _Tpvec v_combine_high(const _Tpvec& a, const _Tpvec& b) \
{ \
    __m128i a1 = cast_from(a.val), b1 = cast_from(b.val); \
    return _Tpvec(cast_to(_mm_unpackhi_epi64(a1, b1))); \
} \
inline void v_recombine(const _Tpvec& a, const _Tpvec& b, _Tpvec& c, _Tpvec& d) \
{ \
    __m128i a1 = cast_from(a.val), b1 = cast_from(b.val); \
    c.val = cast_to(_mm_unpacklo_epi64(a1, b1)); \
    d.val = cast_to(_mm_unpackhi_epi64(a1, b1)); \
}

OPENCV_HAL_IMPL_SSE_UNPACKS(v_uint8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_int8x16, epi8, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_uint16x8, epi16, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_int16x8, epi16, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_uint32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_int32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_float32x4, ps, _mm_castps_si128, _mm_castsi128_ps)
OPENCV_HAL_IMPL_SSE_UNPACKS(v_float64x2, pd, _mm_castpd_si128, _mm_castsi128_pd)

template<int s, typename _Tpvec>
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)
{
    const int w = sizeof(typename _Tpvec::lane_type);
    const int n = _Tpvec::nlanes;
    __m128i ra, rb;
    ra = _mm_srli_si128(a.val, s*w);
    rb = _mm_slli_si128(b.val, (n-s)*w);
    return _Tpvec(_mm_or_si128(ra, rb));
}

inline v_int32x4 v_round(const v_float32x4& a)
{ return v_int32x4(_mm_cvtps_epi32(a.val)); }

inline v_int32x4 v_floor(const v_float32x4& a)
{
    __m128i a1 = _mm_cvtps_epi32(a.val);
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(_mm_cvtepi32_ps(a1), a.val));
    return v_int32x4(_mm_add_epi32(a1, mask));
}

inline v_int32x4 v_ceil(const v_float32x4& a)
{
    __m128i a1 = _mm_cvtps_epi32(a.val);
    __m128i mask = _mm_castps_si128(_mm_cmpgt_ps(a.val, _mm_cvtepi32_ps(a1)));
    return v_int32x4(_mm_sub_epi32(a1, mask));
}

inline v_int32x4 v_trunc(const v_float32x4& a)
{ return v_int32x4(_mm_cvttps_epi32(a.val)); }

inline v_int32x4 v_round(const v_float64x2& a)
{ return v_int32x4(_mm_cvtpd_epi32(a.val)); }

inline v_int32x4 v_floor(const v_float64x2& a)
{
    __m128i a1 = _mm_cvtpd_epi32(a.val);
    __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(_mm_cvtepi32_pd(a1), a.val));
    mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
    return v_int32x4(_mm_add_epi32(a1, mask));
}

inline v_int32x4 v_ceil(const v_float64x2& a)
{
    __m128i a1 = _mm_cvtpd_epi32(a.val);
    __m128i mask = _mm_castpd_si128(_mm_cmpgt_pd(a.val, _mm_cvtepi32_pd(a1)));
    mask = _mm_srli_si128(_mm_slli_si128(mask, 4), 8); // m0 m0 m1 m1 => m0 m1 0 0
    return v_int32x4(_mm_sub_epi32(a1, mask));
}

inline v_int32x4 v_trunc(const v_float64x2& a)
{ return v_int32x4(_mm_cvttpd_epi32(a.val)); }

#define OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(_Tpvec, suffix, cast_from, cast_to) \
inline void v_transpose4x4(const _Tpvec& a0, const _Tpvec& a1, \
                           const _Tpvec& a2, const _Tpvec& a3, \
                           _Tpvec& b0, _Tpvec& b1, \
                           _Tpvec& b2, _Tpvec& b3) \
{ \
    __m128i t0 = cast_from(_mm_unpacklo_##suffix(a0.val, a1.val)); \
    __m128i t1 = cast_from(_mm_unpacklo_##suffix(a2.val, a3.val)); \
    __m128i t2 = cast_from(_mm_unpackhi_##suffix(a0.val, a1.val)); \
    __m128i t3 = cast_from(_mm_unpackhi_##suffix(a2.val, a3.val)); \
\
    b0.val = cast_to(_mm_unpacklo_epi64(t0, t1)); \
    b1.val = cast_to(_mm_unpackhi_epi64(t0, t1)); \
    b2.val = cast_to(_mm_unpacklo_epi64(t2, t3)); \
    b3.val = cast_to(_mm_unpackhi_epi64(t2, t3)); \
}

OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(v_uint32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(v_int32x4, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_TRANSPOSE4x4(v_float32x4, ps, _mm_castps_si128, _mm_castsi128_ps)

// adopted from sse_utils.hpp
inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c)
{
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i t02 = _mm_loadu_si128((const __m128i*)(ptr + 32));

    __m128i t10 = _mm_unpacklo_epi8(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi8(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi8(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi8(t11, _mm_unpackhi_epi64(t12, t12));

    __m128i t30 = _mm_unpacklo_epi8(t20, _mm_unpackhi_epi64(t21, t21));
    __m128i t31 = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t20, t20), t22);
    __m128i t32 = _mm_unpacklo_epi8(t21, _mm_unpackhi_epi64(t22, t22));

    a.val = _mm_unpacklo_epi8(t30, _mm_unpackhi_epi64(t31, t31));
    b.val = _mm_unpacklo_epi8(_mm_unpackhi_epi64(t30, t30), t32);
    c.val = _mm_unpacklo_epi8(t31, _mm_unpackhi_epi64(t32, t32));
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c, v_uint8x16& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1 ...
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 32)); // a8 b8 c8 d8 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 48)); // a12 b12 c12 d12 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2); // a0 a8 b0 b8 ...
    __m128i v1 = _mm_unpackhi_epi8(u0, u2); // a2 a10 b2 b10 ...
    __m128i v2 = _mm_unpacklo_epi8(u1, u3); // a4 a12 b4 b12 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a6 a14 b6 b14 ...

    u0 = _mm_unpacklo_epi8(v0, v2); // a0 a4 a8 a12 ...
    u1 = _mm_unpacklo_epi8(v1, v3); // a2 a6 a10 a14 ...
    u2 = _mm_unpackhi_epi8(v0, v2); // a1 a5 a9 a13 ...
    u3 = _mm_unpackhi_epi8(v1, v3); // a3 a7 a11 a15 ...

    v0 = _mm_unpacklo_epi8(u0, u1); // a0 a2 a4 a6 ...
    v1 = _mm_unpacklo_epi8(u2, u3); // a1 a3 a5 a7 ...
    v2 = _mm_unpackhi_epi8(u0, u1); // c0 c2 c4 c6 ...
    v3 = _mm_unpackhi_epi8(u2, u3); // c1 c3 c5 c7 ...

    a.val = _mm_unpacklo_epi8(v0, v1);
    b.val = _mm_unpackhi_epi8(v0, v1);
    c.val = _mm_unpacklo_epi8(v2, v3);
    d.val = _mm_unpackhi_epi8(v2, v3);
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c)
{
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 8));
    __m128i t02 = _mm_loadu_si128((const __m128i*)(ptr + 16));

    __m128i t10 = _mm_unpacklo_epi16(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi16(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi16(t01, _mm_unpackhi_epi64(t02, t02));

    __m128i t20 = _mm_unpacklo_epi16(t10, _mm_unpackhi_epi64(t11, t11));
    __m128i t21 = _mm_unpacklo_epi16(_mm_unpackhi_epi64(t10, t10), t12);
    __m128i t22 = _mm_unpacklo_epi16(t11, _mm_unpackhi_epi64(t12, t12));

    a.val = _mm_unpacklo_epi16(t20, _mm_unpackhi_epi64(t21, t21));
    b.val = _mm_unpacklo_epi16(_mm_unpackhi_epi64(t20, t20), t22);
    c.val = _mm_unpacklo_epi16(t21, _mm_unpackhi_epi64(t22, t22));
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c, v_uint16x8& d)
{
    __m128i u0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0 c0 d0 a1 b1 c1 d1
    __m128i u1 = _mm_loadu_si128((const __m128i*)(ptr + 8)); // a2 b2 c2 d2 ...
    __m128i u2 = _mm_loadu_si128((const __m128i*)(ptr + 16)); // a4 b4 c4 d4 ...
    __m128i u3 = _mm_loadu_si128((const __m128i*)(ptr + 24)); // a6 b6 c6 d6 ...

    __m128i v0 = _mm_unpacklo_epi16(u0, u2); // a0 a4 b0 b4 ...
    __m128i v1 = _mm_unpackhi_epi16(u0, u2); // a1 a5 b1 b5 ...
    __m128i v2 = _mm_unpacklo_epi16(u1, u3); // a2 a6 b2 b6 ...
    __m128i v3 = _mm_unpackhi_epi16(u1, u3); // a3 a7 b3 b7 ...

    u0 = _mm_unpacklo_epi16(v0, v2); // a0 a2 a4 a6 ...
    u1 = _mm_unpacklo_epi16(v1, v3); // a1 a3 a5 a7 ...
    u2 = _mm_unpackhi_epi16(v0, v2); // c0 c2 c4 c6 ...
    u3 = _mm_unpackhi_epi16(v1, v3); // c1 c3 c5 c7 ...

    a.val = _mm_unpacklo_epi16(u0, u1);
    b.val = _mm_unpackhi_epi16(u0, u1);
    c.val = _mm_unpacklo_epi16(u2, u3);
    d.val = _mm_unpackhi_epi16(u2, u3);
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c)
{
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 4));
    __m128i t02 = _mm_loadu_si128((const __m128i*)(ptr + 8));

    __m128i t10 = _mm_unpacklo_epi32(t00, _mm_unpackhi_epi64(t01, t01));
    __m128i t11 = _mm_unpacklo_epi32(_mm_unpackhi_epi64(t00, t00), t02);
    __m128i t12 = _mm_unpacklo_epi32(t01, _mm_unpackhi_epi64(t02, t02));

    a.val = _mm_unpacklo_epi32(t10, _mm_unpackhi_epi64(t11, t11));
    b.val = _mm_unpacklo_epi32(_mm_unpackhi_epi64(t10, t10), t12);
    c.val = _mm_unpacklo_epi32(t11, _mm_unpackhi_epi64(t12, t12));
}

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b, v_uint32x4& c, v_uint32x4& d)
{
    v_uint32x4 u0(_mm_loadu_si128((const __m128i*)ptr));        // a0 b0 c0 d0
    v_uint32x4 u1(_mm_loadu_si128((const __m128i*)(ptr + 4))); // a1 b1 c1 d1
    v_uint32x4 u2(_mm_loadu_si128((const __m128i*)(ptr + 8))); // a2 b2 c2 d2
    v_uint32x4 u3(_mm_loadu_si128((const __m128i*)(ptr + 12))); // a3 b3 c3 d3

    v_transpose4x4(u0, u1, u2, u3, a, b, c, d);
}

// 2-channel, float only
inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b)
{
    const int mask_lo = _MM_SHUFFLE(2, 0, 2, 0), mask_hi = _MM_SHUFFLE(3, 1, 3, 1);

    __m128 u0 = _mm_loadu_ps(ptr);       // a0 b0 a1 b1
    __m128 u1 = _mm_loadu_ps((ptr + 4)); // a2 b2 a3 b3

    a.val = _mm_shuffle_ps(u0, u1, mask_lo); // a0 a1 a2 a3
    b.val = _mm_shuffle_ps(u0, u1, mask_hi); // b0 b1 ab b3
}

inline void v_store_interleave( short* ptr, const v_int16x8& a, const v_int16x8& b )
{
    __m128i t0, t1;
    t0 = _mm_unpacklo_epi16(a.val, b.val);
    t1 = _mm_unpackhi_epi16(a.val, b.val);
    _mm_storeu_si128((__m128i*)(ptr), t0);
    _mm_storeu_si128((__m128i*)(ptr + 8), t1);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c )
{
    __m128i z = _mm_setzero_si128();
    __m128i ab0 = _mm_unpacklo_epi8(a.val, b.val);
    __m128i ab1 = _mm_unpackhi_epi8(a.val, b.val);
    __m128i c0 = _mm_unpacklo_epi8(c.val, z);
    __m128i c1 = _mm_unpackhi_epi8(c.val, z);

    __m128i p00 = _mm_unpacklo_epi16(ab0, c0);
    __m128i p01 = _mm_unpackhi_epi16(ab0, c0);
    __m128i p02 = _mm_unpacklo_epi16(ab1, c1);
    __m128i p03 = _mm_unpackhi_epi16(ab1, c1);

    __m128i p10 = _mm_unpacklo_epi32(p00, p01);
    __m128i p11 = _mm_unpackhi_epi32(p00, p01);
    __m128i p12 = _mm_unpacklo_epi32(p02, p03);
    __m128i p13 = _mm_unpackhi_epi32(p02, p03);

    __m128i p20 = _mm_unpacklo_epi64(p10, p11);
    __m128i p21 = _mm_unpackhi_epi64(p10, p11);
    __m128i p22 = _mm_unpacklo_epi64(p12, p13);
    __m128i p23 = _mm_unpackhi_epi64(p12, p13);

    p20 = _mm_slli_si128(p20, 1);
    p22 = _mm_slli_si128(p22, 1);

    __m128i p30 = _mm_slli_epi64(_mm_unpacklo_epi32(p20, p21), 8);
    __m128i p31 = _mm_srli_epi64(_mm_unpackhi_epi32(p20, p21), 8);
    __m128i p32 = _mm_slli_epi64(_mm_unpacklo_epi32(p22, p23), 8);
    __m128i p33 = _mm_srli_epi64(_mm_unpackhi_epi32(p22, p23), 8);

    __m128i p40 = _mm_unpacklo_epi64(p30, p31);
    __m128i p41 = _mm_unpackhi_epi64(p30, p31);
    __m128i p42 = _mm_unpacklo_epi64(p32, p33);
    __m128i p43 = _mm_unpackhi_epi64(p32, p33);

    __m128i v0 = _mm_or_si128(_mm_srli_si128(p40, 2), _mm_slli_si128(p41, 10));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(p41, 6), _mm_slli_si128(p42, 6));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(p42, 10), _mm_slli_si128(p43, 2));

    _mm_storeu_si128((__m128i*)(ptr), v0);
    _mm_storeu_si128((__m128i*)(ptr + 16), v1);
    _mm_storeu_si128((__m128i*)(ptr + 32), v2);
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, const v_uint8x16& d)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    __m128i u0 = _mm_unpacklo_epi8(a.val, c.val); // a0 c0 a1 c1 ...
    __m128i u1 = _mm_unpackhi_epi8(a.val, c.val); // a8 c8 a9 c9 ...
    __m128i u2 = _mm_unpacklo_epi8(b.val, d.val); // b0 d0 b1 d1 ...
    __m128i u3 = _mm_unpackhi_epi8(b.val, d.val); // b8 d8 b9 d9 ...

    __m128i v0 = _mm_unpacklo_epi8(u0, u2); // a0 b0 c0 d0 ...
    __m128i v1 = _mm_unpacklo_epi8(u1, u3); // a8 b8 c8 d8 ...
    __m128i v2 = _mm_unpackhi_epi8(u0, u2); // a4 b4 c4 d4 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a12 b12 c12 d12 ...

    _mm_storeu_si128((__m128i*)ptr, v0);
    _mm_storeu_si128((__m128i*)(ptr + 16), v2);
    _mm_storeu_si128((__m128i*)(ptr + 32), v1);
    _mm_storeu_si128((__m128i*)(ptr + 48), v3);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a,
                                const v_uint16x8& b,
                                const v_uint16x8& c )
{
    __m128i z = _mm_setzero_si128();
    __m128i ab0 = _mm_unpacklo_epi16(a.val, b.val);
    __m128i ab1 = _mm_unpackhi_epi16(a.val, b.val);
    __m128i c0 = _mm_unpacklo_epi16(c.val, z);
    __m128i c1 = _mm_unpackhi_epi16(c.val, z);

    __m128i p10 = _mm_unpacklo_epi32(ab0, c0);
    __m128i p11 = _mm_unpackhi_epi32(ab0, c0);
    __m128i p12 = _mm_unpacklo_epi32(ab1, c1);
    __m128i p13 = _mm_unpackhi_epi32(ab1, c1);

    __m128i p20 = _mm_unpacklo_epi64(p10, p11);
    __m128i p21 = _mm_unpackhi_epi64(p10, p11);
    __m128i p22 = _mm_unpacklo_epi64(p12, p13);
    __m128i p23 = _mm_unpackhi_epi64(p12, p13);

    p20 = _mm_slli_si128(p20, 2);
    p22 = _mm_slli_si128(p22, 2);

    __m128i p30 = _mm_unpacklo_epi64(p20, p21);
    __m128i p31 = _mm_unpackhi_epi64(p20, p21);
    __m128i p32 = _mm_unpacklo_epi64(p22, p23);
    __m128i p33 = _mm_unpackhi_epi64(p22, p23);

    __m128i v0 = _mm_or_si128(_mm_srli_si128(p30, 2), _mm_slli_si128(p31, 10));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(p31, 6), _mm_slli_si128(p32, 6));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(p32, 10), _mm_slli_si128(p33, 2));

    _mm_storeu_si128((__m128i*)(ptr), v0);
    _mm_storeu_si128((__m128i*)(ptr + 8), v1);
    _mm_storeu_si128((__m128i*)(ptr + 16), v2);
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                const v_uint16x8& c, const v_uint16x8& d)
{
    // a0 a1 a2 a3 ....
    // b0 b1 b2 b3 ....
    // c0 c1 c2 c3 ....
    // d0 d1 d2 d3 ....
    __m128i u0 = _mm_unpacklo_epi16(a.val, c.val); // a0 c0 a1 c1 ...
    __m128i u1 = _mm_unpackhi_epi16(a.val, c.val); // a4 c4 a5 c5 ...
    __m128i u2 = _mm_unpacklo_epi16(b.val, d.val); // b0 d0 b1 d1 ...
    __m128i u3 = _mm_unpackhi_epi16(b.val, d.val); // b4 d4 b5 d5 ...

    __m128i v0 = _mm_unpacklo_epi16(u0, u2); // a0 b0 c0 d0 ...
    __m128i v1 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    __m128i v2 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    __m128i v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...

    _mm_storeu_si128((__m128i*)ptr, v0);
    _mm_storeu_si128((__m128i*)(ptr + 8), v2);
    _mm_storeu_si128((__m128i*)(ptr + 16), v1);
    _mm_storeu_si128((__m128i*)(ptr + 24), v3);
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                const v_uint32x4& c )
{
    v_uint32x4 z = v_setzero_u32(), u0, u1, u2, u3;
    v_transpose4x4(a, b, c, z, u0, u1, u2, u3);

    __m128i v0 = _mm_or_si128(u0.val, _mm_slli_si128(u1.val, 12));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(u1.val, 4), _mm_slli_si128(u2.val, 8));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(u2.val, 8), _mm_slli_si128(u3.val, 4));

    _mm_storeu_si128((__m128i*)ptr, v0);
    _mm_storeu_si128((__m128i*)(ptr + 4), v1);
    _mm_storeu_si128((__m128i*)(ptr + 8), v2);
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               const v_uint32x4& c, const v_uint32x4& d)
{
    v_uint32x4 t0, t1, t2, t3;
    v_transpose4x4(a, b, c, d, t0, t1, t2, t3);
    v_store(ptr, t0);
    v_store(ptr + 4, t1);
    v_store(ptr + 8, t2);
    v_store(ptr + 12, t3);
}

// 2-channel, float only
inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b)
{
    // a0 a1 a2 a3 ...
    // b0 b1 b2 b3 ...
    __m128 u0 = _mm_unpacklo_ps(a.val, b.val); // a0 b0 a1 b1
    __m128 u1 = _mm_unpackhi_ps(a.val, b.val); // a2 b2 a3 b3

    _mm_storeu_ps(ptr, u0);
    _mm_storeu_ps((ptr + 4), u1);
}

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(_Tpvec, _Tp, suffix, _Tpuvec, _Tpu, usuffix) \
inline void v_load_deinterleave( const _Tp* ptr, _Tpvec& a0, \
                                 _Tpvec& b0, _Tpvec& c0 ) \
{ \
    _Tpuvec a1, b1, c1; \
    v_load_deinterleave((const _Tpu*)ptr, a1, b1, c1); \
    a0 = v_reinterpret_as_##suffix(a1); \
    b0 = v_reinterpret_as_##suffix(b1); \
    c0 = v_reinterpret_as_##suffix(c1); \
} \
inline void v_load_deinterleave( const _Tp* ptr, _Tpvec& a0, \
                                 _Tpvec& b0, _Tpvec& c0, _Tpvec& d0 ) \
{ \
    _Tpuvec a1, b1, c1, d1; \
    v_load_deinterleave((const _Tpu*)ptr, a1, b1, c1, d1); \
    a0 = v_reinterpret_as_##suffix(a1); \
    b0 = v_reinterpret_as_##suffix(b1); \
    c0 = v_reinterpret_as_##suffix(c1); \
    d0 = v_reinterpret_as_##suffix(d1); \
} \
inline void v_store_interleave( _Tp* ptr, const _Tpvec& a0, \
                               const _Tpvec& b0, const _Tpvec& c0 ) \
{ \
    _Tpuvec a1 = v_reinterpret_as_##usuffix(a0); \
    _Tpuvec b1 = v_reinterpret_as_##usuffix(b0); \
    _Tpuvec c1 = v_reinterpret_as_##usuffix(c0); \
    v_store_interleave((_Tpu*)ptr, a1, b1, c1); \
} \
inline void v_store_interleave( _Tp* ptr, const _Tpvec& a0, const _Tpvec& b0, \
                               const _Tpvec& c0, const _Tpvec& d0 ) \
{ \
    _Tpuvec a1 = v_reinterpret_as_##usuffix(a0); \
    _Tpuvec b1 = v_reinterpret_as_##usuffix(b0); \
    _Tpuvec c1 = v_reinterpret_as_##usuffix(c0); \
    _Tpuvec d1 = v_reinterpret_as_##usuffix(d0); \
    v_store_interleave((_Tpu*)ptr, a1, b1, c1, d1); \
}

OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int8x16, schar, s8, v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int16x8, short, s16, v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int32x4, int, s32, v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_float32x4, float, f32, v_uint32x4, unsigned, u32)

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(_mm_cvtepi32_ps(a.val));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    return v_float32x4(_mm_cvtpd_ps(a.val));
}

inline v_float64x2 v_cvt_f64(const v_int32x4& a)
{
    return v_float64x2(_mm_cvtepi32_pd(a.val));
}

inline v_float64x2 v_cvt_f64_high(const v_int32x4& a)
{
    return v_float64x2(_mm_cvtepi32_pd(_mm_srli_si128(a.val,8)));
}

inline v_float64x2 v_cvt_f64(const v_float32x4& a)
{
    return v_float64x2(_mm_cvtps_pd(a.val));
}

inline v_float64x2 v_cvt_f64_high(const v_float32x4& a)
{
    return v_float64x2(_mm_cvtps_pd(_mm_castsi128_ps(_mm_srli_si128(_mm_castps_si128(a.val),8))));
}

#if defined(HAVE_FP16)
inline v_float32x4 v_cvt_f32(const v_float16x4& a)
{
    return v_float32x4(_mm_cvtph_ps(a.val));
}

inline v_float16x4 v_cvt_f16(const v_float32x4& a)
{
    return v_float16x4(_mm_cvtps_ph(a.val, 0));
}
#endif

//! @name Check SIMD support
//! @{
//! @brief Check CPU capability of SIMD operation
static inline bool hasSIMD128()
{
    return checkHardwareSupport(CV_CPU_SSE2);
}

//! @}

//! @endcond

}

#endif
