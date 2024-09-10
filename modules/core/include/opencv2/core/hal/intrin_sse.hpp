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
#define CV_SIMD128_FP16 0  // no native operations with FP16 type.

namespace cv
{

//! @cond IGNORED

//
// Compilation troubleshooting:
// - MSVC: error C2719: 'a': formal parameter with requested alignment of 16 won't be aligned
//   Replace parameter declaration to const reference:
//   -v_int32x4 a
//   +const v_int32x4& a
//

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_BEGIN

///////// Types ////////////

struct v_uint8x16
{
    typedef uchar lane_type;
    typedef __m128i vector_type;
    enum { nlanes = 16 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128i vector_type;
    enum { nlanes = 16 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128i vector_type;
    enum { nlanes = 8 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128i vector_type;
    enum { nlanes = 8 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128i vector_type;
    enum { nlanes = 4 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128i vector_type;
    enum { nlanes = 4 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128 vector_type;
    enum { nlanes = 4 };

    /* coverity[uninit_ctor]: suppress warning */
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
    typedef __m128i vector_type;
    enum { nlanes = 2 };

    /* coverity[uninit_ctor]: suppress warning */
    v_uint64x2() {}
    explicit v_uint64x2(__m128i v) : val(v) {}
    v_uint64x2(uint64 v0, uint64 v1)
    {
#if defined(_MSC_VER) && _MSC_VER >= 1920/*MSVS 2019*/ && defined(_M_X64) && !defined(__clang__)
        val = _mm_setr_epi64x((int64_t)v0, (int64_t)v1);
#elif defined(__GNUC__)
        val = _mm_setr_epi64((__m64)v0, (__m64)v1);
#else
        val = _mm_setr_epi32((int)v0, (int)(v0 >> 32), (int)v1, (int)(v1 >> 32));
#endif
    }

    uint64 get0() const
    {
    #if !defined(__x86_64__) && !defined(_M_X64)
        int a = _mm_cvtsi128_si32(val);
        int b = _mm_cvtsi128_si32(_mm_srli_epi64(val, 32));
        return (unsigned)a | ((uint64)(unsigned)b << 32);
    #else
        return (uint64)_mm_cvtsi128_si64(val);
    #endif
    }

    __m128i val;
};

struct v_int64x2
{
    typedef int64 lane_type;
    typedef __m128i vector_type;
    enum { nlanes = 2 };

    /* coverity[uninit_ctor]: suppress warning */
    v_int64x2() {}
    explicit v_int64x2(__m128i v) : val(v) {}
    v_int64x2(int64 v0, int64 v1)
    {
#if defined(_MSC_VER) && _MSC_VER >= 1920/*MSVS 2019*/ && defined(_M_X64) && !defined(__clang__)
        val = _mm_setr_epi64x((int64_t)v0, (int64_t)v1);
#elif defined(__GNUC__)
        val = _mm_setr_epi64((__m64)v0, (__m64)v1);
#else
        val = _mm_setr_epi32((int)v0, (int)(v0 >> 32), (int)v1, (int)(v1 >> 32));
#endif
    }

    int64 get0() const
    {
    #if !defined(__x86_64__) && !defined(_M_X64)
        int a = _mm_cvtsi128_si32(val);
        int b = _mm_cvtsi128_si32(_mm_srli_epi64(val, 32));
        return (int64)((unsigned)a | ((uint64)(unsigned)b << 32));
    #else
        return _mm_cvtsi128_si64(val);
    #endif
    }

    __m128i val;
};

struct v_float64x2
{
    typedef double lane_type;
    typedef __m128d vector_type;
    enum { nlanes = 2 };

    /* coverity[uninit_ctor]: suppress warning */
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

namespace hal_sse_internal
{
    template <typename to_sse_type, typename from_sse_type>
    to_sse_type v_sse_reinterpret_as(const from_sse_type& val);

#define OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(to_sse_type, from_sse_type, sse_cast_intrin) \
    template<> inline \
    to_sse_type v_sse_reinterpret_as(const from_sse_type& a) \
    { return sse_cast_intrin(a); }

    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128i, __m128i, OPENCV_HAL_NOP)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128i, __m128, _mm_castps_si128)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128i, __m128d, _mm_castpd_si128)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128, __m128i, _mm_castsi128_ps)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128, __m128, OPENCV_HAL_NOP)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128, __m128d, _mm_castpd_ps)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128d, __m128i, _mm_castsi128_pd)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128d, __m128, _mm_castps_pd)
    OPENCV_HAL_IMPL_SSE_REINTERPRET_RAW(__m128d, __m128d, OPENCV_HAL_NOP)
}

#define OPENCV_HAL_IMPL_SSE_INITVEC(_Tpvec, _Tp, suffix, zsuffix, ssuffix, _Tps, cast) \
inline _Tpvec v_setzero_##suffix() { return _Tpvec(_mm_setzero_##zsuffix()); } \
inline _Tpvec v_setall_##suffix(_Tp v) { return _Tpvec(_mm_set1_##ssuffix((_Tps)v)); } \
template<typename _Tpvec0> inline _Tpvec v_reinterpret_as_##suffix(const _Tpvec0& a) \
{ return _Tpvec(cast(a.val)); }

OPENCV_HAL_IMPL_SSE_INITVEC(v_uint8x16, uchar, u8, si128, epi8, schar, OPENCV_HAL_NOP)
OPENCV_HAL_IMPL_SSE_INITVEC(v_int8x16, schar, s8, si128, epi8, schar, OPENCV_HAL_NOP)
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

inline void v_pack_store(schar* ptr, const v_int16x8& a)
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


// byte-wise "mask ? a : b"
inline __m128i v_select_si128(__m128i mask, __m128i a, __m128i b)
{
#if CV_SSE4_1
    return _mm_blendv_epi8(b, a, mask);
#else
    return _mm_xor_si128(b, _mm_and_si128(_mm_xor_si128(a, b), mask));
#endif
}

inline v_uint16x8 v_pack(const v_uint32x4& a, const v_uint32x4& b)
{ return v_uint16x8(_v128_packs_epu32(a.val, b.val)); }

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
#if CV_SSE4_1
    return v_uint16x8(_mm_packus_epi32(a.val, b.val));
#else
    __m128i delta32 = _mm_set1_epi32(32768);

    // preliminary saturate negative values to zero
    __m128i a1 = _mm_and_si128(a.val, _mm_cmpgt_epi32(a.val, _mm_set1_epi32(0)));
    __m128i b1 = _mm_and_si128(b.val, _mm_cmpgt_epi32(b.val, _mm_set1_epi32(0)));

    __m128i r = _mm_packs_epi32(_mm_sub_epi32(a1, delta32), _mm_sub_epi32(b1, delta32));
    return v_uint16x8(_mm_sub_epi16(r, _mm_set1_epi16(-32768)));
#endif
}

inline void v_pack_u_store(ushort* ptr, const v_int32x4& a)
{
#if CV_SSE4_1
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi32(a.val, a.val));
#else
    __m128i delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(a.val, delta32);
    __m128i r = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, r);
#endif
}

template<int n> inline
v_uint16x8 v_rshr_pack_u(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    __m128i delta = _mm_set1_epi32(1 << (n - 1));
    return v_uint16x8(_mm_packus_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n),
                                       _mm_srai_epi32(_mm_add_epi32(b.val, delta), n)));
#else
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    __m128i b1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(b.val, delta), n), delta32);
    __m128i b2 = _mm_sub_epi16(_mm_packs_epi32(b1, b1), _mm_set1_epi16(-32768));
    return v_uint16x8(_mm_unpacklo_epi64(a2, b2));
#endif
}

template<int n> inline
void v_rshr_pack_u_store(ushort* ptr, const v_int32x4& a)
{
#if CV_SSE4_1
    __m128i delta = _mm_set1_epi32(1 << (n - 1));
    __m128i a1 = _mm_srai_epi32(_mm_add_epi32(a.val, delta), n);
    _mm_storel_epi64((__m128i*)ptr, _mm_packus_epi32(a1, a1));
#else
    __m128i delta = _mm_set1_epi32(1 << (n-1)), delta32 = _mm_set1_epi32(32768);
    __m128i a1 = _mm_sub_epi32(_mm_srai_epi32(_mm_add_epi32(a.val, delta), n), delta32);
    __m128i a2 = _mm_sub_epi16(_mm_packs_epi32(a1, a1), _mm_set1_epi16(-32768));
    _mm_storel_epi64((__m128i*)ptr, a2);
#endif
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

// pack boolean
inline v_uint8x16 v_pack_b(const v_uint16x8& a, const v_uint16x8& b)
{
    __m128i ab = _mm_packs_epi16(a.val, b.val);
    return v_uint8x16(ab);
}

inline v_uint8x16 v_pack_b(const v_uint32x4& a, const v_uint32x4& b,
                           const v_uint32x4& c, const v_uint32x4& d)
{
    __m128i ab = _mm_packs_epi32(a.val, b.val);
    __m128i cd = _mm_packs_epi32(c.val, d.val);
    return v_uint8x16(_mm_packs_epi16(ab, cd));
}

inline v_uint8x16 v_pack_b(const v_uint64x2& a, const v_uint64x2& b, const v_uint64x2& c,
                           const v_uint64x2& d, const v_uint64x2& e, const v_uint64x2& f,
                           const v_uint64x2& g, const v_uint64x2& h)
{
    __m128i ab = _mm_packs_epi32(a.val, b.val);
    __m128i cd = _mm_packs_epi32(c.val, d.val);
    __m128i ef = _mm_packs_epi32(e.val, f.val);
    __m128i gh = _mm_packs_epi32(g.val, h.val);

    __m128i abcd = _mm_packs_epi32(ab, cd);
    __m128i efgh = _mm_packs_epi32(ef, gh);
    return v_uint8x16(_mm_packs_epi16(abcd, efgh));
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

inline v_float32x4 v_matmuladd(const v_float32x4& v, const v_float32x4& m0,
                               const v_float32x4& m1, const v_float32x4& m2,
                               const v_float32x4& a)
{
    __m128 v0 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(0, 0, 0, 0)), m0.val);
    __m128 v1 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(1, 1, 1, 1)), m1.val);
    __m128 v2 = _mm_mul_ps(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE(2, 2, 2, 2)), m2.val);

    return v_float32x4(_mm_add_ps(_mm_add_ps(v0, v1), _mm_add_ps(v2, a.val)));
}

#define OPENCV_HAL_IMPL_SSE_BIN_OP(bin_op, _Tpvec, intrin) \
    inline _Tpvec bin_op(const _Tpvec& a, const _Tpvec& b) \
    { \
        return _Tpvec(intrin(a.val, b.val)); \
    }

OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_uint8x16, _mm_adds_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_uint8x16, _mm_subs_epu8)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_int8x16, _mm_adds_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_int8x16, _mm_subs_epi8)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_uint16x8, _mm_adds_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_uint16x8, _mm_subs_epu16)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_int16x8, _mm_adds_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_int16x8, _mm_subs_epi16)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_uint32x4, _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_uint32x4, _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_mul, v_uint32x4, _v128_mullo_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_int32x4, _mm_add_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_int32x4, _mm_sub_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_mul, v_int32x4, _v128_mullo_epi32)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_float32x4, _mm_add_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_float32x4, _mm_sub_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_mul, v_float32x4, _mm_mul_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_div, v_float32x4, _mm_div_ps)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_float64x2, _mm_add_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_float64x2, _mm_sub_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_mul, v_float64x2, _mm_mul_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_div, v_float64x2, _mm_div_pd)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_uint64x2, _mm_add_epi64)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_uint64x2, _mm_sub_epi64)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_add, v_int64x2, _mm_add_epi64)
OPENCV_HAL_IMPL_SSE_BIN_OP(v_sub, v_int64x2, _mm_sub_epi64)

// saturating multiply 8-bit, 16-bit
#define OPENCV_HAL_IMPL_SSE_MUL_SAT(_Tpvec, _Tpwvec)             \
    inline _Tpvec v_mul(const _Tpvec& a, const _Tpvec& b)        \
    {                                                            \
        _Tpwvec c, d;                                            \
        v_mul_expand(a, b, c, d);                                \
        return v_pack(c, d);                                     \
    }

OPENCV_HAL_IMPL_SSE_MUL_SAT(v_uint8x16, v_uint16x8)
OPENCV_HAL_IMPL_SSE_MUL_SAT(v_int8x16,  v_int16x8)
OPENCV_HAL_IMPL_SSE_MUL_SAT(v_uint16x8, v_uint32x4)
OPENCV_HAL_IMPL_SSE_MUL_SAT(v_int16x8,  v_int32x4)

//  Multiply and expand
inline void v_mul_expand(const v_uint8x16& a, const v_uint8x16& b,
                         v_uint16x8& c, v_uint16x8& d)
{
    v_uint16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
}

inline void v_mul_expand(const v_int8x16& a, const v_int8x16& b,
                         v_int16x8& c, v_int16x8& d)
{
    v_int16x8 a0, a1, b0, b1;
    v_expand(a, a0, a1);
    v_expand(b, b0, b1);
    c = v_mul_wrap(a0, b0);
    d = v_mul_wrap(a1, b1);
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

inline v_int16x8 v_mul_hi(const v_int16x8& a, const v_int16x8& b) { return v_int16x8(_mm_mulhi_epi16(a.val, b.val)); }
inline v_uint16x8 v_mul_hi(const v_uint16x8& a, const v_uint16x8& b) { return v_uint16x8(_mm_mulhi_epu16(a.val, b.val)); }

//////// Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b)
{ return v_int32x4(_mm_madd_epi16(a.val, b.val)); }
inline v_int32x4 v_dotprod(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_add(v_dotprod(a, b), c); }

// 32 >> 64
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    __m128i even = _mm_mul_epi32(a.val, b.val);
    __m128i odd = _mm_mul_epi32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    return v_int64x2(_mm_add_epi64(even, odd));
#else
    __m128i even_u = _mm_mul_epu32(a.val, b.val);
    __m128i odd_u = _mm_mul_epu32(_mm_srli_epi64(a.val, 32), _mm_srli_epi64(b.val, 32));
    // convert unsigned to signed high multiplication (from: Agner Fog(veclib) and H S Warren: Hacker's delight, 2003, p. 132)
    __m128i a_sign = _mm_srai_epi32(a.val, 31);
    __m128i b_sign = _mm_srai_epi32(b.val, 31);
    // |x * sign of x
    __m128i axb  = _mm_and_si128(a.val, b_sign);
    __m128i bxa  = _mm_and_si128(b.val, a_sign);
    // sum of sign corrections
    __m128i ssum = _mm_add_epi32(bxa, axb);
    __m128i even_ssum = _mm_slli_epi64(ssum, 32);
    __m128i odd_ssum = _mm_and_si128(ssum, _mm_set_epi32(-1, 0, -1, 0));
    // convert to signed and prod
    return v_int64x2(_mm_add_epi64(_mm_sub_epi64(even_u, even_ssum), _mm_sub_epi64(odd_u, odd_ssum)));
#endif
}
inline v_int64x2 v_dotprod(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_add(v_dotprod(a, b), c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i a0 = _mm_srli_epi16(_mm_slli_si128(a.val, 1), 8); // even
    __m128i a1 = _mm_srli_epi16(a.val, 8); // odd
    __m128i b0 = _mm_srli_epi16(_mm_slli_si128(b.val, 1), 8);
    __m128i b1 = _mm_srli_epi16(b.val, 8);
    __m128i p0 = _mm_madd_epi16(a0, b0);
    __m128i p1 = _mm_madd_epi16(a1, b1);
    return v_uint32x4(_mm_add_epi32(p0, p1));
}
inline v_uint32x4 v_dotprod_expand(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b)
{
    __m128i a0 = _mm_srai_epi16(_mm_slli_si128(a.val, 1), 8); // even
    __m128i a1 = _mm_srai_epi16(a.val, 8); // odd
    __m128i b0 = _mm_srai_epi16(_mm_slli_si128(b.val, 1), 8);
    __m128i b1 = _mm_srai_epi16(b.val, 8);
    __m128i p0 = _mm_madd_epi16(a0, b0);
    __m128i p1 = _mm_madd_epi16(a1, b1);
    return v_int32x4(_mm_add_epi32(p0, p1));
}
inline v_int32x4 v_dotprod_expand(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint32x4 c, d;
    v_mul_expand(a, b, c, d);

    v_uint64x2 c0, c1, d0, d1;
    v_expand(c, c0, c1);
    v_expand(d, d0, d1);

    c0 = v_add(c0, c1); d0 = v_add(d0, d1);
    return v_uint64x2(_mm_add_epi64(
        _mm_unpacklo_epi64(c0.val, d0.val),
        _mm_unpackhi_epi64(c0.val, d0.val)
    ));
}
inline v_uint64x2 v_dotprod_expand(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b)
{
    v_int32x4 prod = v_dotprod(a, b);
    v_int64x2 c, d;
    v_expand(prod, c, d);
    return v_int64x2(_mm_add_epi64(
        _mm_unpacklo_epi64(c.val, d.val),
        _mm_unpackhi_epi64(c.val, d.val)
    ));
}
inline v_int64x2 v_dotprod_expand(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

// 32 >> 64f
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    return v_cvt_f64(v_dotprod(a, b));
#else
    v_float64x2 c = v_mul(v_cvt_f64(a), v_cvt_f64(b));
    v_float64x2 d = v_mul(v_cvt_f64_high(a), v_cvt_f64_high(b));

    return v_float64x2(_mm_add_pd(
        _mm_unpacklo_pd(c.val, d.val),
        _mm_unpackhi_pd(c.val, d.val)
    ));
#endif
}
inline v_float64x2 v_dotprod_expand(const v_int32x4& a, const v_int32x4& b, const v_float64x2& c)
{ return v_add(v_dotprod_expand(a, b), c); }

//////// Fast Dot Product ////////

// 16 >> 32
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b)
{ return v_dotprod(a, b); }
inline v_int32x4 v_dotprod_fast(const v_int16x8& a, const v_int16x8& b, const v_int32x4& c)
{ return v_add(v_dotprod(a, b), c); }

// 32 >> 64
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_dotprod(a, b); }
inline v_int64x2 v_dotprod_fast(const v_int32x4& a, const v_int32x4& b, const v_int64x2& c)
{ return v_add(v_dotprod_fast(a, b), c); }

// 8 >> 32
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i a0 = v_expand_low(a).val;
    __m128i a1 = v_expand_high(a).val;
    __m128i b0 = v_expand_low(b).val;
    __m128i b1 = v_expand_high(b).val;
    __m128i p0 = _mm_madd_epi16(a0, b0);
    __m128i p1 = _mm_madd_epi16(a1, b1);
    return v_uint32x4(_mm_add_epi32(p0, p1));
}
inline v_uint32x4 v_dotprod_expand_fast(const v_uint8x16& a, const v_uint8x16& b, const v_uint32x4& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b)
{
#if CV_SSE4_1
    __m128i a0 = _mm_cvtepi8_epi16(a.val);
    __m128i a1 = v_expand_high(a).val;
    __m128i b0 = _mm_cvtepi8_epi16(b.val);
    __m128i b1 = v_expand_high(b).val;
    __m128i p0 = _mm_madd_epi16(a0, b0);
    __m128i p1 = _mm_madd_epi16(a1, b1);
    return v_int32x4(_mm_add_epi32(p0, p1));
#else
    return v_dotprod_expand(a, b);
#endif
}
inline v_int32x4 v_dotprod_expand_fast(const v_int8x16& a, const v_int8x16& b, const v_int32x4& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

// 16 >> 64
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint32x4 c, d;
    v_mul_expand(a, b, c, d);

    v_uint64x2 c0, c1, d0, d1;
    v_expand(c, c0, c1);
    v_expand(d, d0, d1);

    c0 = v_add(c0, c1); d0 = v_add(d0, d1);
    return v_add(c0, d0);
}
inline v_uint64x2 v_dotprod_expand_fast(const v_uint16x8& a, const v_uint16x8& b, const v_uint64x2& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b)
{
    v_int32x4 prod = v_dotprod(a, b);
    v_int64x2 c, d;
    v_expand(prod, c, d);
    return v_add(c, d);
}
inline v_int64x2 v_dotprod_expand_fast(const v_int16x8& a, const v_int16x8& b, const v_int64x2& c)
{ return v_add(v_dotprod_expand_fast(a, b), c); }

// 32 >> 64f
v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c);
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a, const v_int32x4& b)
{ return v_fma(v_cvt_f64(a), v_cvt_f64(b), v_mul(v_cvt_f64_high(a), v_cvt_f64_high(b))); }
inline v_float64x2 v_dotprod_expand_fast(const v_int32x4& a,   const v_int32x4& b, const v_float64x2& c)
{ return v_fma(v_cvt_f64(a), v_cvt_f64(b), v_fma(v_cvt_f64_high(a), v_cvt_f64_high(b), c)); }

#define OPENCV_HAL_IMPL_SSE_LOGIC_OP(_Tpvec, suffix, not_const) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(v_and, _Tpvec, _mm_and_##suffix) \
    OPENCV_HAL_IMPL_SSE_BIN_OP(v_or, _Tpvec, _mm_or_##suffix)   \
    OPENCV_HAL_IMPL_SSE_BIN_OP(v_xor, _Tpvec, _mm_xor_##suffix) \
    inline _Tpvec v_not(const _Tpvec& a) \
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
    const __m128 _0_5 = _mm_set1_ps(0.5f), _1_5 = _mm_set1_ps(1.5f);
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
    const __m128d v_1 = _mm_set1_pd(1.);
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
#if CV_SSE4_1
    return v_int8x16(_mm_min_epi8(a.val, b.val));
#else
    __m128i delta = _mm_set1_epi8((char)-128);
    return v_int8x16(_mm_xor_si128(delta, _mm_min_epu8(_mm_xor_si128(a.val, delta),
                                                       _mm_xor_si128(b.val, delta))));
#endif
}
inline v_int8x16 v_max(const v_int8x16& a, const v_int8x16& b)
{
#if CV_SSE4_1
    return v_int8x16(_mm_max_epi8(a.val, b.val));
#else
    __m128i delta = _mm_set1_epi8((char)-128);
    return v_int8x16(_mm_xor_si128(delta, _mm_max_epu8(_mm_xor_si128(a.val, delta),
                                                       _mm_xor_si128(b.val, delta))));
#endif
}
inline v_uint16x8 v_min(const v_uint16x8& a, const v_uint16x8& b)
{
#if CV_SSE4_1
    return v_uint16x8(_mm_min_epu16(a.val, b.val));
#else
    return v_uint16x8(_mm_subs_epu16(a.val, _mm_subs_epu16(a.val, b.val)));
#endif
}
inline v_uint16x8 v_max(const v_uint16x8& a, const v_uint16x8& b)
{
#if CV_SSE4_1
    return v_uint16x8(_mm_max_epu16(a.val, b.val));
#else
    return v_uint16x8(_mm_adds_epu16(_mm_subs_epu16(a.val, b.val), b.val));
#endif
}
inline v_uint32x4 v_min(const v_uint32x4& a, const v_uint32x4& b)
{
#if CV_SSE4_1
    return v_uint32x4(_mm_min_epu32(a.val, b.val));
#else
    __m128i delta = _mm_set1_epi32((int)0x80000000);
    __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
    return v_uint32x4(v_select_si128(mask, b.val, a.val));
#endif
}
inline v_uint32x4 v_max(const v_uint32x4& a, const v_uint32x4& b)
{
#if CV_SSE4_1
    return v_uint32x4(_mm_max_epu32(a.val, b.val));
#else
    __m128i delta = _mm_set1_epi32((int)0x80000000);
    __m128i mask = _mm_cmpgt_epi32(_mm_xor_si128(a.val, delta), _mm_xor_si128(b.val, delta));
    return v_uint32x4(v_select_si128(mask, a.val, b.val));
#endif
}
inline v_int32x4 v_min(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    return v_int32x4(_mm_min_epi32(a.val, b.val));
#else
    return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), b.val, a.val));
#endif
}
inline v_int32x4 v_max(const v_int32x4& a, const v_int32x4& b)
{
#if CV_SSE4_1
    return v_int32x4(_mm_max_epi32(a.val, b.val));
#else
    return v_int32x4(v_select_si128(_mm_cmpgt_epi32(a.val, b.val), a.val, b.val));
#endif
}

#define OPENCV_HAL_IMPL_SSE_INT_CMP_OP(_Tpuvec, _Tpsvec, suffix, sbit) \
inline _Tpuvec v_eq(const _Tpuvec& a, const _Tpuvec& b) \
{ return _Tpuvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpuvec v_ne(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpuvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec v_eq(const _Tpsvec& a, const _Tpsvec& b) \
{ return _Tpsvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpsvec v_ne(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpeq_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpuvec v_lt(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask))); \
} \
inline _Tpuvec v_gt(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    return _Tpuvec(_mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask))); \
} \
inline _Tpuvec v_le(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(a.val, smask), _mm_xor_si128(b.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpuvec v_ge(const _Tpuvec& a, const _Tpuvec& b) \
{ \
    __m128i smask = _mm_set1_##suffix(sbit); \
    __m128i not_mask = _mm_set1_epi32(-1); \
    __m128i res = _mm_cmpgt_##suffix(_mm_xor_si128(b.val, smask), _mm_xor_si128(a.val, smask)); \
    return _Tpuvec(_mm_xor_si128(res, not_mask)); \
} \
inline _Tpsvec v_lt(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(b.val, a.val)); \
} \
inline _Tpsvec v_gt(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    return _Tpsvec(_mm_cmpgt_##suffix(a.val, b.val)); \
} \
inline _Tpsvec v_le(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(a.val, b.val), not_mask)); \
} \
inline _Tpsvec v_ge(const _Tpsvec& a, const _Tpsvec& b) \
{ \
    __m128i not_mask = _mm_set1_epi32(-1); \
    return _Tpsvec(_mm_xor_si128(_mm_cmpgt_##suffix(b.val, a.val), not_mask)); \
}

OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint8x16, v_int8x16, epi8, (char)-128)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint16x8, v_int16x8, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_INT_CMP_OP(v_uint32x4, v_int32x4, epi32, (int)0x80000000)

#define OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(_Tpvec, suffix) \
inline _Tpvec v_eq(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpeq_##suffix(a.val, b.val)); } \
inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpneq_##suffix(a.val, b.val)); } \
inline _Tpvec v_lt(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmplt_##suffix(a.val, b.val)); } \
inline _Tpvec v_gt(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpgt_##suffix(a.val, b.val)); } \
inline _Tpvec v_le(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmple_##suffix(a.val, b.val)); } \
inline _Tpvec v_ge(const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpge_##suffix(a.val, b.val)); }

OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(v_float32x4, ps)
OPENCV_HAL_IMPL_SSE_FLT_CMP_OP(v_float64x2, pd)

#if CV_SSE4_1
#define OPENCV_HAL_IMPL_SSE_64BIT_CMP_OP(_Tpvec) \
inline _Tpvec v_eq (const _Tpvec& a, const _Tpvec& b) \
{ return _Tpvec(_mm_cmpeq_epi64(a.val, b.val)); } \
inline _Tpvec v_ne (const _Tpvec& a, const _Tpvec& b) \
{ return v_not(v_eq(a, b)); }
#else
#define OPENCV_HAL_IMPL_SSE_64BIT_CMP_OP(_Tpvec) \
inline _Tpvec v_eq(const _Tpvec& a, const _Tpvec& b) \
{ __m128i cmp = _mm_cmpeq_epi32(a.val, b.val); \
  return _Tpvec(_mm_and_si128(cmp, _mm_shuffle_epi32(cmp, _MM_SHUFFLE(2, 3, 0, 1)))); } \
inline _Tpvec v_ne(const _Tpvec& a, const _Tpvec& b) \
{ return v_not(v_eq(a, b)); }
#endif

OPENCV_HAL_IMPL_SSE_64BIT_CMP_OP(v_uint64x2)
OPENCV_HAL_IMPL_SSE_64BIT_CMP_OP(v_int64x2)

inline v_float32x4 v_not_nan(const v_float32x4& a)
{ return v_float32x4(_mm_cmpord_ps(a.val, a.val)); }
inline v_float64x2 v_not_nan(const v_float64x2& a)
{ return v_float64x2(_mm_cmpord_pd(a.val, a.val)); }

OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_add_wrap, _mm_add_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_add_wrap, _mm_add_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int8x16, v_sub_wrap, _mm_sub_epi8)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_sub_wrap, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_sub_wrap, _mm_sub_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_uint16x8, v_mul_wrap, _mm_mullo_epi16)
OPENCV_HAL_IMPL_SSE_BIN_FUNC(v_int16x8, v_mul_wrap, _mm_mullo_epi16)

inline v_uint8x16 v_mul_wrap(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i ad = _mm_srai_epi16(a.val, 8);
    __m128i bd = _mm_srai_epi16(b.val, 8);
    __m128i p0 = _mm_mullo_epi16(a.val, b.val); // even
    __m128i p1 = _mm_slli_epi16(_mm_mullo_epi16(ad, bd), 8); // odd
    const __m128i b01 = _mm_set1_epi32(0xFF00FF00);
    return v_uint8x16(_v128_blendv_epi8(p0, p1, b01));
}
inline v_int8x16 v_mul_wrap(const v_int8x16& a, const v_int8x16& b)
{
    return v_reinterpret_as_s8(v_mul_wrap(v_reinterpret_as_u8(a), v_reinterpret_as_u8(b)));
}

/** Absolute difference **/

inline v_uint8x16 v_absdiff(const v_uint8x16& a, const v_uint8x16& b)
{ return v_add_wrap(v_sub(a, b),  v_sub(b, a)); }
inline v_uint16x8 v_absdiff(const v_uint16x8& a, const v_uint16x8& b)
{ return v_add_wrap(v_sub(a, b),  v_sub(b, a)); }
inline v_uint32x4 v_absdiff(const v_uint32x4& a, const v_uint32x4& b)
{ return v_sub(v_max(a, b), v_min(a, b)); }

inline v_uint8x16 v_absdiff(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = v_sub_wrap(a, b);
    v_int8x16 m = v_lt(a, b);
    return v_reinterpret_as_u8(v_sub_wrap(v_xor(d, m), m));
}
inline v_uint16x8 v_absdiff(const v_int16x8& a, const v_int16x8& b)
{
    return v_reinterpret_as_u16(v_sub_wrap(v_max(a, b), v_min(a, b)));
}
inline v_uint32x4 v_absdiff(const v_int32x4& a, const v_int32x4& b)
{
    v_int32x4 d = v_sub(a, b);
    v_int32x4 m = v_lt(a, b);
    return v_reinterpret_as_u32(v_sub(v_xor(d, m), m));
}

/** Saturating absolute difference **/
inline v_int8x16 v_absdiffs(const v_int8x16& a, const v_int8x16& b)
{
    v_int8x16 d = v_sub(a, b);
    v_int8x16 m = v_lt(a, b);
    return v_sub(v_xor(d, m), m);
 }
inline v_int16x8 v_absdiffs(const v_int16x8& a, const v_int16x8& b)
{ return v_sub(v_max(a, b), v_min(a, b)); }


inline v_int32x4 v_fma(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_add(v_mul(a, b), c);
}

inline v_int32x4 v_muladd(const v_int32x4& a, const v_int32x4& b, const v_int32x4& c)
{
    return v_fma(a, b, c);
}

inline v_float32x4 v_fma(const v_float32x4& a, const v_float32x4& b, const v_float32x4& c)
{
#if CV_FMA3
    return v_float32x4(_mm_fmadd_ps(a.val, b.val, c.val));
#else
    return v_float32x4(_mm_add_ps(_mm_mul_ps(a.val, b.val), c.val));
#endif
}

inline v_float64x2 v_fma(const v_float64x2& a, const v_float64x2& b, const v_float64x2& c)
{
#if CV_FMA3
    return v_float64x2(_mm_fmadd_pd(a.val, b.val, c.val));
#else
    return v_float64x2(_mm_add_pd(_mm_mul_pd(a.val, b.val), c.val));
#endif
}

#define OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(_Tpvec, _Tp, _Tpreg, suffix, absmask_vec) \
inline _Tpvec v_absdiff(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpreg absmask = _mm_castsi128_##suffix(absmask_vec); \
    return _Tpvec(_mm_and_##suffix(_mm_sub_##suffix(a.val, b.val), absmask)); \
} \
inline _Tpvec v_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    _Tpvec res = v_fma(a, a, v_mul(b, b)); \
    return _Tpvec(_mm_sqrt_##suffix(res.val)); \
} \
inline _Tpvec v_sqr_magnitude(const _Tpvec& a, const _Tpvec& b) \
{ \
    return v_fma(a, a, v_mul(b, b)); \
} \
inline _Tpvec v_muladd(const _Tpvec& a, const _Tpvec& b, const _Tpvec& c) \
{ \
    return v_fma(a, b, c); \
}

OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(v_float32x4, float, __m128, ps, _mm_set1_epi32((int)0x7fffffff))
OPENCV_HAL_IMPL_SSE_MISC_FLT_OP(v_float64x2, double, __m128d, pd, _mm_srli_epi64(_mm_set1_epi32(-1), 1))

#define OPENCV_HAL_IMPL_SSE_SHIFT_OP(_Tpuvec, _Tpsvec, suffix, srai) \
inline _Tpuvec v_shl(const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(_mm_slli_##suffix(a.val, imm)); \
} \
inline _Tpsvec v_shl(const _Tpsvec& a, int imm) \
{ \
    return _Tpsvec(_mm_slli_##suffix(a.val, imm)); \
} \
inline _Tpuvec v_shr(const _Tpuvec& a, int imm) \
{ \
    return _Tpuvec(_mm_srli_##suffix(a.val, imm)); \
} \
inline _Tpsvec v_shr(const _Tpsvec& a, int imm) \
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

namespace hal_sse_internal
{
    template <int imm,
        bool is_invalid = ((imm < 0) || (imm > 16)),
        bool is_first = (imm == 0),
        bool is_half = (imm == 8),
        bool is_second = (imm == 16),
        bool is_other = (((imm > 0) && (imm < 8)) || ((imm > 8) && (imm < 16)))>
    class v_sse_palignr_u8_class;

    template <int imm>
    class v_sse_palignr_u8_class<imm, true, false, false, false, false>;

    template <int imm>
    class v_sse_palignr_u8_class<imm, false, true, false, false, false>
    {
    public:
        inline __m128i operator()(const __m128i& a, const __m128i&) const
        {
            return a;
        }
    };

    template <int imm>
    class v_sse_palignr_u8_class<imm, false, false, true, false, false>
    {
    public:
        inline __m128i operator()(const __m128i& a, const __m128i& b) const
        {
            return _mm_unpacklo_epi64(_mm_unpackhi_epi64(a, a), b);
        }
    };

    template <int imm>
    class v_sse_palignr_u8_class<imm, false, false, false, true, false>
    {
    public:
        inline __m128i operator()(const __m128i&, const __m128i& b) const
        {
            return b;
        }
    };

    template <int imm>
    class v_sse_palignr_u8_class<imm, false, false, false, false, true>
    {
#if CV_SSSE3
    public:
        inline __m128i operator()(const __m128i& a, const __m128i& b) const
        {
            return _mm_alignr_epi8(b, a, imm);
        }
#else
    public:
        inline __m128i operator()(const __m128i& a, const __m128i& b) const
        {
            enum { imm2 = (sizeof(__m128i) - imm) };
            return _mm_or_si128(_mm_srli_si128(a, imm), _mm_slli_si128(b, imm2));
        }
#endif
    };

    template <int imm>
    inline __m128i v_sse_palignr_u8(const __m128i& a, const __m128i& b)
    {
        CV_StaticAssert((imm >= 0) && (imm <= 16), "Invalid imm for v_sse_palignr_u8.");
        return v_sse_palignr_u8_class<imm>()(a, b);
    }
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_right(const _Tpvec &a)
{
    using namespace hal_sse_internal;
    enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
    return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
        _mm_srli_si128(
            v_sse_reinterpret_as<__m128i>(a.val), imm2)));
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_left(const _Tpvec &a)
{
    using namespace hal_sse_internal;
    enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
    return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
        _mm_slli_si128(
            v_sse_reinterpret_as<__m128i>(a.val), imm2)));
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_right(const _Tpvec &a, const _Tpvec &b)
{
    using namespace hal_sse_internal;
    enum { imm2 = (imm * sizeof(typename _Tpvec::lane_type)) };
    return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
        v_sse_palignr_u8<imm2>(
            v_sse_reinterpret_as<__m128i>(a.val),
            v_sse_reinterpret_as<__m128i>(b.val))));
}

template<int imm, typename _Tpvec>
inline _Tpvec v_rotate_left(const _Tpvec &a, const _Tpvec &b)
{
    using namespace hal_sse_internal;
    enum { imm2 = ((_Tpvec::nlanes - imm) * sizeof(typename _Tpvec::lane_type)) };
    return _Tpvec(v_sse_reinterpret_as<typename _Tpvec::vector_type>(
        v_sse_palignr_u8<imm2>(
            v_sse_reinterpret_as<__m128i>(b.val),
            v_sse_reinterpret_as<__m128i>(a.val))));
}

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_INT_OP(_Tpvec, _Tp) \
inline _Tpvec v_load(const _Tp* ptr) \
{ return _Tpvec(_mm_loadu_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_aligned(const _Tp* ptr) \
{ return _Tpvec(_mm_load_si128((const __m128i*)ptr)); } \
inline _Tpvec v_load_low(const _Tp* ptr) \
{ return _Tpvec(_mm_loadl_epi64((const __m128i*)ptr)); } \
inline _Tpvec v_load_halves(const _Tp* ptr0, const _Tp* ptr1) \
{ \
    return _Tpvec(_mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)ptr0), \
                                     _mm_loadl_epi64((const __m128i*)ptr1))); \
} \
inline void v_store(_Tp* ptr, const _Tpvec& a) \
{ _mm_storeu_si128((__m128i*)ptr, a.val); } \
inline void v_store_aligned(_Tp* ptr, const _Tpvec& a) \
{ _mm_store_si128((__m128i*)ptr, a.val); } \
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ _mm_stream_si128((__m128i*)ptr, a.val); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
{ \
    if( mode == hal::STORE_UNALIGNED ) \
        _mm_storeu_si128((__m128i*)ptr, a.val); \
    else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
        _mm_stream_si128((__m128i*)ptr, a.val); \
    else \
        _mm_store_si128((__m128i*)ptr, a.val); \
} \
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
inline _Tpvec v_load_low(const _Tp* ptr) \
{ return _Tpvec(_mm_castsi128_##suffix(_mm_loadl_epi64((const __m128i*)ptr))); } \
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
inline void v_store_aligned_nocache(_Tp* ptr, const _Tpvec& a) \
{ _mm_stream_##suffix(ptr, a.val); } \
inline void v_store(_Tp* ptr, const _Tpvec& a, hal::StoreMode mode) \
{ \
    if( mode == hal::STORE_UNALIGNED ) \
        _mm_storeu_##suffix(ptr, a.val); \
    else if( mode == hal::STORE_ALIGNED_NOCACHE )  \
        _mm_stream_##suffix(ptr, a.val); \
    else \
        _mm_store_##suffix(ptr, a.val); \
} \
inline void v_store_low(_Tp* ptr, const _Tpvec& a) \
{ _mm_storel_epi64((__m128i*)ptr, _mm_cast##suffix##_si128(a.val)); } \
inline void v_store_high(_Tp* ptr, const _Tpvec& a) \
{ \
    __m128i a1 = _mm_cast##suffix##_si128(a.val); \
    _mm_storel_epi64((__m128i*)ptr, _mm_unpackhi_epi64(a1, a1)); \
}

OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(v_float32x4, float, ps)
OPENCV_HAL_IMPL_SSE_LOADSTORE_FLT_OP(v_float64x2, double, pd)

inline unsigned v_reduce_sum(const v_uint8x16& a)
{
    __m128i half = _mm_sad_epu8(a.val, _mm_setzero_si128());
    return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half)));
}
inline int v_reduce_sum(const v_int8x16& a)
{
    __m128i half = _mm_set1_epi8((schar)-128);
    half = _mm_sad_epu8(_mm_xor_si128(a.val, half), _mm_setzero_si128());
    return _mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half))) - 2048;
}
#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_16(func) \
inline schar v_reduce_##func(const v_int8x16& a) \
{ \
    __m128i val = a.val; \
    __m128i smask = _mm_set1_epi8((schar)-128); \
    val = _mm_xor_si128(val, smask); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,2)); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,1)); \
    return (schar)_mm_cvtsi128_si32(val) ^ (schar)-128; \
} \
inline uchar v_reduce_##func(const v_uint8x16& a) \
{ \
    __m128i val = a.val; \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,8)); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,4)); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,2)); \
    val = _mm_##func##_epu8(val, _mm_srli_si128(val,1)); \
    return (uchar)_mm_cvtsi128_si32(val); \
}
OPENCV_HAL_IMPL_SSE_REDUCE_OP_16(max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_16(min)

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
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, max, epi16, (short)-32768)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_8(int16x8, short, min, epi16, (short)-32768)

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(_Tpvec, scalartype, regtype, suffix, cast_from, cast_to, extract) \
inline scalartype v_reduce_sum(const _Tpvec& a) \
{ \
    regtype val = a.val; \
    val = _mm_add_##suffix(val, cast_to(_mm_srli_si128(cast_from(val), 8))); \
    val = _mm_add_##suffix(val, cast_to(_mm_srli_si128(cast_from(val), 4))); \
    return (scalartype)_mm_cvt##extract(val); \
}

#define OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(_Tpvec, scalartype, func, scalar_func) \
inline scalartype v_reduce_##func(const _Tpvec& a) \
{ \
    scalartype CV_DECL_ALIGNED(16) buf[4]; \
    v_store_aligned(buf, a); \
    scalartype s0 = scalar_func(buf[0], buf[1]); \
    scalartype s1 = scalar_func(buf[2], buf[3]); \
    return scalar_func(s0, s1); \
}

OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_uint32x4, unsigned, __m128i, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP, si128_si32)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_int32x4, int, __m128i, epi32, OPENCV_HAL_NOP, OPENCV_HAL_NOP, si128_si32)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4_SUM(v_float32x4, float, __m128, ps, _mm_castps_si128, _mm_castsi128_ps, ss_f32)

inline int v_reduce_sum(const v_int16x8& a)
{ return v_reduce_sum(v_add(v_expand_low(a), v_expand_high(a))); }
inline unsigned v_reduce_sum(const v_uint16x8& a)
{ return v_reduce_sum(v_add(v_expand_low(a), v_expand_high(a))); }

inline uint64 v_reduce_sum(const v_uint64x2& a)
{
    uint64 CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return idx[0] + idx[1];
}
inline int64 v_reduce_sum(const v_int64x2& a)
{
    int64 CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return idx[0] + idx[1];
}
inline double v_reduce_sum(const v_float64x2& a)
{
    double CV_DECL_ALIGNED(32) idx[2];
    v_store_aligned(idx, a);
    return idx[0] + idx[1];
}

inline v_float32x4 v_reduce_sum4(const v_float32x4& a, const v_float32x4& b,
                                 const v_float32x4& c, const v_float32x4& d)
{
#if CV_SSE3
    __m128 ab = _mm_hadd_ps(a.val, b.val);
    __m128 cd = _mm_hadd_ps(c.val, d.val);
    return v_float32x4(_mm_hadd_ps(ab, cd));
#else
    __m128 ac = _mm_add_ps(_mm_unpacklo_ps(a.val, c.val), _mm_unpackhi_ps(a.val, c.val));
    __m128 bd = _mm_add_ps(_mm_unpacklo_ps(b.val, d.val), _mm_unpackhi_ps(b.val, d.val));
    return v_float32x4(_mm_add_ps(_mm_unpacklo_ps(ac, bd), _mm_unpackhi_ps(ac, bd)));
#endif
}

OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_uint32x4, unsigned, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_int32x4, int, min, std::min)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, max, std::max)
OPENCV_HAL_IMPL_SSE_REDUCE_OP_4(v_float32x4, float, min, std::min)

inline unsigned v_reduce_sad(const v_uint8x16& a, const v_uint8x16& b)
{
    __m128i half = _mm_sad_epu8(a.val, b.val);
    return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half)));
}
inline unsigned v_reduce_sad(const v_int8x16& a, const v_int8x16& b)
{
    __m128i half = _mm_set1_epi8(0x7f);
    half = _mm_sad_epu8(_mm_add_epi8(a.val, half), _mm_add_epi8(b.val, half));
    return (unsigned)_mm_cvtsi128_si32(_mm_add_epi32(half, _mm_unpackhi_epi64(half, half)));
}
inline unsigned v_reduce_sad(const v_uint16x8& a, const v_uint16x8& b)
{
    v_uint32x4 l, h;
    v_expand(v_absdiff(a, b), l, h);
    return v_reduce_sum(v_add(l, h));
}
inline unsigned v_reduce_sad(const v_int16x8& a, const v_int16x8& b)
{
    v_uint32x4 l, h;
    v_expand(v_absdiff(a, b), l, h);
    return v_reduce_sum(v_add(l, h));
}
inline unsigned v_reduce_sad(const v_uint32x4& a, const v_uint32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}
inline unsigned v_reduce_sad(const v_int32x4& a, const v_int32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}
inline float v_reduce_sad(const v_float32x4& a, const v_float32x4& b)
{
    return v_reduce_sum(v_absdiff(a, b));
}

inline v_uint8x16 v_popcount(const v_uint8x16& a)
{
    __m128i m1 = _mm_set1_epi32(0x55555555);
    __m128i m2 = _mm_set1_epi32(0x33333333);
    __m128i m4 = _mm_set1_epi32(0x0f0f0f0f);
    __m128i p = a.val;
    p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 1), m1), _mm_and_si128(p, m1));
    p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 2), m2), _mm_and_si128(p, m2));
    p = _mm_add_epi32(_mm_and_si128(_mm_srli_epi32(p, 4), m4), _mm_and_si128(p, m4));
    return v_uint8x16(p);
}
inline v_uint16x8 v_popcount(const v_uint16x8& a)
{
    v_uint8x16 p = v_popcount(v_reinterpret_as_u8(a));
    p = v_add(p, v_rotate_right<1>(p));
    return v_and(v_reinterpret_as_u16(p), v_setall_u16(0x00ff));
}
inline v_uint32x4 v_popcount(const v_uint32x4& a)
{
    v_uint8x16 p = v_popcount(v_reinterpret_as_u8(a));
    p = v_add(p, v_rotate_right<1>(p));
    p = v_add(p, v_rotate_right<2>(p));
    return v_and(v_reinterpret_as_u32(p), v_setall_u32(0x000000ff));
}
inline v_uint64x2 v_popcount(const v_uint64x2& a)
{
    return v_uint64x2(_mm_sad_epu8(v_popcount(v_reinterpret_as_u8(a)).val, _mm_setzero_si128()));
}
inline v_uint8x16 v_popcount(const v_int8x16& a)
{ return v_popcount(v_reinterpret_as_u8(a)); }
inline v_uint16x8 v_popcount(const v_int16x8& a)
{ return v_popcount(v_reinterpret_as_u16(a)); }
inline v_uint32x4 v_popcount(const v_int32x4& a)
{ return v_popcount(v_reinterpret_as_u32(a)); }
inline v_uint64x2 v_popcount(const v_int64x2& a)
{ return v_popcount(v_reinterpret_as_u64(a)); }

#define OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(_Tpvec, suffix, cast_op, allmask) \
inline int v_signmask(const _Tpvec& a)   { return _mm_movemask_##suffix(cast_op(a.val)); } \
inline bool v_check_all(const _Tpvec& a) { return _mm_movemask_##suffix(cast_op(a.val)) == allmask; } \
inline bool v_check_any(const _Tpvec& a) { return _mm_movemask_##suffix(cast_op(a.val)) != 0; }
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint8x16, epi8, OPENCV_HAL_NOP, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int8x16, epi8, OPENCV_HAL_NOP, 65535)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint32x4, ps, _mm_castsi128_ps, 15)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int32x4, ps, _mm_castsi128_ps, 15)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_uint64x2, pd, _mm_castsi128_pd, 3)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_int64x2, pd, _mm_castsi128_pd, 3)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float32x4, ps, OPENCV_HAL_NOP, 15)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS(v_float64x2, pd, OPENCV_HAL_NOP, 3)

#define OPENCV_HAL_IMPL_SSE_CHECK_SIGNS_SHORT(_Tpvec) \
inline int v_signmask(const _Tpvec& a) { return _mm_movemask_epi8(_mm_packs_epi16(a.val, a.val)) & 255; } \
inline bool v_check_all(const _Tpvec& a) { return (_mm_movemask_epi8(a.val) & 0xaaaa) == 0xaaaa; } \
inline bool v_check_any(const _Tpvec& a) { return (_mm_movemask_epi8(a.val) & 0xaaaa) != 0; }
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS_SHORT(v_uint16x8)
OPENCV_HAL_IMPL_SSE_CHECK_SIGNS_SHORT(v_int16x8)

inline int v_scan_forward(const v_int8x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_uint8x16& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))); }
inline int v_scan_forward(const v_int16x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_uint16x8& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 2; }
inline int v_scan_forward(const v_int32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_uint32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_float32x4& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 4; }
inline int v_scan_forward(const v_int64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_uint64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }
inline int v_scan_forward(const v_float64x2& a) { return trailingZeros32(v_signmask(v_reinterpret_as_s8(a))) / 8; }

#if CV_SSE4_1
#define OPENCV_HAL_IMPL_SSE_SELECT(_Tpvec, cast_ret, cast, suffix) \
inline _Tpvec v_select(const _Tpvec& mask, const _Tpvec& a, const _Tpvec& b) \
{ \
    return _Tpvec(cast_ret(_mm_blendv_##suffix(cast(b.val), cast(a.val), cast(mask.val)))); \
}

OPENCV_HAL_IMPL_SSE_SELECT(v_uint8x16, OPENCV_HAL_NOP, OPENCV_HAL_NOP, epi8)
OPENCV_HAL_IMPL_SSE_SELECT(v_int8x16, OPENCV_HAL_NOP, OPENCV_HAL_NOP, epi8)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint16x8, OPENCV_HAL_NOP, OPENCV_HAL_NOP, epi8)
OPENCV_HAL_IMPL_SSE_SELECT(v_int16x8, OPENCV_HAL_NOP, OPENCV_HAL_NOP, epi8)
OPENCV_HAL_IMPL_SSE_SELECT(v_uint32x4, _mm_castps_si128, _mm_castsi128_ps, ps)
OPENCV_HAL_IMPL_SSE_SELECT(v_int32x4, _mm_castps_si128, _mm_castsi128_ps, ps)
// OPENCV_HAL_IMPL_SSE_SELECT(v_uint64x2, TBD, TBD, pd)
// OPENCV_HAL_IMPL_SSE_SELECT(v_int64x2, TBD, TBD, ps)
OPENCV_HAL_IMPL_SSE_SELECT(v_float32x4, OPENCV_HAL_NOP, OPENCV_HAL_NOP, ps)
OPENCV_HAL_IMPL_SSE_SELECT(v_float64x2, OPENCV_HAL_NOP, OPENCV_HAL_NOP, pd)

#else // CV_SSE4_1

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
#endif

/* Expand */
#define OPENCV_HAL_IMPL_SSE_EXPAND(_Tpvec, _Tpwvec, _Tp, intrin)    \
    inline void v_expand(const _Tpvec& a, _Tpwvec& b0, _Tpwvec& b1) \
    {                                                               \
        b0.val = intrin(a.val);                                     \
        b1.val = __CV_CAT(intrin, _high)(a.val);                    \
    }                                                               \
    inline _Tpwvec v_expand_low(const _Tpvec& a)                    \
    { return _Tpwvec(intrin(a.val)); }                              \
    inline _Tpwvec v_expand_high(const _Tpvec& a)                   \
    { return _Tpwvec(__CV_CAT(intrin, _high)(a.val)); }             \
    inline _Tpwvec v_load_expand(const _Tp* ptr)                    \
    {                                                               \
        __m128i a = _mm_loadl_epi64((const __m128i*)ptr);           \
        return _Tpwvec(intrin(a));                                  \
    }

OPENCV_HAL_IMPL_SSE_EXPAND(v_uint8x16, v_uint16x8,  uchar,    _v128_cvtepu8_epi16)
OPENCV_HAL_IMPL_SSE_EXPAND(v_int8x16,  v_int16x8,   schar,    _v128_cvtepi8_epi16)
OPENCV_HAL_IMPL_SSE_EXPAND(v_uint16x8, v_uint32x4,  ushort,   _v128_cvtepu16_epi32)
OPENCV_HAL_IMPL_SSE_EXPAND(v_int16x8,  v_int32x4,   short,    _v128_cvtepi16_epi32)
OPENCV_HAL_IMPL_SSE_EXPAND(v_uint32x4, v_uint64x2,  unsigned, _v128_cvtepu32_epi64)
OPENCV_HAL_IMPL_SSE_EXPAND(v_int32x4,  v_int64x2,   int,      _v128_cvtepi32_epi64)

#define OPENCV_HAL_IMPL_SSE_EXPAND_Q(_Tpvec, _Tp, intrin)          \
    inline _Tpvec v_load_expand_q(const _Tp* ptr)                  \
    {                                                              \
        typedef int CV_DECL_ALIGNED(1) unaligned_int;              \
        __m128i a = _mm_cvtsi32_si128(*(const unaligned_int*)ptr); \
        return _Tpvec(intrin(a));                                  \
    }

OPENCV_HAL_IMPL_SSE_EXPAND_Q(v_uint32x4, uchar, _v128_cvtepu8_epi32)
OPENCV_HAL_IMPL_SSE_EXPAND_Q(v_int32x4,  schar, _v128_cvtepi8_epi32)

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

inline v_uint8x16 v_reverse(const v_uint8x16 &a)
{
#if CV_SSSE3
    static const __m128i perm = _mm_setr_epi8(15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    return v_uint8x16(_mm_shuffle_epi8(a.val, perm));
#else
    uchar CV_DECL_ALIGNED(32) d[16];
    v_store_aligned(d, a);
    return v_uint8x16(d[15], d[14], d[13], d[12], d[11], d[10], d[9], d[8], d[7], d[6], d[5], d[4], d[3], d[2], d[1], d[0]);
#endif
}

inline v_int8x16 v_reverse(const v_int8x16 &a)
{ return v_reinterpret_as_s8(v_reverse(v_reinterpret_as_u8(a))); }

inline v_uint16x8 v_reverse(const v_uint16x8 &a)
{
#if CV_SSSE3
    static const __m128i perm = _mm_setr_epi8(14, 15, 12, 13, 10, 11, 8, 9, 6, 7, 4, 5, 2, 3, 0, 1);
    return v_uint16x8(_mm_shuffle_epi8(a.val, perm));
#else
    __m128i r = _mm_shuffle_epi32(a.val, _MM_SHUFFLE(0, 1, 2, 3));
    r = _mm_shufflelo_epi16(r, _MM_SHUFFLE(2, 3, 0, 1));
    r = _mm_shufflehi_epi16(r, _MM_SHUFFLE(2, 3, 0, 1));
    return v_uint16x8(r);
#endif
}

inline v_int16x8 v_reverse(const v_int16x8 &a)
{ return v_reinterpret_as_s16(v_reverse(v_reinterpret_as_u16(a))); }

inline v_uint32x4 v_reverse(const v_uint32x4 &a)
{
    return v_uint32x4(_mm_shuffle_epi32(a.val, _MM_SHUFFLE(0, 1, 2, 3)));
}

inline v_int32x4 v_reverse(const v_int32x4 &a)
{ return v_reinterpret_as_s32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_float32x4 v_reverse(const v_float32x4 &a)
{ return v_reinterpret_as_f32(v_reverse(v_reinterpret_as_u32(a))); }

inline v_uint64x2 v_reverse(const v_uint64x2 &a)
{
    return v_uint64x2(_mm_shuffle_epi32(a.val, _MM_SHUFFLE(1, 0, 3, 2)));
}

inline v_int64x2 v_reverse(const v_int64x2 &a)
{ return v_reinterpret_as_s64(v_reverse(v_reinterpret_as_u64(a))); }

inline v_float64x2 v_reverse(const v_float64x2 &a)
{ return v_reinterpret_as_f64(v_reverse(v_reinterpret_as_u64(a))); }

template<int s, typename _Tpvec>
inline _Tpvec v_extract(const _Tpvec& a, const _Tpvec& b)
{
    return v_rotate_right<s>(a, b);
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

inline v_int32x4 v_round(const v_float64x2& a, const v_float64x2& b)
{
    __m128i ai = _mm_cvtpd_epi32(a.val), bi = _mm_cvtpd_epi32(b.val);
    return v_int32x4(_mm_unpacklo_epi64(ai, bi));
}

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

// load deinterleave
inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b)
{
    __m128i t00 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t01 = _mm_loadu_si128((const __m128i*)(ptr + 16));

    __m128i t10 = _mm_unpacklo_epi8(t00, t01);
    __m128i t11 = _mm_unpackhi_epi8(t00, t01);

    __m128i t20 = _mm_unpacklo_epi8(t10, t11);
    __m128i t21 = _mm_unpackhi_epi8(t10, t11);

    __m128i t30 = _mm_unpacklo_epi8(t20, t21);
    __m128i t31 = _mm_unpackhi_epi8(t20, t21);

    a.val = _mm_unpacklo_epi8(t30, t31);
    b.val = _mm_unpackhi_epi8(t30, t31);
}

inline void v_load_deinterleave(const uchar* ptr, v_uint8x16& a, v_uint8x16& b, v_uint8x16& c)
{
#if CV_SSE4_1
    const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i s0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i s1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i s2 = _mm_loadu_si128((const __m128i*)(ptr + 32));
    __m128i a0 = _mm_blendv_epi8(_mm_blendv_epi8(s0, s1, m0), s2, m1);
    __m128i b0 = _mm_blendv_epi8(_mm_blendv_epi8(s1, s2, m0), s0, m1);
    __m128i c0 = _mm_blendv_epi8(_mm_blendv_epi8(s2, s0, m0), s1, m1);
    const __m128i sh_b = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14, 1, 4, 7, 10, 13);
    const __m128i sh_g = _mm_setr_epi8(1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15, 2, 5, 8, 11, 14);
    const __m128i sh_r = _mm_setr_epi8(2, 5, 8, 11, 14, 1, 4, 7, 10, 13, 0, 3, 6, 9, 12, 15);
    a0 = _mm_shuffle_epi8(a0, sh_b);
    b0 = _mm_shuffle_epi8(b0, sh_g);
    c0 = _mm_shuffle_epi8(c0, sh_r);
    a.val = a0;
    b.val = b0;
    c.val = c0;
#elif CV_SSSE3
    const __m128i m0 = _mm_setr_epi8(0, 3, 6, 9, 12, 15, 1, 4, 7, 10, 13, 2, 5, 8, 11, 14);
    const __m128i m1 = _mm_alignr_epi8(m0, m0, 11);
    const __m128i m2 = _mm_alignr_epi8(m0, m0, 6);

    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 16));
    __m128i t2 = _mm_loadu_si128((const __m128i*)(ptr + 32));

    __m128i s0 = _mm_shuffle_epi8(t0, m0);
    __m128i s1 = _mm_shuffle_epi8(t1, m1);
    __m128i s2 = _mm_shuffle_epi8(t2, m2);

    t0 = _mm_alignr_epi8(s1, _mm_slli_si128(s0, 10), 5);
    a.val = _mm_alignr_epi8(s2, t0, 5);

    t1 = _mm_alignr_epi8(_mm_srli_si128(s1, 5), _mm_slli_si128(s0, 5), 6);
    b.val = _mm_alignr_epi8(_mm_srli_si128(s2, 5), t1, 5);

    t2 = _mm_alignr_epi8(_mm_srli_si128(s2, 10), s1, 11);
    c.val = _mm_alignr_epi8(t2, s0, 11);
#else
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
#endif
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

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b)
{
    __m128i v0 = _mm_loadu_si128((__m128i*)(ptr));     // a0 b0 a1 b1 a2 b2 a3 b3
    __m128i v1 = _mm_loadu_si128((__m128i*)(ptr + 8)); // a4 b4 a5 b5 a6 b6 a7 b7

    __m128i v2 = _mm_unpacklo_epi16(v0, v1); // a0 a4 b0 b4 a1 a5 b1 b5
    __m128i v3 = _mm_unpackhi_epi16(v0, v1); // a2 a6 b2 b6 a3 a7 b3 b7
    __m128i v4 = _mm_unpacklo_epi16(v2, v3); // a0 a2 a4 a6 b0 b2 b4 b6
    __m128i v5 = _mm_unpackhi_epi16(v2, v3); // a1 a3 a5 a7 b1 b3 b5 b7

    a.val = _mm_unpacklo_epi16(v4, v5); // a0 a1 a2 a3 a4 a5 a6 a7
    b.val = _mm_unpackhi_epi16(v4, v5); // b0 b1 ab b3 b4 b5 b6 b7
}

inline void v_load_deinterleave(const ushort* ptr, v_uint16x8& a, v_uint16x8& b, v_uint16x8& c)
{
#if CV_SSE4_1
    __m128i v0 = _mm_loadu_si128((__m128i*)(ptr));
    __m128i v1 = _mm_loadu_si128((__m128i*)(ptr + 8));
    __m128i v2 = _mm_loadu_si128((__m128i*)(ptr + 16));
    __m128i a0 = _mm_blend_epi16(_mm_blend_epi16(v0, v1, 0x92), v2, 0x24);
    __m128i b0 = _mm_blend_epi16(_mm_blend_epi16(v2, v0, 0x92), v1, 0x24);
    __m128i c0 = _mm_blend_epi16(_mm_blend_epi16(v1, v2, 0x92), v0, 0x24);

    const __m128i sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m128i sh_b = _mm_setr_epi8(2, 3, 8, 9, 14, 15, 4, 5, 10, 11, 0, 1, 6, 7, 12, 13);
    const __m128i sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    a0 = _mm_shuffle_epi8(a0, sh_a);
    b0 = _mm_shuffle_epi8(b0, sh_b);
    c0 = _mm_shuffle_epi8(c0, sh_c);

    a.val = a0;
    b.val = b0;
    c.val = c0;
#else
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
#endif
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

inline void v_load_deinterleave(const unsigned* ptr, v_uint32x4& a, v_uint32x4& b)
{
    __m128i v0 = _mm_loadu_si128((__m128i*)(ptr));     // a0 b0 a1 b1
    __m128i v1 = _mm_loadu_si128((__m128i*)(ptr + 4)); // a2 b2 a3 b3

    __m128i v2 = _mm_unpacklo_epi32(v0, v1); // a0 a2 b0 b2
    __m128i v3 = _mm_unpackhi_epi32(v0, v1); // a1 a3 b1 b3

    a.val = _mm_unpacklo_epi32(v2, v3); // a0 a1 a2 a3
    b.val = _mm_unpackhi_epi32(v2, v3); // b0 b1 ab b3
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
    v_uint32x4 s0(_mm_loadu_si128((const __m128i*)ptr));        // a0 b0 c0 d0
    v_uint32x4 s1(_mm_loadu_si128((const __m128i*)(ptr + 4)));  // a1 b1 c1 d1
    v_uint32x4 s2(_mm_loadu_si128((const __m128i*)(ptr + 8)));  // a2 b2 c2 d2
    v_uint32x4 s3(_mm_loadu_si128((const __m128i*)(ptr + 12))); // a3 b3 c3 d3

    v_transpose4x4(s0, s1, s2, s3, a, b, c, d);
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b)
{
    __m128 u0 = _mm_loadu_ps(ptr);       // a0 b0 a1 b1
    __m128 u1 = _mm_loadu_ps((ptr + 4)); // a2 b2 a3 b3

    a.val = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(2, 0, 2, 0)); // a0 a1 a2 a3
    b.val = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(3, 1, 3, 1)); // b0 b1 ab b3
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c)
{
    __m128 t0 = _mm_loadu_ps(ptr + 0);
    __m128 t1 = _mm_loadu_ps(ptr + 4);
    __m128 t2 = _mm_loadu_ps(ptr + 8);

    __m128 at12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 1, 0, 2));
    a.val = _mm_shuffle_ps(t0, at12, _MM_SHUFFLE(2, 0, 3, 0));

    __m128 bt01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 0, 0, 1));
    __m128 bt12 = _mm_shuffle_ps(t1, t2, _MM_SHUFFLE(0, 2, 0, 3));
    b.val = _mm_shuffle_ps(bt01, bt12, _MM_SHUFFLE(2, 0, 2, 0));

    __m128 ct01 = _mm_shuffle_ps(t0, t1, _MM_SHUFFLE(0, 1, 0, 2));
    c.val = _mm_shuffle_ps(ct01, t2, _MM_SHUFFLE(3, 0, 2, 0));
}

inline void v_load_deinterleave(const float* ptr, v_float32x4& a, v_float32x4& b, v_float32x4& c, v_float32x4& d)
{
    __m128 t0 = _mm_loadu_ps(ptr +  0);
    __m128 t1 = _mm_loadu_ps(ptr +  4);
    __m128 t2 = _mm_loadu_ps(ptr +  8);
    __m128 t3 = _mm_loadu_ps(ptr + 12);
    __m128 t02lo = _mm_unpacklo_ps(t0, t2);
    __m128 t13lo = _mm_unpacklo_ps(t1, t3);
    __m128 t02hi = _mm_unpackhi_ps(t0, t2);
    __m128 t13hi = _mm_unpackhi_ps(t1, t3);
    a.val = _mm_unpacklo_ps(t02lo, t13lo);
    b.val = _mm_unpackhi_ps(t02lo, t13lo);
    c.val = _mm_unpacklo_ps(t02hi, t13hi);
    d.val = _mm_unpackhi_ps(t02hi, t13hi);
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b)
{
    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr);
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 2));

    a = v_uint64x2(_mm_unpacklo_epi64(t0, t1));
    b = v_uint64x2(_mm_unpackhi_epi64(t0, t1));
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a, v_uint64x2& b, v_uint64x2& c)
{
    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr); // a0, b0
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 2)); // c0, a1
    __m128i t2 = _mm_loadu_si128((const __m128i*)(ptr + 4)); // b1, c1

    t1 = _mm_shuffle_epi32(t1, 0x4e); // a1, c0

    a = v_uint64x2(_mm_unpacklo_epi64(t0, t1));
    b = v_uint64x2(_mm_unpacklo_epi64(_mm_unpackhi_epi64(t0, t0), t2));
    c = v_uint64x2(_mm_unpackhi_epi64(t1, t2));
}

inline void v_load_deinterleave(const uint64 *ptr, v_uint64x2& a,
                                v_uint64x2& b, v_uint64x2& c, v_uint64x2& d)
{
    __m128i t0 = _mm_loadu_si128((const __m128i*)ptr); // a0 b0
    __m128i t1 = _mm_loadu_si128((const __m128i*)(ptr + 2)); // c0 d0
    __m128i t2 = _mm_loadu_si128((const __m128i*)(ptr + 4)); // a1 b1
    __m128i t3 = _mm_loadu_si128((const __m128i*)(ptr + 6)); // c1 d1

    a = v_uint64x2(_mm_unpacklo_epi64(t0, t2));
    b = v_uint64x2(_mm_unpackhi_epi64(t0, t2));
    c = v_uint64x2(_mm_unpacklo_epi64(t1, t3));
    d = v_uint64x2(_mm_unpackhi_epi64(t1, t3));
}

// store interleave

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi8(a.val, b.val);
    __m128i v1 = _mm_unpackhi_epi8(a.val, b.val);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 16), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 16), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 16), v1);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
#if CV_SSE4_1
    const __m128i sh_a = _mm_setr_epi8(0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10, 5);
    const __m128i sh_b = _mm_setr_epi8(5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15, 10);
    const __m128i sh_c = _mm_setr_epi8(10, 5, 0, 11, 6, 1, 12, 7, 2, 13, 8, 3, 14, 9, 4, 15);
    __m128i a0 = _mm_shuffle_epi8(a.val, sh_a);
    __m128i b0 = _mm_shuffle_epi8(b.val, sh_b);
    __m128i c0 = _mm_shuffle_epi8(c.val, sh_c);

    const __m128i m0 = _mm_setr_epi8(0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0);
    const __m128i m1 = _mm_setr_epi8(0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0, -1, 0, 0);
    __m128i v0 = _mm_blendv_epi8(_mm_blendv_epi8(a0, b0, m1), c0, m0);
    __m128i v1 = _mm_blendv_epi8(_mm_blendv_epi8(b0, c0, m1), a0, m0);
    __m128i v2 = _mm_blendv_epi8(_mm_blendv_epi8(c0, a0, m1), b0, m0);
#elif CV_SSSE3
    const __m128i m0 = _mm_setr_epi8(0, 6, 11, 1, 7, 12, 2, 8, 13, 3, 9, 14, 4, 10, 15, 5);
    const __m128i m1 = _mm_setr_epi8(5, 11, 0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 4, 10);
    const __m128i m2 = _mm_setr_epi8(10, 0, 5, 11, 1, 6, 12, 2, 7, 13, 3, 8, 14, 4, 9, 15);

    __m128i t0 = _mm_alignr_epi8(b.val, _mm_slli_si128(a.val, 10), 5);
    t0 = _mm_alignr_epi8(c.val, t0, 5);
    __m128i v0 = _mm_shuffle_epi8(t0, m0);

    __m128i t1 = _mm_alignr_epi8(_mm_srli_si128(b.val, 5), _mm_slli_si128(a.val, 5), 6);
    t1 = _mm_alignr_epi8(_mm_srli_si128(c.val, 5), t1, 5);
    __m128i v1 = _mm_shuffle_epi8(t1, m1);

    __m128i t2 = _mm_alignr_epi8(_mm_srli_si128(c.val, 10), b.val, 11);
    t2 = _mm_alignr_epi8(t2, a.val, 11);
    __m128i v2 = _mm_shuffle_epi8(t2, m2);
#else
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
#endif

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 16), v1);
        _mm_stream_si128((__m128i*)(ptr + 32), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 16), v1);
        _mm_store_si128((__m128i*)(ptr + 32), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 16), v1);
        _mm_storeu_si128((__m128i*)(ptr + 32), v2);
    }
}

inline void v_store_interleave( uchar* ptr, const v_uint8x16& a, const v_uint8x16& b,
                                const v_uint8x16& c, const v_uint8x16& d,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
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
    __m128i v1 = _mm_unpackhi_epi8(u0, u2); // a4 b4 c4 d4 ...
    __m128i v2 = _mm_unpacklo_epi8(u1, u3); // a8 b8 c8 d8 ...
    __m128i v3 = _mm_unpackhi_epi8(u1, u3); // a12 b12 c12 d12 ...

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 16), v1);
        _mm_stream_si128((__m128i*)(ptr + 32), v2);
        _mm_stream_si128((__m128i*)(ptr + 48), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 16), v1);
        _mm_store_si128((__m128i*)(ptr + 32), v2);
        _mm_store_si128((__m128i*)(ptr + 48), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 16), v1);
        _mm_storeu_si128((__m128i*)(ptr + 32), v2);
        _mm_storeu_si128((__m128i*)(ptr + 48), v3);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi16(a.val, b.val);
    __m128i v1 = _mm_unpackhi_epi16(a.val, b.val);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 8), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 8), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 8), v1);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a,
                                const v_uint16x8& b, const v_uint16x8& c,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
#if CV_SSE4_1
    const __m128i sh_a = _mm_setr_epi8(0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5, 10, 11);
    const __m128i sh_b = _mm_setr_epi8(10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15, 4, 5);
    const __m128i sh_c = _mm_setr_epi8(4, 5, 10, 11, 0, 1, 6, 7, 12, 13, 2, 3, 8, 9, 14, 15);
    __m128i a0 = _mm_shuffle_epi8(a.val, sh_a);
    __m128i b0 = _mm_shuffle_epi8(b.val, sh_b);
    __m128i c0 = _mm_shuffle_epi8(c.val, sh_c);

    __m128i v0 = _mm_blend_epi16(_mm_blend_epi16(a0, b0, 0x92), c0, 0x24);
    __m128i v1 = _mm_blend_epi16(_mm_blend_epi16(c0, a0, 0x92), b0, 0x24);
    __m128i v2 = _mm_blend_epi16(_mm_blend_epi16(b0, c0, 0x92), a0, 0x24);
#else
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
#endif
    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 8), v1);
        _mm_stream_si128((__m128i*)(ptr + 16), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 8), v1);
        _mm_store_si128((__m128i*)(ptr + 16), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 8), v1);
        _mm_storeu_si128((__m128i*)(ptr + 16), v2);
    }
}

inline void v_store_interleave( ushort* ptr, const v_uint16x8& a, const v_uint16x8& b,
                                const v_uint16x8& c, const v_uint16x8& d,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
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
    __m128i v1 = _mm_unpackhi_epi16(u0, u2); // a2 b2 c2 d2 ...
    __m128i v2 = _mm_unpacklo_epi16(u1, u3); // a4 b4 c4 d4 ...
    __m128i v3 = _mm_unpackhi_epi16(u1, u3); // a6 b6 c6 d6 ...

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 8), v1);
        _mm_stream_si128((__m128i*)(ptr + 16), v2);
        _mm_stream_si128((__m128i*)(ptr + 24), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 8), v1);
        _mm_store_si128((__m128i*)(ptr + 16), v2);
        _mm_store_si128((__m128i*)(ptr + 24), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 8), v1);
        _mm_storeu_si128((__m128i*)(ptr + 16), v2);
        _mm_storeu_si128((__m128i*)(ptr + 24), v3);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi32(a.val, b.val);
    __m128i v1 = _mm_unpackhi_epi32(a.val, b.val);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 4), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 4), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 4), v1);
    }
}

inline void v_store_interleave( unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                                const v_uint32x4& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v_uint32x4 z = v_setzero_u32(), u0, u1, u2, u3;
    v_transpose4x4(a, b, c, z, u0, u1, u2, u3);

    __m128i v0 = _mm_or_si128(u0.val, _mm_slli_si128(u1.val, 12));
    __m128i v1 = _mm_or_si128(_mm_srli_si128(u1.val, 4), _mm_slli_si128(u2.val, 8));
    __m128i v2 = _mm_or_si128(_mm_srli_si128(u2.val, 8), _mm_slli_si128(u3.val, 4));

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 4), v1);
        _mm_stream_si128((__m128i*)(ptr + 8), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 4), v1);
        _mm_store_si128((__m128i*)(ptr + 8), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 4), v1);
        _mm_storeu_si128((__m128i*)(ptr + 8), v2);
    }
}

inline void v_store_interleave(unsigned* ptr, const v_uint32x4& a, const v_uint32x4& b,
                               const v_uint32x4& c, const v_uint32x4& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    v_uint32x4 v0, v1, v2, v3;
    v_transpose4x4(a, b, c, d, v0, v1, v2, v3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0.val);
        _mm_stream_si128((__m128i*)(ptr + 4), v1.val);
        _mm_stream_si128((__m128i*)(ptr + 8), v2.val);
        _mm_stream_si128((__m128i*)(ptr + 12), v3.val);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0.val);
        _mm_store_si128((__m128i*)(ptr + 4), v1.val);
        _mm_store_si128((__m128i*)(ptr + 8), v2.val);
        _mm_store_si128((__m128i*)(ptr + 12), v3.val);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0.val);
        _mm_storeu_si128((__m128i*)(ptr + 4), v1.val);
        _mm_storeu_si128((__m128i*)(ptr + 8), v2.val);
        _mm_storeu_si128((__m128i*)(ptr + 12), v3.val);
    }
}

// 2-channel, float only
inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128 v0 = _mm_unpacklo_ps(a.val, b.val); // a0 b0 a1 b1
    __m128 v1 = _mm_unpackhi_ps(a.val, b.val); // a2 b2 a3 b3

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_ps(ptr, v0);
        _mm_stream_ps(ptr + 4, v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_ps(ptr, v0);
        _mm_store_ps(ptr + 4, v1);
    }
    else
    {
        _mm_storeu_ps(ptr, v0);
        _mm_storeu_ps(ptr + 4, v1);
    }
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128 u0 = _mm_shuffle_ps(a.val, b.val, _MM_SHUFFLE(0, 0, 0, 0));
    __m128 u1 = _mm_shuffle_ps(c.val, a.val, _MM_SHUFFLE(1, 1, 0, 0));
    __m128 v0 = _mm_shuffle_ps(u0, u1, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u2 = _mm_shuffle_ps(b.val, c.val, _MM_SHUFFLE(1, 1, 1, 1));
    __m128 u3 = _mm_shuffle_ps(a.val, b.val, _MM_SHUFFLE(2, 2, 2, 2));
    __m128 v1 = _mm_shuffle_ps(u2, u3, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 u4 = _mm_shuffle_ps(c.val, a.val, _MM_SHUFFLE(3, 3, 2, 2));
    __m128 u5 = _mm_shuffle_ps(b.val, c.val, _MM_SHUFFLE(3, 3, 3, 3));
    __m128 v2 = _mm_shuffle_ps(u4, u5, _MM_SHUFFLE(2, 0, 2, 0));

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_ps(ptr, v0);
        _mm_stream_ps(ptr + 4, v1);
        _mm_stream_ps(ptr + 8, v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_ps(ptr, v0);
        _mm_store_ps(ptr + 4, v1);
        _mm_store_ps(ptr + 8, v2);
    }
    else
    {
        _mm_storeu_ps(ptr, v0);
        _mm_storeu_ps(ptr + 4, v1);
        _mm_storeu_ps(ptr + 8, v2);
    }
}

inline void v_store_interleave(float* ptr, const v_float32x4& a, const v_float32x4& b,
                               const v_float32x4& c, const v_float32x4& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128 u0 = _mm_unpacklo_ps(a.val, c.val);
    __m128 u1 = _mm_unpacklo_ps(b.val, d.val);
    __m128 u2 = _mm_unpackhi_ps(a.val, c.val);
    __m128 u3 = _mm_unpackhi_ps(b.val, d.val);
    __m128 v0 = _mm_unpacklo_ps(u0, u1);
    __m128 v2 = _mm_unpacklo_ps(u2, u3);
    __m128 v1 = _mm_unpackhi_ps(u0, u1);
    __m128 v3 = _mm_unpackhi_ps(u2, u3);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_ps(ptr, v0);
        _mm_stream_ps(ptr + 4, v1);
        _mm_stream_ps(ptr + 8, v2);
        _mm_stream_ps(ptr + 12, v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_ps(ptr, v0);
        _mm_store_ps(ptr + 4, v1);
        _mm_store_ps(ptr + 8, v2);
        _mm_store_ps(ptr + 12, v3);
    }
    else
    {
        _mm_storeu_ps(ptr, v0);
        _mm_storeu_ps(ptr + 4, v1);
        _mm_storeu_ps(ptr + 8, v2);
        _mm_storeu_ps(ptr + 12, v3);
    }
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi64(a.val, b.val);
    __m128i v1 = _mm_unpackhi_epi64(a.val, b.val);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 2), v1);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 2), v1);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 2), v1);
    }
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi64(a.val, b.val);
    __m128i v1 = _mm_unpacklo_epi64(c.val, _mm_unpackhi_epi64(a.val, a.val));
    __m128i v2 = _mm_unpackhi_epi64(b.val, c.val);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 2), v1);
        _mm_stream_si128((__m128i*)(ptr + 4), v2);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 2), v1);
        _mm_store_si128((__m128i*)(ptr + 4), v2);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 2), v1);
        _mm_storeu_si128((__m128i*)(ptr + 4), v2);
    }
}

inline void v_store_interleave(uint64 *ptr, const v_uint64x2& a, const v_uint64x2& b,
                               const v_uint64x2& c, const v_uint64x2& d,
                               hal::StoreMode mode = hal::STORE_UNALIGNED)
{
    __m128i v0 = _mm_unpacklo_epi64(a.val, b.val);
    __m128i v1 = _mm_unpacklo_epi64(c.val, d.val);
    __m128i v2 = _mm_unpackhi_epi64(a.val, b.val);
    __m128i v3 = _mm_unpackhi_epi64(c.val, d.val);

    if( mode == hal::STORE_ALIGNED_NOCACHE )
    {
        _mm_stream_si128((__m128i*)(ptr), v0);
        _mm_stream_si128((__m128i*)(ptr + 2), v1);
        _mm_stream_si128((__m128i*)(ptr + 4), v2);
        _mm_stream_si128((__m128i*)(ptr + 6), v3);
    }
    else if( mode == hal::STORE_ALIGNED )
    {
        _mm_store_si128((__m128i*)(ptr), v0);
        _mm_store_si128((__m128i*)(ptr + 2), v1);
        _mm_store_si128((__m128i*)(ptr + 4), v2);
        _mm_store_si128((__m128i*)(ptr + 6), v3);
    }
    else
    {
        _mm_storeu_si128((__m128i*)(ptr), v0);
        _mm_storeu_si128((__m128i*)(ptr + 2), v1);
        _mm_storeu_si128((__m128i*)(ptr + 4), v2);
        _mm_storeu_si128((__m128i*)(ptr + 6), v3);
    }
}

#define OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(_Tpvec0, _Tp0, suffix0, _Tpvec1, _Tp1, suffix1) \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0 ) \
{ \
    _Tpvec1 a1, b1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0 ) \
{ \
    _Tpvec1 a1, b1, c1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
} \
inline void v_load_deinterleave( const _Tp0* ptr, _Tpvec0& a0, _Tpvec0& b0, _Tpvec0& c0, _Tpvec0& d0 ) \
{ \
    _Tpvec1 a1, b1, c1, d1; \
    v_load_deinterleave((const _Tp1*)ptr, a1, b1, c1, d1); \
    a0 = v_reinterpret_as_##suffix0(a1); \
    b0 = v_reinterpret_as_##suffix0(b1); \
    c0 = v_reinterpret_as_##suffix0(c1); \
    d0 = v_reinterpret_as_##suffix0(d1); \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, mode);      \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, mode);  \
} \
inline void v_store_interleave( _Tp0* ptr, const _Tpvec0& a0, const _Tpvec0& b0, \
                                const _Tpvec0& c0, const _Tpvec0& d0, \
                                hal::StoreMode mode = hal::STORE_UNALIGNED ) \
{ \
    _Tpvec1 a1 = v_reinterpret_as_##suffix1(a0); \
    _Tpvec1 b1 = v_reinterpret_as_##suffix1(b0); \
    _Tpvec1 c1 = v_reinterpret_as_##suffix1(c0); \
    _Tpvec1 d1 = v_reinterpret_as_##suffix1(d0); \
    v_store_interleave((_Tp1*)ptr, a1, b1, c1, d1, mode); \
}

OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int8x16, schar, s8, v_uint8x16, uchar, u8)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int16x8, short, s16, v_uint16x8, ushort, u16)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int32x4, int, s32, v_uint32x4, unsigned, u32)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_int64x2, int64, s64, v_uint64x2, uint64, u64)
OPENCV_HAL_IMPL_SSE_LOADSTORE_INTERLEAVE(v_float64x2, double, f64, v_uint64x2, uint64, u64)

inline v_float32x4 v_cvt_f32(const v_int32x4& a)
{
    return v_float32x4(_mm_cvtepi32_ps(a.val));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a)
{
    return v_float32x4(_mm_cvtpd_ps(a.val));
}

inline v_float32x4 v_cvt_f32(const v_float64x2& a, const v_float64x2& b)
{
    return v_float32x4(_mm_movelh_ps(_mm_cvtpd_ps(a.val), _mm_cvtpd_ps(b.val)));
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
    return v_float64x2(_mm_cvtps_pd(_mm_movehl_ps(a.val, a.val)));
}

// from (Mysticial and wim) https://stackoverflow.com/q/41144668
inline v_float64x2 v_cvt_f64(const v_int64x2& v)
{
    // constants encoded as floating-point
    __m128i magic_i_hi32 = _mm_set1_epi64x(0x4530000080000000); // 2^84 + 2^63
    __m128i magic_i_all  = _mm_set1_epi64x(0x4530000080100000); // 2^84 + 2^63 + 2^52
    __m128d magic_d_all  = _mm_castsi128_pd(magic_i_all);
    // Blend the 32 lowest significant bits of v with magic_int_lo
#if CV_SSE4_1
    __m128i magic_i_lo   = _mm_set1_epi64x(0x4330000000000000); // 2^52
    __m128i v_lo         = _mm_blend_epi16(v.val, magic_i_lo, 0xcc);
#else
    __m128i magic_i_lo   = _mm_set1_epi32(0x43300000); // 2^52
    __m128i v_lo         = _mm_unpacklo_epi32(_mm_shuffle_epi32(v.val, _MM_SHUFFLE(0, 0, 2, 0)), magic_i_lo);
#endif
    // Extract the 32 most significant bits of v
    __m128i v_hi         = _mm_srli_epi64(v.val, 32);
    // Flip the msb of v_hi and blend with 0x45300000
            v_hi         = _mm_xor_si128(v_hi, magic_i_hi32);
    // Compute in double precision
    __m128d v_hi_dbl     = _mm_sub_pd(_mm_castsi128_pd(v_hi), magic_d_all);
    // (v_hi - magic_d_all) + v_lo  Do not assume associativity of floating point addition
    __m128d result       = _mm_add_pd(v_hi_dbl, _mm_castsi128_pd(v_lo));
    return v_float64x2(result);
}

////////////// Lookup table access ////////////////////

inline v_int8x16 v_lut(const schar* tab, const int* idx)
{
#if defined(_MSC_VER)
    return v_int8x16(_mm_setr_epi8(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]],
                                   tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]]));
#else
    return v_int8x16(_mm_setr_epi64(
                        _mm_setr_pi8(tab[idx[0]], tab[idx[1]], tab[idx[ 2]], tab[idx[ 3]], tab[idx[ 4]], tab[idx[ 5]], tab[idx[ 6]], tab[idx[ 7]]),
                        _mm_setr_pi8(tab[idx[8]], tab[idx[9]], tab[idx[10]], tab[idx[11]], tab[idx[12]], tab[idx[13]], tab[idx[14]], tab[idx[15]])
                    ));
#endif
}
inline v_int8x16 v_lut_pairs(const schar* tab, const int* idx)
{
#if defined(_MSC_VER)
    return v_int8x16(_mm_setr_epi16(*(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]), *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3]),
                                    *(const short*)(tab + idx[4]), *(const short*)(tab + idx[5]), *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7])));
#else
    return v_int8x16(_mm_setr_epi64(
                        _mm_setr_pi16(*(const short*)(tab + idx[0]), *(const short*)(tab + idx[1]), *(const short*)(tab + idx[2]), *(const short*)(tab + idx[3])),
                        _mm_setr_pi16(*(const short*)(tab + idx[4]), *(const short*)(tab + idx[5]), *(const short*)(tab + idx[6]), *(const short*)(tab + idx[7]))
                    ));
#endif
}
inline v_int8x16 v_lut_quads(const schar* tab, const int* idx)
{
#if defined(_MSC_VER)
    return v_int8x16(_mm_setr_epi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                                    *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
#else
    return v_int8x16(_mm_setr_epi64(
                        _mm_setr_pi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1])),
                        _mm_setr_pi32(*(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]))
                    ));
#endif
}
inline v_uint8x16 v_lut(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut((const schar *)tab, idx)); }
inline v_uint8x16 v_lut_pairs(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_pairs((const schar *)tab, idx)); }
inline v_uint8x16 v_lut_quads(const uchar* tab, const int* idx) { return v_reinterpret_as_u8(v_lut_quads((const schar *)tab, idx)); }

inline v_int16x8 v_lut(const short* tab, const int* idx)
{
#if defined(_MSC_VER)
    return v_int16x8(_mm_setr_epi16(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]],
                                    tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]]));
#else
    return v_int16x8(_mm_setr_epi64(
                        _mm_setr_pi16(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]),
                        _mm_setr_pi16(tab[idx[4]], tab[idx[5]], tab[idx[6]], tab[idx[7]])
                    ));
#endif
}
inline v_int16x8 v_lut_pairs(const short* tab, const int* idx)
{
#if defined(_MSC_VER)
    return v_int16x8(_mm_setr_epi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1]),
                                    *(const int*)(tab + idx[2]), *(const int*)(tab + idx[3])));
#else
    return v_int16x8(_mm_setr_epi64(
                        _mm_setr_pi32(*(const int*)(tab + idx[0]), *(const int*)(tab + idx[1])),
                        _mm_setr_pi32(*(const int*)(tab + idx[2]), *(const int*)(tab + idx[3]))
                    ));
#endif
}
inline v_int16x8 v_lut_quads(const short* tab, const int* idx)
{
    return v_int16x8(_mm_set_epi64x(*(const int64_t*)(tab + idx[1]), *(const int64_t*)(tab + idx[0])));
}
inline v_uint16x8 v_lut(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut((const short *)tab, idx)); }
inline v_uint16x8 v_lut_pairs(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_pairs((const short *)tab, idx)); }
inline v_uint16x8 v_lut_quads(const ushort* tab, const int* idx) { return v_reinterpret_as_u16(v_lut_quads((const short *)tab, idx)); }

inline v_int32x4 v_lut(const int* tab, const int* idx)
{
#if defined(_MSC_VER)
    return v_int32x4(_mm_setr_epi32(tab[idx[0]], tab[idx[1]],
                                    tab[idx[2]], tab[idx[3]]));
#else
    return v_int32x4(_mm_setr_epi64(
                        _mm_setr_pi32(tab[idx[0]], tab[idx[1]]),
                        _mm_setr_pi32(tab[idx[2]], tab[idx[3]])
                    ));
#endif
}
inline v_int32x4 v_lut_pairs(const int* tab, const int* idx)
{
    return v_int32x4(_mm_set_epi64x(*(const int64_t*)(tab + idx[1]), *(const int64_t*)(tab + idx[0])));
}
inline v_int32x4 v_lut_quads(const int* tab, const int* idx)
{
    return v_int32x4(_mm_loadu_si128((const __m128i*)(tab + idx[0])));
}
inline v_uint32x4 v_lut(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut((const int *)tab, idx)); }
inline v_uint32x4 v_lut_pairs(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_pairs((const int *)tab, idx)); }
inline v_uint32x4 v_lut_quads(const unsigned* tab, const int* idx) { return v_reinterpret_as_u32(v_lut_quads((const int *)tab, idx)); }

inline v_int64x2 v_lut(const int64_t* tab, const int* idx)
{
    return v_int64x2(_mm_set_epi64x(tab[idx[1]], tab[idx[0]]));
}
inline v_int64x2 v_lut_pairs(const int64_t* tab, const int* idx)
{
    return v_int64x2(_mm_loadu_si128((const __m128i*)(tab + idx[0])));
}
inline v_uint64x2 v_lut(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut((const int64_t *)tab, idx)); }
inline v_uint64x2 v_lut_pairs(const uint64_t* tab, const int* idx) { return v_reinterpret_as_u64(v_lut_pairs((const int64_t *)tab, idx)); }

inline v_float32x4 v_lut(const float* tab, const int* idx)
{
    return v_float32x4(_mm_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}
inline v_float32x4 v_lut_pairs(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_pairs((const int *)tab, idx)); }
inline v_float32x4 v_lut_quads(const float* tab, const int* idx) { return v_reinterpret_as_f32(v_lut_quads((const int *)tab, idx)); }

inline v_float64x2 v_lut(const double* tab, const int* idx)
{
    return v_float64x2(_mm_setr_pd(tab[idx[0]], tab[idx[1]]));
}
inline v_float64x2 v_lut_pairs(const double* tab, const int* idx) { return v_float64x2(_mm_castsi128_pd(_mm_loadu_si128((const __m128i*)(tab + idx[0])))); }

inline v_int32x4 v_lut(const int* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);
    return v_int32x4(_mm_setr_epi32(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}

inline v_uint32x4 v_lut(const unsigned* tab, const v_int32x4& idxvec)
{
    return v_reinterpret_as_u32(v_lut((const int *)tab, idxvec));
}

inline v_float32x4 v_lut(const float* tab, const v_int32x4& idxvec)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);
    return v_float32x4(_mm_setr_ps(tab[idx[0]], tab[idx[1]], tab[idx[2]], tab[idx[3]]));
}

inline v_float64x2 v_lut(const double* tab, const v_int32x4& idxvec)
{
    int idx[2];
    v_store_low(idx, idxvec);
    return v_float64x2(_mm_setr_pd(tab[idx[0]], tab[idx[1]]));
}

// loads pairs from the table and deinterleaves them, e.g. returns:
//   x = (tab[idxvec[0], tab[idxvec[1]], tab[idxvec[2]], tab[idxvec[3]]),
//   y = (tab[idxvec[0]+1], tab[idxvec[1]+1], tab[idxvec[2]+1], tab[idxvec[3]+1])
// note that the indices are float's indices, not the float-pair indices.
// in theory, this function can be used to implement bilinear interpolation,
// when idxvec are the offsets within the image.
inline void v_lut_deinterleave(const float* tab, const v_int32x4& idxvec, v_float32x4& x, v_float32x4& y)
{
    int CV_DECL_ALIGNED(32) idx[4];
    v_store_aligned(idx, idxvec);
    __m128 z = _mm_setzero_ps();
    __m128 xy01 = _mm_loadl_pi(z, (__m64*)(tab + idx[0]));
    __m128 xy23 = _mm_loadl_pi(z, (__m64*)(tab + idx[2]));
    xy01 = _mm_loadh_pi(xy01, (__m64*)(tab + idx[1]));
    xy23 = _mm_loadh_pi(xy23, (__m64*)(tab + idx[3]));
    __m128 xxyy02 = _mm_unpacklo_ps(xy01, xy23);
    __m128 xxyy13 = _mm_unpackhi_ps(xy01, xy23);
    x = v_float32x4(_mm_unpacklo_ps(xxyy02, xxyy13));
    y = v_float32x4(_mm_unpackhi_ps(xxyy02, xxyy13));
}

inline void v_lut_deinterleave(const double* tab, const v_int32x4& idxvec, v_float64x2& x, v_float64x2& y)
{
    int idx[2];
    v_store_low(idx, idxvec);
    __m128d xy0 = _mm_loadu_pd(tab + idx[0]);
    __m128d xy1 = _mm_loadu_pd(tab + idx[1]);
    x = v_float64x2(_mm_unpacklo_pd(xy0, xy1));
    y = v_float64x2(_mm_unpackhi_pd(xy0, xy1));
}

inline v_int8x16 v_interleave_pairs(const v_int8x16& vec)
{
#if CV_SSSE3
    return v_int8x16(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0d0e0c0b090a08, 0x0705060403010200)));
#else
    __m128i a = _mm_shufflelo_epi16(vec.val, _MM_SHUFFLE(3, 1, 2, 0));
    a = _mm_shufflehi_epi16(a, _MM_SHUFFLE(3, 1, 2, 0));
    a = _mm_shuffle_epi32(a, _MM_SHUFFLE(3, 1, 2, 0));
    return v_int8x16(_mm_unpacklo_epi8(a, _mm_unpackhi_epi64(a, a)));
#endif
}
inline v_uint8x16 v_interleave_pairs(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_pairs(v_reinterpret_as_s8(vec))); }
inline v_int8x16 v_interleave_quads(const v_int8x16& vec)
{
#if CV_SSSE3
    return v_int8x16(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0b0e0a0d090c08, 0x0703060205010400)));
#else
    __m128i a = _mm_shuffle_epi32(vec.val, _MM_SHUFFLE(3, 1, 2, 0));
    return v_int8x16(_mm_unpacklo_epi8(a, _mm_unpackhi_epi64(a, a)));
#endif
}
inline v_uint8x16 v_interleave_quads(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_interleave_quads(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_interleave_pairs(const v_int16x8& vec)
{
#if CV_SSSE3
    return v_int16x8(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0e0b0a0d0c0908, 0x0706030205040100)));
#else
    __m128i a = _mm_shufflelo_epi16(vec.val, _MM_SHUFFLE(3, 1, 2, 0));
    return v_int16x8(_mm_shufflehi_epi16(a, _MM_SHUFFLE(3, 1, 2, 0)));
#endif
}
inline v_uint16x8 v_interleave_pairs(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_pairs(v_reinterpret_as_s16(vec))); }
inline v_int16x8 v_interleave_quads(const v_int16x8& vec)
{
#if CV_SSSE3
    return v_int16x8(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0x0f0e07060d0c0504, 0x0b0a030209080100)));
#else
    return v_int16x8(_mm_unpacklo_epi16(vec.val, _mm_unpackhi_epi64(vec.val, vec.val)));
#endif
}
inline v_uint16x8 v_interleave_quads(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_interleave_quads(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_interleave_pairs(const v_int32x4& vec)
{
    return v_int32x4(_mm_shuffle_epi32(vec.val, _MM_SHUFFLE(3, 1, 2, 0)));
}
inline v_uint32x4 v_interleave_pairs(const v_uint32x4& vec) { return v_reinterpret_as_u32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }
inline v_float32x4 v_interleave_pairs(const v_float32x4& vec) { return v_reinterpret_as_f32(v_interleave_pairs(v_reinterpret_as_s32(vec))); }

inline v_int8x16 v_pack_triplets(const v_int8x16& vec)
{
#if CV_SSSE3
    return v_int8x16(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0xffffff0f0e0d0c0a, 0x0908060504020100)));
#else
    __m128i mask = _mm_set1_epi64x(0x00000000FFFFFFFF);
    __m128i a = _mm_srli_si128(_mm_or_si128(_mm_andnot_si128(mask, vec.val), _mm_and_si128(mask, _mm_sll_epi32(vec.val, _mm_set_epi64x(0, 8)))), 1);
    return v_int8x16(_mm_srli_si128(_mm_shufflelo_epi16(a, _MM_SHUFFLE(2, 1, 0, 3)), 2));
#endif
}
inline v_uint8x16 v_pack_triplets(const v_uint8x16& vec) { return v_reinterpret_as_u8(v_pack_triplets(v_reinterpret_as_s8(vec))); }

inline v_int16x8 v_pack_triplets(const v_int16x8& vec)
{
#if CV_SSSE3
    return v_int16x8(_mm_shuffle_epi8(vec.val, _mm_set_epi64x(0xffff0f0e0d0c0b0a, 0x0908050403020100)));
#else
    return v_int16x8(_mm_srli_si128(_mm_shufflelo_epi16(vec.val, _MM_SHUFFLE(2, 1, 0, 3)), 2));
#endif
}
inline v_uint16x8 v_pack_triplets(const v_uint16x8& vec) { return v_reinterpret_as_u16(v_pack_triplets(v_reinterpret_as_s16(vec))); }

inline v_int32x4 v_pack_triplets(const v_int32x4& vec) { return vec; }
inline v_uint32x4 v_pack_triplets(const v_uint32x4& vec) { return vec; }
inline v_float32x4 v_pack_triplets(const v_float32x4& vec) { return vec; }

template<int i>
inline uchar v_extract_n(const v_uint8x16& v)
{
#if CV_SSE4_1
    return (uchar)_mm_extract_epi8(v.val, i);
#else
    return v_rotate_right<i>(v).get0();
#endif
}

template<int i>
inline schar v_extract_n(const v_int8x16& v)
{
    return (schar)v_extract_n<i>(v_reinterpret_as_u8(v));
}

template<int i>
inline ushort v_extract_n(const v_uint16x8& v)
{
    return (ushort)_mm_extract_epi16(v.val, i);
}

template<int i>
inline short v_extract_n(const v_int16x8& v)
{
    return (short)v_extract_n<i>(v_reinterpret_as_u16(v));
}

template<int i>
inline uint v_extract_n(const v_uint32x4& v)
{
#if CV_SSE4_1
    return (uint)_mm_extract_epi32(v.val, i);
#else
    return v_rotate_right<i>(v).get0();
#endif
}

template<int i>
inline int v_extract_n(const v_int32x4& v)
{
    return (int)v_extract_n<i>(v_reinterpret_as_u32(v));
}

template<int i>
inline uint64 v_extract_n(const v_uint64x2& v)
{
#ifdef CV__SIMD_NATIVE_mm_extract_epi64
    return (uint64)_v128_extract_epi64<i>(v.val);
#else
    return v_rotate_right<i>(v).get0();
#endif
}

template<int i>
inline int64 v_extract_n(const v_int64x2& v)
{
    return (int64)v_extract_n<i>(v_reinterpret_as_u64(v));
}

template<int i>
inline float v_extract_n(const v_float32x4& v)
{
    union { uint iv; float fv; } d;
    d.iv = v_extract_n<i>(v_reinterpret_as_u32(v));
    return d.fv;
}

template<int i>
inline double v_extract_n(const v_float64x2& v)
{
    union { uint64 iv; double dv; } d;
    d.iv = v_extract_n<i>(v_reinterpret_as_u64(v));
    return d.dv;
}

template<int i>
inline v_int32x4 v_broadcast_element(const v_int32x4& v)
{
    return v_int32x4(_mm_shuffle_epi32(v.val, _MM_SHUFFLE(i,i,i,i)));
}

template<int i>
inline v_uint32x4 v_broadcast_element(const v_uint32x4& v)
{
    return v_uint32x4(_mm_shuffle_epi32(v.val, _MM_SHUFFLE(i,i,i,i)));
}

template<int i>
inline v_float32x4 v_broadcast_element(const v_float32x4& v)
{
    return v_float32x4(_mm_shuffle_ps(v.val, v.val, _MM_SHUFFLE((char)i,(char)i,(char)i,(char)i)));
}

////////////// FP16 support ///////////////////////////

inline v_float32x4 v_load_expand(const hfloat* ptr)
{
#if CV_FP16
    return v_float32x4(_mm_cvtph_ps(_mm_loadu_si128((const __m128i*)ptr)));
#else
    const __m128i z = _mm_setzero_si128(), delta = _mm_set1_epi32(0x38000000);
    const __m128i signmask = _mm_set1_epi32(0x80000000), maxexp = _mm_set1_epi32(0x7c000000);
    const __m128 deltaf = _mm_castsi128_ps(_mm_set1_epi32(0x38800000));
    __m128i bits = _mm_unpacklo_epi16(z, _mm_loadl_epi64((const __m128i*)ptr)); // h << 16
    __m128i e = _mm_and_si128(bits, maxexp), sign = _mm_and_si128(bits, signmask);
    __m128i t = _mm_add_epi32(_mm_srli_epi32(_mm_xor_si128(bits, sign), 3), delta); // ((h & 0x7fff) << 13) + delta
    __m128i zt = _mm_castps_si128(_mm_sub_ps(_mm_castsi128_ps(_mm_add_epi32(t, _mm_set1_epi32(1 << 23))), deltaf));

    t = _mm_add_epi32(t, _mm_and_si128(delta, _mm_cmpeq_epi32(maxexp, e)));
    __m128i zmask = _mm_cmpeq_epi32(e, z);
    __m128i ft = v_select_si128(zmask, zt, t);
    return v_float32x4(_mm_castsi128_ps(_mm_or_si128(ft, sign)));
#endif
}

inline void v_pack_store(hfloat* ptr, const v_float32x4& v)
{
#if CV_FP16
    __m128i fp16_value = _mm_cvtps_ph(v.val, 0);
    _mm_storel_epi64((__m128i*)ptr, fp16_value);
#else
    const __m128i signmask = _mm_set1_epi32(0x80000000);
    const __m128i rval = _mm_set1_epi32(0x3f000000);

    __m128i t = _mm_castps_si128(v.val);
    __m128i sign = _mm_srai_epi32(_mm_and_si128(t, signmask), 16);
    t = _mm_andnot_si128(signmask, t);

    __m128i finitemask = _mm_cmpgt_epi32(_mm_set1_epi32(0x47800000), t);
    __m128i isnan = _mm_cmpgt_epi32(t, _mm_set1_epi32(0x7f800000));
    __m128i naninf = v_select_si128(isnan, _mm_set1_epi32(0x7e00), _mm_set1_epi32(0x7c00));
    __m128i tinymask = _mm_cmpgt_epi32(_mm_set1_epi32(0x38800000), t);
    __m128i tt = _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(t), _mm_castsi128_ps(rval)));
    tt = _mm_sub_epi32(tt, rval);
    __m128i odd = _mm_and_si128(_mm_srli_epi32(t, 13), _mm_set1_epi32(1));
    __m128i nt = _mm_add_epi32(t, _mm_set1_epi32(0xc8000fff));
    nt = _mm_srli_epi32(_mm_add_epi32(nt, odd), 13);
    t = v_select_si128(tinymask, tt, nt);
    t = v_select_si128(finitemask, t, naninf);
    t = _mm_or_si128(t, sign);
    t = _mm_packs_epi32(t, t);
    _mm_storel_epi64((__m128i*)ptr, t);
#endif
}

#include "intrin_math.hpp"
OPENCV_HAL_MATH_IMPL_32F(v, 32x4)
OPENCV_HAL_MATH_IMPL_64F(v, 64x2)

inline void v_cleanup() {}

CV_CPU_OPTIMIZATION_HAL_NAMESPACE_END

//! @endcond

}

#endif
