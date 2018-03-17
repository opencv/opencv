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

#ifndef __OPENCV_ARITHM_SIMD_HPP__
#define __OPENCV_ARITHM_SIMD_HPP__

namespace cv {

struct NOP {};

#if CV_SSE2 || CV_NEON
#define IF_SIMD(op) op
#else
#define IF_SIMD(op) NOP
#endif


#if CV_SSE2 || CV_NEON

#define FUNCTOR_TEMPLATE(name)          \
    template<typename T> struct name {}

FUNCTOR_TEMPLATE(VLoadStore128);
#if CV_SSE2
FUNCTOR_TEMPLATE(VLoadStore64);
FUNCTOR_TEMPLATE(VLoadStore128Aligned);
#if CV_AVX2
FUNCTOR_TEMPLATE(VLoadStore256);
FUNCTOR_TEMPLATE(VLoadStore256Aligned);
#endif
#endif

#endif

#if CV_AVX2

#define FUNCTOR_LOADSTORE_CAST(name, template_arg, register_type, load_body, store_body)         \
    template <>                                                                                  \
    struct name<template_arg>{                                                                   \
        typedef register_type reg_type;                                                          \
        static reg_type load(const template_arg * p) { return load_body ((const reg_type *)p); } \
        static void store(template_arg * p, reg_type v) { store_body ((reg_type *)p, v); }       \
    }

#define FUNCTOR_LOADSTORE(name, template_arg, register_type, load_body, store_body) \
    template <>                                                                     \
    struct name<template_arg>{                                                      \
        typedef register_type reg_type;                                             \
        static reg_type load(const template_arg * p) { return load_body (p); }      \
        static void store(template_arg * p, reg_type v) { store_body (p, v); }      \
    }

#define FUNCTOR_CLOSURE_2arg(name, template_arg, body)                         \
    template<>                                                                 \
    struct name<template_arg>                                                  \
    {                                                                          \
        VLoadStore256<template_arg>::reg_type operator()(                      \
                        const VLoadStore256<template_arg>::reg_type & a,       \
                        const VLoadStore256<template_arg>::reg_type & b) const \
        {                                                                      \
            body;                                                              \
        }                                                                      \
    }

#define FUNCTOR_CLOSURE_1arg(name, template_arg, body)                         \
    template<>                                                                 \
    struct name<template_arg>                                                  \
    {                                                                          \
        VLoadStore256<template_arg>::reg_type operator()(                      \
                        const VLoadStore256<template_arg>::reg_type & a,       \
                        const VLoadStore256<template_arg>::reg_type &  ) const \
        {                                                                      \
            body;                                                              \
        }                                                                      \
    }

FUNCTOR_LOADSTORE_CAST(VLoadStore256,  uchar, __m256i, _mm256_loadu_si256, _mm256_storeu_si256);
FUNCTOR_LOADSTORE_CAST(VLoadStore256,  schar, __m256i, _mm256_loadu_si256, _mm256_storeu_si256);
FUNCTOR_LOADSTORE_CAST(VLoadStore256, ushort, __m256i, _mm256_loadu_si256, _mm256_storeu_si256);
FUNCTOR_LOADSTORE_CAST(VLoadStore256,  short, __m256i, _mm256_loadu_si256, _mm256_storeu_si256);
FUNCTOR_LOADSTORE_CAST(VLoadStore256,    int, __m256i, _mm256_loadu_si256, _mm256_storeu_si256);
FUNCTOR_LOADSTORE(     VLoadStore256,  float, __m256 , _mm256_loadu_ps   , _mm256_storeu_ps   );
FUNCTOR_LOADSTORE(     VLoadStore256, double, __m256d, _mm256_loadu_pd   , _mm256_storeu_pd   );

FUNCTOR_LOADSTORE_CAST(VLoadStore256Aligned,    int, __m256i, _mm256_load_si256, _mm256_store_si256);
FUNCTOR_LOADSTORE(     VLoadStore256Aligned,  float, __m256 , _mm256_load_ps   , _mm256_store_ps   );
FUNCTOR_LOADSTORE(     VLoadStore256Aligned, double, __m256d, _mm256_load_pd   , _mm256_store_pd   );

FUNCTOR_TEMPLATE(VAdd);
FUNCTOR_CLOSURE_2arg(VAdd,  uchar, return _mm256_adds_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  schar, return _mm256_adds_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, ushort, return _mm256_adds_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  short, return _mm256_adds_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,    int, return _mm256_add_epi32 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  float, return _mm256_add_ps    (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, double, return _mm256_add_pd    (a, b));

FUNCTOR_TEMPLATE(VSub);
FUNCTOR_CLOSURE_2arg(VSub,  uchar, return _mm256_subs_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  schar, return _mm256_subs_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub, ushort, return _mm256_subs_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,  short, return _mm256_subs_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,    int, return _mm256_sub_epi32 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  float, return _mm256_sub_ps    (a, b));
FUNCTOR_CLOSURE_2arg(VSub, double, return _mm256_sub_pd    (a, b));

FUNCTOR_TEMPLATE(VMin);
FUNCTOR_CLOSURE_2arg(VMin,  uchar, return _mm256_min_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VMin,  schar, return _mm256_min_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VMin, ushort, return _mm256_min_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,  short, return _mm256_min_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,    int, return _mm256_min_epi32(a, b));
FUNCTOR_CLOSURE_2arg(VMin,  float, return _mm256_min_ps   (a, b));
FUNCTOR_CLOSURE_2arg(VMin, double, return _mm256_min_pd   (a, b));

FUNCTOR_TEMPLATE(VMax);
FUNCTOR_CLOSURE_2arg(VMax,  uchar, return _mm256_max_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VMax,  schar, return _mm256_max_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VMax, ushort, return _mm256_max_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,  short, return _mm256_max_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,    int, return _mm256_max_epi32(a, b));
FUNCTOR_CLOSURE_2arg(VMax,  float, return _mm256_max_ps   (a, b));
FUNCTOR_CLOSURE_2arg(VMax, double, return _mm256_max_pd   (a, b));


static unsigned int CV_DECL_ALIGNED(32) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff,
                                                           0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static unsigned int CV_DECL_ALIGNED(32) v64f_absmask[] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff,
                                                           0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff };

FUNCTOR_TEMPLATE(VAbsDiff);
FUNCTOR_CLOSURE_2arg(VAbsDiff,  uchar,
        return _mm256_add_epi8(_mm256_subs_epu8(a, b), _mm256_subs_epu8(b, a));
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  schar,
        __m256i d = _mm256_subs_epi8(a, b);
        __m256i m = _mm256_cmpgt_epi8(b, a);
        return _mm256_subs_epi8(_mm256_xor_si256(d, m), m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff, ushort,
        return _mm256_add_epi16(_mm256_subs_epu16(a, b), _mm256_subs_epu16(b, a));
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  short,
        __m256i M = _mm256_max_epi16(a, b);
        __m256i m = _mm256_min_epi16(a, b);
        return _mm256_subs_epi16(M, m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,    int,
        __m256i d = _mm256_sub_epi32(a, b);
        __m256i m = _mm256_cmpgt_epi32(b, a);
        return _mm256_sub_epi32(_mm256_xor_si256(d, m), m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  float,
        return _mm256_and_ps(_mm256_sub_ps(a, b), *(const __m256*)v32f_absmask);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff, double,
        return _mm256_and_pd(_mm256_sub_pd(a, b), *(const __m256d*)v64f_absmask);
    );

FUNCTOR_TEMPLATE(VAnd);
FUNCTOR_CLOSURE_2arg(VAnd, uchar, return _mm256_and_si256(a, b));
FUNCTOR_TEMPLATE(VOr);
FUNCTOR_CLOSURE_2arg(VOr , uchar, return _mm256_or_si256 (a, b));
FUNCTOR_TEMPLATE(VXor);
FUNCTOR_CLOSURE_2arg(VXor, uchar, return _mm256_xor_si256(a, b));
FUNCTOR_TEMPLATE(VNot);
FUNCTOR_CLOSURE_1arg(VNot, uchar, return _mm256_xor_si256(_mm256_set1_epi32(-1), a));

#elif CV_SSE2

#define FUNCTOR_LOADSTORE_CAST(name, template_arg, register_type, load_body, store_body)\
    template <>                                                                                  \
    struct name<template_arg>{                                                                   \
        typedef register_type reg_type;                                                          \
        static reg_type load(const template_arg * p) { return load_body ((const reg_type *)p); } \
        static void store(template_arg * p, reg_type v) { store_body ((reg_type *)p, v); }       \
    }

#define FUNCTOR_LOADSTORE(name, template_arg, register_type, load_body, store_body)\
    template <>                                                                \
    struct name<template_arg>{                                                 \
        typedef register_type reg_type;                                        \
        static reg_type load(const template_arg * p) { return load_body (p); } \
        static void store(template_arg * p, reg_type v) { store_body (p, v); } \
    }

#define FUNCTOR_CLOSURE_2arg(name, template_arg, body)\
    template<>                                                                 \
    struct name<template_arg>                                                  \
    {                                                                          \
        VLoadStore128<template_arg>::reg_type operator()(                      \
                        const VLoadStore128<template_arg>::reg_type & a,       \
                        const VLoadStore128<template_arg>::reg_type & b) const \
        {                                                                      \
            body;                                                              \
        }                                                                      \
    }

#define FUNCTOR_CLOSURE_1arg(name, template_arg, body)\
    template<>                                                                 \
    struct name<template_arg>                                                  \
    {                                                                          \
        VLoadStore128<template_arg>::reg_type operator()(                      \
                        const VLoadStore128<template_arg>::reg_type & a,       \
                        const VLoadStore128<template_arg>::reg_type &  ) const \
        {                                                                      \
            body;                                                              \
        }                                                                      \
    }

FUNCTOR_LOADSTORE_CAST(VLoadStore128,  uchar, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128,  schar, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128, ushort, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128,  short, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE_CAST(VLoadStore128,    int, __m128i, _mm_loadu_si128, _mm_storeu_si128);
FUNCTOR_LOADSTORE(     VLoadStore128,  float, __m128 , _mm_loadu_ps   , _mm_storeu_ps   );
FUNCTOR_LOADSTORE(     VLoadStore128, double, __m128d, _mm_loadu_pd   , _mm_storeu_pd   );

FUNCTOR_LOADSTORE_CAST(VLoadStore64,  uchar, __m128i, _mm_loadl_epi64, _mm_storel_epi64);
FUNCTOR_LOADSTORE_CAST(VLoadStore64,  schar, __m128i, _mm_loadl_epi64, _mm_storel_epi64);
FUNCTOR_LOADSTORE_CAST(VLoadStore64, ushort, __m128i, _mm_loadl_epi64, _mm_storel_epi64);
FUNCTOR_LOADSTORE_CAST(VLoadStore64,  short, __m128i, _mm_loadl_epi64, _mm_storel_epi64);

FUNCTOR_LOADSTORE_CAST(VLoadStore128Aligned,    int, __m128i, _mm_load_si128, _mm_store_si128);
FUNCTOR_LOADSTORE(     VLoadStore128Aligned,  float, __m128 , _mm_load_ps   , _mm_store_ps   );
FUNCTOR_LOADSTORE(     VLoadStore128Aligned, double, __m128d, _mm_load_pd   , _mm_store_pd   );

FUNCTOR_TEMPLATE(VAdd);
FUNCTOR_CLOSURE_2arg(VAdd,  uchar, return _mm_adds_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  schar, return _mm_adds_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, ushort, return _mm_adds_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  short, return _mm_adds_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,    int, return _mm_add_epi32 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  float, return _mm_add_ps    (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, double, return _mm_add_pd    (a, b));

FUNCTOR_TEMPLATE(VSub);
FUNCTOR_CLOSURE_2arg(VSub,  uchar, return _mm_subs_epu8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  schar, return _mm_subs_epi8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub, ushort, return _mm_subs_epu16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,  short, return _mm_subs_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,    int, return _mm_sub_epi32 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  float, return _mm_sub_ps    (a, b));
FUNCTOR_CLOSURE_2arg(VSub, double, return _mm_sub_pd    (a, b));

FUNCTOR_TEMPLATE(VMin);
FUNCTOR_CLOSURE_2arg(VMin, uchar, return _mm_min_epu8(a, b));
FUNCTOR_CLOSURE_2arg(VMin, schar,
        __m128i m = _mm_cmpgt_epi8(a, b);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMin, ushort, return _mm_subs_epu16(a, _mm_subs_epu16(a, b)));
FUNCTOR_CLOSURE_2arg(VMin,  short, return _mm_min_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,    int,
        __m128i m = _mm_cmpgt_epi32(a, b);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMin,  float, return _mm_min_ps(a, b));
FUNCTOR_CLOSURE_2arg(VMin, double, return _mm_min_pd(a, b));

FUNCTOR_TEMPLATE(VMax);
FUNCTOR_CLOSURE_2arg(VMax, uchar, return _mm_max_epu8(a, b));
FUNCTOR_CLOSURE_2arg(VMax, schar,
        __m128i m = _mm_cmpgt_epi8(b, a);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMax, ushort, return _mm_adds_epu16(_mm_subs_epu16(a, b), b));
FUNCTOR_CLOSURE_2arg(VMax,  short, return _mm_max_epi16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,    int,
        __m128i m = _mm_cmpgt_epi32(b, a);
        return _mm_xor_si128(a, _mm_and_si128(_mm_xor_si128(a, b), m));
    );
FUNCTOR_CLOSURE_2arg(VMax,  float, return _mm_max_ps(a, b));
FUNCTOR_CLOSURE_2arg(VMax, double, return _mm_max_pd(a, b));


static unsigned int CV_DECL_ALIGNED(16) v32f_absmask[] = { 0x7fffffff, 0x7fffffff, 0x7fffffff, 0x7fffffff };
static unsigned int CV_DECL_ALIGNED(16) v64f_absmask[] = { 0xffffffff, 0x7fffffff, 0xffffffff, 0x7fffffff };

FUNCTOR_TEMPLATE(VAbsDiff);
FUNCTOR_CLOSURE_2arg(VAbsDiff,  uchar,
        return _mm_add_epi8(_mm_subs_epu8(a, b), _mm_subs_epu8(b, a));
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  schar,
        __m128i d = _mm_subs_epi8(a, b);
        __m128i m = _mm_cmpgt_epi8(b, a);
        return _mm_subs_epi8(_mm_xor_si128(d, m), m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff, ushort,
        return _mm_add_epi16(_mm_subs_epu16(a, b), _mm_subs_epu16(b, a));
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  short,
        __m128i M = _mm_max_epi16(a, b);
        __m128i m = _mm_min_epi16(a, b);
        return _mm_subs_epi16(M, m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,    int,
        __m128i d = _mm_sub_epi32(a, b);
        __m128i m = _mm_cmpgt_epi32(b, a);
        return _mm_sub_epi32(_mm_xor_si128(d, m), m);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff,  float,
        return _mm_and_ps(_mm_sub_ps(a,b), *(const __m128*)v32f_absmask);
    );
FUNCTOR_CLOSURE_2arg(VAbsDiff, double,
        return _mm_and_pd(_mm_sub_pd(a,b), *(const __m128d*)v64f_absmask);
    );

FUNCTOR_TEMPLATE(VAnd);
FUNCTOR_CLOSURE_2arg(VAnd, uchar, return _mm_and_si128(a, b));
FUNCTOR_TEMPLATE(VOr);
FUNCTOR_CLOSURE_2arg(VOr , uchar, return _mm_or_si128 (a, b));
FUNCTOR_TEMPLATE(VXor);
FUNCTOR_CLOSURE_2arg(VXor, uchar, return _mm_xor_si128(a, b));
FUNCTOR_TEMPLATE(VNot);
FUNCTOR_CLOSURE_1arg(VNot, uchar, return _mm_xor_si128(_mm_set1_epi32(-1), a));
#endif

#if CV_NEON

#define FUNCTOR_LOADSTORE(name, template_arg, register_type, load_body, store_body)\
    template <>                                                                \
    struct name<template_arg>{                                                 \
        typedef register_type reg_type;                                        \
        static reg_type load(const template_arg * p) { return load_body (p);}; \
        static void store(template_arg * p, reg_type v) { store_body (p, v);}; \
    }

#define FUNCTOR_CLOSURE_2arg(name, template_arg, body)\
    template<>                                                         \
    struct name<template_arg>                                          \
    {                                                                  \
        VLoadStore128<template_arg>::reg_type operator()(              \
                        VLoadStore128<template_arg>::reg_type a,       \
                        VLoadStore128<template_arg>::reg_type b) const \
        {                                                              \
            return body;                                               \
        };                                                             \
    }

#define FUNCTOR_CLOSURE_1arg(name, template_arg, body)\
    template<>                                                         \
    struct name<template_arg>                                          \
    {                                                                  \
        VLoadStore128<template_arg>::reg_type operator()(              \
                        VLoadStore128<template_arg>::reg_type a,       \
                        VLoadStore128<template_arg>::reg_type  ) const \
        {                                                              \
            return body;                                               \
        };                                                             \
    }

FUNCTOR_LOADSTORE(VLoadStore128,  uchar,  uint8x16_t, vld1q_u8 , vst1q_u8 );
FUNCTOR_LOADSTORE(VLoadStore128,  schar,   int8x16_t, vld1q_s8 , vst1q_s8 );
FUNCTOR_LOADSTORE(VLoadStore128, ushort,  uint16x8_t, vld1q_u16, vst1q_u16);
FUNCTOR_LOADSTORE(VLoadStore128,  short,   int16x8_t, vld1q_s16, vst1q_s16);
FUNCTOR_LOADSTORE(VLoadStore128,    int,   int32x4_t, vld1q_s32, vst1q_s32);
FUNCTOR_LOADSTORE(VLoadStore128,  float, float32x4_t, vld1q_f32, vst1q_f32);

FUNCTOR_TEMPLATE(VAdd);
FUNCTOR_CLOSURE_2arg(VAdd,  uchar, vqaddq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  schar, vqaddq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd, ushort, vqaddq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  short, vqaddq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VAdd,    int, vaddq_s32 (a, b));
FUNCTOR_CLOSURE_2arg(VAdd,  float, vaddq_f32 (a, b));

FUNCTOR_TEMPLATE(VSub);
FUNCTOR_CLOSURE_2arg(VSub,  uchar, vqsubq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  schar, vqsubq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VSub, ushort, vqsubq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,  short, vqsubq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VSub,    int, vsubq_s32 (a, b));
FUNCTOR_CLOSURE_2arg(VSub,  float, vsubq_f32 (a, b));

FUNCTOR_TEMPLATE(VMin);
FUNCTOR_CLOSURE_2arg(VMin,  uchar, vminq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VMin,  schar, vminq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VMin, ushort, vminq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,  short, vminq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VMin,    int, vminq_s32(a, b));
FUNCTOR_CLOSURE_2arg(VMin,  float, vminq_f32(a, b));

FUNCTOR_TEMPLATE(VMax);
FUNCTOR_CLOSURE_2arg(VMax,  uchar, vmaxq_u8 (a, b));
FUNCTOR_CLOSURE_2arg(VMax,  schar, vmaxq_s8 (a, b));
FUNCTOR_CLOSURE_2arg(VMax, ushort, vmaxq_u16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,  short, vmaxq_s16(a, b));
FUNCTOR_CLOSURE_2arg(VMax,    int, vmaxq_s32(a, b));
FUNCTOR_CLOSURE_2arg(VMax,  float, vmaxq_f32(a, b));

FUNCTOR_TEMPLATE(VAbsDiff);
FUNCTOR_CLOSURE_2arg(VAbsDiff,  uchar, vabdq_u8  (a, b));
FUNCTOR_CLOSURE_2arg(VAbsDiff,  schar, vqabsq_s8 (vqsubq_s8(a, b)));
FUNCTOR_CLOSURE_2arg(VAbsDiff, ushort, vabdq_u16 (a, b));
FUNCTOR_CLOSURE_2arg(VAbsDiff,  short, vqabsq_s16(vqsubq_s16(a, b)));
FUNCTOR_CLOSURE_2arg(VAbsDiff,    int, vabdq_s32 (a, b));
FUNCTOR_CLOSURE_2arg(VAbsDiff,  float, vabdq_f32 (a, b));

FUNCTOR_TEMPLATE(VAnd);
FUNCTOR_CLOSURE_2arg(VAnd, uchar, vandq_u8(a, b));
FUNCTOR_TEMPLATE(VOr);
FUNCTOR_CLOSURE_2arg(VOr , uchar, vorrq_u8(a, b));
FUNCTOR_TEMPLATE(VXor);
FUNCTOR_CLOSURE_2arg(VXor, uchar, veorq_u8(a, b));
FUNCTOR_TEMPLATE(VNot);
FUNCTOR_CLOSURE_1arg(VNot, uchar, vmvnq_u8(a   ));
#endif


template <typename T>
struct Cmp_SIMD
{
    explicit Cmp_SIMD(int)
    {
    }

    int operator () (const T *, const T *, uchar *, int) const
    {
        return 0;
    }
};

#if CV_NEON

template <>
struct Cmp_SIMD<schar>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdupq_n_u8(255);
    }

    int operator () (const schar * src1, const schar * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, vcgtq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)));
        else if (code == CMP_LE)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, vcleq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)));
        else if (code == CMP_EQ)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, vceqq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)));
        else if (code == CMP_NE)
            for ( ; x <= width - 16; x += 16)
                vst1q_u8(dst + x, veorq_u8(vceqq_s8(vld1q_s8(src1 + x), vld1q_s8(src2 + x)), v_mask));

        return x;
    }

    int code;
    uint8x16_t v_mask;
};

template <>
struct Cmp_SIMD<ushort>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdup_n_u8(255);
    }

    int operator () (const ushort * src1, const ushort * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vcgtq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, vmovn_u16(v_dst));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vcleq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, vmovn_u16(v_dst));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vceqq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, vmovn_u16(v_dst));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_dst = vceqq_u16(vld1q_u16(src1 + x), vld1q_u16(src2 + x));
                vst1_u8(dst + x, veor_u8(vmovn_u16(v_dst), v_mask));
            }

        return x;
    }

    int code;
    uint8x8_t v_mask;
};

template <>
struct Cmp_SIMD<int>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdup_n_u8(255);
    }

    int operator () (const int * src1, const int * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcgtq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vcgtq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcleq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vcleq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vceqq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_s32(vld1q_s32(src1 + x), vld1q_s32(src2 + x));
                uint32x4_t v_dst2 = vceqq_s32(vld1q_s32(src1 + x + 4), vld1q_s32(src2 + x + 4));
                uint8x8_t v_dst = vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2)));
                vst1_u8(dst + x, veor_u8(v_dst, v_mask));
            }

        return x;
    }

    int code;
    uint8x8_t v_mask;
};

template <>
struct Cmp_SIMD<float>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        v_mask = vdup_n_u8(255);
    }

    int operator () (const float * src1, const float * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcgtq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vcgtq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vcleq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vcleq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vceqq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1_u8(dst + x, vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2))));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                uint32x4_t v_dst1 = vceqq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                uint32x4_t v_dst2 = vceqq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                uint8x8_t v_dst = vmovn_u16(vcombine_u16(vmovn_u32(v_dst1), vmovn_u32(v_dst2)));
                vst1_u8(dst + x, veor_u8(v_dst, v_mask));
            }

        return x;
    }

    int code;
    uint8x8_t v_mask;
};

#elif CV_SSE2

template <>
struct Cmp_SIMD<schar>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        haveSSE = checkHardwareSupport(CV_CPU_SSE2);

        v_mask = _mm_set1_epi8(-1);
    }

    int operator () (const schar * src1, const schar * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        if (code == CMP_GT)
            for ( ; x <= width - 16; x += 16)
                _mm_storeu_si128((__m128i *)(dst + x), _mm_cmpgt_epi8(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                                                      _mm_loadu_si128((const __m128i *)(src2 + x))));
        else if (code == CMP_LE)
            for ( ; x <= width - 16; x += 16)
            {
                __m128i v_gt = _mm_cmpgt_epi8(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                              _mm_loadu_si128((const __m128i *)(src2 + x)));
                _mm_storeu_si128((__m128i *)(dst + x), _mm_xor_si128(v_mask, v_gt));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 16; x += 16)
                _mm_storeu_si128((__m128i *)(dst + x), _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                                                      _mm_loadu_si128((const __m128i *)(src2 + x))));
        else if (code == CMP_NE)
            for ( ; x <= width - 16; x += 16)
            {
                __m128i v_eq = _mm_cmpeq_epi8(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                              _mm_loadu_si128((const __m128i *)(src2 + x)));
                _mm_storeu_si128((__m128i *)(dst + x), _mm_xor_si128(v_mask, v_eq));
            }

        return x;
    }

    int code;
    __m128i v_mask;
    bool haveSSE;
};

template <>
struct Cmp_SIMD<int>
{
    explicit Cmp_SIMD(int code_) :
        code(code_)
    {
        // CV_Assert(code == CMP_GT || code == CMP_LE ||
        //           code == CMP_EQ || code == CMP_NE);

        haveSSE = checkHardwareSupport(CV_CPU_SSE2);

        v_mask = _mm_set1_epi32(0xffffffff);
    }

    int operator () (const int * src1, const int * src2, uchar * dst, int width) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        if (code == CMP_GT)
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_dst0 = _mm_cmpgt_epi32(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x)));
                __m128i v_dst1 = _mm_cmpgt_epi32(_mm_loadu_si128((const __m128i *)(src1 + x + 4)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x + 4)));

                _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(_mm_packs_epi32(v_dst0, v_dst1), v_mask));
            }
        else if (code == CMP_LE)
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_dst0 = _mm_cmpgt_epi32(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x)));
                __m128i v_dst1 = _mm_cmpgt_epi32(_mm_loadu_si128((const __m128i *)(src1 + x + 4)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x + 4)));

                _mm_storel_epi64((__m128i *)(dst + x), _mm_xor_si128(_mm_packs_epi16(_mm_packs_epi32(v_dst0, v_dst1), v_mask), v_mask));
            }
        else if (code == CMP_EQ)
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_dst0 = _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x)));
                __m128i v_dst1 = _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i *)(src1 + x + 4)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x + 4)));

                _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(_mm_packs_epi32(v_dst0, v_dst1), v_mask));
            }
        else if (code == CMP_NE)
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_dst0 = _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i *)(src1 + x)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x)));
                __m128i v_dst1 = _mm_cmpeq_epi32(_mm_loadu_si128((const __m128i *)(src1 + x + 4)),
                                                 _mm_loadu_si128((const __m128i *)(src2 + x + 4)));

                _mm_storel_epi64((__m128i *)(dst + x), _mm_xor_si128(v_mask, _mm_packs_epi16(_mm_packs_epi32(v_dst0, v_dst1), v_mask)));
            }

        return x;
    }

    int code;
    __m128i v_mask;
    bool haveSSE;
};

#endif


template <typename T, typename WT>
struct Mul_SIMD
{
    int operator() (const T *, const T *, T *, int, WT) const
    {
        return 0;
    }
};

#if CV_NEON

template <>
struct Mul_SIMD<uchar, float>
{
    int operator() (const uchar * src1, const uchar * src2, uchar * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + x));
                uint16x8_t v_src2 = vmovl_u8(vld1_u8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1_u8(dst + x, vqmovn_u16(v_dst));
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vmovl_u8(vld1_u8(src1 + x));
                uint16x8_t v_src2 = vmovl_u8(vld1_u8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1_u8(dst + x, vqmovn_u16(v_dst));
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<schar, float>
{
    int operator() (const schar * src1, const schar * src2, schar * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vmovl_s8(vld1_s8(src1 + x));
                int16x8_t v_src2 = vmovl_s8(vld1_s8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1_s8(dst + x, vqmovn_s16(v_dst));
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vmovl_s8(vld1_s8(src1 + x));
                int16x8_t v_src2 = vmovl_s8(vld1_s8(src2 + x));

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1_s8(dst + x, vqmovn_s16(v_dst));
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<ushort, float>
{
    int operator() (const ushort * src1, const ushort * src2, ushort * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vld1q_u16(src1 + x), v_src2 = vld1q_u16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1q_u16(dst + x, v_dst);
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                uint16x8_t v_src1 = vld1q_u16(src1 + x), v_src2 = vld1q_u16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))),
                                               vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                uint16x8_t v_dst = vcombine_u16(vqmovn_u32(cv_vrndq_u32_f32(v_dst1)),
                                                vqmovn_u32(cv_vrndq_u32_f32(v_dst2)));
                vst1q_u16(dst + x, v_dst);
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<short, float>
{
    int operator() (const short * src1, const short * src2, short * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vld1q_s16(src1 + x), v_src2 = vld1q_s16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1q_s16(dst + x, v_dst);
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                int16x8_t v_src1 = vld1q_s16(src1 + x), v_src2 = vld1q_s16(src2 + x);

                float32x4_t v_dst1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))));
                v_dst1 = vmulq_f32(v_dst1, v_scale);
                float32x4_t v_dst2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))),
                                               vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                int16x8_t v_dst = vcombine_s16(vqmovn_s32(cv_vrndq_s32_f32(v_dst1)),
                                               vqmovn_s32(cv_vrndq_s32_f32(v_dst2)));
                vst1q_s16(dst + x, v_dst);
            }
        }

        return x;
    }
};

template <>
struct Mul_SIMD<float, float>
{
    int operator() (const float * src1, const float * src2, float * dst, int width, float scale) const
    {
        int x = 0;

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                float32x4_t v_dst1 = vmulq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                float32x4_t v_dst2 = vmulq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                vst1q_f32(dst + x, v_dst1);
                vst1q_f32(dst + x + 4, v_dst2);
            }
        else
        {
            float32x4_t v_scale = vdupq_n_f32(scale);
            for ( ; x <= width - 8; x += 8)
            {
                float32x4_t v_dst1 = vmulq_f32(vld1q_f32(src1 + x), vld1q_f32(src2 + x));
                v_dst1 = vmulq_f32(v_dst1, v_scale);

                float32x4_t v_dst2 = vmulq_f32(vld1q_f32(src1 + x + 4), vld1q_f32(src2 + x + 4));
                v_dst2 = vmulq_f32(v_dst2, v_scale);

                vst1q_f32(dst + x, v_dst1);
                vst1q_f32(dst + x + 4, v_dst2);
            }
        }

        return x;
    }
};

#elif CV_SSE2

#if CV_SSE4_1

template <>
struct Mul_SIMD<ushort, float>
{
    Mul_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator() (const ushort * src1, const ushort * src2, ushort * dst, int width, float scale) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();

        if( scale != 1.0f )
        {
            __m128 v_scale = _mm_set1_ps(scale);
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_src1 = _mm_loadu_si128((__m128i const *)(src1 + x));
                __m128i v_src2 = _mm_loadu_si128((__m128i const *)(src2 + x));

                __m128 v_dst1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src1, v_zero)),
                                           _mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src2, v_zero)));
                v_dst1 = _mm_mul_ps(v_dst1, v_scale);

                __m128 v_dst2 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src1, v_zero)),
                                           _mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src2, v_zero)));
                v_dst2 = _mm_mul_ps(v_dst2, v_scale);

                __m128i v_dsti = _mm_packus_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2));
                _mm_storeu_si128((__m128i *)(dst + x), v_dsti);
            }
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <>
struct Mul_SIMD<schar, float>
{
    Mul_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const schar * src1, const schar * src2, schar * dst, int width, float scale) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();

        if( scale == 1.0f )
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src1 + x));
                __m128i v_src2 = _mm_loadl_epi64((__m128i const *)(src2 + x));

                v_src1 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src1), 8);
                v_src2 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src2), 8);

                __m128 v_dst1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src1), 16)),
                                           _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src2), 16)));

                __m128 v_dst2 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src1), 16)),
                                           _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src2), 16)));

                __m128i v_dsti = _mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2));
                _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dsti, v_zero));
            }
        else
        {
            __m128 v_scale = _mm_set1_ps(scale);
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_src1 = _mm_loadl_epi64((__m128i const *)(src1 + x));
                __m128i v_src2 = _mm_loadl_epi64((__m128i const *)(src2 + x));

                v_src1 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src1), 8);
                v_src2 = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src2), 8);

                __m128 v_dst1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src1), 16)),
                                           _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src2), 16)));
                v_dst1 = _mm_mul_ps(v_dst1, v_scale);

                __m128 v_dst2 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src1), 16)),
                                           _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src2), 16)));
                v_dst2 = _mm_mul_ps(v_dst2, v_scale);

                __m128i v_dsti = _mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2));
                _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dsti, v_zero));
            }
        }

        return x;
    }

    bool haveSSE;
};

template <>
struct Mul_SIMD<short, float>
{
    Mul_SIMD()
    {
        haveSSE = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const short * src1, const short * src2, short * dst, int width, float scale) const
    {
        int x = 0;

        if (!haveSSE)
            return x;

        __m128i v_zero = _mm_setzero_si128();

        if( scale != 1.0f )
        {
            __m128 v_scale = _mm_set1_ps(scale);
            for ( ; x <= width - 8; x += 8)
            {
                __m128i v_src1 = _mm_loadu_si128((__m128i const *)(src1 + x));
                __m128i v_src2 = _mm_loadu_si128((__m128i const *)(src2 + x));

                __m128 v_dst1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src1), 16)),
                                           _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src2), 16)));
                v_dst1 = _mm_mul_ps(v_dst1, v_scale);

                __m128 v_dst2 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src1), 16)),
                                           _mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src2), 16)));
                v_dst2 = _mm_mul_ps(v_dst2, v_scale);

                __m128i v_dsti = _mm_packs_epi32(_mm_cvtps_epi32(v_dst1), _mm_cvtps_epi32(v_dst2));
                _mm_storeu_si128((__m128i *)(dst + x), v_dsti);
            }
        }

        return x;
    }

    bool haveSSE;
};

#endif

template <typename T>
struct Div_SIMD
{
    int operator() (const T *, const T *, T *, int, double) const
    {
        return 0;
    }
};

template <typename T>
struct Recip_SIMD
{
    int operator() (const T *, T *, int, double) const
    {
        return 0;
    }
};


#if CV_SIMD128

template <>
struct Div_SIMD<uchar>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const uchar * src1, const uchar * src2, uchar * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_uint16x8 v_zero = v_setzero_u16();

        for ( ; x <= width - 8; x += 8)
        {
            v_uint16x8 v_src1 = v_load_expand(src1 + x);
            v_uint16x8 v_src2 = v_load_expand(src2 + x);

            v_uint32x4 t0, t1, t2, t3;
            v_expand(v_src1, t0, t1);
            v_expand(v_src2, t2, t3);

            v_float32x4 f0 = v_cvt_f32(v_reinterpret_as_s32(t0));
            v_float32x4 f1 = v_cvt_f32(v_reinterpret_as_s32(t1));

            v_float32x4 f2 = v_cvt_f32(v_reinterpret_as_s32(t2));
            v_float32x4 f3 = v_cvt_f32(v_reinterpret_as_s32(t3));

            f0 = f0 * v_scale / f2;
            f1 = f1 * v_scale / f3;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_uint16x8 res = v_pack_u(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_pack_store(dst + x, res);
        }

        return x;
    }
};


template <>
struct Div_SIMD<schar>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const schar * src1, const schar * src2, schar * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int16x8 v_zero = v_setzero_s16();

        for ( ; x <= width - 8; x += 8)
        {
            v_int16x8 v_src1 = v_load_expand(src1 + x);
            v_int16x8 v_src2 = v_load_expand(src2 + x);

            v_int32x4 t0, t1, t2, t3;
            v_expand(v_src1, t0, t1);
            v_expand(v_src2, t2, t3);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            v_float32x4 f2 = v_cvt_f32(t2);
            v_float32x4 f3 = v_cvt_f32(t3);

            f0 = f0 * v_scale / f2;
            f1 = f1 * v_scale / f3;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_int16x8 res = v_pack(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_pack_store(dst + x, res);
        }

        return x;
    }
};


template <>
struct Div_SIMD<ushort>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const ushort * src1, const ushort * src2, ushort * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_uint16x8 v_zero = v_setzero_u16();

        for ( ; x <= width - 8; x += 8)
        {
            v_uint16x8 v_src1 = v_load(src1 + x);
            v_uint16x8 v_src2 = v_load(src2 + x);

            v_uint32x4 t0, t1, t2, t3;
            v_expand(v_src1, t0, t1);
            v_expand(v_src2, t2, t3);

            v_float32x4 f0 = v_cvt_f32(v_reinterpret_as_s32(t0));
            v_float32x4 f1 = v_cvt_f32(v_reinterpret_as_s32(t1));

            v_float32x4 f2 = v_cvt_f32(v_reinterpret_as_s32(t2));
            v_float32x4 f3 = v_cvt_f32(v_reinterpret_as_s32(t3));

            f0 = f0 * v_scale / f2;
            f1 = f1 * v_scale / f3;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_uint16x8 res = v_pack_u(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_store(dst + x, res);
        }

        return x;
    }
};

template <>
struct Div_SIMD<short>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const short * src1, const short * src2, short * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int16x8 v_zero = v_setzero_s16();

        for ( ; x <= width - 8; x += 8)
        {
            v_int16x8 v_src1 = v_load(src1 + x);
            v_int16x8 v_src2 = v_load(src2 + x);

            v_int32x4 t0, t1, t2, t3;
            v_expand(v_src1, t0, t1);
            v_expand(v_src2, t2, t3);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            v_float32x4 f2 = v_cvt_f32(t2);
            v_float32x4 f3 = v_cvt_f32(t3);

            f0 = f0 * v_scale / f2;
            f1 = f1 * v_scale / f3;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_int16x8 res = v_pack(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_store(dst + x, res);
        }

        return x;
    }
};

template <>
struct Div_SIMD<int>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const int * src1, const int * src2, int * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int32x4 v_zero = v_setzero_s32();

        for ( ; x <= width - 8; x += 8)
        {
            v_int32x4 t0 = v_load(src1 + x);
            v_int32x4 t1 = v_load(src1 + x + 4);
            v_int32x4 t2 = v_load(src2 + x);
            v_int32x4 t3 = v_load(src2 + x + 4);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);
            v_float32x4 f2 = v_cvt_f32(t2);
            v_float32x4 f3 = v_cvt_f32(t3);

            f0 = f0 * v_scale / f2;
            f1 = f1 * v_scale / f3;

            v_int32x4 res0 = v_round(f0), res1 = v_round(f1);

            res0 = v_select(t2 == v_zero, v_zero, res0);
            res1 = v_select(t3 == v_zero, v_zero, res1);
            v_store(dst + x, res0);
            v_store(dst + x + 4, res1);
        }

        return x;
    }
};


template <>
struct Div_SIMD<float>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const float * src1, const float * src2, float * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_float32x4 v_zero = v_setzero_f32();

        for ( ; x <= width - 8; x += 8)
        {
            v_float32x4 f0 = v_load(src1 + x);
            v_float32x4 f1 = v_load(src1 + x + 4);
            v_float32x4 f2 = v_load(src2 + x);
            v_float32x4 f3 = v_load(src2 + x + 4);

            v_float32x4 res0 = f0 * v_scale / f2;
            v_float32x4 res1 = f1 * v_scale / f3;

            res0 = v_select(f2 == v_zero, v_zero, res0);
            res1 = v_select(f3 == v_zero, v_zero, res1);

            v_store(dst + x, res0);
            v_store(dst + x + 4, res1);
        }

        return x;
    }
};


///////////////////////// RECIPROCAL //////////////////////

template <>
struct Recip_SIMD<uchar>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const uchar * src2, uchar * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_uint16x8 v_zero = v_setzero_u16();

        for ( ; x <= width - 8; x += 8)
        {
            v_uint16x8 v_src2 = v_load_expand(src2 + x);

            v_uint32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(v_reinterpret_as_s32(t0));
            v_float32x4 f1 = v_cvt_f32(v_reinterpret_as_s32(t1));

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_uint16x8 res = v_pack_u(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_pack_store(dst + x, res);
        }

        return x;
    }
};


template <>
struct Recip_SIMD<schar>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const schar * src2, schar * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int16x8 v_zero = v_setzero_s16();

        for ( ; x <= width - 8; x += 8)
        {
            v_int16x8 v_src2 = v_load_expand(src2 + x);

            v_int32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_int16x8 res = v_pack(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_pack_store(dst + x, res);
        }

        return x;
    }
};


template <>
struct Recip_SIMD<ushort>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const ushort * src2, ushort * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_uint16x8 v_zero = v_setzero_u16();

        for ( ; x <= width - 8; x += 8)
        {
            v_uint16x8 v_src2 = v_load(src2 + x);

            v_uint32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(v_reinterpret_as_s32(t0));
            v_float32x4 f1 = v_cvt_f32(v_reinterpret_as_s32(t1));

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_uint16x8 res = v_pack_u(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_store(dst + x, res);
        }

        return x;
    }
};

template <>
struct Recip_SIMD<short>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const short * src2, short * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int16x8 v_zero = v_setzero_s16();

        for ( ; x <= width - 8; x += 8)
        {
            v_int16x8 v_src2 = v_load(src2 + x);

            v_int32x4 t0, t1;
            v_expand(v_src2, t0, t1);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 i0 = v_round(f0), i1 = v_round(f1);
            v_int16x8 res = v_pack(i0, i1);

            res = v_select(v_src2 == v_zero, v_zero, res);
            v_store(dst + x, res);
        }

        return x;
    }
};

template <>
struct Recip_SIMD<int>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const int * src2, int * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_int32x4 v_zero = v_setzero_s32();

        for ( ; x <= width - 8; x += 8)
        {
            v_int32x4 t0 = v_load(src2 + x);
            v_int32x4 t1 = v_load(src2 + x + 4);

            v_float32x4 f0 = v_cvt_f32(t0);
            v_float32x4 f1 = v_cvt_f32(t1);

            f0 = v_scale / f0;
            f1 = v_scale / f1;

            v_int32x4 res0 = v_round(f0), res1 = v_round(f1);

            res0 = v_select(t0 == v_zero, v_zero, res0);
            res1 = v_select(t1 == v_zero, v_zero, res1);
            v_store(dst + x, res0);
            v_store(dst + x + 4, res1);
        }

        return x;
    }
};


template <>
struct Recip_SIMD<float>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const float * src2, float * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float32x4 v_scale = v_setall_f32((float)scale);
        v_float32x4 v_zero = v_setzero_f32();

        for ( ; x <= width - 8; x += 8)
        {
            v_float32x4 f0 = v_load(src2 + x);
            v_float32x4 f1 = v_load(src2 + x + 4);

            v_float32x4 res0 = v_scale / f0;
            v_float32x4 res1 = v_scale / f1;

            res0 = v_select(f0 == v_zero, v_zero, res0);
            res1 = v_select(f1 == v_zero, v_zero, res1);

            v_store(dst + x, res0);
            v_store(dst + x + 4, res1);
        }

        return x;
    }
};

#if CV_SIMD128_64F

template <>
struct Div_SIMD<double>
{
    bool haveSIMD;
    Div_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const double * src1, const double * src2, double * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float64x2 v_scale = v_setall_f64(scale);
        v_float64x2 v_zero = v_setzero_f64();

        for ( ; x <= width - 4; x += 4)
        {
            v_float64x2 f0 = v_load(src1 + x);
            v_float64x2 f1 = v_load(src1 + x + 2);
            v_float64x2 f2 = v_load(src2 + x);
            v_float64x2 f3 = v_load(src2 + x + 2);

            v_float64x2 res0 = f0 * v_scale / f2;
            v_float64x2 res1 = f1 * v_scale / f3;

            res0 = v_select(f2 == v_zero, v_zero, res0);
            res1 = v_select(f3 == v_zero, v_zero, res1);

            v_store(dst + x, res0);
            v_store(dst + x + 2, res1);
        }

        return x;
    }
};

template <>
struct Recip_SIMD<double>
{
    bool haveSIMD;
    Recip_SIMD() { haveSIMD = hasSIMD128(); }

    int operator() (const double * src2, double * dst, int width, double scale) const
    {
        int x = 0;

        if (!haveSIMD)
            return x;

        v_float64x2 v_scale = v_setall_f64(scale);
        v_float64x2 v_zero = v_setzero_f64();

        for ( ; x <= width - 4; x += 4)
        {
            v_float64x2 f0 = v_load(src2 + x);
            v_float64x2 f1 = v_load(src2 + x + 2);

            v_float64x2 res0 = v_scale / f0;
            v_float64x2 res1 = v_scale / f1;

            res0 = v_select(f0 == v_zero, v_zero, res0);
            res1 = v_select(f1 == v_zero, v_zero, res1);

            v_store(dst + x, res0);
            v_store(dst + x + 2, res1);
        }

        return x;
    }
};

#endif

#endif


template <typename T, typename WT>
struct AddWeighted_SIMD
{
    int operator() (const T *, const T *, T *, int, WT, WT, WT) const
    {
        return 0;
    }
};

#if CV_SSE2

template <>
struct AddWeighted_SIMD<schar, float>
{
    AddWeighted_SIMD()
    {
        haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const schar * src1, const schar * src2, schar * dst, int width, float alpha, float beta, float gamma) const
    {
        int x = 0;

        if (!haveSSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_alpha = _mm_set1_ps(alpha), v_beta = _mm_set1_ps(beta),
               v_gamma = _mm_set1_ps(gamma);

        for( ; x <= width - 8; x += 8 )
        {
            __m128i v_src1 = _mm_loadl_epi64((const __m128i *)(src1 + x));
            __m128i v_src2 = _mm_loadl_epi64((const __m128i *)(src2 + x));

            __m128i v_src1_p = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src1), 8);
            __m128i v_src2_p = _mm_srai_epi16(_mm_unpacklo_epi8(v_zero, v_src2), 8);

            __m128 v_dstf0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src1_p), 16)), v_alpha);
            v_dstf0 = _mm_add_ps(_mm_add_ps(v_dstf0, v_gamma),
                                 _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src2_p), 16)), v_beta));

            __m128 v_dstf1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src1_p), 16)), v_alpha);
            v_dstf1 = _mm_add_ps(_mm_add_ps(v_dstf1, v_gamma),
                                 _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src2_p), 16)), v_beta));

            __m128i v_dst16 = _mm_packs_epi32(_mm_cvtps_epi32(v_dstf0),
                                              _mm_cvtps_epi32(v_dstf1));

            _mm_storel_epi64((__m128i *)(dst + x), _mm_packs_epi16(v_dst16, v_zero));
        }

        return x;
    }

    bool haveSSE2;
};

template <>
struct AddWeighted_SIMD<short, float>
{
    AddWeighted_SIMD()
    {
        haveSSE2 = checkHardwareSupport(CV_CPU_SSE2);
    }

    int operator() (const short * src1, const short * src2, short * dst, int width, float alpha, float beta, float gamma) const
    {
        int x = 0;

        if (!haveSSE2)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_alpha = _mm_set1_ps(alpha), v_beta = _mm_set1_ps(beta),
               v_gamma = _mm_set1_ps(gamma);

        for( ; x <= width - 8; x += 8 )
        {
            __m128i v_src1 = _mm_loadu_si128((const __m128i *)(src1 + x));
            __m128i v_src2 = _mm_loadu_si128((const __m128i *)(src2 + x));

            __m128 v_dstf0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src1), 16)), v_alpha);
            v_dstf0 = _mm_add_ps(_mm_add_ps(v_dstf0, v_gamma),
                                 _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpacklo_epi16(v_zero, v_src2), 16)), v_beta));

            __m128 v_dstf1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src1), 16)), v_alpha);
            v_dstf1 = _mm_add_ps(_mm_add_ps(v_dstf1, v_gamma),
                                 _mm_mul_ps(_mm_cvtepi32_ps(_mm_srai_epi32(_mm_unpackhi_epi16(v_zero, v_src2), 16)), v_beta));

            _mm_storeu_si128((__m128i *)(dst + x), _mm_packs_epi32(_mm_cvtps_epi32(v_dstf0),
                                                                   _mm_cvtps_epi32(v_dstf1)));
        }

        return x;
    }

    bool haveSSE2;
};

#if CV_SSE4_1

template <>
struct AddWeighted_SIMD<ushort, float>
{
    AddWeighted_SIMD()
    {
        haveSSE4_1 = checkHardwareSupport(CV_CPU_SSE4_1);
    }

    int operator() (const ushort * src1, const ushort * src2, ushort * dst, int width, float alpha, float beta, float gamma) const
    {
        int x = 0;

        if (!haveSSE4_1)
            return x;

        __m128i v_zero = _mm_setzero_si128();
        __m128 v_alpha = _mm_set1_ps(alpha), v_beta = _mm_set1_ps(beta),
               v_gamma = _mm_set1_ps(gamma);

        for( ; x <= width - 8; x += 8 )
        {
            __m128i v_src1 = _mm_loadu_si128((const __m128i *)(src1 + x));
            __m128i v_src2 = _mm_loadu_si128((const __m128i *)(src2 + x));

            __m128 v_dstf0 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src1, v_zero)), v_alpha);
            v_dstf0 = _mm_add_ps(_mm_add_ps(v_dstf0, v_gamma),
                                 _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpacklo_epi16(v_src2, v_zero)), v_beta));

            __m128 v_dstf1 = _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src1, v_zero)), v_alpha);
            v_dstf1 = _mm_add_ps(_mm_add_ps(v_dstf1, v_gamma),
                                 _mm_mul_ps(_mm_cvtepi32_ps(_mm_unpackhi_epi16(v_src2, v_zero)), v_beta));

            _mm_storeu_si128((__m128i *)(dst + x), _mm_packus_epi32(_mm_cvtps_epi32(v_dstf0),
                                                                    _mm_cvtps_epi32(v_dstf1)));
        }

        return x;
    }

    bool haveSSE4_1;
};

#endif

#elif CV_NEON

template <>
struct AddWeighted_SIMD<schar, float>
{
    int operator() (const schar * src1, const schar * src2, schar * dst, int width, float alpha, float beta, float gamma) const
    {
        int x = 0;

        float32x4_t g = vdupq_n_f32 (gamma);

        for( ; x <= width - 8; x += 8 )
        {
            int8x8_t in1 = vld1_s8(src1 + x);
            int16x8_t in1_16 = vmovl_s8(in1);
            float32x4_t in1_f_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(in1_16)));
            float32x4_t in1_f_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(in1_16)));

            int8x8_t in2 = vld1_s8(src2+x);
            int16x8_t in2_16 = vmovl_s8(in2);
            float32x4_t in2_f_l = vcvtq_f32_s32(vmovl_s16(vget_low_s16(in2_16)));
            float32x4_t in2_f_h = vcvtq_f32_s32(vmovl_s16(vget_high_s16(in2_16)));

            float32x4_t out_f_l = vaddq_f32(vmulq_n_f32(in1_f_l, alpha), vmulq_n_f32(in2_f_l, beta));
            float32x4_t out_f_h = vaddq_f32(vmulq_n_f32(in1_f_h, alpha), vmulq_n_f32(in2_f_h, beta));
            out_f_l = vaddq_f32(out_f_l, g);
            out_f_h = vaddq_f32(out_f_h, g);

            int16x4_t out_16_l = vqmovn_s32(cv_vrndq_s32_f32(out_f_l));
            int16x4_t out_16_h = vqmovn_s32(cv_vrndq_s32_f32(out_f_h));

            int16x8_t out_16 = vcombine_s16(out_16_l, out_16_h);
            int8x8_t out = vqmovn_s16(out_16);

            vst1_s8(dst + x, out);
        }

        return x;
    }
};

template <>
struct AddWeighted_SIMD<ushort, float>
{
    int operator() (const ushort * src1, const ushort * src2, ushort * dst, int width, float alpha, float beta, float gamma) const
    {
        int x = 0;

        float32x4_t g = vdupq_n_f32(gamma);

        for( ; x <= width - 8; x += 8 )
        {
            uint16x8_t v_src1 = vld1q_u16(src1 + x), v_src2 = vld1q_u16(src2 + x);

            float32x4_t v_s1 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src1))), alpha);
            float32x4_t v_s2 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_low_u16(v_src2))), beta);
            uint16x4_t v_dst1 = vqmovn_u32(cv_vrndq_u32_f32(vaddq_f32(vaddq_f32(v_s1, v_s2), g)));

            v_s1 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src1))), alpha);
            v_s2 = vmulq_n_f32(vcvtq_f32_u32(vmovl_u16(vget_high_u16(v_src2))), beta);
            uint16x4_t v_dst2 = vqmovn_u32(cv_vrndq_u32_f32(vaddq_f32(vaddq_f32(v_s1, v_s2), g)));

            vst1q_u16(dst + x, vcombine_u16(v_dst1, v_dst2));
        }

        return x;
    }
};

template <>
struct AddWeighted_SIMD<short, float>
{
    int operator() (const short * src1, const short * src2, short * dst, int width, float alpha, float beta, float gamma) const
    {
        int x = 0;

        float32x4_t g = vdupq_n_f32(gamma);

        for( ; x <= width - 8; x += 8 )
        {
            int16x8_t v_src1 = vld1q_s16(src1 + x), v_src2 = vld1q_s16(src2 + x);

            float32x4_t v_s1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src1))), alpha);
            float32x4_t v_s2 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v_src2))), beta);
            int16x4_t v_dst1 = vqmovn_s32(cv_vrndq_s32_f32(vaddq_f32(vaddq_f32(v_s1, v_s2), g)));

            v_s1 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src1))), alpha);
            v_s2 = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(v_src2))), beta);
            int16x4_t v_dst2 = vqmovn_s32(cv_vrndq_s32_f32(vaddq_f32(vaddq_f32(v_s1, v_s2), g)));

            vst1q_s16(dst + x, vcombine_s16(v_dst1, v_dst2));
        }

        return x;
    }
};

#endif

}

#endif // __OPENCV_ARITHM_SIMD_HPP__
