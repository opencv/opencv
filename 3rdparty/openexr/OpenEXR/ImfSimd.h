//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) Autodesk and Contributors of the OpenEXR Project
//

#ifndef INCLUDED_IMF_SIMD_H
#define INCLUDED_IMF_SIMD_H

//
// Compile time SSE detection:
//    IMF_HAVE_SSE2 - Defined if it's safe to compile SSE2 optimizations
//    IMF_HAVE_SSE4_1 - Defined if it's safe to compile SSE4.1 optimizations
//

// GCC and Visual Studio SSE2 compiler flags
#if defined __SSE2__ || (_MSC_VER && (_M_IX86 || _M_X64))
#    define IMF_HAVE_SSE2 1
#endif

#if defined __SSE4_1__ || (_MSC_VER && (_M_IX86 || _M_X64))
#    define IMF_HAVE_SSE4_1 1
#endif

// Compiler flags on e2k (MCST Elbrus 2000) architecture
#if defined(__SSE3__) && defined(__e2k__)
#    define IMF_HAVE_SSE3 1
#endif

#if defined(__SSSE3__) && defined(__e2k__)
#    define IMF_HAVE_SSSE3 1
#endif

#if defined(__SSE4_2__) && defined(__e2k__)
#    define IMF_HAVE_SSE4_2 1
#endif

#if defined(__AVX__) && defined(__e2k__)
#    define IMF_HAVE_AVX 1
#endif

#if defined(__F16C__) && defined(__e2k__)
#    define IMF_HAVE_F16C 1
#endif

#if defined(__ARM_NEON)
#    define IMF_HAVE_NEON
#endif

#if defined(__aarch64__)
#    define IMF_HAVE_NEON_AARCH64 1
#endif

extern "C" {
#ifdef IMF_HAVE_SSE2
#    include <emmintrin.h>
#    include <mmintrin.h>
#endif

#ifdef IMF_HAVE_SSE4_1
#    include <smmintrin.h>
#endif

#ifdef IMF_HAVE_NEON
#    include <arm_neon.h>
#endif

}

#include "OpenEXRConfigInternal.h"
#ifdef OPENEXR_MISSING_ARM_VLD1
/* Workaround for missing vld1q_f32_x2 in older gcc versions.  */

__extension__ extern __inline float32x4x2_t
    __attribute__ ((__always_inline__, __gnu_inline__, __artificial__))
    vld1q_f32_x2 (const float32_t* __a)
{
    float32x4x2_t ret;
    asm ("ld1 {%S0.4s - %T0.4s}, [%1]" : "=w"(ret) : "r"(__a) :);
    return ret;
}
#endif

#endif
