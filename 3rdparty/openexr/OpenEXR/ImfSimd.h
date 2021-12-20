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
#if defined __SSE2__ || (_MSC_VER >= 1300 && !_M_CEE_PURE)
    #define IMF_HAVE_SSE2 1
#endif

#if defined __SSE4_1__
    #define IMF_HAVE_SSE4_1 1
#endif

// Compiler flags on e2k (MCST Elbrus 2000) architecture
#if defined(__SSE3__) && defined(__e2k__)
    #define IMF_HAVE_SSE3 1
#endif

#if defined(__SSSE3__) && defined(__e2k__)
    #define IMF_HAVE_SSSE3 1
#endif

#if defined(__SSE4_2__) && defined(__e2k__)
    #define IMF_HAVE_SSE4_2 1
#endif

#if defined(__AVX__) && defined(__e2k__)
    #define IMF_HAVE_AVX 1
#endif

#if defined(__F16C__) && defined(__e2k__)
    #define IMF_HAVE_F16C 1
#endif

extern "C"
{
#ifdef IMF_HAVE_SSE2
    #include <emmintrin.h>
    #include <mmintrin.h>
#endif

#ifdef IMF_HAVE_SSE4_1
    #include <smmintrin.h>
#endif
}

#endif
