//
// SPDX-License-Identifier: BSD-3-Clause
// Copyright Contributors to the OpenEXR Project.
//

//
// This file contains functions and constants which aren't
// provided by the system libraries, compilers, or includes on
// certain platforms.
//

#ifndef INCLUDED_IMATHPLATFORM_H
#define INCLUDED_IMATHPLATFORM_H

/// @cond Doxygen_Suppress

#include <math.h>

#include "ImathNamespace.h"

#ifdef __cplusplus

IMATH_INTERNAL_NAMESPACE_HEADER_ENTER

//
// Helpful macros for checking which C++ standard we are compiling with.
//
#if (__cplusplus >= 202002L)
#    define IMATH_CPLUSPLUS_VERSION 20
#elif (__cplusplus >= 201703L)
#    define IMATH_CPLUSPLUS_VERSION 17
#elif (__cplusplus >= 201402L) || (defined(_MSC_VER) && _MSC_VER >= 1914)
#    define IMATH_CPLUSPLUS_VERSION 14
#elif (__cplusplus >= 201103L) || (defined(_MSC_VER) && _MSC_VER >= 1900)
#    define IMATH_CPLUSPLUS_VERSION 11
#else
#    error "This version of Imath is meant to work only with C++11 and above"
#endif


//
// Constexpr C++14 conditional definition
//
#if (IMATH_CPLUSPLUS_VERSION >= 14)
  #define IMATH_CONSTEXPR14 constexpr
#else
  #define IMATH_CONSTEXPR14 /* can not be constexpr before c++14 */
#endif

#endif // __cplusplus

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

#ifndef M_PI_2
#    define M_PI_2 1.57079632679489661923 // pi/2
#endif

//-----------------------------------------------------------------------------
//
//    Some, but not all, C++ compilers support the C99 restrict
//    keyword or some variant of it, for example, __restrict.
//
//-----------------------------------------------------------------------------

#if defined(__GNUC__) || defined(__clang__) || defined(_MSC_VER) || defined(__INTEL_COMPILER)
#    define IMATH_RESTRICT __restrict
#else
#    define IMATH_RESTRICT
#endif

#ifdef __cplusplus

IMATH_INTERNAL_NAMESPACE_HEADER_EXIT

#endif // __cplusplus

/// @endcond

#endif // INCLUDED_IMATHPLATFORM_H
