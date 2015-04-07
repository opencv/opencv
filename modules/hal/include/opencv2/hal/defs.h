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

#ifndef __OPENCV_DEF_H__
#define __OPENCV_DEF_H__

#if !defined _CRT_SECURE_NO_DEPRECATE && defined _MSC_VER && _MSC_VER > 1300
#  define _CRT_SECURE_NO_DEPRECATE /* to avoid multiple Visual Studio warnings */
#endif

#if defined __ICL
#  define CV_ICC   __ICL
#elif defined __ICC
#  define CV_ICC   __ICC
#elif defined __ECL
#  define CV_ICC   __ECL
#elif defined __ECC
#  define CV_ICC   __ECC
#elif defined __INTEL_COMPILER
#  define CV_ICC   __INTEL_COMPILER
#endif

#if defined CV_ICC && !defined CV_ENABLE_UNROLLED
#  define CV_ENABLE_UNROLLED 0
#else
#  define CV_ENABLE_UNROLLED 1
#endif

/* CPU features and intrinsics support */
#define CV_CPU_NONE             0
#define CV_CPU_MMX              1
#define CV_CPU_SSE              2
#define CV_CPU_SSE2             3
#define CV_CPU_SSE3             4
#define CV_CPU_SSSE3            5
#define CV_CPU_SSE4_1           6
#define CV_CPU_SSE4_2           7
#define CV_CPU_POPCNT           8

#define CV_CPU_AVX              10
#define CV_CPU_AVX2             11
#define CV_CPU_FMA3             12

#define CV_CPU_AVX_512F         13
#define CV_CPU_AVX_512BW        14
#define CV_CPU_AVX_512CD        15
#define CV_CPU_AVX_512DQ        16
#define CV_CPU_AVX_512ER        17
#define CV_CPU_AVX_512IFMA512   18
#define CV_CPU_AVX_512PF        19
#define CV_CPU_AVX_512VBMI      20
#define CV_CPU_AVX_512VL        21

#define CV_CPU_NEON   100

// when adding to this list remember to update the enum in core/utility.cpp
#define CV_HARDWARE_MAX_FEATURE 255

// do not include SSE/AVX/NEON headers for NVCC compiler
#ifndef __CUDACC__

#if defined __SSE2__ || defined _M_X64  || (defined _M_IX86_FP && _M_IX86_FP >= 2)
#  include <emmintrin.h>
#  define CV_MMX 1
#  define CV_SSE 1
#  define CV_SSE2 1
#  if defined __SSE3__ || (defined _MSC_VER && _MSC_VER >= 1500)
#    include <pmmintrin.h>
#    define CV_SSE3 1
#  endif
#  if defined __SSSE3__  || (defined _MSC_VER && _MSC_VER >= 1500)
#    include <tmmintrin.h>
#    define CV_SSSE3 1
#  endif
#  if defined __SSE4_1__ || (defined _MSC_VER && _MSC_VER >= 1500)
#    include <smmintrin.h>
#    define CV_SSE4_1 1
#  endif
#  if defined __SSE4_2__ || (defined _MSC_VER && _MSC_VER >= 1500)
#    include <nmmintrin.h>
#    define CV_SSE4_2 1
#  endif
#  if defined __POPCNT__ || (defined _MSC_VER && _MSC_VER >= 1500)
#    ifdef _MSC_VER
#      include <nmmintrin.h>
#    else
#      include <popcntintrin.h>
#    endif
#    define CV_POPCNT 1
#  endif
#  if defined __AVX__ || (defined _MSC_VER && _MSC_VER >= 1600 && 0)
// MS Visual Studio 2010 (2012?) has no macro pre-defined to identify the use of /arch:AVX
// See: http://connect.microsoft.com/VisualStudio/feedback/details/605858/arch-avx-should-define-a-predefined-macro-in-x64-and-set-a-unique-value-for-m-ix86-fp-in-win32
#    include <immintrin.h>
#    define CV_AVX 1
#    if defined(_XCR_XFEATURE_ENABLED_MASK)
#      define __xgetbv() _xgetbv(_XCR_XFEATURE_ENABLED_MASK)
#    else
#      define __xgetbv() 0
#    endif
#  endif
#  if defined __AVX2__ || (defined _MSC_VER && _MSC_VER >= 1800 && 0)
#    include <immintrin.h>
#    define CV_AVX2 1
#    if defined __FMA__
#      define CV_FMA3 1
#    endif
#  endif
#endif

#if (defined WIN32 || defined _WIN32) && defined(_M_ARM)
# include <Intrin.h>
# include "arm_neon.h"
# define CV_NEON 1
# define CPU_HAS_NEON_FEATURE (true)
#elif defined(__ARM_NEON__) || (defined (__ARM_NEON) && defined(__aarch64__))
#  include <arm_neon.h>
#  define CV_NEON 1
#endif

#if defined __GNUC__ && defined __arm__ && (defined __ARM_PCS_VFP || defined __ARM_VFPV3__)
#  define CV_VFP 1
#endif

#endif // __CUDACC__

#ifndef CV_POPCNT
#define CV_POPCNT 0
#endif
#ifndef CV_MMX
#  define CV_MMX 0
#endif
#ifndef CV_SSE
#  define CV_SSE 0
#endif
#ifndef CV_SSE2
#  define CV_SSE2 0
#endif
#ifndef CV_SSE3
#  define CV_SSE3 0
#endif
#ifndef CV_SSSE3
#  define CV_SSSE3 0
#endif
#ifndef CV_SSE4_1
#  define CV_SSE4_1 0
#endif
#ifndef CV_SSE4_2
#  define CV_SSE4_2 0
#endif
#ifndef CV_AVX
#  define CV_AVX 0
#endif
#ifndef CV_AVX2
#  define CV_AVX2 0
#endif
#ifndef CV_FMA3
#  define CV_FMA3 0
#endif
#ifndef CV_AVX_512F
#  define CV_AVX_512F 0
#endif
#ifndef CV_AVX_512BW
#  define CV_AVX_512BW 0
#endif
#ifndef CV_AVX_512CD
#  define CV_AVX_512CD 0
#endif
#ifndef CV_AVX_512DQ
#  define CV_AVX_512DQ 0
#endif
#ifndef CV_AVX_512ER
#  define CV_AVX_512ER 0
#endif
#ifndef CV_AVX_512IFMA512
#  define CV_AVX_512IFMA512 0
#endif
#ifndef CV_AVX_512PF
#  define CV_AVX_512PF 0
#endif
#ifndef CV_AVX_512VBMI
#  define CV_AVX_512VBMI 0
#endif
#ifndef CV_AVX_512VL
#  define CV_AVX_512VL 0
#endif

#ifndef CV_NEON
#  define CV_NEON 0
#endif

#ifndef CV_VFP
#  define CV_VFP 0
#endif

/* primitive types */
/*
  schar  - signed 1 byte integer
  uchar  - unsigned 1 byte integer
  short  - signed 2 byte integer
  ushort - unsigned 2 byte integer
  int    - signed 4 byte integer
  uint   - unsigned 4 byte integer
  int64  - signed 8 byte integer
  uint64 - unsigned 8 byte integer
*/

#if !defined _MSC_VER && !defined __BORLANDC__
#  if defined __cplusplus && __cplusplus >= 201103L
#    include <cstdint>
     typedef std::uint32_t uint;
#  else
#    include <stdint.h>
     typedef uint32_t uint;
#  endif
#else
   typedef unsigned uint;
#endif

typedef signed char schar;

#ifndef __IPL_H__
   typedef unsigned char uchar;
   typedef unsigned short ushort;
#endif

#if defined _MSC_VER || defined __BORLANDC__
   typedef __int64 int64;
   typedef unsigned __int64 uint64;
#  define CV_BIG_INT(n)   n##I64
#  define CV_BIG_UINT(n)  n##UI64
#else
   typedef int64_t int64;
   typedef uint64_t uint64;
#  define CV_BIG_INT(n)   n##LL
#  define CV_BIG_UINT(n)  n##ULL
#endif

/* fundamental constants */
#define CV_PI   3.1415926535897932384626433832795
#define CV_2PI 6.283185307179586476925286766559
#define CV_LOG2 0.69314718055994530941723212145818

#endif //__OPENCV_HAL_H__
