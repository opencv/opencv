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

#ifndef OPENCV_CORE_CVDEF_H
#define OPENCV_CORE_CVDEF_H

//! @addtogroup core_utils
//! @{

#if !defined CV_DOXYGEN && !defined CV_IGNORE_DEBUG_BUILD_GUARD
#if (defined(_MSC_VER) && (defined(DEBUG) || defined(_DEBUG))) || \
    (defined(_GLIBCXX_DEBUG) || defined(_GLIBCXX_DEBUG_PEDANTIC))
// Guard to prevent using of binary incompatible binaries / runtimes
// https://github.com/opencv/opencv/pull/9161
#define CV__DEBUG_NS_BEGIN namespace debug_build_guard {
#define CV__DEBUG_NS_END }
namespace cv { namespace debug_build_guard { } using namespace debug_build_guard; }
#endif
#endif

#ifndef CV__DEBUG_NS_BEGIN
#define CV__DEBUG_NS_BEGIN
#define CV__DEBUG_NS_END
#endif


#ifdef __OPENCV_BUILD
#include "cvconfig.h"
#endif

#ifndef __CV_EXPAND
#define __CV_EXPAND(x) x
#endif

#ifndef __CV_CAT
#define __CV_CAT__(x, y) x ## y
#define __CV_CAT_(x, y) __CV_CAT__(x, y)
#define __CV_CAT(x, y) __CV_CAT_(x, y)
#endif


// undef problematic defines sometimes defined by system headers (windows.h in particular)
#undef small
#undef min
#undef max
#undef abs
#undef Complex

#include <limits.h>
#include "opencv2/core/hal/interface.h"

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

#ifndef CV_INLINE
#  if defined __cplusplus
#    define CV_INLINE static inline
#  elif defined _MSC_VER
#    define CV_INLINE __inline
#  else
#    define CV_INLINE static
#  endif
#endif

#if defined CV_DISABLE_OPTIMIZATION || (defined CV_ICC && !defined CV_ENABLE_UNROLLED)
#  define CV_ENABLE_UNROLLED 0
#else
#  define CV_ENABLE_UNROLLED 1
#endif

#ifdef __GNUC__
#  define CV_DECL_ALIGNED(x) __attribute__ ((aligned (x)))
#elif defined _MSC_VER
#  define CV_DECL_ALIGNED(x) __declspec(align(x))
#else
#  define CV_DECL_ALIGNED(x)
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
#define CV_CPU_FP16             9
#define CV_CPU_AVX              10
#define CV_CPU_AVX2             11
#define CV_CPU_FMA3             12

#define CV_CPU_AVX_512F         13
#define CV_CPU_AVX_512BW        14
#define CV_CPU_AVX_512CD        15
#define CV_CPU_AVX_512DQ        16
#define CV_CPU_AVX_512ER        17
#define CV_CPU_AVX_512IFMA512   18 // deprecated
#define CV_CPU_AVX_512IFMA      18
#define CV_CPU_AVX_512PF        19
#define CV_CPU_AVX_512VBMI      20
#define CV_CPU_AVX_512VL        21

#define CV_CPU_NEON   100

#define CV_CPU_VSX 200

// CPU features groups
#define CV_CPU_AVX512_SKX       256

// when adding to this list remember to update the following enum
#define CV_HARDWARE_MAX_FEATURE 512

/** @brief Available CPU features.
*/
enum CpuFeatures {
    CPU_MMX             = 1,
    CPU_SSE             = 2,
    CPU_SSE2            = 3,
    CPU_SSE3            = 4,
    CPU_SSSE3           = 5,
    CPU_SSE4_1          = 6,
    CPU_SSE4_2          = 7,
    CPU_POPCNT          = 8,
    CPU_FP16            = 9,
    CPU_AVX             = 10,
    CPU_AVX2            = 11,
    CPU_FMA3            = 12,

    CPU_AVX_512F        = 13,
    CPU_AVX_512BW       = 14,
    CPU_AVX_512CD       = 15,
    CPU_AVX_512DQ       = 16,
    CPU_AVX_512ER       = 17,
    CPU_AVX_512IFMA512  = 18, // deprecated
    CPU_AVX_512IFMA     = 18,
    CPU_AVX_512PF       = 19,
    CPU_AVX_512VBMI     = 20,
    CPU_AVX_512VL       = 21,

    CPU_NEON            = 100,

    CPU_VSX             = 200,

    CPU_AVX512_SKX      = 256, //!< Skylake-X with AVX-512F/CD/BW/DQ/VL

    CPU_MAX_FEATURE     = 512  // see CV_HARDWARE_MAX_FEATURE
};


#include "cv_cpu_dispatch.h"


/* fundamental constants */
#define CV_PI   3.1415926535897932384626433832795
#define CV_2PI  6.283185307179586476925286766559
#define CV_LOG2 0.69314718055994530941723212145818

#if defined __ARM_FP16_FORMAT_IEEE \
    && !defined __CUDACC__
#  define CV_FP16_TYPE 1
#else
#  define CV_FP16_TYPE 0
#endif

typedef union Cv16suf
{
    short i;
#if CV_FP16_TYPE
    __fp16 h;
#endif
    struct _fp16Format
    {
        unsigned int significand : 10;
        unsigned int exponent    : 5;
        unsigned int sign        : 1;
    } fmt;
}
Cv16suf;

typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
    struct _fp32Format
    {
        unsigned int significand : 23;
        unsigned int exponent    : 8;
        unsigned int sign        : 1;
    } fmt;
}
Cv32suf;

typedef union Cv64suf
{
    int64 i;
    uint64 u;
    double f;
}
Cv64suf;

#define OPENCV_ABI_COMPATIBILITY 300

#ifdef __OPENCV_BUILD
#  define DISABLE_OPENCV_24_COMPATIBILITY
#endif

#ifdef CVAPI_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__)
#   define CV_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4
#   define CV_EXPORTS __attribute__ ((visibility ("default")))
# endif
#endif

#ifndef CV_EXPORTS
# define CV_EXPORTS
#endif

#ifdef _MSC_VER
#   define CV_EXPORTS_TEMPLATE
#else
#   define CV_EXPORTS_TEMPLATE CV_EXPORTS
#endif

#ifndef CV_DEPRECATED
#  if defined(__GNUC__)
#    define CV_DEPRECATED __attribute__ ((deprecated))
#  elif defined(_MSC_VER)
#    define CV_DEPRECATED __declspec(deprecated)
#  else
#    define CV_DEPRECATED
#  endif
#endif

#ifndef CV_EXTERN_C
#  ifdef __cplusplus
#    define CV_EXTERN_C extern "C"
#  else
#    define CV_EXTERN_C
#  endif
#endif

/* special informative macros for wrapper generators */
#define CV_EXPORTS_W CV_EXPORTS
#define CV_EXPORTS_W_SIMPLE CV_EXPORTS
#define CV_EXPORTS_AS(synonym) CV_EXPORTS
#define CV_EXPORTS_W_MAP CV_EXPORTS
#define CV_IN_OUT
#define CV_OUT
#define CV_PROP
#define CV_PROP_RW
#define CV_WRAP
#define CV_WRAP_AS(synonym)

/****************************************************************************************\
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

#define CV_MAT_CN_MASK          ((CV_CN_MAX - 1) << CV_CN_SHIFT)
#define CV_MAT_CN(flags)        ((((flags) & CV_MAT_CN_MASK) >> CV_CN_SHIFT) + 1)
#define CV_MAT_TYPE_MASK        (CV_DEPTH_MAX*CV_CN_MAX - 1)
#define CV_MAT_TYPE(flags)      ((flags) & CV_MAT_TYPE_MASK)
#define CV_MAT_CONT_FLAG_SHIFT  14
#define CV_MAT_CONT_FLAG        (1 << CV_MAT_CONT_FLAG_SHIFT)
#define CV_IS_MAT_CONT(flags)   ((flags) & CV_MAT_CONT_FLAG)
#define CV_IS_CONT_MAT          CV_IS_MAT_CONT
#define CV_SUBMAT_FLAG_SHIFT    15
#define CV_SUBMAT_FLAG          (1 << CV_SUBMAT_FLAG_SHIFT)
#define CV_IS_SUBMAT(flags)     ((flags) & CV_MAT_SUBMAT_FLAG)

/** Size of each channel item,
   0x8442211 = 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define CV_ELEM_SIZE1(type) \
    ((((sizeof(size_t)<<28)|0x8442211) >> CV_MAT_DEPTH(type)*4) & 15)

/** 0x3a50 = 11 10 10 01 01 00 00 ~ array of log2(sizeof(arr_type_elem)) */
#define CV_ELEM_SIZE(type) \
    (CV_MAT_CN(type) << ((((sizeof(size_t)/4+1)*16384|0x3a50) >> CV_MAT_DEPTH(type)*2) & 3))

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

/****************************************************************************************\
*                                    static analysys                                     *
\****************************************************************************************/

// In practice, some macro are not processed correctly (noreturn is not detected).
// We need to use simplified definition for them.
#ifndef CV_STATIC_ANALYSIS
# if defined(__KLOCWORK__) || defined(__clang_analyzer__) || defined(__COVERITY__)
#   define CV_STATIC_ANALYSIS
# endif
#endif

/****************************************************************************************\
*                                    Thread sanitizer                                    *
\****************************************************************************************/
#ifndef CV_THREAD_SANITIZER
# if defined(__has_feature)
#   if __has_feature(thread_sanitizer)
#     define CV_THREAD_SANITIZER
#   endif
# endif
#endif

/****************************************************************************************\
*          exchange-add operation for atomic operations on reference counters            *
\****************************************************************************************/

#ifdef CV_XADD
  // allow to use user-defined macro
#elif defined __GNUC__ || defined __clang__
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)
#    ifdef __ATOMIC_ACQ_REL
#      define CV_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
       // version for gcc >= 4.7
#      define CV_XADD(addr, delta) (int)__atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) (int)__sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define CV_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
   CV_INLINE CV_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
#endif


/****************************************************************************************\
*                                  CV_NORETURN attribute                                 *
\****************************************************************************************/

#ifndef CV_NORETURN
#  if defined(__GNUC__)
#    define CV_NORETURN __attribute__((__noreturn__))
#  elif defined(_MSC_VER) && (_MSC_VER >= 1300)
#    define CV_NORETURN __declspec(noreturn)
#  else
#    define CV_NORETURN /* nothing by default */
#  endif
#endif


/****************************************************************************************\
*                                    C++ 11                                              *
\****************************************************************************************/
#ifndef CV_CXX11
#  if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1800)
#    define CV_CXX11 1
#  endif
#else
#  if CV_CXX11 == 0
#    undef CV_CXX11
#  endif
#endif


/****************************************************************************************\
*                                    C++ Move semantics                                  *
\****************************************************************************************/

#ifndef CV_CXX_MOVE_SEMANTICS
#  if __cplusplus >= 201103L || defined(__GXX_EXPERIMENTAL_CXX0X__) || (defined(_MSC_VER) && _MSC_VER >= 1600)
#    define CV_CXX_MOVE_SEMANTICS 1
#  elif defined(__clang)
#    if __has_feature(cxx_rvalue_references)
#      define CV_CXX_MOVE_SEMANTICS 1
#    endif
#  endif
#else
#  if CV_CXX_MOVE_SEMANTICS == 0
#    undef CV_CXX_MOVE_SEMANTICS
#  endif
#endif

/****************************************************************************************\
*                                    C++11 std::array                                    *
\****************************************************************************************/

#ifndef CV_CXX_STD_ARRAY
#  if __cplusplus >= 201103L || (defined(_MSC_VER) && _MSC_VER >= 1900/*MSVS 2015*/)
#    define CV_CXX_STD_ARRAY 1
#    include <array>
#  endif
#else
#  if CV_CXX_STD_ARRAY == 0
#    undef CV_CXX_STD_ARRAY
#  endif
#endif


/****************************************************************************************\
*                                 C++11 override / final                                 *
\****************************************************************************************/

#ifndef CV_OVERRIDE
#  ifdef CV_CXX11
#    define CV_OVERRIDE override
#  endif
#endif
#ifndef CV_OVERRIDE
#  define CV_OVERRIDE
#endif

#ifndef CV_FINAL
#  ifdef CV_CXX11
#    define CV_FINAL final
#  endif
#endif
#ifndef CV_FINAL
#  define CV_FINAL
#endif



// Integer types portatibility
#ifdef OPENCV_STDINT_HEADER
#include OPENCV_STDINT_HEADER
#else
#if defined(_MSC_VER) && _MSC_VER < 1600 /* MSVS 2010 */
namespace cv {
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed short int16_t;
typedef unsigned short uint16_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;
typedef signed __int64 int64_t;
typedef unsigned __int64 uint64_t;
}
#elif defined(_MSC_VER) || __cplusplus >= 201103L
#include <cstdint>
namespace cv {
using std::int8_t;
using std::uint8_t;
using std::int16_t;
using std::uint16_t;
using std::int32_t;
using std::uint32_t;
using std::int64_t;
using std::uint64_t;
}
#else
#include <stdint.h>
namespace cv {
typedef ::int8_t int8_t;
typedef ::uint8_t uint8_t;
typedef ::int16_t int16_t;
typedef ::uint16_t uint16_t;
typedef ::int32_t int32_t;
typedef ::uint32_t uint32_t;
typedef ::int64_t int64_t;
typedef ::uint64_t uint64_t;
}
#endif
#endif


//! @}

#endif // OPENCV_CORE_CVDEF_H
