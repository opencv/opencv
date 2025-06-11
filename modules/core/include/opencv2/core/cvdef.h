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

#include "opencv2/core/version.hpp"

//! @addtogroup core_utils
//! @{

#ifdef OPENCV_INCLUDE_PORT_FILE  // User-provided header file with custom platform configuration
#include OPENCV_INCLUDE_PORT_FILE
#endif

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

#define __CV_VA_NUM_ARGS_HELPER(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, N, ...) N
#define __CV_VA_NUM_ARGS(...) __CV_EXPAND(__CV_VA_NUM_ARGS_HELPER(__VA_ARGS__, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0))

#ifdef CV_Func
// keep current value (through OpenCV port file)
#elif defined __GNUC__ || (defined (__cpluscplus) && (__cpluscplus >= 201103))
#define CV_Func __func__
#elif defined __clang__ && (__clang_minor__ * 100 + __clang_major__ >= 305)
#define CV_Func __func__
#elif defined(__STDC_VERSION__) && (__STDC_VERSION >= 199901)
#define CV_Func __func__
#elif defined _MSC_VER
#define CV_Func __FUNCTION__
#elif defined(__INTEL_COMPILER) && (_INTEL_COMPILER >= 600)
#define CV_Func __FUNCTION__
#elif defined __IBMCPP__ && __IBMCPP__ >=500
#define CV_Func __FUNCTION__
#elif defined __BORLAND__ && (__BORLANDC__ >= 0x550)
#define CV_Func __FUNC__
#else
#define CV_Func "<unknown>"
#endif

//! @cond IGNORED

//////////////// static assert /////////////////
#define CVAUX_CONCAT_EXP(a, b) a##b
#define CVAUX_CONCAT(a, b) CVAUX_CONCAT_EXP(a,b)

#if defined(__clang__)
#  ifndef __has_extension
#    define __has_extension __has_feature /* compatibility, for older versions of clang */
#  endif
#  if __has_extension(cxx_static_assert)
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  elif __has_extension(c_static_assert)
#    define CV_StaticAssert(condition, reason)    _Static_assert((condition), reason " " #condition)
#  endif
#elif defined(__GNUC__)
#  if (defined(__GXX_EXPERIMENTAL_CXX0X__) || __cplusplus >= 201103L)
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#elif defined(_MSC_VER)
#  if _MSC_VER >= 1600 /* MSVC 10 */
#    define CV_StaticAssert(condition, reason)    static_assert((condition), reason " " #condition)
#  endif
#endif
#ifndef CV_StaticAssert
#  if !defined(__clang__) && defined(__GNUC__) && (__GNUC__*100 + __GNUC_MINOR__ > 302)
#    define CV_StaticAssert(condition, reason) ({ extern int __attribute__((error("CV_StaticAssert: " reason " " #condition))) CV_StaticAssert(); ((condition) ? 0 : CV_StaticAssert()); })
#  else
namespace cv {
     template <bool x> struct CV_StaticAssert_failed;
     template <> struct CV_StaticAssert_failed<true> { enum { val = 1 }; };
     template<int x> struct CV_StaticAssert_test {};
}
#    define CV_StaticAssert(condition, reason)\
       typedef cv::CV_StaticAssert_test< sizeof(cv::CV_StaticAssert_failed< static_cast<bool>(condition) >) > CVAUX_CONCAT(CV_StaticAssert_failed_at_, __LINE__)
#  endif
#endif

// Suppress warning "-Wdeprecated-declarations" / C4996
#if defined(_MSC_VER)
    #define CV_DO_PRAGMA(x) __pragma(x)
#elif defined(__GNUC__)
    #define CV_DO_PRAGMA(x) _Pragma (#x)
#else
    #define CV_DO_PRAGMA(x)
#endif

#ifdef _MSC_VER
#define CV_SUPPRESS_DEPRECATED_START \
    CV_DO_PRAGMA(warning(push)) \
    CV_DO_PRAGMA(warning(disable: 4996))
#define CV_SUPPRESS_DEPRECATED_END CV_DO_PRAGMA(warning(pop))
#elif defined (__clang__) || ((__GNUC__)  && (__GNUC__*100 + __GNUC_MINOR__ > 405))
#define CV_SUPPRESS_DEPRECATED_START \
    CV_DO_PRAGMA(GCC diagnostic push) \
    CV_DO_PRAGMA(GCC diagnostic ignored "-Wdeprecated-declarations")
#define CV_SUPPRESS_DEPRECATED_END CV_DO_PRAGMA(GCC diagnostic pop)
#else
#define CV_SUPPRESS_DEPRECATED_START
#define CV_SUPPRESS_DEPRECATED_END
#endif

#define CV_UNUSED(name) (void)name

//! @endcond

// undef problematic defines sometimes defined by system headers (windows.h in particular)
#undef small
#undef min
#undef max
#undef abs
#undef Complex

#if defined __cplusplus
#include <limits>
#else
#include <limits.h>
#endif

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

#if defined _WIN32
#  define CV_CDECL __cdecl
#  define CV_STDCALL __stdcall
#else
#  define CV_CDECL
#  define CV_STDCALL
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

#ifndef CV_ALWAYS_INLINE
#if defined(__GNUC__) && (__GNUC__ > 3 || (__GNUC__ == 3 && __GNUC_MINOR__ >= 1))
#define CV_ALWAYS_INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define CV_ALWAYS_INLINE __forceinline
#else
#define CV_ALWAYS_INLINE inline
#endif
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
#define CV_CPU_AVX_512VBMI2     22
#define CV_CPU_AVX_512VNNI      23
#define CV_CPU_AVX_512BITALG    24
#define CV_CPU_AVX_512VPOPCNTDQ 25
#define CV_CPU_AVX_5124VNNIW    26
#define CV_CPU_AVX_5124FMAPS    27

#define CV_CPU_NEON             100
#define CV_CPU_NEON_DOTPROD     101
#define CV_CPU_NEON_FP16        102
#define CV_CPU_NEON_BF16        103

#define CV_CPU_MSA              150

#define CV_CPU_RISCVV           170

#define CV_CPU_VSX              200
#define CV_CPU_VSX3             201

#define CV_CPU_RVV              210

#define CV_CPU_LSX              230
#define CV_CPU_LASX             231

// CPU features groups
#define CV_CPU_AVX512_SKX       256
#define CV_CPU_AVX512_COMMON    257
#define CV_CPU_AVX512_KNL       258
#define CV_CPU_AVX512_KNM       259
#define CV_CPU_AVX512_CNL       260
#define CV_CPU_AVX512_CLX       261
#define CV_CPU_AVX512_ICL       262

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
    CPU_AVX_512VBMI2    = 22,
    CPU_AVX_512VNNI     = 23,
    CPU_AVX_512BITALG   = 24,
    CPU_AVX_512VPOPCNTDQ= 25,
    CPU_AVX_5124VNNIW   = 26,
    CPU_AVX_5124FMAPS   = 27,

    CPU_NEON            = 100,
    CPU_NEON_DOTPROD    = 101,
    CPU_NEON_FP16       = 102,
    CPU_NEON_BF16       = 103,

    CPU_MSA             = 150,

    CPU_RISCVV          = 170,

    CPU_VSX             = 200,
    CPU_VSX3            = 201,

    CPU_RVV             = 210,

    CPU_LSX             = 230,
    CPU_LASX            = 231,

    CPU_AVX512_SKX      = 256, //!< Skylake-X with AVX-512F/CD/BW/DQ/VL
    CPU_AVX512_COMMON   = 257, //!< Common instructions AVX-512F/CD for all CPUs that support AVX-512
    CPU_AVX512_KNL      = 258, //!< Knights Landing with AVX-512F/CD/ER/PF
    CPU_AVX512_KNM      = 259, //!< Knights Mill with AVX-512F/CD/ER/PF/4FMAPS/4VNNIW/VPOPCNTDQ
    CPU_AVX512_CNL      = 260, //!< Cannon Lake with AVX-512F/CD/BW/DQ/VL/IFMA/VBMI
    CPU_AVX512_CLX      = 261, //!< Cascade Lake with AVX-512F/CD/BW/DQ/VL/VNNI
    CPU_AVX512_ICL      = 262, //!< Ice Lake with AVX-512F/CD/BW/DQ/VL/IFMA/VBMI/VNNI/VBMI2/BITALG/VPOPCNTDQ

    CPU_MAX_FEATURE     = 512  // see CV_HARDWARE_MAX_FEATURE
};


#include "cv_cpu_dispatch.h"

#if !defined(CV_STRONG_ALIGNMENT) && defined(__arm__) && !(defined(__aarch64__) || defined(_M_ARM64))
// int*, int64* should be propertly aligned pointers on ARMv7
#define CV_STRONG_ALIGNMENT 1
#endif
#if !defined(CV_STRONG_ALIGNMENT)
#define CV_STRONG_ALIGNMENT 0
#endif

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
    ushort u;
#if CV_FP16_TYPE
    __fp16 h;
#endif
}
Cv16suf;

typedef union Cv32suf
{
    int i;
    unsigned u;
    float f;
}
Cv32suf;

typedef union Cv64suf
{
    int64 i;
    uint64 u;
    double f;
}
Cv64suf;

#ifndef OPENCV_ABI_COMPATIBILITY
#define OPENCV_ABI_COMPATIBILITY 400
#endif

#ifdef __OPENCV_BUILD
#  define DISABLE_OPENCV_3_COMPATIBILITY
#  define OPENCV_DISABLE_DEPRECATED_COMPATIBILITY
#endif

#ifndef CV_EXPORTS
# if (defined _WIN32 || defined WINCE || defined __CYGWIN__) && defined(CVAPI_EXPORTS)
#   define CV_EXPORTS __declspec(dllexport)
# elif defined __GNUC__ && __GNUC__ >= 4 && (defined(CVAPI_EXPORTS) || defined(__APPLE__))
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

#ifndef CV_DEPRECATED_EXTERNAL
#  if defined(__OPENCV_BUILD)
#    define CV_DEPRECATED_EXTERNAL /* nothing */
#  else
#    define CV_DEPRECATED_EXTERNAL CV_DEPRECATED
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
#define CV_EXPORTS_W_PARAMS CV_EXPORTS
#define CV_IN_OUT
#define CV_OUT
#define CV_PROP
#define CV_PROP_RW
#define CV_ND // Indicates that input data should be parsed into Mat without channels
#define CV_WRAP
#define CV_WRAP_AS(synonym)
#define CV_WRAP_MAPPABLE(mappable)
#define CV_WRAP_PHANTOM(phantom_header)
#define CV_WRAP_DEFAULT(val)
/* Indicates that the function parameter has filesystem path semantic */
#define CV_WRAP_FILE_PATH

/****************************************************************************************\
*                                  Matrix type (Mat)                                     *
\****************************************************************************************/

#define CV_MAX_DIM              32
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
   0x28442211 = 0010 1000 0100 0100 0010 0010 0001 0001 ~ array of sizeof(arr_type_elem) */
#define CV_ELEM_SIZE1(type) ((0x28442211 >> CV_MAT_DEPTH(type)*4) & 15)

#define CV_ELEM_SIZE(type) (CV_MAT_CN(type)*CV_ELEM_SIZE1(type))

#ifndef MIN
#  define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#  define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif

/** min & max without jumps */
#define CV_IMIN(a, b)  ((a) ^ (((a)^(b)) & (((a) < (b)) - 1)))
#define CV_IMAX(a, b)  ((a) ^ (((a)^(b)) & (((a) > (b)) - 1)))
#define CV_SWAP(a,b,t) ((t) = (a), (a) = (b), (b) = (t))
#define CV_CMP(a,b)    (((a) > (b)) - ((a) < (b)))
#define CV_SIGN(a)     CV_CMP((a),0)

///////////////////////////////////////// Enum operators ///////////////////////////////////////

/**

Provides compatibility operators for both classical and C++11 enum classes,
as well as exposing the C++11 enum class members for backwards compatibility

@code
    // Provides operators required for flag enums
    CV_ENUM_FLAGS(AccessFlag)

    // Exposes the listed members of the enum class AccessFlag to the current namespace
    CV_ENUM_CLASS_EXPOSE(AccessFlag, ACCESS_READ [, ACCESS_WRITE [, ...] ]);
@endcode
*/

#define __CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST)                                              \
static const EnumType MEMBER_CONST = EnumType::MEMBER_CONST;                                          \

#define __CV_ENUM_CLASS_EXPOSE_2(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_1(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_3(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_2(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_4(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_3(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_5(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_4(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_6(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_5(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_7(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_6(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_8(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_7(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_CLASS_EXPOSE_9(EnumType, MEMBER_CONST, ...)                                         \
__CV_ENUM_CLASS_EXPOSE_1(EnumType, MEMBER_CONST);                                                     \
__CV_EXPAND(__CV_ENUM_CLASS_EXPOSE_8(EnumType, __VA_ARGS__));                                         \

#define __CV_ENUM_FLAGS_LOGICAL_NOT(EnumType)                                                         \
static inline bool operator!(const EnumType& val)                                                     \
{                                                                                                     \
    typedef std::underlying_type<EnumType>::type UnderlyingType;                                      \
    return !static_cast<UnderlyingType>(val);                                                         \
}                                                                                                     \

#define __CV_ENUM_FLAGS_LOGICAL_NOT_EQ(Arg1Type, Arg2Type)                                            \
static inline bool operator!=(const Arg1Type& a, const Arg2Type& b)                                   \
{                                                                                                     \
    return static_cast<int>(a) != static_cast<int>(b);                                                \
}                                                                                                     \

#define __CV_ENUM_FLAGS_LOGICAL_EQ(Arg1Type, Arg2Type)                                                \
static inline bool operator==(const Arg1Type& a, const Arg2Type& b)                                   \
{                                                                                                     \
    return static_cast<int>(a) == static_cast<int>(b);                                                \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_NOT(EnumType)                                                         \
static inline EnumType operator~(const EnumType& val)                                                 \
{                                                                                                     \
    typedef std::underlying_type<EnumType>::type UnderlyingType;                                      \
    return static_cast<EnumType>(~static_cast<UnderlyingType>(val));                                  \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_OR(EnumType, Arg1Type, Arg2Type)                                      \
static inline EnumType operator|(const Arg1Type& a, const Arg2Type& b)                                \
{                                                                                                     \
    typedef std::underlying_type<EnumType>::type UnderlyingType;                                      \
    return static_cast<EnumType>(static_cast<UnderlyingType>(a) | static_cast<UnderlyingType>(b));    \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_AND(EnumType, Arg1Type, Arg2Type)                                     \
static inline EnumType operator&(const Arg1Type& a, const Arg2Type& b)                                \
{                                                                                                     \
    typedef std::underlying_type<EnumType>::type UnderlyingType;                                      \
    return static_cast<EnumType>(static_cast<UnderlyingType>(a) & static_cast<UnderlyingType>(b));    \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_XOR(EnumType, Arg1Type, Arg2Type)                                     \
static inline EnumType operator^(const Arg1Type& a, const Arg2Type& b)                                \
{                                                                                                     \
    typedef std::underlying_type<EnumType>::type UnderlyingType;                                      \
    return static_cast<EnumType>(static_cast<UnderlyingType>(a) ^ static_cast<UnderlyingType>(b));    \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_OR_EQ(EnumType, Arg1Type)                                             \
static inline EnumType& operator|=(EnumType& _this, const Arg1Type& val)                              \
{                                                                                                     \
    _this = static_cast<EnumType>(static_cast<int>(_this) | static_cast<int>(val));                   \
    return _this;                                                                                     \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_AND_EQ(EnumType, Arg1Type)                                            \
static inline EnumType& operator&=(EnumType& _this, const Arg1Type& val)                              \
{                                                                                                     \
    _this = static_cast<EnumType>(static_cast<int>(_this) & static_cast<int>(val));                   \
    return _this;                                                                                     \
}                                                                                                     \

#define __CV_ENUM_FLAGS_BITWISE_XOR_EQ(EnumType, Arg1Type)                                            \
static inline EnumType& operator^=(EnumType& _this, const Arg1Type& val)                              \
{                                                                                                     \
    _this = static_cast<EnumType>(static_cast<int>(_this) ^ static_cast<int>(val));                   \
    return _this;                                                                                     \
}                                                                                                     \

#define CV_ENUM_CLASS_EXPOSE(EnumType, ...)                                                           \
__CV_EXPAND(__CV_CAT(__CV_ENUM_CLASS_EXPOSE_, __CV_VA_NUM_ARGS(__VA_ARGS__))(EnumType, __VA_ARGS__)); \

#define CV_ENUM_FLAGS(EnumType)                                                                       \
__CV_ENUM_FLAGS_LOGICAL_NOT      (EnumType)                                                           \
__CV_ENUM_FLAGS_LOGICAL_EQ       (EnumType, int)                                                      \
__CV_ENUM_FLAGS_LOGICAL_NOT_EQ   (EnumType, int)                                                      \
                                                                                                      \
__CV_ENUM_FLAGS_BITWISE_NOT      (EnumType)                                                           \
__CV_ENUM_FLAGS_BITWISE_OR       (EnumType, EnumType, EnumType)                                       \
__CV_ENUM_FLAGS_BITWISE_AND      (EnumType, EnumType, EnumType)                                       \
__CV_ENUM_FLAGS_BITWISE_XOR      (EnumType, EnumType, EnumType)                                       \
                                                                                                      \
__CV_ENUM_FLAGS_BITWISE_OR_EQ    (EnumType, EnumType)                                                 \
__CV_ENUM_FLAGS_BITWISE_AND_EQ   (EnumType, EnumType)                                                 \
__CV_ENUM_FLAGS_BITWISE_XOR_EQ   (EnumType, EnumType)                                                 \

/****************************************************************************************\
*                                    static analysys                                     *
\****************************************************************************************/

// In practice, some macro are not processed correctly (noreturn is not detected).
// We need to use simplified definition for them.
#ifndef CV_STATIC_ANALYSIS
# if defined(__KLOCWORK__) || defined(__clang_analyzer__) || defined(__COVERITY__)
#   define CV_STATIC_ANALYSIS 1
# endif
#else
# if defined(CV_STATIC_ANALYSIS) && !(__CV_CAT(1, CV_STATIC_ANALYSIS) == 1)  // defined and not empty
#   if 0 == CV_STATIC_ANALYSIS
#     undef CV_STATIC_ANALYSIS
#   endif
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
#  if defined __clang__ && __clang_major__ >= 3 && !defined __ANDROID__ && !defined __EMSCRIPTEN__ && !defined(__CUDACC__)  && !defined __INTEL_COMPILER
#    ifdef __ATOMIC_ACQ_REL
#      define CV_XADD(addr, delta) __c11_atomic_fetch_add((_Atomic(int)*)(addr), delta, __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) __atomic_fetch_add((_Atomic(int)*)(addr), delta, 4)
#    endif
#  else
#    if defined __ATOMIC_ACQ_REL && !defined __clang__
       // version for gcc >= 4.7
#      define CV_XADD(addr, delta) __atomic_fetch_add((unsigned*)(addr), (unsigned)(delta), __ATOMIC_ACQ_REL)
#    else
#      define CV_XADD(addr, delta) __sync_fetch_and_add((unsigned*)(addr), (unsigned)(delta))
#    endif
#  endif
#elif defined _MSC_VER && !defined RC_INVOKED
#  include <intrin.h>
#  define CV_XADD(addr, delta) (int)_InterlockedExchangeAdd((long volatile*)addr, delta)
#else
  #ifdef OPENCV_FORCE_UNSAFE_XADD
    CV_INLINE int CV_XADD(int* addr, int delta) { int tmp = *addr; *addr += delta; return tmp; }
  #else
    #error "OpenCV: can't define safe CV_XADD macro for current platform (unsupported). Define CV_XADD macro through custom port header (see OPENCV_INCLUDE_PORT_FILE)"
  #endif
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
*                       CV_NODISCARD_STD attribute (C++17)                               *
* encourages the compiler to issue a warning if the return value is discarded            *
\****************************************************************************************/
#ifndef CV_NODISCARD_STD
#  ifndef __has_cpp_attribute
//   workaround preprocessor non-compliance https://reviews.llvm.org/D57851
#    define __has_cpp_attribute(__x) 0
#  endif
#  if __has_cpp_attribute(nodiscard)
#    if defined(__NVCC__) && __CUDACC_VER_MAJOR__ < 12
#       define CV_NODISCARD_STD
#    else
#       define CV_NODISCARD_STD [[nodiscard]]
#    endif
#  elif __cplusplus >= 201703L
//   available when compiler is C++17 compliant
#    define CV_NODISCARD_STD [[nodiscard]]
#  elif defined(__INTEL_COMPILER)
     // see above, available when C++17 is enabled
#  elif defined(_MSC_VER) && _MSC_VER >= 1911 && _MSVC_LANG >= 201703L
//   available with VS2017 v15.3+ with /std:c++17 or higher; works on functions and classes
#    define CV_NODISCARD_STD [[nodiscard]]
#  elif defined(__GNUC__) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 700) && (__cplusplus >= 201103L)
//   available with GCC 7.0+; works on functions, works or silently fails on classes
#    define CV_NODISCARD_STD [[nodiscard]]
#  elif defined(__GNUC__) && (((__GNUC__ * 100) + __GNUC_MINOR__) >= 408) && (__cplusplus >= 201103L)
//   available with GCC 4.8+ but it usually does nothing and can fail noisily -- therefore not used
//   define CV_NODISCARD_STD [[gnu::warn_unused_result]]
#  endif
#endif
#ifndef CV_NODISCARD_STD
#  define CV_NODISCARD_STD /* nothing by default */
#endif


/****************************************************************************************\
*                                    C++ 11                                              *
\****************************************************************************************/
#ifdef __cplusplus
// MSVC was stuck at __cplusplus == 199711L for a long time, even where it supports C++11,
// so check _MSC_VER instead. See:
// <https://devblogs.microsoft.com/cppblog/msvc-now-correctly-reports-__cplusplus>
#  if defined(_MSC_VER)
#    if _MSC_VER < 1800
#      error "OpenCV 4.x+ requires enabled C++11 support"
#    endif
#  elif __cplusplus < 201103L
#    error "OpenCV 4.x+ requires enabled C++11 support"
#  endif
#endif

#ifndef CV_CXX11
#  define CV_CXX11 1
#endif

#ifndef CV_OVERRIDE
#  define CV_OVERRIDE override
#endif

#ifndef CV_FINAL
#  define CV_FINAL final
#endif

#ifndef CV_NOEXCEPT
#  define CV_NOEXCEPT noexcept
#endif

#ifndef CV_CONSTEXPR
#  define CV_CONSTEXPR constexpr
#endif

// Integer types portability
#ifdef __cplusplus
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
#else // pure C
#include <stdint.h>
#endif

#ifdef __cplusplus
namespace cv
{

class hfloat
{
public:
#if CV_FP16_TYPE

    hfloat() : h(0) {}
    explicit hfloat(float x) { h = (__fp16)x; }
    operator float() const { return (float)h; }
protected:
    __fp16 h;

#else
    hfloat() : w(0) {}
    explicit hfloat(float x)
    {
    #if CV_FP16 && CV_AVX2
        __m128 v = _mm_load_ss(&x);
        w = (ushort)_mm_cvtsi128_si32(_mm_cvtps_ph(v, 0));
    #else
        Cv32suf in;
        in.f = x;
        unsigned sign = in.u & 0x80000000;
        in.u ^= sign;

        if( in.u >= 0x47800000 )
            w = (ushort)(in.u > 0x7f800000 ? 0x7e00 : 0x7c00);
        else
        {
            if (in.u < 0x38800000)
            {
                in.f += 0.5f;
                w = (ushort)(in.u - 0x3f000000);
            }
            else
            {
                unsigned t = in.u + 0xc8000fff;
                w = (ushort)((t + ((in.u >> 13) & 1)) >> 13);
            }
        }

        w = (ushort)(w | (sign >> 16));
    #endif
    }

    operator float() const
    {
    #if CV_FP16 && CV_AVX2
        float f;
        _mm_store_ss(&f, _mm_cvtph_ps(_mm_cvtsi32_si128(w)));
        return f;
    #else
        Cv32suf out;

        unsigned t = ((w & 0x7fff) << 13) + 0x38000000;
        unsigned sign = (w & 0x8000) << 16;
        unsigned e = w & 0x7c00;

        out.u = t + (1 << 23);
        out.u = (e >= 0x7c00 ? t + 0x38000000 :
                 e == 0 ? (static_cast<void>(out.f -= 6.103515625e-05f), out.u) : t) | sign;
        return out.f;
    #endif
    }

protected:
    ushort w;

#endif
};

inline hfloat hfloatFromBits(ushort w) {
#if CV_FP16_TYPE
    Cv16suf u;
    u.u = w;
    hfloat res(float(u.h));
    return res;
#else
    Cv32suf out;

    unsigned t = ((w & 0x7fff) << 13) + 0x38000000;
    unsigned sign = (w & 0x8000) << 16;
    unsigned e = w & 0x7c00;

    out.u = t + (1 << 23);
    out.u = (e >= 0x7c00 ? t + 0x38000000 :
            e == 0 ? (static_cast<void>(out.f -= 6.103515625e-05f), out.u) : t) | sign;
    hfloat res(out.f);
    return res;
#endif
}

#if !defined(__OPENCV_BUILD) && !(defined __STDCPP_FLOAT16_T__) && !(defined __ARM_NEON)
typedef hfloat float16_t;
#endif

}
#endif

/** @brief Constructs the 'fourcc' code, used in video codecs and many other places.
    Simply call it with 4 chars like `CV_FOURCC('I', 'Y', 'U', 'V')`
*/
CV_INLINE int CV_FOURCC(char c1, char c2, char c3, char c4)
{
    return (c1 & 255) + ((c2 & 255) << 8) + ((c3 & 255) << 16) + ((c4 & 255) << 24);
}

//! Macro to construct the fourcc code of the codec. Same as CV_FOURCC()
#define CV_FOURCC_MACRO(c1, c2, c3, c4) (((c1) & 255) + (((c2) & 255) << 8) + (((c3) & 255) << 16) + (((c4) & 255) << 24))

//! @}

#ifndef __cplusplus
#include "opencv2/core/fast_math.hpp" // define cvRound(double)
#endif

#endif // OPENCV_CORE_CVDEF_H
