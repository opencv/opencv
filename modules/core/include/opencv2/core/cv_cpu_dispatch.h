// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if defined __OPENCV_BUILD \

#include "cv_cpu_config.h"
#include "cv_cpu_helper.h"

#ifdef CV_CPU_DISPATCH_MODE
#define CV_CPU_OPTIMIZATION_NAMESPACE __CV_CAT(opt_, CV_CPU_DISPATCH_MODE)
#define CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN namespace __CV_CAT(opt_, CV_CPU_DISPATCH_MODE) {
#define CV_CPU_OPTIMIZATION_NAMESPACE_END }
#else
#define CV_CPU_OPTIMIZATION_NAMESPACE cpu_baseline
#define CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN namespace cpu_baseline {
#define CV_CPU_OPTIMIZATION_NAMESPACE_END }
#define CV_CPU_BASELINE_MODE 1
#endif


#define __CV_CPU_DISPATCH_CHAIN_END(fn, args, mode, ...)  /* done */
#define __CV_CPU_DISPATCH(fn, args, mode, ...) __CV_EXPAND(__CV_CPU_DISPATCH_CHAIN_ ## mode(fn, args, __VA_ARGS__))
#define __CV_CPU_DISPATCH_EXPAND(fn, args, ...) __CV_EXPAND(__CV_CPU_DISPATCH(fn, args, __VA_ARGS__))
#define CV_CPU_DISPATCH(fn, args, ...) __CV_CPU_DISPATCH_EXPAND(fn, args, __VA_ARGS__, END) // expand macros


#if defined CV_ENABLE_INTRINSICS \
    && !defined CV_DISABLE_OPTIMIZATION \
    && !defined __CUDACC__ /* do not include SSE/AVX/NEON headers for NVCC compiler */ \

#ifdef CV_CPU_COMPILE_SSE2
#  include <emmintrin.h>
#  define CV_MMX 1
#  define CV_SSE 1
#  define CV_SSE2 1
#endif
#ifdef CV_CPU_COMPILE_SSE3
#  include <pmmintrin.h>
#  define CV_SSE3 1
#endif
#ifdef CV_CPU_COMPILE_SSSE3
#  include <tmmintrin.h>
#  define CV_SSSE3 1
#endif
#ifdef CV_CPU_COMPILE_SSE4_1
#  include <smmintrin.h>
#  define CV_SSE4_1 1
#endif
#ifdef CV_CPU_COMPILE_SSE4_2
#  include <nmmintrin.h>
#  define CV_SSE4_2 1
#endif
#ifdef CV_CPU_COMPILE_POPCNT
#  ifdef _MSC_VER
#    include <nmmintrin.h>
#    if defined(_M_X64)
#      define CV_POPCNT_U64 (int)_mm_popcnt_u64
#    endif
#    define CV_POPCNT_U32 _mm_popcnt_u32
#  else
#    include <popcntintrin.h>
#    if defined(__x86_64__)
#      define CV_POPCNT_U64 __builtin_popcountll
#    endif
#    define CV_POPCNT_U32 __builtin_popcount
#  endif
#  define CV_POPCNT 1
#endif
#ifdef CV_CPU_COMPILE_AVX
#  include <immintrin.h>
#  define CV_AVX 1
#endif
#ifdef CV_CPU_COMPILE_FP16
#  if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
#    include <arm_neon.h>
#  else
#    include <immintrin.h>
#  endif
#  define CV_FP16 1
#endif
#ifdef CV_CPU_COMPILE_AVX2
#  include <immintrin.h>
#  define CV_AVX2 1
#endif
#ifdef CV_CPU_COMPILE_AVX_512F
#  include <immintrin.h>
#  define CV_AVX_512F 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_COMMON
#  define CV_AVX512_COMMON 1
#  define CV_AVX_512CD 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_KNL
#  define CV_AVX512_KNL 1
#  define CV_AVX_512ER 1
#  define CV_AVX_512PF 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_KNM
#  define CV_AVX512_KNM 1
#  define CV_AVX_5124FMAPS 1
#  define CV_AVX_5124VNNIW 1
#  define CV_AVX_512VPOPCNTDQ 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_SKX
#  define CV_AVX512_SKX 1
#  define CV_AVX_512VL 1
#  define CV_AVX_512BW 1
#  define CV_AVX_512DQ 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_CNL
#  define CV_AVX512_CNL 1
#  define CV_AVX_512IFMA 1
#  define CV_AVX_512VBMI 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_CLX
#  define CV_AVX512_CLX 1
#  define CV_AVX_512VNNI 1
#endif
#ifdef CV_CPU_COMPILE_AVX512_ICL
#  define CV_AVX512_ICL 1
#  undef CV_AVX_512IFMA
#  define CV_AVX_512IFMA 1
#  undef CV_AVX_512VBMI
#  define CV_AVX_512VBMI 1
#  undef CV_AVX_512VNNI
#  define CV_AVX_512VNNI 1
#  define CV_AVX_512VBMI2 1
#  define CV_AVX_512BITALG 1
#  define CV_AVX_512VPOPCNTDQ 1
#endif
#ifdef CV_CPU_COMPILE_FMA3
#  define CV_FMA3 1
#endif

#if defined _WIN32 && (defined(_M_ARM) || defined(_M_ARM64)) && (defined(CV_CPU_COMPILE_NEON) || !defined(_MSC_VER))
# include <Intrin.h>
# include <arm_neon.h>
# define CV_NEON 1
#elif defined(__ARM_NEON__) || (defined (__ARM_NEON) && defined(__aarch64__))
#  include <arm_neon.h>
#  define CV_NEON 1
#endif

#if defined(__riscv) && defined(__riscv_vector) && defined(__riscv_vector_071)
# include<riscv-vector.h>
# define CV_RVV071 1
#endif

#if defined(__ARM_NEON__) || defined(__aarch64__)
#  include <arm_neon.h>
#endif

#ifdef CV_CPU_COMPILE_VSX
#  include <altivec.h>
#  undef vector
#  undef pixel
#  undef bool
#  define CV_VSX 1
#endif

#ifdef CV_CPU_COMPILE_VSX3
#  define CV_VSX3 1
#endif

#ifdef CV_CPU_COMPILE_MSA
#  include "hal/msa_macros.h"
#  define CV_MSA 1
#endif

#ifdef __EMSCRIPTEN__
#  define CV_WASM_SIMD 1
#  include <wasm_simd128.h>
#endif

#if defined CV_CPU_COMPILE_RVV
#  define CV_RVV 1
#  include <riscv_vector.h>
#endif

#endif // CV_ENABLE_INTRINSICS && !CV_DISABLE_OPTIMIZATION && !__CUDACC__

#if defined CV_CPU_COMPILE_AVX && !defined CV_CPU_BASELINE_COMPILE_AVX
struct VZeroUpperGuard {
#ifdef __GNUC__
    __attribute__((always_inline))
#endif
    inline VZeroUpperGuard() { _mm256_zeroupper(); }
#ifdef __GNUC__
    __attribute__((always_inline))
#endif
    inline ~VZeroUpperGuard() { _mm256_zeroupper(); }
};
#define __CV_AVX_GUARD VZeroUpperGuard __vzeroupper_guard; CV_UNUSED(__vzeroupper_guard);
#endif

#ifdef __CV_AVX_GUARD
#define CV_AVX_GUARD __CV_AVX_GUARD
#else
#define CV_AVX_GUARD
#endif

#endif // __OPENCV_BUILD



#if !defined __OPENCV_BUILD /* Compatibility code */ \
    && !defined __CUDACC__ /* do not include SSE/AVX/NEON headers for NVCC compiler */
#if defined __SSE2__ || defined _M_X64 || (defined _M_IX86_FP && _M_IX86_FP >= 2)
#  include <emmintrin.h>
#  define CV_MMX 1
#  define CV_SSE 1
#  define CV_SSE2 1
#elif defined _WIN32 && (defined(_M_ARM) || defined(_M_ARM64)) && (defined(CV_CPU_COMPILE_NEON) || !defined(_MSC_VER))
# include <Intrin.h>
# include <arm_neon.h>
# define CV_NEON 1
#elif defined(__ARM_NEON__) || (defined (__ARM_NEON) && defined(__aarch64__))
#  include <arm_neon.h>
#  define CV_NEON 1
#elif defined(__VSX__) && defined(__PPC64__) && defined(__LITTLE_ENDIAN__)
#  include <altivec.h>
#  undef vector
#  undef pixel
#  undef bool
#  define CV_VSX 1
#endif

#ifdef __F16C__
#  include <immintrin.h>
#  define CV_FP16 1
#endif

#endif // !__OPENCV_BUILD && !__CUDACC (Compatibility code)



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
#ifndef CV_POPCNT
#  define CV_POPCNT 0
#endif
#ifndef CV_AVX
#  define CV_AVX 0
#endif
#ifndef CV_FP16
#  define CV_FP16 0
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
#ifndef CV_AVX_512IFMA
#  define CV_AVX_512IFMA 0
#endif
#define CV_AVX_512IFMA512 CV_AVX_512IFMA // deprecated
#ifndef CV_AVX_512PF
#  define CV_AVX_512PF 0
#endif
#ifndef CV_AVX_512VBMI
#  define CV_AVX_512VBMI 0
#endif
#ifndef CV_AVX_512VL
#  define CV_AVX_512VL 0
#endif
#ifndef CV_AVX_5124FMAPS
#  define CV_AVX_5124FMAPS 0
#endif
#ifndef CV_AVX_5124VNNIW
#  define CV_AVX_5124VNNIW 0
#endif
#ifndef CV_AVX_512VPOPCNTDQ
#  define CV_AVX_512VPOPCNTDQ 0
#endif
#ifndef CV_AVX_512VNNI
#  define CV_AVX_512VNNI 0
#endif
#ifndef CV_AVX_512VBMI2
#  define CV_AVX_512VBMI2 0
#endif
#ifndef CV_AVX_512BITALG
#  define CV_AVX_512BITALG 0
#endif
#ifndef CV_AVX512_COMMON
#  define CV_AVX512_COMMON 0
#endif
#ifndef CV_AVX512_KNL
#  define CV_AVX512_KNL 0
#endif
#ifndef CV_AVX512_KNM
#  define CV_AVX512_KNM 0
#endif
#ifndef CV_AVX512_SKX
#  define CV_AVX512_SKX 0
#endif
#ifndef CV_AVX512_CNL
#  define CV_AVX512_CNL 0
#endif
#ifndef CV_AVX512_CLX
#  define CV_AVX512_CLX 0
#endif
#ifndef CV_AVX512_ICL
#  define CV_AVX512_ICL 0
#endif

#ifndef CV_NEON
#  define CV_NEON 0
#endif

#ifndef CV_RVV071
#  define CV_RVV071 0
#endif

#ifndef CV_VSX
#  define CV_VSX 0
#endif

#ifndef CV_VSX3
#  define CV_VSX3 0
#endif

#ifndef CV_MSA
#  define CV_MSA 0
#endif

#ifndef CV_WASM_SIMD
#  define CV_WASM_SIMD 0
#endif

#ifndef CV_RVV
#  define CV_RVV 0
#endif
