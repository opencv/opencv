// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"
#include "opencv2/core/hal/intrin.hpp"

// VBMI vpermb is the only x86 SIMD path faster than scalar for byte LUT.
// Non-VBMI x86 SIMD (gather-based) is slower than scalar for byte LUT.
// Non-x86 (NEON, RVV, etc.) benefits from v_lut implementation.
#if CV_AVX_512VBMI
#define CV_EQUALIZE_HIST_SIMD 1
#elif (defined(CV_CPU_COMPILE_SSE) || defined(CV_CPU_COMPILE_SSE2) || defined(CV_CPU_COMPILE_SSE3) || \
       defined(CV_CPU_COMPILE_SSSE3) || defined(CV_CPU_COMPILE_SSE4_1) || defined(CV_CPU_COMPILE_SSE4_2) || \
       defined(CV_CPU_COMPILE_AVX) || defined(CV_CPU_COMPILE_AVX2) || defined(CV_CPU_COMPILE_AVX_512F) || \
       defined(CV_CPU_COMPILE_AVX512_COMMON) || defined(CV_CPU_COMPILE_AVX512_SKX) || \
       defined(CV_CPU_COMPILE_AVX512_CNL) || defined(CV_CPU_COMPILE_AVX512_CLX))
#define CV_EQUALIZE_HIST_SIMD 0
#elif CV_SIMD || CV_SIMD_SCALABLE
#define CV_EQUALIZE_HIST_SIMD 1
#else
#define CV_EQUALIZE_HIST_SIMD 0
#endif

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void equalizeHistLut_( const uchar* src, uchar* dst, int len, const uchar* lut );

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

void equalizeHistLut_( const uchar* src, uchar* dst, int len, const uchar* lut )
{
    int x = 0;
#if CV_EQUALIZE_HIST_SIMD
    const int nlanes = VTraits<v_uint8>::vlanes();
    for( ; x <= len - nlanes; x += nlanes )
    {
        v_uint8 idx = vx_load(src + x);
        v_uint8 result = v_lut(lut, idx);
        v_store(dst + x, result);
    }
#endif
    for( ; x <= len - 4; x += 4 )
    {
        uchar v0 = src[x], v1 = src[x+1];
        dst[x]   = lut[v0];
        dst[x+1] = lut[v1];
        v0 = src[x+2]; v1 = src[x+3];
        dst[x+2] = lut[v0];
        dst[x+3] = lut[v1];
    }
    for( ; x < len; x++ )
        dst[x] = lut[src[x]];
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
