// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
// Copyright (C) 2026, Advanced Micro Devices, Inc., all rights reserved.

#include "precomp.hpp"

// VBMI vpermb is the only x86 SIMD path faster than scalar for byte LUT.
// x86 gather (AVX2/SSE) is slower than scalar due to high gather latency.
// Non-x86 (NEON, etc.) benefits from v_lut implementation.
#if CV_AVX_512VBMI
#define CV_LUT8U_SIMD 1
#elif (defined(CV_CPU_COMPILE_SSE) || defined(CV_CPU_COMPILE_SSE2) || defined(CV_CPU_COMPILE_SSE3) || \
       defined(CV_CPU_COMPILE_SSSE3) || defined(CV_CPU_COMPILE_SSE4_1) || defined(CV_CPU_COMPILE_SSE4_2) || \
       defined(CV_CPU_COMPILE_AVX) || defined(CV_CPU_COMPILE_AVX2) || defined(CV_CPU_COMPILE_AVX_512F) || \
       defined(CV_CPU_COMPILE_AVX512_COMMON) || defined(CV_CPU_COMPILE_AVX512_SKX) || \
       defined(CV_CPU_COMPILE_AVX512_CNL) || defined(CV_CPU_COMPILE_AVX512_CLX))
#define CV_LUT8U_SIMD 0
#elif CV_SIMD || CV_SIMD_SCALABLE
#define CV_LUT8U_SIMD 1
#else
#define CV_LUT8U_SIMD 0
#endif

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN

void LUT8u_( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn );
void LUT16u_( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn );

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

void LUT8u_( const uchar* src, const uchar* lut, uchar* dst, int len, int cn, int lutcn )
{
    const int total = len * cn;

    if( lutcn == 1 )
    {
        int i = 0;
#if CV_LUT8U_SIMD
        const int nlanes = VTraits<v_uint8>::vlanes();
        for( ; i <= total - nlanes; i += nlanes )
        {
            v_uint8 idx = vx_load(src + i);
            v_uint8 result = v_lut(lut, idx);
            v_store(dst + i, result);
        }
#endif
        for( ; i < total; i++ )
            dst[i] = lut[src[i]];
    }
#if CV_LUT8U_SIMD
    else if( cn == 3 )
    {
        // Deinterleave the 3-channel LUT into per-channel tables
        uchar CV_DECL_ALIGNED(64) lut0[256], lut1[256], lut2[256];
        const int nlanes = VTraits<v_uint8>::vlanes();
        {
            unsigned j = 0;
            for( ; j + (unsigned)nlanes <= 256; j += (unsigned)nlanes )
            {
                v_uint8 a, b, c;
                v_load_deinterleave(lut + j * 3, a, b, c);
                v_store(lut0 + j, a);
                v_store(lut1 + j, b);
                v_store(lut2 + j, c);
            }
            for( ; j < 256; j++ )
            {
                const unsigned idx = j * 3;
                lut0[j] = lut[idx];
                lut1[j] = lut[idx + 1];
                lut2[j] = lut[idx + 2];
            }
        }
        int i = 0;
        for( ; i <= total - nlanes*3; i += nlanes*3 )
        {
            v_uint8 r, g, b;
            v_load_deinterleave(src + i, r, g, b);
            r = v_lut(lut0, r);
            g = v_lut(lut1, g);
            b = v_lut(lut2, b);
            v_store_interleave(dst + i, r, g, b);
        }
        for( ; i < total; i += 3 )
        {
            dst[i]   = lut0[src[i]];
            dst[i+1] = lut1[src[i+1]];
            dst[i+2] = lut2[src[i+2]];
        }
    }
#endif
    else
    {
        for( int i = 0; i < total; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn + k];
    }
}

void LUT16u_( const uchar* src, const ushort* lut, ushort* dst, int len, int cn, int lutcn )
{
    const int total = len * cn;

    if( lutcn == 1 )
    {
        int i = 0;
#if CV_LUT8U_SIMD
        // Split ushort LUT into low-byte and high-byte tables,
        // then use two byte v_lut (vpermb on VBMI) + interleave.
        uchar CV_DECL_ALIGNED(64) lut_lo[256], lut_hi[256];
        for( int j = 0; j < 256; j++ )
        {
            lut_lo[j] = (uchar)(lut[j]);
            lut_hi[j] = (uchar)(lut[j] >> 8);
        }
        const int nlanes8 = VTraits<v_uint8>::vlanes();
        const int nlanes16 = VTraits<v_uint16>::vlanes();
        for( ; i <= total - nlanes8; i += nlanes8 )
        {
            v_uint8 idx = vx_load(src + i);
            v_uint8 lo = v_lut(lut_lo, idx);
            v_uint8 hi = v_lut(lut_hi, idx);
            // Zip low and high bytes: [l0,h0,l1,h1,...] → ushort values on little-endian
            v_uint8 res0, res1;
            v_zip(lo, hi, res0, res1);
            v_store(dst + i, v_reinterpret_as_u16(res0));
            v_store(dst + i + nlanes16, v_reinterpret_as_u16(res1));
        }
#endif
        for( ; i < total; i++ )
            dst[i] = lut[src[i]];
    }
    else
    {
        for( int i = 0; i < total; i += cn )
            for( int k = 0; k < cn; k++ )
                dst[i+k] = lut[src[i+k]*cn + k];
    }
}

#endif // CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace cv
