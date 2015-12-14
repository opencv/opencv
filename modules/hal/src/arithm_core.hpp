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

#ifndef __OPENCV_HAL_ARITHM_CORE_HPP__
#define __OPENCV_HAL_ARITHM_CORE_HPP__

#include "arithm_simd.hpp"

const uchar g_Saturate8u[] =
{
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,
      0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,  14,  15,
     16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26,  27,  28,  29,  30,  31,
     32,  33,  34,  35,  36,  37,  38,  39,  40,  41,  42,  43,  44,  45,  46,  47,
     48,  49,  50,  51,  52,  53,  54,  55,  56,  57,  58,  59,  60,  61,  62,  63,
     64,  65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,  78,  79,
     80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,
     96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127,
    128, 129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143,
    144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159,
    160, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, 173, 174, 175,
    176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207,
    208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223,
    224, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239,
    240, 241, 242, 243, 244, 245, 246, 247, 248, 249, 250, 251, 252, 253, 254, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
    255
};


#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) && (t) <= 512), g_Saturate8u[(t)+256])
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))

const float g_8x32fTab[] =
{
    -128.f, -127.f, -126.f, -125.f, -124.f, -123.f, -122.f, -121.f,
    -120.f, -119.f, -118.f, -117.f, -116.f, -115.f, -114.f, -113.f,
    -112.f, -111.f, -110.f, -109.f, -108.f, -107.f, -106.f, -105.f,
    -104.f, -103.f, -102.f, -101.f, -100.f,  -99.f,  -98.f,  -97.f,
     -96.f,  -95.f,  -94.f,  -93.f,  -92.f,  -91.f,  -90.f,  -89.f,
     -88.f,  -87.f,  -86.f,  -85.f,  -84.f,  -83.f,  -82.f,  -81.f,
     -80.f,  -79.f,  -78.f,  -77.f,  -76.f,  -75.f,  -74.f,  -73.f,
     -72.f,  -71.f,  -70.f,  -69.f,  -68.f,  -67.f,  -66.f,  -65.f,
     -64.f,  -63.f,  -62.f,  -61.f,  -60.f,  -59.f,  -58.f,  -57.f,
     -56.f,  -55.f,  -54.f,  -53.f,  -52.f,  -51.f,  -50.f,  -49.f,
     -48.f,  -47.f,  -46.f,  -45.f,  -44.f,  -43.f,  -42.f,  -41.f,
     -40.f,  -39.f,  -38.f,  -37.f,  -36.f,  -35.f,  -34.f,  -33.f,
     -32.f,  -31.f,  -30.f,  -29.f,  -28.f,  -27.f,  -26.f,  -25.f,
     -24.f,  -23.f,  -22.f,  -21.f,  -20.f,  -19.f,  -18.f,  -17.f,
     -16.f,  -15.f,  -14.f,  -13.f,  -12.f,  -11.f,  -10.f,   -9.f,
      -8.f,   -7.f,   -6.f,   -5.f,   -4.f,   -3.f,   -2.f,   -1.f,
       0.f,    1.f,    2.f,    3.f,    4.f,    5.f,    6.f,    7.f,
       8.f,    9.f,   10.f,   11.f,   12.f,   13.f,   14.f,   15.f,
      16.f,   17.f,   18.f,   19.f,   20.f,   21.f,   22.f,   23.f,
      24.f,   25.f,   26.f,   27.f,   28.f,   29.f,   30.f,   31.f,
      32.f,   33.f,   34.f,   35.f,   36.f,   37.f,   38.f,   39.f,
      40.f,   41.f,   42.f,   43.f,   44.f,   45.f,   46.f,   47.f,
      48.f,   49.f,   50.f,   51.f,   52.f,   53.f,   54.f,   55.f,
      56.f,   57.f,   58.f,   59.f,   60.f,   61.f,   62.f,   63.f,
      64.f,   65.f,   66.f,   67.f,   68.f,   69.f,   70.f,   71.f,
      72.f,   73.f,   74.f,   75.f,   76.f,   77.f,   78.f,   79.f,
      80.f,   81.f,   82.f,   83.f,   84.f,   85.f,   86.f,   87.f,
      88.f,   89.f,   90.f,   91.f,   92.f,   93.f,   94.f,   95.f,
      96.f,   97.f,   98.f,   99.f,  100.f,  101.f,  102.f,  103.f,
     104.f,  105.f,  106.f,  107.f,  108.f,  109.f,  110.f,  111.f,
     112.f,  113.f,  114.f,  115.f,  116.f,  117.f,  118.f,  119.f,
     120.f,  121.f,  122.f,  123.f,  124.f,  125.f,  126.f,  127.f,
     128.f,  129.f,  130.f,  131.f,  132.f,  133.f,  134.f,  135.f,
     136.f,  137.f,  138.f,  139.f,  140.f,  141.f,  142.f,  143.f,
     144.f,  145.f,  146.f,  147.f,  148.f,  149.f,  150.f,  151.f,
     152.f,  153.f,  154.f,  155.f,  156.f,  157.f,  158.f,  159.f,
     160.f,  161.f,  162.f,  163.f,  164.f,  165.f,  166.f,  167.f,
     168.f,  169.f,  170.f,  171.f,  172.f,  173.f,  174.f,  175.f,
     176.f,  177.f,  178.f,  179.f,  180.f,  181.f,  182.f,  183.f,
     184.f,  185.f,  186.f,  187.f,  188.f,  189.f,  190.f,  191.f,
     192.f,  193.f,  194.f,  195.f,  196.f,  197.f,  198.f,  199.f,
     200.f,  201.f,  202.f,  203.f,  204.f,  205.f,  206.f,  207.f,
     208.f,  209.f,  210.f,  211.f,  212.f,  213.f,  214.f,  215.f,
     216.f,  217.f,  218.f,  219.f,  220.f,  221.f,  222.f,  223.f,
     224.f,  225.f,  226.f,  227.f,  228.f,  229.f,  230.f,  231.f,
     232.f,  233.f,  234.f,  235.f,  236.f,  237.f,  238.f,  239.f,
     240.f,  241.f,  242.f,  243.f,  244.f,  245.f,  246.f,  247.f,
     248.f,  249.f,  250.f,  251.f,  252.f,  253.f,  254.f,  255.f
};

#define CV_8TO32F(x)  g_8x32fTab[(x)+128]

namespace cv {

template<> inline uchar OpAdd<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a + b); }

template<> inline uchar OpSub<uchar>::operator ()(uchar a, uchar b) const
{ return CV_FAST_CAST_8U(a - b); }

template<> inline short OpAbsDiff<short>::operator ()(short a, short b) const
{ return saturate_cast<short>(std::abs(a - b)); }

template<> inline schar OpAbsDiff<schar>::operator ()(schar a, schar b) const
{ return saturate_cast<schar>(std::abs(a - b)); }

template<> inline uchar OpMin<uchar>::operator ()(uchar a, uchar b) const { return CV_MIN_8U(a, b); }

template<> inline uchar OpMax<uchar>::operator ()(uchar a, uchar b) const { return CV_MAX_8U(a, b); }

}

namespace cv { namespace hal {

template<typename T, class Op, class VOp>
void vBinOp(const T* src1, size_t step1, const T* src2, size_t step2, T* dst, size_t step, int width, int height)
{
#if CV_SSE2 || CV_NEON
    VOp vop;
#endif
    Op op;

    for( ; height--; src1 = (const T *)((const uchar *)src1 + step1),
                        src2 = (const T *)((const uchar *)src2 + step2),
                        dst = (T *)((uchar *)dst + step) )
    {
        int x = 0;

#if CV_NEON || CV_SSE2
#if CV_AVX2
        if( USE_AVX2 )
        {
            for( ; x <= width - 32/(int)sizeof(T); x += 32/sizeof(T) )
            {
                typename VLoadStore256<T>::reg_type r0 = VLoadStore256<T>::load(src1 + x);
                r0 = vop(r0, VLoadStore256<T>::load(src2 + x));
                VLoadStore256<T>::store(dst + x, r0);
            }
        }
#else
#if CV_SSE2
        if( USE_SSE2 )
        {
#endif // CV_SSE2
            for( ; x <= width - 32/(int)sizeof(T); x += 32/sizeof(T) )
            {
                typename VLoadStore128<T>::reg_type r0 = VLoadStore128<T>::load(src1 + x               );
                typename VLoadStore128<T>::reg_type r1 = VLoadStore128<T>::load(src1 + x + 16/sizeof(T));
                r0 = vop(r0, VLoadStore128<T>::load(src2 + x               ));
                r1 = vop(r1, VLoadStore128<T>::load(src2 + x + 16/sizeof(T)));
                VLoadStore128<T>::store(dst + x               , r0);
                VLoadStore128<T>::store(dst + x + 16/sizeof(T), r1);
            }
#if CV_SSE2
        }
#endif // CV_SSE2
#endif // CV_AVX2
#endif // CV_NEON || CV_SSE2

#if CV_AVX2
        // nothing
#elif CV_SSE2
        if( USE_SSE2 )
        {
            for( ; x <= width - 8/(int)sizeof(T); x += 8/sizeof(T) )
            {
                typename VLoadStore64<T>::reg_type r = VLoadStore64<T>::load(src1 + x);
                r = vop(r, VLoadStore64<T>::load(src2 + x));
                VLoadStore64<T>::store(dst + x, r);
            }
        }
#endif

#if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            T v0 = op(src1[x], src2[x]);
            T v1 = op(src1[x+1], src2[x+1]);
            dst[x] = v0; dst[x+1] = v1;
            v0 = op(src1[x+2], src2[x+2]);
            v1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = v0; dst[x+3] = v1;
        }
#endif

        for( ; x < width; x++ )
            dst[x] = op(src1[x], src2[x]);
    }
}

template<typename T, class Op, class Op32>
void vBinOp32(const T* src1, size_t step1, const T* src2, size_t step2,
              T* dst, size_t step, int width, int height)
{
#if CV_SSE2 || CV_NEON
    Op32 op32;
#endif
    Op op;

    for( ; height--; src1 = (const T *)((const uchar *)src1 + step1),
                        src2 = (const T *)((const uchar *)src2 + step2),
                        dst = (T *)((uchar *)dst + step) )
    {
        int x = 0;

#if CV_AVX2
        if( USE_AVX2 )
        {
            if( (((size_t)src1|(size_t)src2|(size_t)dst)&31) == 0 )
            {
                for( ; x <= width - 8; x += 8 )
                {
                    typename VLoadStore256Aligned<T>::reg_type r0 = VLoadStore256Aligned<T>::load(src1 + x);
                    r0 = op32(r0, VLoadStore256Aligned<T>::load(src2 + x));
                    VLoadStore256Aligned<T>::store(dst + x, r0);
                }
            }
        }
#elif CV_SSE2
        if( USE_SSE2 )
        {
            if( (((size_t)src1|(size_t)src2|(size_t)dst)&15) == 0 )
            {
                for( ; x <= width - 8; x += 8 )
                {
                    typename VLoadStore128Aligned<T>::reg_type r0 = VLoadStore128Aligned<T>::load(src1 + x    );
                    typename VLoadStore128Aligned<T>::reg_type r1 = VLoadStore128Aligned<T>::load(src1 + x + 4);
                    r0 = op32(r0, VLoadStore128Aligned<T>::load(src2 + x    ));
                    r1 = op32(r1, VLoadStore128Aligned<T>::load(src2 + x + 4));
                    VLoadStore128Aligned<T>::store(dst + x    , r0);
                    VLoadStore128Aligned<T>::store(dst + x + 4, r1);
                }
            }
        }
#endif // CV_AVX2

#if CV_NEON || CV_SSE2
#if CV_AVX2
        if( USE_AVX2 )
        {
            for( ; x <= width - 8; x += 8 )
            {
                typename VLoadStore256<T>::reg_type r0 = VLoadStore256<T>::load(src1 + x);
                r0 = op32(r0, VLoadStore256<T>::load(src2 + x));
                VLoadStore256<T>::store(dst + x, r0);
            }
        }
#else
#if CV_SSE2
        if( USE_SSE2 )
        {
#endif // CV_SSE2
            for( ; x <= width - 8; x += 8 )
            {
                typename VLoadStore128<T>::reg_type r0 = VLoadStore128<T>::load(src1 + x    );
                typename VLoadStore128<T>::reg_type r1 = VLoadStore128<T>::load(src1 + x + 4);
                r0 = op32(r0, VLoadStore128<T>::load(src2 + x    ));
                r1 = op32(r1, VLoadStore128<T>::load(src2 + x + 4));
                VLoadStore128<T>::store(dst + x    , r0);
                VLoadStore128<T>::store(dst + x + 4, r1);
            }
#if CV_SSE2
        }
#endif // CV_SSE2
#endif // CV_AVX2
#endif // CV_NEON || CV_SSE2

#if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            T v0 = op(src1[x], src2[x]);
            T v1 = op(src1[x+1], src2[x+1]);
            dst[x] = v0; dst[x+1] = v1;
            v0 = op(src1[x+2], src2[x+2]);
            v1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = v0; dst[x+3] = v1;
        }
#endif

        for( ; x < width; x++ )
            dst[x] = op(src1[x], src2[x]);
    }
}


template<typename T, class Op, class Op64>
void vBinOp64(const T* src1, size_t step1, const T* src2, size_t step2,
               T* dst, size_t step, int width, int height)
{
#if CV_SSE2
    Op64 op64;
#endif
    Op op;

    for( ; height--; src1 = (const T *)((const uchar *)src1 + step1),
                        src2 = (const T *)((const uchar *)src2 + step2),
                        dst = (T *)((uchar *)dst + step) )
    {
        int x = 0;

#if CV_AVX2
        if( USE_AVX2 )
        {
            if( (((size_t)src1|(size_t)src2|(size_t)dst)&31) == 0 )
            {
                for( ; x <= width - 4; x += 4 )
                {
                    typename VLoadStore256Aligned<T>::reg_type r0 = VLoadStore256Aligned<T>::load(src1 + x);
                    r0 = op64(r0, VLoadStore256Aligned<T>::load(src2 + x));
                    VLoadStore256Aligned<T>::store(dst + x, r0);
                }
            }
        }
#elif CV_SSE2
        if( USE_SSE2 )
        {
            if( (((size_t)src1|(size_t)src2|(size_t)dst)&15) == 0 )
            {
                for( ; x <= width - 4; x += 4 )
                {
                    typename VLoadStore128Aligned<T>::reg_type r0 = VLoadStore128Aligned<T>::load(src1 + x    );
                    typename VLoadStore128Aligned<T>::reg_type r1 = VLoadStore128Aligned<T>::load(src1 + x + 2);
                    r0 = op64(r0, VLoadStore128Aligned<T>::load(src2 + x    ));
                    r1 = op64(r1, VLoadStore128Aligned<T>::load(src2 + x + 2));
                    VLoadStore128Aligned<T>::store(dst + x    , r0);
                    VLoadStore128Aligned<T>::store(dst + x + 2, r1);
                }
            }
        }
#endif

        for( ; x <= width - 4; x += 4 )
        {
            T v0 = op(src1[x], src2[x]);
            T v1 = op(src1[x+1], src2[x+1]);
            dst[x] = v0; dst[x+1] = v1;
            v0 = op(src1[x+2], src2[x+2]);
            v1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = v0; dst[x+3] = v1;
        }

        for( ; x < width; x++ )
            dst[x] = op(src1[x], src2[x]);
    }
}

template<typename T> static void
cmp_(const T* src1, size_t step1, const T* src2, size_t step2,
     uchar* dst, size_t step, int width, int height, int code)
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    if( code == CMP_GE || code == CMP_LT )
    {
        std::swap(src1, src2);
        std::swap(step1, step2);
        code = code == CMP_GE ? CMP_LE : CMP_GT;
    }

    Cmp_SIMD<T> vop(code);

    if( code == CMP_GT || code == CMP_LE )
    {
        int m = code == CMP_GT ? 0 : 255;
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = vop(src1, src2, dst, width);
            #if CV_ENABLE_UNROLLED
            for( ; x <= width - 4; x += 4 )
            {
                int t0, t1;
                t0 = -(src1[x] > src2[x]) ^ m;
                t1 = -(src1[x+1] > src2[x+1]) ^ m;
                dst[x] = (uchar)t0; dst[x+1] = (uchar)t1;
                t0 = -(src1[x+2] > src2[x+2]) ^ m;
                t1 = -(src1[x+3] > src2[x+3]) ^ m;
                dst[x+2] = (uchar)t0; dst[x+3] = (uchar)t1;
            }
            #endif
            for( ; x < width; x++ )
                dst[x] = (uchar)(-(src1[x] > src2[x]) ^ m);
        }
    }
    else if( code == CMP_EQ || code == CMP_NE )
    {
        int m = code == CMP_EQ ? 0 : 255;
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int x = 0;
            #if CV_ENABLE_UNROLLED
            for( ; x <= width - 4; x += 4 )
            {
                int t0, t1;
                t0 = -(src1[x] == src2[x]) ^ m;
                t1 = -(src1[x+1] == src2[x+1]) ^ m;
                dst[x] = (uchar)t0; dst[x+1] = (uchar)t1;
                t0 = -(src1[x+2] == src2[x+2]) ^ m;
                t1 = -(src1[x+3] == src2[x+3]) ^ m;
                dst[x+2] = (uchar)t0; dst[x+3] = (uchar)t1;
            }
            #endif
            for( ; x < width; x++ )
                dst[x] = (uchar)(-(src1[x] == src2[x]) ^ m);
        }
    }
}

template<typename T, typename WT> static void
mul_( const T* src1, size_t step1, const T* src2, size_t step2,
      T* dst, size_t step, int width, int height, WT scale )
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    Mul_SIMD<T, WT> vop;

    if( scale == (WT)1. )
    {
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int i = vop(src1, src2, dst, width, scale);
            #if CV_ENABLE_UNROLLED
            for(; i <= width - 4; i += 4 )
            {
                T t0;
                T t1;
                t0 = saturate_cast<T>(src1[i  ] * src2[i  ]);
                t1 = saturate_cast<T>(src1[i+1] * src2[i+1]);
                dst[i  ] = t0;
                dst[i+1] = t1;

                t0 = saturate_cast<T>(src1[i+2] * src2[i+2]);
                t1 = saturate_cast<T>(src1[i+3] * src2[i+3]);
                dst[i+2] = t0;
                dst[i+3] = t1;
            }
            #endif
            for( ; i < width; i++ )
                dst[i] = saturate_cast<T>(src1[i] * src2[i]);
        }
    }
    else
    {
        for( ; height--; src1 += step1, src2 += step2, dst += step )
        {
            int i = vop(src1, src2, dst, width, scale);
            #if CV_ENABLE_UNROLLED
            for(; i <= width - 4; i += 4 )
            {
                T t0 = saturate_cast<T>(scale*(WT)src1[i]*src2[i]);
                T t1 = saturate_cast<T>(scale*(WT)src1[i+1]*src2[i+1]);
                dst[i] = t0; dst[i+1] = t1;

                t0 = saturate_cast<T>(scale*(WT)src1[i+2]*src2[i+2]);
                t1 = saturate_cast<T>(scale*(WT)src1[i+3]*src2[i+3]);
                dst[i+2] = t0; dst[i+3] = t1;
            }
            #endif
            for( ; i < width; i++ )
                dst[i] = saturate_cast<T>(scale*(WT)src1[i]*src2[i]);
        }
    }
}


template<typename T> static void
div_i( const T* src1, size_t step1, const T* src2, size_t step2,
      T* dst, size_t step, int width, int height, double scale )
{
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    Div_SIMD<T> vop;
    float scale_f = (float)scale;

    for( ; height--; src1 += step1, src2 += step2, dst += step )
    {
        int i = vop(src1, src2, dst, width, scale);
        for( ; i < width; i++ )
        {
            T num = src1[i], denom = src2[i];
            dst[i] = denom != 0 ? saturate_cast<T>(num*scale_f/denom) : (T)0;
        }
    }
}

template<typename T> static void
div_f( const T* src1, size_t step1, const T* src2, size_t step2,
      T* dst, size_t step, int width, int height, double scale )
{
    T scale_f = (T)scale;
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    Div_SIMD<T> vop;

    for( ; height--; src1 += step1, src2 += step2, dst += step )
    {
        int i = vop(src1, src2, dst, width, scale);
        for( ; i < width; i++ )
        {
            T num = src1[i], denom = src2[i];
            dst[i] = denom != 0 ? saturate_cast<T>(num*scale_f/denom) : (T)0;
        }
    }
}

template<typename T> static void
recip_i( const T*, size_t, const T* src2, size_t step2,
         T* dst, size_t step, int width, int height, double scale )
{
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    Recip_SIMD<T> vop;
    float scale_f = (float)scale;

    for( ; height--; src2 += step2, dst += step )
    {
        int i = vop(src2, dst, width, scale);
        for( ; i < width; i++ )
        {
            T denom = src2[i];
            dst[i] = denom != 0 ? saturate_cast<T>(scale_f/denom) : (T)0;
        }
    }
}

template<typename T> static void
recip_f( const T*, size_t, const T* src2, size_t step2,
         T* dst, size_t step, int width, int height, double scale )
{
    T scale_f = (T)scale;
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    Recip_SIMD<T> vop;

    for( ; height--; src2 += step2, dst += step )
    {
        int i = vop(src2, dst, width, scale);
        for( ; i < width; i++ )
        {
            T denom = src2[i];
            dst[i] = denom != 0 ? saturate_cast<T>(scale_f/denom) : (T)0;
        }
    }
}

template<typename T, typename WT> static void
addWeighted_( const T* src1, size_t step1, const T* src2, size_t step2,
              T* dst, size_t step, int width, int height, void* _scalars )
{
    const double* scalars = (const double*)_scalars;
    WT alpha = (WT)scalars[0], beta = (WT)scalars[1], gamma = (WT)scalars[2];
    step1 /= sizeof(src1[0]);
    step2 /= sizeof(src2[0]);
    step /= sizeof(dst[0]);

    AddWeighted_SIMD<T, WT> vop;

    for( ; height--; src1 += step1, src2 += step2, dst += step )
    {
        int x = vop(src1, src2, dst, width, alpha, beta, gamma);
        #if CV_ENABLE_UNROLLED
        for( ; x <= width - 4; x += 4 )
        {
            T t0 = saturate_cast<T>(src1[x]*alpha + src2[x]*beta + gamma);
            T t1 = saturate_cast<T>(src1[x+1]*alpha + src2[x+1]*beta + gamma);
            dst[x] = t0; dst[x+1] = t1;

            t0 = saturate_cast<T>(src1[x+2]*alpha + src2[x+2]*beta + gamma);
            t1 = saturate_cast<T>(src1[x+3]*alpha + src2[x+3]*beta + gamma);
            dst[x+2] = t0; dst[x+3] = t1;
        }
        #endif
        for( ; x < width; x++ )
            dst[x] = saturate_cast<T>(src1[x]*alpha + src2[x]*beta + gamma);
    }
}

}} // cv::hal::


#endif // __OPENCV_HAL_ARITHM_CORE_HPP__
