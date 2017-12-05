/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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

#include "precomp.hpp"
#include <limits.h>
#include "opencl_kernels_imgproc.hpp"
#include <iostream>
#include "hal_replacement.hpp"

/****************************************************************************************\
                     Basic Morphological Operations: Erosion & Dilation
\****************************************************************************************/

using namespace std;

namespace cv
{

template<typename T> struct MinOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(const T a, const T b) const { return std::min(a, b); }
};

template<typename T> struct MaxOp
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(const T a, const T b) const { return std::max(a, b); }
};

#undef CV_MIN_8U
#undef CV_MAX_8U
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))

template<> inline uchar MinOp<uchar>::operator ()(const uchar a, const uchar b) const { return CV_MIN_8U(a, b); }
template<> inline uchar MaxOp<uchar>::operator ()(const uchar a, const uchar b) const { return CV_MAX_8U(a, b); }

struct MorphRowNoVec
{
    MorphRowNoVec(int, int) {}
    int operator()(const uchar*, uchar*, int, int) const { return 0; }
};

struct MorphColumnNoVec
{
    MorphColumnNoVec(int, int) {}
    int operator()(const uchar**, uchar*, int, int, int) const { return 0; }
};

struct MorphNoVec
{
    int operator()(uchar**, int, uchar*, int) const { return 0; }
};

#if CV_SSE2

template<class VecUpdate> struct MorphRowIVec
{
    enum { ESZ = VecUpdate::ESZ };

    MorphRowIVec(int _ksize, int _anchor) : ksize(_ksize), anchor(_anchor) {}
    int operator()(const uchar* src, uchar* dst, int width, int cn) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        cn *= ESZ;
        int i, k, _ksize = ksize*cn;
        width = (width & -4)*cn;
        VecUpdate updateOp;

        for( i = 0; i <= width - 16; i += 16 )
        {
            __m128i s = _mm_loadu_si128((const __m128i*)(src + i));
            for( k = cn; k < _ksize; k += cn )
            {
                __m128i x = _mm_loadu_si128((const __m128i*)(src + i + k));
                s = updateOp(s, x);
            }
            _mm_storeu_si128((__m128i*)(dst + i), s);
        }

        for( ; i < width; i += 4 )
        {
            __m128i s = _mm_cvtsi32_si128(*(const int*)(src + i));
            for( k = cn; k < _ksize; k += cn )
            {
                __m128i x = _mm_cvtsi32_si128(*(const int*)(src + i + k));
                s = updateOp(s, x);
            }
            *(int*)(dst + i) = _mm_cvtsi128_si32(s);
        }

        return i/ESZ;
    }

    int ksize, anchor;
};


template<class VecUpdate> struct MorphRowFVec
{
    MorphRowFVec(int _ksize, int _anchor) : ksize(_ksize), anchor(_anchor) {}
    int operator()(const uchar* src, uchar* dst, int width, int cn) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        int i, k, _ksize = ksize*cn;
        width = (width & -4)*cn;
        VecUpdate updateOp;

        for( i = 0; i < width; i += 4 )
        {
            __m128 s = _mm_loadu_ps((const float*)src + i);
            for( k = cn; k < _ksize; k += cn )
            {
                __m128 x = _mm_loadu_ps((const float*)src + i + k);
                s = updateOp(s, x);
            }
            _mm_storeu_ps((float*)dst + i, s);
        }

        return i;
    }

    int ksize, anchor;
};


template<class VecUpdate> struct MorphColumnIVec
{
    enum { ESZ = VecUpdate::ESZ };

    MorphColumnIVec(int _ksize, int _anchor) : ksize(_ksize), anchor(_anchor) {}
    int operator()(const uchar** src, uchar* dst, int dststep, int count, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        int i = 0, k, _ksize = ksize;
        width *= ESZ;
        VecUpdate updateOp;

        for( i = 0; i < count + ksize - 1; i++ )
            CV_Assert( ((size_t)src[i] & 15) == 0 );

        for( ; _ksize > 1 && count > 1; count -= 2, dst += dststep*2, src += 2 )
        {
            for( i = 0; i <= width - 32; i += 32 )
            {
                const uchar* sptr = src[1] + i;
                __m128i s0 = _mm_load_si128((const __m128i*)sptr);
                __m128i s1 = _mm_load_si128((const __m128i*)(sptr + 16));
                __m128i x0, x1;

                for( k = 2; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    x0 = _mm_load_si128((const __m128i*)sptr);
                    x1 = _mm_load_si128((const __m128i*)(sptr + 16));
                    s0 = updateOp(s0, x0);
                    s1 = updateOp(s1, x1);
                }

                sptr = src[0] + i;
                x0 = _mm_load_si128((const __m128i*)sptr);
                x1 = _mm_load_si128((const __m128i*)(sptr + 16));
                _mm_storeu_si128((__m128i*)(dst + i), updateOp(s0, x0));
                _mm_storeu_si128((__m128i*)(dst + i + 16), updateOp(s1, x1));

                sptr = src[k] + i;
                x0 = _mm_load_si128((const __m128i*)sptr);
                x1 = _mm_load_si128((const __m128i*)(sptr + 16));
                _mm_storeu_si128((__m128i*)(dst + dststep + i), updateOp(s0, x0));
                _mm_storeu_si128((__m128i*)(dst + dststep + i + 16), updateOp(s1, x1));
            }

            for( ; i <= width - 8; i += 8 )
            {
                __m128i s0 = _mm_loadl_epi64((const __m128i*)(src[1] + i)), x0;

                for( k = 2; k < _ksize; k++ )
                {
                    x0 = _mm_loadl_epi64((const __m128i*)(src[k] + i));
                    s0 = updateOp(s0, x0);
                }

                x0 = _mm_loadl_epi64((const __m128i*)(src[0] + i));
                _mm_storel_epi64((__m128i*)(dst + i), updateOp(s0, x0));
                x0 = _mm_loadl_epi64((const __m128i*)(src[k] + i));
                _mm_storel_epi64((__m128i*)(dst + dststep + i), updateOp(s0, x0));
            }
        }

        for( ; count > 0; count--, dst += dststep, src++ )
        {
            for( i = 0; i <= width - 32; i += 32 )
            {
                const uchar* sptr = src[0] + i;
                __m128i s0 = _mm_load_si128((const __m128i*)sptr);
                __m128i s1 = _mm_load_si128((const __m128i*)(sptr + 16));
                __m128i x0, x1;

                for( k = 1; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    x0 = _mm_load_si128((const __m128i*)sptr);
                    x1 = _mm_load_si128((const __m128i*)(sptr + 16));
                    s0 = updateOp(s0, x0);
                    s1 = updateOp(s1, x1);
                }
                _mm_storeu_si128((__m128i*)(dst + i), s0);
                _mm_storeu_si128((__m128i*)(dst + i + 16), s1);
            }

            for( ; i <= width - 8; i += 8 )
            {
                __m128i s0 = _mm_loadl_epi64((const __m128i*)(src[0] + i)), x0;

                for( k = 1; k < _ksize; k++ )
                {
                    x0 = _mm_loadl_epi64((const __m128i*)(src[k] + i));
                    s0 = updateOp(s0, x0);
                }
                _mm_storel_epi64((__m128i*)(dst + i), s0);
            }
        }

        return i/ESZ;
    }

    int ksize, anchor;
};


template<class VecUpdate> struct MorphColumnFVec
{
    MorphColumnFVec(int _ksize, int _anchor) : ksize(_ksize), anchor(_anchor) {}
    int operator()(const uchar** _src, uchar* _dst, int dststep, int count, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        int i = 0, k, _ksize = ksize;
        VecUpdate updateOp;

        for( i = 0; i < count + ksize - 1; i++ )
            CV_Assert( ((size_t)_src[i] & 15) == 0 );

        const float** src = (const float**)_src;
        float* dst = (float*)_dst;
        dststep /= sizeof(dst[0]);

        for( ; _ksize > 1 && count > 1; count -= 2, dst += dststep*2, src += 2 )
        {
            for( i = 0; i <= width - 16; i += 16 )
            {
                const float* sptr = src[1] + i;
                __m128 s0 = _mm_load_ps(sptr);
                __m128 s1 = _mm_load_ps(sptr + 4);
                __m128 s2 = _mm_load_ps(sptr + 8);
                __m128 s3 = _mm_load_ps(sptr + 12);
                __m128 x0, x1, x2, x3;

                for( k = 2; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    x0 = _mm_load_ps(sptr);
                    x1 = _mm_load_ps(sptr + 4);
                    s0 = updateOp(s0, x0);
                    s1 = updateOp(s1, x1);
                    x2 = _mm_load_ps(sptr + 8);
                    x3 = _mm_load_ps(sptr + 12);
                    s2 = updateOp(s2, x2);
                    s3 = updateOp(s3, x3);
                }

                sptr = src[0] + i;
                x0 = _mm_load_ps(sptr);
                x1 = _mm_load_ps(sptr + 4);
                x2 = _mm_load_ps(sptr + 8);
                x3 = _mm_load_ps(sptr + 12);
                _mm_storeu_ps(dst + i, updateOp(s0, x0));
                _mm_storeu_ps(dst + i + 4, updateOp(s1, x1));
                _mm_storeu_ps(dst + i + 8, updateOp(s2, x2));
                _mm_storeu_ps(dst + i + 12, updateOp(s3, x3));

                sptr = src[k] + i;
                x0 = _mm_load_ps(sptr);
                x1 = _mm_load_ps(sptr + 4);
                x2 = _mm_load_ps(sptr + 8);
                x3 = _mm_load_ps(sptr + 12);
                _mm_storeu_ps(dst + dststep + i, updateOp(s0, x0));
                _mm_storeu_ps(dst + dststep + i + 4, updateOp(s1, x1));
                _mm_storeu_ps(dst + dststep + i + 8, updateOp(s2, x2));
                _mm_storeu_ps(dst + dststep + i + 12, updateOp(s3, x3));
            }

            for( ; i <= width - 4; i += 4 )
            {
                __m128 s0 = _mm_load_ps(src[1] + i), x0;

                for( k = 2; k < _ksize; k++ )
                {
                    x0 = _mm_load_ps(src[k] + i);
                    s0 = updateOp(s0, x0);
                }

                x0 = _mm_load_ps(src[0] + i);
                _mm_storeu_ps(dst + i, updateOp(s0, x0));
                x0 = _mm_load_ps(src[k] + i);
                _mm_storeu_ps(dst + dststep + i, updateOp(s0, x0));
            }
        }

        for( ; count > 0; count--, dst += dststep, src++ )
        {
            for( i = 0; i <= width - 16; i += 16 )
            {
                const float* sptr = src[0] + i;
                __m128 s0 = _mm_load_ps(sptr);
                __m128 s1 = _mm_load_ps(sptr + 4);
                __m128 s2 = _mm_load_ps(sptr + 8);
                __m128 s3 = _mm_load_ps(sptr + 12);
                __m128 x0, x1, x2, x3;

                for( k = 1; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    x0 = _mm_load_ps(sptr);
                    x1 = _mm_load_ps(sptr + 4);
                    s0 = updateOp(s0, x0);
                    s1 = updateOp(s1, x1);
                    x2 = _mm_load_ps(sptr + 8);
                    x3 = _mm_load_ps(sptr + 12);
                    s2 = updateOp(s2, x2);
                    s3 = updateOp(s3, x3);
                }
                _mm_storeu_ps(dst + i, s0);
                _mm_storeu_ps(dst + i + 4, s1);
                _mm_storeu_ps(dst + i + 8, s2);
                _mm_storeu_ps(dst + i + 12, s3);
            }

            for( i = 0; i <= width - 4; i += 4 )
            {
                __m128 s0 = _mm_load_ps(src[0] + i), x0;
                for( k = 1; k < _ksize; k++ )
                {
                    x0 = _mm_load_ps(src[k] + i);
                    s0 = updateOp(s0, x0);
                }
                _mm_storeu_ps(dst + i, s0);
            }
        }

        return i;
    }

    int ksize, anchor;
};


template<class VecUpdate> struct MorphIVec
{
    enum { ESZ = VecUpdate::ESZ };

    int operator()(uchar** src, int nz, uchar* dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE2) )
            return 0;

        int i, k;
        width *= ESZ;
        VecUpdate updateOp;

        for( i = 0; i <= width - 32; i += 32 )
        {
            const uchar* sptr = src[0] + i;
            __m128i s0 = _mm_loadu_si128((const __m128i*)sptr);
            __m128i s1 = _mm_loadu_si128((const __m128i*)(sptr + 16));
            __m128i x0, x1;

            for( k = 1; k < nz; k++ )
            {
                sptr = src[k] + i;
                x0 = _mm_loadu_si128((const __m128i*)sptr);
                x1 = _mm_loadu_si128((const __m128i*)(sptr + 16));
                s0 = updateOp(s0, x0);
                s1 = updateOp(s1, x1);
            }
            _mm_storeu_si128((__m128i*)(dst + i), s0);
            _mm_storeu_si128((__m128i*)(dst + i + 16), s1);
        }

        for( ; i <= width - 8; i += 8 )
        {
            __m128i s0 = _mm_loadl_epi64((const __m128i*)(src[0] + i)), x0;

            for( k = 1; k < nz; k++ )
            {
                x0 = _mm_loadl_epi64((const __m128i*)(src[k] + i));
                s0 = updateOp(s0, x0);
            }
            _mm_storel_epi64((__m128i*)(dst + i), s0);
        }

        return i/ESZ;
    }
};


template<class VecUpdate> struct MorphFVec
{
    int operator()(uchar** _src, int nz, uchar* _dst, int width) const
    {
        if( !checkHardwareSupport(CV_CPU_SSE) )
            return 0;

        const float** src = (const float**)_src;
        float* dst = (float*)_dst;
        int i, k;
        VecUpdate updateOp;

        for( i = 0; i <= width - 16; i += 16 )
        {
            const float* sptr = src[0] + i;
            __m128 s0 = _mm_loadu_ps(sptr);
            __m128 s1 = _mm_loadu_ps(sptr + 4);
            __m128 s2 = _mm_loadu_ps(sptr + 8);
            __m128 s3 = _mm_loadu_ps(sptr + 12);
            __m128 x0, x1, x2, x3;

            for( k = 1; k < nz; k++ )
            {
                sptr = src[k] + i;
                x0 = _mm_loadu_ps(sptr);
                x1 = _mm_loadu_ps(sptr + 4);
                x2 = _mm_loadu_ps(sptr + 8);
                x3 = _mm_loadu_ps(sptr + 12);
                s0 = updateOp(s0, x0);
                s1 = updateOp(s1, x1);
                s2 = updateOp(s2, x2);
                s3 = updateOp(s3, x3);
            }
            _mm_storeu_ps(dst + i, s0);
            _mm_storeu_ps(dst + i + 4, s1);
            _mm_storeu_ps(dst + i + 8, s2);
            _mm_storeu_ps(dst + i + 12, s3);
        }

        for( ; i <= width - 4; i += 4 )
        {
            __m128 s0 = _mm_loadu_ps(src[0] + i), x0;

            for( k = 1; k < nz; k++ )
            {
                x0 = _mm_loadu_ps(src[k] + i);
                s0 = updateOp(s0, x0);
            }
            _mm_storeu_ps(dst + i, s0);
        }

        for( ; i < width; i++ )
        {
            __m128 s0 = _mm_load_ss(src[0] + i), x0;

            for( k = 1; k < nz; k++ )
            {
                x0 = _mm_load_ss(src[k] + i);
                s0 = updateOp(s0, x0);
            }
            _mm_store_ss(dst + i, s0);
        }

        return i;
    }
};

struct VMin8u
{
    enum { ESZ = 1 };
    __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_min_epu8(a,b); }
};
struct VMax8u
{
    enum { ESZ = 1 };
    __m128i operator()(const __m128i& a, const __m128i& b) const { return _mm_max_epu8(a,b); }
};
struct VMin16u
{
    enum { ESZ = 2 };
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_subs_epu16(a,_mm_subs_epu16(a,b)); }
};
struct VMax16u
{
    enum { ESZ = 2 };
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_adds_epu16(_mm_subs_epu16(a,b), b); }
};
struct VMin16s
{
    enum { ESZ = 2 };
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_min_epi16(a, b); }
};
struct VMax16s
{
    enum { ESZ = 2 };
    __m128i operator()(const __m128i& a, const __m128i& b) const
    { return _mm_max_epi16(a, b); }
};
struct VMin32f { __m128 operator()(const __m128& a, const __m128& b) const { return _mm_min_ps(a,b); }};
struct VMax32f { __m128 operator()(const __m128& a, const __m128& b) const { return _mm_max_ps(a,b); }};

typedef MorphRowIVec<VMin8u> ErodeRowVec8u;
typedef MorphRowIVec<VMax8u> DilateRowVec8u;
typedef MorphRowIVec<VMin16u> ErodeRowVec16u;
typedef MorphRowIVec<VMax16u> DilateRowVec16u;
typedef MorphRowIVec<VMin16s> ErodeRowVec16s;
typedef MorphRowIVec<VMax16s> DilateRowVec16s;
typedef MorphRowFVec<VMin32f> ErodeRowVec32f;
typedef MorphRowFVec<VMax32f> DilateRowVec32f;

typedef MorphColumnIVec<VMin8u> ErodeColumnVec8u;
typedef MorphColumnIVec<VMax8u> DilateColumnVec8u;
typedef MorphColumnIVec<VMin16u> ErodeColumnVec16u;
typedef MorphColumnIVec<VMax16u> DilateColumnVec16u;
typedef MorphColumnIVec<VMin16s> ErodeColumnVec16s;
typedef MorphColumnIVec<VMax16s> DilateColumnVec16s;
typedef MorphColumnFVec<VMin32f> ErodeColumnVec32f;
typedef MorphColumnFVec<VMax32f> DilateColumnVec32f;

typedef MorphIVec<VMin8u> ErodeVec8u;
typedef MorphIVec<VMax8u> DilateVec8u;
typedef MorphIVec<VMin16u> ErodeVec16u;
typedef MorphIVec<VMax16u> DilateVec16u;
typedef MorphIVec<VMin16s> ErodeVec16s;
typedef MorphIVec<VMax16s> DilateVec16s;
typedef MorphFVec<VMin32f> ErodeVec32f;
typedef MorphFVec<VMax32f> DilateVec32f;

#else

typedef MorphRowNoVec ErodeRowVec8u;
typedef MorphRowNoVec DilateRowVec8u;

typedef MorphColumnNoVec ErodeColumnVec8u;
typedef MorphColumnNoVec DilateColumnVec8u;

typedef MorphRowNoVec ErodeRowVec16u;
typedef MorphRowNoVec DilateRowVec16u;
typedef MorphRowNoVec ErodeRowVec16s;
typedef MorphRowNoVec DilateRowVec16s;
typedef MorphRowNoVec ErodeRowVec32f;
typedef MorphRowNoVec DilateRowVec32f;

typedef MorphColumnNoVec ErodeColumnVec16u;
typedef MorphColumnNoVec DilateColumnVec16u;
typedef MorphColumnNoVec ErodeColumnVec16s;
typedef MorphColumnNoVec DilateColumnVec16s;
typedef MorphColumnNoVec ErodeColumnVec32f;
typedef MorphColumnNoVec DilateColumnVec32f;

typedef MorphNoVec ErodeVec8u;
typedef MorphNoVec DilateVec8u;
typedef MorphNoVec ErodeVec16u;
typedef MorphNoVec DilateVec16u;
typedef MorphNoVec ErodeVec16s;
typedef MorphNoVec DilateVec16s;
typedef MorphNoVec ErodeVec32f;
typedef MorphNoVec DilateVec32f;

#endif

typedef MorphRowNoVec ErodeRowVec64f;
typedef MorphRowNoVec DilateRowVec64f;
typedef MorphColumnNoVec ErodeColumnVec64f;
typedef MorphColumnNoVec DilateColumnVec64f;
typedef MorphNoVec ErodeVec64f;
typedef MorphNoVec DilateVec64f;


template<class Op, class VecOp> struct MorphRowFilter : public BaseRowFilter
{
    typedef typename Op::rtype T;

    MorphRowFilter( int _ksize, int _anchor ) : vecOp(_ksize, _anchor)
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    void operator()(const uchar* src, uchar* dst, int width, int cn)
    {
        int i, j, k, _ksize = ksize*cn;
        const T* S = (const T*)src;
        Op op;
        T* D = (T*)dst;

        if( _ksize == cn )
        {
            for( i = 0; i < width*cn; i++ )
                D[i] = S[i];
            return;
        }

        int i0 = vecOp(src, dst, width, cn);
        width *= cn;

        for( k = 0; k < cn; k++, S++, D++ )
        {
            for( i = i0; i <= width - cn*2; i += cn*2 )
            {
                const T* s = S + i;
                T m = s[cn];
                for( j = cn*2; j < _ksize; j += cn )
                    m = op(m, s[j]);
                D[i] = op(m, s[0]);
                D[i+cn] = op(m, s[j]);
            }

            for( ; i < width; i += cn )
            {
                const T* s = S + i;
                T m = s[0];
                for( j = cn; j < _ksize; j += cn )
                    m = op(m, s[j]);
                D[i] = m;
            }
        }
    }

    VecOp vecOp;
};


template<class Op, class VecOp> struct MorphColumnFilter : public BaseColumnFilter
{
    typedef typename Op::rtype T;

    MorphColumnFilter( int _ksize, int _anchor ) : vecOp(_ksize, _anchor)
    {
        ksize = _ksize;
        anchor = _anchor;
    }

    void operator()(const uchar** _src, uchar* dst, int dststep, int count, int width)
    {
        int i, k, _ksize = ksize;
        const T** src = (const T**)_src;
        T* D = (T*)dst;
        Op op;

        int i0 = vecOp(_src, dst, dststep, count, width);
        dststep /= sizeof(D[0]);

        for( ; _ksize > 1 && count > 1; count -= 2, D += dststep*2, src += 2 )
        {
            i = i0;
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                const T* sptr = src[1] + i;
                T s0 = sptr[0], s1 = sptr[1], s2 = sptr[2], s3 = sptr[3];

                for( k = 2; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    s0 = op(s0, sptr[0]); s1 = op(s1, sptr[1]);
                    s2 = op(s2, sptr[2]); s3 = op(s3, sptr[3]);
                }

                sptr = src[0] + i;
                D[i] = op(s0, sptr[0]);
                D[i+1] = op(s1, sptr[1]);
                D[i+2] = op(s2, sptr[2]);
                D[i+3] = op(s3, sptr[3]);

                sptr = src[k] + i;
                D[i+dststep] = op(s0, sptr[0]);
                D[i+dststep+1] = op(s1, sptr[1]);
                D[i+dststep+2] = op(s2, sptr[2]);
                D[i+dststep+3] = op(s3, sptr[3]);
            }
            #endif
            for( ; i < width; i++ )
            {
                T s0 = src[1][i];

                for( k = 2; k < _ksize; k++ )
                    s0 = op(s0, src[k][i]);

                D[i] = op(s0, src[0][i]);
                D[i+dststep] = op(s0, src[k][i]);
            }
        }

        for( ; count > 0; count--, D += dststep, src++ )
        {
            i = i0;
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                const T* sptr = src[0] + i;
                T s0 = sptr[0], s1 = sptr[1], s2 = sptr[2], s3 = sptr[3];

                for( k = 1; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    s0 = op(s0, sptr[0]); s1 = op(s1, sptr[1]);
                    s2 = op(s2, sptr[2]); s3 = op(s3, sptr[3]);
                }

                D[i] = s0; D[i+1] = s1;
                D[i+2] = s2; D[i+3] = s3;
            }
            #endif
            for( ; i < width; i++ )
            {
                T s0 = src[0][i];
                for( k = 1; k < _ksize; k++ )
                    s0 = op(s0, src[k][i]);
                D[i] = s0;
            }
        }
    }

    VecOp vecOp;
};


template<class Op, class VecOp> struct MorphFilter : BaseFilter
{
    typedef typename Op::rtype T;

    MorphFilter( const Mat& _kernel, Point _anchor )
    {
        anchor = _anchor;
        ksize = _kernel.size();
        CV_Assert( _kernel.type() == CV_8U );

        std::vector<uchar> coeffs; // we do not really the values of non-zero
        // kernel elements, just their locations
        preprocess2DKernel( _kernel, coords, coeffs );
        ptrs.resize( coords.size() );
    }

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int cn)
    {
        const Point* pt = &coords[0];
        const T** kp = (const T**)&ptrs[0];
        int i, k, nz = (int)coords.size();
        Op op;

        width *= cn;
        for( ; count > 0; count--, dst += dststep, src++ )
        {
            T* D = (T*)dst;

            for( k = 0; k < nz; k++ )
                kp[k] = (const T*)src[pt[k].y] + pt[k].x*cn;

            i = vecOp(&ptrs[0], nz, dst, width);
            #if CV_ENABLE_UNROLLED
            for( ; i <= width - 4; i += 4 )
            {
                const T* sptr = kp[0] + i;
                T s0 = sptr[0], s1 = sptr[1], s2 = sptr[2], s3 = sptr[3];

                for( k = 1; k < nz; k++ )
                {
                    sptr = kp[k] + i;
                    s0 = op(s0, sptr[0]); s1 = op(s1, sptr[1]);
                    s2 = op(s2, sptr[2]); s3 = op(s3, sptr[3]);
                }

                D[i] = s0; D[i+1] = s1;
                D[i+2] = s2; D[i+3] = s3;
            }
            #endif
            for( ; i < width; i++ )
            {
                T s0 = kp[0][i];
                for( k = 1; k < nz; k++ )
                    s0 = op(s0, kp[k][i]);
                D[i] = s0;
            }
        }
    }

    std::vector<Point> coords;
    std::vector<uchar*> ptrs;
    VecOp vecOp;
};

}

/////////////////////////////////// External Interface /////////////////////////////////////

cv::Ptr<cv::BaseRowFilter> cv::getMorphologyRowFilter(int op, int type, int ksize, int anchor)
{
    int depth = CV_MAT_DEPTH(type);
    if( anchor < 0 )
        anchor = ksize/2;
    CV_Assert( op == MORPH_ERODE || op == MORPH_DILATE );
    if( op == MORPH_ERODE )
    {
        if( depth == CV_8U )
            return makePtr<MorphRowFilter<MinOp<uchar>,
                                      ErodeRowVec8u> >(ksize, anchor);
        if( depth == CV_16U )
            return makePtr<MorphRowFilter<MinOp<ushort>,
                                      ErodeRowVec16u> >(ksize, anchor);
        if( depth == CV_16S )
            return makePtr<MorphRowFilter<MinOp<short>,
                                      ErodeRowVec16s> >(ksize, anchor);
        if( depth == CV_32F )
            return makePtr<MorphRowFilter<MinOp<float>,
                                      ErodeRowVec32f> >(ksize, anchor);
        if( depth == CV_64F )
            return makePtr<MorphRowFilter<MinOp<double>,
                                      ErodeRowVec64f> >(ksize, anchor);
    }
    else
    {
        if( depth == CV_8U )
            return makePtr<MorphRowFilter<MaxOp<uchar>,
                                      DilateRowVec8u> >(ksize, anchor);
        if( depth == CV_16U )
            return makePtr<MorphRowFilter<MaxOp<ushort>,
                                      DilateRowVec16u> >(ksize, anchor);
        if( depth == CV_16S )
            return makePtr<MorphRowFilter<MaxOp<short>,
                                      DilateRowVec16s> >(ksize, anchor);
        if( depth == CV_32F )
            return makePtr<MorphRowFilter<MaxOp<float>,
                                      DilateRowVec32f> >(ksize, anchor);
        if( depth == CV_64F )
            return makePtr<MorphRowFilter<MaxOp<double>,
                                      DilateRowVec64f> >(ksize, anchor);
    }

    CV_Error_( CV_StsNotImplemented, ("Unsupported data type (=%d)", type));
    return Ptr<BaseRowFilter>();
}

cv::Ptr<cv::BaseColumnFilter> cv::getMorphologyColumnFilter(int op, int type, int ksize, int anchor)
{
    int depth = CV_MAT_DEPTH(type);
    if( anchor < 0 )
        anchor = ksize/2;
    CV_Assert( op == MORPH_ERODE || op == MORPH_DILATE );
    if( op == MORPH_ERODE )
    {
        if( depth == CV_8U )
            return makePtr<MorphColumnFilter<MinOp<uchar>,
                                         ErodeColumnVec8u> >(ksize, anchor);
        if( depth == CV_16U )
            return makePtr<MorphColumnFilter<MinOp<ushort>,
                                         ErodeColumnVec16u> >(ksize, anchor);
        if( depth == CV_16S )
            return makePtr<MorphColumnFilter<MinOp<short>,
                                         ErodeColumnVec16s> >(ksize, anchor);
        if( depth == CV_32F )
            return makePtr<MorphColumnFilter<MinOp<float>,
                                         ErodeColumnVec32f> >(ksize, anchor);
        if( depth == CV_64F )
            return makePtr<MorphColumnFilter<MinOp<double>,
                                         ErodeColumnVec64f> >(ksize, anchor);
    }
    else
    {
        if( depth == CV_8U )
            return makePtr<MorphColumnFilter<MaxOp<uchar>,
                                         DilateColumnVec8u> >(ksize, anchor);
        if( depth == CV_16U )
            return makePtr<MorphColumnFilter<MaxOp<ushort>,
                                         DilateColumnVec16u> >(ksize, anchor);
        if( depth == CV_16S )
            return makePtr<MorphColumnFilter<MaxOp<short>,
                                         DilateColumnVec16s> >(ksize, anchor);
        if( depth == CV_32F )
            return makePtr<MorphColumnFilter<MaxOp<float>,
                                         DilateColumnVec32f> >(ksize, anchor);
        if( depth == CV_64F )
            return makePtr<MorphColumnFilter<MaxOp<double>,
                                         DilateColumnVec64f> >(ksize, anchor);
    }

    CV_Error_( CV_StsNotImplemented, ("Unsupported data type (=%d)", type));
    return Ptr<BaseColumnFilter>();
}


cv::Ptr<cv::BaseFilter> cv::getMorphologyFilter(int op, int type, InputArray _kernel, Point anchor)
{
    Mat kernel = _kernel.getMat();
    int depth = CV_MAT_DEPTH(type);
    anchor = normalizeAnchor(anchor, kernel.size());
    CV_Assert( op == MORPH_ERODE || op == MORPH_DILATE );
    if( op == MORPH_ERODE )
    {
        if( depth == CV_8U )
            return makePtr<MorphFilter<MinOp<uchar>, ErodeVec8u> >(kernel, anchor);
        if( depth == CV_16U )
            return makePtr<MorphFilter<MinOp<ushort>, ErodeVec16u> >(kernel, anchor);
        if( depth == CV_16S )
            return makePtr<MorphFilter<MinOp<short>, ErodeVec16s> >(kernel, anchor);
        if( depth == CV_32F )
            return makePtr<MorphFilter<MinOp<float>, ErodeVec32f> >(kernel, anchor);
        if( depth == CV_64F )
            return makePtr<MorphFilter<MinOp<double>, ErodeVec64f> >(kernel, anchor);
    }
    else
    {
        if( depth == CV_8U )
            return makePtr<MorphFilter<MaxOp<uchar>, DilateVec8u> >(kernel, anchor);
        if( depth == CV_16U )
            return makePtr<MorphFilter<MaxOp<ushort>, DilateVec16u> >(kernel, anchor);
        if( depth == CV_16S )
            return makePtr<MorphFilter<MaxOp<short>, DilateVec16s> >(kernel, anchor);
        if( depth == CV_32F )
            return makePtr<MorphFilter<MaxOp<float>, DilateVec32f> >(kernel, anchor);
        if( depth == CV_64F )
            return makePtr<MorphFilter<MaxOp<double>, DilateVec64f> >(kernel, anchor);
    }

    CV_Error_( CV_StsNotImplemented, ("Unsupported data type (=%d)", type));
    return Ptr<BaseFilter>();
}


cv::Ptr<cv::FilterEngine> cv::createMorphologyFilter( int op, int type, InputArray _kernel,
                                                      Point anchor, int _rowBorderType, int _columnBorderType,
                                                      const Scalar& _borderValue )
{
    Mat kernel = _kernel.getMat();
    anchor = normalizeAnchor(anchor, kernel.size());

    Ptr<BaseRowFilter> rowFilter;
    Ptr<BaseColumnFilter> columnFilter;
    Ptr<BaseFilter> filter2D;

    if( countNonZero(kernel) == kernel.rows*kernel.cols )
    {
        // rectangular structuring element
        rowFilter = getMorphologyRowFilter(op, type, kernel.cols, anchor.x);
        columnFilter = getMorphologyColumnFilter(op, type, kernel.rows, anchor.y);
    }
    else
        filter2D = getMorphologyFilter(op, type, kernel, anchor);

    Scalar borderValue = _borderValue;
    if( (_rowBorderType == BORDER_CONSTANT || _columnBorderType == BORDER_CONSTANT) &&
            borderValue == morphologyDefaultBorderValue() )
    {
        int depth = CV_MAT_DEPTH(type);
        CV_Assert( depth == CV_8U || depth == CV_16U || depth == CV_16S ||
                   depth == CV_32F || depth == CV_64F );
        if( op == MORPH_ERODE )
            borderValue = Scalar::all( depth == CV_8U ? (double)UCHAR_MAX :
                                       depth == CV_16U ? (double)USHRT_MAX :
                                       depth == CV_16S ? (double)SHRT_MAX :
                                       depth == CV_32F ? (double)FLT_MAX : DBL_MAX);
        else
            borderValue = Scalar::all( depth == CV_8U || depth == CV_16U ?
                                           0. :
                                       depth == CV_16S ? (double)SHRT_MIN :
                                       depth == CV_32F ? (double)-FLT_MAX : -DBL_MAX);
    }

    return makePtr<FilterEngine>(filter2D, rowFilter, columnFilter,
                                 type, type, type, _rowBorderType, _columnBorderType, borderValue );
}


cv::Mat cv::getStructuringElement(int shape, Size ksize, Point anchor)
{
    int i, j;
    int r = 0, c = 0;
    double inv_r2 = 0;

    CV_Assert( shape == MORPH_RECT || shape == MORPH_CROSS || shape == MORPH_ELLIPSE );

    anchor = normalizeAnchor(anchor, ksize);

    if( ksize == Size(1,1) )
        shape = MORPH_RECT;

    if( shape == MORPH_ELLIPSE )
    {
        r = ksize.height/2;
        c = ksize.width/2;
        inv_r2 = r ? 1./((double)r*r) : 0;
    }

    Mat elem(ksize, CV_8U);

    for( i = 0; i < ksize.height; i++ )
    {
        uchar* ptr = elem.ptr(i);
        int j1 = 0, j2 = 0;

        if( shape == MORPH_RECT || (shape == MORPH_CROSS && i == anchor.y) )
            j2 = ksize.width;
        else if( shape == MORPH_CROSS )
            j1 = anchor.x, j2 = j1 + 1;
        else
        {
            int dy = i - r;
            if( std::abs(dy) <= r )
            {
                int dx = saturate_cast<int>(c*std::sqrt((r*r - dy*dy)*inv_r2));
                j1 = std::max( c - dx, 0 );
                j2 = std::min( c + dx + 1, ksize.width );
            }
        }

        for( j = 0; j < j1; j++ )
            ptr[j] = 0;
        for( ; j < j2; j++ )
            ptr[j] = 1;
        for( ; j < ksize.width; j++ )
            ptr[j] = 0;
    }

    return elem;
}

namespace cv
{

// ===== 1. replacement implementation

static bool halMorph(int op, int src_type, int dst_type,
              uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int roi_width, int roi_height, int roi_x, int roi_y,
              int roi_width2, int roi_height2, int roi_x2, int roi_y2,
              int kernel_type, uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height, int anchor_x, int anchor_y,
              int borderType, const double borderValue[4], int iterations, bool isSubmatrix)
{
    cvhalFilter2D * ctx;
    int res = cv_hal_morphInit(&ctx, op, src_type, dst_type, width, height,
                               kernel_type, kernel_data, kernel_step, kernel_width, kernel_height,
                               anchor_x, anchor_y,
                               borderType, borderValue,
                               iterations, isSubmatrix, src_data == dst_data);
    if (res != CV_HAL_ERROR_OK)
        return false;

    res = cv_hal_morph(ctx, src_data, src_step, dst_data, dst_step, width, height,
                       roi_width, roi_height,
                       roi_x, roi_y,
                       roi_width2, roi_height2,
                       roi_x2, roi_y2);
    bool success = (res == CV_HAL_ERROR_OK);

    res = cv_hal_morphFree(ctx);
    if (res != CV_HAL_ERROR_OK)
        return false;

    return success;
}

// ===== 2. IPP implementation
#ifdef HAVE_IPP
#ifdef HAVE_IPP_IW
static inline IwiMorphologyType ippiGetMorphologyType(int morphOp)
{
    return morphOp == MORPH_ERODE ? iwiMorphErode :
        morphOp == MORPH_DILATE   ? iwiMorphDilate :
        morphOp == MORPH_OPEN     ? iwiMorphOpen :
        morphOp == MORPH_CLOSE    ? iwiMorphClose :
        morphOp == MORPH_GRADIENT ? iwiMorphGradient :
        morphOp == MORPH_TOPHAT   ? iwiMorphTophat :
        morphOp == MORPH_BLACKHAT ? iwiMorphBlackhat : (IwiMorphologyType)-1;
}
#endif

static bool ippMorph(int op, int src_type, int dst_type,
              const uchar * src_data, size_t src_step,
              uchar * dst_data, size_t dst_step,
              int width, int height,
              int roi_width, int roi_height, int roi_x, int roi_y,
              int roi_width2, int roi_height2, int roi_x2, int roi_y2,
              int kernel_type, uchar * kernel_data, size_t kernel_step,
              int kernel_width, int kernel_height, int anchor_x, int anchor_y,
              int borderType, const double borderValue[4], int iterations, bool isSubmatrix)
{
#ifdef HAVE_IPP_IW
    CV_INSTRUMENT_REGION_IPP()

#if IPP_VERSION_X100 < 201800
    // Problem with SSE42 optimizations performance
    if(cv::ipp::getIppTopFeatures() == ippCPUID_SSE42)
        return false;

    // Different mask flipping
    if(op == MORPH_GRADIENT)
        return false;
#endif

#if IPP_VERSION_X100 < 201801
    // Problem with AVX512 optimizations performance
    if(cv::ipp::getIppTopFeatures()&ippCPUID_AVX512F)
        return false;

    // Multiple iterations on small mask is not effective in current integration
    // Implace imitation for 3x3 kernel is not efficient
    // Advanced morphology for small mask introduces degradations
    if((iterations > 1 || src_data == dst_data || (op != MORPH_ERODE && op != MORPH_DILATE)) && kernel_width*kernel_height < 25)
        return false;

    // Skip even mask sizes for advanced morphology since they can produce out of spec writes
    if((op != MORPH_ERODE && op != MORPH_DILATE) && (!(kernel_width&1) || !(kernel_height&1)))
        return false;
#endif

    IppAutoBuffer<Ipp8u>        kernelTempBuffer;
    ::ipp::IwiBorderSize        iwBorderSize;
    ::ipp::IwiBorderSize        iwBorderSize2;
    ::ipp::IwiBorderType        iwBorderType;
    ::ipp::IwiBorderType        iwBorderType2;
    ::ipp::IwiImage             iwMask;
    ::ipp::IwiImage             iwInter;
    ::ipp::IwiSize              initSize(width, height);
    ::ipp::IwiSize              kernelSize(kernel_width, kernel_height);
    IppDataType                 type        = ippiGetDataType(CV_MAT_DEPTH(src_type));
    int                         channels    = CV_MAT_CN(src_type);
    IwiMorphologyType           morphType   = ippiGetMorphologyType(op);

    CV_UNUSED(isSubmatrix);

    if(morphType < 0)
        return false;

    if(iterations > 1 && morphType != iwiMorphErode && morphType != iwiMorphDilate)
        return false;

    if(src_type != dst_type)
        return false;

    if(!ippiCheckAnchor(anchor_x, anchor_y, kernel_width, kernel_height))
        return false;

    try
    {
        ::ipp::IwiImage iwSrc(initSize, type, channels, ::ipp::IwiBorderSize(roi_x, roi_y, roi_width-roi_x-width, roi_height-roi_y-height), (void*)src_data, src_step);
        ::ipp::IwiImage iwDst(initSize, type, channels, ::ipp::IwiBorderSize(roi_x2, roi_y2, roi_width2-roi_x2-width, roi_height2-roi_y2-height), (void*)dst_data, dst_step);

        iwBorderSize = ::ipp::iwiSizeToBorderSize(kernelSize);
        iwBorderType = ippiGetBorder(iwSrc, borderType, iwBorderSize);
        if(!iwBorderType)
            return false;
        if(iterations > 1)
        {
            // Check dst border for second and later iterations
            iwBorderSize2 = ::ipp::iwiSizeToBorderSize(kernelSize);
            iwBorderType2 = ippiGetBorder(iwDst, borderType, iwBorderSize2);
            if(!iwBorderType2)
                return false;
        }

        if(morphType != iwiMorphErode && morphType != iwiMorphDilate && morphType != iwiMorphGradient)
        {
            // For now complex morphology support only InMem around all sides. This will be improved later.
            if((iwBorderType&ippBorderInMem) && (iwBorderType&ippBorderInMem) != ippBorderInMem)
                return false;

            if((iwBorderType&ippBorderInMem) == ippBorderInMem)
            {
                iwBorderType &= ~ippBorderInMem;
                iwBorderType &=  ippBorderFirstStageInMem;
            }
        }

        if(iwBorderType.StripFlags() == ippBorderConst)
        {
            if(Vec<double, 4>(borderValue) == morphologyDefaultBorderValue())
                iwBorderType.SetType(ippBorderDefault);
            else
                iwBorderType.m_value = ::ipp::IwValueFloat(borderValue[0], borderValue[1], borderValue[2], borderValue[3]);
        }

        iwMask.Init(ippiSize(kernel_width, kernel_height), ippiGetDataType(CV_MAT_DEPTH(kernel_type)), CV_MAT_CN(kernel_type), 0, kernel_data, kernel_step);

        ::ipp::IwiImage iwMaskLoc = iwMask;
        if(morphType == iwiMorphDilate)
        {
            iwMaskLoc.Alloc(iwMask.m_size, iwMask.m_dataType, iwMask.m_channels);
            ::ipp::iwiMirror(iwMask, iwMaskLoc, ippAxsBoth);
            iwMask = iwMaskLoc;
        }

        if(iterations > 1)
        {
            // OpenCV uses in mem border from dst for two and more iterations, so we need to keep this border in intermediate image
            iwInter.Alloc(initSize, type, channels, iwBorderSize2);

            ::ipp::IwiImage *pSwap[2] = {&iwInter, &iwDst};
            CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, iwSrc, iwInter, morphType, iwMask, ::ipp::IwDefault(), iwBorderType);

            // Copy border only
            {
                if(iwBorderSize2.top)
                {
                    ::ipp::IwiRoi   borderRoi(-iwBorderSize2.left, -iwBorderSize2.top, iwDst.m_size.width+iwBorderSize2.left+iwBorderSize2.right, iwBorderSize2.top);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
                if(iwBorderSize2.bottom)
                {
                    ::ipp::IwiRoi   borderRoi(-iwBorderSize2.left, iwDst.m_size.height, iwDst.m_size.width+iwBorderSize2.left+iwBorderSize2.right, iwBorderSize2.bottom);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
                if(iwBorderSize2.left)
                {
                    ::ipp::IwiRoi   borderRoi(-iwBorderSize2.left, 0, iwBorderSize2.left, iwDst.m_size.height);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
                if(iwBorderSize2.right)
                {
                    ::ipp::IwiRoi   borderRoi(iwDst.m_size.width, 0, iwBorderSize2.left, iwDst.m_size.height);
                    ::ipp::IwiImage iwInterRoi = iwInter.GetRoiImage(borderRoi);
                    ::ipp::iwiCopy(iwDst.GetRoiImage(borderRoi), iwInterRoi);
                }
            }

            iwBorderType2.SetType(iwBorderType);
            for(int i = 0; i < iterations-1; i++)
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, *pSwap[i&0x1], *pSwap[(i+1)&0x1], morphType, iwMask, ::ipp::IwDefault(), iwBorderType2);
            if(iterations&0x1)
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiCopy, iwInter, iwDst);
        }
        else
        {
            if(src_data == dst_data)
            {
                iwInter.Alloc(initSize, type, channels);

                CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, iwSrc, iwInter, morphType, iwMask, ::ipp::IwDefault(), iwBorderType);
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiCopy, iwInter, iwDst);
            }
            else
                CV_INSTRUMENT_FUN_IPP(::ipp::iwiFilterMorphology, iwSrc, iwDst, morphType, iwMask, ::ipp::IwDefault(), iwBorderType);
        }
    }
    catch(::ipp::IwException ex)
    {
        return false;
    }

    return true;
#else
    CV_UNUSED(op); CV_UNUSED(src_type); CV_UNUSED(dst_type); CV_UNUSED(src_data); CV_UNUSED(src_step); CV_UNUSED(dst_data);
    CV_UNUSED(dst_step); CV_UNUSED(width); CV_UNUSED(height); CV_UNUSED(roi_width); CV_UNUSED(roi_height);
    CV_UNUSED(roi_x); CV_UNUSED(roi_y); CV_UNUSED(roi_width2); CV_UNUSED(roi_height2); CV_UNUSED(roi_x2); CV_UNUSED(roi_y2);
    CV_UNUSED(kernel_type); CV_UNUSED(kernel_data); CV_UNUSED(kernel_step); CV_UNUSED(kernel_width); CV_UNUSED(kernel_height);
    CV_UNUSED(anchor_x); CV_UNUSED(anchor_y); CV_UNUSED(borderType); CV_UNUSED(borderValue); CV_UNUSED(iterations);
    CV_UNUSED(isSubmatrix);
    return false;
#endif
};

#endif // HAVE_IPP

// ===== 3. Fallback implementation

static void ocvMorph(int op, int src_type, int dst_type,
                     uchar * src_data, size_t src_step,
                     uchar * dst_data, size_t dst_step,
                     int width, int height,
                     int roi_width, int roi_height, int roi_x, int roi_y,
                     int roi_width2, int roi_height2, int roi_x2, int roi_y2,
                     int kernel_type, uchar * kernel_data, size_t kernel_step,
                     int kernel_width, int kernel_height, int anchor_x, int anchor_y,
                     int borderType, const double borderValue[4], int iterations)
{
    Mat kernel(Size(kernel_width, kernel_height), kernel_type, kernel_data, kernel_step);
    Point anchor(anchor_x, anchor_y);
    Vec<double, 4> borderVal(borderValue);
    Ptr<FilterEngine> f = createMorphologyFilter(op, src_type, kernel, anchor, borderType, borderType, borderVal);
    Mat src(Size(width, height), src_type, src_data, src_step);
    Mat dst(Size(width, height), dst_type, dst_data, dst_step);
    {
        Point ofs(roi_x, roi_y);
        Size wsz(roi_width, roi_height);
        f->apply( src, dst, wsz, ofs );
    }
    {
        Point ofs(roi_x2, roi_y2);
        Size wsz(roi_width2, roi_height2);
        for( int i = 1; i < iterations; i++ )
            f->apply( dst, dst, wsz, ofs );
    }
}


// ===== HAL interface implementation

namespace hal {


CV_DEPRECATED Ptr<Morph> Morph::create(int , int , int , int , int ,
                                int , uchar * , size_t ,
                                int , int ,
                                int , int ,
                                int , const double *,
                                int , bool , bool )  { return Ptr<hal::Morph>(); }


void morph(int op, int src_type, int dst_type,
           uchar * src_data, size_t src_step,
           uchar * dst_data, size_t dst_step,
           int width, int height,
           int roi_width, int roi_height, int roi_x, int roi_y,
           int roi_width2, int roi_height2, int roi_x2, int roi_y2,
           int kernel_type, uchar * kernel_data, size_t kernel_step,
           int kernel_width, int kernel_height, int anchor_x, int anchor_y,
           int borderType, const double borderValue[4], int iterations, bool isSubmatrix)
{
    {
        bool res = halMorph(op, src_type, dst_type, src_data, src_step, dst_data, dst_step, width, height,
                            roi_width, roi_height, roi_x, roi_y,
                            roi_width2, roi_height2, roi_x2, roi_y2,
                            kernel_type, kernel_data, kernel_step,
                            kernel_width, kernel_height, anchor_x, anchor_y,
                            borderType, borderValue, iterations, isSubmatrix);
        if (res)
            return;
    }

    CV_IPP_RUN_FAST(ippMorph(op, src_type, dst_type, src_data, src_step, dst_data, dst_step, width, height,
                            roi_width, roi_height, roi_x, roi_y,
                            roi_width2, roi_height2, roi_x2, roi_y2,
                            kernel_type, kernel_data, kernel_step,
                            kernel_width, kernel_height, anchor_x, anchor_y,
                            borderType, borderValue, iterations, isSubmatrix));

    ocvMorph(op, src_type, dst_type, src_data, src_step, dst_data, dst_step, width, height,
             roi_width, roi_height, roi_x, roi_y,
             roi_width2, roi_height2, roi_x2, roi_y2,
             kernel_type, kernel_data, kernel_step,
             kernel_width, kernel_height, anchor_x, anchor_y,
             borderType, borderValue, iterations);
}

} // cv::hal

#ifdef HAVE_OPENCL

#define ROUNDUP(sz, n)      ((sz) + (n) - 1 - (((sz) + (n) - 1) % (n)))

#ifndef __APPLE__
static bool ocl_morph3x3_8UC1( InputArray _src, OutputArray _dst, InputArray _kernel, Point anchor,
                               int op, int actual_op = -1, InputArray _extraMat = noArray())
{
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Size ksize = _kernel.size();

    Mat kernel8u;
    String processing;

    bool haveExtraMat = !_extraMat.empty();
    CV_Assert(actual_op <= 3 || haveExtraMat);

    _kernel.getMat().convertTo(kernel8u, CV_8U);
    for (int y = 0; y < kernel8u.rows; ++y)
        for (int x = 0; x < kernel8u.cols; ++x)
            if (kernel8u.at<uchar>(y, x) != 0)
                processing += format("PROCESS(%d,%d)", y, x);

    if (anchor.x < 0)
        anchor.x = ksize.width / 2;
    if (anchor.y < 0)
        anchor.y = ksize.height / 2;

    if (actual_op < 0)
        actual_op = op;

    if (type != CV_8UC1 ||
        !((_src.offset() == 0) && (_src.step() % 4 == 0)) ||
        !((_src.cols() % 16 == 0) && (_src.rows() % 2 == 0)) ||
        !(anchor.x == 1 && anchor.y == 1) ||
        !(ksize.width == 3 && ksize.height == 3))
        return false;

    Size size = _src.size();
    size_t globalsize[2] = { 0, 0 };
    size_t localsize[2] = { 0, 0 };

    globalsize[0] = size.width / 16;
    globalsize[1] = size.height / 2;

    static const char * const op2str[] = { "OP_ERODE", "OP_DILATE", NULL, NULL, "OP_GRADIENT", "OP_TOPHAT", "OP_BLACKHAT" };
    String opts = format("-D PROCESS_ELEM_=%s -D %s%s", processing.c_str(), op2str[op],
                         actual_op == op ? "" : cv::format(" -D %s", op2str[actual_op]).c_str());

    ocl::Kernel k;
    k.create("morph3x3_8UC1_cols16_rows2", cv::ocl::imgproc::morph3x3_oclsrc, opts);

    if (k.empty())
        return false;

    UMat src = _src.getUMat();
    _dst.create(size, CV_MAKETYPE(depth, cn));
    if (!(_dst.offset() == 0 && _dst.step() % 4 == 0))
        return false;
    UMat dst = _dst.getUMat();
    UMat extraMat = _extraMat.getUMat();

    int idxArg = k.set(0, ocl::KernelArg::PtrReadOnly(src));
    idxArg = k.set(idxArg, (int)src.step);
    idxArg = k.set(idxArg, ocl::KernelArg::PtrWriteOnly(dst));
    idxArg = k.set(idxArg, (int)dst.step);
    idxArg = k.set(idxArg, (int)dst.rows);
    idxArg = k.set(idxArg, (int)dst.cols);

    if (haveExtraMat)
    {
        idxArg = k.set(idxArg, ocl::KernelArg::ReadOnlyNoSize(extraMat));
    }

    return k.run(2, globalsize, (localsize[0] == 0) ? NULL : localsize, false);
}

static bool ocl_morphSmall( InputArray _src, OutputArray _dst, InputArray _kernel, Point anchor, int borderType,
                            int op, int actual_op = -1, InputArray _extraMat = noArray())
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type), esz = CV_ELEM_SIZE(type);
    bool doubleSupport = dev.doubleFPConfig() > 0;

    if (cn > 4 || (!doubleSupport && depth == CV_64F) ||
        _src.offset() % esz != 0 || _src.step() % esz != 0)
        return false;

    bool haveExtraMat = !_extraMat.empty();
    CV_Assert(actual_op <= 3 || haveExtraMat);

    Size ksize = _kernel.size();
    if (anchor.x < 0)
        anchor.x = ksize.width / 2;
    if (anchor.y < 0)
        anchor.y = ksize.height / 2;

    Size size = _src.size(), wholeSize;
    bool isolated = (borderType & BORDER_ISOLATED) != 0;
    borderType &= ~BORDER_ISOLATED;
    int wdepth = depth, wtype = type;
    if (depth == CV_8U)
    {
        wdepth = CV_32S;
        wtype = CV_MAKETYPE(wdepth, cn);
    }
    char cvt[2][40];

    const char * const borderMap[] = { "BORDER_CONSTANT", "BORDER_REPLICATE",
                                       "BORDER_REFLECT", 0, "BORDER_REFLECT_101" };
    size_t globalsize[2] = { (size_t)size.width, (size_t)size.height };

    UMat src = _src.getUMat();
    if (!isolated)
    {
        Point ofs;
        src.locateROI(wholeSize, ofs);
    }

    int h = isolated ? size.height : wholeSize.height;
    int w = isolated ? size.width : wholeSize.width;
    if (w < ksize.width || h < ksize.height)
        return false;

    // Figure out what vector size to use for loading the pixels.
    int pxLoadNumPixels = cn != 1 || size.width % 4 ? 1 : 4;
    int pxLoadVecSize = cn * pxLoadNumPixels;

    // Figure out how many pixels per work item to compute in X and Y
    // directions.  Too many and we run out of registers.
    int pxPerWorkItemX = 1, pxPerWorkItemY = 1;
    if (cn <= 2 && ksize.width <= 4 && ksize.height <= 4)
    {
        pxPerWorkItemX = size.width % 8 ? size.width % 4 ? size.width % 2 ? 1 : 2 : 4 : 8;
        pxPerWorkItemY = size.height % 2 ? 1 : 2;
    }
    else if (cn < 4 || (ksize.width <= 4 && ksize.height <= 4))
    {
        pxPerWorkItemX = size.width % 2 ? 1 : 2;
        pxPerWorkItemY = size.height % 2 ? 1 : 2;
    }
    globalsize[0] = size.width / pxPerWorkItemX;
    globalsize[1] = size.height / pxPerWorkItemY;

    // Need some padding in the private array for pixels
    int privDataWidth = ROUNDUP(pxPerWorkItemX + ksize.width - 1, pxLoadNumPixels);

    // Make the global size a nice round number so the runtime can pick
    // from reasonable choices for the workgroup size
    const int wgRound = 256;
    globalsize[0] = ROUNDUP(globalsize[0], wgRound);

    if (actual_op < 0)
        actual_op = op;

    // build processing
    String processing;
    Mat kernel8u;
    _kernel.getMat().convertTo(kernel8u, CV_8U);
    for (int y = 0; y < kernel8u.rows; ++y)
        for (int x = 0; x < kernel8u.cols; ++x)
            if (kernel8u.at<uchar>(y, x) != 0)
                processing += format("PROCESS(%d,%d)", y, x);


    static const char * const op2str[] = { "OP_ERODE", "OP_DILATE", NULL, NULL, "OP_GRADIENT", "OP_TOPHAT", "OP_BLACKHAT" };
    String opts = format("-D cn=%d "
            "-D ANCHOR_X=%d -D ANCHOR_Y=%d -D KERNEL_SIZE_X=%d -D KERNEL_SIZE_Y=%d "
            "-D PX_LOAD_VEC_SIZE=%d -D PX_LOAD_NUM_PX=%d -D DEPTH_%d "
            "-D PX_PER_WI_X=%d -D PX_PER_WI_Y=%d -D PRIV_DATA_WIDTH=%d -D %s -D %s "
            "-D PX_LOAD_X_ITERATIONS=%d -D PX_LOAD_Y_ITERATIONS=%d "
            "-D srcT=%s -D srcT1=%s -D dstT=srcT -D dstT1=srcT1 -D WT=%s -D WT1=%s "
            "-D convertToWT=%s -D convertToDstT=%s -D PX_LOAD_FLOAT_VEC_CONV=convert_%s -D PROCESS_ELEM_=%s -D %s%s",
            cn, anchor.x, anchor.y, ksize.width, ksize.height,
            pxLoadVecSize, pxLoadNumPixels, depth,
            pxPerWorkItemX, pxPerWorkItemY, privDataWidth, borderMap[borderType],
            isolated ? "BORDER_ISOLATED" : "NO_BORDER_ISOLATED",
            privDataWidth / pxLoadNumPixels, pxPerWorkItemY + ksize.height - 1,
            ocl::typeToStr(type), ocl::typeToStr(depth),
            haveExtraMat ? ocl::typeToStr(wtype):"srcT",//to prevent overflow - WT
            haveExtraMat ? ocl::typeToStr(wdepth):"srcT1",//to prevent overflow - WT1
            haveExtraMat ? ocl::convertTypeStr(depth, wdepth, cn, cvt[0]) : "noconvert",//to prevent overflow - src to WT
            haveExtraMat ? ocl::convertTypeStr(wdepth, depth, cn, cvt[1]) : "noconvert",//to prevent overflow - WT to dst
            ocl::typeToStr(CV_MAKE_TYPE(haveExtraMat ? wdepth : depth, pxLoadVecSize)), //PX_LOAD_FLOAT_VEC_CONV
            processing.c_str(), op2str[op],
            actual_op == op ? "" : cv::format(" -D %s", op2str[actual_op]).c_str());

    ocl::Kernel kernel("filterSmall", cv::ocl::imgproc::filterSmall_oclsrc, opts);
    if (kernel.empty())
        return false;

    _dst.create(size, type);
    UMat dst = _dst.getUMat();

    UMat source;
    if(src.u != dst.u)
        source = src;
    else
    {
        Point ofs;
        int cols =  src.cols, rows = src.rows;
        src.locateROI(wholeSize, ofs);
        src.adjustROI(ofs.y, wholeSize.height - rows - ofs.y, ofs.x, wholeSize.width - cols - ofs.x);
        src.copyTo(source);

        src.adjustROI(-ofs.y, -wholeSize.height + rows + ofs.y, -ofs.x, -wholeSize.width + cols + ofs.x);
        source.adjustROI(-ofs.y, -wholeSize.height + rows + ofs.y, -ofs.x, -wholeSize.width + cols + ofs.x);
        source.locateROI(wholeSize, ofs);
    }

    UMat extraMat = _extraMat.getUMat();

    int idxArg = kernel.set(0, ocl::KernelArg::PtrReadOnly(source));
    idxArg = kernel.set(idxArg, (int)source.step);
    int srcOffsetX = (int)((source.offset % source.step) / source.elemSize());
    int srcOffsetY = (int)(source.offset / source.step);
    int srcEndX = isolated ? srcOffsetX + size.width : wholeSize.width;
    int srcEndY = isolated ? srcOffsetY + size.height : wholeSize.height;
    idxArg = kernel.set(idxArg, srcOffsetX);
    idxArg = kernel.set(idxArg, srcOffsetY);
    idxArg = kernel.set(idxArg, srcEndX);
    idxArg = kernel.set(idxArg, srcEndY);
    idxArg = kernel.set(idxArg, ocl::KernelArg::WriteOnly(dst));

    if (haveExtraMat)
    {
        idxArg = kernel.set(idxArg, ocl::KernelArg::ReadOnlyNoSize(extraMat));
    }

    return kernel.run(2, globalsize, NULL, false);
}
#endif

static bool ocl_morphOp(InputArray _src, OutputArray _dst, InputArray _kernel,
                        Point anchor, int iterations, int op, int borderType,
                        const Scalar &, int actual_op = -1, InputArray _extraMat = noArray())
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    Mat kernel = _kernel.getMat();
    Size ksize = !kernel.empty() ? kernel.size() : Size(3, 3), ssize = _src.size();

    bool doubleSupport = dev.doubleFPConfig() > 0;
    if ((depth == CV_64F && !doubleSupport) || borderType != BORDER_CONSTANT)
        return false;

    bool haveExtraMat = !_extraMat.empty();
    CV_Assert(actual_op <= 3 || haveExtraMat);

    if (kernel.empty())
    {
        kernel = getStructuringElement(MORPH_RECT, Size(1+iterations*2,1+iterations*2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    else if( iterations > 1 && countNonZero(kernel) == kernel.rows*kernel.cols )
    {
        anchor = Point(anchor.x*iterations, anchor.y*iterations);
        kernel = getStructuringElement(MORPH_RECT,
                                       Size(ksize.width + (iterations-1)*(ksize.width-1),
                                            ksize.height + (iterations-1)*(ksize.height-1)),
                                       anchor);
        iterations = 1;
    }

#ifndef __APPLE__
    int esz = CV_ELEM_SIZE(type);
    // try to use OpenCL kernel adopted for small morph kernel
    if (dev.isIntel() &&
        ((ksize.width < 5 && ksize.height < 5 && esz <= 4) ||
         (ksize.width == 5 && ksize.height == 5 && cn == 1)) &&
         (iterations == 1)
         )
    {
        if (ocl_morph3x3_8UC1(_src, _dst, kernel, anchor, op, actual_op, _extraMat))
            return true;

        if (ocl_morphSmall(_src, _dst, kernel, anchor, borderType, op, actual_op, _extraMat))
            return true;
    }
#endif

    if (iterations == 0 || kernel.rows*kernel.cols == 1)
    {
        _src.copyTo(_dst);
        return true;
    }

#ifdef __ANDROID__
    size_t localThreads[2] = { 16, 8 };
#else
    size_t localThreads[2] = { 16, 16 };
#endif
    size_t globalThreads[2] = { (size_t)ssize.width, (size_t)ssize.height };

#ifdef __APPLE__
    if( actual_op != MORPH_ERODE && actual_op != MORPH_DILATE )
        localThreads[0] = localThreads[1] = 4;
#endif

    if (localThreads[0]*localThreads[1] * 2 < (localThreads[0] + ksize.width - 1) * (localThreads[1] + ksize.height - 1))
        return false;

#ifdef __ANDROID__
    if (dev.isNVidia())
        return false;
#endif

    // build processing
    String processing;
    Mat kernel8u;
    kernel.convertTo(kernel8u, CV_8U);
    for (int y = 0; y < kernel8u.rows; ++y)
        for (int x = 0; x < kernel8u.cols; ++x)
            if (kernel8u.at<uchar>(y, x) != 0)
                processing += format("PROCESS(%d,%d)", y, x);

    static const char * const op2str[] = { "OP_ERODE", "OP_DILATE", NULL, NULL, "OP_GRADIENT", "OP_TOPHAT", "OP_BLACKHAT" };

    char cvt[2][50];
    int wdepth = std::max(depth, CV_32F), scalarcn = cn == 3 ? 4 : cn;

    if (actual_op < 0)
        actual_op = op;

    std::vector<ocl::Kernel> kernels(iterations);
    for (int i = 0; i < iterations; i++)
    {
        int current_op = iterations == i + 1 ? actual_op : op;
        String buildOptions = format("-D RADIUSX=%d -D RADIUSY=%d -D LSIZE0=%d -D LSIZE1=%d -D %s%s"
                                     " -D PROCESS_ELEMS=%s -D T=%s -D DEPTH_%d -D cn=%d -D T1=%s"
                                     " -D convertToWT=%s -D convertToT=%s -D ST=%s%s",
                                     anchor.x, anchor.y, (int)localThreads[0], (int)localThreads[1], op2str[op],
                                     doubleSupport ? " -D DOUBLE_SUPPORT" : "", processing.c_str(),
                                     ocl::typeToStr(type), depth, cn, ocl::typeToStr(depth),
                                     ocl::convertTypeStr(depth, wdepth, cn, cvt[0]),
                                     ocl::convertTypeStr(wdepth, depth, cn, cvt[1]),
                                     ocl::typeToStr(CV_MAKE_TYPE(depth, scalarcn)),
                                     current_op == op ? "" : cv::format(" -D %s", op2str[current_op]).c_str());

        kernels[i].create("morph", ocl::imgproc::morph_oclsrc, buildOptions);
        if (kernels[i].empty())
            return false;
    }

    UMat src = _src.getUMat(), extraMat = _extraMat.getUMat();
    _dst.create(src.size(), src.type());
    UMat dst = _dst.getUMat();

    if (iterations == 1 && src.u != dst.u)
    {
        Size wholesize;
        Point ofs;
        src.locateROI(wholesize, ofs);
        int wholecols = wholesize.width, wholerows = wholesize.height;

        if (haveExtraMat)
            kernels[0].args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnlyNoSize(dst),
                        ofs.x, ofs.y, src.cols, src.rows, wholecols, wholerows,
                        ocl::KernelArg::ReadOnlyNoSize(extraMat));
        else
            kernels[0].args(ocl::KernelArg::ReadOnlyNoSize(src), ocl::KernelArg::WriteOnlyNoSize(dst),
                        ofs.x, ofs.y, src.cols, src.rows, wholecols, wholerows);

        return kernels[0].run(2, globalThreads, localThreads, false);
    }

    for (int i = 0; i < iterations; i++)
    {
        UMat source;
        Size wholesize;
        Point ofs;

        if (i == 0)
        {
            int cols =  src.cols, rows = src.rows;
            src.locateROI(wholesize, ofs);
            src.adjustROI(ofs.y, wholesize.height - rows - ofs.y, ofs.x, wholesize.width - cols - ofs.x);
            if(src.u != dst.u)
                source = src;
            else
                src.copyTo(source);

            src.adjustROI(-ofs.y, -wholesize.height + rows + ofs.y, -ofs.x, -wholesize.width + cols + ofs.x);
            source.adjustROI(-ofs.y, -wholesize.height + rows + ofs.y, -ofs.x, -wholesize.width + cols + ofs.x);
        }
        else
        {
            int cols =  dst.cols, rows = dst.rows;
            dst.locateROI(wholesize, ofs);
            dst.adjustROI(ofs.y, wholesize.height - rows - ofs.y, ofs.x, wholesize.width - cols - ofs.x);
            dst.copyTo(source);
            dst.adjustROI(-ofs.y, -wholesize.height + rows + ofs.y, -ofs.x, -wholesize.width + cols + ofs.x);
            source.adjustROI(-ofs.y, -wholesize.height + rows + ofs.y, -ofs.x, -wholesize.width + cols + ofs.x);
        }
        source.locateROI(wholesize, ofs);

        if (haveExtraMat && iterations == i + 1)
            kernels[i].args(ocl::KernelArg::ReadOnlyNoSize(source), ocl::KernelArg::WriteOnlyNoSize(dst),
                ofs.x, ofs.y, source.cols, source.rows, wholesize.width, wholesize.height,
                ocl::KernelArg::ReadOnlyNoSize(extraMat));
        else
            kernels[i].args(ocl::KernelArg::ReadOnlyNoSize(source), ocl::KernelArg::WriteOnlyNoSize(dst),
                ofs.x, ofs.y, source.cols, source.rows, wholesize.width, wholesize.height);

        if (!kernels[i].run(2, globalThreads, localThreads, false))
            return false;
    }

    return true;
}

#endif

static void morphOp( int op, InputArray _src, OutputArray _dst,
                     InputArray _kernel,
                     Point anchor, int iterations,
                     int borderType, const Scalar& borderValue )
{
    CV_INSTRUMENT_REGION()

    Mat kernel = _kernel.getMat();
    Size ksize = !kernel.empty() ? kernel.size() : Size(3,3);
    anchor = normalizeAnchor(anchor, ksize);

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && _src.channels() <= 4 &&
               borderType == cv::BORDER_CONSTANT && borderValue == morphologyDefaultBorderValue() &&
               (op == MORPH_ERODE || op == MORPH_DILATE) &&
               anchor.x == ksize.width >> 1 && anchor.y == ksize.height >> 1,
               ocl_morphOp(_src, _dst, kernel, anchor, iterations, op, borderType, borderValue) )

    if (iterations == 0 || kernel.rows*kernel.cols == 1)
    {
        _src.copyTo(_dst);
        return;
    }

    if (kernel.empty())
    {
        kernel = getStructuringElement(MORPH_RECT, Size(1+iterations*2,1+iterations*2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    else if( iterations > 1 && countNonZero(kernel) == kernel.rows*kernel.cols )
    {
        anchor = Point(anchor.x*iterations, anchor.y*iterations);
        kernel = getStructuringElement(MORPH_RECT,
                                       Size(ksize.width + (iterations-1)*(ksize.width-1),
                                            ksize.height + (iterations-1)*(ksize.height-1)),
                                       anchor);
        iterations = 1;
    }

    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    Point s_ofs;
    Size s_wsz(src.cols, src.rows);
    Point d_ofs;
    Size d_wsz(dst.cols, dst.rows);
    bool isolated = (borderType&BORDER_ISOLATED)?true:false;
    borderType = (borderType&~BORDER_ISOLATED);

    if(!isolated)
    {
        src.locateROI(s_wsz, s_ofs);
        dst.locateROI(d_wsz, d_ofs);
    }

    hal::morph(op, src.type(), dst.type(),
               src.data, src.step,
               dst.data, dst.step,
               src.cols, src.rows,
               s_wsz.width, s_wsz.height, s_ofs.x, s_ofs.y,
               d_wsz.width, d_wsz.height, d_ofs.x, d_ofs.y,
               kernel.type(), kernel.data, kernel.step, kernel.cols, kernel.rows, anchor.x, anchor.y,
               borderType, borderValue.val, iterations,
               (src.isSubmatrix() && !isolated));
}

}

void cv::erode( InputArray src, OutputArray dst, InputArray kernel,
                Point anchor, int iterations,
                int borderType, const Scalar& borderValue )
{
    CV_INSTRUMENT_REGION()

    morphOp( MORPH_ERODE, src, dst, kernel, anchor, iterations, borderType, borderValue );
}


void cv::dilate( InputArray src, OutputArray dst, InputArray kernel,
                 Point anchor, int iterations,
                 int borderType, const Scalar& borderValue )
{
    CV_INSTRUMENT_REGION()

    morphOp( MORPH_DILATE, src, dst, kernel, anchor, iterations, borderType, borderValue );
}

#ifdef HAVE_OPENCL

namespace cv {

static bool ocl_morphologyEx(InputArray _src, OutputArray _dst, int op,
                             InputArray kernel, Point anchor, int iterations,
                             int borderType, const Scalar& borderValue)
{
    _dst.createSameSize(_src, _src.type());
    bool submat = _dst.isSubmatrix();
    UMat temp;
    _OutputArray _temp = submat ? _dst : _OutputArray(temp);

    switch( op )
    {
    case MORPH_ERODE:
        if (!ocl_morphOp( _src, _dst, kernel, anchor, iterations, MORPH_ERODE, borderType, borderValue ))
            return false;
        break;
    case MORPH_DILATE:
        if (!ocl_morphOp( _src, _dst, kernel, anchor, iterations, MORPH_DILATE, borderType, borderValue ))
            return false;
        break;
    case MORPH_OPEN:
        if (!ocl_morphOp( _src, _temp, kernel, anchor, iterations, MORPH_ERODE, borderType, borderValue ))
            return false;
        if (!ocl_morphOp( _temp, _dst, kernel, anchor, iterations, MORPH_DILATE, borderType, borderValue ))
            return false;
        break;
    case MORPH_CLOSE:
        if (!ocl_morphOp( _src, _temp, kernel, anchor, iterations, MORPH_DILATE, borderType, borderValue ))
            return false;
        if (!ocl_morphOp( _temp, _dst, kernel, anchor, iterations, MORPH_ERODE, borderType, borderValue ))
            return false;
        break;
    case MORPH_GRADIENT:
        if (!ocl_morphOp( _src, temp, kernel, anchor, iterations, MORPH_ERODE, borderType, borderValue ))
            return false;
        if (!ocl_morphOp( _src, _dst, kernel, anchor, iterations, MORPH_DILATE, borderType, borderValue, MORPH_GRADIENT, temp ))
            return false;
        break;
    case MORPH_TOPHAT:
        if (!ocl_morphOp( _src, _temp, kernel, anchor, iterations, MORPH_ERODE, borderType, borderValue ))
            return false;
        if (!ocl_morphOp( _temp, _dst, kernel, anchor, iterations, MORPH_DILATE, borderType, borderValue, MORPH_TOPHAT, _src ))
            return false;
        break;
    case MORPH_BLACKHAT:
        if (!ocl_morphOp( _src, _temp, kernel, anchor, iterations, MORPH_DILATE, borderType, borderValue ))
            return false;
        if (!ocl_morphOp( _temp, _dst, kernel, anchor, iterations, MORPH_ERODE, borderType, borderValue, MORPH_BLACKHAT, _src ))
            return false;
        break;
    default:
        CV_Error( CV_StsBadArg, "unknown morphological operation" );
    }

    return true;
}

}
#endif

#define IPP_DISABLE_MORPH_ADV 1
#ifdef HAVE_IPP
#if !IPP_DISABLE_MORPH_ADV
namespace cv {
static bool ipp_morphologyEx(int op, InputArray _src, OutputArray _dst,
                     InputArray _kernel,
                     Point anchor, int iterations,
                     int borderType, const Scalar& borderValue)
{
#if defined HAVE_IPP_IW
    Mat kernel = _kernel.getMat();
    Size ksize = !kernel.empty() ? kernel.size() : Size(3,3);
    anchor = normalizeAnchor(anchor, ksize);

    if (iterations == 0 || kernel.rows*kernel.cols == 1)
    {
        _src.copyTo(_dst);
        return true;
    }

    if (kernel.empty())
    {
        kernel = getStructuringElement(MORPH_RECT, Size(1+iterations*2,1+iterations*2));
        anchor = Point(iterations, iterations);
        iterations = 1;
    }
    else if( iterations > 1 && countNonZero(kernel) == kernel.rows*kernel.cols )
    {
        anchor = Point(anchor.x*iterations, anchor.y*iterations);
        kernel = getStructuringElement(MORPH_RECT,
                                       Size(ksize.width + (iterations-1)*(ksize.width-1),
                                            ksize.height + (iterations-1)*(ksize.height-1)),
                                       anchor);
        iterations = 1;
    }

    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    Point s_ofs;
    Size s_wsz(src.cols, src.rows);
    Point d_ofs;
    Size d_wsz(dst.cols, dst.rows);
    bool isolated = (borderType&BORDER_ISOLATED)?true:false;
    borderType = (borderType&~BORDER_ISOLATED);

    if(!isolated)
    {
        src.locateROI(s_wsz, s_ofs);
        dst.locateROI(d_wsz, d_ofs);
    }

    return ippMorph(op, src.type(), dst.type(),
               src.data, src.step,
               dst.data, dst.step,
               src.cols, src.rows,
               s_wsz.width, s_wsz.height, s_ofs.x, s_ofs.y,
               d_wsz.width, d_wsz.height, d_ofs.x, d_ofs.y,
               kernel.type(), kernel.data, kernel.step, kernel.cols, kernel.rows, anchor.x, anchor.y,
               borderType, borderValue.val, iterations,
               (src.isSubmatrix() && !isolated));
#else
    CV_UNUSED(op); CV_UNUSED(_src); CV_UNUSED(_dst); CV_UNUSED(_kernel); CV_UNUSED(anchor);
    CV_UNUSED(iterations); CV_UNUSED(borderType); CV_UNUSED(borderValue);
    return false;
#endif
}
}
#endif
#endif

void cv::morphologyEx( InputArray _src, OutputArray _dst, int op,
                       InputArray _kernel, Point anchor, int iterations,
                       int borderType, const Scalar& borderValue )
{
    CV_INSTRUMENT_REGION()

    Mat kernel = _kernel.getMat();
    if (kernel.empty())
    {
        kernel = getStructuringElement(MORPH_RECT, Size(3,3), Point(1,1));
    }
#ifdef HAVE_OPENCL
    Size ksize = kernel.size();
    anchor = normalizeAnchor(anchor, ksize);

    CV_OCL_RUN(_dst.isUMat() && _src.dims() <= 2 && _src.channels() <= 4 &&
        anchor.x == ksize.width >> 1 && anchor.y == ksize.height >> 1 &&
        borderType == cv::BORDER_CONSTANT && borderValue == morphologyDefaultBorderValue(),
        ocl_morphologyEx(_src, _dst, op, kernel, anchor, iterations, borderType, borderValue))
#endif

    Mat src = _src.getMat(), temp;
    _dst.create(src.size(), src.type());
    Mat dst = _dst.getMat();

#if !IPP_DISABLE_MORPH_ADV
    CV_IPP_RUN_FAST(ipp_morphologyEx(op, src, dst, kernel, anchor, iterations, borderType, borderValue));
#endif

    switch( op )
    {
    case MORPH_ERODE:
        erode( src, dst, kernel, anchor, iterations, borderType, borderValue );
        break;
    case MORPH_DILATE:
        dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );
        break;
    case MORPH_OPEN:
        erode( src, dst, kernel, anchor, iterations, borderType, borderValue );
        dilate( dst, dst, kernel, anchor, iterations, borderType, borderValue );
        break;
    case MORPH_CLOSE:
        dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );
        erode( dst, dst, kernel, anchor, iterations, borderType, borderValue );
        break;
    case MORPH_GRADIENT:
        erode( src, temp, kernel, anchor, iterations, borderType, borderValue );
        dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );
        dst -= temp;
        break;
    case MORPH_TOPHAT:
        if( src.data != dst.data )
            temp = dst;
        erode( src, temp, kernel, anchor, iterations, borderType, borderValue );
        dilate( temp, temp, kernel, anchor, iterations, borderType, borderValue );
        dst = src - temp;
        break;
    case MORPH_BLACKHAT:
        if( src.data != dst.data )
            temp = dst;
        dilate( src, temp, kernel, anchor, iterations, borderType, borderValue );
        erode( temp, temp, kernel, anchor, iterations, borderType, borderValue );
        dst = temp - src;
        break;
    case MORPH_HITMISS:
        CV_Assert(src.type() == CV_8UC1);
        if(countNonZero(kernel) <=0)
        {
            src.copyTo(dst);
            break;
        }
        {
            Mat k1, k2, e1, e2;
            k1 = (kernel == 1);
            k2 = (kernel == -1);

            if (countNonZero(k1) <= 0)
                e1 = src;
            else
                erode(src, e1, k1, anchor, iterations, borderType, borderValue);

            Mat src_complement;
            bitwise_not(src, src_complement);
            if (countNonZero(k2) <= 0)
                e2 = src_complement;
            else
                erode(src_complement, e2, k2, anchor, iterations, borderType, borderValue);
            dst = e1 & e2;
        }
        break;
    default:
        CV_Error( CV_StsBadArg, "unknown morphological operation" );
    }
}

CV_IMPL IplConvKernel *
cvCreateStructuringElementEx( int cols, int rows,
                              int anchorX, int anchorY,
                              int shape, int *values )
{
    cv::Size ksize = cv::Size(cols, rows);
    cv::Point anchor = cv::Point(anchorX, anchorY);
    CV_Assert( cols > 0 && rows > 0 && anchor.inside(cv::Rect(0,0,cols,rows)) &&
               (shape != CV_SHAPE_CUSTOM || values != 0));

    int i, size = rows * cols;
    int element_size = sizeof(IplConvKernel) + size*sizeof(int);
    IplConvKernel *element = (IplConvKernel*)cvAlloc(element_size + 32);

    element->nCols = cols;
    element->nRows = rows;
    element->anchorX = anchorX;
    element->anchorY = anchorY;
    element->nShiftR = shape < CV_SHAPE_ELLIPSE ? shape : CV_SHAPE_CUSTOM;
    element->values = (int*)(element + 1);

    if( shape == CV_SHAPE_CUSTOM )
    {
        for( i = 0; i < size; i++ )
            element->values[i] = values[i];
    }
    else
    {
        cv::Mat elem = cv::getStructuringElement(shape, ksize, anchor);
        for( i = 0; i < size; i++ )
            element->values[i] = elem.ptr()[i];
    }

    return element;
}


CV_IMPL void
cvReleaseStructuringElement( IplConvKernel ** element )
{
    if( !element )
        CV_Error( CV_StsNullPtr, "" );
    cvFree( element );
}


static void convertConvKernel( const IplConvKernel* src, cv::Mat& dst, cv::Point& anchor )
{
    if(!src)
    {
        anchor = cv::Point(1,1);
        dst.release();
        return;
    }
    anchor = cv::Point(src->anchorX, src->anchorY);
    dst.create(src->nRows, src->nCols, CV_8U);

    int i, size = src->nRows*src->nCols;
    for( i = 0; i < size; i++ )
        dst.ptr()[i] = (uchar)(src->values[i] != 0);
}


CV_IMPL void
cvErode( const CvArr* srcarr, CvArr* dstarr, IplConvKernel* element, int iterations )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), kernel;
    CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
    cv::Point anchor;
    convertConvKernel( element, kernel, anchor );
    cv::erode( src, dst, kernel, anchor, iterations, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvDilate( const CvArr* srcarr, CvArr* dstarr, IplConvKernel* element, int iterations )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), kernel;
    CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
    cv::Point anchor;
    convertConvKernel( element, kernel, anchor );
    cv::dilate( src, dst, kernel, anchor, iterations, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvMorphologyEx( const void* srcarr, void* dstarr, void*,
                IplConvKernel* element, int op, int iterations )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), kernel;
    CV_Assert( src.size() == dst.size() && src.type() == dst.type() );
    cv::Point anchor;
    IplConvKernel* temp_element = NULL;
    if (!element)
    {
        temp_element = cvCreateStructuringElementEx(3, 3, 1, 1, CV_SHAPE_RECT);
    } else {
        temp_element = element;
    }
    convertConvKernel( temp_element, kernel, anchor );
    if (!element)
    {
        cvReleaseStructuringElement(&temp_element);
    }
    cv::morphologyEx( src, dst, op, kernel, anchor, iterations, cv::BORDER_REPLICATE );
}

/* End of file. */
