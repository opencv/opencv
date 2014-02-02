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
#include "opencl_kernels.hpp"

/****************************************************************************************\
                     Basic Morphological Operations: Erosion & Dilation
\****************************************************************************************/

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

#ifdef HAVE_TEGRA_OPTIMIZATION
using tegra::ErodeRowVec8u;
using tegra::DilateRowVec8u;

using tegra::ErodeColumnVec8u;
using tegra::DilateColumnVec8u;
#else
typedef MorphRowNoVec ErodeRowVec8u;
typedef MorphRowNoVec DilateRowVec8u;

typedef MorphColumnNoVec ErodeColumnVec8u;
typedef MorphColumnNoVec DilateColumnVec8u;
#endif

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
        uchar* ptr = elem.data + i*elem.step;
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

class MorphologyRunner : public ParallelLoopBody
{
public:
    MorphologyRunner(Mat _src, Mat _dst, int _nStripes, int _iterations,
                     int _op, Mat _kernel, Point _anchor,
                     int _rowBorderType, int _columnBorderType, const Scalar& _borderValue) :
        borderValue(_borderValue)
    {
        src = _src;
        dst = _dst;

        nStripes = _nStripes;
        iterations = _iterations;

        op = _op;
        kernel = _kernel;
        anchor = _anchor;
        rowBorderType = _rowBorderType;
        columnBorderType = _columnBorderType;
    }

    void operator () ( const Range& range ) const
    {
        int row0 = std::min(cvRound(range.start * src.rows / nStripes), src.rows);
        int row1 = std::min(cvRound(range.end * src.rows / nStripes), src.rows);

        /*if(0)
            printf("Size = (%d, %d), range[%d,%d), row0 = %d, row1 = %d\n",
                   src.rows, src.cols, range.start, range.end, row0, row1);*/

        Mat srcStripe = src.rowRange(row0, row1);
        Mat dstStripe = dst.rowRange(row0, row1);

        Ptr<FilterEngine> f = createMorphologyFilter(op, src.type(), kernel, anchor,
                                                     rowBorderType, columnBorderType, borderValue );

        f->apply( srcStripe, dstStripe );
        for( int i = 1; i < iterations; i++ )
            f->apply( dstStripe, dstStripe );
    }

private:
    Mat src;
    Mat dst;
    int nStripes;
    int iterations;

    int op;
    Mat kernel;
    Point anchor;
    int rowBorderType;
    int columnBorderType;
    Scalar borderValue;
};

#if IPP_VERSION_X100 >= 801
static bool IPPMorphReplicate(int op, const Mat &src, Mat &dst, const Mat &kernel,
                              const Size& ksize, const Point &anchor, bool rectKernel)
{
    int type = src.type();
    const Mat* _src = &src;
    Mat temp;
    if (src.data == dst.data)
    {
        src.copyTo(temp);
        _src = &temp;
    }

    IppiSize roiSize = {src.cols, src.rows};
    IppiSize kernelSize = {ksize.width, ksize.height};

    if (!rectKernel)
    {
#if 1
        if (((kernel.cols - 1) / 2 != anchor.x) || ((kernel.rows - 1) / 2 != anchor.y))
            return false;
        #define IPP_MORPH_CASE(cvtype, flavor, data_type) \
        case cvtype: \
            {\
                int specSize = 0, bufferSize = 0;\
                if (0 > ippiMorphologyBorderGetSize_##flavor(roiSize.width, kernelSize, &specSize, &bufferSize))\
                    return false;\
                IppiMorphState *pSpec = (IppiMorphState*)ippMalloc(specSize);\
                Ipp8u *pBuffer = (Ipp8u*)ippMalloc(bufferSize);\
                if (0 > ippiMorphologyBorderInit_##flavor(roiSize.width, kernel.data, kernelSize, pSpec, pBuffer))\
                {\
                    ippFree(pBuffer);\
                    ippFree(pSpec);\
                    return false;\
                }\
                bool ok = false;\
                if (op == MORPH_ERODE)\
                    ok = (0 <= ippiErodeBorder_##flavor((Ipp##data_type *)_src->data, (int)_src->step[0], (Ipp##data_type *)dst.data, (int)dst.step[0],\
                                            roiSize, ippBorderRepl, 0, pSpec, pBuffer));\
                else\
                    ok = (0 <= ippiDilateBorder_##flavor((Ipp##data_type *)_src->data, (int)_src->step[0], (Ipp##data_type *)dst.data, (int)dst.step[0],\
                                            roiSize, ippBorderRepl, 0, pSpec, pBuffer));\
                ippFree(pBuffer);\
                ippFree(pSpec);\
                return ok;\
            }\
            break;
#else
        IppiPoint point = {anchor.x, anchor.y};
        // this is case, which can be used with the anchor not in center of the kernel, but
        // ippiMorphologyBorderGetSize_, ippiErodeBorderReplicate_ and ippiDilateBorderReplicate_ are deprecated.
        #define IPP_MORPH_CASE(cvtype, flavor, data_type) \
        case cvtype: \
            {\
                int specSize = 0;\
                int bufferSize = 0;\
                if (0 > ippiMorphologyGetSize_##flavor( roiSize.width, kernel.data kernelSize, &specSize))\
                    return false;\
                bool ok = false;\
                IppiMorphState* pState = (IppiMorphState*)ippMalloc(specSize);\
                if (ippiMorphologyInit_##flavor(roiSize.width, kernel.data, kernelSize, point, pState) >= 0)\
                {\
                    if (op == MORPH_ERODE)\
                        ok = ippiErodeBorderReplicate_##flavor((Ipp##data_type *)_src->data, (int)_src->step[0],\
                            (Ipp##data_type *)dst.data, (int)dst.step[0],\
                            roiSize, ippBorderRepl, pState ) >= 0;\
                    else\
                        ok = ippiDilateBorderReplicate_##flavor((Ipp##data_type *)_src->data, (int)_src->step[0],\
                            (Ipp##data_type *)dst.data, (int)dst.step[0],\
                            roiSize, ippBorderRepl, pState ) >= 0;\
                }\
                ippFree(pState);\
                return ok;\
            }\
            break;
#endif
        switch (type)
        {
        IPP_MORPH_CASE(CV_8UC1, 8u_C1R, 8u);
        IPP_MORPH_CASE(CV_8UC3, 8u_C3R, 8u);
        IPP_MORPH_CASE(CV_8UC4, 8u_C4R, 8u);
        IPP_MORPH_CASE(CV_32FC1, 32f_C1R, 32f);
        IPP_MORPH_CASE(CV_32FC3, 32f_C3R, 32f);
        IPP_MORPH_CASE(CV_32FC4, 32f_C4R, 32f);
        default:
            return false;
        }

        #undef IPP_MORPH_CASE
    }
    else
    {
        IppiPoint point = {anchor.x, anchor.y};

        #define IPP_MORPH_CASE(cvtype, flavor, data_type) \
        case cvtype: \
            {\
                int bufSize = 0;\
                if (0 > ippiFilterMinGetBufferSize_##flavor(src.cols, kernelSize, &bufSize))\
                    return false;\
                AutoBuffer<uchar> buf(bufSize + 64);\
                uchar* buffer = alignPtr((uchar*)buf, 32);\
                if (op == MORPH_ERODE)\
                    return (0 <= ippiFilterMinBorderReplicate_##flavor((Ipp##data_type *)_src->data, (int)_src->step[0], (Ipp##data_type *)dst.data, (int)dst.step[0], roiSize, kernelSize, point, buffer));\
                return (0 <= ippiFilterMaxBorderReplicate_##flavor((Ipp##data_type *)_src->data, (int)_src->step[0], (Ipp##data_type *)dst.data, (int)dst.step[0], roiSize, kernelSize, point, buffer));\
            }\
            break;

        switch (type)
        {
        IPP_MORPH_CASE(CV_8UC1, 8u_C1R, 8u);
        IPP_MORPH_CASE(CV_8UC3, 8u_C3R, 8u);
        IPP_MORPH_CASE(CV_8UC4, 8u_C4R, 8u);
        IPP_MORPH_CASE(CV_32FC1, 32f_C1R, 32f);
        IPP_MORPH_CASE(CV_32FC3, 32f_C3R, 32f);
        IPP_MORPH_CASE(CV_32FC4, 32f_C4R, 32f);
        default:
            return false;
        }
        #undef IPP_MORPH_CASE

#if defined(__GNUC__) && __GNUC__ == 4 && __GNUC_MINOR__ >= 8
        return false; /// It disables false positive warning in GCC 4.8 and further
#endif
    }
}

static bool IPPMorphOp(int op, InputArray _src, OutputArray _dst,
    const Mat& _kernel, Point anchor, int iterations,
    int borderType, const Scalar &borderValue)
{
    Mat src = _src.getMat(), kernel = _kernel;
    int type = src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);

    if( !( depth == CV_8U || depth == CV_32F ) || !(cn == 1 || cn == 3 || cn == 4) ||
        !( borderType == cv::BORDER_REPLICATE || (borderType == cv::BORDER_CONSTANT && borderValue == morphologyDefaultBorderValue()) )
        || !( op == MORPH_DILATE || op == MORPH_ERODE) || _src.isSubmatrix() )
        return false;

    if( borderType == cv::BORDER_CONSTANT && kernel.data )
    {
        int x, y;
        for( y = 0; y < kernel.rows; y++ )
        {
            if( kernel.at<uchar>(y, anchor.x) != 0 )
                continue;
            for( x = 0; x < kernel.cols; x++ )
            {
                if( kernel.at<uchar>(y,x) != 0 )
                    return false;
            }
        }
        for( x = 0; x < kernel.cols; x++ )
        {
            if( kernel.at<uchar>(anchor.y, x) != 0 )
                continue;
            for( y = 0; y < kernel.rows; y++ )
            {
                if( kernel.at<uchar>(y,x) != 0 )
                    return false;
            }
        }

    }
    Size ksize = kernel.data ? kernel.size() : Size(3,3);

    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    if( iterations == 0 || kernel.rows*kernel.cols == 1 )
    {
        src.copyTo(dst);
        return true;
    }

    bool rectKernel = false;
    if( !kernel.data )
    {
        ksize = Size(1+iterations*2,1+iterations*2);
        anchor = Point(iterations, iterations);
        rectKernel = true;
        iterations = 1;
    }
    else if( iterations >= 1 && countNonZero(kernel) == kernel.rows*kernel.cols )
    {
        ksize = Size(ksize.width + (iterations-1)*(ksize.width-1),
             ksize.height + (iterations-1)*(ksize.height-1)),
        anchor = Point(anchor.x*iterations, anchor.y*iterations);
        kernel = Mat();
        rectKernel = true;
        iterations = 1;
    }

    // TODO: implement the case of iterations > 1.
    if( iterations > 1 )
        return false;

    return IPPMorphReplicate( op, src, dst, kernel, ksize, anchor, rectKernel );
}
#endif

#ifdef HAVE_OPENCL

static bool ocl_morphOp(InputArray _src, OutputArray _dst, InputArray _kernel,
                        Point anchor, int iterations, int op, int borderType,
                        const Scalar &, int actual_op = -1, InputArray _extraMat = noArray())
{
    const ocl::Device & dev = ocl::Device::getDefault();
    int type = _src.type(), depth = CV_MAT_DEPTH(type), cn = CV_MAT_CN(type);
    bool doubleSupport = dev.doubleFPConfig() > 0;

    if ((depth == CV_64F && !doubleSupport) || borderType != BORDER_CONSTANT)
        return false;

    Mat kernel = _kernel.getMat();
    bool haveExtraMat = !_extraMat.empty();
    Size ksize = kernel.data ? kernel.size() : Size(3, 3), ssize = _src.size();
    CV_Assert(actual_op <= 3 || haveExtraMat);

    if (iterations == 0 || kernel.rows*kernel.cols == 1)
    {
        _src.copyTo(_dst);
        return true;
    }

    if (!kernel.data)
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

#ifdef ANDROID
    size_t localThreads[2] = { 16, 8 };
#else
    size_t localThreads[2] = { 16, 16 };
#endif
    size_t globalThreads[2] = { ssize.width, ssize.height };

    if (localThreads[0]*localThreads[1] * 2 < (localThreads[0] + ksize.width - 1) * (localThreads[1] + ksize.height - 1))
        return false;

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
    Mat kernel = _kernel.getMat();
    Size ksize = kernel.data ? kernel.size() : Size(3,3);
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

    if (!kernel.data)
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

#if IPP_VERSION_X100 >= 801
    if( IPPMorphOp(op, _src, _dst, kernel, anchor, iterations, borderType, borderValue) )
        return;
#endif

    Mat src = _src.getMat();
    _dst.create( src.size(), src.type() );
    Mat dst = _dst.getMat();

    int nStripes = 1;
#if defined HAVE_TEGRA_OPTIMIZATION
    if (src.data != dst.data && iterations == 1 &&  //NOTE: threads are not used for inplace processing
        (borderType & BORDER_ISOLATED) == 0 && //TODO: check border types
        src.rows >= 64 ) //NOTE: just heuristics
        nStripes = 4;
#endif

    parallel_for_(Range(0, nStripes),
                  MorphologyRunner(src, dst, nStripes, iterations, op, kernel, anchor, borderType, borderType, borderValue));
}

}

void cv::erode( InputArray src, OutputArray dst, InputArray kernel,
                Point anchor, int iterations,
                int borderType, const Scalar& borderValue )
{
    morphOp( MORPH_ERODE, src, dst, kernel, anchor, iterations, borderType, borderValue );
}


void cv::dilate( InputArray src, OutputArray dst, InputArray kernel,
                 Point anchor, int iterations,
                 int borderType, const Scalar& borderValue )
{
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

void cv::morphologyEx( InputArray _src, OutputArray _dst, int op,
                       InputArray kernel, Point anchor, int iterations,
                       int borderType, const Scalar& borderValue )
{
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
    case CV_MOP_CLOSE:
        dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );
        erode( dst, dst, kernel, anchor, iterations, borderType, borderValue );
        break;
    case CV_MOP_GRADIENT:
        erode( src, temp, kernel, anchor, iterations, borderType, borderValue );
        dilate( src, dst, kernel, anchor, iterations, borderType, borderValue );
        dst -= temp;
        break;
    case CV_MOP_TOPHAT:
        if( src.data != dst.data )
            temp = dst;
        erode( src, temp, kernel, anchor, iterations, borderType, borderValue );
        dilate( temp, temp, kernel, anchor, iterations, borderType, borderValue );
        dst = src - temp;
        break;
    case CV_MOP_BLACKHAT:
        if( src.data != dst.data )
            temp = dst;
        dilate( src, temp, kernel, anchor, iterations, borderType, borderValue );
        erode( temp, temp, kernel, anchor, iterations, borderType, borderValue );
        dst = temp - src;
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
            element->values[i] = elem.data[i];
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
        dst.data[i] = (uchar)(src->values[i] != 0);
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
