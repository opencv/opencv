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
#include "opencv2/core/hal/intrin.hpp"

/****************************************************************************************\
                     Basic Morphological Operations: Erosion & Dilation
\****************************************************************************************/

namespace cv {
CV_CPU_OPTIMIZATION_NAMESPACE_BEGIN
// forward declarations
Ptr<BaseRowFilter> getMorphologyRowFilter(int op, int type, int ksize, int anchor);
Ptr<BaseColumnFilter> getMorphologyColumnFilter(int op, int type, int ksize, int anchor);
Ptr<BaseFilter> getMorphologyFilter(int op, int type, const Mat& kernel, Point anchor);

#ifndef CV_CPU_OPTIMIZATION_DECLARATIONS_ONLY

namespace {
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


#if !defined(CV_SIMD)  // min/max operation are usually fast enough (without using of control flow 'if' statements)

#undef CV_MIN_8U
#undef CV_MAX_8U
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))

template<> inline uchar MinOp<uchar>::operator ()(const uchar a, const uchar b) const { return CV_MIN_8U(a, b); }
template<> inline uchar MaxOp<uchar>::operator ()(const uchar a, const uchar b) const { return CV_MAX_8U(a, b); }

#endif



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

#if CV_SIMD // TODO: enable for CV_SIMD_SCALABLE, GCC 13 related

template<class VecUpdate> struct MorphRowVec
{
    typedef typename VecUpdate::vtype vtype;
    typedef typename VTraits<vtype>::lane_type stype;
    MorphRowVec(int _ksize, int _anchor) : ksize(_ksize), anchor(_anchor) {}
    int operator()(const uchar* src, uchar* dst, int width, int cn) const
    {
        CV_INSTRUMENT_REGION();

        int i, k, _ksize = ksize*cn;
        width *= cn;
        VecUpdate updateOp;

        for( i = 0; i <= width - 4*VTraits<vtype>::vlanes(); i += 4*VTraits<vtype>::vlanes() )
        {
            vtype s0 = vx_load((const stype*)src + i);
            vtype s1 = vx_load((const stype*)src + i + VTraits<vtype>::vlanes());
            vtype s2 = vx_load((const stype*)src + i + 2*VTraits<vtype>::vlanes());
            vtype s3 = vx_load((const stype*)src + i + 3*VTraits<vtype>::vlanes());
            for (k = cn; k < _ksize; k += cn)
            {
                s0 = updateOp(s0, vx_load((const stype*)src + i + k));
                s1 = updateOp(s1, vx_load((const stype*)src + i + k + VTraits<vtype>::vlanes()));
                s2 = updateOp(s2, vx_load((const stype*)src + i + k + 2*VTraits<vtype>::vlanes()));
                s3 = updateOp(s3, vx_load((const stype*)src + i + k + 3*VTraits<vtype>::vlanes()));
            }
            v_store((stype*)dst + i, s0);
            v_store((stype*)dst + i + VTraits<vtype>::vlanes(), s1);
            v_store((stype*)dst + i + 2*VTraits<vtype>::vlanes(), s2);
            v_store((stype*)dst + i + 3*VTraits<vtype>::vlanes(), s3);
        }
        if( i <= width - 2*VTraits<vtype>::vlanes() )
        {
            vtype s0 = vx_load((const stype*)src + i);
            vtype s1 = vx_load((const stype*)src + i + VTraits<vtype>::vlanes());
            for( k = cn; k < _ksize; k += cn )
            {
                s0 = updateOp(s0, vx_load((const stype*)src + i + k));
                s1 = updateOp(s1, vx_load((const stype*)src + i + k + VTraits<vtype>::vlanes()));
            }
            v_store((stype*)dst + i, s0);
            v_store((stype*)dst + i + VTraits<vtype>::vlanes(), s1);
            i += 2*VTraits<vtype>::vlanes();
        }
        if( i <= width - VTraits<vtype>::vlanes() )
        {
            vtype s = vx_load((const stype*)src + i);
            for( k = cn; k < _ksize; k += cn )
                s = updateOp(s, vx_load((const stype*)src + i + k));
            v_store((stype*)dst + i, s);
            i += VTraits<vtype>::vlanes();
        }
        if( i <= width - VTraits<vtype>::vlanes()/2 )
        {
            vtype s = vx_load_low((const stype*)src + i);
            for( k = cn; k < _ksize; k += cn )
                s = updateOp(s, vx_load_low((const stype*)src + i + k));
            v_store_low((stype*)dst + i, s);
            i += VTraits<vtype>::vlanes()/2;
        }

        return i - i % cn;
    }

    int ksize, anchor;
};


template<class VecUpdate> struct MorphColumnVec
{
    typedef typename VecUpdate::vtype vtype;
    typedef typename VTraits<vtype>::lane_type stype;
    MorphColumnVec(int _ksize, int _anchor) : ksize(_ksize), anchor(_anchor) {}
    int operator()(const uchar** _src, uchar* _dst, int dststep, int count, int width) const
    {
        CV_INSTRUMENT_REGION();

        int i = 0, k, _ksize = ksize;
        VecUpdate updateOp;

        for( i = 0; i < count + ksize - 1; i++ )
            CV_Assert( ((size_t)_src[i] & (VTraits<v_uint8>::vlanes()-1)) == 0 );

        const stype** src = (const stype**)_src;
        stype* dst = (stype*)_dst;
        dststep /= sizeof(dst[0]);

        for( ; _ksize > 1 && count > 1; count -= 2, dst += dststep*2, src += 2 )
        {
            for( i = 0; i <= width - 4*VTraits<vtype>::vlanes(); i += 4*VTraits<vtype>::vlanes())
            {
                const stype* sptr = src[1] + i;
                vtype s0 = vx_load_aligned(sptr);
                vtype s1 = vx_load_aligned(sptr + VTraits<vtype>::vlanes());
                vtype s2 = vx_load_aligned(sptr + 2*VTraits<vtype>::vlanes());
                vtype s3 = vx_load_aligned(sptr + 3*VTraits<vtype>::vlanes());

                for( k = 2; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    s0 = updateOp(s0, vx_load_aligned(sptr));
                    s1 = updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes()));
                    s2 = updateOp(s2, vx_load_aligned(sptr + 2*VTraits<vtype>::vlanes()));
                    s3 = updateOp(s3, vx_load_aligned(sptr + 3*VTraits<vtype>::vlanes()));
                }

                sptr = src[0] + i;
                v_store(dst + i, updateOp(s0, vx_load_aligned(sptr)));
                v_store(dst + i + VTraits<vtype>::vlanes(), updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes())));
                v_store(dst + i + 2*VTraits<vtype>::vlanes(), updateOp(s2, vx_load_aligned(sptr + 2*VTraits<vtype>::vlanes())));
                v_store(dst + i + 3*VTraits<vtype>::vlanes(), updateOp(s3, vx_load_aligned(sptr + 3*VTraits<vtype>::vlanes())));

                sptr = src[k] + i;
                v_store(dst + dststep + i, updateOp(s0, vx_load_aligned(sptr)));
                v_store(dst + dststep + i + VTraits<vtype>::vlanes(), updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes())));
                v_store(dst + dststep + i + 2*VTraits<vtype>::vlanes(), updateOp(s2, vx_load_aligned(sptr + 2*VTraits<vtype>::vlanes())));
                v_store(dst + dststep + i + 3*VTraits<vtype>::vlanes(), updateOp(s3, vx_load_aligned(sptr + 3*VTraits<vtype>::vlanes())));
            }
            if( i <= width - 2*VTraits<vtype>::vlanes() )
            {
                const stype* sptr = src[1] + i;
                vtype s0 = vx_load_aligned(sptr);
                vtype s1 = vx_load_aligned(sptr + VTraits<vtype>::vlanes());

                for( k = 2; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    s0 = updateOp(s0, vx_load_aligned(sptr));
                    s1 = updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes()));
                }

                sptr = src[0] + i;
                v_store(dst + i, updateOp(s0, vx_load_aligned(sptr)));
                v_store(dst + i + VTraits<vtype>::vlanes(), updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes())));

                sptr = src[k] + i;
                v_store(dst + dststep + i, updateOp(s0, vx_load_aligned(sptr)));
                v_store(dst + dststep + i + VTraits<vtype>::vlanes(), updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes())));
                i += 2*VTraits<vtype>::vlanes();
            }
            if( i <= width - VTraits<vtype>::vlanes() )
            {
                vtype s0 = vx_load_aligned(src[1] + i);

                for( k = 2; k < _ksize; k++ )
                    s0 = updateOp(s0, vx_load_aligned(src[k] + i));

                v_store(dst + i, updateOp(s0, vx_load_aligned(src[0] + i)));
                v_store(dst + dststep + i, updateOp(s0, vx_load_aligned(src[k] + i)));
                i += VTraits<vtype>::vlanes();
            }
            if( i <= width - VTraits<vtype>::vlanes()/2 )
            {
                vtype s0 = vx_load_low(src[1] + i);

                for( k = 2; k < _ksize; k++ )
                    s0 = updateOp(s0, vx_load_low(src[k] + i));

                v_store_low(dst + i, updateOp(s0, vx_load_low(src[0] + i)));
                v_store_low(dst + dststep + i, updateOp(s0, vx_load_low(src[k] + i)));
                i += VTraits<vtype>::vlanes()/2;
            }
        }

        for( ; count > 0; count--, dst += dststep, src++ )
        {
            for( i = 0; i <= width - 4*VTraits<vtype>::vlanes(); i += 4*VTraits<vtype>::vlanes())
            {
                const stype* sptr = src[0] + i;
                vtype s0 = vx_load_aligned(sptr);
                vtype s1 = vx_load_aligned(sptr + VTraits<vtype>::vlanes());
                vtype s2 = vx_load_aligned(sptr + 2*VTraits<vtype>::vlanes());
                vtype s3 = vx_load_aligned(sptr + 3*VTraits<vtype>::vlanes());

                for( k = 1; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    s0 = updateOp(s0, vx_load_aligned(sptr));
                    s1 = updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes()));
                    s2 = updateOp(s2, vx_load_aligned(sptr + 2*VTraits<vtype>::vlanes()));
                    s3 = updateOp(s3, vx_load_aligned(sptr + 3*VTraits<vtype>::vlanes()));
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<vtype>::vlanes(), s1);
                v_store(dst + i + 2*VTraits<vtype>::vlanes(), s2);
                v_store(dst + i + 3*VTraits<vtype>::vlanes(), s3);
            }
            if( i <= width - 2*VTraits<vtype>::vlanes() )
            {
                const stype* sptr = src[0] + i;
                vtype s0 = vx_load_aligned(sptr);
                vtype s1 = vx_load_aligned(sptr + VTraits<vtype>::vlanes());

                for( k = 1; k < _ksize; k++ )
                {
                    sptr = src[k] + i;
                    s0 = updateOp(s0, vx_load_aligned(sptr));
                    s1 = updateOp(s1, vx_load_aligned(sptr + VTraits<vtype>::vlanes()));
                }
                v_store(dst + i, s0);
                v_store(dst + i + VTraits<vtype>::vlanes(), s1);
                i += 2*VTraits<vtype>::vlanes();
            }
            if( i <= width - VTraits<vtype>::vlanes() )
            {
                vtype s0 = vx_load_aligned(src[0] + i);

                for( k = 1; k < _ksize; k++ )
                    s0 = updateOp(s0, vx_load_aligned(src[k] + i));
                v_store(dst + i, s0);
                i += VTraits<vtype>::vlanes();
            }
            if( i <= width - VTraits<vtype>::vlanes()/2 )
            {
                vtype s0 = vx_load_low(src[0] + i);

                for( k = 1; k < _ksize; k++ )
                    s0 = updateOp(s0, vx_load_low(src[k] + i));
                v_store_low(dst + i, s0);
                i += VTraits<vtype>::vlanes()/2;
            }
        }

        return i;
    }

    int ksize, anchor;
};


template<class VecUpdate> struct MorphVec
{
    typedef typename VecUpdate::vtype vtype;
    typedef typename VTraits<vtype>::lane_type stype;
    int operator()(uchar** _src, int nz, uchar* _dst, int width) const
    {
        CV_INSTRUMENT_REGION();

        const stype** src = (const stype**)_src;
        stype* dst = (stype*)_dst;
        int i, k;
        VecUpdate updateOp;

        for( i = 0; i <= width - 4*VTraits<vtype>::vlanes(); i += 4*VTraits<vtype>::vlanes() )
        {
            const stype* sptr = src[0] + i;
            vtype s0 = vx_load(sptr);
            vtype s1 = vx_load(sptr + VTraits<vtype>::vlanes());
            vtype s2 = vx_load(sptr + 2*VTraits<vtype>::vlanes());
            vtype s3 = vx_load(sptr + 3*VTraits<vtype>::vlanes());
            for( k = 1; k < nz; k++ )
            {
                sptr = src[k] + i;
                s0 = updateOp(s0, vx_load(sptr));
                s1 = updateOp(s1, vx_load(sptr + VTraits<vtype>::vlanes()));
                s2 = updateOp(s2, vx_load(sptr + 2*VTraits<vtype>::vlanes()));
                s3 = updateOp(s3, vx_load(sptr + 3*VTraits<vtype>::vlanes()));
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<vtype>::vlanes(), s1);
            v_store(dst + i + 2*VTraits<vtype>::vlanes(), s2);
            v_store(dst + i + 3*VTraits<vtype>::vlanes(), s3);
        }
        if( i <= width - 2*VTraits<vtype>::vlanes() )
        {
            const stype* sptr = src[0] + i;
            vtype s0 = vx_load(sptr);
            vtype s1 = vx_load(sptr + VTraits<vtype>::vlanes());
            for( k = 1; k < nz; k++ )
            {
                sptr = src[k] + i;
                s0 = updateOp(s0, vx_load(sptr));
                s1 = updateOp(s1, vx_load(sptr + VTraits<vtype>::vlanes()));
            }
            v_store(dst + i, s0);
            v_store(dst + i + VTraits<vtype>::vlanes(), s1);
            i += 2*VTraits<vtype>::vlanes();
        }
        if( i <= width - VTraits<vtype>::vlanes() )
        {
            vtype s0 = vx_load(src[0] + i);
            for( k = 1; k < nz; k++ )
                s0 = updateOp(s0, vx_load(src[k] + i));
            v_store(dst + i, s0);
            i += VTraits<vtype>::vlanes();
        }
        if( i <= width - VTraits<vtype>::vlanes()/2 )
        {
            vtype s0 = vx_load_low(src[0] + i);
            for( k = 1; k < nz; k++ )
                s0 = updateOp(s0, vx_load_low(src[k] + i));
            v_store_low(dst + i, s0);
            i += VTraits<vtype>::vlanes()/2;
        }
        return i;
    }
};

template <typename T> struct VMin
{
    typedef T vtype;
    vtype operator()(const vtype& a, const vtype& b) const { return v_min(a,b); }
};
template <typename T> struct VMax
{
    typedef T vtype;
    vtype operator()(const vtype& a, const vtype& b) const { return v_max(a,b); }
};

typedef MorphRowVec<VMin<v_uint8> > ErodeRowVec8u;
typedef MorphRowVec<VMax<v_uint8> > DilateRowVec8u;
typedef MorphRowVec<VMin<v_uint16> > ErodeRowVec16u;
typedef MorphRowVec<VMax<v_uint16> > DilateRowVec16u;
typedef MorphRowVec<VMin<v_int16> > ErodeRowVec16s;
typedef MorphRowVec<VMax<v_int16> > DilateRowVec16s;
typedef MorphRowVec<VMin<v_float32> > ErodeRowVec32f;
typedef MorphRowVec<VMax<v_float32> > DilateRowVec32f;

typedef MorphColumnVec<VMin<v_uint8> > ErodeColumnVec8u;
typedef MorphColumnVec<VMax<v_uint8> > DilateColumnVec8u;
typedef MorphColumnVec<VMin<v_uint16> > ErodeColumnVec16u;
typedef MorphColumnVec<VMax<v_uint16> > DilateColumnVec16u;
typedef MorphColumnVec<VMin<v_int16> > ErodeColumnVec16s;
typedef MorphColumnVec<VMax<v_int16> > DilateColumnVec16s;
typedef MorphColumnVec<VMin<v_float32> > ErodeColumnVec32f;
typedef MorphColumnVec<VMax<v_float32> > DilateColumnVec32f;

typedef MorphVec<VMin<v_uint8> > ErodeVec8u;
typedef MorphVec<VMax<v_uint8> > DilateVec8u;
typedef MorphVec<VMin<v_uint16> > ErodeVec16u;
typedef MorphVec<VMax<v_uint16> > DilateVec16u;
typedef MorphVec<VMin<v_int16> > ErodeVec16s;
typedef MorphVec<VMax<v_int16> > DilateVec16s;
typedef MorphVec<VMin<v_float32> > ErodeVec32f;
typedef MorphVec<VMax<v_float32> > DilateVec32f;

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

    void operator()(const uchar* src, uchar* dst, int width, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

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

    void operator()(const uchar** _src, uchar* dst, int dststep, int count, int width) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

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

    void operator()(const uchar** src, uchar* dst, int dststep, int count, int width, int cn) CV_OVERRIDE
    {
        CV_INSTRUMENT_REGION();

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

} // namespace anon

/////////////////////////////////// External Interface /////////////////////////////////////

Ptr<BaseRowFilter> getMorphologyRowFilter(int op, int type, int ksize, int anchor)
{
    CV_INSTRUMENT_REGION();

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

    CV_Error_( cv::Error::StsNotImplemented, ("Unsupported data type (=%d)", type));
}

Ptr<BaseColumnFilter> getMorphologyColumnFilter(int op, int type, int ksize, int anchor)
{
    CV_INSTRUMENT_REGION();

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

    CV_Error_( cv::Error::StsNotImplemented, ("Unsupported data type (=%d)", type));
}

Ptr<BaseFilter> getMorphologyFilter(int op, int type, const Mat& kernel, Point anchor)
{
    CV_INSTRUMENT_REGION();

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

    CV_Error_( cv::Error::StsNotImplemented, ("Unsupported data type (=%d)", type));
}

#endif
CV_CPU_OPTIMIZATION_NAMESPACE_END
} // namespace
