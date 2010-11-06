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

#ifndef __OPENCV_PRECOMP_H__
#define __OPENCV_PRECOMP_H__

#if defined _MSC_VER && _MSC_VER >= 1200
    // disable warnings related to inline functions
    #pragma warning( disable: 4251 4711 4710 4514 )
#endif

#ifdef HAVE_CONFIG_H 
#include <cvconfig.h> 
#endif

#include "opencv2/core/core.hpp"
#include "opencv2/core/core_c.h"
#include "opencv2/core/internal.hpp"

#include <assert.h>
#include <ctype.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CV_MEMCPY_CHAR( dst, src, len )                 \
{                                                       \
    size_t _icv_memcpy_i_, _icv_memcpy_len_ = (len);    \
    char* _icv_memcpy_dst_ = (char*)(dst);              \
    const char* _icv_memcpy_src_ = (const char*)(src);  \
                                                        \
    for( _icv_memcpy_i_ = 0; _icv_memcpy_i_ < _icv_memcpy_len_; _icv_memcpy_i_++ )  \
        _icv_memcpy_dst_[_icv_memcpy_i_] = _icv_memcpy_src_[_icv_memcpy_i_];        \
}


#define CV_MEMCPY_INT( dst, src, len )                  \
{                                                       \
    size_t _icv_memcpy_i_, _icv_memcpy_len_ = (len);    \
    int* _icv_memcpy_dst_ = (int*)(dst);                \
    const int* _icv_memcpy_src_ = (const int*)(src);    \
    assert( ((size_t)_icv_memcpy_src_&(sizeof(int)-1)) == 0 && \
    ((size_t)_icv_memcpy_dst_&(sizeof(int)-1)) == 0 );  \
                                                        \
    for(_icv_memcpy_i_=0;_icv_memcpy_i_<_icv_memcpy_len_;_icv_memcpy_i_++)  \
        _icv_memcpy_dst_[_icv_memcpy_i_] = _icv_memcpy_src_[_icv_memcpy_i_];\
}


#define CV_MEMCPY_AUTO( dst, src, len )                                             \
{                                                                                   \
    size_t _icv_memcpy_i_, _icv_memcpy_len_ = (len);                                \
    char* _icv_memcpy_dst_ = (char*)(dst);                                          \
    const char* _icv_memcpy_src_ = (const char*)(src);                              \
    if( (_icv_memcpy_len_ & (sizeof(int)-1)) == 0 )                                 \
    {                                                                               \
        assert( ((size_t)_icv_memcpy_src_&(sizeof(int)-1)) == 0 &&                  \
                ((size_t)_icv_memcpy_dst_&(sizeof(int)-1)) == 0 );                  \
        for( _icv_memcpy_i_ = 0; _icv_memcpy_i_ < _icv_memcpy_len_;                 \
            _icv_memcpy_i_+=sizeof(int) )                                           \
        {                                                                           \
            *(int*)(_icv_memcpy_dst_+_icv_memcpy_i_) =                              \
            *(const int*)(_icv_memcpy_src_+_icv_memcpy_i_);                         \
        }                                                                           \
    }                                                                               \
    else                                                                            \
    {                                                                               \
        for(_icv_memcpy_i_ = 0; _icv_memcpy_i_ < _icv_memcpy_len_; _icv_memcpy_i_++)\
            _icv_memcpy_dst_[_icv_memcpy_i_] = _icv_memcpy_src_[_icv_memcpy_i_];    \
    }                                                                               \
}


#define CV_ZERO_CHAR( dst, len )                        \
{                                                       \
    size_t _icv_memcpy_i_, _icv_memcpy_len_ = (len);    \
    char* _icv_memcpy_dst_ = (char*)(dst);              \
                                                        \
    for( _icv_memcpy_i_ = 0; _icv_memcpy_i_ < _icv_memcpy_len_; _icv_memcpy_i_++ )  \
        _icv_memcpy_dst_[_icv_memcpy_i_] = '\0';        \
}


#define CV_ZERO_INT( dst, len )                                                     \
{                                                                                   \
    size_t _icv_memcpy_i_, _icv_memcpy_len_ = (len);                                \
    int* _icv_memcpy_dst_ = (int*)(dst);                                            \
    assert( ((size_t)_icv_memcpy_dst_&(sizeof(int)-1)) == 0 );                      \
                                                                                    \
    for(_icv_memcpy_i_=0;_icv_memcpy_i_<_icv_memcpy_len_;_icv_memcpy_i_++)          \
        _icv_memcpy_dst_[_icv_memcpy_i_] = 0;                                       \
}

namespace cv
{

// -128.f ... 255.f
extern const float g_8x32fTab[];
#define CV_8TO32F(x)  cv::g_8x32fTab[(x)+128]

extern const ushort g_8x16uSqrTab[];
#define CV_SQR_8U(x)  cv::g_8x16uSqrTab[(x)+255]

extern const char* g_HersheyGlyphs[];

extern const uchar g_Saturate8u[];
#define CV_FAST_CAST_8U(t)   (assert(-256 <= (t) && (t) <= 512), cv::g_Saturate8u[(t)+256])
#define CV_MIN_8U(a,b)       ((a) - CV_FAST_CAST_8U((a) - (b)))
#define CV_MAX_8U(a,b)       ((a) + CV_FAST_CAST_8U((b) - (a)))

typedef void (*CopyMaskFunc)(const Mat& src, Mat& dst, const Mat& mask);

extern CopyMaskFunc g_copyMaskFuncTab[];

static inline CopyMaskFunc getCopyMaskFunc(int esz)
{
    CV_Assert( (unsigned)esz <= 32U );
    CopyMaskFunc func = g_copyMaskFuncTab[esz];
    CV_Assert( func != 0 );
    return func;
}

#if defined WIN32 || defined _WIN32
void deleteThreadAllocData();
void deleteThreadRNGData();
#endif

    
template<typename T1, typename T2=T1, typename T3=T1> struct OpAdd
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(a + b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(a - b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpRSub
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(b - a); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpMul
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(a * b); }
};

template<typename T1, typename T2=T1, typename T3=T1> struct OpDiv
{
    typedef T1 type1;
    typedef T2 type2;
    typedef T3 rtype;
    T3 operator ()(T1 a, T2 b) const { return saturate_cast<T3>(a / b); }
};

template<typename T> struct OpMin
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(T a, T b) const { return std::min(a, b); }
};

template<typename T> struct OpMax
{
    typedef T type1;
    typedef T type2;
    typedef T rtype;
    T operator ()(T a, T b) const { return std::max(a, b); }
};

inline Size getContinuousSize( const Mat& m1, int widthScale=1 )
{
    return m1.isContinuous() ? Size(m1.cols*m1.rows*widthScale, 1) :
        Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2, int widthScale=1 )
{
    return (m1.flags & m2.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & m4.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

inline Size getContinuousSize( const Mat& m1, const Mat& m2,
                               const Mat& m3, const Mat& m4,
                               const Mat& m5, int widthScale=1 )
{
    return (m1.flags & m2.flags & m3.flags & m4.flags & m5.flags & Mat::CONTINUOUS_FLAG) != 0 ?
        Size(m1.cols*m1.rows*widthScale, 1) : Size(m1.cols*widthScale, m1.rows);
}

struct NoVec
{
    int operator()(const void*, const void*, void*, int) const { return 0; }
};
    
    
template<class Op, class VecOp> static void
binaryOpC1_( const Mat& srcmat1, const Mat& srcmat2, Mat& dstmat )
{
    Op op; VecOp vecOp;
    typedef typename Op::type1 T1;
    typedef typename Op::type2 T2;
    typedef typename Op::rtype DT;

    const T1* src1 = (const T1*)srcmat1.data;
    const T2* src2 = (const T2*)srcmat2.data;
    DT* dst = (DT*)dstmat.data;
    size_t step1 = srcmat1.step/sizeof(src1[0]);
    size_t step2 = srcmat2.step/sizeof(src2[0]);
    size_t step = dstmat.step/sizeof(dst[0]);
    Size size = getContinuousSize( srcmat1, srcmat2, dstmat, dstmat.channels() );

    if( size.width == 1 )
    {
        for( ; size.height--; src1 += step1, src2 += step2, dst += step )
            dst[0] = op( src1[0], src2[0] );
        return;
    }

    for( ; size.height--; src1 += step1, src2 += step2, dst += step )
    {
        int x = vecOp(src1, src2, dst, size.width);
        for( ; x <= size.width - 4; x += 4 )
        {
            DT f0, f1;
            f0 = op( src1[x], src2[x] );
            f1 = op( src1[x+1], src2[x+1] );
            dst[x] = f0;
            dst[x+1] = f1;
            f0 = op(src1[x+2], src2[x+2]);
            f1 = op(src1[x+3], src2[x+3]);
            dst[x+2] = f0;
            dst[x+3] = f1;
        }

        for( ; x < size.width; x++ )
            dst[x] = op( src1[x], src2[x] );
    }
}

typedef void (*BinaryFunc)(const Mat& src1, const Mat& src2, Mat& dst);

template<class Op> static void
binarySOpCn_( const Mat& srcmat, Mat& dstmat, const Scalar& _scalar )
{
    Op op;
    typedef typename Op::type1 T;
    typedef typename Op::type2 WT;
    typedef typename Op::rtype DT;
    const T* src0 = (const T*)srcmat.data;
    DT* dst0 = (DT*)dstmat.data;
    size_t step1 = srcmat.step/sizeof(src0[0]);
    size_t step = dstmat.step/sizeof(dst0[0]);
    int cn = dstmat.channels();
    Size size = getContinuousSize( srcmat, dstmat, cn );
    WT scalar[12];
    scalarToRawData(_scalar, scalar, CV_MAKETYPE(DataType<WT>::depth,cn), 12);

    for( ; size.height--; src0 += step1, dst0 += step )
    {
        int i, len = size.width;
        const T* src = src0;
        T* dst = dst0;

        for( ; (len -= 12) >= 0; dst += 12, src += 12 )
        {
            DT t0 = op(src[0], scalar[0]);
            DT t1 = op(src[1], scalar[1]);
            dst[0] = t0; dst[1] = t1;

            t0 = op(src[2], scalar[2]);
            t1 = op(src[3], scalar[3]);
            dst[2] = t0; dst[3] = t1;

            t0 = op(src[4], scalar[4]);
            t1 = op(src[5], scalar[5]);
            dst[4] = t0; dst[5] = t1;

            t0 = op(src[6], scalar[6]);
            t1 = op(src[7], scalar[7]);
            dst[6] = t0; dst[7] = t1;

            t0 = op(src[8], scalar[8]);
            t1 = op(src[9], scalar[9]);
            dst[8] = t0; dst[9] = t1;

            t0 = op(src[10], scalar[10]);
            t1 = op(src[11], scalar[11]);
            dst[10] = t0; dst[11] = t1;
        }

        for( (len) += 12, i = 0; i < (len); i++ )
            dst[i] = op((WT)src[i], scalar[i]);
    }
}

template<class Op> static void
binarySOpC1_( const Mat& srcmat, Mat& dstmat, double _scalar )
{
    Op op;
    typedef typename Op::type1 T;
    typedef typename Op::type2 WT;
    typedef typename Op::rtype DT;
    WT scalar = saturate_cast<WT>(_scalar);
    const T* src = (const T*)srcmat.data;
    DT* dst = (DT*)dstmat.data;
    size_t step1 = srcmat.step/sizeof(src[0]);
    size_t step = dstmat.step/sizeof(dst[0]);
    Size size = srcmat.size();

    size.width *= srcmat.channels();
    if( srcmat.isContinuous() && dstmat.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( ; size.height--; src += step1, dst += step )
    {
        int x;
        for( x = 0; x <= size.width - 4; x += 4 )
        {
            DT f0 = op( src[x], scalar );
            DT f1 = op( src[x+1], scalar );
            dst[x] = f0;
            dst[x+1] = f1;
            f0 = op( src[x+2], scalar );
            f1 = op( src[x+3], scalar );
            dst[x+2] = f0;
            dst[x+3] = f1;
        }

        for( ; x < size.width; x++ )
            dst[x] = op( src[x], scalar );
    }
}

typedef void (*BinarySFuncCn)(const Mat& src1, Mat& dst, const Scalar& scalar);
typedef void (*BinarySFuncC1)(const Mat& src1, Mat& dst, double scalar);

}

#endif /*_CXCORE_INTERNAL_H_*/
