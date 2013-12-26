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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
#include "opencv2/core/opencl/runtime/opencl_clamdblas.hpp"

#ifdef HAVE_IPP
#include "ippversion.h"
#endif

namespace cv
{

/****************************************************************************************\
*                                         GEMM                                           *
\****************************************************************************************/

static void
GEMM_CopyBlock( const uchar* src, size_t src_step,
                uchar* dst, size_t dst_step,
                Size size, size_t pix_size )
{
    int j;
    size.width *= (int)(pix_size / sizeof(int));

    for( ; size.height--; src += src_step, dst += dst_step )
    {
        j=0;
         #if CV_ENABLE_UNROLLED
        for( ; j <= size.width - 4; j += 4 )
        {
            int t0 = ((const int*)src)[j];
            int t1 = ((const int*)src)[j+1];
            ((int*)dst)[j] = t0;
            ((int*)dst)[j+1] = t1;
            t0 = ((const int*)src)[j+2];
            t1 = ((const int*)src)[j+3];
            ((int*)dst)[j+2] = t0;
            ((int*)dst)[j+3] = t1;
        }
        #endif
        for( ; j < size.width; j++ )
            ((int*)dst)[j] = ((const int*)src)[j];
    }
}


static void
GEMM_TransposeBlock( const uchar* src, size_t src_step,
                     uchar* dst, size_t dst_step,
                     Size size, size_t pix_size )
{
    int i, j;
    for( i = 0; i < size.width; i++, dst += dst_step, src += pix_size )
    {
        const uchar* _src = src;
        switch( pix_size )
        {
        case sizeof(int):
            for( j = 0; j < size.height; j++, _src += src_step )
                ((int*)dst)[j] = ((int*)_src)[0];
            break;
        case sizeof(int)*2:
            for( j = 0; j < size.height*2; j += 2, _src += src_step )
            {
                int t0 = ((int*)_src)[0];
                int t1 = ((int*)_src)[1];
                ((int*)dst)[j] = t0;
                ((int*)dst)[j+1] = t1;
            }
            break;
        case sizeof(int)*4:
            for( j = 0; j < size.height*4; j += 4, _src += src_step )
            {
                int t0 = ((int*)_src)[0];
                int t1 = ((int*)_src)[1];
                ((int*)dst)[j] = t0;
                ((int*)dst)[j+1] = t1;
                t0 = ((int*)_src)[2];
                t1 = ((int*)_src)[3];
                ((int*)dst)[j+2] = t0;
                ((int*)dst)[j+3] = t1;
            }
            break;
        default:
            assert(0);
            return;
        }
    }
}


template<typename T, typename WT> static void
GEMMSingleMul( const T* a_data, size_t a_step,
               const T* b_data, size_t b_step,
               const T* c_data, size_t c_step,
               T* d_data, size_t d_step,
               Size a_size, Size d_size,
               double alpha, double beta, int flags )
{
    int i, j, k, n = a_size.width, m = d_size.width, drows = d_size.height;
    const T *_a_data = a_data, *_b_data = b_data, *_c_data = c_data;
    cv::AutoBuffer<T> _a_buf;
    T* a_buf = 0;
    size_t a_step0, a_step1, c_step0, c_step1, t_step;

    a_step /= sizeof(a_data[0]);
    b_step /= sizeof(b_data[0]);
    c_step /= sizeof(c_data[0]);
    d_step /= sizeof(d_data[0]);
    a_step0 = a_step;
    a_step1 = 1;

    if( !c_data )
        c_step0 = c_step1 = 0;
    else if( !(flags & GEMM_3_T) )
        c_step0 = c_step, c_step1 = 1;
    else
        c_step0 = 1, c_step1 = c_step;

    if( flags & GEMM_1_T )
    {
        CV_SWAP( a_step0, a_step1, t_step );
        n = a_size.height;
        if( a_step > 1 && n > 1 )
        {
            _a_buf.allocate(n);
            a_buf = _a_buf;
        }
    }

    if( n == 1 ) /* external product */
    {
        cv::AutoBuffer<T> _b_buf;
        T* b_buf = 0;

        if( a_step > 1 && a_size.height > 1 )
        {
            _a_buf.allocate(drows);
            a_buf = _a_buf;
            for( k = 0; k < drows; k++ )
                a_buf[k] = a_data[a_step*k];
            a_data = a_buf;
        }

        if( b_step > 1 )
        {
            _b_buf.allocate(d_size.width);
            b_buf = _b_buf;
            for( j = 0; j < d_size.width; j++ )
                b_buf[j] = b_data[j*b_step];
            b_data = b_buf;
        }

        for( i = 0; i < drows; i++, _c_data += c_step0, d_data += d_step )
        {
            WT al = WT(a_data[i])*alpha;
            c_data = _c_data;
            for( j = 0; j <= d_size.width - 2; j += 2, c_data += 2*c_step1 )
            {
                WT s0 = al*WT(b_data[j]);
                WT s1 = al*WT(b_data[j+1]);
                if( !c_data )
                {
                    d_data[j] = T(s0);
                    d_data[j+1] = T(s1);
                }
                else
                {
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
                    d_data[j+1] = T(s1 + WT(c_data[c_step1])*beta);
                }
            }

            for( ; j < d_size.width; j++, c_data += c_step1 )
            {
                WT s0 = al*WT(b_data[j]);
                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else if( flags & GEMM_2_T ) /* A * Bt */
    {
        for( i = 0; i < drows; i++, _a_data += a_step0, _c_data += c_step0, d_data += d_step )
        {
            a_data = _a_data;
            b_data = _b_data;
            c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < d_size.width; j++, b_data += b_step,
                                               c_data += c_step1 )
            {
                WT s0(0), s1(0), s2(0), s3(0);
                k = 0;
                 #if CV_ENABLE_UNROLLED
                for( ; k <= n - 4; k += 4 )
                {
                    s0 += WT(a_data[k])*WT(b_data[k]);
                    s1 += WT(a_data[k+1])*WT(b_data[k+1]);
                    s2 += WT(a_data[k+2])*WT(b_data[k+2]);
                    s3 += WT(a_data[k+3])*WT(b_data[k+3]);
                }
                #endif
                for( ; k < n; k++ )
                    s0 += WT(a_data[k])*WT(b_data[k]);
                s0 = (s0+s1+s2+s3)*alpha;

                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else if( d_size.width*sizeof(d_data[0]) <= 1600 )
    {
        for( i = 0; i < drows; i++, _a_data += a_step0,
                                    _c_data += c_step0,
                                    d_data += d_step )
        {
            a_data = _a_data, c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j <= m - 4; j += 4, c_data += 4*c_step1 )
            {
                const T* b = _b_data + j;
                WT s0(0), s1(0), s2(0), s3(0);

                for( k = 0; k < n; k++, b += b_step )
                {
                    WT a(a_data[k]);
                    s0 += a * WT(b[0]); s1 += a * WT(b[1]);
                    s2 += a * WT(b[2]); s3 += a * WT(b[3]);
                }

                if( !c_data )
                {
                    d_data[j] = T(s0*alpha);
                    d_data[j+1] = T(s1*alpha);
                    d_data[j+2] = T(s2*alpha);
                    d_data[j+3] = T(s3*alpha);
                }
                else
                {
                    s0 = s0*alpha; s1 = s1*alpha;
                    s2 = s2*alpha; s3 = s3*alpha;
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
                    d_data[j+1] = T(s1 + WT(c_data[c_step1])*beta);
                    d_data[j+2] = T(s2 + WT(c_data[c_step1*2])*beta);
                    d_data[j+3] = T(s3 + WT(c_data[c_step1*3])*beta);
                }
            }

            for( ; j < m; j++, c_data += c_step1 )
            {
                const T* b = _b_data + j;
                WT s0(0);

                for( k = 0; k < n; k++, b += b_step )
                    s0 += WT(a_data[k]) * WT(b[0]);

                s0 = s0*alpha;
                if( !c_data )
                    d_data[j] = T(s0);
                else
                    d_data[j] = T(s0 + WT(c_data[0])*beta);
            }
        }
    }
    else
    {
        cv::AutoBuffer<WT> _d_buf(m);
        WT* d_buf = _d_buf;

        for( i = 0; i < drows; i++, _a_data += a_step0, _c_data += c_step0, d_data += d_step )
        {
            a_data = _a_data;
            b_data = _b_data;
            c_data = _c_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = _a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < m; j++ )
                d_buf[j] = WT(0);

            for( k = 0; k < n; k++, b_data += b_step )
            {
                WT al(a_data[k]);
                j=0;
                 #if CV_ENABLE_UNROLLED
                for(; j <= m - 4; j += 4 )
                {
                    WT t0 = d_buf[j] + WT(b_data[j])*al;
                    WT t1 = d_buf[j+1] + WT(b_data[j+1])*al;
                    d_buf[j] = t0;
                    d_buf[j+1] = t1;
                    t0 = d_buf[j+2] + WT(b_data[j+2])*al;
                    t1 = d_buf[j+3] + WT(b_data[j+3])*al;
                    d_buf[j+2] = t0;
                    d_buf[j+3] = t1;
                }
                #endif
                for( ; j < m; j++ )
                    d_buf[j] += WT(b_data[j])*al;
            }

            if( !c_data )
                for( j = 0; j < m; j++ )
                    d_data[j] = T(d_buf[j]*alpha);
            else
                for( j = 0; j < m; j++, c_data += c_step1 )
                {
                    WT t = d_buf[j]*alpha;
                    d_data[j] = T(t + WT(c_data[0])*beta);
                }
        }
    }
}


template<typename T, typename WT> static void
GEMMBlockMul( const T* a_data, size_t a_step,
              const T* b_data, size_t b_step,
              WT* d_data, size_t d_step,
              Size a_size, Size d_size, int flags )
{
    int i, j, k, n = a_size.width, m = d_size.width;
    const T *_a_data = a_data, *_b_data = b_data;
    cv::AutoBuffer<T> _a_buf;
    T* a_buf = 0;
    size_t a_step0, a_step1, t_step;
    int do_acc = flags & 16;

    a_step /= sizeof(a_data[0]);
    b_step /= sizeof(b_data[0]);
    d_step /= sizeof(d_data[0]);

    a_step0 = a_step;
    a_step1 = 1;

    if( flags & GEMM_1_T )
    {
        CV_SWAP( a_step0, a_step1, t_step );
        n = a_size.height;
        _a_buf.allocate(n);
        a_buf = _a_buf;
    }

    if( flags & GEMM_2_T )
    {
        /* second operand is transposed */
        for( i = 0; i < d_size.height; i++, _a_data += a_step0, d_data += d_step )
        {
            a_data = _a_data; b_data = _b_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j < d_size.width; j++, b_data += b_step )
            {
                WT s0 = do_acc ? d_data[j]:WT(0), s1(0);
                for( k = 0; k <= n - 2; k += 2 )
                {
                    s0 += WT(a_data[k])*WT(b_data[k]);
                    s1 += WT(a_data[k+1])*WT(b_data[k+1]);
                }

                for( ; k < n; k++ )
                    s0 += WT(a_data[k])*WT(b_data[k]);

                d_data[j] = s0 + s1;
            }
        }
    }
    else
    {
        for( i = 0; i < d_size.height; i++, _a_data += a_step0, d_data += d_step )
        {
            a_data = _a_data, b_data = _b_data;

            if( a_buf )
            {
                for( k = 0; k < n; k++ )
                    a_buf[k] = a_data[a_step1*k];
                a_data = a_buf;
            }

            for( j = 0; j <= m - 4; j += 4 )
            {
                WT s0, s1, s2, s3;
                const T* b = b_data + j;

                if( do_acc )
                {
                    s0 = d_data[j]; s1 = d_data[j+1];
                    s2 = d_data[j+2]; s3 = d_data[j+3];
                }
                else
                    s0 = s1 = s2 = s3 = WT(0);

                for( k = 0; k < n; k++, b += b_step )
                {
                    WT a(a_data[k]);
                    s0 += a * WT(b[0]); s1 += a * WT(b[1]);
                    s2 += a * WT(b[2]); s3 += a * WT(b[3]);
                }

                d_data[j] = s0; d_data[j+1] = s1;
                d_data[j+2] = s2; d_data[j+3] = s3;
            }

            for( ; j < m; j++ )
            {
                const T* b = b_data + j;
                WT s0 = do_acc ? d_data[j] : WT(0);

                for( k = 0; k < n; k++, b += b_step )
                    s0 += WT(a_data[k]) * WT(b[0]);

                d_data[j] = s0;
            }
        }
    }
}


template<typename T, typename WT> static void
GEMMStore( const T* c_data, size_t c_step,
           const WT* d_buf, size_t d_buf_step,
           T* d_data, size_t d_step, Size d_size,
           double alpha, double beta, int flags )
{
    const T* _c_data = c_data;
    int j;
    size_t c_step0, c_step1;

    c_step /= sizeof(c_data[0]);
    d_buf_step /= sizeof(d_buf[0]);
    d_step /= sizeof(d_data[0]);

    if( !c_data )
        c_step0 = c_step1 = 0;
    else if( !(flags & GEMM_3_T) )
        c_step0 = c_step, c_step1 = 1;
    else
        c_step0 = 1, c_step1 = c_step;

    for( ; d_size.height--; _c_data += c_step0, d_buf += d_buf_step, d_data += d_step )
    {
        if( _c_data )
        {
            c_data = _c_data;
            j=0;
             #if CV_ENABLE_UNROLLED
            for(; j <= d_size.width - 4; j += 4, c_data += 4*c_step1 )
            {
                WT t0 = alpha*d_buf[j];
                WT t1 = alpha*d_buf[j+1];
                t0 += beta*WT(c_data[0]);
                t1 += beta*WT(c_data[c_step1]);
                d_data[j] = T(t0);
                d_data[j+1] = T(t1);
                t0 = alpha*d_buf[j+2];
                t1 = alpha*d_buf[j+3];
                t0 += beta*WT(c_data[c_step1*2]);
                t1 += beta*WT(c_data[c_step1*3]);
                d_data[j+2] = T(t0);
                d_data[j+3] = T(t1);
            }
            #endif
            for( ; j < d_size.width; j++, c_data += c_step1 )
            {
                WT t0 = alpha*d_buf[j];
                d_data[j] = T(t0 + WT(c_data[0])*beta);
            }
        }
        else
        {
            j = 0;
             #if CV_ENABLE_UNROLLED
            for( ; j <= d_size.width - 4; j += 4 )
            {
                WT t0 = alpha*d_buf[j];
                WT t1 = alpha*d_buf[j+1];
                d_data[j] = T(t0);
                d_data[j+1] = T(t1);
                t0 = alpha*d_buf[j+2];
                t1 = alpha*d_buf[j+3];
                d_data[j+2] = T(t0);
                d_data[j+3] = T(t1);
            }
            #endif
            for( ; j < d_size.width; j++ )
                d_data[j] = T(alpha*d_buf[j]);
        }
    }
}


typedef void (*GEMMSingleMulFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, const void* src3, size_t step3,
                   void* dst, size_t dststep, Size srcsize, Size dstsize,
                   double alpha, double beta, int flags );

typedef void (*GEMMBlockMulFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, void* dst, size_t dststep,
                   Size srcsize, Size dstsize, int flags );

typedef void (*GEMMStoreFunc)( const void* src1, size_t step1,
                   const void* src2, size_t step2, void* dst, size_t dststep,
                   Size dstsize, double alpha, double beta, int flags );

static void GEMMSingleMul_32f( const float* a_data, size_t a_step,
              const float* b_data, size_t b_step,
              const float* c_data, size_t c_step,
              float* d_data, size_t d_step,
              Size a_size, Size d_size,
              double alpha, double beta, int flags )
{
    GEMMSingleMul<float,double>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}

static void GEMMSingleMul_64f( const double* a_data, size_t a_step,
                              const double* b_data, size_t b_step,
                              const double* c_data, size_t c_step,
                              double* d_data, size_t d_step,
                              Size a_size, Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<double,double>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}


static void GEMMSingleMul_32fc( const Complexf* a_data, size_t a_step,
                              const Complexf* b_data, size_t b_step,
                              const Complexf* c_data, size_t c_step,
                              Complexf* d_data, size_t d_step,
                              Size a_size, Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<Complexf,Complexd>(a_data, a_step, b_data, b_step, c_data,
                                c_step, d_data, d_step, a_size, d_size,
                                alpha, beta, flags);
}

static void GEMMSingleMul_64fc( const Complexd* a_data, size_t a_step,
                              const Complexd* b_data, size_t b_step,
                              const Complexd* c_data, size_t c_step,
                              Complexd* d_data, size_t d_step,
                              Size a_size, Size d_size,
                              double alpha, double beta, int flags )
{
    GEMMSingleMul<Complexd,Complexd>(a_data, a_step, b_data, b_step, c_data,
                                 c_step, d_data, d_step, a_size, d_size,
                                 alpha, beta, flags);
}

static void GEMMBlockMul_32f( const float* a_data, size_t a_step,
             const float* b_data, size_t b_step,
             double* d_data, size_t d_step,
             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_64f( const double* a_data, size_t a_step,
                             const double* b_data, size_t b_step,
                             double* d_data, size_t d_step,
                             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_32fc( const Complexf* a_data, size_t a_step,
                             const Complexf* b_data, size_t b_step,
                             Complexd* d_data, size_t d_step,
                             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMBlockMul_64fc( const Complexd* a_data, size_t a_step,
                             const Complexd* b_data, size_t b_step,
                             Complexd* d_data, size_t d_step,
                             Size a_size, Size d_size, int flags )
{
    GEMMBlockMul(a_data, a_step, b_data, b_step, d_data, d_step, a_size, d_size, flags);
}


static void GEMMStore_32f( const float* c_data, size_t c_step,
          const double* d_buf, size_t d_buf_step,
          float* d_data, size_t d_step, Size d_size,
          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_64f( const double* c_data, size_t c_step,
                      const double* d_buf, size_t d_buf_step,
                      double* d_data, size_t d_step, Size d_size,
                      double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_32fc( const Complexf* c_data, size_t c_step,
                          const Complexd* d_buf, size_t d_buf_step,
                          Complexf* d_data, size_t d_step, Size d_size,
                          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}


static void GEMMStore_64fc( const Complexd* c_data, size_t c_step,
                          const Complexd* d_buf, size_t d_buf_step,
                          Complexd* d_data, size_t d_step, Size d_size,
                          double alpha, double beta, int flags )
{
    GEMMStore(c_data, c_step, d_buf, d_buf_step, d_data, d_step, d_size, alpha, beta, flags);
}

#ifdef HAVE_CLAMDBLAS

static bool ocl_gemm( InputArray matA, InputArray matB, double alpha,
                      InputArray matC, double beta, OutputArray matD, int flags )
{
    int type = matA.type(), esz = CV_ELEM_SIZE(type);
    bool haveC = matC.kind() != cv::_InputArray::NONE;
    Size sizeA = matA.size(), sizeB = matB.size(), sizeC = haveC ? matC.size() : Size(0, 0);
    bool atrans = (flags & GEMM_1_T) != 0, btrans = (flags & GEMM_2_T) != 0, ctrans = (flags & GEMM_3_T) != 0;

    if (atrans)
        sizeA = Size(sizeA.height, sizeA.width);
    if (btrans)
        sizeB = Size(sizeB.height, sizeB.width);
    if (haveC && ctrans)
        sizeC = Size(sizeC.height, sizeC.width);

    Size sizeD(sizeB.width, sizeA.height);

    CV_Assert( matB.type() == type && (!haveC || matC.type() == type) );
    CV_Assert( sizeA.width == sizeB.height && (!haveC || sizeC == sizeD) );

    matD.create(sizeD, type);
    if ( matA.offset() % esz != 0 || matA.step() % esz != 0 ||
         matB.offset() % esz != 0 || matB.step() % esz != 0 ||
         (haveC && (matC.offset() % esz != 0 || matC.step() % esz != 0)) )
        return false;

    UMat A = matA.getUMat(), B = matB.getUMat(), D = matD.getUMat();
    if (haveC)
        ctrans ? transpose(matC, D) : matC.copyTo(D);
    else
        D.setTo(Scalar::all(0));

    int M = sizeD.height, N = sizeD.width, K = sizeA.width;
    int lda = (int)A.step / esz, ldb = (int)B.step / esz, ldc = (int)D.step / esz;
    int offa = (int)A.offset / esz, offb = (int)B.offset / esz, offc = (int)D.offset / esz;

    cl_command_queue clq = (cl_command_queue)ocl::Queue::getDefault().ptr();
    clAmdBlasTranspose transA = atrans ? clAmdBlasTrans : clAmdBlasNoTrans;
    clAmdBlasTranspose transB = btrans ? clAmdBlasTrans : clAmdBlasNoTrans;
    clAmdBlasOrder order = clAmdBlasRowMajor;
    clAmdBlasStatus status = clAmdBlasSuccess;

    if (type == CV_32FC1)
        status = clAmdBlasSgemmEx(order, transA, transB, M, N, K,
                                  (cl_float)alpha, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                                  (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                                  (cl_float)beta, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                                  1, &clq, 0, NULL, NULL);
    else if (type == CV_64FC1)
        status = clAmdBlasDgemmEx(order, transA, transB, M, N, K,
                                  alpha, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                                  (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                                  beta, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                                  1, &clq, 0, NULL, NULL);
    else if (type == CV_32FC2)
    {
         cl_float2 alpha_2 = { { (cl_float)alpha, 0 } };
         cl_float2 beta_2  = { { (cl_float)beta, 0 } };
         status = clAmdBlasCgemmEx(order, transA, transB, M, N, K,
                                   alpha_2, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                                   (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                                   beta_2, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                                   1, &clq, 0, NULL, NULL);
    }
    else if (type == CV_64FC2)
    {
        cl_double2 alpha_2 = { { alpha, 0 } };
        cl_double2 beta_2  = { { beta, 0 } };
        status = clAmdBlasZgemmEx(order, transA, transB, M, N, K,
                                  alpha_2, (const cl_mem)A.handle(ACCESS_READ), offa, lda,
                                  (const cl_mem)B.handle(ACCESS_READ), offb, ldb,
                                  beta_2, (cl_mem)D.handle(ACCESS_RW), offc, ldc,
                                  1, &clq, 0, NULL, NULL);
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");

    return status == clAmdBlasSuccess;
}

#endif

}

void cv::gemm( InputArray matA, InputArray matB, double alpha,
           InputArray matC, double beta, OutputArray _matD, int flags )
{
#ifdef HAVE_CLAMDBLAS
    if (ocl::haveAmdBlas() && matA.dims() <= 2 && matB.dims() <= 2 && matC.dims() <= 2 &&
            ocl::useOpenCL() && _matD.isUMat() &&
            ocl_gemm(matA, matB, alpha, matC, beta, _matD, flags))
        return;
#endif

    const int block_lin_size = 128;
    const int block_size = block_lin_size * block_lin_size;

    static double zero[] = {0,0,0,0};
    static float zerof[] = {0,0,0,0};

    Mat A = matA.getMat(), B = matB.getMat(), C = beta != 0 ? matC.getMat() : Mat();
    Size a_size = A.size(), d_size;
    int i, len = 0, type = A.type();

    CV_Assert( type == B.type() && (type == CV_32FC1 || type == CV_64FC1 || type == CV_32FC2 || type == CV_64FC2) );

    switch( flags & (GEMM_1_T|GEMM_2_T) )
    {
    case 0:
        d_size = Size( B.cols, a_size.height );
        len = B.rows;
        CV_Assert( a_size.width == len );
        break;
    case 1:
        d_size = Size( B.cols, a_size.width );
        len = B.rows;
        CV_Assert( a_size.height == len );
        break;
    case 2:
        d_size = Size( B.rows, a_size.height );
        len = B.cols;
        CV_Assert( a_size.width == len );
        break;
    case 3:
        d_size = Size( B.rows, a_size.width );
        len = B.cols;
        CV_Assert( a_size.height == len );
        break;
    }

    if( C.data )
    {
        CV_Assert( C.type() == type &&
            (((flags&GEMM_3_T) == 0 && C.rows == d_size.height && C.cols == d_size.width) ||
             ((flags&GEMM_3_T) != 0 && C.rows == d_size.width && C.cols == d_size.height)));
    }

    _matD.create( d_size.height, d_size.width, type );
    Mat D = _matD.getMat();
    if( (flags & GEMM_3_T) != 0 && C.data == D.data )
    {
        transpose( C, C );
        flags &= ~GEMM_3_T;
    }

    if( flags == 0 && 2 <= len && len <= 4 && (len == d_size.width || len == d_size.height) )
    {
        if( type == CV_32F )
        {
            float* d = (float*)D.data;
            const float *a = (const float*)A.data,
                        *b = (const float*)B.data,
                        *c = (const float*)C.data;
            size_t d_step = D.step/sizeof(d[0]),
                a_step = A.step/sizeof(a[0]),
                b_step = B.step/sizeof(b[0]),
                c_step = C.data ? C.step/sizeof(c[0]) : 0;

            if( !c )
                c = zerof;

            switch( len )
            {
            case 2:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                    }
                }
                else
                    break;
                return;
            case 3:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1];
                        float t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                        d[2] = (float)(t2*alpha + c[2]*beta);
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] + a[a_step+2]*b[b_step*2];
                        float t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] + a[a_step*2+2]*b[b_step*2];

                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                        d[d_step*2] = (float)(t2*alpha + c[c_step*2]*beta);
                    }
                }
                else
                    break;
                return;
            case 4:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        float t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1] + a[3]*b[b_step*3+1];
                        float t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2] + a[3]*b[b_step*3+2];
                        float t3 = a[0]*b[3] + a[1]*b[b_step+3] + a[2]*b[b_step*2+3] + a[3]*b[b_step*3+3];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[1] = (float)(t1*alpha + c[1]*beta);
                        d[2] = (float)(t2*alpha + c[2]*beta);
                        d[3] = (float)(t3*alpha + c[3]*beta);
                    }
                }
                else if( len <= 16 && a != d )
                {
                    int c_step0 = 1;
                    if( c == zerof )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        float t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        float t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] +
                                   a[a_step+2]*b[b_step*2] + a[a_step+3]*b[b_step*3];
                        float t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] +
                                   a[a_step*2+2]*b[b_step*2] + a[a_step*2+3]*b[b_step*3];
                        float t3 = a[a_step*3]*b[0] + a[a_step*3+1]*b[b_step] +
                                   a[a_step*3+2]*b[b_step*2] + a[a_step*3+3]*b[b_step*3];
                        d[0] = (float)(t0*alpha + c[0]*beta);
                        d[d_step] = (float)(t1*alpha + c[c_step]*beta);
                        d[d_step*2] = (float)(t2*alpha + c[c_step*2]*beta);
                        d[d_step*3] = (float)(t3*alpha + c[c_step*3]*beta);
                    }
                }
                else
                    break;
                return;
            }
        }

        if( type == CV_64F )
        {
            double* d = (double*)D.data;
            const double *a = (const double*)A.data,
                         *b = (const double*)B.data,
                         *c = (const double*)C.data;
            size_t d_step = D.step/sizeof(d[0]),
                a_step = A.step/sizeof(a[0]),
                b_step = B.step/sizeof(b[0]),
                c_step = C.data ? C.step/sizeof(c[0]) : 0;
            if( !c )
                c = zero;

            switch( len )
            {
            case 2:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step];
                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                    }
                }
                else
                    break;
                return;
            case 3:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1];
                        double t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                        d[2] = t2*alpha + c[2]*beta;
                    }
                }
                else if( a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] + a[a_step+2]*b[b_step*2];
                        double t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] + a[a_step*2+2]*b[b_step*2];

                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                        d[d_step*2] = t2*alpha + c[c_step*2]*beta;
                    }
                }
                else
                    break;
                return;
            case 4:
                if( len == d_size.width && b != d )
                {
                    for( i = 0; i < d_size.height; i++, d += d_step, a += a_step, c += c_step )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        double t1 = a[0]*b[1] + a[1]*b[b_step+1] + a[2]*b[b_step*2+1] + a[3]*b[b_step*3+1];
                        double t2 = a[0]*b[2] + a[1]*b[b_step+2] + a[2]*b[b_step*2+2] + a[3]*b[b_step*3+2];
                        double t3 = a[0]*b[3] + a[1]*b[b_step+3] + a[2]*b[b_step*2+3] + a[3]*b[b_step*3+3];
                        d[0] = t0*alpha + c[0]*beta;
                        d[1] = t1*alpha + c[1]*beta;
                        d[2] = t2*alpha + c[2]*beta;
                        d[3] = t3*alpha + c[3]*beta;
                    }
                }
                else if( d_size.width <= 16 && a != d )
                {
                    int c_step0 = 1;
                    if( c == zero )
                    {
                        c_step0 = 0;
                        c_step = 1;
                    }

                    for( i = 0; i < d_size.width; i++, d++, b++, c += c_step0 )
                    {
                        double t0 = a[0]*b[0] + a[1]*b[b_step] + a[2]*b[b_step*2] + a[3]*b[b_step*3];
                        double t1 = a[a_step]*b[0] + a[a_step+1]*b[b_step] +
                                    a[a_step+2]*b[b_step*2] + a[a_step+3]*b[b_step*3];
                        double t2 = a[a_step*2]*b[0] + a[a_step*2+1]*b[b_step] +
                                    a[a_step*2+2]*b[b_step*2] + a[a_step*2+3]*b[b_step*3];
                        double t3 = a[a_step*3]*b[0] + a[a_step*3+1]*b[b_step] +
                                    a[a_step*3+2]*b[b_step*2] + a[a_step*3+3]*b[b_step*3];
                        d[0] = t0*alpha + c[0]*beta;
                        d[d_step] = t1*alpha + c[c_step]*beta;
                        d[d_step*2] = t2*alpha + c[c_step*2]*beta;
                        d[d_step*3] = t3*alpha + c[c_step*3]*beta;
                    }
                }
                else
                    break;
                return;
            }
        }
    }

    {
    size_t b_step = B.step;
    GEMMSingleMulFunc singleMulFunc;
    GEMMBlockMulFunc blockMulFunc;
    GEMMStoreFunc storeFunc;
    Mat *matD = &D, tmat;
    const uchar* Cdata = C.data;
    size_t Cstep = C.data ? (size_t)C.step : 0;
    AutoBuffer<uchar> buf;

    if( type == CV_32FC1 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_32f;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_32f;
        storeFunc = (GEMMStoreFunc)GEMMStore_32f;
    }
    else if( type == CV_64FC1 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_64f;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_64f;
        storeFunc = (GEMMStoreFunc)GEMMStore_64f;
    }
    else if( type == CV_32FC2 )
    {
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_32fc;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_32fc;
        storeFunc = (GEMMStoreFunc)GEMMStore_32fc;
    }
    else
    {
        CV_Assert( type == CV_64FC2 );
        singleMulFunc = (GEMMSingleMulFunc)GEMMSingleMul_64fc;
        blockMulFunc = (GEMMBlockMulFunc)GEMMBlockMul_64fc;
        storeFunc = (GEMMStoreFunc)GEMMStore_64fc;
    }

    if( D.data == A.data || D.data == B.data )
    {
        buf.allocate(d_size.width*d_size.height*CV_ELEM_SIZE(type));
        tmat = Mat(d_size.height, d_size.width, type, (uchar*)buf );
        matD = &tmat;
    }

    if( (d_size.width == 1 || len == 1) && !(flags & GEMM_2_T) && B.isContinuous() )
    {
        b_step = d_size.width == 1 ? 0 : CV_ELEM_SIZE(type);
        flags |= GEMM_2_T;
    }

    /*if( (d_size.width | d_size.height | len) >= 16 && icvBLAS_GEMM_32f_p != 0 )
    {
        blas_func = type == CV_32FC1 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_32f_p :
                    type == CV_64FC1 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_64f_p :
                    type == CV_32FC2 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_32fc_p :
                    type == CV_64FC2 ? (icvBLAS_GEMM_32f_t)icvBLAS_GEMM_64fc_p : 0;
    }

    if( blas_func )
    {
        const char* transa = flags & GEMM_1_T ? "t" : "n";
        const char* transb = flags & GEMM_2_T ? "t" : "n";
        int lda, ldb, ldd;

        if( C->data.ptr )
        {
            if( C->data.ptr != D->data.ptr )
            {
                if( !(flags & GEMM_3_T) )
                    cvCopy( C, D );
                else
                    cvTranspose( C, D );
            }
        }

        if( CV_MAT_DEPTH(type) == CV_32F )
        {
            Complex32f _alpha, _beta;

            lda = A->step/sizeof(float);
            ldb = b_step/sizeof(float);
            ldd = D->step/sizeof(float);
            _alpha.re = (float)alpha;
            _alpha.im = 0;
            _beta.re = C->data.ptr ? (float)beta : 0;
            _beta.im = 0;
            if( CV_MAT_CN(type) == 2 )
                lda /= 2, ldb /= 2, ldd /= 2;

            blas_func( transb, transa, &d_size.width, &d_size.height, &len,
                   &_alpha, B->data.ptr, &ldb, A->data.ptr, &lda,
                   &_beta, D->data.ptr, &ldd );
        }
        else
        {
            CvComplex64f _alpha, _beta;

            lda = A->step/sizeof(double);
            ldb = b_step/sizeof(double);
            ldd = D->step/sizeof(double);
            _alpha.re = alpha;
            _alpha.im = 0;
            _beta.re = C->data.ptr ? beta : 0;
            _beta.im = 0;
            if( CV_MAT_CN(type) == 2 )
                lda /= 2, ldb /= 2, ldd /= 2;

            blas_func( transb, transa, &d_size.width, &d_size.height, &len,
                   &_alpha, B->data.ptr, &ldb, A->data.ptr, &lda,
                   &_beta, D->data.ptr, &ldd );
        }
    }
    else*/ if( ((d_size.height <= block_lin_size/2 || d_size.width <= block_lin_size/2) &&
        len <= 10000) || len <= 10 ||
        (d_size.width <= block_lin_size &&
        d_size.height <= block_lin_size && len <= block_lin_size) )
    {
        singleMulFunc( A.data, A.step, B.data, b_step, Cdata, Cstep,
                       matD->data, matD->step, a_size, d_size, alpha, beta, flags );
    }
    else
    {
        int is_a_t = flags & GEMM_1_T;
        int is_b_t = flags & GEMM_2_T;
        int elem_size = CV_ELEM_SIZE(type);
        int dk0_1, dk0_2;
        int a_buf_size = 0, b_buf_size, d_buf_size;
        uchar* a_buf = 0;
        uchar* b_buf = 0;
        uchar* d_buf = 0;
        int j, k, di = 0, dj = 0, dk = 0;
        int dm0, dn0, dk0;
        size_t a_step0, a_step1, b_step0, b_step1, c_step0, c_step1;
        int work_elem_size = elem_size << (CV_MAT_DEPTH(type) == CV_32F ? 1 : 0);

        if( !is_a_t )
            a_step0 = A.step, a_step1 = elem_size;
        else
            a_step0 = elem_size, a_step1 = A.step;

        if( !is_b_t )
            b_step0 = b_step, b_step1 = elem_size;
        else
            b_step0 = elem_size, b_step1 = b_step;

        if( !C.data )
        {
            c_step0 = c_step1 = 0;
            flags &= ~GEMM_3_T;
        }
        else if( !(flags & GEMM_3_T) )
            c_step0 = C.step, c_step1 = elem_size;
        else
            c_step0 = elem_size, c_step1 = C.step;

        dm0 = std::min( block_lin_size, d_size.height );
        dn0 = std::min( block_lin_size, d_size.width );
        dk0_1 = block_size / dm0;
        dk0_2 = block_size / dn0;
        dk0 = std::min( dk0_1, dk0_2 );
        dk0 = std::min( dk0, len );
        if( dk0*dm0 > block_size )
            dm0 = block_size / dk0;
        if( dk0*dn0 > block_size )
            dn0 = block_size / dk0;

        dk0_1 = (dn0+dn0/8+2) & -2;
        b_buf_size = (dk0+dk0/8+1)*dk0_1*elem_size;
        d_buf_size = (dk0+dk0/8+1)*dk0_1*work_elem_size;

        if( is_a_t )
        {
            a_buf_size = (dm0+dm0/8+1)*((dk0+dk0/8+2)&-2)*elem_size;
            flags &= ~GEMM_1_T;
        }

        buf.allocate(a_buf_size + b_buf_size + d_buf_size);
        d_buf = (uchar*)buf;
        b_buf = d_buf + d_buf_size;

        if( is_a_t )
            a_buf = b_buf + b_buf_size;

        for( i = 0; i < d_size.height; i += di )
        {
            di = dm0;
            if( i + di >= d_size.height || 8*(i + di) + di > 8*d_size.height )
                di = d_size.height - i;

            for( j = 0; j < d_size.width; j += dj )
            {
                uchar* _d = matD->data + i*matD->step + j*elem_size;
                const uchar* _c = Cdata + i*c_step0 + j*c_step1;
                size_t _d_step = matD->step;
                dj = dn0;

                if( j + dj >= d_size.width || 8*(j + dj) + dj > 8*d_size.width )
                    dj = d_size.width - j;

                flags &= 15;
                if( dk0 < len )
                {
                    _d = d_buf;
                    _d_step = dj*work_elem_size;
                }

                for( k = 0; k < len; k += dk )
                {
                    const uchar* _a = A.data + i*a_step0 + k*a_step1;
                    size_t _a_step = A.step;
                    const uchar* _b = B.data + k*b_step0 + j*b_step1;
                    size_t _b_step = b_step;
                    Size a_bl_size;

                    dk = dk0;
                    if( k + dk >= len || 8*(k + dk) + dk > 8*len )
                        dk = len - k;

                    if( !is_a_t )
                        a_bl_size.width = dk, a_bl_size.height = di;
                    else
                        a_bl_size.width = di, a_bl_size.height = dk;

                    if( a_buf && is_a_t )
                    {
                        _a_step = dk*elem_size;
                        GEMM_TransposeBlock( _a, A.step, a_buf, _a_step, a_bl_size, elem_size );
                        std::swap( a_bl_size.width, a_bl_size.height );
                        _a = a_buf;
                    }

                    if( dj < d_size.width )
                    {
                        Size b_size;
                        if( !is_b_t )
                            b_size.width = dj, b_size.height = dk;
                        else
                            b_size.width = dk, b_size.height = dj;

                        _b_step = b_size.width*elem_size;
                        GEMM_CopyBlock( _b, b_step, b_buf, _b_step, b_size, elem_size );
                        _b = b_buf;
                    }

                    if( dk0 < len )
                        blockMulFunc( _a, _a_step, _b, _b_step, _d, _d_step,
                                      a_bl_size, Size(dj,di), flags );
                    else
                        singleMulFunc( _a, _a_step, _b, _b_step, _c, Cstep,
                                       _d, _d_step, a_bl_size, Size(dj,di), alpha, beta, flags );
                    flags |= 16;
                }

                if( dk0 < len )
                    storeFunc( _c, Cstep, _d, _d_step,
                               matD->data + i*matD->step + j*elem_size,
                               matD->step, Size(dj,di), alpha, beta, flags );
            }
        }
    }

    if( matD != &D )
        matD->copyTo(D);
    }
}

/****************************************************************************************\
*                                        Transform                                       *
\****************************************************************************************/

namespace cv
{

template<typename T, typename WT> static void
transform_( const T* src, T* dst, const WT* m, int len, int scn, int dcn )
{
    int x;

    if( scn == 2 && dcn == 2 )
    {
        for( x = 0; x < len*2; x += 2 )
        {
            WT v0 = src[x], v1 = src[x+1];
            T t0 = saturate_cast<T>(m[0]*v0 + m[1]*v1 + m[2]);
            T t1 = saturate_cast<T>(m[3]*v0 + m[4]*v1 + m[5]);
            dst[x] = t0; dst[x+1] = t1;
        }
    }
    else if( scn == 3 && dcn == 3 )
    {
        for( x = 0; x < len*3; x += 3 )
        {
            WT v0 = src[x], v1 = src[x+1], v2 = src[x+2];
            T t0 = saturate_cast<T>(m[0]*v0 + m[1]*v1 + m[2]*v2 + m[3]);
            T t1 = saturate_cast<T>(m[4]*v0 + m[5]*v1 + m[6]*v2 + m[7]);
            T t2 = saturate_cast<T>(m[8]*v0 + m[9]*v1 + m[10]*v2 + m[11]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
    }
    else if( scn == 3 && dcn == 1 )
    {
        for( x = 0; x < len; x++, src += 3 )
            dst[x] = saturate_cast<T>(m[0]*src[0] + m[1]*src[1] + m[2]*src[2] + m[3]);
    }
    else if( scn == 4 && dcn == 4 )
    {
        for( x = 0; x < len*4; x += 4 )
        {
            WT v0 = src[x], v1 = src[x+1], v2 = src[x+2], v3 = src[x+3];
            T t0 = saturate_cast<T>(m[0]*v0 + m[1]*v1 + m[2]*v2 + m[3]*v3 + m[4]);
            T t1 = saturate_cast<T>(m[5]*v0 + m[6]*v1 + m[7]*v2 + m[8]*v3 + m[9]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<T>(m[10]*v0 + m[11]*v1 + m[12]*v2 + m[13]*v3 + m[14]);
            t1 = saturate_cast<T>(m[15]*v0 + m[16]*v1 + m[17]*v2 + m[18]*v3 + m[19]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
    }
    else
    {
        for( x = 0; x < len; x++, src += scn, dst += dcn )
        {
            const WT* _m = m;
            int j, k;
            for( j = 0; j < dcn; j++, _m += scn + 1 )
            {
                WT s = _m[scn];
                for( k = 0; k < scn; k++ )
                    s += _m[k]*src[k];
                dst[j] = saturate_cast<T>(s);
            }
        }
    }
}

#if CV_SSE2

static inline void
load3x3Matrix( const float* m, __m128& m0, __m128& m1, __m128& m2, __m128& m3 )
{
    m0 = _mm_setr_ps(m[0], m[4], m[8], 0);
    m1 = _mm_setr_ps(m[1], m[5], m[9], 0);
    m2 = _mm_setr_ps(m[2], m[6], m[10], 0);
    m3 = _mm_setr_ps(m[3], m[7], m[11], 0);
}

static inline void
load4x4Matrix( const float* m, __m128& m0, __m128& m1, __m128& m2, __m128& m3, __m128& m4 )
{
    m0 = _mm_setr_ps(m[0], m[5], m[10], m[15]);
    m1 = _mm_setr_ps(m[1], m[6], m[11], m[16]);
    m2 = _mm_setr_ps(m[2], m[7], m[12], m[17]);
    m3 = _mm_setr_ps(m[3], m[8], m[13], m[18]);
    m4 = _mm_setr_ps(m[4], m[9], m[14], m[19]);
}

#endif

static void
transform_8u( const uchar* src, uchar* dst, const float* m, int len, int scn, int dcn )
{
#if CV_SSE2
    const int BITS = 10, SCALE = 1 << BITS;
    const float MAX_M = (float)(1 << (15 - BITS));

    if( USE_SSE2 && scn == 3 && dcn == 3 &&
        std::abs(m[0]) < MAX_M && std::abs(m[1]) < MAX_M && std::abs(m[2]) < MAX_M && std::abs(m[3]) < MAX_M*256 &&
        std::abs(m[4]) < MAX_M && std::abs(m[5]) < MAX_M && std::abs(m[6]) < MAX_M && std::abs(m[7]) < MAX_M*256 &&
        std::abs(m[8]) < MAX_M && std::abs(m[9]) < MAX_M && std::abs(m[10]) < MAX_M && std::abs(m[11]) < MAX_M*256 )
    {
        // faster fixed-point transformation
        short m00 = saturate_cast<short>(m[0]*SCALE), m01 = saturate_cast<short>(m[1]*SCALE),
            m02 = saturate_cast<short>(m[2]*SCALE), m10 = saturate_cast<short>(m[4]*SCALE),
            m11 = saturate_cast<short>(m[5]*SCALE), m12 = saturate_cast<short>(m[6]*SCALE),
            m20 = saturate_cast<short>(m[8]*SCALE), m21 = saturate_cast<short>(m[9]*SCALE),
            m22 = saturate_cast<short>(m[10]*SCALE);
        int m03 = saturate_cast<int>((m[3]+0.5f)*SCALE), m13 = saturate_cast<int>((m[7]+0.5f)*SCALE ),
            m23 = saturate_cast<int>((m[11]+0.5f)*SCALE);

        __m128i m0 = _mm_setr_epi16(0, m00, m01, m02, m00, m01, m02, 0);
        __m128i m1 = _mm_setr_epi16(0, m10, m11, m12, m10, m11, m12, 0);
        __m128i m2 = _mm_setr_epi16(0, m20, m21, m22, m20, m21, m22, 0);
        __m128i m3 = _mm_setr_epi32(m03, m13, m23, 0);
        int x = 0;

        for( ; x <= (len - 8)*3; x += 8*3 )
        {
            __m128i z = _mm_setzero_si128(), t0, t1, t2, r0, r1;
            __m128i v0 = _mm_loadl_epi64((const __m128i*)(src + x));
            __m128i v1 = _mm_loadl_epi64((const __m128i*)(src + x + 8));
            __m128i v2 = _mm_loadl_epi64((const __m128i*)(src + x + 16)), v3;
            v0 = _mm_unpacklo_epi8(v0, z); // b0 g0 r0 b1 g1 r1 b2 g2
            v1 = _mm_unpacklo_epi8(v1, z); // r2 b3 g3 r3 b4 g4 r4 b5
            v2 = _mm_unpacklo_epi8(v2, z); // g5 r5 b6 g6 r6 b7 g7 r7

            v3 = _mm_srli_si128(v2, 2); // ? b6 g6 r6 b7 g7 r7 0
            v2 = _mm_or_si128(_mm_slli_si128(v2, 10), _mm_srli_si128(v1, 6)); // ? b4 g4 r4 b5 g5 r5 ?
            v1 = _mm_or_si128(_mm_slli_si128(v1, 6), _mm_srli_si128(v0, 10)); // ? b2 g2 r2 b3 g3 r3 ?
            v0 = _mm_slli_si128(v0, 2); // 0 b0 g0 r0 b1 g1 r1 ?

            // process pixels 0 & 1
            t0 = _mm_madd_epi16(v0, m0); // a0 b0 a1 b1
            t1 = _mm_madd_epi16(v0, m1); // c0 d0 c1 d1
            t2 = _mm_madd_epi16(v0, m2); // e0 f0 e1 f1
            v0 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
            t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
            t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
            t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
            r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v0, t1), _mm_unpackhi_epi64(v0,t1)), m3); // B0 G0 R0 0
            r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0,t2)), m3); // B1 G1 R1 0
            r0 = _mm_srai_epi32(r0, BITS);
            r1 = _mm_srai_epi32(r1, BITS);
            v0 = _mm_packus_epi16(_mm_packs_epi32(_mm_slli_si128(r0, 4), r1), z); // 0 B0 G0 R0 B1 G1 R1 0

            // process pixels 2 & 3
            t0 = _mm_madd_epi16(v1, m0); // a0 b0 a1 b1
            t1 = _mm_madd_epi16(v1, m1); // c0 d0 c1 d1
            t2 = _mm_madd_epi16(v1, m2); // e0 f0 e1 f1
            v1 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
            t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
            t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
            t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
            r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v1, t1), _mm_unpackhi_epi64(v1,t1)), m3); // B2 G2 R2 0
            r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0,t2)), m3); // B3 G3 R3 0
            r0 = _mm_srai_epi32(r0, BITS);
            r1 = _mm_srai_epi32(r1, BITS);
            v1 = _mm_packus_epi16(_mm_packs_epi32(_mm_slli_si128(r0, 4), r1), z); // 0 B2 G2 R2 B3 G3 R3 0

            // process pixels 4 & 5
            t0 = _mm_madd_epi16(v2, m0); // a0 b0 a1 b1
            t1 = _mm_madd_epi16(v2, m1); // c0 d0 c1 d1
            t2 = _mm_madd_epi16(v2, m2); // e0 f0 e1 f1
            v2 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
            t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
            t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
            t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
            r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v2, t1), _mm_unpackhi_epi64(v2,t1)), m3); // B4 G4 R4 0
            r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0,t2)), m3); // B5 G5 R5 0
            r0 = _mm_srai_epi32(r0, BITS);
            r1 = _mm_srai_epi32(r1, BITS);
            v2 = _mm_packus_epi16(_mm_packs_epi32(_mm_slli_si128(r0, 4), r1), z); // 0 B4 G4 R4 B5 G5 R5 0

            // process pixels 6 & 7
            t0 = _mm_madd_epi16(v3, m0); // a0 b0 a1 b1
            t1 = _mm_madd_epi16(v3, m1); // c0 d0 c1 d1
            t2 = _mm_madd_epi16(v3, m2); // e0 f0 e1 f1
            v3 = _mm_unpacklo_epi32(t0, t1); // a0 c0 b0 d0
            t0 = _mm_unpackhi_epi32(t0, t1); // a1 b1 c1 d1
            t1 = _mm_unpacklo_epi32(t2, z);  // e0 0 f0 0
            t2 = _mm_unpackhi_epi32(t2, z);  // e1 0 f1 0
            r0 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(v3, t1), _mm_unpackhi_epi64(v3,t1)), m3); // B6 G6 R6 0
            r1 = _mm_add_epi32(_mm_add_epi32(_mm_unpacklo_epi64(t0, t2), _mm_unpackhi_epi64(t0,t2)), m3); // B7 G7 R7 0
            r0 = _mm_srai_epi32(r0, BITS);
            r1 = _mm_srai_epi32(r1, BITS);
            v3 = _mm_packus_epi16(_mm_packs_epi32(_mm_slli_si128(r0, 4), r1), z); // 0 B6 G6 R6 B7 G7 R7 0

            v0 = _mm_or_si128(_mm_srli_si128(v0, 1), _mm_slli_si128(v1, 5));
            v1 = _mm_or_si128(_mm_srli_si128(v1, 3), _mm_slli_si128(v2, 3));
            v2 = _mm_or_si128(_mm_srli_si128(v2, 5), _mm_slli_si128(v3, 1));
            _mm_storel_epi64((__m128i*)(dst + x), v0);
            _mm_storel_epi64((__m128i*)(dst + x + 8), v1);
            _mm_storel_epi64((__m128i*)(dst + x + 16), v2);
        }

        for( ; x < len*3; x += 3 )
        {
            int v0 = src[x], v1 = src[x+1], v2 = src[x+2];
            uchar t0 = saturate_cast<uchar>((m00*v0 + m01*v1 + m02*v2 + m03)>>BITS);
            uchar t1 = saturate_cast<uchar>((m10*v0 + m11*v1 + m12*v2 + m13)>>BITS);
            uchar t2 = saturate_cast<uchar>((m20*v0 + m21*v1 + m22*v2 + m23)>>BITS);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
        return;
    }
#endif

    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_16u( const ushort* src, ushort* dst, const float* m, int len, int scn, int dcn )
{
#if CV_SSE2
    if( USE_SSE2 && scn == 3 && dcn == 3 )
    {
        __m128 m0, m1, m2, m3;
        __m128i delta = _mm_setr_epi16(0,-32768,-32768,-32768,-32768,-32768,-32768,0);
        load3x3Matrix(m, m0, m1, m2, m3);
        m3 = _mm_sub_ps(m3, _mm_setr_ps(32768.f, 32768.f, 32768.f, 0.f));

        int x = 0;
        for( ; x <= (len - 4)*3; x += 4*3 )
        {
            __m128i z = _mm_setzero_si128();
            __m128i v0 = _mm_loadu_si128((const __m128i*)(src + x)), v1;
            __m128i v2 = _mm_loadl_epi64((const __m128i*)(src + x + 8)), v3;
            v1 = _mm_unpacklo_epi16(_mm_srli_si128(v0, 6), z); // b1 g1 r1
            v3 = _mm_unpacklo_epi16(_mm_srli_si128(v2, 2), z); // b3 g3 r3
            v2 = _mm_or_si128(_mm_srli_si128(v0, 12), _mm_slli_si128(v2, 4));
            v0 = _mm_unpacklo_epi16(v0, z); // b0 g0 r0
            v2 = _mm_unpacklo_epi16(v2, z); // b2 g2 r2
            __m128 x0 = _mm_cvtepi32_ps(v0), x1 = _mm_cvtepi32_ps(v1);
            __m128 x2 = _mm_cvtepi32_ps(v2), x3 = _mm_cvtepi32_ps(v3);
            __m128 y0 = _mm_add_ps(_mm_add_ps(_mm_add_ps(
                        _mm_mul_ps(m0, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(0,0,0,0))),
                        _mm_mul_ps(m1, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(1,1,1,1)))),
                        _mm_mul_ps(m2, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(2,2,2,2)))), m3);
            __m128 y1 = _mm_add_ps(_mm_add_ps(_mm_add_ps(
                        _mm_mul_ps(m0, _mm_shuffle_ps(x1,x1,_MM_SHUFFLE(0,0,0,0))),
                        _mm_mul_ps(m1, _mm_shuffle_ps(x1,x1,_MM_SHUFFLE(1,1,1,1)))),
                        _mm_mul_ps(m2, _mm_shuffle_ps(x1,x1,_MM_SHUFFLE(2,2,2,2)))), m3);
            __m128 y2 = _mm_add_ps(_mm_add_ps(_mm_add_ps(
                        _mm_mul_ps(m0, _mm_shuffle_ps(x2,x2,_MM_SHUFFLE(0,0,0,0))),
                        _mm_mul_ps(m1, _mm_shuffle_ps(x2,x2,_MM_SHUFFLE(1,1,1,1)))),
                        _mm_mul_ps(m2, _mm_shuffle_ps(x2,x2,_MM_SHUFFLE(2,2,2,2)))), m3);
            __m128 y3 = _mm_add_ps(_mm_add_ps(_mm_add_ps(
                        _mm_mul_ps(m0, _mm_shuffle_ps(x3,x3,_MM_SHUFFLE(0,0,0,0))),
                        _mm_mul_ps(m1, _mm_shuffle_ps(x3,x3,_MM_SHUFFLE(1,1,1,1)))),
                        _mm_mul_ps(m2, _mm_shuffle_ps(x3,x3,_MM_SHUFFLE(2,2,2,2)))), m3);
            v0 = _mm_cvtps_epi32(y0); v1 = _mm_cvtps_epi32(y1);
            v2 = _mm_cvtps_epi32(y2); v3 = _mm_cvtps_epi32(y3);

            v0 = _mm_add_epi16(_mm_packs_epi32(_mm_slli_si128(v0,4), v1), delta); // 0 b0 g0 r0 b1 g1 r1 0
            v2 = _mm_add_epi16(_mm_packs_epi32(_mm_slli_si128(v2,4), v3), delta); // 0 b2 g2 r2 b3 g3 r3 0
            v1 = _mm_or_si128(_mm_srli_si128(v0,2), _mm_slli_si128(v2,10)); // b0 g0 r0 b1 g1 r1 b2 g2
            v2 = _mm_srli_si128(v2, 6); // r2 b3 g3 r3 0 0 0 0
            _mm_storeu_si128((__m128i*)(dst + x), v1);
            _mm_storel_epi64((__m128i*)(dst + x + 8), v2);
        }

        for( ; x < len*3; x += 3 )
        {
            float v0 = src[x], v1 = src[x+1], v2 = src[x+2];
            ushort t0 = saturate_cast<ushort>(m[0]*v0 + m[1]*v1 + m[2]*v2 + m[3]);
            ushort t1 = saturate_cast<ushort>(m[4]*v0 + m[5]*v1 + m[6]*v2 + m[7]);
            ushort t2 = saturate_cast<ushort>(m[8]*v0 + m[9]*v1 + m[10]*v2 + m[11]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
        return;
    }
#endif

    transform_(src, dst, m, len, scn, dcn);
}


static void
transform_32f( const float* src, float* dst, const float* m, int len, int scn, int dcn )
{
#if CV_SSE2
    if( USE_SSE2 )
    {
        int x = 0;
        if( scn == 3 && dcn == 3 )
        {
            __m128 m0, m1, m2, m3;
            load3x3Matrix(m, m0, m1, m2, m3);

            for( ; x < (len - 1)*3; x += 3 )
            {
                __m128 x0 = _mm_loadu_ps(src + x);
                __m128 y0 = _mm_add_ps(_mm_add_ps(_mm_add_ps(
                            _mm_mul_ps(m0, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(0,0,0,0))),
                            _mm_mul_ps(m1, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(1,1,1,1)))),
                            _mm_mul_ps(m2, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(2,2,2,2)))), m3);
                _mm_storel_pi((__m64*)(dst + x), y0);
                _mm_store_ss(dst + x + 2, _mm_movehl_ps(y0,y0));
            }

            for( ; x < len*3; x += 3 )
            {
                float v0 = src[x], v1 = src[x+1], v2 = src[x+2];
                float t0 = saturate_cast<float>(m[0]*v0 + m[1]*v1 + m[2]*v2 + m[3]);
                float t1 = saturate_cast<float>(m[4]*v0 + m[5]*v1 + m[6]*v2 + m[7]);
                float t2 = saturate_cast<float>(m[8]*v0 + m[9]*v1 + m[10]*v2 + m[11]);
                dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
            }
            return;
        }

        if( scn == 4 && dcn == 4 )
        {
            __m128 m0, m1, m2, m3, m4;
            load4x4Matrix(m, m0, m1, m2, m3, m4);

            for( ; x < len*4; x += 4 )
            {
                __m128 x0 = _mm_loadu_ps(src + x);
                __m128 y0 = _mm_add_ps(_mm_add_ps(_mm_add_ps(_mm_add_ps(
                                    _mm_mul_ps(m0, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(0,0,0,0))),
                                    _mm_mul_ps(m1, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(1,1,1,1)))),
                                    _mm_mul_ps(m2, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(2,2,2,2)))),
                                    _mm_mul_ps(m3, _mm_shuffle_ps(x0,x0,_MM_SHUFFLE(3,3,3,3)))), m4);
                _mm_storeu_ps(dst + x, y0);
            }
            return;
        }
    }
#endif

    transform_(src, dst, m, len, scn, dcn);
}


static void
transform_8s(const schar* src, schar* dst, const float* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_16s(const short* src, short* dst, const float* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_32s(const int* src, int* dst, const double* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

static void
transform_64f(const double* src, double* dst, const double* m, int len, int scn, int dcn)
{
    transform_(src, dst, m, len, scn, dcn);
}

template<typename T, typename WT> static void
diagtransform_( const T* src, T* dst, const WT* m, int len, int cn, int )
{
    int x;

    if( cn == 2 )
    {
        for( x = 0; x < len*2; x += 2 )
        {
            T t0 = saturate_cast<T>(m[0]*src[x] + m[2]);
            T t1 = saturate_cast<T>(m[4]*src[x+1] + m[5]);
            dst[x] = t0; dst[x+1] = t1;
        }
    }
    else if( cn == 3 )
    {
        for( x = 0; x < len*3; x += 3 )
        {
            T t0 = saturate_cast<T>(m[0]*src[x] + m[3]);
            T t1 = saturate_cast<T>(m[5]*src[x+1] + m[7]);
            T t2 = saturate_cast<T>(m[10]*src[x+2] + m[11]);
            dst[x] = t0; dst[x+1] = t1; dst[x+2] = t2;
        }
    }
    else if( cn == 4 )
    {
        for( x = 0; x < len*4; x += 4 )
        {
            T t0 = saturate_cast<T>(m[0]*src[x] + m[4]);
            T t1 = saturate_cast<T>(m[6]*src[x+1] + m[9]);
            dst[x] = t0; dst[x+1] = t1;
            t0 = saturate_cast<T>(m[12]*src[x+2] + m[14]);
            t1 = saturate_cast<T>(m[18]*src[x+3] + m[19]);
            dst[x+2] = t0; dst[x+3] = t1;
        }
    }
    else
    {
        for( x = 0; x < len; x++, src += cn, dst += cn )
        {
            const WT* _m = m;
            for( int j = 0; j < cn; j++, _m += cn + 1 )
                dst[j] = saturate_cast<T>(src[j]*_m[j] + _m[cn]);
        }
    }
}

static void
diagtransform_8u(const uchar* src, uchar* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_8s(const schar* src, schar* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_16u(const ushort* src, ushort* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_16s(const short* src, short* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_32s(const int* src, int* dst, const double* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_32f(const float* src, float* dst, const float* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}

static void
diagtransform_64f(const double* src, double* dst, const double* m, int len, int scn, int dcn)
{
    diagtransform_(src, dst, m, len, scn, dcn);
}


typedef void (*TransformFunc)( const uchar* src, uchar* dst, const uchar* m, int, int, int );

static TransformFunc getTransformFunc(int depth)
{
    static TransformFunc transformTab[] =
    {
        (TransformFunc)transform_8u, (TransformFunc)transform_8s, (TransformFunc)transform_16u,
        (TransformFunc)transform_16s, (TransformFunc)transform_32s, (TransformFunc)transform_32f,
        (TransformFunc)transform_64f, 0
    };

    return transformTab[depth];
}

static TransformFunc getDiagTransformFunc(int depth)
{
    static TransformFunc diagTransformTab[] =
    {
        (TransformFunc)diagtransform_8u, (TransformFunc)diagtransform_8s, (TransformFunc)diagtransform_16u,
        (TransformFunc)diagtransform_16s, (TransformFunc)diagtransform_32s, (TransformFunc)diagtransform_32f,
        (TransformFunc)diagtransform_64f, 0
    };

    return diagTransformTab[depth];
}

}

void cv::transform( InputArray _src, OutputArray _dst, InputArray _mtx )
{
    Mat src = _src.getMat(), m = _mtx.getMat();
    int depth = src.depth(), scn = src.channels(), dcn = m.rows;
    CV_Assert( scn == m.cols || scn + 1 == m.cols );
    bool isDiag = false;

    _dst.create( src.size(), CV_MAKETYPE(depth, dcn) );
    Mat dst = _dst.getMat();

    int mtype = depth == CV_32S || depth == CV_64F ? CV_64F : CV_32F;
    AutoBuffer<double> _mbuf;
    double* mbuf;

    if( !m.isContinuous() || m.type() != mtype || m.cols != scn + 1 )
    {
        _mbuf.allocate(dcn*(scn+1));
        mbuf = (double*)_mbuf;
        Mat tmp(dcn, scn+1, mtype, mbuf);
        memset(tmp.data, 0, tmp.total()*tmp.elemSize());
        if( m.cols == scn+1 )
            m.convertTo(tmp, mtype);
        else
        {
            Mat tmppart = tmp.colRange(0, m.cols);
            m.convertTo(tmppart, mtype);
        }
        m = tmp;
    }
    else
        mbuf = (double*)m.data;

    if( scn == dcn )
    {
        int i, j;
        double eps = mtype == CV_32F ? FLT_EPSILON : DBL_EPSILON;

        if( scn == 1 )
        {
            double alpha, beta;
            if( mtype == CV_32F )
                alpha = m.at<float>(0), beta = m.at<float>(1);
            else
                alpha = m.at<double>(0), beta = m.at<double>(1);
            src.convertTo(dst, dst.type(), alpha, beta);
            return;
        }

        for( i = 0, isDiag = true; isDiag && i < scn; i++ )
        {
            for( j = 0; isDiag && j < scn; j++ )
            {
                double v = mtype == CV_32F ? m.at<float>(i, j) : m.at<double>(i, j);
                if( i != j && fabs(v) > eps )
                    isDiag = false;
            }
        }
    }

    TransformFunc func = isDiag ? getDiagTransformFunc(depth): getTransformFunc(depth);
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    size_t i, total = it.size;

    for( i = 0; i < it.nplanes; i++, ++it )
        func( ptrs[0], ptrs[1], (uchar*)mbuf, (int)total, scn, dcn );
}

/****************************************************************************************\
*                                  Perspective Transform                                 *
\****************************************************************************************/

namespace cv
{

template<typename T> static void
perspectiveTransform_( const T* src, T* dst, const double* m, int len, int scn, int dcn )
{
    const double eps = FLT_EPSILON;
    int i;

    if( scn == 2 && dcn == 2 )
    {
        for( i = 0; i < len*2; i += 2 )
        {
            T x = src[i], y = src[i + 1];
            double w = x*m[6] + y*m[7] + m[8];

            if( fabs(w) > eps )
            {
                w = 1./w;
                dst[i] = (T)((x*m[0] + y*m[1] + m[2])*w);
                dst[i+1] = (T)((x*m[3] + y*m[4] + m[5])*w);
            }
            else
                dst[i] = dst[i+1] = (T)0;
        }
    }
    else if( scn == 3 && dcn == 3 )
    {
        for( i = 0; i < len*3; i += 3 )
        {
            T x = src[i], y = src[i + 1], z = src[i + 2];
            double w = x*m[12] + y*m[13] + z*m[14] + m[15];

            if( fabs(w) > eps )
            {
                w = 1./w;
                dst[i] = (T)((x*m[0] + y*m[1] + z*m[2] + m[3]) * w);
                dst[i+1] = (T)((x*m[4] + y*m[5] + z*m[6] + m[7]) * w);
                dst[i+2] = (T)((x*m[8] + y*m[9] + z*m[10] + m[11]) * w);
            }
            else
                dst[i] = dst[i+1] = dst[i+2] = (T)0;
        }
    }
    else if( scn == 3 && dcn == 2 )
    {
        for( i = 0; i < len; i++, src += 3, dst += 2 )
        {
            T x = src[0], y = src[1], z = src[2];
            double w = x*m[8] + y*m[9] + z*m[10] + m[11];

            if( fabs(w) > eps )
            {
                w = 1./w;
                dst[0] = (T)((x*m[0] + y*m[1] + z*m[2] + m[3])*w);
                dst[1] = (T)((x*m[4] + y*m[5] + z*m[6] + m[7])*w);
            }
            else
                dst[0] = dst[1] = (T)0;
        }
    }
    else
    {
        for( i = 0; i < len; i++, src += scn, dst += dcn )
        {
            const double* _m = m + dcn*(scn + 1);
            double w = _m[scn];
            int j, k;
            for( k = 0; k < scn; k++ )
                w += _m[k]*src[k];
            if( fabs(w) > eps )
            {
                _m = m;
                for( j = 0; j < dcn; j++, _m += scn + 1 )
                {
                    double s = _m[scn];
                    for( k = 0; k < scn; k++ )
                        s += _m[k]*src[k];
                    dst[j] = (T)(s*w);
                }
            }
            else
                for( j = 0; j < dcn; j++ )
                    dst[j] = 0;
        }
    }
}


static void
perspectiveTransform_32f(const float* src, float* dst, const double* m, int len, int scn, int dcn)
{
    perspectiveTransform_(src, dst, m, len, scn, dcn);
}

static void
perspectiveTransform_64f(const double* src, double* dst, const double* m, int len, int scn, int dcn)
{
    perspectiveTransform_(src, dst, m, len, scn, dcn);
}

}

void cv::perspectiveTransform( InputArray _src, OutputArray _dst, InputArray _mtx )
{
    Mat src = _src.getMat(), m = _mtx.getMat();
    int depth = src.depth(), scn = src.channels(), dcn = m.rows-1;
    CV_Assert( scn + 1 == m.cols && (depth == CV_32F || depth == CV_64F));

    _dst.create( src.size(), CV_MAKETYPE(depth, dcn) );
    Mat dst = _dst.getMat();

    const int mtype = CV_64F;
    AutoBuffer<double> _mbuf;
    double* mbuf = _mbuf;

    if( !m.isContinuous() || m.type() != mtype )
    {
        _mbuf.allocate((dcn+1)*(scn+1));
        Mat tmp(dcn+1, scn+1, mtype, (double*)_mbuf);
        m.convertTo(tmp, mtype);
        m = tmp;
    }
    else
        mbuf = (double*)m.data;

    TransformFunc func = depth == CV_32F ?
        (TransformFunc)perspectiveTransform_32f :
        (TransformFunc)perspectiveTransform_64f;
    CV_Assert( func != 0 );

    const Mat* arrays[] = {&src, &dst, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    size_t i, total = it.size;

    for( i = 0; i < it.nplanes; i++, ++it )
        func( ptrs[0], ptrs[1], (uchar*)mbuf, (int)total, scn, dcn );
}

/****************************************************************************************\
*                                       ScaleAdd                                         *
\****************************************************************************************/

namespace cv
{

static void scaleAdd_32f(const float* src1, const float* src2, float* dst,
                         int len, float* _alpha)
{
    float alpha = *_alpha;
    int i = 0;
#if CV_SSE2
    if( USE_SSE2 )
    {
        __m128 a4 = _mm_set1_ps(alpha);
        if( (((size_t)src1|(size_t)src2|(size_t)dst) & 15) == 0 )
            for( ; i <= len - 8; i += 8 )
            {
                __m128 x0, x1, y0, y1, t0, t1;
                x0 = _mm_load_ps(src1 + i); x1 = _mm_load_ps(src1 + i + 4);
                y0 = _mm_load_ps(src2 + i); y1 = _mm_load_ps(src2 + i + 4);
                t0 = _mm_add_ps(_mm_mul_ps(x0, a4), y0);
                t1 = _mm_add_ps(_mm_mul_ps(x1, a4), y1);
                _mm_store_ps(dst + i, t0);
                _mm_store_ps(dst + i + 4, t1);
            }
        else
            for( ; i <= len - 8; i += 8 )
            {
                __m128 x0, x1, y0, y1, t0, t1;
                x0 = _mm_loadu_ps(src1 + i); x1 = _mm_loadu_ps(src1 + i + 4);
                y0 = _mm_loadu_ps(src2 + i); y1 = _mm_loadu_ps(src2 + i + 4);
                t0 = _mm_add_ps(_mm_mul_ps(x0, a4), y0);
                t1 = _mm_add_ps(_mm_mul_ps(x1, a4), y1);
                _mm_storeu_ps(dst + i, t0);
                _mm_storeu_ps(dst + i + 4, t1);
            }
    }
    else
#endif
    //vz why do we need unroll here?
    for( ; i <= len - 4; i += 4 )
    {
        float t0, t1;
        t0 = src1[i]*alpha + src2[i];
        t1 = src1[i+1]*alpha + src2[i+1];
        dst[i] = t0; dst[i+1] = t1;
        t0 = src1[i+2]*alpha + src2[i+2];
        t1 = src1[i+3]*alpha + src2[i+3];
        dst[i+2] = t0; dst[i+3] = t1;
    }
    for(; i < len; i++ )
        dst[i] = src1[i]*alpha + src2[i];
}


static void scaleAdd_64f(const double* src1, const double* src2, double* dst,
                         int len, double* _alpha)
{
    double alpha = *_alpha;
    int i = 0;
#if CV_SSE2
    if( USE_SSE2 && (((size_t)src1|(size_t)src2|(size_t)dst) & 15) == 0 )
    {
        __m128d a2 = _mm_set1_pd(alpha);
        for( ; i <= len - 4; i += 4 )
        {
            __m128d x0, x1, y0, y1, t0, t1;
            x0 = _mm_load_pd(src1 + i); x1 = _mm_load_pd(src1 + i + 2);
            y0 = _mm_load_pd(src2 + i); y1 = _mm_load_pd(src2 + i + 2);
            t0 = _mm_add_pd(_mm_mul_pd(x0, a2), y0);
            t1 = _mm_add_pd(_mm_mul_pd(x1, a2), y1);
            _mm_store_pd(dst + i, t0);
            _mm_store_pd(dst + i + 2, t1);
        }
    }
    else
#endif
     //vz why do we need unroll here?
    for( ; i <= len - 4; i += 4 )
    {
        double t0, t1;
        t0 = src1[i]*alpha + src2[i];
        t1 = src1[i+1]*alpha + src2[i+1];
        dst[i] = t0; dst[i+1] = t1;
        t0 = src1[i+2]*alpha + src2[i+2];
        t1 = src1[i+3]*alpha + src2[i+3];
        dst[i+2] = t0; dst[i+3] = t1;
    }
    for(; i < len; i++ )
        dst[i] = src1[i]*alpha + src2[i];
}

typedef void (*ScaleAddFunc)(const uchar* src1, const uchar* src2, uchar* dst, int len, const void* alpha);

}

void cv::scaleAdd( InputArray _src1, double alpha, InputArray _src2, OutputArray _dst )
{
    Mat src1 = _src1.getMat(), src2 = _src2.getMat();
    int depth = src1.depth(), cn = src1.channels();

    CV_Assert( src1.type() == src2.type() );
    if( depth < CV_32F )
    {
        addWeighted(_src1, alpha, _src2, 1, 0, _dst, depth);
        return;
    }

    _dst.create(src1.dims, src1.size, src1.type());
    Mat dst = _dst.getMat();

    float falpha = (float)alpha;
    void* palpha = depth == CV_32F ? (void*)&falpha : (void*)&alpha;

    ScaleAddFunc func = depth == CV_32F ? (ScaleAddFunc)scaleAdd_32f : (ScaleAddFunc)scaleAdd_64f;

    if( src1.isContinuous() && src2.isContinuous() && dst.isContinuous() )
    {
        size_t len = src1.total()*cn;
        func(src1.data, src2.data, dst.data, (int)len, palpha);
        return;
    }

    const Mat* arrays[] = {&src1, &src2, &dst, 0};
    uchar* ptrs[3];
    NAryMatIterator it(arrays, ptrs);
    size_t i, len = it.size*cn;

    for( i = 0; i < it.nplanes; i++, ++it )
        func( ptrs[0], ptrs[1], ptrs[2], (int)len, palpha );
}

/****************************************************************************************\
*                                 Covariation Matrix                                     *
\****************************************************************************************/

void cv::calcCovarMatrix( const Mat* data, int nsamples, Mat& covar, Mat& _mean, int flags, int ctype )
{
    CV_Assert( data && nsamples > 0 );
    Size size = data[0].size();
    int sz = size.width * size.height, esz = (int)data[0].elemSize();
    int type = data[0].type();
    Mat mean;
    ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), _mean.depth()), CV_32F);

    if( (flags & CV_COVAR_USE_AVG) != 0 )
    {
        CV_Assert( _mean.size() == size );
        if( _mean.isContinuous() && _mean.type() == ctype )
            mean = _mean.reshape(1, 1);
        else
        {
            _mean.convertTo(mean, ctype);
            mean = mean.reshape(1, 1);
        }
    }

    Mat _data(nsamples, sz, type);

    for( int i = 0; i < nsamples; i++ )
    {
        CV_Assert( data[i].size() == size && data[i].type() == type );
        if( data[i].isContinuous() )
            memcpy( _data.ptr(i), data[i].data, sz*esz );
        else
        {
            Mat dataRow(size.height, size.width, type, _data.ptr(i));
            data[i].copyTo(dataRow);
        }
    }

    calcCovarMatrix( _data, covar, mean, (flags & ~(CV_COVAR_ROWS|CV_COVAR_COLS)) | CV_COVAR_ROWS, ctype );
    if( (flags & CV_COVAR_USE_AVG) == 0 )
        _mean = mean.reshape(1, size.height);
}

void cv::calcCovarMatrix( InputArray _src, OutputArray _covar, InputOutputArray _mean, int flags, int ctype )
{
    if(_src.kind() == _InputArray::STD_VECTOR_MAT)
    {
        std::vector<cv::Mat> src;
        _src.getMatVector(src);

        CV_Assert( src.size() > 0 );

        Size size = src[0].size();
        int type = src[0].type();

        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), _mean.depth()), CV_32F);

        Mat _data(static_cast<int>(src.size()), size.area(), type);

        int i = 0;
        for(std::vector<cv::Mat>::iterator each = src.begin(); each != src.end(); each++, i++ )
        {
            CV_Assert( (*each).size() == size && (*each).type() == type );
            Mat dataRow(size.height, size.width, type, _data.ptr(i));
            (*each).copyTo(dataRow);
        }

        Mat mean;
        if( (flags & CV_COVAR_USE_AVG) != 0 )
        {
            CV_Assert( _mean.size() == size );

            if( mean.type() != ctype )
            {
                mean = _mean.getMat();
                _mean.create(mean.size(), ctype);
                Mat tmp = _mean.getMat();
                mean.convertTo(tmp, ctype);
                mean = tmp;
            }

            mean = _mean.getMat().reshape(1, 1);
        }

        calcCovarMatrix( _data, _covar, mean, (flags & ~(CV_COVAR_ROWS|CV_COVAR_COLS)) | CV_COVAR_ROWS, ctype );

        if( (flags & CV_COVAR_USE_AVG) == 0 )
        {
            mean = mean.reshape(1, size.height);
            mean.copyTo(_mean);
        }
        return;
    }

    Mat data = _src.getMat(), mean;
    CV_Assert( ((flags & CV_COVAR_ROWS) != 0) ^ ((flags & CV_COVAR_COLS) != 0) );
    bool takeRows = (flags & CV_COVAR_ROWS) != 0;
    int type = data.type();
    int nsamples = takeRows ? data.rows : data.cols;
    CV_Assert( nsamples > 0 );
    Size size = takeRows ? Size(data.cols, 1) : Size(1, data.rows);

    if( (flags & CV_COVAR_USE_AVG) != 0 )
    {
        mean = _mean.getMat();
        ctype = std::max(std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), mean.depth()), CV_32F);
        CV_Assert( mean.size() == size );
        if( mean.type() != ctype )
        {
            _mean.create(mean.size(), ctype);
            Mat tmp = _mean.getMat();
            mean.convertTo(tmp, ctype);
            mean = tmp;
        }
    }
    else
    {
        ctype = std::max(CV_MAT_DEPTH(ctype >= 0 ? ctype : type), CV_32F);
        reduce( _src, _mean, takeRows ? 0 : 1, CV_REDUCE_AVG, ctype );
        mean = _mean.getMat();
    }

    mulTransposed( data, _covar, ((flags & CV_COVAR_NORMAL) == 0) ^ takeRows,
        mean, (flags & CV_COVAR_SCALE) != 0 ? 1./nsamples : 1, ctype );
}

/****************************************************************************************\
*                                        Mahalanobis                                     *
\****************************************************************************************/

double cv::Mahalanobis( InputArray _v1, InputArray _v2, InputArray _icovar )
{
    Mat v1 = _v1.getMat(), v2 = _v2.getMat(), icovar = _icovar.getMat();
    int type = v1.type(), depth = v1.depth();
    Size sz = v1.size();
    int i, j, len = sz.width*sz.height*v1.channels();
    AutoBuffer<double> buf(len);
    double result = 0;

    CV_Assert( type == v2.type() && type == icovar.type() &&
        sz == v2.size() && len == icovar.rows && len == icovar.cols );

    sz.width *= v1.channels();
    if( v1.isContinuous() && v2.isContinuous() )
    {
        sz.width *= sz.height;
        sz.height = 1;
    }

    if( depth == CV_32F )
    {
        const float* src1 = (const float*)v1.data;
        const float* src2 = (const float*)v2.data;
        size_t step1 = v1.step/sizeof(src1[0]);
        size_t step2 = v2.step/sizeof(src2[0]);
        double* diff = buf;
        const float* mat = (const float*)icovar.data;
        size_t matstep = icovar.step/sizeof(mat[0]);

        for( ; sz.height--; src1 += step1, src2 += step2, diff += sz.width )
        {
            for( i = 0; i < sz.width; i++ )
                diff[i] = src1[i] - src2[i];
        }

        diff = buf;
        for( i = 0; i < len; i++, mat += matstep )
        {
            double row_sum = 0;
            j = 0;
             #if CV_ENABLE_UNROLLED
            for(; j <= len - 4; j += 4 )
                row_sum += diff[j]*mat[j] + diff[j+1]*mat[j+1] +
                           diff[j+2]*mat[j+2] + diff[j+3]*mat[j+3];
            #endif
            for( ; j < len; j++ )
                row_sum += diff[j]*mat[j];
            result += row_sum * diff[i];
        }
    }
    else if( depth == CV_64F )
    {
        const double* src1 = (const double*)v1.data;
        const double* src2 = (const double*)v2.data;
        size_t step1 = v1.step/sizeof(src1[0]);
        size_t step2 = v2.step/sizeof(src2[0]);
        double* diff = buf;
        const double* mat = (const double*)icovar.data;
        size_t matstep = icovar.step/sizeof(mat[0]);

        for( ; sz.height--; src1 += step1, src2 += step2, diff += sz.width )
        {
            for( i = 0; i < sz.width; i++ )
                diff[i] = src1[i] - src2[i];
        }

        diff = buf;
        for( i = 0; i < len; i++, mat += matstep )
        {
            double row_sum = 0;
            j = 0;
             #if CV_ENABLE_UNROLLED
            for(; j <= len - 4; j += 4 )
                row_sum += diff[j]*mat[j] + diff[j+1]*mat[j+1] +
                           diff[j+2]*mat[j+2] + diff[j+3]*mat[j+3];
            #endif
            for( ; j < len; j++ )
                row_sum += diff[j]*mat[j];
            result += row_sum * diff[i];
        }
    }
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    return std::sqrt(result);
}

/****************************************************************************************\
*                                        MulTransposed                                   *
\****************************************************************************************/

namespace cv
{

template<typename sT, typename dT> static void
MulTransposedR( const Mat& srcmat, Mat& dstmat, const Mat& deltamat, double scale )
{
    int i, j, k;
    const sT* src = (const sT*)srcmat.data;
    dT* dst = (dT*)dstmat.data;
    const dT* delta = (const dT*)deltamat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    Size size = srcmat.size();
    dT* tdst = dst;
    dT* col_buf = 0;
    dT* delta_buf = 0;
    int buf_size = size.height*sizeof(dT);
    AutoBuffer<uchar> buf;

    if( delta && delta_cols < size.width )
    {
        assert( delta_cols == 1 );
        buf_size *= 5;
    }
    buf.allocate(buf_size);
    col_buf = (dT*)(uchar*)buf;

    if( delta && delta_cols < size.width )
    {
        delta_buf = col_buf + size.height;
        for( i = 0; i < size.height; i++ )
            delta_buf[i*4] = delta_buf[i*4+1] =
                delta_buf[i*4+2] = delta_buf[i*4+3] = delta[i*deltastep];
        delta = delta_buf;
        deltastep = deltastep ? 4 : 0;
    }

    if( !delta )
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            for( k = 0; k < size.height; k++ )
                col_buf[k] = src[k*srcstep+i];

            for( j = i; j <= size.width - 4; j += 4 )
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                {
                    double a = col_buf[k];
                    s0 += a * tsrc[0];
                    s1 += a * tsrc[1];
                    s2 += a * tsrc[2];
                    s3 += a * tsrc[3];
                }

                tdst[j] = (dT)(s0*scale);
                tdst[j+1] = (dT)(s1*scale);
                tdst[j+2] = (dT)(s2*scale);
                tdst[j+3] = (dT)(s3*scale);
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;

                for( k = 0; k < size.height; k++, tsrc += srcstep )
                    s0 += (double)col_buf[k] * tsrc[0];

                tdst[j] = (dT)(s0*scale);
            }
        }
    else
        for( i = 0; i < size.width; i++, tdst += dststep )
        {
            if( !delta_buf )
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta[k*deltastep+i];
            else
                for( k = 0; k < size.height; k++ )
                    col_buf[k] = src[k*srcstep+i] - delta_buf[k*deltastep];

            for( j = i; j <= size.width - 4; j += 4 )
            {
                double s0 = 0, s1 = 0, s2 = 0, s3 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                {
                    double a = col_buf[k];
                    s0 += a * (tsrc[0] - d[0]);
                    s1 += a * (tsrc[1] - d[1]);
                    s2 += a * (tsrc[2] - d[2]);
                    s3 += a * (tsrc[3] - d[3]);
                }

                tdst[j] = (dT)(s0*scale);
                tdst[j+1] = (dT)(s1*scale);
                tdst[j+2] = (dT)(s2*scale);
                tdst[j+3] = (dT)(s3*scale);
            }

            for( ; j < size.width; j++ )
            {
                double s0 = 0;
                const sT *tsrc = src + j;
                const dT *d = delta_buf ? delta_buf : delta + j;

                for( k = 0; k < size.height; k++, tsrc+=srcstep, d+=deltastep )
                    s0 += (double)col_buf[k] * (tsrc[0] - d[0]);

                tdst[j] = (dT)(s0*scale);
            }
        }
}


template<typename sT, typename dT> static void
MulTransposedL( const Mat& srcmat, Mat& dstmat, const Mat& deltamat, double scale )
{
    int i, j, k;
    const sT* src = (const sT*)srcmat.data;
    dT* dst = (dT*)dstmat.data;
    const dT* delta = (const dT*)deltamat.data;
    size_t srcstep = srcmat.step/sizeof(src[0]);
    size_t dststep = dstmat.step/sizeof(dst[0]);
    size_t deltastep = deltamat.rows > 1 ? deltamat.step/sizeof(delta[0]) : 0;
    int delta_cols = deltamat.cols;
    Size size = srcmat.size();
    dT* tdst = dst;

    if( !delta )
        for( i = 0; i < size.height; i++, tdst += dststep )
            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc1 = src + i*srcstep;
                const sT *tsrc2 = src + j*srcstep;

                for( k = 0; k <= size.width - 4; k += 4 )
                    s += (double)tsrc1[k]*tsrc2[k] + (double)tsrc1[k+1]*tsrc2[k+1] +
                         (double)tsrc1[k+2]*tsrc2[k+2] + (double)tsrc1[k+3]*tsrc2[k+3];
                for( ; k < size.width; k++ )
                    s += (double)tsrc1[k] * tsrc2[k];
                tdst[j] = (dT)(s*scale);
            }
    else
    {
        dT delta_buf[4];
        int delta_shift = delta_cols == size.width ? 4 : 0;
        AutoBuffer<uchar> buf(size.width*sizeof(dT));
        dT* row_buf = (dT*)(uchar*)buf;

        for( i = 0; i < size.height; i++, tdst += dststep )
        {
            const sT *tsrc1 = src + i*srcstep;
            const dT *tdelta1 = delta + i*deltastep;

            if( delta_cols < size.width )
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[0];
            else
                for( k = 0; k < size.width; k++ )
                    row_buf[k] = tsrc1[k] - tdelta1[k];

            for( j = i; j < size.height; j++ )
            {
                double s = 0;
                const sT *tsrc2 = src + j*srcstep;
                const dT *tdelta2 = delta + j*deltastep;
                if( delta_cols < size.width )
                {
                    delta_buf[0] = delta_buf[1] =
                        delta_buf[2] = delta_buf[3] = tdelta2[0];
                    tdelta2 = delta_buf;
                }
                for( k = 0; k <= size.width-4; k += 4, tdelta2 += delta_shift )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]) +
                         (double)row_buf[k+1]*(tsrc2[k+1] - tdelta2[1]) +
                         (double)row_buf[k+2]*(tsrc2[k+2] - tdelta2[2]) +
                         (double)row_buf[k+3]*(tsrc2[k+3] - tdelta2[3]);
                for( ; k < size.width; k++, tdelta2++ )
                    s += (double)row_buf[k]*(tsrc2[k] - tdelta2[0]);
                tdst[j] = (dT)(s*scale);
            }
        }
    }
}

typedef void (*MulTransposedFunc)(const Mat& src, Mat& dst, const Mat& delta, double scale);

}

void cv::mulTransposed( InputArray _src, OutputArray _dst, bool ata,
                        InputArray _delta, double scale, int dtype )
{
    Mat src = _src.getMat(), delta = _delta.getMat();
    const int gemm_level = 100; // boundary above which GEMM is faster.
    int stype = src.type();
    dtype = std::max(std::max(CV_MAT_DEPTH(dtype >= 0 ? dtype : stype), delta.depth()), CV_32F);
    CV_Assert( src.channels() == 1 );

    if( delta.data )
    {
        CV_Assert( delta.channels() == 1 &&
            (delta.rows == src.rows || delta.rows == 1) &&
            (delta.cols == src.cols || delta.cols == 1));
        if( delta.type() != dtype )
            delta.convertTo(delta, dtype);
    }

    int dsize = ata ? src.cols : src.rows;
    _dst.create( dsize, dsize, dtype );
    Mat dst = _dst.getMat();

    if( src.data == dst.data || (stype == dtype &&
        (dst.cols >= gemm_level && dst.rows >= gemm_level &&
         src.cols >= gemm_level && src.rows >= gemm_level)))
    {
        Mat src2;
        const Mat* tsrc = &src;
        if( delta.data )
        {
            if( delta.size() == src.size() )
                subtract( src, delta, src2 );
            else
            {
                repeat(delta, src.rows/delta.rows, src.cols/delta.cols, src2);
                subtract( src, src2, src2 );
            }
            tsrc = &src2;
        }
        gemm( *tsrc, *tsrc, scale, Mat(), 0, dst, ata ? GEMM_1_T : GEMM_2_T );
    }
    else
    {
        MulTransposedFunc func = 0;
        if(stype == CV_8U && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<uchar,float>;
            else
                func = MulTransposedL<uchar,float>;
        }
        else if(stype == CV_8U && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<uchar,double>;
            else
                func = MulTransposedL<uchar,double>;
        }
        else if(stype == CV_16U && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<ushort,float>;
            else
                func = MulTransposedL<ushort,float>;
        }
        else if(stype == CV_16U && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<ushort,double>;
            else
                func = MulTransposedL<ushort,double>;
        }
        else if(stype == CV_16S && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<short,float>;
            else
                func = MulTransposedL<short,float>;
        }
        else if(stype == CV_16S && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<short,double>;
            else
                func = MulTransposedL<short,double>;
        }
        else if(stype == CV_32F && dtype == CV_32F)
        {
            if(ata)
                func = MulTransposedR<float,float>;
            else
                func = MulTransposedL<float,float>;
        }
        else if(stype == CV_32F && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<float,double>;
            else
                func = MulTransposedL<float,double>;
        }
        else if(stype == CV_64F && dtype == CV_64F)
        {
            if(ata)
                func = MulTransposedR<double,double>;
            else
                func = MulTransposedL<double,double>;
        }
        if( !func )
            CV_Error( CV_StsUnsupportedFormat, "" );

        func( src, dst, delta, scale );
        completeSymm( dst, false );
    }
}

/****************************************************************************************\
*                                      Dot Product                                       *
\****************************************************************************************/

namespace cv
{

template<typename T> double
dotProd_(const T* src1, const T* src2, int len)
{
    int i = 0;
    double result = 0;
     #if CV_ENABLE_UNROLLED
    for( ; i <= len - 4; i += 4 )
        result += (double)src1[i]*src2[i] + (double)src1[i+1]*src2[i+1] +
            (double)src1[i+2]*src2[i+2] + (double)src1[i+3]*src2[i+3];
    #endif
    for( ; i < len; i++ )
        result += (double)src1[i]*src2[i];

    return result;
}


static double dotProd_8u(const uchar* src1, const uchar* src2, int len)
{
    double r = 0;
#if ARITHM_USE_IPP
    ippiDotProd_8u64f_C1R(src1, (int)(len*sizeof(src1[0])),
                          src2, (int)(len*sizeof(src2[0])),
                          ippiSize(len, 1), &r);
    return r;
#else
    int i = 0;

#if CV_SSE2
    if( USE_SSE2 )
    {
        int j, len0 = len & -4, blockSize0 = (1 << 13), blockSize;
        __m128i z = _mm_setzero_si128();
        while( i < len0 )
        {
            blockSize = std::min(len0 - i, blockSize0);
            __m128i s = _mm_setzero_si128();
            j = 0;
            for( ; j <= blockSize - 16; j += 16 )
            {
                __m128i b0 = _mm_loadu_si128((const __m128i*)(src1 + j));
                __m128i b1 = _mm_loadu_si128((const __m128i*)(src2 + j));
                __m128i s0, s1, s2, s3;
                s0 = _mm_unpacklo_epi8(b0, z);
                s2 = _mm_unpackhi_epi8(b0, z);
                s1 = _mm_unpacklo_epi8(b1, z);
                s3 = _mm_unpackhi_epi8(b1, z);
                s0 = _mm_madd_epi16(s0, s1);
                s2 = _mm_madd_epi16(s2, s3);
                s = _mm_add_epi32(s, s0);
                s = _mm_add_epi32(s, s2);
            }

            for( ; j < blockSize; j += 4 )
            {
                __m128i s0 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src1 + j)), z);
                __m128i s1 = _mm_unpacklo_epi8(_mm_cvtsi32_si128(*(const int*)(src2 + j)), z);
                s0 = _mm_madd_epi16(s0, s1);
                s = _mm_add_epi32(s, s0);
            }
            CV_DECL_ALIGNED(16) int buf[4];
            _mm_store_si128((__m128i*)buf, s);
            r += buf[0] + buf[1] + buf[2] + buf[3];

            src1 += blockSize;
            src2 += blockSize;
            i += blockSize;
        }
    }
#endif
    return r + dotProd_(src1, src2, len - i);
#endif
}


static double dotProd_8s(const schar* src1, const schar* src2, int len)
{
    return dotProd_(src1, src2, len);
}

static double dotProd_16u(const ushort* src1, const ushort* src2, int len)
{
    double r = 0;
    IF_IPP(ippiDotProd_16u64f_C1R(src1, (int)(len*sizeof(src1[0])),
                                  src2, (int)(len*sizeof(src2[0])),
                                  ippiSize(len, 1), &r),
           r = dotProd_(src1, src2, len));
    return r;
}

static double dotProd_16s(const short* src1, const short* src2, int len)
{
    double r = 0;
    IF_IPP(ippiDotProd_16s64f_C1R(src1, (int)(len*sizeof(src1[0])),
                                  src2, (int)(len*sizeof(src2[0])),
                                  ippiSize(len, 1), &r),
           r = dotProd_(src1, src2, len));
    return r;
}

static double dotProd_32s(const int* src1, const int* src2, int len)
{
    double r = 0;
    IF_IPP(ippiDotProd_32s64f_C1R(src1, (int)(len*sizeof(src1[0])),
                                  src2, (int)(len*sizeof(src2[0])),
                                  ippiSize(len, 1), &r),
           r = dotProd_(src1, src2, len));
    return r;
}

static double dotProd_32f(const float* src1, const float* src2, int len)
{
    double r = 0;
    IF_IPP(ippsDotProd_32f64f(src1, src2, len, &r),
           r = dotProd_(src1, src2, len));
    return r;
}

static double dotProd_64f(const double* src1, const double* src2, int len)
{
    double r = 0;
    IF_IPP(ippsDotProd_64f(src1, src2, len, &r),
           r = dotProd_(src1, src2, len));
    return r;
}


typedef double (*DotProdFunc)(const uchar* src1, const uchar* src2, int len);

static DotProdFunc getDotProdFunc(int depth)
{
    static DotProdFunc dotProdTab[] =
    {
        (DotProdFunc)GET_OPTIMIZED(dotProd_8u), (DotProdFunc)GET_OPTIMIZED(dotProd_8s),
        (DotProdFunc)dotProd_16u, (DotProdFunc)dotProd_16s,
        (DotProdFunc)dotProd_32s, (DotProdFunc)GET_OPTIMIZED(dotProd_32f),
        (DotProdFunc)dotProd_64f, 0
    };

    return dotProdTab[depth];
}

double Mat::dot(InputArray _mat) const
{
    Mat mat = _mat.getMat();
    int cn = channels();
    DotProdFunc func = getDotProdFunc(depth());
    CV_Assert( mat.type() == type() && mat.size == size && func != 0 );

    if( isContinuous() && mat.isContinuous() )
    {
        size_t len = total()*cn;
        if( len == (size_t)(int)len )
            return func(data, mat.data, (int)len);
    }

    const Mat* arrays[] = {this, &mat, 0};
    uchar* ptrs[2];
    NAryMatIterator it(arrays, ptrs);
    int len = (int)(it.size*cn);
    double r = 0;

    for( size_t i = 0; i < it.nplanes; i++, ++it )
        r += func( ptrs[0], ptrs[1], len );

    return r;
}

/****************************************************************************************\
*                                          PCA                                           *
\****************************************************************************************/

PCA::PCA() {}

PCA::PCA(InputArray data, InputArray _mean, int flags, int maxComponents)
{
    operator()(data, _mean, flags, maxComponents);
}

PCA::PCA(InputArray data, InputArray _mean, int flags, double retainedVariance)
{
    operator()(data, _mean, flags, retainedVariance);
}

PCA& PCA::operator()(InputArray _data, InputArray __mean, int flags, int maxComponents)
{
    Mat data = _data.getMat(), _mean = __mean.getMat();
    int covar_flags = CV_COVAR_SCALE;
    int i, len, in_count;
    Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= CV_COVAR_COLS;
        mean_sz = Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= CV_COVAR_ROWS;
        mean_sz = Size(len, 1);
    }

    int count = std::min(len, in_count), out_count = count;
    if( maxComponents > 0 )
        out_count = std::min(count, maxComponents);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= CV_COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    Mat covar( count, count, ctype );

    if( _mean.data )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
        covar_flags |= CV_COVAR_USE_AVG;
    }

    calcCovarMatrix( data, covar, mean, covar_flags, ctype );
    eigen( covar, eigenvalues, eigenvectors );

    if( !(covar_flags & CV_COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        Mat tmp_data, tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
        if( data.type() != ctype || tmp_mean.data == mean.data )
        {
            data.convertTo( tmp_data, ctype );
            subtract( tmp_data, tmp_mean, tmp_data );
        }
        else
        {
            subtract( data, tmp_mean, tmp_mean );
            tmp_data = tmp_mean;
        }

        Mat evects1(count, len, ctype);
        gemm( eigenvectors, tmp_data, 1, Mat(), 0, evects1,
            (flags & CV_PCA_DATA_AS_COL) ? CV_GEMM_B_T : 0);
        eigenvectors = evects1;

        // normalize eigenvectors
        for( i = 0; i < out_count; i++ )
        {
            Mat vec = eigenvectors.row(i);
            normalize(vec, vec);
        }
    }

    if( count > out_count )
    {
        // use clone() to physically copy the data and thus deallocate the original matrices
        eigenvalues = eigenvalues.rowRange(0,out_count).clone();
        eigenvectors = eigenvectors.rowRange(0,out_count).clone();
    }
    return *this;
}

void PCA::write(FileStorage& fs ) const
{
    CV_Assert( fs.isOpened() );

    fs << "name" << "PCA";
    fs << "vectors" << eigenvectors;
    fs << "values" << eigenvalues;
    fs << "mean" << mean;
}

void PCA::read(const FileNode& fs)
{
    CV_Assert( !fs.empty() );
    String name = (String)fs["name"];
    CV_Assert( name == "PCA" );

    cv::read(fs["vectors"], eigenvectors);
    cv::read(fs["values"], eigenvalues);
    cv::read(fs["mean"], mean);
}

template <typename T>
int computeCumulativeEnergy(const Mat& eigenvalues, double retainedVariance)
{
    CV_DbgAssert( eigenvalues.type() == DataType<T>::type );

    Mat g(eigenvalues.size(), DataType<T>::type);

    for(int ig = 0; ig < g.rows; ig++)
    {
        g.at<T>(ig, 0) = 0;
        for(int im = 0; im <= ig; im++)
        {
            g.at<T>(ig,0) += eigenvalues.at<T>(im,0);
        }
    }

    int L;

    for(L = 0; L < eigenvalues.rows; L++)
    {
        double energy = g.at<T>(L, 0) / g.at<T>(g.rows - 1, 0);
        if(energy > retainedVariance)
            break;
    }

    L = std::max(2, L);

    return L;
}

PCA& PCA::operator()(InputArray _data, InputArray __mean, int flags, double retainedVariance)
{
    Mat data = _data.getMat(), _mean = __mean.getMat();
    int covar_flags = CV_COVAR_SCALE;
    int i, len, in_count;
    Size mean_sz;

    CV_Assert( data.channels() == 1 );
    if( flags & CV_PCA_DATA_AS_COL )
    {
        len = data.rows;
        in_count = data.cols;
        covar_flags |= CV_COVAR_COLS;
        mean_sz = Size(1, len);
    }
    else
    {
        len = data.cols;
        in_count = data.rows;
        covar_flags |= CV_COVAR_ROWS;
        mean_sz = Size(len, 1);
    }

    CV_Assert( retainedVariance > 0 && retainedVariance <= 1 );

    int count = std::min(len, in_count);

    // "scrambled" way to compute PCA (when cols(A)>rows(A)):
    // B = A'A; B*x=b*x; C = AA'; C*y=c*y -> AA'*y=c*y -> A'A*(A'*y)=c*(A'*y) -> c = b, x=A'*y
    if( len <= in_count )
        covar_flags |= CV_COVAR_NORMAL;

    int ctype = std::max(CV_32F, data.depth());
    mean.create( mean_sz, ctype );

    Mat covar( count, count, ctype );

    if( _mean.data )
    {
        CV_Assert( _mean.size() == mean_sz );
        _mean.convertTo(mean, ctype);
    }

    calcCovarMatrix( data, covar, mean, covar_flags, ctype );
    eigen( covar, eigenvalues, eigenvectors );

    if( !(covar_flags & CV_COVAR_NORMAL) )
    {
        // CV_PCA_DATA_AS_ROW: cols(A)>rows(A). x=A'*y -> x'=y'*A
        // CV_PCA_DATA_AS_COL: rows(A)>cols(A). x=A''*y -> x'=y'*A'
        Mat tmp_data, tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
        if( data.type() != ctype || tmp_mean.data == mean.data )
        {
            data.convertTo( tmp_data, ctype );
            subtract( tmp_data, tmp_mean, tmp_data );
        }
        else
        {
            subtract( data, tmp_mean, tmp_mean );
            tmp_data = tmp_mean;
        }

        Mat evects1(count, len, ctype);
        gemm( eigenvectors, tmp_data, 1, Mat(), 0, evects1,
            (flags & CV_PCA_DATA_AS_COL) ? CV_GEMM_B_T : 0);
        eigenvectors = evects1;

        // normalize all eigenvectors
        for( i = 0; i < eigenvectors.rows; i++ )
        {
            Mat vec = eigenvectors.row(i);
            normalize(vec, vec);
        }
    }

    // compute the cumulative energy content for each eigenvector
    int L;
    if (ctype == CV_32F)
        L = computeCumulativeEnergy<float>(eigenvalues, retainedVariance);
    else
        L = computeCumulativeEnergy<double>(eigenvalues, retainedVariance);

    // use clone() to physically copy the data and thus deallocate the original matrices
    eigenvalues = eigenvalues.rowRange(0,L).clone();
    eigenvectors = eigenvectors.rowRange(0,L).clone();

    return *this;
}

void PCA::project(InputArray _data, OutputArray result) const
{
    Mat data = _data.getMat();
    CV_Assert( mean.data && eigenvectors.data &&
        ((mean.rows == 1 && mean.cols == data.cols) || (mean.cols == 1 && mean.rows == data.rows)));
    Mat tmp_data, tmp_mean = repeat(mean, data.rows/mean.rows, data.cols/mean.cols);
    int ctype = mean.type();
    if( data.type() != ctype || tmp_mean.data == mean.data )
    {
        data.convertTo( tmp_data, ctype );
        subtract( tmp_data, tmp_mean, tmp_data );
    }
    else
    {
        subtract( data, tmp_mean, tmp_mean );
        tmp_data = tmp_mean;
    }
    if( mean.rows == 1 )
        gemm( tmp_data, eigenvectors, 1, Mat(), 0, result, GEMM_2_T );
    else
        gemm( eigenvectors, tmp_data, 1, Mat(), 0, result, 0 );
}

Mat PCA::project(InputArray data) const
{
    Mat result;
    project(data, result);
    return result;
}

void PCA::backProject(InputArray _data, OutputArray result) const
{
    Mat data = _data.getMat();
    CV_Assert( mean.data && eigenvectors.data &&
        ((mean.rows == 1 && eigenvectors.rows == data.cols) ||
         (mean.cols == 1 && eigenvectors.rows == data.rows)));

    Mat tmp_data, tmp_mean;
    data.convertTo(tmp_data, mean.type());
    if( mean.rows == 1 )
    {
        tmp_mean = repeat(mean, data.rows, 1);
        gemm( tmp_data, eigenvectors, 1, tmp_mean, 1, result, 0 );
    }
    else
    {
        tmp_mean = repeat(mean, 1, data.cols);
        gemm( eigenvectors, tmp_data, 1, tmp_mean, 1, result, GEMM_1_T );
    }
}

Mat PCA::backProject(InputArray data) const
{
    Mat result;
    backProject(data, result);
    return result;
}

}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, int maxComponents)
{
    PCA pca;
    pca(data, mean, 0, maxComponents);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}

void cv::PCACompute(InputArray data, InputOutputArray mean,
                    OutputArray eigenvectors, double retainedVariance)
{
    PCA pca;
    pca(data, mean, 0, retainedVariance);
    pca.mean.copyTo(mean);
    pca.eigenvectors.copyTo(eigenvectors);
}

void cv::PCAProject(InputArray data, InputArray mean,
                    InputArray eigenvectors, OutputArray result)
{
    PCA pca;
    pca.mean = mean.getMat();
    pca.eigenvectors = eigenvectors.getMat();
    pca.project(data, result);
}

void cv::PCABackProject(InputArray data, InputArray mean,
                    InputArray eigenvectors, OutputArray result)
{
    PCA pca;
    pca.mean = mean.getMat();
    pca.eigenvectors = eigenvectors.getMat();
    pca.backProject(data, result);
}


/****************************************************************************************\
*                                    Earlier API                                         *
\****************************************************************************************/

CV_IMPL void cvGEMM( const CvArr* Aarr, const CvArr* Barr, double alpha,
                     const CvArr* Carr, double beta, CvArr* Darr, int flags )
{
    cv::Mat A = cv::cvarrToMat(Aarr), B = cv::cvarrToMat(Barr);
    cv::Mat C, D = cv::cvarrToMat(Darr);

    if( Carr )
        C = cv::cvarrToMat(Carr);

    CV_Assert( (D.rows == ((flags & CV_GEMM_A_T) == 0 ? A.rows : A.cols)) &&
               (D.cols == ((flags & CV_GEMM_B_T) == 0 ? B.cols : B.rows)) &&
               D.type() == A.type() );

    gemm( A, B, alpha, C, beta, D, flags );
}


CV_IMPL void
cvTransform( const CvArr* srcarr, CvArr* dstarr,
             const CvMat* transmat, const CvMat* shiftvec )
{
    cv::Mat m = cv::cvarrToMat(transmat), src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    if( shiftvec )
    {
        cv::Mat v = cv::cvarrToMat(shiftvec).reshape(1,m.rows),
            _m(m.rows, m.cols + 1, m.type()), m1 = _m.colRange(0,m.cols), v1 = _m.col(m.cols);
        m.convertTo(m1, m1.type());
        v.convertTo(v1, v1.type());
        m = _m;
    }

    CV_Assert( dst.depth() == src.depth() && dst.channels() == m.rows );
    cv::transform( src, dst, m );
}


CV_IMPL void
cvPerspectiveTransform( const CvArr* srcarr, CvArr* dstarr, const CvMat* mat )
{
    cv::Mat m = cv::cvarrToMat(mat), src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( dst.type() == src.type() && dst.channels() == m.rows-1 );
    cv::perspectiveTransform( src, dst, m );
}


CV_IMPL void cvScaleAdd( const CvArr* srcarr1, CvScalar scale,
                         const CvArr* srcarr2, CvArr* dstarr )
{
    cv::Mat src1 = cv::cvarrToMat(srcarr1), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src1.size == dst.size && src1.type() == dst.type() );
    cv::scaleAdd( src1, scale.val[0], cv::cvarrToMat(srcarr2), dst );
}


CV_IMPL void
cvCalcCovarMatrix( const CvArr** vecarr, int count,
                   CvArr* covarr, CvArr* avgarr, int flags )
{
    cv::Mat cov0 = cv::cvarrToMat(covarr), cov = cov0, mean0, mean;
    CV_Assert( vecarr != 0 && count >= 1 );

    if( avgarr )
        mean = mean0 = cv::cvarrToMat(avgarr);

    if( (flags & CV_COVAR_COLS) != 0 || (flags & CV_COVAR_ROWS) != 0 )
    {

        cv::Mat data = cv::cvarrToMat(vecarr[0]);
        cv::calcCovarMatrix( data, cov, mean, flags, cov.type() );
    }
    else
    {
        std::vector<cv::Mat> data(count);
        for( int i = 0; i < count; i++ )
            data[i] = cv::cvarrToMat(vecarr[i]);
        cv::calcCovarMatrix( &data[0], count, cov, mean, flags, cov.type() );
    }

    if( mean.data != mean0.data && mean0.data )
        mean.convertTo(mean0, mean0.type());

    if( cov.data != cov0.data )
        cov.convertTo(cov0, cov0.type());
}


CV_IMPL double
cvMahalanobis( const CvArr* srcAarr, const CvArr* srcBarr, const CvArr* matarr )
{
    return cv::Mahalanobis(cv::cvarrToMat(srcAarr),
        cv::cvarrToMat(srcBarr), cv::cvarrToMat(matarr));
}

CV_IMPL void
cvMulTransposed( const CvArr* srcarr, CvArr* dstarr,
                 int order, const CvArr* deltaarr, double scale )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst0 = cv::cvarrToMat(dstarr), dst = dst0, delta;
    if( deltaarr )
        delta = cv::cvarrToMat(deltaarr);
    cv::mulTransposed( src, dst, order != 0, delta, scale, dst.type());
    if( dst.data != dst0.data )
        dst.convertTo(dst0, dst0.type());
}

CV_IMPL double cvDotProduct( const CvArr* srcAarr, const CvArr* srcBarr )
{
    return cv::cvarrToMat(srcAarr).dot(cv::cvarrToMat(srcBarr));
}


CV_IMPL void
cvCalcPCA( const CvArr* data_arr, CvArr* avg_arr, CvArr* eigenvals, CvArr* eigenvects, int flags )
{
    cv::Mat data = cv::cvarrToMat(data_arr), mean0 = cv::cvarrToMat(avg_arr);
    cv::Mat evals0 = cv::cvarrToMat(eigenvals), evects0 = cv::cvarrToMat(eigenvects);
    cv::Mat mean = mean0, evals = evals0, evects = evects0;

    cv::PCA pca;
    pca.mean = mean;
    pca.eigenvalues = evals;
    pca.eigenvectors = evects;

    pca(data, (flags & CV_PCA_USE_AVG) ? mean : cv::Mat(),
        flags, evals.data ? evals.rows + evals.cols - 1 : 0);

    if( pca.mean.size() == mean.size() )
        pca.mean.convertTo( mean, mean.type() );
    else
    {
        cv::Mat temp; pca.mean.convertTo( temp, mean.type() );
        transpose( temp, mean );
    }

    evals = pca.eigenvalues;
    evects = pca.eigenvectors;
    int ecount0 = evals0.cols + evals0.rows - 1;
    int ecount = evals.cols + evals.rows - 1;

    CV_Assert( (evals0.cols == 1 || evals0.rows == 1) &&
                ecount0 <= ecount &&
                evects0.cols == evects.cols &&
                evects0.rows == ecount0 );

    cv::Mat temp = evals0;
    if( evals.rows == 1 )
        evals.colRange(0, ecount0).convertTo(temp, evals0.type());
    else
        evals.rowRange(0, ecount0).convertTo(temp, evals0.type());
    if( temp.data != evals0.data )
        transpose(temp, evals0);
    evects.rowRange(0, ecount0).convertTo( evects0, evects0.type() );

    // otherwise some datatype's or size's were incorrect, so the output arrays have been reallocated
    CV_Assert( mean0.data == mean.data );
}


CV_IMPL void
cvProjectPCA( const CvArr* data_arr, const CvArr* avg_arr,
              const CvArr* eigenvects, CvArr* result_arr )
{
    cv::Mat data = cv::cvarrToMat(data_arr), mean = cv::cvarrToMat(avg_arr);
    cv::Mat evects = cv::cvarrToMat(eigenvects), dst0 = cv::cvarrToMat(result_arr), dst = dst0;

    cv::PCA pca;
    pca.mean = mean;
    int n;
    if( mean.rows == 1 )
    {
        CV_Assert(dst.cols <= evects.rows && dst.rows == data.rows);
        n = dst.cols;
    }
    else
    {
        CV_Assert(dst.rows <= evects.rows && dst.cols == data.cols);
        n = dst.rows;
    }
    pca.eigenvectors = evects.rowRange(0, n);

    cv::Mat result = pca.project(data);
    if( result.cols != dst.cols )
        result = result.reshape(1, 1);
    result.convertTo(dst, dst.type());

    CV_Assert(dst0.data == dst.data);
}


CV_IMPL void
cvBackProjectPCA( const CvArr* proj_arr, const CvArr* avg_arr,
                  const CvArr* eigenvects, CvArr* result_arr )
{
    cv::Mat data = cv::cvarrToMat(proj_arr), mean = cv::cvarrToMat(avg_arr);
    cv::Mat evects = cv::cvarrToMat(eigenvects), dst0 = cv::cvarrToMat(result_arr), dst = dst0;

    cv::PCA pca;
    pca.mean = mean;
    int n;
    if( mean.rows == 1 )
    {
        CV_Assert(data.cols <= evects.rows && dst.rows == data.rows);
        n = data.cols;
    }
    else
    {
        CV_Assert(data.rows <= evects.rows && dst.cols == data.cols);
        n = data.rows;
    }
    pca.eigenvectors = evects.rowRange(0, n);

    cv::Mat result = pca.backProject(data);
    result.convertTo(dst, dst.type());

    CV_Assert(dst0.data == dst.data);
}

/* End of file. */
