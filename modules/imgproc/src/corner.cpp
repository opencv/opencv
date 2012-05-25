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
#include <stdio.h>


namespace cv
{

static void
calcMinEigenVal( const Mat& _cov, Mat& _dst )
{
    int i, j;
    Size size = _cov.size();
#if CV_SSE
    volatile bool simd = checkHardwareSupport(CV_CPU_SSE);
#endif
    
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const float* cov = (const float*)(_cov.data + _cov.step*i);
        float* dst = (float*)(_dst.data + _dst.step*i);
        j = 0;
    #if CV_SSE
        if( simd )
        {
            __m128 half = _mm_set1_ps(0.5f);
            for( ; j <= size.width - 5; j += 4 )
            {
                __m128 t0 = _mm_loadu_ps(cov + j*3); // a0 b0 c0 x
                __m128 t1 = _mm_loadu_ps(cov + j*3 + 3); // a1 b1 c1 x
                __m128 t2 = _mm_loadu_ps(cov + j*3 + 6); // a2 b2 c2 x
                __m128 t3 = _mm_loadu_ps(cov + j*3 + 9); // a3 b3 c3 x
                __m128 a, b, c, t;
                t = _mm_unpacklo_ps(t0, t1); // a0 a1 b0 b1
                c = _mm_unpackhi_ps(t0, t1); // c0 c1 x x
                b = _mm_unpacklo_ps(t2, t3); // a2 a3 b2 b3
                c = _mm_movelh_ps(c, _mm_unpackhi_ps(t2, t3)); // c0 c1 c2 c3
                a = _mm_movelh_ps(t, b);
                b = _mm_movehl_ps(b, t);
                a = _mm_mul_ps(a, half);
                c = _mm_mul_ps(c, half);
                t = _mm_sub_ps(a, c);
                t = _mm_add_ps(_mm_mul_ps(t, t), _mm_mul_ps(b,b));
                a = _mm_sub_ps(_mm_add_ps(a, c), _mm_sqrt_ps(t));
                _mm_storeu_ps(dst + j, a);
            }
        }
    #endif
        for( ; j < size.width; j++ )
        {
            float a = cov[j*3]*0.5f;
            float b = cov[j*3+1];
            float c = cov[j*3+2]*0.5f;
            dst[j] = (float)((a + c) - std::sqrt((a - c)*(a - c) + b*b));
        }
    }
}


static void
calcHarris( const Mat& _cov, Mat& _dst, double k )
{
    int i, j;
    Size size = _cov.size();
#if CV_SSE
    volatile bool simd = checkHardwareSupport(CV_CPU_SSE);
#endif
    
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( i = 0; i < size.height; i++ )
    {
        const float* cov = (const float*)(_cov.data + _cov.step*i);
        float* dst = (float*)(_dst.data + _dst.step*i);
        j = 0;

    #if CV_SSE
        if( simd )
        {
            __m128 k4 = _mm_set1_ps((float)k);
            for( ; j <= size.width - 5; j += 4 )
            {
                __m128 t0 = _mm_loadu_ps(cov + j*3); // a0 b0 c0 x
                __m128 t1 = _mm_loadu_ps(cov + j*3 + 3); // a1 b1 c1 x
                __m128 t2 = _mm_loadu_ps(cov + j*3 + 6); // a2 b2 c2 x
                __m128 t3 = _mm_loadu_ps(cov + j*3 + 9); // a3 b3 c3 x
                __m128 a, b, c, t;
                t = _mm_unpacklo_ps(t0, t1); // a0 a1 b0 b1
                c = _mm_unpackhi_ps(t0, t1); // c0 c1 x x
                b = _mm_unpacklo_ps(t2, t3); // a2 a3 b2 b3
                c = _mm_movelh_ps(c, _mm_unpackhi_ps(t2, t3)); // c0 c1 c2 c3
                a = _mm_movelh_ps(t, b);
                b = _mm_movehl_ps(b, t);
                t = _mm_add_ps(a, c);
                a = _mm_sub_ps(_mm_mul_ps(a, c), _mm_mul_ps(b, b));
                t = _mm_mul_ps(_mm_mul_ps(k4, t), t);
                a = _mm_sub_ps(a, t);
                _mm_storeu_ps(dst + j, a);
            }
        }
    #endif

        for( ; j < size.width; j++ )
        {
            float a = cov[j*3];
            float b = cov[j*3+1];
            float c = cov[j*3+2];
            dst[j] = (float)(a*c - b*b - k*(a + c)*(a + c));
        }
    }
}

    
void eigen2x2( const float* cov, float* dst, int n )
{
    for( int j = 0; j < n; j++ )
    {
        double a = cov[j*3];
        double b = cov[j*3+1];
        double c = cov[j*3+2];
        
        double u = (a + c)*0.5;
        double v = std::sqrt((a - c)*(a - c)*0.25 + b*b);
        double l1 = u + v;
        double l2 = u - v;
        
        double x = b;
        double y = l1 - a;
        double e = fabs(x);
        
        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l1 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }
        
        double d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j] = (float)l1;
        dst[6*j + 2] = (float)(x*d);
        dst[6*j + 3] = (float)(y*d);
        
        x = b;
        y = l2 - a;
        e = fabs(x);
        
        if( e + fabs(y) < 1e-4 )
        {
            y = b;
            x = l2 - c;
            e = fabs(x);
            if( e + fabs(y) < 1e-4 )
            {
                e = 1./(e + fabs(y) + FLT_EPSILON);
                x *= e, y *= e;
            }
        }
        
        d = 1./std::sqrt(x*x + y*y + DBL_EPSILON);
        dst[6*j + 1] = (float)l2;
        dst[6*j + 4] = (float)(x*d);
        dst[6*j + 5] = (float)(y*d);
    }
}

static void
calcEigenValsVecs( const Mat& _cov, Mat& _dst )
{
    Size size = _cov.size();
    if( _cov.isContinuous() && _dst.isContinuous() )
    {
        size.width *= size.height;
        size.height = 1;
    }

    for( int i = 0; i < size.height; i++ )
    {
        const float* cov = (const float*)(_cov.data + _cov.step*i);
        float* dst = (float*)(_dst.data + _dst.step*i);

        eigen2x2(cov, dst, size.width);
    }
}


enum { MINEIGENVAL=0, HARRIS=1, EIGENVALSVECS=2 };


static void
cornerEigenValsVecs( const Mat& src, Mat& eigenv, int block_size,
                     int aperture_size, int op_type, double k=0.,
                     int borderType=BORDER_DEFAULT )
{
#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::cornerEigenValsVecs(src, eigenv, block_size, aperture_size, op_type, k, borderType))
        return;
#endif 

    int depth = src.depth();
    double scale = (double)(1 << ((aperture_size > 0 ? aperture_size : 3) - 1)) * block_size;
    if( aperture_size < 0 )
        scale *= 2.;
    if( depth == CV_8U )
        scale *= 255.;
    scale = 1./scale;

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );

    Mat Dx, Dy;
    if( aperture_size > 0 )
    {
        Sobel( src, Dx, CV_32F, 1, 0, aperture_size, scale, 0, borderType );
        Sobel( src, Dy, CV_32F, 0, 1, aperture_size, scale, 0, borderType );
    }
    else
    {
        Scharr( src, Dx, CV_32F, 1, 0, scale, 0, borderType );
        Scharr( src, Dy, CV_32F, 0, 1, scale, 0, borderType );
    }

    Size size = src.size();
    Mat cov( size, CV_32FC3 );
    int i, j;

    for( i = 0; i < size.height; i++ )
    {
        float* cov_data = (float*)(cov.data + i*cov.step);
        const float* dxdata = (const float*)(Dx.data + i*Dx.step);
        const float* dydata = (const float*)(Dy.data + i*Dy.step);

        for( j = 0; j < size.width; j++ )
        {
            float dx = dxdata[j];
            float dy = dydata[j];

            cov_data[j*3] = dx*dx;
            cov_data[j*3+1] = dx*dy;
            cov_data[j*3+2] = dy*dy;
        }
    }

    boxFilter(cov, cov, cov.depth(), Size(block_size, block_size),
        Point(-1,-1), false, borderType );

    if( op_type == MINEIGENVAL )
        calcMinEigenVal( cov, eigenv );
    else if( op_type == HARRIS )
        calcHarris( cov, eigenv, k );
    else if( op_type == EIGENVALSVECS )
        calcEigenValsVecs( cov, eigenv );
}

}

void cv::cornerMinEigenVal( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, MINEIGENVAL, 0, borderType );
}


void cv::cornerHarris( InputArray _src, OutputArray _dst, int blockSize, int ksize, double k, int borderType )
{
    Mat src = _src.getMat();
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, HARRIS, k, borderType );
}


void cv::cornerEigenValsAndVecs( InputArray _src, OutputArray _dst, int blockSize, int ksize, int borderType )
{
    Mat src = _src.getMat();
    Size dsz = _dst.size();
    int dtype = _dst.type();
    
    if( dsz.height != src.rows || dsz.width*CV_MAT_CN(dtype) != src.cols*6 || CV_MAT_DEPTH(dtype) != CV_32F )
        _dst.create( src.size(), CV_32FC(6) );
    Mat dst = _dst.getMat();
    cornerEigenValsVecs( src, dst, blockSize, ksize, EIGENVALSVECS, 0, borderType );
}


void cv::preCornerDetect( InputArray _src, OutputArray _dst, int ksize, int borderType )
{
    Mat Dx, Dy, D2x, D2y, Dxy, src = _src.getMat();

    CV_Assert( src.type() == CV_8UC1 || src.type() == CV_32FC1 );
    _dst.create( src.size(), CV_32F );
    Mat dst = _dst.getMat();
    
    Sobel( src, Dx, CV_32F, 1, 0, ksize, 1, 0, borderType );
    Sobel( src, Dy, CV_32F, 0, 1, ksize, 1, 0, borderType );
    Sobel( src, D2x, CV_32F, 2, 0, ksize, 1, 0, borderType );
    Sobel( src, D2y, CV_32F, 0, 2, ksize, 1, 0, borderType );
    Sobel( src, Dxy, CV_32F, 1, 1, ksize, 1, 0, borderType );

    double factor = 1 << (ksize - 1);
    if( src.depth() == CV_8U )
        factor *= 255;
    factor = 1./(factor * factor * factor);

    Size size = src.size();
    int i, j;
    for( i = 0; i < size.height; i++ )
    {
        float* dstdata = (float*)(dst.data + i*dst.step);
        const float* dxdata = (const float*)(Dx.data + i*Dx.step);
        const float* dydata = (const float*)(Dy.data + i*Dy.step);
        const float* d2xdata = (const float*)(D2x.data + i*D2x.step);
        const float* d2ydata = (const float*)(D2y.data + i*D2y.step);
        const float* dxydata = (const float*)(Dxy.data + i*Dxy.step);
        
        for( j = 0; j < size.width; j++ )
        {
            float dx = dxdata[j];
            float dy = dydata[j];
            dstdata[j] = (float)(factor*(dx*dx*d2ydata[j] + dy*dy*d2xdata[j] - 2*dx*dy*dxydata[j]));
        }
    }
}

CV_IMPL void
cvCornerMinEigenVal( const CvArr* srcarr, CvArr* dstarr,
                     int block_size, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::cornerMinEigenVal( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}

CV_IMPL void
cvCornerHarris( const CvArr* srcarr, CvArr* dstarr,
                int block_size, int aperture_size, double k )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::cornerHarris( src, dst, block_size, aperture_size, k, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvCornerEigenValsAndVecs( const void* srcarr, void* dstarr,
                          int block_size, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.rows == dst.rows && src.cols*6 == dst.cols*dst.channels() && dst.depth() == CV_32F );
    cv::cornerEigenValsAndVecs( src, dst, block_size, aperture_size, cv::BORDER_REPLICATE );
}


CV_IMPL void
cvPreCornerDetect( const void* srcarr, void* dstarr, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);

    CV_Assert( src.size() == dst.size() && dst.type() == CV_32FC1 );
    cv::preCornerDetect( src, dst, aperture_size, cv::BORDER_REPLICATE );
}

/* End of file */
