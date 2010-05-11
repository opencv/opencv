/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                        Intel License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
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
//   * The name of Intel Corporation may not be used to endorse or promote products
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

#ifndef _CV_MATRIX_H_
#define _CV_MATRIX_H_

#define icvCopyVector( src, dst, len ) memcpy( (dst), (src), (len)*sizeof((dst)[0]))
#define icvSetZero( dst, len ) memset( (dst), 0, (len)*sizeof((dst)[0]))

#define icvCopyVector_32f( src, len, dst ) memcpy((dst),(src),(len)*sizeof(float))
#define icvSetZero_32f( dst, cols, rows ) memset((dst),0,(rows)*(cols)*sizeof(float))
#define icvCopyVector_64d( src, len, dst ) memcpy((dst),(src),(len)*sizeof(double))
#define icvSetZero_64d( dst, cols, rows ) memset((dst),0,(rows)*(cols)*sizeof(double))
#define icvCopyMatrix_32f( src, w, h, dst ) memcpy((dst),(src),(w)*(h)*sizeof(float))
#define icvCopyMatrix_64d( src, w, h, dst ) memcpy((dst),(src),(w)*(h)*sizeof(double))

#define icvCreateVector_32f( len )  (float*)cvAlloc( (len)*sizeof(float))
#define icvCreateVector_64d( len )  (double*)cvAlloc( (len)*sizeof(double))
#define icvCreateMatrix_32f( w, h )  (float*)cvAlloc( (w)*(h)*sizeof(float))
#define icvCreateMatrix_64d( w, h )  (double*)cvAlloc( (w)*(h)*sizeof(double))

#define icvDeleteVector( vec )  cvFree( &(vec) )
#define icvDeleteMatrix icvDeleteVector

#define icvAddMatrix_32f( src1, src2, dst, w, h ) \
    icvAddVector_32f( (src1), (src2), (dst), (w)*(h))

#define icvSubMatrix_32f( src1, src2, dst, w, h ) \
    icvSubVector_32f( (src1), (src2), (dst), (w)*(h))

#define icvNormVector_32f( src, len )  \
    sqrt(icvDotProduct_32f( src, src, len ))

#define icvNormVector_64d( src, len )  \
    sqrt(icvDotProduct_64d( src, src, len ))


#define icvDeleteMatrix icvDeleteVector

#define icvCheckVector_64f( ptr, len )
#define icvCheckVector_32f( ptr, len )

CV_INLINE double icvSum_32f( const float* src, int len )
{
    double s = 0;
    for( int i = 0; i < len; i++ ) s += src[i];

    icvCheckVector_64f( &s, 1 );

    return s;
}

CV_INLINE double icvDotProduct_32f( const float* src1, const float* src2, int len )
{
    double s = 0;
    for( int i = 0; i < len; i++ ) s += src1[i]*src2[i];

    icvCheckVector_64f( &s, 1 );

    return s;
}


CV_INLINE double icvDotProduct_64f( const double* src1, const double* src2, int len )
{
    double s = 0;
    for( int i = 0; i < len; i++ ) s += src1[i]*src2[i];

    icvCheckVector_64f( &s, 1 );

    return s;
}


CV_INLINE void icvMulVectors_32f( const float* src1, const float* src2,
                                  float* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src1[i] * src2[i];

    icvCheckVector_32f( dst, len );
}

CV_INLINE void icvMulVectors_64d( const double* src1, const double* src2,
                                  double* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src1[i] * src2[i];

    icvCheckVector_64f( dst, len );
}


CV_INLINE void icvAddVector_32f( const float* src1, const float* src2,
                                  float* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src1[i] + src2[i];

    icvCheckVector_32f( dst, len );
}

CV_INLINE void icvAddVector_64d( const double* src1, const double* src2,
                                  double* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src1[i] + src2[i];

    icvCheckVector_64f( dst, len );
}


CV_INLINE void icvSubVector_32f( const float* src1, const float* src2,
                                  float* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src1[i] - src2[i];

    icvCheckVector_32f( dst, len );
}

CV_INLINE void icvSubVector_64d( const double* src1, const double* src2,
                                  double* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src1[i] - src2[i];

    icvCheckVector_64f( dst, len );
}


#define icvAddMatrix_64d( src1, src2, dst, w, h ) \
    icvAddVector_64d( (src1), (src2), (dst), (w)*(h))

#define icvSubMatrix_64d( src1, src2, dst, w, h ) \
    icvSubVector_64d( (src1), (src2), (dst), (w)*(h))


CV_INLINE void icvSetIdentity_32f( float* dst, int w, int h )
{
    int i, len = MIN( w, h );
    icvSetZero_32f( dst, w, h );
    for( i = 0; len--; i += w+1 )
        dst[i] = 1.f;
}


CV_INLINE void icvSetIdentity_64d( double* dst, int w, int h )
{
    int i, len = MIN( w, h );
    icvSetZero_64d( dst, w, h );
    for( i = 0; len--; i += w+1 )
        dst[i] = 1.;
}


CV_INLINE void icvTrace_32f( const float* src, int w, int h, float* trace )
{
    int i, len = MIN( w, h );
    double sum = 0;
    for( i = 0; len--; i += w+1 )
        sum += src[i];
    *trace = (float)sum;

    icvCheckVector_64f( &sum, 1 );
}


CV_INLINE void icvTrace_64d( const double* src, int w, int h, double* trace )
{
    int i, len = MIN( w, h );
    double sum = 0;
    for( i = 0; len--; i += w+1 )
        sum += src[i];
    *trace = sum;

    icvCheckVector_64f( &sum, 1 );
}


CV_INLINE void icvScaleVector_32f( const float* src, float* dst,
                                   int len, double scale )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = (float)(src[i]*scale);

    icvCheckVector_32f( dst, len );
}


CV_INLINE void icvScaleVector_64d( const double* src, double* dst,
                                   int len, double scale )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src[i]*scale;

    icvCheckVector_64f( dst, len );
}


CV_INLINE void icvTransposeMatrix_32f( const float* src, int w, int h, float* dst )
{
    int i, j;

    for( i = 0; i < w; i++ )
        for( j = 0; j < h; j++ )
            *dst++ = src[j*w + i];
        
    icvCheckVector_32f( dst, w*h );
}

CV_INLINE void icvTransposeMatrix_64d( const double* src, int w, int h, double* dst )
{
    int i, j;

    for( i = 0; i < w; i++ )
        for( j = 0; j < h; j++ )
            *dst++ = src[j*w + i];

    icvCheckVector_64f( dst, w*h );
}

CV_INLINE void icvDetMatrix3x3_64d( const double* mat, double* det )
{
    #define m(y,x) mat[(y)*3 + (x)]
    
    *det = m(0,0)*(m(1,1)*m(2,2) - m(1,2)*m(2,1)) -
           m(0,1)*(m(1,0)*m(2,2) - m(1,2)*m(2,0)) +
           m(0,2)*(m(1,0)*m(2,1) - m(1,1)*m(2,0));

    #undef m

    icvCheckVector_64f( det, 1 );
}


CV_INLINE void icvMulMatrix_32f( const float* src1, int w1, int h1,
                                 const float* src2, int w2, int h2,
                                 float* dst )
{
    int i, j, k;

    if( w1 != h2 )
    {
        assert(0);
        return;
    }

    for( i = 0; i < h1; i++, src1 += w1, dst += w2 )
        for( j = 0; j < w2; j++ )
        {
            double s = 0;
            for( k = 0; k < w1; k++ )
                s += src1[k]*src2[j + k*w2];
            dst[j] = (float)s;
        }

    icvCheckVector_32f( dst, h1*w2 );
}


CV_INLINE void icvMulMatrix_64d( const double* src1, int w1, int h1,
                                 const double* src2, int w2, int h2,
                                 double* dst )
{
    int i, j, k;

    if( w1 != h2 )
    {
        assert(0);
        return;
    }

    for( i = 0; i < h1; i++, src1 += w1, dst += w2 )
        for( j = 0; j < w2; j++ )
        {
            double s = 0;
            for( k = 0; k < w1; k++ )
                s += src1[k]*src2[j + k*w2];
            dst[j] = s;
        }

    icvCheckVector_64f( dst, h1*w2 );
}


#define icvTransformVector_32f( matr, src, dst, w, h ) \
    icvMulMatrix_32f( matr, w, h, src, 1, w, dst )

#define icvTransformVector_64d( matr, src, dst, w, h ) \
    icvMulMatrix_64d( matr, w, h, src, 1, w, dst )


#define icvScaleMatrix_32f( src, dst, w, h, scale ) \
    icvScaleVector_32f( (src), (dst), (w)*(h), (scale) )

#define icvScaleMatrix_64d( src, dst, w, h, scale ) \
    icvScaleVector_64d( (src), (dst), (w)*(h), (scale) )

#define icvDotProduct_64d icvDotProduct_64f


CV_INLINE void icvInvertMatrix_64d( double* A, int n, double* invA )
{
    CvMat Am = cvMat( n, n, CV_64F, A );
    CvMat invAm = cvMat( n, n, CV_64F, invA );

    cvInvert( &Am, &invAm, CV_SVD );
}

CV_INLINE void icvMulTransMatrixR_64d( double* src, int width, int height, double* dst )
{
    CvMat srcMat = cvMat( height, width, CV_64F, src );
    CvMat dstMat = cvMat( width, width, CV_64F, dst );

    cvMulTransposed( &srcMat, &dstMat, 1 );
}

CV_INLINE void icvMulTransMatrixL_64d( double* src, int width, int height, double* dst )
{
    CvMat srcMat = cvMat( height, width, CV_64F, src );
    CvMat dstMat = cvMat( height, height, CV_64F, dst );

    cvMulTransposed( &srcMat, &dstMat, 0 );
}

CV_INLINE void icvMulTransMatrixR_32f( float* src, int width, int height, float* dst )
{
    CvMat srcMat = cvMat( height, width, CV_32F, src );
    CvMat dstMat = cvMat( width, width, CV_32F, dst );

    cvMulTransposed( &srcMat, &dstMat, 1 );
}

CV_INLINE void icvMulTransMatrixL_32f( float* src, int width, int height, float* dst )
{
    CvMat srcMat = cvMat( height, width, CV_32F, src );
    CvMat dstMat = cvMat( height, height, CV_32F, dst );

    cvMulTransposed( &srcMat, &dstMat, 0 );
}

CV_INLINE void icvCvt_32f_64d( const float* src, double* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = src[i];
}

CV_INLINE void icvCvt_64d_32f( const double* src, float* dst, int len )
{
    int i;
    for( i = 0; i < len; i++ )
        dst[i] = (float)src[i];
}

#endif/*_CV_MATRIX_H_*/

/* End of file. */
