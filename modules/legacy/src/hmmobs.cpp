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
#include "precomp.hpp"

//*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: icvImgToObs_DCT_8u32f_C1R
//    Purpose: The function takes as input an image and returns the sequnce of observations
//             to be used with an embedded HMM; Each observation is top-left block of DCT
//             coefficient matrix.
//    Context:
//    Parameters: img     - pointer to the original image ROI
//                imgStep - full row width of the image in bytes
//                roi     - width and height of ROI in pixels
//                obs     - pointer to resultant observation vectors
//                dctSize - size of the block for which DCT is calculated
//                obsSize - size of top-left block of DCT coeffs matrix, which is treated
//                          as observation. Each observation vector consists of
//                          obsSize.width * obsSize.height floats.
//                          The following conditions should be satisfied:
//                          0 < objSize.width <= dctSize.width,
//                          0 < objSize.height <= dctSize.height.
//                delta   - dctBlocks are overlapped and this parameter specifies horizontal
//                          and vertical shift.
//    Returns:
//      CV_NO_ERR or error code
//    Notes:
//      The algorithm is following:
//          1. First, number of observation vectors per row and per column are calculated:
//
//             Nx = floor((roi.width - dctSize.width + delta.width)/delta.width);
//             Ny = floor((roi.height - dctSize.height + delta.height)/delta.height);
//
//             So, total number of observation vectors is Nx*Ny, and total size of
//             array obs must be >= Nx*Ny*obsSize.width*obsSize.height*sizeof(float).
//          2. Observation vectors are calculated in the following loop
//               ( actual implementation may be different ), where 
//               I[x1:x2,y1:y2] means block of pixels from source image with
//               x1 <= x < x2, y1 <= y < y2,
//               D[x1:x2,y1:y2] means sub matrix of DCT matrix D.
//               O[x,y] means observation vector that corresponds to position
//               (x*delta.width,y*delta.height) in the source image
//               ( all indices are counted from 0 ).
//
//               for( y = 0; y < Ny; y++ )
//               {
//                   for( x = 0; x < Nx; x++ )
//                   {
//                       D = DCT(I[x*delta.width : x*delta.width + dctSize.width,
//                                  y*delta.height : y*delta.height + dctSize.height]);
//                       O[x,y] = D[0:obsSize.width, 0:obsSize.height];
//                   }
//               }
//F*/

/*comment out the following line to make DCT be calculated in floating-point arithmetics*/
//#define _CV_INT_DCT

/* for integer DCT only */
#define DCT_SCALE  15

#ifdef _CV_INT_DCT
typedef int work_t;

#define  DESCALE      CV_DESCALE
#define  SCALE(x)     CV_FLT_TO_FIX((x),DCT_SCALE)
#else
typedef float work_t;

#define  DESCALE(x,n) (float)(x)
#define  SCALE(x)     (float)(x)
#endif

/* calculate dct transform matrix */
static void icvCalcDCTMatrix( work_t * cfs, int n );

#define  MAX_DCT_SIZE  32

static CvStatus CV_STDCALL
icvImgToObs_DCT_8u32f_C1R( uchar * img, int imgStep, CvSize roi,
                           float *obs, CvSize dctSize,
                           CvSize obsSize, CvSize delta )
{
    /* dct transform matrices: horizontal and vertical */
    work_t tab_x[MAX_DCT_SIZE * MAX_DCT_SIZE / 2 + 2];
    work_t tab_y[MAX_DCT_SIZE * MAX_DCT_SIZE / 2 + 2];

    /* temporary buffers for dct */
    work_t temp0[MAX_DCT_SIZE * 4];
    work_t temp1[MAX_DCT_SIZE * 4];
    work_t *buffer = 0;
    work_t *buf_limit;

    double s;

    int y;
    int Nx, Ny;

    int n1 = dctSize.height, m1 = n1 / 2;
    int n2 = dctSize.width, m2 = n2 / 2;

    if( !img || !obs )
        return CV_NULLPTR_ERR;

    if( roi.width <= 0 || roi.height <= 0 )
        return CV_BADSIZE_ERR;

    if( delta.width <= 0 || delta.height <= 0 )
        return CV_BADRANGE_ERR;

    if( obsSize.width <= 0 || dctSize.width < obsSize.width ||
        obsSize.height <= 0 || dctSize.height < obsSize.height )
        return CV_BADRANGE_ERR;

    if( dctSize.width > MAX_DCT_SIZE || dctSize.height > MAX_DCT_SIZE )
        return CV_BADRANGE_ERR;

    Nx = (roi.width - dctSize.width + delta.width) / delta.width;
    Ny = (roi.height - dctSize.height + delta.height) / delta.height;

    if( Nx <= 0 || Ny <= 0 )
        return CV_BADRANGE_ERR;

    buffer = (work_t *)cvAlloc( roi.width * obsSize.height * sizeof( buffer[0] ));
    if( !buffer )
        return CV_OUTOFMEM_ERR;

    icvCalcDCTMatrix( tab_x, dctSize.width );
    icvCalcDCTMatrix( tab_y, dctSize.height );

    buf_limit = buffer + obsSize.height * roi.width;

    for( y = 0; y < Ny; y++, img += delta.height * imgStep )
    {
        int x, i, j, k;
        work_t k0 = 0;

        /* do transfroms for each column. Calc only first obsSize.height DCT coefficients */
        for( x = 0; x < roi.width; x++ )
        {
            float is = 0;
            work_t *buf = buffer + x;
            work_t *tab = tab_y + 2;

            if( n1 & 1 )
            {
                is = img[x + m1 * imgStep];
                k0 = ((work_t) is) * tab[-1];
            }

            /* first coefficient */
            for( j = 0; j < m1; j++ )
            {
                float t0 = img[x + j * imgStep];
                float t1 = img[x + (n1 - 1 - j) * imgStep];
                float t2 = t0 + t1;

                t0 -= t1;
                temp0[j] = (work_t) t2;
                is += t2;
                temp1[j] = (work_t) t0;
            }

            buf[0] = DESCALE( is * tab[-2], PASS1_SHIFT );
            if( (buf += roi.width) >= buf_limit )
                continue;

            /* other coefficients */
            for( ;; )
            {
                s = 0;

                for( k = 0; k < m1; k++ )
                    s += temp1[k] * tab[k];

                buf[0] = DESCALE( s, PASS1_SHIFT );
                if( (buf += roi.width) >= buf_limit )
                    break;

                tab += m1;
                s = 0;

                if( n1 & 1 )
                {
                    k0 = -k0;
                    s = k0;
                }
                for( k = 0; k < m1; k++ )
                    s += temp0[k] * tab[k];

                buf[0] = DESCALE( s, PASS1_SHIFT );
                tab += m1;

                if( (buf += roi.width) >= buf_limit )
                    break;
            }
        }

        k0 = 0;

        /* do transforms for rows. */
        for( x = 0; x + dctSize.width <= roi.width; x += delta.width )
        {
            for( i = 0; i < obsSize.height; i++ )
            {
                work_t *buf = buffer + x + roi.width * i;
                work_t *tab = tab_x + 2;
                float *obs_limit = obs + obsSize.width;

                s = 0;

                if( n2 & 1 )
                {
                    s = buf[m2];
                    k0 = (work_t) (s * tab[-1]);
                }

                /* first coefficient */
                for( j = 0; j < m2; j++ )
                {
                    work_t t0 = buf[j];
                    work_t t1 = buf[n2 - 1 - j];
                    work_t t2 = t0 + t1;

                    t0 -= t1;
                    temp0[j] = (work_t) t2;
                    s += t2;
                    temp1[j] = (work_t) t0;
                }

                *obs++ = (float) DESCALE( s * tab[-2], PASS2_SHIFT );

                if( obs == obs_limit )
                    continue;

                /* other coefficients */
                for( ;; )
                {
                    s = 0;

                    for( k = 0; k < m2; k++ )
                        s += temp1[k] * tab[k];

                    obs[0] = (float) DESCALE( s, PASS2_SHIFT );
                    if( ++obs == obs_limit )
                        break;

                    tab += m2;

                    s = 0;

                    if( n2 & 1 )
                    {
                        k0 = -k0;
                        s = k0;
                    }
                    for( k = 0; k < m2; k++ )
                        s += temp0[k] * tab[k];
                    obs[0] = (float) DESCALE( s, PASS2_SHIFT );

                    tab += m2;
                    if( ++obs == obs_limit )
                        break;
                }
            }
        }
    }

    cvFree( &buffer );
    return CV_NO_ERR;
}


static CvStatus CV_STDCALL
icvImgToObs_DCT_32f_C1R( float * img, int imgStep, CvSize roi,
                         float *obs, CvSize dctSize,
                         CvSize obsSize, CvSize delta )
{
    /* dct transform matrices: horizontal and vertical */
    work_t tab_x[MAX_DCT_SIZE * MAX_DCT_SIZE / 2 + 2];
    work_t tab_y[MAX_DCT_SIZE * MAX_DCT_SIZE / 2 + 2];

    /* temporary buffers for dct */
    work_t temp0[MAX_DCT_SIZE * 4];
    work_t temp1[MAX_DCT_SIZE * 4];
    work_t *buffer = 0;
    work_t *buf_limit;

    double s;

    int y;
    int Nx, Ny;

    int n1 = dctSize.height, m1 = n1 / 2;
    int n2 = dctSize.width, m2 = n2 / 2;

    if( !img || !obs )
        return CV_NULLPTR_ERR;

    if( roi.width <= 0 || roi.height <= 0 )
        return CV_BADSIZE_ERR;

    if( delta.width <= 0 || delta.height <= 0 )
        return CV_BADRANGE_ERR;

    if( obsSize.width <= 0 || dctSize.width < obsSize.width ||
        obsSize.height <= 0 || dctSize.height < obsSize.height )
        return CV_BADRANGE_ERR;

    if( dctSize.width > MAX_DCT_SIZE || dctSize.height > MAX_DCT_SIZE )
        return CV_BADRANGE_ERR;

    Nx = (roi.width - dctSize.width + delta.width) / delta.width;
    Ny = (roi.height - dctSize.height + delta.height) / delta.height;

    if( Nx <= 0 || Ny <= 0 )
        return CV_BADRANGE_ERR;

    buffer = (work_t *)cvAlloc( roi.width * obsSize.height * sizeof( buffer[0] ));
    if( !buffer )
        return CV_OUTOFMEM_ERR;

    icvCalcDCTMatrix( tab_x, dctSize.width );
    icvCalcDCTMatrix( tab_y, dctSize.height );

    buf_limit = buffer + obsSize.height * roi.width;

    imgStep /= sizeof(img[0]);

    for( y = 0; y < Ny; y++, img += delta.height * imgStep )
    {
        int x, i, j, k;
        work_t k0 = 0;

        /* do transfroms for each column. Calc only first obsSize.height DCT coefficients */
        for( x = 0; x < roi.width; x++ )
        {
            float is = 0;
            work_t *buf = buffer + x;
            work_t *tab = tab_y + 2;

            if( n1 & 1 )
            {
                is = img[x + m1 * imgStep];
                k0 = ((work_t) is) * tab[-1];
            }

            /* first coefficient */
            for( j = 0; j < m1; j++ )
            {
                float t0 = img[x + j * imgStep];
                float t1 = img[x + (n1 - 1 - j) * imgStep];
                float t2 = t0 + t1;

                t0 -= t1;
                temp0[j] = (work_t) t2;
                is += t2;
                temp1[j] = (work_t) t0;
            }

            buf[0] = DESCALE( is * tab[-2], PASS1_SHIFT );
            if( (buf += roi.width) >= buf_limit )
                continue;

            /* other coefficients */
            for( ;; )
            {
                s = 0;

                for( k = 0; k < m1; k++ )
                    s += temp1[k] * tab[k];

                buf[0] = DESCALE( s, PASS1_SHIFT );
                if( (buf += roi.width) >= buf_limit )
                    break;

                tab += m1;
                s = 0;

                if( n1 & 1 )
                {
                    k0 = -k0;
                    s = k0;
                }
                for( k = 0; k < m1; k++ )
                    s += temp0[k] * tab[k];

                buf[0] = DESCALE( s, PASS1_SHIFT );
                tab += m1;

                if( (buf += roi.width) >= buf_limit )
                    break;
            }
        }

        k0 = 0;

        /* do transforms for rows. */
        for( x = 0; x + dctSize.width <= roi.width; x += delta.width )
        {
            for( i = 0; i < obsSize.height; i++ )
            {
                work_t *buf = buffer + x + roi.width * i;
                work_t *tab = tab_x + 2;
                float *obs_limit = obs + obsSize.width;

                s = 0;

                if( n2 & 1 )
                {
                    s = buf[m2];
                    k0 = (work_t) (s * tab[-1]);
                }

                /* first coefficient */
                for( j = 0; j < m2; j++ )
                {
                    work_t t0 = buf[j];
                    work_t t1 = buf[n2 - 1 - j];
                    work_t t2 = t0 + t1;

                    t0 -= t1;
                    temp0[j] = (work_t) t2;
                    s += t2;
                    temp1[j] = (work_t) t0;
                }

                *obs++ = (float) DESCALE( s * tab[-2], PASS2_SHIFT );

                if( obs == obs_limit )
                    continue;

                /* other coefficients */
                for( ;; )
                {
                    s = 0;

                    for( k = 0; k < m2; k++ )
                        s += temp1[k] * tab[k];

                    obs[0] = (float) DESCALE( s, PASS2_SHIFT );
                    if( ++obs == obs_limit )
                        break;

                    tab += m2;

                    s = 0;

                    if( n2 & 1 )
                    {
                        k0 = -k0;
                        s = k0;
                    }
                    for( k = 0; k < m2; k++ )
                        s += temp0[k] * tab[k];
                    obs[0] = (float) DESCALE( s, PASS2_SHIFT );

                    tab += m2;
                    if( ++obs == obs_limit )
                        break;
                }
            }
        }
    }

    cvFree( &buffer );
    return CV_NO_ERR;
}


static void
icvCalcDCTMatrix( work_t * cfs, int n )
{
    static const double sqrt2 = 1.4142135623730950488016887242097;
    static const double pi = 3.1415926535897932384626433832795;

    static const double sincos[16 * 2] = {
        1.00000000000000000, 0.00000000000000006,
        0.70710678118654746, 0.70710678118654757,
        0.49999999999999994, 0.86602540378443871,
        0.38268343236508978, 0.92387953251128674,
        0.30901699437494740, 0.95105651629515353,
        0.25881904510252074, 0.96592582628906831,
        0.22252093395631439, 0.97492791218182362,
        0.19509032201612825, 0.98078528040323043,
        0.17364817766693033, 0.98480775301220802,
        0.15643446504023087, 0.98768834059513777,
        0.14231483827328514, 0.98982144188093268,
        0.13052619222005157, 0.99144486137381038,
        0.12053668025532305, 0.99270887409805397,
        0.11196447610330786, 0.99371220989324260,
        0.10452846326765346, 0.99452189536827329,
        0.09801714032956060, 0.99518472667219693,
    };

#define ROTATE( c, s, dc, ds ) \
    {                              \
        t = c*dc - s*ds;           \
        s = c*ds + s*dc;           \
        c = t;                     \
    }

#define WRITE2( j, a, b ) \
    {                         \
        cfs[j]   = SCALE(a);  \
        cfs2[j]  = SCALE(b);  \
    }

    double t, scale = 1. / sqrt( (double)n );
    int i, j, m = n / 2;

    cfs[0] = SCALE( scale );
    scale *= sqrt2;
    cfs[1] = SCALE( scale );
    cfs += 2 - m;

    if( n > 1 )
    {
        double a0, b0;
        double da0, db0;
        work_t *cfs2 = cfs + m * n;

        if( n <= 16 )
        {
            da0 = a0 = sincos[2 * n - 1];
            db0 = b0 = sincos[2 * n - 2];
        }
        else
        {
            t = pi / (2 * n);
            da0 = a0 = cos( t );
            db0 = b0 = sin( t );
        }

        /* other rows */
        for( i = 1; i <= m; i++ )
        {
            double a = a0 * scale;
            double b = b0 * scale;
            double da = a0 * a0 - b0 * b0;
            double db = a0 * b0 + a0 * b0;

            cfs += m;
            cfs2 -= m;

            for( j = 0; j < m; j += 2 )
            {
                WRITE2( j, a, b );
                ROTATE( a, b, da, db );
                if( j + 1 < m )
                {
                    WRITE2( j + 1, a, -b );
                    ROTATE( a, b, da, db );
                }
            }

            ROTATE( a0, b0, da0, db0 );
        }
    }
#undef ROTATE
#undef WRITE2
}


CV_IMPL void
cvImgToObs_DCT( const void* arr, float *obs, CvSize dctSize,
                CvSize obsSize, CvSize delta )
{
    CV_FUNCNAME( "cvImgToObs_DCT" );

    __BEGIN__;

    CvMat stub, *mat = (CvMat*)arr;

    CV_CALL( mat = cvGetMat( arr, &stub ));

    switch( CV_MAT_TYPE( mat->type ))
    {
    case CV_8UC1:
        IPPI_CALL( icvImgToObs_DCT_8u32f_C1R( mat->data.ptr, mat->step,
                                           cvGetMatSize(mat), obs,
                                           dctSize, obsSize, delta ));
        break;
    case CV_32FC1:
        IPPI_CALL( icvImgToObs_DCT_32f_C1R( mat->data.fl, mat->step,
                                           cvGetMatSize(mat), obs,
                                           dctSize, obsSize, delta ));
        break;
    default:
        CV_ERROR( CV_StsUnsupportedFormat, "" );
    }

    __END__;
}


/* End of file. */
