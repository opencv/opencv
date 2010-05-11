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
#include "_vm.h"

//#define REAL_ZERO(x) ( (x) < 1e-8 && (x) > -1e-8)

static CvStatus
icvGetNormalVector3( CvMatrix3 * Matrix, float *v )
{
/*  return vector v that is any 3-vector perpendicular
    to all the row vectors of Matrix */

    double *solutions = 0;
    double M[3 * 3];
    double B[3] = { 0.f, 0.f, 0.f };
    int i, j, res;

    if( Matrix == 0 || v == 0 )
        return CV_NULLPTR_ERR;

    for( i = 0; i < 3; i++ )
    {
        for( j = 0; j < 3; j++ )
            M[i * 3 + j] = (double) (Matrix->m[i][j]);
    }                           /* for */

    res = icvGaussMxN( M, B, 3, 3, &solutions );

    if( res == -1 )
        return CV_BADFACTOR_ERR;

    if( res > 0 && solutions )
    {
        v[0] = (float) solutions[0];
        v[1] = (float) solutions[1];
        v[2] = (float) solutions[2];
        res = 0;
    }
    else
        res = 1;

    if( solutions )
        cvFree( &solutions );

    if( res )
        return CV_BADFACTOR_ERR;
    else
        return CV_NO_ERR;

}                               /* icvgetNormalVector3 */


/*=====================================================================================*/

static CvStatus
icvMultMatrixVector3( CvMatrix3 * m, float *src, float *dst )
{
    if( m == 0 || src == 0 || dst == 0 )
        return CV_NULLPTR_ERR;

    dst[0] = m->m[0][0] * src[0] + m->m[0][1] * src[1] + m->m[0][2] * src[2];
    dst[1] = m->m[1][0] * src[0] + m->m[1][1] * src[1] + m->m[1][2] * src[2];
    dst[2] = m->m[2][0] * src[0] + m->m[2][1] * src[1] + m->m[2][2] * src[2];

    return CV_NO_ERR;

}                               /* icvMultMatrixVector3 */


/*=====================================================================================*/

static CvStatus
icvMultMatrixTVector3( CvMatrix3 * m, float *src, float *dst )
{
    if( m == 0 || src == 0 || dst == 0 )
        return CV_NULLPTR_ERR;

    dst[0] = m->m[0][0] * src[0] + m->m[1][0] * src[1] + m->m[2][0] * src[2];
    dst[1] = m->m[0][1] * src[0] + m->m[1][1] * src[1] + m->m[2][1] * src[2];
    dst[2] = m->m[0][2] * src[0] + m->m[1][2] * src[1] + m->m[2][2] * src[2];

    return CV_NO_ERR;

}                               /* icvMultMatrixTVector3 */

/*=====================================================================================*/

static CvStatus
icvCrossLines( float *line1, float *line2, float *cross_point )
{
    float delta;

    if( line1 == 0 && line2 == 0 && cross_point == 0 )
        return CV_NULLPTR_ERR;

    delta = line1[0] * line2[1] - line1[1] * line2[0];

    if( REAL_ZERO( delta ))
        return CV_BADFACTOR_ERR;

    cross_point[0] = (-line1[2] * line2[1] + line1[1] * line2[2]) / delta;
    cross_point[1] = (-line1[0] * line2[2] + line1[2] * line2[0]) / delta;
    cross_point[2] = 1;

    return CV_NO_ERR;
}                               /* icvCrossLines */



/*======================================================================================*/

static CvStatus
icvMakeScanlines( CvMatrix3 * matrix,
                  CvSize imgSize,
                  int *scanlines_1, int *scanlines_2, int *lens_1, int *lens_2, int *numlines )
{

    CvStatus error;

    error = icvGetCoefficient( matrix, imgSize, scanlines_2, scanlines_1, numlines );

    /* Make Length of scanlines */

    if( scanlines_1 == 0 && scanlines_2 == 0 )
        return error;

    icvMakeScanlinesLengths( scanlines_1, *numlines, lens_1 );

    icvMakeScanlinesLengths( scanlines_2, *numlines, lens_2 );

    matrix = matrix;
    return CV_NO_ERR;


}                               /* icvMakeScanlines */


/*======================================================================================*/

CvStatus
icvMakeScanlinesLengths( int *scanlines, int numlines, int *lens )
{
    int index;
    int x1, y1, x2, y2, dx, dy;
    int curr;

    curr = 0;

    for( index = 0; index < numlines; index++ )
    {

        x1 = scanlines[curr++];
        y1 = scanlines[curr++];
        x2 = scanlines[curr++];
        y2 = scanlines[curr++];

        dx = abs( x1 - x2 ) + 1;
        dy = abs( y1 - y2 ) + 1;

        lens[index] = MAX( dx, dy );

    }
    return CV_NO_ERR;
}

/*======================================================================================*/

static CvStatus
icvMakeAlphaScanlines( int *scanlines_1,
                       int *scanlines_2,
                       int *scanlines_a, int *lens, int numlines, float alpha )
{
    int index;
    int x1, y1, x2, y2;
    int curr;
    int dx, dy;
    int curr_len;

    curr = 0;
    curr_len = 0;
    for( index = 0; index < numlines; index++ )
    {

        x1 = (int) (scanlines_1[curr] * alpha + scanlines_2[curr] * (1.0 - alpha));

        scanlines_a[curr++] = x1;

        y1 = (int) (scanlines_1[curr] * alpha + scanlines_2[curr] * (1.0 - alpha));

        scanlines_a[curr++] = y1;

        x2 = (int) (scanlines_1[curr] * alpha + scanlines_2[curr] * (1.0 - alpha));

        scanlines_a[curr++] = x2;

        y2 = (int) (scanlines_1[curr] * alpha + scanlines_2[curr] * (1.0 - alpha));

        scanlines_a[curr++] = y2;

        dx = abs( x1 - x2 ) + 1;
        dy = abs( y1 - y2 ) + 1;

        lens[curr_len++] = MAX( dx, dy );

    }

    return CV_NO_ERR;
}

/*======================================================================================*/







/* //////////////////////////////////////////////////////////////////////////////////// */

CvStatus
icvGetCoefficient( CvMatrix3 * matrix,
                   CvSize imgSize, int *scanlines_1, int *scanlines_2, int *numlines )
{
    float l_epipole[3];
    float r_epipole[3];
    CvMatrix3 *F;
    CvMatrix3 Ft;
    CvStatus error;
    int i, j;

    F = matrix;

    l_epipole[2] = -1;
    r_epipole[2] = -1;

    if( F == 0 )
    {
        error = icvGetCoefficientDefault( matrix,
                                          imgSize, scanlines_1, scanlines_2, numlines );
        return error;
    }


    for( i = 0; i < 3; i++ )
        for( j = 0; j < 3; j++ )
            Ft.m[i][j] = F->m[j][i];


    error = icvGetNormalVector3( &Ft, l_epipole );
    if( error == CV_NO_ERR && !REAL_ZERO( l_epipole[2] ) && !REAL_ZERO( l_epipole[2] - 1 ))
    {

        l_epipole[0] /= l_epipole[2];
        l_epipole[1] /= l_epipole[2];
        l_epipole[2] = 1;
    }                           /* if */

    error = icvGetNormalVector3( F, r_epipole );
    if( error == CV_NO_ERR && !REAL_ZERO( r_epipole[2] ) && !REAL_ZERO( r_epipole[2] - 1 ))
    {

        r_epipole[0] /= r_epipole[2];
        r_epipole[1] /= r_epipole[2];
        r_epipole[2] = 1;
    }                           /* if */

    if( REAL_ZERO( l_epipole[2] - 1 ) && REAL_ZERO( r_epipole[2] - 1 ))
    {
        error = icvGetCoefficientStereo( matrix,
                                         imgSize,
                                         l_epipole,
                                         r_epipole, scanlines_1, scanlines_2, numlines );
        if( error == CV_NO_ERR )
            return CV_NO_ERR;
    }
    else
    {
        if( REAL_ZERO( l_epipole[2] ) && REAL_ZERO( r_epipole[2] ))
        {
            error = icvGetCoefficientOrto( matrix,
                                           imgSize, scanlines_1, scanlines_2, numlines );
            if( error == CV_NO_ERR )
                return CV_NO_ERR;
        }
    }


    error = icvGetCoefficientDefault( matrix, imgSize, scanlines_1, scanlines_2, numlines );

    return error;

}                               /* icvlGetCoefficient */

/*===========================================================================*/
CvStatus
icvGetCoefficientDefault( CvMatrix3 * matrix,
                          CvSize imgSize, int *scanlines_1, int *scanlines_2, int *numlines )
{
    int curr;
    int y;

    *numlines = imgSize.height;

    if( scanlines_1 == 0 && scanlines_2 == 0 )
        return CV_NO_ERR;

    curr = 0;
    for( y = 0; y < imgSize.height; y++ )
    {
        scanlines_1[curr] = 0;
        scanlines_1[curr + 1] = y;
        scanlines_1[curr + 2] = imgSize.width - 1;
        scanlines_1[curr + 3] = y;

        scanlines_2[curr] = 0;
        scanlines_2[curr + 1] = y;
        scanlines_2[curr + 2] = imgSize.width - 1;
        scanlines_2[curr + 3] = y;

        curr += 4;
    }

    matrix = matrix;
    return CV_NO_ERR;

}                               /* icvlGetCoefficientDefault */

/*===========================================================================*/
CvStatus
icvGetCoefficientOrto( CvMatrix3 * matrix,
                       CvSize imgSize, int *scanlines_1, int *scanlines_2, int *numlines )
{
    float l_start_end[4], r_start_end[4];
    double a, b;
    CvStatus error;
    CvMatrix3 *F;

    F = matrix;

    if( F->m[0][2] * F->m[1][2] < 0 )
    {                           /* on left / */

        if( F->m[2][0] * F->m[2][1] < 0 )
        {                       /* on right / */
            error = icvGetStartEnd1( F, imgSize, l_start_end, r_start_end );


        }
        else
        {                       /* on right \ */
            error = icvGetStartEnd2( F, imgSize, l_start_end, r_start_end );
        }                       /* if */

    }
    else
    {                           /* on left \ */

        if( F->m[2][0] * F->m[2][1] < 0 )
        {                       /* on right / */
            error = icvGetStartEnd3( F, imgSize, l_start_end, r_start_end );
        }
        else
        {                       /* on right \ */
            error = icvGetStartEnd4( F, imgSize, l_start_end, r_start_end );
        }                       /* if */
    }                           /* if */

    if( error != CV_NO_ERR )
        return error;

    a = fabs( l_start_end[0] - l_start_end[2] );
    b = fabs( r_start_end[0] - r_start_end[2] );
    if( a > b )
    {

        error = icvBuildScanlineLeft( F,
                                      imgSize,
                                      scanlines_1, scanlines_2, l_start_end, numlines );

    }
    else
    {

        error = icvBuildScanlineRight( F,
                                       imgSize,
                                       scanlines_1, scanlines_2, r_start_end, numlines );

    }                           /* if */

    return error;

}                               /* icvlGetCoefficientOrto */

/*===========================================================================*/
CvStatus
icvGetStartEnd1( CvMatrix3 * matrix, CvSize imgSize, float *l_start_end, float *r_start_end )
{

    CvMatrix3 *F;
    int width, height;
    float l_diagonal[3];
    float r_diagonal[3];
    float l_point[3]={0,0,0}, r_point[3], epiline[3]={0,0,0};
    CvStatus error = CV_OK;

    F = matrix;
    width = imgSize.width - 1;
    height = imgSize.height - 1;

    l_diagonal[0] = (float) 1 / width;
    l_diagonal[1] = (float) 1 / height;
    l_diagonal[2] = -1;

    r_diagonal[0] = (float) 1 / width;
    r_diagonal[1] = (float) 1 / height;
    r_diagonal[2] = -1;

    r_point[0] = (float) width;
    r_point[1] = 0;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );
    error = icvCrossLines( l_diagonal, epiline, l_point );

    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[0] = l_point[0];
        l_start_end[1] = l_point[1];

        r_start_end[0] = r_point[0];
        r_start_end[1] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {
                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }                       /* if */
    }                           /* if */

    r_point[0] = 0;
    r_point[1] = (float) height;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );
    error = icvCrossLines( l_diagonal, epiline, l_point );
    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[2] = l_point[0];
        l_start_end[3] = l_point[1];

        r_start_end[2] = r_point[0];
        r_start_end[3] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }                       /* if */
    }                           /* if */

    return error;

}                               /* icvlGetStartEnd1 */

/*===========================================================================*/
CvStatus
icvGetStartEnd2( CvMatrix3 * matrix, CvSize imgSize, float *l_start_end, float *r_start_end )
{


    CvMatrix3 *F;
    int width, height;
    float l_diagonal[3];
    float r_diagonal[3];
    float l_point[3]={0,0,0}, r_point[3], epiline[3]={0,0,0};
    CvStatus error = CV_OK;

    F = matrix;

    width = imgSize.width - 1;
    height = imgSize.height - 1;

    l_diagonal[0] = (float) 1 / width;
    l_diagonal[1] = (float) 1 / height;
    l_diagonal[2] = -1;

    r_diagonal[0] = (float) height / width;
    r_diagonal[1] = -1;
    r_diagonal[2] = 0;

    r_point[0] = 0;
    r_point[1] = 0;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );

    error = icvCrossLines( l_diagonal, epiline, l_point );

    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[0] = l_point[0];
        l_start_end[1] = l_point[1];

        r_start_end[0] = r_point[0];
        r_start_end[1] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );

            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }                       /* if */
    }                           /* if */

    r_point[0] = (float) width;
    r_point[1] = (float) height;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );
    error = icvCrossLines( l_diagonal, epiline, l_point );
    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[2] = l_point[0];
        l_start_end[3] = l_point[1];

        r_start_end[2] = r_point[0];
        r_start_end[3] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }
    }                           /* if */

    return error;

}                               /* icvlGetStartEnd2 */

/*===========================================================================*/
CvStatus
icvGetStartEnd3( CvMatrix3 * matrix, CvSize imgSize, float *l_start_end, float *r_start_end )
{

    CvMatrix3 *F;
    int width, height;
    float l_diagonal[3];
    float r_diagonal[3];
    float l_point[3]={0,0,0}, r_point[3], epiline[3]={0,0,0};
    CvStatus error = CV_OK;

    F = matrix;

    width = imgSize.width - 1;
    height = imgSize.height - 1;

    l_diagonal[0] = (float) height / width;
    l_diagonal[1] = -1;
    l_diagonal[2] = 0;

    r_diagonal[0] = (float) 1 / width;
    r_diagonal[1] = (float) 1 / height;
    r_diagonal[2] = -1;

    r_point[0] = 0;
    r_point[1] = 0;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );

    error = icvCrossLines( l_diagonal, epiline, l_point );

    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[0] = l_point[0];
        l_start_end[1] = l_point[1];

        r_start_end[0] = r_point[0];
        r_start_end[1] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }                       /* if */
    }                           /* if */

    r_point[0] = (float) width;
    r_point[1] = (float) height;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );
    error = icvCrossLines( l_diagonal, epiline, l_point );
    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[2] = l_point[0];
        l_start_end[3] = l_point[1];

        r_start_end[2] = r_point[0];
        r_start_end[3] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );

            error = icvCrossLines( r_diagonal, epiline, r_point );

            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );

            error = icvCrossLines( r_diagonal, epiline, r_point );

            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }                       /* if */
    }                           /* if */

    return error;

}                               /* icvlGetStartEnd3 */

/*===========================================================================*/
CvStatus
icvGetStartEnd4( CvMatrix3 * matrix, CvSize imgSize, float *l_start_end, float *r_start_end )
{
    CvMatrix3 *F;
    int width, height;
    float l_diagonal[3];
    float r_diagonal[3];
    float l_point[3], r_point[3], epiline[3]={0,0,0};
    CvStatus error;

    F = matrix;

    width = imgSize.width - 1;
    height = imgSize.height - 1;

    l_diagonal[0] = (float) height / width;
    l_diagonal[1] = -1;
    l_diagonal[2] = 0;

    r_diagonal[0] = (float) height / width;
    r_diagonal[1] = -1;
    r_diagonal[2] = 0;

    r_point[0] = 0;
    r_point[1] = 0;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );
    error = icvCrossLines( l_diagonal, epiline, l_point );

    if( error != CV_NO_ERR )
        return error;

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[0] = l_point[0];
        l_start_end[1] = l_point[1];

        r_start_end[0] = r_point[0];
        r_start_end[1] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[0] = l_point[0];
                l_start_end[1] = l_point[1];

                r_start_end[0] = r_point[0];
                r_start_end[1] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }                       /* if */
    }                           /* if */

    r_point[0] = (float) width;
    r_point[1] = (float) height;
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, epiline );
    error = icvCrossLines( l_diagonal, epiline, l_point );
    assert( error == CV_NO_ERR );

    if( l_point[0] >= 0 && l_point[0] <= width )
    {

        l_start_end[2] = l_point[0];
        l_start_end[3] = l_point[1];

        r_start_end[2] = r_point[0];
        r_start_end[3] = r_point[1];

    }
    else
    {

        if( l_point[0] < 0 )
        {

            l_point[0] = 0;
            l_point[1] = 0;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;

        }
        else
        {                       /* if( l_point[0] > width ) */

            l_point[0] = (float) width;
            l_point[1] = (float) height;
            l_point[2] = 1;

            icvMultMatrixTVector3( F, l_point, epiline );
            error = icvCrossLines( r_diagonal, epiline, r_point );
            assert( error == CV_NO_ERR );

            if( r_point[0] >= 0 && r_point[0] <= width )
            {

                l_start_end[2] = l_point[0];
                l_start_end[3] = l_point[1];

                r_start_end[2] = r_point[0];
                r_start_end[3] = r_point[1];
            }
            else
                return CV_BADFACTOR_ERR;
        }                       /* if */
    }                           /* if */

    return CV_NO_ERR;

}                               /* icvlGetStartEnd4 */

/*===========================================================================*/
CvStatus
icvBuildScanlineLeft( CvMatrix3 * matrix,
                      CvSize imgSize,
                      int *scanlines_1, int *scanlines_2, float *l_start_end, int *numlines )
{
    int prewarp_height;
    float l_point[3];
    float r_point[3];
    float height;
    float delta_x;
    float delta_y;
    CvStatus error = CV_OK;
    CvMatrix3 *F;
    float i;
    int offset;
    float epiline[3];
    double a, b;

    assert( l_start_end != 0 );

    a = fabs( l_start_end[2] - l_start_end[0] );
    b = fabs( l_start_end[3] - l_start_end[1] );
    prewarp_height = cvRound( MAX(a, b) );

    *numlines = prewarp_height;

    if( scanlines_1 == 0 && scanlines_2 == 0 )
        return CV_NO_ERR;

    F = matrix;


    l_point[2] = 1;
    height = (float) prewarp_height;

    delta_x = (l_start_end[2] - l_start_end[0]) / height;

    l_start_end[0] += delta_x;
    l_start_end[2] -= delta_x;

    delta_x = (l_start_end[2] - l_start_end[0]) / height;
    delta_y = (l_start_end[3] - l_start_end[1]) / height;

    l_start_end[1] += delta_y;
    l_start_end[3] -= delta_y;

    delta_y = (l_start_end[3] - l_start_end[1]) / height;

    for( i = 0, offset = 0; i < height; i++, offset += 4 )
    {

        l_point[0] = l_start_end[0] + i * delta_x;
        l_point[1] = l_start_end[1] + i * delta_y;

        icvMultMatrixTVector3( F, l_point, epiline );

        error = icvGetCrossEpilineFrame( imgSize, epiline,
                                         scanlines_2 + offset,
                                         scanlines_2 + offset + 1,
                                         scanlines_2 + offset + 2, scanlines_2 + offset + 3 );



        assert( error == CV_NO_ERR );

        r_point[0] = -(float) (*(scanlines_2 + offset));
        r_point[1] = -(float) (*(scanlines_2 + offset + 1));
        r_point[2] = -1;

        icvMultMatrixVector3( F, r_point, epiline );

        error = icvGetCrossEpilineFrame( imgSize, epiline,
                                         scanlines_1 + offset,
                                         scanlines_1 + offset + 1,
                                         scanlines_1 + offset + 2, scanlines_1 + offset + 3 );

        assert( error == CV_NO_ERR );
    }                           /* for */

    *numlines = prewarp_height;

    return error;

} /*icvlBuildScanlineLeft */

/*===========================================================================*/
CvStatus
icvBuildScanlineRight( CvMatrix3 * matrix,
                       CvSize imgSize,
                       int *scanlines_1, int *scanlines_2, float *r_start_end, int *numlines )
{
    int prewarp_height;
    float l_point[3];
    float r_point[3];
    float height;
    float delta_x;
    float delta_y;
    CvStatus error = CV_OK;
    CvMatrix3 *F;
    float i;
    int offset;
    float epiline[3];
    double a, b;

    assert( r_start_end != 0 );

    a = fabs( r_start_end[2] - r_start_end[0] );
    b = fabs( r_start_end[3] - r_start_end[1] );
    prewarp_height = cvRound( MAX(a, b) );

    *numlines = prewarp_height;

    if( scanlines_1 == 0 && scanlines_2 == 0 )
        return CV_NO_ERR;

    F = matrix;

    r_point[2] = 1;
    height = (float) prewarp_height;

    delta_x = (r_start_end[2] - r_start_end[0]) / height;

    r_start_end[0] += delta_x;
    r_start_end[2] -= delta_x;

    delta_x = (r_start_end[2] - r_start_end[0]) / height;
    delta_y = (r_start_end[3] - r_start_end[1]) / height;

    r_start_end[1] += delta_y;
    r_start_end[3] -= delta_y;

    delta_y = (r_start_end[3] - r_start_end[1]) / height;

    for( i = 0, offset = 0; i < height; i++, offset += 4 )
    {

        r_point[0] = r_start_end[0] + i * delta_x;
        r_point[1] = r_start_end[1] + i * delta_y;

        icvMultMatrixVector3( F, r_point, epiline );

        error = icvGetCrossEpilineFrame( imgSize, epiline,
                                         scanlines_1 + offset,
                                         scanlines_1 + offset + 1,
                                         scanlines_1 + offset + 2, scanlines_1 + offset + 3 );


        assert( error == CV_NO_ERR );

        l_point[0] = -(float) (*(scanlines_1 + offset));
        l_point[1] = -(float) (*(scanlines_1 + offset + 1));

        l_point[2] = -1;

        icvMultMatrixTVector3( F, l_point, epiline );
        error = icvGetCrossEpilineFrame( imgSize, epiline,
                                         scanlines_2 + offset,
                                         scanlines_2 + offset + 1,
                                         scanlines_2 + offset + 2, scanlines_2 + offset + 3 );


        assert( error == CV_NO_ERR );
    }                           /* for */

    *numlines = prewarp_height;

    return error;

} /*icvlBuildScanlineRight */

/*===========================================================================*/
#define Abs(x)              ( (x)<0 ? -(x):(x) )
#define Sgn(x)              ( (x)<0 ? -1:1 )    /* Sgn(0) = 1 ! */

static CvStatus
icvBuildScanline( CvSize imgSize, float *epiline, float *kx, float *cx, float *ky, float *cy )
{
    float point[4][2], d;
    int sign[4], i;

    float width, height;

    if( REAL_ZERO( epiline[0] ) && REAL_ZERO( epiline[1] ))
        return CV_BADFACTOR_ERR;

    width = (float) imgSize.width - 1;
    height = (float) imgSize.height - 1;

    sign[0] = Sgn( epiline[2] );
    sign[1] = Sgn( epiline[0] * width + epiline[2] );
    sign[2] = Sgn( epiline[1] * height + epiline[2] );
    sign[3] = Sgn( epiline[0] * width + epiline[1] * height + epiline[2] );

    i = 0;

    if( sign[0] * sign[1] < 0 )
    {

        point[i][0] = -epiline[2] / epiline[0];
        point[i][1] = 0;
        i++;
    }                           /* if */

    if( sign[0] * sign[2] < 0 )
    {

        point[i][0] = 0;
        point[i][1] = -epiline[2] / epiline[1];
        i++;
    }                           /* if */

    if( sign[1] * sign[3] < 0 )
    {

        point[i][0] = width;
        point[i][1] = -(epiline[0] * width + epiline[2]) / epiline[1];
        i++;
    }                           /* if */

    if( sign[2] * sign[3] < 0 )
    {

        point[i][0] = -(epiline[1] * height + epiline[2]) / epiline[0];
        point[i][1] = height;
    }                           /* if */

    if( sign[0] == sign[1] && sign[0] == sign[2] && sign[0] == sign[3] )
        return CV_BADFACTOR_ERR;

    if( !kx && !ky && !cx && !cy )
        return CV_BADFACTOR_ERR;

    if( kx && ky )
    {

        *kx = -epiline[1];
        *ky = epiline[0];

        d = (float) MAX( Abs( *kx ), Abs( *ky ));

        *kx /= d;
        *ky /= d;
    }                           /* if */

    if( cx && cy )
    {

        if( (point[0][0] - point[1][0]) * epiline[1] +
            (point[1][1] - point[0][1]) * epiline[0] > 0 )
        {

            *cx = point[0][0];
            *cy = point[0][1];

        }
        else
        {

            *cx = point[1][0];
            *cy = point[1][1];
        }                       /* if */
    }                           /* if */

    return CV_NO_ERR;

}                               /* icvlBuildScanline */

/*===========================================================================*/
CvStatus
icvGetCoefficientStereo( CvMatrix3 * matrix,
                         CvSize imgSize,
                         float *l_epipole,
                         float *r_epipole, int *scanlines_1, int *scanlines_2, int *numlines )
{
    int i, j, turn;
    float width, height;
    float l_angle[2], r_angle[2];
    float l_radius, r_radius;
    float r_point[3], l_point[3];
    float l_epiline[3], r_epiline[3], x, y;
    float swap;

    float radius1, radius2, radius3, radius4;

    float l_start_end[4], r_start_end[4];
    CvMatrix3 *F;
    CvStatus error;
    float Region[3][3][4] = {
       {{0.f, 0.f, 1.f, 1.f}, {0.f, 1.f, 1.f, 1.f}, {0.f, 1.f, 1.f, 0.f}},
        {{0.f, 0.f, 0.f, 1.f}, {2.f, 2.f, 2.f, 2.f}, {1.f, 1.f, 1.f, 0.f}},
        {{1.f, 0.f, 0.f, 1.f}, {1.f, 0.f, 0.f, 0.f}, {1.f, 1.f, 0.f, 0.f}}
    };


    width = (float) imgSize.width - 1;
    height = (float) imgSize.height - 1;

    F = matrix;

    if( F->m[0][0] * F->m[1][1] - F->m[1][0] * F->m[0][1] > 0 )
        turn = 1;
    else
        turn = -1;

    if( l_epipole[0] < 0 )
        i = 0;
    else if( l_epipole[0] < width )
        i = 1;
    else
        i = 2;

    if( l_epipole[1] < 0 )
        j = 2;
    else if( l_epipole[1] < height )
        j = 1;
    else
        j = 0;

    l_start_end[0] = Region[j][i][0];
    l_start_end[1] = Region[j][i][1];
    l_start_end[2] = Region[j][i][2];
    l_start_end[3] = Region[j][i][3];

    if( r_epipole[0] < 0 )
        i = 0;
    else if( r_epipole[0] < width )
        i = 1;
    else
        i = 2;

    if( r_epipole[1] < 0 )
        j = 2;
    else if( r_epipole[1] < height )
        j = 1;
    else
        j = 0;

    r_start_end[0] = Region[j][i][0];
    r_start_end[1] = Region[j][i][1];
    r_start_end[2] = Region[j][i][2];
    r_start_end[3] = Region[j][i][3];

    radius1 = l_epipole[0] * l_epipole[0] + (l_epipole[1] - height) * (l_epipole[1] - height);

    radius2 = (l_epipole[0] - width) * (l_epipole[0] - width) +
        (l_epipole[1] - height) * (l_epipole[1] - height);

    radius3 = l_epipole[0] * l_epipole[0] + l_epipole[1] * l_epipole[1];

    radius4 = (l_epipole[0] - width) * (l_epipole[0] - width) + l_epipole[1] * l_epipole[1];


    l_radius = (float) sqrt( (double)MAX( MAX( radius1, radius2 ), MAX( radius3, radius4 )));

    radius1 = r_epipole[0] * r_epipole[0] + (r_epipole[1] - height) * (r_epipole[1] - height);

    radius2 = (r_epipole[0] - width) * (r_epipole[0] - width) +
        (r_epipole[1] - height) * (r_epipole[1] - height);

    radius3 = r_epipole[0] * r_epipole[0] + r_epipole[1] * r_epipole[1];

    radius4 = (r_epipole[0] - width) * (r_epipole[0] - width) + r_epipole[1] * r_epipole[1];


    r_radius = (float) sqrt( (double)MAX( MAX( radius1, radius2 ), MAX( radius3, radius4 )));

    if( l_start_end[0] == 2 && r_start_end[0] == 2 )
    {
        if( l_radius > r_radius )
        {

            l_angle[0] = 0.0f;
            l_angle[1] = (float) CV_PI;

            error = icvBuildScanlineLeftStereo( imgSize,
                                                matrix,
                                                l_epipole,
                                                l_angle,
                                                l_radius, scanlines_1, scanlines_2, numlines );

            return error;
        }
        else
        {

            r_angle[0] = 0.0f;
            r_angle[1] = (float) CV_PI;

            error = icvBuildScanlineRightStereo( imgSize,
                                                 matrix,
                                                 r_epipole,
                                                 r_angle,
                                                 r_radius,
                                                 scanlines_1, scanlines_2, numlines );

            return error;
        }                       /* if */
    }

    if( l_start_end[0] == 2 )
    {

        r_angle[0] = (float) atan2( r_start_end[1] * height - r_epipole[1],
                                    r_start_end[0] * width - r_epipole[0] );
        r_angle[1] = (float) atan2( r_start_end[3] * height - r_epipole[1],
                                    r_start_end[2] * width - r_epipole[0] );

        if( r_angle[0] > r_angle[1] )
            r_angle[1] += (float) (CV_PI * 2);

        error = icvBuildScanlineRightStereo( imgSize,
                                             matrix,
                                             r_epipole,
                                             r_angle,
                                             r_radius, scanlines_1, scanlines_2, numlines );

        return error;
    }                           /* if */

    if( r_start_end[0] == 2 )
    {

        l_point[0] = l_start_end[0] * width;
        l_point[1] = l_start_end[1] * height;
        l_point[2] = 1;

        icvMultMatrixTVector3( F, l_point, r_epiline );

        l_angle[0] = (float) atan2( l_start_end[1] * height - l_epipole[1],
                                    l_start_end[0] * width - l_epipole[0] );
        l_angle[1] = (float) atan2( l_start_end[3] * height - l_epipole[1],
                                    l_start_end[2] * width - l_epipole[0] );

        if( l_angle[0] > l_angle[1] )
            l_angle[1] += (float) (CV_PI * 2);

        error = icvBuildScanlineLeftStereo( imgSize,
                                            matrix,
                                            l_epipole,
                                            l_angle,
                                            l_radius, scanlines_1, scanlines_2, numlines );

        return error;

    }                           /* if */

    l_start_end[0] *= width;
    l_start_end[1] *= height;
    l_start_end[2] *= width;
    l_start_end[3] *= height;

    r_start_end[0] *= width;
    r_start_end[1] *= height;
    r_start_end[2] *= width;
    r_start_end[3] *= height;

    r_point[0] = r_start_end[0];
    r_point[1] = r_start_end[1];
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, l_epiline );
    error = icvBuildScanline( imgSize, l_epiline, 0, &x, 0, &y );

    if( error == CV_NO_ERR )
    {

        l_angle[0] = (float) atan2( y - l_epipole[1], x - l_epipole[0] );

        r_angle[0] = (float) atan2( r_point[1] - r_epipole[1], r_point[0] - r_epipole[0] );

    }
    else
    {

        if( turn == 1 )
        {

            l_point[0] = l_start_end[0];
            l_point[1] = l_start_end[1];

        }
        else
        {

            l_point[0] = l_start_end[2];
            l_point[1] = l_start_end[3];
        }                       /* if */

        l_point[2] = 1;

        icvMultMatrixTVector3( F, l_point, r_epiline );
        error = icvBuildScanline( imgSize, r_epiline, 0, &x, 0, &y );

        if( error == CV_NO_ERR )
        {

            r_angle[0] = (float) atan2( y - r_epipole[1], x - r_epipole[0] );

            l_angle[0] = (float) atan2( l_point[1] - l_epipole[1], l_point[0] - l_epipole[0] );

        }
        else
            return CV_BADFACTOR_ERR;
    }                           /* if */

    r_point[0] = r_start_end[2];
    r_point[1] = r_start_end[3];
    r_point[2] = 1;

    icvMultMatrixVector3( F, r_point, l_epiline );
    error = icvBuildScanline( imgSize, l_epiline, 0, &x, 0, &y );

    if( error == CV_NO_ERR )
    {

        l_angle[1] = (float) atan2( y - l_epipole[1], x - l_epipole[0] );

        r_angle[1] = (float) atan2( r_point[1] - r_epipole[1], r_point[0] - r_epipole[0] );

    }
    else
    {

        if( turn == 1 )
        {

            l_point[0] = l_start_end[2];
            l_point[1] = l_start_end[3];

        }
        else
        {

            l_point[0] = l_start_end[0];
            l_point[1] = l_start_end[1];
        }                       /* if */

        l_point[2] = 1;

        icvMultMatrixTVector3( F, l_point, r_epiline );
        error = icvBuildScanline( imgSize, r_epiline, 0, &x, 0, &y );

        if( error == CV_NO_ERR )
        {

            r_angle[1] = (float) atan2( y - r_epipole[1], x - r_epipole[0] );

            l_angle[1] = (float) atan2( l_point[1] - l_epipole[1], l_point[0] - l_epipole[0] );

        }
        else
            return CV_BADFACTOR_ERR;
    }                           /* if */

    if( l_angle[0] > l_angle[1] )
    {

        swap = l_angle[0];
        l_angle[0] = l_angle[1];
        l_angle[1] = swap;
    }                           /* if */

    if( l_angle[1] - l_angle[0] > CV_PI )
    {

        swap = l_angle[0];
        l_angle[0] = l_angle[1];
        l_angle[1] = swap + (float) (CV_PI * 2);
    }                           /* if */

    if( r_angle[0] > r_angle[1] )
    {

        swap = r_angle[0];
        r_angle[0] = r_angle[1];
        r_angle[1] = swap;
    }                           /* if */

    if( r_angle[1] - r_angle[0] > CV_PI )
    {

        swap = r_angle[0];
        r_angle[0] = r_angle[1];
        r_angle[1] = swap + (float) (CV_PI * 2);
    }                           /* if */

    if( l_radius * (l_angle[1] - l_angle[0]) > r_radius * (r_angle[1] - r_angle[0]) )
        error = icvBuildScanlineLeftStereo( imgSize,
                                            matrix,
                                            l_epipole,
                                            l_angle,
                                            l_radius, scanlines_1, scanlines_2, numlines );

    else
        error = icvBuildScanlineRightStereo( imgSize,
                                             matrix,
                                             r_epipole,
                                             r_angle,
                                             r_radius, scanlines_1, scanlines_2, numlines );


    return error;

}                               /* icvGetCoefficientStereo */

/*===========================================================================*/
CvStatus
icvBuildScanlineLeftStereo( CvSize imgSize,
                            CvMatrix3 * matrix,
                            float *l_epipole,
                            float *l_angle,
                            float l_radius, int *scanlines_1, int *scanlines_2, int *numlines )
{
    //int prewarp_width;
    int prewarp_height;
    float i;
    int offset;
    float height;
    float delta;
    float angle;
    float l_point[3];
    float l_epiline[3];
    float r_epiline[3];
    CvStatus error = CV_OK;
    CvMatrix3 *F;


    assert( l_angle != 0 && !REAL_ZERO( l_radius ));

    /*prewarp_width = (int) (sqrt( image_width * image_width +
                                 image_height * image_height ) + 1);*/

    prewarp_height = (int) (l_radius * (l_angle[1] - l_angle[0]));

    *numlines = prewarp_height;

    if( scanlines_1 == 0 && scanlines_2 == 0 )
        return CV_NO_ERR;

    F = matrix;

    l_point[2] = 1;
    height = (float) prewarp_height;

    delta = (l_angle[1] - l_angle[0]) / height;

    l_angle[0] += delta;
    l_angle[1] -= delta;

    delta = (l_angle[1] - l_angle[0]) / height;

    for( i = 0, offset = 0; i < height; i++, offset += 4 )
    {

        angle = l_angle[0] + i * delta;

        l_point[0] = l_epipole[0] + l_radius * (float) cos( angle );
        l_point[1] = l_epipole[1] + l_radius * (float) sin( angle );

        icvMultMatrixTVector3( F, l_point, r_epiline );

        error = icvGetCrossEpilineFrame( imgSize, r_epiline,
                                         scanlines_2 + offset,
                                         scanlines_2 + offset + 1,
                                         scanlines_2 + offset + 2, scanlines_2 + offset + 3 );


        l_epiline[0] = l_point[1] - l_epipole[1];
        l_epiline[1] = l_epipole[0] - l_point[0];
        l_epiline[2] = l_point[0] * l_epipole[1] - l_point[1] * l_epipole[0];

        if( Sgn( l_epiline[0] * r_epiline[0] + l_epiline[1] * r_epiline[1] ) < 0 )
        {

            l_epiline[0] = -l_epiline[0];
            l_epiline[1] = -l_epiline[1];
            l_epiline[2] = -l_epiline[2];
        }                       /* if */

        error = icvGetCrossEpilineFrame( imgSize, l_epiline,
                                         scanlines_1 + offset,
                                         scanlines_1 + offset + 1,
                                         scanlines_1 + offset + 2, scanlines_1 + offset + 3 );

    }                           /* for */

    *numlines = prewarp_height;

    return error;

}                               /* icvlBuildScanlineLeftStereo */

/*===========================================================================*/
CvStatus
icvBuildScanlineRightStereo( CvSize imgSize,
                             CvMatrix3 * matrix,
                             float *r_epipole,
                             float *r_angle,
                             float r_radius,
                             int *scanlines_1, int *scanlines_2, int *numlines )
{
    //int prewarp_width;
    int prewarp_height;
    float i;
    int offset;
    float height;
    float delta;
    float angle;
    float r_point[3];
    float l_epiline[3];
    float r_epiline[3];
    CvStatus error = CV_OK;
    CvMatrix3 *F;

    assert( r_angle != 0 && !REAL_ZERO( r_radius ));

    /*prewarp_width = (int) (sqrt( image_width * image_width +
                                 image_height * image_height ) + 1);*/

    prewarp_height = (int) (r_radius * (r_angle[1] - r_angle[0]));

    *numlines = prewarp_height;

    if( scanlines_1 == 0 && scanlines_2 == 0 )
        return CV_NO_ERR;

    F = matrix;

    r_point[2] = 1;
    height = (float) prewarp_height;

    delta = (r_angle[1] - r_angle[0]) / height;

    r_angle[0] += delta;
    r_angle[1] -= delta;

    delta = (r_angle[1] - r_angle[0]) / height;

    for( i = 0, offset = 0; i < height; i++, offset += 4 )
    {

        angle = r_angle[0] + i * delta;

        r_point[0] = r_epipole[0] + r_radius * (float) cos( angle );
        r_point[1] = r_epipole[1] + r_radius * (float) sin( angle );

        icvMultMatrixVector3( F, r_point, l_epiline );

        error = icvGetCrossEpilineFrame( imgSize, l_epiline,
                                         scanlines_1 + offset,
                                         scanlines_1 + offset + 1,
                                         scanlines_1 + offset + 2, scanlines_1 + offset + 3 );

        assert( error == CV_NO_ERR );

        r_epiline[0] = r_point[1] - r_epipole[1];
        r_epiline[1] = r_epipole[0] - r_point[0];
        r_epiline[2] = r_point[0] * r_epipole[1] - r_point[1] * r_epipole[0];

        if( Sgn( l_epiline[0] * r_epiline[0] + l_epiline[1] * r_epiline[1] ) < 0 )
        {

            r_epiline[0] = -r_epiline[0];
            r_epiline[1] = -r_epiline[1];
            r_epiline[2] = -r_epiline[2];
        }                       /* if */

        error = icvGetCrossEpilineFrame( imgSize, r_epiline,
                                         scanlines_2 + offset,
                                         scanlines_2 + offset + 1,
                                         scanlines_2 + offset + 2, scanlines_2 + offset + 3 );

        assert( error == CV_NO_ERR );
    }                           /* for */

    *numlines = prewarp_height;

    return error;

}                               /* icvlBuildScanlineRightStereo */

/*===========================================================================*/
CvStatus
icvGetCrossEpilineFrame( CvSize imgSize, float *epiline, int *x1, int *y1, int *x2, int *y2 )
{
    int tx, ty;
    float point[2][2];
    int sign[4], i;
    float width, height;
    double tmpvalue;

    if( REAL_ZERO( epiline[0] ) && REAL_ZERO( epiline[1] ))
        return CV_BADFACTOR_ERR;

    width = (float) imgSize.width - 1;
    height = (float) imgSize.height - 1;

    tmpvalue = epiline[2];
    sign[0] = SIGN( tmpvalue );

    tmpvalue = epiline[0] * width + epiline[2];
    sign[1] = SIGN( tmpvalue );

    tmpvalue = epiline[1] * height + epiline[2];
    sign[2] = SIGN( tmpvalue );

    tmpvalue = epiline[0] * width + epiline[1] * height + epiline[2];
    sign[3] = SIGN( tmpvalue );

    i = 0;
    for( tx = 0; tx < 2; tx++ )
    {
        for( ty = 0; ty < 2; ty++ )
        {

            if( sign[ty * 2 + tx] == 0 )
            {

                point[i][0] = width * tx;
                point[i][1] = height * ty;
                i++;

            }                   /* if */
        }                       /* for */
    }                           /* for */

    if( sign[0] * sign[1] < 0 )
    {
        point[i][0] = -epiline[2] / epiline[0];
        point[i][1] = 0;
        i++;
    }                           /* if */

    if( sign[0] * sign[2] < 0 )
    {
        point[i][0] = 0;
        point[i][1] = -epiline[2] / epiline[1];
        i++;
    }                           /* if */

    if( sign[1] * sign[3] < 0 )
    {
        point[i][0] = width;
        point[i][1] = -(epiline[0] * width + epiline[2]) / epiline[1];
        i++;
    }                           /* if */

    if( sign[2] * sign[3] < 0 )
    {
        point[i][0] = -(epiline[1] * height + epiline[2]) / epiline[0];
        point[i][1] = height;
    }                           /* if */

    if( sign[0] == sign[1] && sign[0] == sign[2] && sign[0] == sign[3] )
        return CV_BADFACTOR_ERR;

    if( (point[0][0] - point[1][0]) * epiline[1] +
        (point[1][1] - point[0][1]) * epiline[0] > 0 )
    {
        *x1 = (int) point[0][0];
        *y1 = (int) point[0][1];
        *x2 = (int) point[1][0];
        *y2 = (int) point[1][1];
    }
    else
    {
        *x1 = (int) point[1][0];
        *y1 = (int) point[1][1];
        *x2 = (int) point[0][0];
        *y2 = (int) point[0][1];
    }                           /* if */

    return CV_NO_ERR;
}                               /* icvlGetCrossEpilineFrame */

/*=====================================================================================*/

CV_IMPL void
cvMakeScanlines( const CvMatrix3* matrix, CvSize imgSize,
                 int *scanlines_1, int *scanlines_2,
                 int *lens_1, int *lens_2, int *numlines )
{
    IPPI_CALL( icvMakeScanlines( (CvMatrix3*)matrix, imgSize, scanlines_1,
                                 scanlines_2, lens_1, lens_2, numlines ));
}

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name: cvDeleteMoire
//    Purpose: The functions
//    Context:
//    Parameters:
//
//    Notes:
//F*/
CV_IMPL void
cvMakeAlphaScanlines( int *scanlines_1,
                      int *scanlines_2,
                      int *scanlines_a, int *lens, int numlines, float alpha )
{
    IPPI_CALL( icvMakeAlphaScanlines( scanlines_1, scanlines_2, scanlines_a,
                                      lens, numlines, alpha ));
}
