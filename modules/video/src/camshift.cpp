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


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvMeanShift
//    Purpose: MeanShift algorithm
//    Context:
//    Parameters:
//      imgProb     - 2D object probability distribution
//      windowIn    - CvRect of CAMSHIFT Window intial size
//      numIters    - If CAMSHIFT iterates this many times, stop
//      windowOut   - Location, height and width of converged CAMSHIFT window
//      len         - If != NULL, return equivalent len
//      width       - If != NULL, return equivalent width
//      itersUsed   - Returns number of iterations CAMSHIFT took to converge
//    Returns:
//      The function itself returns the area found
//    Notes:
//F*/
CV_IMPL int
cvMeanShift( const void* imgProb, CvRect windowIn,
             CvTermCriteria criteria, CvConnectedComp* comp )
{
    CvMoments moments;
    int    i = 0, eps;
    CvMat  stub, *mat = (CvMat*)imgProb;
    CvMat  cur_win;
    CvRect cur_rect = windowIn;

    if( comp )
        comp->rect = windowIn;

    moments.m00 = moments.m10 = moments.m01 = 0;

    mat = cvGetMat( mat, &stub );

    if( CV_MAT_CN( mat->type ) > 1 )
        CV_Error( CV_BadNumChannels, cvUnsupportedFormat );

    if( windowIn.height <= 0 || windowIn.width <= 0 )
        CV_Error( CV_StsBadArg, "Input window has non-positive sizes" );

    windowIn = cv::Rect(windowIn) & cv::Rect(0, 0, mat->cols, mat->rows);

    criteria = cvCheckTermCriteria( criteria, 1., 100 );
    eps = cvRound( criteria.epsilon * criteria.epsilon );

    for( i = 0; i < criteria.max_iter; i++ )
    {
        int dx, dy, nx, ny;
        double inv_m00;
        cur_rect = cv::Rect(cur_rect) & cv::Rect(0, 0, mat->cols, mat->rows);
        
        cvGetSubRect( mat, &cur_win, cur_rect ); 
        cvMoments( &cur_win, &moments );

        /* Calculating center of mass */
        if( fabs(moments.m00) < DBL_EPSILON )
            break;

        inv_m00 = moments.inv_sqrt_m00*moments.inv_sqrt_m00;
        dx = cvRound( moments.m10 * inv_m00 - windowIn.width*0.5 );
        dy = cvRound( moments.m01 * inv_m00 - windowIn.height*0.5 );

        nx = cur_rect.x + dx;
        ny = cur_rect.y + dy;

        if( nx < 0 )
            nx = 0;
        else if( nx + cur_rect.width > mat->cols )
            nx = mat->cols - cur_rect.width;

        if( ny < 0 )
            ny = 0;
        else if( ny + cur_rect.height > mat->rows )
            ny = mat->rows - cur_rect.height;

        dx = nx - cur_rect.x;
        dy = ny - cur_rect.y;
        cur_rect.x = nx;
        cur_rect.y = ny;

        /* Check for coverage centers mass & window */
        if( dx*dx + dy*dy < eps )
            break;
    }

    if( comp )
    {
        comp->rect = cur_rect;
        comp->area = (float)moments.m00;
    }

    return i;
}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    cvCamShift
//    Purpose: CAMSHIFT algorithm
//    Context:
//    Parameters:
//      imgProb     - 2D object probability distribution
//      windowIn    - CvRect of CAMSHIFT Window intial size
//      criteria    - criteria of stop finding window
//      windowOut   - Location, height and width of converged CAMSHIFT window
//      orientation - If != NULL, return distribution orientation
//      len         - If != NULL, return equivalent len
//      width       - If != NULL, return equivalent width
//      area        - sum of all elements in result window
//      itersUsed   - Returns number of iterations CAMSHIFT took to converge
//    Returns:
//      The function itself returns the area found
//    Notes:
//F*/
CV_IMPL int
cvCamShift( const void* imgProb, CvRect windowIn,
            CvTermCriteria criteria,
            CvConnectedComp* _comp,
            CvBox2D* box )
{
    const int TOLERANCE = 10;
    CvMoments moments;
    double m00 = 0, m10, m01, mu20, mu11, mu02, inv_m00;
    double a, b, c, xc, yc;
    double rotate_a, rotate_c;
    double theta = 0, square;
    double cs, sn;
    double length = 0, width = 0;
    int itersUsed = 0;
    CvConnectedComp comp;
    CvMat  cur_win, stub, *mat = (CvMat*)imgProb;

    comp.rect = windowIn;

    mat = cvGetMat( mat, &stub );

    itersUsed = cvMeanShift( mat, windowIn, criteria, &comp );
    windowIn = comp.rect;

    windowIn.x -= TOLERANCE;
    if( windowIn.x < 0 )
        windowIn.x = 0;

    windowIn.y -= TOLERANCE;
    if( windowIn.y < 0 )
        windowIn.y = 0;

    windowIn.width += 2 * TOLERANCE;
    if( windowIn.x + windowIn.width > mat->width )
        windowIn.width = mat->width - windowIn.x;

    windowIn.height += 2 * TOLERANCE;
    if( windowIn.y + windowIn.height > mat->height )
        windowIn.height = mat->height - windowIn.y;

    cvGetSubRect( mat, &cur_win, windowIn );

    /* Calculating moments in new center mass */
    cvMoments( &cur_win, &moments );

    m00 = moments.m00;
    m10 = moments.m10;
    m01 = moments.m01;
    mu11 = moments.mu11;
    mu20 = moments.mu20;
    mu02 = moments.mu02;

    if( fabs(m00) < DBL_EPSILON )
        return -1;

    inv_m00 = 1. / m00;
    xc = cvRound( m10 * inv_m00 + windowIn.x );
    yc = cvRound( m01 * inv_m00 + windowIn.y );
    a = mu20 * inv_m00;
    b = mu11 * inv_m00;
    c = mu02 * inv_m00;

    /* Calculating width & height */
    square = sqrt( 4 * b * b + (a - c) * (a - c) );

    /* Calculating orientation */
    theta = atan2( 2 * b, a - c + square );

    /* Calculating width & length of figure */
    cs = cos( theta );
    sn = sin( theta );

    rotate_a = cs * cs * mu20 + 2 * cs * sn * mu11 + sn * sn * mu02;
    rotate_c = sn * sn * mu20 - 2 * cs * sn * mu11 + cs * cs * mu02;
    length = sqrt( rotate_a * inv_m00 ) * 4;
    width = sqrt( rotate_c * inv_m00 ) * 4;

    /* In case, when tetta is 0 or 1.57... the Length & Width may be exchanged */
    if( length < width )
    {
        double t;
        
        CV_SWAP( length, width, t );
        CV_SWAP( cs, sn, t );
        theta = CV_PI*0.5 - theta;
    }

    /* Saving results */
    if( _comp || box )
    {
        int t0, t1;
        int _xc = cvRound( xc );
        int _yc = cvRound( yc );

        t0 = cvRound( fabs( length * cs ));
        t1 = cvRound( fabs( width * sn ));

        t0 = MAX( t0, t1 ) + 2;
        comp.rect.width = MIN( t0, (mat->width - _xc) * 2 );

        t0 = cvRound( fabs( length * sn ));
        t1 = cvRound( fabs( width * cs ));

        t0 = MAX( t0, t1 ) + 2;
        comp.rect.height = MIN( t0, (mat->height - _yc) * 2 );

        comp.rect.x = MAX( 0, _xc - comp.rect.width / 2 );
        comp.rect.y = MAX( 0, _yc - comp.rect.height / 2 );

        comp.rect.width = MIN( mat->width - comp.rect.x, comp.rect.width );
        comp.rect.height = MIN( mat->height - comp.rect.y, comp.rect.height );
        comp.area = (float) m00;
    }

    if( _comp )
        *_comp = comp;
    
    if( box )
    {
        box->size.height = (float)length;
        box->size.width = (float)width;
        box->angle = (float)(theta*180./CV_PI);
        box->center = cvPoint2D32f( comp.rect.x + comp.rect.width*0.5f,
                                    comp.rect.y + comp.rect.height*0.5f);
    }

    return itersUsed;
}

namespace cv
{

RotatedRect CamShift( const Mat& probImage, Rect& window,
                      TermCriteria criteria )
{
    CvConnectedComp comp;
    CvBox2D box;
    CvMat _probImage = probImage;
    cvCamShift(&_probImage, window, (CvTermCriteria)criteria, &comp, &box);
    window = comp.rect;
    return RotatedRect(Point2f(box.center), Size2f(box.size), box.angle);
}

int meanShift( const Mat& probImage, Rect& window, TermCriteria criteria )
{
    CvConnectedComp comp;
    CvMat _probImage = probImage;
    int iters = cvMeanShift(&_probImage, window, (CvTermCriteria)criteria, &comp );
    window = comp.rect;
    return iters;
}

}

/* End of file. */
