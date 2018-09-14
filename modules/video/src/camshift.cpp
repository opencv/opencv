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
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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

int cv::meanShift( InputArray _probImage, Rect& window, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    Size size;
    int cn;
    Mat mat;
    UMat umat;
    bool isUMat = _probImage.isUMat();

    if (isUMat)
        umat = _probImage.getUMat(), cn = umat.channels(), size = umat.size();
    else
        mat = _probImage.getMat(), cn = mat.channels(), size = mat.size();

    Rect cur_rect = window;

    CV_Assert( cn == 1 );

    if( window.height <= 0 || window.width <= 0 )
        CV_Error( Error::StsBadArg, "Input window has non-positive sizes" );

    window = window & Rect(0, 0, size.width, size.height);

    double eps = (criteria.type & TermCriteria::EPS) ? std::max(criteria.epsilon, 0.) : 1.;
    eps = cvRound(eps*eps);
    int i, niters = (criteria.type & TermCriteria::MAX_ITER) ? std::max(criteria.maxCount, 1) : 100;

    for( i = 0; i < niters; i++ )
    {
        cur_rect = cur_rect & Rect(0, 0, size.width, size.height);
        if( cur_rect == Rect() )
        {
            cur_rect.x = size.width/2;
            cur_rect.y = size.height/2;
        }
        cur_rect.width = std::max(cur_rect.width, 1);
        cur_rect.height = std::max(cur_rect.height, 1);

        Moments m = isUMat ? moments(umat(cur_rect)) : moments(mat(cur_rect));

        // Calculating center of mass
        if( fabs(m.m00) < DBL_EPSILON )
            break;

        int dx = cvRound( m.m10/m.m00 - window.width*0.5 );
        int dy = cvRound( m.m01/m.m00 - window.height*0.5 );

        int nx = std::min(std::max(cur_rect.x + dx, 0), size.width - cur_rect.width);
        int ny = std::min(std::max(cur_rect.y + dy, 0), size.height - cur_rect.height);

        dx = nx - cur_rect.x;
        dy = ny - cur_rect.y;
        cur_rect.x = nx;
        cur_rect.y = ny;

        // Check for coverage centers mass & window
        if( dx*dx + dy*dy < eps )
            break;
    }

    window = cur_rect;
    return i;
}


cv::RotatedRect cv::CamShift( InputArray _probImage, Rect& window,
                              TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    const int TOLERANCE = 10;
    Size size;
    Mat mat;
    UMat umat;
    bool isUMat = _probImage.isUMat();

    if (isUMat)
        umat = _probImage.getUMat(), size = umat.size();
    else
        mat = _probImage.getMat(), size = mat.size();

    meanShift( _probImage, window, criteria );

    window.x -= TOLERANCE;
    if( window.x < 0 )
        window.x = 0;

    window.y -= TOLERANCE;
    if( window.y < 0 )
        window.y = 0;

    window.width += 2 * TOLERANCE;
    if( window.x + window.width > size.width )
        window.width = size.width - window.x;

    window.height += 2 * TOLERANCE;
    if( window.y + window.height > size.height )
        window.height = size.height - window.y;

    // Calculating moments in new center mass
    Moments m = isUMat ? moments(umat(window)) : moments(mat(window));

    double m00 = m.m00, m10 = m.m10, m01 = m.m01;
    double mu11 = m.mu11, mu20 = m.mu20, mu02 = m.mu02;

    if( fabs(m00) < DBL_EPSILON )
        return RotatedRect();

    double inv_m00 = 1. / m00;
    int xc = cvRound( m10 * inv_m00 + window.x );
    int yc = cvRound( m01 * inv_m00 + window.y );
    double a = mu20 * inv_m00, b = mu11 * inv_m00, c = mu02 * inv_m00;

    // Calculating width & height
    double square = std::sqrt( 4 * b * b + (a - c) * (a - c) );

    // Calculating orientation
    double theta = atan2( 2 * b, a - c + square );

    // Calculating width & length of figure
    double cs = cos( theta );
    double sn = sin( theta );

    double rotate_a = cs * cs * mu20 + 2 * cs * sn * mu11 + sn * sn * mu02;
    double rotate_c = sn * sn * mu20 - 2 * cs * sn * mu11 + cs * cs * mu02;
    double length = std::sqrt( rotate_a * inv_m00 ) * 4;
    double width = std::sqrt( rotate_c * inv_m00 ) * 4;

    // In case, when tetta is 0 or 1.57... the Length & Width may be exchanged
    if( length < width )
    {
        std::swap( length, width );
        std::swap( cs, sn );
        theta = CV_PI*0.5 - theta;
    }

    // Saving results
    int _xc = cvRound( xc );
    int _yc = cvRound( yc );

    int t0 = cvRound( fabs( length * cs ));
    int t1 = cvRound( fabs( width * sn ));

    t0 = MAX( t0, t1 ) + 2;
    window.width = MIN( t0, (size.width - _xc) * 2 );

    t0 = cvRound( fabs( length * sn ));
    t1 = cvRound( fabs( width * cs ));

    t0 = MAX( t0, t1 ) + 2;
    window.height = MIN( t0, (size.height - _yc) * 2 );

    window.x = MAX( 0, _xc - window.width / 2 );
    window.y = MAX( 0, _yc - window.height / 2 );

    window.width = MIN( size.width - window.x, window.width );
    window.height = MIN( size.height - window.y, window.height );

    RotatedRect box;
    box.size.height = (float)length;
    box.size.width = (float)width;
    box.angle = (float)((CV_PI*0.5+theta)*180./CV_PI);
    while(box.angle < 0)
        box.angle += 360;
    while(box.angle >= 360)
        box.angle -= 360;
    if(box.angle >= 180)
        box.angle -= 180;
    box.center = Point2f( window.x + window.width*0.5f, window.y + window.height*0.5f);

    return box;
}

/* End of file. */
