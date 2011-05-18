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

CV_IMPL void
cvFindCornerSubPix( const void* srcarr, CvPoint2D32f* corners,
                    int count, CvSize win, CvSize zeroZone,
                    CvTermCriteria criteria )
{
    cv::AutoBuffer<float> buffer;
    
    const int MAX_ITERS = 100;
    const float drv_x[] = { -1.f, 0.f, 1.f };
    const float drv_y[] = { 0.f, 0.5f, 0.f };
    float *maskX;
    float *maskY;
    float *mask;
    float *src_buffer;
    float *gx_buffer;
    float *gy_buffer;
    int win_w = win.width * 2 + 1, win_h = win.height * 2 + 1;
    int win_rect_size = (win_w + 4) * (win_h + 4);
    double coeff;
    CvSize size, src_buf_size;
    int i, j, k, pt_i;
    int max_iters = 10;
    double eps = 0;

    CvMat stub, *src = (CvMat*)srcarr;
    src = cvGetMat( srcarr, &stub );

    if( CV_MAT_TYPE( src->type ) != CV_8UC1 )
        CV_Error( CV_StsBadMask, "" );

    if( !corners )
        CV_Error( CV_StsNullPtr, "" );

    if( count < 0 )
        CV_Error( CV_StsBadSize, "" );

    if( count == 0 )
        return;

    if( win.width <= 0 || win.height <= 0 )
        CV_Error( CV_StsBadSize, "" );

    size = cvGetMatSize( src );

    if( size.width < win_w + 4 || size.height < win_h + 4 )
        CV_Error( CV_StsBadSize, "" );

    /* initialize variables, controlling loop termination */
    switch( criteria.type )
    {
    case CV_TERMCRIT_ITER:
        eps = 0.f;
        max_iters = criteria.max_iter;
        break;
    case CV_TERMCRIT_EPS:
        eps = criteria.epsilon;
        max_iters = MAX_ITERS;
        break;
    case CV_TERMCRIT_ITER | CV_TERMCRIT_EPS:
        eps = criteria.epsilon;
        max_iters = criteria.max_iter;
        break;
    default:
        assert( 0 );
        CV_Error( CV_StsBadFlag, "" );
    }

    eps = MAX( eps, 0 );
    eps *= eps;                 /* use square of error in comparsion operations. */

    max_iters = MAX( max_iters, 1 );
    max_iters = MIN( max_iters, MAX_ITERS );

    buffer.allocate( win_rect_size * 5 + win_w + win_h + 32 );

    /* assign pointers */
    maskX = buffer;
    maskY = maskX + win_w + 4;
    mask = maskY + win_h + 4;
    src_buffer = mask + win_w * win_h;
    gx_buffer = src_buffer + win_rect_size;
    gy_buffer = gx_buffer + win_rect_size;

    coeff = 1. / (win.width * win.width);

    /* calculate mask */
    for( i = -win.width, k = 0; i <= win.width; i++, k++ )
    {
        maskX[k] = (float)exp( -i * i * coeff );
    }

    if( win.width == win.height )
    {
        maskY = maskX;
    }
    else
    {
        coeff = 1. / (win.height * win.height);
        for( i = -win.height, k = 0; i <= win.height; i++, k++ )
        {
            maskY[k] = (float) exp( -i * i * coeff );
        }
    }

    for( i = 0; i < win_h; i++ )
    {
        for( j = 0; j < win_w; j++ )
        {
            mask[i * win_w + j] = maskX[j] * maskY[i];
        }
    }


    /* make zero_zone */
    if( zeroZone.width >= 0 && zeroZone.height >= 0 &&
        zeroZone.width * 2 + 1 < win_w && zeroZone.height * 2 + 1 < win_h )
    {
        for( i = win.height - zeroZone.height; i <= win.height + zeroZone.height; i++ )
        {
            for( j = win.width - zeroZone.width; j <= win.width + zeroZone.width; j++ )
            {
                mask[i * win_w + j] = 0;
            }
        }
    }

    /* set sizes of image rectangles, used in convolutions */
    src_buf_size.width = win_w + 2;
    src_buf_size.height = win_h + 2;

    /* do optimization loop for all the points */
    for( pt_i = 0; pt_i < count; pt_i++ )
    {
        CvPoint2D32f cT = corners[pt_i], cI = cT;
        int iter = 0;
        double err;

        do
        {
            CvPoint2D32f cI2;
            double a, b, c, bb1, bb2;

            IPPI_CALL( icvGetRectSubPix_8u32f_C1R( (uchar*)src->data.ptr, src->step, size,
                                        src_buffer, (win_w + 2) * sizeof( src_buffer[0] ),
                                        cvSize( win_w + 2, win_h + 2 ), cI ));

            /* calc derivatives */
            icvSepConvSmall3_32f( src_buffer, src_buf_size.width * sizeof(src_buffer[0]),
                                  gx_buffer, win_w * sizeof(gx_buffer[0]),
                                  src_buf_size, drv_x, drv_y, buffer );

            icvSepConvSmall3_32f( src_buffer, src_buf_size.width * sizeof(src_buffer[0]),
                                  gy_buffer, win_w * sizeof(gy_buffer[0]),
                                  src_buf_size, drv_y, drv_x, buffer );

            a = b = c = bb1 = bb2 = 0;

            /* process gradient */
            for( i = 0, k = 0; i < win_h; i++ )
            {
                double py = i - win.height;

                for( j = 0; j < win_w; j++, k++ )
                {
                    double m = mask[k];
                    double tgx = gx_buffer[k];
                    double tgy = gy_buffer[k];
                    double gxx = tgx * tgx * m;
                    double gxy = tgx * tgy * m;
                    double gyy = tgy * tgy * m;
                    double px = j - win.width;

                    a += gxx;
                    b += gxy;
                    c += gyy;

                    bb1 += gxx * px + gxy * py;
                    bb2 += gxy * px + gyy * py;
                }
            }

            {
                double A[4];
                double InvA[4];
                CvMat matA, matInvA;

                A[0] = a;
                A[1] = A[2] = b;
                A[3] = c;

                cvInitMatHeader( &matA, 2, 2, CV_64F, A );
                cvInitMatHeader( &matInvA, 2, 2, CV_64FC1, InvA );

                cvInvert( &matA, &matInvA, CV_SVD );
                cI2.x = (float)(cI.x + InvA[0]*bb1 + InvA[1]*bb2);
                cI2.y = (float)(cI.y + InvA[2]*bb1 + InvA[3]*bb2);
            }

            err = (cI2.x - cI.x) * (cI2.x - cI.x) + (cI2.y - cI.y) * (cI2.y - cI.y);
            cI = cI2;
        }
        while( ++iter < max_iters && err > eps );

        /* if new point is too far from initial, it means poor convergence.
           leave initial point as the result */
        if( fabs( cI.x - cT.x ) > win.width || fabs( cI.y - cT.y ) > win.height )
        {
            cI = cT;
        }

        corners[pt_i] = cI;     /* store result */
    }
}

void cv::cornerSubPix( const InputArray& _image, InputOutputArray _corners,
                       Size winSize, Size zeroZone,
                       TermCriteria criteria )
{
    Mat corners = _corners.getMat();
    int ncorners = corners.checkVector(2);
    CV_Assert( ncorners >= 0 && corners.depth() == CV_32F );
    Mat image = _image.getMat();
    CvMat c_image = image;
    
    cvFindCornerSubPix( &c_image, (CvPoint2D32f*)corners.data, ncorners,
                        winSize, zeroZone, criteria );
}

/* End of file. */
