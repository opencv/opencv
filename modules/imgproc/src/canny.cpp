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

void cv::Canny( InputArray _src, OutputArray _dst,
                double low_thresh, double high_thresh,
                int aperture_size, bool L2gradient )
{
    Mat src = _src.getMat();
    CV_Assert( src.depth() == CV_8U );
    
    _dst.create(src.size(), CV_8U);
    Mat dst = _dst.getMat();
#ifdef HAVE_TEGRA_OPTIMIZATION
    if (tegra::canny(src, dst, low_thresh, high_thresh, aperture_size, L2gradient))
        return;
#endif
    
    if( low_thresh > high_thresh )
        std::swap(low_thresh, high_thresh);

    if( (aperture_size & 1) == 0 || (aperture_size != -1 && (aperture_size < 3 || aperture_size > 7)) )
        CV_Error( CV_StsBadFlag, "" );

    Mat dx, dy;
    Sobel(src, dx, CV_16S, 1, 0, aperture_size, 1, 0, BORDER_REFLECT_101);
    Sobel(src, dy, CV_16S, 0, 1, aperture_size, 1, 0, BORDER_REFLECT_101);

    int low, high;
    if( L2gradient )
    {
        Cv32suf ul, uh;
        ul.f = (float)low_thresh;
        uh.f = (float)high_thresh;

        low = ul.i;
        high = uh.i;
    }
    else
    {
        low = cvFloor( low_thresh );
        high = cvFloor( high_thresh );
    }

    Size size = src.size();
    int i, j, k, mstep = size.width + 2, cn = src.channels();
    
    Mat mask(size.height + 2, mstep, CV_8U);
    memset( mask.ptr<uchar>(0), 1, mstep );
    memset( mask.ptr<uchar>(size.height+1), 1, mstep );
    
    Mat mag(6+cn, mstep, CV_32S);
    mag = Scalar::all(0);
    int* mag_buf[3] = { mag.ptr<int>(0), mag.ptr<int>(1), mag.ptr<int>(2) };
    short* dxybuf[3] = { (short*)mag.ptr<int>(3), (short*)mag.ptr<int>(4), (short*)mag.ptr<int>(5) }; 
    int* mbuf = mag.ptr<int>(6);
    
    int maxsize = MAX( 1 << 10, size.width*size.height/10 );
    std::vector<uchar*> stack( maxsize );
    uchar **stack_top, **stack_bottom;
    stack_top = stack_bottom = &stack[0];

    /* sector numbers 
       (Top-Left Origin)

        1   2   3
         *  *  * 
          * * *  
        0*******0
          * * *  
         *  *  * 
        3   2   1
    */

    #define CANNY_PUSH(d)    *(d) = (uchar)2, *stack_top++ = (d)
    #define CANNY_POP(d)     (d) = *--stack_top

    // calculate magnitude and angle of gradient, perform non-maxima supression.
    // fill the map with one of the following values:
    //   0 - the pixel might belong to an edge
    //   1 - the pixel can not belong to an edge
    //   2 - the pixel does belong to an edge
    for( i = 0; i <= size.height; i++ )
    {
        int *_mag = mag_buf[(i > 0) + 1] + 1;
        float* _magf = (float*)_mag;
        const short *_dx, *_dy;
        short *_ddx, *_ddy;
        uchar* _map;
        int x, y;
        ptrdiff_t magstep1, magstep2;
        int prev_flag = 0;

        if( i < size.height )
        {
            _dx = dx.ptr<short>(i);
            _dy = dy.ptr<short>(i);
            _ddx = dxybuf[(i > 0) + 1];
            _ddy = _ddx + size.width;
            
            if( cn > 1 )
            {
                _mag = mbuf;
                _magf = (float*)_mag;
            }
            
            if( !L2gradient )
                for( j = 0; j < size.width*cn; j++ )
                    _mag[j] = std::abs(_dx[j]) + std::abs(_dy[j]);
            else
            {
                for( j = 0; j < size.width*cn; j++ )
                {
                    x = _dx[j]; y = _dy[j];
                    _magf[j] = sqrtf((float)x*x + (float)y*y);
                }
            }
            
            if( cn > 1 )
            {
                _mag = mag_buf[(i > 0) + 1] + 1;
                for( j = 0; j < size.width; j++ )
                {
                    _mag[j] = mbuf[(j+1)*cn];
                    _ddx[j] = _dx[j*cn]; _ddy[j] = _dy[j*cn];
                }
                
                for( k = 1; k < cn; k++ )
                {
                    for( j = 0; j < size.width; j++ )
                        if( mbuf[(j+1)*cn + k] > _mag[j] )
                        {
                            _mag[j] = mbuf[(j+1)*cn + k];
                            _ddx[j] = _dx[j*cn + k];
                            _ddy[j] = _dy[j*cn + k];
                        }
                }
            }
            else
            {
                for( j = 0; j < size.width; j++ )
                    _ddx[j] = _dx[j]; _ddy[j] = _dy[j];
            }
            
            _mag[-1] = _mag[size.width] = 0;
        }
        else
            memset( _mag-1, 0, (size.width + 2)*sizeof(_mag[0]) );

        // at the very beginning we do not have a complete ring
        // buffer of 3 magnitude rows for non-maxima suppression
        if( i == 0 )
            continue;

        _map = &mask.at<uchar>(i, 1);
        _map[-1] = _map[size.width] = 1;
        
        _mag = mag_buf[1] + 1; // take the central row
        _dx = dxybuf[1];
        _dy = _dx + size.width;
        
        magstep1 = mag_buf[2] - mag_buf[1];
        magstep2 = mag_buf[0] - mag_buf[1];

        if( (stack_top - stack_bottom) + size.width > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + size.width );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        for( j = 0; j < size.width; j++ )
        {
            #define CANNY_SHIFT 15
            #define TG22  (int)(0.4142135623730950488016887242097*(1<<CANNY_SHIFT) + 0.5)

            x = _dx[j];
            y = _dy[j];
            int s = x ^ y;
            int m = _mag[j];

            x = std::abs(x);
            y = std::abs(y);
            if( m > low )
            {
                int tg22x = x * TG22;
                int tg67x = tg22x + ((x + x) << CANNY_SHIFT);

                y <<= CANNY_SHIFT;

                if( y < tg22x )
                {
                    if( m > _mag[j-1] && m >= _mag[j+1] )
                    {
                        if( m > high && !prev_flag && _map[j-mstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else if( y > tg67x )
                {
                    if( m > _mag[j+magstep2] && m >= _mag[j+magstep1] )
                    {
                        if( m > high && !prev_flag && _map[j-mstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
                else
                {
                    s = s < 0 ? -1 : 1;
                    if( m > _mag[j+magstep2-s] && m > _mag[j+magstep1+s] )
                    {
                        if( m > high && !prev_flag && _map[j-mstep] != 2 )
                        {
                            CANNY_PUSH( _map + j );
                            prev_flag = 1;
                        }
                        else
                            _map[j] = (uchar)0;
                        continue;
                    }
                }
            }
            prev_flag = 0;
            _map[j] = (uchar)1;
        }

        // scroll the ring buffers
        _mag = mag_buf[0];
        mag_buf[0] = mag_buf[1];
        mag_buf[1] = mag_buf[2];
        mag_buf[2] = _mag;
                    
        _ddx = dxybuf[0];
        dxybuf[0] = dxybuf[1];
        dxybuf[1] = dxybuf[2];
        dxybuf[2] = _ddx;
    }

    // now track the edges (hysteresis thresholding)
    while( stack_top > stack_bottom )
    {
        uchar* m;
        if( (stack_top - stack_bottom) + 8 > maxsize )
        {
            int sz = (int)(stack_top - stack_bottom);
            maxsize = MAX( maxsize * 3/2, maxsize + 8 );
            stack.resize(maxsize);
            stack_bottom = &stack[0];
            stack_top = stack_bottom + sz;
        }

        CANNY_POP(m);
    
        if( !m[-1] )
            CANNY_PUSH( m - 1 );
        if( !m[1] )
            CANNY_PUSH( m + 1 );
        if( !m[-mstep-1] )
            CANNY_PUSH( m - mstep - 1 );
        if( !m[-mstep] )
            CANNY_PUSH( m - mstep );
        if( !m[-mstep+1] )
            CANNY_PUSH( m - mstep + 1 );
        if( !m[mstep-1] )
            CANNY_PUSH( m + mstep - 1 );
        if( !m[mstep] )
            CANNY_PUSH( m + mstep );
        if( !m[mstep+1] )
            CANNY_PUSH( m + mstep + 1 );
    }

    // the final pass, form the final image
    for( i = 0; i < size.height; i++ )
    {
        const uchar* _map = mask.ptr<uchar>(i+1) + 1;
        uchar* _dst = dst.ptr<uchar>(i);
        
        for( j = 0; j < size.width; j++ )
            _dst[j] = (uchar)-(_map[j] >> 1);
    }
}

void cvCanny( const CvArr* image, CvArr* edges, double threshold1,
              double threshold2, int aperture_size )
{
    cv::Mat src = cv::cvarrToMat(image), dst = cv::cvarrToMat(edges);
    CV_Assert( src.size == dst.size && src.depth() == CV_8U && dst.type() == CV_8U );
    
    cv::Canny(src, dst, threshold1, threshold2, aperture_size & 255,
              (aperture_size & CV_CANNY_L2_GRADIENT) != 0);
}

/* End of file. */
