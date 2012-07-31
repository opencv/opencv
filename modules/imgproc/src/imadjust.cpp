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

void stretchlimFromHist( const cv::MatND& hist, double* low_value,
                     double* high_value, double low_fract, double high_fract,
                     unsigned int histSum)
{
    CV_Assert( low_fract >= 0 && low_fract < 1.0 );
    CV_Assert( low_fract < high_fract && high_fract <= 1.0);

    unsigned int sum;
    unsigned int low_count = low_fract * histSum;
    sum = 0;
    for( unsigned int i = 0; i < hist.rows; i++ ) {
        if (sum >= low_count) {
            *low_value = i;
            break;
        }

        sum += ((float*)hist.data)[i];
    }

    unsigned int high_count = (1 - high_fract) * histSum;
    sum = 0;
    for( unsigned int i = hist.rows - 1; i >= 0; i-- ) {
        if (sum >= high_count) {
            *high_value = i;
            break;
        }

        sum += ((float*)hist.data)[i];
    }
}

//TODO: surely something like this already exists
int bitsFromDepth( int depth )
{
    if (depth == CV_8U)
        return 8;
    else if (depth == CV_16U)
        return 16;
    else
        return 0;
}

//TODO: handle RGB or force user to do a channel at a time?
void cv::stretchlim( const InputArray _image, double* low_value,
                     double* high_value, double low_fract, double high_fract )
{
    Mat image = _image.getMat();

    CV_Assert( image.type() == CV_8UC1 || image.type() == CV_16UC1 );

    if (low_fract == 0 && high_fract == 1.0) {
        // no need to waste calculating histogram
        *low_value = 0;
        *high_value = 1;
        return;
    }

    int nPixelValues = 1 << bitsFromDepth( image.depth() );
    int channels[] = { 0 };
    MatND hist;
    int histSize[] = { nPixelValues };
    float range[] = { 0, nPixelValues };
    const float* ranges[] = { range };
    calcHist( &image, 1, channels, Mat(), hist, 1, histSize, ranges );
    
    stretchlimFromHist( hist, low_value, high_value, low_fract, high_fract, image.rows * image.cols );

    //TODO: scaling to 0..1 here, but should be in stretchlimFromHist?
    unsigned int maxVal = (1 << bitsFromDepth( _image.depth() )) - 1;
    *low_value /= maxVal;
    *high_value /= maxVal;
}

//TODO: best to determine output depth from _dst or argument?
void cv::imadjust( const InputArray _src, OutputArray _dst, double low_in,
                   double high_in, double low_out, double high_out )
{
    CV_Assert( (low_in == 0 || high_in != low_in) && high_out != low_out );

    //FIXME: use NaN or something else for default values?
    if (low_in == 0 && high_in == 0)
        stretchlim ( _src, &low_in, &high_in );

    double alpha = (high_out - low_out) / (high_in - low_in);
    double beta = high_out - high_in * alpha;

    Mat src = _src.getMat();
    int depth;
    if (_dst.empty())
        depth = _src.depth();
    else
        depth = _dst.depth();

    //TODO: handle more than just 8U/16U
    //adjust alpha/beta to handle to/from different depths
    int max_in = (1 << bitsFromDepth( _src.depth() )) - 1;
    int max_out = (1 << bitsFromDepth( _dst.depth() )) - 1;
    // y = a*x*(outmax/inmax) + b*outmax
    alpha *= max_out / max_in;
    beta *= max_out;

    src.convertTo( _dst, depth, alpha, beta );
}