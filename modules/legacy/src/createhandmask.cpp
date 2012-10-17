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

#define CV_MAX2( a, b ) ((a)>(b) ? (a) : (b))
#define CV_MIN2( a, b ) ((a)<(b) ? (a) : (b))

/****************************************************************************************\

   create hand mask

\****************************************************************************************/

static CvStatus icvCreateHandMask8uC1R(CvSeq * numbers,
                                       uchar * image_mask, int step,
                                       CvSize size, CvRect * roi )
{

    CvSeqReader reader;
    CvPoint pt;
    int k_point;
    int i_min, i_max, j_min, j_max;

    if( numbers == NULL )
        return CV_NULLPTR_ERR;

    if( !CV_IS_SEQ_POINT_SET( numbers ))
        return CV_BADFLAG_ERR;

    i_max = j_max = 0;
    i_min = size.height;
    j_min = size.width;

    cvStartReadSeq( numbers, &reader, 0 );

    k_point = numbers->total;
    assert( k_point > 0 );
    if( k_point <= 0 )
        return CV_BADSIZE_ERR;

    memset( image_mask, 0, step * size.height );

    while( k_point-- > 0 )
    {
        CV_READ_SEQ_ELEM( pt, reader );

        i_min = CV_MIN2( i_min, pt.y );
        i_max = CV_MAX2( i_max, pt.y );
        j_min = CV_MIN2( j_min, pt.x );
        j_max = CV_MAX2( j_max, pt.x );

        *(image_mask + pt.y * step + pt.x) = 255;
    }

    roi->x = j_min;
    roi->y = i_min;
    roi->width = j_max - j_min + 1;
    roi->height = i_max - i_min + 1;

    return CV_OK;

}


/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:     cvCreateHandMask
//    Purpose:  creates hand mask image
//    Context:
//    Parameters:
//      numbers - pointer to the input sequence of the point's indexes inside
//                hand region
//      img_mask - pointer to the result mask image
//      roi      - result hand mask ROI
//
//    Notes:
//F*/
CV_IMPL void
cvCreateHandMask( CvSeq * numbers, IplImage * img_mask, CvRect * roi )
{
    uchar *img_mask_data = 0;
    int img_mask_step = 0;
    CvSize img_mask_size;

    CV_FUNCNAME( "cvCreateHandMask" );

    __BEGIN__;

    if( img_mask->depth != IPL_DEPTH_8U )
        CV_ERROR( CV_BadDepth, cvUnsupportedFormat );

    if( img_mask->nChannels != 1 )
        CV_ERROR( CV_BadNumChannels, "output image have wrong number of channels" );

    cvGetImageRawData( img_mask, &img_mask_data, &img_mask_step, &img_mask_size );

    IPPI_CALL( icvCreateHandMask8uC1R( numbers, img_mask_data,
                                        img_mask_step, img_mask_size, roi ));

    __END__;
}
