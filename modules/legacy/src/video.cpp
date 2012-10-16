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
cvDeInterlace( const CvArr* framearr, CvArr* fieldEven, CvArr* fieldOdd )
{
    CV_FUNCNAME("cvDeInterlace");

    __BEGIN__;

    CvMat frame_stub, *frame = (CvMat*)framearr;
    CvMat even_stub, *even = (CvMat*)fieldEven;
    CvMat odd_stub, *odd = (CvMat*)fieldOdd;
    CvSize size;
    int y;

    CV_CALL( frame = cvGetMat( frame, &frame_stub ));
    CV_CALL( even = cvGetMat( even, &even_stub ));
    CV_CALL( odd = cvGetMat( odd, &odd_stub ));

    if( !CV_ARE_TYPES_EQ( frame, even ) || !CV_ARE_TYPES_EQ( frame, odd ))
        CV_ERROR( CV_StsUnmatchedFormats, "All the input images must have the same type" );

    if( frame->cols != even->cols || frame->cols != odd->cols ||
        frame->rows != even->rows*2 || odd->rows != even->rows )
        CV_ERROR( CV_StsUnmatchedSizes, "Uncorrelated sizes of the input image and output fields" );

    size = cvGetMatSize( even );
    size.width *= CV_ELEM_SIZE( even->type );

    for( y = 0; y < size.height; y++ )
    {
        memcpy( even->data.ptr + even->step*y,
                frame->data.ptr + frame->step*y*2, size.width );
        memcpy( odd->data.ptr + even->step*y,
                frame->data.ptr + frame->step*(y*2+1), size.width );
    }

    __END__;
}

/* End of file. */


