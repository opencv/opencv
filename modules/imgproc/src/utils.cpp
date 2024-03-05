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

CV_IMPL CvSeq* cvPointSeqFromMat( int seq_kind, const CvArr* arr,
                                  CvContour* contour_header, CvSeqBlock* block )
{
    CV_Assert( arr != 0 && contour_header != 0 && block != 0 );

    int eltype;
    CvMat hdr;
    CvMat* mat = (CvMat*)arr;

    if( !CV_IS_MAT( mat ))
        CV_Error( cv::Error::StsBadArg, "Input array is not a valid matrix" );

    if( CV_MAT_CN(mat->type) == 1 && mat->width == 2 )
        mat = cvReshape(mat, &hdr, 2);

    eltype = CV_MAT_TYPE( mat->type );
    if( eltype != CV_32SC2 && eltype != CV_32FC2 )
        CV_Error( cv::Error::StsUnsupportedFormat,
        "The matrix can not be converted to point sequence because of "
        "inappropriate element type" );

    if( (mat->width != 1 && mat->height != 1) || !CV_IS_MAT_CONT(mat->type))
        CV_Error( cv::Error::StsBadArg,
        "The matrix converted to point sequence must be "
        "1-dimensional and continuous" );

    cvMakeSeqHeaderForArray(
            (seq_kind & (CV_SEQ_KIND_MASK|CV_SEQ_FLAG_CLOSED)) | eltype,
            sizeof(CvContour), CV_ELEM_SIZE(eltype), mat->data.ptr,
            mat->width*mat->height, (CvSeq*)contour_header, block );

    return (CvSeq*)contour_header;
}

CV_IMPL void
cvCopyMakeBorder( const CvArr* srcarr, CvArr* dstarr, CvPoint offset,
                  int borderType, CvScalar value )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    int left = offset.x, right = dst.cols - src.cols - left;
    int top = offset.y, bottom = dst.rows - src.rows - top;

    CV_Assert( dst.type() == src.type() );
    cv::copyMakeBorder( src, dst, top, bottom, left, right, borderType, value );
}

/* End of file. */
