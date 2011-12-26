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
    CvMat* mat = (CvMat*)arr;
    
    if( !CV_IS_MAT( mat ))
        CV_Error( CV_StsBadArg, "Input array is not a valid matrix" ); 

    eltype = CV_MAT_TYPE( mat->type );
    if( eltype != CV_32SC2 && eltype != CV_32FC2 )
        CV_Error( CV_StsUnsupportedFormat,
        "The matrix can not be converted to point sequence because of "
        "inappropriate element type" );

    if( (mat->width != 1 && mat->height != 1) || !CV_IS_MAT_CONT(mat->type))
        CV_Error( CV_StsBadArg,
        "The matrix converted to point sequence must be "
        "1-dimensional and continuous" );

    cvMakeSeqHeaderForArray(
            (seq_kind & (CV_SEQ_KIND_MASK|CV_SEQ_FLAG_CLOSED)) | eltype,
            sizeof(CvContour), CV_ELEM_SIZE(eltype), mat->data.ptr,
            mat->width*mat->height, (CvSeq*)contour_header, block );

    return (CvSeq*)contour_header;
}

namespace cv
{

static void copyMakeBorder_8u( const uchar* src, size_t srcstep, Size srcroi,
                               uchar* dst, size_t dststep, Size dstroi,
                               int top, int left, int cn, int borderType )
{
    const int isz = (int)sizeof(int);
    int i, j, k, elemSize = 1;
    bool intMode = false;

    if( (cn | srcstep | dststep | (size_t)src | (size_t)dst) % isz == 0 )
    {
        cn /= isz;
        elemSize = isz;
        intMode = true;
    }

    AutoBuffer<int> _tab((dstroi.width - srcroi.width)*cn);
    int* tab = _tab;
    int right = dstroi.width - srcroi.width - left;
    int bottom = dstroi.height - srcroi.height - top;
    
    for( i = 0; i < left; i++ )
    {
        j = borderInterpolate(i - left, srcroi.width, borderType)*cn;
        for( k = 0; k < cn; k++ )
            tab[i*cn + k] = j + k;
    }
    
    for( i = 0; i < right; i++ )
    {
        j = borderInterpolate(srcroi.width + i, srcroi.width, borderType)*cn;
        for( k = 0; k < cn; k++ )
            tab[(i+left)*cn + k] = j + k;
    }

    srcroi.width *= cn;
    dstroi.width *= cn;
    left *= cn;
    right *= cn;
    
    uchar* dstInner = dst + dststep*top + left*elemSize;

    for( i = 0; i < srcroi.height; i++, dstInner += dststep, src += srcstep )
    {
        if( dstInner != src )
            memcpy(dstInner, src, srcroi.width*elemSize);
        
        if( intMode )
        {
            const int* isrc = (int*)src;
            int* idstInner = (int*)dstInner;
            for( j = 0; j < left; j++ )
                idstInner[j - left] = isrc[tab[j]];
            for( j = 0; j < right; j++ )
                idstInner[j + srcroi.width] = isrc[tab[j + left]];
        }
        else
        {
            for( j = 0; j < left; j++ )
                dstInner[j - left] = src[tab[j]];
            for( j = 0; j < right; j++ )
                dstInner[j + srcroi.width] = src[tab[j + left]];
        }
    }
    
    dstroi.width *= elemSize;
    dst += dststep*top;
    
    for( i = 0; i < top; i++ )
    {
        j = borderInterpolate(i - top, srcroi.height, borderType);
        memcpy(dst + (i - top)*dststep, dst + j*dststep, dstroi.width);
    }
    
    for( i = 0; i < bottom; i++ )
    {
        j = borderInterpolate(i + srcroi.height, srcroi.height, borderType);
        memcpy(dst + (i + srcroi.height)*dststep, dst + j*dststep, dstroi.width);
    }
}


static void copyMakeConstBorder_8u( const uchar* src, size_t srcstep, Size srcroi,
                                    uchar* dst, size_t dststep, Size dstroi,
                                    int top, int left, int cn, const uchar* value )
{
    int i, j;
    AutoBuffer<uchar> _constBuf(dstroi.width*cn);
    uchar* constBuf = _constBuf;
    int right = dstroi.width - srcroi.width - left;
    int bottom = dstroi.height - srcroi.height - top;
    
    for( i = 0; i < dstroi.width; i++ )
    {
        for( j = 0; j < cn; j++ )
            constBuf[i*cn + j] = value[j];
    }
    
    srcroi.width *= cn;
    dstroi.width *= cn;
    left *= cn;
    right *= cn;
    
    uchar* dstInner = dst + dststep*top + left;
    
    for( i = 0; i < srcroi.height; i++, dstInner += dststep, src += srcstep )
    {
        if( dstInner != src )
            memcpy( dstInner, src, srcroi.width );
        memcpy( dstInner - left, constBuf, left );
        memcpy( dstInner + srcroi.width, constBuf, right );
    }
    
    dst += dststep*top;
    
    for( i = 0; i < top; i++ )
        memcpy(dst + (i - top)*dststep, constBuf, dstroi.width);
    
    for( i = 0; i < bottom; i++ )
        memcpy(dst + (i + srcroi.height)*dststep, constBuf, dstroi.width);
}

}
    
void cv::copyMakeBorder( InputArray _src, OutputArray _dst, int top, int bottom,
                         int left, int right, int borderType, const Scalar& value )
{
    Mat src = _src.getMat();
    CV_Assert( top >= 0 && bottom >= 0 && left >= 0 && right >= 0 );
    
    _dst.create( src.rows + top + bottom, src.cols + left + right, src.type() );
    Mat dst = _dst.getMat();
    
    if( src.isSubmatrix() && (borderType & BORDER_ISOLATED) == 0 )
    {
        Size wholeSize;
        Point ofs;
        src.locateROI(wholeSize, ofs);
        int dtop = std::min(ofs.y, top);
        int dbottom = std::min(wholeSize.height - src.rows - ofs.y, bottom);
        int dleft = std::min(ofs.x, left);
        int dright = std::min(wholeSize.width - src.cols - ofs.x, right);
        src.adjustROI(dtop, dbottom, dleft, dright);
        top -= dtop;
        left -= dleft;
        bottom -= dbottom;
        right -= dright;
    }
    
    borderType &= ~BORDER_ISOLATED;
    
    if( borderType != BORDER_CONSTANT )
        copyMakeBorder_8u( src.data, src.step, src.size(),
                           dst.data, dst.step, dst.size(),
                           top, left, (int)src.elemSize(), borderType );
    else
    {
        int cn = src.channels(), cn1 = cn;
        AutoBuffer<double> buf(cn);
        if( cn > 4 )
        {
            CV_Assert( value[0] == value[1] && value[0] == value[2] && value[0] == value[3] );
            cn1 = 1;
        }
        scalarToRawData(value, buf, CV_MAKETYPE(src.depth(), cn1), cn);
        copyMakeConstBorder_8u( src.data, src.step, src.size(),
                                dst.data, dst.step, dst.size(),
                                top, left, (int)src.elemSize(), (uchar*)(double*)buf );
    }
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
