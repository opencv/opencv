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


typedef CvStatus (CV_STDCALL * CvCopyNonConstBorderFunc)(
    const void*, int, CvSize, void*, int, CvSize, int, int );

typedef CvStatus (CV_STDCALL * CvCopyNonConstBorderFuncI)(
    const void*, int, CvSize, CvSize, int, int );

CvStatus CV_STDCALL
icvCopyReplicateBorder_8u( const uchar* src, int srcstep, CvSize srcroi,
                           uchar* dst, int dststep, CvSize dstroi,
                           int top, int left, int cn, const uchar* )
{
    const int isz = (int)sizeof(int);
    int i, j;

    if( (cn | srcstep | dststep | (size_t)src | (size_t)dst) % isz == 0 )
    {
        const int* isrc = (const int*)src;
        int* idst = (int*)dst;
        
        cn /= isz;
        srcstep /= isz;
        dststep /= isz;

        srcroi.width *= cn;
        dstroi.width *= cn;
        left *= cn;

        for( i = 0; i < dstroi.height; i++, idst += dststep )
        {
            if( idst + left != isrc )
                memcpy( idst + left, isrc, srcroi.width*sizeof(idst[0]) );
            for( j = left - 1; j >= 0; j-- )
                idst[j] = idst[j + cn];
            for( j = left+srcroi.width; j < dstroi.width; j++ )
                idst[j] = idst[j - cn];
            if( i >= top && i < top + srcroi.height - 1 )
                isrc += srcstep;
        }
    }
    else
    {
        srcroi.width *= cn;
        dstroi.width *= cn;
        left *= cn;

        for( i = 0; i < dstroi.height; i++, dst += dststep )
        {
            if( dst + left != src )
                memcpy( dst + left, src, srcroi.width );
            for( j = left - 1; j >= 0; j-- )
                dst[j] = dst[j + cn];
            for( j = left+srcroi.width; j < dstroi.width; j++ )
                dst[j] = dst[j - cn];
            if( i >= top && i < top + srcroi.height - 1 )
                src += srcstep;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvCopyReflect101Border_8u( const uchar* src, int srcstep, CvSize srcroi,
                            uchar* dst, int dststep, CvSize dstroi,
                            int top, int left, int cn )
{
    const int isz = (int)sizeof(int);
    int i, j, k, t, dj, tab_size, int_mode = 0;
    const int* isrc = (const int*)src;
    int* idst = (int*)dst, *tab;

    if( (cn | srcstep | dststep | (size_t)src | (size_t)dst) % isz == 0 )
    {
        cn /= isz;
        srcstep /= isz;
        dststep /= isz;

        int_mode = 1;
    }

    srcroi.width *= cn;
    dstroi.width *= cn;
    left *= cn;

    tab_size = dstroi.width - srcroi.width;
    tab = (int*)cvStackAlloc( tab_size*sizeof(tab[0]) );

    if( srcroi.width == 1 )
    {
        for( k = 0; k < cn; k++ )
            for( i = 0; i < tab_size; i += cn )
                tab[i + k] = k + left;
    }
    else
    {
        j = dj = cn;
        for( i = left - cn; i >= 0; i -= cn )
        {
            for( k = 0; k < cn; k++ )
                tab[i + k] = j + k + left;
            if( (unsigned)(j += dj) >= (unsigned)srcroi.width )
                j -= 2*dj, dj = -dj;
        }
        
        j = srcroi.width - cn*2;
        dj = -cn;
        for( i = left; i < tab_size; i += cn )
        {
            for( k = 0; k < cn; k++ )
                tab[i + k] = j + k + left;
            if( (unsigned)(j += dj) >= (unsigned)srcroi.width )
                j -= 2*dj, dj = -dj;
        }
    }

    if( int_mode )
    {
        idst += top*dststep;
        for( i = 0; i < srcroi.height; i++, isrc += srcstep, idst += dststep )
        {
            if( idst + left != isrc )
                memcpy( idst + left, isrc, srcroi.width*sizeof(idst[0]) );
            for( j = 0; j < left; j++ )
            {
                k = tab[j]; 
                idst[j] = idst[k];
            }
            for( ; j < tab_size; j++ )
            {
                k = tab[j];
                idst[j + srcroi.width] = idst[k];
            }
        }
        isrc -= srcroi.height*srcstep;
        idst -= (top + srcroi.height)*dststep;
    }
    else
    {
        dst += top*dststep;
        for( i = 0; i < srcroi.height; i++, src += srcstep, dst += dststep )
        {
            if( dst + left != src )
                memcpy( dst + left, src, srcroi.width );
            for( j = 0; j < left; j++ )
            {
                k = tab[j]; 
                dst[j] = dst[k];
            }
            for( ; j < tab_size; j++ )
            {
                k = tab[j];
                dst[j + srcroi.width] = dst[k];
            }
        }
        src -= srcroi.height*srcstep;
        dst -= (top + srcroi.height)*dststep;
    }

    for( t = 0; t < 2; t++ )
    {
        int i1, i2, di;
        if( t == 0 )
            i1 = top-1, i2 = 0, di = -1, j = 1, dj = 1;
        else
            i1 = top+srcroi.height, i2=dstroi.height, di = 1, j = srcroi.height-2, dj = -1;
        
        for( i = i1; (di > 0 && i < i2) || (di < 0 && i > i2); i += di )
        {
            if( int_mode )
            {
                const int* s = idst + i*dststep;
                int* d = idst + (j+top)*dststep;
                for( k = 0; k < dstroi.width; k++ )
                    d[k] = s[k];
            }
            else
            {
                const uchar* s = dst + i*dststep;
                uchar* d = dst + (j+top)*dststep;
                for( k = 0; k < dstroi.width; k++ )
                    d[k] = s[k];
            }

            if( (unsigned)(j += dj) >= (unsigned)srcroi.height )
                j -= 2*dj, dj = -dj;
        }
    }

    return CV_OK;
}


static CvStatus CV_STDCALL
icvCopyConstBorder_8u( const uchar* src, int srcstep, CvSize srcroi,
                       uchar* dst, int dststep, CvSize dstroi,
                       int top, int left, int cn, const uchar* value )
{
    const int isz = (int)sizeof(int);
    int i, j, k;
    if( (cn | srcstep | dststep | (size_t)src | (size_t)dst | (size_t)value) % isz == 0 )
    {
        const int* isrc = (const int*)src;
        int* idst = (int*)dst;
        const int* ivalue = (const int*)value;
        int v0 = ivalue[0];
        
        cn /= isz;
        srcstep /= isz;
        dststep /= isz;

        srcroi.width *= cn;
        dstroi.width *= cn;
        left *= cn;

        for( j = 1; j < cn; j++ )
            if( ivalue[j] != ivalue[0] )
                break;

        if( j == cn )
            cn = 1;

        if( dstroi.width <= 0 )
            return CV_OK;

        for( i = 0; i < dstroi.height; i++, idst += dststep )
        {
            if( i < top || i >= top + srcroi.height )
            {
                if( cn == 1 )
                {
                    for( j = 0; j < dstroi.width; j++ )
                        idst[j] = v0;
                }
                else
                {
                    for( j = 0; j < cn; j++ )
                        idst[j] = ivalue[j];
                    for( ; j < dstroi.width; j++ )
                        idst[j] = idst[j - cn];
                }
                continue;
            }

            if( cn == 1 )
            {
                for( j = 0; j < left; j++ )
                    idst[j] = v0;
                for( j = srcroi.width + left; j < dstroi.width; j++ )
                    idst[j] = v0;
            }
            else
            {
                for( k = 0; k < cn; k++ )
                {
                    for( j = 0; j < left; j += cn )
                        idst[j+k] = ivalue[k];
                    for( j = srcroi.width + left; j < dstroi.width; j += cn )
                        idst[j+k] = ivalue[k];
                }
            }

            if( idst + left != isrc )
                for( j = 0; j < srcroi.width; j++ )
                    idst[j + left] = isrc[j];
            isrc += srcstep;
        }
    }
    else
    {
        uchar v0 = value[0];
        
        srcroi.width *= cn;
        dstroi.width *= cn;
        left *= cn;

        for( j = 1; j < cn; j++ )
            if( value[j] != value[0] )
                break;

        if( j == cn )
            cn = 1;

        if( dstroi.width <= 0 )
            return CV_OK;

        for( i = 0; i < dstroi.height; i++, dst += dststep )
        {
            if( i < top || i >= top + srcroi.height )
            {
                if( cn == 1 )
                {
                    for( j = 0; j < dstroi.width; j++ )
                        dst[j] = v0;
                }
                else
                {
                    for( j = 0; j < cn; j++ )
                        dst[j] = value[j];
                    for( ; j < dstroi.width; j++ )
                        dst[j] = dst[j - cn];
                }
                continue;
            }

            if( cn == 1 )
            {
                for( j = 0; j < left; j++ )
                    dst[j] = v0;
                for( j = srcroi.width + left; j < dstroi.width; j++ )
                    dst[j] = v0;
            }
            else
            {
                for( k = 0; k < cn; k++ )
                {
                    for( j = 0; j < left; j += cn )
                        dst[j+k] = value[k];
                    for( j = srcroi.width + left; j < dstroi.width; j += cn )
                        dst[j+k] = value[k];
                }
            }

            if( dst + left != src )
                for( j = 0; j < srcroi.width; j++ )
                    dst[j + left] = src[j];
            src += srcstep;
        }
    }

    return CV_OK;
}


CV_IMPL void
cvCopyMakeBorder( const CvArr* srcarr, CvArr* dstarr, CvPoint offset,
                  int bordertype, CvScalar value )
{
    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvSize srcsize, dstsize;
    int srcstep, dststep;
    int pix_size, type;

    if( !CV_IS_MAT(src) )
        src = cvGetMat( src, &srcstub );
    
    if( !CV_IS_MAT(dst) )    
        dst = cvGetMat( dst, &dststub );

    if( offset.x < 0 || offset.y < 0 )
        CV_Error( CV_StsOutOfRange, "Offset (left/top border width) is negative" );

    if( src->rows + offset.y > dst->rows || src->cols + offset.x > dst->cols )
        CV_Error( CV_StsBadSize, "Source array is too big or destination array is too small" );

    if( !CV_ARE_TYPES_EQ( src, dst ))
        CV_Error( CV_StsUnmatchedFormats, "" );

    type = CV_MAT_TYPE(src->type);
    pix_size = CV_ELEM_SIZE(type);
    srcsize = cvGetMatSize(src);
    dstsize = cvGetMatSize(dst);
    srcstep = src->step;
    dststep = dst->step;
    if( srcstep == 0 )
        srcstep = CV_STUB_STEP;
    if( dststep == 0 )
        dststep = CV_STUB_STEP;

    bordertype &= 15;
    if( bordertype == IPL_BORDER_REPLICATE )
    {
        icvCopyReplicateBorder_8u( src->data.ptr, srcstep, srcsize,
                                   dst->data.ptr, dststep, dstsize,
                                   offset.y, offset.x, pix_size );
    }
    else if( bordertype == IPL_BORDER_REFLECT_101 )
    {
        icvCopyReflect101Border_8u( src->data.ptr, srcstep, srcsize,
                                   dst->data.ptr, dststep, dstsize,
                                   offset.y, offset.x, pix_size );
    }
    else if( bordertype == IPL_BORDER_CONSTANT )
    {
        double buf[4];
        cvScalarToRawData( &value, buf, src->type, 0 );
        icvCopyConstBorder_8u( src->data.ptr, srcstep, srcsize,
                               dst->data.ptr, dststep, dstsize,
                               offset.y, offset.x, pix_size, (uchar*)buf );
    }
    else
        CV_Error( CV_StsBadFlag, "Unknown/unsupported border type" );
}

namespace cv
{

void copyMakeBorder( const Mat& src, Mat& dst, int top, int bottom,
                     int left, int right, int borderType, const Scalar& value )
{
    dst.create( src.rows + top + bottom, src.cols + left + right, src.type() );
    CvMat _src = src, _dst = dst;
    cvCopyMakeBorder( &_src, &_dst, Point(left, top), borderType, value );
}

}

/* End of file. */
