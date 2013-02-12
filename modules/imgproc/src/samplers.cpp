/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

namespace cv
{

static const uchar*
adjustRect( const uchar* src, int src_step, int pix_size,
           Size src_size, Size win_size,
           Point ip, Rect* pRect )
{
    Rect rect;

    if( ip.x >= 0 )
    {
        src += ip.x*pix_size;
        rect.x = 0;
    }
    else
    {
        rect.x = -ip.x;
        if( rect.x > win_size.width )
            rect.x = win_size.width;
    }

    if( ip.x + win_size.width < src_size.width )
        rect.width = win_size.width;
    else
    {
        rect.width = src_size.width - ip.x - 1;
        if( rect.width < 0 )
        {
            src += rect.width*pix_size;
            rect.width = 0;
        }
        assert( rect.width <= win_size.width );
    }

    if( ip.y >= 0 )
    {
        src += ip.y * src_step;
        rect.y = 0;
    }
    else
        rect.y = -ip.y;

    if( ip.y + win_size.height < src_size.height )
        rect.height = win_size.height;
    else
    {
        rect.height = src_size.height - ip.y - 1;
        if( rect.height < 0 )
        {
            src += rect.height*src_step;
            rect.height = 0;
        }
    }

    *pRect = rect;
    return src - rect.x*pix_size;
}


enum { SUBPIX_SHIFT=16 };

struct scale_fixpt
{
    int operator()(float a) const { return cvRound(a*(1 << SUBPIX_SHIFT)); }
};

struct cast_8u
{
    uchar operator()(int a) const { return (uchar)((a + (1 << (SUBPIX_SHIFT-1))) >> SUBPIX_SHIFT); }
};

struct cast_flt_8u
{
    uchar operator()(float a) const { return (uchar)cvRound(a); }
};

template<typename _Tp>
struct nop
{
    _Tp operator()(_Tp a) const { return a; }
};


template<typename _Tp, typename _DTp, typename _WTp, class ScaleOp, class CastOp>
void getRectSubPix_Cn_(const _Tp* src, int src_step, Size src_size,
                       _DTp* dst, int dst_step, Size win_size, Point2f center, int cn )
{
    ScaleOp scale_op;
    CastOp cast_op;
    Point ip;
    _WTp a11, a12, a21, a22, b1, b2;
    float a, b;
    int i, j, c;

    center.x -= (win_size.width-1)*0.5f;
    center.y -= (win_size.height-1)*0.5f;

    ip.x = cvFloor( center.x );
    ip.y = cvFloor( center.y );

    a = center.x - ip.x;
    b = center.y - ip.y;
    a11 = scale_op((1.f-a)*(1.f-b));
    a12 = scale_op(a*(1.f-b));
    a21 = scale_op((1.f-a)*b);
    a22 = scale_op(a*b);
    b1 = scale_op(1.f - b);
    b2 = scale_op(b);

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);

    if( 0 <= ip.x && ip.x + win_size.width < src_size.width &&
       0 <= ip.y && ip.y + win_size.height < src_size.height )
    {
        // extracted rectangle is totally inside the image
        src += ip.y * src_step + ip.x*cn;
        win_size.width *= cn;

        for( i = 0; i < win_size.height; i++, src += src_step, dst += dst_step )
        {
            for( j = 0; j <= win_size.width - 2; j += 2 )
            {
                _WTp s0 = src[j]*a11 + src[j+cn]*a12 + src[j+src_step]*a21 + src[j+src_step+cn]*a22;
                _WTp s1 = src[j+1]*a11 + src[j+cn+1]*a12 + src[j+src_step+1]*a21 + src[j+src_step+cn+1]*a22;
                dst[j] = cast_op(s0);
                dst[j+1] = cast_op(s1);
            }

            for( j = 0; j < win_size.width; j++ )
            {
                _WTp s0 = src[j]*a11 + src[j+cn]*a12 + src[j+src_step]*a21 + src[j+src_step+cn]*a22;
                dst[j] = cast_op(s0);
            }
        }
    }
    else
    {
        Rect r;
        src = (const _Tp*)adjustRect( (const uchar*)src, src_step*sizeof(*src),
                                     sizeof(*src)*cn, src_size, win_size, ip, &r);

        for( i = 0; i < win_size.height; i++, dst += dst_step )
        {
            const _Tp *src2 = src + src_step;
            _WTp s0;

            if( i < r.y || i >= r.height )
                src2 -= src_step;

            for( c = 0; c < cn; c++ )
            {
                s0 = src[r.x*cn + c]*b1 + src2[r.x*cn + c]*b2;
                for( j = 0; j < r.x; j++ )
                    dst[j*cn + c] = cast_op(s0);
                s0 = src[r.width*cn + c]*b1 + src2[r.width*cn + c]*b2;
                for( j = r.width; j < win_size.width; j++ )
                    dst[j*cn + c] = cast_op(s0);
            }

            for( j = r.x*cn; j < r.width*cn; j++ )
            {
                s0 = src[j]*a11 + src[j+cn]*a12 + src2[j]*a21 + src2[j+cn]*a22;
                dst[j] = cast_op(s0);
            }

            if( i < r.height )
                src = src2;
        }
    }
}


static void getRectSubPix_8u32f
( const uchar* src, int src_step, Size src_size,
 float* dst, int dst_step, Size win_size, Point2f center0, int cn )
{
    Point2f center = center0;
    Point ip;

    center.x -= (win_size.width-1)*0.5f;
    center.y -= (win_size.height-1)*0.5f;

    ip.x = cvFloor( center.x );
    ip.y = cvFloor( center.y );

    if( cn == 1 &&
       0 <= ip.x && ip.x + win_size.width < src_size.width &&
       0 <= ip.y && ip.y + win_size.height < src_size.height &&
       win_size.width > 0 && win_size.height > 0 )
    {
        float a = center.x - ip.x;
        float b = center.y - ip.y;
        a = MAX(a,0.0001f);
        float a12 = a*(1.f-b);
        float a22 = a*b;
        float b1 = 1.f - b;
        float b2 = b;
        double s = (1. - a)/a;

        src_step /= sizeof(src[0]);
        dst_step /= sizeof(dst[0]);

        // extracted rectangle is totally inside the image
        src += ip.y * src_step + ip.x;

        for( ; win_size.height--; src += src_step, dst += dst_step )
        {
            float prev = (1 - a)*(b1*src[0] + b2*src[src_step]);
            for( int j = 0; j < win_size.width; j++ )
            {
                float t = a12*src[j+1] + a22*src[j+1+src_step];
                dst[j] = prev + t;
                prev = (float)(t*s);
            }
        }
    }
    else
    {
        getRectSubPix_Cn_<uchar, float, float, nop<float>, nop<float> >
        (src, src_step, src_size, dst, dst_step, win_size, center0, cn );
    }
}

}

void cv::getRectSubPix( InputArray _image, Size patchSize, Point2f center,
                       OutputArray _patch, int patchType )
{
    Mat image = _image.getMat();
    int depth = image.depth(), cn = image.channels();
    int ddepth = patchType < 0 ? depth : CV_MAT_DEPTH(patchType);
    
    CV_Assert( cn == 1 || cn == 3 );
    
    _patch.create(patchSize, CV_MAKETYPE(ddepth, cn));
    Mat patch = _patch.getMat();
    
    if( depth == CV_8U && ddepth == CV_8U )
        getRectSubPix_Cn_<uchar, uchar, int, scale_fixpt, cast_8u>
        (image.data, image.step, image.size(), patch.data, patch.step, patch.size(), center, cn);
    else if( depth == CV_8U && ddepth == CV_32F )
        getRectSubPix_8u32f
        (image.data, image.step, image.size(), (float*)patch.data, patch.step, patch.size(), center, cn);
    else if( depth == CV_32F && ddepth == CV_32F )
        getRectSubPix_Cn_<float, float, float, nop<float>, nop<float> >
        ((const float*)image.data, image.step, image.size(), (float*)patch.data, patch.step, patch.size(), center, cn);
    else
        CV_Error( CV_StsUnsupportedFormat, "Unsupported combination of input and output formats");
}


CV_IMPL void
cvGetRectSubPix( const void* srcarr, void* dstarr, CvPoint2D32f center )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr);
    CV_Assert( src.channels() == dst.channels() );

    cv::getRectSubPix(src, dst.size(), center, dst, dst.type());
}


CV_IMPL void
cvGetQuadrangleSubPix( const void* srcarr, void* dstarr, const CvMat* mat )
{
    cv::Mat src = cv::cvarrToMat(srcarr), dst = cv::cvarrToMat(dstarr), m = cv::cvarrToMat(mat);
    cv::Size win_size = dst.size();
    double matrix[6];
    cv::Mat M(2, 3, CV_64F, matrix);
    m.convertTo(M, CV_64F);
    double dx = (win_size.width - 1)*0.5;
    double dy = (win_size.height - 1)*0.5;
    matrix[2] -= matrix[0]*dx + matrix[1]*dy;
    matrix[5] -= matrix[3]*dx + matrix[4]*dy;

    cv::warpAffine(src, dst, M, dst.size(),
                   cv::INTER_LINEAR + cv::WARP_INVERSE_MAP,
                   cv::BORDER_REPLICATE);
}


CV_IMPL int
cvSampleLine( const void* _img, CvPoint pt1, CvPoint pt2,
              void* _buffer, int connectivity )
{
    cv::Mat img = cv::cvarrToMat(_img);
    cv::LineIterator li(img, pt1, pt2, connectivity, false);
    uchar* buffer = (uchar*)_buffer;
    size_t pixsize = img.elemSize();
    
    if( !buffer )
        CV_Error( CV_StsNullPtr, "" );

    for( int i = 0; i < li.count; i++, ++li )
    {
        for( size_t k = 0; k < pixsize; k++ )
            *buffer++ = li.ptr[k];
    }

    return li.count;
}


/* End of file. */
