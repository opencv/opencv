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

/**************************************************************************************\
*                                   line samplers                                      *
\**************************************************************************************/

CV_IMPL int
cvSampleLine( const void* img, CvPoint pt1, CvPoint pt2,
              void* _buffer, int connectivity )
{
    int count = -1;
    
    int i, coi = 0, pix_size;
    CvMat stub, *mat = cvGetMat( img, &stub, &coi );
    CvLineIterator iterator;
    uchar* buffer = (uchar*)_buffer;

    if( coi != 0 )
        CV_Error( CV_BadCOI, "" );

    if( !buffer )
        CV_Error( CV_StsNullPtr, "" );

    count = cvInitLineIterator( mat, pt1, pt2, &iterator, connectivity );

    pix_size = CV_ELEM_SIZE(mat->type);
    for( i = 0; i < count; i++ )
    {
        for( int j = 0; j < pix_size; j++ )
            buffer[j] = iterator.ptr[j];
        buffer += pix_size;
        CV_NEXT_LINE_POINT( iterator );
    }

    return count;
}


static const void*
icvAdjustRect( const void* srcptr, int src_step, int pix_size,
               CvSize src_size, CvSize win_size,
               CvPoint ip, CvRect* pRect )
{
    CvRect rect;
    const char* src = (const char*)srcptr;

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


#define  ICV_DEF_GET_RECT_SUB_PIX_FUNC( flavor, srctype, dsttype, worktype, \
                                        cast_macro, scale_macro, cast_macro2 )\
CvStatus CV_STDCALL icvGetRectSubPix_##flavor##_C1R                         \
( const srctype* src, int src_step, CvSize src_size,                        \
  dsttype* dst, int dst_step, CvSize win_size, CvPoint2D32f center )        \
{                                                                           \
    CvPoint ip;                                                             \
    worktype  a11, a12, a21, a22, b1, b2;                                   \
    float a, b;                                                             \
    int i, j;                                                               \
                                                                            \
    center.x -= (win_size.width-1)*0.5f;                                    \
    center.y -= (win_size.height-1)*0.5f;                                   \
                                                                            \
    ip.x = cvFloor( center.x );                                             \
    ip.y = cvFloor( center.y );                                             \
                                                                            \
    a = center.x - ip.x;                                                    \
    b = center.y - ip.y;                                                    \
    a11 = scale_macro((1.f-a)*(1.f-b));                                     \
    a12 = scale_macro(a*(1.f-b));                                           \
    a21 = scale_macro((1.f-a)*b);                                           \
    a22 = scale_macro(a*b);                                                 \
    b1 = scale_macro(1.f - b);                                              \
    b2 = scale_macro(b);                                                    \
                                                                            \
    src_step /= sizeof(src[0]);                                             \
    dst_step /= sizeof(dst[0]);                                             \
                                                                            \
    if( 0 <= ip.x && ip.x + win_size.width < src_size.width &&              \
        0 <= ip.y && ip.y + win_size.height < src_size.height )             \
    {                                                                       \
        /* extracted rectangle is totally inside the image */               \
        src += ip.y * src_step + ip.x;                                      \
                                                                            \
        for( i = 0; i < win_size.height; i++, src += src_step,              \
                                              dst += dst_step )             \
        {                                                                   \
            for( j = 0; j <= win_size.width - 2; j += 2 )                   \
            {                                                               \
                worktype s0 = cast_macro(src[j])*a11 +                      \
                              cast_macro(src[j+1])*a12 +                    \
                              cast_macro(src[j+src_step])*a21 +             \
                              cast_macro(src[j+src_step+1])*a22;            \
                worktype s1 = cast_macro(src[j+1])*a11 +                    \
                              cast_macro(src[j+2])*a12 +                    \
                              cast_macro(src[j+src_step+1])*a21 +           \
                              cast_macro(src[j+src_step+2])*a22;            \
                                                                            \
                dst[j] = (dsttype)cast_macro2(s0);                          \
                dst[j+1] = (dsttype)cast_macro2(s1);                        \
            }                                                               \
                                                                            \
            for( ; j < win_size.width; j++ )                                \
            {                                                               \
                worktype s0 = cast_macro(src[j])*a11 +                      \
                              cast_macro(src[j+1])*a12 +                    \
                              cast_macro(src[j+src_step])*a21 +             \
                              cast_macro(src[j+src_step+1])*a22;            \
                                                                            \
                dst[j] = (dsttype)cast_macro2(s0);                          \
            }                                                               \
        }                                                                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        CvRect r;                                                           \
                                                                            \
        src = (const srctype*)icvAdjustRect( src, src_step*sizeof(*src),    \
                               sizeof(*src), src_size, win_size,ip, &r);    \
                                                                            \
        for( i = 0; i < win_size.height; i++, dst += dst_step )             \
        {                                                                   \
            const srctype *src2 = src + src_step;                           \
                                                                            \
            if( i < r.y || i >= r.height )                                  \
                src2 -= src_step;                                           \
                                                                            \
            for( j = 0; j < r.x; j++ )                                      \
            {                                                               \
                worktype s0 = cast_macro(src[r.x])*b1 +                     \
                              cast_macro(src2[r.x])*b2;                     \
                                                                            \
                dst[j] = (dsttype)cast_macro2(s0);                          \
            }                                                               \
                                                                            \
            for( ; j < r.width; j++ )                                       \
            {                                                               \
                worktype s0 = cast_macro(src[j])*a11 +                      \
                              cast_macro(src[j+1])*a12 +                    \
                              cast_macro(src2[j])*a21 +                     \
                              cast_macro(src2[j+1])*a22;                    \
                                                                            \
                dst[j] = (dsttype)cast_macro2(s0);                          \
            }                                                               \
                                                                            \
            for( ; j < win_size.width; j++ )                                \
            {                                                               \
                worktype s0 = cast_macro(src[r.width])*b1 +                 \
                              cast_macro(src2[r.width])*b2;                 \
                                                                            \
                dst[j] = (dsttype)cast_macro2(s0);                          \
            }                                                               \
                                                                            \
            if( i < r.height )                                              \
                src = src2;                                                 \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}


#define  ICV_DEF_GET_RECT_SUB_PIX_FUNC_C3( flavor, srctype, dsttype, worktype, \
                                        cast_macro, scale_macro, mul_macro )\
static CvStatus CV_STDCALL icvGetRectSubPix_##flavor##_C3R                  \
( const srctype* src, int src_step, CvSize src_size,                        \
  dsttype* dst, int dst_step, CvSize win_size, CvPoint2D32f center )        \
{                                                                           \
    CvPoint ip;                                                             \
    worktype a, b;                                                          \
    int i, j;                                                               \
                                                                            \
    center.x -= (win_size.width-1)*0.5f;                                    \
    center.y -= (win_size.height-1)*0.5f;                                   \
                                                                            \
    ip.x = cvFloor( center.x );                                             \
    ip.y = cvFloor( center.y );                                             \
                                                                            \
    a = scale_macro( center.x - ip.x );                                     \
    b = scale_macro( center.y - ip.y );                                     \
                                                                            \
    src_step /= sizeof( src[0] );                                           \
    dst_step /= sizeof( dst[0] );                                           \
                                                                            \
    if( 0 <= ip.x && ip.x + win_size.width < src_size.width &&              \
        0 <= ip.y && ip.y + win_size.height < src_size.height )             \
    {                                                                       \
        /* extracted rectangle is totally inside the image */               \
        src += ip.y * src_step + ip.x*3;                                    \
                                                                            \
        for( i = 0; i < win_size.height; i++, src += src_step,              \
                                              dst += dst_step )             \
        {                                                                   \
            for( j = 0; j < win_size.width; j++ )                           \
            {                                                               \
                worktype s0 = cast_macro(src[j*3]);                         \
                worktype s1 = cast_macro(src[j*3 + src_step]);              \
                s0 += mul_macro( a, (cast_macro(src[j*3+3]) - s0));         \
                s1 += mul_macro( a, (cast_macro(src[j*3+3+src_step]) - s1));\
                dst[j*3] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));        \
                                                                            \
                s0 = cast_macro(src[j*3+1]);                                \
                s1 = cast_macro(src[j*3+1 + src_step]);                     \
                s0 += mul_macro( a, (cast_macro(src[j*3+4]) - s0));         \
                s1 += mul_macro( a, (cast_macro(src[j*3+4+src_step]) - s1));\
                dst[j*3+1] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
                                                                            \
                s0 = cast_macro(src[j*3+2]);                                \
                s1 = cast_macro(src[j*3+2 + src_step]);                     \
                s0 += mul_macro( a, (cast_macro(src[j*3+5]) - s0));         \
                s1 += mul_macro( a, (cast_macro(src[j*3+5+src_step]) - s1));\
                dst[j*3+2] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
            }                                                               \
        }                                                                   \
    }                                                                       \
    else                                                                    \
    {                                                                       \
        CvRect r;                                                           \
                                                                            \
        src = (const srctype*)icvAdjustRect( src, src_step*sizeof(*src),    \
                             sizeof(*src)*3, src_size, win_size, ip, &r );  \
                                                                            \
        for( i = 0; i < win_size.height; i++, dst += dst_step )             \
        {                                                                   \
            const srctype *src2 = src + src_step;                           \
                                                                            \
            if( i < r.y || i >= r.height )                                  \
                src2 -= src_step;                                           \
                                                                            \
            for( j = 0; j < r.x; j++ )                                      \
            {                                                               \
                worktype s0 = cast_macro(src[r.x*3]);                       \
                worktype s1 = cast_macro(src2[r.x*3]);                      \
                dst[j*3] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));        \
                                                                            \
                s0 = cast_macro(src[r.x*3+1]);                              \
                s1 = cast_macro(src2[r.x*3+1]);                             \
                dst[j*3+1] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
                                                                            \
                s0 = cast_macro(src[r.x*3+2]);                              \
                s1 = cast_macro(src2[r.x*3+2]);                             \
                dst[j*3+2] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
            }                                                               \
                                                                            \
            for( ; j < r.width; j++ )                                       \
            {                                                               \
                worktype s0 = cast_macro(src[j*3]);                         \
                worktype s1 = cast_macro(src2[j*3]);                        \
                s0 += mul_macro( a, (cast_macro(src[j*3 + 3]) - s0));       \
                s1 += mul_macro( a, (cast_macro(src2[j*3 + 3]) - s1));      \
                dst[j*3] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));        \
                                                                            \
                s0 = cast_macro(src[j*3+1]);                                \
                s1 = cast_macro(src2[j*3+1]);                               \
                s0 += mul_macro( a, (cast_macro(src[j*3 + 4]) - s0));       \
                s1 += mul_macro( a, (cast_macro(src2[j*3 + 4]) - s1));      \
                dst[j*3+1] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
                                                                            \
                s0 = cast_macro(src[j*3+2]);                                \
                s1 = cast_macro(src2[j*3+2]);                               \
                s0 += mul_macro( a, (cast_macro(src[j*3 + 5]) - s0));       \
                s1 += mul_macro( a, (cast_macro(src2[j*3 + 5]) - s1));      \
                dst[j*3+2] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
            }                                                               \
                                                                            \
            for( ; j < win_size.width; j++ )                                \
            {                                                               \
                worktype s0 = cast_macro(src[r.width*3]);                   \
                worktype s1 = cast_macro(src2[r.width*3]);                  \
                dst[j*3] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));        \
                                                                            \
                s0 = cast_macro(src[r.width*3+1]);                          \
                s1 = cast_macro(src2[r.width*3+1]);                         \
                dst[j*3+1] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
                                                                            \
                s0 = cast_macro(src[r.width*3+2]);                          \
                s1 = cast_macro(src2[r.width*3+2]);                         \
                dst[j*3+2] = (dsttype)(s0 + mul_macro( b, (s1 - s0)));      \
            }                                                               \
                                                                            \
            if( i < r.height )                                              \
                src = src2;                                                 \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}



CvStatus CV_STDCALL icvGetRectSubPix_8u32f_C1R
( const uchar* src, int src_step, CvSize src_size,
  float* dst, int dst_step, CvSize win_size, CvPoint2D32f center )
{
    CvPoint ip;
    float  a12, a22, b1, b2;
    float a, b;
    double s = 0;
    int i, j;

    center.x -= (win_size.width-1)*0.5f;
    center.y -= (win_size.height-1)*0.5f;

    ip.x = cvFloor( center.x );
    ip.y = cvFloor( center.y );

    if( win_size.width <= 0 || win_size.height <= 0 )
        return CV_BADRANGE_ERR;

    a = center.x - ip.x;
    b = center.y - ip.y;
    a = MAX(a,0.0001f);
    a12 = a*(1.f-b);
    a22 = a*b;
    b1 = 1.f - b;
    b2 = b;
    s = (1. - a)/a;

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dst[0]);

    if( 0 <= ip.x && ip.x + win_size.width < src_size.width &&
        0 <= ip.y && ip.y + win_size.height < src_size.height )
    {
        // extracted rectangle is totally inside the image
        src += ip.y * src_step + ip.x;

#if 0
        if( icvCopySubpix_8u32f_C1R_p &&
            icvCopySubpix_8u32f_C1R_p( src, src_step, dst,
                dst_step*sizeof(dst[0]), win_size, a, b ) >= 0 )
            return CV_OK;
#endif

        for( ; win_size.height--; src += src_step, dst += dst_step )
        {
            float prev = (1 - a)*(b1*CV_8TO32F(src[0]) + b2*CV_8TO32F(src[src_step]));
            for( j = 0; j < win_size.width; j++ )
            {
                float t = a12*CV_8TO32F(src[j+1]) + a22*CV_8TO32F(src[j+1+src_step]);
                dst[j] = prev + t;
                prev = (float)(t*s);
            }
        }
    }
    else
    {
        CvRect r;

        src = (const uchar*)icvAdjustRect( src, src_step*sizeof(*src),
                               sizeof(*src), src_size, win_size,ip, &r);

        for( i = 0; i < win_size.height; i++, dst += dst_step )
        {
            const uchar *src2 = src + src_step;

            if( i < r.y || i >= r.height )
                src2 -= src_step;

            for( j = 0; j < r.x; j++ )
            {
                float s0 = CV_8TO32F(src[r.x])*b1 +
                           CV_8TO32F(src2[r.x])*b2;

                dst[j] = (float)(s0);
            }

            if( j < r.width )
            {
                float prev = (1 - a)*(b1*CV_8TO32F(src[j]) + b2*CV_8TO32F(src2[j]));

                for( ; j < r.width; j++ )
                {
                    float t = a12*CV_8TO32F(src[j+1]) + a22*CV_8TO32F(src2[j+1]);
                    dst[j] = prev + t;
                    prev = (float)(t*s);
                }
            }

            for( ; j < win_size.width; j++ )
            {
                float s0 = CV_8TO32F(src[r.width])*b1 +
                           CV_8TO32F(src2[r.width])*b2;

                dst[j] = (float)(s0);
            }

            if( i < r.height )
                src = src2;
        }
    }

    return CV_OK;
}



#define ICV_SHIFT             16
#define ICV_SCALE(x)          cvRound((x)*(1 << ICV_SHIFT))
#define ICV_MUL_SCALE(x,y)    (((x)*(y) + (1 << (ICV_SHIFT-1))) >> ICV_SHIFT)
#define ICV_DESCALE(x)        (((x)+(1 << (ICV_SHIFT-1))) >> ICV_SHIFT)

/*icvCopySubpix_8u_C1R_t icvCopySubpix_8u_C1R_p = 0;
icvCopySubpix_8u32f_C1R_t icvCopySubpix_8u32f_C1R_p = 0;
icvCopySubpix_32f_C1R_t icvCopySubpix_32f_C1R_p = 0;*/

ICV_DEF_GET_RECT_SUB_PIX_FUNC( 8u, uchar, uchar, int, CV_NOP, ICV_SCALE, ICV_DESCALE )
//ICV_DEF_GET_RECT_SUB_PIX_FUNC( 8u32f, uchar, float, float, CV_8TO32F, CV_NOP, CV_NOP )
ICV_DEF_GET_RECT_SUB_PIX_FUNC( 32f, float, float, float, CV_NOP, CV_NOP, CV_NOP )

ICV_DEF_GET_RECT_SUB_PIX_FUNC_C3( 8u, uchar, uchar, int, CV_NOP, ICV_SCALE, ICV_MUL_SCALE )
ICV_DEF_GET_RECT_SUB_PIX_FUNC_C3( 8u32f, uchar, float, float, CV_8TO32F, CV_NOP, CV_MUL )
ICV_DEF_GET_RECT_SUB_PIX_FUNC_C3( 32f, float, float, float, CV_NOP, CV_NOP, CV_MUL )


#define  ICV_DEF_INIT_SUBPIX_TAB( FUNCNAME, FLAG )                  \
static void icvInit##FUNCNAME##FLAG##Table( CvFuncTable* tab )      \
{                                                                   \
    tab->fn_2d[CV_8U] = (void*)icv##FUNCNAME##_8u_##FLAG;           \
    tab->fn_2d[CV_32F] = (void*)icv##FUNCNAME##_32f_##FLAG;         \
                                                                    \
    tab->fn_2d[1] = (void*)icv##FUNCNAME##_8u32f_##FLAG;            \
}


ICV_DEF_INIT_SUBPIX_TAB( GetRectSubPix, C1R )
ICV_DEF_INIT_SUBPIX_TAB( GetRectSubPix, C3R )

typedef CvStatus (CV_STDCALL *CvGetRectSubPixFunc)( const void* src, int src_step,
                                                    CvSize src_size, void* dst,
                                                    int dst_step, CvSize win_size,
                                                    CvPoint2D32f center );

CV_IMPL void
cvGetRectSubPix( const void* srcarr, void* dstarr, CvPoint2D32f center )
{
    static CvFuncTable gr_tab[2];
    static int inittab = 0;

    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvSize src_size, dst_size;
    CvGetRectSubPixFunc func;
    int cn, src_step, dst_step;

    if( !inittab )
    {
        icvInitGetRectSubPixC1RTable( gr_tab + 0 );
        icvInitGetRectSubPixC3RTable( gr_tab + 1 );
        inittab = 1;
    }

    if( !CV_IS_MAT(src))
        src = cvGetMat( src, &srcstub );

    if( !CV_IS_MAT(dst))
        dst = cvGetMat( dst, &dststub );

    cn = CV_MAT_CN( src->type );

    if( (cn != 1 && cn != 3) || !CV_ARE_CNS_EQ( src, dst ))
        CV_Error( CV_StsUnsupportedFormat, "" );

    src_size = cvGetMatSize( src );
    dst_size = cvGetMatSize( dst );
    src_step = src->step ? src->step : CV_STUB_STEP;
    dst_step = dst->step ? dst->step : CV_STUB_STEP;

    //if( dst_size.width > src_size.width || dst_size.height > src_size.height )
    //    CV_ERROR( CV_StsBadSize, "destination ROI must be smaller than source ROI" );

    if( CV_ARE_DEPTHS_EQ( src, dst ))
    {
        func = (CvGetRectSubPixFunc)(gr_tab[cn != 1].fn_2d[CV_MAT_DEPTH(src->type)]);
    }
    else
    {
        if( CV_MAT_DEPTH( src->type ) != CV_8U || CV_MAT_DEPTH( dst->type ) != CV_32F )
            CV_Error( CV_StsUnsupportedFormat, "" );

        func = (CvGetRectSubPixFunc)(gr_tab[cn != 1].fn_2d[1]);
    }

    if( !func )
        CV_Error( CV_StsUnsupportedFormat, "" );

    IPPI_CALL( func( src->data.ptr, src_step, src_size,
                     dst->data.ptr, dst_step, dst_size, center ));
}


#define ICV_32F8U(x)  ((uchar)cvRound(x))

#define ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( flavor, srctype, dsttype,      \
                                             worktype, cast_macro, cvt )    \
CvStatus CV_STDCALL                                                         \
icvGetQuadrangleSubPix_##flavor##_C1R                                       \
( const srctype * src, int src_step, CvSize src_size,                       \
  dsttype *dst, int dst_step, CvSize win_size, const float *matrix )        \
{                                                                           \
    int x, y;                                                               \
    double dx = (win_size.width - 1)*0.5;                                   \
    double dy = (win_size.height - 1)*0.5;                                  \
    double A11 = matrix[0], A12 = matrix[1], A13 = matrix[2]-A11*dx-A12*dy; \
    double A21 = matrix[3], A22 = matrix[4], A23 = matrix[5]-A21*dx-A22*dy; \
                                                                            \
    src_step /= sizeof(srctype);                                            \
    dst_step /= sizeof(dsttype);                                            \
                                                                            \
    for( y = 0; y < win_size.height; y++, dst += dst_step )                 \
    {                                                                       \
        double xs = A12*y + A13;                                            \
        double ys = A22*y + A23;                                            \
        double xe = A11*(win_size.width-1) + A12*y + A13;                   \
        double ye = A21*(win_size.width-1) + A22*y + A23;                   \
                                                                            \
        if( (unsigned)(cvFloor(xs)-1) < (unsigned)(src_size.width - 3) &&   \
            (unsigned)(cvFloor(ys)-1) < (unsigned)(src_size.height - 3) &&  \
            (unsigned)(cvFloor(xe)-1) < (unsigned)(src_size.width - 3) &&   \
            (unsigned)(cvFloor(ye)-1) < (unsigned)(src_size.height - 3))    \
        {                                                                   \
            for( x = 0; x < win_size.width; x++ )                           \
            {                                                               \
                int ixs = cvFloor( xs );                                    \
                int iys = cvFloor( ys );                                    \
                const srctype *ptr = src + src_step*iys + ixs;              \
                double a = xs - ixs, b = ys - iys, a1 = 1.f - a;            \
                worktype p0 = cvt(ptr[0])*a1 + cvt(ptr[1])*a;               \
                worktype p1 = cvt(ptr[src_step])*a1 + cvt(ptr[src_step+1])*a;\
                xs += A11;                                                  \
                ys += A21;                                                  \
                                                                            \
                dst[x] = cast_macro(p0 + b * (p1 - p0));                    \
            }                                                               \
        }                                                                   \
        else                                                                \
        {                                                                   \
            for( x = 0; x < win_size.width; x++ )                           \
            {                                                               \
                int ixs = cvFloor( xs ), iys = cvFloor( ys );               \
                double a = xs - ixs, b = ys - iys, a1 = 1.f - a;            \
                const srctype *ptr0, *ptr1;                                 \
                worktype p0, p1;                                            \
                xs += A11; ys += A21;                                       \
                                                                            \
                if( (unsigned)iys < (unsigned)(src_size.height-1) )         \
                    ptr0 = src + src_step*iys, ptr1 = ptr0 + src_step;      \
                else                                                        \
                    ptr0 = ptr1 = src + (iys < 0 ? 0 : src_size.height-1)*src_step; \
                                                                            \
                if( (unsigned)ixs < (unsigned)(src_size.width-1) )          \
                {                                                           \
                    p0 = cvt(ptr0[ixs])*a1 + cvt(ptr0[ixs+1])*a;            \
                    p1 = cvt(ptr1[ixs])*a1 + cvt(ptr1[ixs+1])*a;            \
                }                                                           \
                else                                                        \
                {                                                           \
                    ixs = ixs < 0 ? 0 : src_size.width - 1;                 \
                    p0 = cvt(ptr0[ixs]); p1 = cvt(ptr1[ixs]);               \
                }                                                           \
                dst[x] = cast_macro(p0 + b * (p1 - p0));                    \
            }                                                               \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}


#define ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC_C3( flavor, srctype, dsttype,   \
                                                worktype, cast_macro, cvt ) \
static CvStatus CV_STDCALL                                                  \
icvGetQuadrangleSubPix_##flavor##_C3R                                       \
( const srctype * src, int src_step, CvSize src_size,                       \
  dsttype *dst, int dst_step, CvSize win_size, const float *matrix )        \
{                                                                           \
    int x, y;                                                               \
    double dx = (win_size.width - 1)*0.5;                                   \
    double dy = (win_size.height - 1)*0.5;                                  \
    double A11 = matrix[0], A12 = matrix[1], A13 = matrix[2]-A11*dx-A12*dy; \
    double A21 = matrix[3], A22 = matrix[4], A23 = matrix[5]-A21*dx-A22*dy; \
                                                                            \
    src_step /= sizeof(srctype);                                            \
    dst_step /= sizeof(dsttype);                                            \
                                                                            \
    for( y = 0; y < win_size.height; y++, dst += dst_step )                 \
    {                                                                       \
        double xs = A12*y + A13;                                            \
        double ys = A22*y + A23;                                            \
        double xe = A11*(win_size.width-1) + A12*y + A13;                   \
        double ye = A21*(win_size.width-1) + A22*y + A23;                   \
                                                                            \
        if( (unsigned)(cvFloor(xs)-1) < (unsigned)(src_size.width - 3) &&   \
            (unsigned)(cvFloor(ys)-1) < (unsigned)(src_size.height - 3) &&  \
            (unsigned)(cvFloor(xe)-1) < (unsigned)(src_size.width - 3) &&   \
            (unsigned)(cvFloor(ye)-1) < (unsigned)(src_size.height - 3))    \
        {                                                                   \
            for( x = 0; x < win_size.width; x++ )                           \
            {                                                               \
                int ixs = cvFloor( xs );                                    \
                int iys = cvFloor( ys );                                    \
                const srctype *ptr = src + src_step*iys + ixs*3;            \
                double a = xs - ixs, b = ys - iys, a1 = 1.f - a;            \
                worktype p0, p1;                                            \
                xs += A11;                                                  \
                ys += A21;                                                  \
                                                                            \
                p0 = cvt(ptr[0])*a1 + cvt(ptr[3])*a;                        \
                p1 = cvt(ptr[src_step])*a1 + cvt(ptr[src_step+3])*a;        \
                dst[x*3] = cast_macro(p0 + b * (p1 - p0));                  \
                                                                            \
                p0 = cvt(ptr[1])*a1 + cvt(ptr[4])*a;                        \
                p1 = cvt(ptr[src_step+1])*a1 + cvt(ptr[src_step+4])*a;      \
                dst[x*3+1] = cast_macro(p0 + b * (p1 - p0));                \
                                                                            \
                p0 = cvt(ptr[2])*a1 + cvt(ptr[5])*a;                        \
                p1 = cvt(ptr[src_step+2])*a1 + cvt(ptr[src_step+5])*a;      \
                dst[x*3+2] = cast_macro(p0 + b * (p1 - p0));                \
            }                                                               \
        }                                                                   \
        else                                                                \
        {                                                                   \
            for( x = 0; x < win_size.width; x++ )                           \
            {                                                               \
                int ixs = cvFloor(xs), iys = cvFloor(ys);                   \
                double a = xs - ixs, b = ys - iys;                          \
                const srctype *ptr0, *ptr1;                                 \
                xs += A11; ys += A21;                                       \
                                                                            \
                if( (unsigned)iys < (unsigned)(src_size.height-1) )         \
                    ptr0 = src + src_step*iys, ptr1 = ptr0 + src_step;      \
                else                                                        \
                    ptr0 = ptr1 = src + (iys < 0 ? 0 : src_size.height-1)*src_step; \
                                                                            \
                if( (unsigned)ixs < (unsigned)(src_size.width - 1) )        \
                {                                                           \
                    double a1 = 1.f - a;                                    \
                    worktype p0, p1;                                        \
                    ptr0 += ixs*3; ptr1 += ixs*3;                           \
                    p0 = cvt(ptr0[0])*a1 + cvt(ptr0[3])*a;                  \
                    p1 = cvt(ptr1[0])*a1 + cvt(ptr1[3])*a;                  \
                    dst[x*3] = cast_macro(p0 + b * (p1 - p0));              \
                                                                            \
                    p0 = cvt(ptr0[1])*a1 + cvt(ptr0[4])*a;                  \
                    p1 = cvt(ptr1[1])*a1 + cvt(ptr1[4])*a;                  \
                    dst[x*3+1] = cast_macro(p0 + b * (p1 - p0));            \
                                                                            \
                    p0 = cvt(ptr0[2])*a1 + cvt(ptr0[5])*a;                  \
                    p1 = cvt(ptr1[2])*a1 + cvt(ptr1[5])*a;                  \
                    dst[x*3+2] = cast_macro(p0 + b * (p1 - p0));            \
                }                                                           \
                else                                                        \
                {                                                           \
                    double b1 = 1.f - b;                                    \
                    ixs = ixs < 0 ? 0 : src_size.width - 1;                 \
                    ptr0 += ixs*3; ptr1 += ixs*3;                           \
                                                                            \
                    dst[x*3] = cast_macro(cvt(ptr0[0])*b1 + cvt(ptr1[0])*b);\
                    dst[x*3+1]=cast_macro(cvt(ptr0[1])*b1 + cvt(ptr1[1])*b);\
                    dst[x*3+2]=cast_macro(cvt(ptr0[2])*b1 + cvt(ptr1[2])*b);\
                }                                                           \
            }                                                               \
        }                                                                   \
    }                                                                       \
                                                                            \
    return CV_OK;                                                           \
}


/*#define srctype uchar
#define dsttype uchar
#define worktype float
#define cvt CV_8TO32F
#define cast_macro ICV_32F8U

#undef srctype
#undef dsttype
#undef worktype
#undef cvt
#undef cast_macro*/

ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( 8u, uchar, uchar, double, ICV_32F8U, CV_8TO32F )
ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( 32f, float, float, double, CV_CAST_32F, CV_NOP )
ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( 8u32f, uchar, float, double, CV_CAST_32F, CV_8TO32F )

ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC_C3( 8u, uchar, uchar, double, ICV_32F8U, CV_8TO32F )
ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC_C3( 32f, float, float, double, CV_CAST_32F, CV_NOP )
ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC_C3( 8u32f, uchar, float, double, CV_CAST_32F, CV_8TO32F )

ICV_DEF_INIT_SUBPIX_TAB( GetQuadrangleSubPix, C1R )
ICV_DEF_INIT_SUBPIX_TAB( GetQuadrangleSubPix, C3R )

typedef CvStatus (CV_STDCALL *CvGetQuadrangleSubPixFunc)(
                                         const void* src, int src_step,
                                         CvSize src_size, void* dst,
                                         int dst_step, CvSize win_size,
                                         const float* matrix );

CV_IMPL void
cvGetQuadrangleSubPix( const void* srcarr, void* dstarr, const CvMat* mat )
{
    static  CvFuncTable  gq_tab[2];
    static  int inittab = 0;

    CvMat srcstub, *src = (CvMat*)srcarr;
    CvMat dststub, *dst = (CvMat*)dstarr;
    CvSize src_size, dst_size;
    CvGetQuadrangleSubPixFunc func;
    float m[6];
    int k, cn;

    if( !inittab )
    {
        icvInitGetQuadrangleSubPixC1RTable( gq_tab + 0 );
        icvInitGetQuadrangleSubPixC3RTable( gq_tab + 1 );
        inittab = 1;
    }

    if( !CV_IS_MAT(src))
        src = cvGetMat( src, &srcstub );

    if( !CV_IS_MAT(dst))
        dst = cvGetMat( dst, &dststub );

    if( !CV_IS_MAT(mat))
        CV_Error( CV_StsBadArg, "map matrix is not valid" );

    cn = CV_MAT_CN( src->type );

    if( (cn != 1 && cn != 3) || !CV_ARE_CNS_EQ( src, dst ))
        CV_Error( CV_StsUnsupportedFormat, "" );

    src_size = cvGetMatSize( src );
    dst_size = cvGetMatSize( dst );

    /*if( dst_size.width > src_size.width || dst_size.height > src_size.height )
        CV_ERROR( CV_StsBadSize, "destination ROI must not be larger than source ROI" );*/

    if( mat->rows != 2 || mat->cols != 3 )
        CV_Error( CV_StsBadArg,
        "Transformation matrix must be 2x3" );

    if( CV_MAT_TYPE( mat->type ) == CV_32FC1 )
    {
        for( k = 0; k < 3; k++ )
        {
            m[k] = mat->data.fl[k];
            m[3 + k] = ((float*)(mat->data.ptr + mat->step))[k];
        }
    }
    else if( CV_MAT_TYPE( mat->type ) == CV_64FC1 )
    {
        for( k = 0; k < 3; k++ )
        {
            m[k] = (float)mat->data.db[k];
            m[3 + k] = (float)((double*)(mat->data.ptr + mat->step))[k];
        }
    }
    else
        CV_Error( CV_StsUnsupportedFormat,
            "The transformation matrix should have 32fC1 or 64fC1 type" );

    if( CV_ARE_DEPTHS_EQ( src, dst ))
    {
        func = (CvGetQuadrangleSubPixFunc)(gq_tab[cn != 1].fn_2d[CV_MAT_DEPTH(src->type)]);
    }
    else
    {
        if( CV_MAT_DEPTH( src->type ) != CV_8U || CV_MAT_DEPTH( dst->type ) != CV_32F )
            CV_Error( CV_StsUnsupportedFormat, "" );

        func = (CvGetQuadrangleSubPixFunc)(gq_tab[cn != 1].fn_2d[1]);
    }

    if( !func )
        CV_Error( CV_StsUnsupportedFormat, "" );

    IPPI_CALL( func( src->data.ptr, src->step, src_size,
                     dst->data.ptr, dst->step, dst_size, m ));
}


void cv::getRectSubPix( const Mat& image, Size patchSize, Point2f center,
                        Mat& patch, int patchType )
{
    patch.create(patchSize, patchType < 0 ? image.type() :
        CV_MAKETYPE(CV_MAT_DEPTH(patchType),image.channels()));
    CvMat _image = image, _patch = patch;
    cvGetRectSubPix(&_image, &_patch, center);
}

/* End of file. */
