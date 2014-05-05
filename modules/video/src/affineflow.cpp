/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/video/tracking_c.h"

// to be moved to legacy

static int icvMinimalPyramidSize( CvSize imgSize )
{
    return cvAlign(imgSize.width,8) * imgSize.height / 3;
}


static void
icvInitPyramidalAlgorithm( const CvMat* imgA, const CvMat* imgB,
                           CvMat* pyrA, CvMat* pyrB,
                           int level, CvTermCriteria * criteria,
                           int max_iters, int flags,
                           uchar *** imgI, uchar *** imgJ,
                           int **step, CvSize** size,
                           double **scale, cv::AutoBuffer<uchar>* buffer )
{
    const int ALIGN = 8;
    int pyrBytes, bufferBytes = 0, elem_size;
    int level1 = level + 1;

    int i;
    CvSize imgSize, levelSize;

    *imgI = *imgJ = 0;
    *step = 0;
    *scale = 0;
    *size = 0;

    /* check input arguments */
    if( ((flags & CV_LKFLOW_PYR_A_READY) != 0 && !pyrA) ||
        ((flags & CV_LKFLOW_PYR_B_READY) != 0 && !pyrB) )
        CV_Error( CV_StsNullPtr, "Some of the precomputed pyramids are missing" );

    if( level < 0 )
        CV_Error( CV_StsOutOfRange, "The number of pyramid levels is negative" );

    switch( criteria->type )
    {
    case CV_TERMCRIT_ITER:
        criteria->epsilon = 0.f;
        break;
    case CV_TERMCRIT_EPS:
        criteria->max_iter = max_iters;
        break;
    case CV_TERMCRIT_ITER | CV_TERMCRIT_EPS:
        break;
    default:
        assert( 0 );
        CV_Error( CV_StsBadArg, "Invalid termination criteria" );
    }

    /* compare squared values */
    criteria->epsilon *= criteria->epsilon;

    /* set pointers and step for every level */
    pyrBytes = 0;

    imgSize = cvGetSize(imgA);
    elem_size = CV_ELEM_SIZE(imgA->type);
    levelSize = imgSize;

    for( i = 1; i < level1; i++ )
    {
        levelSize.width = (levelSize.width + 1) >> 1;
        levelSize.height = (levelSize.height + 1) >> 1;

        int tstep = cvAlign(levelSize.width,ALIGN) * elem_size;
        pyrBytes += tstep * levelSize.height;
    }

    assert( pyrBytes <= imgSize.width * imgSize.height * elem_size * 4 / 3 );

    /* buffer_size = <size for patches> + <size for pyramids> */
    bufferBytes = (int)((level1 >= 0) * ((pyrA->data.ptr == 0) +
        (pyrB->data.ptr == 0)) * pyrBytes +
        (sizeof(imgI[0][0]) * 2 + sizeof(step[0][0]) +
         sizeof(size[0][0]) + sizeof(scale[0][0])) * level1);

    buffer->allocate( bufferBytes );

    *imgI = (uchar **) (uchar*)(*buffer);
    *imgJ = *imgI + level1;
    *step = (int *) (*imgJ + level1);
    *scale = (double *) (*step + level1);
    *size = (CvSize *)(*scale + level1);

    imgI[0][0] = imgA->data.ptr;
    imgJ[0][0] = imgB->data.ptr;
    step[0][0] = imgA->step;
    scale[0][0] = 1;
    size[0][0] = imgSize;

    if( level > 0 )
    {
        uchar *bufPtr = (uchar *) (*size + level1);
        uchar *ptrA = pyrA->data.ptr;
        uchar *ptrB = pyrB->data.ptr;

        if( !ptrA )
        {
            ptrA = bufPtr;
            bufPtr += pyrBytes;
        }

        if( !ptrB )
            ptrB = bufPtr;

        levelSize = imgSize;

        /* build pyramids for both frames */
        for( i = 1; i <= level; i++ )
        {
            int levelBytes;
            CvMat prev_level, next_level;

            levelSize.width = (levelSize.width + 1) >> 1;
            levelSize.height = (levelSize.height + 1) >> 1;

            size[0][i] = levelSize;
            step[0][i] = cvAlign( levelSize.width, ALIGN ) * elem_size;
            scale[0][i] = scale[0][i - 1] * 0.5;

            levelBytes = step[0][i] * levelSize.height;
            imgI[0][i] = (uchar *) ptrA;
            ptrA += levelBytes;

            if( !(flags & CV_LKFLOW_PYR_A_READY) )
            {
                prev_level = cvMat( size[0][i-1].height, size[0][i-1].width, CV_8UC1 );
                next_level = cvMat( size[0][i].height, size[0][i].width, CV_8UC1 );
                cvSetData( &prev_level, imgI[0][i-1], step[0][i-1] );
                cvSetData( &next_level, imgI[0][i], step[0][i] );
                cvPyrDown( &prev_level, &next_level );
            }

            imgJ[0][i] = (uchar *) ptrB;
            ptrB += levelBytes;

            if( !(flags & CV_LKFLOW_PYR_B_READY) )
            {
                prev_level = cvMat( size[0][i-1].height, size[0][i-1].width, CV_8UC1 );
                next_level = cvMat( size[0][i].height, size[0][i].width, CV_8UC1 );
                cvSetData( &prev_level, imgJ[0][i-1], step[0][i-1] );
                cvSetData( &next_level, imgJ[0][i], step[0][i] );
                cvPyrDown( &prev_level, &next_level );
            }
        }
    }
}


/* compute dI/dx and dI/dy */
static void
icvCalcIxIy_32f( const float* src, int src_step, float* dstX, float* dstY, int dst_step,
                 CvSize src_size, const float* smooth_k, float* buffer0 )
{
    int src_width = src_size.width, dst_width = src_size.width-2;
    int x, height = src_size.height - 2;
    float* buffer1 = buffer0 + src_width;

    src_step /= sizeof(src[0]);
    dst_step /= sizeof(dstX[0]);

    for( ; height--; src += src_step, dstX += dst_step, dstY += dst_step )
    {
        const float* src2 = src + src_step;
        const float* src3 = src + src_step*2;

        for( x = 0; x < src_width; x++ )
        {
            float t0 = (src3[x] + src[x])*smooth_k[0] + src2[x]*smooth_k[1];
            float t1 = src3[x] - src[x];
            buffer0[x] = t0; buffer1[x] = t1;
        }

        for( x = 0; x < dst_width; x++ )
        {
            float t0 = buffer0[x+2] - buffer0[x];
            float t1 = (buffer1[x] + buffer1[x+2])*smooth_k[0] + buffer1[x+1]*smooth_k[1];
            dstX[x] = t0; dstY[x] = t1;
        }
    }
}


#undef CV_8TO32F
#define CV_8TO32F(a) (a)

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


static CvStatus CV_STDCALL icvGetRectSubPix_8u32f_C1R
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


#define ICV_32F8U(x)  ((uchar)cvRound(x))

#define ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( flavor, srctype, dsttype, worktype, cast_macro, cvt ) \
static CvStatus CV_STDCALL icvGetQuadrangleSubPix_##flavor##_C1R                    \
              ( const srctype * src, int src_step, CvSize src_size,                 \
                dsttype *dst, int dst_step, CvSize win_size, const float *matrix )  \
{                                                                                   \
    int x, y;                                                                       \
    double dx = (win_size.width - 1)*0.5;                                           \
    double dy = (win_size.height - 1)*0.5;                                          \
    double A11 = matrix[0], A12 = matrix[1], A13 = matrix[2]-A11*dx-A12*dy;         \
    double A21 = matrix[3], A22 = matrix[4], A23 = matrix[5]-A21*dx-A22*dy;         \
                                                                                    \
    src_step /= sizeof(srctype);                                                    \
    dst_step /= sizeof(dsttype);                                                    \
                                                                                    \
    for( y = 0; y < win_size.height; y++, dst += dst_step )                         \
    {                                                                               \
        double xs = A12*y + A13;                                                    \
        double ys = A22*y + A23;                                                    \
        double xe = A11*(win_size.width-1) + A12*y + A13;                           \
        double ye = A21*(win_size.width-1) + A22*y + A23;                           \
                                                                                    \
        if( (unsigned)(cvFloor(xs)-1) < (unsigned)(src_size.width - 3) &&           \
            (unsigned)(cvFloor(ys)-1) < (unsigned)(src_size.height - 3) &&          \
            (unsigned)(cvFloor(xe)-1) < (unsigned)(src_size.width - 3) &&           \
            (unsigned)(cvFloor(ye)-1) < (unsigned)(src_size.height - 3))            \
        {                                                                           \
            for( x = 0; x < win_size.width; x++ )                                   \
            {                                                                       \
                int ixs = cvFloor( xs );                                            \
                int iys = cvFloor( ys );                                            \
                const srctype *ptr = src + src_step*iys + ixs;                      \
                double a = xs - ixs, b = ys - iys, a1 = 1.f - a;                    \
                worktype p0 = cvt(ptr[0])*a1 + cvt(ptr[1])*a;                       \
                worktype p1 = cvt(ptr[src_step])*a1 + cvt(ptr[src_step+1])*a;       \
                xs += A11;                                                          \
                ys += A21;                                                          \
                                                                                    \
                dst[x] = cast_macro(p0 + b * (p1 - p0));                            \
            }                                                                       \
        }                                                                           \
        else                                                                        \
        {                                                                           \
            for( x = 0; x < win_size.width; x++ )                                   \
            {                                                                       \
                int ixs = cvFloor( xs ), iys = cvFloor( ys );                       \
                double a = xs - ixs, b = ys - iys, a1 = 1.f - a;                    \
                const srctype *ptr0, *ptr1;                                         \
                worktype p0, p1;                                                    \
                xs += A11; ys += A21;                                               \
                                                                                    \
                if( (unsigned)iys < (unsigned)(src_size.height-1) )                 \
                    ptr0 = src + src_step*iys, ptr1 = ptr0 + src_step;              \
                else                                                                \
                    ptr0 = ptr1 = src + (iys < 0 ? 0 : src_size.height-1)*src_step; \
                                                                                    \
                if( (unsigned)ixs < (unsigned)(src_size.width-1) )                  \
                {                                                                   \
                    p0 = cvt(ptr0[ixs])*a1 + cvt(ptr0[ixs+1])*a;                    \
                    p1 = cvt(ptr1[ixs])*a1 + cvt(ptr1[ixs+1])*a;                    \
                }                                                                   \
                else                                                                \
                {                                                                   \
                    ixs = ixs < 0 ? 0 : src_size.width - 1;                         \
                    p0 = cvt(ptr0[ixs]); p1 = cvt(ptr1[ixs]);                       \
                }                                                                   \
                dst[x] = cast_macro(p0 + b * (p1 - p0));                            \
            }                                                                       \
        }                                                                           \
    }                                                                               \
                                                                                    \
    return CV_OK;                                                                   \
}

ICV_DEF_GET_QUADRANGLE_SUB_PIX_FUNC( 8u32f, uchar, float, double, cv::saturate_cast<float>, CV_8TO32F )

/* Affine tracking algorithm */

CV_IMPL void
cvCalcAffineFlowPyrLK( const void* arrA, const void* arrB,
                       void* pyrarrA, void* pyrarrB,
                       const CvPoint2D32f * featuresA,
                       CvPoint2D32f * featuresB,
                       float *matrices, int count,
                       CvSize winSize, int level,
                       char *status, float *error,
                       CvTermCriteria criteria, int flags )
{
    const int MAX_ITERS = 100;

    cv::AutoBuffer<char> _status;
    cv::AutoBuffer<uchar> buffer;
    cv::AutoBuffer<uchar> pyr_buffer;

    CvMat stubA, *imgA = (CvMat*)arrA;
    CvMat stubB, *imgB = (CvMat*)arrB;
    CvMat pstubA, *pyrA = (CvMat*)pyrarrA;
    CvMat pstubB, *pyrB = (CvMat*)pyrarrB;

    static const float smoothKernel[] = { 0.09375, 0.3125, 0.09375 };  /* 3/32, 10/32, 3/32 */

    int bufferBytes = 0;

    uchar **imgI = 0;
    uchar **imgJ = 0;
    int *step = 0;
    double *scale = 0;
    CvSize* size = 0;

    float *patchI;
    float *patchJ;
    float *Ix;
    float *Iy;

    int i, j, k, l;

    CvSize patchSize = cvSize( winSize.width * 2 + 1, winSize.height * 2 + 1 );
    int patchLen = patchSize.width * patchSize.height;
    int patchStep = patchSize.width * sizeof( patchI[0] );

    CvSize srcPatchSize = cvSize( patchSize.width + 2, patchSize.height + 2 );
    int srcPatchLen = srcPatchSize.width * srcPatchSize.height;
    int srcPatchStep = srcPatchSize.width * sizeof( patchI[0] );
    CvSize imgSize;
    float eps = (float)MIN(winSize.width, winSize.height);

    imgA = cvGetMat( imgA, &stubA );
    imgB = cvGetMat( imgB, &stubB );

    if( CV_MAT_TYPE( imgA->type ) != CV_8UC1 )
        CV_Error( CV_StsUnsupportedFormat, "" );

    if( !CV_ARE_TYPES_EQ( imgA, imgB ))
        CV_Error( CV_StsUnmatchedFormats, "" );

    if( !CV_ARE_SIZES_EQ( imgA, imgB ))
        CV_Error( CV_StsUnmatchedSizes, "" );

    if( imgA->step != imgB->step )
        CV_Error( CV_StsUnmatchedSizes, "imgA and imgB must have equal steps" );

    if( !matrices )
        CV_Error( CV_StsNullPtr, "" );

    imgSize = cv::Size(imgA->cols, imgA->rows);

    if( pyrA )
    {
        pyrA = cvGetMat( pyrA, &pstubA );

        if( pyrA->step*pyrA->height < icvMinimalPyramidSize( imgSize ) )
            CV_Error( CV_StsBadArg, "pyramid A has insufficient size" );
    }
    else
    {
        pyrA = &pstubA;
        pyrA->data.ptr = 0;
    }

    if( pyrB )
    {
        pyrB = cvGetMat( pyrB, &pstubB );

        if( pyrB->step*pyrB->height < icvMinimalPyramidSize( imgSize ) )
            CV_Error( CV_StsBadArg, "pyramid B has insufficient size" );
    }
    else
    {
        pyrB = &pstubB;
        pyrB->data.ptr = 0;
    }

    if( count == 0 )
        return;

    /* check input arguments */
    if( !featuresA || !featuresB || !matrices )
        CV_Error( CV_StsNullPtr, "" );

    if( winSize.width <= 1 || winSize.height <= 1 )
        CV_Error( CV_StsOutOfRange, "the search window is too small" );

    if( count < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    icvInitPyramidalAlgorithm( imgA, imgB,
        pyrA, pyrB, level, &criteria, MAX_ITERS, flags,
        &imgI, &imgJ, &step, &size, &scale, &pyr_buffer );

    /* buffer_size = <size for patches> + <size for pyramids> */
    bufferBytes = (srcPatchLen + patchLen*3)*sizeof(patchI[0]) + (36*2 + 6)*sizeof(double);

    buffer.allocate(bufferBytes);

    if( !status )
    {
        _status.allocate(count);
        status = _status;
    }

    patchI = (float *)(uchar*)buffer;
    patchJ = patchI + srcPatchLen;
    Ix = patchJ + patchLen;
    Iy = Ix + patchLen;

    if( status )
        memset( status, 1, count );

    if( !(flags & CV_LKFLOW_INITIAL_GUESSES) )
    {
        memcpy( featuresB, featuresA, count * sizeof( featuresA[0] ));
        for( i = 0; i < count * 4; i += 4 )
        {
            matrices[i] = matrices[i + 3] = 1.f;
            matrices[i + 1] = matrices[i + 2] = 0.f;
        }
    }

    for( i = 0; i < count; i++ )
    {
        featuresB[i].x = (float)(featuresB[i].x * scale[level] * 0.5);
        featuresB[i].y = (float)(featuresB[i].y * scale[level] * 0.5);
    }

    /* do processing from top pyramid level (smallest image)
       to the bottom (original image) */
    for( l = level; l >= 0; l-- )
    {
        CvSize levelSize = size[l];
        int levelStep = step[l];

        /* find flow for each given point at the particular level */
        for( i = 0; i < count; i++ )
        {
            CvPoint2D32f u;
            float Av[6];
            double G[36];
            double meanI = 0, meanJ = 0;
            int x, y;
            int pt_status = status[i];
            CvMat mat;

            if( !pt_status )
                continue;

            Av[0] = matrices[i*4];
            Av[1] = matrices[i*4+1];
            Av[3] = matrices[i*4+2];
            Av[4] = matrices[i*4+3];

            Av[2] = featuresB[i].x += featuresB[i].x;
            Av[5] = featuresB[i].y += featuresB[i].y;

            u.x = (float) (featuresA[i].x * scale[l]);
            u.y = (float) (featuresA[i].y * scale[l]);

            if( u.x < -eps || u.x >= levelSize.width+eps ||
                u.y < -eps || u.y >= levelSize.height+eps ||
                icvGetRectSubPix_8u32f_C1R( imgI[l], levelStep,
                levelSize, patchI, srcPatchStep, srcPatchSize, u ) < 0 )
            {
                /* point is outside the image. take the next */
                if( l == 0 )
                    status[i] = 0;
                continue;
            }

            icvCalcIxIy_32f( patchI, srcPatchStep, Ix, Iy,
                (srcPatchSize.width-2)*sizeof(patchI[0]), srcPatchSize,
                smoothKernel, patchJ );

            /* repack patchI (remove borders) */
            for( k = 0; k < patchSize.height; k++ )
                memcpy( patchI + k * patchSize.width,
                        patchI + (k + 1) * srcPatchSize.width + 1, patchStep );

            memset( G, 0, sizeof( G ));

            /* calculate G matrix */
            for( y = -winSize.height, k = 0; y <= winSize.height; y++ )
            {
                for( x = -winSize.width; x <= winSize.width; x++, k++ )
                {
                    double ixix = ((double) Ix[k]) * Ix[k];
                    double ixiy = ((double) Ix[k]) * Iy[k];
                    double iyiy = ((double) Iy[k]) * Iy[k];

                    double xx, xy, yy;

                    G[0] += ixix;
                    G[1] += ixiy;
                    G[2] += x * ixix;
                    G[3] += y * ixix;
                    G[4] += x * ixiy;
                    G[5] += y * ixiy;

                    // G[6] == G[1]
                    G[7] += iyiy;
                    // G[8] == G[4]
                    // G[9] == G[5]
                    G[10] += x * iyiy;
                    G[11] += y * iyiy;

                    xx = x * x;
                    xy = x * y;
                    yy = y * y;

                    // G[12] == G[2]
                    // G[13] == G[8] == G[4]
                    G[14] += xx * ixix;
                    G[15] += xy * ixix;
                    G[16] += xx * ixiy;
                    G[17] += xy * ixiy;

                    // G[18] == G[3]
                    // G[19] == G[9]
                    // G[20] == G[15]
                    G[21] += yy * ixix;
                    // G[22] == G[17]
                    G[23] += yy * ixiy;

                    // G[24] == G[4]
                    // G[25] == G[10]
                    // G[26] == G[16]
                    // G[27] == G[22]
                    G[28] += xx * iyiy;
                    G[29] += xy * iyiy;

                    // G[30] == G[5]
                    // G[31] == G[11]
                    // G[32] == G[17]
                    // G[33] == G[23]
                    // G[34] == G[29]
                    G[35] += yy * iyiy;

                    meanI += patchI[k];
                }
            }

            meanI /= patchSize.width*patchSize.height;

            G[8] = G[4];
            G[9] = G[5];
            G[22] = G[17];

            // fill part of G below its diagonal
            for( y = 1; y < 6; y++ )
                for( x = 0; x < y; x++ )
                    G[y * 6 + x] = G[x * 6 + y];

            cvInitMatHeader( &mat, 6, 6, CV_64FC1, G );

            if( cvInvert( &mat, &mat, CV_SVD ) < 1e-4 )
            {
                /* bad matrix. take the next point */
                if( l == 0 )
                    status[i] = 0;
                continue;
            }

            for( j = 0; j < criteria.max_iter; j++ )
            {
                double b[6] = {0,0,0,0,0,0}, eta[6];
                double t0, t1, s = 0;

                if( Av[2] < -eps || Av[2] >= levelSize.width+eps ||
                    Av[5] < -eps || Av[5] >= levelSize.height+eps ||
                    icvGetQuadrangleSubPix_8u32f_C1R( imgJ[l], levelStep,
                    levelSize, patchJ, patchStep, patchSize, Av ) < 0 )
                {
                    pt_status = 0;
                    break;
                }

                for( y = -winSize.height, k = 0, meanJ = 0; y <= winSize.height; y++ )
                    for( x = -winSize.width; x <= winSize.width; x++, k++ )
                        meanJ += patchJ[k];

                meanJ = meanJ / (patchSize.width * patchSize.height) - meanI;

                for( y = -winSize.height, k = 0; y <= winSize.height; y++ )
                {
                    for( x = -winSize.width; x <= winSize.width; x++, k++ )
                    {
                        double t = patchI[k] - patchJ[k] + meanJ;
                        double ixt = Ix[k] * t;
                        double iyt = Iy[k] * t;

                        s += t;

                        b[0] += ixt;
                        b[1] += iyt;
                        b[2] += x * ixt;
                        b[3] += y * ixt;
                        b[4] += x * iyt;
                        b[5] += y * iyt;
                    }
                }

                for( k = 0; k < 6; k++ )
                    eta[k] = G[k*6]*b[0] + G[k*6+1]*b[1] + G[k*6+2]*b[2] +
                        G[k*6+3]*b[3] + G[k*6+4]*b[4] + G[k*6+5]*b[5];

                Av[2] = (float)(Av[2] + Av[0] * eta[0] + Av[1] * eta[1]);
                Av[5] = (float)(Av[5] + Av[3] * eta[0] + Av[4] * eta[1]);

                t0 = Av[0] * (1 + eta[2]) + Av[1] * eta[4];
                t1 = Av[0] * eta[3] + Av[1] * (1 + eta[5]);
                Av[0] = (float)t0;
                Av[1] = (float)t1;

                t0 = Av[3] * (1 + eta[2]) + Av[4] * eta[4];
                t1 = Av[3] * eta[3] + Av[4] * (1 + eta[5]);
                Av[3] = (float)t0;
                Av[4] = (float)t1;

                if( eta[0] * eta[0] + eta[1] * eta[1] < criteria.epsilon )
                    break;
            }

            if( pt_status != 0 || l == 0 )
            {
                status[i] = (char)pt_status;
                featuresB[i].x = Av[2];
                featuresB[i].y = Av[5];

                matrices[i*4] = Av[0];
                matrices[i*4+1] = Av[1];
                matrices[i*4+2] = Av[3];
                matrices[i*4+3] = Av[4];
            }

            if( pt_status && l == 0 && error )
            {
                /* calc error */
                double err = 0;

                for( y = 0, k = 0; y < patchSize.height; y++ )
                {
                    for( x = 0; x < patchSize.width; x++, k++ )
                    {
                        double t = patchI[k] - patchJ[k] + meanJ;
                        err += t * t;
                    }
                }
                error[i] = (float)std::sqrt(err);
            }
        }
    }
}
