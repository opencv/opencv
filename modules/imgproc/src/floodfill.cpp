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

typedef struct CvFFillSegment
{
    ushort y;
    ushort l;
    ushort r;
    ushort prevl;
    ushort prevr;
    short dir;
}
CvFFillSegment;

#define UP 1
#define DOWN -1

#define ICV_PUSH( Y, L, R, PREV_L, PREV_R, DIR )\
{                                               \
    tail->y = (ushort)(Y);                      \
    tail->l = (ushort)(L);                      \
    tail->r = (ushort)(R);                      \
    tail->prevl = (ushort)(PREV_L);             \
    tail->prevr = (ushort)(PREV_R);             \
    tail->dir = (short)(DIR);                   \
    if( ++tail >= buffer_end )                  \
        tail = buffer;                          \
}


#define ICV_POP( Y, L, R, PREV_L, PREV_R, DIR ) \
{                                               \
    Y = head->y;                                \
    L = head->l;                                \
    R = head->r;                                \
    PREV_L = head->prevl;                       \
    PREV_R = head->prevr;                       \
    DIR = head->dir;                            \
    if( ++head >= buffer_end )                  \
        head = buffer;                          \
}


#define ICV_EQ_C3( p1, p2 ) \
    ((p1)[0] == (p2)[0] && (p1)[1] == (p2)[1] && (p1)[2] == (p2)[2])

#define ICV_SET_C3( p, q ) \
    ((p)[0] = (q)[0], (p)[1] = (q)[1], (p)[2] = (q)[2])

/****************************************************************************************\
*              Simple Floodfill (repainting single-color connected component)            *
\****************************************************************************************/

static void
icvFloodFill_8u_CnIR( uchar* pImage, int step, CvSize roi, CvPoint seed,
                      uchar* _newVal, CvConnectedComp* region, int flags,
                      CvFFillSegment* buffer, int buffer_size, int cn )
{
    uchar* img = pImage + step * seed.y;
    int i, L, R;
    int area = 0;
    int val0[] = {0,0,0};
    uchar newVal[] = {0,0,0};
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    CvFFillSegment* buffer_end = buffer + buffer_size, *head = buffer, *tail = buffer;

    L = R = XMin = XMax = seed.x;

    if( cn == 1 )
    {
        val0[0] = img[L];
        newVal[0] = _newVal[0];

        img[L] = newVal[0];

        while( ++R < roi.width && img[R] == val0[0] )
            img[R] = newVal[0];

        while( --L >= 0 && img[L] == val0[0] )
            img[L] = newVal[0];
    }
    else
    {
        assert( cn == 3 );
        ICV_SET_C3( val0, img + L*3 );
        ICV_SET_C3( newVal, _newVal );

        ICV_SET_C3( img + L*3, newVal );

        while( --L >= 0 && ICV_EQ_C3( img + L*3, val0 ))
            ICV_SET_C3( img + L*3, newVal );

        while( ++R < roi.width && ICV_EQ_C3( img + R*3, val0 ))
            ICV_SET_C3( img + R*3, newVal );
    }

    XMax = --R;
    XMin = ++L;
    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir;
        ICV_POP( YC, L, R, PL, PR, dir );

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        if( region )
        {
            area += R - L + 1;

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        for( k = 0/*(unsigned)(YC - dir) >= (unsigned)roi.height*/; k < 3; k++ )
        {
            dir = data[k][0];
            img = pImage + (YC + dir) * step;
            int left = data[k][1];
            int right = data[k][2];

            if( (unsigned)(YC + dir) >= (unsigned)roi.height )
                continue;

            if( cn == 1 )
                for( i = left; i <= right; i++ )
                {
                    if( (unsigned)i < (unsigned)roi.width && img[i] == val0[0] )
                    {
                        int j = i;
                        img[i] = newVal[0];
                        while( --j >= 0 && img[j] == val0[0] )
                            img[j] = newVal[0];

                        while( ++i < roi.width && img[i] == val0[0] )
                            img[i] = newVal[0];

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
            else
                for( i = left; i <= right; i++ )
                {
                    if( (unsigned)i < (unsigned)roi.width && ICV_EQ_C3( img + i*3, val0 ))
                    {
                        int j = i;
                        ICV_SET_C3( img + i*3, newVal );
                        while( --j >= 0 && ICV_EQ_C3( img + j*3, val0 ))
                            ICV_SET_C3( img + j*3, newVal );

                        while( ++i < roi.width && ICV_EQ_C3( img + i*3, val0 ))
                            ICV_SET_C3( img + i*3, newVal );

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
        }
    }

    if( region )
    {
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
        region->value = cvScalar(newVal[0], newVal[1], newVal[2], 0);
    }
}


/* because all the operations on floats that are done during non-gradient floodfill
   are just copying and comparison on equality,
   we can do the whole op on 32-bit integers instead */
static void
icvFloodFill_32f_CnIR( int* pImage, int step, CvSize roi, CvPoint seed,
                       int* _newVal, CvConnectedComp* region, int flags,
                       CvFFillSegment* buffer, int buffer_size, int cn )
{
    int* img = pImage + (step /= sizeof(pImage[0])) * seed.y;
    int i, L, R;
    int area = 0;
    int val0[] = {0,0,0};
    int newVal[] = {0,0,0};
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    CvFFillSegment* buffer_end = buffer + buffer_size, *head = buffer, *tail = buffer;

    L = R = XMin = XMax = seed.x;

    if( cn == 1 )
    {
        val0[0] = img[L];
        newVal[0] = _newVal[0];

        img[L] = newVal[0];

        while( ++R < roi.width && img[R] == val0[0] )
            img[R] = newVal[0];

        while( --L >= 0 && img[L] == val0[0] )
            img[L] = newVal[0];
    }
    else
    {
        assert( cn == 3 );
        ICV_SET_C3( val0, img + L*3 );
        ICV_SET_C3( newVal, _newVal );

        ICV_SET_C3( img + L*3, newVal );

        while( --L >= 0 && ICV_EQ_C3( img + L*3, val0 ))
            ICV_SET_C3( img + L*3, newVal );

        while( ++R < roi.width && ICV_EQ_C3( img + R*3, val0 ))
            ICV_SET_C3( img + R*3, newVal );
    }

    XMax = --R;
    XMin = ++L;
    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir;
        ICV_POP( YC, L, R, PL, PR, dir );

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        if( region )
        {
            area += R - L + 1;

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        for( k = 0/*(unsigned)(YC - dir) >= (unsigned)roi.height*/; k < 3; k++ )
        {
            dir = data[k][0];
            img = pImage + (YC + dir) * step;
            int left = data[k][1];
            int right = data[k][2];

            if( (unsigned)(YC + dir) >= (unsigned)roi.height )
                continue;

            if( cn == 1 )
                for( i = left; i <= right; i++ )
                {
                    if( (unsigned)i < (unsigned)roi.width && img[i] == val0[0] )
                    {
                        int j = i;
                        img[i] = newVal[0];
                        while( --j >= 0 && img[j] == val0[0] )
                            img[j] = newVal[0];

                        while( ++i < roi.width && img[i] == val0[0] )
                            img[i] = newVal[0];

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
            else
                for( i = left; i <= right; i++ )
                {
                    if( (unsigned)i < (unsigned)roi.width && ICV_EQ_C3( img + i*3, val0 ))
                    {
                        int j = i;
                        ICV_SET_C3( img + i*3, newVal );
                        while( --j >= 0 && ICV_EQ_C3( img + j*3, val0 ))
                            ICV_SET_C3( img + j*3, newVal );

                        while( ++i < roi.width && ICV_EQ_C3( img + i*3, val0 ))
                            ICV_SET_C3( img + i*3, newVal );

                        ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                    }
                }
        }
    }

    if( region )
    {
        Cv32suf v0, v1, v2;
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;
        v0.i = newVal[0]; v1.i = newVal[1]; v2.i = newVal[2];
        region->value = cvScalar( v0.f, v1.f, v2.f );
    }
}

/****************************************************************************************\
*                                   Gradient Floodfill                                   *
\****************************************************************************************/

#define DIFF_INT_C1(p1,p2) ((unsigned)((p1)[0] - (p2)[0] + d_lw[0]) <= interval[0])

#define DIFF_INT_C3(p1,p2) ((unsigned)((p1)[0] - (p2)[0] + d_lw[0])<= interval[0] && \
                            (unsigned)((p1)[1] - (p2)[1] + d_lw[1])<= interval[1] && \
                            (unsigned)((p1)[2] - (p2)[2] + d_lw[2])<= interval[2])

#define DIFF_FLT_C1(p1,p2) (fabs((p1)[0] - (p2)[0] + d_lw[0]) <= interval[0])

#define DIFF_FLT_C3(p1,p2) (fabs((p1)[0] - (p2)[0] + d_lw[0]) <= interval[0] && \
                            fabs((p1)[1] - (p2)[1] + d_lw[1]) <= interval[1] && \
                            fabs((p1)[2] - (p2)[2] + d_lw[2]) <= interval[2])

static void
icvFloodFillGrad_8u_CnIR( uchar* pImage, int step, uchar* pMask, int maskStep,
                           CvSize /*roi*/, CvPoint seed, uchar* _newVal, uchar* _d_lw,
                           uchar* _d_up, CvConnectedComp* region, int flags,
                           CvFFillSegment* buffer, int buffer_size, int cn )
{
    uchar* img = pImage + step*seed.y;
    uchar* mask = (pMask += maskStep + 1) + maskStep*seed.y;
    int i, L, R;
    int area = 0;
    int sum[] = {0,0,0}, val0[] = {0,0,0};
    uchar newVal[] = {0,0,0};
    int d_lw[] = {0,0,0};
    unsigned interval[] = {0,0,0};
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    int fixedRange = flags & CV_FLOODFILL_FIXED_RANGE;
    int fillImage = (flags & CV_FLOODFILL_MASK_ONLY) == 0;
    uchar newMaskVal = (uchar)(flags & 0xff00 ? flags >> 8 : 1);
    CvFFillSegment* buffer_end = buffer + buffer_size, *head = buffer, *tail = buffer;

    L = R = seed.x;
    if( mask[L] )
        return;

    mask[L] = newMaskVal;

    for( i = 0; i < cn; i++ )
    {
        newVal[i] = _newVal[i];
        d_lw[i] = _d_lw[i];
        interval[i] = (unsigned)(_d_up[i] + _d_lw[i]);
        if( fixedRange )
            val0[i] = img[L*cn+i];
    }

    if( cn == 1 )
    {
        if( fixedRange )
        {
            while( !mask[R + 1] && DIFF_INT_C1( img + (R+1), val0 ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_INT_C1( img + (L-1), val0 ))
                mask[--L] = newMaskVal;
        }
        else
        {
            while( !mask[R + 1] && DIFF_INT_C1( img + (R+1), img + R ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_INT_C1( img + (L-1), img + L ))
                mask[--L] = newMaskVal;
        }
    }
    else
    {
        if( fixedRange )
        {
            while( !mask[R + 1] && DIFF_INT_C3( img + (R+1)*3, val0 ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_INT_C3( img + (L-1)*3, val0 ))
                mask[--L] = newMaskVal;
        }
        else
        {
            while( !mask[R + 1] && DIFF_INT_C3( img + (R+1)*3, img + R*3 ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_INT_C3( img + (L-1)*3, img + L*3 ))
                mask[--L] = newMaskVal;
        }
    }

    XMax = R;
    XMin = L;
    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir, curstep;
        ICV_POP( YC, L, R, PL, PR, dir );

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        unsigned length = (unsigned)(R-L);

        if( region )
        {
            area += (int)length + 1;

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        if( cn == 1 )
        {
            for( k = 0; k < 3; k++ )
            {
                dir = data[k][0];
                curstep = dir * step;
                img = pImage + (YC + dir) * step;
                mask = pMask + (YC + dir) * maskStep;
                int left = data[k][1];
                int right = data[k][2];

                if( fixedRange )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_INT_C1( img + i, val0 ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_INT_C1( img + j, val0 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] && DIFF_INT_C1( img + i, val0 ))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else if( !_8_connectivity )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_INT_C1( img + i, img - curstep + i ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_INT_C1( img + j, img + (j+1) ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   (DIFF_INT_C1( img + i, img + (i-1) ) ||
                                   (DIFF_INT_C1( img + i, img + i - curstep) && i <= R)))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else
                    for( i = left; i <= right; i++ )
                    {
                        int idx, val[1];

                        if( !mask[i] &&
                            (((val[0] = img[i],
                            (unsigned)(idx = i-L-1) <= length) &&
                            DIFF_INT_C1( val, img - curstep + (i-1))) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_INT_C1( val, img - curstep + i )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_INT_C1( val, img - curstep + (i+1) ))))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_INT_C1( img + j, img + (j+1) ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   ((val[0] = img[i],
                                   DIFF_INT_C1( val, img + (i-1) )) ||
                                   (((unsigned)(idx = i-L-1) <= length &&
                                   DIFF_INT_C1( val, img - curstep + (i-1) ))) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_INT_C1( val, img - curstep + i )) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_INT_C1( val, img - curstep + (i+1) ))))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
            }

            img = pImage + YC * step;
            if( fillImage )
                for( i = L; i <= R; i++ )
                    img[i] = newVal[0];
            else if( region )
                for( i = L; i <= R; i++ )
                    sum[0] += img[i];
        }
        else
        {
            for( k = 0; k < 3; k++ )
            {
                dir = data[k][0];
                curstep = dir * step;
                img = pImage + (YC + dir) * step;
                mask = pMask + (YC + dir) * maskStep;
                int left = data[k][1];
                int right = data[k][2];

                if( fixedRange )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_INT_C3( img + i*3, val0 ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_INT_C3( img + j*3, val0 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] && DIFF_INT_C3( img + i*3, val0 ))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else if( !_8_connectivity )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_INT_C3( img + i*3, img - curstep + i*3 ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_INT_C3( img + j*3, img + (j+1)*3 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   (DIFF_INT_C3( img + i*3, img + (i-1)*3 ) ||
                                   (DIFF_INT_C3( img + i*3, img + i*3 - curstep) && i <= R)))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else
                    for( i = left; i <= right; i++ )
                    {
                        int idx, val[3];

                        if( !mask[i] &&
                            (((ICV_SET_C3( val, img+i*3 ),
                            (unsigned)(idx = i-L-1) <= length) &&
                            DIFF_INT_C3( val, img - curstep + (i-1)*3 )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_INT_C3( val, img - curstep + i*3 )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_INT_C3( val, img - curstep + (i+1)*3 ))))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_INT_C3( img + j*3, img + (j+1)*3 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   ((ICV_SET_C3( val, img + i*3 ),
                                   DIFF_INT_C3( val, img + (i-1)*3 )) ||
                                   (((unsigned)(idx = i-L-1) <= length &&
                                   DIFF_INT_C3( val, img - curstep + (i-1)*3 ))) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_INT_C3( val, img - curstep + i*3 )) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_INT_C3( val, img - curstep + (i+1)*3 ))))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
            }

            img = pImage + YC * step;
            if( fillImage )
                for( i = L; i <= R; i++ )
                    ICV_SET_C3( img + i*3, newVal );
            else if( region )
                for( i = L; i <= R; i++ )
                {
                    sum[0] += img[i*3];
                    sum[1] += img[i*3+1];
                    sum[2] += img[i*3+2];
                }
        }
    }

    if( region )
    {
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;

        if( fillImage )
            region->value = cvScalar(newVal[0], newVal[1], newVal[2]);
        else
        {
            double iarea = area ? 1./area : 0;
            region->value = cvScalar(sum[0]*iarea, sum[1]*iarea, sum[2]*iarea);
        }
    }
}


static void
icvFloodFillGrad_32f_CnIR( float* pImage, int step, uchar* pMask, int maskStep,
                           CvSize /*roi*/, CvPoint seed, float* _newVal, float* _d_lw,
                           float* _d_up, CvConnectedComp* region, int flags,
                           CvFFillSegment* buffer, int buffer_size, int cn )
{
    float* img = pImage + (step /= sizeof(float))*seed.y;
    uchar* mask = (pMask += maskStep + 1) + maskStep*seed.y;
    int i, L, R;
    int area = 0;
    double sum[] = {0,0,0}, val0[] = {0,0,0};
    float newVal[] = {0,0,0};
    float d_lw[] = {0,0,0};
    float interval[] = {0,0,0};
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int _8_connectivity = (flags & 255) == 8;
    int fixedRange = flags & CV_FLOODFILL_FIXED_RANGE;
    int fillImage = (flags & CV_FLOODFILL_MASK_ONLY) == 0;
    uchar newMaskVal = (uchar)(flags & 0xff00 ? flags >> 8 : 1);
    CvFFillSegment* buffer_end = buffer + buffer_size, *head = buffer, *tail = buffer;

    L = R = seed.x;
    if( mask[L] )
        return;

    mask[L] = newMaskVal;

    for( i = 0; i < cn; i++ )
    {
        newVal[i] = _newVal[i];
        d_lw[i] = 0.5f*(_d_lw[i] - _d_up[i]);
        interval[i] = 0.5f*(_d_lw[i] + _d_up[i]);
        if( fixedRange )
            val0[i] = img[L*cn+i];
    }

    if( cn == 1 )
    {
        if( fixedRange )
        {
            while( !mask[R + 1] && DIFF_FLT_C1( img + (R+1), val0 ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_FLT_C1( img + (L-1), val0 ))
                mask[--L] = newMaskVal;
        }
        else
        {
            while( !mask[R + 1] && DIFF_FLT_C1( img + (R+1), img + R ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_FLT_C1( img + (L-1), img + L ))
                mask[--L] = newMaskVal;
        }
    }
    else
    {
        if( fixedRange )
        {
            while( !mask[R + 1] && DIFF_FLT_C3( img + (R+1)*3, val0 ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_FLT_C3( img + (L-1)*3, val0 ))
                mask[--L] = newMaskVal;
        }
        else
        {
            while( !mask[R + 1] && DIFF_FLT_C3( img + (R+1)*3, img + R*3 ))
                mask[++R] = newMaskVal;

            while( !mask[L - 1] && DIFF_FLT_C3( img + (L-1)*3, img + L*3 ))
                mask[--L] = newMaskVal;
        }
    }

    XMax = R;
    XMin = L;
    ICV_PUSH( seed.y, L, R, R + 1, R, UP );

    while( head != tail )
    {
        int k, YC, PL, PR, dir, curstep;
        ICV_POP( YC, L, R, PL, PR, dir );

        int data[][3] =
        {
            {-dir, L - _8_connectivity, R + _8_connectivity},
            {dir, L - _8_connectivity, PL - 1},
            {dir, PR + 1, R + _8_connectivity}
        };

        unsigned length = (unsigned)(R-L);

        if( region )
        {
            area += (int)length + 1;

            if( XMax < R ) XMax = R;
            if( XMin > L ) XMin = L;
            if( YMax < YC ) YMax = YC;
            if( YMin > YC ) YMin = YC;
        }

        if( cn == 1 )
        {
            for( k = 0; k < 3; k++ )
            {
                dir = data[k][0];
                curstep = dir * step;
                img = pImage + (YC + dir) * step;
                mask = pMask + (YC + dir) * maskStep;
                int left = data[k][1];
                int right = data[k][2];

                if( fixedRange )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_FLT_C1( img + i, val0 ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_FLT_C1( img + j, val0 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] && DIFF_FLT_C1( img + i, val0 ))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else if( !_8_connectivity )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_FLT_C1( img + i, img - curstep + i ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_FLT_C1( img + j, img + (j+1) ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   (DIFF_FLT_C1( img + i, img + (i-1) ) ||
                                   (DIFF_FLT_C1( img + i, img + i - curstep) && i <= R)))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else
                    for( i = left; i <= right; i++ )
                    {
                        int idx;
                        float val[1];

                        if( !mask[i] &&
                            (((val[0] = img[i],
                            (unsigned)(idx = i-L-1) <= length) &&
                            DIFF_FLT_C1( val, img - curstep + (i-1) )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_FLT_C1( val, img - curstep + i )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_FLT_C1( val, img - curstep + (i+1) ))))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_FLT_C1( img + j, img + (j+1) ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   ((val[0] = img[i],
                                   DIFF_FLT_C1( val, img + (i-1) )) ||
                                   (((unsigned)(idx = i-L-1) <= length &&
                                   DIFF_FLT_C1( val, img - curstep + (i-1) ))) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_FLT_C1( val, img - curstep + i )) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_FLT_C1( val, img - curstep + (i+1) ))))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
            }

            img = pImage + YC * step;
            if( fillImage )
                for( i = L; i <= R; i++ )
                    img[i] = newVal[0];
            else if( region )
                for( i = L; i <= R; i++ )
                    sum[0] += img[i];
        }
        else
        {
            for( k = 0; k < 3; k++ )
            {
                dir = data[k][0];
                curstep = dir * step;
                img = pImage + (YC + dir) * step;
                mask = pMask + (YC + dir) * maskStep;
                int left = data[k][1];
                int right = data[k][2];

                if( fixedRange )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_FLT_C3( img + i*3, val0 ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_FLT_C3( img + j*3, val0 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] && DIFF_FLT_C3( img + i*3, val0 ))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else if( !_8_connectivity )
                    for( i = left; i <= right; i++ )
                    {
                        if( !mask[i] && DIFF_FLT_C3( img + i*3, img - curstep + i*3 ))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_FLT_C3( img + j*3, img + (j+1)*3 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   (DIFF_FLT_C3( img + i*3, img + (i-1)*3 ) ||
                                   (DIFF_FLT_C3( img + i*3, img + i*3 - curstep) && i <= R)))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
                else
                    for( i = left; i <= right; i++ )
                    {
                        int idx;
                        float val[3];

                        if( !mask[i] &&
                            (((ICV_SET_C3( val, img+i*3 ),
                            (unsigned)(idx = i-L-1) <= length) &&
                            DIFF_FLT_C3( val, img - curstep + (i-1)*3 )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_FLT_C3( val, img - curstep + i*3 )) ||
                            ((unsigned)(++idx) <= length &&
                            DIFF_FLT_C3( val, img - curstep + (i+1)*3 ))))
                        {
                            int j = i;
                            mask[i] = newMaskVal;
                            while( !mask[--j] && DIFF_FLT_C3( img + j*3, img + (j+1)*3 ))
                                mask[j] = newMaskVal;

                            while( !mask[++i] &&
                                   ((ICV_SET_C3( val, img + i*3 ),
                                   DIFF_FLT_C3( val, img + (i-1)*3 )) ||
                                   (((unsigned)(idx = i-L-1) <= length &&
                                   DIFF_FLT_C3( val, img - curstep + (i-1)*3 ))) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_FLT_C3( val, img - curstep + i*3 )) ||
                                   ((unsigned)(++idx) <= length &&
                                   DIFF_FLT_C3( val, img - curstep + (i+1)*3 ))))
                                mask[i] = newMaskVal;

                            ICV_PUSH( YC + dir, j+1, i-1, L, R, -dir );
                        }
                    }
            }

            img = pImage + YC * step;
            if( fillImage )
                for( i = L; i <= R; i++ )
                    ICV_SET_C3( img + i*3, newVal );
            else if( region )
                for( i = L; i <= R; i++ )
                {
                    sum[0] += img[i*3];
                    sum[1] += img[i*3+1];
                    sum[2] += img[i*3+2];
                }
        }
    }

    if( region )
    {
        region->area = area;
        region->rect.x = XMin;
        region->rect.y = YMin;
        region->rect.width = XMax - XMin + 1;
        region->rect.height = YMax - YMin + 1;

        if( fillImage )
            region->value = cvScalar(newVal[0], newVal[1], newVal[2]);
        else
        {
            double iarea = area ? 1./area : 0;
            region->value = cvScalar(sum[0]*iarea, sum[1]*iarea, sum[2]*iarea);
        }
    }
}


/****************************************************************************************\
*                                    External Functions                                  *
\****************************************************************************************/

typedef  void (*CvFloodFillFunc)(
               void* img, int step, CvSize size, CvPoint seed, void* newval,
               CvConnectedComp* comp, int flags, void* buffer, int buffer_size, int cn );

typedef  void (*CvFloodFillGradFunc)(
               void* img, int step, uchar* mask, int maskStep, CvSize size,
               CvPoint seed, void* newval, void* d_lw, void* d_up, void* ccomp,
               int flags, void* buffer, int buffer_size, int cn );

CV_IMPL void
cvFloodFill( CvArr* arr, CvPoint seed_point,
             CvScalar newVal, CvScalar lo_diff, CvScalar up_diff,
             CvConnectedComp* comp, int flags, CvArr* maskarr )
{
    cv::Ptr<CvMat> tempMask;
    cv::AutoBuffer<CvFFillSegment> buffer;
    
    if( comp )
        memset( comp, 0, sizeof(*comp) );

    int i, type, depth, cn, is_simple;
    int buffer_size, connectivity = flags & 255;
    double nv_buf[4] = {0,0,0,0};
    union { uchar b[4]; float f[4]; } ld_buf, ud_buf;
    CvMat stub, *img = cvGetMat(arr, &stub);
    CvMat maskstub, *mask = (CvMat*)maskarr;
    CvSize size;

    type = CV_MAT_TYPE( img->type );
    depth = CV_MAT_DEPTH(type);
    cn = CV_MAT_CN(type);

    if( connectivity == 0 )
        connectivity = 4;
    else if( connectivity != 4 && connectivity != 8 )
        CV_Error( CV_StsBadFlag, "Connectivity must be 4, 0(=4) or 8" );

    is_simple = mask == 0 && (flags & CV_FLOODFILL_MASK_ONLY) == 0;

    for( i = 0; i < cn; i++ )
    {
        if( lo_diff.val[i] < 0 || up_diff.val[i] < 0 )
            CV_Error( CV_StsBadArg, "lo_diff and up_diff must be non-negative" );
        is_simple &= fabs(lo_diff.val[i]) < DBL_EPSILON && fabs(up_diff.val[i]) < DBL_EPSILON;
    }

    size = cvGetMatSize( img );

    if( (unsigned)seed_point.x >= (unsigned)size.width ||
        (unsigned)seed_point.y >= (unsigned)size.height )
        CV_Error( CV_StsOutOfRange, "Seed point is outside of image" );

    cvScalarToRawData( &newVal, &nv_buf, type, 0 );
    buffer_size = MAX( size.width, size.height )*2;
    buffer.allocate( buffer_size );

    if( is_simple )
    {
        int elem_size = CV_ELEM_SIZE(type);
        const uchar* seed_ptr = img->data.ptr + img->step*seed_point.y + elem_size*seed_point.x;
        CvFloodFillFunc func =
            type == CV_8UC1 || type == CV_8UC3 ? (CvFloodFillFunc)icvFloodFill_8u_CnIR :
            type == CV_32FC1 || type == CV_32FC3 ? (CvFloodFillFunc)icvFloodFill_32f_CnIR : 0;
        if( !func )
            CV_Error( CV_StsUnsupportedFormat, "" );
        // check if the new value is different from the current value at the seed point.
        // if they are exactly the same, use the generic version with mask to avoid infinite loops.
        for( i = 0; i < elem_size; i++ )
            if( seed_ptr[i] != ((uchar*)nv_buf)[i] )
                break;
        if( i < elem_size )
        {
            func( img->data.ptr, img->step, size,
                  seed_point, &nv_buf, comp, flags,
                  buffer, buffer_size, cn );
            return;
        }
    }

    CvFloodFillGradFunc func = 
        type == CV_8UC1 || type == CV_8UC3 ? (CvFloodFillGradFunc)icvFloodFillGrad_8u_CnIR :
        type == CV_32FC1 || type == CV_32FC3 ? (CvFloodFillGradFunc)icvFloodFillGrad_32f_CnIR : 0;
    if( !func )
        CV_Error( CV_StsUnsupportedFormat, "" );
    
    if( !mask )
    {
        /* created mask will be 8-byte aligned */
        tempMask = cvCreateMat( size.height + 2, (size.width + 9) & -8, CV_8UC1 );
        mask = tempMask;
    }
    else
    {
        mask = cvGetMat( mask, &maskstub );
        if( !CV_IS_MASK_ARR( mask ))
            CV_Error( CV_StsBadMask, "" );

        if( mask->width != size.width + 2 || mask->height != size.height + 2 )
            CV_Error( CV_StsUnmatchedSizes, "mask must be 2 pixel wider "
                                   "and 2 pixel taller than filled image" );
    }

    int width = tempMask ? mask->step : size.width + 2;
    uchar* mask_row = mask->data.ptr + mask->step;
    memset( mask_row - mask->step, 1, width );

    for( i = 1; i <= size.height; i++, mask_row += mask->step )
    {
        if( tempMask )
            memset( mask_row, 0, width );
        mask_row[0] = mask_row[size.width+1] = (uchar)1;
    }
    memset( mask_row, 1, width );

    if( depth == CV_8U )
        for( i = 0; i < cn; i++ )
        {
            int t = cvFloor(lo_diff.val[i]);
            ld_buf.b[i] = CV_CAST_8U(t);
            t = cvFloor(up_diff.val[i]);
            ud_buf.b[i] = CV_CAST_8U(t);
        }
    else
        for( i = 0; i < cn; i++ )
        {
            ld_buf.f[i] = (float)lo_diff.val[i];
            ud_buf.f[i] = (float)up_diff.val[i];
        }

    func( img->data.ptr, img->step, mask->data.ptr, mask->step,
          size, seed_point, &nv_buf, ld_buf.f, ud_buf.f,
          comp, flags, buffer, buffer_size, cn );
}


int cv::floodFill( InputOutputArray _image, Point seedPoint,
                   Scalar newVal, Rect* rect,
                   Scalar loDiff, Scalar upDiff, int flags )
{
    CvConnectedComp ccomp;
    CvMat c_image = _image.getMat();
    cvFloodFill(&c_image, seedPoint, newVal, loDiff, upDiff, &ccomp, flags, 0);
    if( rect )
        *rect = ccomp.rect;
    return cvRound(ccomp.area);
}

int cv::floodFill( InputOutputArray _image, InputOutputArray _mask,
                   Point seedPoint, Scalar newVal, Rect* rect, 
                   Scalar loDiff, Scalar upDiff, int flags )
{
    CvConnectedComp ccomp;
    CvMat c_image = _image.getMat(), c_mask = _mask.getMat();
    cvFloodFill(&c_image, seedPoint, newVal, loDiff, upDiff, &ccomp, flags, &c_mask);
    if( rect )
        *rect = ccomp.rect;
    return cvRound(ccomp.area);
}

/* End of file. */
