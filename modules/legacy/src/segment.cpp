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

typedef struct Seg
{
    ushort y;
    ushort l;
    ushort r;
    ushort Prevl;
    ushort Prevr;
    short  fl;
}
Seg;

#define UP 1
#define DOWN -1             

#define PUSH(Y,IL,IR,IPL,IPR,FL) {  stack[StIn].y=(ushort)(Y); \
                                    stack[StIn].l=(ushort)(IL); \
                                    stack[StIn].r=(ushort)(IR); \
                                    stack[StIn].Prevl=(ushort)(IPL); \
                                    stack[StIn].Prevr=(ushort)(IPR); \
                                    stack[StIn].fl=(short)(FL); \
                                    StIn++; }

#define POP(Y,IL,IR,IPL,IPR,FL)  {  StIn--; \
                                    Y=stack[StIn].y; \
                                    IL=stack[StIn].l; \
                                    IR=stack[StIn].r;\
                                    IPL=stack[StIn].Prevl; \
                                    IPR=stack[StIn].Prevr; \
                                    FL=stack[StIn].fl; }


#define DIFF(p1,p2) ((unsigned)((p1)[0] - (p2)[0] + d_lw)<=Interval && \
                     (unsigned)((p1)[1] - (p2)[1] + d_lw)<=Interval && \
                     (unsigned)((p1)[2] - (p2)[2] + d_lw)<=Interval)

/*#define DIFF(p1,p2) (CV_IABS((p1)[0] - (p2)[0]) + \
                     CV_IABS((p1)[1] - (p2)[1]) + \
                     CV_IABS((p1)[2] - (p2)[2]) <=Interval )*/

static CvStatus
icvSegmFloodFill_Stage1( uchar* pImage, int step,
                         uchar* pMask, int maskStep,
                         CvSize /*roi*/, CvPoint seed,
                         int* newVal, int d_lw, int d_up,
                         CvConnectedComp * region,
                         void *pStack )
{
    uchar* img = pImage + step * seed.y;
    uchar* mask = pMask + maskStep * (seed.y + 1);
    unsigned Interval = (unsigned) (d_up + d_lw);
    Seg *stack = (Seg*)pStack;
    int StIn = 0;
    int i, L, R; 
    int area = 0;
    int sum[] = { 0, 0, 0 };
    int XMin, XMax, YMin = seed.y, YMax = seed.y;
    int val0[3];

    L = R = seed.x;
    img = pImage + seed.y*step;
    mask = pMask + seed.y*maskStep;
    mask[L] = 1;

    val0[0] = img[seed.x*3];
    val0[1] = img[seed.x*3 + 1];
    val0[2] = img[seed.x*3 + 2];

    while( DIFF( img + (R+1)*3, /*img + R*3*/val0 ) && !mask[R + 1] )
        mask[++R] = 2;

    while( DIFF( img + (L-1)*3, /*img + L*3*/val0 ) && !mask[L - 1] )
        mask[--L] = 2;

    XMax = R;
    XMin = L;
    PUSH( seed.y, L, R, R + 1, R, UP );

    while( StIn )
    {
        int k, YC, PL, PR, flag/*, curstep*/;

        POP( YC, L, R, PL, PR, flag );

        int data[][3] = { {-flag, L, R}, {flag, L, PL-1}, {flag,PR+1,R}};

        if( XMax < R )
            XMax = R;

        if( XMin > L )
            XMin = L;

        if( YMax < YC )
            YMax = YC;

        if( YMin > YC )
            YMin = YC;

        for( k = 0; k < 3; k++ )
        {
            flag = data[k][0];
            /*curstep = flag * step;*/
            img = pImage + (YC + flag) * step;
            mask = pMask + (YC + flag) * maskStep;
            int left = data[k][1];
            int right = data[k][2];

            for( i = left; i <= right; i++ )
            {
                if( !mask[i] && DIFF( img + i*3, /*img - curstep + i*3*/val0 ))
                {
                    int j = i;
                    mask[i] = 2;
                    while( !mask[j - 1] && DIFF( img + (j - 1)*3, /*img + j*3*/val0 ))
                        mask[--j] = 2;

                    while( !mask[i + 1] &&
                           (DIFF( img + (i+1)*3, /*img + i*3*/val0 ) ||
                           (DIFF( img + (i+1)*3, /*img + (i+1)*3 - curstep*/val0) && i < R)))
                        mask[++i] = 2;

                    PUSH( YC + flag, j, i, L, R, -flag );
                    i++;
                }
            }
        }
        
        img = pImage + YC * step;

        for( i = L; i <= R; i++ )
        {
            sum[0] += img[i*3];
            sum[1] += img[i*3 + 1];
            sum[2] += img[i*3 + 2];
        }

        area += R - L + 1;
    }
    
    region->area = area;
    region->rect.x = XMin;
    region->rect.y = YMin;
    region->rect.width = XMax - XMin + 1;
    region->rect.height = YMax - YMin + 1;
    region->value = cvScalarAll(0);

    {
        double inv_area = area ? 1./area : 0;
        newVal[0] = cvRound( sum[0] * inv_area );
        newVal[1] = cvRound( sum[1] * inv_area );
        newVal[2] = cvRound( sum[2] * inv_area );
    }

    return CV_NO_ERR;
}


#undef PUSH
#undef POP
#undef DIFF


static CvStatus
icvSegmFloodFill_Stage2( uchar* pImage, int step,
                         uchar* pMask, int maskStep,
                         CvSize /*roi*/, int* newVal,
                         CvRect rect )
{
    uchar* img = pImage + step * rect.y + rect.x * 3;
    uchar* mask = pMask + maskStep * rect.y + rect.x;
    uchar uv[] = { (uchar)newVal[0], (uchar)newVal[1], (uchar)newVal[2] };
    int x, y;

    for( y = 0; y < rect.height; y++, img += step, mask += maskStep )
        for( x = 0; x < rect.width; x++ )
            if( mask[x] == 2 )
            {
                mask[x] = 1;
                img[x*3] = uv[0];
                img[x*3+1] = uv[1];
                img[x*3+2] = uv[2];
            }

    return CV_OK;
}

#if 0
static void color_derv( const CvArr* srcArr, CvArr* dstArr, int thresh )
{
    static int tab[] = { 0, 2, 2, 1 };
    
    uchar *src = 0, *dst = 0;
    int dst_step, src_step;
    int x, y;
    CvSize size;

    cvGetRawData( srcArr, (uchar**)&src, &src_step, &size );
    cvGetRawData( dstArr, (uchar**)&dst, &dst_step, 0 );

    memset( dst, 0, size.width*sizeof(dst[0]));
    memset( (uchar*)dst + dst_step*(size.height-1), 0, size.width*sizeof(dst[0]));
    src += 3;

    #define  CV_IABS(a)     (((a) ^ ((a) < 0 ? -1 : 0)) - ((a) < 0 ? -1 : 0))
    
    for( y = 1; y < size.height - 1; y++ )
    {
        src += src_step;
        dst += dst_step;
        uchar* src0 = src;
        
        dst[0] = dst[size.width - 1] = 0;

        for( x = 1; x < size.width - 1; x++, src += 3 )
        {
            /*int d[3];
            int ad[3];
            int f0, f1;
            int val;*/
            int m[3];
            double val;
            //double xx, yy;
            int dh[3];
            int dv[3];
            dh[0] = src[0] - src[-3];
            dv[0] = src[0] - src[-src_step];
            dh[1] = src[1] - src[-2];
            dv[1] = src[1] - src[1-src_step];
            dh[2] = src[2] - src[-1];
            dv[2] = src[2] - src[2-src_step];

            m[0] = dh[0]*dh[0] + dh[1]*dh[1] + dh[2]*dh[2];
            m[2] = dh[0]*dv[0] + dh[1]*dv[1] + dh[2]*dv[2];
            m[1] = dv[0]*dv[0] + dv[1]*dv[1] + dh[2]*dh[2];

            val = (m[0] + m[2]) + 
                sqrt(((double)((double)m[0] - m[2]))*(m[0] - m[2]) + (4.*m[1])*m[1]);

            /*

            xx = m[1];
            yy = v - m[0];
            v /= sqrt(xx*xx + yy*yy) + 1e-7;
            xx *= v;
            yy *= v;
            
            dx[x] = (short)cvRound(xx);
            dy[x] = (short)cvRound(yy);

            //dx[x] = (short)cvRound(v);

            //dx[x] = dy[x] = (short)v;
            d[0] = src[0] - src[-3];
            ad[0] = CV_IABS(d[0]);

            d[1] = src[1] - src[-2];
            ad[1] = CV_IABS(d[1]);

            d[2] = src[2] - src[-1];
            ad[2] = CV_IABS(d[2]);

            f0 = ad[1] > ad[0];
            f1 = ad[2] > ad[f0];  

            val = d[tab[f0*2 + f1]];

            d[0] = src[0] - src[-src_step];
            ad[0] = CV_IABS(d[0]);

            d[1] = src[1] - src[1-src_step];
            ad[1] = CV_IABS(d[1]);

            d[2] = src[2] - src[2-src_step];
            ad[2] = CV_IABS(d[2]);

            f0 = ad[1] > ad[0];
            f1 = ad[2] > ad[f0];  

            dst[x] = (uchar)(val + d[tab[f0*2 + f1]] > thresh ? 255 : 0);*/
            dst[x] = (uchar)(val > thresh);
        }

        src = src0;
    }

}
#endif

const CvPoint icvCodeDeltas[8] =
    { {1, 0}, {1, -1}, {0, -1}, {-1, -1}, {-1, 0}, {-1, 1}, {0, 1}, {1, 1} };

static CvSeq*
icvGetComponent( uchar* img, int step, CvRect rect,
                 CvMemStorage* storage )
{
    const char nbd = 4;
    int  deltas[16];
    int  x, y;
    CvSeq* exterior = 0;
    char* ptr;

    /* initialize local state */
    CV_INIT_3X3_DELTAS( deltas, step, 1 );
    memcpy( deltas + 8, deltas, 8 * sizeof( deltas[0] ));

    ptr = (char*)(img + step*rect.y);
    rect.width += rect.x;
    rect.height += rect.y;

    for( y = rect.y; y < rect.height; y++, ptr += step )
    {
        int prev = ptr[rect.x - 1] & -2;
        
        for( x = rect.x; x < rect.width; x++ )
        {
            int p = ptr[x] & -2;

            //assert( exterior || ((p | prev) & -4) == 0 );

            if( p != prev )
            {
                CvSeq *seq = 0;
                int is_hole = 0;
                CvSeqWriter  writer;
                char  *i0, *i1, *i3, *i4 = 0;
                int  prev_s = -1, s, s_end;
                CvPoint pt = { x, y };

                if( !(prev == 0 && p == 2) )    /* if not external contour */
                {
                    /* check hole */
                    if( p != 0 || prev < 1 )
                    {
                        prev = p;
                        continue;
                    }

                    is_hole = 1;
                    if( !exterior )
                    {
                        assert(0);
                        return 0;
                    }
                }

                cvStartWriteSeq( CV_SEQ_CONTOUR | (is_hole ? CV_SEQ_FLAG_HOLE : 0),
                                 sizeof(CvContour), sizeof(CvPoint), storage, &writer );
                s_end = s = is_hole ? 0 : 4;
                i0 = ptr + x - is_hole;

                do
                {
                    s = (s - 1) & 7;
                    i1 = i0 + deltas[s];
                    if( (*i1 & -2) != 0 )
                        break;
                }
                while( s != s_end );

                if( s == s_end )            /* single pixel domain */
                {
                    *i0 = (char) (nbd | -128);
                    CV_WRITE_SEQ_ELEM( pt, writer );
                }
                else
                {
                    i3 = i0;
                    prev_s = s ^ 4;

                    /* follow border */
                    for( ;; )
                    {
                        s_end = s;

                        for( ;; )
                        {
                            i4 = i3 + deltas[++s];
                            if( (*i4 & -2) != 0 )
                                break;
                        }
                        s &= 7;

                        /* check "right" bound */
                        if( (unsigned) (s - 1) < (unsigned) s_end )
                        {
                            *i3 = (char) (nbd | -128);
                        }
                        else if( *i3 > 0 )
                        {
                            *i3 = nbd;
                        }

                        if( s != prev_s )
                        {
                            CV_WRITE_SEQ_ELEM( pt, writer );
                            prev_s = s;
                        }

                        pt.x += icvCodeDeltas[s].x;
                        pt.y += icvCodeDeltas[s].y;

                        if( i4 == i0 && i3 == i1 )
                            break;

                        i3 = i4;
                        s = (s + 4) & 7;
                    }                       /* end of border following loop */
                }

                seq = cvEndWriteSeq( &writer );
                cvContourBoundingRect( seq, 1 );

                if( !is_hole )
                    exterior = seq;
                else
                {
                    seq->v_prev = exterior;
                    seq->h_next = exterior->v_next;
                    if( seq->h_next )
                        seq->h_next->h_prev = seq;
                    exterior->v_next = seq;
                }

                prev = ptr[x] & -2;
            }
        }
    }

    return exterior;
}



CV_IMPL CvSeq*
cvSegmentImage( const CvArr* srcarr, CvArr* dstarr,
                double canny_threshold,
                double ffill_threshold,
                CvMemStorage* storage )
{
    CvSeq* root = 0;
    CvMat* gray = 0;
    CvMat* canny = 0;
    //CvMat* temp = 0;
    void* stack = 0;
    
    CV_FUNCNAME( "cvSegmentImage" );

    __BEGIN__;

    CvMat srcstub, *src;
    CvMat dststub, *dst;
    CvMat* mask;
    CvSize size;
    CvPoint pt;
    int ffill_lw_up = cvRound( fabs(ffill_threshold) );
    CvSeq* prev_seq = 0;

    CV_CALL( src = cvGetMat( srcarr, &srcstub ));
    CV_CALL( dst = cvGetMat( dstarr, &dststub ));

    size = cvGetSize( src );

    CV_CALL( gray = cvCreateMat( size.height, size.width, CV_8UC1 ));
    CV_CALL( canny = cvCreateMat( size.height, size.width, CV_8UC1 ));
    //CV_CALL( temp = cvCreateMat( size.height/2, size.width/2, CV_8UC3 ));

    CV_CALL( stack = cvAlloc( size.width * size.height * sizeof(Seg)));

    cvCvtColor( src, gray, CV_BGR2GRAY );
    cvCanny( gray, canny, 0/*canny_threshold*0.4*/, canny_threshold, 3 );
    cvThreshold( canny, canny, 1, 1, CV_THRESH_BINARY );
    //cvZero( canny );
    //color_derv( src, canny, canny_threshold );

    //cvPyrDown( src, temp );
    //cvPyrUp( temp, dst );

    //src = dst;
    mask = canny; // a new name for new role

    // make a non-zero border.
    cvRectangle( mask, cvPoint(0,0), cvPoint(size.width-1,size.height-1), cvScalarAll(1), 1 );

    for( pt.y = 0; pt.y < size.height; pt.y++ )
    {
        for( pt.x = 0; pt.x < size.width; pt.x++ )
        {
            if( mask->data.ptr[mask->step*pt.y + pt.x] == 0 )
            {
                CvConnectedComp region;
                int avgVal[3] = { 0, 0, 0 };
                
                icvSegmFloodFill_Stage1( src->data.ptr, src->step,
                                         mask->data.ptr, mask->step,
                                         size, pt, avgVal,
                                         ffill_lw_up, ffill_lw_up,
                                         &region, stack );

                /*avgVal[0] = (avgVal[0] + 15) & -32;
                if( avgVal[0] > 255 )
                    avgVal[0] = 255;
                avgVal[1] = (avgVal[1] + 15) & -32;
                if( avgVal[1] > 255 )
                    avgVal[1] = 255;
                avgVal[2] = (avgVal[2] + 15) & -32;
                if( avgVal[2] > 255 )
                    avgVal[2] = 255;*/

                if( storage )
                {
                    CvSeq* tmpseq = icvGetComponent( mask->data.ptr, mask->step,
                                                     region.rect, storage );
                    if( tmpseq != 0 )
                    {
                        ((CvContour*)tmpseq)->color = avgVal[0] + (avgVal[1] << 8) + (avgVal[2] << 16);
                        tmpseq->h_prev = prev_seq;
                        if( prev_seq )
                            prev_seq->h_next = tmpseq;
                        else
                            root = tmpseq;
                        prev_seq = tmpseq;
                    }
                }

                icvSegmFloodFill_Stage2( dst->data.ptr, dst->step,
                                         mask->data.ptr, mask->step,
                                         size, avgVal,
                                         region.rect );
            }
        }
    }

    __END__;

    //cvReleaseMat( &temp );
    cvReleaseMat( &gray );
    cvReleaseMat( &canny );
    cvFree( &stack );

    return root;
}

/* End of file. */
