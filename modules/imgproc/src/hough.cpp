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
#include "_list.h"

#define halfPi ((float)(CV_PI*0.5))
#define Pi     ((float)CV_PI)
#define a0  0 /*-4.172325e-7f*/   /*(-(float)0x7)/((float)0x1000000); */
#define a1 1.000025f        /*((float)0x1922253)/((float)0x1000000)*2/Pi; */
#define a2 -2.652905e-4f    /*(-(float)0x2ae6)/((float)0x1000000)*4/(Pi*Pi); */
#define a3 -0.165624f       /*(-(float)0xa45511)/((float)0x1000000)*8/(Pi*Pi*Pi); */
#define a4 -1.964532e-3f    /*(-(float)0x30fd3)/((float)0x1000000)*16/(Pi*Pi*Pi*Pi); */
#define a5 1.02575e-2f      /*((float)0x191cac)/((float)0x1000000)*32/(Pi*Pi*Pi*Pi*Pi); */
#define a6 -9.580378e-4f    /*(-(float)0x3af27)/((float)0x1000000)*64/(Pi*Pi*Pi*Pi*Pi*Pi); */

#define _sin(x) ((((((a6*(x) + a5)*(x) + a4)*(x) + a3)*(x) + a2)*(x) + a1)*(x) + a0)
#define _cos(x) _sin(halfPi - (x))

/****************************************************************************************\
*                               Classical Hough Transform                                *
\****************************************************************************************/

typedef struct CvLinePolar
{
    float rho;
    float angle;
}
CvLinePolar;

/*=====================================================================================*/

#define hough_cmp_gt(l1,l2) (aux[l1] > aux[l2])

static CV_IMPLEMENT_QSORT_EX( icvHoughSortDescent32s, int, hough_cmp_gt, const int* )

/*
Here image is an input raster;
step is it's step; size characterizes it's ROI;
rho and theta are discretization steps (in pixels and radians correspondingly).
threshold is the minimum number of pixels in the feature for it
to be a candidate for line. lines is the output
array of (rho, theta) pairs. linesMax is the buffer size (number of pairs).
Functions return the actual number of found lines.
*/
static void
icvHoughLinesStandard( const CvMat* img, float rho, float theta,
                       int threshold, CvSeq *lines, int linesMax )
{
    cv::AutoBuffer<int> _accum, _sort_buf;
    cv::AutoBuffer<float> _tabSin, _tabCos;

    const uchar* image;
    int step, width, height;
    int numangle, numrho;
    int total = 0;
    float ang;
    int r, n;
    int i, j;
    float irho = 1 / rho;
    double scale;

    CV_Assert( CV_IS_MAT(img) && CV_MAT_TYPE(img->type) == CV_8UC1 );

    image = img->data.ptr;
    step = img->step;
    width = img->cols;
    height = img->rows;

    numangle = cvRound(CV_PI / theta);
    numrho = cvRound(((width + height) * 2 + 1) / rho);

    _accum.allocate((numangle+2) * (numrho+2));
    _sort_buf.allocate(numangle * numrho);
    _tabSin.allocate(numangle);
    _tabCos.allocate(numangle);
    int *accum = _accum, *sort_buf = _sort_buf;
    float *tabSin = _tabSin, *tabCos = _tabCos;
    
    memset( accum, 0, sizeof(accum[0]) * (numangle+2) * (numrho+2) );

    for( ang = 0, n = 0; n < numangle; ang += theta, n++ )
    {
        tabSin[n] = (float)(sin(ang) * irho);
        tabCos[n] = (float)(cos(ang) * irho);
    }

    // stage 1. fill accumulator
    for( i = 0; i < height; i++ )
        for( j = 0; j < width; j++ )
        {
            if( image[i * step + j] != 0 )
                for( n = 0; n < numangle; n++ )
                {
                    r = cvRound( j * tabCos[n] + i * tabSin[n] );
                    r += (numrho - 1) / 2;
                    accum[(n+1) * (numrho+2) + r+1]++;
                }
        }

    // stage 2. find local maximums
    for( r = 0; r < numrho; r++ )
        for( n = 0; n < numangle; n++ )
        {
            int base = (n+1) * (numrho+2) + r+1;
            if( accum[base] > threshold &&
                accum[base] > accum[base - 1] && accum[base] >= accum[base + 1] &&
                accum[base] > accum[base - numrho - 2] && accum[base] >= accum[base + numrho + 2] )
                sort_buf[total++] = base;
        }

    // stage 3. sort the detected lines by accumulator value
    icvHoughSortDescent32s( sort_buf, total, accum );

    // stage 4. store the first min(total,linesMax) lines to the output buffer
    linesMax = MIN(linesMax, total);
    scale = 1./(numrho+2);
    for( i = 0; i < linesMax; i++ )
    {
        CvLinePolar line;
        int idx = sort_buf[i];
        int n = cvFloor(idx*scale) - 1;
        int r = idx - (n+1)*(numrho+2) - 1;
        line.rho = (r - (numrho - 1)*0.5f) * rho;
        line.angle = n * theta;
        cvSeqPush( lines, &line );
    }
}


/****************************************************************************************\
*                     Multi-Scale variant of Classical Hough Transform                   *
\****************************************************************************************/

#if defined _MSC_VER && _MSC_VER >= 1200
#pragma warning( disable: 4714 )
#endif

//DECLARE_AND_IMPLEMENT_LIST( _index, h_ );
IMPLEMENT_LIST( _index, h_ )

static void
icvHoughLinesSDiv( const CvMat* img,
                   float rho, float theta, int threshold,
                   int srn, int stn,
                   CvSeq* lines, int linesMax )
{
    std::vector<uchar> _caccum, _buffer;
    std::vector<float> _sinTable;
    std::vector<int> _x, _y;
    float* sinTable;
    int *x, *y;
    uchar *caccum, *buffer;
    _CVLIST* list = 0;

#define _POINT(row, column)\
    (image_src[(row)*step+(column)])

    uchar *mcaccum = 0;
    int rn, tn;                 /* number of rho and theta discrete values */
    int index, i;
    int ri, ti, ti1, ti0;
    int row, col;
    float r, t;                 /* Current rho and theta */
    float rv;                   /* Some temporary rho value */
    float irho;
    float itheta;
    float srho, stheta;
    float isrho, istheta;

    const uchar* image_src;
    int w, h, step;
    int fn = 0;
    float xc, yc;

    const float d2r = (float)(Pi / 180);
    int sfn = srn * stn;
    int fi;
    int count;
    int cmax = 0;

    CVPOS pos;
    _index *pindex;
    _index vi;

    CV_Assert( CV_IS_MAT(img) && CV_MAT_TYPE(img->type) == CV_8UC1 );
    CV_Assert( linesMax > 0 && rho > 0 && theta > 0 );

    threshold = MIN( threshold, 255 );

    image_src = img->data.ptr;
    step = img->step;
    w = img->cols;
    h = img->rows;

    irho = 1 / rho;
    itheta = 1 / theta;
    srho = rho / srn;
    stheta = theta / stn;
    isrho = 1 / srho;
    istheta = 1 / stheta;

    rn = cvFloor( sqrt( (double)w * w + (double)h * h ) * irho );
    tn = cvFloor( 2 * Pi * itheta );

    list = h_create_list__index( linesMax < 1000 ? linesMax : 1000 );
    vi.value = threshold;
    vi.rho = -1;
    h_add_head__index( list, &vi );

    /* Precalculating sin */
    _sinTable.resize( 5 * tn * stn );
    sinTable = &_sinTable[0];
    
    for( index = 0; index < 5 * tn * stn; index++ )
        sinTable[index] = (float)cos( stheta * index * 0.2f );

    _caccum.resize(rn * tn);
    caccum = &_caccum[0];
    memset( caccum, 0, rn * tn * sizeof( caccum[0] ));

    /* Counting all feature pixels */
    for( row = 0; row < h; row++ )
        for( col = 0; col < w; col++ )
            fn += _POINT( row, col ) != 0;

    _x.resize(fn);
    _y.resize(fn);
    x = &_x[0];
    y = &_y[0];

    /* Full Hough Transform (it's accumulator update part) */
    fi = 0;
    for( row = 0; row < h; row++ )
    {
        for( col = 0; col < w; col++ )
        {
            if( _POINT( row, col ))
            {
                int halftn;
                float r0;
                float scale_factor;
                int iprev = -1;
                float phi, phi1;
                float theta_it;     /* Value of theta for iterating */

                /* Remember the feature point */
                x[fi] = col;
                y[fi] = row;
                fi++;

                yc = (float) row + 0.5f;
                xc = (float) col + 0.5f;

                /* Update the accumulator */
                t = (float) fabs( cvFastArctan( yc, xc ) * d2r );
                r = (float) sqrt( (double)xc * xc + (double)yc * yc );
                r0 = r * irho;
                ti0 = cvFloor( (t + Pi / 2) * itheta );

                caccum[ti0]++;

                theta_it = rho / r;
                theta_it = theta_it < theta ? theta_it : theta;
                scale_factor = theta_it * itheta;
                halftn = cvFloor( Pi / theta_it );
                for( ti1 = 1, phi = theta_it - halfPi, phi1 = (theta_it + t) * itheta;
                     ti1 < halftn; ti1++, phi += theta_it, phi1 += scale_factor )
                {
                    rv = r0 * _cos( phi );
                    i = cvFloor( rv ) * tn;
                    i += cvFloor( phi1 );
                    assert( i >= 0 );
                    assert( i < rn * tn );
                    caccum[i] = (uchar) (caccum[i] + ((i ^ iprev) != 0));
                    iprev = i;
                    if( cmax < caccum[i] )
                        cmax = caccum[i];
                }
            }
        }
    }

    /* Starting additional analysis */
    count = 0;
    for( ri = 0; ri < rn; ri++ )
    {
        for( ti = 0; ti < tn; ti++ )
        {
            if( caccum[ri * tn + ti > threshold] )
            {
                count++;
            }
        }
    }

    if( count * 100 > rn * tn )
    {
        icvHoughLinesStandard( img, rho, theta, threshold, lines, linesMax );
        return;
    }

    _buffer.resize(srn * stn + 2);
    buffer = &_buffer[0];
    mcaccum = buffer + 1;

    count = 0;
    for( ri = 0; ri < rn; ri++ )
    {
        for( ti = 0; ti < tn; ti++ )
        {
            if( caccum[ri * tn + ti] > threshold )
            {
                count++;
                memset( mcaccum, 0, sfn * sizeof( uchar ));

                for( index = 0; index < fn; index++ )
                {
                    int ti2;
                    float r0;

                    yc = (float) y[index] + 0.5f;
                    xc = (float) x[index] + 0.5f;

                    /* Update the accumulator */
                    t = (float) fabs( cvFastArctan( yc, xc ) * d2r );
                    r = (float) sqrt( (double)xc * xc + (double)yc * yc ) * isrho;
                    ti0 = cvFloor( (t + Pi * 0.5f) * istheta );
                    ti2 = (ti * stn - ti0) * 5;
                    r0 = (float) ri *srn;

                    for( ti1 = 0 /*, phi = ti*theta - Pi/2 - t */ ; ti1 < stn; ti1++, ti2 += 5
                         /*phi += stheta */  )
                    {
                        /*rv = r*_cos(phi) - r0; */
                        rv = r * sinTable[(int) (abs( ti2 ))] - r0;
                        i = cvFloor( rv ) * stn + ti1;

                        i = CV_IMAX( i, -1 );
                        i = CV_IMIN( i, sfn );
                        mcaccum[i]++;
                        assert( i >= -1 );
                        assert( i <= sfn );
                    }
                }

                /* Find peaks in maccum... */
                for( index = 0; index < sfn; index++ )
                {
                    i = 0;
                    pos = h_get_tail_pos__index( list );
                    if( h_get_prev__index( &pos )->value < mcaccum[index] )
                    {
                        vi.value = mcaccum[index];
                        vi.rho = index / stn * srho + ri * rho;
                        vi.theta = index % stn * stheta + ti * theta - halfPi;
                        while( h_is_pos__index( pos ))
                        {
                            if( h_get__index( pos )->value > mcaccum[index] )
                            {
                                h_insert_after__index( list, pos, &vi );
                                if( h_get_count__index( list ) > linesMax )
                                {
                                    h_remove_tail__index( list );
                                }
                                break;
                            }
                            h_get_prev__index( &pos );
                        }
                        if( !h_is_pos__index( pos ))
                        {
                            h_add_head__index( list, &vi );
                            if( h_get_count__index( list ) > linesMax )
                            {
                                h_remove_tail__index( list );
                            }
                        }
                    }
                }
            }
        }
    }

    pos = h_get_head_pos__index( list );
    if( h_get_count__index( list ) == 1 )
    {
        if( h_get__index( pos )->rho < 0 )
        {
            h_clear_list__index( list );
        }
    }
    else
    {
        while( h_is_pos__index( pos ))
        {
            CvLinePolar line;
            pindex = h_get__index( pos );
            if( pindex->rho < 0 )
            {
                /* This should be the last element... */
                h_get_next__index( &pos );
                assert( !h_is_pos__index( pos ));
                break;
            }
            line.rho = pindex->rho;
            line.angle = pindex->theta;
            cvSeqPush( lines, &line );

            if( lines->total >= linesMax )
                break;
            h_get_next__index( &pos );
        }
    }
    
    h_destroy_list__index(list);
}


/****************************************************************************************\
*                              Probabilistic Hough Transform                             *
\****************************************************************************************/

static void
icvHoughLinesProbabalistic( CvMat* image,
                            float rho, float theta, int threshold,
                            int lineLength, int lineGap,
                            CvSeq *lines, int linesMax )
{
    cv::Mat accum, mask;
    cv::vector<float> trigtab;
    cv::MemStorage storage(cvCreateMemStorage(0));

    CvSeq* seq;
    CvSeqWriter writer;
    int width, height;
    int numangle, numrho;
    float ang;
    int r, n, count;
    CvPoint pt;
    float irho = 1 / rho;
    CvRNG rng = cvRNG(-1);
    const float* ttab;
    uchar* mdata0;

    CV_Assert( CV_IS_MAT(image) && CV_MAT_TYPE(image->type) == CV_8UC1 );

    width = image->cols;
    height = image->rows;

    numangle = cvRound(CV_PI / theta);
    numrho = cvRound(((width + height) * 2 + 1) / rho);

    accum.create( numangle, numrho, CV_32SC1 );
    mask.create( height, width, CV_8UC1 );
    trigtab.resize(numangle*2);
    accum = cv::Scalar(0);

    for( ang = 0, n = 0; n < numangle; ang += theta, n++ )
    {
        trigtab[n*2] = (float)(cos(ang) * irho);
        trigtab[n*2+1] = (float)(sin(ang) * irho);
    }
    ttab = &trigtab[0];
    mdata0 = mask.data;

    cvStartWriteSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage, &writer );

    // stage 1. collect non-zero image points
    for( pt.y = 0, count = 0; pt.y < height; pt.y++ )
    {
        const uchar* data = image->data.ptr + pt.y*image->step;
        uchar* mdata = mdata0 + pt.y*width;
        for( pt.x = 0; pt.x < width; pt.x++ )
        {
            if( data[pt.x] )
            {
                mdata[pt.x] = (uchar)1;
                CV_WRITE_SEQ_ELEM( pt, writer );
            }
            else
                mdata[pt.x] = 0;
        }
    }

    seq = cvEndWriteSeq( &writer );
    count = seq->total;

    // stage 2. process all the points in random order
    for( ; count > 0; count-- )
    {
        // choose random point out of the remaining ones
        int idx = cvRandInt(&rng) % count;
        int max_val = threshold-1, max_n = 0;
        CvPoint* pt = (CvPoint*)cvGetSeqElem( seq, idx );
        CvPoint line_end[2] = {{0,0}, {0,0}};
        float a, b;
        int* adata = (int*)accum.data;
        int i, j, k, x0, y0, dx0, dy0, xflag;
        int good_line;
        const int shift = 16;

        i = pt->y;
        j = pt->x;

        // "remove" it by overriding it with the last element
        *pt = *(CvPoint*)cvGetSeqElem( seq, count-1 );

        // check if it has been excluded already (i.e. belongs to some other line)
        if( !mdata0[i*width + j] )
            continue;

        // update accumulator, find the most probable line
        for( n = 0; n < numangle; n++, adata += numrho )
        {
            r = cvRound( j * ttab[n*2] + i * ttab[n*2+1] );
            r += (numrho - 1) / 2;
            int val = ++adata[r];
            if( max_val < val )
            {
                max_val = val;
                max_n = n;
            }
        }

        // if it is too "weak" candidate, continue with another point
        if( max_val < threshold )
            continue;

        // from the current point walk in each direction
        // along the found line and extract the line segment
        a = -ttab[max_n*2+1];
        b = ttab[max_n*2];
        x0 = j;
        y0 = i;
        if( fabs(a) > fabs(b) )
        {
            xflag = 1;
            dx0 = a > 0 ? 1 : -1;
            dy0 = cvRound( b*(1 << shift)/fabs(a) );
            y0 = (y0 << shift) + (1 << (shift-1));
        }
        else
        {
            xflag = 0;
            dy0 = b > 0 ? 1 : -1;
            dx0 = cvRound( a*(1 << shift)/fabs(b) );
            x0 = (x0 << shift) + (1 << (shift-1));
        }

        for( k = 0; k < 2; k++ )
        {
            int gap = 0, x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                if( j1 < 0 || j1 >= width || i1 < 0 || i1 >= height )
                    break;

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    gap = 0;
                    line_end[k].y = i1;
                    line_end[k].x = j1;
                }
                else if( ++gap > lineGap )
                    break;
            }
        }

        good_line = abs(line_end[1].x - line_end[0].x) >= lineLength ||
                    abs(line_end[1].y - line_end[0].y) >= lineLength;

        for( k = 0; k < 2; k++ )
        {
            int x = x0, y = y0, dx = dx0, dy = dy0;

            if( k > 0 )
                dx = -dx, dy = -dy;

            // walk along the line using fixed-point arithmetics,
            // stop at the image border or in case of too big gap
            for( ;; x += dx, y += dy )
            {
                uchar* mdata;
                int i1, j1;

                if( xflag )
                {
                    j1 = x;
                    i1 = y >> shift;
                }
                else
                {
                    j1 = x >> shift;
                    i1 = y;
                }

                mdata = mdata0 + i1*width + j1;

                // for each non-zero point:
                //    update line end,
                //    clear the mask element
                //    reset the gap
                if( *mdata )
                {
                    if( good_line )
                    {
                        adata = (int*)accum.data;
                        for( n = 0; n < numangle; n++, adata += numrho )
                        {
                            r = cvRound( j1 * ttab[n*2] + i1 * ttab[n*2+1] );
                            r += (numrho - 1) / 2;
                            adata[r]--;
                        }
                    }
                    *mdata = 0;
                }

                if( i1 == line_end[k].y && j1 == line_end[k].x )
                    break;
            }
        }

        if( good_line )
        {
            CvRect lr = { line_end[0].x, line_end[0].y, line_end[1].x, line_end[1].y };
            cvSeqPush( lines, &lr );
            if( lines->total >= linesMax )
                return;
        }
    }
}

/* Wrapper function for standard hough transform */
CV_IMPL CvSeq*
cvHoughLines2( CvArr* src_image, void* lineStorage, int method,
               double rho, double theta, int threshold,
               double param1, double param2 )
{
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* lines = 0;
    CvSeq lines_header;
    CvSeqBlock lines_block;
    int lineType, elemSize;
    int linesMax = INT_MAX;
    int iparam1, iparam2;

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !lineStorage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( rho <= 0 || theta <= 0 || threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "rho, theta and threshold must be positive" );

    if( method != CV_HOUGH_PROBABILISTIC )
    {
        lineType = CV_32FC2;
        elemSize = sizeof(float)*2;
    }
    else
    {
        lineType = CV_32SC4;
        elemSize = sizeof(int)*4;
    }

    if( CV_IS_STORAGE( lineStorage ))
    {
        lines = cvCreateSeq( lineType, sizeof(CvSeq), elemSize, (CvMemStorage*)lineStorage );
    }
    else if( CV_IS_MAT( lineStorage ))
    {
        mat = (CvMat*)lineStorage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        if( CV_MAT_TYPE( mat->type ) != lineType )
            CV_Error( CV_StsBadArg,
            "The destination matrix data type is inappropriate, see the manual" );

        lines = cvMakeSeqHeaderForArray( lineType, sizeof(CvSeq), elemSize, mat->data.ptr,
                                         mat->rows + mat->cols - 1, &lines_header, &lines_block );
        linesMax = lines->total;
        cvClearSeq( lines );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );
    
    iparam1 = cvRound(param1);
    iparam2 = cvRound(param2);

    switch( method )
    {
    case CV_HOUGH_STANDARD:
          icvHoughLinesStandard( img, (float)rho,
                (float)theta, threshold, lines, linesMax );
          break;
    case CV_HOUGH_MULTI_SCALE:
          icvHoughLinesSDiv( img, (float)rho, (float)theta,
                threshold, iparam1, iparam2, lines, linesMax );
          break;
    case CV_HOUGH_PROBABILISTIC:
          icvHoughLinesProbabalistic( img, (float)rho, (float)theta,
                threshold, iparam1, iparam2, lines, linesMax );
          break;
    default:
        CV_Error( CV_StsBadArg, "Unrecognized method id" );
    }

    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = lines->total;
        else
            mat->rows = lines->total;
    }
    else
        result = lines;

    return result;
}


/****************************************************************************************\
*                                     Circle Detection                                   *
\****************************************************************************************/

static void
icvHoughCirclesGradient( CvMat* img, float dp, float min_dist,
                         int min_radius, int max_radius,
                         int canny_threshold, int acc_threshold,
                         CvSeq* circles, int circles_max )
{
    const int SHIFT = 10, ONE = 1 << SHIFT, R_THRESH = 30;
    cv::Ptr<CvMat> dx, dy;
    cv::Ptr<CvMat> edges, accum, dist_buf;
    std::vector<int> sort_buf;
    cv::Ptr<CvMemStorage> storage;

    int x, y, i, j, center_count, nz_count;
    int rows, cols, arows, acols;
    int astep, *adata;
    float* ddata;
    CvSeq *nz, *centers;
    float idp, dr;
    CvSeqReader reader;

    edges = cvCreateMat( img->rows, img->cols, CV_8UC1 );
    cvCanny( img, edges, MAX(canny_threshold/5,1), canny_threshold, 3 );

    dx = cvCreateMat( img->rows, img->cols, CV_16SC1 );
    dy = cvCreateMat( img->rows, img->cols, CV_16SC1 );
    cvSobel( img, dx, 1, 0, 3 );
    cvSobel( img, dy, 0, 1, 3 );

    if( dp < 1.f )
        dp = 1.f;
    idp = 1.f/dp;
    accum = cvCreateMat( cvCeil(img->rows*idp)+2, cvCeil(img->cols*idp)+2, CV_32SC1 );
    cvZero(accum);

    storage = cvCreateMemStorage();
    nz = cvCreateSeq( CV_32SC2, sizeof(CvSeq), sizeof(CvPoint), storage );
    centers = cvCreateSeq( CV_32SC1, sizeof(CvSeq), sizeof(int), storage );

    rows = img->rows;
    cols = img->cols;
    arows = accum->rows - 2;
    acols = accum->cols - 2;
    adata = accum->data.i;
    astep = accum->step/sizeof(adata[0]);

    for( y = 0; y < rows; y++ )
    {
        const uchar* edges_row = edges->data.ptr + y*edges->step;
        const short* dx_row = (const short*)(dx->data.ptr + y*dx->step);
        const short* dy_row = (const short*)(dy->data.ptr + y*dy->step);

        for( x = 0; x < cols; x++ )
        {
            float vx, vy;
            int sx, sy, x0, y0, x1, y1, r, k;
            CvPoint pt;

            vx = dx_row[x];
            vy = dy_row[x];

            if( !edges_row[x] || (vx == 0 && vy == 0) )
                continue;

            float mag = sqrt(vx*vx+vy*vy);
            assert( mag >= 1 );
            sx = cvRound((vx*idp)*ONE/mag);
            sy = cvRound((vy*idp)*ONE/mag);

            x0 = cvRound((x*idp)*ONE);
            y0 = cvRound((y*idp)*ONE);

            for( k = 0; k < 2; k++ )
            {
                x1 = x0 + min_radius * sx;
                y1 = y0 + min_radius * sy;

                for( r = min_radius; r <= max_radius; x1 += sx, y1 += sy, r++ )
                {
                    int x2 = x1 >> SHIFT, y2 = y1 >> SHIFT;
                    if( (unsigned)x2 >= (unsigned)acols ||
                        (unsigned)y2 >= (unsigned)arows )
                        break;
                    adata[y2*astep + x2]++;
                }

                sx = -sx; sy = -sy;
            }

            pt.x = x; pt.y = y;
            cvSeqPush( nz, &pt );
        }
    }

    nz_count = nz->total;
    if( !nz_count )
        return;

    for( y = 1; y < arows - 1; y++ )
    {
        for( x = 1; x < acols - 1; x++ )
        {
            int base = y*(acols+2) + x;
            if( adata[base] > acc_threshold &&
                adata[base] > adata[base-1] && adata[base] > adata[base+1] &&
                adata[base] > adata[base-acols-2] && adata[base] > adata[base+acols+2] )
                cvSeqPush(centers, &base);
        }
    }

    center_count = centers->total;
    if( !center_count )
        return;

    sort_buf.resize( MAX(center_count,nz_count) );
    cvCvtSeqToArray( centers, &sort_buf[0] );

    icvHoughSortDescent32s( &sort_buf[0], center_count, adata );
    cvClearSeq( centers );
    cvSeqPushMulti( centers, &sort_buf[0], center_count );

    dist_buf = cvCreateMat( 1, nz_count, CV_32FC1 );
    ddata = dist_buf->data.fl;

    dr = dp;
    min_dist = MAX( min_dist, dp );
    min_dist *= min_dist;

    for( i = 0; i < centers->total; i++ )
    {
        int ofs = *(int*)cvGetSeqElem( centers, i );
        y = ofs/(acols+2) - 1;
        x = ofs - (y+1)*(acols+2) - 1;
        float cx = (float)(x*dp), cy = (float)(y*dp);
        int start_idx = nz_count - 1;
        float start_dist, dist_sum;
        float r_best = 0, c[3];
        int max_count = R_THRESH;

        for( j = 0; j < circles->total; j++ )
        {
            float* c = (float*)cvGetSeqElem( circles, j );
            if( (c[0] - cx)*(c[0] - cx) + (c[1] - cy)*(c[1] - cy) < min_dist )
                break;
        }

        if( j < circles->total )
            continue;

        cvStartReadSeq( nz, &reader );
        for( j = 0; j < nz_count; j++ )
        {
            CvPoint pt;
            float _dx, _dy;
            CV_READ_SEQ_ELEM( pt, reader );
            _dx = cx - pt.x; _dy = cy - pt.y;
            ddata[j] = _dx*_dx + _dy*_dy;
            sort_buf[j] = j;
        }

        cvPow( dist_buf, dist_buf, 0.5 );
        icvHoughSortDescent32s( &sort_buf[0], nz_count, (int*)ddata );

        dist_sum = start_dist = ddata[sort_buf[nz_count-1]];
        for( j = nz_count - 2; j >= 0; j-- )
        {
            float d = ddata[sort_buf[j]];

            if( d > max_radius )
                break;

            if( d - start_dist > dr )
            {
                float r_cur = ddata[sort_buf[(j + start_idx)/2]];
                if( (start_idx - j)*r_best >= max_count*r_cur ||
                    (r_best < FLT_EPSILON && start_idx - j >= max_count) )
                {
                    r_best = r_cur;
                    max_count = start_idx - j;
                }
                start_dist = d;
                start_idx = j;
                dist_sum = 0;
            }
            dist_sum += d;
        }

        if( max_count > R_THRESH )
        {
            c[0] = cx;
            c[1] = cy;
            c[2] = (float)r_best;
            cvSeqPush( circles, c );
            if( circles->total > circles_max )
                return;
        }
    }
}

CV_IMPL CvSeq*
cvHoughCircles( CvArr* src_image, void* circle_storage,
                int method, double dp, double min_dist,
                double param1, double param2,
                int min_radius, int max_radius )
{
    CvSeq* result = 0;

    CvMat stub, *img = (CvMat*)src_image;
    CvMat* mat = 0;
    CvSeq* circles = 0;
    CvSeq circles_header;
    CvSeqBlock circles_block;
    int circles_max = INT_MAX;
    int canny_threshold = cvRound(param1);
    int acc_threshold = cvRound(param2);

    img = cvGetMat( img, &stub );

    if( !CV_IS_MASK_ARR(img))
        CV_Error( CV_StsBadArg, "The source image must be 8-bit, single-channel" );

    if( !circle_storage )
        CV_Error( CV_StsNullPtr, "NULL destination" );

    if( dp <= 0 || min_dist <= 0 || canny_threshold <= 0 || acc_threshold <= 0 )
        CV_Error( CV_StsOutOfRange, "dp, min_dist, canny_threshold and acc_threshold must be all positive numbers" );

    min_radius = MAX( min_radius, 0 );
    if( max_radius <= 0 )
        max_radius = MAX( img->rows, img->cols );
    else if( max_radius <= min_radius )
        max_radius = min_radius + 2;

    if( CV_IS_STORAGE( circle_storage ))
    {
        circles = cvCreateSeq( CV_32FC3, sizeof(CvSeq),
            sizeof(float)*3, (CvMemStorage*)circle_storage );
    }
    else if( CV_IS_MAT( circle_storage ))
    {
        mat = (CvMat*)circle_storage;

        if( !CV_IS_MAT_CONT( mat->type ) || (mat->rows != 1 && mat->cols != 1) ||
            CV_MAT_TYPE(mat->type) != CV_32FC3 )
            CV_Error( CV_StsBadArg,
            "The destination matrix should be continuous and have a single row or a single column" );

        circles = cvMakeSeqHeaderForArray( CV_32FC3, sizeof(CvSeq), sizeof(float)*3,
                mat->data.ptr, mat->rows + mat->cols - 1, &circles_header, &circles_block );
        circles_max = circles->total;
        cvClearSeq( circles );
    }
    else
        CV_Error( CV_StsBadArg, "Destination is not CvMemStorage* nor CvMat*" );

    switch( method )
    {
    case CV_HOUGH_GRADIENT:
        icvHoughCirclesGradient( img, (float)dp, (float)min_dist,
                                min_radius, max_radius, canny_threshold,
                                acc_threshold, circles, circles_max );
          break;
    default:
        CV_Error( CV_StsBadArg, "Unrecognized method id" );
    }

    if( mat )
    {
        if( mat->cols > mat->rows )
            mat->cols = circles->total;
        else
            mat->rows = circles->total;
    }
    else
        result = circles;

    return result;
}


namespace cv
{

const int STORAGE_SIZE = 1 << 12;

void HoughLines( const Mat& image, vector<Vec2f>& lines,
                 double rho, double theta, int threshold,
                 double srn, double stn )
{
    CvMemStorage* storage = cvCreateMemStorage(STORAGE_SIZE);
    CvMat _image = image;
    CvSeq* seq = cvHoughLines2( &_image, storage, srn == 0 && stn == 0 ?
                    CV_HOUGH_STANDARD : CV_HOUGH_MULTI_SCALE,
                    rho, theta, threshold, srn, stn );
    Seq<Vec2f>(seq).copyTo(lines);
	cvReleaseMemStorage(&storage);
}

void HoughLinesP( Mat& image, vector<Vec4i>& lines,
                  double rho, double theta, int threshold,
                  double minLineLength, double maxGap )
{
    CvMemStorage* storage = cvCreateMemStorage(STORAGE_SIZE);
    CvMat _image = image;
    CvSeq* seq = cvHoughLines2( &_image, storage, CV_HOUGH_PROBABILISTIC,
                    rho, theta, threshold, minLineLength, maxGap );
    Seq<Vec4i>(seq).copyTo(lines);
	cvReleaseMemStorage(&storage);
}

void HoughCircles( const Mat& image, vector<Vec3f>& circles,
                   int method, double dp, double min_dist,
                   double param1, double param2,
                   int minRadius, int maxRadius )
{
    CvMemStorage* storage = cvCreateMemStorage(STORAGE_SIZE);
    CvMat _image = image;
    CvSeq* seq = cvHoughCircles( &_image, storage, method,
                    dp, min_dist, param1, param2, minRadius, maxRadius );
    Seq<Vec3f>(seq).copyTo(circles);
	cvReleaseMemStorage(&storage);
}

}

/* End of file. */
