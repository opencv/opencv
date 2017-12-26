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

/****************************************************************************************\
*                                  Chain Approximation                                   *
\****************************************************************************************/

typedef struct _CvPtInfo
{
    CvPoint pt;
    int k;                      /* support region */
    int s;                      /* curvature value */
    struct _CvPtInfo *next;
}
_CvPtInfo;


/* curvature: 0 - 1-curvature, 1 - k-cosine curvature. */
CvSeq* icvApproximateChainTC89( CvChain* chain, int header_size,
                                CvMemStorage* storage, int method )
{
    static const int abs_diff[] = { 1, 2, 3, 4, 3, 2, 1, 0, 1, 2, 3, 4, 3, 2, 1 };

    cv::AutoBuffer<_CvPtInfo> buf(chain->total + 8);

    _CvPtInfo       temp;
    _CvPtInfo       *array = buf, *first = 0, *current = 0, *prev_current = 0;
    int             i, j, i1, i2, s, len;
    int             count = chain->total;

    CvChainPtReader reader;
    CvSeqWriter     writer;
    CvPoint         pt = chain->origin;

    CV_Assert( CV_IS_SEQ_CHAIN_CONTOUR( chain ));
    CV_Assert( header_size >= (int)sizeof(CvContour) );

    cvStartWriteSeq( (chain->flags & ~CV_SEQ_ELTYPE_MASK) | CV_SEQ_ELTYPE_POINT,
                     header_size, sizeof( CvPoint ), storage, &writer );

    if( chain->total == 0 )
    {
        CV_WRITE_SEQ_ELEM( pt, writer );
        return cvEndWriteSeq( &writer );
    }

    reader.code = 0;
    cvStartReadChainPoints( chain, &reader );

    temp.next = 0;
    current = &temp;

    /* Pass 0.
       Restores all the digital curve points from the chain code.
       Removes the points (from the resultant polygon)
       that have zero 1-curvature */
    for( i = 0; i < count; i++ )
    {
        int prev_code = *reader.prev_elem;

        reader.prev_elem = reader.ptr;
        CV_READ_CHAIN_POINT( pt, reader );

        /* calc 1-curvature */
        s = abs_diff[reader.code - prev_code + 7];

        if( method <= CV_CHAIN_APPROX_SIMPLE )
        {
            if( method == CV_CHAIN_APPROX_NONE || s != 0 )
            {
                CV_WRITE_SEQ_ELEM( pt, writer );
            }
        }
        else
        {
            if( s != 0 )
                current = current->next = array + i;
            array[i].s = s;
            array[i].pt = pt;
        }
    }

    //assert( pt.x == chain->origin.x && pt.y == chain->origin.y );

    if( method <= CV_CHAIN_APPROX_SIMPLE )
        return cvEndWriteSeq( &writer );

    current->next = 0;

    len = i;
    current = temp.next;

    assert( current );

    /* Pass 1.
       Determines support region for all the remained points */
    do
    {
        CvPoint pt0;
        int k, l = 0, d_num = 0;

        i = (int)(current - array);
        pt0 = array[i].pt;

        /* determine support region */
        for( k = 1;; k++ )
        {
            int lk, dk_num;
            int dx, dy;
            Cv32suf d;

            assert( k <= len );

            /* calc indices */
            i1 = i - k;
            i1 += i1 < 0 ? len : 0;
            i2 = i + k;
            i2 -= i2 >= len ? len : 0;

            dx = array[i2].pt.x - array[i1].pt.x;
            dy = array[i2].pt.y - array[i1].pt.y;

            /* distance between p_(i - k) and p_(i + k) */
            lk = dx * dx + dy * dy;

            /* distance between p_i and the line (p_(i-k), p_(i+k)) */
            dk_num = (pt0.x - array[i1].pt.x) * dy - (pt0.y - array[i1].pt.y) * dx;
            d.f = (float) (((double) d_num) * lk - ((double) dk_num) * l);

            if( k > 1 && (l >= lk || ((d_num > 0 && d.i <= 0) || (d_num < 0 && d.i >= 0))))
                break;

            d_num = dk_num;
            l = lk;
        }

        current->k = --k;

        /* determine cosine curvature if it should be used */
        if( method == CV_CHAIN_APPROX_TC89_KCOS )
        {
            /* calc k-cosine curvature */
            for( j = k, s = 0; j > 0; j-- )
            {
                double temp_num;
                int dx1, dy1, dx2, dy2;
                Cv32suf sk;

                i1 = i - j;
                i1 += i1 < 0 ? len : 0;
                i2 = i + j;
                i2 -= i2 >= len ? len : 0;

                dx1 = array[i1].pt.x - pt0.x;
                dy1 = array[i1].pt.y - pt0.y;
                dx2 = array[i2].pt.x - pt0.x;
                dy2 = array[i2].pt.y - pt0.y;

                if( (dx1 | dy1) == 0 || (dx2 | dy2) == 0 )
                    break;

                temp_num = dx1 * dx2 + dy1 * dy2;
                temp_num =
                    (float) (temp_num /
                             sqrt( ((double)dx1 * dx1 + (double)dy1 * dy1) *
                                   ((double)dx2 * dx2 + (double)dy2 * dy2) ));
                sk.f = (float) (temp_num + 1.1);

                assert( 0 <= sk.f && sk.f <= 2.2 );
                if( j < k && sk.i <= s )
                    break;

                s = sk.i;
            }
            current->s = s;
        }
        current = current->next;
    }
    while( current != 0 );

    prev_current = &temp;
    current = temp.next;

    /* Pass 2.
       Performs non-maxima suppression */
    do
    {
        int k2 = current->k >> 1;

        s = current->s;
        i = (int)(current - array);

        for( j = 1; j <= k2; j++ )
        {
            i2 = i - j;
            i2 += i2 < 0 ? len : 0;

            if( array[i2].s > s )
                break;

            i2 = i + j;
            i2 -= i2 >= len ? len : 0;

            if( array[i2].s > s )
                break;
        }

        if( j <= k2 )           /* exclude point */
        {
            prev_current->next = current->next;
            current->s = 0;     /* "clear" point */
        }
        else
            prev_current = current;
        current = current->next;
    }
    while( current != 0 );

    /* Pass 3.
       Removes non-dominant points with 1-length support region */
    current = temp.next;
    assert( current );
    prev_current = &temp;

    do
    {
        if( current->k == 1 )
        {
            s = current->s;
            i = (int)(current - array);

            i1 = i - 1;
            i1 += i1 < 0 ? len : 0;

            i2 = i + 1;
            i2 -= i2 >= len ? len : 0;

            if( s <= array[i1].s || s <= array[i2].s )
            {
                prev_current->next = current->next;
                current->s = 0;
            }
            else
                prev_current = current;
        }
        else
            prev_current = current;
        current = current->next;
    }
    while( current != 0 );

    if( method == CV_CHAIN_APPROX_TC89_KCOS )
        goto copy_vect;

    /* Pass 4.
       Cleans remained couples of points */
    assert( temp.next );

    if( array[0].s != 0 && array[len - 1].s != 0 )      /* specific case */
    {
        for( i1 = 1; i1 < len && array[i1].s != 0; i1++ )
        {
            array[i1 - 1].s = 0;
        }
        if( i1 == len )
            goto copy_vect;     /* all points survived */
        i1--;

        for( i2 = len - 2; i2 > 0 && array[i2].s != 0; i2-- )
        {
            array[i2].next = 0;
            array[i2 + 1].s = 0;
        }
        i2++;

        if( i1 == 0 && i2 == len - 1 )  /* only two points */
        {
            i1 = (int)(array[0].next - array);
            array[len] = array[0];      /* move to the end */
            array[len].next = 0;
            array[len - 1].next = array + len;
        }
        temp.next = array + i1;
    }

    current = temp.next;
    first = prev_current = &temp;
    count = 1;

    /* do last pass */
    do
    {
        if( current->next == 0 || current->next - current != 1 )
        {
            if( count >= 2 )
            {
                if( count == 2 )
                {
                    int s1 = prev_current->s;
                    int s2 = current->s;

                    if( s1 > s2 || (s1 == s2 && prev_current->k <= current->k) )
                        /* remove second */
                        prev_current->next = current->next;
                    else
                        /* remove first */
                        first->next = current;
                }
                else
                    first->next->next = current;
            }
            first = current;
            count = 1;
        }
        else
            count++;
        prev_current = current;
        current = current->next;
    }
    while( current != 0 );

copy_vect:

    // gather points
    current = temp.next;
    assert( current );

    do
    {
        CV_WRITE_SEQ_ELEM( current->pt, writer );
        current = current->next;
    }
    while( current != 0 );

    return cvEndWriteSeq( &writer );
}


/*Applies some approximation algorithm to chain-coded contour(s) and
  converts it/them to polygonal representation */
CV_IMPL CvSeq*
cvApproxChains( CvSeq*              src_seq,
                CvMemStorage*       storage,
                int                 method,
                double              /*parameter*/,
                int                 minimal_perimeter,
                int                 recursive )
{
    CvSeq *prev_contour = 0, *parent = 0;
    CvSeq *dst_seq = 0;

    if( !src_seq || !storage )
        CV_Error( CV_StsNullPtr, "" );
    if( method > CV_CHAIN_APPROX_TC89_KCOS || method <= 0 || minimal_perimeter < 0 )
        CV_Error( CV_StsOutOfRange, "" );

    while( src_seq != 0 )
    {
        int len = src_seq->total;

        if( len >= minimal_perimeter )
        {
            CvSeq *contour = 0;

            switch( method )
            {
            case CV_CHAIN_APPROX_NONE:
            case CV_CHAIN_APPROX_SIMPLE:
            case CV_CHAIN_APPROX_TC89_L1:
            case CV_CHAIN_APPROX_TC89_KCOS:
                contour = icvApproximateChainTC89( (CvChain *) src_seq, sizeof( CvContour ), storage, method );
                break;
            default:
                CV_Error( CV_StsOutOfRange, "" );
            }

            if( contour->total > 0 )
            {
                cvBoundingRect( contour, 1 );

                contour->v_prev = parent;
                contour->h_prev = prev_contour;

                if( prev_contour )
                    prev_contour->h_next = contour;
                else if( parent )
                    parent->v_next = contour;
                prev_contour = contour;
                if( !dst_seq )
                    dst_seq = prev_contour;
            }
            else                /* if resultant contour has zero length, skip it */
            {
                len = -1;
            }
        }

        if( !recursive )
            break;

        if( src_seq->v_next && len >= minimal_perimeter )
        {
            assert( prev_contour != 0 );
            parent = prev_contour;
            prev_contour = 0;
            src_seq = src_seq->v_next;
        }
        else
        {
            while( src_seq->h_next == 0 )
            {
                src_seq = src_seq->v_prev;
                if( src_seq == 0 )
                    break;
                prev_contour = parent;
                if( parent )
                    parent = parent->v_prev;
            }
            if( src_seq )
                src_seq = src_seq->h_next;
        }
    }

    return dst_seq;
}


/****************************************************************************************\
*                               Polygonal Approximation                                  *
\****************************************************************************************/

/* Ramer-Douglas-Peucker algorithm for polygon simplification */

namespace cv
{

template<typename T> static int
approxPolyDP_( const Point_<T>* src_contour, int count0, Point_<T>* dst_contour,
              bool is_closed0, double eps, AutoBuffer<Range>* _stack )
{
    #define PUSH_SLICE(slice) \
        if( top >= stacksz ) \
        { \
            _stack->resize(stacksz*3/2); \
            stack = *_stack; \
            stacksz = _stack->size(); \
        } \
        stack[top++] = slice

    #define READ_PT(pt, pos) \
        pt = src_contour[pos]; \
        if( ++pos >= count ) pos = 0

    #define READ_DST_PT(pt, pos) \
        pt = dst_contour[pos]; \
        if( ++pos >= count ) pos = 0

    #define WRITE_PT(pt) \
        dst_contour[new_count++] = pt

    typedef cv::Point_<T> PT;
    int             init_iters = 3;
    Range           slice(0, 0), right_slice(0, 0);
    PT              start_pt((T)-1000000, (T)-1000000), end_pt(0, 0), pt(0,0);
    int             i = 0, j, pos = 0, wpos, count = count0, new_count=0;
    int             is_closed = is_closed0;
    bool            le_eps = false;
    size_t top = 0, stacksz = _stack->size();
    Range*          stack = *_stack;

    if( count == 0  )
        return 0;

    eps *= eps;

    if( !is_closed )
    {
        right_slice.start = count;
        end_pt = src_contour[0];
        start_pt = src_contour[count-1];

        if( start_pt.x != end_pt.x || start_pt.y != end_pt.y )
        {
            slice.start = 0;
            slice.end = count - 1;
            PUSH_SLICE(slice);
        }
        else
        {
            is_closed = 1;
            init_iters = 1;
        }
    }

    if( is_closed )
    {
        // 1. Find approximately two farthest points of the contour
        right_slice.start = 0;

        for( i = 0; i < init_iters; i++ )
        {
            double dist, max_dist = 0;
            pos = (pos + right_slice.start) % count;
            READ_PT(start_pt, pos);

            for( j = 1; j < count; j++ )
            {
                double dx, dy;

                READ_PT(pt, pos);
                dx = pt.x - start_pt.x;
                dy = pt.y - start_pt.y;

                dist = dx * dx + dy * dy;

                if( dist > max_dist )
                {
                    max_dist = dist;
                    right_slice.start = j;
                }
            }

            le_eps = max_dist <= eps;
        }

        // 2. initialize the stack
        if( !le_eps )
        {
            right_slice.end = slice.start = pos % count;
            slice.end = right_slice.start = (right_slice.start + slice.start) % count;

            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
        else
            WRITE_PT(start_pt);
    }

    // 3. run recursive process
    while( top > 0 )
    {
        slice = stack[--top];
        end_pt = src_contour[slice.end];
        pos = slice.start;
        READ_PT(start_pt, pos);

        if( pos != slice.end )
        {
            double dx, dy, dist, max_dist = 0;

            dx = end_pt.x - start_pt.x;
            dy = end_pt.y - start_pt.y;

            assert( dx != 0 || dy != 0 );

            while( pos != slice.end )
            {
                READ_PT(pt, pos);
                dist = fabs((pt.y - start_pt.y) * dx - (pt.x - start_pt.x) * dy);

                if( dist > max_dist )
                {
                    max_dist = dist;
                    right_slice.start = (pos+count-1)%count;
                }
            }

            le_eps = max_dist * max_dist <= eps * (dx * dx + dy * dy);
        }
        else
        {
            le_eps = true;
            // read starting point
            start_pt = src_contour[slice.start];
        }

        if( le_eps )
        {
            WRITE_PT(start_pt);
        }
        else
        {
            right_slice.end = slice.end;
            slice.end = right_slice.start;
            PUSH_SLICE(right_slice);
            PUSH_SLICE(slice);
        }
    }

    if( !is_closed )
        WRITE_PT( src_contour[count-1] );

    // last stage: do final clean-up of the approximated contour -
    // remove extra points on the [almost] stright lines.
    is_closed = is_closed0;
    count = new_count;
    pos = is_closed ? count - 1 : 0;
    READ_DST_PT(start_pt, pos);
    wpos = pos;
    READ_DST_PT(pt, pos);

    for( i = !is_closed; i < count - !is_closed && new_count > 2; i++ )
    {
        double dx, dy, dist, successive_inner_product;
        READ_DST_PT( end_pt, pos );

        dx = end_pt.x - start_pt.x;
        dy = end_pt.y - start_pt.y;
        dist = fabs((pt.x - start_pt.x)*dy - (pt.y - start_pt.y)*dx);
        successive_inner_product = (pt.x - start_pt.x) * (end_pt.x - pt.x) +
        (pt.y - start_pt.y) * (end_pt.y - pt.y);

        if( dist * dist <= 0.5*eps*(dx*dx + dy*dy) && dx != 0 && dy != 0 &&
           successive_inner_product >= 0 )
        {
            new_count--;
            dst_contour[wpos] = start_pt = end_pt;
            if(++wpos >= count) wpos = 0;
            READ_DST_PT(pt, pos);
            i++;
            continue;
        }
        dst_contour[wpos] = start_pt = pt;
        if(++wpos >= count) wpos = 0;
        pt = end_pt;
    }

    if( !is_closed )
        dst_contour[wpos] = pt;

    return new_count;
}

}

void cv::approxPolyDP( InputArray _curve, OutputArray _approxCurve,
                      double epsilon, bool closed )
{
    CV_INSTRUMENT_REGION()

    Mat curve = _curve.getMat();
    int npoints = curve.checkVector(2), depth = curve.depth();
    CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F));

    if( npoints == 0 )
    {
        _approxCurve.release();
        return;
    }

    AutoBuffer<Point> _buf(npoints);
    AutoBuffer<Range> _stack(npoints);
    Point* buf = _buf;
    int nout = 0;

    if( depth == CV_32S )
        nout = approxPolyDP_(curve.ptr<Point>(), npoints, buf, closed, epsilon, &_stack);
    else if( depth == CV_32F )
        nout = approxPolyDP_(curve.ptr<Point2f>(), npoints, (Point2f*)buf, closed, epsilon, &_stack);
    else
        CV_Error( CV_StsUnsupportedFormat, "" );

    Mat(nout, 1, CV_MAKETYPE(depth, 2), buf).copyTo(_approxCurve);
}


CV_IMPL CvSeq*
cvApproxPoly( const void* array, int header_size,
             CvMemStorage* storage, int method,
             double parameter, int parameter2 )
{
    cv::AutoBuffer<cv::Point> _buf;
    cv::AutoBuffer<cv::Range> stack(100);
    CvSeq* dst_seq = 0;
    CvSeq *prev_contour = 0, *parent = 0;
    CvContour contour_header;
    CvSeq* src_seq = 0;
    CvSeqBlock block;
    int recursive = 0;

    if( CV_IS_SEQ( array ))
    {
        src_seq = (CvSeq*)array;
        if( !CV_IS_SEQ_POLYLINE( src_seq ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );

        recursive = parameter2;

        if( !storage )
            storage = src_seq->storage;
    }
    else
    {
        src_seq = cvPointSeqFromMat(
                                    CV_SEQ_KIND_CURVE | (parameter2 ? CV_SEQ_FLAG_CLOSED : 0),
                                    array, &contour_header, &block );
    }

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer " );

    if( header_size < 0 )
        CV_Error( CV_StsOutOfRange, "header_size is negative. "
                 "Pass 0 to make the destination header_size == input header_size" );

    if( header_size == 0 )
        header_size = src_seq->header_size;

    if( !CV_IS_SEQ_POLYLINE( src_seq ))
    {
        if( CV_IS_SEQ_CHAIN( src_seq ))
        {
            CV_Error( CV_StsBadArg, "Input curves are not polygonal. "
                     "Use cvApproxChains first" );
        }
        else
        {
            CV_Error( CV_StsBadArg, "Input curves have uknown type" );
        }
    }

    if( header_size == 0 )
        header_size = src_seq->header_size;

    if( header_size < (int)sizeof(CvContour) )
        CV_Error( CV_StsBadSize, "New header size must be non-less than sizeof(CvContour)" );

    if( method != CV_POLY_APPROX_DP )
        CV_Error( CV_StsOutOfRange, "Unknown approximation method" );

    while( src_seq != 0 )
    {
        CvSeq *contour = 0;

        switch (method)
        {
        case CV_POLY_APPROX_DP:
            if( parameter < 0 )
                CV_Error( CV_StsOutOfRange, "Accuracy must be non-negative" );

            CV_Assert( CV_SEQ_ELTYPE(src_seq) == CV_32SC2 ||
                      CV_SEQ_ELTYPE(src_seq) == CV_32FC2 );

            {
            int npoints = src_seq->total, nout = 0;
            _buf.allocate(npoints*2);
            cv::Point *src = _buf, *dst = src + npoints;
            bool closed = CV_IS_SEQ_CLOSED(src_seq);

            if( src_seq->first->next == src_seq->first )
                src = (cv::Point*)src_seq->first->data;
            else
                cvCvtSeqToArray(src_seq, src);

            if( CV_SEQ_ELTYPE(src_seq) == CV_32SC2 )
                nout = cv::approxPolyDP_(src, npoints, dst, closed, parameter, &stack);
            else if( CV_SEQ_ELTYPE(src_seq) == CV_32FC2 )
                nout = cv::approxPolyDP_((cv::Point2f*)src, npoints,
                                         (cv::Point2f*)dst, closed, parameter, &stack);
            else
                CV_Error( CV_StsUnsupportedFormat, "" );

            contour = cvCreateSeq( src_seq->flags, header_size,
                                  src_seq->elem_size, storage );
            cvSeqPushMulti(contour, dst, nout);
            }
            break;
        default:
            CV_Error( CV_StsBadArg, "Invalid approximation method" );
        }

        assert( contour );

        if( header_size >= (int)sizeof(CvContour))
            cvBoundingRect( contour, 1 );

        contour->v_prev = parent;
        contour->h_prev = prev_contour;

        if( prev_contour )
            prev_contour->h_next = contour;
        else if( parent )
            parent->v_next = contour;
        prev_contour = contour;
        if( !dst_seq )
            dst_seq = prev_contour;

        if( !recursive )
            break;

        if( src_seq->v_next )
        {
            assert( prev_contour != 0 );
            parent = prev_contour;
            prev_contour = 0;
            src_seq = src_seq->v_next;
        }
        else
        {
            while( src_seq->h_next == 0 )
            {
                src_seq = src_seq->v_prev;
                if( src_seq == 0 )
                    break;
                prev_contour = parent;
                if( parent )
                    parent = parent->v_prev;
            }
            if( src_seq )
                src_seq = src_seq->h_next;
        }
    }

    return dst_seq;
}

/* End of file. */
