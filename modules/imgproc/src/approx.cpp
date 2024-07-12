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
*                               Polygonal Approximation                                  *
\****************************************************************************************/

/* Ramer-Douglas-Peucker algorithm for polygon simplification */

namespace cv
{

template<typename T> static int
approxPolyDP_( const Point_<T>* src_contour, int count0, Point_<T>* dst_contour,
              bool is_closed0, double eps, AutoBuffer<Range>& _stack )
{
    #define PUSH_SLICE(slice) \
        if( top >= stacksz ) \
        { \
            _stack.resize(stacksz*3/2); \
            stack = _stack.data(); \
            stacksz = _stack.size(); \
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
    size_t top = 0, stacksz = _stack.size();
    Range*          stack = _stack.data();

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

            CV_Assert( dx != 0 || dy != 0 );

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
    // remove extra points on the [almost] straight lines.
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
    CV_INSTRUMENT_REGION();

    //Prevent unreasonable error values (Douglas-Peucker algorithm)
    //from being used.
    if (epsilon < 0.0 || !(epsilon < 1e30))
    {
        CV_Error(cv::Error::StsOutOfRange, "Epsilon not valid.");
    }

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
    Point* buf = _buf.data();
    int nout = 0;

    if( depth == CV_32S )
        nout = approxPolyDP_(curve.ptr<Point>(), npoints, buf, closed, epsilon, _stack);
    else if( depth == CV_32F )
        nout = approxPolyDP_(curve.ptr<Point2f>(), npoints, (Point2f*)buf, closed, epsilon, _stack);
    else
        CV_Error( cv::Error::StsUnsupportedFormat, "" );

    Mat(nout, 1, CV_MAKETYPE(depth, 2), buf).copyTo(_approxCurve);
}
