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
#include <iostream>

namespace cv
{

template<typename _Tp, typename _DotTp>
static int Sklansky_( Point_<_Tp>** array, int start, int end, int* stack, int nsign, int sign2 )
{
    int incr = end > start ? 1 : -1;
    // prepare first triangle
    int pprev = start, pcur = pprev + incr, pnext = pcur + incr;
    int stacksize = 3;

    if( start == end ||
       (array[start]->x == array[end]->x &&
        array[start]->y == array[end]->y) )
    {
        stack[0] = start;
        return 1;
    }

    stack[0] = pprev;
    stack[1] = pcur;
    stack[2] = pnext;

    end += incr; // make end = afterend

    while( pnext != end )
    {
        // check the angle p1,p2,p3
        _Tp cury = array[pcur]->y;
        _Tp nexty = array[pnext]->y;
        _Tp by = nexty - cury;

        if( CV_SIGN( by ) != nsign )
        {
            _Tp ax = array[pcur]->x - array[pprev]->x;
            _Tp bx = array[pnext]->x - array[pcur]->x;
            _Tp ay = cury - array[pprev]->y;
            _DotTp convexity = (_DotTp)ay*bx - (_DotTp)ax*by; // if >0 then convex angle

            if( CV_SIGN( convexity ) == sign2 && (ax != 0 || ay != 0) )
            {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext;
                stacksize++;
            }
            else
            {
                if( pprev == start )
                {
                    pcur = pnext;
                    stack[1] = pcur;
                    pnext += incr;
                    stack[2] = pnext;
                }
                else
                {
                    stack[stacksize-2] = pnext;
                    pcur = pprev;
                    pprev = stack[stacksize-4];
                    stacksize--;
                }
            }
        }
        else
        {
            pnext += incr;
            stack[stacksize-1] = pnext;
        }
    }

    return --stacksize;
}


template<typename _Tp>
struct CHullCmpPoints
{
    bool operator()(const Point_<_Tp>* p1, const Point_<_Tp>* p2) const
    {
        if( p1->x != p2->x )
            return p1->x < p2->x;
        if( p1->y != p2->y )
            return p1->y < p2->y;
        return p1 < p2;
    }
};


void convexHull( InputArray _points, OutputArray _hull, bool clockwise, bool returnPoints )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!_points.pointsTo(_hull));
    Mat points = _points.getMat();
    int i, total = points.checkVector(2), depth = points.depth(), nout = 0;
    int miny_ind = 0, maxy_ind = 0;
    CV_Assert(total >= 0 && (depth == CV_32F || depth == CV_32S));

    if( total == 0 )
    {
        _hull.release();
        return;
    }

    returnPoints = !_hull.fixedType() ? returnPoints : _hull.type() != CV_32S;

    bool is_float = depth == CV_32F;
    AutoBuffer<Point*> _pointer(total);
    AutoBuffer<int> _stack(total + 2), _hullbuf(total);
    Point** pointer = _pointer.data();
    Point2f** pointerf = (Point2f**)pointer;
    Point* data0 = points.ptr<Point>();
    int* stack = _stack.data();
    int* hullbuf = _hullbuf.data();

    CV_Assert(points.isContinuous());

    for( i = 0; i < total; i++ )
        pointer[i] = &data0[i];

    // sort the point set by x-coordinate, find min and max y
    if( !is_float )
    {
        std::sort(pointer, pointer + total, CHullCmpPoints<int>());
        for( i = 1; i < total; i++ )
        {
            int y = pointer[i]->y;
            if( pointer[miny_ind]->y > y )
                miny_ind = i;
            if( pointer[maxy_ind]->y < y )
                maxy_ind = i;
        }
    }
    else
    {
        std::sort(pointerf, pointerf + total, CHullCmpPoints<float>());
        for( i = 1; i < total; i++ )
        {
            float y = pointerf[i]->y;
            if( pointerf[miny_ind]->y > y )
                miny_ind = i;
            if( pointerf[maxy_ind]->y < y )
                maxy_ind = i;
        }
    }

    if( pointer[0]->x == pointer[total-1]->x &&
        pointer[0]->y == pointer[total-1]->y )
    {
        hullbuf[nout++] = 0;
    }
    else
    {
        // upper half
        int *tl_stack = stack;
        int tl_count = !is_float ?
            Sklansky_<int, int64>( pointer, 0, maxy_ind, tl_stack, -1, 1) :
            Sklansky_<float, double>( pointerf, 0, maxy_ind, tl_stack, -1, 1);
        int *tr_stack = stack + tl_count;
        int tr_count = !is_float ?
            Sklansky_<int, int64>( pointer, total-1, maxy_ind, tr_stack, -1, -1) :
            Sklansky_<float, double>( pointerf, total-1, maxy_ind, tr_stack, -1, -1);

        // gather upper part of convex hull to output
        if( !clockwise )
        {
            std::swap( tl_stack, tr_stack );
            std::swap( tl_count, tr_count );
        }

        for( i = 0; i < tl_count-1; i++ )
            hullbuf[nout++] = int(pointer[tl_stack[i]] - data0);
        for( i = tr_count - 1; i > 0; i-- )
            hullbuf[nout++] = int(pointer[tr_stack[i]] - data0);
        int stop_idx = tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;

        // lower half
        int *bl_stack = stack;
        int bl_count = !is_float ?
            Sklansky_<int, int64>( pointer, 0, miny_ind, bl_stack, 1, -1) :
            Sklansky_<float, double>( pointerf, 0, miny_ind, bl_stack, 1, -1);
        int *br_stack = stack + bl_count;
        int br_count = !is_float ?
            Sklansky_<int, int64>( pointer, total-1, miny_ind, br_stack, 1, 1) :
            Sklansky_<float, double>( pointerf, total-1, miny_ind, br_stack, 1, 1);

        if( clockwise )
        {
            std::swap( bl_stack, br_stack );
            std::swap( bl_count, br_count );
        }

        if( stop_idx >= 0 )
        {
            int check_idx = bl_count > 2 ? bl_stack[1] :
            bl_count + br_count > 2 ? br_stack[2-bl_count] : -1;
            if( check_idx == stop_idx || (check_idx >= 0 &&
                                          pointer[check_idx]->x == pointer[stop_idx]->x &&
                                          pointer[check_idx]->y == pointer[stop_idx]->y) )
            {
                // if all the points lie on the same line, then
                // the bottom part of the convex hull is the mirrored top part
                // (except the exteme points).
                bl_count = MIN( bl_count, 2 );
                br_count = MIN( br_count, 2 );
            }
        }

        for( i = 0; i < bl_count-1; i++ )
            hullbuf[nout++] = int(pointer[bl_stack[i]] - data0);
        for( i = br_count-1; i > 0; i-- )
            hullbuf[nout++] = int(pointer[br_stack[i]] - data0);

        // try to make the convex hull indices form
        // an ascending or descending sequence by the cyclic
        // shift of the output sequence.
        if( nout >= 3 )
        {
            int min_idx = 0, max_idx = 0, lt = 0;
            for( i = 1; i < nout; i++ )
            {
                int idx = hullbuf[i];
                lt += hullbuf[i-1] < idx;
                if( lt > 1 && lt <= i-2 )
                    break;
                if( idx < hullbuf[min_idx] )
                    min_idx = i;
                if( idx > hullbuf[max_idx] )
                    max_idx = i;
            }
            int mmdist = std::abs(max_idx - min_idx);
            if( (mmdist == 1 || mmdist == nout-1) && (lt <= 1 || lt >= nout-2) )
            {
                int ascending = (max_idx + 1) % nout == min_idx;
                int i0 = ascending ? min_idx : max_idx, j = i0;
                if( i0 > 0 )
                {
                    for( i = 0; i < nout; i++ )
                    {
                        int curr_idx = stack[i] = hullbuf[j];
                        int next_j = j+1 < nout ? j+1 : 0;
                        int next_idx = hullbuf[next_j];
                        if( i < nout-1 && (ascending != (curr_idx < next_idx)) )
                            break;
                        j = next_j;
                    }
                    if( i == nout )
                        memcpy(hullbuf, stack, nout*sizeof(hullbuf[0]));
                }
            }
        }
    }

    if( !returnPoints )
        Mat(nout, 1, CV_32S, hullbuf).copyTo(_hull);
    else
    {
        _hull.create(nout, 1, CV_MAKETYPE(depth, 2));
        Mat hull = _hull.getMat();
        size_t step = !hull.isContinuous() ? hull.step[0] : sizeof(Point);
        for( i = 0; i < nout; i++ )
            *(Point*)(hull.ptr() + i*step) = data0[hullbuf[i]];
    }
}


void convexityDefects( InputArray _points, InputArray _hull, OutputArray _defects )
{
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();
    int i, j = 0, npoints = points.checkVector(2, CV_32S);
    CV_Assert( npoints >= 0 );

    if( npoints <= 3 )
    {
        _defects.release();
        return;
    }

    Mat hull = _hull.getMat();
    int hpoints = hull.checkVector(1, CV_32S);
    CV_Assert( hpoints > 0 );

    const Point* ptr = points.ptr<Point>();
    const int* hptr = hull.ptr<int>();
    std::vector<Vec4i> defects;
    if ( hpoints < 3 ) //if hull consists of one or two points, contour is always convex
    {
        _defects.release();
        return;
    }

    // 1. recognize co-orientation of the contour and its hull
    bool rev_orientation = ((hptr[1] > hptr[0]) + (hptr[2] > hptr[1]) + (hptr[0] > hptr[2])) != 2;

    // 2. cycle through points and hull, compute defects
    int hcurr = hptr[rev_orientation ? 0 : hpoints-1];
    CV_Assert( 0 <= hcurr && hcurr < npoints );

    int increasing_idx = -1;

    for( i = 0; i < hpoints; i++ )
    {
        int hnext = hptr[rev_orientation ? hpoints - i - 1 : i];
        CV_Assert( 0 <= hnext && hnext < npoints );

        Point pt0 = ptr[hcurr], pt1 = ptr[hnext];
        if( increasing_idx < 0 )
            increasing_idx = !(hcurr < hnext);
        else if( increasing_idx != (hcurr < hnext))
        {
            CV_Error(Error::StsBadArg,
            "The convex hull indices are not monotonous, which can be in the case when the input contour contains self-intersections");
        }

        double dx0 = pt1.x - pt0.x;
        double dy0 = pt1.y - pt0.y;
        double scale = dx0 == 0 && dy0 == 0 ? 0. : 1./std::sqrt(dx0*dx0 + dy0*dy0);

        int defect_deepest_point = -1;
        double defect_depth = 0;
        bool is_defect = false;
        j=hcurr;
        for(;;)
        {
            // go through points to achieve next hull point
            j++;
            j &= j >= npoints ? 0 : -1;
            if( j == hnext )
                break;

            // compute distance from current point to hull edge
            double dx = ptr[j].x - pt0.x;
            double dy = ptr[j].y - pt0.y;
            double dist = fabs(-dy0*dx + dx0*dy) * scale;

            if( dist > defect_depth )
            {
                defect_depth = dist;
                defect_deepest_point = j;
                is_defect = true;
            }
        }

        if( is_defect )
        {
            int idepth = cvRound(defect_depth*256);
            defects.push_back(Vec4i(hcurr, hnext, defect_deepest_point, idepth));
        }

        hcurr = hnext;
    }

    Mat(defects).copyTo(_defects);
}


template<typename _Tp>
static bool isContourConvex_( const Point_<_Tp>* p, int n )
{
    Point_<_Tp> prev_pt = p[(n-2+n) % n];
    Point_<_Tp> cur_pt = p[n-1];

    _Tp dx0 = cur_pt.x - prev_pt.x;
    _Tp dy0 = cur_pt.y - prev_pt.y;
    int orientation = 0;

    for( int i = 0; i < n; i++ )
    {
        _Tp dxdy0, dydx0;
        _Tp dx, dy;

        prev_pt = cur_pt;
        cur_pt = p[i];

        dx = cur_pt.x - prev_pt.x;
        dy = cur_pt.y - prev_pt.y;
        dxdy0 = dx * dy0;
        dydx0 = dy * dx0;

        // find orientation
        // orient = -dy0 * dx + dx0 * dy;
        // orientation |= (orient > 0) ? 1 : 2;
        orientation |= (dydx0 > dxdy0) ? 1 : ((dydx0 < dxdy0) ? 2 : 3);
        if( orientation == 3 )
            return false;

        dx0 = dx;
        dy0 = dy;
    }

    return true;
}


bool isContourConvex( InputArray _contour )
{
    Mat contour = _contour.getMat();
    int total = contour.checkVector(2), depth = contour.depth();
    CV_Assert(total >= 0 && (depth == CV_32F || depth == CV_32S));

    if( total == 0 )
        return false;

    return depth == CV_32S ?
    isContourConvex_(contour.ptr<Point>(), total ) :
    isContourConvex_(contour.ptr<Point2f>(), total );
}

}
