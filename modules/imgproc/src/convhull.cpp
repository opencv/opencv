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

template<typename _Tp>
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
            _Tp convexity = ay*bx - ax*by; // if >0 then convex angle

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
    { return p1->x < p2->x || (p1->x == p2->x && p1->y < p2->y); }
};


void convexHull( InputArray _points, OutputArray _hull, bool clockwise, bool returnPoints )
{
    CV_INSTRUMENT_REGION();

    CV_Assert(_points.getObj() != _hull.getObj());
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
            Sklansky_( pointer, 0, maxy_ind, tl_stack, -1, 1) :
            Sklansky_( pointerf, 0, maxy_ind, tl_stack, -1, 1);
        int *tr_stack = stack + tl_count;
        int tr_count = !is_float ?
            Sklansky_( pointer, total-1, maxy_ind, tr_stack, -1, -1) :
            Sklansky_( pointerf, total-1, maxy_ind, tr_stack, -1, -1);

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
            Sklansky_( pointer, 0, miny_ind, bl_stack, 1, -1) :
            Sklansky_( pointerf, 0, miny_ind, bl_stack, 1, -1);
        int *br_stack = stack + bl_count;
        int br_count = !is_float ?
            Sklansky_( pointer, total-1, miny_ind, br_stack, 1, 1) :
            Sklansky_( pointerf, total-1, miny_ind, br_stack, 1, 1);

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

    for( i = 0; i < hpoints; i++ )
    {
        int hnext = hptr[rev_orientation ? hpoints - i - 1 : i];
        CV_Assert( 0 <= hnext && hnext < npoints );

        Point pt0 = ptr[hcurr], pt1 = ptr[hnext];
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

CV_IMPL CvSeq*
cvConvexHull2( const CvArr* array, void* hull_storage,
               int orientation, int return_points )
{
    CvMat* mat = 0;
    CvContour contour_header;
    CvSeq hull_header;
    CvSeqBlock block, hullblock;
    CvSeq* ptseq = 0;
    CvSeq* hullseq = 0;

    if( CV_IS_SEQ( array ))
    {
        ptseq = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );
        if( hull_storage == 0 )
            hull_storage = ptseq->storage;
    }
    else
    {
        ptseq = cvPointSeqFromMat( CV_SEQ_KIND_GENERIC, array, &contour_header, &block );
    }

    bool isStorage = isStorageOrMat(hull_storage);

    if(isStorage)
    {
        if( return_points )
        {
            hullseq = cvCreateSeq(CV_SEQ_KIND_CURVE|CV_SEQ_ELTYPE(ptseq)|
                                  CV_SEQ_FLAG_CLOSED|CV_SEQ_FLAG_CONVEX,
                                  sizeof(CvContour), sizeof(CvPoint),(CvMemStorage*)hull_storage );
        }
        else
        {
            hullseq = cvCreateSeq(
                                  CV_SEQ_KIND_CURVE|CV_SEQ_ELTYPE_PPOINT|
                                  CV_SEQ_FLAG_CLOSED|CV_SEQ_FLAG_CONVEX,
                                  sizeof(CvContour), sizeof(CvPoint*), (CvMemStorage*)hull_storage );
        }
    }
    else
    {
        mat = (CvMat*)hull_storage;

        if( (mat->cols != 1 && mat->rows != 1) || !CV_IS_MAT_CONT(mat->type))
            CV_Error( CV_StsBadArg,
                     "The hull matrix should be continuous and have a single row or a single column" );

        if( mat->cols + mat->rows - 1 < ptseq->total )
            CV_Error( CV_StsBadSize, "The hull matrix size might be not enough to fit the hull" );

        if( CV_MAT_TYPE(mat->type) != CV_SEQ_ELTYPE(ptseq) &&
           CV_MAT_TYPE(mat->type) != CV_32SC1 )
            CV_Error( CV_StsUnsupportedFormat,
                     "The hull matrix must have the same type as input or 32sC1 (integers)" );

        hullseq = cvMakeSeqHeaderForArray(
                                          CV_SEQ_KIND_CURVE|CV_MAT_TYPE(mat->type)|CV_SEQ_FLAG_CLOSED,
                                          sizeof(hull_header), CV_ELEM_SIZE(mat->type), mat->data.ptr,
                                          mat->cols + mat->rows - 1, &hull_header, &hullblock );
        cvClearSeq( hullseq );
    }

    int hulltype = CV_SEQ_ELTYPE(hullseq);
    int total = ptseq->total;
    if( total == 0 )
    {
        if( !isStorage )
            CV_Error( CV_StsBadSize,
                     "Point sequence can not be empty if the output is matrix" );
        return 0;
    }

    cv::AutoBuffer<double> _ptbuf;
    cv::Mat h0;
    cv::convexHull(cv::cvarrToMat(ptseq, false, false, 0, &_ptbuf), h0,
                   orientation == CV_CLOCKWISE, CV_MAT_CN(hulltype) == 2);


    if( hulltype == CV_SEQ_ELTYPE_PPOINT )
    {
        const int* idx = h0.ptr<int>();
        int ctotal = (int)h0.total();
        for( int i = 0; i < ctotal; i++ )
        {
            void* ptr = cvGetSeqElem(ptseq, idx[i]);
            cvSeqPush( hullseq, &ptr );
        }
    }
    else
        cvSeqPushMulti(hullseq, h0.ptr(), (int)h0.total());

    if (isStorage)
    {
        return hullseq;
    }
    else
    {
        if( mat->rows > mat->cols )
            mat->rows = hullseq->total;
        else
            mat->cols = hullseq->total;
        return 0;
    }
}


/* contour must be a simple polygon */
/* it must have more than 3 points  */
CV_IMPL CvSeq* cvConvexityDefects( const CvArr* array,
                                  const CvArr* hullarray,
                                  CvMemStorage* storage )
{
    CvSeq* defects = 0;

    int i, index;
    CvPoint* hull_cur;

    /* is orientation of hull different from contour one */
    int rev_orientation;

    CvContour contour_header;
    CvSeq hull_header;
    CvSeqBlock block, hullblock;
    CvSeq *ptseq = (CvSeq*)array, *hull = (CvSeq*)hullarray;

    CvSeqReader hull_reader;
    CvSeqReader ptseq_reader;
    CvSeqWriter writer;
    int is_index;

    if( CV_IS_SEQ( ptseq ))
    {
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( CV_StsUnsupportedFormat,
                     "Input sequence is not a sequence of points" );
        if( !storage )
            storage = ptseq->storage;
    }
    else
    {
        ptseq = cvPointSeqFromMat( CV_SEQ_KIND_GENERIC, array, &contour_header, &block );
    }

    if( CV_SEQ_ELTYPE( ptseq ) != CV_32SC2 )
        CV_Error( CV_StsUnsupportedFormat, "Floating-point coordinates are not supported here" );

    if( CV_IS_SEQ( hull ))
    {
        int hulltype = CV_SEQ_ELTYPE( hull );
        if( hulltype != CV_SEQ_ELTYPE_PPOINT && hulltype != CV_SEQ_ELTYPE_INDEX )
            CV_Error( CV_StsUnsupportedFormat,
                     "Convex hull must represented as a sequence "
                     "of indices or sequence of pointers" );
        if( !storage )
            storage = hull->storage;
    }
    else
    {
        CvMat* mat = (CvMat*)hull;

        if( !CV_IS_MAT( hull ))
            CV_Error(CV_StsBadArg, "Convex hull is neither sequence nor matrix");

        if( (mat->cols != 1 && mat->rows != 1) ||
           !CV_IS_MAT_CONT(mat->type) || CV_MAT_TYPE(mat->type) != CV_32SC1 )
            CV_Error( CV_StsBadArg,
                     "The matrix should be 1-dimensional and continuous array of int's" );

        if( mat->cols + mat->rows - 1 > ptseq->total )
            CV_Error( CV_StsBadSize, "Convex hull is larger than the point sequence" );

        hull = cvMakeSeqHeaderForArray(
                                       CV_SEQ_KIND_CURVE|CV_MAT_TYPE(mat->type)|CV_SEQ_FLAG_CLOSED,
                                       sizeof(hull_header), CV_ELEM_SIZE(mat->type), mat->data.ptr,
                                       mat->cols + mat->rows - 1, &hull_header, &hullblock );
    }

    is_index = CV_SEQ_ELTYPE(hull) == CV_SEQ_ELTYPE_INDEX;

    if( !storage )
        CV_Error( CV_StsNullPtr, "NULL storage pointer" );

    defects = cvCreateSeq( CV_SEQ_KIND_GENERIC, sizeof(CvSeq), sizeof(CvConvexityDefect), storage );

    if( ptseq->total < 4 || hull->total < 3)
    {
        //CV_ERROR( CV_StsBadSize,
        //    "point seq size must be >= 4, convex hull size must be >= 3" );
        return defects;
    }

    /* recognize co-orientation of ptseq and its hull */
    {
        int sign = 0;
        int index1, index2, index3;

        if( !is_index )
        {
            CvPoint* pos = *CV_SEQ_ELEM( hull, CvPoint*, 0 );
            index1 = cvSeqElemIdx( ptseq, pos );

            pos = *CV_SEQ_ELEM( hull, CvPoint*, 1 );
            index2 = cvSeqElemIdx( ptseq, pos );

            pos = *CV_SEQ_ELEM( hull, CvPoint*, 2 );
            index3 = cvSeqElemIdx( ptseq, pos );
        }
        else
        {
            index1 = *CV_SEQ_ELEM( hull, int, 0 );
            index2 = *CV_SEQ_ELEM( hull, int, 1 );
            index3 = *CV_SEQ_ELEM( hull, int, 2 );
        }

        sign += (index2 > index1) ? 1 : 0;
        sign += (index3 > index2) ? 1 : 0;
        sign += (index1 > index3) ? 1 : 0;

        rev_orientation = (sign == 2) ? 0 : 1;
    }

    cvStartReadSeq( ptseq, &ptseq_reader, 0 );
    cvStartReadSeq( hull, &hull_reader, rev_orientation );

    if( !is_index )
    {
        hull_cur = *(CvPoint**)hull_reader.prev_elem;
        index = cvSeqElemIdx( ptseq, (char*)hull_cur, 0 );
    }
    else
    {
        index = *(int*)hull_reader.prev_elem;
        hull_cur = CV_GET_SEQ_ELEM( CvPoint, ptseq, index );
    }
    cvSetSeqReaderPos( &ptseq_reader, index );
    cvStartAppendToSeq( defects, &writer );

    /* cycle through ptseq and hull with computing defects */
    for( i = 0; i < hull->total; i++ )
    {
        CvConvexityDefect defect;
        int is_defect = 0;
        double dx0, dy0;
        double depth = 0, scale;
        CvPoint* hull_next;

        if( !is_index )
            hull_next = *(CvPoint**)hull_reader.ptr;
        else
        {
            int t = *(int*)hull_reader.ptr;
            hull_next = CV_GET_SEQ_ELEM( CvPoint, ptseq, t );
        }
        CV_Assert(hull_next != NULL && hull_cur != NULL);

        dx0 = (double)hull_next->x - (double)hull_cur->x;
        dy0 = (double)hull_next->y - (double)hull_cur->y;
        assert( dx0 != 0 || dy0 != 0 );
        scale = 1./std::sqrt(dx0*dx0 + dy0*dy0);

        defect.start = hull_cur;
        defect.end = hull_next;

        for(;;)
        {
            /* go through ptseq to achieve next hull point */
            CV_NEXT_SEQ_ELEM( sizeof(CvPoint), ptseq_reader );

            if( ptseq_reader.ptr == (schar*)hull_next )
                break;
            else
            {
                CvPoint* cur = (CvPoint*)ptseq_reader.ptr;

                /* compute distance from current point to hull edge */
                double dx = (double)cur->x - (double)hull_cur->x;
                double dy = (double)cur->y - (double)hull_cur->y;

                /* compute depth */
                double dist = fabs(-dy0*dx + dx0*dy) * scale;

                if( dist > depth )
                {
                    depth = dist;
                    defect.depth_point = cur;
                    defect.depth = (float)depth;
                    is_defect = 1;
                }
            }
        }
        if( is_defect )
        {
            CV_WRITE_SEQ_ELEM( defect, writer );
        }

        hull_cur = hull_next;
        if( rev_orientation )
        {
            CV_PREV_SEQ_ELEM( hull->elem_size, hull_reader );
        }
        else
        {
            CV_NEXT_SEQ_ELEM( hull->elem_size, hull_reader );
        }
    }

    return cvEndWriteSeq( &writer );
}


CV_IMPL int
cvCheckContourConvexity( const CvArr* array )
{
    CvContour contour_header;
    CvSeqBlock block;
    CvSeq* contour = (CvSeq*)array;

    if( CV_IS_SEQ(contour) )
    {
        if( !CV_IS_SEQ_POINT_SET(contour))
            CV_Error( CV_StsUnsupportedFormat,
                     "Input sequence must be polygon (closed 2d curve)" );
    }
    else
    {
        contour = cvPointSeqFromMat(CV_SEQ_KIND_CURVE|
            CV_SEQ_FLAG_CLOSED, array, &contour_header, &block );
    }

    if( contour->total == 0 )
        return -1;

    cv::AutoBuffer<double> _buf;
    return cv::isContourConvex(cv::cvarrToMat(contour, false, false, 0, &_buf)) ? 1 : 0;
}

/* End of file. */
