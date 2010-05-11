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

static int
icvSklansky_32s( CvPoint** array, int start, int end, int* stack, int nsign, int sign2 )
{
    int incr = end > start ? 1 : -1;
    /* prepare first triangle */
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

    end += incr; /* make end = afterend */

    while( pnext != end )
    {
        /* check the angle p1,p2,p3 */
        int cury = array[pcur]->y;
        int nexty = array[pnext]->y;
        int by = nexty - cury;

        if( CV_SIGN(by) != nsign )
        {
            int ax = array[pcur]->x - array[pprev]->x;
            int bx = array[pnext]->x - array[pcur]->x;
            int ay = cury - array[pprev]->y;
            int convexity = ay*bx - ax*by;/* if >0 then convex angle */

            if( CV_SIGN(convexity) == sign2 && (ax != 0 || ay != 0) )
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


static int
icvSklansky_32f( CvPoint2D32f** array, int start, int end, int* stack, int nsign, int sign2 )
{
    int incr = end > start ? 1 : -1;
    /* prepare first triangle */
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

    end += incr; /* make end = afterend */

    while( pnext != end )
    {
        /* check the angle p1,p2,p3 */
        float cury = array[pcur]->y;
        float nexty = array[pnext]->y;
        float by = nexty - cury;

        if( CV_SIGN( by ) != nsign )
        {
            float ax = array[pcur]->x - array[pprev]->x;
            float bx = array[pnext]->x - array[pcur]->x;
            float ay = cury - array[pprev]->y;
            float convexity = ay*bx - ax*by;/* if >0 then convex angle */

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

typedef int (*sklansky_func)( CvPoint** points, int start, int end,
                              int* stack, int sign, int sign2 );

#define cmp_pts( pt1, pt2 )  \
    ((pt1)->x < (pt2)->x || ((pt1)->x <= (pt2)->x && (pt1)->y < (pt2)->y))
static CV_IMPLEMENT_QSORT( icvSortPointsByPointers_32s, CvPoint*, cmp_pts )
static CV_IMPLEMENT_QSORT( icvSortPointsByPointers_32f, CvPoint2D32f*, cmp_pts )

static void
icvCalcAndWritePtIndices( CvPoint** pointer, int* stack, int start, int end,
                          CvSeq* ptseq, CvSeqWriter* writer )
{
    int i, incr = start < end ? 1 : -1;
    int idx, first_idx = ptseq->first->start_index;

    for( i = start; i != end; i += incr )
    {
        CvPoint* ptr = (CvPoint*)pointer[stack[i]];
        CvSeqBlock* block = ptseq->first;
        while( (unsigned)(idx = (int)(ptr - (CvPoint*)block->data)) >= (unsigned)block->count )
        {
            block = block->next;
            if( block == ptseq->first )
                CV_Error( CV_StsError, "Internal error" );
        }
        idx += block->start_index - first_idx;
        CV_WRITE_SEQ_ELEM( idx, *writer );
    }
}


CV_IMPL CvSeq*
cvConvexHull2( const CvArr* array, void* hull_storage,
               int orientation, int return_points )
{
    union { CvContour* c; CvSeq* s; } hull;
    cv::AutoBuffer<CvPoint*> _pointer;
    CvPoint** pointer;
    CvPoint2D32f** pointerf = 0;
    cv::AutoBuffer<int> _stack;
    int* stack;

    hull.s = 0;

    CvMat* mat = 0;
    CvSeqReader reader;
    CvSeqWriter writer;
    CvContour contour_header;
    union { CvContour c; CvSeq s; } hull_header;
    CvSeqBlock block, hullblock;
    CvSeq* ptseq = 0;
    CvSeq* hullseq = 0;
    int is_float;
    int* t_stack;
    int t_count;
    int i, miny_ind = 0, maxy_ind = 0, total;
    int hulltype;
    int stop_idx;
    sklansky_func sklansky;

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

    if( CV_IS_STORAGE( hull_storage ))
    {
        if( return_points )
        {
            hullseq = cvCreateSeq(
                CV_SEQ_KIND_CURVE|CV_SEQ_ELTYPE(ptseq)|
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
        if( !CV_IS_MAT( hull_storage ))
            CV_Error(CV_StsBadArg, "Destination must be valid memory storage or matrix");

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
            sizeof(contour_header), CV_ELEM_SIZE(mat->type), mat->data.ptr,
            mat->cols + mat->rows - 1, &hull_header.s, &hullblock );

        cvClearSeq( hullseq );
    }

    total = ptseq->total;
    if( total == 0 )
    {
        if( mat )
            CV_Error( CV_StsBadSize,
            "Point sequence can not be empty if the output is matrix" );
        return hull.s;
    }

    cvStartAppendToSeq( hullseq, &writer );

    is_float = CV_SEQ_ELTYPE(ptseq) == CV_32FC2;
    hulltype = CV_SEQ_ELTYPE(hullseq);
    sklansky = !is_float ? (sklansky_func)icvSklansky_32s :
                           (sklansky_func)icvSklansky_32f;

    _pointer.allocate( ptseq->total );
    _stack.allocate( ptseq->total + 2);
    pointer = _pointer;
    pointerf = (CvPoint2D32f**)pointer;
    stack = _stack;

    cvStartReadSeq( ptseq, &reader );

    for( i = 0; i < total; i++ )
    {
        pointer[i] = (CvPoint*)reader.ptr;
        CV_NEXT_SEQ_ELEM( ptseq->elem_size, reader );
    }

    // sort the point set by x-coordinate, find min and max y
    if( !is_float )
    {
        icvSortPointsByPointers_32s( pointer, total, 0 );
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
        icvSortPointsByPointers_32f( pointerf, total, 0 );
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
        if( hulltype == CV_SEQ_ELTYPE_PPOINT )
        {
            CV_WRITE_SEQ_ELEM( pointer[0], writer );
        }
        else if( hulltype == CV_SEQ_ELTYPE_INDEX )
        {
            int index = 0;
            CV_WRITE_SEQ_ELEM( index, writer );
        }
        else
        {
            CvPoint pt = pointer[0][0];
            CV_WRITE_SEQ_ELEM( pt, writer );
        }
        goto finish_hull;
    }

    /*upper half */
    {
        int *tl_stack = stack;
        int tl_count = sklansky( pointer, 0, maxy_ind, tl_stack, -1, 1 );
        int *tr_stack = tl_stack + tl_count;
        int tr_count = sklansky( pointer, ptseq->total - 1, maxy_ind, tr_stack, -1, -1 );

        /* gather upper part of convex hull to output */
        if( orientation == CV_COUNTER_CLOCKWISE )
        {
            CV_SWAP( tl_stack, tr_stack, t_stack );
            CV_SWAP( tl_count, tr_count, t_count );
        }

        if( hulltype == CV_SEQ_ELTYPE_PPOINT )
        {
            for( i = 0; i < tl_count - 1; i++ )
                CV_WRITE_SEQ_ELEM( pointer[tl_stack[i]], writer );

            for( i = tr_count - 1; i > 0; i-- )
                CV_WRITE_SEQ_ELEM( pointer[tr_stack[i]], writer );
        }
        else if( hulltype == CV_SEQ_ELTYPE_INDEX )
        {
            icvCalcAndWritePtIndices( pointer, tl_stack, 0, tl_count-1, ptseq, &writer );
            icvCalcAndWritePtIndices( pointer, tr_stack, tr_count-1, 0, ptseq, &writer );
        }
        else
        {
            for( i = 0; i < tl_count - 1; i++ )
                CV_WRITE_SEQ_ELEM( pointer[tl_stack[i]][0], writer );

            for( i = tr_count - 1; i > 0; i-- )
                CV_WRITE_SEQ_ELEM( pointer[tr_stack[i]][0], writer );
        }
        stop_idx = tr_count > 2 ? tr_stack[1] : tl_count > 2 ? tl_stack[tl_count - 2] : -1;
    }

    /* lower half */
    {
        int *bl_stack = stack;
        int bl_count = sklansky( pointer, 0, miny_ind, bl_stack, 1, -1 );
        int *br_stack = stack + bl_count;
        int br_count = sklansky( pointer, ptseq->total - 1, miny_ind, br_stack, 1, 1 );

        if( orientation != CV_COUNTER_CLOCKWISE )
        {
            CV_SWAP( bl_stack, br_stack, t_stack );
            CV_SWAP( bl_count, br_count, t_count );
        }

        if( stop_idx >= 0 )
        {
            int check_idx = bl_count > 2 ? bl_stack[1] :
                            bl_count + br_count > 2 ? br_stack[2-bl_count] : -1;
            if( check_idx == stop_idx || (check_idx >= 0 &&
                pointer[check_idx]->x == pointer[stop_idx]->x &&
                pointer[check_idx]->y == pointer[stop_idx]->y) )
            {
                /* if all the points lie on the same line, then
                   the bottom part of the convex hull is the mirrored top part
                   (except the exteme points).*/
                bl_count = MIN( bl_count, 2 );
                br_count = MIN( br_count, 2 );
            }
        }

        if( hulltype == CV_SEQ_ELTYPE_PPOINT )
        {
            for( i = 0; i < bl_count - 1; i++ )
                CV_WRITE_SEQ_ELEM( pointer[bl_stack[i]], writer );

            for( i = br_count - 1; i > 0; i-- )
                CV_WRITE_SEQ_ELEM( pointer[br_stack[i]], writer );
        }
        else if( hulltype == CV_SEQ_ELTYPE_INDEX )
        {
            icvCalcAndWritePtIndices( pointer, bl_stack, 0, bl_count-1, ptseq, &writer );
            icvCalcAndWritePtIndices( pointer, br_stack, br_count-1, 0, ptseq, &writer );
        }
        else
        {
            for( i = 0; i < bl_count - 1; i++ )
                CV_WRITE_SEQ_ELEM( pointer[bl_stack[i]][0], writer );

            for( i = br_count - 1; i > 0; i-- )
                CV_WRITE_SEQ_ELEM( pointer[br_stack[i]][0], writer );
        }
    }

finish_hull:
    cvEndWriteSeq( &writer );

    if( mat )
    {
        if( mat->rows > mat->cols )
            mat->rows = hullseq->total;
        else
            mat->cols = hullseq->total;
    }
    else
    {
        hull.s = hullseq;
        hull.c->rect = cvBoundingRect( ptseq,
            ptseq->header_size < (int)sizeof(CvContour) ||
            &ptseq->flags == &contour_header.flags );

        /*if( ptseq != (CvSeq*)&contour_header )
            hullseq->v_prev = ptseq;*/
    }

    return hull.s;
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
    union { CvContour c; CvSeq s; } hull_header;
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
            sizeof(CvContour), CV_ELEM_SIZE(mat->type), mat->data.ptr,
            mat->cols + mat->rows - 1, &hull_header.s, &hullblock );
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

        dx0 = (double)hull_next->x - (double)hull_cur->x;
        dy0 = (double)hull_next->y - (double)hull_cur->y;
        assert( dx0 != 0 || dy0 != 0 );
        scale = 1./sqrt(dx0*dx0 + dy0*dy0);

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
    int flag = -1;

    int i;
    int orientation = 0;
    CvSeqReader reader;
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
        contour = cvPointSeqFromMat(CV_SEQ_KIND_CURVE|CV_SEQ_FLAG_CLOSED, array, &contour_header, &block );
    }

    if( contour->total == 0 )
        return -1;

    cvStartReadSeq( contour, &reader, 0 );
    flag = 1;

    if( CV_SEQ_ELTYPE( contour ) == CV_32SC2 )
    {
        CvPoint *prev_pt = (CvPoint*)reader.prev_elem;
        CvPoint *cur_pt = (CvPoint*)reader.ptr;

        int dx0 = cur_pt->x - prev_pt->x;
        int dy0 = cur_pt->y - prev_pt->y;

        for( i = 0; i < contour->total; i++ )
        {
            int dxdy0, dydx0;
            int dx, dy;

            /*int orient; */
            CV_NEXT_SEQ_ELEM( sizeof(CvPoint), reader );
            prev_pt = cur_pt;
            cur_pt = (CvPoint *) reader.ptr;

            dx = cur_pt->x - prev_pt->x;
            dy = cur_pt->y - prev_pt->y;
            dxdy0 = dx * dy0;
            dydx0 = dy * dx0;

            /* find orientation */
            /*orient = -dy0 * dx + dx0 * dy;
               orientation |= (orient > 0) ? 1 : 2;
             */
            orientation |= (dydx0 > dxdy0) ? 1 : ((dydx0 < dxdy0) ? 2 : 3);

            if( orientation == 3 )
            {
                flag = 0;
                break;
            }

            dx0 = dx;
            dy0 = dy;
        }
    }
    else
    {
        CV_Assert( CV_SEQ_ELTYPE(contour) == CV_32FC2 );

        CvPoint2D32f *prev_pt = (CvPoint2D32f*)reader.prev_elem;
        CvPoint2D32f *cur_pt = (CvPoint2D32f*)reader.ptr;

        float dx0 = cur_pt->x - prev_pt->x;
        float dy0 = cur_pt->y - prev_pt->y;

        for( i = 0; i < contour->total; i++ )
        {
            float dxdy0, dydx0;
            float dx, dy;

            /*int orient; */
            CV_NEXT_SEQ_ELEM( sizeof(CvPoint2D32f), reader );
            prev_pt = cur_pt;
            cur_pt = (CvPoint2D32f*) reader.ptr;

            dx = cur_pt->x - prev_pt->x;
            dy = cur_pt->y - prev_pt->y;
            dxdy0 = dx * dy0;
            dydx0 = dy * dx0;

            /* find orientation */
            /*orient = -dy0 * dx + dx0 * dy;
               orientation |= (orient > 0) ? 1 : 2;
             */
            orientation |= (dydx0 > dxdy0) ? 1 : ((dydx0 < dxdy0) ? 2 : 3);

            if( orientation == 3 )
            {
                flag = 0;
                break;
            }

            dx0 = dx;
            dy0 = dy;
        }
    }

    return flag;
}


/* End of file. */
