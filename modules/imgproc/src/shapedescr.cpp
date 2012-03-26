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

/* calculates length of a curve (e.g. contour perimeter) */
CV_IMPL  double
cvArcLength( const void *array, CvSlice slice, int is_closed )
{
    double perimeter = 0;

    int i, j = 0, count;
    const int N = 16;
    float buf[N];
    CvMat buffer = cvMat( 1, N, CV_32F, buf ); 
    CvSeqReader reader;
    CvContour contour_header;
    CvSeq* contour = 0;
    CvSeqBlock block;

    if( CV_IS_SEQ( array ))
    {
        contour = (CvSeq*)array;
        if( !CV_IS_SEQ_POLYLINE( contour ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );
        if( is_closed < 0 )
            is_closed = CV_IS_SEQ_CLOSED( contour );
    }
    else
    {
        is_closed = is_closed > 0;
        contour = cvPointSeqFromMat(
            CV_SEQ_KIND_CURVE | (is_closed ? CV_SEQ_FLAG_CLOSED : 0),
            array, &contour_header, &block );
    }

    if( contour->total > 1 )
    {
        int is_float = CV_SEQ_ELTYPE( contour ) == CV_32FC2;
        
        cvStartReadSeq( contour, &reader, 0 );
        cvSetSeqReaderPos( &reader, slice.start_index );
        count = cvSliceLength( slice, contour );

        count -= !is_closed && count == contour->total;

        /* scroll the reader by 1 point */
        reader.prev_elem = reader.ptr;
        CV_NEXT_SEQ_ELEM( sizeof(CvPoint), reader );

        for( i = 0; i < count; i++ )
        {
            float dx, dy;

            if( !is_float )
            {
                CvPoint* pt = (CvPoint*)reader.ptr;
                CvPoint* prev_pt = (CvPoint*)reader.prev_elem;

                dx = (float)pt->x - (float)prev_pt->x;
                dy = (float)pt->y - (float)prev_pt->y;
            }
            else
            {
                CvPoint2D32f* pt = (CvPoint2D32f*)reader.ptr;
                CvPoint2D32f* prev_pt = (CvPoint2D32f*)reader.prev_elem;

                dx = pt->x - prev_pt->x;
                dy = pt->y - prev_pt->y;
            }

            reader.prev_elem = reader.ptr;
            CV_NEXT_SEQ_ELEM( contour->elem_size, reader );

            buffer.data.fl[j] = dx * dx + dy * dy;
            if( ++j == N || i == count - 1 )
            {
                buffer.cols = j;
                cvPow( &buffer, &buffer, 0.5 );
                for( ; j > 0; j-- )
                    perimeter += buffer.data.fl[j-1];
            }
        }
    }

    return perimeter;
}


static CvStatus
icvFindCircle( CvPoint2D32f pt0, CvPoint2D32f pt1,
               CvPoint2D32f pt2, CvPoint2D32f * center, float *radius )
{
    double x1 = (pt0.x + pt1.x) * 0.5;
    double dy1 = pt0.x - pt1.x;
    double x2 = (pt1.x + pt2.x) * 0.5;
    double dy2 = pt1.x - pt2.x;
    double y1 = (pt0.y + pt1.y) * 0.5;
    double dx1 = pt1.y - pt0.y;
    double y2 = (pt1.y + pt2.y) * 0.5;
    double dx2 = pt2.y - pt1.y;
    double t = 0;

    CvStatus result = CV_OK;

    if( icvIntersectLines( x1, dx1, y1, dy1, x2, dx2, y2, dy2, &t ) >= 0 )
    {
        center->x = (float) (x2 + dx2 * t);
        center->y = (float) (y2 + dy2 * t);
        *radius = (float) icvDistanceL2_32f( *center, pt0 );
    }
    else
    {
        center->x = center->y = 0.f;
        radius = 0;
        result = CV_NOTDEFINED_ERR;
    }

    return result;
}


CV_INLINE double icvIsPtInCircle( CvPoint2D32f pt, CvPoint2D32f center, float radius )
{
    double dx = pt.x - center.x;
    double dy = pt.y - center.y;
    return (double)radius*radius - dx*dx - dy*dy;
}


static int
icvFindEnslosingCicle4pts_32f( CvPoint2D32f * pts, CvPoint2D32f * _center, float *_radius )
{
    int shuffles[4][4] = { {0, 1, 2, 3}, {0, 1, 3, 2}, {2, 3, 0, 1}, {2, 3, 1, 0} };

    int idxs[4] = { 0, 1, 2, 3 };
    int i, j, k = 1, mi = 0;
    float max_dist = 0;
    CvPoint2D32f center;
    CvPoint2D32f min_center;
    float radius, min_radius = FLT_MAX;
    CvPoint2D32f res_pts[4];

    center = min_center = pts[0];
    radius = 1.f;

    for( i = 0; i < 4; i++ )
        for( j = i + 1; j < 4; j++ )
        {
            float dist = icvDistanceL2_32f( pts[i], pts[j] );

            if( max_dist < dist )
            {
                max_dist = dist;
                idxs[0] = i;
                idxs[1] = j;
            }
        }

    if( max_dist == 0 )
        goto function_exit;

    k = 2;
    for( i = 0; i < 4; i++ )
    {
        for( j = 0; j < k; j++ )
            if( i == idxs[j] )
                break;
        if( j == k )
            idxs[k++] = i;
    }

    center = cvPoint2D32f( (pts[idxs[0]].x + pts[idxs[1]].x)*0.5f,
                           (pts[idxs[0]].y + pts[idxs[1]].y)*0.5f );
    radius = (float)(icvDistanceL2_32f( pts[idxs[0]], center )*1.03);
    if( radius < 1.f )
        radius = 1.f;

    if( icvIsPtInCircle( pts[idxs[2]], center, radius ) >= 0 &&
        icvIsPtInCircle( pts[idxs[3]], center, radius ) >= 0 )
    {
        k = 2; //rand()%2+2;
    }
    else
    {
        mi = -1;
        for( i = 0; i < 4; i++ )
        {
            if( icvFindCircle( pts[shuffles[i][0]], pts[shuffles[i][1]],
                               pts[shuffles[i][2]], &center, &radius ) >= 0 )
            {
                radius *= 1.03f;
                if( radius < 2.f )
                    radius = 2.f;

                if( icvIsPtInCircle( pts[shuffles[i][3]], center, radius ) >= 0 &&
                    min_radius > radius )
                {
                    min_radius = radius;
                    min_center = center;
                    mi = i;
                }
            }
        }
        assert( mi >= 0 );
        if( mi < 0 )
            mi = 0;
        k = 3;
        center = min_center;
        radius = min_radius;
        for( i = 0; i < 4; i++ )
            idxs[i] = shuffles[mi][i];
    }

  function_exit:

    *_center = center;
    *_radius = radius;

    /* reorder output points */
    for( i = 0; i < 4; i++ )
        res_pts[i] = pts[idxs[i]];

    for( i = 0; i < 4; i++ )
    {
        pts[i] = res_pts[i];
        assert( icvIsPtInCircle( pts[i], center, radius ) >= 0 );
    }

    return k;
}


CV_IMPL int
cvMinEnclosingCircle( const void* array, CvPoint2D32f * _center, float *_radius )
{
    const int max_iters = 100;
    const float eps = FLT_EPSILON*2;
    CvPoint2D32f center = { 0, 0 };
    float radius = 0;
    int result = 0;

    if( _center )
        _center->x = _center->y = 0.f;
    if( _radius )
        *_radius = 0;

    CvSeqReader reader;
    int i, k, count;
    CvPoint2D32f pts[8];
    CvContour contour_header;
    CvSeqBlock block;
    CvSeq* sequence = 0;
    int is_float;

    if( !_center || !_radius )
        CV_Error( CV_StsNullPtr, "Null center or radius pointers" );

    if( CV_IS_SEQ(array) )
    {
        sequence = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( sequence ))
            CV_Error( CV_StsBadArg, "The passed sequence is not a valid contour" );
    }
    else
    {
        sequence = cvPointSeqFromMat(
            CV_SEQ_KIND_GENERIC, array, &contour_header, &block );
    }

    if( sequence->total <= 0 )
        CV_Error( CV_StsBadSize, "" );

    cvStartReadSeq( sequence, &reader, 0 );

    count = sequence->total;
    is_float = CV_SEQ_ELTYPE(sequence) == CV_32FC2;

    if( !is_float )
    {
        CvPoint *pt_left, *pt_right, *pt_top, *pt_bottom;
        CvPoint pt;
        pt_left = pt_right = pt_top = pt_bottom = (CvPoint *)(reader.ptr);
        CV_READ_SEQ_ELEM( pt, reader );

        for( i = 1; i < count; i++ )
        {
            CvPoint* pt_ptr = (CvPoint*)reader.ptr;
            CV_READ_SEQ_ELEM( pt, reader );

            if( pt.x < pt_left->x )
                pt_left = pt_ptr;
            if( pt.x > pt_right->x )
                pt_right = pt_ptr;
            if( pt.y < pt_top->y )
                pt_top = pt_ptr;
            if( pt.y > pt_bottom->y )
                pt_bottom = pt_ptr;
        }

        pts[0] = cvPointTo32f( *pt_left );
        pts[1] = cvPointTo32f( *pt_right );
        pts[2] = cvPointTo32f( *pt_top );
        pts[3] = cvPointTo32f( *pt_bottom );
    }
    else
    {
        CvPoint2D32f *pt_left, *pt_right, *pt_top, *pt_bottom;
        CvPoint2D32f pt;
        pt_left = pt_right = pt_top = pt_bottom = (CvPoint2D32f *) (reader.ptr);
        CV_READ_SEQ_ELEM( pt, reader );

        for( i = 1; i < count; i++ )
        {
            CvPoint2D32f* pt_ptr = (CvPoint2D32f*)reader.ptr;
            CV_READ_SEQ_ELEM( pt, reader );

            if( pt.x < pt_left->x )
                pt_left = pt_ptr;
            if( pt.x > pt_right->x )
                pt_right = pt_ptr;
            if( pt.y < pt_top->y )
                pt_top = pt_ptr;
            if( pt.y > pt_bottom->y )
                pt_bottom = pt_ptr;
        }

        pts[0] = *pt_left;
        pts[1] = *pt_right;
        pts[2] = *pt_top;
        pts[3] = *pt_bottom;
    }

    for( k = 0; k < max_iters; k++ )
    {
        double min_delta = 0, delta;
		CvPoint2D32f ptfl, farAway = { 0, 0};
		/*only for first iteration because the alg is repared at the loop's foot*/
		if(k==0)
			icvFindEnslosingCicle4pts_32f( pts, &center, &radius );

        cvStartReadSeq( sequence, &reader, 0 );

        for( i = 0; i < count; i++ )
        {
            if( !is_float )
            {
                ptfl.x = (float)((CvPoint*)reader.ptr)->x;
                ptfl.y = (float)((CvPoint*)reader.ptr)->y;
            }
            else
            {
                ptfl = *(CvPoint2D32f*)reader.ptr;
            }
            CV_NEXT_SEQ_ELEM( sequence->elem_size, reader );

            delta = icvIsPtInCircle( ptfl, center, radius );
            if( delta < min_delta )
            {
                min_delta = delta;
                farAway = ptfl;
            }
        }
        result = min_delta >= 0;
        if( result )
            break;

		CvPoint2D32f ptsCopy[4];
		/* find good replacement partner for the point which is at most far away, 
		starting with the one that lays in the actual circle (i=3) */
		for(int i = 3; i >=0; i-- )
		{
			for(int j = 0; j < 4; j++ )
			{
				ptsCopy[j]=(i != j)? pts[j]: farAway;
			}

			icvFindEnslosingCicle4pts_32f(ptsCopy, &center, &radius );
			if( icvIsPtInCircle( pts[i], center, radius )>=0){ // replaced one again in the new circle?
				pts[i] = farAway;
				break;
			}
		}
    }

    if( !result )
    {
        cvStartReadSeq( sequence, &reader, 0 );
        radius = 0.f;

        for( i = 0; i < count; i++ )
        {
            CvPoint2D32f ptfl;
            float t, dx, dy;

            if( !is_float )
            {
                ptfl.x = (float)((CvPoint*)reader.ptr)->x;
                ptfl.y = (float)((CvPoint*)reader.ptr)->y;
            }
            else
            {
                ptfl = *(CvPoint2D32f*)reader.ptr;
            }

            CV_NEXT_SEQ_ELEM( sequence->elem_size, reader );
            dx = center.x - ptfl.x;
            dy = center.y - ptfl.y;
            t = dx*dx + dy*dy;
            radius = MAX(radius,t);
        }

        radius = (float)(sqrt(radius)*(1 + eps));
        result = 1;
    }

    *_center = center;
    *_radius = radius;

    return result;
}


/* area of a whole sequence */
static CvStatus
icvContourArea( const CvSeq* contour, double *area )
{
    if( contour->total )
    {
        CvSeqReader reader;
        int lpt = contour->total;
        double a00 = 0, xi_1, yi_1;
        int is_float = CV_SEQ_ELTYPE(contour) == CV_32FC2;

        cvStartReadSeq( contour, &reader, 0 );

        if( !is_float )
        {
            xi_1 = ((CvPoint*)(reader.ptr))->x;
            yi_1 = ((CvPoint*)(reader.ptr))->y;
        }
        else
        {
            xi_1 = ((CvPoint2D32f*)(reader.ptr))->x;
            yi_1 = ((CvPoint2D32f*)(reader.ptr))->y;
        }
        CV_NEXT_SEQ_ELEM( contour->elem_size, reader );
        
        while( lpt-- > 0 )
        {
            double dxy, xi, yi;

            if( !is_float )
            {
                xi = ((CvPoint*)(reader.ptr))->x;
                yi = ((CvPoint*)(reader.ptr))->y;
            }
            else
            {
                xi = ((CvPoint2D32f*)(reader.ptr))->x;
                yi = ((CvPoint2D32f*)(reader.ptr))->y;
            }
            CV_NEXT_SEQ_ELEM( contour->elem_size, reader );

            dxy = xi_1 * yi - xi * yi_1;
            a00 += dxy;
            xi_1 = xi;
            yi_1 = yi;
        }

        *area = a00 * 0.5;
    }
    else
        *area = 0;

    return CV_OK;
}


/****************************************************************************************\

 copy data from one buffer to other buffer 

\****************************************************************************************/

static CvStatus
icvMemCopy( double **buf1, double **buf2, double **buf3, int *b_max )
{
    int bb;

    if( (*buf1 == NULL && *buf2 == NULL) || *buf3 == NULL )
        return CV_NULLPTR_ERR;

    bb = *b_max;
    if( *buf2 == NULL )
    {
        *b_max = 2 * (*b_max);
        *buf2 = (double *)cvAlloc( (*b_max) * sizeof( double ));

        if( *buf2 == NULL )
            return CV_OUTOFMEM_ERR;

        memcpy( *buf2, *buf3, bb * sizeof( double ));

        *buf3 = *buf2;
        cvFree( buf1 );
        *buf1 = NULL;
    }
    else
    {
        *b_max = 2 * (*b_max);
        *buf1 = (double *) cvAlloc( (*b_max) * sizeof( double ));

        if( *buf1 == NULL )
            return CV_OUTOFMEM_ERR;

        memcpy( *buf1, *buf3, bb * sizeof( double ));

        *buf3 = *buf1;
        cvFree( buf2 );
        *buf2 = NULL;
    }
    return CV_OK;
}


/* area of a contour sector */
static CvStatus icvContourSecArea( CvSeq * contour, CvSlice slice, double *area )
{
    CvPoint pt;                 /*  pointer to points   */
    CvPoint pt_s, pt_e;         /*  first and last points  */
    CvSeqReader reader;         /*  points reader of contour   */

    int p_max = 2, p_ind;
    int lpt, flag, i;
    double a00;                 /* unnormalized moments m00    */
    double xi, yi, xi_1, yi_1, x0, y0, dxy, sk, sk1, t;
    double x_s, y_s, nx, ny, dx, dy, du, dv;
    double eps = 1.e-5;
    double *p_are1, *p_are2, *p_are;

    assert( contour != NULL );

    if( contour == NULL )
        return CV_NULLPTR_ERR;

    if( !CV_IS_SEQ_POINT_SET( contour ))
        return CV_BADFLAG_ERR;

    lpt = cvSliceLength( slice, contour );
    /*if( n2 >= n1 )
        lpt = n2 - n1 + 1;
    else
        lpt = contour->total - n1 + n2 + 1;*/

    if( contour->total && lpt > 2 )
    {
        a00 = x0 = y0 = xi_1 = yi_1 = 0;
        sk1 = 0;
        flag = 0;
        dxy = 0;
        p_are1 = (double *) cvAlloc( p_max * sizeof( double ));

        if( p_are1 == NULL )
            return CV_OUTOFMEM_ERR;

        p_are = p_are1;
        p_are2 = NULL;

        cvStartReadSeq( contour, &reader, 0 );
        cvSetSeqReaderPos( &reader, slice.start_index );
        CV_READ_SEQ_ELEM( pt_s, reader );
        p_ind = 0;
        cvSetSeqReaderPos( &reader, slice.end_index );
        CV_READ_SEQ_ELEM( pt_e, reader );

/*    normal coefficients    */
        nx = pt_s.y - pt_e.y;
        ny = pt_e.x - pt_s.x;
        cvSetSeqReaderPos( &reader, slice.start_index );

        while( lpt-- > 0 )
        {
            CV_READ_SEQ_ELEM( pt, reader );

            if( flag == 0 )
            {
                xi_1 = (double) pt.x;
                yi_1 = (double) pt.y;
                x0 = xi_1;
                y0 = yi_1;
                sk1 = 0;
                flag = 1;
            }
            else
            {
                xi = (double) pt.x;
                yi = (double) pt.y;

/****************   edges intersection examination   **************************/
                sk = nx * (xi - pt_s.x) + ny * (yi - pt_s.y);
                if( (fabs( sk ) < eps && lpt > 0) || sk * sk1 < -eps )
                {
                    if( fabs( sk ) < eps )
                    {
                        dxy = xi_1 * yi - xi * yi_1;
                        a00 = a00 + dxy;
                        dxy = xi * y0 - x0 * yi;
                        a00 = a00 + dxy;

                        if( p_ind >= p_max )
                            icvMemCopy( &p_are1, &p_are2, &p_are, &p_max );

                        p_are[p_ind] = a00 / 2.;
                        p_ind++;
                        a00 = 0;
                        sk1 = 0;
                        x0 = xi;
                        y0 = yi;
                        dxy = 0;
                    }
                    else
                    {
/*  define intersection point    */
                        dv = yi - yi_1;
                        du = xi - xi_1;
                        dx = ny;
                        dy = -nx;
                        if( fabs( du ) > eps )
                            t = ((yi_1 - pt_s.y) * du + dv * (pt_s.x - xi_1)) /
                                (du * dy - dx * dv);
                        else
                            t = (xi_1 - pt_s.x) / dx;
                        if( t > eps && t < 1 - eps )
                        {
                            x_s = pt_s.x + t * dx;
                            y_s = pt_s.y + t * dy;
                            dxy = xi_1 * y_s - x_s * yi_1;
                            a00 += dxy;
                            dxy = x_s * y0 - x0 * y_s;
                            a00 += dxy;
                            if( p_ind >= p_max )
                                icvMemCopy( &p_are1, &p_are2, &p_are, &p_max );

                            p_are[p_ind] = a00 / 2.;
                            p_ind++;

                            a00 = 0;
                            sk1 = 0;
                            x0 = x_s;
                            y0 = y_s;
                            dxy = x_s * yi - xi * y_s;
                        }
                    }
                }
                else
                    dxy = xi_1 * yi - xi * yi_1;

                a00 += dxy;
                xi_1 = xi;
                yi_1 = yi;
                sk1 = sk;

            }
        }

        xi = x0;
        yi = y0;
        dxy = xi_1 * yi - xi * yi_1;

        a00 += dxy;

        if( p_ind >= p_max )
            icvMemCopy( &p_are1, &p_are2, &p_are, &p_max );

        p_are[p_ind] = a00 / 2.;
        p_ind++;

/*     common area calculation    */
        *area = 0;
        for( i = 0; i < p_ind; i++ )
            (*area) += fabs( p_are[i] );

        if( p_are1 != NULL )
            cvFree( &p_are1 );
        else if( p_are2 != NULL )
            cvFree( &p_are2 );

        return CV_OK;
    }
    else
        return CV_BADSIZE_ERR;
}


/* external contour area function */
CV_IMPL double
cvContourArea( const void *array, CvSlice slice, int oriented )
{
    double area = 0;

    CvContour contour_header;
    CvSeq* contour = 0;
    CvSeqBlock block;

    if( CV_IS_SEQ( array ))
    {
        contour = (CvSeq*)array;
        if( !CV_IS_SEQ_POLYLINE( contour ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );
    }
    else
    {
        contour = cvPointSeqFromMat( CV_SEQ_KIND_CURVE, array, &contour_header, &block );
    }

    if( cvSliceLength( slice, contour ) == contour->total )
    {
        IPPI_CALL( icvContourArea( contour, &area ));
    }
    else
    {
        if( CV_SEQ_ELTYPE( contour ) != CV_32SC2 )
            CV_Error( CV_StsUnsupportedFormat,
            "Only curves with integer coordinates are supported in case of contour slice" );
        IPPI_CALL( icvContourSecArea( contour, slice, &area ));
    }

    return oriented ? area : fabs(area);
}


/* for now this function works bad with singular cases
   You can see in the code, that when some troubles with
   matrices or some variables occur -
   box filled with zero values is returned.
   However in general function works fine.
*/
static void
icvFitEllipse_F( CvSeq* points, CvBox2D* box )
{
    cv::Ptr<CvMat> D;
    double S[36], C[36], T[36];

    int i, j;
    double eigenvalues[6], eigenvectors[36];
    double a, b, c, d, e, f;
    double x0, y0, idet, scale, offx = 0, offy = 0;

    int n = points->total;
    CvSeqReader reader;
    int is_float = CV_SEQ_ELTYPE(points) == CV_32FC2;

    CvMat matS = cvMat(6,6,CV_64F,S), matC = cvMat(6,6,CV_64F,C), matT = cvMat(6,6,CV_64F,T);
    CvMat _EIGVECS = cvMat(6,6,CV_64F,eigenvectors), _EIGVALS = cvMat(6,1,CV_64F,eigenvalues);

    /* create matrix D of  input points */
    D = cvCreateMat( n, 6, CV_64F );
    
    cvStartReadSeq( points, &reader );

    /* shift all points to zero */
    for( i = 0; i < n; i++ )
    {
        if( !is_float )
        {
            offx += ((CvPoint*)reader.ptr)->x;
            offy += ((CvPoint*)reader.ptr)->y;
        }
        else
        {
            offx += ((CvPoint2D32f*)reader.ptr)->x;
            offy += ((CvPoint2D32f*)reader.ptr)->y;
        }
        CV_NEXT_SEQ_ELEM( points->elem_size, reader );
    }

    offx /= n;
    offy /= n;

    // fill matrix rows as (x*x, x*y, y*y, x, y, 1 )
    for( i = 0; i < n; i++ )
    {
        double x, y;
        double* Dptr = D->data.db + i*6;
        
        if( !is_float )
        {
            x = ((CvPoint*)reader.ptr)->x - offx;
            y = ((CvPoint*)reader.ptr)->y - offy;
        }
        else
        {
            x = ((CvPoint2D32f*)reader.ptr)->x - offx;
            y = ((CvPoint2D32f*)reader.ptr)->y - offy;
        }
        CV_NEXT_SEQ_ELEM( points->elem_size, reader );
        
        Dptr[0] = x * x;
        Dptr[1] = x * y;
        Dptr[2] = y * y;
        Dptr[3] = x;
        Dptr[4] = y;
        Dptr[5] = 1.;
    }

    // S = D^t*D
    cvMulTransposed( D, &matS, 1 );
    cvSVD( &matS, &_EIGVALS, &_EIGVECS, 0, CV_SVD_MODIFY_A + CV_SVD_U_T );

    for( i = 0; i < 6; i++ )
    {
        double a = eigenvalues[i];
        a = a < DBL_EPSILON ? 0 : 1./sqrt(sqrt(a));
        for( j = 0; j < 6; j++ )
            eigenvectors[i*6 + j] *= a;
    }

    // C = Q^-1 = transp(INVEIGV) * INVEIGV
    cvMulTransposed( &_EIGVECS, &matC, 1 );
    
    cvZero( &matS );
    S[2] = 2.;
    S[7] = -1.;
    S[12] = 2.;

    // S = Q^-1*S*Q^-1
    cvMatMul( &matC, &matS, &matT );
    cvMatMul( &matT, &matC, &matS );

    // and find its eigenvalues and vectors too
    //cvSVD( &matS, &_EIGVALS, &_EIGVECS, 0, CV_SVD_MODIFY_A + CV_SVD_U_T );
    cvEigenVV( &matS, &_EIGVECS, &_EIGVALS, 0 );

    for( i = 0; i < 3; i++ )
        if( eigenvalues[i] > 0 )
            break;

    if( i >= 3 /*eigenvalues[0] < DBL_EPSILON*/ )
    {
        box->center.x = box->center.y = 
        box->size.width = box->size.height = 
        box->angle = 0.f;
        return;
    }

    // now find truthful eigenvector
    _EIGVECS = cvMat( 6, 1, CV_64F, eigenvectors + 6*i );
    matT = cvMat( 6, 1, CV_64F, T );
    // Q^-1*eigenvecs[0]
    cvMatMul( &matC, &_EIGVECS, &matT );
    
    // extract vector components
    a = T[0]; b = T[1]; c = T[2]; d = T[3]; e = T[4]; f = T[5];
    
    ///////////////// extract ellipse axes from above values ////////////////

    /* 
       1) find center of ellipse 
       it satisfy equation  
       | a     b/2 | *  | x0 | +  | d/2 | = |0 |
       | b/2    c  |    | y0 |    | e/2 |   |0 |

     */
    idet = a * c - b * b * 0.25;
    idet = idet > DBL_EPSILON ? 1./idet : 0;

    // we must normalize (a b c d e f ) to fit (4ac-b^2=1)
    scale = sqrt( 0.25 * idet );

    if( scale < DBL_EPSILON ) 
    {
        box->center.x = (float)offx;
        box->center.y = (float)offy;
        box->size.width = box->size.height = box->angle = 0.f;
        return;
    }
       
    a *= scale;
    b *= scale;
    c *= scale;
    d *= scale;
    e *= scale;
    f *= scale;

    x0 = (-d * c + e * b * 0.5) * 2.;
    y0 = (-a * e + d * b * 0.5) * 2.;

    // recover center
    box->center.x = (float)(x0 + offx);
    box->center.y = (float)(y0 + offy);

    // offset ellipse to (x0,y0)
    // new f == F(x0,y0)
    f += a * x0 * x0 + b * x0 * y0 + c * y0 * y0 + d * x0 + e * y0;

    if( fabs(f) < DBL_EPSILON ) 
    {
        box->size.width = box->size.height = box->angle = 0.f;
        return;
    }

    scale = -1. / f;
    // normalize to f = 1
    a *= scale;
    b *= scale;
    c *= scale;

    // extract axis of ellipse
    // one more eigenvalue operation
    S[0] = a;
    S[1] = S[2] = b * 0.5;
    S[3] = c;

    matS = cvMat( 2, 2, CV_64F, S );
    _EIGVECS = cvMat( 2, 2, CV_64F, eigenvectors );
    _EIGVALS = cvMat( 1, 2, CV_64F, eigenvalues );
    cvSVD( &matS, &_EIGVALS, &_EIGVECS, 0, CV_SVD_MODIFY_A + CV_SVD_U_T );

    // exteract axis length from eigenvectors
    box->size.width = (float)(2./sqrt(eigenvalues[0]));
    box->size.height = (float)(2./sqrt(eigenvalues[1]));

    // calc angle
    box->angle = (float)(180 - atan2(eigenvectors[2], eigenvectors[3])*180/CV_PI);
}


CV_IMPL CvBox2D
cvFitEllipse2( const CvArr* array )
{
    CvBox2D box;
    cv::AutoBuffer<double> Ad, bd;
    memset( &box, 0, sizeof(box));

    CvContour contour_header;
    CvSeq* ptseq = 0;
    CvSeqBlock block;
    int n;

    if( CV_IS_SEQ( array ))
    {
        ptseq = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );
    }
    else
    {
        ptseq = cvPointSeqFromMat(CV_SEQ_KIND_GENERIC, array, &contour_header, &block);
    }

    n = ptseq->total;
    if( n < 5 )
        CV_Error( CV_StsBadSize, "Number of points should be >= 5" );
#if 1
    icvFitEllipse_F( ptseq, &box );
#else
    /*
     *	New fitellipse algorithm, contributed by Dr. Daniel Weiss
     */
    double gfp[5], rp[5], t;
    CvMat A, b, x;
    const double min_eps = 1e-6;
    int i, is_float;
    CvSeqReader reader;

    Ad.allocate(n*5);
    bd.allocate(n);

    // first fit for parameters A - E
    A = cvMat( n, 5, CV_64F, Ad );
    b = cvMat( n, 1, CV_64F, bd );
    x = cvMat( 5, 1, CV_64F, gfp );

    cvStartReadSeq( ptseq, &reader );
    is_float = CV_SEQ_ELTYPE(ptseq) == CV_32FC2;

    for( i = 0; i < n; i++ )
    {
        CvPoint2D32f p;
        if( is_float )
            p = *(CvPoint2D32f*)(reader.ptr);
        else
        {
            p.x = (float)((int*)reader.ptr)[0];
            p.y = (float)((int*)reader.ptr)[1];
        }
        CV_NEXT_SEQ_ELEM( sizeof(p), reader );

        bd[i] = 10000.0; // 1.0?
        Ad[i*5] = -(double)p.x * p.x; // A - C signs inverted as proposed by APP
        Ad[i*5 + 1] = -(double)p.y * p.y;
        Ad[i*5 + 2] = -(double)p.x * p.y;
        Ad[i*5 + 3] = p.x;
        Ad[i*5 + 4] = p.y;
    }
    
    cvSolve( &A, &b, &x, CV_SVD );

    // now use general-form parameters A - E to find the ellipse center:
    // differentiate general form wrt x/y to get two equations for cx and cy
    A = cvMat( 2, 2, CV_64F, Ad );
    b = cvMat( 2, 1, CV_64F, bd );
    x = cvMat( 2, 1, CV_64F, rp );
    Ad[0] = 2 * gfp[0];
    Ad[1] = Ad[2] = gfp[2];
    Ad[3] = 2 * gfp[1];
    bd[0] = gfp[3];
    bd[1] = gfp[4];
    cvSolve( &A, &b, &x, CV_SVD );

    // re-fit for parameters A - C with those center coordinates
    A = cvMat( n, 3, CV_64F, Ad );
    b = cvMat( n, 1, CV_64F, bd );
    x = cvMat( 3, 1, CV_64F, gfp );
    for( i = 0; i < n; i++ )
    {
        CvPoint2D32f p;
        if( is_float )
            p = *(CvPoint2D32f*)(reader.ptr);
        else
        {
            p.x = (float)((int*)reader.ptr)[0];
            p.y = (float)((int*)reader.ptr)[1];
        }
        CV_NEXT_SEQ_ELEM( sizeof(p), reader );
        bd[i] = 1.0;
        Ad[i * 3] = (p.x - rp[0]) * (p.x - rp[0]);
        Ad[i * 3 + 1] = (p.y - rp[1]) * (p.y - rp[1]);
        Ad[i * 3 + 2] = (p.x - rp[0]) * (p.y - rp[1]);
    }
    cvSolve(&A, &b, &x, CV_SVD);

    // store angle and radii
    rp[4] = -0.5 * atan2(gfp[2], gfp[1] - gfp[0]); // convert from APP angle usage
    t = sin(-2.0 * rp[4]);
    if( fabs(t) > fabs(gfp[2])*min_eps )
        t = gfp[2]/t;
    else
        t = gfp[1] - gfp[0];
    rp[2] = fabs(gfp[0] + gfp[1] - t);
    if( rp[2] > min_eps )
        rp[2] = sqrt(2.0 / rp[2]);
    rp[3] = fabs(gfp[0] + gfp[1] + t);
    if( rp[3] > min_eps )
        rp[3] = sqrt(2.0 / rp[3]);

    box.center.x = (float)rp[0];
    box.center.y = (float)rp[1];
    box.size.width = (float)(rp[2]*2);
    box.size.height = (float)(rp[3]*2);
    if( box.size.width > box.size.height )
    {
        float tmp;
        CV_SWAP( box.size.width, box.size.height, tmp );
        box.angle = (float)(90 + rp[4]*180/CV_PI);
    }
    if( box.angle < -180 )
        box.angle += 360;
    if( box.angle > 360 )
        box.angle -= 360;
#endif

    return box;
}


/* Calculates bounding rectagnle of a point set or retrieves already calculated */
CV_IMPL  CvRect
cvBoundingRect( CvArr* array, int update )
{
    CvSeqReader reader;
    CvRect  rect = { 0, 0, 0, 0 };
    CvContour contour_header;
    CvSeq* ptseq = 0;
    CvSeqBlock block;

    CvMat stub, *mat = 0;
    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i, j, k;
    int calculate = update;

    if( CV_IS_SEQ( array ))
    {
        ptseq = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );

        if( ptseq->header_size < (int)sizeof(CvContour))
        {
            /*if( update == 1 )
                CV_Error( CV_StsBadArg, "The header is too small to fit the rectangle, "
                                        "so it could not be updated" );*/
            update = 0;
            calculate = 1;
        }
    }
    else
    {
        mat = cvGetMat( array, &stub );
        if( CV_MAT_TYPE(mat->type) == CV_32SC2 ||
            CV_MAT_TYPE(mat->type) == CV_32FC2 )
        {
            ptseq = cvPointSeqFromMat(CV_SEQ_KIND_GENERIC, mat, &contour_header, &block);
            mat = 0;
        }
        else if( CV_MAT_TYPE(mat->type) != CV_8UC1 &&
                CV_MAT_TYPE(mat->type) != CV_8SC1 )
            CV_Error( CV_StsUnsupportedFormat,
                "The image/matrix format is not supported by the function" );
        update = 0;
        calculate = 1;
    }

    if( !calculate )
        return ((CvContour*)ptseq)->rect;

    if( mat )
    {
        CvSize size = cvGetMatSize(mat);
        xmin = size.width;
        ymin = -1;

        for( i = 0; i < size.height; i++ )
        {
            uchar* _ptr = mat->data.ptr + i*mat->step;
            uchar* ptr = (uchar*)cvAlignPtr(_ptr, 4);
            int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
            j = 0;
            offset = MIN(offset, size.width);
            for( ; j < offset; j++ )
                if( _ptr[j] )
                {
                    have_nz = 1;
                    break;
                }
            if( j < offset )
            {
                if( j < xmin )
                    xmin = j;
                if( j > xmax )
                    xmax = j;
            }
            if( offset < size.width )
            {
                xmin -= offset;
                xmax -= offset;
                size.width -= offset;
                j = 0;
                for( ; j <= xmin - 4; j += 4 )
                    if( *((int*)(ptr+j)) )
                        break;
                for( ; j < xmin; j++ )
                    if( ptr[j] )
                    {
                        xmin = j;
                        if( j > xmax )
                            xmax = j;
                        have_nz = 1;
                        break;
                    }
                k_min = MAX(j-1, xmax);
                k = size.width - 1;
                for( ; k > k_min && (k&3) != 3; k-- )
                    if( ptr[k] )
                        break;
                if( k > k_min && (k&3) == 3 )
                {
                    for( ; k > k_min+3; k -= 4 )
                        if( *((int*)(ptr+k-3)) )
                            break;
                }
                for( ; k > k_min; k-- )
                    if( ptr[k] )
                    {
                        xmax = k;
                        have_nz = 1;
                        break;
                    }
                if( !have_nz )
                {
                    j &= ~3;
                    for( ; j <= k - 3; j += 4 )
                        if( *((int*)(ptr+j)) )
                            break;
                    for( ; j <= k; j++ )
                        if( ptr[j] )
                        {
                            have_nz = 1;
                            break;
                        }
                }
                xmin += offset;
                xmax += offset;
                size.width += offset;
            }
            if( have_nz )
            {
                if( ymin < 0 )
                    ymin = i;
                ymax = i;
            }
        }

        if( xmin >= size.width )
            xmin = ymin = 0;
    }
    else if( ptseq->total )
    {   
        int  is_float = CV_SEQ_ELTYPE(ptseq) == CV_32FC2;
        cvStartReadSeq( ptseq, &reader, 0 );

        if( !is_float )
        {
            CvPoint pt;
            /* init values */
            CV_READ_SEQ_ELEM( pt, reader );
            xmin = xmax = pt.x;
            ymin = ymax = pt.y;

            for( i = 1; i < ptseq->total; i++ )
            {            
                CV_READ_SEQ_ELEM( pt, reader );
        
                if( xmin > pt.x )
                    xmin = pt.x;
        
                if( xmax < pt.x )
                    xmax = pt.x;

                if( ymin > pt.y )
                    ymin = pt.y;

                if( ymax < pt.y )
                    ymax = pt.y;
            }
        }
        else
        {
            CvPoint pt;
            Cv32suf v;
            /* init values */
            CV_READ_SEQ_ELEM( pt, reader );
            xmin = xmax = CV_TOGGLE_FLT(pt.x);
            ymin = ymax = CV_TOGGLE_FLT(pt.y);

            for( i = 1; i < ptseq->total; i++ )
            {            
                CV_READ_SEQ_ELEM( pt, reader );
                pt.x = CV_TOGGLE_FLT(pt.x);
                pt.y = CV_TOGGLE_FLT(pt.y);
        
                if( xmin > pt.x )
                    xmin = pt.x;
        
                if( xmax < pt.x )
                    xmax = pt.x;

                if( ymin > pt.y )
                    ymin = pt.y;

                if( ymax < pt.y )
                    ymax = pt.y;
            }

            v.i = CV_TOGGLE_FLT(xmin); xmin = cvFloor(v.f);
            v.i = CV_TOGGLE_FLT(ymin); ymin = cvFloor(v.f);
            /* because right and bottom sides of
               the bounding rectangle are not inclusive
               (note +1 in width and height calculation below),
               cvFloor is used here instead of cvCeil */
            v.i = CV_TOGGLE_FLT(xmax); xmax = cvFloor(v.f);
            v.i = CV_TOGGLE_FLT(ymax); ymax = cvFloor(v.f);
        }
    }

    rect.x = xmin;
    rect.y = ymin;
    rect.width = xmax - xmin + 1;
    rect.height = ymax - ymin + 1;

    if( update )
        ((CvContour*)ptseq)->rect = rect;
    
    return rect;
}


/* End of file. */
