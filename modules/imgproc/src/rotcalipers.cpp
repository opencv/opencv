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

typedef struct
{
    int bottom;
    int left;
    float height;
    float width;
    float base_a;
    float base_b;
}
icvMinAreaState;

#define CV_CALIPERS_MAXHEIGHT      0
#define CV_CALIPERS_MINAREARECT    1
#define CV_CALIPERS_MAXDIST        2

/*F///////////////////////////////////////////////////////////////////////////////////////
//    Name:    icvRotatingCalipers
//    Purpose:
//      Rotating calipers algorithm with some applications
//
//    Context:
//    Parameters:
//      points      - convex hull vertices ( any orientation )
//      n           - number of vertices
//      mode        - concrete application of algorithm
//                    can be  CV_CALIPERS_MAXDIST   or
//                            CV_CALIPERS_MINAREARECT
//      left, bottom, right, top - indexes of extremal points
//      out         - output info.
//                    In case CV_CALIPERS_MAXDIST it points to float value -
//                    maximal height of polygon.
//                    In case CV_CALIPERS_MINAREARECT
//                    ((CvPoint2D32f*)out)[0] - corner
//                    ((CvPoint2D32f*)out)[1] - vector1
//                    ((CvPoint2D32f*)out)[0] - corner2
//
//                      ^
//                      |
//              vector2 |
//                      |
//                      |____________\
//                    corner         /
//                               vector1
//
//    Returns:
//    Notes:
//F*/

/* we will use usual cartesian coordinates */
static void
icvRotatingCalipers( CvPoint2D32f* points, int n, int mode, float* out )
{
    float minarea = FLT_MAX;
    float max_dist = 0;
    char buffer[32] = {};
    int i, k;
    CvPoint2D32f* vect = (CvPoint2D32f*)cvAlloc( n * sizeof(vect[0]) );
    float* inv_vect_length = (float*)cvAlloc( n * sizeof(inv_vect_length[0]) );
    int left = 0, bottom = 0, right = 0, top = 0;
    int seq[4] = { -1, -1, -1, -1 };

    /* rotating calipers sides will always have coordinates
       (a,b) (-b,a) (-a,-b) (b, -a)
     */
    /* this is a first base bector (a,b) initialized by (1,0) */
    float orientation = 0;
    float base_a;
    float base_b = 0;

    float left_x, right_x, top_y, bottom_y;
    CvPoint2D32f pt0 = points[0];

    left_x = right_x = pt0.x;
    top_y = bottom_y = pt0.y;

    for( i = 0; i < n; i++ )
    {
        double dx, dy;

        if( pt0.x < left_x )
            left_x = pt0.x, left = i;

        if( pt0.x > right_x )
            right_x = pt0.x, right = i;

        if( pt0.y > top_y )
            top_y = pt0.y, top = i;

        if( pt0.y < bottom_y )
            bottom_y = pt0.y, bottom = i;

        CvPoint2D32f pt = points[(i+1) & (i+1 < n ? -1 : 0)];

        dx = pt.x - pt0.x;
        dy = pt.y - pt0.y;

        vect[i].x = (float)dx;
        vect[i].y = (float)dy;
        inv_vect_length[i] = (float)(1./sqrt(dx*dx + dy*dy));

        pt0 = pt;
    }

    //cvbInvSqrt( inv_vect_length, inv_vect_length, n );

    /* find convex hull orientation */
    {
        double ax = vect[n-1].x;
        double ay = vect[n-1].y;

        for( i = 0; i < n; i++ )
        {
            double bx = vect[i].x;
            double by = vect[i].y;

            double convexity = ax * by - ay * bx;

            if( convexity != 0 )
            {
                orientation = (convexity > 0) ? 1.f : (-1.f);
                break;
            }
            ax = bx;
            ay = by;
        }
        assert( orientation != 0 );
    }
    base_a = orientation;

/*****************************************************************************************/
/*                         init calipers position                                        */
    seq[0] = bottom;
    seq[1] = right;
    seq[2] = top;
    seq[3] = left;
/*****************************************************************************************/
/*                         Main loop - evaluate angles and rotate calipers               */

    /* all of edges will be checked while rotating calipers by 90 degrees */
    for( k = 0; k < n; k++ )
    {
        /* sinus of minimal angle */
        /*float sinus;*/

        /* compute cosine of angle between calipers side and polygon edge */
        /* dp - dot product */
        float dp[4] = {
            +base_a * vect[seq[0]].x + base_b * vect[seq[0]].y,
            -base_b * vect[seq[1]].x + base_a * vect[seq[1]].y,
            -base_a * vect[seq[2]].x - base_b * vect[seq[2]].y,
            +base_b * vect[seq[3]].x - base_a * vect[seq[3]].y,
        };

        float maxcos = dp[0] * inv_vect_length[seq[0]];

        /* number of calipers edges, that has minimal angle with edge */
        int main_element = 0;

        /* choose minimal angle */
        for ( i = 1; i < 4; ++i )
        {
            float cosalpha = dp[i] * inv_vect_length[seq[i]];
            if (cosalpha > maxcos)
            {
                main_element = i;
                maxcos = cosalpha;
            }
        }

        /*rotate calipers*/
        {
            //get next base
            int pindex = seq[main_element];
            float lead_x = vect[pindex].x*inv_vect_length[pindex];
            float lead_y = vect[pindex].y*inv_vect_length[pindex];
            switch( main_element )
            {
            case 0:
                base_a = lead_x;
                base_b = lead_y;
                break;
            case 1:
                base_a = lead_y;
                base_b = -lead_x;
                break;
            case 2:
                base_a = -lead_x;
                base_b = -lead_y;
                break;
            case 3:
                base_a = -lead_y;
                base_b = lead_x;
                break;
            default: assert(0);
            }
        }
        /* change base point of main edge */
        seq[main_element] += 1;
        seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];


        switch (mode)
        {
        case CV_CALIPERS_MAXHEIGHT:
            {
                /* now main element lies on edge alligned to calipers side */

                /* find opposite element i.e. transform  */
                /* 0->2, 1->3, 2->0, 3->1                */
                int opposite_el = main_element ^ 2;

                float dx = points[seq[opposite_el]].x - points[seq[main_element]].x;
                float dy = points[seq[opposite_el]].y - points[seq[main_element]].y;
                float dist;

                if( main_element & 1 )
                    dist = (float)fabs(dx * base_a + dy * base_b);
                else
                    dist = (float)fabs(dx * (-base_b) + dy * base_a);

                if( dist > max_dist )
                    max_dist = dist;

                break;
            }
        case CV_CALIPERS_MINAREARECT:
            /* find area of rectangle */
            {
                float height;
                float area;

                /* find vector left-right */
                float dx = points[seq[1]].x - points[seq[3]].x;
                float dy = points[seq[1]].y - points[seq[3]].y;

                /* dotproduct */
                float width = dx * base_a + dy * base_b;

                /* find vector left-right */
                dx = points[seq[2]].x - points[seq[0]].x;
                dy = points[seq[2]].y - points[seq[0]].y;

                /* dotproduct */
                height = -dx * base_b + dy * base_a;

                area = width * height;
                if( area <= minarea )
                {
                    float *buf = (float *) buffer;

                    minarea = area;
                    /* leftist point */
                    ((int *) buf)[0] = seq[3];
                    buf[1] = base_a;
                    buf[2] = width;
                    buf[3] = base_b;
                    buf[4] = height;
                    /* bottom point */
                    ((int *) buf)[5] = seq[0];
                    buf[6] = area;
                }
                break;
            }
        }                       /*switch */
    }                           /* for */

    switch (mode)
    {
    case CV_CALIPERS_MINAREARECT:
        {
            float *buf = (float *) buffer;

            float A1 = buf[1];
            float B1 = buf[3];

            float A2 = -buf[3];
            float B2 = buf[1];

            float C1 = A1 * points[((int *) buf)[0]].x + points[((int *) buf)[0]].y * B1;
            float C2 = A2 * points[((int *) buf)[5]].x + points[((int *) buf)[5]].y * B2;

            float idet = 1.f / (A1 * B2 - A2 * B1);

            float px = (C1 * B2 - C2 * B1) * idet;
            float py = (A1 * C2 - A2 * C1) * idet;

            out[0] = px;
            out[1] = py;

            out[2] = A1 * buf[2];
            out[3] = B1 * buf[2];

            out[4] = A2 * buf[4];
            out[5] = B2 * buf[4];
        }
        break;
    case CV_CALIPERS_MAXHEIGHT:
        {
            out[0] = max_dist;
        }
        break;
    }

    cvFree( &vect );
    cvFree( &inv_vect_length );
}


CV_IMPL  CvBox2D
cvMinAreaRect2( const CvArr* array, CvMemStorage* storage )
{
    cv::Ptr<CvMemStorage> temp_storage;
    CvBox2D box;
    cv::AutoBuffer<CvPoint2D32f> _points;
    CvPoint2D32f* points;

    memset(&box, 0, sizeof(box));

    int i, n;
    CvSeqReader reader;
    CvContour contour_header;
    CvSeqBlock block;
    CvSeq* ptseq = (CvSeq*)array;
    CvPoint2D32f out[3];

    if( CV_IS_SEQ(ptseq) )
    {
        if( !CV_IS_SEQ_POINT_SET(ptseq) &&
            (CV_SEQ_KIND(ptseq) != CV_SEQ_KIND_CURVE ||
            CV_SEQ_ELTYPE(ptseq) != CV_SEQ_ELTYPE_PPOINT ))
            CV_Error( CV_StsUnsupportedFormat,
                "Input sequence must consist of 2d points or pointers to 2d points" );
        if( !storage )
            storage = ptseq->storage;
    }
    else
    {
        ptseq = cvPointSeqFromMat( CV_SEQ_KIND_GENERIC, array, &contour_header, &block );
    }

    if( storage )
    {
        temp_storage = cvCreateChildMemStorage( storage );
    }
    else
    {
        temp_storage = cvCreateMemStorage(1 << 10);
    }

    ptseq = cvConvexHull2( ptseq, temp_storage, CV_CLOCKWISE, 1 );
    n = ptseq->total;

    _points.allocate(n);
    points = _points;
    cvStartReadSeq( ptseq, &reader );

    if( CV_SEQ_ELTYPE( ptseq ) == CV_32SC2 )
    {
        for( i = 0; i < n; i++ )
        {
            CvPoint pt;
            CV_READ_SEQ_ELEM( pt, reader );
            points[i].x = (float)pt.x;
            points[i].y = (float)pt.y;
        }
    }
    else
    {
        for( i = 0; i < n; i++ )
        {
            CV_READ_SEQ_ELEM( points[i], reader );
        }
    }

    if( n > 2 )
    {
        icvRotatingCalipers( points, n, CV_CALIPERS_MINAREARECT, (float*)out );
        box.center.x = out[0].x + (out[1].x + out[2].x)*0.5f;
        box.center.y = out[0].y + (out[1].y + out[2].y)*0.5f;
        box.size.width = (float)sqrt((double)out[1].x*out[1].x + (double)out[1].y*out[1].y);
        box.size.height = (float)sqrt((double)out[2].x*out[2].x + (double)out[2].y*out[2].y);
        box.angle = (float)atan2( (double)out[1].y, (double)out[1].x );
    }
    else if( n == 2 )
    {
        box.center.x = (points[0].x + points[1].x)*0.5f;
        box.center.y = (points[0].y + points[1].y)*0.5f;
        double dx = points[1].x - points[0].x;
        double dy = points[1].y - points[0].y;
        box.size.width = (float)sqrt(dx*dx + dy*dy);
        box.size.height = 0;
        box.angle = (float)atan2( dy, dx );
    }
    else
    {
        if( n == 1 )
            box.center = points[0];
    }

    box.angle = (float)(box.angle*180/CV_PI);
    return box;
}
