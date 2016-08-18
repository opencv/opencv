/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
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
//   * The name of OpenCV Foundation may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the OpenCV Foundation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
#include "precomp.hpp"

namespace cv
{

struct MinAreaState
{
    int bottom;
    int left;
    float height;
    float width;
    float base_a;
    float base_b;
};

enum { CALIPERS_MAXHEIGHT=0, CALIPERS_MINAREARECT=1, CALIPERS_MAXDIST=2 };

/*F///////////////////////////////////////////////////////////////////////////////////////
 //    Name:    rotatingCalipers
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
static void rotatingCalipers( const Point2f* points, int n, int mode, float* out )
{
    float minarea = FLT_MAX;
    float max_dist = 0;
    char buffer[32] = {};
    int i, k;
    AutoBuffer<float> abuf(n*3);
    float* inv_vect_length = abuf;
    Point2f* vect = (Point2f*)(inv_vect_length + n);
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
    Point2f pt0 = points[0];

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

        Point2f pt = points[(i+1) & (i+1 < n ? -1 : 0)];

        dx = pt.x - pt0.x;
        dy = pt.y - pt0.y;

        vect[i].x = (float)dx;
        vect[i].y = (float)dy;
        inv_vect_length[i] = (float)(1./std::sqrt(dx*dx + dy*dy));

        pt0 = pt;
    }

    // find convex hull orientation
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
        CV_Assert( orientation != 0 );
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
            default:
                CV_Error(CV_StsError, "main_element should be 0, 1, 2 or 3");
            }
        }
        /* change base point of main edge */
        seq[main_element] += 1;
        seq[main_element] = (seq[main_element] == n) ? 0 : seq[main_element];

        switch (mode)
        {
        case CALIPERS_MAXHEIGHT:
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
            }
            break;
        case CALIPERS_MINAREARECT:
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
            }
            break;
        }                       /*switch */
    }                           /* for */

    switch (mode)
    {
    case CALIPERS_MINAREARECT:
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
    case CALIPERS_MAXHEIGHT:
        {
        out[0] = max_dist;
        }
        break;
    }
}

}


cv::RotatedRect cv::minAreaRect( InputArray _points )
{
    CV_INSTRUMENT_REGION()

    Mat hull;
    Point2f out[3];
    RotatedRect box;

    convexHull(_points, hull, true, true);

    if( hull.depth() != CV_32F )
    {
        Mat temp;
        hull.convertTo(temp, CV_32F);
        hull = temp;
    }

    int n = hull.checkVector(2);
    const Point2f* hpoints = hull.ptr<Point2f>();

    if( n > 2 )
    {
        rotatingCalipers( hpoints, n, CALIPERS_MINAREARECT, (float*)out );
        box.center.x = out[0].x + (out[1].x + out[2].x)*0.5f;
        box.center.y = out[0].y + (out[1].y + out[2].y)*0.5f;
        box.size.width = (float)std::sqrt((double)out[1].x*out[1].x + (double)out[1].y*out[1].y);
        box.size.height = (float)std::sqrt((double)out[2].x*out[2].x + (double)out[2].y*out[2].y);
        box.angle = (float)atan2( (double)out[1].y, (double)out[1].x );
    }
    else if( n == 2 )
    {
        box.center.x = (hpoints[0].x + hpoints[1].x)*0.5f;
        box.center.y = (hpoints[0].y + hpoints[1].y)*0.5f;
        double dx = hpoints[1].x - hpoints[0].x;
        double dy = hpoints[1].y - hpoints[0].y;
        box.size.width = (float)std::sqrt(dx*dx + dy*dy);
        box.size.height = 0;
        box.angle = (float)atan2( dy, dx );
    }
    else
    {
        if( n == 1 )
            box.center = hpoints[0];
    }

    box.angle = (float)(box.angle*180/CV_PI);
    return box;
}


CV_IMPL CvBox2D
cvMinAreaRect2( const CvArr* array, CvMemStorage* /*storage*/ )
{
    cv::AutoBuffer<double> abuf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &abuf);

    cv::RotatedRect rr = cv::minAreaRect(points);
    return (CvBox2D)rr;
}

void cv::boxPoints(cv::RotatedRect box, OutputArray _pts)
{
    CV_INSTRUMENT_REGION()

    _pts.create(4, 2, CV_32F);
    Mat pts = _pts.getMat();
    box.points(pts.ptr<Point2f>());
}
