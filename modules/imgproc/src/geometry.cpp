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


CV_IMPL CvRect
cvMaxRect( const CvRect* rect1, const CvRect* rect2 )
{
    if( rect1 && rect2 )
    {
        CvRect max_rect;
        int a, b;

        max_rect.x = a = rect1->x;
        b = rect2->x;
        if( max_rect.x > b )
            max_rect.x = b;

        max_rect.width = a += rect1->width;
        b += rect2->width;

        if( max_rect.width < b )
            max_rect.width = b;
        max_rect.width -= max_rect.x;

        max_rect.y = a = rect1->y;
        b = rect2->y;
        if( max_rect.y > b )
            max_rect.y = b;

        max_rect.height = a += rect1->height;
        b += rect2->height;

        if( max_rect.height < b )
            max_rect.height = b;
        max_rect.height -= max_rect.y;
        return max_rect;
    }
    else if( rect1 )
        return *rect1;
    else if( rect2 )
        return *rect2;
    else
        return cvRect(0,0,0,0);
}


CV_IMPL void
cvBoxPoints( CvBox2D box, CvPoint2D32f pt[4] )
{
    if( !pt )
        CV_Error( CV_StsNullPtr, "NULL vertex array pointer" );
    cv::RotatedRect(box).points((cv::Point2f*)pt);
}


double cv::pointPolygonTest( InputArray _contour, Point2f pt, bool measureDist )
{
    double result = 0;
    Mat contour = _contour.getMat();
    int i, total = contour.checkVector(2), counter = 0;
    int depth = contour.depth();
    CV_Assert( total >= 0 && (depth == CV_32S || depth == CV_32F));

    bool is_float = depth == CV_32F;
    double min_dist_num = FLT_MAX, min_dist_denom = 1;
    Point ip(cvRound(pt.x), cvRound(pt.y));

    if( total == 0 )
        return measureDist ? -DBL_MAX : -1;

    const Point* cnt = contour.ptr<Point>();
    const Point2f* cntf = (const Point2f*)cnt;

    if( !is_float && !measureDist && ip.x == pt.x && ip.y == pt.y )
    {
        // the fastest "purely integer" branch
        Point v0, v = cnt[total-1];

        for( i = 0; i < total; i++ )
        {
            int dist;
            v0 = v;
            v = cnt[i];

            if( (v0.y <= ip.y && v.y <= ip.y) ||
               (v0.y > ip.y && v.y > ip.y) ||
               (v0.x < ip.x && v.x < ip.x) )
            {
                if( ip.y == v.y && (ip.x == v.x || (ip.y == v0.y &&
                    ((v0.x <= ip.x && ip.x <= v.x) || (v.x <= ip.x && ip.x <= v0.x)))) )
                    return 0;
                continue;
            }

            dist = (ip.y - v0.y)*(v.x - v0.x) - (ip.x - v0.x)*(v.y - v0.y);
            if( dist == 0 )
                return 0;
            if( v.y < v0.y )
                dist = -dist;
            counter += dist > 0;
        }

        result = counter % 2 == 0 ? -1 : 1;
    }
    else
    {
        Point2f v0, v;
        Point iv;

        if( is_float )
        {
            v = cntf[total-1];
        }
        else
        {
            v = cnt[total-1];
        }

        if( !measureDist )
        {
            for( i = 0; i < total; i++ )
            {
                double dist;
                v0 = v;
                if( is_float )
                    v = cntf[i];
                else
                    v = cnt[i];

                if( (v0.y <= pt.y && v.y <= pt.y) ||
                   (v0.y > pt.y && v.y > pt.y) ||
                   (v0.x < pt.x && v.x < pt.x) )
                {
                    if( pt.y == v.y && (pt.x == v.x || (pt.y == v0.y &&
                        ((v0.x <= pt.x && pt.x <= v.x) || (v.x <= pt.x && pt.x <= v0.x)))) )
                        return 0;
                    continue;
                }

                dist = (double)(pt.y - v0.y)*(v.x - v0.x) - (double)(pt.x - v0.x)*(v.y - v0.y);
                if( dist == 0 )
                    return 0;
                if( v.y < v0.y )
                    dist = -dist;
                counter += dist > 0;
            }

            result = counter % 2 == 0 ? -1 : 1;
        }
        else
        {
            for( i = 0; i < total; i++ )
            {
                double dx, dy, dx1, dy1, dx2, dy2, dist_num, dist_denom = 1;

                v0 = v;
                if( is_float )
                    v = cntf[i];
                else
                    v = cnt[i];

                dx = v.x - v0.x; dy = v.y - v0.y;
                dx1 = pt.x - v0.x; dy1 = pt.y - v0.y;
                dx2 = pt.x - v.x; dy2 = pt.y - v.y;

                if( dx1*dx + dy1*dy <= 0 )
                    dist_num = dx1*dx1 + dy1*dy1;
                else if( dx2*dx + dy2*dy >= 0 )
                    dist_num = dx2*dx2 + dy2*dy2;
                else
                {
                    dist_num = (dy1*dx - dx1*dy);
                    dist_num *= dist_num;
                    dist_denom = dx*dx + dy*dy;
                }

                if( dist_num*min_dist_denom < min_dist_num*dist_denom )
                {
                    min_dist_num = dist_num;
                    min_dist_denom = dist_denom;
                    if( min_dist_num == 0 )
                        break;
                }

                if( (v0.y <= pt.y && v.y <= pt.y) ||
                   (v0.y > pt.y && v.y > pt.y) ||
                   (v0.x < pt.x && v.x < pt.x) )
                    continue;

                dist_num = dy1*dx - dx1*dy;
                if( dy < 0 )
                    dist_num = -dist_num;
                counter += dist_num > 0;
            }

            result = std::sqrt(min_dist_num/min_dist_denom);
            if( counter % 2 == 0 )
                result = -result;
        }
    }

    return result;
}


CV_IMPL double
cvPointPolygonTest( const CvArr* _contour, CvPoint2D32f pt, int measure_dist )
{
    cv::AutoBuffer<double> abuf;
    cv::Mat contour = cv::cvarrToMat(_contour, false, false, 0, &abuf);
    return cv::pointPolygonTest(contour, pt, measure_dist != 0);
}

/*
 This code is described in "Computational Geometry in C" (Second Edition),
 Chapter 7.  It is not written to be comprehensible without the
 explanation in that book.

 Written by Joseph O'Rourke.
 Last modified: December 1997
 Questions to orourke@cs.smith.edu.
 --------------------------------------------------------------------
 This code is Copyright 1997 by Joseph O'Rourke.  It may be freely
 redistributed in its entirety provided that this copyright notice is
 not removed.
 --------------------------------------------------------------------
 */

namespace cv
{
typedef enum { Pin, Qin, Unknown } tInFlag;

static int areaSign( Point2f a, Point2f b, Point2f c )
{
    static const double eps = 1e-5;
    double area2 = (b.x - a.x) * (double)(c.y - a.y) - (c.x - a.x ) * (double)(b.y - a.y);
    return area2 > eps ? 1 : area2 < -eps ? -1 : 0;
}

//---------------------------------------------------------------------
// Returns true iff point c lies on the closed segement ab.
// Assumes it is already known that abc are collinear.
//---------------------------------------------------------------------
static bool between( Point2f a, Point2f b, Point2f c )
{
    Point2f ba, ca;

    // If ab not vertical, check betweenness on x; else on y.
    if ( a.x != b.x )
        return ((a.x <= c.x) && (c.x <= b.x)) ||
        ((a.x >= c.x) && (c.x >= b.x));
    else
        return ((a.y <= c.y) && (c.y <= b.y)) ||
        ((a.y >= c.y) && (c.y >= b.y));
}

static char parallelInt( Point2f a, Point2f b, Point2f c, Point2f d, Point2f& p, Point2f& q )
{
    char code = 'e';
    if( areaSign(a, b, c) != 0 )
        code = '0';
    else if( between(a, b, c) && between(a, b, d))
        p = c, q = d;
    else if( between(c, d, a) && between(c, d, b))
        p = a, q = b;
    else if( between(a, b, c) && between(c, d, b))
        p = c, q = b;
    else if( between(a, b, c) && between(c, d, a))
        p = c, q = a;
    else if( between(a, b, d) && between(c, d, b))
        p = d, q = b;
    else if( between(a, b, d) && between(c, d, a))
        p = d, q = a;
    else
        code = '0';
    return code;
}

//---------------------------------------------------------------------
// segSegInt: Finds the point of intersection p between two closed
// segments ab and cd.  Returns p and a char with the following meaning:
// 'e': The segments collinearly overlap, sharing a point.
// 'v': An endpoint (vertex) of one segment is on the other segment,
// but 'e' doesn't hold.
// '1': The segments intersect properly (i.e., they share a point and
// neither 'v' nor 'e' holds).
// '0': The segments do not intersect (i.e., they share no points).
// Note that two collinear segments that share just one point, an endpoint
// of each, returns 'e' rather than 'v' as one might expect.
//---------------------------------------------------------------------
static char segSegInt( Point2f a, Point2f b, Point2f c, Point2f d, Point2f& p, Point2f& q )
{
    double  s, t;       // The two parameters of the parametric eqns.
    double num, denom;  // Numerator and denoninator of equations.
    char code = '?';    // Return char characterizing intersection.

    denom = a.x * (double)( d.y - c.y ) +
    b.x * (double)( c.y - d.y ) +
    d.x * (double)( b.y - a.y ) +
    c.x * (double)( a.y - b.y );

    // If denom is zero, then segments are parallel: handle separately.
    if (denom == 0.0)
        return parallelInt(a, b, c, d, p, q);

    num =    a.x * (double)( d.y - c.y ) +
    c.x * (double)( a.y - d.y ) +
    d.x * (double)( c.y - a.y );
    if ( (num == 0.0) || (num == denom) ) code = 'v';
    s = num / denom;

    num = -( a.x * (double)( c.y - b.y ) +
            b.x * (double)( a.y - c.y ) +
            c.x * (double)( b.y - a.y ) );
    if ( (num == 0.0) || (num == denom) ) code = 'v';
    t = num / denom;

    if      ( (0.0 < s) && (s < 1.0) &&
             (0.0 < t) && (t < 1.0) )
        code = '1';
    else if ( (0.0 > s) || (s > 1.0) ||
             (0.0 > t) || (t > 1.0) )
        code = '0';

    p.x = (float)(a.x + s*(b.x - a.x));
    p.y = (float)(a.y + s*(b.y - a.y));

    return code;
}

static tInFlag inOut( Point2f p, tInFlag inflag, int aHB, int bHA, Point2f*& result )
{
    if( p != result[-1] )
        *result++ = p;
    // Update inflag.
    return aHB > 0 ? Pin : bHA > 0 ? Qin : inflag;
}

//---------------------------------------------------------------------
// Advances and prints out an inside vertex if appropriate.
//---------------------------------------------------------------------
static int advance( int a, int *aa, int n, bool inside, Point2f v, Point2f*& result )
{
    if( inside && v != result[-1] )
        *result++ = v;
    (*aa)++;
    return  (a+1) % n;
}

static void addSharedSeg( Point2f p, Point2f q, Point2f*& result )
{
    if( p != result[-1] )
        *result++ = p;
    if( q != result[-1] )
        *result++ = q;
}


static int intersectConvexConvex_( const Point2f* P, int n, const Point2f* Q, int m,
                                   Point2f* result, float* _area )
{
    Point2f* result0 = result;
    // P has n vertices, Q has m vertices.
    int     a=0, b=0;       // indices on P and Q (resp.)
    Point2f Origin(0,0);
    tInFlag inflag=Unknown; // {Pin, Qin, Unknown}: which inside
    int     aa=0, ba=0;     // # advances on a & b indices (after 1st inter.)
    bool    FirstPoint=true;// Is this the first point? (used to initialize).
    Point2f p0;             // The first point.
    *result++ = Point2f(FLT_MAX, FLT_MAX);

    do
    {
        // Computations of key variables.
        int a1 = (a + n - 1) % n; // a-1, b-1 (resp.)
        int b1 = (b + m - 1) % m;

        Point2f A = P[a] - P[a1], B = Q[b] - Q[b1]; // directed edges on P and Q (resp.)

        int cross = areaSign( Origin, A, B );    // sign of z-component of A x B
        int aHB = areaSign( Q[b1], Q[b], P[a] ); // a in H(b).
        int bHA = areaSign( P[a1], P[a], Q[b] ); // b in H(A);

        // If A & B intersect, update inflag.
        Point2f p, q;
        int code = segSegInt( P[a1], P[a], Q[b1], Q[b], p, q );
        if( code == '1' || code == 'v' )
        {
            if( inflag == Unknown && FirstPoint )
            {
                aa = ba = 0;
                FirstPoint = false;
                p0 = p;
                *result++ = p;
            }
            inflag = inOut( p, inflag, aHB, bHA, result );
        }

        //-----Advance rules-----

        // Special case: A & B overlap and oppositely oriented.
        if( code == 'e' && A.ddot(B) < 0 )
        {
            addSharedSeg( p, q, result );
            return (int)(result - result0);
        }

        // Special case: A & B parallel and separated.
        if( cross == 0 && aHB < 0 && bHA < 0 )
            return (int)(result - result0);

        // Special case: A & B collinear.
        else if ( cross == 0 && aHB == 0 && bHA == 0 ) {
            // Advance but do not output point.
            if ( inflag == Pin )
                b = advance( b, &ba, m, inflag == Qin, Q[b], result );
            else
                a = advance( a, &aa, n, inflag == Pin, P[a], result );
        }

        // Generic cases.
        else if( cross >= 0 )
        {
            if( bHA > 0)
                a = advance( a, &aa, n, inflag == Pin, P[a], result );
            else
                b = advance( b, &ba, m, inflag == Qin, Q[b], result );
        }
        else
        {
            if( aHB > 0)
                b = advance( b, &ba, m, inflag == Qin, Q[b], result );
            else
                a = advance( a, &aa, n, inflag == Pin, P[a], result );
        }
        // Quit when both adv. indices have cycled, or one has cycled twice.
    }
    while ( ((aa < n) || (ba < m)) && (aa < 2*n) && (ba < 2*m) );

    // Deal with special cases: not implemented.
    if( inflag == Unknown )
    {
        // The boundaries of P and Q do not cross.
        // ...
    }

    int i, nr = (int)(result - result0);
    double area = 0;
    Point2f prev = result0[nr-1];
    for( i = 1; i < nr; i++ )
    {
        result0[i-1] = result0[i];
        area += (double)prev.x*result0[i].y - (double)prev.y*result0[i].x;
        prev = result0[i];
    }

    *_area = (float)(area*0.5);

    if( result0[nr-2] == result0[0] && nr > 1 )
        nr--;
    return nr-1;
}

}

float cv::intersectConvexConvex( InputArray _p1, InputArray _p2, OutputArray _p12, bool handleNested )
{
    Mat p1 = _p1.getMat(), p2 = _p2.getMat();
    CV_Assert( p1.depth() == CV_32S || p1.depth() == CV_32F );
    CV_Assert( p2.depth() == CV_32S || p2.depth() == CV_32F );

    int n = p1.checkVector(2, p1.depth(), true);
    int m = p2.checkVector(2, p2.depth(), true);

    CV_Assert( n >= 0 && m >= 0 );

    if( n < 2 || m < 2 )
    {
        _p12.release();
        return 0.f;
    }

    AutoBuffer<Point2f> _result(n*2 + m*2 + 1);
    Point2f *fp1 = _result, *fp2 = fp1 + n;
    Point2f* result = fp2 + m;
    int orientation = 0;

    for( int k = 1; k <= 2; k++ )
    {
        Mat& p = k == 1 ? p1 : p2;
        int len = k == 1 ? n : m;
        Point2f* dst = k == 1 ? fp1 : fp2;

        Mat temp(p.size(), CV_MAKETYPE(CV_32F, p.channels()), dst);
        p.convertTo(temp, CV_32F);
        CV_Assert( temp.ptr<Point2f>() == dst );
        Point2f diff0 = dst[0] - dst[len-1];
        for( int i = 1; i < len; i++ )
        {
            double s = diff0.cross(dst[i] - dst[i-1]);
            if( s != 0 )
            {
                if( s < 0 )
                {
                    orientation++;
                    flip( temp, temp, temp.rows > 1 ? 0 : 1 );
                }
                break;
            }
        }
    }

    float area = 0.f;
    int nr = intersectConvexConvex_(fp1, n, fp2, m, result, &area);
    if( nr == 0 )
    {
        if( !handleNested )
        {
            _p12.release();
            return 0.f;
        }

        if( pointPolygonTest(_InputArray(fp1, n), fp2[0], false) >= 0 )
        {
            result = fp2;
            nr = m;
        }
        else if( pointPolygonTest(_InputArray(fp2, n), fp1[0], false) >= 0 )
        {
            result = fp1;
            nr = n;
        }
        else
        {
            _p12.release();
            return 0.f;
        }
        area = (float)contourArea(_InputArray(result, nr), false);
    }

    if( _p12.needed() )
    {
        Mat temp(nr, 1, CV_32FC2, result);
        // if both input contours were reflected,
        // let's orient the result as the input vectors
        if( orientation == 2 )
            flip(temp, temp, 0);

        temp.copyTo(_p12);
    }
    return (float)fabs(area);
}
