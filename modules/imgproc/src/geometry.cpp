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
#include "opencv2/core/hal/intrin.hpp"

using namespace cv;

CV_IMPL CvRect
cvMaxRect( const CvRect* rect1, const CvRect* rect2 )
{
    if( rect1 && rect2 )
    {
        cv::Rect max_rect;
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
        return cvRect(max_rect);
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
        CV_Error( cv::Error::StsNullPtr, "NULL vertex array pointer" );
    cv::RotatedRect(box).points((cv::Point2f*)pt);
}


double cv::pointPolygonTest( InputArray _contour, Point2f pt, bool measureDist )
{
    CV_INSTRUMENT_REGION();

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

            int64 dist = static_cast<int64>(ip.y - v0.y)*(v.x - v0.x)
                       - static_cast<int64>(ip.x - v0.x)*(v.y - v0.y);
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
// Returns true iff point c lies on the closed segment ab.
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

enum LineSegmentIntersection
{
    LS_NO_INTERSECTION = 0,
    LS_SINGLE_INTERSECTION = 1,
    LS_OVERLAP = 2,
    LS_ENDPOINT_INTERSECTION = 3
};

static LineSegmentIntersection parallelInt( Point2f a, Point2f b, Point2f c, Point2f d, Point2f& p, Point2f& q )
{
    LineSegmentIntersection code = LS_OVERLAP;
    if( areaSign(a, b, c) != 0 )
        code = LS_NO_INTERSECTION;
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
        code = LS_NO_INTERSECTION;
    return code;
}

// Finds intersection of two line segments: (a, b) and (c, d).
static LineSegmentIntersection intersectLineSegments( Point2f a, Point2f b, Point2f c,
                                                      Point2f d, Point2f& p, Point2f& q )
{
    double denom = (a.x - b.x) * (double)(d.y - c.y) - (a.y - b.y) * (double)(d.x - c.x);

    // If denom is zero, then segments are parallel: handle separately.
    if( denom == 0. )
        return parallelInt(a, b, c, d, p, q);

    double num = (d.y - a.y) * (double)(a.x - c.x) + (a.x - d.x) * (double)(a.y - c.y);
    double s = num / denom;

    num = (b.y - a.y) * (double)(a.x - c.x) + (c.y - a.y) * (double)(b.x - a.x);
    double t = num / denom;

    p.x = (float)(a.x + s*(b.x - a.x));
    p.y = (float)(a.y + s*(b.y - a.y));
    q = p;

    static const double eps = 1e-5;
    return s < - eps || s > 1.+ eps || t < - eps || t > 1. + eps ? LS_NO_INTERSECTION :
           s < eps || s > 1. - eps || t < eps || t > 1. - eps ? LS_ENDPOINT_INTERSECTION :
           LS_SINGLE_INTERSECTION;
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

// Note: The function and subroutings use direct pointer arithmetics instead of arrays indexing.
// Each loop iteration may push to result array up to 3 times.
// It means that we need +3 spare result elements against result_size
// to catch agorithmic overflow and prevent actual result array overflow.
static int intersectConvexConvex_( const Point2f* P, int n, const Point2f* Q, int m,
                                   Point2f* result, int result_size, float* _area )
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
        LineSegmentIntersection code = intersectLineSegments( P[a1], P[a], Q[b1], Q[b], p, q );
        if( code == LS_SINGLE_INTERSECTION || code == LS_ENDPOINT_INTERSECTION )
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
        if( code == LS_OVERLAP && A.ddot(B) < 0 )
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
    while ( ((aa < n) || (ba < m)) && (aa < 2*n) && (ba < 2*m) && ((int)(result - result0) <= result_size) );

    // Deal with special cases: not implemented.
    if( inflag == Unknown )
    {
        // The boundaries of P and Q do not cross.
        // ...
    }

    int nr = (int)(result - result0);
    if (nr > result_size)
    {
        *_area = -1.f;
        return -1;
    }

    double area = 0;
    Point2f prev = result0[nr-1];
    for(int i = 1; i < nr; i++ )
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
    CV_INSTRUMENT_REGION();

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

    AutoBuffer<Point2f> _result(n + m + n+m+1+3);
    Point2f* fp1 = _result.data();
    Point2f* fp2 = fp1 + n;
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
    int nr = intersectConvexConvex_(fp1, n, fp2, m, result, n+m+1, &area);

    if (nr < 0)
    {
        // The algorithm did not converge, e.g. some of inputs is not convex
        _p12.release();
        return -1.f;
    }

    if( nr == 0 )
    {
        if( !handleNested )
        {
            _p12.release();
            return 0.f;
        }

        bool intersected = false;

        // check if all of fp2's vertices is inside/on the edge of fp1.
        int nVertices = 0;
        for (int i=0; i<m; ++i)
            nVertices += pointPolygonTest(_InputArray(fp1, n), fp2[i], false) >= 0;

        // if all of fp2's vertices is inside/on the edge of fp1.
        if (nVertices == m)
        {
            intersected = true;
            result = fp2;
            nr = m;
        }
        else // otherwise check if fp2 is inside fp1.
        {
            nVertices = 0;
            for (int i=0; i<n; ++i)
                nVertices += pointPolygonTest(_InputArray(fp2, m), fp1[i], false) >= 0;

            // // if all of fp1's vertices is inside/on the edge of fp2.
            if (nVertices == n)
            {
                intersected = true;
                result = fp1;
                nr = n;
            }
        }

        if (!intersected)
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

static Rect maskBoundingRect( const Mat& img )
{
    CV_Assert( img.depth() <= CV_8S && img.channels() == 1 );

    Size size = img.size();
    int xmin = size.width, ymin = -1, xmax = -1, ymax = -1, i, j, k;

    for( i = 0; i < size.height; i++ )
    {
        const uchar* _ptr = img.ptr(i);
        const uchar* ptr = (const uchar*)alignPtr(_ptr, 4);
        int have_nz = 0, k_min, offset = (int)(ptr - _ptr);
        j = 0;
        offset = MIN(offset, size.width);
        for( ; j < offset; j++ )
            if( _ptr[j] )
            {
                if( j < xmin )
                    xmin = j;
                if( j > xmax )
                    xmax = j;
                have_nz = 1;
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
    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

// Calculates bounding rectangle of a point set or retrieves already calculated
static Rect pointSetBoundingRect( const Mat& points )
{
    int npoints = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    bool is_float = depth == CV_32F;

    if( npoints == 0 )
        return Rect();

#if CV_SIMD // TODO: enable for CV_SIMD_SCALABLE, loop tail related.
    if( !is_float )
    {
        const int32_t* pts = points.ptr<int32_t>();
        int64_t firstval = 0;
        std::memcpy(&firstval, pts, sizeof(pts[0]) * 2);
        v_int32 minval, maxval;
        minval = maxval = v_reinterpret_as_s32(vx_setall_s64(firstval)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - VTraits<v_int32>::vlanes()/2; i+= VTraits<v_int32>::vlanes()/2 )
        {
            v_int32 ptXY2 = vx_load(pts + 2 * i);
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - VTraits<v_int32>::vlanes()/4 )
        {
            v_int32 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + 2 * i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += VTraits<v_int64>::vlanes()/2;
        }
        for(int j = 16; j < VTraits<v_uint8>::vlanes(); j*=2)
        {
            minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = v_get0(minval);
        xmax = v_get0(maxval);
        ymin = v_get0(v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
        ymax = v_get0(v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_int32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + 2 * i))));
            for( i++; i < npoints; i++ )
            {
                v_int32x4 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + 2 * i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, v_get0(minval2));
            xmax = max(xmax, v_get0(maxval2));
            ymin = min(ymin, v_get0(v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval2)))));
            ymax = max(ymax, v_get0(v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval2)))));
        }
#endif // CV_SIMD
    }
    else
    {
        const float* pts = points.ptr<float>();
        int64_t firstval = 0;
        std::memcpy(&firstval, pts, sizeof(pts[0]) * 2);
        v_float32 minval, maxval;
        minval = maxval = v_reinterpret_as_f32(vx_setall_s64(firstval)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - VTraits<v_float32>::vlanes()/2; i+= VTraits<v_float32>::vlanes()/2 )
        {
            v_float32 ptXY2 = vx_load(pts + 2 * i);
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - VTraits<v_float32>::vlanes()/4 )
        {
            v_float32 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + 2 * i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += VTraits<v_float32>::vlanes()/4;
        }
        for(int j = 16; j < VTraits<v_uint8>::vlanes(); j*=2)
        {
            minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = cvFloor(v_get0(minval));
        xmax = cvFloor(v_get0(maxval));
        ymin = cvFloor(v_get0(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval)))));
        ymax = cvFloor(v_get0(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval)))));
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_float32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + 2 * i))));
            for( i++; i < npoints; i++ )
            {
                v_float32x4 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + 2 * i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, cvFloor(v_get0(minval2)));
            xmax = max(xmax, cvFloor(v_get0(maxval2)));
            ymin = min(ymin, cvFloor(v_get0(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval2))))));
            ymax = max(ymax, cvFloor(v_get0(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval2))))));
        }
#endif
    }
#else
    const Point* pts = points.ptr<Point>();
    Point pt = pts[0];

    if( !is_float )
    {
        xmin = xmax = pt.x;
        ymin = ymax = pt.y;

        for( i = 1; i < npoints; i++ )
        {
            pt = pts[i];

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
        Cv32suf v;
        // init values
        xmin = xmax = CV_TOGGLE_FLT(pt.x);
        ymin = ymax = CV_TOGGLE_FLT(pt.y);

        for( i = 1; i < npoints; i++ )
        {
            pt = pts[i];
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
        // because right and bottom sides of the bounding rectangle are not inclusive
        // (note +1 in width and height calculation below), cvFloor is used here instead of cvCeil
        v.i = CV_TOGGLE_FLT(xmax); xmax = cvFloor(v.f);
        v.i = CV_TOGGLE_FLT(ymax); ymax = cvFloor(v.f);
    }
#endif

    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}


cv::Rect cv::boundingRect(InputArray array)
{
    CV_INSTRUMENT_REGION();

    Mat m = array.getMat();
    return m.depth() <= CV_8U ? maskBoundingRect(m) : pointSetBoundingRect(m);
}


/* Calculates bounding rectangle of a point set or retrieves already calculated */
CV_IMPL  CvRect
cvBoundingRect( CvArr* array, int update )
{
    cv::Rect rect;
    CvContour contour_header;
    CvSeq* ptseq = 0;
    CvSeqBlock block;

    CvMat stub, *mat = 0;
    int calculate = update;

    if( CV_IS_SEQ( array ))
    {
        ptseq = (CvSeq*)array;
        if( !CV_IS_SEQ_POINT_SET( ptseq ))
            CV_Error( cv::Error::StsBadArg, "Unsupported sequence type" );

        if( ptseq->header_size < (int)sizeof(CvContour))
        {
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
            CV_Error( cv::Error::StsUnsupportedFormat,
                "The image/matrix format is not supported by the function" );
        update = 0;
        calculate = 1;
    }

    if( !calculate )
        return ((CvContour*)ptseq)->rect;

    if( mat )
    {
        rect = cvRect(maskBoundingRect(cv::cvarrToMat(mat)));
    }
    else if( ptseq->total )
    {
        cv::AutoBuffer<double> abuf;
        rect = cvRect(pointSetBoundingRect(cv::cvarrToMat(ptseq, false, false, 0, &abuf)));
    }
    if( update )
        ((CvContour*)ptseq)->rect = cvRect(rect);
    return cvRect(rect);
}
