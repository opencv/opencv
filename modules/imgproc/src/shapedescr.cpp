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

namespace cv
{

const float EPS = 1.0e-4f;

static void findCircle3pts(Point2f *pts, Point2f &center, float &radius)
{
    // two edges of the triangle v1, v2
    Point2f v1 = pts[1] - pts[0];
    Point2f v2 = pts[2] - pts[0];

    // center is intersection of midperpendicular lines of the two edges v1, v2
    // a1*x + b1*y = c1 where a1 = v1.x, b1 = v1.y
    // a2*x + b2*y = c2 where a2 = v2.x, b2 = v2.y
    Point2f midPoint1 = (pts[0] + pts[1]) / 2.0f;
    float c1 = midPoint1.x * v1.x + midPoint1.y * v1.y;
    Point2f midPoint2 = (pts[0] + pts[2]) / 2.0f;
    float c2 = midPoint2.x * v2.x + midPoint2.y * v2.y;
    float det = v1.x * v2.y - v1.y * v2.x;
    float cx = (c1 * v2.y - c2 * v1.y) / det;
    float cy = (v1.x * c2 - v2.x * c1) / det;
    center.x = (float)cx;
    center.y = (float)cy;
    cx -= pts[0].x;
    cy -= pts[0].y;
    radius = (float)(std::sqrt(cx *cx + cy * cy)) + EPS;
}

template<typename PT>
static void findThirdPoint(const PT *pts, int i, int j, Point2f &center, float &radius)
{
    center.x = (float)(pts[j].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[j].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[j].x - pts[i].x);
    float dy = (float)(pts[j].y - pts[i].y);
    radius = (float)norm(Point2f(dx, dy)) / 2.0f + EPS;

    for (int k = 0; k < j; ++k)
    {
        dx = center.x - (float)pts[k].x;
        dy = center.y - (float)pts[k].y;
        if (norm(Point2f(dx, dy)) < radius)
        {
            continue;
        }
        else
        {
            Point2f ptsf[3];
            ptsf[0] = (Point2f)pts[i];
            ptsf[1] = (Point2f)pts[j];
            ptsf[2] = (Point2f)pts[k];
            findCircle3pts(ptsf, center, radius);
        }
    }
}


template<typename PT>
void findSecondPoint(const PT *pts, int i, Point2f &center, float &radius)
{
    center.x = (float)(pts[0].x + pts[i].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[i].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[i].x);
    float dy = (float)(pts[0].y - pts[i].y);
    radius = (float)norm(Point2f(dx, dy)) / 2.0f + EPS;

    for (int j = 1; j < i; ++j)
    {
        dx = center.x - (float)pts[j].x;
        dy = center.y - (float)pts[j].y;
        if (norm(Point2f(dx, dy)) < radius)
        {
            continue;
        }
        else
        {
            findThirdPoint(pts, i, j, center, radius);
        }
    }
}


template<typename PT>
static void findMinEnclosingCircle(const PT *pts, int count, Point2f &center, float &radius)
{
    center.x = (float)(pts[0].x + pts[1].x) / 2.0f;
    center.y = (float)(pts[0].y + pts[1].y) / 2.0f;
    float dx = (float)(pts[0].x - pts[1].x);
    float dy = (float)(pts[0].y - pts[1].y);
    radius = (float)norm(Point2f(dx, dy)) / 2.0f + EPS;

    for (int i = 2; i < count; ++i)
    {
        dx = (float)pts[i].x - center.x;
        dy = (float)pts[i].y - center.y;
        float d = (float)norm(Point2f(dx, dy));
        if (d < radius)
        {
            continue;
        }
        else
        {
            findSecondPoint(pts, i, center, radius);
        }
    }
}
} // namespace cv

// see Welzl, Emo. Smallest enclosing disks (balls and ellipsoids). Springer Berlin Heidelberg, 1991.
void cv::minEnclosingCircle( InputArray _points, Point2f& _center, float& _radius )
{
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();
    int count = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(count >= 0 && (depth == CV_32F || depth == CV_32S));

    _center.x = _center.y = 0.f;
    _radius = 0.f;

    if( count == 0 )
        return;

    bool is_float = depth == CV_32F;
    const Point* ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();

    switch (count)
    {
        case 1:
        {
            _center = (is_float) ? ptsf[0] : Point2f((float)ptsi[0].x, (float)ptsi[0].y);
            _radius = EPS;
            break;
        }
        case 2:
        {
            Point2f p1 = (is_float) ? ptsf[0] : Point2f((float)ptsi[0].x, (float)ptsi[0].y);
            Point2f p2 = (is_float) ? ptsf[1] : Point2f((float)ptsi[1].x, (float)ptsi[1].y);
            _center.x = (p1.x + p2.x) / 2.0f;
            _center.y = (p1.y + p2.y) / 2.0f;
            _radius = (float)(norm(p1 - p2) / 2.0) + EPS;
            break;
        }
        default:
        {
            Point2f center;
            float radius = 0.f;
            if (is_float)
            {
                findMinEnclosingCircle<Point2f>(ptsf, count, center, radius);
                #if 0
                    for (size_t m = 0; m < count; ++m)
                    {
                        float d = (float)norm(ptsf[m] - center);
                        if (d > radius)
                        {
                            printf("error!\n");
                        }
                    }
                #endif
            }
            else
            {
                findMinEnclosingCircle<Point>(ptsi, count, center, radius);
                #if 0
                    for (size_t m = 0; m < count; ++m)
                    {
                        double dx = ptsi[m].x - center.x;
                        double dy = ptsi[m].y - center.y;
                        double d = std::sqrt(dx * dx + dy * dy);
                        if (d > radius)
                        {
                            printf("error!\n");
                        }
                    }
                #endif
            }
            _center = center;
            _radius = radius;
            break;
        }
    }
}


// calculates length of a curve (e.g. contour perimeter)
double cv::arcLength( InputArray _curve, bool is_closed )
{
    CV_INSTRUMENT_REGION();

    Mat curve = _curve.getMat();
    int count = curve.checkVector(2);
    int depth = curve.depth();
    CV_Assert( count >= 0 && (depth == CV_32F || depth == CV_32S));
    double perimeter = 0;

    int i;

    if( count <= 1 )
        return 0.;

    bool is_float = depth == CV_32F;
    int last = is_closed ? count-1 : 0;
    const Point* pti = curve.ptr<Point>();
    const Point2f* ptf = curve.ptr<Point2f>();

    Point2f prev = is_float ? ptf[last] : Point2f((float)pti[last].x,(float)pti[last].y);

    for( i = 0; i < count; i++ )
    {
        Point2f p = is_float ? ptf[i] : Point2f((float)pti[i].x,(float)pti[i].y);
        float dx = p.x - prev.x, dy = p.y - prev.y;
        perimeter += std::sqrt(dx*dx + dy*dy);

        prev = p;
    }

    return perimeter;
}

// area of a whole sequence
double cv::contourArea( InputArray _contour, bool oriented )
{
    CV_INSTRUMENT_REGION();

    Mat contour = _contour.getMat();
    int npoints = contour.checkVector(2);
    int depth = contour.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    if( npoints == 0 )
        return 0.;

    double a00 = 0;
    bool is_float = depth == CV_32F;
    const Point* ptsi = contour.ptr<Point>();
    const Point2f* ptsf = contour.ptr<Point2f>();
    Point2f prev = is_float ? ptsf[npoints-1] : Point2f((float)ptsi[npoints-1].x, (float)ptsi[npoints-1].y);

    for( int i = 0; i < npoints; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        a00 += (double)prev.x * p.y - (double)prev.y * p.x;
        prev = p;
    }

    a00 *= 0.5;
    if( !oriented )
        a00 = fabs(a00);

    return a00;
}


cv::RotatedRect cv::fitEllipse( InputArray _points )
{
    CV_INSTRUMENT_REGION();

    Mat points = _points.getMat();
    int i, n = points.checkVector(2);
    int depth = points.depth();
    CV_Assert( n >= 0 && (depth == CV_32F || depth == CV_32S));

    RotatedRect box;

    if( n < 5 )
        CV_Error( CV_StsBadSize, "There should be at least 5 points to fit the ellipse" );

    // New fitellipse algorithm, contributed by Dr. Daniel Weiss
    Point2f c(0,0);
    double gfp[5] = {0}, rp[5] = {0}, t;
    const double min_eps = 1e-8;
    bool is_float = depth == CV_32F;
    const Point* ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();

    AutoBuffer<double> _Ad(n*5), _bd(n);
    double *Ad = _Ad.data(), *bd = _bd.data();

    // first fit for parameters A - E
    Mat A( n, 5, CV_64F, Ad );
    Mat b( n, 1, CV_64F, bd );
    Mat x( 5, 1, CV_64F, gfp );

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        c += p;
    }
    c.x /= n;
    c.y /= n;

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        p -= c;

        bd[i] = 10000.0; // 1.0?
        Ad[i*5] = -(double)p.x * p.x; // A - C signs inverted as proposed by APP
        Ad[i*5 + 1] = -(double)p.y * p.y;
        Ad[i*5 + 2] = -(double)p.x * p.y;
        Ad[i*5 + 3] = p.x;
        Ad[i*5 + 4] = p.y;
    }

    solve(A, b, x, DECOMP_SVD);

    // now use general-form parameters A - E to find the ellipse center:
    // differentiate general form wrt x/y to get two equations for cx and cy
    A = Mat( 2, 2, CV_64F, Ad );
    b = Mat( 2, 1, CV_64F, bd );
    x = Mat( 2, 1, CV_64F, rp );
    Ad[0] = 2 * gfp[0];
    Ad[1] = Ad[2] = gfp[2];
    Ad[3] = 2 * gfp[1];
    bd[0] = gfp[3];
    bd[1] = gfp[4];
    solve( A, b, x, DECOMP_SVD );

    // re-fit for parameters A - C with those center coordinates
    A = Mat( n, 3, CV_64F, Ad );
    b = Mat( n, 1, CV_64F, bd );
    x = Mat( 3, 1, CV_64F, gfp );
    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        p -= c;
        bd[i] = 1.0;
        Ad[i * 3] = (p.x - rp[0]) * (p.x - rp[0]);
        Ad[i * 3 + 1] = (p.y - rp[1]) * (p.y - rp[1]);
        Ad[i * 3 + 2] = (p.x - rp[0]) * (p.y - rp[1]);
    }
    solve(A, b, x, DECOMP_SVD);

    // store angle and radii
    rp[4] = -0.5 * atan2(gfp[2], gfp[1] - gfp[0]); // convert from APP angle usage
    if( fabs(gfp[2]) > min_eps )
        t = gfp[2]/sin(-2.0 * rp[4]);
    else // ellipse is rotated by an integer multiple of pi/2
        t = gfp[1] - gfp[0];
    rp[2] = fabs(gfp[0] + gfp[1] - t);
    if( rp[2] > min_eps )
        rp[2] = std::sqrt(2.0 / rp[2]);
    rp[3] = fabs(gfp[0] + gfp[1] + t);
    if( rp[3] > min_eps )
        rp[3] = std::sqrt(2.0 / rp[3]);

    box.center.x = (float)rp[0] + c.x;
    box.center.y = (float)rp[1] + c.y;
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

    return box;
}

cv::RotatedRect cv::fitEllipseAMS( InputArray _points )
{
    Mat points = _points.getMat();
    int i, n = points.checkVector(2);
    int depth = points.depth();
    CV_Assert( n >= 0 && (depth == CV_32F || depth == CV_32S));

    RotatedRect box;

    if( n < 5 )
        CV_Error( CV_StsBadSize, "There should be at least 5 points to fit the ellipse" );

    Point2f c(0,0);

    bool is_float = depth == CV_32F;
    const Point* ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();

    Mat A( n, 6, CV_64F);
    Matx<double, 6, 6> DM;
    Matx<double, 5, 5> M;
    Matx<double, 5, 1> pVec;
    Matx<double, 6, 1> coeffs;

    double x0, y0, a, b, theta;

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        c += p;
    }
    c.x /= (float)n;
    c.y /= (float)n;

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        p -= c;

        A.at<double>(i,0) = (double)(p.x)*(p.x);
        A.at<double>(i,1) = (double)(p.x)*(p.y);
        A.at<double>(i,2) = (double)(p.y)*(p.y);
        A.at<double>(i,3) = (double)p.x;
        A.at<double>(i,4) = (double)p.y;
        A.at<double>(i,5) = 1.0;
    }
    cv::mulTransposed( A, DM, true, noArray(), 1.0, -1 );
    DM *= (1.0/n);
    double dnm = ( DM(2,5)*(DM(0,5) + DM(2,5)) - (DM(1,5)*DM(1,5)) );
    double ddm =  (4.*(DM(0,5) + DM(2,5))*( (DM(0,5)*DM(2,5)) - (DM(1,5)*DM(1,5))));
    double ddmm = (2.*(DM(0,5) + DM(2,5))*( (DM(0,5)*DM(2,5)) - (DM(1,5)*DM(1,5))));

    M(0,0)=((-DM(0,0) + DM(0,2) + DM(0,5)*DM(0,5))*(DM(1,5)*DM(1,5)) + (-2*DM(0,1)*DM(1,5) + DM(0,5)*(DM(0,0) \
            - (DM(0,5)*DM(0,5)) + (DM(1,5)*DM(1,5))))*DM(2,5) + (DM(0,0) - (DM(0,5)*DM(0,5)))*(DM(2,5)*DM(2,5))) / ddm;
    M(0,1)=((DM(1,5)*DM(1,5))*(-DM(0,1) + DM(1,2) + DM(0,5)*DM(1,5)) + (DM(0,1)*DM(0,5) - ((DM(0,5)*DM(0,5)) + 2*DM(1,1))*DM(1,5) + \
            (DM(1,5)*DM(1,5)*DM(1,5)))*DM(2,5) + (DM(0,1) - DM(0,5)*DM(1,5))*(DM(2,5)*DM(2,5))) / ddm;
    M(0,2)=(-2*DM(1,2)*DM(1,5)*DM(2,5) - DM(0,5)*(DM(2,5)*DM(2,5))*(DM(0,5) + DM(2,5)) + DM(0,2)*dnm + \
            (DM(1,5)*DM(1,5))*(DM(2,2) + DM(2,5)*(DM(0,5) + DM(2,5))))/ddm;
    M(0,3)=(DM(1,5)*(DM(1,5)*DM(2,3) - 2*DM(1,3)*DM(2,5)) + DM(0,3)*dnm) / ddm;
    M(0,4)=(DM(1,5)*(DM(1,5)*DM(2,4) - 2*DM(1,4)*DM(2,5)) + DM(0,4)*dnm) / ddm;
    M(1,0)=(-(DM(0,2)*DM(0,5)*DM(1,5)) + (2*DM(0,1)*DM(0,5) - DM(0,0)*DM(1,5))*DM(2,5))/ddmm;
    M(1,1)=(-(DM(0,1)*DM(1,5)*DM(2,5)) + DM(0,5)*(-(DM(1,2)*DM(1,5)) + 2*DM(1,1)*DM(2,5)))/ddmm;
    M(1,2)=(-(DM(0,2)*DM(1,5)*DM(2,5)) + DM(0,5)*(-(DM(1,5)*DM(2,2)) + 2*DM(1,2)*DM(2,5)))/ddmm;
    M(1,3)=(-(DM(0,3)*DM(1,5)*DM(2,5)) + DM(0,5)*(-(DM(1,5)*DM(2,3)) + 2*DM(1,3)*DM(2,5)))/ddmm;
    M(1,4)=(-(DM(0,4)*DM(1,5)*DM(2,5)) + DM(0,5)*(-(DM(1,5)*DM(2,4)) + 2*DM(1,4)*DM(2,5)))/ddmm;
    M(2,0)=(-2*DM(0,1)*DM(0,5)*DM(1,5) + (DM(0,0) + (DM(0,5)*DM(0,5)))*(DM(1,5)*DM(1,5)) + DM(0,5)*(-(DM(0,5)*DM(0,5)) \
            + (DM(1,5)*DM(1,5)))*DM(2,5) - (DM(0,5)*DM(0,5))*(DM(2,5)*DM(2,5)) + DM(0,2)*(-(DM(1,5)*DM(1,5)) + DM(0,5)*(DM(0,5) + DM(2,5)))) / ddm;
    M(2,1)=((DM(0,5)*DM(0,5))*(DM(1,2) - DM(1,5)*DM(2,5)) + (DM(1,5)*DM(1,5))*(DM(0,1) - DM(1,2) + DM(1,5)*DM(2,5)) \
            + DM(0,5)*(DM(1,2)*DM(2,5) + DM(1,5)*(-2*DM(1,1) + (DM(1,5)*DM(1,5)) - (DM(2,5)*DM(2,5))))) / ddm;
    M(2,2)=((DM(0,5)*DM(0,5))*(DM(2,2) - (DM(2,5)*DM(2,5))) + (DM(1,5)*DM(1,5))*(DM(0,2) - DM(2,2) + (DM(2,5)*DM(2,5))) + \
             DM(0,5)*(-2*DM(1,2)*DM(1,5) + DM(2,5)*((DM(1,5)*DM(1,5)) + DM(2,2) - (DM(2,5)*DM(2,5))))) / ddm;
    M(2,3)=((DM(1,5)*DM(1,5))*(DM(0,3) - DM(2,3)) + (DM(0,5)*DM(0,5))*DM(2,3) + DM(0,5)*(-2*DM(1,3)*DM(1,5) + DM(2,3)*DM(2,5))) / ddm;
    M(2,4)=((DM(1,5)*DM(1,5))*(DM(0,4) - DM(2,4)) + (DM(0,5)*DM(0,5))*DM(2,4) + DM(0,5)*(-2*DM(1,4)*DM(1,5) + DM(2,4)*DM(2,5))) / ddm;
    M(3,0)=DM(0,3);
    M(3,1)=DM(1,3);
    M(3,2)=DM(2,3);
    M(3,3)=DM(3,3);
    M(3,4)=DM(3,4);
    M(4,0)=DM(0,4);
    M(4,1)=DM(1,4);
    M(4,2)=DM(2,4);
    M(4,3)=DM(3,4);
    M(4,4)=DM(4,4);

    if (fabs(cv::determinant(M)) > 1.0e-10) {
            Mat eVal, eVec;
            eigenNonSymmetric(M, eVal, eVec);

            // Select the eigen vector {a,b,c,d,e} which has the lowest eigenvalue
            int minpos = 0;
            double normi, normEVali, normMinpos, normEValMinpos;
            normMinpos = sqrt(eVec.at<double>(minpos,0)*eVec.at<double>(minpos,0) + eVec.at<double>(minpos,1)*eVec.at<double>(minpos,1) + \
                              eVec.at<double>(minpos,2)*eVec.at<double>(minpos,2) + eVec.at<double>(minpos,3)*eVec.at<double>(minpos,3) + \
                              eVec.at<double>(minpos,4)*eVec.at<double>(minpos,4) );
            normEValMinpos = eVal.at<double>(minpos,0) * normMinpos;
            for (i=1; i<5; i++) {
                normi = sqrt(eVec.at<double>(i,0)*eVec.at<double>(i,0) + eVec.at<double>(i,1)*eVec.at<double>(i,1) + \
                             eVec.at<double>(i,2)*eVec.at<double>(i,2) + eVec.at<double>(i,3)*eVec.at<double>(i,3) + \
                             eVec.at<double>(i,4)*eVec.at<double>(i,4) );
                normEVali = eVal.at<double>(i,0) * normi;
                if (normEVali < normEValMinpos) {
                    minpos = i;
                    normMinpos=normi;
                    normEValMinpos=normEVali;
                }
            };

            pVec(0) =eVec.at<double>(minpos,0) / normMinpos;
            pVec(1) =eVec.at<double>(minpos,1) / normMinpos;
            pVec(2) =eVec.at<double>(minpos,2) / normMinpos;
            pVec(3) =eVec.at<double>(minpos,3) / normMinpos;
            pVec(4) =eVec.at<double>(minpos,4) / normMinpos;

            coeffs(0) =pVec(0) ;
            coeffs(1) =pVec(1) ;
            coeffs(2) =pVec(2) ;
            coeffs(3) =pVec(3) ;
            coeffs(4) =pVec(4) ;
            coeffs(5) =-pVec(0) *DM(0,5)-pVec(1) *DM(1,5)-coeffs(2) *DM(2,5);

        // Check that an elliptical solution has been found. AMS sometimes produces Parabolic solutions.
        bool is_ellipse = (coeffs(0)  < 0 && \
                           coeffs(2)  < (coeffs(1) *coeffs(1) )/(4.*coeffs(0) ) && \
                           coeffs(5)  > (-(coeffs(2) *(coeffs(3) *coeffs(3) )) + coeffs(1) *coeffs(3) *coeffs(4)  - coeffs(0) *(coeffs(4) *coeffs(4) )) / \
                                        ((coeffs(1) *coeffs(1) ) - 4*coeffs(0) *coeffs(2) )) || \
                          (coeffs(0)  > 0 && \
                           coeffs(2)  > (coeffs(1) *coeffs(1) )/(4.*coeffs(0) ) && \
                           coeffs(5)  < (-(coeffs(2) *(coeffs(3) *coeffs(3) )) + coeffs(1) *coeffs(3) *coeffs(4)  - coeffs(0) *(coeffs(4) *coeffs(4) )) / \
                                        ( (coeffs(1) *coeffs(1) ) - 4*coeffs(0) *coeffs(2) ));
        if (is_ellipse) {
            double u1 = pVec(2) *pVec(3) *pVec(3)  - pVec(1) *pVec(3) *pVec(4)  + pVec(0) *pVec(4) *pVec(4)  + pVec(1) *pVec(1) *coeffs(5) ;
            double u2 = pVec(0) *pVec(2) *coeffs(5) ;
            double l1 = sqrt(pVec(1) *pVec(1)  + (pVec(0)  - pVec(2) )*(pVec(0)  - pVec(2) ));
            double l2 = pVec(0)  + pVec(2) ;
            double l3 = pVec(1) *pVec(1)  - 4.0*pVec(0) *pVec(2) ;
            double p1 = 2.0*pVec(2) *pVec(3)  - pVec(1) *pVec(4) ;
            double p2 = 2.0*pVec(0) *pVec(4) -(pVec(1) *pVec(3) );

            x0 = p1/l3 + c.x;
            y0 = p2/l3 + c.y;
            a = std::sqrt(2.)*sqrt((u1 - 4.0*u2)/((l1 - l2)*l3));
            b = std::sqrt(2.)*sqrt(-1.0*((u1 - 4.0*u2)/((l1 + l2)*l3)));
            if (pVec(1)  == 0) {
                if (pVec(0)  < pVec(2) ) {
                    theta = 0;
                } else {
                    theta = CV_PI/2.;
                }
            } else {
                theta = CV_PI/2. + 0.5*std::atan2(pVec(1) , (pVec(0)  - pVec(2) ));
            }

            box.center.x = (float)x0; // +c.x;
            box.center.y = (float)y0; // +c.y;
            box.size.width = (float)(2.0*a);
            box.size.height = (float)(2.0*b);
            if( box.size.width > box.size.height )
            {
                float tmp;
                CV_SWAP( box.size.width, box.size.height, tmp );
                box.angle = (float)(90 + theta*180/CV_PI);
            } else {
                box.angle = (float)(fmod(theta*180/CV_PI,180.0));
            };


        } else {
            box = cv::fitEllipseDirect( points );
        }
    } else {
        box = cv::fitEllipse( points );
    }

    return box;
}

cv::RotatedRect cv::fitEllipseDirect( InputArray _points )
{
    Mat points = _points.getMat();
    int i, n = points.checkVector(2);
    int depth = points.depth();
    CV_Assert( n >= 0 && (depth == CV_32F || depth == CV_32S));

    RotatedRect box;

    if( n < 5 )
        CV_Error( CV_StsBadSize, "There should be at least 5 points to fit the ellipse" );

    Point2f c(0,0);

    bool is_float = (depth == CV_32F);
    const Point*   ptsi = points.ptr<Point>();
    const Point2f* ptsf = points.ptr<Point2f>();

    Mat A( n, 6, CV_64F);
    Matx<double, 6, 6> DM;
    Matx33d M, TM, Q;
    Matx<double, 3, 1> pVec;

    double x0, y0, a, b, theta, Ts;

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        c += p;
    }
    c.x /= (float)n;
    c.y /= (float)n;

    for( i = 0; i < n; i++ )
    {
        Point2f p = is_float ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
        p -= c;

        A.at<double>(i,0) = (double)(p.x)*(p.x);
        A.at<double>(i,1) = (double)(p.x)*(p.y);
        A.at<double>(i,2) = (double)(p.y)*(p.y);
        A.at<double>(i,3) = (double)p.x;
        A.at<double>(i,4) = (double)p.y;
        A.at<double>(i,5) = 1.0;
    }
    cv::mulTransposed( A, DM, true, noArray(), 1.0, -1 );
    DM *= (1.0/n);

    TM(0,0) = DM(0,5)*DM(3,5)*DM(4,4) - DM(0,5)*DM(3,4)*DM(4,5) - DM(0,4)*DM(3,5)*DM(5,4) + \
              DM(0,3)*DM(4,5)*DM(5,4) + DM(0,4)*DM(3,4)*DM(5,5) - DM(0,3)*DM(4,4)*DM(5,5);
    TM(0,1) = DM(1,5)*DM(3,5)*DM(4,4) - DM(1,5)*DM(3,4)*DM(4,5) - DM(1,4)*DM(3,5)*DM(5,4) + \
              DM(1,3)*DM(4,5)*DM(5,4) + DM(1,4)*DM(3,4)*DM(5,5) - DM(1,3)*DM(4,4)*DM(5,5);
    TM(0,2) = DM(2,5)*DM(3,5)*DM(4,4) - DM(2,5)*DM(3,4)*DM(4,5) - DM(2,4)*DM(3,5)*DM(5,4) + \
              DM(2,3)*DM(4,5)*DM(5,4) + DM(2,4)*DM(3,4)*DM(5,5) - DM(2,3)*DM(4,4)*DM(5,5);
    TM(1,0) = DM(0,5)*DM(3,3)*DM(4,5) - DM(0,5)*DM(3,5)*DM(4,3) + DM(0,4)*DM(3,5)*DM(5,3) - \
              DM(0,3)*DM(4,5)*DM(5,3) - DM(0,4)*DM(3,3)*DM(5,5) + DM(0,3)*DM(4,3)*DM(5,5);
    TM(1,1) = DM(1,5)*DM(3,3)*DM(4,5) - DM(1,5)*DM(3,5)*DM(4,3) + DM(1,4)*DM(3,5)*DM(5,3) - \
              DM(1,3)*DM(4,5)*DM(5,3) - DM(1,4)*DM(3,3)*DM(5,5) + DM(1,3)*DM(4,3)*DM(5,5);
    TM(1,2) = DM(2,5)*DM(3,3)*DM(4,5) - DM(2,5)*DM(3,5)*DM(4,3) + DM(2,4)*DM(3,5)*DM(5,3) - \
              DM(2,3)*DM(4,5)*DM(5,3) - DM(2,4)*DM(3,3)*DM(5,5) + DM(2,3)*DM(4,3)*DM(5,5);
    TM(2,0) = DM(0,5)*DM(3,4)*DM(4,3) - DM(0,5)*DM(3,3)*DM(4,4) - DM(0,4)*DM(3,4)*DM(5,3) + \
              DM(0,3)*DM(4,4)*DM(5,3) + DM(0,4)*DM(3,3)*DM(5,4) - DM(0,3)*DM(4,3)*DM(5,4);
    TM(2,1) = DM(1,5)*DM(3,4)*DM(4,3) - DM(1,5)*DM(3,3)*DM(4,4) - DM(1,4)*DM(3,4)*DM(5,3) + \
              DM(1,3)*DM(4,4)*DM(5,3) + DM(1,4)*DM(3,3)*DM(5,4) - DM(1,3)*DM(4,3)*DM(5,4);
    TM(2,2) = DM(2,5)*DM(3,4)*DM(4,3) - DM(2,5)*DM(3,3)*DM(4,4) - DM(2,4)*DM(3,4)*DM(5,3) + \
              DM(2,3)*DM(4,4)*DM(5,3) + DM(2,4)*DM(3,3)*DM(5,4) - DM(2,3)*DM(4,3)*DM(5,4);

    Ts=(-(DM(3,5)*DM(4,4)*DM(5,3)) + DM(3,4)*DM(4,5)*DM(5,3) + DM(3,5)*DM(4,3)*DM(5,4) - \
          DM(3,3)*DM(4,5)*DM(5,4)  - DM(3,4)*DM(4,3)*DM(5,5) + DM(3,3)*DM(4,4)*DM(5,5));

    M(0,0) = (DM(2,0) + (DM(2,3)*TM(0,0) + DM(2,4)*TM(1,0) + DM(2,5)*TM(2,0))/Ts)/2.;
    M(0,1) = (DM(2,1) + (DM(2,3)*TM(0,1) + DM(2,4)*TM(1,1) + DM(2,5)*TM(2,1))/Ts)/2.;
    M(0,2) = (DM(2,2) + (DM(2,3)*TM(0,2) + DM(2,4)*TM(1,2) + DM(2,5)*TM(2,2))/Ts)/2.;
    M(1,0) = -DM(1,0) - (DM(1,3)*TM(0,0) + DM(1,4)*TM(1,0) + DM(1,5)*TM(2,0))/Ts;
    M(1,1) = -DM(1,1) - (DM(1,3)*TM(0,1) + DM(1,4)*TM(1,1) + DM(1,5)*TM(2,1))/Ts;
    M(1,2) = -DM(1,2) - (DM(1,3)*TM(0,2) + DM(1,4)*TM(1,2) + DM(1,5)*TM(2,2))/Ts;
    M(2,0) = (DM(0,0) + (DM(0,3)*TM(0,0) + DM(0,4)*TM(1,0) + DM(0,5)*TM(2,0))/Ts)/2.;
    M(2,1) = (DM(0,1) + (DM(0,3)*TM(0,1) + DM(0,4)*TM(1,1) + DM(0,5)*TM(2,1))/Ts)/2.;
    M(2,2) = (DM(0,2) + (DM(0,3)*TM(0,2) + DM(0,4)*TM(1,2) + DM(0,5)*TM(2,2))/Ts)/2.;

    if (fabs(cv::determinant(M)) > 1.0e-10) {
        Mat eVal, eVec;
        eigenNonSymmetric(M, eVal, eVec);

        // Select the eigen vector {a,b,c} which satisfies 4ac-b^2 > 0
        double cond[3];
        cond[0]=(4.0 * eVec.at<double>(0,0) * eVec.at<double>(0,2) - eVec.at<double>(0,1) * eVec.at<double>(0,1));
        cond[1]=(4.0 * eVec.at<double>(1,0) * eVec.at<double>(1,2) - eVec.at<double>(1,1) * eVec.at<double>(1,1));
        cond[2]=(4.0 * eVec.at<double>(2,0) * eVec.at<double>(2,2) - eVec.at<double>(2,1) * eVec.at<double>(2,1));
        if (cond[0]<cond[1]) {
            i = (cond[1]<cond[2]) ? 2 : 1;
        } else {
            i = (cond[0]<cond[2]) ? 2 : 0;
        }
        double norm = std::sqrt(eVec.at<double>(i,0)*eVec.at<double>(i,0) + eVec.at<double>(i,1)*eVec.at<double>(i,1) + eVec.at<double>(i,2)*eVec.at<double>(i,2));
        if (((eVec.at<double>(i,0)<0.0  ? -1 : 1) * (eVec.at<double>(i,1)<0.0  ? -1 : 1) * (eVec.at<double>(i,2)<0.0  ? -1 : 1)) <= 0.0) {
                norm=-1.0*norm;
            }
        pVec(0) =eVec.at<double>(i,0)/norm; pVec(1) =eVec.at<double>(i,1)/norm;pVec(2) =eVec.at<double>(i,2)/norm;

    //  Q = (TM . pVec)/Ts;
        Q(0,0) = (TM(0,0)*pVec(0) +TM(0,1)*pVec(1) +TM(0,2)*pVec(2) )/Ts;
        Q(0,1) = (TM(1,0)*pVec(0) +TM(1,1)*pVec(1) +TM(1,2)*pVec(2) )/Ts;
        Q(0,2) = (TM(2,0)*pVec(0) +TM(2,1)*pVec(1) +TM(2,2)*pVec(2) )/Ts;

    // We compute the ellipse properties in the shifted coordinates as doing so improves the numerical accuracy.

        double u1 = pVec(2)*Q(0,0)*Q(0,0) - pVec(1)*Q(0,0)*Q(0,1) + pVec(0)*Q(0,1)*Q(0,1) + pVec(1)*pVec(1)*Q(0,2);
        double u2 = pVec(0)*pVec(2)*Q(0,2);
        double l1 = sqrt(pVec(1)*pVec(1) + (pVec(0) - pVec(2))*(pVec(0) - pVec(2)));
        double l2 = pVec(0) + pVec(2) ;
        double l3 = pVec(1)*pVec(1) - 4*pVec(0)*pVec(2) ;
        double p1 = 2*pVec(2)*Q(0,0) - pVec(1)*Q(0,1);
        double p2 = 2*pVec(0)*Q(0,1) - pVec(1)*Q(0,0);

        x0 = p1/l3 + c.x;
        y0 = p2/l3 + c.y;
        a = sqrt(2.)*sqrt((u1 - 4.0*u2)/((l1 - l2)*l3));
        b = sqrt(2.)*sqrt(-1.0*((u1 - 4.0*u2)/((l1 + l2)*l3)));
        if (pVec(1)  == 0) {
            if (pVec(0)  < pVec(2) ) {
                theta = 0;
            } else {
                theta = CV_PI/2.;
            }
        } else {
                theta = CV_PI/2. + 0.5*std::atan2(pVec(1) , (pVec(0)  - pVec(2) ));
        }

        box.center.x = (float)x0;
        box.center.y = (float)y0;
        box.size.width = (float)(2.0*a);
        box.size.height = (float)(2.0*b);
        if( box.size.width > box.size.height )
        {
            float tmp;
            CV_SWAP( box.size.width, box.size.height, tmp );
            box.angle = (float)(fmod((90 + theta*180/CV_PI),180.0)) ;
        } else {
            box.angle = (float)(fmod(theta*180/CV_PI,180.0));
        };
    } else {
        box = cv::fitEllipse( points );
    }
    return box;
}


namespace cv
{

// Calculates bounding rectagnle of a point set or retrieves already calculated
static Rect pointSetBoundingRect( const Mat& points )
{
    int npoints = points.checkVector(2);
    int depth = points.depth();
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_32S));

    int  xmin = 0, ymin = 0, xmax = -1, ymax = -1, i;
    bool is_float = depth == CV_32F;

    if( npoints == 0 )
        return Rect();

#if CV_SIMD
    const int64_t* pts = points.ptr<int64_t>();

    if( !is_float )
    {
        v_int32 minval, maxval;
        minval = maxval = v_reinterpret_as_s32(vx_setall_s64(*pts)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - v_int32::nlanes/2; i+= v_int32::nlanes/2 )
        {
            v_int32 ptXY2 = v_reinterpret_as_s32(vx_load(pts + i));
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - v_int32::nlanes/4 )
        {
            v_int32 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += v_int64::nlanes/2;
        }
        for(int j = 16; j < CV_SIMD_WIDTH; j*=2)
        {
            minval = v_min(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = minval.get0();
        xmax = maxval.get0();
        ymin = v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval))).get0();
        ymax = v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval))).get0();
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_int32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
            for( i++; i < npoints; i++ )
            {
                v_int32x4 ptXY = v_reinterpret_as_s32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, minval2.get0());
            xmax = max(xmax, maxval2.get0());
            ymin = min(ymin, v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(minval2))).get0());
            ymax = max(ymax, v_reinterpret_as_s32(v_expand_high(v_reinterpret_as_u32(maxval2))).get0());
        }
#endif
    }
    else
    {
        v_float32 minval, maxval;
        minval = maxval = v_reinterpret_as_f32(vx_setall_s64(*pts)); //min[0]=pt.x, min[1]=pt.y, min[2]=pt.x, min[3]=pt.y
        for( i = 1; i <= npoints - v_float32::nlanes/2; i+= v_float32::nlanes/2 )
        {
            v_float32 ptXY2 = v_reinterpret_as_f32(vx_load(pts + i));
            minval = v_min(ptXY2, minval);
            maxval = v_max(ptXY2, maxval);
        }
        minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
        maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        if( i <= npoints - v_float32::nlanes/4 )
        {
            v_float32 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(vx_load_low(pts + i))));
            minval = v_min(ptXY, minval);
            maxval = v_max(ptXY, maxval);
            i += v_float32::nlanes/4;
        }
        for(int j = 16; j < CV_SIMD_WIDTH; j*=2)
        {
            minval = v_min(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(minval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))));
            maxval = v_max(v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(maxval))), v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))));
        }
        xmin = cvFloor(minval.get0());
        xmax = cvFloor(maxval.get0());
        ymin = cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval))).get0());
        ymax = cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval))).get0());
#if CV_SIMD_WIDTH > 16
        if( i < npoints )
        {
            v_float32x4 minval2, maxval2;
            minval2 = maxval2 = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
            for( i++; i < npoints; i++ )
            {
                v_float32x4 ptXY = v_reinterpret_as_f32(v_expand_low(v_reinterpret_as_u32(v_load_low(pts + i))));
                minval2 = v_min(ptXY, minval2);
                maxval2 = v_max(ptXY, maxval2);
            }
            xmin = min(xmin, cvFloor(minval2.get0()));
            xmax = max(xmax, cvFloor(maxval2.get0()));
            ymin = min(ymin, cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(minval2))).get0()));
            ymax = max(ymax, cvFloor(v_reinterpret_as_f32(v_expand_high(v_reinterpret_as_u32(maxval2))).get0()));
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
    return Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1);
}

}

cv::Rect cv::boundingRect(InputArray array)
{
    CV_INSTRUMENT_REGION();

    Mat m = array.getMat();
    return m.depth() <= CV_8U ? maskBoundingRect(m) : pointSetBoundingRect(m);
}

////////////////////////////////////////////// C API ///////////////////////////////////////////

CV_IMPL int
cvMinEnclosingCircle( const void* array, CvPoint2D32f * _center, float *_radius )
{
    cv::AutoBuffer<double> abuf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &abuf);
    cv::Point2f center;
    float radius;

    cv::minEnclosingCircle(points, center, radius);
    if(_center)
        *_center = cvPoint2D32f(center);
    if(_radius)
        *_radius = radius;
    return 1;
}

static void
icvMemCopy( double **buf1, double **buf2, double **buf3, int *b_max )
{
    CV_Assert( (*buf1 != NULL || *buf2 != NULL) && *buf3 != NULL );

    int bb = *b_max;
    if( *buf2 == NULL )
    {
        *b_max = 2 * (*b_max);
        *buf2 = (double *)cvAlloc( (*b_max) * sizeof( double ));

        memcpy( *buf2, *buf3, bb * sizeof( double ));

        *buf3 = *buf2;
        cvFree( buf1 );
        *buf1 = NULL;
    }
    else
    {
        *b_max = 2 * (*b_max);
        *buf1 = (double *) cvAlloc( (*b_max) * sizeof( double ));

        memcpy( *buf1, *buf3, bb * sizeof( double ));

        *buf3 = *buf1;
        cvFree( buf2 );
        *buf2 = NULL;
    }
}


/* area of a contour sector */
static double icvContourSecArea( CvSeq * contour, CvSlice slice )
{
    cv::Point pt;                 /*  pointer to points   */
    cv::Point pt_s, pt_e;         /*  first and last points  */
    CvSeqReader reader;         /*  points reader of contour   */

    int p_max = 2, p_ind;
    int lpt, flag, i;
    double a00;                 /* unnormalized moments m00    */
    double xi, yi, xi_1, yi_1, x0, y0, dxy, sk, sk1, t;
    double x_s, y_s, nx, ny, dx, dy, du, dv;
    double eps = 1.e-5;
    double *p_are1, *p_are2, *p_are;
    double area = 0;

    CV_Assert( contour != NULL && CV_IS_SEQ_POINT_SET( contour ));

    lpt = cvSliceLength( slice, contour );
    /*if( n2 >= n1 )
        lpt = n2 - n1 + 1;
    else
        lpt = contour->total - n1 + n2 + 1;*/

    if( contour->total <= 0 || lpt <= 2 )
        return 0.;

    a00 = x0 = y0 = xi_1 = yi_1 = 0;
    sk1 = 0;
    flag = 0;
    dxy = 0;
    p_are1 = (double *) cvAlloc( p_max * sizeof( double ));

    p_are = p_are1;
    p_are2 = NULL;

    cvStartReadSeq( contour, &reader, 0 );
    cvSetSeqReaderPos( &reader, slice.start_index );
    { CvPoint pt_s_ = CV_STRUCT_INITIALIZER; CV_READ_SEQ_ELEM(pt_s_, reader); pt_s = pt_s_; }
    p_ind = 0;
    cvSetSeqReaderPos( &reader, slice.end_index );
    { CvPoint pt_e_ = CV_STRUCT_INITIALIZER; CV_READ_SEQ_ELEM(pt_e_, reader); pt_e = pt_e_; }

/*    normal coefficients    */
    nx = pt_s.y - pt_e.y;
    ny = pt_e.x - pt_s.x;
    cvSetSeqReaderPos( &reader, slice.start_index );

    while( lpt-- > 0 )
    {
        { CvPoint pt_ = CV_STRUCT_INITIALIZER; CV_READ_SEQ_ELEM(pt_, reader); pt = pt_; }

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

    // common area calculation
    area = 0;
    for( i = 0; i < p_ind; i++ )
        area += fabs( p_are[i] );

    if( p_are1 != NULL )
        cvFree( &p_are1 );
    else if( p_are2 != NULL )
        cvFree( &p_are2 );

    return area;
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
        cv::AutoBuffer<double> abuf;
        cv::Mat points = cv::cvarrToMat(contour, false, false, 0, &abuf);
        return cv::contourArea( points, oriented !=0 );
    }

    if( CV_SEQ_ELTYPE( contour ) != CV_32SC2 )
        CV_Error( CV_StsUnsupportedFormat,
        "Only curves with integer coordinates are supported in case of contour slice" );
    area = icvContourSecArea( contour, slice );
    return oriented ? area : fabs(area);
}


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

        // scroll the reader by 1 point
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
            // Bugfix by Axel at rubico.com 2010-03-22, affects closed slices only
            // wraparound not handled by CV_NEXT_SEQ_ELEM
            if( is_closed && i == count - 2 )
                cvSetSeqReaderPos( &reader, slice.start_index );

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


CV_IMPL CvBox2D
cvFitEllipse2( const CvArr* array )
{
    cv::AutoBuffer<double> abuf;
    cv::Mat points = cv::cvarrToMat(array, false, false, 0, &abuf);
    return cvBox2D(cv::fitEllipse(points));
}

/* Calculates bounding rectagnle of a point set or retrieves already calculated */
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
            CV_Error( CV_StsBadArg, "Unsupported sequence type" );

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
            CV_Error( CV_StsUnsupportedFormat,
                "The image/matrix format is not supported by the function" );
        update = 0;
        calculate = 1;
    }

    if( !calculate )
        return ((CvContour*)ptseq)->rect;

    if( mat )
    {
        rect = cvRect(cv::maskBoundingRect(cv::cvarrToMat(mat)));
    }
    else if( ptseq->total )
    {
        cv::AutoBuffer<double> abuf;
        rect = cvRect(cv::pointSetBoundingRect(cv::cvarrToMat(ptseq, false, false, 0, &abuf)));
    }
    if( update )
        ((CvContour*)ptseq)->rect = cvRect(rect);
    return cvRect(rect);
}

/* End of file. */
