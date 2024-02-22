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
// Copyright (C) 2009, Intel Corporation and others, all rights reserved.
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
#include <opencv2/core/hal/hal.hpp>

namespace cv {

// correctMatches function is Copyright (C) 2009, Jostein Austvik Jacobsen.
// triangulatePoints function is derived from reconstructPointsFor3View, originally by Valery Mosyagin.

// HZ, R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, Cambridge Univ. Press, 2003.

// This method is the same as reconstructPointsFor3View, with only a few numbers adjusted for two-view geometry
void triangulatePoints( InputArray _P1, InputArray _P2,
                        InputArray _points1, InputArray _points2,
                        OutputArray _points4D )
{
    CV_INSTRUMENT_REGION();

    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    int depth1 = points1.depth(), depth2 = points2.depth();
    const float *p1f = depth1 == CV_32F ? points1.ptr<float>() : 0;
    const float *p2f = depth2 == CV_32F ? points2.ptr<float>() : 0;
    const double *p1d = depth1 == CV_64F ? points1.ptr<double>() : 0;
    const double *p2d = depth2 == CV_64F ? points2.ptr<double>() : 0;
    int pstep1, ystep1, pstep2, ystep2, npoints1, npoints2;

    CV_Assert(depth1 == depth2 && (depth1 == CV_32F || depth1 == CV_64F));

    if ((points1.rows == 1 || points1.cols == 1) && points1.channels() == 2)
    {
        npoints1 = points1.rows + points1.cols - 1;
        ystep1 = 1;
        pstep1 = points1.rows == 1 ? 2 : (int)(points1.step/points1.elemSize());
    }
    else
    {
        npoints1 = points1.cols;
        ystep1 = (int)(points1.step/points1.elemSize());
        pstep1 = 1;
    }

    if ((points2.rows == 1 || points2.cols == 1) && points2.channels() == 2)
    {
        npoints2 = points2.rows + points2.cols - 1;
        ystep2 =  1;
        pstep2 = points2.rows == 1 ? 2 : (int)(points2.step/points2.elemSize());
    }
    else
    {
        npoints2 = points2.cols;
        ystep2 = (int)(points2.step/points2.elemSize());
        pstep2 = 1;
    }

    CV_Assert(npoints1 == npoints2);

    _points4D.create(4, npoints1, depth1);
    Mat points4D = _points4D.getMat();

    Matx<double, 4, 4> matrA;
    Matx<double, 4, 4> matrU;
    Matx<double, 4, 1> matrW;
    Matx<double, 4, 4> matrV;
    size_t step4 = 4*sizeof(double);
    Matx<double, 3, 4> P1;
    Matx<double, 3, 4> P2;
    _P1.getMat().convertTo(P1, CV_64F);
    _P2.getMat().convertTo(P2, CV_64F);

    // Solve system for each point
    for( int i = 0; i < npoints1; i++ )
    {
        // Fill matrix for current point
        double x1 = p1f ? (double)p1f[pstep1*i] : p1d[pstep1*i];
        double y1 = p1f ? (double)p1f[pstep1*i + ystep1] : p1d[pstep1*i + ystep1];
        double x2 = p2f ? (double)p2f[pstep2*i] : p2d[pstep2*i];
        double y2 = p2f ? (double)p2f[pstep2*i + ystep2] : p2d[pstep2*i + ystep2];

        for(int k = 0; k < 4; k++)
        {
            matrA(k, 0) = x1*P1(2, k) - P1(0, k);
            matrA(k, 1) = y1*P1(2, k) - P1(1, k);
            matrA(k, 2) = x2*P2(2, k) - P2(0, k);
            matrA(k, 3) = y2*P2(2, k) - P2(1, k);
        }

        // Solve system for current point
        hal::SVD64f(matrA.val, step4, matrW.val, matrU.val, step4, matrV.val, step4, 4, 4, 4);

        // Copy computed point
        if(depth1 == CV_32F)
        {
            points4D.at<float>(0, i) = (float)matrV(3, 0);
            points4D.at<float>(1, i) = (float)matrV(3, 1);
            points4D.at<float>(2, i) = (float)matrV(3, 2);
            points4D.at<float>(3, i) = (float)matrV(3, 3);
        }
        else
        {
            points4D.at<double>(0, i) = matrV(3, 0);
            points4D.at<double>(1, i) = matrV(3, 1);
            points4D.at<double>(2, i) = matrV(3, 2);
            points4D.at<double>(3, i) = matrV(3, 3);
        }
    }
}

/*
 *	The Optimal Triangulation Method (see HZ for details)
 *		For each given point correspondence points1[i] <-> points2[i], and a fundamental matrix F,
 *		computes the corrected correspondences new_points1[i] <-> new_points2[i] that minimize the
 *		geometric error d(points1[i],new_points1[i])^2 + d(points2[i],new_points2[i])^2 (where d(a,b)
 *		is the geometric distance between points a and b) subject to the epipolar constraint
 *		new_points2' * F * new_points1 = 0.
 *
 *		_F			:	3x3 fundamental matrix
 *		_points1	:	1xN matrix containing the first set of points
 *		_points2	:	1xN matrix containing the second set of points
 *		_newPoints1	:	the optimized _points1.
 *		_newPoints2	:	the optimized -points2.
 */
void correctMatches( InputArray _F, InputArray _points1, InputArray _points2,
                     OutputArray _newPoints1, OutputArray _newPoints2 )
{
    CV_INSTRUMENT_REGION();

    Mat points1 = _points1.getMat(), points2 = _points2.getMat();

    int depth1 = points1.depth(), depth2 = points2.depth();
    CV_Assert((depth1 == CV_32F || depth1 == CV_64F) && depth1 == depth2);

    CV_Assert(points1.size() == points2.size());
    CV_Assert(points1.rows == 1 || points1.cols == 1);
    if (points1.channels() != 2)
        CV_Error( cv::Error::StsUnmatchedSizes, "The first set of points must contain two channels; one for x and one for y" );
    if (points2.channels() != 2)
        CV_Error( cv::Error::StsUnmatchedSizes, "The second set of points must contain two channels; one for x and one for y" );

    _newPoints1.create(points1.size(), points1.type());
    _newPoints2.create(points2.size(), points2.type());
    Mat newPoints1 = _newPoints1.getMat(), newPoints2 = _newPoints2.getMat();

    Matx33d F, U, Vt;
    Matx31d S;
    int npoints = points1.rows + points1.cols - 1;

    // Make sure F uses double precision
    _F.getMat().convertTo(F, CV_64F);

    for (int p = 0; p < npoints; ++p) {
        // Replace F by T2-t * F * T1-t
        double x1, y1, x2, y2;
        if (depth1 == CV_32F) {
            Point2f p1 = points1.at<Point2f>(p);
            Point2f p2 = points2.at<Point2f>(p);
            x1 = p1.x; y1 = p1.y;
            x2 = p2.x; y2 = p2.y;
        } else {
            Point2d p1 = points1.at<Point2d>(p);
            Point2d p2 = points2.at<Point2d>(p);
            x1 = p1.x; y1 = p1.y;
            x2 = p2.x; y2 = p2.y;
        }

        Matx33d T1i(1, 0, x1,
                    0, 1, y1,
                    0, 0, 1);
        Matx33d T2i(1, 0, x2,
                    0, 1, y2,
                    0, 0, 1);
        Matx33d TFT = T2i.t()*F*T1i;

        // Compute the right epipole e1 from F * e1 = 0
        SVDecomp(TFT, S, U, Vt);
        double scale = sqrt(Vt(2, 0)*Vt(2, 0) + Vt(2, 1)*Vt(2, 1));

        Vec3d e1(Vt(2, 0)/scale, Vt(2, 1)/scale, Vt(2, 2)/scale);
        if (e1(2) < 0)
            e1 = -e1;

        // Compute the left epipole e2 from e2' * F = 0  =>  F' * e2 = 0
        scale = sqrt(U(0, 2)*U(0, 2) + U(1, 2)*U(1, 2));

        Vec3d e2(U(0, 2)/scale, U(1, 2)/scale, U(2, 2)/scale);
        if (e2(2) < 0)
            e2 = -e2;

        // Replace F by R2 * F * R1'
        Matx33d R1_t(e1(0), -e1(1), 0,
                     e1(1), e1(0), 0,
                     0, 0, 1);
        Matx33d R2(e2(0), e2(1), 0,
                   -e2(1), e2(0), 0,
                   0, 0, 1);
        Matx33d RTFTR = R2*TFT*R1_t;

        // Set f1 = e1(3), f2 = e2(3), a = F22, b = F23, c = F32, d = F33
        double f1 = e1(2);
        double f2 = e2(2);
        double a = RTFTR(1,1);
        double b = RTFTR(1,2);
        double c = RTFTR(2,1);
        double d = RTFTR(2,2);

        // Form the polynomial g(t) = k6*t^6 + k5*t^5 + k4*t^4 + k3*t^3 + k2*t^2 + k1*t + k0
        // from f1, f2, a, b, c and d
        Vec<double, 7> polynomial(
            -a*d*d*b+b*b*c*d,
            +f2*f2*f2*f2*d*d*d*d+b*b*b*b+2*b*b*f2*f2*d*d-a*a*d*d+b*b*c*c,
            +4*a*b*b*b+4*b*b*f2*f2*c*d+4*f2*f2*f2*f2*c*d*d*d-a*a*d*c+b*c*c*a+4*a*b*f2*f2*d*d-2*a*d*d*f1*f1*b+2*b*b*c*f1*f1*d,
            +6*a*a*b*b+6*f2*f2*f2*f2*c*c*d*d+2*b*b*f2*f2*c*c+2*a*a*f2*f2*d*d-2*a*a*d*d*f1*f1+2*b*b*c*c*f1*f1+8*a*b*f2*f2*c*d,
            +4*a*a*a*b+2*b*c*c*f1*f1*a+4*f2*f2*f2*f2*c*c*c*d+4*a*b*f2*f2*c*c+4*a*a*f2*f2*c*d-2*a*a*d*f1*f1*c-a*d*d*f1*f1*f1*f1*b+b*b*c*f1*f1*f1*f1*d,
            +f2*f2*f2*f2*c*c*c*c+2*a*a*f2*f2*c*c-a*a*d*d*f1*f1*f1*f1+b*b*c*c*f1*f1*f1*f1+a*a*a*a,
            +b*c*c*f1*f1*f1*f1*a-a*a*d*f1*f1*f1*f1*c);

        // Solve g(t) for t to get 6 roots
        double rdata[6*2];
        Mat result(6, 1, CV_64FC2, rdata);
        solvePoly(polynomial, result);

        // Evaluate the cost function s(t) at the real part of the 6 roots
        double t_min = DBL_MAX;
        double s_val = 1./(f1*f1) + (c*c)/(a*a+f2*f2*c*c);
        for (int ti = 0; ti < 6; ++ti) {
            Vec2d root_i = result.at<Vec2d>(ti);
            double t = root_i(0);
            double s = (t*t)/(1 + f1*f1*t*t) + ((c*t + d)*(c*t + d))/((a*t + b)*(a*t + b) + f2*f2*(c*t + d)*(c*t + d));
            if (s < s_val) {
                s_val = s;
                t_min = t;
            }
        }

        // find the optimal x1 and y1 as the points on l1 and l2 closest to the origin
        scale = t_min*t_min*f1*f1+1;
        Vec3d tmp31(t_min*t_min*f1/scale, t_min/scale, 1);
        Vec3d tmp31_2 = T1i*(R1_t*tmp31);
        x1 = tmp31_2(0);
        y1 = tmp31_2(1);

        scale = f2*f2*(c*t_min+d)*(c*t_min+d) + (a*t_min+b)*(a*t_min+b);
        tmp31 = Vec3d(f2*(c*t_min+d)*(c*t_min+d)/scale, -(a*t_min+b)*(c*t_min+d)/scale, 1);
        tmp31_2 = T2i*(R2.t()*tmp31);
        x2 = tmp31_2(0);
        y2 = tmp31_2(1);

        // Return the points in the matrix format that the user wants
        if (depth1 == CV_32F) {
            newPoints1.at<Point2f>(p) = Point2f((float)x1, (float)y1);
            newPoints2.at<Point2f>(p) = Point2f((float)x2, (float)y2);
        } else {
            newPoints1.at<Point2d>(p) = Point2d(x1, y1);
            newPoints2.at<Point2d>(p) = Point2d(x2, y2);
        }
    }
}

}
