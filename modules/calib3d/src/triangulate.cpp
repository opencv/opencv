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
#include "opencv2/core/core_c.h"

// cvCorrectMatches function is Copyright (C) 2009, Jostein Austvik Jacobsen.
// cvTriangulatePoints function is derived from icvReconstructPointsFor3View, originally by Valery Mosyagin.

// HZ, R. Hartley and A. Zisserman, Multiple View Geometry in Computer Vision, Cambridge Univ. Press, 2003.



// This method is the same as icvReconstructPointsFor3View, with only a few numbers adjusted for two-view geometry
static void
icvTriangulatePoints(CvMat* projMatr1, CvMat* projMatr2, CvMat* projPoints1, CvMat* projPoints2, CvMat* points4D)
{
    if( projMatr1 == 0 || projMatr2 == 0 ||
      projPoints1 == 0 || projPoints2 == 0 ||
      points4D == 0)
      CV_Error( cv::Error::StsNullPtr, "Some of parameters is a NULL pointer" );

    if( !CV_IS_MAT(projMatr1) || !CV_IS_MAT(projMatr2) ||
      !CV_IS_MAT(projPoints1) || !CV_IS_MAT(projPoints2) ||
      !CV_IS_MAT(points4D) )
      CV_Error( cv::Error::StsUnsupportedFormat, "Input parameters must be matrices" );

    int numPoints = projPoints1->cols;

    if( numPoints < 1 )
        CV_Error( cv::Error::StsOutOfRange, "Number of points must be more than zero" );

    if( projPoints2->cols != numPoints || points4D->cols != numPoints )
        CV_Error( cv::Error::StsUnmatchedSizes, "Number of points must be the same" );

    if( projPoints1->rows != 2 || projPoints2->rows != 2)
        CV_Error( cv::Error::StsUnmatchedSizes, "Number of proj points coordinates must be == 2" );

    if( points4D->rows != 4 )
        CV_Error( cv::Error::StsUnmatchedSizes, "Number of world points coordinates must be == 4" );

    if( projMatr1->cols != 4 || projMatr1->rows != 3 ||
       projMatr2->cols != 4 || projMatr2->rows != 3)
        CV_Error( cv::Error::StsUnmatchedSizes, "Size of projection matrices must be 3x4" );

    // preallocate SVD matrices on stack
    cv::Matx<double, 4, 4> matrA;
    cv::Matx<double, 4, 4> matrU;
    cv::Matx<double, 4, 1> matrW;
    cv::Matx<double, 4, 4> matrV;

    CvMat* projPoints[2] = {projPoints1, projPoints2};
    CvMat* projMatrs[2] = {projMatr1, projMatr2};

    /* Solve system for each point */
    for( int i = 0; i < numPoints; i++ )/* For each point */
    {
        /* Fill matrix for current point */
        for( int j = 0; j < 2; j++ )/* For each view */
        {
            double x,y;
            x = cvmGet(projPoints[j],0,i);
            y = cvmGet(projPoints[j],1,i);
            for( int k = 0; k < 4; k++ )
            {
                matrA(j*2+0, k) = x * cvmGet(projMatrs[j],2,k) - cvmGet(projMatrs[j],0,k);
                matrA(j*2+1, k) = y * cvmGet(projMatrs[j],2,k) - cvmGet(projMatrs[j],1,k);
            }
        }
        /* Solve system for current point */
        cv::SVD::compute(matrA, matrW, matrU, matrV);

        /* Copy computed point */
        cvmSet(points4D,0,i,matrV(3,0));/* X */
        cvmSet(points4D,1,i,matrV(3,1));/* Y */
        cvmSet(points4D,2,i,matrV(3,2));/* Z */
        cvmSet(points4D,3,i,matrV(3,3));/* W */
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
 *		F_			:	3x3 fundamental matrix
 *		points1_	:	1xN matrix containing the first set of points
 *		points2_	:	1xN matrix containing the second set of points
 *		new_points1	:	the optimized points1_. if this is NULL, the corrected points are placed back in points1_
 *		new_points2	:	the optimized points2_. if this is NULL, the corrected points are placed back in points2_
 */
static void
icvCorrectMatches(CvMat *F_, CvMat *points1_, CvMat *points2_, CvMat *new_points1, CvMat *new_points2)
{
    cv::Ptr<CvMat> tmp33;
    cv::Ptr<CvMat> tmp31, tmp31_2;
    cv::Ptr<CvMat> T1i, T2i;
    cv::Ptr<CvMat> R1, R2;
    cv::Ptr<CvMat> TFT, TFTt, RTFTR;
    cv::Ptr<CvMat> U, S, V;
    cv::Ptr<CvMat> e1, e2;
    cv::Ptr<CvMat> polynomial;
    cv::Ptr<CvMat> result;
    cv::Ptr<CvMat> points1, points2;
    cv::Ptr<CvMat> F;

    if (!CV_IS_MAT(F_) || !CV_IS_MAT(points1_) || !CV_IS_MAT(points2_) )
        CV_Error( cv::Error::StsUnsupportedFormat, "Input parameters must be matrices" );
    if (!( F_->cols == 3 && F_->rows == 3))
        CV_Error( cv::Error::StsUnmatchedSizes, "The fundamental matrix must be a 3x3 matrix");
    if (!(((F_->type & CV_MAT_TYPE_MASK) >> 3) == 0 ))
        CV_Error( cv::Error::StsUnsupportedFormat, "The fundamental matrix must be a single-channel matrix" );
    if (!(points1_->rows == 1 && points2_->rows == 1 && points1_->cols == points2_->cols))
        CV_Error( cv::Error::StsUnmatchedSizes, "The point-matrices must have one row, and an equal number of columns" );
    if (((points1_->type & CV_MAT_TYPE_MASK) >> 3) != 1 )
        CV_Error( cv::Error::StsUnmatchedSizes, "The first set of points must contain two channels; one for x and one for y" );
    if (((points2_->type & CV_MAT_TYPE_MASK) >> 3) != 1 )
        CV_Error( cv::Error::StsUnmatchedSizes, "The second set of points must contain two channels; one for x and one for y" );
    if (new_points1 != NULL) {
        CV_Assert(CV_IS_MAT(new_points1));
        if (new_points1->cols != points1_->cols || new_points1->rows != 1)
            CV_Error( cv::Error::StsUnmatchedSizes, "The first output matrix must have the same dimensions as the input matrices" );
        if (CV_MAT_CN(new_points1->type) != 2)
            CV_Error( cv::Error::StsUnsupportedFormat, "The first output matrix must have two channels; one for x and one for y" );
    }
    if (new_points2 != NULL) {
        CV_Assert(CV_IS_MAT(new_points2));
        if (new_points2->cols != points2_->cols || new_points2->rows != 1)
            CV_Error( cv::Error::StsUnmatchedSizes, "The second output matrix must have the same dimensions as the input matrices" );
        if (CV_MAT_CN(new_points2->type) != 2)
            CV_Error( cv::Error::StsUnsupportedFormat, "The second output matrix must have two channels; one for x and one for y" );
    }

    // Make sure F uses double precision
    F.reset(cvCreateMat(3,3,CV_64FC1));
    cvConvert(F_, F);

    // Make sure points1 uses double precision
    points1.reset(cvCreateMat(points1_->rows,points1_->cols,CV_64FC2));
    cvConvert(points1_, points1);

    // Make sure points2 uses double precision
    points2.reset(cvCreateMat(points2_->rows,points2_->cols,CV_64FC2));
    cvConvert(points2_, points2);

    tmp33.reset(cvCreateMat(3,3,CV_64FC1));
    tmp31.reset(cvCreateMat(3,1,CV_64FC1)), tmp31_2.reset(cvCreateMat(3,1,CV_64FC1));
    T1i.reset(cvCreateMat(3,3,CV_64FC1)), T2i.reset(cvCreateMat(3,3,CV_64FC1));
    R1.reset(cvCreateMat(3,3,CV_64FC1)), R2.reset(cvCreateMat(3,3,CV_64FC1));
    TFT.reset(cvCreateMat(3,3,CV_64FC1)), TFTt.reset(cvCreateMat(3,3,CV_64FC1)), RTFTR.reset(cvCreateMat(3,3,CV_64FC1));
    U.reset(cvCreateMat(3,3,CV_64FC1));
    S.reset(cvCreateMat(3,3,CV_64FC1));
    V.reset(cvCreateMat(3,3,CV_64FC1));
    e1.reset(cvCreateMat(3,1,CV_64FC1)), e2.reset(cvCreateMat(3,1,CV_64FC1));

    double x1, y1, x2, y2;
    double scale;
    double f1, f2, a, b, c, d;
    polynomial.reset(cvCreateMat(1,7,CV_64FC1));
    result.reset(cvCreateMat(1,6,CV_64FC2));
    double t_min, s_val, t, s;
    for (int p = 0; p < points1->cols; ++p) {
        // Replace F by T2-t * F * T1-t
        x1 = points1->data.db[p*2];
        y1 = points1->data.db[p*2+1];
        x2 = points2->data.db[p*2];
        y2 = points2->data.db[p*2+1];

        cvSetZero(T1i);
        cvSetReal2D(T1i,0,0,1);
        cvSetReal2D(T1i,1,1,1);
        cvSetReal2D(T1i,2,2,1);
        cvSetReal2D(T1i,0,2,x1);
        cvSetReal2D(T1i,1,2,y1);
        cvSetZero(T2i);
        cvSetReal2D(T2i,0,0,1);
        cvSetReal2D(T2i,1,1,1);
        cvSetReal2D(T2i,2,2,1);
        cvSetReal2D(T2i,0,2,x2);
        cvSetReal2D(T2i,1,2,y2);
        cvGEMM(T2i,F,1,0,0,tmp33,CV_GEMM_A_T);
        cvSetZero(TFT);
        cvGEMM(tmp33,T1i,1,0,0,TFT);

        // Compute the right epipole e1 from F * e1 = 0
        cvSetZero(U);
        cvSetZero(S);
        cvSetZero(V);
        cvSVD(TFT,S,U,V);
        scale = sqrt(cvGetReal2D(V,0,2)*cvGetReal2D(V,0,2) + cvGetReal2D(V,1,2)*cvGetReal2D(V,1,2));
        cvSetReal2D(e1,0,0,cvGetReal2D(V,0,2)/scale);
        cvSetReal2D(e1,1,0,cvGetReal2D(V,1,2)/scale);
        cvSetReal2D(e1,2,0,cvGetReal2D(V,2,2)/scale);
        if (cvGetReal2D(e1,2,0) < 0) {
            cvSetReal2D(e1,0,0,-cvGetReal2D(e1,0,0));
            cvSetReal2D(e1,1,0,-cvGetReal2D(e1,1,0));
            cvSetReal2D(e1,2,0,-cvGetReal2D(e1,2,0));
        }

        // Compute the left epipole e2 from e2' * F = 0  =>  F' * e2 = 0
        cvSetZero(TFTt);
        cvTranspose(TFT, TFTt);
        cvSetZero(U);
        cvSetZero(S);
        cvSetZero(V);
        cvSVD(TFTt,S,U,V);
        cvSetZero(e2);
        scale = sqrt(cvGetReal2D(V,0,2)*cvGetReal2D(V,0,2) + cvGetReal2D(V,1,2)*cvGetReal2D(V,1,2));
        cvSetReal2D(e2,0,0,cvGetReal2D(V,0,2)/scale);
        cvSetReal2D(e2,1,0,cvGetReal2D(V,1,2)/scale);
        cvSetReal2D(e2,2,0,cvGetReal2D(V,2,2)/scale);
        if (cvGetReal2D(e2,2,0) < 0) {
            cvSetReal2D(e2,0,0,-cvGetReal2D(e2,0,0));
            cvSetReal2D(e2,1,0,-cvGetReal2D(e2,1,0));
            cvSetReal2D(e2,2,0,-cvGetReal2D(e2,2,0));
        }

        // Replace F by R2 * F * R1'
        cvSetZero(R1);
        cvSetReal2D(R1,0,0,cvGetReal2D(e1,0,0));
        cvSetReal2D(R1,0,1,cvGetReal2D(e1,1,0));
        cvSetReal2D(R1,1,0,-cvGetReal2D(e1,1,0));
        cvSetReal2D(R1,1,1,cvGetReal2D(e1,0,0));
        cvSetReal2D(R1,2,2,1);
        cvSetZero(R2);
        cvSetReal2D(R2,0,0,cvGetReal2D(e2,0,0));
        cvSetReal2D(R2,0,1,cvGetReal2D(e2,1,0));
        cvSetReal2D(R2,1,0,-cvGetReal2D(e2,1,0));
        cvSetReal2D(R2,1,1,cvGetReal2D(e2,0,0));
        cvSetReal2D(R2,2,2,1);
        cvGEMM(R2,TFT,1,0,0,tmp33);
        cvGEMM(tmp33,R1,1,0,0,RTFTR,CV_GEMM_B_T);

        // Set f1 = e1(3), f2 = e2(3), a = F22, b = F23, c = F32, d = F33
        f1 = cvGetReal2D(e1,2,0);
        f2 = cvGetReal2D(e2,2,0);
        a = cvGetReal2D(RTFTR,1,1);
        b = cvGetReal2D(RTFTR,1,2);
        c = cvGetReal2D(RTFTR,2,1);
        d = cvGetReal2D(RTFTR,2,2);

        // Form the polynomial g(t) = k6*t^6 + k5*t^5 + k4*t^4 + k3*t^3 + k2*t^2 + k1*t + k0
        // from f1, f2, a, b, c and d
        cvSetReal2D(polynomial,0,6,( +b*c*c*f1*f1*f1*f1*a-a*a*d*f1*f1*f1*f1*c ));
        cvSetReal2D(polynomial,0,5,( +f2*f2*f2*f2*c*c*c*c+2*a*a*f2*f2*c*c-a*a*d*d*f1*f1*f1*f1+b*b*c*c*f1*f1*f1*f1+a*a*a*a ));
        cvSetReal2D(polynomial,0,4,( +4*a*a*a*b+2*b*c*c*f1*f1*a+4*f2*f2*f2*f2*c*c*c*d+4*a*b*f2*f2*c*c+4*a*a*f2*f2*c*d-2*a*a*d*f1*f1*c-a*d*d*f1*f1*f1*f1*b+b*b*c*f1*f1*f1*f1*d ));
        cvSetReal2D(polynomial,0,3,( +6*a*a*b*b+6*f2*f2*f2*f2*c*c*d*d+2*b*b*f2*f2*c*c+2*a*a*f2*f2*d*d-2*a*a*d*d*f1*f1+2*b*b*c*c*f1*f1+8*a*b*f2*f2*c*d ));
        cvSetReal2D(polynomial,0,2,( +4*a*b*b*b+4*b*b*f2*f2*c*d+4*f2*f2*f2*f2*c*d*d*d-a*a*d*c+b*c*c*a+4*a*b*f2*f2*d*d-2*a*d*d*f1*f1*b+2*b*b*c*f1*f1*d ));
        cvSetReal2D(polynomial,0,1,( +f2*f2*f2*f2*d*d*d*d+b*b*b*b+2*b*b*f2*f2*d*d-a*a*d*d+b*b*c*c ));
        cvSetReal2D(polynomial,0,0,( -a*d*d*b+b*b*c*d ));

        // Solve g(t) for t to get 6 roots
        cvSetZero(result);
        cvSolvePoly(polynomial, result, 100, 20);

        // Evaluate the cost function s(t) at the real part of the 6 roots
        t_min = DBL_MAX;
        s_val = 1./(f1*f1) + (c*c)/(a*a+f2*f2*c*c);
        for (int ti = 0; ti < 6; ++ti) {
            t = result->data.db[2*ti];
            s = (t*t)/(1 + f1*f1*t*t) + ((c*t + d)*(c*t + d))/((a*t + b)*(a*t + b) + f2*f2*(c*t + d)*(c*t + d));
            if (s < s_val) {
                s_val = s;
                t_min = t;
            }
        }

        // find the optimal x1 and y1 as the points on l1 and l2 closest to the origin
        tmp31->data.db[0] = t_min*t_min*f1;
        tmp31->data.db[1] = t_min;
        tmp31->data.db[2] = t_min*t_min*f1*f1+1;
        tmp31->data.db[0] /= tmp31->data.db[2];
        tmp31->data.db[1] /= tmp31->data.db[2];
        tmp31->data.db[2] /= tmp31->data.db[2];
        cvGEMM(T1i,R1,1,0,0,tmp33,CV_GEMM_B_T);
        cvGEMM(tmp33,tmp31,1,0,0,tmp31_2);
        x1 = tmp31_2->data.db[0];
        y1 = tmp31_2->data.db[1];

        tmp31->data.db[0] = f2*pow(c*t_min+d,2);
        tmp31->data.db[1] = -(a*t_min+b)*(c*t_min+d);
        tmp31->data.db[2] = f2*f2*pow(c*t_min+d,2) + pow(a*t_min+b,2);
        tmp31->data.db[0] /= tmp31->data.db[2];
        tmp31->data.db[1] /= tmp31->data.db[2];
        tmp31->data.db[2] /= tmp31->data.db[2];
        cvGEMM(T2i,R2,1,0,0,tmp33,CV_GEMM_B_T);
        cvGEMM(tmp33,tmp31,1,0,0,tmp31_2);
        x2 = tmp31_2->data.db[0];
        y2 = tmp31_2->data.db[1];

        // Return the points in the matrix format that the user wants
        points1->data.db[p*2] = x1;
        points1->data.db[p*2+1] = y1;
        points2->data.db[p*2] = x2;
        points2->data.db[p*2+1] = y2;
    }

    if( new_points1 )
        cvConvert( points1, new_points1 );
    if( new_points2 )
        cvConvert( points2, new_points2 );
}

void cv::triangulatePoints( InputArray _projMatr1, InputArray _projMatr2,
                            InputArray _projPoints1, InputArray _projPoints2,
                            OutputArray _points4D )
{
    CV_INSTRUMENT_REGION();

    Mat matr1 = _projMatr1.getMat(), matr2 = _projMatr2.getMat();
    Mat points1 = _projPoints1.getMat(), points2 = _projPoints2.getMat();

    if((points1.rows == 1 || points1.cols == 1) && points1.channels() == 2)
        points1 = points1.reshape(1, static_cast<int>(points1.total())).t();

    if((points2.rows == 1 || points2.cols == 1) && points2.channels() == 2)
        points2 = points2.reshape(1, static_cast<int>(points2.total())).t();

    CvMat cvMatr1 = cvMat(matr1), cvMatr2 = cvMat(matr2);
    CvMat cvPoints1 = cvMat(points1), cvPoints2 = cvMat(points2);

    _points4D.create(4, points1.cols, points1.type());
    Mat cvPoints4D_ = _points4D.getMat();
    CvMat cvPoints4D = cvMat(cvPoints4D_);

    icvTriangulatePoints(&cvMatr1, &cvMatr2, &cvPoints1, &cvPoints2, &cvPoints4D);
}

void cv::correctMatches( InputArray _F, InputArray _points1, InputArray _points2,
                         OutputArray _newPoints1, OutputArray _newPoints2 )
{
    CV_INSTRUMENT_REGION();

    Mat F = _F.getMat();
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();

    CvMat cvPoints1 = cvMat(points1), cvPoints2 = cvMat(points2);
    CvMat cvF = cvMat(F);

    _newPoints1.create(points1.size(), points1.type());
    _newPoints2.create(points2.size(), points2.type());
    Mat cvNewPoints1_ = _newPoints1.getMat(), cvNewPoints2_ = _newPoints2.getMat();
    CvMat cvNewPoints1 = cvMat(cvNewPoints1_), cvNewPoints2 = cvMat(cvNewPoints2_);

    icvCorrectMatches(&cvF, &cvPoints1, &cvPoints2, &cvNewPoints1, &cvNewPoints2);
}
