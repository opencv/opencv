/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                          License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000, Intel Corporation, all rights reserved.
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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

static bool haveCollinearPoints( const Mat& m, int count )
{
    int j, k, i = count-1;
    const Point2f* ptr = m.ptr<Point2f>();

    // check that the i-th selected point does not belong
    // to a line connecting some previously selected points
    for( j = 0; j < i; j++ )
    {
        double dx1 = ptr[j].x - ptr[i].x;
        double dy1 = ptr[j].y - ptr[i].y;
        for( k = 0; k < j; k++ )
        {
            double dx2 = ptr[k].x - ptr[i].x;
            double dy2 = ptr[k].y - ptr[i].y;
            if( fabs(dx2*dy1 - dy2*dx1) <= FLT_EPSILON*(fabs(dx1) + fabs(dy1) + fabs(dx2) + fabs(dy2)))
                return true;
        }
    }
    return false;
}


template<typename T> int compressPoints( T* ptr, const uchar* mask, int mstep, int count )
{
    int i, j;
    for( i = j = 0; i < count; i++ )
        if( mask[i*mstep] )
        {
            if( i > j )
                ptr[j] = ptr[i];
            j++;
        }
    return j;
}


class HomographyEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        if( haveCollinearPoints(ms1, count) || haveCollinearPoints(ms2, count) )
            return false;

        // We check whether the minimal set of points for the homography estimation
        // are geometrically consistent. We check if every 3 correspondences sets
        // fulfills the constraint.
        //
        // The usefullness of this constraint is explained in the paper:
        //
        // "Speeding-up homography estimation in mobile devices"
        // Journal of Real-Time Image Processing. 2013. DOI: 10.1007/s11554-012-0314-1
        // Pablo Marquez-Neila, Javier Lopez-Alberca, Jose M. Buenaposada, Luis Baumela
        if( count == 4 )
        {
            static const int tt[][3] = {{0, 1, 2}, {1, 2, 3}, {0, 2, 3}, {0, 1, 3}};
            const Point2f* src = ms1.ptr<Point2f>();
            const Point2f* dst = ms2.ptr<Point2f>();
            int negative = 0;

            for( int i = 0; i < 4; i++ )
            {
                const int* t = tt[i];
                Matx33d A(src[t[0]].x, src[t[0]].y, 1., src[t[1]].x, src[t[1]].y, 1., src[t[2]].x, src[t[2]].y, 1.);
                Matx33d B(dst[t[0]].x, dst[t[0]].y, 1., dst[t[1]].x, dst[t[1]].y, 1., dst[t[2]].x, dst[t[2]].y, 1.);

                negative += determinant(A)*determinant(B) < 0;
            }
            if( negative != 0 && negative != 4 )
                return false;
        }

        return true;
    }

    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int i, count = m1.checkVector(2);
        const Point2f* M = m1.ptr<Point2f>();
        const Point2f* m = m2.ptr<Point2f>();

        double LtL[9][9], W[9][1], V[9][9];
        Mat _LtL( 9, 9, CV_64F, &LtL[0][0] );
        Mat matW( 9, 1, CV_64F, W );
        Mat matV( 9, 9, CV_64F, V );
        Mat _H0( 3, 3, CV_64F, V[8] );
        Mat _Htemp( 3, 3, CV_64F, V[7] );
        Point2d cM(0,0), cm(0,0), sM(0,0), sm(0,0);

        for( i = 0; i < count; i++ )
        {
            cm.x += m[i].x; cm.y += m[i].y;
            cM.x += M[i].x; cM.y += M[i].y;
        }

        cm.x /= count;
        cm.y /= count;
        cM.x /= count;
        cM.y /= count;

        for( i = 0; i < count; i++ )
        {
            sm.x += fabs(m[i].x - cm.x);
            sm.y += fabs(m[i].y - cm.y);
            sM.x += fabs(M[i].x - cM.x);
            sM.y += fabs(M[i].y - cM.y);
        }

        if( fabs(sm.x) < DBL_EPSILON || fabs(sm.y) < DBL_EPSILON ||
            fabs(sM.x) < DBL_EPSILON || fabs(sM.y) < DBL_EPSILON )
            return 0;
        sm.x = count/sm.x; sm.y = count/sm.y;
        sM.x = count/sM.x; sM.y = count/sM.y;

        double invHnorm[9] = { 1./sm.x, 0, cm.x, 0, 1./sm.y, cm.y, 0, 0, 1 };
        double Hnorm2[9] = { sM.x, 0, -cM.x*sM.x, 0, sM.y, -cM.y*sM.y, 0, 0, 1 };
        Mat _invHnorm( 3, 3, CV_64FC1, invHnorm );
        Mat _Hnorm2( 3, 3, CV_64FC1, Hnorm2 );

        _LtL.setTo(Scalar::all(0));
        for( i = 0; i < count; i++ )
        {
            double x = (m[i].x - cm.x)*sm.x, y = (m[i].y - cm.y)*sm.y;
            double X = (M[i].x - cM.x)*sM.x, Y = (M[i].y - cM.y)*sM.y;
            double Lx[] = { X, Y, 1, 0, 0, 0, -x*X, -x*Y, -x };
            double Ly[] = { 0, 0, 0, X, Y, 1, -y*X, -y*Y, -y };
            int j, k;
            for( j = 0; j < 9; j++ )
                for( k = j; k < 9; k++ )
                    LtL[j][k] += Lx[j]*Lx[k] + Ly[j]*Ly[k];
        }
        completeSymm( _LtL );

        eigen( _LtL, matW, matV );
        _Htemp = _invHnorm*_H0;
        _H0 = _Htemp*_Hnorm2;
        _H0.convertTo(_model, _H0.type(), 1./_H0.at<double>(2,2) );

        return 1;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat m1 = _m1.getMat(), m2 = _m2.getMat(), model = _model.getMat();
        int i, count = m1.checkVector(2);
        const Point2f* M = m1.ptr<Point2f>();
        const Point2f* m = m2.ptr<Point2f>();
        const double* H = model.ptr<double>();
        float Hf[] = { (float)H[0], (float)H[1], (float)H[2], (float)H[3], (float)H[4], (float)H[5], (float)H[6], (float)H[7] };

        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();

        for( i = 0; i < count; i++ )
        {
            float ww = 1.f/(Hf[6]*M[i].x + Hf[7]*M[i].y + 1.f);
            float dx = (Hf[0]*M[i].x + Hf[1]*M[i].y + Hf[2])*ww - m[i].x;
            float dy = (Hf[3]*M[i].x + Hf[4]*M[i].y + Hf[5])*ww - m[i].y;
            err[i] = (float)(dx*dx + dy*dy);
        }
    }
};


class HomographyRefineCallback : public LMSolver::Callback
{
public:
    HomographyRefineCallback(InputArray _src, InputArray _dst)
    {
        src = _src.getMat();
        dst = _dst.getMat();
    }

    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const
    {
        int i, count = src.checkVector(2);
        Mat param = _param.getMat();
        _err.create(count*2, 1, CV_64F);
        Mat err = _err.getMat(), J;
        if( _Jac.needed())
        {
            _Jac.create(count*2, param.rows, CV_64F);
            J = _Jac.getMat();
            CV_Assert( J.isContinuous() && J.cols == 8 );
        }

        const Point2f* M = src.ptr<Point2f>();
        const Point2f* m = dst.ptr<Point2f>();
        const double* h = param.ptr<double>();
        double* errptr = err.ptr<double>();
        double* Jptr = J.data ? J.ptr<double>() : 0;

        for( i = 0; i < count; i++ )
        {
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            errptr[i*2] = xi - m[i].x;
            errptr[i*2+1] = yi - m[i].y;

            if( Jptr )
            {
                Jptr[0] = Mx*ww; Jptr[1] = My*ww; Jptr[2] = ww;
                Jptr[3] = Jptr[4] = Jptr[5] = 0.;
                Jptr[6] = -Mx*ww*xi; Jptr[7] = -My*ww*xi;
                Jptr[8] = Jptr[9] = Jptr[10] = 0.;
                Jptr[11] = Mx*ww; Jptr[12] = My*ww; Jptr[13] = ww;
                Jptr[14] = -Mx*ww*yi; Jptr[15] = -My*ww*yi;

                Jptr += 16;
            }
        }

        return true;
    }

    Mat src, dst;
};

}


cv::Mat cv::findHomography( InputArray _points1, InputArray _points2,
                            int method, double ransacReprojThreshold, OutputArray _mask )
{
    const double confidence = 0.995;
    const int maxIters = 2000;
    const double defaultRANSACReprojThreshold = 3;
    bool result = false;

    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat src, dst, H, tempMask;
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        Mat& p = i == 1 ? points1 : points2;
        Mat& m = i == 1 ? src : dst;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            if( npoints < 0 )
                CV_Error(Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if( npoints == 0 )
                return Mat();
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }

    CV_Assert( src.checkVector(2) == dst.checkVector(2) );

    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    Ptr<PointSetRegistrator::Callback> cb = new HomographyEstimatorCallback;

    if( method == 0 || npoints == 4 )
    {
        tempMask = Mat::ones(npoints, 1, CV_8U);
        result = cb->runKernel(src, dst, H) > 0;
    }
    else if( method == RANSAC )
        result = createRANSACPointSetRegistrator(cb, 4, ransacReprojThreshold, confidence, maxIters)->run(src, dst, H, tempMask);
    else if( method == LMEDS )
        result = createLMeDSPointSetRegistrator(cb, 4, confidence, maxIters)->run(src, dst, H, tempMask);
    else
        CV_Error(Error::StsBadArg, "Unknown estimation method");

    if( result && npoints > 4 )
    {
        compressPoints( src.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, npoints );
        npoints = compressPoints( dst.ptr<Point2f>(), tempMask.ptr<uchar>(), 1, npoints );
        if( npoints > 0 )
        {
            Mat src1 = src.rowRange(0, npoints);
            Mat dst1 = dst.rowRange(0, npoints);
            src = src1;
            dst = dst1;
            if( method == RANSAC || method == LMEDS )
                cb->runKernel( src, dst, H );
            Mat H8(8, 1, CV_64F, H.ptr<double>());
            createLMSolver(new HomographyRefineCallback(src, dst), 10)->run(H8);
        }
    }

    if( result )
    {
        if( _mask.needed() )
            tempMask.copyTo(_mask);
    }
    else
        H.release();

    return H;
}

cv::Mat cv::findHomography( InputArray _points1, InputArray _points2,
                           OutputArray _mask, int method, double ransacReprojThreshold )
{
    return cv::findHomography(_points1, _points2, method, ransacReprojThreshold, _mask);
}



/* Estimation of Fundamental Matrix from point correspondences.
   The original code has been written by Valery Mosyagin */

/* The algorithms (except for RANSAC) and the notation have been taken from
   Zhengyou Zhang's research report
   "Determining the Epipolar Geometry and its Uncertainty: A Review"
   that can be found at http://www-sop.inria.fr/robotvis/personnel/zzhang/zzhang-eng.html */

/************************************** 7-point algorithm *******************************/
namespace cv
{

static int run7Point( const Mat& _m1, const Mat& _m2, Mat& _fmatrix )
{
    double a[7*9], w[7], u[9*9], v[9*9], c[4], r[3];
    double* f1, *f2;
    double t0, t1, t2;
    Mat A( 7, 9, CV_64F, a );
    Mat U( 7, 9, CV_64F, u );
    Mat Vt( 9, 9, CV_64F, v );
    Mat W( 7, 1, CV_64F, w );
    Mat coeffs( 1, 4, CV_64F, c );
    Mat roots( 1, 3, CV_64F, r );
    const Point2f* m1 = _m1.ptr<Point2f>();
    const Point2f* m2 = _m2.ptr<Point2f>();
    double* fmatrix = _fmatrix.ptr<double>();
    int i, k, n;

    // form a linear system: i-th row of A(=a) represents
    // the equation: (m2[i], 1)'*F*(m1[i], 1) = 0
    for( i = 0; i < 7; i++ )
    {
        double x0 = m1[i].x, y0 = m1[i].y;
        double x1 = m2[i].x, y1 = m2[i].y;

        a[i*9+0] = x1*x0;
        a[i*9+1] = x1*y0;
        a[i*9+2] = x1;
        a[i*9+3] = y1*x0;
        a[i*9+4] = y1*y0;
        a[i*9+5] = y1;
        a[i*9+6] = x0;
        a[i*9+7] = y0;
        a[i*9+8] = 1;
    }

    // A*(f11 f12 ... f33)' = 0 is singular (7 equations for 9 variables), so
    // the solution is linear subspace of dimensionality 2.
    // => use the last two singular vectors as a basis of the space
    // (according to SVD properties)
    SVDecomp( A, W, U, Vt, SVD::MODIFY_A + SVD::FULL_UV );
    f1 = v + 7*9;
    f2 = v + 8*9;

    // f1, f2 is a basis => lambda*f1 + mu*f2 is an arbitrary f. matrix.
    // as it is determined up to a scale, normalize lambda & mu (lambda + mu = 1),
    // so f ~ lambda*f1 + (1 - lambda)*f2.
    // use the additional constraint det(f) = det(lambda*f1 + (1-lambda)*f2) to find lambda.
    // it will be a cubic equation.
    // find c - polynomial coefficients.
    for( i = 0; i < 9; i++ )
        f1[i] -= f2[i];

    t0 = f2[4]*f2[8] - f2[5]*f2[7];
    t1 = f2[3]*f2[8] - f2[5]*f2[6];
    t2 = f2[3]*f2[7] - f2[4]*f2[6];

    c[3] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2;

    c[2] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2 -
    f1[3]*(f2[1]*f2[8] - f2[2]*f2[7]) +
    f1[4]*(f2[0]*f2[8] - f2[2]*f2[6]) -
    f1[5]*(f2[0]*f2[7] - f2[1]*f2[6]) +
    f1[6]*(f2[1]*f2[5] - f2[2]*f2[4]) -
    f1[7]*(f2[0]*f2[5] - f2[2]*f2[3]) +
    f1[8]*(f2[0]*f2[4] - f2[1]*f2[3]);

    t0 = f1[4]*f1[8] - f1[5]*f1[7];
    t1 = f1[3]*f1[8] - f1[5]*f1[6];
    t2 = f1[3]*f1[7] - f1[4]*f1[6];

    c[1] = f2[0]*t0 - f2[1]*t1 + f2[2]*t2 -
    f2[3]*(f1[1]*f1[8] - f1[2]*f1[7]) +
    f2[4]*(f1[0]*f1[8] - f1[2]*f1[6]) -
    f2[5]*(f1[0]*f1[7] - f1[1]*f1[6]) +
    f2[6]*(f1[1]*f1[5] - f1[2]*f1[4]) -
    f2[7]*(f1[0]*f1[5] - f1[2]*f1[3]) +
    f2[8]*(f1[0]*f1[4] - f1[1]*f1[3]);

    c[0] = f1[0]*t0 - f1[1]*t1 + f1[2]*t2;

    // solve the cubic equation; there can be 1 to 3 roots ...
    n = solveCubic( coeffs, roots );

    if( n < 1 || n > 3 )
        return n;

    for( k = 0; k < n; k++, fmatrix += 9 )
    {
        // for each root form the fundamental matrix
        double lambda = r[k], mu = 1.;
        double s = f1[8]*r[k] + f2[8];

        // normalize each matrix, so that F(3,3) (~fmatrix[8]) == 1
        if( fabs(s) > DBL_EPSILON )
        {
            mu = 1./s;
            lambda *= mu;
            fmatrix[8] = 1.;
        }
        else
            fmatrix[8] = 0.;

        for( i = 0; i < 8; i++ )
            fmatrix[i] = f1[i]*lambda + f2[i]*mu;
    }

    return n;
}


static int run8Point( const Mat& _m1, const Mat& _m2, Mat& _fmatrix )
{
    double a[9*9], w[9], v[9*9];
    Mat W( 9, 1, CV_64F, w );
    Mat V( 9, 9, CV_64F, v );
    Mat A( 9, 9, CV_64F, a );
    Mat U, F0, TF;

    Point2d m1c(0,0), m2c(0,0);
    double t, scale1 = 0, scale2 = 0;

    const Point2f* m1 = _m1.ptr<Point2f>();
    const Point2f* m2 = _m2.ptr<Point2f>();
    double* fmatrix = _fmatrix.ptr<double>();
    CV_Assert( (_m1.cols == 1 || _m1.rows == 1) && _m1.size() == _m2.size());
    int i, j, k, count = _m1.checkVector(2);

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < count; i++ )
    {
        double x = m1[i].x, y = m1[i].y;
        m1c.x += x; m1c.y += y;

        x = m2[i].x, y = m2[i].y;
        m2c.x += x; m2c.y += y;
    }

    // calculate the normalizing transformations for each of the point sets:
    // after the transformation each set will have the mass center at the coordinate origin
    // and the average distance from the origin will be ~sqrt(2).
    t = 1./count;
    m1c.x *= t; m1c.y *= t;
    m2c.x *= t; m2c.y *= t;

    for( i = 0; i < count; i++ )
    {
        double x = m1[i].x - m1c.x, y = m1[i].y - m1c.y;
        scale1 += std::sqrt(x*x + y*y);

        x = m2[i].x - m2c.x, y = m2[i].y - m2c.y;
        scale2 += std::sqrt(x*x + y*y);
    }

    scale1 *= t;
    scale2 *= t;

    if( scale1 < FLT_EPSILON || scale2 < FLT_EPSILON )
        return 0;

    scale1 = std::sqrt(2.)/scale1;
    scale2 = std::sqrt(2.)/scale2;

    A.setTo(Scalar::all(0));

    // form a linear system Ax=0: for each selected pair of points m1 & m2,
    // the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
    // to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0.
    for( i = 0; i < count; i++ )
    {
        double x1 = (m1[i].x - m1c.x)*scale1;
        double y1 = (m1[i].y - m1c.y)*scale1;
        double x2 = (m2[i].x - m2c.x)*scale2;
        double y2 = (m2[i].y - m2c.y)*scale2;
        double r[9] = { x2*x1, x2*y1, x2, y2*x1, y2*y1, y2, x1, y1, 1 };
        for( j = 0; j < 9; j++ )
            for( k = 0; k < 9; k++ )
                a[j*9+k] += r[j]*r[k];
    }

    eigen(A, W, V);

    for( i = 0; i < 9; i++ )
    {
        if( fabs(w[i]) < DBL_EPSILON )
            break;
    }

    if( i < 8 )
        return 0;

    F0 = Mat( 3, 3, CV_64F, v + 9*8 ); // take the last column of v as a solution of Af = 0

    // make F0 singular (of rank 2) by decomposing it with SVD,
    // zeroing the last diagonal element of W and then composing the matrices back.

    // use v as a temporary storage for different 3x3 matrices
    W = U = V = TF = F0;
    W = Mat(3, 1, CV_64F, v);
    U = Mat(3, 3, CV_64F, v + 9);
    V = Mat(3, 3, CV_64F, v + 18);
    TF = Mat(3, 3, CV_64F, v + 27);

    SVDecomp( F0, W, U, V, SVD::MODIFY_A );
    W.at<double>(2) = 0.;

    // F0 <- U*diag([W(1), W(2), 0])*V'
    gemm( U, Mat::diag(W), 1., 0, 0., TF, GEMM_1_T );
    gemm( TF, V, 1., 0, 0., F0, 0/*CV_GEMM_B_T*/ );

    // apply the transformation that is inverse
    // to what we used to normalize the point coordinates
    double tt1[] = { scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 };
    double tt2[] = { scale2, 0, -scale2*m2c.x, 0, scale2, -scale2*m2c.y, 0, 0, 1 };
    Mat T1(3, 3, CV_64F, tt1), T2(3, 3, CV_64F, tt2);

    // F0 <- T2'*F0*T1
    gemm( T2, F0, 1., 0, 0., TF, GEMM_1_T );
    F0 = Mat(3, 3, CV_64F, fmatrix);
    gemm( TF, T1, 1., 0, 0., F0, 0 );

    // make F(3,3) = 1
    if( fabs(F0.at<double>(2,2)) > FLT_EPSILON )
        F0 *= 1./F0.at<double>(2,2);

    return 1;
}


class FMEstimatorCallback : public PointSetRegistrator::Callback
{
public:
    bool checkSubset( InputArray _ms1, InputArray _ms2, int count ) const
    {
        Mat ms1 = _ms1.getMat(), ms2 = _ms2.getMat();
        return !haveCollinearPoints(ms1, count) && !haveCollinearPoints(ms2, count);
    }

    int runKernel( InputArray _m1, InputArray _m2, OutputArray _model ) const
    {
        double f[9*3];
        Mat m1 = _m1.getMat(), m2 = _m2.getMat();
        int count = m1.checkVector(2);
        Mat F(count == 7 ? 9 : 3, 3, CV_64F, f);
        int n = count == 7 ? run7Point(m1, m2, F) : run8Point(m1, m2, F);

        if( n == 0 )
            _model.release();
        else
            F.rowRange(0, n*3).copyTo(_model);

        return n;
    }

    void computeError( InputArray _m1, InputArray _m2, InputArray _model, OutputArray _err ) const
    {
        Mat __m1 = _m1.getMat(), __m2 = _m2.getMat(), __model = _model.getMat();
        int i, count = __m1.checkVector(2);
        const Point2f* m1 = __m1.ptr<Point2f>();
        const Point2f* m2 = __m2.ptr<Point2f>();
        const double* F = __model.ptr<double>();
        _err.create(count, 1, CV_32F);
        float* err = _err.getMat().ptr<float>();

        for( i = 0; i < count; i++ )
        {
            double a, b, c, d1, d2, s1, s2;

            a = F[0]*m1[i].x + F[1]*m1[i].y + F[2];
            b = F[3]*m1[i].x + F[4]*m1[i].y + F[5];
            c = F[6]*m1[i].x + F[7]*m1[i].y + F[8];

            s2 = 1./(a*a + b*b);
            d2 = m2[i].x*a + m2[i].y*b + c;

            a = F[0]*m2[i].x + F[3]*m2[i].y + F[6];
            b = F[1]*m2[i].x + F[4]*m2[i].y + F[7];
            c = F[2]*m2[i].x + F[5]*m2[i].y + F[8];

            s1 = 1./(a*a + b*b);
            d1 = m1[i].x*a + m1[i].y*b + c;

            err[i] = (float)std::max(d1*d1*s1, d2*d2*s2);
        }
    }
};

}

cv::Mat cv::findFundamentalMat( InputArray _points1, InputArray _points2,
                                int method, double param1, double param2,
                                OutputArray _mask )
{
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    Mat m1, m2, F;
    int npoints = -1;

    for( int i = 1; i <= 2; i++ )
    {
        Mat& p = i == 1 ? points1 : points2;
        Mat& m = i == 1 ? m1 : m2;
        npoints = p.checkVector(2, -1, false);
        if( npoints < 0 )
        {
            npoints = p.checkVector(3, -1, false);
            if( npoints < 0 )
                CV_Error(Error::StsBadArg, "The input arrays should be 2D or 3D point sets");
            if( npoints == 0 )
                return Mat();
            convertPointsFromHomogeneous(p, p);
        }
        p.reshape(2, npoints).convertTo(m, CV_32F);
    }

    CV_Assert( m1.checkVector(2) == m2.checkVector(2) );

    if( npoints < 7 )
        return Mat();

    Ptr<PointSetRegistrator::Callback> cb = new FMEstimatorCallback;
    int result;

    if( npoints == 7 || method == FM_8POINT )
    {
        result = cb->runKernel(m1, m2, F);
        if( _mask.needed() )
        {
            _mask.create(npoints, 1, CV_8U, -1, true);
            Mat mask = _mask.getMat();
            CV_Assert( (mask.cols == 1 || mask.rows == 1) && (int)mask.total() == npoints );
            mask.setTo(Scalar::all(1));
        }
    }
    else
    {
        if( param1 <= 0 )
            param1 = 3;
        if( param2 < DBL_EPSILON || param2 > 1 - DBL_EPSILON )
            param2 = 0.99;

        if( (method & ~3) == FM_RANSAC && npoints >= 15 )
            result = createRANSACPointSetRegistrator(cb, 7, param1, param2)->run(m1, m2, F, _mask);
        else
            result = createLMeDSPointSetRegistrator(cb, 7, param2)->run(m1, m2, F, _mask);
    }

    if( result <= 0 )
        return Mat();

    return F;
}

cv::Mat cv::findFundamentalMat( InputArray _points1, InputArray _points2,
                               OutputArray _mask, int method, double param1, double param2 )
{
    return cv::findFundamentalMat(_points1, _points2, method, param1, param2, _mask);
}


void cv::computeCorrespondEpilines( InputArray _points, int whichImage,
                                    InputArray _Fmat, OutputArray _lines )
{
    double f[9];
    Mat tempF(3, 3, CV_64F, f);
    Mat points = _points.getMat(), F = _Fmat.getMat();

    if( !points.isContinuous() )
        points = points.clone();

    int npoints = points.checkVector(2);
    if( npoints < 0 )
    {
        npoints = points.checkVector(3);
        if( npoints < 0 )
            CV_Error( Error::StsBadArg, "The input should be a 2D or 3D point set");
        Mat temp;
        convertPointsFromHomogeneous(points, temp);
        points = temp;
    }
    int depth = points.depth();
    CV_Assert( depth == CV_32F || depth == CV_32S || depth == CV_64F );

    CV_Assert(F.size() == Size(3,3));
    F.convertTo(tempF, CV_64F);
    if( whichImage == 2 )
        transpose(tempF, tempF);

    int ltype = CV_MAKETYPE(MAX(depth, CV_32F), 3);
    _lines.create(npoints, 1, ltype);
    Mat lines = _lines.getMat();
    if( !lines.isContinuous() )
    {
        _lines.release();
        _lines.create(npoints, 1, ltype);
        lines = _lines.getMat();
    }
    CV_Assert( lines.isContinuous());

    if( depth == CV_32S || depth == CV_32F )
    {
        const Point* ptsi = (const Point*)points.data;
        const Point2f* ptsf = (const Point2f*)points.data;
        Point3f* dstf = lines.ptr<Point3f>();
        for( int i = 0; i < npoints; i++ )
        {
            Point2f pt = depth == CV_32F ? ptsf[i] : Point2f((float)ptsi[i].x, (float)ptsi[i].y);
            double a = f[0]*pt.x + f[1]*pt.y + f[2];
            double b = f[3]*pt.x + f[4]*pt.y + f[5];
            double c = f[6]*pt.x + f[7]*pt.y + f[8];
            double nu = a*a + b*b;
            nu = nu ? 1./std::sqrt(nu) : 1.;
            a *= nu; b *= nu; c *= nu;
            dstf[i] = Point3f((float)a, (float)b, (float)c);
        }
    }
    else
    {
        const Point2d* ptsd = (const Point2d*)points.data;
        Point3d* dstd = lines.ptr<Point3d>();
        for( int i = 0; i < npoints; i++ )
        {
            Point2d pt = ptsd[i];
            double a = f[0]*pt.x + f[1]*pt.y + f[2];
            double b = f[3]*pt.x + f[4]*pt.y + f[5];
            double c = f[6]*pt.x + f[7]*pt.y + f[8];
            double nu = a*a + b*b;
            nu = nu ? 1./std::sqrt(nu) : 1.;
            a *= nu; b *= nu; c *= nu;
            dstd[i] = Point3d(a, b, c);
        }
    }
}

void cv::convertPointsFromHomogeneous( InputArray _src, OutputArray _dst )
{
    Mat src = _src.getMat();
    if( !src.isContinuous() )
        src = src.clone();
    int i, npoints = src.checkVector(3), depth = src.depth(), cn = 3;
    if( npoints < 0 )
    {
        npoints = src.checkVector(4);
        CV_Assert(npoints >= 0);
        cn = 4;
    }
    CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F || depth == CV_64F));

    int dtype = CV_MAKETYPE(depth <= CV_32F ? CV_32F : CV_64F, cn-1);
    _dst.create(npoints, 1, dtype);
    Mat dst = _dst.getMat();
    if( !dst.isContinuous() )
    {
        _dst.release();
        _dst.create(npoints, 1, dtype);
        dst = _dst.getMat();
    }
    CV_Assert( dst.isContinuous() );

    if( depth == CV_32S )
    {
        if( cn == 3 )
        {
            const Point3i* sptr = (const Point3i*)src.data;
            Point2f* dptr = (Point2f*)dst.data;
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i].z != 0 ? 1.f/sptr[i].z : 1.f;
                dptr[i] = Point2f(sptr[i].x*scale, sptr[i].y*scale);
            }
        }
        else
        {
            const Vec4i* sptr = (const Vec4i*)src.data;
            Point3f* dptr = (Point3f*)dst.data;
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i][3] != 0 ? 1.f/sptr[i][3] : 1.f;
                dptr[i] = Point3f(sptr[i][0]*scale, sptr[i][1]*scale, sptr[i][2]*scale);
            }
        }
    }
    else if( depth == CV_32F )
    {
        if( cn == 3 )
        {
            const Point3f* sptr = (const Point3f*)src.data;
            Point2f* dptr = (Point2f*)dst.data;
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i].z != 0.f ? 1.f/sptr[i].z : 1.f;
                dptr[i] = Point2f(sptr[i].x*scale, sptr[i].y*scale);
            }
        }
        else
        {
            const Vec4f* sptr = (const Vec4f*)src.data;
            Point3f* dptr = (Point3f*)dst.data;
            for( i = 0; i < npoints; i++ )
            {
                float scale = sptr[i][3] != 0.f ? 1.f/sptr[i][3] : 1.f;
                dptr[i] = Point3f(sptr[i][0]*scale, sptr[i][1]*scale, sptr[i][2]*scale);
            }
        }
    }
    else if( depth == CV_64F )
    {
        if( cn == 3 )
        {
            const Point3d* sptr = (const Point3d*)src.data;
            Point2d* dptr = (Point2d*)dst.data;
            for( i = 0; i < npoints; i++ )
            {
                double scale = sptr[i].z != 0. ? 1./sptr[i].z : 1.;
                dptr[i] = Point2d(sptr[i].x*scale, sptr[i].y*scale);
            }
        }
        else
        {
            const Vec4d* sptr = (const Vec4d*)src.data;
            Point3d* dptr = (Point3d*)dst.data;
            for( i = 0; i < npoints; i++ )
            {
                double scale = sptr[i][3] != 0.f ? 1./sptr[i][3] : 1.;
                dptr[i] = Point3d(sptr[i][0]*scale, sptr[i][1]*scale, sptr[i][2]*scale);
            }
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");
}


void cv::convertPointsToHomogeneous( InputArray _src, OutputArray _dst )
{
    Mat src = _src.getMat();
    if( !src.isContinuous() )
        src = src.clone();
    int i, npoints = src.checkVector(2), depth = src.depth(), cn = 2;
    if( npoints < 0 )
    {
        npoints = src.checkVector(3);
        CV_Assert(npoints >= 0);
        cn = 3;
    }
    CV_Assert( npoints >= 0 && (depth == CV_32S || depth == CV_32F || depth == CV_64F));

    int dtype = CV_MAKETYPE(depth <= CV_32F ? CV_32F : CV_64F, cn+1);
    _dst.create(npoints, 1, dtype);
    Mat dst = _dst.getMat();
    if( !dst.isContinuous() )
    {
        _dst.release();
        _dst.create(npoints, 1, dtype);
        dst = _dst.getMat();
    }
    CV_Assert( dst.isContinuous() );

    if( depth == CV_32S )
    {
        if( cn == 2 )
        {
            const Point2i* sptr = (const Point2i*)src.data;
            Point3i* dptr = (Point3i*)dst.data;
            for( i = 0; i < npoints; i++ )
                dptr[i] = Point3i(sptr[i].x, sptr[i].y, 1);
        }
        else
        {
            const Point3i* sptr = (const Point3i*)src.data;
            Vec4i* dptr = (Vec4i*)dst.data;
            for( i = 0; i < npoints; i++ )
                dptr[i] = Vec4i(sptr[i].x, sptr[i].y, sptr[i].z, 1);
        }
    }
    else if( depth == CV_32F )
    {
        if( cn == 2 )
        {
            const Point2f* sptr = (const Point2f*)src.data;
            Point3f* dptr = (Point3f*)dst.data;
            for( i = 0; i < npoints; i++ )
                dptr[i] = Point3f(sptr[i].x, sptr[i].y, 1.f);
        }
        else
        {
            const Point3f* sptr = (const Point3f*)src.data;
            Vec4f* dptr = (Vec4f*)dst.data;
            for( i = 0; i < npoints; i++ )
                dptr[i] = Vec4f(sptr[i].x, sptr[i].y, sptr[i].z, 1.f);
        }
    }
    else if( depth == CV_64F )
    {
        if( cn == 2 )
        {
            const Point2d* sptr = (const Point2d*)src.data;
            Point3d* dptr = (Point3d*)dst.data;
            for( i = 0; i < npoints; i++ )
                dptr[i] = Point3d(sptr[i].x, sptr[i].y, 1.);
        }
        else
        {
            const Point3d* sptr = (const Point3d*)src.data;
            Vec4d* dptr = (Vec4d*)dst.data;
            for( i = 0; i < npoints; i++ )
                dptr[i] = Vec4d(sptr[i].x, sptr[i].y, sptr[i].z, 1.);
        }
    }
    else
        CV_Error(Error::StsUnsupportedFormat, "");
}


void cv::convertPointsHomogeneous( InputArray _src, OutputArray _dst )
{
    int stype = _src.type(), dtype = _dst.type();
    CV_Assert( _dst.fixedType() );

    if( CV_MAT_CN(stype) > CV_MAT_CN(dtype) )
        convertPointsFromHomogeneous(_src, _dst);
    else
        convertPointsToHomogeneous(_src, _dst);
}

/* End of file. */
