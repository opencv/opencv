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
#include "_modelest.h"

using namespace cv;

template<typename T> int icvCompressPoints( T* ptr, const uchar* mask, int mstep, int count )
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

class CvHomographyEstimator : public CvModelEstimator2
{
public:
    CvHomographyEstimator( int modelPoints );

    virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
    virtual bool refine( const CvMat* m1, const CvMat* m2,
                         CvMat* model, int maxIters );
protected:
    virtual void computeReprojError( const CvMat* m1, const CvMat* m2,
                                     const CvMat* model, CvMat* error );
};


CvHomographyEstimator::CvHomographyEstimator(int _modelPoints)
    : CvModelEstimator2(_modelPoints, cvSize(3,3), 1)
{
    assert( _modelPoints == 4 || _modelPoints == 5 );
    checkPartialSubsets = false;
}

int CvHomographyEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* H )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;

    double LtL[9][9], W[9][9], V[9][9];
    CvMat _LtL = cvMat( 9, 9, CV_64F, LtL );
    CvMat matW = cvMat( 9, 9, CV_64F, W );
    CvMat matV = cvMat( 9, 9, CV_64F, V );
    CvMat _H0 = cvMat( 3, 3, CV_64F, V[8] );
    CvMat _Htemp = cvMat( 3, 3, CV_64F, V[7] );
    CvPoint2D64f cM={0,0}, cm={0,0}, sM={0,0}, sm={0,0};

    for( i = 0; i < count; i++ )
    {
        cm.x += m[i].x; cm.y += m[i].y;
        cM.x += M[i].x; cM.y += M[i].y;
    }

    cm.x /= count; cm.y /= count;
    cM.x /= count; cM.y /= count;

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
    CvMat _invHnorm = cvMat( 3, 3, CV_64FC1, invHnorm );
    CvMat _Hnorm2 = cvMat( 3, 3, CV_64FC1, Hnorm2 );

    cvZero( &_LtL );
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
    cvCompleteSymm( &_LtL );

    //cvSVD( &_LtL, &matW, 0, &matV, CV_SVD_MODIFY_A + CV_SVD_V_T );
    cvEigenVV( &_LtL, &matV, &matW );
    cvMatMul( &_invHnorm, &_H0, &_Htemp );
    cvMatMul( &_Htemp, &_Hnorm2, &_H0 );
    cvConvertScale( &_H0, H, 1./_H0.data.db[8] );

    return 1;
}


void CvHomographyEstimator::computeReprojError( const CvMat* m1, const CvMat* m2,
                                                const CvMat* model, CvMat* _err )
{
    int i, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    const double* H = model->data.db;
    float* err = _err->data.fl;

    for( i = 0; i < count; i++ )
    {
        double ww = 1./(H[6]*M[i].x + H[7]*M[i].y + 1.);
        double dx = (H[0]*M[i].x + H[1]*M[i].y + H[2])*ww - m[i].x;
        double dy = (H[3]*M[i].x + H[4]*M[i].y + H[5])*ww - m[i].y;
        err[i] = (float)(dx*dx + dy*dy);
    }
}

bool CvHomographyEstimator::refine( const CvMat* m1, const CvMat* m2, CvMat* model, int maxIters )
{
    CvLevMarq solver(8, 0, cvTermCriteria(CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, maxIters, DBL_EPSILON));
    int i, j, k, count = m1->rows*m1->cols;
    const CvPoint2D64f* M = (const CvPoint2D64f*)m1->data.ptr;
    const CvPoint2D64f* m = (const CvPoint2D64f*)m2->data.ptr;
    CvMat modelPart = cvMat( solver.param->rows, solver.param->cols, model->type, model->data.ptr );
    cvCopy( &modelPart, solver.param );

    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;

        if( !solver.updateAlt( _param, _JtJ, _JtErr, _errNorm ))
            break;

        for( i = 0; i < count; i++ )
        {
            const double* h = _param->data.db;
            double Mx = M[i].x, My = M[i].y;
            double ww = h[6]*Mx + h[7]*My + 1.;
            ww = fabs(ww) > DBL_EPSILON ? 1./ww : 0;
            double _xi = (h[0]*Mx + h[1]*My + h[2])*ww;
            double _yi = (h[3]*Mx + h[4]*My + h[5])*ww;
            double err[] = { _xi - m[i].x, _yi - m[i].y };
            if( _JtJ || _JtErr )
            {
                double J[][8] =
                {
                    { Mx*ww, My*ww, ww, 0, 0, 0, -Mx*ww*_xi, -My*ww*_xi },
                    { 0, 0, 0, Mx*ww, My*ww, ww, -Mx*ww*_yi, -My*ww*_yi }
                };

                for( j = 0; j < 8; j++ )
                {
                    for( k = j; k < 8; k++ )
                        _JtJ->data.db[j*8+k] += J[0][j]*J[0][k] + J[1][j]*J[1][k];
                    _JtErr->data.db[j] += J[0][j]*err[0] + J[1][j]*err[1];
                }
            }
            if( _errNorm )
                *_errNorm += err[0]*err[0] + err[1]*err[1];
        }
    }

    cvCopy( solver.param, &modelPart );
    return true;
}


CV_IMPL int
cvFindHomography( const CvMat* objectPoints, const CvMat* imagePoints,
                  CvMat* __H, int method, double ransacReprojThreshold,
                  CvMat* mask )
{
    const double confidence = 0.995;
    const int maxIters = 2000;
    const double defaultRANSACReprojThreshold = 3;
    bool result = false;
    Ptr<CvMat> m, M, tempMask;

    double H[9];
    CvMat matH = cvMat( 3, 3, CV_64FC1, H );
    int count;

    CV_Assert( CV_IS_MAT(imagePoints) && CV_IS_MAT(objectPoints) );

    count = MAX(imagePoints->cols, imagePoints->rows);
    CV_Assert( count >= 4 );
    if( ransacReprojThreshold <= 0 )
        ransacReprojThreshold = defaultRANSACReprojThreshold;

    m = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( imagePoints, m );

    M = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( objectPoints, M );

    if( mask )
    {
        CV_Assert( CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
            (mask->rows == 1 || mask->cols == 1) &&
            mask->rows*mask->cols == count );
    }
    if( mask || count > 4 )
        tempMask = cvCreateMat( 1, count, CV_8U );
    if( !tempMask.empty() )
        cvSet( tempMask, cvScalarAll(1.) );

    CvHomographyEstimator estimator( MIN(count, 4) );
    if( count == 4 )
        method = 0;
    if( method == CV_LMEDS )
        result = estimator.runLMeDS( M, m, &matH, tempMask, confidence, maxIters );
    else if( method == CV_RANSAC )
        result = estimator.runRANSAC( M, m, &matH, tempMask, ransacReprojThreshold, confidence, maxIters);
    else
        result = estimator.runKernel( M, m, &matH ) > 0;

    if( result && count > 4 )
    {
        icvCompressPoints( (CvPoint2D64f*)M->data.ptr, tempMask->data.ptr, 1, count );
        count = icvCompressPoints( (CvPoint2D64f*)m->data.ptr, tempMask->data.ptr, 1, count );
        M->cols = m->cols = count;
        if( method == CV_RANSAC )
            estimator.runKernel( M, m, &matH );
        estimator.refine( M, m, &matH, 10 );
    }

    if( result )
        cvConvert( &matH, __H );
    
    if( mask && tempMask )
    {
        if( CV_ARE_SIZES_EQ(mask, tempMask) )
           cvCopy( tempMask, mask );
        else
           cvTranspose( tempMask, mask );
    }

    return (int)result;
}


/* Evaluation of Fundamental Matrix from point correspondences.
   The original code has been written by Valery Mosyagin */

/* The algorithms (except for RANSAC) and the notation have been taken from
   Zhengyou Zhang's research report
   "Determining the Epipolar Geometry and its Uncertainty: A Review"
   that can be found at http://www-sop.inria.fr/robotvis/personnel/zzhang/zzhang-eng.html */

/************************************** 7-point algorithm *******************************/
class CvFMEstimator : public CvModelEstimator2
{
public:
    CvFMEstimator( int _modelPoints );

    virtual int runKernel( const CvMat* m1, const CvMat* m2, CvMat* model );
    virtual int run7Point( const CvMat* m1, const CvMat* m2, CvMat* model );
    virtual int run8Point( const CvMat* m1, const CvMat* m2, CvMat* model );
protected:
    virtual void computeReprojError( const CvMat* m1, const CvMat* m2,
                                     const CvMat* model, CvMat* error );
};

CvFMEstimator::CvFMEstimator( int _modelPoints )
: CvModelEstimator2( _modelPoints, cvSize(3,3), _modelPoints == 7 ? 3 : 1 )
{
    assert( _modelPoints == 7 || _modelPoints == 8 );
}


int CvFMEstimator::runKernel( const CvMat* m1, const CvMat* m2, CvMat* model )
{
    return modelPoints == 7 ? run7Point( m1, m2, model ) : run8Point( m1, m2, model );
}

int CvFMEstimator::run7Point( const CvMat* _m1, const CvMat* _m2, CvMat* _fmatrix )
{
    double a[7*9], w[7], v[9*9], c[4], r[3];
    double* f1, *f2;
    double t0, t1, t2;
    CvMat A = cvMat( 7, 9, CV_64F, a );
    CvMat V = cvMat( 9, 9, CV_64F, v );
    CvMat W = cvMat( 7, 1, CV_64F, w );
    CvMat coeffs = cvMat( 1, 4, CV_64F, c );
    CvMat roots = cvMat( 1, 3, CV_64F, r );
    const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
    const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
    double* fmatrix = _fmatrix->data.db;
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
    cvSVD( &A, &W, 0, &V, CV_SVD_MODIFY_A + CV_SVD_V_T );
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
    n = cvSolveCubic( &coeffs, &roots );

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


int CvFMEstimator::run8Point( const CvMat* _m1, const CvMat* _m2, CvMat* _fmatrix )
{
    double a[9*9], w[9], v[9*9];
    CvMat W = cvMat( 1, 9, CV_64F, w );
    CvMat V = cvMat( 9, 9, CV_64F, v );
    CvMat A = cvMat( 9, 9, CV_64F, a );
    CvMat U, F0, TF;

    CvPoint2D64f m0c = {0,0}, m1c = {0,0};
    double t, scale0 = 0, scale1 = 0;

    const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
    const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
    double* fmatrix = _fmatrix->data.db;
    CV_Assert( (_m1->cols == 1 || _m1->rows == 1) && CV_ARE_SIZES_EQ(_m1, _m2));
    int i, j, k, count = _m1->cols*_m1->rows;

    // compute centers and average distances for each of the two point sets
    for( i = 0; i < count; i++ )
    {
        double x = m1[i].x, y = m1[i].y;
        m0c.x += x; m0c.y += y;

        x = m2[i].x, y = m2[i].y;
        m1c.x += x; m1c.y += y;
    }

    // calculate the normalizing transformations for each of the point sets:
    // after the transformation each set will have the mass center at the coordinate origin
    // and the average distance from the origin will be ~sqrt(2).
    t = 1./count;
    m0c.x *= t; m0c.y *= t;
    m1c.x *= t; m1c.y *= t;

    for( i = 0; i < count; i++ )
    {
        double x = m1[i].x - m0c.x, y = m1[i].y - m0c.y;
        scale0 += sqrt(x*x + y*y);

        x = m2[i].x - m1c.x, y = m2[i].y - m1c.y;
        scale1 += sqrt(x*x + y*y);
    }

    scale0 *= t;
    scale1 *= t;

    if( scale0 < FLT_EPSILON || scale1 < FLT_EPSILON )
        return 0;

    scale0 = sqrt(2.)/scale0;
    scale1 = sqrt(2.)/scale1;
    
    cvZero( &A );

    // form a linear system Ax=0: for each selected pair of points m1 & m2,
    // the row of A(=a) represents the coefficients of equation: (m2, 1)'*F*(m1, 1) = 0
    // to save computation time, we compute (At*A) instead of A and then solve (At*A)x=0. 
    for( i = 0; i < count; i++ )
    {
        double x0 = (m1[i].x - m0c.x)*scale0;
        double y0 = (m1[i].y - m0c.y)*scale0;
        double x1 = (m2[i].x - m1c.x)*scale1;
        double y1 = (m2[i].y - m1c.y)*scale1;
        double r[9] = { x1*x0, x1*y0, x1, y1*x0, y1*y0, y1, x0, y0, 1 };
        for( j = 0; j < 9; j++ )
            for( k = 0; k < 9; k++ )
                a[j*9+k] += r[j]*r[k];
    }

    cvSVD( &A, &W, 0, &V, CV_SVD_MODIFY_A + CV_SVD_V_T );

    for( i = 0; i < 8; i++ )
    {
        if( fabs(w[i]) < DBL_EPSILON )
            break;
    }

    if( i < 7 )
        return 0;

    F0 = cvMat( 3, 3, CV_64F, v + 9*8 ); // take the last column of v as a solution of Af = 0

    // make F0 singular (of rank 2) by decomposing it with SVD,
    // zeroing the last diagonal element of W and then composing the matrices back.

    // use v as a temporary storage for different 3x3 matrices
    W = U = V = TF = F0;
    W.data.db = v;
    U.data.db = v + 9;
    V.data.db = v + 18;
    TF.data.db = v + 27;

    cvSVD( &F0, &W, &U, &V, CV_SVD_MODIFY_A + CV_SVD_U_T + CV_SVD_V_T );
    W.data.db[8] = 0.;

    // F0 <- U*diag([W(1), W(2), 0])*V'
    cvGEMM( &U, &W, 1., 0, 0., &TF, CV_GEMM_A_T );
    cvGEMM( &TF, &V, 1., 0, 0., &F0, 0/*CV_GEMM_B_T*/ );

    // apply the transformation that is inverse
    // to what we used to normalize the point coordinates
    {
        double tt0[] = { scale0, 0, -scale0*m0c.x, 0, scale0, -scale0*m0c.y, 0, 0, 1 };
        double tt1[] = { scale1, 0, -scale1*m1c.x, 0, scale1, -scale1*m1c.y, 0, 0, 1 };
        CvMat T0, T1;
        T0 = T1 = F0;
        T0.data.db = tt0;
        T1.data.db = tt1;

        // F0 <- T1'*F0*T0
        cvGEMM( &T1, &F0, 1., 0, 0., &TF, CV_GEMM_A_T );
        F0.data.db = fmatrix;
        cvGEMM( &TF, &T0, 1., 0, 0., &F0, 0 );

        // make F(3,3) = 1
        if( fabs(F0.data.db[8]) > FLT_EPSILON )
            cvScale( &F0, &F0, 1./F0.data.db[8] );
    }

    return 1;
}


void CvFMEstimator::computeReprojError( const CvMat* _m1, const CvMat* _m2,
                                        const CvMat* model, CvMat* _err )
{
    int i, count = _m1->rows*_m1->cols;
    const CvPoint2D64f* m1 = (const CvPoint2D64f*)_m1->data.ptr;
    const CvPoint2D64f* m2 = (const CvPoint2D64f*)_m2->data.ptr;
    const double* F = model->data.db;
    float* err = _err->data.fl;
    
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


CV_IMPL int cvFindFundamentalMat( const CvMat* points1, const CvMat* points2,
                                  CvMat* fmatrix, int method,
                                  double param1, double param2, CvMat* mask )
{
    int result = 0;
    Ptr<CvMat> m1, m2, tempMask;

    double F[3*9];
    CvMat _F3x3 = cvMat( 3, 3, CV_64FC1, F ), _F9x3 = cvMat( 9, 3, CV_64FC1, F );
    int count;

    CV_Assert( CV_IS_MAT(points1) && CV_IS_MAT(points2) && CV_ARE_SIZES_EQ(points1, points2) );
    CV_Assert( CV_IS_MAT(fmatrix) && fmatrix->cols == 3 &&
        (fmatrix->rows == 3 || (fmatrix->rows == 9 && method == CV_FM_7POINT)) );

    count = MAX(points1->cols, points1->rows);
    if( count < 7 )
        return 0;

    m1 = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( points1, m1 );

    m2 = cvCreateMat( 1, count, CV_64FC2 );
    cvConvertPointsHomogeneous( points2, m2 );

    if( mask )
    {
        CV_Assert( CV_IS_MASK_ARR(mask) && CV_IS_MAT_CONT(mask->type) &&
            (mask->rows == 1 || mask->cols == 1) &&
            mask->rows*mask->cols == count );
    }
    if( mask || count > 8 )
        tempMask = cvCreateMat( 1, count, CV_8U );
    if( !tempMask.empty() )
        cvSet( tempMask, cvScalarAll(1.) );

    CvFMEstimator estimator(7);
    if( count == 7 )
        result = estimator.run7Point(m1, m2, &_F9x3);
    else if( count == 8 || method == CV_FM_8POINT )
        result = estimator.run8Point(m1, m2, &_F3x3);
    else if( count >= 8 )
    {
        if( param1 <= 0 )
            param1 = 3;
        if( param2 < DBL_EPSILON || param2 > 1 - DBL_EPSILON )
            param2 = 0.99;
        
        if( (method & ~3) == CV_RANSAC && count >= 15 )
            result = estimator.runRANSAC(m1, m2, &_F3x3, tempMask, param1, param2 );
        else
            result = estimator.runLMeDS(m1, m2, &_F3x3, tempMask, param2 );
        if( result <= 0 )
            return 0;
        /*icvCompressPoints( (CvPoint2D64f*)m1->data.ptr, tempMask->data.ptr, 1, count );
        count = icvCompressPoints( (CvPoint2D64f*)m2->data.ptr, tempMask->data.ptr, 1, count );
        assert( count >= 8 );
        m1->cols = m2->cols = count;
        estimator.run8Point(m1, m2, &_F3x3);*/
    }

    if( result )
        cvConvert( fmatrix->rows == 3 ? &_F3x3 : &_F9x3, fmatrix );
    
    if( mask && tempMask )
    {
        if( CV_ARE_SIZES_EQ(mask, tempMask) )
            cvCopy( tempMask, mask );
        else
            cvTranspose( tempMask, mask );
    }

    return result;
}


CV_IMPL void cvComputeCorrespondEpilines( const CvMat* points, int pointImageID,
                                          const CvMat* fmatrix, CvMat* lines )
{
    int abc_stride, abc_plane_stride, abc_elem_size;
    int plane_stride, stride, elem_size;
    int i, dims, count, depth, cn, abc_dims, abc_count, abc_depth, abc_cn;
    uchar *ap, *bp, *cp;
    const uchar *xp, *yp, *zp;
    double f[9];
    CvMat F = cvMat( 3, 3, CV_64F, f );

    if( !CV_IS_MAT(points) )
        CV_Error( !points ? CV_StsNullPtr : CV_StsBadArg, "points parameter is not a valid matrix" );

    depth = CV_MAT_DEPTH(points->type);
    cn = CV_MAT_CN(points->type);
    if( (depth != CV_32F && depth != CV_64F) || (cn != 1 && cn != 2 && cn != 3) )
        CV_Error( CV_StsUnsupportedFormat, "The format of point matrix is unsupported" );

    if( points->rows > points->cols )
    {
        dims = cn*points->cols;
        count = points->rows;
    }
    else
    {
        if( (points->rows > 1 && cn > 1) || (points->rows == 1 && cn == 1) )
            CV_Error( CV_StsBadSize, "The point matrix does not have a proper layout (2xn, 3xn, nx2 or nx3)" );
        dims = cn * points->rows;
        count = points->cols;
    }

    if( dims != 2 && dims != 3 )
        CV_Error( CV_StsOutOfRange, "The dimensionality of points must be 2 or 3" );

    if( !CV_IS_MAT(fmatrix) )
        CV_Error( !fmatrix ? CV_StsNullPtr : CV_StsBadArg, "fmatrix is not a valid matrix" );

    if( CV_MAT_TYPE(fmatrix->type) != CV_32FC1 && CV_MAT_TYPE(fmatrix->type) != CV_64FC1 )
        CV_Error( CV_StsUnsupportedFormat, "fundamental matrix must have 32fC1 or 64fC1 type" );

    if( fmatrix->cols != 3 || fmatrix->rows != 3 )
        CV_Error( CV_StsBadSize, "fundamental matrix must be 3x3" );

    if( !CV_IS_MAT(lines) )
        CV_Error( !lines ? CV_StsNullPtr : CV_StsBadArg, "lines parameter is not a valid matrix" );

    abc_depth = CV_MAT_DEPTH(lines->type);
    abc_cn = CV_MAT_CN(lines->type);
    if( (abc_depth != CV_32F && abc_depth != CV_64F) || (abc_cn != 1 && abc_cn != 3) )
        CV_Error( CV_StsUnsupportedFormat, "The format of the matrix of lines is unsupported" );

    if( lines->rows > lines->cols )
    {
        abc_dims = abc_cn*lines->cols;
        abc_count = lines->rows;
    }
    else
    {
        if( (lines->rows > 1 && abc_cn > 1) || (lines->rows == 1 && abc_cn == 1) )
            CV_Error( CV_StsBadSize, "The lines matrix does not have a proper layout (3xn or nx3)" );
        abc_dims = abc_cn * lines->rows;
        abc_count = lines->cols;
    }

    if( abc_dims != 3 )
        CV_Error( CV_StsOutOfRange, "The lines matrix does not have a proper layout (3xn or nx3)" );

    if( abc_count != count )
        CV_Error( CV_StsUnmatchedSizes, "The numbers of points and lines are different" );

    elem_size = CV_ELEM_SIZE(depth);
    abc_elem_size = CV_ELEM_SIZE(abc_depth);

    if( points->rows == dims )
    {
        plane_stride = points->step;
        stride = elem_size;
    }
    else
    {
        plane_stride = elem_size;
        stride = points->rows == 1 ? dims*elem_size : points->step;
    }

    if( lines->rows == 3 )
    {
        abc_plane_stride = lines->step;
        abc_stride = abc_elem_size;
    }
    else
    {
        abc_plane_stride = abc_elem_size;
        abc_stride = lines->rows == 1 ? 3*abc_elem_size : lines->step;
    }

    cvConvert( fmatrix, &F );
    if( pointImageID == 2 )
        cvTranspose( &F, &F );

    xp = points->data.ptr;
    yp = xp + plane_stride;
    zp = dims == 3 ? yp + plane_stride : 0;

    ap = lines->data.ptr;
    bp = ap + abc_plane_stride;
    cp = bp + abc_plane_stride;

    for( i = 0; i < count; i++ )
    {
        double x, y, z = 1.;
        double a, b, c, nu;

        if( depth == CV_32F )
        {
            x = *(float*)xp; y = *(float*)yp;
            if( zp )
                z = *(float*)zp, zp += stride;
        }
        else
        {
            x = *(double*)xp; y = *(double*)yp;
            if( zp )
                z = *(double*)zp, zp += stride;
        }

        xp += stride; yp += stride;

        a = f[0]*x + f[1]*y + f[2]*z;
        b = f[3]*x + f[4]*y + f[5]*z;
        c = f[6]*x + f[7]*y + f[8]*z;
        nu = a*a + b*b;
        nu = nu ? 1./sqrt(nu) : 1.;
        a *= nu; b *= nu; c *= nu;

        if( abc_depth == CV_32F )
        {
            *(float*)ap = (float)a;
            *(float*)bp = (float)b;
            *(float*)cp = (float)c;
        }
        else
        {
            *(double*)ap = a;
            *(double*)bp = b;
            *(double*)cp = c;
        }

        ap += abc_stride;
        bp += abc_stride;
        cp += abc_stride;
    }
}


CV_IMPL void cvConvertPointsHomogeneous( const CvMat* src, CvMat* dst )
{
    Ptr<CvMat> temp, denom;

    int i, s_count, s_dims, d_count, d_dims;
    CvMat _src, _dst, _ones;
    CvMat* ones = 0;

    if( !CV_IS_MAT(src) )
        CV_Error( !src ? CV_StsNullPtr : CV_StsBadArg,
        "The input parameter is not a valid matrix" );

    if( !CV_IS_MAT(dst) )
        CV_Error( !dst ? CV_StsNullPtr : CV_StsBadArg,
        "The output parameter is not a valid matrix" );

    if( src == dst || src->data.ptr == dst->data.ptr )
    {
        if( src != dst && (!CV_ARE_TYPES_EQ(src, dst) || !CV_ARE_SIZES_EQ(src,dst)) )
            CV_Error( CV_StsBadArg, "Invalid inplace operation" );
        return;
    }

    if( src->rows > src->cols )
    {
        if( !((src->cols > 1) ^ (CV_MAT_CN(src->type) > 1)) )
            CV_Error( CV_StsBadSize, "Either the number of channels or columns or rows must be =1" );

        s_dims = CV_MAT_CN(src->type)*src->cols;
        s_count = src->rows;
    }
    else
    {
        if( !((src->rows > 1) ^ (CV_MAT_CN(src->type) > 1)) )
            CV_Error( CV_StsBadSize, "Either the number of channels or columns or rows must be =1" );

        s_dims = CV_MAT_CN(src->type)*src->rows;
        s_count = src->cols;
    }

    if( src->rows == 1 || src->cols == 1 )
        src = cvReshape( src, &_src, 1, s_count );

    if( dst->rows > dst->cols )
    {
        if( !((dst->cols > 1) ^ (CV_MAT_CN(dst->type) > 1)) )
            CV_Error( CV_StsBadSize,
            "Either the number of channels or columns or rows in the input matrix must be =1" );

        d_dims = CV_MAT_CN(dst->type)*dst->cols;
        d_count = dst->rows;
    }
    else
    {
        if( !((dst->rows > 1) ^ (CV_MAT_CN(dst->type) > 1)) )
            CV_Error( CV_StsBadSize,
            "Either the number of channels or columns or rows in the output matrix must be =1" );

        d_dims = CV_MAT_CN(dst->type)*dst->rows;
        d_count = dst->cols;
    }

    if( dst->rows == 1 || dst->cols == 1 )
        dst = cvReshape( dst, &_dst, 1, d_count );

    if( s_count != d_count )
        CV_Error( CV_StsUnmatchedSizes, "Both matrices must have the same number of points" );

    if( CV_MAT_DEPTH(src->type) < CV_32F || CV_MAT_DEPTH(dst->type) < CV_32F )
        CV_Error( CV_StsUnsupportedFormat,
        "Both matrices must be floating-point (single or double precision)" );

    if( s_dims < 2 || s_dims > 4 || d_dims < 2 || d_dims > 4 )
        CV_Error( CV_StsOutOfRange,
        "Both input and output point dimensionality must be 2, 3 or 4" );

    if( s_dims < d_dims - 1 || s_dims > d_dims + 1 )
        CV_Error( CV_StsUnmatchedSizes,
        "The dimensionalities of input and output point sets differ too much" );

    if( s_dims == d_dims - 1 )
    {
        if( d_count == dst->rows )
        {
            ones = cvGetSubRect( dst, &_ones, cvRect( s_dims, 0, 1, d_count ));
            dst = cvGetSubRect( dst, &_dst, cvRect( 0, 0, s_dims, d_count ));
        }
        else
        {
            ones = cvGetSubRect( dst, &_ones, cvRect( 0, s_dims, d_count, 1 ));
            dst = cvGetSubRect( dst, &_dst, cvRect( 0, 0, d_count, s_dims ));
        }
    }

    if( s_dims <= d_dims )
    {
        if( src->rows == dst->rows && src->cols == dst->cols )
        {
            if( CV_ARE_TYPES_EQ( src, dst ) )
                cvCopy( src, dst );
            else
                cvConvert( src, dst );
        }
        else
        {
            if( !CV_ARE_TYPES_EQ( src, dst ))
            {
                temp = cvCreateMat( src->rows, src->cols, dst->type );
                cvConvert( src, temp );
                src = temp;
            }
            cvTranspose( src, dst );
        }

        if( ones )
            cvSet( ones, cvRealScalar(1.) );
    }
    else
    {
        int s_plane_stride, s_stride, d_plane_stride, d_stride, elem_size;

        if( !CV_ARE_TYPES_EQ( src, dst ))
        {
            temp = cvCreateMat( src->rows, src->cols, dst->type );
            cvConvert( src, temp );
            src = temp;
        }

        elem_size = CV_ELEM_SIZE(src->type);

        if( s_count == src->cols )
            s_plane_stride = src->step / elem_size, s_stride = 1;
        else
            s_stride = src->step / elem_size, s_plane_stride = 1;

        if( d_count == dst->cols )
            d_plane_stride = dst->step / elem_size, d_stride = 1;
        else
            d_stride = dst->step / elem_size, d_plane_stride = 1;

        denom = cvCreateMat( 1, d_count, dst->type );

        if( CV_MAT_DEPTH(dst->type) == CV_32F )
        {
            const float* xs = src->data.fl;
            const float* ys = xs + s_plane_stride;
            const float* zs = 0;
            const float* ws = xs + (s_dims - 1)*s_plane_stride;

            float* iw = denom->data.fl;

            float* xd = dst->data.fl;
            float* yd = xd + d_plane_stride;
            float* zd = 0;

            if( d_dims == 3 )
            {
                zs = ys + s_plane_stride;
                zd = yd + d_plane_stride;
            }

            for( i = 0; i < d_count; i++, ws += s_stride )
            {
                float t = *ws;
                iw[i] = fabs((double)t) > FLT_EPSILON ? t : 1.f;
            }

            cvDiv( 0, denom, denom );

            if( d_dims == 3 )
                for( i = 0; i < d_count; i++ )
                {
                    float w = iw[i];
                    float x = *xs * w, y = *ys * w, z = *zs * w;
                    xs += s_stride; ys += s_stride; zs += s_stride;
                    *xd = x; *yd = y; *zd = z;
                    xd += d_stride; yd += d_stride; zd += d_stride;
                }
            else
                for( i = 0; i < d_count; i++ )
                {
                    float w = iw[i];
                    float x = *xs * w, y = *ys * w;
                    xs += s_stride; ys += s_stride;
                    *xd = x; *yd = y;
                    xd += d_stride; yd += d_stride;
                }
        }
        else
        {
            const double* xs = src->data.db;
            const double* ys = xs + s_plane_stride;
            const double* zs = 0;
            const double* ws = xs + (s_dims - 1)*s_plane_stride;

            double* iw = denom->data.db;

            double* xd = dst->data.db;
            double* yd = xd + d_plane_stride;
            double* zd = 0;

            if( d_dims == 3 )
            {
                zs = ys + s_plane_stride;
                zd = yd + d_plane_stride;
            }

            for( i = 0; i < d_count; i++, ws += s_stride )
            {
                double t = *ws;
                iw[i] = fabs(t) > DBL_EPSILON ? t : 1.;
            }

            cvDiv( 0, denom, denom );

            if( d_dims == 3 )
                for( i = 0; i < d_count; i++ )
                {
                    double w = iw[i];
                    double x = *xs * w, y = *ys * w, z = *zs * w;
                    xs += s_stride; ys += s_stride; zs += s_stride;
                    *xd = x; *yd = y; *zd = z;
                    xd += d_stride; yd += d_stride; zd += d_stride;
                }
            else
                for( i = 0; i < d_count; i++ )
                {
                    double w = iw[i];
                    double x = *xs * w, y = *ys * w;
                    xs += s_stride; ys += s_stride;
                    *xd = x; *yd = y;
                    xd += d_stride; yd += d_stride;
                }
        }
    }
}

cv::Mat cv::findHomography( const InputArray& _points1, const InputArray& _points2,
                            int method, double ransacReprojThreshold, OutputArray _mask )
{
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
               points1.type() == points2.type());
    
    Mat H(3, 3, CV_64F);
    CvMat _pt1 = points1, _pt2 = points2;
    CvMat matH = H, c_mask, *p_mask = 0;
    if( _mask.needed() )
    {
        _mask.create(npoints, 1, CV_8U, -1, true);
        p_mask = &(c_mask = _mask.getMat());
    }
    bool ok = cvFindHomography( &_pt1, &_pt2, &matH, method, ransacReprojThreshold, p_mask ) > 0;
    if( !ok )
        H = Scalar(0);
    return H;
}

cv::Mat cv::findHomography( const InputArray& _points1, const InputArray& _points2,
                            OutputArray _mask, int method, double ransacReprojThreshold )
{
    return cv::findHomography(_points1, _points2, method, ransacReprojThreshold, _mask);
}

cv::Mat cv::findFundamentalMat( const InputArray& _points1, const InputArray& _points2,
                               int method, double param1, double param2,
                               OutputArray _mask )
{
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    int npoints = points1.checkVector(2);
    CV_Assert( npoints >= 0 && points2.checkVector(2) == npoints &&
              points1.type() == points2.type());
    
    Mat F(3, 3, CV_64F);
    CvMat _pt1 = points1, _pt2 = points2;
    CvMat matF = F, c_mask, *p_mask = 0;
    if( _mask.needed() )
    {
        _mask.create(npoints, 1, CV_8U, -1, true);
        p_mask = &(c_mask = _mask.getMat());
    }
    int n = cvFindFundamentalMat( &_pt1, &_pt2, &matF, method, param1, param2, p_mask );
    if( n <= 0 )
        F = Scalar(0);
    return F;
}

cv::Mat cv::findFundamentalMat( const InputArray& _points1, const InputArray& _points2,
                                OutputArray _mask, int method, double param1, double param2 )
{
    return cv::findFundamentalMat(_points1, _points2, method, param1, param2, _mask);
}


void cv::computeCorrespondEpilines( const InputArray& _points, int whichImage,
                                    const InputArray& _Fmat, OutputArray _lines )
{
    Mat points = _points.getMat(), F = _Fmat.getMat();
    int npoints = points.checkVector(2);
    CV_Assert( npoints >= 0 && (points.depth() == CV_32F || points.depth() == CV_32S));
    
    _lines.create(npoints, 1, CV_32FC3, -1, true);
    CvMat c_points = points, c_lines = _lines.getMat(), c_F = F;
    cvComputeCorrespondEpilines(&c_points, whichImage, &c_F, &c_lines);
}

void cv::convertPointsFromHomogeneous( const InputArray& _src, OutputArray _dst )
{
    Mat src = _src.getMat();
    int npoints = src.checkVector(3), cn = 3;
    if( npoints < 0 )
    {
        npoints = src.checkVector(4);
        if( npoints >= 0 )
            cn = 4;
    }
    CV_Assert( npoints >= 0 && (src.depth() == CV_32F || src.depth() == CV_32S));
    
    _dst.create(npoints, 1, CV_MAKETYPE(CV_32F, cn-1));
    CvMat c_src = src, c_dst = _dst.getMat();
    cvConvertPointsHomogeneous(&c_src, &c_dst);
}

void cv::convertPointsToHomogeneous( const InputArray& _src, OutputArray _dst )
{
    Mat src = _src.getMat();
    int npoints = src.checkVector(2), cn = 2;
    if( npoints < 0 )
    {
        npoints = src.checkVector(3);
        if( npoints >= 0 )
            cn = 3;
    }
    CV_Assert( npoints >= 0 && (src.depth() == CV_32F || src.depth() == CV_32S));
    
    _dst.create(npoints, 1, CV_MAKETYPE(CV_32F, cn+1));
    CvMat c_src = src, c_dst = _dst.getMat();
    cvConvertPointsHomogeneous(&c_src, &c_dst);
}

void cv::convertPointsHomogeneous( const InputArray& _src, OutputArray _dst )
{
    int stype = _src.type(), dtype = _dst.type();
    CV_Assert( _dst.fixedType() );
    
    if( CV_MAT_CN(stype) > CV_MAT_CN(dtype) )
        convertPointsFromHomogeneous(_src, _dst);
    else
        convertPointsToHomogeneous(_src, _dst);
}

/* End of file. */
