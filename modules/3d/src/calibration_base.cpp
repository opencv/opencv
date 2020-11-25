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
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
#include "opencv2/imgproc/imgproc_c.h"
#include "distortion_model.hpp"
#include <stdio.h>
#include <iterator>

using namespace cv;

/*
    This is straight-forward port v3 of Matlab calibration engine by Jean-Yves Bouguet
    that is (in a large extent) based on the paper:
    Z. Zhang. "A flexible new technique for camera calibration".
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
    The 1st initial port was done by Valery Mosyagin.
*/

// reimplementation of dAB.m
static void cvCalcMatMulDeriv( const CvMat* A, const CvMat* B, CvMat* dABdA, CvMat* dABdB )
{
    int i, j, M, N, L;
    int bstep;

    CV_Assert( CV_IS_MAT(A) && CV_IS_MAT(B) );
    CV_Assert( CV_ARE_TYPES_EQ(A, B) &&
        (CV_MAT_TYPE(A->type) == CV_32F || CV_MAT_TYPE(A->type) == CV_64F) );
    CV_Assert( A->cols == B->rows );

    M = A->rows;
    L = A->cols;
    N = B->cols;
    bstep = B->step/CV_ELEM_SIZE(B->type);

    if( dABdA )
    {
        CV_Assert( CV_ARE_TYPES_EQ(A, dABdA) &&
            dABdA->rows == A->rows*B->cols && dABdA->cols == A->rows*A->cols );
    }

    if( dABdB )
    {
        CV_Assert( CV_ARE_TYPES_EQ(A, dABdB) &&
            dABdB->rows == A->rows*B->cols && dABdB->cols == B->rows*B->cols );
    }

    if( CV_MAT_TYPE(A->type) == CV_32F )
    {
        for( i = 0; i < M*N; i++ )
        {
            int i1 = i / N,  i2 = i % N;

            if( dABdA )
            {
                float* dcda = (float*)(dABdA->data.ptr + dABdA->step*i);
                const float* b = (const float*)B->data.ptr + i2;

                for( j = 0; j < M*L; j++ )
                    dcda[j] = 0;
                for( j = 0; j < L; j++ )
                    dcda[i1*L + j] = b[j*bstep];
            }

            if( dABdB )
            {
                float* dcdb = (float*)(dABdB->data.ptr + dABdB->step*i);
                const float* a = (const float*)(A->data.ptr + A->step*i1);

                for( j = 0; j < L*N; j++ )
                    dcdb[j] = 0;
                for( j = 0; j < L; j++ )
                    dcdb[j*N + i2] = a[j];
            }
        }
    }
    else
    {
        for( i = 0; i < M*N; i++ )
        {
            int i1 = i / N,  i2 = i % N;

            if( dABdA )
            {
                double* dcda = (double*)(dABdA->data.ptr + dABdA->step*i);
                const double* b = (const double*)B->data.ptr + i2;

                for( j = 0; j < M*L; j++ )
                    dcda[j] = 0;
                for( j = 0; j < L; j++ )
                    dcda[i1*L + j] = b[j*bstep];
            }

            if( dABdB )
            {
                double* dcdb = (double*)(dABdB->data.ptr + dABdB->step*i);
                const double* a = (const double*)(A->data.ptr + A->step*i1);

                for( j = 0; j < L*N; j++ )
                    dcdb[j] = 0;
                for( j = 0; j < L; j++ )
                    dcdb[j*N + i2] = a[j];
            }
        }
    }
}

static int cvRodrigues2( const CvMat* src, CvMat* dst, CvMat* jacobian=0 )
{
    double J[27] = {0};
    CvMat matJ = cvMat( 3, 9, CV_64F, J );

    if( !CV_IS_MAT(src) )
        CV_Error( !src ? CV_StsNullPtr : CV_StsBadArg, "Input argument is not a valid matrix" );

    if( !CV_IS_MAT(dst) )
        CV_Error( !dst ? CV_StsNullPtr : CV_StsBadArg,
        "The first output argument is not a valid matrix" );

    int depth = CV_MAT_DEPTH(src->type);
    int elem_size = CV_ELEM_SIZE(depth);

    if( depth != CV_32F && depth != CV_64F )
        CV_Error( CV_StsUnsupportedFormat, "The matrices must have 32f or 64f data type" );

    if( !CV_ARE_DEPTHS_EQ(src, dst) )
        CV_Error( CV_StsUnmatchedFormats, "All the matrices must have the same data type" );

    if( jacobian )
    {
        if( !CV_IS_MAT(jacobian) )
            CV_Error( CV_StsBadArg, "Jacobian is not a valid matrix" );

        if( !CV_ARE_DEPTHS_EQ(src, jacobian) || CV_MAT_CN(jacobian->type) != 1 )
            CV_Error( CV_StsUnmatchedFormats, "Jacobian must have 32fC1 or 64fC1 datatype" );

        if( (jacobian->rows != 9 || jacobian->cols != 3) &&
            (jacobian->rows != 3 || jacobian->cols != 9))
            CV_Error( CV_StsBadSize, "Jacobian must be 3x9 or 9x3" );
    }

    if( src->cols == 1 || src->rows == 1 )
    {
        int step = src->rows > 1 ? src->step / elem_size : 1;

        if( src->rows + src->cols*CV_MAT_CN(src->type) - 1 != 3 )
            CV_Error( CV_StsBadSize, "Input matrix must be 1x3, 3x1 or 3x3" );

        if( dst->rows != 3 || dst->cols != 3 || CV_MAT_CN(dst->type) != 1 )
            CV_Error( CV_StsBadSize, "Output matrix must be 3x3, single-channel floating point matrix" );

        Point3d r;
        if( depth == CV_32F )
        {
            r.x = src->data.fl[0];
            r.y = src->data.fl[step];
            r.z = src->data.fl[step*2];
        }
        else
        {
            r.x = src->data.db[0];
            r.y = src->data.db[step];
            r.z = src->data.db[step*2];
        }

        double theta = norm(r);

        if( theta < DBL_EPSILON )
        {
            cvSetIdentity( dst );

            if( jacobian )
            {
                memset( J, 0, sizeof(J) );
                J[5] = J[15] = J[19] = -1;
                J[7] = J[11] = J[21] = 1;
            }
        }
        else
        {
            double c = cos(theta);
            double s = sin(theta);
            double c1 = 1. - c;
            double itheta = theta ? 1./theta : 0.;

            r *= itheta;

            Matx33d rrt( r.x*r.x, r.x*r.y, r.x*r.z, r.x*r.y, r.y*r.y, r.y*r.z, r.x*r.z, r.y*r.z, r.z*r.z );
            Matx33d r_x(    0, -r.z,  r.y,
                          r.z,    0, -r.x,
                         -r.y,  r.x,    0 );

            // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
            Matx33d R = c*Matx33d::eye() + c1*rrt + s*r_x;

            Mat(R).convertTo(cvarrToMat(dst), dst->type);

            if( jacobian )
            {
                const double I[] = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
                double drrt[] = { r.x+r.x, r.y, r.z, r.y, 0, 0, r.z, 0, 0,
                                  0, r.x, 0, r.x, r.y+r.y, r.z, 0, r.z, 0,
                                  0, 0, r.x, 0, 0, r.y, r.x, r.y, r.z+r.z };
                double d_r_x_[] = { 0, 0, 0, 0, 0, -1, 0, 1, 0,
                                    0, 0, 1, 0, 0, 0, -1, 0, 0,
                                    0, -1, 0, 1, 0, 0, 0, 0, 0 };
                for( int i = 0; i < 3; i++ )
                {
                    double ri = i == 0 ? r.x : i == 1 ? r.y : r.z;
                    double a0 = -s*ri, a1 = (s - 2*c1*itheta)*ri, a2 = c1*itheta;
                    double a3 = (c - s*itheta)*ri, a4 = s*itheta;
                    for( int k = 0; k < 9; k++ )
                        J[i*9+k] = a0*I[k] + a1*rrt.val[k] + a2*drrt[i*9+k] +
                                   a3*r_x.val[k] + a4*d_r_x_[i*9+k];
                }
            }
        }
    }
    else if( src->cols == 3 && src->rows == 3 )
    {
        Matx33d U, Vt;
        Vec3d W;
        double theta, s, c;
        int step = dst->rows > 1 ? dst->step / elem_size : 1;

        if( (dst->rows != 1 || dst->cols*CV_MAT_CN(dst->type) != 3) &&
            (dst->rows != 3 || dst->cols != 1 || CV_MAT_CN(dst->type) != 1))
            CV_Error( CV_StsBadSize, "Output matrix must be 1x3 or 3x1" );

        Matx33d R = cvarrToMat(src);

        if( !checkRange(R, true, NULL, -100, 100) )
        {
            cvZero(dst);
            if( jacobian )
                cvZero(jacobian);
            return 0;
        }

        SVD::compute(R, W, U, Vt);
        R = U*Vt;

        Point3d r(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));

        s = std::sqrt((r.x*r.x + r.y*r.y + r.z*r.z)*0.25);
        c = (R(0, 0) + R(1, 1) + R(2, 2) - 1)*0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;
        theta = acos(c);

        if( s < 1e-5 )
        {
            double t;

            if( c > 0 )
                r = Point3d(0, 0, 0);
            else
            {
                t = (R(0, 0) + 1)*0.5;
                r.x = std::sqrt(MAX(t,0.));
                t = (R(1, 1) + 1)*0.5;
                r.y = std::sqrt(MAX(t,0.))*(R(0, 1) < 0 ? -1. : 1.);
                t = (R(2, 2) + 1)*0.5;
                r.z = std::sqrt(MAX(t,0.))*(R(0, 2) < 0 ? -1. : 1.);
                if( fabs(r.x) < fabs(r.y) && fabs(r.x) < fabs(r.z) && (R(1, 2) > 0) != (r.y*r.z > 0) )
                    r.z = -r.z;
                theta /= norm(r);
                r *= theta;
            }

            if( jacobian )
            {
                memset( J, 0, sizeof(J) );
                if( c > 0 )
                {
                    J[5] = J[15] = J[19] = -0.5;
                    J[7] = J[11] = J[21] = 0.5;
                }
            }
        }
        else
        {
            double vth = 1/(2*s);

            if( jacobian )
            {
                double t, dtheta_dtr = -1./s;
                // var1 = [vth;theta]
                // var = [om1;var1] = [om1;vth;theta]
                double dvth_dtheta = -vth*c/s;
                double d1 = 0.5*dvth_dtheta*dtheta_dtr;
                double d2 = 0.5*dtheta_dtr;
                // dvar1/dR = dvar1/dtheta*dtheta/dR = [dvth/dtheta; 1] * dtheta/dtr * dtr/dR
                double dvardR[5*9] =
                {
                    0, 0, 0, 0, 0, 1, 0, -1, 0,
                    0, 0, -1, 0, 0, 0, 1, 0, 0,
                    0, 1, 0, -1, 0, 0, 0, 0, 0,
                    d1, 0, 0, 0, d1, 0, 0, 0, d1,
                    d2, 0, 0, 0, d2, 0, 0, 0, d2
                };
                // var2 = [om;theta]
                double dvar2dvar[] =
                {
                    vth, 0, 0, r.x, 0,
                    0, vth, 0, r.y, 0,
                    0, 0, vth, r.z, 0,
                    0, 0, 0, 0, 1
                };
                double domegadvar2[] =
                {
                    theta, 0, 0, r.x*vth,
                    0, theta, 0, r.y*vth,
                    0, 0, theta, r.z*vth
                };

                CvMat _dvardR = cvMat( 5, 9, CV_64FC1, dvardR );
                CvMat _dvar2dvar = cvMat( 4, 5, CV_64FC1, dvar2dvar );
                CvMat _domegadvar2 = cvMat( 3, 4, CV_64FC1, domegadvar2 );
                double t0[3*5];
                CvMat _t0 = cvMat( 3, 5, CV_64FC1, t0 );

                cvMatMul( &_domegadvar2, &_dvar2dvar, &_t0 );
                cvMatMul( &_t0, &_dvardR, &matJ );

                // transpose every row of matJ (treat the rows as 3x3 matrices)
                CV_SWAP(J[1], J[3], t); CV_SWAP(J[2], J[6], t); CV_SWAP(J[5], J[7], t);
                CV_SWAP(J[10], J[12], t); CV_SWAP(J[11], J[15], t); CV_SWAP(J[14], J[16], t);
                CV_SWAP(J[19], J[21], t); CV_SWAP(J[20], J[24], t); CV_SWAP(J[23], J[25], t);
            }

            vth *= theta;
            r *= vth;
        }

        if( depth == CV_32F )
        {
            dst->data.fl[0] = (float)r.x;
            dst->data.fl[step] = (float)r.y;
            dst->data.fl[step*2] = (float)r.z;
        }
        else
        {
            dst->data.db[0] = r.x;
            dst->data.db[step] = r.y;
            dst->data.db[step*2] = r.z;
        }
    }
    else
    {
        CV_Error(CV_StsBadSize, "Input matrix must be 1x3 or 3x1 for a rotation vector, or 3x3 for a rotation matrix");
    }

    if( jacobian )
    {
        if( depth == CV_32F )
        {
            if( jacobian->rows == matJ.rows )
                cvConvert( &matJ, jacobian );
            else
            {
                float Jf[3*9];
                CvMat _Jf = cvMat( matJ.rows, matJ.cols, CV_32FC1, Jf );
                cvConvert( &matJ, &_Jf );
                cvTranspose( &_Jf, jacobian );
            }
        }
        else if( jacobian->rows == matJ.rows )
            cvCopy( &matJ, jacobian );
        else
            cvTranspose( &matJ, jacobian );
    }

    return 1;
}

// reimplementation of compose_motion.m
static void cvComposeRT( const CvMat* _rvec1, const CvMat* _tvec1,
             const CvMat* _rvec2, const CvMat* _tvec2,
             CvMat* _rvec3, CvMat* _tvec3,
             CvMat* dr3dr1, CvMat* dr3dt1,
             CvMat* dr3dr2, CvMat* dr3dt2,
             CvMat* dt3dr1, CvMat* dt3dt1,
             CvMat* dt3dr2, CvMat* dt3dt2 )
{
    double _r1[3], _r2[3];
    double _R1[9], _d1[9*3], _R2[9], _d2[9*3];
    CvMat r1 = cvMat(3,1,CV_64F,_r1), r2 = cvMat(3,1,CV_64F,_r2);
    CvMat R1 = cvMat(3,3,CV_64F,_R1), R2 = cvMat(3,3,CV_64F,_R2);
    CvMat dR1dr1 = cvMat(9,3,CV_64F,_d1), dR2dr2 = cvMat(9,3,CV_64F,_d2);

    CV_Assert( CV_IS_MAT(_rvec1) && CV_IS_MAT(_rvec2) );

    CV_Assert( CV_MAT_TYPE(_rvec1->type) == CV_32F ||
               CV_MAT_TYPE(_rvec1->type) == CV_64F );

    CV_Assert( _rvec1->rows == 3 && _rvec1->cols == 1 && CV_ARE_SIZES_EQ(_rvec1, _rvec2) );

    cvConvert( _rvec1, &r1 );
    cvConvert( _rvec2, &r2 );

    cvRodrigues2( &r1, &R1, &dR1dr1 );
    cvRodrigues2( &r2, &R2, &dR2dr2 );

    if( _rvec3 || dr3dr1 || dr3dr2 )
    {
        double _r3[3], _R3[9], _dR3dR1[9*9], _dR3dR2[9*9], _dr3dR3[9*3];
        double _W1[9*3], _W2[3*3];
        CvMat r3 = cvMat(3,1,CV_64F,_r3), R3 = cvMat(3,3,CV_64F,_R3);
        CvMat dR3dR1 = cvMat(9,9,CV_64F,_dR3dR1), dR3dR2 = cvMat(9,9,CV_64F,_dR3dR2);
        CvMat dr3dR3 = cvMat(3,9,CV_64F,_dr3dR3);
        CvMat W1 = cvMat(3,9,CV_64F,_W1), W2 = cvMat(3,3,CV_64F,_W2);

        cvMatMul( &R2, &R1, &R3 );
        cvCalcMatMulDeriv( &R2, &R1, &dR3dR2, &dR3dR1 );

        cvRodrigues2( &R3, &r3, &dr3dR3 );

        if( _rvec3 )
            cvConvert( &r3, _rvec3 );

        if( dr3dr1 )
        {
            cvMatMul( &dr3dR3, &dR3dR1, &W1 );
            cvMatMul( &W1, &dR1dr1, &W2 );
            cvConvert( &W2, dr3dr1 );
        }

        if( dr3dr2 )
        {
            cvMatMul( &dr3dR3, &dR3dR2, &W1 );
            cvMatMul( &W1, &dR2dr2, &W2 );
            cvConvert( &W2, dr3dr2 );
        }
    }

    if( dr3dt1 )
        cvZero( dr3dt1 );
    if( dr3dt2 )
        cvZero( dr3dt2 );

    if( _tvec3 || dt3dr2 || dt3dt1 )
    {
        double _t1[3], _t2[3], _t3[3], _dxdR2[3*9], _dxdt1[3*3], _W3[3*3];
        CvMat t1 = cvMat(3,1,CV_64F,_t1), t2 = cvMat(3,1,CV_64F,_t2);
        CvMat t3 = cvMat(3,1,CV_64F,_t3);
        CvMat dxdR2 = cvMat(3, 9, CV_64F, _dxdR2);
        CvMat dxdt1 = cvMat(3, 3, CV_64F, _dxdt1);
        CvMat W3 = cvMat(3, 3, CV_64F, _W3);

        CV_Assert( CV_IS_MAT(_tvec1) && CV_IS_MAT(_tvec2) );
        CV_Assert( CV_ARE_SIZES_EQ(_tvec1, _tvec2) && CV_ARE_SIZES_EQ(_tvec1, _rvec1) );

        cvConvert( _tvec1, &t1 );
        cvConvert( _tvec2, &t2 );
        cvMatMulAdd( &R2, &t1, &t2, &t3 );

        if( _tvec3 )
            cvConvert( &t3, _tvec3 );

        if( dt3dr2 || dt3dt1 )
        {
            cvCalcMatMulDeriv( &R2, &t1, &dxdR2, &dxdt1 );
            if( dt3dr2 )
            {
                cvMatMul( &dxdR2, &dR2dr2, &W3 );
                cvConvert( &W3, dt3dr2 );
            }
            if( dt3dt1 )
                cvConvert( &dxdt1, dt3dt1 );
        }
    }

    if( dt3dt2 )
        cvSetIdentity( dt3dt2 );
    if( dt3dr1 )
        cvZero( dt3dr1 );
}

static const char* cvDistCoeffErr = "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8, 8x1, 1x12, 12x1, 1x14 or 14x1 floating-point vector";

static void cvProjectPoints2Internal( const CvMat* objectPoints,
                  const CvMat* r_vec,
                  const CvMat* t_vec,
                  const CvMat* A,
                  const CvMat* distCoeffs,
                  CvMat* imagePoints, CvMat* dpdr CV_DEFAULT(NULL),
                  CvMat* dpdt CV_DEFAULT(NULL), CvMat* dpdf CV_DEFAULT(NULL),
                  CvMat* dpdc CV_DEFAULT(NULL), CvMat* dpdk CV_DEFAULT(NULL),
                  CvMat* dpdo CV_DEFAULT(NULL),
                  double aspectRatio CV_DEFAULT(0) )
{
    Ptr<CvMat> matM, _m;
    Ptr<CvMat> _dpdr, _dpdt, _dpdc, _dpdf, _dpdk;
    Ptr<CvMat> _dpdo;

    int i, j, count;
    int calc_derivatives;
    const CvPoint3D64f* M;
    CvPoint2D64f* m;
    double r[3], R[9], dRdr[27], t[3], a[9], k[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0}, fx, fy, cx, cy;
    Matx33d matTilt = Matx33d::eye();
    Matx33d dMatTiltdTauX(0,0,0,0,0,0,0,-1,0);
    Matx33d dMatTiltdTauY(0,0,0,0,0,0,1,0,0);
    CvMat _r, _t, _a = cvMat( 3, 3, CV_64F, a ), _k;
    CvMat matR = cvMat( 3, 3, CV_64F, R ), _dRdr = cvMat( 3, 9, CV_64F, dRdr );
    double *dpdr_p = 0, *dpdt_p = 0, *dpdk_p = 0, *dpdf_p = 0, *dpdc_p = 0;
    double* dpdo_p = 0;
    int dpdr_step = 0, dpdt_step = 0, dpdk_step = 0, dpdf_step = 0, dpdc_step = 0;
    int dpdo_step = 0;
    bool fixedAspectRatio = aspectRatio > FLT_EPSILON;

    if( !CV_IS_MAT(objectPoints) || !CV_IS_MAT(r_vec) ||
        !CV_IS_MAT(t_vec) || !CV_IS_MAT(A) ||
        /*!CV_IS_MAT(distCoeffs) ||*/ !CV_IS_MAT(imagePoints) )
        CV_Error( CV_StsBadArg, "One of required arguments is not a valid matrix" );

    int total = objectPoints->rows * objectPoints->cols * CV_MAT_CN(objectPoints->type);
    if(total % 3 != 0)
    {
        //we have stopped support of homogeneous coordinates because it cause ambiguity in interpretation of the input data
        CV_Error( CV_StsBadArg, "Homogeneous coordinates are not supported" );
    }
    count = total / 3;

    if( CV_IS_CONT_MAT(objectPoints->type) &&
        (CV_MAT_DEPTH(objectPoints->type) == CV_32F || CV_MAT_DEPTH(objectPoints->type) == CV_64F)&&
        ((objectPoints->rows == 1 && CV_MAT_CN(objectPoints->type) == 3) ||
        (objectPoints->rows == count && CV_MAT_CN(objectPoints->type)*objectPoints->cols == 3) ||
        (objectPoints->rows == 3 && CV_MAT_CN(objectPoints->type) == 1 && objectPoints->cols == count)))
    {
        matM.reset(cvCreateMat( objectPoints->rows, objectPoints->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(objectPoints->type)) ));
        cvConvert(objectPoints, matM);
    }
    else
    {
//        matM = cvCreateMat( 1, count, CV_64FC3 );
//        cvConvertPointsHomogeneous( objectPoints, matM );
        CV_Error( CV_StsBadArg, "Homogeneous coordinates are not supported" );
    }

    if( CV_IS_CONT_MAT(imagePoints->type) &&
        (CV_MAT_DEPTH(imagePoints->type) == CV_32F || CV_MAT_DEPTH(imagePoints->type) == CV_64F) &&
        ((imagePoints->rows == 1 && CV_MAT_CN(imagePoints->type) == 2) ||
        (imagePoints->rows == count && CV_MAT_CN(imagePoints->type)*imagePoints->cols == 2) ||
        (imagePoints->rows == 2 && CV_MAT_CN(imagePoints->type) == 1 && imagePoints->cols == count)))
    {
        _m.reset(cvCreateMat( imagePoints->rows, imagePoints->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(imagePoints->type)) ));
        cvConvert(imagePoints, _m);
    }
    else
    {
//        _m = cvCreateMat( 1, count, CV_64FC2 );
        CV_Error( CV_StsBadArg, "Homogeneous coordinates are not supported" );
    }

    M = (CvPoint3D64f*)matM->data.db;
    m = (CvPoint2D64f*)_m->data.db;

    if( (CV_MAT_DEPTH(r_vec->type) != CV_64F && CV_MAT_DEPTH(r_vec->type) != CV_32F) ||
        (((r_vec->rows != 1 && r_vec->cols != 1) ||
        r_vec->rows*r_vec->cols*CV_MAT_CN(r_vec->type) != 3) &&
        ((r_vec->rows != 3 && r_vec->cols != 3) || CV_MAT_CN(r_vec->type) != 1)))
        CV_Error( CV_StsBadArg, "Rotation must be represented by 1x3 or 3x1 "
                  "floating-point rotation vector, or 3x3 rotation matrix" );

    if( r_vec->rows == 3 && r_vec->cols == 3 )
    {
        _r = cvMat( 3, 1, CV_64FC1, r );
        cvRodrigues2( r_vec, &_r );
        cvRodrigues2( &_r, &matR, &_dRdr );
        cvCopy( r_vec, &matR );
    }
    else
    {
        _r = cvMat( r_vec->rows, r_vec->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(r_vec->type)), r );
        cvConvert( r_vec, &_r );
        cvRodrigues2( &_r, &matR, &_dRdr );
    }

    if( (CV_MAT_DEPTH(t_vec->type) != CV_64F && CV_MAT_DEPTH(t_vec->type) != CV_32F) ||
        (t_vec->rows != 1 && t_vec->cols != 1) ||
        t_vec->rows*t_vec->cols*CV_MAT_CN(t_vec->type) != 3 )
        CV_Error( CV_StsBadArg,
            "Translation vector must be 1x3 or 3x1 floating-point vector" );

    _t = cvMat( t_vec->rows, t_vec->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(t_vec->type)), t );
    cvConvert( t_vec, &_t );

    if( (CV_MAT_TYPE(A->type) != CV_64FC1 && CV_MAT_TYPE(A->type) != CV_32FC1) ||
        A->rows != 3 || A->cols != 3 )
        CV_Error( CV_StsBadArg, "Intrinsic parameters must be 3x3 floating-point matrix" );

    cvConvert( A, &_a );
    fx = a[0]; fy = a[4];
    cx = a[2]; cy = a[5];

    if( fixedAspectRatio )
        fx = fy*aspectRatio;

    if( distCoeffs )
    {
        if( !CV_IS_MAT(distCoeffs) ||
            (CV_MAT_DEPTH(distCoeffs->type) != CV_64F &&
            CV_MAT_DEPTH(distCoeffs->type) != CV_32F) ||
            (distCoeffs->rows != 1 && distCoeffs->cols != 1) ||
            (distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) != 4 &&
            distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) != 5 &&
            distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) != 8 &&
            distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) != 12 &&
            distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) != 14) )
            CV_Error( CV_StsBadArg, cvDistCoeffErr );

        _k = cvMat( distCoeffs->rows, distCoeffs->cols,
                    CV_MAKETYPE(CV_64F,CV_MAT_CN(distCoeffs->type)), k );
        cvConvert( distCoeffs, &_k );
        if(k[12] != 0 || k[13] != 0)
        {
            cv::_3d::computeTiltProjectionMatrix(k[12], k[13],
                                        &matTilt, &dMatTiltdTauX, &dMatTiltdTauY);
        }
    }

    if( dpdr )
    {
        if( !CV_IS_MAT(dpdr) ||
            (CV_MAT_TYPE(dpdr->type) != CV_32FC1 &&
            CV_MAT_TYPE(dpdr->type) != CV_64FC1) ||
            dpdr->rows != count*2 || dpdr->cols != 3 )
            CV_Error( CV_StsBadArg, "dp/drot must be 2Nx3 floating-point matrix" );

        if( CV_MAT_TYPE(dpdr->type) == CV_64FC1 )
        {
            _dpdr.reset(cvCloneMat(dpdr));
        }
        else
            _dpdr.reset(cvCreateMat( 2*count, 3, CV_64FC1 ));
        dpdr_p = _dpdr->data.db;
        dpdr_step = _dpdr->step/sizeof(dpdr_p[0]);
    }

    if( dpdt )
    {
        if( !CV_IS_MAT(dpdt) ||
            (CV_MAT_TYPE(dpdt->type) != CV_32FC1 &&
            CV_MAT_TYPE(dpdt->type) != CV_64FC1) ||
            dpdt->rows != count*2 || dpdt->cols != 3 )
            CV_Error( CV_StsBadArg, "dp/dT must be 2Nx3 floating-point matrix" );

        if( CV_MAT_TYPE(dpdt->type) == CV_64FC1 )
        {
            _dpdt.reset(cvCloneMat(dpdt));
        }
        else
            _dpdt.reset(cvCreateMat( 2*count, 3, CV_64FC1 ));
        dpdt_p = _dpdt->data.db;
        dpdt_step = _dpdt->step/sizeof(dpdt_p[0]);
    }

    if( dpdf )
    {
        if( !CV_IS_MAT(dpdf) ||
            (CV_MAT_TYPE(dpdf->type) != CV_32FC1 && CV_MAT_TYPE(dpdf->type) != CV_64FC1) ||
            dpdf->rows != count*2 || dpdf->cols != 2 )
            CV_Error( CV_StsBadArg, "dp/df must be 2Nx2 floating-point matrix" );

        if( CV_MAT_TYPE(dpdf->type) == CV_64FC1 )
        {
            _dpdf.reset(cvCloneMat(dpdf));
        }
        else
            _dpdf.reset(cvCreateMat( 2*count, 2, CV_64FC1 ));
        dpdf_p = _dpdf->data.db;
        dpdf_step = _dpdf->step/sizeof(dpdf_p[0]);
    }

    if( dpdc )
    {
        if( !CV_IS_MAT(dpdc) ||
            (CV_MAT_TYPE(dpdc->type) != CV_32FC1 && CV_MAT_TYPE(dpdc->type) != CV_64FC1) ||
            dpdc->rows != count*2 || dpdc->cols != 2 )
            CV_Error( CV_StsBadArg, "dp/dc must be 2Nx2 floating-point matrix" );

        if( CV_MAT_TYPE(dpdc->type) == CV_64FC1 )
        {
            _dpdc.reset(cvCloneMat(dpdc));
        }
        else
            _dpdc.reset(cvCreateMat( 2*count, 2, CV_64FC1 ));
        dpdc_p = _dpdc->data.db;
        dpdc_step = _dpdc->step/sizeof(dpdc_p[0]);
    }

    if( dpdk )
    {
        if( !CV_IS_MAT(dpdk) ||
            (CV_MAT_TYPE(dpdk->type) != CV_32FC1 && CV_MAT_TYPE(dpdk->type) != CV_64FC1) ||
            dpdk->rows != count*2 || (dpdk->cols != 14 && dpdk->cols != 12 && dpdk->cols != 8 && dpdk->cols != 5 && dpdk->cols != 4 && dpdk->cols != 2) )
            CV_Error( CV_StsBadArg, "dp/df must be 2Nx14, 2Nx12, 2Nx8, 2Nx5, 2Nx4 or 2Nx2 floating-point matrix" );

        if( !distCoeffs )
            CV_Error( CV_StsNullPtr, "distCoeffs is NULL while dpdk is not" );

        if( CV_MAT_TYPE(dpdk->type) == CV_64FC1 )
        {
            _dpdk.reset(cvCloneMat(dpdk));
        }
        else
            _dpdk.reset(cvCreateMat( dpdk->rows, dpdk->cols, CV_64FC1 ));
        dpdk_p = _dpdk->data.db;
        dpdk_step = _dpdk->step/sizeof(dpdk_p[0]);
    }

    if( dpdo )
    {
        if( !CV_IS_MAT( dpdo ) || ( CV_MAT_TYPE( dpdo->type ) != CV_32FC1
                                    && CV_MAT_TYPE( dpdo->type ) != CV_64FC1 )
            || dpdo->rows != count * 2 || dpdo->cols != count * 3 )
            CV_Error( CV_StsBadArg, "dp/do must be 2Nx3N floating-point matrix" );

        if( CV_MAT_TYPE( dpdo->type ) == CV_64FC1 )
        {
            _dpdo.reset( cvCloneMat( dpdo ) );
        }
        else
            _dpdo.reset( cvCreateMat( 2 * count, 3 * count, CV_64FC1 ) );
        cvZero(_dpdo);
        dpdo_p = _dpdo->data.db;
        dpdo_step = _dpdo->step / sizeof( dpdo_p[0] );
    }

    calc_derivatives = dpdr || dpdt || dpdf || dpdc || dpdk || dpdo;

    for( i = 0; i < count; i++ )
    {
        double X = M[i].x, Y = M[i].y, Z = M[i].z;
        double x = R[0]*X + R[1]*Y + R[2]*Z + t[0];
        double y = R[3]*X + R[4]*Y + R[5]*Z + t[1];
        double z = R[6]*X + R[7]*Y + R[8]*Z + t[2];
        double r2, r4, r6, a1, a2, a3, cdist, icdist2;
        double xd, yd, xd0, yd0, invProj;
        Vec3d vecTilt;
        Vec3d dVecTilt;
        Matx22d dMatTilt;
        Vec2d dXdYd;

        double z0 = z;
        z = z ? 1./z : 1;
        x *= z; y *= z;

        r2 = x*x + y*y;
        r4 = r2*r2;
        r6 = r4*r2;
        a1 = 2*x*y;
        a2 = r2 + 2*x*x;
        a3 = r2 + 2*y*y;
        cdist = 1 + k[0]*r2 + k[1]*r4 + k[4]*r6;
        icdist2 = 1./(1 + k[5]*r2 + k[6]*r4 + k[7]*r6);
        xd0 = x*cdist*icdist2 + k[2]*a1 + k[3]*a2 + k[8]*r2+k[9]*r4;
        yd0 = y*cdist*icdist2 + k[2]*a3 + k[3]*a1 + k[10]*r2+k[11]*r4;

        // additional distortion by projecting onto a tilt plane
        vecTilt = matTilt*Vec3d(xd0, yd0, 1);
        invProj = vecTilt(2) ? 1./vecTilt(2) : 1;
        xd = invProj * vecTilt(0);
        yd = invProj * vecTilt(1);

        m[i].x = xd*fx + cx;
        m[i].y = yd*fy + cy;

        if( calc_derivatives )
        {
            if( dpdc_p )
            {
                dpdc_p[0] = 1; dpdc_p[1] = 0; // dp_xdc_x; dp_xdc_y
                dpdc_p[dpdc_step] = 0;
                dpdc_p[dpdc_step+1] = 1;
                dpdc_p += dpdc_step*2;
            }

            if( dpdf_p )
            {
                if( fixedAspectRatio )
                {
                    dpdf_p[0] = 0; dpdf_p[1] = xd*aspectRatio; // dp_xdf_x; dp_xdf_y
                    dpdf_p[dpdf_step] = 0;
                    dpdf_p[dpdf_step+1] = yd;
                }
                else
                {
                    dpdf_p[0] = xd; dpdf_p[1] = 0;
                    dpdf_p[dpdf_step] = 0;
                    dpdf_p[dpdf_step+1] = yd;
                }
                dpdf_p += dpdf_step*2;
            }
            for (int row = 0; row < 2; ++row)
                for (int col = 0; col < 2; ++col)
                    dMatTilt(row,col) = matTilt(row,col)*vecTilt(2)
                      - matTilt(2,col)*vecTilt(row);
            double invProjSquare = (invProj*invProj);
            dMatTilt *= invProjSquare;
            if( dpdk_p )
            {
                dXdYd = dMatTilt*Vec2d(x*icdist2*r2, y*icdist2*r2);
                dpdk_p[0] = fx*dXdYd(0);
                dpdk_p[dpdk_step] = fy*dXdYd(1);
                dXdYd = dMatTilt*Vec2d(x*icdist2*r4, y*icdist2*r4);
                dpdk_p[1] = fx*dXdYd(0);
                dpdk_p[dpdk_step+1] = fy*dXdYd(1);
                if( _dpdk->cols > 2 )
                {
                    dXdYd = dMatTilt*Vec2d(a1, a3);
                    dpdk_p[2] = fx*dXdYd(0);
                    dpdk_p[dpdk_step+2] = fy*dXdYd(1);
                    dXdYd = dMatTilt*Vec2d(a2, a1);
                    dpdk_p[3] = fx*dXdYd(0);
                    dpdk_p[dpdk_step+3] = fy*dXdYd(1);
                    if( _dpdk->cols > 4 )
                    {
                        dXdYd = dMatTilt*Vec2d(x*icdist2*r6, y*icdist2*r6);
                        dpdk_p[4] = fx*dXdYd(0);
                        dpdk_p[dpdk_step+4] = fy*dXdYd(1);

                        if( _dpdk->cols > 5 )
                        {
                            dXdYd = dMatTilt*Vec2d(
                              x*cdist*(-icdist2)*icdist2*r2, y*cdist*(-icdist2)*icdist2*r2);
                            dpdk_p[5] = fx*dXdYd(0);
                            dpdk_p[dpdk_step+5] = fy*dXdYd(1);
                            dXdYd = dMatTilt*Vec2d(
                              x*cdist*(-icdist2)*icdist2*r4, y*cdist*(-icdist2)*icdist2*r4);
                            dpdk_p[6] = fx*dXdYd(0);
                            dpdk_p[dpdk_step+6] = fy*dXdYd(1);
                            dXdYd = dMatTilt*Vec2d(
                              x*cdist*(-icdist2)*icdist2*r6, y*cdist*(-icdist2)*icdist2*r6);
                            dpdk_p[7] = fx*dXdYd(0);
                            dpdk_p[dpdk_step+7] = fy*dXdYd(1);
                            if( _dpdk->cols > 8 )
                            {
                                dXdYd = dMatTilt*Vec2d(r2, 0);
                                dpdk_p[8] = fx*dXdYd(0); //s1
                                dpdk_p[dpdk_step+8] = fy*dXdYd(1); //s1
                                dXdYd = dMatTilt*Vec2d(r4, 0);
                                dpdk_p[9] = fx*dXdYd(0); //s2
                                dpdk_p[dpdk_step+9] = fy*dXdYd(1); //s2
                                dXdYd = dMatTilt*Vec2d(0, r2);
                                dpdk_p[10] = fx*dXdYd(0);//s3
                                dpdk_p[dpdk_step+10] = fy*dXdYd(1); //s3
                                dXdYd = dMatTilt*Vec2d(0, r4);
                                dpdk_p[11] = fx*dXdYd(0);//s4
                                dpdk_p[dpdk_step+11] = fy*dXdYd(1); //s4
                                if( _dpdk->cols > 12 )
                                {
                                    dVecTilt = dMatTiltdTauX * Vec3d(xd0, yd0, 1);
                                    dpdk_p[12] = fx * invProjSquare * (
                                      dVecTilt(0) * vecTilt(2) - dVecTilt(2) * vecTilt(0));
                                    dpdk_p[dpdk_step+12] = fy*invProjSquare * (
                                      dVecTilt(1) * vecTilt(2) - dVecTilt(2) * vecTilt(1));
                                    dVecTilt = dMatTiltdTauY * Vec3d(xd0, yd0, 1);
                                    dpdk_p[13] = fx * invProjSquare * (
                                      dVecTilt(0) * vecTilt(2) - dVecTilt(2) * vecTilt(0));
                                    dpdk_p[dpdk_step+13] = fy * invProjSquare * (
                                      dVecTilt(1) * vecTilt(2) - dVecTilt(2) * vecTilt(1));
                                }
                            }
                        }
                    }
                }
                dpdk_p += dpdk_step*2;
            }

            if( dpdt_p )
            {
                double dxdt[] = { z, 0, -x*z }, dydt[] = { 0, z, -y*z };
                for( j = 0; j < 3; j++ )
                {
                    double dr2dt = 2*x*dxdt[j] + 2*y*dydt[j];
                    double dcdist_dt = k[0]*dr2dt + 2*k[1]*r2*dr2dt + 3*k[4]*r4*dr2dt;
                    double dicdist2_dt = -icdist2*icdist2*(k[5]*dr2dt + 2*k[6]*r2*dr2dt + 3*k[7]*r4*dr2dt);
                    double da1dt = 2*(x*dydt[j] + y*dxdt[j]);
                    double dmxdt = (dxdt[j]*cdist*icdist2 + x*dcdist_dt*icdist2 + x*cdist*dicdist2_dt +
                                       k[2]*da1dt + k[3]*(dr2dt + 4*x*dxdt[j]) + k[8]*dr2dt + 2*r2*k[9]*dr2dt);
                    double dmydt = (dydt[j]*cdist*icdist2 + y*dcdist_dt*icdist2 + y*cdist*dicdist2_dt +
                                       k[2]*(dr2dt + 4*y*dydt[j]) + k[3]*da1dt + k[10]*dr2dt + 2*r2*k[11]*dr2dt);
                    dXdYd = dMatTilt*Vec2d(dmxdt, dmydt);
                    dpdt_p[j] = fx*dXdYd(0);
                    dpdt_p[dpdt_step+j] = fy*dXdYd(1);
                }
                dpdt_p += dpdt_step*2;
            }

            if( dpdr_p )
            {
                double dx0dr[] =
                {
                    X*dRdr[0] + Y*dRdr[1] + Z*dRdr[2],
                    X*dRdr[9] + Y*dRdr[10] + Z*dRdr[11],
                    X*dRdr[18] + Y*dRdr[19] + Z*dRdr[20]
                };
                double dy0dr[] =
                {
                    X*dRdr[3] + Y*dRdr[4] + Z*dRdr[5],
                    X*dRdr[12] + Y*dRdr[13] + Z*dRdr[14],
                    X*dRdr[21] + Y*dRdr[22] + Z*dRdr[23]
                };
                double dz0dr[] =
                {
                    X*dRdr[6] + Y*dRdr[7] + Z*dRdr[8],
                    X*dRdr[15] + Y*dRdr[16] + Z*dRdr[17],
                    X*dRdr[24] + Y*dRdr[25] + Z*dRdr[26]
                };
                for( j = 0; j < 3; j++ )
                {
                    double dxdr = z*(dx0dr[j] - x*dz0dr[j]);
                    double dydr = z*(dy0dr[j] - y*dz0dr[j]);
                    double dr2dr = 2*x*dxdr + 2*y*dydr;
                    double dcdist_dr = (k[0] + 2*k[1]*r2 + 3*k[4]*r4)*dr2dr;
                    double dicdist2_dr = -icdist2*icdist2*(k[5] + 2*k[6]*r2 + 3*k[7]*r4)*dr2dr;
                    double da1dr = 2*(x*dydr + y*dxdr);
                    double dmxdr = (dxdr*cdist*icdist2 + x*dcdist_dr*icdist2 + x*cdist*dicdist2_dr +
                                       k[2]*da1dr + k[3]*(dr2dr + 4*x*dxdr) + (k[8] + 2*r2*k[9])*dr2dr);
                    double dmydr = (dydr*cdist*icdist2 + y*dcdist_dr*icdist2 + y*cdist*dicdist2_dr +
                                       k[2]*(dr2dr + 4*y*dydr) + k[3]*da1dr + (k[10] + 2*r2*k[11])*dr2dr);
                    dXdYd = dMatTilt*Vec2d(dmxdr, dmydr);
                    dpdr_p[j] = fx*dXdYd(0);
                    dpdr_p[dpdr_step+j] = fy*dXdYd(1);
                }
                dpdr_p += dpdr_step*2;
            }

            if( dpdo_p )
            {
                double dxdo[] = { z * ( R[0] - x * z * z0 * R[6] ),
                                  z * ( R[1] - x * z * z0 * R[7] ),
                                  z * ( R[2] - x * z * z0 * R[8] ) };
                double dydo[] = { z * ( R[3] - y * z * z0 * R[6] ),
                                  z * ( R[4] - y * z * z0 * R[7] ),
                                  z * ( R[5] - y * z * z0 * R[8] ) };
                for( j = 0; j < 3; j++ )
                {
                    double dr2do = 2 * x * dxdo[j] + 2 * y * dydo[j];
                    double dr4do = 2 * r2 * dr2do;
                    double dr6do = 3 * r4 * dr2do;
                    double da1do = 2 * y * dxdo[j] + 2 * x * dydo[j];
                    double da2do = dr2do + 4 * x * dxdo[j];
                    double da3do = dr2do + 4 * y * dydo[j];
                    double dcdist_do
                        = k[0] * dr2do + k[1] * dr4do + k[4] * dr6do;
                    double dicdist2_do = -icdist2 * icdist2
                        * ( k[5] * dr2do + k[6] * dr4do + k[7] * dr6do );
                    double dxd0_do = cdist * icdist2 * dxdo[j]
                        + x * icdist2 * dcdist_do + x * cdist * dicdist2_do
                        + k[2] * da1do + k[3] * da2do + k[8] * dr2do
                        + k[9] * dr4do;
                    double dyd0_do = cdist * icdist2 * dydo[j]
                        + y * icdist2 * dcdist_do + y * cdist * dicdist2_do
                        + k[2] * da3do + k[3] * da1do + k[10] * dr2do
                        + k[11] * dr4do;
                    dXdYd = dMatTilt * Vec2d( dxd0_do, dyd0_do );
                    dpdo_p[i * 3 + j] = fx * dXdYd( 0 );
                    dpdo_p[dpdo_step + i * 3 + j] = fy * dXdYd( 1 );
                }
                dpdo_p += dpdo_step * 2;
            }
        }
    }

    if( _m != imagePoints )
        cvConvert( _m, imagePoints );

    if( _dpdr != dpdr )
        cvConvert( _dpdr, dpdr );

    if( _dpdt != dpdt )
        cvConvert( _dpdt, dpdt );

    if( _dpdf != dpdf )
        cvConvert( _dpdf, dpdf );

    if( _dpdc != dpdc )
        cvConvert( _dpdc, dpdc );

    if( _dpdk != dpdk )
        cvConvert( _dpdk, dpdk );

    if( _dpdo != dpdo )
        cvConvert( _dpdo, dpdo );
}

static void cvProjectPoints2( const CvMat* objectPoints,
                  const CvMat* r_vec,
                  const CvMat* t_vec,
                  const CvMat* A,
                  const CvMat* distCoeffs,
                  CvMat* imagePoints, CvMat* dpdr,
                  CvMat* dpdt, CvMat* dpdf,
                  CvMat* dpdc, CvMat* dpdk,
                  double aspectRatio )
{
    cvProjectPoints2Internal( objectPoints, r_vec, t_vec, A, distCoeffs, imagePoints, dpdr, dpdt,
                              dpdf, dpdc, dpdk, NULL, aspectRatio );
}

static void cvRQDecomp3x3( const CvMat *matrixM, CvMat *matrixR, CvMat *matrixQ,
               CvMat *matrixQx, CvMat *matrixQy, CvMat *matrixQz,
               CvPoint3D64f *eulerAngles)
{
    double matM[3][3], matR[3][3], matQ[3][3];
    CvMat M = cvMat(3, 3, CV_64F, matM);
    CvMat R = cvMat(3, 3, CV_64F, matR);
    CvMat Q = cvMat(3, 3, CV_64F, matQ);
    double z, c, s;

    /* Validate parameters. */
    CV_Assert( CV_IS_MAT(matrixM) && CV_IS_MAT(matrixR) && CV_IS_MAT(matrixQ) &&
        matrixM->cols == 3 && matrixM->rows == 3 &&
        CV_ARE_SIZES_EQ(matrixM, matrixR) && CV_ARE_SIZES_EQ(matrixM, matrixQ));

    cvConvert(matrixM, &M);

    /* Find Givens rotation Q_x for x axis (left multiplication). */
    /*
         ( 1  0  0 )
    Qx = ( 0  c  s ), c = m33/sqrt(m32^2 + m33^2), s = m32/sqrt(m32^2 + m33^2)
         ( 0 -s  c )
    */
    s = matM[2][1];
    c = matM[2][2];
    z = 1./std::sqrt(c * c + s * s + DBL_EPSILON);
    c *= z;
    s *= z;

    double _Qx[3][3] = { {1, 0, 0}, {0, c, s}, {0, -s, c} };
    CvMat Qx = cvMat(3, 3, CV_64F, _Qx);

    cvMatMul(&M, &Qx, &R);
    assert(fabs(matR[2][1]) < FLT_EPSILON);
    matR[2][1] = 0;

    /* Find Givens rotation for y axis. */
    /*
         ( c  0 -s )
    Qy = ( 0  1  0 ), c = m33/sqrt(m31^2 + m33^2), s = -m31/sqrt(m31^2 + m33^2)
         ( s  0  c )
    */
    s = -matR[2][0];
    c = matR[2][2];
    z = 1./std::sqrt(c * c + s * s + DBL_EPSILON);
    c *= z;
    s *= z;

    double _Qy[3][3] = { {c, 0, -s}, {0, 1, 0}, {s, 0, c} };
    CvMat Qy = cvMat(3, 3, CV_64F, _Qy);
    cvMatMul(&R, &Qy, &M);

    assert(fabs(matM[2][0]) < FLT_EPSILON);
    matM[2][0] = 0;

    /* Find Givens rotation for z axis. */
    /*
         ( c  s  0 )
    Qz = (-s  c  0 ), c = m22/sqrt(m21^2 + m22^2), s = m21/sqrt(m21^2 + m22^2)
         ( 0  0  1 )
    */

    s = matM[1][0];
    c = matM[1][1];
    z = 1./std::sqrt(c * c + s * s + DBL_EPSILON);
    c *= z;
    s *= z;

    double _Qz[3][3] = { {c, s, 0}, {-s, c, 0}, {0, 0, 1} };
    CvMat Qz = cvMat(3, 3, CV_64F, _Qz);

    cvMatMul(&M, &Qz, &R);
    assert(fabs(matR[1][0]) < FLT_EPSILON);
    matR[1][0] = 0;

    // Solve the decomposition ambiguity.
    // Diagonal entries of R, except the last one, shall be positive.
    // Further rotate R by 180 degree if necessary
    if( matR[0][0] < 0 )
    {
        if( matR[1][1] < 0 )
        {
            // rotate around z for 180 degree, i.e. a rotation matrix of
            // [-1,  0,  0],
            // [ 0, -1,  0],
            // [ 0,  0,  1]
            matR[0][0] *= -1;
            matR[0][1] *= -1;
            matR[1][1] *= -1;

            _Qz[0][0] *= -1;
            _Qz[0][1] *= -1;
            _Qz[1][0] *= -1;
            _Qz[1][1] *= -1;
        }
        else
        {
            // rotate around y for 180 degree, i.e. a rotation matrix of
            // [-1,  0,  0],
            // [ 0,  1,  0],
            // [ 0,  0, -1]
            matR[0][0] *= -1;
            matR[0][2] *= -1;
            matR[1][2] *= -1;
            matR[2][2] *= -1;

            cvTranspose( &Qz, &Qz );

            _Qy[0][0] *= -1;
            _Qy[0][2] *= -1;
            _Qy[2][0] *= -1;
            _Qy[2][2] *= -1;
        }
    }
    else if( matR[1][1] < 0 )
    {
        // ??? for some reason, we never get here ???

        // rotate around x for 180 degree, i.e. a rotation matrix of
        // [ 1,  0,  0],
        // [ 0, -1,  0],
        // [ 0,  0, -1]
        matR[0][1] *= -1;
        matR[0][2] *= -1;
        matR[1][1] *= -1;
        matR[1][2] *= -1;
        matR[2][2] *= -1;

        cvTranspose( &Qz, &Qz );
        cvTranspose( &Qy, &Qy );

        _Qx[1][1] *= -1;
        _Qx[1][2] *= -1;
        _Qx[2][1] *= -1;
        _Qx[2][2] *= -1;
    }

    // calculate the euler angle
    if( eulerAngles )
    {
        eulerAngles->x = acos(_Qx[1][1]) * (_Qx[1][2] >= 0 ? 1 : -1) * (180.0 / CV_PI);
        eulerAngles->y = acos(_Qy[0][0]) * (_Qy[2][0] >= 0 ? 1 : -1) * (180.0 / CV_PI);
        eulerAngles->z = acos(_Qz[0][0]) * (_Qz[0][1] >= 0 ? 1 : -1) * (180.0 / CV_PI);
    }

    /* Calculate orthogonal matrix. */
    /*
    Q = QzT * QyT * QxT
    */
    cvGEMM( &Qz, &Qy, 1, 0, 0, &M, CV_GEMM_A_T + CV_GEMM_B_T );
    cvGEMM( &M, &Qx, 1, 0, 0, &Q, CV_GEMM_B_T );

    /* Save R and Q matrices. */
    cvConvert( &R, matrixR );
    cvConvert( &Q, matrixQ );

    if( matrixQx )
        cvConvert(&Qx, matrixQx);
    if( matrixQy )
        cvConvert(&Qy, matrixQy);
    if( matrixQz )
        cvConvert(&Qz, matrixQz);
}


static void
cvDecomposeProjectionMatrix( const CvMat *projMatr, CvMat *calibMatr,
                             CvMat *rotMatr, CvMat *posVect,
                             CvMat *rotMatrX, CvMat *rotMatrY,
                             CvMat *rotMatrZ, CvPoint3D64f *eulerAngles)
{
    double tmpProjMatrData[16], tmpMatrixDData[16], tmpMatrixVData[16];
    CvMat tmpProjMatr = cvMat(4, 4, CV_64F, tmpProjMatrData);
    CvMat tmpMatrixD = cvMat(4, 4, CV_64F, tmpMatrixDData);
    CvMat tmpMatrixV = cvMat(4, 4, CV_64F, tmpMatrixVData);
    CvMat tmpMatrixM;

    /* Validate parameters. */
    if(projMatr == 0 || calibMatr == 0 || rotMatr == 0 || posVect == 0)
        CV_Error(CV_StsNullPtr, "Some of parameters is a NULL pointer!");

    if(!CV_IS_MAT(projMatr) || !CV_IS_MAT(calibMatr) || !CV_IS_MAT(rotMatr) || !CV_IS_MAT(posVect))
        CV_Error(CV_StsUnsupportedFormat, "Input parameters must be a matrices!");

    if(projMatr->cols != 4 || projMatr->rows != 3)
        CV_Error(CV_StsUnmatchedSizes, "Size of projection matrix must be 3x4!");

    if(calibMatr->cols != 3 || calibMatr->rows != 3 || rotMatr->cols != 3 || rotMatr->rows != 3)
        CV_Error(CV_StsUnmatchedSizes, "Size of calibration and rotation matrices must be 3x3!");

    if(posVect->cols != 1 || posVect->rows != 4)
        CV_Error(CV_StsUnmatchedSizes, "Size of position vector must be 4x1!");

    /* Compute position vector. */
    cvSetZero(&tmpProjMatr); // Add zero row to make matrix square.
    int i, k;
    for(i = 0; i < 3; i++)
        for(k = 0; k < 4; k++)
            cvmSet(&tmpProjMatr, i, k, cvmGet(projMatr, i, k));

    cvSVD(&tmpProjMatr, &tmpMatrixD, NULL, &tmpMatrixV, CV_SVD_MODIFY_A + CV_SVD_V_T);

    /* Save position vector. */
    for(i = 0; i < 4; i++)
        cvmSet(posVect, i, 0, cvmGet(&tmpMatrixV, 3, i)); // Solution is last row of V.

    /* Compute calibration and rotation matrices via RQ decomposition. */
    cvGetCols(projMatr, &tmpMatrixM, 0, 3); // M is first square matrix of P.

    CV_Assert(cvDet(&tmpMatrixM) != 0.0); // So far only finite cameras could be decomposed, so M has to be nonsingular [det(M) != 0].

    cvRQDecomp3x3(&tmpMatrixM, calibMatr, rotMatr, rotMatrX, rotMatrY, rotMatrZ, eulerAngles);
}

class SolvePnPCallback CV_FINAL : public cv3d::LMSolver::Callback
{
public:
    SolvePnPCallback(const Mat& _objpt, const Mat& _imgpt,
                     const Mat& _cameraMatrix, const Mat& _distCoeffs)
    {
        objpt = _objpt;
        imgpt = _imgpt;
        cameraMatrix = _cameraMatrix;
        distCoeffs = _distCoeffs;
    }

    bool compute(InputArray _param, OutputArray _err, OutputArray _Jac) const CV_OVERRIDE
    {
        Mat param = _param.getMat();
        CV_Assert((param.cols == 1 || param.rows == 1) && param.total() == 6 && param.type() == CV_64F);
        double* pdata = param.ptr<double>();
        Mat rvec(3, 1, CV_64F, pdata);
        Mat tvec(3, 1, CV_64F, pdata + 3);
        int count = objpt.rows + objpt.cols - 1;
        _err.create(count*2, 1, CV_64F);
        Mat err = _err.getMat();
        err = err.reshape(2, count);
        if( _Jac.needed() )
        {
            _Jac.create(count*2, 6, CV_64F);
            Mat Jac = _Jac.getMat();
            Mat dpdr = Jac.colRange(0, 3);
            Mat dpdt = Jac.colRange(3, 6);
            CvMat objpt_c = cvMat(objpt);
            CvMat err_c = cvMat(err);
            CvMat rvec_c = cvMat(rvec);
            CvMat tvec_c = cvMat(tvec);
            CvMat A_c = cvMat(cameraMatrix);
            CvMat dk_c = cvMat(distCoeffs);
            CvMat dpdr_c = cvMat(dpdr);
            CvMat dpdt_c = cvMat(dpdt);
            cvProjectPoints2( &objpt_c, &rvec_c, &tvec_c, &A_c,
                              distCoeffs.empty() ? 0 : &dk_c,
                              &err_c, &dpdr_c, &dpdt_c, 0, 0, 0, 0 );
        }
        else
        {
            cv3d::projectPoints( objpt, rvec, tvec, cameraMatrix, distCoeffs, err);
        }
        err = err - imgpt;
        err = err.reshape(1, 2*count);
        return true;
    }

    Mat objpt, imgpt, cameraMatrix, distCoeffs;
};


void cv3d::findExtrinsicCameraParams2( const Mat& objectPoints,
                  const Mat& imagePoints, const Mat& A,
                  const Mat& distCoeffs, Mat& rvec, Mat& tvec,
                  int useExtrinsicGuess )
{
    const int max_iter = 20;
    Mat matM, _m, _mn;

    int i, count;
    double a[9], ar[9]={1,0,0,0,1,0,0,0,1}, R[9];
    double MM[9] = { 0 }, U[9] = { 0 }, V[9] = { 0 }, W[3] = { 0 };
    double param[6] = { 0 };
    Mat matA( 3, 3, CV_64F, a );
    Mat _Ar( 3, 3, CV_64F, ar );
    Mat matR( 3, 3, CV_64F, R );
    Mat _r( 3, 1, CV_64F, param );
    Mat _t( 3, 1, CV_64F, param + 3 );
    Mat _MM( 3, 3, CV_64F, MM );
    Mat matU( 3, 3, CV_64F, U );
    Mat matV( 3, 3, CV_64F, V );
    Mat matW( 3, 1, CV_64F, W );
    Mat _param( 6, 1, CV_64F, param );
    Mat _dpdr, _dpdt;

    count = MAX(objectPoints.cols, objectPoints.rows);
    if (objectPoints.checkVector(3) > 0)
        objectPoints.convertTo(matM, CV_64F);
    else
        convertPointsFromHomogeneous(objectPoints, matM, CV_64F);
    if (imagePoints.checkVector(2) > 0)
        imagePoints.convertTo(_m, CV_64F);
    else
        convertPointsFromHomogeneous(imagePoints, _m, CV_64F);
    A.convertTo(matA, CV_64F);

    CV_Assert((count >= 4) || (count == 3 && useExtrinsicGuess)); // it is unsafe to call LM optimisation without an extrinsic guess in the case of 3 points. This is because there is no guarantee that it will converge on the correct solution.

    // normalize image points
    // (unapply the intrinsic matrix transformation and distortion)
    cv3d::undistortPoints(_m, _mn, matA, distCoeffs, Mat(), _Ar);

    if( useExtrinsicGuess )
    {
        CV_Assert((rvec.rows == 1 || rvec.cols == 1) && rvec.total()*rvec.channels() == 3);
        CV_Assert((tvec.rows == 1 || tvec.cols == 1) && tvec.total()*tvec.channels() == 3);
        Mat _r_temp(rvec.rows, rvec.cols, CV_MAKETYPE(CV_64F,rvec.channels()), param);
        Mat _t_temp(tvec.rows, tvec.cols, CV_MAKETYPE(CV_64F,tvec.channels()), param + 3);
        rvec.convertTo(_r_temp, CV_64F);
        tvec.convertTo(_t_temp, CV_64F);
    }
    else
    {
        Scalar Mc = mean(matM);
        Mat _Mc( 1, 3, CV_64F, Mc.val );

        matM = matM.reshape(1, count);
        mulTransposed(matM, _MM, true, _Mc);
        SVDecomp(_MM, matW, noArray(), matV, SVD::MODIFY_A);
        CV_Assert(matW.ptr<double>() == W);
        CV_Assert(matV.ptr<double>() == V);

        // initialize extrinsic parameters
        if( W[2]/W[1] < 1e-3)
        {
            // a planar structure case (all M's lie in the same plane)
            double tt[3];
            Mat R_transform = matV;
            Mat T_transform( 3, 1, CV_64F, tt );

            if( V[2]*V[2] + V[5]*V[5] < 1e-10 )
                R_transform = Mat::eye(3, 3, CV_64F);

            if( determinant(R_transform) < 0 )
                R_transform *= -1.;

            gemm( R_transform, _Mc, -1, Mat(), 0, T_transform, GEMM_2_T );

            const double* Rp = R_transform.ptr<double>();
            const double* Tp = T_transform.ptr<double>();

            Mat _Mxy(count, 1, CV_64FC2);
            const double* src = (double*)matM.data;
            double* dst = (double*)_Mxy.data;

            for( i = 0; i < count; i++, src += 3, dst += 2 )
            {
                dst[0] = Rp[0]*src[0] + Rp[1]*src[1] + Rp[2]*src[2] + Tp[0];
                dst[1] = Rp[3]*src[0] + Rp[4]*src[1] + Rp[5]*src[2] + Tp[1];
            }

            Mat matH = cv3d::findHomography(_Mxy, _mn);

            if( checkRange(matH, true))
            {
                Mat _h1 = matH.col(0);
                Mat _h2 = matH.col(1);
                Mat _h3 = matH.col(2);
                double* h = matH.ptr<double>();
                CV_Assert(matH.isContinuous());
                double h1_norm = std::sqrt(h[0]*h[0] + h[3]*h[3] + h[6]*h[6]);
                double h2_norm = std::sqrt(h[1]*h[1] + h[4]*h[4] + h[7]*h[7]);

                _h1 *= 1./MAX(h1_norm, DBL_EPSILON);
                _h2 *= 1./MAX(h2_norm, DBL_EPSILON);
                _t = _h3 * (2./MAX(h1_norm + h2_norm, DBL_EPSILON));
                _h1.cross(_h2).copyTo(_h3);

                cv3d::Rodrigues( matH, _r );
                cv3d::Rodrigues( _r, matH );
                _t += matH*T_transform;
                matR = matH * R_transform;
            }
            else
            {
                setIdentity(matR);
                _t.setTo(Scalar::all(0));
            }

            Rodrigues( matR, _r );
        }
        else
        {
            // non-planar structure. Use DLT method
            CV_CheckGE(count, 6, "DLT algorithm needs at least 6 points for pose estimation from 3D-2D point correspondences.");
            double LL[12*12], LW[12], LV[12*12], sc;
            Mat _LL( 12, 12, CV_64F, LL );
            Mat _LW( 12, 1, CV_64F, LW );
            Mat _LV( 12, 12, CV_64F, LV );
            Point3d* M = (Point3d*)matM.data;
            Point2d* mn = (Point2d*)_mn.data;

            Mat matL(2*count, 12, CV_64F);
            double* L = matL.ptr<double>();

            for( i = 0; i < count; i++, L += 24 )
            {
                double x = -mn[i].x, y = -mn[i].y;
                L[0] = L[16] = M[i].x;
                L[1] = L[17] = M[i].y;
                L[2] = L[18] = M[i].z;
                L[3] = L[19] = 1.;
                L[4] = L[5] = L[6] = L[7] = 0.;
                L[12] = L[13] = L[14] = L[15] = 0.;
                L[8] = x*M[i].x;
                L[9] = x*M[i].y;
                L[10] = x*M[i].z;
                L[11] = x;
                L[20] = y*M[i].x;
                L[21] = y*M[i].y;
                L[22] = y*M[i].z;
                L[23] = y;
            }

            mulTransposed( matL, _LL, true );
            SVDecomp( _LL, _LW, noArray(), _LV, SVD::MODIFY_A );
            Mat _RRt( 3, 4, CV_64F, LV + 11*12 );
            Mat _RR = _RRt.colRange(0, 3);
            Mat _tt = _RRt.col(3);
            if( determinant(_RR) < 0 )
                _RRt *= -1.0;
            sc = norm(_RR, NORM_L2);
            CV_Assert(fabs(sc) > DBL_EPSILON);
            SVDecomp(_RR, matW, matU, matV, SVD::MODIFY_A);
            matR = matU*matV;
            _tt.convertTo(_t, CV_64F, norm(matR, NORM_L2)/sc);
            cv3d::Rodrigues(matR, _r);
        }
    }

    matM = matM.reshape(3, 1);
    _mn = _mn.reshape(2, 1);

    // refine extrinsic parameters using iterative algorithm
    Ptr<LMSolver::Callback> callb = makePtr<SolvePnPCallback>(matM, _m, matA, distCoeffs);
    Ptr<LMSolver> solver = LMSolver::create(callb, max_iter, (double)FLT_EPSILON);
    solver->run(_param);
    _param.rowRange(0, 3).copyTo(rvec);
    _param.rowRange(3, 6).copyTo(tvec);
}


void cv3d::Rodrigues(InputArray _src, OutputArray _dst, OutputArray _jacobian)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    const Size srcSz = src.size();
    CV_Check(srcSz, srcSz == Size(3, 1) || srcSz == Size(1, 3) ||
             (srcSz == Size(1, 1) && src.channels() == 3) ||
             srcSz == Size(3, 3),
             "Input matrix must be 1x3 or 3x1 for a rotation vector, or 3x3 for a rotation matrix");

    bool v2m = src.cols == 1 || src.rows == 1;
    _dst.create(3, v2m ? 3 : 1, src.depth());
    Mat dst = _dst.getMat();
    CvMat _csrc = cvMat(src), _cdst = cvMat(dst), _cjacobian;
    if( _jacobian.needed() )
    {
        _jacobian.create(v2m ? Size(9, 3) : Size(3, 9), src.depth());
        _cjacobian = cvMat(_jacobian.getMat());
    }
    bool ok = cvRodrigues2(&_csrc, &_cdst, _jacobian.needed() ? &_cjacobian : 0) > 0;
    if( !ok )
        dst = Scalar(0);
}

void cv3d::matMulDeriv( InputArray _Amat, InputArray _Bmat,
                      OutputArray _dABdA, OutputArray _dABdB )
{
    CV_INSTRUMENT_REGION();

    Mat A = _Amat.getMat(), B = _Bmat.getMat();
    _dABdA.create(A.rows*B.cols, A.rows*A.cols, A.type());
    _dABdB.create(A.rows*B.cols, B.rows*B.cols, A.type());
    Mat dABdA = _dABdA.getMat(), dABdB = _dABdB.getMat();
    CvMat matA = cvMat(A), matB = cvMat(B), c_dABdA = cvMat(dABdA), c_dABdB = cvMat(dABdB);
    cvCalcMatMulDeriv(&matA, &matB, &c_dABdA, &c_dABdB);
}


void cv3d::composeRT( InputArray _rvec1, InputArray _tvec1,
                    InputArray _rvec2, InputArray _tvec2,
                    OutputArray _rvec3, OutputArray _tvec3,
                    OutputArray _dr3dr1, OutputArray _dr3dt1,
                    OutputArray _dr3dr2, OutputArray _dr3dt2,
                    OutputArray _dt3dr1, OutputArray _dt3dt1,
                    OutputArray _dt3dr2, OutputArray _dt3dt2 )
{
    Mat rvec1 = _rvec1.getMat(), tvec1 = _tvec1.getMat();
    Mat rvec2 = _rvec2.getMat(), tvec2 = _tvec2.getMat();
    int rtype = rvec1.type();
    _rvec3.create(rvec1.size(), rtype);
    _tvec3.create(tvec1.size(), rtype);
    Mat rvec3 = _rvec3.getMat(), tvec3 = _tvec3.getMat();

    CvMat c_rvec1 = cvMat(rvec1), c_tvec1 = cvMat(tvec1), c_rvec2 = cvMat(rvec2),
          c_tvec2 = cvMat(tvec2), c_rvec3 = cvMat(rvec3), c_tvec3 = cvMat(tvec3);
    CvMat c_dr3dr1, c_dr3dt1, c_dr3dr2, c_dr3dt2, c_dt3dr1, c_dt3dt1, c_dt3dr2, c_dt3dt2;
    CvMat *p_dr3dr1=0, *p_dr3dt1=0, *p_dr3dr2=0, *p_dr3dt2=0, *p_dt3dr1=0, *p_dt3dt1=0, *p_dt3dr2=0, *p_dt3dt2=0;
#define CV_COMPOSE_RT_PARAM(name) \
    Mat name; \
    if (_ ## name.needed())\
    { \
        _ ## name.create(3, 3, rtype); \
        name = _ ## name.getMat(); \
        p_ ## name = &(c_ ## name = cvMat(name)); \
    }

    CV_COMPOSE_RT_PARAM(dr3dr1); CV_COMPOSE_RT_PARAM(dr3dt1);
    CV_COMPOSE_RT_PARAM(dr3dr2); CV_COMPOSE_RT_PARAM(dr3dt2);
    CV_COMPOSE_RT_PARAM(dt3dr1); CV_COMPOSE_RT_PARAM(dt3dt1);
    CV_COMPOSE_RT_PARAM(dt3dr2); CV_COMPOSE_RT_PARAM(dt3dt2);
#undef CV_COMPOSE_RT_PARAM

    cvComposeRT(&c_rvec1, &c_tvec1, &c_rvec2, &c_tvec2, &c_rvec3, &c_tvec3,
                p_dr3dr1, p_dr3dt1, p_dr3dr2, p_dr3dt2,
                p_dt3dr1, p_dt3dt1, p_dt3dr2, p_dt3dt2);
}


void cv3d::projectPoints( InputArray _opoints,
                        InputArray _rvec,
                        InputArray _tvec,
                        InputArray _cameraMatrix,
                        InputArray _distCoeffs,
                        OutputArray _ipoints,
                        OutputArray _jacobian,
                        double aspectRatio )
{
    Mat opoints = _opoints.getMat();
    int npoints = opoints.checkVector(3), depth = opoints.depth();
    if (npoints < 0)
        opoints = opoints.t();
    npoints = opoints.checkVector(3);
    CV_Assert(npoints >= 0 && (depth == CV_32F || depth == CV_64F));

    if (opoints.cols == 3)
        opoints = opoints.reshape(3);

    CvMat dpdrot, dpdt, dpdf, dpdc, dpddist;
    CvMat *pdpdrot=0, *pdpdt=0, *pdpdf=0, *pdpdc=0, *pdpddist=0;

    CV_Assert( _ipoints.needed() );

    _ipoints.create(npoints, 1, CV_MAKETYPE(depth, 2), -1, true);
    Mat imagePoints = _ipoints.getMat();
    CvMat c_imagePoints = cvMat(imagePoints);
    CvMat c_objectPoints = cvMat(opoints);
    Mat cameraMatrix = _cameraMatrix.getMat();

    Mat rvec = _rvec.getMat(), tvec = _tvec.getMat();
    CvMat c_cameraMatrix = cvMat(cameraMatrix);
    CvMat c_rvec = cvMat(rvec), c_tvec = cvMat(tvec);

    double dc0buf[5]={0};
    Mat dc0(5,1,CV_64F,dc0buf);
    Mat distCoeffs = _distCoeffs.getMat();
    if( distCoeffs.empty() )
        distCoeffs = dc0;
    CvMat c_distCoeffs = cvMat(distCoeffs);
    int ndistCoeffs = distCoeffs.rows + distCoeffs.cols - 1;

    Mat jacobian;
    if( _jacobian.needed() )
    {
        _jacobian.create(npoints*2, 3+3+2+2+ndistCoeffs, CV_64F);
        jacobian = _jacobian.getMat();
        pdpdrot = &(dpdrot = cvMat(jacobian.colRange(0, 3)));
        pdpdt = &(dpdt = cvMat(jacobian.colRange(3, 6)));
        pdpdf = &(dpdf = cvMat(jacobian.colRange(6, 8)));
        pdpdc = &(dpdc = cvMat(jacobian.colRange(8, 10)));
        pdpddist = &(dpddist = cvMat(jacobian.colRange(10, 10+ndistCoeffs)));
    }

    cvProjectPoints2( &c_objectPoints, &c_rvec, &c_tvec, &c_cameraMatrix, &c_distCoeffs,
                      &c_imagePoints, pdpdrot, pdpdt, pdpdf, pdpdc, pdpddist, aspectRatio );
}

void cv3d::getUndistortRectangles(InputArray _cameraMatrix, InputArray _distCoeffs,
              InputArray R, InputArray newCameraMatrix, Size imgSize,
              Rect_<float>& inner, Rect_<float>& outer )
{
    const int N = 9;
    int x, y, k;
    Mat _pts(1, N*N, CV_32FC2);
    Point2f* pts = _pts.ptr<Point2f>();

    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
            pts[k++] = Point2f((float)x*imgSize.width/(N-1), (float)y*imgSize.height/(N-1));

    undistortPoints(_pts, _pts, _cameraMatrix, _distCoeffs, R, newCameraMatrix);

    float iX0=-FLT_MAX, iX1=FLT_MAX, iY0=-FLT_MAX, iY1=FLT_MAX;
    float oX0=FLT_MAX, oX1=-FLT_MAX, oY0=FLT_MAX, oY1=-FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
        {
            Point2f p = pts[k++];
            oX0 = MIN(oX0, p.x);
            oX1 = MAX(oX1, p.x);
            oY0 = MIN(oY0, p.y);
            oY1 = MAX(oY1, p.y);

            if( x == 0 )
                iX0 = MAX(iX0, p.x);
            if( x == N-1 )
                iX1 = MIN(iX1, p.x);
            if( y == 0 )
                iY0 = MAX(iY0, p.y);
            if( y == N-1 )
                iY1 = MIN(iY1, p.y);
        }
    inner = Rect_<float>(iX0, iY0, iX1-iX0, iY1-iY0);
    outer = Rect_<float>(oX0, oY0, oX1-oX0, oY1-oY0);
}

cv::Mat cv3d::getOptimalNewCameraMatrix( InputArray _cameraMatrix, InputArray _distCoeffs,
                                  Size imgSize, double alpha, Size newImgSize,
                                  Rect* validPixROI, bool centerPrincipalPoint )
{
    Rect_<float> inner, outer;
    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imgSize;

    Mat cameraMatrix = _cameraMatrix.getMat(), M;
    cameraMatrix.convertTo(M, CV_64F);
    CV_Assert(M.isContinuous());

    if( centerPrincipalPoint )
    {
        double cx0 = M.at<double>(0, 2);
        double cy0 = M.at<double>(1, 2);
        double cx = (newImgSize.width-1)*0.5;
        double cy = (newImgSize.height-1)*0.5;

        getUndistortRectangles( _cameraMatrix, _distCoeffs, Mat(), cameraMatrix, imgSize, inner, outer );
        double s0 = std::max(std::max(std::max((double)cx/(cx0 - inner.x), (double)cy/(cy0 - inner.y)),
                                      (double)cx/(inner.x + inner.width - cx0)),
                             (double)cy/(inner.y + inner.height - cy0));
        double s1 = std::min(std::min(std::min((double)cx/(cx0 - outer.x), (double)cy/(cy0 - outer.y)),
                                      (double)cx/(outer.x + outer.width - cx0)),
                             (double)cy/(outer.y + outer.height - cy0));
        double s = s0*(1 - alpha) + s1*alpha;

        M.at<double>(0, 0) *= s;
        M.at<double>(1, 1) *= s;
        M.at<double>(0, 2) = cx;
        M.at<double>(1, 2) = cy;

        if( validPixROI )
        {
            inner = cv::Rect_<float>((float)((inner.x - cx0)*s + cx),
                                     (float)((inner.y - cy0)*s + cy),
                                     (float)(inner.width*s),
                                     (float)(inner.height*s));
            Rect r(cvCeil(inner.x), cvCeil(inner.y), cvFloor(inner.width), cvFloor(inner.height));
            r &= Rect(0, 0, newImgSize.width, newImgSize.height);
            *validPixROI = r;
        }
    }
    else
    {
        // Get inscribed and circumscribed rectangles in normalized
        // (independent of camera matrix) coordinates
        getUndistortRectangles( _cameraMatrix, _distCoeffs, Mat(), Mat(), imgSize, inner, outer );

        // Projection mapping inner rectangle to viewport
        double fx0 = (newImgSize.width  - 1) / inner.width;
        double fy0 = (newImgSize.height - 1) / inner.height;
        double cx0 = -fx0 * inner.x;
        double cy0 = -fy0 * inner.y;

        // Projection mapping outer rectangle to viewport
        double fx1 = (newImgSize.width  - 1) / outer.width;
        double fy1 = (newImgSize.height - 1) / outer.height;
        double cx1 = -fx1 * outer.x;
        double cy1 = -fy1 * outer.y;

        // Interpolate between the two optimal projections
        M.at<double>(0, 0) = fx0*(1 - alpha) + fx1*alpha;
        M.at<double>(1, 1) = fy0*(1 - alpha) + fy1*alpha;
        M.at<double>(0, 2) = cx0*(1 - alpha) + cx1*alpha;
        M.at<double>(1, 2) = cy0*(1 - alpha) + cy1*alpha;

        if( validPixROI )
        {
            getUndistortRectangles( _cameraMatrix, _distCoeffs, Mat(), M, imgSize, inner, outer );
            Rect r = inner;
            r &= Rect(0, 0, newImgSize.width, newImgSize.height);
            *validPixROI = r;
        }
    }

    return M;
}

cv::Vec3d cv3d::RQDecomp3x3( InputArray _Mmat,
                   OutputArray _Rmat,
                   OutputArray _Qmat,
                   OutputArray _Qx,
                   OutputArray _Qy,
                   OutputArray _Qz )
{
    CV_INSTRUMENT_REGION();

    Mat M = _Mmat.getMat();
    _Rmat.create(3, 3, M.type());
    _Qmat.create(3, 3, M.type());
    Mat Rmat = _Rmat.getMat();
    Mat Qmat = _Qmat.getMat();
    Vec3d eulerAngles;

    CvMat matM = cvMat(M), matR = cvMat(Rmat), matQ = cvMat(Qmat);
#define CV_RQDecomp3x3_PARAM(name) \
    Mat name; \
    CvMat c_ ## name, *p ## name = NULL; \
    if( _ ## name.needed() ) \
    { \
        _ ## name.create(3, 3, M.type()); \
        name = _ ## name.getMat(); \
        c_ ## name = cvMat(name); p ## name = &c_ ## name; \
    }

    CV_RQDecomp3x3_PARAM(Qx);
    CV_RQDecomp3x3_PARAM(Qy);
    CV_RQDecomp3x3_PARAM(Qz);
#undef CV_RQDecomp3x3_PARAM
    cvRQDecomp3x3(&matM, &matR, &matQ, pQx, pQy, pQz, (CvPoint3D64f*)&eulerAngles[0]);
    return eulerAngles;
}


void cv3d::decomposeProjectionMatrix( InputArray _projMatrix, OutputArray _cameraMatrix,
                                    OutputArray _rotMatrix, OutputArray _transVect,
                                    OutputArray _rotMatrixX, OutputArray _rotMatrixY,
                                    OutputArray _rotMatrixZ, OutputArray _eulerAngles )
{
    CV_INSTRUMENT_REGION();

    Mat projMatrix = _projMatrix.getMat();
    int type = projMatrix.type();
    _cameraMatrix.create(3, 3, type);
    _rotMatrix.create(3, 3, type);
    _transVect.create(4, 1, type);
    Mat cameraMatrix = _cameraMatrix.getMat();
    Mat rotMatrix = _rotMatrix.getMat();
    Mat transVect = _transVect.getMat();
    CvMat c_projMatrix = cvMat(projMatrix), c_cameraMatrix = cvMat(cameraMatrix);
    CvMat c_rotMatrix = cvMat(rotMatrix), c_transVect = cvMat(transVect);
    CvPoint3D64f *p_eulerAngles = 0;

#define CV_decomposeProjectionMatrix_PARAM(name) \
    Mat name; \
    CvMat c_ ## name, *p_ ## name = NULL; \
    if( _ ## name.needed() ) \
    { \
        _ ## name.create(3, 3, type); \
        name = _ ## name.getMat(); \
        c_ ## name = cvMat(name); p_ ## name = &c_ ## name; \
    }

    CV_decomposeProjectionMatrix_PARAM(rotMatrixX);
    CV_decomposeProjectionMatrix_PARAM(rotMatrixY);
    CV_decomposeProjectionMatrix_PARAM(rotMatrixZ);
#undef CV_decomposeProjectionMatrix_PARAM

    if( _eulerAngles.needed() )
    {
        _eulerAngles.create(3, 1, CV_64F, -1, true);
        p_eulerAngles = _eulerAngles.getMat().ptr<CvPoint3D64f>();
    }

    cvDecomposeProjectionMatrix(&c_projMatrix, &c_cameraMatrix, &c_rotMatrix,
                                &c_transVect, p_rotMatrixX, p_rotMatrixY,
                                p_rotMatrixZ, p_eulerAngles);
}

#if 0
namespace cv {
namespace _3d {

static void adjust3rdMatrix(InputArrayOfArrays _imgpt1_0,
                            InputArrayOfArrays _imgpt3_0,
                            const Mat& cameraMatrix1, const Mat& distCoeffs1,
                            const Mat& cameraMatrix3, const Mat& distCoeffs3,
                            const Mat& R1, const Mat& R3, const Mat& P1, Mat& P3 )
{
    size_t n1 = _imgpt1_0.total(), n3 = _imgpt3_0.total();
    std::vector<Point2f> imgpt1, imgpt3;

    for( int i = 0; i < (int)std::min(n1, n3); i++ )
    {
        Mat pt1 = _imgpt1_0.getMat(i), pt3 = _imgpt3_0.getMat(i);
        int ni1 = pt1.checkVector(2, CV_32F), ni3 = pt3.checkVector(2, CV_32F);
        CV_Assert( ni1 > 0 && ni1 == ni3 );
        const Point2f* pt1data = pt1.ptr<Point2f>();
        const Point2f* pt3data = pt3.ptr<Point2f>();
        std::copy(pt1data, pt1data + ni1, std::back_inserter(imgpt1));
        std::copy(pt3data, pt3data + ni3, std::back_inserter(imgpt3));
    }

    undistortPoints(imgpt1, imgpt1, cameraMatrix1, distCoeffs1, R1, P1);
    undistortPoints(imgpt3, imgpt3, cameraMatrix3, distCoeffs3, R3, P3);

    double y1_ = 0, y2_ = 0, y1y1_ = 0, y1y2_ = 0;
    size_t n = imgpt1.size();
    CV_DbgAssert(n > 0);

    for( size_t i = 0; i < n; i++ )
    {
        double y1 = imgpt3[i].y, y2 = imgpt1[i].y;

        y1_ += y1; y2_ += y2;
        y1y1_ += y1*y1; y1y2_ += y1*y2;
    }

    y1_ /= n;
    y2_ /= n;
    y1y1_ /= n;
    y1y2_ /= n;

    double a = (y1y2_ - y1_*y2_)/(y1y1_ - y1_*y1_);
    double b = y2_ - a*y1_;

    P3.at<double>(0,0) *= a;
    P3.at<double>(1,1) *= a;
    P3.at<double>(0,2) = P3.at<double>(0,2)*a;
    P3.at<double>(1,2) = P3.at<double>(1,2)*a + b;
    P3.at<double>(0,3) *= a;
    P3.at<double>(1,3) *= a;
}

}}

float cv3d::rectify3Collinear( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                   InputArray _cameraMatrix2, InputArray _distCoeffs2,
                   InputArray _cameraMatrix3, InputArray _distCoeffs3,
                   InputArrayOfArrays _imgpt1,
                   InputArrayOfArrays _imgpt3,
                   Size imageSize, InputArray _Rmat12, InputArray _Tmat12,
                   InputArray _Rmat13, InputArray _Tmat13,
                   OutputArray _Rmat1, OutputArray _Rmat2, OutputArray _Rmat3,
                   OutputArray _Pmat1, OutputArray _Pmat2, OutputArray _Pmat3,
                   OutputArray _Qmat,
                   double alpha, Size newImgSize,
                   Rect* roi1, Rect* roi2, int flags )
{
    // first, rectify the 1-2 stereo pair
    stereoRectify( _cameraMatrix1, _distCoeffs1, _cameraMatrix2, _distCoeffs2,
                   imageSize, _Rmat12, _Tmat12, _Rmat1, _Rmat2, _Pmat1, _Pmat2, _Qmat,
                   flags, alpha, newImgSize, roi1, roi2 );

    Mat R12 = _Rmat12.getMat(), R13 = _Rmat13.getMat(), T12 = _Tmat12.getMat(), T13 = _Tmat13.getMat();

    _Rmat3.create(3, 3, CV_64F);
    _Pmat3.create(3, 4, CV_64F);

    Mat P1 = _Pmat1.getMat(), P2 = _Pmat2.getMat();
    Mat R3 = _Rmat3.getMat(), P3 = _Pmat3.getMat();

    // recompute rectification transforms for cameras 1 & 2.
    Mat om, r_r, r_r13;

    if( R13.size() != Size(3,3) )
        Rodrigues(R13, r_r13);
    else
        R13.copyTo(r_r13);

    if( R12.size() == Size(3,3) )
        Rodrigues(R12, om);
    else
        R12.copyTo(om);

    om *= -0.5;
    Rodrigues(om, r_r); // rotate cameras to same orientation by averaging
    Mat_<double> t12 = r_r * T12;

    int idx = fabs(t12(0,0)) > fabs(t12(1,0)) ? 0 : 1;
    double c = t12(idx,0), nt = norm(t12, CV_L2);
    CV_Assert(fabs(nt) > 0);
    Mat_<double> uu = Mat_<double>::zeros(3,1);
    uu(idx, 0) = c > 0 ? 1 : -1;

    // calculate global Z rotation
    Mat_<double> ww = t12.cross(uu), wR;
    double nw = norm(ww, CV_L2);
    CV_Assert(fabs(nw) > 0);
    ww *= acos(fabs(c)/nt)/nw;
    Rodrigues(ww, wR);

    // now rotate camera 3 to make its optical axis parallel to cameras 1 and 2.
    R3 = wR*r_r.t()*r_r13.t();
    Mat_<double> t13 = R3 * T13;

    P2.copyTo(P3);
    Mat t = P3.col(3);
    t13.copyTo(t);
    P3.at<double>(0,3) *= P3.at<double>(0,0);
    P3.at<double>(1,3) *= P3.at<double>(1,1);

    if( !_imgpt1.empty() && !_imgpt3.empty() )
        adjust3rdMatrix(_imgpt1, _imgpt3, _cameraMatrix1.getMat(), _distCoeffs1.getMat(),
                        _cameraMatrix3.getMat(), _distCoeffs3.getMat(), _Rmat1.getMat(), R3, P1, P3);

    return (float)((P3.at<double>(idx,3)/P3.at<double>(idx,idx))/
                   (P2.at<double>(idx,3)/P2.at<double>(idx,idx)));
}
#endif

/* End of file. */
