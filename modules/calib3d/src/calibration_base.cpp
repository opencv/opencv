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
#include "hal_replacement.hpp"
#include "distortion_model.hpp"
#include "opencv2/calib3d/calib3d_c.h"
#include "opencv2/core/core_c.h"
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
void cv::matMulDeriv( InputArray A_, InputArray B_, OutputArray dABdA_, OutputArray dABdB_ )
{
    CV_INSTRUMENT_REGION();

    Mat A = A_.getMat(), B = B_.getMat();
    int type = A.type();
    CV_Assert(type == B.type());
    CV_Assert(type == CV_32F || type == CV_64F);
    CV_Assert(A.cols == B.rows);

    dABdA_.create(A.rows*B.cols, A.rows*A.cols, type);
    dABdB_.create(A.rows*B.cols, B.rows*B.cols, type);
    Mat dABdA = dABdA_.getMat(), dABdB = dABdB_.getMat();

    int M = A.rows, L = A.cols, N = B.cols;
    int bstep = (int)(B.step/B.elemSize());

    if( type == CV_32F )
    {
        for( int i = 0; i < M*N; i++ )
        {
            int j, i1 = i / N,  i2 = i % N;

            const float* a = A.ptr<float>(i1);
            const float* b = B.ptr<float>() + i2;
            float* dcda = dABdA.ptr<float>(i);
            float* dcdb = dABdB.ptr<float>(i);

            memset(dcda, 0, M*L*sizeof(dcda[0]));
            memset(dcdb, 0, L*N*sizeof(dcdb[0]));

            for( j = 0; j < L; j++ )
            {
                dcda[i1*L + j] = b[j*bstep];
                dcdb[j*N + i2] = a[j];
            }
        }
    }
    else
    {
        for( int i = 0; i < M*N; i++ )
        {
            int j, i1 = i / N,  i2 = i % N;

            const double* a = A.ptr<double>(i1);
            const double* b = B.ptr<double>() + i2;
            double* dcda = dABdA.ptr<double>(i);
            double* dcdb = dABdB.ptr<double>(i);

            memset(dcda, 0, M*L*sizeof(dcda[0]));
            memset(dcdb, 0, L*N*sizeof(dcdb[0]));

            for( j = 0; j < L; j++ )
            {
                dcda[i1*L + j] = b[j*bstep];
                dcdb[j*N + i2] = a[j];
            }
        }
    }
}

void cv::Rodrigues(InputArray _src, OutputArray _dst, OutputArray _jacobian)
{
    CV_INSTRUMENT_REGION();

    Mat src = _src.getMat();
    const Size srcSz = src.size();
    int srccn = src.channels();
    int depth = src.depth();
    CV_Check(srcSz, ((srcSz == Size(3, 1) || srcSz == Size(1, 3)) && srccn == 1) ||
             (srcSz == Size(1, 1) && srccn == 3) ||
             (srcSz == Size(3, 3) && srccn == 1),
             "Input matrix must be 1x3 or 3x1 for a rotation vector, or 3x3 for a rotation matrix");

    bool v2m = src.cols == 1 || src.rows == 1;
    _dst.create(3, v2m ? 3 : 1, depth);
    Mat dst = _dst.getMat(), jacobian;
    if( _jacobian.needed() )
    {
        _jacobian.create(v2m ? Size(9, 3) : Size(3, 9), src.depth());
        jacobian = _jacobian.getMat();
    }

    double J[27] = {0};
    Mat matJ( 3, 9, CV_64F, J);

    dst.setTo(0);

    if( depth != CV_32F && depth != CV_64F )
        CV_Error( cv::Error::StsUnsupportedFormat, "The matrices must have 32f or 64f data type" );

    if( v2m )
    {
        int sstep = src.rows > 1 ? (int)src.step1() : 1;

        Point3d r;
        if( depth == CV_32F )
        {
            const float* sptr = src.ptr<float>();
            r.x = sptr[0];
            r.y = sptr[sstep];
            r.z = sptr[sstep*2];
        }
        else
        {
            const double* sptr = src.ptr<double>();
            r.x = sptr[0];
            r.y = sptr[sstep];
            r.z = sptr[sstep*2];
        }

        double theta = norm(r);
        if( theta < DBL_EPSILON )
        {
            dst = Mat::eye(3, 3, depth);
            if( jacobian.data )
            {
                memset( J, 0, sizeof(J) );
                J[5] = J[15] = J[19] = -1;
                J[7] = J[11] = J[21] = 1;
            }
        }
        else
        {
            double c = std::cos(theta);
            double s = std::sin(theta);
            double c1 = 1. - c;
            double itheta = theta ? 1./theta : 0.;

            r *= itheta;

            Matx33d rrt( r.x*r.x, r.x*r.y, r.x*r.z, r.x*r.y, r.y*r.y, r.y*r.z, r.x*r.z, r.y*r.z, r.z*r.z );
            Matx33d r_x(    0, -r.z,  r.y,
                          r.z,    0, -r.x,
                         -r.y,  r.x,    0 );

            // R = cos(theta)*I + (1 - cos(theta))*r*rT + sin(theta)*[r_x]
            Matx33d R = c*Matx33d::eye() + c1*rrt + s*r_x;
            Mat(R).convertTo(dst, depth);

            if( jacobian.data )
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
    else
    {
        Matx33d U, Vt;
        Vec3d W;
        double theta, s, c;
        int dstep = dst.rows > 1 ? (int)dst.step1() : 1;

        Matx33d R;
        src.convertTo(R, CV_64F);

        if( !checkRange(R, true, NULL, -100, 100) )
        {
            dst.setTo(0);
            if (jacobian.data)
                jacobian.setTo(0);
            return;
        }

        SVD::compute(R, W, U, Vt);
        R = U*Vt;

        Point3d r(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));

        s = std::sqrt((r.x*r.x + r.y*r.y + r.z*r.z)*0.25);
        c = (R(0, 0) + R(1, 1) + R(2, 2) - 1)*0.5;
        c = c > 1. ? 1. : c < -1. ? -1. : c;
        theta = std::acos(c);

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

            if( jacobian.data )
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

            if( jacobian.data )
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

                Mat _dvardR( 5, 9, CV_64FC1, dvardR );
                Mat _dvar2dvar( 4, 5, CV_64FC1, dvar2dvar );
                Mat _domegadvar2( 3, 4, CV_64FC1, domegadvar2 );
                double t0[3*5];
                Mat _t0( 3, 5, CV_64FC1, t0 );

                gemm(_domegadvar2, _dvar2dvar, 1, noArray(), 0, _t0);
                gemm(_t0, _dvardR, 1, noArray(), 0, matJ);
                CV_Assert(matJ.ptr<double>() == J);

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
            float* dptr = dst.ptr<float>();
            dptr[0] = (float)r.x;
            dptr[dstep] = (float)r.y;
            dptr[dstep*2] = (float)r.z;
        }
        else
        {
            double* dptr = dst.ptr<double>();
            dptr[0] = r.x;
            dptr[dstep] = r.y;
            dptr[dstep*2] = r.z;
        }
    }

    if( jacobian.data )
    {
        if( depth == CV_32F )
        {
            if( jacobian.rows == matJ.rows )
                matJ.convertTo(jacobian, CV_32F);
            else
            {
                float Jf[3*9];
                Mat _Jf( matJ.rows, matJ.cols, CV_32FC1, Jf );
                matJ.convertTo(_Jf, CV_32F);
                transpose(_Jf, jacobian);
            }
        }
        else if( jacobian.rows == matJ.rows )
            matJ.copyTo(jacobian);
        else
            transpose(matJ, jacobian);
    }
}

// reimplementation of compose_motion.m
void cv::composeRT( InputArray _rvec1, InputArray _tvec1,
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

    CV_Assert(rtype == CV_32F || rtype == CV_64F);
    Size rsz = rvec1.size();
    CV_Assert(rsz == Size(3, 1) || rsz == Size(1, 3));
    CV_Assert(rsz == rvec2.size() && rsz == tvec1.size() && rsz == tvec2.size());

    Mat dr3dr1, dr3dt1, dr3dr2, dr3dt2;
    Mat dt3dr1, dt3dt1, dt3dr2, dt3dt2;
    if(_dr3dr1.needed()) {
        _dr3dr1.create(3, 3, rtype);
        dr3dr1 = _dr3dr1.getMat();
    }
    if(_dr3dt1.needed()) {
        _dr3dt1.create(3, 3, rtype);
        dr3dt1 = _dr3dt1.getMat();
    }
    if(_dr3dr2.needed()) {
        _dr3dr2.create(3, 3, rtype);
        dr3dr2 = _dr3dr2.getMat();
    }
    if(_dr3dt2.needed()) {
        _dr3dt2.create(3, 3, rtype);
        dr3dt2 = _dr3dt2.getMat();
    }
    if(_dt3dr1.needed()) {
        _dt3dr1.create(3, 3, rtype);
        dt3dr1 = _dt3dr1.getMat();
    }
    if(_dt3dt1.needed()) {
        _dt3dt1.create(3, 3, rtype);
        dt3dt1 = _dt3dt1.getMat();
    }
    if(_dt3dr2.needed()) {
        _dt3dr2.create(3, 3, rtype);
        dt3dr2 = _dt3dr2.getMat();
    }
    if(_dt3dt2.needed()) {
        _dt3dt2.create(3, 3, rtype);
        dt3dt2 = _dt3dt2.getMat();
    }

    double _r1[3], _r2[3];
    double _R1[9], _d1[9*3], _R2[9], _d2[9*3];
    Mat r1(rsz,CV_64F,_r1), r2(rsz,CV_64F,_r2);
    Mat R1(3,3,CV_64F,_R1), R2(3,3,CV_64F,_R2);
    Mat dR1dr1(3,9,CV_64F,_d1), dR2dr2(3,9,CV_64F,_d2);

    rvec1.convertTo(r1, CV_64F);
    rvec2.convertTo(r2, CV_64F);

    Rodrigues(r1, R1, dR1dr1);
    Rodrigues(r2, R2, dR2dr2);
    CV_Assert(dR1dr1.ptr<double>() == _d1);
    CV_Assert(dR2dr2.ptr<double>() == _d2);

    double _r3[3], _R3[9], _dR3dR1[9*9], _dR3dR2[9*9], _dr3dR3[9*3];
    double _W1[9*3], _W2[3*3];
    Mat r3(3,1,CV_64F,_r3), R3(3,3,CV_64F,_R3);
    Mat dR3dR1(9,9,CV_64F,_dR3dR1), dR3dR2(9,9,CV_64F,_dR3dR2);
    Mat dr3dR3(9,3,CV_64F,_dr3dR3);
    Mat W1(3,9,CV_64F,_W1), W2(3,3,CV_64F,_W2);

    R3 = R2*R1;
    matMulDeriv(R2, R1, dR3dR2, dR3dR1);
    Rodrigues(R3, r3, dr3dR3);
    CV_Assert(dr3dR3.ptr<double>() == _dr3dR3);

    r3.convertTo(_rvec3, rtype);

    if( dr3dr1.data )
    {
        gemm(dr3dR3, dR3dR1, 1, noArray(), 0, W1, GEMM_1_T);
        gemm(W1, dR1dr1, 1, noArray(), 0, W2, GEMM_2_T);
        W2.convertTo(dr3dr1, rtype);
    }

    if( dr3dr2.data )
    {
        gemm(dr3dR3, dR3dR2, 1, noArray(), 0, W1, GEMM_1_T);
        gemm(W1, dR2dr2, 1, noArray(), 0, W2, GEMM_2_T);
        W2.convertTo(dr3dr2, rtype);
    }

    if( dr3dt1.data )
        dr3dt1.setTo(0);
    if( dr3dt2.data )
        dr3dt2.setTo(0);

    double _t1[3], _t2[3], _t3[3], _dxdR2[3*9], _dxdt1[3*3], _W3[3*3];
    Mat t1(3,1,CV_64F,_t1), t2(3,1,CV_64F,_t2);
    Mat t3(3,1,CV_64F,_t3);
    Mat dxdR2(3, 9, CV_64F, _dxdR2);
    Mat dxdt1(3, 3, CV_64F, _dxdt1);
    Mat W3(3, 3, CV_64F, _W3);

    tvec1.convertTo(t1, CV_64F);
    tvec2.convertTo(t2, CV_64F);
    gemm(R2, t1, 1, t2, 1, t3);
    t3.convertTo(_tvec3, rtype);

    if( dt3dr2.data || dt3dt1.data )
    {
        matMulDeriv(R2, t1, dxdR2, dxdt1);
        if( dt3dr2.data )
        {
            gemm(dxdR2, dR2dr2, 1, noArray(), 0, W3, GEMM_2_T);
            W3.convertTo(dt3dr2, rtype);
        }
        if( dt3dt1.data )
            dxdt1.convertTo(dt3dt1, rtype);
    }

    if( dt3dt2.data )
        setIdentity(dt3dt2);
    if( dt3dr1.data )
        dt3dr1.setTo(0);
}

static const char* cvDistCoeffErr =
    "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8, 8x1, 1x12, 12x1, 1x14 or 14x1 floating-point vector";

void cv::projectPoints( InputArray _objectPoints,
                        InputArray _rvec, InputArray _tvec,
                        InputArray _cameraMatrix, InputArray _distCoeffs,
                        OutputArray _imagePoints, OutputArray _dpdr,
                        OutputArray _dpdt, OutputArray _dpdf,
                        OutputArray _dpdc, OutputArray _dpdk,
                        OutputArray _dpdo, double aspectRatio)
{
    Mat _m, objectPoints = _objectPoints.getMat();
    Mat dpdr, dpdt, dpdc, dpdf, dpdk, dpdo;

    int i, j;
    double R[9], dRdr[27], t[3], a[9], k[14] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0}, fx, fy, cx, cy;
    Matx33d matTilt = Matx33d::eye();
    Matx33d dMatTiltdTauX(0,0,0,0,0,0,0,-1,0);
    Matx33d dMatTiltdTauY(0,0,0,0,0,0,1,0,0);
    Mat matR( 3, 3, CV_64F, R ), _dRdr( 3, 9, CV_64F, dRdr );
    double *dpdr_p = 0, *dpdt_p = 0, *dpdk_p = 0, *dpdf_p = 0, *dpdc_p = 0;
    double* dpdo_p = 0;
    int dpdr_step = 0, dpdt_step = 0, dpdk_step = 0, dpdf_step = 0, dpdc_step = 0;
    int dpdo_step = 0;
    bool fixedAspectRatio = aspectRatio > FLT_EPSILON;

    int objpt_depth = objectPoints.depth();
    int objpt_cn = objectPoints.channels();
    int total = (int)(objectPoints.total()*objectPoints.channels());
    int count = total / 3;
    if(total % 3 != 0)
    {
        //we have stopped support of homogeneous coordinates because it cause ambiguity in interpretation of the input data
        CV_Error( cv::Error::StsBadArg, "Homogeneous coordinates are not supported" );
    }
    count = total / 3;
    CV_Assert(objpt_depth == CV_32F || objpt_depth == CV_64F);
    CV_Assert((objectPoints.rows == 1 && objpt_cn == 3) ||
              (objectPoints.rows == count && objpt_cn*objectPoints.cols == 3) ||
              (objectPoints.rows == 3 && objpt_cn == 1 && objectPoints.cols == count));

    if (objectPoints.rows == 3 && objectPoints.cols == count) {
        Mat temp;
        transpose(objectPoints, temp);
        objectPoints = temp;
    }

    CV_Assert( _imagePoints.needed() );
    _imagePoints.create(count, 1, CV_MAKETYPE(objpt_depth, 2), -1, true);
    Mat ipoints = _imagePoints.getMat();

    Mat rvec = _rvec.getMat(), tvec = _tvec.getMat();
    if(!((rvec.depth() == CV_32F || rvec.depth() == CV_64F) &&
        (rvec.size() == Size(3, 3) ||
        (rvec.rows == 1 && rvec.cols*rvec.channels() == 3) ||
        (rvec.rows == 3 && rvec.cols*rvec.channels() == 1)))) {
        CV_Error(cv::Error::StsBadArg, "rvec must be 3x3 or 1x3 or 3x1 floating-point array");
    }

    if( rvec.size() == Size(3, 3) )
    {
        rvec.convertTo(matR, CV_64F);
        Vec3d rvec_d;
        Rodrigues(matR, rvec_d);
        Rodrigues(rvec_d, matR, _dRdr);
        rvec.convertTo(matR, CV_64F);
    }
    else
    {
        double r[3];
        Mat _r(rvec.size(), CV_64FC(rvec.channels()), r);
        rvec.convertTo(_r, CV_64F);
        Rodrigues(_r, matR, _dRdr);
    }

    if(!((tvec.depth() == CV_32F || tvec.depth() == CV_64F) &&
        ((tvec.rows == 1 && tvec.cols*tvec.channels() == 3) ||
        (tvec.rows == 3 && tvec.cols*tvec.channels() == 1)))) {
        CV_Error(cv::Error::StsBadArg, "tvec must be 1x3 or 3x1 floating-point array");
    }

    Mat _t(tvec.size(), CV_64FC(tvec.channels()), t);
    tvec.convertTo(_t, CV_64F);

    Mat cameraMatrix = _cameraMatrix.getMat();

    if(cameraMatrix.size() != Size(3, 3) || cameraMatrix.channels() != 1)
        CV_Error( cv::Error::StsBadArg, "Intrinsic parameters must be 3x3 floating-point matrix" );
    Mat _a(3, 3, CV_64F, a);
    cameraMatrix.convertTo(_a, CV_64F);

    fx = a[0]; fy = a[4];
    cx = a[2]; cy = a[5];

    if( fixedAspectRatio )
        fx = fy*aspectRatio;

    Mat distCoeffs = _distCoeffs.getMat();
    int ktotal = 0;
    if( distCoeffs.data )
    {
        int kcn = distCoeffs.channels();
        ktotal = (int)distCoeffs.total()*kcn;
        if( (distCoeffs.rows != 1 && distCoeffs.cols != 1) ||
            (ktotal != 4 && ktotal != 5 && ktotal != 8 && ktotal != 12 && ktotal != 14))
            CV_Error( cv::Error::StsBadArg, cvDistCoeffErr );

        Mat _k(distCoeffs.size(), CV_64FC(kcn), k);
        distCoeffs.convertTo(_k, CV_64F);
        if(k[12] != 0 || k[13] != 0)
            detail::computeTiltProjectionMatrix(k[12], k[13], &matTilt, &dMatTiltdTauX, &dMatTiltdTauY);
    }

    if( _dpdr.needed() )
    {
        dpdr.create(count*2, 3, CV_64F);
        dpdr_p = dpdr.ptr<double>();
        dpdr_step = (int)dpdr.step1();
    }
    if( _dpdt.needed() )
    {
        dpdt.create(count*2, 3, CV_64F);
        dpdt_p = dpdt.ptr<double>();
        dpdt_step = (int)dpdt.step1();
    }
    if( _dpdf.needed() )
    {
        dpdf.create(count*2, 2, CV_64F);
        dpdf_p = dpdf.ptr<double>();
        dpdf_step = (int)dpdf.step1();
    }
    if( _dpdc.needed() )
    {
        dpdc.create(count*2, 2, CV_64F);
        dpdc_p = dpdc.ptr<double>();
        dpdc_step = (int)dpdc.step1();
    }
    if( _dpdk.needed() )
    {
        dpdk.create(count*2, ktotal, CV_64F);
        dpdk_p = dpdk.ptr<double>();
        dpdk_step = (int)dpdk.step1();
    }
    if( _dpdo.needed() )
    {
        dpdo = Mat::zeros(count*2, count*3, CV_64F);
        dpdo_p = dpdo.ptr<double>();
        dpdo_step = (int)dpdo.step1();
    }

    bool calc_derivatives = dpdr.data || dpdt.data || dpdf.data ||
                            dpdc.data || dpdk.data || dpdo.data;

    if (!calc_derivatives)
    {
        if (objpt_depth == CV_32F && ipoints.type() == CV_32F)
        {
            float rtMatrix[12] = { (float)R[0], (float)R[1], (float)R[2], (float)t[0],
                                (float)R[3], (float)R[4], (float)R[5], (float)t[1],
                                (float)R[6], (float)R[7], (float)R[8], (float)t[2] };

            cv_camera_intrinsics_pinhole_32f intr;
            intr.fx = (float)fx; intr.fy = (float)fy;
            intr.cx = (float)cx; intr.cy = (float)cy;
            intr.amt_k = 0; intr.amt_p = 0; intr.amt_s = 0; intr.use_tau = false;

            switch (ktotal)
            {
            case  0: break;
            case  4: // [k_1, k_2, p_1, p_2]
                intr.amt_k = 2; intr.amt_p = 2;
                break;
            case  5: // [k_1, k_2, p_1, p_2, k_3]
                intr.amt_k = 3; intr.amt_p = 2;
                break;
            case  8: // [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6]
                intr.amt_k = 6; intr.amt_p = 2;
                break;
            case 12: // [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4]
                intr.amt_k = 6; intr.amt_p = 2; intr.amt_s = 4;
                break;
            case 14: // [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, tau_x, tau_y]
                intr.amt_k = 6; intr.amt_p = 2; intr.amt_s = 4; intr.use_tau = true;
                break;
            default:
                CV_Error(cv::Error::StsInternal, "Wrong number of distortion coefficients");
            }

            intr.k[0] = (float)k[0];
            intr.k[1] = (float)k[1];
            intr.k[2] = (float)k[4];
            intr.k[3] = (float)k[5];
            intr.k[4] = (float)k[6];
            intr.k[5] = (float)k[7];

            intr.p[0] = (float)k[2];
            intr.p[1] = (float)k[3];

            for (int ctr = 0; ctr < 4; ctr++)
            {
                intr.s[ctr] = (float)k[8+ctr];
            }

            intr.tau_x = (float)k[12];
            intr.tau_y = (float)k[13];

            CALL_HAL(projectPoints, cv_hal_project_points_pinhole32f,
                     (float*)objectPoints.data, objectPoints.step, count,
                     (float*)ipoints.data, ipoints.step, rtMatrix, &intr);
        }

        if (objpt_depth == CV_64F && ipoints.type() == CV_64F)
        {
            double rtMatrix[12] = { R[0], R[1], R[2], t[0],
                                    R[3], R[4], R[5], t[1],
                                    R[6], R[7], R[8], t[2] };

            cv_camera_intrinsics_pinhole_64f intr;
            intr.fx = fx; intr.fy = fy;
            intr.cx = cx; intr.cy = cy;
            intr.amt_k = 0; intr.amt_p = 0; intr.amt_s = 0; intr.use_tau = false;

            switch (ktotal)
            {
            case  0: break;
            case  4: // [k_1, k_2, p_1, p_2]
                intr.amt_k = 2; intr.amt_p = 2;
                break;
            case  5: // [k_1, k_2, p_1, p_2, k_3]
                intr.amt_k = 3; intr.amt_p = 2;
                break;
            case  8: // [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6]
                intr.amt_k = 6; intr.amt_p = 2;
                break;
            case 12: // [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4]
                intr.amt_k = 6; intr.amt_p = 2; intr.amt_s = 4;
                break;
            case 14: // [k_1, k_2, p_1, p_2, k_3, k_4, k_5, k_6, s_1, s_2, s_3, s_4, tau_x, tau_y]
                intr.amt_k = 6; intr.amt_p = 2; intr.amt_s = 4; intr.use_tau = true;
                break;
            default:
                CV_Error(cv::Error::StsInternal, "Wrong number of distortion coefficients");
            }

            intr.k[0] = k[0];
            intr.k[1] = k[1];
            intr.k[2] = k[4];
            intr.k[3] = k[5];
            intr.k[4] = k[6];
            intr.k[5] = k[7];

            intr.p[0] = k[2];
            intr.p[1] = k[3];

            for (int ctr = 0; ctr < 4; ctr++)
            {
                intr.s[ctr] = k[8+ctr];
            }

            intr.tau_x = k[12];
            intr.tau_y = k[13];

            CALL_HAL(projectPoints, cv_hal_project_points_pinhole64f,
                     (double*)objectPoints.data, objectPoints.step, count,
                     (double*)ipoints.data, ipoints.step, rtMatrix, &intr);
        }
    }

    Mat matM(objectPoints.size(), CV_64FC(objpt_cn));
    objectPoints.convertTo(matM, CV_64F);
    ipoints.convertTo(_m, CV_64F);
    const Point3d* M = matM.ptr<Point3d>();
    Point2d* m = _m.ptr<Point2d>();

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
            if( dpdc.data )
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
                    dMatTilt(row,col) = matTilt(row,col)*vecTilt(2) - matTilt(2,col)*vecTilt(row);
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
                if( dpdk.cols > 2 )
                {
                    dXdYd = dMatTilt*Vec2d(a1, a3);
                    dpdk_p[2] = fx*dXdYd(0);
                    dpdk_p[dpdk_step+2] = fy*dXdYd(1);
                    dXdYd = dMatTilt*Vec2d(a2, a1);
                    dpdk_p[3] = fx*dXdYd(0);
                    dpdk_p[dpdk_step+3] = fy*dXdYd(1);
                    if( dpdk.cols > 4 )
                    {
                        dXdYd = dMatTilt*Vec2d(x*icdist2*r6, y*icdist2*r6);
                        dpdk_p[4] = fx*dXdYd(0);
                        dpdk_p[dpdk_step+4] = fy*dXdYd(1);

                        if( dpdk.cols > 5 )
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
                            if( dpdk.cols > 8 )
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
                                if( dpdk.cols > 12 )
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

    _m.convertTo(_imagePoints, objpt_depth);

    int depth = CV_64F;//cameraMatrix.depth();
    if( _dpdr.needed() )
        dpdr.convertTo(_dpdr, depth);

    if( _dpdt.needed() )
        dpdt.convertTo(_dpdt, depth);

    if( _dpdf.needed() )
        dpdf.convertTo(_dpdf, depth);

    if( _dpdc.needed() )
        dpdc.convertTo(_dpdc, depth);

    if( _dpdk.needed() )
        dpdk.convertTo(_dpdk, depth);

    if( _dpdo.needed() )
        dpdo.convertTo(_dpdo, depth);
}

cv::Vec3d cv::RQDecomp3x3( InputArray _Marr,
                   OutputArray _Rarr,
                   OutputArray _Qarr,
                   OutputArray _Qx,
                   OutputArray _Qy,
                   OutputArray _Qz )
{
    CV_INSTRUMENT_REGION();

    Matx33d M, Q;
    double z, c, s;
    Mat Mmat = _Marr.getMat();
    int depth = Mmat.depth();
    Mmat.convertTo(M, CV_64F);

    /* Find Givens rotation Q_x for x axis (left multiplication). */
    /*
         ( 1  0  0 )
    Qx = ( 0  c  s ), c = m33/sqrt(m32^2 + m33^2), s = m32/sqrt(m32^2 + m33^2)
         ( 0 -s  c )
    */
    s = std::abs(M(2, 1)) > DBL_EPSILON ? M(2, 1): 0.;
    c = std::abs(M(2, 1)) > DBL_EPSILON ? M(2, 2): 1.;
    z = 1./std::sqrt(c * c + s * s);
    c *= z;
    s *= z;

    Matx33d Qx(1, 0, 0, 0, c, s, 0, -s, c);
    Matx33d R = M*Qx;

    assert(fabs(R(2, 1)) < FLT_EPSILON);
    R(2, 1) = 0;

    /* Find Givens rotation for y axis. */
    /*
         ( c  0 -s )
    Qy = ( 0  1  0 ), c = m33/sqrt(m31^2 + m33^2), s = -m31/sqrt(m31^2 + m33^2)
         ( s  0  c )
    */
    s = std::abs(R(2, 0)) > DBL_EPSILON ? -R(2, 0): 0.;
    c = std::abs(R(2, 0)) > DBL_EPSILON ? R(2, 2): 1.;
    z = 1./std::sqrt(c * c + s * s);
    c *= z;
    s *= z;

    Matx33d Qy(c, 0, -s, 0, 1, 0, s, 0, c);
    M = R*Qy;

    CV_Assert(fabs(M(2, 0)) < FLT_EPSILON);
    M(2, 0) = 0;

    /* Find Givens rotation for z axis. */
    /*
         ( c  s  0 )
    Qz = (-s  c  0 ), c = m22/sqrt(m21^2 + m22^2), s = m21/sqrt(m21^2 + m22^2)
         ( 0  0  1 )
    */

    s = std::abs(M(1, 0)) > DBL_EPSILON ? M(1, 0): 0.;
    c = std::abs(M(1, 0)) > DBL_EPSILON ? M(1, 1): 1.;
    z = 1./std::sqrt(c * c + s * s);
    c *= z;
    s *= z;

    Matx33d Qz(c, s, 0, -s, c, 0, 0, 0, 1);
    R = M*Qz;

    CV_Assert(fabs(R(1, 0)) < FLT_EPSILON);
    R(1, 0) = 0;

    // Solve the decomposition ambiguity.
    // Diagonal entries of R, except the last one, shall be positive.
    // Further rotate R by 180 degree if necessary
    if( R(0, 0) < 0 )
    {
        if( R(1, 1) < 0 )
        {
            // rotate around z for 180 degree, i.e. a rotation matrix of
            // [-1,  0,  0],
            // [ 0, -1,  0],
            // [ 0,  0,  1]
            R(0, 0) *= -1;
            R(0, 1) *= -1;
            R(1, 1) *= -1;

            Qz(0, 0) *= -1;
            Qz(0, 1) *= -1;
            Qz(1, 0) *= -1;
            Qz(1, 1) *= -1;
        }
        else
        {
            // rotate around y for 180 degree, i.e. a rotation matrix of
            // [-1,  0,  0],
            // [ 0,  1,  0],
            // [ 0,  0, -1]
            R(0, 0) *= -1;
            R(0, 2) *= -1;
            R(1, 2) *= -1;
            R(2, 2) *= -1;

            Qz = Qz.t();

            Qy(0, 0) *= -1;
            Qy(0, 2) *= -1;
            Qy(2, 0) *= -1;
            Qy(2, 2) *= -1;
        }
    }
    else if( R(1, 1) < 0 )
    {
        // ??? for some reason, we never get here ???

        // rotate around x for 180 degree, i.e. a rotation matrix of
        // [ 1,  0,  0],
        // [ 0, -1,  0],
        // [ 0,  0, -1]
        R(0, 1) *= -1;
        R(0, 2) *= -1;
        R(1, 1) *= -1;
        R(1, 2) *= -1;
        R(2, 2) *= -1;

        Qz = Qz.t();
        Qy = Qy.t();

        Qx(1, 1) *= -1;
        Qx(1, 2) *= -1;
        Qx(2, 1) *= -1;
        Qx(2, 2) *= -1;
    }

    // calculate the euler angle
    Vec3d eulerAngles(
        std::acos(Qx(1, 1)) * (Qx(1, 2) >= 0 ? 1 : -1) * (180.0 / CV_PI),
        std::acos(Qy(0, 0)) * (Qy(2, 0) >= 0 ? 1 : -1) * (180.0 / CV_PI),
        std::acos(Qz(0, 0)) * (Qz(0, 1) >= 0 ? 1 : -1) * (180.0 / CV_PI));

    /* Calculate orthogonal matrix. */
    /*
    Q = QzT * QyT * QxT
    */
    M = Qz.t()*Qy.t();
    Q = M*Qx.t();

    /* Save R and Q matrices. */
    Mat(R).convertTo(_Rarr, depth);
    Mat(Q).convertTo(_Qarr, depth);

    if(_Qx.needed())
        Mat(Qx).convertTo(_Qx, depth);
    if(_Qy.needed())
        Mat(Qy).convertTo(_Qy, depth);
    if(_Qz.needed())
        Mat(Qz).convertTo(_Qz, depth);
    return eulerAngles;
}

void cv::decomposeProjectionMatrix( InputArray _projMatrix, OutputArray _cameraMatrix,
                                    OutputArray _rotMatrix, OutputArray _transVect,
                                    OutputArray _rotMatrixX, OutputArray _rotMatrixY,
                                    OutputArray _rotMatrixZ, OutputArray _eulerAngles )
{
    CV_INSTRUMENT_REGION();

    Mat projMatrix = _projMatrix.getMat();
    int depth = projMatrix.depth();
    Matx34d P;
    projMatrix.convertTo(P, CV_64F);
    Matx44d Px(P(0, 0), P(0, 1), P(0, 2), P(0, 3),
               P(1, 0), P(1, 1), P(1, 2), P(1, 3),
               P(2, 0), P(2, 1), P(2, 2), P(2, 3),
               0, 0, 0, 0), U, Vt;
    Matx41d W;
    SVDecomp(Px, W, U, Vt, SVD::MODIFY_A);
    Vec4d t(Vt(3, 0), Vt(3, 1), Vt(3, 2), Vt(3, 3));
    Matx33d M(P(0, 0), P(0, 1), P(0, 2),
              P(1, 0), P(1, 1), P(1, 2),
              P(2, 0), P(2, 1), P(2, 2));
    Mat(t).convertTo(_transVect, depth);
    Vec3d eulerAngles = RQDecomp3x3(M, _cameraMatrix, _rotMatrix, _rotMatrixX, _rotMatrixY, _rotMatrixZ);
    if (_eulerAngles.needed())
        Mat(eulerAngles).convertTo(_eulerAngles, depth);
}

void cv::findExtrinsicCameraParams2( const Mat& objectPoints,
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
    else {
        convertPointsFromHomogeneous(objectPoints, matM);
        matM.convertTo(matM, CV_64F);
    }
    if (imagePoints.checkVector(2) > 0)
        imagePoints.convertTo(_m, CV_64F);
    else {
        convertPointsFromHomogeneous(imagePoints, _m);
        _m.convertTo(_m, CV_64F);
    }
    A.convertTo(matA, CV_64F);

    CV_Assert((count >= 4) || (count == 3 && useExtrinsicGuess)); // it is unsafe to call LM optimisation without an extrinsic guess in the case of 3 points. This is because there is no guarantee that it will converge on the correct solution.

    // normalize image points
    // (unapply the intrinsic matrix transformation and distortion)
    undistortPoints(_m, _mn, matA, distCoeffs, Mat(), _Ar);

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

            Mat matH = findHomography(_Mxy, _mn);

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

                Rodrigues( matH, _r );
                Rodrigues( _r, matH );
                _t += matH*T_transform;
                matR = matH * R_transform;
            }
            else
            {
                setIdentity(matR);
                _t.setTo(0);
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
            Rodrigues(matR, _r);
        }
    }

    matM = matM.reshape(3, 1);
    _mn = _mn.reshape(2, 1);

    // refine extrinsic parameters using iterative algorithm
#if 0
    // The C++ LMSolver is not as good as CvLevMarq to pass the tests, maybe due to _completeSymmFlag in CvLevMarq.
    class RefineLMCallback CV_FINAL : public LMSolver::Callback
    {
    public:
    RefineLMCallback(const Mat &matM, const Mat &_m, const Mat &matA, const Mat& distCoeffs) : matM_(matM), _m_(_m), matA_(matA), distCoeffs_(distCoeffs) {
    }
    bool compute(InputArray param_, OutputArray _err, OutputArray _Jac) const CV_OVERRIDE
    {
        const Mat& objpt = matM_;
        const Mat& imgpt = _m_;
        const Mat& cameraMatrix = matA_;
        Mat x = param_.getMat();
        CV_Assert((x.cols == 1 || x.rows == 1) && x.total() == 6 && x.type() == CV_64F);
        double* pdata = x.ptr<double>();
        Mat rv(3, 1, CV_64F, pdata);
        Mat tv(3, 1, CV_64F, pdata + 3);
        int errCount = objpt.rows + objpt.cols - 1;
        _err.create(errCount * 2, 1, CV_64F);
        Mat err = _err.getMat();
        err = err.reshape(2, errCount);
        if (_Jac.needed())
        {
            _Jac.create(errCount * 2, 6, CV_64F);
            Mat Jac = _Jac.getMat();
            Mat dpdr = Jac.colRange(0, 3);
            Mat dpdt = Jac.colRange(3, 6);
            projectPoints(objpt, rv, tv, cameraMatrix, distCoeffs_,
                err, dpdr, dpdt, noArray(), noArray(), noArray(), noArray());
        }
        else
        {
            projectPoints(objpt, rv, tv, cameraMatrix, distCoeffs_, err);
        }
        err = err - (imgpt.rows == 1 ? imgpt.t() : imgpt);
        err = err.reshape(1, 2 * errCount);
        return true;
    };
    private:
    const Mat &matM_, &_m_, &matA_, &distCoeffs_;
    };

    LMSolver::create(makePtr<RefineLMCallback>(matM, _m, matA, distCoeffs), max_iter, FLT_EPSILON)->run(_param);
#else
    CvLevMarq solver( 6, count*2, cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER,max_iter,FLT_EPSILON), true);
    _param.copyTo(cvarrToMat(solver.param));

    for(;;)
    {
        CvMat *matJ = 0, *_err = 0;
        const CvMat *__param = 0;
        bool proceed = solver.update( __param, matJ, _err );
        cvarrToMat(__param).copyTo(_param );
        if( !proceed || !_err )
            break;
        int errCount = matM.rows + matM.cols - 1;
        Mat err = cvarrToMat(_err);
        err = err.reshape(2, errCount);
        if( matJ )
        {
            Mat Jac = cvarrToMat(matJ);
            Mat dpdr = Jac.colRange(0, 3);
            Mat dpdt = Jac.colRange(3, 6);
            projectPoints(matM, _r, _t, matA, distCoeffs,
                err, dpdr, dpdt, noArray(), noArray(), noArray(), noArray());
        }
        else
        {
            projectPoints(matM, _r, _t, matA, distCoeffs, err);
        }
        subtract(err, _m.rows == 1 ? _m.t() : _m, err);
        cvReshape( _err, _err, 1, 2*count );
    }
    cvarrToMat(solver.param).copyTo(_param );
#endif

    _param.rowRange(0, 3).convertTo(rvec, rvec.depth());
    _param.rowRange(3, 6).convertTo(tvec, tvec.depth());
}

void cv::projectPoints( InputArray _opoints,
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

    CV_Assert( _ipoints.needed() );

    double dc0buf[5]={0};
    Mat dc0(5,1,CV_64F,dc0buf);
    Mat distCoeffs = _distCoeffs.getMat();
    if( distCoeffs.empty() )
        distCoeffs = dc0;
    int ndistCoeffs = distCoeffs.rows + distCoeffs.cols - 1;

    if( _jacobian.needed() )
    {
        _jacobian.create(npoints*2, 3+3+2+2+ndistCoeffs, CV_64F);
        Mat jacobian = _jacobian.getMat();
        Mat dpdr = jacobian.colRange(0, 3);
        Mat dpdt = jacobian.colRange(3, 6);
        Mat dpdf = jacobian.colRange(6, 8);
        Mat dpdc = jacobian.colRange(8, 10);
        Mat dpdk = jacobian.colRange(10, 10+ndistCoeffs);

        projectPoints(opoints, _rvec, _tvec, _cameraMatrix, distCoeffs, _ipoints,
                      dpdr, dpdt, dpdf, dpdc, dpdk, noArray(), aspectRatio);
    }
    else
    {
        projectPoints(opoints, _rvec, _tvec, _cameraMatrix, distCoeffs, _ipoints,
                      noArray(), noArray(), noArray(), noArray(), noArray(), noArray(), aspectRatio);
    }
}

void cv::getUndistortRectangles(InputArray _cameraMatrix, InputArray _distCoeffs,
              InputArray R, InputArray newCameraMatrix, Size imgSize,
              Rect_<double>& inner, Rect_<double>& outer )
{
    const int N = 9;
    int x, y, k;
    Mat _pts(1, 4*(N-2), CV_64FC2);
    Point2d* pts = _pts.ptr<Point2d>();

    // generate a grid of points across the image to estimate the distortion deformation
    double stepX = (imgSize.width - 1) / static_cast<double>(N - 1);
    double stepY = (imgSize.height - 1) / static_cast<double>(N - 1);
    for( y = k = 0; y < N; y++ )
    {
        for( x = 0; x < N; x++ )
        {
            if (x != 0 && x != N - 1 && y != 0 && y != N - 1)
            {
                // skip all points except those on the image border, because inner grid points
                // have no influence on the two deformation rectangles that are calculated below
                continue;
            }
            if ((x == 0 || x == N - 1) && (y == 0 || y == N - 1))
            {
                // skip corners, because undistortPoints is likely to fail and return the same
                // value
                continue;
            }
            pts[k++] = Point2d(x * stepX, y * stepY);
        }
    }

    undistortPoints(_pts, _pts, _cameraMatrix, _distCoeffs, R, newCameraMatrix);

    double iX0=-FLT_MAX, iX1=FLT_MAX, iY0=-FLT_MAX, iY1=FLT_MAX;
    double oX0=FLT_MAX, oX1=-FLT_MAX, oY0=FLT_MAX, oY1=-FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for( y = k = 0; y < N; y++ )
    {
        for( x = 0; x < N; x++ )
        {
            if (x != 0 && x != N - 1 && y != 0 && y != N - 1)
            {
                continue;
            }
            if ((x == 0 || x == N - 1) && (y == 0 || y == N - 1))
            {
                continue;
            }

            Point2d p = pts[k++];
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
    }
    inner = Rect_<double>(iX0, iY0, iX1-iX0, iY1-iY0);
    outer = Rect_<double>(oX0, oY0, oX1-oX0, oY1-oY0);
}

cv::Mat cv::getOptimalNewCameraMatrix( InputArray _cameraMatrix, InputArray _distCoeffs,
                                  Size imgSize, double alpha, Size newImgSize,
                                  Rect* validPixROI, bool centerPrincipalPoint )
{
    Rect_<double> inner, outer;
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
            inner = cv::Rect_<double>((double)((inner.x - cx0)*s + cx),
                                      (double)((inner.y - cy0)*s + cy),
                                      (double)(inner.width*s),
                                      (double)(inner.height*s));
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
    M.convertTo(M, cameraMatrix.type());

    return M;
}

/* End of file. */
