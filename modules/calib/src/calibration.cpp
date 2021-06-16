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
#include "distortion_model.hpp"
#include <stdio.h>
#include <iterator>
#include <iostream>

/*
    This is straight-forward port v3 of Matlab calibration engine by Jean-Yves Bouguet
    that is (in a large extent) based on the paper:
    Z. Zhang. "A flexible new technique for camera calibration".
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
    The 1st initial port was done by Valery Mosyagin.
*/

namespace cv {

static void initIntrinsicParams2D( const Mat& objectPoints,
                         const Mat& imagePoints, const Mat& npoints,
                         Size imageSize, OutputArray cameraMatrix,
                         double aspectRatio )
{
    int i, j, pos;
    double a[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    double H[9] = {0}, f[2] = {0};
    Mat _a( 3, 3, CV_64F, a );
    Mat matH( 3, 3, CV_64F, H );
    Mat _f( 2, 1, CV_64F, f );

    CV_Assert(npoints.type() == CV_32SC1 && (npoints.rows == 1 || npoints.cols == 1) && npoints.isContinuous());
    int nimages = npoints.rows + npoints.cols - 1;

    CV_Assert( objectPoints.type() == CV_32FC3 ||
               objectPoints.type() == CV_64FC3 );
    CV_Assert( imagePoints.type() == CV_32FC2 ||
               imagePoints.type() == CV_64FC2 );

    if( objectPoints.rows != 1 || imagePoints.rows != 1 )
        CV_Error( CV_StsBadSize, "object points and image points must be a single-row matrices" );

    Mat matA( 2*nimages, 2, CV_64F );
    Mat _b( 2*nimages, 1, CV_64F );
    a[2] = (!imageSize.width) ? 0.5 : (imageSize.width - 1)*0.5;
    a[5] = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;
    Mat _allH( nimages, 9, CV_64F );

    // extract vanishing points in order to obtain initial value for the focal length
    for( i = 0, pos = 0; i < nimages; i++ )
    {
        double* Ap = (double*)matA.data + i*4;
        double* bp = (double*)_b.data + i*2;
        int ni = npoints.at<int>(i);
        double h[3], v[3], d1[3], d2[3];
        double n[4] = {0,0,0,0};
        Mat matM = objectPoints.colRange(pos, pos + ni);
        Mat _m = imagePoints.colRange(pos, pos + ni);
        pos += ni;

        Mat matH0 = findHomography(matM, _m);
        CV_Assert(matH0.size() == Size(3, 3));
        matH0.convertTo(matH, CV_64F);

        H[0] -= H[6]*a[2]; H[1] -= H[7]*a[2]; H[2] -= H[8]*a[2];
        H[3] -= H[6]*a[5]; H[4] -= H[7]*a[5]; H[5] -= H[8]*a[5];

        for( j = 0; j < 3; j++ )
        {
            double t0 = H[j*3], t1 = H[j*3+1];
            h[j] = t0; v[j] = t1;
            d1[j] = (t0 + t1)*0.5;
            d2[j] = (t0 - t1)*0.5;
            n[0] += t0*t0; n[1] += t1*t1;
            n[2] += d1[j]*d1[j]; n[3] += d2[j]*d2[j];
        }

        for( j = 0; j < 4; j++ )
            n[j] = 1./std::sqrt(n[j]);

        for( j = 0; j < 3; j++ )
        {
            h[j] *= n[0]; v[j] *= n[1];
            d1[j] *= n[2]; d2[j] *= n[3];
        }

        Ap[0] = h[0]*v[0]; Ap[1] = h[1]*v[1];
        Ap[2] = d1[0]*d2[0]; Ap[3] = d1[1]*d2[1];
        bp[0] = -h[2]*v[2]; bp[1] = -d1[2]*d2[2];
    }

    solve(matA, _b, _f, DECOMP_NORMAL + DECOMP_SVD);
    CV_Assert((double*)_f.data == f);
    a[0] = std::sqrt(fabs(1./f[0]));
    a[4] = std::sqrt(fabs(1./f[1]));
    if( aspectRatio != 0 )
    {
        double tf = (a[0] + a[4])/(aspectRatio + 1.);
        a[0] = aspectRatio*tf;
        a[4] = tf;
    }
    _a.copyTo(cameraMatrix);
}

static void subMatrix(const Mat& src, Mat& dst,
                      const std::vector<uchar>& cols,
                      const std::vector<uchar>& rows)
{
    CV_Assert(src.type() == CV_64F && dst.type() == CV_64F);
    int m = (int)rows.size(), n = (int)cols.size();
    int i1 = 0, j1 = 0;
    const uchar* colsdata = cols.empty() ? 0 : &cols[0];
    for(int i = 0; i < m; i++)
    {
        if(rows[i])
        {
            const double* srcptr = src.ptr<double>(i);
            double* dstptr = dst.ptr<double>(i1++);

            for(int j = j1 = 0; j < n; j++)
            {
                if(colsdata[j])
                    dstptr[j1++] = srcptr[j];
            }
        }
    }
}

static void cameraCalcJErr(const Mat& objectPoints, const Mat& imagePoints,
                           const Mat& npoints, Mat& allErrors,
                           Mat& _param, Mat* _JtErr, Mat* _JtJ, double* _errnorm,
                           double aspectRatio, Mat* perViewErrors,
                           int flags, bool optimizeObjPoints)
{
    const int NINTRINSIC = CALIB_NINTRINSIC;
    int ni = 0, nimages = (int)npoints.total();
    double k[14] = {0};
    Mat _k(14, 1, CV_64F, k);
    double* param = _param.ptr<double>();
    int nparams = (int)_param.total();
    bool calcJ = _JtErr != 0;
    int ni0 = npoints.at<int>(0);
    Mat _Je(ni0*2, 6, CV_64F), _Ji(ni0*2, NINTRINSIC, CV_64F), _Jo, _err(ni*2, 1, CV_64F);

    if( flags & CALIB_FIX_ASPECT_RATIO )
    {
        param[0] = param[1]*aspectRatio;
        //pparam[0] = pparam[1]*aspectRatio;
    }

    Matx33d A(param[0], 0, param[2],
              0, param[1], param[3],
              0, 0, 1);
    std::copy(param + 4, param + 4 + 14, k);

    if (_JtJ)
        _JtJ->setZero();
    if (_JtErr)
        _JtErr->setZero();

    if(optimizeObjPoints)
        _Jo.create(ni0*2, ni0*3, CV_64F);

    double reprojErr = 0;
    int maxPoints = 0;
    for( int i = 0; i < nimages; i++ )
        maxPoints = max(maxPoints, npoints.at<int>(i));

    for( int i = 0, pos = 0; i < nimages; i++, pos += ni )
    {
        ni = npoints.at<int>(i);
        Mat _ri = _param.rowRange(NINTRINSIC + i*6, NINTRINSIC + i*6 + 3);
        Mat _ti = _param.rowRange(NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6);

        Mat _Mi = objectPoints.colRange(pos, pos + ni);
        if( optimizeObjPoints )
        {
            _Mi = _param.rowRange(NINTRINSIC + nimages * 6,
                       NINTRINSIC + nimages * 6 + ni * 3);
            _Mi = _Mi.reshape(3, 1);
        }
        Mat _mi = imagePoints.colRange(pos, pos + ni);
        Mat _me = allErrors.colRange(pos, pos + ni);

        _Je.resize(ni*2);
        _Ji.resize(ni*2);
        _err.resize(ni*2);
        if (optimizeObjPoints)
            _Jo.resize(ni*2);
        Mat _mp = _err.reshape(2, 1);

        if( calcJ )
        {
            Mat _dpdr = _Je.colRange(0, 3);
            Mat _dpdt = _Je.colRange(3, 6);
            Mat _dpdf = _Ji.colRange(0, 2);
            Mat _dpdc = _Ji.colRange(2, 4);
            Mat _dpdk = _Ji.colRange(4, NINTRINSIC);
            Mat _dpdo = _Jo.empty() ? Mat() : _Jo.colRange(0, ni * 3);
            double* dpdr_p = _dpdr.ptr<double>();
            double* dpdt_p = _dpdt.ptr<double>();
            double* dpdf_p = _dpdf.ptr<double>();
            double* dpdc_p = _dpdc.ptr<double>();
            double* dpdk_p = _dpdk.ptr<double>();
            double* dpdo_p = _dpdo.ptr<double>();

            projectPoints(_Mi, _ri, _ti, A, _k, _mp, _dpdr, _dpdt,
                          (flags & CALIB_FIX_FOCAL_LENGTH) ? _OutputArray() : _OutputArray(_dpdf),
                          (flags & CALIB_FIX_PRINCIPAL_POINT) ? _OutputArray() : _OutputArray(_dpdc),
                          _dpdk, _Jo.empty() ? _OutputArray() : _OutputArray(_dpdo),
                          (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0.);
            CV_Assert(_mp.ptr<double>() == _err.ptr<double>() &&
                      dpdr_p == _dpdr.ptr<double>() && dpdt_p == _dpdt.ptr<double>() &&
                      dpdf_p == _dpdf.ptr<double>() && dpdc_p == _dpdc.ptr<double>() &&
                      dpdk_p == _dpdk.ptr<double>() && dpdo_p == _dpdo.ptr<double>());
        }
        else
            projectPoints( _Mi, _ri, _ti, A, _k, _mp,
                           noArray(), noArray(), noArray(),
                           noArray(), noArray(), noArray(), 0.);

        subtract( _mp, _mi, _mp );
        _mp.copyTo(_me);

        if( calcJ )
        {
            Mat JtJ = *_JtJ, JtErr = *_JtErr;

            // see HZ: (A6.14) for details on the structure of the Jacobian
            JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += _Ji.t() * _Ji;
            JtJ(Rect(NINTRINSIC + i * 6, NINTRINSIC + i * 6, 6, 6)) = _Je.t() * _Je;
            JtJ(Rect(NINTRINSIC + i * 6, 0, 6, NINTRINSIC)) = _Ji.t() * _Je;
            if( optimizeObjPoints )
            {
                JtJ(Rect(NINTRINSIC + nimages * 6, 0, maxPoints * 3, NINTRINSIC)) += _Ji.t() * _Jo;
                JtJ(Rect(NINTRINSIC + nimages * 6, NINTRINSIC + i * 6, maxPoints * 3, 6))
                    += _Je.t() * _Jo;
                JtJ(Rect(NINTRINSIC + nimages * 6, NINTRINSIC + nimages * 6, maxPoints * 3, maxPoints * 3))
                    += _Jo.t() * _Jo;
            }

            JtErr.rowRange(0, NINTRINSIC) += _Ji.t() * _err;
            JtErr.rowRange(NINTRINSIC + i * 6, NINTRINSIC + (i + 1) * 6) = _Je.t() * _err;
            if( optimizeObjPoints )
            {
                JtErr.rowRange(NINTRINSIC + nimages * 6, nparams) += _Jo.t() * _err;
            }
        }

        double viewErr = norm(_err, NORM_L2SQR);
        /*if (i == 0 || i == nimages-1) {
            printf("image %d.", i);
            for(int j = 0; j < 10; j++) {
                printf(" %.2g", _err.at<double>(j));
            }
            printf("\n");
        }*/

        if( perViewErrors )
            perViewErrors->at<double>(i) = std::sqrt(viewErr / ni);

        reprojErr += viewErr;
    }

    if(_errnorm)
        *_errnorm = reprojErr;
}

static double calibrateCameraInternal( const Mat& objectPoints,
                    const Mat& imagePoints, const Mat& npoints,
                    Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs,
                    Mat* rvecs, Mat* tvecs, Mat* newObjPoints, Mat* stdDevs,
                    Mat* perViewErrors, int flags, const TermCriteria& termCrit )
{
    const int NINTRINSIC = CALIB_NINTRINSIC;

    Matx33d A;
    double k[14] = {0};
    Mat matA(3, 3, CV_64F, A.val);
    int i, maxPoints = 0, ni = 0, pos, total = 0, nparams, cn;
    double aspectRatio = 0.;
    int nimages = npoints.checkVector(1, CV_32S);
    CV_Assert(nimages >= 1);
    int ndistCoeffs = (int)distCoeffs.total();
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints.at<int>(0) - 1;

    // 0. check the parameters & allocate buffers
    if( imageSize.width <= 0 || imageSize.height <= 0 )
        CV_Error( CV_StsOutOfRange, "image width and height must be positive" );

    if(flags & CALIB_TILTED_MODEL)
    {
        //when the tilted sensor model is used the distortion coefficients matrix must have 14 parameters
        if (ndistCoeffs != 14)
            CV_Error( CV_StsBadArg, "The tilted sensor model must have 14 parameters in the distortion matrix" );
    }
    else
    {
        //when the thin prism model is used the distortion coefficients matrix must have 12 parameters
        if(flags & CALIB_THIN_PRISM_MODEL)
            if (ndistCoeffs != 12)
                CV_Error( CV_StsBadArg, "Thin prism model must have 12 parameters in the distortion matrix" );
    }

    if( rvecs )
    {
        cn = rvecs->channels();
        CV_Assert(rvecs->depth() == CV_32F || rvecs->depth() == CV_64F);
        CV_Assert(rvecs->rows == nimages);
        CV_Assert((rvecs->rows == nimages && (rvecs->cols*cn == 3 || rvecs->cols*cn == 3)) ||
                  (rvecs->rows == 1 && rvecs->cols == nimages && cn == 3));
    }

    if( tvecs )
    {
        cn = tvecs->channels();
        CV_Assert(tvecs->depth() == CV_32F || tvecs->depth() == CV_64F);
        CV_Assert(tvecs->rows == nimages);
        CV_Assert((tvecs->rows == nimages && tvecs->cols*cn == 3) ||
                  (tvecs->rows == 1 && tvecs->cols == nimages && cn == 3));
    }

    CV_Assert(cameraMatrix.type() == CV_32F || cameraMatrix.type() == CV_64F);
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);

    CV_Assert(distCoeffs.type() == CV_32F || distCoeffs.type() == CV_64F);
    CV_Assert(distCoeffs.rows == 1 || distCoeffs.cols == 1);
    CV_Assert(ndistCoeffs == 4 || ndistCoeffs == 5 || ndistCoeffs == 8 ||
              ndistCoeffs == 12 || ndistCoeffs == 14);

    for( i = 0; i < nimages; i++ )
    {
        ni = npoints.at<int>(i);
        if( ni < 4 )
        {
            CV_Error_( CV_StsOutOfRange, ("The number of points in the view #%d is < 4", i));
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    if( newObjPoints )
    {
        cn = newObjPoints->channels();
        CV_Assert(newObjPoints->depth() == CV_32F || newObjPoints->depth() == CV_64F);
        CV_Assert(rvecs->rows == nimages);
        CV_Assert((newObjPoints->rows == maxPoints && newObjPoints->cols*cn == 3) ||
                  (newObjPoints->rows == 1 && newObjPoints->cols == maxPoints && cn == 3));
    }

    if( stdDevs )
    {
        cn = stdDevs->channels();
        CV_Assert(stdDevs->depth() == CV_32F || stdDevs->depth() == CV_64F);
        int nstddev = nimages*6 + NINTRINSIC + (releaseObject ? maxPoints*3 : 0);

        CV_Assert((stdDevs->rows == nstddev && stdDevs->cols*cn == 1) ||
                  (stdDevs->rows == 1 && stdDevs->cols == nstddev && cn == 1));
    }

    Mat matM( 1, total, CV_64FC3 );
    Mat _m( 1, total, CV_64FC2 );
    Mat allErrors(1, total, CV_64FC2);

    if(objectPoints.channels() == 3)
        objectPoints.convertTo(matM, CV_64F);
    else
        convertPointsToHomogeneous(objectPoints, matM, CV_64F);

    if(imagePoints.channels() == 2)
        imagePoints.convertTo(_m, CV_64F);
    else
        convertPointsFromHomogeneous(imagePoints, _m, CV_64F);

    nparams = NINTRINSIC + nimages*6;
    if( releaseObject )
        nparams += maxPoints * 3;

    Mat _k( distCoeffs.rows, distCoeffs.cols, CV_64F, k);
    if( distCoeffs.total() < 8 )
    {
        if( distCoeffs.total() < 5 )
            flags |= CALIB_FIX_K3;
        flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
    }
    const double minValidAspectRatio = 0.01;
    const double maxValidAspectRatio = 100.0;

    cameraMatrix.convertTo(matA, CV_64F);

    // 1. initialize intrinsic parameters & LM solver
    if( flags & CALIB_USE_INTRINSIC_GUESS )
    {
        if( A(0, 0) <= 0 || A(1, 1) <= 0 )
            CV_Error( CV_StsOutOfRange, "Focal length (fx and fy) must be positive" );
        if( A(0, 2) < 0 || A(0, 2) >= imageSize.width ||
            A(1, 2) < 0 || A(1, 2) >= imageSize.height )
            CV_Error( CV_StsOutOfRange, "Principal point must be within the image" );
        if( fabs(A(0, 1)) > 1e-5 )
            CV_Error( CV_StsOutOfRange, "Non-zero skew is not supported by the function" );
        if( fabs(A(1, 0)) > 1e-5 || fabs(A(2, 0)) > 1e-5 ||
            fabs(A(2, 1)) > 1e-5 || fabs(A(2,2)-1) > 1e-5 )
            CV_Error( CV_StsOutOfRange,
                "The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape" );
        A(0, 1) = A(1, 0) = A(2, 0) = A(2, 1) = 0.;
        A(2, 2) = 1.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = A(0, 0)/A(1, 1);

            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( CV_StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        distCoeffs.convertTo(_k, CV_64F);
    }
    else
    {
        Scalar mean, sdv;
        meanStdDev(matM, mean, sdv);
        if( fabs(mean[2]) > 1e-5 || fabs(sdv[2]) > 1e-5 )
            CV_Error( CV_StsBadArg,
            "For non-planar calibration rigs the initial intrinsic matrix must be specified" );
        for( i = 0; i < total; i++ )
            matM.at<Point3d>(i).z = 0.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = A(0, 0);
            aspectRatio /= A(1, 1);
            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( CV_StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        initIntrinsicParams2D( matM, _m, npoints, imageSize, A, aspectRatio );
    }

    //std::cout << "A0: " << A << std::endl;
    //std::cout << "dist0:" << _k << std::endl;

    Mat _Ji( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
    Mat _Je( maxPoints*2, 6, CV_64FC1 );
    Mat _err( maxPoints*2, 1, CV_64FC1 );
    Mat param0( nparams, 1, CV_64FC1 );
    Mat mask0 = Mat::ones( nparams, 1, CV_8UC1 );

    const bool allocJo = stdDevs || releaseObject;
    Mat _Jo = allocJo ? Mat( maxPoints*2, maxPoints*3, CV_64FC1, Scalar(0) ) : Mat();
    int solveMethod = DECOMP_EIG;

    if(flags & CALIB_USE_LU) {
        solveMethod = DECOMP_LU;
    }
    else if(flags & CALIB_USE_QR) {
        solveMethod = DECOMP_QR;
    }

    double* param = param0.ptr<double>();
    uchar* mask = mask0.ptr<uchar>();

    param[0] = A(0, 0); param[1] = A(1, 1); param[2] = A(0, 2); param[3] = A(1, 2);
    std::copy(k, k + 14, param + 4);

    if(flags & CALIB_FIX_ASPECT_RATIO)
        mask[0] = 0;
    if( flags & CALIB_FIX_FOCAL_LENGTH )
        mask[0] = mask[1] = 0;
    if( flags & CALIB_FIX_PRINCIPAL_POINT )
        mask[2] = mask[3] = 0;
    if( flags & CALIB_ZERO_TANGENT_DIST )
    {
        param[6] = param[7] = 0;
        mask[6] = mask[7] = 0;
    }
    if( !(flags & CALIB_RATIONAL_MODEL) )
        flags |= CALIB_FIX_K4 + CALIB_FIX_K5 + CALIB_FIX_K6;
    if( !(flags & CALIB_THIN_PRISM_MODEL))
        flags |= CALIB_FIX_S1_S2_S3_S4;
    if( !(flags & CALIB_TILTED_MODEL))
        flags |= CALIB_FIX_TAUX_TAUY;

    mask[ 4] = !(flags & CALIB_FIX_K1);
    mask[ 5] = !(flags & CALIB_FIX_K2);
    if( flags & CALIB_FIX_TANGENT_DIST )
        mask[6]  = mask[7]  = 0;
    mask[ 8] = !(flags & CALIB_FIX_K3);
    mask[ 9] = !(flags & CALIB_FIX_K4);
    mask[10] = !(flags & CALIB_FIX_K5);
    mask[11] = !(flags & CALIB_FIX_K6);

    if(flags & CALIB_FIX_S1_S2_S3_S4)
    {
        mask[12] = 0;
        mask[13] = 0;
        mask[14] = 0;
        mask[15] = 0;
    }
    if(flags & CALIB_FIX_TAUX_TAUY)
        mask[16] = mask[17] = 0;

    if(releaseObject)
    {
        // copy object points
        std::copy( matM.ptr<double>(), matM.ptr<double>( 0, maxPoints - 1 ) + 3,
                   param + NINTRINSIC + nimages * 6 );
        // fix points
        mask[NINTRINSIC + nimages * 6] = 0;
        mask[NINTRINSIC + nimages * 6 + 1] = 0;
        mask[NINTRINSIC + nimages * 6 + 2] = 0;
        mask[NINTRINSIC + nimages * 6 + iFixedPoint * 3] = 0;
        mask[NINTRINSIC + nimages * 6 + iFixedPoint * 3 + 1] = 0;
        mask[NINTRINSIC + nimages * 6 + iFixedPoint * 3 + 2] = 0;
        mask[nparams - 1] = 0;
    }

    // 2. initialize extrinsic parameters
    for( i = 0, pos = 0; i < nimages; i++, pos += ni )
    {
        ni = npoints.at<int>(i);

        Mat _ri = param0.rowRange(NINTRINSIC + i*6, NINTRINSIC + i*6 + 3);
        Mat _ti = param0.rowRange(NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6);

        Mat _Mi = matM.colRange(pos, pos + ni);
        Mat _mi = _m.colRange(pos, pos + ni);

        solvePnP(_Mi, _mi, matA, _k, _ri, _ti, false);
    }

    //std::cout << "single camera calib. param before LM: " << param0.t() << "\n";
    //std::cout << "single camera calib. mask: " << mask0.t() << "\n";

    // 3. run the optimization
    LMSolver::runAlt(param0, mask0, termCrit, solveMethod, false,
        [&](Mat& _param, Mat* _JtErr, Mat* _JtJ, double* _errnorm)
    {
        cameraCalcJErr(matM, _m, npoints, allErrors, _param, _JtErr, _JtJ, _errnorm,
                       aspectRatio, perViewErrors, flags, releaseObject);
        return true;
    });

    //std::cout << "single camera calib. param after LM: " << param0.t() << "\n";

    Mat JtErr(nparams, 1, CV_64F), JtJ(nparams, nparams, CV_64F), JtJinv, JtJN;
    double reprojErr = 0;
    JtErr.setZero(); JtJ.setZero();
    cameraCalcJErr(matM, _m, npoints, allErrors, param0,
                   stdDevs ? &JtErr : 0, stdDevs ? &JtJ : 0, &reprojErr,
                   aspectRatio, 0, flags, releaseObject);
    if (stdDevs)
    {
        int nparams_nz = countNonZero(mask0);
        JtJN.create(nparams_nz, nparams_nz, CV_64F);

        subMatrix(JtJ, JtJN, mask0, mask0);
        completeSymm(JtJN, false);
        cv::invert(JtJN, JtJinv, DECOMP_EIG);
        // sigma2 is deviation of the noise
        // see any papers about variance of the least squares estimator for
        // detailed description of the variance estimation methods
        double sigma2 = norm(allErrors, NORM_L2SQR) / (total - nparams_nz);
        int j = 0;
        for ( int s = 0; s < nparams; s++ )
            if( mask0.at<uchar>(s) )
            {
                stdDevs->at<double>(s) = std::sqrt(JtJinv.at<double>(j,j) * sigma2);
                j++;
            }
            else
                stdDevs->at<double>(s) = 0.;
    }

    // 4. store the results
    A = Matx33d(param[0], 0, param[2], 0, param[1], param[3], 0, 0, 1);
    A.convertTo(cameraMatrix, cameraMatrix.type());
    _k = Mat(distCoeffs.size(), CV_64F, param + 4);
    _k.convertTo(distCoeffs, distCoeffs.type());
    if( newObjPoints && releaseObject )
    {
        Mat _Mi = param0.rowRange(NINTRINSIC + nimages * 6,
                   NINTRINSIC + nimages * 6 + maxPoints * 3);
        _Mi.reshape(3, 1).convertTo(*newObjPoints, newObjPoints->type());
    }

    for( i = 0, pos = 0; i < nimages; i++ )
    {
        if( rvecs )
        {
            Mat src = Mat(3, 1, CV_64F, param + NINTRINSIC + i*6);
            if( rvecs->rows == nimages && rvecs->cols*rvecs->channels() == 9 )
            {
                Mat dst(3, 3, rvecs->depth(), rvecs->ptr(i));
                Rodrigues(src, A);
                A.convertTo(dst, dst.type());
            }
            else
            {
                Mat dst(3, 1, rvecs->depth(), rvecs->rows == 1 ?
                    rvecs->data + i*rvecs->elemSize1() : rvecs->ptr(i));
                src.convertTo(dst, dst.type());
            }
        }
        if( tvecs )
        {
            Mat src(3, 1, CV_64F, param + NINTRINSIC + i*6 + 3);
            Mat dst(3, 1, tvecs->depth(), tvecs->rows == 1 ?
                    tvecs->data + i*tvecs->elemSize1() : tvecs->ptr(i));
            src.convertTo(dst, dst.type());
        }
    }

    return std::sqrt(reprojErr/total);
}

//////////////////////////////// Stereo Calibration ///////////////////////////////////

static int dbCmp( const void* _a, const void* _b )
{
    double a = *(const double*)_a;
    double b = *(const double*)_b;

    return (a > b) - (a < b);
}

static double stereoCalibrateImpl(
        const Mat& _objectPoints, const Mat& _imagePoints1,
        const Mat& _imagePoints2, const Mat& _npoints,
        Mat& _cameraMatrix1, Mat& _distCoeffs1,
        Mat& _cameraMatrix2, Mat& _distCoeffs2,
        Size imageSize, Mat* matR, Mat* matT,
        Mat* matE, Mat* matF,
        Mat* perViewErr, int flags,
        TermCriteria termCrit )
{
    const int NINTRINSIC = 18;
    double reprojErr = 0;

    double dk[2][14]={{0}};
    Mat Dist[2];
    Matx33d A[2], R_LR;
    int i, k, p, ni = 0, pos, pointsTotal = 0, maxPoints = 0, nparams;
    bool recomputeIntrinsics = false;
    double aspectRatio[2] = {0};

    CV_Assert( _imagePoints1.type() == _imagePoints2.type() &&
               _imagePoints1.depth() == _objectPoints.depth() );

    CV_Assert( (_npoints.cols == 1 || _npoints.rows == 1) &&
               _npoints.type() == CV_32S );

    int nimages = (int)_npoints.total();
    for( i = 0; i < nimages; i++ )
    {
        ni = _npoints.at<int>(i);
        maxPoints = MAX(maxPoints, ni);
        pointsTotal += ni;
    }

    Mat objectPoints, imagePoints[2];
    _objectPoints.convertTo(objectPoints, CV_64F);
    objectPoints = objectPoints.reshape(3, 1);

    for( k = 0; k < 2; k++ )
    {
        const Mat& points = k == 0 ? _imagePoints1 : _imagePoints2;
        const Mat& cameraMatrix = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const Mat& distCoeffs = k == 0 ? _distCoeffs1 : _distCoeffs2;

        int depth = points.depth();
        int cn = points.channels();
        CV_Assert( (depth == CV_32F || depth == CV_64F) &&
                   ((points.rows == pointsTotal && points.cols*cn == 2) ||
                   (points.rows == 1 && points.cols == pointsTotal && cn == 2)));

        A[k] = Matx33d(1, 0, 0, 0, 1, 0, 0, 0, 1);
        Dist[k] = Mat(1,14,CV_64F,dk[k]);

        points.convertTo(imagePoints[k], CV_64F);
        imagePoints[k] = imagePoints[k].reshape(2, 1);

        if( flags & (CALIB_FIX_INTRINSIC|CALIB_USE_INTRINSIC_GUESS|
            CALIB_FIX_ASPECT_RATIO|CALIB_FIX_FOCAL_LENGTH) )
            cameraMatrix.convertTo(A[k], CV_64F);

        if( flags & (CALIB_FIX_INTRINSIC|CALIB_USE_INTRINSIC_GUESS|
            CALIB_FIX_K1|CALIB_FIX_K2|CALIB_FIX_K3|CALIB_FIX_K4|CALIB_FIX_K5|CALIB_FIX_K6|CALIB_FIX_TANGENT_DIST) )
        {
            Mat tdist( distCoeffs.size(), CV_MAKETYPE(CV_64F, distCoeffs.channels()), dk[k] );
            distCoeffs.convertTo(tdist, CV_64F);
        }

        if( !(flags & (CALIB_FIX_INTRINSIC|CALIB_USE_INTRINSIC_GUESS)))
        {
            Mat matA(A[k], false);
            calibrateCameraInternal(objectPoints, imagePoints[k],
                                    _npoints, imageSize, 0, matA, Dist[k],
                                    0, 0, 0, 0, 0, flags, termCrit);
            //std::cout << "K(" << k << "): " << A[k] << "\n";
            //std::cout << "Dist(" << k << "): " << Dist[k] << "\n";
        }
    }

    if( flags & CALIB_SAME_FOCAL_LENGTH )
    {
        A[0](0, 0) = A[1](0, 0) = (A[0](0, 0) + A[1](0, 0))*0.5;
        A[0](0, 2) = A[1](0, 2) = (A[0](0, 2) + A[1](0, 2))*0.5;
        A[0](1, 1) = A[1](1, 1) = (A[0](1, 1) + A[1](1, 1))*0.5;
        A[0](1, 2) = A[1](1, 2) = (A[0](1, 2) + A[1](1, 2))*0.5;
    }

    if( flags & CALIB_FIX_ASPECT_RATIO )
    {
        for( k = 0; k < 2; k++ )
            aspectRatio[k] = A[k](0, 0)/A[k](1, 1);
    }

    recomputeIntrinsics = (flags & CALIB_FIX_INTRINSIC) == 0;

    Mat err( maxPoints*2, 1, CV_64F );
    Mat Je( maxPoints*2, 6, CV_64F );
    Mat J_LR( maxPoints*2, 6, CV_64F );
    Mat Ji( maxPoints*2, NINTRINSIC, CV_64F, Scalar(0) );

    // we optimize for the inter-camera R(3),t(3), then, optionally,
    // for intrinisic parameters of each camera ((fx,fy,cx,cy,k1,k2,p1,p2) ~ 8 parameters).
    nparams = 6*(nimages+1) + (recomputeIntrinsics ? NINTRINSIC*2 : 0);

    std::vector<uchar> mask(nparams, (uchar)0);
    std::vector<double> param(nparams, 0.);

    if( recomputeIntrinsics )
    {
        uchar* imask = &mask[0] + nparams - NINTRINSIC*2;
        if( !(flags & CALIB_RATIONAL_MODEL) )
            flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
        if( !(flags & CALIB_THIN_PRISM_MODEL) )
            flags |= CALIB_FIX_S1_S2_S3_S4;
        if( !(flags & CALIB_TILTED_MODEL) )
            flags |= CALIB_FIX_TAUX_TAUY;
        if( flags & CALIB_FIX_ASPECT_RATIO )
            imask[0] = imask[NINTRINSIC] = 0;
        if( flags & CALIB_FIX_FOCAL_LENGTH )
            imask[0] = imask[1] = imask[NINTRINSIC] = imask[NINTRINSIC+1] = 0;
        if( flags & CALIB_FIX_PRINCIPAL_POINT )
            imask[2] = imask[3] = imask[NINTRINSIC+2] = imask[NINTRINSIC+3] = 0;
        if( flags & (CALIB_ZERO_TANGENT_DIST|CALIB_FIX_TANGENT_DIST) )
            imask[6] = imask[7] = imask[NINTRINSIC+6] = imask[NINTRINSIC+7] = 0;
        if( flags & CALIB_FIX_K1 )
            imask[4] = imask[NINTRINSIC+4] = 0;
        if( flags & CALIB_FIX_K2 )
            imask[5] = imask[NINTRINSIC+5] = 0;
        if( flags & CALIB_FIX_K3 )
            imask[8] = imask[NINTRINSIC+8] = 0;
        if( flags & CALIB_FIX_K4 )
            imask[9] = imask[NINTRINSIC+9] = 0;
        if( flags & CALIB_FIX_K5 )
            imask[10] = imask[NINTRINSIC+10] = 0;
        if( flags & CALIB_FIX_K6 )
            imask[11] = imask[NINTRINSIC+11] = 0;
        if( flags & CALIB_FIX_S1_S2_S3_S4 )
        {
            imask[12] = imask[NINTRINSIC+12] = 0;
            imask[13] = imask[NINTRINSIC+13] = 0;
            imask[14] = imask[NINTRINSIC+14] = 0;
            imask[15] = imask[NINTRINSIC+15] = 0;
        }
        if( flags & CALIB_FIX_TAUX_TAUY )
        {
            imask[16] = imask[NINTRINSIC+16] = 0;
            imask[17] = imask[NINTRINSIC+17] = 0;
        }
    }

    // storage for initial [om(R){i}|t{i}] (in order to compute the median for each component)
    Mat RT0(6, nimages, CV_64F);
    double* RT0data = RT0.ptr<double>();
    /*
       Compute initial estimate of pose
       For each image, compute:
          R(om) is the rotation matrix of om
          om(R) is the rotation vector of R
          R_ref = R(om_right) * R(om_left)'
          T_ref_list = [T_ref_list; T_right - R_ref * T_left]
          om_ref_list = {om_ref_list; om(R_ref)]
       om = median(om_ref_list)
       T = median(T_ref_list)
    */
    for( i = pos = 0; i < nimages; pos += ni, i++ )
    {
        ni = _npoints.at<int>(i);
        Mat objpt_i(1, ni, CV_64FC3, objectPoints.ptr<double>() + pos*3);
        Matx33d R[2];
        Vec3d rv, T[2];
        for( k = 0; k < 2; k++ )
        {
            Mat imgpt_ik = Mat(1, ni, CV_64FC2, imagePoints[k].ptr<double>() + pos*2);
            solvePnP(objpt_i, imgpt_ik, A[k], Dist[k], rv, T[k], false, SOLVEPNP_ITERATIVE );
            Rodrigues(rv, R[k]);

            if( k == 0 )
            {
                // save initial om_left and T_left
                param[(i+1)*6] = rv[0];
                param[(i+1)*6 + 1] = rv[1];
                param[(i+1)*6 + 2] = rv[2];
                param[(i+1)*6 + 3] = T[0][0];
                param[(i+1)*6 + 4] = T[0][1];
                param[(i+1)*6 + 5] = T[0][2];
            }
        }
        R[0] = R[1]*R[0].t();
        T[1] -= R[0]*T[0];

        Rodrigues(R[0], rv);

        RT0data[i] = rv[0];
        RT0data[i + nimages] = rv[1];
        RT0data[i + nimages*2] = rv[2];
        RT0data[i + nimages*3] = T[1][0];
        RT0data[i + nimages*4] = T[1][1];
        RT0data[i + nimages*5] = T[1][2];
    }

    if(flags & CALIB_USE_EXTRINSIC_GUESS)
    {
        Vec3d R, T;
        matT->convertTo(T, CV_64F);

        if( matR->rows == 3 && matR->cols == 3 )
            Rodrigues(*matR, R);
        else
            matR->convertTo(R, CV_64F);

        param[0] = R[0];
        param[1] = R[1];
        param[2] = R[2];
        param[3] = T[0];
        param[4] = T[1];
        param[5] = T[2];
    }
    else
    {
        // find the medians and save the first 6 parameters
        for( i = 0; i < 6; i++ )
        {
            double* rti = RT0data + i*nimages;
            qsort( rti, nimages, sizeof(*rti), dbCmp );
            param[i] = nimages % 2 != 0 ? rti[nimages/2] : (rti[nimages/2 - 1] + rti[nimages/2])*0.5;
        }
    }

    if( recomputeIntrinsics )
        for( k = 0; k < 2; k++ )
        {
            double* iparam = &param[(nimages+1)*6 + k*NINTRINSIC];
            if( flags & CALIB_ZERO_TANGENT_DIST )
                dk[k][2] = dk[k][3] = 0;
            iparam[0] = A[k](0, 0); iparam[1] = A[k](1, 1); iparam[2] = A[k](0, 2); iparam[3] = A[k](1, 2);
            iparam[4] = dk[k][0]; iparam[5] = dk[k][1]; iparam[6] = dk[k][2];
            iparam[7] = dk[k][3]; iparam[8] = dk[k][4]; iparam[9] = dk[k][5];
            iparam[10] = dk[k][6]; iparam[11] = dk[k][7];
            iparam[12] = dk[k][8];
            iparam[13] = dk[k][9];
            iparam[14] = dk[k][10];
            iparam[15] = dk[k][11];
            iparam[16] = dk[k][12];
            iparam[17] = dk[k][13];
        }

    A[0] = A[1] = Matx33d(1, 0, 0, 0, 1, 0, 0, 0, 1);

    //std::cout << "param before LM: " << Mat(param, false).t() << "\n";

    LMSolver::runAlt(param, mask, termCrit, DECOMP_SVD, false,
    [&](Mat& _param, Mat* _JtErr, Mat* _JtJ, double* _errnorm)
    {
        double* param_p = _param.ptr<double>();
        Vec3d om_LR(param_p[0], param_p[1], param_p[2]);
        Vec3d T_LR(param_p[3], param_p[4], param_p[5]);
        Vec3d om[2], T[2];
        Matx33d dr3dr1, dr3dr2, dt3dr2, dt3dt1, dt3dt2;

        reprojErr = 0;
        Rodrigues(om_LR, R_LR);

        if( recomputeIntrinsics )
        {
            double* iparam = param_p + (nimages+1)*6;
            //double* ipparam = solver.prevParam->data.db + (nimages+1)*6;

            if( flags & CALIB_SAME_FOCAL_LENGTH )
            {
                iparam[NINTRINSIC] = iparam[0];
                iparam[NINTRINSIC+1] = iparam[1];
                //ipparam[NINTRINSIC] = ipparam[0];
                //ipparam[NINTRINSIC+1] = ipparam[1];
            }
            if( flags & CALIB_FIX_ASPECT_RATIO )
            {
                iparam[0] = iparam[1]*aspectRatio[0];
                iparam[NINTRINSIC] = iparam[NINTRINSIC+1]*aspectRatio[1];
                //ipparam[0] = ipparam[1]*aspectRatio[0];
                //ipparam[NINTRINSIC] = ipparam[NINTRINSIC+1]*aspectRatio[1];
            }
            for( k = 0; k < 2; k++ )
            {
                A[k] = Matx33d(iparam[k*NINTRINSIC+0], 0, iparam[k*NINTRINSIC+2],
                               0, iparam[k*NINTRINSIC+1], iparam[k*NINTRINSIC+3],
                               0, 0, 1);
                for(int j = 0; j < 14; j++)
                    dk[k][j] = iparam[k*NINTRINSIC+4+j];
            }
        }

        for( i = pos = 0; i < nimages; pos += ni, i++ )
        {
            ni = _npoints.at<int>(i);

            double* pi = param_p + (i+1)*6;
            om[0] = Vec3d(pi[0], pi[1], pi[2]);
            T[0] = Vec3d(pi[3], pi[4], pi[5]);

            if( _JtJ || _JtErr )
                composeRT( om[0], T[0], om_LR, T_LR, om[1], T[1], dr3dr1, noArray(),
                           dr3dr2, noArray(), noArray(), dt3dt1, dt3dr2, dt3dt2 );
            else
                composeRT( om[0], T[0], om_LR, T_LR, om[1], T[1] );

            Mat objpt_i(1, ni, CV_64FC3, objectPoints.ptr<double>() + pos*3);
            err.resize(ni*2); Je.resize(ni*2); J_LR.resize(ni*2); Ji.resize(ni*2);

            Mat tmpImagePoints = err.reshape(2, 1);
            Mat dpdf = Ji.colRange(0, 2);
            Mat dpdc = Ji.colRange(2, 4);
            Mat dpdk = Ji.colRange(4, NINTRINSIC);
            Mat dpdrot = Je.colRange(0, 3);
            Mat dpdt = Je.colRange(3, 6);

            for( k = 0; k < 2; k++ )
            {
                Mat imgpt_ik(1, ni, CV_64FC2, imagePoints[k].ptr<double>() + pos*2);

                if( _JtJ || _JtErr )
                    projectPoints(objpt_i, om[k], T[k], A[k], Dist[k],
                            tmpImagePoints, dpdrot, dpdt, dpdf, dpdc, dpdk, noArray(),
                            (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio[k] : 0.);
                else
                    projectPoints(objpt_i, om[k], T[k], A[k], Dist[k], tmpImagePoints);
                subtract( tmpImagePoints, imgpt_ik, tmpImagePoints );

                if( _JtJ )
                {
                    Mat& JtErr = *_JtErr;
                    Mat& JtJ = *_JtJ;
                    int iofs = (nimages+1)*6 + k*NINTRINSIC, eofs = (i+1)*6;
                    assert( _JtJ && _JtErr );

                    if( k == 1 )
                    {
                        // d(err_{x|y}R) ~ de3
                        // convert de3/{dr3,dt3} => de3{dr1,dt1} & de3{dr2,dt2}
                        for( p = 0; p < ni*2; p++ )
                        {
                            Mat de3dr3( 1, 3, CV_64F, Je.ptr(p));
                            Mat de3dt3( 1, 3, CV_64F, de3dr3.ptr<double>() + 3 );
                            Mat de3dr2( 1, 3, CV_64F, J_LR.ptr(p) );
                            Mat de3dt2( 1, 3, CV_64F, de3dr2.ptr<double>() + 3 );
                            double _de3dr1[3], _de3dt1[3];
                            Mat de3dr1( 1, 3, CV_64F, _de3dr1 );
                            Mat de3dt1( 1, 3, CV_64F, _de3dt1 );

                            gemm(de3dr3, dr3dr1, 1, noArray(), 0, de3dr1);
                            gemm(de3dt3, dt3dt1, 1, noArray(), 0, de3dt1);

                            gemm(de3dr3, dr3dr2, 1, noArray(), 0, de3dr2);
                            gemm(de3dt3, dt3dr2, 1, de3dr2, 1, de3dr2);
                            gemm(de3dt3, dt3dt2, 1, noArray(), 0, de3dt2);

                            de3dr1.copyTo(de3dr3);
                            de3dt1.copyTo(de3dt3);
                        }

                        JtJ(Rect(0, 0, 6, 6)) += J_LR.t()*J_LR;
                        JtJ(Rect(eofs, 0, 6, 6)) = J_LR.t()*Je;
                        JtErr.rowRange(0, 6) += J_LR.t()*err;
                    }

                    JtJ(Rect(eofs, eofs, 6, 6)) += Je.t()*Je;
                    JtErr.rowRange(eofs, eofs + 6) += Je.t()*err;

                    if( recomputeIntrinsics )
                    {
                        JtJ(Rect(iofs, iofs, NINTRINSIC, NINTRINSIC)) += Ji.t()*Ji;
                        JtJ(Rect(iofs, eofs, NINTRINSIC, 6)) += Je.t()*Ji;
                        if( k == 1 )
                        {
                            JtJ(Rect(iofs, 0, NINTRINSIC, 6)) += J_LR.t()*Ji;
                        }
                        JtErr.rowRange(iofs, iofs + NINTRINSIC) += Ji.t()*err;
                    }
                }

                double viewErr = norm(err, NORM_L2SQR);
                if(perViewErr)
                    perViewErr->at<double>(i, k) = std::sqrt(viewErr/ni);
                reprojErr += viewErr;
            }
        }
        if(_errnorm)
            *_errnorm = reprojErr;
        return true;
    });

    Vec3d om_LR(param[0], param[1], param[2]);
    Vec3d T_LR(param[3], param[4], param[5]);
    Rodrigues( om_LR, R_LR );
    if( matR->rows == 1 || matR->cols == 1 )
        om_LR.convertTo(*matR, matR->depth());
    else
        R_LR.convertTo(*matR, matR->depth());
    T_LR.convertTo(*matT, matT->depth());
    for( k = 0; k < 2; k++ )
    {
        double* iparam = &param[(nimages+1)*6 + k*NINTRINSIC];
        A[k] = Matx33d(iparam[0], 0, iparam[2], 0, iparam[1], iparam[3], 0, 0, 1);
    }

    if( recomputeIntrinsics )
    {
        for( k = 0; k < 2; k++ )
        {
            Mat& cameraMatrix = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
            Mat& distCoeffs = k == 0 ? _distCoeffs1 : _distCoeffs2;
            A[k].convertTo(cameraMatrix, cameraMatrix.depth());
            Mat tdist( distCoeffs.size(), CV_MAKETYPE(CV_64F,distCoeffs.channels()), Dist[k].data );
            tdist.convertTo(distCoeffs, distCoeffs.depth());
        }
    }

    if( matE || matF )
    {
        Matx33d Tx(0, -T_LR[2], T_LR[1],
                   T_LR[2], 0, -T_LR[0],
                   -T_LR[1], T_LR[0], 0);
        Matx33d E = Tx*R_LR;
        if (matE)
            E.convertTo(*matE, matE->depth());
        if( matF )
        {
            Matx33d iA0 = A[0].inv(), iA1 = A[1].inv();
            Matx33d F = iA1.t()*E*iA0;
            F.convertTo(*matF, matF->depth(), fabs(F(2,2)) > 0 ? 1./F(2,2) : 1.);
        }
    }

    return std::sqrt(reprojErr/(pointsTotal*2));
}

static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    int iFixedPoint,
                                    Mat& objPtMat, Mat& imgPtMat1, Mat* imgPtMat2,
                                    Mat& npoints )
{
    int nimages = (int)objectPoints.total();
    int total = 0;
    CV_Assert(nimages > 0);
    CV_CheckEQ(nimages, (int)imagePoints1.total(), "");
    if (imgPtMat2)
        CV_CheckEQ(nimages, (int)imagePoints2.total(), "");

    for (int i = 0; i < nimages; i++)
    {
        Mat objectPoint = objectPoints.getMat(i);
        if (objectPoint.empty())
            CV_Error(CV_StsBadSize, "objectPoints should not contain empty vector of vectors of points");
        int numberOfObjectPoints = objectPoint.checkVector(3, CV_32F);
        if (numberOfObjectPoints <= 0)
            CV_Error(CV_StsUnsupportedFormat, "objectPoints should contain vector of vectors of points of type Point3f");

        Mat imagePoint1 = imagePoints1.getMat(i);
        if (imagePoint1.empty())
            CV_Error(CV_StsBadSize, "imagePoints1 should not contain empty vector of vectors of points");
        int numberOfImagePoints = imagePoint1.checkVector(2, CV_32F);
        if (numberOfImagePoints <= 0)
            CV_Error(CV_StsUnsupportedFormat, "imagePoints1 should contain vector of vectors of points of type Point2f");
        CV_CheckEQ(numberOfObjectPoints, numberOfImagePoints, "Number of object and image points must be equal");

        total += numberOfObjectPoints;
    }

    npoints.create(1, (int)nimages, CV_32S);
    objPtMat.create(1, (int)total, CV_32FC3);
    imgPtMat1.create(1, (int)total, CV_32FC2);
    Point2f* imgPtData2 = 0;

    if (imgPtMat2)
    {
        imgPtMat2->create(1, (int)total, CV_32FC2);
        imgPtData2 = imgPtMat2->ptr<Point2f>();
    }

    Point3f* objPtData = objPtMat.ptr<Point3f>();
    Point2f* imgPtData1 = imgPtMat1.ptr<Point2f>();

    for (int i = 0, j = 0; i < nimages; i++)
    {
        Mat objpt = objectPoints.getMat(i);
        Mat imgpt1 = imagePoints1.getMat(i);
        int numberOfObjectPoints = objpt.checkVector(3, CV_32F);
        npoints.at<int>(i) = numberOfObjectPoints;
        for (int n = 0; n < numberOfObjectPoints; ++n)
        {
            objPtData[j + n] = objpt.ptr<Point3f>()[n];
            imgPtData1[j + n] = imgpt1.ptr<Point2f>()[n];
        }

        if (imgPtData2)
        {
            Mat imgpt2 = imagePoints2.getMat(i);
            int numberOfImage2Points = imgpt2.checkVector(2, CV_32F);
            CV_CheckEQ(numberOfObjectPoints, numberOfImage2Points, "Number of object and image(2) points must be equal");
            for (int n = 0; n < numberOfImage2Points; ++n)
            {
                imgPtData2[j + n] = imgpt2.ptr<Point2f>()[n];
            }
        }

        j += numberOfObjectPoints;
    }

    int ni = npoints.at<int>(0);
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < ni - 1;
    // check object points. If not qualified, report errors.
    if( releaseObject )
    {
        for (int i = 1; i < nimages; i++)
        {
            if( npoints.at<int>(i) != ni )
            {
                CV_Error( CV_StsBadArg, "All objectPoints[i].size() should be equal when "
                                        "object-releasing method is requested." );
            }
            Mat ocmp = objPtMat.colRange(ni * i, ni * i + ni) != objPtMat.colRange(0, ni);
            ocmp = ocmp.reshape(1);
            if( countNonZero(ocmp) )
            {
                CV_Error( CV_StsBadArg, "All objectPoints[i] should be identical when object-releasing"
                                        " method is requested." );
            }
        }
    }
}

static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    Mat& objPtMat, Mat& imgPtMat1, Mat* imgPtMat2,
                                    Mat& npoints )
{
    collectCalibrationData( objectPoints, imagePoints1, imagePoints2, -1, objPtMat, imgPtMat1,
                            imgPtMat2, npoints );
}

static Mat prepareCameraMatrix(Mat& cameraMatrix0, int rtype, int flags)
{
    Mat cameraMatrix = Mat::eye(3, 3, rtype);
    if( cameraMatrix0.size() == cameraMatrix.size() )
        cameraMatrix0.convertTo(cameraMatrix, rtype);
    else if( flags & CALIB_USE_INTRINSIC_GUESS )
        CV_Error(Error::StsBadArg, "CALIB_USE_INTRINSIC_GUESS flag is set, but the camera matrix is not 3x3");
    return cameraMatrix;
}

Mat initCameraMatrix2D( InputArrayOfArrays objectPoints,
                        InputArrayOfArrays imagePoints,
                        Size imageSize, double aspectRatio )
{
    CV_INSTRUMENT_REGION();

    Mat objPt, imgPt, npoints, cameraMatrix;
    collectCalibrationData( objectPoints, imagePoints, noArray(),
                            objPt, imgPt, 0, npoints );
    initIntrinsicParams2D( objPt, imgPt, npoints, imageSize, cameraMatrix, aspectRatio );
    return cameraMatrix;
}

static Mat prepareDistCoeffs(Mat& distCoeffs0, int rtype, int outputSize = 14)
{
    CV_Assert((int)distCoeffs0.total() <= outputSize);
    Mat distCoeffs = Mat::zeros(distCoeffs0.cols == 1 ? Size(1, outputSize) : Size(outputSize, 1), rtype);
    if( distCoeffs0.size() == Size(1, 4) ||
       distCoeffs0.size() == Size(1, 5) ||
       distCoeffs0.size() == Size(1, 8) ||
       distCoeffs0.size() == Size(1, 12) ||
       distCoeffs0.size() == Size(1, 14) ||
       distCoeffs0.size() == Size(4, 1) ||
       distCoeffs0.size() == Size(5, 1) ||
       distCoeffs0.size() == Size(8, 1) ||
       distCoeffs0.size() == Size(12, 1) ||
       distCoeffs0.size() == Size(14, 1) )
    {
        Mat dstCoeffs(distCoeffs, Rect(0, 0, distCoeffs0.cols, distCoeffs0.rows));
        distCoeffs0.convertTo(dstCoeffs, rtype);
    }
    return distCoeffs;
}

double calibrateCamera( InputArrayOfArrays _objectPoints,
                        InputArrayOfArrays _imagePoints,
                        Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                        OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    return calibrateCamera(_objectPoints, _imagePoints, imageSize, _cameraMatrix, _distCoeffs,
                                         _rvecs, _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

double calibrateCamera(InputArrayOfArrays _objectPoints,
                       InputArrayOfArrays _imagePoints,
                       Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                       OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                       OutputArray stdDeviationsIntrinsics,
                       OutputArray stdDeviationsExtrinsics,
                       OutputArray _perViewErrors, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    return calibrateCameraRO(_objectPoints, _imagePoints, imageSize, -1, _cameraMatrix, _distCoeffs,
                             _rvecs, _tvecs, noArray(), stdDeviationsIntrinsics, stdDeviationsExtrinsics,
                             noArray(), _perViewErrors, flags, criteria);
}

double calibrateCameraRO(InputArrayOfArrays _objectPoints,
                         InputArrayOfArrays _imagePoints,
                         Size imageSize, int iFixedPoint, InputOutputArray _cameraMatrix,
                         InputOutputArray _distCoeffs,
                         OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                         OutputArray newObjPoints,
                         int flags, TermCriteria criteria)
{
    CV_INSTRUMENT_REGION();

    return calibrateCameraRO(_objectPoints, _imagePoints, imageSize, iFixedPoint, _cameraMatrix,
                             _distCoeffs, _rvecs, _tvecs, newObjPoints, noArray(), noArray(),
                             noArray(), noArray(), flags, criteria);
}

double calibrateCameraRO(InputArrayOfArrays _objectPoints,
                         InputArrayOfArrays _imagePoints,
                         Size imageSize, int iFixedPoint, InputOutputArray _cameraMatrix,
                         InputOutputArray _distCoeffs,
                         OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                         OutputArray newObjPoints,
                         OutputArray stdDeviationsIntrinsics,
                         OutputArray stdDeviationsExtrinsics,
                         OutputArray stdDeviationsObjPoints,
                         OutputArray _perViewErrors, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    int rtype = CV_64F;

    CV_Assert( _cameraMatrix.needed() );
    CV_Assert( _distCoeffs.needed() );

    Mat cameraMatrix = _cameraMatrix.getMat();
    cameraMatrix = prepareCameraMatrix(cameraMatrix, rtype, flags);
    Mat distCoeffs = _distCoeffs.getMat();
    distCoeffs =
        (flags & CALIB_THIN_PRISM_MODEL) &&
        !(flags & CALIB_TILTED_MODEL) ?
            prepareDistCoeffs(distCoeffs, rtype, 12) :
            prepareDistCoeffs(distCoeffs, rtype);
    if( !(flags & CALIB_RATIONAL_MODEL) &&
    (!(flags & CALIB_THIN_PRISM_MODEL)) &&
    (!(flags & CALIB_TILTED_MODEL)))
        distCoeffs = distCoeffs.rows == 1 ? distCoeffs.colRange(0, 5) : distCoeffs.rowRange(0, 5);

    int nimages = int(_objectPoints.total());
    CV_Assert( nimages > 0 );
    Mat objPt, imgPt, npoints, rvecM, tvecM, stdDeviationsM, errorsM;

    bool rvecs_needed = _rvecs.needed(), tvecs_needed = _tvecs.needed(),
            stddev_needed = stdDeviationsIntrinsics.needed(), errors_needed = _perViewErrors.needed(),
            stddev_ext_needed = stdDeviationsExtrinsics.needed();
    bool newobj_needed = newObjPoints.needed();
    bool stddev_obj_needed = stdDeviationsObjPoints.needed();

    bool rvecs_mat_vec = _rvecs.isMatVector();
    bool tvecs_mat_vec = _tvecs.isMatVector();

    if( rvecs_needed )
    {
        _rvecs.create(nimages, 1, CV_64FC3);

        if(rvecs_mat_vec)
            rvecM.create(nimages, 3, CV_64F);
        else
            rvecM = _rvecs.getMat();
    }

    if( tvecs_needed )
    {
        _tvecs.create(nimages, 1, CV_64FC3);

        if(tvecs_mat_vec)
            tvecM.create(nimages, 3, CV_64F);
        else
            tvecM = _tvecs.getMat();
    }

    collectCalibrationData( _objectPoints, _imagePoints, noArray(), iFixedPoint,
                            objPt, imgPt, 0, npoints );
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints.at<int>(0) - 1;

    newobj_needed = newobj_needed && releaseObject;
    int np = npoints.at<int>( 0 );
    Mat newObjPt;
    if( newobj_needed ) {
        newObjPoints.create( 1, np, CV_32FC3 );
        newObjPt = newObjPoints.getMat();
    }

    stddev_obj_needed = stddev_obj_needed && releaseObject;
    bool stddev_any_needed = stddev_needed || stddev_ext_needed || stddev_obj_needed;
    if( stddev_any_needed )
    {
        if( releaseObject )
            stdDeviationsM.create(nimages*6 + CALIB_NINTRINSIC + np * 3, 1, CV_64F);
        else
            stdDeviationsM.create(nimages*6 + CALIB_NINTRINSIC, 1, CV_64F);
    }

    if( errors_needed )
    {
        _perViewErrors.create(nimages, 1, CV_64F);
        errorsM = _perViewErrors.getMat();
    }

    double reprojErr = calibrateCameraInternal(
            objPt, imgPt, npoints, imageSize, iFixedPoint,
            cameraMatrix, distCoeffs,
            rvecs_needed ? &rvecM : NULL,
            tvecs_needed ? &tvecM : NULL,
            newobj_needed ? &newObjPt : NULL,
            stddev_any_needed ? &stdDeviationsM : NULL,
            errors_needed ? &errorsM : NULL, flags, cvTermCriteria(criteria));

    if( newobj_needed )
        newObjPt.copyTo(newObjPoints);

    if( stddev_needed )
    {
        stdDeviationsIntrinsics.create(CALIB_NINTRINSIC, 1, CV_64F);
        Mat stdDeviationsIntrinsicsMat = stdDeviationsIntrinsics.getMat();
        std::memcpy(stdDeviationsIntrinsicsMat.ptr(), stdDeviationsM.ptr(),
                    CALIB_NINTRINSIC*sizeof(double));
    }

    if ( stddev_ext_needed )
    {
        stdDeviationsExtrinsics.create(nimages*6, 1, CV_64F);
        Mat stdDeviationsExtrinsicsMat = stdDeviationsExtrinsics.getMat();
        std::memcpy(stdDeviationsExtrinsicsMat.ptr(),
                    stdDeviationsM.ptr() + CALIB_NINTRINSIC*sizeof(double),
                    nimages*6*sizeof(double));
    }

    if( stddev_obj_needed )
    {
        stdDeviationsObjPoints.create( np * 3, 1, CV_64F );
        Mat stdDeviationsObjPointsMat = stdDeviationsObjPoints.getMat();
        std::memcpy( stdDeviationsObjPointsMat.ptr(), stdDeviationsM.ptr()
                         + ( CALIB_NINTRINSIC + nimages * 6 ) * sizeof( double ),
                     np * 3 * sizeof( double ) );
    }

    // overly complicated and inefficient rvec/ tvec handling to support vector<Mat>
    for(int i = 0; i < nimages; i++ )
    {
        if( rvecs_needed && rvecs_mat_vec)
        {
            _rvecs.create(3, 1, CV_64F, i, true);
            Mat rv = _rvecs.getMat(i);
            memcpy(rv.ptr(), rvecM.ptr(i), 3*sizeof(double));
        }
        if( tvecs_needed && tvecs_mat_vec)
        {
            _tvecs.create(3, 1, CV_64F, i, true);
            Mat tv = _tvecs.getMat(i);
            memcpy(tv.ptr(), tvecM.ptr(i), 3*sizeof(double));
        }
    }

    cameraMatrix.copyTo(_cameraMatrix);
    distCoeffs.copyTo(_distCoeffs);

    return reprojErr;
}


void calibrationMatrixValues( InputArray _cameraMatrix, Size imageSize,
                              double apertureWidth, double apertureHeight,
                              double& fovx, double& fovy, double& focalLength,
                              Point2d& principalPoint, double& aspectRatio )
{
    CV_INSTRUMENT_REGION();

    if(_cameraMatrix.size() != Size(3, 3))
        CV_Error(CV_StsUnmatchedSizes, "Size of cameraMatrix must be 3x3!");

    Matx33d A;
    _cameraMatrix.getMat().convertTo(A, CV_64F);
    CV_DbgAssert(imageSize.width != 0 && imageSize.height != 0 && A(0, 0) != 0.0 && A(1, 1) != 0.0);

    /* Calculate pixel aspect ratio. */
    aspectRatio = A(1, 1) / A(0, 0);

    /* Calculate number of pixel per realworld unit. */
    double mx, my;
    if(apertureWidth != 0.0 && apertureHeight != 0.0) {
        mx = imageSize.width / apertureWidth;
        my = imageSize.height / apertureHeight;
    } else {
        mx = 1.0;
        my = aspectRatio;
    }

    /* Calculate fovx and fovy. */
    fovx = atan2(A(0, 2), A(0, 0)) + atan2(imageSize.width  - A(0, 2), A(0, 0));
    fovy = atan2(A(1, 2), A(1, 1)) + atan2(imageSize.height - A(1, 2), A(1, 1));
    fovx *= 180.0 / CV_PI;
    fovy *= 180.0 / CV_PI;

    /* Calculate focal length. */
    focalLength = A(0, 0) / mx;

    /* Calculate principle point. */
    principalPoint = Point2d(A(0, 2) / mx, A(1, 2) / my);
}

double stereoCalibrate( InputArrayOfArrays _objectPoints,
                        InputArrayOfArrays _imagePoints1,
                        InputArrayOfArrays _imagePoints2,
                        InputOutputArray _cameraMatrix1, InputOutputArray _distCoeffs1,
                        InputOutputArray _cameraMatrix2, InputOutputArray _distCoeffs2,
                        Size imageSize, OutputArray _Rmat, OutputArray _Tmat,
                        OutputArray _Emat, OutputArray _Fmat, int flags,
                        TermCriteria criteria)
{
    if (flags & CALIB_USE_EXTRINSIC_GUESS)
        CV_Error(Error::StsBadFlag, "stereoCalibrate does not support CALIB_USE_EXTRINSIC_GUESS.");

    Mat Rmat, Tmat;
    double ret = stereoCalibrate(_objectPoints, _imagePoints1, _imagePoints2, _cameraMatrix1, _distCoeffs1,
                                 _cameraMatrix2, _distCoeffs2, imageSize, Rmat, Tmat, _Emat, _Fmat,
                                 noArray(), flags, criteria);
    Rmat.copyTo(_Rmat);
    Tmat.copyTo(_Tmat);
    return ret;
}

double stereoCalibrate( InputArrayOfArrays _objectPoints,
                        InputArrayOfArrays _imagePoints1,
                        InputArrayOfArrays _imagePoints2,
                        InputOutputArray _cameraMatrix1, InputOutputArray _distCoeffs1,
                        InputOutputArray _cameraMatrix2, InputOutputArray _distCoeffs2,
                        Size imageSize, InputOutputArray _Rmat, InputOutputArray _Tmat,
                        OutputArray _Emat, OutputArray _Fmat,
                        OutputArray _perViewErrors, int flags ,
                        TermCriteria criteria)
{
    int rtype = CV_64F;
    Mat cameraMatrix1 = _cameraMatrix1.getMat();
    Mat cameraMatrix2 = _cameraMatrix2.getMat();
    Mat distCoeffs1 = _distCoeffs1.getMat();
    Mat distCoeffs2 = _distCoeffs2.getMat();
    cameraMatrix1 = prepareCameraMatrix(cameraMatrix1, rtype, flags);
    cameraMatrix2 = prepareCameraMatrix(cameraMatrix2, rtype, flags);
    distCoeffs1 = prepareDistCoeffs(distCoeffs1, rtype);
    distCoeffs2 = prepareDistCoeffs(distCoeffs2, rtype);

    if( !(flags & CALIB_RATIONAL_MODEL) &&
    (!(flags & CALIB_THIN_PRISM_MODEL)) &&
    (!(flags & CALIB_TILTED_MODEL)))
    {
        distCoeffs1 = distCoeffs1.rows == 1 ? distCoeffs1.colRange(0, 5) : distCoeffs1.rowRange(0, 5);
        distCoeffs2 = distCoeffs2.rows == 1 ? distCoeffs2.colRange(0, 5) : distCoeffs2.rowRange(0, 5);
    }

    if((flags & CALIB_USE_EXTRINSIC_GUESS) == 0)
    {
        _Rmat.create(3, 3, rtype);
        _Tmat.create(3, 1, rtype);
    }

    Mat objPt, imgPt, imgPt2, npoints;

    collectCalibrationData( _objectPoints, _imagePoints1, _imagePoints2,
                            objPt, imgPt, &imgPt2, npoints );
    Mat matR = _Rmat.getMat(), matT = _Tmat.getMat();

    bool E_needed = _Emat.needed(), F_needed = _Fmat.needed(), errors_needed = _perViewErrors.needed();

    Mat matE, matF, matErr;
    if( E_needed )
    {
        _Emat.create(3, 3, rtype);
        matE = _Emat.getMat();
    }
    if( F_needed )
    {
        _Fmat.create(3, 3, rtype);
        matF = _Fmat.getMat();
    }

    if( errors_needed )
    {
        int nimages = int(_objectPoints.total());
        _perViewErrors.create(nimages, 2, CV_64F);
        matErr = _perViewErrors.getMat();
    }

    double err = stereoCalibrateImpl(objPt, imgPt, imgPt2, npoints, cameraMatrix1,
                                     distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
                                     &matR, &matT, E_needed ? &matE : NULL, F_needed ? &matF : NULL,
                                     errors_needed ? &matErr : NULL, flags, criteria);
    cameraMatrix1.copyTo(_cameraMatrix1);
    cameraMatrix2.copyTo(_cameraMatrix2);
    distCoeffs1.copyTo(_distCoeffs1);
    distCoeffs2.copyTo(_distCoeffs2);

    return err;
}

}

/* End of file. */
