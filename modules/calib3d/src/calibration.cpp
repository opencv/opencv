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
#include "opencv2/core/core_c.h"
#include "opencv2/calib3d/calib3d_c.h"
#include <stdio.h>
#include <iterator>

/*
    This is straight-forward port v3 of Matlab calibration engine by Jean-Yves Bouguet
    that is (in a large extent) based on the paper:
    Z. Zhang. "A flexible new technique for camera calibration".
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
    The 1st initial port was done by Valery Mosyagin.
*/

namespace cv {

static const char* cvDistCoeffErr = "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8, 8x1, 1x12, 12x1, 1x14 or 14x1 floating-point vector";

static void initIntrinsicParams2D( const Mat& objectPoints,
                         const Mat& imagePoints, const Mat& npoints,
                         Size imageSize, OutputArray cameraMatrix,
                         double aspectRatio )
{
    CV_Assert(npoints.type() == CV_32SC1 && (npoints.rows == 1 || npoints.cols == 1) && npoints.isContinuous());
    int nimages = npoints.rows + npoints.cols - 1;

    CV_Assert( objectPoints.type() == CV_32FC3 ||
               objectPoints.type() == CV_64FC3 );
    CV_Assert( imagePoints.type() == CV_32FC2 ||
               imagePoints.type() == CV_64FC2 );

    if( objectPoints.rows != 1 || imagePoints.rows != 1 )
        CV_Error( cv::Error::StsBadSize, "object points and image points must be a single-row matrices" );

    Mat_<double> matA(2*nimages, 2);
    Mat_<double> matb(2*nimages, 1, CV_64F );
    double fx, fy, cx, cy;
    cx = (!imageSize.width ) ? 0.5 : (imageSize.width - 1)*0.5;
    cy = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;

    // extract vanishing points in order to obtain initial value for the focal length
    int pos = 0;
    for(int i = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);
        Mat matM = objectPoints.colRange(pos, pos + ni);
        Mat _m = imagePoints.colRange(pos, pos + ni);
        pos += ni;

        Matx33d H;
        Mat matH0 = findHomography(matM, _m);
        CV_Assert(matH0.size() == Size(3, 3));
        matH0.convertTo(H, CV_64F);

        H(0, 0) -= H(2, 0)*cx; H(0, 1) -= H(2, 1)*cx; H(0, 2) -= H(2, 2)*cx;
        H(1, 0) -= H(2, 0)*cy; H(1, 1) -= H(2, 1)*cy; H(1, 2) -= H(2, 2)*cy;

        Vec3d h, v, d1, d2;
        Vec4d n;
        for(int j = 0; j < 3; j++ )
        {
            double t0 = H(j, 0), t1 = H(j, 1);
            h[j] = t0; v[j] = t1;
            d1[j] = (t0 + t1)*0.5;
            d2[j] = (t0 - t1)*0.5;
            n[0] += t0*t0; n[1] += t1*t1;
            n[2] += d1[j]*d1[j]; n[3] += d2[j]*d2[j];
        }

        for(int j = 0; j < 4; j++ )
            n[j] = 1./std::sqrt(n[j]);

        for(int j = 0; j < 3; j++ )
        {
            h[j] *= n[0]; v[j] *= n[1];
            d1[j] *= n[2]; d2[j] *= n[3];
        }

        matA(i*2+0, 0) = h[0]*v[0];   matA(i*2+0, 1) = h[1]*v[1];
        matA(i*2+1, 0) = d1[0]*d2[0]; matA(i*2+1, 1) = d1[1]*d2[1];
        matb(i*2+0) = -h[2]*v[2];
        matb(i*2+1) = -d1[2]*d2[2];
    }

    Vec2d f;
    solve(matA, matb, f, DECOMP_NORMAL + DECOMP_SVD);
    fx = std::sqrt(fabs(1./f[0]));
    fy = std::sqrt(fabs(1./f[1]));
    if( aspectRatio != 0 )
    {
        double tf = (fx + fy)/(aspectRatio + 1.);
        fx = aspectRatio*tf;
        fy = tf;
    }
    Mat(Matx33d(fx,  0, cx,
             0, fy, cy,
             0,  0,  1)).copyTo(cameraMatrix);
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

static double calibrateCameraInternal( const CvMat* objectPoints,
                    const CvMat* imagePoints, const CvMat* npoints,
                    CvSize imageSize, int iFixedPoint, CvMat* cameraMatrix, CvMat* distCoeffs,
                                       Mat rvecs, Mat tvecs, Mat newObjPoints, Mat stdDevs,
                                       Mat perViewErr, int flags, const CvTermCriteria& termCrit )
{
    const int NINTRINSIC = CV_CALIB_NINTRINSIC;
    double reprojErr = 0;

    Matx33d A;
    double k[14] = {0};
    CvMat matA = cvMat(3, 3, CV_64F, A.val), _k;
    int nimages, maxPoints = 0, ni = 0, total = 0, nparams, npstep;
    double aspectRatio = 0.;

    // 0. check the parameters & allocate buffers
    if( !CV_IS_MAT(objectPoints) || !CV_IS_MAT(imagePoints) ||
        !CV_IS_MAT(npoints) || !CV_IS_MAT(cameraMatrix) || !CV_IS_MAT(distCoeffs) )
        CV_Error( cv::Error::StsBadArg, "One of required vector arguments is not a valid matrix" );

    if( imageSize.width <= 0 || imageSize.height <= 0 )
        CV_Error( cv::Error::StsOutOfRange, "image width and height must be positive" );

    if( CV_MAT_TYPE(npoints->type) != CV_32SC1 ||
        (npoints->rows != 1 && npoints->cols != 1) )
        CV_Error( cv::Error::StsUnsupportedFormat,
            "the array of point counters must be 1-dimensional integer vector" );
    if(flags & CALIB_TILTED_MODEL)
    {
        //when the tilted sensor model is used the distortion coefficients matrix must have 14 parameters
        if (distCoeffs->cols*distCoeffs->rows != 14)
            CV_Error( cv::Error::StsBadArg, "The tilted sensor model must have 14 parameters in the distortion matrix" );
    }
    else
    {
        //when the thin prism model is used the distortion coefficients matrix must have 12 parameters
        if(flags & CALIB_THIN_PRISM_MODEL)
            if (distCoeffs->cols*distCoeffs->rows != 12)
                CV_Error( cv::Error::StsBadArg, "Thin prism model must have 12 parameters in the distortion matrix" );
    }

    nimages = npoints->rows*npoints->cols;
    npstep = npoints->rows == 1 ? 1 : npoints->step/CV_ELEM_SIZE(npoints->type);


    if( !rvecs.empty() )
    {
        int cn = rvecs.channels();
        CV_Assert(rvecs.depth() == CV_32F || rvecs.depth() == CV_64F);
        CV_Assert((rvecs.rows == nimages && (rvecs.cols*cn == 3 || rvecs.cols*cn == 3)) ||
                  (rvecs.rows == 1 && rvecs.cols == nimages && cn == 3));
    }

    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints->data.i[0] - 1;

    if( !tvecs.empty() )
    {
        int cn = tvecs.channels();
        CV_Assert(tvecs.depth() == CV_32F || tvecs.depth() == CV_64F);
        CV_Assert((tvecs.rows == nimages && tvecs.cols*cn == 3) ||
                  (tvecs.rows == 1 && tvecs.cols == nimages && cn == 3));
    }

    if( (CV_MAT_TYPE(cameraMatrix->type) != CV_32FC1 &&
        CV_MAT_TYPE(cameraMatrix->type) != CV_64FC1) ||
        cameraMatrix->rows != 3 || cameraMatrix->cols != 3 )
        CV_Error( cv::Error::StsBadArg,
            "Intrinsic parameters must be 3x3 floating-point matrix" );

    if( (CV_MAT_TYPE(distCoeffs->type) != CV_32FC1 &&
        CV_MAT_TYPE(distCoeffs->type) != CV_64FC1) ||
        (distCoeffs->cols != 1 && distCoeffs->rows != 1) ||
        (distCoeffs->cols*distCoeffs->rows != 4 &&
        distCoeffs->cols*distCoeffs->rows != 5 &&
        distCoeffs->cols*distCoeffs->rows != 8 &&
        distCoeffs->cols*distCoeffs->rows != 12 &&
        distCoeffs->cols*distCoeffs->rows != 14) )
        CV_Error( cv::Error::StsBadArg, cvDistCoeffErr );

    for(int i = 0; i < nimages; i++ )
    {
        ni = npoints->data.i[i*npstep];
        if( ni < 4 )
        {
            CV_Error_( cv::Error::StsOutOfRange, ("The number of points in the view #%d is < 4", i));
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    if( !newObjPoints.empty() )
    {
        int cn = newObjPoints.channels();
        CV_Assert(newObjPoints.depth() == CV_32F || newObjPoints.depth() == CV_64F);
        CV_Assert((newObjPoints.rows == maxPoints && newObjPoints.cols*cn == 3) ||
                  (newObjPoints.rows == 1 && newObjPoints.cols == maxPoints && cn == 3));
    }

    if( !stdDevs.empty() && releaseObject )
    {
        int cn = stdDevs.channels();
        CV_Assert(stdDevs.depth() == CV_32F || stdDevs.depth() == CV_64F);
        int nstddev = nimages*6 + NINTRINSIC + (releaseObject ? maxPoints*3 : 0);

        CV_Assert((stdDevs.rows == nstddev && stdDevs.cols*cn == 1) ||
                  (stdDevs.rows == 1 && stdDevs.cols == nstddev && cn == 1));
    }

    Mat matM( 1, total, CV_64FC3 );
    Mat _m( 1, total, CV_64FC2 );
    Mat allErrors(1, total, CV_64FC2);

    if(CV_MAT_CN(objectPoints->type) == 3) {
        cvarrToMat(objectPoints).convertTo(matM, CV_64F);
    } else {
        convertPointsHomogeneous(cvarrToMat(objectPoints), matM);
    }

    if(CV_MAT_CN(imagePoints->type) == 2) {
        cvarrToMat(imagePoints).convertTo(_m, CV_64F);
    } else {
        convertPointsHomogeneous(cvarrToMat(imagePoints), _m);
    }

    nparams = NINTRINSIC + nimages*6;
    if( releaseObject )
        nparams += maxPoints * 3;

    _k = cvMat( distCoeffs->rows, distCoeffs->cols, CV_MAKETYPE(CV_64F,CV_MAT_CN(distCoeffs->type)), k);
    if( distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 8 )
    {
        if( distCoeffs->rows*distCoeffs->cols*CV_MAT_CN(distCoeffs->type) < 5 )
            flags |= CALIB_FIX_K3;
        flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
    }
    const double minValidAspectRatio = 0.01;
    const double maxValidAspectRatio = 100.0;

    // 1. initialize intrinsic parameters & LM solver
    if( flags & CALIB_USE_INTRINSIC_GUESS )
    {
        cvConvert( cameraMatrix, &matA );
        if( A(0, 0) <= 0 || A(1, 1) <= 0 )
            CV_Error( cv::Error::StsOutOfRange, "Focal length (fx and fy) must be positive" );
        if( A(0, 2) < 0 || A(0, 2) >= imageSize.width ||
            A(1, 2) < 0 || A(1, 2) >= imageSize.height )
            CV_Error( cv::Error::StsOutOfRange, "Principal point must be within the image" );
        if( fabs(A(0, 1)) > 1e-5 )
            CV_Error( cv::Error::StsOutOfRange, "Non-zero skew is not supported by the function" );
        if( fabs(A(1, 0)) > 1e-5 || fabs(A(2, 0)) > 1e-5 ||
            fabs(A(2, 1)) > 1e-5 || fabs(A(2,2)-1) > 1e-5 )
            CV_Error( cv::Error::StsOutOfRange,
                "The intrinsic matrix must have [fx 0 cx; 0 fy cy; 0 0 1] shape" );
        A(0, 1) = A(1, 0) = A(2, 0) = A(2, 1) = 0.;
        A(2, 2) = 1.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = A(0, 0)/A(1, 1);

            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        cvConvert( distCoeffs, &_k );
    }
    else
    {
        Scalar mean, sdv;
        meanStdDev(matM, mean, sdv);
        if( fabs(mean[2]) > 1e-5 || fabs(sdv[2]) > 1e-5 )
            CV_Error( cv::Error::StsBadArg,
            "For non-planar calibration rigs the initial intrinsic matrix must be specified" );
        for(int i = 0; i < total; i++ )
            matM.at<Point3d>(i).z = 0.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = cvmGet(cameraMatrix,0,0);
            aspectRatio /= cvmGet(cameraMatrix,1,1);
            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        initIntrinsicParams2D( matM, _m, cvarrToMat(npoints), imageSize, cvarrToMat(&matA), aspectRatio );
    }

    CvLevMarq solver( nparams, 0, termCrit );

    Mat _Ji( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
    Mat _Je( maxPoints*2, 6, CV_64FC1 );
    Mat _err( maxPoints*2, 1, CV_64FC1 );

    const bool allocJo = (solver.state == CvLevMarq::CALC_J) || !stdDevs.empty() || releaseObject;
    Mat _Jo = allocJo ? Mat( maxPoints*2, maxPoints*3, CV_64FC1, Scalar(0) ) : Mat();

    if(flags & CALIB_USE_LU) {
        solver.solveMethod = DECOMP_LU;
    }
    else if(flags & CALIB_USE_QR) {
        solver.solveMethod = DECOMP_QR;
    }

    {
    double* param = solver.param->data.db;
    uchar* mask = solver.mask->data.ptr;

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
        int s = NINTRINSIC + nimages * 6;

        std::copy( matM.ptr<double>(), matM.ptr<double>( 0, maxPoints - 1 ) + 3,
                   param + NINTRINSIC + nimages * 6 );
        // fix points
        mask[s + 0] = 0;
        mask[s + 1] = 0;
        mask[s + 2] = 0;
        mask[s + iFixedPoint * 3 + 0] = 0;
        mask[s + iFixedPoint * 3 + 1] = 0;
        mask[s + iFixedPoint * 3 + 2] = 0;
        mask[nparams - 1] = 0;
    }
    }

    Mat mask = cvarrToMat(solver.mask);
    int nparams_nz = countNonZero(mask);
    if (nparams_nz >= 2 * total)
        CV_Error_(Error::StsBadArg,
                  ("There should be less vars to optimize (having %d) than the number of residuals (%d = 2 per point)", nparams_nz, 2 * total));

    // 2. initialize extrinsic parameters
    for(int i = 0, pos = 0; i < nimages; i++, pos += ni )
    {
        CvMat _ri, _ti;
        ni = npoints->data.i[i*npstep];

        cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
        cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

        CvMat _Mi = cvMat(matM.colRange(pos, pos + ni));
        CvMat _mi = cvMat(_m.colRange(pos, pos + ni));

        Mat r_mat = cvarrToMat(&_ri), t_mat = cvarrToMat(&_ti);
        findExtrinsicCameraParams2( cvarrToMat(&_Mi), cvarrToMat(&_mi), cvarrToMat(&matA),
                                    cvarrToMat(&_k), r_mat, t_mat, /*useExtrinsicGuess=*/0 );
    }

    //std::cout << "single camera calib. param before LM: " << param0.t() << "\n";
    //std::cout << "single camera calib. mask: " << mask0.t() << "\n";

    // 3. run the optimization
    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;
        bool proceed = solver.updateAlt( _param, _JtJ, _JtErr, _errNorm );
        double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;
        bool calcJ = solver.state == CvLevMarq::CALC_J || (!proceed && !stdDevs.empty());

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            param[0] = param[1]*aspectRatio;
            pparam[0] = pparam[1]*aspectRatio;
        }

        A(0, 0) = param[0]; A(1, 1) = param[1]; A(0, 2) = param[2]; A(1, 2) = param[3];
        std::copy(param + 4, param + 4 + 14, k);

        if ( !proceed && stdDevs.empty() && perViewErr.empty() )
            break;
        else if ( !proceed && !stdDevs.empty() )
            cvZero(_JtJ);

        reprojErr = 0;

        for(int i = 0, pos = 0; i < nimages; i++, pos += ni )
        {
            CvMat _ri, _ti;
            ni = npoints->data.i[i*npstep];

            cvGetRows( solver.param, &_ri, NINTRINSIC + i*6, NINTRINSIC + i*6 + 3 );
            cvGetRows( solver.param, &_ti, NINTRINSIC + i*6 + 3, NINTRINSIC + i*6 + 6 );

            CvMat _Mi = cvMat(matM.colRange(pos, pos + ni));
            if( releaseObject )
            {
                cvGetRows( solver.param, &_Mi, NINTRINSIC + nimages * 6,
                           NINTRINSIC + nimages * 6 + ni * 3 );
                cvReshape( &_Mi, &_Mi, 3, 1 );
            }
            CvMat _mi = cvMat(_m.colRange(pos, pos + ni));
            CvMat _me = cvMat(allErrors.colRange(pos, pos + ni));

            _Je.resize(ni*2); _Ji.resize(ni*2); _err.resize(ni*2);
            _Jo.resize(ni*2);

            CvMat _mp = cvMat(_err.reshape(2, 1));

            if( calcJ )
            {
                projectPoints( cvarrToMat(&_Mi), cvarrToMat(&_ri), cvarrToMat(&_ti), cvarrToMat(&matA),
                               cvarrToMat(&_k), cvarrToMat(&_mp), _Je.colRange(0, 3), _Je.colRange(3, 6),
                               (flags & CALIB_FIX_FOCAL_LENGTH) ? noArray() : _Ji.colRange(0, 2),
                               (flags & CALIB_FIX_PRINCIPAL_POINT) ? noArray() : _Ji.colRange(2, 4),
                               _Ji.colRange(4, 4 + _k.cols * _k.rows), (_Jo.empty()) ? noArray() : _Jo.colRange(0, ni * 3),
                               (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0);
            }
            else
                projectPoints( cvarrToMat(&_Mi), cvarrToMat(&_ri), cvarrToMat(&_ti), cvarrToMat(&matA),
                               cvarrToMat(&_k), cvarrToMat(&_mp) );

            cvSub( &_mp, &_mi, &_mp );
            if (!perViewErr.empty() || !stdDevs.empty())
                cvCopy(&_mp, &_me);

            if( calcJ )
            {
                Mat JtJ(cvarrToMat(_JtJ)), JtErr(cvarrToMat(_JtErr));

                // see HZ: (A6.14) for details on the structure of the Jacobian
                JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += _Ji.t() * _Ji;
                JtJ(Rect(NINTRINSIC + i * 6, NINTRINSIC + i * 6, 6, 6)) = _Je.t() * _Je;
                JtJ(Rect(NINTRINSIC + i * 6, 0, 6, NINTRINSIC)) = _Ji.t() * _Je;
                if( releaseObject )
                {
                    JtJ(Rect(NINTRINSIC + nimages * 6, 0, maxPoints * 3, NINTRINSIC)) += _Ji.t() * _Jo;
                    JtJ(Rect(NINTRINSIC + nimages * 6, NINTRINSIC + i * 6, maxPoints * 3, 6))
                        += _Je.t() * _Jo;
                    JtJ(Rect(NINTRINSIC + nimages * 6, NINTRINSIC + nimages * 6, maxPoints * 3, maxPoints * 3))
                        += _Jo.t() * _Jo;
                }

                JtErr.rowRange(0, NINTRINSIC) += _Ji.t() * _err;
                JtErr.rowRange(NINTRINSIC + i * 6, NINTRINSIC + (i + 1) * 6) = _Je.t() * _err;
                if( releaseObject )
                {
                    JtErr.rowRange(NINTRINSIC + nimages * 6, nparams) += _Jo.t() * _err;
                }
            }

            double viewErr = norm(_err, NORM_L2SQR);

            if( !perViewErr.empty() )
                perViewErr.at<double>(i) = std::sqrt(viewErr / ni);

            reprojErr += viewErr;
        }
        if( _errNorm )
            *_errNorm = reprojErr;

        if( !proceed )
        {
            if( !stdDevs.empty() )
            {
                Mat JtJinv, JtJN;
                JtJN.create(nparams_nz, nparams_nz, CV_64F);
                subMatrix(cvarrToMat(_JtJ), JtJN, mask, mask);
                completeSymm(JtJN, false);
                cv::invert(JtJN, JtJinv, DECOMP_SVD);
                // an explanation of that denominator correction can be found here:
                // R. Hartley, A. Zisserman, Multiple View Geometry in Computer Vision, 2004, section 5.1.3, page 134
                // see the discussion for more details: https://github.com/opencv/opencv/pull/22992
                int nErrors = 2 * total - nparams_nz;
                double sigma2 = norm(allErrors, NORM_L2SQR) / nErrors;
                int j = 0;
                for ( int s = 0; s < nparams; s++ )
                {
                    stdDevs.at<double>(s) = mask.data[s] ? std::sqrt(JtJinv.at<double>(j,j) * sigma2) : 0.0;
                    if( mask.data[s] )
                        j++;
                }
            }
            break;
        }
    }

    // 4. store the results
    cvConvert( &matA, cameraMatrix );
    cvConvert( &_k, distCoeffs );
    if( !newObjPoints.empty() && releaseObject )
    {
        int s = NINTRINSIC + nimages * 6;
        Mat _Mi = cvarrToMat(solver.param).rowRange(s, s + maxPoints * 3);
        _Mi.reshape(3, 1).convertTo(newObjPoints, newObjPoints.type());
    }

    for(int i = 0; i < nimages; i++ )
    {
        if( !rvecs.empty() )
        {
            Mat src = Mat(3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6);
            if( rvecs.rows == nimages && rvecs.cols*rvecs.channels() == 9 )
            {
                Mat dst(3, 3, rvecs.depth(), rvecs.ptr(i));
                Rodrigues(src, A);
                Mat(A).convertTo(dst, dst.type());
            }
            else
            {
                Mat dst(3, 1, rvecs.depth(), rvecs.rows == 1 ?
                    rvecs.data + i*rvecs.elemSize() : rvecs.ptr(i));
                src.convertTo(dst, dst.type());
            }
        }
        if( !tvecs.empty() )
        {
            Mat src(3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 + 3);
            Mat dst(3, 1, tvecs.depth(), tvecs.rows == 1 ?
                    tvecs.data + i*tvecs.elemSize() : tvecs.ptr(i));
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

static double stereoCalibrateImpl( const CvMat* _objectPoints, const CvMat* _imagePoints1,
                        const CvMat* _imagePoints2, const CvMat* _npoints,
                        CvMat* _cameraMatrix1, CvMat* _distCoeffs1,
                        CvMat* _cameraMatrix2, CvMat* _distCoeffs2,
                        CvSize imageSize, CvMat* matR, CvMat* matT,
        Mat matE, Mat matF,
        Mat rvecs, Mat tvecs,
        Mat perViewErr, int flags,
        CvTermCriteria termCrit )
{
    const int NINTRINSIC = 18;
    Ptr<CvMat> npoints, imagePoints[2], objectPoints, RT0;
    double reprojErr = 0;

    double A[2][9], dk[2][14]={{0}}, rlr[9];
    CvMat K[2], Dist[2], om_LR, T_LR;
    CvMat R_LR = cvMat(3, 3, CV_64F, rlr);
    int i, k, p, ni = 0, ofs, nimages, pointsTotal, maxPoints = 0;
    int nparams;
    bool recomputeIntrinsics = false;
    double aspectRatio[2] = {0, 0};

    CV_Assert( CV_IS_MAT(_imagePoints1) && CV_IS_MAT(_imagePoints2) &&
               CV_IS_MAT(_objectPoints) && CV_IS_MAT(_npoints) &&
               CV_IS_MAT(matR) && CV_IS_MAT(matT) );

    CV_Assert( CV_ARE_TYPES_EQ(_imagePoints1, _imagePoints2) &&
               CV_ARE_DEPTHS_EQ(_imagePoints1, _objectPoints) );

    CV_Assert( (_npoints->cols == 1 || _npoints->rows == 1) &&
               CV_MAT_TYPE(_npoints->type) == CV_32SC1 );

    nimages = _npoints->cols + _npoints->rows - 1;
    npoints.reset(cvCreateMat( _npoints->rows, _npoints->cols, _npoints->type ));
    cvCopy( _npoints, npoints );

    for( i = 0, pointsTotal = 0; i < nimages; i++ )
    {
        maxPoints = MAX(maxPoints, npoints->data.i[i]);
        pointsTotal += npoints->data.i[i];
    }

    objectPoints.reset(cvCreateMat( _objectPoints->rows, _objectPoints->cols,
                                    CV_64FC(CV_MAT_CN(_objectPoints->type))));
    cvConvert( _objectPoints, objectPoints );
    cvReshape( objectPoints, objectPoints, 3, 1 );

    if( !rvecs.empty() )
    {
        int cn = rvecs.channels();
        int depth = rvecs.depth();
        if( (depth != CV_32F && depth != CV_64F) ||
            ((rvecs.rows != nimages || (rvecs.cols*cn != 3 && rvecs.cols*cn != 9)) &&
             (rvecs.rows != 1 || rvecs.cols != nimages || cn != 3)) )
            CV_Error( cv::Error::StsBadArg, "the output array of rotation vectors must be 3-channel "
                "1xn or nx1 array or 1-channel nx3 or nx9 array, where n is the number of views" );
    }
    if( !tvecs.empty() )
    {
        int cn = tvecs.channels();
        int depth = tvecs.depth();
        if( (depth != CV_32F && depth != CV_64F) ||
            ((tvecs.rows != nimages || tvecs.cols*cn != 3) &&
             (tvecs.rows != 1 || tvecs.cols != nimages || cn != 3)) )
            CV_Error( cv::Error::StsBadArg, "the output array of translation vectors must be 3-channel "
                "1xn or nx1 array or 1-channel nx3 array, where n is the number of views" );
    }

    for( k = 0; k < 2; k++ )
    {
        const CvMat* points = k == 0 ? _imagePoints1 : _imagePoints2;
        const CvMat* cameraMatrix = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const CvMat* distCoeffs = k == 0 ? _distCoeffs1 : _distCoeffs2;

        int cn = CV_MAT_CN(_imagePoints1->type);
        CV_Assert( (CV_MAT_DEPTH(_imagePoints1->type) == CV_32F ||
                CV_MAT_DEPTH(_imagePoints1->type) == CV_64F) &&
               ((_imagePoints1->rows == pointsTotal && _imagePoints1->cols*cn == 2) ||
                (_imagePoints1->rows == 1 && _imagePoints1->cols == pointsTotal && cn == 2)) );

        K[k] = cvMat(3,3,CV_64F,A[k]);
        Dist[k] = cvMat(1,14,CV_64F,dk[k]);

        imagePoints[k].reset(cvCreateMat( points->rows, points->cols, CV_64FC(CV_MAT_CN(points->type))));
        cvConvert( points, imagePoints[k] );
        cvReshape( imagePoints[k], imagePoints[k], 2, 1 );

        if( flags & (CALIB_FIX_INTRINSIC|CALIB_USE_INTRINSIC_GUESS|
            CALIB_FIX_ASPECT_RATIO|CALIB_FIX_FOCAL_LENGTH) )
            cvConvert( cameraMatrix, &K[k] );

        if( flags & (CALIB_FIX_INTRINSIC|CALIB_USE_INTRINSIC_GUESS|
            CALIB_FIX_K1|CALIB_FIX_K2|CALIB_FIX_K3|CALIB_FIX_K4|CALIB_FIX_K5|CALIB_FIX_K6|CALIB_FIX_TANGENT_DIST) )
        {
            CvMat tdist = cvMat( distCoeffs->rows, distCoeffs->cols,
                CV_MAKETYPE(CV_64F,CV_MAT_CN(distCoeffs->type)), Dist[k].data.db );
            cvConvert( distCoeffs, &tdist );
        }

        if( !(flags & (CALIB_FIX_INTRINSIC|CALIB_USE_INTRINSIC_GUESS)))
        {
            calibrateCameraInternal( objectPoints, imagePoints[k],
                npoints, imageSize, -1, &K[k], &Dist[k], Mat(), Mat(),
                                    Mat(), Mat(), Mat(), flags, termCrit );
        }
    }

    if( flags & CALIB_SAME_FOCAL_LENGTH )
    {
        static const int avg_idx[] = { 0, 4, 2, 5, -1 };
        for( k = 0; avg_idx[k] >= 0; k++ )
            A[0][avg_idx[k]] = A[1][avg_idx[k]] = (A[0][avg_idx[k]] + A[1][avg_idx[k]])*0.5;
    }

    if( flags & CALIB_FIX_ASPECT_RATIO )
    {
        for( k = 0; k < 2; k++ )
            aspectRatio[k] = A[k][0]/A[k][4];
    }

    recomputeIntrinsics = (flags & CALIB_FIX_INTRINSIC) == 0;

    Mat err( maxPoints*2, 1, CV_64F );
    Mat Je( maxPoints*2, 6, CV_64F );
    Mat J_LR( maxPoints*2, 6, CV_64F );
    Mat Ji( maxPoints*2, NINTRINSIC, CV_64F, Scalar(0) );

    // we optimize for the inter-camera R(3),t(3), then, optionally,
    // for intrinisic parameters of each camera ((fx,fy,cx,cy,k1,k2,p1,p2) ~ 8 parameters).
    // Param mapping is:
    // - from 0 next 6: stereo pair Rt, from 6+i*6 next 6: Rt for each ith camera of nimages,
    // - from 6*(nimages+1) next NINTRINSICS: intrinsics for 1st camera: fx, fy, cx, cy, 14 x dist
    // - next NINTRINSICS: the same for for 2nd camera
    nparams = 6*(nimages+1) + (recomputeIntrinsics ? NINTRINSIC*2 : 0);

    CvLevMarq solver( nparams, 0, termCrit );

    if(flags & CALIB_USE_LU) {
        solver.solveMethod = DECOMP_LU;
    }

    if( recomputeIntrinsics )
    {
        uchar* imask = solver.mask->data.ptr + nparams - NINTRINSIC*2;
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
    RT0.reset(cvCreateMat( 6, nimages, CV_64F ));
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
    for( i = ofs = 0; i < nimages; ofs += ni, i++ )
    {
        ni = npoints->data.i[i];
        CvMat objpt_i;
        double _om[2][3], r[2][9], t[2][3];
        CvMat om[2], R[2], T[2], imgpt_i[2];

        objpt_i = cvMat(1, ni, CV_64FC3, objectPoints->data.db + ofs*3);
        for( k = 0; k < 2; k++ )
        {
            imgpt_i[k] = cvMat(1, ni, CV_64FC2, imagePoints[k]->data.db + ofs*2);
            om[k] = cvMat(3, 1, CV_64F, _om[k]);
            R[k] = cvMat(3, 3, CV_64F, r[k]);
            T[k] = cvMat(3, 1, CV_64F, t[k]);

            Mat r_mat = cvarrToMat(&om[k]), t_mat = cvarrToMat(&T[k]);
            findExtrinsicCameraParams2( cvarrToMat(&objpt_i), cvarrToMat(&imgpt_i[k]),
                                        cvarrToMat(&K[k]), cvarrToMat(&Dist[k]),
                                        r_mat, t_mat, /*useExtrinsicGuess=*/0 );
            Rodrigues( cvarrToMat(&om[k]), cvarrToMat(&R[k]) );
            if( k == 0 )
            {
                // save initial om_left and T_left
                solver.param->data.db[(i+1)*6] = _om[0][0];
                solver.param->data.db[(i+1)*6 + 1] = _om[0][1];
                solver.param->data.db[(i+1)*6 + 2] = _om[0][2];
                solver.param->data.db[(i+1)*6 + 3] = t[0][0];
                solver.param->data.db[(i+1)*6 + 4] = t[0][1];
                solver.param->data.db[(i+1)*6 + 5] = t[0][2];
            }
        }
        cvGEMM( &R[1], &R[0], 1, 0, 0, &R[0], CV_GEMM_B_T );
        cvGEMM( &R[0], &T[0], -1, &T[1], 1, &T[1] );
        Rodrigues( cvarrToMat(&R[0]), cvarrToMat(&T[0]) );
        RT0->data.db[i] = t[0][0];
        RT0->data.db[i + nimages] = t[0][1];
        RT0->data.db[i + nimages*2] = t[0][2];
        RT0->data.db[i + nimages*3] = t[1][0];
        RT0->data.db[i + nimages*4] = t[1][1];
        RT0->data.db[i + nimages*5] = t[1][2];
    }

    if(flags & CALIB_USE_EXTRINSIC_GUESS)
    {
        Vec3d R, T;
        cvarrToMat(matT).convertTo(T, CV_64F);

        if( matR->rows == 3 && matR->cols == 3 )
            Rodrigues(cvarrToMat(matR), R);
        else
            cvarrToMat(matR).convertTo(R, CV_64F);

        solver.param->data.db[0] = R[0];
        solver.param->data.db[1] = R[1];
        solver.param->data.db[2] = R[2];
        solver.param->data.db[3] = T[0];
        solver.param->data.db[4] = T[1];
        solver.param->data.db[5] = T[2];
    }
    else
    {
        // find the medians and save the first 6 parameters
        for( i = 0; i < 6; i++ )
        {
            qsort( RT0->data.db + i*nimages, nimages, CV_ELEM_SIZE(RT0->type), dbCmp );
            solver.param->data.db[i] = nimages % 2 != 0 ? RT0->data.db[i*nimages + nimages/2] :
                (RT0->data.db[i*nimages + nimages/2 - 1] + RT0->data.db[i*nimages + nimages/2])*0.5;
        }
    }

    if( recomputeIntrinsics )
        for( k = 0; k < 2; k++ )
        {
            double* iparam = solver.param->data.db + (nimages+1)*6 + k*NINTRINSIC;
            if( flags & CALIB_ZERO_TANGENT_DIST )
                dk[k][2] = dk[k][3] = 0;
            iparam[0] = A[k][0]; iparam[1] = A[k][4]; iparam[2] = A[k][2]; iparam[3] = A[k][5];
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

    om_LR = cvMat(3, 1, CV_64F, solver.param->data.db);
    T_LR = cvMat(3, 1, CV_64F, solver.param->data.db + 3);

    for(;;)
    {
        const CvMat* param = 0;
        CvMat *JtJ = 0, *JtErr = 0;
        double *_errNorm = 0;
        double _omR[3], _tR[3];
        double _dr3dr1[9], _dr3dr2[9], /*_dt3dr1[9],*/ _dt3dr2[9], _dt3dt1[9], _dt3dt2[9];
        CvMat dr3dr1 = cvMat(3, 3, CV_64F, _dr3dr1);
        CvMat dr3dr2 = cvMat(3, 3, CV_64F, _dr3dr2);
        //CvMat dt3dr1 = cvMat(3, 3, CV_64F, _dt3dr1);
        CvMat dt3dr2 = cvMat(3, 3, CV_64F, _dt3dr2);
        CvMat dt3dt1 = cvMat(3, 3, CV_64F, _dt3dt1);
        CvMat dt3dt2 = cvMat(3, 3, CV_64F, _dt3dt2);
        CvMat om[2], T[2], imgpt_i[2];

        if( !solver.updateAlt( param, JtJ, JtErr, _errNorm ))
            break;
        reprojErr = 0;

        Rodrigues( cvarrToMat(&om_LR), cvarrToMat(&R_LR) );
        om[1] = cvMat(3,1,CV_64F,_omR);
        T[1] = cvMat(3,1,CV_64F,_tR);

        if( recomputeIntrinsics )
        {
            double* iparam = solver.param->data.db + (nimages+1)*6;
            double* ipparam = solver.prevParam->data.db + (nimages+1)*6;

            if( flags & CALIB_SAME_FOCAL_LENGTH )
            {
                iparam[NINTRINSIC] = iparam[0];
                iparam[NINTRINSIC+1] = iparam[1];
                ipparam[NINTRINSIC] = ipparam[0];
                ipparam[NINTRINSIC+1] = ipparam[1];
            }
            if( flags & CALIB_FIX_ASPECT_RATIO )
            {
                iparam[0] = iparam[1]*aspectRatio[0];
                iparam[NINTRINSIC] = iparam[NINTRINSIC+1]*aspectRatio[1];
                ipparam[0] = ipparam[1]*aspectRatio[0];
                ipparam[NINTRINSIC] = ipparam[NINTRINSIC+1]*aspectRatio[1];
            }
            for( k = 0; k < 2; k++ )
            {
                A[k][0] = iparam[k*NINTRINSIC+0];
                A[k][4] = iparam[k*NINTRINSIC+1];
                A[k][2] = iparam[k*NINTRINSIC+2];
                A[k][5] = iparam[k*NINTRINSIC+3];
                dk[k][0] = iparam[k*NINTRINSIC+4];
                dk[k][1] = iparam[k*NINTRINSIC+5];
                dk[k][2] = iparam[k*NINTRINSIC+6];
                dk[k][3] = iparam[k*NINTRINSIC+7];
                dk[k][4] = iparam[k*NINTRINSIC+8];
                dk[k][5] = iparam[k*NINTRINSIC+9];
                dk[k][6] = iparam[k*NINTRINSIC+10];
                dk[k][7] = iparam[k*NINTRINSIC+11];
                dk[k][8] = iparam[k*NINTRINSIC+12];
                dk[k][9] = iparam[k*NINTRINSIC+13];
                dk[k][10] = iparam[k*NINTRINSIC+14];
                dk[k][11] = iparam[k*NINTRINSIC+15];
                dk[k][12] = iparam[k*NINTRINSIC+16];
                dk[k][13] = iparam[k*NINTRINSIC+17];
            }
        }

        for( i = ofs = 0; i < nimages; ofs += ni, i++ )
        {
            ni = npoints->data.i[i];
            CvMat objpt_i;

            om[0] = cvMat(3,1,CV_64F,solver.param->data.db+(i+1)*6);
            T[0] = cvMat(3,1,CV_64F,solver.param->data.db+(i+1)*6+3);

            if( JtJ || JtErr )
                composeRT( cvarrToMat(&om[0]), cvarrToMat(&T[0]), cvarrToMat(&om_LR),
                           cvarrToMat(&T_LR), cvarrToMat(&om[1]), cvarrToMat(&T[1]),
                           cvarrToMat(&dr3dr1), noArray(), cvarrToMat(&dr3dr2),
                           noArray(), noArray(), cvarrToMat(&dt3dt1), cvarrToMat(&dt3dr2),
                           cvarrToMat(&dt3dt2 ) );
            else
                composeRT( cvarrToMat(&om[0]), cvarrToMat(&T[0]), cvarrToMat(&om_LR),
                           cvarrToMat(&T_LR), cvarrToMat(&om[1]), cvarrToMat(&T[1]) );

            objpt_i = cvMat(1, ni, CV_64FC3, objectPoints->data.db + ofs*3);
            err.resize(ni*2); Je.resize(ni*2); J_LR.resize(ni*2); Ji.resize(ni*2);

            CvMat tmpimagePoints = cvMat(err.reshape(2, 1));

            for( k = 0; k < 2; k++ )
            {
                imgpt_i[k] = cvMat(1, ni, CV_64FC2, imagePoints[k]->data.db + ofs*2);

                if( JtJ || JtErr )
                    projectPoints( cvarrToMat(&objpt_i), cvarrToMat(&om[k]), cvarrToMat(&T[k]),
                            cvarrToMat(&K[k]), cvarrToMat(&Dist[k]),
                            err.reshape(2, 1), Je.colRange(0, 3), Je.colRange(3, 6),
                            Ji.colRange(0, 2), Ji.colRange(2, 4), Ji.colRange(4, 4 + Dist[k].cols * Dist[k].rows), noArray(),
                            (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio[k] : 0);
                else
                    projectPoints( cvarrToMat(&objpt_i), cvarrToMat(&om[k]), cvarrToMat(&T[k]),
                                   cvarrToMat(&K[k]), cvarrToMat(&Dist[k]), cvarrToMat(&tmpimagePoints) );
                cvSub( &tmpimagePoints, &imgpt_i[k], &tmpimagePoints );

                if( solver.state == CvLevMarq::CALC_J )
                {
                    int iofs = (nimages+1)*6 + k*NINTRINSIC, eofs = (i+1)*6;
                    CV_Assert( JtJ && JtErr );

                    Mat _JtJ(cvarrToMat(JtJ)), _JtErr(cvarrToMat(JtErr));

                    if( k == 1 )
                    {
                        // d(err_{x|y}R) ~ de3
                        // convert de3/{dr3,dt3} => de3{dr1,dt1} & de3{dr2,dt2}
                        for( p = 0; p < ni*2; p++ )
                        {
                            CvMat de3dr3 = cvMat( 1, 3, CV_64F, Je.ptr(p));
                            CvMat de3dt3 = cvMat( 1, 3, CV_64F, de3dr3.data.db + 3 );
                            CvMat de3dr2 = cvMat( 1, 3, CV_64F, J_LR.ptr(p) );
                            CvMat de3dt2 = cvMat( 1, 3, CV_64F, de3dr2.data.db + 3 );
                            double _de3dr1[3], _de3dt1[3];
                            CvMat de3dr1 = cvMat( 1, 3, CV_64F, _de3dr1 );
                            CvMat de3dt1 = cvMat( 1, 3, CV_64F, _de3dt1 );

                            cvMatMul( &de3dr3, &dr3dr1, &de3dr1 );
                            cvMatMul( &de3dt3, &dt3dt1, &de3dt1 );

                            cvMatMul( &de3dr3, &dr3dr2, &de3dr2 );
                            cvMatMulAdd( &de3dt3, &dt3dr2, &de3dr2, &de3dr2 );

                            cvMatMul( &de3dt3, &dt3dt2, &de3dt2 );

                            cvCopy( &de3dr1, &de3dr3 );
                            cvCopy( &de3dt1, &de3dt3 );
                        }

                        _JtJ(Rect(0, 0, 6, 6)) += J_LR.t()*J_LR;
                        _JtJ(Rect(eofs, 0, 6, 6)) = J_LR.t()*Je;
                        _JtErr.rowRange(0, 6) += J_LR.t()*err;
                    }

                    _JtJ(Rect(eofs, eofs, 6, 6)) += Je.t()*Je;
                    _JtErr.rowRange(eofs, eofs + 6) += Je.t()*err;

                    if( recomputeIntrinsics )
                    {
                        _JtJ(Rect(iofs, iofs, NINTRINSIC, NINTRINSIC)) += Ji.t()*Ji;
                        _JtJ(Rect(iofs, eofs, NINTRINSIC, 6)) += Je.t()*Ji;
                        if( k == 1 )
                        {
                            _JtJ(Rect(iofs, 0, NINTRINSIC, 6)) += J_LR.t()*Ji;
                        }
                        _JtErr.rowRange(iofs, iofs + NINTRINSIC) += Ji.t()*err;
                    }
                }

                double viewErr = norm(err, NORM_L2SQR);
                if(!perViewErr.empty())
                    perViewErr.at<double>(i, k) = std::sqrt(viewErr/ni);
                reprojErr += viewErr;
            }
        }
        if(_errNorm)
            *_errNorm = reprojErr;
    }

    Rodrigues( cvarrToMat(&om_LR), cvarrToMat(&R_LR) );
    if( matR->rows == 1 || matR->cols == 1 )
        cvConvert( &om_LR, matR );
    else
        cvConvert( &R_LR, matR );
    cvConvert( &T_LR, matT );

    if( recomputeIntrinsics )
    {
        cvConvert( &K[0], _cameraMatrix1 );
        cvConvert( &K[1], _cameraMatrix2 );

        for( k = 0; k < 2; k++ )
        {
            CvMat* distCoeffs = k == 0 ? _distCoeffs1 : _distCoeffs2;
            CvMat tdist = cvMat( distCoeffs->rows, distCoeffs->cols,
                CV_MAKETYPE(CV_64F,CV_MAT_CN(distCoeffs->type)), Dist[k].data.db );
            cvConvert( &tdist, distCoeffs );
        }
    }

    if( !matE.empty() || !matF.empty() )
    {
        double* t = T_LR.data.db;
        Matx33d Tx(0, -t[2], t[1],
                   t[2], 0, -t[0],
                  -t[1], t[0], 0);
        Matx33d E = Mat(Tx*cvarrToMat(&R_LR));
        if( !matE.empty() )
            Mat(E).convertTo(matE, matE.depth());
        if( !matF.empty())
        {
            Matx33d iA0 = Mat(cvarrToMat(&K[0]).inv()), iA1 = Mat(cvarrToMat(&K[1]).inv());
            Matx33d F = iA1.t() * E * iA0;
            Mat(F).convertTo(matF, matF.depth(), fabs(F(2,2)) > 0 ? 1./F(2,2) : 1.);
        }
    }

    double* param = solver.param->data.db;
    Mat r1d = rvecs.empty() ? Mat() : rvecs.reshape(1, nimages);
    Mat t1d = tvecs.empty() ? Mat() : tvecs.reshape(1, nimages);
    for( i = 0; i < nimages; i++ )
    {
        int idx = (i + 1) * 6;

        if( !rvecs.empty() )
        {
            Vec3d srcR(param[idx + 0], param[idx + 1], param[idx + 2]);
            if( rvecs.rows * rvecs.cols * rvecs.channels() == nimages * 9 )
            {
                Matx33d rod;
                Rodrigues(srcR, rod);
                Mat(rod).convertTo(r1d.row(i).reshape(1, 3), rvecs.depth());
            }
            else if (rvecs.rows * rvecs.cols * rvecs.channels() == nimages * 3 )
            {
                Mat(Mat(srcR).t()).convertTo(r1d.row(i), rvecs.depth());
            }
        }
        if( !tvecs.empty() )
        {
            Vec3d srcT(param[idx + 3], param[idx + 4], param[idx + 5]);
            Mat(Mat(srcT).t()).convertTo(t1d.row(i), tvecs.depth());
        }
    }

    return std::sqrt(reprojErr/(pointsTotal*2));
}

static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    int iFixedPoint,
                                    OutputArray objPt, OutputArray imgPt1, OutputArray imgPt2,
                                    OutputArray npoints )
{
    int nimages = (int)objectPoints.total();
    int total = 0;
    CV_Assert(nimages > 0);
    CV_CheckEQ(nimages, (int)imagePoints1.total(), "");
    if (!imagePoints2.empty())
    {
        CV_CheckEQ(nimages, (int)imagePoints2.total(), "");
        CV_Assert(imgPt2.needed());
    }

    for (int i = 0; i < nimages; i++)
    {
        Mat objectPoint = objectPoints.getMat(i);
        if (objectPoint.empty())
            CV_Error(cv::Error::StsBadSize, "objectPoints should not contain empty vector of vectors of points");
        int numberOfObjectPoints = objectPoint.checkVector(3, CV_32F);
        if (numberOfObjectPoints <= 0)
            CV_Error(cv::Error::StsUnsupportedFormat, "objectPoints should contain vector of vectors of points of type Point3f");

        Mat imagePoint1 = imagePoints1.getMat(i);
        if (imagePoint1.empty())
            CV_Error(cv::Error::StsBadSize, "imagePoints1 should not contain empty vector of vectors of points");
        int numberOfImagePoints = imagePoint1.checkVector(2, CV_32F);
        if (numberOfImagePoints <= 0)
            CV_Error(cv::Error::StsUnsupportedFormat, "imagePoints1 should contain vector of vectors of points of type Point2f");
        CV_CheckEQ(numberOfObjectPoints, numberOfImagePoints, "Number of object and image points must be equal");

        total += numberOfObjectPoints;
    }

    npoints.create(1, (int)nimages, CV_32S);
    objPt.create(1, (int)total, CV_32FC3);
    imgPt1.create(1, (int)total, CV_32FC2);
    Point2f* imgPtData2 = 0;

    Mat imgPt1Mat = imgPt1.getMat();
    if (!imagePoints2.empty())
    {
        imgPt2.create(1, (int)total, CV_32FC2);
        imgPtData2 = imgPt2.getMat().ptr<Point2f>();
    }

    Mat nPointsMat = npoints.getMat();
    Mat objPtMat = objPt.getMat();
    Point3f* objPtData = objPtMat.ptr<Point3f>();
    Point2f* imgPtData1 = imgPt1.getMat().ptr<Point2f>();

    for (int i = 0, j = 0; i < nimages; i++)
    {
        Mat objpt = objectPoints.getMat(i);
        Mat imgpt1 = imagePoints1.getMat(i);
        int numberOfObjectPoints = objpt.checkVector(3, CV_32F);
        nPointsMat.at<int>(i) = numberOfObjectPoints;
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

    int ni = nPointsMat.at<int>(0);
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < ni - 1;
    // check object points. If not qualified, report errors.
    if( releaseObject )
    {
        for (int i = 1; i < nimages; i++)
        {
            if( nPointsMat.at<int>(i) != ni )
            {
                CV_Error( cv::Error::StsBadArg, "All objectPoints[i].size() should be equal when "
                                        "object-releasing method is requested." );
            }
            Mat ocmp = objPtMat.colRange(ni * i, ni * i + ni) != objPtMat.colRange(0, ni);
            ocmp = ocmp.reshape(1);
            if( countNonZero(ocmp) )
            {
                CV_Error( cv::Error::StsBadArg, "All objectPoints[i] should be identical when object-releasing"
                                        " method is requested." );
            }
        }
    }
}

static void collectCalibrationData( InputArrayOfArrays objectPoints,
                                    InputArrayOfArrays imagePoints1,
                                    InputArrayOfArrays imagePoints2,
                                    OutputArray objPt, OutputArray imgPtMat1, OutputArray imgPtMat2,
                                    OutputArray npoints )
{
    collectCalibrationData( objectPoints, imagePoints1, imagePoints2, -1, objPt, imgPtMat1,
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
                            objPt, imgPt, noArray(), npoints );
    initIntrinsicParams2D( objPt, imgPt, npoints, imageSize, cameraMatrix, aspectRatio );
    return cameraMatrix;
}

static Mat prepareDistCoeffs(Mat& distCoeffs0, int rtype, int outputSize)
{
    Size sz = distCoeffs0.size();
    int n = sz.area();
    if (n > 0)
        CV_Assert(sz.width == 1 || sz.height == 1);
    CV_Assert((int)distCoeffs0.total() <= outputSize);
    Mat distCoeffs = Mat::zeros(sz.width == 1 ? Size(1, outputSize) : Size(outputSize, 1), rtype);
    if( n ==  4 || n ==  5 || n ==  8 || n == 12 || n == 14 )
    {
        distCoeffs0.convertTo(distCoeffs(Rect(Point(), sz)), rtype);
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
            prepareDistCoeffs(distCoeffs, rtype, 14);
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
                            objPt, imgPt, noArray(), npoints );
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
        int sz = nimages*6 + CALIB_NINTRINSIC + (releaseObject ? np * 3 : 0);
        stdDeviationsM.create(sz, 1, CV_64F);
    }

    if( errors_needed )
    {
        _perViewErrors.create(nimages, 1, CV_64F);
        errorsM = _perViewErrors.getMat();
    }

    CvMat c_objPt = cvMat(objPt), c_imgPt = cvMat(imgPt), c_npoints = cvMat(npoints);
    CvMat c_cameraMatrix = cvMat(cameraMatrix), c_distCoeffs = cvMat(distCoeffs);

    double reprojErr = calibrateCameraInternal(
            &c_objPt, &c_imgPt, &c_npoints, cvSize(imageSize), iFixedPoint,
            &c_cameraMatrix, &c_distCoeffs,
            rvecM, tvecM,
            newObjPt,
            stdDeviationsM,
            errorsM, flags, cvTermCriteria(criteria));

    if( stddev_needed )
    {
        stdDeviationsM.rowRange(0, CALIB_NINTRINSIC).copyTo(stdDeviationsIntrinsics);
    }

    if ( stddev_ext_needed )
    {
        int s = CALIB_NINTRINSIC;
        stdDeviationsM.rowRange(s, s + nimages*6).copyTo(stdDeviationsExtrinsics);
    }

    if( stddev_obj_needed )
    {
        int s = CALIB_NINTRINSIC + nimages*6;
        stdDeviationsM.rowRange(s, s + np*3).copyTo(stdDeviationsObjPoints);
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
        CV_Error(cv::Error::StsUnmatchedSizes, "Size of cameraMatrix must be 3x3!");

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
                        OutputArray _perViewErrors, int flags,
                        TermCriteria criteria)
{
    return stereoCalibrate(_objectPoints, _imagePoints1, _imagePoints2, _cameraMatrix1, _distCoeffs1,
                           _cameraMatrix2, _distCoeffs2, imageSize, _Rmat, _Tmat, _Emat, _Fmat,
                           noArray(), noArray(), _perViewErrors, flags, criteria);
}

double stereoCalibrate( InputArrayOfArrays _objectPoints,
                        InputArrayOfArrays _imagePoints1,
                        InputArrayOfArrays _imagePoints2,
                        InputOutputArray _cameraMatrix1, InputOutputArray _distCoeffs1,
                        InputOutputArray _cameraMatrix2, InputOutputArray _distCoeffs2,
                        Size imageSize, InputOutputArray _Rmat, InputOutputArray _Tmat,
                        OutputArray _Emat, OutputArray _Fmat,
                        OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs,
                        OutputArray _perViewErrors, int flags,
                        TermCriteria criteria)
{
    int rtype = CV_64F;
    Mat cameraMatrix1 = _cameraMatrix1.getMat();
    Mat cameraMatrix2 = _cameraMatrix2.getMat();
    Mat distCoeffs1 = _distCoeffs1.getMat();
    Mat distCoeffs2 = _distCoeffs2.getMat();
    cameraMatrix1 = prepareCameraMatrix(cameraMatrix1, rtype, flags);
    cameraMatrix2 = prepareCameraMatrix(cameraMatrix2, rtype, flags);
    distCoeffs1 = prepareDistCoeffs(distCoeffs1, rtype, 14);
    distCoeffs2 = prepareDistCoeffs(distCoeffs2, rtype, 14);

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

    int nimages = int(_objectPoints.total());
    CV_Assert( nimages > 0 );

    Mat objPt, imgPt, imgPt2, npoints, rvecLM, tvecLM;

    collectCalibrationData( _objectPoints, _imagePoints1, _imagePoints2,
                            objPt, imgPt, imgPt2, npoints );
    CvMat c_objPt = cvMat(objPt), c_imgPt = cvMat(imgPt), c_imgPt2 = cvMat(imgPt2), c_npoints = cvMat(npoints);
    CvMat c_cameraMatrix1 = cvMat(cameraMatrix1), c_distCoeffs1 = cvMat(distCoeffs1);
    CvMat c_cameraMatrix2 = cvMat(cameraMatrix2), c_distCoeffs2 = cvMat(distCoeffs2);
    Mat matR = _Rmat.getMat(), matT = _Tmat.getMat();
    CvMat c_matR = cvMat(matR), c_matT = cvMat(matT);

    bool E_needed = _Emat.needed(), F_needed = _Fmat.needed();
    bool rvecs_needed = _rvecs.needed(), tvecs_needed = _tvecs.needed();
    bool errors_needed = _perViewErrors.needed();

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

    bool rvecs_mat_vec = _rvecs.isMatVector();
    bool tvecs_mat_vec = _tvecs.isMatVector();

    if( rvecs_needed )
    {
        _rvecs.create(nimages, 1, CV_64FC3);

        if( rvecs_mat_vec )
            rvecLM.create(nimages, 3, CV_64F);
        else
            rvecLM = _rvecs.getMat();
    }
    if( tvecs_needed )
    {
        _tvecs.create(nimages, 1, CV_64FC3);

        if( tvecs_mat_vec )
            tvecLM.create(nimages, 3, CV_64F);
        else
            tvecLM = _tvecs.getMat();
    }

    if( errors_needed )
    {
        _perViewErrors.create(nimages, 2, CV_64F);
        matErr = _perViewErrors.getMat();
    }

    double err = stereoCalibrateImpl(&c_objPt, &c_imgPt, &c_imgPt2, &c_npoints, &c_cameraMatrix1,
                                     &c_distCoeffs1, &c_cameraMatrix2, &c_distCoeffs2, cvSize(imageSize),
                                     &c_matR, &c_matT, matE, matF, rvecLM, tvecLM,
                                     matErr, flags, cvTermCriteria(criteria));
    cameraMatrix1.copyTo(_cameraMatrix1);
    cameraMatrix2.copyTo(_cameraMatrix2);
    distCoeffs1.copyTo(_distCoeffs1);
    distCoeffs2.copyTo(_distCoeffs2);

    for(int i = 0; i < nimages; i++ )
    {
        if( rvecs_needed && rvecs_mat_vec )
        {
            _rvecs.create(3, 1, CV_64F, i, true);
            Mat rv = _rvecs.getMat(i);
            Mat(rvecLM.row(i).t()).copyTo(rv);
        }
        if( tvecs_needed && tvecs_mat_vec )
        {
            _tvecs.create(3, 1, CV_64F, i, true);
            Mat tv = _tvecs.getMat(i);
            Mat(tvecLM.row(i).t()).copyTo(tv);
        }
    }

    return err;
}
}

/* End of file. */
