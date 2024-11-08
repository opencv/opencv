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
#include "calib3d_c_api.h"
#include <stdio.h>
#include <iterator>

/*
    This is straight-forward port v3 of Matlab calibration engine by Jean-Yves Bouguet
    that is (in a large extent) based on the paper:
    Z. Zhang. "A flexible new technique for camera calibration".
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 22(11):1330-1334, 2000.
    The 1st initial port was done by Valery Mosyagin.
*/

using namespace cv;

static const char* cvDistCoeffErr = "Distortion coefficients must be 1x4, 4x1, 1x5, 5x1, 1x8, 8x1, 1x12, 12x1, 1x14 or 14x1 floating-point vector";

CV_IMPL void cvInitIntrinsicParams2D( const CvMat* objectPoints,
                         const CvMat* imagePoints, const CvMat* npoints,
                         CvSize imageSize, CvMat* cameraMatrix,
                         double aspectRatio )
{
    Ptr<CvMat> matA, _b, _allH;

    int i, j, pos, nimages, ni = 0;
    double a[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 1 };
    double H[9] = {0}, f[2] = {0};
    CvMat _a = cvMat( 3, 3, CV_64F, a );
    CvMat matH = cvMat( 3, 3, CV_64F, H );
    CvMat _f = cvMat( 2, 1, CV_64F, f );

    CV_Assert(npoints);
    CV_Assert(CV_MAT_TYPE(npoints->type) == CV_32SC1);
    CV_Assert(CV_IS_MAT_CONT(npoints->type));
    nimages = npoints->rows + npoints->cols - 1;

    if( (CV_MAT_TYPE(objectPoints->type) != CV_32FC3 &&
        CV_MAT_TYPE(objectPoints->type) != CV_64FC3) ||
        (CV_MAT_TYPE(imagePoints->type) != CV_32FC2 &&
        CV_MAT_TYPE(imagePoints->type) != CV_64FC2) )
        CV_Error( cv::Error::StsUnsupportedFormat, "Both object points and image points must be 2D" );

    if( objectPoints->rows != 1 || imagePoints->rows != 1 )
        CV_Error( cv::Error::StsBadSize, "object points and image points must be a single-row matrices" );

    matA.reset(cvCreateMat( 2*nimages, 2, CV_64F ));
    _b.reset(cvCreateMat( 2*nimages, 1, CV_64F ));
    a[2] = (!imageSize.width) ? 0.5 : (imageSize.width - 1)*0.5;
    a[5] = (!imageSize.height) ? 0.5 : (imageSize.height - 1)*0.5;
    _allH.reset(cvCreateMat( nimages, 9, CV_64F ));

    // extract vanishing points in order to obtain initial value for the focal length
    for( i = 0, pos = 0; i < nimages; i++, pos += ni )
    {
        CV_DbgAssert(npoints->data.i);
        CV_DbgAssert(matA && matA->data.db);
        CV_DbgAssert(_b && _b->data.db);
        double* Ap = matA->data.db + i*4;
        double* bp = _b->data.db + i*2;
        ni = npoints->data.i[i];
        double h[3], v[3], d1[3], d2[3];
        double n[4] = {0,0,0,0};
        CvMat _m, matM;
        cvGetCols( objectPoints, &matM, pos, pos + ni );
        cvGetCols( imagePoints, &_m, pos, pos + ni );

        cvFindHomography( &matM, &_m, &matH );
        CV_DbgAssert(_allH && _allH->data.db);
        memcpy( _allH->data.db + i*9, H, sizeof(H) );

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

    cvSolve( matA, _b, &_f, CV_NORMAL + CV_SVD );
    a[0] = std::sqrt(fabs(1./f[0]));
    a[4] = std::sqrt(fabs(1./f[1]));
    if( aspectRatio != 0 )
    {
        double tf = (a[0] + a[4])/(aspectRatio + 1.);
        a[0] = aspectRatio*tf;
        a[4] = tf;
    }

    cvConvert( &_a, cameraMatrix );
}

static void subMatrix(const cv::Mat& src, cv::Mat& dst, const std::vector<uchar>& cols,
                      const std::vector<uchar>& rows) {
    int nonzeros_cols = cv::countNonZero(cols);
    cv::Mat tmp(src.rows, nonzeros_cols, CV_64FC1);

    for (int i = 0, j = 0; i < (int)cols.size(); i++)
    {
        if (cols[i])
        {
            src.col(i).copyTo(tmp.col(j++));
        }
    }

    int nonzeros_rows  = cv::countNonZero(rows);
    dst.create(nonzeros_rows, nonzeros_cols, CV_64FC1);
    for (int i = 0, j = 0; i < (int)rows.size(); i++)
    {
        if (rows[i])
        {
            tmp.row(i).copyTo(dst.row(j++));
        }
    }
}

static double cvCalibrateCamera2Internal( const CvMat* objectPoints,
                    const CvMat* imagePoints, const CvMat* npoints,
                    CvSize imageSize, int iFixedPoint, CvMat* cameraMatrix, CvMat* distCoeffs,
                    CvMat* rvecs, CvMat* tvecs, CvMat* newObjPoints, CvMat* stdDevs,
                    CvMat* perViewErrors, int flags, CvTermCriteria termCrit )
{
    const int NINTRINSIC = CV_CALIB_NINTRINSIC;
    double reprojErr = 0;

    Matx33d A;
    double k[14] = {0};
    CvMat matA = cvMat(3, 3, CV_64F, A.val), _k;
    int i, nimages, maxPoints = 0, ni = 0, pos, total = 0, nparams, npstep, cn;
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

    if( rvecs )
    {
        cn = CV_MAT_CN(rvecs->type);
        if( !CV_IS_MAT(rvecs) ||
            (CV_MAT_DEPTH(rvecs->type) != CV_32F && CV_MAT_DEPTH(rvecs->type) != CV_64F) ||
            ((rvecs->rows != nimages || (rvecs->cols*cn != 3 && rvecs->cols*cn != 9)) &&
            (rvecs->rows != 1 || rvecs->cols != nimages || cn != 3)) )
            CV_Error( cv::Error::StsBadArg, "the output array of rotation vectors must be 3-channel "
                "1xn or nx1 array or 1-channel nx3 or nx9 array, where n is the number of views" );
    }

    if( tvecs )
    {
        cn = CV_MAT_CN(tvecs->type);
        if( !CV_IS_MAT(tvecs) ||
            (CV_MAT_DEPTH(tvecs->type) != CV_32F && CV_MAT_DEPTH(tvecs->type) != CV_64F) ||
            ((tvecs->rows != nimages || tvecs->cols*cn != 3) &&
            (tvecs->rows != 1 || tvecs->cols != nimages || cn != 3)) )
            CV_Error( cv::Error::StsBadArg, "the output array of translation vectors must be 3-channel "
                "1xn or nx1 array or 1-channel nx3 array, where n is the number of views" );
    }

    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints->data.i[0] - 1;

    if( stdDevs && !releaseObject )
    {
        cn = CV_MAT_CN(stdDevs->type);
        if( !CV_IS_MAT(stdDevs) ||
            (CV_MAT_DEPTH(stdDevs->type) != CV_32F && CV_MAT_DEPTH(stdDevs->type) != CV_64F) ||
            ((stdDevs->rows != (nimages*6 + NINTRINSIC) || stdDevs->cols*cn != 1) &&
            (stdDevs->rows != 1 || stdDevs->cols != (nimages*6 + NINTRINSIC) || cn != 1)) )
#define STR__(x) #x
#define STR_(x) STR__(x)
            CV_Error( cv::Error::StsBadArg, "the output array of standard deviations vectors must be 1-channel "
                "1x(n*6 + NINTRINSIC) or (n*6 + NINTRINSIC)x1 array, where n is the number of views,"
                " NINTRINSIC = " STR_(CV_CALIB_NINTRINSIC));
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

    for( i = 0; i < nimages; i++ )
    {
        ni = npoints->data.i[i*npstep];
        if( ni < 4 )
        {
            CV_Error_( cv::Error::StsOutOfRange, ("The number of points in the view #%d is < 4", i));
        }
        maxPoints = MAX( maxPoints, ni );
        total += ni;
    }

    if( newObjPoints )
    {
        cn = CV_MAT_CN(newObjPoints->type);
        if( !CV_IS_MAT(newObjPoints) ||
            (CV_MAT_DEPTH(newObjPoints->type) != CV_32F && CV_MAT_DEPTH(newObjPoints->type) != CV_64F) ||
            ((newObjPoints->rows != maxPoints || newObjPoints->cols*cn != 3) &&
            (newObjPoints->rows != 1 || newObjPoints->cols != maxPoints || cn != 3)) )
            CV_Error( cv::Error::StsBadArg, "the output array of refined object points must be 3-channel "
                "1xn or nx1 array or 1-channel nx3 array, where n is the number of object points per view" );
    }

    if( stdDevs && releaseObject )
    {
        cn = CV_MAT_CN(stdDevs->type);
        if( !CV_IS_MAT(stdDevs) ||
            (CV_MAT_DEPTH(stdDevs->type) != CV_32F && CV_MAT_DEPTH(stdDevs->type) != CV_64F) ||
            ((stdDevs->rows != (nimages*6 + NINTRINSIC + maxPoints*3) || stdDevs->cols*cn != 1) &&
            (stdDevs->rows != 1 || stdDevs->cols != (nimages*6 + NINTRINSIC + maxPoints*3) || cn != 1)) )
            CV_Error( cv::Error::StsBadArg, "the output array of standard deviations vectors must be 1-channel "
                "1x(n*6 + NINTRINSIC + m*3) or (n*6 + NINTRINSIC + m*3)x1 array, where n is the number of views,"
                " NINTRINSIC = " STR_(CV_CALIB_NINTRINSIC) ", m is the number of object points per view");
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
        for( i = 0; i < total; i++ )
            matM.at<Point3d>(i).z = 0.;

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            aspectRatio = cvmGet(cameraMatrix,0,0);
            aspectRatio /= cvmGet(cameraMatrix,1,1);
            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        CvMat _matM = cvMat(matM), m = cvMat(_m);
        cvInitIntrinsicParams2D( &_matM, &m, npoints, imageSize, &matA, aspectRatio );
    }

    CvLevMarq solver( nparams, 0, termCrit );

    Mat _Ji( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
    Mat _Je( maxPoints*2, 6, CV_64FC1 );
    Mat _err( maxPoints*2, 1, CV_64FC1 );

    const bool allocJo = (solver.state == CvLevMarq::CALC_J) || stdDevs || releaseObject;
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
    {
      mask[6]  = mask[7]  = 0;
    }
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
    {
        mask[16] = 0;
        mask[17] = 0;
    }

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
    }

    Mat mask = cvarrToMat(solver.mask);
    int nparams_nz = countNonZero(mask);
    if (nparams_nz >= 2 * total)
        CV_Error_(cv::Error::StsBadArg,
                  ("There should be less vars to optimize (having %d) than the number of residuals (%d = 2 per point)", nparams_nz, 2 * total));

    // 2. initialize extrinsic parameters
    for( i = 0, pos = 0; i < nimages; i++, pos += ni )
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

    // 3. run the optimization
    for(;;)
    {
        const CvMat* _param = 0;
        CvMat *_JtJ = 0, *_JtErr = 0;
        double* _errNorm = 0;
        bool proceed = solver.updateAlt( _param, _JtJ, _JtErr, _errNorm );
        double *param = solver.param->data.db, *pparam = solver.prevParam->data.db;
        bool calcJ = solver.state == CvLevMarq::CALC_J || (!proceed && stdDevs);

        if( flags & CALIB_FIX_ASPECT_RATIO )
        {
            param[0] = param[1]*aspectRatio;
            pparam[0] = pparam[1]*aspectRatio;
        }

        A(0, 0) = param[0]; A(1, 1) = param[1]; A(0, 2) = param[2]; A(1, 2) = param[3];
        std::copy(param + 4, param + 4 + 14, k);

        if ( !proceed && !stdDevs && !perViewErrors )
            break;
        else if ( !proceed && stdDevs )
            cvZero(_JtJ);

        reprojErr = 0;

        for( i = 0, pos = 0; i < nimages; i++, pos += ni )
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
            if (perViewErrors || stdDevs)
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

            if( perViewErrors )
                perViewErrors->data.db[i] = std::sqrt(viewErr / ni);

            reprojErr += viewErr;
        }
        if( _errNorm )
            *_errNorm = reprojErr;

        if( !proceed )
        {
            if( stdDevs )
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
                Mat stdDevsM = cvarrToMat(stdDevs);
                int j = 0;
                for ( int s = 0; s < nparams; s++ )
                {
                    stdDevsM.at<double>(s) = mask.data[s] ? std::sqrt(JtJinv.at<double>(j,j) * sigma2) : 0.0;
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
    if( newObjPoints && releaseObject )
    {
        CvMat _Mi;
        cvGetRows( solver.param, &_Mi, NINTRINSIC + nimages * 6,
                   NINTRINSIC + nimages * 6 + maxPoints * 3 );
        cvReshape( &_Mi, &_Mi, 3, 1 );
        cvConvert( &_Mi, newObjPoints );
    }

    for( i = 0, pos = 0; i < nimages; i++ )
    {
        CvMat src, dst;

        if( rvecs )
        {
            src = cvMat( 3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 );
            if( rvecs->rows == nimages && rvecs->cols*CV_MAT_CN(rvecs->type) == 9 )
            {
                dst = cvMat( 3, 3, CV_MAT_DEPTH(rvecs->type),
                    rvecs->data.ptr + rvecs->step*i );
                Rodrigues( cvarrToMat(&src), cvarrToMat(&matA) );
                cvConvert( &matA, &dst );
            }
            else
            {
                dst = cvMat( 3, 1, CV_MAT_DEPTH(rvecs->type), rvecs->rows == 1 ?
                    rvecs->data.ptr + i*CV_ELEM_SIZE(rvecs->type) :
                    rvecs->data.ptr + rvecs->step*i );
                cvConvert( &src, &dst );
            }
        }
        if( tvecs )
        {
            src = cvMat( 3, 1, CV_64F, solver.param->data.db + NINTRINSIC + i*6 + 3 );
            dst = cvMat( 3, 1, CV_MAT_DEPTH(tvecs->type), tvecs->rows == 1 ?
                    tvecs->data.ptr + i*CV_ELEM_SIZE(tvecs->type) :
                    tvecs->data.ptr + tvecs->step*i );
            cvConvert( &src, &dst );
         }
    }

    return std::sqrt(reprojErr/total);
}


/* finds intrinsic and extrinsic camera parameters
   from a few views of known calibration pattern */
CV_IMPL double cvCalibrateCamera2( const CvMat* objectPoints,
                    const CvMat* imagePoints, const CvMat* npoints,
                    CvSize imageSize, CvMat* cameraMatrix, CvMat* distCoeffs,
                    CvMat* rvecs, CvMat* tvecs, int flags, CvTermCriteria termCrit )
{
    return cvCalibrateCamera2Internal(objectPoints, imagePoints, npoints, imageSize, -1, cameraMatrix,
                                      distCoeffs, rvecs, tvecs, NULL, NULL, NULL, flags, termCrit);
}

CV_IMPL double cvCalibrateCamera4( const CvMat* objectPoints,
                    const CvMat* imagePoints, const CvMat* npoints,
                    CvSize imageSize, int iFixedPoint, CvMat* cameraMatrix, CvMat* distCoeffs,
                    CvMat* rvecs, CvMat* tvecs, CvMat* newObjPoints, int flags, CvTermCriteria termCrit )
{
    if( !CV_IS_MAT(npoints) )
        CV_Error( cv::Error::StsBadArg, "npoints is not a valid matrix" );
    if( CV_MAT_TYPE(npoints->type) != CV_32SC1 ||
        (npoints->rows != 1 && npoints->cols != 1) )
        CV_Error( cv::Error::StsUnsupportedFormat,
            "the array of point counters must be 1-dimensional integer vector" );

    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints->data.i[0] - 1;
    int nimages = npoints->rows * npoints->cols;
    int npstep = npoints->rows == 1 ? 1 : npoints->step / CV_ELEM_SIZE(npoints->type);
    int i, ni;
    // check object points. If not qualified, report errors.
    if( releaseObject )
    {
        if( !CV_IS_MAT(objectPoints) )
            CV_Error( cv::Error::StsBadArg, "objectPoints is not a valid matrix" );
        Mat matM;
        if(CV_MAT_CN(objectPoints->type) == 3) {
            matM = cvarrToMat(objectPoints);
        } else {
            convertPointsHomogeneous(cvarrToMat(objectPoints), matM);
        }

        matM = matM.reshape(3, 1);
        ni = npoints->data.i[0];
        for( i = 1; i < nimages; i++ )
        {
            if( npoints->data.i[i * npstep] != ni )
            {
                CV_Error( cv::Error::StsBadArg, "All objectPoints[i].size() should be equal when "
                                        "object-releasing method is requested." );
            }
            Mat ocmp = matM.colRange(ni * i, ni * i + ni) != matM.colRange(0, ni);
            ocmp = ocmp.reshape(1);
            if( countNonZero(ocmp) )
            {
                CV_Error( cv::Error::StsBadArg, "All objectPoints[i] should be identical when object-releasing"
                                        " method is requested." );
            }
        }
    }

    return cvCalibrateCamera2Internal(objectPoints, imagePoints, npoints, imageSize, iFixedPoint,
                                      cameraMatrix, distCoeffs, rvecs, tvecs, newObjPoints, NULL,
                                      NULL, flags, termCrit);
}

void cvCalibrationMatrixValues( const CvMat *calibMatr, CvSize imgSize,
    double apertureWidth, double apertureHeight, double *fovx, double *fovy,
    double *focalLength, CvPoint2D64f *principalPoint, double *pasp )
{
    /* Validate parameters. */
    if(calibMatr == 0)
        CV_Error(cv::Error::StsNullPtr, "Some of parameters is a NULL pointer!");

    if(!CV_IS_MAT(calibMatr))
        CV_Error(cv::Error::StsUnsupportedFormat, "Input parameters must be matrices!");

    double dummy = .0;
    Point2d pp;
    cv::calibrationMatrixValues(cvarrToMat(calibMatr), imgSize, apertureWidth, apertureHeight,
            fovx ? *fovx : dummy,
            fovy ? *fovy : dummy,
            focalLength ? *focalLength : dummy,
            pp,
            pasp ? *pasp : dummy);

    if(principalPoint)
        *principalPoint = cvPoint2D64f(pp.x, pp.y);
}


//////////////////////////////// Stereo Calibration ///////////////////////////////////

static int dbCmp( const void* _a, const void* _b )
{
    double a = *(const double*)_a;
    double b = *(const double*)_b;

    return (a > b) - (a < b);
}

static double cvStereoCalibrateImpl( const CvMat* _objectPoints, const CvMat* _imagePoints1,
                        const CvMat* _imagePoints2, const CvMat* _npoints,
                        CvMat* _cameraMatrix1, CvMat* _distCoeffs1,
                        CvMat* _cameraMatrix2, CvMat* _distCoeffs2,
                        CvSize imageSize, CvMat* matR, CvMat* matT,
                        CvMat* matE, CvMat* matF,
                        CvMat* rvecs, CvMat* tvecs, CvMat* perViewErr, int flags,
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
    double aspectRatio[2] = {0};

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

    if( rvecs )
    {
        int cn = CV_MAT_CN(rvecs->type);
        if( !CV_IS_MAT(rvecs) ||
            (CV_MAT_DEPTH(rvecs->type) != CV_32F && CV_MAT_DEPTH(rvecs->type) != CV_64F) ||
            ((rvecs->rows != nimages || (rvecs->cols*cn != 3 && rvecs->cols*cn != 9)) &&
            (rvecs->rows != 1 || rvecs->cols != nimages || cn != 3)) )
            CV_Error( cv::Error::StsBadArg, "the output array of rotation vectors must be 3-channel "
                "1xn or nx1 array or 1-channel nx3 or nx9 array, where n is the number of views" );
    }

    if( tvecs )
    {
        int cn = CV_MAT_CN(tvecs->type);
        if( !CV_IS_MAT(tvecs) ||
            (CV_MAT_DEPTH(tvecs->type) != CV_32F && CV_MAT_DEPTH(tvecs->type) != CV_64F) ||
            ((tvecs->rows != nimages || tvecs->cols*cn != 3) &&
            (tvecs->rows != 1 || tvecs->cols != nimages || cn != 3)) )
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
            cvCalibrateCamera2( objectPoints, imagePoints[k],
                npoints, imageSize, &K[k], &Dist[k], NULL, NULL, flags );
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

                if(perViewErr)
                    perViewErr->data.db[i*2 + k] = std::sqrt(viewErr/ni);

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

    if( matE || matF )
    {
        double* t = T_LR.data.db;
        double tx[] =
        {
            0, -t[2], t[1],
            t[2], 0, -t[0],
            -t[1], t[0], 0
        };
        CvMat Tx = cvMat(3, 3, CV_64F, tx);
        double e[9], f[9];
        CvMat E = cvMat(3, 3, CV_64F, e);
        CvMat F = cvMat(3, 3, CV_64F, f);
        cvMatMul( &Tx, &R_LR, &E );
        if( matE )
            cvConvert( &E, matE );
        if( matF )
        {
            double ik[9];
            CvMat iK = cvMat(3, 3, CV_64F, ik);
            cvInvert(&K[1], &iK);
            cvGEMM( &iK, &E, 1, 0, 0, &E, CV_GEMM_A_T );
            cvInvert(&K[0], &iK);
            cvMatMul(&E, &iK, &F);
            cvConvertScale( &F, matF, fabs(f[8]) > 0 ? 1./f[8] : 1 );
        }
    }

    CvMat tmp = cvMat(3, 3, CV_64F);
    for( i = 0; i < nimages; i++ )
    {
        CvMat src, dst;

        if( rvecs )
        {
            src = cvMat(3, 1, CV_64F, solver.param->data.db+(i+1)*6);
            if( rvecs->rows == nimages && rvecs->cols*CV_MAT_CN(rvecs->type) == 9 )
            {
                dst = cvMat(3, 3, CV_MAT_DEPTH(rvecs->type),
                    rvecs->data.ptr + rvecs->step*i);
                Rodrigues( cvarrToMat(&src), cvarrToMat(&tmp) );
                cvConvert( &tmp, &dst );
            }
            else
            {
                dst = cvMat(3, 1, CV_MAT_DEPTH(rvecs->type), rvecs->rows == 1 ?
                    rvecs->data.ptr + i*CV_ELEM_SIZE(rvecs->type) :
                    rvecs->data.ptr + rvecs->step*i);
                cvConvert( &src, &dst );
            }
        }
        if( tvecs )
        {
            src = cvMat(3, 1,CV_64F,solver.param->data.db+(i+1)*6+3);
            dst = cvMat(3, 1, CV_MAT_DEPTH(tvecs->type), tvecs->rows == 1 ?
                    tvecs->data.ptr + i*CV_ELEM_SIZE(tvecs->type) :
                    tvecs->data.ptr + tvecs->step*i);
            cvConvert( &src, &dst );
         }
    }

    return std::sqrt(reprojErr/(pointsTotal*2));
}
double cvStereoCalibrate( const CvMat* _objectPoints, const CvMat* _imagePoints1,
                        const CvMat* _imagePoints2, const CvMat* _npoints,
                        CvMat* _cameraMatrix1, CvMat* _distCoeffs1,
                        CvMat* _cameraMatrix2, CvMat* _distCoeffs2,
                        CvSize imageSize, CvMat* matR, CvMat* matT,
                        CvMat* matE, CvMat* matF,
                        int flags,
                        CvTermCriteria termCrit )
{
    return cvStereoCalibrateImpl(_objectPoints, _imagePoints1, _imagePoints2, _npoints, _cameraMatrix1,
                                 _distCoeffs1, _cameraMatrix2, _distCoeffs2, imageSize, matR, matT, matE,
                                 matF, NULL, NULL, NULL, flags, termCrit);
}

static void
icvGetRectangles( const CvMat* cameraMatrix, const CvMat* distCoeffs,
                 const CvMat* R, const CvMat* newCameraMatrix, CvSize imgSize,
                 cv::Rect_<double>& inner, cv::Rect_<double>& outer )
{
    const int N = 9;
    int x, y, k;
    cv::Ptr<CvMat> _pts(cvCreateMat(1, N*N, CV_64FC2));
    CvPoint2D64f* pts = (CvPoint2D64f*)(_pts->data.ptr);

    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
            pts[k++] = cvPoint2D64f((double)x*(imgSize.width-1)/(N-1),
                                    (double)y*(imgSize.height-1)/(N-1));

    cvUndistortPoints(_pts, _pts, cameraMatrix, distCoeffs, R, newCameraMatrix);

    double iX0=-FLT_MAX, iX1=FLT_MAX, iY0=-FLT_MAX, iY1=FLT_MAX;
    double oX0=FLT_MAX, oX1=-FLT_MAX, oY0=FLT_MAX, oY1=-FLT_MAX;
    // find the inscribed rectangle.
    // the code will likely not work with extreme rotation matrices (R) (>45%)
    for( y = k = 0; y < N; y++ )
        for( x = 0; x < N; x++ )
        {
            CvPoint2D64f p = pts[k++];
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
    inner = cv::Rect_<double>(iX0, iY0, iX1-iX0, iY1-iY0);
    outer = cv::Rect_<double>(oX0, oY0, oX1-oX0, oY1-oY0);
}


void cvStereoRectify( const CvMat* _cameraMatrix1, const CvMat* _cameraMatrix2,
                      const CvMat* _distCoeffs1, const CvMat* _distCoeffs2,
                      CvSize imageSize, const CvMat* matR, const CvMat* matT,
                      CvMat* _R1, CvMat* _R2, CvMat* _P1, CvMat* _P2,
                      CvMat* matQ, int flags, double alpha, CvSize newImgSize,
                      CvRect* roi1, CvRect* roi2 )
{
    double _om[3], _t[3] = {0}, _uu[3]={0,0,0}, _r_r[3][3], _pp[3][4];
    double _ww[3], _wr[3][3], _z[3] = {0,0,0}, _ri[3][3];
    cv::Rect_<double> inner1, inner2, outer1, outer2;

    CvMat om  = cvMat(3, 1, CV_64F, _om);
    CvMat t   = cvMat(3, 1, CV_64F, _t);
    CvMat uu  = cvMat(3, 1, CV_64F, _uu);
    CvMat r_r = cvMat(3, 3, CV_64F, _r_r);
    CvMat pp  = cvMat(3, 4, CV_64F, _pp);
    CvMat ww  = cvMat(3, 1, CV_64F, _ww); // temps
    CvMat wR  = cvMat(3, 3, CV_64F, _wr);
    CvMat Z   = cvMat(3, 1, CV_64F, _z);
    CvMat Ri  = cvMat(3, 3, CV_64F, _ri);
    double nx = imageSize.width, ny = imageSize.height;
    int i, k;

    if( matR->rows == 3 && matR->cols == 3 )
        Rodrigues(cvarrToMat(matR), cvarrToMat(&om));          // get vector rotation
    else
        cvConvert(matR, &om); // it's already a rotation vector
    cvConvertScale(&om, &om, -0.5); // get average rotation
    Rodrigues(cvarrToMat(&om), cvarrToMat(&r_r));        // rotate cameras to same orientation by averaging
    cvMatMul(&r_r, matT, &t);

    int idx = fabs(_t[0]) > fabs(_t[1]) ? 0 : 1;
    double c = _t[idx], nt = cvNorm(&t, 0, CV_L2);
    _uu[idx] = c > 0 ? 1 : -1;

    CV_Assert(nt > 0.0);

    // calculate global Z rotation
    cvCrossProduct(&t,&uu,&ww);
    double nw = cvNorm(&ww, 0, CV_L2);
    if (nw > 0.0)
        cvConvertScale(&ww, &ww, acos(fabs(c)/nt)/nw);
    Rodrigues(cvarrToMat(&ww), cvarrToMat(&wR));

    // apply to both views
    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, CV_GEMM_B_T);
    cvConvert( &Ri, _R1 );
    cvGEMM(&wR, &r_r, 1, 0, 0, &Ri, 0);
    cvConvert( &Ri, _R2 );
    cvMatMul(&Ri, matT, &t);

    // calculate projection/camera matrices
    // these contain the relevant rectified image internal params (fx, fy=fx, cx, cy)
    double fc_new = DBL_MAX;
    CvPoint2D64f cc_new[2] = {};

    newImgSize = newImgSize.width * newImgSize.height != 0 ? newImgSize : imageSize;
    const double ratio_x = (double)newImgSize.width / imageSize.width / 2;
    const double ratio_y = (double)newImgSize.height / imageSize.height / 2;
    const double ratio = idx == 1 ? ratio_x : ratio_y;
    fc_new = (cvmGet(_cameraMatrix1, idx ^ 1, idx ^ 1) + cvmGet(_cameraMatrix2, idx ^ 1, idx ^ 1)) * ratio;

    for( k = 0; k < 2; k++ )
    {
        const CvMat* A = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const CvMat* Dk = k == 0 ? _distCoeffs1 : _distCoeffs2;
        CvPoint2D32f _pts[4] = {};
        CvPoint3D32f _pts_3[4] = {};
        CvMat pts = cvMat(1, 4, CV_32FC2, _pts);
        CvMat pts_3 = cvMat(1, 4, CV_32FC3, _pts_3);

        for( i = 0; i < 4; i++ )
        {
            int j = (i<2) ? 0 : 1;
            _pts[i].x = (float)((i % 2)*(nx-1));
            _pts[i].y = (float)(j*(ny-1));
        }
        cvUndistortPoints( &pts, &pts, A, Dk, 0, 0 );
        cvConvertPointsHomogeneous( &pts, &pts_3 );

        //Change camera matrix to have cc=[0,0] and fc = fc_new
        double _a_tmp[3][3];
        CvMat A_tmp  = cvMat(3, 3, CV_64F, _a_tmp);
        _a_tmp[0][0]=fc_new;
        _a_tmp[1][1]=fc_new;
        _a_tmp[0][2]=0.0;
        _a_tmp[1][2]=0.0;
        projectPoints( cvarrToMat(&pts_3), cvarrToMat(k == 0 ? _R1 : _R2), cvarrToMat(&Z),
                       cvarrToMat(&A_tmp), noArray(), cvarrToMat(&pts) );
        CvScalar avg = cvAvg(&pts);
        cc_new[k].x = (nx-1)/2 - avg.val[0];
        cc_new[k].y = (ny-1)/2 - avg.val[1];
    }

    // vertical focal length must be the same for both images to keep the epipolar constraint
    // (for horizontal epipolar lines -- TBD: check for vertical epipolar lines)
    // use fy for fx also, for simplicity

    // For simplicity, set the principal points for both cameras to be the average
    // of the two principal points (either one of or both x- and y- coordinates)
    if( flags & CALIB_ZERO_DISPARITY )
    {
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    }
    else if( idx == 0 ) // horizontal stereo
        cc_new[0].y = cc_new[1].y = (cc_new[0].y + cc_new[1].y)*0.5;
    else // vertical stereo
        cc_new[0].x = cc_new[1].x = (cc_new[0].x + cc_new[1].x)*0.5;

    cvZero( &pp );
    _pp[0][0] = _pp[1][1] = fc_new;
    _pp[0][2] = cc_new[0].x;
    _pp[1][2] = cc_new[0].y;
    _pp[2][2] = 1;
    cvConvert(&pp, _P1);

    _pp[0][2] = cc_new[1].x;
    _pp[1][2] = cc_new[1].y;
    _pp[idx][3] = _t[idx]*fc_new; // baseline * focal length
    cvConvert(&pp, _P2);

    alpha = MIN(alpha, 1.);

    icvGetRectangles( _cameraMatrix1, _distCoeffs1, _R1, _P1, imageSize, inner1, outer1 );
    icvGetRectangles( _cameraMatrix2, _distCoeffs2, _R2, _P2, imageSize, inner2, outer2 );

    {
    newImgSize = newImgSize.width*newImgSize.height != 0 ? newImgSize : imageSize;
    double cx1_0 = cc_new[0].x;
    double cy1_0 = cc_new[0].y;
    double cx2_0 = cc_new[1].x;
    double cy2_0 = cc_new[1].y;
    double cx1 = newImgSize.width*cx1_0/imageSize.width;
    double cy1 = newImgSize.height*cy1_0/imageSize.height;
    double cx2 = newImgSize.width*cx2_0/imageSize.width;
    double cy2 = newImgSize.height*cy2_0/imageSize.height;
    double s = 1.;

    if( alpha >= 0 )
    {
        double s0 = std::max(std::max(std::max((double)cx1/(cx1_0 - inner1.x), (double)cy1/(cy1_0 - inner1.y)),
                            (double)(newImgSize.width - 1 - cx1)/(inner1.x + inner1.width - cx1_0)),
                        (double)(newImgSize.height - 1 - cy1)/(inner1.y + inner1.height - cy1_0));
        s0 = std::max(std::max(std::max(std::max((double)cx2/(cx2_0 - inner2.x), (double)cy2/(cy2_0 - inner2.y)),
                         (double)(newImgSize.width - 1 - cx2)/(inner2.x + inner2.width - cx2_0)),
                     (double)(newImgSize.height - 1 - cy2)/(inner2.y + inner2.height - cy2_0)),
                 s0);

        double s1 = std::min(std::min(std::min((double)cx1/(cx1_0 - outer1.x), (double)cy1/(cy1_0 - outer1.y)),
                            (double)(newImgSize.width - 1 - cx1)/(outer1.x + outer1.width - cx1_0)),
                        (double)(newImgSize.height - 1 - cy1)/(outer1.y + outer1.height - cy1_0));
        s1 = std::min(std::min(std::min(std::min((double)cx2/(cx2_0 - outer2.x), (double)cy2/(cy2_0 - outer2.y)),
                         (double)(newImgSize.width - 1 - cx2)/(outer2.x + outer2.width - cx2_0)),
                     (double)(newImgSize.height - 1 - cy2)/(outer2.y + outer2.height - cy2_0)),
                 s1);

        s = s0*(1 - alpha) + s1*alpha;
    }

    fc_new *= s;
    cc_new[0] = cvPoint2D64f(cx1, cy1);
    cc_new[1] = cvPoint2D64f(cx2, cy2);

    cvmSet(_P1, 0, 0, fc_new);
    cvmSet(_P1, 1, 1, fc_new);
    cvmSet(_P1, 0, 2, cx1);
    cvmSet(_P1, 1, 2, cy1);

    cvmSet(_P2, 0, 0, fc_new);
    cvmSet(_P2, 1, 1, fc_new);
    cvmSet(_P2, 0, 2, cx2);
    cvmSet(_P2, 1, 2, cy2);
    cvmSet(_P2, idx, 3, s*cvmGet(_P2, idx, 3));

    if(roi1)
    {
        *roi1 = cvRect(
            cv::Rect(cvCeil((inner1.x - cx1_0)*s + cx1),
                     cvCeil((inner1.y - cy1_0)*s + cy1),
                     cvFloor(inner1.width*s), cvFloor(inner1.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height)
        );
    }

    if(roi2)
    {
        *roi2 = cvRect(
            cv::Rect(cvCeil((inner2.x - cx2_0)*s + cx2),
                     cvCeil((inner2.y - cy2_0)*s + cy2),
                     cvFloor(inner2.width*s), cvFloor(inner2.height*s))
            & cv::Rect(0, 0, newImgSize.width, newImgSize.height)
        );
    }
    }

    if( matQ )
    {
        double q[] =
        {
            1, 0, 0, -cc_new[0].x,
            0, 1, 0, -cc_new[0].y,
            0, 0, 0, fc_new,
            0, 0, -1./_t[idx],
            (idx == 0 ? cc_new[0].x - cc_new[1].x : cc_new[0].y - cc_new[1].y)/_t[idx]
        };
        CvMat Q = cvMat(4, 4, CV_64F, q);
        cvConvert( &Q, matQ );
    }
}


CV_IMPL int cvStereoRectifyUncalibrated(
    const CvMat* _points1, const CvMat* _points2,
    const CvMat* F0, CvSize imgSize,
    CvMat* _H1, CvMat* _H2, double threshold )
{
    Ptr<CvMat> _m1, _m2, _lines1, _lines2;

    int i, j, npoints;
    double cx, cy;
    double u[9], v[9], w[9], f[9], h1[9], h2[9], h0[9], e2[3] = {0};
    CvMat E2 = cvMat( 3, 1, CV_64F, e2 );
    CvMat U = cvMat( 3, 3, CV_64F, u );
    CvMat V = cvMat( 3, 3, CV_64F, v );
    CvMat W = cvMat( 3, 3, CV_64F, w );
    CvMat F = cvMat( 3, 3, CV_64F, f );
    CvMat H1 = cvMat( 3, 3, CV_64F, h1 );
    CvMat H2 = cvMat( 3, 3, CV_64F, h2 );
    CvMat H0 = cvMat( 3, 3, CV_64F, h0 );

    CvPoint2D64f* m1;
    CvPoint2D64f* m2;
    CvPoint3D64f* lines1;
    CvPoint3D64f* lines2;

    CV_Assert( CV_IS_MAT(_points1) && CV_IS_MAT(_points2) &&
        CV_ARE_SIZES_EQ(_points1, _points2) );

    npoints = _points1->rows * _points1->cols * CV_MAT_CN(_points1->type) / 2;

    _m1.reset(cvCreateMat( _points1->rows, _points1->cols, CV_64FC(CV_MAT_CN(_points1->type)) ));
    _m2.reset(cvCreateMat( _points2->rows, _points2->cols, CV_64FC(CV_MAT_CN(_points2->type)) ));
    _lines1.reset(cvCreateMat( 1, npoints, CV_64FC3 ));
    _lines2.reset(cvCreateMat( 1, npoints, CV_64FC3 ));

    cvConvert( F0, &F );

    cvSVD( (CvMat*)&F, &W, &U, &V, CV_SVD_U_T + CV_SVD_V_T );
    W.data.db[8] = 0.;
    cvGEMM( &U, &W, 1, 0, 0, &W, CV_GEMM_A_T );
    cvMatMul( &W, &V, &F );

    cx = cvRound( (imgSize.width-1)*0.5 );
    cy = cvRound( (imgSize.height-1)*0.5 );

    cvZero( _H1 );
    cvZero( _H2 );

    cvConvert( _points1, _m1 );
    cvConvert( _points2, _m2 );
    cvReshape( _m1, _m1, 2, 1 );
    cvReshape( _m2, _m2, 2, 1 );

    m1 = (CvPoint2D64f*)_m1->data.ptr;
    m2 = (CvPoint2D64f*)_m2->data.ptr;
    lines1 = (CvPoint3D64f*)_lines1->data.ptr;
    lines2 = (CvPoint3D64f*)_lines2->data.ptr;

    if( threshold > 0 )
    {
        cvComputeCorrespondEpilines( _m1, 1, &F, _lines1 );
        cvComputeCorrespondEpilines( _m2, 2, &F, _lines2 );

        // measure distance from points to the corresponding epilines, mark outliers
        for( i = j = 0; i < npoints; i++ )
        {
            if( fabs(m1[i].x*lines2[i].x +
                     m1[i].y*lines2[i].y +
                     lines2[i].z) <= threshold &&
                fabs(m2[i].x*lines1[i].x +
                     m2[i].y*lines1[i].y +
                     lines1[i].z) <= threshold )
            {
                if( j < i )
                {
                    m1[j] = m1[i];
                    m2[j] = m2[i];
                }
                j++;
            }
        }

        npoints = j;
        if( npoints == 0 )
            return 0;
    }

    _m1->cols = _m2->cols = npoints;
    memcpy( E2.data.db, U.data.db + 6, sizeof(e2));
    cvScale( &E2, &E2, e2[2] > 0 ? 1 : -1 );

    double t[] =
    {
        1, 0, -cx,
        0, 1, -cy,
        0, 0, 1
    };
    CvMat T = cvMat(3, 3, CV_64F, t);
    cvMatMul( &T, &E2, &E2 );

    int mirror = e2[0] < 0;
    double d = MAX(std::sqrt(e2[0]*e2[0] + e2[1]*e2[1]),DBL_EPSILON);
    double alpha = e2[0]/d;
    double beta = e2[1]/d;
    double r[] =
    {
        alpha, beta, 0,
        -beta, alpha, 0,
        0, 0, 1
    };
    CvMat R = cvMat(3, 3, CV_64F, r);
    cvMatMul( &R, &T, &T );
    cvMatMul( &R, &E2, &E2 );
    double invf = fabs(e2[2]) < 1e-6*fabs(e2[0]) ? 0 : -e2[2]/e2[0];
    double k[] =
    {
        1, 0, 0,
        0, 1, 0,
        invf, 0, 1
    };
    CvMat K = cvMat(3, 3, CV_64F, k);
    cvMatMul( &K, &T, &H2 );
    cvMatMul( &K, &E2, &E2 );

    double it[] =
    {
        1, 0, cx,
        0, 1, cy,
        0, 0, 1
    };
    CvMat iT = cvMat( 3, 3, CV_64F, it );
    cvMatMul( &iT, &H2, &H2 );

    memcpy( E2.data.db, U.data.db + 6, sizeof(e2));
    cvScale( &E2, &E2, e2[2] > 0 ? 1 : -1 );

    double e2_x[] =
    {
        0, -e2[2], e2[1],
       e2[2], 0, -e2[0],
       -e2[1], e2[0], 0
    };
    double e2_111[] =
    {
        e2[0], e2[0], e2[0],
        e2[1], e2[1], e2[1],
        e2[2], e2[2], e2[2],
    };
    CvMat E2_x = cvMat(3, 3, CV_64F, e2_x);
    CvMat E2_111 = cvMat(3, 3, CV_64F, e2_111);
    cvMatMulAdd(&E2_x, &F, &E2_111, &H0 );
    cvMatMul(&H2, &H0, &H0);
    CvMat E1=cvMat(3, 1, CV_64F, V.data.db+6);
    cvMatMul(&H0, &E1, &E1);

    cvPerspectiveTransform( _m1, _m1, &H0 );
    cvPerspectiveTransform( _m2, _m2, &H2 );
    CvMat A = cvMat( 1, npoints, CV_64FC3, lines1 ), BxBy, B;
    double x[3] = {0};
    CvMat X = cvMat( 3, 1, CV_64F, x );
    cvConvertPointsHomogeneous( _m1, &A );
    cvReshape( &A, &A, 1, npoints );
    cvReshape( _m2, &BxBy, 1, npoints );
    cvGetCol( &BxBy, &B, 0 );
    cvSolve( &A, &B, &X, CV_SVD );

    double ha[] =
    {
        x[0], x[1], x[2],
        0, 1, 0,
        0, 0, 1
    };
    CvMat Ha = cvMat(3, 3, CV_64F, ha);
    cvMatMul( &Ha, &H0, &H1 );
    cvPerspectiveTransform( _m1, _m1, &Ha );

    if( mirror )
    {
        double mm[] = { -1, 0, cx*2, 0, -1, cy*2, 0, 0, 1 };
        CvMat MM = cvMat(3, 3, CV_64F, mm);
        cvMatMul( &MM, &H1, &H1 );
        cvMatMul( &MM, &H2, &H2 );
    }

    cvConvert( &H1, _H1 );
    cvConvert( &H2, _H2 );

    return 1;
}


void cv::reprojectImageTo3D( InputArray _disparity,
                             OutputArray __3dImage, InputArray _Qmat,
                             bool handleMissingValues, int dtype )
{
    CV_INSTRUMENT_REGION();

    Mat disparity = _disparity.getMat(), Q = _Qmat.getMat();
    int stype = disparity.type();

    CV_Assert( stype == CV_8UC1 || stype == CV_16SC1 ||
               stype == CV_32SC1 || stype == CV_32FC1 );
    CV_Assert( Q.size() == Size(4,4) );

    if( dtype >= 0 )
        dtype = CV_MAKETYPE(CV_MAT_DEPTH(dtype), 3);

    if( __3dImage.fixedType() )
    {
        int dtype_ = __3dImage.type();
        CV_Assert( dtype == -1 || dtype == dtype_ );
        dtype = dtype_;
    }

    if( dtype < 0 )
        dtype = CV_32FC3;
    else
        CV_Assert( dtype == CV_16SC3 || dtype == CV_32SC3 || dtype == CV_32FC3 );

    __3dImage.create(disparity.size(), dtype);
    Mat _3dImage = __3dImage.getMat();

    const float bigZ = 10000.f;
    Matx44d _Q;
    Q.convertTo(_Q, CV_64F);

    int x, cols = disparity.cols;
    CV_Assert( cols >= 0 );

    std::vector<float> _sbuf(cols);
    std::vector<Vec3f> _dbuf(cols);
    float* sbuf = &_sbuf[0];
    Vec3f* dbuf = &_dbuf[0];
    double minDisparity = FLT_MAX;

    // NOTE: here we quietly assume that at least one pixel in the disparity map is not defined.
    // and we set the corresponding Z's to some fixed big value.
    if( handleMissingValues )
        cv::minMaxIdx( disparity, &minDisparity, 0, 0, 0 );

    for( int y = 0; y < disparity.rows; y++ )
    {
        float* sptr = sbuf;
        Vec3f* dptr = dbuf;

        if( stype == CV_8UC1 )
        {
            const uchar* sptr0 = disparity.ptr<uchar>(y);
            for( x = 0; x < cols; x++ )
                sptr[x] = (float)sptr0[x];
        }
        else if( stype == CV_16SC1 )
        {
            const short* sptr0 = disparity.ptr<short>(y);
            for( x = 0; x < cols; x++ )
                sptr[x] = (float)sptr0[x];
        }
        else if( stype == CV_32SC1 )
        {
            const int* sptr0 = disparity.ptr<int>(y);
            for( x = 0; x < cols; x++ )
                sptr[x] = (float)sptr0[x];
        }
        else
            sptr = disparity.ptr<float>(y);

        if( dtype == CV_32FC3 )
            dptr = _3dImage.ptr<Vec3f>(y);

        for( x = 0; x < cols; x++)
        {
            double d = sptr[x];
            Vec4d homg_pt = _Q*Vec4d(x, y, d, 1.0);
            dptr[x] = Vec3d(homg_pt.val);
            dptr[x] /= homg_pt[3];

            if( fabs(d-minDisparity) <= FLT_EPSILON )
                dptr[x][2] = bigZ;
        }

        if( dtype == CV_16SC3 )
        {
            Vec3s* dptr0 = _3dImage.ptr<Vec3s>(y);
            for( x = 0; x < cols; x++ )
            {
                dptr0[x] = dptr[x];
            }
        }
        else if( dtype == CV_32SC3 )
        {
            Vec3i* dptr0 = _3dImage.ptr<Vec3i>(y);
            for( x = 0; x < cols; x++ )
            {
                dptr0[x] = dptr[x];
            }
        }
    }
}


void cvReprojectImageTo3D( const CvArr* disparityImage,
                           CvArr* _3dImage, const CvMat* matQ,
                           int handleMissingValues )
{
    cv::Mat disp = cv::cvarrToMat(disparityImage);
    cv::Mat _3dimg = cv::cvarrToMat(_3dImage);
    cv::Mat mq = cv::cvarrToMat(matQ);
    CV_Assert( disp.size() == _3dimg.size() );
    int dtype = _3dimg.type();
    CV_Assert( dtype == CV_16SC3 || dtype == CV_32SC3 || dtype == CV_32FC3 );

    cv::reprojectImageTo3D(disp, _3dimg, mq, handleMissingValues != 0, dtype );
}



namespace cv
{

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

} // namespace cv

cv::Mat cv::initCameraMatrix2D( InputArrayOfArrays objectPoints,
                                InputArrayOfArrays imagePoints,
                                Size imageSize, double aspectRatio )
{
    CV_INSTRUMENT_REGION();

    Mat objPt, imgPt, npoints, cameraMatrix(3, 3, CV_64F);
    collectCalibrationData( objectPoints, imagePoints, noArray(),
                            objPt, imgPt, 0, npoints );
    CvMat _objPt = cvMat(objPt), _imgPt = cvMat(imgPt), _npoints = cvMat(npoints), _cameraMatrix = cvMat(cameraMatrix);
    cvInitIntrinsicParams2D( &_objPt, &_imgPt, &_npoints,
                             cvSize(imageSize), &_cameraMatrix, aspectRatio );
    return cameraMatrix;
}



double cv::calibrateCamera( InputArrayOfArrays _objectPoints,
                            InputArrayOfArrays _imagePoints,
                            Size imageSize, InputOutputArray _cameraMatrix, InputOutputArray _distCoeffs,
                            OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, int flags, TermCriteria criteria )
{
    CV_INSTRUMENT_REGION();

    return calibrateCamera(_objectPoints, _imagePoints, imageSize, _cameraMatrix, _distCoeffs,
                                         _rvecs, _tvecs, noArray(), noArray(), noArray(), flags, criteria);
}

double cv::calibrateCamera(InputArrayOfArrays _objectPoints,
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

double cv::calibrateCameraRO(InputArrayOfArrays _objectPoints,
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

double cv::calibrateCameraRO(InputArrayOfArrays _objectPoints,
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
    distCoeffs = (flags & CALIB_THIN_PRISM_MODEL) && !(flags & CALIB_TILTED_MODEL)  ? prepareDistCoeffs(distCoeffs, rtype, 12) :
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
            stdDeviationsM.create(nimages*6 + CV_CALIB_NINTRINSIC + np * 3, 1, CV_64F);
        else
            stdDeviationsM.create(nimages*6 + CV_CALIB_NINTRINSIC, 1, CV_64F);
    }

    if( errors_needed )
    {
        _perViewErrors.create(nimages, 1, CV_64F);
        errorsM = _perViewErrors.getMat();
    }

    CvMat c_objPt = cvMat(objPt), c_imgPt = cvMat(imgPt), c_npoints = cvMat(npoints);
    CvMat c_cameraMatrix = cvMat(cameraMatrix), c_distCoeffs = cvMat(distCoeffs);
    CvMat c_rvecM = cvMat(rvecM), c_tvecM = cvMat(tvecM), c_stdDev = cvMat(stdDeviationsM), c_errors = cvMat(errorsM);
    CvMat c_newObjPt = cvMat( newObjPt );

    double reprojErr = cvCalibrateCamera2Internal(&c_objPt, &c_imgPt, &c_npoints, cvSize(imageSize),
                                          iFixedPoint,
                                          &c_cameraMatrix, &c_distCoeffs,
                                          rvecs_needed ? &c_rvecM : NULL,
                                          tvecs_needed ? &c_tvecM : NULL,
                                          newobj_needed ? &c_newObjPt : NULL,
                                          stddev_any_needed ? &c_stdDev : NULL,
                                          errors_needed ? &c_errors : NULL, flags, cvTermCriteria(criteria));

    if( newobj_needed )
        newObjPt.copyTo(newObjPoints);

    if( stddev_needed )
    {
        stdDeviationsIntrinsics.create(CV_CALIB_NINTRINSIC, 1, CV_64F);
        Mat stdDeviationsIntrinsicsMat = stdDeviationsIntrinsics.getMat();
        std::memcpy(stdDeviationsIntrinsicsMat.ptr(), stdDeviationsM.ptr(),
                    CV_CALIB_NINTRINSIC*sizeof(double));
    }

    if ( stddev_ext_needed )
    {
        stdDeviationsExtrinsics.create(nimages*6, 1, CV_64F);
        Mat stdDeviationsExtrinsicsMat = stdDeviationsExtrinsics.getMat();
        std::memcpy(stdDeviationsExtrinsicsMat.ptr(),
                    stdDeviationsM.ptr() + CV_CALIB_NINTRINSIC*sizeof(double),
                    nimages*6*sizeof(double));
    }

    if( stddev_obj_needed )
    {
        stdDeviationsObjPoints.create( np * 3, 1, CV_64F );
        Mat stdDeviationsObjPointsMat = stdDeviationsObjPoints.getMat();
        std::memcpy( stdDeviationsObjPointsMat.ptr(), stdDeviationsM.ptr()
                         + ( CV_CALIB_NINTRINSIC + nimages * 6 ) * sizeof( double ),
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


void cv::calibrationMatrixValues( InputArray _cameraMatrix, Size imageSize,
                                  double apertureWidth, double apertureHeight,
                                  double& fovx, double& fovy, double& focalLength,
                                  Point2d& principalPoint, double& aspectRatio )
{
    CV_INSTRUMENT_REGION();

    if(_cameraMatrix.size() != Size(3, 3))
        CV_Error(cv::Error::StsUnmatchedSizes, "Size of cameraMatrix must be 3x3!");

    Matx33d K = _cameraMatrix.getMat();

    CV_DbgAssert(imageSize.width != 0 && imageSize.height != 0 && K(0, 0) != 0.0 && K(1, 1) != 0.0);

    /* Calculate pixel aspect ratio. */
    aspectRatio = K(1, 1) / K(0, 0);

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
    fovx = atan2(K(0, 2), K(0, 0)) + atan2(imageSize.width  - K(0, 2), K(0, 0));
    fovy = atan2(K(1, 2), K(1, 1)) + atan2(imageSize.height - K(1, 2), K(1, 1));
    fovx *= 180.0 / CV_PI;
    fovy *= 180.0 / CV_PI;

    /* Calculate focal length. */
    focalLength = K(0, 0) / mx;

    /* Calculate principle point. */
    principalPoint = Point2d(K(0, 2) / mx, K(1, 2) / my);
}

double cv::stereoCalibrate( InputArrayOfArrays _objectPoints,
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

double cv::stereoCalibrate( InputArrayOfArrays _objectPoints,
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

double cv::stereoCalibrate( InputArrayOfArrays _objectPoints,
                          InputArrayOfArrays _imagePoints1,
                          InputArrayOfArrays _imagePoints2,
                          InputOutputArray _cameraMatrix1, InputOutputArray _distCoeffs1,
                          InputOutputArray _cameraMatrix2, InputOutputArray _distCoeffs2,
                          Size imageSize, InputOutputArray _Rmat, InputOutputArray _Tmat,
                          OutputArray _Emat, OutputArray _Fmat,
                          OutputArrayOfArrays _rvecs, OutputArrayOfArrays _tvecs, OutputArray _perViewErrors, int flags,
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
    (!(flags & CALIB_TILTED_MODEL)) )
    {
        distCoeffs1 = distCoeffs1.rows == 1 ? distCoeffs1.colRange(0, 5) : distCoeffs1.rowRange(0, 5);
        distCoeffs2 = distCoeffs2.rows == 1 ? distCoeffs2.colRange(0, 5) : distCoeffs2.rowRange(0, 5);
    }

    if( (flags & CALIB_USE_EXTRINSIC_GUESS) == 0 )
    {
        _Rmat.create(3, 3, rtype);
        _Tmat.create(3, 1, rtype);
    }

    int nimages = int(_objectPoints.total());
    CV_Assert( nimages > 0 );

    Mat objPt, imgPt, imgPt2, npoints, rvecLM, tvecLM;

    collectCalibrationData( _objectPoints, _imagePoints1, _imagePoints2,
                            objPt, imgPt, &imgPt2, npoints );
    CvMat c_objPt = cvMat(objPt), c_imgPt = cvMat(imgPt), c_imgPt2 = cvMat(imgPt2), c_npoints = cvMat(npoints);
    CvMat c_cameraMatrix1 = cvMat(cameraMatrix1), c_distCoeffs1 = cvMat(distCoeffs1);
    CvMat c_cameraMatrix2 = cvMat(cameraMatrix2), c_distCoeffs2 = cvMat(distCoeffs2);
    Mat matR_ = _Rmat.getMat(), matT_ = _Tmat.getMat();
    CvMat c_matR = cvMat(matR_), c_matT = cvMat(matT_), c_matE, c_matF, c_matErr;

    bool E_needed = _Emat.needed(), F_needed = _Fmat.needed(), rvecs_needed = _rvecs.needed(), tvecs_needed = _tvecs.needed(), errors_needed = _perViewErrors.needed();

    Mat matE_, matF_, matErr_;
    if( E_needed )
    {
        _Emat.create(3, 3, rtype);
        matE_ = _Emat.getMat();
        c_matE = cvMat(matE_);
    }
    if( F_needed )
    {
        _Fmat.create(3, 3, rtype);
        matF_ = _Fmat.getMat();
        c_matF = cvMat(matF_);
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
    CvMat c_rvecLM = cvMat(rvecLM), c_tvecLM = cvMat(tvecLM);

    if( errors_needed )
    {
        _perViewErrors.create(nimages, 2, CV_64F);
        matErr_ = _perViewErrors.getMat();
        c_matErr = cvMat(matErr_);
    }

    double err = cvStereoCalibrateImpl(&c_objPt, &c_imgPt, &c_imgPt2, &c_npoints, &c_cameraMatrix1,
                                       &c_distCoeffs1, &c_cameraMatrix2, &c_distCoeffs2, cvSize(imageSize), &c_matR,
                                       &c_matT, E_needed ? &c_matE : NULL, F_needed ? &c_matF : NULL,
                                       rvecs_needed ? &c_rvecLM : NULL, tvecs_needed ? &c_tvecLM : NULL,
                                       errors_needed ? &c_matErr : NULL, flags, cvTermCriteria(criteria));

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
            memcpy(rv.ptr(), rvecLM.ptr(i), 3*sizeof(double));
        }
        if( tvecs_needed && tvecs_mat_vec )
        {
            _tvecs.create(3, 1, CV_64F, i, true);
            Mat tv = _tvecs.getMat(i);
            memcpy(tv.ptr(), tvecLM.ptr(i), 3*sizeof(double));
        }
    }

    return err;
}


void cv::stereoRectify( InputArray _cameraMatrix1, InputArray _distCoeffs1,
                        InputArray _cameraMatrix2, InputArray _distCoeffs2,
                        Size imageSize, InputArray _Rmat, InputArray _Tmat,
                        OutputArray _Rmat1, OutputArray _Rmat2,
                        OutputArray _Pmat1, OutputArray _Pmat2,
                        OutputArray _Qmat, int flags,
                        double alpha, Size newImageSize,
                        Rect* validPixROI1, Rect* validPixROI2 )
{
    Mat cameraMatrix1 = _cameraMatrix1.getMat(), cameraMatrix2 = _cameraMatrix2.getMat();
    Mat distCoeffs1 = _distCoeffs1.getMat(), distCoeffs2 = _distCoeffs2.getMat();
    Mat Rmat = _Rmat.getMat(), Tmat = _Tmat.getMat();
    CvMat c_cameraMatrix1 = cvMat(cameraMatrix1);
    CvMat c_cameraMatrix2 = cvMat(cameraMatrix2);
    CvMat c_distCoeffs1 = cvMat(distCoeffs1);
    CvMat c_distCoeffs2 = cvMat(distCoeffs2);
    CvMat c_R = cvMat(Rmat), c_T = cvMat(Tmat);

    int rtype = CV_64F;
    _Rmat1.create(3, 3, rtype);
    _Rmat2.create(3, 3, rtype);
    _Pmat1.create(3, 4, rtype);
    _Pmat2.create(3, 4, rtype);
    Mat R1 = _Rmat1.getMat(), R2 = _Rmat2.getMat(), P1 = _Pmat1.getMat(), P2 = _Pmat2.getMat(), Q;
    CvMat c_R1 = cvMat(R1), c_R2 = cvMat(R2), c_P1 = cvMat(P1), c_P2 = cvMat(P2);
    CvMat c_Q, *p_Q = 0;

    if( _Qmat.needed() )
    {
        _Qmat.create(4, 4, rtype);
        p_Q = &(c_Q = cvMat(Q = _Qmat.getMat()));
    }

    CvMat *p_distCoeffs1 = distCoeffs1.empty() ? NULL : &c_distCoeffs1;
    CvMat *p_distCoeffs2 = distCoeffs2.empty() ? NULL : &c_distCoeffs2;
    cvStereoRectify( &c_cameraMatrix1, &c_cameraMatrix2, p_distCoeffs1, p_distCoeffs2,
        cvSize(imageSize), &c_R, &c_T, &c_R1, &c_R2, &c_P1, &c_P2, p_Q, flags, alpha,
        cvSize(newImageSize), (CvRect*)validPixROI1, (CvRect*)validPixROI2);
}

bool cv::stereoRectifyUncalibrated( InputArray _points1, InputArray _points2,
                                    InputArray _Fmat, Size imgSize,
                                    OutputArray _Hmat1, OutputArray _Hmat2, double threshold )
{
    CV_INSTRUMENT_REGION();

    int rtype = CV_64F;
    _Hmat1.create(3, 3, rtype);
    _Hmat2.create(3, 3, rtype);
    Mat F = _Fmat.getMat();
    Mat points1 = _points1.getMat(), points2 = _points2.getMat();
    CvMat c_pt1 = cvMat(points1), c_pt2 = cvMat(points2);
    Mat H1 = _Hmat1.getMat(), H2 = _Hmat2.getMat();
    CvMat c_F, *p_F=0, c_H1 = cvMat(H1), c_H2 = cvMat(H2);
    if( F.size() == Size(3, 3) )
        p_F = &(c_F = cvMat(F));
    return cvStereoRectifyUncalibrated(&c_pt1, &c_pt2, p_F, cvSize(imgSize), &c_H1, &c_H2, threshold) > 0;
}

namespace cv
{

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

}

float cv::rectify3Collinear( InputArray _cameraMatrix1, InputArray _distCoeffs1,
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


/* End of file. */
