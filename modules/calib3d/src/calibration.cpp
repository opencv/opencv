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

static double calibrateCameraInternal( const Mat& objectPoints,
                                       const Mat& imagePoints, const Mat& npoints,
                                       Size imageSize, int iFixedPoint, Mat& cameraMatrix, Mat& distCoeffs,
                                       Mat rvecs, Mat tvecs, Mat newObjPoints, Mat stdDevs,
                                       Mat perViewErr, int flags, const TermCriteria& termCrit )
{
    int NINTRINSIC = CALIB_NINTRINSIC;
    double reprojErr = 0;

    double aspectRatio = 0.;
    int nimages = npoints.checkVector(1, CV_32S);
    CV_Assert(nimages >= 1);
    int ndistCoeffs = (int)distCoeffs.total();
    bool releaseObject = iFixedPoint > 0 && iFixedPoint < npoints.at<int>(0) - 1;

    // 0. check the parameters & allocate buffers
    if( imageSize.width <= 0 || imageSize.height <= 0 )
        CV_Error( cv::Error::StsOutOfRange, "image width and height must be positive" );

    if(flags & CALIB_TILTED_MODEL)
    {
        //when the tilted sensor model is used the distortion coefficients matrix must have 14 parameters
        if (ndistCoeffs != 14)
            CV_Error( cv::Error::StsBadArg, "The tilted sensor model must have 14 parameters in the distortion matrix" );
    }
    else
    {
        //when the thin prism model is used the distortion coefficients matrix must have 12 parameters
        if(flags & CALIB_THIN_PRISM_MODEL)
            if (ndistCoeffs != 12)
                CV_Error( cv::Error::StsBadArg, "Thin prism model must have 12 parameters in the distortion matrix" );
    }

    if( !rvecs.empty() )
    {
        int cn = rvecs.channels();
        CV_Assert(rvecs.depth() == CV_32F || rvecs.depth() == CV_64F);
        CV_Assert((rvecs.rows == nimages && (rvecs.cols*cn == 3 || rvecs.cols*cn == 3)) ||
                  (rvecs.rows == 1 && rvecs.cols == nimages && cn == 3));
    }

    if( !tvecs.empty() )
    {
        int cn = tvecs.channels();
        CV_Assert(tvecs.depth() == CV_32F || tvecs.depth() == CV_64F);
        CV_Assert((tvecs.rows == nimages && tvecs.cols*cn == 3) ||
                  (tvecs.rows == 1 && tvecs.cols == nimages && cn == 3));
    }

    CV_Assert(cameraMatrix.type() == CV_32F || cameraMatrix.type() == CV_64F);
    CV_Assert(cameraMatrix.rows == 3 && cameraMatrix.cols == 3);

    CV_Assert(distCoeffs.type() == CV_32F || distCoeffs.type() == CV_64F);
    CV_Assert(distCoeffs.rows == 1 || distCoeffs.cols == 1);
    CV_Assert(ndistCoeffs == 4 || ndistCoeffs == 5 || ndistCoeffs == 8 ||
              ndistCoeffs == 12 || ndistCoeffs == 14);

    int total = 0, maxPoints = 0;
    for(int i = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);
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

    if( !stdDevs.empty() )
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

    if(objectPoints.channels() == 3)
        objectPoints.convertTo(matM, CV_64F);
    else {
        convertPointsToHomogeneous(objectPoints, matM);
        matM.convertTo(matM, CV_64F);
    }

    if(imagePoints.channels() == 2)
        imagePoints.convertTo(_m, CV_64F);
    else {
        convertPointsFromHomogeneous(imagePoints, _m);
        _m.convertTo(_m, CV_64F);
    }

    int nparams = NINTRINSIC + nimages*6;
    if( releaseObject )
        nparams += maxPoints * 3;

    std::vector<double> k(14, 0.0);
    Mat _k( distCoeffs.rows, distCoeffs.cols, CV_64F, k.data());
    if( distCoeffs.total() < 8 )
    {
        if( distCoeffs.total() < 5 )
            flags |= CALIB_FIX_K3;
        flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
    }
    const double minValidAspectRatio = 0.01;
    const double maxValidAspectRatio = 100.0;

    Matx33d A;
    cameraMatrix.convertTo(A, CV_64F);

    // 1. initialize intrinsic parameters & LM solver
    if( flags & CALIB_USE_INTRINSIC_GUESS )
    {
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
        distCoeffs.convertTo(_k, CV_64F);
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
            aspectRatio = A(0, 0);
            aspectRatio /= A(1, 1);
            if( aspectRatio < minValidAspectRatio || aspectRatio > maxValidAspectRatio )
                CV_Error( cv::Error::StsOutOfRange,
                    "The specified aspect ratio (= cameraMatrix[0][0] / cameraMatrix[1][1]) is incorrect" );
        }
        initIntrinsicParams2D( matM, _m, npoints, imageSize, A, aspectRatio );
    }

    CvLevMarq solver( nparams, 0, cvTermCriteria(termCrit) );

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
    std::copy(k.begin(), k.end(), param + 4);

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

    Mat_<double> param_m = cvarrToMat(solver.param);
    Mat mask = cvarrToMat(solver.mask);
    int nparams_nz = countNonZero(mask);

    if (nparams_nz >= 2 * total)
        CV_Error_(Error::StsBadArg,
                  ("There should be less vars to optimize (having %d) than the number of residuals (%d = 2 per point)", nparams_nz, 2 * total));

    // 2. initialize extrinsic parameters
    for(int i = 0, pos = 0; i < nimages; i++ )
    {
        int ni = npoints.at<int>(i);

        int s = NINTRINSIC + i*6;
        Mat _ri = param_m.rowRange(s, s + 3);
        Mat _ti = param_m.rowRange(s + 3, s + 6);

        Mat _Mi = matM.colRange(pos, pos + ni);
        Mat _mi = _m.colRange(pos, pos + ni);

        findExtrinsicCameraParams2( _Mi, _mi, Mat(A), _k, _ri, _ti, /*useExtrinsicGuess=*/0 );

        pos += ni;
    }

    // 3. run the optimization

    Mat errBuf( maxPoints*2, 1, CV_64FC1 );
    Mat JiBuf ( maxPoints*2, NINTRINSIC, CV_64FC1, Scalar(0));
    Mat JeBuf ( maxPoints*2, 6, CV_64FC1 );
    Mat JoBuf;
    if (releaseObject)
        JoBuf = Mat( maxPoints*2, maxPoints*3, CV_64FC1);

    for(;;)
    {
        bool optimizeObjPoints = releaseObject;

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

        double fx = param_m(0), fy = param_m(1), cx = param_m(2), cy = param_m(3);
        Matx33d intrin(fx,  0, cx,
                        0, fy, cy,
                        0,  0,  1);
        Mat dist = param_m.rowRange(4, 4+14);

        if ( !proceed && stdDevs.empty() && perViewErr.empty() )
            break;
        else if ( !proceed && !stdDevs.empty() )
            cvZero(_JtJ);

        reprojErr = 0;

        int so = NINTRINSIC + nimages * 6;
        int pos = 0;
        for( int i = 0; i < nimages; i++ )
        {
            int si = NINTRINSIC + i * 6;

            int ni = npoints.at<int>(i);
            Mat _ri = param_m.rowRange(si, si + 3);
            Mat _ti = param_m.rowRange(si + 3, si + 6);

            Mat _Mi = matM.colRange(pos, pos + ni);
            if( optimizeObjPoints )
            {
                _Mi = param_m.rowRange(so, so + ni * 3);
                _Mi = _Mi.reshape(3, 1);
            }
            Mat _mi = _m.colRange(pos, pos + ni);
            Mat _me = allErrors.colRange(pos, pos + ni);

            Mat Jo;
            if (optimizeObjPoints)
                Jo = JoBuf(Range(0, ni*2), Range(0, ni*3));

            Mat Je  = JeBuf.rowRange(0, ni*2);
            Mat Ji  = JiBuf.rowRange(0, ni*2);
            Mat err = errBuf.rowRange(0, ni*2);
            Mat _mp = err.reshape(2, 1);

            if( calcJ )
            {
                Mat _dpdr = Je.colRange(0, 3);
                Mat _dpdt = Je.colRange(3, 6);
                Mat _dpdf = (flags & CALIB_FIX_FOCAL_LENGTH) ? Mat() : Ji.colRange(0, 2);
                Mat _dpdc = (flags & CALIB_FIX_PRINCIPAL_POINT) ? Mat() : Ji.colRange(2, 4);
                Mat _dpdk = Ji.colRange(4, NINTRINSIC);
                Mat _dpdo = Jo.empty() ? Mat() : Jo.colRange(0, ni * 3);

                projectPoints(_Mi, _ri, _ti, intrin, dist, _mp, _dpdr, _dpdt,
                              (_dpdf.empty() ? noArray() : _dpdf),
                              (_dpdc.empty() ? noArray() : _dpdc),
                              _dpdk, (_dpdo.empty() ? noArray() : _dpdo),
                              (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio : 0.);
            }
            else
                projectPoints( _Mi, _ri, _ti, intrin, dist, _mp);

            subtract( _mp, _mi, _mp );
            _mp.copyTo(_me);

            if( calcJ )
            {
                Mat JtJ(cvarrToMat(_JtJ)), JtErr(cvarrToMat(_JtErr));

                // see HZ: (A6.14) for details on the structure of the Jacobian
                JtJ(Rect(0, 0, NINTRINSIC, NINTRINSIC)) += Ji.t() * Ji;
                JtJ(Rect(si, si, 6, 6)) = Je.t() * Je;
                JtJ(Rect(si,  0, 6, NINTRINSIC)) = Ji.t() * Je;
                if( optimizeObjPoints )
                {
                    JtJ(Rect(so, 0, maxPoints * 3, NINTRINSIC)) += Ji.t() * Jo;
                    JtJ(Rect(so, si, maxPoints * 3, 6)) += Je.t() * Jo;
                    JtJ(Rect(so, so, maxPoints * 3, maxPoints * 3)) += Jo.t() * Jo;
                }

                JtErr.rowRange(0, NINTRINSIC) += Ji.t() * err;
                JtErr.rowRange(si, si + 6) = Je.t() * err;
                if( optimizeObjPoints )
                {
                    JtErr.rowRange(so, nparams) += Jo.t() * err;
                }
            }

            double viewErr = norm(err, NORM_L2SQR);
            if( !perViewErr.empty() )
                perViewErr.at<double>(i) = std::sqrt(viewErr / ni);

            reprojErr += viewErr;
            pos += ni;
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
    double * param = solver.param->data.db;
    A = Matx33d(param[0], 0, param[2], 0, param[1], param[3], 0, 0, 1);
    Mat(A).convertTo(cameraMatrix, cameraMatrix.type());
    _k = Mat(distCoeffs.size(), CV_64F, param + 4);
    _k.convertTo(distCoeffs, distCoeffs.type());

    if( !newObjPoints.empty() && releaseObject )
    {
        int s = NINTRINSIC + nimages * 6;
        Mat _Mi = param_m.rowRange(s, s + maxPoints * 3);
        _Mi.reshape(3, 1).convertTo(newObjPoints, newObjPoints.type());
    }

    for(int i = 0; i < nimages; i++ )
    {
        if( !rvecs.empty() )
        {
            Mat src = Mat(3, 1, CV_64F, param + NINTRINSIC + i*6);
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
            Mat src(3, 1, CV_64F, param + NINTRINSIC + i*6 + 3);
            Mat dst(3, 1, tvecs.depth(), tvecs.rows == 1 ?
                    tvecs.data + i*tvecs.elemSize() : tvecs.ptr(i));
            src.convertTo(dst, dst.type());
        }
    }

    return std::sqrt(reprojErr/total);
}

//////////////////////////////// Stereo Calibration ///////////////////////////////////

static double stereoCalibrateImpl(
        const Mat& _objectPoints, const Mat& _imagePoints1,
        const Mat& _imagePoints2, const Mat& _npoints,
        Mat& _cameraMatrix1, Mat& _distCoeffs1,
        Mat& _cameraMatrix2, Mat& _distCoeffs2,
        Size imageSize, Mat matR, Mat matT,
        Mat matE, Mat matF,
        Mat rvecs, Mat tvecs,
        Mat perViewErr, int flags,
        TermCriteria termCrit )
{
    int NINTRINSIC = CALIB_NINTRINSIC;
    double reprojErr = 0;

    // initial camera intrinsicss
    Vec<double, 14> distInitial[2];
    Matx33d A[2];
    int pointsTotal = 0, maxPoints = 0, nparams;
    bool recomputeIntrinsics = false;
    double aspectRatio[2] = {0, 0};

    CV_Assert( _imagePoints1.type() == _imagePoints2.type() &&
               _imagePoints1.depth() == _objectPoints.depth() );

    CV_Assert( (_npoints.cols == 1 || _npoints.rows == 1) &&
                _npoints.type() == CV_32S );

    int nimages = (int)_npoints.total();
    for(int i = 0; i < nimages; i++ )
    {
        int ni = _npoints.at<int>(i);
        maxPoints = std::max(maxPoints, ni);
        pointsTotal += ni;
    }

    Mat objectPoints;
    Mat imagePoints[2];
    _objectPoints.convertTo(objectPoints, CV_64F);
    objectPoints = objectPoints.reshape(3, 1);

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

    for(int k = 0; k < 2; k++ )
    {
        const Mat& points = k == 0 ? _imagePoints1 : _imagePoints2;
        const Mat& cameraMatrix = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
        const Mat& distCoeffs = k == 0 ? _distCoeffs1 : _distCoeffs2;

        int depth = points.depth();
        int cn = points.channels();
        CV_Assert( (depth == CV_32F || depth == CV_64F) &&
                   ((points.rows == pointsTotal && points.cols*cn == 2) ||
                   (points.rows == 1 && points.cols == pointsTotal && cn == 2)));

        A[k] = Matx33d::eye();

        points.convertTo(imagePoints[k], CV_64F);
        imagePoints[k] = imagePoints[k].reshape(2, 1);

        if( flags & ( CALIB_FIX_INTRINSIC | CALIB_USE_INTRINSIC_GUESS |
                      CALIB_FIX_ASPECT_RATIO | CALIB_FIX_FOCAL_LENGTH ) )
            cameraMatrix.convertTo(A[k], CV_64F);

        if( flags & ( CALIB_FIX_INTRINSIC | CALIB_USE_INTRINSIC_GUESS |
                      CALIB_FIX_K1 | CALIB_FIX_K2 | CALIB_FIX_K3 | CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6 |
                      CALIB_FIX_TANGENT_DIST) )
        {
            Mat tdist( distCoeffs.size(), CV_MAKETYPE(CV_64F, distCoeffs.channels()), distInitial[k].val);
            distCoeffs.convertTo(tdist, CV_64F);
        }

        if( !(flags & (CALIB_FIX_INTRINSIC | CALIB_USE_INTRINSIC_GUESS)))
        {
            Mat mIntr(A[k], /* copyData = */ false);
            Mat mDist(distInitial[k], /* copyData = */ false);
            calibrateCameraInternal(objectPoints, imagePoints[k],
                                    _npoints, imageSize, 0, mIntr, mDist,
                                    Mat(), Mat(), Mat(), Mat(), Mat(), flags, termCrit);
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
        for(int k = 0; k < 2; k++ )
            aspectRatio[k] = A[k](0, 0) / A[k](1, 1);
    }

    recomputeIntrinsics = (flags & CALIB_FIX_INTRINSIC) == 0;

    // we optimize for the inter-camera R(3),t(3), then, optionally,
    // for intrinisic parameters of each camera ((fx,fy,cx,cy,k1,k2,p1,p2) ~ 8 parameters).
    // Param mapping is:
    // - from 0 next 6: stereo pair Rt, from 6+i*6 next 6: Rt for each ith camera of nimages,
    // - from 6*(nimages+1) next NINTRINSICS: intrinsics for 1st camera: fx, fy, cx, cy, 14 x dist
    // - next NINTRINSICS: the same for for 2nd camera
    nparams = 6*(nimages+1) + (recomputeIntrinsics ? NINTRINSIC*2 : 0);

    CvLevMarq solver( nparams, 0, cvTermCriteria(termCrit) );
    double * param = solver.param->data.db;
    Mat paramM = Mat(solver.param->rows, solver.param->cols, CV_64F, param);
    uchar* mask = solver.mask->data.ptr;

    if(flags & CALIB_USE_LU) {
        solver.solveMethod = DECOMP_LU;
    }

    if( recomputeIntrinsics )
    {
        size_t idx = nparams - NINTRINSIC*2;
        if( !(flags & CALIB_RATIONAL_MODEL) )
            flags |= CALIB_FIX_K4 | CALIB_FIX_K5 | CALIB_FIX_K6;
        if( !(flags & CALIB_THIN_PRISM_MODEL) )
            flags |= CALIB_FIX_S1_S2_S3_S4;
        if( !(flags & CALIB_TILTED_MODEL) )
            flags |= CALIB_FIX_TAUX_TAUY;
        if( flags & CALIB_FIX_ASPECT_RATIO )
            mask[idx + 0] = mask[idx + NINTRINSIC] = 0;
        if ( flags & CALIB_SAME_FOCAL_LENGTH)
            mask[idx + NINTRINSIC] = mask[idx + NINTRINSIC + 1] = 0;
        if( flags & CALIB_FIX_FOCAL_LENGTH )
            mask[idx + 0] = mask[idx + 1] = mask[idx + NINTRINSIC] = mask[idx + NINTRINSIC+1] = 0;
        if( flags & CALIB_FIX_PRINCIPAL_POINT )
            mask[idx + 2] = mask[idx + 3] = mask[idx + NINTRINSIC+2] = mask[idx + NINTRINSIC+3] = 0;
        if( flags & (CALIB_ZERO_TANGENT_DIST|CALIB_FIX_TANGENT_DIST) )
            mask[idx + 6] = mask[idx + 7] = mask[idx + NINTRINSIC+6] = mask[idx + NINTRINSIC+7] = 0;
        if( flags & CALIB_FIX_K1 )
            mask[idx + 4] = mask[idx + NINTRINSIC+4] = 0;
        if( flags & CALIB_FIX_K2 )
            mask[idx + 5] = mask[idx + NINTRINSIC+5] = 0;
        if( flags & CALIB_FIX_K3 )
            mask[idx + 8] = mask[idx + NINTRINSIC+8] = 0;
        if( flags & CALIB_FIX_K4 )
            mask[idx + 9] = mask[idx + NINTRINSIC+9] = 0;
        if( flags & CALIB_FIX_K5 )
            mask[idx + 10] = mask[idx + NINTRINSIC+10] = 0;
        if( flags & CALIB_FIX_K6 )
            mask[idx + 11] = mask[idx + NINTRINSIC+11] = 0;
        if( flags & CALIB_FIX_S1_S2_S3_S4 )
        {
            mask[idx + 12] = mask[idx + NINTRINSIC+12] = 0;
            mask[idx + 13] = mask[idx + NINTRINSIC+13] = 0;
            mask[idx + 14] = mask[idx + NINTRINSIC+14] = 0;
            mask[idx + 15] = mask[idx + NINTRINSIC+15] = 0;
        }
        if( flags & CALIB_FIX_TAUX_TAUY )
        {
            mask[idx + 16] = mask[idx + NINTRINSIC+16] = 0;
            mask[idx + 17] = mask[idx + NINTRINSIC+17] = 0;
        }
    }

    // storage for initial [om(R){i}|t{i}] (in order to compute the median for each component)
    std::vector<double> rtsort(nimages*6);
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
    int pos = 0;
    for(int i = 0; i < nimages; i++ )
    {
        int ni = _npoints.at<int>(i);
        Mat objpt_i = objectPoints.colRange(pos, pos + ni);
        Matx33d R[2];
        Mat_<double> rv, T[2];
        for(int k = 0; k < 2; k++ )
        {
            Mat imgpt_ik = imagePoints[k].colRange(pos, pos + ni);
            solvePnP(objpt_i, imgpt_ik, A[k], distInitial[k], rv, T[k], false, SOLVEPNP_ITERATIVE );
            findExtrinsicCameraParams2( objpt_i, imgpt_ik, Mat(A[k]), Mat(distInitial[k]),
                                        rv, T[k], /*useExtrinsicGuess=*/0 );
            Rodrigues(rv, R[k]);

            if( k == 0 )
            {
                // save initial om_left and T_left
                param[(i+1)*6 + 0] = rv(0);
                param[(i+1)*6 + 1] = rv(1);
                param[(i+1)*6 + 2] = rv(2);
                param[(i+1)*6 + 3] = T[0](0);
                param[(i+1)*6 + 4] = T[0](1);
                param[(i+1)*6 + 5] = T[0](2);
            }
        }
        R[0] = R[1]*R[0].t();
        T[1] -= R[0]*T[0];

        Rodrigues(R[0], rv);

        rtsort[i + nimages*0] = rv(0);
        rtsort[i + nimages*1] = rv(1);
        rtsort[i + nimages*2] = rv(2);
        rtsort[i + nimages*3] = T[1](0);
        rtsort[i + nimages*4] = T[1](1);
        rtsort[i + nimages*5] = T[1](2);

        pos += ni;
    }

    if(flags & CALIB_USE_EXTRINSIC_GUESS)
    {
        Vec3d R, T;
        matT.convertTo(T, CV_64F);

        if( matR.rows == 3 && matR.cols == 3 )
            Rodrigues(matR, R);
        else
            matR.convertTo(R, CV_64F);

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
        for(int i = 0; i < 6; i++ )
        {
            size_t idx = i*nimages;
            std::nth_element(rtsort.begin() + idx,
                             rtsort.begin() + idx + nimages/2,
                             rtsort.begin() + idx + nimages);
            double h = rtsort[idx + nimages/2];
            param[i] = (nimages % 2 == 0) ? (h + rtsort[idx + nimages/2 - 1]) * 0.5 : h;
        }
    }

    if( recomputeIntrinsics )
    {
        for(int k = 0; k < 2; k++ )
        {
            size_t idx = (nimages+1)*6 + k*NINTRINSIC;
            if( flags & CALIB_ZERO_TANGENT_DIST )
                distInitial[k][2] = distInitial[k][3] = 0;
            param[idx +  0] = A[k](0, 0); param[idx + 1] = A[k](1, 1); param[idx + 2] = A[k](0, 2); param[idx + 3] = A[k](1, 2);
            for (int i = 0; i < 14; i++)
            {
                param[idx + 4 + i] = distInitial[k][i];
            }
        }
    }

    // Preallocated place for callback calculations
    Mat errBuf( maxPoints*2, 1, CV_64F );
    Mat JeBuf( maxPoints*2, 6, CV_64F );
    Mat J_LRBuf( maxPoints*2, 6, CV_64F );
    Mat JiBuf( maxPoints*2, NINTRINSIC, CV_64F, Scalar(0) );

    for(;;)
    {
        const CvMat* tmp_param = 0;
        CvMat *JtJ = 0, *JtErr = 0;
        double *_errNorm = 0;
        Mat_<double> param_m(1,nparams, solver.param->data.db);
        Vec3d om_LR(param_m(0), param_m(1), param_m(2));
        Vec3d T_LR(param_m(3), param_m(4), param_m(5));
        Vec3d om[2], T[2];
        Matx33d dr3dr1, dr3dr2, dt3dr2, dt3dt1, dt3dt2;
        Matx33d intrin[2];
        if( !solver.updateAlt( tmp_param, JtJ, JtErr, _errNorm ))
            break;
        std::vector< std::vector<double> > distCoeffs(2, std::vector<double>(14, 0.0));

        reprojErr = 0;

        if( recomputeIntrinsics )
        {
            int idx = (nimages+1)*6;

            if( flags & CALIB_SAME_FOCAL_LENGTH )
            {
                param_m(idx + NINTRINSIC  ) = param_m(idx + 0);
                param_m(idx + NINTRINSIC+1) = param_m(idx + 1);
            }
            if( flags & CALIB_FIX_ASPECT_RATIO )
            {
                param_m(idx + 0)          = aspectRatio[0]*param_m(idx + 1             );
                param_m(idx + NINTRINSIC) = aspectRatio[1]*param_m(idx + 1 + NINTRINSIC);
            }
            for(int k = 0; k < 2; k++ )
            {
                double fx = param_m(idx + k*NINTRINSIC+0), fy = param_m(idx + k*NINTRINSIC+1);
                double cx = param_m(idx + k*NINTRINSIC+2), cy = param_m(idx + k*NINTRINSIC+3);
                intrin[k] = Matx33d(fx,  0, cx,
                                     0, fy, cy,
                                     0,  0,  1);
                for(int j = 0; j < 14; j++)
                    distCoeffs[k][j] = param_m(idx + k*NINTRINSIC+4+j);
            }
        }
        else
        {
            for (int k = 0; k < 2; k++)
            {
                intrin[k] = A[k];
                for(int j = 0; j < 14; j++)
                    distCoeffs[k][j] = distInitial[k][j];
            }
        }

        int ptPos = 0;
        for(int i = 0; i < nimages; i++ )
        {
            int ni = _npoints.at<int>(i);

            int idx = (i+1)*6;
            om[0] = Vec3d(param_m(idx + 0), param_m(idx + 1), param_m(idx + 2));
            T[0]  = Vec3d(param_m(idx + 3), param_m(idx + 4), param_m(idx + 5));

            if( JtJ || JtErr )
                composeRT( om[0], T[0], om_LR, T_LR, om[1], T[1], dr3dr1, noArray(),
                           dr3dr2, noArray(), noArray(), dt3dt1, dt3dr2, dt3dt2 );
            else
                composeRT( om[0], T[0], om_LR, T_LR, om[1], T[1] );

            Mat objpt_i = objectPoints(Range::all(), Range(ptPos, ptPos + ni));
            Mat err  = errBuf (Range(0, ni*2), Range::all());
            Mat Je   = JeBuf  (Range(0, ni*2), Range::all());
            Mat J_LR = J_LRBuf(Range(0, ni*2), Range::all());
            Mat Ji   = JiBuf  (Range(0, ni*2), Range::all());

            Mat tmpImagePoints = err.reshape(2, 1);
            Mat dpdf = Ji.colRange(0, 2);
            Mat dpdc = Ji.colRange(2, 4);
            Mat dpdk = Ji.colRange(4, NINTRINSIC);
            Mat dpdrot = Je.colRange(0, 3);
            Mat dpdt = Je.colRange(3, 6);

            for(int k = 0; k < 2; k++ )
            {
                Mat imgpt_ik = imagePoints[k](Range::all(), Range(ptPos, ptPos + ni));

                if( JtJ || JtErr )
                    projectPoints(objpt_i, om[k], T[k], intrin[k], distCoeffs[k],
                                  tmpImagePoints, dpdrot, dpdt, dpdf, dpdc, dpdk, noArray(),
                                  (flags & CALIB_FIX_ASPECT_RATIO) ? aspectRatio[k] : 0.);
                else
                    projectPoints(objpt_i, om[k], T[k], intrin[k], distCoeffs[k], tmpImagePoints);
                subtract( tmpImagePoints, imgpt_ik, tmpImagePoints );

                if( solver.state == CvLevMarq::CALC_J )
                {
                    int iofs = (nimages+1)*6 + k*NINTRINSIC, eofs = (i+1)*6;
                    CV_Assert( JtJ && JtErr );

                    Mat _JtJ(cvarrToMat(JtJ)), _JtErr(cvarrToMat(JtErr));

                    if( k == 1 )
                    {
                        // d(err_{x|y}R) ~ de3
                        // convert de3/{dr3,dt3} => de3{dr1,dt1} & de3{dr2,dt2}
                        for(int p = 0; p < ni*2; p++ )
                        {
                            Matx13d de3dr3, de3dt3, de3dr2, de3dt2, de3dr1, de3dt1;
                            for(int j = 0; j < 3; j++)
                                de3dr3(j) = Je.at<double>(p, j);

                            for(int j = 0; j < 3; j++)
                                de3dt3(j) = Je.at<double>(p, 3+j);

                            for(int j = 0; j < 3; j++)
                                de3dr2(j) = J_LR.at<double>(p, j);

                            for(int j = 0; j < 3; j++)
                                de3dt2(j) = J_LR.at<double>(p, 3+j);

                            de3dr1 = de3dr3 * dr3dr1;
                            de3dt1 = de3dt3 * dt3dt1;
                            de3dr2  = de3dr3 * dr3dr2 + de3dt3 * dt3dr2;
                            de3dt2  = de3dt3 * dt3dt2;

                            for(int j = 0; j < 3; j++)
                                Je.at<double>(p, j) = de3dr1(j);

                            for(int j = 0; j < 3; j++)
                                Je.at<double>(p, 3+j) = de3dt1(j);

                            for(int j = 0; j < 3; j++)
                                J_LR.at<double>(p, j) = de3dr2(j);

                            for(int j = 0; j < 3; j++)
                                J_LR.at<double>(p, 3+j) = de3dt2(j);
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
            ptPos += ni;
        }
        if(_errNorm)
            *_errNorm = reprojErr;
    }

    // Extract optimized params from the param vector

    Vec3d om_LR(param[0], param[1], param[2]);
    Vec3d T_LR(param[3], param[4], param[5]);
    Matx33d R_LR;
    Rodrigues( om_LR, R_LR );
    if( matR.rows == 1 || matR.cols == 1 )
        Mat(om_LR).convertTo(matR, matR.depth());
    else
        Mat(R_LR).convertTo(matR, matR.depth());
    Mat(T_LR).convertTo(matT, matT.depth());

    if( recomputeIntrinsics )
    {
        for(int k = 0; k < 2; k++ )
        {
            size_t idx = (nimages+1)*6 + k*NINTRINSIC;
            A[k] = Matx33d(param[idx + 0], 0, param[idx + 2], 0, param[idx + 1], param[idx + 3], 0, 0, 1);

            Mat& cameraMatrix = k == 0 ? _cameraMatrix1 : _cameraMatrix2;
            Mat& distCoeffs = k == 0 ? _distCoeffs1 : _distCoeffs2;
            Mat(A[k]).convertTo(cameraMatrix, cameraMatrix.depth());

            std::vector<double> vdist(14);
            for(int j = 0; j < 14; j++)
                vdist[j] = param[idx + 4 + j];

            Mat tdist( distCoeffs.size(), CV_MAKETYPE(CV_64F, distCoeffs.channels()), vdist.data());
            tdist.convertTo(distCoeffs, distCoeffs.depth());
        }
    }

    if( !matE.empty() || !matF.empty() )
    {
        Matx33d Tx(0, -T_LR[2], T_LR[1],
                   T_LR[2], 0, -T_LR[0],
                  -T_LR[1], T_LR[0], 0);
        Matx33d E = Tx*R_LR;
        if( !matE.empty() )
            Mat(E).convertTo(matE, matE.depth());
        if( !matF.empty())
        {
            Matx33d iA0 = A[0].inv(), iA1 = A[1].inv();
            Matx33d F = iA1.t() * E * iA0;
            Mat(F).convertTo(matF, matF.depth(), fabs(F(2,2)) > 0 ? 1./F(2,2) : 1.);
        }
    }

    Mat r1d = rvecs.empty() ? Mat() : rvecs.reshape(1, nimages);
    Mat t1d = tvecs.empty() ? Mat() : tvecs.reshape(1, nimages);
    for(int i = 0; i < nimages; i++ )
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

    double reprojErr = calibrateCameraInternal(
            objPt, imgPt, npoints, imageSize, iFixedPoint,
            cameraMatrix, distCoeffs,
            rvecM, tvecM,
            newObjPt,
            stdDeviationsM,
            errorsM, flags, criteria);

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
    Mat matR = _Rmat.getMat(), matT = _Tmat.getMat();

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

    double err = stereoCalibrateImpl(objPt, imgPt, imgPt2, npoints, cameraMatrix1,
                                     distCoeffs1, cameraMatrix2, distCoeffs2, imageSize,
                                     matR, matT, matE, matF, rvecLM, tvecLM,
                                     matErr, flags, criteria);
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
