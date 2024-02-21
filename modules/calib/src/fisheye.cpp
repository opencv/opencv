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
// Copyright (C) 2009-2011, Willow Garage Inc., all rights reserved.
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
#include "fisheye.hpp"
#include <limits>

namespace cv {
namespace {

void subMatrix(const Mat& src, Mat& dst, const std::vector<uchar>& cols, const std::vector<uchar>& rows);

}}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::calibrate

double cv::fisheye::calibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints, const Size& image_size,
                                    InputOutputArray K, InputOutputArray D, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                    int flags , cv::TermCriteria criteria)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!objectPoints.empty() && !imagePoints.empty() && objectPoints.total() == imagePoints.total());
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    CV_Assert(imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2);
    CV_Assert(K.empty() || (K.size() == Size(3,3)));
    CV_Assert(D.empty() || (D.total() == 4));
    CV_Assert(rvecs.empty() || (rvecs.channels() == 3));
    CV_Assert(tvecs.empty() || (tvecs.channels() == 3));

    CV_Assert((!K.empty() && !D.empty()) || !(flags & CALIB_USE_INTRINSIC_GUESS));

    using namespace cv::internal;
    //-------------------------------Initialization
    IntrinsicParams finalParam;
    IntrinsicParams currentParam;
    IntrinsicParams errors;

    finalParam.isEstimate[0] = flags & CALIB_FIX_FOCAL_LENGTH ? 0 : 1;
    finalParam.isEstimate[1] = flags & CALIB_FIX_FOCAL_LENGTH ? 0 : 1;
    finalParam.isEstimate[2] = flags & CALIB_FIX_PRINCIPAL_POINT ? 0 : 1;
    finalParam.isEstimate[3] = flags & CALIB_FIX_PRINCIPAL_POINT ? 0 : 1;
    finalParam.isEstimate[4] = flags & CALIB_FIX_SKEW ? 0 : 1;
    finalParam.isEstimate[5] = flags & CALIB_FIX_K1 ? 0 : 1;
    finalParam.isEstimate[6] = flags & CALIB_FIX_K2 ? 0 : 1;
    finalParam.isEstimate[7] = flags & CALIB_FIX_K3 ? 0 : 1;
    finalParam.isEstimate[8] = flags & CALIB_FIX_K4 ? 0 : 1;

    const int recompute_extrinsic = flags & CALIB_RECOMPUTE_EXTRINSIC ? 1: 0;
    const int check_cond = flags & CALIB_CHECK_COND ? 1 : 0;

    const double alpha_smooth = 0.4;
    const double thresh_cond = 1e6;
    double change = 1;
    Vec2d err_std;

    Matx33d _K;
    Vec4d _D;
    if (flags & CALIB_USE_INTRINSIC_GUESS)
    {
        K.getMat().convertTo(_K, CV_64FC1);
        D.getMat().convertTo(_D, CV_64FC1);
        finalParam.Init(Vec2d(_K(0,0), _K(1, 1)),
                        Vec2d(_K(0,2), _K(1, 2)),
                        Vec4d(flags & CALIB_FIX_K1 ? 0 : _D[0],
                              flags & CALIB_FIX_K2 ? 0 : _D[1],
                              flags & CALIB_FIX_K3 ? 0 : _D[2],
                              flags & CALIB_FIX_K4 ? 0 : _D[3]),
                        _K(0, 1) / _K(0, 0));
    }
    else
    {
        finalParam.Init(Vec2d(max(image_size.width, image_size.height) / 2., max(image_size.width, image_size.height) / 2.),
                        Vec2d(image_size.width  / 2.0 - 0.5, image_size.height / 2.0 - 0.5));
    }

    errors.isEstimate = finalParam.isEstimate;

    std::vector<Vec3d> omc(objectPoints.total()), Tc(objectPoints.total());

    CalibrateExtrinsics(objectPoints, imagePoints, finalParam, check_cond, thresh_cond, omc, Tc);


    //-------------------------------Optimization
    for(int iter = 0; iter < std::numeric_limits<int>::max(); ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;

        double alpha_smooth2 = 1 - std::pow(1 - alpha_smooth, iter + 1.0);

        Mat JJ2, ex3;
        ComputeJacobians(objectPoints, imagePoints, finalParam, omc, Tc, check_cond,thresh_cond, JJ2, ex3);

        Mat G;
        solve(JJ2, ex3, G);
        currentParam = finalParam + alpha_smooth2*G;

        change = norm(Vec4d(currentParam.f[0], currentParam.f[1], currentParam.c[0], currentParam.c[1]) -
                Vec4d(finalParam.f[0], finalParam.f[1], finalParam.c[0], finalParam.c[1]))
                / norm(Vec4d(currentParam.f[0], currentParam.f[1], currentParam.c[0], currentParam.c[1]));

        finalParam = currentParam;

        if (recompute_extrinsic)
        {
            CalibrateExtrinsics(objectPoints,  imagePoints, finalParam, check_cond,
                                    thresh_cond, omc, Tc);
        }
    }

    //-------------------------------Validation
    double rms;
    EstimateUncertainties(objectPoints, imagePoints, finalParam,  omc, Tc, errors, err_std, thresh_cond,
                              check_cond, rms);

    //-------------------------------
    _K = Matx33d(finalParam.f[0], finalParam.f[0] * finalParam.alpha, finalParam.c[0],
            0,                    finalParam.f[1], finalParam.c[1],
            0,                                  0,               1);

    if (K.needed()) Mat(_K).convertTo(K, K.empty() ? CV_64FC1 : K.type());
    if (D.needed()) Mat(finalParam.k).convertTo(D, D.empty() ? CV_64FC1 : D.type());
    if (rvecs.isMatVector())
    {
        int N = (int)objectPoints.total();

        if(rvecs.empty())
            rvecs.create(N, 1, CV_64FC3);

        if(tvecs.empty())
            tvecs.create(N, 1, CV_64FC3);

        for(int i = 0; i < N; i++ )
        {
            rvecs.create(3, 1, CV_64F, i, true);
            tvecs.create(3, 1, CV_64F, i, true);
            memcpy(rvecs.getMat(i).ptr(), omc[i].val, sizeof(Vec3d));
            memcpy(tvecs.getMat(i).ptr(), Tc[i].val, sizeof(Vec3d));
        }
    }
    else
    {
        if (rvecs.needed()) Mat(omc).reshape(3, (int)omc.size()).convertTo(rvecs, rvecs.empty() ? CV_64FC3 : rvecs.type());
        if (tvecs.needed()) Mat(Tc).reshape(3, (int)Tc.size()).convertTo(tvecs, tvecs.empty() ? CV_64FC3 : tvecs.type());
    }

    return rms;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// cv::fisheye::stereoCalibrate

double cv::fisheye::stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                    InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                    OutputArray R, OutputArray T, int flags, TermCriteria criteria)
{
    return cv::fisheye::stereoCalibrate(objectPoints, imagePoints1, imagePoints2, K1, D1, K2, D2, imageSize, R, T,
                                        noArray(), noArray(), flags, criteria);
}

double cv::fisheye::stereoCalibrate(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints1, InputArrayOfArrays imagePoints2,
                                    InputOutputArray K1, InputOutputArray D1, InputOutputArray K2, InputOutputArray D2, Size imageSize,
                                    OutputArray R, OutputArray T, OutputArrayOfArrays rvecs, OutputArrayOfArrays tvecs,
                                    int flags, TermCriteria criteria)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!objectPoints.empty() && !imagePoints1.empty() && !imagePoints2.empty());
    CV_Assert(objectPoints.total() == imagePoints1.total() || imagePoints1.total() == imagePoints2.total());
    CV_Assert(objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3);
    CV_Assert(imagePoints1.type() == CV_32FC2 || imagePoints1.type() == CV_64FC2);
    CV_Assert(imagePoints2.type() == CV_32FC2 || imagePoints2.type() == CV_64FC2);

    CV_Assert(K1.empty() || (K1.size() == Size(3,3)));
    CV_Assert(D1.empty() || (D1.total() == 4));
    CV_Assert(K2.empty() || (K2.size() == Size(3,3)));
    CV_Assert(D2.empty() || (D2.total() == 4));

    CV_Assert((!K1.empty() && !K2.empty() && !D1.empty() && !D2.empty()) || !(flags & CALIB_FIX_INTRINSIC));

    //-------------------------------Initialization

    const int threshold = 50;
    const double thresh_cond = 1e6;
    const int check_cond = 1;

    int n_points = (int)objectPoints.getMat(0).total();
    int n_images = (int)objectPoints.total();

    double change = 1;

    cv::internal::IntrinsicParams intrinsicLeft;
    cv::internal::IntrinsicParams intrinsicRight;

    cv::internal::IntrinsicParams intrinsicLeft_errors;
    cv::internal::IntrinsicParams intrinsicRight_errors;

    Matx33d _K1, _K2;
    Vec4d _D1, _D2;
    if (!K1.empty()) K1.getMat().convertTo(_K1, CV_64FC1);
    if (!D1.empty()) D1.getMat().convertTo(_D1, CV_64FC1);
    if (!K2.empty()) K2.getMat().convertTo(_K2, CV_64FC1);
    if (!D2.empty()) D2.getMat().convertTo(_D2, CV_64FC1);

    std::vector<Vec3d> rvecs1(n_images), tvecs1(n_images), rvecs2(n_images), tvecs2(n_images);

    if (!(flags & CALIB_FIX_INTRINSIC))
    {
        calibrate(objectPoints, imagePoints1, imageSize, _K1, _D1, rvecs1, tvecs1, flags, TermCriteria(3, 20, 1e-6));
        calibrate(objectPoints, imagePoints2, imageSize, _K2, _D2, rvecs2, tvecs2, flags, TermCriteria(3, 20, 1e-6));
    }

    intrinsicLeft.Init(Vec2d(_K1(0,0), _K1(1, 1)), Vec2d(_K1(0,2), _K1(1, 2)),
                       Vec4d(_D1[0], _D1[1], _D1[2], _D1[3]), _K1(0, 1) / _K1(0, 0));

    intrinsicRight.Init(Vec2d(_K2(0,0), _K2(1, 1)), Vec2d(_K2(0,2), _K2(1, 2)),
                        Vec4d(_D2[0], _D2[1], _D2[2], _D2[3]), _K2(0, 1) / _K2(0, 0));

    if ((flags & CALIB_FIX_INTRINSIC))
    {
        cv::internal::CalibrateExtrinsics(objectPoints,  imagePoints1, intrinsicLeft, check_cond, thresh_cond, rvecs1, tvecs1);
        cv::internal::CalibrateExtrinsics(objectPoints,  imagePoints2, intrinsicRight, check_cond, thresh_cond, rvecs2, tvecs2);
    }

    intrinsicLeft.isEstimate[0] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[1] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[2] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[3] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicLeft.isEstimate[4] = flags & (CALIB_FIX_SKEW | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[5] = flags & (CALIB_FIX_K1 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[6] = flags & (CALIB_FIX_K2 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[7] = flags & (CALIB_FIX_K3 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicLeft.isEstimate[8] = flags & (CALIB_FIX_K4 | CALIB_FIX_INTRINSIC) ? 0 : 1;

    intrinsicRight.isEstimate[0] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[1] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[2] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[3] = flags & CALIB_FIX_INTRINSIC ? 0 : 1;
    intrinsicRight.isEstimate[4] = flags & (CALIB_FIX_SKEW | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[5] = flags & (CALIB_FIX_K1 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[6] = flags & (CALIB_FIX_K2 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[7] = flags & (CALIB_FIX_K3 | CALIB_FIX_INTRINSIC) ? 0 : 1;
    intrinsicRight.isEstimate[8] = flags & (CALIB_FIX_K4 | CALIB_FIX_INTRINSIC) ? 0 : 1;

    intrinsicLeft_errors.isEstimate = intrinsicLeft.isEstimate;
    intrinsicRight_errors.isEstimate = intrinsicRight.isEstimate;

    std::vector<uchar> selectedParams;
    std::vector<uchar> tmp(6 * (n_images + 1), 1);
    selectedParams.insert(selectedParams.end(), intrinsicLeft.isEstimate.begin(), intrinsicLeft.isEstimate.end());
    selectedParams.insert(selectedParams.end(), intrinsicRight.isEstimate.begin(), intrinsicRight.isEstimate.end());
    selectedParams.insert(selectedParams.end(), tmp.begin(), tmp.end());

    //Init values for rotation and translation between two views
    Mat om_list(1, n_images, CV_64FC3), T_list(1, n_images, CV_64FC3);
    Mat om_ref, R_ref, T_ref, R1, R2;
    for (int image_idx = 0; image_idx < n_images; ++image_idx)
    {
        Rodrigues(rvecs1[image_idx], R1);
        Rodrigues(rvecs2[image_idx], R2);
        R_ref = R2 * R1.t();
        T_ref = Mat(tvecs2[image_idx]) - R_ref * Mat(tvecs1[image_idx]);
        Rodrigues(R_ref, om_ref);
        om_ref.reshape(3, 1).copyTo(om_list.col(image_idx));
        T_ref.reshape(3, 1).copyTo(T_list.col(image_idx));
    }
    Vec3d omcur = cv::internal::median3d(om_list);
    Vec3d Tcur  = cv::internal::median3d(T_list);

    Mat J = Mat::zeros(4 * n_points * n_images, 18 + 6 * (n_images + 1), CV_64FC1),
            e = Mat::zeros(4 * n_points * n_images, 1, CV_64FC1), Jkk, ekk;

    for(int iter = 0; ; ++iter)
    {
        if ((criteria.type == 1 && iter >= criteria.maxCount)  ||
            (criteria.type == 2 && change <= criteria.epsilon) ||
            (criteria.type == 3 && (change <= criteria.epsilon || iter >= criteria.maxCount)))
            break;

        J.create(4 * n_points * n_images, 18 + 6 * (n_images + 1), CV_64FC1);
        e.create(4 * n_points * n_images, 1, CV_64FC1);
        Jkk.create(4 * n_points, 18 + 6 * (n_images + 1), CV_64FC1);
        ekk.create(4 * n_points, 1, CV_64FC1);

        Mat omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT;

        for (int image_idx = 0; image_idx < n_images; ++image_idx)
        {
            Jkk = Mat::zeros(4 * n_points, 18 + 6 * (n_images + 1), CV_64FC1);

            Mat object  = objectPoints.getMat(image_idx).clone();
            Mat imageLeft  = imagePoints1.getMat(image_idx).clone();
            Mat imageRight  = imagePoints2.getMat(image_idx).clone();
            Mat jacobians, projected;

            //left camera jacobian
            Mat rvec = Mat(rvecs1[image_idx]);
            Mat tvec  = Mat(tvecs1[image_idx]);
            cv::internal::projectPoints(object, projected, rvec, tvec, intrinsicLeft, jacobians);
            Mat pt_diff = imageLeft.reshape(1, n_points*2) -
                          projected.reshape(1, n_points*2);
            pt_diff.copyTo(ekk.rowRange(0, 2 * n_points));
            jacobians.colRange(8, 11).copyTo(Jkk.colRange(24 + image_idx * 6, 27 + image_idx * 6).rowRange(0, 2 * n_points));
            jacobians.colRange(11, 14).copyTo(Jkk.colRange(27 + image_idx * 6, 30 + image_idx * 6).rowRange(0, 2 * n_points));
            jacobians.colRange(0, 2).copyTo(Jkk.colRange(0, 2).rowRange(0, 2 * n_points));
            jacobians.colRange(2, 4).copyTo(Jkk.colRange(2, 4).rowRange(0, 2 * n_points));
            jacobians.colRange(4, 8).copyTo(Jkk.colRange(5, 9).rowRange(0, 2 * n_points));
            jacobians.col(14).copyTo(Jkk.col(4).rowRange(0, 2 * n_points));

            //right camera jacobian
            cv::internal::compose_motion(rvec, tvec, omcur, Tcur, omr, Tr, domrdomckk, domrdTckk, domrdom, domrdT, dTrdomckk, dTrdTckk, dTrdom, dTrdT);
            rvec = Mat(rvecs2[image_idx]);
            tvec  = Mat(tvecs2[image_idx]);

            cv::internal::projectPoints(object, projected, omr, Tr, intrinsicRight, jacobians);

            pt_diff = imageRight.reshape(1, n_points*2) -
                      projected.reshape(1, n_points*2);
            pt_diff.copyTo(ekk.rowRange(2 * n_points, 4 * n_points));
            Mat dxrdom = jacobians.colRange(8, 11) * domrdom + jacobians.colRange(11, 14) * dTrdom;
            Mat dxrdT = jacobians.colRange(8, 11) * domrdT + jacobians.colRange(11, 14)* dTrdT;
            Mat dxrdomckk = jacobians.colRange(8, 11) * domrdomckk + jacobians.colRange(11, 14) * dTrdomckk;
            Mat dxrdTckk = jacobians.colRange(8, 11) * domrdTckk + jacobians.colRange(11, 14) * dTrdTckk;

            dxrdom.copyTo(Jkk.colRange(18, 21).rowRange(2 * n_points, 4 * n_points));
            dxrdT.copyTo(Jkk.colRange(21, 24).rowRange(2 * n_points, 4 * n_points));
            dxrdomckk.copyTo(Jkk.colRange(24 + image_idx * 6, 27 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
            dxrdTckk.copyTo(Jkk.colRange(27 + image_idx * 6, 30 + image_idx * 6).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(0, 2).copyTo(Jkk.colRange(9 + 0, 9 + 2).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(2, 4).copyTo(Jkk.colRange(9 + 2, 9 + 4).rowRange(2 * n_points, 4 * n_points));
            jacobians.colRange(4, 8).copyTo(Jkk.colRange(9 + 5, 9 + 9).rowRange(2 * n_points, 4 * n_points));
            jacobians.col(14).copyTo(Jkk.col(9 + 4).rowRange(2 * n_points, 4 * n_points));

            //check goodness of sterepair
            double abs_max  = 0;
            for (int i = 0; i < 4 * n_points; i++)
            {
                if (fabs(ekk.at<double>(i)) > abs_max)
                {
                    abs_max = fabs(ekk.at<double>(i));
                }
            }

            CV_Assert(abs_max < threshold); // bad stereo pair

            Jkk.copyTo(J.rowRange(image_idx * 4 * n_points, (image_idx + 1) * 4 * n_points));
            ekk.copyTo(e.rowRange(image_idx * 4 * n_points, (image_idx + 1) * 4 * n_points));
        }

        Vec6d oldTom(Tcur[0], Tcur[1], Tcur[2], omcur[0], omcur[1], omcur[2]);

        //update all parameters
        cv::subMatrix(J, J, selectedParams, std::vector<uchar>(J.rows, 1));
        int a = cv::countNonZero(intrinsicLeft.isEstimate);
        int b = cv::countNonZero(intrinsicRight.isEstimate);
        Mat deltas;
        solve(J.t() * J, J.t()*e, deltas);
        if (a > 0)
            intrinsicLeft = intrinsicLeft + deltas.rowRange(0, a);
        if (b > 0)
            intrinsicRight = intrinsicRight + deltas.rowRange(a, a + b);
        omcur = omcur + Vec3d(deltas.rowRange(a + b, a + b + 3));
        Tcur = Tcur + Vec3d(deltas.rowRange(a + b + 3, a + b + 6));
        for (int image_idx = 0; image_idx < n_images; ++image_idx)
        {
            rvecs1[image_idx] = Mat(Mat(rvecs1[image_idx]) + deltas.rowRange(a + b + 6 + image_idx * 6, a + b + 9 + image_idx * 6));
            tvecs1[image_idx] = Mat(Mat(tvecs1[image_idx]) + deltas.rowRange(a + b + 9 + image_idx * 6, a + b + 12 + image_idx * 6));
        }

        Vec6d newTom(Tcur[0], Tcur[1], Tcur[2], omcur[0], omcur[1], omcur[2]);
        change = cv::norm(newTom - oldTom) / cv::norm(newTom);
    }

    double rms = 0;
    const Vec2d* ptr_e = e.ptr<Vec2d>();
    for (size_t i = 0; i < e.total() / 2; i++)
    {
        rms += ptr_e[i][0] * ptr_e[i][0] + ptr_e[i][1] * ptr_e[i][1];
    }

    rms /= ((double)e.total() / 2.0);
    rms = sqrt(rms);

    _K1 = Matx33d(intrinsicLeft.f[0], intrinsicLeft.f[0] * intrinsicLeft.alpha, intrinsicLeft.c[0],
                                   0,                       intrinsicLeft.f[1], intrinsicLeft.c[1],
                                   0,                                        0,                 1);

    _K2 = Matx33d(intrinsicRight.f[0], intrinsicRight.f[0] * intrinsicRight.alpha, intrinsicRight.c[0],
                                    0,                        intrinsicRight.f[1], intrinsicRight.c[1],
                                    0,                                          0,                  1);

    Mat _R;
    Rodrigues(omcur, _R);

    if (K1.needed()) Mat(_K1).convertTo(K1, K1.empty() ? CV_64FC1 : K1.type());
    if (K2.needed()) Mat(_K2).convertTo(K2, K2.empty() ? CV_64FC1 : K2.type());
    if (D1.needed()) Mat(intrinsicLeft.k).convertTo(D1, D1.empty() ? CV_64FC1 : D1.type());
    if (D2.needed()) Mat(intrinsicRight.k).convertTo(D2, D2.empty() ? CV_64FC1 : D2.type());
    if (R.needed()) _R.convertTo(R, R.empty() ? CV_64FC1 : R.type());
    if (T.needed()) Mat(Tcur).convertTo(T, T.empty() ? CV_64FC1 : T.type());
    if (rvecs.isMatVector())
    {
        if(rvecs.empty())
            rvecs.create(n_images, 1, CV_64FC3);

        if(tvecs.empty())
            tvecs.create(n_images, 1, CV_64FC3);

        for(int i = 0; i < n_images; i++ )
        {
            rvecs.create(3, 1, CV_64F, i, true);
            tvecs.create(3, 1, CV_64F, i, true);
            rvecs1[i].copyTo(rvecs.getMat(i));
            tvecs1[i].copyTo(tvecs.getMat(i));
        }
    }
    else
    {
        if (rvecs.needed()) cv::Mat(rvecs1).convertTo(rvecs, rvecs.empty() ? CV_64FC3 : rvecs.type());
        if (tvecs.needed()) cv::Mat(tvecs1).convertTo(tvecs, tvecs.empty() ? CV_64FC3 : tvecs.type());
    }

    return rms;
}

namespace cv{ namespace {
void subMatrix(const Mat& src, Mat& dst, const std::vector<uchar>& cols, const std::vector<uchar>& rows)
{
    CV_Assert(src.channels() == 1);

    int nonzeros_cols = cv::countNonZero(cols);
    Mat tmp(src.rows, nonzeros_cols, CV_64F);

    for (int i = 0, j = 0; i < (int)cols.size(); i++)
    {
        if (cols[i])
        {
            src.col(i).copyTo(tmp.col(j++));
        }
    }

    int nonzeros_rows  = cv::countNonZero(rows);
    dst.create(nonzeros_rows, nonzeros_cols, CV_64F);
    for (int i = 0, j = 0; i < (int)rows.size(); i++)
    {
        if (rows[i])
        {
            tmp.row(i).copyTo(dst.row(j++));
        }
    }
}

}}

cv::internal::IntrinsicParams::IntrinsicParams():
    f(Vec2d::all(0)), c(Vec2d::all(0)), k(Vec4d::all(0)), alpha(0), isEstimate(9,0)
{
}

cv::internal::IntrinsicParams::IntrinsicParams(Vec2d _f, Vec2d _c, Vec4d _k, double _alpha):
    f(_f), c(_c), k(_k), alpha(_alpha), isEstimate(9,0)
{
}

cv::internal::IntrinsicParams cv::internal::IntrinsicParams::operator+(const Mat& a)
{
    CV_Assert(a.type() == CV_64FC1);
    IntrinsicParams tmp;
    const double* ptr = a.ptr<double>();

    int j = 0;
    tmp.f[0]    = this->f[0]    + (isEstimate[0] ? ptr[j++] : 0);
    tmp.f[1]    = this->f[1]    + (isEstimate[1] ? ptr[j++] : 0);
    tmp.c[0]    = this->c[0]    + (isEstimate[2] ? ptr[j++] : 0);
    tmp.c[1]    = this->c[1]    + (isEstimate[3] ? ptr[j++] : 0);
    tmp.alpha   = this->alpha   + (isEstimate[4] ? ptr[j++] : 0);
    tmp.k[0]    = this->k[0]    + (isEstimate[5] ? ptr[j++] : 0);
    tmp.k[1]    = this->k[1]    + (isEstimate[6] ? ptr[j++] : 0);
    tmp.k[2]    = this->k[2]    + (isEstimate[7] ? ptr[j++] : 0);
    tmp.k[3]    = this->k[3]    + (isEstimate[8] ? ptr[j++] : 0);

    tmp.isEstimate = isEstimate;
    return tmp;
}

cv::internal::IntrinsicParams& cv::internal::IntrinsicParams::operator =(const Mat& a)
{
    CV_Assert(a.type() == CV_64FC1);
    const double* ptr = a.ptr<double>();

    int j = 0;

    this->f[0]  = isEstimate[0] ? ptr[j++] : 0;
    this->f[1]  = isEstimate[1] ? ptr[j++] : 0;
    this->c[0]  = isEstimate[2] ? ptr[j++] : 0;
    this->c[1]  = isEstimate[3] ? ptr[j++] : 0;
    this->alpha = isEstimate[4] ? ptr[j++] : 0;
    this->k[0]  = isEstimate[5] ? ptr[j++] : 0;
    this->k[1]  = isEstimate[6] ? ptr[j++] : 0;
    this->k[2]  = isEstimate[7] ? ptr[j++] : 0;
    this->k[3]  = isEstimate[8] ? ptr[j++] : 0;

    return *this;
}

void cv::internal::IntrinsicParams::Init(const Vec2d& _f, const Vec2d& _c, const Vec4d& _k, const double& _alpha)
{
    this->c = _c;
    this->f = _f;
    this->k = _k;
    this->alpha = _alpha;
}

void cv::internal::projectPoints(cv::InputArray objectPoints, cv::OutputArray imagePoints,
                   cv::InputArray _rvec,cv::InputArray _tvec,
                   const IntrinsicParams& param, cv::OutputArray jacobian)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
    Matx33d K(param.f[0], param.f[0] * param.alpha, param.c[0],
                       0,               param.f[1], param.c[1],
                       0,                        0,         1);
    fisheye::projectPoints(objectPoints, imagePoints, _rvec, _tvec, K, param.k, param.alpha, jacobian);
}

void cv::internal::ComputeExtrinsicRefine(const Mat& imagePoints, const Mat& objectPoints, Mat& rvec,
                            Mat&  tvec, Mat& J, const int MaxIter,
                            const IntrinsicParams& param, const double thresh_cond)
{
    CV_Assert(!objectPoints.empty() && objectPoints.type() == CV_64FC3);
    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);
    CV_Assert(rvec.total() > 2 && tvec.total() > 2);
    Vec6d extrinsics(rvec.at<double>(0), rvec.at<double>(1), rvec.at<double>(2),
                    tvec.at<double>(0), tvec.at<double>(1), tvec.at<double>(2));
    double change = 1;
    int iter = 0;

    while (change > 1e-10 && iter < MaxIter)
    {
        std::vector<Point2d> x;
        Mat jacobians;
        projectPoints(objectPoints, x, rvec, tvec, param, jacobians);

        Mat ex = imagePoints - Mat(x);
        ex = ex.reshape(1, 2);

        J = jacobians.colRange(8, 14).clone();

        SVD svd(J, SVD::NO_UV);
        double condJJ = svd.w.at<double>(0)/svd.w.at<double>(5);

        if (condJJ > thresh_cond)
            change = 0;
        else
        {
            Vec6d param_innov;
            solve(J, ex.reshape(1, (int)ex.total()), param_innov, DECOMP_SVD + DECOMP_NORMAL);

            Vec6d param_up = extrinsics + param_innov;
            change = norm(param_innov)/norm(param_up);
            extrinsics = param_up;
            iter = iter + 1;

            rvec = Mat(Vec3d(extrinsics.val));
            tvec = Mat(Vec3d(extrinsics.val+3));
        }
    }
}

cv::Mat cv::internal::ComputeHomography(Mat m, Mat M)
{
    CV_INSTRUMENT_REGION();

    int Np = m.cols;

    if (m.rows < 3)
    {
        vconcat(m, Mat::ones(1, Np, CV_64FC1), m);
    }
    if (M.rows < 3)
    {
        vconcat(M, Mat::ones(1, Np, CV_64FC1), M);
    }

    divide(m, Mat::ones(3, 1, CV_64FC1) * m.row(2), m);
    divide(M, Mat::ones(3, 1, CV_64FC1) * M.row(2), M);

    Mat ax = m.row(0).clone();
    Mat ay = m.row(1).clone();

    double mxx = mean(ax)[0];
    double myy = mean(ay)[0];

    ax = ax - mxx;
    ay = ay - myy;

    double scxx = mean(abs(ax))[0];
    double scyy = mean(abs(ay))[0];

    Mat Hnorm (Matx33d( 1/scxx,        0.0,     -mxx/scxx,
                         0.0,     1/scyy,     -myy/scyy,
                         0.0,        0.0,           1.0 ));

    Mat inv_Hnorm (Matx33d( scxx,     0,   mxx,
                                    0,  scyy,   myy,
                                    0,     0,     1 ));
    Mat mn =  Hnorm * m;

    Mat L = Mat::zeros(2*Np, 9, CV_64FC1);

    for (int i = 0; i < Np; ++i)
    {
        for (int j = 0; j < 3; j++)
        {
            L.at<double>(2 * i, j) = M.at<double>(j, i);
            L.at<double>(2 * i + 1, j + 3) = M.at<double>(j, i);
            L.at<double>(2 * i, j + 6) = -mn.at<double>(0,i) * M.at<double>(j, i);
            L.at<double>(2 * i + 1, j + 6) = -mn.at<double>(1,i) * M.at<double>(j, i);
        }
    }

    if (Np > 4) L = L.t() * L;
    SVD svd(L);
    Mat hh = svd.vt.row(8) / svd.vt.row(8).at<double>(8);
    Mat Hrem = hh.reshape(1, 3);
    Mat H = inv_Hnorm * Hrem;

    if (Np > 4)
    {
        Mat hhv = H.reshape(1, 9)(Rect(0, 0, 1, 8)).clone();
        for (int iter = 0; iter < 10; iter++)
        {
            Mat mrep = H * M;
            Mat J = Mat::zeros(2 * Np, 8, CV_64FC1);
            Mat MMM;
            divide(M, Mat::ones(3, 1, CV_64FC1) * mrep(Rect(0, 2, mrep.cols, 1)), MMM);
            divide(mrep, Mat::ones(3, 1, CV_64FC1) * mrep(Rect(0, 2, mrep.cols, 1)), mrep);
            Mat m_err = m(Rect(0,0, m.cols, 2)) - mrep(Rect(0,0, mrep.cols, 2));
            m_err = Mat(m_err.t()).reshape(1, m_err.cols * m_err.rows);
            Mat MMM2, MMM3;
            multiply(Mat::ones(3, 1, CV_64FC1) * mrep(Rect(0, 0, mrep.cols, 1)), MMM, MMM2);
            multiply(Mat::ones(3, 1, CV_64FC1) * mrep(Rect(0, 1, mrep.cols, 1)), MMM, MMM3);

            for (int i = 0; i < Np; ++i)
            {
                for (int j = 0; j < 3; ++j)
                {
                    J.at<double>(2 * i, j)         = -MMM.at<double>(j, i);
                    J.at<double>(2 * i + 1, j + 3) = -MMM.at<double>(j, i);
                }

                for (int j = 0; j < 2; ++j)
                {
                    J.at<double>(2 * i, j + 6)     = MMM2.at<double>(j, i);
                    J.at<double>(2 * i + 1, j + 6) = MMM3.at<double>(j, i);
                }
            }
            divide(M, Mat::ones(3, 1, CV_64FC1) * mrep(Rect(0,2,mrep.cols,1)), MMM);
            Mat hh_innov = (J.t() * J).inv() * (J.t()) * m_err;
            Mat hhv_up = hhv - hh_innov;
            Mat tmp;
            vconcat(hhv_up, Mat::ones(1,1,CV_64FC1), tmp);
            Mat H_up = tmp.reshape(1,3);
            hhv = hhv_up;
            H = H_up;
        }
    }
    return H;
}

cv::Mat cv::internal::NormalizePixels(const Mat& imagePoints, const IntrinsicParams& param)
{
    CV_INSTRUMENT_REGION();

    CV_Assert(!imagePoints.empty() && imagePoints.type() == CV_64FC2);

    Mat distorted((int)imagePoints.total(), 1, CV_64FC2), undistorted;
    const Vec2d* ptr   = imagePoints.ptr<Vec2d>();
    Vec2d* ptr_d = distorted.ptr<Vec2d>();
    for (size_t i = 0; i < imagePoints.total(); ++i)
    {
        ptr_d[i] = (ptr[i] - param.c).mul(Vec2d(1.0 / param.f[0], 1.0 / param.f[1]));
        ptr_d[i][0] -= param.alpha * ptr_d[i][1];
    }
    cv::fisheye::undistortPoints(distorted, undistorted, Matx33d::eye(), param.k);
    return undistorted;
}

void cv::internal::InitExtrinsics(const Mat& _imagePoints, const Mat& _objectPoints, const IntrinsicParams& param, Mat& omckk, Mat& Tckk)
{
    CV_Assert(!_objectPoints.empty() && _objectPoints.type() == CV_64FC3);
    CV_Assert(!_imagePoints.empty() && _imagePoints.type() == CV_64FC2);

    Mat imagePointsNormalized = NormalizePixels(_imagePoints, param).reshape(1).t();
    Mat objectPoints = _objectPoints.reshape(1).t();
    Mat objectPointsMean, covObjectPoints;
    Mat Rckk;
    int Np = imagePointsNormalized.cols;
    calcCovarMatrix(objectPoints, covObjectPoints, objectPointsMean, COVAR_NORMAL | COVAR_COLS);
    SVD svd(covObjectPoints);
    Mat R(svd.vt);
    if (norm(R(Rect(2, 0, 1, 2))) < 1e-6)
        R = Mat::eye(3,3, CV_64FC1);
    if (determinant(R) < 0)
        R = -R;
    Mat T = -R * objectPointsMean;
    Mat X_new = R * objectPoints + T * Mat::ones(1, Np, CV_64FC1);
    Mat H = ComputeHomography(imagePointsNormalized, X_new(Rect(0,0,X_new.cols,2)));
    double sc = .5 * (norm(H.col(0)) + norm(H.col(1)));
    H = H / sc;
    Mat u1 = H.col(0).clone();
    double norm_u1 = norm(u1);
    CV_Assert(fabs(norm_u1) > 0);
    u1  = u1 / norm_u1;
    Mat u2 = H.col(1).clone() - u1.dot(H.col(1).clone()) * u1;
    double norm_u2 = norm(u2);
    CV_Assert(fabs(norm_u2) > 0);
    u2 = u2 / norm_u2;
    Mat u3 = u1.cross(u2);
    Mat RRR;
    hconcat(u1, u2, RRR);
    hconcat(RRR, u3, RRR);
    Rodrigues(RRR, omckk);
    Rodrigues(omckk, Rckk);
    Tckk = H.col(2).clone();
    Tckk = Tckk + Rckk * T;
    Rckk = Rckk * R;
    Rodrigues(Rckk, omckk);
}

void cv::internal::CalibrateExtrinsics(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                         const IntrinsicParams& param, const int check_cond,
                         const double thresh_cond, InputOutputArray omc, InputOutputArray Tc)
{
    CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
    CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));
    CV_Assert(omc.type() == CV_64FC3 || Tc.type() == CV_64FC3);

    if (omc.empty()) omc.create(1, (int)objectPoints.total(), CV_64FC3);
    if (Tc.empty()) Tc.create(1, (int)objectPoints.total(), CV_64FC3);

    const int maxIter = 20;

    for(int image_idx = 0; image_idx < (int)imagePoints.total(); ++image_idx)
    {
        Mat omckk, Tckk, JJ_kk;
        Mat image, object;

        objectPoints.getMat(image_idx).convertTo(object,  CV_64FC3);
        imagePoints.getMat (image_idx).convertTo(image, CV_64FC2);

        bool imT = image.rows < image.cols;
        bool obT = object.rows < object.cols;

        InitExtrinsics(imT ? image.t() : image, obT ? object.t() : object, param, omckk, Tckk);

        ComputeExtrinsicRefine(!imT ? image.t() : image, !obT ? object.t() : object, omckk, Tckk, JJ_kk, maxIter, param, thresh_cond);
        if (check_cond)
        {
            SVD svd(JJ_kk, SVD::NO_UV);
            if(svd.w.at<double>(0) / svd.w.at<double>((int)svd.w.total() - 1) > thresh_cond )
                CV_Error( cv::Error::StsInternal, format("CALIB_CHECK_COND - Ill-conditioned matrix for input array %d",image_idx));
        }
        omckk.reshape(3,1).copyTo(omc.getMat().col(image_idx));
        Tckk.reshape(3,1).copyTo(Tc.getMat().col(image_idx));
    }
}

void cv::internal::ComputeJacobians(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                      const IntrinsicParams& param,  InputArray omc, InputArray Tc,
                      const int& check_cond, const double& thresh_cond, Mat& JJ2, Mat& ex3)
{
    CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
    CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));

    CV_Assert(!omc.empty() && omc.type() == CV_64FC3);
    CV_Assert(!Tc.empty() && Tc.type() == CV_64FC3);

    int n = (int)objectPoints.total();

    JJ2 = Mat::zeros(9 + 6 * n, 9 + 6 * n, CV_64FC1);
    ex3 = Mat::zeros(9 + 6 * n, 1, CV_64FC1 );

    for (int image_idx = 0; image_idx < n; ++image_idx)
    {
        Mat image, object;
        objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
        imagePoints.getMat (image_idx).convertTo(image, CV_64FC2);

        bool imT = image.channels() == 1 && image.rows > image.cols;
        Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

        std::vector<Point2d> x;
        Mat jacobians;
        projectPoints(object, x, om, T, param, jacobians);
        Mat exkk = (imT ? image.t() : image) - Mat(x);

        Mat A(jacobians.rows, 9, CV_64FC1);
        jacobians.colRange(0, 4).copyTo(A.colRange(0, 4));
        jacobians.col(14).copyTo(A.col(4));
        jacobians.colRange(4, 8).copyTo(A.colRange(5, 9));

        A = A.t();

        Mat B = jacobians.colRange(8, 14).clone();
        B = B.t();

        JJ2(Rect(0, 0, 9, 9)) += A * A.t();
        JJ2(Rect(9 + 6 * image_idx, 9 + 6 * image_idx, 6, 6)) = B * B.t();

        JJ2(Rect(9 + 6 * image_idx, 0, 6, 9)) = A * B.t();
        JJ2(Rect(0, 9 + 6 * image_idx, 9, 6)) = JJ2(Rect(9 + 6 * image_idx, 0, 6, 9)).t();

        Mat exkk_col = exkk.reshape(1, 2 * (int)exkk.total());
        ex3.rowRange(0, 9) += A * exkk_col;
        ex3.rowRange(9 + 6 * image_idx, 9 + 6 * (image_idx + 1)) = B *  exkk_col;

        if (check_cond)
        {
            Mat JJ_kk = B.t();
            SVD svd(JJ_kk, SVD::NO_UV);
            CV_Assert(svd.w.at<double>(0) / svd.w.at<double>(svd.w.rows - 1) < thresh_cond);
        }
    }

    std::vector<uchar> idxs(param.isEstimate);
    idxs.insert(idxs.end(), 6 * n, 1);

    subMatrix(JJ2, JJ2, idxs, idxs);
    subMatrix(ex3, ex3, std::vector<uchar>(1, 1), idxs);
}

void cv::internal::EstimateUncertainties(InputArrayOfArrays objectPoints, InputArrayOfArrays imagePoints,
                           const IntrinsicParams& params, InputArray omc, InputArray Tc,
                           IntrinsicParams& errors, Vec2d& std_err, double thresh_cond, int check_cond, double& rms)
{
    CV_Assert(!objectPoints.empty() && (objectPoints.type() == CV_32FC3 || objectPoints.type() == CV_64FC3));
    CV_Assert(!imagePoints.empty() && (imagePoints.type() == CV_32FC2 || imagePoints.type() == CV_64FC2));

    CV_Assert(!omc.empty() && omc.type() == CV_64FC3);
    CV_Assert(!Tc.empty() && Tc.type() == CV_64FC3);

    int total_ex = 0;
    for (int image_idx = 0; image_idx < (int)objectPoints.total(); ++image_idx)
    {
        total_ex += (int)objectPoints.getMat(image_idx).total();
    }
    Mat ex(total_ex, 1, CV_64FC2);
    int insert_idx = 0;
    for (int image_idx = 0; image_idx < (int)objectPoints.total(); ++image_idx)
    {
        Mat image, object;
        objectPoints.getMat(image_idx).convertTo(object, CV_64FC3);
        imagePoints.getMat (image_idx).convertTo(image, CV_64FC2);

        bool imT = image.channels() == 1 && image.rows > image.cols;

        Mat om(omc.getMat().col(image_idx)), T(Tc.getMat().col(image_idx));

        std::vector<Point2d> x;
        projectPoints(object, x, om, T, params, noArray());
        Mat ex_ = (imT ? image.t() : image) - Mat(x);
        ex_ = ex_.reshape(2, (int)ex_.total());
        ex_.copyTo(ex.rowRange(insert_idx, insert_idx + ex_.rows));
        insert_idx += ex_.rows;
    }

    meanStdDev(ex, noArray(), std_err);
    std_err *= sqrt((double)ex.total()/((double)ex.total() - 1.0));

    Vec<double, 1> sigma_x;
    meanStdDev(ex.reshape(1, 1), noArray(), sigma_x);

    Mat JJ2, ex3;
    ComputeJacobians(objectPoints, imagePoints, params, omc, Tc, check_cond, thresh_cond, JJ2, ex3);

    sqrt(JJ2.inv(), JJ2);

    int nParams = JJ2.rows;
    // an explanation of that denominator correction can be found here:
    // R. Hartley, A. Zisserman, Multiple View Geometry in Computer Vision, 2004, section 5.1.3, page 134
    // see the discussion for more details: https://github.com/opencv/opencv/pull/22992
    sigma_x  *= sqrt(2.0 * (double)ex.total()/(2.0 * (double)ex.total() - nParams));

    errors = 3 * sigma_x(0) * JJ2.diag();
    rms = sqrt(norm(ex, NORM_L2SQR)/ex.total());
}

void cv::internal::dAB(InputArray A, InputArray B, OutputArray dABdA, OutputArray dABdB)
{
    CV_Assert(A.getMat().cols == B.getMat().rows);
    CV_Assert(A.type() == CV_64FC1 && B.type() == CV_64FC1);

    int p = A.getMat().rows;
    int n = A.getMat().cols;
    int q = B.getMat().cols;

    dABdA.create(p * q, p * n, CV_64FC1);
    dABdB.create(p * q, q * n, CV_64FC1);

    dABdA.getMat() = Mat::zeros(p * q, p * n, CV_64FC1);
    dABdB.getMat() = Mat::zeros(p * q, q * n, CV_64FC1);

    for (int i = 0; i < q; ++i)
    {
        for (int j = 0; j < p; ++j)
        {
            int ij = j + i * p;
            for (int k = 0; k < n; ++k)
            {
                int kj = j + k * p;
                dABdA.getMat().at<double>(ij, kj) = B.getMat().at<double>(k, i);
            }
        }
    }

    for (int i = 0; i < q; ++i)
    {
        A.getMat().copyTo(dABdB.getMat().rowRange(i * p, i * p + p).colRange(i * n, i * n + n));
    }
}

void cv::internal::JRodriguesMatlab(const Mat& src, Mat& dst)
{
    Mat tmp(src.cols, src.rows, src.type());
    if (src.rows == 9)
    {
        Mat(src.row(0).t()).copyTo(tmp.col(0));
        Mat(src.row(1).t()).copyTo(tmp.col(3));
        Mat(src.row(2).t()).copyTo(tmp.col(6));
        Mat(src.row(3).t()).copyTo(tmp.col(1));
        Mat(src.row(4).t()).copyTo(tmp.col(4));
        Mat(src.row(5).t()).copyTo(tmp.col(7));
        Mat(src.row(6).t()).copyTo(tmp.col(2));
        Mat(src.row(7).t()).copyTo(tmp.col(5));
        Mat(src.row(8).t()).copyTo(tmp.col(8));
    }
    else
    {
        Mat(src.col(0).t()).copyTo(tmp.row(0));
        Mat(src.col(1).t()).copyTo(tmp.row(3));
        Mat(src.col(2).t()).copyTo(tmp.row(6));
        Mat(src.col(3).t()).copyTo(tmp.row(1));
        Mat(src.col(4).t()).copyTo(tmp.row(4));
        Mat(src.col(5).t()).copyTo(tmp.row(7));
        Mat(src.col(6).t()).copyTo(tmp.row(2));
        Mat(src.col(7).t()).copyTo(tmp.row(5));
        Mat(src.col(8).t()).copyTo(tmp.row(8));
    }
    dst = tmp.clone();
}

void cv::internal::compose_motion(InputArray _om1, InputArray _T1, InputArray _om2, InputArray _T2,
                    Mat& om3, Mat& T3, Mat& dom3dom1, Mat& dom3dT1, Mat& dom3dom2,
                    Mat& dom3dT2, Mat& dT3dom1, Mat& dT3dT1, Mat& dT3dom2, Mat& dT3dT2)
{
    Mat om1 = _om1.getMat();
    Mat om2 = _om2.getMat();
    Mat T1 = _T1.getMat().reshape(1, 3);
    Mat T2 = _T2.getMat().reshape(1, 3);

    //% Rotations:
    Mat R1, R2, R3, dR1dom1(9, 3, CV_64FC1), dR2dom2;
    Rodrigues(om1, R1, dR1dom1);
    Rodrigues(om2, R2, dR2dom2);
    JRodriguesMatlab(dR1dom1, dR1dom1);
    JRodriguesMatlab(dR2dom2, dR2dom2);
    R3 = R2 * R1;
    Mat dR3dR2, dR3dR1;
    dAB(R2, R1, dR3dR2, dR3dR1);
    Mat dom3dR3;
    Rodrigues(R3, om3, dom3dR3);
    JRodriguesMatlab(dom3dR3, dom3dR3);
    dom3dom1 = dom3dR3 * dR3dR1 * dR1dom1;
    dom3dom2 = dom3dR3 * dR3dR2 * dR2dom2;
    dom3dT1 = Mat::zeros(3, 3, CV_64FC1);
    dom3dT2 = Mat::zeros(3, 3, CV_64FC1);

    //% Translations:
    Mat T3t = R2 * T1;
    Mat dT3tdR2, dT3tdT1;
    dAB(R2, T1, dT3tdR2, dT3tdT1);
    Mat dT3tdom2 = dT3tdR2 * dR2dom2;
    T3 = T3t + T2;
    dT3dT1 = dT3tdT1;
    dT3dT2 = Mat::eye(3, 3, CV_64FC1);
    dT3dom2 = dT3tdom2;
    dT3dom1 = Mat::zeros(3, 3, CV_64FC1);
}

double cv::internal::median(const Mat& row)
{
    CV_Assert(row.type() == CV_64FC1);
    CV_Assert(!row.empty() && row.rows == 1);
    Mat tmp = row.clone();
    sort(tmp, tmp, 0);
    if ((int)tmp.total() % 2) return tmp.at<double>((int)tmp.total() / 2);
    else return 0.5 *(tmp.at<double>((int)tmp.total() / 2) + tmp.at<double>((int)tmp.total() / 2 - 1));
}

cv::Vec3d cv::internal::median3d(InputArray m)
{
    CV_Assert(m.depth() == CV_64F && m.getMat().rows == 1);
    Mat M = Mat(m.getMat().t()).reshape(1).t();
    return Vec3d(median(M.row(0)), median(M.row(1)), median(M.row(2)));
}
