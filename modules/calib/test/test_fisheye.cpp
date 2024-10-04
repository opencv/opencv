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

#include "test_precomp.hpp"
#include <opencv2/ts/cuda_test.hpp> // EXPECT_MAT_NEAR
#include "../src/fisheye.hpp"
#include "opencv2/videoio.hpp"

namespace opencv_test { namespace {

class fisheyeTest : public ::testing::Test {

protected:
    const static cv::Size imageSize;
    const static cv::Matx33d K;
    const static cv::Vec4d D;
    const static cv::Matx33d R;
    const static cv::Vec3d T;
    std::string datasets_repository_path;

    virtual void SetUp() {
        datasets_repository_path = combine(cvtest::TS::ptr()->get_data_path(), "cv/cameracalibration/fisheye");
    }

protected:
    std::string combine(const std::string& _item1, const std::string& _item2);
};

const cv::Size fisheyeTest::imageSize(1280, 800);

const cv::Matx33d fisheyeTest::K(558.478087865323,               0, 620.458515360843,
                              0, 560.506767351568, 381.939424848348,
                              0,               0,                1);

const cv::Vec4d fisheyeTest::D(-0.0014613319981768, -0.00329861110580401, 0.00605760088590183, -0.00374209380722371);


const cv::Matx33d fisheyeTest::R ( 9.9756700084424932e-01, 6.9698277640183867e-02, 1.4929569991321144e-03,
                            -6.9711825162322980e-02, 9.9748249845531767e-01, 1.2997180766418455e-02,
                            -5.8331736398316541e-04,-1.3069635393884985e-02, 9.9991441852366736e-01);

const cv::Vec3d fisheyeTest::T(-9.9217369356044638e-02, 3.1741831972356663e-03, 1.8551007952921010e-04);

std::string fisheyeTest::combine(const std::string& _item1, const std::string& _item2)
{
    std::string item1 = _item1, item2 = _item2;
    std::replace(item1.begin(), item1.end(), '\\', '/');
    std::replace(item2.begin(), item2.end(), '\\', '/');

    if (item1.empty())
        return item2;

    if (item2.empty())
        return item1;

    char last = item1[item1.size()-1];
    return item1 + (last != '/' ? "/" : "") + item2;
}

TEST_F(fisheyeTest, Calibration)
{
    const int n_images = 34;

    const cv::Matx33d goldK(558.4780870585967, 0, 620.4585053962692,
                            0, 560.5067667343917, 381.9394122875291,
                            0, 0, 1);
    const cv::Vec4d goldD(-0.00146136, -0.00329847, 0.00605742, -0.00374201);

    std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");
    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> imagePoints[i];
    fs_left.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;

    cv::Matx33d theK;
    cv::Vec4d theD;

    cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, theK, theD,
                           cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));

    EXPECT_MAT_NEAR(theK, goldK, 1e-8);
    EXPECT_MAT_NEAR(theD, goldD, 1e-8);
}

TEST_F(fisheyeTest, CalibrationWithFixedFocalLength)
{
    const int n_images = 34;

    std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    const std::string folder =combine(datasets_repository_path, "calib-3_stereo_from_JY");
    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> imagePoints[i];
    fs_left.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;
    flag |= cv::CALIB_FIX_FOCAL_LENGTH;
    flag |= cv::CALIB_USE_INTRINSIC_GUESS;

    cv::Matx33d theK = this->K;
    const cv::Matx33d newK(
        558.478088, 0.000000, 620.458461,
        0.000000, 560.506767, 381.939362,
        0.000000, 0.000000, 1.000000);

    cv::Vec4d theD;
    const cv::Vec4d newD(-0.001461, -0.003298, 0.006057, -0.003742);

    cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, theK, theD,
                           cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));

    // ensure that CALIB_FIX_FOCAL_LENGTH works and focal lenght has not changed
    EXPECT_EQ(theK(0,0), K(0,0));
    EXPECT_EQ(theK(1,1), K(1,1));

    EXPECT_MAT_NEAR(theK, newK, 1e-6);
    EXPECT_MAT_NEAR(theD, newD, 1e-6);
}

TEST_F(fisheyeTest, Homography)
{
    const int n_images = 1;

    std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");
    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> imagePoints[i];
    fs_left.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    cv::internal::IntrinsicParams param;
    param.Init(cv::Vec2d(cv::max(imageSize.width, imageSize.height) / CV_PI, cv::max(imageSize.width, imageSize.height) / CV_PI),
               cv::Vec2d(imageSize.width  / 2.0 - 0.5, imageSize.height / 2.0 - 0.5));

    cv::Mat _imagePoints (imagePoints[0]);
    cv::Mat _objectPoints(objectPoints[0]);

    cv::Mat imagePointsNormalized = NormalizePixels(_imagePoints, param).reshape(1).t();
    _objectPoints = _objectPoints.reshape(1, (int)_objectPoints.total()).t();
    cv::Mat objectPointsMean, covObjectPoints;

    int Np = imagePointsNormalized.cols;
    cv::calcCovarMatrix(_objectPoints, covObjectPoints, objectPointsMean, cv::COVAR_NORMAL | cv::COVAR_COLS);
    cv::SVD svd(covObjectPoints);
    cv::Mat theR(svd.vt);

    if (cv::norm(theR(cv::Rect(2, 0, 1, 2))) < 1e-6)
        theR = cv::Mat::eye(3,3, CV_64FC1);
    if (cv::determinant(theR) < 0)
        theR = -theR;

    cv::Mat theT = -theR * objectPointsMean;
    cv::Mat X_new = theR * _objectPoints + theT * cv::Mat::ones(1, Np, CV_64FC1);
    cv::Mat H = cv::internal::ComputeHomography(imagePointsNormalized, X_new.rowRange(0, 2));

    cv::Mat M = cv::Mat::ones(3, X_new.cols, CV_64FC1);
    X_new.rowRange(0, 2).copyTo(M.rowRange(0, 2));
    cv::Mat mrep = H * M;

    cv::divide(mrep, cv::Mat::ones(3,1, CV_64FC1) * mrep.row(2).clone(), mrep);

    cv::Mat merr = (mrep.rowRange(0, 2) - imagePointsNormalized).t();

    cv::Vec2d std_err;
    cv::meanStdDev(merr.reshape(2), cv::noArray(), std_err);
    std_err *= sqrt((double)merr.reshape(2).total() / (merr.reshape(2).total() - 1));

    cv::Vec2d correct_std_err(0.00516740156010384, 0.00644205331553901);
    EXPECT_MAT_NEAR(std_err, correct_std_err, 1e-12);
}

TEST_F(fisheyeTest, EstimateUncertainties)
{
    const int n_images = 34;

    std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    const std::string folder =combine(datasets_repository_path, "calib-3_stereo_from_JY");
    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> imagePoints[i];
    fs_left.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;

    cv::Matx33d theK;
    cv::Vec4d theD;
    std::vector<cv::Vec3d> rvec;
    std::vector<cv::Vec3d> tvec;

    cv::fisheye::calibrate(objectPoints, imagePoints, imageSize, theK, theD,
                           rvec, tvec, flag, cv::TermCriteria(3, 20, 1e-6));

    cv::internal::IntrinsicParams param, errors;
    cv::Vec2d err_std;
    double thresh_cond = 1e6;
    int check_cond = 1;
    param.Init(cv::Vec2d(theK(0,0), theK(1,1)), cv::Vec2d(theK(0,2), theK(1, 2)), theD);
    param.isEstimate = std::vector<uchar>(9, 1);
    param.isEstimate[4] = 0;

    errors.isEstimate = param.isEstimate;

    double rms;

    cv::internal::EstimateUncertainties(objectPoints, imagePoints, param,  rvec, tvec,
                                        errors, err_std, thresh_cond, check_cond, rms);

    EXPECT_MAT_NEAR(errors.f, cv::Vec2d(1.34250246865020720, 1.36037536429654530), 1e-6);
    EXPECT_MAT_NEAR(errors.c, cv::Vec2d(0.92070526160049848, 0.84383585812851514), 1e-6);
    EXPECT_MAT_NEAR(errors.k, cv::Vec4d(0.0053379581373996041, 0.017389792901700545, 0.022036256089491224, 0.0094714594258908952), 1e-7);
    EXPECT_MAT_NEAR(err_std, cv::Vec2d(0.187475975266883, 0.185678953263995), 1e-7);
    CV_Assert(fabs(rms - 0.263782587133546) < 1e-10);
    CV_Assert(errors.alpha == 0);
}

TEST_F(fisheyeTest, stereoCalibrate)
{
    const int n_images = 34;

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");

    std::vector<std::vector<cv::Point2d> > leftPoints(n_images);
    std::vector<std::vector<cv::Point2d> > rightPoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> leftPoints[i];
    fs_left.release();

    cv::FileStorage fs_right(combine(folder, "right.xml"), cv::FileStorage::READ);
    CV_Assert(fs_right.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_right[cv::format("image_%d", i )] >> rightPoints[i];
    fs_right.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    cv::Matx33d K1, K2, theR;
    cv::Vec3d theT;
    cv::Vec4d D1, D2;

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;

    cv::fisheye::stereoCalibrate(objectPoints, leftPoints, rightPoints,
                    K1, D1, K2, D2, imageSize, theR, theT, flag,
                    cv::TermCriteria(3, 12, 0));

    cv::Matx33d R_correct(   0.9975587205950972,   0.06953016383322372, 0.006492709911733523,
                           -0.06956823121068059,    0.9975601387249519, 0.005833595226966235,
                          -0.006071257768382089, -0.006271040135405457, 0.9999619062167968);
    cv::Vec3d T_correct(-0.099402724724121, 0.00270812139265413, 0.00129330292472699);
    cv::Matx33d K1_correct (561.195925927249,                0, 621.282400272412,
                                   0, 562.849402029712, 380.555455380889,
                                   0,                0,                1);

    cv::Matx33d K2_correct (560.395452535348,                0, 678.971652040359,
                                   0,  561.90171021422, 380.401340535339,
                                   0,                0,                1);

    cv::Vec4d D1_correct (-7.44253716539556e-05, -0.00702662033932424, 0.00737569823650885, -0.00342230256441771);
    cv::Vec4d D2_correct (-0.0130785435677431, 0.0284434505383497, -0.0360333869900506, 0.0144724062347222);

    EXPECT_MAT_NEAR(theR, R_correct, 1e-10);
    EXPECT_MAT_NEAR(theT, T_correct, 1e-10);

    EXPECT_MAT_NEAR(K1, K1_correct, 1e-10);
    EXPECT_MAT_NEAR(K2, K2_correct, 1e-10);

    EXPECT_MAT_NEAR(D1, D1_correct, 1e-10);
    EXPECT_MAT_NEAR(D2, D2_correct, 1e-10);

}

TEST_F(fisheyeTest, stereoCalibrateFixIntrinsic)
{
    const int n_images = 34;

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");

    std::vector<std::vector<cv::Point2d> > leftPoints(n_images);
    std::vector<std::vector<cv::Point2d> > rightPoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> leftPoints[i];
    fs_left.release();

    cv::FileStorage fs_right(combine(folder, "right.xml"), cv::FileStorage::READ);
    CV_Assert(fs_right.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_right[cv::format("image_%d", i )] >> rightPoints[i];
    fs_right.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    cv::Matx33d theR;
    cv::Vec3d theT;

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;
    flag |= cv::CALIB_FIX_INTRINSIC;

    cv::Matx33d K1 (561.195925927249,                0, 621.282400272412,
                                   0, 562.849402029712, 380.555455380889,
                                   0,                0,                1);

    cv::Matx33d K2 (560.395452535348,                0, 678.971652040359,
                                   0,  561.90171021422, 380.401340535339,
                                   0,                0,                1);

    cv::Vec4d D1 (-7.44253716539556e-05, -0.00702662033932424, 0.00737569823650885, -0.00342230256441771);
    cv::Vec4d D2 (-0.0130785435677431, 0.0284434505383497, -0.0360333869900506, 0.0144724062347222);

    cv::fisheye::stereoCalibrate(objectPoints, leftPoints, rightPoints,
                    K1, D1, K2, D2, imageSize, theR, theT, flag,
                    cv::TermCriteria(3, 12, 0));

    cv::Matx33d R_correct(   0.9975587205950972,   0.06953016383322372, 0.006492709911733523,
                           -0.06956823121068059,    0.9975601387249519, 0.005833595226966235,
                          -0.006071257768382089, -0.006271040135405457, 0.9999619062167968);
    cv::Vec3d T_correct(-0.099402724724121, 0.00270812139265413, 0.00129330292472699);


    EXPECT_MAT_NEAR(theR, R_correct, 1e-10);
    EXPECT_MAT_NEAR(theT, T_correct, 1e-10);
}

TEST_F(fisheyeTest, CalibrationWithDifferentPointsNumber)
{
    const int n_images = 2;

    std::vector<std::vector<cv::Point2d> > imagePoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    std::vector<cv::Point2d> imgPoints1(10);
    std::vector<cv::Point2d> imgPoints2(15);

    std::vector<cv::Point3d> objectPoints1(imgPoints1.size());
    std::vector<cv::Point3d> objectPoints2(imgPoints2.size());

    for (size_t i = 0; i < imgPoints1.size(); i++)
    {
        imgPoints1[i] = cv::Point2d((double)i, (double)i);
        objectPoints1[i] = cv::Point3d((double)i, (double)i, 10.0);
    }

    for (size_t i = 0; i < imgPoints2.size(); i++)
    {
        imgPoints2[i] = cv::Point2d(i + 0.5, i + 0.5);
        objectPoints2[i] = cv::Point3d(i + 0.5, i + 0.5, 10.0);
    }

    imagePoints[0] = imgPoints1;
    imagePoints[1] = imgPoints2;
    objectPoints[0] = objectPoints1;
    objectPoints[1] = objectPoints2;

    cv::Matx33d theK = cv::Matx33d::eye();
    cv::Vec4d theD;

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_USE_INTRINSIC_GUESS;
    flag |= cv::CALIB_FIX_SKEW;

    cv::fisheye::calibrate(objectPoints, imagePoints, cv::Size(100, 100), theK, theD,
        cv::noArray(), cv::noArray(), flag, cv::TermCriteria(3, 20, 1e-6));
}


TEST_F(fisheyeTest, stereoCalibrateWithPerViewTransformations)
{
    const int n_images = 34;

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");

    std::vector<std::vector<cv::Point2d> > leftPoints(n_images);
    std::vector<std::vector<cv::Point2d> > rightPoints(n_images);
    std::vector<std::vector<cv::Point3d> > objectPoints(n_images);

    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> leftPoints[i];
    fs_left.release();

    cv::FileStorage fs_right(combine(folder, "right.xml"), cv::FileStorage::READ);
    CV_Assert(fs_right.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_right[cv::format("image_%d", i )] >> rightPoints[i];
    fs_right.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    cv::Matx33d K1, K2, theR;
    cv::Vec3d theT;
    cv::Vec4d D1, D2;

    std::vector<cv::Mat> rvecs, tvecs;

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;

    double rmsErrorStereoCalib = cv::fisheye::stereoCalibrate(objectPoints, leftPoints, rightPoints,
                                                              K1, D1, K2, D2, imageSize, theR, theT, rvecs, tvecs, flag,
                                                              cv::TermCriteria(3, 12, 0));

    std::vector<cv::Point2d> reprojectedImgPts[2] = { std::vector<cv::Point2d>(n_images),
                                                      std::vector<cv::Point2d>(n_images) };
    size_t totalPoints = 0;
    double totalMSError[2] = { 0, 0 };
    for( size_t i = 0; i < n_images; i++ )
    {
        cv::Matx33d viewRotMat1, viewRotMat2;
        cv::Vec3d viewT1, viewT2;
        cv::Mat rVec;
        cv::Rodrigues( rvecs[i], rVec );
        rVec.convertTo(viewRotMat1, CV_64F);
        tvecs[i].convertTo(viewT1, CV_64F);

        viewRotMat2 = theR * viewRotMat1;
        cv::Vec3d T2t = theR * viewT1;
        viewT2 = T2t + theT;

        cv::Vec3d viewRotVec1, viewRotVec2;
        cv::Rodrigues(viewRotMat1, viewRotVec1);
        cv::Rodrigues(viewRotMat2, viewRotVec2);

        double alpha1 = K1(0, 1) / K1(0, 0);
        double alpha2 = K2(0, 1) / K2(0, 0);
        cv::fisheye::projectPoints(objectPoints[i], reprojectedImgPts[0], viewRotVec1, viewT1, K1, D1, alpha1);
        cv::fisheye::projectPoints(objectPoints[i], reprojectedImgPts[1], viewRotVec2, viewT2, K2, D2, alpha2);

        double viewMSError[2] = {
            cv::norm(leftPoints[i], reprojectedImgPts[0], cv::NORM_L2SQR),
            cv::norm(rightPoints[i], reprojectedImgPts[1], cv::NORM_L2SQR)
        };

        size_t n = objectPoints[i].size();
        totalMSError[0] += viewMSError[0];
        totalMSError[1] += viewMSError[1];
        totalPoints += n;
    }
    double rmsErrorFromReprojectedImgPts = std::sqrt((totalMSError[0] + totalMSError[1]) / (2 * totalPoints));

    cv::Matx33d R_correct(   0.9975587205950972,   0.06953016383322372, 0.006492709911733523,
                           -0.06956823121068059,    0.9975601387249519, 0.005833595226966235,
                          -0.006071257768382089, -0.006271040135405457, 0.9999619062167968);
    cv::Vec3d T_correct(-0.099402724724121, 0.00270812139265413, 0.00129330292472699);
    cv::Matx33d K1_correct (561.195925927249,                0, 621.282400272412,
                                           0, 562.849402029712, 380.555455380889,
                                           0,                0,                1);

    cv::Matx33d K2_correct (560.395452535348,               0, 678.971652040359,
                                           0, 561.90171021422, 380.401340535339,
                                           0,               0,                1);

    cv::Vec4d D1_correct (-7.44253716539556e-05, -0.00702662033932424, 0.00737569823650885, -0.00342230256441771);
    cv::Vec4d D2_correct (-0.0130785435677431, 0.0284434505383497, -0.0360333869900506, 0.0144724062347222);

    EXPECT_MAT_NEAR(theR, R_correct, 1e-10);
    EXPECT_MAT_NEAR(theT, T_correct, 1e-10);

    EXPECT_MAT_NEAR(K1, K1_correct, 1e-10);
    EXPECT_MAT_NEAR(K2, K2_correct, 1e-10);

    EXPECT_MAT_NEAR(D1, D1_correct, 1e-10);
    EXPECT_MAT_NEAR(D2, D2_correct, 1e-10);

    EXPECT_NEAR(rmsErrorStereoCalib, rmsErrorFromReprojectedImgPts, 1e-4);
}

TEST_F(fisheyeTest, multiview_calibration)
{
    const int n_images = 34;

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");

    std::vector<std::vector<cv::Point2f> > leftPoints(n_images);
    std::vector<std::vector<cv::Point2f> > rightPoints(n_images);
    std::vector<std::vector<cv::Point3f> > objectPoints(n_images);

    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> leftPoints[i];
    fs_left.release();

    cv::FileStorage fs_right(combine(folder, "right.xml"), cv::FileStorage::READ);
    CV_Assert(fs_right.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_right[cv::format("image_%d", i )] >> rightPoints[i];
    fs_right.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    std::vector<std::vector<cv::Mat>> image_points_all(2, std::vector<cv::Mat>(leftPoints.size()));
    for (int i = 0; i < (int)leftPoints.size(); i++) {
        cv::Mat left_pts(leftPoints[i], false) , right_pts(rightPoints[i], false);
        left_pts.copyTo(image_points_all[0][i]);
        right_pts.copyTo(image_points_all[1][i]);
    }
    std::vector<cv::Size> image_sizes(2, imageSize);
    cv::Mat visibility_mat = cv::Mat_<uchar>::ones(2, (int)leftPoints.size());
    std::vector<cv::Mat> Rs, Ts, Ks, distortions;
    std::vector<uchar> models(2, cv::CALIB_MODEL_FISHEYE);
    std::vector<int> all_flags(2, cv::CALIB_RECOMPUTE_EXTRINSIC | cv::CALIB_CHECK_COND | cv::CALIB_FIX_SKEW);

    calibrateMultiview(objectPoints, image_points_all, image_sizes, visibility_mat,
                       models, Ks, distortions, Rs, Ts, all_flags);
    cv::Matx33d R_correct(   0.9975587205950972,   0.06953016383322372, 0.006492709911733523,
                           -0.06956823121068059,    0.9975601387249519, 0.005833595226966235,
                          -0.006071257768382089, -0.006271040135405457, 0.9999619062167968);
    cv::Vec3d T_correct(-0.099402724724121, 0.00270812139265413, 0.00129330292472699);
    cv::Matx33d K1_correct (561.195925927249,                0, 621.282400272412,
                                   0, 562.849402029712, 380.555455380889,
                                   0,                0,                1);

    cv::Matx33d K2_correct (560.395452535348,                0, 678.971652040359,
                                   0,  561.90171021422, 380.401340535339,
                                   0,                0,                1);

    cv::Vec4d D1_correct (-7.44253716539556e-05, -0.00702662033932424, 0.00737569823650885, -0.00342230256441771);
    cv::Vec4d D2_correct (-0.0130785435677431, 0.0284434505383497, -0.0360333869900506, 0.0144724062347222);

    cv::Mat theR;
    cv::Rodrigues(Rs[1], theR);

    EXPECT_MAT_NEAR(theR, R_correct, 1e-2);
    EXPECT_MAT_NEAR(Ts[1], T_correct, 5e-3);

    EXPECT_MAT_NEAR(Ks[0], K1_correct, 4);
    EXPECT_MAT_NEAR(Ks[1], K2_correct, 5);

    EXPECT_MAT_NEAR(distortions[0], D1_correct, 1e-2);
    EXPECT_MAT_NEAR(distortions[1], D2_correct, 5e-2);
}

TEST_F(fisheyeTest, cameraRegistrationWithPerViewTransformations)
{
    const int n_images = 34;

    const std::string folder = combine(datasets_repository_path, "calib-3_stereo_from_JY");

    std::vector<std::vector<cv::Point2f> > leftPoints(n_images);
    std::vector<std::vector<cv::Point2f> > rightPoints(n_images);
    std::vector<std::vector<cv::Point3f> > objectPoints(n_images);

    cv::FileStorage fs_left(combine(folder, "left.xml"), cv::FileStorage::READ);
    CV_Assert(fs_left.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_left[cv::format("image_%d", i )] >> leftPoints[i];
    fs_left.release();

    cv::FileStorage fs_right(combine(folder, "right.xml"), cv::FileStorage::READ);
    CV_Assert(fs_right.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_right[cv::format("image_%d", i )] >> rightPoints[i];
    fs_right.release();

    cv::FileStorage fs_object(combine(folder, "object.xml"), cv::FileStorage::READ);
    CV_Assert(fs_object.isOpened());
    for(int i = 0; i < n_images; ++i)
        fs_object[cv::format("image_%d", i )] >> objectPoints[i];
    fs_object.release();

    cv::Matx33d K1, K2, theR;
    cv::Vec3d theT;
    cv::Vec4d D1, D2;

    int flag = 0;
    flag |= cv::CALIB_RECOMPUTE_EXTRINSIC;
    flag |= cv::CALIB_CHECK_COND;
    flag |= cv::CALIB_FIX_SKEW;

    cv::fisheye::stereoCalibrate(objectPoints, leftPoints, rightPoints,
                                 K1, D1, K2, D2, imageSize, theR, theT,flag, cv::TermCriteria(3, 12, 0));

    cv::Mat E, F, perViewErrors;
    std::vector<cv::Mat> rvecs, tvecs;
    flag = 0;
    double rmsErrorRegisterCamera = cv::registerCameras(objectPoints, objectPoints, leftPoints, rightPoints,
                                                        K1, D1, CALIB_MODEL_FISHEYE,
                                                        K2, D2, CALIB_MODEL_FISHEYE,
                                                        theR, theT, E, F, rvecs, tvecs, perViewErrors, flag,
                                                        cv::TermCriteria(3, 12, 0));
    std::vector<cv::Point2f> reprojectedImgPts[2] = { std::vector<cv::Point2f>(n_images),
                                                      std::vector<cv::Point2f>(n_images) };
    size_t totalPoints = 0;
    double totalMSError[2] = { 0, 0 };
    for( size_t i = 0; i < n_images; i++ )
    {
        cv::Matx33d viewRotMat1, viewRotMat2;
        cv::Vec3d viewT1, viewT2;
        cv::Mat rVec;
        cv::Rodrigues( rvecs[i], rVec );
        rVec.convertTo(viewRotMat1, CV_64F);
        tvecs[i].convertTo(viewT1, CV_64F);

        viewRotMat2 = theR * viewRotMat1;
        cv::Vec3d T2t = theR * viewT1;
        viewT2 = T2t + theT;

        cv::Vec3d viewRotVec1, viewRotVec2;
        cv::Rodrigues(viewRotMat1, viewRotVec1);
        cv::Rodrigues(viewRotMat2, viewRotVec2);

        double alpha1 = K1(0, 1) / K1(0, 0);
        double alpha2 = K2(0, 1) / K2(0, 0);
        cv::fisheye::projectPoints(objectPoints[i], reprojectedImgPts[0], viewRotVec1, viewT1, K1, D1, alpha1);
        cv::fisheye::projectPoints(objectPoints[i], reprojectedImgPts[1], viewRotVec2, viewT2, K2, D2, alpha2);

        double viewMSError[2] = {
            cv::norm(leftPoints[i], reprojectedImgPts[0], cv::NORM_L2SQR),
            cv::norm(rightPoints[i], reprojectedImgPts[1], cv::NORM_L2SQR)
        };

        size_t n = objectPoints[i].size();
        totalMSError[0] += viewMSError[0];
        totalMSError[1] += viewMSError[1];
        totalPoints += n;
    }

    double rmsErrorFromReprojectedImgPts = std::sqrt((totalMSError[0] + totalMSError[1]) / (2 * totalPoints));

    cv::Matx33d R_correct(   0.9975587205950972,   0.06953016383322372, 0.006492709911733523,
                           -0.06956823121068059,    0.9975601387249519, 0.005833595226966235,
                          -0.006071257768382089, -0.006271040135405457, 0.9999619062167968);
    cv::Vec3d T_correct(-0.099402724724121, 0.00270812139265413, 0.00129330292472699);
    cv::Matx33d K1_correct (561.195925927249,                0, 621.282400272412,
                                           0, 562.849402029712, 380.555455380889,
                                           0,                0,                1);

    cv::Matx33d K2_correct (560.395452535348,               0, 678.971652040359,
                                           0, 561.90171021422, 380.401340535339,
                                           0,               0,                1);

    cv::Vec4d D1_correct (-7.44253716539556e-05, -0.00702662033932424, 0.00737569823650885, -0.00342230256441771);
    cv::Vec4d D2_correct (-0.0130785435677431, 0.0284434505383497, -0.0360333869900506, 0.0144724062347222);

    EXPECT_MAT_NEAR(theR, R_correct, 1e-6);
    EXPECT_MAT_NEAR(theT, T_correct, 1e-6);

    EXPECT_MAT_NEAR(K1, K1_correct, 1e-4);
    EXPECT_MAT_NEAR(K2, K2_correct, 1e-4);

    EXPECT_MAT_NEAR(D1, D1_correct, 1e-5);
    EXPECT_MAT_NEAR(D2, D2_correct, 1e-5);

    EXPECT_NEAR(rmsErrorRegisterCamera, rmsErrorFromReprojectedImgPts, 1e-4);
}

}} // namespace
