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

#include "test_precomp.hpp"
#include "test_chessboardgenerator.hpp"
#include "opencv2/core/types_c.h"

namespace opencv_test { namespace {

class CV_CameraCalibrationBadArgTest : public cvtest::BadArgTest
{
public:
    CV_CameraCalibrationBadArgTest() {}
    ~CV_CameraCalibrationBadArgTest() {}
protected:
    void run(int);
    void run_func(void) {}

    struct C_Caller
    {
        _InputArray imgPts_arg;
        _InputArray objPts_arg;
        _OutputArray rvecs_arg;
        _OutputArray tvecs_arg;
        _OutputArray newObjPts_arg;
        _InputOutputArray cameraMatrix_arg;
        _InputOutputArray distCoeffs_arg;

        std::vector<std::vector<Point2f> > imgPts;
        std::vector<std::vector<Point3f> > objPts;

        Size imageSize0, imageSize;
        int iFixedPoint0, iFixedPoint;
        Mat cameraMatrix;
        Mat distCoeffs;
        std::vector<Mat> rvecs;
        std::vector<Mat> tvecs;
        std::vector<Point3f> newObjPts;
        int flags0, flags;

        void initArgs()
        {
            imgPts_arg = imgPts;
            objPts_arg = objPts;
            rvecs_arg = rvecs;
            tvecs_arg = tvecs;
            newObjPts_arg = newObjPts;
            cameraMatrix_arg = cameraMatrix;
            distCoeffs_arg = distCoeffs;
            imageSize = imageSize0;
            flags = flags0;
            iFixedPoint = iFixedPoint0;
        }

        void operator()() const
        {
            calibrateCameraRO(objPts_arg, imgPts_arg, imageSize, iFixedPoint,
                              cameraMatrix_arg, distCoeffs_arg, rvecs_arg, tvecs_arg,
                              newObjPts_arg, flags);
        }
    };
};


void CV_CameraCalibrationBadArgTest::run( int /* start_from */ )
{
    const int M = 2;
    Size imgSize(800, 600);
    Mat_<float> camMat(3, 3);
    Mat_<float> distCoeffs0(1, 5);

    camMat << 300.f, 0.f, imgSize.width/2.f, 0, 300.f, imgSize.height/2.f, 0.f, 0.f, 1.f;
    distCoeffs0 << 1.2f, 0.2f, 0.f, 0.f, 0.f;

    ChessBoardGenerator cbg(Size(8,6));
    Size corSize = cbg.cornersSize();
    vector<Point2f> corners;
    cbg(Mat(imgSize, CV_8U, Scalar(0)), camMat, distCoeffs0, corners);

    C_Caller caller;
    caller.imageSize0 = imgSize;
    caller.iFixedPoint0 = -1;
    caller.flags0 = 0;

    /////////////////////////////
    Mat cameraMatrix_cpp;
    Mat distCoeffs_cpp;
    Mat rvecs_cpp;
    Mat tvecs_cpp;
    Mat newObjPts_cpp;

    std::vector<Point3f> objPts_cpp;
    for(int y = 0; y < corSize.height; ++y)
        for(int x = 0; x < corSize.width; ++x)
            objPts_cpp.push_back(Point3f((float)x, (float)y, 0.f));
    caller.objPts.resize(M);
    caller.imgPts.resize(M);
    for(int i = 0; i < M; i++)
    {
        caller.objPts[i] = objPts_cpp;
        caller.imgPts[i] = corners;
    }
    caller.cameraMatrix.create(3, 3, CV_32F);
    caller.distCoeffs.create(5, 1, CV_32F);
    caller.rvecs.clear();
    caller.tvecs.clear();
    caller.newObjPts.clear();

    /* /*//*/ */
    int errors = 0;

    caller.initArgs();
    caller.objPts_arg = noArray();
    errors += run_test_case( CV_StsBadArg, "None passed in objPts", caller);

    caller.initArgs();
    caller.imgPts_arg = noArray();
    errors += run_test_case( CV_StsBadArg, "None passed in imgPts", caller );

    caller.initArgs();
    caller.cameraMatrix_arg = noArray();
    errors += run_test_case( CV_StsBadArg, "Zero passed in cameraMatrix", caller );

    caller.initArgs();
    caller.distCoeffs_arg = noArray();
    errors += run_test_case( CV_StsBadArg, "Zero passed in distCoeffs", caller );

    caller.initArgs();
    caller.imageSize.width = -1;
    errors += run_test_case( CV_StsOutOfRange, "Bad image width", caller );

    caller.initArgs();
    caller.imageSize.height = -1;
    errors += run_test_case( CV_StsOutOfRange, "Bad image height", caller );

    caller.initArgs();
    caller.imgPts[0].clear();
    errors += run_test_case( CV_StsUnsupportedFormat, "Bad imgpts[0]", caller );
    caller.imgPts[0] = caller.imgPts[1];

    caller.initArgs();
    caller.objPts[1].clear();
    errors += run_test_case( CV_StsUnsupportedFormat, "Bad objpts[1]", caller );
    caller.objPts[1] = caller.objPts[0];

    caller.initArgs();
    Mat badCM = Mat::zeros(4, 4, CV_64F);
    caller.cameraMatrix_arg = badCM;
    caller.flags = CALIB_USE_INTRINSIC_GUESS;
    errors += run_test_case( CV_StsBadArg, "Bad camearaMatrix header", caller );

    caller.initArgs();
    Mat badDC = Mat::zeros(10, 10, CV_64F);
    caller.distCoeffs_arg = badDC;
    caller.flags = CALIB_USE_INTRINSIC_GUESS;
    errors += run_test_case( CV_StsBadArg, "Bad camearaMatrix header", caller );

    if (errors)
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    else
        ts->set_failed_test_info(cvtest::TS::OK);
}


class CV_Rodrigues2BadArgTest : public cvtest::BadArgTest
{
public:
    CV_Rodrigues2BadArgTest() {}
    ~CV_Rodrigues2BadArgTest() {}
protected:
    void run_func(void) {}

    struct C_Caller
    {
        _InputArray src_arg;
        _OutputArray dst_arg, j_arg;

        Mat src;
        Mat dst;
        Mat jacobian;

        void initArgs()
        {
            src_arg = src;
            dst_arg = dst;
            j_arg = jacobian;
        }

        void operator()()
        {
            cv::Rodrigues(src_arg, dst_arg, j_arg);
        }
    };

    void run(int /* start_from */ )
    {
        Mat zeros(1, sizeof(CvMat), CV_8U, Scalar(0));

        Mat src_cpp(3, 1, CV_32F);
        Mat dst_cpp(3, 3, CV_32F);

        C_Caller caller;

        /*/*//*/*/
        int errors = 0;

        caller.initArgs();
        caller.src_arg = noArray();
        errors += run_test_case( CV_StsBadArg, "Src is empty matrix", caller );

        caller.initArgs();
        caller.src = Mat::zeros(3, 1, CV_8U);
        errors += run_test_case( CV_StsUnsupportedFormat, "Bad src formart", caller );

        caller.initArgs();
        caller.src = Mat::zeros(1, 1, CV_32F);
        errors += run_test_case( CV_StsBadSize, "Bad src size", caller );

        if (errors)
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        else
            ts->set_failed_test_info(cvtest::TS::OK);
    }
};


//////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////
class CV_ProjectPoints2BadArgTest : public cvtest::BadArgTest
{
public:
    CV_ProjectPoints2BadArgTest() : camMat(3, 3), distCoeffs(1, 5)
    {
        Size imsSize(800, 600);
        camMat << 300.f, 0.f, imsSize.width/2.f, 0, 300.f, imsSize.height/2.f, 0.f, 0.f, 1.f;
        distCoeffs << 1.2f, 0.2f, 0.f, 0.f, 0.f;
    }
    ~CV_ProjectPoints2BadArgTest() {}
protected:
    void run_func(void) {}

    Mat_<float> camMat;
    Mat_<float> distCoeffs;

    struct C_Caller
    {
        _InputArray objectPoints_arg, rvec_arg, tvec_arg, A_arg, DC_arg;
        _OutputArray imagePoints_arg;
        Mat objectPoints;
        Mat r_vec;
        Mat t_vec;
        Mat A;
        Mat distCoeffs;
        Mat imagePoints;
        Mat J;
        double aspectRatio0, aspectRatio;

        void initArgs()
        {
            objectPoints_arg = objectPoints;
            imagePoints_arg = imagePoints;
            rvec_arg = r_vec;
            tvec_arg = t_vec;
            A_arg = A;
            DC_arg = distCoeffs;
            aspectRatio = aspectRatio0;
        }

        void operator()()
        {
            projectPoints(objectPoints_arg, rvec_arg, tvec_arg, A_arg, DC_arg,
                          imagePoints_arg, J, aspectRatio );
        }
    };

    void run(int /* start_from */ )
    {
        C_Caller caller;

        const int n = 10;

        Mat objectPoints_cpp(1, n, CV_32FC3);
        randu(objectPoints_cpp, Scalar::all(1), Scalar::all(10));
        caller.objectPoints = objectPoints_cpp;
        caller.t_vec = Mat::zeros(1, 3, CV_32F);
        cvtest::Rodrigues(Mat::eye(3, 3, CV_32F), caller.r_vec);
        caller.A = Mat::eye(3, 3, CV_32F);
        caller.distCoeffs = Mat::zeros(1, 5, CV_32F);
        caller.aspectRatio0 = 1.0;

        /********************/
        int errors = 0;

        caller.initArgs();
        caller.objectPoints_arg = noArray();
        errors += run_test_case( CV_StsBadArg, "Zero objectPoints", caller );

        caller.initArgs();
        caller.rvec_arg = noArray();
        errors += run_test_case( CV_StsBadArg, "Zero r_vec", caller );

        caller.initArgs();
        caller.tvec_arg = noArray();
        errors += run_test_case( CV_StsBadArg, "Zero t_vec", caller );

        caller.initArgs();
        caller.A_arg = noArray();
        errors += run_test_case( CV_StsBadArg, "Zero camMat", caller );

        caller.initArgs();
        caller.imagePoints_arg = noArray();
        errors += run_test_case( CV_StsBadArg, "Zero imagePoints", caller );

        Mat save_rvec = caller.r_vec;
        caller.initArgs();
        caller.r_vec.create(2, 2, CV_32F);
        errors += run_test_case( CV_StsBadArg, "Bad rvec format", caller );

        caller.initArgs();
        caller.r_vec.create(1, 3, CV_8U);
        errors += run_test_case( CV_StsBadArg, "Bad rvec format", caller );
        caller.r_vec = save_rvec;

        /****************************/
        Mat save_tvec = caller.t_vec;
        caller.initArgs();
        caller.t_vec.create(3, 3, CV_32F);
        errors += run_test_case( CV_StsBadArg, "Bad tvec format", caller );

        caller.initArgs();
        caller.t_vec.create(1, 3, CV_8U);
        errors += run_test_case( CV_StsBadArg, "Bad tvec format", caller );
        caller.t_vec = save_tvec;

        /****************************/
        Mat save_A = caller.A;
        caller.initArgs();
        caller.A.create(2, 2, CV_32F);
        errors += run_test_case( CV_StsBadArg, "Bad A format", caller );
        caller.A = save_A;

        /****************************/
        Mat save_DC = caller.distCoeffs;
        caller.initArgs();
        caller.distCoeffs.create(3, 3, CV_32F);
        errors += run_test_case( CV_StsBadArg, "Bad distCoeffs format", caller );
        caller.distCoeffs = save_DC;

        if (errors)
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        else
            ts->set_failed_test_info(cvtest::TS::OK);
    }
};


TEST(Calib3d_CalibrateCamera_CPP, badarg) { CV_CameraCalibrationBadArgTest test; test.safe_run(); }
TEST(Calib3d_Rodrigues_CPP, badarg) { CV_Rodrigues2BadArgTest test; test.safe_run(); }
TEST(Calib3d_ProjectPoints_CPP, badarg) { CV_ProjectPoints2BadArgTest test; test.safe_run(); }

}} // namespace
