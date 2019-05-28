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
#include "opencv2/core/core_c.h"

namespace opencv_test { namespace {

class CV_UndistortPointsBadArgTest : public cvtest::BadArgTest
{
public:
    CV_UndistortPointsBadArgTest();
protected:
    void run(int);
    void run_func();

private:
    //common
    cv::Size img_size;
    //static const int N_POINTS = 1;
    static const int N_POINTS2 = 2;

    //C++
    cv::Mat camera_mat;
    cv::Mat R;
    cv::Mat P;
    cv::Mat distortion_coeffs;
    cv::Mat src_points;
    std::vector<cv::Point2f> dst_points;

};

CV_UndistortPointsBadArgTest::CV_UndistortPointsBadArgTest ()
{
}

void CV_UndistortPointsBadArgTest::run_func()
{
    cv::undistortPoints(src_points,dst_points,camera_mat,distortion_coeffs,R,P);
}

void CV_UndistortPointsBadArgTest::run(int)
{
    //RNG& rng = ts->get_rng();
    int errcount = 0;
//initializing
    img_size.width = 800;
    img_size.height = 600;
    double cam[9] = {150.f, 0.f, img_size.width/2.f, 0, 300.f, img_size.height/2.f, 0.f, 0.f, 1.f};
    double dist[4] = {0.01,0.02,0.001,0.0005};
    double s_points[N_POINTS2] = {
        static_cast<double>(img_size.width) / 4.0,
        static_cast<double>(img_size.height) / 4.0,
    };
    double p[9] = {155.f, 0.f, img_size.width/2.f+img_size.width/50.f, 0, 310.f, img_size.height/2.f+img_size.height/50.f, 0.f, 0.f, 1.f};
    double r[9] = {1,0,0,0,1,0,0,0,1};

    CvMat _camera_mat_orig = cvMat(3,3,CV_64F,cam);
    CvMat _distortion_coeffs_orig = cvMat(1,4,CV_64F,dist);
    CvMat _P_orig = cvMat(3,3,CV_64F,p);
    CvMat _R_orig = cvMat(3,3,CV_64F,r);
    CvMat _src_points_orig = cvMat(1,4,CV_64FC2,s_points);

    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    P = cv::cvarrToMat(&_P_orig);
    R = cv::cvarrToMat(&_R_orig);
    src_points = cv::cvarrToMat(&_src_points_orig);

    src_points.create(2, 2, CV_32FC2);
    errcount += run_test_case( CV_StsAssert, "Invalid input data matrix size" );
    src_points = cv::cvarrToMat(&_src_points_orig);

    src_points.create(1, 4, CV_64FC2);
    errcount += run_test_case( CV_StsAssert, "Invalid input data matrix type" );
    src_points = cv::cvarrToMat(&_src_points_orig);

    src_points = cv::Mat();
    errcount += run_test_case( CV_StsAssert, "Input data matrix is not continuous" );
    src_points = cv::cvarrToMat(&_src_points_orig);

//------------
    ts->set_failed_test_info(errcount > 0 ? cvtest::TS::FAIL_BAD_ARG_CHECK : cvtest::TS::OK);
}


//=========
class CV_InitUndistortRectifyMapBadArgTest : public cvtest::BadArgTest
{
public:
    CV_InitUndistortRectifyMapBadArgTest();
protected:
    void run(int);
    void run_func();

private:
    cv::Size img_size;
    cv::Mat camera_mat;
    cv::Mat R;
    cv::Mat new_camera_mat;
    cv::Mat distortion_coeffs;
    cv::Mat mapx;
    cv::Mat mapy;
    int mat_type;

};

CV_InitUndistortRectifyMapBadArgTest::CV_InitUndistortRectifyMapBadArgTest ()
{
}

void CV_InitUndistortRectifyMapBadArgTest::run_func()
{
    cv::initUndistortRectifyMap(camera_mat,distortion_coeffs,R,new_camera_mat,img_size,mat_type,mapx,mapy);
}

void CV_InitUndistortRectifyMapBadArgTest::run(int)
{
    int errcount = 0;
//initializing
    img_size.width = 800;
    img_size.height = 600;
    double cam[9] = {150.f, 0.f, img_size.width/2.f, 0, 300.f, img_size.height/2.f, 0.f, 0.f, 1.f};
    double dist[4] = {0.01,0.02,0.001,0.0005};
    std::vector<float> arr_mapx(img_size.width*img_size.height);
    std::vector<float> arr_mapy(img_size.width*img_size.height);
    double arr_new_camera_mat[9] = {155.f, 0.f, img_size.width/2.f+img_size.width/50.f, 0, 310.f, img_size.height/2.f+img_size.height/50.f, 0.f, 0.f, 1.f};
    double r[9] = {1,0,0,0,1,0,0,0,1};

    CvMat _camera_mat_orig = cvMat(3,3,CV_64F,cam);
    CvMat _distortion_coeffs_orig = cvMat(1,4,CV_64F,dist);
    CvMat _new_camera_mat_orig = cvMat(3,3,CV_64F,arr_new_camera_mat);
    CvMat _R_orig = cvMat(3,3,CV_64F,r);
    CvMat _mapx_orig = cvMat(img_size.height,img_size.width,CV_32FC1,&arr_mapx[0]);
    CvMat _mapy_orig = cvMat(img_size.height,img_size.width,CV_32FC1,&arr_mapy[0]);
    int mat_type_orig = CV_32FC1;

    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    new_camera_mat = cv::cvarrToMat(&_new_camera_mat_orig);
    R = cv::cvarrToMat(&_R_orig);
    mapx = cv::cvarrToMat(&_mapx_orig);
    mapy = cv::cvarrToMat(&_mapy_orig);

    mat_type = CV_64F;
    errcount += run_test_case( CV_StsAssert, "Invalid map matrix type" );
    mat_type = mat_type_orig;

    camera_mat.create(3, 2, CV_32F);
    errcount += run_test_case( CV_StsAssert, "Invalid camera data matrix size" );
    camera_mat = cv::cvarrToMat(&_camera_mat_orig);

    R.create(4, 3, CV_32F);
    errcount += run_test_case( CV_StsAssert, "Invalid R data matrix size" );
    R = cv::cvarrToMat(&_R_orig);

    distortion_coeffs.create(6, 1, CV_32F);
    errcount += run_test_case( CV_StsAssert, "Invalid distortion coefficients data matrix size" );
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);

//------------
    ts->set_failed_test_info(errcount > 0 ? cvtest::TS::FAIL_BAD_ARG_CHECK : cvtest::TS::OK);
}


//=========
class CV_UndistortBadArgTest : public cvtest::BadArgTest
{
public:
    CV_UndistortBadArgTest();
protected:
    void run(int);
    void run_func();

private:
    //common
    cv::Size img_size;

    cv::Mat camera_mat;
    cv::Mat new_camera_mat;
    cv::Mat distortion_coeffs;
    cv::Mat src;
    cv::Mat dst;

};

CV_UndistortBadArgTest::CV_UndistortBadArgTest ()
{
}

void CV_UndistortBadArgTest::run_func()
{
    cv::undistort(src,dst,camera_mat,distortion_coeffs,new_camera_mat);
}

void CV_UndistortBadArgTest::run(int)
{
    int errcount = 0;
//initializing
    img_size.width = 800;
    img_size.height = 600;
    double cam[9] = {150.f, 0.f, img_size.width/2.f, 0, 300.f, img_size.height/2.f, 0.f, 0.f, 1.f};
    double dist[4] = {0.01,0.02,0.001,0.0005};
    std::vector<float> arr_src(img_size.width*img_size.height);
    std::vector<float> arr_dst(img_size.width*img_size.height);
    double arr_new_camera_mat[9] = {155.f, 0.f, img_size.width/2.f+img_size.width/50.f, 0, 310.f, img_size.height/2.f+img_size.height/50.f, 0.f, 0.f, 1.f};

    CvMat _camera_mat_orig = cvMat(3,3,CV_64F,cam);
    CvMat _distortion_coeffs_orig = cvMat(1,4,CV_64F,dist);
    CvMat _new_camera_mat_orig = cvMat(3,3,CV_64F,arr_new_camera_mat);
    CvMat _src_orig = cvMat(img_size.height,img_size.width,CV_32FC1,&arr_src[0]);
    CvMat _dst_orig = cvMat(img_size.height,img_size.width,CV_32FC1,&arr_dst[0]);

    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    new_camera_mat = cv::cvarrToMat(&_new_camera_mat_orig);
    src = cv::cvarrToMat(&_src_orig);
    dst = cv::cvarrToMat(&_dst_orig);

    camera_mat.create(5, 5, CV_64F);
    errcount += run_test_case( CV_StsAssert, "Invalid camera data matrix size" );

//------------
    ts->set_failed_test_info(errcount > 0 ? cvtest::TS::FAIL_BAD_ARG_CHECK : cvtest::TS::OK);
}

TEST(Calib3d_UndistortPoints, badarg) { CV_UndistortPointsBadArgTest test; test.safe_run(); }
TEST(Calib3d_InitUndistortRectifyMap, badarg) { CV_InitUndistortRectifyMapBadArgTest test; test.safe_run(); }
TEST(Calib3d_Undistort, badarg) { CV_UndistortBadArgTest test; test.safe_run(); }

}} // namespace
/* End of file. */
