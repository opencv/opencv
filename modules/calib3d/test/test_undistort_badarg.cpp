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
#include "opencv2/imgproc/imgproc_c.h"

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
    bool useCPlus;
    //static const int N_POINTS = 1;
    static const int N_POINTS2 = 2;

    //C
    CvMat* _camera_mat;
    CvMat* matR;
    CvMat* matP;
    CvMat* _distortion_coeffs;
    CvMat* _src_points;
    CvMat* _dst_points;


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
    useCPlus = false;
    _camera_mat = matR = matP = _distortion_coeffs = _src_points = _dst_points = NULL;
}

void CV_UndistortPointsBadArgTest::run_func()
{
    if (useCPlus)
    {
        cv::undistortPoints(src_points,dst_points,camera_mat,distortion_coeffs,R,P);
    }
    else
    {
        cvUndistortPoints(_src_points,_dst_points,_camera_mat,_distortion_coeffs,matR,matP);
    }
}

void CV_UndistortPointsBadArgTest::run(int)
{
    //RNG& rng = ts->get_rng();
    int errcount = 0;
    useCPlus = false;
//initializing
    img_size.width = 800;
    img_size.height = 600;
    double cam[9] = {150.f, 0.f, img_size.width/2.f, 0, 300.f, img_size.height/2.f, 0.f, 0.f, 1.f};
    double dist[4] = {0.01,0.02,0.001,0.0005};
    double s_points[N_POINTS2] = {
        static_cast<double>(img_size.width) / 4.0,
        static_cast<double>(img_size.height) / 4.0,
    };
    double d_points[N_POINTS2];
    double p[9] = {155.f, 0.f, img_size.width/2.f+img_size.width/50.f, 0, 310.f, img_size.height/2.f+img_size.height/50.f, 0.f, 0.f, 1.f};
    double r[9] = {1,0,0,0,1,0,0,0,1};

    CvMat _camera_mat_orig = cvMat(3,3,CV_64F,cam);
    CvMat _distortion_coeffs_orig = cvMat(1,4,CV_64F,dist);
    CvMat _P_orig = cvMat(3,3,CV_64F,p);
    CvMat _R_orig = cvMat(3,3,CV_64F,r);
    CvMat _src_points_orig = cvMat(1,4,CV_64FC2,s_points);
    CvMat _dst_points_orig = cvMat(1,4,CV_64FC2,d_points);

    _camera_mat = &_camera_mat_orig;
    _distortion_coeffs = &_distortion_coeffs_orig;
    matP = &_P_orig;
    matR = &_R_orig;
    _src_points = &_src_points_orig;
    _dst_points = &_dst_points_orig;

//tests
    CvMat* temp1;
    CvMat* temp;
    IplImage* temp_img = cvCreateImage(cvSize(img_size.width,img_size.height),8,3);

//-----------
    temp = (CvMat*)temp_img;
    _src_points = temp;
    errcount += run_test_case( CV_StsAssert, "Input data is not CvMat*" );
    _src_points = &_src_points_orig;

    temp = (CvMat*)temp_img;
    _dst_points = temp;
    errcount += run_test_case( CV_StsAssert, "Output data is not CvMat*" );
    _dst_points = &_dst_points_orig;

    temp = cvCreateMat(2,3,CV_64F);
    _src_points = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid input data matrix size" );
    _src_points = &_src_points_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(2,3,CV_64F);
    _dst_points = temp;
    errcount += run_test_case(CV_StsAssert, "Invalid output data matrix size" );
    _dst_points = &_dst_points_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(1,3,CV_64F);
    temp1 = cvCreateMat(4,1,CV_64F);
    _dst_points = temp;
    _src_points = temp1;
    errcount += run_test_case(CV_StsAssert, "Output and input data sizes mismatch" );
    _dst_points = &_dst_points_orig;
    _src_points = &_src_points_orig;
    cvReleaseMat(&temp);
    cvReleaseMat(&temp1);

    temp = cvCreateMat(1,3,CV_32S);
    _dst_points = temp;
    errcount += run_test_case(CV_StsAssert, "Invalid output data matrix type" );
    _dst_points = &_dst_points_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(1,3,CV_32S);
    _src_points = temp;
    errcount += run_test_case(CV_StsAssert, "Invalid input data matrix type" );
    _src_points = &_src_points_orig;
    cvReleaseMat(&temp);
//------------
    temp = cvCreateMat(2,3,CV_64F);
    _camera_mat = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid camera data matrix size" );
    _camera_mat = &_camera_mat_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(3,4,CV_64F);
    _camera_mat = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid camera data matrix size" );
    _camera_mat = &_camera_mat_orig;
    cvReleaseMat(&temp);

    temp = (CvMat*)temp_img;
    _camera_mat = temp;
    errcount += run_test_case( CV_StsAssert, "Camera data is not CvMat*" );
    _camera_mat = &_camera_mat_orig;
//----------

    temp = (CvMat*)temp_img;
    _distortion_coeffs = temp;
    errcount += run_test_case( CV_StsAssert, "Distortion coefficients data is not CvMat*" );
    _distortion_coeffs = &_distortion_coeffs_orig;

    temp = cvCreateMat(1,6,CV_64F);
    _distortion_coeffs = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid distortion coefficients data matrix size" );
    _distortion_coeffs = &_distortion_coeffs_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(3,3,CV_64F);
    _distortion_coeffs = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid distortion coefficients data matrix size" );
    _distortion_coeffs = &_distortion_coeffs_orig;
    cvReleaseMat(&temp);
//----------
    temp = (CvMat*)temp_img;
    matR = temp;
    errcount += run_test_case( CV_StsAssert, "R data is not CvMat*" );
    matR = &_R_orig;

    temp = cvCreateMat(4,3,CV_64F);
    matR = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid R data matrix size" );
    matR = &_R_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(3,2,CV_64F);
    matR = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid R data matrix size" );
    matR = &_R_orig;
    cvReleaseMat(&temp);

//-----------
    temp = (CvMat*)temp_img;
    matP = temp;
    errcount += run_test_case( CV_StsAssert, "P data is not CvMat*" );
    matP = &_P_orig;

    temp = cvCreateMat(4,3,CV_64F);
    matP = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid P data matrix size" );
    matP = &_P_orig;
    cvReleaseMat(&temp);

    temp = cvCreateMat(3,2,CV_64F);
    matP = temp;
    errcount += run_test_case( CV_StsAssert, "Invalid P data matrix size" );
    matP = &_P_orig;
    cvReleaseMat(&temp);
//------------
    //C++ tests
    useCPlus = true;

    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    P = cv::cvarrToMat(&_P_orig);
    R = cv::cvarrToMat(&_R_orig);
    src_points = cv::cvarrToMat(&_src_points_orig);

    temp = cvCreateMat(2,2,CV_32FC2);
    src_points = cv::cvarrToMat(temp);
    errcount += run_test_case( CV_StsAssert, "Invalid input data matrix size" );
    src_points = cv::cvarrToMat(&_src_points_orig);
    cvReleaseMat(&temp);

    temp = cvCreateMat(1,4,CV_64FC2);
    src_points = cv::cvarrToMat(temp);
    errcount += run_test_case( CV_StsAssert, "Invalid input data matrix type" );
    src_points = cv::cvarrToMat(&_src_points_orig);
    cvReleaseMat(&temp);

    src_points = cv::Mat();
    errcount += run_test_case( CV_StsAssert, "Input data matrix is not continuous" );
    src_points = cv::cvarrToMat(&_src_points_orig);
    cvReleaseMat(&temp);



//------------
    cvReleaseImage(&temp_img);
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
    //common
    cv::Size img_size;
    bool useCPlus;

    //C
    CvMat* _camera_mat;
    CvMat* matR;
    CvMat* _new_camera_mat;
    CvMat* _distortion_coeffs;
    CvMat* _mapx;
    CvMat* _mapy;


    //C++
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
    useCPlus = false;
    _camera_mat = matR = _new_camera_mat = _distortion_coeffs = _mapx = _mapy = NULL;
}

void CV_InitUndistortRectifyMapBadArgTest::run_func()
{
    if (useCPlus)
    {
        cv::initUndistortRectifyMap(camera_mat,distortion_coeffs,R,new_camera_mat,img_size,mat_type,mapx,mapy);
    }
    else
    {
        cvInitUndistortRectifyMap(_camera_mat,_distortion_coeffs,matR,_new_camera_mat,_mapx,_mapy);
    }
}

void CV_InitUndistortRectifyMapBadArgTest::run(int)
{
    int errcount = 0;
//initializing
    img_size.width = 800;
    img_size.height = 600;
    double cam[9] = {150.f, 0.f, img_size.width/2.f, 0, 300.f, img_size.height/2.f, 0.f, 0.f, 1.f};
    double dist[4] = {0.01,0.02,0.001,0.0005};
    float* arr_mapx = new float[img_size.width*img_size.height];
    float* arr_mapy = new float[img_size.width*img_size.height];
    double arr_new_camera_mat[9] = {155.f, 0.f, img_size.width/2.f+img_size.width/50.f, 0, 310.f, img_size.height/2.f+img_size.height/50.f, 0.f, 0.f, 1.f};
    double r[9] = {1,0,0,0,1,0,0,0,1};

    CvMat _camera_mat_orig = cvMat(3,3,CV_64F,cam);
    CvMat _distortion_coeffs_orig = cvMat(1,4,CV_64F,dist);
    CvMat _new_camera_mat_orig = cvMat(3,3,CV_64F,arr_new_camera_mat);
    CvMat _R_orig = cvMat(3,3,CV_64F,r);
    CvMat _mapx_orig = cvMat(img_size.height,img_size.width,CV_32FC1,arr_mapx);
    CvMat _mapy_orig = cvMat(img_size.height,img_size.width,CV_32FC1,arr_mapy);
    int mat_type_orig = CV_32FC1;

    _camera_mat = &_camera_mat_orig;
    _distortion_coeffs = &_distortion_coeffs_orig;
    _new_camera_mat = &_new_camera_mat_orig;
    matR = &_R_orig;
    _mapx = &_mapx_orig;
    _mapy = &_mapy_orig;
    mat_type = mat_type_orig;

//tests
    useCPlus = true;
    CvMat* temp;

    //C++ tests
    useCPlus = true;

    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    new_camera_mat = cv::cvarrToMat(&_new_camera_mat_orig);
    R = cv::cvarrToMat(&_R_orig);
    mapx = cv::cvarrToMat(&_mapx_orig);
    mapy = cv::cvarrToMat(&_mapy_orig);


    mat_type = CV_64F;
    errcount += run_test_case( CV_StsAssert, "Invalid map matrix type" );
    mat_type = mat_type_orig;

    temp = cvCreateMat(3,2,CV_32FC1);
    camera_mat = cv::cvarrToMat(temp);
    errcount += run_test_case( CV_StsAssert, "Invalid camera data matrix size" );
    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    cvReleaseMat(&temp);

    temp = cvCreateMat(4,3,CV_32FC1);
    R = cv::cvarrToMat(temp);
    errcount += run_test_case( CV_StsAssert, "Invalid R data matrix size" );
    R = cv::cvarrToMat(&_R_orig);
    cvReleaseMat(&temp);

    temp = cvCreateMat(6,1,CV_32FC1);
    distortion_coeffs = cv::cvarrToMat(temp);
    errcount += run_test_case( CV_StsAssert, "Invalid distortion coefficients data matrix size" );
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    cvReleaseMat(&temp);

//------------
    delete[] arr_mapx;
    delete[] arr_mapy;
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
    bool useCPlus;

    //C
    CvMat* _camera_mat;
    CvMat* _new_camera_mat;
    CvMat* _distortion_coeffs;
    CvMat* _src;
    CvMat* _dst;


    //C++
    cv::Mat camera_mat;
    cv::Mat new_camera_mat;
    cv::Mat distortion_coeffs;
    cv::Mat src;
    cv::Mat dst;

};

CV_UndistortBadArgTest::CV_UndistortBadArgTest ()
{
    useCPlus = false;
    _camera_mat = _new_camera_mat = _distortion_coeffs = _src = _dst = NULL;
}

void CV_UndistortBadArgTest::run_func()
{
    if (useCPlus)
    {
        cv::undistort(src,dst,camera_mat,distortion_coeffs,new_camera_mat);
    }
    else
    {
        cvUndistort2(_src,_dst,_camera_mat,_distortion_coeffs,_new_camera_mat);
    }
}

void CV_UndistortBadArgTest::run(int)
{
    int errcount = 0;
//initializing
    img_size.width = 800;
    img_size.height = 600;
    double cam[9] = {150.f, 0.f, img_size.width/2.f, 0, 300.f, img_size.height/2.f, 0.f, 0.f, 1.f};
    double dist[4] = {0.01,0.02,0.001,0.0005};
    float* arr_src = new float[img_size.width*img_size.height];
    float* arr_dst = new float[img_size.width*img_size.height];
    double arr_new_camera_mat[9] = {155.f, 0.f, img_size.width/2.f+img_size.width/50.f, 0, 310.f, img_size.height/2.f+img_size.height/50.f, 0.f, 0.f, 1.f};

    CvMat _camera_mat_orig = cvMat(3,3,CV_64F,cam);
    CvMat _distortion_coeffs_orig = cvMat(1,4,CV_64F,dist);
    CvMat _new_camera_mat_orig = cvMat(3,3,CV_64F,arr_new_camera_mat);
    CvMat _src_orig = cvMat(img_size.height,img_size.width,CV_32FC1,arr_src);
    CvMat _dst_orig = cvMat(img_size.height,img_size.width,CV_32FC1,arr_dst);

    _camera_mat = &_camera_mat_orig;
    _distortion_coeffs = &_distortion_coeffs_orig;
    _new_camera_mat = &_new_camera_mat_orig;
    _src = &_src_orig;
    _dst = &_dst_orig;

//tests
    useCPlus = true;
    CvMat* temp;
    CvMat* temp1;

//C tests
    useCPlus = false;

    temp = cvCreateMat(800,600,CV_32F);
    temp1 = cvCreateMat(800,601,CV_32F);
    _src = temp;
    _dst = temp1;
    errcount += run_test_case( CV_StsAssert, "Input and output data matrix sizes mismatch" );
    _src = &_src_orig;
    _dst = &_dst_orig;
    cvReleaseMat(&temp);
    cvReleaseMat(&temp1);

    temp = cvCreateMat(800,600,CV_32F);
    temp1 = cvCreateMat(800,600,CV_64F);
    _src = temp;
    _dst = temp1;
    errcount += run_test_case( CV_StsAssert, "Input and output data matrix types mismatch" );
    _src = &_src_orig;
    _dst = &_dst_orig;
    cvReleaseMat(&temp);
    cvReleaseMat(&temp1);

    //C++ tests
    useCPlus = true;

    camera_mat = cv::cvarrToMat(&_camera_mat_orig);
    distortion_coeffs = cv::cvarrToMat(&_distortion_coeffs_orig);
    new_camera_mat = cv::cvarrToMat(&_new_camera_mat_orig);
    src = cv::cvarrToMat(&_src_orig);
    dst = cv::cvarrToMat(&_dst_orig);

//------------
    delete[] arr_src;
    delete[] arr_dst;
    ts->set_failed_test_info(errcount > 0 ? cvtest::TS::FAIL_BAD_ARG_CHECK : cvtest::TS::OK);
}

TEST(Calib3d_UndistortPoints, badarg) { CV_UndistortPointsBadArgTest test; test.safe_run(); }
TEST(Calib3d_InitUndistortRectifyMap, badarg) { CV_InitUndistortRectifyMapBadArgTest test; test.safe_run(); }
TEST(Calib3d_Undistort, badarg) { CV_UndistortBadArgTest test; test.safe_run(); }

}} // namespace
/* End of file. */
