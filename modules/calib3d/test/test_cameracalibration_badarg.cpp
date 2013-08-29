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

#include <iostream>

using namespace cv;
using namespace std;

class CV_CameraCalibrationBadArgTest : public cvtest::BadArgTest
{
public:
    CV_CameraCalibrationBadArgTest() : imgSize(800, 600) {}
    ~CV_CameraCalibrationBadArgTest() {}
protected:
    void run(int);
    void run_func(void) {};

    const static int M = 1;

    Size imgSize;
    Size corSize;
    Mat chessBoard;
    Mat corners;

    struct C_Caller
    {
        CvMat* objPts;
        CvMat* imgPts;
        CvMat* npoints;
        Size imageSize;
        CvMat *cameraMatrix;
        CvMat *distCoeffs;
        CvMat *rvecs;
        CvMat *tvecs;
        int flags;

        void operator()() const
        {
            cvCalibrateCamera2(objPts, imgPts, npoints, imageSize,
                cameraMatrix, distCoeffs, rvecs, tvecs, flags );
        }
    };
};


void CV_CameraCalibrationBadArgTest::run( int /* start_from */ )
{
    Mat_<float> camMat(3, 3);
    Mat_<float> distCoeffs0(1, 5);

    camMat << 300.f, 0.f, imgSize.width/2.f, 0, 300.f, imgSize.height/2.f, 0.f, 0.f, 1.f;
    distCoeffs0 << 1.2f, 0.2f, 0.f, 0.f, 0.f;

    ChessBoardGenerator cbg(Size(8,6));
    corSize = cbg.cornersSize();
    vector<Point2f> exp_corn;
    chessBoard = cbg(Mat(imgSize, CV_8U, Scalar(0)), camMat, distCoeffs0, exp_corn);
    Mat_<Point2f>(corSize.height, corSize.width, (Point2f*)&exp_corn[0]).copyTo(corners);

    CvMat objPts, imgPts, npoints, cameraMatrix, distCoeffs, rvecs, tvecs;
    Mat zeros(1, sizeof(CvMat), CV_8U, Scalar(0));

    C_Caller caller, bad_caller;
    caller.imageSize = imgSize;
    caller.objPts = &objPts;
    caller.imgPts = &imgPts;
    caller.npoints = &npoints;
    caller.cameraMatrix = &cameraMatrix;
    caller.distCoeffs = &distCoeffs;
    caller.rvecs = &rvecs;
    caller.tvecs = &tvecs;

    /////////////////////////////
    Mat objPts_cpp;
    Mat imgPts_cpp;
    Mat npoints_cpp;
    Mat cameraMatrix_cpp;
    Mat distCoeffs_cpp;
    Mat rvecs_cpp;
    Mat tvecs_cpp;

    objPts_cpp.create(corSize, CV_32FC3);
    for(int j = 0; j < corSize.height; ++j)
        for(int i = 0; i < corSize.width; ++i)
            objPts_cpp.at<Point3f>(j, i) = Point3i(i, j, 0);
    objPts_cpp = objPts_cpp.reshape(3, 1);

    imgPts_cpp = corners.clone().reshape(2, 1);
    npoints_cpp = Mat_<int>(M, 1, corSize.width * corSize.height);
    cameraMatrix_cpp.create(3, 3, CV_32F);
    distCoeffs_cpp.create(5, 1, CV_32F);
    rvecs_cpp.create(M, 1, CV_32FC3);
    tvecs_cpp.create(M, 1, CV_32FC3);

    caller.flags = 0;
    //CV_CALIB_USE_INTRINSIC_GUESS;    //CV_CALIB_FIX_ASPECT_RATIO
    //CV_CALIB_USE_INTRINSIC_GUESS    //CV_CALIB_FIX_ASPECT_RATIO
    //CV_CALIB_FIX_PRINCIPAL_POINT    //CV_CALIB_ZERO_TANGENT_DIST
    //CV_CALIB_FIX_FOCAL_LENGTH    //CV_CALIB_FIX_K1    //CV_CALIB_FIX_K2    //CV_CALIB_FIX_K3

    objPts = objPts_cpp;
    imgPts = imgPts_cpp;
    npoints = npoints_cpp;
    cameraMatrix = cameraMatrix_cpp;
    distCoeffs = distCoeffs_cpp;
    rvecs = rvecs_cpp;
    tvecs = tvecs_cpp;

    /* /*//*/ */
    int errors = 0;

    bad_caller = caller;
    bad_caller.objPts = 0;
    errors += run_test_case( CV_StsBadArg, "Zero passed in objPts", bad_caller);

    bad_caller = caller;
    bad_caller.imgPts = 0;
    errors += run_test_case( CV_StsBadArg, "Zero passed in imgPts", bad_caller );

    bad_caller = caller;
    bad_caller.npoints = 0;
    errors += run_test_case( CV_StsBadArg, "Zero passed in npoints", bad_caller );

    bad_caller = caller;
    bad_caller.cameraMatrix = 0;
    errors += run_test_case( CV_StsBadArg, "Zero passed in cameraMatrix", bad_caller );

    bad_caller = caller;
    bad_caller.distCoeffs = 0;
    errors += run_test_case( CV_StsBadArg, "Zero passed in distCoeffs", bad_caller );

    bad_caller = caller;
    bad_caller.imageSize.width = -1;
    errors += run_test_case( CV_StsOutOfRange, "Bad image width", bad_caller );

    bad_caller = caller;
    bad_caller.imageSize.height = -1;
    errors += run_test_case( CV_StsOutOfRange, "Bad image height", bad_caller );

    Mat bad_nts_cpp1 = Mat_<float>(M, 1, 1.f);
    Mat bad_nts_cpp2 = Mat_<int>(3, 3, corSize.width * corSize.height);
    CvMat bad_npts_c1 = bad_nts_cpp1;
    CvMat bad_npts_c2 = bad_nts_cpp2;

    bad_caller = caller;
    bad_caller.npoints = &bad_npts_c1;
    errors += run_test_case( CV_StsUnsupportedFormat, "Bad npoints format", bad_caller );

    bad_caller = caller;
    bad_caller.npoints = &bad_npts_c2;
    errors += run_test_case( CV_StsUnsupportedFormat, "Bad npoints size", bad_caller );

    bad_caller = caller;
    bad_caller.rvecs = (CvMat*)zeros.ptr();
    errors += run_test_case( CV_StsBadArg, "Bad rvecs header", bad_caller );

    bad_caller = caller;
    bad_caller.tvecs = (CvMat*)zeros.ptr();
    errors += run_test_case( CV_StsBadArg, "Bad tvecs header", bad_caller );

    Mat bad_rvecs_cpp1(M+1, 1, CV_32FC3); CvMat bad_rvecs_c1 = bad_rvecs_cpp1;
    Mat bad_tvecs_cpp1(M+1, 1, CV_32FC3); CvMat bad_tvecs_c1 = bad_tvecs_cpp1;



    Mat bad_rvecs_cpp2(M, 2, CV_32FC3); CvMat bad_rvecs_c2 = bad_rvecs_cpp2;
    Mat bad_tvecs_cpp2(M, 2, CV_32FC3); CvMat bad_tvecs_c2 = bad_tvecs_cpp2;

    bad_caller = caller;
    bad_caller.rvecs = &bad_rvecs_c1;
    errors += run_test_case( CV_StsBadArg, "Bad tvecs header", bad_caller );

    bad_caller = caller;
    bad_caller.rvecs = &bad_rvecs_c2;
    errors += run_test_case( CV_StsBadArg, "Bad tvecs header", bad_caller );

    bad_caller = caller;
    bad_caller.tvecs = &bad_tvecs_c1;
    errors += run_test_case( CV_StsBadArg, "Bad tvecs header", bad_caller );

    bad_caller = caller;
    bad_caller.tvecs = &bad_tvecs_c2;
    errors += run_test_case( CV_StsBadArg, "Bad tvecs header", bad_caller );

    Mat bad_cameraMatrix_cpp1(3, 3, CV_32S); CvMat bad_cameraMatrix_c1 = bad_cameraMatrix_cpp1;
    Mat bad_cameraMatrix_cpp2(2, 3, CV_32F); CvMat bad_cameraMatrix_c2 = bad_cameraMatrix_cpp2;
    Mat bad_cameraMatrix_cpp3(3, 2, CV_64F); CvMat bad_cameraMatrix_c3 = bad_cameraMatrix_cpp3;



    bad_caller = caller;
    bad_caller.cameraMatrix = &bad_cameraMatrix_c1;
    errors += run_test_case( CV_StsBadArg, "Bad camearaMatrix header", bad_caller );

    bad_caller = caller;
    bad_caller.cameraMatrix = &bad_cameraMatrix_c2;
    errors += run_test_case( CV_StsBadArg, "Bad camearaMatrix header", bad_caller );

    bad_caller = caller;
    bad_caller.cameraMatrix = &bad_cameraMatrix_c3;
    errors += run_test_case( CV_StsBadArg, "Bad camearaMatrix header", bad_caller );

    Mat bad_distCoeffs_cpp1(1, 5, CV_32S); CvMat bad_distCoeffs_c1 = bad_distCoeffs_cpp1;
    Mat bad_distCoeffs_cpp2(2, 2, CV_64F); CvMat bad_distCoeffs_c2 = bad_distCoeffs_cpp2;
    Mat bad_distCoeffs_cpp3(1, 6, CV_64F); CvMat bad_distCoeffs_c3 = bad_distCoeffs_cpp3;



    bad_caller = caller;
    bad_caller.distCoeffs = &bad_distCoeffs_c1;
    errors += run_test_case( CV_StsBadArg, "Bad distCoeffs header", bad_caller );

    bad_caller = caller;
    bad_caller.distCoeffs = &bad_distCoeffs_c2;
    errors += run_test_case( CV_StsBadArg, "Bad distCoeffs header", bad_caller );


    bad_caller = caller;
    bad_caller.distCoeffs = &bad_distCoeffs_c3;
    errors += run_test_case( CV_StsBadArg, "Bad distCoeffs header", bad_caller );

    double CM[] = {0, 0, 0, /**/0, 0, 0, /**/0, 0, 0};
    Mat bad_cameraMatrix_cpp4(3, 3, CV_64F, CM); CvMat bad_cameraMatrix_c4 = bad_cameraMatrix_cpp4;

    bad_caller = caller;
    bad_caller.flags |= CV_CALIB_USE_INTRINSIC_GUESS;
    bad_caller.cameraMatrix = &bad_cameraMatrix_c4;
    CM[0] = 0; //bad fx
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[0] = 500; CM[4] = 0;  //bad fy
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[0] = 500; CM[4] = 500; CM[2] = -1; //bad cx
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[0] = 500; CM[4] = 500; CM[2] = imgSize.width*2; //bad cx
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[0] = 500; CM[4] = 500; CM[2] = imgSize.width/2;  CM[5] = -1; //bad cy
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[0] = 500; CM[4] = 500; CM[2] = imgSize.width/2;  CM[5] = imgSize.height*2; //bad cy
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[0] = 500; CM[4] = 500; CM[2] = imgSize.width/2; CM[5] = imgSize.height/2;
    CM[1] = 0.1; //Non-zero skew
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[1] = 0;
    CM[3] = 0.1; /* mad matrix shape */
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[3] = 0; CM[6] = 0.1; /* mad matrix shape */
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[3] = 0; CM[6] = 0; CM[7] = 0.1; /* mad matrix shape */
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );

    CM[3] = 0; CM[6] = 0; CM[7] = 0; CM[8] = 1.1; /* mad matrix shape */
    errors += run_test_case( CV_StsOutOfRange, "Bad camearaMatrix data", bad_caller );
    CM[8] = 1.0;

    /////////////////////////////////////////////////////////////////////////////////////
    bad_caller = caller;
    Mat bad_objPts_cpp5 = objPts_cpp.clone(); CvMat bad_objPts_c5 = bad_objPts_cpp5;
    bad_caller.objPts = &bad_objPts_c5;

    cv::RNG& rng = theRNG();
    for(int i = 0; i < bad_objPts_cpp5.rows; ++i)
        bad_objPts_cpp5.at<Point3f>(0, i).z += ((float)rng - 0.5f);

    errors += run_test_case( CV_StsBadArg, "Bad objPts data", bad_caller );

    if (errors)
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    else
        ts->set_failed_test_info(cvtest::TS::OK);

    //try { caller(); }
    //catch (...)
    //{
    //    ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    //    printf("+!");
    //}
}


class CV_Rodrigues2BadArgTest : public cvtest::BadArgTest
{
public:
    CV_Rodrigues2BadArgTest() {}
    ~CV_Rodrigues2BadArgTest() {}
protected:
    void run_func(void) {};

    struct C_Caller
    {
        CvMat* src;
        CvMat* dst;
        CvMat* jacobian;

        void operator()() { cvRodrigues2(src, dst, jacobian); }
    };

    void run(int /* start_from */ )
    {
        Mat zeros(1, sizeof(CvMat), CV_8U, Scalar(0));
        CvMat src_c, dst_c, jacobian_c;

        Mat src_cpp(3, 1, CV_32F); src_c = src_cpp;
        Mat dst_cpp(3, 3, CV_32F); dst_c = dst_cpp;
        Mat jacobian_cpp(3, 9, CV_32F); jacobian_c = jacobian_cpp;

        C_Caller caller, bad_caller;
        caller.src = &src_c;
        caller.dst = &dst_c;
        caller.jacobian = &jacobian_c;

       /* try { caller(); }
        catch (...)
        {
            printf("badasfas");
        }*/

        /*/*//*/*/
        int errors = 0;

        bad_caller = caller;
        bad_caller.src = 0;
        errors += run_test_case( CV_StsNullPtr, "Src is zero pointer", bad_caller );

        bad_caller = caller;
        bad_caller.dst = 0;
        errors += run_test_case( CV_StsNullPtr, "Dst is zero pointer", bad_caller );

        Mat bad_src_cpp1(3, 1, CV_8U); CvMat bad_src_c1 = bad_src_cpp1;
        Mat bad_dst_cpp1(3, 1, CV_8U); CvMat bad_dst_c1 = bad_dst_cpp1;
        Mat bad_jac_cpp1(3, 1, CV_8U); CvMat bad_jac_c1 = bad_jac_cpp1;
        Mat bad_jac_cpp2(3, 1, CV_32FC2); CvMat bad_jac_c2 = bad_jac_cpp2;
        Mat bad_jac_cpp3(3, 1, CV_32F); CvMat bad_jac_c3 = bad_jac_cpp3;

        bad_caller = caller;
        bad_caller.src = &bad_src_c1;
        errors += run_test_case( CV_StsUnsupportedFormat, "Bad src formart", bad_caller );

        bad_caller = caller;
        bad_caller.dst = &bad_dst_c1;
        errors += run_test_case( CV_StsUnmatchedFormats, "Bad dst formart", bad_caller );

        bad_caller = caller;
        bad_caller.jacobian = (CvMat*)zeros.ptr();
        errors += run_test_case( CV_StsBadArg, "Bad jacobian ", bad_caller );

        bad_caller = caller;
        bad_caller.jacobian = &bad_jac_c1;
        errors += run_test_case( CV_StsUnmatchedFormats, "Bad jacobian format", bad_caller );

        bad_caller = caller;
        bad_caller.jacobian = &bad_jac_c2;
        errors += run_test_case( CV_StsUnmatchedFormats, "Bad jacobian format", bad_caller );

        bad_caller = caller;
        bad_caller.jacobian = &bad_jac_c3;
        errors += run_test_case( CV_StsBadSize, "Bad jacobian format", bad_caller );

        Mat bad_src_cpp2(1, 1, CV_32F); CvMat bad_src_c2 = bad_src_cpp2;

        bad_caller = caller;
        bad_caller.src = &bad_src_c2;
        errors += run_test_case( CV_StsBadSize, "Bad src format", bad_caller );

        Mat bad_dst_cpp2(2, 1, CV_32F); CvMat bad_dst_c2 = bad_dst_cpp2;
        Mat bad_dst_cpp3(3, 2, CV_32F); CvMat bad_dst_c3 = bad_dst_cpp3;
        Mat bad_dst_cpp4(3, 3, CV_32FC2); CvMat bad_dst_c4 = bad_dst_cpp4;

        bad_caller = caller;
        bad_caller.dst = &bad_dst_c2;
        errors += run_test_case( CV_StsBadSize, "Bad dst format", bad_caller );

        bad_caller = caller;
        bad_caller.dst = &bad_dst_c3;
        errors += run_test_case( CV_StsBadSize, "Bad dst format", bad_caller );

        bad_caller = caller;
        bad_caller.dst = &bad_dst_c4;
        errors += run_test_case( CV_StsBadSize, "Bad dst format", bad_caller );


        /********/
        src_cpp.create(3, 3, CV_32F); src_c = src_cpp;
        dst_cpp.create(3, 1, CV_32F); dst_c = dst_cpp;


        Mat bad_dst_cpp5(5, 5, CV_32F); CvMat bad_dst_c5 = bad_dst_cpp5;

        bad_caller = caller;
        bad_caller.dst = &bad_dst_c5;
        errors += run_test_case( CV_StsBadSize, "Bad dst format", bad_caller );


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
    };
    ~CV_ProjectPoints2BadArgTest() {} ;
protected:
    void run_func(void) {};

    Mat_<float> camMat;
    Mat_<float> distCoeffs;

    struct C_Caller
    {
        CvMat* objectPoints;
        CvMat* r_vec;
        CvMat* t_vec;
        CvMat* A;
        CvMat* distCoeffs;
        CvMat* imagePoints;
        CvMat* dpdr;
        CvMat* dpdt;
        CvMat* dpdf;
        CvMat* dpdc;
        CvMat* dpdk;
        double aspectRatio;

        void operator()()
        {
            cvProjectPoints2( objectPoints, r_vec, t_vec, A, distCoeffs, imagePoints,
                dpdr, dpdt, dpdf, dpdc, dpdk, aspectRatio );
        }
    };

    void run(int /* start_from */ )
    {
        CvMat zeros;
        memset(&zeros, 0, sizeof(zeros));

        C_Caller caller, bad_caller;
        CvMat objectPoints_c, r_vec_c, t_vec_c, A_c, distCoeffs_c, imagePoints_c,
            dpdr_c, dpdt_c, dpdf_c, dpdc_c, dpdk_c;

        const int n = 10;

        Mat imagePoints_cpp(1, n, CV_32FC2); imagePoints_c = imagePoints_cpp;

        Mat objectPoints_cpp(1, n, CV_32FC3);
        randu(objectPoints_cpp, Scalar::all(1), Scalar::all(10));
        objectPoints_c = objectPoints_cpp;

        Mat t_vec_cpp(Mat::zeros(1, 3, CV_32F)); t_vec_c = t_vec_cpp;
        Mat r_vec_cpp;
        Rodrigues(Mat::eye(3, 3, CV_32F), r_vec_cpp); r_vec_c = r_vec_cpp;

        Mat A_cpp = camMat.clone(); A_c = A_cpp;
        Mat distCoeffs_cpp = distCoeffs.clone(); distCoeffs_c = distCoeffs_cpp;

        Mat dpdr_cpp(2*n, 3, CV_32F); dpdr_c = dpdr_cpp;
        Mat dpdt_cpp(2*n, 3, CV_32F); dpdt_c = dpdt_cpp;
        Mat dpdf_cpp(2*n, 2, CV_32F); dpdf_c = dpdf_cpp;
        Mat dpdc_cpp(2*n, 2, CV_32F); dpdc_c = dpdc_cpp;
        Mat dpdk_cpp(2*n, 4, CV_32F); dpdk_c = dpdk_cpp;

        caller.aspectRatio = 1.0;
        caller.objectPoints = &objectPoints_c;
        caller.r_vec = &r_vec_c;
        caller.t_vec = &t_vec_c;
        caller.A = &A_c;
        caller.distCoeffs = &distCoeffs_c;
        caller.imagePoints = &imagePoints_c;
        caller.dpdr = &dpdr_c;
        caller.dpdt = &dpdt_c;
        caller.dpdf = &dpdf_c;
        caller.dpdc = &dpdc_c;
        caller.dpdk = &dpdk_c;

        /********************/
        int errors = 0;


        bad_caller = caller;
        bad_caller.objectPoints = 0;
        errors += run_test_case( CV_StsBadArg, "Zero objectPoints", bad_caller );

        bad_caller = caller;
        bad_caller.r_vec = 0;
        errors += run_test_case( CV_StsBadArg, "Zero r_vec", bad_caller );

        bad_caller = caller;
        bad_caller.t_vec = 0;
        errors += run_test_case( CV_StsBadArg, "Zero t_vec", bad_caller );

        bad_caller = caller;
        bad_caller.A = 0;
        errors += run_test_case( CV_StsBadArg, "Zero camMat", bad_caller );

        bad_caller = caller;
        bad_caller.imagePoints = 0;
        errors += run_test_case( CV_StsBadArg, "Zero imagePoints", bad_caller );

        /****************************/
        Mat bad_r_vec_cpp1(r_vec_cpp.size(), CV_32S); CvMat bad_r_vec_c1 = bad_r_vec_cpp1;
        Mat bad_r_vec_cpp2(2, 2, CV_32F); CvMat bad_r_vec_c2 = bad_r_vec_cpp2;
        Mat bad_r_vec_cpp3(r_vec_cpp.size(), CV_32FC2); CvMat bad_r_vec_c3 = bad_r_vec_cpp3;

        bad_caller = caller;
        bad_caller.r_vec = &bad_r_vec_c1;
        errors += run_test_case( CV_StsBadArg, "Bad rvec format", bad_caller );

        bad_caller = caller;
        bad_caller.r_vec = &bad_r_vec_c2;
        errors += run_test_case( CV_StsBadArg, "Bad rvec format", bad_caller );

        bad_caller = caller;
        bad_caller.r_vec = &bad_r_vec_c3;
        errors += run_test_case( CV_StsBadArg, "Bad rvec format", bad_caller );

        /****************************/
        Mat bad_t_vec_cpp1(t_vec_cpp.size(), CV_32S); CvMat bad_t_vec_c1 = bad_t_vec_cpp1;
        Mat bad_t_vec_cpp2(2, 2, CV_32F); CvMat bad_t_vec_c2 = bad_t_vec_cpp2;
        Mat bad_t_vec_cpp3(1, 1, CV_32FC2); CvMat bad_t_vec_c3 = bad_t_vec_cpp3;

        bad_caller = caller;
        bad_caller.t_vec = &bad_t_vec_c1;
        errors += run_test_case( CV_StsBadArg, "Bad tvec format", bad_caller );

        bad_caller = caller;
        bad_caller.t_vec = &bad_t_vec_c2;
        errors += run_test_case( CV_StsBadArg, "Bad tvec format", bad_caller );

        bad_caller = caller;
        bad_caller.t_vec = &bad_t_vec_c3;
        errors += run_test_case( CV_StsBadArg, "Bad tvec format", bad_caller );

        /****************************/
        Mat bad_A_cpp1(A_cpp.size(), CV_32S); CvMat bad_A_c1 = bad_A_cpp1;
        Mat bad_A_cpp2(2, 2, CV_32F); CvMat bad_A_c2 = bad_A_cpp2;

        bad_caller = caller;
        bad_caller.A = &bad_A_c1;
        errors += run_test_case( CV_StsBadArg, "Bad A format", bad_caller );

        bad_caller = caller;
        bad_caller.A = &bad_A_c2;
        errors += run_test_case( CV_StsBadArg, "Bad A format", bad_caller );

        /****************************/
        Mat bad_distCoeffs_cpp1(distCoeffs_cpp.size(), CV_32S); CvMat bad_distCoeffs_c1 = bad_distCoeffs_cpp1;
        Mat bad_distCoeffs_cpp2(2, 2, CV_32F); CvMat bad_distCoeffs_c2 = bad_distCoeffs_cpp2;
        Mat bad_distCoeffs_cpp3(1, 7, CV_32F); CvMat bad_distCoeffs_c3 = bad_distCoeffs_cpp3;

        bad_caller = caller;
        bad_caller.distCoeffs = &zeros;
        errors += run_test_case( CV_StsBadArg, "Bad distCoeffs format", bad_caller );

        bad_caller = caller;
        bad_caller.distCoeffs = &bad_distCoeffs_c1;
        errors += run_test_case( CV_StsBadArg, "Bad distCoeffs format", bad_caller );

        bad_caller = caller;
        bad_caller.distCoeffs = &bad_distCoeffs_c2;
        errors += run_test_case( CV_StsBadArg, "Bad distCoeffs format", bad_caller );

        bad_caller = caller;
        bad_caller.distCoeffs = &bad_distCoeffs_c3;
        errors += run_test_case( CV_StsBadArg, "Bad distCoeffs format", bad_caller );


        /****************************/
        Mat bad_dpdr_cpp1(dpdr_cpp.size(), CV_32S); CvMat bad_dpdr_c1 = bad_dpdr_cpp1;
        Mat bad_dpdr_cpp2(dpdr_cpp.cols+1, 3, CV_32F); CvMat bad_dpdr_c2 = bad_dpdr_cpp2;
        Mat bad_dpdr_cpp3(dpdr_cpp.cols, 7, CV_32F); CvMat bad_dpdr_c3 = bad_dpdr_cpp3;

        bad_caller = caller;
        bad_caller.dpdr = &zeros;
        errors += run_test_case( CV_StsBadArg, "Bad dpdr format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdr = &bad_dpdr_c1;
        errors += run_test_case( CV_StsBadArg, "Bad dpdr format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdr = &bad_dpdr_c2;
        errors += run_test_case( CV_StsBadArg, "Bad dpdr format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdr = &bad_dpdr_c3;
        errors += run_test_case( CV_StsBadArg, "Bad dpdr format", bad_caller );

        /****************************/

        bad_caller = caller;
        bad_caller.dpdt = &zeros;
        errors += run_test_case( CV_StsBadArg, "Bad dpdt format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdt = &bad_dpdr_c1;
        errors += run_test_case( CV_StsBadArg, "Bad dpdt format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdt = &bad_dpdr_c2;
        errors += run_test_case( CV_StsBadArg, "Bad dpdt format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdt = &bad_dpdr_c3;
        errors += run_test_case( CV_StsBadArg, "Bad dpdt format", bad_caller );

        /****************************/

        Mat bad_dpdf_cpp2(dpdr_cpp.cols+1, 2, CV_32F); CvMat bad_dpdf_c2 = bad_dpdf_cpp2;

        bad_caller = caller;
        bad_caller.dpdf = &zeros;
        errors += run_test_case( CV_StsBadArg, "Bad dpdf format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdf = &bad_dpdr_c1;
        errors += run_test_case( CV_StsBadArg, "Bad dpdf format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdf = &bad_dpdf_c2;
        errors += run_test_case( CV_StsBadArg, "Bad dpdf format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdf = &bad_dpdr_c3;
        errors += run_test_case( CV_StsBadArg, "Bad dpdf format", bad_caller );

        /****************************/

        bad_caller = caller;
        bad_caller.dpdc = &zeros;
        errors += run_test_case( CV_StsBadArg, "Bad dpdc format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdc = &bad_dpdr_c1;
        errors += run_test_case( CV_StsBadArg, "Bad dpdc format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdc = &bad_dpdf_c2;
        errors += run_test_case( CV_StsBadArg, "Bad dpdc format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdc = &bad_dpdr_c3;
        errors += run_test_case( CV_StsBadArg, "Bad dpdc format", bad_caller );

        /****************************/

        bad_caller = caller;
        bad_caller.dpdk = &zeros;
        errors += run_test_case( CV_StsBadArg, "Bad dpdk format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdk = &bad_dpdr_c1;
        errors += run_test_case( CV_StsBadArg, "Bad dpdk format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdk = &bad_dpdf_c2;
        errors += run_test_case( CV_StsBadArg, "Bad dpdk format", bad_caller );

        bad_caller = caller;
        bad_caller.dpdk = &bad_dpdr_c3;
        errors += run_test_case( CV_StsBadArg, "Bad dpdk format", bad_caller );

        bad_caller = caller;
        bad_caller.distCoeffs = 0;
        errors += run_test_case( CV_StsNullPtr, "distCoeffs is NULL while dpdk is not", bad_caller );


        if (errors)
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        else
            ts->set_failed_test_info(cvtest::TS::OK);
    }
};


TEST(Calib3d_CalibrateCamera_C, badarg) { CV_CameraCalibrationBadArgTest test; test.safe_run(); }
TEST(Calib3d_Rodrigues_C, badarg) { CV_Rodrigues2BadArgTest test; test.safe_run(); }
TEST(Calib3d_ProjectPoints_C, badarg) { CV_ProjectPoints2BadArgTest test; test.safe_run(); }
