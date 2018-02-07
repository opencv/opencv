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
#include "opencv2/video/tracking_c.h"

namespace opencv_test { namespace {

/* ///////////////////// pyrlk_test ///////////////////////// */

class CV_OptFlowPyrLKTest : public cvtest::BaseTest
{
public:
    CV_OptFlowPyrLKTest();
protected:
    void run(int);
};


CV_OptFlowPyrLKTest::CV_OptFlowPyrLKTest() {}

void CV_OptFlowPyrLKTest::run( int )
{
    int code = cvtest::TS::OK;

    const double success_error_level = 0.3;
    const int bad_points_max = 8;

    /* test parameters */
    double  max_err = 0., sum_err = 0;
    int     pt_cmpd = 0;
    int     pt_exceed = 0;
    int     merr_i = 0, merr_j = 0, merr_k = 0, merr_nan = 0;
    char    filename[1000];

    CvPoint2D32f *u = 0, *v = 0, *v2 = 0;
    CvMat *_u = 0, *_v = 0, *_v2 = 0;
    char* status = 0;

    IplImage imgI;
    IplImage imgJ;
    cv::Mat  imgI2, imgJ2;

    int  n = 0, i = 0;

    sprintf( filename, "%soptflow/%s", ts->get_data_path().c_str(), "lk_prev.dat" );
    _u = (CvMat*)cvLoad( filename );

    if( !_u )
    {
        ts->printf( cvtest::TS::LOG, "could not read %s\n", filename );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    sprintf( filename, "%soptflow/%s", ts->get_data_path().c_str(), "lk_next.dat" );
    _v = (CvMat*)cvLoad( filename );

    if( !_v )
    {
        ts->printf( cvtest::TS::LOG, "could not read %s\n", filename );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    if( _u->cols != 2 || CV_MAT_TYPE(_u->type) != CV_32F ||
        _v->cols != 2 || CV_MAT_TYPE(_v->type) != CV_32F || _v->rows != _u->rows )
    {
        ts->printf( cvtest::TS::LOG, "the loaded matrices of points are not valid\n" );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;

    }

    u = (CvPoint2D32f*)_u->data.fl;
    v = (CvPoint2D32f*)_v->data.fl;

    /* allocate adidtional buffers */
    _v2 = cvCloneMat( _u );
    v2 = (CvPoint2D32f*)_v2->data.fl;

    /* read first image */
    sprintf( filename, "%soptflow/%s", ts->get_data_path().c_str(), "rock_1.bmp" );
    imgI2 = cv::imread( filename, cv::IMREAD_UNCHANGED );
    imgI = imgI2;

    if( imgI2.empty() )
    {
        ts->printf( cvtest::TS::LOG, "could not read %s\n", filename );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    /* read second image */
    sprintf( filename, "%soptflow/%s", ts->get_data_path().c_str(), "rock_2.bmp" );
    imgJ2 = cv::imread( filename, cv::IMREAD_UNCHANGED );
    imgJ = imgJ2;

    if( imgJ2.empty() )
    {
        ts->printf( cvtest::TS::LOG, "could not read %s\n", filename );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    n = _u->rows;
    status = (char*)cvAlloc(n*sizeof(status[0]));

    /* calculate flow */
    cvCalcOpticalFlowPyrLK( &imgI, &imgJ, 0, 0, u, v2, n, cvSize( 41, 41 ),
                            4, status, 0, cvTermCriteria( CV_TERMCRIT_ITER|
                            CV_TERMCRIT_EPS, 30, 0.01f ), 0 );

    /* compare results */
    for( i = 0; i < n; i++ )
    {
        if( status[i] != 0 )
        {
            double err;
            if( cvIsNaN(v[i].x) || cvIsNaN(v[i].y) )
            {
                merr_j++;
                continue;
            }

            if( cvIsNaN(v2[i].x) || cvIsNaN(v2[i].y) )
            {
                merr_nan++;
                continue;
            }

            err = fabs(v2[i].x - v[i].x) + fabs(v2[i].y - v[i].y);
            if( err > max_err )
            {
                max_err = err;
                merr_i = i;
            }

            pt_exceed += err > success_error_level;
            sum_err += err;
            pt_cmpd++;
        }
        else
        {
            if( !cvIsNaN( v[i].x ))
            {
                merr_i = i;
                merr_k++;
                ts->printf( cvtest::TS::LOG, "The algorithm lost the point #%d\n", i );
                code = cvtest::TS::FAIL_BAD_ACCURACY;
                goto _exit_;
            }
        }
    }

    if( pt_exceed > bad_points_max )
    {
        ts->printf( cvtest::TS::LOG,
                   "The number of poorly tracked points is too big (>=%d)\n", pt_exceed );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    if( max_err > 1 )
    {
        ts->printf( cvtest::TS::LOG, "Maximum tracking error is too big (=%g) at %d\n", max_err, merr_i );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    if( merr_nan > 0 )
    {
        ts->printf( cvtest::TS::LOG, "NAN tracking result with status != 0 (%d times)\n", merr_nan );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    cvFree( &status );
    cvReleaseMat( &_u );
    cvReleaseMat( &_v );
    cvReleaseMat( &_v2 );

    if( code < 0 )
        ts->set_failed_test_info( code );
}


TEST(Video_OpticalFlowPyrLK, accuracy) { CV_OptFlowPyrLKTest test; test.safe_run(); }

TEST(Video_OpticalFlowPyrLK, submat)
{
    // see bug #2075
    std::string path = cvtest::TS::ptr()->get_data_path() + "../cv/shared/lena.png";

    cv::Mat lenaImg = cv::imread(path);
    ASSERT_FALSE(lenaImg.empty());

    cv::Mat wholeImage;
    cv::resize(lenaImg, wholeImage, cv::Size(1024, 1024), 0, 0, cv::INTER_LINEAR_EXACT);

    cv::Mat img1 = wholeImage(cv::Rect(0, 0, 640, 360)).clone();
    cv::Mat img2 = wholeImage(cv::Rect(40, 60, 640, 360));

    std::vector<uchar> status;
    std::vector<float> error;
    std::vector<cv::Point2f> prev;
    std::vector<cv::Point2f> next;

    cv::RNG rng(123123);

    for(int i = 0; i < 50; ++i)
    {
        int x = rng.uniform(0, 640);
        int y = rng.uniform(0, 360);

        prev.push_back(cv::Point2f((float)x, (float)y));
    }

    ASSERT_NO_THROW(cv::calcOpticalFlowPyrLK(img1, img2, prev, next, status, error));
}

}} // namespace
