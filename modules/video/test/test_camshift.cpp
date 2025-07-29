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
#include "opencv2/video/tracking.hpp"

namespace opencv_test { namespace {

using namespace cv;

class CV_TrackBaseTest : public cvtest::BaseTest
{
public:
    CV_TrackBaseTest();
    virtual ~CV_TrackBaseTest();
    void clear();

protected:
    int read_params( const cv::FileStorage& fs );
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    void generate_object();

    int min_log_size, max_log_size;
    Mat img;
    RotatedRect box0;
    Size img_size;
    TermCriteria criteria;
    int img_type;
};


CV_TrackBaseTest::CV_TrackBaseTest()
{
    img = 0;
    test_case_count = 100;
    min_log_size = 5;
    max_log_size = 8;
}


CV_TrackBaseTest::~CV_TrackBaseTest()
{
    clear();
}


void CV_TrackBaseTest::clear()
{
    img.release();
    cvtest::BaseTest::clear();
}


int CV_TrackBaseTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::BaseTest::read_params( fs );
    if( code < 0 )
        return code;

    read( find_param( fs, "test_case_count" ), test_case_count, test_case_count );
    read( find_param( fs, "min_log_size" ), min_log_size, min_log_size );
    read( find_param( fs, "max_log_size" ), max_log_size, max_log_size );

    min_log_size = cvtest::clipInt( min_log_size, 1, 10 );
    max_log_size = cvtest::clipInt( max_log_size, 1, 10 );
    if( min_log_size > max_log_size )
    {
        std::swap( min_log_size, max_log_size );
    }

    return 0;
}


void CV_TrackBaseTest::generate_object()
{
    int x, y;
    double cx = box0.center.x;
    double cy = box0.center.y;
    double width = box0.size.width*0.5;
    double height = box0.size.height*0.5;
    double angle = box0.angle*CV_PI/180.;
    double a = sin(angle), b = -cos(angle);
    double inv_ww = 1./(width*width), inv_hh = 1./(height*height);

    img = Mat::zeros( img_size.height, img_size.width, img_type );

    // use the straightforward algorithm: for every pixel check if it is inside the ellipse
    for( y = 0; y < img_size.height; y++ )
    {
        uchar* ptr = img.ptr(y);
        float* fl = (float*)ptr;
        double x_ = (y - cy)*b, y_ = (y - cy)*a;

        for( x = 0; x < img_size.width; x++ )
        {
            double x1 = (x - cx)*a - x_;
            double y1 = (x - cx)*b + y_;

            if( x1*x1*inv_hh + y1*y1*inv_ww <= 1. )
            {
                if( img_type == CV_8U )
                    ptr[x] = (uchar)1;
                else
                    fl[x] = (float)1.f;
            }
        }
    }
}


int CV_TrackBaseTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    cvtest::BaseTest::prepare_test_case( test_case_idx );
    float m;

    clear();

    box0.size.width = (float)exp((cvtest::randReal(rng) * (max_log_size - min_log_size) + min_log_size)*CV_LOG2);
    box0.size.height = (float)exp((cvtest::randReal(rng) * (max_log_size - min_log_size) + min_log_size)*CV_LOG2);
    box0.angle = (float)(cvtest::randReal(rng)*180.);

    if( box0.size.width > box0.size.height )
    {
        std::swap( box0.size.width, box0.size.height );
    }

    m = MAX( box0.size.width, box0.size.height );
    img_size.width = cvRound(cvtest::randReal(rng)*m*0.5 + m + 1);
    img_size.height = cvRound(cvtest::randReal(rng)*m*0.5 + m + 1);
    img_type = cvtest::randInt(rng) % 2 ? CV_32F : CV_8U;
    img_type = CV_8U;

    box0.center.x = (float)(img_size.width*0.5 + (cvtest::randReal(rng)-0.5)*(img_size.width - m));
    box0.center.y = (float)(img_size.height*0.5 + (cvtest::randReal(rng)-0.5)*(img_size.height - m));

    criteria = TermCriteria( TermCriteria::EPS + TermCriteria::MAX_ITER, 10, 0.1 );

    generate_object();

    return 1;
}


void CV_TrackBaseTest::run_func(void)
{
}


int CV_TrackBaseTest::validate_test_results( int /*test_case_idx*/ )
{
    return 0;
}



///////////////////////// CamShift //////////////////////////////

class CV_CamShiftTest : public CV_TrackBaseTest
{
public:
    CV_CamShiftTest();

protected:
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    void generate_object();

    RotatedRect box;
    Rect init_rect;
    int area0;
};


CV_CamShiftTest::CV_CamShiftTest()
{
}


int CV_CamShiftTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    double m;
    int code = CV_TrackBaseTest::prepare_test_case( test_case_idx );
    int i, area;

    if( code <= 0 )
        return code;

    area0 = countNonZero(img);

    for(i = 0; i < 100; i++)
    {
        m = MAX(box0.size.width,box0.size.height)*0.8;
        init_rect.x = cvFloor(box0.center.x - m*(0.45 + cvtest::randReal(rng)*0.2));
        init_rect.y = cvFloor(box0.center.y - m*(0.45 + cvtest::randReal(rng)*0.2));
        init_rect.width = cvCeil(box0.center.x + m*(0.45 + cvtest::randReal(rng)*0.2) - init_rect.x);
        init_rect.height = cvCeil(box0.center.y + m*(0.45 + cvtest::randReal(rng)*0.2) - init_rect.y);

        if( init_rect.x < 0 || init_rect.y < 0 ||
            init_rect.x + init_rect.width >= img_size.width ||
            init_rect.y + init_rect.height >= img_size.height )
            continue;

        Mat temp = img(init_rect);
        area = countNonZero( temp );

        if( area >= 0.1*area0 )
            break;
    }

    return i < 100 ? code : 0;
}


void CV_CamShiftTest::run_func(void)
{
    box = CamShift( img, init_rect, criteria );
}


int CV_CamShiftTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;

    double m = MAX(box0.size.width, box0.size.height);
    double diff_angle;

    if( cvIsNaN(box.size.width) || cvIsInf(box.size.width) || box.size.width <= 0 ||
        cvIsNaN(box.size.height) || cvIsInf(box.size.height) || box.size.height <= 0 ||
        cvIsNaN(box.center.x) || cvIsInf(box.center.x) ||
        cvIsNaN(box.center.y) || cvIsInf(box.center.y) ||
        cvIsNaN(box.angle) || cvIsInf(box.angle) || box.angle < -180 || box.angle > 180 )
    {
        ts->printf( cvtest::TS::LOG, "Invalid RotatedRect was returned by CamShift\n" );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
        goto _exit_;
    }

    box.angle = (float)(180 - box.angle);

    if( fabs(box.size.width - box0.size.width) > box0.size.width*0.2 ||
        fabs(box.size.height - box0.size.height) > box0.size.height*0.3 )
    {
        ts->printf( cvtest::TS::LOG, "Incorrect CvBox2D size (=%.1f x %.1f, should be %.1f x %.1f)\n",
            box.size.width, box.size.height, box0.size.width, box0.size.height );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    if( fabs(box.center.x - box0.center.x) > m*0.1 ||
        fabs(box.center.y - box0.center.y) > m*0.1 )
    {
        ts->printf( cvtest::TS::LOG, "Incorrect CvBox2D position (=(%.1f, %.1f), should be (%.1f, %.1f))\n",
            box.center.x, box.center.y, box0.center.x, box0.center.y );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    if( box.angle < 0 )
        box.angle += 180;

    diff_angle = fabs(box0.angle - box.angle);
    diff_angle = MIN( diff_angle, fabs(box0.angle - box.angle + 180));

    if( fabs(diff_angle) > 30 && box0.size.height > box0.size.width*1.2 )
    {
        ts->printf( cvtest::TS::LOG, "Incorrect CvBox2D angle (=%1.f, should be %1.f)\n",
            box.angle, box0.angle );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

_exit_:

    if( code < 0 )
    {
        ts->set_failed_test_info( code );
    }
    return code;
}


///////////////////////// MeanShift //////////////////////////////

class CV_MeanShiftTest : public CV_TrackBaseTest
{
public:
    CV_MeanShiftTest();

protected:
    void run_func(void);
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    void generate_object();

    Rect init_rect, rect;
    int area0, area;
};


CV_MeanShiftTest::CV_MeanShiftTest()
{
}


int CV_MeanShiftTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    double m;
    int code = CV_TrackBaseTest::prepare_test_case( test_case_idx );
    int i;

    if( code <= 0 )
        return code;

    area0 = countNonZero(img);

    for(i = 0; i < 100; i++)
    {
        m = (box0.size.width + box0.size.height)*0.5;
        init_rect.x = cvFloor(box0.center.x - m*(0.4 + cvtest::randReal(rng)*0.2));
        init_rect.y = cvFloor(box0.center.y - m*(0.4 + cvtest::randReal(rng)*0.2));
        init_rect.width = cvCeil(box0.center.x + m*(0.4 + cvtest::randReal(rng)*0.2) - init_rect.x);
        init_rect.height = cvCeil(box0.center.y + m*(0.4 + cvtest::randReal(rng)*0.2) - init_rect.y);

        if( init_rect.x < 0 || init_rect.y < 0 ||
            init_rect.x + init_rect.width >= img_size.width ||
            init_rect.y + init_rect.height >= img_size.height )
            continue;

        Mat temp = img(init_rect);
        area = countNonZero( temp );

        if( area >= 0.5*area0 )
            break;
    }

    return i < 100 ? code : 0;
}


void CV_MeanShiftTest::run_func(void)
{
    rect = init_rect;
    meanShift( img, rect, criteria );
}


int CV_MeanShiftTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;
    Point2f c;
    double m = MAX(box0.size.width, box0.size.height), delta;

    c.x = (float)(rect.x + rect.width*0.5);
    c.y = (float)(rect.y + rect.height*0.5);

    if( fabs(c.x - box0.center.x) > m*0.1 ||
        fabs(c.y - box0.center.y) > m*0.1 )
    {
        ts->printf( cvtest::TS::LOG, "Incorrect CvBox2D position (=(%.1f, %.1f), should be (%.1f, %.1f))\n",
            c.x, c.y, box0.center.x, box0.center.y );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }

    delta = m*0.7;

    if( rect.x < box0.center.x - delta ||
        rect.y < box0.center.y - delta ||
        rect.x + rect.width > box0.center.x + delta ||
        rect.y + rect.height > box0.center.y + delta )
    {
        ts->printf( cvtest::TS::LOG,
            "Incorrect ConnectedComp ((%d,%d,%d,%d) is not within (%.1f,%.1f,%.1f,%.1f))\n",
            rect.x, rect.y, rect.x + rect.width, rect.y + rect.height,
            box0.center.x - delta, box0.center.y - delta, box0.center.x + delta, box0.center.y + delta );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

_exit_:

    if( code < 0 )
    {
        ts->set_failed_test_info( code );
    }
    return code;
}


TEST(Video_CAMShift, accuracy) { CV_CamShiftTest test; test.safe_run(); }
TEST(Video_MeanShift, accuracy) { CV_MeanShiftTest test; test.safe_run(); }

}} // namespace
/* End of file. */
