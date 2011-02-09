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

using namespace cv;
using namespace std;

///////////////////// base MHI class ///////////////////////
class CV_MHIBaseTest : public cvtest::ArrayTest
{
public:
    CV_MHIBaseTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    int prepare_test_case( int test_case_idx );
    double timestamp, duration, max_log_duration;
    int mhi_i, mhi_ref_i;
    double silh_ratio;
};


CV_MHIBaseTest::CV_MHIBaseTest()
{
    timestamp = duration = 0;
    max_log_duration = 9;
    mhi_i = mhi_ref_i = -1;
    
    silh_ratio = 0.25;
}


void CV_MHIBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT && CV_MAT_DEPTH(type) == CV_8U )
    {
        low = Scalar::all(cvRound(-1./silh_ratio)+2.);
        high = Scalar::all(2);
    }
    else if( i == mhi_i || i == mhi_ref_i )
    {
        low = Scalar::all(-exp(max_log_duration));
        high = Scalar::all(0.);
    }
}


void CV_MHIBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[INPUT][0] = CV_8UC1;
    types[mhi_i][0] = types[mhi_ref_i][0] = CV_32FC1;
    duration = exp(cvtest::randReal(rng)*max_log_duration);
    timestamp = duration + cvtest::randReal(rng)*30.-10.;
}


int CV_MHIBaseTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        Mat& mat = test_mat[mhi_i][0];
        mat += Scalar::all(duration);
        cv::max(mat, 0, mat);
        if( mhi_i != mhi_ref_i )
        {
            Mat& mat0 = test_mat[mhi_ref_i][0];
            cvtest::copy( mat, mat0 );
        }
    }

    return code;
}


///////////////////// update motion history ////////////////////////////

static void test_updateMHI( const Mat& silh, Mat& mhi, double timestamp, double duration )
{
    int i, j;
    float delbound = (float)(timestamp - duration);
    for( i = 0; i < mhi.rows; i++ )
    {
        const uchar* silh_row = silh.ptr(i);
        float* mhi_row = mhi.ptr<float>(i);

        for( j = 0; j < mhi.cols; j++ )
        {
            if( silh_row[j] )
                mhi_row[j] = (float)timestamp;
            else if( mhi_row[j] < delbound )
                mhi_row[j] = 0.f;
        }
    }
}


class CV_UpdateMHITest : public CV_MHIBaseTest
{
public:
    CV_UpdateMHITest();

protected:
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );
};


CV_UpdateMHITest::CV_UpdateMHITest()
{
    test_array[INPUT].push_back(NULL);
    test_array[INPUT_OUTPUT].push_back(NULL);
    test_array[REF_INPUT_OUTPUT].push_back(NULL);
    mhi_i = INPUT_OUTPUT; mhi_ref_i = REF_INPUT_OUTPUT;
}


double CV_UpdateMHITest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_UpdateMHITest::run_func()
{
    cvUpdateMotionHistory( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], timestamp, duration );
}


void CV_UpdateMHITest::prepare_to_validation( int /*test_case_idx*/ )
{
    test_updateMHI( test_mat[INPUT][0], test_mat[REF_INPUT_OUTPUT][0], timestamp, duration );
}


///////////////////// calc motion gradient ////////////////////////////

static void test_MHIGradient( const Mat& mhi, Mat& mask, Mat& orientation,
                              double delta1, double delta2, int aperture_size )
{
    Point anchor( aperture_size/2, aperture_size/2 );
    double limit = 1e-4*aperture_size*aperture_size;
    
    Mat dx, dy, min_mhi, max_mhi;
    
    Mat kernel = cvtest::calcSobelKernel2D( 1, 0, aperture_size );
    cvtest::filter2D( mhi, dx, CV_32F, kernel, anchor, 0, BORDER_REPLICATE );
    kernel = cvtest::calcSobelKernel2D( 0, 1, aperture_size );
    cvtest::filter2D( mhi, dy, CV_32F, kernel, anchor, 0, BORDER_REPLICATE );

    kernel = Mat::ones(aperture_size, aperture_size, CV_8U);
    cvtest::erode(mhi, min_mhi, kernel, anchor, 0, BORDER_REPLICATE);
    cvtest::dilate(mhi, max_mhi, kernel, anchor, 0, BORDER_REPLICATE);

    if( delta1 > delta2 )
    {
        double t;
        CV_SWAP( delta1, delta2, t );
    }

    for( int i = 0; i < mhi.rows; i++ )
    {
        uchar* mask_row = mask.ptr(i);
        float* orient_row = orientation.ptr<float>(i);
        const float* dx_row = dx.ptr<float>(i);
        const float* dy_row = dy.ptr<float>(i);
        const float* min_row = min_mhi.ptr<float>(i);
        const float* max_row = max_mhi.ptr<float>(i);

        for( int j = 0; j < mhi.cols; j++ )
        {
            double delta = max_row[j] - min_row[j];
            double _dx = dx_row[j], _dy = dy_row[j];

            if( delta1 <= delta && delta <= delta2 &&
                (fabs(_dx) > limit || fabs(_dy) > limit) )
            {
                mask_row[j] = 1;
                double angle = atan2( _dy, _dx ) * (180/CV_PI);
                if( angle < 0 )
                    angle += 360.;
                orient_row[j] = (float)angle;
            }
            else
            {
                mask_row[j] = 0;
                orient_row[j] = 0.f;
            }
        }
    }
}


class CV_MHIGradientTest : public CV_MHIBaseTest
{
public:
    CV_MHIGradientTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    double delta1, delta2, delta_range_log;
    int aperture_size;
};


CV_MHIGradientTest::CV_MHIGradientTest()
{
    mhi_i = mhi_ref_i = INPUT;
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    delta1 = delta2 = 0;
    aperture_size = 0;
    delta_range_log = 4;
}


void CV_MHIGradientTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_MHIBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8UC1;
    types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_32FC1;
    delta1 = exp(cvtest::randReal(rng)*delta_range_log + 1.);
    delta2 = exp(cvtest::randReal(rng)*delta_range_log + 1.);
    aperture_size = (cvtest::randInt(rng)%3)*2+3;
    //duration = exp(cvtest::randReal(rng)*max_log_duration);
    //timestamp = duration + cvtest::randReal(rng)*30.-10.;
}


double CV_MHIGradientTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int j )
{
    return j == 0 ? 0 : 2e-1;
}


void CV_MHIGradientTest::run_func()
{
    cvCalcMotionGradient( test_array[INPUT][0], test_array[OUTPUT][0],
                          test_array[OUTPUT][1], delta1, delta2, aperture_size );
}


void CV_MHIGradientTest::prepare_to_validation( int /*test_case_idx*/ )
{
    test_MHIGradient( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                      test_mat[REF_OUTPUT][1], delta1, delta2, aperture_size );
    test_mat[REF_OUTPUT][0] += Scalar::all(1);
    test_mat[OUTPUT][0] += Scalar::all(1);
}


////////////////////// calc global orientation /////////////////////////

static double test_calcGlobalOrientation( const Mat& orient, const Mat& mask,
                                          const Mat& mhi, double timestamp, double duration )
{
    const int HIST_SIZE = 12;
    int      y, x;
    int      histogram[HIST_SIZE];
    int      max_bin = 0;

    double   base_orientation = 0, delta_orientation = 0, weight = 0;
    double   low_time, global_orientation;

    memset( histogram, 0, sizeof( histogram ));
    timestamp = 0;

    for( y = 0; y < orient.rows; y++ )
    {
        const float* orient_data = orient.ptr<float>(y);
        const uchar* mask_data = mask.ptr(y);
        const float* mhi_data = mhi.ptr<float>(y);
        for( x = 0; x < orient.cols; x++ )
            if( mask_data[x] )
            {
                int bin = cvFloor( (orient_data[x]*HIST_SIZE)/360 );
                histogram[bin < 0 ? 0 : bin >= HIST_SIZE ? HIST_SIZE-1 : bin]++;
                if( mhi_data[x] > timestamp )
                    timestamp = mhi_data[x];
            }
    }

    low_time = timestamp - duration;

    for( x = 1; x < HIST_SIZE; x++ )
    {
        if( histogram[x] > histogram[max_bin] )
            max_bin = x;
    }

    base_orientation = ((double)max_bin*360)/HIST_SIZE;

    for( y = 0; y < orient.rows; y++ )
    {
        const float* orient_data = orient.ptr<float>(y);
        const float* mhi_data = mhi.ptr<float>(y);
        const uchar* mask_data = mask.ptr(y);
        
        for( x = 0; x < orient.cols; x++ )
        {
            if( mask_data[x] && mhi_data[x] > low_time )
            {
                double diff = orient_data[x] - base_orientation;
                double delta_weight = (((mhi_data[x] - low_time)/duration)*254 + 1)/255;

                if( diff < -180 ) diff += 360;
                if( diff > 180 ) diff -= 360;

                if( delta_weight > 0 && fabs(diff) < 45 )
                {
                    delta_orientation += diff*delta_weight;
                    weight += delta_weight;
                }
            }
        }
    }

    if( weight == 0 )
        global_orientation = base_orientation;
    else
    {
        global_orientation = base_orientation + delta_orientation/weight;
        if( global_orientation < 0 ) global_orientation += 360;
        if( global_orientation > 360 ) global_orientation -= 360;
    }
    
    return global_orientation;
}


class CV_MHIGlobalOrientTest : public CV_MHIBaseTest
{
public:
    CV_MHIGlobalOrientTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int validate_test_results( int test_case_idx );
    void run_func();
    double angle, min_angle, max_angle;
};


CV_MHIGlobalOrientTest::CV_MHIGlobalOrientTest()
{
    mhi_i = mhi_ref_i = INPUT;
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    min_angle = max_angle = 0;
}


void CV_MHIGlobalOrientTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_MHIBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize size = sizes[INPUT][0];

    size.width = MAX( size.width, 16 );
    size.height = MAX( size.height, 16 );
    sizes[INPUT][0] = sizes[INPUT][1] = sizes[INPUT][2] = size;

    types[INPUT][1] = CV_8UC1; // mask
    types[INPUT][2] = CV_32FC1; // orientation

    min_angle = cvtest::randReal(rng)*359.9;
    max_angle = cvtest::randReal(rng)*359.9;
    if( min_angle >= max_angle )
    {
        double t;
        CV_SWAP( min_angle, max_angle, t );
    }
    max_angle += 0.1;
    duration = exp(cvtest::randReal(rng)*max_log_duration);
    timestamp = duration + cvtest::randReal(rng)*30.-10.;
}


void CV_MHIGlobalOrientTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    CV_MHIBaseTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT && j == 2 )
    {
        low = Scalar::all(min_angle);
        high = Scalar::all(max_angle);
    }
}


double CV_MHIGlobalOrientTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 15;
}


void CV_MHIGlobalOrientTest::run_func()
{
    angle = cvCalcGlobalOrientation( test_array[INPUT][2], test_array[INPUT][1],
                                     test_array[INPUT][0], timestamp, duration );
}


int CV_MHIGlobalOrientTest::validate_test_results( int test_case_idx )
{
    //printf("%d. rows=%d, cols=%d, nzmask=%d\n", test_case_idx, test_mat[INPUT][1].rows, test_mat[INPUT][1].cols,
    //       cvCountNonZero(test_array[INPUT][1]));
    
    double ref_angle = test_calcGlobalOrientation( test_mat[INPUT][2], test_mat[INPUT][1],
                                                   test_mat[INPUT][0], timestamp, duration );
    double err_level = get_success_error_level( test_case_idx, 0, 0 );
    int code = cvtest::TS::OK;
    int nz = cvCountNonZero( test_array[INPUT][1] );

    if( nz > 32 && !(min_angle - err_level <= angle &&
          max_angle + err_level >= angle) &&
        !(min_angle - err_level <= angle+360 &&
          max_angle + err_level >= angle+360) )
    {
        ts->printf( cvtest::TS::LOG, "The angle=%g is outside (%g,%g) range\n",
                    angle, min_angle - err_level, max_angle + err_level );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }
    else if( fabs(angle - ref_angle) > err_level &&
             fabs(360 - fabs(angle - ref_angle)) > err_level )
    {
        ts->printf( cvtest::TS::LOG, "The angle=%g differs too much from reference value=%g\n",
                    angle, ref_angle );
        code = cvtest::TS::FAIL_BAD_ACCURACY;
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


TEST(Video_MHIUpdate, accuracy) { CV_UpdateMHITest test; test.safe_run(); }
TEST(Video_MHIGradient, accuracy) { CV_MHIGradientTest test; test.safe_run(); }
TEST(Video_MHIGlobalOrient, accuracy) { CV_MHIGlobalOrientTest test; test.safe_run(); }
