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

#include "cvtest.h"

static const int motempl_silh_ratio[] = { 10, 50 };
static const int motempl_duration[] = { 200, 2000 };
static const int motempl_gradient_aperture[] = { 3, 5 };
static const char* motempl_update_param_names[] = { "silh_ratio", "duration", "size", 0 };
static const char* motempl_gradient_param_names[] = { "silh_ratio", "duration", "aperture", "size", 0 };
static const char* motempl_global_param_names[] = { "silh_ratio", "duration", "size", 0 };
static const CvSize motempl_sizes[] = {{320, 240}, {720,480}, {-1,-1}};

///////////////////// base MHI class ///////////////////////
class CV_MHIBaseTest : public CvArrTest
{
public:
    CV_MHIBaseTest( const char* test_name, const char* test_funcs );

protected:
    int write_default_params(CvFileStorage* fs);
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    int prepare_test_case( int test_case_idx );
    double timestamp, duration, max_log_duration;
    int mhi_i, mhi_ref_i;
    double silh_ratio;
};


CV_MHIBaseTest::CV_MHIBaseTest( const char* test_name, const char* test_funcs )
    : CvArrTest( test_name, test_funcs )
{
    timestamp = duration = 0;
    max_log_duration = 9;
    mhi_i = mhi_ref_i = -1;
    
    size_list = whole_size_list = strcmp( test_funcs, "" ) == 0 ? motempl_sizes : 0;
    depth_list = 0;
    cn_list = 0;
    default_timing_param_names = 0;

    silh_ratio = 0.25;
}


int CV_MHIBaseTest::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE && strcmp(tested_functions, "") == 0 )
    {
        start_write_param( fs );        
        write_int_list( fs, "silh_ratio", motempl_silh_ratio, CV_DIM(motempl_silh_ratio) );
        write_int_list( fs, "duration", motempl_duration, CV_DIM(motempl_duration) );
    }

    return code;
}


void CV_MHIBaseTest::get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high )
{
    CvArrTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT && CV_MAT_DEPTH(type) == CV_8U )
    {
        *low = cvScalarAll(cvRound(-1./silh_ratio)+2.);
        *high = cvScalarAll(2);
    }
    else if( i == mhi_i || i == mhi_ref_i )
    {
        *low = cvScalarAll(-exp(max_log_duration));
        *high = cvScalarAll(0.);
    }
}


void CV_MHIBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[INPUT][0] = CV_8UC1;
    types[mhi_i][0] = types[mhi_ref_i][0] = CV_32FC1;
    duration = exp(cvTsRandReal(rng)*max_log_duration);
    timestamp = duration + cvTsRandReal(rng)*30.-10.;
}


void CV_MHIBaseTest::get_timing_test_array_types_and_sizes( int test_case_idx,
            CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    types[INPUT][0] = CV_8UC1;
    types[mhi_i][0] = CV_32FC1;
    duration = cvReadInt( find_timing_param( "duration" ), 500 );
    silh_ratio = cvReadInt( find_timing_param( "silh_ratio" ), 25 )*0.01;
    timestamp = duration;
}


void CV_MHIBaseTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "ratio=%d%%,duration=%dms,", cvRound(silh_ratio*100), cvRound(duration) );
    ptr += strlen(ptr);
    params_left -= 2;

    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


int CV_MHIBaseTest::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        CvMat* mat = &test_mat[mhi_i][0];
        cvTsAdd( mat, cvScalarAll(1.), 0, cvScalarAll(0.), cvScalarAll(duration), mat, 0 ); 
        cvTsMinMaxS( mat, 0, mat, CV_TS_MAX );
        if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE && mhi_i != mhi_ref_i )
        {
            CvMat* mat0 = &test_mat[mhi_ref_i][0];
            cvTsCopy( mat, mat0 );
        }
    }

    return code;
}


CV_MHIBaseTest mhi_base_test( "mhi", "" );


///////////////////// update motion history ////////////////////////////

static void cvTsUpdateMHI( const CvMat* silh, CvMat* mhi, double timestamp, double duration )
{
    int i, j;
    float delbound = (float)(timestamp - duration);
    for( i = 0; i < mhi->rows; i++ )
    {
        const uchar* silh_row = silh->data.ptr + i*silh->step;
        float* mhi_row = (float*)(mhi->data.ptr + i*mhi->step);

        for( j = 0; j < mhi->cols; j++ )
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
    : CV_MHIBaseTest( "mhi-update", "cvUpdateMotionHistory" )
{
    test_array[INPUT].push(NULL);
    test_array[INPUT_OUTPUT].push(NULL);
    test_array[REF_INPUT_OUTPUT].push(NULL);
    mhi_i = INPUT_OUTPUT; mhi_ref_i = REF_INPUT_OUTPUT;

    default_timing_param_names = motempl_update_param_names;
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
    cvTsUpdateMHI( &test_mat[INPUT][0], &test_mat[REF_INPUT_OUTPUT][0], timestamp, duration );
}


CV_UpdateMHITest mhi_update_test;


///////////////////// calc motion gradient ////////////////////////////

static void cvTsMHIGradient( const CvMat* mhi, CvMat* mask, CvMat* orientation,
                             double delta1, double delta2, int aperture_size )
{
    CvPoint anchor = { aperture_size/2, aperture_size/2 };
    CvMat* src = cvCreateMat( mhi->rows + aperture_size - 1, mhi->cols + aperture_size - 1, CV_32FC1 );
    CvMat* kernel = cvCreateMat( aperture_size, aperture_size, CV_32FC1 );
    CvMat* dx = cvCreateMat( mhi->rows, mhi->cols, CV_32FC1 );
    CvMat* dy = cvCreateMat( mhi->rows, mhi->cols, CV_32FC1 );
    CvMat* min_mhi = cvCreateMat( mhi->rows, mhi->cols, CV_32FC1 );
    CvMat* max_mhi = cvCreateMat( mhi->rows, mhi->cols, CV_32FC1 );
    IplConvKernel* element = cvCreateStructuringElementEx( aperture_size, aperture_size,
                                        aperture_size/2, aperture_size/2, CV_SHAPE_RECT );
    int i, j;
    double limit = 1e-4*aperture_size*aperture_size;
    
    cvTsPrepareToFilter( mhi, src, anchor );
    cvTsCalcSobelKernel2D( 1, 0, aperture_size, 0, kernel );
    cvTsConvolve2D( src, dx, kernel, anchor );
    cvTsCalcSobelKernel2D( 0, 1, aperture_size, 0, kernel );
    cvTsConvolve2D( src, dy, kernel, anchor );
    cvReleaseMat( &kernel );

    cvTsMinMaxFilter( src, min_mhi, element, CV_TS_MIN );
    cvTsMinMaxFilter( src, max_mhi, element, CV_TS_MAX );
    cvReleaseMat( &src );
    cvReleaseStructuringElement( &element );

    if( delta1 > delta2 )
    {
        double t;
        CV_SWAP( delta1, delta2, t );
    }

    for( i = 0; i < mhi->rows; i++ )
    {
        uchar* mask_row = mask->data.ptr + i*mask->step;
        float* orient_row = (float*)(orientation->data.ptr + i*orientation->step);
        const float* dx_row = (float*)(dx->data.ptr + i*dx->step);
        const float* dy_row = (float*)(dy->data.ptr + i*dy->step);
        const float* min_row = (float*)(min_mhi->data.ptr + i*min_mhi->step);
        const float* max_row = (float*)(max_mhi->data.ptr + i*max_mhi->step);

        for( j = 0; j < mhi->cols; j++ )
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

    cvReleaseMat( &dx );
    cvReleaseMat( &dy );
    cvReleaseMat( &min_mhi );
    cvReleaseMat( &max_mhi );
}


class CV_MHIGradientTest : public CV_MHIBaseTest
{
public:
    CV_MHIGradientTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );
    int write_default_params(CvFileStorage* fs);

    double delta1, delta2, delta_range_log;
    int aperture_size;
};


CV_MHIGradientTest::CV_MHIGradientTest()
    : CV_MHIBaseTest( "mhi-gradient", "cvCalcMotionGradient" )
{
    mhi_i = mhi_ref_i = INPUT;
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    delta1 = delta2 = 0;
    aperture_size = 0;
    delta_range_log = 4;

    default_timing_param_names = motempl_gradient_param_names;
}


int CV_MHIGradientTest::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        start_write_param( fs );        
        write_int_list( fs, "aperture", motempl_gradient_aperture, CV_DIM(motempl_gradient_aperture) );
    }

    return code;
}


void CV_MHIGradientTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_MHIBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8UC1;
    types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_32FC1;
    delta1 = exp(cvTsRandReal(rng)*delta_range_log + 1.);
    delta2 = exp(cvTsRandReal(rng)*delta_range_log + 1.);
    aperture_size = (cvTsRandInt(rng)%3)*2+3;
    //duration = exp(cvTsRandReal(rng)*max_log_duration);
    //timestamp = duration + cvTsRandReal(rng)*30.-10.;
}


void CV_MHIGradientTest::get_timing_test_array_types_and_sizes( int test_case_idx,
            CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_MHIBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                           whole_sizes, are_images );
    types[OUTPUT][0] = CV_8UC1;
    types[OUTPUT][1] = CV_32FC1;
    aperture_size = cvReadInt( find_timing_param( "aperture" ), 3 );
    delta1 = duration*0.02;
    delta2 = duration*0.2;
}


void CV_MHIGradientTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "aperture=%d,", aperture_size );
    ptr += strlen(ptr);
    params_left--;

    CV_MHIBaseTest::print_timing_params( test_case_idx, ptr, params_left );
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
    cvTsMHIGradient( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0],
                     &test_mat[REF_OUTPUT][1], delta1, delta2, aperture_size );
    cvTsAdd( &test_mat[REF_OUTPUT][1], cvScalarAll(1.), 0, cvScalarAll(0.),
             cvScalarAll(1.), &test_mat[REF_OUTPUT][1], 0 );
    cvTsAdd( &test_mat[OUTPUT][1], cvScalarAll(1.), 0, cvScalarAll(0.),
             cvScalarAll(1.), &test_mat[OUTPUT][1], 0 );
}


CV_MHIGradientTest mhi_gradient_test;


////////////////////// calc global orientation /////////////////////////

static double
cvTsCalcGlobalOrientation( const CvMat* orient, const CvMat* mask, const CvMat* mhi,
                           double timestamp, double duration )
{
    const int HIST_SIZE = 12;
    int      y, x;
    int      histogram[HIST_SIZE];
    int      max_bin = 0;

    double   base_orientation = 0, delta_orientation = 0, weight = 0;
    double   low_time, global_orientation;

    memset( histogram, 0, sizeof( histogram ));
    timestamp = 0;

    for( y = 0; y < orient->rows; y++ )
    {
        const float* orient_data = (const float*)(orient->data.ptr + y*orient->step);
        const uchar* mask_data = mask->data.ptr + y*mask->step;
        const float* mhi_data = (const float*)(mhi->data.ptr + y*mhi->step);
        for( x = 0; x < orient->cols; x++ )
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

    for( y = 0; y < orient->rows; y++ )
    {
        const float* orient_data = (const float*)(orient->data.ptr + y*orient->step);
        const float* mhi_data = (const float*)(mhi->data.ptr + y*mhi->step);
        const uchar* mask_data = mask->data.ptr + y*mask->step;
        
        for( x = 0; x < orient->cols; x++ )
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
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int validate_test_results( int test_case_idx );
    void run_func();
    double angle, min_angle, max_angle;
};


CV_MHIGlobalOrientTest::CV_MHIGlobalOrientTest()
    : CV_MHIBaseTest( "mhi-global", "cvCalcGlobalOrientation" )
{
    mhi_i = mhi_ref_i = INPUT;
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    min_angle = max_angle = 0;

    default_timing_param_names = motempl_global_param_names;
}


void CV_MHIGlobalOrientTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_MHIBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize size = sizes[INPUT][0];

    size.width = MAX( size.width, 16 );
    size.height = MAX( size.height, 16 );
    sizes[INPUT][0] = sizes[INPUT][1] = sizes[INPUT][2] = size;

    types[INPUT][1] = CV_8UC1; // mask
    types[INPUT][2] = CV_32FC1; // orientation

    min_angle = cvTsRandReal(rng)*359.9;
    max_angle = cvTsRandReal(rng)*359.9;
    if( min_angle >= max_angle )
    {
        double t;
        CV_SWAP( min_angle, max_angle, t );
    }
    max_angle += 0.1;
    duration = exp(cvTsRandReal(rng)*max_log_duration);
    timestamp = duration + cvTsRandReal(rng)*30.-10.;
}


void CV_MHIGlobalOrientTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_MHIBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                           whole_sizes, are_images );
    types[INPUT][1] = CV_8UC1;
    types[INPUT][2] = CV_32FC1;
}


void CV_MHIGlobalOrientTest::get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high )
{
    CV_MHIBaseTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT && j == 2 )
    {
        *low = cvScalarAll(min_angle);
        *high = cvScalarAll(max_angle);
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
    
    double ref_angle = cvTsCalcGlobalOrientation( &test_mat[INPUT][2], &test_mat[INPUT][1],
                                                  &test_mat[INPUT][0], timestamp, duration );
    double err_level = get_success_error_level( test_case_idx, 0, 0 );
    int code = CvTS::OK;
    int nz = cvCountNonZero( test_array[INPUT][1] );

    if( nz > 32 && !(min_angle - err_level <= angle &&
          max_angle + err_level >= angle) &&
        !(min_angle - err_level <= angle+360 &&
          max_angle + err_level >= angle+360) )
    {
        ts->printf( CvTS::LOG, "The angle=%g is outside (%g,%g) range\n",
                    angle, min_angle - err_level, max_angle + err_level );
        code = CvTS::FAIL_BAD_ACCURACY;
    }
    else if( fabs(angle - ref_angle) > err_level &&
             fabs(360 - fabs(angle - ref_angle)) > err_level )
    {
        ts->printf( CvTS::LOG, "The angle=%g differs too much from reference value=%g\n",
                    angle, ref_angle );
        code = CvTS::FAIL_BAD_ACCURACY;
    }

    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_MHIGlobalOrientTest mhi_global_orient_test;

