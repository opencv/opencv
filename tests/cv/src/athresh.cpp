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

static const char* thresh_param_names[] = { "size", "depth", "thresh_type", 0 };
static const CvSize thresh_sizes[] = {{30,30}, {320, 240}, {720,480}, {-1,-1}};
static const CvSize thresh_whole_sizes[] = {{320,240}, {320, 240}, {720,480}, {-1,-1}};
static const int thresh_depths[] = { CV_8U, CV_32F, -1 };
static const char* thresh_types[] = { "binary", "binary_inv", "trunc", "tozero", "tozero_inv", 0 };

class CV_ThreshTest : public CvArrTest
{
public:
    CV_ThreshTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );
    
    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );

    int thresh_type;
    float thresh_val;
    float max_val;
};


CV_ThreshTest::CV_ThreshTest()
    : CvArrTest( "thresh-simple", "cvThreshold", "" )
{
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    optional_mask = false;
    element_wise_relative_error = true;

    default_timing_param_names = thresh_param_names;
    depth_list = thresh_depths;
    size_list = thresh_sizes;
    whole_size_list = thresh_whole_sizes;
    cn_list = 0;
}


void CV_ThreshTest::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % 2, cn = cvTsRandInt(rng) % 4 + 1;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : CV_32F;

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth,cn);
    thresh_type = cvTsRandInt(rng) % 5;

    if( depth == CV_8U )
    {
        thresh_val = (float)(cvTsRandReal(rng)*350. - 50.);
        max_val = (float)(cvTsRandReal(rng)*350. - 50.);
        if( cvTsRandInt(rng)%4 == 0 )
            max_val = 255;
    }
    else
    {
        thresh_val = (float)(cvTsRandReal(rng)*1000. - 500.);
        max_val = (float)(cvTsRandReal(rng)*1000. - 500.);
    }
}


double CV_ThreshTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return FLT_EPSILON*10;
}


void CV_ThreshTest::run_func()
{
    cvThreshold( test_array[INPUT][0], test_array[OUTPUT][0],
                 thresh_val, max_val, thresh_type );
}


static void cvTsThreshold( const CvMat* _src, CvMat* _dst,
                           float thresh, float maxval, int thresh_type )
{
    int i, j;
    int depth = CV_MAT_DEPTH(_src->type), cn = CV_MAT_CN(_src->type);
    int width_n = _src->cols*cn, height = _src->rows;
    int ithresh = cvFloor(thresh), ithresh2, imaxval = cvRound(maxval);
    const uchar* src = _src->data.ptr;
    uchar* dst = _dst->data.ptr;
    
    ithresh2 = CV_CAST_8U(ithresh);
    imaxval = CV_CAST_8U(imaxval);

    assert( depth == CV_8U || depth == CV_32F );
    
    switch( thresh_type )
    {
    case CV_THRESH_BINARY:
        for( i = 0; i < height; i++, src += _src->step, dst += _dst->step )
        {
            if( depth == CV_8U )
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? imaxval : 0);
            else
                for( j = 0; j < width_n; j++ )
                    ((float*)dst)[j] = ((const float*)src)[j] > thresh ? maxval : 0.f;
        }
        break;
    case CV_THRESH_BINARY_INV:
        for( i = 0; i < height; i++, src += _src->step, dst += _dst->step )
        {
            if( depth == CV_8U )
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? 0 : imaxval);
            else
                for( j = 0; j < width_n; j++ )
                    ((float*)dst)[j] = ((const float*)src)[j] > thresh ? 0.f : maxval;
        }
        break;
    case CV_THRESH_TRUNC:
        for( i = 0; i < height; i++, src += _src->step, dst += _dst->step )
        {
            if( depth == CV_8U )
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? ithresh2 : s);
                }
            else
                for( j = 0; j < width_n; j++ )
                {
                    float s = ((const float*)src)[j];
                    ((float*)dst)[j] = s > thresh ? thresh : s;
                }
        }
        break;
    case CV_THRESH_TOZERO:
        for( i = 0; i < height; i++, src += _src->step, dst += _dst->step )
        {
            if( depth == CV_8U )
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? s : 0);
                }
            else
                for( j = 0; j < width_n; j++ )
                {
                    float s = ((const float*)src)[j];
                    ((float*)dst)[j] = s > thresh ? s : 0.f;
                }
        }
        break;
    case CV_THRESH_TOZERO_INV:
        for( i = 0; i < height; i++, src += _src->step, dst += _dst->step )
        {
            if( depth == CV_8U )
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? 0 : s);
                }
            else
                for( j = 0; j < width_n; j++ )
                {
                    float s = ((const float*)src)[j];
                    ((float*)dst)[j] = s > thresh ? 0.f : s;
                }
        }
        break;
    default:
        assert(0);
    }
}


void CV_ThreshTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsThreshold( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0],
                   thresh_val, max_val, thresh_type );
}


int CV_ThreshTest::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code < 0 )
        return code;
    
    if( ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        start_write_param( fs );        
        write_string_list( fs, "thresh_type", thresh_types );
    }

    return code;
}


void CV_ThreshTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    const char* thresh_str = cvReadString( find_timing_param( "thresh_type" ), "binary" );
    thresh_type = strcmp( thresh_str, "binary" ) == 0 ? CV_THRESH_BINARY :
        strcmp( thresh_str, "binary_inv" ) == 0 ? CV_THRESH_BINARY_INV :
        strcmp( thresh_str, "trunc" ) == 0 ? CV_THRESH_TRUNC :
        strcmp( thresh_str, "tozero" ) == 0 ? CV_THRESH_TOZERO :
        CV_THRESH_TOZERO_INV;

    if( CV_MAT_DEPTH(types[INPUT][0]) == CV_8U )
    {
        thresh_val = 128;
        max_val = 255;
    }
    else
    {
        thresh_val = 500.;
        max_val = 1.;
    }
}


void CV_ThreshTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", cvReadString( find_timing_param( "thresh_type" ), "binary" ) );
    ptr += strlen(ptr);
    params_left--;

    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


CV_ThreshTest thresh_test;

