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

static const char* accum_param_names[] = { "size", "channels", "depth", "use_mask", 0 };
static const CvSize accum_sizes[] = {{30,30}, {320, 240}, {720,480}, {-1,-1}};
static const CvSize accum_whole_sizes[] = {{320,240}, {320, 240}, {720,480}, {-1,-1}};
static const int accum_depths[] = { CV_8U, CV_32F, CV_64F, -1 };
static const int accum_channels[] = { 1, 3, -1 };

class CV_AccumBaseTestImpl : public CvArrTest
{
public:
    CV_AccumBaseTestImpl( const char* test_name, const char* test_funcs );

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    double alpha;
};


CV_AccumBaseTestImpl::CV_AccumBaseTestImpl( const char* test_name, const char* test_funcs )
    : CvArrTest( test_name, test_funcs, "" )
{
    test_array[INPUT].push(NULL);
    test_array[INPUT_OUTPUT].push(NULL);
    test_array[REF_INPUT_OUTPUT].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[MASK].push(NULL);
    optional_mask = true;
    element_wise_relative_error = false;

    default_timing_param_names = 0;
    depth_list = accum_depths;
    size_list = accum_sizes;
    whole_size_list = accum_whole_sizes;
    cn_list = accum_channels;
}


void CV_AccumBaseTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % 3, cn = cvTsRandInt(rng) & 1 ? 3 : 1;
    int accdepth = std::max((int)(cvTsRandInt(rng) % 2 + 1), depth);
    int i, input_count = test_array[INPUT].size();
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_32F : CV_64F;
    accdepth = accdepth == 1 ? CV_32F : CV_64F;
    accdepth = MAX(accdepth, depth);

    for( i = 0; i < input_count; i++ )
        types[INPUT][i] = CV_MAKETYPE(depth,cn);
    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = types[TEMP][0] = CV_MAKETYPE(accdepth,cn);

    alpha = cvTsRandReal(rng);
}


double CV_AccumBaseTestImpl::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return CV_MAT_DEPTH(test_mat[INPUT_OUTPUT][0].type) < CV_64F ||
        CV_MAT_DEPTH(test_mat[INPUT][0].type) == CV_32F ? FLT_EPSILON*100 : DBL_EPSILON*1000;
}


void CV_AccumBaseTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    types[INPUT_OUTPUT][0] = CV_MAKETYPE(MAX(CV_32F, CV_MAT_DEPTH(types[INPUT][0])),
        CV_MAT_CN(types[INPUT][0]));
    alpha = 0.333333333333333;
}


CV_AccumBaseTestImpl accum_base( "accum", "" );


class CV_AccumBaseTest : public CV_AccumBaseTestImpl
{
public:
    CV_AccumBaseTest( const char* test_name, const char* test_funcs );
};


CV_AccumBaseTest::CV_AccumBaseTest( const char* test_name, const char* test_funcs )
    : CV_AccumBaseTestImpl( test_name, test_funcs )
{
    depth_list = 0;
    size_list = 0;
    whole_size_list = 0;
    cn_list = 0;

    default_timing_param_names = accum_param_names;
}


/// acc
class CV_AccTest : public CV_AccumBaseTest
{
public:
    CV_AccTest();
protected:
    void run_func();
    void prepare_to_validation( int );
};

CV_AccTest::CV_AccTest()
    : CV_AccumBaseTest( "accum-acc", "cvAcc" )
{
}


void CV_AccTest::run_func()
{
    cvAcc( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
}


void CV_AccTest::prepare_to_validation( int )
{
    const CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* temp = &test_mat[TEMP][0];
    const CvMat* mask = test_array[MASK][0] ? &test_mat[MASK][0] : 0;

    cvTsAdd( src, cvScalarAll(1.), dst, cvScalarAll(1.), cvScalarAll(0.), temp, 0 );
    cvTsCopy( temp, dst, mask );
}

CV_AccTest acc_test;


/// square acc
class CV_SquareAccTest : public CV_AccumBaseTest
{
public:
    CV_SquareAccTest();
protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_SquareAccTest::CV_SquareAccTest()
    : CV_AccumBaseTest( "accum-squareacc", "cvSquareAcc" )
{
}


void CV_SquareAccTest::run_func()
{
    cvSquareAcc( test_array[INPUT][0], test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
}


void CV_SquareAccTest::prepare_to_validation( int )
{
    const CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* temp = &test_mat[TEMP][0];
    const CvMat* mask = test_array[MASK][0] ? &test_mat[MASK][0] : 0;

    cvTsMul( src, src, cvScalarAll(1.), temp );
    cvTsAdd( temp, cvScalarAll(1.), dst, cvScalarAll(1.), cvScalarAll(0.), temp, 0 );
    cvTsCopy( temp, dst, mask );
}

CV_SquareAccTest squareacc_test;


/// multiply acc
class CV_MultiplyAccTest : public CV_AccumBaseTest
{
public:
    CV_MultiplyAccTest();
protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_MultiplyAccTest::CV_MultiplyAccTest()
    : CV_AccumBaseTest( "accum-mulacc", "cvMultiplyAcc" )
{
    test_array[INPUT].push(NULL);
}


void CV_MultiplyAccTest::run_func()
{
    cvMultiplyAcc( test_array[INPUT][0], test_array[INPUT][1],
                   test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
}


void CV_MultiplyAccTest::prepare_to_validation( int )
{
    const CvMat* src1 = &test_mat[INPUT][0];
    const CvMat* src2 = &test_mat[INPUT][1];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* temp = &test_mat[TEMP][0];
    const CvMat* mask = test_array[MASK][0] ? &test_mat[MASK][0] : 0;

    cvTsMul( src1, src2, cvScalarAll(1.), temp );
    cvTsAdd( temp, cvScalarAll(1.), dst, cvScalarAll(1.), cvScalarAll(0.), temp, 0 );
    cvTsCopy( temp, dst, mask );
}

CV_MultiplyAccTest mulacc_test;


/// running average
class CV_RunningAvgTest : public CV_AccumBaseTest
{
public:
    CV_RunningAvgTest();
protected:
    void run_func();
    void prepare_to_validation( int );
};


CV_RunningAvgTest::CV_RunningAvgTest()
    : CV_AccumBaseTest( "accum-runavg", "cvRunningAvg" )
{
}


void CV_RunningAvgTest::run_func()
{
    cvRunningAvg( test_array[INPUT][0], test_array[INPUT_OUTPUT][0],
                  alpha, test_array[MASK][0] );
}


void CV_RunningAvgTest::prepare_to_validation( int )
{
    const CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_INPUT_OUTPUT][0];
    CvMat* temp = &test_mat[TEMP][0];
    const CvMat* mask = test_array[MASK][0] ? &test_mat[MASK][0] : 0;
    double a[1], b[1];
    int accdepth = CV_MAT_DEPTH(test_mat[INPUT_OUTPUT][0].type);
    CvMat A = cvMat(1,1,accdepth,a), B = cvMat(1,1,accdepth,b);
    cvSetReal1D( &A, 0, alpha);
    cvSetReal1D( &B, 0, 1 - cvGetReal1D(&A, 0));

    cvTsAdd( src, cvScalarAll(cvGetReal1D(&A, 0)), dst, cvScalarAll(cvGetReal1D(&B, 0)), cvScalarAll(0.), temp, 0 );
    cvTsCopy( temp, dst, mask );
}

CV_RunningAvgTest runavg_test;

