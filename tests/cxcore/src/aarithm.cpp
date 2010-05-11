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

//////////////////////////////////////////////////////////////////////////////////////////
////////////////// tests for arithmetic, logic and statistical functions /////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "cxcoretest.h"
#include <float.h>

static const CvSize arithm_sizes[] = {{10,10}, {100,100}, {720,480}, {-1,-1}};
static const CvSize arithm_whole_sizes[] = {{10,10}, {720,480}, {720,480}, {-1,-1}};
static const int arithm_depths[] = { CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, -1 };
static const int arithm_channels[] = { 1, 2, 3, 4, -1 };
static const char* arithm_mask_param_names[] = { "size", "channels", "depth", "use_mask", 0 };
static const char* arithm_param_names[] = { "size", "channels", "depth", 0 };
static const char* minmax_param_names[] = { "size", "depth", 0 };

class CxCore_ArithmTestImpl : public CvArrTest
{
public:
    CxCore_ArithmTestImpl( const char* test_name, const char* test_funcs,
                           int _generate_scalars=0, bool _allow_mask=true, bool _calc_abs=false );
protected:
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int /*test_case_idx*/,
                        CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images );
    void generate_scalars( int depth );
    void finalize_scalar( CvScalar& s );
    CvScalar alpha, beta, gamma;
    int gen_scalars;
    bool calc_abs;
    bool test_nd;
};


CxCore_ArithmTestImpl::CxCore_ArithmTestImpl( const char* test_name, const char* test_funcs,
                                              int _generate_scalars, bool _allow_mask, bool _calc_abs )
    : CvArrTest( test_name, test_funcs, "" ),
    gen_scalars(_generate_scalars), calc_abs(_calc_abs)
{
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    optional_mask = _allow_mask;

    if( optional_mask )
    {
        test_array[INPUT_OUTPUT].push(NULL);
        test_array[REF_INPUT_OUTPUT].push(NULL);
        test_array[TEMP].push(NULL);
        test_array[MASK].push(NULL);
    }
    else
    {
        test_array[OUTPUT].push(NULL);
        test_array[REF_OUTPUT].push(NULL);
    }
    alpha = beta = gamma = cvScalarAll(0);

    size_list = arithm_sizes;
    whole_size_list = arithm_whole_sizes;
    depth_list = arithm_depths;
    cn_list = arithm_channels;
    test_nd = false;
}


void CxCore_ArithmTestImpl::generate_scalars( int depth )
{
    bool is_timing = ts->get_testing_mode() == CvTS::TIMING_MODE;
    double ab_min_val = -1.;
    double ab_max_val = 1.;
    double gamma_min_val = depth == CV_8U ? -100 : depth < CV_32F ? -10000 : -1e6;
    double gamma_max_val = depth == CV_8U ? 100 : depth < CV_32F ? 10000 : 1e6;
    
    if( gen_scalars )
    {
        CvRNG* rng = ts->get_rng();
        int i;
        double m = 3.;
        for( i = 0; i < 4; i++ )
        {
            if( gen_scalars & 1 )
            {
                alpha.val[i] = exp((cvTsRandReal(rng)-0.5)*m*2*CV_LOG2);
                alpha.val[i] *= (cvTsRandInt(rng) & 1) ? 1 : -1;
                if( is_timing )
                {
                    alpha.val[i] = MAX( alpha.val[i], ab_min_val );
                    alpha.val[i] = MIN( alpha.val[i], ab_max_val );
                }
            }
            if( gen_scalars & 2 )
            {
                beta.val[i] = exp((cvTsRandReal(rng)-0.5)*m*2*CV_LOG2);
                beta.val[i] *= (cvTsRandInt(rng) & 1) ? 1 : -1;
                if( is_timing )
                {
                    beta.val[i] = MAX( beta.val[i], ab_min_val );
                    beta.val[i] = MIN( beta.val[i], ab_max_val );
                }
            }
            if( gen_scalars & 4 )
            {
                gamma.val[i] = exp((cvTsRandReal(rng)-0.5)*m*2*CV_LOG2);
                gamma.val[i] *= (cvTsRandInt(rng) & 1) ? 1 : -1;
                if( is_timing )
                {
                    gamma.val[i] = MAX( gamma.val[i], gamma_min_val );
                    gamma.val[i] = MIN( gamma.val[i], gamma_max_val );
                }
            }
        }
    }

    if( depth == CV_32F )
    {
        CvMat fl = cvMat( 1, 4, CV_32F, buf );
        CvMat db = cvMat( 1, 4, CV_64F, 0 );

        db.data.db = alpha.val;
        cvTsConvert( &db, &fl );
        cvTsConvert( &fl, &db );

        db.data.db = beta.val;
        cvTsConvert( &db, &fl );
        cvTsConvert( &fl, &db );

        db.data.db = gamma.val;
        cvTsConvert( &db, &fl );
        cvTsConvert( &fl, &db );
    }
}

void CxCore_ArithmTestImpl::finalize_scalar( CvScalar& s )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    if( depth < CV_32F )
        s = cvScalar(cvRound(s.val[0]), cvRound(s.val[1]), cvRound(s.val[2]), cvRound(s.val[3]));
}

void CxCore_ArithmTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                            CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng)%(CV_64F+1);
    int cn = cvTsRandInt(rng) % 4 + 1;
    int i, j;
    depth += depth == CV_8S;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    generate_scalars( depth );

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        int type = i != MASK ? CV_MAKETYPE(depth, cn) : CV_8UC1;
        for( j = 0; j < count; j++ )
        {
            types[i][j] = type;
        }
    }
    test_nd = cvTsRandInt(rng)%3 == 0;
}


void CxCore_ArithmTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    generate_scalars( types[INPUT][0] );
    test_nd = false;
}


void CxCore_ArithmTestImpl::prepare_to_validation( int /*test_case_idx*/ )
{
    const CvMat* mask = test_array[MASK].size() > 0 && test_array[MASK][0] ? &test_mat[MASK][0] : 0;
    CvMat* output = test_array[REF_INPUT_OUTPUT].size() > 0 ?
        &test_mat[REF_INPUT_OUTPUT][0] : &test_mat[REF_OUTPUT][0];
    CvMat* temp_dst = mask ? &test_mat[TEMP][0] : output;
    cvTsAdd( &test_mat[INPUT][0], alpha,
             test_array[INPUT].size() > 1 ? &test_mat[INPUT][1] : 0, beta,
             gamma, temp_dst, calc_abs );
    if( mask )
        cvTsCopy( temp_dst, output, mask );
}


CxCore_ArithmTestImpl arithm( "arithm", "", 0, false );


class CxCore_ArithmTest : public CxCore_ArithmTestImpl
{
public:
    CxCore_ArithmTest( const char* test_name, const char* test_funcs,
                       int _generate_scalars=0, bool _allow_mask=true, bool _calc_abs=false );
};


CxCore_ArithmTest::CxCore_ArithmTest( const char* test_name, const char* test_funcs,
                                      int _generate_scalars, bool _allow_mask, bool _calc_abs ) :
    CxCore_ArithmTestImpl( test_name, test_funcs, _generate_scalars, _allow_mask, _calc_abs )
{
    default_timing_param_names = optional_mask ? arithm_mask_param_names : arithm_param_names;
        
    // inherit the default parameters from arithmetical test
    size_list = 0;
    whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;
}


////////////////////////////// add /////////////////////////////

class CxCore_AddTest : public CxCore_ArithmTest
{
public:
    CxCore_AddTest();
protected:
    void run_func();
};

CxCore_AddTest::CxCore_AddTest()
    : CxCore_ArithmTest( "arithm-add", "cvAdd", 0, true )
{
    alpha = beta = cvScalarAll(1.);
}

void CxCore_AddTest::run_func()
{
    if(!test_nd)
    {
        cvAdd( test_array[INPUT][0], test_array[INPUT][1],
            test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND b = cv::cvarrToMatND(test_array[INPUT][1]);
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        if( !test_array[MASK][0] )
            cv::add(a, b, c);
        else
            cv::add(a, b, c, cv::cvarrToMatND(test_array[MASK][0]));
    }
}

CxCore_AddTest add_test;

////////////////////////////// sub /////////////////////////////

class CxCore_SubTest : public CxCore_ArithmTest
{
public:
    CxCore_SubTest();
protected:
    void run_func();
};

CxCore_SubTest::CxCore_SubTest()
    : CxCore_ArithmTest( "arithm-sub", "cvSub", 0, true )
{
    alpha = cvScalarAll(1.);
    beta = cvScalarAll(-1.);
}

void CxCore_SubTest::run_func()
{
    if(!test_nd)
    {
        cvSub( test_array[INPUT][0], test_array[INPUT][1],
            test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND b = cv::cvarrToMatND(test_array[INPUT][1]);
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        if( !test_array[MASK][0] )
            cv::subtract(a, b, c);
        else
            cv::subtract(a, b, c, cv::cvarrToMatND(test_array[MASK][0]));
    }
}

CxCore_SubTest sub_test;


////////////////////////////// adds /////////////////////////////

class CxCore_AddSTest : public CxCore_ArithmTest
{
public:
    CxCore_AddSTest();
protected:
    void run_func();
};

CxCore_AddSTest::CxCore_AddSTest()
    : CxCore_ArithmTest( "arithm-adds", "cvAddS", 4, true )
{
    test_array[INPUT].pop();
    alpha = cvScalarAll(1.);
}

void CxCore_AddSTest::run_func()
{
    finalize_scalar(gamma);
    if(!test_nd)
    {
        if( test_mat[INPUT][0].cols % 2 == 0 )
            cvAddS( test_array[INPUT][0], gamma,
                test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
        else
        {
            cv::Mat a = cv::cvarrToMat(test_array[INPUT][0]),
                c = cv::cvarrToMat(test_array[INPUT_OUTPUT][0]);
                cv::subtract(a, -cv::Scalar(gamma), c, test_array[MASK][0] ?
                    cv::cvarrToMat(test_array[MASK][0]) : cv::Mat());
        }
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::add( cv::cvarrToMatND(test_array[INPUT][0]),
                 gamma, c, test_array[MASK][0] ?
                 cv::cvarrToMatND(test_array[MASK][0]) : cv::MatND());
    }
}

CxCore_AddSTest adds_test;

////////////////////////////// subrs /////////////////////////////

class CxCore_SubRSTest : public CxCore_ArithmTest
{
public:
    CxCore_SubRSTest();
protected:
    void run_func();
};

CxCore_SubRSTest::CxCore_SubRSTest()
    : CxCore_ArithmTest( "arithm-subrs", "cvSubRS", 4, true )
{
    test_array[INPUT].pop();
    alpha = cvScalarAll(-1.);
}

void CxCore_SubRSTest::run_func()
{
    finalize_scalar(gamma);
    if(!test_nd)
    {
        cvSubRS( test_array[INPUT][0], gamma,
                test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::subtract( gamma,
                cv::cvarrToMatND(test_array[INPUT][0]),
                c, test_array[MASK][0] ?
                    cv::cvarrToMatND(test_array[MASK][0]) : cv::MatND());
    }
}

CxCore_SubRSTest subrs_test;

////////////////////////////// addweighted /////////////////////////////

class CxCore_AddWeightedTest : public CxCore_ArithmTest
{
public:
    CxCore_AddWeightedTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                          CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
};

CxCore_AddWeightedTest::CxCore_AddWeightedTest()
    : CxCore_ArithmTest( "arithm-addweighted", "cvAddWeighted", 7, false )
{
}

void CxCore_AddWeightedTest::get_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types )
{
    CxCore_ArithmTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    alpha = cvScalarAll(alpha.val[0]);
    beta = cvScalarAll(beta.val[0]);
    gamma = cvScalarAll(gamma.val[0]);
}


double CxCore_AddWeightedTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int type = cvGetElemType(test_array[i][j]), depth = CV_MAT_DEPTH(type);
    if( depth <= CV_32S )
        return 2;
    if( depth == CV_32F )
    {
        CvScalar low=cvScalarAll(0), high=low;
        get_minmax_bounds(i,j,type, &low, &high);
        double a = (fabs(alpha.val[0])+fabs(beta.val[0]))*(fabs(low.val[0])+fabs(high.val[0]));
        double b = fabs(gamma.val[0]);
        return (a+b)*500*FLT_EPSILON;
    }
    return CvArrTest::get_success_error_level( test_case_idx, i, j );
}


void CxCore_AddWeightedTest::run_func()
{
    if(!test_nd)
    {
        cvAddWeighted( test_array[INPUT][0], alpha.val[0],
                    test_array[INPUT][1], beta.val[0],
                    gamma.val[0], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::addWeighted(cv::cvarrToMatND(test_array[INPUT][0]),
                alpha.val[0],
                cv::cvarrToMatND(test_array[INPUT][1]),
                beta.val[0], gamma.val[0], c);
    }
}

CxCore_AddWeightedTest addweighted_test;


////////////////////////////// absdiff /////////////////////////////

class CxCore_AbsDiffTest : public CxCore_ArithmTest
{
public:
    CxCore_AbsDiffTest();
protected:
    void run_func();
};

CxCore_AbsDiffTest::CxCore_AbsDiffTest()
    : CxCore_ArithmTest( "arithm-absdiff", "cvAbsDiff", 0, false, true )
{
    alpha = cvScalarAll(1.);
    beta = cvScalarAll(-1.);
}

void CxCore_AbsDiffTest::run_func()
{
    if(!test_nd)
    {
        cvAbsDiff( test_array[INPUT][0], test_array[INPUT][1], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::absdiff(cv::cvarrToMatND(test_array[INPUT][0]),
                cv::cvarrToMatND(test_array[INPUT][1]),
                 c );
    }
}

CxCore_AbsDiffTest absdiff_test;

////////////////////////////// absdiffs /////////////////////////////

class CxCore_AbsDiffSTest : public CxCore_ArithmTest
{
public:
    CxCore_AbsDiffSTest();
protected:
    void run_func();
};

CxCore_AbsDiffSTest::CxCore_AbsDiffSTest()
    : CxCore_ArithmTest( "arithm-absdiffs", "cvAbsDiffS", 4, false, true )
{
    alpha = cvScalarAll(-1.);
    test_array[INPUT].pop();
}

void CxCore_AbsDiffSTest::run_func()
{
    finalize_scalar(gamma);
    if(!test_nd)
    {
        cvAbsDiffS( test_array[INPUT][0], test_array[OUTPUT][0], gamma );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::absdiff(cv::cvarrToMatND(test_array[INPUT][0]),
                gamma, c);
    }
}

CxCore_AbsDiffSTest absdiffs_test;


////////////////////////////// mul /////////////////////////////

static const char* mul_param_names[] = { "size", "scale", "channels", "depth", 0 };
static const char* mul_scale_flags[] = { "scale==1", "scale!=1", 0 };

class CxCore_MulTest : public CxCore_ArithmTest
{
public:
    CxCore_MulTest();
protected:
    void run_func();
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    double get_success_error_level( int test_case_idx, int i, int j );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void prepare_to_validation( int test_case_idx );
    int write_default_params( CvFileStorage* fs );
};


CxCore_MulTest::CxCore_MulTest()
    : CxCore_ArithmTest( "arithm-mul", "cvMul", 4, false, false )
{
    default_timing_param_names = mul_param_names;
}


int CxCore_MulTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_ArithmTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "scale", mul_scale_flags );
    return code;
}


void CxCore_MulTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_ArithmTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* scale_flag_str = cvReadString( find_timing_param( "scale" ), "scale==1" );
    if( strstr( scale_flag_str, "==1" ) )
        alpha.val[0] = 1.;
    else
    {
        double val = alpha.val[0];
        int depth = CV_MAT_DEPTH(types[INPUT][0]);
        if( val == 1. )
            val = 1./CV_PI;
        if( depth == CV_16U || depth == CV_16S || depth == CV_32S )
        {
            double minmax = 1./cvTsMaxVal(depth);
            if( val < -minmax )
                val = -minmax;
            else if( val > minmax )
                val = minmax;
            if( depth == CV_16U && val < 0 )
                val = -val;
        }
        alpha.val[0] = val;
        ts->printf( CvTS::LOG, "alpha = %g\n", alpha.val[0] );
    }
}


void CxCore_MulTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", alpha.val[0] == 1. ? "scale==1" : "scale!=1" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_ArithmTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CxCore_MulTest::get_success_error_level( int test_case_idx, int i, int j )
{
    if( CV_MAT_DEPTH(cvGetElemType(test_array[i][j])) <= CV_32S )
    {
        return gamma.val[0] != cvRound(gamma.val[0]);
    }
    else
        return CvArrTest::get_success_error_level( test_case_idx, i, j );
}


void CxCore_MulTest::run_func()
{
    if(!test_nd)
    {
        cvMul( test_array[INPUT][0], test_array[INPUT][1],
              test_array[OUTPUT][0], alpha.val[0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::multiply(cv::cvarrToMatND(test_array[INPUT][0]),
                     cv::cvarrToMatND(test_array[INPUT][1]),
                     c, alpha.val[0]);
    }
}

void CxCore_MulTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsMul( &test_mat[INPUT][0], &test_mat[INPUT][1],
             cvScalarAll(alpha.val[0]),
             &test_mat[REF_OUTPUT][0] );
}

CxCore_MulTest mul_test;

////////////////////////////// div /////////////////////////////

class CxCore_DivTest : public CxCore_ArithmTest
{
public:
    CxCore_DivTest();
protected:
    void run_func();
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void prepare_to_validation( int /*test_case_idx*/ );
};

CxCore_DivTest::CxCore_DivTest()
    : CxCore_ArithmTest( "arithm-div", "cvDiv", 4, false, false )
{
}

void CxCore_DivTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "s*A(i)/B(i)," );
    ptr += strlen(ptr);
    params_left--;
    CxCore_ArithmTest::print_timing_params( test_case_idx, ptr, params_left );
}

void CxCore_DivTest::run_func()
{
    if(!test_nd)
    {
        cvDiv( test_array[INPUT][0], test_array[INPUT][1],
              test_array[OUTPUT][0], alpha.val[0] );
    }
    else
    {
        cv::MatND b = cv::cvarrToMatND(test_array[INPUT][1]);
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::divide(cv::cvarrToMatND(test_array[INPUT][0]),
                   b, c, alpha.val[0]);
    }
}

void CxCore_DivTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsDiv( &test_mat[INPUT][0], &test_mat[INPUT][1],
             cvScalarAll(alpha.val[0]),
             &test_mat[REF_OUTPUT][0] );
}

CxCore_DivTest div_test;

////////////////////////////// recip /////////////////////////////

class CxCore_RecipTest : public CxCore_ArithmTest
{
public:
    CxCore_RecipTest();
protected:
    void run_func();
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void prepare_to_validation( int /*test_case_idx*/ );
};

CxCore_RecipTest::CxCore_RecipTest()
    : CxCore_ArithmTest( "arithm-recip", "cvDiv", 4, false, false )
{
    test_array[INPUT].pop();
}

void CxCore_RecipTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "s/B(i)," );
    ptr += strlen(ptr);
    params_left--;
    CxCore_ArithmTest::print_timing_params( test_case_idx, ptr, params_left );
}

void CxCore_RecipTest::run_func()
{
    if(!test_nd)
    {
        cvDiv( 0, test_array[INPUT][0],
              test_array[OUTPUT][0], gamma.val[0] );
    }
    else
    {
        cv::MatND b = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::divide(gamma.val[0], b, c);
    }
}

void CxCore_RecipTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsDiv( 0, &test_mat[INPUT][0],
             cvScalarAll(gamma.val[0]),
             &test_mat[REF_OUTPUT][0] );
}

CxCore_RecipTest recip_test;


///////////////// matrix copy/initializing/permutations /////////////////////
                                                   
class CxCore_MemTestImpl : public CxCore_ArithmTestImpl
{
public:
    CxCore_MemTestImpl( const char* test_name, const char* test_funcs,
                        int _generate_scalars=0, bool _allow_mask=true );
protected:
    double get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ );
};

CxCore_MemTestImpl::CxCore_MemTestImpl( const char* test_name, const char* test_funcs,
                                        int _generate_scalars, bool _allow_mask ) :
    CxCore_ArithmTestImpl( test_name, test_funcs, _generate_scalars, _allow_mask, false )
{
}

double CxCore_MemTestImpl::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}

CxCore_MemTestImpl mem_test( "mem", "", 0, false );


class CxCore_MemTest : public CxCore_MemTestImpl
{
public:
    CxCore_MemTest( const char* test_name, const char* test_funcs,
                    int _generate_scalars=0, bool _allow_mask=true );
};

CxCore_MemTest::CxCore_MemTest( const char* test_name, const char* test_funcs,
                                int _generate_scalars, bool _allow_mask ) :
    CxCore_MemTestImpl( test_name, test_funcs, _generate_scalars, _allow_mask )
{
    default_timing_param_names = optional_mask ? arithm_mask_param_names : arithm_param_names;
        
    // inherit the default parameters from arithmerical test
    size_list = 0;
    whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;
}


///////////////// setidentity /////////////////////

class CxCore_SetIdentityTest : public CxCore_MemTest
{
public:
    CxCore_SetIdentityTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_SetIdentityTest::CxCore_SetIdentityTest() :
    CxCore_MemTest( "mem-setidentity", "cvSetIdentity", 4, false )
{
    test_array[INPUT].clear();
}


void CxCore_SetIdentityTest::run_func()
{
    if(!test_nd)
        cvSetIdentity(test_array[OUTPUT][0], gamma);
    else
    {
        cv::Mat a = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::setIdentity(a, gamma);
    }
}


void CxCore_SetIdentityTest::prepare_to_validation( int )
{
    cvTsSetIdentity( &test_mat[REF_OUTPUT][0], gamma );
}

CxCore_SetIdentityTest setidentity_test;


///////////////// SetZero /////////////////////

class CxCore_SetZeroTest : public CxCore_MemTest
{
public:
    CxCore_SetZeroTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_SetZeroTest::CxCore_SetZeroTest() :
    CxCore_MemTest( "mem-setzero", "cvSetZero", 0, false )
{
    test_array[INPUT].clear();
}


void CxCore_SetZeroTest::run_func()
{
    if(!test_nd)
        cvSetZero(test_array[OUTPUT][0]);
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[OUTPUT][0]);
        a.setTo(cv::Scalar());
    }
}


void CxCore_SetZeroTest::prepare_to_validation( int )
{
    cvTsZero( &test_mat[REF_OUTPUT][0] );
}

CxCore_SetZeroTest setzero_test;


///////////////// Set /////////////////////

class CxCore_FillTest : public CxCore_MemTest
{
public:
    CxCore_FillTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_FillTest::CxCore_FillTest() :
    CxCore_MemTest( "mem-fill", "cvSet", 4, true )
{
    test_array[INPUT].clear();
}


void CxCore_FillTest::run_func()
{
    const CvArr* mask = test_array[MASK][0];
    if(!test_nd)
        cvSet(test_array[INPUT_OUTPUT][0], gamma, mask);
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        a.setTo(gamma, mask ? cv::cvarrToMatND(mask) : cv::MatND());
    }
}


void CxCore_FillTest::prepare_to_validation( int )
{
    if( test_array[MASK][0] )
    {
        cvTsAdd( 0, cvScalarAll(0.), 0, cvScalarAll(0.), gamma, &test_mat[TEMP][0], 0 );
        cvTsCopy( &test_mat[TEMP][0], &test_mat[REF_INPUT_OUTPUT][0], &test_mat[MASK][0] );
    }
    else
    {
        cvTsAdd( 0, cvScalarAll(0.), 0, cvScalarAll(0.), gamma, &test_mat[REF_INPUT_OUTPUT][0], 0 );
    }
}

CxCore_FillTest fill_test;


///////////////// Copy /////////////////////

class CxCore_CopyTest : public CxCore_MemTest
{
public:
    CxCore_CopyTest();
protected:
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_CopyTest::CxCore_CopyTest() :
    CxCore_MemTest( "mem-copy", "cvCopy", 0, true )
{
    test_array[INPUT].pop();
}


double CxCore_CopyTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CxCore_CopyTest::run_func()
{
    const CvArr* mask = test_array[MASK][0];
    if(!test_nd)
        cvCopy(test_array[INPUT][0], test_array[INPUT_OUTPUT][0], mask);
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        if(!mask)
            a.copyTo(c);
        else
            a.copyTo(c, cv::cvarrToMatND(mask));
    }
}


void CxCore_CopyTest::prepare_to_validation( int )
{
    cvTsCopy( &test_mat[INPUT][0], &test_mat[REF_INPUT_OUTPUT][0],
              test_array[MASK].size() > 0 && test_array[MASK][0] ? &test_mat[MASK][0] : 0 );
}

CxCore_CopyTest copy_test;

///////////////// Transpose /////////////////////

class CxCore_TransposeTest : public CxCore_MemTest
{
public:
    CxCore_TransposeTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    bool inplace;
};


CxCore_TransposeTest::CxCore_TransposeTest() :
    CxCore_MemTest( "mem-transpose", "cvTranspose", 0, false ), inplace(false)
{
    test_array[INPUT].pop();
}


void CxCore_TransposeTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    int bits = cvTsRandInt(ts->get_rng());
    CxCore_MemTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    inplace = false;
    if( bits & 1 )
    {
        sizes[INPUT][0].height = sizes[INPUT][0].width;
        inplace = (bits & 2) != 0;
    }

    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(sizes[INPUT][0].height, sizes[INPUT][0].width );
}


void CxCore_TransposeTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CxCore_MemTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    CvSize size = sizes[INPUT][0];
    if( size.width != size.height )
    {
        sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] =
        whole_sizes[OUTPUT][0] = whole_sizes[REF_OUTPUT][0] = cvSize(size.height,size.width);
    }
}


int CxCore_TransposeTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MemTest::prepare_test_case( test_case_idx );
    if( inplace && code > 0 )
        cvTsCopy( &test_mat[INPUT][0], &test_mat[OUTPUT][0] );
    return code;
}

void CxCore_TransposeTest::run_func()
{
    cvTranspose( inplace ? test_array[OUTPUT][0] : test_array[INPUT][0], test_array[OUTPUT][0]);
}


void CxCore_TransposeTest::prepare_to_validation( int )
{
    cvTsTranspose( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0] );
}

CxCore_TransposeTest transpose_test;


///////////////// Flip /////////////////////

static const int flip_codes[] = { 0, 1, -1, INT_MIN };
static const char* flip_strings[] = { "center", "vert", "horiz", 0 };
static const char* flip_param_names[] = { "size", "flip_op", "channels", "depth", 0 };

class CxCore_FlipTest : public CxCore_MemTest
{
public:
    CxCore_FlipTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int prepare_test_case( int test_case_idx );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int write_default_params( CvFileStorage* fs );
    int flip_type;
    bool inplace;
};


CxCore_FlipTest::CxCore_FlipTest() :
    CxCore_MemTest( "mem-flip", "cvFlip", 0, false ), flip_type(0), inplace(false)
{
    test_array[INPUT].pop();
    default_timing_param_names = flip_param_names;
}


int CxCore_FlipTest::write_default_params( CvFileStorage* fs )
{
    int i, code = CxCore_MemTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    start_write_param( fs );
    cvStartWriteStruct( fs, "flip_op", CV_NODE_SEQ + CV_NODE_FLOW );
    for( i = 0; flip_codes[i] != INT_MIN; i++ )
        cvWriteString( fs, 0, flip_strings[flip_codes[i]+1] );
    cvEndWriteStruct(fs);
    return code;
}


void CxCore_FlipTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    int bits = cvTsRandInt(ts->get_rng());
    CxCore_MemTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    flip_type = (bits & 3) - 2;
    flip_type += flip_type == -2;
    inplace = (bits & 4) != 0;
}


void CxCore_FlipTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MemTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* flip_op_str = cvReadString( find_timing_param( "flip_op" ), "center" );
    if( strcmp( flip_op_str, "vert" ) == 0 )
        flip_type = 0;
    else if( strcmp( flip_op_str, "horiz" ) == 0 )
        flip_type = 1;
    else
        flip_type = -1;
}


void CxCore_FlipTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", flip_type > 0 ? "horiz" : flip_type < 0 ? "center" : "vert" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_MemTest::print_timing_params( test_case_idx, ptr, params_left );
}


int CxCore_FlipTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MemTest::prepare_test_case( test_case_idx );
    if( inplace && code > 0 )
        cvTsCopy( &test_mat[INPUT][0], &test_mat[OUTPUT][0] );
    return code;
}


void CxCore_FlipTest::run_func()
{
    cvFlip(inplace ? test_array[OUTPUT][0] : test_array[INPUT][0], test_array[OUTPUT][0], flip_type);
}


void CxCore_FlipTest::prepare_to_validation( int )
{
    cvTsFlip( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0], flip_type );
}

CxCore_FlipTest flip_test;


///////////////// Split/Merge /////////////////////

static const char* split_merge_types[] = { "all", "single", 0 };
static int split_merge_channels[] = { 2, 3, 4, -1 };
static const char* split_merge_param_names[] = { "size", "planes", "channels", "depth", 0 };

class CxCore_SplitMergeBaseTest : public CxCore_MemTest
{
public:
    CxCore_SplitMergeBaseTest( const char* test_name, const char* test_funcs, int _is_split );
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int prepare_test_case( int test_case_idx );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void prepare_to_validation( int test_case_idx );
    int write_default_params( CvFileStorage* fs );
    bool are_images;
    int is_split, coi; 
    void* hdrs[4];
};


CxCore_SplitMergeBaseTest::CxCore_SplitMergeBaseTest( const char* test_name,
    const char* test_funcs, int _is_split )
    : CxCore_MemTest( test_name, test_funcs, 0, false ), are_images(false), is_split(_is_split), coi(0)
{
    test_array[INPUT].pop();
    if( is_split )
        ;
    else
    {
        test_array[OUTPUT].clear();
        test_array[REF_OUTPUT].clear();
        test_array[INPUT_OUTPUT].push(NULL);
        test_array[REF_INPUT_OUTPUT].push(NULL);
    }
    memset( hdrs, 0, sizeof(hdrs) );

    default_timing_param_names = split_merge_param_names;
    cn_list = split_merge_channels;
}


int CxCore_SplitMergeBaseTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_MemTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "planes", split_merge_types );
    return code;
}


void CxCore_SplitMergeBaseTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    int cn, depth;
    CvRNG* rng = ts->get_rng();
    CxCore_MemTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    cn = cvTsRandInt(rng)%3 + 2;
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    
    if( is_split )
    {
        types[INPUT][0] = CV_MAKETYPE(depth, cn);
        types[OUTPUT][0] = types[REF_OUTPUT][0] = depth;
    }
    else
    {
        types[INPUT][0] = depth;
        types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] = CV_MAKETYPE(depth, cn);
    }

    if( (cvTsRandInt(rng) & 3) != 0 )
    {
        coi = cvTsRandInt(rng) % cn;
    }
    else
    {
        CvSize size = sizes[INPUT][0];
        size.height *= cn;

        if( is_split )
            sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = size;
        else
            sizes[INPUT][0] = size;
        coi = -1;
    }

    are_images = cvTsRandInt(rng)%2 != 0;
}


void CxCore_SplitMergeBaseTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                    CvSize** sizes, int** types, CvSize** whole_sizes, bool* _are_images )
{
    CxCore_MemTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, _are_images );
    const char* split_merge_type = cvReadString( find_timing_param( "planes" ), "all" );
    int type0 = types[INPUT][0];
    int depth = CV_MAT_DEPTH(type0);
    int cn = CV_MAT_CN(type0);
    CvSize size = sizes[INPUT][0];

    if( strcmp( split_merge_type, "single" ) == 0 )
        coi = cvTsRandInt(ts->get_rng()) % cn;
    else
    {
        coi = -1;
        size.height *= cn;
    }

    if( is_split )
    {
        types[OUTPUT][0] = types[REF_OUTPUT][0] = depth;
        sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = size;
        
        // planes are put into separate arrays, not ROI's
        whole_sizes[OUTPUT][0] = whole_sizes[REF_OUTPUT][0] = size;
    }
    else
    {
        types[INPUT][0] = depth;
        sizes[INPUT][0] = size;
        
        // planes are put into separate arrays, not ROI's
        whole_sizes[INPUT][0] = size;
    }

    are_images = false;
}


void CxCore_SplitMergeBaseTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    int i;
    
    sprintf( ptr, "%s,", coi >= 0 ? "single" : "all" );
    ptr += strlen(ptr);
    params_left--;

    // at once, delete the headers, though is not very good from structural point of view ...
    for( i = 0; i < 4; i++ )
        cvRelease( &hdrs[i] );

    CxCore_MemTest::print_timing_params( test_case_idx, ptr, params_left );
}


int CxCore_SplitMergeBaseTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MemTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        CvMat* input = &test_mat[INPUT][0];
        CvMat* output = &test_mat[is_split ? OUTPUT : INPUT_OUTPUT][0];
        CvMat* merged = is_split ? input : output;
        CvMat* planes = is_split ? output : input;
        int depth = CV_MAT_DEPTH(merged->type);
        int i, cn = CV_MAT_CN(merged->type), y = 0;
        CvSize sz = cvGetMatSize(merged);

        for( i = 0; i < cn; i++ )
        {
            if( coi < 0 || coi == i )
            {
                if( are_images )
                    hdrs[i] = cvCreateImageHeader( sz, cvIplDepth(depth), 1 );
                else
                    hdrs[i] = cvCreateMatHeader( sz.height, sz.width, depth );
                cvSetData( hdrs[i], planes->data.ptr + planes->step*y, planes->step );
                y += sz.height;
            }
        }
    }

    return code;
}


void CxCore_SplitMergeBaseTest::prepare_to_validation( int )
{
    CvMat* input = &test_mat[INPUT][0];
    CvMat* output = &test_mat[is_split ? REF_OUTPUT : REF_INPUT_OUTPUT][0];
    CvMat* merged = is_split ? input : output;
    CvMat* planes = is_split ? output : input;
    int i, cn = CV_MAT_CN(merged->type), y = 0;
    CvSize sz = cvGetSize(merged);

    for( i = 0; i < cn; i++ )
    {
        if( coi < 0 || coi == i )
        {
            CvMat stub, *h;
            cvSetData( hdrs[i], planes->data.ptr + planes->step*y, planes->step );
            h = cvGetMat( hdrs[i], &stub );
            if( is_split )
                cvTsExtract( input, h, i );
            else
                cvTsInsert( h, output, i );
            cvSetData( hdrs[i], 0, 0 );
            cvRelease( &hdrs[i] );
            y += sz.height;
        }
    }
}


class CxCore_SplitTest : public CxCore_SplitMergeBaseTest
{
public:
    CxCore_SplitTest();
protected:
    void run_func();
};


CxCore_SplitTest::CxCore_SplitTest() :
    CxCore_SplitMergeBaseTest( "mem-split", "cvSplit", 1 )
{
}


void CxCore_SplitTest::run_func()
{
    int i, nz = (hdrs[0] != 0) + (hdrs[1] != 0) + (hdrs[2] != 0) + (hdrs[3] != 0);
    
    if(!test_nd || nz != CV_MAT_CN(test_mat[INPUT][0].type))
        cvSplit( test_array[INPUT][0], hdrs[0], hdrs[1], hdrs[2], hdrs[3] );
    else
    {
        cv::MatND _hdrs[4];
        for( i = 0; i < nz; i++ )
            _hdrs[i] = cv::cvarrToMatND(hdrs[i]);
        cv::split(cv::cvarrToMatND(test_array[INPUT][0]), _hdrs);
    }
}

CxCore_SplitTest split_test;

class CxCore_MergeTest : public CxCore_SplitMergeBaseTest
{
public:
    CxCore_MergeTest();
protected:
    void run_func();
};


CxCore_MergeTest::CxCore_MergeTest() :
    CxCore_SplitMergeBaseTest( "mem-merge", "cvMerge", 0 )
{
}


void CxCore_MergeTest::run_func()
{
    int i, nz = (hdrs[0] != 0) + (hdrs[1] != 0) + (hdrs[2] != 0) + (hdrs[3] != 0);
    
    if(!test_nd || nz != CV_MAT_CN(test_mat[INPUT_OUTPUT][0].type))
        cvMerge( hdrs[0], hdrs[1], hdrs[2], hdrs[3], test_array[INPUT_OUTPUT][0] );
    else
    {
        cv::MatND _hdrs[4], dst = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        for( i = 0; i < nz; i++ )
            _hdrs[i] = cv::cvarrToMatND(hdrs[i]);
        cv::merge(_hdrs, nz, dst);
    }
}

CxCore_MergeTest merge_test;

///////////////// CompleteSymm /////////////////////

class CxCore_CompleteSymm : public CvArrTest
{
public:
    CxCore_CompleteSymm();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
	int LtoR; //flags 
};

CxCore_CompleteSymm::CxCore_CompleteSymm() :
    CvArrTest("matrix-symm", "cvCompleteSymm", "Test of cvCompleteSymm function")
{
	/*Generates 1 input and 1 outputs (by default we have 2 inputs and 1 output)*/
	test_array[INPUT].clear();
	test_array[INPUT].push(NULL);
	test_array[OUTPUT].clear();
	test_array[OUTPUT].push(NULL);
	test_array[REF_OUTPUT].clear();
	test_array[REF_OUTPUT].push(NULL);
}


void CxCore_CompleteSymm::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] =sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(sizes[INPUT][0].height, sizes[INPUT][0].height );

	/*Making input and output matrixes one-channel*/
	int type;
	switch (test_case_idx % 3)
	{
		case 0:
			type = CV_32FC1;
			break;
		case 1:
			type = CV_32SC1;
			break;
		default:
			type = CV_64FC1;
	}
	types[OUTPUT][0] = types[INPUT][0] = types[REF_OUTPUT][0] = type;
}

int CxCore_CompleteSymm::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
	if (code)
	{
		CvRNG* rng = ts->get_rng();
		unsigned val = cvRandInt(rng);
		LtoR = val % 2;
		cvConvert(&test_mat[INPUT][0], &test_mat[OUTPUT][0]);
	}
	return code;
}

void CxCore_CompleteSymm::run_func()
{
	cvCompleteSymm(&test_mat[OUTPUT][0],LtoR);
}

void CxCore_CompleteSymm::prepare_to_validation( int )
{
	CvMat* ref_output = cvCreateMat(test_mat[OUTPUT][0].rows, test_mat[OUTPUT][0].cols, CV_64F); 
	CvMat* input = cvCreateMat(test_mat[INPUT][0].rows, test_mat[INPUT][0].cols, CV_64F);
	cvConvert(&test_mat[INPUT][0], input);
	
	for (int i=0;i<input->rows;i++)
	{
		ref_output->data.db[i*input->cols+i]=input->data.db[i*input->cols+i];
		if (LtoR)
		{
			for (int j=0;j<i;j++)
			{
				ref_output->data.db[j*input->cols+i] = ref_output->data.db[i*input->cols+j]=input->data.db[i*input->cols+j];
			}
				
		}
		else 
		{
			for (int j=0;j<i;j++)
			{
				ref_output->data.db[j*input->cols+i] = ref_output->data.db[i*input->cols+j]=input->data.db[j*input->cols+i];
			}
		}
	}

	cvConvert(ref_output, &test_mat[REF_OUTPUT][0]);
	cvReleaseMat(&input);
	cvReleaseMat(&ref_output);
}

CxCore_CompleteSymm complete_symm;


////////////////////////////// Sort /////////////////////////////////

class CxCore_SortTest : public CxCore_MemTest
{
public:
    CxCore_SortTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
	int flags; //flags for sorting
private:
	static int compareIndexes (const void * a, const void * b); // comparing two elements of the matrix with pointers sorting
	static int compare(const void * a, const void * b); // comparing two elements of the matrix with pointers sorting
	bool useIndexMatrix;
	bool useInPlaceSort;
	CvMat* input;

};

CxCore_SortTest::CxCore_SortTest() :
    CxCore_MemTest( "matrix-sort", "cvSort", 0, false )
{
	/*Generates 1 input and 2 outputs (by default we have 2 inputs and 1 output)*/
	test_array[INPUT].clear();
	test_array[INPUT].push(NULL);
	test_array[OUTPUT].push(NULL);
	test_array[REF_OUTPUT].push(NULL);
}


void CxCore_SortTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CxCore_MemTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sizes[OUTPUT][1] = sizes[REF_OUTPUT][1] = cvSize(sizes[INPUT][0].height, sizes[INPUT][0].width );
	types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_32SC1;

	/*Making input and output matrixes one-channel*/
	types[OUTPUT][0] = types[INPUT][0] = CV_MAKETYPE(CV_MAT_DEPTH(types[INPUT][0]), 1);
	types[REF_OUTPUT][0] = CV_MAKETYPE(CV_MAT_DEPTH(types[REF_OUTPUT][0]), 1);
}

int CxCore_SortTest::prepare_test_case( int test_case_idx )
{
	if (test_case_idx==0)
	{
		useIndexMatrix=true;
		useInPlaceSort=false;
	}
   int code = CxCore_MemTest::prepare_test_case( test_case_idx );

   if( code > 0 )
	{
		//Copying input data
		input = cvCreateMat(test_mat[INPUT][0].rows, test_mat[INPUT][0].cols, CV_64F);
		cvConvert(&test_mat[INPUT][0], input);
		CvRNG* rng = ts->get_rng();
		unsigned val = cvRandInt(rng);
        // Setting up flags
		switch (val%4)
		{
			case 0:
				flags = CV_SORT_EVERY_ROW + CV_SORT_DESCENDING;
				break;
			case 1:
				flags = CV_SORT_EVERY_ROW + CV_SORT_ASCENDING;
				break;
			case 2:
				flags = CV_SORT_EVERY_COLUMN + CV_SORT_DESCENDING;
				break;
			case 3:
				flags = CV_SORT_EVERY_COLUMN + CV_SORT_ASCENDING;
				break;
		}
		if (val%3) 
			useIndexMatrix = !useIndexMatrix;

		if (val%5) 
			useInPlaceSort = !useInPlaceSort;

	}
    return code;
}

void CxCore_SortTest::run_func()
{
	//test_mat[OUTPUT][0] is sorted matrix
	//test_mat[OUTPUT][1] is index matrix
	if (useInPlaceSort)
	{
		cvConvert(&test_mat[INPUT][0], &test_mat[OUTPUT][0]);
		if (useIndexMatrix)
			cvSort(&(test_mat[OUTPUT][0]),&(test_mat[OUTPUT][0]),&(test_mat[OUTPUT][1]),flags);
		else
		{
			cvSort(&(test_mat[OUTPUT][0]),&(test_mat[OUTPUT][0]),0,flags);
		}

	}
	else
	{
		if (useIndexMatrix)
			cvSort(&(test_mat[INPUT][0]),&(test_mat[OUTPUT][0]),&(test_mat[OUTPUT][1]),flags);
		else
		{
			cvSort(&(test_mat[INPUT][0]),&(test_mat[OUTPUT][0]),0,flags);
		}
	}
}

int CxCore_SortTest::compareIndexes (const void * a, const void * b)
{
	double zero = 1e-30;
	double res=(**((double**)a)-**((double**)b));
	return res<-zero?-1:(res>zero?1:0);
}
int CxCore_SortTest::compare (const void * a, const void * b)
{
	return *((int*)a)-*((int*)b);
}

void CxCore_SortTest::prepare_to_validation(int)
{
	/*Creating matrixes copies to work with*/
	CvMat* ref_indexes = cvCreateMat(test_mat[REF_OUTPUT][1].rows, test_mat[REF_OUTPUT][1].cols, CV_32SC1); 
	CvMat* indexes = cvCreateMat(test_mat[OUTPUT][1].rows, test_mat[OUTPUT][1].cols, CV_32SC1); 
	CvMat* ref_output = cvCreateMat(test_mat[OUTPUT][0].rows, test_mat[OUTPUT][0].cols,CV_64F); 
	
	/*Copying data*/
	cvConvert(&test_mat[REF_OUTPUT][1], ref_indexes);
	cvConvert(&test_mat[OUTPUT][1], indexes);

	/*Following block generates REF_OUTPUT indexes matrix*/
	if ((flags == (CV_SORT_EVERY_ROW+CV_SORT_ASCENDING)) ||(flags == (CV_SORT_EVERY_ROW+CV_SORT_DESCENDING)))
	for (int i=0;i<test_mat[REF_OUTPUT][1].rows;i++)
		for (int j=0;j<test_mat[REF_OUTPUT][1].cols;j++)
			ref_indexes->data.i[ref_indexes->cols*i + j]=j;
	else 
	for (int i=0;i<test_mat[REF_OUTPUT][1].rows;i++)
		for (int j=0;j<test_mat[REF_OUTPUT][1].cols;j++)
			ref_indexes->data.i[ref_indexes->cols*i + j]=i;
	cvConvert(ref_indexes, &test_mat[REF_OUTPUT][1]);
	/*End of block*/

	/* Matrix User's Sorting Algorithm */
	int order = -1; // order of sorting (ASCENDING or DESCENDING)
	//// Following to variables are for sorting rows or cols in one block without any conditions (if statements)
	short rowsSort=0;
	short colsSort=0;
	if ((flags == CV_SORT_EVERY_ROW+CV_SORT_ASCENDING)||(flags == CV_SORT_EVERY_COLUMN+CV_SORT_ASCENDING)) order=1;
	if ((flags == CV_SORT_EVERY_ROW+CV_SORT_ASCENDING)||(flags == CV_SORT_EVERY_ROW+CV_SORT_DESCENDING)) rowsSort=1;
	else colsSort=1;
	int i,j;
	
	// For accessing [i,j] element using index matrix we can use following formula
	// input->data.db[(input->cols*i+ref_indexes->cols*i+j)*rowsSort+(cols*(ref_indexes->cols*i+j)+j)*colsSort];

    if ((flags == CV_SORT_EVERY_ROW+CV_SORT_ASCENDING)||(flags == CV_SORT_EVERY_ROW+CV_SORT_DESCENDING))
	{
		double** row = new double*[input->cols];
		for (i=0;i<input->rows; i++)
		{
			for (int j=0;j<input->cols;j++)
				row[j]=&(input->data.db[(input->cols*i+j)]);
			qsort(row,input->cols,sizeof(row[0]),&CxCore_SortTest::compareIndexes);
			for (int j=0;j<ref_indexes->cols;j++)
			{
				if (order==1)
					ref_indexes->data.i[ref_indexes->cols*i+j]=(int)(row[j]-&(input->data.db[input->cols*i]));
				else
					ref_indexes->data.i[ref_indexes->cols*(i+1)-1-j]=(int)(row[j]-&(input->data.db[input->cols*i]));
			}
		}
		delete[] row;
	}
	else
	{
		double** col = new double*[input->rows];
		for (j=0;j<input->cols; j++)
		{
			for (int i=0;i<input->rows;i++)
				col[i]=&(input->data.db[(input->cols*i+j)]);
			qsort(col,input->rows,sizeof(col[0]),&CxCore_SortTest::compareIndexes);
			for (int i=0;i<ref_indexes->rows;i++)
			{
				if (order==1)
					ref_indexes->data.i[ref_indexes->cols*i+j]=(int)((col[i]-&(input->data.db[j]))/(ref_output->cols));
				else
					ref_indexes->data.i[ref_indexes->cols*(ref_indexes->rows-1-i)+j]=(int)(col[i]-&(input->data.db[j]))/(ref_output->cols);
			}
		}
		delete[] col;
	}

	/*End of Sort*/

	int n;
	for (i=0;i<input->rows;i++)
		for (j=0;j<input->cols;j++)
		{
			n=(input->cols*i+ref_indexes->data.i[ref_indexes->cols*i+j])*rowsSort+
			(input->cols*(ref_indexes->data.i[ref_indexes->cols*i+j])+j)*colsSort;
			ref_output->data.db[ref_output->cols*i+j] = input->data.db[n];
		}

	if (useIndexMatrix)
	{
		/* Comparing indexes matrixes */
		if ((flags == CV_SORT_EVERY_ROW+CV_SORT_ASCENDING)||(flags == CV_SORT_EVERY_ROW+CV_SORT_DESCENDING))
		{
			int begin=0,end=0;
			double temp;
			for (i=0;i<indexes->rows;i++)
			{
				for (j=0;j<indexes->cols-1;j++)
					if (ref_output->data.db[ref_output->cols*i+j]==ref_output->data.db[ref_output->cols*i+j+1])
					{
						temp=ref_output->data.db[ref_output->cols*i+j];
						begin=j++;
						while ((j<ref_output->cols)&&(temp==ref_output->data.db[ref_output->cols*i+j])) j++;
						end=--j;
						int* row = new int[end-begin+1];
						int* row1 = new int[end-begin+1];

						for (int k=0;k<=end-begin;k++)
						{
							row[k]=ref_indexes->data.i[ref_indexes->cols*i+k+begin];
							row1[k]=indexes->data.i[indexes->cols*i+k+begin];
						}
						qsort(row,end-begin+1,sizeof(row[0]),&CxCore_SortTest::compare);
						qsort(row1,end-begin+1,sizeof(row1[0]),&CxCore_SortTest::compare);
						for (int k=0;k<=end-begin;k++)
						{
							ref_indexes->data.i[ref_indexes->cols*i+k+begin]=row[k];
							indexes->data.i[indexes->cols*i+k+begin]=row1[k];
						}	
						delete[] row;
						delete[] row1;
					}
			}
		}
		else
		{
			int begin=0,end=0;
			double temp;
			for (j=0;j<indexes->cols;j++)
			{
				for (i=0;i<indexes->rows-1;i++)
					if (ref_output->data.db[ref_output->cols*i+j]==ref_output->data.db[ref_output->cols*(i+1)+j])
					{
						temp=ref_output->data.db[ref_output->cols*i+j];
						begin=i++;
						while ((i<ref_output->rows)&&(temp==ref_output->data.db[ref_output->cols*i+j])) i++;
						end=--i;

						int* col = new int[end-begin+1];
						int* col1 = new int[end-begin+1];

						for (int k=0;k<=end-begin;k++)
						{
							col[k]=ref_indexes->data.i[ref_indexes->cols*(k+begin)+j];
							col1[k]=indexes->data.i[indexes->cols*(k+begin)+j];
						}
						qsort(col,end-begin+1,sizeof(col[0]),&CxCore_SortTest::compare);
						qsort(col1,end-begin+1,sizeof(col1[0]),&CxCore_SortTest::compare);
						for (int k=0;k<=end-begin;k++)
						{
							ref_indexes->data.i[ref_indexes->cols*(k+begin)+j]=col[k];
							indexes->data.i[indexes->cols*(k+begin)+j]=col1[k];
						}	
						delete[] col;
						delete[] col1;
					}
			}
		}
	/* End of compare*/
	cvConvert(ref_indexes, &test_mat[REF_OUTPUT][1]);
	cvConvert(indexes, &test_mat[OUTPUT][1]);
	}
	else
	{
		cvConvert(ref_indexes, &test_mat[REF_OUTPUT][1]);
		cvConvert(ref_indexes, &test_mat[OUTPUT][1]);
	}

	cvConvert(ref_output, &test_mat[REF_OUTPUT][0]);

	/*releasing matrixes*/
	cvReleaseMat(&ref_output); 
	cvReleaseMat(&input); 
	cvReleaseMat(&indexes); 
	cvReleaseMat(&ref_indexes);   
}

CxCore_SortTest sort_test;

////////////////////////////// min/max  /////////////////////////////

class CxCore_MinMaxBaseTest : public CxCore_ArithmTest
{
public:
    CxCore_MinMaxBaseTest( const char* test_name, const char* test_funcs,
                           int _op_type, int _generate_scalars=0 );
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ );
    void prepare_to_validation( int /*test_case_idx*/ );
    int op_type;
};

CxCore_MinMaxBaseTest::CxCore_MinMaxBaseTest( const char* test_name, const char* test_funcs,
                                              int _op_type, int _generate_scalars )
    : CxCore_ArithmTest( test_name, test_funcs, _generate_scalars, false, false ), op_type(_op_type)
{
    if( _generate_scalars )
        test_array[INPUT].pop();
    default_timing_param_names = minmax_param_names;
}

double CxCore_MinMaxBaseTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}

void CxCore_MinMaxBaseTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    int i, j;
    CxCore_ArithmTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        for( j = 0; j < count; j++ )
        {
            types[i][j] &= ~CV_MAT_CN_MASK;            
        }
    }
}

void CxCore_MinMaxBaseTest::prepare_to_validation( int /*test_case_idx*/ )
{
    if( !gen_scalars )
        cvTsMinMax( &test_mat[INPUT][0], &test_mat[INPUT][1],
                    &test_mat[REF_OUTPUT][0], op_type );
    else
        cvTsMinMaxS( &test_mat[INPUT][0], gamma.val[0],
                     &test_mat[REF_OUTPUT][0], op_type );
}


class CxCore_MinTest : public CxCore_MinMaxBaseTest
{
public:
    CxCore_MinTest();
protected:
    void run_func();
};


CxCore_MinTest::CxCore_MinTest()
    : CxCore_MinMaxBaseTest( "arithm-min", "cvMin", CV_TS_MIN, 0 )
{
}

void CxCore_MinTest::run_func()
{
    if(!test_nd)
    {
        cvMin( test_array[INPUT][0], test_array[INPUT][1], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::min(cv::cvarrToMatND(test_array[INPUT][0]),
                cv::cvarrToMatND(test_array[INPUT][1]), c);
    }
}

CxCore_MinTest min_test;


////////////////////////////// max /////////////////////////////

class CxCore_MaxTest : public CxCore_MinMaxBaseTest
{
public:
    CxCore_MaxTest();
protected:
    void run_func();
};

CxCore_MaxTest::CxCore_MaxTest()
    : CxCore_MinMaxBaseTest( "arithm-max", "cvMax", CV_TS_MAX, 0 )
{
}

void CxCore_MaxTest::run_func()
{
    if(!test_nd)
    {
        cvMax( test_array[INPUT][0], test_array[INPUT][1], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::max(cv::cvarrToMatND(test_array[INPUT][0]),
                cv::cvarrToMatND(test_array[INPUT][1]), c);
    }
}

CxCore_MaxTest max_test;


////////////////////////////// mins /////////////////////////////

class CxCore_MinSTest : public CxCore_MinMaxBaseTest
{
public:
    CxCore_MinSTest();
protected:
    void run_func();
};

CxCore_MinSTest::CxCore_MinSTest()
    : CxCore_MinMaxBaseTest( "arithm-mins", "cvMinS", CV_TS_MIN, 4 )
{
}

void CxCore_MinSTest::run_func()
{
    if(!test_nd)
    {
        cvMinS( test_array[INPUT][0], gamma.val[0], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::min(cv::cvarrToMatND(test_array[INPUT][0]),
                gamma.val[0], c);
    }
}

CxCore_MinSTest mins_test;

////////////////////////////// maxs /////////////////////////////

class CxCore_MaxSTest : public CxCore_MinMaxBaseTest
{
public:
    CxCore_MaxSTest();
protected:
    void run_func();
};

CxCore_MaxSTest::CxCore_MaxSTest()
    : CxCore_MinMaxBaseTest( "arithm-maxs", "cvMaxS", CV_TS_MAX, 4 )
{
}

void CxCore_MaxSTest::run_func()
{
    if(!test_nd)
    {
        cvMaxS( test_array[INPUT][0], gamma.val[0], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::max(cv::cvarrToMatND(test_array[INPUT][0]),
                gamma.val[0], c);
    }
}

CxCore_MaxSTest maxs_test;


//////////////////////////////// logic ///////////////////////////////////////

class CxCore_LogicTestImpl : public CxCore_ArithmTestImpl
{
public:
    CxCore_LogicTestImpl( const char* test_name, const char* test_funcs, int _logic_op,
                      int _generate_scalars=0, bool _allow_mask=true );
protected:
    void prepare_to_validation( int test_case_idx );
    int logic_op;
};

CxCore_LogicTestImpl::CxCore_LogicTestImpl( const char* test_name, const char* test_funcs,
                            int _logic_op, int _generate_scalars, bool _allow_mask )
    : CxCore_ArithmTestImpl( test_name, test_funcs, _generate_scalars, _allow_mask, false ),
    logic_op(_logic_op)
{
    if( _generate_scalars )
        test_array[INPUT].pop();
}

void CxCore_LogicTestImpl::prepare_to_validation( int /*test_case_idx*/ )
{
    int ref_output_idx = optional_mask ? REF_INPUT_OUTPUT : REF_OUTPUT;
    int output_idx = optional_mask ? INPUT_OUTPUT : OUTPUT;
    const CvMat* mask = test_array[MASK].size() > 0 && test_array[MASK][0] ? &test_mat[MASK][0] : 0;
    CvMat* dst = mask ? &test_mat[TEMP][0] : &test_mat[ref_output_idx][0];
    int i;
    if( test_array[INPUT].size() > 1 )
    {
        cvTsLogic( &test_mat[INPUT][0], &test_mat[INPUT][1], dst, logic_op );
    }
    else
    {
        cvTsLogicS( &test_mat[INPUT][0], gamma, dst, logic_op );
    }
    if( mask )
        cvTsCopy( dst, &test_mat[ref_output_idx][0], mask );
    
    for( i = 0; i < 2; i++ )
    {
        dst = i == 0 ? &test_mat[ref_output_idx][0] : &test_mat[output_idx][0];

        if( CV_IS_MAT(dst) )
        {
            CvMat* mat = (CvMat*)dst;
            mat->cols *= CV_ELEM_SIZE(mat->type);
            mat->type = (mat->type & ~CV_MAT_TYPE_MASK) | CV_8UC1;
        }
        else
        {
            IplImage* img = (IplImage*)dst;
            int elem_size;
        
            assert( CV_IS_IMAGE(dst) );
            elem_size = ((img->depth & 255)>>3)*img->nChannels;
            img->width *= elem_size;
        
            if( img->roi )
            {
                img->roi->xOffset *= elem_size;
                img->roi->width *= elem_size;
            }
            img->depth = IPL_DEPTH_8U;
            img->nChannels = 1;
        }
    }
}

CxCore_LogicTestImpl logic_test("logic", "", -1, 0, false );

class CxCore_LogicTest : public CxCore_LogicTestImpl
{
public:
    CxCore_LogicTest( const char* test_name, const char* test_funcs, int _logic_op,
                      int _generate_scalars=0, bool _allow_mask=true );
};

CxCore_LogicTest::CxCore_LogicTest( const char* test_name, const char* test_funcs,
                            int _logic_op, int _generate_scalars, bool _allow_mask )
    : CxCore_LogicTestImpl( test_name, test_funcs, _logic_op, _generate_scalars, _allow_mask )
{
    default_timing_param_names = optional_mask ? arithm_mask_param_names : arithm_param_names;

    // inherit the default parameters from arithmerical test
    size_list = 0;
    whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;
}


///////////////////////// and //////////////////////////

class CxCore_AndTest : public CxCore_LogicTest
{
public:
    CxCore_AndTest();
protected:
    void run_func();
};

CxCore_AndTest::CxCore_AndTest()
    : CxCore_LogicTest( "logic-and", "cvAnd", CV_TS_LOGIC_AND )
{
}

void CxCore_AndTest::run_func()
{
    if(!test_nd)
    {
        cvAnd( test_array[INPUT][0], test_array[INPUT][1],
              test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::bitwise_and(cv::cvarrToMatND(test_array[INPUT][0]),
                        cv::cvarrToMatND(test_array[INPUT][1]),
                        c, cv::cvarrToMatND(test_array[MASK][0]));
    }
}

CxCore_AndTest and_test;


class CxCore_AndSTest : public CxCore_LogicTest
{
public:
    CxCore_AndSTest();
protected:
    void run_func();
};

CxCore_AndSTest::CxCore_AndSTest()
    : CxCore_LogicTest( "logic-ands", "cvAndS", CV_TS_LOGIC_AND, 4 )
{
}

void CxCore_AndSTest::run_func()
{
    if(!test_nd)
    {
        cvAndS( test_array[INPUT][0], gamma,
              test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::bitwise_and(cv::cvarrToMatND(test_array[INPUT][0]),
                        gamma, c,
                        cv::cvarrToMatND(test_array[MASK][0]));
    }
}

CxCore_AndSTest ands_test;


///////////////////////// or /////////////////////////

class CxCore_OrTest : public CxCore_LogicTest
{
public:
    CxCore_OrTest();
protected:
    void run_func();
};

CxCore_OrTest::CxCore_OrTest()
    : CxCore_LogicTest( "logic-or", "cvOr", CV_TS_LOGIC_OR )
{
}

void CxCore_OrTest::run_func()
{
    if(!test_nd)
    {
        cvOr( test_array[INPUT][0], test_array[INPUT][1],
              test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::bitwise_or(cv::cvarrToMatND(test_array[INPUT][0]),
                        cv::cvarrToMatND(test_array[INPUT][1]),
                        c, cv::cvarrToMatND(test_array[MASK][0]));
    }
    
}

CxCore_OrTest or_test;


class CxCore_OrSTest : public CxCore_LogicTest
{
public:
    CxCore_OrSTest();
protected:
    void run_func();
};

CxCore_OrSTest::CxCore_OrSTest()
    : CxCore_LogicTest( "logic-ors", "cvOrS", CV_TS_LOGIC_OR, 4 )
{
}

void CxCore_OrSTest::run_func()
{
    if(!test_nd)
    {
        cvOrS( test_array[INPUT][0], gamma,
               test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::bitwise_or(cv::cvarrToMatND(test_array[INPUT][0]),
                        gamma, c,
                        cv::cvarrToMatND(test_array[MASK][0]));
    }
}

CxCore_OrSTest ors_test;


////////////////////////// xor ////////////////////////////

class CxCore_XorTest : public CxCore_LogicTest
{
public:
    CxCore_XorTest();
protected:
    void run_func();
};

CxCore_XorTest::CxCore_XorTest()
    : CxCore_LogicTest( "logic-xor", "cvXor", CV_TS_LOGIC_XOR )
{
}

void CxCore_XorTest::run_func()
{
    if(!test_nd)
    {
        cvXor( test_array[INPUT][0], test_array[INPUT][1],
               test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::bitwise_xor(cv::cvarrToMatND(test_array[INPUT][0]),
                        cv::cvarrToMatND(test_array[INPUT][1]),
                        c, cv::cvarrToMatND(test_array[MASK][0]));
    }
    
}

CxCore_XorTest xor_test;


class CxCore_XorSTest : public CxCore_LogicTest
{
public:
    CxCore_XorSTest();
protected:
    void run_func();
};

CxCore_XorSTest::CxCore_XorSTest()
    : CxCore_LogicTest( "logic-xors", "cvXorS", CV_TS_LOGIC_XOR, 4 )
{
}

void CxCore_XorSTest::run_func()
{
    if(!test_nd)
    {
        cvXorS( test_array[INPUT][0], gamma,
               test_array[INPUT_OUTPUT][0], test_array[MASK][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[INPUT_OUTPUT][0]);
        cv::bitwise_xor(cv::cvarrToMatND(test_array[INPUT][0]),
                        gamma, c,
                        cv::cvarrToMatND(test_array[MASK][0]));
    }
}

CxCore_XorSTest xors_test;


////////////////////////// not ////////////////////////////

class CxCore_NotTest : public CxCore_LogicTest
{
public:
    CxCore_NotTest();
protected:
    void run_func();
};

CxCore_NotTest::CxCore_NotTest()
    : CxCore_LogicTest( "logic-not", "cvNot", CV_TS_LOGIC_NOT, 4, false )
{
}

void CxCore_NotTest::run_func()
{
    if(!test_nd)
    {
        cvNot( test_array[INPUT][0], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::bitwise_not(cv::cvarrToMatND(test_array[INPUT][0]), c);
    }
}

CxCore_NotTest nots_test;

///////////////////////// cmp //////////////////////////////

static int cmp_op_values[] = { CV_CMP_GE, CV_CMP_EQ, CV_CMP_NE, -1 };

class CxCore_CmpBaseTestImpl : public CxCore_ArithmTestImpl
{
public:
    CxCore_CmpBaseTestImpl( const char* test_name, const char* test_funcs,
                            int in_range, int _generate_scalars=0 );
protected:
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_test_array_types_and_sizes( int test_case_idx,
                                         CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes,
                            int** types, CvSize** whole_sizes, bool* are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void prepare_to_validation( int test_case_idx );
    int write_default_params( CvFileStorage* fs );
    int in_range;
    int cmp_op;
    enum { CMP_OP_COUNT=6 };
    const char* cmp_op_strings[CMP_OP_COUNT];
};

CxCore_CmpBaseTestImpl::CxCore_CmpBaseTestImpl( const char* test_name, const char* test_funcs,
                                        int _in_range, int _generate_scalars )
    : CxCore_ArithmTestImpl( test_name, test_funcs, _generate_scalars, 0, 0 ), in_range(_in_range)
{
    static const char* cmp_param_names[] = { "size", "cmp_op", "depth", 0 };
    static const char* inrange_param_names[] = { "size", "channels", "depth", 0 };

    if( in_range )
    {
        test_array[INPUT].push(NULL);
        test_array[TEMP].push(NULL);
        test_array[TEMP].push(NULL);
        if( !gen_scalars )
            test_array[TEMP].push(NULL);
    }
    if( gen_scalars )
        test_array[INPUT].pop();

    default_timing_param_names = in_range == 1 ? inrange_param_names : cmp_param_names;

    cmp_op_strings[CV_CMP_EQ] = "eq";
    cmp_op_strings[CV_CMP_LT] = "lt";
    cmp_op_strings[CV_CMP_LE] = "le";
    cmp_op_strings[CV_CMP_GE] = "ge";
    cmp_op_strings[CV_CMP_GT] = "gt";
    cmp_op_strings[CV_CMP_NE] = "ne";

    cmp_op = -1;
}

double CxCore_CmpBaseTestImpl::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CxCore_CmpBaseTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types )
{
    int j, count;
    CxCore_ArithmTestImpl::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8UC1;
    if( in_range == 0 )
    {
        // for cmp tests make all the input arrays single-channel
        count = test_array[INPUT].size();
        for( j = 0; j < count; j++ )
            types[INPUT][j] &= ~CV_MAT_CN_MASK;

        cmp_op = cvTsRandInt(ts->get_rng()) % 6; // == > >= < <= !=
    }
    else if( in_range == 1 )
    {
        types[TEMP][0] = CV_8UC1;
        types[TEMP][1] &= ~CV_MAT_CN_MASK;
        if( !gen_scalars )
            types[TEMP][2] &= ~CV_MAT_CN_MASK;
    }
}


int CxCore_CmpBaseTestImpl::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_ArithmTestImpl::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    if( in_range == 0 )
    {
        start_write_param( fs );
        int i;
        cvStartWriteStruct( fs, "cmp_op", CV_NODE_SEQ + CV_NODE_FLOW );
        for( i = 0; cmp_op_values[i] >= 0; i++ )
            cvWriteString( fs, 0, cmp_op_strings[cmp_op_values[i]] );
        cvEndWriteStruct(fs);
    }
    return code;
}


void CxCore_CmpBaseTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_ArithmTestImpl::get_timing_test_array_types_and_sizes( test_case_idx,
                                            sizes, types, whole_sizes, are_images );
    types[OUTPUT][0] = CV_8UC1;
    if( in_range == 0 )
    {
        const char* cmp_op_str = cvReadString( find_timing_param( "cmp_op" ), "ge" );
        int i;
        cmp_op = CV_CMP_GE;
        for( i = 0; i < CMP_OP_COUNT; i++ )
            if( strcmp( cmp_op_str, cmp_op_strings[i] ) == 0 )
            {
                cmp_op = i;
                break;
            }
    }
}


void CxCore_CmpBaseTestImpl::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    if( in_range == 0 )
    {
        sprintf( ptr, "%s,", cmp_op_strings[cmp_op] );
        ptr += strlen(ptr);
        params_left--;
    }
    CxCore_ArithmTestImpl::print_timing_params( test_case_idx, ptr, params_left );
}


void CxCore_CmpBaseTestImpl::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* dst = &test_mat[REF_OUTPUT][0];
    if( !in_range )
    {
        if( test_array[INPUT].size() > 1 )
        {
            cvTsCmp( &test_mat[INPUT][0], &test_mat[INPUT][1], dst, cmp_op );
        }
        else
        {
            cvTsCmpS( &test_mat[INPUT][0], gamma.val[0], dst, cmp_op );
        }
    }
    else
    {
        int el_type = CV_MAT_TYPE( test_mat[INPUT][0].type );
        int i, cn = CV_MAT_CN(el_type);
        CvMat* tdst = dst;

        for( i = 0; i < cn*2; i++ )
        {
            int coi = i / 2, is_lower = (i % 2) == 0;
            int cmp_op = is_lower ? CV_CMP_GE : CV_CMP_LT;
            const CvMat* src = &test_mat[INPUT][0];
            const CvMat* lu = gen_scalars ? 0 : &test_mat[INPUT][is_lower?1:2];
            double luS = is_lower ? alpha.val[coi] : gamma.val[coi];
            
            if( cn > 1 )
            {
                cvTsExtract( src, &test_mat[TEMP][1], coi );
                src = &test_mat[TEMP][1];

                if( !gen_scalars )
                {
                    cvTsExtract( lu, &test_mat[TEMP][2], coi );
                    lu = &test_mat[TEMP][2];
                }
            }

            if( !gen_scalars )
                cvTsCmp( src, lu, tdst, cmp_op );
            else
                cvTsCmpS( src, luS, tdst, cmp_op );
            if( i > 0 )
                cvTsLogic( tdst, dst, dst, CV_TS_LOGIC_AND );
            tdst = &test_mat[TEMP][0];
        }
    }
}


CxCore_CmpBaseTestImpl cmpbase_test( "cmp", "", -1 );


class CxCore_CmpBaseTest : public CxCore_CmpBaseTestImpl
{
public:
    CxCore_CmpBaseTest( const char* test_name, const char* test_funcs,
                        int in_range, int _generate_scalars=0 );
};

CxCore_CmpBaseTest::CxCore_CmpBaseTest( const char* test_name, const char* test_funcs,
                                        int _in_range, int _generate_scalars )
    : CxCore_CmpBaseTestImpl( test_name, test_funcs, _in_range, _generate_scalars )
{
    // inherit the default parameters from arithmerical test
    size_list = 0;
    depth_list = 0;
    cn_list = 0;
}


class CxCore_CmpTest : public CxCore_CmpBaseTest
{
public:
    CxCore_CmpTest();
protected:
    void run_func();
};

CxCore_CmpTest::CxCore_CmpTest()
    : CxCore_CmpBaseTest( "cmp-cmp", "cvCmp", 0, 0 )
{
}

void CxCore_CmpTest::run_func()
{
    if(!test_nd)
    {
        cvCmp( test_array[INPUT][0], test_array[INPUT][1],
              test_array[OUTPUT][0], cmp_op );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::compare(cv::cvarrToMatND(test_array[INPUT][0]),
                    cv::cvarrToMatND(test_array[INPUT][1]),
                    c, cmp_op);
    }
}

CxCore_CmpTest cmp_test;


class CxCore_CmpSTest : public CxCore_CmpBaseTest
{
public:
    CxCore_CmpSTest();
protected:
    void run_func();
};

CxCore_CmpSTest::CxCore_CmpSTest()
    : CxCore_CmpBaseTest( "cmp-cmps", "cvCmpS", 0, 4 )
{
}

void CxCore_CmpSTest::run_func()
{
    if(!test_nd)
    {
        cvCmpS( test_array[INPUT][0], gamma.val[0],
            test_array[OUTPUT][0], cmp_op );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::compare(cv::cvarrToMatND(test_array[INPUT][0]),
                    gamma.val[0], c, cmp_op);
    }
}

CxCore_CmpSTest cmps_test;


class CxCore_InRangeTest : public CxCore_CmpBaseTest
{
public:
    CxCore_InRangeTest();
protected:
    void run_func();
};

CxCore_InRangeTest::CxCore_InRangeTest()
    : CxCore_CmpBaseTest( "cmp-inrange", "cvInRange", 1, 0 )
{
}

void CxCore_InRangeTest::run_func()
{
    if(!test_nd)
    {
        cvInRange( test_array[INPUT][0], test_array[INPUT][1],
                  test_array[INPUT][2], test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::inRange(cv::cvarrToMatND(test_array[INPUT][0]),
                    cv::cvarrToMatND(test_array[INPUT][1]),
                    cv::cvarrToMatND(test_array[INPUT][2]),
                    c);
    }
}

CxCore_InRangeTest inrange_test;


class CxCore_InRangeSTest : public CxCore_CmpBaseTest
{
public:
    CxCore_InRangeSTest();
protected:
    void run_func();
};

CxCore_InRangeSTest::CxCore_InRangeSTest()
    : CxCore_CmpBaseTest( "cmp-inranges", "cvInRangeS", 1, 5 )
{
}

void CxCore_InRangeSTest::run_func()
{
    if(!test_nd)
    {
        cvInRangeS( test_array[INPUT][0], alpha, gamma, test_array[OUTPUT][0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::inRange(cv::cvarrToMatND(test_array[INPUT][0]), alpha, gamma, c);
    }
}

CxCore_InRangeSTest inranges_test;


/////////////////////////// convertscale[abs] ////////////////////////////////////////

static const char* cvt_param_names[] = { "size", "scale", "dst_depth", "depth", 0 };
static const char* cvt_abs_param_names[] = { "size", "depth", 0 };
static const int cvt_scale_flags[] = { 0, 1 };

class CxCore_CvtBaseTestImpl : public CxCore_ArithmTestImpl
{
public:
    CxCore_CvtBaseTestImpl( const char* test_name, const char* test_funcs, bool calc_abs );
protected:
    void get_test_array_types_and_sizes( int test_case_idx,
                                         CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                        CvSize** sizes, int** types,
                                        CvSize** whole_sizes, bool *are_images );
    double get_success_error_level( int test_case_idx, int i, int j );

    int prepare_test_case( int test_case_idx );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    int write_default_params( CvFileStorage* fs );

    void prepare_to_validation( int test_case_idx );
};


CxCore_CvtBaseTestImpl::CxCore_CvtBaseTestImpl( const char* test_name,
                                                const char* test_funcs,
                                                bool _calc_abs )
    : CxCore_ArithmTestImpl( test_name, test_funcs, 5, false, _calc_abs )
{
    test_array[INPUT].pop();
    default_timing_param_names = 0;
    cn_list = 0;
}


// unlike many other arithmetic functions, conversion operations support 8s type,
// also, for cvCvtScale output array depth may be arbitrary and
// for cvCvtScaleAbs output depth = CV_8U
void CxCore_CvtBaseTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CxCore_ArithmTestImpl::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvRNG* rng = ts->get_rng();
    int depth = CV_8U, rbits;
    types[INPUT][0] = (types[INPUT][0] & ~CV_MAT_DEPTH_MASK)|
                    cvTsRandInt(rng)%(CV_64F+1);
    if( !calc_abs )
        depth = cvTsRandInt(rng) % (CV_64F+1);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = (types[INPUT][0] & ~CV_MAT_DEPTH_MASK)|depth;

    rbits = cvTsRandInt(rng);
    // check special cases: shift=0 and/or scale=1.
    if( (rbits & 3) == 0 )
        gamma.val[0] = 0;
    if( (rbits & 12) == 0 )
        alpha.val[0] = 1;
}


double CxCore_CvtBaseTestImpl::get_success_error_level( int, int, int )
{
    if( CV_MAT_DEPTH(test_mat[OUTPUT][0].type) <= CV_32S )
        return alpha.val[0] != cvRound(alpha.val[0]) ||
               beta.val[0] != cvRound(beta.val[0]) ||
               gamma.val[0] != cvRound(gamma.val[0]);

    CvScalar l1, h1, l2, h2;
    int stype = CV_MAT_TYPE(test_mat[INPUT][0].type);
    int dtype = CV_MAT_TYPE(test_mat[OUTPUT][0].type);
    get_minmax_bounds( INPUT, 0, stype, &l1, &h1 );
    get_minmax_bounds( OUTPUT, 0, dtype, &l2, &h2 );
    double maxval = 0;
    for( int i = 0; i < 4; i++ )
    {
        maxval = MAX(maxval, fabs(l1.val[i]));
        maxval = MAX(maxval, fabs(h1.val[i]));
        maxval = MAX(maxval, fabs(l2.val[i]));
        maxval = MAX(maxval, fabs(h2.val[i]));
    }
    double max_err = (CV_MAT_DEPTH(stype) == CV_64F || CV_MAT_DEPTH(dtype) == CV_64F ?
        DBL_EPSILON : FLT_EPSILON)*maxval*MAX(fabs(alpha.val[0]), 1.)*100;
    return max_err;
}


void CxCore_CvtBaseTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                    CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CxCore_ArithmTestImpl::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    bool scale = true;
    int dst_depth = CV_8U;
    int cn = CV_MAT_CN(types[INPUT][0]);
    if( !calc_abs )
    {
        scale = cvReadInt( find_timing_param( "scale" ), 1 ) != 0;
        dst_depth = cvTsTypeByName( cvReadString(find_timing_param( "dst_depth" ), "8u") );
    }

    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(dst_depth, cn);

    if( scale )
    {
        alpha.val[0] = 2.1;
        gamma.val[0] = -100.;
    }
    else
    {
        alpha.val[0] = 1.;
        gamma.val[0] = 0.;
    }
}


int CxCore_CvtBaseTestImpl::prepare_test_case( int test_case_idx )
{
    int code = CxCore_ArithmTestImpl::prepare_test_case( test_case_idx );

    if( code > 0 && ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        if( CV_ARE_TYPES_EQ( &test_mat[INPUT][0], &test_mat[OUTPUT][0] ) &&
            !calc_abs && alpha.val[0] == 1 && gamma.val[0] == 0 )
            code = 0; // skip the case when no any transformation is done
    }

    return code;
}


void CxCore_CvtBaseTestImpl::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", alpha.val[0] == 1. && gamma.val[0] == 0. ? "no_scale" : "scale" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_ArithmTestImpl::print_timing_params( test_case_idx, ptr, params_left );
}


int CxCore_CvtBaseTestImpl::write_default_params( CvFileStorage* fs )
{
    int i, code = CxCore_ArithmTestImpl::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    if( !calc_abs )
    {
        start_write_param( fs );
        cvStartWriteStruct( fs, "dst_depth", CV_NODE_SEQ + CV_NODE_FLOW );
        for( i = 0; arithm_depths[i] >= 0; i++ )
            cvWriteString( fs, 0, cvTsGetTypeName(arithm_depths[i]) );
        cvEndWriteStruct(fs);
        write_int_list( fs, "scale", cvt_scale_flags, CV_DIM(cvt_scale_flags) );
    }
    return code;
}


void CxCore_CvtBaseTestImpl::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsAdd( &test_mat[INPUT][0], cvScalarAll(alpha.val[0]), 0, beta,
             cvScalarAll(gamma.val[0]), &test_mat[REF_OUTPUT][0], calc_abs );
}

CxCore_CvtBaseTestImpl cvt_test( "cvt", "", true );


class CxCore_CvtBaseTest : public CxCore_CvtBaseTestImpl
{
public:
    CxCore_CvtBaseTest( const char* test_name, const char* test_funcs, bool calc_abs );
};


CxCore_CvtBaseTest::CxCore_CvtBaseTest( const char* test_name, const char* test_funcs, bool _calc_abs )
    : CxCore_CvtBaseTestImpl( test_name, test_funcs, _calc_abs )
{
    // inherit the default parameters from arithmerical test
    size_list = 0;
    whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;
}


class CxCore_CvtScaleTest : public CxCore_CvtBaseTest
{
public:
    CxCore_CvtScaleTest();
protected:
    void run_func();
};

CxCore_CvtScaleTest::CxCore_CvtScaleTest()
    : CxCore_CvtBaseTest( "cvt-scale", "cvCvtScale", false )
{
    default_timing_param_names = cvt_param_names;
}

void CxCore_CvtScaleTest::run_func()
{
    if(!test_nd)
    {
        cvConvertScale( test_array[INPUT][0], test_array[OUTPUT][0],
                       alpha.val[0], gamma.val[0] );
    }
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::cvarrToMatND(test_array[INPUT][0]).convertTo(c,c.type(),alpha.val[0], gamma.val[0]);
    }
}

CxCore_CvtScaleTest cvtscale_test;


class CxCore_CvtScaleAbsTest : public CxCore_CvtBaseTest
{
public:
    CxCore_CvtScaleAbsTest();
protected:
    void run_func();
};

CxCore_CvtScaleAbsTest::CxCore_CvtScaleAbsTest()
    : CxCore_CvtBaseTest( "cvt-scaleabs", "cvCvtScaleAbs", true )
{
    default_timing_param_names = cvt_abs_param_names;
}

void CxCore_CvtScaleAbsTest::run_func()
{
    if(!test_nd)
    {
        cvConvertScaleAbs( test_array[INPUT][0], test_array[OUTPUT][0],
                       alpha.val[0], gamma.val[0] );
    }
    else
    {
        cv::Mat c = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::convertScaleAbs(cv::cvarrToMat(test_array[INPUT][0]),c,alpha.val[0], gamma.val[0]);
    }
}

CxCore_CvtScaleAbsTest cvtscaleabs_test;


/////////////////////////////// statistics //////////////////////////////////

static const char* stat_param_names[] = { "size", "coi", "channels", "depth", 0 };
static const char* stat_mask_param_names[] = { "size", "coi", "channels", "depth", "use_mask", 0 };
static const char* stat_single_param_names[] = { "size", "channels", "depth", 0 };
static const char* stat_single_mask_param_names[] = { "size", "channels", "depth", "use_mask", 0 };
static const char* stat_coi_modes[] = { "all", "single", 0 };

class CxCore_StatTestImpl : public CvArrTest
{
public:
    CxCore_StatTestImpl( const char* test_name, const char* test_funcs,
                     int _output_count, bool _single_channel,
                     bool _allow_mask=true, bool _is_binary=false );
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    int write_default_params( CvFileStorage* fs );    
    int prepare_test_case( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );

    int coi;
    int output_count;
    bool single_channel;
    bool is_binary;
    bool test_nd;
};


CxCore_StatTestImpl::CxCore_StatTestImpl( const char* test_name,
                        const char* test_funcs, int _output_count,
                        bool _single_channel, bool _allow_mask, bool _is_binary )
    : CvArrTest( test_name, test_funcs, "" ), output_count(_output_count),
    single_channel(_single_channel), is_binary(_is_binary)
{
    test_array[INPUT].push(NULL);
    if( is_binary )
        test_array[INPUT].push(NULL);
    optional_mask = _allow_mask;
    if( optional_mask )
        test_array[MASK].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    coi = 0;

    size_list = arithm_sizes;
    whole_size_list = arithm_whole_sizes;
    depth_list = arithm_depths;
    cn_list = arithm_channels;
    test_nd = false;
}


void CxCore_StatTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                            CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng)%(CV_64F+1);
    int cn = cvTsRandInt(rng) % 4 + 1;
    int j, count = test_array[INPUT].size();
    
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth += depth == CV_8S;

    for( j = 0; j < count; j++ )
        types[INPUT][j] = CV_MAKETYPE(depth, cn);

    // regardless of the test case, the output is always a fixed-size tuple of numbers
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize( output_count, 1 );
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;

    coi = 0;
    cvmat_allowed = true;
    if( cn > 1 && (single_channel || (cvTsRandInt(rng) & 3) == 0) )
    {
        coi = cvTsRandInt(rng) % cn + 1;
        cvmat_allowed = false;
    }
    test_nd = cvTsRandInt(rng) % 3 == 0;
}


void CxCore_StatTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    const char* coi_mode_str = cvReadString(find_timing_param("coi"), single_channel ? "single" : "all");

    // regardless of the test case, the output is always a fixed-size tuple of numbers
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize( output_count, 1 );
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;

    int cn = CV_MAT_CN(types[INPUT][0]);
    coi = 0;
    cvmat_allowed = true;
    if( strcmp( coi_mode_str, "single" ) == 0 )
    {
        CvRNG* rng = ts->get_rng();
        coi = cvTsRandInt(rng) % cn + 1;
        cvmat_allowed = false;
        *are_images = true;
    }
}


int CxCore_StatTestImpl::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    if( !single_channel )
        write_string_list( fs, "coi", stat_coi_modes );
    return code;
}


int CxCore_StatTestImpl::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    
    if( coi && code > 0 )
    {
        int j, count = test_array[INPUT].size();

        if( ts->get_testing_mode() == CvTS::TIMING_MODE && CV_MAT_CN(test_mat[INPUT][0].type) == 1 )
            return 0;

        for( j = 0; j < count; j++ )
        {
            IplImage* img = (IplImage*)test_array[INPUT][j];
            if( img )
                cvSetImageCOI( img, coi );
        }
    }

    return code;
}


void CxCore_StatTestImpl::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", coi > 0 || CV_MAT_CN(test_mat[INPUT][0].type) == 1 ? "single" : "all" );
    ptr += strlen(ptr);
    params_left--;
    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CxCore_StatTestImpl::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0]));
    if( depth == CV_32F )
        return FLT_EPSILON*1000;
    if( depth == CV_64F )
        return DBL_EPSILON*100000;
    else
        return CvArrTest::get_success_error_level( test_case_idx, i, j );
}

CxCore_StatTestImpl stat_test( "stat", "", 0, true, false );


class CxCore_StatTest : public CxCore_StatTestImpl
{
public:
    CxCore_StatTest( const char* test_name, const char* test_funcs,
                     int _output_count, bool _single_channel,
                     bool _allow_mask=1, bool _is_binary=0 );
};

CxCore_StatTest::CxCore_StatTest( const char* test_name, const char* test_funcs,
                     int _output_count, bool _single_channel,
                     bool _allow_mask, bool _is_binary )
    : CxCore_StatTestImpl( test_name, test_funcs, _output_count, _single_channel, _allow_mask, _is_binary )
{
    if( !single_channel )
        default_timing_param_names = optional_mask ? stat_single_mask_param_names : stat_single_param_names;
    else
        default_timing_param_names = optional_mask ? stat_mask_param_names : stat_param_names;
    
    // inherit the default parameters from arithmerical test
    size_list = 0;
    whole_size_list = 0;
    depth_list = 0;
    cn_list = 0;
}

////////////////// sum /////////////////
class CxCore_SumTest : public CxCore_StatTest
{
public:
    CxCore_SumTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CxCore_SumTest::CxCore_SumTest()
    : CxCore_StatTest( "stat-sum", "cvSum", 4 /* CvScalar */, false, false, false )
{
}

double CxCore_SumTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int depth = CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0]));
    if( depth == CV_32F )
        return FLT_EPSILON*1000;
    return DBL_EPSILON*100000;
}


void CxCore_SumTest::run_func()
{
    if(!test_nd || coi)
    {
        *(CvScalar*)(test_mat[OUTPUT][0].data.db) = cvSum(test_array[INPUT][0]);
    }
    else
    {
        *(cv::Scalar*)(test_mat[OUTPUT][0].data.db) = cv::sum(cv::cvarrToMatND(test_array[INPUT][0]));
    }
}

void CxCore_SumTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvScalar mean;
    int nonzero = cvTsMeanStdDevNonZero( &test_mat[INPUT][0], 0, &mean, 0, coi );

    *(CvScalar*)(test_mat[REF_OUTPUT][0].data.db) = mean;
    mean = *(CvScalar*)(test_mat[OUTPUT][0].data.db);

    mean.val[0] /= nonzero;
    mean.val[1] /= nonzero;
    mean.val[2] /= nonzero;
    mean.val[3] /= nonzero;
    *(CvScalar*)(test_mat[OUTPUT][0].data.db) = mean;
}

CxCore_SumTest sum_test;


////////////////// nonzero /////////////////
class CxCore_NonZeroTest : public CxCore_StatTest
{
public:
    CxCore_NonZeroTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx,
                                         CvSize** sizes, int** types );
};


CxCore_NonZeroTest::CxCore_NonZeroTest()
    : CxCore_StatTest( "stat-nonzero", "cvCountNonZero", 1 /* int */, true, false, false )
{
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
}

void CxCore_NonZeroTest::run_func()
{
    if(!test_nd || coi)
    {
        test_mat[OUTPUT][0].data.db[0] = cvCountNonZero(test_array[INPUT][0]);
    }
    else
    {
        test_mat[OUTPUT][0].data.db[0] = cv::countNonZero(cv::cvarrToMatND(test_array[INPUT][0]));
    }
}

void CxCore_NonZeroTest::get_test_array_types_and_sizes( int test_case_idx,
                                              CvSize** sizes, int** types )
{
    CxCore_StatTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[TEMP][0] = CV_8UC1;
    if( CV_MAT_CN(types[INPUT][0]) > 1 )
        types[TEMP][1] = types[INPUT][0] & ~CV_MAT_CN_MASK;
    else
        sizes[TEMP][1] = cvSize(0,0);
}


void CxCore_NonZeroTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* plane = &test_mat[INPUT][0];
    if( CV_MAT_CN(plane->type) > 1 )
    {
        plane = &test_mat[TEMP][1];
        assert( coi > 0 );
        cvTsExtract( &test_mat[INPUT][0], plane, coi-1 );
    }
    cvTsCmpS( plane, 0, &test_mat[TEMP][0], CV_CMP_NE );
    int nonzero = cvTsMeanStdDevNonZero( &test_mat[INPUT][0], &test_mat[TEMP][0], 0, 0, coi );
    test_mat[REF_OUTPUT][0].data.db[0] = nonzero;
}


CxCore_NonZeroTest nonzero_test;


/////////////////// mean //////////////////////
class CxCore_MeanTest : public CxCore_StatTest
{
public:
    CxCore_MeanTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_MeanTest::CxCore_MeanTest()
    : CxCore_StatTest( "stat-mean", "cvAvg", 4 /* CvScalar */, false, true, false )
{
}

void CxCore_MeanTest::run_func()
{
    if(!test_nd || coi)
    {
        *(CvScalar*)(test_mat[OUTPUT][0].data.db) =
            cvAvg(test_array[INPUT][0], test_array[MASK][0]);
    }
    else
    {
        *(cv::Scalar*)(test_mat[OUTPUT][0].data.db) = cv::mean(
                    cv::cvarrToMatND(test_array[INPUT][0]),
                    cv::cvarrToMatND(test_array[MASK][0]));
    }
}

void CxCore_MeanTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvScalar mean;
    cvTsMeanStdDevNonZero( &test_mat[INPUT][0],
        test_array[MASK][0] ? &test_mat[MASK][0] : 0,
        &mean, 0, coi );
    *(CvScalar*)(test_mat[REF_OUTPUT][0].data.db) = mean;
}

CxCore_MeanTest mean_test;


/////////////////// mean_stddev //////////////////////
class CxCore_MeanStdDevTest : public CxCore_StatTest
{
public:
    CxCore_MeanStdDevTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CxCore_MeanStdDevTest::CxCore_MeanStdDevTest()
    : CxCore_StatTest( "stat-mean_stddev", "cvAvgSdv", 8 /* CvScalar x 2 */, false, true, false )
{
}

void CxCore_MeanStdDevTest::run_func()
{
    if(!test_nd || coi)
    {
        cvAvgSdv( test_array[INPUT][0],
                  &((CvScalar*)(test_mat[OUTPUT][0].data.db))[0],
                  &((CvScalar*)(test_mat[OUTPUT][0].data.db))[1],
                  test_array[MASK][0] );
    }
    else
    {
        cv::meanStdDev(cv::cvarrToMatND(test_array[INPUT][0]),
                       ((cv::Scalar*)(test_mat[OUTPUT][0].data.db))[0],
                       ((cv::Scalar*)(test_mat[OUTPUT][0].data.db))[1],
                       cv::cvarrToMatND(test_array[MASK][0]) );
    }
}

double CxCore_MeanStdDevTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0]));
    if( depth < CV_64F && depth != CV_32S )
        return CxCore_StatTest::get_success_error_level( test_case_idx, i, j );
    return DBL_EPSILON*1e6;
}

void CxCore_MeanStdDevTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvScalar mean, stddev;
    int i;
    CvMat* output = &test_mat[OUTPUT][0];
    CvMat* ref_output = &test_mat[REF_OUTPUT][0];
    cvTsMeanStdDevNonZero( &test_mat[INPUT][0],
        test_array[MASK][0] ? &test_mat[MASK][0] : 0,
        &mean, &stddev, coi );
    ((CvScalar*)(ref_output->data.db))[0] = mean;
    ((CvScalar*)(ref_output->data.db))[1] = stddev;
    for( i = 0; i < 4; i++ )
    {
        output->data.db[i] *= output->data.db[i];
        output->data.db[i+4] = output->data.db[i+4]*output->data.db[i+4] + 1000;
        ref_output->data.db[i] *= ref_output->data.db[i];
        ref_output->data.db[i+4] = ref_output->data.db[i+4]*ref_output->data.db[i+4] + 1000;
    }
}

CxCore_MeanStdDevTest mean_stddev_test;


/////////////////// minmaxloc //////////////////////
class CxCore_MinMaxLocTest : public CxCore_StatTest
{
public:
    CxCore_MinMaxLocTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_MinMaxLocTest::CxCore_MinMaxLocTest()
    : CxCore_StatTest( "stat-minmaxloc", "cvMinMaxLoc", 6 /* double x 2 + CvPoint x 2 */, true, true, false )
{
}

void CxCore_MinMaxLocTest::run_func()
{
    CvPoint minloc = {0,0}, maxloc = {0,0};
    double* output = test_mat[OUTPUT][0].data.db;

    cvMinMaxLoc( test_array[INPUT][0],
        output, output+1, &minloc, &maxloc,
        test_array[MASK][0] );
    output[2] = minloc.x;
    output[3] = minloc.y;
    output[4] = maxloc.x;
    output[5] = maxloc.y;
}

void CxCore_MinMaxLocTest::prepare_to_validation( int /*test_case_idx*/ )
{
    double minval = 0, maxval = 0;
    CvPoint minloc = {0,0}, maxloc = {0,0};
    double* ref_output = test_mat[REF_OUTPUT][0].data.db;
    cvTsMinMaxLoc( &test_mat[INPUT][0], test_array[MASK][0] ?
        &test_mat[MASK][0] : 0, &minval, &maxval, &minloc, &maxloc, coi );
    ref_output[0] = minval;
    ref_output[1] = maxval;
    ref_output[2] = minloc.x;
    ref_output[3] = minloc.y;
    ref_output[4] = maxloc.x;
    ref_output[5] = maxloc.y;
}

CxCore_MinMaxLocTest minmaxloc_test;


/////////////////// norm //////////////////////

static const char* stat_norm_param_names[] = { "size", "coi", "norm_type", "channels", "depth", "use_mask", 0 };
static const char* stat_norm_type_names[] = { "Inf", "L1", "L2", "diff_Inf", "diff_L1", "diff_L2", 0 };

class CxCore_NormTest : public CxCore_StatTest
{
public:
    CxCore_NormTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
    void get_test_array_types_and_sizes( int test_case_idx,
                                         CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int /*test_case_idx*/,
        CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images );
    int prepare_test_case( int test_case_idx );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    int write_default_params( CvFileStorage* fs );
    double get_success_error_level( int test_case_idx, int i, int j );
    int norm_type;
};


CxCore_NormTest::CxCore_NormTest()
    : CxCore_StatTest( "stat-norm", "cvNorm", 1 /* double */, false, true, true )
{
    test_array[TEMP].push(NULL);
    default_timing_param_names = stat_norm_param_names;
}


double CxCore_NormTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0]));
    if( (depth == CV_16U || depth == CV_16S) /*&& (norm_type&3) != CV_C*/ )
        return 1e-4;
    else
        return CxCore_StatTest::get_success_error_level( test_case_idx, i, j );
}


void CxCore_NormTest::get_test_array_types_and_sizes( int test_case_idx,
                                               CvSize** sizes, int** types )
{
    int intype;
    int norm_kind;
    CxCore_StatTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    norm_type = cvTsRandInt(ts->get_rng()) % 3; // CV_C, CV_L1 or CV_L2
    norm_kind = cvTsRandInt(ts->get_rng()) % 3; // simple, difference or relative difference
    if( norm_kind == 0 )
        sizes[INPUT][1] = cvSize(0,0);
    norm_type = (1 << norm_type) | (norm_kind*8);
    intype = types[INPUT][0];
    if( CV_MAT_CN(intype) > 1 && coi == 0 )
        sizes[MASK][0] = cvSize(0,0);
    sizes[TEMP][0] = cvSize(0,0);
    if( (norm_type & (CV_DIFF|CV_RELATIVE)) && CV_MAT_DEPTH(intype) <= CV_32F )
    {
        sizes[TEMP][0] = sizes[INPUT][0];
        types[TEMP][0] = (intype & ~CV_MAT_DEPTH_MASK)|
            (CV_MAT_DEPTH(intype) < CV_32F ? CV_32S : CV_64F);
    }
}


void CxCore_NormTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_StatTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* norm_type_str = cvReadString( find_timing_param( "norm_type" ), "L2" );
    bool diff = false;
    if( strncmp( norm_type_str, "diff_", 5 ) == 0 )
    {
        diff = true;
        norm_type_str += 5;
    }
    
    if( strcmp( norm_type_str, "L1" ) == 0 )
        norm_type = CV_L1;
    else if( strcmp( norm_type_str, "L2" ) == 0 )
        norm_type = CV_L2;
    else
        norm_type = CV_C;

    if( diff )
        norm_type += CV_DIFF;
    else
        sizes[INPUT][1] = cvSize(0,0);
}


int CxCore_NormTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_StatTest::prepare_test_case( test_case_idx );
    if( code > 0 && ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        // currently it is not supported
        if( test_array[MASK][0] && CV_MAT_CN(test_mat[INPUT][0].type) > 1 && coi == 0 )
            return 0;
    }
    return code;
}


int CxCore_NormTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_StatTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "norm_type", stat_norm_type_names );
    return code;
}


void CxCore_NormTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    int nt = norm_type & CV_NORM_MASK;
    sprintf( ptr, "%s%s,", norm_type & CV_DIFF ? "diff_" : "",
             nt == CV_C ? "Inf" : nt == CV_L1 ? "L1" : "L2" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_StatTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CxCore_NormTest::run_func()
{
    if(!test_nd || coi)
    {
        test_mat[OUTPUT][0].data.db[0] = cvNorm( test_array[INPUT][0],
                 test_array[INPUT][1], norm_type, test_array[MASK][0] );
    }
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND b = cv::cvarrToMatND(test_array[INPUT][1]);
        cv::MatND mask = cv::cvarrToMatND(test_array[MASK][0]);
        test_mat[OUTPUT][0].data.db[0] = b.data ?
            cv::norm( a, b, norm_type, mask ) :
            cv::norm( a, norm_type, mask );
    }
}

void CxCore_NormTest::prepare_to_validation( int /*test_case_idx*/ )
{
    double a_norm = 0, b_norm = 0;
    CvMat* a = &test_mat[INPUT][0];
    CvMat* b = &test_mat[INPUT][1];
    CvMat* mask = test_array[MASK][0] ? &test_mat[MASK][0] : 0;
    CvMat* diff = a;

    if( norm_type & (CV_DIFF|CV_RELATIVE) )
    {
        diff = test_array[TEMP][0] ? &test_mat[TEMP][0] : a;
        cvTsAdd( a, cvScalarAll(1.), b, cvScalarAll(-1.),
                 cvScalarAll(0.), diff, 0 );
    }
    a_norm = cvTsNorm( diff, mask, norm_type & CV_NORM_MASK, coi );
    if( norm_type & CV_RELATIVE )
    {
        b_norm = cvTsNorm( b, mask, norm_type & CV_NORM_MASK, coi );
        a_norm /= (b_norm + DBL_EPSILON );
    }
    test_mat[REF_OUTPUT][0].data.db[0] = a_norm;
}

CxCore_NormTest norm_test;

// TODO: repeat(?), reshape(?), lut

/* End of file. */
