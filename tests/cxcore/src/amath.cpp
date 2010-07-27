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
/////////////////// tests for matrix operations and math functions ///////////////////////
//////////////////////////////////////////////////////////////////////////////////////////

#include "cxcoretest.h"
#include <float.h>
#include <math.h>

/// !!! NOTE !!! These tests happily avoid overflow cases & out-of-range arguments
/// so that output arrays contain neigher Inf's nor Nan's.
/// Handling such cases would require special modification of check function
/// (validate_test_results) => TBD.
/// Also, need some logarithmic-scale generation of input data. Right now it is done (in some tests)
/// by generating min/max boundaries for random data in logarimithic scale, but
/// within the same test case all the input array elements are of the same order.

static const CvSize math_sizes[] = {{10,1}, {100,1}, {10000,1}, {-1,-1}};
static const int math_depths[] = { CV_32F, CV_64F, -1 };
static const char* math_param_names[] = { "size", "depth", 0 };

static const CvSize matrix_sizes[] = {{3,3}, {4,4}, {10,10}, {30,30}, {100,100}, {500,500}, {-1,-1}};

class CxCore_MathTestImpl : public CvArrTest
{
public:
    CxCore_MathTestImpl( const char* test_name, const char* test_funcs );
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int /*test_case_idx*/, int i, int j );
    bool test_nd;
};


CxCore_MathTestImpl::CxCore_MathTestImpl( const char* test_name, const char* test_funcs )
    : CvArrTest( test_name, test_funcs, "" )
{
    optional_mask = false;

    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);

    default_timing_param_names = math_param_names;

    size_list = math_sizes;
    whole_size_list = 0;
    depth_list = math_depths;
    cn_list = 0;
    test_nd = false;
}


double CxCore_MathTestImpl::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    return CV_MAT_DEPTH(test_mat[i][j].type) == CV_32F ? FLT_EPSILON*128 : DBL_EPSILON*1024;
}


void CxCore_MathTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                      CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng)%2 + CV_32F;
    int cn = cvTsRandInt(rng) % 4 + 1, type = CV_MAKETYPE(depth, cn);
    int i, j;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        for( j = 0; j < count; j++ )
            types[i][j] = type;
    }
    test_nd = cvTsRandInt(rng)%3 == 0;
}

CxCore_MathTestImpl math_test( "math", "" );


class CxCore_MathTest : public CxCore_MathTestImpl
{
public:
    CxCore_MathTest( const char* test_name, const char* test_funcs );
};


CxCore_MathTest::CxCore_MathTest( const char* test_name, const char* test_funcs )
    : CxCore_MathTestImpl( test_name, test_funcs )
{
    size_list = 0;
    depth_list = 0;
}


////////// exp /////////////
class CxCore_ExpTest : public CxCore_MathTest
{
public:
    CxCore_ExpTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    double get_success_error_level( int /*test_case_idx*/, int i, int j );
    int prepare_test_case( int test_case );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int out_type;
};


CxCore_ExpTest::CxCore_ExpTest()
    : CxCore_MathTest( "math-exp", "cvExp" )
{
    out_type = 0;
}


double CxCore_ExpTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    int in_depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    int out_depth = CV_MAT_DEPTH(test_mat[OUTPUT][0].type);
    int min_depth = MIN(in_depth, out_depth);
    return min_depth == CV_32F ? 1e-5 : 1e-8;
}


void CxCore_ExpTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CxCore_MathTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    out_type = types[OUTPUT][0];
    /*if( CV_MAT_DEPTH(types[INPUT][0]) == CV_32F && (cvRandInt(ts->get_rng()) & 3) == 0 )
        types[OUTPUT][0] = types[REF_OUTPUT][0] =
            out_type = (types[INPUT][0] & ~CV_MAT_DEPTH_MASK)|CV_64F;*/
}

void CxCore_ExpTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    double l = cvTsRandReal(ts->get_rng())*10+1;
    double u = cvTsRandReal(ts->get_rng())*10+1;
    l *= -l;
    u *= u;
    *low = cvScalarAll(l);
    *high = cvScalarAll(CV_MAT_DEPTH(out_type)==CV_64F? u : u*0.5);
}

int CxCore_ExpTest::prepare_test_case( int test_case )
{
    int code = CxCore_MathTest::prepare_test_case(test_case);
    if( code < 0 )
        return code;

    CvRNG* rng = ts->get_rng();

    int i, j, k, count = cvTsRandInt(rng) % 10;
    CvMat* src = &test_mat[INPUT][0];
    int depth = CV_MAT_DEPTH(src->type);

    // add some extremal values
    for( k = 0; k < count; k++ )
    {
        i = cvTsRandInt(rng) % src->rows;
        j = cvTsRandInt(rng) % (src->cols*CV_MAT_CN(src->type));
        int sign = cvTsRandInt(rng) % 2 ? 1 : -1;
        if( depth == CV_32F )
            ((float*)(src->data.ptr + src->step*i))[j] = FLT_MAX*sign;
        else
            ((double*)(src->data.ptr + src->step*i))[j] = DBL_MAX*sign;
    }

    return code;
}


void CxCore_ExpTest::run_func()
{
    if(!test_nd)
        cvExp( test_array[INPUT][0], test_array[OUTPUT][0] );
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND b = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::exp(a, b);
    }
}


void CxCore_ExpTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* a = &test_mat[INPUT][0];
    CvMat* b = &test_mat[REF_OUTPUT][0];

    int a_depth = CV_MAT_DEPTH(a->type);
    int b_depth = CV_MAT_DEPTH(b->type);
    int ncols = test_mat[INPUT][0].cols*CV_MAT_CN(a->type);
    int i, j;

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + i*a->step;
        uchar* b_data = b->data.ptr + i*b->step;

        if( a_depth == CV_32F && b_depth == CV_32F )
        {
            for( j = 0; j < ncols; j++ )
                ((float*)b_data)[j] = (float)exp((double)((float*)a_data)[j]);
        }
        else if( a_depth == CV_32F && b_depth == CV_64F )
        {
            for( j = 0; j < ncols; j++ )
                ((double*)b_data)[j] = exp((double)((float*)a_data)[j]);
        }
        else
        {
            assert( a_depth == CV_64F && b_depth == CV_64F );
            for( j = 0; j < ncols; j++ )
                ((double*)b_data)[j] = exp(((double*)a_data)[j]);
        }
    }
}

CxCore_ExpTest exp_test;


////////// log /////////////
class CxCore_LogTest : public CxCore_MathTest
{
public:
    CxCore_LogTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_LogTest::CxCore_LogTest()
    : CxCore_MathTest( "math-log", "cvLog" )
{
}


void CxCore_LogTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CxCore_MathTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    /*if( CV_MAT_DEPTH(types[INPUT][0]) == CV_32F && (cvRandInt(ts->get_rng()) & 3) == 0 )
        types[INPUT][0] = (types[INPUT][0] & ~CV_MAT_DEPTH_MASK)|CV_64F;*/
}


void CxCore_LogTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    double l = cvTsRandReal(ts->get_rng())*15-5;
    double u = cvTsRandReal(ts->get_rng())*15-5;
    double t;
    l = exp(l);
    u = exp(u);
    if( l > u )
        CV_SWAP( l, u, t );
    *low = cvScalarAll(l);
    *high = cvScalarAll(u);
}


void CxCore_LogTest::run_func()
{
    if(!test_nd)
        cvLog( test_array[INPUT][0], test_array[OUTPUT][0] );
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND b = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::log(a, b);
    }
}


void CxCore_LogTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* a = &test_mat[INPUT][0];
    CvMat* b = &test_mat[REF_OUTPUT][0];

    int a_depth = CV_MAT_DEPTH(a->type);
    int b_depth = CV_MAT_DEPTH(b->type);
    int ncols = test_mat[INPUT][0].cols*CV_MAT_CN(a->type);
    int i, j;

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + i*a->step;
        uchar* b_data = b->data.ptr + i*b->step;

        if( a_depth == CV_32F && b_depth == CV_32F )
        {
            for( j = 0; j < ncols; j++ )
                ((float*)b_data)[j] = (float)log((double)((float*)a_data)[j]);
        }
        else if( a_depth == CV_64F && b_depth == CV_32F )
        {
            for( j = 0; j < ncols; j++ )
                ((float*)b_data)[j] = (float)log(((double*)a_data)[j]);
        }
        else
        {
            assert( a_depth == CV_64F && b_depth == CV_64F );
            for( j = 0; j < ncols; j++ )
                ((double*)b_data)[j] = log(((double*)a_data)[j]);
        }
    }
}

CxCore_LogTest log_test;


////////// pow /////////////

static const double math_pow_values[] = { 2., 5., 0.5, -0.5, 1./3, -1./3, CV_PI };
static const char* math_pow_param_names[] = { "size", "power", "depth", 0 };
static const int math_pow_depths[] = { CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, -1 };

class CxCore_PowTest : public CxCore_MathTest
{
public:
    CxCore_PowTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int write_default_params( CvFileStorage* fs );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void run_func();
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int test_case_idx );
    double get_success_error_level( int test_case_idx, int i, int j );
    double power;
};


CxCore_PowTest::CxCore_PowTest()
    : CxCore_MathTest( "math-pow", "cvPow" )
{
    power = 0;
    default_timing_param_names = math_pow_param_names;
    depth_list = math_pow_depths;
}


void CxCore_PowTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % (CV_64F+1);
    int cn = cvTsRandInt(rng) % 4 + 1;
    int i, j;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth += depth == CV_8S;

    if( depth < CV_32F || cvTsRandInt(rng)%8 == 0 )
        // integer power
        power = (int)(cvTsRandInt(rng)%21 - 10);
    else
    {
        i = cvTsRandInt(rng)%17;
        power = i == 16 ? 1./3 : i == 15 ? 0.5 : i == 14 ? -0.5 : cvTsRandReal(rng)*10 - 5;
    }

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        int type = CV_MAKETYPE(depth, cn);
        for( j = 0; j < count; j++ )
            types[i][j] = type;
    }
    test_nd = cvTsRandInt(rng)%3 == 0;
}


void CxCore_PowTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MathTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    power = cvReadReal( find_timing_param( "power" ), 0.2 );
}


int CxCore_PowTest::write_default_params( CvFileStorage* fs )
{
    int i, code = CxCore_MathTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    start_write_param( fs );
    cvStartWriteStruct( fs, "power", CV_NODE_SEQ + CV_NODE_FLOW );
    for( i = 0; i < CV_DIM(math_pow_values); i++ )
        cvWriteReal( fs, 0, math_pow_values[i] );
    cvEndWriteStruct(fs);
    return code;
}


int CxCore_PowTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MathTest::prepare_test_case( test_case_idx );
    if( code > 0 && ts->get_testing_mode() == CvTS::TIMING_MODE )
    {
        if( cvRound(power) != power && CV_MAT_DEPTH(test_mat[INPUT][0].type) < CV_32F )
            return 0;
    }
    return code;
}


void CxCore_PowTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%g,", power );
    ptr += strlen(ptr);
    params_left--;
    CxCore_MathTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CxCore_PowTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int type = cvGetElemType( test_array[i][j] );
    if( CV_MAT_DEPTH(type) < CV_32F )
        return power == cvRound(power) && power >= 0 ? 0 : 1;
    else
        return CxCore_MathTest::get_success_error_level( test_case_idx, i, j );
}


void CxCore_PowTest::get_minmax_bounds( int /*i*/, int /*j*/, int type, CvScalar* low, CvScalar* high )
{
    double l, u = cvTsRandInt(ts->get_rng())%1000 + 1;
    if( power > 0 )
    {
        double mval = cvTsMaxVal(type);
        double u1 = pow(mval,1./power)*2;
        u = MIN(u,u1);
    }

    l = power == cvRound(power) ? -u : FLT_EPSILON;
    *low = cvScalarAll(l);
    *high = cvScalarAll(u);
}


void CxCore_PowTest::run_func()
{
    if(!test_nd)
    {
        if( fabs(power-1./3) <= DBL_EPSILON && CV_MAT_DEPTH(test_mat[INPUT][0].type) == CV_32F )
        {
            cv::Mat a(&test_mat[INPUT][0]), b(&test_mat[OUTPUT][0]);
            
            a = a.reshape(1);
            b = b.reshape(1);
            for( int i = 0; i < a.rows; i++ )
            {
                b.at<float>(i,0) = (float)fabs(cvCbrt(a.at<float>(i,0)));
                for( int j = 1; j < a.cols; j++ )
                    b.at<float>(i,j) = (float)fabs(cv::cubeRoot(a.at<float>(i,j)));
            }
        }
        else
            cvPow( test_array[INPUT][0], test_array[OUTPUT][0], power );
    }
    else
    {
        cv::MatND a = cv::cvarrToMatND(test_array[INPUT][0]);
        cv::MatND b = cv::cvarrToMatND(test_array[OUTPUT][0]);
        if(power == 0.5)
            cv::sqrt(a, b);
        else
            cv::pow(a, power, b);
    }
}


inline static int ipow( int a, int power )
{
    int b = 1;
    while( power > 0 )
    {
        if( power&1 )
            b *= a, power--;
        else
            a *= a, power >>= 1;
    }
    return b;
}


inline static double ipow( double a, int power )
{
    double b = 1.;
    while( power > 0 )
    {
        if( power&1 )
            b *= a, power--;
        else
            a *= a, power >>= 1;
    }
    return b;
}


void CxCore_PowTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* a = &test_mat[INPUT][0];
    CvMat* b = &test_mat[REF_OUTPUT][0];

    int depth = CV_MAT_DEPTH(a->type);
    int ncols = test_mat[INPUT][0].cols*CV_MAT_CN(a->type);
    int ipower = cvRound(power), apower = abs(ipower);
    int i, j;

    for( i = 0; i < a->rows; i++ )
    {
        uchar* a_data = a->data.ptr + i*a->step;
        uchar* b_data = b->data.ptr + i*b->step;

        switch( depth )
        {
        case CV_8U:
            if( ipower < 0 )
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((uchar*)a_data)[j];
                    ((uchar*)b_data)[j] = (uchar)(val <= 1 ? val :
                                        val == 2 && ipower == -1 ? 1 : 0);
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((uchar*)a_data)[j];
                    val = ipow( val, ipower );
                    ((uchar*)b_data)[j] = CV_CAST_8U(val);
                }
            break;
        case CV_8S:
            if( ipower < 0 )
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((char*)a_data)[j];
                    ((char*)b_data)[j] = (char)((val&~1)==0 ? val :
                                          val ==-1 ? 1-2*(ipower&1) :
                                          val == 2 && ipower == -1 ? 1 : 0);
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((char*)a_data)[j];
                    val = ipow( val, ipower );
                    ((char*)b_data)[j] = CV_CAST_8S(val);
                }
            break;
        case CV_16U:
            if( ipower < 0 )
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((ushort*)a_data)[j];
                    ((ushort*)b_data)[j] = (ushort)((val&~1)==0 ? val :
                                          val ==-1 ? 1-2*(ipower&1) :
                                          val == 2 && ipower == -1 ? 1 : 0);
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((ushort*)a_data)[j];
                    val = ipow( val, ipower );
                    ((ushort*)b_data)[j] = CV_CAST_16U(val);
                }
            break;
        case CV_16S:
            if( ipower < 0 )
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((short*)a_data)[j];
                    ((short*)b_data)[j] = (short)((val&~1)==0 ? val :
                                          val ==-1 ? 1-2*(ipower&1) :
                                          val == 2 && ipower == -1 ? 1 : 0);
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((short*)a_data)[j];
                    val = ipow( val, ipower );
                    ((short*)b_data)[j] = CV_CAST_16S(val);
                }
            break;
        case CV_32S:
            if( ipower < 0 )
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((int*)a_data)[j];
                    ((int*)b_data)[j] = (val&~1)==0 ? val :
                                        val ==-1 ? 1-2*(ipower&1) :
                                        val == 2 && ipower == -1 ? 1 : 0;
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    int val = ((int*)a_data)[j];
                    val = ipow( val, ipower );
                    ((int*)b_data)[j] = val;
                }
            break;
        case CV_32F:
            if( power != ipower )
                for( j = 0; j < ncols; j++ )
                {
                    double val = ((float*)a_data)[j];
                    val = pow( fabs(val), power );
                    ((float*)b_data)[j] = CV_CAST_32F(val);
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    double val = ((float*)a_data)[j];
                    if( ipower < 0 )
                        val = 1./val;
                    val = ipow( val, apower );
                    ((float*)b_data)[j] = (float)val;
                }
            break;
        case CV_64F:
            if( power != ipower )
                for( j = 0; j < ncols; j++ )
                {
                    double val = ((double*)a_data)[j];
                    val = pow( fabs(val), power );
                    ((double*)b_data)[j] = CV_CAST_64F(val);
                }
            else
                for( j = 0; j < ncols; j++ )
                {
                    double val = ((double*)a_data)[j];
                    if( ipower < 0 )
                        val = 1./val;
                    val = ipow( val, apower );
                    ((double*)b_data)[j] = (double)val;
                }
            break;
        }
    }
}

CxCore_PowTest pow_test;



////////// cart2polar /////////////
class CxCore_CartToPolarTest : public CxCore_MathTest
{
public:
    CxCore_CartToPolarTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int use_degrees;
};


CxCore_CartToPolarTest::CxCore_CartToPolarTest()
    : CxCore_MathTest( "math-cart2polar", "cvCartToPolar" )
{
    use_degrees = 0;
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
}


void CxCore_CartToPolarTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CxCore_MathTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    use_degrees = cvTsRandInt(rng) & 1;
    if( cvTsRandInt(rng) % 4 == 0 ) // check missing magnitude/angle cases
    {
        int idx = cvTsRandInt(rng) & 1;
        sizes[OUTPUT][idx] = sizes[REF_OUTPUT][idx] = cvSize(0,0);
    }
}


void CxCore_CartToPolarTest::run_func()
{
    if(!test_nd)
    {
        cvCartToPolar( test_array[INPUT][0], test_array[INPUT][1],
                    test_array[OUTPUT][0], test_array[OUTPUT][1], use_degrees );
    }
    else
    {
        cv::Mat X = cv::cvarrToMat(test_array[INPUT][0]);
        cv::Mat Y = cv::cvarrToMat(test_array[INPUT][1]);
        cv::Mat mag = test_array[OUTPUT][0] ? cv::cvarrToMat(test_array[OUTPUT][0]) : cv::Mat();
        cv::Mat ph = test_array[OUTPUT][1] ? cv::cvarrToMat(test_array[OUTPUT][1]) : cv::Mat();
        if(!mag.data)
            cv::phase(X, Y, ph, use_degrees != 0);
        else if(!ph.data)
            cv::magnitude(X, Y, mag);
        else
            cv::cartToPolar(X, Y, mag, ph, use_degrees != 0);
    }
}


double CxCore_CartToPolarTest::get_success_error_level( int test_case_idx, int i, int j )
{
    return j == 1 ? 0.5*(use_degrees ? 1 : CV_PI/180.) :
        CxCore_MathTest::get_success_error_level( test_case_idx, i, j );
}


void CxCore_CartToPolarTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* x = &test_mat[INPUT][0];
    CvMat* y = &test_mat[INPUT][1];
    CvMat* mag = test_array[REF_OUTPUT][0] ? &test_mat[REF_OUTPUT][0] : 0;
    CvMat* angle = test_array[REF_OUTPUT][1] ? &test_mat[REF_OUTPUT][1] : 0;
    double C = use_degrees ? 180./CV_PI : 1.;

    int depth = CV_MAT_DEPTH(x->type);
    int ncols = x->cols*CV_MAT_CN(x->type);
    int i, j;

    for( i = 0; i < x->rows; i++ )
    {
        uchar* x_data = x->data.ptr + i*x->step;
        uchar* y_data = y->data.ptr + i*y->step;
        uchar* mag_data = mag ? mag->data.ptr + i*mag->step : 0;
        uchar* angle_data = angle ? angle->data.ptr + i*angle->step : 0;

        if( depth == CV_32F )
        {
            for( j = 0; j < ncols; j++ )
            {
                double xval = ((float*)x_data)[j];
                double yval = ((float*)y_data)[j];

                if( mag_data )
                    ((float*)mag_data)[j] = (float)sqrt(xval*xval + yval*yval);
                if( angle_data )
                {
                    double a = atan2( yval, xval );
                    if( a < 0 )
                        a += CV_PI*2;
                    a *= C;
                    ((float*)angle_data)[j] = (float)a;
                }
            }
        }
        else
        {
            assert( depth == CV_64F );
            for( j = 0; j < ncols; j++ )
            {
                double xval = ((double*)x_data)[j];
                double yval = ((double*)y_data)[j];

                if( mag_data )
                    ((double*)mag_data)[j] = sqrt(xval*xval + yval*yval);
                if( angle_data )
                {
                    double a = atan2( yval, xval );
                    if( a < 0 )
                        a += CV_PI*2;
                    a *= C;
                    ((double*)angle_data)[j] = a;
                }
            }
        }
    }

    if( angle )
    {
        // hack: increase angle value by 1 (so that alpha becomes 1+alpha)
        // to hide large relative errors in case of very small angles
        cvTsAdd( &test_mat[OUTPUT][1], cvScalarAll(1.), 0, cvScalarAll(0.),
                 cvScalarAll(1.), &test_mat[OUTPUT][1], 0 );
        cvTsAdd( &test_mat[REF_OUTPUT][1], cvScalarAll(1.), 0, cvScalarAll(0.),
                 cvScalarAll(1.), &test_mat[REF_OUTPUT][1], 0 );
    }
}

CxCore_CartToPolarTest cart2polar_test;



////////// polar2cart /////////////
class CxCore_PolarToCartTest : public CxCore_MathTest
{
public:
    CxCore_PolarToCartTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int use_degrees;
};


CxCore_PolarToCartTest::CxCore_PolarToCartTest()
    : CxCore_MathTest( "math-polar2cart", "cvPolarToCart" )
{
    use_degrees = 0;
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
}


void CxCore_PolarToCartTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CxCore_MathTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    use_degrees = cvTsRandInt(rng) & 1;
    if( cvTsRandInt(rng) % 4 == 0 ) // check missing magnitude case
        sizes[INPUT][1] = cvSize(0,0);

    if( cvTsRandInt(rng) % 4 == 0 ) // check missing x/y cases
    {
        int idx = cvTsRandInt(rng) & 1;
        sizes[OUTPUT][idx] = sizes[REF_OUTPUT][idx] = cvSize(0,0);
    }
}


void CxCore_PolarToCartTest::run_func()
{
    if(!test_nd)
    {
        cvPolarToCart( test_array[INPUT][1], test_array[INPUT][0],
                    test_array[OUTPUT][0], test_array[OUTPUT][1], use_degrees );
    }
    else
    {
        cv::Mat X = test_array[OUTPUT][0] ? cv::cvarrToMat(test_array[OUTPUT][0]) : cv::Mat();
        cv::Mat Y = test_array[OUTPUT][1] ? cv::cvarrToMat(test_array[OUTPUT][1]) : cv::Mat();
        cv::Mat mag = test_array[INPUT][1] ? cv::cvarrToMat(test_array[INPUT][1]) : cv::Mat();
        cv::Mat ph = test_array[INPUT][0] ? cv::cvarrToMat(test_array[INPUT][0]) : cv::Mat();
        cv::polarToCart(mag, ph, X, Y, use_degrees != 0);
    }
}


double CxCore_PolarToCartTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return FLT_EPSILON*100;
}


void CxCore_PolarToCartTest::prepare_to_validation( int /*test_case_idx*/ )
{
    CvMat* x = test_array[REF_OUTPUT][0] ? &test_mat[REF_OUTPUT][0] : 0;
    CvMat* y = test_array[REF_OUTPUT][1] ? &test_mat[REF_OUTPUT][1] : 0;
    CvMat* angle = &test_mat[INPUT][0];
    CvMat* mag = test_array[INPUT][1] ? &test_mat[INPUT][1] : 0;
    double C = use_degrees ? CV_PI/180. : 1.;

    int depth = CV_MAT_DEPTH(angle->type);
    int ncols = angle->cols*CV_MAT_CN(angle->type);
    int i, j;

    for( i = 0; i < angle->rows; i++ )
    {
        uchar* x_data = x ? x->data.ptr + i*x->step : 0;
        uchar* y_data = y ? y->data.ptr + i*y->step : 0;
        uchar* mag_data = mag ? mag->data.ptr + i*mag->step : 0;
        uchar* angle_data = angle->data.ptr + i*angle->step;

        if( depth == CV_32F )
        {
            for( j = 0; j < ncols; j++ )
            {
                double a = ((float*)angle_data)[j]*C;
                double m = mag_data ? ((float*)mag_data)[j] : 1.;

                if( x_data )
                    ((float*)x_data)[j] = (float)(m*cos(a));
                if( y_data )
                    ((float*)y_data)[j] = (float)(m*sin(a));
            }
        }
        else
        {
            assert( depth == CV_64F );
            for( j = 0; j < ncols; j++ )
            {
                double a = ((double*)angle_data)[j]*C;
                double m = mag_data ? ((double*)mag_data)[j] : 1.;

                if( x_data )
                    ((double*)x_data)[j] = m*cos(a);
                if( y_data )
                    ((double*)y_data)[j] = m*sin(a);
            }
        }
    }
}

CxCore_PolarToCartTest polar2cart_test;

///////////////////////////////////////// matrix tests ////////////////////////////////////////////

static const int matrix_all_depths[] = { CV_8U, CV_16U, CV_16S, CV_32S, CV_32F, CV_64F, -1 };

class CxCore_MatrixTestImpl : public CvArrTest
{
public:
    CxCore_MatrixTestImpl( const char* test_name, const char* test_funcs, int in_count, int out_count,
                       bool allow_int, bool scalar_output, int max_cn );
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    double get_success_error_level( int test_case_idx, int i, int j );
    bool allow_int;
    bool scalar_output;
    int max_cn;
};


CxCore_MatrixTestImpl::CxCore_MatrixTestImpl( const char* test_name, const char* test_funcs,
                                      int in_count, int out_count,
                                      bool _allow_int, bool _scalar_output, int _max_cn )
    : CvArrTest( test_name, test_funcs, "" ),
    allow_int(_allow_int), scalar_output(_scalar_output), max_cn(_max_cn)
{
    int i;
    for( i = 0; i < in_count; i++ )
        test_array[INPUT].push(NULL);

    for( i = 0; i < out_count; i++ )
    {
        test_array[OUTPUT].push(NULL);
        test_array[REF_OUTPUT].push(NULL);
    }

    element_wise_relative_error = false;

    default_timing_param_names = math_param_names;

    size_list = (CvSize*)matrix_sizes;
    whole_size_list = 0;
    depth_list = (int*)math_depths;
    cn_list = 0;
}


void CxCore_MatrixTestImpl::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % (allow_int ? CV_64F+1 : 2);
    int cn = cvTsRandInt(rng) % max_cn + 1;
    int i, j;

    if( allow_int )
        depth += depth == CV_8S;
    else
        depth += CV_32F;

    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    for( i = 0; i < max_arr; i++ )
    {
        int count = test_array[i].size();
        int flag = (i == OUTPUT || i == REF_OUTPUT) && scalar_output;
        int type = !flag ? CV_MAKETYPE(depth, cn) : CV_64FC1;

        for( j = 0; j < count; j++ )
        {
            types[i][j] = type;
            if( flag )
                sizes[i][j] = cvSize( 4, 1 );
        }
    }
}


void CxCore_MatrixTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx,
                              sizes, types, whole_sizes, are_images );
    if( scalar_output )
    {
        types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_64FC1;
        sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize( 4, 1 );
        whole_sizes[OUTPUT][0] = whole_sizes[REF_OUTPUT][0] = cvSize( 4, 1 );
    }
}


double CxCore_MatrixTestImpl::get_success_error_level( int test_case_idx, int i, int j )
{
    int input_depth = CV_MAT_DEPTH(cvGetElemType( test_array[INPUT][0] ));
    double input_precision = input_depth < CV_32F ? 0 : input_depth == CV_32F ?
                            5e-5 : 1e-10;
    double output_precision = CvArrTest::get_success_error_level( test_case_idx, i, j );
    return MAX(input_precision, output_precision);
}

CxCore_MatrixTestImpl matrix_test( "matrix", "", 0, 0, false, false, 0 );


class CxCore_MatrixTest : public CxCore_MatrixTestImpl
{
public:
    CxCore_MatrixTest( const char* test_name, const char* test_funcs, int in_count, int out_count,
                       bool allow_int, bool scalar_output, int max_cn );
};


CxCore_MatrixTest::CxCore_MatrixTest( const char* test_name, const char* test_funcs,
                                      int in_count, int out_count, bool _allow_int,
                                      bool _scalar_output, int _max_cn )
    : CxCore_MatrixTestImpl( test_name, test_funcs, in_count, out_count,
                             _allow_int, _scalar_output, _max_cn )
{
    size_list = 0;
    depth_list = 0;
}


///////////////// Trace /////////////////////

class CxCore_TraceTest : public CxCore_MatrixTest
{
public:
    CxCore_TraceTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_TraceTest::CxCore_TraceTest() :
    CxCore_MatrixTest( "matrix-trace", "cvTrace", 1, 1, true, true, 4 )
{
}


void CxCore_TraceTest::run_func()
{
    *((CvScalar*)(test_mat[OUTPUT][0].data.db)) = cvTrace(test_array[INPUT][0]);
}


void CxCore_TraceTest::prepare_to_validation( int )
{
    CvMat* mat = &test_mat[INPUT][0];
    int i, j, count = MIN( mat->rows, mat->cols );
    CvScalar trace = {{0,0,0,0}};

    for( i = 0; i < count; i++ )
    {
        CvScalar el = cvGet2D( mat, i, i );
        for( j = 0; j < 4; j++ )
            trace.val[j] += el.val[j];
    }

    *((CvScalar*)(test_mat[REF_OUTPUT][0].data.db)) = trace;
}

CxCore_TraceTest trace_test;


///////// dotproduct //////////

class CxCore_DotProductTest : public CxCore_MatrixTest
{
public:
    CxCore_DotProductTest();
protected:
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_DotProductTest::CxCore_DotProductTest() :
    CxCore_MatrixTest( "matrix-dotproduct", "cvDotProduct", 2, 1, true, true, 4 )
{
    depth_list = matrix_all_depths;
}


void CxCore_DotProductTest::run_func()
{
    *((CvScalar*)(test_mat[OUTPUT][0].data.ptr)) =
        cvRealScalar(cvDotProduct( test_array[INPUT][0], test_array[INPUT][1] ));
}


void CxCore_DotProductTest::prepare_to_validation( int )
{
    *((CvScalar*)(test_mat[REF_OUTPUT][0].data.ptr)) =
        cvRealScalar(cvTsCrossCorr( &test_mat[INPUT][0], &test_mat[INPUT][1] ));
}

CxCore_DotProductTest dotproduct_test;


///////// crossproduct //////////

static const CvSize cross_product_sizes[] = {{3,1}, {-1,-1}};

class CxCore_CrossProductTest : public CxCore_MatrixTest
{
public:
    CxCore_CrossProductTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_CrossProductTest::CxCore_CrossProductTest() :
    CxCore_MatrixTest( "matrix-crossproduct", "cvCrossProduct", 2, 1, false, false, 1 )
{
    size_list = cross_product_sizes;
}


void CxCore_CrossProductTest::get_test_array_types_and_sizes( int /*test_case_idx*/, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % 2 + CV_32F;
    int cn = cvTsRandInt(rng) & 1 ? 3 : 1, type = CV_MAKETYPE(depth, cn);
    CvSize sz;

    types[INPUT][0] = types[INPUT][1] = types[OUTPUT][0] = types[REF_OUTPUT][0] = type;

    if( cn == 3 )
        sz = cvSize(1,1);
    else if( cvTsRandInt(rng) & 1 )
        sz = cvSize(3,1);
    else
        sz = cvSize(1,3);

    sizes[INPUT][0] = sizes[INPUT][1] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = sz;
}


void CxCore_CrossProductTest::run_func()
{
    cvCrossProduct( test_array[INPUT][0], test_array[INPUT][1], test_array[OUTPUT][0] );
}


void CxCore_CrossProductTest::prepare_to_validation( int )
{
    CvScalar a = {{0,0,0,0}}, b = {{0,0,0,0}}, c = {{0,0,0,0}};

    if( test_mat[INPUT][0].rows > 1 )
    {
        a.val[0] = cvGetReal2D( &test_mat[INPUT][0], 0, 0 );
        a.val[1] = cvGetReal2D( &test_mat[INPUT][0], 1, 0 );
        a.val[2] = cvGetReal2D( &test_mat[INPUT][0], 2, 0 );

        b.val[0] = cvGetReal2D( &test_mat[INPUT][1], 0, 0 );
        b.val[1] = cvGetReal2D( &test_mat[INPUT][1], 1, 0 );
        b.val[2] = cvGetReal2D( &test_mat[INPUT][1], 2, 0 );
    }
    else if( test_mat[INPUT][0].cols > 1 )
    {
        a.val[0] = cvGetReal1D( &test_mat[INPUT][0], 0 );
        a.val[1] = cvGetReal1D( &test_mat[INPUT][0], 1 );
        a.val[2] = cvGetReal1D( &test_mat[INPUT][0], 2 );

        b.val[0] = cvGetReal1D( &test_mat[INPUT][1], 0 );
        b.val[1] = cvGetReal1D( &test_mat[INPUT][1], 1 );
        b.val[2] = cvGetReal1D( &test_mat[INPUT][1], 2 );
    }
    else
    {
        a = cvGet1D( &test_mat[INPUT][0], 0 );
        b = cvGet1D( &test_mat[INPUT][1], 0 );
    }

    c.val[2] = a.val[0]*b.val[1] - a.val[1]*b.val[0];
    c.val[1] = -a.val[0]*b.val[2] + a.val[2]*b.val[0];
    c.val[0] = a.val[1]*b.val[2] - a.val[2]*b.val[1];

    if( test_mat[REF_OUTPUT][0].rows > 1 )
    {
        cvSetReal2D( &test_mat[REF_OUTPUT][0], 0, 0, c.val[0] );
        cvSetReal2D( &test_mat[REF_OUTPUT][0], 1, 0, c.val[1] );
        cvSetReal2D( &test_mat[REF_OUTPUT][0], 2, 0, c.val[2] );
    }
    else if( test_mat[REF_OUTPUT][0].cols > 1 )
    {
        cvSetReal1D( &test_mat[REF_OUTPUT][0], 0, c.val[0] );
        cvSetReal1D( &test_mat[REF_OUTPUT][0], 1, c.val[1] );
        cvSetReal1D( &test_mat[REF_OUTPUT][0], 2, c.val[2] );
    }
    else
    {
        cvSet1D( &test_mat[REF_OUTPUT][0], 0, c );
    }
}

CxCore_CrossProductTest crossproduct_test;


///////////////// scaleadd /////////////////////

class CxCore_ScaleAddTest : public CxCore_MatrixTest
{
public:
    CxCore_ScaleAddTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    CvScalar alpha;
    bool test_nd;
};

CxCore_ScaleAddTest::CxCore_ScaleAddTest() :
    CxCore_MatrixTest( "matrix-scaleadd", "cvScaleAdd", 3, 1, false, false, 4 )
{
    alpha = cvScalarAll(0);
    test_nd = false;
}


void CxCore_ScaleAddTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][2] = cvSize(1,1);
    types[INPUT][2] &= CV_MAT_DEPTH_MASK;
    test_nd = cvTsRandInt(ts->get_rng()) % 2 != 0;
}


void CxCore_ScaleAddTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                              whole_sizes, are_images );
    sizes[INPUT][2] = cvSize(1,1);
    types[INPUT][2] &= CV_MAT_DEPTH_MASK;
}


int CxCore_ScaleAddTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
        alpha = cvGet1D( &test_mat[INPUT][2], 0 );
    if( test_nd )
        alpha.val[1] = 0;
    return code;
}


void CxCore_ScaleAddTest::run_func()
{
    if(!test_nd)
        cvScaleAdd( test_array[INPUT][0], alpha, test_array[INPUT][1], test_array[OUTPUT][0] );
    else
    {
        cv::MatND c = cv::cvarrToMatND(test_array[OUTPUT][0]);
        cv::scaleAdd( cv::cvarrToMatND(test_array[INPUT][0]), alpha.val[0],
                      cv::cvarrToMatND(test_array[INPUT][1]), c);
    }
}


void CxCore_ScaleAddTest::prepare_to_validation( int )
{
    cvTsAdd( &test_mat[INPUT][0], cvScalarAll(alpha.val[0]),
             &test_mat[INPUT][1], cvScalarAll(1.),
             cvScalarAll(0.), &test_mat[REF_OUTPUT][0], 0 );
}

CxCore_ScaleAddTest scaleadd_test;


///////////////// gemm /////////////////////

static const char* matrix_gemm_param_names[] = { "size", "add_c", "mul_type", "depth", 0 };
static const char* matrix_gemm_mul_types[] = { "AB", "AtB", "ABt", "AtBt", 0 };
static const int matrix_gemm_add_c_flags[] = { 0, 1 };

class CxCore_GEMMTest : public CxCore_MatrixTest
{
public:
    CxCore_GEMMTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int write_default_params( CvFileStorage* fs );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int tabc_flag;
    double alpha, beta;
};

CxCore_GEMMTest::CxCore_GEMMTest() :
    CxCore_MatrixTest( "matrix-gemm", "cvGEMM", 5, 1, false, false, 2 )
{
    test_case_count = 100;
    max_log_array_size = 10;
    default_timing_param_names = matrix_gemm_param_names;
    alpha = beta = 0;
}


void CxCore_GEMMTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CvSize sizeA;
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizeA = sizes[INPUT][0];
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = sizeA;
    sizes[INPUT][2] = sizes[INPUT][3] = cvSize(1,1);
    types[INPUT][2] = types[INPUT][3] &= ~CV_MAT_CN_MASK;

    tabc_flag = cvTsRandInt(rng) & 7;

    switch( tabc_flag & (CV_GEMM_A_T|CV_GEMM_B_T) )
    {
    case 0:
        sizes[INPUT][1].height = sizes[INPUT][0].width;
        sizes[OUTPUT][0].height = sizes[INPUT][0].height;
        sizes[OUTPUT][0].width = sizes[INPUT][1].width;
        break;
    case CV_GEMM_B_T:
        sizes[INPUT][1].width = sizes[INPUT][0].width;
        sizes[OUTPUT][0].height = sizes[INPUT][0].height;
        sizes[OUTPUT][0].width = sizes[INPUT][1].height;
        break;
    case CV_GEMM_A_T:
        sizes[INPUT][1].height = sizes[INPUT][0].height;
        sizes[OUTPUT][0].height = sizes[INPUT][0].width;
        sizes[OUTPUT][0].width = sizes[INPUT][1].width;
        break;
    case CV_GEMM_A_T | CV_GEMM_B_T:
        sizes[INPUT][1].width = sizes[INPUT][0].height;
        sizes[OUTPUT][0].height = sizes[INPUT][0].width;
        sizes[OUTPUT][0].width = sizes[INPUT][1].height;
        break;
    }

    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];

    if( cvTsRandInt(rng) & 1 )
        sizes[INPUT][4] = cvSize(0,0);
    else if( !(tabc_flag & CV_GEMM_C_T) )
        sizes[INPUT][4] = sizes[OUTPUT][0];
    else
    {
        sizes[INPUT][4].width = sizes[OUTPUT][0].height;
        sizes[INPUT][4].height = sizes[OUTPUT][0].width;
    }
}


void CxCore_GEMMTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* mul_type = cvReadString( find_timing_param("mul_type"), "AB" );
    if( strcmp( mul_type, "AtB" ) == 0 )
        tabc_flag = CV_GEMM_A_T;
    else if( strcmp( mul_type, "ABt" ) == 0 )
        tabc_flag = CV_GEMM_B_T;
    else if( strcmp( mul_type, "AtBt" ) == 0 )
        tabc_flag = CV_GEMM_A_T + CV_GEMM_B_T;
    else
        tabc_flag = 0;

    if( cvReadInt( find_timing_param( "add_c" ), 0 ) == 0 )
        sizes[INPUT][4] = cvSize(0,0);
}


int CxCore_GEMMTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_MatrixTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "mul_type", matrix_gemm_mul_types );
    write_int_list( fs, "add_c", matrix_gemm_add_c_flags, CV_DIM(matrix_gemm_add_c_flags) );
    return code;
}


void CxCore_GEMMTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s%s,%s,",
        tabc_flag & CV_GEMM_A_T ? "At" : "A",
        tabc_flag & CV_GEMM_B_T ? "Bt" : "B",
        test_array[INPUT][4] ? "plusC" : "" );
    ptr += strlen(ptr);
    params_left -= 2;
    CxCore_MatrixTest::print_timing_params( test_case_idx, ptr, params_left );
}


int CxCore_GEMMTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        alpha = cvmGet( &test_mat[INPUT][2], 0, 0 );
        beta = cvmGet( &test_mat[INPUT][3], 0, 0 );
    }
    return code;
}


void CxCore_GEMMTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-10.);
    *high = cvScalarAll(10.);
}


void CxCore_GEMMTest::run_func()
{
    cvGEMM( test_array[INPUT][0], test_array[INPUT][1], alpha,
            test_array[INPUT][4], beta, test_array[OUTPUT][0], tabc_flag );
}


void CxCore_GEMMTest::prepare_to_validation( int )
{
    cvTsGEMM( &test_mat[INPUT][0], &test_mat[INPUT][1], alpha,
        test_array[INPUT][4] ? &test_mat[INPUT][4] : 0,
        beta, &test_mat[REF_OUTPUT][0], tabc_flag );
}

CxCore_GEMMTest gemm_test;


///////////////// multransposed /////////////////////

static const char* matrix_multrans_param_names[] = { "size", "use_delta", "mul_type", "depth", 0 };
static const int matrix_multrans_use_delta_flags[] = { 0, 1 };
static const char* matrix_multrans_mul_types[] = { "AAt", "AtA", 0 };

class CxCore_MulTransposedTest : public CxCore_MatrixTest
{
public:
    CxCore_MulTransposedTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int write_default_params( CvFileStorage* fs );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int order;
};


CxCore_MulTransposedTest::CxCore_MulTransposedTest() :
    CxCore_MatrixTest( "matrix-multransposed", "cvMulTransposed, cvRepeat", 2, 1, false, false, 1 )
{
    test_case_count = 100;
    order = 0;
    test_array[TEMP].push(NULL);
    default_timing_param_names = matrix_multrans_param_names;
}


void CxCore_MulTransposedTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    int src_type = cvTsRandInt(rng) % 5;
    int dst_type = cvTsRandInt(rng) % 2;

    src_type = src_type == 0 ? CV_8U : src_type == 1 ? CV_16U : src_type == 2 ? CV_16S :
               src_type == 3 ? CV_32F : CV_64F;
    dst_type = dst_type == 0 ? CV_32F : CV_64F;
    dst_type = MAX( dst_type, src_type );

    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( bits & 1 )
        sizes[INPUT][1] = cvSize(0,0);
    else
    {
        sizes[INPUT][1] = sizes[INPUT][0];
        if( bits & 2 )
            sizes[INPUT][1].height = 1;
        if( bits & 4 )
            sizes[INPUT][1].width = 1;
    }

    sizes[TEMP][0] = sizes[INPUT][0];
    types[INPUT][0] = src_type;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][1] = types[TEMP][0] = dst_type;

    order = (bits & 8) != 0;
    sizes[OUTPUT][0].width = sizes[OUTPUT][0].height = order == 0 ?
        sizes[INPUT][0].height : sizes[INPUT][0].width;
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


void CxCore_MulTransposedTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* mul_type = cvReadString( find_timing_param("mul_type"), "AAt" );
    order = strcmp( mul_type, "AtA" ) == 0;

    if( cvReadInt( find_timing_param( "use_delta" ), 0 ) == 0 )
        sizes[INPUT][1] = cvSize(0,0);
}


int CxCore_MulTransposedTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_MatrixTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "mul_type", matrix_multrans_mul_types );
    write_int_list( fs, "use_delta", matrix_multrans_use_delta_flags,
                    CV_DIM(matrix_multrans_use_delta_flags) );
    return code;
}


void CxCore_MulTransposedTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,%s,", order == 0 ? "AAt" : "AtA", test_array[INPUT][1] ? "delta" : "" );
    ptr += strlen(ptr);
    params_left -= 2;
    CxCore_MatrixTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CxCore_MulTransposedTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-10.);
    *high = cvScalarAll(10.);
}


void CxCore_MulTransposedTest::run_func()
{
    cvMulTransposed( test_array[INPUT][0], test_array[OUTPUT][0],
                     order, test_array[INPUT][1] );
}


void CxCore_MulTransposedTest::prepare_to_validation( int )
{
    CvMat* delta = test_array[INPUT][1] ? &test_mat[INPUT][1] : 0;
    if( delta )
    {
        if( test_mat[INPUT][1].rows < test_mat[INPUT][0].rows ||
            test_mat[INPUT][1].cols < test_mat[INPUT][0].cols )
        {
            cvRepeat( delta, &test_mat[TEMP][0] );
            delta = &test_mat[TEMP][0];
        }
        cvTsAdd( &test_mat[INPUT][0], cvScalarAll(1.), delta, cvScalarAll(-1.),
                 cvScalarAll(0.), &test_mat[TEMP][0], 0 );
    }
    else
        cvTsConvert( &test_mat[INPUT][0], &test_mat[TEMP][0] );
    delta = &test_mat[TEMP][0];

    cvTsGEMM( delta, delta, 1., 0, 0, &test_mat[REF_OUTPUT][0], order == 0 ? CV_GEMM_B_T : CV_GEMM_A_T );
}

CxCore_MulTransposedTest multransposed_test;


///////////////// Transform /////////////////////

static const CvSize matrix_transform_sizes[] = {{10,10}, {100,100}, {720,480}, {-1,-1}};
static const CvSize matrix_transform_whole_sizes[] = {{10,10}, {720,480}, {720,480}, {-1,-1}};
static const int matrix_transform_channels[] = { 2, 3, 4, -1 };
static const char* matrix_transform_param_names[] = { "size", "channels", "depth", 0 };

class CxCore_TransformTest : public CxCore_MatrixTest
{
public:
    CxCore_TransformTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int prepare_test_case( int test_case_idx );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void run_func();
    void prepare_to_validation( int test_case_idx );

    double scale;
    bool diagMtx;
};


CxCore_TransformTest::CxCore_TransformTest() :
    CxCore_MatrixTest( "matrix-transform", "cvTransform", 3, 1, true, false, 4 )
{
    default_timing_param_names = matrix_transform_param_names;
    cn_list = matrix_transform_channels;
    depth_list = matrix_all_depths;
    size_list = matrix_transform_sizes;
    whole_size_list = matrix_transform_whole_sizes;
}


void CxCore_TransformTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    int depth, dst_cn, mat_cols, mattype;
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    mat_cols = CV_MAT_CN(types[INPUT][0]);
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    dst_cn = cvTsRandInt(rng) % 4 + 1;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, dst_cn);

    mattype = depth < CV_32S ? CV_32F : depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;
    types[INPUT][2] = CV_MAKETYPE(mattype, dst_cn);

    scale = 1./((cvTsRandInt(rng)%4)*50+1);

    if( bits & 2 )
    {
        sizes[INPUT][2] = cvSize(0,0);
        mat_cols += (bits & 4) != 0;
    }
    else if( bits & 4 )
        sizes[INPUT][2] = cvSize(1,1);
    else
    {
        if( bits & 8 )
            sizes[INPUT][2] = cvSize(dst_cn,1);
        else
            sizes[INPUT][2] = cvSize(1,dst_cn);
        types[INPUT][2] &= ~CV_MAT_CN_MASK;
    }
    diagMtx = (bits & 16) != 0;

    sizes[INPUT][1] = cvSize(mat_cols,dst_cn);
}


void CxCore_TransformTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    int cn = CV_MAT_CN(types[INPUT][0]);
    sizes[INPUT][1] = cvSize(cn + (cn < 4), cn);
    sizes[INPUT][2] = cvSize(0,0);
    types[INPUT][1] = types[INPUT][2] = CV_64FC1;
    scale = 1./1000;
}

int CxCore_TransformTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        cvTsAdd(&test_mat[INPUT][1], cvScalarAll(scale), &test_mat[INPUT][1],
                cvScalarAll(0), cvScalarAll(0), &test_mat[INPUT][1], 0 );
        if(diagMtx)
        {
            CvMat* w = cvCloneMat(&test_mat[INPUT][1]);
            cvSetIdentity(w, cvScalarAll(1));
            cvMul(w, &test_mat[INPUT][1], &test_mat[INPUT][1]);
            cvReleaseMat(&w);
        }
    }
    return code;
}

void CxCore_TransformTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    CvSize size = cvGetMatSize(&test_mat[INPUT][1]);
    sprintf( ptr, "matrix=%dx%d,", size.height, size.width );
    ptr += strlen(ptr);
    params_left--;
    CxCore_MatrixTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CxCore_TransformTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth <= CV_8S ? 1 : depth <= CV_32S ? 9 :
        CxCore_MatrixTest::get_success_error_level( test_case_idx, i, j );
}

void CxCore_TransformTest::run_func()
{
    cvTransform( test_array[INPUT][0], test_array[OUTPUT][0], &test_mat[INPUT][1],
                 test_array[INPUT][2] ? &test_mat[INPUT][2] : 0);
}


void CxCore_TransformTest::prepare_to_validation( int )
{
    CvMat* transmat = &test_mat[INPUT][1];
    CvMat* shift = test_array[INPUT][2] ? &test_mat[INPUT][2] : 0;

    cvTsTransform( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0], transmat, shift );
}

CxCore_TransformTest transform_test;


///////////////// PerspectiveTransform /////////////////////

static const int matrix_perspective_transform_channels[] = { 2, 3, -1 };

class CxCore_PerspectiveTransformTest : public CxCore_MatrixTest
{
public:
    CxCore_PerspectiveTransformTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_PerspectiveTransformTest::CxCore_PerspectiveTransformTest() :
    CxCore_MatrixTest( "matrix-perspective", "cvPerspectiveTransform", 2, 1, false, false, 2 )
{
    default_timing_param_names = matrix_transform_param_names;
    cn_list = matrix_perspective_transform_channels;
    size_list = matrix_transform_sizes;
    whole_size_list = matrix_transform_whole_sizes;
}


void CxCore_PerspectiveTransformTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    int depth, cn, mattype;
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    cn = CV_MAT_CN(types[INPUT][0]) + 1;
    depth = CV_MAT_DEPTH(types[INPUT][0]);
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, cn);

    mattype = depth == CV_64F ? CV_64F : bits & 1 ? CV_32F : CV_64F;
    types[INPUT][1] = mattype;
    sizes[INPUT][1] = cvSize(cn + 1, cn + 1);
}


double CxCore_PerspectiveTransformTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    return depth == CV_32F ? 1e-4 : depth == CV_64F ? 1e-8 :
		CxCore_MatrixTest::get_success_error_level(test_case_idx, i, j);
}


void CxCore_PerspectiveTransformTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                        CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    int cn = CV_MAT_CN(types[INPUT][0]);
    sizes[INPUT][1] = cvSize(cn + 1, cn + 1);
    types[INPUT][1] = CV_64FC1;
}


void CxCore_PerspectiveTransformTest::run_func()
{
    cvPerspectiveTransform( test_array[INPUT][0], test_array[OUTPUT][0], &test_mat[INPUT][1] );
}


static void cvTsPerspectiveTransform( const CvArr* _src, CvArr* _dst, const CvMat* transmat )
{
    int i, j, cols;
    int cn, depth, mat_depth;
    CvMat astub, bstub, *a, *b;
    double mat[16], *buf;

    a = cvGetMat( _src, &astub, 0, 0 );
    b = cvGetMat( _dst, &bstub, 0, 0 );

    cn = CV_MAT_CN(a->type);
    depth = CV_MAT_DEPTH(a->type);
    mat_depth = CV_MAT_DEPTH(transmat->type);
    cols = transmat->cols;

    // prepare cn x (cn + 1) transform matrix
    if( mat_depth == CV_32F )
    {
        for( i = 0; i < transmat->rows; i++ )
            for( j = 0; j < cols; j++ )
                mat[i*cols + j] = ((float*)(transmat->data.ptr + transmat->step*i))[j];
    }
    else
    {
        assert( mat_depth == CV_64F );
        for( i = 0; i < transmat->rows; i++ )
            for( j = 0; j < cols; j++ )
                mat[i*cols + j] = ((double*)(transmat->data.ptr + transmat->step*i))[j];
    }

    // transform data
    cols = a->cols * cn;
    buf = (double*)cvStackAlloc( cols * sizeof(double) );

    for( i = 0; i < a->rows; i++ )
    {
        uchar* src = a->data.ptr + i*a->step;
        uchar* dst = b->data.ptr + i*b->step;

        switch( depth )
        {
        case CV_32F:
            for( j = 0; j < cols; j++ )
                buf[j] = ((float*)src)[j];
            break;
        case CV_64F:
            for( j = 0; j < cols; j++ )
                buf[j] = ((double*)src)[j];
            break;
        default:
            assert(0);
        }

        switch( cn )
        {
        case 2:
            for( j = 0; j < cols; j += 2 )
            {
                double t0 = buf[j]*mat[0] + buf[j+1]*mat[1] + mat[2];
                double t1 = buf[j]*mat[3] + buf[j+1]*mat[4] + mat[5];
                double w = buf[j]*mat[6] + buf[j+1]*mat[7] + mat[8];
                w = w ? 1./w : 0;
                buf[j] = t0*w;
                buf[j+1] = t1*w;
            }
            break;
        case 3:
            for( j = 0; j < cols; j += 3 )
            {
                double t0 = buf[j]*mat[0] + buf[j+1]*mat[1] + buf[j+2]*mat[2] + mat[3];
                double t1 = buf[j]*mat[4] + buf[j+1]*mat[5] + buf[j+2]*mat[6] + mat[7];
                double t2 = buf[j]*mat[8] + buf[j+1]*mat[9] + buf[j+2]*mat[10] + mat[11];
                double w = buf[j]*mat[12] + buf[j+1]*mat[13] + buf[j+2]*mat[14] + mat[15];
                w = w ? 1./w : 0;
                buf[j] = t0*w;
                buf[j+1] = t1*w;
                buf[j+2] = t2*w;
            }
            break;
        default:
            assert(0);
        }

        switch( depth )
        {
        case CV_32F:
            for( j = 0; j < cols; j++ )
                ((float*)dst)[j] = (float)buf[j];
            break;
        case CV_64F:
            for( j = 0; j < cols; j++ )
                ((double*)dst)[j] = buf[j];
            break;
        default:
            assert(0);
        }
    }
}


void CxCore_PerspectiveTransformTest::prepare_to_validation( int )
{
    CvMat* transmat = &test_mat[INPUT][1];
    cvTsPerspectiveTransform( test_array[INPUT][0], test_array[REF_OUTPUT][0], transmat );
}

CxCore_PerspectiveTransformTest perspective_test;


///////////////// Mahalanobis /////////////////////

class CxCore_MahalanobisTest : public CxCore_MatrixTest
{
public:
    CxCore_MahalanobisTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_MahalanobisTest::CxCore_MahalanobisTest() :
    CxCore_MatrixTest( "matrix-mahalanobis", "cvMahalanobis", 3, 1, false, true, 1 )
{
    test_case_count = 100;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
}


void CxCore_MahalanobisTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( cvTsRandInt(rng) & 1 )
        sizes[INPUT][0].width = sizes[INPUT][1].width = 1;
    else
        sizes[INPUT][0].height = sizes[INPUT][1].height = 1;

    sizes[TEMP][0] = sizes[TEMP][1] = sizes[INPUT][0];
    sizes[INPUT][2].width = sizes[INPUT][2].height = sizes[INPUT][0].width + sizes[INPUT][0].height - 1;
    sizes[TEMP][2] = sizes[INPUT][2];
    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[INPUT][0];
}


void CxCore_MahalanobisTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                        CvSize** sizes, int** types, CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    sizes[INPUT][0].height = sizes[INPUT][1].height = 1;
}


int CxCore_MahalanobisTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 && ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        // make sure that the inverted "covariation" matrix is symmetrix and positively defined.
        cvTsGEMM( &test_mat[INPUT][2], &test_mat[INPUT][2], 1., 0, 0., &test_mat[TEMP][2], CV_GEMM_B_T );
        cvTsCopy( &test_mat[TEMP][2], &test_mat[INPUT][2] );
    }

    return code;
}


void CxCore_MahalanobisTest::run_func()
{
    *((CvScalar*)(test_mat[OUTPUT][0].data.db)) =
        cvRealScalar(cvMahalanobis(test_array[INPUT][0], test_array[INPUT][1], test_array[INPUT][2]));
}

void CxCore_MahalanobisTest::prepare_to_validation( int )
{
    cvTsAdd( &test_mat[INPUT][0], cvScalarAll(1.),
             &test_mat[INPUT][1], cvScalarAll(-1.),
             cvScalarAll(0.), &test_mat[TEMP][0], 0 );
    if( test_mat[INPUT][0].rows == 1 )
        cvTsGEMM( &test_mat[TEMP][0], &test_mat[INPUT][2], 1.,
                  0, 0., &test_mat[TEMP][1], 0 );
    else
        cvTsGEMM( &test_mat[INPUT][2], &test_mat[TEMP][0], 1.,
                  0, 0., &test_mat[TEMP][1], 0 );

    *((CvScalar*)(test_mat[REF_OUTPUT][0].data.db)) =
        cvRealScalar(sqrt(cvTsCrossCorr(&test_mat[TEMP][0], &test_mat[TEMP][1])));
}

CxCore_MahalanobisTest mahalanobis_test;


///////////////// covarmatrix /////////////////////

class CxCore_CovarMatrixTest : public CxCore_MatrixTest
{
public:
    CxCore_CovarMatrixTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    CvTestPtrVec temp_hdrs;
    uchar* hdr_data;
    int flags, t_flag, len, count;
    bool are_images;
};


CxCore_CovarMatrixTest::CxCore_CovarMatrixTest() :
    CxCore_MatrixTest( "matrix-covar", "cvCalcCovarMatrix", 1, 1, true, false, 1 ),
        flags(0), t_flag(0), are_images(false)
{
    test_case_count = 100;
    test_array[INPUT_OUTPUT].push(NULL);
    test_array[REF_INPUT_OUTPUT].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}


void CxCore_CovarMatrixTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    int i, single_matrix;
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    flags = bits & (CV_COVAR_NORMAL | CV_COVAR_USE_AVG | CV_COVAR_SCALE | CV_COVAR_ROWS );
    single_matrix = flags & CV_COVAR_ROWS;
    t_flag = (bits & 256) != 0;

    const int min_count = 2;

    if( !t_flag )
    {
        len = sizes[INPUT][0].width;
        count = sizes[INPUT][0].height;
        count = MAX(count, min_count);
        sizes[INPUT][0] = cvSize(len, count);
    }
    else
    {
        len = sizes[INPUT][0].height;
        count = sizes[INPUT][0].width;
        count = MAX(count, min_count);
        sizes[INPUT][0] = cvSize(count, len);
    }

    if( single_matrix && t_flag )
        flags = (flags & ~CV_COVAR_ROWS) | CV_COVAR_COLS;

    if( CV_MAT_DEPTH(types[INPUT][0]) == CV_32S )
        types[INPUT][0] = (types[INPUT][0] & ~CV_MAT_DEPTH_MASK) | CV_32F;

    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = flags & CV_COVAR_NORMAL ? cvSize(len,len) : cvSize(count,count);
    sizes[INPUT_OUTPUT][0] = sizes[REF_INPUT_OUTPUT][0] = !t_flag ? cvSize(len,1) : cvSize(1,len);
    sizes[TEMP][0] = sizes[INPUT][0];

    types[INPUT_OUTPUT][0] = types[REF_INPUT_OUTPUT][0] =
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[TEMP][0] =
        CV_MAT_DEPTH(types[INPUT][0]) == CV_64F || (bits & 512) ? CV_64F : CV_32F;

    are_images = (bits & 1024) != 0;
    for( i = 0; i < (single_matrix ? 1 : count); i++ )
        temp_hdrs.push(NULL);
}


int CxCore_CovarMatrixTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        int i;
        int single_matrix = flags & (CV_COVAR_ROWS|CV_COVAR_COLS);
        int hdr_size = are_images ? sizeof(IplImage) : sizeof(CvMat);

        hdr_data = (uchar*)cvAlloc( count*hdr_size );
        if( single_matrix )
        {
            if( !are_images )
                *((CvMat*)hdr_data) = test_mat[INPUT][0];
            else
                cvGetImage( &test_mat[INPUT][0], (IplImage*)hdr_data );
            temp_hdrs[0] = hdr_data;
        }
        else
            for( i = 0; i < count; i++ )
            {
                CvMat part;
                void* ptr = hdr_data + i*hdr_size;

                if( !t_flag )
                    cvGetRow( &test_mat[INPUT][0], &part, i );
                else
                    cvGetCol( &test_mat[INPUT][0], &part, i );

                if( !are_images )
                    *((CvMat*)ptr) = part;
                else
                    cvGetImage( &part, (IplImage*)ptr );

                temp_hdrs[i] = ptr;
            }
    }

    return code;
}


void CxCore_CovarMatrixTest::run_func()
{
    cvCalcCovarMatrix( (const void**)&temp_hdrs[0], count,
                       test_array[OUTPUT][0], test_array[INPUT_OUTPUT][0], flags );
}


void CxCore_CovarMatrixTest::prepare_to_validation( int )
{
    CvMat* avg = &test_mat[REF_INPUT_OUTPUT][0];
    double scale = 1.;

    if( !(flags & CV_COVAR_USE_AVG) )
    {
        int i;
        cvTsZero( avg );

        for( i = 0; i < count; i++ )
        {
            CvMat stub, *vec = 0;
            if( flags & CV_COVAR_ROWS )
                vec = cvGetRow( temp_hdrs[0], &stub, i );
            else if( flags & CV_COVAR_COLS )
                vec = cvGetCol( temp_hdrs[0], &stub, i );
            else
                vec = cvGetMat( temp_hdrs[i], &stub );

            cvTsAdd( avg, cvScalarAll(1.), vec,
                     cvScalarAll(1.), cvScalarAll(0.), avg, 0 );
        }

        cvTsAdd( avg, cvScalarAll(1./count), 0,
                 cvScalarAll(0.), cvScalarAll(0.), avg, 0 );
    }

    if( flags & CV_COVAR_SCALE )
    {
        scale = 1./count;
    }

    cvRepeat( avg, &test_mat[TEMP][0] );
    cvTsAdd( &test_mat[INPUT][0], cvScalarAll(1.),
             &test_mat[TEMP][0], cvScalarAll(-1.),
             cvScalarAll(0.), &test_mat[TEMP][0], 0 );

    cvTsGEMM( &test_mat[TEMP][0], &test_mat[TEMP][0],
              scale, 0, 0., &test_mat[REF_OUTPUT][0],
              t_flag ^ ((flags & CV_COVAR_NORMAL) != 0) ?
              CV_GEMM_A_T : CV_GEMM_B_T );

    cvFree( &hdr_data );
    temp_hdrs.clear();
}

CxCore_CovarMatrixTest covarmatrix_test;


static void cvTsFloodWithZeros( CvMat* mat, CvRNG* rng )
{
    int k, total = mat->rows*mat->cols;
    int zero_total = cvTsRandInt(rng) % total;
    assert( CV_MAT_TYPE(mat->type) == CV_32FC1 ||
            CV_MAT_TYPE(mat->type) == CV_64FC1 );

    for( k = 0; k < zero_total; k++ )
    {
        int i = cvTsRandInt(rng) % mat->rows;
        int j = cvTsRandInt(rng) % mat->cols;
        uchar* row = mat->data.ptr + mat->step*i;

        if( CV_MAT_DEPTH(mat->type) == CV_32FC1 )
            ((float*)row)[j] = 0.f;
        else
            ((double*)row)[j] = 0.;
    }
}


///////////////// determinant /////////////////////

class CxCore_DetTest : public CxCore_MatrixTest
{
public:
    CxCore_DetTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CxCore_DetTest::CxCore_DetTest() :
    CxCore_MatrixTest( "matrix-det", "cvDet", 1, 1, false, true, 1 )
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push(NULL);
}


void CxCore_DetTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    sizes[INPUT][0].width = sizes[INPUT][0].height = sizes[INPUT][0].height;
    sizes[TEMP][0] = sizes[INPUT][0];
    types[TEMP][0] = CV_64FC1;
}


void CxCore_DetTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-2.);
    *high = cvScalarAll(2.);
}


double CxCore_DetTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0])) == CV_32F ? 1e-2 : 1e-5;
}


int CxCore_DetTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
        cvTsFloodWithZeros( &test_mat[INPUT][0], ts->get_rng() );

    return code;
}


void CxCore_DetTest::run_func()
{
    *((CvScalar*)(test_mat[OUTPUT][0].data.db)) = cvRealScalar(cvDet(test_array[INPUT][0]));
}


// LU method that chooses the optimal in a column pivot element
static double cvTsLU( CvMat* a, CvMat* b=NULL, CvMat* x=NULL, int* rank=0 )
{
    int i, j, k, N = a->rows, N1 = a->cols, Nm = MIN(N, N1), step = a->step/sizeof(double);
    int M = b ? b->cols : 0, b_step = b ? b->step/sizeof(double) : 0;
    int x_step = x ? x->step/sizeof(double) : 0;
    double *a0 = a->data.db, *b0 = b ? b->data.db : 0;
    double *x0 = x ? x->data.db : 0;
    double t, det = 1.;
    assert( CV_MAT_TYPE(a->type) == CV_64FC1 &&
            (!b || CV_ARE_TYPES_EQ(a,b)) && (!x || CV_ARE_TYPES_EQ(a,x)));

    for( i = 0; i < Nm; i++ )
    {
        double max_val = fabs(a0[i*step + i]);
        double *a1, *a2, *b1 = 0, *b2 = 0;
        k = i;

        for( j = i+1; j < N; j++ )
        {
            t = fabs(a0[j*step + i]);
            if( max_val < t )
            {
                max_val = t;
                k = j;
            }
        }

        if( k != i )
        {
            for( j = i; j < N1; j++ )
                CV_SWAP( a0[i*step + j], a0[k*step + j], t );

            for( j = 0; j < M; j++ )
                CV_SWAP( b0[i*b_step + j], b0[k*b_step + j], t );
            det = -det;
        }

        if( max_val == 0 )
        {
            if( rank )
                *rank = i;
            return 0.;
        }

        a1 = a0 + i*step;
        a2 = a1 + step;
        b1 = b0 + i*b_step;
        b2 = b1 + b_step;

        for( j = i+1; j < N; j++, a2 += step, b2 += b_step )
        {
            t = a2[i]/a1[i];
            for( k = i+1; k < N1; k++ )
                a2[k] -= t*a1[k];

            for( k = 0; k < M; k++ )
                b2[k] -= t*b1[k];
        }

        det *= a1[i];
    }

    if( x )
    {
        assert( b );

        for( i = N-1; i >= 0; i-- )
        {
            double* a1 = a0 + i*step;
            double* b1 = b0 + i*b_step;
            for( j = 0; j < M; j++ )
            {
                t = b1[j];
                for( k = i+1; k < N1; k++ )
                    t -= a1[k]*x0[k*x_step + j];
                x0[i*x_step + j] = t/a1[i];
            }
        }
    }

    if( rank )
        *rank = i;
    return det;
}


void CxCore_DetTest::prepare_to_validation( int )
{
    if( !CV_ARE_TYPES_EQ( &test_mat[INPUT][0], &test_mat[TEMP][0] ))
        cvTsConvert( &test_mat[INPUT][0], &test_mat[TEMP][0] );
    else
        cvTsCopy( &test_mat[INPUT][0], &test_mat[TEMP][0], 0 );

    *((CvScalar*)(test_mat[REF_OUTPUT][0].data.db)) = cvRealScalar(cvTsLU(&test_mat[TEMP][0], 0, 0));
}

CxCore_DetTest det_test;



///////////////// invert /////////////////////

static const char* matrix_solve_invert_param_names[] = { "size", "method", "depth", 0 };
static const char* matrix_solve_invert_methods[] = { "LU", "SVD", 0 };

class CxCore_InvertTest : public CxCore_MatrixTest
{
public:
    CxCore_InvertTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int write_default_params( CvFileStorage* fs );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int method, rank;
    double result;
};


CxCore_InvertTest::CxCore_InvertTest() :
    CxCore_MatrixTest( "matrix-invert", "cvInvert, cvSVD, cvSVBkSb", 1, 1, false, false, 1 ), method(0), rank(0), result(0.)
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);

    default_timing_param_names = matrix_solve_invert_param_names;
}


void CxCore_InvertTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );

    if( (bits & 3) == 0 )
    {
        method = CV_SVD;
        if( bits & 4 )
        {
            sizes[INPUT][0] = cvSize(min_size, min_size);
            if( bits & 16 )
                method = CV_CHOLESKY;
        }
    }
    else
    {
        method = CV_LU;
        sizes[INPUT][0] = cvSize(min_size, min_size);
    }

    sizes[TEMP][0].width = sizes[INPUT][0].height;
    sizes[TEMP][0].height = sizes[INPUT][0].width;
    sizes[TEMP][1] = sizes[INPUT][0];
    types[TEMP][0] = types[INPUT][0];
    types[TEMP][1] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(min_size, min_size);
}


void CxCore_InvertTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* method_str = cvReadString( find_timing_param("method"), "LU" );
    method = strcmp( method_str, "LU" ) == 0 ? CV_LU : CV_SVD;
}


int CxCore_InvertTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_MatrixTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "method", matrix_solve_invert_methods );
    return code;
}


void CxCore_InvertTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", method == CV_LU ? "LU" : "SVD" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_MatrixTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CxCore_InvertTest::get_success_error_level( int /*test_case_idx*/, int, int )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[OUTPUT][0])) == CV_32F ? 1e-2 : 1e-6;
}

int CxCore_InvertTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        cvTsFloodWithZeros( &test_mat[INPUT][0], ts->get_rng() );

        if( method == CV_CHOLESKY )
        {
            cvTsGEMM( &test_mat[INPUT][0], &test_mat[INPUT][0], 1.,
                      0, 0., &test_mat[TEMP][0], CV_GEMM_B_T );
            cvTsCopy( &test_mat[TEMP][0], &test_mat[INPUT][0] );
        }
    }

    return code;
}



void CxCore_InvertTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-1.);
    *high = cvScalarAll(1.);
}


void CxCore_InvertTest::run_func()
{
    result = cvInvert(test_array[INPUT][0], test_array[TEMP][0], method);
}


static double cvTsSVDet( CvMat* mat, double* ratio )
{
    int type = CV_MAT_TYPE(mat->type);
    int i, nm = MIN( mat->rows, mat->cols );
    CvMat* w = cvCreateMat( nm, 1, type );
    double det = 1.;

    cvSVD( mat, w, 0, 0, 0 );

    if( type == CV_32FC1 )
    {
        for( i = 0; i < nm; i++ )
            det *= w->data.fl[i];
        *ratio = w->data.fl[nm-1] < FLT_EPSILON ? FLT_MAX : w->data.fl[nm-1]/w->data.fl[0];
    }
    else
    {
        for( i = 0; i < nm; i++ )
            det *= w->data.db[i];
        *ratio = w->data.db[nm-1] < FLT_EPSILON ? DBL_MAX : w->data.db[nm-1]/w->data.db[0];
    }

    cvReleaseMat( &w );
    return det;
}

void CxCore_InvertTest::prepare_to_validation( int )
{
    CvMat* input = &test_mat[INPUT][0];
    double ratio = 0, det = cvTsSVDet( input, &ratio );
    double threshold = (CV_MAT_DEPTH(input->type) == CV_32F ? FLT_EPSILON : DBL_EPSILON)*1000;

    if( CV_MAT_TYPE(input->type) == CV_32FC1 )
        cvTsConvert( input, &test_mat[TEMP][1] );
    else
        cvTsCopy( input, &test_mat[TEMP][1], 0 );

    if( det < threshold ||
        ((method == CV_LU || method == CV_CHOLESKY) && (result == 0 || ratio < threshold)) ||
        ((method == CV_SVD || method == CV_SVD_SYM) && result < threshold) )
    {
        cvTsZero( &test_mat[OUTPUT][0] );
        cvTsZero( &test_mat[REF_OUTPUT][0] );
        //cvTsAdd( 0, cvScalarAll(0.), 0, cvScalarAll(0.), cvScalarAll(fabs(det)>1e-3),
        //         &test_mat[REF_OUTPUT][0], 0 );
        return;
    }

    if( input->rows >= input->cols )
        cvTsGEMM( &test_mat[TEMP][0], input, 1., 0, 0., &test_mat[OUTPUT][0], 0 );
    else
        cvTsGEMM( input, &test_mat[TEMP][0], 1., 0, 0., &test_mat[OUTPUT][0], 0 );

    cvTsSetIdentity( &test_mat[REF_OUTPUT][0], cvScalarAll(1.) );
}

CxCore_InvertTest invert_test;


///////////////// solve /////////////////////

class CxCore_SolveTest : public CxCore_MatrixTest
{
public:
    CxCore_SolveTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    int write_default_params( CvFileStorage* fs );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int method, rank;
    double result;
};


CxCore_SolveTest::CxCore_SolveTest() :
    CxCore_MatrixTest( "matrix-solve", "cvSolve, cvSVD, cvSVBkSb", 2, 1, false, false, 1 ), method(0), rank(0), result(0.)
{
    test_case_count = 100;
    max_log_array_size = 7;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);

    default_timing_param_names = matrix_solve_invert_param_names;
}


void CxCore_SolveTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    CvSize in_sz = sizes[INPUT][0];
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    sizes[INPUT][0] = in_sz;
    int min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );

    if( (bits & 3) == 0 )
    {
        method = CV_SVD;
        if( bits & 4 )
        {
            sizes[INPUT][0] = cvSize(min_size, min_size);
            /*if( bits & 8 )
                method = CV_SVD_SYM;*/
        }
    }
    else
    {
        method = CV_LU;
        sizes[INPUT][0] = cvSize(min_size, min_size);
    }

    sizes[INPUT][1].height = sizes[INPUT][0].height;
    sizes[TEMP][0].width = sizes[INPUT][1].width;
    sizes[TEMP][0].height = sizes[INPUT][0].width;
    sizes[TEMP][1] = sizes[INPUT][0];
    types[TEMP][0] = types[INPUT][0];
    types[TEMP][1] = CV_64FC1;
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(sizes[INPUT][1].width, min_size);
}

void CxCore_SolveTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* method_str = cvReadString( find_timing_param("method"), "LU" );
    sizes[INPUT][1].width = sizes[TEMP][0].width = sizes[OUTPUT][0].width = sizes[REF_OUTPUT][0].width = 1;
    method = strcmp( method_str, "LU" ) == 0 ? CV_LU : CV_SVD;
}


int CxCore_SolveTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_MatrixTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "method", matrix_solve_invert_methods );
    return code;
}


void CxCore_SolveTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", method == CV_LU ? "LU" : "SVD" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_MatrixTest::print_timing_params( test_case_idx, ptr, params_left );
}


int CxCore_SolveTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );

    /*if( method == CV_SVD_SYM )
    {
        cvTsGEMM( test_array[INPUT][0], test_array[INPUT][0], 1.,
                  0, 0., test_array[TEMP][0], CV_GEMM_B_T );
        cvTsCopy( test_array[TEMP][0], test_array[INPUT][0] );
    }*/

    return code;
}


void CxCore_SolveTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-1.);
    *high = cvScalarAll(1.);
}


double CxCore_SolveTest::get_success_error_level( int /*test_case_idx*/, int, int )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[OUTPUT][0])) == CV_32F ? 5e-2 : 1e-8;
}


void CxCore_SolveTest::run_func()
{
    result = cvSolve(test_array[INPUT][0], test_array[INPUT][1], test_array[TEMP][0], method);
}

void CxCore_SolveTest::prepare_to_validation( int )
{
    //int rank = test_mat[REF_OUTPUT][0].rows;
    CvMat* dst;
    CvMat* input = &test_mat[INPUT][0];

    if( method == CV_LU )
    {
        if( result == 0 )
        {
            if( CV_MAT_TYPE(input->type) == CV_32FC1 )
                cvTsConvert( input, &test_mat[TEMP][1] );
            else
                cvTsCopy( input, &test_mat[TEMP][1], 0 );

            cvTsZero( &test_mat[OUTPUT][0] );
            double det = cvTsLU( &test_mat[TEMP][1], 0, 0 );
            cvTsAdd( 0, cvScalarAll(0.), 0, cvScalarAll(0.), cvScalarAll(det != 0),
                     &test_mat[REF_OUTPUT][0], 0 );
            return;
        }
     
        double threshold = (CV_MAT_DEPTH(input->type) == CV_32F ? FLT_EPSILON : DBL_EPSILON)*1000;
        double ratio = 0, det = cvTsSVDet( input, &ratio );
        if( det < threshold || ratio < threshold )
        {
            cvTsZero( &test_mat[OUTPUT][0] );
            cvTsZero( &test_mat[REF_OUTPUT][0] );
            return;
        }
    }
        

    dst = input->rows <= input->cols ? &test_mat[OUTPUT][0] : &test_mat[INPUT][1];

    cvTsGEMM( input, &test_mat[TEMP][0], 1., &test_mat[INPUT][1], -1., dst, 0 );
    if( dst != &test_mat[OUTPUT][0] )
        cvTsGEMM( input, dst, 1., 0, 0., &test_mat[OUTPUT][0], CV_GEMM_A_T );
    cvTsZero( &test_mat[REF_OUTPUT][0] );
}

CxCore_SolveTest solve_test;


///////////////// SVD /////////////////////

static const char* matrix_svd_param_names[] = { "size", "output", "depth", 0 };
static const char* matrix_svd_output_modes[] = { "w", "all", 0 };

class CxCore_SVDTest : public CxCore_MatrixTest
{
public:
    CxCore_SVDTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    double get_success_error_level( int test_case_idx, int i, int j );
    int write_default_params( CvFileStorage* fs );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int flags;
    bool have_u, have_v, symmetric, compact, vector_w;
};


CxCore_SVDTest::CxCore_SVDTest() :
    CxCore_MatrixTest( "matrix-svd", "cvSVD", 1, 4, false, false, 1 ),
        flags(0), have_u(false), have_v(false), symmetric(false), compact(false), vector_w(false)
{
    test_case_count = 100;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);

    default_timing_param_names = matrix_svd_param_names;
}


void CxCore_SVDTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size, i, m, n;

    min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );

    flags = bits & (CV_SVD_MODIFY_A+CV_SVD_U_T+CV_SVD_V_T);
    have_u = (bits & 8) != 0;
    have_v = (bits & 16) != 0;
    symmetric = (bits & 32) != 0;
    compact = (bits & 64) != 0;
    vector_w = (bits & 128) != 0;

    if( symmetric )
        sizes[INPUT][0] = cvSize(min_size, min_size);

    m = sizes[INPUT][0].height;
    n = sizes[INPUT][0].width;

    if( compact )
        sizes[TEMP][0] = cvSize(min_size, min_size);
    else
        sizes[TEMP][0] = sizes[INPUT][0];
    sizes[TEMP][3] = cvSize(0,0);

    if( vector_w )
    {
        sizes[TEMP][3] = sizes[TEMP][0];
        if( bits & 256 )
            sizes[TEMP][0] = cvSize(1, min_size);
        else
            sizes[TEMP][0] = cvSize(min_size, 1);
    }

    if( have_u )
    {
        sizes[TEMP][1] = compact ? cvSize(min_size, m) : cvSize(m, m);

        if( flags & CV_SVD_U_T )
            CV_SWAP( sizes[TEMP][1].width, sizes[TEMP][1].height, i );
    }
    else
        sizes[TEMP][1] = cvSize(0,0);

    if( have_v )
    {
        sizes[TEMP][2] = compact ? cvSize(n, min_size) : cvSize(n, n);

        if( !(flags & CV_SVD_V_T) )
            CV_SWAP( sizes[TEMP][2].width, sizes[TEMP][2].height, i );
    }
    else
        sizes[TEMP][2] = cvSize(0,0);

    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[TEMP][3] = types[INPUT][0];
    types[OUTPUT][0] = types[OUTPUT][1] = types[OUTPUT][2] = types[INPUT][0];
    types[OUTPUT][3] = CV_8UC1;
    sizes[OUTPUT][0] = !have_u || !have_v ? cvSize(0,0) : sizes[INPUT][0];
    sizes[OUTPUT][1] = !have_u ? cvSize(0,0) : compact ? cvSize(min_size,min_size) : cvSize(m,m);
    sizes[OUTPUT][2] = !have_v ? cvSize(0,0) : compact ? cvSize(min_size,min_size) : cvSize(n,n);
    sizes[OUTPUT][3] = cvSize(min_size,1);

    for( i = 0; i < 4; i++ )
    {
        sizes[REF_OUTPUT][i] = sizes[OUTPUT][i];
        types[REF_OUTPUT][i] = types[OUTPUT][i];
    }
}


void CxCore_SVDTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    const char* output_str = cvReadString( find_timing_param("output"), "all" );
    bool need_all = strcmp( output_str, "all" ) == 0;
    int i, count = test_array[OUTPUT].size();
    vector_w = true;
    symmetric = false;
    compact = true;
    sizes[TEMP][0] = cvSize(1,sizes[INPUT][0].height);
    if( need_all )
    {
        have_u = have_v = true;
    }
    else
    {
        have_u = have_v = false;
        sizes[TEMP][1] = sizes[TEMP][2] = cvSize(0,0);
    }

    flags = CV_SVD_U_T + CV_SVD_V_T;
    for( i = 0; i < count; i++ )
        sizes[OUTPUT][i] = sizes[REF_OUTPUT][i] = cvSize(0,0);
    sizes[OUTPUT][0] = cvSize(1,1);
}


int CxCore_SVDTest::write_default_params( CvFileStorage* fs )
{
    int code = CxCore_MatrixTest::write_default_params(fs);
    if( code < 0 || ts->get_testing_mode() != CvTS::TIMING_MODE )
        return code;
    write_string_list( fs, "output", matrix_svd_output_modes );
    return code;
}


void CxCore_SVDTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", have_u ? "all" : "w" );
    ptr += strlen(ptr);
    params_left--;
    CxCore_MatrixTest::print_timing_params( test_case_idx, ptr, params_left );
}


int CxCore_SVDTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        CvMat* input = &test_mat[INPUT][0];
        cvTsFloodWithZeros( input, ts->get_rng() );

        if( symmetric && (have_u || have_v) )
        {
            CvMat* temp = &test_mat[TEMP][have_u ? 1 : 2];
            cvTsGEMM( input, input, 1.,
                      0, 0., temp, CV_GEMM_B_T );
            cvTsCopy( temp, input );
        }

        if( (flags & CV_SVD_MODIFY_A) && test_array[OUTPUT][0] )
            cvTsCopy( input, &test_mat[OUTPUT][0] );
    }

    return code;
}


void CxCore_SVDTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-2.);
    *high = cvScalarAll(2.);
}

double CxCore_SVDTest::get_success_error_level( int test_case_idx, int i, int j )
{
    int input_depth = CV_MAT_DEPTH(cvGetElemType( test_array[INPUT][0] ));
    double input_precision = input_depth < CV_32F ? 0 : input_depth == CV_32F ?
                            5e-5 : 5e-11;
    double output_precision = CvArrTest::get_success_error_level( test_case_idx, i, j );
    return MAX(input_precision, output_precision);
}

void CxCore_SVDTest::run_func()
{
    CvArr* src = test_array[!(flags & CV_SVD_MODIFY_A) ? INPUT : OUTPUT][0];
    if( !src )
        src = test_array[INPUT][0];
    cvSVD( src, test_array[TEMP][0], test_array[TEMP][1], test_array[TEMP][2], flags );
}


void CxCore_SVDTest::prepare_to_validation( int )
{
    CvMat* input = &test_mat[INPUT][0];
    int m = input->rows, n = input->cols, min_size = MIN(m, n);
    CvMat *src, *dst, *w;
    double prev = 0, threshold = CV_MAT_TYPE(input->type) == CV_32FC1 ? FLT_EPSILON : DBL_EPSILON;
    int i, j = 0, step;

    if( have_u )
    {
        src = &test_mat[TEMP][1];
        dst = &test_mat[OUTPUT][1];
        cvTsGEMM( src, src, 1., 0, 0., dst, src->rows == dst->rows ? CV_GEMM_B_T : CV_GEMM_A_T );
        cvTsSetIdentity( &test_mat[REF_OUTPUT][1], cvScalarAll(1.) );
    }

    if( have_v )
    {
        src = &test_mat[TEMP][2];
        dst = &test_mat[OUTPUT][2];
        cvTsGEMM( src, src, 1., 0, 0., dst, src->rows == dst->rows ? CV_GEMM_B_T : CV_GEMM_A_T );
        cvTsSetIdentity( &test_mat[REF_OUTPUT][2], cvScalarAll(1.) );
    }

    w = &test_mat[TEMP][0];
    step = w->rows == 1 ? 1 : w->step/CV_ELEM_SIZE(w->type);
    for( i = 0; i < min_size; i++ )
    {
        double norm = 0, aii;
        uchar* row_ptr;
        if( w->rows > 1 && w->cols > 1 )
        {
            CvMat row;
            cvGetRow( w, &row, i );
            norm = cvNorm( &row, 0, CV_L1 );
            j = i;
            row_ptr = row.data.ptr;
        }
        else
        {
            row_ptr = w->data.ptr;
            j = i*step;
        }

        aii = CV_MAT_TYPE(w->type) == CV_32FC1 ?
            (double)((float*)row_ptr)[j] : ((double*)row_ptr)[j];
        if( w->rows == 1 || w->cols == 1 )
            norm = aii;
        norm = fabs(norm - aii);
        test_mat[OUTPUT][3].data.ptr[i] = aii >= 0 && norm < threshold && (i == 0 || aii <= prev);
        prev = aii;
    }

    cvTsAdd( 0, cvScalarAll(0.), 0, cvScalarAll(0.),
             cvScalarAll(1.), &test_mat[REF_OUTPUT][3], 0 );

    if( have_u && have_v )
    {
        if( vector_w )
        {
            cvTsZero( &test_mat[TEMP][3] );
            for( i = 0; i < min_size; i++ )
            {
                double val = cvGetReal1D( w, i );
                cvSetReal2D( &test_mat[TEMP][3], i, i, val );
            }
            w = &test_mat[TEMP][3];
        }

        if( m >= n )
        {
            cvTsGEMM( &test_mat[TEMP][1], w, 1., 0, 0., &test_mat[REF_OUTPUT][0],
                      flags & CV_SVD_U_T ? CV_GEMM_A_T : 0 );
            cvTsGEMM( &test_mat[REF_OUTPUT][0], &test_mat[TEMP][2], 1., 0, 0.,
                      &test_mat[OUTPUT][0], flags & CV_SVD_V_T ? 0 : CV_GEMM_B_T );
        }
        else
        {
            cvTsGEMM( w, &test_mat[TEMP][2], 1., 0, 0., &test_mat[REF_OUTPUT][0],
                      flags & CV_SVD_V_T ? 0 : CV_GEMM_B_T );
            cvTsGEMM( &test_mat[TEMP][1], &test_mat[REF_OUTPUT][0], 1., 0, 0.,
                      &test_mat[OUTPUT][0], flags & CV_SVD_U_T ? CV_GEMM_A_T : 0 );
        }

        cvTsCopy( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0], 0 );
    }
}


CxCore_SVDTest svd_test;


///////////////// SVBkSb /////////////////////

class CxCore_SVBkSbTest : public CxCore_MatrixTest
{
public:
    CxCore_SVBkSbTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool* are_images );
    double get_success_error_level( int test_case_idx, int i, int j );
    void get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int test_case_idx );
    int flags;
    bool have_b, symmetric, compact, vector_w;
};


CxCore_SVBkSbTest::CxCore_SVBkSbTest() :
    CxCore_MatrixTest( "matrix-svbksb", "cvSVBkSb", 2, 1, false, false, 1 ),
        flags(0), have_b(false), symmetric(false), compact(false), vector_w(false)
{
    test_case_count = 100;
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
    test_array[TEMP].push(NULL);
}


void CxCore_SVBkSbTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int bits = cvTsRandInt(rng);
    CxCore_MatrixTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int min_size, i, m, n;
    CvSize b_size;

    min_size = MIN( sizes[INPUT][0].width, sizes[INPUT][0].height );

    flags = bits & (CV_SVD_MODIFY_A+CV_SVD_U_T+CV_SVD_V_T);
    have_b = (bits & 16) != 0;
    symmetric = (bits & 32) != 0;
    compact = (bits & 64) != 0;
    vector_w = (bits & 128) != 0;

    if( symmetric )
        sizes[INPUT][0] = cvSize(min_size, min_size);

    m = sizes[INPUT][0].height;
    n = sizes[INPUT][0].width;

    sizes[INPUT][1] = cvSize(0,0);
    b_size = cvSize(m,m);
    if( have_b )
    {
        sizes[INPUT][1].height = sizes[INPUT][0].height;
        sizes[INPUT][1].width = cvTsRandInt(rng) % 100 + 1;
        b_size = sizes[INPUT][1];
    }

    if( compact )
        sizes[TEMP][0] = cvSize(min_size, min_size);
    else
        sizes[TEMP][0] = sizes[INPUT][0];

    if( vector_w )
    {
        if( bits & 256 )
            sizes[TEMP][0] = cvSize(1, min_size);
        else
            sizes[TEMP][0] = cvSize(min_size, 1);
    }

    sizes[TEMP][1] = compact ? cvSize(min_size, m) : cvSize(m, m);

    if( flags & CV_SVD_U_T )
        CV_SWAP( sizes[TEMP][1].width, sizes[TEMP][1].height, i );

    sizes[TEMP][2] = compact ? cvSize(n, min_size) : cvSize(n, n);

    if( !(flags & CV_SVD_V_T) )
        CV_SWAP( sizes[TEMP][2].width, sizes[TEMP][2].height, i );

    types[TEMP][0] = types[TEMP][1] = types[TEMP][2] = types[INPUT][0];
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][0];
    sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize( b_size.width, n );
}


void CxCore_SVBkSbTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                                                    CvSize** sizes, int** types,
                                                    CvSize** whole_sizes, bool* are_images )
{
    CxCore_MatrixTest::get_timing_test_array_types_and_sizes( test_case_idx,
                                    sizes, types, whole_sizes, are_images );
    have_b = true;
    vector_w = true;
    compact = true;
    sizes[TEMP][0] = cvSize(1,sizes[INPUT][0].height);
    sizes[INPUT][1] = sizes[OUTPUT][0] = sizes[REF_OUTPUT][0] = cvSize(1,sizes[INPUT][0].height);
    flags = CV_SVD_U_T + CV_SVD_V_T;
}


int CxCore_SVBkSbTest::prepare_test_case( int test_case_idx )
{
    int code = CxCore_MatrixTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        CvMat* input = &test_mat[INPUT][0];
        cvTsFloodWithZeros( input, ts->get_rng() );

        if( symmetric )
        {
            CvMat* temp = &test_mat[TEMP][1];
            cvTsGEMM( input, input, 1., 0, 0., temp, CV_GEMM_B_T );
            cvTsCopy( temp, input );
        }

        cvSVD( input, test_array[TEMP][0], test_array[TEMP][1], test_array[TEMP][2], flags );
    }

    return code;
}


void CxCore_SVBkSbTest::get_minmax_bounds( int /*i*/, int /*j*/, int /*type*/, CvScalar* low, CvScalar* high )
{
    *low = cvScalarAll(-2.);
    *high = cvScalarAll(2.);
}


double CxCore_SVBkSbTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return CV_MAT_DEPTH(cvGetElemType(test_array[INPUT][0])) == CV_32F ? 1e-3 : 1e-7;
}


void CxCore_SVBkSbTest::run_func()
{
    cvSVBkSb( test_array[TEMP][0], test_array[TEMP][1], test_array[TEMP][2],
              test_array[INPUT][1], test_array[OUTPUT][0], flags );
}


void CxCore_SVBkSbTest::prepare_to_validation( int )
{
    CvMat* input = &test_mat[INPUT][0];
    int i, m = input->rows, n = input->cols, min_size = MIN(m, n), nb;
    bool is_float = CV_MAT_DEPTH(input->type) == CV_32F;
    CvSize w_size = compact ? cvSize(min_size,min_size) : cvSize(m,n);
    CvMat* w = &test_mat[TEMP][0];
    CvMat* wdb = cvCreateMat( w_size.height, w_size.width, CV_64FC1 );
    // use exactly the same threshold as in icvSVD... ,
    // so the changes in the library and here should be synchronized.
    double threshold = cvSum( w ).val[0]*2*(is_float ? FLT_EPSILON : DBL_EPSILON);
    CvMat *u, *v, *b, *t0, *t1;

    cvTsZero(wdb);
    for( i = 0; i < min_size; i++ )
    {
        double wii = vector_w ? cvGetReal1D(w,i) : cvGetReal2D(w,i,i);
        cvSetReal2D( wdb, i, i, wii > threshold ? 1./wii : 0. );
    }

    u = &test_mat[TEMP][1];
    v = &test_mat[TEMP][2];
    b = 0;
    nb = m;

    if( test_array[INPUT][1] )
    {
        b = &test_mat[INPUT][1];
        nb = b->cols;
    }

    if( is_float )
    {
        u = cvCreateMat( u->rows, u->cols, CV_64F );
        cvTsConvert( &test_mat[TEMP][1], u );
        if( b )
        {
            b = cvCreateMat( b->rows, b->cols, CV_64F );
            cvTsConvert( &test_mat[INPUT][1], b );
        }
    }

    t0 = cvCreateMat( wdb->cols, nb, CV_64F );

    if( b )
        cvTsGEMM( u, b, 1., 0, 0., t0, !(flags & CV_SVD_U_T) ? CV_GEMM_A_T : 0 );
    else if( flags & CV_SVD_U_T )
        cvTsCopy( u, t0 );
    else
        cvTsTranspose( u, t0 );

    if( is_float )
    {
        cvReleaseMat( &b );

        if( !symmetric )
        {
            cvReleaseMat( &u );
            v = cvCreateMat( v->rows, v->cols, CV_64F );
        }
        else
        {
            v = u;
            u = 0;
        }
        cvTsConvert( &test_mat[TEMP][2], v );
    }

    t1 = cvCreateMat( wdb->rows, nb, CV_64F );
    cvTsGEMM( wdb, t0, 1, 0, 0, t1, 0 );

    if( !is_float || !symmetric )
    {
        cvReleaseMat( &t0 );
        t0 = !is_float ? &test_mat[REF_OUTPUT][0] : cvCreateMat( test_mat[REF_OUTPUT][0].rows, nb, CV_64F );
    }

    cvTsGEMM( v, t1, 1, 0, 0, t0, flags & CV_SVD_V_T ? CV_GEMM_A_T : 0 );
    cvReleaseMat( &t1 );

    if( t0 != &test_mat[REF_OUTPUT][0] )
    {
        cvTsConvert( t0, &test_mat[REF_OUTPUT][0] );
        cvReleaseMat( &t0 );
    }

    if( v != &test_mat[TEMP][2] )
        cvReleaseMat( &v );

    cvReleaseMat( &wdb );
}


CxCore_SVBkSbTest svbksb_test;


// TODO: eigenvv, invsqrt, cbrt, fastarctan, (round, floor, ceil(?)),

/* End of file. */

