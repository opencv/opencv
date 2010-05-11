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

static const char* templmatch_param_names[] = { "template_size", "method", "size", "channels", "depth", 0 };
static const int templmatch_depths[] = { CV_8U, CV_32F, -1 };
static const int templmatch_channels[] = { 1, 3, -1 };

static const CvSize templmatch_sizes[] = {{320, 240}, {1024,768}, {-1,-1}};
static const CvSize templmatch_whole_sizes[] = {{320,240}, {1024,768}, {-1,-1}};
static const CvSize templmatch_template_sizes[] = {{15,15}, {60,60}, {-1,-1}};
static const char* templmatch_methods[] = { "sqdiff", "sqdiff_norm", "ccorr", "ccorr_normed", "ccoeff", "ccoeff_normed", 0 };

class CV_TemplMatchTest : public CvArrTest
{
public:
    CV_TemplMatchTest();

protected:
    int read_params( CvFileStorage* fs );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int write_default_params(CvFileStorage* fs);
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void print_timing_params( int test_case_idx, char* ptr, int params_left );

    int max_template_size;
    int method;
    bool test_cpp;
};


CV_TemplMatchTest::CV_TemplMatchTest()
    : CvArrTest( "match-template", "cvMatchTemplate", "" )
{
    test_array[INPUT].push(NULL);
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    element_wise_relative_error = false;
    max_template_size = 100;
    method = 0;

    size_list = templmatch_sizes;
    whole_size_list = templmatch_whole_sizes;
    cn_list = templmatch_channels;
    depth_list = templmatch_depths;

    default_timing_param_names = templmatch_param_names;
    test_cpp = false;
}


int CV_TemplMatchTest::read_params( CvFileStorage* fs )
{
    int code = CvArrTest::read_params( fs );
    if( code < 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        max_template_size = cvReadInt( find_param( fs, "max_template_size" ), max_template_size );
        max_template_size = cvTsClipInt( max_template_size, 1, 100 );
    }

    return code;
}


int CV_TemplMatchTest::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code < 0 )
        return code;

    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        write_param( fs, "max_template_size", max_template_size );
    }
    else
    {
        int i;
        start_write_param( fs );

        cvStartWriteStruct( fs, "template_size", CV_NODE_SEQ+CV_NODE_FLOW );
        for( i = 0; templmatch_template_sizes[i].width >= 0; i++ )
        {
            cvStartWriteStruct( fs, 0, CV_NODE_SEQ+CV_NODE_FLOW );
            cvWriteInt( fs, 0, templmatch_template_sizes[i].width );
            cvWriteInt( fs, 0, templmatch_template_sizes[i].height );
            cvEndWriteStruct(fs);
        }
        cvEndWriteStruct(fs);

        write_string_list( fs, "method", templmatch_methods );
    }

    return code;
}


void CV_TemplMatchTest::get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high )
{
    CvArrTest::get_minmax_bounds( i, j, type, low, high );
    int depth = CV_MAT_DEPTH(type);
    if( depth == CV_32F )
    {
        *low = cvScalarAll(-10.);
        *high = cvScalarAll(10.);
    }
}


void CV_TemplMatchTest::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth = cvTsRandInt(rng) % 2, cn = cvTsRandInt(rng) & 1 ? 3 : 1;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : CV_32F;

    types[INPUT][0] = types[INPUT][1] = CV_MAKETYPE(depth,cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_32FC1;

    sizes[INPUT][1].width = cvTsRandInt(rng)%MIN(sizes[INPUT][1].width,max_template_size) + 1;
    sizes[INPUT][1].height = cvTsRandInt(rng)%MIN(sizes[INPUT][1].height,max_template_size) + 1;
    sizes[OUTPUT][0].width = sizes[INPUT][0].width - sizes[INPUT][1].width + 1;
    sizes[OUTPUT][0].height = sizes[INPUT][0].height - sizes[INPUT][1].height + 1;
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];

    method = cvTsRandInt(rng)%6;
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


void CV_TemplMatchTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    const char* method_str = cvReadString( find_timing_param( "method" ), "ccorr" );
    const CvFileNode* node = find_timing_param( "template_size" );
    CvSize templ_size, result_size;

    assert( node && CV_NODE_IS_SEQ( node->tag ));

    method = strncmp( method_str, "sqdiff", 6 ) == 0 ? CV_TM_SQDIFF :
             strncmp( method_str, "ccorr", 5 ) == 0 ? CV_TM_CCORR : CV_TM_CCOEFF;
    method += strstr( method_str, "_normed" ) != 0;

    cvReadRawData( ts->get_file_storage(), node, &templ_size, "2i" );

    sizes[INPUT][1] = whole_sizes[INPUT][1] = templ_size;
    result_size.width = sizes[INPUT][0].width - templ_size.width + 1;
    result_size.height = sizes[INPUT][0].height - templ_size.height + 1;
    assert( result_size.width > 0 && result_size.height > 0 );
    sizes[OUTPUT][0] = whole_sizes[OUTPUT][0] = result_size;

    types[OUTPUT][0] = CV_32FC1;
}


void CV_TemplMatchTest::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", cvReadString( find_timing_param( "method" ), "ccorr" ) );
    ptr += strlen(ptr);
    sprintf( ptr, "templ_size=%dx%d,", test_mat[INPUT][1].width, test_mat[INPUT][1].height );
    ptr += strlen(ptr);
    params_left -= 2;

    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


double CV_TemplMatchTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    if( CV_MAT_DEPTH(test_mat[INPUT][1].type) == CV_8U ||
        (method >= CV_TM_CCOEFF && test_mat[INPUT][1].cols*test_mat[INPUT][1].rows <= 2) )
        return 1e-2;
    else
        return 1e-3;
}


void CV_TemplMatchTest::run_func()
{
    if(!test_cpp)
        cvMatchTemplate( test_array[INPUT][0], test_array[INPUT][1], test_array[OUTPUT][0], method );
    else
    {
        cv::Mat _out = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::matchTemplate(cv::cvarrToMat(test_array[INPUT][0]), cv::cvarrToMat(test_array[INPUT][1]), _out, method);
    }
}


static void cvTsMatchTemplate( const CvMat* img, const CvMat* templ, CvMat* result, int method )
{
    int i, j, k, l;
    int depth = CV_MAT_DEPTH(img->type), cn = CV_MAT_CN(img->type);
    int width_n = templ->cols*cn, height = templ->rows;
    int a_step = img->step / CV_ELEM_SIZE(img->type & CV_MAT_DEPTH_MASK);
    int b_step = templ->step / CV_ELEM_SIZE(templ->type & CV_MAT_DEPTH_MASK);
    CvScalar b_mean, b_sdv;
    double b_denom = 1., b_sum2 = 0;
    int area = templ->rows*templ->cols;

    cvTsMeanStdDevNonZero( templ, 0, &b_mean, &b_sdv, 0 );

    for( i = 0; i < cn; i++ )
        b_sum2 += (b_sdv.val[i]*b_sdv.val[i] + b_mean.val[i]*b_mean.val[i])*area;

    if( CV_SQR(b_sdv.val[0]) + CV_SQR(b_sdv.val[1]) +
        CV_SQR(b_sdv.val[2]) + CV_SQR(b_sdv.val[3]) < DBL_EPSILON &&
        method == CV_TM_CCOEFF_NORMED )
    {
        cvSet( result, cvScalarAll(1.) );
        return;
    }

    if( method & 1 )
    {
        b_denom = 0;
        if( method != CV_TM_CCOEFF_NORMED )
        {
            b_denom = b_sum2;
        }
        else
        {
            for( i = 0; i < cn; i++ )
                b_denom += b_sdv.val[i]*b_sdv.val[i]*area;
        }
        b_denom = sqrt(b_denom);
        if( b_denom == 0 )
            b_denom = 1.;
    }

    assert( CV_TM_SQDIFF <= method && method <= CV_TM_CCOEFF_NORMED );

    for( i = 0; i < result->rows; i++ )
    {
        for( j = 0; j < result->cols; j++ )
        {
            CvScalar a_sum = {{ 0, 0, 0, 0 }}, a_sum2 = {{ 0, 0, 0, 0 }};
            CvScalar ccorr = {{ 0, 0, 0, 0 }};
            double value = 0.;

            if( depth == CV_8U )
            {
                const uchar* a = img->data.ptr + i*img->step + j*cn;
                const uchar* b = templ->data.ptr;

                if( cn == 1 || method < CV_TM_CCOEFF )
                {
                    for( k = 0; k < height; k++, a += a_step, b += b_step )
                        for( l = 0; l < width_n; l++ )
                        {
                            ccorr.val[0] += a[l]*b[l];
                            a_sum.val[0] += a[l];
                            a_sum2.val[0] += a[l]*a[l];
                        }
                }
                else
                {
                    for( k = 0; k < height; k++, a += a_step, b += b_step )
                        for( l = 0; l < width_n; l += 3 )
                        {
                            ccorr.val[0] += a[l]*b[l];
                            ccorr.val[1] += a[l+1]*b[l+1];
                            ccorr.val[2] += a[l+2]*b[l+2];
                            a_sum.val[0] += a[l];
                            a_sum.val[1] += a[l+1];
                            a_sum.val[2] += a[l+2];
                            a_sum2.val[0] += a[l]*a[l];
                            a_sum2.val[1] += a[l+1]*a[l+1];
                            a_sum2.val[2] += a[l+2]*a[l+2];
                        }
                }
            }
            else
            {
                const float* a = (const float*)(img->data.ptr + i*img->step) + j*cn;
                const float* b = (const float*)templ->data.ptr;

                if( cn == 1 || method < CV_TM_CCOEFF )
                {
                    for( k = 0; k < height; k++, a += a_step, b += b_step )
                        for( l = 0; l < width_n; l++ )
                        {
                            ccorr.val[0] += a[l]*b[l];
                            a_sum.val[0] += a[l];
                            a_sum2.val[0] += a[l]*a[l];
                        }
                }
                else
                {
                    for( k = 0; k < height; k++, a += a_step, b += b_step )
                        for( l = 0; l < width_n; l += 3 )
                        {
                            ccorr.val[0] += a[l]*b[l];
                            ccorr.val[1] += a[l+1]*b[l+1];
                            ccorr.val[2] += a[l+2]*b[l+2];
                            a_sum.val[0] += a[l];
                            a_sum.val[1] += a[l+1];
                            a_sum.val[2] += a[l+2];
                            a_sum2.val[0] += a[l]*a[l];
                            a_sum2.val[1] += a[l+1]*a[l+1];
                            a_sum2.val[2] += a[l+2]*a[l+2];
                        }
                }
            }

            switch( method )
            {
            case CV_TM_CCORR:
            case CV_TM_CCORR_NORMED:
                value = ccorr.val[0];
                break;
            case CV_TM_SQDIFF:
            case CV_TM_SQDIFF_NORMED:
                value = (a_sum2.val[0] + b_sum2 - 2*ccorr.val[0]);
                break;
            default:
                value = (ccorr.val[0] - a_sum.val[0]*b_mean.val[0]+
                         ccorr.val[1] - a_sum.val[1]*b_mean.val[1]+
                         ccorr.val[2] - a_sum.val[2]*b_mean.val[2]);
            }

            if( method & 1 )
            {
                double denom;

                // calc denominator
                if( method != CV_TM_CCOEFF_NORMED )
                {
                    denom = a_sum2.val[0] + a_sum2.val[1] + a_sum2.val[2];
                }
                else
                {
                    denom = a_sum2.val[0] - (a_sum.val[0]*a_sum.val[0])/area;
                    denom += a_sum2.val[1] - (a_sum.val[1]*a_sum.val[1])/area;
                    denom += a_sum2.val[2] - (a_sum.val[2]*a_sum.val[2])/area;
                }
                denom = sqrt(MAX(denom,0))*b_denom;
                if( fabs(value) < denom )
                    value /= denom;
                else if( fabs(value) < denom*1.125 )
                    value = value > 0 ? 1 : -1;
                else
                    value = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
            }

            ((float*)(result->data.ptr + result->step*i))[j] = (float)value;
        }
    }
}


void CV_TemplMatchTest::prepare_to_validation( int /*test_case_idx*/ )
{
    cvTsMatchTemplate( &test_mat[INPUT][0], &test_mat[INPUT][1],
                       &test_mat[REF_OUTPUT][0], method );

    //if( ts->get_current_test_info()->test_case_idx == 0 )
    /*{
        CvFileStorage* fs = cvOpenFileStorage( "_match_template.yml", 0, CV_STORAGE_WRITE );
        cvWrite( fs, "image", &test_mat[INPUT][0] );
        cvWrite( fs, "template", &test_mat[INPUT][1] );
        cvWrite( fs, "ref", &test_mat[REF_OUTPUT][0] );
        cvWrite( fs, "opencv", &test_mat[OUTPUT][0] );
        cvWriteInt( fs, "method", method );
        cvReleaseFileStorage( &fs );
    }*/

    if( method >= CV_TM_CCOEFF )
    {
        // avoid numerical stability problems in singular cases (when the results are near to 0)
        const double delta = 10.;
        cvTsAdd( &test_mat[REF_OUTPUT][0], cvScalar(1.), 0, cvScalar(0.),
                 cvScalar(delta), &test_mat[REF_OUTPUT][0], 0 );
        cvTsAdd( &test_mat[OUTPUT][0], cvScalar(1.), 0, cvScalar(0.),
                 cvScalar(delta), &test_mat[OUTPUT][0], 0 );
    }
}


CV_TemplMatchTest templ_match;
