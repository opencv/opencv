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


/////////////////////////// base test class for color transformations /////////////////////////

static const char* cvtcolor_param_names[] = { "op", "size", "depth", 0 };
static const int cvtcolor_depths_8_16_32[] = { CV_8U, CV_16U, CV_32F, -1 };
static const int cvtcolor_depths_8_32[] = { CV_8U, CV_32F, -1 };
static const int cvtcolor_depths_8[] = { CV_8U, -1 };

static const CvSize cvtcolor_sizes[] = {{30,30}, {320, 240}, {720,480}, {-1,-1}};
static const CvSize cvtcolor_whole_sizes[] = {{320,240}, {320, 240}, {720,480}, {-1,-1}};

class CV_ColorCvtBaseTestImpl : public CvArrTest
{
public:
    CV_ColorCvtBaseTestImpl( const char* test_name, const char* test_funcs,
                         bool custom_inv_transform, bool allow_32f, bool allow_16u );

protected:
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high );
    int write_default_params(CvFileStorage* fs);
    void print_timing_params( int test_case_idx, char* ptr, int params_left );

    // input --- fwd_transform -> ref_output[0]
    virtual void convert_forward( const CvMat* src, CvMat* dst );
    // ref_output[0] --- inv_transform ---> ref_output[1] (or input -- copy --> ref_output[1])
    virtual void convert_backward( const CvMat* src, const CvMat* dst, CvMat* dst2 );

    // called from default implementation of convert_forward
    virtual void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );

    // called from default implementation of convert_backward
    virtual void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );

    const char* fwd_code_str;
    const char* inv_code_str;

    void run_func();
    bool allow_16u, allow_32f;
    int blue_idx;
    bool inplace;
    bool custom_inv_transform;
    int fwd_code, inv_code;
    int timing_code;
    bool test_cpp;
    bool hue_channel;
};


CV_ColorCvtBaseTestImpl::CV_ColorCvtBaseTestImpl( const char* test_name, const char* test_funcs,
                                          bool _custom_inv_transform, bool _allow_32f, bool _allow_16u )
    : CvArrTest( test_name, test_funcs, "" )
{
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    allow_16u = _allow_16u;
    allow_32f = _allow_32f;
    custom_inv_transform = _custom_inv_transform;
    fwd_code = inv_code = timing_code = -1;
    element_wise_relative_error = false;

    size_list = cvtcolor_sizes;
    whole_size_list = cvtcolor_whole_sizes;
    depth_list = cvtcolor_depths_8_16_32;

    fwd_code_str = inv_code_str = 0;

    default_timing_param_names = 0;
    test_cpp = false;
    hue_channel = false;
}


int CV_ColorCvtBaseTestImpl::write_default_params( CvFileStorage* fs )
{
    int code = CvArrTest::write_default_params( fs );
    if( code >= 0 && ts->get_testing_mode() == CvTS::TIMING_MODE && fwd_code_str != 0 )
    {
        start_write_param( fs );
        cvStartWriteStruct( fs, "op", CV_NODE_SEQ+CV_NODE_FLOW );
        if( strcmp( fwd_code_str, "" ) != 0 )
            cvWriteString( fs, 0, fwd_code_str );
        if( strcmp( inv_code_str, "" ) != 0 )
            cvWriteString( fs, 0, inv_code_str );
        cvEndWriteStruct(fs);
    }
    return code;
}


void CV_ColorCvtBaseTestImpl::get_minmax_bounds( int i, int j, int type, CvScalar* low, CvScalar* high )
{
    CvArrTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT )
    {
        int depth = CV_MAT_DEPTH(type);
        *low = cvScalarAll(0.);
        *high = cvScalarAll( depth == CV_8U ? 256 : depth == CV_16U ? 65536 : 1. );
    }
}


void CV_ColorCvtBaseTestImpl::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    int depth, cn;
    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( allow_16u && allow_32f )
    {
        depth = cvTsRandInt(rng) % 3;
        depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    }
    else if( allow_16u || allow_32f )
    {
        depth = cvTsRandInt(rng) % 2;
        depth = depth == 0 ? CV_8U : allow_16u ? CV_16U : CV_32F;
    }
    else
        depth = CV_8U;

    cn = (cvTsRandInt(rng) & 1) + 3;
    blue_idx = cvTsRandInt(rng) & 1 ? 2 : 0;

    types[INPUT][0] = CV_MAKETYPE(depth, cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, 3);
    if( test_array[OUTPUT].size() > 1 )
        types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(depth, cn);

    inplace = cn == 3 && cvTsRandInt(rng) % 2 != 0;
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


void CV_ColorCvtBaseTestImpl::get_timing_test_array_types_and_sizes( int test_case_idx,
                    CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CvArrTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                      whole_sizes, are_images );
    int depth = CV_MAT_DEPTH(types[INPUT][0]);
    const char* op_str = cvReadString( find_timing_param( "op" ), "none" );
    timing_code = strcmp( op_str, inv_code_str ) == 0 ? inv_code : fwd_code;
    types[INPUT][0] = types[OUTPUT][0] = CV_MAKETYPE(depth, 3);
    if( test_array[OUTPUT].size() > 1 )
        types[OUTPUT][1] = types[OUTPUT][0];
}


int CV_ColorCvtBaseTestImpl::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        if( inplace )
            cvTsCopy( &test_mat[INPUT][0], &test_mat[OUTPUT][0] );
        if( ts->get_testing_mode() == CvTS::TIMING_MODE && timing_code != fwd_code )
        {
            int save_timing_code = timing_code;
            timing_code = fwd_code;
            run_func(); // initialize the intermediate image for backward color space transformation
            timing_code = save_timing_code;
        }
    }
    return code;
}


void CV_ColorCvtBaseTestImpl::print_timing_params( int test_case_idx, char* ptr, int params_left )
{
    sprintf( ptr, "%s,", cvReadString( find_timing_param( "op" ), fwd_code_str ));
    ptr += strlen(ptr);
    params_left--;

    CvArrTest::print_timing_params( test_case_idx, ptr, params_left );
}


void CV_ColorCvtBaseTestImpl::run_func()
{
    if( ts->get_testing_mode() == CvTS::CORRECTNESS_CHECK_MODE )
    {
        CvArr* out0 = test_array[OUTPUT][0];
        cv::Mat _out0 = cv::cvarrToMat(out0), _out1 = cv::cvarrToMat(test_array[OUTPUT][1]);
        
        if(!test_cpp)
            cvCvtColor( inplace ? out0 : test_array[INPUT][0], out0, fwd_code );
        else
            cv::cvtColor( cv::cvarrToMat(inplace ? out0 : test_array[INPUT][0]), _out0, fwd_code, _out0.channels());
        
        if( inplace )
        {
            cvCopy( out0, test_array[OUTPUT][1] );
            out0 = test_array[OUTPUT][1];
        }
        if(!test_cpp)
            cvCvtColor( out0, test_array[OUTPUT][1], inv_code );
        else
            cv::cvtColor(cv::cvarrToMat(out0), _out1, inv_code, _out1.channels());
    }
    else if( timing_code == fwd_code )
        cvCvtColor( test_array[INPUT][0], test_array[OUTPUT][0], timing_code );
    else
        cvCvtColor( test_array[OUTPUT][0], test_array[OUTPUT][1], timing_code );
}


void CV_ColorCvtBaseTestImpl::prepare_to_validation( int /*test_case_idx*/ )
{
    convert_forward( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0] );
    convert_backward( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0],
                      &test_mat[REF_OUTPUT][1] );
    int depth = CV_MAT_DEPTH(test_mat[REF_OUTPUT][0].type);
    if( depth == CV_8U && hue_channel )
    {
        for( int y = 0; y < test_mat[REF_OUTPUT][0].rows; y++ )
            for( int x = 0; x < test_mat[REF_OUTPUT][0].cols; x++ )
            {
                uchar* h0 = test_mat[REF_OUTPUT][0].data.ptr + test_mat[REF_OUTPUT][0].step*y + x*3;
                uchar* h = test_mat[OUTPUT][0].data.ptr + test_mat[OUTPUT][0].step*y + x*3;
                if( abs(*h - *h0) == 180 )
                    if( *h == 0 ) *h = 180;
            }
    }
}


void CV_ColorCvtBaseTestImpl::convert_forward( const CvMat* src, CvMat* dst )
{
    const float c8u = 0.0039215686274509803f; // 1./255
    const float c16u = 1.5259021896696422e-005f; // 1./65535
    int depth = CV_MAT_DEPTH(src->type);
    int cn = CV_MAT_CN(src->type), dst_cn = CV_MAT_CN(dst->type);
    int cols = src->cols, dst_cols_n = dst->cols*dst_cn;
    float* src_buf = (float*)cvAlloc( src->cols*3*sizeof(src_buf[0]) );
    float* dst_buf = (float*)cvAlloc( dst->cols*3*sizeof(dst_buf[0]) );
    int i, j;

    assert( (cn == 3 || cn == 4) && (dst_cn == 3 || dst_cn == 1) );

    for( i = 0; i < src->rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = (const uchar*)(src->data.ptr + i*src->step);
                uchar* dst_row = (uchar*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    src_buf[j*3] = src_row[j*cn + blue_idx]*c8u;
                    src_buf[j*3+1] = src_row[j*cn + 1]*c8u;
                    src_buf[j*3+2] = src_row[j*cn + (blue_idx^2)]*c8u;
                }

                convert_row_bgr2abc_32f_c3( src_buf, dst_buf, cols );

                for( j = 0; j < dst_cols_n; j++ )
                {
                    int t = cvRound( dst_buf[j] );
                    dst_row[j] = CV_CAST_8U(t);
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = (const ushort*)(src->data.ptr + i*src->step);
                ushort* dst_row = (ushort*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    src_buf[j*3] = src_row[j*cn + blue_idx]*c16u;
                    src_buf[j*3+1] = src_row[j*cn + 1]*c16u;
                    src_buf[j*3+2] = src_row[j*cn + (blue_idx^2)]*c16u;
                }

                convert_row_bgr2abc_32f_c3( src_buf, dst_buf, cols );

                for( j = 0; j < dst_cols_n; j++ )
                {
                    int t = cvRound( dst_buf[j] );
                    dst_row[j] = CV_CAST_16U(t);
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = (const float*)(src->data.ptr + i*src->step);
                float* dst_row = (float*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    src_buf[j*3] = src_row[j*cn + blue_idx];
                    src_buf[j*3+1] = src_row[j*cn + 1];
                    src_buf[j*3+2] = src_row[j*cn + (blue_idx^2)];
                }

                convert_row_bgr2abc_32f_c3( src_buf, dst_row, cols );
            }
            break;
        default:
            assert(0);
        }
    }

    cvFree( &src_buf );
    cvFree( &dst_buf );
}


void CV_ColorCvtBaseTestImpl::convert_row_bgr2abc_32f_c3( const float* /*src_row*/,
                                                      float* /*dst_row*/, int /*n*/ )
{
}


void CV_ColorCvtBaseTestImpl::convert_row_abc2bgr_32f_c3( const float* /*src_row*/,
                                                      float* /*dst_row*/, int /*n*/ )
{
}


void CV_ColorCvtBaseTestImpl::convert_backward( const CvMat* src, const CvMat* dst, CvMat* dst2 )
{
    if( custom_inv_transform )
    {
        int depth = CV_MAT_DEPTH(src->type);
        int src_cn = CV_MAT_CN(dst->type), cn = CV_MAT_CN(dst2->type);
        int cols_n = src->cols*src_cn, dst_cols = dst->cols;
        float* src_buf = (float*)cvAlloc( src->cols*3*sizeof(src_buf[0]) );
        float* dst_buf = (float*)cvAlloc( dst->cols*3*sizeof(dst_buf[0]) );
        int i, j;

        assert( cn == 3 || cn == 4 );

        for( i = 0; i < src->rows; i++ )
        {
            switch( depth )
            {
            case CV_8U:
                {
                    const uchar* src_row = (const uchar*)(dst->data.ptr + i*dst->step);
                    uchar* dst_row = (uchar*)(dst2->data.ptr + i*dst2->step);

                    for( j = 0; j < cols_n; j++ )
                        src_buf[j] = src_row[j];

                    convert_row_abc2bgr_32f_c3( src_buf, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        int b = cvRound( dst_buf[j*3]*255. );
                        int g = cvRound( dst_buf[j*3+1]*255. );
                        int r = cvRound( dst_buf[j*3+2]*255. );
                        dst_row[j*cn + blue_idx] = CV_CAST_8U(b);
                        dst_row[j*cn + 1] = CV_CAST_8U(g);
                        dst_row[j*cn + (blue_idx^2)] = CV_CAST_8U(r);
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 255;
                    }
                }
                break;
            case CV_16U:
                {
                    const ushort* src_row = (const ushort*)(dst->data.ptr + i*dst->step);
                    ushort* dst_row = (ushort*)(dst2->data.ptr + i*dst2->step);

                    for( j = 0; j < cols_n; j++ )
                        src_buf[j] = src_row[j];

                    convert_row_abc2bgr_32f_c3( src_buf, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        int b = cvRound( dst_buf[j*3]*65535. );
                        int g = cvRound( dst_buf[j*3+1]*65535. );
                        int r = cvRound( dst_buf[j*3+2]*65535. );
                        dst_row[j*cn + blue_idx] = CV_CAST_16U(b);
                        dst_row[j*cn + 1] = CV_CAST_16U(g);
                        dst_row[j*cn + (blue_idx^2)] = CV_CAST_16U(r);
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 65535;
                    }
                }
                break;
            case CV_32F:
                {
                    const float* src_row = (const float*)(dst->data.ptr + i*dst->step);
                    float* dst_row = (float*)(dst2->data.ptr + i*dst2->step);

                    convert_row_abc2bgr_32f_c3( src_row, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        float b = dst_buf[j*3];
                        float g = dst_buf[j*3+1];
                        float r = dst_buf[j*3+2];
                        dst_row[j*cn + blue_idx] = b;
                        dst_row[j*cn + 1] = g;
                        dst_row[j*cn + (blue_idx^2)] = r;
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 1.f;
                    }
                }
                break;
            default:
                assert(0);
            }
        }

        cvFree( &src_buf );
        cvFree( &dst_buf );
    }
    else
    {
        int i, j, k;
        int elem_size = CV_ELEM_SIZE(src->type), elem_size1 = CV_ELEM_SIZE(src->type & CV_MAT_DEPTH_MASK);
        int width_n = src->cols*elem_size;

        for( i = 0; i < src->rows; i++ )
        {
            memcpy( dst2->data.ptr + i*dst2->step, src->data.ptr + i*src->step, width_n );
            if( CV_MAT_CN(src->type) == 4 )
            {
                // clear the alpha channel
                uchar* ptr = dst2->data.ptr + i*dst2->step + elem_size1*3;
                for( j = 0; j < width_n; j += elem_size )
                {
                    for( k = 0; k < elem_size1; k++ )
                        ptr[j + k] = 0;
                }
            }
        }
    }
}


CV_ColorCvtBaseTestImpl cvtcolor( "color", "", false, false, false );


class CV_ColorCvtBaseTest : public CV_ColorCvtBaseTestImpl
{
public:
    CV_ColorCvtBaseTest( const char* test_name, const char* test_funcs,
                         bool custom_inv_transform, bool allow_32f, bool allow_16u );
};


CV_ColorCvtBaseTest::CV_ColorCvtBaseTest( const char* test_name, const char* test_funcs,
                                          bool _custom_inv_transform, bool _allow_32f, bool _allow_16u )
    : CV_ColorCvtBaseTestImpl( test_name, test_funcs, _custom_inv_transform, _allow_32f, _allow_16u )
{
    default_timing_param_names = cvtcolor_param_names;
    depth_list = 0;
    cn_list = 0;
    size_list = whole_size_list = 0;
}

#undef INIT_FWD_INV_CODES
#define INIT_FWD_INV_CODES( fwd, inv )          \
    fwd_code = CV_##fwd; inv_code = CV_##inv;   \
    fwd_code_str = #fwd; inv_code_str = #inv

//// rgb <=> gray
class CV_ColorGrayTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorGrayTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_ColorGrayTest::CV_ColorGrayTest()
    : CV_ColorCvtBaseTest( "color-gray", "cvCvtColor", true, true, true )
{
    INIT_FWD_INV_CODES( BGR2GRAY, GRAY2BGR );
}


void CV_ColorGrayTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = CV_MAT_CN(types[INPUT][0]);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][0] & CV_MAT_DEPTH_MASK;
    inplace = false;

    if( cn == 3 )
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGR2GRAY, inv_code = CV_GRAY2BGR;
        else
            fwd_code = CV_RGB2GRAY, inv_code = CV_GRAY2RGB;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGRA2GRAY, inv_code = CV_GRAY2BGRA;
        else
            fwd_code = CV_RGBA2GRAY, inv_code = CV_GRAY2RGBA;
    }
}


void CV_ColorGrayTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                    CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ColorCvtBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                                whole_sizes, are_images );
    types[OUTPUT][0] &= CV_MAT_DEPTH_MASK;
}


double CV_ColorGrayTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? 2 : depth == CV_16U ? 16 : 1e-5;
}


void CV_ColorGrayTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    double scale = depth == CV_8U ? 255 : depth == CV_16U ? 65535 : 1;
    double cr = 0.299*scale;
    double cg = 0.587*scale;
    double cb = 0.114*scale;
    int j;

    for( j = 0; j < n; j++ )
        dst_row[j] = (float)(src_row[j*3]*cb + src_row[j*3+1]*cg + src_row[j*3+2]*cr);
}


void CV_ColorGrayTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int j, depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float scale = depth == CV_8U ? (1.f/255) : depth == CV_16U ? 1.f/65535 : 1.f;
    for( j = 0; j < n; j++ )
        dst_row[j*3] = dst_row[j*3+1] = dst_row[j*3+2] = src_row[j]*scale;
}


CV_ColorGrayTest color_gray_test;


//// rgb <=> ycrcb
class CV_ColorYCrCbTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorYCrCbTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorYCrCbTest::CV_ColorYCrCbTest()
    : CV_ColorCvtBaseTest( "color-ycc", "cvCvtColor", true, true, true )
{
    INIT_FWD_INV_CODES( BGR2YCrCb, YCrCb2BGR );
}


void CV_ColorYCrCbTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2YCrCb, inv_code = CV_YCrCb2BGR;
    else
        fwd_code = CV_RGB2YCrCb, inv_code = CV_YCrCb2RGB;
}


double CV_ColorYCrCbTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? 2 : depth == CV_16U ? 32 : 1e-3;
}


void CV_ColorYCrCbTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    double scale = depth == CV_8U ? 255 : depth == CV_16U ? 65535 : 1;
    double bias = depth == CV_8U ? 128 : depth == CV_16U ? 32768 : 0.5;

    double M[] = { 0.299, 0.587, 0.114,
                   0.49981,  -0.41853,  -0.08128,
                   -0.16864,  -0.33107,   0.49970 };
    int j;
    for( j = 0; j < 9; j++ )
        M[j] *= scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double r = src_row[j+2];
        double g = src_row[j+1];
        double b = src_row[j];
        double y = M[0]*r + M[1]*g + M[2]*b;
        double cr = M[3]*r + M[4]*g + M[5]*b + bias;
        double cb = M[6]*r + M[7]*g + M[8]*b + bias;
        dst_row[j] = (float)y;
        dst_row[j+1] = (float)cr;
        dst_row[j+2] = (float)cb;
    }
}


void CV_ColorYCrCbTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    double bias = depth == CV_8U ? 128 : depth == CV_16U ? 32768 : 0.5;
    double scale = depth == CV_8U ? 1./255 : depth == CV_16U ? 1./65535 : 1;
    double M[] = { 1,   1.40252,  0,
                   1,  -0.71440,  -0.34434,
                   1,   0,   1.77305 };
    int j;
    for( j = 0; j < 9; j++ )
        M[j] *= scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double y = src_row[j];
        double cr = src_row[j+1] - bias;
        double cb = src_row[j+2] - bias;
        double r = M[0]*y + M[1]*cr + M[2]*cb;
        double g = M[3]*y + M[4]*cr + M[5]*cb;
        double b = M[6]*y + M[7]*cr + M[8]*cb;
        dst_row[j] = (float)b;
        dst_row[j+1] = (float)g;
        dst_row[j+2] = (float)r;
    }
}


CV_ColorYCrCbTest color_ycrcb_test;



//// rgb <=> hsv
class CV_ColorHSVTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorHSVTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorHSVTest::CV_ColorHSVTest()
    : CV_ColorCvtBaseTest( "color-hsv", "cvCvtColor", true, true, false )
{
    INIT_FWD_INV_CODES( BGR2HSV, HSV2BGR );
    depth_list = cvtcolor_depths_8_32;
    hue_channel = true;
}


void CV_ColorHSVTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2HSV, inv_code = CV_HSV2BGR;
    else
        fwd_code = CV_RGB2HSV, inv_code = CV_HSV2RGB;
}


double CV_ColorHSVTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? (j == 0 ? 4 : 16) : depth == CV_16U ? 32 : 1e-3;
}


void CV_ColorHSVTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float h_scale = depth == CV_8U ? 30.f : 60.f;
    float scale = depth == CV_8U ? 255.f : depth == CV_16U ? 65535.f : 1.f;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];
        float vmin = MIN(r,g);
        float v = MAX(r,g);
        float s, h, diff;
        vmin = MIN(vmin,b);
        v = MAX(v,b);
        diff = v - vmin;
        if( diff == 0 )
            s = h = 0;
        else
        {
            s = diff/(v + FLT_EPSILON);
            diff = 1.f/diff;

            h = r == v ? (g - b)*diff :
                g == v ? 2 + (b - r)*diff : 4 + (r - g)*diff;

            if( h < 0 )
                h += 6;
        }

        dst_row[j] = h*h_scale;
        dst_row[j+1] = s*scale;
        dst_row[j+2] = v*scale;
    }
}

// taken from http://www.cs.rit.edu/~ncs/color/t_convert.html
void CV_ColorHSVTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float h_scale = depth == CV_8U ? 1.f/30 : 1.f/60;
    float scale = depth == CV_8U ? 1.f/255 : depth == CV_16U ? 1.f/65535 : 1;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float h = src_row[j]*h_scale;
        float s = src_row[j+1]*scale;
        float v = src_row[j+2]*scale;
        float r = v, g = v, b = v;

        if( h < 0 )
            h += 6;
        else if( h >= 6 )
            h -= 6;

        if( s != 0 )
        {
            int i = cvFloor(h);
            float f = h - i;
            float p = v*(1 - s);
            float q = v*(1 - s*f);
            float t = v*(1 - s*(1 - f));

            if( i == 0 )
                r = v, g = t, b = p;
            else if( i == 1 )
                r = q, g = v, b = p;
            else if( i == 2 )
                r = p, g = v, b = t;
            else if( i == 3 )
                r = p, g = q, b = v;
            else if( i == 4 )
                r = t, g = p, b = v;
            else
                r = v, g = p, b = q;
        }

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


CV_ColorHSVTest color_hsv_test;



//// rgb <=> hls
class CV_ColorHLSTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorHLSTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorHLSTest::CV_ColorHLSTest()
    : CV_ColorCvtBaseTest( "color-hls", "cvCvtColor", true, true, false )
{
    INIT_FWD_INV_CODES( BGR2HLS, HLS2BGR );
    depth_list = cvtcolor_depths_8_32;
    hue_channel = true;
}


void CV_ColorHLSTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2HLS, inv_code = CV_HLS2BGR;
    else
        fwd_code = CV_RGB2HLS, inv_code = CV_HLS2RGB;
}


double CV_ColorHLSTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? (j == 0 ? 4 : 16) : depth == CV_16U ? 32 : 1e-4;
}


void CV_ColorHLSTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float h_scale = depth == CV_8U ? 30.f : 60.f;
    float scale = depth == CV_8U ? 255.f : depth == CV_16U ? 65535.f : 1.f;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];
        float vmin = MIN(r,g);
        float v = MAX(r,g);
        float s, h, l, diff;
        vmin = MIN(vmin,b);
        v = MAX(v,b);
        diff = v - vmin;

        if( diff == 0 )
            s = h = 0, l = v;
        else
        {
            l = (v + vmin)*0.5f;
            s = l <= 0.5f ? diff / (v + vmin) : diff / (2 - v - vmin);
            diff = 1.f/diff;

            h = r == v ? (g - b)*diff :
                g == v ? 2 + (b - r)*diff : 4 + (r - g)*diff;

            if( h < 0 )
                h += 6;
        }

        dst_row[j] = h*h_scale;
        dst_row[j+1] = l*scale;
        dst_row[j+2] = s*scale;
    }
}


void CV_ColorHLSTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float h_scale = depth == CV_8U ? 1.f/30 : 1.f/60;
    float scale = depth == CV_8U ? 1.f/255 : depth == CV_16U ? 1.f/65535 : 1;
    int j;

    for( j = 0; j < n*3; j += 3 )
    {
        float h = src_row[j]*h_scale;
        float l = src_row[j+1]*scale;
        float s = src_row[j+2]*scale;
        float r = l, g = l, b = l;

        if( h < 0 )
            h += 6;
        else if( h >= 6 )
            h -= 6;

        if( s != 0 )
        {
            float m2 = l <= 0.5f ? l*(1.f + s) : l + s - l*s;
            float m1 = 2*l - m2;
            float h1 = h + 2;

            if( h1 >= 6 )
                h1 -= 6;
            if( h1 < 1 )
                r = m1 + (m2 - m1)*h1;
            else if( h1 < 3 )
                r = m2;
            else if( h1 < 4 )
                r = m1 + (m2 - m1)*(4 - h1);
            else
                r = m1;

            h1 = h;

            if( h1 < 1 )
                g = m1 + (m2 - m1)*h1;
            else if( h1 < 3 )
                g = m2;
            else if( h1 < 4 )
                g = m1 + (m2 - m1)*(4 - h1);
            else
                g = m1;

            h1 = h - 2;
            if( h1 < 0 )
                h1 += 6;

            if( h1 < 1 )
                b = m1 + (m2 - m1)*h1;
            else if( h1 < 3 )
                b = m2;
            else if( h1 < 4 )
                b = m1 + (m2 - m1)*(4 - h1);
            else
                b = m1;
        }

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


CV_ColorHLSTest color_hls_test;


static const double RGB2XYZ[] =
{
     0.412453, 0.357580, 0.180423,
     0.212671, 0.715160, 0.072169,
     0.019334, 0.119193, 0.950227
};


static const double XYZ2RGB[] =
{
    3.240479, -1.53715, -0.498535,
   -0.969256, 1.875991, 0.041556,
    0.055648, -0.204043, 1.057311
};

static const float Xn = 0.950456f;
static const float Zn = 1.088754f;


//// rgb <=> xyz
class CV_ColorXYZTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorXYZTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorXYZTest::CV_ColorXYZTest()
    : CV_ColorCvtBaseTest( "color-xyz", "cvCvtColor", true, true, true )
{
    INIT_FWD_INV_CODES( BGR2XYZ, XYZ2BGR );
}


void CV_ColorXYZTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_BGR2XYZ, inv_code = CV_XYZ2BGR;
    else
        fwd_code = CV_RGB2XYZ, inv_code = CV_XYZ2RGB;
}


double CV_ColorXYZTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? (j == 0 ? 2 : 8) : depth == CV_16U ? (j == 0 ? 64 : 128) : 1e-1;
}


void CV_ColorXYZTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    double scale = depth == CV_8U ? 255 : depth == CV_16U ? 65535 : 1;

    double M[9];
    int j;
    for( j = 0; j < 9; j++ )
        M[j] = RGB2XYZ[j]*scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double r = src_row[j+2];
        double g = src_row[j+1];
        double b = src_row[j];
        double x = M[0]*r + M[1]*g + M[2]*b;
        double y = M[3]*r + M[4]*g + M[5]*b;
        double z = M[6]*r + M[7]*g + M[8]*b;
        dst_row[j] = (float)x;
        dst_row[j+1] = (float)y;
        dst_row[j+2] = (float)z;
    }
}


void CV_ColorXYZTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    double scale = depth == CV_8U ? 1./255 : depth == CV_16U ? 1./65535 : 1;

    double M[9];
    int j;
    for( j = 0; j < 9; j++ )
        M[j] = XYZ2RGB[j]*scale;

    for( j = 0; j < n*3; j += 3 )
    {
        double x = src_row[j];
        double y = src_row[j+1];
        double z = src_row[j+2];
        double r = M[0]*x + M[1]*y + M[2]*z;
        double g = M[3]*x + M[4]*y + M[5]*z;
        double b = M[6]*x + M[7]*y + M[8]*z;
        dst_row[j] = (float)b;
        dst_row[j+1] = (float)g;
        dst_row[j+2] = (float)r;
    }
}


CV_ColorXYZTest color_xyz_test;


//// rgb <=> L*a*b*
class CV_ColorLabTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorLabTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorLabTest::CV_ColorLabTest()
    : CV_ColorCvtBaseTest( "color-lab", "cvCvtColor", true, true, false )
{
    INIT_FWD_INV_CODES( BGR2Lab, Lab2BGR );
    depth_list = cvtcolor_depths_8_32;
}


void CV_ColorLabTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_LBGR2Lab, inv_code = CV_Lab2LBGR;
    else
        fwd_code = CV_LRGB2Lab, inv_code = CV_Lab2LRGB;
}


double CV_ColorLabTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? 16 : depth == CV_16U ? 32 : 1e-3;
}


static const double _1_3 = 0.333333333333;

void CV_ColorLabTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    int j;
    float M[9];

    for( j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];

        float X = (r*M[0] + g*M[1] + b*M[2])*(1.f/Xn);
        float Y = r*M[3] + g*M[4] + b*M[5];
        float Z = (r*M[6] + g*M[7] + b*M[8])*(1.f/Zn);
        float fX, fY, fZ;

        float L, a;

        if( Y > 0.008856 )
        {
            fY = (float)pow((double)Y,_1_3);
            L = 116.f*fY - 16.f;
        }
        else
        {
            fY = 7.787f*Y + 16.f/116.f;
            L = 903.3f*Y;
        }

        if( X > 0.008856 )
            fX = (float)pow((double)X,_1_3);
        else
            fX = 7.787f*X + 16.f/116.f;

        if( Z > 0.008856 )
            fZ = (float)pow((double)Z,_1_3);
        else
            fZ = 7.787f*Z + 16.f/116.f;

        a = 500.f*(fX - fY);
        b = 200.f*(fY - fZ);

        dst_row[j] = L*Lscale;
        dst_row[j+1] = a + ab_bias;
        dst_row[j+2] = b + ab_bias;
    }
}


void CV_ColorLabTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float Lscale = depth == CV_8U ? 100.f/255.f : depth == CV_16U ? 100.f/65535.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    int j;
    float M[9];

    for( j = 0; j < 9; j++ )
        M[j] = (float)XYZ2RGB[j];

    for( j = 0; j < n*3; j += 3 )
    {
        float L = src_row[j]*Lscale;
        float a = src_row[j+1] - ab_bias;
        float b = src_row[j+2] - ab_bias;

        float P = (L + 16.f)*(1.f/116.f);
        float X = (P + a*0.002f);
        float Z = (P - b*0.005f);
        float Y = P*P*P;
        X = Xn*X*X*X;
        Z = Zn*Z*Z*Z;

        float r = M[0]*X + M[1]*Y + M[2]*Z;
        float g = M[3]*X + M[4]*Y + M[5]*Z;
        b = M[6]*X + M[7]*Y + M[8]*Z;

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


CV_ColorLabTest color_lab_test;


//// rgb <=> L*u*v*
class CV_ColorLuvTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorLuvTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorLuvTest::CV_ColorLuvTest()
    : CV_ColorCvtBaseTest( "color-luv", "cvCvtColor", true, true, false )
{
    INIT_FWD_INV_CODES( BGR2Luv, Luv2BGR );
    depth_list = cvtcolor_depths_8_32;
}


void CV_ColorLuvTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = CV_LBGR2Luv, inv_code = CV_Luv2LBGR;
    else
        fwd_code = CV_LRGB2Luv, inv_code = CV_Luv2LRGB;
}


double CV_ColorLuvTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = CV_MAT_DEPTH(test_mat[i][j].type);
    return depth == CV_8U ? 48 : depth == CV_16U ? 32 : 1e-2;
}


void CV_ColorLuvTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    int j;

    float M[9];
    float un = 4.f*Xn/(Xn + 15.f*1.f + 3*Zn);
    float vn = 9.f*1.f/(Xn + 15.f*1.f + 3*Zn);
    float u_scale = 1.f, u_bias = 0.f;
    float v_scale = 1.f, v_bias = 0.f;

    for( j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    if( depth == CV_8U )
    {
        u_scale = 0.720338983f;
        u_bias = 96.5254237f;
        v_scale = 0.99609375f;
        v_bias = 139.453125f;
    }

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];

        float X = r*M[0] + g*M[1] + b*M[2];
        float Y = r*M[3] + g*M[4] + b*M[5];
        float Z = r*M[6] + g*M[7] + b*M[8];
        float d = X + 15*Y + 3*Z, L, u, v;

        if( d == 0 )
            L = u = v = 0;
        else
        {
            if( Y > 0.008856f )
                L = (float)(116.*pow((double)Y,_1_3) - 16.);
            else
                L = 903.3f * Y;

            d = 1.f/d;
            u = 13*L*(4*X*d - un);
            v = 13*L*(9*Y*d - vn);
        }
        dst_row[j] = L*Lscale;
        dst_row[j+1] = u*u_scale + u_bias;
        dst_row[j+2] = v*v_scale + v_bias;
    }
}


void CV_ColorLuvTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = CV_MAT_DEPTH(test_mat[INPUT][0].type);
    float Lscale = depth == CV_8U ? 100.f/255.f : depth == CV_16U ? 100.f/65535.f : 1.f;
    int j;
    float M[9];
    float un = 4.f*Xn/(Xn + 15.f*1.f + 3*Zn);
    float vn = 9.f*1.f/(Xn + 15.f*1.f + 3*Zn);
    float u_scale = 1.f, u_bias = 0.f;
    float v_scale = 1.f, v_bias = 0.f;

    for( j = 0; j < 9; j++ )
        M[j] = (float)XYZ2RGB[j];

    if( depth == CV_8U )
    {
        u_scale = 1.f/0.720338983f;
        u_bias = 96.5254237f;
        v_scale = 1.f/0.99609375f;
        v_bias = 139.453125f;
    }

    for( j = 0; j < n*3; j += 3 )
    {
        float L = src_row[j]*Lscale;
        float u = (src_row[j+1] - u_bias)*u_scale;
        float v = (src_row[j+2] - v_bias)*v_scale;
        float X, Y, Z;

        if( L >= 8 )
        {
            Y = (L + 16.f)*(1.f/116.f);
            Y = Y*Y*Y;
        }
        else
        {
            Y = L * (1.f/903.3f);
            if( L == 0 )
                L = 0.001f;
        }

        u = u/(13*L) + un;
        v = v/(13*L) + vn;

        X = -9*Y*u/((u - 4)*v - u*v);
        Z = (9*Y - 15*v*Y - v*X)/(3*v);

        float r = M[0]*X + M[1]*Y + M[2]*Z;
        float g = M[3]*X + M[4]*Y + M[5]*Z;
        float b = M[6]*X + M[7]*Y + M[8]*Z;

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


CV_ColorLuvTest color_luv_test;



//// rgb <=> another rgb
class CV_ColorRGBTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorRGBTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_forward( const CvMat* src, CvMat* dst );
    void convert_backward( const CvMat* src, const CvMat* dst, CvMat* dst2 );
    int dst_bits;
};


CV_ColorRGBTest::CV_ColorRGBTest()
    : CV_ColorCvtBaseTest( "color-rgb", "cvCvtColor", true, true, true )
{
    dst_bits = 0;
    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
}


void CV_ColorRGBTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = CV_MAT_CN(types[INPUT][0]);

    dst_bits = 24;

    if( cvTsRandInt(rng) % 3 == 0 )
    {
        types[INPUT][0] = types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(CV_8U,cn);
        types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(CV_8U,2);
        if( cvTsRandInt(rng) & 1 )
        {
            if( blue_idx == 0 )
                fwd_code = CV_BGR2BGR565, inv_code = CV_BGR5652BGR;
            else
                fwd_code = CV_RGB2BGR565, inv_code = CV_BGR5652RGB;
            dst_bits = 16;
        }
        else
        {
            if( blue_idx == 0 )
                fwd_code = CV_BGR2BGR555, inv_code = CV_BGR5552BGR;
            else
                fwd_code = CV_RGB2BGR555, inv_code = CV_BGR5552RGB;
            dst_bits = 15;
        }
    }
    else
    {
        if( cn == 3 )
        {
            fwd_code = CV_RGB2BGR, inv_code = CV_BGR2RGB;
            blue_idx = 2;
        }
        else if( blue_idx == 0 )
            fwd_code = CV_BGRA2BGR, inv_code = CV_BGR2BGRA;
        else
            fwd_code = CV_RGBA2BGR, inv_code = CV_BGR2RGBA;
    }

    if( CV_MAT_CN(types[INPUT][0]) != CV_MAT_CN(types[OUTPUT][0]) )
        inplace = false;
}


double CV_ColorRGBTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_ColorRGBTest::convert_forward( const CvMat* src, CvMat* dst )
{
    int depth = CV_MAT_DEPTH(src->type);
    int cn = CV_MAT_CN(src->type);
/*#if defined _DEBUG || defined DEBUG
    int dst_cn = CV_MAT_CN(dst->type);
#endif*/
    int i, j, cols = src->cols;
    int g_rshift = dst_bits == 16 ? 2 : 3;
    int r_lshift = dst_bits == 16 ? 11 : 10;

    //assert( (cn == 3 || cn == 4) && (dst_cn == 3 || (dst_cn == 2 && depth == CV_8U)) );

    for( i = 0; i < src->rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = (const uchar*)(src->data.ptr + i*src->step);
                uchar* dst_row = (uchar*)(dst->data.ptr + i*dst->step);

                if( dst_bits == 24 )
                {
                    for( j = 0; j < cols; j++ )
                    {
                        uchar b = src_row[j*cn + blue_idx];
                        uchar g = src_row[j*cn + 1];
                        uchar r = src_row[j*cn + (blue_idx^2)];
                        dst_row[j*3] = b;
                        dst_row[j*3+1] = g;
                        dst_row[j*3+2] = r;
                    }
                }
                else
                {
                    for( j = 0; j < cols; j++ )
                    {
                        int b = src_row[j*cn + blue_idx] >> 3;
                        int g = src_row[j*cn + 1] >> g_rshift;
                        int r = src_row[j*cn + (blue_idx^2)] >> 3;
                        ((ushort*)dst_row)[j] = (ushort)(b | (g << 5) | (r << r_lshift));
                        if( cn == 4 && src_row[j*4+3] )
                            ((ushort*)dst_row)[j] |= 1 << (r_lshift+5);
                    }
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = (const ushort*)(src->data.ptr + i*src->step);
                ushort* dst_row = (ushort*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    ushort b = src_row[j*cn + blue_idx];
                    ushort g = src_row[j*cn + 1];
                    ushort r = src_row[j*cn + (blue_idx^2)];
                    dst_row[j*3] = b;
                    dst_row[j*3+1] = g;
                    dst_row[j*3+2] = r;
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = (const float*)(src->data.ptr + i*src->step);
                float* dst_row = (float*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    float b = src_row[j*cn + blue_idx];
                    float g = src_row[j*cn + 1];
                    float r = src_row[j*cn + (blue_idx^2)];
                    dst_row[j*3] = b;
                    dst_row[j*3+1] = g;
                    dst_row[j*3+2] = r;
                }
            }
            break;
        default:
            assert(0);
        }
    }
}


void CV_ColorRGBTest::convert_backward( const CvMat* /*src*/, const CvMat* src, CvMat* dst )
{
    int depth = CV_MAT_DEPTH(src->type);
    int cn = CV_MAT_CN(dst->type);
/*#if defined _DEBUG || defined DEBUG
    int src_cn = CV_MAT_CN(src->type);
#endif*/
    int i, j, cols = src->cols;
    int g_lshift = dst_bits == 16 ? 2 : 3;
    int r_rshift = dst_bits == 16 ? 11 : 10;

    //assert( (cn == 3 || cn == 4) && (src_cn == 3 || (src_cn == 2 && depth == CV_8U)) );

    for( i = 0; i < src->rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = (const uchar*)(src->data.ptr + i*src->step);
                uchar* dst_row = (uchar*)(dst->data.ptr + i*dst->step);

                if( dst_bits == 24 )
                {
                    for( j = 0; j < cols; j++ )
                    {
                        uchar b = src_row[j*3];
                        uchar g = src_row[j*3 + 1];
                        uchar r = src_row[j*3 + 2];

                        dst_row[j*cn + blue_idx] = b;
                        dst_row[j*cn + 1] = g;
                        dst_row[j*cn + (blue_idx^2)] = r;

                        if( cn == 4 )
                            dst_row[j*cn + 3] = 255;
                    }
                }
                else
                {
                    for( j = 0; j < cols; j++ )
                    {
                        ushort val = ((ushort*)src_row)[j];
                        uchar b = (uchar)(val << 3);
                        uchar g = (uchar)((val >> 5) << g_lshift);
                        uchar r = (uchar)((val >> r_rshift) << 3);

                        dst_row[j*cn + blue_idx] = b;
                        dst_row[j*cn + 1] = g;
                        dst_row[j*cn + (blue_idx^2)] = r;

                        if( cn == 4 )
                        {
                            uchar alpha = r_rshift == 11 || (val & 0x8000) != 0 ? 255 : 0;
                            dst_row[j*cn + 3] = alpha;
                        }
                    }
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = (const ushort*)(src->data.ptr + i*src->step);
                ushort* dst_row = (ushort*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    ushort b = src_row[j*3];
                    ushort g = src_row[j*3 + 1];
                    ushort r = src_row[j*3 + 2];

                    dst_row[j*cn + blue_idx] = b;
                    dst_row[j*cn + 1] = g;
                    dst_row[j*cn + (blue_idx^2)] = r;

                    if( cn == 4 )
                        dst_row[j*cn + 3] = 65535;
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = (const float*)(src->data.ptr + i*src->step);
                float* dst_row = (float*)(dst->data.ptr + i*dst->step);

                for( j = 0; j < cols; j++ )
                {
                    float b = src_row[j*3];
                    float g = src_row[j*3 + 1];
                    float r = src_row[j*3 + 2];

                    dst_row[j*cn + blue_idx] = b;
                    dst_row[j*cn + 1] = g;
                    dst_row[j*cn + (blue_idx^2)] = r;

                    if( cn == 4 )
                        dst_row[j*cn + 3] = 1.f;
                }
            }
            break;
        default:
            assert(0);
        }
    }
}


CV_ColorRGBTest color_rgb_test;


//// rgb <=> bayer

static const char* cvtcolor_bayer_param_names[] = { "size", "depth", 0 };

class CV_ColorBayerTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorBayerTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    void get_timing_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types,
                                                CvSize** whole_sizes, bool *are_images );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CV_ColorBayerTest::CV_ColorBayerTest()
    : CV_ColorCvtBaseTest( "color-bayer", "cvCvtColor", false, false, false )
{
    test_array[OUTPUT].pop();
    test_array[REF_OUTPUT].pop();

    fwd_code_str = "BayerBG2BGR";
    inv_code_str = "";
    fwd_code = CV_BayerBG2BGR;
    inv_code = -1;

    default_timing_param_names = cvtcolor_bayer_param_names;
    depth_list = cvtcolor_depths_8;
}


void CV_ColorBayerTest::get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[INPUT][0] = CV_8UC1;
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8UC3;
    inplace = false;

    fwd_code = cvTsRandInt(rng)%4 + CV_BayerBG2BGR;
}


void CV_ColorBayerTest::get_timing_test_array_types_and_sizes( int test_case_idx,
                    CvSize** sizes, int** types, CvSize** whole_sizes, bool *are_images )
{
    CV_ColorCvtBaseTest::get_timing_test_array_types_and_sizes( test_case_idx, sizes, types,
                                                                whole_sizes, are_images );
    types[INPUT][0] &= CV_MAT_DEPTH_MASK;
}


double CV_ColorBayerTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1;
}


void CV_ColorBayerTest::run_func()
{
    if(!test_cpp)
        cvCvtColor( test_array[INPUT][0], test_array[OUTPUT][0], fwd_code );
    else
    {
        cv::Mat _out = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::cvtColor(cv::cvarrToMat(test_array[INPUT][0]), _out, fwd_code, _out.channels());
    }
}


void CV_ColorBayerTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const CvMat* src = &test_mat[INPUT][0];
    CvMat* dst = &test_mat[REF_OUTPUT][0];
    int i, j, cols = src->cols - 2;
    int code = fwd_code;
    int bi = 0;
    int step = src->step;

    memset( dst->data.ptr, 0, dst->cols*3 );
    memset( dst->data.ptr + (dst->rows-1)*dst->step, 0, dst->cols*3 );

    if( fwd_code == CV_BayerRG2BGR || fwd_code == CV_BayerGR2BGR )
        bi ^= 2;

    for( i = 1; i < src->rows - 1; i++ )
    {
        const uchar* ptr = src->data.ptr + i*step + 1;
        uchar* dst_row = dst->data.ptr + i*dst->step + 3;
        int save_code = code;
        dst_row[-3] = dst_row[-2] = dst_row[-1] = 0;
        dst_row[cols*3] = dst_row[cols*3+1] = dst_row[cols*3+2] = 0;

        for( j = 0; j < cols; j++ )
        {
            int b, g, r;
            if( !(code & 1) )
            {
                b = ptr[j];
                g = (ptr[j-1] + ptr[j+1] + ptr[j-step] + ptr[j+step])>>2;
                r = (ptr[j-step-1] + ptr[j-step+1] + ptr[j+step-1] + ptr[j+step+1]) >> 2;
            }
            else
            {
                b = (ptr[j-1] + ptr[j+1]) >> 1;
                g = ptr[j];
                r = (ptr[j-step] + ptr[j+step]) >> 1;
            }
            code ^= 1;
            dst_row[j*3 + bi] = (uchar)b;
            dst_row[j*3 + 1] = (uchar)g;
            dst_row[j*3 + (bi^2)] = (uchar)r;
        }
        code = save_code ^ 1;
        bi ^= 2;
    }
}

CV_ColorBayerTest color_bayer_test;
