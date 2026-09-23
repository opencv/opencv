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

namespace opencv_test { namespace {

/////////////////////////// base test class for color transformations /////////////////////////

class CV_ColorCvtBaseTest : public cvtest::ArrayTest
{
public:
    CV_ColorCvtBaseTest( bool custom_inv_transform, bool allow_32f, bool allow_16u );

protected:
    int prepare_test_case( int test_case_idx );
    void prepare_to_validation( int /*test_case_idx*/ );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high );

    // input --- fwd_transform -> ref_output[0]
    virtual void convert_forward( const Mat& src, Mat& dst );
    // ref_output[0] --- inv_transform ---> ref_output[1] (or input -- copy --> ref_output[1])
    virtual void convert_backward( const Mat& src, const Mat& dst, Mat& dst2 );

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
    int hue_range;
    bool srgb;
};


CV_ColorCvtBaseTest::CV_ColorCvtBaseTest( bool _custom_inv_transform, bool _allow_32f, bool _allow_16u )
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    allow_16u = _allow_16u;
    allow_32f = _allow_32f;
    custom_inv_transform = _custom_inv_transform;
    fwd_code = inv_code = -1;
    element_wise_relative_error = false;

    fwd_code_str = inv_code_str = 0;

    hue_range = 0;
    blue_idx = 0;
    srgb = false;
    inplace = false;
}


void CV_ColorCvtBaseTest::get_minmax_bounds( int i, int j, int type, Scalar& low, Scalar& high )
{
    cvtest::ArrayTest::get_minmax_bounds( i, j, type, low, high );
    if( i == INPUT )
    {
        int depth = CV_MAT_DEPTH(type);
        low = Scalar::all(0.);
        high = Scalar::all( depth == CV_8U ? 256 : depth == CV_16U ? 65536 : 1. );
    }
}


void CV_ColorCvtBaseTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth, cn;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( allow_16u && allow_32f )
    {
        depth = cvtest::randInt(rng) % 3;
        depth = depth == 0 ? CV_8U : depth == 1 ? CV_16U : CV_32F;
    }
    else if( allow_16u || allow_32f )
    {
        depth = cvtest::randInt(rng) % 2;
        depth = depth == 0 ? CV_8U : allow_16u ? CV_16U : CV_32F;
    }
    else
        depth = CV_8U;

    cn = (cvtest::randInt(rng) & 1) + 3;
    blue_idx = cvtest::randInt(rng) & 1 ? 2 : 0;
    srgb = (cvtest::randInt(rng) & 1) != 0;

    types[INPUT][0] = CV_MAKETYPE(depth, cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, 3);
    if( test_array[OUTPUT].size() > 1 )
        types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(depth, cn);

    inplace = cn == 3 && cvtest::randInt(rng) % 2 != 0;
}


int CV_ColorCvtBaseTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 && inplace )
        cvtest::copy( test_mat[INPUT][0], test_mat[OUTPUT][0] );
    return code;
}

void CV_ColorCvtBaseTest::run_func()
{
    cv::Mat out0 = test_mat[OUTPUT][0];
    cv::Mat _out0 = out0, _out1 = test_mat[OUTPUT][1];

    cv::cvtColor( inplace ? out0 : test_mat[INPUT][0], _out0, fwd_code, _out0.channels());

    if( inplace )
    {
        out0.copyTo(test_mat[OUTPUT][1]);
        out0 = test_mat[OUTPUT][1];
    }
    cv::cvtColor(out0, _out1, inv_code, _out1.channels());
}


void CV_ColorCvtBaseTest::prepare_to_validation( int /*test_case_idx*/ )
{
    convert_forward( test_mat[INPUT][0], test_mat[REF_OUTPUT][0] );
    convert_backward( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                      test_mat[REF_OUTPUT][1] );
    int depth = test_mat[REF_OUTPUT][0].depth();
    if( depth == CV_8U && hue_range )
    {
        for( int y = 0; y < test_mat[REF_OUTPUT][0].rows; y++ )
        {
            uchar* h0 = test_mat[REF_OUTPUT][0].ptr(y);
            uchar* h = test_mat[OUTPUT][0].ptr(y);

            for( int x = 0; x < test_mat[REF_OUTPUT][0].cols; x++, h0 += 3, h += 3 )
            {
                if( abs(*h - *h0) >= hue_range-1 && (*h <= 1 || *h0 <= 1) )
                    *h = *h0 = 0;
            }
        }
    }
}


void CV_ColorCvtBaseTest::convert_forward( const Mat& src, Mat& dst )
{
    const float c8u = 0.0039215686274509803f; // 1./255
    const float c16u = 1.5259021896696422e-005f; // 1./65535
    int depth = src.depth();
    int cn = src.channels(), dst_cn = dst.channels();
    int cols = src.cols, dst_cols_n = dst.cols*dst_cn;
    vector<float> _src_buf(src.cols*3);
    vector<float> _dst_buf(dst.cols*3);
    float* src_buf = &_src_buf[0];
    float* dst_buf = &_dst_buf[0];
    int i, j;

    CV_Assert( (cn == 3 || cn == 4) && (dst_cn == 3 || dst_cn == 1) );

    for( i = 0; i < src.rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = src.ptr(i);
                uchar* dst_row = dst.ptr(i);

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
                    dst_row[j] = saturate_cast<uchar>(t);
                }
            }
            break;
        case CV_16U:
            {
                const ushort* src_row = src.ptr<ushort>(i);
                ushort* dst_row = dst.ptr<ushort>(i);

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
                    dst_row[j] = saturate_cast<ushort>(t);
                }
            }
            break;
        case CV_32F:
            {
                const float* src_row = src.ptr<float>(i);
                float* dst_row = dst.ptr<float>(i);

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
            CV_Assert(0);
        }
    }
}


void CV_ColorCvtBaseTest::convert_row_bgr2abc_32f_c3( const float* /*src_row*/,
                                                      float* /*dst_row*/, int /*n*/ )
{
}


void CV_ColorCvtBaseTest::convert_row_abc2bgr_32f_c3( const float* /*src_row*/,
                                                      float* /*dst_row*/, int /*n*/ )
{
}


void CV_ColorCvtBaseTest::convert_backward( const Mat& src, const Mat& dst, Mat& dst2 )
{
    if( custom_inv_transform )
    {
        int depth = src.depth();
        int src_cn = dst.channels(), cn = dst2.channels();
        int cols_n = src.cols*src_cn, dst_cols = dst.cols;
        vector<float> _src_buf(src.cols*3);
        vector<float> _dst_buf(dst.cols*3);
        float* src_buf = &_src_buf[0];
        float* dst_buf = &_dst_buf[0];
        int i, j;

        CV_Assert( cn == 3 || cn == 4 );

        for( i = 0; i < src.rows; i++ )
        {
            switch( depth )
            {
            case CV_8U:
                {
                    const uchar* src_row = dst.ptr(i);
                    uchar* dst_row = dst2.ptr(i);

                    for( j = 0; j < cols_n; j++ )
                        src_buf[j] = src_row[j];

                    convert_row_abc2bgr_32f_c3( src_buf, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        int b = cvRound( dst_buf[j*3]*255. );
                        int g = cvRound( dst_buf[j*3+1]*255. );
                        int r = cvRound( dst_buf[j*3+2]*255. );
                        dst_row[j*cn + blue_idx] = saturate_cast<uchar>(b);
                        dst_row[j*cn + 1] = saturate_cast<uchar>(g);
                        dst_row[j*cn + (blue_idx^2)] = saturate_cast<uchar>(r);
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 255;
                    }
                }
                break;
            case CV_16U:
                {
                    const ushort* src_row = dst.ptr<ushort>(i);
                    ushort* dst_row = dst2.ptr<ushort>(i);

                    for( j = 0; j < cols_n; j++ )
                        src_buf[j] = src_row[j];

                    convert_row_abc2bgr_32f_c3( src_buf, dst_buf, dst_cols );

                    for( j = 0; j < dst_cols; j++ )
                    {
                        int b = cvRound( dst_buf[j*3]*65535. );
                        int g = cvRound( dst_buf[j*3+1]*65535. );
                        int r = cvRound( dst_buf[j*3+2]*65535. );
                        dst_row[j*cn + blue_idx] = saturate_cast<ushort>(b);
                        dst_row[j*cn + 1] = saturate_cast<ushort>(g);
                        dst_row[j*cn + (blue_idx^2)] = saturate_cast<ushort>(r);
                        if( cn == 4 )
                            dst_row[j*cn + 3] = 65535;
                    }
                }
                break;
            case CV_32F:
                {
                    const float* src_row = dst.ptr<float>(i);
                    float* dst_row = dst2.ptr<float>(i);

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
                CV_Assert(0);
            }
        }
    }
    else
    {
        int i, j, k;
        int elem_size = (int)src.elemSize(), elem_size1 = (int)src.elemSize1();
        int width_n = src.cols*elem_size;

        for( i = 0; i < src.rows; i++ )
        {
            memcpy( dst2.ptr(i), src.ptr(i), width_n );
            if( src.channels() == 4 )
            {
                // clear the alpha channel
                uchar* ptr = dst2.ptr(i) + elem_size1*3;
                for( j = 0; j < width_n; j += elem_size )
                {
                    for( k = 0; k < elem_size1; k++ )
                        ptr[j + k] = 0;
                }
            }
        }
    }
}


#undef INIT_FWD_INV_CODES
#define INIT_FWD_INV_CODES( fwd, inv )          \
    fwd_code = COLOR_##fwd; inv_code = COLOR_##inv;   \
    fwd_code_str = #fwd; inv_code_str = #inv

//// rgb <=> gray
class CV_ColorGrayTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorGrayTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
    double get_success_error_level( int test_case_idx, int i, int j );
};


CV_ColorGrayTest::CV_ColorGrayTest() : CV_ColorCvtBaseTest( true, true, true )
{
    INIT_FWD_INV_CODES( BGR2GRAY, GRAY2BGR );
}


void CV_ColorGrayTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = CV_MAT_CN(types[INPUT][0]);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = types[INPUT][0] & CV_MAT_DEPTH_MASK;
    inplace = false;

    if( cn == 3 )
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_BGR2GRAY, inv_code = COLOR_GRAY2BGR;
        else
            fwd_code = COLOR_RGB2GRAY, inv_code = COLOR_GRAY2RGB;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_BGRA2GRAY, inv_code = COLOR_GRAY2BGRA;
        else
            fwd_code = COLOR_RGBA2GRAY, inv_code = COLOR_GRAY2RGBA;
    }
}


double CV_ColorGrayTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 1 : depth == CV_16U ? 2 : 1e-5;
}


void CV_ColorGrayTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
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
    int j, depth = test_mat[INPUT][0].depth();
    float scale = depth == CV_8U ? (1.f/255) : depth == CV_16U ? 1.f/65535 : 1.f;
    for( j = 0; j < n; j++ )
        dst_row[j*3] = dst_row[j*3+1] = dst_row[j*3+2] = src_row[j]*scale;
}


//// rgb <=> ycrcb
class CV_ColorYCrCbTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorYCrCbTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorYCrCbTest::CV_ColorYCrCbTest() : CV_ColorCvtBaseTest( true, true, true )
{
    INIT_FWD_INV_CODES( BGR2YCrCb, YCrCb2BGR );
}


void CV_ColorYCrCbTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = COLOR_BGR2YCrCb, inv_code = COLOR_YCrCb2BGR;
    else
        fwd_code = COLOR_RGB2YCrCb, inv_code = COLOR_YCrCb2RGB;
}


double CV_ColorYCrCbTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 2 : depth == CV_16U ? 32 : 1e-3;
}


void CV_ColorYCrCbTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
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
    int depth = test_mat[INPUT][0].depth();
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


//// rgb <=> hsv
class CV_ColorHSVTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorHSVTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorHSVTest::CV_ColorHSVTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2HSV, HSV2BGR );
    hue_range = 180;
}


void CV_ColorHSVTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    RNG& rng = ts->get_rng();

    bool full_hrange = (rng.next() & 256) != 0;
    if( full_hrange )
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_BGR2HSV_FULL, inv_code = COLOR_HSV2BGR_FULL;
        else
            fwd_code = COLOR_RGB2HSV_FULL, inv_code = COLOR_HSV2RGB_FULL;
        hue_range = 256;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_BGR2HSV, inv_code = COLOR_HSV2BGR;
        else
            fwd_code = COLOR_RGB2HSV, inv_code = COLOR_HSV2RGB;
        hue_range = 180;
    }
}


double CV_ColorHSVTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 4 : 16) : depth == CV_16U ? 32 : 1e-3;
}


void CV_ColorHSVTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float h_scale = depth == CV_8U ? hue_range*30.f/180 : 60.f;
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
    int depth = test_mat[INPUT][0].depth();
    float h_scale = depth == CV_8U ? 180/(hue_range*30.f) : 1.f/60;
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


//// rgb <=> hls
class CV_ColorHLSTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorHLSTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorHLSTest::CV_ColorHLSTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2HLS, HLS2BGR );
    hue_range = 180;
}


void CV_ColorHLSTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = COLOR_BGR2HLS, inv_code = COLOR_HLS2BGR;
    else
        fwd_code = COLOR_RGB2HLS, inv_code = COLOR_HLS2RGB;
}


double CV_ColorHLSTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 4 : 16) : depth == CV_16U ? 32 : 1e-4;
}


void CV_ColorHLSTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
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
    int depth = test_mat[INPUT][0].depth();
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

// 0.412453, 0.357580, 0.180423,
// 0.212671, 0.715160, 0.072169,
// 0.019334, 0.119193, 0.950227
static const softdouble RGB2XYZ[] =
{
    softdouble::fromRaw(0x3fda65a14488c60d),
    softdouble::fromRaw(0x3fd6e297396d0918),
    softdouble::fromRaw(0x3fc71819d2391d58),
    softdouble::fromRaw(0x3fcb38cda6e75ff6),
    softdouble::fromRaw(0x3fe6e297396d0918),
    softdouble::fromRaw(0x3fb279aae6c8f755),
    softdouble::fromRaw(0x3f93cc4ac6cdaf4b),
    softdouble::fromRaw(0x3fbe836eb4e98138),
    softdouble::fromRaw(0x3fee68427418d691)
};

//  3.240479, -1.53715, -0.498535,
// -0.969256, 1.875991, 0.041556,
//  0.055648, -0.204043, 1.057311
static const softdouble XYZ2RGB[] =
{
    softdouble::fromRaw(0x4009ec804102ff8f),
    softdouble::fromRaw(0xbff8982a9930be0e),
    softdouble::fromRaw(0xbfdfe7ff583a53b9),
    softdouble::fromRaw(0xbfef042528ae74f3),
    softdouble::fromRaw(0x3ffe040f23897204),
    softdouble::fromRaw(0x3fa546d3f9e7b80b),
    softdouble::fromRaw(0x3fac7de5082cf52c),
    softdouble::fromRaw(0xbfca1e14bdfd2631),
    softdouble::fromRaw(0x3ff0eabef06b3786)
};

//0.950456
static const softdouble Xn = softdouble::fromRaw(0x3fee6a22b3892ee8);
//1.088754
static const softdouble Zn = softdouble::fromRaw(0x3ff16b8950763a19);


//// rgb <=> xyz
class CV_ColorXYZTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorXYZTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorXYZTest::CV_ColorXYZTest() : CV_ColorCvtBaseTest( true, true, true )
{
    INIT_FWD_INV_CODES( BGR2XYZ, XYZ2BGR );
}


void CV_ColorXYZTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if( blue_idx == 0 )
        fwd_code = COLOR_BGR2XYZ, inv_code = COLOR_XYZ2BGR;
    else
        fwd_code = COLOR_RGB2XYZ, inv_code = COLOR_XYZ2RGB;
}


double CV_ColorXYZTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 2 : 8) : depth == CV_16U ? (j == 0 ? 64 : 128) : 1e-1;
}


void CV_ColorXYZTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    softdouble scale(depth == CV_8U  ? 255 :
                     depth == CV_16U ? 65535 : 1);

    double M[9];
    int j;
    for( j = 0; j < 9; j++ )
        M[j] = (double)(RGB2XYZ[j]*scale);

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
    int depth = test_mat[INPUT][0].depth();
    softdouble scale(depth == CV_8U  ? 1./255 :
                     depth == CV_16U ? 1./65535 : 1);

    double M[9];
    int j;
    for( j = 0; j < 9; j++ )
        M[j] = (double)(XYZ2RGB[j]*scale);

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


//// rgb <=> L*a*b*

//taken from color.cpp

//all constants should be presented through integers to keep bit-exactness
static const softdouble gammaThreshold    = softdouble(809)/softdouble(20000);    //  0.04045
static const softdouble gammaInvThreshold = softdouble(7827)/softdouble(2500000); //  0.0031308
static const softdouble gammaLowScale     = softdouble(323)/softdouble(25);       // 12.92
static const softdouble gammaPower        = softdouble(12)/softdouble(5);         //  2.4
static const softdouble gammaXshift       = softdouble(11)/softdouble(200);       // 0.055

static inline softfloat applyGamma(softfloat x)
{
    //return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
    softdouble xd = x;
    return (xd <= gammaThreshold ?
                xd/gammaLowScale :
                pow((xd + gammaXshift)/(softdouble::one()+gammaXshift), gammaPower));
}

static inline softfloat applyInvGamma(softfloat x)
{
    //return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
    softdouble xd = x;
    return (xd <= gammaInvThreshold ?
                xd*gammaLowScale :
                pow(xd, softdouble::one()/gammaPower)*(softdouble::one()+gammaXshift) - gammaXshift);
}

static inline float applyGamma(float x)
{
    return x <= 0.04045f ? x*(1.f/12.92f) : (float)std::pow((double)(x + 0.055)*(1./1.055), 2.4);
}

static inline float applyInvGamma(float x)
{
    return x <= 0.0031308 ? x*12.92f : (float)(1.055*std::pow((double)x, 1./2.4) - 0.055);
}

class CV_ColorLabTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorLabTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorLabTest::CV_ColorLabTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2Lab, Lab2BGR );
}


void CV_ColorLabTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if(srgb)
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_BGR2Lab, inv_code = COLOR_Lab2BGR;
        else
            fwd_code = COLOR_RGB2Lab, inv_code = COLOR_Lab2RGB;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_LBGR2Lab, inv_code = COLOR_Lab2LBGR;
        else
            fwd_code = COLOR_LRGB2Lab, inv_code = COLOR_Lab2LRGB;
    }
}


double CV_ColorLabTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    // j == 0 is for forward code, j == 1 is for inverse code
    return (depth ==  CV_8U) ? (srgb ? 37 : 8) :
           //(depth == CV_16U) ? 32 : // 16u is disabled
           srgb ? ((j == 0) ? 0.4 : 0.0055) : 1e-3;
}


void CV_ColorLabTest::convert_row_bgr2abc_32f_c3(const float* src_row, float* dst_row, int n)
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    float M[9];

    // 7.787f = (29/3)^3/(29*4)
    static const float lowScale = 29.f*29.f/(27.f*4.f);
    // 0.008856f = (6/29)^3
    static const float lthresh = 6.f*6.f*6.f/(29.f*29.f*29.f);
    // 903.3 = (29/3)^3
    static const float yscale = 29.f*29.f*29.f/27.f;

    static const float f16of116 = 16.f/116.f;

    for (int j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    float xn = (float)Xn, zn = (float)Zn;
    for (int x = 0; x < n*3; x += 3)
    {
        float R = src_row[x + 2];
        float G = src_row[x + 1];
        float B = src_row[x];

        R = std::min(std::max(R, 0.f), 1.f);
        G = std::min(std::max(G, 0.f), 1.f);
        B = std::min(std::max(B, 0.f), 1.f);
        if (srgb)
        {
            R = applyGamma(R);
            G = applyGamma(G);
            B = applyGamma(B);
        }

        float X = (R * M[0] + G * M[1] + B * M[2]) / xn;
        float Y =  R * M[3] + G * M[4] + B * M[5];
        float Z = (R * M[6] + G * M[7] + B * M[8]) / zn;

        float fX = X > lthresh ? cubeRoot(X) : (lowScale * X + f16of116);
        float fY = Y > lthresh ? cubeRoot(Y) : (lowScale * Y + f16of116);
        float fZ = Z > lthresh ? cubeRoot(Z) : (lowScale * Z + f16of116);

        float L = Y > lthresh ? (116.f*fY - 16.f) : (yscale*Y);
        float a = 500.f * (fX - fY);
        float b = 200.f * (fY - fZ);

        dst_row[x] = L * Lscale;
        dst_row[x + 1] = a + ab_bias;
        dst_row[x + 2] = b + ab_bias;
    }
}


void CV_ColorLabTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 100.f/255.f : depth == CV_16U ? 100.f/65535.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    float M[9];

    for(int j = 0; j < 9; j++ )
        M[j] = (float)XYZ2RGB[j];

    // 0.008856f * 903.3f  = (6/29)^3*(29/3)^3 = 8
    static const float lThresh = 8.f;
    // 7.787f * 0.008856f + 16.0f / 116.0f = 6/29
    static const float fThresh = 6.f/29.f;
    static const float lbias = 16.f/116.f;
    // 7.787f = (29/3)^3/(29*4)
    static const float lowScale = 29.f*29.f/(27.f*4.f);
    // 903.3 = (29/3)^3
    static const float yscale = 29.f*29.f*29.f/27.f;

    float xn = (float)Xn, zn = (float)Zn;
    for (int x = 0, end = n * 3; x < end; x += 3)
    {
        float L = src_row[x] * Lscale;
        float a = src_row[x + 1] - ab_bias;
        float b = src_row[x + 2] - ab_bias;

        float FY = 0.0f, Y = 0.0f;
        if (L <= lThresh)
        {
            Y = L / yscale;
            FY = lowScale * Y + lbias;
        }
        else
        {
            FY = (L + 16.0f) / 116.0f;
            Y = FY * FY * FY;
        }

        float FX = a / 500.0f + FY;
        float FZ = FY - b / 200.0f;

        float FXZ[] = { FX, FZ };
        for (int k = 0; k < 2; ++k)
        {
            if (FXZ[k] <= fThresh)
                FXZ[k] = (FXZ[k] - lbias) / lowScale;
            else
                FXZ[k] = FXZ[k] * FXZ[k] * FXZ[k];
        }
        float X = FXZ[0] * xn;
        float Z = FXZ[1] * zn;

        float R = M[0] * X + M[1] * Y + M[2] * Z;
        float G = M[3] * X + M[4] * Y + M[5] * Z;
        float B = M[6] * X + M[7] * Y + M[8] * Z;

        R = std::min(std::max(R, 0.f), 1.f);
        G = std::min(std::max(G, 0.f), 1.f);
        B = std::min(std::max(B, 0.f), 1.f);
        if (srgb)
        {
            R = applyInvGamma(R);
            G = applyInvGamma(G);
            B = applyInvGamma(B);
        }

        dst_row[x] = B;
        dst_row[x + 1] = G;
        dst_row[x + 2] = R;
    }
}


//// rgb <=> L*u*v*
class CV_ColorLuvTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorLuvTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n );
    void convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n );
};


CV_ColorLuvTest::CV_ColorLuvTest() : CV_ColorCvtBaseTest( true, true, false )
{
    INIT_FWD_INV_CODES( BGR2Luv, Luv2BGR );
}


void CV_ColorLuvTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    if(srgb)
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_BGR2Luv, inv_code = COLOR_Luv2BGR;
        else
            fwd_code = COLOR_RGB2Luv, inv_code = COLOR_Luv2RGB;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = COLOR_LBGR2Luv, inv_code = COLOR_Luv2LBGR;
        else
            fwd_code = COLOR_LRGB2Luv, inv_code = COLOR_Luv2LRGB;
    }
}


double CV_ColorLuvTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    // j == 0 is for forward code, j == 1 is for inverse code
    return (depth ==  CV_8U) ? (srgb ? 37 : 8) :
           //(depth == CV_16U) ? 32 : // 16u is disabled
           5e-2;
}


void CV_ColorLuvTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    static const float uLow = -134.f, uHigh = 220.f, uRange = uHigh - uLow;
    static const float vLow = -140.f, vHigh = 122.f, vRange = vHigh - vLow;
    int j;

    float M[9];
    // Yn == 1
    float xn = (float)Xn, zn = (float)Zn;
    float dd = xn + 15.f*1.f + 3.f*zn;
    float un = 4.f*13.f*xn/dd;
    float vn = 9.f*13.f/dd;

    float u_scale = 1.f, u_bias = 0.f;
    float v_scale = 1.f, v_bias = 0.f;

    for( j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    //0.72033 = 255/(220+134), 96.525 = 134*255/(220+134)
    //0.9732 = 255/(140+122), 136.259 = 140*255/(140+122)
    if( depth == CV_8U )
    {
        u_scale = 255.f/uRange;
        u_bias = -uLow*255.f/uRange;
        v_scale = 255.f/vRange;
        v_bias = -vLow*255.f/vRange;
    }

    // 0.008856f = (6/29)^3
    static const float lthresh = 6.f*6.f*6.f/(29.f*29.f*29.f);
    // 903.3 = (29/3)^3
    static const float yscale = 29.f*29.f*29.f/27.f;

    for( j = 0; j < n*3; j += 3 )
    {
        float r = src_row[j+2];
        float g = src_row[j+1];
        float b = src_row[j];

        r = std::min(std::max(r, 0.f), 1.f);
        g = std::min(std::max(g, 0.f), 1.f);
        b = std::min(std::max(b, 0.f), 1.f);
        if( srgb )
        {
            r = applyGamma(r);
            g = applyGamma(g);
            b = applyGamma(b);
        }

        float X = r*M[0] + g*M[1] + b*M[2];
        float Y = r*M[3] + g*M[4] + b*M[5];
        float Z = r*M[6] + g*M[7] + b*M[8];
        float d = X + 15*Y + 3*Z, L, u, v;

        if( d == 0 )
            L = u = v = 0;
        else
        {
            if( Y > lthresh )
                L = 116.f*cubeRoot(Y) - 16.f;
            else
                L = yscale * Y;

            d = 4.f*13.f/d;
            u = L*(X*d - un);
            v = L*(9.f/4.f*Y*d - vn);
        }
        dst_row[j] = L*Lscale;
        dst_row[j+1] = u*u_scale + u_bias;
        dst_row[j+2] = v*v_scale + v_bias;
    }
}


void CV_ColorLuvTest::convert_row_abc2bgr_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 100.f/255.f : depth == CV_16U ? 100.f/65535.f : 1.f;
    static const float uLow = -134.f, uHigh = 220.f, uRange = uHigh - uLow;
    static const float vLow = -140.f, vHigh = 122.f, vRange = vHigh - vLow;

    int j;
    float M[9];
    // Yn == 1
    float xn = (float)Xn, zn = (float)Zn;
    float dd = xn + 15.f*1.f + 3.f*zn;
    float un = 4*13.f*xn/dd;
    float vn = 9*13.f*1.f/dd;

    float u_scale = 1.f, u_bias = 0.f;
    float v_scale = 1.f, v_bias = 0.f;

    for( j = 0; j < 9; j++ )
        M[j] = (float)XYZ2RGB[j];

    //0.72033 = 255/(220+134), 96.525 = 134*255/(220+134)
    //0.9732 = 255/(140+122), 136.259 = 140*255/(140+122)
    if( depth == CV_8U )
    {
        u_scale = uRange/255.f;
        u_bias = -uLow*255.f/uRange;
        v_scale = vRange/255.f;
        v_bias = -vLow*255.f/vRange;
    }

    // (1 / 903.3) = (3/29)^3
    static const float yscale = 27.f/(29.f*29.f*29.f);
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
            Y = L * yscale;
        }

        float up = 3.f*(u + L*un);
        float vp = 0.25f/(v + L*vn);
        if(vp >  0.25f) vp =  0.25f;
        if(vp < -0.25f) vp = -0.25f;
        X = Y*3.f*up*vp;
        Z = Y*(((12.f*13.f)*L - up)*vp - 5.f);

        float r = M[0]*X + M[1]*Y + M[2]*Z;
        float g = M[3]*X + M[4]*Y + M[5]*Z;
        float b = M[6]*X + M[7]*Y + M[8]*Z;

        r = std::min(std::max(r, 0.f), 1.f);
        g = std::min(std::max(g, 0.f), 1.f);
        b = std::min(std::max(b, 0.f), 1.f);

        if( srgb )
        {
            r = applyInvGamma(r);
            g = applyInvGamma(g);
            b = applyInvGamma(b);
        }

        dst_row[j] = b;
        dst_row[j+1] = g;
        dst_row[j+2] = r;
    }
}


//// rgb <=> another rgb
class CV_ColorRGBTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorRGBTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void convert_forward( const Mat& src, Mat& dst );
    void convert_backward( const Mat& src, const Mat& dst, Mat& dst2 );
    int dst_bits;
};


CV_ColorRGBTest::CV_ColorRGBTest() : CV_ColorCvtBaseTest( true, true, true )
{
    dst_bits = 0;
}


void CV_ColorRGBTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    int cn = CV_MAT_CN(types[INPUT][0]);

    dst_bits = 24;

    if( cvtest::randInt(rng) % 3 == 0 )
    {
        types[INPUT][0] = types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(CV_8U,cn);
        types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(CV_8U,2);
        if( cvtest::randInt(rng) & 1 )
        {
            if( blue_idx == 0 )
                fwd_code = COLOR_BGR2BGR565, inv_code = COLOR_BGR5652BGR;
            else
                fwd_code = COLOR_RGB2BGR565, inv_code = COLOR_BGR5652RGB;
            dst_bits = 16;
        }
        else
        {
            if( blue_idx == 0 )
                fwd_code = COLOR_BGR2BGR555, inv_code = COLOR_BGR5552BGR;
            else
                fwd_code = COLOR_RGB2BGR555, inv_code = COLOR_BGR5552RGB;
            dst_bits = 15;
        }
    }
    else
    {
        if( cn == 3 )
        {
            fwd_code = COLOR_RGB2BGR, inv_code = COLOR_BGR2RGB;
            blue_idx = 2;
        }
        else if( blue_idx == 0 )
            fwd_code = COLOR_BGRA2BGR, inv_code = COLOR_BGR2BGRA;
        else
            fwd_code = COLOR_RGBA2BGR, inv_code = COLOR_BGR2RGBA;
    }

    if( CV_MAT_CN(types[INPUT][0]) != CV_MAT_CN(types[OUTPUT][0]) )
        inplace = false;
}


double CV_ColorRGBTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_ColorRGBTest::convert_forward( const Mat& src, Mat& dst )
{
    int depth = src.depth(), cn = src.channels();
    int i, j, cols = src.cols;
    int g_rshift = dst_bits == 16 ? 2 : 3;
    int r_lshift = dst_bits == 16 ? 11 : 10;

    //CV_Assert( (cn == 3 || cn == 4) && (dst_cn == 3 || (dst_cn == 2 && depth == CV_8U)) );

    for( i = 0; i < src.rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = src.ptr(i);
                uchar* dst_row = dst.ptr(i);

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
                const ushort* src_row = src.ptr<ushort>(i);
                ushort* dst_row = dst.ptr<ushort>(i);

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
                const float* src_row = src.ptr<float>(i);
                float* dst_row = dst.ptr<float>(i);

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
            CV_Assert(0);
        }
    }
}


void CV_ColorRGBTest::convert_backward( const Mat& /*src*/, const Mat& src, Mat& dst )
{
    int depth = src.depth(), cn = dst.channels();
    int i, j, cols = src.cols;
    int g_lshift = dst_bits == 16 ? 2 : 3;
    int r_rshift = dst_bits == 16 ? 11 : 10;

    //CV_Assert( (cn == 3 || cn == 4) && (src_cn == 3 || (src_cn == 2 && depth == CV_8U)) );

    for( i = 0; i < src.rows; i++ )
    {
        switch( depth )
        {
        case CV_8U:
            {
                const uchar* src_row = src.ptr(i);
                uchar* dst_row = dst.ptr(i);

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
                const ushort* src_row = src.ptr<ushort>(i);
                ushort* dst_row = dst.ptr<ushort>(i);

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
                const float* src_row = src.ptr<float>(i);
                float* dst_row = dst.ptr<float>(i);

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
            CV_Assert(0);
        }
    }
}


//// rgb <=> bayer

class CV_ColorBayerTest : public CV_ColorCvtBaseTest
{
public:
    CV_ColorBayerTest();
protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int test_case_idx );
};


CV_ColorBayerTest::CV_ColorBayerTest() : CV_ColorCvtBaseTest( false, false, true )
{
    test_array[OUTPUT].pop_back();
    test_array[REF_OUTPUT].pop_back();

    fwd_code_str = "BayerBG2BGR";
    inv_code_str = "";
    fwd_code = COLOR_BayerBG2BGR;
    inv_code = -1;
}


void CV_ColorBayerTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[INPUT][0] = CV_MAT_DEPTH(types[INPUT][0]);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(CV_MAT_DEPTH(types[INPUT][0]), 3);
    inplace = false;

    fwd_code = cvtest::randInt(rng)%4 + COLOR_BayerBG2BGR;
}


double CV_ColorBayerTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1;
}


void CV_ColorBayerTest::run_func()
{
    cv::Mat _out = test_mat[OUTPUT][0];
    cv::cvtColor(test_mat[INPUT][0], _out, fwd_code, _out.channels());
}


template<typename T>
static void bayer2BGR_(const Mat& src, Mat& dst, int code)
{
    int i, j, cols = src.cols - 2;
    int bi = 0;
    int step = (int)(src.step/sizeof(T));

    if( code == COLOR_BayerRG2BGR || code == COLOR_BayerGR2BGR )
        bi ^= 2;

    for( i = 1; i < src.rows - 1; i++ )
    {
        const T* ptr = src.ptr<T>(i) + 1;
        T* dst_row = dst.ptr<T>(i) + 3;
        int save_code = code;
        if( cols <= 0 )
        {
            dst_row[-3] = dst_row[-2] = dst_row[-1] = 0;
            dst_row[cols*3] = dst_row[cols*3+1] = dst_row[cols*3+2] = 0;
            continue;
        }

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
            dst_row[j*3 + bi] = (T)b;
            dst_row[j*3 + 1] = (T)g;
            dst_row[j*3 + (bi^2)] = (T)r;
        }

        dst_row[-3] = dst_row[0];
        dst_row[-2] = dst_row[1];
        dst_row[-1] = dst_row[2];
        dst_row[cols*3] = dst_row[cols*3-3];
        dst_row[cols*3+1] = dst_row[cols*3-2];
        dst_row[cols*3+2] = dst_row[cols*3-1];

        code = save_code ^ 1;
        bi ^= 2;
    }

    if( src.rows <= 2 )
    {
        memset( dst.ptr(), 0, (cols+2)*3*sizeof(T) );
        memset( dst.ptr(dst.rows-1), 0, (cols+2)*3*sizeof(T) );
    }
    else
    {
        T* top_row = dst.ptr<T>();
        T* bottom_row = dst.ptr<T>(dst.rows-1);
        int dstep = (int)(dst.step/sizeof(T));

        for( j = 0; j < (cols+2)*3; j++ )
        {
            top_row[j] = top_row[j + dstep];
            bottom_row[j] = bottom_row[j - dstep];
        }
    }
}


void CV_ColorBayerTest::prepare_to_validation( int /*test_case_idx*/ )
{
    const Mat& src = test_mat[INPUT][0];
    Mat& dst = test_mat[REF_OUTPUT][0];
    int depth = src.depth();
    if( depth == CV_8U )
        bayer2BGR_<uchar>(src, dst, fwd_code);
    else if( depth == CV_16U )
        bayer2BGR_<ushort>(src, dst, fwd_code);
    else
        CV_Error(cv::Error::StsUnsupportedFormat, "");
}


/////////////////////////////////////////////////////////////////////////////////////////////////

TEST(Imgproc_ColorGray, accuracy) { CV_ColorGrayTest test; test.safe_run(); }
TEST(Imgproc_ColorYCrCb, accuracy) { CV_ColorYCrCbTest test; test.safe_run(); }
TEST(Imgproc_ColorHSV, accuracy) { CV_ColorHSVTest test; test.safe_run(); }
TEST(Imgproc_ColorHLS, accuracy) { CV_ColorHLSTest test; test.safe_run(); }
TEST(Imgproc_ColorXYZ, accuracy) { CV_ColorXYZTest test; test.safe_run(); }
TEST(Imgproc_ColorLab, accuracy) { CV_ColorLabTest test; test.safe_run(); }
TEST(Imgproc_ColorLuv, accuracy) { CV_ColorLuvTest test; test.safe_run(); }
TEST(Imgproc_ColorRGB, accuracy) { CV_ColorRGBTest test; test.safe_run(); }
TEST(Imgproc_ColorBayer, accuracy) { CV_ColorBayerTest test; test.safe_run(); }

TEST(Imgproc_ColorLuv, Overflow_21112)
{
    const Size sz(107, 16);  // unaligned size to run both SIMD and generic code
    Mat luv_init(sz, CV_8UC3, Scalar(49, 205, 23));
    Mat rgb;
    cvtColor(luv_init, rgb, COLOR_Luv2RGB);
    // Convert to normal Luv coordinates for floats.
    Mat luv_initf(sz, CV_32FC3, Scalar(49.0f/255.f*100, 205.0f*354/255.f - 134, 23.0f*262/255.f - 140));
    Mat rgbf;
    cvtColor(luv_initf, rgbf, COLOR_Luv2RGB);
    Mat rgb_converted;
    rgb.convertTo(rgb_converted, CV_32F);
    EXPECT_LE(cvtest::norm(255.f*rgbf, rgb_converted, NORM_INF), 1e-5);
}

TEST(Imgproc_ColorBayer, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();

    Mat given = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    Mat gold = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_gold.png", IMREAD_UNCHANGED);
    Mat result;

    CV_Assert( !given.empty() && !gold.empty() );

    cvtColor(given, result, COLOR_BayerBG2GRAY);

    EXPECT_EQ(gold.type(), result.type());
    EXPECT_EQ(gold.cols, result.cols);
    EXPECT_EQ(gold.rows, result.rows);

    Mat diff;
    absdiff(gold, result, diff);

    EXPECT_EQ(0, countNonZero(diff.reshape(1) > 1));
}

TEST(Imgproc_ColorBayer2Gray, regression_25823)
{
    const int n = 100;
    Mat src(n, n, CV_8UC1);
    Mat dst;

    for (int i = 0; i < src.rows; ++i)
    {
        for (int j = 0; j < src.cols; ++j)
        {
            src.at<uchar>(i, j) = (i + j) % 2;
        }
    }

    cvtColor(src, dst, COLOR_BayerBG2GRAY);

    Mat gold(n, n, CV_8UC1, Scalar(1));
    EXPECT_EQ(0, cv::norm(dst, gold, NORM_INF));
}

TEST(Imgproc_ColorBayerVNG, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();

    Mat given = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    string goldfname = string(ts->get_data_path()) + "/cvtcolor/bayerVNG_gold.png";
    Mat gold = imread(goldfname, IMREAD_UNCHANGED);
    Mat result;

    CV_Assert( !given.empty() );

    cvtColor(given, result, COLOR_BayerBG2BGR_VNG, 3);

    if (gold.empty())
        imwrite(goldfname, result);
    else
    {
        EXPECT_EQ(gold.type(), result.type());
        EXPECT_EQ(gold.cols, result.cols);
        EXPECT_EQ(gold.rows, result.rows);

        Mat diff;
        absdiff(gold, result, diff);

        EXPECT_EQ(0, countNonZero(diff.reshape(1) > 1));
    }
}

// See https://github.com/opencv/opencv/issues/5089
// See https://github.com/opencv/opencv/issues/27225
typedef tuple<cv::ColorConversionCodes, cv::ColorConversionCodes> VNGandINT;
typedef testing::TestWithParam<VNGandINT> Imgproc_ColorBayerVNG_Codes;

TEST_P(Imgproc_ColorBayerVNG_Codes, regression27225)
{
    const cv::ColorConversionCodes codeVNG = get<0>(GetParam());
    const int margin = (codeVNG == cv::COLOR_BayerGB2BGR_VNG || codeVNG == cv::COLOR_BayerGR2BGR_VNG)? 5 : 4;

    cv::Mat in = cv::Mat::eye(16, 16, CV_8UC1) * 255;
    cv::resize(in, in, {}, 2, 2, cv::INTER_NEAREST);

    cv::Mat out;
    EXPECT_NO_THROW(cv::cvtColor(in, out, codeVNG));

    for(int iy=2; iy < out.size().height-2; iy++) {
        for(int ix=2; ix < out.size().width-2; ix++) {
            // Avoid to test around main diagonal pixels.
            if(cv::abs(ix - iy) < margin) {
                continue;
            }
            // Others should be completely black.
            const Vec3b pixel = out.at<Vec3b>(iy, ix);
            EXPECT_EQ(pixel[0], 0) << cv::format(" - iy = %d, ix = %d", iy, ix);
            EXPECT_EQ(pixel[1], 0) << cv::format(" - iy = %d, ix = %d", iy, ix);
            EXPECT_EQ(pixel[2], 0) << cv::format(" - iy = %d, ix = %d", iy, ix);
        }
    }
}

TEST_P(Imgproc_ColorBayerVNG_Codes, regression27225_small)
{
    // for too small images use the simple interpolation algorithm
    const cv::ColorConversionCodes codeVNG = get<0>(GetParam());
    const cv::ColorConversionCodes codeINT = get<1>(GetParam());
    cv::Mat in = cv::Mat::eye(7, 7, CV_8UC1) * 255;

    cv::Mat outVNG;
    EXPECT_NO_THROW(cv::cvtColor(in, outVNG, codeVNG));
    cv::Mat outINT;
    EXPECT_NO_THROW(cv::cvtColor(in, outINT, codeINT));

    Mat diff;
    absdiff(outVNG, outINT, diff);

    EXPECT_EQ(0, countNonZero(diff.reshape(1) > 1));
}

INSTANTIATE_TEST_CASE_P(/**/, Imgproc_ColorBayerVNG_Codes,
    testing::Values(
        make_tuple(cv::COLOR_BayerBG2BGR_VNG, cv::COLOR_BayerBG2BGR),
        make_tuple(cv::COLOR_BayerGB2BGR_VNG, cv::COLOR_BayerGB2BGR),
        make_tuple(cv::COLOR_BayerRG2BGR_VNG, cv::COLOR_BayerRG2BGR),
        make_tuple(cv::COLOR_BayerGR2BGR_VNG, cv::COLOR_BayerGR2BGR)));

// creating Bayer pattern
template <typename T, int depth>
static void calculateBayerPattern(const Mat& src, Mat& bayer, const char* pattern)
{
    Size ssize = src.size();
    const int scn = 1;
    bayer.create(ssize, CV_MAKETYPE(depth, scn));

    if (!strcmp(pattern, "bg"))
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
    else if (!strcmp(pattern, "gb"))
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
    else if (!strcmp(pattern, "rg"))
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
    else
    {
        for (int y = 0; y < ssize.height; ++y)
            for (int x = 0; x < ssize.width; ++x)
            {
                if ((x + y) % 2 == 0)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[1]);
                else if (x % 2)
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[0]);
                else
                    bayer.at<T>(y, x) = static_cast<T>(src.at<Vec3b>(y, x)[2]);
            }
    }
}

TEST(Imgproc_ColorBayerVNG_Strict, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    const char pattern[][3] = { "bg", "gb", "rg", "gr" };
    const std::string image_name = "lena.png";
    const std::string parent_path = string(ts->get_data_path()) + "/cvtcolor_strict/";

    Mat src, dst, bayer, reference;
    std::string full_path = parent_path + image_name;
    src = imread(full_path, IMREAD_UNCHANGED);

    if ( src.empty() )
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        ts->printf(cvtest::TS::SUMMARY, "No input image\n");
        ts->set_gtest_status();
        return;
    }

    for (int i = 0; i < 4; ++i)
    {
        calculateBayerPattern<uchar, CV_8U>(src, bayer, pattern[i]);
        CV_Assert(!bayer.empty() && bayer.type() == CV_8UC1);

        // calculating a dst image
        cvtColor(bayer, dst, COLOR_BayerBG2BGR_VNG + i);

        // reading a reference image
        full_path = parent_path + pattern[i] + image_name;
        reference = imread(full_path, IMREAD_UNCHANGED);
        if ( reference.empty() )
        {
            imwrite(full_path, dst);
            continue;
        }

        if (reference.depth() != dst.depth() || reference.channels() != dst.channels() ||
            reference.size() != dst.size())
        {
            std::cout << reference(Rect(0, 0, 5, 5)) << std::endl << std::endl << std::endl;
            ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
            ts->printf(cvtest::TS::SUMMARY, "\nReference channels: %d\n"
                "Actual channels: %d\n", reference.channels(), dst.channels());
            ts->printf(cvtest::TS::SUMMARY, "\nReference depth: %d\n"
                "Actual depth: %d\n", reference.depth(), dst.depth());
            ts->printf(cvtest::TS::SUMMARY, "\nReference rows: %d\n"
                "Actual rows: %d\n", reference.rows, dst.rows);
            ts->printf(cvtest::TS::SUMMARY, "\nReference cols: %d\n"
                "Actual cols: %d\n", reference.cols, dst.cols);
            ts->set_gtest_status();

            return;
        }

        Mat diff;
        absdiff(reference, dst, diff);

        int nonZero = countNonZero(diff.reshape(1) > 1);
        if (nonZero != 0)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
            ts->printf(cvtest::TS::SUMMARY, "\nCount non zero in absdiff: %d\n", nonZero);
            ts->set_gtest_status();
            return;
        }
    }
}

static void getTestMatrix(Mat& src)
{
    Size ssize(1000, 1000);
    src.create(ssize, CV_32FC3);
    int szm = ssize.width - 1;
    float pi2 = 2 * 3.1415f;
    // Generate a pretty test image
    for (int i = 0; i < ssize.height; i++)
    {
        for (int j = 0; j < ssize.width; j++)
        {
            float b = (1 + cos((szm - i) * (szm - j) * pi2 / (10 * float(szm)))) / 2;
            float g = (1 + cos((szm - i) * j * pi2 / (10 * float(szm)))) / 2;
            float r = (1 + sin(i * j * pi2 / (10 * float(szm)))) / 2;

            // The following lines aren't necessary, but just to prove that
            // the BGR values all lie in [0,1]...
            if (b < 0) b = 0; else if (b > 1) b = 1;
            if (g < 0) g = 0; else if (g > 1) g = 1;
            if (r < 0) r = 0; else if (r > 1) r = 1;
            src.at<cv::Vec3f>(i, j) = cv::Vec3f(b, g, r);
        }
    }
}

static void validateResult(const Mat& reference, const Mat& actual, const Mat& src = Mat(), int mode = -1)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    Size ssize = reference.size();

    int cn = reference.channels();
    ssize.width *= cn;
    bool next = true;
    //RGB2Lab_f works through LUT and brings additional error
    static const float maxErr = 1.f/192.f;

    for (int y = 0; y < ssize.height && next; ++y)
    {
        const float* rD = reference.ptr<float>(y);
        const float* D = actual.ptr<float>(y);
        for (int x = 0; x < ssize.width && next; ++x)
            if(fabs(rD[x] - D[x]) > maxErr)
            {
                next = false;
                ts->printf(cvtest::TS::SUMMARY, "Error in: (%d, %d)\n", x / cn,  y);
                ts->printf(cvtest::TS::SUMMARY, "Reference value: %f\n", rD[x]);
                ts->printf(cvtest::TS::SUMMARY, "Actual value: %f\n", D[x]);
                if (!src.empty())
                    ts->printf(cvtest::TS::SUMMARY, "Src value: %f\n", src.ptr<float>(y)[x]);
                ts->printf(cvtest::TS::SUMMARY, "Size: (%d, %d)\n", reference.rows, reference.cols);

                if (mode >= 0)
                {
                    cv::Mat lab;
                    cv::cvtColor(src, lab, mode);
                    std::cout << "lab: " << lab(cv::Rect(y, x / cn, 1, 1)) << std::endl;
                }
                std::cout << "src: " << src(cv::Rect(y, x / cn, 1, 1)) << std::endl;

                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                ts->set_gtest_status();
            }
    }
}

TEST(Imgproc_ColorLab_Full, accuracy)
{
    Mat src;
    getTestMatrix(src);
    Size ssize = src.size();
    CV_Assert(ssize.width == ssize.height);

    for(int i = 0; i < 4; i++)
    {
        int blueInd = (i%2) > 0 ? 0 : 2;
        bool srgb = i > 1;

        // Convert test image to LAB
        cv::Mat lab;
        int forward_code = blueInd ? srgb ? COLOR_BGR2Lab : COLOR_LBGR2Lab : srgb ? COLOR_RGB2Lab : COLOR_LRGB2Lab;
        int inverse_code = blueInd ? srgb ? COLOR_Lab2BGR : COLOR_Lab2LBGR : srgb ? COLOR_Lab2RGB : COLOR_Lab2LRGB;
        cv::cvtColor(src, lab, forward_code);
        // Convert LAB image back to BGR(RGB)
        cv::Mat recons;
        cv::cvtColor(lab, recons, inverse_code);

        validateResult(src, recons, src, forward_code);
    }
}


static uint32_t adler32(Mat m)
{
    uint32_t s1 = 1, s2 = 0;
    for(int y = 0; y < m.rows; y++)
    {
        uchar* py = m.ptr(y);
        for(size_t x = 0; x < m.cols*m.elemSize(); x++)
        {
            s1 = (s1 + py[x]) % 65521;
            s2 = (s1 + s2   ) % 65521;
        }
    }
    return (s2 << 16) + s1;
}


// taken from color.cpp

static ushort sRGBGammaTab_b[256], linearGammaTab_b[256];
enum { inv_gamma_shift = 12, INV_GAMMA_TAB_SIZE = (1 << inv_gamma_shift) };
static ushort sRGBInvGammaTab_b[INV_GAMMA_TAB_SIZE], linearInvGammaTab_b[INV_GAMMA_TAB_SIZE];
#undef lab_shift
// #define lab_shift xyz_shift
#define lab_shift 12
#define gamma_shift 3
#define lab_shift2 (lab_shift + gamma_shift)
#define LAB_CBRT_TAB_SIZE_B (256*3/2*(1<<gamma_shift))
static ushort LabCbrtTab_b[LAB_CBRT_TAB_SIZE_B];

enum
{
    lab_lut_shift = 5,
    LAB_LUT_DIM = (1 << lab_lut_shift)+1,
    lab_base_shift = 14,
    LAB_BASE = (1 << lab_base_shift),
    trilinear_shift = 8 - lab_lut_shift + 1,
    TRILINEAR_BASE = (1 << trilinear_shift)
};

static int16_t trilinearLUT[TRILINEAR_BASE*TRILINEAR_BASE*TRILINEAR_BASE*8];
static int16_t RGB2LuvLUT_s16[LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3*8];
static const softfloat uLow(-134), uHigh(220), uRange(uHigh-uLow);
static const softfloat vLow(-140), vHigh(122), vRange(vHigh-vLow);
static int LuToUp_b[256*256];
static int LvToVp_b[256*256];
static long long int LvToVpl_b[256*256];

#define  CV_DESCALE(x,n)     (((x) + (1 << ((n)-1))) >> (n))

static ushort LabToYF_b[256*2];
static const int minABvalue = -8145;
static int abToXZ_b[LAB_BASE*9/4];

static void initLabTabs()
{
    static bool initialized = false;
    if(!initialized)
    {
        static const softfloat lthresh = softfloat(216) / softfloat(24389); // 0.008856f = (6/29)^3
        static const softfloat lscale  = softfloat(841) / softfloat(108); // 7.787f = (29/3)^3/(29*4)
        static const softfloat lbias = softfloat(16) / softfloat(116);
        static const softfloat f255(255);

        static const softfloat intScale(255*(1 << gamma_shift));
        for(int i = 0; i < 256; i++)
        {
            softfloat x = softfloat(i)/f255;
            sRGBGammaTab_b[i] = (ushort)(cvRound(intScale*applyGamma(x)));
            linearGammaTab_b[i] = (ushort)(i*(1 << gamma_shift));
        }
        static const softfloat invScale = softfloat::one()/softfloat((int)INV_GAMMA_TAB_SIZE);
        for(int i = 0; i < INV_GAMMA_TAB_SIZE; i++)
        {
            softfloat x = invScale*softfloat(i);
            sRGBInvGammaTab_b[i] = (ushort)(cvRound(f255*applyInvGamma(x)));
            linearInvGammaTab_b[i] = (ushort)(cvTrunc(f255*x));
        }

        static const softfloat cbTabScale(softfloat::one()/(f255*(1 << gamma_shift)));
        static const softfloat lshift2(1 << lab_shift2);
        for(int i = 0; i < LAB_CBRT_TAB_SIZE_B; i++)
        {
            softfloat x = cbTabScale*softfloat(i);
            LabCbrtTab_b[i] = (ushort)(cvRound(lshift2 * (x < lthresh ? mulAdd(x, lscale, lbias) : cbrt(x))));
        }

        //Lookup table for L to y and ify calculations
        static const int BASE = (1 << 14);
        for(int i = 0; i < 256; i++)
        {
            int y, ify;
            //8 * 255.0 / 100.0 == 20.4
            if( i <= 20)
            {
                //yy = li / 903.3f;
                //y = L*100/903.3f; 903.3f = (29/3)^3, 255 = 17*3*5
                y = cvRound(softfloat(i*BASE*20*9)/softfloat(17*29*29*29));
                //fy = 7.787f * yy + 16.0f / 116.0f; 7.787f = (29/3)^3/(29*4)
                ify = cvRound(softfloat(BASE)*(softfloat(16)/softfloat(116) + softfloat(i*5)/softfloat(3*17*29)));
            }
            else
            {
                //fy = (li + 16.0f) / 116.0f;
                softfloat fy = (softfloat(i*100*BASE)/softfloat(255*116) +
                                softfloat(16*BASE)/softfloat(116));
                ify = cvRound(fy);
                //yy = fy * fy * fy;
                y = cvRound(fy*fy*fy/softfloat(BASE*BASE));
            }

            LabToYF_b[i*2  ] = (ushort)y;   // 2260 <= y <= BASE
            LabToYF_b[i*2+1] = (ushort)ify; // 0 <= ify <= BASE
        }

        //Lookup table for a,b to x,z conversion
        for(int i = minABvalue; i < LAB_BASE*9/4+minABvalue; i++)
        {
            int v;
            //6.f/29.f*BASE = 3389.730
            if(i <= 3390)
            {
                //fxz[k] = (fxz[k] - 16.0f / 116.0f) / 7.787f;
                // 7.787f = (29/3)^3/(29*4)
                v = i*108/841 - BASE*16/116*108/841;
            }
            else
            {
                //fxz[k] = fxz[k] * fxz[k] * fxz[k];
                v = i*i/BASE*i/BASE;
            }
            abToXZ_b[i-minABvalue] = v; // -1335 <= v <= 88231
        }

        softdouble D65[] = { Xn, softdouble::one(), Zn };
        softfloat dd = (D65[0] + D65[1]*softdouble(15) + D65[2]*softdouble(3));
        dd = softfloat::one()/max(dd, softfloat::eps());
        softfloat un = dd*softfloat(13*4)*D65[0];
        softfloat vn = dd*softfloat(13*9)*D65[1];

        //Luv LUT
        softfloat oneof4 = softfloat::one()/softfloat(4);

        for(int LL = 0; LL < 256; LL++)
        {
            softfloat L = softfloat(LL*100)/f255;
            for(int uu = 0; uu < 256; uu++)
            {
                softfloat u = softfloat(uu)*uRange/f255 + uLow;
                softfloat up = softfloat(9)*(u + L*un);
                LuToUp_b[LL*256+uu] = cvRound(up*softfloat(BASE/1024));//1024 is OK, 2048 gave maxerr 3
            }
            for(int vv = 0; vv < 256; vv++)
            {
                softfloat v = softfloat(vv)*vRange/f255 + vLow;
                softfloat vp = oneof4/(v + L*vn);
                if(vp >  oneof4) vp =  oneof4;
                if(vp < -oneof4) vp = -oneof4;
                int ivp = cvRound(vp*softfloat(BASE*1024));
                LvToVp_b[LL*256+vv] = ivp;
                int vpl = ivp*LL;
                LvToVpl_b[LL*256+vv] = (12*13*100*(BASE/1024))*(long long)vpl;
            }
        }

        softfloat coeffs[9];
        for(int i = 0; i < 3; i++ )
        {
            coeffs[i*3+2] = RGB2XYZ[i*3  ];
            coeffs[i*3+1] = RGB2XYZ[i*3+1];
            coeffs[i*3  ] = RGB2XYZ[i*3+2];
        }

        softfloat C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2],
                  C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5],
                  C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

        //u, v: [-134.0, 220.0], [-140.0, 122.0]
        static const softfloat lld(LAB_LUT_DIM - 1), f116(116), f16(16);
        static const softfloat f100(100), lbase((int)LAB_BASE);
        static const softfloat f9of4 = softfloat(9)/softfloat(4);
        static const softfloat f15(15), f3(3);
        AutoBuffer<int16_t> RGB2Luvprev(LAB_LUT_DIM*LAB_LUT_DIM*LAB_LUT_DIM*3);
        for(int p = 0; p < LAB_LUT_DIM; p++)
        {
            for(int q = 0; q < LAB_LUT_DIM; q++)
            {
                for(int r = 0; r < LAB_LUT_DIM; r++)
                {
                    int idx = p*3 + q*LAB_LUT_DIM*3 + r*LAB_LUT_DIM*LAB_LUT_DIM*3;
                    softfloat R = softfloat(p)/lld;
                    softfloat G = softfloat(q)/lld;
                    softfloat B = softfloat(r)/lld;

                    R = applyGamma(R);
                    G = applyGamma(G);
                    B = applyGamma(B);

                    //RGB 2 Luv LUT building
                    {
                        softfloat X = R*C0 + G*C1 + B*C2;
                        softfloat Y = R*C3 + G*C4 + B*C5;
                        softfloat Z = R*C6 + G*C7 + B*C8;

                        softfloat L = Y < lthresh ? mulAdd(Y, lscale, lbias) : cbrt(Y);
                        L = L*f116 - f16;

                        softfloat d = softfloat(4*13)/max(X + f15 * Y + f3 * Z, softfloat(FLT_EPSILON));
                        softfloat u = L*(X*d - un);
                        softfloat v = L*(f9of4*Y*d - vn);

                        RGB2Luvprev[idx  ] = (int16_t)cvRound(lbase*L/f100);
                        RGB2Luvprev[idx+1] = (int16_t)cvRound(lbase*(u-uLow)/uRange);
                        RGB2Luvprev[idx+2] = (int16_t)cvRound(lbase*(v-vLow)/vRange);
                    }
                }
            }
        }
        for(int p = 0; p < LAB_LUT_DIM; p++)
        {
            for(int q = 0; q < LAB_LUT_DIM; q++)
            {
                for(int r = 0; r < LAB_LUT_DIM; r++)
                {
                    #define FILL(_p, _q, _r) \
                        do {\
                        int idxold = 0;\
                        idxold += std::min(p+(_p), (int)(LAB_LUT_DIM-1))*3;\
                        idxold += std::min(q+(_q), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*3;\
                        idxold += std::min(r+(_r), (int)(LAB_LUT_DIM-1))*LAB_LUT_DIM*LAB_LUT_DIM*3;\
                        int idxnew = p*3*8 + q*LAB_LUT_DIM*3*8 + r*LAB_LUT_DIM*LAB_LUT_DIM*3*8+4*(_p)+2*(_q)+(_r);\
                        RGB2LuvLUT_s16[idxnew]    = RGB2Luvprev[idxold];\
                        RGB2LuvLUT_s16[idxnew+8]  = RGB2Luvprev[idxold+1];\
                        RGB2LuvLUT_s16[idxnew+16] = RGB2Luvprev[idxold+2];\
                        } while(0)

                    FILL(0, 0, 0); FILL(0, 0, 1);
                    FILL(0, 1, 0); FILL(0, 1, 1);
                    FILL(1, 0, 0); FILL(1, 0, 1);
                    FILL(1, 1, 0); FILL(1, 1, 1);

                    #undef FILL
                }
            }
        }

        for(int16_t p = 0; p < TRILINEAR_BASE; p++)
        {
            int16_t pp = TRILINEAR_BASE - p;
            for(int16_t q = 0; q < TRILINEAR_BASE; q++)
            {
                int16_t qq = TRILINEAR_BASE - q;
                for(int16_t r = 0; r < TRILINEAR_BASE; r++)
                {
                    int16_t rr = TRILINEAR_BASE - r;
                    int16_t* w = &trilinearLUT[8*p + 8*TRILINEAR_BASE*q + 8*TRILINEAR_BASE*TRILINEAR_BASE*r];
                    w[0]  = pp * qq * rr; w[1]  = pp * qq * r ; w[2]  = pp * q  * rr; w[3]  = pp * q  * r ;
                    w[4]  = p  * qq * rr; w[5]  = p  * qq * r ; w[6]  = p  * q  * rr; w[7]  = p  * q  * r ;
                }
            }
        }

        initialized = true;
    }
}

static int row8uRGB2Lab(const uchar* src_row, uchar *dst_row, int n, int cn, int blue_idx, bool srgb)
{
    int coeffs[9];
    softdouble whitept[3] = {Xn, softdouble::one(), Zn};

    static const softdouble lshift(1 << lab_shift);
    for(int i = 0; i < 3; i++)
    {
        coeffs[i*3 + (blue_idx^2)] = cvRound(lshift*RGB2XYZ[i*3  ]/whitept[i]);
        coeffs[i*3 + 1           ] = cvRound(lshift*RGB2XYZ[i*3+1]/whitept[i]);
        coeffs[i*3 + (blue_idx  )] = cvRound(lshift*RGB2XYZ[i*3+2]/whitept[i]);
    }

    const int Lscale = (116*255+50)/100;
    const int Lshift = -((16*255*(1 << lab_shift2) + 50)/100);
    const ushort* tab = srgb ? sRGBGammaTab_b : linearGammaTab_b;
    for (int x = 0; x < n; x++)
    {
        int R = src_row[x*cn + 0],
            G = src_row[x*cn + 1],
            B = src_row[x*cn + 2];
        R = tab[R], G = tab[G], B = tab[B];
        int fX = LabCbrtTab_b[CV_DESCALE(R*coeffs[0] + G*coeffs[1] + B*coeffs[2], lab_shift)];
        int fY = LabCbrtTab_b[CV_DESCALE(R*coeffs[3] + G*coeffs[4] + B*coeffs[5], lab_shift)];
        int fZ = LabCbrtTab_b[CV_DESCALE(R*coeffs[6] + G*coeffs[7] + B*coeffs[8], lab_shift)];

        int L = CV_DESCALE( Lscale*fY + Lshift, lab_shift2 );
        int a = CV_DESCALE( 500*(fX - fY) + 128*(1 << lab_shift2), lab_shift2 );
        int b = CV_DESCALE( 200*(fY - fZ) + 128*(1 << lab_shift2), lab_shift2 );

        dst_row[x*3    ] = saturate_cast<uchar>(L);
        dst_row[x*3 + 1] = saturate_cast<uchar>(a);
        dst_row[x*3 + 2] = saturate_cast<uchar>(b);
    }

    return n;
}


int row8uLab2RGB(const uchar* src_row, uchar *dst_row, int n, int cn, int blue_idx, bool srgb)
{
    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);

    int coeffs[9];
    softdouble whitept[3] = {Xn, softdouble::one(), Zn};

    static const softdouble lshift(1 << lab_shift);
    for(int i = 0; i < 3; i++)
    {
        coeffs[i+(blue_idx  )*3] = cvRound(lshift*XYZ2RGB[i  ]*whitept[i]);
        coeffs[i+           1*3] = cvRound(lshift*XYZ2RGB[i+3]*whitept[i]);
        coeffs[i+(blue_idx^2)*3] = cvRound(lshift*XYZ2RGB[i+6]*whitept[i]);
    }
    ushort* tab = srgb ? sRGBInvGammaTab_b : linearInvGammaTab_b;

    for(int x = 0; x < n; x++)
    {
        uchar LL = src_row[x*3    ];
        uchar aa = src_row[x*3 + 1];
        uchar bb = src_row[x*3 + 2];

        int ro, go, bo, xx, yy, zz, ify;

        yy  = LabToYF_b[LL*2  ];
        ify = LabToYF_b[LL*2+1];

        int adiv, bdiv;
        //adiv = aa*BASE/500 - 128*BASE/500, bdiv = bb*BASE/200 - 128*BASE/200;
        //approximations with reasonable precision
        adiv = ((5*aa*53687 + (1 << 7)) >> 13) - 128*BASE/500;
        bdiv = ((  bb*41943 + (1 << 4)) >>  9) - 128*BASE/200+1;

        int ifxz[] = {ify + adiv, ify - bdiv};

        for(int k = 0; k < 2; k++)
        {
            int& v = ifxz[k];
            v = abToXZ_b[v-minABvalue];
        }
        xx = ifxz[0]; /* yy = yy */; zz = ifxz[1];

        ro = CV_DESCALE(coeffs[0]*xx + coeffs[1]*yy + coeffs[2]*zz, shift);
        go = CV_DESCALE(coeffs[3]*xx + coeffs[4]*yy + coeffs[5]*zz, shift);
        bo = CV_DESCALE(coeffs[6]*xx + coeffs[7]*yy + coeffs[8]*zz, shift);

        ro = std::max(0, std::min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = std::max(0, std::min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = std::max(0, std::min((int)INV_GAMMA_TAB_SIZE-1, bo));

        ro = tab[ro];
        go = tab[go];
        bo = tab[bo];

        dst_row[x*cn    ] = saturate_cast<uchar>(bo);
        dst_row[x*cn + 1] = saturate_cast<uchar>(go);
        dst_row[x*cn + 2] = saturate_cast<uchar>(ro);
        if(cn == 4) dst_row[x*cn + 3] = 255;
    }

    return n;
}


int row8uRGB2Luv(const uchar* src_row, uchar *dst_row, int n, int cn, int blue_idx)
{
    for (int x = 0; x < n; x++)
    {
        int R = src_row[x*cn + (blue_idx)],
            G = src_row[x*cn + 1],
            B = src_row[x*cn + (blue_idx^2)];

        // (LAB_BASE/255) gives more accuracy but not very much
        static const int baseDiv = LAB_BASE/256;
        // cx, cy, cz are in [0; LAB_BASE]
        int cx = R*baseDiv, cy = G*baseDiv, cz = B*baseDiv;
        int L, u, v;

        //LUT idx of origin pt of cube
        int tx = cx >> (lab_base_shift - lab_lut_shift);
        int ty = cy >> (lab_base_shift - lab_lut_shift);
        int tz = cz >> (lab_base_shift - lab_lut_shift);

        int16_t* baseLUT = &RGB2LuvLUT_s16[3*8*tx + (3*8*LAB_LUT_DIM)*ty + (3*8*LAB_LUT_DIM*LAB_LUT_DIM)*tz];
        int aa[8], bb[8], cc[8];
        for(int i = 0; i < 8; i++)
        {
            aa[i] = baseLUT[i]; bb[i] = baseLUT[i+8]; cc[i] = baseLUT[i+16];
        }

        //x, y, z are [0; TRILINEAR_BASE)
        static const int bitMask = (1 << trilinear_shift) - 1;
        int xx = (cx >> (lab_base_shift - 8 - 1)) & bitMask;
        int yy = (cy >> (lab_base_shift - 8 - 1)) & bitMask;
        int zz = (cz >> (lab_base_shift - 8 - 1)) & bitMask;

        int w[8];
        for(int i = 0; i < 8; i++)
        {
            w[i] = trilinearLUT[8*xx + 8*TRILINEAR_BASE*yy + 8*TRILINEAR_BASE*TRILINEAR_BASE*zz + i];
        }

        L = aa[0]*w[0]+aa[1]*w[1]+aa[2]*w[2]+aa[3]*w[3]+aa[4]*w[4]+aa[5]*w[5]+aa[6]*w[6]+aa[7]*w[7];
        u = bb[0]*w[0]+bb[1]*w[1]+bb[2]*w[2]+bb[3]*w[3]+bb[4]*w[4]+bb[5]*w[5]+bb[6]*w[6]+bb[7]*w[7];
        v = cc[0]*w[0]+cc[1]*w[1]+cc[2]*w[2]+cc[3]*w[3]+cc[4]*w[4]+cc[5]*w[5]+cc[6]*w[6]+cc[7]*w[7];

        L = CV_DESCALE(L, trilinear_shift*3);
        u = CV_DESCALE(u, trilinear_shift*3);
        v = CV_DESCALE(v, trilinear_shift*3);

        dst_row[x*3    ] = saturate_cast<uchar>(L/baseDiv);
        dst_row[x*3 + 1] = saturate_cast<uchar>(u/baseDiv);
        dst_row[x*3 + 2] = saturate_cast<uchar>(v/baseDiv);
    }

    return n;
}

int row8uLuv2RGB(const uchar* src_row, uchar *dst_row, int n, int cn, int blue_idx, bool srgb)
{
    static const int base_shift = 14;
    static const int BASE = (1 << base_shift);
    static const int shift = lab_shift+(base_shift-inv_gamma_shift);
    int coeffs[9];

    static const softdouble lshift(1 << lab_shift);
    for(int i = 0; i < 3; i++)
    {
        coeffs[i+(blue_idx  )*3] = cvRound(lshift*XYZ2RGB[i  ]);
        coeffs[i+           1*3] = cvRound(lshift*XYZ2RGB[i+3]);
        coeffs[i+(blue_idx^2)*3] = cvRound(lshift*XYZ2RGB[i+6]);
    }

    ushort *tab = srgb ? sRGBInvGammaTab_b : linearInvGammaTab_b;

    int C0 = coeffs[0], C1 = coeffs[1], C2 = coeffs[2];
    int C3 = coeffs[3], C4 = coeffs[4], C5 = coeffs[5];
    int C6 = coeffs[6], C7 = coeffs[7], C8 = coeffs[8];

    for(int xx = 0; xx < n; xx++)
    {
        uchar LL = src_row[xx*3    ];
        uchar uu = src_row[xx*3 + 1];
        uchar vv = src_row[xx*3 + 2];

        ushort y = LabToYF_b[LL*2];

        int up = LuToUp_b[LL*256+uu];
        int vp = LvToVp_b[LL*256+vv];

        long long int xv = ((int)up)*(long long)vp;
        int x = (int)(xv/BASE);
        x = ((long long int)y)*x/BASE;

        long long int vpl = LvToVpl_b[LL*256+vv];
        long long int zp = vpl - xv*(255/3);
        zp /= BASE;
        long long int zq = zp - (long long)(5*255*BASE);
        int zm = (int)(y*zq/BASE);
        int z = zm/256 + zm/65536;

        //limit X, Y, Z to [0, 2] to fit white point
        x = std::max(0, std::min(2*BASE, x)); z = std::max(0, std::min(2*BASE, z));

        int ro, go, bo;
        ro = CV_DESCALE(C0 * x + C1 * y + C2 * z, shift);
        go = CV_DESCALE(C3 * x + C4 * y + C5 * z, shift);
        bo = CV_DESCALE(C6 * x + C7 * y + C8 * z, shift);

        ro = max(0, std::min((int)INV_GAMMA_TAB_SIZE-1, ro));
        go = max(0, std::min((int)INV_GAMMA_TAB_SIZE-1, go));
        bo = max(0, std::min((int)INV_GAMMA_TAB_SIZE-1, bo));

        ro = tab[ro];
        go = tab[go];
        bo = tab[bo];

        dst_row[xx*cn    ] = saturate_cast<uchar>(bo);
        dst_row[xx*cn + 1] = saturate_cast<uchar>(go);
        dst_row[xx*cn + 2] = saturate_cast<uchar>(ro);
        if(cn == 4) dst_row[xx*cn + 3] = 255;
    }

    return n;
}


int row8uLabChoose(const uchar* src_row, uchar *dst_row, int n, bool forward, int blue_idx, bool srgb)
{
    if(forward)
        return row8uRGB2Lab(src_row, dst_row, n, 3, blue_idx, srgb);
    else
        return row8uLab2RGB(src_row, dst_row, n, 3, blue_idx, srgb);
}

int row8uLuvChoose(const uchar* src_row, uchar *dst_row, int n, bool forward, int blue_idx, bool srgb)
{
    if(forward)
        return row8uRGB2Luv(src_row, dst_row, n, 3, blue_idx);
    else
        return row8uLuv2RGB(src_row, dst_row, n, 3, blue_idx, srgb);
}


TEST(Imgproc_ColorLab_Full, bitExactness)
{
    int codes[] = { COLOR_BGR2Lab, COLOR_RGB2Lab, COLOR_LBGR2Lab, COLOR_LRGB2Lab,
                    COLOR_Lab2BGR, COLOR_Lab2RGB, COLOR_Lab2LBGR, COLOR_Lab2LRGB};
    string names[] = { "COLOR_BGR2Lab", "COLOR_RGB2Lab", "COLOR_LBGR2Lab", "COLOR_LRGB2Lab",
                       "COLOR_Lab2BGR", "COLOR_Lab2RGB", "COLOR_Lab2LBGR", "COLOR_Lab2LRGB" };

    // need to be recalculated each time we change Lab algorithms, RNG or test system
    const int nIterations = 8;
    uint32_t hashes[] = {
        0xca7d94c4, 0x34aeb79a, 0x7272c2cf, 0x62c2efed, 0x047cab77, 0x5e8dfb85, 0x10fed613, 0x34d2f4aa,
        0x048bea9a, 0xbbe20ef2, 0x3274e88f, 0x710e9272, 0x9fd6cd59, 0x69d67639, 0x04742095, 0x9ef2b60b,
        0x75b78f5b, 0x3fda9801, 0x374cc472, 0x3239e8ad, 0x94749b2d, 0x9362ac0c, 0xa4d7dd36, 0xe25ef694,
        0x51d1b01d, 0xb0f6e3f5, 0x2b72a228, 0xb7429fa0, 0x799ba6bd, 0x2141d3d2, 0xb4dde471, 0x813b6e0f,
        0x9c029161, 0xb51eb5ec, 0x460c3a09, 0x27724f63, 0xb446c9a8, 0x3adf1b61, 0xe6b0d30f, 0xd1078779,
        0xfaa7525b, 0x5b6ea158, 0xdf3511f7, 0xf01dc02d, 0x5c663841, 0xce611ed4, 0x758ad851, 0xa43c3a1c,
        0xed30f68c, 0xcb6babd9, 0xf38262b5, 0x608cb3db, 0x13425e5a, 0x6dc5fdc7, 0x9519090a, 0x87aa73d0,
        0x8e9bf980, 0x46b98728, 0x0064591c, 0x7e1efc9b, 0xf0ec2465, 0x89a75c8d, 0x0d162fa7, 0xffea7a2f,
    };

    RNG rng(0);
    // blueIdx x srgb x direction
    bool next = true;
    for(int c = 0; next && c < 8; c++)
    {
        int v = c;
        int  blueIdx = (v % 2 != 0) ? 2 : 0; v /=2;
        bool    srgb = (v % 2 == 0); v /= 2;
        bool forward = (v % 2 == 0);

        for(int iter = 0; next && iter < nIterations; iter++)
        {
            Mat probe(256, 256, CV_8UC3), result;
            rng.fill(probe, RNG::UNIFORM, 0, 255, true);

            cvtColor(probe, result, codes[c], 0, ALGO_HINT_ACCURATE);

            uint32_t h = adler32(result);
            uint32_t goodHash = hashes[c*nIterations + iter];

            if(h != goodHash)
            {
                initLabTabs();

                vector<uchar> goldBuf(probe.cols*4);
                uchar* goldRow = &goldBuf[0];
                for(int y = 0; next && y < probe.rows; y++)
                {
                    uchar* probeRow = probe.ptr(y);
                    uchar* resultRow = result.ptr(y);
                    row8uLabChoose(probeRow, goldRow, probe.cols, forward, blueIdx, srgb);

                    for(int x = 0; next && x < probe.cols; x++)
                    {
                        uchar* px = probeRow  + x*3;
                        uchar* gx = goldRow   + x*3;
                        uchar* rx = resultRow + x*3;
                        if(gx[0] != rx[0] || gx[1] != rx[1] || gx[2] != rx[2])
                        {
                            next = false;

                            FAIL() << "Bad accuracy" << endl
                                   << "Conversion code: " << names[c] << endl
                                   << "Iteration: " << iter << endl
                                   << "Hash vs Correct hash: " << h << ", " << goodHash << endl
                                   << "Error in: (" << x << ", " << y << ")" << endl
                                   << "Reference value: " << int(gx[0]) << " " << int(gx[1]) << " " << int(gx[2]) << endl
                                   << "Actual value: "    << int(rx[0]) << " " << int(rx[1]) << " " << int(rx[2]) << endl
                                   << "Src value: " << int(px[0]) << " " << int(px[1]) << " " << int(px[2]) << endl
                                   << "Size: (" << probe.rows << ", " << probe.cols << ")" << endl;

                            break;
                        }
                    }
                }
                if(next)
                    // this place should never be reached
                    throw std::runtime_error("Test system error: hash function mismatch when results are the same");
            }
        }
    }
}


TEST(Imgproc_ColorLuv_Full, bitExactness)
{
    int codes[] = { COLOR_BGR2Luv, COLOR_RGB2Luv, COLOR_LBGR2Luv, COLOR_LRGB2Luv,
                    COLOR_Luv2BGR, COLOR_Luv2RGB, COLOR_Luv2LBGR, COLOR_Luv2LRGB};
    string names[] = { "COLOR_BGR2Luv", "COLOR_RGB2Luv", "COLOR_LBGR2Luv", "COLOR_LRGB2Luv",
                       "COLOR_Luv2BGR", "COLOR_Luv2RGB", "COLOR_Luv2LBGR", "COLOR_Luv2LRGB" };
    /* to be enabled when bit-exactness is done for other codes */
    bool codeEnabled[] = { true, true, false, false, true, true, true, true };

    size_t nCodes = sizeof(codes)/sizeof(codes[0]);

    // need to be recalculated each time we change Luv algorithms, RNG or test system
    const int nIterations = 8;
    uint32_t hashes[] = {
        0x9d4d983a, 0xd3d7b220, 0xd503b661, 0x73581d9b, 0x3beec8a6, 0xea6dfc16, 0xc867f4cd, 0x2c97f43a,
        0x8152fbc9, 0xd7e764a6, 0x5e01f9a3, 0x53e8961e, 0x6a64f1f7, 0x4fa89a44, 0x67096871, 0x4f3bce87,

        0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0,

        0x4bff0e00, 0x76bbff01, 0x80735725, 0xb5e0f137, 0x96abb417, 0xfb2cf5cf, 0x314cf55e, 0x77bde10e,
        0x2ab24209, 0x81caa6F0, 0x3019b8eb, 0x427c505f, 0x5bba7d77, 0xf29cb4d6, 0x760f65ca, 0xf6b4536c,

        0xb5cd0704, 0x82144fd4, 0x4e6f4843, 0x106bc505, 0xf587fc97, 0x3665d9a3, 0x3ea014a8, 0xec664953,
        0x6ec9e59e, 0xf9201e08, 0xf3676fb8, 0xe4e42c10, 0x92d33f64, 0x13b923f7, 0x308f7f50, 0xca98b420,
    };

    RNG rng(0);
    // blueIdx x srgb x direction
    bool next = true;
    for(size_t c = 0; next && c < nCodes; c++)
    {
        if(!codeEnabled[c]) continue;
        size_t v = c;
        int  blueIdx = (v % 2 != 0) ? 2 : 0; v /=2;
        bool    srgb = (v % 2 == 0); v /= 2;
        bool forward = (v % 2 == 0);

        for(int iter = 0; next && iter < nIterations; iter++)
        {
            Mat probe(256, 256, CV_8UC3), result;
            rng.fill(probe, RNG::UNIFORM, 0, 255, true);

            cvtColor(probe, result, codes[c], 0, ALGO_HINT_ACCURATE);

            uint32_t h = adler32(result);
            uint32_t goodHash = hashes[c*nIterations + iter];

            if(h != goodHash)
            {
                initLabTabs();

                vector<uchar> goldBuf(probe.cols*4);
                uchar* goldRow = &goldBuf[0];
                for(int y = 0; next && y < probe.rows; y++)
                {
                    uchar* probeRow = probe.ptr(y);
                    uchar* resultRow = result.ptr(y);

                    row8uLuvChoose(probeRow, goldRow, probe.cols, forward, blueIdx, srgb);

                    for(int x = 0; next && x < probe.cols; x++)
                    {
                        uchar* px = probeRow  + x*3;
                        uchar* gx = goldRow   + x*3;
                        uchar* rx = resultRow + x*3;
                        if(gx[0] != rx[0] || gx[1] != rx[1] || gx[2] != rx[2])
                        {
                            next = false;

                            FAIL() << "Bad accuracy" << endl
                                   << "Conversion code: " << names[c] << endl
                                   << "Iteration: " << iter << endl
                                   << "Hash vs Correct hash: " << h << ", " << goodHash << endl
                                   << "Error in: (" << x << ", " << y << ")" << endl
                                   << "Reference value: " << int(gx[0]) << " " << int(gx[1]) << " " << int(gx[2]) << endl
                                   << "Actual value: "    << int(rx[0]) << " " << int(rx[1]) << " " << int(rx[2]) << endl
                                   << "Src value: " << int(px[0]) << " " << int(px[1]) << " " << int(px[2]) << endl
                                   << "Size: (" << probe.rows << ", " << probe.cols << ")" << endl;

                            break;
                        }
                    }
                }
                if(next)
                    // this place should never be reached
                    throw std::runtime_error("Test system error: hash function mismatch when results are the same");
            }
        }
    }
}


static
void runCvtColorBitExactCheck(ColorConversionCodes code, int inputType, uint32_t hash, Size sz = Size(263, 255), int rngSeed = 0)
{
    RNG rng(rngSeed);

    Mat src(sz, inputType, Scalar::all(0));
    Mat dst;
    rng.fill(src, RNG::UNIFORM, 0, 255, true);

    cv::cvtColor(src, dst, code, 0, ALGO_HINT_ACCURATE);

    uint32_t dst_hash = adler32(dst);

    EXPECT_EQ(hash, dst_hash) << cv::format("0x%08llx", (long long int)dst_hash);

    if (cvtest::debugLevel > 0)
    {
        const ::testing::TestInfo* const test_info = ::testing::UnitTest::GetInstance()->current_test_info();
        CV_Assert(test_info);
        std::string name = (std::string(test_info->test_case_name()) + "--" + test_info->name() + ".xml");
        cv::FileStorage fs(name, cv::FileStorage::WRITE);
        fs << "dst" << dst;
    }
}

TEST(Imgproc_cvtColor_BE, COLOR_RGB2GRAY)  { runCvtColorBitExactCheck(COLOR_RGB2GRAY,  CV_8UC3, 0x416bd44a); }
TEST(Imgproc_cvtColor_BE, COLOR_RGBA2GRAY) { runCvtColorBitExactCheck(COLOR_RGBA2GRAY, CV_8UC3, 0x416bd44a); }
TEST(Imgproc_cvtColor_BE, COLOR_BGR2GRAY)  { runCvtColorBitExactCheck(COLOR_BGR2GRAY,  CV_8UC3, 0x3008c6b8); }
TEST(Imgproc_cvtColor_BE, COLOR_BGRA2GRAY) { runCvtColorBitExactCheck(COLOR_BGRA2GRAY, CV_8UC3, 0x3008c6b8); }

TEST(Imgproc_cvtColor_BE, COLOR_BGR2YUV) { runCvtColorBitExactCheck(COLOR_BGR2YUV, CV_8UC3, 0xc2cbcfda); }
TEST(Imgproc_cvtColor_BE, COLOR_RGB2YUV) { runCvtColorBitExactCheck(COLOR_RGB2YUV, CV_8UC3, 0x4e98e757); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR) { runCvtColorBitExactCheck(COLOR_YUV2BGR, CV_8UC3, 0xb2c62a3f); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB) { runCvtColorBitExactCheck(COLOR_YUV2RGB, CV_8UC3, 0x6d242a3f); }

// packed input
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_NV12) { runCvtColorBitExactCheck(COLOR_YUV2RGB_NV12, CV_8UC1, 0x46a1bb76, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_NV12) { runCvtColorBitExactCheck(COLOR_YUV2BGR_NV12, CV_8UC1, 0x3843bb76, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_NV21) { runCvtColorBitExactCheck(COLOR_YUV2RGB_NV21, CV_8UC1, 0xf3fdf2ea, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_NV21) { runCvtColorBitExactCheck(COLOR_YUV2BGR_NV21, CV_8UC1, 0x6e84f2ea, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_NV12) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_NV12, CV_8UC1, 0xb6a16bd3, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_NV12) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_NV12, CV_8UC1, 0xa8436bd3, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_NV21) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_NV21, CV_8UC1, 0x1c7fa347, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_NV21) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_NV21, CV_8UC1, 0x96f7a347, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_YV12) { runCvtColorBitExactCheck(COLOR_YUV2RGB_YV12, CV_8UC1, 0xc5da1651, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_YV12) { runCvtColorBitExactCheck(COLOR_YUV2BGR_YV12, CV_8UC1, 0x12161651, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_IYUV) { runCvtColorBitExactCheck(COLOR_YUV2RGB_IYUV, CV_8UC1, 0xb4e62ea5, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_IYUV) { runCvtColorBitExactCheck(COLOR_YUV2BGR_IYUV, CV_8UC1, 0xfa632ea5, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_YV12) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_YV12, CV_8UC1, 0x0db4c69f, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_YV12) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_YV12, CV_8UC1, 0x59e1c69f, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_IYUV) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_IYUV, CV_8UC1, 0xfe09def3, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_IYUV) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_IYUV, CV_8UC1, 0x4395def3, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2GRAY_420) { runCvtColorBitExactCheck(COLOR_YUV2GRAY_420, CV_8UC1, 0xf672b440, Size(262, 510)); }

TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_UYVY) { runCvtColorBitExactCheck(COLOR_YUV2RGB_UYVY, CV_8UC2, 0x69bea2c1, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_UYVY) { runCvtColorBitExactCheck(COLOR_YUV2BGR_UYVY, CV_8UC2, 0xdc51a2c1, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_UYVY) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_UYVY, CV_8UC2, 0x851eab45, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_UYVY) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_UYVY, CV_8UC2, 0xf7b1ab45, Size(262, 510)); }

TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_YUY2) { runCvtColorBitExactCheck(COLOR_YUV2RGB_YUY2, CV_8UC2, 0x607e8889, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_YUY2) { runCvtColorBitExactCheck(COLOR_YUV2BGR_YUY2, CV_8UC2, 0xfb148889, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGB_YVYU) { runCvtColorBitExactCheck(COLOR_YUV2RGB_YVYU, CV_8UC2, 0x239b13d4, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGR_YVYU) { runCvtColorBitExactCheck(COLOR_YUV2BGR_YVYU, CV_8UC2, 0x402b13d4, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_YUY2) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_YUY2, CV_8UC2, 0xf6af910d, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_YUY2) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_YUY2, CV_8UC2, 0x9154910d, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2RGBA_YVYU) { runCvtColorBitExactCheck(COLOR_YUV2RGBA_YVYU, CV_8UC2, 0x14481c58, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2BGRA_YVYU) { runCvtColorBitExactCheck(COLOR_YUV2BGRA_YVYU, CV_8UC2, 0x30d81c58, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2GRAY_UYVY) { runCvtColorBitExactCheck(COLOR_YUV2GRAY_UYVY, CV_8UC2, 0x228e669c, Size(262, 510)); }
TEST(Imgproc_cvtColor_BE, COLOR_YUV2GRAY_YUY2) { runCvtColorBitExactCheck(COLOR_YUV2GRAY_YUY2, CV_8UC2, 0x125c62fd, Size(262, 510)); }

TEST(Imgproc_cvtColor_BE, COLOR_RGB2YUV_I420) { runCvtColorBitExactCheck(COLOR_RGB2YUV_I420, CV_8UC3, 0x44bb076a, Size(262, 254)); }
TEST(Imgproc_cvtColor_BE, COLOR_BGR2YUV_I420) { runCvtColorBitExactCheck(COLOR_BGR2YUV_I420, CV_8UC3, 0xf908ff52, Size(262, 254)); }
TEST(Imgproc_cvtColor_BE, COLOR_RGBA2YUV_I420) { runCvtColorBitExactCheck(COLOR_RGBA2YUV_I420, CV_8UC3, 0x44bb076a, Size(262, 254)); }
TEST(Imgproc_cvtColor_BE, COLOR_BGRA2YUV_I420) { runCvtColorBitExactCheck(COLOR_BGRA2YUV_I420, CV_8UC3, 0xf908ff52, Size(262, 254)); }

TEST(Imgproc_cvtColor_BE, COLOR_RGB2YUV_YV12) { runCvtColorBitExactCheck(COLOR_RGB2YUV_YV12, CV_8UC3, 0x1b0d076a, Size(262, 254)); }
TEST(Imgproc_cvtColor_BE, COLOR_BGR2YUV_YV12) { runCvtColorBitExactCheck(COLOR_BGR2YUV_YV12, CV_8UC3, 0xda8aff52, Size(262, 254)); }
TEST(Imgproc_cvtColor_BE, COLOR_RGBA2YUV_YV12) { runCvtColorBitExactCheck(COLOR_RGBA2YUV_YV12, CV_8UC3, 0x1b0d076a, Size(262, 254)); }
TEST(Imgproc_cvtColor_BE, COLOR_BGRA2YUV_YV12) { runCvtColorBitExactCheck(COLOR_BGRA2YUV_YV12, CV_8UC3, 0xda8aff52, Size(262, 254)); }


static void test_Bayer2RGB_EdgeAware_8u(const Mat& src, Mat& dst, int code)
{
    if (dst.empty())
        dst.create(src.size(), CV_MAKETYPE(src.depth(), 3));
    Size size = src.size();
    size.width -= 1;
    size.height -= 1;

    int dcn = dst.channels();
    CV_Assert(dcn == 3);

    int step = (int)src.step;
    const uchar* S = src.ptr<uchar>(1) + 1;
    uchar* D = dst.ptr<uchar>(1) + dcn;

    int start_with_green = code == COLOR_BayerGB2BGR_EA || code == COLOR_BayerGR2BGR_EA ? 1 : 0;
    int blue = code == COLOR_BayerGB2BGR_EA || code == COLOR_BayerBG2BGR_EA ? 1 : 0;

    for (int y = 1; y < size.height; ++y)
    {
        S = src.ptr<uchar>(y) + 1;
        D = dst.ptr<uchar>(y) + dcn;

        if (start_with_green)
        {
            for (int x = 1; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                // red
                D[0] = (S[-1] + S[1]) / 2;
                D[1] = S[0];
                D[2] = (S[-step] + S[step]) / 2;
                if (!blue)
                    std::swap(D[0], D[2]);
            }

            S = src.ptr<uchar>(y) + 2;
            D = dst.ptr<uchar>(y) + 2*dcn;

            for (int x = 2; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                // red
                D[0] = S[0];
                D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[step] - S[-step]) ? (S[step] + S[-step] + 1) : (S[-1] + S[1] + 1)) / 2;
                D[2] = ((S[-step-1] + S[-step+1] + S[step-1] + S[step+1] + 2) / 4);
                if (!blue)
                    std::swap(D[0], D[2]);
            }
        }
        else
        {
            for (int x = 1; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                D[0] = S[0];
                D[1] = (std::abs(S[-1] - S[1]) > std::abs(S[step] - S[-step]) ? (S[step] + S[-step] + 1) : (S[-1] + S[1] + 1)) / 2;
                D[2] = ((S[-step-1] + S[-step+1] + S[step-1] + S[step+1] + 2) / 4);
                if (!blue)
                    std::swap(D[0], D[2]);
            }

            S = src.ptr<uchar>(y) + 2;
            D = dst.ptr<uchar>(y) + 2*dcn;

            for (int x = 2; x < size.width; x += 2, S += 2, D += 2*dcn)
            {
                D[0] = (S[-1] + S[1] + 1) / 2;
                D[1] = S[0];
                D[2] = (S[-step] + S[step] + 1) / 2;
                if (!blue)
                    std::swap(D[0], D[2]);
            }
        }

        D = dst.ptr<uchar>(y + 1) - dcn;
        for (int i = 0; i < dcn; ++i)
        {
            D[i] = D[-dcn + i];
            D[-static_cast<int>(dst.step)+dcn+i] = D[-static_cast<int>(dst.step)+(dcn<<1)+i];
        }

        start_with_green ^= 1;
        blue ^= 1;
    }

    ++size.width;
    uchar* firstRow = dst.ptr(), *lastRow = dst.ptr(size.height);
    size.width *= dcn;
    for (int x = 0; x < size.width; ++x)
    {
        firstRow[x] = firstRow[dst.step + x];
        lastRow[x] = lastRow[-static_cast<int>(dst.step)+x];
    }
}

template <typename T>
static void checkData(const Mat& actual, const Mat& reference, cvtest::TS* ts, const char* type,
    bool& next, const char* bayer_type)
{
    EXPECT_EQ(actual.size(), reference.size());
    EXPECT_EQ(actual.channels(), reference.channels());
    EXPECT_EQ(actual.depth(), reference.depth());

    Size size = reference.size();
    int dcn = reference.channels();
    size.width *= dcn;

    for (int y = 0; y < size.height && next; ++y)
    {
        const T* A = actual.ptr<T>(y);
        const T* R = reference.ptr<T>(y);

        for (int x = 0; x < size.width && next; ++x)
            if (std::abs(A[x] - R[x]) > 1)
            {
                #define SUM cvtest::TS::SUMMARY
                ts->printf(SUM, "\nReference value: %d\n", static_cast<int>(R[x]));
                ts->printf(SUM, "Actual value: %d\n", static_cast<int>(A[x]));
                ts->printf(SUM, "(y, x): (%d, %d)\n", y, x / reference.channels());
                ts->printf(SUM, "Channel pos: %d\n", x % reference.channels());
                ts->printf(SUM, "Pattern: %s\n", type);
                ts->printf(SUM, "Bayer image type: %s", bayer_type);
                #undef SUM

                Mat diff;
                absdiff(actual, reference, diff);
                EXPECT_EQ(countNonZero(diff.reshape(1) > 1), 0);

                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                ts->set_gtest_status();

                next = false;
            }
    }
}

TEST(ImgProc_BayerEdgeAwareDemosaicing, accuracy)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    const std::string image_name = "lena.png";
    const std::string parent_path = string(ts->get_data_path()) + "/cvtcolor_strict/";

    Mat src, bayer;
    std::string full_path = parent_path + image_name;
    src = imread(full_path, IMREAD_UNCHANGED);

    if (src.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISSING_TEST_DATA);
        ts->printf(cvtest::TS::SUMMARY, "No input image\n");
        ts->set_gtest_status();
        return;
    }

    /*
    COLOR_BayerBG2BGR_EA = 127,
    COLOR_BayerGB2BGR_EA = 128,
    COLOR_BayerRG2BGR_EA = 129,
    COLOR_BayerGR2BGR_EA = 130,
    */

    bool next = true;
    const char* types[] = { "bg", "gb", "rg", "gr" };
    for (int i = 0; i < 4 && next; ++i)
    {
        calculateBayerPattern<uchar, CV_8U>(src, bayer, types[i]);
        Mat reference;
        test_Bayer2RGB_EdgeAware_8u(bayer, reference, COLOR_BayerBG2BGR_EA + i);

        for (int t = 0; t <= 1; ++t)
        {
            if (t == 1)
                calculateBayerPattern<unsigned short int, CV_16U>(src, bayer, types[i]);

            CV_Assert(!bayer.empty() && (bayer.type() == CV_8UC1 || bayer.type() == CV_16UC1));

            Mat actual;
            cv::demosaicing(bayer, actual, COLOR_BayerBG2BGR_EA + i);

            if (t == 0)
                checkData<unsigned char>(actual, reference, ts, types[i], next, "CV_8U");
            else
            {
                Mat tmp;
                reference.convertTo(tmp, CV_16U);
                checkData<unsigned short int>(actual, tmp, ts, types[i], next, "CV_16U");
            }
        }
    }
}

TEST(ImgProc_Bayer2RGBA, accuracy)
{
    cvtest::TS* ts = cvtest::TS::ptr();
    Mat raw = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    Mat rgb, reference;

    CV_Assert(raw.channels() == 1);
    CV_Assert(raw.depth() == CV_8U);
    CV_Assert(!raw.empty());

    for (int code = COLOR_BayerBG2BGR; code <= COLOR_BayerGR2BGR; ++code)
    {
        cvtColor(raw, rgb, code);
        cvtColor(rgb, reference, COLOR_BGR2BGRA);

        Mat actual;
        cvtColor(raw, actual, code, 4);

        EXPECT_EQ(reference.size(), actual.size());
        EXPECT_EQ(reference.depth(), actual.depth());
        EXPECT_EQ(reference.channels(), actual.channels());

        Size ssize = raw.size();
        int cn = reference.channels();
        ssize.width *= cn;
        bool next = true;
        for (int y = 0; y < ssize.height && next; ++y)
        {
            const uchar* rD = reference.ptr<uchar>(y);
            const uchar* D = actual.ptr<uchar>(y);
            for (int x = 0; x < ssize.width && next; ++x)
                if (abs(rD[x] - D[x]) >= 1)
                {
                    next = false;
                    ts->printf(cvtest::TS::SUMMARY, "Error in: (%d, %d)\n", x / cn,  y);
                    ts->printf(cvtest::TS::SUMMARY, "Reference value: %d\n", rD[x]);
                    ts->printf(cvtest::TS::SUMMARY, "Actual value: %d\n", D[x]);
                    ts->printf(cvtest::TS::SUMMARY, "Src value: %d\n", raw.ptr<uchar>(y)[x]);
                    ts->printf(cvtest::TS::SUMMARY, "Size: (%d, %d)\n", reference.rows, reference.cols);

                    Mat diff;
                    absdiff(actual, reference, diff);
                    EXPECT_EQ(countNonZero(diff.reshape(1) > 1), 0);

                    ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
                    ts->set_gtest_status();
                }
        }
    }
}

TEST(ImgProc_BGR2RGBA, regression_8696)
{
    Mat src(Size(80, 10), CV_8UC4);
    src.setTo(Scalar(255, 0, 200, 100));

    Mat dst;
    cvtColor(src, dst, COLOR_BGR2BGRA);

    EXPECT_DOUBLE_EQ(cvtest::norm(dst - src, NORM_INF), 0.);
}

TEST(ImgProc_BGR2RGBA, 3ch24ch)
{
    Mat src(Size(80, 10), CV_8UC3);
    src.setTo(Scalar(200, 0, 200));

    Mat dst;
    cvtColor(src, dst, COLOR_BGR2BGRA);

    Mat expected(Size(80, 10), CV_8UC4);
    expected.setTo(Scalar(80, 0, 200, 255));

    EXPECT_DOUBLE_EQ(cvtest::norm(expected - dst, NORM_INF), 0.);
}

TEST(ImgProc_RGB2YUV, regression_13668)
{
    Mat src(Size(32, 4), CV_8UC3, Scalar(9, 250,  82));  // Ensure that SIMD code path works
    Mat dst;
    cvtColor(src, dst, COLOR_RGB2YUV);
    Vec3b res = dst.at<Vec3b>(0, 0);
    Vec3b ref(159, 90, 0);
    EXPECT_EQ(res, ref);
}

TEST(ImgProc_cvtColorTwoPlane, y_plane_padding_differs_from_uv_plane_padding_17036)
{
    RNG &rng = theRNG();

    std::vector<uchar> y_reference(640 * 480);
    std::vector<uchar> uv_reference(640 * 240);
    std::vector<uchar> y_padded(700 * 480);
    std::vector<uchar> uv_padded(700 * 240);

    Mat y_reference_mat(480, 640, CV_8UC1, y_reference.data());
    Mat uv_reference_mat(240, 320, CV_8UC2, uv_reference.data());
    Mat y_padded_mat(480, 640, CV_8UC1, y_padded.data(), 700);
    Mat uv_padded_mat(240, 320, CV_8UC2, uv_padded.data(), 700);

    rng.fill(y_reference_mat, RNG::UNIFORM, 16, 235 + 1);
    rng.fill(uv_reference_mat, RNG::UNIFORM, 16, 240 + 1);

    y_reference_mat.copyTo(y_padded_mat(Rect(0, 0, y_reference_mat.cols, y_reference_mat.rows)));
    uv_reference_mat.copyTo(uv_padded_mat(Rect(0, 0, uv_reference_mat.cols, uv_reference_mat.rows)));

    Mat rgb_reference_mat, rgb_y_padded_mat, rgb_uv_padded_mat;

    cvtColorTwoPlane(y_reference_mat, uv_reference_mat, rgb_reference_mat, COLOR_YUV2RGB_NV21);
    cvtColorTwoPlane(y_padded_mat, uv_reference_mat, rgb_y_padded_mat, COLOR_YUV2RGB_NV21);
    cvtColorTwoPlane(y_reference_mat, uv_padded_mat, rgb_uv_padded_mat, COLOR_YUV2RGB_NV21);

    EXPECT_DOUBLE_EQ(cvtest::norm(rgb_reference_mat, rgb_y_padded_mat, NORM_INF), .0);
    EXPECT_DOUBLE_EQ(cvtest::norm(rgb_reference_mat, rgb_uv_padded_mat, NORM_INF), .0);
}

TEST(ImgProc_RGB2Lab, NaN_21111)
{
    const float kNaN = std::numeric_limits<float>::quiet_NaN();
    cv::Mat3f src(1, 111, Vec3f::all(kNaN)), dst;
    // Make some entries with only one NaN.
    src(0, 0) = src(0, 27) = src(0, 81) = src(0, 108) = cv::Vec3f(0, 0, kNaN);
    src(0, 1) = src(0, 28) = src(0, 82) = src(0, 109) = cv::Vec3f(0, kNaN, 0);
    src(0, 2) = src(0, 29) = src(0, 83) = src(0, 110) = cv::Vec3f(kNaN, 0, 0);
    EXPECT_NO_THROW(cvtColor(src, dst, COLOR_RGB2Lab));
    EXPECT_NO_THROW(cvtColor(src, dst, COLOR_RGB2Luv));
    EXPECT_NO_THROW(cvtColor(src, dst, COLOR_Luv2RGB));

#if 0  // no NaN propagation guarantee
    for (int i = 0; i < 20; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_TRUE(cvIsNaN(dst(0, i)[j]));
        }
    }
#endif
}

// See https://github.com/opencv/opencv/issues/25971
// If num of channels is not suitable for selected cv::ColorConversionCodes,
// e.code must be cv::Error::BadNumChannels.
TEST(ImgProc_cvtColor_InvalidNumOfChannels, regression_25971)
{
    try {
        cv::Mat src = cv::Mat::zeros(100, 100, CV_8UC1);
        cv::Mat dst;
        EXPECT_THROW(cv::cvtColor(src, dst, COLOR_RGB2GRAY), cv::Exception);
    }catch(const cv::Exception& e) {
        EXPECT_EQ(e.code, cv::Error::BadNumChannels);
    }catch(...) {
        FAIL() << "Unexpected exception is happened.";
    }
}

}} // namespace
