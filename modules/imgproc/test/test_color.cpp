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
    bool test_cpp;
    int hue_range;
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

    test_cpp = false;
    hue_range = 0;
    blue_idx = 0;
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

    types[INPUT][0] = CV_MAKETYPE(depth, cn);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth, 3);
    if( test_array[OUTPUT].size() > 1 )
        types[OUTPUT][1] = types[REF_OUTPUT][1] = CV_MAKETYPE(depth, cn);

    inplace = cn == 3 && cvtest::randInt(rng) % 2 != 0;
    test_cpp = (cvtest::randInt(rng) & 256) == 0;
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

    assert( (cn == 3 || cn == 4) && (dst_cn == 3 || dst_cn == 1) );

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
            assert(0);
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

        assert( cn == 3 || cn == 4 );

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
                assert(0);
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
    fwd_code = CV_##fwd; inv_code = CV_##inv;   \
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


double CV_ColorGrayTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 2 : depth == CV_16U ? 16 : 1e-5;
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
        fwd_code = CV_BGR2YCrCb, inv_code = CV_YCrCb2BGR;
    else
        fwd_code = CV_RGB2YCrCb, inv_code = CV_YCrCb2RGB;
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
            fwd_code = CV_BGR2HSV_FULL, inv_code = CV_HSV2BGR_FULL;
        else
            fwd_code = CV_RGB2HSV_FULL, inv_code = CV_HSV2RGB_FULL;
        hue_range = 256;
    }
    else
    {
        if( blue_idx == 0 )
            fwd_code = CV_BGR2HSV, inv_code = CV_HSV2BGR;
        else
            fwd_code = CV_RGB2HSV, inv_code = CV_HSV2RGB;
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
        fwd_code = CV_BGR2HLS, inv_code = CV_HLS2BGR;
    else
        fwd_code = CV_RGB2HLS, inv_code = CV_HLS2RGB;
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
        fwd_code = CV_BGR2XYZ, inv_code = CV_XYZ2BGR;
    else
        fwd_code = CV_RGB2XYZ, inv_code = CV_XYZ2RGB;
}


double CV_ColorXYZTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? (j == 0 ? 2 : 8) : depth == CV_16U ? (j == 0 ? 64 : 128) : 1e-1;
}


void CV_ColorXYZTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
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
    int depth = test_mat[INPUT][0].depth();
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


//// rgb <=> L*a*b*
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

    if( blue_idx == 0 )
        fwd_code = CV_LBGR2Lab, inv_code = CV_Lab2LBGR;
    else
        fwd_code = CV_LRGB2Lab, inv_code = CV_Lab2LRGB;
}


double CV_ColorLabTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 16 : depth == CV_16U ? 32 : 1e-3;
}


static const double _1_3 = 0.333333333333;
const static float _1_3f = static_cast<float>(_1_3);


void CV_ColorLabTest::convert_row_bgr2abc_32f_c3(const float* src_row, float* dst_row, int n)
{
    int depth = test_mat[INPUT][0].depth();
    float Lscale = depth == CV_8U ? 255.f/100.f : depth == CV_16U ? 65535.f/100.f : 1.f;
    float ab_bias = depth == CV_8U ? 128.f : depth == CV_16U ? 32768.f : 0.f;
    float M[9];

    for (int j = 0; j < 9; j++ )
        M[j] = (float)RGB2XYZ[j];

    for (int x = 0; x < n*3; x += 3)
    {
        float R = src_row[x + 2];
        float G = src_row[x + 1];
        float B = src_row[x];

        float X = (R * M[0] + G * M[1] + B * M[2]) / Xn;
        float Y = R * M[3] + G * M[4] + B * M[5];
        float Z = (R * M[6] + G * M[7] + B * M[8]) / Zn;
        float fX = X > 0.008856f ? pow(X, _1_3f) :
            (7.787f * X + 16.f / 116.f);
        float fZ = Z > 0.008856f ? pow(Z, _1_3f):
            (7.787f * Z + 16.f / 116.f);

        float L = 0.0f, fY = 0.0f;
        if (Y > 0.008856f)
        {
            fY = pow(Y, _1_3f);
            L = 116.f * fY - 16.f;
        }
        else
        {
            fY = 7.787f * Y + 16.f / 116.f;
            L = 903.3f * Y;
        }

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

    static const float lthresh = 903.3f * 0.008856f;
    static const float thresh = 7.787f * 0.008856f + 16.0f / 116.0f;
    for (int x = 0, end = n * 3; x < end; x += 3)
    {
        float L = src_row[x] * Lscale;
        float a = src_row[x + 1] - ab_bias;
        float b = src_row[x + 2] - ab_bias;

        float FY = 0.0f, Y = 0.0f;
        if (L <= lthresh)
        {
            Y = L / 903.3f;
            FY = 7.787f * Y + 16.0f / 116.0f;
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
            if (FXZ[k] <= thresh)
                FXZ[k] = (FXZ[k] - 16.0f / 116.0f) / 7.787f;
            else
                FXZ[k] = FXZ[k] * FXZ[k] * FXZ[k];
        }
        float X = FXZ[0] * Xn;
        float Z = FXZ[1] * Zn;

        float R = M[0] * X + M[1] * Y + M[2] * Z;
        float G = M[3] * X + M[4] * Y + M[5] * Z;
        float B = M[6] * X + M[7] * Y + M[8] * Z;

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

    if( blue_idx == 0 )
        fwd_code = CV_LBGR2Luv, inv_code = CV_Luv2LBGR;
    else
        fwd_code = CV_LRGB2Luv, inv_code = CV_Luv2LRGB;
}


double CV_ColorLuvTest::get_success_error_level( int /*test_case_idx*/, int i, int j )
{
    int depth = test_mat[i][j].depth();
    return depth == CV_8U ? 48 : depth == CV_16U ? 32 : 5e-2;
}


void CV_ColorLuvTest::convert_row_bgr2abc_32f_c3( const float* src_row, float* dst_row, int n )
{
    int depth = test_mat[INPUT][0].depth();
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
    int depth = test_mat[INPUT][0].depth();
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


void CV_ColorRGBTest::convert_forward( const Mat& src, Mat& dst )
{
    int depth = src.depth(), cn = src.channels();
/*#if defined _DEBUG || defined DEBUG
    int dst_cn = CV_MAT_CN(dst->type);
#endif*/
    int i, j, cols = src.cols;
    int g_rshift = dst_bits == 16 ? 2 : 3;
    int r_lshift = dst_bits == 16 ? 11 : 10;

    //assert( (cn == 3 || cn == 4) && (dst_cn == 3 || (dst_cn == 2 && depth == CV_8U)) );

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
            assert(0);
        }
    }
}


void CV_ColorRGBTest::convert_backward( const Mat& /*src*/, const Mat& src, Mat& dst )
{
    int depth = src.depth(), cn = dst.channels();
/*#if defined _DEBUG || defined DEBUG
    int src_cn = CV_MAT_CN(src->type);
#endif*/
    int i, j, cols = src.cols;
    int g_lshift = dst_bits == 16 ? 2 : 3;
    int r_rshift = dst_bits == 16 ? 11 : 10;

    //assert( (cn == 3 || cn == 4) && (src_cn == 3 || (src_cn == 2 && depth == CV_8U)) );

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
            assert(0);
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
    fwd_code = CV_BayerBG2BGR;
    inv_code = -1;
}


void CV_ColorBayerTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    CV_ColorCvtBaseTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );

    types[INPUT][0] = CV_MAT_DEPTH(types[INPUT][0]);
    types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(CV_MAT_DEPTH(types[INPUT][0]), 3);
    inplace = false;

    fwd_code = cvtest::randInt(rng)%4 + CV_BayerBG2BGR;
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


template<typename T>
static void bayer2BGR_(const Mat& src, Mat& dst, int code)
{
    int i, j, cols = src.cols - 2;
    int bi = 0;
    int step = (int)(src.step/sizeof(T));

    if( code == CV_BayerRG2BGR || code == CV_BayerGR2BGR )
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
        CV_Error(CV_StsUnsupportedFormat, "");
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

TEST(Imgproc_ColorBayer, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();

    Mat given = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    Mat gold = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_gold.png", IMREAD_UNCHANGED);
    Mat result;

    CV_Assert( !given.empty() && !gold.empty() );

    cvtColor(given, result, CV_BayerBG2GRAY);

    EXPECT_EQ(gold.type(), result.type());
    EXPECT_EQ(gold.cols, result.cols);
    EXPECT_EQ(gold.rows, result.rows);

    Mat diff;
    absdiff(gold, result, diff);

    EXPECT_EQ(0, countNonZero(diff.reshape(1) > 1));
}

TEST(Imgproc_ColorBayerVNG, regression)
{
    cvtest::TS* ts = cvtest::TS::ptr();

    Mat given = imread(string(ts->get_data_path()) + "/cvtcolor/bayer_input.png", IMREAD_GRAYSCALE);
    string goldfname = string(ts->get_data_path()) + "/cvtcolor/bayerVNG_gold.png";
    Mat gold = imread(goldfname, IMREAD_UNCHANGED);
    Mat result;

    CV_Assert( !given.empty() );

    cvtColor(given, result, CV_BayerBG2BGR_VNG, 3);

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
        cvtColor(bayer, dst, CV_BayerBG2BGR_VNG + i);

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

    for (int y = 0; y < ssize.height && next; ++y)
    {
        const float* rD = reference.ptr<float>(y);
        const float* D = actual.ptr<float>(y);
        for (int x = 0; x < ssize.width && next; ++x)
            if (fabs(rD[x] - D[x]) > 0.0001f)
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

    RNG& rng = cvtest::TS::ptr()->get_rng();
    int blueInd = rng.uniform(0., 1.) > 0.5 ? 0 : 2;
    bool srgb = rng.uniform(0., 1.) > 0.5;

    // Convert test image to LAB
    cv::Mat lab;
    int forward_code = blueInd ? srgb ? CV_BGR2Lab : CV_LBGR2Lab : srgb ? CV_RGB2Lab : CV_LRGB2Lab;
    int inverse_code = blueInd ? srgb ? CV_Lab2BGR : CV_Lab2LBGR : srgb ? CV_Lab2RGB : CV_Lab2LRGB;
    cv::cvtColor(src, lab, forward_code);
    // Convert LAB image back to BGR(RGB)
    cv::Mat recons;
    cv::cvtColor(lab, recons, inverse_code);

    validateResult(src, recons, src, forward_code);
}

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

    int start_with_green = code == CV_BayerGB2BGR_EA || code == CV_BayerGR2BGR_EA ? 1 : 0;
    int blue = code == CV_BayerGB2BGR_EA || code == CV_BayerBG2BGR_EA ? 1 : 0;

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
        test_Bayer2RGB_EdgeAware_8u(bayer, reference, CV_BayerBG2BGR_EA + i);

        for (int t = 0; t <= 1; ++t)
        {
            if (t == 1)
                calculateBayerPattern<unsigned short int, CV_16U>(src, bayer, types[i]);

            CV_Assert(!bayer.empty() && (bayer.type() == CV_8UC1 || bayer.type() == CV_16UC1));

            Mat actual;
            cv::demosaicing(bayer, actual, CV_BayerBG2BGR_EA + i);

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

    for (int code = CV_BayerBG2BGR; code <= CV_BayerGR2BGR; ++code)
    {
        cvtColor(raw, rgb, code);
        cvtColor(rgb, reference, CV_BGR2BGRA);

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
