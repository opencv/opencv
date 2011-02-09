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

class CV_ThreshTest : public cvtest::ArrayTest
{
public:
    CV_ThreshTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int thresh_type;
    float thresh_val;
    float max_val;
};


CV_ThreshTest::CV_ThreshTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    optional_mask = false;
    element_wise_relative_error = true;
}


void CV_ThreshTest::get_test_array_types_and_sizes( int test_case_idx,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int depth = cvtest::randInt(rng) % 2, cn = cvtest::randInt(rng) % 4 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : CV_32F;

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth,cn);
    thresh_type = cvtest::randInt(rng) % 5;

    if( depth == CV_8U )
    {
        thresh_val = (float)(cvtest::randReal(rng)*350. - 50.);
        max_val = (float)(cvtest::randReal(rng)*350. - 50.);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = 255;
    }
    else
    {
        thresh_val = (float)(cvtest::randReal(rng)*1000. - 500.);
        max_val = (float)(cvtest::randReal(rng)*1000. - 500.);
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


static void test_threshold( const Mat& _src, Mat& _dst,
                            float thresh, float maxval, int thresh_type )
{
    int i, j;
    int depth = _src.depth(), cn = _src.channels();
    int width_n = _src.cols*cn, height = _src.rows;
    int ithresh = cvFloor(thresh), ithresh2, imaxval = cvRound(maxval);
    const uchar* src = _src.data;
    uchar* dst = _dst.data;
    size_t srcstep = _src.step, dststep = _dst.step;
    
    ithresh2 = saturate_cast<uchar>(ithresh);
    imaxval = saturate_cast<uchar>(imaxval);

    assert( depth == CV_8U || depth == CV_32F );
    
    switch( thresh_type )
    {
    case CV_THRESH_BINARY:
        for( i = 0; i < height; i++, src += srcstep, dst += dststep )
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
        for( i = 0; i < height; i++, src += srcstep, dst += dststep )
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
        for( i = 0; i < height; i++, src += srcstep, dst += dststep )
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
        for( i = 0; i < height; i++, src += srcstep, dst += dststep )
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
        for( i = 0; i < height; i++, src += srcstep, dst += dststep )
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
    test_threshold( test_mat[INPUT][0], test_mat[REF_OUTPUT][0],
                   thresh_val, max_val, thresh_type );
}

TEST(Imgproc_Threshold, accuracy) { CV_ThreshTest test; test.safe_run(); }

