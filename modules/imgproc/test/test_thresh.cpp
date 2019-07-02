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
    double thresh_val;
    double max_val;
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
    int depth = cvtest::randInt(rng) % 5, cn = cvtest::randInt(rng) % 4 + 1;
    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    depth = depth == 0 ? CV_8U : depth == 1 ? CV_16S : depth == 2 ? CV_16U : depth == 3 ? CV_32F : CV_64F;

    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_MAKETYPE(depth,cn);
    thresh_type = cvtest::randInt(rng) % 5;

    if( depth == CV_8U )
    {
        thresh_val = (cvtest::randReal(rng)*350. - 50.);
        max_val = (cvtest::randReal(rng)*350. - 50.);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = 255.f;
    }
    else if( depth == CV_16S )
    {
        double min_val = SHRT_MIN-100.f;
        max_val = SHRT_MAX+100.f;
        thresh_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        max_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = (double)SHRT_MAX;
    }
    else if( depth == CV_16U )
    {
        double min_val = -100.f;
        max_val = USHRT_MAX+100.f;
        thresh_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        max_val = (cvtest::randReal(rng)*(max_val - min_val) + min_val);
        if( cvtest::randInt(rng)%4 == 0 )
            max_val = (double)USHRT_MAX;
    }
    else
    {
        thresh_val = (cvtest::randReal(rng)*1000. - 500.);
        max_val = (cvtest::randReal(rng)*1000. - 500.);
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
                            double thresh, double maxval, int thresh_type )
{
    int i, j;
    int depth = _src.depth(), cn = _src.channels();
    int width_n = _src.cols*cn, height = _src.rows;
    int ithresh = cvFloor(thresh);
    int imaxval, ithresh2;

    if( depth == CV_8U )
    {
        ithresh2 = saturate_cast<uchar>(ithresh);
        imaxval = saturate_cast<uchar>(maxval);
    }
    else if( depth == CV_16S )
    {
        ithresh2 = saturate_cast<short>(ithresh);
        imaxval = saturate_cast<short>(maxval);
    }
    else if( depth == CV_16U )
    {
        ithresh2 = saturate_cast<ushort>(ithresh);
        imaxval = saturate_cast<ushort>(maxval);
    }
    else
    {
        ithresh2 = cvRound(ithresh);
        imaxval = cvRound(maxval);
    }

    assert( depth == CV_8U || depth == CV_16S || depth == CV_16U || depth == CV_32F || depth == CV_64F );

    switch( thresh_type )
    {
    case CV_THRESH_BINARY:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (short)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (ushort)(src[j] > ithresh ? imaxval : 0);
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (float)(src[j] > thresh ? maxval : 0.f);
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = src[j] > thresh ? maxval : 0.0;
            }
        }
        break;
    case CV_THRESH_BINARY_INV:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (uchar)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (short)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (ushort)(src[j] > ithresh ? 0 : imaxval);
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = (float)(src[j] > thresh ? 0.f : maxval);
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                    dst[j] = src[j] > thresh ? 0.0 : maxval;
            }
        }
        break;
    case CV_THRESH_TRUNC:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (ushort)(s > ithresh ? ithresh2 : s);
                }
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = (float)(s > thresh ? thresh : s);
                }
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = s > thresh ? thresh : s;
                }
            }
        }
        break;
    case CV_THRESH_TOZERO:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (ushort)(s > ithresh ? s : 0);
                }
            }
            else if( depth == CV_32F )
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = s > thresh ? s : 0.f;
                }
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = s > thresh ? s : 0.0;
                }
            }
        }
        break;
    case CV_THRESH_TOZERO_INV:
        for( i = 0; i < height; i++ )
        {
            if( depth == CV_8U )
            {
                const uchar* src = _src.ptr<uchar>(i);
                uchar* dst = _dst.ptr<uchar>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (uchar)(s > ithresh ? 0 : s);
                }
            }
            else if( depth == CV_16S )
            {
                const short* src = _src.ptr<short>(i);
                short* dst = _dst.ptr<short>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (short)(s > ithresh ? 0 : s);
                }
            }
            else if( depth == CV_16U )
            {
                const ushort* src = _src.ptr<ushort>(i);
                ushort* dst = _dst.ptr<ushort>(i);
                for( j = 0; j < width_n; j++ )
                {
                    int s = src[j];
                    dst[j] = (ushort)(s > ithresh ? 0 : s);
                }
            }
            else if (depth == CV_32F)
            {
                const float* src = _src.ptr<float>(i);
                float* dst = _dst.ptr<float>(i);
                for( j = 0; j < width_n; j++ )
                {
                    float s = src[j];
                    dst[j] = s > thresh ? 0.f : s;
                }
            }
            else
            {
                const double* src = _src.ptr<double>(i);
                double* dst = _dst.ptr<double>(i);
                for( j = 0; j < width_n; j++ )
                {
                    double s = src[j];
                    dst[j] = s > thresh ? 0.0 : s;
                }
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

BIGDATA_TEST(Imgproc_Threshold, huge)
{
    Mat m(65000, 40000, CV_8U);
    ASSERT_FALSE(m.isContinuous());

    uint64 i, n = (uint64)m.rows*m.cols;
    for( i = 0; i < n; i++ )
        m.data[i] = (uchar)(i & 255);

    cv::threshold(m, m, 127, 255, cv::THRESH_BINARY);
    int nz = cv::countNonZero(m);  // FIXIT 'int' is not enough here (overflow is possible with other inputs)
    ASSERT_EQ((uint64)nz, n / 2);
}

}} // namespace
