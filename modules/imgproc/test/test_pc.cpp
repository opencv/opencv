/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
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
//   * The name of the copyright holders may not be used to endorse or promote products
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
#include "opencv2/core/core_c.h"

#define CV_DXT_MUL_CONJ 8

namespace opencv_test { namespace {

/// phase correlation
class CV_PhaseCorrelatorTest : public cvtest::ArrayTest
{
public:
    CV_PhaseCorrelatorTest();
protected:
    void run( int );
};

CV_PhaseCorrelatorTest::CV_PhaseCorrelatorTest() {}

void CV_PhaseCorrelatorTest::run( int )
{
    ts->set_failed_test_info(cvtest::TS::OK);

    Mat r1 = Mat::ones(Size(129, 128), CV_64F);
    Mat r2 = Mat::ones(Size(129, 128), CV_64F);

    double expectedShiftX = -10.0;
    double expectedShiftY = -20.0;

    // draw 10x10 rectangles @ (100, 100) and (90, 80) should see ~(-10, -20) shift here...
    cv::rectangle(r1, Point(100, 100), Point(110, 110), Scalar(0, 0, 0), cv::FILLED);
    cv::rectangle(r2, Point(90, 80), Point(100, 90), Scalar(0, 0, 0), cv::FILLED);

    Mat hann;
    createHanningWindow(hann, r1.size(), CV_64F);
    Point2d phaseShift = phaseCorrelate(r1, r2, hann);

    // test accuracy should be less than 1 pixel...
    if(std::abs(expectedShiftX - phaseShift.x) >= 1 || std::abs(expectedShiftY - phaseShift.y) >= 1)
    {
         ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    }
}

TEST(Imgproc_PhaseCorrelatorTest, accuracy) { CV_PhaseCorrelatorTest test; test.safe_run(); }

TEST(Imgproc_PhaseCorrelatorTest, accuracy_real_img)
{
    Mat img = imread(cvtest::TS::ptr()->get_data_path() + "shared/airplane.png", IMREAD_GRAYSCALE);
    img.convertTo(img, CV_64FC1);

    const int xLen = 129;
    const int yLen = 129;
    const int xShift = 40;
    const int yShift = 14;

    Mat roi1 = img(Rect(xShift, yShift, xLen, yLen));
    Mat roi2 = img(Rect(0, 0, xLen, yLen));

    Mat hann;
    createHanningWindow(hann, roi1.size(), CV_64F);
    Point2d phaseShift = phaseCorrelate(roi1, roi2, hann);

    ASSERT_NEAR(phaseShift.x, (double)xShift, 1.);
    ASSERT_NEAR(phaseShift.y, (double)yShift, 1.);
}

TEST(Imgproc_PhaseCorrelatorTest, accuracy_1d_odd_fft) {
    Mat r1 = Mat::ones(Size(129, 1), CV_64F)*255; // 129 will be completed to 135 before FFT
    Mat r2 = Mat::ones(Size(129, 1), CV_64F)*255;

    const int xShift = 10;

    for(int i = 6; i < 20; i++)
    {
        r1.at<double>(i) = 1;
        r2.at<double>(i + xShift) = 1;
    }

    Point2d phaseShift = phaseCorrelate(r1, r2);

    ASSERT_NEAR(phaseShift.x, (double)xShift, 1.);
}

TEST(Imgproc_PhaseCorrelatorTest, float32_overflow) {
    // load
    Mat im = imread(cvtest::TS::ptr()->get_data_path() + "shared/baboon.png", IMREAD_GRAYSCALE);
    ASSERT_EQ(im.type(), CV_8UC1);

    // convert to 32F, scale values as if original image was 16U
    constexpr auto u8Max = std::numeric_limits<std::uint8_t>::max();
    constexpr auto u16Max = std::numeric_limits<std::uint16_t>::max();
    im.convertTo(im, CV_32FC1, double(u16Max) / double(u8Max));

    // enlarge and create ROIs
    const auto w = im.cols * 5;
    const auto h = im.rows * 5;
    const auto roiW = (w * 2) / 3; // 50% overlap
    Mat imLarge;
    resize(im, imLarge, { w, h });
    const auto roiLeft = imLarge(Rect(0, 0, roiW, h));
    const auto roiRight = imLarge(Rect(w - roiW, 0, roiW, h));

    // correlate
    double response = 0.0;
    Point2d phaseShift = phaseCorrelate(roiLeft, roiRight, cv::noArray(), &response);
    ASSERT_TRUE(std::isnormal(phaseShift.x) || 0.0 == phaseShift.x);
    ASSERT_TRUE(std::isnormal(phaseShift.y) || 0.0 == phaseShift.y);
    ASSERT_TRUE(std::isnormal(response) || 0.0 == response);
    EXPECT_NEAR(std::abs(phaseShift.x), w / 3.0, 1.0);
    EXPECT_NEAR(std::abs(phaseShift.y), 0.0, 1.0);
}

////////////////////// DivSpectrums ////////////////////////
class CV_DivSpectrumsTest : public cvtest::ArrayTest
{
public:
    CV_DivSpectrumsTest();
protected:
    void run_func();
    void get_test_array_types_and_sizes( int, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void prepare_to_validation( int test_case_idx );
    int flags;
};


CV_DivSpectrumsTest::CV_DivSpectrumsTest() : flags(0)
{
    // Allocate test matrices.
    test_array[INPUT].push_back(NULL);  // first input DFT as a CCS-packed array or complex matrix.
    test_array[INPUT].push_back(NULL);  // second input DFT as a CCS-packed array or complex matrix.
    test_array[OUTPUT].push_back(NULL);  // output DFT as a complex matrix.
    test_array[REF_OUTPUT].push_back(NULL);  // reference output DFT as a complex matrix.
    test_array[TEMP].push_back(NULL);  // first input DFT converted to a complex matrix.
    test_array[TEMP].push_back(NULL);  // second input DFT converted to a complex matrix.
    test_array[TEMP].push_back(NULL);  // output DFT as a CCV-packed array.
}

void CV_DivSpectrumsTest::get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    cvtest::ArrayTest::get_test_array_types_and_sizes(test_case_idx, sizes, types);
    RNG& rng = ts->get_rng();

    // Get the flag of the input.
    const int rand_int_flags = cvtest::randInt(rng);
    flags = rand_int_flags & (CV_DXT_MUL_CONJ | DFT_ROWS);

    // Get input type.
    const int rand_int_type = cvtest::randInt(rng);
    int type;

    if (rand_int_type % 4)
    {
        type = CV_32FC1;
    }
    else if (rand_int_type % 4 == 1)
    {
        type = CV_32FC2;
    }
    else if (rand_int_type % 4 == 2)
    {
        type = CV_64FC1;
    }
    else
    {
        type = CV_64FC2;
    }

    for( size_t i = 0; i < types.size(); i++ )
    {
        for( size_t j = 0; j < types[i].size(); j++ )
        {
            types[i][j] = type;
        }
    }

    // Inputs are CCS-packed arrays.  Prepare outputs and temporary inputs as complex matrices.
    if( type == CV_32FC1 || type == CV_64FC1 )
    {
        types[OUTPUT][0] += CV_DEPTH_MAX;
        types[REF_OUTPUT][0] += CV_DEPTH_MAX;
        types[TEMP][0] += CV_DEPTH_MAX;
        types[TEMP][1] += CV_DEPTH_MAX;
    }
}

/// Helper function to convert a ccs array of depth_t into a complex matrix.
template<typename depth_t>
static void convert_from_ccs_helper( const Mat& src0, const Mat& src1, Mat& dst )
{
    const int cn = src0.channels();
    int srcstep = cn;
    int dststep = 1;

    if( !dst.isContinuous() )
        dststep = (int)(dst.step/dst.elemSize());

    if( !src0.isContinuous() )
        srcstep = (int)(src0.step/src0.elemSize1());

    Complex<depth_t> *dst_data = dst.ptr<Complex<depth_t> >();
    const depth_t* src0_data = src0.ptr<depth_t>();
    const depth_t* src1_data = src1.ptr<depth_t>();
    dst_data->re = src0_data[0];
    dst_data->im = 0;
    const int n = dst.cols + dst.rows - 1;
    const int n2 = (n+1) >> 1;

    if( (n & 1) == 0 )
    {
        dst_data[n2*dststep].re = src0_data[(cn == 1 ? n-1 : n2)*srcstep];
        dst_data[n2*dststep].im = 0;
    }

    int delta0 = srcstep;
    int delta1 = delta0 + (cn == 1 ? srcstep : 1);

    if( cn == 1 )
        srcstep *= 2;

    for( int i = 1; i < n2; i++, delta0 += srcstep, delta1 += srcstep )
    {
        depth_t t0 = src0_data[delta0];
        depth_t t1 = src0_data[delta1];

        dst_data[i*dststep].re = t0;
        dst_data[i*dststep].im = t1;

        t0 = src1_data[delta0];
        t1 = -src1_data[delta1];

        dst_data[(n-i)*dststep].re = t0;
        dst_data[(n-i)*dststep].im = t1;
    }
}

/// Helper function to convert a ccs array into a complex matrix.
static void convert_from_ccs( const Mat& src0, const Mat& src1, Mat& dst, const int flags )
{
    if( dst.rows > 1 && (dst.cols > 1 || (flags & DFT_ROWS)) )
    {
        const int count = dst.rows;
        const int len = dst.cols;
        const bool is2d = (flags & DFT_ROWS) == 0;
        for( int i = 0; i < count; i++ )
        {
            const int j = !is2d || i == 0 ? i : count - i;
            const Mat& src0row = src0.row(i);
            const Mat& src1row = src1.row(j);
            Mat dstrow = dst.row(i);
            convert_from_ccs( src0row, src1row, dstrow, 0 );
        }

        if( is2d )
        {
            const Mat& src0row = src0.col(0);
            Mat dstrow = dst.col(0);
            convert_from_ccs( src0row, src0row, dstrow, 0 );

            if( (len & 1) == 0 )
            {
                const Mat& src0row_even = src0.col(src0.cols - 1);
                Mat dstrow_even = dst.col(len/2);
                convert_from_ccs( src0row_even, src0row_even, dstrow_even, 0 );
            }
        }
    }
    else
    {
        if( dst.depth() == CV_32F )
        {
            convert_from_ccs_helper<float>( src0, src1, dst );
        }
        else
        {
            convert_from_ccs_helper<double>( src0, src1, dst );
        }
    }
}

/// Helper function to compute complex number (nu_re + nu_im * i) / (de_re + de_im * i).
static std::pair<double, double> divide_complex_numbers( const double nu_re, const double nu_im,
                                                         const double de_re, const double de_im,
                                                         const bool conj_de )
{
    if ( conj_de )
    {
        return divide_complex_numbers( nu_re, nu_im, de_re, -de_im, false /* conj_de */ );
    }

    const double result_de = de_re * de_re + de_im * de_im + DBL_EPSILON;
    const double result_re = nu_re * de_re + nu_im * de_im;
    const double result_im = nu_re * (-de_im) + nu_im * de_re;
    return std::pair<double, double>(result_re / result_de, result_im / result_de);
}

/// Helper function to divide a DFT in src1 by a DFT in src2 with depths depth_t.  The DFTs are
/// complex matrices.
template <typename depth_t>
static void div_complex_helper( const Mat& src1, const Mat& src2, Mat& dst, int flags )
{
    CV_Assert( src1.size == src2.size && src1.type() == src2.type() );
    dst.create( src1.rows, src1.cols, src1.type() );
    const int cn = src1.channels();
    int cols = src1.cols * cn;

    for( int i = 0; i < dst.rows; i++ )
    {
        const depth_t *src1_data = src1.ptr<depth_t>(i);
        const depth_t *src2_data = src2.ptr<depth_t>(i);
        depth_t *dst_data = dst.ptr<depth_t>(i);
        for( int j = 0; j < cols; j += 2 )
        {
            std::pair<double, double> result =
                    divide_complex_numbers( src1_data[j], src1_data[j + 1],
                                            src2_data[j], src2_data[j + 1],
                                            (flags & CV_DXT_MUL_CONJ) != 0 );
            dst_data[j] = (depth_t)result.first;
            dst_data[j + 1] = (depth_t)result.second;
        }
    }
}

/// Helper function to divide a DFT in src1 by a DFT in src2.  The DFTs are complex matrices.
static void div_complex( const Mat& src1, const Mat& src2, Mat& dst, const int flags )
{
    const int type = src1.type();
    CV_Assert( type == CV_32FC2 || type == CV_64FC2 );

    if ( src1.depth() == CV_32F )
    {
        return div_complex_helper<float>( src1, src2, dst, flags );
    }
    else
    {
        return div_complex_helper<double>( src1, src2, dst, flags );
    }
}

void CV_DivSpectrumsTest::prepare_to_validation( int /* test_case_idx */ )
{
    Mat &src1 = test_mat[INPUT][0];
    Mat &src2 = test_mat[INPUT][1];
    Mat &ref_dst = test_mat[REF_OUTPUT][0];
    const int cn = src1.channels();
    // Inputs are CCS-packed arrays.  Convert them to complex matrices and get the expected output
    // as a complex matrix.
    if( cn == 1 )
    {
        Mat &converted_src1 = test_mat[TEMP][0];
        Mat &converted_src2 = test_mat[TEMP][1];
        convert_from_ccs( src1, src1, converted_src1, flags );
        convert_from_ccs( src2, src2, converted_src2, flags );
        div_complex( converted_src1, converted_src2, ref_dst, flags );
    }
    // Inputs are complex matrices.  Get the expected output as a complex matrix.
    else
    {
        div_complex( src1, src2, ref_dst, flags );
    }
}

void CV_DivSpectrumsTest::run_func()
{
    const Mat &src1 = test_mat[INPUT][0];
    const Mat &src2 = test_mat[INPUT][1];
    const int cn = src1.channels();

    // Inputs are CCS-packed arrays.  Get the output as a CCS-packed array and convert it to a
    // complex matrix.
    if ( cn == 1 )
    {
        Mat &dst = test_mat[TEMP][2];
        cv::divSpectrums( src1, src2, dst, flags, (flags & CV_DXT_MUL_CONJ) != 0 );
        Mat &converted_dst = test_mat[OUTPUT][0];
        convert_from_ccs( dst, dst, converted_dst, flags );
    }
    // Inputs are complex matrices.  Get the output as a complex matrix.
    else
    {
        Mat &dst = test_mat[OUTPUT][0];
        cv::divSpectrums( src1, src2, dst, flags, (flags & CV_DXT_MUL_CONJ) != 0 );
    }
}

TEST(Imgproc_DivSpectrums, accuracy) { CV_DivSpectrumsTest test; test.safe_run(); }

}} // namespace
