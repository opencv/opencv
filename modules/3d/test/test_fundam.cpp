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

namespace opencv_test {

static void test_convertHomogeneous( const Mat& _src, Mat& _dst )
{
    Mat src = _src, dst = _dst;
    int i, count, sdims, ddims;
    int sstep1, sstep2, dstep1, dstep2;

    if( src.depth() != CV_64F )
        _src.convertTo(src, CV_64F);

    if( dst.depth() != CV_64F )
        dst.create(dst.size(), CV_MAKETYPE(CV_64F, _dst.channels()));

    if( src.rows > src.cols )
    {
        count = src.rows;
        sdims = src.channels()*src.cols;
        sstep1 = (int)(src.step/sizeof(double));
        sstep2 = 1;
    }
    else
    {
        count = src.cols;
        sdims = src.channels()*src.rows;
        if( src.rows == 1 )
        {
            sstep1 = sdims;
            sstep2 = 1;
        }
        else
        {
            sstep1 = 1;
            sstep2 = (int)(src.step/sizeof(double));
        }
    }

    if( dst.rows > dst.cols )
    {
        CV_Assert( count == dst.rows );
        ddims = dst.channels()*dst.cols;
        dstep1 = (int)(dst.step/sizeof(double));
        dstep2 = 1;
    }
    else
    {
        CV_Assert( count == dst.cols );
        ddims = dst.channels()*dst.rows;
        if( dst.rows == 1 )
        {
            dstep1 = ddims;
            dstep2 = 1;
        }
        else
        {
            dstep1 = 1;
            dstep2 = (int)(dst.step/sizeof(double));
        }
    }

    double* s = src.ptr<double>();
    double* d = dst.ptr<double>();

    if( sdims <= ddims )
    {
        int wstep = dstep2*(ddims - 1);

        for( i = 0; i < count; i++, s += sstep1, d += dstep1 )
        {
            double x = s[0];
            double y = s[sstep2];

            d[wstep] = 1;
            d[0] = x;
            d[dstep2] = y;

            if( sdims >= 3 )
            {
                d[dstep2*2] = s[sstep2*2];
                if( sdims == 4 )
                    d[dstep2*3] = s[sstep2*3];
            }
        }
    }
    else
    {
        int wstep = sstep2*(sdims - 1);

        for( i = 0; i < count; i++, s += sstep1, d += dstep1 )
        {
            double w = s[wstep];
            double x = s[0];
            double y = s[sstep2];

            w = w ? 1./w : 1;

            d[0] = x*w;
            d[dstep2] = y*w;

            if( ddims == 3 )
                d[dstep2*2] = s[sstep2*2]*w;
        }
    }

    if( dst.data != _dst.data )
        dst.convertTo(_dst, _dst.depth());
}

namespace {

/********************************** convert homogeneous *********************************/

class CV_ConvertHomogeneousTest : public cvtest::ArrayTest
{
public:
    CV_ConvertHomogeneousTest();

protected:
    int read_params( const cv::FileStorage& fs );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int dims1, dims2;
    int pt_count;
};


CV_ConvertHomogeneousTest::CV_ConvertHomogeneousTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    element_wise_relative_error = false;

    pt_count = dims1 = dims2 = 0;
}


int CV_ConvertHomogeneousTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    return code;
}


void CV_ConvertHomogeneousTest::get_test_array_types_and_sizes( int /*test_case_idx*/,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int pt_depth1 = cvtest::randInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int pt_depth2 = pt_depth1;//cvtest::randInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    double pt_count_exp = cvtest::randReal(rng)*6 + 1;
    int t;

    pt_count = cvRound(exp(pt_count_exp));
    pt_count = MAX( pt_count, 5 );

    dims1 = 2 + (cvtest::randInt(rng) % 2);
    dims2 = dims1 + 1;

    if( cvtest::randInt(rng) % 2 )
        CV_SWAP( dims1, dims2, t );

    types[INPUT][0] = CV_MAKETYPE(pt_depth1, 1);

    sizes[INPUT][0] = Size(dims1, pt_count);
    if( cvtest::randInt(rng) % 2 )
    {
        types[INPUT][0] = CV_MAKETYPE(pt_depth1, dims1);
        if( cvtest::randInt(rng) % 2 )
            sizes[INPUT][0] = Size(pt_count, 1);
        else
            sizes[INPUT][0] = Size(1, pt_count);
    }

    types[OUTPUT][0] = CV_MAKETYPE(pt_depth2, dims2);
    sizes[OUTPUT][0] = Size(1, pt_count);

    types[REF_OUTPUT][0] = types[OUTPUT][0];
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


double CV_ConvertHomogeneousTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-5;
}


void CV_ConvertHomogeneousTest::fill_array( int /*test_case_idx*/, int /*i*/, int /*j*/, Mat& arr )
{
    Mat temp( 1, pt_count, CV_MAKETYPE(CV_64FC1,dims1) );
    RNG& rng = ts->get_rng();
    Scalar low = Scalar::all(0), high = Scalar::all(10);

    if( dims1 > dims2 )
        low.val[dims1-1] = 1.;

    cvtest::randUni( rng, temp, low, high );
    test_convertHomogeneous( temp, arr );
}


void CV_ConvertHomogeneousTest::run_func()
{
    cv::Mat _input = test_mat[INPUT][0], &_output = test_mat[OUTPUT][0];
    if( dims1 > dims2 )
        cv::convertPointsFromHomogeneous(_input, _output);
    else
        cv::convertPointsToHomogeneous(_input, _output);
}


void CV_ConvertHomogeneousTest::prepare_to_validation( int /*test_case_idx*/ )
{
    test_convertHomogeneous( test_mat[INPUT][0], test_mat[REF_OUTPUT][0] );
}


/************************** compute corresponding epipolar lines ************************/

class CV_ComputeEpilinesTest : public cvtest::ArrayTest
{
public:
    CV_ComputeEpilinesTest();

protected:
    int read_params( const cv::FileStorage& fs );
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    void fill_array( int test_case_idx, int i, int j, Mat& arr );
    double get_success_error_level( int test_case_idx, int i, int j );
    void run_func();
    void prepare_to_validation( int );

    int which_image;
    int dims;
    int pt_count;
};


CV_ComputeEpilinesTest::CV_ComputeEpilinesTest()
{
    test_array[INPUT].push_back(NULL);
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    element_wise_relative_error = false;

    pt_count = dims = which_image = 0;
}


int CV_ComputeEpilinesTest::read_params( const cv::FileStorage& fs )
{
    int code = cvtest::ArrayTest::read_params( fs );
    return code;
}


void CV_ComputeEpilinesTest::get_test_array_types_and_sizes( int /*test_case_idx*/,
                                                vector<vector<Size> >& sizes, vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    int fm_depth = cvtest::randInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int pt_depth = cvtest::randInt(rng) % 2 == 0 ? CV_32F : CV_64F;
    int ln_depth = pt_depth;
    double pt_count_exp = cvtest::randReal(rng)*6;

    which_image = 1 + (cvtest::randInt(rng) % 2);

    pt_count = cvRound(exp(pt_count_exp));
    pt_count = MAX( pt_count, 1 );
    bool few_points = pt_count < 5;

    dims = 2 + (cvtest::randInt(rng) % 2);

    types[INPUT][0] = CV_MAKETYPE(pt_depth, 1);

    sizes[INPUT][0] = Size(dims, pt_count);
    if( cvtest::randInt(rng) % 2 || few_points )
    {
        types[INPUT][0] = CV_MAKETYPE(pt_depth, dims);
        if( cvtest::randInt(rng) % 2 )
            sizes[INPUT][0] = Size(pt_count, 1);
        else
            sizes[INPUT][0] = Size(1, pt_count);
    }

    types[INPUT][1] = CV_MAKETYPE(fm_depth, 1);
    sizes[INPUT][1] = Size(3, 3);

    types[OUTPUT][0] = CV_MAKETYPE(ln_depth, 3);
    sizes[OUTPUT][0] = Size(1, pt_count);

    types[REF_OUTPUT][0] = types[OUTPUT][0];
    sizes[REF_OUTPUT][0] = sizes[OUTPUT][0];
}


double CV_ComputeEpilinesTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 1e-5;
}


void CV_ComputeEpilinesTest::fill_array( int test_case_idx, int i, int j, Mat& arr )
{
    RNG& rng = ts->get_rng();

    if( i == INPUT && j == 0 )
    {
        Mat temp( 1, pt_count, CV_MAKETYPE(CV_64FC1,dims) );
        cvtest::randUni( rng, temp, Scalar(0,0,1), Scalar::all(10) );
        test_convertHomogeneous( temp, arr );
    }
    else if( i == INPUT && j == 1 )
        cvtest::randUni( rng, arr, Scalar::all(0), Scalar::all(10) );
    else
        cvtest::ArrayTest::fill_array( test_case_idx, i, j, arr );
}


void CV_ComputeEpilinesTest::run_func()
{
    cv::Mat _points = test_mat[INPUT][0], _F = test_mat[INPUT][1], &_lines = test_mat[OUTPUT][0];
    cv::computeCorrespondEpilines( _points, which_image, _F, _lines );
}


void CV_ComputeEpilinesTest::prepare_to_validation( int /*test_case_idx*/ )
{
    Mat pt( 1, pt_count, CV_MAKETYPE(CV_64F, 3) );
    Mat lines( 1, pt_count, CV_MAKETYPE(CV_64F, 3) );
    double f[9];
    Mat F( 3, 3, CV_64F, f );

    test_convertHomogeneous( test_mat[INPUT][0], pt );
    test_mat[INPUT][1].convertTo(F, CV_64F);
    if( which_image == 2 )
        cv::transpose( F, F );

    for( int i = 0; i < pt_count; i++ )
    {
        double* p = pt.ptr<double>() + i*3;
        double* l = lines.ptr<double>() + i*3;
        double t0 = f[0]*p[0] + f[1]*p[1] + f[2]*p[2];
        double t1 = f[3]*p[0] + f[4]*p[1] + f[5]*p[2];
        double t2 = f[6]*p[0] + f[7]*p[1] + f[8]*p[2];
        double d = sqrt(t0*t0 + t1*t1);
        d = d ? 1./d : 1.;
        l[0] = t0*d; l[1] = t1*d; l[2] = t2*d;
    }

    test_convertHomogeneous( lines, test_mat[REF_OUTPUT][0] );
}

TEST(Calib3d_ConvertHomogeneoous, accuracy) { CV_ConvertHomogeneousTest test; test.safe_run(); }
TEST(Calib3d_ComputeEpilines, accuracy) { CV_ComputeEpilinesTest test; test.safe_run(); }

TEST(Calib3d_FindFundamentalMat, correctMatches)
{
    double fdata[] = {0, 0, 0, 0, 0, -1, 0, 1, 0};
    double p1data[] = {200, 0, 1};
    double p2data[] = {170, 0, 1};

    Mat F(3, 3, CV_64F, fdata);
    Mat p1(1, 1, CV_64FC2, p1data);
    Mat p2(1, 1, CV_64FC2, p2data);
    Mat np1, np2;

    correctMatches(F, p1, p2, np1, np2);

    cout << np1 << endl;
    cout << np2 << endl;
}

}} // namespace
/* End of file. */
