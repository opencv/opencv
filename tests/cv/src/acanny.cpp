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

class CV_CannyTest : public CvArrTest
{
public:
    CV_CannyTest();

protected:
    void get_test_array_types_and_sizes( int test_case_idx, CvSize** sizes, int** types );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int );
    int validate_test_results( int /*test_case_idx*/ );

    int aperture_size, use_true_gradient;
    double threshold1, threshold2;
    bool test_cpp;
};


CV_CannyTest::CV_CannyTest()
    : CvArrTest( "canny", "cvCanny, cvSobel", "" )
{
    test_array[INPUT].push(NULL);
    test_array[OUTPUT].push(NULL);
    test_array[REF_OUTPUT].push(NULL);
    element_wise_relative_error = true;
    aperture_size = use_true_gradient = 0;
    threshold1 = threshold2 = 0;

    support_testing_modes = CvTS::CORRECTNESS_CHECK_MODE;
    default_timing_param_names = 0;
    test_cpp = false;
}


void CV_CannyTest::get_test_array_types_and_sizes( int test_case_idx,
                                                CvSize** sizes, int** types )
{
    CvRNG* rng = ts->get_rng();
    double thresh_range;

    CvArrTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8U;

    aperture_size = cvTsRandInt(rng) % 2 ? 5 : 3;
    thresh_range = aperture_size == 3 ? 300 : 1000;

    threshold1 = cvTsRandReal(rng)*thresh_range;
    threshold2 = cvTsRandReal(rng)*thresh_range*0.3;

    if( cvTsRandInt(rng) % 2 )
        CV_SWAP( threshold1, threshold2, thresh_range );

    use_true_gradient = cvTsRandInt(rng) % 2;
    test_cpp = (cvTsRandInt(rng) & 256) == 0;
}


int CV_CannyTest::prepare_test_case( int test_case_idx )
{
    int code = CvArrTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        CvMat* src = &test_mat[INPUT][0];
        cvSmooth( src, src, CV_GAUSSIAN, 11, 11, 5, 5 );
    }

    return code;
}


double CV_CannyTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_CannyTest::run_func()
{
    if(!test_cpp)
        cvCanny( test_array[INPUT][0], test_array[OUTPUT][0], threshold1, threshold2,
                aperture_size + (use_true_gradient ? CV_CANNY_L2_GRADIENT : 0));
    else
    {
        cv::Mat _out = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::Canny(cv::cvarrToMat(test_array[INPUT][0]), _out, threshold1, threshold2,
                aperture_size + (use_true_gradient ? CV_CANNY_L2_GRADIENT : 0));
    }
}


static void
icvTsCannyFollow( int x, int y, float lowThreshold, const CvMat* mag, CvMat* dst )
{
    static const int ofs[][2] = {{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1},{1,1}};
    int i;

    dst->data.ptr[dst->step*y + x] = (uchar)255;

    for( i = 0; i < 8; i++ )
    {
        int x1 = x + ofs[i][0];
        int y1 = y + ofs[i][1];
        if( (unsigned)x1 < (unsigned)mag->cols &&
            (unsigned)y1 < (unsigned)mag->rows &&
            mag->data.fl[y1*mag->cols+x1] > lowThreshold &&
            !dst->data.ptr[dst->step*y1+x1] )
            icvTsCannyFollow( x1, y1, lowThreshold, mag, dst );
    }
}


static void
icvTsCanny( const CvMat* src, CvMat* dst,
            double threshold1, double threshold2,
            int aperture_size, int use_true_gradient )
{
    int m = aperture_size;
    CvMat* _src = cvCreateMat( src->rows + m - 1, src->cols + m - 1, CV_16S );
    CvMat* dx = cvCreateMat( src->rows, src->cols, CV_16S );
    CvMat* dy = cvCreateMat( src->rows, src->cols, CV_16S );
    CvMat* kernel = cvCreateMat( m, m, CV_32F );
    CvPoint anchor = {m/2, m/2};
    CvMat* mag = cvCreateMat( src->rows, src->cols, CV_32F );
    const double tan_pi_8 = tan(CV_PI/8.);
    const double tan_3pi_8 = tan(CV_PI*3/8);
    float lowThreshold = (float)MIN(threshold1, threshold2);
    float highThreshold = (float)MAX(threshold1, threshold2);

    int x, y, width = src->cols, height = src->rows;

    cvTsConvert( src, dx );
    cvTsPrepareToFilter( dx, _src, anchor, CV_TS_BORDER_REPLICATE );
    cvTsCalcSobelKernel2D( 1, 0, m, 0, kernel );
    cvTsConvolve2D( _src, dx, kernel, anchor );
    cvTsCalcSobelKernel2D( 0, 1, m, 0, kernel );
    cvTsConvolve2D( _src, dy, kernel, anchor );

    /* estimate magnitude and angle */
    for( y = 0; y < height; y++ )
    {
        const short* _dx = (short*)(dx->data.ptr + dx->step*y);
        const short* _dy = (short*)(dy->data.ptr + dy->step*y);
        float* _mag = (float*)(mag->data.ptr + mag->step*y);

        for( x = 0; x < width; x++ )
        {
            float mval = use_true_gradient ?
                (float)sqrt((double)(_dx[x]*_dx[x] + _dy[x]*_dy[x])) :
                (float)(abs(_dx[x]) + abs(_dy[x]));
            _mag[x] = mval;
        }
    }

    /* nonmaxima suppression */
    for( y = 0; y < height; y++ )
    {
        const short* _dx = (short*)(dx->data.ptr + dx->step*y);
        const short* _dy = (short*)(dy->data.ptr + dy->step*y);
        float* _mag = (float*)(mag->data.ptr + mag->step*y);

        for( x = 0; x < width; x++ )
        {
            int y1 = 0, y2 = 0, x1 = 0, x2 = 0;
            double tg;
            float a = _mag[x], b = 0, c = 0;

            if( a <= lowThreshold )
                continue;

            if( _dx[x] )
                tg = (double)_dy[x]/_dx[x];
            else
                tg = DBL_MAX*CV_SIGN(_dy[x]);

            if( fabs(tg) < tan_pi_8 )
            {
                y1 = y2 = y; x1 = x + 1; x2 = x - 1;
            }
            else if( tan_pi_8 <= tg && tg <= tan_3pi_8 )
            {
                y1 = y + 1; y2 = y - 1; x1 = x + 1; x2 = x - 1;
            }
            else if( -tan_3pi_8 <= tg && tg <= -tan_pi_8 )
            {
                y1 = y - 1; y2 = y + 1; x1 = x + 1; x2 = x - 1;
            }
            else
            {
                assert( fabs(tg) > tan_3pi_8 );
                x1 = x2 = x; y1 = y + 1; y2 = y - 1;
            }

            if( (unsigned)y1 < (unsigned)height && (unsigned)x1 < (unsigned)width )
                b = (float)fabs((double)mag->data.fl[y1*width+x1]);

            if( (unsigned)y2 < (unsigned)height && (unsigned)x2 < (unsigned)width )
                c = (float)fabs((double)mag->data.fl[y2*width+x2]);

            if( (a > b || (a == b && ((x1 == x+1 && y1 == y) || (x1 == x && y1 == y+1)))) && a > c )
                ;
            else
                _mag[x] = -a;
        }
    }

    cvTsZero( dst );

    /* hysteresis threshold */
    for( y = 0; y < height; y++ )
    {
        const float* _mag = (float*)(mag->data.ptr + mag->step*y);
        uchar* _dst = dst->data.ptr + dst->step*y;

        for( x = 0; x < width; x++ )
            if( _mag[x] > highThreshold && !_dst[x] )
                icvTsCannyFollow( x, y, lowThreshold, mag, dst );
    }

    cvReleaseMat( &_src );
    cvReleaseMat( &dx );
    cvReleaseMat( &dy );
    cvReleaseMat( &kernel );
    cvReleaseMat( &mag );
}


void CV_CannyTest::prepare_to_validation( int )
{
    icvTsCanny( &test_mat[INPUT][0], &test_mat[REF_OUTPUT][0],
                threshold1, threshold2, aperture_size, use_true_gradient );
    
    /*cv::Mat output(&test_mat[OUTPUT][0]);
    cv::Mat ref_output(&test_mat[REF_OUTPUT][0]);
    
    cv::absdiff(output, ref_output, output);
    cv::namedWindow("ref test", 0);
    cv::imshow("ref test", ref_output);
    cv::namedWindow("test", 0);
    cv::imshow("test", output);
    cv::waitKey();*/
}


int CV_CannyTest::validate_test_results( int test_case_idx )
{
    int code = CvTS::OK, nz0;
    prepare_to_validation(test_case_idx);
    
    double err = cvNorm(&test_mat[OUTPUT][0], &test_mat[REF_OUTPUT][0], CV_L1);
    if( err == 0 )
        goto _exit_;
    
    if( err != cvRound(err) || cvRound(err)%255 != 0 )
    {
        ts->printf( CvTS::LOG, "Some of the pixels, produced by Canny, are not 0's or 255's; the difference is %g\n", err );
        code = CvTS::FAIL_INVALID_OUTPUT;
        goto _exit_;
    }
    
    nz0 = cvCountNonZero(&test_mat[REF_OUTPUT][0]);
    err = (err/255/MAX(nz0,100))*100;
    if( err > 1 )
    {
        ts->printf( CvTS::LOG, "Too high percentage of non-matching edge pixels = %g%%\n", err);
        code = CvTS::FAIL_BAD_ACCURACY;
        goto _exit_;
    }
    
_exit_:
    if( code < 0 )
        ts->set_failed_test_info( code );
    return code;
}


CV_CannyTest canny_test;

/* End of file. */
