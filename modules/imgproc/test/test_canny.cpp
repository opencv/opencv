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

class CV_CannyTest : public cvtest::ArrayTest
{
public:
    CV_CannyTest(bool custom_deriv = false);

protected:
    void get_test_array_types_and_sizes( int test_case_idx, vector<vector<Size> >& sizes, vector<vector<int> >& types );
    double get_success_error_level( int test_case_idx, int i, int j );
    int prepare_test_case( int test_case_idx );
    void run_func();
    void prepare_to_validation( int );
    int validate_test_results( int /*test_case_idx*/ );

    int aperture_size;
    bool use_true_gradient;
    double threshold1, threshold2;
    bool test_cpp;
    bool test_custom_deriv;

    Mat img;
};


CV_CannyTest::CV_CannyTest(bool custom_deriv)
{
    test_array[INPUT].push_back(NULL);
    test_array[OUTPUT].push_back(NULL);
    test_array[REF_OUTPUT].push_back(NULL);
    element_wise_relative_error = true;
    aperture_size = 0;
    use_true_gradient = false;
    threshold1 = threshold2 = 0;
    test_custom_deriv = custom_deriv;

    const char imgPath[] = "shared/fruits.png";
    img = cv::imread(cvtest::TS::ptr()->get_data_path() + imgPath, IMREAD_GRAYSCALE);
}


void CV_CannyTest::get_test_array_types_and_sizes( int test_case_idx,
                                                  vector<vector<Size> >& sizes,
                                                  vector<vector<int> >& types )
{
    RNG& rng = ts->get_rng();
    double thresh_range;

    cvtest::ArrayTest::get_test_array_types_and_sizes( test_case_idx, sizes, types );
    types[INPUT][0] = types[OUTPUT][0] = types[REF_OUTPUT][0] = CV_8U;

    aperture_size = cvtest::randInt(rng) % 2 ? 5 : 3;
    thresh_range = aperture_size == 3 ? 300 : 1000;

    threshold1 = cvtest::randReal(rng)*thresh_range;
    threshold2 = cvtest::randReal(rng)*thresh_range*0.3;

    if( cvtest::randInt(rng) % 2 )
        CV_SWAP( threshold1, threshold2, thresh_range );

    use_true_gradient = cvtest::randInt(rng) % 2 != 0;
    test_cpp = (cvtest::randInt(rng) & 256) == 0;

    ts->printf(cvtest::TS::LOG, "Canny(size = %d x %d, aperture_size = %d, threshold1 = %g, threshold2 = %g, L2 = %s) test_cpp = %s (test case #%d)\n",
        sizes[0][0].width, sizes[0][0].height, aperture_size, threshold1, threshold2, use_true_gradient ? "TRUE" : "FALSE", test_cpp ? "TRUE" : "FALSE", test_case_idx);
}


int CV_CannyTest::prepare_test_case( int test_case_idx )
{
    int code = cvtest::ArrayTest::prepare_test_case( test_case_idx );
    if( code > 0 )
    {
        RNG& rng = ts->get_rng();
        Mat& src = test_mat[INPUT][0];
        //GaussianBlur(src, src, Size(11, 11), 5, 5);
        if(src.cols > img.cols || src.rows > img.rows)
            resize(img, src, src.size(), 0, 0, INTER_LINEAR_EXACT);
        else
            img(
                Rect(
                    cvtest::randInt(rng) % (img.cols-src.cols),
                    cvtest::randInt(rng) % (img.rows-src.rows),
                    src.cols,
                    src.rows
                )
            ).copyTo(src);
        GaussianBlur(src, src, Size(5, 5), 0);
    }

    return code;
}


double CV_CannyTest::get_success_error_level( int /*test_case_idx*/, int /*i*/, int /*j*/ )
{
    return 0;
}


void CV_CannyTest::run_func()
{
    if (test_custom_deriv)
    {
        cv::Mat _out = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::Mat src = cv::cvarrToMat(test_array[INPUT][0]);
        cv::Mat dx, dy;
        int m = aperture_size;
        Point anchor(m/2, m/2);
        Mat dxkernel = cvtest::calcSobelKernel2D( 1, 0, m, 0 );
        Mat dykernel = cvtest::calcSobelKernel2D( 0, 1, m, 0 );
        cvtest::filter2D(src, dx, CV_16S, dxkernel, anchor, 0, BORDER_REPLICATE);
        cvtest::filter2D(src, dy, CV_16S, dykernel, anchor, 0, BORDER_REPLICATE);
        cv::Canny(dx, dy, _out, threshold1, threshold2, use_true_gradient);
    }
    else
    {
        cv::Mat _out = cv::cvarrToMat(test_array[OUTPUT][0]);
        cv::Canny(cv::cvarrToMat(test_array[INPUT][0]), _out, threshold1, threshold2,
                aperture_size + (use_true_gradient ? CV_CANNY_L2_GRADIENT : 0));
    }
}


static void
cannyFollow( int x, int y, float lowThreshold, const Mat& mag, Mat& dst )
{
    static const int ofs[][2] = {{1,0},{1,-1},{0,-1},{-1,-1},{-1,0},{-1,1},{0,1},{1,1}};
    int i;

    dst.at<uchar>(y, x) = (uchar)255;

    for( i = 0; i < 8; i++ )
    {
        int x1 = x + ofs[i][0];
        int y1 = y + ofs[i][1];
        if( (unsigned)x1 < (unsigned)mag.cols &&
            (unsigned)y1 < (unsigned)mag.rows &&
            mag.at<float>(y1, x1) > lowThreshold &&
            !dst.at<uchar>(y1, x1) )
            cannyFollow( x1, y1, lowThreshold, mag, dst );
    }
}


static void
test_Canny( const Mat& src, Mat& dst,
            double threshold1, double threshold2,
            int aperture_size, bool use_true_gradient )
{
    int m = aperture_size;
    Point anchor(m/2, m/2);
    const double tan_pi_8 = tan(CV_PI/8.);
    const double tan_3pi_8 = tan(CV_PI*3/8);
    float lowThreshold = (float)MIN(threshold1, threshold2);
    float highThreshold = (float)MAX(threshold1, threshold2);

    int x, y, width = src.cols, height = src.rows;

    Mat dxkernel = cvtest::calcSobelKernel2D( 1, 0, m, 0 );
    Mat dykernel = cvtest::calcSobelKernel2D( 0, 1, m, 0 );
    Mat dx, dy, mag(height, width, CV_32F);
    cvtest::filter2D(src, dx, CV_32S, dxkernel, anchor, 0, BORDER_REPLICATE);
    cvtest::filter2D(src, dy, CV_32S, dykernel, anchor, 0, BORDER_REPLICATE);

    // calc gradient magnitude
    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < width; x++ )
        {
            int dxval = dx.at<int>(y, x), dyval = dy.at<int>(y, x);
            mag.at<float>(y, x) = use_true_gradient ?
                (float)sqrt((double)(dxval*dxval + dyval*dyval)) :
                (float)(fabs((double)dxval) + fabs((double)dyval));
        }
    }

    // calc gradient direction, do nonmaxima suppression
    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < width; x++ )
        {

            float a = mag.at<float>(y, x), b = 0, c = 0;
            int y1 = 0, y2 = 0, x1 = 0, x2 = 0;

            if( a <= lowThreshold )
                continue;

            int dxval = dx.at<int>(y, x);
            int dyval = dy.at<int>(y, x);

            double tg = dxval ? (double)dyval/dxval : DBL_MAX*CV_SIGN(dyval);

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
                b = (float)fabs(mag.at<float>(y1, x1));

            if( (unsigned)y2 < (unsigned)height && (unsigned)x2 < (unsigned)width )
                c = (float)fabs(mag.at<float>(y2, x2));

            if( (a > b || (a == b && ((x1 == x+1 && y1 == y) || (x1 == x && y1 == y+1)))) && a > c )
                ;
            else
                mag.at<float>(y, x) = -a;
        }
    }

    dst = Scalar::all(0);

    // hysteresis threshold
    for( y = 0; y < height; y++ )
    {
        for( x = 0; x < width; x++ )
            if( mag.at<float>(y, x) > highThreshold && !dst.at<uchar>(y, x) )
                cannyFollow( x, y, lowThreshold, mag, dst );
    }
}


void CV_CannyTest::prepare_to_validation( int )
{
    Mat src = test_mat[INPUT][0], dst = test_mat[REF_OUTPUT][0];
    test_Canny( src, dst, threshold1, threshold2, aperture_size, use_true_gradient );
}


int CV_CannyTest::validate_test_results( int test_case_idx )
{
    int code = cvtest::TS::OK, nz0;
    prepare_to_validation(test_case_idx);

    double err = cvtest::norm(test_mat[OUTPUT][0], test_mat[REF_OUTPUT][0], CV_L1);
    if( err == 0 )
        return code;

    if( err != cvRound(err) || cvRound(err)%255 != 0 )
    {
        ts->printf( cvtest::TS::LOG, "Some of the pixels, produced by Canny, are not 0's or 255's; the difference is %g\n", err );
        ts->set_failed_test_info( cvtest::TS::FAIL_INVALID_OUTPUT );
        return code;
    }

    nz0 = cvRound(cvtest::norm(test_mat[REF_OUTPUT][0], CV_L1)/255);
    err = (err/255/MAX(nz0,100))*100;
    if( err > 1 )
    {
        ts->printf( cvtest::TS::LOG, "Too high percentage of non-matching edge pixels = %g%%\n", err);
        ts->set_failed_test_info( cvtest::TS::FAIL_BAD_ACCURACY );
    }

    return code;
}

TEST(Imgproc_Canny, accuracy) { CV_CannyTest test; test.safe_run(); }
TEST(Imgproc_Canny, accuracy_deriv) { CV_CannyTest test(true); test.safe_run(); }


/*
 * Comparing OpenVX based implementation with the main one
*/

#ifndef IMPLEMENT_PARAM_CLASS
#define IMPLEMENT_PARAM_CLASS(name, type) \
    class name \
    { \
    public: \
        name ( type arg = type ()) : val_(arg) {} \
        operator type () const {return val_;} \
    private: \
        type val_; \
    }; \
    inline void PrintTo( name param, std::ostream* os) \
    { \
        *os << #name <<  "(" << testing::PrintToString(static_cast< type >(param)) << ")"; \
    }
#endif // IMPLEMENT_PARAM_CLASS

IMPLEMENT_PARAM_CLASS(ImagePath, string)
IMPLEMENT_PARAM_CLASS(ApertureSize, int)
IMPLEMENT_PARAM_CLASS(L2gradient, bool)

PARAM_TEST_CASE(CannyVX, ImagePath, ApertureSize, L2gradient)
{
    string imgPath;
    int kSize;
    bool useL2;
    Mat src, dst;

    virtual void SetUp()
    {
        imgPath = GET_PARAM(0);
        kSize = GET_PARAM(1);
        useL2 = GET_PARAM(2);
    }

    void loadImage()
    {
        src = cv::imread(cvtest::TS::ptr()->get_data_path() + imgPath, IMREAD_GRAYSCALE);
        ASSERT_FALSE(src.empty()) << "can't load image: " << imgPath;
    }
};

TEST_P(CannyVX, Accuracy)
{
    if(haveOpenVX())
    {
        loadImage();

        setUseOpenVX(false);
        Mat canny;
        cv::Canny(src, canny, 100, 150, 3);

        setUseOpenVX(true);
        Mat cannyVX;
        cv::Canny(src, cannyVX, 100, 150, 3);

        // 'smart' diff check (excluding isolated pixels)
        Mat diff, diff1;
        absdiff(canny, cannyVX, diff);
        boxFilter(diff, diff1, -1, Size(3,3));
        const int minPixelsAroud = 3; // empirical number
        diff1 = diff1 > 255/9 * minPixelsAroud;
        erode(diff1, diff1, Mat());
        double error = cv::norm(diff1, NORM_L1) / 255;
        const int maxError = std::min(10, diff.size().area()/100); // empirical number
        if(error > maxError)
        {
            string outPath =
                    string("CannyVX-diff-") +
                    imgPath + '-' +
                    'k' + char(kSize+'0') + '-' +
                    (useL2 ? "l2" : "l1");
            std::replace(outPath.begin(), outPath.end(), '/', '_');
            std::replace(outPath.begin(), outPath.end(), '\\', '_');
            std::replace(outPath.begin(), outPath.end(), '.', '_');
            imwrite(outPath+".png", diff);
        }
        ASSERT_LE(error, maxError);

    }
}

    INSTANTIATE_TEST_CASE_P(
                ImgProc, CannyVX,
                testing::Combine(
                    testing::Values(
                        string("shared/baboon.png"),
                        string("shared/fruits.png"),
                        string("shared/lena.png"),
                        string("shared/pic1.png"),
                        string("shared/pic3.png"),
                        string("shared/pic5.png"),
                        string("shared/pic6.png")
                    ),
                    testing::Values(ApertureSize(3), ApertureSize(5)),
                    testing::Values(L2gradient(false), L2gradient(true))
                )
    );

}} // namespace
/* End of file. */
