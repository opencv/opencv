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
#include <opencv2/highgui.hpp>

namespace opencv_test { namespace {

class CV_FindContourTest : public cvtest::BaseTest
{
public:
    enum { NUM_IMG = 4 };

    CV_FindContourTest();
    ~CV_FindContourTest();
    void clear();

protected:
    int read_params( CvFileStorage* fs );
    int prepare_test_case( int test_case_idx );
    int validate_test_results( int test_case_idx );
    void run_func();

    int min_blob_size, max_blob_size;
    int blob_count, max_log_blob_count;
    int retr_mode, approx_method;

    int min_log_img_width, max_log_img_width;
    int min_log_img_height, max_log_img_height;
    Size img_size;
    int count, count2;

    IplImage* img[NUM_IMG];
    CvMemStorage* storage;
    CvSeq *contours, *contours2, *chain;

    static const bool useVeryWideImages =
#if SIZE_MAX <= 0xffffffff
        // 32-bit: don't even try the very wide images
        false
#else
        // 64-bit: test with very wide images
        true
#endif
        ;
};


CV_FindContourTest::CV_FindContourTest()
{
    int i;

    test_case_count    = useVeryWideImages ? 10 : 300;
    min_blob_size      = 1;
    max_blob_size      = 50;
    max_log_blob_count = 10;

    min_log_img_width  = useVeryWideImages ? 17 : 3;
    max_log_img_width  = useVeryWideImages ? 17 : 10;

    min_log_img_height = 3;
    max_log_img_height = 10;

    for( i = 0; i < NUM_IMG; i++ )
        img[i] = 0;

    storage = 0;
}


CV_FindContourTest::~CV_FindContourTest()
{
    clear();
}


void CV_FindContourTest::clear()
{
    int i;

    cvtest::BaseTest::clear();

    for( i = 0; i < NUM_IMG; i++ )
        cvReleaseImage( &img[i] );

    cvReleaseMemStorage( &storage );
}


int CV_FindContourTest::read_params( CvFileStorage* fs )
{
    int t;
    int code = cvtest::BaseTest::read_params( fs );

    if( code < 0 )
        return code;

    min_blob_size      = cvReadInt( find_param( fs, "min_blob_size" ), min_blob_size );
    max_blob_size      = cvReadInt( find_param( fs, "max_blob_size" ), max_blob_size );
    max_log_blob_count = cvReadInt( find_param( fs, "max_log_blob_count" ), max_log_blob_count );
    min_log_img_width  = cvReadInt( find_param( fs, "min_log_img_width" ), min_log_img_width );
    max_log_img_width  = cvReadInt( find_param( fs, "max_log_img_width" ), max_log_img_width );
    min_log_img_height = cvReadInt( find_param( fs, "min_log_img_height"), min_log_img_height );
    max_log_img_height = cvReadInt( find_param( fs, "max_log_img_height"), max_log_img_height );

    min_blob_size = cvtest::clipInt( min_blob_size, 1, 100 );
    max_blob_size = cvtest::clipInt( max_blob_size, 1, 100 );

    if( min_blob_size > max_blob_size )
        CV_SWAP( min_blob_size, max_blob_size, t );

    max_log_blob_count = cvtest::clipInt( max_log_blob_count, 1, 10 );

    min_log_img_width  = cvtest::clipInt( min_log_img_width, 1, useVeryWideImages ? 17 : 10 );
    min_log_img_width  = cvtest::clipInt( max_log_img_width, 1, useVeryWideImages ? 17 : 10 );
    min_log_img_height = cvtest::clipInt( min_log_img_height, 1, 10 );
    min_log_img_height = cvtest::clipInt( max_log_img_height, 1, 10 );

    if( min_log_img_width > max_log_img_width )
        std::swap( min_log_img_width, max_log_img_width );

    if (min_log_img_height > max_log_img_height)
        std::swap(min_log_img_height, max_log_img_height);

    return 0;
}


static void
cvTsGenerateBlobImage( IplImage* img, int min_blob_size, int max_blob_size,
                       int blob_count, int min_brightness, int max_brightness,
                       RNG& rng )
{
    int i;
    Size size;

    CV_Assert(img->depth == IPL_DEPTH_8U && img->nChannels == 1);

    cvZero( img );

    // keep the border clear
    cvSetImageROI( img, cvRect(1,1,img->width-2,img->height-2) );
    size = cvGetSize( img );

    for( i = 0; i < blob_count; i++ )
    {
        Point center;
        Size  axes;
        int angle = cvtest::randInt(rng) % 180;
        int brightness = cvtest::randInt(rng) %
                         (max_brightness - min_brightness) + min_brightness;
        center.x = cvtest::randInt(rng) % size.width;
        center.y = cvtest::randInt(rng) % size.height;

        axes.width = (cvtest::randInt(rng) %
                     (max_blob_size - min_blob_size) + min_blob_size + 1)/2;
        axes.height = (cvtest::randInt(rng) %
                      (max_blob_size - min_blob_size) + min_blob_size + 1)/2;

        cvEllipse( img, cvPoint(center), cvSize(axes), angle, 0, 360, cvScalar(brightness), CV_FILLED );
    }

    cvResetImageROI( img );
}


static void
cvTsMarkContours( IplImage* img, int val )
{
    int i, j;
    int step = img->widthStep;

    assert( img->depth == IPL_DEPTH_8U && img->nChannels == 1 && (val&1) != 0);

    for( i = 1; i < img->height - 1; i++ )
        for( j = 1; j < img->width - 1; j++ )
        {
            uchar* t = (uchar*)(img->imageData + img->widthStep*i + j);
            if( *t == 1 && (t[-step] == 0 || t[-1] == 0 || t[1] == 0 || t[step] == 0))
                *t = (uchar)val;
        }

    cvThreshold( img, img, val - 2, val, CV_THRESH_BINARY );
}


int CV_FindContourTest::prepare_test_case( int test_case_idx )
{
    RNG& rng = ts->get_rng();
    const int  min_brightness = 0, max_brightness = 2;
    int i, code = cvtest::BaseTest::prepare_test_case( test_case_idx );

    if( code < 0 )
        return code;

    clear();

    blob_count = cvRound(exp(cvtest::randReal(rng)*max_log_blob_count*CV_LOG2));

    img_size.width = cvRound(exp((cvtest::randReal(rng)*
        (max_log_img_width - min_log_img_width) + min_log_img_width)*CV_LOG2));
    img_size.height = cvRound(exp((cvtest::randReal(rng)*
        (max_log_img_height - min_log_img_height) + min_log_img_height)*CV_LOG2));

    approx_method = cvtest::randInt( rng ) % 4 + 1;
    retr_mode = cvtest::randInt( rng ) % 4;

    storage = cvCreateMemStorage( 1 << 10 );

    for( i = 0; i < NUM_IMG; i++ )
        img[i] = cvCreateImage( cvSize(img_size), 8, 1 );

    cvTsGenerateBlobImage( img[0], min_blob_size, max_blob_size,
        blob_count, min_brightness, max_brightness, rng );

    cvCopy( img[0], img[1] );
    cvCopy( img[0], img[2] );

    cvTsMarkContours( img[1], 255 );

    return 1;
}


void CV_FindContourTest::run_func()
{
    contours = contours2 = chain = 0;
    count = cvFindContours( img[2], storage, &contours, sizeof(CvContour), retr_mode, approx_method );

    cvZero( img[3] );

    if( contours && retr_mode != CV_RETR_EXTERNAL && approx_method < CV_CHAIN_APPROX_TC89_L1 )
        cvDrawContours( img[3], contours, cvScalar(255), cvScalar(255), INT_MAX, -1 );

    cvCopy( img[0], img[2] );

    count2 = cvFindContours( img[2], storage, &chain, sizeof(CvChain), retr_mode, CV_CHAIN_CODE );

    if( chain )
        contours2 = cvApproxChains( chain, storage, approx_method, 0, 0, 1 );

    cvZero( img[2] );

    if( contours && retr_mode != CV_RETR_EXTERNAL && approx_method < CV_CHAIN_APPROX_TC89_L1 )
        cvDrawContours( img[2], contours2, cvScalar(255), cvScalar(255), INT_MAX );
}


// the whole testing is done here, run_func() is not utilized in this test
int CV_FindContourTest::validate_test_results( int /*test_case_idx*/ )
{
    int code = cvtest::TS::OK;

    cvCmpS( img[0], 0, img[0], CV_CMP_GT );

    if( count != count2 )
    {
        ts->printf( cvtest::TS::LOG, "The number of contours retrieved with different "
            "approximation methods is not the same\n"
            "(%d contour(s) for method %d vs %d contour(s) for method %d)\n",
            count, approx_method, count2, CV_CHAIN_CODE );
        code = cvtest::TS::FAIL_INVALID_OUTPUT;
    }

    if( retr_mode != CV_RETR_EXTERNAL && approx_method < CV_CHAIN_APPROX_TC89_L1 )
    {
        Mat _img[4];
        for( int i = 0; i < 4; i++ )
            _img[i] = cvarrToMat(img[i]);

        code = cvtest::cmpEps2(ts, _img[0], _img[3], 0, true, "Comparing original image with the map of filled contours" );

        if( code < 0 )
            goto _exit_;

        code = cvtest::cmpEps2( ts, _img[1], _img[2], 0, true,
            "Comparing contour outline vs manually produced edge map" );

        if( code < 0 )
            goto _exit_;
    }

    if( contours )
    {
        CvTreeNodeIterator iterator1;
        CvTreeNodeIterator iterator2;
        int count3;

        for(int i = 0; i < 2; i++ )
        {
            CvTreeNodeIterator iterator;
            cvInitTreeNodeIterator( &iterator, i == 0 ? contours : contours2, INT_MAX );

            for( count3 = 0; cvNextTreeNode( &iterator ) != 0; count3++ )
                ;

            if( count3 != count )
            {
                ts->printf( cvtest::TS::LOG,
                    "The returned number of retrieved contours (using the approx_method = %d) does not match\n"
                    "to the actual number of contours in the tree/list (returned %d, actual %d)\n",
                    i == 0 ? approx_method : 0, count, count3 );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }
        }

        cvInitTreeNodeIterator( &iterator1, contours, INT_MAX );
        cvInitTreeNodeIterator( &iterator2, contours2, INT_MAX );

        for( count3 = 0; count3 < count; count3++ )
        {
            CvSeq* seq1 = (CvSeq*)cvNextTreeNode( &iterator1 );
            CvSeq* seq2 = (CvSeq*)cvNextTreeNode( &iterator2 );
            CvSeqReader reader1;
            CvSeqReader reader2;

            if( !seq1 || !seq2 )
            {
                ts->printf( cvtest::TS::LOG,
                    "There are NULL pointers in the original contour tree or the "
                    "tree produced by cvApproxChains\n" );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }

            cvStartReadSeq( seq1, &reader1 );
            cvStartReadSeq( seq2, &reader2 );

            if( seq1->total != seq2->total )
            {
                ts->printf( cvtest::TS::LOG,
                    "The original contour #%d has %d points, while the corresponding contour has %d point\n",
                    count3, seq1->total, seq2->total );
                code = cvtest::TS::FAIL_INVALID_OUTPUT;
                goto _exit_;
            }

            for(int i = 0; i < seq1->total; i++ )
            {
                CvPoint pt1 = {0, 0};
                CvPoint pt2 = {0, 0};

                CV_READ_SEQ_ELEM( pt1, reader1 );
                CV_READ_SEQ_ELEM( pt2, reader2 );

                if( pt1.x != pt2.x || pt1.y != pt2.y )
                {
                    ts->printf( cvtest::TS::LOG,
                    "The point #%d in the contour #%d is different from the corresponding point "
                    "in the approximated chain ((%d,%d) vs (%d,%d)", count3, i, pt1.x, pt1.y, pt2.x, pt2.y );
                    code = cvtest::TS::FAIL_INVALID_OUTPUT;
                    goto _exit_;
                }
            }
        }
    }

_exit_:
    if( code < 0 )
    {
#if 0
        cvNamedWindow( "test", 0 );
        cvShowImage( "test", img[0] );
        cvWaitKey();
#endif
        ts->set_failed_test_info( code );
    }

    return code;
}

TEST(Imgproc_FindContours, accuracy) { CV_FindContourTest test; test.safe_run(); }

//rotate/flip a quadrant appropriately
static void rot(int n, int *x, int *y, int rx, int ry)
{
    if (ry == 0) {
        if (rx == 1) {
            *x = n-1 - *x;
            *y = n-1 - *y;
        }

        //Swap x and y
        int t  = *x;
        *x = *y;
        *y = t;
    }
}

static void d2xy(int n, int d, int *x, int *y)
{
    int rx, ry, s, t=d;
    *x = *y = 0;
    for (s=1; s<n; s*=2)
    {
        rx = 1 & (t/2);
        ry = 1 & (t ^ rx);
        rot(s, x, y, rx, ry);
        *x += s * rx;
        *y += s * ry;
        t /= 4;
    }
}

TEST(Imgproc_FindContours, hilbert)
{
    int n = 64, n2 = n*n, scale = 10, w = (n + 2)*scale;
    Point ofs(scale, scale);
    Mat img(w, w, CV_8U);
    img.setTo(Scalar::all(0));

    Point p(0,0);
    for( int i = 0; i < n2; i++ )
    {
        Point q(0,0);
        d2xy(n2, i, &q.x, &q.y);
        line(img, p*scale + ofs, q*scale + ofs, Scalar::all(255));
        p = q;
    }
    dilate(img, img, Mat());
    vector<vector<Point> > contours;
    findContours(img, contours, noArray(), RETR_LIST, CHAIN_APPROX_SIMPLE);
    printf("ncontours = %d, contour[0].npoints=%d\n", (int)contours.size(), (int)contours[0].size());
    img.setTo(Scalar::all(0));

    drawContours(img, contours, 0, Scalar::all(255), 1);

    ASSERT_EQ(1, (int)contours.size());
    ASSERT_EQ(9832, (int)contours[0].size());
}

TEST(Imgproc_FindContours, border)
{
    Mat img;
    cv::copyMakeBorder(Mat::zeros(8, 10, CV_8U), img, 1, 1, 1, 1, BORDER_CONSTANT, Scalar(1));

    std::vector<std::vector<cv::Point> > contours;
    findContours(img, contours, RETR_LIST, CHAIN_APPROX_NONE);

    Mat img_draw_contours = Mat::zeros(img.size(), CV_8U);
    for (size_t cpt = 0; cpt < contours.size(); cpt++)
    {
      drawContours(img_draw_contours, contours, static_cast<int>(cpt), cv::Scalar(1));
    }

    ASSERT_EQ(0, cvtest::norm(img, img_draw_contours, NORM_INF));
}

TEST(Imgproc_PointPolygonTest, regression_10222)
{
    vector<Point> contour;
    contour.push_back(Point(0, 0));
    contour.push_back(Point(0, 100000));
    contour.push_back(Point(100000, 100000));
    contour.push_back(Point(100000, 50000));
    contour.push_back(Point(100000, 0));

    const Point2f point(40000, 40000);
    const double result = cv::pointPolygonTest(contour, point, false);
    EXPECT_GT(result, 0) << "Desired result: point is inside polygon - actual result: point is not inside polygon";
}

}} // namespace
/* End of file. */
