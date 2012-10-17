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
#include "test_chessboardgenerator.hpp"

#include <limits>

using namespace std;
using namespace cv;

class CV_ChessboardDetectorBadArgTest : public cvtest::BadArgTest
{
public:
    CV_ChessboardDetectorBadArgTest();
protected:
    void run(int);
    bool checkByGenerator();

    bool cpp;

    /* cpp interface */
    Mat img;
    Size pattern_size;
    int flags;
    vector<Point2f> corners;

    /* c interface */
    CvMat arr;
    CvPoint2D32f* out_corners;
    int* out_corner_count;


    /* c interface draw  corners */
    bool drawCorners;
    CvMat drawCorImg;
    bool was_found;

    void run_func()
    {
        if (cpp)
            findChessboardCorners(img, pattern_size, corners, flags);
        else
            if (!drawCorners)
                cvFindChessboardCorners( &arr, pattern_size, out_corners, out_corner_count, flags );
            else
                cvDrawChessboardCorners( &drawCorImg, pattern_size,
                    (CvPoint2D32f*)(corners.empty() ? 0 : &corners[0]),
                    (int)corners.size(), was_found);
    }
};

CV_ChessboardDetectorBadArgTest::CV_ChessboardDetectorBadArgTest() {}

/* ///////////////////// chess_corner_test ///////////////////////// */
void CV_ChessboardDetectorBadArgTest::run( int /*start_from */)
{
    Mat bg(800, 600, CV_8U, Scalar(0));
    Mat_<float> camMat(3, 3);
    camMat << 300.f, 0.f, bg.cols/2.f, 0, 300.f, bg.rows/2.f, 0.f, 0.f, 1.f;
    Mat_<float> distCoeffs(1, 5);
    distCoeffs << 1.2f, 0.2f, 0.f, 0.f, 0.f;

    ChessBoardGenerator cbg(Size(8,6));
    vector<Point2f> exp_corn;
    Mat cb = cbg(bg, camMat, distCoeffs, exp_corn);

    /* /*//*/ */
    int errors = 0;
    flags = CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_NORMALIZE_IMAGE;
    cpp = true;

    img = cb.clone();
    pattern_size = Size(2,2);
    errors += run_test_case( CV_StsOutOfRange, "Invlid pattern size" );

    pattern_size = cbg.cornersSize();
    cb.convertTo(img, CV_32F);
    errors += run_test_case( CV_StsUnsupportedFormat, "Not 8-bit image" );

    cv::merge(vector<Mat>(2, cb), img);
    errors += run_test_case( CV_StsUnsupportedFormat, "2 channel image" );

    cpp = false;
    drawCorners = false;

    img = cb.clone();
    arr = img;
    out_corner_count = 0;
    out_corners = 0;
    errors += run_test_case( CV_StsNullPtr, "Null pointer to corners" );

    drawCorners = true;
    Mat cvdrawCornImg(img.size(), CV_8UC2);
    drawCorImg = cvdrawCornImg;
    was_found = true;
    errors += run_test_case( CV_StsUnsupportedFormat, "2 channel image" );


    if (errors)
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
    else
        ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Calib3d_ChessboardDetector, badarg) { CV_ChessboardDetectorBadArgTest test; test.safe_run(); }

/* End of file. */
