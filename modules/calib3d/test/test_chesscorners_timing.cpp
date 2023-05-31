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
#include "opencv2/imgproc.hpp"
#include "opencv2/calib3d.hpp"

namespace opencv_test { namespace {

class CV_ChessboardDetectorTimingTest : public cvtest::BaseTest
{
public:
    CV_ChessboardDetectorTimingTest();
protected:
    void run(int);
};


CV_ChessboardDetectorTimingTest::CV_ChessboardDetectorTimingTest()
{
}

/* ///////////////////// chess_corner_test ///////////////////////// */
void CV_ChessboardDetectorTimingTest::run( int start_from )
{
    int code = cvtest::TS::OK;

    /* test parameters */
    std::string   filepath;
    std::string   filename;

    std::vector<Point2f> v;
    Mat img, gray, thresh;

    int  idx, max_idx;
    int  progress = 0;

    filepath = cv::format("%scv/cameracalibration/", ts->get_data_path().c_str() );
    filename = cv::format("%schessboard_timing_list.dat", filepath.c_str() );
    cv::FileStorage fs( filename, FileStorage::READ );
    cv::FileNode board_list = fs["boards"];
    cv::FileNodeIterator bl_it = board_list.begin();

    if( !fs.isOpened() || !board_list.isSeq() || board_list.size() % 4 != 0 )
    {
        ts->printf( cvtest::TS::LOG, "chessboard_timing_list.dat can not be read or is not valid" );
        code = cvtest::TS::FAIL_MISSING_TEST_DATA;
        goto _exit_;
    }

    max_idx = (int)(board_list.size()/4);
    for( idx = 0; idx < start_from; idx++ )
    {
        bl_it += 4;
    }

    for( idx = start_from; idx < max_idx; idx++ )
    {
        Size pattern_size;

        std::string imgname; read(*bl_it++, imgname, "dummy.txt");
        int is_chessboard = 0;
        read(*bl_it++, is_chessboard, 0);
        read(*bl_it++, pattern_size.width, -1);
        read(*bl_it++, pattern_size.height, -1);

        ts->update_context( this, idx-1, true );

        /* read the image */
        filename = cv::format("%s%s", filepath.c_str(), imgname.c_str() );

        img = cv::imread( filename );
        if( img.empty() )
        {
            ts->printf( cvtest::TS::LOG, "one of chessboard images can't be read: %s\n", filename.c_str() );
            code = cvtest::TS::FAIL_MISSING_TEST_DATA;
            continue;
        }

        ts->printf(cvtest::TS::LOG, "%s: chessboard %d:\n", imgname.c_str(), is_chessboard);

        cvtColor(img, gray, COLOR_BGR2GRAY);

        int64 _time0 = cv::getTickCount();
        bool result = cv::checkChessboard(gray, pattern_size);
        int64 _time01 = cv::getTickCount();
        bool result1 = findChessboardCorners(gray, pattern_size, v, 15);
        int64 _time1 = cv::getTickCount();

        if( result != (is_chessboard != 0))
        {
            ts->printf( cvtest::TS::LOG, "Error: chessboard was %sdetected in the image %s\n",
                       result ? "" : "not ", imgname.c_str() );
            code = cvtest::TS::FAIL_INVALID_OUTPUT;
            goto _exit_;
        }
        if(result != result1)
        {
            ts->printf( cvtest::TS::LOG, "Warning: results differ cvCheckChessboard %d, cvFindChessboardCorners %d\n",
                       (int)result, (int)result1);
        }

        int num_pixels = gray.cols*gray.rows;
        float check_chessboard_time = float(_time01 - _time0)/(float)cv::getTickFrequency(); // in us
        ts->printf(cvtest::TS::LOG, "    cvCheckChessboard time s: %f, us per pixel: %f\n",
                   check_chessboard_time*1e-6, check_chessboard_time/num_pixels);

        float find_chessboard_time = float(_time1 - _time01)/(float)cv::getTickFrequency();
        ts->printf(cvtest::TS::LOG, "    cvFindChessboard time s: %f, us per pixel: %f\n",
                   find_chessboard_time*1e-6, find_chessboard_time/num_pixels);
        progress = update_progress( progress, idx-1, max_idx, 0 );
    }

_exit_:

    if( code < 0 )
        ts->set_failed_test_info( code );
}

TEST(Calib3d_ChessboardDetector, timing) { CV_ChessboardDetectorTimingTest test; test.safe_run(); }

}} // namespace
/* End of file. */
