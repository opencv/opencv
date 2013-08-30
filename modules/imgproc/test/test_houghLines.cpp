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

using namespace cv;
using namespace std;

class CV_HoughLinesTest : public cvtest::BaseTest
{
public:
    enum {STANDART = 0, PROBABILISTIC};
    CV_HoughLinesTest() {}
    ~CV_HoughLinesTest() {}
protected:
    void run_test(int type);
};

class CV_StandartHoughLinesTest : public CV_HoughLinesTest
{
public:
    CV_StandartHoughLinesTest() {}
    ~CV_StandartHoughLinesTest() {}
    virtual void run(int);
};

class CV_ProbabilisticHoughLinesTest : public CV_HoughLinesTest
{
public:
    CV_ProbabilisticHoughLinesTest() {}
    ~CV_ProbabilisticHoughLinesTest() {}
    virtual void run(int);
};

void CV_StandartHoughLinesTest::run(int)
{
    run_test(STANDART);
}

void CV_ProbabilisticHoughLinesTest::run(int)
{
    run_test(PROBABILISTIC);
}

void CV_HoughLinesTest::run_test(int type)
{
    Mat src = imread(string(ts->get_data_path()) + "shared/pic1.png");
    if (src.empty())
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
        return;
    }

    string xml;
    if (type == STANDART)
        xml = string(ts->get_data_path()) + "imgproc/HoughLines.xml";
    else if (type == PROBABILISTIC)
        xml = string(ts->get_data_path()) + "imgproc/HoughLinesP.xml";
    else
    {
        ts->printf(cvtest::TS::LOG, "Error: unknown HoughLines algorithm type.\n");
        ts->set_failed_test_info(cvtest::TS::FAIL_GENERIC);
        return;
    }

    Mat dst;
    Canny(src, dst, 50, 200, 3);

    Mat lines;
    if (type == STANDART)
        HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0);
    else if (type == PROBABILISTIC)
        HoughLinesP(dst, lines, 1, CV_PI/180, 100, 0, 0);

    FileStorage fs(xml, FileStorage::READ);
    if (!fs.isOpened())
    {
        fs.open(xml, FileStorage::WRITE);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
        fs << "exp_lines" << lines;
        fs.release();
        fs.open(xml, FileStorage::READ);
        if (!fs.isOpened())
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }
    }

    Mat exp_lines;
    read( fs["exp_lines"], exp_lines, Mat() );
    fs.release();

    if ( exp_lines.size != lines.size || norm(exp_lines, lines, NORM_INF) > 1e-4 )
    {
        ts->set_failed_test_info(cvtest::TS::FAIL_MISMATCH);
        return;
    }

    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Imgproc_HoughLines, regression) { CV_StandartHoughLinesTest test; test.safe_run(); }

TEST(Imgproc_HoughLinesP, regression) { CV_ProbabilisticHoughLinesTest test; test.safe_run(); }
