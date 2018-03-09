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

class Core_ConcatenationTest : public cvtest::BaseTest
{
public:
    Core_ConcatenationTest(bool horizontal, bool firstEmpty, bool secondEmpty);
protected:
    int prepare_test_case( int );
    void run_func();
    int validate_test_results( int );

    Mat mat0x5;
    Mat mat10x5;
    Mat mat20x5;

    Mat mat5x0;
    Mat mat5x10;
    Mat mat5x20;

    Mat result;

    bool horizontal;
    bool firstEmpty;
    bool secondEmpty;

private:
    static bool areEqual(const Mat& m1, const Mat& m2);

};

Core_ConcatenationTest::Core_ConcatenationTest(bool horizontal_, bool firstEmpty_, bool secondEmpty_)
    : horizontal(horizontal_)
    , firstEmpty(firstEmpty_)
    , secondEmpty(secondEmpty_)
{
    test_case_count = 1;

    mat0x5 = Mat::ones(0,5, CV_8U);
    mat10x5 = Mat::ones(10,5, CV_8U);
    mat20x5 = Mat::ones(20,5, CV_8U);

    mat5x0 = Mat::ones(5,0, CV_8U);
    mat5x10 = Mat::ones(5,10, CV_8U);
    mat5x20 = Mat::ones(5,20, CV_8U);
}

int Core_ConcatenationTest::prepare_test_case( int test_case_idx )
{
    cvtest::BaseTest::prepare_test_case( test_case_idx );
    return 1;
}

void Core_ConcatenationTest::run_func()
{
    if (horizontal)
    {
        cv::hconcat((firstEmpty ? mat5x0 : mat5x10),
                    (secondEmpty ? mat5x0 : mat5x10),
                    result);
    } else {
        cv::vconcat((firstEmpty ? mat0x5 : mat10x5),
                    (secondEmpty ? mat0x5 : mat10x5),
                    result);
    }
}

int Core_ConcatenationTest::validate_test_results( int )
{
    Mat expected;

    if (firstEmpty && secondEmpty)
        expected = (horizontal ? mat5x0 : mat0x5);
    else if ((firstEmpty && !secondEmpty) || (!firstEmpty && secondEmpty))
        expected = (horizontal ? mat5x10 : mat10x5);
    else
        expected = (horizontal ? mat5x20 : mat20x5);

    if (areEqual(expected, result))
    {
        return cvtest::TS::OK;
    } else
    {
        ts->printf( cvtest::TS::LOG, "Concatenation failed");
        ts->set_failed_test_info( cvtest::TS::FAIL_MISMATCH );
    }

    return cvtest::TS::OK;
}

bool Core_ConcatenationTest::areEqual(const Mat &m1, const Mat &m2)
{
    return m1.size() == m2.size()
            && m1.type() == m2.type()
            && countNonZero(m1 != m2) == 0;
}

TEST(Core_Concatenation, hconcat_empty_nonempty) { Core_ConcatenationTest test(true, true, false); test.safe_run(); }
TEST(Core_Concatenation, hconcat_nonempty_empty) { Core_ConcatenationTest test(true, false, true); test.safe_run(); }
TEST(Core_Concatenation, hconcat_empty_empty) { Core_ConcatenationTest test(true, true, true); test.safe_run(); }

TEST(Core_Concatenation, vconcat_empty_nonempty) { Core_ConcatenationTest test(false, true, false); test.safe_run(); }
TEST(Core_Concatenation, vconcat_nonempty_empty) { Core_ConcatenationTest test(false, false, true); test.safe_run(); }
TEST(Core_Concatenation, vconcat_empty_empty) { Core_ConcatenationTest test(false, true, true); test.safe_run(); }

}} // namespace
