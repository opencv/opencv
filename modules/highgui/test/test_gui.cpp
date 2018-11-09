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

namespace opencv_test { namespace {

#if defined HAVE_GTK || defined HAVE_QT || defined HAVE_WIN32UI || defined HAVE_COCOA

class CV_HighGuiOnlyGuiTest : public cvtest::BaseTest
{
protected:
    void run(int);
};

static void Foo(int /*k*/, void* /*z*/) {}

void CV_HighGuiOnlyGuiTest::run( int /*start_from */)
{
    ts->printf(ts->LOG, "GUI 0\n");
    destroyAllWindows();

    ts->printf(ts->LOG, "GUI 1\n");
    namedWindow("Win");

    ts->printf(ts->LOG, "GUI 2\n");
    Mat m(256, 256, CV_8U);
    m = Scalar(128);

    ts->printf(ts->LOG, "GUI 3\n");
    imshow("Win", m);

    ts->printf(ts->LOG, "GUI 4\n");
    int value = 50;

    ts->printf(ts->LOG, "GUI 5\n");
    createTrackbar( "trackbar", "Win", &value, 100, Foo, &value);

    ts->printf(ts->LOG, "GUI 6\n");
    getTrackbarPos( "trackbar", "Win" );

    ts->printf(ts->LOG, "GUI 7\n");
    waitKey(500);

    ts->printf(ts->LOG, "GUI 8\n");
    Rect rc = getWindowImageRect("Win");
    std::cout << "window image rect: " << rc << std::endl;

    ts->printf(ts->LOG, "GUI 9\n");
    destroyAllWindows();
    ts->set_failed_test_info(cvtest::TS::OK);
}

TEST(Highgui_GUI,    regression) { CV_HighGuiOnlyGuiTest test; test.safe_run(); }

#endif

}} // namespace
