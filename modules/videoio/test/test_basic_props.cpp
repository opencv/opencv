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
#include "opencv2/videoio.hpp"
#include "opencv2/ts.hpp"
#include <stdio.h>

#if BUILD_WITH_VIDEO_INPUT_SUPPORT

using namespace cv;
using namespace std;
using namespace cvtest;

#ifdef HAVE_GSTREAMER
const string ext[] = {"avi"};
#else
const string ext[] = {"avi", "mov", "mp4"};
#endif

TEST(Videoio_Video, prop_resolution)
{
    const size_t n = sizeof(ext)/sizeof(ext[0]);
    const string src_dir = TS::ptr()->get_data_path();

    TS::ptr()->printf(cvtest::TS::LOG, "\n\nSource files directory: %s\n", (src_dir+"video/").c_str());

    for (size_t i = 0; i < n; ++i)
    {
        string file_path = src_dir+"video/big_buck_bunny."+ext[i];
        VideoCapture cap(file_path);
        if (!cap.isOpened())
        {
            TS::ptr()->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
            TS::ptr()->printf(cvtest::TS::LOG, "Error: cannot read source video file.\n");
            TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        ASSERT_EQ(672, cap.get(CAP_PROP_FRAME_WIDTH));
        ASSERT_EQ(384, cap.get(CAP_PROP_FRAME_HEIGHT));
    }
}

TEST(Videoio_Video, actual_resolution)
{
    const size_t n = sizeof(ext)/sizeof(ext[0]);
    const string src_dir = TS::ptr()->get_data_path();

    TS::ptr()->printf(cvtest::TS::LOG, "\n\nSource files directory: %s\n", (src_dir+"video/").c_str());

    for (size_t i = 0; i < n; ++i)
    {
        string file_path = src_dir+"video/big_buck_bunny."+ext[i];
        VideoCapture cap(file_path);
        if (!cap.isOpened())
        {
            TS::ptr()->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
            TS::ptr()->printf(cvtest::TS::LOG, "Error: cannot read source video file.\n");
            TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        Mat frame;
        cap >> frame;

        ASSERT_EQ(672, frame.cols);
        ASSERT_EQ(384, frame.rows);
    }
}

TEST(Videoio_Video, DISABLED_prop_fps)
{
    const size_t n = sizeof(ext)/sizeof(ext[0]);
    const string src_dir = TS::ptr()->get_data_path();

    TS::ptr()->printf(cvtest::TS::LOG, "\n\nSource files directory: %s\n", (src_dir+"video/").c_str());

    for (size_t i = 0; i < n; ++i)
    {
        string file_path = src_dir+"video/big_buck_bunny."+ext[i];
        VideoCapture cap(file_path);
        if (!cap.isOpened())
        {
            TS::ptr()->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
            TS::ptr()->printf(cvtest::TS::LOG, "Error: cannot read source video file.\n");
            TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        ASSERT_EQ(24, cap.get(CAP_PROP_FPS));
    }
}

TEST(Videoio_Video, prop_framecount)
{
    const size_t n = sizeof(ext)/sizeof(ext[0]);
    const string src_dir = TS::ptr()->get_data_path();

    TS::ptr()->printf(cvtest::TS::LOG, "\n\nSource files directory: %s\n", (src_dir+"video/").c_str());

    for (size_t i = 0; i < n; ++i)
    {
        string file_path = src_dir+"video/big_buck_bunny."+ext[i];
        VideoCapture cap(file_path);
        if (!cap.isOpened())
        {
            TS::ptr()->printf(cvtest::TS::LOG, "\nFile information (video %d): \n\nName: big_buck_bunny.%s\nFAILED\n\n", i+1, ext[i].c_str());
            TS::ptr()->printf(cvtest::TS::LOG, "Error: cannot read source video file.\n");
            TS::ptr()->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            return;
        }

        ASSERT_EQ(125, cap.get(CAP_PROP_FRAME_COUNT));
    }
}

#endif
