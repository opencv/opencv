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
#include "opencv2/videoio/videoio_c.h"
#include "opencv2/highgui.hpp"
#include <cstdio>

using namespace cv;
using namespace std;
using namespace std::tr1;

class Videoio_Test_Base
{
protected:
    string ext;
    string video_file;
protected:
    Videoio_Test_Base() {}
    virtual ~Videoio_Test_Base() {}
    virtual void checkFrameContent(Mat &, int) {}
    virtual void checkFrameCount(int &) {}
    void checkFrameRead(int idx, VideoCapture & cap)
    {
        Mat img; cap >> img;
        ASSERT_FALSE(img.empty());
        checkFrameContent(img, idx);
    }
    void checkFrameSeek(int idx, VideoCapture & cap)
    {
        ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, idx));
        ASSERT_EQ(idx, (int)cap.get(CAP_PROP_POS_FRAMES));
        checkFrameRead(idx, cap);
    }
public:
    void doTest()
    {
        VideoCapture cap(video_file);
        ASSERT_TRUE(cap.isOpened());

        int n_frames = (int)cap.get(CAP_PROP_FRAME_COUNT);
        ASSERT_GT(n_frames, 0);
        checkFrameCount(n_frames);

        {
            SCOPED_TRACE("consecutive read");
            for (int k = 0; k < n_frames; ++k)
            {
                checkFrameRead(k, cap);
            }
        }

        if (ext != "mpg" && ext != "wmv")
        {
            SCOPED_TRACE("random seek");
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, 0));
            for (int k = 0; k < 10; ++k)
            {
                checkFrameSeek(cvtest::TS::ptr()->get_rng().uniform(0, n_frames), cap);
            }
        }

        if (ext != "wmv")
        {
            SCOPED_TRACE("progressive seek");
            ASSERT_TRUE(cap.set(CAP_PROP_POS_FRAMES, 0));
            for (int k = 1; k < n_frames; k += 20)
            {
                checkFrameSeek(k, cap);
            }
        }
    }
};

//==================================================================================================

class Videoio_Bunny : public Videoio_Test_Base, public testing::TestWithParam<string>
{
public:
    Videoio_Bunny()
    {
        ext = GetParam();
        video_file = cvtest::TS::ptr()->get_data_path() + "video/big_buck_bunny." + ext;
    }
    void doFrameCountTest()
    {
        VideoCapture cap(video_file);
        ASSERT_TRUE(cap.isOpened());

        const int width_gt = 672;
        const int height_gt = 384;
        const int fps_gt = 24;
        const double time_gt = 5.21;
        const int count_gt = cvRound(fps_gt * time_gt); // 5.21 sec * 24 fps

        EXPECT_EQ(width_gt, cap.get(CAP_PROP_FRAME_WIDTH));
        EXPECT_EQ(height_gt, cap.get(CAP_PROP_FRAME_HEIGHT));

        int fps_prop = (int)cap.get(CAP_PROP_FPS);
        EXPECT_EQ(fps_gt, fps_prop);

        int count_prop = (int)cap.get(CAP_PROP_FRAME_COUNT);
        ASSERT_GT(count_prop, 0);
        // mpg file reports 5.08 sec * 24 fps => property returns 122 frames
        // but actual number of frames returned is 125
        if (ext != "mpg")
        {
            EXPECT_EQ(count_gt, count_prop);
        }

        int count_actual = 0;
        while (cap.isOpened())
        {
            Mat frame;
            cap >> frame;
            if (frame.empty())
                break;
            EXPECT_EQ(width_gt, frame.cols);
            EXPECT_EQ(height_gt, frame.rows);
            count_actual += 1;
        }
        EXPECT_EQ(count_gt, count_actual);
    }
};

typedef tuple<string, string, float> Ext_Fourcc_PSNR;
typedef tuple<Size, Ext_Fourcc_PSNR> Size_Ext_Fourcc_PSNR;

class Videoio_Synthetic : public Videoio_Test_Base, public testing::TestWithParam<Size_Ext_Fourcc_PSNR>
{
    Size frame_size;
    int fourcc;
    float PSNR_GT;
    int frame_count;
    double fps;
public:
    Videoio_Synthetic()
    {
        frame_size = get<0>(GetParam());
        const Ext_Fourcc_PSNR &param = get<1>(GetParam());
        ext = get<0>(param);
        fourcc = fourccFromString(get<1>(param));
        PSNR_GT = get<2>(param);
        video_file = cv::tempfile((fourccToString(fourcc) + "." + ext).c_str());
        frame_count = 100;
        fps = 25.;
    }
    void SetUp()
    {
        Mat img(frame_size, CV_8UC3);
        VideoWriter writer(video_file, fourcc, fps, frame_size, true);
        ASSERT_TRUE(writer.isOpened());
        for(int i = 0; i < frame_count; ++i )
        {
            generateFrame(i, frame_count, img);
            writer << img;
        }
        writer.release();
    }
    void TearDown()
    {
        remove(video_file.c_str());
    }
    virtual void checkFrameContent(Mat & img, int idx)
    {
        Mat imgGT(frame_size, CV_8UC3);
        generateFrame(idx, frame_count, imgGT);
        double psnr = cvtest::PSNR(img, imgGT);
        ASSERT_GT(psnr, PSNR_GT) << "frame " << idx;
    }
    virtual void checkFrameCount(int &actual)
    {
        Range expected_frame_count = Range(frame_count, frame_count);

        // Hack! Newer FFmpeg versions in this combination produce a file
        // whose reported duration is one frame longer than needed, and so
        // the calculated frame count is also off by one. Ideally, we'd want
        // to fix both writing (to produce the correct duration) and reading
        // (to correctly report frame count for such files), but I don't know
        // how to do either, so this is a workaround for now.
        if (fourcc == VideoWriter::fourcc('M', 'P', 'E', 'G') && ext == "mkv")
            expected_frame_count.end += 1;

        // Hack! Some GStreamer encoding pipelines drop last frame in the video
#ifdef HAVE_GSTREAMER
        expected_frame_count.start -= 1;
#endif

        ASSERT_LE(expected_frame_count.start, actual);
        ASSERT_GE(expected_frame_count.end, actual);

        actual = expected_frame_count.start; // adjust actual frame boundary to possible minimum
    }
};

//==================================================================================================


string bunny_params[] = {
#ifdef HAVE_VIDEO_INPUT
    string("avi"),
    string("mov"),
    string("mp4"),
    string("mpg"),
    string("wmv"),
#endif
    string("mjpg.avi")
};

TEST_P(Videoio_Bunny, read_position) { doTest(); }

TEST_P(Videoio_Bunny, frame_count) { doFrameCountTest(); }

INSTANTIATE_TEST_CASE_P(videoio, Videoio_Bunny,
                        testing::ValuesIn(bunny_params));


//==================================================================================================

inline Ext_Fourcc_PSNR makeParam(const char * ext, const char * fourcc, float psnr)
{
    return make_tuple(string(ext), string(fourcc), (float)psnr);
}

Ext_Fourcc_PSNR synthetic_params[] = {

#if defined(HAVE_VIDEO_INPUT) && defined(HAVE_VIDEO_OUTPUT) && !defined(__APPLE__)

#ifdef HAVE_MSMF

#if !defined(_M_ARM)
    makeParam("wmv", "WMV1", 39.f),
    makeParam("wmv", "WMV2", 39.f),
#endif
    makeParam("wmv", "WMV3", 39.f),
    makeParam("avi", "H264", 39.f),
    makeParam("wmv", "WVC1", 39.f),

#else // HAVE_MSMF

    makeParam("avi", "XVID", 35.f),
    makeParam("avi", "MPEG", 35.f),
    makeParam("avi", "IYUV", 35.f),
    makeParam("mkv", "XVID", 35.f),
    makeParam("mkv", "MPEG", 35.f),
    makeParam("mkv", "MJPG", 35.f),
#ifndef HAVE_GSTREAMER
    makeParam("mov", "mp4v", 35.f),
#endif

#endif // HAVE_MSMF

#endif // HAVE_VIDEO_INPUT && HAVE_VIDEO_OUTPUT ...

    makeParam("avi", "MJPG", 41.f)
};

Size all_sizes[] = {
    Size(640, 480),
    Size(976, 768)
};

TEST_P(Videoio_Synthetic, write_read_position) { doTest(); }

INSTANTIATE_TEST_CASE_P(videoio, Videoio_Synthetic,
                        testing::Combine(
                            testing::ValuesIn(all_sizes),
                            testing::ValuesIn(synthetic_params)));
