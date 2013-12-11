/*M///////////////////////////////////////////////////////////////////////////////////////
 //
 //  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
 //
 //  By downloading, copying, installing or using the software you agree to this license.
 //  If you do not agree to this license, do not download, install,
 //  copy or use the software.
 //
 //
 //                          License Agreement
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
#include "opencv2/highgui.hpp"

using namespace cv;
using namespace std;

class CV_PositioningTest : public cvtest::BaseTest
{
public:
    CV_PositioningTest()
    {
        framesize = Size(640, 480);
    }

    Mat drawFrame(int i)
    {
        Mat mat = Mat::zeros(framesize, CV_8UC3);

        mat = Scalar(fabs(cos(i*0.08)*255), fabs(sin(i*0.05)*255), i);
        putText(mat, format("%03d", i), Point(10, 350), 0, 10, Scalar(128, 255, 255), 15);
        return mat;
    }

    string getFilename(const cvtest::VideoFormat& fmt)
    {
        return cv::tempfile((cvtest::fourccToString(fmt.fourcc) + "." + fmt.ext).c_str());
    }

    bool CreateTestVideo(const cvtest::VideoFormat& fmt, int framecount, string filename)
    {
        VideoWriter writer(filename, fmt.fourcc, 25, framesize, true);
        if( !writer.isOpened() )
            return false;

        for (int i = 0; i < framecount; ++i)
        {
            Mat img = drawFrame(i);
            writer << img;
        }
        return true;
    }

    void run(int)
    {
        int n_frames = 100;

        for( int testcase = 0; ; testcase++ )
        {
            const cvtest::VideoFormat& fmt = cvtest::g_specific_fmt_list[testcase];
            if( fmt.empty() )
                break;
            string filename = getFilename(fmt);
            ts->printf(ts->LOG, "\nFile: %s\n", filename.c_str());

            if( !CreateTestVideo(fmt, n_frames, filename) )
            {
                ts->printf(ts->LOG, "\nError: cannot create video file");
                ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                return;
            }

            VideoCapture cap(filename);

            if (!cap.isOpened())
            {
                ts->printf(ts->LOG, "\nError: cannot read video file.");
                ts->set_failed_test_info(ts->FAIL_INVALID_TEST_DATA);
                return;
            }

            int N0 = (int)cap.get(CAP_PROP_FRAME_COUNT);
            cap.set(CAP_PROP_POS_FRAMES, 0);
            int N = (int)cap.get(CAP_PROP_FRAME_COUNT);

            // See the same hack in CV_HighGuiTest::SpecificVideoTest for explanation.
            int allowed_extra_frames = 0;
            if (fmt.fourcc == VideoWriter::fourcc('M', 'P', 'E', 'G') && fmt.ext == "mkv")
                allowed_extra_frames = 1;

            if (N < n_frames || N > n_frames + allowed_extra_frames || N != N0)
            {
                ts->printf(ts->LOG, "\nError: returned frame count (N0=%d, N=%d) is different from the reference number %d\n", N0, N, n_frames);
                ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                return;
            }

            for (int k = 0; k < n_frames; ++k)
            {
                int idx = theRNG().uniform(0, n_frames);

                if( !cap.set(CAP_PROP_POS_FRAMES, idx) )
                {
                    ts->printf(ts->LOG, "\nError: cannot seek to frame %d.\n", idx);
                    ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                    return;
                }

                int idx1 = (int)cap.get(CAP_PROP_POS_FRAMES);

                Mat img; cap >> img;
                Mat img0 = drawFrame(idx);

                if( idx != idx1 )
                {
                    ts->printf(ts->LOG, "\nError: the current position (%d) after seek is different from specified (%d)\n",
                               idx1, idx);
                    ts->printf(ts->LOG, "Saving both frames ...\n");
                    ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                    // imwrite("opencv_test_highgui_postest_actual.png", img);
                    // imwrite("opencv_test_highgui_postest_expected.png", img0);
                    return;
                }

                if (img.empty())
                {
                    ts->printf(ts->LOG, "\nError: cannot read a frame at position %d.\n", idx);
                    ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                    return;
                }

                double err = PSNR(img, img0);

                if( err < 20 )
                {
                    ts->printf(ts->LOG, "The frame read after positioning to %d is incorrect (PSNR=%g)\n", idx, err);
                    ts->printf(ts->LOG, "Saving both frames ...\n");
                    ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
                    // imwrite("opencv_test_highgui_postest_actual.png", img);
                    // imwrite("opencv_test_highgui_postest_expected.png", img0);
                    return;
                }
            }
        }
    }

    Size framesize;
};

#if BUILD_WITH_VIDEO_INPUT_SUPPORT && BUILD_WITH_VIDEO_OUTPUT_SUPPORT && defined HAVE_FFMPEG
TEST(Highgui_Video, seek_random_synthetic) { CV_PositioningTest test; test.safe_run(); }
#endif
