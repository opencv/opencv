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
#include "opencv2/highgui/highgui.hpp"

#ifdef HAVE_FFMPEG

#include "ffmpeg_codecs.hpp"

using namespace cv;
using namespace std;

class CV_PositioningTest : public cvtest::BaseTest
{
public:
    void CreateTestVideo(const string& format, int codec, int framecount = 125);
    void run(int);
};

void CV_PositioningTest::CreateTestVideo(const string& format, int codec, int framecount)
{
 stringstream s; s << codec;

 cv::VideoWriter writer(ts->get_data_path()+"video/test_video_"+s.str()+"."+format, codec, 25, cv::Size(640, 480), false);

 for (int i = 0; i < framecount; ++i)
 {
   cv::Mat mat(480, 640, CV_8UC1);
   size_t n = 32, tmp = i;

   vector<char> tmp_code; tmp_code.clear();

   while ( tmp > 1 )
   {
       tmp_code.push_back(tmp%2);
       tmp /= 2;
   }
   tmp_code.push_back(tmp);

   vector<char> i_code;

   for (size_t j = 0; j < n; ++j)
   {
    char val = j < n - tmp_code.size() ? 0 : tmp_code.at(n-1-j);
    i_code.push_back(val);
   }

   const size_t w = 480/n;

   for (size_t j = 0; j < n; ++j)
   {
       cv::Scalar color = i_code[j] ? 255 : 0;
       rectangle(mat, Rect(0, w*j, 640, w), color, -1);
   }

   writer << mat;
 }

 writer.~VideoWriter();
}

void CV_PositioningTest::run(int)
{
#if defined WIN32 || (defined __linux__ && !defined ANDROID)
#if !defined HAVE_GSTREAMER || defined HAVE_GSTREAMER_APP

    const string format[] =  {"avi", "mov", "mp4", "mpg", "wmv", "3gp"};

    const char codec[][4] = { {'X', 'V', 'I', 'D'},
                              {'M', 'P', 'G', '2'},
                              {'M', 'J', 'P', 'G'} };

    size_t n_format = sizeof(format)/sizeof(format[0]),
           n_codec = sizeof(codec)/sizeof(codec[0]);

    for (size_t i = 0; i < n_format; ++i)
    for (size_t j = 0; j < n_codec; ++j)
    {
      CreateTestVideo(format[i], CV_FOURCC(codec[j][0], codec[j][1], codec[j][2], codec[j][3]), 125);

      stringstream s; s << CV_FOURCC(codec[j][0], codec[j][1], codec[j][2], codec[j][3]); //codec_bmp_tags[j].tag;

      const string file_path = ts->get_data_path()+"video/test_video_"+s.str()+"."+format[i];

      bool error = false; int failed = 0;

      cv::VideoCapture cap(file_path);

      if (!cap.isOpened())
      {
        ts->printf(ts->LOG, "\nFile: %s\n", file_path.c_str());
        ts->printf(ts->LOG, "\nVideo codec: %s\n", string(&codec[j][0], 4).c_str());
        ts->printf(ts->LOG, "\nError: cannot read video file.\n");
        ts->set_failed_test_info(ts->FAIL_INVALID_TEST_DATA);
        error = true;
      }

      cap.set(CV_CAP_PROP_POS_FRAMES, 0);

      int N = cap.get(CV_CAP_PROP_FRAME_COUNT);

      if (N != 125)
      {
        if (!error)
        {
            ts->printf(ts->LOG, "\nFile: %s\n", file_path.c_str());
            ts->printf(ts->LOG, "\nVideo codec: %s\n", string(&codec[j][0], 4).c_str());
            error = true;
        }
        ts->printf(ts->LOG, "\nError: returned frame count in clip is incorrect.\n");
        ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
      }

      vector <int> idx;

      RNG rng(N);
      idx.clear();
      for( int k = 0; k < N-1; k++ )
      idx.push_back(rng.uniform(0, N));
      idx.push_back(N-1);
      std::swap(idx.at(rng.uniform(0, N-1)), idx.at(N-1));

      for (int k = 0; k < N; ++k)
      {
        cap.set(CV_CAP_PROP_POS_FRAMES, (double)idx[k]);

        cv::Mat img; cap.retrieve(img);

        if (img.empty())
        {
            if (!error)
            {
                ts->printf(ts->LOG, "\nFile: %s\n", file_path.c_str());
                ts->printf(ts->LOG, "\nVideo codec: %s\n", string(&codec[j][0], 4).c_str());
                error = true;
            }
            ts->printf(ts->LOG, "\nError: cannot read a frame in position %d.\n", idx[k]);
            ts->set_failed_test_info(ts->FAIL_EXCEPTION);
        }

        const double thresh = 100;

        const size_t n = 32, w = img.rows/n;

        int index = 0, deg = n-1;

        for (size_t l = 0; l < n; ++l)
        {
            cv::Mat mat = img.rowRange(w*l, w*(l+1)-1);

            Scalar mat_mean = cv::mean(mat);

            if (mat_mean[0] > thresh) index += (int)std::pow(2.0, 1.0*deg);

            deg--;
        }

        if (index != idx[k])
        {
            if (!error)
            {
                ts->printf(ts->LOG, "\nFile: %s\n", file_path.c_str());
                ts->printf(ts->LOG, "\nVideo codec: %s\n\n", string(&codec[j][0], 4).c_str());
                error = true;
            }
            ts->printf(ts->LOG, "Required position: %d   Returned position: %d FAILED\n", idx[k], index);
            ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
            failed++;
        }
      }

      if (!error) ts->printf(ts->LOG, "\nFile: %s\n", file_path.c_str());

      ts->printf(ts->LOG, "\nSuccessfull iterations: %d(%d%%)   Failed iterations: %d(%d%%)\n", N-failed, (N-failed)*100/N, failed, failed*100/N);
      ts->printf(ts->LOG, "\n----------\n");
    }

#endif
#endif
}

TEST(Highgui_Positioning, regression) { CV_PositioningTest test; test.safe_run(); }

#endif
