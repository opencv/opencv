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
#include "opencv2/highgui/highgui.hpp"

#ifdef HAVE_FFMPEG

#include "ffmpeg_codecs.hpp"

using namespace cv;
using namespace std;

class CV_FFmpegWriteBigVideoTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        const int img_r = 4096;
        const int img_c = 4096;
        const double fps0 = 15;
        const double time_sec = 1;
        
        const size_t n = sizeof(codec_bmp_tags)/sizeof(codec_bmp_tags[0]);

        bool created = false;

        for (size_t j = 0; j < n; ++j)
        {
        stringstream s; s << codec_bmp_tags[j].tag;
        int tag = codec_bmp_tags[j].tag;
        
        if( tag != MKTAG('H', '2', '6', '3') &&
            tag != MKTAG('H', '2', '6', '1') &&
            //tag != MKTAG('D', 'I', 'V', 'X') &&
            tag != MKTAG('D', 'X', '5', '0') &&
            tag != MKTAG('X', 'V', 'I', 'D') &&
            tag != MKTAG('m', 'p', '4', 'v') &&
            //tag != MKTAG('D', 'I', 'V', '3') &&
            //tag != MKTAG('W', 'M', 'V', '1') &&
            //tag != MKTAG('W', 'M', 'V', '2') &&
            tag != MKTAG('M', 'P', 'E', 'G') &&
            tag != MKTAG('M', 'J', 'P', 'G') &&
            //tag != MKTAG('j', 'p', 'e', 'g') &&
            tag != 0 &&
            tag != MKTAG('I', '4', '2', '0') &&
            //tag != MKTAG('Y', 'U', 'Y', '2') &&
            tag != MKTAG('F', 'L', 'V', '1') )
            continue;

        const string filename = "output_"+s.str()+".avi";

        try
        {
            double fps = fps0;
            Size frame_s = Size(img_c, img_r);
            
            if( tag == CV_FOURCC('H', '2', '6', '1') )
                frame_s = Size(352, 288);
            else if( tag == CV_FOURCC('H', '2', '6', '3') )
                frame_s = Size(704, 576);
            /*else if( tag == CV_FOURCC('M', 'J', 'P', 'G') ||
                     tag == CV_FOURCC('j', 'p', 'e', 'g') )
                frame_s = Size(1920, 1080);*/
            
            if( tag == CV_FOURCC('M', 'P', 'E', 'G') )
                fps = 25;
            
            VideoWriter writer(filename, tag, fps, frame_s);

            if (writer.isOpened() == false)
            {
                ts->printf(ts->LOG, "\n\nFile name: %s\n", filename.c_str());
                ts->printf(ts->LOG, "Codec id: %d   Codec tag: %c%c%c%c\n", j,
                           tag & 255, (tag >> 8) & 255, (tag >> 16) & 255, (tag >> 24) & 255);
                ts->printf(ts->LOG, "Error: cannot create video file.");
                ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
            }
            else
            {
                Mat img(frame_s, CV_8UC3, Scalar::all(0));
                const int coeff = cvRound(cv::min(frame_s.width, frame_s.height)/(fps0 * time_sec));

                for (int i = 0 ; i < static_cast<int>(fps * time_sec); i++ )
                {
                    //circle(img, Point2i(img_c / 2, img_r / 2), cv::min(img_r, img_c) / 2 * (i + 1), Scalar(255, 0, 0, 0), 2);
                    rectangle(img, Point2i(coeff * i, coeff * i), Point2i(coeff * (i + 1), coeff * (i + 1)),
                              Scalar::all(255 * (1.0 - static_cast<double>(i) / (fps * time_sec * 2) )), -1);
                    writer << img;
                }

                if (!created) created = true;
                else remove(filename.c_str());
            }
        }
        catch(...)
        {
            ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
        }
        ts->set_failed_test_info(cvtest::TS::OK);

        }
    }
};

TEST(Highgui_Video, ffmpeg_writebig) { CV_FFmpegWriteBigVideoTest test; test.safe_run(); }

class CV_FFmpegReadImageTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            string filename = ts->get_data_path() + "../cv/features2d/tsukuba.png";
            VideoCapture cap(filename);
            Mat img0 = imread(filename, 1);
            Mat img, img_next;
            cap >> img;
            cap >> img_next;
            
            CV_Assert( !img0.empty() && !img.empty() && img_next.empty() );
            
            double diff = norm(img0, img, CV_C);
            CV_Assert( diff == 0 );
        }
        catch(...)
        {
            ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Highgui_Video, ffmpeg_image) { CV_FFmpegReadImageTest test; test.safe_run(); }

#endif
