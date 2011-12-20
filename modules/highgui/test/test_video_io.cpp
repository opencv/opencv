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

using namespace cv;
using namespace std;

class CV_HighGuiTest : public cvtest::BaseTest
{
protected:
    void ImagesTest(const string& dir);
    void VideoTest (const string& dir, int fourcc);

public:
    void run(int);
};

double PSNR(const Mat& m1, const Mat& m2)
{		
    Mat tmp;
    absdiff( m1.reshape(1), m2.reshape(1), tmp);
    multiply(tmp, tmp, tmp);

    double MSE = 1.0/(tmp.cols * tmp.rows) * sum(tmp)[0];

    return 20 * log10(255.0 / sqrt(MSE));
}

void CV_HighGuiTest::ImagesTest(const string& dir)
{
    string _name = dir + string("../cv/shared/baboon.jpg");
    ts->printf(ts->LOG, "reading image : %s\n", _name.c_str());

    Mat image = imread(_name);
    image.convertTo(image, CV_8UC3);

    if (image.empty())
    {
        ts->set_failed_test_info(ts->FAIL_MISSING_TEST_DATA);
        return;
    }

    const string exts[] = {"png", "bmp", "tiff", "jpg", "jp2", "ppm", "ras" };
    const size_t ext_num = sizeof(exts)/sizeof(exts[0]);

    for(size_t i = 0; i < ext_num; ++i)
    {
        string ext = exts[i];
        string full_name = dir + "img." + ext;
        ts->printf(ts->LOG, " full_name : %s\n", full_name.c_str());

        imwrite(full_name, image);

        Mat loaded = imread(full_name);
        if (loaded.empty())
        {
            ts->printf(ts->LOG, "Reading failed at fmt=%s\n", ext.c_str());
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

        const double thresDbell = 20;
        double psnr = PSNR(loaded, image);
        if (psnr < thresDbell)
        {
            ts->printf(ts->LOG, "Reading image from file: too big difference (=%g) with fmt=%s\n", psnr, ext.c_str());
            ts->set_failed_test_info(ts->FAIL_BAD_ACCURACY);
            continue;
        }

        vector<uchar> from_file;

        FILE *f = fopen(full_name.c_str(), "rb");
        fseek(f, 0, SEEK_END);
        long len = ftell(f);
        from_file.resize((size_t)len);
        fseek(f, 0, SEEK_SET);
        from_file.resize(fread(&from_file[0], 1, from_file.size(), f));
        fclose(f);
	
        vector<uchar> buf;
        imencode("." + exts[i], image, buf);

        if (buf != from_file)
        {
            ts->printf(ts->LOG, "Encoding failed with fmt=%s\n", ext.c_str());
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

        Mat buf_loaded = imdecode(Mat(buf), 1);

        if (buf_loaded.empty())
        {
            ts->printf(ts->LOG, "Decoding failed with fmt=%s\n", ext.c_str());
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

        psnr = PSNR(buf_loaded, image);

        if (psnr < thresDbell)
        {
            ts->printf(ts->LOG, "Decoding image from memory: too small PSNR (=%gdb) with fmt=%s\n", psnr, ext.c_str());
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

    }

    ts->printf(ts->LOG, "end test function : ImagesTest \n");
    ts->set_failed_test_info(ts->OK);
}

void CV_HighGuiTest::VideoTest(const string& dir, int fourcc)
{	
    string src_file = dir + "../cv/shared/video_for_test.avi";
    string tmp_name = dir + "video.avi";

    ts->printf(ts->LOG, "reading video : %s\n", src_file.c_str());

    CvCapture* cap = cvCaptureFromFile(src_file.c_str());

    if (!cap)
    {
        ts->set_failed_test_info(ts->FAIL_MISMATCH);
        return;
    }

    CvVideoWriter* writer = 0;

    int counter = 0;
    for(;;)
    {
        IplImage * img = cvQueryFrame( cap );

        if (!img)
            break;

        if (writer == 0)
        {
            writer = cvCreateVideoWriter(tmp_name.c_str(), fourcc, 24, cvGetSize(img));
            if (writer == 0)
            {
                ts->printf(ts->LOG, "can't create writer (with fourcc : %d)\n", fourcc);
                cvReleaseCapture( &cap );
                ts->set_failed_test_info(ts->FAIL_MISMATCH);
                return;
            }
        }

        cvWriteFrame(writer, img);
    }

    cvReleaseVideoWriter( &writer );
    cvReleaseCapture( &cap );

    cap = cvCaptureFromFile(src_file.c_str());

    CvCapture *saved = cvCaptureFromFile(tmp_name.c_str());
    if (!saved)
    {
        ts->set_failed_test_info(ts->FAIL_MISMATCH);
        return;
    }

    const double thresDbell = 20;

    counter = 0;
    for(;;)
    {
        IplImage* ipl  = cvQueryFrame( cap );
        IplImage* ipl1 = cvQueryFrame( saved );

        if (!ipl || !ipl1)
            break;

        Mat img(ipl);
        Mat img1(ipl1);

        if (PSNR(img1, img) < thresDbell)
        {
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            break;
        }
    }

    cvReleaseCapture( &cap );
    cvReleaseCapture( &saved );

    ts->printf(ts->LOG, "end test function : ImagesVideo \n");
}


void CV_HighGuiTest::run( int /*start_from */)
{
    ImagesTest(ts->get_data_path());

#if defined WIN32 || (defined __linux__ && !defined ANDROID)
#if !defined HAVE_GSTREAMER || defined HAVE_GSTREAMER_APP  

    VideoTest(ts->get_data_path(), CV_FOURCC_DEFAULT);

    VideoTest(ts->get_data_path(), CV_FOURCC('X', 'V', 'I', 'D'));

    VideoTest(ts->get_data_path(), CV_FOURCC('M', 'P', 'G', '2'));

    VideoTest(ts->get_data_path(), CV_FOURCC('M', 'J', 'P', 'G'));

#endif
#endif
}

TEST(Highgui_HighGui, regression) { CV_HighGuiTest  test; test.safe_run(); }
