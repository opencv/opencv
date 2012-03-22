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
    void ImageTest(const string& dir);
    void VideoTest (const string& dir, int fourcc);
    void SpecificImageTest (const string& dir);
    void SpecificVideoFileTest (const string& dir, const char codecchars[4]);
    void SpecificVideoCameraTest (const string& dir, const char codecchars[4]);

public:
    CV_HighGuiTest();
    ~CV_HighGuiTest();
    virtual void run(int) = 0;
};

class CV_ImageTest : public CV_HighGuiTest
{
public:
    CV_ImageTest();
    ~CV_ImageTest();
    void run(int);
};

class CV_SpecificImageTest : public CV_HighGuiTest
{
public:
    CV_SpecificImageTest();
    ~CV_SpecificImageTest();
    void run(int);
};

class CV_VideoTest : public CV_HighGuiTest
{
public:
    CV_VideoTest();
    ~CV_VideoTest();
    void run(int);
};

class CV_SpecificVideoFileTest : public CV_HighGuiTest
{
public:
    CV_SpecificVideoFileTest();
    ~CV_SpecificVideoFileTest();
    void run(int);
};

class CV_SpecificVideoCameraTest : public CV_HighGuiTest
{
public:
    CV_SpecificVideoCameraTest();
    ~CV_SpecificVideoCameraTest();
    void run(int);
};

CV_HighGuiTest::CV_HighGuiTest() {}
CV_HighGuiTest::~CV_HighGuiTest() {}

CV_ImageTest::CV_ImageTest() : CV_HighGuiTest() {}
CV_VideoTest::CV_VideoTest() : CV_HighGuiTest() {}
CV_SpecificImageTest::CV_SpecificImageTest() : CV_HighGuiTest() {}
CV_SpecificVideoFileTest::CV_SpecificVideoFileTest() : CV_HighGuiTest() {}
CV_SpecificVideoCameraTest::CV_SpecificVideoCameraTest() : CV_HighGuiTest() {}

CV_ImageTest::~CV_ImageTest() {}
CV_VideoTest::~CV_VideoTest() {}
CV_SpecificImageTest::~CV_SpecificImageTest() {}
CV_SpecificVideoFileTest::~CV_SpecificVideoFileTest() {}
CV_SpecificVideoCameraTest::~CV_SpecificVideoCameraTest() {}

double PSNR(const Mat& m1, const Mat& m2)
{		
    Mat tmp;
    absdiff( m1.reshape(1), m2.reshape(1), tmp);
    multiply(tmp, tmp, tmp);

    double MSE = 1.0/(tmp.cols * tmp.rows) * sum(tmp)[0];

    return 20 * log10(255.0 / sqrt(MSE));
}

void CV_HighGuiTest::ImageTest(const string& dir)
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

void CV_HighGuiTest::SpecificImageTest(const string& dir)
{
    const size_t IMAGE_COUNT = 10;

    for (size_t i = 0; i < IMAGE_COUNT; ++i)
    {
        stringstream s; s << i;
        string file_path = dir+"../python/images/QCIF_0"+s.str()+".bmp";
        Mat image = imread(file_path);

        if (image.empty())
        {
            ts->set_failed_test_info(ts->FAIL_MISSING_TEST_DATA);
            return;
        }

        cv::resize(image, image, cv::Size(968, 757), 0.0, 0.0, cv::INTER_CUBIC);

        stringstream s_digit; s_digit << i;

        string full_name = dir + "img_"+s_digit.str()+".bmp";
        ts->printf(ts->LOG, " full_name : %s\n", full_name.c_str());

        imwrite(full_name, image);

        Mat loaded = imread(full_name);
        if (loaded.empty())
        {
            ts->printf(ts->LOG, "Reading failed at fmt=bmp\n");
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

        const double thresDbell = 20;
        double psnr = PSNR(loaded, image);
        if (psnr < thresDbell)
        {
            ts->printf(ts->LOG, "Reading image from file: too big difference (=%g) with fmt=bmp\n", psnr);
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
        imencode(".bmp", image, buf);

        if (buf != from_file)
        {
            ts->printf(ts->LOG, "Encoding failed with fmt=bmp\n");
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

        Mat buf_loaded = imdecode(Mat(buf), 1);

        if (buf_loaded.empty())
        {
            ts->printf(ts->LOG, "Decoding failed with fmt=bmp\n");
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }

        psnr = PSNR(buf_loaded, image);

        if (psnr < thresDbell)
        {
            ts->printf(ts->LOG, "Decoding image from memory: too small PSNR (=%gdb) with fmt=bmp\n", psnr);
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
            continue;
        }
    }

    ts->printf(ts->LOG, "end test function : SpecificImageTest \n");
    ts->set_failed_test_info(ts->OK);
}

void CV_HighGuiTest::SpecificVideoFileTest(const string& dir, const char codecchars[4])
{
    const string ext[] = {"avi", "mov", "mp4", "mpg", "wmv"};

    const size_t n = sizeof(ext)/sizeof(ext[0]);

    for (size_t j = 0; j < n; ++j) 
		if ((ext[j]!="mp4")||(string(&codecchars[0], 4)!="IYUV"))
	#if defined WIN32 || defined _WIN32
		if (((ext[j]!="mov")||(string(&codecchars[0], 4)=="XVID"))&&(ext[j]!="mp4"))
	#endif
    {
        const string video_file = dir + "video_" + string(&codecchars[0], 4) + "." + ext[j];

        VideoWriter writer = cv::VideoWriter(video_file, CV_FOURCC(codecchars[0], codecchars[1], codecchars[2], codecchars[3]), 25, cv::Size(968, 757), true);

		if (!writer.isOpened())
		{
			ts->printf(ts->LOG, "Creating a video in %s...\n", video_file.c_str());
			ts->printf(ts->LOG, "Cannot create VideoWriter object with codec %s.\n", string(&codecchars[0], 4).c_str());
			ts->set_failed_test_info(ts->FAIL_MISMATCH);
			continue;
		}

        const size_t IMAGE_COUNT = 30;

        for(size_t i = 0; i < IMAGE_COUNT; ++i)
        {
            stringstream s_digit;
            if (i < 10) {s_digit << "0"; s_digit << i;}
            else s_digit <<  i;

            const string file_path = dir+"../python/images/QCIF_"+s_digit.str()+".bmp";

            cv::Mat img = imread(file_path, CV_LOAD_IMAGE_COLOR);

            if (img.empty())
            {
                ts->printf(ts->LOG, "Creating a video in %s...\n", video_file.c_str());
                ts->printf(ts->LOG, "Error: cannot read frame from %s.\n", (ts->get_data_path()+"../python/images/QCIF_"+s_digit.str()+".bmp").c_str());
                ts->printf(ts->LOG, "Continue creating the video file...\n");
                ts->set_failed_test_info(ts->FAIL_INVALID_TEST_DATA);
                continue;
            }

            cv::resize(img, img, Size(968, 757), 0.0, 0.0, cv::INTER_CUBIC);

            for (int k = 0; k < img.rows; ++k)
                for (int l = 0; l < img.cols; ++l)
                    if (img.at<Vec3b>(k, l) == Vec3b::all(0))
                        img.at<Vec3b>(k, l) = Vec3b(0, 255, 0);
            else img.at<Vec3b>(k, l) = Vec3b(0, 0, 255);

            imwrite(dir+"QCIF_"+s_digit.str()+".bmp", img);

            writer << img;
        }

        writer.~VideoWriter();

        cv::VideoCapture cap(video_file);

        size_t FRAME_COUNT = (size_t)cap.get(CV_CAP_PROP_FRAME_COUNT);

        if (FRAME_COUNT != IMAGE_COUNT)
        {
            ts->printf(ts->LOG, "\nFrame count checking for video_%s.%s...\n", string(&codecchars[0], 4).c_str(), ext[j].c_str());
            ts->printf(ts->LOG, "Video codec: %s\n", string(&codecchars[0], 4).c_str());
            ts->printf(ts->LOG, "Required frame count: %d; Returned frame count: %d\n", IMAGE_COUNT, FRAME_COUNT);
            ts->printf(ts->LOG, "Error: Incorrect frame count in the video.\n");
			ts->printf(ts->LOG, "Continue checking...\n");
            ts->set_failed_test_info(ts->FAIL_BAD_ACCURACY);
        }

        cap.set(CV_CAP_PROP_POS_FRAMES, -1);

		for (int i = -1; i < (int)std::min<size_t>(FRAME_COUNT, IMAGE_COUNT)-1; i++)
        {
            cv::Mat frame; cap >> frame;
            if (frame.empty())
            {
                ts->printf(ts->LOG, "\nVideo file directory: %s\n", dir.c_str());
                ts->printf(ts->LOG, "File name: video_%s.%s\n", string(&codecchars[0], 4).c_str(), ext[i].c_str());
                ts->printf(ts->LOG, "Video codec: %s\n", string(&codecchars[0], 4).c_str());
				ts->printf(ts->LOG, "Error: cannot read the next frame with index %d.\n", i+1);
                ts->set_failed_test_info(ts->FAIL_MISSING_TEST_DATA);
                break;
            }

            stringstream s_digit;
            if (i+1 < 10) {s_digit << "0"; s_digit << i+1;}
            else s_digit << i+1;

            cv::Mat img = imread(dir+"QCIF_"+s_digit.str()+".bmp", CV_LOAD_IMAGE_COLOR);

            if (img.empty())
            {
				ts->printf(ts->LOG, "\nError: cannot read an image from %s.\n", (dir+"QCIF_"+s_digit.str()+".bmp").c_str());
                ts->set_failed_test_info(ts->FAIL_MISMATCH);
                continue;
            }

            const double thresDbell = 20;

            double psnr = PSNR(img, frame);

            if (psnr > thresDbell)
            {
                ts->printf(ts->LOG, "\nReading frame from the file video_%s.%s...\n", string(&codecchars[0], 4).c_str(), ext[j].c_str());
				ts->printf(ts->LOG, "Frame index: %d\n", i+1);
                ts->printf(ts->LOG, "Difference between saved and original images: %g\n", psnr);
                ts->printf(ts->LOG, "Maximum allowed difference: %g\n", thresDbell);
                ts->printf(ts->LOG, "Error: too big difference between saved and original images.\n");
                continue;
            }

        }

        cap.~VideoCapture();
    }
}

void CV_HighGuiTest::SpecificVideoCameraTest(const string& dir, const char codecchars[4])
{
    const string ext[] = {"avi", "mov", "mp4", "mpg", "wmv"};

    const size_t n = sizeof(ext)/sizeof(ext[0]);

    const int IMAGE_COUNT = 125;

    cv::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        ts->printf(ts->LOG, "\nError: cannot start working with device.\n");
		ts->set_failed_test_info(ts->OK);
        return;
    }

    for (size_t i = 0; i < n; ++i) 
		if ((ext[i]!="mp4")||(string(&codecchars[0], 4)!="IYUV"))
	#if defined WIN32 || defined _WIN32
		if (((ext[i]!="mov")||(string(&codecchars[0], 4)=="XVID"))&&(ext[i]!="mp4"))
	#endif
    {
        Mat frame; int framecount = 0;
        cv::VideoWriter writer;

        std::vector <cv::Mat> tmp_img(IMAGE_COUNT);

        writer.open(dir+"video_"+string(&codecchars[0], 4)+"."+ext[i], CV_FOURCC(codecchars[0], codecchars[1], codecchars[2], codecchars[3]), 25, Size(968, 757), true);

        if (!writer.isOpened())
        {
            ts->printf(ts->LOG, "\nVideo file directory: %s\n", dir.c_str());
            ts->printf(ts->LOG, "Video codec: %s\n", std::string(&codecchars[0], 4).c_str());
            ts->printf(ts->LOG, "Error: cannot create VideoWriter object for video_%s.%s.\n", string(&codecchars[0]).c_str(), ext[i].c_str());
            ts->set_failed_test_info(ts->FAIL_EXCEPTION);
            continue;
        }

        for (;;)
        {
            cap >> frame;

            if (frame.empty())
            {
                ts->printf(ts->LOG, "\nVideo file directory: %s\n", dir.c_str());
                ts->printf(ts->LOG, "File name: video_%s.%s\n", string(&codecchars[0], 4).c_str(), ext[i].c_str());
                ts->printf(ts->LOG, "Video codec: %s\n", string(&codecchars[0], 4).c_str());
				ts->printf(ts->LOG, "Error: cannot read next frame with index %d from the device.\n", framecount);
                break;
            }

            cv::resize(frame, frame, Size(968, 757), 0, 0, INTER_CUBIC);
            writer << frame; tmp_img[framecount] = frame;

            framecount++;
            if (framecount == IMAGE_COUNT) break;
        }

        frame.~Mat();
        writer.~VideoWriter();

        cv::VideoCapture vcap(dir+"video_"+string(&codecchars[0], 4)+"."+ext[i]);

        if (!vcap.isOpened())
        {
            ts->printf(ts->LOG, "\nVideo file directory: %s\n", dir.c_str());
            ts->printf(ts->LOG, "File name: video_%s.%s\n",  string(&codecchars[0], 4).c_str(), ext[i].c_str());
            ts->printf(ts->LOG, "Video codec: %s\n", string(&codecchars[0], 4).c_str());
            ts->printf(ts->LOG, "Error: cannot open video file.\n");
            continue;
        }

        int FRAME_COUNT = (int)vcap.get(CV_CAP_PROP_FRAME_COUNT);

        if (FRAME_COUNT != IMAGE_COUNT)
        {
            ts->printf(ts->LOG, "\nChecking frame count...\n");
            ts->printf(ts->LOG, "Video file directory: %s\n", dir.c_str());
            ts->printf(ts->LOG, "File name: video_%s.%s\n", string(&codecchars[0], 4).c_str(), ext[i].c_str());
            ts->printf(ts->LOG, "Video codec: %s\n", string(&codecchars[0], 4).c_str());
            ts->printf(ts->LOG, "Required frame count: %d  Returned frame count: %d\n", IMAGE_COUNT, FRAME_COUNT);
            ts->printf(ts->LOG, "Error: required and returned frame count are not matched.\n");
			ts->printf(ts->LOG, "Continue checking...\n");
            ts->set_failed_test_info(ts->FAIL_INVALID_OUTPUT);
        }

        cv::Mat img; framecount = 0;
        vcap.set(CV_CAP_PROP_POS_FRAMES, 0);

        for ( ; framecount < std::min<int>(FRAME_COUNT, IMAGE_COUNT); framecount++ )
        {
            vcap >> img;

            if (img.empty())
            {
                ts->printf(ts->LOG, "\nVideo file directory: %s\n", dir.c_str());
                ts->printf(ts->LOG, "File name: video_%s.%s\n", string(&codecchars[0], 4).c_str(), ext[i].c_str());
                ts->printf(ts->LOG, "Video codec: %s\n", string(&codecchars[0], 4).c_str());
                ts->printf(ts->LOG, "Error: cannot read frame with index %d from the video.\n", framecount);
                break;
            }

            const double thresDbell = 20;
            double psnr = PSNR(img, tmp_img[framecount]);

            if (psnr > thresDbell)
            {
                ts->printf(ts->LOG, "\nReading frame from the file video_%s.%s...\n", string(&codecchars[0], 4).c_str(), ext[i].c_str());
				ts->printf(ts->LOG, "Frame index: %d\n", framecount);
                ts->printf(ts->LOG, "Difference between saved and original images: %g\n", psnr);
                ts->printf(ts->LOG, "Maximum allowed difference: %g\n", thresDbell);
                ts->printf(ts->LOG, "Error: too big difference between saved and original images.\n");
                continue;
            }
        }

        img.~Mat();
        vcap.~VideoCapture();
    }

    cap.~VideoCapture();
}

void CV_ImageTest::run(int)
{
    ImageTest(ts->get_data_path());
}

void CV_SpecificImageTest::run(int)
{
    SpecificImageTest(ts->get_data_path());
}

void CV_VideoTest::run(int)
{
    const char codecs[][4] = { {'I', 'Y', 'U', 'V'},
                               {'X', 'V', 'I', 'D'},
                               {'M', 'P', 'G', '2'},
                               {'M', 'J', 'P', 'G'} };

    printf("%s", ts->get_data_path().c_str());

    int count = sizeof(codecs)/(4*sizeof(char));

    for (int i = 0; i < count; ++i)
    {
        VideoTest(ts->get_data_path(), CV_FOURCC(codecs[i][0], codecs[i][1], codecs[i][2], codecs[i][3]));
    }
}

void CV_SpecificVideoFileTest::run(int)
{
    const char codecs[][4] = { {'M', 'P', 'G', '2'},
                               {'X', 'V', 'I', 'D'},
                               {'M', 'J', 'P', 'G'},
                               {'I', 'Y', 'U', 'V'} };

    int count = sizeof(codecs)/(4*sizeof(char));

    for (int i = 0; i < count; ++i)
    {
        SpecificVideoFileTest(ts->get_data_path(), codecs[i]);
    }
}

void CV_SpecificVideoCameraTest::run(int)
{
    const char codecs[][4] = { {'M', 'P', 'G', '2'},
                               {'X', 'V', 'I', 'D'},
                               {'M', 'J', 'P', 'G'},
                               {'I', 'Y', 'U', 'V'} };

    int count = sizeof(codecs)/(4*sizeof(char));

    for (int i = 0; i < count; ++i)
    {
        SpecificVideoCameraTest(ts->get_data_path(), codecs[i]);
    }
}

TEST(Highgui_Image, regression) { CV_ImageTest test; test.safe_run(); }
TEST(Highgui_Video, regression) { CV_VideoTest test; test.safe_run(); }
TEST(Highgui_SpecificImage, regression) { CV_SpecificImageTest test; test.safe_run(); }
TEST(Highgui_SpecificVideoFile, regression) { CV_SpecificVideoFileTest test; test.safe_run(); }
TEST(Highgui_SpecificVideoCamera, regression) { CV_SpecificVideoCameraTest test; test.safe_run(); }
