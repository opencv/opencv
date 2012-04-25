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

string cvtest::fourccToString(int fourcc)
{
    return format("%c%c%c%c", fourcc & 255, (fourcc >> 8) & 255, (fourcc >> 16) & 255, (fourcc >> 24) & 255);
}

struct VideoFmt
{
    VideoFmt() { fourcc = -1; }
    VideoFmt(const string& _ext, int _fourcc) : ext(_ext), fourcc(_fourcc) {}
    bool empty() const { return ext.empty(); }
    
    string ext;
    int fourcc;
};

static const VideoFmt specific_fmt_list[] =
{
    VideoFmt("avi", CV_FOURCC('m', 'p', 'e', 'g')),
    VideoFmt("avi", CV_FOURCC('M', 'J', 'P', 'G')),
    VideoFmt("avi", CV_FOURCC('I', 'Y', 'U', 'V')),
    VideoFmt("mkv", CV_FOURCC('X', 'V', 'I', 'D')),
    VideoFmt("mov", CV_FOURCC('m', 'p', '4', 'v')),
    VideoFmt()
};

class CV_HighGuiTest : public cvtest::BaseTest
{
protected:
    void ImageTest(const string& dir);
    void VideoTest (const string& dir, const VideoFmt& fmt);
    void SpecificImageTest (const string& dir);
    void SpecificVideoTest (const string& dir, const VideoFmt& fmt);

    CV_HighGuiTest() {}
    ~CV_HighGuiTest() {}
    virtual void run(int) = 0;
};

class CV_ImageTest : public CV_HighGuiTest
{
public:
    CV_ImageTest() {}
    ~CV_ImageTest() {}
    void run(int);
};

class CV_SpecificImageTest : public CV_HighGuiTest
{
public:
    CV_SpecificImageTest() {}
    ~CV_SpecificImageTest() {}
    void run(int);
};

class CV_VideoTest : public CV_HighGuiTest
{
public:
    CV_VideoTest() {}
    ~CV_VideoTest() {}
    void run(int);
};

class CV_SpecificVideoTest : public CV_HighGuiTest
{
public:
    CV_SpecificVideoTest() {}
    ~CV_SpecificVideoTest() {}
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

    const string exts[] = {
#ifdef HAVE_PNG
        "png",
#endif
#ifdef HAVE_TIFF
        "tiff",
#endif
#ifdef HAVE_JPEG
        "jpg",
#endif
#ifdef HAVE_JASPER
        "jp2",
#endif
#ifdef HAVE_OPENEXR
        "exr",
#endif
        "bmp",
        "ppm",
        "ras"
        };
    const size_t ext_num = sizeof(exts)/sizeof(exts[0]);

    for(size_t i = 0; i < ext_num; ++i)
    {
        string ext = exts[i];
        string full_name = "img." + ext;
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


void CV_HighGuiTest::VideoTest(const string& dir, const VideoFmt& fmt)
{
    string src_file = dir + "../cv/shared/video_for_test.avi";
    string tmp_name = format("video.%s", fmt.ext.c_str());

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
            writer = cvCreateVideoWriter(tmp_name.c_str(), fmt.fourcc, 24, cvGetSize(img));
            if (writer == 0)
            {
                ts->printf(ts->LOG, "can't create writer (with fourcc : %d)\n",
                           cvtest::fourccToString(fmt.fourcc).c_str());
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

        resize(image, image, Size(968, 757), 0.0, 0.0, INTER_CUBIC);

        stringstream s_digit; s_digit << i;

        string full_name = "img_"+s_digit.str()+".bmp";
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


void CV_HighGuiTest::SpecificVideoTest(const string& dir, const VideoFmt& fmt)
{
    string ext = fmt.ext;
    int fourcc = fmt.fourcc;
        
    string fourcc_str = cvtest::fourccToString(fourcc);
    const string video_file = "video_" + fourcc_str + "." + ext;

    Size frame_size(968 & -2, 757 & -2);
    VideoWriter writer(video_file, fourcc, 25, frame_size, true);

    if (!writer.isOpened())
    {
        // call it repeatedly for easier debugging
        VideoWriter writer(video_file, fourcc, 25, frame_size, true);
        ts->printf(ts->LOG, "Creating a video in %s...\n", video_file.c_str());
        ts->printf(ts->LOG, "Cannot create VideoWriter object with codec %s.\n", fourcc_str.c_str());
        ts->set_failed_test_info(ts->FAIL_MISMATCH);
        return;
    }

    const size_t IMAGE_COUNT = 30;
    vector<Mat> images;
    
    for( size_t i = 0; i < IMAGE_COUNT; ++i )
    {
        string file_path = format("%s../python/images/QCIF_%02d.bmp", dir.c_str(), i);
        Mat img = imread(file_path, CV_LOAD_IMAGE_COLOR);

        if (img.empty())
        {
            ts->printf(ts->LOG, "Creating a video in %s...\n", video_file.c_str());
            ts->printf(ts->LOG, "Error: cannot read frame from %s.\n", file_path.c_str());
            ts->printf(ts->LOG, "Continue creating the video file...\n");
            ts->set_failed_test_info(ts->FAIL_INVALID_TEST_DATA);
            break;
        }

        for (int k = 0; k < img.rows; ++k)
            for (int l = 0; l < img.cols; ++l)
                if (img.at<Vec3b>(k, l) == Vec3b::all(0))
                    img.at<Vec3b>(k, l) = Vec3b(0, 255, 0);
                else img.at<Vec3b>(k, l) = Vec3b(0, 0, 255);
        
        resize(img, img, frame_size, 0.0, 0.0, INTER_CUBIC);

        images.push_back(img);
        writer << img;
    }

    writer.release();
    VideoCapture cap(video_file);

    size_t FRAME_COUNT = (size_t)cap.get(CV_CAP_PROP_FRAME_COUNT);

    if (FRAME_COUNT != IMAGE_COUNT )
    {
        ts->printf(ts->LOG, "\nFrame count checking for video_%s.%s...\n", fourcc_str.c_str(), ext.c_str());
        ts->printf(ts->LOG, "Video codec: %s\n", fourcc_str.c_str());
        ts->printf(ts->LOG, "Required frame count: %d; Returned frame count: %d\n", IMAGE_COUNT, FRAME_COUNT);
        ts->printf(ts->LOG, "Error: Incorrect frame count in the video.\n");
        ts->printf(ts->LOG, "Continue checking...\n");
        ts->set_failed_test_info(ts->FAIL_BAD_ACCURACY);
        return;
    }

    for (int i = 0; i < FRAME_COUNT; i++)
    {
        Mat frame; cap >> frame;
        if (frame.empty())
        {
            ts->printf(ts->LOG, "\nVideo file directory: %s\n", ".");
            ts->printf(ts->LOG, "File name: video_%s.%s\n", fourcc_str.c_str(), ext.c_str());
            ts->printf(ts->LOG, "Video codec: %s\n", fourcc_str.c_str());
            ts->printf(ts->LOG, "Error: cannot read the next frame with index %d.\n", i+1);
            ts->set_failed_test_info(ts->FAIL_MISSING_TEST_DATA);
            break;
        }

        Mat img = images[i];

        const double thresDbell = 40;
        double psnr = PSNR(img, frame);

        if (psnr > thresDbell)
        {
            ts->printf(ts->LOG, "\nReading frame from the file video_%s.%s...\n", fourcc_str.c_str(), ext.c_str());
            ts->printf(ts->LOG, "Frame index: %d\n", i+1);
            ts->printf(ts->LOG, "Difference between saved and original images: %g\n", psnr);
            ts->printf(ts->LOG, "Maximum allowed difference: %g\n", thresDbell);
            ts->printf(ts->LOG, "Error: too big difference between saved and original images.\n");
            break;
        }
    }
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
    for (int i = 0; !specific_fmt_list[i].empty(); ++i)
    {
        VideoTest(ts->get_data_path(), specific_fmt_list[i]);
    }
}

void CV_SpecificVideoTest::run(int)
{
    for (int i = 0; !specific_fmt_list[i].empty(); ++i)
    {
        SpecificVideoTest(ts->get_data_path(), specific_fmt_list[i]);
    }
}

#ifdef HAVE_JPEG
TEST(Highgui_Image, regression) { CV_ImageTest test; test.safe_run(); }
#endif

#if BUILD_WITH_VIDEO_INPUT_SUPPORT && BUILD_WITH_VIDEO_OUTPUT_SUPPORT
TEST(Highgui_Video, regression) { CV_VideoTest test; test.safe_run(); }
TEST(Highgui_SpecificVideoFile, regression) { CV_SpecificVideoTest test; test.safe_run(); }
#endif

TEST(Highgui_SpecificImage, regression) { CV_SpecificImageTest test; test.safe_run(); }
