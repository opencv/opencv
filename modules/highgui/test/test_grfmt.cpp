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


class CV_GrfmtWriteBigImageTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            ts->printf(cvtest::TS::LOG, "start  reading big image\n");
            Mat img = imread(string(ts->get_data_path()) + "readwrite/read.png");
            ts->printf(cvtest::TS::LOG, "finish reading big image\n");
            if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            ts->printf(cvtest::TS::LOG, "start  writing big image\n");
            imwrite(cv::tempfile(".png"), img);
            ts->printf(cvtest::TS::LOG, "finish writing big image\n");
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

string ext_from_int(int ext)
{
#ifdef HAVE_PNG
    if (ext == 0) return ".png";
#endif
    if (ext == 1) return ".bmp";
    if (ext == 2) return ".pgm";
#ifdef HAVE_TIFF
    if (ext == 3) return ".tiff";
#endif
    return "";
}

class CV_GrfmtWriteSequenceImageTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            const int img_r = 640;
            const int img_c = 480;

            for (int k = 1; k <= 5; ++k)
            {
                for (int ext = 0; ext < 4; ++ext) // 0 - png, 1 - bmp, 2 - pgm, 3 - tiff
                {
                    if(ext_from_int(ext).empty())
                        continue;
                    for (int num_channels = 1; num_channels <= 3; num_channels+=2)
                    {
                        ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_8U, num_channels, ext_from_int(ext).c_str());
                        Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_8U, num_channels), Scalar::all(0));
                        circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));

                        string img_path = cv::tempfile(ext_from_int(ext).c_str());
                        ts->printf(ts->LOG, "writing      image : %s\n", img_path.c_str());
                        imwrite(img_path, img);

                        ts->printf(ts->LOG, "reading test image : %s\n", img_path.c_str());
                        Mat img_test = imread(img_path, CV_LOAD_IMAGE_UNCHANGED);

                        if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

                        CV_Assert(img.size() == img_test.size());
                        CV_Assert(img.type() == img_test.type());

                        double n = norm(img, img_test);
                        if ( n > 1.0)
                        {
                            ts->printf(ts->LOG, "norm = %f \n", n);
                            ts->set_failed_test_info(ts->FAIL_MISMATCH);
                        }
                    }
                }

#ifdef HAVE_JPEG
                for (int num_channels = 1; num_channels <= 3; num_channels+=2)
                {
                    // jpeg
                    ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_8U, num_channels, ".jpg");
                    Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_8U, num_channels), Scalar::all(0));
                    circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));

                    string filename = cv::tempfile(".jpg");
                    imwrite(filename, img);
                    img = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

                    filename = string(ts->get_data_path() + "readwrite/test_" + char(k + 48) + "_c" + char(num_channels + 48) + ".jpg");
                    ts->printf(ts->LOG, "reading test image : %s\n", filename.c_str());
                    Mat img_test = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

                    if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

                    CV_Assert(img.size() == img_test.size());
                    CV_Assert(img.type() == img_test.type());

                    double n = norm(img, img_test);
                    if ( n > 1.0)
                    {
                        ts->printf(ts->LOG, "norm = %f \n", n);
                        ts->set_failed_test_info(ts->FAIL_MISMATCH);
                    }
                }
#endif

#ifdef HAVE_TIFF
                for (int num_channels = 1; num_channels <= 3; num_channels+=2)
                {
                    // tiff
                    ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_16U, num_channels, ".tiff");
                    Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_16U, num_channels), Scalar::all(0));
                    circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), cv::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));

                    string filename = cv::tempfile(".tiff");
                    imwrite(filename, img);
                    ts->printf(ts->LOG, "reading test image : %s\n", filename.c_str());
                    Mat img_test = imread(filename, CV_LOAD_IMAGE_UNCHANGED);

                    if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

                    CV_Assert(img.size() == img_test.size());

                    ts->printf(ts->LOG, "img      : %d ; %d \n", img.channels(), img.depth());
                    ts->printf(ts->LOG, "img_test : %d ; %d \n", img_test.channels(), img_test.depth());

                    CV_Assert(img.type() == img_test.type());


                    double n = norm(img, img_test);
                    if ( n > 1.0)
                    {
                        ts->printf(ts->LOG, "norm = %f \n", n);
                        ts->set_failed_test_info(ts->FAIL_MISMATCH);
                    }
                }
#endif
            }
        }
        catch(const cv::Exception & e)
        {
            ts->printf(ts->LOG, "Exception: %s\n" , e.what());
            ts->set_failed_test_info(ts->FAIL_MISMATCH);
        }
    }
};

class CV_GrfmtReadBMPRLE8Test : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            Mat rle = imread(string(ts->get_data_path()) + "readwrite/rle8.bmp");
            Mat bmp = imread(string(ts->get_data_path()) + "readwrite/ordinary.bmp");
            if (norm(rle-bmp)>1.e-10)
                ts->set_failed_test_info(cvtest::TS::FAIL_BAD_ACCURACY);
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};


#ifdef HAVE_PNG
TEST(Highgui_Image, write_big) { CV_GrfmtWriteBigImageTest test; test.safe_run(); }
#endif

TEST(Highgui_Image, write_imageseq) { CV_GrfmtWriteSequenceImageTest test; test.safe_run(); }

TEST(Highgui_Image, read_bmp_rle8) { CV_GrfmtReadBMPRLE8Test test; test.safe_run(); }

#ifdef HAVE_PNG
class CV_GrfmtPNGEncodeTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            vector<uchar> buff;
            Mat im = Mat::zeros(1000,1000, CV_8U);
            //randu(im, 0, 256);
            vector<int> param;
            param.push_back(CV_IMWRITE_PNG_COMPRESSION);
            param.push_back(3); //default(3) 0-9.
            cv::imencode(".png" ,im ,buff, param);

            // hangs
            Mat im2 = imdecode(buff,CV_LOAD_IMAGE_ANYDEPTH);
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Highgui_Image, encode_png) { CV_GrfmtPNGEncodeTest test; test.safe_run(); }

TEST(Highgui_ImreadVSCvtColor, regression)
{
    cvtest::TS& ts = *cvtest::TS::ptr();

    const int MAX_MEAN_DIFF = 1;
    const int MAX_ABS_DIFF = 10;

    string imgName = string(ts.get_data_path()) + "/../cv/shared/lena.png";
    Mat original_image = imread(imgName);
    Mat gray_by_codec = imread(imgName, 0);
    Mat gray_by_cvt;

    cvtColor(original_image, gray_by_cvt, CV_BGR2GRAY);

    Mat diff;
    absdiff(gray_by_codec, gray_by_cvt, diff);

    double actual_avg_diff = (double)mean(diff)[0];
    double actual_maxval, actual_minval;
    minMaxLoc(diff, &actual_minval, &actual_maxval);
    //printf("actual avg = %g, actual maxdiff = %g, npixels = %d\n", actual_avg_diff, actual_maxval, (int)diff.total());

    EXPECT_LT(actual_avg_diff, MAX_MEAN_DIFF);
    EXPECT_LT(actual_maxval, MAX_ABS_DIFF);
}

//Test OpenCV issue 3075 is solved
class CV_GrfmtReadPNGColorPaletteWithAlphaTest : public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            // First Test : Read PNG with alpha, imread flag -1
            Mat img = imread(string(ts->get_data_path()) + "readwrite/color_palette_alpha.png",-1);
            if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);

            ASSERT_TRUE(img.channels() == 4);

            unsigned char* img_data = (unsigned char*)img.data;

            // Verification first pixel is red in BGRA
            ASSERT_TRUE(img_data[0] == 0x00);
            ASSERT_TRUE(img_data[1] == 0x00);
            ASSERT_TRUE(img_data[2] == 0xFF);
            ASSERT_TRUE(img_data[3] == 0xFF);

            // Verification second pixel is red in BGRA
            ASSERT_TRUE(img_data[4] == 0x00);
            ASSERT_TRUE(img_data[5] == 0x00);
            ASSERT_TRUE(img_data[6] == 0xFF);
            ASSERT_TRUE(img_data[7] == 0xFF);

            // Second Test : Read PNG without alpha, imread flag -1
            img = imread(string(ts->get_data_path()) + "readwrite/color_palette_no_alpha.png",-1);
            if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);

            ASSERT_TRUE(img.channels() == 3);

            img_data = (unsigned char*)img.data;

            // Verification first pixel is red in BGR
            ASSERT_TRUE(img_data[0] == 0x00);
            ASSERT_TRUE(img_data[1] == 0x00);
            ASSERT_TRUE(img_data[2] == 0xFF);

            // Verification second pixel is red in BGR
            ASSERT_TRUE(img_data[3] == 0x00);
            ASSERT_TRUE(img_data[4] == 0x00);
            ASSERT_TRUE(img_data[5] == 0xFF);

            // Third Test : Read PNG with alpha, imread flag 1
            img = imread(string(ts->get_data_path()) + "readwrite/color_palette_alpha.png",1);
            if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);

            ASSERT_TRUE(img.channels() == 3);

            img_data = (unsigned char*)img.data;

            // Verification first pixel is red in BGR
            ASSERT_TRUE(img_data[0] == 0x00);
            ASSERT_TRUE(img_data[1] == 0x00);
            ASSERT_TRUE(img_data[2] == 0xFF);

            // Verification second pixel is red in BGR
            ASSERT_TRUE(img_data[3] == 0x00);
            ASSERT_TRUE(img_data[4] == 0x00);
            ASSERT_TRUE(img_data[5] == 0xFF);

            // Fourth Test : Read PNG without alpha, imread flag 1
            img = imread(string(ts->get_data_path()) + "readwrite/color_palette_no_alpha.png",1);
            if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);

            ASSERT_TRUE(img.channels() == 3);

            img_data = (unsigned char*)img.data;

            // Verification first pixel is red in BGR
            ASSERT_TRUE(img_data[0] == 0x00);
            ASSERT_TRUE(img_data[1] == 0x00);
            ASSERT_TRUE(img_data[2] == 0xFF);

            // Verification second pixel is red in BGR
            ASSERT_TRUE(img_data[3] == 0x00);
            ASSERT_TRUE(img_data[4] == 0x00);
            ASSERT_TRUE(img_data[5] == 0xFF);
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
    }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Highgui_Image, read_png_color_palette_with_alpha) { CV_GrfmtReadPNGColorPaletteWithAlphaTest test; test.safe_run(); }
#endif

#ifdef HAVE_JPEG
TEST(Highgui_Jpeg, encode_empty)
{
    cv::Mat img;
    std::vector<uchar> jpegImg;

    ASSERT_THROW(cv::imencode(".jpg", img, jpegImg), cv::Exception);
}
#endif


#ifdef HAVE_TIFF

// these defines are used to resolve conflict between tiff.h and opencv2/core/types_c.h
#define uint64 uint64_hack_
#define int64 int64_hack_
#include "tiff.h"

TEST(Highgui_Tiff, decode_tile16384x16384)
{
    // see issue #2161
    cv::Mat big(16384, 16384, CV_8UC1, cv::Scalar::all(0));
    string file3 = cv::tempfile(".tiff");
    string file4 = cv::tempfile(".tiff");

    std::vector<int> params;
    params.push_back(TIFFTAG_ROWSPERSTRIP);
    params.push_back(big.rows);
    cv::imwrite(file4, big, params);
    cv::imwrite(file3, big.colRange(0, big.cols - 1), params);
    big.release();

    try
    {
        cv::imread(file3);
        EXPECT_NO_THROW(cv::imread(file4));
    }
    catch(const std::bad_alloc&)
    {
        // have no enough memory
    }

    remove(file3.c_str());
    remove(file4.c_str());
}
#endif
