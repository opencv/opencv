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
                    for (int num_channels = 1; num_channels <= 4; num_channels++)
                    {
                        if (num_channels == 2) continue;
                        if (num_channels == 4 && ext!=3 /*TIFF*/) continue;

                        ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_8U, num_channels, ext_from_int(ext).c_str());
                        Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_8U, num_channels), Scalar::all(0));
                        circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), std::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));

                        string img_path = cv::tempfile(ext_from_int(ext).c_str());
                        ts->printf(ts->LOG, "writing      image : %s\n", img_path.c_str());
                        imwrite(img_path, img);

                        ts->printf(ts->LOG, "reading test image : %s\n", img_path.c_str());
                        Mat img_test = imread(img_path, IMREAD_UNCHANGED);

                        if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

                        CV_Assert(img.size() == img_test.size());
                        CV_Assert(img.type() == img_test.type());
                        CV_Assert(num_channels == img_test.channels());

                        double n = cvtest::norm(img, img_test, NORM_L2);
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
                    circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), std::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));

                    string filename = cv::tempfile(".jpg");
                    imwrite(filename, img);
                    ts->printf(ts->LOG, "reading test image : %s\n", filename.c_str());
                    Mat img_test = imread(filename, IMREAD_UNCHANGED);

                    if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

                    CV_Assert(img.size() == img_test.size());
                    CV_Assert(img.type() == img_test.type());

                    double n = cvtest::norm(img, img_test, NORM_L2);
                    if ( n > 1.0)
                    {
                        ts->printf(ts->LOG, "norm = %f \n", n);
                        ts->set_failed_test_info(ts->FAIL_MISMATCH);
                    }
                }
#endif

#ifdef HAVE_TIFF
                for (int num_channels = 1; num_channels <= 4; num_channels++)
                {
                    if (num_channels == 2) continue;
                    // tiff
                    ts->printf(ts->LOG, "image type depth:%d   channels:%d   ext: %s\n", CV_16U, num_channels, ".tiff");
                    Mat img(img_r * k, img_c * k, CV_MAKETYPE(CV_16U, num_channels), Scalar::all(0));
                    circle(img, Point2i((img_c * k) / 2, (img_r * k) / 2), std::min((img_r * k), (img_c * k)) / 4 , Scalar::all(255));

                    string filename = cv::tempfile(".tiff");
                    imwrite(filename, img);
                    ts->printf(ts->LOG, "reading test image : %s\n", filename.c_str());
                    Mat img_test = imread(filename, IMREAD_UNCHANGED);

                    if (img_test.empty()) ts->set_failed_test_info(ts->FAIL_MISMATCH);

                    CV_Assert(img.size() == img_test.size());

                    ts->printf(ts->LOG, "img      : %d ; %d \n", img.channels(), img.depth());
                    ts->printf(ts->LOG, "img_test : %d ; %d \n", img_test.channels(), img_test.depth());

                    CV_Assert(img.type() == img_test.type());


                    double n = cvtest::norm(img, img_test, NORM_L2);
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
            if (cvtest::norm(rle-bmp, NORM_L2)>1.e-10)
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
TEST(Imgcodecs_Image, write_big) { CV_GrfmtWriteBigImageTest test; test.safe_run(); }
#endif

TEST(Imgcodecs_Image, write_imageseq) { CV_GrfmtWriteSequenceImageTest test; test.safe_run(); }

TEST(Imgcodecs_Image, read_bmp_rle8) { CV_GrfmtReadBMPRLE8Test test; test.safe_run(); }

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
            param.push_back(IMWRITE_PNG_COMPRESSION);
            param.push_back(3); //default(3) 0-9.
            cv::imencode(".png" ,im ,buff, param);

            // hangs
            Mat im2 = imdecode(buff,IMREAD_ANYDEPTH);
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Imgcodecs_Image, encode_png) { CV_GrfmtPNGEncodeTest test; test.safe_run(); }

TEST(Imgcodecs_ImreadVSCvtColor, regression)
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

TEST(Imgcodecs_Image, read_png_color_palette_with_alpha) { CV_GrfmtReadPNGColorPaletteWithAlphaTest test; test.safe_run(); }
#endif

#ifdef HAVE_JPEG
TEST(Imgcodecs_Jpeg, encode_empty)
{
    cv::Mat img;
    std::vector<uchar> jpegImg;

    ASSERT_THROW(cv::imencode(".jpg", img, jpegImg), cv::Exception);
}

TEST(Imgcodecs_Jpeg, encode_decode_progressive_jpeg)
{
    cvtest::TS& ts = *cvtest::TS::ptr();
    string input = string(ts.get_data_path()) + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_PROGRESSIVE);
    params.push_back(1);

    string output_progressive = cv::tempfile(".jpg");
    EXPECT_NO_THROW(cv::imwrite(output_progressive, img, params));
    cv::Mat img_jpg_progressive = cv::imread(output_progressive);

    string output_normal = cv::tempfile(".jpg");
    EXPECT_NO_THROW(cv::imwrite(output_normal, img));
    cv::Mat img_jpg_normal = cv::imread(output_normal);

    EXPECT_EQ(0, cvtest::norm(img_jpg_progressive, img_jpg_normal, NORM_INF));

    remove(output_progressive.c_str());
}

TEST(Imgcodecs_Jpeg, encode_decode_optimize_jpeg)
{
    cvtest::TS& ts = *cvtest::TS::ptr();
    string input = string(ts.get_data_path()) + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_OPTIMIZE);
    params.push_back(1);

    string output_optimized = cv::tempfile(".jpg");
    EXPECT_NO_THROW(cv::imwrite(output_optimized, img, params));
    cv::Mat img_jpg_optimized = cv::imread(output_optimized);

    string output_normal = cv::tempfile(".jpg");
    EXPECT_NO_THROW(cv::imwrite(output_normal, img));
    cv::Mat img_jpg_normal = cv::imread(output_normal);

    EXPECT_EQ(0, cvtest::norm(img_jpg_optimized, img_jpg_normal, NORM_INF));

    remove(output_optimized.c_str());
}

TEST(Imgcodecs_Jpeg, encode_decode_rst_jpeg)
{
    cvtest::TS& ts = *cvtest::TS::ptr();
    string input = string(ts.get_data_path()) + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_RST_INTERVAL);
    params.push_back(1);

    string output_rst = cv::tempfile(".jpg");
    EXPECT_NO_THROW(cv::imwrite(output_rst, img, params));
    cv::Mat img_jpg_rst = cv::imread(output_rst);

    string output_normal = cv::tempfile(".jpg");
    EXPECT_NO_THROW(cv::imwrite(output_normal, img));
    cv::Mat img_jpg_normal = cv::imread(output_normal);

    EXPECT_EQ(0, cvtest::norm(img_jpg_rst, img_jpg_normal, NORM_INF));

    remove(output_rst.c_str());
}

#endif


#ifdef HAVE_TIFF

// these defines are used to resolve conflict between tiff.h and opencv2/core/types_c.h
#define uint64 uint64_hack_
#define int64 int64_hack_
#include "tiff.h"

#ifdef ANDROID
// Test disabled as it uses a lot of memory.
// It is killed with SIGKILL by out of memory killer.
TEST(Imgcodecs_Tiff, DISABLED_decode_tile16384x16384)
#else
TEST(Imgcodecs_Tiff, decode_tile16384x16384)
#endif
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
        cv::imread(file3, IMREAD_UNCHANGED);
        EXPECT_NO_THROW(cv::imread(file4, IMREAD_UNCHANGED));
    }
    catch(const std::bad_alloc&)
    {
        // have no enough memory
    }

    remove(file3.c_str());
    remove(file4.c_str());
}

TEST(Imgcodecs_Tiff, write_read_16bit_big_little_endian)
{
    // see issue #2601 "16-bit Grayscale TIFF Load Failures Due to Buffer Underflow and Endianness"

    // Setup data for two minimal 16-bit grayscale TIFF files in both endian formats
    uchar tiff_sample_data[2][86] = { {
        // Little endian
        0x49, 0x49, 0x2a, 0x00, 0x0c, 0x00, 0x00, 0x00, 0xad, 0xde, 0xef, 0xbe, 0x06, 0x00, 0x00, 0x01,
        0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x01, 0x01, 0x03, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x06, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x11, 0x01,
        0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00, 0x17, 0x01, 0x04, 0x00, 0x01, 0x00,
        0x00, 0x00, 0x04, 0x00, 0x00, 0x00 }, {
        // Big endian
        0x4d, 0x4d, 0x00, 0x2a, 0x00, 0x00, 0x00, 0x0c, 0xde, 0xad, 0xbe, 0xef, 0x00, 0x06, 0x01, 0x00,
        0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x02, 0x00, 0x00, 0x01, 0x01, 0x00, 0x03, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x02, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x10,
        0x00, 0x00, 0x01, 0x06, 0x00, 0x03, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0x01, 0x11,
        0x00, 0x04, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x01, 0x17, 0x00, 0x04, 0x00, 0x00,
        0x00, 0x01, 0x00, 0x00, 0x00, 0x04 }
        };

    // Test imread() for both a little endian TIFF and big endian TIFF
    for (int i = 0; i < 2; i++)
    {
        string filename = cv::tempfile(".tiff");

        // Write sample TIFF file
        FILE* fp = fopen(filename.c_str(), "wb");
        ASSERT_TRUE(fp != NULL);
        ASSERT_EQ((size_t)1, fwrite(tiff_sample_data, 86, 1, fp));
        fclose(fp);

        Mat img = imread(filename, IMREAD_UNCHANGED);

        EXPECT_EQ(1, img.rows);
        EXPECT_EQ(2, img.cols);
        EXPECT_EQ(CV_16U, img.type());
        EXPECT_EQ(sizeof(ushort), img.elemSize());
        EXPECT_EQ(1, img.channels());
        EXPECT_EQ(0xDEAD, img.at<ushort>(0,0));
        EXPECT_EQ(0xBEEF, img.at<ushort>(0,1));

        remove(filename.c_str());
    }
}

class CV_GrfmtReadTifTiledWithNotFullTiles: public cvtest::BaseTest
{
public:
    void run(int)
    {
        try
        {
            /* see issue #3472 - dealing with tiled images where the tile size is
             * not a multiple of image size.
             * The tiled images were created with 'convert' from ImageMagick,
             * using the command 'convert <input> -define tiff:tile-geometry=128x128 -depth [8|16] <output>
             * Note that the conversion to 16 bits expands the range from 0-255 to 0-255*255,
             * so the test converts back but rounding errors cause small differences.
             */
            cv::Mat img = imread(string(ts->get_data_path()) + "readwrite/non_tiled.tif",-1);
            if (img.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            ASSERT_TRUE(img.channels() == 3);
            cv::Mat tiled8 = imread(string(ts->get_data_path()) + "readwrite/tiled_8.tif", -1);
            if (tiled8.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            ASSERT_PRED_FORMAT2(cvtest::MatComparator(0, 0), img, tiled8);

            cv::Mat tiled16 = imread(string(ts->get_data_path()) + "readwrite/tiled_16.tif", -1);
            if (tiled16.empty()) ts->set_failed_test_info(cvtest::TS::FAIL_INVALID_TEST_DATA);
            ASSERT_TRUE(tiled16.elemSize() == 6);
            tiled16.convertTo(tiled8, CV_8UC3, 1./256.);
            ASSERT_PRED_FORMAT2(cvtest::MatComparator(2, 0), img, tiled8);
            // What about 32, 64 bit?
        }
        catch(...)
        {
            ts->set_failed_test_info(cvtest::TS::FAIL_EXCEPTION);
        }
        ts->set_failed_test_info(cvtest::TS::OK);
    }
};

TEST(Imgcodecs_Tiff, decode_tile_remainder)
{
    CV_GrfmtReadTifTiledWithNotFullTiles test; test.safe_run();
}

#endif

#ifdef HAVE_WEBP

TEST(Imgcodecs_WebP, encode_decode_lossless_webp)
{
    cvtest::TS& ts = *cvtest::TS::ptr();
    string input = string(ts.get_data_path()) + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    string output = cv::tempfile(".webp");
    EXPECT_NO_THROW(cv::imwrite(output, img)); // lossless

    cv::Mat img_webp = cv::imread(output);

    std::vector<unsigned char> buf;

    FILE * wfile = NULL;

    wfile = fopen(output.c_str(), "rb");
    if (wfile != NULL)
    {
        fseek(wfile, 0, SEEK_END);
        size_t wfile_size = ftell(wfile);
        fseek(wfile, 0, SEEK_SET);

        buf.resize(wfile_size);

        size_t data_size = fread(&buf[0], 1, wfile_size, wfile);

        if(wfile)
        {
            fclose(wfile);
        }

        if (data_size != wfile_size)
        {
            EXPECT_TRUE(false);
        }
    }

    remove(output.c_str());

    cv::Mat decode = cv::imdecode(buf, IMREAD_COLOR);
    ASSERT_FALSE(decode.empty());
    EXPECT_TRUE(cvtest::norm(decode, img_webp, NORM_INF) == 0);

    ASSERT_FALSE(img_webp.empty());

    EXPECT_TRUE(cvtest::norm(img, img_webp, NORM_INF) == 0);
}

TEST(Imgcodecs_WebP, encode_decode_lossy_webp)
{
    cvtest::TS& ts = *cvtest::TS::ptr();
    std::string input = std::string(ts.get_data_path()) + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    for(int q = 100; q>=0; q-=20)
    {
        std::vector<int> params;
        params.push_back(IMWRITE_WEBP_QUALITY);
        params.push_back(q);
        string output = cv::tempfile(".webp");

        EXPECT_NO_THROW(cv::imwrite(output, img, params));
        cv::Mat img_webp = cv::imread(output);
        remove(output.c_str());
        EXPECT_FALSE(img_webp.empty());
        EXPECT_EQ(3,   img_webp.channels());
        EXPECT_EQ(512, img_webp.cols);
        EXPECT_EQ(512, img_webp.rows);
    }
}

TEST(Imgcodecs_WebP, encode_decode_with_alpha_webp)
{
    cvtest::TS& ts = *cvtest::TS::ptr();
    std::string input = std::string(ts.get_data_path()) + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Mat> imgs;
    cv::split(img, imgs);
    imgs.push_back(cv::Mat(imgs[0]));
    imgs[imgs.size() - 1] = cv::Scalar::all(128);
    cv::merge(imgs, img);

    string output = cv::tempfile(".webp");

    EXPECT_NO_THROW(cv::imwrite(output, img));
    cv::Mat img_webp = cv::imread(output);
    remove(output.c_str());
    EXPECT_FALSE(img_webp.empty());
    EXPECT_EQ(4,   img_webp.channels());
    EXPECT_EQ(512, img_webp.cols);
    EXPECT_EQ(512, img_webp.rows);
}

#endif

TEST(Imgcodecs_Hdr, regression)
{
    string folder = string(cvtest::TS::ptr()->get_data_path()) + "/readwrite/";
    string name_rle = folder + "rle.hdr";
    string name_no_rle = folder + "no_rle.hdr";
    Mat img_rle = imread(name_rle, -1);
    ASSERT_FALSE(img_rle.empty()) << "Could not open " << name_rle;
    Mat img_no_rle = imread(name_no_rle, -1);
    ASSERT_FALSE(img_no_rle.empty()) << "Could not open " << name_no_rle;

    double min = 0.0, max = 1.0;
    minMaxLoc(abs(img_rle - img_no_rle), &min, &max);
    ASSERT_FALSE(max > DBL_EPSILON);
    string tmp_file_name = tempfile(".hdr");
    vector<int>param(1);
    for(int i = 0; i < 2; i++) {
        param[0] = i;
        imwrite(tmp_file_name, img_rle, param);
        Mat written_img = imread(tmp_file_name, -1);
        ASSERT_FALSE(written_img.empty()) << "Could not open " << tmp_file_name;
        minMaxLoc(abs(img_rle - written_img), &min, &max);
        ASSERT_FALSE(max > DBL_EPSILON);
    }
}
