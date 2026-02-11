// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_WEBP

static void readFileBytes(const std::string& fname, std::vector<unsigned char>& buf)
{
    FILE * wfile = fopen(fname.c_str(), "rb");
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

        EXPECT_EQ(data_size, wfile_size);
    }
}

TEST(Imgcodecs_WebP, encode_decode_lossless_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    string filename = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(filename);
    ASSERT_FALSE(img.empty());

    string output = cv::tempfile(".webp");
    EXPECT_NO_THROW(cv::imwrite(output, img)); // lossless

    cv::Mat img_webp = cv::imread(output);

    std::vector<unsigned char> buf;
    readFileBytes(output, buf);
    EXPECT_EQ(0, remove(output.c_str()));

    cv::Mat decode = cv::imdecode(buf, IMREAD_COLOR);
    ASSERT_FALSE(decode.empty());
    EXPECT_TRUE(cvtest::norm(decode, img_webp, NORM_INF) == 0);

    cv::Mat decode_rgb = cv::imdecode(buf, IMREAD_COLOR_RGB);
    ASSERT_FALSE(decode_rgb.empty());

    cvtColor(decode_rgb, decode_rgb, COLOR_RGB2BGR);
    EXPECT_TRUE(cvtest::norm(decode_rgb, img_webp, NORM_INF) == 0);

    ASSERT_FALSE(img_webp.empty());

    EXPECT_TRUE(cvtest::norm(img, img_webp, NORM_INF) == 0);
}

TEST(Imgcodecs_WebP, encode_decode_lossy_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    std::string input = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    for(int q = 100; q>=0; q-=20)
    {
        std::vector<int> params;
        params.push_back(IMWRITE_WEBP_QUALITY);
        params.push_back(MAX(q,1));
        string output = cv::tempfile(".webp");

        EXPECT_NO_THROW(cv::imwrite(output, img, params));
        cv::Mat img_webp = cv::imread(output);
        EXPECT_EQ(0, remove(output.c_str()));
        EXPECT_FALSE(img_webp.empty());
        EXPECT_EQ(3,   img_webp.channels());
        EXPECT_EQ(512, img_webp.cols);
        EXPECT_EQ(512, img_webp.rows);
    }
}

TEST(Imgcodecs_WebP, encode_decode_with_alpha_webp)
{
    const string root = cvtest::TS::ptr()->get_data_path();
    std::string input = root + "../cv/shared/lena.png";
    cv::Mat img = cv::imread(input);
    ASSERT_FALSE(img.empty());

    std::vector<cv::Mat> imgs;
    cv::split(img, imgs);
    imgs.push_back(cv::Mat(imgs[0]));
    imgs[imgs.size() - 1] = cv::Scalar::all(128);
    cv::merge(imgs, img);

    string output = cv::tempfile(".webp");

    EXPECT_NO_THROW(cv::imwrite(output, img));
    cv::Mat img_webp = cv::imread(output, IMREAD_UNCHANGED);
    cv::Mat img_webp_bgr = cv::imread(output); // IMREAD_COLOR by default
    EXPECT_EQ(0, remove(output.c_str()));
    EXPECT_FALSE(img_webp.empty());
    EXPECT_EQ(4,   img_webp.channels());
    EXPECT_EQ(512, img_webp.cols);
    EXPECT_EQ(512, img_webp.rows);
    EXPECT_FALSE(img_webp_bgr.empty());
    EXPECT_EQ(3,   img_webp_bgr.channels());
    EXPECT_EQ(512, img_webp_bgr.cols);
    EXPECT_EQ(512, img_webp_bgr.rows);
}

// See https://github.com/opencv/opencv/issues/28503
TEST(Imgcodecs_WebP, encode_decode_LOSSLESS_MODE)
{
    cv::Mat img(cv::Size(64,4), CV_8UC4, cv::Scalar(124,64,67,0) );
    for(int ix = 0; ix < img.size().width; ix++)
    {
        img.at<Vec4b>(0, ix)[3] = 0;   // Transpacency pixel
        img.at<Vec4b>(1, ix)[3] = 1;
        img.at<Vec4b>(2, ix)[3] = 254;
        img.at<Vec4b>(3, ix)[3] = 255;
    }

    std::vector<uint8_t> work;
    EXPECT_NO_THROW(cv::imencode(".webp", img, work, {IMWRITE_WEBP_LOSSLESS_MODE, IMWRITE_WEBP_LOSSLESS_ON}));
    cv::Mat img_ON = cv::imdecode(work, IMREAD_UNCHANGED);
    EXPECT_NO_THROW(cv::imencode(".webp", img, work, {IMWRITE_WEBP_LOSSLESS_MODE, IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR}));
    cv::Mat img_PRESERVE_COLOR = cv::imdecode(work, IMREAD_UNCHANGED);

    for(int ix = 0; ix < img.size().width; ix++)
    {
        EXPECT_EQ(img_ON.at<Vec4b>(0, ix),             Vec4b(0, 0, 0, 0));  // LOSSLESS_ON -> COLOR will be optimized/dropped.
        EXPECT_EQ(img_PRESERVE_COLOR.at<Vec4b>(0, ix), Vec4b(124,64,67,0)); // PRESERVE_COLOR

        EXPECT_EQ(img_ON.at<Vec4b>(1, ix), img_PRESERVE_COLOR.at<Vec4b>(1, ix) );
        EXPECT_EQ(img_ON.at<Vec4b>(2, ix), img_PRESERVE_COLOR.at<Vec4b>(2, ix) );
        EXPECT_EQ(img_ON.at<Vec4b>(3, ix), img_PRESERVE_COLOR.at<Vec4b>(3, ix) );
    }
}

#endif // HAVE_WEBP

}} // namespace
