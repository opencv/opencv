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

// Expected result categories for WebP encoding tests.
enum ImencodeLosslessResult {
    LOSSY,      // Expect lossy compression (pixel differences allowed)
    LOSSLESS,   // Expect standard lossless compression (pixel values match)
    EXACT       // Expect exact lossless (preserving RGB values of transparent pixels)
};

typedef std::tuple<int, int, ImencodeLosslessResult> WebPModePriorityParams;
class Imgcodecs_WebP_Mode_Priority : public testing::TestWithParam<WebPModePriorityParams> {};

TEST_P(Imgcodecs_WebP_Mode_Priority, encode_webp_mode_priority)
{
    const int mode    = std::get<0>(GetParam());
    const int quality = std::get<1>(GetParam());
    const ImencodeLosslessResult expected = std::get<2>(GetParam());

    // Generate a 100x100 RGBA test image.
    // Set a transparent pixel with specific color (Blue) to verify EXACT mode.
    Mat src(100, 100, CV_8UC4, Scalar(255, 255, 255, 255));
    src.at<Vec4b>(0, 0) = Vec4b(255, 0, 0, 0); // Transparent Blue (B:255, G:0, R:0, A:0)

    // Build the imwrite parameter vector dynamically.
    std::vector<int> params;
    if (mode != -1) {
        params.push_back(IMWRITE_WEBP_LOSSLESS_MODE);
        params.push_back(mode);
    }
    if (quality != -1) {
        params.push_back(IMWRITE_WEBP_QUALITY);
        params.push_back(quality);
    }

    // Encode to memory and decode back.
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".webp", src, buf, params));
    Mat dst = imdecode(buf, IMREAD_UNCHANGED);
    ASSERT_FALSE(dst.empty());

    // Validation logic
    if (expected == LOSSY) {
        // We expect some differences in lossy mode
        double diff = cv::norm(src, dst, NORM_INF);
        EXPECT_GT(diff, 0) << "Should be lossy (Quality: " << quality << ")";
    }
    else if (expected == LOSSLESS) {
        // Standard lossless: we allow the library to modify RGB values
        // of fully transparent pixels (A=0) to improve compression ratio.
        // Thus, we compare only visible pixels or check with a slightly relaxed condition.

        // Option A: If you want to allow RGB changes on A=0:
        // We can't use cv::norm directly if A=0 pixels are modified.
        // Let's check if they are identical except for the transparent pixel.
        Mat diff;
        absdiff(src, dst, diff);
        Scalar total_diff = sum(diff);
        // If only the (0,0) pixel changed from (255,0,0,0) to (0,0,0,0),
        // total_diff will be 255.
        EXPECT_LE(total_diff[0] + total_diff[1] + total_diff[2], 255)
            << "Standard lossless should not have significant pixel differences";
        EXPECT_EQ(src.at<Vec4b>(0,0)[3], dst.at<Vec4b>(0,0)[3]) << "Alpha must be preserved";
    }
    else if (expected == EXACT) {
        // Exact lossless: Every single bit must match, including transparent pixels.
        double diff = cv::norm(src, dst, NORM_INF);
        EXPECT_EQ(0, diff) << "Exact mode must preserve all pixel values perfectly";
        EXPECT_EQ(src.at<Vec4b>(0, 0), dst.at<Vec4b>(0, 0))
            << "RGB values of transparent pixels must be preserved in EXACT mode";
    }
    else {
        FAIL() << "Unknown expectation type";
    }
}

/**
 * Helper to generate human-readable test names in gtest output.
 */
static std::string getModeStr(int m) {
    if (m == -1) return "OMIT";
    if (m == IMWRITE_WEBP_LOSSLESS_OFF) return "OFF";
    if (m == IMWRITE_WEBP_LOSSLESS_ON) return "ON";
    if (m == IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR) return "PRESERVE";
    return "UNKNOWN";
}

static std::string getExpectStr(ImencodeLosslessResult r) {
    return (r == LOSSY) ? "LOSSY" : (r == EXACT) ? "EXACT" : "LOSSLESS";
}

INSTANTIATE_TEST_CASE_P(Imgcodecs, Imgcodecs_WebP_Mode_Priority,
    testing::Values(
        // Default (OMIT mode) cases
        WebPModePriorityParams(-1, -1,  LOSSLESS),
        WebPModePriorityParams(-1, 80,  LOSSY),
        WebPModePriorityParams(-1, 101, LOSSLESS),

        // LOSSLESS_OFF (Explicitly off)
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_OFF, -1,  LOSSLESS),
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_OFF, 80,  LOSSY),
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_OFF, 101, LOSSLESS),

        // LOSSLESS_ON (Force lossless)
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_ON, -1,  LOSSLESS),
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_ON, 80,  LOSSLESS),
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_ON, 101, LOSSLESS),

        // PRESERVE_COLOR (Exact lossless)
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR, -1,  EXACT),
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR, 80,  EXACT),
        WebPModePriorityParams(IMWRITE_WEBP_LOSSLESS_PRESERVE_COLOR, 101, EXACT)
    ),
    [](const testing::TestParamInfo<WebPModePriorityParams>& info_) {
        std::string mode = getModeStr(std::get<0>(info_.param));
        int q = std::get<1>(info_.param);
        std::string q_str = (q == -1) ? "omit" : std::to_string(q);
        return mode + "_q" + q_str + "_" + getExpectStr(std::get<2>(info_.param));
    }
);

#endif // HAVE_WEBP

}} // namespace
