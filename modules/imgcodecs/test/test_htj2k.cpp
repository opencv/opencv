// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
#include "test_precomp.hpp"

namespace opencv_test { namespace {

#ifdef HAVE_OPENHTJ2K

// Uniform random samples over the whole range of the type, so that lossless coding is
// exercised on every bit plane.
static Mat randomImage(Size size, int type)
{
    Mat img(size, type);
    RNG& rng = theRNG();
    switch (CV_MAT_DEPTH(type))
    {
    case CV_8U:  rng.fill(img, RNG::UNIFORM, 0, 256); break;
    case CV_8S:  rng.fill(img, RNG::UNIFORM, -128, 128); break;
    case CV_16U: rng.fill(img, RNG::UNIFORM, 0, 65536); break;
    case CV_16S: rng.fill(img, RNG::UNIFORM, -32768, 32768); break;
    default: CV_Error(Error::StsBadArg, "unsupported type");
    }
    return img;
}

// A smooth colour image with some texture, for lossy coding.
static Mat smoothImage(Size size)
{
    Mat img(size, CV_8UC3);
    for (int y = 0; y < size.height; y++)
        for (int x = 0; x < size.width; x++)
            img.at<Vec3b>(y, x) = Vec3b(saturate_cast<uchar>(x * 255 / size.width),
                                        saturate_cast<uchar>(y * 255 / size.height),
                                        saturate_cast<uchar>(128 + 100 * std::sin(x * 0.05) * std::cos(y * 0.07)));
    circle(img, Point(size.width / 3, size.height / 2), size.height / 4, Scalar(30, 200, 90), FILLED);
    Mat noise(size, CV_8UC3);
    theRNG().fill(noise, RNG::UNIFORM, 0, 16);
    img += noise;
    GaussianBlur(img, img, Size(3, 3), 0);
    return img;
}

//==================================================================================================
// Lossless round trip (the default): every supported type, both containers.

typedef tuple<perf::MatType, string> HTJ2K_Type_Ext;
typedef testing::TestWithParam<HTJ2K_Type_Ext> Imgcodecs_HTJ2K_Lossless;

TEST_P(Imgcodecs_HTJ2K_Lossless, encode_decode)
{
    const int type = get<0>(GetParam());
    const string ext = get<1>(GetParam());
    const Mat src = randomImage(Size(127, 61), type);

    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(ext, src, buf));
    const Mat dst = imdecode(buf, IMREAD_UNCHANGED);
    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(src.type(), dst.type());
    EXPECT_EQ(0, cvtest::norm(src, dst, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(/**/, Imgcodecs_HTJ2K_Lossless, testing::Combine(
    testing::Values(CV_8UC1, CV_8UC3, CV_8UC4, CV_8SC1, CV_16UC1, CV_16UC3, CV_16UC4, CV_16SC1, CV_16SC3),
    testing::Values(".jph", ".jhc")));

typedef testing::TestWithParam<Size> Imgcodecs_HTJ2K_Size;

TEST_P(Imgcodecs_HTJ2K_Size, encode_decode)
{
    const Mat src = randomImage(GetParam(), CV_8UC3);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf));
    const Mat dst = imdecode(buf, IMREAD_UNCHANGED);
    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(0, cvtest::norm(src, dst, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(/**/, Imgcodecs_HTJ2K_Size, testing::Values(
    Size(1, 1), Size(3, 2), Size(1, 300), Size(300, 1), Size(64, 64), Size(257, 129)));

TEST(Imgcodecs_HTJ2K, imwrite_imread)
{
    const Mat src = randomImage(Size(100, 50), CV_16UC3);
    const string fname = cv::tempfile(".jph");
    ASSERT_TRUE(imwrite(fname, src));
    const Mat dst = imread(fname, IMREAD_UNCHANGED);
    EXPECT_EQ(0, remove(fname.c_str()));
    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(0, cvtest::norm(src, dst, NORM_INF));
}

//==================================================================================================
// Lossy coding.

TEST(Imgcodecs_HTJ2K, lossy_quality)
{
    const Mat src = smoothImage(Size(320, 240));
    std::vector<uchar> lossless, q90, q50;
    ASSERT_TRUE(imencode(".jph", src, lossless));
    ASSERT_TRUE(imencode(".jph", src, q90, { IMWRITE_HTJ2K_QUALITY_SCALE, 90 }));
    ASSERT_TRUE(imencode(".jph", src, q50, { IMWRITE_HTJ2K_QUALITY_SCALE, 50 }));
    EXPECT_LT(q90.size(), lossless.size());
    EXPECT_LT(q50.size(), q90.size());

    const Mat d90 = imdecode(q90, IMREAD_COLOR);
    const Mat d50 = imdecode(q50, IMREAD_COLOR);
    ASSERT_FALSE(d90.empty());
    ASSERT_FALSE(d50.empty());
    const double psnr90 = cvtest::PSNR(src, d90);
    const double psnr50 = cvtest::PSNR(src, d50);
    EXPECT_GT(psnr90, 35);
    EXPECT_GT(psnr50, 28);
    EXPECT_GT(psnr90, psnr50);
}

TEST(Imgcodecs_HTJ2K, lossy_bgra)
{
    const Mat bgr = smoothImage(Size(320, 240));
    Mat alpha(bgr.size(), CV_8U);
    for (int y = 0; y < alpha.rows; y++)
        for (int x = 0; x < alpha.cols; x++)
            alpha.at<uchar>(y, x) = saturate_cast<uchar>(x * 255 / (alpha.cols - 1));
    Mat src;
    const Mat planes[] = { bgr, alpha };
    merge(planes, 2, src);
    ASSERT_EQ(CV_8UC4, src.type());

    std::vector<uchar> q90, q50;
    ASSERT_TRUE(imencode(".jph", src, q90, { IMWRITE_HTJ2K_QUALITY_SCALE, 90 }));
    ASSERT_TRUE(imencode(".jph", src, q50, { IMWRITE_HTJ2K_QUALITY_SCALE, 50 }));
    EXPECT_LT(q50.size(), q90.size());
    const Mat d90 = imdecode(q90, IMREAD_UNCHANGED);
    const Mat d50 = imdecode(q50, IMREAD_UNCHANGED);
    ASSERT_FALSE(d90.empty());
    ASSERT_FALSE(d50.empty());
    EXPECT_EQ(CV_8UC4, d90.type());
    std::vector<Mat> c90, c50, cs;
    split(d90, c90);
    split(d50, c50);
    split(src, cs);
    for (int c = 0; c < 4; c++)
    {
        EXPECT_GT(cvtest::PSNR(cs[c], c90[c]), 35) << "channel " << c;
        EXPECT_GT(cvtest::PSNR(cs[c], c50[c]), 28) << "channel " << c;
    }
}

TEST(Imgcodecs_HTJ2K, lossy_16bit)
{
    Mat src;
    smoothImage(Size(160, 120)).convertTo(src, CV_16U, 256);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf, { IMWRITE_HTJ2K_QUALITY_SCALE, 90 }));
    const Mat dst = imdecode(buf, IMREAD_UNCHANGED);
    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(CV_16UC3, dst.type());
    EXPECT_GT(cv::PSNR(src, dst, 65535), 35);
}

//==================================================================================================
// Output conversions requested through the imread flags.

// Reference for reading a 16-bit image as 8 bits: the top 8 bits of each sample.
static Mat top8(const Mat& src16, int add = 0)
{
    Mat flat = src16.reshape(1), dst(flat.size(), CV_8U);
    for (int y = 0; y < flat.rows; y++)
        for (int x = 0; x < flat.cols; x++)
        {
            const int v = flat.depth() == CV_16S ? flat.at<short>(y, x) : flat.at<ushort>(y, x);
            dst.at<uchar>(y, x) = (uchar)((v + add) >> 8);
        }
    return dst.reshape(src16.channels());
}

TEST(Imgcodecs_HTJ2K, read_16bit_as_8bit)
{
    const Mat src = randomImage(Size(33, 17), CV_16UC3);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf));
    const Mat dst = imdecode(buf, IMREAD_COLOR);
    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(CV_8UC3, dst.type());
    EXPECT_EQ(0, cvtest::norm(top8(src), dst, NORM_INF));
}

TEST(Imgcodecs_HTJ2K, read_signed_as_unsigned)
{
    const Mat src = randomImage(Size(40, 20), CV_16SC1);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jhc", src, buf));
    const Mat dst = imdecode(buf, IMREAD_GRAYSCALE);
    ASSERT_FALSE(dst.empty());
    EXPECT_EQ(CV_8UC1, dst.type());
    EXPECT_EQ(0, cvtest::norm(top8(src, 32768), dst, NORM_INF));
}

TEST(Imgcodecs_HTJ2K, read_rgb_order)
{
    const Mat src = randomImage(Size(33, 17), CV_8UC3);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf));
    const Mat dst = imdecode(buf, IMREAD_COLOR_RGB);
    ASSERT_FALSE(dst.empty());
    Mat expected;
    cvtColor(src, expected, COLOR_BGR2RGB);
    EXPECT_EQ(0, cvtest::norm(expected, dst, NORM_INF));
}

TEST(Imgcodecs_HTJ2K, read_color_as_gray)
{
    const Mat src = randomImage(Size(33, 17), CV_8UC3);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf));
    const Mat dst = imdecode(buf, IMREAD_GRAYSCALE);
    ASSERT_FALSE(dst.empty());
    Mat expected;
    cvtColor(src, expected, COLOR_BGR2GRAY);
    EXPECT_LE(cvtest::norm(expected, dst, NORM_INF), 1);
}

TEST(Imgcodecs_HTJ2K, read_gray_as_color)
{
    const Mat src = randomImage(Size(33, 17), CV_8UC1);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf));
    const Mat dst = imdecode(buf, IMREAD_COLOR);
    ASSERT_FALSE(dst.empty());
    Mat expected;
    cvtColor(src, expected, COLOR_GRAY2BGR);
    EXPECT_EQ(0, cvtest::norm(expected, dst, NORM_INF));
}

//==================================================================================================
// Colour spaces of JPH files from other writers. The encoder writes sRGB and greyscale only, so
// the tests set the EnumCS of the colour specification box (ISO/IEC 15444-1 I.5.3.3) themselves.

static void setColourSpace(std::vector<uchar>& jph, uint32_t enumcs)
{
    static const uchar colr[4] = { 'c', 'o', 'l', 'r' };
    auto it = std::search(jph.begin(), jph.end(), colr, colr + 4);
    ASSERT_TRUE(it != jph.end());
    const size_t pos = (size_t)(it - jph.begin()) + 4;  // METH, PREC, APPROX, EnumCS
    ASSERT_LE(pos + 7, jph.size());
    ASSERT_EQ(1, jph[pos]);  // enumerated colour space
    for (int i = 0; i < 4; i++)
        jph[pos + 3 + i] = (uchar)(enumcs >> (24 - 8 * i));
}

TEST(Imgcodecs_HTJ2K, read_ycc)
{
    // Read as sYCC (EnumCS 18), the components R, G, B of src are Y, Cb, Cr.
    const Mat src = randomImage(Size(33, 17), CV_8UC4);
    std::vector<uchar> buf;
    ASSERT_TRUE(imencode(".jph", src, buf));
    ASSERT_NO_FATAL_FAILURE(setColourSpace(buf, 18));
    std::vector<Mat> ycc;  // B, G, R, A of src = Cr, Cb, Y, A
    split(src, ycc);

    const Mat gray = imdecode(buf, IMREAD_GRAYSCALE);
    ASSERT_FALSE(gray.empty());
    EXPECT_EQ(0, cvtest::norm(ycc[2], gray, NORM_INF));  // Y

    const Mat bgra = imdecode(buf, IMREAD_UNCHANGED);
    ASSERT_FALSE(bgra.empty());
    ASSERT_EQ(CV_8UC4, bgra.type());
    Mat expected(src.size(), CV_8UC4);
    for (int y = 0; y < src.rows; y++)
        for (int x = 0; x < src.cols; x++)
        {
            const Vec4b v = src.at<Vec4b>(y, x);
            const double Y = v[2], Cb = v[1] - 128.0, Cr = v[0] - 128.0;
            expected.at<Vec4b>(y, x) = Vec4b(saturate_cast<uchar>(Y + 1.772 * Cb),
                                             saturate_cast<uchar>(Y - 0.344136 * Cb - 0.714136 * Cr),
                                             saturate_cast<uchar>(Y + 1.402 * Cr), v[3]);
        }
    EXPECT_LE(cvtest::norm(expected, bgra, NORM_INF), 1);
}

TEST(Imgcodecs_HTJ2K, unsupported_colour_space)
{
    struct Case { int type; uint32_t enumcs; } cases[] = {
        { CV_8UC4, 12 },  // CMYK
        { CV_8UC3, 14 },  // CIELab
        { CV_8UC3, 24 },  // e-sYCC
        { CV_8UC3, 17 },  // greyscale with three components
        { CV_8UC1, 16 },  // sRGB with one component
    };
    for (const Case& c : cases)
    {
        std::vector<uchar> buf;
        ASSERT_TRUE(imencode(".jph", randomImage(Size(16, 16), c.type), buf));
        ASSERT_NO_FATAL_FAILURE(setColourSpace(buf, c.enumcs));
        EXPECT_TRUE(imdecode(buf, IMREAD_UNCHANGED).empty()) << "EnumCS " << c.enumcs;
        EXPECT_TRUE(imdecode(buf, IMREAD_COLOR).empty()) << "EnumCS " << c.enumcs;
    }
}

//==================================================================================================
// Format detection.

TEST(Imgcodecs_HTJ2K, readers_and_writers)
{
    EXPECT_TRUE(haveImageWriter(".jph"));
    EXPECT_TRUE(haveImageWriter(".jhc"));
    for (const char* ext : { ".jph", ".jhc" })
    {
        const string fname = cv::tempfile(ext);
        ASSERT_TRUE(imwrite(fname, randomImage(Size(16, 16), CV_8UC3)));
        EXPECT_TRUE(haveImageReader(fname)) << ext;
        EXPECT_EQ(0, remove(fname.c_str()));
    }
}

// A truncated stream is refused, wherever it is cut: in the file boxes, the main header, a
// tile-part or the EOC marker. Cuts before the CAP marker of a codestream go to another decoder.
TEST(Imgcodecs_HTJ2K, truncated_input)
{
    const Mat src = randomImage(Size(32, 32), CV_8UC3);
    for (const char* ext : { ".jph", ".jhc" })
    {
        std::vector<uchar> buf;
        ASSERT_TRUE(imencode(ext, src, buf));
        ASSERT_FALSE(imdecode(buf, IMREAD_UNCHANGED).empty()) << ext;
        for (size_t len = 1; len < buf.size(); len++)
        {
            const std::vector<uchar> cut(buf.begin(), buf.begin() + len);
            EXPECT_TRUE(imdecode(cut, IMREAD_UNCHANGED).empty()) << ext << " cut to " << len << " bytes";
        }
    }
}

#endif  // HAVE_OPENHTJ2K

}}  // namespace
