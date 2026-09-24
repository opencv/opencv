// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

static void requireOpenCvMjpeg()
{
    if (!videoio_registry::hasBackend(CAP_OPENCV_MJPEG))
        throw SkipTestException("CAP_OPENCV_MJPEG is not available");
}

// Smooth gradients so JPEG Q75 round-trip PSNR is dominated by the encoder,
// not by MCU zero-padding of tiny frames or hard 8x8 edges.
static void fillMjpegGradient(Mat& frame, bool rgba)
{
    CV_Assert(frame.depth() == CV_8U);
    const int cn = frame.channels();
    for (int y = 0; y < frame.rows; y++)
    {
        uchar* row = frame.ptr<uchar>(y);
        for (int x = 0; x < frame.cols; x++)
        {
            const int gx = (x * 255) / std::max(frame.cols - 1, 1);
            const int gy = (y * 255) / std::max(frame.rows - 1, 1);
            if (cn == 1)
            {
                row[x] = (uchar)((gx + gy) / 2);
            }
            else if (rgba)
            {
                row[x * cn + 0] = (uchar)gx;             // R
                row[x * cn + 1] = (uchar)gy;             // G
                row[x * cn + 2] = (uchar)((255 - gx + gy) / 2); // B
                row[x * cn + 3] = 255;
            }
            else
            {
                row[x * cn + 0] = (uchar)gx;             // B
                row[x * cn + 1] = (uchar)gy;             // G
                row[x * cn + 2] = (uchar)((255 - gx + gy) / 2); // R
            }
        }
    }
}

static Mat encodeDecodeMjpeg(const Mat& src, Size writerSize, bool isColor,
                             double quality, int nstripes)
{
    const std::string filename = cv::tempfile(".avi");
    const int fourcc = VideoWriter::fourcc('M', 'J', 'P', 'G');
    VideoWriter writer;
    EXPECT_NO_THROW(writer.open(filename, CAP_OPENCV_MJPEG, fourcc, 25.0, writerSize, isColor));
    EXPECT_TRUE(writer.isOpened()) << "failed to open OpenCV MJPEG writer";
    if (quality >= 0)
    {
        EXPECT_TRUE(writer.set(VIDEOWRITER_PROP_QUALITY, quality));
    }
    if (nstripes > 0)
    {
        EXPECT_TRUE(writer.set(VIDEOWRITER_PROP_NSTRIPES, nstripes));
    }

    EXPECT_NO_THROW(writer.write(src));
    EXPECT_NO_THROW(writer.release());

    VideoCapture cap;
    EXPECT_NO_THROW(cap.open(filename, CAP_OPENCV_MJPEG));
    EXPECT_TRUE(cap.isOpened()) << "failed to open OpenCV MJPEG reader: " << filename;

    Mat decoded;
    EXPECT_NO_THROW(cap >> decoded);
    EXPECT_FALSE(decoded.empty());
    EXPECT_EQ(writerSize, decoded.size());
    cap.release();
    EXPECT_EQ(0, remove(filename.c_str()));
    return decoded;
}

typedef tuple<Size, int /*quality*/, int /*nstripes*/> MJPEGAccuracyParam;

class Videoio_MJPEG_Accuracy : public testing::TestWithParam<MJPEGAccuracyParam>
{
};

TEST_P(Videoio_MJPEG_Accuracy, encode_decode)
{
    requireOpenCvMjpeg();

    const Size sz = get<0>(GetParam());
    const int quality = get<1>(GetParam());
    const int nstripes = get<2>(GetParam());

    Mat src(sz, CV_8UC3);
    fillMjpegGradient(src, false);

    const Mat decoded = encodeDecodeMjpeg(src, sz, true, quality, nstripes);
    ASSERT_FALSE(decoded.empty());
    ASSERT_EQ(CV_8UC3, decoded.type());

    const double psnr = cvtest::PSNR(decoded, src);
    // Existing videoio_synthetic MJPEG check is 30 dB. These sizes include
    // leftover MCUs (17/33/641) that hit scalar tails next to the SIMD 16-wide path.
    const double minPsnr = 30.0;
    EXPECT_GT(psnr, minPsnr)
        << "size=" << sz << " Q=" << quality << " stripes=" << nstripes << " PSNR=" << psnr;
}

INSTANTIATE_TEST_CASE_P(
    videoio,
    Videoio_MJPEG_Accuracy,
    testing::Combine(
        testing::Values(
            Size(16, 16),   // exact color MCU (SIMD convertToYUV)
            Size(49, 41),   // SIMD MCU + scalar leftover (not 16-aligned)
            Size(32, 31),   // leftover height
            Size(33, 32),   // leftover width
            Size(64, 48),
            Size(320, 240),
            Size(641, 481)
        ),
        testing::Values(75, 95),
        testing::Values(1, 4)
    )
);

TEST(Videoio_MJPEG, accuracy_constant)
{
    requireOpenCvMjpeg();

    // MCU-aligned so padded 8x8 tails are not mixed with zeros.
    Mat src(Size(96, 64), CV_8UC3, Scalar(12, 180, 240));
    const Mat decoded = encodeDecodeMjpeg(src, src.size(), true, 75, 1);
    ASSERT_EQ(src.size(), decoded.size());
    EXPECT_GT(cvtest::PSNR(decoded, src), 45.0);
    EXPECT_LE(cv::norm(decoded, src, NORM_INF), 4.0);
}

TEST(Videoio_MJPEG, accuracy_rgba)
{
    requireOpenCvMjpeg();

    Mat src(Size(65, 49), CV_8UC4); // leftover MCU; encoder reads RGB from first 3 bytes
    fillMjpegGradient(src, true);
    const Mat decoded = encodeDecodeMjpeg(src, src.size(), true, 75, 1);
    ASSERT_EQ(CV_8UC3, decoded.type());

    Mat expected;
    cvtColor(src, expected, COLOR_RGBA2BGR);
    EXPECT_GT(cvtest::PSNR(decoded, expected), 30.0);
}

TEST(Videoio_MJPEG, accuracy_yuv444p)
{
    requireOpenCvMjpeg();

    const Size frameSize(37, 25); // leftover MCU on YUV444 16-wide SIMD path
    Mat yuv(frameSize.height * 3, frameSize.width, CV_8UC1);
    Mat y = yuv.rowRange(0, frameSize.height);
    Mat u = yuv.rowRange(frameSize.height, frameSize.height * 2);
    Mat v = yuv.rowRange(frameSize.height * 2, frameSize.height * 3);
    for (int r = 0; r < frameSize.height; r++)
    {
        for (int c = 0; c < frameSize.width; c++)
        {
            y.at<uchar>(r, c) = (uchar)((r * 3 + c * 5) & 255);
            u.at<uchar>(r, c) = (uchar)(80 + (c & 31));
            v.at<uchar>(r, c) = (uchar)(160 + (r & 31));
        }
    }

    const Mat decoded = encodeDecodeMjpeg(yuv, frameSize, true, 95, 1);
    ASSERT_EQ(frameSize, decoded.size());
    ASSERT_EQ(CV_8UC3, decoded.type());

    // Color MJPEG is 4:2:0, so chroma is subsampled. Luma should still track the Y plane.
    Mat decodedY;
    cvtColor(decoded, decodedY, COLOR_BGR2GRAY);
    EXPECT_GT(cvtest::PSNR(decodedY, y), 30.0);
}

TEST(Videoio_MJPEG, accuracy_gray_roundtrip)
{
    requireOpenCvMjpeg();

    Mat src(Size(320, 240), CV_8UC1);
    fillMjpegGradient(src, false);
    // One stripe and several stripes share DC prediction, so both must round-trip.
    for (int nstripes : {1, 4})
    {
        const Mat decoded = encodeDecodeMjpeg(src, src.size(), false, 75, nstripes);
        ASSERT_FALSE(decoded.empty()) << "stripes=" << nstripes;
        ASSERT_EQ(src.size(), decoded.size());

        Mat decodedGray;
        if (decoded.channels() == 3)
            cvtColor(decoded, decodedGray, COLOR_BGR2GRAY);
        else
            decodedGray = decoded;

        // Gray samples are level-shifted like luma, so the libjpeg round-trip
        // matches the color gradient check.
        EXPECT_GT(cvtest::PSNR(decodedGray, src), 30.0) << "stripes=" << nstripes;
    }
}

TEST(Videoio_MJPEG, accuracy_real_image)
{
    requireOpenCvMjpeg();

    Mat src = imread(cvtest::TS::ptr()->get_data_path() + "../cv/shared/lena.png", IMREAD_COLOR);
    if (src.empty())
        throw SkipTestException("testdata cv/shared/lena.png is not available");

    const Size sizes[] = { Size(128, 128), Size(320, 240), Size(513, 257) };
    for (Size sz : sizes)
    {
        Mat image;
        resize(src, image, sz);
        const Mat decoded = encodeDecodeMjpeg(image, sz, true, 75, 1);
        const double psnr = cvtest::PSNR(decoded, image);
        EXPECT_GT(psnr, 30.0) << "lena " << sz << " PSNR=" << psnr;
    }
}

}} // namespace
