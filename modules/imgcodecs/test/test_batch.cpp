// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"

namespace opencv_test { namespace {

// The images are synthesized instead of being read from the test data repository, so that the
// batch tests exercise the codecs even when opencv_extra is not available.
static Mat makeTestImage(Size size, int type, int seed)
{
    Mat img(size, type);
    RNG rng((uint64)seed * 7919 + 13);

    CV_Assert(img.depth() == CV_8U);

    // A smooth gradient plus a few shapes: noise alone compresses badly and hides codec artifacts.
    for (int y = 0; y < img.rows; y++)
    {
        for (int x = 0; x < img.cols; x++)
        {
            const int gx = x * 255 / std::max(1, img.cols - 1);
            const int gy = y * 255 / std::max(1, img.rows - 1);
            const int v = (gx + gy) / 2;
            switch (img.channels())
            {
                case 1: img.at<uchar>(y, x) = (uchar)v; break;
                case 3: img.at<Vec3b>(y, x) = Vec3b((uchar)v, (uchar)((v + 85) % 256), (uchar)((v + 170) % 256)); break;
                // Alpha stays >= 1: libwebp is free to discard the colour of fully transparent
                // pixels, which would make a lossless round trip look lossy for the wrong reason.
                case 4: img.at<Vec4b>(y, x) = Vec4b((uchar)v, (uchar)((v + 85) % 256), (uchar)((v + 170) % 256),
                                                    (uchar)std::max(1, gx)); break;
                default: CV_Assert(false);
            }
        }
    }

    for (int i = 0; i < 4; i++)
    {
        const Point c(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
        const int r = rng.uniform(3, std::max(4, std::min(img.cols, img.rows) / 4));
        circle(img, c, r, Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256), 255), -1);
    }

    return img;
}

static std::vector<Mat> makeBatch(size_t n, int type, Size size = Size(64, 48))
{
    std::vector<Mat> batch;
    for (size_t i = 0; i < n; i++)
        batch.push_back(makeTestImage(size, type, (int)i));
    return batch;
}

// ---------------------------------------------------------------------------------------------
// Determinism: the batch call must produce exactly the bytes a sequential imencode() loop does.
// ---------------------------------------------------------------------------------------------

static void checkEncodeMatchesLoop(const String& ext, int type, const std::vector<int>& params)
{
    const std::vector<Mat> images = makeBatch(6, type);

    std::vector<std::vector<uchar> > batched;
    ASSERT_TRUE(imencodeBatch(ext, images, batched, params)) << ext;
    ASSERT_EQ(images.size(), batched.size());

    for (size_t i = 0; i < images.size(); i++)
    {
        std::vector<uchar> single;
        ASSERT_TRUE(imencode(ext, images[i], single, params)) << ext << " #" << i;
        EXPECT_EQ(single, batched[i]) << ext << " #" << i;
    }
}

static void checkDecodeMatchesLoop(const String& ext, int type, int flags, const std::vector<int>& params)
{
    const std::vector<Mat> images = makeBatch(6, type);

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(ext, images, buffers, params)) << ext;

    std::vector<Mat> batched;
    ASSERT_TRUE(imdecodeBatch(buffers, flags, batched)) << ext;
    ASSERT_EQ(buffers.size(), batched.size());

    for (size_t i = 0; i < buffers.size(); i++)
    {
        const Mat single = imdecode(buffers[i], flags);
        ASSERT_FALSE(single.empty()) << ext << " #" << i;
        EXPECT_EQ(single.size(), batched[i].size()) << ext << " #" << i;
        EXPECT_EQ(single.type(), batched[i].type()) << ext << " #" << i;
        EXPECT_EQ(0, cv::norm(single, batched[i], NORM_INF)) << ext << " #" << i;
    }
}

#ifdef HAVE_PNG
TEST(Imgcodecs_Batch, png_encode_matches_sequential_calls)
{
    checkEncodeMatchesLoop(".png", CV_8UC1, std::vector<int>());
    checkEncodeMatchesLoop(".png", CV_8UC3, std::vector<int>());
    checkEncodeMatchesLoop(".png", CV_8UC4, std::vector<int>());
}

TEST(Imgcodecs_Batch, png_decode_matches_sequential_calls)
{
    checkDecodeMatchesLoop(".png", CV_8UC3, IMREAD_COLOR, std::vector<int>());
    checkDecodeMatchesLoop(".png", CV_8UC4, IMREAD_UNCHANGED, std::vector<int>());
}

// PNG is lossless, so the round trip must be bit exact.
TEST(Imgcodecs_Batch, png_roundtrip_is_lossless)
{
    const std::vector<Mat> images = makeBatch(5, CV_8UC3);

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(".png", images, buffers, std::vector<int>()));

    std::vector<Mat> decoded;
    ASSERT_TRUE(imdecodeBatch(buffers, IMREAD_COLOR, decoded));
    ASSERT_EQ(images.size(), decoded.size());

    for (size_t i = 0; i < images.size(); i++)
        EXPECT_EQ(0, cv::norm(images[i], decoded[i], NORM_INF)) << "#" << i;
}

// Independent buffers do not require homogeneous images.
TEST(Imgcodecs_Batch, mixed_sizes_and_channels)
{
    std::vector<Mat> images;
    images.push_back(makeTestImage(Size(17, 5), CV_8UC3, 0));
    images.push_back(makeTestImage(Size(64, 64), CV_8UC1, 1));
    images.push_back(makeTestImage(Size(128, 33), CV_8UC4, 2));
    images.push_back(makeTestImage(Size(1, 1), CV_8UC3, 3));

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(".png", images, buffers, std::vector<int>()));
    ASSERT_EQ(images.size(), buffers.size());

    std::vector<Mat> decoded;
    ASSERT_TRUE(imdecodeBatch(buffers, IMREAD_UNCHANGED, decoded));
    ASSERT_EQ(images.size(), decoded.size());

    for (size_t i = 0; i < images.size(); i++)
    {
        EXPECT_EQ(images[i].size(), decoded[i].size()) << "#" << i;
        EXPECT_EQ(images[i].type(), decoded[i].type()) << "#" << i;
        EXPECT_EQ(0, cv::norm(images[i], decoded[i], NORM_INF)) << "#" << i;
    }
}

// The output must not depend on how many threads OpenCV happens to be using.
TEST(Imgcodecs_Batch, output_is_independent_of_thread_count)
{
    const std::vector<Mat> images = makeBatch(8, CV_8UC3);
    const int saved_threads = getNumThreads();

    std::vector<std::vector<uchar> > single_threaded;
    setNumThreads(1);
    ASSERT_TRUE(imencodeBatch(".png", images, single_threaded, std::vector<int>()));

    std::vector<std::vector<uchar> > multi_threaded;
    setNumThreads(saved_threads);
    ASSERT_TRUE(imencodeBatch(".png", images, multi_threaded, std::vector<int>()));

    EXPECT_EQ(single_threaded, multi_threaded);
}
#endif // HAVE_PNG

#ifdef HAVE_JPEG
TEST(Imgcodecs_Batch, jpeg_encode_matches_sequential_calls)
{
    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
    params.push_back(90);

    checkEncodeMatchesLoop(".jpg", CV_8UC1, params);
    checkEncodeMatchesLoop(".jpg", CV_8UC3, params);
}

TEST(Imgcodecs_Batch, jpeg_decode_matches_sequential_calls)
{
    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
    params.push_back(90);

    checkDecodeMatchesLoop(".jpg", CV_8UC3, IMREAD_COLOR, params);
    checkDecodeMatchesLoop(".jpg", CV_8UC1, IMREAD_GRAYSCALE, params);
}

// Quality is shared by the whole batch, and it must actually reach every image.
TEST(Imgcodecs_Batch, jpeg_quality_applies_to_every_image)
{
    const std::vector<Mat> images = makeBatch(4, CV_8UC3, Size(128, 128));

    std::vector<int> low, high;
    low.push_back(IMWRITE_JPEG_QUALITY);  low.push_back(20);
    high.push_back(IMWRITE_JPEG_QUALITY); high.push_back(95);

    std::vector<std::vector<uchar> > low_buffers, high_buffers;
    ASSERT_TRUE(imencodeBatch(".jpg", images, low_buffers, low));
    ASSERT_TRUE(imencodeBatch(".jpg", images, high_buffers, high));

    for (size_t i = 0; i < images.size(); i++)
        EXPECT_LT(low_buffers[i].size(), high_buffers[i].size()) << "#" << i;
}

// Lossy round trip: close to the source, not identical to it.
TEST(Imgcodecs_Batch, jpeg_roundtrip_is_close)
{
    const std::vector<Mat> images = makeBatch(4, CV_8UC3, Size(96, 96));

    std::vector<int> params;
    params.push_back(IMWRITE_JPEG_QUALITY);
    params.push_back(95);

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(".jpg", images, buffers, params));

    std::vector<Mat> decoded;
    ASSERT_TRUE(imdecodeBatch(buffers, IMREAD_COLOR, decoded));
    ASSERT_EQ(images.size(), decoded.size());

    // The synthetic images are close to a worst case for JPEG: hard-edged shapes on top of a
    // gradient that wraps around per channel. A photo lands near 0.015 at this quality, these
    // land near 0.08, so the bound only asserts that the round trip stays in the right ballpark.
    // Bit exactness against a sequential imencode/imdecode pair is covered separately.
    for (size_t i = 0; i < images.size(); i++)
    {
        ASSERT_EQ(images[i].size(), decoded[i].size()) << "#" << i;
        EXPECT_LT(cvtest::norm(images[i], decoded[i], NORM_L2 | NORM_RELATIVE), 0.1) << "#" << i;
    }
}
#endif // HAVE_JPEG

#ifdef HAVE_WEBP
TEST(Imgcodecs_Batch, webp_encode_matches_sequential_calls)
{
    std::vector<int> params;
    params.push_back(IMWRITE_WEBP_QUALITY);
    params.push_back(90);

    checkEncodeMatchesLoop(".webp", CV_8UC3, params);
    checkEncodeMatchesLoop(".webp", CV_8UC4, params);
}

TEST(Imgcodecs_Batch, webp_decode_matches_sequential_calls)
{
    std::vector<int> params;
    params.push_back(IMWRITE_WEBP_QUALITY);
    params.push_back(90);

    checkDecodeMatchesLoop(".webp", CV_8UC3, IMREAD_COLOR, params);
    checkDecodeMatchesLoop(".webp", CV_8UC4, IMREAD_UNCHANGED, params);
}

// Lossless WebP (quality > 100) must survive the round trip untouched, alpha included.
TEST(Imgcodecs_Batch, webp_lossless_roundtrip_preserves_alpha)
{
    const std::vector<Mat> images = makeBatch(4, CV_8UC4);

    std::vector<int> params;
    params.push_back(IMWRITE_WEBP_QUALITY);
    params.push_back(101);

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(".webp", images, buffers, params));

    std::vector<Mat> decoded;
    ASSERT_TRUE(imdecodeBatch(buffers, IMREAD_UNCHANGED, decoded));
    ASSERT_EQ(images.size(), decoded.size());

    for (size_t i = 0; i < images.size(); i++)
    {
        ASSERT_EQ(4, decoded[i].channels()) << "#" << i;
        EXPECT_EQ(0, cv::norm(images[i], decoded[i], NORM_INF)) << "#" << i;
    }
}
#endif // HAVE_WEBP

// ---------------------------------------------------------------------------------------------
// Contract: empty batches, failure reporting and argument validation.
// ---------------------------------------------------------------------------------------------

TEST(Imgcodecs_Batch, empty_batch_succeeds)
{
    std::vector<std::vector<uchar> > buffers(3);  // pre-filled to prove the output is cleared
    EXPECT_TRUE(imencodeBatch(".png", std::vector<Mat>(), buffers, std::vector<int>()));
    EXPECT_TRUE(buffers.empty());

    std::vector<Mat> images(3);
    EXPECT_TRUE(imdecodeBatch(std::vector<std::vector<uchar> >(), IMREAD_COLOR, images));
    EXPECT_TRUE(images.empty());
}

TEST(Imgcodecs_Batch, unknown_extension_throws)
{
    const std::vector<Mat> images = makeBatch(2, CV_8UC3);
    std::vector<std::vector<uchar> > buffers;
    EXPECT_THROW(imencodeBatch(".notaformat", images, buffers, std::vector<int>()), cv::Exception);
}

TEST(Imgcodecs_Batch, malformed_params_throw)
{
    const std::vector<Mat> images = makeBatch(2, CV_8UC3);
    std::vector<std::vector<uchar> > buffers;
    // An odd number of entries cannot be read as key-value pairs.
    EXPECT_THROW(imencodeBatch(".png", images, buffers, std::vector<int>(1, IMWRITE_PNG_COMPRESSION)),
                 cv::Exception);
}

#ifdef HAVE_PNG
// A failing image must not shorten the result nor take its neighbours down with it.
TEST(Imgcodecs_Batch, encode_failure_is_isolated_to_its_index)
{
    std::vector<Mat> images = makeBatch(4, CV_8UC3);
    images[2] = Mat(8, 8, CV_8UC2, Scalar::all(0));  // no encoder accepts two channels

    std::vector<std::vector<uchar> > buffers;
    EXPECT_FALSE(imencodeBatch(".png", images, buffers, std::vector<int>()));
    ASSERT_EQ(images.size(), buffers.size()) << "the result must not be shortened";

    EXPECT_TRUE(buffers[2].empty());
    for (size_t i = 0; i < buffers.size(); i++)
    {
        if (i == 2)
            continue;
        EXPECT_FALSE(buffers[i].empty()) << "#" << i;

        std::vector<uchar> expected;
        ASSERT_TRUE(imencode(".png", images[i], expected, std::vector<int>()));
        EXPECT_EQ(expected, buffers[i]) << "#" << i;
    }
}

TEST(Imgcodecs_Batch, decode_failure_is_isolated_to_its_index)
{
    const std::vector<Mat> images = makeBatch(4, CV_8UC3);

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(".png", images, buffers, std::vector<int>()));

    buffers[1] = std::vector<uchar>(64, 0x7f);  // garbage: no decoder recognizes the signature
    buffers[3].clear();                         // empty buffer

    std::vector<Mat> decoded;
    EXPECT_FALSE(imdecodeBatch(buffers, IMREAD_COLOR, decoded));
    ASSERT_EQ(buffers.size(), decoded.size()) << "the result must not be shortened";

    EXPECT_TRUE(decoded[1].empty());
    EXPECT_TRUE(decoded[3].empty());
    EXPECT_EQ(0, cv::norm(images[0], decoded[0], NORM_INF));
    EXPECT_EQ(0, cv::norm(images[2], decoded[2], NORM_INF));
}

// The C++ contract in the issue passes buffers as std::vector<std::vector<uchar>>; the same call
// must also accept the std::vector<Mat> that other OpenCV entry points hand out.
TEST(Imgcodecs_Batch, decode_accepts_vector_of_mats)
{
    const std::vector<Mat> images = makeBatch(3, CV_8UC3);

    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(".png", images, buffers, std::vector<int>()));

    std::vector<Mat> buffer_mats;
    for (size_t i = 0; i < buffers.size(); i++)
        buffer_mats.push_back(Mat(buffers[i], /*copyData*/ true));

    std::vector<Mat> decoded;
    ASSERT_TRUE(imdecodeBatch(buffer_mats, IMREAD_COLOR, decoded));
    ASSERT_EQ(images.size(), decoded.size());

    for (size_t i = 0; i < images.size(); i++)
        EXPECT_EQ(0, cv::norm(images[i], decoded[i], NORM_INF)) << "#" << i;
}
#endif // HAVE_PNG

}} // namespace
