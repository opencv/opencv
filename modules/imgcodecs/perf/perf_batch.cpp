// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html
#include "perf_precomp.hpp"

namespace opencv_test
{

using namespace perf;

// Compares cv::imencodeBatch / cv::imdecodeBatch against the sequential imencode / imdecode loop
// they are meant to replace. Both variants of an operation take the same parameters, so the pairs
// can be read straight off the report:
//
//   imencodeBatch / batch_size_16_1024x1024_jpg_q90   vs   imencodeLoop / <same suffix>
//
// The images are synthesized so that the benchmark does not depend on the opencv_extra data repo.

static Mat makeBenchImage(Size size, int seed)
{
    Mat img(size, CV_8UC3);
    RNG rng((uint64)seed * 7919 + 13);

    for (int y = 0; y < img.rows; y++)
    {
        Vec3b* row = img.ptr<Vec3b>(y);
        const int gy = y * 255 / std::max(1, img.rows - 1);
        for (int x = 0; x < img.cols; x++)
        {
            const int gx = x * 255 / std::max(1, img.cols - 1);
            const int v = (gx + gy) / 2;
            row[x] = Vec3b((uchar)v, (uchar)((v + 85) % 256), (uchar)((v + 170) % 256));
        }
    }

    // Some structure on top of the gradient, otherwise the codecs see an unrealistically easy image.
    for (int i = 0; i < 24; i++)
    {
        const Point c(rng.uniform(0, img.cols), rng.uniform(0, img.rows));
        const int r = rng.uniform(3, std::max(4, std::min(img.cols, img.rows) / 6));
        circle(img, c, r, Scalar(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256)), -1);
    }

    return img;
}

static std::vector<Mat> makeBenchBatch(Size size, int batch_size)
{
    std::vector<Mat> images;
    for (int i = 0; i < batch_size; i++)
        images.push_back(makeBenchImage(size, i));
    return images;
}

static std::vector<int> qualityParams(const String& ext, int quality)
{
    std::vector<int> params;
    if (ext == ".jpg")
        params.push_back(IMWRITE_JPEG_QUALITY);
    else if (ext == ".webp")
        params.push_back(IMWRITE_WEBP_QUALITY);
    else
        return params;
    params.push_back(quality);
    return params;
}

// size, batch size, extension, quality
typedef TestBaseWithParam< tuple<Size, int, String, int> > Imgcodecs_Batch;

#define BATCH_PERF_PARAMS testing::Combine(                                        \
    testing::Values(Size(256, 256), Size(512, 512), Size(1024, 1024)),             \
    testing::Values(2, 4, 8, 16),                                                  \
    testing::Values(String(".jpg"), String(".webp")),                              \
    testing::Values(50, 90))

static bool haveCodecFor(const String& ext)
{
#ifndef HAVE_JPEG
    if (ext == ".jpg")
        return false;
#endif
#ifndef HAVE_WEBP
    if (ext == ".webp")
        return false;
#endif
    return haveImageWriter("test" + ext);
}

PERF_TEST_P(Imgcodecs_Batch, imencodeBatch, BATCH_PERF_PARAMS)
{
    const Size size = get<0>(GetParam());
    const int batch_size = get<1>(GetParam());
    const String ext = get<2>(GetParam());
    const int quality = get<3>(GetParam());

    if (!haveCodecFor(ext))
        throw SkipTestException("Codec is not available: " + ext);

    const std::vector<Mat> images = makeBenchBatch(size, batch_size);
    const std::vector<int> params = qualityParams(ext, quality);
    std::vector<std::vector<uchar> > buffers;

    TEST_CYCLE() ASSERT_TRUE(imencodeBatch(ext, images, buffers, params));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Imgcodecs_Batch, imencodeLoop, BATCH_PERF_PARAMS)
{
    const Size size = get<0>(GetParam());
    const int batch_size = get<1>(GetParam());
    const String ext = get<2>(GetParam());
    const int quality = get<3>(GetParam());

    if (!haveCodecFor(ext))
        throw SkipTestException("Codec is not available: " + ext);

    const std::vector<Mat> images = makeBenchBatch(size, batch_size);
    const std::vector<int> params = qualityParams(ext, quality);
    std::vector<std::vector<uchar> > buffers(images.size());

    TEST_CYCLE()
    {
        for (size_t i = 0; i < images.size(); i++)
            ASSERT_TRUE(imencode(ext, images[i], buffers[i], params));
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Imgcodecs_Batch, imdecodeBatch, BATCH_PERF_PARAMS)
{
    const Size size = get<0>(GetParam());
    const int batch_size = get<1>(GetParam());
    const String ext = get<2>(GetParam());
    const int quality = get<3>(GetParam());

    if (!haveCodecFor(ext))
        throw SkipTestException("Codec is not available: " + ext);

    const std::vector<Mat> images = makeBenchBatch(size, batch_size);
    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(ext, images, buffers, qualityParams(ext, quality)));

    std::vector<Mat> decoded;

    TEST_CYCLE() ASSERT_TRUE(imdecodeBatch(buffers, IMREAD_COLOR_BGR, decoded));

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P(Imgcodecs_Batch, imdecodeLoop, BATCH_PERF_PARAMS)
{
    const Size size = get<0>(GetParam());
    const int batch_size = get<1>(GetParam());
    const String ext = get<2>(GetParam());
    const int quality = get<3>(GetParam());

    if (!haveCodecFor(ext))
        throw SkipTestException("Codec is not available: " + ext);

    const std::vector<Mat> images = makeBenchBatch(size, batch_size);
    std::vector<std::vector<uchar> > buffers;
    ASSERT_TRUE(imencodeBatch(ext, images, buffers, qualityParams(ext, quality)));

    std::vector<Mat> decoded(buffers.size());

    TEST_CYCLE()
    {
        for (size_t i = 0; i < buffers.size(); i++)
        {
            decoded[i] = imdecode(buffers[i], IMREAD_COLOR_BGR);
            ASSERT_FALSE(decoded[i].empty());
        }
    }

    SANITY_CHECK_NOTHING();
}

} // namespace
