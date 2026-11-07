// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <cmath>
#include <limits>

namespace opencv_test { namespace {

typedef testing::TestWithParam<tuple<int, int, bool> > ResizeHalf;

static void expectHalfReference(const Mat& actual, const Mat& reference)
{
    ASSERT_EQ(reference.size(), actual.size());
    ASSERT_EQ(reference.type(), actual.type());
    Mat a, b;
    actual.convertTo(a, CV_32F);
    reference.convertTo(b, CV_32F);
    for (int y = 0; y < a.rows; ++y)
        for (int x = 0; x < a.cols * a.channels(); ++x)
        {
            const float av = a.ptr<float>(y)[x], bv = b.ptr<float>(y)[x];
            if (std::isnan(bv))
                ASSERT_TRUE(std::isnan(av)) << y << ", " << x;
            else if (std::isinf(bv))
                ASSERT_EQ(bv, av) << y << ", " << x;
            else
                // Both paths accumulate in float. Different vector contraction/order
                // may round an answer to the adjacent half, especially near zero.
                ASSERT_NEAR(bv, av, std::max(2e-5f, std::abs(bv) / 512.f))
                    << y << ", " << x;
        }
}

TEST_P(ResizeHalf, matchesFloatWorkingImage)
{
    const int mode = get<0>(GetParam()), cn = get<1>(GetParam());
    const bool roi = get<2>(GetParam());
    const Size inputs[] = {Size(1, 1), Size(1, 7), Size(7, 1), Size(17, 13),
                           Size(64, 48), Size(65, 49)};
    const Size outputs[] = {Size(1, 1), Size(2, 3), Size(9, 7), Size(34, 26)};
    RNG rng(0x30216);
    for (const Size& size : inputs)
    {
        Mat storage(size.height + 2, size.width + 4, CV_MAKETYPE(CV_16F, cn));
        Mat src = roi ? storage(Rect(1, 1, size.width, size.height))
                      : Mat(size, CV_MAKETYPE(CV_16F, cn));
        Mat random(size, CV_MAKETYPE(CV_32F, cn));
        rng.fill(random, RNG::UNIFORM, -1.f, 1.f);
        random.convertTo(src, CV_16F);
        Mat working;
        src.convertTo(working, CV_32F);
        for (const Size& output : outputs)
        {
            SCOPED_TRACE(testing::Message() << size << " -> " << output);
            Mat ref32, ref, dst;
            try
            {
                resize(working, ref32, output, 0, 0, mode);
            }
            catch (const cv::Exception&)
            {
                // Generic INTER_AREA retains the existing four-channel restriction.
                // Let the float path decide which scale factors take that path.
                if (mode != INTER_AREA || cn <= 4)
                    throw;
                EXPECT_THROW(resize(src, dst, output, 0, 0, mode), cv::Exception);
                continue;
            }
            ref32.convertTo(ref, CV_16F);
            Mat dstStorage(output.height + 2, output.width + 4, src.type());
            dst = roi ? dstStorage(Rect(1, 1, output.width, output.height))
                      : Mat(output, src.type());
            uchar* destination = dst.data;
            resize(src, dst, output, 0, 0, mode);
            EXPECT_EQ(destination, dst.data);
            expectHalfReference(dst, ref);
        }
    }
}

INSTANTIATE_TEST_CASE_P(Imgproc, ResizeHalf, testing::Combine(
    testing::Values(INTER_NEAREST, INTER_NEAREST_EXACT, INTER_LINEAR,
                    INTER_LINEAR_EXACT, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA),
    testing::Values(1, 2, 3, 4, 5, 9), testing::Bool()));

TEST(Imgproc_ResizeHalf, nearestPreservesAllBitPatterns)
{
    Mat src(256, 256, CV_16F);
    for (int i = 0; i < 65536; ++i)
        reinterpret_cast<ushort*>(src.data)[i] = static_cast<ushort>(i);
    for (int mode : {INTER_NEAREST, INTER_NEAREST_EXACT})
    {
        Mat dst;
        resize(src, dst, Size(512, 512), 0, 0, mode);
        for (int y = 0; y < dst.rows; ++y)
            for (int x = 0; x < dst.cols; ++x)
                ASSERT_EQ(reinterpret_cast<const ushort*>(src.ptr(y / 2))[x / 2],
                          reinterpret_cast<const ushort*>(dst.ptr(y))[x]);
    }
}

TEST(Imgproc_ResizeHalf, inplaceAndExplicitScales)
{
    Mat values(33, 47, CV_32FC3);
    randu(values, -1.f, 1.f);
    Mat src;
    values.convertTo(src, CV_16F);
    for (int mode : {INTER_LINEAR, INTER_LINEAR_EXACT, INTER_CUBIC,
                    INTER_LANCZOS4, INTER_AREA})
    {
        Mat ref32, ref, working;
        src.convertTo(working, CV_32F);
        resize(working, ref32, Size(), 0.7, 1.3, mode);
        ref32.convertTo(ref, CV_16F);
        Mat inplace = src.clone();
        resize(inplace, inplace, Size(), 0.7, 1.3, mode);
        expectHalfReference(inplace, ref);
    }
}

TEST(Imgproc_ResizeHalf, floatAccumulation)
{
    // A half accumulator would overflow before the average is taken.
    Mat src32(8, 8, CV_32F, Scalar(60000));
    Mat src, dst, actual;
    src32.convertTo(src, CV_16F);
    for (int mode : {INTER_LINEAR, INTER_LINEAR_EXACT, INTER_AREA})
    {
        resize(src, dst, Size(2, 2), 0, 0, mode);
        dst.convertTo(actual, CV_32F);
        EXPECT_EQ(0, cv::norm(actual, Mat(2, 2, CV_32F, Scalar(60000)), NORM_INF));
    }
}

TEST(Imgproc_ResizeHalf, integerDecimationAndPartialEdges)
{
    RNG rng(30216);
    for (int cn : {1, 2, 3, 4, 5, 9})
        for (int width : {3, 7, 8, 15, 16, 31, 32, 65})
            for (int height : {3, 4, 9, 16})
            {
                Mat values(height, width, CV_MAKETYPE(CV_32F, cn));
                rng.fill(values, RNG::UNIFORM, -100.f, 100.f);
                Mat storage(height + 2, width + 2, CV_MAKETYPE(CV_16F, cn));
                Mat src = storage(Rect(1, 1, width, height)), working;
                values.convertTo(src, CV_16F);
                src.convertTo(working, CV_32F);
                for (const Size scale : {Size(2, 2), Size(3, 3), Size(4, 2),
                                         Size(1, 4), Size(4, 1)})
                {
                    SCOPED_TRACE(testing::Message() << src.size() << " C" << cn
                                 << " / " << scale);
                    Mat ref32, ref, dst;
                    resize(working, ref32, Size(), 1. / scale.width,
                           1. / scale.height, INTER_AREA);
                    ref32.convertTo(ref, CV_16F);
                    // A strided destination with guard pixels on every side.
                    Mat dstStorage(ref.rows + 2, ref.cols + 2, src.type());
                    std::fill(dstStorage.ptr<hfloat>(),
                              dstStorage.ptr<hfloat>() + dstStorage.total() * cn,
                              hfloat(42.f));
                    dst = dstStorage(Rect(1, 1, ref.cols, ref.rows));
                    resize(src, dst, Size(), 1. / scale.width,
                           1. / scale.height, INTER_AREA);
                    expectHalfReference(dst, ref);
                    for (int y = 0; y < dstStorage.rows; ++y)
                        for (int x = 0; x < dstStorage.cols; ++x)
                            if (y == 0 || y == dstStorage.rows - 1 ||
                                x == 0 || x == dstStorage.cols - 1)
                                for (int c = 0; c < cn; ++c)
                                    ASSERT_EQ(42.f, static_cast<float>(
                                        dstStorage.ptr<hfloat>(y)[x * cn + c]));
                }
            }
}

TEST(Imgproc_ResizeHalf, subnormalValues)
{
    Mat src(16, 32, CV_16F);
    for (int i = 0; i < 512; ++i)
        src.ptr<hfloat>()[i] = cv::hfloatFromBits(static_cast<ushort>(i * 2));
    Mat working;
    src.convertTo(working, CV_32F);
    for (int mode : {INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA})
        for (Size size : {Size(16, 8), Size(11, 7), Size(49, 27)})
        {
            Mat dst, ref32, ref, actual32, expected32;
            resize(src, dst, size, 0, 0, mode);
            resize(working, ref32, size, 0, 0, mode);
            ref32.convertTo(ref, CV_16F);
            dst.convertTo(actual32, CV_32F);
            ref.convertTo(expected32, CV_32F);
            // One binary16 subnormal unit; in particular, do not flush to zero.
            EXPECT_LE(cv::norm(actual32, expected32, NORM_INF), std::ldexp(1., -24));
        }
}

TEST(Imgproc_ResizeHalf, largeOutputs)
{
    // Enough output pixels to exercise per-stripe row buffers in parallel builds.
    Mat values(257, 513, CV_32FC3);
    randu(values, -1.f, 1.f);
    Mat src, working;
    values.convertTo(src, CV_16F);
    src.convertTo(working, CV_32F);
    for (int mode : {INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA})
    {
        Mat ref32, ref, dst;
        resize(working, ref32, Size(769, 515), 0, 0, mode);
        ref32.convertTo(ref, CV_16F);
        resize(src, dst, ref.size(), 0, 0, mode);
        expectHalfReference(dst, ref);
    }
}

TEST(Imgproc_ResizeHalf, nonFiniteValues)
{
    Mat src32(9, 17, CV_32F, Scalar(1));
    src32.at<float>(3, 5) = std::numeric_limits<float>::infinity();
    src32.at<float>(5, 7) = -std::numeric_limits<float>::infinity();
    src32.at<float>(6, 11) = std::numeric_limits<float>::quiet_NaN();
    Mat src;
    src32.convertTo(src, CV_16F);
    for (int mode : {INTER_LINEAR, INTER_CUBIC, INTER_LANCZOS4, INTER_AREA})
    {
        Mat ref32, ref, dst;
        resize(src32, ref32, Size(13, 7), 0, 0, mode);
        ref32.convertTo(ref, CV_16F);
        resize(src, dst, Size(13, 7), 0, 0, mode);
        expectHalfReference(dst, ref);
    }
}

}} // namespace
