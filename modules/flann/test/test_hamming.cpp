// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include "opencv2/flann/dist.h"

namespace opencv_test { namespace {

TEST(Flann_Hamming, tail_bounds_and_correctness)
{
    // Regression test for issue #30940:
    // When size is not a multiple of 16, Hamming distance must not read past the vector
    // boundaries into subsequent memory.
    const size_t maxLen = 65;
    const size_t bufSize = maxLen + 32;

    std::vector<unsigned char> a(bufSize);
    std::vector<unsigned char> b(bufSize);

    for (size_t len = 1; len <= maxLen; ++len)
    {
        for (size_t i = 0; i < bufSize; ++i)
        {
            a[i] = static_cast<unsigned char>(i * 17 + 1);
            // Agree for i < len, invert for i >= len
            b[i] = (i < len) ? a[i] : static_cast<unsigned char>(~a[i]);
        }

        int dist = cvflann::Hamming<unsigned char>()(a.data(), b.data(), len);
        EXPECT_EQ(0, dist) << "Hamming distance over " << len
                           << " bytes must be 0 when vectors match within range";
    }
}

TEST(Flann_Hamming, zero_iterator_tail_bounds)
{
    const size_t maxLen = 65;
    const size_t bufSize = maxLen + 32;

    std::vector<unsigned char> a(bufSize);

    for (size_t len = 1; len <= maxLen; ++len)
    {
        for (size_t i = 0; i < bufSize; ++i)
        {
            // Zero within range, non-zero beyond range
            a[i] = (i < len) ? 0 : 0xFF;
        }

        int dist = cvflann::Hamming<unsigned char>()(a.data(), cvflann::ZeroIterator<unsigned char>(), len);
        EXPECT_EQ(0, dist) << "Distance to ZeroIterator over " << len
                           << " bytes must be 0 when vectors are zero within range";
    }
}

TEST(Flann_Hamming, matches_norm_hamming)
{
    RNG& rng = theRNG();
    const size_t testSizes[] = {1, 2, 7, 8, 15, 16, 17, 31, 32, 48, 61, 64, 65, 127, 128};

    for (size_t len : testSizes)
    {
        std::vector<unsigned char> a(len);
        std::vector<unsigned char> b(len);
        for (size_t i = 0; i < len; ++i)
        {
            a[i] = static_cast<unsigned char>(rng.uniform(0, 256));
            b[i] = static_cast<unsigned char>(rng.uniform(0, 256));
        }

        int dist = cvflann::Hamming<unsigned char>()(a.data(), b.data(), len);
        int lutDist = cvflann::HammingLUT()(a.data(), b.data(), len);
        int expectedNorm = static_cast<int>(cv::norm(Mat(1, static_cast<int>(len), CV_8U, a.data()),
                                                     Mat(1, static_cast<int>(len), CV_8U, b.data()),
                                                     NORM_HAMMING));

        EXPECT_EQ(expectedNorm, dist) << "Distance mismatch against cv::norm for length " << len;
        EXPECT_EQ(lutDist, dist) << "Distance mismatch against HammingLUT for length " << len;

        int zeroDist = cvflann::Hamming<unsigned char>()(a.data(), cvflann::ZeroIterator<unsigned char>(), len);
        int expectedZeroNorm = static_cast<int>(cv::norm(Mat(1, static_cast<int>(len), CV_8U, a.data()), NORM_HAMMING));
        EXPECT_EQ(expectedZeroNorm, zeroDist) << "ZeroIterator distance mismatch against cv::norm for length " << len;
    }
}

}} // namespace opencv_test
