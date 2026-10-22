// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <cstring>

namespace opencv_test { namespace {

typedef std::tuple<int, Size, int, int, bool> FlipByteStepParams;
class Core_FlipByteStep : public testing::TestWithParam<FlipByteStepParams> {};

TEST_P(Core_FlipByteStep, accuracy)
{
    const int type = get<0>(GetParam());
    const Size size = get<1>(GetParam());
    const int code = get<2>(GetParam());
    const int layout = get<3>(GetParam());
    const bool inplace = get<4>(GetParam());
    const size_t esz = CV_ELEM_SIZE(type);
    const size_t offset = layout == 0 ? 1 : 0;
    const size_t srcStep = size.width * esz + ((layout & 1) ? 1 : 0);
    const size_t dstStep = size.width * esz + ((layout & 2) ? 1 : 0);
    std::vector<uchar> srcBuffer(offset + srcStep * size.height + 16);
    std::vector<uchar> dstBuffer(offset + dstStep * size.height + 16, 0xa5);
    Mat srcBytes(1, static_cast<int>(srcBuffer.size()), CV_8U, srcBuffer.data());
    theRNG().fill(srcBytes, RNG::UNIFORM, 0, 256);
    const std::vector<uchar> original = srcBuffer;
    std::vector<uchar> expected = inplace ? srcBuffer : dstBuffer;
    const size_t actualStep = inplace ? srcStep : dstStep;

    // Reference each destination pixel directly from the original bytes.
    for (int y = 0; y < size.height; ++y)
        for (int x = 0; x < size.width; ++x)
        {
            const int dy = code <= 0 ? size.height - 1 - y : y;
            const int dx = code != 0 ? size.width - 1 - x : x;
            std::memcpy(expected.data() + offset + dy * actualStep + dx * esz,
                        original.data() + offset + y * srcStep + x * esz, esz);
        }

    Mat src(size, type, srcBuffer.data() + offset, srcStep);
    Mat dst(size, type, inplace ? srcBuffer.data() + offset : dstBuffer.data() + offset, actualStep);
    const uchar* dstData = dst.data;
    cv::flip(src, dst, code);
    ASSERT_EQ(dstData, dst.data);
    ASSERT_EQ(actualStep, dst.step[0]);
    const std::vector<uchar>& actual = inplace ? srcBuffer : dstBuffer;
    EXPECT_EQ(0, std::memcmp(expected.data(), actual.data(), expected.size()));
    if (!inplace)
    {
        EXPECT_EQ(0, std::memcmp(original.data(), srcBuffer.data(), original.size()));
    }
}

INSTANTIATE_TEST_CASE_P(Core, Core_FlipByteStep, testing::Combine(
    testing::Values(CV_8UC2, CV_8UC4, CV_8UC(8), CV_8UC(6), CV_8UC(12), CV_8UC(24)),
    testing::Values(Size(17, 9), Size(257, 7), Size(256, 8), Size(1, 3)),
    testing::Values(-1, 0, 1), testing::Values(0, 1, 2, 3), testing::Bool()));

}} // namespace opencv_test
