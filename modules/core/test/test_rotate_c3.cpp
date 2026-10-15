// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include "test_precomp.hpp"
#include <cstring>
#include <cstdint>

namespace opencv_test { namespace {

typedef std::tuple<int, Size, int, bool> RotateC3Params;
class Core_RotateC3 : public testing::TestWithParam<RotateC3Params> {};

static void expectRotateBytes(const Mat& expected, const Mat& actual)
{
    ASSERT_EQ(expected.size(), actual.size());
    ASSERT_EQ(expected.type(), actual.type());
    for (int y = 0; y < expected.rows; ++y)
        ASSERT_EQ(0, std::memcmp(expected.ptr(y), actual.ptr(y),
                                 expected.cols * expected.elemSize())) << "row " << y;
}

TEST_P(Core_RotateC3, BitExactAndPaddedRows)
{
    const int type = CV_MAKETYPE(get<0>(GetParam()), 3);
    const Size size = get<1>(GetParam());
    const int code = get<2>(GetParam());
    const bool roi = get<3>(GetParam());
    const Rect srcArea = roi ? Rect(1, 1, size.width, size.height) : Rect(Point(), size);
    Mat srcParent(roi ? size + Size(5, 3) : size, type);
    // Rotation is a bit permutation. Cover NaN payloads and signed zeros too.
    for (int y = 0; y < srcParent.rows; ++y)
        theRNG().fill(Mat(1, static_cast<int>(srcParent.cols * srcParent.elemSize()),
                          CV_8U, srcParent.ptr(y)), RNG::UNIFORM, 0, 256);
    Mat src = srcParent(srcArea);
    if (src.depth() == CV_16F)
    {
        const uint16_t bits[] = { 0x8000, 0x7e01, 0x7c01 };
        std::memcpy(src.ptr(), bits, sizeof(bits));
    }
    else if (src.depth() == CV_32F)
    {
        const uint32_t bits[] = { 0x80000000U, 0x7fc01234U, 0x7f800001U };
        std::memcpy(src.ptr(), bits, sizeof(bits));
    }
    else if (src.depth() == CV_64F)
    {
        const uint64_t bits[] = { 0x8000000000000000ULL, 0x7ff8000000001234ULL,
                                  0x7ff0000000000001ULL };
        std::memcpy(src.ptr(), bits, sizeof(bits));
    }
    const Mat originalSrcParent = srcParent.clone();
    const Size dstSize = code == ROTATE_180 ? size : Size(size.height, size.width);
    Mat expected(dstSize, type);
    const size_t esz = src.elemSize();
    for (int y = 0; y < src.rows; ++y)
        for (int x = 0; x < src.cols; ++x)
        {
            int dx, dy;
            if (code == ROTATE_90_CLOCKWISE)
            {
                dx = src.rows - 1 - y;
                dy = x;
            }
            else if (code == ROTATE_180)
            {
                dx = src.cols - 1 - x;
                dy = src.rows - 1 - y;
            }
            else
            {
                dx = y;
                dy = src.cols - 1 - x;
            }
            std::memcpy(expected.ptr(dy) + dx * esz, src.ptr(y) + x * esz, esz);
        }

    const Rect dstArea = roi ? Rect(1, 1, dstSize.width, dstSize.height) : Rect(Point(), dstSize);
    Mat dstParent(roi ? dstSize + Size(7, 4) : dstSize, type, Scalar::all(0));
    Mat expectedParent = dstParent.clone();
    expected.copyTo(expectedParent(dstArea));
    Mat actual = dstParent(dstArea);
    rotate(src, actual, code);
    expectRotateBytes(expectedParent, dstParent); // Check padding, not just the ROI.
    expectRotateBytes(originalSrcParent, srcParent);

    Mat inPlaceParent = srcParent.clone();
    Mat inPlace = inPlaceParent(srcArea);
    rotate(inPlace, inPlace, code);
    expectRotateBytes(expected, inPlace);
    if (code == ROTATE_180 || size.width == size.height)
    {
        Mat expectedInPlaceParent = srcParent.clone();
        expected.copyTo(expectedInPlaceParent(srcArea));
        expectRotateBytes(expectedInPlaceParent, inPlaceParent);
    }
    else
        expectRotateBytes(originalSrcParent, inPlaceParent);
}

INSTANTIATE_TEST_CASE_P(Core, Core_RotateC3, testing::Combine(
    testing::Values(CV_8U, CV_8S, CV_16U, CV_16S, CV_16F, CV_32S, CV_32F, CV_64F),
    testing::Values(Size(1, 1), Size(1, 19), Size(17, 1), Size(17, 19),
                    Size(32, 8), Size(129, 7), Size(513, 9), Size(1025, 3)),
    testing::Values(ROTATE_90_CLOCKWISE, ROTATE_180, ROTATE_90_COUNTERCLOCKWISE),
    testing::Bool()));

}} // namespace opencv_test
