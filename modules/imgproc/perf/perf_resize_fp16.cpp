// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test {

typedef tuple<Size, Size, int, int> ResizeHalfParams;
typedef TestBaseWithParam<ResizeHalfParams> ResizeHalfPerf;

PERF_TEST_P(ResizeHalfPerf, resize,
    testing::Combine(
        testing::Values(Size(320, 240), Size(1920, 1080)),
        testing::Values(Size(224, 224), Size(640, 480)),
        testing::Values(1, 3, 4),
        testing::Values(INTER_LINEAR, INTER_CUBIC, INTER_AREA, INTER_LANCZOS4)))
{
    const Size sourceSize = get<0>(GetParam()), destinationSize = get<1>(GetParam());
    const int channels = get<2>(GetParam()), interpolation = get<3>(GetParam());
    Mat input32(sourceSize, CV_MAKETYPE(CV_32F, channels));
    randu(input32, 0.f, 1.f);
    Mat src, dst(destinationSize, CV_MAKETYPE(CV_16F, channels));
    input32.convertTo(src, CV_16F);
    declare.in(src).out(dst);

    TEST_CYCLE() cv::resize(src, dst, destinationSize, 0, 0, interpolation);

    Mat result32;
    dst.convertTo(result32, CV_32F);
    SANITY_CHECK(result32, 2e-3, ERROR_RELATIVE);
}

} // namespace
