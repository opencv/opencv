// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{

typedef perf::TestBaseWithParam<Size> Size_Denoising;

PERF_TEST_P(Size_Denoising, fastNlMeansDenoising,
            testing::Values(::perf::szVGA, ::perf::sz720p, Size(2592, 1944)))
{
    const Size size = GetParam();

    Mat original = imread(getDataPath("cv/denoising/lena_noised_gaussian_sigma=10.png"),
                          IMREAD_GRAYSCALE);
    ASSERT_FALSE(original.empty()) << "Could not load input image";

    Mat tiled;
    repeat(original, (size.height + original.rows - 1) / original.rows,
                     (size.width + original.cols - 1) / original.cols, tiled);
    Mat src = tiled(Rect(0, 0, size.width, size.height)).clone();
    Mat dst(size, CV_8UC1);

    declare.in(src).out(dst).time(120);

    TEST_CYCLE() cv::fastNlMeansDenoising(src, dst, 10, 7, 21);

    SANITY_CHECK_NOTHING();
}

} // namespace
