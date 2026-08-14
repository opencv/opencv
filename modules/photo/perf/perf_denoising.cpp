// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{

typedef perf::TestBaseWithParam<Size> Size_Denoising;

// CPU coverage for fastNlMeansDenoising: the other denoising perf tests are gated behind
// HAVE_CUDA (perf_cuda.cpp) and HAVE_OPENCL (opencl/perf_denoising.cpp).
//
// szVGA and sz720p, and the h/template/search parameters, match
// CUDA_DENOISING_IMAGE_SIZES in perf_cuda.cpp so the figures stay comparable. The
// 2592x1944 case is the only one above 0.92 MP, where parallel_for_ scaling shows;
// it is also the slowest, hence time(120).
PERF_TEST_P(Size_Denoising, fastNlMeansDenoising,
            testing::Values(::perf::szVGA, ::perf::sz720p, Size(2592, 1944)))
{
    const Size size = GetParam();

    Mat src(size, CV_8UC1);
    Mat dst(size, CV_8UC1);

    declare.in(src, WARMUP_RNG).out(dst).time(120);

    TEST_CYCLE() cv::fastNlMeansDenoising(src, dst, 10, 7, 21);

    SANITY_CHECK_NOTHING();
}

} // namespace
