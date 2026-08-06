// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "perf_precomp.hpp"

namespace opencv_test
{

typedef perf::TestBaseWithParam<Size> Size_Denoising;

// Unconditional CPU performance coverage for fastNlMeansDenoising.
//
// Every pre-existing CPU measurement of this function was conditional:
//   - perf/perf_cuda.cpp:139 holds a real CPU TEST_CYCLE(), but the whole file sits
//     inside "#if defined(HAVE_CUDA) && defined(HAVE_OPENCV_CUDAARITHM) &&
//     defined(HAVE_OPENCV_CUDAIMGPROC)", so it requires a CUDA toolchain to exist at
//     all, plus --perf_impl=plain at runtime to take the non-CUDA branch.
//   - perf/opencl/perf_denoising.cpp sits inside "#ifdef HAVE_OPENCL" and reaches the
//     CPU path only incidentally, on machines with no OpenCL device, where UMat falls
//     back to Mat. With a device attached it measures the GPU instead.
// A build with OpenCL enabled and a device present therefore measured no CPU path.
//
// szVGA and sz720p deliberately match CUDA_DENOISING_IMAGE_SIZES in perf_cuda.cpp so
// the numbers are directly comparable with the CUDA figures. h, templateWindowSize
// and searchWindowSize match that test as well.
//
// The 2592x1944 case carries over TEST(Photo_Denoising, speed) from
// test/test_denoising.cpp, which timed a single call on cv/shared/5MP.png with
// getTickCount() and printf'd the result from inside the accuracy suite, where no
// tooling collected it. That image is 5.04 MP; nothing else in the tree measures
// denoising above 0.92 MP, which is the regime where parallel_for_ scaling and
// memory pressure show up. It is by far the slowest case here — hence time(120).
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
