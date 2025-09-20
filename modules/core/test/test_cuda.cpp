// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#if defined(HAVE_CUDA)

#include "test_precomp.hpp"
#include <cuda_runtime.h>
#include "opencv2/core/cuda.hpp"

namespace opencv_test { namespace {

TEST(CUDA_Stream, construct_cudaFlags)
{
    cv::cuda::Stream stream(cudaStreamNonBlocking);
    EXPECT_NE(stream.cudaPtr(), nullptr);
}

typedef testing::TestWithParam< tuple<perf::MatType, perf::MatType> > GpuMat;
TEST_P(GpuMat, convertTo)
{
    int sdepth = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    if (sdepth == CV_16F || sdepth == CV_Bool || sdepth == CV_16BF)
        throw SkipTestException("Unsupported src type");
    if (ddepth == CV_16F || ddepth == CV_Bool || ddepth == CV_16BF)
        throw SkipTestException("Unsupported dst type");

    Mat ref(16, 20, CV_8U), testMat;
    randu(ref, 0, 128);
    ref.convertTo(ref, sdepth);

    cv::cuda::GpuMat src, dst;
    src.upload(ref);
    EXPECT_EQ(sdepth, src.depth());
    src.convertTo(dst, ddepth);
    EXPECT_EQ(ddepth, dst.depth());
    dst.download(testMat);

    Mat expected;
    ref.convertTo(expected, ddepth);
    EXPECT_EQ(ddepth, testMat.depth());
    ASSERT_EQ(0, cvtest::norm(expected, testMat, NORM_INF));
}

TEST_P(GpuMat, convertToScale)
{
    int sdepth = get<0>(GetParam());
    int ddepth = get<1>(GetParam());
    if (sdepth == CV_16F || sdepth == CV_Bool || sdepth == CV_16BF)
        throw SkipTestException("Unsupported src type");
    if (ddepth == CV_16F || ddepth == CV_Bool || ddepth == CV_16BF)
        throw SkipTestException("Unsupported dst type");

    Mat ref(16, 20, CV_8U), testMat;
    randu(ref, 10, 50);
    ref.convertTo(ref, sdepth);

    cv::cuda::GpuMat src, dst;
    src.upload(ref);
    EXPECT_EQ(sdepth, src.depth());
    src.convertTo(dst, ddepth, 2, -1);
    EXPECT_EQ(ddepth, dst.depth());
    dst.download(testMat);

    Mat expected;
    ref.convertTo(expected, ddepth, 2, -1);
    EXPECT_EQ(ddepth, testMat.depth());
    ASSERT_EQ(0, cvtest::norm(expected, testMat, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(/*nothing*/, GpuMat, testing::Combine(
    testing::Range(perf::MatType(CV_8U), perf::MatType(CV_DEPTH_CURR_MAX)),
    testing::Range(perf::MatType(CV_8U), perf::MatType(CV_DEPTH_CURR_MAX))
));

}} // namespace

#endif
