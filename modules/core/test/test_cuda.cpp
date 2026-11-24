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

TEST(CUDA_GpuMat, output_array_setTo_multi_channel_value)
{
    if (cv::cuda::getCudaEnabledDeviceCount() <= 0)
        throw cvtest::SkipTestException("CUDA device is not available");
    cv::cuda::GpuMat dst(3, 4, CV_8UC3);
    cv::Mat value(1, 1, CV_8UC3, cv::Scalar(4, 5, 6));
    cv::_OutputArray output(dst);
    output.setTo(value);
    cv::Mat actual;
    dst.download(actual);
    const cv::Mat expected(3, 4, CV_8UC3, cv::Scalar(4, 5, 6));
    EXPECT_EQ(0.0, cv::norm(actual, expected, cv::NORM_INF));
}

}} // namespace

#endif
