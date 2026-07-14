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

}} // namespace

#endif
