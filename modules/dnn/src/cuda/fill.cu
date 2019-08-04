// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void fill(span<T> output, T value) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = value;
        }
    }

    template <class T>
    void fill(const Stream& stream, span<T> output, T value) {
        auto kernel = raw::fill<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, value);
    }

    template void fill(const Stream&, span<__half>, __half);
    template void fill(const Stream&, span<float>, float);
    template void fill(const Stream&, span<double>, double);

}}}} /* cv::dnn::cuda4dnn::kernels */
