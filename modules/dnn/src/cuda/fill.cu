// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/pointer.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void fill(span<T> output, T value)
        {
            for (auto i : grid_stride_range(output.size()))
                output[i] = value;
        }
    }

    template <class T>
    void fill(const Stream& stream, span<T> output, T value)
    {
        auto kernel = raw::fill<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, output, value);
    }

    template void fill<float>(const Stream&, span<float>, float);
    template void fill<double>(const Stream&, span<double>, double);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
