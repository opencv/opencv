// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, std::size_t N>
        __global__ void fill_vec(Span<T> output, T value) {
            using vector_type = get_vector_type_t<T, N>;

            auto output_vPtr = vector_type::get_pointer(output.data());
            for (auto i : grid_stride_range(output.size() / vector_type::size())) {
                vector_type vec;
                for (int j = 0; j < vector_type::size(); j++)
                    vec.data[j] = value;
                v_store(output_vPtr[i], vec);
            }
        }
    }

    template <class T, std::size_t N> static
    void launch_vectorized_fill(const Stream& stream, Span<T> output, T value) {
        CV_Assert(is_fully_aligned<T>(output, N));

        auto kernel = raw::fill_vec<T, N>;
        auto policy = make_policy(kernel, output.size() / N, 0, stream);
        launch_kernel(kernel, policy, output, value);
    }

    template <class T>
    void fill(const Stream& stream, Span<T> output, T value) {
        if (is_fully_aligned<T>(output, 4)) {
            launch_vectorized_fill<T, 4>(stream, output, value);
        } else if (is_fully_aligned<T>(output, 2)) {
            launch_vectorized_fill<T, 2>(stream, output, value);
        } else {
            launch_vectorized_fill<T, 1>(stream, output, value);
        }
    }

    template void fill(const Stream&, Span<__half>, __half);
    template void fill(const Stream&, Span<float>, float);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
