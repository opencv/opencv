// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "math.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T>
        __global__ void eltwise_max_2(span<T> output, view<T> x, view<T> y) {
            assert(x.size() == y.size());
            assert(output.size() >= y.size());

            for (auto i : grid_stride_range(output.size())) {
                using utils::max;
                output[i] = max(x[i], y[i]);
            }
        }

        template <class T>
        __global__ void eltwise_sum_2(span<T> output, view<T> x, view<T> y) {
            assert(x.size() == y.size());
            assert(output.size() >= y.size());

            for (auto i : grid_stride_range(output.size()))
                output[i] = x[i] + y[i];
        }

        template <class T>
        __global__ void eltwise_sum_coeff_2(span<T> output, T coeff_x, view<T> x, T coeff_y, view<T> y) {
            assert(x.size() == y.size());
            assert(output.size() >= y.size());

            for (auto i : grid_stride_range(output.size()))
                output[i] = coeff_x * x[i] + coeff_y * y[i];
        }

        template <class T>
        __global__ void eltwise_prod_2(span<T> output, view<T> x, view<T> y) {
            assert(x.size() == y.size());
            assert(output.size() >= y.size());

            for (auto i : grid_stride_range(output.size()))
                output[i] = x[i] * y[i];
        }
    }

    template <class T>
    void eltwise_max_2(const Stream& stream, span<T> output, view<T> x, view<T> y) {
        auto policy = make_policy(raw::eltwise_max_2<T>, 0, stream);
        launch_kernel(raw::eltwise_max_2<T>, policy, output, x, y);
    }

    template void eltwise_max_2(const Stream& stream, span<float> output, view<float> x, view<float> y);
    template void eltwise_max_2(const Stream& stream, span<double> output, view<double> x, view<double> y);

    template <class T>
    void eltwise_sum_2(const Stream& stream, span<T> output, view<T> x, view<T> y) {
        auto policy = make_policy(raw::eltwise_sum_2<T>, 0, stream);
        launch_kernel(raw::eltwise_sum_2<T>, policy, output, x, y);
    }

    template void eltwise_sum_2(const Stream& stream, span<float> output, view<float> x, view<float> y);
    template void eltwise_sum_2(const Stream& stream, span<double> output, view<double> x, view<double> y);

    template <class T>
    void eltwise_sum_coeff_2(const Stream& stream, span<T> output, T coeff_x, view<T> x, T coeff_y, view<T> y) {
        auto policy = make_policy(raw::eltwise_sum_coeff_2<T>, 0, stream);
        launch_kernel(raw::eltwise_sum_coeff_2<T>, policy, output, coeff_x, x, coeff_y, y);
    }

    template void eltwise_sum_coeff_2(const Stream&, span<float>, float, view<float>, float, view<float>);
    template void eltwise_sum_coeff_2(const Stream&, span<double>, double, view<double>, double, view<double>);

    template <class T>
    void eltwise_prod_2(const Stream& stream, span<T> output, view<T> x, view<T> y) {
        auto policy = make_policy(raw::eltwise_prod_2<T>, 0, stream);
        launch_kernel(raw::eltwise_prod_2<T>, policy, output, x, y);
    }

    template void eltwise_prod_2(const Stream& stream, span<float> output, view<float> x, view<float> y);
    template void eltwise_prod_2(const Stream& stream, span<double> output, view<double> x, view<double> y);

}}}}} /* cv::dnn::cuda4dnn::csl::kernels */
