// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "array.hpp"
#include "math.hpp"
#include "reduce.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/kernel_utils.hpp"
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/stream.hpp"

#include <cstddef>
#include <cuda_runtime.h>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        template <class T, bool Normalize>
        __global__ void prior_box(
            span<T> output,
            view<T> boxWidth, view<T> boxHeight, view<T> offsetX, view<T> offsetY,
            std::size_t layerWidth, std::size_t layerHeight,
            std::size_t imageWidth, std::size_t imageHeight,
            T stepX, T stepY)
        {
            /* num_points contains the number of points in the feature map of interest
             * each iteration of the stride loop selects a point and generates prior boxes for it
             */
            std::size_t num_points = layerWidth * layerHeight;
            for (auto idx : grid_stride_range(num_points)) {
                auto x = idx % layerWidth,
                     y = idx / layerWidth;

                DevicePtr<T> output_ptr = output.data() + idx * 4 * offsetX.size() * boxWidth.size();

                for (int i = 0; i < boxWidth.size(); i++) {
                    for (int j = 0; j < offsetX.size(); j++) {
                        float center_x = (x + offsetX[j]) * stepX;
                        float center_y = (y + offsetY[j]) * stepY;

                        if(Normalize) {
                            output_ptr[0] = (center_x - boxWidth[i] * 0.5f) / imageWidth;
                            output_ptr[1] = (center_y - boxHeight[i] * 0.5f) / imageHeight;
                            output_ptr[2] = (center_x + boxWidth[i] * 0.5f) / imageWidth;
                            output_ptr[3] = (center_y + boxHeight[i] * 0.5f) / imageHeight;
                        } else {
                            output_ptr[0] = center_x - boxWidth[i] * 0.5f;
                            output_ptr[1] = center_y - boxHeight[i] * 0.5f;
                            output_ptr[2] = center_x + boxWidth[i] * 0.5f - 1.0f;
                            output_ptr[3] = center_y + boxHeight[i] * 0.5f - 1.0f;
                        }

                        output_ptr += 4;
                    }
                }
            }
        }

        template <class T>
        __global__ void prior_box_clip(span<T> output) {
            for (auto i : grid_stride_range(output.size())) {
                using utils::clamp;
                output[i] = clamp<T>(output[i], 0.0, 1.0);
            }
        }

        template <class T>
        __global__ void prior_box_set_variance1(span<T> output, T variance) {
            for (auto i : grid_stride_range(output.size()))
                output[i] = variance;
        }

        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T>
        __global__ void prior_box_set_variance4(span<T> output, array<T, 4> variance) {
            for (auto i : grid_stride_range(output.size())) {
                const auto vidx = i % variance.size();
                output[i] = variance[vidx];
            }
        }
    }

    template <class T, bool Normalize> static
    void launch_prior_box_kernel(
        const Stream& stream,
        span<T> output, view<T> boxWidth, view<T> boxHeight, view<T> offsetX, view<T> offsetY,
        std::size_t layerWidth, std::size_t layerHeight, std::size_t imageWidth, std::size_t imageHeight,
        T stepX, T stepY)
    {
        auto num_points = layerWidth * layerHeight;
        auto kernel = raw::prior_box<T, Normalize>;
        auto policy = make_policy(kernel, num_points, 0, stream);
        launch_kernel(kernel, policy,
            output, boxWidth, boxHeight, offsetX, offsetY,
            layerWidth, layerHeight, imageWidth, imageHeight,
            stepX, stepY);
    }

    template <class T>
    void generate_prior_boxes(
        const Stream& stream,
        span<T> output,
        view<T> boxWidth, view<T> boxHeight, view<T> offsetX, view<T> offsetY,
        std::vector<T> variance,
        std::size_t numPriors,
        std::size_t layerWidth, std::size_t layerHeight,
        std::size_t imageWidth, std::size_t imageHeight,
        T stepX, T stepY,
        bool normalize, bool clip)
    {
        if (normalize) {
            launch_prior_box_kernel<T, true>(
                stream, output, boxWidth, boxHeight, offsetX, offsetY,
                layerWidth, layerHeight, imageWidth, imageHeight, stepX, stepY
            );
        } else {
            launch_prior_box_kernel<T, false>(
                stream, output, boxWidth, boxHeight, offsetX, offsetY,
                layerWidth, layerHeight, imageWidth, imageHeight, stepX, stepY
            );
        }

        std::size_t channel_size = layerHeight * layerWidth * numPriors * 4;
        CV_Assert(channel_size * 2 == output.size());

        if (clip) {
            auto output_span_c1 = span<T>(output.data(), channel_size);
            auto kernel = raw::prior_box_clip<T>;
            auto policy = make_policy(kernel, output_span_c1.size(), 0, stream);
            launch_kernel(kernel, policy, output_span_c1);
        }

        auto output_span_c2 = span<T>(output.data() + channel_size, channel_size);
        if (variance.size() == 1) {
            auto kernel = raw::prior_box_set_variance1<T>;
            auto policy = make_policy(kernel, output_span_c2.size(), 0, stream);
            launch_kernel(kernel, policy, output_span_c2, variance[0]);
        } else {
            utils::array<T, 4> variance_k;
            variance_k.assign(std::begin(variance), std::end(variance));
            auto kernel = raw::prior_box_set_variance4<T>;
            auto policy = make_policy(kernel, output_span_c2.size(), 0, stream);
            launch_kernel(kernel, policy, output_span_c2, variance_k);
        }
    }

    template void generate_prior_boxes(const Stream&, span<float>, view<float>, view<float>, view<float>, view<float>,
        std::vector<float>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, float, float, bool, bool);

    template void generate_prior_boxes(const Stream&, span<double>, view<double>, view<double>, view<double>, view<double>,
        std::vector<double>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, double, double, bool, bool);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
