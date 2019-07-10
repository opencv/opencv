// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "array.hpp"
#include "math.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_loop.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/kernels.hpp"
#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace cv { namespace dnn { namespace cuda4dnn { namespace csl  { namespace kernels {

    namespace raw {
        using index_type = gpu::index_type;
        using size_type = gpu::size_type;

        template <class T, bool Normalize>
        __global__ void prior_box(
            span<T> output,
            view<T> boxWidth, view<T> boxHeight, view<T> offsetX, view<T> offsetY,
            size_type layerWidth, size_type layerHeight,
            size_type imageWidth, size_type imageHeight,
            T stepX, T stepY)
        {
            /* each box consists of two pair of coordinates and hence 4 values in total */
            /* since the entire output consists (first channel at least) of these boxes,
             * we are garunteeed that the output is aligned to a boundary of 4 values
             */
            using vector_type = typename get_vector_type<T, 4>::type;
            vector_type* outputPtr_v4 = reinterpret_cast<vector_type*>(output.data().get());

            /* num_points contains the number of points in the feature map of interest
             * each iteration of the stride loop selects a point and generates prior boxes for it
             */
            size_type num_points = layerWidth * layerHeight;
            for (auto idx : grid_stride_range(num_points)) {
                const index_type x = idx % layerWidth,
                                 y = idx / layerWidth;

                index_type output_offset_v4 = idx * offsetX.size() * boxWidth.size();
                for (int i = 0; i < boxWidth.size(); i++) {
                    for (int j = 0; j < offsetX.size(); j++) {
                        float center_x = (x + offsetX[j]) * stepX;
                        float center_y = (y + offsetY[j]) * stepY;

                        vector_type vec;
                        if(Normalize) {
                            vec.x = (center_x - boxWidth[i] * 0.5f) / imageWidth;
                            vec.y = (center_y - boxHeight[i] * 0.5f) / imageHeight;
                            vec.z = (center_x + boxWidth[i] * 0.5f) / imageWidth;
                            vec.w = (center_y + boxHeight[i] * 0.5f) / imageHeight;
                        } else {
                            vec.x = center_x - boxWidth[i] * 0.5f;
                            vec.y = center_y - boxHeight[i] * 0.5f;
                            vec.z = center_x + boxWidth[i] * 0.5f - 1.0f;
                            vec.w = center_y + boxHeight[i] * 0.5f - 1.0f;
                        }

                        outputPtr_v4[output_offset_v4] = vec;
                        output_offset_v4++;
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
            using vector_type = typename get_vector_type<T, 4>::type;
            vector_type* outputPtr_v4 = reinterpret_cast<vector_type*>(output.data().get());
            for (auto i : grid_stride_range(output.size() / 4)) {
                vector_type vec;
                vec.x = variance;
                vec.y = variance;
                vec.z = variance;
                vec.w = variance;
                outputPtr_v4[i] = vec;
            }
        }

        template <class T, std::size_t N>
        using array = utils::array<T, N>;

        template <class T>
        __global__ void prior_box_set_variance4(span<T> output, array<T, 4> variance) {
            using vector_type = typename get_vector_type<T, 4>::type;
            vector_type* outputPtr_v4 = reinterpret_cast<vector_type*>(output.data().get());
            for (auto i : grid_stride_range(output.size() / 4)) {
                vector_type vec;
                vec.x = variance[0];
                vec.y = variance[1];
                vec.z = variance[2];
                vec.w = variance[3];
                outputPtr_v4[i] = vec;
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
            auto policy = make_policy(kernel, output_span_c2.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output_span_c2, variance[0]);
        } else {
            utils::array<T, 4> variance_k;
            variance_k.assign(std::begin(variance), std::end(variance));
            auto kernel = raw::prior_box_set_variance4<T>;
            auto policy = make_policy(kernel, output_span_c2.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output_span_c2, variance_k);
        }
    }

    template void generate_prior_boxes(const Stream&, span<float>, view<float>, view<float>, view<float>, view<float>,
        std::vector<float>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, float, float, bool, bool);

    template void generate_prior_boxes(const Stream&, span<double>, view<double>, view<double>, view<double>, view<double>,
        std::vector<double>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, double, double, bool, bool);

}}}}} /*  cv::dnn::cuda4dnn::csl::kernels */
