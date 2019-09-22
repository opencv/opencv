// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "array.hpp"
#include "math.hpp"
#include "types.hpp"
#include "vector_traits.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        template <class T, bool Normalize>
        __global__ void prior_box(
            Span<T> output,
            View<float> boxWidth, View<float> boxHeight, View<float> offsetX, View<float> offsetY, float stepX, float stepY,
            size_type layerWidth, size_type layerHeight,
            size_type imageWidth, size_type imageHeight)
        {
            /* each box consists of two pair of coordinates and hence 4 values in total */
            /* since the entire output consists (first channel at least) of these boxes,
             * we are garunteeed that the output is aligned to a boundary of 4 values
             */
            using vector_type = get_vector_type_t<T, 4>;
            auto output_vPtr = vector_type::get_pointer(output.data());

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
                            vec.data[0] = (center_x - boxWidth[i] * 0.5f) / imageWidth;
                            vec.data[1] = (center_y - boxHeight[i] * 0.5f) / imageHeight;
                            vec.data[2] = (center_x + boxWidth[i] * 0.5f) / imageWidth;
                            vec.data[3] = (center_y + boxHeight[i] * 0.5f) / imageHeight;
                        } else {
                            vec.data[0] = center_x - boxWidth[i] * 0.5f;
                            vec.data[1] = center_y - boxHeight[i] * 0.5f;
                            vec.data[2] = center_x + boxWidth[i] * 0.5f - 1.0f;
                            vec.data[3] = center_y + boxHeight[i] * 0.5f - 1.0f;
                        }

                        v_store(output_vPtr[output_offset_v4], vec);
                        output_offset_v4++;
                    }
                }
            }
        }

        template <class T>
        __global__ void prior_box_clip(Span<T> output) {
            for (auto i : grid_stride_range(output.size())) {
                using device::clamp;
                output[i] = clamp<T>(output[i], 0.0, 1.0);
            }
        }

        template <class T>
        __global__ void prior_box_set_variance1(Span<T> output, float variance) {
            using vector_type = get_vector_type_t<T, 4>;
            auto output_vPtr = vector_type::get_pointer(output.data());
            for (auto i : grid_stride_range(output.size() / 4)) {
                vector_type vec;
                for (int j = 0; j < 4; j++)
                    vec.data[j] = variance;
                v_store(output_vPtr[i], vec);
            }
        }

        template <class T>
        __global__ void prior_box_set_variance4(Span<T> output, array<float, 4> variance) {
            using vector_type = get_vector_type_t<T, 4>;
            auto output_vPtr = vector_type::get_pointer(output.data());
            for (auto i : grid_stride_range(output.size() / 4)) {
                vector_type vec;
                for(int j = 0; j < 4; j++)
                    vec.data[j] = variance[j];
                v_store(output_vPtr[i], vec);
            }
        }
    }

    template <class T, bool Normalize> static
    void launch_prior_box_kernel(
        const Stream& stream,
        Span<T> output, View<float> boxWidth, View<float> boxHeight, View<float> offsetX, View<float> offsetY, float stepX, float stepY,
        std::size_t layerWidth, std::size_t layerHeight, std::size_t imageWidth, std::size_t imageHeight)
    {
        auto num_points = layerWidth * layerHeight;
        auto kernel = raw::prior_box<T, Normalize>;
        auto policy = make_policy(kernel, num_points, 0, stream);
        launch_kernel(kernel, policy,
            output, boxWidth, boxHeight, offsetX, offsetY, stepX, stepY,
            layerWidth, layerHeight, imageWidth, imageHeight);
    }

    template <class T>
    void generate_prior_boxes(
        const Stream& stream,
        Span<T> output,
        View<float> boxWidth, View<float> boxHeight, View<float> offsetX, View<float> offsetY, float stepX, float stepY,
        std::vector<float> variance,
        std::size_t numPriors,
        std::size_t layerWidth, std::size_t layerHeight,
        std::size_t imageWidth, std::size_t imageHeight,
        bool normalize, bool clip)
    {
        if (normalize) {
            launch_prior_box_kernel<T, true>(
                stream, output, boxWidth, boxHeight, offsetX, offsetY, stepX, stepY,
                layerWidth, layerHeight, imageWidth, imageHeight
            );
        } else {
            launch_prior_box_kernel<T, false>(
                stream, output, boxWidth, boxHeight, offsetX, offsetY, stepX, stepY,
                layerWidth, layerHeight, imageWidth, imageHeight
            );
        }

        std::size_t channel_size = layerHeight * layerWidth * numPriors * 4;
        CV_Assert(channel_size * 2 == output.size());

        if (clip) {
            auto output_span_c1 = Span<T>(output.data(), channel_size);
            auto kernel = raw::prior_box_clip<T>;
            auto policy = make_policy(kernel, output_span_c1.size(), 0, stream);
            launch_kernel(kernel, policy, output_span_c1);
        }

        auto output_span_c2 = Span<T>(output.data() + channel_size, channel_size);
        if (variance.size() == 1) {
            auto kernel = raw::prior_box_set_variance1<T>;
            auto policy = make_policy(kernel, output_span_c2.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output_span_c2, variance[0]);
        } else {
            array<float, 4> variance_k;
            variance_k.assign(std::begin(variance), std::end(variance));
            auto kernel = raw::prior_box_set_variance4<T>;
            auto policy = make_policy(kernel, output_span_c2.size() / 4, 0, stream);
            launch_kernel(kernel, policy, output_span_c2, variance_k);
        }
    }

    template void generate_prior_boxes(const Stream&, Span<__half>, View<float>, View<float>, View<float>, View<float>, float, float,
        std::vector<float>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, bool, bool);

    template void generate_prior_boxes(const Stream&, Span<float>, View<float>, View<float>, View<float>, View<float>, float, float,
        std::vector<float>, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, bool, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
