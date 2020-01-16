// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "math.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"
#include "limits.hpp"
#include "vector_traits.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"

#include <opencv2/core.hpp>

#include <cstddef>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {

        template <class T>
        __global__ void region_box(
            Span<T> output, View<T> input, View<T> bias,
            size_type boxes_per_cell, size_type box_size,
            size_type rows, size_type cols,
            size_type height_norm, size_type width_norm,
            T object_prob_cutoff)
        {
            using vector2_type = get_vector_type_t<T, 2>;
            auto bias_vPtr = vector2_type::get_pointer(bias.data());

            for (auto box_index : grid_stride_range(output.size() / box_size)) {
                const auto box_of_the_cell = box_index % boxes_per_cell; /* box number within a cell */
                const auto box_offset = box_index * box_size;

                const auto batch_inner_size = rows * cols * boxes_per_cell;
                const auto row_inner_size = cols * boxes_per_cell;
                const auto col_inner_size = boxes_per_cell;

                const auto y = (box_index % batch_inner_size) / row_inner_size;
                const auto x = (box_index % row_inner_size) / col_inner_size;

                using device::sigmoid;
                output[box_offset + 0] = (T(x) + sigmoid(input[box_offset + 0])) / T(cols);
                output[box_offset + 1] = (T(y) + sigmoid(input[box_offset + 1])) / T(rows);

                vector2_type bias_xy;
                v_load(bias_xy, bias_vPtr[box_of_the_cell]);

                using device::exp;
                output[box_offset + 2] = exp(input[box_offset + 2]) * bias_xy.data[0] / T(width_norm);
                output[box_offset + 3] = exp(input[box_offset + 3]) * bias_xy.data[1] / T(height_norm);

                /* squash objectness score into a probability */
                using device::sigmoid;
                T objectness_prob = sigmoid(input[box_offset + 4]);

                /* ignore prediction if the objectness probability is less than the cutoff */
                if (objectness_prob < object_prob_cutoff)
                    objectness_prob = 0;

                output[box_offset + 4] = objectness_prob;
            }
        }

        template <class T>
        __global__ void region_sigmoid_class_score(Span<T> output, View<T> input, T class_prob_cutoff, size_type box_size)
        {
            for (auto idx : grid_stride_range(output.size())) {
                const index_type box_no = idx / box_size;
                const index_type start_of_box = box_no * box_size;
                const index_type box_offset = idx % box_size;

                if (box_offset < 5) {
                    /* continue as we have already processed these in region_box */
                    continue;
                }

                auto objectness_prob = output[start_of_box + 4];

                /* the class probabilities we currently have are conditional class probabilities
                 * given the object
                 *
                 * to obtain the actual class probability, we multiply the conditional probability
                 * with the object probability
                 */
                auto actual_class_prob = objectness_prob * sigmoid(input[idx]);
                if (actual_class_prob <= class_prob_cutoff)
                    actual_class_prob = T(0);
                output[idx] = actual_class_prob;
            }
        }

        template <class T>
        __global__ void region_softmax_class_score(Span<T> output, View<T> input, T class_prob_cutoff, size_type box_size) {
            for (auto box_no : grid_stride_range(output.size() / box_size)) {
                const index_type start_of_box = box_no * box_size;
                const index_type start_idx = start_of_box + 5;
                const index_type end_idx = start_of_box + box_size;

                auto largest = numeric_limits<T>::lowest();
                for (int idx = start_idx; idx < end_idx; idx++) {
                    using device::max;
                    largest = max(largest, input[idx]);
                }

                auto sum = T(0);
                for (int idx = start_idx; idx < end_idx; idx++) {
                    using device::exp;
                    auto temp = exp(input[idx] - largest);
                    sum += temp;
                    output[idx] = temp;
                }

                for (int idx = start_idx; idx < end_idx; idx++) {
                    auto softmax_score = output[idx] / sum;

                    /* the class probabilities we currently have are conditional class probabilities
                     * given the object
                     *
                     * to obtain the actual class probability, we multiply the conditional probability
                     * with the object probability
                     */
                    auto objectness_prob = output[start_of_box + 4];
                    auto actual_class_prob = objectness_prob * softmax_score;
                    if (actual_class_prob <= class_prob_cutoff)
                        actual_class_prob = T(0);
                    output[idx] = actual_class_prob;
                }
            }
        }
    }

    template <class T>
    void region(const Stream& stream, Span<T> output, View<T> input, View<T> bias,
        T object_prob_cutoff, T class_prob_cutoff,
        std::size_t boxes_per_cell, std::size_t box_size,
        std::size_t rows, std::size_t cols,
        std::size_t height_norm, std::size_t width_norm,
        bool if_true_sigmoid_else_softmax /* true = sigmoid, false = softmax */)
    {
        CV_Assert(output.size() == input.size());
        CV_Assert(output.size() % box_size == 0);
        CV_Assert(is_fully_aligned(bias, 2));

        auto box_kernel = raw::region_box<T>;
        auto box_policy = make_policy(box_kernel, output.size() / box_size, 0, stream);
        launch_kernel(box_kernel, box_policy,
            output, input, bias, boxes_per_cell, box_size,
            rows, cols, height_norm, width_norm,
            object_prob_cutoff);

        if (if_true_sigmoid_else_softmax) {
            auto kernel_score = raw::region_sigmoid_class_score<T>;
            auto policy_score = make_policy(kernel_score, output.size(), 0, stream);
            launch_kernel(kernel_score, policy_score, output, input, class_prob_cutoff, box_size);
        } else {
            auto kernel_score = raw::region_softmax_class_score<T>;
            auto policy_score = make_policy(kernel_score, output.size(), 0, stream);
            launch_kernel(kernel_score, policy_score, output, input, class_prob_cutoff, box_size);
        }
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void region(const Stream&, Span<__half>, View<__half>, View<__half>,
        __half, __half, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, bool);
#endif

    template void region(const Stream&, Span<float>, View<float>, View<float>,
        float, float, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, bool);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
