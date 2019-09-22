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
        __global__ void sigmoid_strided(Span<T> output, View<T> input, size_type n, size_type stride, size_type offset) {
            /* - the input is divided into equal blocks strided by `stride`
             * - we must apply sigmoid to a continuous range of `n` values starting from `offset` in every block
             */
            for (auto i : grid_stride_range(n * output.size() / stride)) {
                auto block_idx = i / n;
                auto index = block_idx * stride + offset + (i % n);

                using device::sigmoid;
                output[index] = sigmoid(input[index]);
            }
        }

        template <class T>
        __global__ void softmax_strided(Span<T> output, View<T> input, size_type n, size_type stride, size_type offset_) {
            for (auto idx : grid_stride_range(output.size() / stride)) {
                index_type offset = idx * stride + offset_;

                auto largest = numeric_limits<T>::lowest();
                for (int i = 0; i < n; i++) {
                    using device::max;
                    largest = max(largest, output[offset + i]);
                }

                auto sum = T(0);
                for (int i = 0; i < n; i++) {
                    using device::exp;
                    auto temp = exp(output[offset + i] - largest);
                    sum += temp;
                    output[offset + i] = temp;
                }

                for (int i = 0; i < n; i++) {
                    output[offset + i] /= sum;
                }
            }
        }

        template <class T>
        __global__ void region_finalize(Span<T> output, View<T> input, View<T> bias,
            T object_prob_cutoff, T class_prob_cutoff,
            size_type height_norm, size_type width_norm,
            size_type rows, size_type cols,
            size_type boxes_per_cell,
            size_type box_size,
            size_type classes)
        {
            for (auto box_index : grid_stride_range(output.size() / box_size)) {
                auto box_of_the_cell = box_index % boxes_per_cell; /* box number within a cell */
                auto box_offset = box_index * box_size;

                auto batch_inner_size = rows * cols * boxes_per_cell;
                auto row_inner_size = cols * boxes_per_cell;
                auto col_inner_size = boxes_per_cell;

                auto y = (box_index % batch_inner_size) / row_inner_size;
                auto x = (box_index % row_inner_size) / col_inner_size;

                using device::sigmoid;
                using device::exp;
                output[box_offset + 0] = (T(x) + sigmoid(input[box_offset + 0])) / T(cols);
                output[box_offset + 1] = (T(y) + sigmoid(input[box_offset + 1])) / T(rows);
                output[box_offset + 2] = exp(input[box_offset + 2]) * bias[2 * box_of_the_cell + 0] / T(width_norm);
                output[box_offset + 3] = exp(input[box_offset + 3]) * bias[2 * box_of_the_cell + 1] / T(height_norm);

                /* squash objectness score into a probability */
                using device::sigmoid;
                T objectness_prob = sigmoid(output[box_offset + 4]);
                output[box_offset + 4] = objectness_prob;

                /* ignore prediction if the objectness probability is less than the cutoff */
                if (objectness_prob < object_prob_cutoff)
                    objectness_prob = 0;

                /* the class probabilities we have currently are conditional class probabilities
                 * given the object
                 *
                 * to obtain the actual class probability, we multiply the conditional probability
                 * with the object probability
                 */
                const index_type class_begin = box_offset + 5; /* 4 box coordinates, 1 obj prob, class probs... */
                const index_type class_end = class_begin + classes;
                index_type offset = class_begin;

                using vector_type = get_vector_type_t<T, 4>;

                /* process each class independently until the offset is aligned to an n-element boundary */
                while (offset % vector_type::size() != 0 && offset < class_end) {
                    T actual_class_prob = objectness_prob * output[offset];
                    if (actual_class_prob <= class_prob_cutoff)
                        actual_class_prob = T(0);
                    output[offset] = actual_class_prob;
                    offset++;
                }

                auto output_vPtr = vector_type::get_pointer(output.data() + offset);
                auto input_vPtr = vector_type::get_pointer(input.data() + offset);
                for (int i = 0; (offset + vector_type::size()) < class_end; i++) {
                    vector_type vec;
                    v_load(vec, output_vPtr[i]);
                    for (int j = 0; j < vector_type::size(); j++) {
                        T actual_class_prob = objectness_prob * vec.data[j];
                        if (actual_class_prob <= class_prob_cutoff)
                            actual_class_prob = T(0);
                        vec.data[j] = actual_class_prob;
                    }
                    v_store(output_vPtr[i], vec);
                    offset += vector_type::size();
                }

                /* process the remaining classes */
                while (offset < class_end) {
                    T actual_class_prob = objectness_prob * output[offset];
                    if (actual_class_prob <= class_prob_cutoff)
                        actual_class_prob = T(0);
                    output[offset] = actual_class_prob;
                    offset++;
                }
            }
        }
    }

    template <class T>
    void sigmoid_strided(const Stream& stream, Span<T> output, View<T> input, std::size_t n, std::size_t stride, std::size_t offset) {
        CV_Assert(output.size() % stride == 0);

        auto kernel = raw::sigmoid_strided<T>;
        auto policy = make_policy(kernel, n * output.size() / stride, 0, stream);
        launch_kernel(kernel, policy, output, input, n, stride, offset);
    }

    template void sigmoid_strided(const Stream&, Span<__half>, View<__half>, std::size_t, std::size_t, std::size_t);
    template void sigmoid_strided(const Stream&, Span<float>, View<float>, std::size_t, std::size_t, std::size_t);

    template <class T>
    void softmax_strided(const Stream& stream, Span<T> output, View<T> input, std::size_t n, std::size_t stride, std::size_t offset) {
        CV_Assert(output.size() % stride == 0);

        auto kernel = raw::softmax_strided<T>;
        auto policy = make_policy(kernel, output.size() / stride, 0, stream);
        launch_kernel(kernel, policy, output, input, n, stride, offset);
    }

    template void softmax_strided(const Stream&, Span<__half>, View<__half>, std::size_t, std::size_t, std::size_t);
    template void softmax_strided(const Stream&, Span<float>, View<float>, std::size_t, std::size_t, std::size_t);

    template <class T>
    void region_finalize(const Stream& stream, Span<T> output, View<T> input, View<T> bias,
        T object_prob_cutoff, T class_prob_cutoff,
        std::size_t height_norm, std::size_t width_norm,
        std::size_t rows, std::size_t cols,
        std::size_t boxes_per_cell,
        std::size_t box_size,
        std::size_t classes)
    {
        CV_Assert(output.size() % box_size == 0);

        auto kernel = raw::region_finalize<T>;
        auto policy = make_policy(kernel, output.size() / box_size, 0, stream);
        launch_kernel(kernel, policy, output, input, bias,
            object_prob_cutoff, class_prob_cutoff,
            height_norm, width_norm,
            rows, cols, boxes_per_cell, box_size, classes);
    }

    template void region_finalize(const Stream&, Span<__half>, View<__half>, View<__half>,
        __half, __half, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

    template void region_finalize(const Stream&, Span<float>, View<float>, View<float>,
        float, float, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t, std::size_t);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
