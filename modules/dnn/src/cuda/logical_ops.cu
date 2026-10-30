// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "types.hpp"
#include "grid_stride_range.hpp"
#include "execution.hpp"

#include "../cuda4dnn/csl/stream.hpp"
#include "../cuda4dnn/csl/span.hpp"
#include "../cuda4dnn/csl/tensor.hpp"

#include "../cuda4dnn/kernels/logical_ops.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace cv::dnn::cuda4dnn::csl;
using namespace cv::dnn::cuda4dnn::csl::device;

namespace cv { namespace dnn { namespace cuda4dnn { namespace kernels {

    namespace raw {
        struct BroadcastIndexLayout {
            int rank{};
            std::int64_t out_dims[CSL_MAX_TENSOR_RANK]{};
            std::int64_t strides[3][CSL_MAX_TENSOR_RANK]{};
        };

        struct EqualFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a == b; }
        };

        struct GreaterFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a > b; }
        };

        struct GreaterEqualFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a >= b; }
        };

        struct LessFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a < b; }
        };

        struct LessEqualFunctor {
            template <class T> __device__ bool operator()(T a, T b) const { return a <= b; }
        };

        struct AndFunctor {
            __device__ bool operator()(bool a, bool b) const { return a && b; }
        };

        struct OrFunctor {
            __device__ bool operator()(bool a, bool b) const { return a || b; }
        };

        struct XorFunctor {
            __device__ bool operator()(bool a, bool b) const { return a != b; }
        };

        template <class T, class Functor>
        __global__ void binary_to_bool(Span<bool> output, View<T> x, View<T> y, BroadcastIndexLayout layout)
        {
            Functor functor;
            for (auto id : grid_stride_range(output.size())) {
                std::int64_t remaining = id;
                std::int64_t x_index = 0, y_index = 0;
                for (int axis = layout.rank - 1; axis >= 0; --axis) {
                    const std::int64_t coord = remaining % layout.out_dims[axis];
                    remaining /= layout.out_dims[axis];
                    x_index += coord * layout.strides[0][axis];
                    y_index += coord * layout.strides[1][axis];
                }
                output[id] = functor(x.data().get()[x_index], y.data().get()[y_index]);
            }
        }

        template <class T>
        __global__ void where(Span<T> output, View<bool> cond, View<T> x, View<T> y, BroadcastIndexLayout layout)
        {
            for (auto id : grid_stride_range(output.size())) {
                std::int64_t remaining = id;
                std::int64_t c_index = 0, x_index = 0, y_index = 0;
                for (int axis = layout.rank - 1; axis >= 0; --axis) {
                    const std::int64_t coord = remaining % layout.out_dims[axis];
                    remaining /= layout.out_dims[axis];
                    c_index += coord * layout.strides[0][axis];
                    x_index += coord * layout.strides[1][axis];
                    y_index += coord * layout.strides[2][axis];
                }
                output[id] = cond.data().get()[c_index] ? x.data().get()[x_index] : y.data().get()[y_index];
            }
        }

        __global__ void logical_not(Span<bool> output, View<bool> input)
        {
            for (auto id : grid_stride_range(output.size()))
                output[id] = !input[id];
        }
    }

    template <class Shape> static
    raw::BroadcastIndexLayout make_broadcast_layout(Shape out_shape, std::vector<Shape> in_shapes)
    {
        CV_Assert(in_shapes.size() <= 3);

        if (out_shape.empty())
            out_shape.assign(1, 1);
        bool same = true;
        for (Shape& shape : in_shapes) {
            if (shape.empty())
                shape.assign(1, 1);
            same = same && shape == out_shape;
        }
        if (same) {
            std::size_t total = 1;
            for (auto dim : out_shape)
                total *= dim;
            out_shape.assign(1, total);
            for (Shape& shape : in_shapes)
                shape = out_shape;
        }

        const int rank = static_cast<int>(out_shape.size());
        CV_Assert(rank >= 1 && rank <= CSL_MAX_TENSOR_RANK);

        raw::BroadcastIndexLayout layout;
        layout.rank = rank;
        for (int axis = 0; axis < rank; axis++)
            layout.out_dims[axis] = static_cast<std::int64_t>(out_shape[axis]);

        for (std::size_t k = 0; k < in_shapes.size(); k++) {
            const Shape& shape = in_shapes[k];
            CV_Assert(shape.size() <= out_shape.size());
            const int pad = rank - static_cast<int>(shape.size());
            std::int64_t stride = 1;
            for (int axis = rank - 1; axis >= 0; --axis) {
                const std::int64_t dim = axis < pad ? 1 : static_cast<std::int64_t>(shape[axis - pad]);
                CV_Assert(dim == 1 || dim == layout.out_dims[axis]);
                layout.strides[k][axis] = dim == 1 ? 0 : stride;
                stride *= dim;
            }
        }
        return layout;
    }

    template <class T, class Functor> static
    void launch_binary_to_bool(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y)
    {
        if (output.size() == 0)
            return;

        auto layout = make_broadcast_layout(output.shape_as_vector(), {x.shape_as_vector(), y.shape_as_vector()});
        auto kernel = raw::binary_to_bool<T, Functor>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, Span<bool>(output), View<T>(x), View<T>(y), layout);
    }

    template <class T>
    void compare_equal(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::EqualFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_greater(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::GreaterFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_greater_equal(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::GreaterEqualFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_less(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::LessFunctor>(stream, output, x, y);
    }

    template <class T>
    void compare_less_equal(const Stream& stream, TensorSpan<bool> output, TensorView<T> x, TensorView<T> y) {
        launch_binary_to_bool<T, raw::LessEqualFunctor>(stream, output, x, y);
    }

    void logical_and(const Stream& stream, TensorSpan<bool> output, TensorView<bool> x, TensorView<bool> y) {
        launch_binary_to_bool<bool, raw::AndFunctor>(stream, output, x, y);
    }

    void logical_or(const Stream& stream, TensorSpan<bool> output, TensorView<bool> x, TensorView<bool> y) {
        launch_binary_to_bool<bool, raw::OrFunctor>(stream, output, x, y);
    }

    void logical_xor(const Stream& stream, TensorSpan<bool> output, TensorView<bool> x, TensorView<bool> y) {
        launch_binary_to_bool<bool, raw::XorFunctor>(stream, output, x, y);
    }

    void logical_not(const Stream& stream, TensorSpan<bool> output, TensorView<bool> input) {
        CV_Assert(output.size() == input.size());
        if (output.size() == 0)
            return;

        auto kernel = raw::logical_not;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, Span<bool>(output), View<bool>(input));
    }

    template <class T>
    void where(const Stream& stream, TensorSpan<T> output, TensorView<bool> cond, TensorView<T> x, TensorView<T> y) {
        if (output.size() == 0)
            return;

        auto layout = make_broadcast_layout(output.shape_as_vector(),
                                            {cond.shape_as_vector(), x.shape_as_vector(), y.shape_as_vector()});
        auto kernel = raw::where<T>;
        auto policy = make_policy(kernel, output.size(), 0, stream);
        launch_kernel(kernel, policy, Span<T>(output), View<bool>(cond), View<T>(x), View<T>(y), layout);
    }

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 530)
    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<__half>, TensorView<__half>);
    template void where(const Stream&, TensorSpan<__half>, TensorView<bool>, TensorView<__half>, TensorView<__half>);
#endif
    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<float>, TensorView<float>);
    template void where(const Stream&, TensorSpan<float>, TensorView<bool>, TensorView<float>, TensorView<float>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<int8_t>, TensorView<int8_t>);
    template void where(const Stream&, TensorSpan<int8_t>, TensorView<bool>, TensorView<int8_t>, TensorView<int8_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<uint8_t>, TensorView<uint8_t>);
    template void where(const Stream&, TensorSpan<uint8_t>, TensorView<bool>, TensorView<uint8_t>, TensorView<uint8_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<int32_t>, TensorView<int32_t>);
    template void where(const Stream&, TensorSpan<int32_t>, TensorView<bool>, TensorView<int32_t>, TensorView<int32_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<int64_t>, TensorView<int64_t>);
    template void where(const Stream&, TensorSpan<int64_t>, TensorView<bool>, TensorView<int64_t>, TensorView<int64_t>);

    template void compare_equal(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_greater(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_greater_equal(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_less(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void compare_less_equal(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>);
    template void where(const Stream&, TensorSpan<bool>, TensorView<bool>, TensorView<bool>, TensorView<bool>);

}}}} /* namespace cv::dnn::cuda4dnn::kernels */
