// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_CUDNN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_CUDNN_HPP

#include "../../csl/workspace.hpp"

#include "../../csl/cudnn/cudnn.hpp"
#include "../../csl/cudnn/convolution.hpp"
#include "../../csl/tensor.hpp"
#include "../../csl/tensor_ops.hpp"

#include <cstddef>
#include <vector>
#include <algorithm>
#include <numeric>

namespace cv { namespace dnn { namespace cuda4dnn {

template <class T>
class cuDNNConvolution {
    using TensorDescriptor = csl::cudnn::TensorDescriptor<T>;
    using FilterDescriptor = csl::cudnn::FilterDescriptor<T>;
    using ConvolutionDescriptor = csl::cudnn::ConvolutionDescriptor<T>;
    using ConvolutionAlgorithm = csl::cudnn::ConvolutionAlgorithm;
    using ActivationDescriptor = csl::cudnn::ActivationDescriptor;
    using MathType = csl::cudnn::MathType;

public:
    using ActivationType = ActivationDescriptor::ActivationType;

    struct params_type {
        std::vector<std::size_t> input_shape;
        std::vector<std::size_t> filter_shape;
        std::vector<std::size_t> padding_left;
        std::vector<std::size_t> padding_right;
        std::vector<std::size_t> stride;
        std::vector<std::size_t> dilation;
        std::size_t groups;

        ConvolutionAlgorithm algorithm;
        MathType math_type;

        /* bias and activation (only RELU supported) */
        std::vector<std::size_t> bias_shape;
        ActivationType activation_type; /* MUST BE identity if there is no bias and ReLU if there is bias */
    };

    cuDNNConvolution() = default;
    cuDNNConvolution(const cuDNNConvolution&) = delete;
    cuDNNConvolution(cuDNNConvolution&&) = default;
    cuDNNConvolution(csl::cudnn::Handle handle, csl::WorkspaceBuilder& builder, const params_type& params) {
        cudnnHandle = std::move(handle);

        const auto& input_shape = params.input_shape;
        auto padding_left = params.padding_left;
        auto padding_right = params.padding_right;

        std::vector<std::size_t> common_padding;
        for (int i = 0; i < padding_left.size(); i++)
        {
            const auto sym_padding = std::min(padding_left[i], padding_right[i]);
            padding_left[i] -= sym_padding;
            padding_right[i] -= sym_padding;
            common_padding.push_back(sym_padding);
        }

        auto is_not_zero = [](std::size_t i) { return i != 0; };
        if(std::any_of(std::begin(padding_left), std::end(padding_left), is_not_zero) ||
            std::any_of(std::begin(padding_right), std::end(padding_right), is_not_zero))
        {
            /* cuDNN does not support asymmetric padding. We deal with this by copying the input
                * to a bigger tensor and manually padding the ends.
                */
            transformed_input_shape = input_shape;
            for (int i = 0; i < padding_left.size(); i++)
                transformed_input_shape[i + 2] += padding_left[i] + padding_right[i];

            std::vector<std::size_t> tpad_left(input_shape.size(), 0), tpad_right(input_shape.size(), 0);
            for (int i = 0; i < padding_left.size(); i++)
            {
                tpad_left[i + 2] = padding_left[i];
                tpad_right[i + 2] = padding_right[i];
            }
            inputTransformer = csl::TensorTransform<T>(cudnnHandle, tpad_left, tpad_right);
        }

        // cuDNN convolution API sees the transformed input if the input was manually padded
        auto conv_input_shape = transformed_input_shape.empty() ? input_shape : transformed_input_shape;
        inputTensorDesc = TensorDescriptor(conv_input_shape);
        filterDesc = FilterDescriptor(params.filter_shape);
        convDesc = ConvolutionDescriptor(common_padding, params.stride, params.dilation, params.groups, params.math_type);

        if (!params.bias_shape.empty()) {
            biasTensorDesc = TensorDescriptor(params.bias_shape);
            activationDesc = ActivationDescriptor(params.activation_type, 0.0);
        } else {
            CV_Assert(params.activation_type == ActivationType::IDENTITY);
        }

        std::vector<int> output_dims;
        getConvolutionForwardOutputDim(convDesc, filterDesc, inputTensorDesc, output_dims);
        outputTensorDesc = TensorDescriptor(output_dims);

        if (!transformed_input_shape.empty())
        {
            const auto& shape = transformed_input_shape;
            const auto sz = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<std::size_t>());
            builder.require<T>(sz);
        }

        algorithm = params.algorithm;
        conv_workspace_size_in_bytes = get_convolution_workspace_size(cudnnHandle, convDesc, filterDesc, inputTensorDesc, outputTensorDesc, algorithm);
        builder.require(conv_workspace_size_in_bytes);
    }

    cuDNNConvolution& operator=(const cuDNNConvolution&) = delete;
    cuDNNConvolution& operator=(cuDNNConvolution&&) = default;

    void convolve(csl::TensorSpan<T> output, csl::TensorView<T> input, csl::TensorView<T> filters, csl::WorkspaceInstance scratchpad) {
        csl::WorkspaceAllocator allocator(scratchpad);

        if (!transformed_input_shape.empty()) {
            auto& shape = transformed_input_shape;
            auto transformed_input = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
            inputTransformer.transform(input, transformed_input);
            input = transformed_input;
        }

        auto conv_scratchpad = allocator.get_instance(conv_workspace_size_in_bytes);
        csl::cudnn::convolve<T>(
            cudnnHandle,
            convDesc, algorithm, conv_scratchpad,
            filterDesc, filters.get(),
            inputTensorDesc, input.get(),
            1.0, 0.0, outputTensorDesc, output.get()
        );
    }

    void convolve_with_bias_activation(csl::TensorSpan<T> output, csl::TensorView<T> input, csl::TensorView<T> filters, csl::TensorView<T> bias, csl::WorkspaceInstance scratchpad) {
        csl::WorkspaceAllocator allocator(scratchpad);

        if (!transformed_input_shape.empty()) {
            auto& shape = transformed_input_shape;
            auto transformed_input = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
            inputTransformer.transform(input, transformed_input);
            input = transformed_input;
        }

        auto conv_scratchpad = allocator.get_instance(conv_workspace_size_in_bytes);
        csl::cudnn::convolve_with_bias_activation<T>(
            cudnnHandle,
            1.0, convDesc, algorithm, conv_scratchpad,
            filterDesc, filters.get(),
            inputTensorDesc, input.get(),
            biasTensorDesc, bias.get(),
            activationDesc,
            outputTensorDesc, output.get()
        );
    }

private:
    csl::cudnn::Handle cudnnHandle;
    TensorDescriptor inputTensorDesc, outputTensorDesc;
    FilterDescriptor filterDesc;
    ConvolutionDescriptor convDesc;
    TensorDescriptor biasTensorDesc;
    ActivationDescriptor activationDesc;

    ConvolutionAlgorithm algorithm;
    std::size_t conv_workspace_size_in_bytes;

    std::vector<std::size_t> transformed_input_shape;
    csl::TensorTransform<T> inputTransformer;
};

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_CUDNN_HPP */