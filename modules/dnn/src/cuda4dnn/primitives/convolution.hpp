// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_HPP

#include "../../op_cuda.hpp"

#include "./convolution/cudnn.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/stream.hpp"
#include "../csl/event.hpp"
#include "../csl/tensor.hpp"
#include "../kernels/scale_shift.hpp"
#include "../kernels/activations.hpp"
#include "../kernels/bias_activation.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct ConvolutionConfiguration {
        /* the size of the following vectors must be equal to the kernel size */
        std::vector<std::size_t> kernel_size;
        std::vector<std::size_t> dilations, strides;

        enum class PaddingMode {
            MANUAL, /* uses explicit padding values provided in `pads_begin` and `pads_end` */
            VALID, /* no padding is added */
            SAME /* TensorFlow logic is used for same padding */
        };

        /* explicit paddings are used if and only if padMode is set to manual */
        PaddingMode padMode;
        std::vector<std::size_t> pads_begin, pads_end;

        /* full shape inclusive of channel and batch axis */
        std::vector<std::size_t> input_shape;
        std::vector<std::size_t> output_shape;

        /* group count for grouped convolution */
        std::size_t groups;

        enum class ActivationType {
            IDENTITY,
            RELU, /* uses value provided in `relu_negative_slope` */
            CLIPPED_RELU, /* uses values provided in `crelu_floor` and `crelu_ceil` */
            POWER, /* scale and shift fused beforehand (fuseWeights); only `power_exp` is handled by CUDA */
            TANH,
            SIGMOID,
            SWISH,
            MISH
        };

        ActivationType activation_type;
        float relu_negative_slope, crelu_floor, crelu_ceil, power_exp;
    };

    template <class T>
    class ConvolutionOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ConvolutionOp(csl::Stream stream_, csl::cudnn::Handle handle, const ConvolutionConfiguration& config, const Mat& filters, const Mat& bias)
            : stream(std::move(stream_)), cudnnHandle(std::move(handle))
        {
            const auto& kernel_size = config.kernel_size;
            const auto& dilations = config.dilations;
            const auto& strides = config.strides;

            const auto convolution_order = kernel_size.size();
            CV_Assert(convolution_order > 1);

            CV_Assert(convolution_order == dilations.size());
            CV_Assert(convolution_order == strides.size());

            const auto& input_shape = config.input_shape;
            const auto& output_shape = config.output_shape;
            CV_Assert(input_shape.size() == output_shape.size());
            CV_Assert(input_shape.size() == convolution_order + 2);

            const auto groups = config.groups;

            if (convolution_order > 3)
                CV_Error(Error::StsNotImplemented, "Only 2D/3D convolution is supported.");

            const auto rank = input_shape.size();
            const auto output_feature_maps = output_shape[1];
            const auto input_feature_maps = input_shape[1];
            const auto input_feature_maps_per_group = input_feature_maps / groups;
            CV_Assert(input_feature_maps % groups == 0);

            filtersTensor = csl::makeTensorHeader<T>(filters);
            csl::copyMatToTensor<T>(filters, filtersTensor, stream);

            std::vector<std::size_t> filter_shape(rank);
            filter_shape[0] = output_feature_maps;
            filter_shape[1] = input_feature_maps_per_group;
            std::copy(std::begin(kernel_size), std::end(kernel_size), std::begin(filter_shape) + 2);

            std::vector<std::size_t> bias_shape;
            if (!bias.empty())
            {
                bias_shape.resize(rank, 1);
                bias_shape[1] = output_feature_maps;
                biasTensor = csl::makeTensorHeader<T>(bias);
                csl::copyMatToTensor<T>(bias, biasTensor, stream);
            }

            /* left and right are misleading as the padding is applicable for any number of dimensions
             * but we use those identifiers to avoid confusion with `pads_begin` and `pads_end`
             *
             * `common_padding` contains the amount of padding that has to be added to both sides
             * `padding_left` and `padding_right` contains the amount of padding that needs to be added
             * to a particular side in addition to the common padding
             */
            std::vector<std::size_t> common_padding(rank, 0);
            std::vector<std::size_t> padding_left(rank, 0), padding_right(rank, 0);
            if (config.padMode == ConvolutionConfiguration::PaddingMode::MANUAL)
            {
                const auto& pads_begin = config.pads_begin;
                const auto& pads_end = config.pads_end;

                CV_Assert(convolution_order == pads_begin.size());
                CV_Assert(convolution_order == pads_end.size());

                for (int i = 2; i < common_padding.size(); i++)
                {
                    common_padding[i] = std::min(pads_begin[i - 2], pads_end[i - 2]);
                    padding_left[i] = pads_begin[i - 2] - common_padding[i];
                    padding_right[i] = pads_end[i - 2] - common_padding[i];
                }
            }
            else if (config.padMode == ConvolutionConfiguration::PaddingMode::VALID)
            {
                /* nothing to do as the paddings are already preset to zero */
            }
            else if (config.padMode == ConvolutionConfiguration::PaddingMode::SAME)
            {
                /* TensorFlow Logic:
                 * total_padding[i] = (o[i] - 1) * s[i] + effective_k[i] - i[i]
                 *
                 * if total padding is odd, the extra is added towards the end
                 */
                for (int i = 2; i < rank; i++)
                {
                    const auto j = i - 2; /* filter index */
                    const auto effective_kernel_size = dilations[j] * (kernel_size[j] - 1) + 1;
                    const auto required_total_padding =
                        std::max<std::int64_t>(0, (output_shape[i] - 1) * strides[j] + effective_kernel_size - input_shape[i]);

                    common_padding[i] = required_total_padding / 2;
                    padding_left[i] = 0;
                    padding_right[i] = required_total_padding % 2;
                }
            }

            /* in some scenarios, the extra padding at the end may not change the output at all */
            for (int i = 2; i < rank; i++) {
                const auto j = i - 2; /* filter idx */
                const auto total_padding = common_padding[i] * 2 + padding_left[i] + padding_right[i];
                const auto effective_kernel_size = dilations[j] * (kernel_size[j] - 1) + 1;
                std::int64_t rem = (input_shape[i] + total_padding - effective_kernel_size) % strides[j];

                /* the output shape doesn't change if we decrease the total padding by at most `rem`
                 * provided that we decrease from the right
                 */
                if (rem && padding_right[i] > 0)
                    padding_right[i] = std::max<std::int64_t>(0, padding_right[i] - rem);
            }

            /* POWER's scale and shift are already fused with the weights
             * Hence, if the `power_exp` is unity, it's effectively identity activation.
             */
            if (activation == ConvolutionConfiguration::ActivationType::POWER && power_exp == 1.0f)
                activation = ConvolutionConfiguration::ActivationType::IDENTITY;

            activation = config.activation_type;
            relu_negative_slope = config.relu_negative_slope;
            crelu_floor = config.crelu_floor;
            crelu_ceil = config.crelu_ceil;
            power_exp = config.power_exp;

            typename cuDNNConvolution<T>::params_type params;
            params.input_shape.assign(std::begin(input_shape), std::end(input_shape));
            params.filter_shape.assign(std::begin(filter_shape), std::end(filter_shape));
            for (int i = 2; i < input_shape.size(); i++)
            {
                params.padding_left.push_back(common_padding[i] + padding_left[i]);
                params.padding_right.push_back(common_padding[i] + padding_right[i]);
            }
            params.stride = strides;
            params.dilation = dilations;
            params.groups = config.groups;

            // Runtime Tuning
            {
                using csl::cudnn::ConvolutionAlgorithm;
                using csl::cudnn::MathType;

                struct algorithm_stats_t {
                    float mean_time, stddev_time;
                    std::size_t workspace_size; // in bytes

                    MathType math_type;
                    ConvolutionAlgorithm algorithm;
                    bool bias_activation_fused;
                };

                std::vector<algorithm_stats_t> stats;

                csl::Workspace workspace;
                csl::Tensor<T> dummy_input(std::begin(input_shape), std::end(input_shape));
                csl::Tensor<T> dummy_output(std::begin(output_shape), std::end(output_shape));

                auto start_event = csl::Event(true, true);
                auto end_event = csl::Event(true, true);

                auto profile_conv_fn = [&] {
                    if (cudnn_fused_bias_relu)
                    {
                        params.bias_shape = bias_shape;
                        params.activation_type = cuDNNConvolution<T>::ActivationType::RELU;
                    }
                    else
                    {
                        params.bias_shape.clear();
                        params.activation_type = cuDNNConvolution<T>::ActivationType::IDENTITY;
                    }

                    csl::WorkspaceBuilder builder;
                    auto convoluter = cuDNNConvolution<T>(cudnnHandle, builder, params);

                    /* This reallocates memory and hence can fail. It however provides a strong exception guarantee.
                     * Hence, workspace is still safe for use after an exception.
                     */
                    workspace.require(builder.required_workspace_size());

                    std::vector<float> runtimes;
                    for (int i = 0; i < 3; i++)
                    {
                        start_event.record(stream);
                        execute_cudnn_convolution(convoluter, dummy_output, dummy_input, workspace);
                        end_event.record(stream);

                        stream.synchronize();

                        auto time_in_ms = csl::TimeElapsedBetweenEvents(start_event, end_event);
                        runtimes.push_back(time_in_ms);
                    }

                    auto sum = std::accumulate(std::begin(runtimes), std::end(runtimes), 0.0f);
                    auto squared_sum = std::inner_product(std::begin(runtimes), std::end(runtimes), std::begin(runtimes), 0.0f);
                    auto mean = sum / runtimes.size();
                    auto stddev = std::sqrt(squared_sum / runtimes.size() - mean * mean);

                    algorithm_stats_t result;
                    result.algorithm = params.algorithm;
                    result.math_type = params.math_type;
                    result.bias_activation_fused = cudnn_fused_bias_relu;
                    result.mean_time = mean;
                    result.stddev_time = stddev;
                    result.workspace_size = builder.required_workspace_size();
                    stats.push_back(result);
                };

                // DEFAULT_MATH
                {
                    auto algos_to_try =
                    {
                        ConvolutionAlgorithm::IMPLICIT_GEMM,
                        ConvolutionAlgorithm::IMPLICIT_PRECOMP_GEMM,
                        ConvolutionAlgorithm::GEMM,
                        ConvolutionAlgorithm::DIRECT,
                        ConvolutionAlgorithm::WINOGRAD,
                        ConvolutionAlgorithm::WINOGRAD_NONFUSED,
                        ConvolutionAlgorithm::FFT,
                        ConvolutionAlgorithm::FFT_TILING
                    };

                    std::vector<bool> fusion_options_to_try;
                    fusion_options_to_try.push_back(false);
                    if (!bias_shape.empty() && activation == ConvolutionConfiguration::ActivationType::RELU && relu_negative_slope == 0.0)
                        fusion_options_to_try.push_back(true);

                    for (auto fusion : fusion_options_to_try)
                    {
                        for (auto algo : algos_to_try)
                        {
                            params.math_type = csl::cudnn::MathType::DEFAULT_MATH;
                            params.algorithm = algo;
                            cudnn_fused_bias_relu = fusion;

                            try {
                                profile_conv_fn();
                            } catch(...) {
                                // ignore
                            }
                        }
                    }
                }

                // TENSOR_OP
                if (std::is_same<T, half>::value)
                {
                    auto algos_to_try =
                    {
                        ConvolutionAlgorithm::IMPLICIT_PRECOMP_GEMM,
                        ConvolutionAlgorithm::WINOGRAD_NONFUSED
                    };

                    std::vector<bool> fusion_options_to_try;
                    fusion_options_to_try.push_back(false);
                    if (!bias_shape.empty() && activation == ConvolutionConfiguration::ActivationType::RELU && relu_negative_slope == 0.0)
                        fusion_options_to_try.push_back(true);

                    for (auto fusion : fusion_options_to_try)
                    {
                        for (auto algo : algos_to_try)
                        {
                            params.math_type = csl::cudnn::MathType::TENSOR_OP_MATH;
                            params.algorithm = algo;
                            cudnn_fused_bias_relu = fusion;

                            try {
                                profile_conv_fn();
                            } catch(...) {
                                // ignore
                            }
                        }
                    }
                }

                CV_Assert(!stats.empty());

                auto& best = *std::min_element(std::begin(stats), std::end(stats),
                    [] (const algorithm_stats_t& lhs, const algorithm_stats_t& rhs) { return lhs.mean_time < rhs.mean_time; });
                params.algorithm = best.algorithm;
                params.math_type = best.math_type;
                cudnn_fused_bias_relu = best.bias_activation_fused;
            }

            if (cudnn_fused_bias_relu)
            {
                params.bias_shape.assign(std::begin(bias_shape), std::end(bias_shape));
                params.activation_type = cuDNNConvolution<T>::ActivationType::RELU;
            }
            else
            {
                params.bias_shape.clear();
                params.activation_type = cuDNNConvolution<T>::ActivationType::IDENTITY;
            }

            csl::WorkspaceBuilder builder;
            cudnn_convoluter = cuDNNConvolution<T>(cudnnHandle, builder, params);
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            execute_cudnn_convolution(cudnn_convoluter, output, input, workspace);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        void execute_cudnn_convolution(cuDNNConvolution<T>& convoluter, csl::TensorSpan<T> output, csl::TensorView<T> input, csl::Workspace workspace)
        {
            csl::WorkspaceAllocator allocator(workspace);

            if (cudnn_fused_bias_relu)
            {
                convoluter.convolve_with_bias_activation(output, input, filtersTensor, biasTensor, allocator.get_instance());
            }
            else
            {
                convoluter.convolve(output, input, filtersTensor, allocator.get_instance());
                if (!biasTensor.empty())
                {
                    std::size_t inner_size = output.size_range(2, output.rank());
                    switch(activation)
                    {
                        case ConvolutionConfiguration::ActivationType::IDENTITY:
                            kernels::biasN<T>(stream, output, output, inner_size, biasTensor);
                            break;
                        case ConvolutionConfiguration::ActivationType::RELU:
                            kernels::biasN_relu_inplace<T>(stream, output, inner_size, biasTensor, relu_negative_slope);
                            break;
                        case ConvolutionConfiguration::ActivationType::CLIPPED_RELU:
                            kernels::biasN_clipped_relu_inplace<T>(stream, output, inner_size, biasTensor, crelu_floor, crelu_ceil);
                            break;
                        case ConvolutionConfiguration::ActivationType::POWER:
                            kernels::biasN_power_inplace<T>(stream, output, inner_size, biasTensor, power_exp, T(1.0), T(0.0));
                            break;
                        case ConvolutionConfiguration::ActivationType::TANH:
                            kernels::biasN_tanh_inplace<T>(stream, output, inner_size, biasTensor);
                            break;
                        case ConvolutionConfiguration::ActivationType::SIGMOID:
                            kernels::biasN_sigmoid_inplace<T>(stream, output, inner_size, biasTensor);
                            break;
                        case ConvolutionConfiguration::ActivationType::SWISH:
                            kernels::biasN_swish_inplace<T>(stream, output, inner_size, biasTensor);
                            break;
                        case ConvolutionConfiguration::ActivationType::MISH:
                            kernels::biasN_mish_inplace<T>(stream, output, inner_size, biasTensor);
                            break;
                    }
                }
                else
                {
                    switch(activation)
                    {
                        case ConvolutionConfiguration::ActivationType::IDENTITY:
                            break;
                        case ConvolutionConfiguration::ActivationType::RELU:
                            kernels::relu<T>(stream, output, output, relu_negative_slope);
                            break;
                        case ConvolutionConfiguration::ActivationType::CLIPPED_RELU:
                            kernels::clipped_relu<T>(stream, output, output, crelu_floor, crelu_ceil);
                            break;
                        case ConvolutionConfiguration::ActivationType::POWER:
                            kernels::power<T>(stream, output, output, power_exp, 1.0, 0.0);
                            break;
                        case ConvolutionConfiguration::ActivationType::TANH:
                            kernels::tanh<T>(stream, output, output);
                            break;
                        case ConvolutionConfiguration::ActivationType::SIGMOID:
                            kernels::sigmoid<T>(stream, output, output);
                            break;
                        case ConvolutionConfiguration::ActivationType::SWISH:
                            kernels::swish<T>(stream, output, output);
                            break;
                        case ConvolutionConfiguration::ActivationType::MISH:
                            kernels::mish<T>(stream, output, output);
                            break;
                    }
                }
            }
        }

        csl::Stream stream;
        csl::cudnn::Handle cudnnHandle;
        csl::Tensor<T> filtersTensor, biasTensor;

        std::size_t scratch_mem_in_bytes;

        ConvolutionConfiguration::ActivationType activation;
        float relu_negative_slope, crelu_floor, crelu_ceil, power_exp;

        bool cudnn_fused_bias_relu;
        cuDNNConvolution<T> cudnn_convoluter;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_HPP */
