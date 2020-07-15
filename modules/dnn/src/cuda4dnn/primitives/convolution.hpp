// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/scale_shift.hpp"
#include "../kernels/activations.hpp"
#include "../kernels/activation_eltwise.hpp"
#include "../kernels/bias_activation.hpp"
#include "../kernels/bias_eltwise_activation.hpp"
#include "../kernels/bias_activation_eltwise.hpp"
#include "../kernels/activation_eltwise.hpp"
#include "../kernels/eltwise_activation.hpp"
#include "../kernels/eltwise_ops.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>

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

        enum class FusionMode {
            NONE,
            ACTIVATION, /* act(conv) */
            ELTWISE_SUM, /* eltwise + conv */ /* eltwise tensor is passed as second input to forward */
            ELTWISE_SUM_THEN_ACTIVATION, /* act(conv + eltwise) */
            ACTIVATION_THEN_ELTWISE_SUM, /* act(conv) + eltwise */
        };

        FusionMode fusion_mode;

        enum class ActivationType {
            IDENTITY,
            RELU, /* uses value provided in `relu_negative_slope` */
            CLIPPED_RELU, /* uses values provided in `crelu_floor` and `crelu_ceil` */
            POWER, /* scale and shift fused with weights and bias; only `power_exp` is handled here */
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

        ConvolutionOp(csl::Stream stream_, csl::cudnn::Handle handle_, const ConvolutionConfiguration& config, const Mat& filters, const Mat& bias)
            : stream(std::move(stream_)), cudnnHandle(std::move(handle_))
        {
            const auto& kernel_size = config.kernel_size;
            const auto& dilations = config.dilations;
            const auto& strides = config.strides;

            const auto convolution_order = kernel_size.size();
            CV_Assert(convolution_order == dilations.size());
            CV_Assert(convolution_order == strides.size());

            const auto& input_shape = config.input_shape;
            const auto& output_shape = config.output_shape;
            CV_Assert(input_shape.size() == output_shape.size());
            CV_Assert(input_shape.size() == convolution_order + 2);

            const auto groups = config.groups;

            CV_Assert (1 < convolution_order && convolution_order <= 3);

            const auto rank = input_shape.size();
            const auto output_feature_maps = output_shape[1];
            const auto input_feature_maps = input_shape[1];
            const auto input_feature_maps_per_group = input_feature_maps / groups;
            CV_Assert(input_feature_maps % groups == 0);

            filtersTensor = csl::makeTensorHeader<T>(filters);
            csl::copyMatToTensor<T>(filters, filtersTensor, stream);

            if (!bias.empty())
            {
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

            auto is_not_zero = [](std::size_t i) { return i != 0; };
            if(std::any_of(std::begin(padding_left), std::end(padding_left), is_not_zero) ||
               std::any_of(std::begin(padding_right), std::end(padding_right), is_not_zero))
            {
                /* csl::Convolution supports symmetric padding only; hence, we deal with asymmetric padding by
                 * copying the input to a bigger tensor and padding the ends manually
                 */
                transformed_shape = input_shape;
                for (int i = 0; i < rank; i++)
                    transformed_shape[i] += padding_left[i] + padding_right[i];

                inputTransformer = csl::TensorTransform<T>(cudnnHandle, padding_left, padding_right);
            }

            typename csl::Convolution<T>::params_type params;
            if (transformed_shape.empty())
            {
                params.input_shape.assign(std::begin(input_shape), std::end(input_shape));
            }
            else
            {
                /* the convolution operation will be seeing the transformed input */
                params.input_shape.assign(std::begin(transformed_shape), std::end(transformed_shape));
            }

            auto& fshape = params.filter_shape;
            fshape.resize(rank);
            fshape[0] = output_feature_maps;
            fshape[1] = input_feature_maps_per_group;
            std::copy(std::begin(kernel_size), std::end(kernel_size), std::begin(fshape) + 2);
            CV_Assert(fshape.size() == kernel_size.size() + 2);

            params.padding.assign(std::begin(common_padding) + 2, std::end(common_padding));
            params.stride = strides;
            params.dilation = dilations;
            params.groups = config.groups;

            fusion_mode = config.fusion_mode;
            activation = config.activation_type;
            relu_negative_slope = config.relu_negative_slope;
            crelu_floor = config.crelu_floor;
            crelu_ceil = config.crelu_ceil;
            power_exp = config.power_exp;

            /* the scale and shift parameters of POWER have already been fused with weights and bias */
            if (activation == ConvolutionConfiguration::ActivationType::POWER && power_exp == 1.0f)
                activation = ConvolutionConfiguration::ActivationType::IDENTITY;

            /* we normally use cuDNN for convolution and perform bias, activation and eltwise ops ourselves
             * hence, the activation for cuDNN is IDENTITY by default
             */
            fusion_location = InternalFusionLocation::NATIVE; /* i.e. we perform bias, act and eltwise */
            params.eltwise = false;
            params.activation_type = csl::Convolution<T>::ActivationType::IDENTITY;

            /* cuDNN can fuse the operations with convolution in some cases; try if it's possible */
            if (!biasTensor.empty() && 0 &&
                 biasTensor.size() == output_feature_maps &&                       /* cuDNN requirement */
                 activation == ConvolutionConfiguration::ActivationType::RELU &&   /* cuDNN requirement */
                 relu_negative_slope == 0.0 &&                                     /* cuDNN requirement */
                 (fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION || /* act(conv + bias) */
                  fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION) /* act(conv + bias + eltwise) */
               )
            {
                bool do_not_fuse = false;
                if(std::is_same<T, half>::value)
                {
                    /* performance degrades if fused with tensor core based convolutions in most cases */
                    int device;
                    CUDA4DNN_CHECK_CUDA(cudaGetDevice(&device));

                    int cc_major;
                    CUDA4DNN_CHECK_CUDA(cudaDeviceGetAttribute(&cc_major, cudaDevAttrComputeCapabilityMajor, device));

                    if (cc_major >= 7)
                        do_not_fuse = true;
                }

                if (!do_not_fuse)
                {
                    fusion_location = InternalFusionLocation::CUDNN;
                    auto bias_shape = std::vector<std::size_t>(rank, 1);
                    bias_shape[1] = output_feature_maps;
                    params.bias_shape = bias_shape;

                    if (config.fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION)
                        params.eltwise = true;

                    params.activation_type = csl::Convolution<T>::ActivationType::RELU;
                }
            }

            convoluter = csl::Convolution<T>(cudnnHandle, params);

            csl::WorkspaceBuilder builder;
            if (!transformed_shape.empty())
            {
                auto& shape = transformed_shape;
                auto sz = std::accumulate(std::begin(shape), std::end(shape), 1, std::multiplies<std::size_t>());
                builder.require<T>(sz);
            }
            builder.require(convoluter.get_workspace_size());
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            /* input[0] = conv input, input[1] = bias (from fused eltwise layer) */
            CV_Assert(inputs.size() == 1 || inputs.size() == 2);
            CV_Assert(outputs.size() == 1);

            csl::WorkspaceAllocator allocator(workspace);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            if (!transformed_shape.empty())
            {
                auto& shape = transformed_shape;
                auto transformed_input = allocator.get_tensor_span<T>(std::begin(shape), std::end(shape));
                inputTransformer.transform(input, transformed_input);
                input = transformed_input;
            }

            auto conv_scratchpad = allocator.get_instance();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            if (fusion_location == InternalFusionLocation::CUDNN)
            {
                try
                {
                    if (fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION)
                        convoluter.convolve_with_bias_activation(output, input, filtersTensor, biasTensor, conv_scratchpad);
                    else if (fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION)
                    {
                        auto eltwise_wrapper = inputs[1].dynamicCast<wrapper_type>();
                        auto eltwise = eltwise_wrapper->getView();
                        CV_Assert(is_shape_same(eltwise, output));

                        convoluter.convolve_with_bias_eltwise_activation(output, input, filtersTensor, biasTensor, eltwise, conv_scratchpad);
                    }
                }
                catch(const csl::cudnn::cuDNNException& ex)
                {
                    if (ex.getCUDNNStatus() == CUDNN_STATUS_NOT_SUPPORTED)
                    {
                        /* drop cuDNN fusion and use the native fusion path */
                        fusion_location = InternalFusionLocation::NATIVE;
                    }
                    else
                        throw;
                }
            }

            if (fusion_location == InternalFusionLocation::NATIVE)
            {
                convoluter.convolve(output, input, filtersTensor, conv_scratchpad);

                if (fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM ||
                    fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION ||
                    fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM)
                {
                    CV_Assert(inputs.size() == 2);
                }

                if (!biasTensor.empty() && inputs.size() == 2)
                {
                    /* bias and eltwise */
                    CV_Assert(fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM ||
                              fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION ||
                              fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM);

                    auto eltwise_wrapper = inputs[1].dynamicCast<wrapper_type>();
                    auto eltwise = eltwise_wrapper->getView();
                    CV_Assert(is_shape_same(eltwise, output));

                    std::size_t inner_size = output.size_range(2, output.rank());

                    if (fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM)
                    {
                        kernels::biasN_eltwise_sum_2_identity_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                    }
                    else if (fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION)
                    {
                        /* activation(conv + bias + eltwise) */
                        switch (activation)
                        {
                        case ConvolutionConfiguration::ActivationType::IDENTITY:
                            kernels::biasN_eltwise_sum_2_identity_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::RELU:
                            kernels::biasN_eltwise_sum_2_relu_inplace<T>(stream, output, inner_size, biasTensor, eltwise, relu_negative_slope);
                            break;
                        case ConvolutionConfiguration::ActivationType::CLIPPED_RELU:
                            kernels::biasN_eltwise_sum_2_clipped_relu_inplace<T>(stream, output, inner_size, biasTensor, eltwise, crelu_floor, crelu_ceil);
                            break;
                        case ConvolutionConfiguration::ActivationType::POWER:
                            kernels::biasN_eltwise_sum_2_power_inplace<T>(stream, output, inner_size, biasTensor, eltwise, power_exp, 1.0, 0.0);
                            break;
                        case ConvolutionConfiguration::ActivationType::TANH:
                            kernels::biasN_eltwise_sum_2_tanh_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SIGMOID:
                            kernels::biasN_eltwise_sum_2_sigmoid_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SWISH:
                            kernels::biasN_eltwise_sum_2_swish_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::MISH:
                            kernels::biasN_eltwise_sum_2_mish_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        }
                    }
                    else if (fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM)
                    {
                        /* activation(conv + bias) + eltwise */
                        switch (activation)
                        {
                        case ConvolutionConfiguration::ActivationType::IDENTITY:
                            kernels::biasN_eltwise_sum_2_identity_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::RELU:
                            kernels::biasN_relu_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise, relu_negative_slope);
                            break;
                        case ConvolutionConfiguration::ActivationType::CLIPPED_RELU:
                            kernels::biasN_clipped_relu_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise, crelu_floor, crelu_ceil);
                            break;
                        case ConvolutionConfiguration::ActivationType::POWER:
                            kernels::biasN_power_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise, power_exp, 1.0, 0.0);
                            break;
                        case ConvolutionConfiguration::ActivationType::TANH:
                            kernels::biasN_tanh_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SIGMOID:
                            kernels::biasN_sigmoid_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SWISH:
                            kernels::biasN_swish_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::MISH:
                            kernels::biasN_mish_eltwise_sum_2_inplace<T>(stream, output, inner_size, biasTensor, eltwise);
                            break;
                        }
                    }
                }
                else if (!biasTensor.empty() && inputs.size() == 1)
                {
                    /* bias but no eltwise */
                    CV_Assert(fusion_mode == ConvolutionConfiguration::FusionMode::NONE ||
                              fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION);

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
                            kernels::biasN_power_inplace<T>(stream, output, inner_size, biasTensor, power_exp, 1.0, 0.0);
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
                else if (biasTensor.empty() && inputs.size() == 2)
                {
                    /* no bias but eltwise */
                    CV_Assert(fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM ||
                              fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION ||
                              fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM);

                    auto eltwise_wrapper = inputs[1].dynamicCast<wrapper_type>();
                    auto eltwise = eltwise_wrapper->getView();
                    CV_Assert(is_shape_same(eltwise, output));

                    /* we pass `eltwise` as `bias` (with `inner_size` as one) to bias-activation kernels */

                    if (fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM)
                    {
                        kernels::eltwise_sum_2<T>(stream, output, output, eltwise);
                    }
                    else if (fusion_mode == ConvolutionConfiguration::FusionMode::ELTWISE_SUM_THEN_ACTIVATION)
                    {
                        switch (activation)
                        {
                        case ConvolutionConfiguration::ActivationType::IDENTITY:
                            kernels::eltwise_sum_2<T>(stream, output, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::RELU:
                            kernels::eltwise_sum_2_relu<T>(stream, output, output, eltwise, relu_negative_slope);
                            break;
                        case ConvolutionConfiguration::ActivationType::CLIPPED_RELU:
                            kernels::eltwise_sum_2_clipped_relu<T>(stream, output, output, eltwise, crelu_floor, crelu_ceil);
                            break;
                        case ConvolutionConfiguration::ActivationType::POWER:
                            kernels::eltwise_sum_2_power<T>(stream, output, output, eltwise, power_exp, 1.0, 0.0);
                            break;
                        case ConvolutionConfiguration::ActivationType::TANH:
                            kernels::eltwise_sum_2_tanh<T>(stream, output, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SIGMOID:
                            kernels::eltwise_sum_2_sigmoid<T>(stream, output, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SWISH:
                            kernels::eltwise_sum_2_swish<T>(stream, output, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::MISH:
                            kernels::eltwise_sum_2_mish<T>(stream, output, output, eltwise);
                            break;
                        }
                    }
                    else if (fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION_THEN_ELTWISE_SUM)
                    {
                        switch (activation)
                        {
                        case ConvolutionConfiguration::ActivationType::IDENTITY:
                            kernels::eltwise_sum_2<T>(stream, output, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::RELU:
                            kernels::relu_eltwise_sum_2_inplace<T>(stream, output, eltwise, relu_negative_slope);
                            break;
                        case ConvolutionConfiguration::ActivationType::CLIPPED_RELU:
                            kernels::clipped_relu_eltwise_sum_2_inplace<T>(stream, output, eltwise, crelu_floor, crelu_ceil);
                            break;
                        case ConvolutionConfiguration::ActivationType::POWER:
                            kernels::power_eltwise_sum_2_inplace<T>(stream, output, eltwise, power_exp, 1.0, 0.0);
                            break;
                        case ConvolutionConfiguration::ActivationType::TANH:
                            kernels::tanh_eltwise_sum_2_inplace<T>(stream, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SIGMOID:
                            kernels::sigmoid_eltwise_sum_2_inplace<T>(stream, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::SWISH:
                            kernels::swish_eltwise_sum_2_inplace<T>(stream, output, eltwise);
                            break;
                        case ConvolutionConfiguration::ActivationType::MISH:
                            kernels::mish_eltwise_sum_2_inplace<T>(stream, output, eltwise);
                            break;
                        }
                    }
                }
                else if(biasTensor.empty() && inputs.size() == 1)
                {
                    /* no bias and no eltwise */
                    CV_Assert(fusion_mode == ConvolutionConfiguration::FusionMode::NONE ||
                              fusion_mode == ConvolutionConfiguration::FusionMode::ACTIVATION);

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

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        csl::cudnn::Handle cudnnHandle;
        csl::Tensor<T> filtersTensor, biasTensor;
        csl::Convolution<T> convoluter;

        std::vector<std::size_t> transformed_shape;
        csl::TensorTransform<T> inputTransformer;

        std::size_t scratch_mem_in_bytes;

        ConvolutionConfiguration::FusionMode fusion_mode;
        ConvolutionConfiguration::ActivationType activation;
        float relu_negative_slope, crelu_floor, crelu_ceil, power_exp;

        enum class InternalFusionLocation {
            CUDNN,
            NATIVE
        } fusion_location;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CONVOLUTION_HPP */
