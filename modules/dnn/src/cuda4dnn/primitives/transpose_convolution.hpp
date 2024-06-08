// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRANSPOSE_CONVOLUTION_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRANSPOSE_CONVOLUTION_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include "../kernels/scale_shift.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct TransposeConvolutionConfiguration {
        /* other than `input_shape` and `output_shape`, all the configuration values must be provided
         * for the corresponding convolution operation (not transpose convolution)
         */

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
    };

    template <class T>
    class TransposeConvolutionOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        TransposeConvolutionOp(csl::Stream stream_, csl::cudnn::Handle handle, const TransposeConvolutionConfiguration& config, const Mat& filters, const Mat& bias)
            : stream(std::move(stream_)), cudnnHandle(std::move(handle))
        {
            /* we make use of backward pass of convolution to perform forward pass of transpose convolution
             * hence, we must setup configuration for the convolution operation and perform backward pass
             */
            const auto& kernel_size = config.kernel_size;
            const auto& dilations = config.dilations;
            const auto& strides = config.strides;

            const auto convolution_order = kernel_size.size();
            CV_Assert(convolution_order >= 1);

            CV_Assert(convolution_order == dilations.size());
            CV_Assert(convolution_order == strides.size());

            const auto& input_shape = config.input_shape;
            const auto& output_shape = config.output_shape;
            CV_Assert(input_shape.size() == output_shape.size());
            CV_Assert(input_shape.size() == convolution_order + 2);

            const auto groups = config.groups;

            if (convolution_order > 3)
                CV_Error(Error::StsNotImplemented, "Only 1D/2D/3D transpose convolution is supported.");

            const auto rank = input_shape.size();
            const auto input_feature_maps = input_shape[1];
            const auto output_feature_maps = output_shape[1];
            const auto output_feature_maps_per_group = output_feature_maps / groups;
            CV_Assert(output_feature_maps % groups == 0);

            filtersTensor = csl::makeTensorHeader<T>(filters);
            csl::copyMatToTensor<T>(filters, filtersTensor, stream);

            if (!bias.empty())
            {
                CV_Assert(bias.total() == output_feature_maps);
                biasTensor = csl::makeTensorHeader<T>(bias);
                csl::copyMatToTensor<T>(bias, biasTensor, stream);
            }

            /* left and right are misleading as the padding is applicable for any number of dimensions
             * but we use those identifiers to avoid confusion with `pads_begin` and `pads_end`
             *
             * `common_padding` contains the amount of padding that has to be added to both sides
             * `padding_left` and `padding_right` contains the amount of padding that needs to be added
             * to a particular side in addition to the common padding
             *
             * note that we compute the padding for the convolution operation
             */
            std::vector<std::size_t> common_padding(rank, 0);
            std::vector<std::size_t> padding_left(rank, 0), padding_right(rank, 0);
            if (config.padMode == TransposeConvolutionConfiguration::PaddingMode::MANUAL)
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
            else if (config.padMode == TransposeConvolutionConfiguration::PaddingMode::VALID)
            {
                /* nothing to do as the paddings are already preset to zero */
            }
            else if (config.padMode == TransposeConvolutionConfiguration::PaddingMode::SAME)
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
                        std::max<std::int64_t>(0, (input_shape[i] - 1) * strides[j] + effective_kernel_size - output_shape[i]);

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
                CV_Error(Error::StsNotImplemented, "Padding configuration requires asymmetric padding and hence is not supported.");
            }

            typename csl::TransposeConvolution<T>::params_type params;
            params.input_shape.assign(std::begin(input_shape), std::end(input_shape));
            params.output_shape.assign(std::begin(output_shape), std::end(output_shape));

            auto& fshape = params.filter_shape;
            fshape.resize(rank);
            fshape[0] = input_feature_maps;
            fshape[1] = output_feature_maps_per_group;
            std::copy(std::begin(kernel_size), std::end(kernel_size), std::begin(fshape) + 2);
            CV_Assert(fshape.size() == kernel_size.size() + 2);

            params.padding.assign(std::begin(common_padding) + 2, std::end(common_padding));
            params.stride = strides;
            params.dilation = dilations;
            params.groups = config.groups;

            convoluter = csl::TransposeConvolution<T>(cudnnHandle, params);

            csl::WorkspaceBuilder builder;
            builder.require(convoluter.get_workspace_size());
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

            csl::WorkspaceAllocator allocator(workspace);
            convoluter.transpose_convolve(output, input, filtersTensor, allocator.get_instance());
            if (!biasTensor.empty())
            {
                std::size_t inner_size = total(output_wrapper->getShape(), 2, -1);
                kernels::biasN<T>(stream, output, output, inner_size, biasTensor);
            }
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        csl::cudnn::Handle cudnnHandle;
        csl::Tensor<T> filtersTensor, biasTensor;
        csl::TransposeConvolution<T> convoluter;

        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_TRANSPOSE_CONVOLUTION_HPP */
