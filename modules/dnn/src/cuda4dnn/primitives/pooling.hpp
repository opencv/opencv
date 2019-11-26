// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_POOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>
#include <utility>
#include <algorithm>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct PoolingConfiguration {
        enum class PoolingMode {
            MAX,
            AVERAGE_INCLUDE_PADDING, /* include padding while calculating average */
            AVERAGE_EXCLUDE_PADDING /* exclude padding while calculating average */
        };

        PoolingMode poolMode;

        /* the size of the following vectors must be equal to the window size */
        std::vector<std::size_t> window_size;
        std::vector<std::size_t> strides;

        enum class PaddingMode {
            MANUAL, /* uses explicit padding values provided in `pads_begin` and `pads_end` */
            VALID, /* no padding is added */
            SAME /* TensorFlow logic is used for same padding */
        };

        PaddingMode padMode;

        /* explicit paddings are used if and only if padMode is set to manual */
        std::vector<std::size_t> pads_begin, pads_end;

        /* the output shape is calculated using the following formula:
         * output_dim = func[(input_dim + padding_left + padding_right - kernel_dim)/stride] + 1
         *
         * rounding mode decides what is used as `func`
         */
        enum class RoundingMode {
            CEIL, /* uses ceil */
            FLOOR
        };

        RoundingMode roundMode;

        /* full shape inclusive of channel and batch axis */
        std::vector<std::size_t> input_shape;
    };

    template <class T>
    class PoolingOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        PoolingOp(csl::cudnn::Handle handle, const PoolingConfiguration& config)
            : cudnnHandle(std::move(handle))
        {
            const auto& window_size = config.window_size;

            const auto pooling_order = window_size.size();
            CV_Assert(pooling_order >= 1);

            const auto& strides = config.strides;
            CV_Assert(pooling_order == strides.size());

            const auto& input_shape = config.input_shape;
            CV_Assert(input_shape.size() == pooling_order + 2);

            if (pooling_order > 3)
                CV_Error(Error::StsNotImplemented, "Only 1D/2D/3D pooling are supported.");

            const auto rank = input_shape.size();

            /* left and right are misleading as the padding is applicable for any number of dimensions
             * but we use those identifiers to avoid confusion with `pads_begin` and `pads_end`
             *
             * `common_padding` contains the amount of padding that has to be added to both sides
             * `padding_left` and `padding_right` contains the amount of padding that needs to be added
             * to a particular side in addition to the common padding
             */
            std::vector<std::size_t> common_padding(rank, 0);
            std::vector<std::size_t> padding_left(rank, 0), padding_right(rank, 0);
            if (config.padMode == PoolingConfiguration::PaddingMode::MANUAL)
            {
                const auto& pads_begin = config.pads_begin;
                const auto& pads_end = config.pads_end;

                CV_Assert(pooling_order == pads_begin.size());
                CV_Assert(pooling_order == pads_end.size());

                /* cuDNN rounds down by default; hence, if ceilMode is false, we do nothing
                 * otherwise, we add extra padding towards the end so that the convolution arithmetic yeilds
                 * the correct output size without having to deal with fancy fractional sizes
                 */
                auto pads_end_modified = pads_end;
                if (config.roundMode == PoolingConfiguration::RoundingMode::CEIL)
                {
                    for (int i = 0; i < window_size.size(); i++) {
                        auto rem = (input_shape[i + 2] + pads_begin[i] + pads_end[i] - window_size[i]) % strides[i];
                        if (rem)
                            pads_end_modified[i] += strides[i] - rem;
                    }
                }

                for (int i = 2; i < common_padding.size(); i++)
                {
                    common_padding[i] = std::min(pads_begin[i - 2], pads_end_modified[i - 2]);
                    padding_left[i] = pads_begin[i - 2] - common_padding[i];
                    padding_right[i] = pads_end_modified[i - 2] - common_padding[i];
                }
            }
            else if (config.padMode == PoolingConfiguration::PaddingMode::VALID)
            {
                /* nothing to do as the paddings are already preset to zero */
            }
            else if (config.padMode == PoolingConfiguration::PaddingMode::SAME)
            {
                /* TensorFlow Logic:
                 * total_padding[i] = (o[i] - 1) * s[i] + effective_k[i] - i[i]
                 *
                 * if total padding is odd, the extra is added towards the end
                 */
                for (int i = 2; i < rank; i++)
                {
                    const auto j = i - 2; /* filter index */
                    const auto output_dim = (input_shape[i] - 1 + strides[j]) / strides[j];
                    const auto required_total_padding =
                        std::max<std::int64_t>(0, (output_dim - 1) * strides[j] + window_size[j] - input_shape[i]);

                    common_padding[i] = required_total_padding / 2;
                    padding_left[i] = 0;
                    padding_right[i] = required_total_padding % 2;
                }
            }

            /* in some scenarios, the extra padding at the end may not change the output at all */
            for (int i = 2; i < rank; i++) {
                const auto j = i - 2; /* filter idx */
                const auto total_padding = common_padding[i] * 2 + padding_left[i] + padding_right[i];
                std::int64_t rem = (input_shape[i] + total_padding - window_size[j]) % strides[j];

                /* the output shape doesn't change if we decrease the total padding by at most `rem`
                 * provided that we decrease from the right
                 */
                if (rem && padding_right[i] > 0)
                    padding_right[i] = std::max<std::int64_t>(0, padding_right[i] - rem);
            }

            auto is_not_zero = [](std::size_t i) { return i != 0; };
            if (std::any_of(std::begin(padding_left), std::end(padding_left), is_not_zero) ||
                std::any_of(std::begin(padding_right), std::end(padding_right), is_not_zero))
            {
                /* csl::Pooling does not fully support asymmetric padding; hence, we deal with asymmetric padding by
                 * copying the input to a bigger tensor and padding the ends manually
                 *
                 * But we first try to avoid the transformation using cuDNN's flexibility. cuDNN can accept a smaller or
                 * a bigger output shape. This effectively allows us to have arbitrary padding at the right.
                 */
                if (std::any_of(std::begin(padding_left), std::end(padding_left), is_not_zero))
                {
                    /* there is padding on the left and we are forced to transform */
                    auto transformed_input_shape = input_shape;
                    for (int i = 0; i < rank; i++)
                        transformed_input_shape[i] += padding_left[i] + padding_right[i];

                    transformedInput.resize(std::begin(transformed_input_shape), std::end(transformed_input_shape));
                    inputTransformer = csl::TensorTransform<T>(cudnnHandle, padding_left, padding_right);
                }
            }

            typename csl::Pooling<T>::params_type params;
            if (transformedInput.empty())
            {
                /* no transform => use original input shape */
                params.input_shape.assign(std::begin(input_shape), std::end(input_shape));
            }
            else
            {
                /* the pooling operation will be seeing the transformed input */
                auto transformed_input_shape = transformedInput.shape_as_vector();
                params.input_shape.assign(std::begin(transformed_input_shape), std::end(transformed_input_shape));
            }

            auto output_shape = input_shape;
            for (int i = 2; i < rank; i++)
            {
                auto total_padding = common_padding[i] * 2 + padding_left[i] + padding_right[i];
                output_shape[i] = (params.input_shape[i] + total_padding - window_size[i - 2]) / strides[i - 2] + 1;
            }

            params.output_shape.assign(std::begin(output_shape), std::end(output_shape));
            params.window_size = window_size;
            params.padding.assign(std::begin(common_padding) + 2, std::end(common_padding));
            params.stride = strides;

            if (config.poolMode == PoolingConfiguration::PoolingMode::MAX)
            {
                params.type = csl::Pooling<T>::PoolingType::MAX;
            }
            else if (config.poolMode == PoolingConfiguration::PoolingMode::AVERAGE_INCLUDE_PADDING)
            {
                params.type = csl::Pooling<T>::PoolingType::AVERAGE_INCLUDE_PADDING;
            }
            else if (config.poolMode == PoolingConfiguration::PoolingMode::AVERAGE_EXCLUDE_PADDING)
            {
                params.type = csl::Pooling<T>::PoolingType::AVERAGE_EXCLUDE_PADDING;
            }

            pooler = csl::Pooling<T>(cudnnHandle, params);
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            if (!transformedInput.empty())
            {
                inputTransformer.transform(input, transformedInput);
                input = csl::TensorView<T>(transformedInput);
            }

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            pooler.pool(input, output);
        }

    private:
        csl::cudnn::Handle cudnnHandle;
        csl::Pooling<T> pooler;

        csl::Tensor<T> transformedInput;
        csl::TensorTransform<T> inputTransformer;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_POOLING_HPP */
