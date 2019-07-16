// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_PRIMITIVES_NORMALIZE_BBOX_HPP
#define OPENCV_DNN_CUDA4DNN_PRIMITIVES_NORMALIZE_BBOX_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"
#include "../csl/kernels.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    struct NormalizeConfiguration {
        std::vector<std::size_t> input_shape;

        /* axis range across which values are normalized
         *
         * [0, axis_start) = outer range
         * [axis_start, axis_end) = mid range
         * [axis_end + 1, -1) = inner range
         *
         * for each location in the outer and inner range, all the values in the mid range are
         * normalized together
         */
        std::size_t axis_start, axis_end;

        /* 1 for L1 norm, 2 for L2 norm */
        std::size_t norm;

        /* epsilon to use to avoid divison by zero */
        T eps;
    };

    template <class T>
    class NormalizeOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        template <class V>
        NormalizeOp(csl::Stream stream_, const Mat& weights, const NormalizeConfiguration<V>& config)
            : stream(std::move(stream_)), weight{ 1.0 }
        {
            norm_order = config.norm;
            epsilon = config.eps;
            axis_start = config.axis_start;
            axis_end = config.axis_end;

            if (!weights.empty())
            {
                if (weights.total() == 1)
                {
                    weight = weights.at<T>(0, 0);
                }
                else
                {
                    weightsTensor = csl::makeTensorHeader<T>(weights);
                    csl::copyMatToTensor<T>(weightsTensor, weights, stream);
                }
            }

            std::size_t outer_size = 1;
            for (int i = 0; i < axis_start; i++)
                outer_size *= config.input_shape[i];

            std::size_t inner_size = 1;
            for (int i = axis_end; i < config.input_shape.size(); i++)
                inner_size *= config.input_shape[i];

            scratch_mem_in_bytes = outer_size * inner_size * sizeof(T);
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();
            auto input_shape = input_wrapper->getShape();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            std::size_t outer_size = 1;
            for (int i = 0; i < axis_start; i++)
                outer_size *= input_shape[i];

            std::size_t mid_size = 1;
            for (int i = axis_start; i < axis_end; i++)
                mid_size *= input_shape[i];

            std::size_t inner_size = 1;
            for (int i = axis_end; i < input_shape.size(); i++)
                inner_size *= input_shape[i];

            auto scratch_ptr = reinterpret_cast<T*>(csl::WorkspaceAccessor::get(workspace).get());
            auto scratch = csl::span<T>(csl::DevicePtr<T>(scratch_ptr), workspace.size());
            csl::kernels::normalize<T>(stream, output, input, outer_size, mid_size, inner_size, norm_order, epsilon, scratch);

            /* there might be a single weight in which case `weight` will be not equal to 1.0
             * or there might be several weights
             * or we don't have to scale
             */
            if (weight != 1.0)
            {
                csl::kernels::scale1<T>(stream, output, input, weight);
            }
            else if (!weightsTensor.empty())
            {
                CV_Assert(weightsTensor.size() != 1); /* constructor should have set up to use `weight` */
                CV_Assert(weightsTensor.size() == mid_size);
                csl::kernels::scaleN<T>(stream, output, input, inner_size, weightsTensor);
            }
        }

        std::size_t get_workspace_memory_in_bytes() { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        csl::Tensor<T> weightsTensor;
        T weight; /* if there is only one weight, we use this */

        T epsilon;
        std::size_t norm_order;
        std::size_t axis_start, axis_end;

        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_CUDA4DNN_PRIMITIVES_NORMALIZE_BBOX_HPP */
