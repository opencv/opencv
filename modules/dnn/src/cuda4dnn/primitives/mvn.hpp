// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MVN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MVN_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/span.hpp"
#include "../csl/tensor.hpp"
#include "../csl/workspace.hpp"

#include "../kernels/fill_copy.hpp"
#include "../kernels/mvn.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct MVNConfiguration {
        std::vector<std::vector<std::size_t>> input_shapes;

        /*
         * [0, split_axis) = outer range
         * [split_axis, -1] = inner range
         *
         * for each location in the outer range, all the values in the inner range are normalized as a group
         */
        std::size_t split_axis;

        /* The group (described above) is centered always. The following parameter controls whether the variance
         * is also normalized.
         */
        bool normalize_variance;
        float epsilon;
    };

    template <class T>
    class MVNOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        MVNOp(csl::Stream stream_, const MVNConfiguration& config)
            : stream(std::move(stream_))
        {
            split_axis = config.split_axis;
            normalize_variance = config.normalize_variance;
            epsilon = config.epsilon;

            std::size_t max_outer_size = 0;
            const auto& input_shapes = config.input_shapes;
            for (int i = 0; i < input_shapes.size(); i++)
            {
                std::size_t outer_size = 1;
                for (int j = 0; j < split_axis; j++)
                    outer_size *= input_shapes[i][j];
                max_outer_size = std::max(max_outer_size, outer_size);
            }

            csl::WorkspaceBuilder builder;
            builder.require<float>(max_outer_size);
            if (normalize_variance)
                builder.require<float>(max_outer_size);
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == outputs.size());

            for (int i = 0; i < inputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                auto outer_size = input.size_range(0, split_axis);
                auto inner_size = input.size_range(split_axis, input.rank());
                if (inner_size == 1)
                {
                    kernels::fill<T>(stream, output, 0.0f);
                    return;
                }
                else
                {
                    auto ws_allocator = csl::WorkspaceAllocator(workspace);

                    auto means = ws_allocator.get_span<float>(outer_size);
                    kernels::fill<float>(stream, means, 0);

                    if (normalize_variance)
                    {
                        auto scales = ws_allocator.get_span<float>(outer_size);
                        kernels::fill<float>(stream, scales, 0);

                        kernels::reduce_mean_sqr_sum<T>(stream, means, scales, input, inner_size);
                        kernels::compute_normalization_scale(stream, scales, means, scales, inner_size, epsilon);
                        kernels::normalize_mean_variance<T>(stream, output, input, means, scales, inner_size);
                    }
                    else
                    {
                        kernels::reduce_mean<T>(stream, means, input, inner_size);
                        kernels::normalize_mean<T>(stream, output, input, means, inner_size);
                    }
                }
            }
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;

        bool normalize_variance;
        float epsilon;
        std::size_t split_axis;

        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_MVN_HPP */
