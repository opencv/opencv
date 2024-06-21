// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DEPTH_SPACE_OPS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DEPTH_SPACE_OPS_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"
#include "../csl/tensor_ops.hpp"
#include "../csl/memory.hpp"
#include "../kernels/permute.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class DepthSpaceOps final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        DepthSpaceOps(csl::Stream stream_, const std::vector<int> &internal_shape_,
                     const std::vector<size_t> &permutation_)
            : stream(std::move(stream_)), internal_shape(internal_shape_),
              permutation(permutation_)
        {
            transposed_internal_shape = std::vector<int>(internal_shape.size());
            for (size_t i = 0; i < permutation.size(); i++) {
                transposed_internal_shape[i] = internal_shape[permutation[i]];
            }

            size_t num_elements = std::accumulate(internal_shape.begin(), internal_shape.end(), 1, std::multiplies<size_t>());
            csl::WorkspaceBuilder builder;
            builder.require<T>(num_elements);
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(const std::vector<cv::Ptr<BackendWrapper>> &inputs,
                     const std::vector<cv::Ptr<BackendWrapper>> &outputs,
                     csl::Workspace &workspace) override {
            CV_CheckEQ(inputs.size(), size_t(1), "DepthSpaceOps: only one input is accepted");
            CV_CheckEQ(outputs.size(), size_t(1), "DepthSpaceOps: only one output is accepted");

            auto input_wrapper = inputs.front().dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();
            CV_CheckEQ(input.rank(), size_t(4), "DepthSpaceOps: input needs to be 4-dimensional [N, C, H, W]");
            auto output_wrapper = outputs.front().dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();
            auto ws_allocator = csl::WorkspaceAllocator(workspace);
            auto transposed_internal = ws_allocator.get_tensor_span<T>(transposed_internal_shape.begin(), transposed_internal_shape.end());

            // Call reshape on input so that it has the correct shape for permutation
            input.reshape(internal_shape.begin(), internal_shape.end());
            kernels::permute(stream, transposed_internal, input, permutation);
            // Only copying is needed as output already has the expected shape
            auto t = csl::TensorView<T>(transposed_internal);
            csl::memcpy(output.get(), t.get(), output.size(), stream);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        std::vector<int> internal_shape;
        std::vector<size_t> permutation;
        std::vector<int> transposed_internal_shape;

        std::size_t scratch_mem_in_bytes;
    };

}}} // namespace cv::dnn::cuda4dnn

#endif // OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_DEPTH_SPACE_OPS_HPP
