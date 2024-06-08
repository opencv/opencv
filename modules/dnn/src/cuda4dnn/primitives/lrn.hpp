// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LRN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LRN_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/tensor_ops.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class LRNType {
        ACROSS_CHANNELS,
        WITHIN_CHANNEL
    };

    template <class T>
    class LRNOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        LRNOp(csl::cudnn::Handle handle, LRNType type_, std::size_t local_size, T alpha, T beta, T bias, std::size_t largestInputSize)
            : scratch_mem_in_bytes { 0 }
        {
            typename csl::LRN<T>::LRNType type{};
            switch (type_) {
            case LRNType::ACROSS_CHANNELS: type = csl::LRN<T>::LRNType::ACROSS_CHANNELS; break;
            case LRNType::WITHIN_CHANNEL: type = csl::LRN<T>::LRNType::WITHIN_CHANNEL; break;
            }
            lrn = csl::LRN<T>(std::move(handle), local_size, alpha, beta, bias, type);

            csl::WorkspaceBuilder builder;
            if (type_ == LRNType::WITHIN_CHANNEL) {
                /* this is not a bug; we require two of these */
                builder.require<T>(largestInputSize);
                builder.require<T>(largestInputSize);
            }

            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            for (int i = 0; i < inputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                csl::WorkspaceAllocator allocator(workspace);
                lrn.normalize(input, output, allocator.get_instance());
            }
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::LRN<T> lrn;
        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LRN_HPP */
