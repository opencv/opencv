// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LAYER_NORM_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LAYER_NORM_HPP

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

    template <class T>
    class LayerNormOp final : public CUDABackendNode {
     public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        LayerNormOp(csl::Stream stream_, int normalized_axis, float epsilon_, size_t loops)
            : stream(std::move(stream_)), epsilon(epsilon_) {
            CV_CheckGE(normalized_axis, 0, "LayerNorm/CUDA: axis needs to be normalized");
            axis = static_cast<size_t>(normalized_axis);

            csl::WorkspaceBuilder builder;
            builder.require<float>(loops);
            builder.require<float>(loops);
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(const std::vector<cv::Ptr<BackendWrapper>>& inputs,
                     const std::vector<cv::Ptr<BackendWrapper>>& outputs,
                     csl::Workspace& workspace) override {
            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto scale_wrapper = inputs[1].dynamicCast<wrapper_type>();

            auto input = input_wrapper->getView();
            auto scale = scale_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            auto loops = input.size_range(0, axis);
            auto norm_size = input.size_range(axis, input.rank());
            if (norm_size == 1) {
                kernels::fill<T>(stream, output, 0.f);
                return;
            } else {
                auto ws_allocator = csl::WorkspaceAllocator(workspace);

                auto mean = ws_allocator.get_span<float>(loops);
                kernels::fill<float>(stream, mean, 0.f);

                auto inv_stddev = ws_allocator.get_span<float>(loops);
                kernels::fill<float>(stream, inv_stddev, 0.f);

                kernels::reduce_mean_sqr_sum<T>(stream, mean, inv_stddev, input, norm_size);
                kernels::compute_normalization_scale(stream, inv_stddev, mean, inv_stddev, norm_size, epsilon);
                if (inputs.size() == 3) {
                    auto bias_wrapper = inputs[2].dynamicCast<wrapper_type>();
                    auto bias = bias_wrapper->getView();
                    kernels::normalize_mean_variance_layernorm<T>(stream, output, input, scale, bias, mean, inv_stddev, norm_size);
                } else {
                    kernels::normalize_mean_variance_layernorm<T>(stream, output, input, scale, mean, inv_stddev, norm_size);
                }
            }
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

     private:
        csl::Stream stream;

        float epsilon;
        size_t axis;

        std::size_t scratch_mem_in_bytes;
    };

}}} // cv::dnn::cuda4dnn

#endif // OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_LAYER_NORM_HPP
