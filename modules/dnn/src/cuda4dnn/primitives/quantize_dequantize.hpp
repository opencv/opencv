// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_QUANTIZE_DEQUANTIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_QUANTIZE_DEQUANTIZE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/workspace.hpp"
#include "../kernels/quantize_dequantize.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#include <cstdint>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    // Per-tensor and per-axis only (block_size == 0); gated by the layer's supportBackend.
    template <class T>
    class DequantizeLinearOp final : public CUDABackendNode {
    public:
        DequantizeLinearOp(csl::Stream stream_, int axis_) : stream(std::move(stream_)), axis(axis_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 || inputs.size() == 3);
            CV_Assert(outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto scale = csl::viewOf<float>(inputs[1]);
            auto output = csl::spanOf<float>(outputs[0]);
            csl::View<T> zero_point = inputs.size() == 3 ? csl::viewOf<T>(inputs[2]) : csl::View<T>();
            CV_Assert(zero_point.empty() || zero_point.size() == scale.size());

            MatShape inpshape = shape(inputs[0]);
            int ndims = inpshape.dims;
            int ax = ndims > 0 ? normalize_axis(axis, ndims) : 0;
            std::size_t sz_a = scale.size() > 1 ? (std::size_t)inpshape[ax] : 1;
            CV_Assert(scale.size() == 1 || scale.size() == sz_a);
            std::size_t slice_size = 1;
            if (sz_a > 1)
                for (int i = ax + 1; i < ndims; i++)
                    slice_size *= inpshape[i];

            kernels::dequantize_linear<T>(stream, output, input, scale, zero_point, sz_a, slice_size);
        }

    private:
        csl::Stream stream;
        int axis;
    };

    template <class T>
    class QuantizeLinearOp final : public CUDABackendNode {
    public:
        QuantizeLinearOp(csl::Stream stream_, int axis_) : stream(std::move(stream_)), axis(axis_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 2 || inputs.size() == 3);
            CV_Assert(outputs.size() == 1);

            auto input = csl::viewOf<float>(inputs[0]);
            auto scale = csl::viewOf<float>(inputs[1]);
            auto output = csl::spanOf<T>(outputs[0]);
            csl::View<T> zero_point = inputs.size() == 3 ? csl::viewOf<T>(inputs[2]) : csl::View<T>();
            CV_Assert(zero_point.empty() || zero_point.size() == scale.size());

            MatShape inpshape = shape(inputs[0]);
            int ndims = inpshape.dims;
            int ax = ndims > 0 ? normalize_axis(axis, ndims) : 0;
            std::size_t sz_a = scale.size() > 1 ? (std::size_t)inpshape[ax] : 1;
            CV_Assert(scale.size() == 1 || scale.size() == sz_a);
            std::size_t slice_size = 1;
            if (sz_a > 1)
                for (int i = ax + 1; i < ndims; i++)
                    slice_size *= inpshape[i];

            kernels::quantize_linear<T>(stream, output, input, scale, zero_point, sz_a, slice_size);
        }

    private:
        csl::Stream stream;
        int axis;
    };

    // ONNX DynamicQuantizeLinear: float32 x -> uint8 y, scalar y_scale and y_zero_point, all on the device.
    class DynamicQuantizeLinearOp final : public CUDABackendNode {
    public:
        DynamicQuantizeLinearOp(csl::Stream stream_, std::size_t total) : stream(std::move(stream_))
        {
            partials = kernels::dynamic_quantize_partials(total);
            csl::WorkspaceBuilder builder;
            builder.require<float>(2 * partials);
            scratch_mem_in_bytes = builder.required_workspace_size();
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 3);

            auto input = csl::viewOf<float>(inputs[0]);
            auto output = csl::spanOf<std::uint8_t>(outputs[0]);
            auto scale = csl::spanOf<float>(outputs[1]);
            auto zero_point = csl::spanOf<std::uint8_t>(outputs[2]);

            auto allocator = csl::WorkspaceAllocator(workspace);
            auto scratch = allocator.get_span<float>(2 * partials);
            kernels::dynamic_quantize_scale(stream, scale, zero_point, input, scratch);
            // same per-tensor formula as QuantizeLinear, with the scale and zero point read on the device
            kernels::quantize_linear<std::uint8_t>(stream, output, input, scale, zero_point, 1, 1);
        }

        std::size_t get_workspace_memory_in_bytes() const noexcept override { return scratch_mem_in_bytes; }

    private:
        csl::Stream stream;
        std::size_t partials;
        std::size_t scratch_mem_in_bytes;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_QUANTIZE_DEQUANTIZE_HPP */
