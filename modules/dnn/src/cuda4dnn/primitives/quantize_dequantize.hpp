// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_QUANTIZE_DEQUANTIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_QUANTIZE_DEQUANTIZE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
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

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_QUANTIZE_DEQUANTIZE_HPP */
