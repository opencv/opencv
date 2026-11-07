// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/resize.hpp"
#include "../csl/span.hpp"

#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class InterpolationType {
        NEAREST_NEIGHBOUR,
        BILINEAR
    };

    struct ResizeConfiguration {
        InterpolationType type;
        bool align_corners;
        bool half_pixel_centers;
    };

    template <class T>
    class ResizeOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ResizeOp(csl::Stream stream_, const ResizeConfiguration& config)
            : stream(std::move(stream_))
        {
            type = config.type;
            align_corners = config.align_corners;
            half_pixel_centers = config.half_pixel_centers;
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            // roi/scales/sizes may follow the data input; the output shape already carries them
            CV_Assert(inputs.size() >= 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            const auto compute_scale = [this](std::size_t input_size, std::size_t output_size) {
                return (align_corners && output_size > 1) ?
                            static_cast<float>(input_size - 1) / (output_size - 1) :
                            static_cast<float>(input_size) / output_size;
            };

            auto out_height = output.get_axis_size(-2), out_width = output.get_axis_size(-1);
            auto in_height = input.get_axis_size(-2), in_width = input.get_axis_size(-1);
            float scale_height = compute_scale(in_height, out_height),
                  scale_width = compute_scale(in_width, out_width);

            if (type == InterpolationType::NEAREST_NEIGHBOUR)
                kernels::resize_nn<T>(stream, output, input, scale_height, scale_width, align_corners, half_pixel_centers);
            else if (type == InterpolationType::BILINEAR)
                kernels::resize_bilinear<T>(stream, output, input, scale_height, scale_width, half_pixel_centers);
        }

    private:
        csl::Stream stream;
        InterpolationType type;
        bool align_corners, half_pixel_centers;
    };

    // per-axis taps built on the host by the CPU Resize2 builders; idx -1 skips a tap
    struct ResizeTables {
        int taps_y = 1, taps_x = 1;
        std::vector<int> idx_y, idx_x, oob_y, oob_x;
        std::vector<float> w_y, w_x;
        float extrapolation_value = 0.f;
    };

    // every ONNX Resize mode as one gather; the tables carry the coordinate and filter math
    template <class T>
    class ResizeTablesOp final : public CUDABackendNode {
    public:
        ResizeTablesOp(csl::Stream stream_, const ResizeTables& t)
            : stream(std::move(stream_)), taps_y(t.taps_y), taps_x(t.taps_x), extrapolation(t.extrapolation_value)
        {
            idx_y = upload(t.idx_y); w_y = upload(t.w_y); oob_y = upload(t.oob_y);
            idx_x = upload(t.idx_x); w_x = upload(t.w_x); oob_x = upload(t.oob_x);
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 1 && outputs.size() == 1);
            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);
            kernels::resize_separable<T>(stream, output, input,
                                         view<int>(idx_y), view<float>(w_y), view<int>(oob_y), taps_y,
                                         view<int>(idx_x), view<float>(w_x), view<int>(oob_x), taps_x, extrapolation);
        }

    private:
        // device copy on the CUDA UMat allocator, like every other device tensor
        template <class V>
        static UMat upload(const std::vector<V>& v)
        {
            CV_Assert(!v.empty());
            Mat host(1, (int)v.size(), DataType<V>::type, const_cast<V*>(v.data()));
            UMat u;
            u.allocator = cv::cuda::getCudaAllocator();
            u.fitSameSize(host, host.type());
            checkCudaAllocated(u);
            host.copyTo(u);
            return u;
        }

        template <class V>
        static csl::View<V> view(const UMat& u) { return csl::View<V>(csl::viewOf<V>(u).get(), u.total()); }

        csl::Stream stream;
        int taps_y, taps_x;
        float extrapolation;
        UMat idx_y, oob_y, idx_x, oob_x, w_y, w_x;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RESIZE_HPP */
