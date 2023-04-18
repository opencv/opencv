// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_POOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/roi_pooling.hpp"

#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class ROIPoolingOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        ROIPoolingOp(csl::Stream stream_, float spatial_scale)
            : stream(std::move(stream_)), spatial_scale{spatial_scale} { }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 2 && outputs.size() == 1);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            auto rois_wrapper = inputs[1].dynamicCast<wrapper_type>();
            auto rois = rois_wrapper->getView();

            auto output_wrapper = outputs[0].dynamicCast<wrapper_type>();
            auto output = output_wrapper->getSpan();

            kernels::roi_pooling<T>(stream, output, input, rois, spatial_scale);
        }

    private:
        csl::Stream stream;
        float spatial_scale;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_POOLING_HPP */
