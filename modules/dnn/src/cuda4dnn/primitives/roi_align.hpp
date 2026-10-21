// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_ALIGN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_ALIGN_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/roi_align.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class RoiAlignOp final : public CUDABackendNode {
    public:
        RoiAlignOp(csl::Stream stream_, const kernels::RoiAlignParams& params_)
            : stream(std::move(stream_)), params(params_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() == 3 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);
            auto rois = csl::viewOf<float>(inputs[1]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            CV_Assert(inShape.dims == 4);

            /* The spatial extents are only known once shapes are resolved, so the
             * per-run fields are filled here and the attribute-derived ones come from
             * the layer. */
            kernels::RoiAlignParams p = params;
            p.channels = inShape[1];
            p.height   = inShape[2];
            p.width    = inShape[3];

            const std::size_t batch_index_element_size = inputs[2].elemSize();
            // device pointer, same idiom the gather primitive uses for its index tensor
            const void* batch_indices = reinterpret_cast<const uchar*>(inputs[2].u->handle) + inputs[2].offset;

            kernels::roi_align<T>(stream, output, input, rois,
                                  batch_indices, batch_index_element_size, p);
        }

    private:
        csl::Stream stream;
        kernels::RoiAlignParams params;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_ROI_ALIGN_HPP */
