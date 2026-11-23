// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CUM_SCAN_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CUM_SCAN_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/cum_scan.hpp"

#include <opencv2/core.hpp>
#include <opencv2/dnn/shape_utils.hpp>

#include <cstddef>
#include <utility>
#include <vector>

namespace cv { namespace dnn { namespace cuda4dnn {

    /* Backs CumSum and CumProd. The axis is resolved by the layer at init time -- it can
     * arrive as an input tensor, and reading it here would mean a device-to-host stall
     * inside forward(). */
    template <class T>
    class CumScanOp final : public CUDABackendNode {
    public:
        CumScanOp(csl::Stream stream_, int axis_, bool is_prod_, bool exclusive_, bool reverse_)
            : stream(std::move(stream_)), axis(axis_),
              is_prod(is_prod_), exclusive(exclusive_), reverse(reverse_) { }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_UNUSED(workspace);
            CV_Assert(inputs.size() >= 1 && outputs.size() == 1);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            const int a = normalize_axis(axis, inShape.dims);
            CV_Assert(a >= 0 && a < inShape.dims);

            std::size_t outer_size = 1, inner_size = 1;
            for (int i = 0; i < a; i++) outer_size *= static_cast<std::size_t>(inShape[i]);
            for (int i = a + 1; i < inShape.dims; i++) inner_size *= static_cast<std::size_t>(inShape[i]);
            const std::size_t target_size = static_cast<std::size_t>(inShape[a]);

            kernels::cum_scan<T>(stream, output, input,
                                 outer_size, target_size, inner_size,
                                 is_prod, exclusive, reverse);
        }

    private:
        csl::Stream stream;
        int axis;
        bool is_prod, exclusive, reverse;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CUM_SCAN_HPP */
