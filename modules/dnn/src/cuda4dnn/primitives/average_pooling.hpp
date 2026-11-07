// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.
#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_AVERAGE_POOLING_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_AVERAGE_POOLING_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"
#include "../csl/tensor.hpp"

#include "../kernels/average_pooling.hpp"
#include "../kernels/reduce.hpp"

#include <opencv2/core.hpp>

#include <cstdint>
#include <vector>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    struct AveragePoolingConfiguration {
        std::vector<std::int64_t> kernel_shape;
        std::vector<std::int64_t> strides;
        std::vector<std::int64_t> pads;
        std::vector<std::int64_t> dilations;
        bool count_include_pad;
    };

    template <class T>
    class AveragePoolingOp final : public CUDABackendNode {
    public:
        AveragePoolingOp(csl::Stream stream_, const AveragePoolingConfiguration& config)
            : stream(std::move(stream_)),
              kernel_shape(config.kernel_shape),
              strides(config.strides),
              pads(config.pads),
              dilations(config.dilations),
              count_include_pad(config.count_include_pad)
        {
        }

        void forward(
            const std::vector<UMat>& inputs,
            const std::vector<UMat>& outputs,
            csl::Workspace& workspace) override
        {
            CV_Assert(inputs.size() == 1 && outputs.size() == 1);
            CV_UNUSED(workspace);

            auto input = csl::viewOf<T>(inputs[0]);
            auto output = csl::spanOf<T>(outputs[0]);

            MatShape inShape = cv::dnn::shape(inputs[0]);
            MatShape outShape = cv::dnn::shape(outputs[0]);
            std::vector<std::int64_t> in_shape(inShape.begin(), inShape.end());
            std::vector<std::int64_t> out_shape(outShape.begin(), outShape.end());

            // a window over every spatial axis without padding is a per-channel mean, which the
            // reduction kernel spreads over a block per channel instead of one thread per output
            if (isGlobal(in_shape)) {
                std::vector<std::int64_t> axes;
                for (std::size_t k = 2; k < in_shape.size(); k++)
                    axes.push_back(static_cast<std::int64_t>(k));
                kernels::reduce_mean<T>(stream, output, input, in_shape, axes);
                return;
            }

            kernels::average_pool<T>(stream, output, input,
                                     in_shape, out_shape, kernel_shape, strides, pads, dilations, count_include_pad);
        }

    private:
        bool isGlobal(const std::vector<std::int64_t>& in_shape) const
        {
            const std::size_t nspatial = kernel_shape.size();
            if (nspatial == 0 || in_shape.size() != nspatial + 2)
                return false;
            for (std::size_t i = 0; i < nspatial; i++)
                if (kernel_shape[i] != in_shape[2 + i] || (!dilations.empty() && dilations[i] != 1))
                    return false;
            for (std::int64_t p : pads)
                if (p != 0)
                    return false;
            return true;
        }

        csl::Stream stream;
        std::vector<std::int64_t> kernel_shape, strides, pads, dilations;
        bool count_include_pad;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_AVERAGE_POOLING_HPP */
