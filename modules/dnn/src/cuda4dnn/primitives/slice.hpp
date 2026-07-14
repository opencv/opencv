// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SLICE_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SLICE_HPP

#include "../../op_cuda.hpp"

#include "../csl/stream.hpp"

#include "../kernels/slice.hpp"
#include "../kernels/fill_copy.hpp"

#include <opencv2/core.hpp>

#include <cstddef>
#include <vector>
#include <algorithm>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    template <class T>
    class SliceOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        /* offsets is indexed by output number and each subvector is indexed by axis number */
        SliceOp(csl::Stream stream_, std::vector<std::vector<std::size_t>> offsets)
            : stream(std::move(stream_)), offsets(std::move(offsets))
        {
        }

        void forward(
            const std::vector<cv::Ptr<BackendWrapper>>& inputs,
            const std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            /* sometimes the output shape is passed in the form of a second input tensor
             * it's only required for initialization and not here
             */
            CV_Assert(inputs.size() == 1 || inputs.size() == 2);

            auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
            auto input = input_wrapper->getView();

            CV_Assert(offsets.size() == outputs.size());

            for (int i = 0; i < outputs.size(); ++i)
            {
                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                kernels::slice<T>(stream, output, input, offsets[i]);
            }
        }

    private:
        csl::Stream stream;
        std::vector<std::vector<std::size_t>> offsets;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_SLICE_HPP */
