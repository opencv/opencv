// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_CUDA4DNN_PRIMITIVES_LRN_HPP
#define OPENCV_DNN_CUDA4DNN_PRIMITIVES_LRN_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/tensor_ops.hpp"

#include <cstddef>
#include <utility>

namespace cv { namespace dnn { namespace cuda4dnn {

    enum class lrn_type {
        across_channels
    };

    template <class T>
    class LRNOp final : public CUDABackendNode {
    public:
        using wrapper_type = GetCUDABackendWrapperType<T>;

        LRNOp(csl::cudnn::Handle handle, lrn_type type, std::size_t local_size, T alpha, T beta, T bias)
        {
            if(type == lrn_type::across_channels)
                lrn = csl::LRN<T>(std::move(handle), local_size, alpha, beta, bias, csl::LRN<T>::lrn_type::ACROSS_CHANNELS);
        }

        void forward(
            std::vector<cv::Ptr<BackendWrapper>>& inputs,
            std::vector<cv::Ptr<BackendWrapper>>& outputs,
            csl::Workspace& workspace) override
        {
            for (int i = 0; i < inputs.size(); i++)
            {
                auto input_wrapper = inputs[i].dynamicCast<wrapper_type>();
                auto input = input_wrapper->getView();

                auto output_wrapper = outputs[i].dynamicCast<wrapper_type>();
                auto output = output_wrapper->getSpan();

                lrn.normalize(input, output);
            }
        }

    private:
        csl::LRN<T> lrn;
    };

}}} /* namespace cv::dnn::cuda4dnn */

#endif /* OPENCV_DNN_CUDA4DNN_PRIMITIVES_LRN_HPP */
