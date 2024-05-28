// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CELLS_HPP
#define OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_CELLS_HPP

#include "../../op_cuda.hpp"

#include "../csl/cudnn.hpp"
#include "../csl/tensor_ops.hpp"
#include "../csl/cudnn/recurrent.hpp"

namespace cv { namespace dnn { namespace cuda4dnn {

struct RNNConfiguration
{
    int seqLength;
    int numLayers;
    int hiddenSize;
    int inputSize;
    int miniBatch;
    bool bidirectional;
};

template<class T>
class LSTMOp final : public CUDABackendNode
{
public:
    using wrapper_type = GetCUDABackendWrapperType<T>;

    LSTMOp(csl::Stream stream_, csl::cudnn::Handle handle, const Mat& filters, const Mat& h0,
           const Mat& c0, const RNNConfiguration& config)
            : stream(std::move(stream_))
    {
        typename csl::LSTM<T>::params_type params{
                {filters.total(), 1, 1}, // reshape
                config.seqLength,
                config.numLayers,
                config.hiddenSize,
                config.inputSize,
                config.miniBatch,
                config.bidirectional,
                0.0, /* dropout */
                csl::cudnn::RNNDescriptor<T>::RNNMode::LSTM
        };

        lstm = csl::LSTM<T>(handle, params);
        auto correct_shape_filters = filters.reshape(1, {static_cast<int>(filters.total()), 1, 1});
        filtersTensor = csl::makeTensorHeader<T>(correct_shape_filters);
        csl::copyMatToTensor<T>(correct_shape_filters, filtersTensor, stream);

        h0Tensor = csl::makeTensorHeader<T>(h0);
        csl::copyMatToTensor<T>(h0, h0Tensor, stream);

        c0Tensor = csl::makeTensorHeader<T>(c0);
        csl::copyMatToTensor<T>(c0, c0Tensor, stream);
    }

    void forward(const std::vector<cv::Ptr<BackendWrapper>>& inputs,
                 const std::vector<cv::Ptr<BackendWrapper>>& outputs,
                 csl::Workspace& workspace) override
    {
        CV_Assert(inputs.size() == 1 && !outputs.empty());

        auto input_wrapper = inputs[0].dynamicCast<wrapper_type>();
        auto input = input_wrapper->getView();

        auto y_output_wrapper = outputs[0].dynamicCast<wrapper_type>();
        auto y_output = y_output_wrapper->getSpan();

        Ptr<wrapper_type> yc_output_wrapper = outputs.size() == 2 ? outputs[1].dynamicCast<wrapper_type>() : Ptr<wrapper_type>();
        csl::TensorSpan<T> yc_output = yc_output_wrapper.empty() ? csl::TensorSpan<T>() : yc_output_wrapper->getSpan();

        lstm.inference(input, y_output, yc_output, filtersTensor, h0Tensor, c0Tensor, workspace);
    }

    std::size_t get_workspace_memory_in_bytes() const noexcept override
    {
        return lstm.get_workspace_memory_in_bytes();
    }

private:
    csl::LSTM<T> lstm;
    csl::Stream stream;
    csl::Tensor<T> filtersTensor;
    csl::Tensor<T> h0Tensor;
    csl::Tensor<T> c0Tensor;
};

}}} /* namespace cv::dnn::cuda4dnn */

#endif //OPENCV_DNN_SRC_CUDA4DNN_PRIMITIVES_RECURRENT_CELLS_HPP
