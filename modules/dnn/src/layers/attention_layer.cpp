// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"

namespace cv { namespace dnn {

// Operator spec: https://github.com/microsoft/onnxruntime/blob/v1.16.1/docs/ContribOperators.md#com.microsoft.Attention
class AttentionLayerImpl CV_FINAL : public AttentionLayer {
 public:
    AttentionLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        CV_CheckTrue(params.has("num_heads"), "DNN/Attention: num_heads is required but missing");
        num_heads = params.get<int>("num_heads"); // required, no default value

        CV_CheckTrue(params.has("qkv_hidden_sizes"), "DNN/Attention: qkv_hidden_sizes is required but missing");
        auto param_qkv_hidden_sizes = params.get("qkv_hidden_sizes");
        CV_CheckEQ(param_qkv_hidden_sizes.size(), 3, "DNN/Attention: qkv_hidden_sizes must and only have three elements");
        qkv_hidden_sizes.resize(3);
        qkv_hidden_sizes[0] = param_qkv_hidden_sizes.get<int>(0);
        qkv_hidden_sizes[1] = param_qkv_hidden_sizes.get<int>(1);
        qkv_hidden_sizes[2] = param_qkv_hidden_sizes.get<int>(2);

        qk_head_size = static_cast<int>(qkv_hidden_sizes[0] / num_heads);
        v_head_size = static_cast<int>(qkv_hidden_sizes[2] / num_heads);

        scale = params.get<float>("scale", sqrt(1.f / qk_head_size));
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE {
        const auto &input = inputs[0];
        const auto &weight = inputs[1];
        const auto &bias = inputs[2];
        int dim_bias = std::accumulate(bias.begin(), bias.end(), 1, std::multiplies<int>());

        CV_CheckEQ(input.back(), weight[weight.size() - 2], "DNN/Attention: invalid input shape");
        CV_CheckEQ(weight.back(), qkv_hidden_sizes[0] + qkv_hidden_sizes[1] + qkv_hidden_sizes[2], "DNN/Attention: invalid weight shape");
        CV_CheckEQ(dim_bias, qkv_hidden_sizes[0] + qkv_hidden_sizes[1] + qkv_hidden_sizes[2], "DNN/Attention: invalid bias shape");

        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        // TODO: impl
    }

 private:
    int qk_head_size;
    int v_head_size;
};

Ptr<AttentionLayer> AttentionLayer::create(const LayerParams &params) {
    return makePtr<AttentionLayerImpl>(params);
}

}} // cv::dnn
