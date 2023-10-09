// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv { namespace dnn {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#InstanceNormalization
class InstanceNormLayerImpl CV_FINAL : public InstanceNormLayer {
public:
    InstanceNormLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        epsilon = params.get<float>("epsilon", 1e-5);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE {
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

        const auto &input = inputs[0];
        const auto &scale = inputs[1];
        const auto &bias = inputs[2];

        const auto input_shape = shape(input);
        CV_CheckGE(input_shape.size(), static_cast<size_t>(3), "InstanceNorm: input data should be at least three dimensional");
        size_t N = static_cast<size_t>(input_shape[0]),
               C = static_cast<size_t>(input_shape[1]);
        CV_CheckEQ(scale.total(), C, "InstanceNorm: scale should be a 1d tensor and match the channel of input data");
        CV_CheckEQ(bias.total(), C, "InstanceNorm: bias should be a 1d tensor and match the channel of input data");

        // This implementation assumes [N, C, D1, D2, ..., Dn], where N is the batch size and C is the channel.
        const auto *input_data = input.ptr<const float>();
        const auto *scale_data = scale.ptr<const float>();
        const auto *bias_data = bias.ptr<const float>();
        auto *output_data = outputs[0].ptr<float>();
        size_t loops = N * C, step = static_cast<size_t>(total(input_shape, 2));
        auto compute_squared_norm = [step] (const float *data, float mean) {
            float sum = 0.f;
            for (size_t i = 0; i < step; i++) {
                float d = data[i] - mean;
                sum +=  d * d;
            }
            return sum / step;
        };
        for (size_t i = 0; i < loops; i++) {
            float mean = std::accumulate(input_data, input_data + step, 0.f) / step;
            float inv_stdev = 1.0f / std::sqrt(compute_squared_norm(input_data, mean) + epsilon);

            size_t channel = i % C;
            float s = scale_data[channel], b = bias_data[channel];
            // y = scale * (x - mean) / sqrt(variance + epsilon) + B
            for (size_t j = 0; j < step; j++) {
                output_data[j] = s * (input_data[j] - mean) * inv_stdev + b;
            }
            input_data += step;
            output_data += step;
        }
    }
};

Ptr<InstanceNormLayer> InstanceNormLayer::create(const LayerParams &params) {
    return Ptr<InstanceNormLayer>(new InstanceNormLayerImpl(params));
}

}} // cv::dnn
