// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <numeric>

namespace cv { 
namespace dnn {

// Group Normalization Layer
class GroupNormLayerImpl CV_FINAL : public GroupNormLayer {
public:
    GroupNormLayerImpl(const LayerParams &params) {
        setParamsFrom(params);

        epsilon = params.get<float>("epsilon", 1e-5);
        num_groups = params.get<int>("num_groups");
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE {
        const auto &input = inputs[0];
        CV_CheckGE(input.size(), static_cast<size_t>(3), "DNN/GroupNorm: input dimension >= 3 is required");

        outputs.assign(1, inputs[0]);
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const auto& input = inputs[0];
        const auto& scale = inputs[1];
        const auto& bias = inputs[2];

        const auto input_shape = shape(input);
        CV_CheckGE(input_shape.size(), static_cast<size_t>(3), "GroupNorm: input data should be at least three dimensional");
        size_t N = static_cast<size_t>(input_shape[0]);
        size_t C = static_cast<size_t>(input_shape[1]);
        CV_CheckEQ(scale.total(), C, "GroupNorm: scale should be a 1d tensor and match the channel of input data");
        CV_CheckEQ(bias.total(), C, "GroupNorm: bias should be a 1d tensor and match the channel of input data");

        // This implementation assumes [N, C, D1, D2, ..., Dn], where N is the batch size and C is the channel.
        const auto* input_data = input.ptr<const float>();
        const auto* scale_data = scale.ptr<const float>();
        const auto* bias_data = bias.ptr<const float>();
        auto* output_data = outputs[0].ptr<float>();

        size_t channels_per_group = C / num_groups;
        size_t loops = N * num_groups;
        size_t step = static_cast<size_t>(total(input_shape, 2) * channels_per_group);

        auto compute_mean = [](const float* data, size_t size) {
            return std::accumulate(data, data + size, 0.0f) / size;
        };

        auto compute_variance = [](const float* data, size_t size, float mean) {
            return std::inner_product(data, data + size, data, 0.0f,
                                      [](float a, float b) { return a + b; },
                                      [mean](float a, float b) { return (a - mean) * (b - mean); }) / size;
        };

        auto normalize_data = [](float* output_data, const float* input_data, size_t size, float mean, float inv_stdev, float scale, float bias) {
            for (size_t i = 0; i < size; i++) {
                output_data[i] = scale * (input_data[i] - mean) * inv_stdev + bias;
            }
        };

        std::vector<float> mean_data(loops);
        std::vector<float> inv_stdev_data(loops);

        for (size_t i = 0; i < loops; i++) {
            const auto* x = input_data + step * i;
            auto* y = output_data + step * i;

            auto mean = compute_mean(x, step);
            mean_data[i] = mean;

            auto variance = compute_variance(x, step, mean);
            float inv_stdev = 1.0f / std::sqrt(variance + epsilon);
            inv_stdev_data[i] = inv_stdev;
        }

        loops = N * C;
        step = static_cast<size_t>(total(input_shape, 2));

        for (size_t i = 0; i < loops; i++) {
            const auto* x = input_data + step * i;
            auto* y = output_data + step * i;

            size_t group_idx = i / channels_per_group;
            size_t channel_idx = i % C;

            float s = scale_data[channel_idx];
            float b = bias_data[channel_idx];

            normalize_data(y, x, step, mean_data[group_idx], inv_stdev_data[group_idx], s, b);
        }
    }

private:
    float epsilon;
    size_t num_groups;
};

Ptr<GroupNormLayer> GroupNormLayer::create(const LayerParams &params) {
    return Ptr<GroupNormLayer>(new GroupNormLayerImpl(params));
}

}} // cv::dnn