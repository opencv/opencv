// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"

namespace cv { namespace dnn {

// Operator spec: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.SkipSimplifiedLayerNormalization
class SkipSimplifiedLayerNormalizationLayerImpl CV_FINAL : public SkipSimplifiedLayerNormalizationLayer {
public:
    float epsilon;
    Ptr<RMSNormLayer> rms;

    SkipSimplifiedLayerNormalizationLayerImpl(const LayerParams& params) {
        setParamsFrom(params);
        epsilon = params.get<float>("epsilon", 1e-5f);

        LayerParams rmsParams;
        rmsParams.set("axis", -1);
        rmsParams.set("epsilon", epsilon);
        rms = RMSNormLayer::create(rmsParams);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual void getTypes(const std::vector<MatType>& inputs,
                          const int requiredOutputs,
                          const int requiredInternals,
                          std::vector<MatType>& outputs,
                          std::vector<MatType>& internals) const CV_OVERRIDE {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_16F, "");
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE {
        CV_CheckGE((int)inputs.size(), 3, "SkipSimplifiedLayerNormalization: expects input, skip, gamma [, bias]");

        MatShape reducedShape = inputs[0];
        if (!reducedShape.empty())
            reducedShape.back() = 1;

        outputs.resize(requiredOutputs);
        for (int i = 0; i < requiredOutputs; ++i)
            outputs[i] = (i == 1 || i == 2) ? reducedShape : inputs[0];
        internals.clear();
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE {
        CV_TRACE_FUNCTION();

        if (inputs_arr.depth() == CV_16F) {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat& input = inputs[0];
        const Mat& skip = inputs[1];
        const Mat& gamma = inputs[2];
        const bool hasBias = inputs.size() > 3 && !inputs[3].empty();
        const int numOutputs = (int)outputs.size();
        CV_CheckGE(numOutputs, 1, "SkipSimplifiedLayerNormalization: at least one output required");

        const int hidden = (int)gamma.total();
        const int rows = (int)(input.total() / hidden);

        Mat sumScratch;
        Mat& sumOut = (numOutputs > 3) ? outputs[3] : sumScratch;

        cv::add(input, skip, sumOut);
        if (hasBias) {
            CV_CheckEQ((int)inputs[3].total(), (int)gamma.total(),
                       "SkipSimplifiedLayerNormalization: bias/gamma size mismatch");
            float* s = sumOut.ptr<float>();
            const float* b = inputs[3].ptr<float>();
            parallel_for_(Range(0, rows), [&](const Range& r) {
                for (int i = r.start; i < r.end; ++i) {
                    float* row = s + (size_t)i * hidden;
                    for (int c = 0; c < hidden; ++c) row[c] += b[c];
                }
            });
        }

        std::vector<Mat> rmsInputs = {sumOut, gamma};
        std::vector<Mat> rmsOutputs = {outputs[0]};
        std::vector<Mat> rmsInternals;
        rms->forward(rmsInputs, rmsOutputs, rmsInternals);

        if (numOutputs > 1) {
            Mat& meanOut = outputs[1];
            meanOut.setTo(Scalar(0));

            if (numOutputs > 2) {
                Mat& invStdOut = outputs[2];
                const float* s = sumOut.ptr<float>();
                float* iv = invStdOut.ptr<float>();
                parallel_for_(Range(0, rows), [&](const Range& r) {
                    for (int i = r.start; i < r.end; ++i) {
                        const float* row = s + (size_t)i * hidden;
                        double sumSq = 0.0;
                        for (int c = 0; c < hidden; ++c) sumSq += (double)row[c] * row[c];
                        iv[i] = (float)(1.0 / std::sqrt(sumSq / hidden + epsilon));
                    }
                });
            }
        }
    }
};

Ptr<SkipSimplifiedLayerNormalizationLayer> SkipSimplifiedLayerNormalizationLayer::create(const LayerParams& params) {
    return makePtr<SkipSimplifiedLayerNormalizationLayerImpl>(params);
}

}} // namespace cv::dnn