// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

#include <cmath>

namespace cv {
namespace dnn {

/*
    Implementation of CausalConvWithState, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__CausalConvWithState.html

    Opset 27 is covered.
*/
class CausalConvWithStateLayerImpl CV_FINAL : public CausalConvWithStateLayer
{
public:
    CausalConvWithStateLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        std::string act = params.get<std::string>("activation", "");
        silu = (act == "silu" || act == "swish");
        CV_Check(act, act.empty() || silu, "CausalConvWithState: unsupported activation");
    }

    bool supportBackend(int backendId) CV_OVERRIDE { return backendId == DNN_BACKEND_OPENCV; }

    static bool present(const std::vector<Mat>& in, size_t i) { return in.size() > i && !in[i].empty(); }

    void getTypes(const std::vector<MatType>& inputs, const int requiredOutputs, const int,
                  std::vector<MatType>& outputs, std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckType(inputs[0], inputs[0] == CV_32F || inputs[0] == CV_16F, "");
        outputs.assign(requiredOutputs, inputs[0]);
        internals.clear();
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs, const int,
                         std::vector<MatShape>& outputs, std::vector<MatShape>&) const CV_OVERRIDE
    {
        CV_CheckGE(inputs.size(), (size_t)2, "CausalConvWithState needs input and weight");
        CV_CheckEQ(inputs[0].dims, 3, "input must be [batch, channels, seq]");
        CV_CheckEQ(inputs[1].dims, 3, "weight must be [channels, 1, kernel]");
        const int B = inputs[0][0], C = inputs[0][1], T = inputs[0][2], K = inputs[1][2];
        outputs.assign(1, MatShape{B, C, T});
        outputs.push_back(MatShape{B, C, K - 1});
        return false;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> rawIn, rawOut;
        inputs_arr.getMatVector(rawIn);
        outputs_arr.getMatVector(rawOut);

        const bool fp16 = rawIn[0].depth() == CV_16F;
        std::vector<Mat> in32, out32;
        if (fp16)
        {
            in32.resize(rawIn.size());
            for (size_t i = 0; i < rawIn.size(); ++i)
                if (!rawIn[i].empty()) rawIn[i].convertTo(in32[i], CV_32F);
            out32.resize(rawOut.size());
            for (size_t i = 0; i < rawOut.size(); ++i)
                out32[i].create(rawOut[i].dims, rawOut[i].size.p, CV_32F);
        }
        std::vector<Mat>& inputs  = fp16 ? in32  : rawIn;
        std::vector<Mat>& outputs = fp16 ? out32 : rawOut;

        const Mat& input  = inputs[0];
        const Mat& weight = inputs[1];
        const bool has_bias = present(inputs, 2);
        const bool has_past = present(inputs, 3);

        const int B = input.size[0], C = input.size[1], T = input.size[2];
        const int K = weight.size[2];
        const int P = K - 1;   // state / left-pad width

        const float* Ip = input.ptr<float>();
        const float* Wp = weight.ptr<float>();
        const float* Bp = has_bias ? inputs[2].ptr<float>() : nullptr;
        const float* Sp = has_past ? inputs[3].ptr<float>() : nullptr;
        float* Op = outputs[0].ptr<float>();
        float* PSp = outputs[1].ptr<float>();

        parallel_for_(Range(0, B * C), [&](const Range& r)
        {
            std::vector<float> pad(P + T);
            for (int bc = r.start; bc < r.end; ++bc)
            {
                const int c = bc % C;
                const float* x = Ip + (size_t)bc * T;
                const float* w = Wp + (size_t)c * K;

                for (int j = 0; j < P; ++j)
                    pad[j] = has_past ? Sp[(size_t)bc * P + j] : 0.f;
                for (int t = 0; t < T; ++t)
                    pad[P + t] = x[t];

                const float bias = has_bias ? Bp[c] : 0.f;
                float* o = Op + (size_t)bc * T;
                for (int t = 0; t < T; ++t)
                {
                    float acc = bias;
                    for (int k = 0; k < K; ++k)
                        acc += w[k] * pad[t + k];
                    o[t] = acc;
                }
                if (silu)
                    for (int t = 0; t < T; ++t)
                        o[t] = o[t] / (1.f + std::exp(-o[t]));

                float* ps = PSp + (size_t)bc * P;
                for (int j = 0; j < P; ++j)
                    ps[j] = pad[T + j];
            }
        });

        if (fp16)
            for (size_t i = 0; i < rawOut.size(); ++i)
                out32[i].convertTo(rawOut[i], CV_16F);
    }

    int64 getFLOPS(const std::vector<MatShape>& inputs, const std::vector<MatShape>&) const CV_OVERRIDE
    {
        const int64 B = inputs[0][0], C = inputs[0][1], T = inputs[0][2], K = inputs[1][2];
        return B * C * T * (2 * K + 4);
    }

private:
    bool silu = false;
};

Ptr<CausalConvWithStateLayer> CausalConvWithStateLayer::create(const LayerParams& params)
{
    return makePtr<CausalConvWithStateLayerImpl>(params);
}

}} // namespace cv::dnn
