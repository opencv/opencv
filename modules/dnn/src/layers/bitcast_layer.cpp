// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2026, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"

#include <cstring>

namespace cv {
namespace dnn {

/*
    Implementation of BitCast, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__BitCast.html

    Opset 26 is covered.
*/
class BitCastLayerImpl CV_FINAL : public BitCastLayer
{
public:
    BitCastLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        outputType = params.get<int>("outputType");
    }

    bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int /*requiredOutputs*/,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& /*internals*/) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)1, "BitCast takes exactly one input");
        outputs.assign(1, inputs[0]);
        return false;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int /*requiredInternals*/,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_CheckEQ(inputs.size(), (size_t)1, "");
        CV_CheckEQ(CV_ELEM_SIZE1(inputs[0]), CV_ELEM_SIZE1(outputType),
                   "BitCast: only equal-width type reinterpretation is supported");
        outputs.assign(requiredOutputs, MatType(outputType));
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        const Mat src = inputs[0].isContinuous() ? inputs[0] : inputs[0].clone();
        Mat& dst = outputs[0];
        CV_CheckEQ(src.elemSize(), dst.elemSize(), "BitCast: element sizes must match");
        CV_CheckEQ(src.total(), dst.total(), "");
        std::memcpy(dst.ptr(), src.ptr(), src.total() * src.elemSize());
    }

private:
    int outputType;
};

Ptr<BitCastLayer> BitCastLayer::create(const LayerParams& params)
{
    return makePtr<BitCastLayerImpl>(params);
}

}} // namespace cv::dnn
