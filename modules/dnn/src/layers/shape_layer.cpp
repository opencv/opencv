// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Shape layer.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"

namespace cv
{
namespace dnn
{

class ShapeLayerImpl CV_FINAL : public ShapeLayer
{
public:
    ShapeLayerImpl(const cv::dnn::LayerParams& params)
    {
        setParamsFrom(params);
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
        const int requiredOutputs,
        std::vector<MatShape>& outputs,
        std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(!inputs.empty());
        outputs.clear();
        outputs.push_back({ (int)inputs[0].size() });
        return false;
    }

    virtual void forward(cv::InputArrayOfArrays inputs_arr,
        cv::OutputArrayOfArrays outputs_arr,
        cv::OutputArrayOfArrays internals_arr) {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        CV_Assert(!inputs.empty());

        float* outptr = outputs[0].ptr<float>();
        for (int i = 0; i < inputs[0].size.dims(); ++i)
        {
            outptr[i] = (float)inputs[0].size[i];
        }
    }
};

Ptr<Layer> ShapeLayer::create(const LayerParams& params) {
    return cv::Ptr<Layer>(new ShapeLayerImpl(params));
}

}
}
