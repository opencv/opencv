// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv {
namespace dnn {

class SizeLayerImpl CV_FINAL : public SizeLayer
{
public:
    SizeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        outputs.assign(1, MatShape({1}));
        return false;
    }

    void getTypes(const std::vector<MatType>& /*inputs*/,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        outputs.assign(requiredOutputs, MatType(CV_64S));
        internals.assign(requiredInternals, MatType(CV_64S));
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays /*internals_arr*/) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(inputs.size() == 1);
        const Mat& x = inputs[0];

        size_t total = x.total()*x.channels();

        outputs[0].create(1, 1, CV_64S);
        outputs[0].at<int64_t>(0) = static_cast<int64_t>(total);
    }
};

Ptr<SizeLayer> SizeLayer::create(const LayerParams& params)
{
    return Ptr<SizeLayer>(new SizeLayerImpl(params));
}

}}
