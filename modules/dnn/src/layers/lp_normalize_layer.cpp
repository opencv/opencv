// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <iostream>
namespace cv { namespace dnn {

class LPNormalizeLayerImpl : public LPNormalizeLayer
{
public:

    LPNormalizeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        pnorm = params.get<float>("p", 2);
        epsilon = params.get<float>("eps", 1e-10f);
        CV_Assert(pnorm > 0);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        if (pnorm != 1 && pnorm != 2)
        {
            internals.resize(1, inputs[0]);
        }
        return true;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_Assert(inputs[0]->total() == outputs[0].total());
        float norm;
        if (pnorm == 1)
            norm = cv::norm(*inputs[0], NORM_L1);
        else if (pnorm == 2)
            norm = cv::norm(*inputs[0], NORM_L2);
        else
        {
            pow(abs(*inputs[0]), pnorm, internals[0]);
            norm = pow(sum(internals[0])[0], 1.0f / pnorm);
        }
        multiply(*inputs[0], 1.0f / (norm + epsilon), outputs[0]);
    }

    int64 getFLOPS(const std::vector<MatShape> &inputs,
                  const std::vector<MatShape> &) const
    {
        int64 flops = 0;
        for (int i = 0; i < inputs.size(); i++)
            flops += 3 * total(inputs[i]);
        return flops;
    }
};

Ptr<LPNormalizeLayer> LPNormalizeLayer::create(const LayerParams& params)
{
    return Ptr<LPNormalizeLayer>(new LPNormalizeLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
