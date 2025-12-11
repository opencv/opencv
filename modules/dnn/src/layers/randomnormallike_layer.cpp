// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include <opencv2/dnn/layer.details.hpp>
#include <cmath>

namespace cv { namespace dnn {

class RandomNormalLikeLayerImpl CV_FINAL : public Layer
{
public:
    RandomNormalLikeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        mean = params.get<double>("mean", 0.0);
        scale = params.get<double>("scale", 1.0);

        hasSeed = params.has("seed");
        if (hasSeed)
        {
            double seedAttr = params.get<double>("seed");
            uint64_t s = static_cast<uint64_t>(std::llround(seedAttr));
            seed = s ? s : 1;
        }

        depth = params.get<int>("depth", CV_32F);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape>& inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape>& outputs,
                                 std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_UNUSED(requiredOutputs);
        CV_UNUSED(internals);
        CV_CheckEQ(inputs.size(), 1ull, "RandomNormalLike: one input is expected");

        outputs.assign(1, inputs[0]);
        return false;
    }

    virtual void forward(InputArrayOfArrays inputs_arr,
                         OutputArrayOfArrays outputs_arr,
                         OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_UNUSED(internals_arr);
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert(!inputs.empty());
        CV_Assert(!outputs.empty());

        Mat& out = outputs[0];

        RNG rng = hasSeed ? RNG(seed) : theRNG();

        if (out.depth() == CV_32F || out.depth() == CV_64F || out.depth() == CV_16F)
        {
            rng.fill(out, RNG::NORMAL, mean, scale);
        }
        else
        {
            Mat tmp(out.size.dims(), out.size.p, CV_32F);
            rng.fill(tmp, RNG::NORMAL, mean, scale);
            tmp.convertTo(out, out.type());
        }
    }

private:
    double mean;
    double scale;
    bool hasSeed = false;
    uint64_t seed = 0;
    int depth;
};

CV_DNN_REGISTER_LAYER_CLASS_STATIC(RandomNormalLike, RandomNormalLikeLayerImpl);

}} // namespace cv::dnn
