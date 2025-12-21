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
            seed = params.get<double>("seed");
        }

        depth = params.get<int>("depth", CV_32F);
        if (params.has("dtype"))
        {
            depth = onnxDataTypeToCV(static_cast<OnnxDataType>(params.get<int>("dtype")));
        }
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

        Mat out = outputs[0];
        const int desiredDepth = depth;
        const bool needRecreate = out.depth() != desiredDepth;
        Mat outBlob;
        if (needRecreate)
        {
            const int dims = out.dims;
            const int* sizes = out.size.p;
            outBlob = Mat(dims, sizes, CV_MAKETYPE(desiredDepth, out.channels()));
        }
        else
        {
            outBlob = out;
        }

        RNG seededRng;
        RNG* rng = &theRNG();
        if (hasSeed)
        {
            Cv64suf u;
            u.f = seed;
            seededRng = RNG(u.u ? u.u : 1);
            rng = &seededRng;
        }

        if (outBlob.depth() == CV_32F || outBlob.depth() == CV_64F || outBlob.depth() == CV_16F)
        {
            rng->fill(outBlob, RNG::NORMAL, mean, scale);
        }
        else
        {
            Mat tmp(outBlob.size.dims(), outBlob.size.p, CV_32F);
            rng->fill(tmp, RNG::NORMAL, mean, scale);
            tmp.convertTo(outBlob, outBlob.type());
        }

        if (needRecreate)
        {
            outputs_arr.assign(std::vector<Mat>{outBlob});
        }
    }

private:
    double mean;
    double scale;
    bool hasSeed = false;
    double seed = 0.0;
    int depth;
};

CV_DNN_REGISTER_LAYER_CLASS_STATIC(RandomNormalLike, RandomNormalLikeLayerImpl);

}} // namespace cv::dnn
