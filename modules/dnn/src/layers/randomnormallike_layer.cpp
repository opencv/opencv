// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
// Copyright (C) 2025, BigVision LLC, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"
#include <cmath>

namespace cv
{
namespace dnn
{

/*
    RandomNormalLike layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html

    Supported Opsets: 1-22
*/

namespace
{
    void fillRandomNormal(OutputArray out, float mean, float scale,
                          bool has_seed, float seed)
    {
        CV_Assert(out.isMat() || out.isUMat());
        const Scalar mean_s = Scalar::all(mean);
        const Scalar scale_s = Scalar::all(scale);

        RNG local_rng;
        if (has_seed)
        {
            uint64 seed_u64 = (uint64)std::llround((double)seed);
            if (!seed_u64)
                seed_u64 = 0x12345678ULL;
            local_rng = RNG(seed_u64);
        }

        RNG& rng = has_seed ? local_rng : theRNG();

        if (out.isMat())
        {
            Mat& m = out.getMatRef();
            rng.fill(m, RNG::NORMAL, mean_s, scale_s);
        }
        else
        {
            UMat& u = out.getUMatRef();
            rng.fill(u, RNG::NORMAL, mean_s, scale_s);
        }
    }
}

class RandomNormalLikeLayerImpl CV_FINAL : public RandomNormalLikeLayer
{
public:
    RandomNormalLikeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        mean = params.get<float>("mean", 0.f);
        scale = params.get<float>("scale", 1.f);
        int dt = params.get<int>("output_dtype", -1);
        outputType = dt >= 0 ? onnxDataTypeToCV(static_cast<OnnxDataType>(dt)) : -1;
        has_seed = params.has("seed");
        seed = has_seed ? params.get<float>("seed") : 0.f;

        CV_Assert(scale >= 0.f);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

    virtual bool dynamicOutputShapes() const CV_OVERRIDE
    {
        return false;
    }

    bool getMemoryShapes(const std::vector<MatShape>& inputs,
                         const int requiredOutputs,
                         std::vector<MatShape>& outputs,
                         std::vector<MatShape>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == (size_t)1);
        CV_Assert(requiredOutputs == 1);
        outputs.assign(1, inputs[0]);
        internals.clear();
        return true;
    }

    void getTypes(const std::vector<MatType>& inputs,
                  const int requiredOutputs,
                  const int requiredInternals,
                  std::vector<MatType>& outputs,
                  std::vector<MatType>& internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == (size_t)1);
        int outType = outputType >= 0 ? outputType : inputs[0];
        outputs.assign(1, outType);
        CV_Assert(requiredInternals == 0);
        internals.clear();
    }

    void forward(InputArrayOfArrays inputs_arr,
                 OutputArrayOfArrays outputs_arr,
                 OutputArrayOfArrays) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Size size = inputs_arr.size();
        int ninputs = size.area();
        CV_Assert(ninputs == 1);

        Mat inp = inputs_arr.getMat(0);
        MatShape outShape = inp.shape();

        int outType = outputType >= 0 ? outputType : inp.type();

        auto kind = outputs_arr.kind();
        if (kind == _InputArray::STD_VECTOR_MAT) {
            std::vector<Mat>& outs = outputs_arr.getMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, outType);
            fillRandomNormal(outs[0], mean, scale, has_seed, seed);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, outType);
            fillRandomNormal(outs[0], mean, scale, has_seed, seed);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }

private:
    int outputType;
};

Ptr<Layer> RandomNormalLikeLayer::create(const LayerParams& params)
{
    return makePtr<RandomNormalLikeLayerImpl>(params);
}

}
}
