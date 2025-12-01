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
    template<typename T>
    void fillRandomNormalImpl(Mat& out, float mean, float scale,
                              bool has_seed, float seed)
    {
        CV_Assert(CV_MAT_DEPTH(out.type()) == DataType<T>::depth);
        CV_Assert(out.isContinuous());

        RNG rng;
        if (has_seed)
        {
            uint64 seed_u64 = (uint64)std::llround((double)seed);
            rng.state = seed_u64 ? seed_u64 : 0x12345678ULL;
        }

        const double mean_d = static_cast<double>(mean);
        const double scale_d = static_cast<double>(scale);

        T* data = out.ptr<T>();
        const size_t N = static_cast<size_t>(out.total() * out.channels());
        for (size_t i = 0; i < N; ++i)
        {
            const double v = rng.gaussian(scale_d) + mean_d;
            data[i] = saturate_cast<T>(v);
        }
    }

    void fillRandomNormal(Mat& out, float mean, float scale,
                          bool has_seed, float seed)
    {
        const int depth = CV_MAT_DEPTH(out.type());
        switch (depth)
        {
        case CV_32F:
            fillRandomNormalImpl<float>(out, mean, scale, has_seed, seed);
            break;
        case CV_64F:
            fillRandomNormalImpl<double>(out, mean, scale, has_seed, seed);
            break;
        case CV_8U:
            fillRandomNormalImpl<uint8_t>(out, mean, scale, has_seed, seed);
            break;
        case CV_8S:
            fillRandomNormalImpl<int8_t>(out, mean, scale, has_seed, seed);
            break;
        case CV_32S:
            fillRandomNormalImpl<int32_t>(out, mean, scale, has_seed, seed);
            break;
        case CV_64S:
            fillRandomNormalImpl<int64_t>(out, mean, scale, has_seed, seed);
            break;
        case CV_16F:
        case CV_16BF:
        {
            Mat tmp(out.size(), CV_MAKETYPE(CV_32F, out.channels()));
            fillRandomNormalImpl<float>(tmp, mean, scale, has_seed, seed);
            tmp.convertTo(out, out.type());
            break;
        }
        default:
            CV_Error_(Error::StsNotImplemented,
                      ("RandomNormalLike: invalid/unsupported tensor type: %s",
                       typeToString(out.type()).c_str()));
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
        output_dtype = params.get<int>("output_dtype", -1);
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
        int outType = output_dtype >= 0 ? output_dtype : inputs[0];
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

        int outType = output_dtype >= 0 ? output_dtype : inp.type();

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
            Mat temp(outShape, outType);
            fillRandomNormal(temp, mean, scale, has_seed, seed);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }

private:
    int output_dtype; // OpenCV cv::Mat type (e.g., CV_32F). -1 means follow input type
};

Ptr<Layer> RandomNormalLikeLayer::create(const LayerParams& params)
{
    return makePtr<RandomNormalLikeLayerImpl>(params);
}

}
}
