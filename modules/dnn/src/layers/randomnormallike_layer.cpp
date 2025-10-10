// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../net_impl.hpp"

namespace cv
{
namespace dnn
{

/*
    RandomNormalLike layer, as defined in ONNX specification:
    https://onnx.ai/onnx/operators/onnx__RandomNormalLike.html

    Opset's 1+ are covered (attributes: dtype, mean, scale, seed).
*/

class RandomNormalLikeLayerImpl CV_FINAL : public Layer
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

    void finalize(InputArrayOfArrays, OutputArrayOfArrays) CV_OVERRIDE {}

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
            fillWithRandomNormal(outs[0]);
        } else if (kind == _InputArray::STD_VECTOR_UMAT) {
            std::vector<UMat>& outs = outputs_arr.getUMatVecRef();
            outs.resize(1);
            outs[0].fit(outShape, outType);
            Mat temp(outShape, outType);
            fillWithRandomNormal(temp);
            temp.copyTo(outs[0]);
        } else {
            CV_Error(Error::StsNotImplemented, "");
        }
    }

private:
    void fillWithRandomNormal(Mat& out)
    {
        // Only basic numeric types supported
        int t = out.type();
        if (t != CV_32F && t != CV_64F && t != CV_16F && t != CV_16BF && t != CV_8U && t != CV_8S && t != CV_32S && t != CV_64S)
            CV_Error_(Error::StsNotImplemented, ("invalid/unsupported tensor type: %s", typeToString(t).c_str()));

        RNG rng;
        if (has_seed) {
            uint64 seed_u64 = (uint64)std::llround((double)seed);
            rng.state = seed_u64 ? seed_u64 : 0x12345678ULL;
        }

        if (t == CV_32F) {
            rng.fill(out, RNG::NORMAL, Scalar(mean), Scalar(scale));
        } else if (t == CV_64F) {
            rng.fill(out, RNG::NORMAL, Scalar((double)mean), Scalar((double)scale));
        } else if (t == CV_16F || t == CV_16BF) {
            Mat tmp(out.size, CV_32F);
            rng.fill(tmp, RNG::NORMAL, Scalar(mean), Scalar(scale));
            tmp.convertTo(out, t);
        } else if (t == CV_8U || t == CV_8S || t == CV_32S || t == CV_64S) {
            Mat tmp(out.size, CV_32F);
            rng.fill(tmp, RNG::NORMAL, Scalar(mean), Scalar(scale));
            tmp.convertTo(out, t);
        } else {
            CV_Error_(Error::StsNotImplemented, ("invalid/unsupported tensor type: %s", typeToString(t).c_str()));
        }
    }

    float mean;
    float scale;
    int output_dtype; // OpenCV cv::Mat type (e.g., CV_32F). -1 means follow input type
    bool has_seed;
    float seed;
};

Ptr<Layer> RandomNormalLikeLayer::create(const LayerParams& params)
{
    return makePtr<RandomNormalLikeLayerImpl>(params);
}

}
}
