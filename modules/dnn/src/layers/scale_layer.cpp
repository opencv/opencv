// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of Scale layer.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "op_halide.hpp"
#include "op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>

namespace cv
{
namespace dnn
{

class ScaleLayerImpl : public ScaleLayer
{
public:
    ScaleLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasBias = params.get<bool>("bias_term", false);
        axis = params.get<int>("axis", 1);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() == 2 && blobs.empty() || blobs.size() == 1 + hasBias);
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide() ||
               backendId == DNN_BACKEND_INFERENCE_ENGINE && haveInfEngine();
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());
        CV_Assert(outputs.size() == 1, !blobs.empty() || inputs.size() == 2);

        Mat &inpBlob = *inputs[0];
        Mat &outBlob = outputs[0];
        Mat &weights = blobs.empty() ? *inputs[1] : blobs[0];
        Mat bias = hasBias ? blobs.back() : Mat();
        MatShape inpShape = shape(inpBlob);
        const int numWeights = weights.total();

        int endAxis;
        for (endAxis = axis + 1; endAxis <= inpBlob.dims; ++endAxis)
        {
            if (total(inpShape, axis, endAxis) == numWeights)
                break;
        }
        CV_Assert(total(inpShape, axis, endAxis) == numWeights,
                  !hasBias || numWeights == bias.total(),
                  inpBlob.type() == CV_32F && outBlob.type() == CV_32F);

        int numSlices = total(inpShape, 0, axis);
        float* inpData = (float*)inpBlob.data;
        float* outData = (float*)outBlob.data;

        if (endAxis != inpBlob.dims)
        {
            float* weightsData = (float*)weights.data;
            float* biasesData = hasBias ? (float*)bias.data : 0;
            int spatialSize = total(inpShape, endAxis);  // spatialSize != 1
            for (int i = 0; i < numSlices; ++i)
            {
                for (int j = 0; j < numWeights; ++j)
                {
                    float w = weightsData[j];
                    float b = hasBias ? biasesData[j] : 0;
                    Mat inpSlice(1, spatialSize, CV_32F, inpData);
                    Mat outSlice(1, spatialSize, CV_32F, outData);
                    inpSlice.convertTo(outSlice, CV_32F, w, b);
                    inpData += spatialSize;
                    outData += spatialSize;
                }
            }
        }
        else
        {
            for (int i = 0; i < numSlices; ++i)
            {
                Mat inpSlice(weights.dims, weights.size, CV_32F, inpData);
                Mat outSlice(weights.dims, weights.size, CV_32F, outData);
                multiply(inpSlice, weights, outSlice);
                if (hasBias)
                    add(outSlice, bias, outSlice);

                inpData += numWeights;
                outData += numWeights;
            }
        }
    }

    virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node)
    {
        switch (node->backendId)
        {
            case DNN_BACKEND_HALIDE:
            {
#ifdef HAVE_HALIDE
                auto base = node.dynamicCast<HalideBackendNode>();
                Halide::Func& input = base->funcs.back();
                Halide::Var x("x"), y("y"), c("c"), n("n");
                Halide::Func top = attachHalide(input(x, y, c, n));
                return Ptr<BackendNode>(new HalideBackendNode(base, top));
#endif  // HAVE_HALIDE
                break;
            }
            case DNN_BACKEND_INFERENCE_ENGINE:
            {
#ifdef HAVE_INF_ENGINE
                auto base = node.dynamicCast<InfEngineBackendNode>();
                auto conv = std::dynamic_pointer_cast<InferenceEngine::ConvolutionLayer>(base->layer);
                if (conv)
                {
                    Mat bias = hasBias ? blobs[1] : Mat();
                    fuseConvWeights(conv, blobs[0], bias);
                    return base;
                }
#endif  // HAVE_INF_ENGINE
                break;
            }
        }
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> input = halideBuffer(inputs[0]);
        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = attachHalide(input(x, y, c, n));
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_HALIDE
    // attachHalide can work both with Halide::Buffer and Halide::Func. In the
    // second case it will be a fusion.
    Halide::Func attachHalide(const Halide::Expr& input)
    {
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Var x("x"), y("y"), c("c"), n("n");

        const int numChannels = blobs[0].total();

        auto weights = wrapToHalideBuffer(blobs[0], {numChannels});
        Halide::Expr topExpr = input * weights(c);
        if (hasBias)
        {
            auto bias = wrapToHalideBuffer(blobs[1], {numChannels});
            topExpr += bias(c);
        }
        top(x, y, c, n) = topExpr;
        return top;
    }
#endif  // HAVE_HALIDE

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&)
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "ScaleShift";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::ScaleShiftLayer> ieLayer(new InferenceEngine::ScaleShiftLayer(lp));

        ieLayer->_weights = wrapToInfEngineBlob(blobs[0]);
        if (hasBias)
            ieLayer->_biases = wrapToInfEngineBlob(blobs[1]);

        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    void getScaleShift(Mat& scale, Mat& shift) const
    {
        scale = !blobs.empty() ? blobs[0] : Mat();
        shift = hasBias ? blobs[1] : Mat();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        long flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 2*total(inputs[i]);
        }
        return flops;
    }
};


Ptr<ScaleLayer> ScaleLayer::create(const LayerParams& params)
{
    return Ptr<ScaleLayer>(new ScaleLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
