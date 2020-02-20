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
#include "../op_cuda.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/scale_shift.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class ScaleLayerImpl CV_FINAL : public ScaleLayer
{
public:
    ScaleLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasBias = params.get<bool>("bias_term", false);
        axis = params.get<int>("axis", 1);
        hasWeights = false;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        hasWeights = blobs.size() == 2 || (blobs.size() == 1 && !hasBias);
        CV_Assert((inputs.size() == 2 && blobs.empty()) || blobs.size() == (int)hasWeights + (int)hasBias);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_HALIDE ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && axis == 1) ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && axis > 0);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert_N(outputs.size() == 1, !blobs.empty() || inputs.size() == 2);

        Mat &inpBlob = inputs[0];
        Mat &outBlob = outputs[0];
        // There is a mode when we multiply a first blob by a second one
        // instead of trainable weights.
        Mat weights = blobs.empty() ? inputs[1] : (hasWeights ? blobs[0] : Mat());
        Mat bias = hasBias ? blobs.back().reshape(1, 1) : Mat();
        if (!weights.empty())
            weights = weights.reshape(1, 1);
        MatShape inpShape = shape(inpBlob);
        const int numWeights = !weights.empty() ? weights.total() : bias.total();
        CV_Assert(numWeights != 0);
        if (hasWeights && hasBias)
            CV_CheckEQ(weights.total(), bias.total(), "Incompatible weights/bias blobs");

        int endAxis;
        for (endAxis = axis + 1; endAxis <= inpBlob.dims; ++endAxis)
        {
            if (total(inpShape, axis, endAxis) == numWeights)
                break;
        }
        CV_Assert(total(inpShape, axis, endAxis) == numWeights);
        CV_Assert(!hasBias || numWeights == bias.total());
        CV_CheckTypeEQ(inpBlob.type(), CV_32FC1, ""); CV_CheckTypeEQ(outBlob.type(), CV_32FC1, "");

        int numSlices = total(inpShape, 0, axis);
        float* inpData = (float*)inpBlob.data;
        float* outData = (float*)outBlob.data;

        if (endAxis != inpBlob.dims)
        {
            float* weightsData = !weights.empty() ? (float*)weights.data : 0;
            float* biasesData = hasBias ? (float*)bias.data : 0;
            int spatialSize = total(inpShape, endAxis);  // spatialSize != 1
            for (int i = 0; i < numSlices; ++i)
            {
                for (int j = 0; j < numWeights; ++j)
                {
                    float w = weightsData ? weightsData[j] : 1;
                    float b = biasesData ? biasesData[j] : 0;
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
                Mat inpSlice(1, numWeights, CV_32F, inpData);
                Mat outSlice(1, numWeights, CV_32F, outData);
                if (!weights.empty())
                {
                    multiply(inpSlice, weights, outSlice);
                    if (hasBias)
                        add(outSlice, bias, outSlice);
                }
                else if (hasBias)
                    add(inpSlice, bias, outSlice);
                inpData += numWeights;
                outData += numWeights;
            }
        }
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        CV_Assert(!blobs.empty() || inputs.size() == 2);

        cv::Mat weightsMat = hasWeights ? blobs[0] : Mat();

        /* if the weights are provided, bias will be in blobs[1]; otherwise, it will be in blobs[0]
         * in either case, it is at the end of the blobs vector => bias = blobs.back()
         */
        cv::Mat biasMat = hasBias ? blobs.back() : Mat();

        return make_cuda_node<cuda4dnn::ScaleShiftOp>(preferableTarget, std::move(context->stream), axis, weightsMat, biasMat);
    }
#endif

    virtual Ptr<BackendNode> tryAttach(const Ptr<BackendNode>& node) CV_OVERRIDE
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
        }
        return Ptr<BackendNode>();
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
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

        Halide::Expr topExpr = input;
        if (hasWeights)
        {
            auto weights = wrapToHalideBuffer(blobs[0], {numChannels});
            topExpr *= weights(c);
        }
        if (hasBias)
        {
            auto bias = wrapToHalideBuffer(blobs.back(), {numChannels});
            topExpr += bias(c);
        }
        top(x, y, c, n) = topExpr;
        return top;
    }
#endif  // HAVE_HALIDE

#ifdef HAVE_INF_ENGINE
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        InferenceEngine::Builder::Layer l = InferenceEngine::Builder::ScaleShiftLayer(name);

        CV_Assert(!blobs.empty());
        const size_t numChannels = blobs[0].total();
        if (hasWeights)
        {
            addConstantData("weights", wrapToInfEngineBlob(blobs[0], {numChannels}, InferenceEngine::Layout::C), l);
        }
        else
        {
            auto weights = InferenceEngine::make_shared_blob<float>({
                               InferenceEngine::Precision::FP32, {(size_t)numChannels},
                               InferenceEngine::Layout::C
                           });
            weights->allocate();
            float* buf = weights->buffer().as<float*>();
            std::fill(buf, buf + numChannels, 1);
            addConstantData("weights", weights, l);
        }
        if (hasBias)
            addConstantData("biases", wrapToInfEngineBlob(blobs.back(), {numChannels}, InferenceEngine::Layout::C), l);
        return Ptr<BackendNode>(new InfEngineBackendNode(l));
    }
#endif  // HAVE_INF_ENGINE


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(!blobs.empty());
        const size_t numChannels = blobs[0].total();
        auto ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        std::vector<size_t> shape(ieInpNode->get_shape().size(), 1);
        int cAxis = clamp(axis, shape.size());
        shape[cAxis] = numChannels;

        auto node = ieInpNode;
        if (hasWeights)
        {
            auto weight = std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                                                 ngraph::Shape(shape), blobs[0].data);
            node = std::make_shared<ngraph::op::v1::Multiply>(node, weight, ngraph::op::AutoBroadcastType::NUMPY);
        }
        if (hasBias || !hasWeights)
        {
            auto bias = hasBias ?
                        std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                                               ngraph::Shape(shape), blobs.back().data) :
                        std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                                               ngraph::Shape(shape), std::vector<float>(numChannels, 0).data());
            node = std::make_shared<ngraph::op::v1::Add>(node, bias, ngraph::op::AutoBroadcastType::NUMPY);
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif  // HAVE_DNN_NGRAPH

    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale = hasWeights ? blobs[0] : Mat();
        shift = hasBias ? blobs.back() : Mat();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning
        long flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 2*total(inputs[i]);
        }
        return flops;
    }

private:
    bool hasWeights;
};


Ptr<ScaleLayer> ScaleLayer::create(const LayerParams& params)
{
    return Ptr<ScaleLayer>(new ScaleLayerImpl(params));
}

Ptr<Layer> ShiftLayer::create(const LayerParams& params)
{
    LayerParams scaleParams;
    scaleParams.name = params.name;
    scaleParams.type = "Scale";
    scaleParams.blobs = params.blobs;
    scaleParams.set("bias_term", true);
    scaleParams.set("axis", 0);
    return Ptr<ScaleLayer>(new ScaleLayerImpl(scaleParams));
}

}  // namespace dnn
}  // namespace cv
