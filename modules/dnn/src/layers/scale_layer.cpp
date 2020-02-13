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
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

#include <opencv2/dnn/shape_utils.hpp>

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
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_HALIDE ||
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

class DataAugmentationLayerImpl CV_FINAL : public DataAugmentationLayer
{
public:
    DataAugmentationLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        recompute_mean = params.get<int>("recompute_mean", 0);
        mean_per_pixel = params.get<bool>("mean_per_pixel", false);
        num_iter = 0;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(blobs[0].total() == 1);
        CV_Assert(blobs[1].total() == total(inputs[0], 1));
        CV_Assert(blobs[2].total() == inputs[0][1]);
        outputs.assign(1, inputs[0]);
        return true;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 1);
        CV_Assert((!mean_per_pixel && blobs.size() == 3) || blobs.size() >= 2);
        ++num_iter;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert_N(outputs.size() == 1, blobs.size() == 3, inputs.size() == 1);

        Mat &inpBlob = inputs[0];
        Mat &outBlob = outputs[0];

        float* inpData = (float*)inpBlob.data;
        float* outData = (float*)outBlob.data;

        Mat data_mean_cpu = blobs[1];
        Mat data_mean_per_channel_cpu = blobs[2];

        const int numWeights = blobs[1].total();
        CV_Assert(numWeights != 0);

        if (num_iter <= recompute_mean) {
            data_mean_cpu *= (num_iter - 1);
            const int batch = inputs[0].size[0];
            float alpha = 1.0 / batch;

            for (int i = 0; i < batch; ++i)
            {
                Mat inpSlice(1, numWeights, CV_32F, inpData);
                inpSlice = alpha * inpSlice;

                add(data_mean_cpu.reshape(1, 1), inpSlice, data_mean_cpu.reshape(1, 1));
                inpData += numWeights;
            }
            data_mean_cpu *= (1.0 / num_iter);

            int newsize[] = {blobs[1].size[1], (int)blobs[1].total(2)};
            reduce(data_mean_cpu.reshape(1, 2, &newsize[0]), data_mean_per_channel_cpu, 1, REDUCE_SUM, CV_32F);

            int area = blobs[1].total(2);
            data_mean_per_channel_cpu *= (1.0 / area);
        }

        MatShape inpShape = shape(inpBlob);

        inpData = (float*)inpBlob.data;
        if (mean_per_pixel) {
            int numSlices = inputs[0].size[0];
            for (int i = 0; i < numSlices; ++i)
            {
                Mat inpSlice(1, numWeights, CV_32F, inpData);
                Mat outSlice(1, numWeights, CV_32F, outData);

                add(inpSlice, (-1) * data_mean_cpu, outSlice);
                inpData += numWeights;
                outData += numWeights;
            }
        } else {
            int numSlices = inpShape[1];
            int count = numWeights / numSlices;

            for (int i = 0; i < numSlices; ++i)
            {
                Mat inpSlice(1, count, CV_32F, inpData);
                Mat outSlice(1, count, CV_32F, outData);
                float coeff = data_mean_per_channel_cpu.reshape(1, 1).at<float>(0, i);
                outSlice = inpSlice - coeff;

                inpData += count;
                outData += count;
            }
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV;
    }

private:
    int num_iter;
    int recompute_mean;
    bool mean_per_pixel;
    Mat data_mean_cpu;
    Mat data_mean_per_channel_cpu;
};

Ptr<DataAugmentationLayer> DataAugmentationLayer::create(const LayerParams& params)
{
    return Ptr<DataAugmentationLayer>(new DataAugmentationLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
