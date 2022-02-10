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
#include "../op_webnn.hpp"

#include <opencv2/imgproc.hpp>
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
#ifdef HAVE_WEBNN
    mutable int dims;
    mutable int numChannels;
#endif
    ScaleLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        hasBias = params.get<bool>("bias_term", false);
        axis = params.get<int>("axis", 1);
        hasWeights = false;
        mode = params.get<String>("mode", "scale");
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        outputs.assign(1, inputs[0]);
#ifdef HAVE_WEBNN
        dims = inputs[0].size();
        numChannels = 1;
        if (inputs.size() > 1)
        {
            for (const size_t& dim : inputs[1])
                numChannels *= dim;
        }
#endif
        return true;
    }

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        hasWeights = blobs.size() == 2 || (blobs.size() <= 1 && !hasBias);
        CV_Assert((inputs.size() == 2 && blobs.empty()) || blobs.size() == (int)hasWeights + (int)hasBias);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (mode != "scale")
        {
            return backendId == DNN_BACKEND_OPENCV;
        }
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               backendId == DNN_BACKEND_HALIDE ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && axis == 1 && !blobs.empty()) ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && axis > 0) ||
               (backendId == DNN_BACKEND_WEBNN && axis >0);
    }

    template<typename T>
    void handleCompare(const Mat& a, const T& b, Mat& dst, const int spatialSize)
    {
        Mat out(1, spatialSize, CV_8U);
        if (mode == "equal")
            compare(a, b, out, CMP_EQ);
        else if (mode == "greater")
            compare(a, b, out, CMP_GT);
        else
            compare(a, b, out, CMP_LT);

        out.convertTo(dst, CV_32F, 1. / 255.);
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
        Mat weights = hasWeights ? (blobs.empty() ? inputs[1] : blobs[0]).reshape(1, 1) : Mat();;
        Mat bias = hasBias ? (blobs.empty() ? inputs[1] : blobs.back()).reshape(1, 1) : Mat();

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

                    if (mode == "scale")
                    {
                        inpSlice.convertTo(outSlice, CV_32F, w, b);
                    }
                    else
                    {
                        handleCompare(inpSlice, b, outSlice, spatialSize);
                    }

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
                {
                    if (mode == "scale")
                    {
                        add(inpSlice, bias, outSlice);
                    }
                    else
                    {
                        handleCompare(inpSlice, bias, outSlice, numWeights);
                    }
                }
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

        auto weightsMat = Mat(), biasMat = Mat();

        cuda4dnn::ScaleShiftConfiguration config;
        if (hasWeights)
        {
            if (blobs.empty())
            {
                config.scaleMode = cuda4dnn::ScaleShiftConfiguration::OpMode::UNTRAINABLE;
            }
            else
            {
                weightsMat = blobs[0];
                config.scaleMode = cuda4dnn::ScaleShiftConfiguration::OpMode::TRAINABLE;
            }
        }
        else
        {
            config.scaleMode = cuda4dnn::ScaleShiftConfiguration::OpMode::NONE;
        }

        if (hasBias)
        {
            if(blobs.empty())
            {
                config.shiftMode = cuda4dnn::ScaleShiftConfiguration::OpMode::UNTRAINABLE;
            }
            else
            {
                /* if the weights are provided, bias will be in blobs[1]; otherwise, it will be in blobs[0]
                 * in either case, it is at the end of the blobs vector => bias = blobs.back()
                 */
                biasMat = blobs.back();
                config.shiftMode = cuda4dnn::ScaleShiftConfiguration::OpMode::TRAINABLE;
            }
        }
        else
        {
            config.shiftMode = cuda4dnn::ScaleShiftConfiguration::OpMode::NONE;
        }

        config.axis = axis;

        return make_cuda_node<cuda4dnn::ScaleShiftOp>(preferableTarget, std::move(context->stream), config, weightsMat, biasMat);
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

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
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
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto ieInpNode0 = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        auto ieInpNode1 = nodes.size() > 1 ? nodes[1].dynamicCast<InfEngineNgraphNode>()->node : nullptr;

        size_t numChannels = 1;
        if (blobs.empty())
            for (const size_t& dim : ieInpNode1->get_shape())
                numChannels *= dim;
        else
            numChannels = blobs[0].total();

        std::vector<size_t> shape(ieInpNode0->get_shape().size(), 1);
        int cAxis = normalize_axis(axis, shape.size());
        shape[cAxis] = numChannels;

        auto node = ieInpNode0;
        if (hasWeights)
        {
            auto weight = blobs.empty() ? ieInpNode1 :
                          std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape(shape), blobs[0].data);

#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2021_2)
            node = std::make_shared<ngraph::op::v1::Multiply>(node, weight, ngraph::op::AutoBroadcastType::NUMPY);
#else
            node = std::make_shared<ngraph::op::v0::Multiply>(node, weight, ngraph::op::AutoBroadcastType::NUMPY);
#endif
        }
        if (hasBias || !hasWeights)
        {
            std::shared_ptr<ngraph::Node> bias;
            if (hasBias)
            {
                bias = blobs.empty() ? ieInpNode1 :
                       std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                                              ngraph::Shape(shape), blobs.back().data);
            }
            else
                bias = std::make_shared<ngraph::op::Constant>(ngraph::element::f32,
                                                              ngraph::Shape(shape), std::vector<float>(numChannels, 0).data());
            node = std::make_shared<ngraph::op::v1::Add>(node, bias, ngraph::op::AutoBroadcastType::NUMPY);
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(node));
    }
#endif  // HAVE_DNN_NGRAPH

#ifdef HAVE_WEBNN
    virtual Ptr<BackendNode> initWebnn(const std::vector<Ptr<BackendWrapper> >& inputs, const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        Ptr<WebnnBackendNode> node = nodes[0].dynamicCast<WebnnBackendNode>();
        auto& webnnInpOperand0 = node->operand;
        auto& webnnGraphBuilder = node->net->builder;
        auto webnnInpOperand1 = nodes.size() > 1 ? nodes[1].dynamicCast<WebnnBackendNode>()->operand : nullptr;
        auto webnnInpOperand2 = nodes.size() > 2 ? nodes[1].dynamicCast<WebnnBackendNode>()->operand : nullptr;
        std::vector<int32_t> shape(dims, 1);

        size_t channels = 1;
        if (blobs.empty())
            channels = numChannels;
        else
            channels = blobs[0].total();

        int cAxis = normalize_axis(axis, shape.size());
        shape[cAxis] = channels;

        ml::Operand operand = webnnInpOperand0;
        if (hasWeights)
        {
            ml::Operand webnnWeights = blobs.empty() ? webnnInpOperand1 : webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs[0]), blobs[0].data, blobs[0].total()*blobs[0].elemSize(), ml::OperandType::Float32);
            webnnWeights = webnnGraphBuilder.Reshape(webnnWeights, shape.data(), shape.size());
            operand = webnnGraphBuilder.Mul(operand, webnnWeights);
        }
        if (hasBias)
        {
            ml::Operand webnnBias;
            if(!hasWeights)
                webnnBias = blobs.empty() ? webnnInpOperand1 : webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs.back()), blobs.back().data, blobs.back().total()*blobs.back().elemSize(), ml::OperandType::Float32);
            else
                webnnBias = blobs.empty() ? webnnInpOperand2 : webnn::BuildConstant(webnnGraphBuilder, webnn::getShape(blobs.back()), blobs.back().data, blobs.back().total()*blobs.back().elemSize(), ml::OperandType::Float32);
            webnnBias = webnnGraphBuilder.Reshape(webnnBias, shape.data(), shape.size());
            operand = webnnGraphBuilder.Add(operand, webnnBias);
        }

        return Ptr<BackendNode>(new WebnnBackendNode(operand));
    }
#endif


    void getScaleShift(Mat& scale, Mat& shift) const CV_OVERRIDE
    {
        scale = (hasWeights && !blobs.empty()) ? blobs[0] : Mat();
        shift = (hasBias && !blobs.empty()) ? blobs.back() : Mat();
    }

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        params.set("input_scales", DictValue::arrayReal(scales[0].data(), scales[0].size()));
        params.set("input_zeropoints", DictValue::arrayInt(zeropoints[0].data(), zeropoints[0].size()));
        return true;
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

Ptr<Layer> CompareLayer::create(const LayerParams& params)
{
    LayerParams compareParams;
    compareParams.name = params.name;
    compareParams.type = "Scale";
    compareParams.blobs = params.blobs;
    compareParams.set("bias_term", true);
    compareParams.set("axis", 0);
    compareParams.set("mode", params.get<String>("mode"));
    return Ptr<ScaleLayer>(new ScaleLayerImpl(compareParams));
}

class DataAugmentationLayerImpl CV_FINAL : public DataAugmentationLayer
{
public:
    DataAugmentationLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        recompute_mean = params.get<int>("recompute_mean", 1);
        CV_CheckGT(recompute_mean, 0, "");
        mean_per_pixel = params.get<bool>("mean_per_pixel", false);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert_N(inputs.size() == 1, blobs.size() == 3);
        CV_Assert_N(blobs[0].total() == 1,
                    blobs[2].total() == inputs[0][1]);

        outputs.assign(1, inputs[0]);
        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        CV_Assert_N(outputs.size() == 1, blobs.size() == 3, inputs.size() == 1);
        int num_iter = 0;

        float* inpData = inputs[0].ptr<float>();
        float* outData = outputs[0].ptr<float>();

        Mat data_mean_cpu = blobs[1].clone();
        Mat mean_resize = Mat(inputs[0].size[3], inputs[0].size[2], CV_32FC3);
        Mat mean_3d = Mat(data_mean_cpu.size[3], data_mean_cpu.size[2], CV_32FC3, data_mean_cpu.ptr<float>(0));
        resize(mean_3d, mean_resize, Size(inputs[0].size[3], inputs[0].size[2]));
        int new_size[] = {1, mean_resize.channels(), mean_resize.cols, mean_resize.rows};
        Mat data_mean_cpu_resize = mean_resize.reshape(1, *new_size);
        Mat data_mean_per_channel_cpu = blobs[2].clone();

        const int numWeights = data_mean_cpu_resize.total();
        CV_Assert(numWeights != 0);

        ++num_iter;
        if (num_iter <= recompute_mean)
        {
            data_mean_cpu_resize *= (num_iter - 1);
            const int batch = inputs[0].size[0];
            float alpha = 1.0 / batch;

            for (int i = 0; i < batch; ++i)
            {
                Mat inpSlice(1, numWeights, CV_32F, inpData);
                inpSlice = alpha * inpSlice;

                add(data_mean_cpu_resize.reshape(1, 1), inpSlice, data_mean_cpu_resize.reshape(1, 1));
                inpData += numWeights;
            }
            data_mean_cpu_resize *= (1.0 / num_iter);

            int newsize[] = {inputs[0].size[1], (int)inputs[0].total(2)};
            reduce(data_mean_cpu_resize.reshape(1, 2, &newsize[0]), data_mean_per_channel_cpu, 1, REDUCE_SUM, CV_32F);

            int area = inputs[0].total(2);
            data_mean_per_channel_cpu *= (1.0 / area);
        }

        MatShape inpShape = shape(inputs[0]);

        inpData = inputs[0].ptr<float>();
        if (mean_per_pixel)
        {
            int numSlices = inputs[0].size[0];
            for (int i = 0; i < numSlices; ++i)
            {
                Mat inpSlice(1, numWeights, CV_32F, inpData);
                Mat outSlice(1, numWeights, CV_32F, outData);

                add(inpSlice, (-1) * data_mean_cpu_resize, outSlice);
                inpData += numWeights;
                outData += numWeights;
            }
        }
        else
        {
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

private:
    int recompute_mean;
    bool mean_per_pixel;
};

Ptr<DataAugmentationLayer> DataAugmentationLayer::create(const LayerParams& params)
{
    return Ptr<DataAugmentationLayer>(new DataAugmentationLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
