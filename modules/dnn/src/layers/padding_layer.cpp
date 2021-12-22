// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of padding layer, which adds paddings to input blob.
*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "../op_cuda.hpp"
#include "../op_halide.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

#include <vector>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/padding.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class PaddingLayerImpl CV_FINAL : public PaddingLayer
{
public:
    PaddingLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        paddingValue = params.get<float>("value", 0);
        inputDims = params.get<int>("input_dims", -1);
        paddingType = params.get<String>("type", "constant");

        CV_Assert(params.has("paddings"));
        const DictValue& paddingsParam = params.get("paddings");
        CV_Assert((paddingsParam.size() & 1) == 0);

        paddings.resize(paddingsParam.size() / 2);
        for (int i = 0; i < paddings.size(); ++i)
        {
            paddings[i].first = paddingsParam.get<int>(i * 2);  // Pad before.
            paddings[i].second = paddingsParam.get<int>(i * 2 + 1);  // Pad after.
            CV_Assert_N(paddings[i].first >= 0, paddings[i].second >= 0);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        const MatShape& inpShape = inputs[0];
        CV_Assert(inpShape.size() >= paddings.size());
        CV_Assert(inputDims == -1 || inpShape.size() == inputDims || inpShape.size() > paddings.size());

        outputs.resize(1, inpShape);
        int offset = (inputDims == -1 ? 0 : (inpShape.size() > inputDims ? 1 : 0));
        for (int i = 0; i < paddings.size(); ++i)
        {
            outputs[0][offset + i] = inpShape[offset + i] + paddings[i].first + paddings[i].second;
        }
        return false;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        // Compute dstRanges.
        const MatSize& inpShape = inputs[0].size;

        if (inputDims != -1 && inputs[0].dims != inputDims)
        {
            paddings.insert(paddings.begin(), std::make_pair(0, 0));
        }

        dstRanges.resize(paddings.size());
        for (int i = 0; i < paddings.size(); ++i)
        {
            dstRanges[i].start = paddings[i].first;
            dstRanges[i].end = paddings[i].first + inpShape[i];
        }

        // Add the rest of dimensions.
        for (int i = dstRanges.size(); i < inputs[0].dims; ++i)
        {
            dstRanges.push_back(Range::all());
            paddings.push_back(std::make_pair(0, 0));
        }
        inputDims = -1;  // Next time paddings are filled for all the dimensions.
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            bool isMyriad = preferableTarget == DNN_TARGET_MYRIAD || preferableTarget == DNN_TARGET_HDDL;
            if (INF_ENGINE_VER_MAJOR_GE(INF_ENGINE_RELEASE_2019R1) && isMyriad)
                return dstRanges.size() == 4 && paddings[0].first == 0 && paddings[0].second == 0;

            return (dstRanges.size() <= 4 || !isArmComputePlugin());
        }
#endif
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               (backendId == DNN_BACKEND_HALIDE && haveHalide() && dstRanges.size() == 4);
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        if (paddingType == "constant")
        {
            if (inputs_arr.depth() == CV_16S)
            {
                std::vector<float> paddingValue_fp32(1, paddingValue);
                std::vector<int16_t> paddingValue_fp16(1);
                cv::convertFp16(paddingValue_fp32, paddingValue_fp16);
                outputs[0].setTo(paddingValue_fp16[0]);
            }
            else if (inputs_arr.depth() == CV_8S)
                outputs[0].setTo(saturate_cast<int8_t>(paddingValue));
            else
                outputs[0].setTo(paddingValue);
            inputs[0].copyTo(outputs[0](dstRanges));
        }
        else if (paddingType == "reflect" || paddingType == "edge")
        {
            CV_Assert(inputs.size() == 1);
            CV_Assert(outputs.size() == 1);
            CV_Assert(inputs[0].dims == 4);
            CV_Assert(outputs[0].dims == 4);
            int borderType = paddingType == "reflect" ? BORDER_REFLECT_101 : BORDER_REPLICATE;

            if (inputs[0].size[0] != outputs[0].size[0] || inputs[0].size[1] != outputs[0].size[1])
                CV_Error(Error::StsNotImplemented, "Only spatial reflection padding is supported.");

            const int inpHeight = inputs[0].size[2];
            const int inpWidth = inputs[0].size[3];
            const int outHeight = outputs[0].size[2];
            const int outWidth = outputs[0].size[3];
            const int padTop = dstRanges[2].start;
            const int padBottom = outHeight - dstRanges[2].end;
            const int padLeft = dstRanges[3].start;
            const int padRight = outWidth - dstRanges[3].end;
            CV_CheckLE(padTop, inpHeight, ""); CV_CheckLE(padBottom, inpHeight, "");
            CV_CheckLE(padLeft, inpWidth, ""); CV_CheckLE(padRight, inpWidth, "");

            for (size_t n = 0; n < inputs[0].size[0]; ++n)
            {
                for (size_t ch = 0; ch < inputs[0].size[1]; ++ch)
                {
                    copyMakeBorder(getPlane(inputs[0], n, ch),
                                   getPlane(outputs[0], n, ch),
                                   padTop, padBottom, padLeft, padRight,
                                   borderType);
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown padding type: " + paddingType);
    }

#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);

        cuda4dnn::PaddingType ptype;
        if (paddingType == "constant")
            ptype = PaddingType::CONSTANT;
        else if (paddingType == "reflect")
            ptype = PaddingType::REFLECTION101;
        else
            CV_Error(Error::StsNotImplemented, "Unsupported padding mode");

        return make_cuda_node<cuda4dnn::PaddingOp>(preferableTarget, std::move(context->stream), ptype, paddingValue, dstRanges);
    }
#endif

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs) CV_OVERRIDE
    {
#ifdef HAVE_HALIDE
        int inW, inH, inC, inN;
        int minN = std::max(dstRanges[0].start, 0);
        int minC = std::max(dstRanges[1].start, 0);
        int minY = std::max(dstRanges[2].start, 0);
        int minX = std::max(dstRanges[3].start, 0);
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        getCanonicalSize(inputBuffer, &inW, &inH, &inC, &inN);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Func padded =
            Halide::BoundaryConditions::constant_exterior(inputBuffer, paddingValue);
        top(x, y, c, n) = padded(x - minX, y - minY, c - minC, n - minN);
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        InferenceEngine::Builder::Layer ieLayer(name);
        ieLayer.setName(name);
        ieLayer.setType("Pad");

        std::vector<int> begins(paddings.size(), 0), ends(paddings.size(), 0);
        for (int i = 0; i < paddings.size(); ++i)
        {
            begins[i] = paddings[i].first;
            ends[i] = paddings[i].second;
        }
        ieLayer.getParameters()["pads_begin"] = begins;
        ieLayer.getParameters()["pads_end"] = ends;
        ieLayer.getParameters()["pad_mode"] = paddingType;
        if (paddingType == "constant")
            ieLayer.getParameters()["pad_value"] = paddingValue;

        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(1));
        ieLayer.setOutputPorts(std::vector<InferenceEngine::Port>(1));
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        std::vector<int64_t> begins(paddings.size(), 0), ends(paddings.size(), 0);
        for (int i = 0; i < paddings.size(); ++i)
        {
            begins[i] = static_cast<int64_t>(paddings[i].first);
            ends[i]   = static_cast<int64_t>(paddings[i].second);
        }
        auto padding_below = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{begins.size()}, begins.data());
        auto padding_above = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ends.size()}, ends.data());
        auto pad_mode = paddingType == "constant" ? ngraph::op::PadMode::CONSTANT : ngraph::op::PadMode::REFLECT; // SYMMETRIC
        auto arg_pad_value = std::make_shared<ngraph::op::Constant>(ngraph::element::f32, ngraph::Shape{}, &paddingValue);;

        auto pad = paddingType == "constant" ?
             std::make_shared<ngraph::op::v1::Pad>(ieInpNode, padding_below, padding_above, arg_pad_value, pad_mode) :
             std::make_shared<ngraph::op::v1::Pad>(ieInpNode, padding_below, padding_above, pad_mode);
        return Ptr<BackendNode>(new InfEngineNgraphNode(pad));
    }
#endif

    virtual bool tryQuantize(const std::vector<std::vector<float> > &scales,
                             const std::vector<std::vector<int> > &zeropoints, LayerParams& params) CV_OVERRIDE
    {
        float outputScale = scales[1][0];
        int outputZp = zeropoints[1][0];
        float padValue = outputZp + std::round(params.get<float>("value", 0)/outputScale);
        params.set("value", padValue);
        return true;
    }

private:
    std::vector<std::pair<int, int> > paddings;  // Pairs pad before, pad after.
    std::vector<Range> dstRanges;
    int inputDims;
    float paddingValue;
    std::string paddingType;
};

Ptr<PaddingLayer> PaddingLayer::create(const LayerParams &params)
{
    return Ptr<PaddingLayer>(new PaddingLayerImpl(params));
}

}
}
