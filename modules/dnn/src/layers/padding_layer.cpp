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
#include "../op_halide.hpp"
#include <vector>

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
        dstRanges.resize(paddings.size());

        int offset = 0;
        if (inputDims != -1 && inputs[0].dims != inputDims)
        {
            dstRanges.insert(dstRanges.begin(), Range::all());
            offset = 1;
        }

        for (int i = 0; i < paddings.size(); ++i)
        {
            dstRanges[offset + i].start = paddings[i].first;
            dstRanges[offset + i].end = paddings[i].first + inpShape[offset + i];
        }

        // Add the rest of dimensions.
        for (int i = dstRanges.size(); i < inputs[0].dims; ++i)
            dstRanges.push_back(Range::all());
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_HALIDE && haveHalide() && dstRanges.size() == 4;
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
                convertFp16(paddingValue_fp32, paddingValue_fp16);
                outputs[0].setTo(paddingValue_fp16[0]);
            }
            else
                outputs[0].setTo(paddingValue);
            inputs[0].copyTo(outputs[0](dstRanges));
        }
        else if (paddingType == "reflect")
        {
            CV_Assert(inputs.size() == 1);
            CV_Assert(outputs.size() == 1);
            CV_Assert(inputs[0].dims == 4);
            CV_Assert(outputs[0].dims == 4);

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
            CV_CheckLT(padTop, inpHeight, ""); CV_CheckLT(padBottom, inpHeight, "");
            CV_CheckLT(padLeft, inpWidth, ""); CV_CheckLT(padRight, inpWidth, "");

            for (size_t n = 0; n < inputs[0].size[0]; ++n)
            {
                for (size_t ch = 0; ch < inputs[0].size[1]; ++ch)
                {
                    copyMakeBorder(getPlane(inputs[0], n, ch),
                                   getPlane(outputs[0], n, ch),
                                   padTop, padBottom, padLeft, padRight,
                                   BORDER_REFLECT_101);
                }
            }
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown padding type: " + paddingType);
    }

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
