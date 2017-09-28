// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of padding layer, which adds paddings to input blob.
*/

#include "../precomp.hpp"
#include "op_halide.hpp"
#include <vector>

namespace cv
{
namespace dnn
{

class PaddingLayerImpl : public PaddingLayer
{
public:
    PaddingLayerImpl(const LayerParams &params)
    {
        setParamsFrom(params);
        paddingValue = params.get<float>("value", 0);
        inputDims = params.get<int>("input_dims", -1);

        CV_Assert(params.has("paddings"));
        const DictValue& paddingsParam = params.get("paddings");
        CV_Assert((paddingsParam.size() & 1) == 0);

        paddings.resize(paddingsParam.size() / 2);
        for (int i = 0; i < paddings.size(); ++i)
        {
            paddings[i].first = paddingsParam.get<int>(i * 2);  // Pad before.
            paddings[i].second = paddingsParam.get<int>(i * 2 + 1);  // Pad after.
            CV_Assert(paddings[i].first >= 0, paddings[i].second >= 0);
        }
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
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

    void finalize(const std::vector<Mat*> &inputs, std::vector<Mat> &outputs)
    {
        // Compute dstRanges.
        const MatSize& inpShape = inputs[0]->size;
        dstRanges.resize(paddings.size());

        int offset = 0;
        if (inputDims != -1 && inputs[0]->dims != inputDims)
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
        for (int i = dstRanges.size(); i < inputs[0]->dims; ++i)
            dstRanges.push_back(Range::all());
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide() && dstRanges.size() == 4;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        outputs[0].setTo(paddingValue);
        inputs[0]->copyTo(outputs[0](dstRanges));
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
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
};

Ptr<PaddingLayer> PaddingLayer::create(const LayerParams &params)
{
    return Ptr<PaddingLayer>(new PaddingLayerImpl(params));
}

}
}
