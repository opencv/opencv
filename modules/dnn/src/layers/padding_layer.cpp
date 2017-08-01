// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
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
        paddingDim = params.get<int>("padding_dim");
        padding = params.get<int>("padding");
        inputDims = params.get<int>("input_dims", 0);
        index = params.get<int>("index", 0);
        paddingValue = params.get<double>("value", 0);

        if(paddingDim < 0 || padding < 0)
            CV_Error(cv::Error::StsNotImplemented, "Negative padding and dim aren't supported");
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        outputs.clear();
        for(int i = 0; i < inputs.size(); i++)
        {
            MatShape shape = inputs[i];
            int dim = getPadDim(shape);
            CV_Assert(dim < shape.size());

            shape[dim] += padding;
            outputs.push_back(shape);
        }

        return false;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide();
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        for(int i = 0; i < inputs.size(); i++)
        {
            outputs[i] = paddingValue;
            const Mat& inp = *inputs[i];
            Mat& out = outputs[i];
            int dims = inp.dims;
            MatShape inShape(inp.size.p, inp.size.p + dims);
            MatShape outShape(out.size.p, out.size.p + dims);
            int dim = getPadDim(inShape);

            int actualIndex = index;
            if(index == 0)
                actualIndex = inShape[dim];

            std::vector<std::pair<Range, Range> > srcDstRanges;
            srcDstRanges.push_back(std::make_pair(Range(0, actualIndex), Range(0, actualIndex)));
            srcDstRanges.push_back(std::make_pair(Range(actualIndex, inShape[dim]),
                                                  Range(actualIndex + padding, outShape[dim])));

            std::vector<Range> srcRanges(dims, Range::all()), dstRanges = srcRanges;

            for(int j = 0; j < srcDstRanges.size(); j++)
            {
                if(!srcDstRanges[j].first.empty())
                {
                    srcRanges[dim] = srcDstRanges[j].first;
                    dstRanges[dim] = srcDstRanges[j].second;
                    Mat dst = out(&dstRanges[0]);
                    Mat src = inp(&srcRanges[0]).clone();
                    src.copyTo(dst);
                }
            }
        }
    }

    int getPadDim(const MatShape& shape) const
    {
        return inputDims > 0 && (int)shape.size() > inputDims ? paddingDim + 1 : paddingDim;
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        int inW, inH, inC, inN;
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        getCanonicalSize(inputBuffer, &inW, &inH, &inC, &inN);

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));
        Halide::Func padded =
            Halide::BoundaryConditions::constant_exterior(inputBuffer, paddingValue);
        top(x, y, c, n) = padded(x, y, c, n);
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

    int paddingDim, padding, inputDims, index;
    float paddingValue;
};

Ptr<PaddingLayer> PaddingLayer::create(const LayerParams &params)
{
    return Ptr<PaddingLayer>(new PaddingLayerImpl(params));
}

}
}
