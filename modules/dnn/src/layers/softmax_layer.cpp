/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Copyright (C) 2017, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#include "../precomp.hpp"
#include "layers_common.hpp"
#include "op_halide.hpp"
#include <algorithm>
#include <stdlib.h>
using std::max;

namespace cv
{
namespace dnn
{

class SoftMaxLayerImpl : public SoftmaxLayer
{
public:

    SoftMaxLayerImpl(const LayerParams& params)
    {
        axisRaw = params.get<int>("axis", 1);
        logSoftMax = params.get<int>("log_softmax", false);
        setParamsFrom(params);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        bool inplace = Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        MatShape shape = inputs[0];
        int cAxis = clamp(axisRaw, shape.size());
        shape[cAxis] = 1;
        internals.assign(1, shape);
        return inplace;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT ||
               backendId == DNN_BACKEND_HALIDE && haveHalide() && axisRaw == 1;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        const Mat &src = *inputs[0];
        Mat &dst = outputs[0];

        int axis = clamp(axisRaw, src.dims);
        size_t outerSize = src.total(0, axis), channels = src.size[axis],
                innerSize = src.total(axis + 1);

        CV_Assert(src.type() == CV_32F);
        CV_Assert(src.isContinuous() && dst.isContinuous());

        const float *srcPtr = src.ptr<float>();
        float *dstPtr = dst.ptr<float>();
        float *bufPtr = internals[0].ptr<float>();

        size_t outerStep = src.total(axis);
        size_t cnStep = src.total(axis + 1);

        //compute max along axis
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            memcpy(bufPtr + bufOffset, srcPtr + srcOffset, innerSize * sizeof(float));

            for (size_t cnDim = 1; cnDim < channels; cnDim++)
            {
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] = std::max(bufPtr[bufOffset + i], srcPtr[srcOffset + cnDim * cnStep + i]);
            }
        }

        //subtract max
        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                const int offset = srcOffset + cnDim * cnStep;
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[offset + i] = srcPtr[offset + i] - bufPtr[bufOffset + i];
            }
        }

        cv::exp(dst, dst);

        for (size_t outerDim = 0; outerDim < outerSize; outerDim++)
        {
            size_t srcOffset = outerDim * outerStep;
            size_t bufOffset = outerDim * cnStep;

            //sum exp along axis
            for (size_t i = 0; i < innerSize; i++)
                bufPtr[bufOffset + i] = 0.f;

            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                const int offset = srcOffset + cnDim * cnStep;
                for (size_t i = 0; i < innerSize; i++)
                    bufPtr[bufOffset + i] += dstPtr[offset + i];
            }

            //divide by computed sum
            for (size_t cnDim = 0; cnDim < channels; cnDim++)
            {
                const int offset = srcOffset + cnDim * cnStep;
                for (size_t i = 0; i < innerSize; i++)
                    dstPtr[offset + i] /= bufPtr[bufOffset + i];
            }
            if (logSoftMax)
            {
                for (size_t cnDim = 0; cnDim < channels; cnDim++)
                {
                    const int offset = srcOffset + cnDim * cnStep;
                    for (size_t i = 0; i < innerSize; i++)
                        dstPtr[offset + i] = log(dstPtr[offset + i]);
                }
            }
        }
    }

    virtual Ptr<BackendNode> initHalide(const std::vector<Ptr<BackendWrapper> > &inputs)
    {
#ifdef HAVE_HALIDE
        Halide::Buffer<float> inputBuffer = halideBuffer(inputs[0]);
        int inW, inH, inC, inN;
        getCanonicalSize(inputBuffer, &inW, &inH, &inC, &inN);

        if (inW != 1 || inH != 1)
            CV_Error(cv::Error::StsNotImplemented,
                     "Halide backend for SoftMax with spatial size "
                     "more than 1x1 is not implemented");

        Halide::Var x("x"), y("y"), c("c"), n("n");
        Halide::Func top = (name.empty() ? Halide::Func() : Halide::Func(name));

        Halide::Func expInput("expInput");
        Halide::RDom r(0, inW, 0, inH, 0, inC);
        expInput(x, y, c, n) = exp(inputBuffer(x, y, c, n));
        Halide::Expr globalSum = sum(expInput(r.x, r.y, r.z, n));
        top(x, y, c, n) = expInput(x, y, c, n) / globalSum;
        return Ptr<BackendNode>(new HalideBackendNode(top));
#endif  // HAVE_HALIDE
        return Ptr<BackendNode>();
    }

    int64 getFLOPS(const std::vector<MatShape> &inputs,
                  const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning
        int64 flops = 0;

        for (int i = 0; i < inputs.size(); i++)
        {
            flops += 4*total(inputs[i]);
        }

        return flops;
    }

    int axisRaw;
};

Ptr<SoftmaxLayer> SoftmaxLayer::create(const LayerParams& params)
{
    return Ptr<SoftmaxLayer>(new SoftMaxLayerImpl(params));
}

}
}
