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
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/reshape.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

static void computeShapeByReshapeMask(const MatShape &srcShape,
                                      const MatShape &maskShape,
                                      Range srcRange /*= Range::all()*/,
                                      MatShape& dstShape)
{
    int srcShapeSize = (int)srcShape.size();
    int maskShapeSize = (int)maskShape.size();

    if (srcRange == Range::all())
        srcRange = Range(0, srcShapeSize);
    else
    {
        int sz = srcRange.size();
        srcRange.start = clamp(srcRange.start, srcShapeSize);
        srcRange.end = srcRange.end == INT_MAX ? srcShapeSize : srcRange.start + sz;
    }

    bool explicitMask = !maskShape.empty();  // All mask values are positive.
    for (int i = 0, n = maskShape.size(); i < n && explicitMask; ++i)
    {
        explicitMask = maskShape[i] > 0;
    }
    // Working range of source shape is a range where area(src) == area(mask).
    if (explicitMask)
    {
        int maskTotal = total(maskShape);
        // Go from the end of mask until we collect required total.
        bool matched = false;
        for (int i = srcRange.end - 1; i >= srcRange.start; --i)
        {
            if (matched)
            {
                if (total(srcShape, i, srcRange.end) != maskTotal)
                {
                    srcRange.start = i + 1;
                    break;
                }
                else if (i == 0)
                {
                    srcRange.start = 0;
                    break;
                }
            }
            else
            {
                matched = total(srcShape, i, srcRange.end) == maskTotal;
            }
        }
        while (total(srcShape, srcRange.start, srcRange.end) != maskTotal && srcRange.start > 0)
        {
            srcRange.start -= 1;
        }
        CV_Assert(total(srcShape, srcRange.start, srcRange.end) == maskTotal);
    }

    CV_Assert(0 <= srcRange.start && srcRange.start <= srcRange.end && srcRange.end <= srcShapeSize);
    int dstShapeSize = srcShapeSize - srcRange.size() + maskShapeSize;
    dstShape.resize(dstShapeSize);

    std::copy(srcShape.begin(), srcShape.begin() + srcRange.start, dstShape.begin());
    std::copy(srcShape.begin() + srcRange.end, srcShape.begin() + srcShapeSize, dstShape.begin() + srcRange.start + maskShapeSize);

    int inferDim = -1;
    for (int i = 0; i < maskShapeSize; i++)
    {
        if (maskShape[i] > 0)
        {
            dstShape[srcRange.start + i] = maskShape[i];
        }
        else if (maskShape[i] == 0)
        {
            if (srcRange.start + i >= srcShapeSize)
                CV_Error(Error::StsBadArg, format("Copy dim[%d] (which has zero size) is out of the source shape bounds", srcRange.start + i));
            dstShape[srcRange.start + i] = srcShape[srcRange.start + i];
        }
        else if (maskShape[i] == -1)
        {
            if (inferDim != -1)
                CV_Error(Error::StsAssert, "Duplicate of inferred dim (which is denoted by -1)");
            inferDim = srcRange.start + i;
            dstShape[inferDim] = 1;
        }
        else
            CV_Error(Error::StsBadArg, "maskShape[i] >= -1");
    }

    size_t srcTotal = total(srcShape);
    size_t dstTotal = total(dstShape);
    CV_Assert(dstTotal != 0);

    if (inferDim != -1)
    {
        if (srcTotal % dstTotal != 0)
            CV_Error(Error::StsBackTrace, "Can't infer a dim denoted by -1");

        dstShape[inferDim] = (int)(srcTotal / dstTotal);
    }
    else
    {
        CV_Assert(srcTotal == dstTotal);
    }
}


class ReshapeLayerImpl CV_FINAL : public ReshapeLayer
{
public:
    ReshapeLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        int axis = params.get<int>("axis", 0);
        int numAxes = params.get<int>("num_axes", -1);
        CV_Assert(numAxes >= -1);
        newShapeRange = (numAxes == -1) ? Range(axis, INT_MAX) : Range(axis, axis + numAxes);

        newShapeDesc.clear();
        if (params.has("dim"))
        {
            const DictValue &paramShape = params.get("dim");
            int i, dims = paramShape.size();
            newShapeDesc.resize(dims);
            for (i = 0; i < dims; i++)
                newShapeDesc[i] = paramShape.get<int>(i);
        }
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               backendId == DNN_BACKEND_CUDA ||
               ((backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH) && haveInfEngine());
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        if (inputs.size() == 1 || inputs.size() == requiredOutputs)
        {
            outputs.clear();
            for (size_t i = 0; i < inputs.size(); i++)
            {
                outputs.push_back(MatShape());
                computeShapeByReshapeMask(inputs[i], newShapeDesc, newShapeRange, outputs.back());
            }
        }
        else
        {
            CV_Assert_N(inputs.size() == 2, total(inputs[0]) == total(inputs[1]));
            outputs.assign(1, inputs[1]);
        }
        return true;
    }

    void finalize(InputArrayOfArrays, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);

        CV_Assert(!outputs.empty());
        outShapes.resize(outputs.size());
        for (int i = 0; i < outputs.size(); ++i)
            outShapes[i] = shape(outputs[i]);
    }

    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        for (size_t i = 0; i < outputs.size(); i++)
        {
            UMat srcBlob = inputs[i];
            void *src_handle = inputs[i].handle(ACCESS_READ);
            void *dst_handle = outputs[i].handle(ACCESS_WRITE);
            if (src_handle != dst_handle)
            {
                UMat umat = srcBlob.reshape(1, (int)outShapes[i].size(), &outShapes[i][0]);
                umat.copyTo(outputs[i]);
            }
        }
        outs.assign(outputs);

        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        for (size_t i = 0; i < outputs.size(); i++)
        {
            Mat srcBlob = inputs[i];
            if (outputs[i].data != srcBlob.data)
                srcBlob.reshape(1, shape(outputs[i])).copyTo(outputs[i]);
        }
    }


#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
        InferenceEngine::Builder::ReshapeLayer ieLayer(name);
        CV_Assert(outShapes.size() == 1);
        ieLayer.setDims(outShapes[0]);
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        CV_Assert(outShapes.size() == 1);
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;

        std::vector<int64_t> out(outShapes[0].begin(), outShapes[0].end());
        auto shape   = std::make_shared<ngraph::op::Constant>(ngraph::element::i64,
                       ngraph::Shape{out.size()}, out.data());
        auto reshape = std::make_shared<ngraph::op::v1::Reshape>(ieInpNode, shape, true);
        return Ptr<BackendNode>(new InfEngineNgraphNode(reshape));
    }
#endif  // HAVE_DNN_NGRAPH


#ifdef HAVE_CUDA
    Ptr<BackendNode> initCUDA(
        void *context_,
        const std::vector<Ptr<BackendWrapper>>& inputs,
        const std::vector<Ptr<BackendWrapper>>& outputs
    ) override
    {
        auto context = reinterpret_cast<csl::CSLContext*>(context_);
        return make_cuda_node<cuda4dnn::ReshapeOp>(preferableTarget, std::move(context->stream));
    }
#endif


private:
    std::vector<MatShape> outShapes;
};

Ptr<ReshapeLayer> ReshapeLayer::create(const LayerParams& params)
{
    return Ptr<ReshapeLayer>(new ReshapeLayerImpl(params));
}


}
}
