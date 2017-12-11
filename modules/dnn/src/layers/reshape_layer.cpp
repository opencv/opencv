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
#include <opencv2/dnn/shape_utils.hpp>

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
                if (i == 0 || total(srcShape, i, srcRange.end) != maskTotal)
                {
                    srcRange.start = i + 1;
                    break;
                }
            }
            else
            {
                matched = total(srcShape, i, srcRange.end) == maskTotal;
            }
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


class ReshapeLayerImpl : public ReshapeLayer
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

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        outputs.clear();

        for (size_t i = 0; i < inputs.size(); i++)
        {
            outputs.push_back(MatShape());
            computeShapeByReshapeMask(inputs[i], newShapeDesc, newShapeRange, outputs.back());
        }
        internals = outputs;

        return true;
    }

    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            UMat srcBlob = inputs[i];
            void *src_handle = inputs[i].handle(ACCESS_READ);
            void *dst_handle = outputs[i].handle(ACCESS_WRITE);
            if (src_handle != dst_handle)
            {
                MatShape outShape = shape(outputs[i]);
                UMat umat = srcBlob.reshape(1, (int)outShape.size(), &outShape[0]);
                umat.copyTo(outputs[i]);
            }
        }
        outs.assign(outputs);

        return true;
    }

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN((preferableTarget == DNN_TARGET_OPENCL) &&
                   OCL_PERFORMANCE_CHECK(ocl::Device::getDefault().isIntel()),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        Layer::forward_fallback(inputs_arr, outputs_arr, internals_arr);
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        for (size_t i = 0; i < inputs.size(); i++)
        {
            Mat srcBlob = *inputs[i];
            if (outputs[i].data != srcBlob.data)
                srcBlob.reshape(1, shape(outputs[i])).copyTo(outputs[i]);
        }
    }
};

Ptr<ReshapeLayer> ReshapeLayer::create(const LayerParams& params)
{
    return Ptr<ReshapeLayer>(new ReshapeLayerImpl(params));
}


}
}
