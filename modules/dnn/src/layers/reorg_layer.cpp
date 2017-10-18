/*M ///////////////////////////////////////////////////////////////////////////////////////
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
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <iostream>

namespace cv
{
namespace dnn
{

class ReorgLayerImpl : public ReorgLayer
{
    int reorgStride;
public:

    ReorgLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);

        reorgStride = params.get<int>("reorg_stride", 2);
        CV_Assert(reorgStride > 0);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() > 0);
        outputs = std::vector<MatShape>(inputs.size(), shape(
            inputs[0][0],
            inputs[0][1] * reorgStride * reorgStride,
            inputs[0][2] / reorgStride,
            inputs[0][3] / reorgStride));

        CV_Assert(outputs[0][0] > 0 && outputs[0][1] > 0 && outputs[0][2] > 0 && outputs[0][3] > 0);
        CV_Assert(total(outputs[0]) == total(inputs[0]));

        return false;
    }

    virtual bool supportBackend(int backendId)
    {
        return backendId == DNN_BACKEND_DEFAULT;
    }
    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        for (size_t i = 0; i < inputs.size(); i++)
        {
            Mat srcBlob = *inputs[i];
            MatShape inputShape = shape(srcBlob), outShape = shape(outputs[i]);
            float *dstData = outputs[0].ptr<float>();
            const float *srcData = srcBlob.ptr<float>();

            int channels = inputShape[1], height = inputShape[2], width = inputShape[3];

            int out_c = channels / (reorgStride*reorgStride);

            for (int k = 0; k < channels; ++k) {
                for (int j = 0; j < height; ++j) {
                    for (int i = 0; i < width; ++i) {
                        int out_index = i + width*(j + height*k);
                        int c2 = k % out_c;
                        int offset = k / out_c;
                        int w2 = i*reorgStride + offset % reorgStride;
                        int h2 = j*reorgStride + offset / reorgStride;
                        int in_index = w2 + width*reorgStride*(h2 + height*reorgStride*c2);
                        dstData[out_index] = srcData[in_index];
                    }
                }
            }
        }
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const
    {
        (void)outputs; // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 21*total(inputs[i]);
        }
        return flops;
    }
};

Ptr<ReorgLayer> ReorgLayer::create(const LayerParams& params)
{
    return Ptr<ReorgLayer>(new ReorgLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
