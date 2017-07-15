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

#include <float.h>
#include <algorithm>

namespace cv
{
namespace dnn
{

namespace
{
    const std::string layerName = "NormalizeBBox";
}

class NormalizeBBoxLayerImpl : public NormalizeBBoxLayer
{
    float _eps;
    bool _across_spatial;
    bool _channel_shared;
public:
    bool getParameterDict(const LayerParams &params,
                          const std::string &parameterName,
                          DictValue& result)
    {
        if (!params.has(parameterName))
        {
            return false;
        }

        result = params.get(parameterName);
        return true;
    }

    template<typename T>
    T getParameter(const LayerParams &params,
                   const std::string &parameterName,
                   const size_t &idx=0,
                   const bool required=true,
                   const T& defaultValue=T())
    {
        DictValue dictValue;
        bool success = getParameterDict(params, parameterName, dictValue);
        if(!success)
        {
            if(required)
            {
                std::string message = layerName;
                message += " layer parameter does not contain ";
                message += parameterName;
                message += " parameter.";
                CV_Error(Error::StsBadArg, message);
            }
            else
            {
                return defaultValue;
            }
        }
        return dictValue.get<T>(idx);
    }

    NormalizeBBoxLayerImpl(const LayerParams &params)
    {
        _eps = getParameter<float>(params, "eps", 0, false, 1e-10f);
        _across_spatial = getParameter<bool>(params, "across_spatial");
        _channel_shared = getParameter<bool>(params, "channel_shared");
        setParamsFrom(params);
    }

    void checkInputs(const std::vector<Mat*> &inputs)
    {
        CV_Assert(inputs.size() > 0);
        CV_Assert(inputs[0]->dims == 4 && inputs[0]->type() == CV_32F);
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i]->dims == 4 && inputs[i]->type() == CV_32F);
            CV_Assert(inputs[i]->size == inputs[0]->size);
        }
        CV_Assert(inputs[0]->dims > 2);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        bool inplace = Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        size_t channels = inputs[0][1];
        size_t rows = inputs[0][2];
        size_t cols = inputs[0][3];
        size_t channelSize = rows * cols;

        internals.assign(1, shape(channels, channelSize));
        internals.push_back(shape(channels, 1));
        internals.push_back(shape(1, channelSize));

        return inplace;
    }

    void forward(std::vector<Mat*> &inputs, std::vector<Mat> &outputs, std::vector<Mat> &internals)
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        checkInputs(inputs);

        Mat& buffer = internals[0], sumChannelMultiplier = internals[1],
                sumSpatialMultiplier = internals[2];

        sumChannelMultiplier.setTo(1.0);
        sumSpatialMultiplier.setTo(1.0);

        const Mat& inp0 = *inputs[0];
        size_t num = inp0.size[0];
        size_t channels = inp0.size[1];
        size_t channelSize = inp0.size[2] * inp0.size[3];

        Mat zeroBuffer(channels, channelSize, CV_32F, Scalar(0));
        Mat absDiff;
        Mat scale = blobs[0];
        for (size_t j = 0; j < inputs.size(); j++)
        {
            for (size_t n = 0; n < num; ++n)
            {
                Mat src = Mat(channels, channelSize, CV_32F, inputs[j]->ptr<float>(n));
                Mat dst = Mat(channels, channelSize, CV_32F, outputs[j].ptr<float>(n));

                buffer = src.mul(src);

                if (_across_spatial)
                {
                    absdiff(buffer, zeroBuffer, absDiff);

                    // add eps to avoid overflow
                    double absSum = sum(absDiff)[0] + _eps;

                    float norm = sqrt(absSum);
                    dst = src / norm;
                }
                else
                {
                    Mat norm(channelSize, 1, buffer.type()); // 1 x channelSize

                    // (_channels x channelSize)T * _channels x 1 -> channelSize x 1
                    gemm(buffer, sumChannelMultiplier, 1, norm, 0, norm, GEMM_1_T);

                    // compute norm
                    pow(norm, 0.5f, norm);

                    // scale the layer
                    // _channels x 1 * (channelSize x 1)T -> _channels x channelSize
                    gemm(sumChannelMultiplier, norm, 1, buffer, 0, buffer, GEMM_2_T);

                    dst = src / buffer;
                }

                // scale the output
                if (_channel_shared)
                {
                    // _scale: 1 x 1
                    dst *= scale.at<float>(0, 0);
                }
                else
                {
                    // _scale: _channels x 1
                    // _channels x 1 * 1 x channelSize -> _channels x channelSize
                    gemm(scale, sumSpatialMultiplier, 1, buffer, 0, buffer);

                    dst = dst.mul(buffer);
                }
            }
        }
    }

};


Ptr<NormalizeBBoxLayer> NormalizeBBoxLayer::create(const LayerParams &params)
{
    return Ptr<NormalizeBBoxLayer>(new NormalizeBBoxLayerImpl(params));
}

}
}
