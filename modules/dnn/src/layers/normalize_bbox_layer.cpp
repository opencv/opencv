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

namespace cv { namespace dnn {

class NormalizeBBoxLayerImpl : public NormalizeBBoxLayer
{
public:
    NormalizeBBoxLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        pnorm = params.get<float>("p", 2);
        epsilon = params.get<float>("eps", 1e-10f);
        acrossSpatial = params.get<bool>("across_spatial", true);
        CV_Assert(pnorm > 0);
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const
    {
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        internals.resize(1, inputs[0]);
        internals[0][0] = 1;  // Batch size.
        return true;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        internals_.getUMatVector(internals);

        CV_Assert(inputs.size() == 1 && outputs.size() == 1);
        CV_Assert(inputs[0].total() == outputs[0].total());

        const UMat& inp0 = inputs[0];
        UMat& buffer = internals[0];
        size_t num = inp0.size[0];
        size_t channels = inp0.size[1];
        size_t channelSize = inp0.total() / (num * channels);
        for (size_t i = 0; i < num; ++i)
        {
            MatShape s = shape(channels, channelSize);
            UMat src = inputs[i].reshape(1, s.size(), &s[0]);
            UMat dst = outputs[i].reshape(1, s.size(), &s[0]);

            UMat abs_mat;
            absdiff(src, cv::Scalar::all(0), abs_mat);
            pow(abs_mat, pnorm, buffer);

            if (acrossSpatial)
            {
                // add eps to avoid overflow
                float absSum = sum(buffer)[0] + epsilon;
                float norm = pow(absSum, 1.0f / pnorm);
                multiply(src, 1.0f / norm, dst);
            }
            else
            {
                Mat norm;
                reduce(buffer, norm, 0, REDUCE_SUM);
                norm += epsilon;

                // compute inverted norm to call multiply instead divide
                cv::pow(norm, -1.0f / pnorm, norm);

                repeat(norm, channels, 1, buffer);
                multiply(src, buffer, dst);
            }

            if (!blobs.empty())
            {
                // scale the output
                Mat scale = blobs[0];
                if (scale.total() == 1)
                {
                    // _scale: 1 x 1
                    multiply(dst, scale.at<float>(0, 0), dst);
                }
                else
                {
                    // _scale: _channels x 1
                    CV_Assert(scale.total() == channels);
                    repeat(scale, 1, dst.cols, buffer);
                    multiply(dst, buffer, dst);
                }
            }
        }
        return true;
    }
#endif

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

        CV_Assert(inputs.size() == 1 && outputs.size() == 1);
        CV_Assert(inputs[0]->total() == outputs[0].total());

        const Mat& inp0 = *inputs[0];
        Mat& buffer = internals[0];
        size_t num = inp0.size[0];
        size_t channels = inp0.size[1];
        size_t channelSize = inp0.total() / (num * channels);
        for (size_t n = 0; n < num; ++n)
        {
            Mat src = Mat(channels, channelSize, CV_32F, (void*)inp0.ptr<float>(n));
            Mat dst = Mat(channels, channelSize, CV_32F, (void*)outputs[0].ptr<float>(n));

            cv::pow(abs(src), pnorm, buffer);

            if (acrossSpatial)
            {
                // add eps to avoid overflow
                float absSum = sum(buffer)[0] + epsilon;
                float norm = pow(absSum, 1.0f / pnorm);
                multiply(src, 1.0f / norm, dst);
            }
            else
            {
                Mat norm;
                reduce(buffer, norm, 0, REDUCE_SUM);
                norm += epsilon;

                // compute inverted norm to call multiply instead divide
                cv::pow(norm, -1.0f / pnorm, norm);

                repeat(norm, channels, 1, buffer);
                multiply(src, buffer, dst);
            }

            if (!blobs.empty())
            {
                // scale the output
                Mat scale = blobs[0];
                if (scale.total() == 1)
                {
                    // _scale: 1 x 1
                    dst *= scale.at<float>(0, 0);
                }
                else
                {
                    // _scale: _channels x 1
                    CV_Assert(scale.total() == channels);
                    repeat(scale, 1, dst.cols, buffer);
                    multiply(dst, buffer, dst);
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
