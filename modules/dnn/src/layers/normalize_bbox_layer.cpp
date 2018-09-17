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
#include "../op_inf_engine.hpp"

namespace cv { namespace dnn {

class NormalizeBBoxLayerImpl CV_FINAL : public NormalizeBBoxLayer
{
public:
    NormalizeBBoxLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        pnorm = params.get<float>("p", 2);
        epsilon = params.get<float>("eps", 1e-10f);
        acrossSpatial = params.get<bool>("across_spatial", true);
        startAxis = params.get<int>("start_axis", 1);
        CV_Assert(!params.has("across_spatial") || !params.has("end_axis"));
        endAxis = params.get<int>("end_axis", acrossSpatial ? -1 : startAxis);
        CV_Assert(pnorm > 0);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
        {
            if (pnorm != 2)
                return false;
            if (!blobs.empty())
                return true;
            if (preferableTarget == DNN_TARGET_MYRIAD)
                return !acrossSpatial;
            return startAxis == 1 && (!acrossSpatial || endAxis > 1);
        }
        else
            return backendId == DNN_BACKEND_OPENCV;
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() == 1);
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        internals.resize(1, inputs[0]);
        internals[0][0] = 1;  // Batch size.
        return true;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);
        CV_Assert(inputs.size() == 1);
        endAxis = endAxis == -1 ? (inputs[0].dims - 1) : endAxis;
        startAxis = startAxis == -1 ? (inputs[0].dims - 1) : startAxis;
        acrossSpatial = (startAxis == 1 && endAxis == inputs[0].dims - 1);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;
        std::vector<UMat> internals;

        if (inputs_.depth() == CV_16S)
            return false;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);
        internals_.getUMatVector(internals);

        CV_Assert(inputs.size() == 1 && outputs.size() == 1);
        CV_Assert(inputs[0].total() == outputs[0].total());

        const UMat& inp0 = inputs[0];
        UMat& buffer = internals[0];
        startAxis = clamp(startAxis, inp0.dims);
        endAxis = clamp(endAxis, inp0.dims);

        size_t num = total(shape(inp0.size), 0, startAxis);
        size_t numPlanes = total(shape(inp0.size), startAxis, endAxis + 1);
        size_t planeSize = inp0.total() / (num * numPlanes);
        MatShape s = shape(1, inputs[0].total());
        UMat inp = inputs[0].reshape(1, s.size(), &s[0]).reshape(1, num);
        UMat out = outputs[0].reshape(1, s.size(), &s[0]).reshape(1, num);
        for (size_t i = 0; i < num; ++i)
        {
            s = shape(numPlanes, planeSize);
            UMat src = inp.row(i).reshape(1, s.size(), &s[0]);
            UMat dst = out.row(i).reshape(1, s.size(), &s[0]);

            UMat abs_mat;
            absdiff(src, cv::Scalar::all(0), abs_mat);
            pow(abs_mat, pnorm, buffer);

            if (planeSize == 1)
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

                repeat(norm, numPlanes, 1, buffer);
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
                    CV_Assert(scale.total() == numPlanes);
                    repeat(scale, 1, dst.cols, buffer);
                    multiply(dst, buffer, dst);
                }
            }
        }
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        if (inputs_arr.depth() == CV_16S)
        {
            forward_fallback(inputs_arr, outputs_arr, internals_arr);
            return;
        }

        std::vector<Mat> inputs, outputs, internals;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);
        internals_arr.getMatVector(internals);

        CV_Assert(inputs.size() == 1 && outputs.size() == 1);
        CV_Assert(inputs[0].total() == outputs[0].total());

        const Mat& inp0 = inputs[0];
        Mat& buffer = internals[0];
        startAxis = clamp(startAxis, inp0.dims);
        endAxis = clamp(endAxis, inp0.dims);

        const float* inpData = inp0.ptr<float>();
        float* outData = outputs[0].ptr<float>();

        size_t num = total(shape(inp0.size), 0, startAxis);
        size_t numPlanes = total(shape(inp0.size), startAxis, endAxis + 1);
        CV_Assert(num * numPlanes != 0);
        size_t planeSize = inp0.total() / (num * numPlanes);
        for (size_t n = 0; n < num; ++n)
        {
            Mat src = Mat(numPlanes, planeSize, CV_32F, (void*)inpData);
            Mat dst = Mat(numPlanes, planeSize, CV_32F, (void*)outData);
            cv::pow(abs(src), pnorm, buffer);

            if (planeSize == 1)
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

                repeat(norm, numPlanes, 1, buffer);
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
                    CV_Assert(scale.total() == numPlanes);
                    repeat(scale, 1, dst.cols, buffer);
                    multiply(dst, buffer, dst);
                }
            }
            inpData += numPlanes * planeSize;
            outData += numPlanes * planeSize;
        }
    }

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::DataPtr input = infEngineDataNode(inputs[0]);

        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.precision = InferenceEngine::Precision::FP32;

        if (input->dims.size() == 4)
        {
            const int numChannels = input->dims[2];  // NOTE: input->dims are reversed (whcn)

            lp.type = "Normalize";
            std::shared_ptr<InferenceEngine::CNNLayer> ieLayer(new InferenceEngine::CNNLayer(lp));
            if (blobs.empty())
            {
                auto weights = InferenceEngine::make_shared_blob<float>(InferenceEngine::Precision::FP32,
                                                                        InferenceEngine::Layout::C,
                                                                        {(size_t)numChannels});
                weights->allocate();
                std::vector<float> ones(numChannels, 1);
                weights->set(ones);
                ieLayer->blobs["weights"] = weights;
                ieLayer->params["channel_shared"] = "0";
            }
            else
            {
                CV_Assert(numChannels == blobs[0].total());
                ieLayer->blobs["weights"] = wrapToInfEngineBlob(blobs[0], {(size_t)numChannels}, InferenceEngine::Layout::C);
                ieLayer->params["channel_shared"] = blobs[0].total() == 1 ? "1" : "0";
            }
            ieLayer->params["eps"] = format("%f", epsilon);
            ieLayer->params["across_spatial"] = acrossSpatial ? "1" : "0";
            return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
        }
        else
        {
            InferenceEngine::LayerParams lp;
            lp.name = name;
            lp.type = "GRN";
            lp.precision = InferenceEngine::Precision::FP32;
            std::shared_ptr<InferenceEngine::CNNLayer> ieLayer(new InferenceEngine::CNNLayer(lp));
            ieLayer->params["bias"] = format("%f", epsilon);
            return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
        }
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

private:
    int startAxis, endAxis;
};


Ptr<NormalizeBBoxLayer> NormalizeBBoxLayer::create(const LayerParams &params)
{
    return Ptr<NormalizeBBoxLayer>(new NormalizeBBoxLayerImpl(params));
}

}
}
