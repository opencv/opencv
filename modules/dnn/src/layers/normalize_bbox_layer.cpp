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
#include "../ie_ngraph.hpp"

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
        if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 || backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            if (pnorm != 2)
                return false;

            if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019 && preferableTarget == DNN_TARGET_MYRIAD)
                return !acrossSpatial;

            return startAxis == 1;
        }
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
        startAxis = normalize_axis(startAxis, inp0.dims);
        endAxis = normalize_axis(endAxis, inp0.dims);

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
        startAxis = normalize_axis(startAxis, inp0.dims);
        endAxis = normalize_axis(endAxis, inp0.dims);

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

#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
        InferenceEngine::DataPtr input = infEngineDataNode(inputs[0]);
        std::vector<size_t> dims = input->getDims();
        if (dims.size() == 4)
        {
            InferenceEngine::Builder::NormalizeLayer ieLayer(name);

            ieLayer.setChannelShared(false);
            ieLayer.setAcrossMaps(acrossSpatial);
            ieLayer.setEpsilon(epsilon);

            InferenceEngine::Builder::Layer l = ieLayer;
            const int numChannels = dims[1];
            InferenceEngine::Blob::Ptr weights;
            if (blobs.empty())
            {
                weights = InferenceEngine::make_shared_blob<float>({
                              InferenceEngine::Precision::FP32,
                              {(size_t)numChannels}, InferenceEngine::Layout::C
                          });
                weights->allocate();

                Mat weightsMat = infEngineBlobToMat(weights).reshape(1, numChannels);
                Mat(numChannels, 1, CV_32F, Scalar(1)).copyTo(weightsMat);
                l.getParameters()["channel_shared"] = false;
            }
            else
            {
                CV_Assert(numChannels == blobs[0].total());
                weights = wrapToInfEngineBlob(blobs[0], {(size_t)numChannels}, InferenceEngine::Layout::C);
                l.getParameters()["channel_shared"] = blobs[0].total() == 1;
            }
            addConstantData("weights", weights, l);
            l.getParameters()["across_spatial"] = acrossSpatial;
            return Ptr<BackendNode>(new InfEngineBackendNode(l));
        }
        else
        {
            InferenceEngine::Builder::GRNLayer ieLayer(name);
            ieLayer.setBeta(epsilon);

            InferenceEngine::Builder::Layer l = ieLayer;
            l.getParameters()["bias"] = epsilon;

            return Ptr<BackendNode>(new InfEngineBackendNode(l));
        }
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019

#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        const size_t batch = ieInpNode->get_shape()[0];
        const size_t numChannels = ieInpNode->get_shape()[1];

        std::vector<int64_t> axes_data;
        if (!acrossSpatial) {
            axes_data.push_back(1);
        } else {
            axes_data.resize(ieInpNode->get_shape().size() - 1);
            std::iota(axes_data.begin(), axes_data.end(), 1);
        }
        auto axes = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{axes_data.size()}, axes_data);
        auto norm = std::make_shared<ngraph::op::v0::NormalizeL2>(ieInpNode, axes, epsilon, ngraph::op::EpsMode::ADD);

        CV_Assert(blobs.empty() || numChannels == blobs[0].total());
        std::vector<size_t> shape(ieInpNode->get_shape().size(), 1);
        shape[0] = blobs.empty() ? 1 : batch;
        shape[1] = numChannels;
        if (!blobs.empty())
        {
            auto weight = std::make_shared<ngraph::op::Constant>(
                                      ngraph::element::f32, ngraph::Shape(shape), blobs[0].data);
#if INF_ENGINE_VER_MAJOR_GT(INF_ENGINE_RELEASE_2021_2)
            auto mul = std::make_shared<ngraph::op::v1::Multiply>(norm, weight, ngraph::op::AutoBroadcastType::NUMPY);
#else
            auto mul = std::make_shared<ngraph::op::v0::Multiply>(norm, weight, ngraph::op::AutoBroadcastType::NUMPY);
#endif
            return Ptr<BackendNode>(new InfEngineNgraphNode(mul));
        }
        return Ptr<BackendNode>(new InfEngineNgraphNode(norm));
    }
#endif  // HAVE_DNN_NGRAPH

private:
    int startAxis, endAxis;
};


Ptr<NormalizeBBoxLayer> NormalizeBBoxLayer::create(const LayerParams &params)
{
    return Ptr<NormalizeBBoxLayer>(new NormalizeBBoxLayerImpl(params));
}

}
}
