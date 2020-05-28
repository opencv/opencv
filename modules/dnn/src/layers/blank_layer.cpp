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
#include "../op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include "../ie_ngraph.hpp"

#ifdef HAVE_CUDA
#include "../cuda4dnn/primitives/reshape.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{
class BlankLayerImpl CV_FINAL : public BlankLayer
{
public:
    BlankLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
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
        Layer::getMemoryShapes(inputs, requiredOutputs, outputs, internals);
        return true;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_, OutputArrayOfArrays outputs_, OutputArrayOfArrays internals_)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inputs_.getUMatVector(inputs);
        outputs_.getUMatVector(outputs);

        for (int i = 0, n = outputs.size(); i < n; ++i)
        {
            void *src_handle = inputs[i].handle(ACCESS_READ);
            void *dst_handle = outputs[i].handle(ACCESS_WRITE);
            if (src_handle != dst_handle)
                inputs[i].copyTo(outputs[i]);
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

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        for (int i = 0, n = outputs.size(); i < n; ++i)
            if (outputs[i].data != inputs[i].data)
                inputs[i].copyTo(outputs[i]);
    }


#ifdef HAVE_DNN_IE_NN_BUILDER_2019
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
        InferenceEngine::DataPtr input = infEngineDataNode(inputs[0]);
        std::vector<size_t> dims = input->getDims();
        CV_Assert(!dims.empty());

        InferenceEngine::Builder::Layer ieLayer(name);
        ieLayer.setName(name);
        if (preferableTarget == DNN_TARGET_MYRIAD)
        {
            ieLayer.setType("Copy");
        }
        else
        {
            ieLayer.setType("Split");
            ieLayer.getParameters()["axis"] = dims.size() - 1;
            ieLayer.getParameters()["out_sizes"] = dims[0];
        }
        ieLayer.setInputPorts({InferenceEngine::Port(dims)});
        ieLayer.setOutputPorts(std::vector<InferenceEngine::Port>(1));
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_DNN_IE_NN_BUILDER_2019


#ifdef HAVE_DNN_NGRAPH
    virtual Ptr<BackendNode> initNgraph(const std::vector<Ptr<BackendWrapper> >& inputs,
                                        const std::vector<Ptr<BackendNode> >& nodes) CV_OVERRIDE
    {
        auto& ieInpNode = nodes[0].dynamicCast<InfEngineNgraphNode>()->node;
        ngraph::OutputVector inp{ieInpNode};
        auto blank = std::make_shared<ngraph::op::Concat>(inp, 0);
        return Ptr<BackendNode>(new InfEngineNgraphNode(blank));
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

};

Ptr<Layer> BlankLayer::create(const LayerParams& params)
{
    // In case of Caffe's Dropout layer from Faster-RCNN framework,
    // https://github.com/rbgirshick/caffe-fast-rcnn/tree/faster-rcnn
    // return Power layer.
    if (!params.get<bool>("scale_train", true))
    {
        float scale = 1 - params.get<float>("dropout_ratio", 0.5f);
        CV_Assert(scale > 0);

        LayerParams powerParams;
        powerParams.name = params.name;
        powerParams.type = "Power";
        powerParams.set("scale", scale);

        return PowerLayer::create(powerParams);
    }
    else
        return Ptr<BlankLayer>(new BlankLayerImpl(params));
}

}
}
