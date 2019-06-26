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
#include "op_cuda.hpp"
#include "../op_inf_engine.hpp"
#include <float.h>
#include <algorithm>
#include <opencv2/dnn/shape_utils.hpp>

#ifdef HAVE_CUDA
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/tensor_ops.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv
{
namespace dnn
{

class FlattenLayerImpl CV_FINAL : public FlattenLayer
{
public:
    FlattenLayerImpl(const LayerParams &params)
    {
        _startAxis = params.get<int>("axis", 1);
        _endAxis = params.get<int>("end_axis", -1);
        setParamsFrom(params);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
               (backendId == DNN_BACKEND_CUDA && haveCUDA()) ||
               (backendId == DNN_BACKEND_INFERENCE_ENGINE && haveInfEngine());
    }

    bool getMemoryShapes(const std::vector<MatShape> &inputs,
                         const int requiredOutputs,
                         std::vector<MatShape> &outputs,
                         std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.size() > 0);
        for (size_t i = 1; i < inputs.size(); i++)
        {
            CV_Assert(inputs[i] == inputs[0]);
        }

        int numAxes = inputs[0].size();
        int startAxis = clamp(_startAxis, numAxes);
        int endAxis = clamp(_endAxis, numAxes);

        CV_Assert(startAxis >= 0);
        CV_Assert(endAxis >= startAxis && endAxis < (int)numAxes);

        size_t flattenedDimensionSize = total(inputs[0], startAxis, endAxis + 1);

        MatShape outputShapeVec;
        for (int i = 0; i < startAxis; i++)
        {
            outputShapeVec.push_back(inputs[0][i]);
        }
        outputShapeVec.push_back(flattenedDimensionSize);
        for (size_t i = endAxis + 1; i < numAxes; i++)
        {
            outputShapeVec.push_back(inputs[0][i]);
        }
        CV_Assert(outputShapeVec.size() <= 4);

        outputs.resize(inputs.size(), outputShapeVec);

        return true;
    }

    void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays) CV_OVERRIDE
    {
        std::vector<Mat> inputs;
        inputs_arr.getMatVector(inputs);

        int numAxes = inputs[0].dims;
        _startAxis = clamp(_startAxis, numAxes);
        _endAxis = clamp(_endAxis, numAxes);
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr)
    {
        std::vector<UMat> inpvec;
        std::vector<UMat> outputs;

        inputs_arr.getUMatVector(inpvec);
        outputs_arr.getUMatVector(outputs);

        std::vector<UMat*> inputs(inpvec.size());
        for (int i = 0; i < inpvec.size(); i++)
            inputs[i] = &inpvec[i];

        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape outShape = shape(outputs[i]);
            UMat& output = outputs_arr.getUMatRef(i);
            output = inputs[i]->reshape(1, (int)outShape.size(), &outShape[0]);
        }

        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget) &&
                   outputs_arr.isUMatVector(),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        for (size_t i = 0; i < inputs.size(); i++)
        {
            MatShape outShape = shape(outputs[i]);
            if (inputs[i].data != outputs[i].data)
            {
                inputs[i].reshape(1, (int)outShape.size(), &outShape[0]).copyTo(outputs[i]);
            }
        }
    }

#ifdef HAVE_CUDA
    void forwardCUDA(
        std::vector<cv::Ptr<BackendWrapper>>& inputs,
        std::vector<cv::Ptr<BackendWrapper>>& outputs,
        csl::Workspace& workspace
    )
    {
        CV_UNUSED(workspace);
        CV_Assert(outputs.size() == 1);

        for (std::size_t i = 0; i < inputs.size(); i++)
        {
            auto input_wrapper = inputs[i].dynamicCast<CUDABackendWrapperFP32>();
            auto input = input_wrapper->getView();

            auto output_wrapper = outputs[i].dynamicCast<CUDABackendWrapperFP32>();
            auto output = output_wrapper->getSpan();

            if (input.get() != output.get())
            {
                input.reshape_as(output);
                csl::tensor_ops::copy(stream, output, input);
            }
        }
    }

    void initCUDA(
        csl::Stream stream_,
        csl::cublas::Handle cublas_handle,
        csl::cudnn::Handle cudnn_handle,
        std::size_t& scratch_mem_in_bytes
    )
    {
        stream = std::move(stream_);
    }

    csl::Stream stream;
#endif

#ifdef HAVE_INF_ENGINE
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >& inputs) CV_OVERRIDE
    {
        InferenceEngine::Builder::Layer ieLayer(name);
        ieLayer.setName(name);
        ieLayer.setType("Flatten");
        ieLayer.getParameters()["axis"] = (size_t)_startAxis;
        ieLayer.getParameters()["end_axis"] = _endAxis;  // Do not cast to size_t because it might be negative.
        ieLayer.setInputPorts(std::vector<InferenceEngine::Port>(1));
        ieLayer.setOutputPorts(std::vector<InferenceEngine::Port>(1));
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_INF_ENGINE

    int _startAxis;
    int _endAxis;
};

Ptr<FlattenLayer> FlattenLayer::create(const LayerParams& params)
{
    return Ptr<FlattenLayer>(new FlattenLayerImpl(params));
}

}
}
