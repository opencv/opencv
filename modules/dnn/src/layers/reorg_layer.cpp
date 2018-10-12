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
#include "../op_inf_engine.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/all_layers.hpp>

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

namespace cv
{
namespace dnn
{

class ReorgLayerImpl CV_FINAL : public ReorgLayer
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
                         std::vector<MatShape> &internals) const CV_OVERRIDE
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

    virtual void finalize(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr) CV_OVERRIDE
    {
        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        Mat inp = inputs[0];
        Mat out = outputs[0];
        int batchSize = inp.size[0];

        LayerParams permParams;
        if (batchSize == 1)
        {
            int order[] = {1, 3, 0, 2};
            permParams.set("order", DictValue::arrayInt(&order[0], 4));

            permuteInpShape.resize(4);
            permuteInpShape[0] = inp.size[1] * inp.size[2] / (reorgStride * reorgStride);  // (channels*height)/(r*r)
            permuteInpShape[1] = reorgStride;
            permuteInpShape[2] = inp.size[3];  // width
            permuteInpShape[3] = reorgStride;

            permuteOutShape.resize(4);
            for (int i = 0; i < 4; ++i)
                permuteOutShape[i] = permuteInpShape[order[i]];
        }
        else
        {
            int order[] = {0, 2, 4, 1, 3};
            permParams.set("order", DictValue::arrayInt(&order[0], 5));

            permuteInpShape.resize(5);
            permuteInpShape[0] = batchSize;
            permuteInpShape[1] = inp.size[1] * inp.size[2] / (reorgStride * reorgStride);  // (channels*height)/(r*r)
            permuteInpShape[2] = reorgStride;
            permuteInpShape[3] = inp.size[3];  // width
            permuteInpShape[4] = reorgStride;

            permuteOutShape.resize(5);
            for (int i = 0; i < 5; ++i)
                permuteOutShape[i] = permuteInpShape[order[i]];
        }
        permute = PermuteLayer::create(permParams);
        std::vector<Mat> permuteInputs(1, inp.reshape(1, permuteInpShape));
        std::vector<Mat> permuteOutputs(1, out.reshape(1, permuteOutShape));
        permute->finalize(permuteInputs, permuteOutputs);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_INFERENCE_ENGINE;
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> inputs;
        std::vector<UMat> outputs;

        inps.getUMatVector(inputs);
        outs.getUMatVector(outputs);

        inputs[0] = inputs[0].reshape(1, permuteInpShape.size(), &permuteInpShape[0]);
        outputs[0] = outputs[0].reshape(1, permuteOutShape.size(), &permuteOutShape[0]);
        permute->preferableTarget = preferableTarget;
        permute->forward(inputs, outputs, internals);
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

        std::vector<Mat> inputs, outputs;
        inputs_arr.getMatVector(inputs);
        outputs_arr.getMatVector(outputs);

        inputs[0] = inputs[0].reshape(1, permuteInpShape);
        outputs[0] = outputs[0].reshape(1, permuteOutShape);
        permute->forward(inputs, outputs, internals_arr);
    }

    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
#ifdef HAVE_INF_ENGINE
        InferenceEngine::LayerParams lp;
        lp.name = name;
        lp.type = "ReorgYolo";
        lp.precision = InferenceEngine::Precision::FP32;
        std::shared_ptr<InferenceEngine::CNNLayer> ieLayer(new InferenceEngine::CNNLayer(lp));
        ieLayer->params["stride"] = format("%d", reorgStride);
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
#endif  // HAVE_INF_ENGINE
        return Ptr<BackendNode>();
    }

    virtual int64 getFLOPS(const std::vector<MatShape> &inputs,
                           const std::vector<MatShape> &outputs) const CV_OVERRIDE
    {
        CV_UNUSED(outputs); // suppress unused variable warning

        int64 flops = 0;
        for(int i = 0; i < inputs.size(); i++)
        {
            flops += 21*total(inputs[i]);
        }
        return flops;
    }

private:
    Ptr<PermuteLayer> permute;
    std::vector<int> permuteInpShape, permuteOutShape;
};

Ptr<ReorgLayer> ReorgLayer::create(const LayerParams& params)
{
    return Ptr<ReorgLayer>(new ReorgLayerImpl(params));
}

}  // namespace dnn
}  // namespace cv
