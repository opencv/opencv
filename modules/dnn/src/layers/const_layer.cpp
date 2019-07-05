// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../op_inf_engine.hpp"
#include "../op_cuda.hpp"
#include "layers_common.hpp"

#ifdef HAVE_OPENCL
#include "opencl_kernels_dnn.hpp"
#endif

#ifdef HAVE_CUDA
#include "../cuda4dnn/csl/tensor.hpp"
#include "../cuda4dnn/csl/tensor_ops.hpp"
using namespace cv::dnn::cuda4dnn;
#endif

namespace cv { namespace dnn {

class ConstLayerImpl CV_FINAL : public ConstLayer
{
public:
    ConstLayerImpl(const LayerParams& params)
    {
        setParamsFrom(params);
        CV_Assert(blobs.size() == 1);
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV || backendId == DNN_BACKEND_INFERENCE_ENGINE;
    }

    virtual bool getMemoryShapes(const std::vector<MatShape> &inputs,
                                 const int requiredOutputs,
                                 std::vector<MatShape> &outputs,
                                 std::vector<MatShape> &internals) const CV_OVERRIDE
    {
        CV_Assert(inputs.empty());
        outputs.assign(1, shape(blobs[0]));
        return false;
    }

    virtual bool supportBackend(int backendId) CV_OVERRIDE
    {
        return backendId == DNN_BACKEND_OPENCV ||
              (backendId == DNN_BACKEND_CUDA && haveCUDA());
    }

#ifdef HAVE_OPENCL
    bool forward_ocl(InputArrayOfArrays inps, OutputArrayOfArrays outs, OutputArrayOfArrays internals)
    {
        std::vector<UMat> outputs;
        outs.getUMatVector(outputs);
        if (outs.depth() == CV_16S)
            convertFp16(blobs[0], outputs[0]);
        else
            blobs[0].copyTo(outputs[0]);
        return true;
    }
#endif

    void forward(InputArrayOfArrays inputs_arr, OutputArrayOfArrays outputs_arr, OutputArrayOfArrays internals_arr) CV_OVERRIDE
    {
        CV_TRACE_FUNCTION();
        CV_TRACE_ARG_VALUE(name, "name", name.c_str());

        CV_OCL_RUN(IS_DNN_OPENCL_TARGET(preferableTarget),
                   forward_ocl(inputs_arr, outputs_arr, internals_arr))

        std::vector<Mat> outputs;
        outputs_arr.getMatVector(outputs);
        blobs[0].copyTo(outputs[0]);
    }

#ifdef HAVE_INF_ENGINE
    virtual Ptr<BackendNode> initInfEngine(const std::vector<Ptr<BackendWrapper> >&) CV_OVERRIDE
    {
        InferenceEngine::Builder::ConstLayer ieLayer(name);
        ieLayer.setData(wrapToInfEngineBlob(blobs[0]));
        return Ptr<BackendNode>(new InfEngineBackendNode(ieLayer));
    }
#endif  // HAVE_INF_ENGINE

#ifdef HAVE_CUDA
    void forwardCUDA(
        std::vector<cv::Ptr<BackendWrapper>>& inputs,
        std::vector<cv::Ptr<BackendWrapper>>& outputs,
        csl::Workspace& workspace
    ) override
    {
        auto output_wrapper = outputs[0].dynamicCast<CUDABackendWrapperFP32>();
        csl::tensor_ops::copy<float>(stream, output_wrapper->getSpan(), constTensor);
    }

    void initCUDA(
        csl::Stream stream_,
        csl::cublas::Handle cublas_handle,
        csl::cudnn::Handle cudnn_handle,
        std::size_t& scratch_mem_in_bytes,
        const std::vector<cv::Ptr<BackendWrapper>>& inputs
    ) override
    {
        /* host to device copy is more expensive than device to device copy; hence, we keep a copy
         * of the blob in device memory and use it as the source for copy
         */
        stream = std::move(stream_);
        constTensor = createTensorHeaderFromMat(blobs[0]);
        copyMatToTensor<float>(constTensor, blobs[0], stream);
    }

    csl::Stream stream;
    csl::Tensor<float> constTensor;
#endif

};

Ptr<Layer> ConstLayer::create(const LayerParams& params)
{
    return Ptr<Layer>(new ConstLayerImpl(params));
}

}}  // namespace cv::dnn
