// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"
#include "legacy_backend.hpp"

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN


Ptr<BackendWrapper> Net::Impl::wrap(Mat& host)
{
    if (preferableBackend == DNN_BACKEND_OPENCV && preferableTarget == DNN_TARGET_CPU)
        return Ptr<BackendWrapper>();

    MatShape shape(host.dims);
    for (int i = 0; i < host.dims; ++i)
        shape[i] = host.size[i];

    void* data = host.data;
    if (backendWrappers.find(data) != backendWrappers.end())
    {
        Ptr<BackendWrapper> baseBuffer = backendWrappers[data];
        if (preferableBackend == DNN_BACKEND_OPENCV)
        {
#ifdef HAVE_OPENCL
            CV_Assert(IS_DNN_OPENCL_TARGET(preferableTarget));
            return OpenCLBackendWrapper::create(baseBuffer, host);
#else
            CV_Error(Error::StsInternal, "");
#endif
        }
        else if (preferableBackend == DNN_BACKEND_HALIDE)
        {
            CV_Assert(haveHalide());
#ifdef HAVE_HALIDE
            return Ptr<BackendWrapper>(new HalideBackendWrapper(baseBuffer, shape));
#endif
        }
        else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019)
        {
            CV_ERROR_DNN_BACKEND_INFERENCE_ENGINE_NN_BUILDER_2019;
        }
        else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
        {
            return wrapMat(preferableBackend, preferableTarget, host);
        }
        else if (preferableBackend == DNN_BACKEND_WEBNN)
        {
#ifdef HAVE_WEBNN
            return wrapMat(preferableBackend, preferableTarget, host);
#endif
        }
        else if (preferableBackend == DNN_BACKEND_VKCOM)
        {
#ifdef HAVE_VULKAN
            return Ptr<BackendWrapper>(new VkComBackendWrapper(baseBuffer, host));
#endif
        }
        else if (preferableBackend == DNN_BACKEND_CUDA)
        {
            CV_Assert(haveCUDA());
#ifdef HAVE_CUDA
            switch (preferableTarget)
            {
            case DNN_TARGET_CUDA:
                return CUDABackendWrapperFP32::create(baseBuffer, shape);
            case DNN_TARGET_CUDA_FP16:
                return CUDABackendWrapperFP16::create(baseBuffer, shape);
            default:
                CV_Assert(IS_DNN_CUDA_TARGET(preferableTarget));
            }
#endif
        }
        else if (preferableBackend == DNN_BACKEND_TIMVX)
        {
#ifdef HAVE_TIMVX
            return Ptr<BackendWrapper>(new TimVXBackendWrapper(baseBuffer, host));
#endif
        }
        else
            CV_Error(Error::StsNotImplemented, "Unknown backend identifier");
    }

    Ptr<BackendWrapper> wrapper = wrapMat(preferableBackend, preferableTarget, host);
    backendWrappers[data] = wrapper;
    return wrapper;
}


void Net::Impl::initBackend(const std::vector<LayerPin>& blobsToKeep_)
{
    CV_TRACE_FUNCTION();
    if (preferableBackend == DNN_BACKEND_OPENCV)
    {
        CV_Assert(preferableTarget == DNN_TARGET_CPU || IS_DNN_OPENCL_TARGET(preferableTarget));
    }
    else if (preferableBackend == DNN_BACKEND_HALIDE)
    {
#ifdef HAVE_HALIDE
        initHalideBackend();
#else
        CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of Halide");
#endif
    }
    else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
#ifdef HAVE_DNN_NGRAPH
        initNgraphBackend(blobsToKeep_);
#else
        CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of OpenVINO");
#endif
    }
    else if (preferableBackend == DNN_BACKEND_WEBNN)
    {
#ifdef HAVE_WEBNN
        initWebnnBackend(blobsToKeep_);
#else
        CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of WebNN");
#endif
    }
    else if (preferableBackend == DNN_BACKEND_VKCOM)
    {
#ifdef HAVE_VULKAN
        initVkComBackend();
#else
        CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of Vulkan");
#endif
    }
    else if (preferableBackend == DNN_BACKEND_CUDA)
    {
#ifdef HAVE_CUDA
        initCUDABackend(blobsToKeep_);
#else
        CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of CUDA/CUDNN");
#endif
    }
    else if (preferableBackend == DNN_BACKEND_TIMVX)
    {
#ifdef HAVE_TIMVX
        initTimVXBackend();
#else
        CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of TimVX");
#endif
    }
    else
    {
        CV_Error(Error::StsNotImplemented, cv::format("Unknown backend identifier: %d", preferableBackend));
    }
}


void Net::Impl::setPreferableBackend(int backendId)
{
    if (backendId == DNN_BACKEND_DEFAULT)
        backendId = (Backend)getParam_DNN_BACKEND_DEFAULT();

    if (netWasQuantized && backendId != DNN_BACKEND_OPENCV && backendId != DNN_BACKEND_TIMVX)
    {
        CV_LOG_WARNING(NULL, "DNN: Only default and TIMVX backends support quantized networks");
        backendId = DNN_BACKEND_OPENCV;
    }

#ifdef HAVE_INF_ENGINE
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
        backendId = DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;
#endif

    if (preferableBackend != backendId)
    {
        preferableBackend = backendId;
        clear();
    }
}

void Net::Impl::setPreferableTarget(int targetId)
{
    if (netWasQuantized && targetId != DNN_TARGET_CPU &&
        targetId != DNN_TARGET_OPENCL && targetId != DNN_TARGET_OPENCL_FP16 && targetId != DNN_TARGET_NPU)
    {
        CV_LOG_WARNING(NULL, "DNN: Only CPU, OpenCL/OpenCL FP16 and NPU targets are supported by quantized networks");
        targetId = DNN_TARGET_CPU;
    }

    if (preferableTarget != targetId)
    {
        preferableTarget = targetId;
        if (IS_DNN_OPENCL_TARGET(targetId))
        {
#ifndef HAVE_OPENCL
#ifdef HAVE_INF_ENGINE
            if (preferableBackend == DNN_BACKEND_OPENCV)
#else
            if (preferableBackend == DNN_BACKEND_DEFAULT ||
                preferableBackend == DNN_BACKEND_OPENCV)
#endif  // HAVE_INF_ENGINE
                preferableTarget = DNN_TARGET_CPU;
#else
            bool fp16 = ocl::Device::getDefault().isExtensionSupported("cl_khr_fp16");
            if (!fp16 && targetId == DNN_TARGET_OPENCL_FP16)
                preferableTarget = DNN_TARGET_OPENCL;
#endif
        }
        clear();
    }
}


CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
