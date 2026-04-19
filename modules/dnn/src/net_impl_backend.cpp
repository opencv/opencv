// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "net_impl.hpp"
#include "legacy_backend.hpp"

#include "backend.hpp"
#include "factory.hpp"

#ifdef HAVE_CUDA
#include "cuda4dnn/init.hpp"
#endif

#ifdef HAVE_ONNXRUNTIME
#include <onnxruntime_cxx_api.h>
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_ONNXRUNTIME
void Net::Impl::refreshOrtMainGraphOutputs()
{
    CV_Assert(mainGraph && ort_session);

    Ort::Session& session = *ort_session;
    Ort::AllocatorWithDefaultOptions allocator;
    const size_t noutputs = session.GetOutputCount();
    CV_Assert(noutputs > 0);

    std::vector<Arg> outs;
    outs.reserve(noutputs);
    for (size_t i = 0; i < noutputs; ++i)
    {
        Ort::AllocatedStringPtr out = session.GetOutputNameAllocated(i, allocator);
        std::string outName = out ? std::string(out.get()) : std::string();
        if (outName.empty())
            outName = format("output_%zu", i);

        if (haveArg(outName))
        {
            const int existingIdx = (int)argnames[outName];
            if (args[(size_t)existingIdx].kind == DNN_ARG_OUTPUT)
            {
                outs.push_back(Arg(existingIdx));
                continue;
            }

            std::string uniqueOutName = format("%s_%zu", outName.c_str(), i);
            while (haveArg(uniqueOutName))
                uniqueOutName += "_";
            outName = uniqueOutName;
        }

        outs.push_back(newArg(outName, DNN_ARG_OUTPUT));
    }
    mainGraph->setOutputs(outs);
}

void Net::Impl::finalizeOrt()
{
    if (ort_session && !ortNeedsReinit)
        return;
    CV_Assert(!modelFileName.empty());

    int target = IS_DNN_CUDA_TARGET(preferableTarget) ? preferableTarget : DNN_TARGET_CPU;
    ort_session.reset();
    ort_names_cache.reset();

    static auto s_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "OpenCV_DNN_ORT");
    ort_env = s_env;

    Ort::SessionOptions opts;
    if (IS_DNN_CUDA_TARGET(target))
    {
        OrtCUDAProviderOptions cuda{};
        cuda.device_id = 0;
        try { opts.AppendExecutionProvider_CUDA(cuda); }
        catch (const std::exception& e)
        {
            CV_LOG_WARNING(NULL, "DNN/ONNX/ORT: CUDA EP unavailable (" << e.what() << "), using CPU");
            target = DNN_TARGET_CPU;
            opts = Ort::SessionOptions();
        }
    }

#ifdef _WIN32
    std::wstring wpath(modelFileName.begin(), modelFileName.end());
    ort_session = std::make_shared<Ort::Session>(*ort_env, wpath.c_str(), opts);
#else
    ort_session = std::make_shared<Ort::Session>(*ort_env, modelFileName.c_str(), opts);
#endif
    preferableTarget = target;
    ortNeedsReinit = false;

    refreshOrtMainGraphOutputs();
    applyStagedOrtInputs();
}
#endif

Ptr<BackendWrapper> Net::Impl::wrap(Mat& host)
{
    if (preferableBackend == DNN_BACKEND_OPENCV &&
            (preferableTarget == DNN_TARGET_CPU || preferableTarget == DNN_TARGET_CPU_FP16))
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
            CV_CheckType(host.depth(), host.depth() == CV_32F || host.depth() == CV_8S || host.depth() == CV_8U || host.depth() == CV_32S || host.depth() == CV_64S || host.depth() == CV_Bool, "Unsupported type for CUDA");
            CV_Assert(IS_DNN_CUDA_TARGET(preferableTarget));
            switch (host.depth())
            {
            case CV_32F:
                if (preferableTarget == DNN_TARGET_CUDA_FP16)
                    return CUDABackendWrapperFP16::create(baseBuffer, shape);
                else
                    return CUDABackendWrapperFP32::create(baseBuffer, shape);
            case CV_8S:
                return CUDABackendWrapperINT8::create(baseBuffer, shape);
            case CV_8U:
                return CUDABackendWrapperUINT8::create(baseBuffer, shape);
            case CV_32S:
                return CUDABackendWrapperINT32::create(baseBuffer, shape);
            case CV_64S:
                return CUDABackendWrapperINT64::create(baseBuffer, shape);
            case CV_Bool:
                return CUDABackendWrapperBOOL::create(baseBuffer, shape);
            default:
                CV_Error(Error::BadDepth, "Unsupported mat type for CUDA");
            }
#endif
        }
        else if (preferableBackend == DNN_BACKEND_TIMVX)
        {
#ifdef HAVE_TIMVX
            return Ptr<BackendWrapper>(new TimVXBackendWrapper(baseBuffer, host));
#endif
        }
        else if (preferableBackend == DNN_BACKEND_CANN)
        {
            CV_Assert(0 && "Internal error: DNN_BACKEND_CANN must be implemented through inheritance");
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
        CV_Assert(preferableTarget == DNN_TARGET_CPU || preferableTarget == DNN_TARGET_CPU_FP16 || IS_DNN_OPENCL_TARGET(preferableTarget));
    }
    else if (preferableBackend == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        CV_Assert(0 && "Inheritance must be used with OpenVINO backend");
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
    else if (preferableBackend == DNN_BACKEND_CANN)
    {
        CV_Assert(0 && "Internal error: DNN_BACKEND_CANN must be implemented through inheritance");
    }
    else
    {
        CV_Error(Error::StsNotImplemented, cv::format("Unknown backend identifier: %d", preferableBackend));
    }
}


void Net::Impl::setPreferableBackend(Net& net, int backendId)
{
    if (backendId == DNN_BACKEND_DEFAULT)
        backendId = (Backend)getParam_DNN_BACKEND_DEFAULT();

    if (backendId == DNN_BACKEND_INFERENCE_ENGINE)
        backendId = DNN_BACKEND_INFERENCE_ENGINE_NGRAPH;

    if (netWasQuantized && backendId != DNN_BACKEND_OPENCV && backendId != DNN_BACKEND_TIMVX &&
        backendId != DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
        CV_LOG_WARNING(NULL, "DNN: Only default, TIMVX and OpenVINO backends support quantized networks");
        backendId = DNN_BACKEND_OPENCV;
    }
#ifdef HAVE_DNN_NGRAPH
    if (netWasQuantized && backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH && INF_ENGINE_VER_MAJOR_LT(INF_ENGINE_RELEASE_2023_0))
    {
        CV_LOG_WARNING(NULL, "DNN: OpenVINO 2023.0 and higher is required to supports quantized networks");
        backendId = DNN_BACKEND_OPENCV;
    }
#endif

    if (preferableBackend == backendId)
        return;

#ifdef HAVE_ONNXRUNTIME
    if (useOrtEngine && mainGraph && modelFormat == DNN_MODEL_ONNX && !modelFileName.empty())
    {
        preferableBackend = backendId;
        ortNeedsReinit = true;  // will be applied on finalizeNet()
        return;
    }
#endif

    if (mainGraph)
    {
        CV_LOG_WARNING(NULL, "Back-ends are not supported by the new graph engine for now");
        preferableBackend = backendId;
        return;
    }

    clear();
    if (backendId == DNN_BACKEND_INFERENCE_ENGINE_NGRAPH)
    {
#if defined(HAVE_INF_ENGINE)
        switchToOpenVINOBackend(net);
#elif defined(ENABLE_PLUGINS)
        auto& networkBackend = dnn_backend::createPluginDNNNetworkBackend("openvino");
        networkBackend.switchBackend(net);
#else
        CV_Error(Error::StsNotImplemented, "OpenVINO backend is not available in the current OpenCV build");
#endif
    }
    else if (backendId == DNN_BACKEND_CANN)
    {
#ifdef HAVE_CANN
        switchToCannBackend(net);
#else
        CV_Error(Error::StsNotImplemented, "CANN backend is not availlable in the current OpenCV build");
#endif
    }
    else
    {
        preferableBackend = backendId;
    }
}

void Net::Impl::setPreferableTarget(int targetId)
{
#ifdef HAVE_ONNXRUNTIME
    if (useOrtEngine && mainGraph && modelFormat == DNN_MODEL_ONNX && !modelFileName.empty())
    {
        int resolved = IS_DNN_CUDA_TARGET(targetId) ? targetId : DNN_TARGET_CPU;
        if (preferableTarget != resolved)
        {
            preferableTarget = resolved;
            ortNeedsReinit = true;  // will be applied on finalizeNet()
        }
        return;
    }
#endif

    if (mainGraph)
    {
        CV_LOG_WARNING(NULL, "Targets are not supported by the new graph engine for now");
        return;
    }
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

        if (IS_DNN_CUDA_TARGET(targetId))
        {
            preferableTarget = DNN_TARGET_CPU;
#ifdef HAVE_CUDA
            if (cuda4dnn::doesDeviceSupportFP16() && targetId == DNN_TARGET_CUDA_FP16)
                preferableTarget = DNN_TARGET_CUDA_FP16;
            else
                preferableTarget = DNN_TARGET_CUDA;
#endif
        }
#if !defined(__arm64__) || !__arm64__
        if (targetId == DNN_TARGET_CPU_FP16)
        {
            CV_LOG_WARNING(NULL, "DNN: fall back to DNN_TARGET_CPU. Only ARM v8 CPU is supported by DNN_TARGET_CPU_FP16.");
            targetId = DNN_TARGET_CPU;
        }
#endif

        clear();

        if (targetId == DNN_TARGET_CPU_FP16)
        {
            if (useWinograd) {
                CV_LOG_INFO(NULL, "DNN: DNN_TARGET_CPU_FP16 is set => Winograd convolution is disabled by default to preserve accuracy. If needed, enable it explicitly using enableWinograd(true).");
                enableWinograd(false);
            }
        }
    }
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
