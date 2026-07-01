// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#include "op_cuda.hpp"
#include "net_impl.hpp"

#ifdef HAVE_CUDA
#include "cuda4dnn/init.hpp"
#endif

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_CUDA
namespace {

struct CudaInfo_t final : public Net::Impl::CudaInfoBase
{
    CudaInfo_t(cuda4dnn::csl::CSLContext ctxt, cuda4dnn::csl::Stream d2h_stream_)
        : context(std::move(ctxt))
        , d2h_stream(std::move(d2h_stream_))
    {}
    cuda4dnn::csl::CSLContext context;
    cuda4dnn::csl::Stream d2h_stream;
    cuda4dnn::csl::Workspace workspace;
};

CudaInfo_t& getCudaInfo(const std::unique_ptr<Net::Impl::CudaInfoBase>& info)
{
    CV_Assert(info);
    return *static_cast<CudaInfo_t*>(info.get());
}

}  // namespace
#endif

bool Net::Impl::isCUDABackendAvailable() const
{
    return haveCUDA();
}

int Net::Impl::getEffectiveCUDATarget(int targetId) const
{
#ifdef HAVE_CUDA
    if (cuda4dnn::doesDeviceSupportFP16() && targetId == DNN_TARGET_CUDA_FP16)
        return DNN_TARGET_CUDA_FP16;
    return DNN_TARGET_CUDA;
#else
    CV_UNUSED(targetId);
    return DNN_TARGET_CPU;
#endif
}

Ptr<BackendWrapper> Net::Impl::wrapCUDA(Mat& host)
{
#ifdef HAVE_CUDA
    CV_Assert(haveCUDA());
    CV_CheckType(host.depth(), host.depth() == CV_32F || host.depth() == CV_8S || host.depth() == CV_8U || host.depth() == CV_32S || host.depth() == CV_64S || host.depth() == CV_Bool, "Unsupported type for CUDA");
    CV_Assert(IS_DNN_CUDA_TARGET(preferableTarget));
    switch (host.depth())
    {
    case CV_32F:
        if (preferableTarget == DNN_TARGET_CUDA_FP16)
            return CUDABackendWrapperFP16::create(host);
        else
            return CUDABackendWrapperFP32::create(host);
    case CV_8S:
        return CUDABackendWrapperINT8::create(host);
    case CV_8U:
        return CUDABackendWrapperUINT8::create(host);
    case CV_32S:
        return CUDABackendWrapperINT32::create(host);
    case CV_64S:
        return CUDABackendWrapperINT64::create(host);
    case CV_Bool:
        return CUDABackendWrapperBOOL::create(host);
    default:
        CV_Error(Error::BadDepth, "Unsupported mat type for CUDA");
    }
#else
    CV_UNUSED(host);
    CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of CUDA/CUDNN");
#endif
}

Ptr<BackendWrapper> Net::Impl::wrapCUDA(const Ptr<BackendWrapper>& baseBuffer, const MatShape& shape, int hostDepth)
{
#ifdef HAVE_CUDA
    CV_Assert(haveCUDA());
    CV_CheckType(hostDepth, hostDepth == CV_32F || hostDepth == CV_8S || hostDepth == CV_8U || hostDepth == CV_32S || hostDepth == CV_64S || hostDepth == CV_Bool, "Unsupported type for CUDA");
    CV_Assert(IS_DNN_CUDA_TARGET(preferableTarget));
    switch (hostDepth)
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
#else
    CV_UNUSED(baseBuffer);
    CV_UNUSED(shape);
    CV_UNUSED(hostDepth);
    CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of CUDA/CUDNN");
#endif
}

void Net::Impl::initCUDABackend(const std::vector<LayerPin>& blobsToKeep_)
{
#ifdef HAVE_CUDA
    CV_Assert(preferableBackend == DNN_BACKEND_CUDA);

    if (!cudaInfo) /* we need to check only once */
        cuda4dnn::checkVersions();

    if (cuda4dnn::getDeviceCount() <= 0)
        CV_Error(Error::StsError, "No CUDA capable device found.");

    if (cuda4dnn::getDevice() < 0)
        CV_Error(Error::StsError, "No CUDA capable device selected.");

    if (!cuda4dnn::isDeviceCompatible())
        CV_Error(Error::GpuNotSupported, "OpenCV was not built to work with the selected device. Please check CUDA_ARCH_PTX or CUDA_ARCH_BIN in your build configuration.");

    if (preferableTarget == DNN_TARGET_CUDA_FP16 && !cuda4dnn::doesDeviceSupportFP16())
    {
        CV_LOG_WARNING(NULL, "The selected CUDA device does not support FP16 target; switching to FP32 target.");
        preferableTarget = DNN_TARGET_CUDA;
    }

    if (!cudaInfo)
    {
        cuda4dnn::csl::CSLContext context;
        context.stream = cuda4dnn::csl::Stream(true);
        context.cublas_handle = cuda4dnn::csl::cublas::Handle(context.stream);
        context.cudnn_handle = cuda4dnn::csl::cudnn::Handle(context.stream);

        auto d2h_stream = cuda4dnn::csl::Stream(true);  // stream for background D2H data transfers
        cudaInfo = std::unique_ptr<CudaInfoBase>(new CudaInfo_t(std::move(context), std::move(d2h_stream)));
    }

    CudaInfo_t& info = getCudaInfo(cudaInfo);
    info.workspace = cuda4dnn::csl::Workspace();  // release workspace memory if any

    for (auto& layer : layers)
    {
        auto& ld = layer.second;

        if (ld.id == 0 && netInputLayer->supportBackend(preferableBackend))
        {
            for (auto& wrapper : ld.inputBlobsWrappers)
            {
                auto cudaWrapper = wrapper.dynamicCast<CUDABackendWrapper>();
                cudaWrapper->setStream(info.context.stream, info.d2h_stream);
            }
        }

        for (auto& wrapper : ld.outputBlobsWrappers)
        {
            auto cudaWrapper = wrapper.dynamicCast<CUDABackendWrapper>();
            cudaWrapper->setStream(info.context.stream, info.d2h_stream);
        }
    }

    for (auto& layer : layers)
    {
        auto& ld = layer.second;
        auto& layerInstance = ld.layerInstance;

        if (!layerInstance->supportBackend(DNN_BACKEND_CUDA))
        {
            std::ostringstream os;
            os << "CUDA backend will fallback to the CPU implementation for the layer \"" << ld.name
               << "\" of type " << ld.type << '\n';
            CV_LOG_INFO(NULL, os.str().c_str());
            continue;
        }

        /* we make a copy so that `initCUDA` doesn't modify `cudaInfo->context` */
        auto context = info.context;
        auto node = layerInstance->initCUDA(&context, ld.inputBlobsWrappers, ld.outputBlobsWrappers);
        ld.backendNodes[DNN_BACKEND_CUDA] = node;

        if(!node.empty())
        {
            auto cudaNode = node.dynamicCast<CUDABackendNode>();
            info.workspace.require(cudaNode->get_workspace_memory_in_bytes());
        }
    }

    if (blobsToKeep_.size() > 1)
    {
        for (const auto& pin : blobsToKeep_)
        {
            LayerData& ld = layers[pin.lid];
            ld.cudaD2HBackgroundTransfers.push_back(pin.oid);
        }
    }
#else
    CV_UNUSED(blobsToKeep_);
    CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of CUDA/CUDNN");
#endif
}

void Net::Impl::forwardCUDALayer(LayerData& ld)
{
#ifdef HAVE_CUDA
    Ptr<BackendNode> node = ld.backendNodes[DNN_BACKEND_CUDA];
    CV_Assert(!node.empty());
    Ptr<CUDABackendNode> cudaNode = node.dynamicCast<CUDABackendNode>();
    CV_Assert(!cudaNode.empty());

    CudaInfo_t& info = getCudaInfo(cudaInfo);
    cudaNode->forward(ld.inputBlobsWrappers, ld.outputBlobsWrappers, info.workspace);

    for (auto id : ld.cudaD2HBackgroundTransfers)
    {
        auto wrapper = ld.outputBlobsWrappers[id].dynamicCast<CUDABackendWrapper>();
        wrapper->copyToHostInBackground();
    }
#else
    CV_UNUSED(ld);
    CV_Error(Error::StsNotImplemented, "This OpenCV version is built without support of CUDA/CUDNN");
#endif
}

void Net::Impl::synchronizeCUDABackend()
{
#ifdef HAVE_CUDA
    if (cudaInfo)
        getCudaInfo(cudaInfo).context.stream.synchronize();
#endif
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

namespace cv { namespace dnn {

bool haveCUDA()
{
#ifdef HAVE_CUDA
    int dev = 0;
    static bool ret = (cudaGetDevice(&dev) == cudaSuccess);
    return ret;
#else
    return false;
#endif
}

}}  // namespace cv::dnn
