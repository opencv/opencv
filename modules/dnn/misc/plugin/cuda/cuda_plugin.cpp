// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../../src/precomp.hpp"

#include <opencv2/core/utils/logger.hpp>

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "../../../src/backend.hpp"
#include "../../../src/cuda4dnn/init.hpp"
#include "../../../src/net_impl.hpp"
#define ABI_VERSION 0
#define API_VERSION 0

#include "../../../src/plugin_api.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

class NetImplCUDA CV_FINAL : public Net::Impl
{
public:
    typedef Net::Impl Base;

    explicit NetImplCUDA(const Ptr<Net::Impl>& basePtr)
        : Net::Impl()
    {
        basePtr_ = basePtr;
        init();
    }

    void init()
    {
        CV_TRACE_FUNCTION();
        CV_Assert(basePtr_);

        Net::Impl& base = *basePtr_;
        CV_Assert(!base.netWasAllocated);

        netInputLayer = base.netInputLayer;
        blobsToKeep = base.blobsToKeep;
        layers = base.layers;
        for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
        {
            LayerData& ld = it->second;
            ld.resetAllocation();
        }
        layerNameToId = base.layerNameToId;
        outputNameToId = base.outputNameToId;
        preferableBackend = DNN_BACKEND_CUDA;
        preferableTarget = IS_DNN_CUDA_TARGET(base.preferableTarget) ? base.preferableTarget : DNN_TARGET_CUDA;
        hasDynamicShapes = base.hasDynamicShapes;
        CV_Assert(base.backendWrappers.empty());
        lastLayerId = base.lastLayerId;
        netWasAllocated = base.netWasAllocated;
        netWasQuantized = base.netWasQuantized;
        fusion = base.fusion;
        isAsync = base.isAsync;
        useWinograd = base.useWinograd;
        useKVCache = base.useKVCache;
        modelFileName = base.modelFileName;
        modelFormat = base.modelFormat;
        originalLayout = base.originalLayout;
        accuracy = base.accuracy;
        defaultC0 = base.defaultC0;
        enableFP16 = base.enableFP16;
        haveFP16 = base.haveFP16;
        prepared = base.prepared;
        finalizeLayers = base.finalizeLayers;
        tracingMode = base.tracingMode;
        profilingMode = base.profilingMode;
    }

    void setPreferableBackend(Net& net, int backendId) CV_OVERRIDE
    {
        if (backendId == DNN_BACKEND_DEFAULT)
            backendId = (Backend)getParam_DNN_BACKEND_DEFAULT();

        if (backendId == DNN_BACKEND_CUDA)
            return;

        if (!basePtr_)
            CV_Error(Error::StsError, "DNN: Can't switch backend from CUDA plugin without a base network");

        Ptr<Net::Impl>& implPtrRef = accessor::DnnNetAccessor::getImplPtrRef(net);
        implPtrRef = basePtr_;
        basePtr_->setPreferableBackend(net, backendId);
    }

    void setPreferableTarget(int targetId) CV_OVERRIDE
    {
        if (!IS_DNN_CUDA_TARGET(targetId))
            CV_Error(Error::StsError, "DNN/CUDA backend supports only CUDA targets");

        int effectiveTarget = getEffectiveCUDATarget(targetId);
        if (preferableTarget != effectiveTarget)
        {
            preferableTarget = effectiveTarget;
            clear();
        }
    }

    Ptr<Layer> createLayerInstance(const LayerData& ld) const CV_OVERRIDE
    {
        Ptr<Layer> instance = LayerFactory::createLayerInstance(ld.type, const_cast<LayerParams&>(ld.params));
        if (!instance)
            instance = Base::createLayerInstance(ld);
        return instance;
    }
};

void switchToCudaBackend(Net& net)
{
    CV_TRACE_FUNCTION();
    Ptr<Net::Impl>& implPtrRef = accessor::DnnNetAccessor::getImplPtrRef(net);
    CV_Assert(implPtrRef);
    CV_LOG_INFO(NULL, "DNN: switching to CUDA plugin backend... (networkID=" << implPtrRef->networkId << ")");
    Ptr<NetImplCUDA> cudaImpl = makePtr<NetImplCUDA>(implPtrRef);
    implPtrRef = cudaImpl;
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn

namespace cv { namespace dnn_backend {

using namespace cv::dnn;

static bool isCudaTargetAvailable(Target target)
{
    if (target != DNN_TARGET_CUDA && target != DNN_TARGET_CUDA_FP16)
        return false;

    try
    {
        bool hasCudaCompatible = false;
        bool hasCudaFP16 = false;
        for (int i = 0; i < cuda4dnn::getDeviceCount(); i++)
        {
            if (cuda4dnn::isDeviceCompatible(i))
            {
                hasCudaCompatible = true;
                if (cuda4dnn::doesDeviceSupportFP16(i))
                {
                    hasCudaFP16 = true;
                    break;
                }
            }
        }

        if (target == DNN_TARGET_CUDA)
            return hasCudaCompatible;
        return hasCudaCompatible && hasCudaFP16;
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_WARNING(NULL, "DNN/CUDA plugin target check failed: " << e.what());
        return false;
    }
}

static bool areCudaDnnDependenciesAvailable()
{
    try
    {
        cuda4dnn::checkVersions();
    }
    catch (const cv::Exception& e)
    {
        CV_LOG_WARNING(NULL, "DNN/CUDA plugin version check failed: " << e.what());
        return false;
    }

    cublasHandle_t cublasHandle = NULL;
    if (cublasCreate(&cublasHandle) != CUBLAS_STATUS_SUCCESS)
        return false;
    cublasDestroy(cublasHandle);

    cudnnHandle_t cudnnHandle = NULL;
    if (cudnnCreate(&cudnnHandle) != CUDNN_STATUS_SUCCESS)
        return false;
    cudnnDestroy(cudnnHandle);

    return cudnnGetVersion() > 0;
}

class NetworkBackendCUDA CV_FINAL : public NetworkBackend
{
public:
    void switchBackend(Net& net) CV_OVERRIDE
    {
        switchToCudaBackend(net);
    }

    Net readNetwork(const std::string& loaderID, const std::string& model, const std::string& config) CV_OVERRIDE
    {
        CV_UNUSED(loaderID);
        CV_UNUSED(model);
        CV_UNUSED(config);
        CV_Error(Error::StsNotImplemented, "DNN/CUDA plugin does not provide a model loader");
    }

    Net readNetwork(
        const std::string& loaderID,
        const uchar* bufferModelConfigPtr, size_t bufferModelConfigSize,
        const uchar* bufferWeightsPtr, size_t bufferWeightsSize
    ) CV_OVERRIDE
    {
        CV_UNUSED(loaderID);
        CV_UNUSED(bufferModelConfigPtr);
        CV_UNUSED(bufferModelConfigSize);
        CV_UNUSED(bufferWeightsPtr);
        CV_UNUSED(bufferWeightsSize);
        CV_Error(Error::StsNotImplemented, "DNN/CUDA plugin does not provide a model loader");
    }

    bool checkTarget(Target target) CV_OVERRIDE
    {
        return areCudaDnnDependenciesAvailable() && isCudaTargetAvailable(target);
    }
};

static std::shared_ptr<NetworkBackendCUDA>& getInstanceNetworkBackendCUDA()
{
    static std::shared_ptr<NetworkBackendCUDA> instance = std::make_shared<NetworkBackendCUDA>();
    return instance;
}

}}  // namespace cv::dnn_backend

static CvResult cv_getInstanceNetworkBackend(CV_OUT CvPluginDNNNetworkBackend* handle) CV_NOEXCEPT
{
    try
    {
        if (!handle)
            return CV_ERROR_FAIL;
        *handle = cv::dnn_backend::getInstanceNetworkBackendCUDA().get();
        return CV_ERROR_OK;
    }
    catch (...)
    {
        return CV_ERROR_FAIL;
    }
}

static const OpenCV_DNN_Plugin_API plugin_api =
{
    {
        sizeof(OpenCV_DNN_Plugin_API), ABI_VERSION, API_VERSION,
        CV_VERSION_MAJOR, CV_VERSION_MINOR, CV_VERSION_REVISION, CV_VERSION_STATUS,
        "CUDA OpenCV DNN plugin"
    },
    {
        /*  1*/cv_getInstanceNetworkBackend
    }
};

const OpenCV_DNN_Plugin_API* CV_API_CALL opencv_dnn_plugin_init_v0(
        int requested_abi_version, int requested_api_version, void* /*reserved=NULL*/) CV_NOEXCEPT
{
    if (requested_abi_version == ABI_VERSION && requested_api_version <= API_VERSION)
        return &plugin_api;
    return NULL;
}
