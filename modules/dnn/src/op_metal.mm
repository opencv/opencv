// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"

#ifdef HAVE_METAL

#include "op_metal.hpp"
#include "net_impl.hpp"

#include "metal/metal.hpp"
#include "metal/ops/operation.hpp"
#include "metal/runtime/context.hpp"
#include "metal/runtime/device.hpp"
#include "metal/runtime/tensor.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void Net::Impl::initMetalBackend()
{
    CV_CheckTrue(metal::Device::instance() != nullptr, "No usable Metal device was found");

    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); ++it)
    {
        LayerData& ld = it->second;
        Ptr<Layer> layer = ld.layerInstance;
        if (!layer->supportBackend(DNN_BACKEND_METAL))
        {
            // CPU fallback must observe a CPU target.  Some layers (for example
            // 3D Pooling) use preferableTarget when reporting OpenCV backend
            // support and when selecting their host implementation.
            layer->preferableTarget = DNN_TARGET_CPU;
            continue;
        }

        try
        {
            ld.backendNodes[DNN_BACKEND_METAL] =
                layer->initMetal(ld.inputBlobsWrappers, ld.outputBlobsWrappers);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "Metal layer initialization failed; falling back to CPU. " << e.what());
            ld.backendNodes[DNN_BACKEND_METAL] = Ptr<BackendNode>();
            layer->preferableTarget = DNN_TARGET_CPU;
        }
    }
}

MetalBackendWrapper::MetalBackendWrapper(Mat& host)
    : BackendWrapper(DNN_BACKEND_METAL, DNN_TARGET_METAL), tensor_(metal::Tensor::create(host))
{
    hostMatDepth = host.depth();
}

MetalBackendWrapper::MetalBackendWrapper(const Ptr<BackendWrapper>& base, Mat& host)
    : BackendWrapper(DNN_BACKEND_METAL, DNN_TARGET_METAL)
{
    Ptr<MetalBackendWrapper> metalBase = base.dynamicCast<MetalBackendWrapper>();
    CV_CheckTrue(!metalBase.empty(), "Base wrapper is not a Metal wrapper");
    tensor_ = metal::Tensor::reshape(metalBase->tensor_, shape(host));
    hostMatDepth = host.depth();
}

void MetalBackendWrapper::copyToHost()
{
    tensor_->copyToHost();
}

void MetalBackendWrapper::setHostDirty()
{
    tensor_->setHostDirty();
}

void MetalBackendWrapper::copyToDevice()
{
    tensor_->copyToDevice();
}

void MetalBackendWrapper::setDeviceDirty()
{
    tensor_->setDeviceDirty();
}

const std::shared_ptr<metal::Tensor>& MetalBackendWrapper::tensor() const
{
    return tensor_;
}

MetalBackendNode::MetalBackendNode(std::unique_ptr<metal::Operation> operation)
    : BackendNode(DNN_BACKEND_METAL), operation_(std::move(operation))
{
}

MetalBackendNode::~MetalBackendNode() = default;

void MetalBackendNode::forward()
{
    operation_->forward(metal::Context::get());
}

Ptr<BackendNode> makeMetalBackendNode(std::unique_ptr<metal::Operation> operation)
{
    return Ptr<BackendNode>(new MetalBackendNode(std::move(operation)));
}

void forwardMetal(const Ptr<BackendNode>& node)
{
    Ptr<MetalBackendNode> metalNode = node.dynamicCast<MetalBackendNode>();
    metalNode->forward();
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // HAVE_METAL

#ifdef HAVE_METAL
namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

bool haveMetal()
{
    return metal::isAvailable();
}

CV__DNN_INLINE_NS_END
}}  // namespace cv::dnn
#endif  // HAVE_METAL
