// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <stdlib.h> // for atexit
#include <opencv2/dnn/shape_utils.hpp>
#include "op_ascendcl.hpp"
#include "net_impl.hpp"

namespace cv
{
namespace dnn
{

#ifdef HAVE_ASCENDCL

CV__DNN_INLINE_NS_BEGIN

void Net::Impl::initAscendCLBackend()
{
    CV_TRACE_FUNCTION();
    CV_Assert(preferableBackend == DNN_BACKEND_ASCENDCL);

    // Init AscendCL, called only once every process!
    if (++cannRefCount == 1)
    {
        aclError ret = aclInit(NULL);
        CV_Assert(ret == ACL_SUCCESS);
    }
    // aclEnvGuard = AclEnvGuard::GetAclEnv();
    // init ascend client for acl context and stream
    if (!cannInfo)
    {
        cannInfo = std::unique_ptr<CannInfo>(new CannInfo());
        cannInfo->init(0);
    }

    for (MapIdToLayerData::iterator it = layers.begin(); it != layers.end(); it++)
    {
        LayerData& ld = it->second;
        Ptr<Layer> layer = ld.layerInstance;
        if (!layer->supportBackend(preferableBackend))
        {
            continue;
        }

        ld.skip = false;

        try
        {
            ld.backendNodes[DNN_BACKEND_ASCENDCL] =
                layer->initAscendCL(cannInfo.get(), ld.inputBlobsWrappers, ld.outputBlobsWrappers);
        }
        catch (const cv::Exception& e)
        {
            CV_LOG_ERROR(NULL, "initAscendCL failed, fall back to CPU implementation! " << e.what());
            ld.backendNodes[DNN_BACKEND_ASCENDCL] = Ptr<BackendNode>();
        }
    }
}

CV__DNN_INLINE_NS_END


void copyToTensor(std::shared_ptr<ascendcl::Tensor> dst, const Mat& src)
{
    CV_Assert(src.isContinuous());

    std::vector<int> mat_shape = shape(src);

    // determine data type for ascendcl
    aclDataType dtype = ACL_DT_UNDEFINED;
    const int m_dtype = src.type();
    switch(m_dtype)
    {
        case CV_32F: dtype = ACL_FLOAT; break;
        case CV_16F: dtype = ACL_FLOAT16; break;
        default: CV_Error(cv::Error::StsBadArg, "Unsupported type.");
    }

    // determine data layerout format for ascendcl
    aclFormat fmt = ACL_FORMAT_UNDEFINED;
    switch(mat_shape.size())
    {
        case 4: fmt = ACL_FORMAT_NCHW; break;
        case 2:
        case 1: fmt = ACL_FORMAT_ND; break;
        default: CV_Error(cv::Error::StsBadArg, "Unsupported data layout.");
    }

    dst->reshape((const void*)src.data, mat_shape, dtype, fmt);
}

void copyToMat(Mat& dst, std::shared_ptr<ascendcl::Tensor> src)
{
    CV_Assert(dst.type() == CV_32F || dst.type() == CV_16F);

    src->toMat(dst);
}

void setDirty(std::vector<Ptr<BackendWrapper> >& ptrs)
{
    for (const Ptr<BackendWrapper>& ptr : ptrs)
    {
        ptr.dynamicCast<AscendCLBackendWrapper>()->setDeviceDirty();
    }
}

void CannInfo::init(int deviceId)
{
    device_id = deviceId;

    // set device for ascend
    aclError ret = aclrtSetDevice(device_id);
    CV_Assert(ret == ACL_SUCCESS);
    // create context from device
    ret = aclrtCreateContext(&context, device_id);
    CV_Assert(ret == ACL_SUCCESS);
    // create stream
    ret = aclrtCreateStream(&stream);
    CV_Assert(ret == ACL_SUCCESS);
}

CannInfo::~CannInfo()
{
    // destroy stream
    aclError ret;
    if (stream != nullptr)
    {
        ret = aclrtDestroyStream(stream);
        CV_Assert(ret == ACL_SUCCESS);
        stream = nullptr;
    }
    // destroy context
    if (context != nullptr)
    {
        ret = aclrtDestroyContext(context);
        CV_Assert(ret == ACL_SUCCESS);
        context = nullptr;
    }
    // reset device
    if (context == nullptr && stream == nullptr)
    {
        ret = aclrtResetDevice(device_id);
        CV_Assert(ret == ACL_SUCCESS);
    }
}

aclrtStream CannInfo::getStream() const
{
    return stream;
}

void CannInfo::syncStream()
{
    aclError ret = aclrtSynchronizeStream(stream);
    CV_Assert(ret == ACL_SUCCESS);
}

AscendCLBackendNode::AscendCLBackendNode(const aclrtStream stream,
                                         const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                                         const std::shared_ptr<ascendcl::Operator>& op,
                                         const std::vector<Ptr<BackendWrapper> >& blobsWrapper)
    : BackendNode(DNN_BACKEND_ASCENDCL), stream_(stream)
{
    operator_ = op;

    inputs_wrapper = inputsWrapper;
    blobs_wrapper = blobsWrapper;
}

bool AscendCLBackendNode::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> outs)
{
    std::vector<std::shared_ptr<ascendcl::Tensor>> inputs_and_blobs;
    for (size_t i = 0; i < inputs_wrapper.size(); i++)
    {
        auto cann_backend_wrapper = inputs_wrapper[i].dynamicCast<AscendCLBackendWrapper>();
        cann_backend_wrapper->copyToDevice();
        auto _tensor = cann_backend_wrapper->getTensor();
        inputs_and_blobs.push_back(_tensor);
    }
    if (!blobs_wrapper.empty())
    {
        for (size_t i = 0; i < blobs_wrapper.size(); i++)
        {
            auto cann_backend_wrapper = blobs_wrapper[i].dynamicCast<AscendCLBackendWrapper>();
            cann_backend_wrapper->copyToDevice();
            auto _tensor = cann_backend_wrapper->getTensor();
            inputs_and_blobs.push_back(_tensor);
        }
    }

    return operator_->forward(inputs_and_blobs, outs, stream_);
}

AscendCLBackendWrapper::AscendCLBackendWrapper(Mat m) : BackendWrapper(DNN_BACKEND_ASCENDCL, DNN_TARGET_NPU)
{
    is_tensor = false;
    host = m;
    hostDirty = false;
    deviceDirty = false;
}

AscendCLBackendWrapper::AscendCLBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat m)
    : BackendWrapper(DNN_BACKEND_ASCENDCL, DNN_TARGET_NPU)
{
    Ptr<AscendCLBackendWrapper> base = baseBuffer.dynamicCast<AscendCLBackendWrapper>();
    CV_Assert(!base.empty());

    is_tensor = false;
    host = m;
    tensor = base->tensor;
    hostDirty = base->hostDirty;
    deviceDirty = base->deviceDirty;
}

void AscendCLBackendWrapper::createTensor()
{
    if (is_tensor)
        return ;

    tensor = std::shared_ptr<ascendcl::Tensor>(new ascendcl::Tensor());
    if (host.type() == CV_32F)
        host.convertTo(host, CV_16F);
    copyToTensor(tensor, host);
    is_tensor = true;
}

void AscendCLBackendWrapper::copyToHost()
{
    if (deviceDirty)
        copyToMat(host, tensor);
}

void AscendCLBackendWrapper::setHostDirty()
{
    hostDirty = true;
}

void AscendCLBackendWrapper::setDeviceDirty()
{
    deviceDirty = true;
}

void AscendCLBackendWrapper::copyToDevice()
{
    if (hostDirty)
    {
        if (host.type() == CV_32F)
            host.convertTo(host, CV_16F);
        copyToTensor(tensor, host);
        hostDirty = false;
    }
}

std::shared_ptr<ascendcl::Tensor> AscendCLBackendWrapper::getTensor()
{
    return tensor;
}

Mat AscendCLBackendWrapper::getMat() const
{
    return host;
}

int AscendCLBackendWrapper::getShapeAt(int axis) const
{
    CV_Assert(axis >= 0 && axis < shape(host).size());
    return shape(host)[axis];
}

#endif // HAVE_ASCENDCL

void forwardAscendCL(std::vector<Ptr<BackendWrapper> >& outputs,
                    const Ptr<BackendNode>& node)
{
#ifdef HAVE_ASCENDCL
    CV_Assert(!node.empty());

    Ptr<AscendCLBackendNode> node_ = node.dynamicCast<AscendCLBackendNode>();
    std::vector<std::shared_ptr<ascendcl::Tensor>> outs;
    for (size_t i = 0; i < outputs.size(); i++)
        outs.push_back(outputs[i].dynamicCast<AscendCLBackendWrapper>()->getTensor());
    node_->forward(outs);
    setDirty(outputs);
#endif // HAVE_ASCENDCL
}

bool haveAscendCL()
{
#ifdef HAVE_ASCENDCL
    return true;
#else
    return false;
#endif
}

} // namespace dnn
} // namespace cv
