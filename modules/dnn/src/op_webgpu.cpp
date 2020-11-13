// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_webgpu.hpp"

namespace cv
{
namespace dnn
{
#ifdef HAVE_WEBGPU
void copyToTensor(webgpu::Tensor &dst, const Mat &src)
{
    CV_Assert(src.isContinuous() && src.type() == CV_32F);

    std::vector<int> mat_shape = shape(src);
    dst.reshape((const char*)src.data, mat_shape);
}

void copyToMat(Mat &dst, webgpu::Tensor &src)
{
    CV_Assert(dst.type() == CV_32F);

    std::vector<int> shape = src.getShape();
    void *data = const_cast<void *>(src.mapRead() );
    Mat tmp(shape, CV_32F, data);
    tmp.copyTo(dst);
    src.unMap();
}

webgpu::Tensor WGPUTensor(const Ptr<BackendWrapper>& ptr)
{
    CV_Assert(!ptr.empty());
    return ptr.dynamicCast<WGPUBackendWrapper>()->getTensor();
}

std::vector<webgpu::Tensor>  WGPUTensors(const std::vector<Ptr<BackendWrapper> >& ptrs)
{
    std::vector<webgpu::Tensor> vec;
    vec.reserve(ptrs.size());
    for(const Ptr<BackendWrapper>& ptr : ptrs) {
        vec.push_back(WGPUTensor(ptr));
    }
    return vec;
}

WGPUBackendWrapper::WGPUBackendWrapper(Mat& m) : BackendWrapper(DNN_BACKEND_WEBGPU, DNN_TARGET_WEBGPU)
{
    copyToTensor(tensor, m);
    host = &m;
    hostDirty = false;
    deviceDirty = false;
}

WGPUBackendWrapper::WGPUBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
    : BackendWrapper(DNN_BACKEND_WEBGPU, DNN_TARGET_WEBGPU)
{
    Ptr<WGPUBackendWrapper> base = baseBuffer.dynamicCast<WGPUBackendWrapper>();
    CV_Assert(!base.empty());

    host = &m;
    tensor = base->tensor;
    CV_Assert(tensor.count() >= m.total());
    tensor.reshape(0, shape(m));
    hostDirty = false;
    deviceDirty = false;
}

void WGPUBackendWrapper::copyToHost()
{
    if(deviceDirty)
    {
        copyToMat(*host, tensor);
        deviceDirty = false;
    }
}

void WGPUBackendWrapper::copyToDevice()
{
    if(hostDirty)
    {
        copyToTensor(tensor, *host);
        hostDirty = false;
    }
}

void WGPUBackendWrapper::setHostDirty()
{
    hostDirty = true;
}

void WGPUBackendWrapper::setDeviceDirty()
{
    deviceDirty = true;
}

webgpu::Tensor WGPUBackendWrapper::getTensor()
{
    return tensor;
}

WGPUBackendNode::WGPUBackendNode(const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                        const std::shared_ptr<webgpu::OpBase> &op,
                        const std::vector<Ptr<BackendWrapper> >& blobsWrapper)
                        :BackendNode(DNN_BACKEND_WEBGPU)
{
    operation = op;
    inputsWrapper_ = inputsWrapper;
    ins = WGPUTensors(inputsWrapper_);
    if(!blobsWrapper.empty()) {
        blobs = WGPUTensors(blobsWrapper);
    }
}

bool WGPUBackendNode::forward(std::vector<webgpu::Tensor>& outs)
{
    for(int i = 0; i < inputsWrapper_.size(); i ++) {
        inputsWrapper_[i].dynamicCast<WGPUBackendWrapper>()->copyToDevice();
    }

    return operation->forward(ins, blobs, outs);
}

void setBackendWrappersDirty(std::vector<Ptr<BackendWrapper> >& ptrs)
{
    for (const Ptr<BackendWrapper>& ptr : ptrs)
    {
        ptr.dynamicCast<WGPUBackendWrapper>()->setDeviceDirty();
    }
}
#endif  // HAVE_WEBGPU

void forwardWGPU(std::vector<Ptr<BackendWrapper> > &outputs,
                const Ptr<BackendNode>& node)
{
#ifdef HAVE_WEBGPU
    CV_Assert(!node.empty());

    Ptr<WGPUBackendNode> node_ = node.dynamicCast<WGPUBackendNode>();
    std::vector<webgpu::Tensor> outs = WGPUTensors(outputs);
    node_->forward(outs);
    setBackendWrappersDirty(outputs);
#endif  // HAVE_WEBGPU
}

    bool haveWGPU()
    {
#ifdef HAVE_WEBGPU
        return webgpu::isAvailable();
#else
        return false;
#endif  // HAVE_WEBGPU
    }

}   //namespace dnn

}   //namespace cv