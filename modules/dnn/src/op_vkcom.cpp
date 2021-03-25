// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "precomp.hpp"
#include <opencv2/dnn/shape_utils.hpp>
#include "op_vkcom.hpp"

namespace cv
{
namespace dnn
{
#ifdef HAVE_VULKAN
    void copyToTensor(vkcom::Tensor &dst, const Mat &src)
    {
        CV_Assert(src.isContinuous() && src.type() == CV_32F);

        std::vector<int> mat_shape = shape(src);
        dst.reshape((const char*)src.data, mat_shape);
    }

    void copyToMat(Mat &dst, vkcom::Tensor &src)
    {
        CV_Assert(dst.type() == CV_32F);

        std::vector<int> shape = src.getShape();
        void *data = src.map();
        Mat tmp(shape, CV_32F, data);
        tmp.copyTo(dst);
        src.unMap();
    }

    vkcom::Tensor VkComTensor(const Ptr<BackendWrapper>& ptr)
    {
        CV_Assert(!ptr.empty());
        return ptr.dynamicCast<VkComBackendWrapper>()->getTensor();
    }

    void setDirty(std::vector<Ptr<BackendWrapper> >& ptrs)
    {
        for (const Ptr<BackendWrapper>& ptr : ptrs)
        {
            ptr.dynamicCast<VkComBackendWrapper>()->setDeviceDirty();
        }
    }

    std::vector<vkcom::Tensor> VkComTensors(const std::vector<Ptr<BackendWrapper> >& ptrs)
    {
        std::vector<vkcom::Tensor> vec;
        vec.reserve(ptrs.size());
        for (const Ptr<BackendWrapper>& ptr : ptrs)
        {
            vec.push_back(VkComTensor(ptr));
        }
        return vec;
    }

    VkComBackendNode::VkComBackendNode(const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                                       const std::shared_ptr<vkcom::OpBase>& op,
                                       const std::vector<Ptr<BackendWrapper> >& blobsWrapper)
                                       : BackendNode(DNN_BACKEND_VKCOM)
    {
        operation = op;

        inputsWrapper_ = inputsWrapper;
        ins = VkComTensors(inputsWrapper_);

        if (!blobsWrapper.empty())
        {
            blobs = VkComTensors(blobsWrapper);
        }
    }

    bool VkComBackendNode::forward(std::vector<vkcom::Tensor>& outs)
    {
        for (int i = 0, n = inputsWrapper_.size(); i < n; ++i)
        {
            inputsWrapper_[i].dynamicCast<VkComBackendWrapper>()->copyToDevice();
        }

        return operation->forward(ins, blobs, outs);
    }

    VkComBackendWrapper::VkComBackendWrapper(Mat& m) : BackendWrapper(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN)
    {
        copyToTensor(tensor, m);
        host = &m;
        hostDirty = false;
        deviceDirty = false;
    }

    VkComBackendWrapper::VkComBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m)
        : BackendWrapper(DNN_BACKEND_VKCOM, DNN_TARGET_VULKAN)
    {
        Ptr<VkComBackendWrapper> base = baseBuffer.dynamicCast<VkComBackendWrapper>();
        CV_Assert(!base.empty());

        host = &m;
        tensor = base->tensor;
        CV_Assert(tensor.count() >= m.total());
        tensor.reshape(0, shape(m));
        hostDirty = false;
        deviceDirty = false;
    }

    void VkComBackendWrapper::copyToHost()
    {
        if (deviceDirty)
            copyToMat(*host, tensor);
    }

    void VkComBackendWrapper::setHostDirty()
    {
        hostDirty = true;
    };

    void VkComBackendWrapper::setDeviceDirty()
    {
        deviceDirty = true;
    };

    void VkComBackendWrapper::copyToDevice()
    {
        if (hostDirty)
        {
            copyToTensor(tensor, *host);
            hostDirty = false;
        }
    }

    vkcom::Tensor VkComBackendWrapper::getTensor()
    {
        return tensor;
    }
#endif
    void forwardVkCom(std::vector<Ptr<BackendWrapper> > &outputs,
                      const Ptr<BackendNode>& node)
    {
#ifdef HAVE_VULKAN
        CV_Assert(!node.empty());

        Ptr<VkComBackendNode> node_ = node.dynamicCast<VkComBackendNode>();
        std::vector<vkcom::Tensor> outs = VkComTensors(outputs);
        node_->forward(outs);
        setDirty(outputs);
#endif
    }

    bool haveVulkan()
    {
#ifdef HAVE_VULKAN
        return vkcom::isAvailable();
#else
        return false;
#endif  // HAVE_VULKAN
    }

}  // namespace dnn
}  // namespace cv
