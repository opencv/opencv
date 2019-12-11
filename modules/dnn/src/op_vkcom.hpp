// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_OP_VKCOM_HPP
#define OPENCV_DNN_OP_VKCOM_HPP

#include <opencv2/dnn/shape_utils.hpp>
#ifdef HAVE_VULKAN
#include "vkcom/include/vkcom.hpp"
#endif  // HAVE_VULKAN

namespace cv
{
namespace dnn
{
#ifdef HAVE_VULKAN
    std::vector<vkcom::Tensor> VkComTensors(const std::vector<Ptr<BackendWrapper> >& ptrs);

    vkcom::Tensor VkComTensor(const Ptr<BackendWrapper>& ptr);

    // Data copied from/to Mat to/from Tensor. Change the shape of dst if
    // needed to make it the same shape as src
    void copyToTensor(vkcom::Tensor &dst, const Mat &src);

    void copyToMat(Mat &dst, const vkcom::Tensor &src);

    class VkComBackendNode : public BackendNode
    {
    public:
        VkComBackendNode(const std::vector<Ptr<BackendWrapper> >& inputsWrapper,
                         const std::shared_ptr<vkcom::OpBase> &op,
                         const std::vector<Ptr<BackendWrapper> >& blobsWrapper =
                         std::vector<Ptr<BackendWrapper> >());

        bool forward(std::vector<vkcom::Tensor>& outs);

    private:
        std::vector<vkcom::Tensor> ins;
        std::vector<vkcom::Tensor> blobs;
        std::vector<Ptr<BackendWrapper> > inputsWrapper_;
        std::shared_ptr<vkcom::OpBase> operation;
    };

    class VkComBackendWrapper : public BackendWrapper
    {
    public:
        VkComBackendWrapper(Mat& m);
        VkComBackendWrapper(const Ptr<BackendWrapper>& baseBuffer, Mat& m);

        virtual void copyToHost() CV_OVERRIDE;
        virtual void setHostDirty() CV_OVERRIDE;
        void setDeviceDirty();
        void copyToDevice();
        vkcom::Tensor getTensor();

    private:
        vkcom::Tensor tensor;
        Mat* host;
        bool hostDirty;
        bool deviceDirty;
    };
#endif // HAVE_VULKAN

    void forwardVkCom(std::vector<Ptr<BackendWrapper> > &outputs, const Ptr<BackendNode>& node);

    bool haveVulkan();
}  // namespace dnn
}  // namespace cv

#endif  // OPENCV_DNN_OP_VKCOM_HPP
