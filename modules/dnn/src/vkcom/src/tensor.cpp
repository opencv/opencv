// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "internal.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

Tensor::Tensor(Format fmt, VkBufferUsageFlags usageFlag) : size_in_byte_(0), format_(fmt), usageFlag_(usageFlag | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
{
}

Tensor::Tensor(const char* data, std::vector<int>& shape, Format fmt, VkBufferUsageFlags usageFlag)
               : size_in_byte_(0), format_(fmt), usageFlag_(usageFlag | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT)
{
    reshape(data, shape);
}

void* Tensor::map()
{
    void *p;

    VK_CHECK_RESULT(vkMapMemory(kDevice, buffer_->getVkMemory(),
                                0, size_in_byte_, 0, (void **)&p));

    return p;
}

void Tensor::unMap()
{
    vkUnmapMemory(kDevice, buffer_->getVkMemory());
}

void* Tensor::mapHost()
{
    if (!buffer_->isDeviceOnly())
    {
        return map();
    }

    VkBufferCopy pRegion;
    pRegion.srcOffset = 0;
    pRegion.dstOffset = 0;
    pRegion.size = size_in_byte_;

    hostBuffer_.reset(new Buffer(size_in_byte_, nullptr, usageFlag_, false));

    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();
    VkCommandBuffer cmdBufferReal = cmdBuffer->get();

    cmdBuffer->beginRecord();
    vkCmdCopyBuffer(cmdBufferReal, buffer_->getVkBuffer(), hostBuffer_->getVkBuffer(), 1, &pRegion);
    cmdBuffer->endRecord();
    cmdPoolPtr->submitAndWait(cmdBufferReal);

    void *p;

    VK_CHECK_RESULT(vkMapMemory(kDevice, hostBuffer_->getVkMemory(),
                                0, size_in_byte_, 0, (void **)&p));
    CV_LOG_DEBUG(NULL, "mapped to host.");
    return p;
}

void Tensor::unMapHostReadOnly()
{
    if (!buffer_->isDeviceOnly())
    {
        unMap();
        return;
    }
    CV_DbgAssert(hostBuffer_ != nullptr);
    vkUnmapMemory(kDevice, hostBuffer_->getVkMemory());
}
    
void Tensor::unMapHostWriteToDevice()
{
    if (!buffer_->isDeviceOnly())
    {
        unMap();
        return;
    }
    CV_DbgAssert(hostBuffer_ != nullptr);
    
    VkBufferCopy pRegion;
    pRegion.srcOffset = 0;
    pRegion.dstOffset = 0;
    pRegion.size = size_in_byte_;

    Ptr<CommandBuffer> cmdBuffer = cmdPoolPtr->allocBuffer();
    VkCommandBuffer cmdBufferReal = cmdBuffer->get();

    cmdBuffer->beginRecord();
    vkCmdCopyBuffer(cmdBufferReal, hostBuffer_->getVkBuffer(), buffer_->getVkBuffer(), 1, &pRegion);
    cmdBuffer->endRecord();
    cmdPoolPtr->submitAndWait(cmdBufferReal);

    vkUnmapMemory(kDevice, hostBuffer_->getVkMemory());
}

Shape Tensor::getShape() const
{
    return shape_;
}

int Tensor::count(const int start_axis, const int end_axis) const
{
    return shapeCount(shape_, start_axis, end_axis);
}

int Tensor::dimSize(const int axis) const
{
    CV_Assert(axis >= 0);
    CV_Assert(axis < shape_.size());

    return shape_[axis];
}

int Tensor::dimNum() const
{
    return shape_.size();
}

Tensor Tensor::reshape(const char* data, const std::vector<int>& shape, bool alloc, Format fmt)
{
    if (kDevice == VK_NULL_HANDLE)
    {
        CV_Error(Error::StsError, "device is NULL!");
        return *this;
    }

    CV_Assert(shape.size() > 0 && shape.size() <= 6);

    if (shape_ != shape) shape_ = shape;
    if (checkFormat(fmt) && fmt != format_) format_ = fmt;

    size_t new_size = shapeCount(shape_) * elementSize(format_);
    if (alloc || new_size > size_in_byte_)
        alloc = true;
    size_in_byte_ = new_size;

    if (alloc)
    {
        buffer_.reset(new Buffer(size_in_byte_, nullptr, usageFlag_));
        if (data)
        {
            void *p = mapHost();
            memcpy(p, data, size_in_byte_);
            unMapHostWriteToDevice();
        }
    }
    else if (data)
    {
        void *p = mapHost();
        memcpy(p, data, size_in_byte_);
        unMapHostWriteToDevice();
    }

    return *this;
}

void Tensor::setTo(float val)
{
    if (kDevice == VK_NULL_HANDLE)
    {
        CV_Error(Error::StsError, "device is NULL!");
        return;
    }

    CV_Assert(format_ == kFormatFp32);

    float* p = (float *)mapHost();
    int cnt = count();
    for (int i = 0; i < cnt; i++)
        *p++ = val;
    unMapHostWriteToDevice();
}

int Tensor::getFormat() const
{
    return format_;
}

void Tensor::copyTo(Tensor& dst)
{
    void *p = mapHost();
    dst.reshape((const char*)p, shape_, format_);
    unMapHostReadOnly();
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
