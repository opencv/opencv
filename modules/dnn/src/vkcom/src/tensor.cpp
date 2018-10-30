// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

Tensor::Tensor(Format fmt) : size_in_byte_(0), format_(fmt)
{
    Context *ctx = getContext();
    device_ = ctx->device;
}

Tensor::Tensor(const char* data, std::vector<int>& shape, Format fmt)
               : size_in_byte_(0), format_(fmt)
{
    Context *ctx = getContext();
    device_ = ctx->device;
    reshape(data, shape);
}

void* Tensor::map()
{
    void *p;

    VK_CHECK_RESULT(vkMapMemory(device_, buffer_->getVkMemory(),
                                0, size_in_byte_, 0, (void **)&p));

    return p;
}

void Tensor::unMap()
{
    vkUnmapMemory(device_, buffer_->getVkMemory());
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
    if (device_ == VK_NULL_HANDLE)
    {
        CV_Error(Error::StsError, "device is NULL");
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
        buffer_.reset(new Buffer(device_, size_in_byte_, data));
    }
    else if (data)
    {
        void* p = map();
        memcpy(p, data, size_in_byte_);
        unMap();
    }

    return *this;
}

void Tensor::setTo(float val)
{
    if (device_ == VK_NULL_HANDLE)
    {
        CV_Error(Error::StsError, "device is NULL");
        return;
    }

    CV_Assert(format_ == kFormatFp32);

    float* p = (float *)map();
    int cnt = count();
    for (int i = 0; i < cnt; i++)
        *p++ = val;
    unMap();
}

int Tensor::getFormat() const
{
    return format_;
}

void Tensor::copyTo(Tensor& dst)
{
    void* p = map();
    dst.reshape((const char*)p, shape_, format_);
    unMap();
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
