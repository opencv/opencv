// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "webgpu_internal.hpp"
#include <unistd.h>
namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU
Tensor::Tensor(Format fmt) : size_in_byte_(0), format_(fmt)
{
    createContext();
    device_ = wDevice;
}

Tensor::Tensor(const void* data, std::vector<int>& shape, Format fmt)
{
    createContext();
    device_ = wDevice;
    size_in_byte_ = 0;
    format_ = fmt;
    reshape(data, shape);
}

const void* Tensor::mapRead()
{
    return buffer_->MapReadAsyncAndWait();

}

void Tensor::unMap()
{
    buffer_->unMap();
}

Shape Tensor::getShape() const{
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

Tensor Tensor::reshape(const void* data, const std::vector<int>& shape,
                       bool alloc, Format fmt)
{
    if (device_ == nullptr)
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
    if(alloc || !buffer_)
    {
        buffer_.reset(new Buffer(device_, data, size_in_byte_, usage_));
    }
    else if (data)
    {
        buffer_->setBufferData(data, size_in_byte_);
    }
    return * this;
}

int Tensor::getFormat() const
{
    return format_;
}

void Tensor::copyTo(Tensor & dst)
{
    dst.reshape(buffer_->MapReadAsyncAndWait(), shape_, true, format_);
}

#endif   //HAVE_WEBGPU

}}}  //namespace cv::dnn:webgpu