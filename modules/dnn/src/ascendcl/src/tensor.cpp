// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

Tensor::Tensor(const void* data, const std::vector<int>& shape, aclDataType data_type, aclFormat format) : data_type_(data_type), format_(format), size_in_byte_(0)
{
    reshape(data, shape, data_type, format);
}

void Tensor::toMat(Mat& m)
{
    ASCENDCL_CHECK_RET(aclrtMemcpy(m.data, size_in_byte_, data_, size_in_byte_, ACL_MEMCPY_DEVICE_TO_HOST));
}

void Tensor::reshape(const void* data, const std::vector<int>& shape, aclDataType data_type, aclFormat format, bool alloc)
{
    CV_Assert(shape.size() > 0);

    if (shape_.toInt32() != shape) shape_.setShape(shape);
    if (data_type_ != data_type) data_type_ = data_type;
    if (format_ != format) format_ = format;

    size_t new_size = shapeCount(shape_.toInt32()) * elementSize(data_type);
    if (alloc || new_size > size_in_byte_) alloc = true;
    size_in_byte_ = new_size;

    if (alloc)
    {
        ASCENDCL_CHECK_RET(aclrtMalloc(&data_, size_in_byte_, ACL_MEM_MALLOC_HUGE_FIRST));
        ASCENDCL_CHECK_RET(aclrtMemcpy(data_, size_in_byte_, data, size_in_byte_, ACL_MEMCPY_HOST_TO_DEVICE));
    }
    else if (data)
    {
        ASCENDCL_CHECK_RET(aclrtMemcpy(data_, size_in_byte_, data, size_in_byte_, ACL_MEMCPY_HOST_TO_DEVICE));
    }
}

void Tensor::empty_like(std::shared_ptr<Tensor> src, aclDataType dtype)
{
    shape_.setShape(src->getShape());
    data_type_ = dtype;
    format_ = src->getFormat();
    size_in_byte_ = shapeCount(shape_.toInt32()) * elementSize(data_type_);

    ASCENDCL_CHECK_RET(aclrtMalloc(&data_, size_in_byte_, ACL_MEM_MALLOC_HUGE_FIRST));
}

std::shared_ptr<aclTensorDesc> Tensor::createTensorDesc()
{
    std::shared_ptr<aclTensorDesc> tensor_desc(
        aclCreateTensorDesc(data_type_, shape_.v.size(), shape_.v.data(), format_),
        aclDestroyTensorDesc
    );
    CV_Assert(tensor_desc != nullptr);
    return tensor_desc;
}

std::shared_ptr<aclDataBuffer> Tensor::createDataBuffer()
{
    std::shared_ptr<aclDataBuffer> databuf(
        aclCreateDataBuffer(data_, size_in_byte_),
        aclDestroyDataBuffer
    );
    CV_Assert(databuf != nullptr);
    return databuf;
}

int Tensor::getShapeAt(int axis) const
{
    CV_Assert(axis >= 0);
    CV_Assert(axis < shape_.v.size());

    return (int)shape_.v[axis];
}

std::vector<int> Tensor::getShape() const
{
    return shape_.toInt32();
}

int Tensor::total() const
{
    return shapeCount(shape_.toInt32());
}

aclDataType Tensor::getDataType() const
{
    return data_type_;
}

aclFormat Tensor::getFormat() const
{
    return format_;
}

size_t Tensor::getSizeInByte() const
{
    return size_in_byte_;
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
