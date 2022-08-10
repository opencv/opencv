// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_TENSOR_HPP
#define OPENCV_DNN_ASCENDCL_TENSOR_HPP

#ifdef HAVE_ASCENDCL
#include "acl/acl.h"
#endif // HAVE_ASCENDCL
#include <vector>
#include "ascendcl.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class Tensor
{
public:
    Tensor() : data_(nullptr), data_type_(ACL_DT_UNDEFINED), format_(ACL_FORMAT_UNDEFINED), size_in_byte_(0) { }
    ~Tensor()
    {
        if (data_ != nullptr)
        {
            aclError ret = aclrtFree(data_);
            CV_Assert(ret == ACL_SUCCESS);
            data_ = nullptr;
        }
    }

    Tensor(const void* data, const std::vector<int>& shape, aclDataType data_type = ACL_FLOAT, aclFormat format = ACL_FORMAT_NCHW);

    void toMat(Mat& m);
    void reshape(const void* data, const std::vector<int>& shape, aclDataType data_type = ACL_FLOAT, aclFormat format = ACL_FORMAT_NCHW, bool alloc = false);

    void empty_like(std::shared_ptr<Tensor> src, aclDataType dtype);

    std::shared_ptr<aclTensorDesc> createTensorDesc();
    std::shared_ptr<aclDataBuffer> createDataBuffer();

    std::vector<int> getShape() const;
    int getShapeAt(int axis) const;
    int total() const;

    aclDataType getDataType() const;
    aclFormat getFormat() const;
    size_t getSizeInByte() const;

private:
    void* data_;
    ShapeInt64 shape_; // typedef std::vector<int64_t> Shape;
    aclDataType data_type_;
    aclFormat format_;
    size_t size_in_byte_;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_TENSOR_HPP
