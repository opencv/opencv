// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OP_CONV2D_HPP
#define OPENCV_DNN_ASCENDCL_OP_CONV2D_HPP

#include "ascendcl.hpp"
#include "operator.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class Conv2D : public Operator
{
public:
    Conv2D(std::vector<int64_t> strides, std::vector<int64_t> pads,
           std::vector<int64_t> dilations, int groups,
           int offset_x = 0);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    // std::vector<int64_t> toInt64(const std::vector<int>& v);
    std::vector<int64_t> strides_;
    std::vector<int64_t> pads_;
    std::vector<int64_t> dilations_;
    int groups_;
    int offset_x_;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_OP_CONV2D_HPP
