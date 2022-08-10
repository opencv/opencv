// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OP_POOLING_HPP
#define OPENCV_DNN_ASCENDCL_OP_POOLING_HPP

#include "ascendcl.hpp"
#include "operator.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class MaxPool : public Operator
{
public:
    MaxPool(std::vector<int64_t> ksize, std::vector<int64_t> strides, String padding_mode,
            std::vector<int64_t> pads, bool global_pooling, bool ceil_mode);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    std::vector<int64_t> ksize_;
    std::vector<int64_t> strides_;
    String padding_mode_;
    std::vector<int64_t> pads_;
    String data_format_;
    bool global_pooling_;
    bool ceil_mode_;
};

class AvgPool : public Operator
{
public:
    AvgPool(std::vector<int64_t> ksize, std::vector<int64_t> strides, String padding_mode,
            std::vector<int64_t> pads, bool global_pooling, bool ceil_mode);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    std::vector<int64_t> ksize_;
    std::vector<int64_t> strides_;
    String padding_mode_;
    std::vector<int64_t> pads_;
    String data_format_;
    bool global_pooling_;
    bool ceil_mode_;
    bool exclusive_;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_OP_POOLING_HPP
