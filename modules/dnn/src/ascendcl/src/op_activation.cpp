// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_activation.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

ReLU::ReLU()
{
    op_name_ = "Relu";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
}

bool ReLU::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                   std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                   aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

ReLU6::ReLU6()
{
    op_name_ = "Relu6";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
}

bool ReLU6::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                    std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                    aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

Tanh::Tanh()
{
    op_name_ = "Tanh";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
}

bool Tanh::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                   std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                   aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

Softmax::Softmax(std::vector<int64_t> axes)
    : axes_(axes)
{
    op_name_ = "SoftmaxV2";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set axes
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "axes", axes_.size(), axes_.data()));
}

bool Softmax::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                      std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                      aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

LogSoftmax::LogSoftmax(std::vector<int64_t> axes)
    : axes_(axes)
{
    op_name_ = "LogSoftmaxV2";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set axes
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "axes", axes_.size(), axes_.data()));
}

bool LogSoftmax::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}


LeakyRelu::LeakyRelu(float negative_slope)
    : negative_slope_(negative_slope)
{
    op_name_ = "LeakyRelu";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set axes
    ASCENDCL_CHECK_RET(aclopSetAttrFloat(attr_.get(), "negative_slope", negative_slope_));
}

bool LeakyRelu::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
