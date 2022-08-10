// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_flatten.hpp"

namespace cv { namespace dnn { namespace ascendcl {

#ifdef HAVE_ASCENDCL

Flatten::Flatten(int axis)
    : axis_(axis)
{
    op_name_ = "Flatten";

    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    ASCENDCL_CHECK_RET(aclopSetAttrInt(attr_.get(), "axis", axis_));
}

bool Flatten::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                      std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                      aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

#endif // HAVE_ASCENDCL

}}} // namespace cv::dnn::ascendcl
