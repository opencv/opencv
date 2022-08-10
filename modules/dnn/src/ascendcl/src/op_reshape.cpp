// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_reshape.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

Reshape::Reshape(int axis, int num_axes)
    : axis_(axis), num_axes_(num_axes)
{
    op_name_ = "Reshape";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set depth_radius
    ASCENDCL_CHECK_RET(aclopSetAttrInt(attr_.get(), "axis", axis_));
    //  * set norm_region
    ASCENDCL_CHECK_RET(aclopSetAttrInt(attr_.get(), "num_axes", num_axes_));
}

bool Reshape::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                      std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                      aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}


#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
