// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_norm.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

LRN::LRN(int depth_radius, float bias, float alpha, float beta, String norm_region)
    : depth_radius_(depth_radius), bias_(bias), alpha_(alpha), beta_(beta), norm_region_(norm_region)
{
    CV_Assert(bias > 0);
    CV_Assert(norm_region_ == "ACROSS_CHANNELS" || norm_region_ == "WITHIN_CHANNELv");
    op_name_ = "LRN";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set depth_radius
    ASCENDCL_CHECK_RET(aclopSetAttrInt(attr_.get(), "depth_radius", depth_radius_));
    //  * set bias
    ASCENDCL_CHECK_RET(aclopSetAttrFloat(attr_.get(), "bias", bias_));
    //  * set alpha
    ASCENDCL_CHECK_RET(aclopSetAttrFloat(attr_.get(), "alpha", alpha_));
    //  * set beta
    ASCENDCL_CHECK_RET(aclopSetAttrFloat(attr_.get(), "beta", beta_));
    //  * set norm_region
    ASCENDCL_CHECK_RET(aclopSetAttrString(attr_.get(), "norm_region", norm_region_.c_str()));
}

bool LRN::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                  std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                  aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

BatchNorm::BatchNorm(float epsilon)
    : epsilon_(epsilon), data_format_("NHWC"), is_training_(false)
{
    op_name_ = "BatchNorm";

    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    ASCENDCL_CHECK_RET(aclopSetAttrFloat(attr_.get(), "epsilon", epsilon));
    ASCENDCL_CHECK_RET(aclopSetAttrString(attr_.get(), "data_format", data_format_.c_str()));
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "is_training", is_training_));
}

bool BatchNorm::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                        std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                        aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
