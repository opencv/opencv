// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_pooling.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

MaxPool::MaxPool(std::vector<int64_t> ksize, std::vector<int64_t> strides, String padding_mode,
                     std::vector<int64_t> pads, bool global_pooling, bool ceil_mode)
    : ksize_(ksize), strides_(strides), padding_mode_(padding_mode), pads_(pads), data_format_("NCHW"), global_pooling_(global_pooling), ceil_mode_(ceil_mode)
{
    CV_Assert(ksize_.size() == 4);
    CV_Assert(strides_.size() == 4);
    CV_Assert(padding_mode_ == "SAME" || padding_mode_ == "VALID" || padding_mode_ == "CALCULATED");
    CV_Assert(pads_.size() == 4);
    op_name_ = "MaxPoolV3";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set ksize
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "ksize", 4, ksize_.data()));
    //  * set strides
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "strides", 4, strides_.data()));
    //  * set padding_mode
    ASCENDCL_CHECK_RET(aclopSetAttrString(attr_.get(), "padding_mode", padding_mode_.c_str()));
    //  * set pads; valid only when padding_mode == "CALCULATED"
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "pads", 4, pads_.data()));
    //  * set data_format
    ASCENDCL_CHECK_RET(aclopSetAttrString(attr_.get(), "data_format", data_format_.c_str()));
    //  * set global_pooling
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "global_pooling", global_pooling_));
    //  * set ceil_mode
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "ceil_mode", ceil_mode_));
}

bool MaxPool::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                      std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                      aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

AvgPool::AvgPool(std::vector<int64_t> ksize, std::vector<int64_t> strides, String padding_mode,
                     std::vector<int64_t> pads, bool global_pooling, bool ceil_mode)
    : ksize_(ksize), strides_(strides), padding_mode_(padding_mode), pads_(pads), data_format_("NCHW"), global_pooling_(global_pooling), ceil_mode_(ceil_mode), exclusive_(true)
{
    CV_Assert(ksize_.size() == 4);
    CV_Assert(strides_.size() == 4);
    CV_Assert(padding_mode_ == "SAME" || padding_mode_ == "VALID" || padding_mode_ == "CALCULATED");
    CV_Assert(pads_.size() == 4);
    op_name_ = "AvgPoolV2";

    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "ksize", 4, ksize_.data()));
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "strides", 4, strides_.data()));
    ASCENDCL_CHECK_RET(aclopSetAttrString(attr_.get(), "padding_mode", padding_mode_.c_str()));
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "pads", 4, pads_.data()));
    ASCENDCL_CHECK_RET(aclopSetAttrString(attr_.get(), "data_format", data_format_.c_str()));
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "global_pooling", global_pooling_));
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "ceil_mode", ceil_mode_));
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "exclusive", exclusive_));
}

bool AvgPool::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                      std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                      aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
