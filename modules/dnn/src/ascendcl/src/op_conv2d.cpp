// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_conv2d.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

Conv2D::Conv2D(std::vector<int64_t> strides, std::vector<int64_t> pads,
               std::vector<int64_t> dilations, int groups,
               int offset_x)
    : strides_(strides), pads_(pads), dilations_(dilations), groups_(groups), offset_x_(offset_x)
{
    op_name_ = "Conv2D";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set strides
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "strides", strides_.size(), strides_.data()));
    //  * set pads
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "pads", pads_.size(), pads_.data()));
    //  * set dilations
    ASCENDCL_CHECK_RET(aclopSetAttrListInt(attr_.get(), "dilations", dilations_.size(), dilations_.data()));
    //  * set groups
    ASCENDCL_CHECK_RET(aclopSetAttrInt(attr_.get(), "groups", groups));
    //  * set offset_x
    ASCENDCL_CHECK_RET(aclopSetAttrInt(attr_.get(), "offset_x", offset_x_));
}


// steps:
//  1. create tensor description for inputs and outputs
//  2. create data buffer for inputs and outputs
//  3. forward
bool Conv2D::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                     std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                     aclrtStream stream)
{
    // create tensor description for inputs
    std::vector<std::shared_ptr<aclTensorDesc>> input_desc;
    for (int i = 0; i < inputs.size(); i++)
        input_desc.push_back(inputs[i]->createTensorDesc());
    if (input_desc.size() < 3) // no bias
    {
        std::shared_ptr<aclTensorDesc> bias_desc(aclCreateTensorDesc(ACL_DT_UNDEFINED, 0,
                                                                      nullptr, ACL_FORMAT_UNDEFINED),
                                                  aclDestroyTensorDesc);
        CV_Assert(bias_desc != nullptr);
        input_desc.push_back(bias_desc);
    }
    // create tensor description for outputs
    std::vector<std::shared_ptr<aclTensorDesc>> output_desc;
    for (int i = 0; i < outputs.size(); i++)
        output_desc.push_back(outputs[i]->createTensorDesc());

    // create data buffer for inputs
    std::vector<std::shared_ptr<aclDataBuffer>> input_databuf;
    for (int i = 0; i < inputs.size(); i++)
        input_databuf.push_back(inputs[i]->createDataBuffer());
    if (input_databuf.size() < 3) // no bias
    {
        std::shared_ptr<aclDataBuffer> bias_databuf(aclCreateDataBuffer(nullptr, 0),
                                                     aclDestroyDataBuffer);
        CV_Assert(bias_databuf != nullptr);
        input_databuf.push_back(bias_databuf);
    }
    // create data buffer for outputs
    std::vector<std::shared_ptr<aclDataBuffer>> output_databuf;
    for (int i = 0; i < outputs.size(); i++)
        output_databuf.push_back(outputs[i]->createDataBuffer());

    // convertions
    std::vector<aclTensorDesc*> raw_input_desc;
    for (auto desc : input_desc)
        raw_input_desc.push_back(desc.get());
    std::vector<aclTensorDesc*> raw_output_desc;
    for (auto desc : output_desc)
        raw_output_desc.push_back(desc.get());
    std::vector<aclDataBuffer*> raw_input_databuf;
    for (auto buf : input_databuf)
        raw_input_databuf.push_back(buf.get());
    std::vector<aclDataBuffer*> raw_output_databuf;
    for (auto buf : output_databuf)
        raw_output_databuf.push_back(buf.get());
    ASCENDCL_CHECK_RET(aclopCompileAndExecute(op_name_.c_str(),
                       raw_input_desc.size(), raw_input_desc.data(), raw_input_databuf.data(),
                       raw_output_desc.size(), raw_output_desc.data(), raw_output_databuf.data(),
                       attr_.get(), ACL_ENGINE_SYS, ACL_COMPILE_SYS, NULL, stream));

    return true;
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
