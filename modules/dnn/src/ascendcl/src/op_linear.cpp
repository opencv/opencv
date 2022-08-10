// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/op_linear.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

Identity::Identity()
{
    op_name_ = "Identity";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
}

bool Identity::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                       std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                       aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

MatMul::MatMul(bool transpose_x1, bool transpose_x2)
    : transpose_x1_(transpose_x1), transpose_x2_(transpose_x2)
{
    op_name_ = "MatMulV2";

    // create aclopAttr
    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
    //  * set transpose_a
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "transpose_x1", transpose_x1_));
    //  * set transpose_b
    ASCENDCL_CHECK_RET(aclopSetAttrBool(attr_.get(), "transpose_x2", transpose_x2_));
}

// inputs: [x1, x2, bias]
/*
 * Y = x1' * x2' + bias
 */
bool MatMul::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                     std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                     aclrtStream stream)
{
    // create tensor description for inputs
    std::vector<std::shared_ptr<aclTensorDesc>> input_desc;
    for (int i = 0; i < inputs.size(); i++)
        input_desc.push_back(inputs[i]->createTensorDesc());
    if (input_desc.size() < 3) // no bias
    {
        std::shared_ptr<aclTensorDesc> bias_desc(
            aclCreateTensorDesc(ACL_DT_UNDEFINED, 0, nullptr, ACL_FORMAT_UNDEFINED),
            aclDestroyTensorDesc
        );
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
        std::shared_ptr<aclDataBuffer> bias_databuf(
            aclCreateDataBuffer(nullptr, 0),
            aclDestroyDataBuffer
        );
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

Add::Add()
{
    op_name_ = "AddV2";

    attr_.reset(aclopCreateAttr(), aclopDestroyAttr);
}

bool Add::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                  std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                  aclrtStream stream)
{
    return Operator::forward(inputs, outputs, stream);
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
