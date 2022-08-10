// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#include "../../precomp.hpp"
#include "common.hpp"
#include "internal.hpp"
#include "../include/operator.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

Operator::Operator() {}

Operator::~Operator() {}

bool Operator::forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                       std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                       aclrtStream stream)
{
    // create tensor description for inputs
    std::vector<std::shared_ptr<aclTensorDesc>> input_desc;
    for (int i = 0; i < inputs.size(); i++)
        input_desc.push_back(inputs[i]->createTensorDesc());
    // create tensor description for outputs
    std::vector<std::shared_ptr<aclTensorDesc>> output_desc;
    for (int i = 0; i < outputs.size(); i++)
        output_desc.push_back(outputs[i]->createTensorDesc());

    // create data buffer for inputs
    std::vector<std::shared_ptr<aclDataBuffer>> input_databuf;
    for (int i = 0; i < inputs.size(); i++)
        input_databuf.push_back(inputs[i]->createDataBuffer());
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

String Operator::getName() const
{
    return op_name_;
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv
