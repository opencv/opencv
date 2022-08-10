// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OP_ACTIVATION_HPP
#define OPENCV_DNN_ASCENDCL_OP_ACTIVATION_HPP

#include "ascendcl.hpp"
#include "operator.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class ReLU : public Operator
{
public:
    ReLU();
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
};

class ReLU6 : public Operator
{
public:
    ReLU6();
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
};

class Tanh : public Operator
{
public:
    Tanh();
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
};

class Softmax : public Operator
{
public:
    Softmax(std::vector<int64_t> axes);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    std::vector<int64_t> axes_;
};

class LogSoftmax : public Operator
{
public:
    LogSoftmax(std::vector<int64_t> axes);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    std::vector<int64_t> axes_;
};

class LeakyRelu : public Operator
{
public:
    LeakyRelu(float negative_slope = 0.0f);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    float negative_slope_;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv


#endif // OPENCV_DNN_ASCENDCL_OP_ACTIVATION_HPP
