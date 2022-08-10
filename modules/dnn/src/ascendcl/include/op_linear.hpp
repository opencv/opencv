// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OP_LINEAR_HPP
#define OPENCV_DNN_ASCENDCL_OP_LINEAR_HPP

#include "ascendcl.hpp"
#include "operator.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class Identity : public Operator
{
public:
    Identity();
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
};

/*
 * Y = x1' * x2' + bias
 */
class MatMul : public Operator
{
public:
    MatMul(bool transpose_x1, bool transpose_x2);
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
private:
    bool transpose_x1_;
    bool transpose_x2_;
};

class Add : public Operator
{
public:
    Add();
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream) CV_OVERRIDE;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_OP_LINEAR_HPP
