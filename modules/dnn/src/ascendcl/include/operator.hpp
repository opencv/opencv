// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_OPERATOR_HPP
#define OPENCV_DNN_ASCENDCL_OPERATOR_HPP

#include "../../precomp.hpp"
#include "ascendcl.hpp"
#include <memory> // for shared_ptr

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

class Operator
{
public:
    Operator();
    virtual ~Operator();
    virtual bool forward(std::vector<std::shared_ptr<ascendcl::Tensor>> inputs,
                         std::vector<std::shared_ptr<ascendcl::Tensor>> outputs,
                         aclrtStream stream);
    String getName() const;

protected:
    String op_name_;
    // attributes of operator
    std::shared_ptr<aclopAttr> attr_;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_OPERATOR_HPP
