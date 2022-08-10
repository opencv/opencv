// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_HPP
#define OPENCV_DNN_ASCENDCL_HPP

#include "../../precomp.hpp"
#ifdef HAVE_ASCENDCL
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#endif // HAVE_ASCENDCL
#include <vector>
#include <algorithm> // for std::transform

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

//typedef std::vector<int64_t> Shape;
struct ShapeInt64 {
    ShapeInt64() {}
    ShapeInt64(const std::vector<int>& _v)
    {
        std::transform(_v.begin(), _v.end(), std::back_inserter(v), [] (int s) { return (int64_t)s; });
    }
    void setShape(const std::vector<int>& _v)
    {
        v.clear();

        // FIXIT: when taking a 2D shape and the last dimension is 1, treat it as 1D shape. Fix this behavior when 1D Mat is supported.
        if (_v.size() == 2 and _v[1] == 1)
        {
            v.push_back((int64_t)_v[0]);
            return;
        }

        std::transform(_v.begin(), _v.end(), std::back_inserter(v), [] (int s) { return (int64_t)s; });
    }
    std::vector<int> toInt32() const
    {
        std::vector<int> _v;
        std::transform(v.begin(), v.end(), std::back_inserter(_v), [] (int s) { return (int)s; });
        return _v;
    }

    std::vector<int64_t> v;
};

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namepsace dnn
} // namespace cv

#include "tensor.hpp"
#include "operator.hpp"
#include "op_conv2d.hpp"
#include "op_activation.hpp"
#include "op_norm.hpp"
#include "op_pooling.hpp"
#include "op_linear.hpp"
#include "op_reshape.hpp"
#include "op_cast.hpp"
#include "op_transdata.hpp"

#endif // OPENCV_DNN_ASCENDCL_HPP
