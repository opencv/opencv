// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_INTERNAL_HPP
#define OPENCV_DNN_ASCENDCL_INTERNAL_HPP

#include "../include/ascendcl.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

inline size_t elementSize(aclDataType dtype)
{
    if (dtype == ACL_FLOAT)
    {
        return 4;
    }
    else if (dtype == ACL_FLOAT16)
    {
        return 2;
    }
    else
    {
        CV_Error(Error::StsError, format("Unsupported format %d", dtype));
    }
    return 0;
}

inline int shapeCount(const std::vector<int>& shape, int start = -1, int end = -1)
{
    if (start == -1) start = 0;
    if (end == -1) end = (int)shape.size();

    if (shape.empty())
        return 0;

    int elems = 1;
    assert(start <= (int)shape.size() &&
           end <= (int)shape.size() &&
           start <= end);
    for(int i = start; i < end; i++)
    {
        elems *= shape[i];
    }
    return elems;
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_INTERNAL_HPP
