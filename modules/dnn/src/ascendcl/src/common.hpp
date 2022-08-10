// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_DNN_ASCENDCL_COMMON_HPP
#define OPENCV_DNN_ASCENDCL_COMMON_HPP

#ifdef HAVE_ASCENDCL
#include "acl/acl.h"
#include "acl/acl_op_compiler.h"
#endif
#include "opencv2/core/utils/logger.hpp"
#include "../include/ascendcl.hpp"

namespace cv
{
namespace dnn
{
namespace ascendcl
{

#ifdef HAVE_ASCENDCL

#define ASCENDCL_CHECK_RET(f) \
{ \
    if (f != ACL_SUCCESS) \
    { \
        CV_LOG_ERROR(NULL, "AscendCL check failed, result = " << (int)f); \
        CV_Error(Error::StsError, "AscendCL check failed"); \
    } \
}

#endif // HAVE_ASCENDCL

} // namespace ascendcl
} // namespace dnn
} // namespace cv

#endif // OPENCV_DNN_ASCENDCL_COMMON_HPP
