// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../../precomp.hpp"
#include "internal.hpp"
#include "../include/op_base.hpp"

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

OpBase::OpBase()
{
}

OpBase::~OpBase()
{
}

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom
