// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_VKCOM_HPP
#define OPENCV_DNN_VKCOM_HPP

#include <vector>

namespace cv { namespace dnn { namespace vkcom {

#ifdef HAVE_VULKAN

enum Format{
    kFormatInvalid = -1,
    kFormatFp16,
    kFormatFp32,
    kFormatFp64,
    kFormatInt32,
    kFormatNum
};

enum OpType {
    kOpTypeNull = -1,
    kOpTypeConv,
    kOpTypeMatMul,
};

enum FusedActivationType {
    kFusedActivUnsupport = -1,
    kFusedActivNone = 0,
    kFusedActivRelu = 1,
    kFusedActivRelu6 = 2,
};
typedef std::vector<int> Shape;

bool isAvailable();

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#include "buffer.hpp"
#include "tensor.hpp"
#include "context.hpp"

// layer
#include "op_base.hpp"
#include "op_conv.hpp"
#include "op_matmul.hpp"
#include "op_naryeltwise.hpp"

#endif // OPENCV_DNN_VKCOM_HPP
