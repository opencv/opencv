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
    kOpTypeConv,
    kOpTypePool,
    kOpTypeDWConv,
    kOpTypeLRN,
    kOpTypeConcat,
    kOpTypeSoftmax,
    kOpTypeReLU,
    kOpTypePriorBox,
    kOpTypePermute,
    kOpTypeNum
};
enum PaddingMode { kPaddingModeSame, kPaddingModeValid, kPaddingModeCaffe, kPaddingModeNum };
enum FusedActivationType { kNone, kRelu, kRelu1, kRelu6, kActivationNum };
typedef std::vector<int> Shape;

/* context APIs */
bool initPerThread();
void deinitPerThread();
bool isAvailable();

#endif // HAVE_VULKAN

}}} // namespace cv::dnn::vkcom

#include "tensor.hpp"
#include "buffer.hpp"
#include "op_base.hpp"
#include "op_concat.hpp"
#include "op_conv.hpp"
#include "op_lrn.hpp"
#include "op_softmax.hpp"
#include "op_relu.hpp"
#include "op_pool.hpp"
#include "op_prior_box.hpp"
#include "op_permute.hpp"

#endif // OPENCV_DNN_VKCOM_HPP
