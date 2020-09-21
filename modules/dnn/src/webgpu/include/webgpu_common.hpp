// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef OPENCV_DNN_WEBGPU_HPP
#define OPENCV_DNN_WEBGPU_HPP

#include <vector>

namespace cv { namespace dnn { namespace webgpu {
#ifdef HAVE_WEBGPU

enum Format
{
    wFormatInvalid = -1,
    wFormatFp16,
    wFormatFp32,
    wFormatFp64,
    wFormatInt32,
    wFormatNum
};

enum OpType
{
    wOpTypeConv,
    wOpTypePool,
    wOpTypeDWConv,
    wOpTypeLRN,
    wOpTypeConcat,
    wOpTypeSoftmax,
    wOpTypeReLU,
    wOpTypePriorBox,
    wOpTypePermute,
    wOpTypeNum
};

enum PaddingMode
{
    wPaddingModeSame,
    wPaddingModeValid,
    wPaddingModeCaffe,
    wPaddingModeNum
};

enum FusedActivationType
{
    wNone,
    wRelu,
    wRelu1,
    wRelu6,
    wActivationNum
};

typedef std::vector<int> Shape;
bool isAvailable();
#endif  // HAVE_WEBGPU

}}} //namespace cv::dnn::webgpu

#include "webgpu_buffer.hpp"
#include "webgpu_tensor.hpp"
#include "webgpu_op_base.hpp"
#include "webgpu_concat.hpp"
#include "webgpu_conv.hpp"
#include "webgpu_lrn.hpp"
#include "webgpu_softmax.hpp"
#include "webgpu_relu.hpp"
#include "webgpu_pool.hpp"
#include "webgpu_prior_box.hpp"
#include "webgpu_permute.hpp"

#endif//    OPENCV_DNN_WEBGPU_HPP