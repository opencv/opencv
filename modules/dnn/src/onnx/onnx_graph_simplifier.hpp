// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
#define __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void simplifySubgraphs(opencv_onnx::GraphProto& net);

template<typename T1, typename T2>
void convertInt64ToInt32(const T1& src, T2& dst, int size)
{
    for (int i = 0; i < size; i++)
    {
        dst[i] = saturate_cast<int32_t>(src[i]);
    }
}

Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto);

CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv

#endif  // __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
