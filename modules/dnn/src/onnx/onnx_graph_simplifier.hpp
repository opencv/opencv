// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2020, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
#define __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
#ifdef HAVE_PROTOBUF

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

/** @brief converts tensor to Mat, preserving the tensor data type
 *  @param uint8ToInt8 if true, handles uint8 tensor as quantized weight. So output Mat = int8(int32(uint8_tensor) - 128)).
 *  if false, just returns uint8 Mat.
*/
Mat getMatFromTensor(const opencv_onnx::TensorProto& tensor_proto, bool uint8ToInt8=true, const std::string base_path = "");

CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv

#endif  // HAVE_PROTOBUF
#endif  // __OPENCV_DNN_ONNX_SIMPLIFIER_HPP__
