// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#ifndef __OPENCV_DNN_TF_SIMPLIFIER_HPP__
#define __OPENCV_DNN_TF_SIMPLIFIER_HPP__

#include "../precomp.hpp"

#ifdef HAVE_PROTOBUF

#include "tf_io.hpp"

namespace cv { namespace dnn {
CV__DNN_INLINE_NS_BEGIN

void RemoveIdentityOps(tensorflow::GraphDef& net);

void simplifySubgraphs(tensorflow::GraphDef& net);

Mat getTensorContent(const tensorflow::TensorProto &tensor);

void releaseTensor(tensorflow::TensorProto* tensor);

void sortByExecutionOrder(tensorflow::GraphDef& net);

CV__DNN_INLINE_NS_END
}}  // namespace dnn, namespace cv

#endif  // HAVE_PROTOBUF
#endif  // __OPENCV_DNN_TF_SIMPLIFIER_HPP__
