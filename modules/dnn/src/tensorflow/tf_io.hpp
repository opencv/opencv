// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Declaration of various functions which are related to Tensorflow models reading.
*/

#ifndef __OPENCV_DNN_TF_IO_HPP__
#define __OPENCV_DNN_TF_IO_HPP__
#if HAVE_PROTOBUF

#include "graph.pb.h"

namespace cv {
namespace dnn {

// Read parameters from a file into a GraphDef proto message.
void ReadTFNetParamsFromBinaryFileOrDie(const char* param_file,
                                      tensorflow::GraphDef* param);

}
}

#endif
#endif
