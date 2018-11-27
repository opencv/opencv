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
#ifdef HAVE_PROTOBUF

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "graph.pb.h"

#include <google/protobuf/message.h>
#include <google/protobuf/text_format.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

namespace tensorflow { using namespace opencv_tensorflow; }

namespace cv {
namespace dnn {

// Read parameters from a file into a GraphDef proto message.
void ReadTFNetParamsFromBinaryFileOrDie(const char* param_file,
                                      tensorflow::GraphDef* param);

void ReadTFNetParamsFromTextFileOrDie(const char* param_file,
                                      tensorflow::GraphDef* param);

// Read parameters from a memory buffer into a GraphDef proto message.
void ReadTFNetParamsFromBinaryBufferOrDie(const char* data, size_t len,
                                          tensorflow::GraphDef* param);

void ReadTFNetParamsFromTextBufferOrDie(const char* data, size_t len,
                                        tensorflow::GraphDef* param);

}
}

#endif
#endif
