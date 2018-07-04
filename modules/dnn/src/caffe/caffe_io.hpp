/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2013, OpenCV Foundation, all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

/*M///////////////////////////////////////////////////////////////////////////////////////
//COPYRIGHT
//
//All contributions by the University of California:
//Copyright (c) 2014, The Regents of the University of California (Regents)
//All rights reserved.
//
//All other contributions:
//Copyright (c) 2014, the respective contributors
//All rights reserved.
//
//Caffe uses a shared copyright model: each contributor holds copyright over
//their contributions to Caffe. The project versioning records all such
//contribution and copyright details. If a contributor wants to further mark
//their specific copyright on a particular contribution, they should indicate
//their copyright solely in the commit message of the change when it is
//committed.
//
//LICENSE
//
//Redistribution and use in source and binary forms, with or without
//modification, are permitted provided that the following conditions are met:
//
//1. Redistributions of source code must retain the above copyright notice, this
//   list of conditions and the following disclaimer.
//2. Redistributions in binary form must reproduce the above copyright notice,
//   this list of conditions and the following disclaimer in the documentation
//   and/or other materials provided with the distribution.
//
//THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
//ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
//WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
//DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
//ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
//(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
//LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
//ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
//(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
//SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
//CONTRIBUTION AGREEMENT
//
//By contributing to the BVLC/caffe repository through pull-request, comment,
//or otherwise, the contributor releases their content to the
//license and copyright terms herein.
//
//M*/

#ifndef __OPENCV_DNN_CAFFE_IO_HPP__
#define __OPENCV_DNN_CAFFE_IO_HPP__
#ifdef HAVE_PROTOBUF

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-caffe.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

namespace caffe { using namespace opencv_caffe; } // avoid massive renames from caffe proto package

namespace cv {
namespace dnn {

// Read parameters from a file into a NetParameter proto message.
void ReadNetParamsFromTextFileOrDie(const char* param_file,
                                    caffe::NetParameter* param);
void ReadNetParamsFromBinaryFileOrDie(const char* param_file,
                                      caffe::NetParameter* param);

// Read parameters from a memory buffer into a NetParammeter proto message.
void ReadNetParamsFromBinaryBufferOrDie(const char* data, size_t len,
                                        caffe::NetParameter* param);
void ReadNetParamsFromTextBufferOrDie(const char* data, size_t len,
                                      caffe::NetParameter* param);

// Utility functions used internally by Caffe and TensorFlow loaders
bool ReadProtoFromTextFile(const char* filename, ::google::protobuf::Message* proto);
bool ReadProtoFromBinaryFile(const char* filename, ::google::protobuf::Message* proto);
bool ReadProtoFromTextBuffer(const char* data, size_t len, ::google::protobuf::Message* proto);
bool ReadProtoFromBinaryBuffer(const char* data, size_t len, ::google::protobuf::Message* proto);

}
}
#endif
#endif
