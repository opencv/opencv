// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2016, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

/*
Implementation of various functions which are related to Tensorflow models reading.
*/

#if HAVE_PROTOBUF
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <opencv2/core.hpp>

#include <map>
#include <string>
#include <fstream>
#include <vector>

#include "graph.pb.h"
#include "tf_io.hpp"
#include "../caffe/glog_emulator.hpp"

namespace cv {
namespace dnn {

using std::string;
using std::map;
using namespace tensorflow;
using namespace ::google::protobuf;
using namespace ::google::protobuf::io;

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

// TODO: remove Caffe duplicate
bool ReadProtoFromBinaryFileTF(const char* filename, Message* proto) {
    std::ifstream fs(filename, std::ifstream::in | std::ifstream::binary);
    CHECK(fs.is_open()) << "Can't open \"" << filename << "\"";
    ZeroCopyInputStream* raw_input = new IstreamInputStream(&fs);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

    bool success = proto->ParseFromCodedStream(coded_input);

    delete coded_input;
    delete raw_input;
    fs.close();
    return success;
}

void ReadTFNetParamsFromBinaryFileOrDie(const char* param_file,
                                      tensorflow::GraphDef* param) {
  CHECK(ReadProtoFromBinaryFileTF(param_file, param))
      << "Failed to parse GraphDef file: " << param_file;
}

}
}
#endif
