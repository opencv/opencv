// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.

// Copyright (C) 2018, Intel Corporation, all rights reserved.
// Third party copyrights are property of their respective owners.

#include "../precomp.hpp"
#include "../net_impl.hpp"

#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/dnn/layer_reg.private.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <opencv2/core/utils/fp_control_utils.hpp>
#include <opencv2/core/utils/logger.defines.hpp>
#undef CV_LOG_STRIP_LEVEL
#define CV_LOG_STRIP_LEVEL CV_LOG_LEVEL_VERBOSE + 1
#include <opencv2/core/utils/logger.hpp>

#include <opencv2/core/utils/configuration.private.hpp>


#ifdef HAVE_PROTOBUF

#include <array>
#include <iostream>
#include <fstream>
#include <string>
#include <limits>
#include <algorithm>


#if defined _MSC_VER && _MSC_VER < 1910/*MSVS 2017*/
#pragma warning(push)
#pragma warning(disable: 4503)  // decorated name length exceeded, name was truncated
#endif

#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-override"
#endif
#include "opencv-onnx.pb.h"
#if defined(__GNUC__) && __GNUC__ >= 5
#pragma GCC diagnostic pop
#endif

#include "onnx_graph_simplifier.hpp"
#endif

namespace cv {
namespace dnn {
CV__DNN_INLINE_NS_BEGIN

#ifdef HAVE_PROTOBUF

// ENGINE_CLASSIC/ENGINE_AUTO have been removed. Resolve any engine request to a
// supported one (ENGINE_NEW or ENGINE_ORT), honoring the OPENCV_FORCE_DNN_ENGINE override.
static int resolveOnnxEngine(int engine)
{
    static const int engine_forced =
        (int)utils::getConfigurationParameterSizeT("OPENCV_FORCE_DNN_ENGINE", ENGINE_NEW);
    if (engine_forced == ENGINE_NEW || engine_forced == ENGINE_ORT)
        engine = engine_forced;
    if (engine != ENGINE_NEW && engine != ENGINE_ORT)
    {
        CV_LOG_WARNING(NULL, "DNN/ONNX: only ENGINE_NEW and ENGINE_ORT are supported; "
                             "ENGINE_CLASSIC/ENGINE_AUTO are deprecated, falling back to ENGINE_NEW.");
        engine = ENGINE_NEW;
    }
    return engine;
}

Net readNetFromONNX(const String& onnxFile, int engine)
{
    if (resolveOnnxEngine(engine) == ENGINE_ORT)
    {
#ifdef HAVE_ONNXRUNTIME
        Net net = readNetFromONNX2_ORT(onnxFile);
        if (net.empty())
            CV_Error(Error::StsError, "DNN/ONNX/ORT: failed to load model");
        if (!net.getImpl() || net.getImpl()->modelFileName.empty())
            CV_Error(Error::StsError, "DNN/ONNX/ORT: ONNX Runtime model metadata was not initialized");
        return net;
#else
        CV_LOG_WARNING(NULL, "DNN/ONNX/ORT: OpenCV was built without ONNX Runtime (WITH_ONNXRUNTIME=OFF). Falling back to ENGINE_NEW.");
#endif
    }
    return readNetFromONNX2(onnxFile);
}

Net readNetFromONNX(const char* buffer, size_t sizeBuffer, int engine)
{
    if (resolveOnnxEngine(engine) == ENGINE_ORT)
    {
#ifdef HAVE_ONNXRUNTIME
        CV_Error(Error::StsNotImplemented, "DNN/ONNX/ORT: loading from memory buffer is not supported");
#else
        CV_LOG_WARNING(NULL, "DNN/ONNX/ORT: OpenCV was built without ONNX Runtime (WITH_ONNXRUNTIME=OFF). Falling back to ENGINE_NEW.");
#endif
    }
    return readNetFromONNX2(buffer, sizeBuffer);
}

Net readNetFromONNX(const std::vector<uchar>& buffer, int engine)
{
    if (resolveOnnxEngine(engine) == ENGINE_ORT)
    {
#ifdef HAVE_ONNXRUNTIME
        CV_Error(Error::StsNotImplemented, "DNN/ONNX/ORT: loading from memory buffer is not supported");
#else
        CV_LOG_WARNING(NULL, "DNN/ONNX/ORT: OpenCV was built without ONNX Runtime (WITH_ONNXRUNTIME=OFF). Falling back to ENGINE_NEW.");
#endif
    }
    return readNetFromONNX2(buffer);
}

static int onnxDataTypeToCvDepth(int onnxType)
{
    switch (onnxType)
    {
        case opencv_onnx::TensorProto_DataType_FLOAT:    return CV_32F;
        case opencv_onnx::TensorProto_DataType_UINT8:    return CV_8U;
        case opencv_onnx::TensorProto_DataType_UINT16:   return CV_16U;
        case opencv_onnx::TensorProto_DataType_FLOAT16:  return CV_16F;
        case opencv_onnx::TensorProto_DataType_INT8:     return CV_8S;
        case opencv_onnx::TensorProto_DataType_INT16:    return CV_16S;
        case opencv_onnx::TensorProto_DataType_INT32:    return CV_32S;
        case opencv_onnx::TensorProto_DataType_INT64:    return CV_64S;
        case opencv_onnx::TensorProto_DataType_BOOL:     return CV_Bool;
        case opencv_onnx::TensorProto_DataType_DOUBLE:   return CV_64F;
        case opencv_onnx::TensorProto_DataType_BFLOAT16: return CV_16BF;
        default: return CV_32F;
    }
}

static void releaseONNXTensor(opencv_onnx::TensorProto& tensor_proto)
{
    if (!tensor_proto.raw_data().empty()) {
        delete tensor_proto.release_raw_data();
    }
}

Mat readTensorFromONNX(const String& path)
{
    std::fstream input(path.c_str(), std::ios::in | std::ios::binary);
    if (!input)
    {
        CV_Error(Error::StsBadArg, cv::format("Can't read ONNX file: %s", path.c_str()));
    }

    opencv_onnx::TensorProto tensor_proto = opencv_onnx::TensorProto();
    if (!tensor_proto.ParseFromIstream(&input))
    {
        CV_Error(Error::StsUnsupportedFormat, cv::format("Failed to parse ONNX data: %s", path.c_str()));
    }
    Mat mat = getMatFromTensor(tensor_proto, false);
    int dims = (int)tensor_proto.dims_size();
    if (dims > 0 && mat.total() == 0) {
        int cv_type = onnxDataTypeToCvDepth(tensor_proto.data_type());

        std::vector<int> sizes(dims);
        for (int i = 0; i < dims; ++i) sizes[i] = (int)tensor_proto.dims(i);
        mat.create(dims, sizes.data(), cv_type);
    }
    releaseONNXTensor(tensor_proto);
    return mat;
}

#else  // HAVE_PROTOBUF

#define DNN_PROTOBUF_UNSUPPORTED() CV_Error(Error::StsError, "DNN/ONNX: Build OpenCV with Protobuf to import ONNX models")

Net readNetFromONNX(const String&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromONNX(const char*, size_t) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Net readNetFromONNX(const std::vector<uchar>&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

Mat readTensorFromONNX(const String&) {
    DNN_PROTOBUF_UNSUPPORTED();
}

#endif  // HAVE_PROTOBUF

CV__DNN_INLINE_NS_END
}} // namespace
