// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "backends/onnx/dml_ep.hpp"
#include "logger.hpp"

#ifdef HAVE_ONNX
#include <onnxruntime_cxx_api.h>

#ifdef HAVE_ONNX_DML
#include "../providers/dml/dml_provider_factory.h"

void cv::gimpl::onnx::addDMLExecutionProvider(Ort::SessionOptions *session_options,
                                              const cv::gapi::onnx::ep::DirectML &dml_ep) {
    namespace ep = cv::gapi::onnx::ep;
    GAPI_Assert(cv::util::holds_alternative<int>(dml_ep.ddesc));
    const int device_id = cv::util::get<int>(dml_ep.ddesc);
    try {
        OrtSessionOptionsAppendExecutionProvider_DML(*session_options, device_id);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "ONNX Backend: Failed to enable DirectML"
           << " Execution Provider: " << e.what();
        cv::util::throw_error(std::runtime_error(ss.str()));
    }
}

#else  // HAVE_ONNX_DML

void cv::gimpl::onnx::addDMLExecutionProvider(Ort::SessionOptions*,
                                              const cv::gapi::onnx::ep::DirectML&) {
     util::throw_error(std::runtime_error("G-API has been compiled with ONNXRT"
                                          " without DirectML support"));
}

#endif  // HAVE_ONNX_DML
#endif  // HAVE_ONNX
