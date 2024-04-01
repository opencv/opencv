// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#include "backends/onnx/coreml_ep.hpp"
#include "logger.hpp"

#ifdef HAVE_ONNX
#include <onnxruntime_cxx_api.h>

#ifdef HAVE_ONNX_COREML
#include "../providers/coreml/coreml_provider_factory.h"

void cv::gimpl::onnx::addCoreMLExecutionProvider(Ort::SessionOptions *session_options,
                                                 const cv::gapi::onnx::ep::CoreML &coreml_ep) {
    uint32_t flags = 0u;
    if (coreml_ep.use_cpu_only) {
        flags |= COREML_FLAG_USE_CPU_ONLY;
    }

    if (coreml_ep.enable_on_subgraph) {
        flags |= COREML_FLAG_ENABLE_ON_SUBGRAPH;
    }

    if (coreml_ep.enable_only_ane) {
        flags |= COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE;
    }

    try {
        OrtSessionOptionsAppendExecutionProvider_CoreML(*session_options, flags);
    } catch (const std::exception &e) {
        std::stringstream ss;
        ss << "ONNX Backend: Failed to enable CoreML"
           << " Execution Provider: " << e.what();
        cv::util::throw_error(std::runtime_error(ss.str()));
    }
}

#else  // HAVE_ONNX_COREML

void cv::gimpl::onnx::addCoreMLExecutionProvider(Ort::SessionOptions*,
                                                 const cv::gapi::onnx::ep::CoreML&) {
     util::throw_error(std::runtime_error("G-API has been compiled with ONNXRT"
                                          " without CoreML support"));
}

#endif  // HAVE_ONNX_COREML
#endif  // HAVE_ONNX
