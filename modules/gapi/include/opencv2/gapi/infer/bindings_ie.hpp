// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#ifndef OPENCV_GAPI_INFER_BINDINGS_IE_HPP
#define OPENCV_GAPI_INFER_BINDINGS_IE_HPP

#include <opencv2/gapi/util/any.hpp>
#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp>     // GKernelPackage
#include <opencv2/gapi/infer/ie.hpp>    // Params

#include <string>

namespace cv {
namespace gapi {
namespace ie {

// NB: Used by python wrapper
// This class can be marked as SIMPLE, because it's implemented as pimpl
class GAPI_EXPORTS_W_SIMPLE PyParams {
public:
    PyParams() = default;

    PyParams(const std::string &tag,
             const std::string &model,
             const std::string &weights,
             const std::string &device);

    PyParams(const std::string &tag,
             const std::string &model,
             const std::string &device);

    GBackend      backend() const;
    std::string   tag()     const;
    cv::util::any params()  const;

private:
    std::shared_ptr<Params<cv::gapi::Generic>> m_priv;
};

GAPI_EXPORTS_W PyParams params(const std::string &tag,
                               const std::string &model,
                               const std::string &weights,
                               const std::string &device);

GAPI_EXPORTS_W PyParams params(const std::string &tag,
                               const std::string &model,
                               const std::string &device);
} // namespace ie
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_BINDINGS_IE_HPP
