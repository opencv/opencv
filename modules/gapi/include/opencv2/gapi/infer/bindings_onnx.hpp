// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#ifndef OPENCV_GAPI_INFER_BINDINGS_ONNX_HPP
#define OPENCV_GAPI_INFER_BINDINGS_ONNX_HPP

#include <opencv2/gapi/gkernel.hpp>     // GKernelPackage
#include <opencv2/gapi/infer/onnx.hpp>  // Params
#include "opencv2/gapi/own/exports.hpp"  // GAPI_EXPORTS
#include <opencv2/gapi/util/any.hpp>

#include <string>

namespace cv {
namespace gapi {
namespace onnx {

// NB: Used by python wrapper
// This class can be marked as SIMPLE, because it's implemented as pimpl
class GAPI_EXPORTS_W_SIMPLE PyParams {
public:
    GAPI_WRAP
    PyParams() = default;

    GAPI_WRAP
    PyParams(const std::string& tag, const std::string& model_path);

    GAPI_WRAP
    PyParams& cfgMeanStd(const std::string &layer_name,
                         const cv::Scalar &m,
                         const cv::Scalar &s);
    GAPI_WRAP
    PyParams& cfgNormalize(const std::string &layer_name, bool flag);

    GAPI_WRAP
    PyParams& cfgAddExecutionProvider(ep::OpenVINO ep);

    GAPI_WRAP
    PyParams& cfgAddExecutionProvider(ep::DirectML ep);

    GAPI_WRAP
    PyParams& cfgAddExecutionProvider(ep::CoreML ep);

    GAPI_WRAP
    PyParams& cfgAddExecutionProvider(ep::CUDA ep);

    GAPI_WRAP
    PyParams& cfgAddExecutionProvider(ep::TensorRT ep);

    GAPI_WRAP
    PyParams& cfgDisableMemPattern();

    GAPI_WRAP
    PyParams& cfgSessionOptions(const std::map<std::string, std::string>& options);

    GBackend backend() const;
    std::string tag() const;
    cv::util::any params() const;

private:
    std::shared_ptr<Params<cv::gapi::Generic>> m_priv;
};

GAPI_EXPORTS_W PyParams params(const std::string& tag, const std::string& model_path);

}  // namespace onnx
}  // namespace gapi
}  // namespace cv

#endif  // OPENCV_GAPI_INFER_BINDINGS_ONNX_HPP
