// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level
// directory of this distribution and at http://opencv.org/license.html.

#include <opencv2/gapi/infer/bindings_onnx.hpp>

cv::gapi::onnx::PyParams::PyParams(const std::string& tag,
                                   const std::string& model_path)
    : m_priv(std::make_shared<Params<cv::gapi::Generic>>(tag, model_path)) {}

cv::gapi::onnx::PyParams& cv::gapi::onnx::PyParams::cfgMeanStd(const std::string &layer_name,
                                                               const cv::Scalar &m,
                                                               const cv::Scalar &s) {
    m_priv->cfgMeanStdDev(layer_name, m, s);
    return *this;
}

cv::gapi::onnx::PyParams& cv::gapi::onnx::PyParams::cfgNormalize(const std::string &layer_name,
                                                                 bool flag) {
    m_priv->cfgNormalize(layer_name, flag);
    return *this;
}

cv::gapi::GBackend cv::gapi::onnx::PyParams::backend() const {
    return m_priv->backend();
}

std::string cv::gapi::onnx::PyParams::tag() const { return m_priv->tag(); }

cv::util::any cv::gapi::onnx::PyParams::params() const {
    return m_priv->params();
}

cv::gapi::onnx::PyParams cv::gapi::onnx::params(
    const std::string& tag, const std::string& model_path) {
    return {tag, model_path};
}
