// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_INFER_BINDINGS_OV_HPP
#define OPENCV_GAPI_INFER_BINDINGS_OV_HPP

#include <opencv2/gapi/util/any.hpp>
#include "opencv2/gapi/own/exports.hpp" // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp>     // GKernelPackage
#include <opencv2/gapi/infer/ov.hpp>    // Params

#include <string>

namespace cv {
namespace gapi {
namespace ov {

// NB: Used by python wrapper
// This class can be marked as SIMPLE, because it's implemented as pimpl
class GAPI_EXPORTS_W_SIMPLE PyParams {
public:
    GAPI_WRAP
    PyParams() = default;

    GAPI_WRAP
    PyParams(const std::string &tag,
             const std::string &model_path,
             const std::string &bin_path,
             const std::string &device);

    GAPI_WRAP
    PyParams(const std::string &tag,
             const std::string &blob_path,
             const std::string &device);

    GAPI_WRAP
    PyParams& cfgPluginConfig(
            const std::map<std::string, std::string> &config);

    GAPI_WRAP
    PyParams& cfgInputTensorLayout(std::string tensor_layout);

    GAPI_WRAP
    PyParams& cfgInputTensorLayout(
            std::map<std::string, std::string> layout_map);

    GAPI_WRAP
    PyParams& cfgInputModelLayout(std::string tensor_layout);

    GAPI_WRAP
    PyParams& cfgInputModelLayout(
            std::map<std::string, std::string> layout_map);

    GAPI_WRAP
    PyParams& cfgOutputTensorLayout(std::string tensor_layout);

    GAPI_WRAP
    PyParams& cfgOutputTensorLayout(
            std::map<std::string, std::string> layout_map);

    GAPI_WRAP
    PyParams& cfgOutputModelLayout(std::string tensor_layout);

    GAPI_WRAP
    PyParams& cfgOutputModelLayout(
            std::map<std::string, std::string> layout_map);

    GAPI_WRAP
    PyParams& cfgOutputTensorPrecision(int precision);

    GAPI_WRAP
    PyParams& cfgOutputTensorPrecision(
            std::map<std::string, int> precision_map);

    GAPI_WRAP
    PyParams& cfgReshape(std::vector<size_t> new_shape);

    GAPI_WRAP
    PyParams& cfgReshape(
            std::map<std::string, std::vector<size_t>> new_shape_map);

    GAPI_WRAP
    PyParams& cfgNumRequests(const size_t nireq);

    GAPI_WRAP
    PyParams& cfgMean(std::vector<float> mean_values);

    GAPI_WRAP
    PyParams& cfgMean(
            std::map<std::string, std::vector<float>> mean_map);

    GAPI_WRAP
    PyParams& cfgScale(std::vector<float> scale_values);

    GAPI_WRAP
    PyParams& cfgScale(
            std::map<std::string, std::vector<float>> scale_map);

    GAPI_WRAP
    PyParams& cfgResize(int interpolation);

    GAPI_WRAP
    PyParams& cfgResize(std::map<std::string, int> interpolation);

    GBackend      backend() const;
    std::string   tag()     const;
    cv::util::any params()  const;

private:
    std::shared_ptr<Params<cv::gapi::Generic>> m_priv;
};

GAPI_EXPORTS_W PyParams params(const std::string &tag,
                               const std::string &model_path,
                               const std::string &weights,
                               const std::string &device);

GAPI_EXPORTS_W PyParams params(const std::string &tag,
                               const std::string &bin_path,
                               const std::string &device);
} // namespace ov
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_BINDINGS_OV_HPP
