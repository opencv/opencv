// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation

#ifndef OPENCV_GAPI_INFER_OV_HPP
#define OPENCV_GAPI_INFER_OV_HPP

#include <string>

#include <opencv2/gapi/util/any.hpp>
#include <opencv2/gapi/own/exports.hpp>  // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp>   // GKernelType[M], GBackend

namespace cv {
namespace gapi {

/**
 * @brief This namespace contains G-API OpenVINO 2.0 backend functions,
 * structures, and symbols.
 */
namespace ov {

GAPI_EXPORTS cv::gapi::GBackend backend();

namespace detail {

struct ParamDesc {
    std::string xml_path;
    std::string bin_path;
    std::string device;

    std::size_t num_in;
    std::size_t num_out;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
};

} // namespace detail

template<typename Net> class Params {
public:
    Params(const std::string &xml,
           const std::string &bin,
           const std::string &device)
        : desc{ xml
              , bin
              , device
              , std::tuple_size<typename Net::InArgs>::value
              , std::tuple_size<typename Net::OutArgs>::value
              , {}
              , {} } {
    }

    Params<Net>& cfgInputLayers(const std::vector<std::string> &input_names) {
        desc.input_names = input_names;
        return *this;
    }

    Params<Net>& cfgOutputLayers(const std::vector<std::string> &output_names) {
        desc.output_names = output_names;
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::ov::backend(); }
    std::string   tag()     const { return Net::tag(); }
    cv::util::any params()  const { return { desc }; }
    // END(G-API's network parametrization API)

protected:
    detail::ParamDesc desc;
};

} // namespace ov
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_OV_HPP
