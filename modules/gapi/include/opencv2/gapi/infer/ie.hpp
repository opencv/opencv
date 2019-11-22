// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation

#ifndef OPENCV_GAPI_INFER_IE_HPP
#define OPENCV_GAPI_INFER_IE_HPP

#include <unordered_map>
#include <string>
#include <array>
#include <tuple> // tuple, tuple_size

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/any.hpp>

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
// FIXME: introduce a new sub-namespace for NN?
namespace ie {

GAPI_EXPORTS cv::gapi::GBackend backend();

namespace detail {
    struct ParamDesc {
        std::string model_path;
        std::string weights_path;
        std::string device_id;

        // NB: Here order follows the `Net` API
        std::vector<std::string> input_names;
        std::vector<std::string> output_names;

        std::unordered_map<std::string, cv::Mat> const_inputs;

        // NB: nun_* may differ from topology's real input/output port numbers
        // (e.g. topology's partial execution)
        std::size_t num_in;  // How many inputs are defined in the operation
        std::size_t num_out; // How many outputs are defined in the operation
    };
} // namespace detail

// FIXME: this is probably a shared (reusable) thing
template<typename Net>
struct PortCfg {
    using In = std::array
        < std::string
        , std::tuple_size<typename Net::InArgs>::value >;
    using Out = std::array
        < std::string
        , std::tuple_size<typename Net::OutArgs>::value >;
};

template<typename Net> class Params {
public:
    Params(const std::string &model,
           const std::string &weights,
           const std::string &device)
        : desc{ model, weights, device, {}, {}, {}
              , std::tuple_size<typename Net::InArgs>::value
              , std::tuple_size<typename Net::OutArgs>::value
              } {
    };

    Params<Net>& cfgInputLayers(const typename PortCfg<Net>::In &ll) {
        desc.input_names.clear();
        desc.input_names.reserve(ll.size());
        std::copy(ll.begin(), ll.end(),
                  std::back_inserter(desc.input_names));
        return *this;
    }

    Params<Net>& cfgOutputLayers(const typename PortCfg<Net>::Out &ll) {
        desc.output_names.clear();
        desc.output_names.reserve(ll.size());
        std::copy(ll.begin(), ll.end(),
                  std::back_inserter(desc.output_names));
        return *this;
    }

    Params<Net>& constInput(const std::string &layer_name,
                            const cv::Mat &data) {
        desc.const_inputs[layer_name] = data;
        return *this;
    }

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::ie::backend();  }
    std::string   tag()     const { return Net::tag(); }
    cv::util::any params()  const { return { desc }; }
    // END(G-API's network parametrization API)

protected:
    detail::ParamDesc desc;
};

} // namespace ie
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_INFER_HPP
