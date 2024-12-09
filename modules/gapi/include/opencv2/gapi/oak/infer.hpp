// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation

#ifndef OPENCV_GAPI_OAK_INFER_HPP
#define OPENCV_GAPI_OAK_INFER_HPP

#include <unordered_map>
#include <string>
#include <array>
#include <tuple>

#include <opencv2/gapi/opencv_includes.hpp>
#include <opencv2/gapi/util/any.hpp>

#include <opencv2/core/cvdef.h>     // GAPI_EXPORTS
#include <opencv2/gapi/gkernel.hpp> // GKernelPackage

namespace cv {
namespace gapi {
namespace oak {

namespace detail {
/**
* @brief This structure contains description of inference parameters
* which is specific to OAK models.
*/
struct ParamDesc {
    std::string blob_file;
};
} // namespace detail

/**
 * Contains description of inference parameters and kit of functions that
 * fill this parameters.
 */
template<typename Net> class Params {
public:
    /** @brief Class constructor.

    Constructs Params based on model information and sets default values for other
    inference description parameters.

    @param model Path to model (.blob file)
    */
    explicit Params(const std::string &model) {
        desc.blob_file = model;
    };

    // BEGIN(G-API's network parametrization API)
    GBackend      backend() const { return cv::gapi::oak::backend(); }
    std::string   tag()     const { return Net::tag(); }
    cv::util::any params()  const { return { desc }; }
    // END(G-API's network parametrization API)

protected:
    detail::ParamDesc desc;
};

} // namespace oak
} // namespace gapi
} // namespace cv

#endif // OPENCV_GAPI_OAK_INFER_HPP
