// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#ifndef OPENCV_GAPI_PLAIDML_PLAIDML_HPP
#define OPENCV_GAPI_PLAIDML_PLAIDML_HPP

#include <string>
#include <opencv2/gapi/gcommon.hpp> // CompileArgTag

namespace cv
{
namespace gapi
{

/**
 * @brief This namespace contains G-API PlaidML backend functions,
 * structures, and symbols.
 */
namespace plaidml
{

/** \addtogroup gapi_compile_args
 * @{
 */
/**
 * @brief This structure represents the basic parameters for the experimental
 * PlaidML backend.
 */
struct config
{
    std::string dev_id; //!< Device ID. Refer to PlaidML documentation for details.
    std::string trg_id; //!< Target ID. Refer to PlaidML documentation for details.
};
/** @} gapi_compile_args */

} // namespace plaidml
} // namespace gapi

namespace detail
{
    template<> struct CompileArgTag<cv::gapi::plaidml::config>
    {
        static const char* tag() { return "gapi.plaidml.config"; }
    };
} // namespace detail

} // namespace cv

#endif // OPENCV_GAPI_PLAIDML_PLAIDML_HPP
