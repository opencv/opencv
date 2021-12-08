// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_CFG_PARAM_PARSER_HPP
#define GAPI_STREAMING_ONEVPL_CFG_PARAM_PARSER_HPP

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

#include <map>
#include <string>

#include <opencv2/gapi/streaming/onevpl/source.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

template<typename ValueType>
std::vector<ValueType> get_params_from_string(const std::string& str);

template <typename ReturnType>
struct ParamCreator {
    template<typename ValueType>
    ReturnType create(const std::string& name, ValueType&& value, bool is_major = false);
};

mfxVariant cfg_param_to_mfx_variant(const CfgParam& value);

size_t strtoull_or_throw(const char* str);
int64_t strtoll_or_throw(const char* str);

} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_CFG_PARAM_PARSER_HPP
