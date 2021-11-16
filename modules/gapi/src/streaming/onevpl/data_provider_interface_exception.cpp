// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#include <vpl/mfxjpeg.h>
#endif // HAVE_ONEVPL

#include <errno.h>
#include <string.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
DataProviderSystemErrorException::DataProviderSystemErrorException(int error_code,
                                                                   const std::string& description) {
    reason = description + ", error code: " + std::to_string(error_code) + " - " + strerror(error_code);
}

DataProviderSystemErrorException::~DataProviderSystemErrorException() = default;

const char* DataProviderSystemErrorException::what() const noexcept {
    return reason.c_str();
}

DataProviderUnsupportedException::DataProviderUnsupportedException(const std::string& description) {
    reason = description;
}

DataProviderUnsupportedException::~DataProviderUnsupportedException() = default;

const char* DataProviderUnsupportedException::what() const noexcept {
    return reason.c_str();
}

DataProviderImplementationException::DataProviderImplementationException(const std::string& description) {
    reason = description;
}

DataProviderImplementationException::~DataProviderImplementationException() = default;

const char* DataProviderImplementationException::what() const noexcept {
    return reason.c_str();
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
