// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"
#endif // HAVE_ONEVPL

#include <errno.h>
#include <string.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

DataProviderException::DataProviderException(const std::string& descr) :
    reason(descr) {
}
DataProviderException::DataProviderException(std::string&& descr) :
    reason(std::move(descr)) {
}

const char* DataProviderException::what() const noexcept {
    return reason.c_str();
}

DataProviderSystemErrorException::DataProviderSystemErrorException(int error_code,
                                                                   const std::string& description) :
    DataProviderException(description + ", error code: " + std::to_string(error_code) + " - " + strerror(error_code)) {

}

DataProviderUnsupportedException::DataProviderUnsupportedException(const std::string& description) :
    DataProviderException(description) {
}

DataProviderImplementationException::DataProviderImplementationException(const std::string& description) :
    DataProviderException(description) {
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
