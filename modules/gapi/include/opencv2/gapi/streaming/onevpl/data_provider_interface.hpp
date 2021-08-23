// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#include <exception>
#include <string>

#include <opencv2/gapi/own/exports.hpp> // GAPI_EXPORTS

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

struct GAPI_EXPORTS DataProviderException : public std::exception {
    virtual ~DataProviderException() {};
};

struct GAPI_EXPORTS DataProviderSystemErrorException : public DataProviderException {
    DataProviderSystemErrorException(int error_code, const std::string& desription = std::string());
    virtual ~DataProviderSystemErrorException();
    virtual const char* what() const noexcept override;

private:
    std::string reason;
};

/**
 * @brief The Data provider interface allows to customize extraction of video data stream used by
 * gapi::streaming::wip::onevpl::GSource instead of reading stream from file (by default).
 *
 * Implementation constructor MUST provide entire valid object.
 * If error happens implementation SHOULD throw `DataProviderException` kind exceptions
 *
 * @note Implementation MUST manage stream resources by itself to avoid any kind of leask.
 * For implementation example please see `StreamDataProvider` in `tests/streaming/gapi_streaming_tests.cpp`
 */
struct GAPI_EXPORTS IDataProvider {
    using Ptr = std::shared_ptr<IDataProvider>;

    virtual ~IDataProvider() {};

    /**
     * The function is used by onevpl::GSource to extract binary data stream from @ref IDataProvider
     * implementation.
     *
     * It MUST throw `DataProviderException` kind exceptions in fail cases.
     * It MUST return 0 in EOF which considered as not-fail case.
     *
     * @param out_data_bytes_size the available capacity of @ref out_data buffer.
     * @param out_data the output consumer buffer with capacity @ref out_data_bytes_size.
     * @return fetched bytes count.
     */
    virtual size_t fetch_data(size_t out_data_bytes_size, void* out_data) = 0;

    /**
     * The function is used by onevpl::GSource to check more binary data availability.
     *
     * It MUST return TRUE in case of EOF and NO_THROW exceptions.
     */
    virtual bool empty() const = 0;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
