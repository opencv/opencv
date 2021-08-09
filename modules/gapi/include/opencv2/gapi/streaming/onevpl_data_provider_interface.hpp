// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#define GAPI_STREAMING_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
#include <stdexcept>
#include <string>

namespace cv {
namespace gapi {
namespace wip {

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

struct GAPI_EXPORTS IDataProvider {
    ~IDataProvider() {};
    virtual size_t provide_data(size_t out_data_bytes_size, void* out_data) = 0;
    virtual bool empty() const = 0;
};
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_DATA_PROVIDER_INTERFACE_HPP
