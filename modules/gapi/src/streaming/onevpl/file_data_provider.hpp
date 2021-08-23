// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
#include <stdio.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct FileDataProvider : public IDataProvider {

    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;
    FileDataProvider(const std::string& file_path);
    ~FileDataProvider();

    size_t fetch_data(size_t out_data_bytes_size, void* out_data) override;
    bool empty() const override;
private:
    file_ptr source_handle;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
