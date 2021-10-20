// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DEMUX_MFP_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_DEMUX_MFP_DATA_PROVIDER_HPP
#include <stdio.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

class IMFMediaSource;
class IMFSourceReader;
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GAPI_EXPORTS MFPDemuxDataProvider : public IDataProvider {

    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;
    MFPDemuxDataProvider(const std::string& file_path);
    ~MFPDemuxDataProvider();

    size_t fetch_data(size_t out_data_bytes_size, void* out_data) override;
    bool empty() const override;
private:
    file_ptr source_handle;
    IMFMediaSource *source_ptr;
    IMFSourceReader *reader;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_DEMUX_MFP_DATA_PROVIDER_HPP
