// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DEMUX_MFP_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_DEMUX_MFP_DATA_PROVIDER_HPP
#include <stdio.h>

#ifdef HAVE_ONEVPL
#ifdef _WIN32
#define NOMINMAX
#include <atlbase.h>
#undef NOMINMAX
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>

struct IMFMediaSource;
struct IMFSourceReader;

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GAPI_EXPORTS MFPDemuxDataProvider : public IDataProvider {
    MFPDemuxDataProvider(const std::string& file_path);
    ~MFPDemuxDataProvider();

    size_t fetch_data(size_t out_data_bytes_size, void* out_data) override;
    bool empty() const override;
private:
    IMFMediaSource *source_ptr;
    IMFSourceReader *reader;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#else // _WIN32
namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GAPI_EXPORTS MFPDemuxDataProvider : public IDataProvider {
    MFPDemuxDataProvider(const std::string&);
    size_t fetch_data(size_t, void*) override;
    bool empty() const override;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // _WIN32
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_DEMUX_MFP_DATA_PROVIDER_HPP
