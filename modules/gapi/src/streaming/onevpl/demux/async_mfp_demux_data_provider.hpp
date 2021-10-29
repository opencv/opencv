// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ASYNC_DEMUX_MFP_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_ASYNC_DEMUX_MFP_DATA_PROVIDER_HPP
#include <stdio.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

#ifdef HAVE_ONEVPL
#ifdef _WIN32
#define NOMINMAX
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>
#include <mfidl.h>
#include <mftransform.h>
#include <mferror.h>
#include <shlwapi.h>
#include <wmcontainer.h>
#include <wmcodecdsp.h>
#undef NOMINMAX

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/utils.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GAPI_EXPORTS MFPAsyncDemuxDataProvider : public IDataProvider,
                                                public IMFSourceReaderCallback {
    MFPAsyncDemuxDataProvider(const std::string& file_path);
    ~MFPAsyncDemuxDataProvider();

    CodecID get_codec() const override;
    size_t fetch_data(size_t out_data_bytes_size, void* out_data) override;
    mfxStatus fetch_bitstream_data(std::shared_ptr<mfxBitstream> &out_bitsream);
    bool empty() const override;

private:
    // IUnknown methods forbidden for current implemenations
    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) override;
    STDMETHODIMP_(ULONG) AddRef() override;
    STDMETHODIMP_(ULONG) Release() override;

    // IMFSourceReaderCallback methods
    STDMETHODIMP OnReadSample(HRESULT status, DWORD stream_index,
                              DWORD stream_flag, LONGLONG timestamp,
                              IMFSample *sample_ptr) override;

    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent *) override;
    STDMETHODIMP OnFlush(DWORD) override;

    ComPtrGuard<IMFMediaSource> source;
    ComPtrGuard<IMFSourceReader> source_reader;

    CodecID codec;

    std::queue<ComPtrGuard<IMFMediaBuffer>> sample_buffer_storage;
    std::queue<std::shared_ptr<mfxBitstream>> locked_buffer_storage;
    std::condition_variable buffer_storage_fill;
    mutable std::mutex buffer_storage_mutex;

    std::atomic_flag submit_read_request;
    std::atomic<ULONG> com_interface_reference_count;

    size_t get_locked_buffer_size() const;
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
struct GAPI_EXPORTS MFPAsyncDemuxDataProvider : public IDataProvider {
    MFPAsyncDemuxDataProvider(const std::string&);
    size_t fetch_data(size_t, void*) override;
    bool empty() const override;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // _WIN32
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_ASYNC_DEMUX_MFP_DATA_PROVIDER_HPP
