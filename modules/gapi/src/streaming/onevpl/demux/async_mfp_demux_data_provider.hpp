// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_DEMUX_ASYNC_MFP_DEMUX_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_DEMUX_ASYNC_MFP_DEMUX_DATA_PROVIDER_HPP

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <queue>

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

#ifdef _WIN32
#define NOMINMAX
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>
#include <mftransform.h>
#include <mferror.h>
#include <shlwapi.h>
#include <wmcontainer.h>
#include <wmcodecdsp.h>
#undef NOMINMAX

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/data_provider_defines.hpp"
#include "streaming/onevpl/utils.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct GAPI_EXPORTS MFPAsyncDemuxDataProvider : public IDataProvider,
                                                public IMFSourceReaderCallback {
    MFPAsyncDemuxDataProvider(const std::string& file_path,
                              size_t keep_preprocessed_buf_count_value = 3);
    ~MFPAsyncDemuxDataProvider();

    mfx_codec_id_type get_mfx_codec_id() const override;
    bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &out_bitsream) override;
    bool empty() const override;

protected: /* For Unit tests only */
    enum class State {
        InProgress,
        Exhausted
    };

    // IUnknown methods forbidden for current implementations
    STDMETHODIMP QueryInterface(REFIID iid, void** ppv) override;
    STDMETHODIMP_(ULONG) AddRef() override;
    STDMETHODIMP_(ULONG) Release() override;

    // IMFSourceReaderCallback methods
    virtual STDMETHODIMP OnReadSample(HRESULT status, DWORD stream_index,
                                      DWORD stream_flag, LONGLONG timestamp,
                                      IMFSample *sample_ptr) override;
    STDMETHODIMP OnEvent(DWORD, IMFMediaEvent *) override;
    STDMETHODIMP OnFlush(DWORD) override;

    // implementation methods
    void flush();
    HRESULT request_next(HRESULT hr, DWORD stream_flag,
                         size_t worker_buffer_count);
    void consume_worker_data();
    virtual size_t produce_worker_data(void *key,
                                       ComPtrGuard<IMFMediaBuffer> &&buffer,
                                       std::shared_ptr<mfx_bitstream> &&staging_stream);
    size_t get_locked_buffer_size() const;

private:
    static bool select_supported_video_stream(ComPtrGuard<IMFPresentationDescriptor> &descriptor,
                                              mfx_codec_id_type &out_codec_id,
                                              void *source_id);
    // members
    size_t keep_preprocessed_buf_count;

    // COM members
    ComPtrGuard<IMFMediaSource> source;
    ComPtrGuard<IMFSourceReader> source_reader;
    std::atomic<ULONG> com_interface_reference_count;

    mfx_codec_id_type codec;

    // worker & processing buffers
    std::map<void*, ComPtrGuard<IMFMediaBuffer>> worker_key_to_buffer_mapping_storage;
    std::map<void*, ComPtrGuard<IMFMediaBuffer>> processing_key_to_buffer_mapping_storage;
    std::queue<std::shared_ptr<mfx_bitstream>> worker_locked_buffer_storage;
    std::queue<std::shared_ptr<mfx_bitstream>> processing_locked_buffer_storage;
    std::condition_variable buffer_storage_non_empty_cond;
    mutable std::mutex buffer_storage_mutex;

    std::atomic_flag submit_read_request;
    std::atomic<State> provider_state;
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
    explicit MFPAsyncDemuxDataProvider(const std::string&);

    mfx_codec_id_type get_mfx_codec_id() const override;
    bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &out_bitsream) override;
    bool empty() const override;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // _WIN32
#endif // HAVE_ONEVPL
#endif // GAPI_STREAMING_ONEVPL_DEMUX_ASYNC_MFP_DEMUX_DATA_PROVIDER_HPP
