// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation
#ifdef HAVE_ONEVPL
#ifdef _WIN32
#include <errno.h>
#include <atlstr.h>

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>
#include <mfidl.h>
#include <mftransform.h>
#include <mferror.h>
#include <wmcontainer.h>
#include <wmcodecdsp.h>

#pragma comment(lib,"Mf.lib")
#pragma comment(lib,"Mfuuid.lib")
#pragma comment(lib,"Mfplat.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid") //???
#endif // _WIN32

#include "streaming/onevpl/demux/mfp_demux_data_provider.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
#ifdef _WIN32
HRESULT CreateMediaSource(const std::string& url, IMFMediaSource **ppSource) {
    CStringW sURL(url.c_str());

    HRESULT hr = S_OK;

    IMFSourceResolver *pSourceResolver = nullptr;
    IUnknown *pSourceUnk = nullptr;

    hr = MFCreateSourceResolver(&pSourceResolver);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "cannot create MFCreateSourceResolver from URI: " +
                                               url);
    }

    MF_OBJECT_TYPE ObjectType = MF_OBJECT_INVALID;
    DWORD resolver_flags = MF_RESOLUTION_MEDIASOURCE | MF_RESOLUTION_READ |
                           MF_RESOLUTION_KEEP_BYTE_STREAM_ALIVE_ON_FAIL;
    do {
        hr = pSourceResolver->CreateObjectFromURL(sURL,
                                                  resolver_flags,
                                                  nullptr, &ObjectType,
                                                  &pSourceUnk);
        if (FAILED(hr)) {
            resolver_flags ^= MF_RESOLUTION_KEEP_BYTE_STREAM_ALIVE_ON_FAIL;
            resolver_flags ^= MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE ;
            GAPI_LOG_DEBUG(nullptr, "Cannot create MF_RESOLUTION_MEDIASOURCE using file extension, "
                                    "try special mode");
            continue;
        }
    } while (FAILED(hr) &&
             (resolver_flags & MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE));

    if (FAILED(hr)) {
        GAPI_LOG_WARNING(nullptr, "Cannot create MF_RESOLUTION_MEDIASOURCE from URI: " <<
                                  url);
        pSourceResolver->Release();
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "cannot create CreateObjectFromURL");
    }

    hr = pSourceUnk->QueryInterface(__uuidof(IMFMediaSource), (void**)ppSource);
    if (FAILED(hr)) {
        pSourceResolver->Release();
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "cannot query IMFMediaSource");
    }
    if (pSourceResolver) {
        pSourceResolver->Release();
        pSourceResolver = nullptr;
    }
    if (pSourceUnk) {
        pSourceUnk->Release();
        pSourceUnk = nullptr;
    }

    return hr;
}

MFPDemuxDataProvider::MFPDemuxDataProvider(const std::string& file_path) {
    HRESULT hr = S_OK;
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot initialize MFStartup");
    }

    source_ptr = nullptr;
    GAPI_LOG_INFO(nullptr, "IDataProvider: " << this <<
                            " - initializing, URI " << file_path);
    hr = CreateMediaSource(file_path, &source_ptr);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create IMFMediaSource");
    }

    GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            ", URI: " << file_path <<
                            " - start creating source attributes");
    IMFAttributes *pAttributes = nullptr;
    IMFMediaType *pType = nullptr;

    hr = MFCreateAttributes(&pAttributes, 2);

    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot MFCreateAttributes");
    }

    hr = pAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);

    // Set the callback pointer.
    /*if (SUCCEEDED(hr))
    {
        hr = pAttributes->SetUnknown(
            MF_SOURCE_READER_ASYNC_CALLBACK,
            this
            );
    }*/

    reader = nullptr;
    hr = MFCreateSourceReaderFromMediaSource(source_ptr, pAttributes,
                                             &reader);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create MFCreateSourceReaderFromMediaSource");
    }

    GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            " - created IMFSourceReader: " << reader);
    // Try to find a suitable output type.
    if (SUCCEEDED(hr)) {
        for (DWORD i = 0; ; i++) {
            hr = reader->GetNativeMediaType(
                (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                i,
                &pType
                );

            if (FAILED(hr)) { break; }

            /////////////////////
            BOOL bFound = FALSE;
            GUID subtype = { 0 };

            hr = pType->GetGUID(MF_MT_SUBTYPE, &subtype);

            if (FAILED(hr)) {
                break;
            }
            /////////////////////


            pType->Release();
            pType = nullptr;

            if (SUCCEEDED(hr)) {
                // Found an output type.
                break;
            }
        }
    }

    if (SUCCEEDED(hr)) {
        // Ask for the first sample.
        hr = reader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,
            NULL,
            NULL,
            NULL,
            NULL
            );
    } else {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot ReadSample");
    }
    GAPI_LOG_INFO(nullptr, "IDataProvider: " << this <<
                            " - initialized");
}

MFPDemuxDataProvider::~MFPDemuxDataProvider() {
    GAPI_LOG_INFO(nullptr, "IDataProvider: " << this <<
                            " - deinitializing");
    if (reader) {
        reader->Release();
        reader = nullptr;
    }

    if (source_ptr) {
        source_ptr->Shutdown();
        source_ptr = nullptr;
    }

    (void)MFShutdown();
    GAPI_LOG_INFO(nullptr, "IDataProvider: " << this <<
                            " - deinitialized");
}

size_t MFPDemuxDataProvider::fetch_data(size_t out_data_bytes_size, void* out_data) {
    if (empty()) {
        return 0;
    }

    GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            " - dst bytes count: " << out_data_bytes_size <<
                            ", dst: " << out_data);

    BYTE *mapped_buffer_data = nullptr;
    DWORD mapped_buffer_size = 0;
    IMFMediaBuffer *contiguous_buffer = 0;
    IMFSample *retrieved_sample = nullptr;
    DWORD retrieved_stream_flag = 0;

    HRESULT hr = S_OK;
    do {
        GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            " - retrieve sample from source");
        hr = reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                0,
                                nullptr,
                                &retrieved_stream_flag,
                                nullptr,
                                &retrieved_sample);

        if (FAILED(hr)) {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - cannot ReadSample");
        }

        if (retrieved_stream_flag & MF_SOURCE_READERF_ENDOFSTREAM) {
            GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            " - EOF");
            reader->Release();
            reader = nullptr;
        }

        if (retrieved_stream_flag & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED){
            // Type change. Get the new format.
            //hr = GetVideoFormat(&m_format);
            GAPI_LOG_WARNING(nullptr, "IDataProvider: " << this <<
                            " - Media type changing is UNSUPPORTED");
            throw DataProviderSystemErrorException(HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - TODO");
        }
    }
    while (!retrieved_sample && !empty());

    if (retrieved_sample) {
        GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                                " - sample retrieved");

        retrieved_sample->AddRef();
        hr = retrieved_sample->ConvertToContiguousBuffer(&contiguous_buffer);
        if (FAILED(hr)) {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - ConvertToContiguousBuffer failed");
        }

        //TODO
        hr = contiguous_buffer->Lock(&mapped_buffer_data, nullptr, &mapped_buffer_size);
        if (FAILED(hr)) {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - canno Lock buffer");
        }

        if (mapped_buffer_data)
        {
            GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            " - fetch buffer from mapped data with size: " << mapped_buffer_size);
            memcpy(out_data, mapped_buffer_data, std::min<size_t>(mapped_buffer_size, out_data_bytes_size));
            contiguous_buffer->Unlock();
        }
        contiguous_buffer->Release();
        retrieved_sample->Release();
    }
    GAPI_LOG_DEBUG(nullptr, "IDataProvider: " << this <<
                            " - bytes fetched: " << mapped_buffer_size);
    return mapped_buffer_size;
}

bool MFPDemuxDataProvider::empty() const {
    return !reader;
}
#else

MFPDemuxDataProvider::MFPDemuxDataProvider(const std::string&) {
    GAPI_Assert(false && "Unsupported: Microsoft Media Foundation is not available");
}
size_t MFPDemuxDataProvider::fetch_data(size_t, void*) {
}
bool MFPDemuxDataProvider::empty() const override {
}
#endif // _WIN32
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
