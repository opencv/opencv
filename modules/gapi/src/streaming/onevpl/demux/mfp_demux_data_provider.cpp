// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation
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

#include <fstream>

#include "streaming/onevpl/demux/mfp_demux_data_provider.hpp"

#pragma comment(lib,"Mf.lib")
#pragma comment(lib,"Mfuuid.lib")
#pragma comment(lib,"Mfplat.lib")
#pragma comment(lib, "shlwapi.lib")
#pragma comment(lib, "mfreadwrite.lib")


#pragma comment(lib, "mfuuid") //???


namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

inline HRESULT CreateMediaSource(const std::string& url, IMFMediaSource **ppSource)
    {
        CStringW sURL(url.c_str());

        HRESULT hr = S_OK;

        IMFSourceResolver   *pSourceResolver = nullptr;
        IUnknown            *pSourceUnk = nullptr;

        // Create the source resolver.
        if (SUCCEEDED(hr))
        {
            hr = MFCreateSourceResolver(&pSourceResolver);
        } else {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                                   "cannot create MFCreateSourceResolver");
        }

        // Use the source resolver to create the media source.
        if (SUCCEEDED(hr))
        {
            MF_OBJECT_TYPE ObjectType = MF_OBJECT_INVALID;

            hr = pSourceResolver->CreateObjectFromURL(
                sURL,                       // URL of the source.
                MF_RESOLUTION_MEDIASOURCE,  // Create a source object.
                nullptr,                       // Optional property store.
                &ObjectType,                // Receives the created object type.
                &pSourceUnk                 // Receives a pointer to the media source.
                );
        } else {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                                   "cannot create CreateObjectFromURL");
        }

        // Get the IMFMediaSource interface from the media source.
        if (SUCCEEDED(hr))
        {
            hr = pSourceUnk->QueryInterface(__uuidof(IMFMediaSource), (void**)ppSource);
        } else {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                                   "cannot query IMFMediaSource");
        }

        // Clean up
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

MFPDemuxDataProvider::MFPDemuxDataProvider(const std::string& file_path) :
    source_handle(fopen(file_path.c_str(), "rb"), &fclose) {
    if (!source_handle) {
        throw DataProviderSystemErrorException(errno,
                                               "MFPDemuxDataProvider: cannot open source file: " + file_path);
    }

    HRESULT hr = S_OK;
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot initialize MFStartup");
    }

    source_ptr = nullptr;
    hr = CreateMediaSource(file_path, &source_ptr);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create IMFMediaSource");
    }

///////////////////
    IMFAttributes   *pAttributes = nullptr;
    IMFMediaType    *pType = nullptr;
    reader = nullptr;

    //
    // Create the source reader.
    //

    // Create an attribute store to hold initialization settings.

    if (SUCCEEDED(hr)) {
        hr = MFCreateAttributes(&pAttributes, 2);
    }
    if (SUCCEEDED(hr)) {
        hr = pAttributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);
    }

    // Set the callback pointer.
    /*if (SUCCEEDED(hr))
    {
        hr = pAttributes->SetUnknown(
            MF_SOURCE_READER_ASYNC_CALLBACK,
            this
            );
    }*/

    if (SUCCEEDED(hr)) {
        hr = MFCreateSourceReaderFromMediaSource(
            source_ptr,
            pAttributes,
            &reader
            );
    } else {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create MFCreateSourceReaderFromMediaSource");
    }

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


    MFT_REGISTER_TYPE_INFO tinfo;
    tinfo.guidMajorType = MFMediaType_Video;
    tinfo.guidSubtype = MFVideoFormat_MPEG2;

    CLSID *pDecoderCLSIDs = NULL;   // Pointer to an array of CLISDs.
    UINT32 cDecoderCLSIDs = 0;   // Size of the array.


    uint32_t flags = MFT_ENUM_FLAG_ALL;
    IMFActivate** activate = nullptr;
	UINT32 count = 0;

    hr = MFTEnumEx(MFT_CATEGORY_DEMULTIPLEXER, flags, &tinfo, NULL, &activate, &count);
    /*hr = MFTEnum(
            MFT_CATEGORY_DEMULTIPLEXER,
            flags,                  // Reserved
            NULL,             // Input type to match. (Encoded type.)
            NULL,               // Output type to match. (Don't care.)
            &pDecoderCLSIDs,    // Receives a pointer to an array of CLSIDs.
            &cDecoderCLSIDs     // Receives the size of the array.
            ));*/

    std::wfstream out(L"MFT.txt", std::ios_base::out | std::ios_base::trunc);

    if (SUCCEEDED(hr) )
    {
        out << "Success, count: " << count << std::endl;
			for (int i = 0; i < count; ++i)
			{
				UINT32 l = 0;
				UINT32 l1 = 0;
				activate[i]->GetStringLength(MFT_FRIENDLY_NAME_Attribute, &l);
				std::unique_ptr<wchar_t[]> name(new wchar_t[l + 1]);
				memset(name.get(), 0, l + 1);
				hr = activate[i]->GetString(MFT_FRIENDLY_NAME_Attribute, name.get(), l + 1, &l1);
				out << name.get() << std::endl;
				activate[i]->Release();
			}
			//CoTaskMemFree(activate);
    } else {
        out <<HRESULT_CODE(hr);
    }

}

MFPDemuxDataProvider::~MFPDemuxDataProvider() {
    if (reader) {
        reader->Release();
        reader = nullptr;
    }

    if (source_ptr) {
        source_ptr->Shutdown();
        source_ptr = nullptr;
    }

    (void)MFShutdown();
}

size_t MFPDemuxDataProvider::fetch_data(size_t out_data_bytes_size, void* out_data) {
    if (empty()) {
        return 0;
    }
    DWORD       dwFlags = 0;
    BYTE        *pBitmapData = NULL;    // Bitmap data
    DWORD       cbBitmapData = 0;       // Size of data, in bytes
    IMFMediaBuffer *pBuffer = 0;
    IMFSample *pSample = NULL;

    HRESULT     hr = S_OK;
    hr = reader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
            0,
            NULL,
            &dwFlags,
            NULL,
            &pSample
            );

    if (FAILED(hr)) {
        throw DataProviderSystemErrorException (HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - cannot ReadSample");
    }

    if (dwFlags & MF_SOURCE_READERF_ENDOFSTREAM)
    {
        reader->Release();
        reader = nullptr;
    }

    if (dwFlags & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED)
    {
        // Type change. Get the new format.
        //hr = GetVideoFormat(&m_format);
        throw DataProviderSystemErrorException (HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - TODO");
    }

    if (pSample == NULL)
    {
        throw DataProviderSystemErrorException (HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - sample is NULL");
    }

    // We got a sample. Hold onto it.
    pSample->AddRef();

    hr = pSample->ConvertToContiguousBuffer(&pBuffer);

    if (FAILED(hr))
    {
        throw DataProviderSystemErrorException (HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - ConvertToContiguousBuffer failed");
    }

    hr = pBuffer->Lock(&pBitmapData, NULL, &cbBitmapData);
    if (FAILED(hr))
    {
        throw DataProviderSystemErrorException (HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - canno Lock buffer");
    }

    if (pBitmapData)
    {
        memcpy(out_data, pBitmapData, std::min<size_t>(cbBitmapData, out_data_bytes_size));
        pBuffer->Unlock();
    }
    pBuffer->Release();
    pSample->Release();
    return cbBitmapData;
}

bool MFPDemuxDataProvider::empty() const {
    return !reader;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
