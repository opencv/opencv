// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation
#ifdef HAVE_ONEVPL
#include <errno.h>
#ifdef _WIN32
#define NOMINMAX
#include <atlstr.h> //TODO If you see it then put comment to remove it, please

#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <mfobjects.h>
#include <mfidl.h>
#include <mftransform.h>
#include <mferror.h>
#include <wmcontainer.h>
#include <wmcodecdsp.h>
#undef NOMINMAX

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
        pSourceUnk->Release();
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

/*
 * The next part of converting GUID into string was copied and modified from
 * https://docs.microsoft.com/en-us/windows/win32/medfound/media-type-debugging-code
 */
#ifndef IF_EQUAL_RETURN
#define IF_EQUAL_RETURN(param, val) if(val == param) return #val
#endif

const char* GetGUIDNameConst(const GUID& guid)
{
    IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_MAJOR_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_SUBTYPE);
    IF_EQUAL_RETURN(guid, MF_MT_ALL_SAMPLES_INDEPENDENT);
    IF_EQUAL_RETURN(guid, MF_MT_FIXED_SIZE_SAMPLES);
    IF_EQUAL_RETURN(guid, MF_MT_COMPRESSED);
    IF_EQUAL_RETURN(guid, MF_MT_SAMPLE_SIZE);
    IF_EQUAL_RETURN(guid, MF_MT_WRAPPED_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_NUM_CHANNELS);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FLOAT_SAMPLES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_AVG_BYTES_PER_SECOND);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BLOCK_ALIGNMENT);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_BITS_PER_SAMPLE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_VALID_BITS_PER_SAMPLE);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_SAMPLES_PER_BLOCK);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_CHANNEL_MASK);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_FOLDDOWN_MATRIX);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKREF);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_PEAKTARGET);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGREF);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_WMADRC_AVGTARGET);
    IF_EQUAL_RETURN(guid, MF_MT_AUDIO_PREFER_WAVEFORMATEX);
    IF_EQUAL_RETURN(guid, MF_MT_AAC_PAYLOAD_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_AAC_AUDIO_PROFILE_LEVEL_INDICATION);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_SIZE);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MAX);
    IF_EQUAL_RETURN(guid, MF_MT_FRAME_RATE_RANGE_MIN);
    IF_EQUAL_RETURN(guid, MF_MT_PIXEL_ASPECT_RATIO);
    IF_EQUAL_RETURN(guid, MF_MT_DRM_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_PAD_CONTROL_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_SOURCE_CONTENT_HINT);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_CHROMA_SITING);
    IF_EQUAL_RETURN(guid, MF_MT_INTERLACE_MODE);
    IF_EQUAL_RETURN(guid, MF_MT_TRANSFER_FUNCTION);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_PRIMARIES);
    IF_EQUAL_RETURN(guid, MF_MT_CUSTOM_VIDEO_PRIMARIES);
    IF_EQUAL_RETURN(guid, MF_MT_YUV_MATRIX);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_LIGHTING);
    IF_EQUAL_RETURN(guid, MF_MT_VIDEO_NOMINAL_RANGE);
    IF_EQUAL_RETURN(guid, MF_MT_GEOMETRIC_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_MINIMUM_DISPLAY_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_APERTURE);
    IF_EQUAL_RETURN(guid, MF_MT_PAN_SCAN_ENABLED);
    IF_EQUAL_RETURN(guid, MF_MT_AVG_BITRATE);
    IF_EQUAL_RETURN(guid, MF_MT_AVG_BIT_ERROR_RATE);
    IF_EQUAL_RETURN(guid, MF_MT_MAX_KEYFRAME_SPACING);
    IF_EQUAL_RETURN(guid, MF_MT_DEFAULT_STRIDE);
    IF_EQUAL_RETURN(guid, MF_MT_PALETTE);
    IF_EQUAL_RETURN(guid, MF_MT_USER_DATA);
    IF_EQUAL_RETURN(guid, MF_MT_AM_FORMAT_TYPE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG_START_TIME_CODE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_PROFILE);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_LEVEL);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG2_FLAGS);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG_SEQUENCE_HEADER);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_0);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_0);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_SRC_PACK_1);
    IF_EQUAL_RETURN(guid, MF_MT_DV_AAUX_CTRL_PACK_1);
    IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_SRC_PACK);
    IF_EQUAL_RETURN(guid, MF_MT_DV_VAUX_CTRL_PACK);
    IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_HEADER);
    IF_EQUAL_RETURN(guid, MF_MT_ARBITRARY_FORMAT);
    IF_EQUAL_RETURN(guid, MF_MT_IMAGE_LOSS_TOLERANT);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG4_SAMPLE_DESCRIPTION);
    IF_EQUAL_RETURN(guid, MF_MT_MPEG4_CURRENT_SAMPLE_ENTRY);
    IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_4CC);
    IF_EQUAL_RETURN(guid, MF_MT_ORIGINAL_WAVE_FORMAT_TAG);

    // Media types

    IF_EQUAL_RETURN(guid, MFMediaType_Audio);
    IF_EQUAL_RETURN(guid, MFMediaType_Video);
    IF_EQUAL_RETURN(guid, MFMediaType_Protected);
    IF_EQUAL_RETURN(guid, MFMediaType_SAMI);
    IF_EQUAL_RETURN(guid, MFMediaType_Script);
    IF_EQUAL_RETURN(guid, MFMediaType_Image);
    IF_EQUAL_RETURN(guid, MFMediaType_HTML);
    IF_EQUAL_RETURN(guid, MFMediaType_Binary);
    IF_EQUAL_RETURN(guid, MFMediaType_FileTransfer);

    IF_EQUAL_RETURN(guid, MFVideoFormat_AI44); //     FCC('AI44')
    IF_EQUAL_RETURN(guid, MFVideoFormat_ARGB32); //   D3DFMT_A8R8G8B8
    IF_EQUAL_RETURN(guid, MFVideoFormat_AV1);
    IF_EQUAL_RETURN(guid, MFVideoFormat_AYUV); //     FCC('AYUV')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DV25); //     FCC('dv25')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DV50); //     FCC('dv50')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVH1); //     FCC('dvh1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVSD); //     FCC('dvsd')
    IF_EQUAL_RETURN(guid, MFVideoFormat_DVSL); //     FCC('dvsl')
    IF_EQUAL_RETURN(guid, MFVideoFormat_H264); //     FCC('H264')
    IF_EQUAL_RETURN(guid, MFVideoFormat_H265);
    IF_EQUAL_RETURN(guid, MFVideoFormat_HEVC);
    IF_EQUAL_RETURN(guid, MFVideoFormat_HEVC_ES);
    IF_EQUAL_RETURN(guid, MFVideoFormat_I420); //     FCC('I420')
    IF_EQUAL_RETURN(guid, MFVideoFormat_IYUV); //     FCC('IYUV')
    IF_EQUAL_RETURN(guid, MFVideoFormat_M4S2); //     FCC('M4S2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MJPG);
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP43); //     FCC('MP43')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP4S); //     FCC('MP4S')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MP4V); //     FCC('MP4V')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MPG1); //     FCC('MPG1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MSS1); //     FCC('MSS1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_MSS2); //     FCC('MSS2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_NV11); //     FCC('NV11')
    IF_EQUAL_RETURN(guid, MFVideoFormat_NV12); //     FCC('NV12')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P010); //     FCC('P010')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P016); //     FCC('P016')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P210); //     FCC('P210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_P216); //     FCC('P216')
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB24); //    D3DFMT_R8G8B8
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB32); //    D3DFMT_X8R8G8B8
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB555); //   D3DFMT_X1R5G5B5
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB565); //   D3DFMT_R5G6B5
    IF_EQUAL_RETURN(guid, MFVideoFormat_RGB8);
    IF_EQUAL_RETURN(guid, MFVideoFormat_UYVY); //     FCC('UYVY')
    IF_EQUAL_RETURN(guid, MFVideoFormat_v210); //     FCC('v210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_v410); //     FCC('v410')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV1); //     FCC('WMV1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV2); //     FCC('WMV2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WMV3); //     FCC('WMV3')
    IF_EQUAL_RETURN(guid, MFVideoFormat_WVC1); //     FCC('WVC1')
    IF_EQUAL_RETURN(guid, MFVideoFormat_VP90);
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y210); //     FCC('Y210')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y216); //     FCC('Y216')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y410); //     FCC('Y410')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y416); //     FCC('Y416')
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y41P);
    IF_EQUAL_RETURN(guid, MFVideoFormat_Y41T);
    IF_EQUAL_RETURN(guid, MFVideoFormat_YUY2); //     FCC('YUY2')
    IF_EQUAL_RETURN(guid, MFVideoFormat_YV12); //     FCC('YV12')
    IF_EQUAL_RETURN(guid, MFVideoFormat_YVYU);

    IF_EQUAL_RETURN(guid, MFAudioFormat_PCM); //              WAVE_FORMAT_PCM
    IF_EQUAL_RETURN(guid, MFAudioFormat_Float); //            WAVE_FORMAT_IEEE_FLOAT
    IF_EQUAL_RETURN(guid, MFAudioFormat_DTS); //              WAVE_FORMAT_DTS
    IF_EQUAL_RETURN(guid, MFAudioFormat_Dolby_AC3_SPDIF); //  WAVE_FORMAT_DOLBY_AC3_SPDIF
    IF_EQUAL_RETURN(guid, MFAudioFormat_DRM); //              WAVE_FORMAT_DRM
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV8); //        WAVE_FORMAT_WMAUDIO2
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudioV9); //        WAVE_FORMAT_WMAUDIO3
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMAudio_Lossless); // WAVE_FORMAT_WMAUDIO_LOSSLESS
    IF_EQUAL_RETURN(guid, MFAudioFormat_WMASPDIF); //         WAVE_FORMAT_WMASPDIF
    IF_EQUAL_RETURN(guid, MFAudioFormat_MSP1); //             WAVE_FORMAT_WMAVOICE9
    IF_EQUAL_RETURN(guid, MFAudioFormat_MP3); //              WAVE_FORMAT_MPEGLAYER3
    IF_EQUAL_RETURN(guid, MFAudioFormat_MPEG); //             WAVE_FORMAT_MPEG
    IF_EQUAL_RETURN(guid, MFAudioFormat_AAC); //              WAVE_FORMAT_MPEG_HEAAC
    IF_EQUAL_RETURN(guid, MFAudioFormat_ADTS); //             WAVE_FORMAT_MPEG_ADTS_AAC

    return NULL;
}

IDataProvider::CodecID convert_to_CodecId(const GUID& guid) {
    if (guid== MFVideoFormat_H264) {
        return IDataProvider::CodecID::AVC;
    } else if (guid == MFVideoFormat_H265 ||
               guid == MFVideoFormat_HEVC ||
               guid == MFVideoFormat_HEVC_ES) {
        return IDataProvider::CodecID::HEVC;
    } else if (guid == MFAudioFormat_MPEG) {
        return IDataProvider::CodecID::MPEG2;
    } else if (guid == MFVideoFormat_WVC1) {
        return IDataProvider::CodecID::VC1;
    } else if (guid == MFVideoFormat_VP90) {
        return IDataProvider::CodecID::VP9;
    } else if (guid == MFVideoFormat_AV1) {
        return IDataProvider::CodecID::AV1;
    } else if (guid == MFVideoFormat_MJPG) {
        return IDataProvider::CodecID::JPEG;
    }
    throw std::runtime_error(std::string("unsupported codec type: ") +
                             GetGUIDNameConst(guid));
}

MFPDemuxDataProvider::MFPDemuxDataProvider(const std::string& file_path) :
codec(CodecID::UNCOMPRESSED) {
    HRESULT hr = S_OK;
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot initialize MFStartup");
    }

    source_ptr = nullptr;
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                           " initializing, URI " << file_path);
    hr = CreateMediaSource(file_path, &source_ptr);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create IMFMediaSource");
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            " start creating source attributes");
    IMFAttributes *pAttributes = nullptr;
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

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "is getting presentation description");
    IMFPresentationDescriptor* descriptor = nullptr;
    hr = source_ptr->CreatePresentationDescriptor(&descriptor);
    if (FAILED(hr)) {
        source_ptr->Release();
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "cannot CreatePresentationDescriptor");
    }

    DWORD stream_count = 0;
    BOOL is_stream_selected = false;
    descriptor->GetStreamDescriptorCount(&stream_count);
    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "received stream count: " << stream_count);
    for (DWORD stream_index = 0;
        stream_index < stream_count && !is_stream_selected; stream_index++) {
        IMFStreamDescriptor *stream_descriptor = nullptr;
        descriptor->GetStreamDescriptorByIndex(stream_index, &is_stream_selected,
                                               &stream_descriptor);

        is_stream_selected = false; // deselect until find supported stream
        IMFMediaTypeHandler *handler = nullptr;
        stream_descriptor->GetMediaTypeHandler(&handler);
        if (handler) {
            GUID guidMajorType;
            if (SUCCEEDED(handler->GetMajorType(&guidMajorType))) {

                if (MFMediaType_Video == guidMajorType) {
                    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                                            "video stream detected by index: " << stream_index);
                    IMFMediaType *pMediaType = nullptr;
                    handler->GetCurrentMediaType(&pMediaType);
                    if (pMediaType) {
                        GUID subtype;
                        if (SUCCEEDED(pMediaType->GetGUID(MF_MT_SUBTYPE, &subtype))) {
                            GAPI_LOG_DEBUG(nullptr, " video type:" << GetGUIDNameConst(subtype));

                            std::string is_codec_supported("unsupported, skip...");
                            try {
                                codec = convert_to_CodecId(subtype);
                                is_stream_selected = true;
                                is_codec_supported = "selected!";
                            } catch (...) {
                            }

                            GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                                          " by URI: " << file_path <<
                                          ", video stream index: " << stream_index <<
                                          ", codec: " << GetGUIDNameConst(subtype) <<
                                          " - " << is_codec_supported)
                        }
                        pMediaType->Release();
                        pMediaType = nullptr;
                    }
                }
            }
            handler->Release();
            handler = nullptr;
        }
        stream_descriptor->Release();
    }
    if (descriptor) {
        descriptor->Release();
        descriptor = nullptr;
    }

    if (!is_stream_selected) {
        GAPI_LOG_INFO(nullptr, "[" << this << "] couldn't select video stream with supported params");
        throw DataProviderUnsupportedException("couldn't find supported video stream");
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "is creating media source");
    reader = nullptr;
    hr = MFCreateSourceReaderFromMediaSource(source_ptr, pAttributes,
                                             &reader);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create MFCreateSourceReaderFromMediaSource");
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "created IMFSourceReader: " << reader);

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
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                            "initialized");
}

MFPDemuxDataProvider::~MFPDemuxDataProvider() {
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                            "deinitializing");
    if (reader) {
        reader->Release();
        reader = nullptr;
    }

    if (source_ptr) {
        source_ptr->Shutdown();
        source_ptr = nullptr;
    }

    (void)MFShutdown();
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                           "deinitialized");
}

MFPDemuxDataProvider::CodecID MFPDemuxDataProvider::get_codec() const {
    return codec;
}

size_t MFPDemuxDataProvider::fetch_data(size_t out_data_bytes_size, void* out_data) {
    if (empty()) {
        return 0;
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "dst bytes count: " << out_data_bytes_size <<
                            ", dst: " << out_data);

    BYTE *mapped_buffer_data = nullptr;
    DWORD mapped_buffer_size = 0;
    IMFMediaBuffer *contiguous_buffer = 0;
    IMFSample *retrieved_sample = nullptr;
    DWORD retrieved_stream_flag = 0;

    HRESULT hr = S_OK;
    do {
        GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                                "retrieve sample from source");
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
            GAPI_LOG_DEBUG(nullptr, "[" << this << "] EOF");
            reader->Release();
            reader = nullptr;
        }

        if (retrieved_stream_flag & MF_SOURCE_READERF_CURRENTMEDIATYPECHANGED){
            // Type change. Get the new format.
            //hr = GetVideoFormat(&m_format);
            GAPI_LOG_WARNING(nullptr, "[" << this << "] " <<
                                      "media type changing is UNSUPPORTED");
            throw DataProviderSystemErrorException(HRESULT_CODE(hr), "MFPDemuxDataProvider::fetch_data - TODO");
        }
    }
    while (!retrieved_sample && !empty());

    if (retrieved_sample) {
        GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                                "sample retrieved");

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
            GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                           "fetch buffer from mapped data with size: " <<
                           mapped_buffer_size);
            memcpy(out_data, mapped_buffer_data, std::min<size_t>(mapped_buffer_size, out_data_bytes_size));
            contiguous_buffer->Unlock();
        }
        contiguous_buffer->Release();
        retrieved_sample->Release();
    }
    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "bytes fetched: " << mapped_buffer_size);
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
