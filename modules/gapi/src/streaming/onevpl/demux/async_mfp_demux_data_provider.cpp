// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation
#ifdef HAVE_ONEVPL
#include <errno.h>

#include <opencv2/gapi/own/assert.hpp>
#include "streaming/onevpl/demux/async_mfp_demux_data_provider.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
#ifdef _WIN32
static HRESULT create_media_source(const std::string& url, IMFMediaSource **ppSource) {
    wchar_t sURL[MAX_PATH];
    GAPI_Assert(url.size() < MAX_PATH && "Windows MAX_PATH limit was reached");
    size_t ret_url_length = 0;
    mbstowcs_s(&ret_url_length, sURL, url.data(), url.size());

    HRESULT hr = S_OK;
    ComPtrGuard<IMFSourceResolver> source_resolver = createCOMPtrGuard<IMFSourceResolver>();
    {
        IMFSourceResolver *source_resolver_tmp = nullptr;
        hr = MFCreateSourceResolver(&source_resolver_tmp);
        if (FAILED(hr)) {
            throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                                   "cannot create MFCreateSourceResolver from URI: " +
                                                   url);
        }
        source_resolver.reset(source_resolver_tmp);
    }

    MF_OBJECT_TYPE ObjectType = MF_OBJECT_INVALID;
    /**
     * NB:
     * CreateObjectFromURL throws exception if actual container type is mismatched with
     * file extension. To overcome this situation by MFP it is possible to apply 2 step
     * approach: at first step we pass special flag
     * `MF_RESOLUTION_KEEP_BYTE_STREAM_ALIVE_ON_FAIL` which claims to fail with error
     * in any case of input instead exception throwing;
     * at the second step we must cease `MF_RESOLUTION_KEEP_BYTE_STREAM_ALIVE_ON_FAIL`
     * flag AND set another special flag
     * `MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE`
     * to filter out container type & file extension mismatch errors.
     *
     * If it failed at second phase then some other errors were not related
     * to types-extension disturbance would happen and data provider must fail ultimately.
     *
     * If second step passed then data provider would continue execution
     */
    IUnknown *source_unknown_tmp = nullptr;
    DWORD resolver_flags = MF_RESOLUTION_MEDIASOURCE | MF_RESOLUTION_READ |
                           MF_RESOLUTION_KEEP_BYTE_STREAM_ALIVE_ON_FAIL;
    hr = source_resolver->CreateObjectFromURL(sURL,
                                              resolver_flags,
                                              nullptr, &ObjectType,
                                              &source_unknown_tmp);
    if (FAILED(hr)) {
        GAPI_LOG_DEBUG(nullptr, "Cannot create MF_RESOLUTION_MEDIASOURCE using file extension, "
                                " looks like actual media container type doesn't match to file extension. "
                                "Try special mode");
        resolver_flags ^= MF_RESOLUTION_KEEP_BYTE_STREAM_ALIVE_ON_FAIL;
        resolver_flags ^= MF_RESOLUTION_CONTENT_DOES_NOT_HAVE_TO_MATCH_EXTENSION_OR_MIME_TYPE;
        hr = source_resolver->CreateObjectFromURL(sURL, resolver_flags,
                                                  nullptr, &ObjectType,
                                                  &source_unknown_tmp);
        if (FAILED(hr)) {
            GAPI_LOG_WARNING(nullptr, "Cannot create MF_RESOLUTION_MEDIASOURCE from URI: " <<
                                      url << ". Abort");
            throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                                   "CreateObjectFromURL failed");
        }
    }

    ComPtrGuard<IUnknown> source_unknown = createCOMPtrGuard(source_unknown_tmp);
    hr = source_unknown->QueryInterface(__uuidof(IMFMediaSource), (void**)ppSource);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "QueryInterface for IMFMediaSource failed");
    }

    return hr;
}

/*
 * The next part of converting GUID into string function GetGUIDNameConst
 * was copied and modified from
 * https://docs.microsoft.com/en-us/windows/win32/medfound/media-type-debugging-code
 */
#ifndef IF_EQUAL_RETURN
#define IF_EQUAL_RETURN(param, val) if(val == param) return #val
#endif

static const char* GetGUIDNameConst(const GUID& guid)
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

    return "<unknown>";
}

static IDataProvider::mfx_codec_id_type convert_to_mfx_codec_id(const GUID& guid) {
    if (guid == MFVideoFormat_H264) {
        return MFX_CODEC_AVC;
    } else if (guid == MFVideoFormat_H265 ||
               guid == MFVideoFormat_HEVC ||
               guid == MFVideoFormat_HEVC_ES) {
        return MFX_CODEC_HEVC;
    } else if (guid == MFAudioFormat_MPEG) {
        return MFX_CODEC_MPEG2;
    } else if (guid == MFVideoFormat_WVC1) {
        return MFX_CODEC_VC1;
    } else if (guid == MFVideoFormat_VP90) {
        return MFX_CODEC_VP9;
    } else if (guid == MFVideoFormat_AV1) {
        return MFX_CODEC_AV1;
    } else if (guid == MFVideoFormat_MJPG) {
        return MFX_CODEC_JPEG;
    }

    throw DataProviderUnsupportedException(std::string("unsupported codec type: ") +
                                           GetGUIDNameConst(guid));
}

bool MFPAsyncDemuxDataProvider::select_supported_video_stream(
                            ComPtrGuard<IMFPresentationDescriptor> &descriptor,
                            mfx_codec_id_type &out_codec_id,
                            void *source_id) {
    DWORD stream_count = 0;
    BOOL is_stream_selected = false;
    descriptor->GetStreamDescriptorCount(&stream_count);
    GAPI_LOG_DEBUG(nullptr, "[" << source_id << "] " <<
                            "received stream count: " << stream_count);
    for (DWORD stream_index = 0;
        stream_index < stream_count && !is_stream_selected; stream_index++) {

        GAPI_LOG_DEBUG(nullptr, "[" << source_id << "] " <<
                                "check stream info by index: " << stream_index);
        IMFStreamDescriptor *stream_descriptor_tmp = nullptr;
        descriptor->GetStreamDescriptorByIndex(stream_index, &is_stream_selected,
                                               &stream_descriptor_tmp);
        if (!stream_descriptor_tmp) {
            GAPI_LOG_WARNING(nullptr, "[" << source_id << "] " <<
                                      "Cannot get stream descriptor by index: " <<
                                      stream_index);
            continue;
        }

        ComPtrGuard<IMFStreamDescriptor> stream_descriptor =
                                    createCOMPtrGuard(stream_descriptor_tmp);
        is_stream_selected = false; // deselect until supported stream found
        IMFMediaTypeHandler *handler_tmp = nullptr;
        stream_descriptor->GetMediaTypeHandler(&handler_tmp);
        if (!handler_tmp) {
            GAPI_LOG_WARNING(nullptr, "[" << source_id << "] " <<
                                      "Cannot get media type handler for stream by index: " <<
                                      stream_index);
            continue;
        }

        ComPtrGuard<IMFMediaTypeHandler> handler = createCOMPtrGuard(handler_tmp);
        GUID guidMajorType;
        if (FAILED(handler->GetMajorType(&guidMajorType))) {
            GAPI_LOG_WARNING(nullptr, "[" << source_id << "] " <<
                                      "Cannot get major GUID type for stream by index: " <<
                                      stream_index);
            continue;
        }

        if (guidMajorType != MFMediaType_Video) {
            GAPI_LOG_DEBUG(nullptr, "[" << source_id << "] " <<
                                    "Skipping non-video stream");
            continue;
        }
        GAPI_LOG_DEBUG(nullptr, "[" << source_id << "] " <<
                                "video stream detected");
        IMFMediaType *media_type_tmp = nullptr;
        handler->GetCurrentMediaType(&media_type_tmp);
        if (!media_type_tmp) {
            GAPI_LOG_WARNING(nullptr, "[" << source_id << "] " <<
                                      "Cannot determine media type for stream by index: " <<
                                      stream_index);
            continue;
        }

        ComPtrGuard<IMFMediaType> media_type = createCOMPtrGuard(media_type_tmp);
        GUID subtype;
        if (SUCCEEDED(media_type->GetGUID(MF_MT_SUBTYPE, &subtype))) {
            GAPI_LOG_DEBUG(nullptr, "[" << source_id << "] " <<
                                    "video type: " << GetGUIDNameConst(subtype));

            std::string is_codec_supported("unsupported, skip...");
            try {
                out_codec_id = convert_to_mfx_codec_id(subtype);
                is_stream_selected = true;
                is_codec_supported = "selected!";
            } catch (...) {}

            GAPI_LOG_INFO(nullptr, "[" << source_id << "] " <<
                          "video stream index: " << stream_index <<
                          ", codec: " << GetGUIDNameConst(subtype) <<
                          " - " << is_codec_supported)
        } else {
            GAPI_LOG_WARNING(nullptr, "[" << source_id << "] " <<
                                      "Cannot get media GUID subtype for stream by index: " <<
                                      stream_index);
            continue;
        }
    }
    return is_stream_selected;
}

MFPAsyncDemuxDataProvider::MFPAsyncDemuxDataProvider(const std::string& file_path,
                                                     size_t keep_preprocessed_buf_count_value) :
  keep_preprocessed_buf_count(keep_preprocessed_buf_count_value),
  source(createCOMPtrGuard<IMFMediaSource>()),
  source_reader(createCOMPtrGuard<IMFSourceReader>()),
  codec(std::numeric_limits<uint32_t>::max()),
  provider_state(State::InProgress) {

    submit_read_request.clear();
    com_interface_reference_count = 1; // object itself

    HRESULT hr = S_OK;
    hr = MFStartup(MF_VERSION);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot initialize MFStartup");
    }

    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                           " initializing, URI " << file_path);
    IMFMediaSource *source_tmp = nullptr;
    hr = create_media_source(file_path, &source_tmp);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot create IMFMediaSource");
    }
    source.reset(source_tmp);

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            " start creating source attributes");
    IMFAttributes *attrs_tmp = nullptr;

    // NB: create 2 attributes for disable converters & async callback capability
    const UINT32 relevant_attributes_count = 2;
    hr = MFCreateAttributes(&attrs_tmp, relevant_attributes_count);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "MFCreateAttributes failed");
    }

    ComPtrGuard<IMFAttributes> attributes = createCOMPtrGuard(attrs_tmp);
    hr = attributes->SetUINT32(MF_READWRITE_DISABLE_CONVERTERS, TRUE);

    // set the callback pointer.
    if (SUCCEEDED(hr))
    {
        hr = attributes->SetUnknown(
            MF_SOURCE_READER_ASYNC_CALLBACK,
            this
            );
    }
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr), "Cannot set MFP async callback ");
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "is getting presentation descriptor");
    IMFPresentationDescriptor* descriptor_tmp = nullptr;
    hr = source->CreatePresentationDescriptor(&descriptor_tmp);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "CreatePresentationDescriptor failed");
    }
    ComPtrGuard<IMFPresentationDescriptor> descriptor = createCOMPtrGuard(descriptor_tmp);
    if (!MFPAsyncDemuxDataProvider::select_supported_video_stream(descriptor, codec, this)) {
        // NB: let's pretty notify clients about list of supported codecs to keep
        // contract in explicit way to avoid continuous troubleshooting
        const auto &supported_codecs = get_supported_mfx_codec_ids();
        std::string ss;
        for (mfxU32 id : supported_codecs) {
            ss += mfx_codec_id_to_cstr(id);
            ss += ", ";
        }
        if (!ss.empty()) {
            ss.erase(ss.size() - 2, 2);
        }

        GAPI_LOG_WARNING(nullptr, "[" << this << "] "
                         "couldn't find video stream with supported params, "
                         "expected codecs: " << ss);
        throw DataProviderUnsupportedException("couldn't find supported video stream");
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "is creating media source");
    IMFSourceReader *source_reader_tmp = nullptr;
    hr = MFCreateSourceReaderFromMediaSource(source.get(), attributes.get(),
                                             &source_reader_tmp);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "MFCreateSourceReaderFromMediaSource failed");
    }
    source_reader = createCOMPtrGuard(source_reader_tmp);

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "created IMFSourceReader: " << source_reader);

    // Ask for the first sample.
    hr = request_next(hr, 0, 0);
    if (FAILED(hr)) {
        throw DataProviderSystemErrorException(HRESULT_CODE(hr),
                                               "ReadSample failed while requesting initial sample");
    }
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                            "initialized");
}

MFPAsyncDemuxDataProvider::~MFPAsyncDemuxDataProvider() {
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                            "begin deinitializing");

    flush();

    {
        std::unique_lock<std::mutex> l(buffer_storage_mutex);
        GAPI_LOG_INFO(nullptr, "Clean up async storage, count: " <<
                               worker_key_to_buffer_mapping_storage.size());
        for (auto& buffer : worker_key_to_buffer_mapping_storage) {
            if (buffer.second) {
                buffer.second->Unlock();
            }
        }
        worker_key_to_buffer_mapping_storage.clear();
    }

    GAPI_LOG_INFO(nullptr, "Clean data storage, elapsed buffer count: " <<
                           processing_key_to_buffer_mapping_storage.size());
    for (auto& buffer : processing_key_to_buffer_mapping_storage) {
        if (buffer.second) {
            buffer.second->Unlock();
        }
    }
    processing_key_to_buffer_mapping_storage.clear();

    // release COM object before overall MFP shutdown
    source_reader.reset();
    source.reset();

    MFShutdown();
    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                           "deinitialized");
}


ULONG MFPAsyncDemuxDataProvider::AddRef() {
    // align behavior with InterlockedIncrement
    return com_interface_reference_count.fetch_add(1) + 1;
}

ULONG MFPAsyncDemuxDataProvider::Release() {
    auto count = com_interface_reference_count.fetch_sub(1);
    GAPI_Assert(count != 0 && "Incorrect reference counting for MFPAsyncDemuxDataProvider");
    count -= 1; // align behavior with InterlockedDecrement
    return count;
}

HRESULT MFPAsyncDemuxDataProvider::QueryInterface(REFIID riid, void** ppv)
{
    static const QITAB qit[] =
    {
        QITABENT(MFPAsyncDemuxDataProvider, IMFSourceReaderCallback),
        { 0 },
    };
    return QISearch(this, qit, riid, ppv);
}


STDMETHODIMP
MFPAsyncDemuxDataProvider::OnReadSample(HRESULT status, DWORD,
                                        DWORD stream_flag, LONGLONG,
                                        IMFSample *sample_ptr) {
    GAPI_LOG_DEBUG(nullptr, "[" << this << "] status: " << std::to_string(HRESULT_CODE(status)) <<
                            ", stream flags: " << stream_flag <<
                            ", sample: " << sample_ptr);
    HRESULT hr = S_OK;
    if (FAILED(status)) {
        hr = status;
    }

    // check EOF
    if (stream_flag & MF_SOURCE_READERF_ENDOFSTREAM) {
        GAPI_LOG_DEBUG(nullptr, "[" << this << "] EOF");

        // close reader
        provider_state.store(State::Exhausted);
        buffer_storage_non_empty_cond.notify_all();
        return hr;
    }

    submit_read_request.clear();

    // extract stream data
    size_t worker_buffer_count = 0;
    if (SUCCEEDED(hr)) {
        if (sample_ptr) {
            // Get the video frame buffer from the sample.
            IMFMediaBuffer *buffer_ptr = nullptr;
            hr = sample_ptr->ConvertToContiguousBuffer(&buffer_ptr);
            GAPI_Assert(SUCCEEDED(hr) &&
                        "MFPAsyncDemuxDataProvider::OnReadSample - ConvertToContiguousBuffer failed");

            DWORD max_buffer_size = 0;
            DWORD curr_size = 0;

            // lock buffer directly into mfx bitstream
            std::shared_ptr<mfx_bitstream> staging_stream = std::make_shared<mfx_bitstream>();
            staging_stream->Data = nullptr;

            hr = buffer_ptr->Lock(&staging_stream->Data, &max_buffer_size, &curr_size);
            GAPI_Assert(SUCCEEDED(hr) &&
                        "MFPAsyncDemuxDataProvider::OnReadSample - Lock failed");

            staging_stream->MaxLength = max_buffer_size;
            staging_stream->DataLength = curr_size;
            staging_stream->CodecId = get_mfx_codec_id();

            GAPI_LOG_DEBUG(nullptr, "[" << this << "] bitstream created, data: " <<
                                    static_cast<void*>(staging_stream->Data) <<
                                    ", MaxLength: " << staging_stream->MaxLength <<
                                    ", DataLength: "  << staging_stream->DataLength);

            worker_buffer_count = produce_worker_data(staging_stream->Data,
                                                      createCOMPtrGuard(buffer_ptr),
                                                      std::move(staging_stream));
        }
    } else {
        GAPI_LOG_WARNING(nullptr, "[" << this << "] callback failed"
                                  ", status: " << std::to_string(HRESULT_CODE(status)) <<
                                  ", stream flags: " << stream_flag <<
                                  ", sample: " << sample_ptr);
    }

    hr = request_next(hr, stream_flag, worker_buffer_count);
    return hr;
}

size_t MFPAsyncDemuxDataProvider::get_locked_buffer_size() const {
    std::unique_lock<std::mutex> l(buffer_storage_mutex);
    return worker_locked_buffer_storage.size();
}

STDMETHODIMP MFPAsyncDemuxDataProvider::OnEvent(DWORD, IMFMediaEvent *) {
    return S_OK;
}

STDMETHODIMP MFPAsyncDemuxDataProvider::OnFlush(DWORD) {
    provider_state.store(State::Exhausted);
    buffer_storage_non_empty_cond.notify_all();
    return S_OK;
}

void MFPAsyncDemuxDataProvider::flush() {
    if(source_reader) {
        GAPI_LOG_INFO(nullptr, "[" << this << "] set flush");
        source_reader->Flush(static_cast<DWORD>(MF_SOURCE_READER_ALL_STREAMS));
    }

    size_t iterations = 0;
    const int waiting_ms = 100;
    const size_t warning_iteration_wait_count = 300; // approx 30 sec
    while (provider_state.load() != State::Exhausted) {
        iterations++;
        if (iterations > warning_iteration_wait_count) {
            GAPI_LOG_WARNING(nullptr, "[" << this << "] is still waiting for flush finishing, "
                                      "iteration: " << iterations);
        } else {
            GAPI_LOG_DEBUG(nullptr, "[" << this << "] is waiting for flush finishing, "
                                    "iteration: " << iterations);
        }
        std::unique_lock<std::mutex> l(buffer_storage_mutex);
        buffer_storage_non_empty_cond.wait_for(l, std::chrono::milliseconds(waiting_ms));
    }

    GAPI_LOG_INFO(nullptr, "[" << this << "] has flushed in: " <<
                           iterations * waiting_ms << "ms interval");
}

HRESULT MFPAsyncDemuxDataProvider::request_next(HRESULT hr,
                                                DWORD stream_flag,
                                                size_t worker_buffer_count) {
    GAPI_LOG_DEBUG(nullptr, "[" << this << "] status: " <<
                            std::to_string(HRESULT_CODE(hr)) <<
                            ", stream flags: " << stream_flag <<
                            ", worker buffer count: (" << worker_buffer_count <<
                            "/" << keep_preprocessed_buf_count << ")");
    // check gap in stream
    if (stream_flag & MF_SOURCE_READERF_STREAMTICK ) {
        GAPI_LOG_INFO(nullptr, "[" << this << "] stream gap detected");
        return hr;
    }

    if (FAILED(hr)) {
        GAPI_LOG_WARNING(nullptr, "[" << this << "] callback error "
                                  ", status: " << std::to_string(HRESULT_CODE(hr)) <<
                                  ", stream flags: " << stream_flag);
    }

    // put on worker buffers available ready
    if (worker_buffer_count < keep_preprocessed_buf_count) {
        // only one consumer might make submit
        if (!submit_read_request.test_and_set()) {
            hr = source_reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                            0, NULL, NULL, NULL, NULL);
            GAPI_LOG_DEBUG(nullptr, "[" << this << "] submit read sample, status: " <<
                                    std::to_string(HRESULT_CODE(hr)));
        }
    }
    return hr;
}

void MFPAsyncDemuxDataProvider::consume_worker_data() {
    // wait callback exchange
    std::unique_lock<std::mutex> l(buffer_storage_mutex);
    buffer_storage_non_empty_cond.wait(l, [this] {
        bool empty = worker_locked_buffer_storage.empty();
        if (empty) {
            if (!submit_read_request.test_and_set()) {
                (void)source_reader->ReadSample((DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM,
                                                0, NULL, NULL, NULL, NULL);
            }
        } else {
            worker_key_to_buffer_mapping_storage.swap(processing_key_to_buffer_mapping_storage);
            worker_locked_buffer_storage.swap(processing_locked_buffer_storage);
        }

        return !empty || provider_state == State::Exhausted;
    });
}

size_t MFPAsyncDemuxDataProvider::produce_worker_data(void *key,
                                                      ComPtrGuard<IMFMediaBuffer> &&buffer,
                                                      std::shared_ptr<mfx_bitstream> &&staging_stream) {
    size_t bitstream_count = 0;
    size_t worker_buffer_count = 0;
    {
        std::unique_lock<std::mutex> l(buffer_storage_mutex);

        // remember sample buffer to keep data safe
        worker_key_to_buffer_mapping_storage.emplace(key, std::move(buffer));
        worker_buffer_count = worker_key_to_buffer_mapping_storage.size();

        // remember bitstream for consuming
        worker_locked_buffer_storage.push(std::move(staging_stream));
        bitstream_count = worker_locked_buffer_storage.size();
        buffer_storage_non_empty_cond.notify_all();
    }
    GAPI_DbgAssert(worker_buffer_count == bitstream_count &&
                   "worker_key_to_buffer_mapping_storage & worker_locked_buffer_storage"
                   " must be the same size" );
    GAPI_LOG_DEBUG(nullptr, "[" << this << "] created dmux buffer by key: " <<
                            key << ", ready bitstream count: " <<
                            bitstream_count);

    return worker_buffer_count;
}

/////////////// IDataProvider methods ///////////////
IDataProvider::mfx_codec_id_type MFPAsyncDemuxDataProvider::get_mfx_codec_id() const {
    return codec;
}

bool MFPAsyncDemuxDataProvider::fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &out_bitsream) {
    if (empty()) {
        GAPI_LOG_DEBUG(nullptr, "[" << this << "] empty");
        return false;
    }

    // utilize consumed bitstream portion allocated at prev step
    if (out_bitsream) {
        // make dmux buffer unlock for not empty bitstream
        GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                                "bitstream before fetch: " <<
                                out_bitsream.get() <<
                                ", DataOffset: " <<
                                out_bitsream->DataOffset <<
                                ", DataLength: " <<
                                out_bitsream->DataLength);
        if (out_bitsream->DataOffset < out_bitsream->DataLength) {
            return true;
        }

        // cleanup
        auto it = processing_key_to_buffer_mapping_storage.find(out_bitsream->Data);
        if (it == processing_key_to_buffer_mapping_storage.end()) {
            GAPI_LOG_WARNING(nullptr, "[" << this << "] " <<
                                      "cannot find appropriate dmux buffer by key: " <<
                                      static_cast<void*>(out_bitsream->Data));
            GAPI_Error("invalid bitstream key");
        }
        if (it->second) {
            it->second->Unlock();
        }
        processing_key_to_buffer_mapping_storage.erase(it);
    }

    // consume new bitstream portion
    if (processing_locked_buffer_storage.empty() &&
        provider_state.load() == State::InProgress) {
        // get worker data collected from another thread
        consume_worker_data();
    }

    // EOF check: nothing to process at this point
    if (processing_locked_buffer_storage.empty()) {
        GAPI_DbgAssert(provider_state == State::Exhausted && "Source reader must be drained");
        out_bitsream.reset();
        return false;
    }

    out_bitsream = processing_locked_buffer_storage.front();
    processing_locked_buffer_storage.pop();

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] "
                            "bitstream after fetch: " <<
                            out_bitsream.get() <<
                            ", DataOffset: " <<
                            out_bitsream->DataOffset <<
                            ", DataLength: " <<
                            out_bitsream->DataLength);
    return true;
}

bool MFPAsyncDemuxDataProvider::empty() const {
    return (provider_state.load() == State::Exhausted) &&
           (processing_locked_buffer_storage.size() == 0) &&
           (get_locked_buffer_size() == 0);
}
#else // _WIN32

MFPAsyncDemuxDataProvider::MFPAsyncDemuxDataProvider(const std::string&) {
    GAPI_Error("Unsupported: Microsoft Media Foundation is not available");
}
IDataProvider::mfx_codec_id_type MFPAsyncDemuxDataProvider::get_mfx_codec_id() const {
    GAPI_Error("Unsupported: Microsoft Media Foundation is not available");
    return std::numeric_limits<mfx_codec_id_type>::max();
}

bool MFPAsyncDemuxDataProvider::fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &) {
    GAPI_Error("Unsupported: Microsoft Media Foundation is not available");
    return false;
}

bool MFPAsyncDemuxDataProvider::empty() const {
    GAPI_Error("Unsupported: Microsoft Media Foundation is not available");
    return true;
}
#endif // _WIN32
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // HAVE_ONEVPL
