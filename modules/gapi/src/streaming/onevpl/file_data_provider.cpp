// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include <errno.h>

#include "streaming/onevpl/file_data_provider.hpp"
#include "streaming/onevpl/cfg_params_parser.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

#ifdef HAVE_ONEVPL
FileDataProvider::FileDataProvider(const std::string& file_path,
                                   const std::vector<CfgParam> codec_params) :
    source_handle(nullptr, &fclose) {

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "check codec Id from CfgParam, total param count: " <<
                            codec_params.size());
    auto codec_it =
        std::find_if(codec_params.begin(), codec_params.end(), [] (const CfgParam& value) {
            return value.get_name() == "mfxImplDescription.mfxDecoderDescription.decoder.CodecID";
        });
    if (codec_it == codec_params.end())
    {
        GAPI_LOG_WARNING(nullptr, "[" << this << "] " <<
                                  "\"mfxImplDescription.mfxDecoderDescription.decoder.CodecID\" "
                                  "is absent, total param count" << codec_params.size());
        return;
    }

    mfxVariant requested_codec = cfg_param_to_mfx_variant(*codec_it);
    switch(requested_codec.Data.U32) {
        case MFX_CODEC_AVC:
            codec = IDataProvider::CodecID::AVC;
            break;
        case MFX_CODEC_HEVC:
            codec = IDataProvider::CodecID::HEVC;
            break;
        case MFX_CODEC_MPEG2:
            codec = IDataProvider::CodecID::MPEG2;
            break;
        case MFX_CODEC_VC1:
            codec = IDataProvider::CodecID::AVC;
            break;
        case MFX_CODEC_VP9:
            codec = IDataProvider::CodecID::VP9;
            break;
        case MFX_CODEC_AV1:
            codec = IDataProvider::CodecID::AV1;
            break;
        case MFX_CODEC_JPEG:
            codec = IDataProvider::CodecID::JPEG;
            break;
        default:
            GAPI_LOG_WARNING(nullptr, "[" << this << "] " <<
                                      "unsupported CodecId requested: " << requested_codec.Data.U32 <<
                                      " check CfgParam \"mfxImplDescription.mfxDecoderDescription.decoder.CodecID\"");
            throw DataProviderUnsupportedException{
                "unsupported \"mfxImplDescription.mfxDecoderDescription.decoder.CodecID\""
                " requested for demultiplexed raw data"
            };
    }

    GAPI_LOG_DEBUG(nullptr, "[" << this << "] " <<
                            "opening file: " << file_path);
    source_handle.reset(fopen(file_path.c_str(), "rb"));
    if (!source_handle) {
        throw DataProviderSystemErrorException(errno,
                                               "FileDataProvider: cannot open source file: " + file_path);
    }

    GAPI_LOG_INFO(nullptr, "[" << this << "] " <<
                            "file: " << file_path << " opened, codec requested: " << to_cstr(codec));
}

FileDataProvider::~FileDataProvider() = default;

FileDataProvider::CodecID FileDataProvider::get_codec() const {
    return codec;
}

size_t FileDataProvider::fetch_data(size_t out_data_bytes_size, void* out_data) {
    if (empty()) {
        return 0;
    }

    size_t ret = fread(out_data, 1, out_data_bytes_size, source_handle.get());
    if (ret == 0) {
        if (feof(source_handle.get())) {
            source_handle.reset();
        } else {
            throw DataProviderSystemErrorException (errno, "FileDataProvider::fetch_data error read");
        }
    }
    return ret;
}

bool FileDataProvider::empty() const {
    return !source_handle;
}

#else

FileDataProvider::FileDataProvider(const std::string&,
                                   const std::vector<CfgParam>) :
    source_handle(nullptr, &fclose) {
    GAPI_Assert(false && "Unsupported: G-API compiled without `WITH_GAPI_ONEVPL=ON`");
}

FileDataProvider::~FileDataProvider() = default;

FileDataProvider::CodecID FileDataProvider::get_codec() const {
    return codec;
}

size_t FileDataProvider::fetch_data(size_t, void*) {
    return 0;
}

bool FileDataProvider::empty() const {
    return true;
}
#endif // HAVE_ONEVPL
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
