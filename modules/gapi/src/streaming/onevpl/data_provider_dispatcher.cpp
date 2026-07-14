// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include "streaming/onevpl/data_provider_dispatcher.hpp"
#include "streaming/onevpl/file_data_provider.hpp"
#include "streaming/onevpl/demux/async_mfp_demux_data_provider.hpp"
#include "logger.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {

IDataProvider::Ptr DataProviderDispatcher::create(const std::string& file_path,
                                                  const std::vector<CfgParam> &cfg_params) {
    GAPI_LOG_INFO(nullptr, "try select suitable IDataProvider for source: " <<
                           file_path);

    IDataProvider::Ptr provider;

    // Look-up CodecId from input params
    // If set then raw data provider is preferred
    GAPI_LOG_DEBUG(nullptr, "try find explicit cfg param \"" <<
                            CfgParam::decoder_id_name() <<"\"");
    auto codec_it =
        std::find_if(cfg_params.begin(), cfg_params.end(), [] (const CfgParam& value) {
            return value.get_name() == CfgParam::decoder_id_name();
        });
    if (codec_it != cfg_params.end()) {
        GAPI_LOG_DEBUG(nullptr, "Dispatcher found \"" << CfgParam::decoder_id_name() << "\""
                                " so try on raw data provider at first");

        try {
            provider = std::make_shared<FileDataProvider>(file_path, cfg_params);
            GAPI_LOG_INFO(nullptr, "raw data provider created");
        } catch (const DataProviderUnsupportedException& ex) {
            GAPI_LOG_INFO(nullptr, "raw data provider creation is failed, reason: " <<
                                    ex.what());
        }
    }

    if (!provider) {
        GAPI_LOG_DEBUG(nullptr, "Try on MFP data provider");
        try {
            provider = std::make_shared<MFPAsyncDemuxDataProvider>(file_path);
            GAPI_LOG_INFO(nullptr, "MFP data provider created");
        } catch (const DataProviderUnsupportedException& ex) {
            GAPI_LOG_INFO(nullptr, "MFP data provider creation is failed, reason: " <<
                                   ex.what());
        }
    }

    // final check
    if (!provider) {
        GAPI_LOG_WARNING(nullptr, "Cannot find suitable data provider");
        throw DataProviderUnsupportedException("Unsupported source or configuration parameters");;
    }
    return provider;
}
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv
#endif // HAVE_ONEVPL
