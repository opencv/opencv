// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP

#include <stdio.h>

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>

#include "streaming/onevpl/data_provider_defines.hpp"

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct FileDataProvider : public IDataProvider {

    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;
    FileDataProvider(const std::string& file_path,
                     const std::vector<CfgParam> &codec_params = {},
                     uint32_t bitstream_data_size_value = 2000000);
    ~FileDataProvider();

    mfx_codec_id_type get_mfx_codec_id() const override;
    bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &out_bitsream) override;
    bool empty() const override;
private:
    file_ptr source_handle;
    mfx_codec_id_type codec;
    const uint32_t bitstream_data_size;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
