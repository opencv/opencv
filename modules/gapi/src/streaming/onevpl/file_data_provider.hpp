// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
#define GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP

#include <stdio.h>

#ifdef HAVE_ONEVPL
#include <vpl/mfxvideo.h>
#endif // HAVE_ONEVPL

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include <opencv2/gapi/streaming/onevpl/cfg_params.hpp>

namespace cv {
namespace gapi {
namespace wip {
namespace onevpl {
struct FileDataProvider : public IDataProvider {

    using file_ptr = std::unique_ptr<FILE, decltype(&fclose)>;
    FileDataProvider(const std::string& file_path,
                     const std::vector<CfgParam> codec_params = {},
                     size_t bitstream_data_size_value = 2000000);
    ~FileDataProvider();

    CodecID get_codec() const override;

#ifdef HAVE_ONEVPL
    mfxStatus fetch_bitstream_data(std::shared_ptr<mfxBitstream> &out_bitsream) override;
#endif // HAVE_ONEVPL

    bool empty() const override;
private:
    file_ptr source_handle;
    CodecID codec;
    const size_t bitstream_data_size;
};
} // namespace onevpl
} // namespace wip
} // namespace gapi
} // namespace cv

#endif // GAPI_STREAMING_ONEVPL_ONEVPL_FILE_DATA_PROVIDER_HPP
