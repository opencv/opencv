// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifndef OPENCV_GAPI_STREAMING_TESTS_COMMON_HPP
#define OPENCV_GAPI_STREAMING_TESTS_COMMON_HPP

#include "gapi_tests_common.hpp"
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/data_provider_defines.hpp"

#ifdef HAVE_ONEVPL
#include "streaming/onevpl/onevpl_export.hpp"

namespace opencv_test {
namespace streaming {
namespace onevpl {

struct StreamDataProvider : public cv::gapi::wip::onevpl::IDataProvider {

    StreamDataProvider(std::istream& in) : data_stream (in) {
        EXPECT_TRUE(in);
    }

mfx_codec_id_type get_mfx_codec_id() const override {
        return MFX_CODEC_HEVC;
    }

    bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &out_bitstream) override {
        if (empty()) {
            return false;
        }

        if (!out_bitstream) {
            out_bitstream = std::make_shared<mfx_bitstream>();
            out_bitstream->MaxLength = 2000000;
            out_bitstream->Data = (mfxU8 *)calloc(out_bitstream->MaxLength, sizeof(mfxU8));
            if(!out_bitstream->Data) {
                throw std::runtime_error("Cannot allocate bitstream.Data bytes: " +
                                         std::to_string(out_bitstream->MaxLength * sizeof(mfxU8)));
            }
            out_bitstream->CodecId = get_mfx_codec_id();
        }

        mfxU8 *p0 = out_bitstream->Data;
        mfxU8 *p1 = out_bitstream->Data + out_bitstream->DataOffset;
        EXPECT_FALSE(out_bitstream->DataOffset > out_bitstream->MaxLength - 1);
        EXPECT_FALSE(out_bitstream->DataLength + out_bitstream->DataOffset > out_bitstream->MaxLength);

        std::copy_n(p1, out_bitstream->DataLength, p0);

        out_bitstream->DataOffset = 0;
        out_bitstream->DataLength += static_cast<mfxU32>(fetch_data(out_bitstream->MaxLength - out_bitstream->DataLength,
                                                         out_bitstream->Data + out_bitstream->DataLength));
        return out_bitstream->DataLength != 0;
    }

    size_t fetch_data(size_t out_data_size, void* out_data_buf) {
        data_stream.read(reinterpret_cast<char*>(out_data_buf), out_data_size);
        return data_stream.gcount();
    }
    bool empty() const override {
        return data_stream.eof() || data_stream.bad();
    }
private:
    std::istream& data_stream;
};

static const unsigned char hevc_header[] = {
 0x00, 0x00, 0x00, 0x01, 0x40, 0x01, 0x0C, 0x06, 0xFF, 0xFF, 0x01, 0x40, 0x00,
 0x00, 0x03, 0x00, 0x80, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x78, 0x00,
 0x00, 0x04, 0x02, 0x10, 0x30, 0x00, 0x00, 0x03, 0x00, 0x10, 0x00, 0x00, 0x03,
 0x01, 0xE5, 0x00, 0x00, 0x00, 0x01, 0x42, 0x01, 0x06, 0x01, 0x40, 0x00, 0x00,
 0x03, 0x00, 0x80, 0x00, 0x00, 0x03, 0x00, 0x00, 0x03, 0x00, 0x78, 0x00, 0x00,
 0xA0, 0x10, 0x20, 0x61, 0x63, 0x41, 0x00, 0x86, 0x49, 0x1B, 0x2B, 0x20, 0x00,
 0x00, 0x00, 0x01, 0x44, 0x01, 0xC0, 0x71, 0xC0, 0xD9, 0x20, 0x00, 0x00, 0x00,
 0x01, 0x26, 0x01, 0xAF, 0x0C
};
} // namespace onevpl
} // namespace streaming
} // namespace opencv_test
#endif // HAVE_ONEVPL
#endif // OPENCV_GAPI_STREAMING_TESTS_HPP
