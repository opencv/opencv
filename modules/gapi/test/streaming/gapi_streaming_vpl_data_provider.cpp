// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include "../test_precomp.hpp"

#include "../common/gapi_tests_common.hpp"
#include "streaming/onevpl/data_provider_dispatcher.hpp"
#include "streaming/onevpl/file_data_provider.hpp"
#include "streaming/onevpl/demux/mfp_demux_data_provider.hpp"
#include "streaming/onevpl/source_priv.hpp"

namespace opencv_test
{
namespace
{
using source_t = std::string;
using dd_valid_t = bool;
using demux_valid_t = bool;
using dec_valid_t = bool;
using array_element_t =
            std::tuple<source_t, dd_valid_t, demux_valid_t, dec_valid_t>;
array_element_t files[] = {
    array_element_t {"highgui/video/VID00003-20100701-2204.3GP",
                                    false,     true,           false},
    array_element_t {"highgui/video/VID00003-20100701-2204.avi",
                                    false,     true,           false},
    array_element_t {"highgui/video/VID00003-20100701-2204.mpg",
                                    false,     true,           false},
    array_element_t {"highgui/video/VID00003-20100701-2204.wmv",
                                    false,     true,           false},
    array_element_t {"highgui/video/sample_322x242_15frames.yuv420p.libaom-av1.mp4",
                                    true,      true,           true},
    array_element_t {"highgui/video/sample_322x242_15frames.yuv420p.libvpx-vp9.mp4",
                                    true,      true,           true},
    array_element_t {"highgui/video/sample_322x242_15frames.yuv420p.libx264.avi",
                                    true,      true,           true},
    array_element_t {"highgui/video/sample_322x242_15frames.yuv420p.libx264.mp4",
                                    true,      true,           true},
    array_element_t {"highgui/video/sample_322x242_15frames.yuv420p.libx265.mp4",
                                    true,      true,           true},
    array_element_t {"highgui/video/sample_322x242_15frames.yuv420p.mjpeg.mp4",
    /* MFP cannot extract video MJPEG subtype from that */
                                    false,     false,          true},
    array_element_t {"highgui/video/big_buck_bunny.h264",
                                    false,     false,          false},
    array_element_t {"highgui/video/big_buck_bunny.h265",
                                    false,     false,          false}
};

// Read encoded stream from file
mfxStatus read_encoded_stream(mfxBitstream &bs,
                            cv::gapi::wip::onevpl::IDataProvider& data_provider) {
    mfxU8 *p0 = bs.Data;
    mfxU8 *p1 = bs.Data + bs.DataOffset;
    if (bs.DataOffset > bs.MaxLength - 1) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }
    if (bs.DataLength + bs.DataOffset > bs.MaxLength) {
        return MFX_ERR_NOT_ENOUGH_BUFFER;
    }

    std::copy_n(p1, bs.DataLength, p0);

    bs.DataOffset = 0;
    bs.DataLength += static_cast<mfxU32>(data_provider.fetch_data(bs.MaxLength - bs.DataLength,
                                                                   bs.Data + bs.DataLength));
    if (bs.DataLength == 0)
        return MFX_ERR_MORE_DATA;

    return MFX_ERR_NONE;
}


class OneVPLSourceMFPDispatcherTest : public ::testing::TestWithParam<array_element_t> {};

TEST_P(OneVPLSourceMFPDispatcherTest, open_and_decode_file)
{
    if (!initTestDataPathSilent()) {
        throw SkipTestException("env variable OPENCV_TEST_DATA_PATH was not configured");
    }
    using namespace cv::gapi::wip::onevpl;

    source_t path = findDataFile(std::get<0>(GetParam()), false);
    dd_valid_t dd_result = std::get<1>(GetParam());
    dec_valid_t dec_result = std::get<3>(GetParam());

    // open demux source & check format support
    std::unique_ptr<MFPDemuxDataProvider> provider_ptr;
    try {
        provider_ptr.reset(new MFPDemuxDataProvider(path));
    } catch (...) {
        EXPECT_FALSE(dd_result);
        GTEST_SUCCEED();
        return;
    }
    EXPECT_TRUE(dd_result);

    // initialize MFX
    mfxLoader mfx_handle = MFXLoad();

    mfxConfig cfg_inst_0 = MFXCreateConfig(mfx_handle);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = codec_id_to_mfx(provider_ptr->get_codec());
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)"mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
                                                    mfx_param_0), MFX_ERR_NONE);

    // create MFX session
    mfxSession mfx_session{};
    mfxStatus sts = MFXCreateSession(mfx_handle, 0, &mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // create proper bitstream
    mfxBitstream bitstream{};
    const int BITSTREAM_BUFFER_SIZE = 2000000;
    bitstream.MaxLength = BITSTREAM_BUFFER_SIZE;
    bitstream.CodecId = mfx_param_0.Data.U32;
    bitstream.Data = (mfxU8 *)calloc(bitstream.MaxLength, sizeof(mfxU8));
    EXPECT_TRUE(bitstream.Data);

    // prepare dec params
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = bitstream.CodecId;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    do {
        sts = read_encoded_stream(bitstream, *provider_ptr);
        EXPECT_TRUE(MFX_ERR_NONE == sts || MFX_ERR_MORE_DATA == sts);
        sts = MFXVideoDECODE_DecodeHeader(mfx_session, &bitstream, &mfxDecParams);
        EXPECT_TRUE(MFX_ERR_NONE == sts || MFX_ERR_MORE_DATA == sts);
    } while (sts == MFX_ERR_MORE_DATA && !provider_ptr->empty());

    if (dec_result) {
        EXPECT_EQ(MFX_ERR_NONE, sts);
    } else {
        EXPECT_FALSE(MFX_ERR_NONE == sts);
    }

    MFXVideoDECODE_Close(mfx_session);
    MFXClose(mfx_session);
    MFXUnload(mfx_handle);
}


TEST_P(OneVPLSourceMFPDispatcherTest, choose_dmux_provider)
{
    if (!initTestDataPathSilent()) {
        throw SkipTestException("env variable OPENCV_TEST_DATA_PATH was not configured");
    }
    using namespace cv::gapi::wip::onevpl;


    source_t path = findDataFile(std::get<0>(GetParam()), false);
    dd_valid_t dd_result = std::get<1>(GetParam());

    std::shared_ptr<IDataProvider> provider_ptr;

    // choose demux provider for empty CfgParams
    try {
        provider_ptr = DataProviderDispatcher::create(path);
    } catch (...) {
        EXPECT_FALSE(dd_result);
        provider_ptr = DataProviderDispatcher::create(path,
                                { CfgParam::create<std::string>(
                                            "mfxImplDescription.mfxDecoderDescription.decoder.CodecID",
                                            "MFX_CODEC_HEVC") /* Doesn't matter what codec for RAW here*/});
        EXPECT_TRUE(std::dynamic_pointer_cast<FileDataProvider>(provider_ptr));
        GTEST_SUCCEED();
        return;
    }

    EXPECT_TRUE(dd_result);
    EXPECT_TRUE(std::dynamic_pointer_cast<MFPDemuxDataProvider>(provider_ptr));
}

INSTANTIATE_TEST_CASE_P(MFP_VPL_DecodeHeaderTests, OneVPLSourceMFPDispatcherTest,
                        testing::ValuesIn(files));
}
} // namespace opencv_test

#endif // HAVE_ONEVPL
