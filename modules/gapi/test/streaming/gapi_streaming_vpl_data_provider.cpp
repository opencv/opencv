// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include <future>

#include "../test_precomp.hpp"

#include "../common/gapi_tests_common.hpp"
#include "streaming/onevpl/data_provider_dispatcher.hpp"
#include "streaming/onevpl/file_data_provider.hpp"
#include "streaming/onevpl/demux/async_mfp_demux_data_provider.hpp"
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

class OneVPL_Source_MFPAsyncDispatcherTest : public ::testing::TestWithParam<array_element_t> {};
TEST_P(OneVPL_Source_MFPAsyncDispatcherTest, open_and_decode_file)
{
    using namespace cv::gapi::wip::onevpl;

    source_t path = findDataFile(std::get<0>(GetParam()));
    dd_valid_t dd_result = std::get<1>(GetParam());
    dec_valid_t dec_result = std::get<3>(GetParam());

    // open demux source & check format support
    std::unique_ptr<MFPAsyncDemuxDataProvider> provider_ptr;
    try {
        provider_ptr.reset(new MFPAsyncDemuxDataProvider(path));
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
    mfx_param_0.Data.U32 = provider_ptr->get_mfx_codec_id();
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)CfgParam::decoder_id_name(),
                                                    mfx_param_0), MFX_ERR_NONE);

    // create MFX session
    mfxSession mfx_session{};
    mfxStatus sts = MFXCreateSession(mfx_handle, 0, &mfx_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // create proper bitstream
    std::shared_ptr<IDataProvider::mfx_bitstream> bitstream{};

    // prepare dec params
    mfxVideoParam mfxDecParams {};
    mfxDecParams.mfx.CodecId = mfx_param_0.Data.U32;
    mfxDecParams.IOPattern = MFX_IOPATTERN_OUT_SYSTEM_MEMORY;
    do {
        bool fetched = provider_ptr->fetch_bitstream_data(bitstream);
        if (dec_result) {
            EXPECT_TRUE(fetched);
        }
        sts = MFXVideoDECODE_DecodeHeader(mfx_session, bitstream.get(), &mfxDecParams);
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


TEST_P(OneVPL_Source_MFPAsyncDispatcherTest, choose_dmux_provider)
{
    using namespace cv::gapi::wip::onevpl;


    source_t path = findDataFile(std::get<0>(GetParam()));
    dd_valid_t dd_result = std::get<1>(GetParam());

    std::shared_ptr<IDataProvider> provider_ptr;

    // choose demux provider for empty CfgParams
    try {
        provider_ptr = DataProviderDispatcher::create(path);
    } catch (...) {
        EXPECT_FALSE(dd_result);
        provider_ptr = DataProviderDispatcher::create(path,
                                { CfgParam::create<std::string>(
                                            CfgParam::decoder_id_name(),
                                            "MFX_CODEC_HEVC") /* Doesn't matter what codec for RAW here*/});
        EXPECT_TRUE(std::dynamic_pointer_cast<FileDataProvider>(provider_ptr));
        GTEST_SUCCEED();
        return;
    }

    EXPECT_TRUE(dd_result);
    EXPECT_TRUE(std::dynamic_pointer_cast<MFPAsyncDemuxDataProvider>(provider_ptr));
}

INSTANTIATE_TEST_CASE_P(MFP_VPL_DecodeHeaderTests, OneVPL_Source_MFPAsyncDispatcherTest,
                        testing::ValuesIn(files));

namespace test {
    struct IntrusiveAsyncDemuxDataProvider :
            public cv::gapi::wip::onevpl::MFPAsyncDemuxDataProvider {

    using base_t = cv::gapi::wip::onevpl::MFPAsyncDemuxDataProvider;
    using base_t::base_t;

    ~IntrusiveAsyncDemuxDataProvider() {
        destroyed = true;
    }

    STDMETHODIMP OnReadSample(HRESULT status, DWORD stream_index,
                              DWORD stream_flag, LONGLONG timestamp,
                              IMFSample *sample_ptr) override {
        if (IntrusiveAsyncDemuxDataProvider::need_request_next) {
            return base_t::OnReadSample(status, stream_index, stream_flag,
                                        timestamp, sample_ptr);
        }
        return status;
    }


    // implementation methods
    size_t produce_worker_data(void *key,
                               cv::gapi::wip::onevpl::ComPtrGuard<IMFMediaBuffer> &&buffer,
                               std::shared_ptr<mfx_bitstream> &&staging_stream) override {
        return base_t::produce_worker_data(key, std::move(buffer),
                                           std::move(staging_stream));
    }

    static bool need_request_next;
    static bool destroyed;
};

bool IntrusiveAsyncDemuxDataProvider::need_request_next{};
bool IntrusiveAsyncDemuxDataProvider::destroyed{};
} // namespace test

TEST(OneVPL_Source_MFPAsyncDemux, sync_flush) {
    using namespace cv::gapi::wip::onevpl;

    source_t path = findDataFile("highgui/video/sample_322x242_15frames.yuv420p.libx265.mp4");
    test::IntrusiveAsyncDemuxDataProvider::need_request_next = false;
    const size_t preprocessed_samples_count = 3;
    {
        test::IntrusiveAsyncDemuxDataProvider provider(path, preprocessed_samples_count);
        size_t produce_buffer_count = 199 * preprocessed_samples_count;
        std::thread producer([&provider, produce_buffer_count]() {
            size_t total_produced_count = 0;
            for (size_t i = 0; i < produce_buffer_count; i ++) {
                total_produced_count += provider.produce_worker_data(
                                                    reinterpret_cast<void*>(i),
                                                    createCOMPtrGuard<IMFMediaBuffer>(nullptr),
                                                    {});
            }
        });
        producer.join();
    }

    EXPECT_EQ(test::IntrusiveAsyncDemuxDataProvider::destroyed, true);
}

TEST(OneVPL_Source_MFPAsyncDemux, async_flush) {
    using namespace cv::gapi::wip::onevpl;

    source_t path = findDataFile("highgui/video/sample_322x242_15frames.yuv420p.libx265.mp4");
    test::IntrusiveAsyncDemuxDataProvider::need_request_next = true;
    const size_t preprocessed_samples_count = 999;
    {
        std::shared_ptr<IDataProvider::mfx_bitstream> stream;
        test::IntrusiveAsyncDemuxDataProvider provider(path, preprocessed_samples_count);
        EXPECT_TRUE(provider.fetch_bitstream_data(stream));
        EXPECT_TRUE(stream);
    }

    EXPECT_EQ(test::IntrusiveAsyncDemuxDataProvider::destroyed, true);
}

TEST(OneVPL_Source_MFPAsyncDemux, eof_async_detection) {
    using namespace cv::gapi::wip::onevpl;

    source_t path = findDataFile("highgui/video/sample_322x242_15frames.yuv420p.libx265.mp4");
    test::IntrusiveAsyncDemuxDataProvider::need_request_next = false;
    const size_t preprocessed_samples_count = 0; // do not ask sample at start
    test::IntrusiveAsyncDemuxDataProvider provider(path, preprocessed_samples_count);
    std::promise<void> start_consume_data;
    std::future<void> wait_consume_data = start_consume_data.get_future();

    std::thread fetcher([&provider, &start_consume_data]() {
        std::shared_ptr<IDataProvider::mfx_bitstream> stream;
        start_consume_data.set_value();
        EXPECT_FALSE(provider.fetch_bitstream_data(stream));
        EXPECT_FALSE(stream);
    });

    wait_consume_data.wait();
    std::this_thread::sleep_for(std::chrono::seconds(2));   // hope fetched has slept on condition

    test::IntrusiveAsyncDemuxDataProvider::need_request_next = true;
    provider.OnReadSample(S_OK, 0, MF_SOURCE_READERF_ENDOFSTREAM, 0, nullptr);
    fetcher.join();
}

TEST(OneVPL_Source_MFPAsyncDemux, produce_consume) {
    using namespace cv::gapi::wip::onevpl;

    source_t path = findDataFile("highgui/video/sample_322x242_15frames.yuv420p.libx265.mp4");
    test::IntrusiveAsyncDemuxDataProvider::need_request_next = false;
    const size_t preprocessed_samples_count = 3;
    test::IntrusiveAsyncDemuxDataProvider provider(path, preprocessed_samples_count);

    std::promise<void> start_consume_data;
    std::future<void> wait_consume_data = start_consume_data.get_future();
    size_t produce_buffer_count = 199 * preprocessed_samples_count;
    std::thread producer([&provider, &wait_consume_data, produce_buffer_count]() {
        wait_consume_data.wait();
        size_t total_produced_count = 0;
        for (size_t i = 0; i < produce_buffer_count; i ++) {
            std::shared_ptr<IDataProvider::mfx_bitstream> dummy_stream =
                                std::make_shared<IDataProvider::mfx_bitstream>();
            dummy_stream->DataLength = static_cast<mfxU32>(i); // control block
            dummy_stream->DataOffset = static_cast<mfxU32>(i); // control block
            dummy_stream->Data = reinterpret_cast<mfxU8*>(i);
            total_produced_count = provider.produce_worker_data(
                                                    dummy_stream->Data,
                                                    createCOMPtrGuard<IMFMediaBuffer>(nullptr),
                                                    std::move(dummy_stream));
            EXPECT_TRUE(total_produced_count <= produce_buffer_count);
        }
    });

    std::thread consumer([&provider, &start_consume_data, produce_buffer_count]() {

        start_consume_data.set_value();
        size_t total_consumed_count = 0;
        std::shared_ptr<IDataProvider::mfx_bitstream> dummy_stream;
        size_t stream_idx = 0;
        do {
            EXPECT_TRUE(provider.fetch_bitstream_data(dummy_stream));
            EXPECT_TRUE(dummy_stream);
            EXPECT_EQ(dummy_stream->DataLength, stream_idx);
            stream_idx ++;
            total_consumed_count++;
        } while (total_consumed_count != produce_buffer_count);
    });

    producer.join();
    consumer.join();
}
}
} // namespace opencv_test

#endif // HAVE_ONEVPL
