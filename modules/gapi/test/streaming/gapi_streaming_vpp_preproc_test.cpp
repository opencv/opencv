// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2022 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_tests_common.hpp"
#include "../common/gapi_streaming_tests_common.hpp"

#include <chrono>
#include <future>
#include <tuple>

#include <opencv2/gapi/media.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#include <opencv2/gapi/ocl/core.hpp>
#include <opencv2/gapi/ocl/imgproc.hpp>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/streaming/desync.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#ifdef HAVE_ONEVPL

#include <opencv2/gapi/streaming/onevpl/data_provider_interface.hpp>
#include "streaming/onevpl/file_data_provider.hpp"
#include "streaming/onevpl/cfg_param_device_selector.hpp"

#include "streaming/onevpl/accelerators/surface/surface.hpp"
#include "streaming/onevpl/accelerators/surface/cpu_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/surface/dx11_frame_adapter.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"
#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#define private public
#define protected public
#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/engine/decode/decode_session.hpp"

#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"

#include "streaming/onevpl/engine/transcode/transcode_engine_legacy.hpp"
#include "streaming/onevpl/engine/transcode/transcode_session.hpp"
#undef protected
#undef private
#include "logger.hpp"

#define ALIGN16(value)           (((value + 15) >> 4) << 4)

namespace opencv_test
{
namespace
{
template<class ProcessingEngine>
cv::MediaFrame extract_decoded_frame(mfxSession sessId, ProcessingEngine& engine) {
    using namespace cv::gapi::wip::onevpl;
    ProcessingEngineBase::ExecutionStatus status = ProcessingEngineBase::ExecutionStatus::Continue;
    while (0 == engine.get_ready_frames_count() &&
           status == ProcessingEngineBase::ExecutionStatus::Continue) {
        status = engine.process(sessId);
    }

    if (engine.get_ready_frames_count() == 0) {
        GAPI_LOG_WARNING(nullptr, "failed: cannot obtain preprocessed frames, last status: " <<
                                  ProcessingEngineBase::status_to_string(status));
        throw std::runtime_error("cannot finalize VPP preprocessing operation");
    }
    cv::gapi::wip::Data data;
    engine.get_frame(data);
    return cv::util::get<cv::MediaFrame>(data);
}

std::tuple<mfxLoader, mfxConfig> prepare_mfx(int mfx_codec, int mfx_accel_mode) {
    using namespace cv::gapi::wip::onevpl;
    mfxLoader mfx = MFXLoad();
    mfxConfig cfg_inst_0 = MFXCreateConfig(mfx);
    EXPECT_TRUE(cfg_inst_0);
    mfxVariant mfx_param_0;
    mfx_param_0.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_0.Data.U32 = MFX_IMPL_TYPE_HARDWARE;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_0,(mfxU8 *)CfgParam::implementation_name(),
                                                    mfx_param_0), MFX_ERR_NONE);

    mfxConfig cfg_inst_1 = MFXCreateConfig(mfx);
    EXPECT_TRUE(cfg_inst_1);
    mfxVariant mfx_param_1;
    mfx_param_1.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_1.Data.U32 = mfx_accel_mode;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_1,(mfxU8 *)CfgParam::acceleration_mode_name(),
                                                    mfx_param_1), MFX_ERR_NONE);

    mfxConfig cfg_inst_2 = MFXCreateConfig(mfx);
    EXPECT_TRUE(cfg_inst_2);
    mfxVariant mfx_param_2;
    mfx_param_2.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_2.Data.U32 = mfx_codec;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_2,(mfxU8 *)CfgParam::decoder_id_name(),
                                                    mfx_param_2), MFX_ERR_NONE);

    mfxConfig cfg_inst_3 = MFXCreateConfig(mfx);
    EXPECT_TRUE(cfg_inst_3);
    mfxVariant mfx_param_3;
    mfx_param_3.Type = MFX_VARIANT_TYPE_U32;
    mfx_param_3.Data.U32 = MFX_EXTBUFF_VPP_SCALING;
    EXPECT_EQ(MFXSetConfigFilterProperty(cfg_inst_3,
                                         (mfxU8 *)"mfxImplDescription.mfxVPPDescription.filter.FilterFourCC",
                                         mfx_param_3), MFX_ERR_NONE);
    return std::make_tuple(mfx, cfg_inst_3);
}

class SafeQueue {
public:
    void push(cv::MediaFrame&& f) {
        std::unique_lock<std::mutex> lock(mutex);
        queue.push(std::move(f));
        cv.notify_all();
    }

    cv::MediaFrame pop() {
        cv::MediaFrame ret;
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [this] () {
            return !queue.empty();
        });
        ret = queue.front();
        queue.pop();
        return ret;
    }

    void push_stop() {
        push(cv::MediaFrame::Create<IStopAdapter>());
    }

    static bool is_stop(const cv::MediaFrame &f) {
        try {
            return f.get<IStopAdapter>();
        } catch(...) {}
        return false;
    }

private:
    struct IStopAdapter final : public cv::MediaFrame::IAdapter {
        ~IStopAdapter() {}
        cv::GFrameDesc meta() const { return {}; };
        MediaFrame::View access(MediaFrame::Access) { return {{}, {}}; };
    };
private:
    std::condition_variable cv;
    std::mutex mutex;
    std::queue<cv::MediaFrame> queue;
};

struct EmptyDataProvider : public cv::gapi::wip::onevpl::IDataProvider {

    bool empty() const override {
        return true;
    }
    mfx_codec_id_type get_mfx_codec_id() const override {
        return std::numeric_limits<uint32_t>::max();
    }
    bool fetch_bitstream_data(std::shared_ptr<mfx_bitstream> &) override {
        return false;
    }
};
}

using source_t          = std::string;
using decoder_t         = int;
using acceleration_t    = int;
using out_frame_info_t  = cv::GFrameDesc;
using preproc_args_t    = std::tuple<source_t, decoder_t, acceleration_t, out_frame_info_t>;

class VPPPreprocParams : public ::testing::TestWithParam<preproc_args_t> {};

preproc_args_t files[] = {
    preproc_args_t {"highgui/video/big_buck_bunny.h264",
                    MFX_CODEC_AVC,     MFX_ACCEL_MODE_VIA_D3D11,
                    cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1080}}},
    preproc_args_t {"highgui/video/big_buck_bunny.h265",
                    MFX_CODEC_HEVC,     MFX_ACCEL_MODE_VIA_D3D11,
                    cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1280}}}
};

#ifdef HAVE_DIRECTX
#ifdef HAVE_D3D11
TEST(OneVPL_Source_PreprocEngine, functional_single_thread)
{
    using namespace cv::gapi::wip::onevpl;
    using namespace cv::gapi::wip;

    std::vector<CfgParam> cfg_params_w_dx11;
    cfg_params_w_dx11.push_back(CfgParam::create_acceleration_mode(MFX_ACCEL_MODE_VIA_D3D11));
    std::unique_ptr<VPLAccelerationPolicy> decode_accel_policy (
                    new VPLDX11AccelerationPolicy(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_dx11)));

    // create file data provider
    std::string file_path = findDataFile("highgui/video/big_buck_bunny.h265");
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                    {CfgParam::create_decoder_id(MFX_CODEC_HEVC)}));

    mfxLoader mfx{};
    mfxConfig mfx_cfg{};
    std::tie(mfx, mfx_cfg) = prepare_mfx(MFX_CODEC_HEVC, MFX_ACCEL_MODE_VIA_D3D11);

    // create decode session
    mfxSession mfx_decode_session{};
    mfxStatus sts = MFXCreateSession(mfx, 0, &mfx_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // create decode engine
    auto device_selector = decode_accel_policy->get_device_selector();
    VPLLegacyDecodeEngine decode_engine(std::move(decode_accel_policy));
    auto sess_ptr = decode_engine.initialize_session(mfx_decode_session,
                                                     cfg_params_w_dx11,
                                                     data_provider);

    // simulate net info
    cv::GFrameDesc required_frame_param {cv::MediaFormat::NV12,
                                         {1920, 1080}};

    // create VPP preproc engine
    VPPPreprocEngine preproc_engine(std::unique_ptr<VPLAccelerationPolicy>{
                                                new VPLDX11AccelerationPolicy(device_selector)});

    // launch pipeline
    // 1) decode frame
    cv::MediaFrame first_decoded_frame;
    ASSERT_NO_THROW(first_decoded_frame = extract_decoded_frame(sess_ptr->session, decode_engine));
    cv::GFrameDesc first_frame_decoded_desc = first_decoded_frame.desc();

    // 1.5) create preproc session based on frame description & network info
    cv::util::optional<pp_params> first_pp_params = preproc_engine.is_applicable(first_decoded_frame);
    ASSERT_TRUE(first_pp_params.has_value());
    pp_session first_pp_sess = preproc_engine.initialize_preproc(first_pp_params.value(),
                                                                 required_frame_param);

    // 2) make preproc using incoming decoded frame & preproc session
    cv::MediaFrame first_pp_frame = preproc_engine.run_sync(first_pp_sess, first_decoded_frame);
    cv::GFrameDesc first_outcome_pp_desc = first_pp_frame.desc();
    ASSERT_FALSE(first_frame_decoded_desc == first_outcome_pp_desc);

    // do not hold media frames because they share limited DX11 surface pool resources
    first_decoded_frame = cv::MediaFrame();
    first_pp_frame = cv::MediaFrame();

    // make test in loop
    bool in_progress = false;
    size_t frames_processed_count = 1;
    const auto &first_pp_param_value_impl =
        cv::util::get<cv::gapi::wip::onevpl::vpp_pp_params>(first_pp_params.value().value);
    try {
        while(true) {
            cv::MediaFrame decoded_frame = extract_decoded_frame(sess_ptr->session, decode_engine);
            in_progress = true;
            ASSERT_EQ(decoded_frame.desc(), first_frame_decoded_desc);

            cv::util::optional<pp_params> params = preproc_engine.is_applicable(decoded_frame);
            ASSERT_TRUE(params.has_value());
            const auto &cur_pp_param_value_impl =
                cv::util::get<cv::gapi::wip::onevpl::vpp_pp_params>(params.value().value);

            ASSERT_EQ(first_pp_param_value_impl.handle, cur_pp_param_value_impl.handle);
            ASSERT_TRUE(FrameInfoComparator::equal_to(first_pp_param_value_impl.info, cur_pp_param_value_impl.info));

            pp_session pp_sess = preproc_engine.initialize_preproc(params.value(),
                                                                   required_frame_param);
            ASSERT_EQ(pp_sess.get<EngineSession>().get(),
                      first_pp_sess.get<EngineSession>().get());

            cv::MediaFrame pp_frame = preproc_engine.run_sync(pp_sess, decoded_frame);
            cv::GFrameDesc pp_desc = pp_frame.desc();
            ASSERT_TRUE(pp_desc == first_outcome_pp_desc);
            in_progress = false;
            frames_processed_count++;
        }
    } catch (...) {}

    // test if interruption has happened
    ASSERT_FALSE(in_progress);
    ASSERT_NE(frames_processed_count, 1);
}


TEST_P(VPPPreprocParams, functional_different_threads)
{
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;
    source_t file_path;
    decoder_t decoder_id;
    acceleration_t accel;
    out_frame_info_t required_frame_param;
    std::tie(file_path, decoder_id, accel, required_frame_param) = GetParam();

    file_path = findDataFile(file_path);

    std::vector<CfgParam> cfg_params_w_dx11;
    cfg_params_w_dx11.push_back(CfgParam::create_acceleration_mode(accel));
    std::unique_ptr<VPLAccelerationPolicy> decode_accel_policy (
                    new VPLDX11AccelerationPolicy(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_dx11)));

    // create file data provider
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                    {CfgParam::create_decoder_id(decoder_id)}));

    mfxLoader mfx{};
    mfxConfig mfx_cfg{};
    std::tie(mfx, mfx_cfg) = prepare_mfx(decoder_id, accel);

    // create decode session
    mfxSession mfx_decode_session{};
    mfxStatus sts = MFXCreateSession(mfx, 0, &mfx_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // create decode engine
    auto device_selector = decode_accel_policy->get_device_selector();
    VPLLegacyDecodeEngine decode_engine(std::move(decode_accel_policy));
    auto sess_ptr = decode_engine.initialize_session(mfx_decode_session,
                                                     cfg_params_w_dx11,
                                                     data_provider);

    // create VPP preproc engine
    VPPPreprocEngine preproc_engine(std::unique_ptr<VPLAccelerationPolicy>{
                                                new VPLDX11AccelerationPolicy(device_selector)});

    // launch threads
    SafeQueue queue;
    size_t decoded_number = 1;
    size_t preproc_number = 0;

    std::thread decode_thread([&decode_engine, sess_ptr,
                               &queue, &decoded_number] () {
        // decode first frame
        {
            cv::MediaFrame decoded_frame;
            ASSERT_NO_THROW(decoded_frame = extract_decoded_frame(sess_ptr->session, decode_engine));
            queue.push(std::move(decoded_frame));
        }

        // launch pipeline
        try {
            while(true) {
                queue.push(extract_decoded_frame(sess_ptr->session, decode_engine));
                decoded_number++;
            }
        } catch (...) {}

        // send stop
        queue.push_stop();
    });

    std::thread preproc_thread([&preproc_engine, &queue, &preproc_number, required_frame_param] () {
        // create preproc session based on frame description & network info
        cv::MediaFrame first_decoded_frame = queue.pop();
        cv::util::optional<pp_params> first_pp_params = preproc_engine.is_applicable(first_decoded_frame);
        ASSERT_TRUE(first_pp_params.has_value());
        pp_session first_pp_sess =
                    preproc_engine.initialize_preproc(first_pp_params.value(), required_frame_param);

        // make preproc using incoming decoded frame & preproc session
        cv::MediaFrame first_pp_frame = preproc_engine.run_sync(first_pp_sess, first_decoded_frame);
        cv::GFrameDesc first_outcome_pp_desc = first_pp_frame.desc();

        // do not hold media frames because they share limited DX11 surface pool resources
        first_decoded_frame = cv::MediaFrame();
        first_pp_frame = cv::MediaFrame();

        // launch pipeline
        bool in_progress = false;
        // let's allow counting of preprocessed frames to check this value later:
        // Currently, it looks redundant to implement any kind of gracefull shutdown logic
        // in this test - so let's apply agreement that media source is processed
        // succesfully when preproc_number != 1 in result
        preproc_number = 1;
        try {
            while(true) {
                cv::MediaFrame decoded_frame = queue.pop();
                if (SafeQueue::is_stop(decoded_frame)) {
                    break;
                }
                in_progress = true;

                cv::util::optional<pp_params> params = preproc_engine.is_applicable(decoded_frame);
                ASSERT_TRUE(params.has_value());
                ASSERT_TRUE(0 == memcmp(&params.value(), &first_pp_params.value(), sizeof(pp_params)));

                pp_session pp_sess = preproc_engine.initialize_preproc(params.value(),
                                                                       required_frame_param);
                ASSERT_EQ(pp_sess.get<EngineSession>().get(),
                          first_pp_sess.get<EngineSession>().get());

                cv::MediaFrame pp_frame = preproc_engine.run_sync(pp_sess, decoded_frame);
                cv::GFrameDesc pp_desc = pp_frame.desc();
                ASSERT_TRUE(pp_desc == first_outcome_pp_desc);
                in_progress = false;
                preproc_number++;
            }
        } catch (...) {}

        // test if interruption has happened
        ASSERT_FALSE(in_progress);
        ASSERT_NE(preproc_number, 1);
    });

    decode_thread.join();
    preproc_thread.join();
    ASSERT_EQ(preproc_number, decoded_number);
}

INSTANTIATE_TEST_CASE_P(OneVPL_Source_PreprocEngine, VPPPreprocParams,
                        testing::ValuesIn(files));

using VPPInnerPreprocParams = VPPPreprocParams;
TEST_P(VPPInnerPreprocParams, functional_inner_preproc_size)
{
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;
    source_t file_path;
    decoder_t decoder_id;
    acceleration_t accel;
    out_frame_info_t required_frame_param;
    std::tie(file_path, decoder_id, accel, required_frame_param) = GetParam();

    file_path = findDataFile(file_path);

    std::vector<CfgParam> cfg_params_w_dx11_vpp;

    // create accel policy
    cfg_params_w_dx11_vpp.push_back(CfgParam::create_acceleration_mode(accel));
    std::unique_ptr<VPLAccelerationPolicy> accel_policy (
                    new VPLDX11AccelerationPolicy(std::make_shared<CfgParamDeviceSelector>(cfg_params_w_dx11_vpp)));

    // create file data provider
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                    {CfgParam::create_decoder_id(decoder_id)}));

    // create decode session
    mfxLoader mfx{};
    mfxConfig mfx_cfg{};
    std::tie(mfx, mfx_cfg) = prepare_mfx(decoder_id, accel);

    mfxSession mfx_decode_session{};
    mfxStatus sts = MFXCreateSession(mfx, 0, &mfx_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // fill vpp params beforehand: resolution
    cfg_params_w_dx11_vpp.push_back(CfgParam::create_vpp_out_width(
                                        static_cast<uint16_t>(required_frame_param.size.width)));
    cfg_params_w_dx11_vpp.push_back(CfgParam::create_vpp_out_height(
                                        static_cast<uint16_t>(required_frame_param.size.height)));

    // create transcode engine
    auto device_selector = accel_policy->get_device_selector();
    VPLLegacyTranscodeEngine engine(std::move(accel_policy));
    auto sess_ptr = engine.initialize_session(mfx_decode_session,
                                              cfg_params_w_dx11_vpp,
                                              data_provider);
   // make test in loop
    bool in_progress = false;
    size_t frames_processed_count = 1;
    try {
        while(true) {
            cv::MediaFrame decoded_frame = extract_decoded_frame(sess_ptr->session, engine);
            in_progress = true;
            ASSERT_EQ(decoded_frame.desc().size.width,
                      ALIGN16(required_frame_param.size.width));
            ASSERT_EQ(decoded_frame.desc().size.height,
                      ALIGN16(required_frame_param.size.height));
            ASSERT_EQ(decoded_frame.desc().fmt, required_frame_param.fmt);
            frames_processed_count++;
            in_progress = false;
        }
    } catch (...) {}

    // test if interruption has happened
    ASSERT_FALSE(in_progress);
    ASSERT_NE(frames_processed_count, 1);
}

INSTANTIATE_TEST_CASE_P(OneVPL_Source_PreprocInner, VPPInnerPreprocParams,
                        testing::ValuesIn(files));
#endif // HAVE_DIRECTX
#endif // HAVE_D3D11
} // namespace opencv_test
#endif // HAVE_ONEVPL
