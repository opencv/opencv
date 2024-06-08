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
#include "streaming/onevpl/accelerators/accel_policy_va_api.hpp"
#include "streaming/onevpl/accelerators/dx11_alloc_resource.hpp"
#include "streaming/onevpl/accelerators/utils/shared_lock.hpp"
#define private public
#define protected public
#include "streaming/onevpl/engine/decode/decode_engine_legacy.hpp"
#include "streaming/onevpl/engine/decode/decode_session.hpp"

#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"
#include "streaming/onevpl/engine/preproc/preproc_dispatcher.hpp"

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

static std::unique_ptr<cv::gapi::wip::onevpl::VPLAccelerationPolicy>
create_accel_policy_from_int(int accel,
                             std::shared_ptr<cv::gapi::wip::onevpl::IDeviceSelector> selector) {
    using namespace cv::gapi::wip::onevpl;
    std::unique_ptr<VPLAccelerationPolicy> decode_accel_policy;
    if (accel == MFX_ACCEL_MODE_VIA_D3D11) {
        decode_accel_policy.reset (new VPLDX11AccelerationPolicy(selector));
    } else if (accel == MFX_ACCEL_MODE_VIA_VAAPI) {
        decode_accel_policy.reset (new VPLVAAPIAccelerationPolicy(selector));
    }
    EXPECT_TRUE(decode_accel_policy.get());
    return decode_accel_policy;
}

static std::unique_ptr<cv::gapi::wip::onevpl::VPLAccelerationPolicy>
create_accel_policy_from_int(int &accel,
                             std::vector<cv::gapi::wip::onevpl::CfgParam> &out_cfg_params) {
    using namespace cv::gapi::wip::onevpl;
    out_cfg_params.push_back(CfgParam::create_acceleration_mode(accel));
    return create_accel_policy_from_int(accel, std::make_shared<CfgParamDeviceSelector>(out_cfg_params));
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

static cv::util::optional<cv::Rect> empty_roi;

class VPPPreprocParams : public ::testing::TestWithParam<preproc_args_t> {};

#if defined(HAVE_DIRECTX) && defined(HAVE_D3D11)
    #define UT_ACCEL_TYPE MFX_ACCEL_MODE_VIA_D3D11
#elif __linux__
    #define UT_ACCEL_TYPE MFX_ACCEL_MODE_VIA_VAAPI
#else
    #define UT_ACCEL_TYPE -1
#endif

preproc_args_t files[] = {
    preproc_args_t {"highgui/video/big_buck_bunny.h264",
                    MFX_CODEC_AVC,     UT_ACCEL_TYPE,
                    cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1080}}},
    preproc_args_t {"highgui/video/big_buck_bunny.h265",
                    MFX_CODEC_HEVC,     UT_ACCEL_TYPE,
                    cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1280}}}
};

class OneVPL_PreproEngineTest : public ::testing::TestWithParam<acceleration_t> {};
TEST_P(OneVPL_PreproEngineTest, functional_single_thread)
{
    using namespace cv::gapi::wip::onevpl;
    using namespace cv::gapi::wip;

    int accel_type = GetParam();
    std::vector<CfgParam> cfg_params_w_accel;
    std::unique_ptr<VPLAccelerationPolicy> decode_accel_policy = create_accel_policy_from_int(accel_type, cfg_params_w_accel);

    // create file data provider
    std::string file_path = findDataFile("highgui/video/big_buck_bunny.h265");
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                    {CfgParam::create_decoder_id(MFX_CODEC_HEVC)}));

    mfxLoader mfx{};
    mfxConfig mfx_cfg{};
    std::tie(mfx, mfx_cfg) = prepare_mfx(MFX_CODEC_HEVC, accel_type);

    // create decode session
    mfxSession mfx_decode_session{};
    mfxStatus sts = MFXCreateSession(mfx, 0, &mfx_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // create decode engine
    auto device_selector = decode_accel_policy->get_device_selector();
    VPLLegacyDecodeEngine decode_engine(std::move(decode_accel_policy));
    auto sess_ptr = decode_engine.initialize_session(mfx_decode_session,
                                                     cfg_params_w_accel,
                                                     data_provider);

    // simulate net info
    cv::GFrameDesc required_frame_param {cv::MediaFormat::NV12,
                                         {1920, 1080}};

    // create VPP preproc engine
    VPPPreprocEngine preproc_engine(create_accel_policy_from_int(accel_type, device_selector));

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
    cv::MediaFrame first_pp_frame = preproc_engine.run_sync(first_pp_sess,
                                                            first_decoded_frame,
                                                            empty_roi);
    cv::GFrameDesc first_outcome_pp_desc = first_pp_frame.desc();
    ASSERT_FALSE(first_frame_decoded_desc == first_outcome_pp_desc);

    // do not hold media frames because they share limited DX11 surface pool resources
    first_decoded_frame = cv::MediaFrame();
    first_pp_frame = cv::MediaFrame();

    // make test in loop
    bool in_progress = false;
    int frames_processed_count = 1;
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
            ASSERT_EQ(pp_sess.get<vpp_pp_session>().handle.get(),
                      first_pp_sess.get<vpp_pp_session>().handle.get());

            cv::MediaFrame pp_frame = preproc_engine.run_sync(pp_sess,
                                                              decoded_frame,
                                                              empty_roi);
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

INSTANTIATE_TEST_CASE_P(OneVPL_Source_PreprocEngine, OneVPL_PreproEngineTest,
                        testing::Values(UT_ACCEL_TYPE));

static void decode_function(cv::gapi::wip::onevpl::VPLLegacyDecodeEngine &decode_engine,
                            cv::gapi::wip::onevpl::ProcessingEngineBase::session_ptr sess_ptr,
                            SafeQueue &queue, int &decoded_number) {
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
}

static void preproc_function(cv::gapi::wip::IPreprocEngine &preproc_engine, SafeQueue&queue,
                             int &preproc_number, const out_frame_info_t &required_frame_param,
                             const cv::util::optional<cv::Rect> &roi_rect = {}) {
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;
    // create preproc session based on frame description & network info
    cv::MediaFrame first_decoded_frame = queue.pop();
    cv::util::optional<pp_params> first_pp_params = preproc_engine.is_applicable(first_decoded_frame);
    ASSERT_TRUE(first_pp_params.has_value());
    pp_session first_pp_sess =
                    preproc_engine.initialize_preproc(first_pp_params.value(),
                                                      required_frame_param);

    // make preproc using incoming decoded frame & preproc session
    cv::MediaFrame first_pp_frame = preproc_engine.run_sync(first_pp_sess,
                                                            first_decoded_frame,
                                                            roi_rect);
    cv::GFrameDesc first_outcome_pp_desc = first_pp_frame.desc();

    // do not hold media frames because they share limited DX11 surface pool resources
    first_decoded_frame = cv::MediaFrame();
    first_pp_frame = cv::MediaFrame();

    // launch pipeline
    bool in_progress = false;
    // let's allow counting of preprocessed frames to check this value later:
    // Currently, it looks redundant to implement any kind of graceful shutdown logic
    // in this test - so let's apply agreement that media source is processed
    // successfully when preproc_number != 1 in result.
    // Specific validation logic which adhere to explicit counter value may be implemented
    // in particular test scope
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
            const auto &vpp_params = params.value().get<vpp_pp_params>();
            const auto &first_vpp_params = first_pp_params.value().get<vpp_pp_params>();
            ASSERT_EQ(vpp_params.handle, first_vpp_params.handle);
            ASSERT_TRUE(0 == memcmp(&vpp_params.info, &first_vpp_params.info, sizeof(mfxFrameInfo)));

            pp_session pp_sess = preproc_engine.initialize_preproc(params.value(),
                                                                   required_frame_param);
            ASSERT_EQ(pp_sess.get<vpp_pp_session>().handle.get(),
                      first_pp_sess.get<vpp_pp_session>().handle.get());

            cv::MediaFrame pp_frame = preproc_engine.run_sync(pp_sess, decoded_frame, empty_roi);
            cv::GFrameDesc pp_desc = pp_frame.desc();
            ASSERT_TRUE(pp_desc == first_outcome_pp_desc);
            in_progress = false;
            preproc_number++;
        }
    } catch (...) {}

    // test if interruption has happened
    ASSERT_FALSE(in_progress);
    ASSERT_NE(preproc_number, 1);
}

#ifdef __WIN32__
static void multi_source_preproc_function(size_t source_num,
                                          cv::gapi::wip::IPreprocEngine &preproc_engine, SafeQueue&queue,
                                          int &preproc_number, const out_frame_info_t &required_frame_param,
                                          const cv::util::optional<cv::Rect> &roi_rect = {}) {
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;
    // create preproc session based on frame description & network info
    cv::MediaFrame first_decoded_frame = queue.pop();
    cv::util::optional<pp_params> first_pp_params = preproc_engine.is_applicable(first_decoded_frame);
    ASSERT_TRUE(first_pp_params.has_value());
    pp_session first_pp_sess =
                    preproc_engine.initialize_preproc(first_pp_params.value(),
                                                      required_frame_param);

    // make preproc using incoming decoded frame & preproc session
    cv::MediaFrame first_pp_frame = preproc_engine.run_sync(first_pp_sess,
                                                            first_decoded_frame,
                                                            roi_rect);
    cv::GFrameDesc first_outcome_pp_desc = first_pp_frame.desc();

    // do not hold media frames because they share limited DX11 surface pool resources
    first_decoded_frame = cv::MediaFrame();
    first_pp_frame = cv::MediaFrame();

    // launch pipeline
    bool in_progress = false;
    preproc_number = 1;
    size_t received_stop_count = 0;
    try {
        while(received_stop_count != source_num) {
            cv::MediaFrame decoded_frame = queue.pop();
            if (SafeQueue::is_stop(decoded_frame)) {
                ++received_stop_count;
                continue;
            }
            in_progress = true;

            cv::util::optional<pp_params> params = preproc_engine.is_applicable(decoded_frame);
            ASSERT_TRUE(params.has_value());

            pp_session pp_sess = preproc_engine.initialize_preproc(params.value(),
                                                                   required_frame_param);
            cv::MediaFrame pp_frame = preproc_engine.run_sync(pp_sess, decoded_frame, empty_roi);
            cv::GFrameDesc pp_desc = pp_frame.desc();
            ASSERT_TRUE(pp_desc == first_outcome_pp_desc);
            in_progress = false;
            decoded_frame = cv::MediaFrame();
            preproc_number++;
        }
    } catch (const std::exception& ex) {
        GAPI_LOG_WARNING(nullptr, "Caught exception in preproc worker: " << ex.what());
    }

    // test if interruption has happened
    if (in_progress) {
        while (true) {
            cv::MediaFrame decoded_frame = queue.pop();
            if (SafeQueue::is_stop(decoded_frame)) {
                break;
            }
        }
    }
    ASSERT_FALSE(in_progress);
    ASSERT_NE(preproc_number, 1);
}
#endif // __WIN32__

using roi_t = cv::util::optional<cv::Rect>;
using preproc_roi_args_t = decltype(std::tuple_cat(std::declval<preproc_args_t>(),
                                                   std::declval<std::tuple<roi_t>>()));
class VPPPreprocROIParams : public ::testing::TestWithParam<preproc_roi_args_t> {};
TEST_P(VPPPreprocROIParams, functional_roi_different_threads)
{
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;
    source_t file_path;
    decoder_t decoder_id;
    acceleration_t accel;
    out_frame_info_t required_frame_param;
    roi_t opt_roi;
    std::tie(file_path, decoder_id, accel, required_frame_param, opt_roi) = GetParam();

    file_path = findDataFile(file_path);

    std::vector<CfgParam> cfg_params_w_accel;
    std::unique_ptr<VPLAccelerationPolicy> decode_accel_policy = create_accel_policy_from_int(accel, cfg_params_w_accel);

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
                                                     cfg_params_w_accel,
                                                     data_provider);

    // create VPP preproc engine
    VPPPreprocEngine preproc_engine(create_accel_policy_from_int(accel, device_selector));

    // launch threads
    SafeQueue queue;
    int decoded_number = 1;
    int preproc_number = 0;

    std::thread decode_thread(decode_function, std::ref(decode_engine), sess_ptr,
                              std::ref(queue), std::ref(decoded_number));
    std::thread preproc_thread(preproc_function, std::ref(preproc_engine),
                               std::ref(queue), std::ref(preproc_number),
                               std::cref(required_frame_param),
                               std::cref(opt_roi));

    decode_thread.join();
    preproc_thread.join();
    ASSERT_EQ(preproc_number, decoded_number);
}

preproc_roi_args_t files_w_roi[] = {
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h264",
                    MFX_CODEC_AVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1080}}},
                    roi_t{cv::Rect{0,0,50,50}}},
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h264",
                    MFX_CODEC_AVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1080}}},
                    roi_t{}},
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h264",
                    MFX_CODEC_AVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1080}}},
                    roi_t{cv::Rect{0,0,100,100}}},
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h264",
                    MFX_CODEC_AVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1080}}},
                    roi_t{cv::Rect{100,100,200,200}}},
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h265",
                    MFX_CODEC_HEVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1280}}},
                    roi_t{cv::Rect{0,0,100,100}}},
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h265",
                    MFX_CODEC_HEVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1280}}},
                    roi_t{}},
    preproc_roi_args_t {"highgui/video/big_buck_bunny.h265",
                    MFX_CODEC_HEVC,     UT_ACCEL_TYPE,
                    out_frame_info_t{cv::GFrameDesc {cv::MediaFormat::NV12, {1920, 1280}}},
                    roi_t{cv::Rect{100,100,200,200}}}
};

INSTANTIATE_TEST_CASE_P(OneVPL_Source_PreprocEngineROI, VPPPreprocROIParams,
                        testing::ValuesIn(files_w_roi));


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

    std::vector<CfgParam> cfg_params_w_accel_vpp;

    // create accel policy
    std::unique_ptr<VPLAccelerationPolicy> accel_policy = create_accel_policy_from_int(accel, cfg_params_w_accel_vpp);

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
    cfg_params_w_accel_vpp.push_back(CfgParam::create_vpp_out_width(
                                        static_cast<uint16_t>(required_frame_param.size.width)));
    cfg_params_w_accel_vpp.push_back(CfgParam::create_vpp_out_height(
                                        static_cast<uint16_t>(required_frame_param.size.height)));

    // create transcode engine
    auto device_selector = accel_policy->get_device_selector();
    VPLLegacyTranscodeEngine engine(std::move(accel_policy));
    auto sess_ptr = engine.initialize_session(mfx_decode_session,
                                              cfg_params_w_accel_vpp,
                                              data_provider);
   // make test in loop
    bool in_progress = false;
    int frames_processed_count = 1;
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

// enable only for WIN32 because there are not CPU processing on Linux by default
#ifdef __WIN32__
class VPPPreprocDispatcherROIParams : public ::testing::TestWithParam<preproc_roi_args_t> {};
TEST_P(VPPPreprocDispatcherROIParams, functional_roi_different_threads)
{
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;
    source_t file_path;
    decoder_t decoder_id;
    acceleration_t accel = 0;
    out_frame_info_t required_frame_param;
    roi_t opt_roi;
    std::tie(file_path, decoder_id, accel, required_frame_param, opt_roi) = GetParam();

    file_path = findDataFile(file_path);

    std::vector<CfgParam> cfg_params_w_accel;
    std::unique_ptr<VPLAccelerationPolicy> decode_accel_policy = create_accel_policy_from_int(accel, cfg_params_w_accel);

    // create file data provider
    std::shared_ptr<IDataProvider> data_provider(new FileDataProvider(file_path,
                                                 {CfgParam::create_decoder_id(decoder_id)}));
    std::shared_ptr<IDataProvider> cpu_data_provider(new FileDataProvider(file_path,
                                                     {CfgParam::create_decoder_id(decoder_id)}));

    mfxLoader mfx{};
    mfxConfig mfx_cfg{};
    std::tie(mfx, mfx_cfg) = prepare_mfx(decoder_id, accel);

    // create decode session
    mfxSession mfx_decode_session{};
    mfxStatus sts = MFXCreateSession(mfx, 0, &mfx_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    mfxSession mfx_cpu_decode_session{};
    sts = MFXCreateSession(mfx, 0, &mfx_cpu_decode_session);
    EXPECT_EQ(MFX_ERR_NONE, sts);

    // create decode engines
    auto device_selector = decode_accel_policy->get_device_selector();
    VPLLegacyDecodeEngine decode_engine(std::move(decode_accel_policy));
    auto sess_ptr = decode_engine.initialize_session(mfx_decode_session,
                                                     cfg_params_w_accel,
                                                     data_provider);
    std::vector<CfgParam> cfg_params_cpu;
    auto cpu_device_selector = std::make_shared<CfgParamDeviceSelector>(cfg_params_cpu);
    VPLLegacyDecodeEngine cpu_decode_engine(std::unique_ptr<VPLAccelerationPolicy>{
                                                            new VPLCPUAccelerationPolicy(cpu_device_selector)});
    auto cpu_sess_ptr = cpu_decode_engine.initialize_session(mfx_cpu_decode_session,
                                                             cfg_params_cpu,
                                                             cpu_data_provider);

    // create VPP preproc engines
    VPPPreprocDispatcher preproc_dispatcher;
    preproc_dispatcher.insert_worker<VPPPreprocEngine>(create_accel_policy_from_int(accel, device_selector));
    preproc_dispatcher.insert_worker<VPPPreprocEngine>(std::unique_ptr<VPLAccelerationPolicy>{
                                                            new VPLCPUAccelerationPolicy(cpu_device_selector)});

    // launch threads
    SafeQueue queue;
    int decoded_number = 1;
    int cpu_decoded_number = 1;
    int preproc_number = 0;

    std::thread decode_thread(decode_function, std::ref(decode_engine), sess_ptr,
                              std::ref(queue), std::ref(decoded_number));
    std::thread cpu_decode_thread(decode_function, std::ref(cpu_decode_engine), cpu_sess_ptr,
                                  std::ref(queue), std::ref(cpu_decoded_number));
    std::thread preproc_thread(multi_source_preproc_function,
                               preproc_dispatcher.size(),
                               std::ref(preproc_dispatcher),
                               std::ref(queue), std::ref(preproc_number),
                               std::cref(required_frame_param),
                               std::cref(opt_roi));

    decode_thread.join();
    cpu_decode_thread.join();
    preproc_thread.join();
    ASSERT_EQ(preproc_number, decoded_number + cpu_decoded_number);
}

INSTANTIATE_TEST_CASE_P(OneVPL_Source_PreprocDispatcherROI, VPPPreprocDispatcherROIParams,
                        testing::ValuesIn(files_w_roi));

#endif // __WIN32__
} // namespace opencv_test
#endif // HAVE_ONEVPL
