// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#ifdef HAVE_ONEVPL

#include "../perf_precomp.hpp"
#include "../../test/common/gapi_tests_common.hpp"
#include <opencv2/gapi/streaming/onevpl/source.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

#ifdef HAVE_INF_ENGINE
#include <inference_engine.hpp>
#endif // HAVE_INF_ENGINE

#include "streaming/onevpl/engine/preproc/preproc_engine.hpp"
#include "streaming/onevpl/engine/preproc/preproc_session.hpp"
#include "streaming/onevpl/accelerators/accel_policy_interface.hpp"
#include "streaming/onevpl/cfg_param_device_selector.hpp"
#include "streaming/onevpl/accelerators/accel_policy_dx11.hpp"
#include "streaming/onevpl/accelerators/accel_policy_cpu.hpp"

namespace opencv_test
{
using namespace perf;

const std::string files[] = {
    "highgui/video/big_buck_bunny.h265",
    "highgui/video/big_buck_bunny.h264",
    "highgui/video/sample_322x242_15frames.yuv420p.libx265.mp4",
};

const std::string codec[] = {
    "MFX_CODEC_HEVC",
    "MFX_CODEC_AVC",
    "",
};

using source_t = std::string;
using codec_t = std::string;
using accel_mode_t = std::string;
using source_description_t = std::tuple<source_t, codec_t, accel_mode_t>;

class OneVPLSourcePerf_Test : public TestPerfParams<source_description_t> {};
class VideoCapSourcePerf_Test : public TestPerfParams<source_t> {};

PERF_TEST_P_(OneVPLSourcePerf_Test, TestPerformance)
{
    using namespace cv::gapi::wip::onevpl;

    const auto params = GetParam();
    source_t src = findDataFile(get<0>(params));
    codec_t type = get<1>(params);
    accel_mode_t mode = get<2>(params);

    std::vector<CfgParam> cfg_params {
        CfgParam::create_implementation("MFX_IMPL_TYPE_HARDWARE"),
    };

    if (!type.empty()) {
        cfg_params.push_back(CfgParam::create_decoder_id(type.c_str()));
    }

    if (!mode.empty()) {
        cfg_params.push_back(CfgParam::create_acceleration_mode(mode.c_str()));
    }

    auto source_ptr = cv::gapi::wip::make_onevpl_src(src, cfg_params);

    cv::gapi::wip::Data out;
    TEST_CYCLE()
    {
        source_ptr->pull(out);
    }

    SANITY_CHECK_NOTHING();
}

PERF_TEST_P_(VideoCapSourcePerf_Test, TestPerformance)
{
    using namespace cv::gapi::wip;

    source_t src = findDataFile(GetParam());
    auto source_ptr = make_src<GCaptureSource>(src);
    Data out;
    TEST_CYCLE()
    {
        source_ptr->pull(out);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(Streaming, OneVPLSourcePerf_Test,
                        Values(source_description_t(files[0], codec[0], ""),
                               source_description_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11"),
                               source_description_t(files[1], codec[1], ""),
                               source_description_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11"),
                               source_description_t(files[2], codec[2], ""),
                               source_description_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11")));

INSTANTIATE_TEST_CASE_P(Streaming, VideoCapSourcePerf_Test,
                        Values(files[0],
                               files[1],
                               files[2]));

using resize_t = std::pair<uint16_t, uint16_t>;
using source_description_preproc_t = decltype(std::tuple_cat(std::declval<source_description_t>(),
                                                             std::declval<std::tuple<resize_t>>()));
class OneVPLSourcePerf_PP_Test : public TestPerfParams<source_description_preproc_t> {};

PERF_TEST_P_(OneVPLSourcePerf_PP_Test, TestPerformance)
{
    using namespace cv::gapi::wip::onevpl;

    const auto params = GetParam();
    source_t src = findDataFile(get<0>(params));
    codec_t type = get<1>(params);
    accel_mode_t mode = get<2>(params);
    resize_t res = get<3>(params);

    std::vector<CfgParam> cfg_params {
        CfgParam::create_implementation("MFX_IMPL_TYPE_HARDWARE"),
    };

    if (!type.empty()) {
        cfg_params.push_back(CfgParam::create_decoder_id(type.c_str()));
    }

    if (!mode.empty()) {
        cfg_params.push_back(CfgParam::create_acceleration_mode(mode.c_str()));
    }

    cfg_params.push_back(CfgParam::create_vpp_out_width(res.first));
    cfg_params.push_back(CfgParam::create_vpp_out_height(res.second));
    cfg_params.push_back(CfgParam::create_vpp_out_crop_x(0));
    cfg_params.push_back(CfgParam::create_vpp_out_crop_y(0));
    cfg_params.push_back(CfgParam::create_vpp_out_crop_w(res.first));
    cfg_params.push_back(CfgParam::create_vpp_out_crop_h(res.second));

    auto source_ptr = cv::gapi::wip::make_onevpl_src(src, cfg_params);

    cv::gapi::wip::Data out;
    TEST_CYCLE()
    {
        source_ptr->pull(out);
    }

    SANITY_CHECK_NOTHING();
}
static resize_t full_hd = resize_t{static_cast<uint16_t>(1920),
                                   static_cast<uint16_t>(1080)};

static resize_t cif = resize_t{static_cast<uint16_t>(352),
                               static_cast<uint16_t>(288)};

INSTANTIATE_TEST_CASE_P(Streaming_Source_PP, OneVPLSourcePerf_PP_Test,
                        Values(source_description_preproc_t(files[0], codec[0], "", full_hd),
                               source_description_preproc_t(files[0], codec[0], "", cif),
                               source_description_preproc_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11", full_hd),
                               source_description_preproc_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11", cif),
                               source_description_preproc_t(files[1], codec[1], "", full_hd),
                               source_description_preproc_t(files[1], codec[1], "", cif),
                               source_description_preproc_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11",full_hd),
                               source_description_preproc_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11",cif),
                               source_description_preproc_t(files[2], codec[2], "", full_hd),
                               source_description_preproc_t(files[2], codec[2], "", cif),
                               source_description_preproc_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11", full_hd),
                               source_description_preproc_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11", cif)));

#ifdef HAVE_INF_ENGINE
class OneVPLSourcePerf_PP_Engine_Test : public TestPerfParams<source_description_preproc_t> {};
InferenceEngine::InputInfo::CPtr mock_network_info(size_t width, size_t height) {
    auto net_input = std::make_shared<InferenceEngine::InputInfo>();
    InferenceEngine::SizeVector dims_src = {1         /* batch, N*/,
                                            height,
                                            width,
                                            3 /*Channels,*/,
                                            };
    InferenceEngine::DataPtr dataPtr(
        new InferenceEngine::Data("data", InferenceEngine::TensorDesc(InferenceEngine::Precision::FP32, dims_src, InferenceEngine::NHWC)));
    net_input->setInputData(dataPtr);
    InferenceEngine::InputInfo::CPtr cptr = std::make_shared<InferenceEngine::InputInfo>(*net_input);
    return cptr;
}

PERF_TEST_P_(OneVPLSourcePerf_PP_Engine_Test, TestPerformance)
{
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;

    const auto params = GetParam();
    source_t src = findDataFile(get<0>(params));
    codec_t type = get<1>(params);
    accel_mode_t mode = get<2>(params);
    resize_t res = get<3>(params);

    std::vector<CfgParam> cfg_params {
        CfgParam::create_implementation("MFX_IMPL_TYPE_HARDWARE"),
    };

    if (!type.empty()) {
        cfg_params.push_back(CfgParam::create_decoder_id(type.c_str()));
    }

    if (!mode.empty()) {
        cfg_params.push_back(CfgParam::create_acceleration_mode(mode.c_str()));
    }

    auto device_selector = std::make_shared<CfgParamDeviceSelector>(cfg_params);
    auto source_ptr = cv::gapi::wip::make_onevpl_src(src, cfg_params, device_selector);

    // create VPP preproc engine
    InferenceEngine::InputInfo::CPtr cptr = mock_network_info(res.first, res.second);
    std::unique_ptr<VPLAccelerationPolicy> policy;
    if (mode == "MFX_ACCEL_MODE_VIA_D3D11") {
        policy.reset(new VPLDX11AccelerationPolicy(device_selector));
    } else {
        policy.reset(new VPLCPUAccelerationPolicy(device_selector));
    }
    VPPPreprocEngine preproc_engine(std::move(policy));
    cv::gapi::wip::Data out;
    cv::util::optional<cv::Rect> empty_roi;
    TEST_CYCLE()
    {
        source_ptr->pull(out);
        cv::MediaFrame frame = cv::util::get<cv::MediaFrame>(out);
        cv::util::optional<pp_params> param = preproc_engine.is_applicable(frame);
        pp_session sess = preproc_engine.initialize_preproc(param.value(),
                                                            cptr);
        (void)preproc_engine.run_sync(sess, frame, empty_roi);
    }

    SANITY_CHECK_NOTHING();
}

INSTANTIATE_TEST_CASE_P(Streaming_Engine_PP, OneVPLSourcePerf_PP_Engine_Test,
                        Values(source_description_preproc_t(files[0], codec[0], "", full_hd),
                               source_description_preproc_t(files[0], codec[0], "", cif),
                               source_description_preproc_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11", full_hd),
                               source_description_preproc_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11", cif),
                               source_description_preproc_t(files[1], codec[1], "", full_hd),
                               source_description_preproc_t(files[1], codec[1], "", cif),
                               source_description_preproc_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11",full_hd),
                               source_description_preproc_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11",cif),
                               source_description_preproc_t(files[2], codec[2], "", full_hd),
                               source_description_preproc_t(files[2], codec[2], "", cif),
                               source_description_preproc_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11", full_hd),
                               source_description_preproc_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11", cif)));

class OneVPLSourcePerf_PP_Engine_Bypass_Test : public TestPerfParams<source_description_preproc_t> {};

PERF_TEST_P_(OneVPLSourcePerf_PP_Engine_Bypass_Test, TestPerformance)
{
    using namespace cv::gapi::wip;
    using namespace cv::gapi::wip::onevpl;

    const auto params = GetParam();
    source_t src = findDataFile(get<0>(params));
    codec_t type = get<1>(params);
    accel_mode_t mode = get<2>(params);
    resize_t res = get<3>(params);

    std::vector<CfgParam> cfg_params {
        CfgParam::create_implementation("MFX_IMPL_TYPE_HARDWARE"),
    };

    if (!type.empty()) {
        cfg_params.push_back(CfgParam::create_decoder_id(type.c_str()));
    }

    if (!mode.empty()) {
        cfg_params.push_back(CfgParam::create_acceleration_mode(mode.c_str()));
    }

    auto device_selector = std::make_shared<CfgParamDeviceSelector>(cfg_params);
    auto source_ptr = cv::gapi::wip::make_onevpl_src(src, cfg_params, device_selector);

    // create VPP preproc engine
    InferenceEngine::InputInfo::CPtr cptr = mock_network_info(res.first, res.second);
    std::unique_ptr<VPLAccelerationPolicy> policy;
    if (mode == "MFX_ACCEL_MODE_VIA_D3D11") {
        policy.reset(new VPLDX11AccelerationPolicy(device_selector));
    } else {
        policy.reset(new VPLCPUAccelerationPolicy(device_selector));
    }
    VPPPreprocEngine preproc_engine(std::move(policy));
    cv::gapi::wip::Data out;
    cv::util::optional<cv::Rect> empty_roi;
    TEST_CYCLE()
    {
        source_ptr->pull(out);
        cv::MediaFrame frame = cv::util::get<cv::MediaFrame>(out);
        cv::util::optional<pp_params> param = preproc_engine.is_applicable(frame);
        pp_session sess = preproc_engine.initialize_preproc(param.value(),
                                                            cptr);
        (void)preproc_engine.run_sync(sess, frame, empty_roi);
    }

    SANITY_CHECK_NOTHING();
}

static resize_t res_672x384 = resize_t{static_cast<uint16_t>(672),
                                       static_cast<uint16_t>(384)};
static resize_t res_336x256 = resize_t{static_cast<uint16_t>(336),
                                       static_cast<uint16_t>(256)};

INSTANTIATE_TEST_CASE_P(Streaming_Engine_PP_Bypass, OneVPLSourcePerf_PP_Engine_Bypass_Test,
                        Values(source_description_preproc_t(files[0], codec[0], "", res_672x384),
                               source_description_preproc_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11", res_672x384),
                               source_description_preproc_t(files[1], codec[1], "", res_672x384),
                               source_description_preproc_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11", res_672x384),
                               source_description_preproc_t(files[2], codec[2], "", res_336x256),
                               source_description_preproc_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11", res_336x256)));
#endif // HAVE_INF_ENGINE
} // namespace opencv_test

#endif // HAVE_ONEVPL
