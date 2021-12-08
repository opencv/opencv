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

class OneVPLSourcePerfTest : public TestPerfParams<source_description_t> {};
class VideoCapSourcePerfTest : public TestPerfParams<source_t> {};

PERF_TEST_P_(OneVPLSourcePerfTest, TestPerformance)
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

PERF_TEST_P_(VideoCapSourcePerfTest, TestPerformance)
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

INSTANTIATE_TEST_CASE_P(Streaming, OneVPLSourcePerfTest,
                        Values(source_description_t(files[0], codec[0], ""),
                               source_description_t(files[0], codec[0], "MFX_ACCEL_MODE_VIA_D3D11"),
                               source_description_t(files[1], codec[1], ""),
                               source_description_t(files[1], codec[1], "MFX_ACCEL_MODE_VIA_D3D11"),
                               source_description_t(files[2], codec[2], ""),
                               source_description_t(files[2], codec[2], "MFX_ACCEL_MODE_VIA_D3D11")));

INSTANTIATE_TEST_CASE_P(Streaming, VideoCapSourcePerfTest,
                        Values(files[0],
                               files[1],
                               files[2]));
} // namespace opencv_test

#endif // HAVE_ONEVPL
