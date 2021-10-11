// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test_precomp.hpp"

#ifdef WITH_OAK_BACKEND

#include <opencv2/gapi/oak/oak.hpp>
#include <opencv2/gapi/oak/oak_media_adapter.hpp>

namespace opencv_test
{

TEST(OAK, SimpleCamera)
{
    cv::GFrame in, h265;
    h265 = cv::gapi::oak::encode(in, {});

    auto args = cv::compile_args(cv::gapi::oak::ColorCameraParams{}, cv::gapi::oak::kernels());

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(h265)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    cv::MediaFrame out_frame;
    std::ofstream out_h265_file;

    // Open H265 file for writing
    out_h265_file.open("output.h265", std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    // Pull 300 frames from the camera
    uint32_t frames = 300;
    uint32_t pulled = 0;

    while (pulled++ < frames &&
           pipeline.pull(cv::gout(out_frame))) {
        cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);
        auto adapter = out_frame.get<cv::gapi::oak::OAKMediaAdapter>();
        out_h265_file.write(reinterpret_cast<const char*>(view.ptr[0]), adapter->getDataSize());
    }
}
} // opencv_test

#endif
