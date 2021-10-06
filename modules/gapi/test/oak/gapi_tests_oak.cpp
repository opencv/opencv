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
namespace {
class TestMediaBGR final : public cv::MediaFrame::IAdapter {
    cv::Mat m_mat;
public:
    explicit TestMediaBGR(cv::Mat m)
        : m_mat(m) {
    }
    cv::GFrameDesc meta() const override {
        return cv::GFrameDesc{ cv::MediaFormat::BGR, cv::Size(m_mat.cols, m_mat.rows) };
    }
    cv::MediaFrame::View access(cv::MediaFrame::Access) override {
        cv::MediaFrame::View::Ptrs pp = { m_mat.ptr(), nullptr, nullptr, nullptr };
        cv::MediaFrame::View::Strides ss = { m_mat.step, 0u, 0u, 0u };
        return cv::MediaFrame::View(std::move(pp), std::move(ss));
    }
};
} // anonymous namespace

TEST(OAK, SimpleCamera)
{
    cv::GFrame in, h265;
    h265 = cv::gapi::oak::encode(in, {});

    auto args = cv::compile_args(cv::gapi::oak::ColorCameraParams{}, cv::gapi::oak::kernels());

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(h265)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    cv::MediaFrame out_frame;// = cv::MediaFrame::Create<cv::gapi::oak::OAKMediaAdapter>();
    std::ofstream out_h265_file;

    // Open H265 file for writing
    out_h265_file.open("output.h265", std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    // Pull 300 frames from the camera
    uint32_t frames = 30;
    uint32_t pulled = 0;

    while (pulled++ < frames &&
           pipeline.pull(cv::gout(out_frame))) {
        std::cout << "pulled" << std::endl;
        cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);
        if (view.ptr[0] == nullptr) {
            std::cout << "nullptr" << std::endl;
        }
        // FIXME: fix (8 * 3) multiplier
        out_h265_file.write(reinterpret_cast<const char*>(view.ptr[0]), out_frame.desc().size.width *
                                                                        out_frame.desc().size.height * 8 * 3);
    }
}
} // opencv_test

#endif
