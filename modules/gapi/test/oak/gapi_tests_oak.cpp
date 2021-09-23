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

TEST(OAK, SimpleCamera)
{
    cv::GFrame in, h265;
    h265 = cv::gapi::oak::encode(in);

    auto args = cv::compile_args(cv::gapi::oak::ColorCameraParams{}, cv::gapi::oak::kernels());

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(h265)).compileStreaming(std::move(args));

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    cv::MediaFrame out_frame = cv::MediaFrame::Create<OAKMediaBGR>();
    //std::vector<uint8_t> out_h265;

    std::ofstream out_h265_file;

    // Open H265 file for writing
    out_h265_file.open("oak_stream.h265", std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    // Pull 300 frames from the camera
    uint32_t frames = 300;
    uint32_t pulled = 0;

    while (pulled++ < frames &&
           pipeline.pull(cv::gout(out_frame))) {
               cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);

        std::cout.width(6);  std::cout << std::left << "h265: ";
        //std::cout.width(15); std::cout << std::left << out_frame.desc().size +
        //                                               " bytes, ";
        out_h265_file.write(reinterpret_cast<const char*>(view.ptr[0]), out_frame.desc().size.width *
                                                                        out_frame.desc().size.height * 8 *3);
    }
}
/*
namespace {
    G_TYPED_KERNEL(TestKernel, <GFrame(GFrame)>, "test.oak.kernel")
    {
        static GFrameDesc outMeta(const GFrameDesc& desc) {
            return desc;
        }
    };

    GAPI_OCV_KERNEL(OCVTestKernel, TestKernel)
    {
        static void run(const cv::MediaFrame& in, cv::MediaFrame& out)
        {
            out = in;
        }
    };
};

TEST(OAK, SimpleFile)
{
    cv::GFrame in, out;
    out = cv::gapi::oak::edgeDetector(in);

    auto args = cv::compile_args(cv::gapi::oak::kernels());

    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    cv::Mat bgr = cv::Mat(1920, 1080, CV_8UC3);
    cv::randu(bgr, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::MediaFrame in_frame = cv::MediaFrame::Create<OAKMediaBGR>(); // fixme: fill data
    cv::MediaFrame out_frame = cv::MediaFrame::Create<OAKMediaBGR>();

    c.apply(cv::gin(in_frame), cv::gout(out_frame), args);

    cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);
    cv::Mat out_mat(out_frame.desc().size, CV_8UC3, view.ptr[0], view.stride[0]);
    // add proper check here?
    cv::imwrite("oak_mat.png", out_mat);
}

TEST(OAK, SimpleFileHetero)
{
    cv::GFrame in, out;

    auto tmp = TestKernel::on(in);
    out = cv::gapi::oak::edgeDetector(tmp);

    auto pkg = cv::gapi::kernels<OCVTestKernel>;
    auto args = cv::compile_args(cv::gapi::combine(cv::gapi::oak::kernels(), pkg));

    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    cv::Mat bgr = cv::Mat(1920, 1080, CV_8UC3);
    cv::randu(bgr, cv::Scalar::all(0), cv::Scalar::all(255));
    cv::MediaFrame in_frame = cv::MediaFrame::Create<OAKMediaBGR>(); // fixme: fill data
    cv::MediaFrame out_frame = cv::MediaFrame::Create<OAKMediaBGR>();

    c.apply(cv::gin(in_frame), cv::gout(out_frame), args);

    cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);
    cv::Mat out_mat(out_frame.desc().size, CV_8UC3, view.ptr[0], view.stride[0]);
    // add proper check here?
    cv::imwrite("oak_mat.png", out_mat);
}*/

} // namespace

#endif