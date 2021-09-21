// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test_precomp.hpp"

#ifdef WITH_OAK_BACKEND

#include "depthai/depthai.hpp"

namespace opencv_test
{

TEST(OAK, SimpleCamera)
{
    cv::GFrame in;
    cv::GArray<uint8_t> h264;
    h264 = cv::gapi::streaming::encH264(bgr);

    auto args = cv::compile_args(cv::gapi::oak::kernels());

    auto pipeline = cv::GComputation(cv::GIn(in), cv::GOut(h264))
        .compileStreaming(args);

    // Graph execution /////////////////////////////////////////////////////////
    pipeline.setSource(cv::gapi::wip::make_src<cv::gapi::oak::ColorCamera>());
    pipeline.start();

    std::vector<uint8_t> out_h264;

    std::ofstream out_h264_file;

    cv::GOptRunArgsP pipeline_outs = cv::gout(out_h264);

    // Open H264 file for writing
    out_h264_file.open(opt_h264_out, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);

    while (pipeline.pull(std::move(pipeline_outs))) {
        const std::vector<uint8_t> &packet = *out_h264;
        CV_Assert(!packet.empty());
        std::cout << "h264: " << packet.size() << " bytes" << std::endl;

        if (out_h264_file.is_open()) {
            out_h264_file.write(reinterpret_cast<const char*>(packet.data()), packet.size());
        }
    }
}

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
    cv::MediaFrame in_frame = cv::MediaFrame::Create<TestMediaBGR>(bgr);
    cv::MediaFrame out_frame = cv::MediaFrame::Create<TestMediaBGR>(bgr);

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
    cv::MediaFrame in_frame = cv::MediaFrame::Create<TestMediaBGR>(bgr);
    cv::MediaFrame out_frame = cv::MediaFrame::Create<TestMediaBGR>(bgr);

    c.apply(cv::gin(in_frame), cv::gout(out_frame), args);

    cv::MediaFrame::View view = out_frame.access(cv::MediaFrame::Access::R);
    cv::Mat out_mat(out_frame.desc().size, CV_8UC3, view.ptr[0], view.stride[0]);
    // add proper check here?
    cv::imwrite("oak_mat.png", out_mat);
}

} // namespace

#endif