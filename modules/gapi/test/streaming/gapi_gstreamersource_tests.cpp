// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test/common/gapi_tests_common.hpp"

#include <opencv2/gapi/streaming/gstreamer/gstreamerpipeline.hpp>
#include <opencv2/gapi/streaming/gstreamer/gstreamersource.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/streaming/meta.hpp>
#include <opencv2/gapi/streaming/format.hpp>

#include <opencv2/gapi/gkernel.hpp>
#include <opencv2/gapi/cpu/gcpukernel.hpp>
#include <opencv2/gapi/gcomputation.hpp>

#include <opencv2/ts.hpp>

#include <regex>

#ifdef HAVE_GSTREAMER

namespace opencv_test
{

struct GStreamerSourceTest : public TestWithParam<std::tuple<std::string, cv::Size, std::size_t>>
{ };


TEST_P(GStreamerSourceTest, AccuracyTest)
{
    std::string pipeline;
    cv::Size expectedFrameSize;
    std::size_t streamLength { };
    std::tie(pipeline, expectedFrameSize, streamLength) = GetParam();

    // Graph declaration:
    cv::GMat in;
    auto out = cv::gapi::copy(in);
    cv::GComputation c(cv::GIn(in), cv::GOut(out));

    // Graph compilation for streaming mode:
    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // GStreamer streaming source configuration:
    ccomp.setSource<cv::gapi::wip::GStreamerSource>(pipeline);

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    cv::Mat in_mat_gapi;

    EXPECT_TRUE(ccomp.pull(cv::gout(in_mat_gapi)));
    EXPECT_TRUE(!in_mat_gapi.empty());
    EXPECT_EQ(expectedFrameSize, in_mat_gapi.size());
    EXPECT_EQ(CV_8UC3, in_mat_gapi.type());

    std::size_t framesCount = 1UL;
    while (ccomp.pull(cv::gout(in_mat_gapi))) {
        EXPECT_TRUE(!in_mat_gapi.empty());
        EXPECT_EQ(expectedFrameSize, in_mat_gapi.size());
        EXPECT_EQ(CV_8UC3, in_mat_gapi.type());

        framesCount++;
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());

    EXPECT_EQ(streamLength, framesCount);
}

TEST_P(GStreamerSourceTest, TimestampsTest)
{
    std::string pipeline;
    std::size_t streamLength { };
    std::tie(pipeline, std::ignore, streamLength) = GetParam();

    // Graph declaration:
    cv::GMat in;
    cv::GMat copied = cv::gapi::copy(in);
    cv::GOpaque<int64_t> outId = cv::gapi::streaming::seq_id(copied);
    cv::GOpaque<int64_t> outTs = cv::gapi::streaming::timestamp(copied);
    cv::GComputation c(cv::GIn(in), cv::GOut(outId, outTs));

    // Graph compilation for streaming mode:
    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // GStreamer streaming source configuration:
    ccomp.setSource<cv::gapi::wip::GStreamerSource>(pipeline);

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    int64_t seqId;
    int64_t timestamp;

    std::vector<int64_t> allSeqIds;
    std::vector<int64_t> allTimestamps;

    while (ccomp.pull(cv::gout(seqId, timestamp))) {
        allSeqIds.push_back(seqId);
        allTimestamps.push_back(timestamp);
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());

    EXPECT_EQ(0L, allSeqIds.front());
    EXPECT_EQ(int64_t(streamLength) - 1, allSeqIds.back());
    EXPECT_EQ(streamLength, allSeqIds.size());
    EXPECT_TRUE(std::is_sorted(allSeqIds.begin(), allSeqIds.end()));
    EXPECT_EQ(allSeqIds.size(), std::set<int64_t>(allSeqIds.begin(), allSeqIds.end()).size());

    EXPECT_EQ(streamLength, allTimestamps.size());
    EXPECT_TRUE(std::is_sorted(allTimestamps.begin(), allTimestamps.end()));
}

G_TYPED_KERNEL(GGstFrameCopyToNV12, <std::tuple<cv::GMat,cv::GMat>(GFrame)>,
    "org.opencv.test.gstframe_copy_to_nv12")
{
    static std::tuple<GMatDesc, GMatDesc> outMeta(GFrameDesc desc) {
        GMatDesc y  { CV_8U, 1, desc.size, false };
        GMatDesc uv { CV_8U, 2, desc.size / 2, false };

        return std::make_tuple(y, uv);
    }
};

G_TYPED_KERNEL(GGstFrameCopyToGRAY8, <cv::GMat(GFrame)>,
    "org.opencv.test.gstframe_copy_to_gray8")
{
    static GMatDesc outMeta(GFrameDesc desc) {
        GMatDesc y{ CV_8U, 1, desc.size, false };
        return y;
    }
};


GAPI_OCV_KERNEL(GOCVGstFrameCopyToNV12, GGstFrameCopyToNV12)
{
    static void run(const cv::MediaFrame& in, cv::Mat& y, cv::Mat& uv)
    {
        auto view = in.access(cv::MediaFrame::Access::R);
        cv::Mat ly(y.size(), y.type(), view.ptr[0], view.stride[0]);
        cv::Mat luv(uv.size(), uv.type(), view.ptr[1], view.stride[1]);

        ly.copyTo(y);
        luv.copyTo(uv);
    }
};

GAPI_OCV_KERNEL(GOCVGstFrameCopyToGRAY8, GGstFrameCopyToGRAY8)
{
    static void run(const cv::MediaFrame & in, cv::Mat & y)
    {
        auto view = in.access(cv::MediaFrame::Access::R);
        cv::Mat ly(y.size(), y.type(), view.ptr[0], view.stride[0]);
        ly.copyTo(y);
    }
};


TEST_P(GStreamerSourceTest, GFrameTest)
{
    std::string pipeline;
    cv::Size expectedFrameSize;
    std::size_t streamLength { };
    bool isNV12 = false;
    std::tie(pipeline, expectedFrameSize, streamLength) = GetParam();

    //Check if pipline string contains NV12 sub-string
    if (pipeline.find("NV12") != std::string::npos) {
        isNV12 = true;
    }

    // Graph declaration:
    cv::GFrame in;
    cv::GMat copiedY, copiedUV;
    if (isNV12) {
        std::tie(copiedY, copiedUV) = GGstFrameCopyToNV12::on(in);
    }
    else {
        copiedY = GGstFrameCopyToGRAY8::on(in);
    }

    cv::GComputation c(cv::GIn(in), isNV12 ? cv::GOut(copiedY, copiedUV) : cv::GOut(copiedY));

    // Graph compilation for streaming mode:
    cv::GStreamingCompiled ccomp;
    if (isNV12) {
        ccomp = c.compileStreaming(cv::compile_args(cv::gapi::kernels<GOCVGstFrameCopyToNV12>()));
    } else {
        ccomp = c.compileStreaming(cv::compile_args(cv::gapi::kernels<GOCVGstFrameCopyToGRAY8>()));
    }


    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    // GStreamer streaming source configuration:
    ccomp.setSource<cv::gapi::wip::GStreamerSource>
        (pipeline, cv::gapi::wip::GStreamerSource::OutputType::FRAME);

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    cv::Mat y_mat, uv_mat;

    EXPECT_TRUE(isNV12 ? ccomp.pull(cv::gout(y_mat, uv_mat)) : ccomp.pull(cv::gout(y_mat)));
    EXPECT_TRUE(!y_mat.empty());
    if (isNV12) {
        EXPECT_TRUE(!uv_mat.empty());
    }

    cv::Size expectedYSize = expectedFrameSize;
    cv::Size expectedUVSize = expectedFrameSize / 2;

    EXPECT_EQ(expectedYSize, y_mat.size());
    if (isNV12) {
        EXPECT_EQ(expectedUVSize, uv_mat.size());
    }

    EXPECT_EQ(CV_8UC1, y_mat.type());
    if (isNV12) {
        EXPECT_EQ(CV_8UC2, uv_mat.type());
    }

    std::size_t framesCount = 1UL;
    while (isNV12 ? ccomp.pull(cv::gout(y_mat, uv_mat)) : ccomp.pull(cv::gout(y_mat))) {
        EXPECT_TRUE(!y_mat.empty());
        if (isNV12) {
            EXPECT_TRUE(!uv_mat.empty());
        }

        EXPECT_EQ(expectedYSize, y_mat.size());
        if (isNV12) {
            EXPECT_EQ(expectedUVSize, uv_mat.size());
        }

        EXPECT_EQ(CV_8UC1, y_mat.type());
        if (isNV12) {
            EXPECT_EQ(CV_8UC2, uv_mat.type());
        }

        framesCount++;
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());

    EXPECT_EQ(streamLength, framesCount);
}


// FIXME: Need to launch with sudo. May be infrastructure problems.
// TODO: It is needed to add tests for streaming from native KMB camera: kmbcamsrc
//       GStreamer element.
INSTANTIATE_TEST_CASE_P(CameraEmulatingPipeline, GStreamerSourceTest,
                        Combine(Values("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=NV12,width=1920,height=1080,framerate=3/1 ! "
                                       "appsink",
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=GRAY8,width=1920,height=1080,framerate=3/1 ! "
                                       "appsink"),
                                Values(cv::Size(1920, 1080)),
                                Values(10UL)));


INSTANTIATE_TEST_CASE_P(FileEmulatingPipeline, GStreamerSourceTest,
                        Combine(Values("videotestsrc pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=NV12,width=640,height=420,framerate=3/1 ! "
                                       "appsink",
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "videorate ! videoscale ! "
                                       "video/x-raw,format=GRAY8,width=640,height=420,framerate=3/1 ! "
                                       "appsink"),
                                Values(cv::Size(640, 420)),
                                Values(10UL)));


INSTANTIATE_TEST_CASE_P(MultipleLiveSources, GStreamerSourceTest,
                        Combine(Values("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=NV12,width=1280,height=720 ! appsink "
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "fakesink",
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=GRAY8,width=1280,height=720 ! appsink "
                                       "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                                       "fakesink"),
                                Values(cv::Size(1280, 720)),
                                Values(10UL)));


INSTANTIATE_TEST_CASE_P(MultipleNotLiveSources, GStreamerSourceTest,
                        Combine(Values("videotestsrc pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=NV12,width=1280,height=720 ! appsink "
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "fakesink",
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "videoscale ! video/x-raw,format=GRAY8,width=1280,height=720 ! appsink "
                                       "videotestsrc pattern=colors num-buffers=10 ! "
                                       "fakesink"),
                                Values(cv::Size(1280, 720)),
                                Values(10UL)));


TEST(GStreamerMultiSourceSmokeTest, Test)
{
    // Graph declaration:
    cv::GMat in1, in2;
    auto out = cv::gapi::add(in1, in2);
    cv::GComputation c(cv::GIn(in1, in2), cv::GOut(out));

    // Graph compilation for streaming mode:
    auto ccomp = c.compileStreaming();

    EXPECT_TRUE(ccomp);
    EXPECT_FALSE(ccomp.running());

    cv::gapi::wip::GStreamerPipeline
        pipeline("videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                 "videorate ! videoscale ! "
                 "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                 "appsink name=sink1 "
                 "videotestsrc is-live=true pattern=colors num-buffers=10 ! "
                 "videorate ! videoscale ! "
                 "video/x-raw,width=1920,height=1080,framerate=3/1 ! "
                 "appsink name=sink2");

    // GStreamer streaming sources configuration:
    auto src1 = pipeline.getStreamingSource("sink1");
    auto src2 = pipeline.getStreamingSource("sink2");

    ccomp.setSource(cv::gin(src1, src2));

    // Start of streaming:
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    // Streaming - pulling of frames until the end:
    cv::Mat in_mat_gapi;

    EXPECT_TRUE(ccomp.pull(cv::gout(in_mat_gapi)));
    EXPECT_TRUE(!in_mat_gapi.empty());
    EXPECT_EQ(CV_8UC3, in_mat_gapi.type());

    while (ccomp.pull(cv::gout(in_mat_gapi))) {
        EXPECT_TRUE(!in_mat_gapi.empty());
        EXPECT_EQ(CV_8UC3, in_mat_gapi.type());
    }

    EXPECT_FALSE(ccomp.running());
    ccomp.stop();

    EXPECT_FALSE(ccomp.running());
}

struct GStreamerMultiSourceTestNV12 :
    public TestWithParam<std::tuple<cv::GComputation, cv::gapi::wip::GStreamerSource::OutputType>>
{ };

TEST_P(GStreamerMultiSourceTestNV12, ImageDataTest)
{
    std::string pathToLeftIm = findDataFile("cv/stereomatching/datasets/tsukuba/im6.png");
    std::string pathToRightIm = findDataFile("cv/stereomatching/datasets/tsukuba/im2.png");

    std::string pipelineToReadImage("filesrc location=LOC ! pngdec ! videoconvert ! "
        "videoscale ! video/x-raw,format=NV12 ! appsink");

    cv::gapi::wip::GStreamerSource leftImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToLeftIm));
    cv::gapi::wip::GStreamerSource rightImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToRightIm));

    cv::gapi::wip::Data leftImData, rightImData;
    leftImageProvider.pull(leftImData);
    rightImageProvider.pull(rightImData);

    cv::Mat leftRefMat =  cv::util::get<cv::Mat>(leftImData);
    cv::Mat rightRefMat = cv::util::get<cv::Mat>(rightImData);

    // Retrieve test parameters:
    std::tuple<cv::GComputation, cv::gapi::wip::GStreamerSource::OutputType> params = GetParam();
    cv::GComputation extractImage = std::move(std::get<0>(params));
    cv::gapi::wip::GStreamerSource::OutputType outputType = std::get<1>(params);

    // Graph compilation for streaming mode:
    auto compiled =
        extractImage.compileStreaming();

    EXPECT_TRUE(compiled);
    EXPECT_FALSE(compiled.running());

    cv::gapi::wip::GStreamerPipeline
        pipeline(std::string("multifilesrc location=" + pathToLeftIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=NV12 ! "
                 "appsink name=sink1 ") +
                 std::string("multifilesrc location=" + pathToRightIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=NV12 ! "
                 "appsink name=sink2"));

    // GStreamer streaming sources configuration:
    auto src1 = pipeline.getStreamingSource("sink1", outputType);
    auto src2 = pipeline.getStreamingSource("sink2", outputType);

    compiled.setSource(cv::gin(src1, src2));

    // Start of streaming:
    compiled.start();
    EXPECT_TRUE(compiled.running());

    // Streaming - pulling of frames:
    cv::Mat in_mat1, in_mat2;

    std::size_t counter { }, limit { 10 };
    while(compiled.pull(cv::gout(in_mat1, in_mat2)) && (counter < limit)) {
        EXPECT_EQ(0, cv::norm(in_mat1, leftRefMat, cv::NORM_INF));
        EXPECT_EQ(0, cv::norm(in_mat2, rightRefMat, cv::NORM_INF));
        ++counter;
    }

    compiled.stop();

    EXPECT_FALSE(compiled.running());
}

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGMatsTest, GStreamerMultiSourceTestNV12,
                        Combine(Values(cv::GComputation([]()
                                       {
                                           cv::GMat in1, in2;
                                           return cv::GComputation(cv::GIn(in1, in2),
                                                                   cv::GOut(cv::gapi::copy(in1),
                                                                            cv::gapi::copy(in2)));
                                       })),
                               Values(cv::gapi::wip::GStreamerSource::OutputType::MAT)));

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGFramesTest, GStreamerMultiSourceTestNV12,
                        Combine(Values(cv::GComputation([]()
                                       {
                                           cv::GFrame in1, in2;
                                           return cv::GComputation(cv::GIn(in1, in2),
                                                cv::GOut(cv::gapi::streaming::BGR(in1),
                                                         cv::gapi::streaming::BGR(in2)));
                                       })),
                               Values(cv::gapi::wip::GStreamerSource::OutputType::FRAME)));

struct GStreamerMultiSourceTestGRAY8 :
    public TestWithParam<std::tuple<cv::GComputation, cv::gapi::wip::GStreamerSource::OutputType>>
{ };

TEST_P(GStreamerMultiSourceTestGRAY8, ImageDataTest)
{
    std::string pathToLeftIm = findDataFile("cv/stereomatching/datasets/tsukuba/im6.png");
    std::string pathToRightIm = findDataFile("cv/stereomatching/datasets/tsukuba/im2.png");

    std::string pipelineToReadImage("filesrc location=LOC ! pngdec ! videoconvert ! "
        "videoscale ! video/x-raw,format=GRAY8 ! appsink");

    cv::gapi::wip::GStreamerSource leftImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToLeftIm));
    cv::gapi::wip::GStreamerSource rightImageProvider(
        std::regex_replace(pipelineToReadImage, std::regex("LOC"), pathToRightIm));

    cv::gapi::wip::Data leftImData, rightImData;
    leftImageProvider.pull(leftImData);
    rightImageProvider.pull(rightImData);

    cv::Mat leftRefMat =  cv::util::get<cv::Mat>(leftImData);
    cv::Mat rightRefMat = cv::util::get<cv::Mat>(rightImData);

    // Retrieve test parameters:
    std::tuple<cv::GComputation, cv::gapi::wip::GStreamerSource::OutputType> params = GetParam();
    cv::GComputation extractImage = std::move(std::get<0>(params));
    cv::gapi::wip::GStreamerSource::OutputType outputType = std::get<1>(params);

    // Graph compilation for streaming mode:
    auto compiled =
        extractImage.compileStreaming();

    EXPECT_TRUE(compiled);
    EXPECT_FALSE(compiled.running());

    cv::gapi::wip::GStreamerPipeline
        pipeline(std::string("multifilesrc location=" + pathToLeftIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! "
                 "appsink name=sink1 ") +
                 std::string("multifilesrc location=" + pathToRightIm + " index=0 loop=true ! "
                 "pngdec ! videoconvert ! videoscale ! video/x-raw,format=GRAY8 ! "
                 "appsink name=sink2"));

    // GStreamer streaming sources configuration:
    auto src1 = pipeline.getStreamingSource("sink1", outputType);
    auto src2 = pipeline.getStreamingSource("sink2", outputType);

    compiled.setSource(cv::gin(src1, src2));

    // Start of streaming:
    compiled.start();
    EXPECT_TRUE(compiled.running());

    // Streaming - pulling of frames:
    cv::Mat in_mat1, in_mat2;

    std::size_t counter { }, limit { 10 };
    while(compiled.pull(cv::gout(in_mat1, in_mat2)) && (counter < limit)) {
        EXPECT_EQ(0, cv::norm(in_mat1, leftRefMat, cv::NORM_INF));
        EXPECT_EQ(0, cv::norm(in_mat2, rightRefMat, cv::NORM_INF));
        ++counter;
    }

    compiled.stop();

    EXPECT_FALSE(compiled.running());
}

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGMatsTest, GStreamerMultiSourceTestGRAY8,
                        Combine(Values(cv::GComputation([]()
                                       {
                                           cv::GMat in1, in2;
                                           return cv::GComputation(cv::GIn(in1, in2),
                                                                   cv::GOut(cv::gapi::copy(in1),
                                                                            cv::gapi::copy(in2)));
                                       })),
                               Values(cv::gapi::wip::GStreamerSource::OutputType::MAT)));

INSTANTIATE_TEST_CASE_P(GStreamerMultiSourceViaGFramesTest, GStreamerMultiSourceTestGRAY8,
                        Combine(Values(cv::GComputation([]()
                                       {
                                           cv::GFrame in1, in2;
                                           return cv::GComputation(cv::GIn(in1, in2),
                                                cv::GOut(cv::gapi::streaming::BGR(in1),
                                                         cv::gapi::streaming::BGR(in2)));
                                       })),
                               Values(cv::gapi::wip::GStreamerSource::OutputType::FRAME)));

} // namespace opencv_test

#endif // HAVE_GSTREAMER
