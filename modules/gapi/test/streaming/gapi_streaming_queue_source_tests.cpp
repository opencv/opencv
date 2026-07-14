// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation


#include "../test_precomp.hpp"

#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/streaming/queue_source.hpp>
#include <opencv2/gapi/streaming/cap.hpp>

namespace opencv_test
{

TEST(GAPI_Streaming_Queue_Source, SmokeTest) {
    // This is more like an example on G-API Queue Source

    cv::GMat in;
    cv::GMat out = in + 1;
    cv::GStreamingCompiled comp = cv::GComputation(in, out).compileStreaming();

    // Queue source needs to know format information to maintain contracts
    auto src = std::make_shared<cv::gapi::wip::QueueSource<cv::Mat> >
        (cv::GMatDesc{CV_8U, 1, cv::Size{128, 128}});

    comp.setSource(cv::gin(src->ptr()));
    comp.start();

    // It is perfectly legal to start a pipeline at this point - the source was passed.
    // Now we can push data through the source and get the pipeline results.

    cv::Mat eye = cv::Mat::eye(cv::Size{128, 128}, CV_8UC1);
    src->push(eye);    // Push I (identity matrix)
    src->push(eye*2);  // Push I*2

    // Now its time to pop. The data could be already processed at this point.
    // Note the queue source queues are unbounded to avoid deadlocks

    cv::Mat result;
    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(eye + 1, result, NORM_INF));

    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(eye*2 + 1, result, NORM_INF));
}

TEST(GAPI_Streaming_Queue_Source, Mixed) {
    // Mixing a regular "live" source (which runs on its own) with a
    // manually controlled queue source may make a little sense, but
    // is perfectly legal and possible.

    cv::GMat in1;
    cv::GMat in2;
    cv::GMat out = in2 - in1;
    cv::GStreamingCompiled comp = cv::GComputation(in1, in2, out).compileStreaming();

    // Queue source needs to know format information to maintain contracts
    auto src1 = std::make_shared<cv::gapi::wip::QueueSource<cv::Mat> >
        (cv::GMatDesc{CV_8U, 3, cv::Size{768, 576}});

    std::shared_ptr<cv::gapi::wip::IStreamSource> src2;
    auto path = findDataFile("cv/video/768x576.avi");
    try {
        src2 = cv::gapi::wip::make_src<cv::gapi::wip::GCaptureSource>(path);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    comp.setSource(cv::gin(src1->ptr(), src2)); // FIXME: quite inconsistent
    comp.start();

    cv::Mat eye = cv::Mat::eye(cv::Size{768, 576}, CV_8UC3);
    src1->push(eye);    // Push I (identity matrix)
    src1->push(eye);    // Push I (again)

    cv::Mat ref, result;
    cv::VideoCapture cap(path);

    cap >> ref;
    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(ref - eye, result, NORM_INF));

    cap >> ref;
    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(ref - eye, result, NORM_INF));
}

TEST(GAPI_Streaming_Queue_Input, SmokeTest) {

    // Queue Input: a tiny wrapper atop of multiple queue sources.
    // Allows users to pass all input data at once.

    cv::GMat in1;
    cv::GScalar in2;
    cv::GMat out = in1 + in2;
    cv::GStreamingCompiled comp = cv::GComputation(cv::GIn(in1, in2), cv::GOut(out))
        .compileStreaming();

    // FIXME: This API is too raw
    cv::gapi::wip::QueueInput input({
            cv::GMetaArg{ cv::GMatDesc{CV_8U, 1, cv::Size{64,64} } },
            cv::GMetaArg{ cv::empty_scalar_desc() }
        });
    comp.setSource(input); // Implicit conversion allows it to be passed as-is.
    comp.start();

    // Push data via queue input
    cv::Mat eye = cv::Mat::eye(cv::Size{64, 64}, CV_8UC1);
    input.push(cv::gin(eye, cv::Scalar(1)));
    input.push(cv::gin(eye, cv::Scalar(2)));
    input.push(cv::gin(eye, cv::Scalar(3)));

    // Pop data and validate
    cv::Mat result;
    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(eye+1, result, NORM_INF));

    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(eye+2, result, NORM_INF));

    ASSERT_TRUE(comp.pull(cv::gout(result)));
    EXPECT_EQ(0, cvtest::norm(eye+3, result, NORM_INF));
}

} // namespace opencv_test
