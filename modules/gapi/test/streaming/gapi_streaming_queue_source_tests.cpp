// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2023 Intel Corporation


#include "../test_precomp.hpp"

#include <opencv2/gapi/gstreaming.hpp>
#include <opencv2/gapi/streaming/queue_source.hpp>

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

} // namespace opencv_test
