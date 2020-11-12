// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include <tuple>

#include "test_precomp.hpp"
#include "opencv2/gapi/streaming/meta.hpp"
#include "opencv2/gapi/streaming/cap.hpp"

namespace opencv_test {

namespace {
void initTestDataPath() {
#ifndef WINRT
    static bool initialized = false;
    if (!initialized)
    {
        // Since G-API has no own test data (yet), it is taken from the common space
        const char* testDataPath = getenv("OPENCV_TEST_DATA_PATH");
        if (testDataPath != nullptr) {
            cvtest::addDataSearchPath(testDataPath);
            initialized = true;
        }
    }
#endif // WINRT
}
} // anonymous namespace

TEST(GraphMeta, Trad_AccessInput) {
    cv::GMat in;
    cv::GMat out1 = cv::gapi::blur(in, cv::Size(3,3));
    cv::GOpaque<int> out2 = cv::gapi::streaming::meta<int>(in, "foo");
    cv::GComputation graph(cv::GIn(in), cv::GOut(out1, out2));

    cv::Mat in_mat = cv::Mat::eye(cv::Size(64, 64), CV_8UC1);
    cv::Mat out_mat;
    int out_meta = 0;

    // manually set metadata in the input fields
    auto inputs = cv::gin(in_mat);
    inputs[0].meta["foo"] = 42;

    graph.apply(std::move(inputs), cv::gout(out_mat, out_meta));
    EXPECT_EQ(42, out_meta);
}

TEST(GraphMeta, Trad_AccessTmp) {
    cv::GMat in;
    cv::GMat tmp = cv::gapi::blur(in, cv::Size(3,3));
    cv::GMat out1 = tmp+1;
    cv::GOpaque<float> out2 = cv::gapi::streaming::meta<float>(tmp, "bar");
    cv::GComputation graph(cv::GIn(in), cv::GOut(out1, out2));

    cv::Mat in_mat = cv::Mat::eye(cv::Size(64, 64), CV_8UC1);
    cv::Mat out_mat;
    float out_meta = 0.f;

    // manually set metadata in the input fields
    auto inputs = cv::gin(in_mat);
    inputs[0].meta["bar"] = 1.f;

    graph.apply(std::move(inputs), cv::gout(out_mat, out_meta));
    EXPECT_EQ(1.f, out_meta);
}

TEST(GraphMeta, Trad_AccessOutput) {
    cv::GMat in;
    cv::GMat out1 = cv::gapi::blur(in, cv::Size(3,3));
    cv::GOpaque<std::string> out2 = cv::gapi::streaming::meta<std::string>(out1, "baz");
    cv::GComputation graph(cv::GIn(in), cv::GOut(out1, out2));

    cv::Mat in_mat = cv::Mat::eye(cv::Size(64, 64), CV_8UC1);
    cv::Mat out_mat;
    std::string out_meta;

    // manually set metadata in the input fields
    auto inputs = cv::gin(in_mat);

    // NOTE: Assigning explicitly an std::string is important,
    // otherwise a "const char*" will be stored and won't be
    // translated properly by util::any since std::string is
    // used within the graph.
    inputs[0].meta["baz"] = std::string("opencv");

    graph.apply(std::move(inputs), cv::gout(out_mat, out_meta));
    EXPECT_EQ("opencv", out_meta);
}

TEST(GraphMeta, Streaming_AccessInput)
{
    initTestDataPath();

    cv::GMat in;
    cv::GMat out1 = cv::gapi::blur(in, cv::Size(3,3));
    cv::GOpaque<int64_t> out2 = cv::gapi::streaming::seq_id(in);
    cv::GComputation graph(cv::GIn(in), cv::GOut(out1, out2));

    auto ccomp = graph.compileStreaming();
    ccomp.setSource<cv::gapi::wip::GCaptureSource>(findDataFile("cv/video/768x576.avi"));
    ccomp.start();

    cv::Mat out_mat;
    int64_t out_meta = 0;
    int64_t expected_counter = 0;

    while (ccomp.pull(cv::gout(out_mat, out_meta))) {
        EXPECT_EQ(expected_counter, out_meta);
        ++expected_counter;
    }
}


} // namespace opencv_test
