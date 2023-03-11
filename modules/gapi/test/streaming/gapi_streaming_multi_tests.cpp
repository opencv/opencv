// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_streaming_tests_common.hpp"

#include <chrono>
#include <thread>
#include <unordered_set>
#include <unordered_map>

#include <opencv2/gapi/cpu/core.hpp>
#include <opencv2/gapi/cpu/imgproc.hpp>

#include <opencv2/gapi/fluid/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/fluid/gfluidkernel.hpp>

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/gstreaming.hpp>


namespace opencv_test
{
namespace
{

class MockSource final: public cv::gapi::wip::IStreamSource {
    cv::Mat m_prototype;
    int m_frame_rate;
    int m_num_frames;
    int m_usec_delay;

    std::chrono::time_point<std::chrono::system_clock> m_last_point;
    int64_t m_counter;

public:
    MockSource(const cv::Mat &prototype, int frame_rate, int num_frames)
        : m_prototype(prototype)
        , m_frame_rate(frame_rate)
        , m_num_frames(num_frames) {
        GAPI_Assert(m_frame_rate > 0);
        GAPI_Assert(m_num_frames > 0);
        m_usec_delay = 1000000 / frame_rate;
    }

    cv::GMetaArg descr_of() const override {
        return cv::GMetaArg{cv::descr_of(m_prototype)};
    }

    bool pull(cv::gapi::wip::Data &data) override {
        if (m_counter == m_num_frames) {
            return false;
        }

        // Check how long the source should sleep to maintain the frame rate
        // (not thinking of frame drops here)
        using ms_t = std::chrono::microseconds;
        const auto now = std::chrono::system_clock::now();
        auto dur = std::chrono::duration_cast<ms_t>(now.time_since_epoch());

        const auto time_passed = std::chrono::duration_cast<ms_t>(now - m_last_point);
        if (time_passed.count() < m_usec_delay) {
            std::this_thread::sleep_for(ms_t{m_usec_delay - time_passed.count()});
        }
        data = m_prototype;

        // Tag data with seq_id/ts
        data.meta[cv::gapi::streaming::meta_tag::timestamp] = int64_t{dur.count()};
        data.meta[cv::gapi::streaming::meta_tag::seq_id]    = int64_t{m_counter++};
        m_last_point = now;
        return true;
    }
};

struct GAPI_Streaming_Multi_Base {
    GAPI_Streaming_Multi_Base() {
        // Define a simple image processing graph
        cv::GMat in;
        cv::GMat tmp = cv::gapi::resize(in, cv::Size(320, 240));
        cv::GMat out = tmp + 1.0; // Add C
        cv::GOpaque<int64_t> out_id = cv::gapi::streaming::seq_id(out);

        // Compile graph for streaming & add multiple sources
        ccomp = cv::GComputation(cv::GIn(in), cv::GOut(out, out_id))
            .compileStreaming();
    }

    cv::GStreamingCompiled ccomp;

    cv::gapi::streaming::tag out_tag{};
    cv::Mat out_mat;
    int64_t out_seq_id = 0;

    std::unordered_map<int, int> frames;
    std::unordered_set<int> end_of_streams;

    template<typename F>
    void run_with_check(F f) {
        ccomp.start();
        // Process and count frames we use
        while (ccomp.pull(out_tag, cv::gout(out_mat, out_seq_id))) {
            if (out_tag.eos) {
                end_of_streams.insert(out_tag.id); // end-of-stream received
            } else {
                frames[out_tag.id]++;              // data received
                f();                               // run check
            }
        }
    }

    void run() {
        run_with_check([](){});
    }
};

struct GAPI_Streaming_Multi_Completion:
        public GAPI_Streaming_Multi_Base,
        public TestWithParam<int>
{ };

TEST_P(GAPI_Streaming_Multi_Completion, TestEOS)
{
    const auto num_streams = GetParam();
    const auto path = findDataFile("cv/video/768x576.avi");
    std::vector<int> stream_ids;
    try {
        for (int i = 0; i < num_streams; i++) {
            stream_ids.push_back
                (ccomp.addSource<cv::gapi::wip::GCaptureSource>(path));
        }
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }
    run();

    // Should receive EOS for every stream
    EXPECT_EQ(num_streams, static_cast<int>(end_of_streams.size()));

    // Every stream should complete the same number of frames
    EXPECT_GT(frames[stream_ids[0]], 0);
    for (int i = 1; i < num_streams; i++) {
        EXPECT_EQ(frames.at(stream_ids[0]), frames.at(stream_ids[i]));
    }
}

INSTANTIATE_TEST_CASE_P(TestEOS, GAPI_Streaming_Multi_Completion,
                        Values(1, 2, 3));

struct GAPI_Streaming_Multi: public GAPI_Streaming_Multi_Base,
                             public ::testing::Test {};

TEST_F(GAPI_Streaming_Multi, TestDifferentCompletionTime_Frames) {
    // Add streams with a different completion time (based on num frames)
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    int stream_ids[2] = {
        ccomp.addSource<MockSource>(proto, 30, 50), // 30 fps, 50 frames
        ccomp.addSource<MockSource>(proto, 30, 75), // 30 fps, 75 frames
    };

    run_with_check([this](){
        // As one stream finishes earlier than other, check
        // if finished stream's messages don't pop up here
        if (end_of_streams.size() > 0u) {
            EXPECT_EQ(0u, end_of_streams.count(out_tag.id));
        }
    });

    // Check completions
    EXPECT_EQ(50, frames.at(stream_ids[0]));
    EXPECT_EQ(75, frames.at(stream_ids[1]));
}

TEST_F(GAPI_Streaming_Multi, TestDifferentCompletionTime_Rate) {
    // Same test as before but now use sources with different
    // latency
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    int stream_ids[2] = {
        ccomp.addSource<MockSource>(proto, 30, 50), // 30 fps, 50 frames
        ccomp.addSource<MockSource>(proto, 60, 50), // 50 fps, 50 frames
    };

    run_with_check([this](){
        // As one stream finishes earlier than other, check
        // if finished stream's messages don't pop up here
        if (end_of_streams.size() > 0u) {
            EXPECT_EQ(0u, end_of_streams.count(out_tag.id));
        }
    });

    // Check completions
    EXPECT_EQ(50, frames.at(stream_ids[0]));
    EXPECT_EQ(50, frames.at(stream_ids[1]));
}

} // namespace

} // namespace opencv_test
