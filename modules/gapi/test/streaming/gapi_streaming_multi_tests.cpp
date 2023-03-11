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
        cv::GMat outg = tmp + 1.0; // Add C
        cv::GOpaque<int64_t> out_id = cv::gapi::streaming::seq_id(outg);

        // Compile graph for streaming & add multiple sources
        ccomp = cv::GComputation(cv::GIn(in), cv::GOut(outg, out_id))
            .compileStreaming();
    }

    cv::GStreamingCompiled ccomp;


    struct Output {
        cv::gapi::streaming::tag tag{};
        cv::Mat mat;
        int64_t seq_id = 0;

        std::unordered_map<int, int> frames;
        std::unordered_set<int> eos;
    } out;

    template<typename F>
    void run_with_check(F f) {
        ccomp.start();
        // Process and count frames we use
        while (ccomp.pull(out.tag, cv::gout(out.mat, out.seq_id))) {
            if (out.tag.eos) {
                out.eos.insert(out.tag.id); // end-of-stream received
            } else {
                out.frames[out.tag.id]++;   // data received
                f();                        // run user-defined check
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
    EXPECT_EQ(num_streams, static_cast<int>(out.eos.size()));

    // Every stream should complete the same number of frames
    EXPECT_LT(0, out.frames[stream_ids[0]]);
    for (int i = 1; i < num_streams; i++) {
        EXPECT_EQ(out.frames.at(stream_ids[0]), out.frames.at(stream_ids[i]));
    }
}

INSTANTIATE_TEST_CASE_P(TestEOS, GAPI_Streaming_Multi_Completion,
                        Values(1, 2, 3));

struct GAPI_Streaming_Multi: public GAPI_Streaming_Multi_Base,
                             public ::testing::Test {};

TEST_F(GAPI_Streaming_Multi, TestDifferentCompletionTime_Frames)
{
    // Add streams with a different completion time (based on num frames)
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    int stream_ids[2] = {
        ccomp.addSource<MockSource>(proto, 30, 50), // 30 fps, 50 frames
        ccomp.addSource<MockSource>(proto, 30, 75), // 30 fps, 75 frames
    };

    run_with_check([this](){
        // As one stream finishes earlier than other, check
        // if finished stream's messages don't pop up here
        if (out.eos.size() > 0u) {
            EXPECT_EQ(0u, out.eos.count(out.tag.id));
        }
    });

    // Check completions
    EXPECT_EQ(50, out.frames.at(stream_ids[0]));
    EXPECT_EQ(75, out.frames.at(stream_ids[1]));
}

TEST_F(GAPI_Streaming_Multi, TestDifferentCompletionTime_Rate)
{
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
        if (out.eos.size() > 0u) {
            EXPECT_EQ(0u, out.eos.count(out.tag.id));
        }
    });

    // Check completions
    EXPECT_EQ(50, out.frames.at(stream_ids[0]));
    EXPECT_EQ(50, out.frames.at(stream_ids[1]));
}

TEST_F(GAPI_Streaming_Multi, TestAddStreamDuringExecution)
{
    // The idea of this test is to add more streams to the running
    // G-API pipeline like this:
    //
    //  ...................................(time)
    // | stream 0  |
    //       | stream 1  |
    //              | stream 2 |
    //
    // Assuming a stream is running at 30 fps for 90 frames, we'll add
    // stream 1 at stream 0's 30th frame, and stream 2 at stream 0's
    // 60th frame.

    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);

    int stream_idx = 1; // next id to use
    int stream_ids[3] = {
        ccomp.addSource<MockSource>(proto, 30, 90), // 30 fps 90 frames
        -1,
        -1
    };

    run_with_check([&](){
        // Start new streams on 30's and 60's frame of the first stream
        if (out.tag.id == stream_ids[0]
            && (out.seq_id == 0 || out.seq_id == 59)) {
            stream_ids[stream_idx++] = ccomp.addSource<MockSource>(proto, 30, 90);
        }
    });

    // Test that all streams have been completed
    EXPECT_EQ(90, out.frames.at(stream_ids[0]));
    EXPECT_EQ(90, out.frames.at(stream_ids[1]));
    EXPECT_EQ(90, out.frames.at(stream_ids[2]));
}

TEST_F(GAPI_Streaming_Multi, TestStopAll)
{
    // Run two streams in parallel, then stop the 2nd one.
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    const int stream_ids[2] = {
        // 30 fps, 50 frames
        ccomp.addSource<MockSource>(proto, 30, 50),
        ccomp.addSource<MockSource>(proto, 30, 50),
    };

    bool stop_called = false;
    run_with_check([&]() {
        ASSERT_FALSE(stop_called); // Should never enter this place after stop
        if (out.frames.at(stream_ids[0]) >= 10
            && out.frames.at(stream_ids[1]) >= 10) {
            ccomp.stop();
            stop_called = true;
        }
    });

    // Completions for both streams has been registered
    EXPECT_EQ(1u, out.eos.count(stream_ids[0]));
    EXPECT_EQ(1u, out.eos.count(stream_ids[1]));

    // NB: this check may be scheduler-dependant (as well as the
    // condition under run_with_check()
    // In theory with multi-stream G-API could complete the first stream
    // first, and then run the second one. All contracts still satisfy,
    // but in this case completed frames for #0 will be 50, and
    // for #1 it should be >= 10 to trigger stop.
    EXPECT_GT(50, out.frames.at(stream_ids[0]));
    EXPECT_GT(50, out.frames.at(stream_ids[1]));
}

TEST_F(GAPI_Streaming_Multi, TestStopOne)
{
    // Run two streams in parallel, then stop the 2nd one.
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    const int stream_ids[2] = {
        // 30 fps, 50 frames
        ccomp.addSource<MockSource>(proto, 30, 50),
        ccomp.addSource<MockSource>(proto, 30, 50),
    };

    bool called_stop = false;
    run_with_check([&](){
        if (out.tag.id == stream_ids[1] && out.seq_id == 24) {
            ccomp.stop(stream_ids[1]);
            called_stop = true;
        }
        if (called_stop) {
            // Note: this code repeats for all streams' received
            // frames.  As it is legit to continue calling pull(),
            // pull() will receive stop signal for S[1] (checked at
            // the end).
            EXPECT_NE(stream_ids[1], out.tag.id);
        }
    });
    EXPECT_EQ(50, out.frames.at(stream_ids[0]));
    EXPECT_EQ(25, out.frames.at(stream_ids[1]));
    EXPECT_EQ(1u, out.eos.count(stream_ids[0]));
    EXPECT_EQ(1u, out.eos.count(stream_ids[1])); // Even for stopped one!
}

TEST_F(GAPI_Streaming_Multi, TwoConsecutiveRuns)
{
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    int stream_id[2] = {-1,-1};

    stream_id[0] = ccomp.addSource<MockSource>(proto, 30, 50);
    run();
    EXPECT_EQ(50, out.frames.at(stream_id[0]));

    // Reset counters after run
    out = {};

    // Note: since the stream #0 has completed, a new stream
    // added after that _can_ get the same ID.
    // But so far it is not specified so relying on this is UB.
    stream_id[1] = ccomp.addSource<MockSource>(proto, 60, 25);
    run();
    EXPECT_EQ(25, out.frames.at(stream_id[1]));
}

TEST_F(GAPI_Streaming_Multi, TwoConsecutiveRunsNoAddSource)
{
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    const int stream_id = ccomp.addSource<MockSource>(proto, 30, 50);
    run();
    EXPECT_EQ(50, out.frames.at(stream_id));

    // Just calling start should throw
    EXPECT_ANY_THROW(ccomp.start());
}

TEST_F(GAPI_Streaming_Multi, TestRunAfterStop)
{
    // Like in the single-stream case, a new addSource<> needs to be called
    // if the pipeline was stopped using stop().
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    int stream_id[] = {-1, -1};

    stream_id[0] = ccomp.addSource<MockSource>(proto, 30, 50);
    run_with_check([&](){
        if (out.seq_id > 10) {
            ccomp.stop();
        }
    });
    EXPECT_GT(50, out.frames.at(stream_id[0]));

    // Reset counters after fist run & run again
    out = {};
    stream_id[1] = ccomp.addSource<MockSource>(proto, 30, 50);
    run();
    EXPECT_EQ(50, out.frames.at(stream_id[1]));
}

TEST_F(GAPI_Streaming_Multi, TestRunAfterStopNoAddSource)
{
    cv::Mat proto = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    const int stream_id = ccomp.addSource<MockSource>(proto, 30, 50);
    run_with_check([&](){
        if (out.seq_id > 10) {
            ccomp.stop();
        }
    });
    EXPECT_GT(50, out.frames.at(stream_id));

    // Reset counters after fist run & run again
    out = {};
    EXPECT_ANY_THROW(ccomp.start()); // no addSource<> - throw
}

TEST_F(GAPI_Streaming_Multi, TestDifferentMeta) {
    // Sources must produce identical image formats, otherwise
    // this will throw
    cv::Mat p1 = cv::Mat::eye(cv::Size(640, 480), CV_8UC1);
    ccomp.addSource<MockSource>(p1, 30, 50);

    cv::Mat p2 = cv::Mat::eye(cv::Size(320, 240), CV_8UC3);
    EXPECT_ANY_THROW(ccomp.addSource<MockSource>(p2, 30, 50));
}

} // namespace
} // namespace opencv_test
