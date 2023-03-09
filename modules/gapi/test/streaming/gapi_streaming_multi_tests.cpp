// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019-2021 Intel Corporation


#include "../test_precomp.hpp"

#include "../common/gapi_streaming_tests_common.hpp"

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
            stream_ids.push_back(ccomp.addSource<cv::gapi::wip::GCaptureSource>(path));
        }
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    ccomp.start();

    std::unordered_map<int, int> frames;
    std::unordered_set<int> end_of_streams;

    while (ccomp.pull(out_tag, cv::gout(out_mat, out_seq_id))) {
        if (out_tag.eos) {
            // end-of-stream received
            end_of_streams.insert(out_tag.id);
        } else {
            // data received. just increment frame counter
            frames[out_tag.id]++;
        }
    }

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

} // namespace

} // namespace opencv_test
