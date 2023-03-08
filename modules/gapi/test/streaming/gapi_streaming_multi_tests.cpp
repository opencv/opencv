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

TEST(GAPI_Streaming, Multi_Simple)
{
    // Define a simple image processing graph
    cv::GMat in;
    cv::GMat tmp = cv::gapi::resize(in, cv::Size(320, 240));
    cv::GMat out = tmp + 1.0; // Add C
    cv::GOpaque<int64_t> out_id = cv::gapi::streaming::seq_id(out);

    // Compile graph for streaming & add multiple sources
    auto ccomp = cv::GComputation(cv::GIn(in), cv::GOut(out, out_id))
        .compileStreaming();

    auto path = findDataFile("cv/video/768x576.avi");
    int stream_tag[2] = {0,0};
    try {
        std::cout << path << std::endl;
        stream_tag[0] = ccomp.addSource<cv::gapi::wip::GCaptureSource>(path);
        stream_tag[1] = ccomp.addSource<cv::gapi::wip::GCaptureSource>(path);
    } catch(...) {
        throw SkipTestException("Video file can not be opened");
    }

    // Start the execution & collect the outputs
    ccomp.start();
    EXPECT_TRUE(ccomp.running());

    int out_tag{};
    cv::Mat out_mat;
    int64_t out_seq_id{};

    std::unordered_map<int, std::unordered_set<int64_t> > ids;

    // FIXME: How to check individual stream completion status??

    while (ccomp.pull(out_tag, cv::gout(out_mat, out_seq_id))) {
        // The output must be tagged by either stream
        EXPECT_TRUE(out_tag == stream_tag[0] || out_tag == stream_tag[1]);
        ids[out_tag].insert(out_seq_id);
    }

    // We should get frames for each stream
    EXPECT_GT(ids[stream_tag[0]].size(), 0u);
    EXPECT_GT(ids[stream_tag[1]].size(), 0u);
    EXPECT_EQ(ids[stream_tag[0]], ids[stream_tag[1]]);
    std::cout << ids[stream_tag[0]].size() << std::endl;
}

} // namespace

} // namespace opencv_test
