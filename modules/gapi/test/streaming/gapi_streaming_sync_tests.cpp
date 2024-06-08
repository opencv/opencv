// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2021 Intel Corporation

#include "../test_precomp.hpp"

#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/core.hpp>
#include <opencv2/gapi/fluid/imgproc.hpp>
#include <opencv2/gapi/streaming/cap.hpp>
#include <opencv2/gapi/streaming/sync.hpp>

namespace opencv_test {
namespace {

using ts_t = int64_t;
using ts_vec = std::vector<ts_t>;
using cv::gapi::streaming::sync_policy;

ts_t calcLeastCommonMultiple(const ts_vec& values) {
    ts_t res = *std::max_element(values.begin(), values.end());
    auto isDivisor = [&](ts_t v) { return res % v == 0; };
    while(!std::all_of(values.begin(), values.end(), isDivisor)) {
        res++;
    }
    return res;
}

struct TimestampGenerationParams {
    const ts_vec frame_times;
    sync_policy policy;
    ts_t end_time;
    TimestampGenerationParams(const ts_vec& ft, sync_policy sp, ts_t et = 25)
        : frame_times(ft), policy(sp), end_time(et) {
    }
};

class MultiFrameSource {
    class SingleSource : public cv::gapi::wip::IStreamSource {
        MultiFrameSource& m_source;
        std::size_t m_idx;
    public:
        SingleSource(MultiFrameSource& s, std::size_t idx)
            : m_source(s)
            , m_idx(idx)
        {}
        virtual bool pull(cv::gapi::wip::Data& data) {
            return m_source.pull(data, m_idx);
        }
        virtual GMetaArg descr_of() const { return GMetaArg{m_source.desc()}; }
    };

    TimestampGenerationParams p;
    ts_vec m_current_times;
    cv::Mat m_mat;

public:
    MultiFrameSource(const TimestampGenerationParams& params)
        : p(params)
        , m_current_times(p.frame_times.size(), 0u)
        , m_mat(8, 8, CV_8UC1) {
    }

    bool pull(cv::gapi::wip::Data& data, std::size_t idx) {
        cv::randn(m_mat, 127, 32);
        GAPI_Assert(idx < p.frame_times.size());
        m_current_times[idx] += p.frame_times[idx];
        if (m_current_times[idx] >= p.end_time) {
            return false;
        }
        data = m_mat.clone();
        data.meta[cv::gapi::streaming::meta_tag::timestamp] = m_current_times[idx];
        return true;
    }

    cv::gapi::wip::IStreamSource::Ptr getSource(std::size_t idx) {
        return cv::gapi::wip::IStreamSource::Ptr{new SingleSource(*this, idx)};
    }

    GMatDesc desc() const { return cv::descr_of(m_mat); }
};

class TimestampChecker {
    TimestampGenerationParams p;
    ts_t m_synced_time = 0u;
    ts_t m_synced_frame_time = 0u;
public:
    TimestampChecker(const TimestampGenerationParams& params)
        : p(params)
        , m_synced_frame_time(calcLeastCommonMultiple(p.frame_times)) {
    }

    void checkNext(const ts_vec& timestamps) {
        if (p.policy == sync_policy::dont_sync) {
            // don't check timestamps if the policy is dont_sync
            return;
        }
        m_synced_time += m_synced_frame_time;
        for (const auto& ts : timestamps) {
            EXPECT_EQ(m_synced_time, ts);
        }
    }

    std::size_t nFrames() const {
        auto frame_time = p.policy == sync_policy::dont_sync
                          ? *std::max_element(p.frame_times.begin(), p.frame_times.end())
                          : m_synced_frame_time;
        auto n_frames = p.end_time / frame_time;
        GAPI_Assert(n_frames > 0u);
        return (std::size_t)n_frames;
    }
};

struct TimestampSyncTest : public ::testing::TestWithParam<sync_policy> {
    void run(cv::GProtoInputArgs&& ins, cv::GProtoOutputArgs&& outs,
             const ts_vec& frame_times) {
        auto video_in_n = frame_times.size();
        GAPI_Assert(video_in_n <= ins.m_args.size());
        // Assume that all remaining inputs are const
        auto const_in_n = ins.m_args.size() - video_in_n;
        auto out_n = outs.m_args.size();
        auto policy = GetParam();
        TimestampGenerationParams ts_params(frame_times, policy);
        MultiFrameSource source(ts_params);

        GRunArgs gins;
        for (std::size_t i = 0; i < video_in_n; i++) {
            gins += cv::gin(source.getSource(i));
        }
        auto desc = source.desc();
        cv::Mat const_mat = cv::Mat::eye(desc.size.height,
                                         desc.size.width,
                                         CV_MAKE_TYPE(desc.depth, desc.chan));
        for (std::size_t i = 0; i < const_in_n; i++) {
            gins += cv::gin(const_mat);
        }
        ts_vec out_timestamps(out_n);
        cv::GRunArgsP gouts{};
        for (auto& t : out_timestamps) {
            gouts += cv::gout(t);
        }

        auto pipe = cv::GComputation(std::move(ins), std::move(outs))
                    .compileStreaming(cv::compile_args(policy));

        pipe.setSource(std::move(gins));
        pipe.start();

        std::size_t frames = 0u;
        TimestampChecker checker(ts_params);
        while(pipe.pull(std::move(gouts))) {
            checker.checkNext(out_timestamps);
            frames++;
        }

        EXPECT_EQ(checker.nFrames(), frames);
    }
};

} // anonymous namespace

TEST_P(TimestampSyncTest, Basic)
{
    cv::GMat in1, in2;
    auto out = cv::gapi::add(in1, in2);
    auto ts = cv::gapi::streaming::timestamp(out);

    run(cv::GIn(in1, in2), cv::GOut(ts), {1,2});
}

TEST_P(TimestampSyncTest, ThreeInputs)
{
    cv::GMat in1, in2, in3;
    auto tmp = cv::gapi::add(in1, in2);
    auto out = cv::gapi::add(tmp, in3);
    auto ts = cv::gapi::streaming::timestamp(out);

    run(cv::GIn(in1, in2, in3), cv::GOut(ts), {2,4,3});
}

TEST_P(TimestampSyncTest, TwoOutputs)
{
    cv::GMat in1, in2, in3;
    auto out1 = cv::gapi::add(in1, in3);
    auto out2 = cv::gapi::add(in2, in3);
    auto ts1 = cv::gapi::streaming::timestamp(out1);
    auto ts2 = cv::gapi::streaming::timestamp(out2);

    run(cv::GIn(in1, in2, in3), cv::GOut(ts1, ts2), {1,4,2});
}

TEST_P(TimestampSyncTest, ConstInput)
{
    cv::GMat in1, in2, in3;
    auto out1 = cv::gapi::add(in1, in3);
    auto out2 = cv::gapi::add(in2, in3);
    auto ts1 = cv::gapi::streaming::timestamp(out1);
    auto ts2 = cv::gapi::streaming::timestamp(out2);

    run(cv::GIn(in1, in2, in3), cv::GOut(ts1, ts2), {1,2});
}

TEST_P(TimestampSyncTest, ChangeSource)
{
    cv::GMat in1, in2, in3;
    auto out1 = cv::gapi::add(in1, in3);
    auto out2 = cv::gapi::add(in2, in3);
    auto ts1 = cv::gapi::streaming::timestamp(out1);
    auto ts2 = cv::gapi::streaming::timestamp(out2);

    run(cv::GIn(in1, in2, in3), cv::GOut(ts1, ts2), {1,2});
    run(cv::GIn(in1, in2, in3), cv::GOut(ts1, ts2), {1,2});
}

INSTANTIATE_TEST_CASE_P(InputSynchronization, TimestampSyncTest,
                        Values(sync_policy::dont_sync,
                               sync_policy::drop));
} // namespace opencv_test
