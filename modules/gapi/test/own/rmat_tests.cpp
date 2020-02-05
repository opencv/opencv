// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"
#include <opencv2/gapi/own/rmat.hpp>

namespace opencv_test
{
using GMatDesc = cv::GMatDesc;
using RMat = cv::gapi::own::RMat;

class RMatAdapterRef : public RMat::Adapter
{
    cv::Mat& m_mat;
public:
    RMatAdapterRef(cv::Mat& m) : m_mat(m) {}
    virtual cv::Mat access() const override { return m_mat; }
    virtual GMatDesc desc() const override { return cv::descr_of(m_mat); }
    virtual void flush() const override {}
};

class RMatAdapterCopy : public RMat::Adapter
{
    cv::Mat& m_mat;
    cv::Mat  m_copy;

public:
    RMatAdapterCopy(cv::Mat& m) : m_mat(m), m_copy(m.clone()) {}

    virtual cv::Mat access() const override { return m_copy; }
    virtual GMatDesc desc() const override { return cv::descr_of(m_copy); }
    virtual void flush() const override { m_copy.copyTo(m_mat); }
};

TEST(RMat, SmokeTest)
{
    cv::Size sz{8, 8};
    cv::Mat mat(sz, CV_8UC1);
    cv::randu(mat, cv::Scalar::all(127), cv::Scalar::all(40));

    auto rmat = cv::gapi::own::make_rmat<RMatAdapterRef>(mat);
    auto matFromDevice = rmat.access();
    EXPECT_TRUE(cv::descr_of(mat) == cv::descr_of(matFromDevice));
    EXPECT_EQ(0, cvtest::norm(mat, matFromDevice, NORM_INF));
}

struct RMatTestBase
{
    cv::Mat in_mat;
    cv::Mat out_mat;
    cv::Mat out_mat_ref;
    cv::GComputation comp;

    static constexpr int w = 8;
    static constexpr int h = 8;

    RMatTestBase()
        : in_mat(h, w, CV_8UC1)
        , out_mat(h, w, CV_8UC1)
        , out_mat_ref(h, w, CV_8UC1)
        , comp([](){
              cv::GMat in;
              auto tmp = cv::gapi::blur(in, {3,3});
              auto out = cv::gapi::blur(tmp, {3,3});
              cv::gapi::island("test", cv::GIn(in), cv::GOut(tmp));
              return cv::GComputation(in, out);
          })
    {
        cv::randu(in_mat, cv::Scalar::all(127), cv::Scalar::all(40));
    }

    void check()
    {
        comp.apply(in_mat, out_mat_ref);
        EXPECT_EQ(0, cvtest::norm(out_mat_ref, out_mat, NORM_INF));
    }
};

struct RMatTest : public RMatTestBase
{
    template<typename In, typename Out>
    void run(const In& in, Out& out)
    {
        for (int i = 0; i < 2; i++)
        {
            comp.apply(cv::gin(in), cv::gout(out));
        }

        check();
    }
};

struct RMatTestStreaming : public RMatTestBase
{
    cv::GMatDesc getDesc(const RMat& m)
    {
        return cv::gapi::own::descr_of(m);
    }

    cv::GMatDesc getDesc(const cv::Mat& m)
    {
        return cv::descr_of(m);
    }

    void checkOutput(const cv::Mat&) { check(); }

    void checkOutput(const RMat& rm)
    {
        out_mat = rm.access();
        check();
    }

    template<typename In, typename Out>
    void run(const In& in, Out& out)
    {
        auto sc = comp.compileStreaming(getDesc(in));

        sc.setSource(cv::gin(in));
        sc.start();

        std::size_t num_frames = 0u;
        while (sc.pull(cv::gout(out)) && num_frames < 10u)
        {
            num_frames++;
            checkOutput(out);
        }
        EXPECT_EQ(10u, num_frames);
    }
};

template<typename RMatAdapterT>
struct RMatCreator
{
    RMat createRMat(cv::Mat& mat) { return {cv::gapi::own::make_rmat<RMatAdapterT>(mat)}; }
};

struct RMatTestCpuRef  : public RMatTest, RMatCreator<RMatAdapterRef>  {};
struct RMatTestCpuCopy : public RMatTest, RMatCreator<RMatAdapterCopy> {};
struct RMatTestCpuRefStreaming  : public RMatTestStreaming, RMatCreator<RMatAdapterRef>  {};
struct RMatTestCpuCopyStreaming : public RMatTestStreaming, RMatCreator<RMatAdapterCopy> {};

template<typename T>
struct RMatTypedTest : public ::testing::Test, public T {};

using RMatTestTypes = ::testing::Types< RMatTestCpuRef
                                      , RMatTestCpuCopy
                                      , RMatTestCpuRefStreaming
                                      , RMatTestCpuCopyStreaming
                                      >;

TYPED_TEST_CASE(RMatTypedTest, RMatTestTypes);

TYPED_TEST(RMatTypedTest, In)
{
    auto in_rmat = this->createRMat(this->in_mat);
    this->run(in_rmat, this->out_mat);
}

TYPED_TEST(RMatTypedTest, Out)
{
    auto out_rmat = this->createRMat(this->out_mat);
    this->run(this->in_mat, out_rmat);
}

TYPED_TEST(RMatTypedTest, InOut)
{
    auto  in_rmat = this->createRMat(this->in_mat);
    auto out_rmat = this->createRMat(this->out_mat);
    this->run(in_rmat, out_rmat);
}

class RMatAdapterForBackend : public RMat::Adapter
{
    int m_i;
public:
    RMatAdapterForBackend(int i) : m_i(i) {}
    virtual cv::Mat access() const override { return {}; }
    virtual GMatDesc desc() const override { return {}; }
    virtual void flush() const override {}
    int deviceSpecificData() const { return m_i; }
};

TEST(RMat, UsageInBackend)
{
    int i = std::rand();
    auto rmat = cv::gapi::own::make_rmat<RMatAdapterForBackend>(i);

    EXPECT_TRUE(rmat.is<RMatAdapterForBackend>());
    auto& adapter = rmat.as<RMatAdapterForBackend>();
    EXPECT_EQ(i, adapter.deviceSpecificData());
}

} // namespace opencv_test
