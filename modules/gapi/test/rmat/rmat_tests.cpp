// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"
#include <opencv2/gapi/rmat.hpp>
#include "rmat_test_common.hpp"

namespace opencv_test {
namespace {
void randomizeMat(cv::Mat& m) {
    auto ref = m.clone();
    while (cv::norm(m, ref, cv::NORM_INF) == 0) {
        cv::randu(m, cv::Scalar::all(127), cv::Scalar::all(40));
    }
}

template <typename RMatAdapterT>
struct RMatTest {
    using AdapterT = RMatAdapterT;
    RMatTest()
        : m_deviceMat(cv::Mat::zeros(8,8,CV_8UC1))
        , m_rmat(make_rmat<RMatAdapterT>(m_deviceMat, m_callbackCalled)) {
        randomizeMat(m_deviceMat);
        expectNoCallbackCalled();
    }

    RMat& rmat() { return m_rmat; }
    cv::Mat cloneDeviceMat() { return m_deviceMat.clone(); }
    void expectCallbackCalled() { EXPECT_TRUE(m_callbackCalled); }
    void expectNoCallbackCalled() { EXPECT_FALSE(m_callbackCalled); }

    void expectDeviceDataEqual(const cv::Mat& mat) {
        EXPECT_EQ(0, cv::norm(mat, m_deviceMat, NORM_INF));
    }
    void expectDeviceDataNotEqual(const cv::Mat& mat) {
        EXPECT_NE(0, cv::norm(mat, m_deviceMat, NORM_INF));
    }

private:
    cv::Mat m_deviceMat;
    bool m_callbackCalled = false;
    cv::RMat m_rmat;
};
} // anonymous namespace

template<typename T>
struct RMatTypedTest : public ::testing::Test, public T { using Type = T; };

using RMatTestTypes = ::testing::Types< RMatTest<RMatAdapterRef>
                                      , RMatTest<RMatAdapterCopy>
                                      >;

TYPED_TEST_CASE(RMatTypedTest, RMatTestTypes);

TYPED_TEST(RMatTypedTest, Smoke) {
    auto view = this->rmat().access(RMat::Access::R);
    auto matFromDevice = cv::Mat(view.size(), view.type(), view.ptr());
    EXPECT_TRUE(cv::descr_of(this->cloneDeviceMat()) == this->rmat().desc());
    this->expectDeviceDataEqual(matFromDevice);
}

static Mat asMat(RMat::View& view) {
    return Mat(view.size(), view.type(), view.ptr(), view.step());
}

TYPED_TEST(RMatTypedTest, BasicWorkflow) {
    {
        auto view = this->rmat().access(RMat::Access::R);
        this->expectDeviceDataEqual(asMat(view));
    }
    this->expectNoCallbackCalled();

    cv::Mat dataToWrite = this->cloneDeviceMat();
    randomizeMat(dataToWrite);
    this->expectDeviceDataNotEqual(dataToWrite);
    {
        auto view = this->rmat().access(RMat::Access::W);
        dataToWrite.copyTo(asMat(view));
    }
    this->expectCallbackCalled();
    this->expectDeviceDataEqual(dataToWrite);
}

TEST(RMat, TestEmptyAdapter) {
    RMat rmat;
    EXPECT_ANY_THROW(rmat.get<RMatAdapterCopy>());
}

TYPED_TEST(RMatTypedTest, CorrectAdapterCast) {
    using T = typename TestFixture::Type::AdapterT;
    EXPECT_NE(nullptr, this->rmat().template get<T>());
}

class DummyAdapter : public RMat::Adapter {
    virtual RMat::View access(RMat::Access) override { return {}; }
    virtual cv::GMatDesc desc() const override { return {}; }
};

TYPED_TEST(RMatTypedTest, IncorrectAdapterCast) {
    EXPECT_EQ(nullptr, this->rmat().template get<DummyAdapter>());
}

class RMatAdapterForBackend : public RMat::Adapter {
    int m_i;
public:
    RMatAdapterForBackend(int i) : m_i(i) {}
    virtual RMat::View access(RMat::Access) override { return {}; }
    virtual GMatDesc desc() const override { return {}; }
    int deviceSpecificData() const { return m_i; }
};

// RMat's usage scenario in the backend:
// we have some specific data hidden under RMat,
// test that we can obtain it via RMat.as<T>() method
TEST(RMat, UsageInBackend) {
    int i = 123456;
    auto rmat = cv::make_rmat<RMatAdapterForBackend>(i);

    auto adapter = rmat.get<RMatAdapterForBackend>();
    ASSERT_NE(nullptr, adapter);
    EXPECT_EQ(i, adapter->deviceSpecificData());
}
} // namespace opencv_test
