// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"
#include <opencv2/gapi/rmat.hpp>

#include <opencv2/gapi/util/compiler_hints.hpp>
#include "../src/backends/common/gbackend.hpp"

namespace opencv_test
{
using cv::GMatDesc;
using View = cv::RMat::View;
using cv::Mat;
using namespace ::testing;

static void expect_eq_desc(const GMatDesc& desc, const View& view) {
    EXPECT_EQ(desc.size, view.size());
    EXPECT_EQ(desc.dims, view.dims());
    EXPECT_EQ(desc.size.width, view.cols());
    EXPECT_EQ(desc.size.height, view.rows());
    EXPECT_EQ(CV_MAKE_TYPE(desc.depth,desc.chan), view.type());
    EXPECT_EQ(desc.depth, view.depth());
    EXPECT_EQ(desc.chan, view.chan());
}

TEST(RMatView, TestDefaultConstruction) {
    View view;
    GMatDesc desc{};
    expect_eq_desc(desc, view);
    EXPECT_EQ(nullptr, view.ptr());
    EXPECT_EQ(0u, view.step());
}

struct RMatViewTest : public TestWithParam<int /*dataType*/>{};
TEST_P(RMatViewTest, ConstructionFromMat) {
    auto type = GetParam();
    Mat mat(8,8,type);
    const auto desc = cv::descr_of(mat);
    View view(cv::descr_of(mat), mat.ptr(), mat.step1());
    expect_eq_desc(desc, view);
    EXPECT_EQ(mat.ptr(), view.ptr());
    EXPECT_EQ(mat.step1(), view.step());
}

TEST(RMatView, TestConstructionFromMatND) {
    std::vector<int> dims(4, 8);
    Mat mat(dims, CV_8UC1);
    const auto desc = cv::descr_of(mat);
    View view(cv::descr_of(mat), mat.ptr());
    expect_eq_desc(desc, view);
    EXPECT_EQ(mat.ptr(), view.ptr());
}

TEST_P(RMatViewTest, DefaultStep) {
    auto type = GetParam();
    GMatDesc desc;
    desc.chan = CV_MAT_CN(type);
    desc.depth = CV_MAT_DEPTH(type);
    desc.size = {8,8};
    std::vector<unsigned char> data(desc.size.width*desc.size.height*CV_ELEM_SIZE(type));
    View view(desc, data.data());
    EXPECT_EQ(static_cast<size_t>(desc.size.width)*CV_ELEM_SIZE(type), view.step());
}

static Mat asMat(View& view) {
    return Mat(view.size(), view.type(), view.ptr(), view.step());
}

TEST_P(RMatViewTest, NonDefaultStepInput) {
    auto type = GetParam();
    Mat bigMat(16,16,type);
    cv::randn(bigMat, cv::Scalar::all(127), cv::Scalar::all(40));
    Mat mat = bigMat(cv::Rect{4,4,8,8});
    View view(cv::descr_of(mat), mat.data, mat.step);
    const auto viewMat = asMat(view);
    Mat ref, out;
    cv::Size ksize{1,1};
    cv::blur(viewMat, out, ksize);
    cv::blur(    mat, ref, ksize);
    EXPECT_EQ(0, cvtest::norm(ref, out, NORM_INF));
}

TEST_P(RMatViewTest, NonDefaultStepOutput) {
    auto type = GetParam();
    Mat mat(8,8,type);
    cv::randn(mat, cv::Scalar::all(127), cv::Scalar::all(40));
    Mat bigMat = Mat::zeros(16,16,type);
    Mat out = bigMat(cv::Rect{4,4,8,8});
    View view(cv::descr_of(out), out.ptr(), out.step);
    auto viewMat = asMat(view);
    Mat ref;
    cv::Size ksize{1,1};
    cv::blur(mat, viewMat, ksize);
    cv::blur(mat, ref,     ksize);
    EXPECT_EQ(0, cvtest::norm(ref, out, NORM_INF));
}

INSTANTIATE_TEST_CASE_P(Test, RMatViewTest,
                        Values(CV_8UC1, CV_8UC3, CV_32FC1));

struct RMatViewCallbackTest : public ::testing::Test {
    RMatViewCallbackTest()
        : mat(8,8,CV_8UC1) {
        cv::randn(mat, cv::Scalar::all(127), cv::Scalar::all(40));
    }
    View getView() { return {cv::descr_of(mat), mat.ptr(), mat.step1(), [this](){ callbackCalls++; }}; }
    int callbackCalls = 0;
    Mat mat;
};

TEST_F(RMatViewCallbackTest, MoveCtor) {
    {
        View copy(getView());
        cv::util::suppress_unused_warning(copy);
        EXPECT_EQ(0, callbackCalls);
    }
    EXPECT_EQ(1, callbackCalls);
}

TEST_F(RMatViewCallbackTest, MoveCopy) {
    {
        View copy;
        copy = getView();
        cv::util::suppress_unused_warning(copy);
        EXPECT_EQ(0, callbackCalls);
    }
    EXPECT_EQ(1, callbackCalls);
}

static int firstElement(const View& view) { return *view.ptr(); }
static void setFirstElement(View& view, uchar value) { *view.ptr() = value; }

TEST_F(RMatViewCallbackTest, MagazineInteraction) {
    cv::gimpl::magazine::Class<View> mag;
    constexpr int rc = 1;
    constexpr uchar value = 11;
    mag.slot<View>()[rc] = getView();
    {
        auto& mag_view = mag.slot<View>()[rc];
        setFirstElement(mag_view, value);
        auto mag_el = firstElement(mag_view);
        EXPECT_EQ(value, mag_el);
    }
    {
        const auto& mag_view = mag.slot<View>()[rc];
        auto mag_el = firstElement(mag_view);
        EXPECT_EQ(value, mag_el);
    }
    EXPECT_EQ(0, callbackCalls);
    mag.slot<View>().erase(rc);
    EXPECT_EQ(1, callbackCalls);
}
} // namespace opencv_test
