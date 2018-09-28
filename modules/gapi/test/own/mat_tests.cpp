// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "opencv2/gapi/own/mat.hpp"
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning

namespace opencv_test
{
using Mat = cv::gapi::own::Mat;

TEST(OwnMat, DefaultConstruction)
{
    Mat m;
    ASSERT_EQ(m.data, nullptr);
    ASSERT_EQ(m.cols, 0);
    ASSERT_EQ(m.rows, 0);
    ASSERT_EQ(m.cols, 0);
    ASSERT_EQ(m.type(), 0);
    ASSERT_EQ(m.depth(), 0);
}

TEST(OwnMat, Create)
{
    auto size = cv::gapi::own::Size{32,16};
    Mat m;
    m.create(size, CV_8UC1);

    ASSERT_NE(m.data, nullptr);
    ASSERT_EQ((cv::gapi::own::Size{m.cols, m.rows}), size);

    ASSERT_EQ(m.type(), CV_8UC1);
    ASSERT_EQ(m.depth(), CV_8U);
    ASSERT_EQ(m.channels(), 1);
    ASSERT_EQ(m.step,   sizeof(uint8_t) * m.cols);
}

struct NonEmptyMat {
    cv::gapi::own::Size size{32,16};
    Mat m;
    NonEmptyMat() {
        m.create(size, CV_8UC1);
    }
};

struct OwnMatSharedSemantics : NonEmptyMat, ::testing::Test {};


namespace {
    auto state_of = [](Mat const& mat) {
        return std::make_tuple(
                mat.data,
                cv::Size{mat.cols, mat.rows},
                mat.type(),
                mat.depth(),
                mat.channels()
        );
    };

    void ensure_mats_are_same(Mat const& copy, Mat const& m){
        EXPECT_NE(copy.data, nullptr);
        EXPECT_EQ(state_of(copy), state_of(m));
    }
}
TEST_F(OwnMatSharedSemantics, CopyConstruction)
{
    Mat copy(m);
    ensure_mats_are_same(copy, m);
}

TEST_F(OwnMatSharedSemantics, CopyAssignment)
{
    Mat copy;
    copy = m;
    ensure_mats_are_same(copy, m);
}

struct OwnMatMoveSemantics : NonEmptyMat, ::testing::Test {
    Mat& moved_from = m;
    decltype(state_of(moved_from)) initial_state = state_of(moved_from);

    void ensure_state_moved_to(Mat const& moved_to)
    {
        EXPECT_EQ(state_of(moved_to),     initial_state);
        EXPECT_EQ(state_of(moved_from),   state_of(Mat{}));
    }
};

TEST_F(OwnMatMoveSemantics, MoveConstruction)
{
    Mat moved_to(std::move(moved_from));

    ensure_state_moved_to(moved_to);
}

TEST_F(OwnMatMoveSemantics, MoveAssignment)
{
    Mat moved_to(std::move(moved_from));
    ensure_state_moved_to(moved_to);
}

struct OwnMatNonOwningView : NonEmptyMat, ::testing::Test {
    decltype(state_of(m)) initial_state = state_of(m);

    void TearDown() override {
        EXPECT_EQ(state_of(m), initial_state)<<"State of the source matrix changed?";
        //ASAN should complain here if memory is freed here (e.g. by bug in non owning logic of own::Mat)
        volatile uchar dummy =  m.data[0];
        cv::util::suppress_unused_warning(dummy);
    }

};

TEST_F(OwnMatNonOwningView, Construction)
{
    Mat non_owning_view(m.rows, m.cols, m.type(), static_cast<void*>(m.data));

    ensure_mats_are_same(non_owning_view, m);
}

TEST_F(OwnMatNonOwningView, CopyConstruction)
{
    Mat non_owning_view{m.rows, m.cols, m.type(), static_cast<void*>(m.data)};

    Mat non_owning_view_copy = non_owning_view;
    ensure_mats_are_same(non_owning_view_copy, m);
}

TEST_F(OwnMatNonOwningView, Assignment)
{
    Mat non_owning_view{m.rows, m.cols, m.type(), static_cast<void*>(m.data)};
    Mat non_owning_view_copy;

    non_owning_view_copy = non_owning_view;
    ensure_mats_are_same(non_owning_view_copy, m);
}

TEST(OwnMatConversion, WithStep)
{
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<int, height * stepInPixels> data;
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<int>(i);
    }
    cv::Mat cvMat(cv::Size{width, height}, CV_32S, data.data(), stepInPixels * sizeof(int));

    auto ownMat = to_own(cvMat);
    auto cvMatFromOwn = cv::gapi::own::to_ocv(ownMat);

    EXPECT_EQ(0, cv::countNonZero(cvMat != cvMatFromOwn))
    << cvMat << std::endl
    << (cvMat != cvMatFromOwn);
}


} // namespace opencv_test
