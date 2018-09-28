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

    ASSERT_EQ(m.total(), static_cast<size_t>(size.height*size.width));
    ASSERT_EQ(m.type(), CV_8UC1);
    ASSERT_EQ(m.depth(), CV_8U);
    ASSERT_EQ(m.channels(), 1);
    ASSERT_EQ(m.elemSize(), sizeof(uint8_t));
    ASSERT_EQ(m.step,   sizeof(uint8_t) * m.cols);
}

TEST(OwnMat, CreateOverload)
{
    auto size = cv::gapi::own::Size{32,16};
    Mat m;
    m.create(size.height,size.width, CV_8UC1);

    ASSERT_NE(m.data, nullptr);
    ASSERT_EQ((cv::Size{m.cols, m.rows}), size);

    ASSERT_EQ(m.total(), static_cast<size_t>(size.height*size.width));
    ASSERT_EQ(m.type(), CV_8UC1);
    ASSERT_EQ(m.depth(), CV_8U);
    ASSERT_EQ(m.channels(), 1);
    ASSERT_EQ(m.elemSize(), sizeof(uint8_t));
    ASSERT_EQ(m.step,   sizeof(uint8_t) * m.cols);
}
TEST(OwnMat, Create3chan)
{
    auto size = cv::Size{32,16};
    Mat m;
    m.create(size, CV_8UC3);

    ASSERT_NE(m.data, nullptr);
    ASSERT_EQ((cv::Size{m.cols, m.rows}), size);

    ASSERT_EQ(m.type(), CV_8UC3);
    ASSERT_EQ(m.depth(), CV_8U);
    ASSERT_EQ(m.channels(), 3);
    ASSERT_EQ(m.elemSize(), 3 * sizeof(uint8_t));
    ASSERT_EQ(m.step,       3*  sizeof(uint8_t) * m.cols);
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

TEST(OwnMat, PtrWithStep)
{
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<int, height * stepInPixels> data;
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<int>(i);
    }
    Mat mat(height, width, CV_32S, data.data(), stepInPixels * sizeof(int));

    EXPECT_EQ(& data[0],                reinterpret_cast<int*>(mat.ptr(0)));
    EXPECT_EQ(& data[1],                reinterpret_cast<int*>(mat.ptr(0, 1)));
    EXPECT_EQ(& data[stepInPixels],     reinterpret_cast<int*>(mat.ptr(1)));
    EXPECT_EQ(& data[stepInPixels +1],  reinterpret_cast<int*>(mat.ptr(1,1)));

    auto const& cmat = mat;

    EXPECT_EQ(& data[0],                reinterpret_cast<const int*>(cmat.ptr(0)));
    EXPECT_EQ(& data[1],                reinterpret_cast<const int*>(cmat.ptr(0, 1)));
    EXPECT_EQ(& data[stepInPixels],     reinterpret_cast<const int*>(cmat.ptr(1)));
    EXPECT_EQ(& data[stepInPixels +1],  reinterpret_cast<const int*>(cmat.ptr(1,1)));
}

TEST(OwnMat, CopyToWithStep)
{
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<int, height * stepInPixels> data;
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<int>(i);
    }
    Mat mat(height, width, CV_32S, data.data(), stepInPixels * sizeof(int));

    Mat dst;
    mat.copyTo(dst);

    EXPECT_NE(mat.data, dst.data);
    EXPECT_EQ(0, cv::countNonZero(to_ocv(mat) != to_ocv(dst)))
    << to_ocv(mat) << std::endl
    << (to_ocv(mat) != to_ocv(dst));
}

TEST(OwnMat, ScalarAssign32SC1)
{
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<int, height * stepInPixels> data;
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<int>(i);
    }
    Mat mat(height, width, CV_32S, data.data(), stepInPixels * sizeof(data[0]));

    mat = cv::gapi::own::Scalar{-1};

    std::array<int, height * stepInPixels> expected;

    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < stepInPixels; col++)
        {
            auto index = row*stepInPixels + col;
            expected[index] = col < width ? -1 : static_cast<int>(index);
        }
    }

    auto cmp_result_mat = (cv::Mat{height, stepInPixels, CV_32S, data.data()} != cv::Mat{height, stepInPixels, CV_32S, expected.data()});
    EXPECT_EQ(0, cv::countNonZero(cmp_result_mat))
    << cmp_result_mat << std::endl;
}

TEST(OwnMat, ScalarAssign8UC1)
{
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<uchar, height * stepInPixels> data;
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<uchar>(i);
    }
    Mat mat(height, width, CV_8U, data.data(), stepInPixels * sizeof(data[0]));

    mat = cv::gapi::own::Scalar{-1};

    std::array<uchar, height * stepInPixels> expected;

    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < stepInPixels; col++)
        {
            auto index = row*stepInPixels + col;
            expected[index] = col < width ? cv::saturate_cast<uchar>(-1) : static_cast<uchar>(index);
        }
    }

    auto cmp_result_mat = (cv::Mat{height, stepInPixels, CV_8U, data.data()} != cv::Mat{height, stepInPixels, CV_8U, expected.data()});
    EXPECT_EQ(0, cv::countNonZero(cmp_result_mat))
    << cmp_result_mat << std::endl;
}

TEST(OwnMat, ScalarAssign8UC3)
{
    constexpr auto cv_type = CV_8SC3;
    constexpr int channels = 3;
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<schar, height * stepInPixels * channels> data;
    for (size_t i = 0; i < data.size(); i+= channels)
    {
        data[i + 0] = static_cast<schar>(10 * i + 0);
        data[i + 1] = static_cast<schar>(10 * i + 1);
        data[i + 2] = static_cast<schar>(10 * i + 2);
    }

    Mat mat(height, width, cv_type, data.data(), channels * stepInPixels * sizeof(data[0]));

    mat = cv::gapi::own::Scalar{-10, -11, -12};

    std::array<schar, data.size()> expected;

    for (size_t row = 0; row < height; row++)
    {
        for (size_t col = 0; col < stepInPixels; col++)
        {
            int index = static_cast<int>(channels * (row*stepInPixels + col));
            expected[index + 0] = static_cast<schar>(col < width ? -10 : 10 * index + 0);
            expected[index + 1] = static_cast<schar>(col < width ? -11 : 10 * index + 1);
            expected[index + 2] = static_cast<schar>(col < width ? -12 : 10 * index + 2);
        }
    }

    auto cmp_result_mat = (cv::Mat{height, stepInPixels, cv_type, data.data()} != cv::Mat{height, stepInPixels, cv_type, expected.data()});
    EXPECT_EQ(0, cv::countNonZero(cmp_result_mat))
    << cmp_result_mat << std::endl
    << "data : " << std::endl
    << cv::Mat{height, stepInPixels, cv_type, data.data()}     << std::endl
    << "expected : " << std::endl
    << cv::Mat{height, stepInPixels, cv_type, expected.data()} << std::endl;
}

TEST(OwnMat, ROIView)
{
    constexpr int width  = 8;
    constexpr int height = 8;
    constexpr int stepInPixels = 16;

    std::array<uchar, height * stepInPixels> data;
    for (size_t i = 0; i < data.size(); i++)
    {
        data[i] = static_cast<uchar>(i);
    }


//    std::cout<<cv::Mat{height, stepInPixels, CV_8U, data.data()}<<std::endl;

    std::array<uchar, 4 * 4> expected;

    for (size_t row = 0; row < 4; row++)
    {
        for (size_t col = 0; col < 4; col++)
        {
            expected[row*4 +col] = static_cast<uchar>(stepInPixels * (2 + row) + 2 + col);
        }
    }

    Mat mat(height, width, CV_8U, data.data(), stepInPixels * sizeof(data[0]));
    Mat roi_view (mat, cv::gapi::own::Rect{2,2,4,4});

//    std::cout<<cv::Mat{4, 4, CV_8U, expected.data()}<<std::endl;
//
    auto expected_cv_mat = cv::Mat{4, 4, CV_8U, expected.data()};

    auto cmp_result_mat = (to_ocv(roi_view) != expected_cv_mat);
    EXPECT_EQ(0, cv::countNonZero(cmp_result_mat))
    << cmp_result_mat   << std::endl
    << to_ocv(roi_view) << std::endl
    << expected_cv_mat  << std::endl;
}
} // namespace opencv_test
