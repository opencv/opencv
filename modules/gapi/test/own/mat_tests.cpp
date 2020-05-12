// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"
#include <opencv2/gapi/own/mat.hpp>
#include <opencv2/gapi/own/convert.hpp>
#include <opencv2/gapi/util/compiler_hints.hpp> //suppress_unused_warning

namespace opencv_test
{
using Mat = cv::gapi::own::Mat;
using Dims = std::vector<int>;

namespace {
inline std::size_t multiply_dims(Dims const& dims){
    return std::accumulate(dims.begin(), dims.end(), static_cast<size_t>(1), std::multiplies<std::size_t>());
}
}

TEST(OwnMat, DefaultConstruction)
{
    Mat m;
    ASSERT_EQ(m.data, nullptr);
    ASSERT_EQ(m.cols, 0);
    ASSERT_EQ(m.rows, 0);
    ASSERT_EQ(m.cols, 0);
    ASSERT_EQ(m.type(), 0);
    ASSERT_EQ(m.depth(), 0);
    ASSERT_TRUE(m.dims.empty());
    ASSERT_TRUE(m.empty());
}

TEST(OwnMat, Create)
{
    auto size = cv::gapi::own::Size{32,16};
    Mat m;
    m.create(size, CV_8UC1);

    ASSERT_NE(m.data, nullptr);
    ASSERT_EQ((cv::gapi::own::Size{m.cols, m.rows}), size);

    ASSERT_EQ(m.total(), static_cast<size_t>(size.height) * size.width);
    ASSERT_EQ(m.type(), CV_8UC1);
    ASSERT_EQ(m.depth(), CV_8U);
    ASSERT_EQ(m.channels(), 1);
    ASSERT_EQ(m.elemSize(), sizeof(uint8_t));
    ASSERT_EQ(m.step,   sizeof(uint8_t) * m.cols);
    ASSERT_TRUE(m.dims.empty());
    ASSERT_FALSE(m.empty());
}

TEST(OwnMat, CreateND)
{
    Dims dims = {1,1,32,32};
    Mat m;
    m.create(dims, CV_32F);

    ASSERT_NE(nullptr        , m.data      );
    ASSERT_EQ((cv::gapi::own::Size{0,0}), (cv::gapi::own::Size{m.cols, m.rows}));

    ASSERT_EQ(multiply_dims(dims), m.total());
    ASSERT_EQ(CV_32F         , m.type()    );
    ASSERT_EQ(CV_32F         , m.depth()   );
    ASSERT_EQ(-1             , m.channels());
    ASSERT_EQ(sizeof(float)  , m.elemSize());
    ASSERT_EQ(0u             , m.step      );
    ASSERT_EQ(dims           , m.dims      );
    ASSERT_FALSE(m.empty());
}

TEST(OwnMat, CreateOverload)
{
    auto size = cv::gapi::own::Size{32,16};
    Mat m;
    m.create(size.height,size.width, CV_8UC1);

    ASSERT_NE(m.data, nullptr);
    ASSERT_EQ((cv::Size{m.cols, m.rows}), size);

    ASSERT_EQ(m.total(), static_cast<size_t>(size.height) * size.width);
    ASSERT_EQ(m.type(), CV_8UC1);
    ASSERT_EQ(m.depth(), CV_8U);
    ASSERT_EQ(m.channels(), 1);
    ASSERT_EQ(m.elemSize(), sizeof(uint8_t));
    ASSERT_EQ(m.step,   sizeof(uint8_t) * m.cols);
    ASSERT_TRUE(m.dims.empty());
    ASSERT_FALSE(m.empty());
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
    ASSERT_TRUE(m.dims.empty());
    ASSERT_FALSE(m.empty());
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
                mat.channels(),
                mat.dims,
                mat.empty()
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

    EXPECT_EQ(0, cvtest::norm(cvMat, cvMatFromOwn, NORM_INF))
    << cvMat << std::endl
    << (cvMat != cvMatFromOwn);
}

TEST(OwnMatConversion, WithND)
{
    const Dims dims = {1,3,8,8};
    std::vector<uint8_t> data(dims[0]*dims[1]*dims[2]*dims[3]);
    for (size_t i = 0u; i < data.size(); i++)
    {
        data[i] = static_cast<uint8_t>(i);
    }
    cv::Mat cvMat(dims, CV_8U, data.data());
    auto ownMat = to_own(cvMat);
    auto cvMatFromOwn = cv::gapi::own::to_ocv(ownMat);

    EXPECT_EQ(0, cv::norm(cvMat, cvMatFromOwn, NORM_INF))
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
    EXPECT_EQ(0, cvtest::norm(to_ocv(mat), to_ocv(dst), NORM_INF))
    << to_ocv(mat) << std::endl
    << (to_ocv(mat) != to_ocv(dst));
}

TEST(OwnMat, AssignNDtoRegular)
{
    const auto sz   = cv::gapi::own::Size{32,32};
    const auto dims = Dims{1,3,224,224};

    Mat a;
    a.create(sz, CV_8U);
    const auto *old_ptr = a.data;

    ASSERT_NE(nullptr , a.data);
    ASSERT_EQ(sz      , (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(static_cast<size_t>(sz.width) * sz.height, a.total());
    ASSERT_EQ(CV_8U   , a.type());
    ASSERT_EQ(CV_8U   , a.depth());
    ASSERT_EQ(1       , a.channels());
    ASSERT_EQ(sizeof(uint8_t), a.elemSize());
    ASSERT_EQ(static_cast<size_t>(sz.width), a.step);
    ASSERT_TRUE(a.dims.empty());

    Mat b;
    b.create(dims, CV_32F);
    a = b;

    ASSERT_NE(nullptr , a.data);
    ASSERT_NE(old_ptr , a.data);
    ASSERT_EQ((cv::gapi::own::Size{0,0}), (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(multiply_dims(dims), a.total());
    ASSERT_EQ(CV_32F  , a.type());
    ASSERT_EQ(CV_32F  , a.depth());
    ASSERT_EQ(-1      , a.channels());
    ASSERT_EQ(sizeof(float), a.elemSize());
    ASSERT_EQ(0u      , a.step);
    ASSERT_EQ(dims    , a.dims);
}

TEST(OwnMat, AssignRegularToND)
{
    const auto sz   = cv::gapi::own::Size{32,32};
    const auto dims = Dims{1,3,224,224};

    Mat a;
    a.create(dims, CV_32F);
    const auto *old_ptr = a.data;

    ASSERT_NE(nullptr , a.data);
    ASSERT_EQ((cv::gapi::own::Size{0,0}), (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(multiply_dims(dims), a.total());
    ASSERT_EQ(CV_32F  , a.type());
    ASSERT_EQ(CV_32F  , a.depth());
    ASSERT_EQ(-1      , a.channels());
    ASSERT_EQ(sizeof(float), a.elemSize());
    ASSERT_EQ(0u      , a.step);
    ASSERT_EQ(dims    , a.dims);

    Mat b;
    b.create(sz, CV_8U);
    a = b;

    ASSERT_NE(nullptr , a.data);
    ASSERT_NE(old_ptr , a.data);
    ASSERT_EQ(sz      , (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(static_cast<size_t>(sz.width) * sz.height, a.total());
    ASSERT_EQ(CV_8U   , a.type());
    ASSERT_EQ(CV_8U   , a.depth());
    ASSERT_EQ(1       , a.channels());
    ASSERT_EQ(sizeof(uint8_t), a.elemSize());
    ASSERT_EQ(static_cast<size_t>(sz.width), a.step);
    ASSERT_TRUE(a.dims.empty());
}

TEST(OwnMat, CopyNDtoRegular)
{
    const auto sz   = cv::gapi::own::Size{32,32};
    const auto dims = Dims{1,3,224,224};

    Mat a;
    a.create(sz, CV_8U);
    const auto *old_ptr = a.data;

    ASSERT_NE(nullptr , a.data);
    ASSERT_EQ(sz      , (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(static_cast<size_t>(sz.width) * sz.height, a.total());
    ASSERT_EQ(CV_8U   , a.type());
    ASSERT_EQ(CV_8U   , a.depth());
    ASSERT_EQ(1       , a.channels());
    ASSERT_EQ(sizeof(uint8_t), a.elemSize());
    ASSERT_EQ(static_cast<size_t>(sz.width), a.step);
    ASSERT_TRUE(a.dims.empty());

    Mat b;
    b.create(dims, CV_32F);
    b.copyTo(a);

    ASSERT_NE(nullptr , a.data);
    ASSERT_NE(old_ptr , a.data);
    ASSERT_NE(b.data  , a.data);
    ASSERT_EQ((cv::gapi::own::Size{0,0}), (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(multiply_dims(dims), a.total());
    ASSERT_EQ(CV_32F  , a.type());
    ASSERT_EQ(CV_32F  , a.depth());
    ASSERT_EQ(-1      , a.channels());
    ASSERT_EQ(sizeof(float), a.elemSize());
    ASSERT_EQ(0u      , a.step);
    ASSERT_EQ(dims    , a.dims);
}

TEST(OwnMat, CopyRegularToND)
{
    const auto sz   = cv::gapi::own::Size{32,32};
    const auto dims = Dims{1,3,224,224};

    Mat a;
    a.create(dims, CV_32F);
    const auto *old_ptr = a.data;


    ASSERT_NE(nullptr , a.data);
    ASSERT_EQ((cv::gapi::own::Size{0,0}), (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(multiply_dims(dims), a.total());
    ASSERT_EQ(CV_32F  , a.type());
    ASSERT_EQ(CV_32F  , a.depth());
    ASSERT_EQ(-1      , a.channels());
    ASSERT_EQ(sizeof(float), a.elemSize());
    ASSERT_EQ(0u      , a.step);
    ASSERT_EQ(dims    , a.dims);

    Mat b;
    b.create(sz, CV_8U);
    b.copyTo(a);

    ASSERT_NE(nullptr , a.data);
    ASSERT_NE(old_ptr , a.data);
    ASSERT_NE(b.data  , a.data);
    ASSERT_EQ(sz      , (cv::gapi::own::Size{a.cols, a.rows}));
    ASSERT_EQ(static_cast<size_t>(sz.width) * sz.height, a.total());
    ASSERT_EQ(CV_8U   , a.type());
    ASSERT_EQ(CV_8U   , a.depth());
    ASSERT_EQ(1       , a.channels());
    ASSERT_EQ(sizeof(uint8_t), a.elemSize());
    ASSERT_EQ(static_cast<size_t>(sz.width), a.step);
    ASSERT_TRUE(a.dims.empty());
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
    EXPECT_EQ(0, cvtest::norm(cmp_result_mat, NORM_INF))
        << cmp_result_mat;
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
    EXPECT_EQ(0, cvtest::norm(cmp_result_mat, NORM_INF))
        << cmp_result_mat;
}

TEST(OwnMat, ScalarAssignND)
{
    std::vector<int> dims = {1,1000};
    Mat m;
    m.create(dims, CV_32F);
    m = cv::gapi::own::Scalar{-1};
    const float *ptr = reinterpret_cast<float*>(m.data);

    for (auto i = 0u; i < m.total(); i++) {
        EXPECT_EQ(-1.f, ptr[i]);
    }
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
    EXPECT_EQ(0, cvtest::norm(cmp_result_mat, NORM_INF))
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
    EXPECT_EQ(0, cvtest::norm(cmp_result_mat, NORM_INF))
        << cmp_result_mat   << std::endl
        << to_ocv(roi_view) << std::endl
        << expected_cv_mat  << std::endl;
}

TEST(OwnMat, CreateWithNegativeDims)
{
    Mat own_mat;
    ASSERT_ANY_THROW(own_mat.create(cv::Size{-1, -1}, CV_8U));
}

TEST(OwnMat, CreateWithNegativeWidth)
{
    Mat own_mat;
    ASSERT_ANY_THROW(own_mat.create(cv::Size{-1, 1}, CV_8U));
}

TEST(OwnMat, CreateWithNegativeHeight)
{
    Mat own_mat;
    ASSERT_ANY_THROW(own_mat.create(cv::Size{1, -1}, CV_8U));
}

TEST(OwnMat, ZeroHeightMat)
{
    cv::GMat in, a, b, c, d;
    std::tie(a, b, c, d) = cv::gapi::split4(in);
    cv::GMat out = cv::gapi::merge3(a, b, c);
    cv::Mat in_mat(cv::Size(8, 0), CV_8UC4);
    cv::Mat out_mat(cv::Size(8, 8), CV_8UC3);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    ASSERT_ANY_THROW(comp.apply(cv::gin(in_mat), cv::gout(out_mat),
        cv::compile_args(cv::gapi::core::fluid::kernels())));
}

TEST(OwnMat, ZeroWidthMat)
{
    cv::GMat in, a, b, c, d;
    std::tie(a, b, c, d) = cv::gapi::split4(in);
    cv::GMat out = cv::gapi::merge3(a, b, c);
    cv::Mat in_mat(cv::Size(0, 8), CV_8UC4);
    cv::Mat out_mat(cv::Size(8, 8), CV_8UC3);
    cv::GComputation comp(cv::GIn(in), cv::GOut(out));
    ASSERT_ANY_THROW(comp.apply(cv::gin(in_mat), cv::gout(out_mat),
        cv::compile_args(cv::gapi::core::fluid::kernels())));
}

} // namespace opencv_test
