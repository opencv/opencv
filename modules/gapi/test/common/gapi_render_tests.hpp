// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_RENDER_TESTS_HPP
#define OPENCV_GAPI_RENDER_TESTS_HPP

#include "gapi_tests_common.hpp"

namespace opencv_test
{

template<typename ...SpecificParams>
struct RenderParams : public Params<SpecificParams...>
{
    using common_params_t = std::tuple<cv::Size>;
    using specific_params_t = std::tuple<SpecificParams...>;
    using params_t = std::tuple<cv::Size, SpecificParams...>;

    static constexpr const size_t common_params_size = std::tuple_size<common_params_t>::value;
    static constexpr const size_t specific_params_size = std::tuple_size<specific_params_t>::value;

    template<size_t I>
    static const typename std::tuple_element<I, common_params_t>::type&
    getCommon(const params_t& t)
    {
        static_assert(I < common_params_size, "Index out of range");
        return std::get<I>(t);
    }

    template<size_t I>
    static const typename std::tuple_element<I, specific_params_t>::type&
    getSpecific(const params_t& t)
    {
        static_assert(specific_params_size > 0,
            "Impossible to call this function: no specific parameters specified");
        static_assert(I < specific_params_size, "Index out of range");
        return std::get<common_params_size + I>(t);
    }
};

template<typename ...SpecificParams>
struct RenderTestBase : public TestWithParam<typename RenderParams<SpecificParams...>::params_t>
{
    using AllParams = RenderParams<SpecificParams...>;

    // Get common (pre-defined) parameter value by index
    template<size_t I>
    inline auto getCommonParam() const
        -> decltype(AllParams::template getCommon<I>(this->GetParam()))
    {
        return AllParams::template getCommon<I>(this->GetParam());
    }

    // Get specific (user-defined) parameter value by index
    template<size_t I>
    inline auto getSpecificParam() const
        -> decltype(AllParams::template getSpecific<I>(this->GetParam()))
    {
        return AllParams::template getSpecific<I>(this->GetParam());
    }

    cv::Size sz_ = getCommonParam<0>();
};

template <typename ...Args>
class RenderBGRTestBase : public RenderTestBase<Args...>
{
protected:
    void Init(const cv::Size& sz)
    {
        MatType type = CV_8UC3;

        ref_mat.create(sz, type);
        gapi_mat.create(sz, type);

        cv::randu(ref_mat, cv::Scalar::all(0), cv::Scalar::all(255));
        ref_mat.copyTo(gapi_mat);
    }

    cv::Mat gapi_mat, ref_mat;
};

template <typename ...Args>
class RenderNV12TestBase : public RenderTestBase<Args...>
{
protected:
    void Init(const cv::Size& sz)
    {
        auto create_rand_mats = [](const cv::Size& size, MatType type, cv::Mat& ref_mat, cv::Mat& gapi_mat) {
            ref_mat.create(size, type);
            cv::randu(ref_mat, cv::Scalar::all(0), cv::Scalar::all(255));
            ref_mat.copyTo(gapi_mat);
        };

        create_rand_mats(sz,     CV_8UC1, y_ref_mat  , y_gapi_mat);
        create_rand_mats(sz / 2, CV_8UC2, uv_ref_mat , uv_gapi_mat);
    }

    cv::Mat y_ref_mat, uv_ref_mat, y_gapi_mat, uv_gapi_mat;
};

cv::Scalar cvtBGRToYUVC(const cv::Scalar& bgr);
void drawMosaicRef(const cv::Mat& mat, const cv::Rect &rect, int cellSz);
void blendImageRef(cv::Mat& mat,
                   const cv::Point& org,
                   const cv::Mat& img,
                   const cv::Mat& alpha);

#define GAPI_RENDER_TEST_FIXTURE_NV12(Fixture, API, Number, ...)  \
struct Fixture : public RenderNV12TestBase API {                  \
    __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__))   \
    Fixture() {                                                   \
        Init(sz_);                                                \
    };                                                            \
};

#define GAPI_RENDER_TEST_FIXTURE_BGR(Fixture, API, Number, ...)  \
struct Fixture : public RenderBGRTestBase API {                  \
    __WRAP_VAARGS(DEFINE_SPECIFIC_PARAMS_##Number(__VA_ARGS__))   \
    Fixture() {                                                   \
        Init(sz_);                                                \
    };                                                            \
};

#define GET_VA_ARGS(...) __VA_ARGS__
#define GAPI_RENDER_TEST_FIXTURES(Fixture, API, Number, ...)                    \
    GAPI_RENDER_TEST_FIXTURE_BGR(RenderBGR##Fixture,   GET_VA_ARGS(API), Number, __VA_ARGS__) \
    GAPI_RENDER_TEST_FIXTURE_NV12(RenderNV12##Fixture, GET_VA_ARGS(API), Number, __VA_ARGS__) \

using Points = std::vector<cv::Point>;
GAPI_RENDER_TEST_FIXTURES(TestTexts,     FIXTURE_API(std::string, cv::Point, double, cv::Scalar), 4, text, org, fs, color)
GAPI_RENDER_TEST_FIXTURES(TestRects,     FIXTURE_API(cv::Rect, cv::Scalar, int),                  3, rect, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestCircles,   FIXTURE_API(cv::Point, int, cv::Scalar, int),            4, center, radius, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestLines,     FIXTURE_API(cv::Point, cv::Point, cv::Scalar, int),      4, pt1, pt2, color, thick)
GAPI_RENDER_TEST_FIXTURES(TestMosaics,   FIXTURE_API(cv::Rect, int, int),                         3, mos, cellsz, decim)
GAPI_RENDER_TEST_FIXTURES(TestImages,    FIXTURE_API(cv::Rect, cv::Scalar, double),               3, rect, color, transparency)
GAPI_RENDER_TEST_FIXTURES(TestPolylines, FIXTURE_API(Points, cv::Scalar, int),                    3, points, color, thick)

} // opencv_test

#endif //OPENCV_GAPI_RENDER_TESTS_HPP
