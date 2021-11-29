// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#include <type_traits>

// NB: emulate top-level of own classes:
// Linux gcc require to define own classes and `to_ocv` functions
// right before SFINAE `has_to_ocv` is declared
// This requirement satisfied in regular gapi include hierarchy
// so we need to repeat the same hierarchy here
namespace cv
{
struct Own {};
namespace gapi
{
namespace own
{
struct ConvertibleToOwn{};
struct NotConvertibleToOwn{};

static cv::Own to_ocv(const cv::gapi::own::ConvertibleToOwn&) {
    return {};
}
} // own
} // gapi
} // cv

#include "test_precomp.hpp"
#include <opencv2/gapi/gtype_traits.hpp>
#include <opencv2/gapi/util/util.hpp>

struct NoGhape {};
struct HasGShape {
    static constexpr cv::GShape shape = cv::GShape::GFRAME;
};
struct MimicGShape {
     static constexpr int shape = 0;
};

namespace opencv_test
{

TEST(GAPIUtil, AllSatisfy)
{
    static_assert(true == cv::detail::all_satisfy<std::is_integral, long, int, char>::value,
                  "[long, int, char] are all integral types");
    static_assert(true == cv::detail::all_satisfy<std::is_integral, char>::value,
                  "char is an integral type");

    static_assert(false == cv::detail::all_satisfy<std::is_integral, float, int, char>::value,
                  "[float, int, char] are NOT all integral types");
    static_assert(false == cv::detail::all_satisfy<std::is_integral, int, char, float>::value,
                  "[int, char, float] are NOT all integral types");
    static_assert(false == cv::detail::all_satisfy<std::is_integral, float>::value,
                  "float is not an integral types");
}

TEST(GAPIUtil, AllButLast)
{
    using test1 = cv::detail::all_but_last<long, int, float>::type;
    static_assert(true == cv::detail::all_satisfy<std::is_integral, test1>::value,
                  "[long, int] are all integral types (float skipped)");

    using test2 = cv::detail::all_but_last<int, float, char>::type;
    static_assert(false == cv::detail::all_satisfy<std::is_integral, test2>::value,
                  "[int, float] are NOT all integral types");
}

TEST(GAPIUtil, GShaped)
{
    static_assert(!cv::detail::has_gshape<NoGhape>::value, "NoGhape hasn't got any shape");
    static_assert(cv::detail::has_gshape<HasGShape>::value, "HasGShape has got GShape shape");
    static_assert(!cv::detail::has_gshape<MimicGShape>::value, "MimicGShape hasn't got right shape");
}

TEST(GAPIUtil, ToOcv)
{
    (void)to_ocv(cv::gapi::own::ConvertibleToOwn {});
    static_assert(!cv::detail::has_to_ocv<cv::gapi::own::NotConvertibleToOwn>::value, "NotConvertibleToOwn hasn't got `to_ocv`");
    static_assert(cv::detail::has_to_ocv<cv::gapi::own::ConvertibleToOwn>::value, "ConvertibleToOwn has got `to_ocv`");
}
} // namespace opencv_test
