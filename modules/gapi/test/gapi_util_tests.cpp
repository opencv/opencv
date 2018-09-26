// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <type_traits>

#include "opencv2/gapi/util/util.hpp"

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

} // namespace opencv_test
