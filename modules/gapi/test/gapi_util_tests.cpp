// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include <type_traits>

#include <opencv2/gapi/util/util.hpp>
#include <opencv2/gapi/gtags.hpp>

namespace cv {
namespace gapi{
namespace tag {
struct Tag1 {};
struct Tag2 {};
struct Tag3 {};
}
}
}
class NonTagged {};

class HasTag1 {
public:
    GAPI_OBJECT (Tag1)
};
class HasTag2 {
public:
    GAPI_OBJECT (Tag2)
};
class HasTag23 {
public:
    GAPI_OBJECT_2 (Tag2, Tag3)
};
class HasTag123 {
public:
    GAPI_OBJECT_3 (Tag1, Tag2, Tag3)
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

TEST(GAPIUtil, GTags_NonTagged)
{
    using namespace cv::gapi::tag;
    static_assert(!cv::gapi::has_tag<NonTagged, Tag1>::value, "NonTagged hasn't got Tag1");
    static_assert(!cv::gapi::has_tag<NonTagged, Tag2>::value, "NonTagged hasn't got Tag2");
    static_assert(!cv::gapi::has_tag<NonTagged, Tag3>::value, "NonTagged hasn't got Tag3");
}

TEST(GAPIUtil, GTags_HasTag1)
{
    using namespace cv::gapi::tag;
    static_assert(cv::gapi::has_tag<HasTag1, Tag1>::value, "HasTag1 has got Tag1");
    static_assert(!cv::gapi::has_tag<HasTag1, Tag2>::value, "HasTag1 hasn't got Tag2");
    static_assert(!cv::gapi::has_tag<HasTag1, Tag3>::value, "HasTag1 hasn't got Tag3");
}

TEST(GAPIUtil, GTags_HasTag2)
{
    using namespace cv::gapi::tag;
    static_assert(!cv::gapi::has_tag<HasTag2, Tag1>::value, "HasTag2 hasn't got Tag1");
    static_assert(cv::gapi::has_tag<HasTag2, Tag2>::value, "HasTag2 has got Tag2");
    static_assert(!cv::gapi::has_tag<HasTag2, Tag3>::value, "HasTag2 hasn't got Tag3");
}

TEST(GAPIUtil, GTags_HasTag23)
{
    using namespace cv::gapi::tag;
    static_assert(!cv::gapi::has_tag<HasTag23, Tag1>::value, "HasTag23 hasn't got Tag1");
    static_assert(cv::gapi::has_tag<HasTag23, Tag2>::value, "HasTag23 has got Tag2 & Tag3 both");
    static_assert(cv::gapi::has_tag<HasTag23, Tag3>::value, "HasTag23 has got Tag2 & Tag3 both");
}

TEST(GAPIUtil, GTags_HasTag123)
{
    using namespace cv::gapi::tag;
    static_assert(cv::gapi::has_tag<HasTag123, Tag1>::value, "HasTag123 has got Tag1");
    static_assert(cv::gapi::has_tag<HasTag123, Tag2>::value, "HasTag123 has got Tag2");
    static_assert(cv::gapi::has_tag<HasTag123, Tag3>::value, "HasTag123 has got Tag3");
}
} // namespace opencv_test
