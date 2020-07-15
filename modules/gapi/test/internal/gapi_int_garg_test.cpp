// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "../test_precomp.hpp"

namespace opencv_test {
// Tests on T/Spec/Kind matching ///////////////////////////////////////////////
// {{

template<class T, cv::detail::ArgKind Exp>
struct Expected
{
    using type = T;
    static const constexpr cv::detail::ArgKind kind = Exp;
};

template<typename T>
struct GArgKind: public ::testing::Test
{
    using Type = typename T::type;
    const cv::detail::ArgKind Kind = T::kind;
};

// The reason here is to _manually_ list types and their kinds
// (and NOT reuse cv::detail::ArgKind::Traits<>, since it is a subject of testing)
using GArg_Test_Types = ::testing::Types
   <
  // G-API types
     Expected<cv::GMat,                 cv::detail::ArgKind::GMAT>
   , Expected<cv::GMatP,                cv::detail::ArgKind::GMATP>
   , Expected<cv::GFrame,               cv::detail::ArgKind::GFRAME>
   , Expected<cv::GScalar,              cv::detail::ArgKind::GSCALAR>
   , Expected<cv::GArray<int>,          cv::detail::ArgKind::GARRAY>
   , Expected<cv::GArray<float>,        cv::detail::ArgKind::GARRAY>
   , Expected<cv::GArray<cv::Point>,    cv::detail::ArgKind::GARRAY>
   , Expected<cv::GArray<cv::Rect>,     cv::detail::ArgKind::GARRAY>
   , Expected<cv::GOpaque<int>,         cv::detail::ArgKind::GOPAQUE>
   , Expected<cv::GOpaque<float>,       cv::detail::ArgKind::GOPAQUE>
   , Expected<cv::GOpaque<cv::Point>,   cv::detail::ArgKind::GOPAQUE>
   , Expected<cv::GOpaque<cv::Rect>,    cv::detail::ArgKind::GOPAQUE>

 // Built-in types
   , Expected<int,                      cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<float,                    cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<int*,                     cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<cv::Point,                cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<std::string,              cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<cv::Mat,                  cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<std::vector<int>,         cv::detail::ArgKind::OPAQUE_VAL>
   , Expected<std::vector<cv::Point>,   cv::detail::ArgKind::OPAQUE_VAL>
   >;

TYPED_TEST_CASE(GArgKind, GArg_Test_Types);

TYPED_TEST(GArgKind, LocalVar)
{
    typename TestFixture::Type val{};
    cv::GArg arg(val);
    EXPECT_EQ(TestFixture::Kind, arg.kind);
}

TYPED_TEST(GArgKind, ConstLocalVar)
{
    const typename TestFixture::Type val{};
    cv::GArg arg(val);
    EXPECT_EQ(TestFixture::Kind, arg.kind);
}

TYPED_TEST(GArgKind, RValue)
{
    cv::GArg arg = cv::GArg(typename TestFixture::Type());
    EXPECT_EQ(TestFixture::Kind, arg.kind);
}

// Repeat the same for Spec

template<class T, cv::detail::ArgSpec Exp>
struct ExpectedS
{
    using type = T;
    static const constexpr cv::detail::ArgSpec spec = Exp;
};

template<typename T>
struct ArgSpec: public ::testing::Test
{
    using Type = typename T::type;
    const cv::detail::ArgSpec Spec = T::spec;
};

using Arg_Spec_Types = ::testing::Types
   <
  // G-API types
     ExpectedS<cv::GMat,                 cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GMatP,                cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GFrame,               cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GScalar,              cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GArray<int>,          cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GArray<float>,        cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GArray<cv::Point>,    cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GArray<cv::Rect>,     cv::detail::ArgSpec::RECT>
   , ExpectedS<cv::GArray<cv::GMat>,     cv::detail::ArgSpec::GMAT>
   , ExpectedS<cv::GOpaque<int>,         cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GOpaque<float>,       cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GOpaque<cv::Point>,   cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::GOpaque<cv::Rect>,    cv::detail::ArgSpec::RECT>
// FIXME: causes internal conflicts in GOpaque/descr_of
// , ExpectedS<cv::GOpaque<cv::Mat>,     cv::detail::ArgSpec::GMAT>

 // Built-in types
   , ExpectedS<int,                      cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<float,                    cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<int*,                     cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::Point,                cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<std::string,              cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<cv::Mat,                  cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<std::vector<int>,         cv::detail::ArgSpec::OPAQUE_SPEC>
   , ExpectedS<std::vector<cv::Point>,   cv::detail::ArgSpec::OPAQUE_SPEC>
   >;

TYPED_TEST_CASE(ArgSpec, Arg_Spec_Types);

TYPED_TEST(ArgSpec, Basic)
{
    const auto this_spec = cv::detail::GTypeTraits<typename TestFixture::Type>::spec;
    EXPECT_EQ(TestFixture::Spec, this_spec);
}

// }}
////////////////////////////////////////////////////////////////////////////////

TEST(GArg, HasWrap)
{
    static_assert(!cv::detail::has_custom_wrap<cv::GMat>::value,
                  "GMat has no custom marshalling logic");
    static_assert(!cv::detail::has_custom_wrap<cv::GScalar>::value,
                  "GScalar has no custom marshalling logic");

    static_assert(cv::detail::has_custom_wrap<cv::GArray<int> >::value,
                  "GArray<int> has custom marshalling logic");
    static_assert(cv::detail::has_custom_wrap<cv::GArray<std::string> >::value,
                  "GArray<int> has custom marshalling logic");

    static_assert(cv::detail::has_custom_wrap<cv::GOpaque<int> >::value,
                  "GOpaque<int> has custom marshalling logic");
    static_assert(cv::detail::has_custom_wrap<cv::GOpaque<std::string> >::value,
                  "GOpaque<int> has custom marshalling logic");
}

TEST(GArg, GArrayU)
{
    // Placing a GArray<T> into GArg automatically strips it to GArrayU
    cv::GArg arg1 = cv::GArg(cv::GArray<int>());
    EXPECT_NO_THROW(arg1.get<cv::detail::GArrayU>());

    cv::GArg arg2 = cv::GArg(cv::GArray<cv::Point>());
    EXPECT_NO_THROW(arg2.get<cv::detail::GArrayU>());
}

TEST(GArg, GOpaqueU)
{
    // Placing a GOpaque<T> into GArg automatically strips it to GOpaqueU
    cv::GArg arg1 = cv::GArg(cv::GOpaque<int>());
    EXPECT_NO_THROW(arg1.get<cv::detail::GOpaqueU>());

    cv::GArg arg2 = cv::GArg(cv::GOpaque<cv::Point>());
    EXPECT_NO_THROW(arg2.get<cv::detail::GOpaqueU>());
}


} // namespace opencv_test
