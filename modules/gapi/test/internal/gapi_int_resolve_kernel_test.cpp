// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "gapi_mock_kernels.hpp"

namespace opencv_test
{

TEST(Lookup, CreateOrder)
{
    const auto order = cv::gapi::lookup_order({Jupiter::backend(),
                                               Saturn::backend()});
    EXPECT_EQ(2u, order.size());
    EXPECT_EQ(Jupiter::backend(), order[0]);
    EXPECT_EQ(Saturn ::backend(), order[1]);
}

TEST(Lookup, NoOrder)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz,
                                       S::Foo, S::Bar, S::Baz>();

    EXPECT_NO_THROW (pkg.lookup<I::Foo>());
    EXPECT_NO_THROW (pkg.lookup<I::Bar>());
    EXPECT_NO_THROW (pkg.lookup<I::Baz>());
    EXPECT_ANY_THROW(pkg.lookup<I::Qux>());
}

TEST(Lookup, Only_Jupiter)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz,
                                       S::Foo, S::Bar, S::Baz>();

    auto order = cv::gapi::lookup_order({J::backend()});

    EXPECT_EQ(J::backend(), pkg.lookup<I::Foo>(order));
    EXPECT_EQ(J::backend(), pkg.lookup<I::Bar>(order));
    EXPECT_EQ(J::backend(), pkg.lookup<I::Baz>(order));
    EXPECT_ANY_THROW(pkg.lookup<I::Qux>(order));
}

TEST(Lookup, Only_Saturn)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz,
                                       S::Foo, S::Bar, S::Baz>();

    auto order = cv::gapi::lookup_order({S::backend()});

    EXPECT_EQ(S::backend(), pkg.lookup<I::Foo>(order));
    EXPECT_EQ(S::backend(), pkg.lookup<I::Bar>(order));
    EXPECT_EQ(S::backend(), pkg.lookup<I::Baz>(order));
    EXPECT_ANY_THROW(pkg.lookup<I::Qux>(order));
}

TEST(Lookup, With_Order)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz,
                                       S::Foo, S::Bar, S::Baz>();

    auto prefer_j = cv::gapi::lookup_order({J::backend(), S::backend()});
    EXPECT_EQ(J::backend(), pkg.lookup<I::Foo>(prefer_j));
    EXPECT_EQ(J::backend(), pkg.lookup<I::Bar>(prefer_j));
    EXPECT_EQ(J::backend(), pkg.lookup<I::Baz>(prefer_j));
    EXPECT_ANY_THROW(pkg.lookup<I::Qux>(prefer_j));

    auto prefer_s = cv::gapi::lookup_order({S::backend(), J::backend()});
    EXPECT_EQ(S::backend(), pkg.lookup<I::Foo>(prefer_s));
    EXPECT_EQ(S::backend(), pkg.lookup<I::Bar>(prefer_s));
    EXPECT_EQ(S::backend(), pkg.lookup<I::Baz>(prefer_s));
    EXPECT_ANY_THROW(pkg.lookup<I::Qux>(prefer_s));
}

TEST(Lookup, NoOverlap)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar, S::Baz, S::Qux>();
    EXPECT_EQ(J::backend(), pkg.lookup<I::Foo>());
    EXPECT_EQ(J::backend(), pkg.lookup<I::Bar>());
    EXPECT_EQ(S::backend(), pkg.lookup<I::Baz>());
    EXPECT_EQ(S::backend(), pkg.lookup<I::Qux>());
}

TEST(Lookup, ExtraBackend)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    const auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();

    // Even if pkg doesn't contain S kernels while S is preferable,
    // it should work.
    const auto prefer_sj = cv::gapi::lookup_order({S::backend(), J::backend()});
    EXPECT_EQ(J::backend(), pkg.lookup<I::Foo>(prefer_sj));
    EXPECT_EQ(J::backend(), pkg.lookup<I::Bar>(prefer_sj));
    EXPECT_EQ(J::backend(), pkg.lookup<I::Baz>(prefer_sj));

    // If search scope is limited to S only, neither J nor S  kernels
    // shouldn't be found
    const auto only_s = cv::gapi::lookup_order({S::backend()});
    EXPECT_ANY_THROW(pkg.lookup<I::Foo>(only_s));
    EXPECT_ANY_THROW(pkg.lookup<I::Bar>(only_s));
    EXPECT_ANY_THROW(pkg.lookup<I::Baz>(only_s));
}

} // namespace opencv_test
