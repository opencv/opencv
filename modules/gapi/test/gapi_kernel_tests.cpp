// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"
#include "gapi_mock_kernels.hpp"

namespace opencv_test
{

namespace
{
    G_TYPED_KERNEL(GClone, <GMat(GMat)>, "org.opencv.test.clone")
    {
        static GMatDesc outMeta(GMatDesc in) { return in;  }

    };

    GAPI_OCV_KERNEL(GCloneImpl, GClone)
    {
        static void run(const cv::Mat& in, cv::Mat &out)
        {
            out = in.clone();
        }
    };
}

TEST(KernelPackage, Create)
{
    namespace J = Jupiter;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    EXPECT_EQ(3u, pkg.size());
}

TEST(KernelPackage, Includes)
{
    namespace J = Jupiter;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    EXPECT_TRUE (pkg.includes<J::Foo>());
    EXPECT_TRUE (pkg.includes<J::Bar>());
    EXPECT_TRUE (pkg.includes<J::Baz>());
    EXPECT_FALSE(pkg.includes<J::Qux>());
}

TEST(KernelPackage, IncludesAPI)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, S::Bar>();
    EXPECT_TRUE (pkg.includesAPI<I::Foo>());
    EXPECT_TRUE (pkg.includesAPI<I::Bar>());
    EXPECT_FALSE(pkg.includesAPI<I::Baz>());
    EXPECT_FALSE(pkg.includesAPI<I::Qux>());
}

TEST(KernelPackage, IncludesAPI_Overlapping)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, S::Foo, S::Bar>();
    EXPECT_TRUE (pkg.includesAPI<I::Foo>());
    EXPECT_TRUE (pkg.includesAPI<I::Bar>());
    EXPECT_FALSE(pkg.includesAPI<I::Baz>());
    EXPECT_FALSE(pkg.includesAPI<I::Qux>());
}

TEST(KernelPackage, Include_Add)
{
    namespace J = Jupiter;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    EXPECT_FALSE(pkg.includes<J::Qux>());

    pkg.include<J::Qux>();
    EXPECT_TRUE(pkg.includes<J::Qux>());
}

TEST(KernelPackage, Include_KEEP)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    EXPECT_FALSE(pkg.includes<S::Foo>());
    EXPECT_FALSE(pkg.includes<S::Bar>());

    pkg.include<S::Bar>(); // default (KEEP)
    EXPECT_TRUE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Bar>());

    pkg.include<S::Foo>(cv::unite_policy::KEEP); // explicit (KEEP)
    EXPECT_TRUE(pkg.includes<J::Foo>());
    EXPECT_TRUE(pkg.includes<S::Foo>());
}

TEST(KernelPackage, Include_REPLACE)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    EXPECT_FALSE(pkg.includes<S::Bar>());

    pkg.include<S::Bar>(cv::unite_policy::REPLACE);
    EXPECT_FALSE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Bar>());
}

TEST(KernelPackage, RemoveBackend)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, S::Foo>();
    EXPECT_TRUE(pkg.includes<J::Foo>());
    EXPECT_TRUE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Foo>());

    pkg.remove(J::backend());
    EXPECT_FALSE(pkg.includes<J::Foo>());
    EXPECT_FALSE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Foo>());
};

TEST(KernelPackage, RemoveAPI)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, S::Foo, S::Bar>();
    EXPECT_TRUE(pkg.includes<J::Foo>());
    EXPECT_TRUE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Foo>());

    pkg.remove<I::Foo>();
    EXPECT_TRUE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Bar>());
    EXPECT_FALSE(pkg.includes<J::Foo>());
    EXPECT_FALSE(pkg.includes<S::Foo>());
};

TEST(KernelPackage, CreateHetero)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz, S::Qux>();
    EXPECT_EQ(4u, pkg.size());
}

TEST(KernelPackage, IncludesHetero)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz, S::Qux>();
    EXPECT_TRUE (pkg.includes<J::Foo>());
    EXPECT_TRUE (pkg.includes<J::Bar>());
    EXPECT_TRUE (pkg.includes<J::Baz>());
    EXPECT_FALSE(pkg.includes<J::Qux>());
    EXPECT_TRUE (pkg.includes<S::Qux>());
}

TEST(KernelPackage, IncludeHetero)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    EXPECT_FALSE(pkg.includes<J::Qux>());
    EXPECT_FALSE(pkg.includes<S::Qux>());

    pkg.include<S::Qux>();
    EXPECT_FALSE(pkg.includes<J::Qux>());
    EXPECT_TRUE (pkg.includes<S::Qux>());
}

TEST(KernelPackage, Combine_REPLACE_Full)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto j_pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    auto s_pkg = cv::gapi::kernels<S::Foo, S::Bar, S::Baz>();
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg, cv::unite_policy::REPLACE);

    EXPECT_EQ(3u, u_pkg.size());
    EXPECT_FALSE(u_pkg.includes<J::Foo>());
    EXPECT_FALSE(u_pkg.includes<J::Bar>());
    EXPECT_FALSE(u_pkg.includes<J::Baz>());
    EXPECT_TRUE (u_pkg.includes<S::Foo>());
    EXPECT_TRUE (u_pkg.includes<S::Bar>());
    EXPECT_TRUE (u_pkg.includes<S::Baz>());
}

TEST(KernelPackage, Combine_REPLACE_Partial)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto j_pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    auto s_pkg = cv::gapi::kernels<S::Bar>();
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg, cv::unite_policy::REPLACE);

    EXPECT_EQ(2u, u_pkg.size());
    EXPECT_TRUE (u_pkg.includes<J::Foo>());
    EXPECT_FALSE(u_pkg.includes<J::Bar>());
    EXPECT_TRUE (u_pkg.includes<S::Bar>());
}

TEST(KernelPackage, Combine_REPLACE_Append)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto j_pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    auto s_pkg = cv::gapi::kernels<S::Qux>();
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg, cv::unite_policy::REPLACE);

    EXPECT_EQ(3u, u_pkg.size());
    EXPECT_TRUE(u_pkg.includes<J::Foo>());
    EXPECT_TRUE(u_pkg.includes<J::Bar>());
    EXPECT_TRUE(u_pkg.includes<S::Qux>());
}

TEST(KernelPackage, Combine_KEEP_AllDups)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto j_pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    auto s_pkg = cv::gapi::kernels<S::Foo, S::Bar, S::Baz>();
    auto u_pkg = cv::gapi::combine(j_pkg ,s_pkg, cv::unite_policy::KEEP);

    EXPECT_EQ(6u, u_pkg.size());
    EXPECT_TRUE(u_pkg.includes<J::Foo>());
    EXPECT_TRUE(u_pkg.includes<J::Bar>());
    EXPECT_TRUE(u_pkg.includes<J::Baz>());
    EXPECT_TRUE(u_pkg.includes<S::Foo>());
    EXPECT_TRUE(u_pkg.includes<S::Bar>());
    EXPECT_TRUE(u_pkg.includes<S::Baz>());
}

TEST(KernelPackage, Combine_KEEP_Append_NoDups)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto j_pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    auto s_pkg = cv::gapi::kernels<S::Qux>();
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg, cv::unite_policy::KEEP);

    EXPECT_EQ(3u, u_pkg.size());
    EXPECT_TRUE(u_pkg.includes<J::Foo>());
    EXPECT_TRUE(u_pkg.includes<J::Bar>());
    EXPECT_TRUE(u_pkg.includes<S::Qux>());
}

TEST(KernelPackage, TestWithEmptyLHS)
{
    namespace J = Jupiter;
    auto lhs = cv::gapi::kernels<>();
    auto rhs = cv::gapi::kernels<J::Foo>();
    auto pkg = cv::gapi::combine(lhs, rhs, cv::unite_policy::KEEP);

    EXPECT_EQ(1u, pkg.size());
    EXPECT_TRUE(pkg.includes<J::Foo>());
}

TEST(KernelPackage, TestWithEmptyRHS)
{
    namespace J = Jupiter;
    auto lhs = cv::gapi::kernels<J::Foo>();
    auto rhs = cv::gapi::kernels<>();
    auto pkg = cv::gapi::combine(lhs, rhs, cv::unite_policy::KEEP);

    EXPECT_EQ(1u, pkg.size());
    EXPECT_TRUE(pkg.includes<J::Foo>());
}

TEST(KernelPackage, Can_Use_Custom_Kernel)
{
    cv::GMat in[2];
    auto out = GClone::on(cv::gapi::add(in[0], in[1]));
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::Size(32,32)});

    auto pkg = cv::gapi::kernels<GCloneImpl>();

    EXPECT_NO_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
                        compile({in_meta, in_meta}, cv::compile_args(pkg)));
}

} // namespace opencv_test
