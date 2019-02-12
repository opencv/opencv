// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "opencv2/gapi/cpu/gcpukernel.hpp"
#include "gapi_mock_kernels.hpp"

#include "opencv2/gapi/ocl/goclkernel.hpp"     // ocl::backend
#include "opencv2/gapi/cpu/gcpukernel.hpp"     // cpu::backend
#include "opencv2/gapi/fluid/gfluidkernel.hpp" // fluid::backend

namespace opencv_test
{

namespace
{
    namespace I
    {
        G_TYPED_KERNEL(GClone, <GMat(GMat)>, "org.opencv.test.clone")
        {
            static GMatDesc outMeta(GMatDesc in) { return in; }
        };
    }

    enum class KernelTags {
        CPU_CUSTOM_BGR2GRAY,
        CPU_CUSTOM_ADD,
        CPU_CUSTOM_CLONE,
        OCL_CUSTOM_BGR2GRAY,
        OCL_CUSTOM_ADD,
        OCL_CUSTOM_CLONE,
        FLUID_CUSTOM_BGR2GRAY,
        FLUID_CUSTOM_CLONE
    };

    struct KernelTagsHash
    {
        std::size_t operator() (KernelTags t) const { return std::hash<int>()(static_cast<int>(t)); }
    };

    struct GraphFixture
    {
        cv::GMat in[2], out;
        cv::Size sz_out;

        static std::unordered_map<std::string, std::unordered_map<KernelTags, bool, KernelTagsHash>> log;

        GraphFixture()
        {
            auto tmp = I::GClone::on(cv::gapi::add(in[0], in[1]));
            out = cv::gapi::imgproc::GBGR2Gray::on(tmp);
        }

        static void registerCallKernel(KernelTags kernel_tag) {
            std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
            log[test_name][kernel_tag] = true;
        }

        bool checkCallKernel(KernelTags kernel_tag) {
            std::string test_name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
            return log[test_name][kernel_tag];
        }
    };

    namespace ocl {
        GAPI_OCL_KERNEL(GAdd, cv::gapi::core::GAdd)
        {
            static void run(const cv::UMat&, const cv::UMat&, int, cv::UMat&)
            {
                GraphFixture::registerCallKernel(KernelTags::OCL_CUSTOM_ADD);
            }
        };

        GAPI_OCL_KERNEL(GClone, I::GClone)
        {
            static void run(const cv::UMat&, cv::UMat&)
            {
                GraphFixture::registerCallKernel(KernelTags::OCL_CUSTOM_CLONE);
            }
        };

        GAPI_OCL_KERNEL(BGR2Gray, cv::gapi::imgproc::GBGR2Gray)
        {
            static void run(const cv::UMat&, cv::UMat&)
            {
                GraphFixture::registerCallKernel(KernelTags::OCL_CUSTOM_BGR2GRAY);
            }
        };
    }

    namespace cpu
    {
        GAPI_OCV_KERNEL(GAdd, cv::gapi::core::GAdd)
        {
            static void run(const cv::Mat&, const cv::Mat&, int, cv::Mat&)
            {
                GraphFixture::registerCallKernel(KernelTags::CPU_CUSTOM_ADD);
            }
        };

        GAPI_OCV_KERNEL(GClone, I::GClone)
        {
            static void run(const cv::Mat&, cv::Mat&)
            {
                GraphFixture::registerCallKernel(KernelTags::CPU_CUSTOM_CLONE);
            }
        };

        GAPI_OCV_KERNEL(BGR2Gray, cv::gapi::imgproc::GBGR2Gray)
        {
            static void run(const cv::Mat&, cv::Mat&)
            {
                GraphFixture::registerCallKernel(KernelTags::CPU_CUSTOM_BGR2GRAY);
            }
        };
    }

    namespace fluid
    {
        GAPI_FLUID_KERNEL(GClone, I::GClone, false)
        {
            static const int Window = 1;
            static void run(const cv::gapi::fluid::View&, cv::gapi::fluid::Buffer&)
            {
                GraphFixture::registerCallKernel(KernelTags::FLUID_CUSTOM_CLONE);
            }
        };

        GAPI_FLUID_KERNEL(BGR2Gray, cv::gapi::imgproc::GBGR2Gray, false)
        {
            static const int Window = 1;
            static void run(const cv::gapi::fluid::View&, cv::gapi::fluid::Buffer&)
            {
                GraphFixture::registerCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY);
            }
        };
    }

    std::unordered_map<std::string, std::unordered_map<KernelTags, bool, KernelTagsHash>> GraphFixture::log;
    struct HeteroGraph: public ::testing::Test, public GraphFixture {};
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

    EXPECT_EQ(6u, u_pkg.size());
    EXPECT_TRUE(u_pkg.includes<J::Foo>());
    EXPECT_TRUE(u_pkg.includes<J::Bar>());
    EXPECT_TRUE(u_pkg.includes<J::Baz>());
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

    EXPECT_EQ(3u, u_pkg.size());
    EXPECT_TRUE (u_pkg.includes<J::Foo>());
    EXPECT_TRUE(u_pkg.includes<J::Bar>());
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
    auto out = I::GClone::on(cv::gapi::add(in[0], in[1]));
    const auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,cv::Size(32,32)});

    auto pkg = cv::gapi::kernels<cpu::GClone>();

    EXPECT_NO_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
                        compile({in_meta, in_meta}, cv::compile_args(pkg)));
}


TEST(KernelPackage, Unite_REPLACE_Same_Backend)
{
    namespace J = Jupiter;
    namespace S = Saturn;

    auto j_pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    auto s_pkg = cv::gapi::kernels<J::Bar, S::Baz>();
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg, cv::unite_policy::REPLACE);

    EXPECT_EQ(3u, u_pkg.size());
    EXPECT_TRUE(u_pkg.includes<J::Foo>());
    EXPECT_TRUE(u_pkg.includes<J::Bar>());
    EXPECT_TRUE(u_pkg.includes<S::Baz>());
}

TEST_F(HeteroGraph, Correct_Use_Custom_Kernel)
{
    // in0 -> cv::gapi::GAdd -> tmp -> cpu::GClone -> cv::gapi::BGR2Gray -> out
    //             ^
    //             |
    // in1 ------- `

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC3),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_CLONE));
}

TEST_F(HeteroGraph, Replace_Default)
{
    // in0 -> cpu::GAdd -> tmp -> cpu::GClone -> cv::gapi::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC3),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GAdd, cpu::GClone>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_ADD));
    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_CLONE));
}

TEST_F(HeteroGraph, User_Kernel_Not_Found)
{
    // in0 -> cv::gapi::GAdd -> tmp -> cpu::GClone -> cv::gapi::BGR2Gray -> out
    //             ^
    //             |
    // in1 --------`

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC3),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC3);

    EXPECT_ANY_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        compile(cv::descr_of(in_mat1), cv::descr_of(in_mat2)));
}

TEST_F(HeteroGraph, Replace_Default_Another_Backend)
{
     //in0 -> cv::gapi::GAdd -> tmp -> cpu::GClone -> fluid::BGR2Gray -> out
     //            ^
     //            |
     //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone, fluid::BGR2Gray>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Conflict_Customs)
{
    // in0 -> cv::gapi::GAdd -> tmp -> cpu::GClone -> (ocl::BGR2Gray/fluid::BGR2Gray) -> out
    //             ^
    //             |
    // in1 --------`

    cv::Mat in_mat1 = cv::Mat::eye(300, 300, CV_8UC3),
            in_mat2 = cv::Mat::eye(300, 300, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,{3, 3}});
    auto pkg = cv::gapi::kernels<cpu::GClone, fluid::BGR2Gray, ocl::BGR2Gray>();

    EXPECT_ANY_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out))
                         .compile({in_meta, in_meta}, cv::compile_args(pkg)));
}

TEST_F(HeteroGraph, Resolve_Custom_Conflict)
{
    // in0 -> gapi::GAdd -> tmp -> cpu::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(300, 300, CV_8UC3),
            in_mat2 = cv::Mat::eye(300, 300, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone, ocl::BGR2Gray, fluid::BGR2Gray>();
    cv::gapi::GLookupOrder lookup_order = { cv::gapi::fluid::backend() };

    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg, lookup_order));

    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Dont_Pass_Default_To_Lookup)
{
    // in0 -> cv::gapi::GAdd -> tmp -> cpu::GClone -> fluid::BGR2Gray -> out
    //             ^
    //             |
    // in1 --------`

    auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,{3, 3}});
    auto pkg = cv::gapi::kernels<cpu::GClone, ocl::BGR2Gray, fluid::BGR2Gray>();

    // Lookup order contains only ocl backend
    // CPU backend for GClone and gapi::GAdd pass implicitly
    cv::gapi::GLookupOrder lookup_order = { cv::gapi::fluid::backend() };

    EXPECT_NO_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out))
                         .compile({in_meta, in_meta}, cv::compile_args(pkg, lookup_order)));
}

TEST_F(HeteroGraph, Resolve_Default_Conflict)
{
    // in0 -> cpu::GAdd -> tmp -> cpu::GClone -> gapi::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(300, 300, CV_8UC3),
            in_mat2 = cv::Mat::eye(300, 300, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone, cpu::GAdd, ocl::GAdd>();
    cv::gapi::GLookupOrder lookup_order = { cv::gapi::cpu::backend() };

    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg, lookup_order));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_ADD));
}

TEST_F(HeteroGraph, Not_Resolve_Conflict)
{
    // in0 -> gapi::GAdd -> tmp -> cpu::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,{3, 3}});
    auto pkg = cv::gapi::kernels<cpu::GClone, fluid::BGR2Gray, ocl::BGR2Gray>();
    cv::gapi::GLookupOrder lookup_order = { cv::gapi::cpu::backend() };

    EXPECT_ANY_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out))
                         .compile({in_meta, in_meta}, cv::compile_args(pkg, lookup_order)));
}

TEST_F(HeteroGraph, DISABLED_Implicit_Pass_To_Lookup)
{
    // in0 -> ocl::GAdd -> tmp -> cpu::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(300, 300, CV_8UC3),
            in_mat2 = cv::Mat::eye(300, 300, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone, fluid::GClone, ocl::BGR2Gray, ocl::GAdd>();

    // ocl backend pass implicity
    cv::gapi::GLookupOrder lookup_order = { cv::gapi::fluid::backend() };

    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg, lookup_order));

    EXPECT_TRUE(checkCallKernel(KernelTags::OCL_CUSTOM_ADD));
    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_CLONE));
    EXPECT_TRUE(checkCallKernel(KernelTags::OCL_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Priority_Backend)
{
    // in0 -> cpu::GAdd -> tmp -> fluid::GClone -> cpu::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(300, 300, CV_8UC3),
            in_mat2 = cv::Mat::eye(300, 300, CV_8UC3),
            out_mat,
            ref_mat,
            tmp_mat;

    auto pkg = cv::gapi::kernels<cpu::GAdd,
                                 fluid::GClone,
                                 ocl::GClone,
                                 cpu::BGR2Gray,
                                 ocl::BGR2Gray>();

    // For GClone fluid backend more priority
    cv::gapi::GLookupOrder lookup_order = { cv::gapi::fluid::backend(), cv::gapi::cpu::backend() };

    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg, lookup_order));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_ADD));
    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_CLONE));
    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Unused_Backend)
{
    // in0 -> cv::gapi::GAdd -> tmp -> cpu::GClone -> cv::gapi::BGR2Gray -> out
    //             ^
    //             |
    // in1 --------`

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC1),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC1),
            out_mat,
            ref_mat,
            tmp_mat;

    auto in_meta = cv::GMetaArg(cv::GMatDesc{CV_8U,1,{3, 3}});
    auto pkg = cv::gapi::kernels<cpu::GClone>();
    auto lookup_order = { cv::gapi::fluid::backend() };

    EXPECT_NO_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out))
            .compile({in_meta, in_meta}, cv::compile_args(pkg, lookup_order)));
}

}// namespace opencv_test
