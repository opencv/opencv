// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "gapi_mock_kernels.hpp"

#include <opencv2/gapi/cpu/gcpukernel.hpp>     // cpu::backend
#include <opencv2/gapi/fluid/gfluidkernel.hpp> // fluid::backend

namespace opencv_test
{

namespace
{
    namespace I
    {
        G_TYPED_KERNEL(GClone, <GMat(GMat)>, "org.opencv.test.clone")
        {
            static GMatDesc outMeta(GMatDesc in) { return in;  }
        };
    }

    enum class KernelTags
    {
        CPU_CUSTOM_BGR2GRAY,
        CPU_CUSTOM_CLONE,
        CPU_CUSTOM_ADD,
        FLUID_CUSTOM_BGR2GRAY,
        FLUID_CUSTOM_CLONE,
        FLUID_CUSTOM_ADD
    };

    class HeteroGraph: public ::testing::Test
    {
    public:
        HeteroGraph()
        {
            auto tmp = I::GClone::on(cv::gapi::add(in[0], in[1]));
            out = cv::gapi::imgproc::GBGR2Gray::on(tmp);
        }

        static void registerCallKernel(KernelTags kernel_tag) {
            kernel_calls.insert(kernel_tag);
        }

        bool checkCallKernel(KernelTags kernel_tag) {
            return ade::util::contains(kernel_calls, kernel_tag);
        }

    protected:
        void SetUp() override
        {
            if (!kernel_calls.empty())
                cv::util::throw_error(std::logic_error("Kernel call log has not been cleared!!!"));
        }

        void TearDown() override
        {
            kernel_calls.clear();
        }

    protected:
        cv::GMat in[2], out;
        static std::set<KernelTags> kernel_calls;
    };

    namespace cpu
    {
        GAPI_OCV_KERNEL(GClone, I::GClone)
        {
            static void run(const cv::Mat&, cv::Mat)
            {
                HeteroGraph::registerCallKernel(KernelTags::CPU_CUSTOM_CLONE);
            }
        };

        GAPI_OCV_KERNEL(BGR2Gray, cv::gapi::imgproc::GBGR2Gray)
        {
            static void run(const cv::Mat&, cv::Mat&)
            {
                HeteroGraph::registerCallKernel(KernelTags::CPU_CUSTOM_BGR2GRAY);
            }
        };

        GAPI_OCV_KERNEL(GAdd, cv::gapi::core::GAdd)
        {
            static void run(const cv::Mat&, const cv::Mat&, int, cv::Mat&)
            {
                HeteroGraph::registerCallKernel(KernelTags::CPU_CUSTOM_ADD);
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
                HeteroGraph::registerCallKernel(KernelTags::FLUID_CUSTOM_CLONE);
            }
        };

        GAPI_FLUID_KERNEL(BGR2Gray, cv::gapi::imgproc::GBGR2Gray, false)
        {
            static const int Window = 1;
            static void run(const cv::gapi::fluid::View&, cv::gapi::fluid::Buffer&)
            {
                HeteroGraph::registerCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY);
            }
        };

        GAPI_FLUID_KERNEL(GAdd, cv::gapi::core::GAdd, false)
        {
            static const int Window = 1;
            static void run(const cv::gapi::fluid::View&, const cv::gapi::fluid::View&,
                            int, cv::gapi::fluid::Buffer&)
            {
                HeteroGraph::registerCallKernel(KernelTags::FLUID_CUSTOM_ADD);
            }
        };
    }

    std::set<KernelTags> HeteroGraph::kernel_calls;
} // anonymous namespace

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

TEST(KernelPackage, Include_Add)
{
    namespace J = Jupiter;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, J::Baz>();
    EXPECT_FALSE(pkg.includes<J::Qux>());

    pkg.include<J::Qux>();
    EXPECT_TRUE(pkg.includes<J::Qux>());
}

TEST(KernelPackage, Include_REPLACE)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    EXPECT_FALSE(pkg.includes<S::Bar>());

    pkg.include<S::Bar>();
    EXPECT_FALSE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Bar>());
}

TEST(KernelPackage, RemoveBackend)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar, S::Baz>();
    EXPECT_TRUE(pkg.includes<J::Foo>());
    EXPECT_TRUE(pkg.includes<J::Bar>());

    pkg.remove(J::backend());
    EXPECT_FALSE(pkg.includes<J::Foo>());
    EXPECT_FALSE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Baz>());
};

TEST(KernelPackage, RemoveAPI)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto pkg = cv::gapi::kernels<J::Foo, J::Bar>();
    EXPECT_TRUE(pkg.includes<J::Foo>());
    EXPECT_TRUE(pkg.includes<J::Bar>());

    pkg.remove<I::Foo>();
    EXPECT_TRUE(pkg.includes<J::Bar>());
    EXPECT_FALSE(pkg.includes<J::Foo>());
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
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg);

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
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg);

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
    auto u_pkg = cv::gapi::combine(j_pkg, s_pkg);

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
    auto pkg = cv::gapi::combine(lhs, rhs);

    EXPECT_EQ(1u, pkg.size());
    EXPECT_TRUE(pkg.includes<J::Foo>());
}

TEST(KernelPackage, TestWithEmptyRHS)
{
    namespace J = Jupiter;
    auto lhs = cv::gapi::kernels<J::Foo>();
    auto rhs = cv::gapi::kernels<>();
    auto pkg = cv::gapi::combine(lhs, rhs);

    EXPECT_EQ(1u, pkg.size());
    EXPECT_TRUE(pkg.includes<J::Foo>());
}

TEST(KernelPackage, Return_Unique_Backends)
{
    auto pkg = cv::gapi::kernels<cpu::GClone, fluid::BGR2Gray, fluid::GAdd>();
    EXPECT_EQ(2u, pkg.backends().size());
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

TEST(KernelPackage, CombineMultiple)
{
    namespace J = Jupiter;
    namespace S = Saturn;
    auto a = cv::gapi::kernels<J::Foo>();
    auto b = cv::gapi::kernels<J::Bar>();
    auto c = cv::gapi::kernels<S::Qux>();
    auto pkg = cv::gapi::combine(a, b, c);

    EXPECT_EQ(3u, pkg.size());
    EXPECT_TRUE(pkg.includes<J::Foo>());
    EXPECT_TRUE(pkg.includes<J::Bar>());
    EXPECT_TRUE(pkg.includes<S::Qux>());
}

TEST_F(HeteroGraph, Call_Custom_Kernel_Default_Backend)
{
    // in0 -> GCPUAdd -> tmp -> cpu::GClone -> GCPUBGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC3),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC3),
            out_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_CLONE));
}

TEST_F(HeteroGraph, Call_Custom_Kernel_Not_Default_Backend)
{
    // in0 -> GCPUAdd -> tmp -> fluid::GClone -> GCPUBGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC3),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC3),
            out_mat;

    auto pkg = cv::gapi::kernels<fluid::GClone>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_CLONE));
}

TEST_F(HeteroGraph, Replace_Default_To_Same_Backend)
{
    // in0 -> GCPUAdd -> tmp -> cpu::GClone -> cpu::BGR2Gray -> out
    //            ^
    //            |
    // in1 -------`

    cv::Mat in_mat1 = cv::Mat::eye(3, 3, CV_8UC3),
            in_mat2 = cv::Mat::eye(3, 3, CV_8UC3),
            out_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone, cpu::BGR2Gray>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Replace_Default_To_Another_Backend)
{
    //in0 -> GCPUAdd -> tmp -> cpu::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
            out_mat;

    auto pkg = cv::gapi::kernels<cpu::GClone, fluid::BGR2Gray>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(pkg));

    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Use_Only_Same_Backend)
{
    //in0 -> cpu::GAdd -> tmp -> cpu::GClone -> cpu::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
        out_mat;

    auto pkg = cv::gapi::kernels<cpu::GAdd, cpu::GClone, cpu::BGR2Gray>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(cv::gapi::use_only{pkg}));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_ADD));
    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_CLONE));
    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Use_Only_Another_Backend)
{
    //in0 -> fluid::GAdd -> tmp -> fluid::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
        out_mat;

    auto pkg = cv::gapi::kernels<fluid::GAdd, fluid::GClone, fluid::BGR2Gray>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(cv::gapi::use_only{pkg}));

    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_ADD));
    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_CLONE));
    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Use_Only_Hetero_Backend)
{
    //in0 -> cpu::GAdd -> tmp -> fluid::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
        out_mat;

    auto pkg = cv::gapi::kernels<cpu::GAdd, fluid::GClone, fluid::BGR2Gray>();
    cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(cv::gapi::use_only{pkg}));

    EXPECT_TRUE(checkCallKernel(KernelTags::CPU_CUSTOM_ADD));
    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_CLONE));
    EXPECT_TRUE(checkCallKernel(KernelTags::FLUID_CUSTOM_BGR2GRAY));
}

TEST_F(HeteroGraph, Use_Only_Not_Found_Default)
{
    //in0 -> GCPUAdd -> tmp -> fluid::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
        out_mat;

    auto pkg = cv::gapi::kernels<fluid::GClone, fluid::BGR2Gray>();
    EXPECT_ANY_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(cv::gapi::use_only{pkg})));
}

TEST_F(HeteroGraph, Use_Only_Not_Found_Custom)
{
    //in0 -> cpu::GAdd -> tmp -> fluid::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
        out_mat;

    auto pkg = cv::gapi::kernels<cpu::GAdd, fluid::BGR2Gray>();
    EXPECT_ANY_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat), cv::compile_args(cv::gapi::use_only{pkg})));
}

TEST_F(HeteroGraph, Use_Only_Other_Package_Ignored)
{
    //in0 -> cpu::GAdd -> tmp -> fluid::GClone -> fluid::BGR2Gray -> out
    //            ^
    //            |
    //in1 --------`

    cv::Mat in_mat1(300, 300, CV_8UC3),
            in_mat2(300, 300, CV_8UC3),
        out_mat;

    auto pkg = cv::gapi::kernels<cpu::GAdd, fluid::BGR2Gray>();
    auto clone_pkg = cv::gapi::kernels<cpu::GClone>();

    EXPECT_ANY_THROW(cv::GComputation(cv::GIn(in[0], in[1]), cv::GOut(out)).
        apply(cv::gin(in_mat1, in_mat2), cv::gout(out_mat),
              cv::compile_args(clone_pkg, cv::gapi::use_only{pkg})));
}

} // namespace opencv_test
