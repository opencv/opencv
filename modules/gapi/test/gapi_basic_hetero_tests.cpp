// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "gapi_mock_kernels.hpp"

#include <opencv2/gapi/fluid/gfluidkernel.hpp>

namespace opencv_test
{

namespace
{
    GAPI_OCV_KERNEL(OCVFoo, I::Foo)
    {
        static void run(const cv::Mat &in, cv::Mat &out)
        {
            out = in + 2;
        }
    };

    GAPI_OCV_KERNEL(OCVBar, I::Bar)
    {
        static void run(const cv::Mat &a, const cv::Mat &b, cv::Mat &out)
        {
            out = 4*(a + b);
        }
    };

    void FluidFooRow(const uint8_t* in, uint8_t* out, int length)
    {
        for (int i = 0; i < length; i++)
        {
            out[i] = in[i] + 3;
        }
    }

    void FluidBarRow(const uint8_t* in1, const uint8_t* in2, uint8_t* out, int length)
    {
        for (int i = 0; i < length; i++)
        {
            out[i] = 3*(in1[i] + in2[i]);
        }
    }

    GAPI_FLUID_KERNEL(FFoo, I::Foo, false)
    {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View   &in,
                              cv::gapi::fluid::Buffer &out)
        {
            FluidFooRow(in.InLineB(0), out.OutLineB(), in.length());
        }
    };

    GAPI_FLUID_KERNEL(FBar, I::Bar, false)
    {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View   &in1,
                        const cv::gapi::fluid::View   &in2,
                              cv::gapi::fluid::Buffer &out)
        {
            FluidBarRow(in1.InLineB(0), in2.InLineB(0), out.OutLineB(), in1.length());
        }
    };

    G_TYPED_KERNEL(FluidFooI, <cv::GMat(cv::GMat)>, "test.kernels.fluid_foo")
    {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in) { return in; }
    };

    G_TYPED_KERNEL(FluidBarI, <cv::GMat(cv::GMat,cv::GMat)>, "test.kernels.fluid_bar")
    {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) { return in; }
    };

    GAPI_FLUID_KERNEL(FluidFoo, FluidFooI, false)
    {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View   &in,
                              cv::gapi::fluid::Buffer &out)
        {
            FluidFooRow(in.InLineB(0), out.OutLineB(), in.length());
        }
    };

    GAPI_FLUID_KERNEL(FluidBar, FluidBarI, false)
    {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View   &in1,
                        const cv::gapi::fluid::View   &in2,
                              cv::gapi::fluid::Buffer &out)
        {
            FluidBarRow(in1.InLineB(0), in2.InLineB(0), out.OutLineB(), in1.length());
        }
    };

    GAPI_FLUID_KERNEL(FluidFoo2lpi, FluidFooI, false)
    {
        static const int Window = 1;
        static const int LPI    = 2;

        static void run(const cv::gapi::fluid::View   &in,
                              cv::gapi::fluid::Buffer &out)
        {
            for (int l = 0; l < out.lpi(); l++)
            {
                FluidFooRow(in.InLineB(l), out.OutLineB(l), in.length());
            }
        }
    };

    cv::Mat ocvFoo(const cv::Mat &in)
    {
        cv::Mat out;
        OCVFoo::run(in, out);
        return out;
    }
    cv::Mat ocvBar(const cv::Mat &in1, const cv::Mat &in2)
    {
        cv::Mat out;
        OCVBar::run(in1, in2, out);
        return out;
    }
    cv::Mat fluidFoo(const cv::Mat &in)
    {
        cv::Mat out(in.rows, in.cols, in.type());
        for (int y = 0; y < in.rows; y++)
        {
            FluidFooRow(in.ptr(y), out.ptr(y), in.cols);
        }
        return out;
    }
    cv::Mat fluidBar(const cv::Mat &in1, const cv::Mat &in2)
    {
        cv::Mat out(in1.rows, in1.cols, in1.type());
        for (int y = 0; y < in1.rows; y++)
        {
            FluidBarRow(in1.ptr(y), in2.ptr(y), out.ptr(y), in1.cols);
        }
        return out;
    }
} // anonymous namespace

struct GAPIHeteroTest: public ::testing::Test
{
    cv::GComputation m_comp;
    cv::gapi::GKernelPackage m_ocv_kernels;
    cv::gapi::GKernelPackage m_fluid_kernels;
    cv::gapi::GKernelPackage m_hetero_kernels;

    cv::Mat m_in_mat;
    cv::Mat m_out_mat;

    GAPIHeteroTest();
};

GAPIHeteroTest::GAPIHeteroTest()
    : m_comp([](){
            cv::GMat in;
            cv::GMat out = I::Bar::on(I::Foo::on(in),
                                      I::Foo::on(in));
            return cv::GComputation(in, out);
        })
    , m_ocv_kernels(cv::gapi::kernels<OCVFoo, OCVBar>())
    , m_fluid_kernels(cv::gapi::kernels<FFoo, FBar>())
    , m_hetero_kernels(cv::gapi::kernels<OCVFoo, FBar>())
    , m_in_mat(cv::Mat::eye(cv::Size(64, 64), CV_8UC1))
{
}

TEST_F(GAPIHeteroTest, TestOCV)
{
    EXPECT_TRUE(cv::gapi::cpu::backend() == m_ocv_kernels.lookup<I::Foo>());
    EXPECT_TRUE(cv::gapi::cpu::backend() == m_ocv_kernels.lookup<I::Bar>());

    cv::Mat ref = ocvBar(ocvFoo(m_in_mat), ocvFoo(m_in_mat));
    EXPECT_NO_THROW(m_comp.apply(m_in_mat, m_out_mat, cv::compile_args(m_ocv_kernels)));
    EXPECT_EQ(0, cv::countNonZero(ref != m_out_mat));
}

TEST_F(GAPIHeteroTest, TestFluid)
{
    EXPECT_TRUE(cv::gapi::fluid::backend() == m_fluid_kernels.lookup<I::Foo>());
    EXPECT_TRUE(cv::gapi::fluid::backend() == m_fluid_kernels.lookup<I::Bar>());

    cv::Mat ref = fluidBar(fluidFoo(m_in_mat), fluidFoo(m_in_mat));
    EXPECT_NO_THROW(m_comp.apply(m_in_mat, m_out_mat, cv::compile_args(m_fluid_kernels)));
    EXPECT_EQ(0, cv::countNonZero(ref != m_out_mat));
}

TEST_F(GAPIHeteroTest, TestBoth)
{
    EXPECT_TRUE(cv::gapi::cpu::backend()   == m_hetero_kernels.lookup<I::Foo>());
    EXPECT_TRUE(cv::gapi::fluid::backend() == m_hetero_kernels.lookup<I::Bar>());

    cv::Mat ref = fluidBar(ocvFoo(m_in_mat), ocvFoo(m_in_mat));
    EXPECT_NO_THROW(m_comp.apply(m_in_mat, m_out_mat, cv::compile_args(m_hetero_kernels)));
    EXPECT_EQ(0, cv::countNonZero(ref != m_out_mat));
}

struct GAPIBigHeteroTest : public ::testing::TestWithParam<std::array<int, 9>>
{
    cv::GComputation m_comp;
    cv::gapi::GKernelPackage m_kernels;

    cv::Mat m_in_mat;
    cv::Mat m_out_mat1;
    cv::Mat m_out_mat2;

    cv::Mat m_ref_mat1;
    cv::Mat m_ref_mat2;

    GAPIBigHeteroTest();
};

//                                    Foo7
//                .-> Foo2 -> Foo3 -<
//   Foo0 -> Foo1                     Bar -> Foo6
//                `-> Foo4 -> Foo5 -`

GAPIBigHeteroTest::GAPIBigHeteroTest()
    : m_comp([&](){
        auto flags = GetParam();
        std::array<std::function<cv::GMat(cv::GMat)>, 8> foos;

        for (int i = 0; i < 8; i++)
        {
            foos[i] = flags[i] ? &I::Foo::on : &FluidFooI::on;
        }
        auto bar = flags[8] ? &I::Bar::on : &FluidBarI::on;

        cv::GMat in;
        auto foo1Out = foos[1](foos[0](in));
        auto foo3Out = foos[3](foos[2](foo1Out));
        auto foo6Out = foos[6](bar(foo3Out,
                               foos[5](foos[4](foo1Out))));
        auto foo7Out = foos[7](foo3Out);

        return cv::GComputation(GIn(in), GOut(foo6Out, foo7Out));
    })
    , m_kernels(cv::gapi::kernels<OCVFoo, OCVBar, FluidFoo, FluidBar>())
    , m_in_mat(cv::Mat::eye(cv::Size(64, 64), CV_8UC1))
{
    auto flags = GetParam();
    std::array<std::function<cv::Mat(cv::Mat)>, 8> foos;

    for (int i = 0; i < 8; i++)
    {
        foos[i] = flags[i] ? ocvFoo : fluidFoo;
    }
    auto bar = flags[8] ? ocvBar : fluidBar;

    cv::Mat foo1OutMat = foos[1](foos[0](m_in_mat));
    cv::Mat foo3OutMat = foos[3](foos[2](foo1OutMat));

    m_ref_mat1 = foos[6](bar(foo3OutMat,
                             foos[5](foos[4](foo1OutMat))));

    m_ref_mat2 = foos[7](foo3OutMat);
}

TEST_P(GAPIBigHeteroTest, Test)
{
    EXPECT_NO_THROW(m_comp.apply(gin(m_in_mat), gout(m_out_mat1, m_out_mat2), cv::compile_args(m_kernels)));
    EXPECT_EQ(0, cv::countNonZero(m_ref_mat1 != m_out_mat1));
    EXPECT_EQ(0, cv::countNonZero(m_ref_mat2 != m_out_mat2));
}

static auto configurations = []()
{
    // Fill all possible configurations
    // from 000000000 to 111111111
    std::array<std::array<int, 9>, 512> arr;
    for (auto n = 0; n < 512; n++)
    {
        for (auto i = 0; i < 9; i++)
        {
            arr[n][i] = (n >> (8 - i)) & 1;
        }
    }
    return arr;
}();

INSTANTIATE_TEST_CASE_P(GAPIBigHeteroTest, GAPIBigHeteroTest,
                        ::testing::ValuesIn(configurations));

TEST(GAPIHeteroTestLPI, Test)
{
    cv::GMat in;
    auto mid = FluidFooI::on(in);
    auto out = FluidFooI::on(mid);
    cv::gapi::island("isl0", GIn(in),  GOut(mid));
    cv::gapi::island("isl1", GIn(mid), GOut(out));
    cv::GComputation c(in, out);

    cv::Mat in_mat = cv::Mat::eye(cv::Size(64, 64), CV_8UC1);
    cv::Mat out_mat;
    EXPECT_NO_THROW(c.apply(in_mat, out_mat, cv::compile_args(cv::gapi::kernels<FluidFoo2lpi>())));
    cv::Mat ref = fluidFoo(fluidFoo(in_mat));
    EXPECT_EQ(0, cv::countNonZero(ref != out_mat));
}

}  // namespace opencv_test
