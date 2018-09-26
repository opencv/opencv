// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"
#include "gapi_mock_kernels.hpp"

#include "opencv2/gapi/fluid/gfluidkernel.hpp"

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

    GAPI_FLUID_KERNEL(FFoo, I::Foo, false)
    {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View   &in,
                              cv::gapi::fluid::Buffer &out)
        {
            const uint8_t* in_ptr = in.InLine<uint8_t>(0);
            uint8_t *out_ptr = out.OutLine<uint8_t>();
            for (int i = 0; i < in.length(); i++)
            {
                out_ptr[i] = in_ptr[i] + 3;
            }
        }
    };

    GAPI_FLUID_KERNEL(FBar, I::Bar, false)
    {
        static const int Window = 1;

        static void run(const cv::gapi::fluid::View   &in1,
                        const cv::gapi::fluid::View   &in2,
                              cv::gapi::fluid::Buffer &out)
        {
            const uint8_t* in1_ptr = in1.InLine<uint8_t>(0);
            const uint8_t* in2_ptr = in2.InLine<uint8_t>(0);
            uint8_t *out_ptr = out.OutLine<uint8_t>();
            for (int i = 0; i < in1.length(); i++)
            {
                out_ptr[i] = 3*(in1_ptr[i] + in2_ptr[i]);
            }
        }
    };
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

    cv::Mat ref = 4*(m_in_mat+2 + m_in_mat+2);
    EXPECT_NO_THROW(m_comp.apply(m_in_mat, m_out_mat, cv::compile_args(m_ocv_kernels)));
    EXPECT_EQ(0, cv::countNonZero(ref != m_out_mat));
}

TEST_F(GAPIHeteroTest, TestFluid)
{
    EXPECT_TRUE(cv::gapi::fluid::backend() == m_fluid_kernels.lookup<I::Foo>());
    EXPECT_TRUE(cv::gapi::fluid::backend() == m_fluid_kernels.lookup<I::Bar>());

    cv::Mat ref = 3*(m_in_mat+3 + m_in_mat+3);
    EXPECT_NO_THROW(m_comp.apply(m_in_mat, m_out_mat, cv::compile_args(m_fluid_kernels)));
    EXPECT_EQ(0, cv::countNonZero(ref != m_out_mat));
}

TEST_F(GAPIHeteroTest, TestBoth_ExpectFailure)
{
    EXPECT_TRUE(cv::gapi::cpu::backend()   == m_hetero_kernels.lookup<I::Foo>());
    EXPECT_TRUE(cv::gapi::fluid::backend() == m_hetero_kernels.lookup<I::Bar>());
    EXPECT_ANY_THROW(m_comp.apply(m_in_mat, m_out_mat, cv::compile_args(m_hetero_kernels)));
}


}  // namespace opencv_test
