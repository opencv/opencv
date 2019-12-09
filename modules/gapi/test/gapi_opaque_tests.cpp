// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2019 Intel Corporation


#include "test_precomp.hpp"

namespace opencv_test
{

namespace ThisTest
{
using GPointOpaque = cv::GOpaque<cv::Point>;
G_TYPED_KERNEL(GeneratePoint, <GPointOpaque(GMat)>, "test.opaque.out_const")
{
    static GOpaqueDesc outMeta(const GMatDesc&) { return empty_opaque_desc(); }
};
} // namespace ThisTest

namespace
{
GAPI_OCV_KERNEL(OCVGeneratePoint, ThisTest::GeneratePoint)
{
    static void run(cv::Mat, cv::Point &out)
    {
        out = cv::Point(42, 42);
    }
};
} // (anonymous namespace)

TEST(GOpaque, TestReturnValue)
{
    // FIXME: Make .apply() able to take compile arguments
    cv::GComputationT<ThisTest::GPointOpaque(cv::GMat)> c(ThisTest::GeneratePoint::on);
    auto cc = c.compile(cv::GMatDesc{CV_8U,1,{52,52}},
                        cv::compile_args(cv::gapi::kernels<OCVGeneratePoint>()));

    // Prepare input matrix
    cv::Mat input = cv::Mat(52, 52, CV_8U);

    cv::Point point;
    cc(input, point);

    EXPECT_TRUE(point == cv::Point(42, 42));
}

TEST(GOpaque_OpaqueRef, TestMov)
{
    // Warning: this test is testing some not-very-public APIs
    // Test how OpaqueRef's mov() (aka poor man's move()) is working.

    using I = int;
    const I gold = 42;
    I test = gold;
    const I* ptr = &test;

    cv::detail::OpaqueRef ref(test);
    cv::detail::OpaqueRef mov;
    mov.reset<I>();

    //std::cout<<gold<<' '<<ref.rref<I>()<<std::endl;
    EXPECT_EQ(gold, ref.rref<I>());

    std::cout<<gold<<' '<<mov.rref<I>()<<std::endl;
    std::cout<<ptr<<' '<<&mov.rref<I>()<<std::endl;
    std::cout<<I{}<<' '<<ref.rref<I>()<<std::endl;
    mov.mov(ref);
    std::cout<<"after mov:"<<std::endl;
    std::cout<<gold<<' '<<mov.rref<I>()<<std::endl;
    std::cout<<ptr<<' '<<&mov.rref<I>()<<std::endl;
    std::cout<<I{}<<' '<<ref.rref<I>()<<std::endl;
    EXPECT_EQ(gold, mov.rref<I>());
    EXPECT_EQ(ptr, &mov.rref<I>());
    EXPECT_EQ(I{},  ref.rref<I>());
    EXPECT_EQ(I{},  test);
}
} // namespace opencv_test
