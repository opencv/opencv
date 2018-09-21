// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include "test_precomp.hpp"

#include "opencv2/gapi/cpu/gcpukernel.hpp"

namespace opencv_test
{

namespace
{
    G_TYPED_KERNEL(KTest, <cv::GScalar(cv::GScalar)>, "org.opencv.test.scalar_kernel") {
        static cv::GScalarDesc outMeta(cv::GScalarDesc in) { return in; }
    };
    GAPI_OCV_KERNEL(GOCVScalarTest, KTest)
    {
        static void run(const cv::Scalar &in, cv::Scalar &out) { out = in+cv::Scalar(1); }
    };
}

TEST(GAPI_MetaDesc, MatDesc)
{
    cv::Mat m1(240, 320, CV_8U);
    const auto desc1 = cv::descr_of(m1);
    EXPECT_EQ(CV_8U, desc1.depth);
    EXPECT_EQ(1,     desc1.chan);
    EXPECT_EQ(320,   desc1.size.width);
    EXPECT_EQ(240,   desc1.size.height);

    cv::Mat m2(480, 640, CV_8UC3);
    const auto desc2 = cv::descr_of(m2);
    EXPECT_EQ(CV_8U, desc2.depth);
    EXPECT_EQ(3,       desc2.chan);
    EXPECT_EQ(640,     desc2.size.width);
    EXPECT_EQ(480,     desc2.size.height);
}

TEST(GAPI_MetaDesc, Compare_Equal_MatDesc)
{
    const auto desc1 = cv::GMatDesc{CV_8U, 1, {64, 64}};
    const auto desc2 = cv::GMatDesc{CV_8U, 1, {64, 64}};

    EXPECT_TRUE(desc1 == desc2);
}

TEST(GAPI_MetaDesc, Compare_Not_Equal_MatDesc)
{
    const auto desc1 = cv::GMatDesc{CV_8U,  1, {64, 64}};
    const auto desc2 = cv::GMatDesc{CV_32F, 1, {64, 64}};

    EXPECT_TRUE(desc1 != desc2);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaNumber_1)
{
    cv::GMat in;
    cv::GComputation cc(in, in+in);

    const auto desc1 = cv::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = cv::GMatDesc{CV_32F,1,{128,128}};

    EXPECT_NO_THROW(cc.compile(desc1));
    EXPECT_NO_THROW(cc.compile(desc2));

    // FIXME: custom exception type?
    // It is worth checking if compilation fails with different number
    // of meta parameters
    EXPECT_THROW(cc.compile(desc1, desc1),        std::logic_error);
    EXPECT_THROW(cc.compile(desc1, desc2, desc2), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaNumber_2)
{
    cv::GMat a, b;
    cv::GComputation cc(cv::GIn(a, b), cv::GOut(a+b));

    const auto desc1 = cv::GMatDesc{CV_8U,1,{64,64}};
    EXPECT_NO_THROW(cc.compile(desc1, desc1));

    const auto desc2 = cv::GMatDesc{CV_32F,1,{128,128}};
    EXPECT_NO_THROW(cc.compile(desc2, desc2));

    // FIXME: custom exception type?
    EXPECT_THROW(cc.compile(desc1),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2, desc2, desc2), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaType_Mat)
{
    cv::GMat in;
    cv::GComputation cc(in, in+in);

    EXPECT_NO_THROW(cc.compile(cv::GMatDesc{CV_8U,1,{64,64}}));

    // FIXME: custom exception type?
    EXPECT_THROW(cc.compile(cv::empty_scalar_desc()), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaType_Scalar)
{
    cv::GScalar in;
    cv::GComputation cc(cv::GIn(in), cv::GOut(KTest::on(in)));

    const auto desc1 = cv::descr_of(cv::Scalar(128));
    const auto desc2 = cv::GMatDesc{CV_8U,1,{64,64}};
    const auto pkg   = cv::gapi::kernels<GOCVScalarTest>();
    EXPECT_NO_THROW(cc.compile(desc1, cv::compile_args(pkg)));

    // FIXME: custom exception type?
    EXPECT_THROW(cc.compile(desc2, cv::compile_args(pkg)), std::logic_error);
}

TEST(GAPI_MetaDesc, Compile_MatchMetaType_Mixed)
{
    cv::GMat a;
    cv::GScalar v;
    cv::GComputation cc(cv::GIn(a, v), cv::GOut(cv::gapi::addC(a, v)));

    const auto desc1 = cv::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = cv::descr_of(cv::Scalar(4));

    EXPECT_NO_THROW(cc.compile(desc1, desc2));

    // FIXME: custom exception type(s)?
    EXPECT_THROW(cc.compile(desc1),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2),               std::logic_error);
    EXPECT_THROW(cc.compile(desc2, desc1),        std::logic_error);
    EXPECT_THROW(cc.compile(desc1, desc1, desc1), std::logic_error);
    EXPECT_THROW(cc.compile(desc1, desc2, desc1), std::logic_error);
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaNumber_1)
{
    cv::GComputationT<cv::GMat(cv::GMat)> cc([](cv::GMat in)
    {
        return in+in;
    });

    const auto desc1 = cv::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = cv::GMatDesc{CV_32F,1,{128,128}};

    EXPECT_NO_THROW(cc.compile(desc1));
    EXPECT_NO_THROW(cc.compile(desc2));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaNumber_2)
{
    cv::GComputationT<cv::GMat(cv::GMat,cv::GMat)> cc([](cv::GMat a, cv::GMat b)
    {
        return a + b;
    });

    const auto desc1 = cv::GMatDesc{CV_8U,1,{64,64}};
    EXPECT_NO_THROW(cc.compile(desc1, desc1));

    const auto desc2 = cv::GMatDesc{CV_32F,1,{128,128}};
    EXPECT_NO_THROW(cc.compile(desc2, desc2));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaType_Mat)
{
    cv::GComputationT<cv::GMat(cv::GMat)> cc([](cv::GMat in)
    {
        return in+in;
    });

    EXPECT_NO_THROW(cc.compile(cv::GMatDesc{CV_8U,1,{64,64}}));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaType_Scalar)
{
    cv::GComputationT<cv::GScalar(cv::GScalar)> cc([](cv::GScalar in)
    {
        return KTest::on(in);
    });

    const auto desc1 = cv::descr_of(cv::Scalar(128));
    const auto pkg = cv::gapi::kernels<GOCVScalarTest>();
    //     EXPECT_NO_THROW(cc.compile(desc1, cv::compile_args(pkg)));
    cc.compile(desc1, cv::compile_args(pkg));
}

TEST(GAPI_MetaDesc, Typed_Compile_MatchMetaType_Mixed)
{
    cv::GComputationT<cv::GMat(cv::GMat,cv::GScalar)> cc([](cv::GMat a, cv::GScalar v)
    {
        return cv::gapi::addC(a, v);
    });

    const auto desc1 = cv::GMatDesc{CV_8U,1,{64,64}};
    const auto desc2 = cv::descr_of(cv::Scalar(4));

    EXPECT_NO_THROW(cc.compile(desc1, desc2));
}

} // namespace opencv_test
