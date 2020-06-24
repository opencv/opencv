// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation


#include "test_precomp.hpp"

namespace opencv_test
{
    struct CustomArg
    {
        int number;
    };
}

namespace cv
{
    namespace detail
    {
        template<> struct CompileArgTag<opencv_test::CustomArg>
        {
            static const char* tag() { return "org.opencv.test.custom_arg"; }
        };
    }
}


namespace opencv_test
{
namespace
{
G_TYPED_KERNEL(GTestOp, <GMat(GMat)>, "org.opencv.test.test_op")
{
    static GMatDesc outMeta(GMatDesc in) { return in; }
};

GAPI_OCV_KERNEL(GOCVTestOp, GTestOp)
{
    static void run(const cv::Mat &/* in */, cv::Mat &/* out */) { }
};
} // anonymous namespace

TEST(GetCompileArgTest, PredefinedArgs)
{
    cv::gapi::GKernelPackage pkg = cv::gapi::kernels<GOCVTestOp>();
    cv::GCompileArg arg0 { pkg },
                    arg1 { cv::gapi::use_only { pkg } },
                    arg2 { cv::graph_dump_path { "fake_path" } };

    GCompileArgs compArgs { arg0, arg1, arg2 };

    auto kernelPkgOpt = cv::gapi::getCompileArg<cv::gapi::GKernelPackage>(compArgs);
    GAPI_Assert(kernelPkgOpt.has_value());
    EXPECT_NO_THROW(kernelPkgOpt.value().lookup("org.opencv.test.test_op"));

    auto hasUseOnlyOpt = cv::gapi::getCompileArg<cv::gapi::use_only>(compArgs);
    GAPI_Assert(hasUseOnlyOpt.has_value());
    EXPECT_NO_THROW(hasUseOnlyOpt.value().pkg.lookup("org.opencv.test.test_op"));

    auto dumpInfoOpt = cv::gapi::getCompileArg<cv::graph_dump_path>(compArgs);
    GAPI_Assert(dumpInfoOpt.has_value());
    EXPECT_EQ("fake_path", dumpInfoOpt.value().m_dump_path);
}

TEST(GetCompileArg, CustomArgs)
{;
    cv::GCompileArgs compArgs{ GCompileArg { CustomArg { 7 } } };

    auto customArgOpt = cv::gapi::getCompileArg<CustomArg>(compArgs);
    GAPI_Assert(customArgOpt.has_value());
    EXPECT_EQ(7, customArgOpt.value().number);
}
} // namespace opencv_test
