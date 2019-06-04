// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#include <opencv2/gapi/cpu/gcpukernel.hpp>

#include "api/gbackend_priv.hpp" // directly instantiate GBackend::Priv

namespace opencv_test
{
namespace {
    // FIXME: Currently every Kernel implementation in this test file has
    // its own backend() method and it is incorrect! API classes should
    // provide it out of the box.

namespace I
{
    G_TYPED_KERNEL(Foo, <cv::GMat(cv::GMat)>, "test.kernels.foo")
    {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in) { return in; }
    };

    G_TYPED_KERNEL(Bar, <cv::GMat(cv::GMat,cv::GMat)>, "test.kernels.bar")
    {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GMatDesc &) { return in; }
    };

    G_TYPED_KERNEL(Baz, <cv::GScalar(cv::GMat)>, "test.kernels.baz")
    {
        static cv::GScalarDesc outMeta(const cv::GMatDesc &) { return cv::empty_scalar_desc(); }
    };

    G_TYPED_KERNEL(Qux, <cv::GMat(cv::GMat, cv::GScalar)>, "test.kernels.qux")
    {
        static cv::GMatDesc outMeta(const cv::GMatDesc &in, const cv::GScalarDesc &) { return in; }
    };

    G_TYPED_KERNEL(Quux, <cv::GMat(cv::GScalar, cv::GMat)>, "test.kernels.quux")
    {
        static cv::GMatDesc outMeta(const cv::GScalarDesc &, const cv::GMatDesc& in) { return in; }
    };
}

// Kernel implementations for imaginary Jupiter device
namespace Jupiter
{
    namespace detail
    {
        static cv::gapi::GBackend backend(std::make_shared<cv::gapi::GBackend::Priv>());
    }

    inline cv::gapi::GBackend backend() { return detail::backend; }

    GAPI_OCV_KERNEL(Foo, I::Foo)
    {
        static void run(const cv::Mat &, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Bar, I::Bar)
    {
        static void run(const cv::Mat &, const cv::Mat &, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Baz, I::Baz)
    {
        static void run(const cv::Mat &, cv::Scalar &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Qux, I::Qux)
    {
        static void run(const cv::Mat &, const cv::Scalar&, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };

    GAPI_OCV_KERNEL(Quux, I::Quux)
    {
        static void run(const cv::Scalar&, const cv::Mat&, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
} // namespace Jupiter

// Kernel implementations for imaginary Saturn device
namespace Saturn
{
    namespace detail
    {
        static cv::gapi::GBackend backend(std::make_shared<cv::gapi::GBackend::Priv>());
    }

    inline cv::gapi::GBackend backend() { return detail::backend; }

    GAPI_OCV_KERNEL(Foo, I::Foo)
    {
        static void run(const cv::Mat &, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Bar, I::Bar)
    {
        static void run(const cv::Mat &, const cv::Mat &, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Baz, I::Baz)
    {
        static void run(const cv::Mat &, cv::Scalar &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
    GAPI_OCV_KERNEL(Qux, I::Qux)
    {
        static void run(const cv::Mat &, const cv::Scalar&, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };

    GAPI_OCV_KERNEL(Quux, I::Quux)
    {
        static void run(const cv::Scalar&, const cv::Mat&, cv::Mat &) { /*Do nothing*/ }
        static cv::gapi::GBackend backend() { return detail::backend; } // FIXME: Must be removed
    };
} // namespace Saturn
} // anonymous namespace
} // namespace opencv_test
