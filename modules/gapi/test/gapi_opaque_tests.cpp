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

G_TYPED_KERNEL(GeneratePoint, <GPointOpaque(GMat)>, "test.opaque.gen_point")
{
    static GOpaqueDesc outMeta(const GMatDesc&) { return empty_gopaque_desc(); }
};

G_TYPED_KERNEL(FillMat, <GMat(cv::GOpaque<int>, int, int, cv::Size)>, "test.opaque.fill_mat")
{
    static GMatDesc outMeta(const GOpaqueDesc&, int depth, int chan, cv::Size size)
    {
        return cv::GMatDesc{depth, chan, size};
    }
};

G_TYPED_KERNEL(PaintPoint, <GMat(GPointOpaque, int, int, cv::Size)>, "test.opaque.paint_point")
{
    static GMatDesc outMeta(const GOpaqueDesc&, int depth, int chan, cv::Size size)
    {
        return cv::GMatDesc{depth, chan, size};
    }
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

GAPI_OCV_KERNEL(OCVFillMat, ThisTest::FillMat)
{
    static void run(int a, int, int, cv::Size, cv::Mat &out)
    {
        out = cv::Scalar(a);
    }
};

GAPI_OCV_KERNEL(OCVPaintPoint, ThisTest::PaintPoint)
{
    static void run(cv::Point a, int, int, cv::Size, cv::Mat &out)
    {
        out.at<uint8_t>(a) = 77;
    }
};
} // (anonymous namespace)

namespace
{
    // copypasted to be used later
    // a possible implementation from c++14 standart
    template<class T, class U = T>
    T exchange(T& obj, U&& new_value)
    {
        T old_value = std::move(obj);
        obj = std::forward<U>(new_value);
        return old_value;
    }
} // (anonymous namespace)

TEST(GOpaque, TestOpaqueOut)
{
    cv::Mat input = cv::Mat(52, 52, CV_8U);
    cv::Point point;

    cv::GMat in;
    auto out = ThisTest::GeneratePoint::on(in);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(input), cv::gout(point), cv::compile_args(cv::gapi::kernels<OCVGeneratePoint>()));

    EXPECT_TRUE(point == cv::Point(42, 42));
}

TEST(GOpaque, TestOpaqueIn)
{
    cv::Size sz = {42, 42};
    int depth = CV_8U;
    int chan = 1;
    cv::Mat mat = cv::Mat(sz, CV_MAKETYPE(depth, chan));
    int fill;

    cv::GOpaque<int> in;
    auto out = ThisTest::FillMat::on(in, depth, chan, sz);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(fill), cv::gout(mat), cv::compile_args(cv::gapi::kernels<OCVFillMat>()));

    auto diff = cv::Mat(sz, CV_MAKETYPE(depth, chan), cv::Scalar(fill)) - mat;
    EXPECT_EQ(cv::countNonZero(diff), 0);
}

TEST(GOpaque, TestOpaqueBetween)
{
    cv::Size sz = {50, 50};
    int depth = CV_8U;
    int chan = 1;
    cv::Mat mat_in = cv::Mat::zeros(sz, CV_MAKETYPE(depth, chan));
    cv::Mat mat_out = cv::Mat::zeros(sz, CV_MAKETYPE(depth, chan));

    cv::GMat in, out;
    auto betw = ThisTest::GeneratePoint::on(in);
    out = ThisTest::PaintPoint::on(betw, depth, chan, sz);

    cv::GComputation c(cv::GIn(in), cv::GOut(out));
    c.apply(cv::gin(mat_in), cv::gout(mat_out), cv::compile_args(cv::gapi::kernels<OCVGeneratePoint, OCVPaintPoint>()));

    int painted = mat_out.at<uint8_t>(42, 42);
    EXPECT_EQ(painted, 77);
}

TEST(GOpaque_OpaqueRef, TestMov)
{
    // Warning: this test is testing some not-very-public APIs
    // Test how OpaqueRef's mov() (aka poor man's move()) is working.

    // a helper class to verify move semantics in GOpaque
    struct Foo
    {
        char* cstring;

        Foo(const char* s = "")
        : cstring(nullptr)
        {
            if (s) {
                std::size_t n = std::strlen(s) + 1;
                cstring = new char[n];
                std::memcpy(cstring, s, n);
            }
        }

        ~Foo()
        {
            delete[] cstring;
        }

        Foo(const Foo& other)
        : Foo(other.cstring)
        {}

        Foo(Foo&& other) noexcept
        : cstring(exchange(other.cstring, nullptr))
        {}

        Foo& operator=(const Foo& other)
        {
             return *this = Foo(other);
        }

        Foo& operator=(Foo&& other) noexcept
        {
            std::swap(cstring, other.cstring);
            return *this;
        }
    };

    // compare 2 instances of Foo
    auto foo_are_eq = [](const Foo& a, const Foo& b, std::size_t n){
        for(std::size_t i = 0; i < n; ++i)
            if(a.cstring[i] != b.cstring[i])
                return false;
        return true;
    };

    using I = Foo;

    const char *word = "a word";
    const I gold(word);

    I test = gold;
    const char* ptr = test.cstring;

    cv::detail::OpaqueRef ref(test);
    cv::detail::OpaqueRef mov;
    mov.reset<I>();

    EXPECT_EQ(foo_are_eq(gold, ref.rref<I>(), 7), true);  // ref = gold

    mov.mov(ref);
    EXPECT_EQ(foo_are_eq(gold, mov.rref<I>(), 7), true);  // mov obtained the data
    EXPECT_EQ(ptr, mov.rref<I>().cstring);                // pointer is unchanged (same data)
    EXPECT_EQ(foo_are_eq(test, ref.rref<I>(), 7), true);  // ref = test
    EXPECT_EQ(foo_are_eq(test, mov.rref<I>(), 7), false); // ref lost the data
}
} // namespace opencv_test
