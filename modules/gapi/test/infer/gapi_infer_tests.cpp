// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2020 Intel Corporation

#include "../test_precomp.hpp"

// These tests verify some parts of cv::gapi::infer<> API
// regardless of the backend used

namespace opencv_test {
namespace {
template<class A, class B> using Check = cv::detail::valid_infer2_types<A, B>;

TEST(Infer, ValidInfer2Types)
{
    // Compiled == passed!

    // Argument block 1
    static_assert(Check< std::tuple<cv::GMat>   // Net
                       , std::tuple<cv::GMat> > // Call
                  ::value == true, "Must work");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::GMat, cv::GMat> > // Call
                  ::value == true, "Must work");

    // Argument block 2
    static_assert(Check< std::tuple<cv::GMat>             // Net
                       , std::tuple<cv::Rect> >           // Call
                  ::value == true, "Must work");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::Rect, cv::Rect> > // Call
                  ::value == true, "Must work");

    // Argument block 3 (mixed cases)
    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::GMat, cv::Rect> > // Call
                  ::value == true, "Must work");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::Rect, cv::GMat> > // Call
                  ::value == true, "Must work");

    // Argument block 4 (super-mixed)
    static_assert(Check< std::tuple<cv::GMat, cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::Rect, cv::GMat, cv::Rect> > // Call
                  ::value == true, "Must work");

    // Argument block 5 (mainly negative)
    static_assert(Check< std::tuple<cv::GMat>             // Net
                       , std::tuple<int> >                // Call
                  ::value == false, "This type(s) shouldn't pass");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<int, cv::Rect> >      // Call
                  ::value == false, "This type(s) shouldn't pass");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::Rect, cv::Point> >// Call
                  ::value == false, "This type(s) shouldn't pass");

    // Argument block 5 (wrong args length)
    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::GMat> >           // Call
                  ::value == false, "Should fail -- not enough args");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>   // Net
                       , std::tuple<cv::Rect> >           // Call
                  ::value == false, "Should fail -- not enough args");

    static_assert(Check< std::tuple<cv::GMat, cv::GMat>             // Net
                       , std::tuple<cv::Rect, cv::Rect, cv::GMat> > // Call
                  ::value == false, "Should fail -- too much args");
}
} // anonymous namespace
} // namespace opencv_test
