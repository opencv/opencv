// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_ASSERT_HPP
#define OPENCV_GAPI_OWN_ASSERT_HPP

#if !defined(GAPI_STANDALONE)
#include <opencv2/core/base.hpp>
#define GAPI_Assert CV_Assert
#define GAPI_DbgAssert CV_DbgAssert

#else
#include <stdexcept>
#include <sstream>
#include <opencv2/gapi/util/throw.hpp>

namespace detail
{
    inline void assert_abort(const char* str, int line, const char* file, const char* func)
    {
        std::stringstream ss;
        ss << file << ":" << line << ": Assertion " << str << " in function " << func << " failed\n";
        cv::util::throw_error(std::logic_error(ss.str()));
    }
}

#define GAPI_Assert(expr) \
{ if (!(expr)) ::detail::assert_abort(#expr, __LINE__, __FILE__, __func__); }


#ifdef NDEBUG
#  define GAPI_DbgAssert(expr)
#else
#  define GAPI_DbgAssert(expr) GAPI_Assert(expr)
#endif

#endif // GAPI_STANDALONE

#endif // OPENCV_GAPI_OWN_ASSERT_HPP
