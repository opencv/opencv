// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_UTIL_THROW_HPP
#define OPENCV_GAPI_UTIL_THROW_HPP

#include <utility>  // std::forward

#if !defined(__EXCEPTIONS)
#include <stdlib.h>
#include <stdio.h>
#endif

namespace cv
{
namespace util
{
template <class ExceptionType>
[[noreturn]] void throw_error(ExceptionType &&e)
{
#if defined(__EXCEPTIONS) || defined(_CPPUNWIND)
    throw std::forward<ExceptionType>(e);
#else
    fprintf(stderr, "An exception thrown! %s\n" , e.what());
    fflush(stderr);
    abort();
#endif
}
} // namespace util
} // namespace cv

#endif // OPENCV_GAPI_UTIL_THROW_HPP
