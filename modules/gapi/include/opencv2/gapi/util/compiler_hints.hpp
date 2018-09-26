// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation

#ifndef OPENCV_GAPI_UTIL_COMPILER_HINTS_HPP
#define OPENCV_GAPI_UTIL_COMPILER_HINTS_HPP

namespace cv
{
namespace util
{
    //! Utility template function to prevent "unused" warnings by various compilers.
    template<typename T> void suppress_unused_warning( const T& ) {}
} // namespace util
} // namespace cv

#define UNUSED(x) cv::util::suppress_unused_warning(x)

#endif /* OPENCV_GAPI_UTIL_COMPILER_HINTS_HPP */
