// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html.
//
// Copyright (C) 2018 Intel Corporation


#ifndef OPENCV_GAPI_OWN_SATURATE_HPP
#define OPENCV_GAPI_OWN_SATURATE_HPP

#include <cmath>

#include <limits>
#include <type_traits>

#include <opencv2/gapi/own/assert.hpp>

namespace cv { namespace gapi { namespace own {
//-----------------------------
//
// Numeric cast with saturation
//
//-----------------------------

template<typename DST, typename SRC>
static inline DST saturate(SRC x)
{
    // only integral types please!
    GAPI_DbgAssert(std::is_integral<DST>::value &&
                   std::is_integral<SRC>::value);

    if (std::is_same<DST, SRC>::value)
        return static_cast<DST>(x);

    if (sizeof(DST) > sizeof(SRC))
        return static_cast<DST>(x);

    // compiler must recognize this saturation,
    // so compile saturate<s16>(a + b) with adds
    // instruction (e.g.: _mm_adds_epi16 if x86)
    return x < std::numeric_limits<DST>::min()?
               std::numeric_limits<DST>::min():
           x > std::numeric_limits<DST>::max()?
               std::numeric_limits<DST>::max():
           static_cast<DST>(x);
}

// Note, that OpenCV rounds differently:
// - like std::round() for add, subtract
// - like std::rint() for multiply, divide
template<typename DST, typename SRC, typename R>
static inline DST saturate(SRC x, R round)
{
    if (std::is_floating_point<DST>::value)
    {
        return static_cast<DST>(x);
    }
    else if (std::is_integral<SRC>::value)
    {
        GAPI_DbgAssert(std::is_integral<DST>::value &&
                       std::is_integral<SRC>::value);
        return saturate<DST>(x);
    }
    else
    {
        GAPI_DbgAssert(std::is_integral<DST>::value &&
                 std::is_floating_point<SRC>::value);
#ifdef _WIN32
// Suppress warning about convering x to floating-point
// Note that x is already floating-point at this point
#pragma warning(disable: 4244)
#endif
        int ix = static_cast<int>(round(x));
#ifdef _WIN32
#pragma warning(default: 4244)
#endif
        return saturate<DST>(ix);
    }
}

// explicit suffix 'd' for double type
inline double  ceild(double x) { return std::ceil(x); }
inline double floord(double x) { return std::floor(x); }
inline double roundd(double x) { return std::round(x); }
inline double  rintd(double x) { return std::rint(x); }

} //namespace own
} //namespace gapi
} //namespace cv
#endif /* OPENCV_GAPI_OWN_SATURATE_HPP */
